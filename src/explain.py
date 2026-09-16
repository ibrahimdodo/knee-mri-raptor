"""Where the model looks: across windows (attention), and within a window (Grad-CAM).

Across windows the explanation is exact rather than approximate. The backbone sees each 3-slice window on
its own, and the head computes, for finding n,
    logit[n] = sum_k a[k, n] * window_logit[k, n]
so every study logit splits into additive per-window contributions. What that does NOT say is what would
happen if a window were removed: the softmax would hand its attention to the others. That causal question is
Phase 4's ablations.

Within a window, Grad-CAM weights the last CoAtNet feature map (12 x 12 at 384 px, so 32 px cells) by the
gradient of that window's logit for the finding.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

# Slot layout of the native 384 dense stack, as slice index ranges [start, end).
SLOTS = [("Sagittal fluid", 0, 18), ("Sagittal other", 18, 32), ("Coronal fluid", 32, 44),
         ("Coronal other", 44, 52), ("Axial", 52, 64)]
CENTRES = np.arange(1, 63)          # all window centres of a 64-slice stack


def slot_of_slice(s: int) -> int:
    return next(i for i, (_, a, b) in enumerate(SLOTS) if a <= s < b)


def window_slots(centres=CENTRES):
    """[K] slot index of each window's centre slice, and [K] bool: does the window straddle two slots."""
    own = np.array([slot_of_slice(c) for c in centres])
    straddle = np.array([slot_of_slice(c - 1) != slot_of_slice(c + 1) for c in centres])
    return own, straddle


@torch.no_grad()
def head_outputs(model, feats: np.ndarray):
    """feats [N, K, F] -> (logits [N, n], attention [N, K, n], window_logits [N, K, n]) as float64."""
    out = model.head_detailed(torch.from_numpy(feats.astype(np.float32)))
    return tuple(t.numpy().astype(np.float64) for t in out)


def effective_windows(attn: np.ndarray) -> np.ndarray:
    """1 / sum a^2 over windows: 1 if all attention is on one window, K if spread evenly."""
    return 1.0 / (attn ** 2).sum(axis=-2)


def grad_cam(model, window: torch.Tensor, finding: int, device: str = "cpu", stage: int = -1):
    """Grad-CAM for one window [3, H, W] (already normalised) and one finding.

    `stage` picks the feature map. In CoAtNet stages 0-1 are convolutional (48 x 48 at 384 px for stage 1)
    and stages 2-3 are transformer blocks whose positions can mix globally, so the last stage's map is
    the coarsest and the least spatially faithful.
    Returns (cam [H, W] in [0, 1], the window's logit for the finding).
    """
    acts = {}
    hook = model.backbone.stages[stage].register_forward_hook(lambda m, i, o: acts.__setitem__("a", o))
    try:
        with torch.enable_grad():
            feat = model.backbone(window[None].to(device))
            logit = (model.norm(feat)[0] * model.clsW[finding]).sum() + model.clsb[finding]
            act = acts["a"]
            grads = torch.autograd.grad(logit, act)[0]
    finally:
        hook.remove()
    weights = grads.mean(dim=(2, 3), keepdim=True)           # one importance per channel
    cam = F.relu((weights * act).sum(1, keepdim=True))       # [1, 1, h, w]
    cam = F.interpolate(cam, size=window.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = cam / (cam.max() + 1e-8)
    return cam.detach().cpu().numpy(), float(logit.detach())


@torch.no_grad()
def occlusion(model, window: torch.Tensor, finding: int, device: str = "cpu",
              patch: int = 48, stride: int = 24, batch: int = 16):
    """How much the window's logit for `finding` drops when each patch is replaced by the window mean.

    Model-agnostic and directly causal for the window, at the cost of one forward pass per patch position.
    Returns (drop map [H, W], averaged over the patches covering each pixel; the unoccluded logit).
    """
    def logits(x):
        feat = model.backbone(x.to(device))
        return ((model.norm(feat) * model.clsW[finding]).sum(-1) + model.clsb[finding]).float().cpu()

    H, W = window.shape[-2:]
    base = float(logits(window[None])[0])
    fill = window.mean(dim=(1, 2), keepdim=True)
    coords = [(y, x) for y in range(0, H - patch + 1, stride) for x in range(0, W - patch + 1, stride)]
    drops = torch.empty(len(coords))
    for i in range(0, len(coords), batch):
        xs = window.repeat(len(coords[i:i + batch]), 1, 1, 1)
        for b, (y, x) in enumerate(coords[i:i + batch]):
            xs[b, :, y:y + patch, x:x + patch] = fill
        drops[i:i + batch] = base - logits(xs)
    total = torch.zeros(H, W); count = torch.zeros(H, W)
    for d, (y, x) in zip(drops, coords):
        total[y:y + patch, x:x + patch] += d
        count[y:y + patch, x:x + patch] += 1
    return (total / count.clamp(min=1)).numpy(), base


# ============================================================================
# Laterality
# ============================================================================
_RIGHT = ("RIGHT", "DERECH", " RT", "RT.", "RT_", "_RT", "(RT", " DCHA", "DCHA")
_LEFT = ("LEFT", "IZQUIER", " LT", "LT.", "LT_", "_LT", "(LT", " IZDA", "IZDA")


def knee_side(*texts) -> str | None:
    """'R', 'L' or None from DICOM laterality tags or free text (English/Spanish), first decisive wins."""
    for t in texts:
        if t is None or (isinstance(t, float) and np.isnan(t)):
            continue
        u = " " + str(t).upper().strip()
        if u.strip() in ("R", "L"):
            return u.strip()
        r, l = any(k in u for k in _RIGHT), any(k in u for k in _LEFT)
        if r != l:
            return "R" if r else "L"
    return None


def medial_at_high_index(normal_x: float, side: str) -> bool | None:
    """Does increasing sagittal slice index move towards the medial side of the knee?

    Slices are sorted by position along the slice normal. In DICOM patient coordinates +x points to the
    patient's left, so index increases towards the patient's left when normal_x > 0. The medial side of a
    right knee faces the patient's left (towards the other leg); for a left knee it faces the right.
    """
    if side not in ("R", "L") or normal_x is None or abs(normal_x) < 0.5:
        return None
    return (normal_x > 0) == (side == "R")
