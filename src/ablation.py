"""Change the input, re-score, and measure what the score depends on.

Two ways to take a series away:

- **drop**: remove that series' windows from the attention pool. The softmax renormalises over the rest. This is
  "the model never had those windows", and needs only the head, so it is exact and instant from saved features.
- **blank**: zero the series' slices and rerun the pipeline from the stack, exactly as for a study that lacks the
  series. Windows now see black slices, blank windows still receive some attention, and if the blanked slot sits
  at either end of the stack the window centres move. Only windows that touch a changed slice need the backbone,
  so this is cheap too.

The attention head has no positional encoding: shuffling the windows leaves the output unchanged. Anything the
model knows about where a window came from (plane, medial or lateral side) must come from the pixels.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch

import raptor_core as rc


class FeatureBank:
    """Backbone features per study and window, reusing the saved ones wherever the window is unchanged.

    feats: [N, 62, F] saved features for centres 1..62 of the unmodified stacks.
    """

    def __init__(self, model, feats: np.ndarray, device: str = "cpu", batch: int = 16):
        self.model, self.device, self.batch = model, device, batch
        self.base = feats.astype(np.float32)
        self._blank = None

    @torch.no_grad()
    def _encode(self, windows: torch.Tensor) -> np.ndarray:
        out = [self.model.backbone(windows[i:i + self.batch].to(self.device)).float().cpu()
               for i in range(0, len(windows), self.batch)]
        return torch.cat(out).numpy()

    def blank_feature(self, res: int = 384) -> np.ndarray:
        """Feature of a window whose three slices are all black (identical for every study)."""
        if self._blank is None:
            self._blank = self._encode(rc.make_windows(np.zeros((3, res, res), np.uint8), [1], res=res))[0]
        return self._blank

    def features(self, i: int, vol: np.ndarray, mask: np.ndarray, changed: np.ndarray, k: int = 62):
        """[k, F] features of study i after its slices `changed` (bool [64]) were modified in `vol`."""
        centres = rc.eval_centers(mask, k)
        out = np.empty((len(centres), self.base.shape[-1]), np.float32)
        todo = []
        for j, c in enumerate(centres):
            touched = changed[c - 1:c + 2]
            if not touched.any():
                out[j] = self.base[i, c - 1]
            elif not vol[c - 1:c + 2].any():
                out[j] = self.blank_feature(vol.shape[-1])
            else:
                todo.append((j, c))
        if todo:
            wins = rc.make_windows(vol, [c for _, c in todo], res=vol.shape[-1])
            for (j, _), f in zip(todo, self._encode(wins)):
                out[j] = f
        return out


def blank_slots(vol: np.ndarray, mask: np.ndarray, slice_ranges):
    """Zero the given [start, end) slice ranges. Returns (vol, mask, changed)."""
    vol, mask = vol.copy(), mask.copy()
    changed = np.zeros(len(mask), bool)
    for a, b in slice_ranges:
        vol[a:b] = 0
        mask[a:b] = 0
        changed[a:b] = True
    return vol, mask, changed


@torch.no_grad()
def head_logits(model, feats: np.ndarray) -> np.ndarray:
    """[N, K, F] -> [N, n] logits (float64)."""
    return model.head(torch.from_numpy(feats.astype(np.float32))).numpy().astype(np.float64)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# ============================================================================
# Acquisition-like perturbations of a whole stack
# ============================================================================
def zoom(vol: np.ndarray, factor: float) -> np.ndarray:
    """Magnify every slice about its centre by `factor` (> 1), as when a series' field of view is under 140 mm."""
    H = vol.shape[-1]
    c = int(round(H / factor))
    o = (H - c) // 2
    out = np.empty_like(vol)
    for s in range(len(vol)):
        out[s] = cv2.resize(vol[s, o:o + c, o:o + c], (H, H), interpolation=cv2.INTER_LINEAR)
    return out


def gamma(vol: np.ndarray, g: float) -> np.ndarray:
    """Remap intensities through x ** g (g > 1 darkens mid-tones, g < 1 brightens). Black stays black."""
    lut = np.clip(np.round(255.0 * (np.arange(256) / 255.0) ** g), 0, 255).astype(np.uint8)
    return lut[vol]
