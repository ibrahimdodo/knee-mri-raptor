"""Core of the Raptor knee-MRI pipeline: DICOM study -> fixed slice stack -> 2.5D windows -> model.

Re-implemented from the author's published inference and training notebooks
(dreaddevelopment/knee-mri-twelve-findings-from-a-single-model and
.../knee-mri-training-the-twelve-finding-model). Every step that the checkpoint depends on is kept
behaviourally identical; the differences are that the geometry is a parameter instead of module
constants, and that the model can also return what it attended to.

This file is deliberately self-contained (no imports from elsewhere in the repo) because the Kaggle
notebook is built by pasting it in front of a small main script.
"""
from __future__ import annotations

import glob
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
       "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]

THREADS = 8          # parallel DICOM reads; results are ordered deterministically regardless

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


# ============================================================================
# Geometry: how a study with a variable number of series and slices becomes one fixed stack
# ============================================================================
@dataclass(frozen=True)
class Geometry:
    img: int = 384              # stored slice size in pixels
    crop_mm: float = 140.0      # physical field of view kept around the image centre
    span_lo: float = 0.02       # slices are sampled from this fraction of each series...
    span_hi: float = 0.98       # ...to this one, so the outermost 2% at each end are skipped
    # (plane, fluid preference, slice count). fluid 1 = prefer fluid-sensitive, 0 = prefer not,
    # -1 = take the first series in that plane. Slots are filled in this order.
    slots: tuple = field(default=(("Sagittal", 1, 18), ("Sagittal", 0, 14), ("Coronal", 1, 12),
                                  ("Coronal", 0, 8), ("Axial", -1, 12)))

    @property
    def n_slices(self) -> int:
        return sum(s[2] for s in self.slots)


# Geometry of raptor_ft_coatnet_v10_full.pt (dataset raptor-knee-native384dense).
NATIVE384_DENSE = Geometry()
# Geometry of the earlier public 0.924 notebook (raptor-knee-widedense), for comparison.
WIDEDENSE_336 = Geometry(img=336, span_lo=0.06, span_hi=0.94)


# ============================================================================
# DICOM reading
# ============================================================================
def order_series(series_dir: str):
    """Sort a series' files by physical position along the slice normal.

    File names and InstanceNumber are not reliable slice orders. The position that is: project
    ImagePositionPatient (the 3D corner of the slice) onto the normal of the image plane, which is
    the cross product of the row and column direction cosines in ImageOrientationPatient.
    Returns ([(path, pixel_spacing_mm)], median_pixel_spacing).
    """
    import pydicom

    def header(f):
        try:
            h = pydicom.dcmread(f, stop_before_pixels=True)
            iop = getattr(h, "ImageOrientationPatient", None)
            ipp = getattr(h, "ImagePositionPatient", None)
            if iop is not None and ipp is not None and len(iop) == 6:
                normal = np.cross(np.array(iop[:3], float), np.array(iop[3:], float))
                pos = float(np.dot(np.array(ipp, float), normal))
            else:
                pos = float(getattr(h, "InstanceNumber", 0) or 0)
            ps = getattr(h, "PixelSpacing", None)
            return pos, f, (float(ps[0]) if ps is not None else 0.5), True
        except Exception:
            return 0.0, f, 0.5, False

    files = glob.glob(series_dir + "/*.dcm")
    # Reading headers is disk-bound, and a series can hold hundreds of files. Threads keep the order
    # deterministic (results stay in `files` order before the stable sort below).
    with ThreadPoolExecutor(max_workers=THREADS) as pool:
        recs = list(pool.map(header, files))
    ps_list = [ps for _, _, ps, ok in recs if ok]
    recs.sort(key=lambda x: x[0])
    med_ps = float(np.median(ps_list)) if ps_list else 0.5
    return [(f, ps) for _, f, ps, _ in recs], med_ps


def read_pixels(path: str) -> np.ndarray:
    """Pixel data in modality units (rescale slope/intercept applied), bright = high signal."""
    import pydicom
    try:
        from pydicom.pixels import apply_modality_lut                     # pydicom >= 3
    except ImportError:
        from pydicom.pixel_data_handlers.util import apply_modality_lut   # pydicom 2.x (Kaggle)

    d = pydicom.dcmread(path)
    a = apply_modality_lut(d.pixel_array, d).astype(np.float32)
    if str(getattr(d, "PhotometricInterpretation", "")) == "MONOCHROME1":
        a = a.max() - a
    return a


def crop_mm_and_resize(a: np.ndarray, pixel_spacing: float, geom: Geometry) -> np.ndarray:
    """Centre-crop a square of geom.crop_mm millimetres, then resize to geom.img pixels.

    Cropping in millimetres rather than pixels makes every study cover the same physical field of
    view whatever its matrix size, so a meniscus is roughly the same number of pixels everywhere.
    """
    import cv2

    h, w = a.shape
    cpx = int(round(geom.crop_mm / max(pixel_spacing, 1e-3)))
    cpx = min(cpx, min(h, w))
    y0, x0 = (h - cpx) // 2, (w - cpx) // 2
    a = a[y0:y0 + cpx, x0:x0 + cpx]
    return cv2.resize(a, (geom.img, geom.img), interpolation=cv2.INTER_AREA)


def pick_series_for_slot(rows: list[dict], plane: str, fluid: int, used: set):
    """First unused series in `plane`, preferring the requested fluid sensitivity.

    `rows` must be in train_series.csv row order: ties are broken by that order, exactly as the
    checkpoint's corpus was built.
    """
    cands = [r for r in rows if r["Anatomical_Plane"] == plane and r["SeriesInstanceUID"] not in used]
    if fluid in (0, 1):
        pref = [r for r in cands if _as_int(r.get("Fluid_Sensitive", 0)) == fluid]
        if pref:
            return pref[0]
    return cands[0] if cands else None


def _as_int(v) -> int:
    try:
        return int(v or 0)
    except (TypeError, ValueError):     # NaN
        return 0


def build_study(study_dir: str, series_rows: list[dict], geom: Geometry = NATIVE384_DENSE):
    """One study -> (vol uint8 [D,img,img], mask uint8 [D], slot_info list).

    Five slots are filled in a fixed order. A slot with no matching series stays all zeros and its
    mask entries are 0. Within a series, slices are spread evenly over [span_lo, span_hi] of the
    sorted stack, and intensities are windowed to that series' 2nd-98th percentile, so contrast is
    comparable across scanners whose raw units differ by orders of magnitude.
    """
    D = geom.n_slices
    vol = np.zeros((D, geom.img, geom.img), np.uint8)
    idx, used, slot_info = 0, set(), []
    for plane, fluid, k in geom.slots:
        info = {"plane": plane, "fluid_pref": fluid, "k": k, "start": idx, "series": None}
        r = pick_series_for_slot(series_rows, plane, fluid, used)
        if r is None:
            slot_info.append(info); idx += k; continue
        used.add(r["SeriesInstanceUID"])
        files, med_ps = order_series(f"{study_dir}/{r['SeriesInstanceUID']}")
        info.update(series=r["SeriesInstanceUID"], n_files=len(files), pixel_spacing=med_ps,
                    fluid=_as_int(r.get("Fluid_Sensitive", 0)))
        if not files:
            slot_info.append(info); idx += k; continue
        n = len(files)
        lo, hi = int(n * geom.span_lo), int(n * geom.span_hi) - 1
        hi = max(hi, lo)
        picks = np.linspace(lo, hi, k).round().astype(int) if n > 1 else np.zeros(k, int)
        info["picks"] = [int(p) for p in picks]
        picked = [files[min(p, n - 1)] for p in picks]

        def load(fp_ps):
            fp, ps = fp_ps
            try:
                return read_pixels(fp), ps
            except Exception:
                return None, med_ps

        with ThreadPoolExecutor(max_workers=THREADS) as pool:
            loaded = list(pool.map(load, picked))
        arrs = [a for a, _ in loaded]
        pss = [ps for _, ps in loaded]
        valid = [a for a in arrs if a is not None]
        if valid:
            loq, hiq = np.percentile(np.concatenate([a.ravel() for a in valid]), [2.0, 98.0])
        else:
            loq, hiq = 0.0, 1.0
        for a, ps in zip(arrs, pss):
            if idx >= D:
                break
            if a is None:
                idx += 1; continue
            aw = np.clip((a - loq) / (hiq - loq + 1e-6), 0, 1)
            aw = crop_mm_and_resize(aw, ps if ps > 0 else med_ps, geom)
            vol[idx] = (aw * 255).astype(np.uint8)
            idx += 1
        slot_info.append(info)
        if idx >= D:
            break
    mask = (vol.reshape(D, -1).sum(1) > 0).astype(np.uint8)
    return vol, mask, slot_info


# ============================================================================
# 2.5D windows
# ============================================================================
def eval_centers(mask: np.ndarray, k: int) -> list[int]:
    """k evenly spaced window centres between the first and last non-empty slice.

    A centre needs a slice on each side, so a full 64-slice stack has 62 possible centres; k=62
    uses every one. Windows can straddle two slots (e.g. slice 17 is the last fluid-sensitive
    sagittal, slice 18 the first non-fluid one), which is how the checkpoint was trained.
    """
    D = len(mask)
    valid = np.where(mask > 0)[0]
    if len(valid) < 3:
        valid = np.arange(min(3, D))
    lo, hi = int(valid.min()), int(valid.max())
    cs = list(range(lo + 1, hi))
    if not cs:
        cs = [max(1, min((lo + hi) // 2, D - 2))]
    idx = np.linspace(0, len(cs) - 1, k).round().astype(int)
    return [cs[i] for i in idx]


def make_windows(vol: np.ndarray, centers: list[int], res: int, norm: str = "imagenet") -> torch.Tensor:
    """Stack slices (c-1, c, c+1) as the RGB channels of one image per centre -> [K,3,res,res]."""
    D = vol.shape[0]
    out = torch.empty((len(centers), 3, res, res), dtype=torch.float32)
    for j, c in enumerate(centers):
        c = max(1, min(c, D - 2))
        tri = torch.from_numpy(np.stack([vol[c - 1], vol[c], vol[c + 1]], 0).astype(np.float32) / 255.0)
        if tri.shape[-1] != res:
            tri = F.interpolate(tri[None], size=(res, res), mode="bilinear", align_corners=False)[0]
        out[j] = tri
    if norm == "imagenet":
        out = (out - IMAGENET_MEAN) / IMAGENET_STD
    return out


# ============================================================================
# Model
# ============================================================================
def build_backbone(arch: str, pretrained: bool = False) -> nn.Module:
    # CoAtNet/MaxViT/ConvNeXt have no CLS token, so they average-pool; the "vit" substring in
    # "maxvit" must not send them down the ViT path.
    import timm

    hybrid = arch.startswith(("maxvit", "maxxvit", "coatnet", "coat_", "convnext"))
    is_vit = (not hybrid) and any(k in arch for k in ("vit", "deit", "dinov2", "eva", "beit"))
    kw = dict(pretrained=pretrained, num_classes=0, in_chans=3)
    kw.update(global_pool="token", dynamic_img_size=True) if is_vit else kw.update(global_pool="avg")
    return timm.create_model(arch, **kw)


class RaptorClassifier(nn.Module):
    """Backbone per window + attention pooling with separate weights for each finding.

    For a study with K windows the backbone gives features h[k] (dim F). For finding n:
        a[k, n]  = softmax over k of  att(h[k])[n]        -- where to look, per finding
        p[n]     = sum_k a[k, n] * h[k]                    -- finding-specific study vector
        logit[n] = <p[n], clsW[n]> + clsb[n]
    """

    def __init__(self, backbone: nn.Module, F_dim: int = 768, n: int = 12, drop: float = 0.2):
        super().__init__()
        self.backbone = backbone
        self.norm = nn.LayerNorm(F_dim)
        self.att = nn.Sequential(nn.Linear(F_dim, 256), nn.Tanh(), nn.Dropout(drop), nn.Linear(256, n))
        self.clsW = nn.Parameter(torch.zeros(n, F_dim))
        self.clsb = nn.Parameter(torch.zeros(n))
        nn.init.trunc_normal_(self.clsW, std=0.02)
        self.n = n

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        B, K = x.shape[:2]
        return self.backbone(x.flatten(0, 1)).view(B, K, -1)

    def head_detailed(self, feats: torch.Tensor):
        """feats [B,K,F] -> (logits [B,n], attention [B,K,n], per-window logits [B,K,n])."""
        h = self.norm(feats)
        a = torch.softmax(self.att(h), dim=1)
        pooled = torch.einsum("bkn,bkf->bnf", a, h)
        logits = (pooled * self.clsW).sum(-1) + self.clsb
        # What each window alone would score: logits are linear in the pooled vector, so the study
        # logit is exactly sum_k a[k,n] * window_logit[k,n].
        window_logits = torch.einsum("bkf,nf->bkn", h, self.clsW) + self.clsb
        return logits, a, window_logits

    def head(self, feats: torch.Tensor) -> torch.Tensor:
        return self.head_detailed(feats)[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(x))


def load_checkpoint(path: str, device: str = "cpu"):
    """-> (model in eval mode on device, checkpoint metadata without the weights)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    bb = build_backbone(ck["arch"], pretrained=False)
    model = RaptorClassifier(bb, F_dim=bb.num_features)
    model.load_state_dict(ck.pop("model"), strict=True)
    return model.eval().to(device), ck


# ============================================================================
# Metrics
# ============================================================================
def macro_auc(y_true: np.ndarray, y_prob: np.ndarray, labels=LAB):
    """Mean ROC-AUC over findings that have both classes present, plus the per-finding dict."""
    from sklearn.metrics import roc_auc_score

    aucs = {}
    for j, name in enumerate(labels):
        yj = y_true[:, j].astype(int)
        if len(set(yj)) > 1:
            aucs[name] = float(roc_auc_score(yj, y_prob[:, j]))
    return float(np.mean(list(aucs.values()))), aucs
