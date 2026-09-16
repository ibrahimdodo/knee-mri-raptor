"""Backbone features for all 58 gold stacks under acquisition-like perturbations (runs on MPS, ~4 min each).

Writes outputs/local/feats_<condition>.npy [58, 62, 1024] float16. `mps_base` is the unperturbed stack computed
on the same device, so every perturbation is compared against features of identical precision.
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import ablation as ab
import raptor_core as rc

CONDITIONS = {
    "mps_base": lambda v: v,
    "zoom_1.25": lambda v: ab.zoom(v, 1.25),
    "zoom_1.5": lambda v: ab.zoom(v, 1.5),
    "gamma_0.67": lambda v: ab.gamma(v, 0.67),
    "gamma_1.5": lambda v: ab.gamma(v, 1.5),
}

run = ROOT / "outputs/kaggle/gold_run"
out = ROOT / "outputs/local"; out.mkdir(parents=True, exist_ok=True)
vols = np.load(run / "gold_vols_A.npy", mmap_mode="r")
device = "mps" if torch.backends.mps.is_available() else "cpu"
model, _ = rc.load_checkpoint(str(ROOT / "models/raptor_ft_coatnet_v10_full.pt"), device)
bank = ab.FeatureBank(model, np.zeros((1, 62, 1024), np.float32), device)
for name, fn in CONDITIONS.items():
    path = out / f"feats_{name}.npy"
    if path.exists():
        print(f"{name}: exists, skipped", flush=True); continue
    t0 = time.time()
    feats = np.zeros((len(vols), 62, 1024), np.float16)
    for i in range(len(vols)):
        v = fn(np.asarray(vols[i]))
        feats[i] = bank._encode(rc.make_windows(v, list(range(1, 63)), res=384))
    np.save(path, feats)
    print(f"{name}: {time.time() - t0:.0f}s", flush=True)
