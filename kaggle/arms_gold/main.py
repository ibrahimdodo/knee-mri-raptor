"""Kaggle job: every public Raptor CoAtNet arm on the 58 gold studies, for the ensemble study (Phase 7).

Arm settings are those of the public ensemble (maxspan-v5 at 336 px, v10 = our checkpoint, v8 on the older
44-slice layout). For each checkpoint and both channel orders ("reverse" = slices c+1, c, c-1 in the RGB
channels, the public stack's extra view), it saves:
  feats_<arm>_<normal|flip>.npy   [58, K, 1024] float16, all window centres of that geometry
  head_<arm>.pt                   the checkpoint's head weights, so any window subset can be re-scored locally
  meta_<arm>.json                 what the checkpoint stores about itself (gold AUCs, epoch)
The 64-slice layout has 62 window centres and the 44-slice one 42; smaller window counts are subsets.
"""
import gc
import json
import os
import time

import numpy as np
import pandas as pd
import torch

import raptor_core as rc  # replaced by the build script

INPUT = os.environ.get("RAPTOR_INPUT", "/kaggle/input")
OUT = os.environ.get("RAPTOR_OUT", "/kaggle/working")
CHUNK = 16
SLOTS44 = (("Sagittal", 1, 12), ("Sagittal", 0, 10), ("Coronal", 1, 8), ("Coronal", 0, 6), ("Axial", -1, 8))
GEOMS = {
    "g336": rc.Geometry(img=336),                                          # maxspan-v5
    "g384": rc.NATIVE384_DENSE,                                            # native384dense-v10 (ours)
    "g384_44": rc.Geometry(img=384, span_lo=0.06, span_hi=0.94, slots=SLOTS44),   # native384-v8
}
ARMS = [("v5", "raptor_ft_coatnet_v5_full_swa.pt", "g336"),
        ("v10", "raptor_ft_coatnet_v10_full.pt", "g384"),
        ("v8", "raptor_ft_coatnet_v8_full_swa.pt", "g384_44")]

torch.backends.cudnn.benchmark = True


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def find_root():
    for b in (f"{INPUT}/competitions/rsna-knee-abnormality-detection", f"{INPUT}/rsna-knee-abnormality-detection"):
        if os.path.exists(b + "/train.csv"):
            return b
    raise SystemExit("competition data not attached")


def find_file(name):
    for d, dirs, fs in os.walk(INPUT):
        dirs[:] = [x for x in dirs if "competition" not in x and x != "rsna-knee-abnormality-detection"]
        if name in fs:
            return os.path.join(d, name)
    raise SystemExit(f"{name} not attached")


@torch.no_grad()
def encode(model, windows, dev):
    out = []
    for i in range(0, len(windows), CHUNK):
        xb = windows[i:i + CHUNK].to(dev)
        with torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
            out.append(model.backbone(xb).float().cpu())
    return torch.cat(out).numpy().astype(np.float16)


def main():
    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    root = find_root()
    tr = pd.read_csv(root + "/train.csv")
    tr["StudyInstanceUID"] = tr["StudyInstanceUID"].astype(str)
    ser = pd.read_csv(root + "/train_series.csv")
    ser["StudyInstanceUID"] = ser["StudyInstanceUID"].astype(str)
    ser["SeriesInstanceUID"] = ser["SeriesInstanceUID"].astype(str)
    SER = {k: v.to_dict("records") for k, v in ser.groupby("StudyInstanceUID", sort=False)}
    gold = tr[tr[rc.LAB].notna().all(axis=1)].reset_index(drop=True)
    ids = gold["StudyInstanceUID"].tolist()
    gold.to_csv(f"{OUT}/gold_labels.csv", index=False)
    log(f"{len(ids)} gold studies | device {dev}")

    for name, fname, gkey in ARMS:
        geom = GEOMS[gkey]
        model, ck = rc.load_checkpoint(find_file(fname), dev)
        meta = {k: v for k, v in ck.items() if k in ("gold_auc", "aucs", "epoch", "res", "arch", "src", "swa_over")}
        json.dump(meta, open(f"{OUT}/meta_{name}.json", "w"), indent=1, default=str)
        torch.save({k: v.cpu() for k, v in model.state_dict().items() if not k.startswith("backbone.")},
                   f"{OUT}/head_{name}.pt")
        n_centres = geom.n_slices - 2
        feats = {v: np.zeros((len(ids), n_centres, model.clsW.shape[1]), np.float16) for v in ("normal", "flip")}
        for i, sid in enumerate(ids):
            vol, mask, _ = rc.build_study(f"{root}/train_series/{sid}", SER.get(sid, []), geom)
            windows = rc.make_windows(vol, rc.eval_centers(mask, n_centres), res=int(ck["res"]))
            feats["normal"][i] = encode(model, windows, dev)
            feats["flip"][i] = encode(model, windows.flip(1).contiguous(), dev)
        for v, f in feats.items():
            np.save(f"{OUT}/feats_{name}_{v}.npy", f)
        log(f"{name}: {geom.img}px, {geom.n_slices} slices, span {geom.span_lo}-{geom.span_hi} | stored gold AUC "
            f"{ck.get('gold_auc')} | {time.time() - t0:.0f}s")
        del model, feats
        gc.collect()
        if dev == "cuda":
            torch.cuda.empty_cache()
    log(f"DONE {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
