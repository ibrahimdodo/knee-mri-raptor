"""Kaggle job: backbone features for every training study, so the head can be retrained locally.

The backbone stays frozen, so its 1024-d output per window is all the head ever sees. Computing it once
here (about 2.3 s per study on a T4) turns head training into minutes of CPU work on a laptop.

SHARD / SHARDS split the studies into equal parts, run as separate versions of this notebook, so a failure
costs one shard rather than the whole set. Each shard writes:
  feats_shard<K>.npy   [n, 62, 1024] float16, window features in eval_centers order
  ids_shard<K>.npy     [n] study UIDs, the row order of the features
  meta_shard<K>.csv    per study: empty slices, which series filled each slot, seconds taken
"""
import os
import time

import numpy as np
import pandas as pd
import torch

import raptor_core as rc  # replaced by the build script

SHARD = int(os.environ.get("RAPTOR_SHARD", "0"))
SHARDS = int(os.environ.get("RAPTOR_SHARDS", "2"))
K_EVAL = 62
CHUNK = 16
INPUT = os.environ.get("RAPTOR_INPUT", "/kaggle/input")
OUT = os.environ.get("RAPTOR_OUT", "/kaggle/working")

torch.backends.cudnn.benchmark = True


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def find_root():
    for b in (f"{INPUT}/competitions/rsna-knee-abnormality-detection",
              f"{INPUT}/rsna-knee-abnormality-detection"):
        if os.path.exists(b + "/train.csv"):
            return b
    for d, dirs, fs in os.walk(INPUT):
        dirs[:] = [x for x in dirs if x not in ("train_series", "test_series", "train_images", "test_images")]
        if "train.csv" in fs:
            return d
    raise SystemExit("competition data not attached")


def find_checkpoint(name="raptor_ft_coatnet_v10_full.pt"):
    for d, dirs, fs in os.walk(INPUT):
        dirs[:] = [x for x in dirs if "competition" not in x and x != "rsna-knee-abnormality-detection"]
        if name in fs:
            return os.path.join(d, name)
    raise SystemExit(f"{name} not attached")


@torch.no_grad()
def encode(model, vol, mask, dev):
    x = rc.make_windows(vol, rc.eval_centers(mask, K_EVAL), res=vol.shape[-1])
    out = []
    for i in range(0, len(x), CHUNK):
        xb = x[i:i + CHUNK].to(dev)
        try:
            with torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
                out.append(model.backbone(xb).float().cpu())
        except RuntimeError:
            if dev == "cuda":
                torch.cuda.empty_cache()
            out.append(model.backbone(xb.float()).float().cpu())
    return torch.cat(out).numpy().astype(np.float16)


def main():
    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    root = find_root()
    series_root = root + "/train_series"
    tr = pd.read_csv(root + "/train.csv")
    tr["StudyInstanceUID"] = tr["StudyInstanceUID"].astype(str)
    ser = pd.read_csv(root + "/train_series.csv")
    ser["StudyInstanceUID"] = ser["StudyInstanceUID"].astype(str)
    ser["SeriesInstanceUID"] = ser["SeriesInstanceUID"].astype(str)
    SER = {k: v.to_dict("records") for k, v in ser.groupby("StudyInstanceUID", sort=False)}

    ids = tr["StudyInstanceUID"].tolist()
    mine = ids[SHARD::SHARDS]                       # every SHARDS-th study: both shards see the same mix
    log(f"shard {SHARD + 1}/{SHARDS}: {len(mine)} of {len(ids)} studies | device {dev}")

    model, _ = rc.load_checkpoint(find_checkpoint(), dev)
    feats = np.zeros((len(mine), K_EVAL, model.clsW.shape[1]), np.float16)
    rows = []
    for i, sid in enumerate(mine):
        t1 = time.time()
        rec = {"StudyInstanceUID": sid}
        try:
            vol, mask, slots = rc.build_study(f"{series_root}/{sid}", SER.get(sid, []), rc.NATIVE384_DENSE)
            if not mask.any():
                raise ValueError("no usable slices")
            feats[i] = encode(model, vol, mask, dev)
            rec.update(empty_slices=int((mask == 0).sum()),
                       **{f"slot{j}": (s["series"] or "") for j, s in enumerate(slots)})
            del vol, mask
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"
            log(f"  {sid[:16]} FAILED ({rec['error']})")
        rec["seconds"] = round(time.time() - t1, 2)
        rows.append(rec)
        if (i + 1) % 100 == 0 or i + 1 == len(mine):
            done = time.time() - t0
            log(f"{i + 1}/{len(mine)} | {done / 60:.1f} min | {done / (i + 1):.2f}s per study | "
                f"eta {(len(mine) - i - 1) * done / (i + 1) / 60:.0f} min")

    np.save(f"{OUT}/feats_shard{SHARD}.npy", feats)
    np.save(f"{OUT}/ids_shard{SHARD}.npy", np.array(mine))
    meta = pd.DataFrame(rows)
    meta.to_csv(f"{OUT}/meta_shard{SHARD}.csv", index=False)
    failed = int(meta.get("error", pd.Series(dtype=str)).notna().sum())
    log(f"wrote {feats.shape} features | {failed} failed studies | {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
