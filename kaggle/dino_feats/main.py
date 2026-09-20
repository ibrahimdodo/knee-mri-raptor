"""Kaggle job: frozen DINOv2 features for every training study, for the Phase 8 diversity test.

Phase 7 showed that averaging models pays in proportion to how much they disagree, and that two CoAtNets by the
same author disagree only a little. DINOv2 is a different family: a self-supervised vision transformer, not a
convolution-attention hybrid trained on ImageNet labels. This job encodes the same 64-slice stacks with it, so an
attention head can be trained on those features locally exactly as in Phase 6, and the two families compared.

Windows are resized to 336 px, a multiple of DINOv2's patch size of 14. Each window becomes the CLS token
concatenated with the mean patch token (768 dimensions for the small model), the pooling the public notebooks use.

SHARD / SHARDS split the studies, as in kaggle/train_feats.
"""
import gc
import os
import time

import numpy as np
import pandas as pd
import torch

import raptor_core as rc  # replaced by the build script

SHARD = int(os.environ.get("RAPTOR_SHARD", "0"))
SHARDS = int(os.environ.get("RAPTOR_SHARDS", "2"))
K_EVAL = 62
RES = 336
CHUNK = 32
INPUT = os.environ.get("RAPTOR_INPUT", "/kaggle/input")
OUT = os.environ.get("RAPTOR_OUT", "/kaggle/working")

torch.backends.cudnn.benchmark = True


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def find_root():
    for b in (f"{INPUT}/competitions/rsna-knee-abnormality-detection", f"{INPUT}/rsna-knee-abnormality-detection"):
        if os.path.exists(b + "/train.csv"):
            return b
    raise SystemExit("competition data not attached")


def find_dinov2():
    """The mounted Kaggle model: a directory holding a Hugging Face config.json and weights."""
    for d, dirs, fs in os.walk(INPUT):
        dirs[:] = [x for x in dirs if "competition" not in x and x != "rsna-knee-abnormality-detection"]
        if "config.json" in fs and "dino" in d.lower():
            return d
    raise SystemExit("no DINOv2 model directory attached")


@torch.no_grad()
def encode(model, windows, dev):
    """[K, 768] float16: CLS token and mean patch token of each window."""
    out = []
    for i in range(0, len(windows), CHUNK):
        xb = windows[i:i + CHUNK].to(dev)
        with torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
            h = model(pixel_values=xb).last_hidden_state
        out.append(torch.cat([h[:, 0], h[:, 1:].mean(1)], dim=-1).float().cpu())
    return torch.cat(out).numpy().astype(np.float16)


def main():
    t0 = time.time()
    from transformers import AutoModel

    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    root = find_root()
    tr = pd.read_csv(root + "/train.csv")
    tr["StudyInstanceUID"] = tr["StudyInstanceUID"].astype(str)
    ser = pd.read_csv(root + "/train_series.csv")
    ser["StudyInstanceUID"] = ser["StudyInstanceUID"].astype(str)
    ser["SeriesInstanceUID"] = ser["SeriesInstanceUID"].astype(str)
    SER = {k: v.to_dict("records") for k, v in ser.groupby("StudyInstanceUID", sort=False)}
    ids = tr["StudyInstanceUID"].tolist()
    mine = ids[SHARD::SHARDS]

    path = find_dinov2()
    model = AutoModel.from_pretrained(path).eval().to(dev)
    dim = model.config.hidden_size * 2
    log(f"shard {SHARD + 1}/{SHARDS}: {len(mine)} of {len(ids)} studies | device {dev} | {path} | {dim}-d features")

    feats = np.zeros((len(mine), K_EVAL, dim), np.float16)
    rows = []
    for i, sid in enumerate(mine):
        rec = {"StudyInstanceUID": sid}
        try:
            vol, mask, _ = rc.build_study(f"{root}/train_series/{sid}", SER.get(sid, []), rc.NATIVE384_DENSE)
            if not mask.any():
                raise ValueError("no usable slices")
            feats[i] = encode(model, rc.make_windows(vol, rc.eval_centers(mask, K_EVAL), res=RES), dev)
            rec["empty_slices"] = int((mask == 0).sum())
            del vol, mask
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"
            log(f"  {sid[:16]} FAILED ({rec['error']})")
        rows.append(rec)
        if (i + 1) % 100 == 0 or i + 1 == len(mine):
            done = time.time() - t0
            log(f"{i + 1}/{len(mine)} | {done / 60:.1f} min | {done / (i + 1):.2f}s per study | "
                f"eta {(len(mine) - i - 1) * done / (i + 1) / 60:.0f} min")
        if (i + 1) % 200 == 0:
            gc.collect()

    np.save(f"{OUT}/dino_feats_shard{SHARD}.npy", feats)
    np.save(f"{OUT}/dino_ids_shard{SHARD}.npy", np.array(mine))
    meta = pd.DataFrame(rows)
    meta.to_csv(f"{OUT}/dino_meta_shard{SHARD}.csv", index=False)
    log(f"wrote {feats.shape} | {int(meta.get('error', pd.Series(dtype=str)).notna().sum())} failed | "
        f"{(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
