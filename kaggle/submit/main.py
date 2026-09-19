"""Kaggle submission notebook: score the hidden test studies with raptor_ft_coatnet_v10_full.pt.

Same pipeline as the Phase 0 reproduction, pointed at the test set. VARIANT picks what to submit:

  A62    the checkpoint as its release describes it: 2-98% slice span, 62 windows per study
  A24    the configuration that produced the stored 0.917 (24 windows)
  AB62   the mean of the 2-98% and 6-94% spans at 62 windows (two passes, so about twice the time)
  RAPTOR4  the public ensemble's Raptor branch: v5 (plus its channel-flipped view), v10 and v8, at fixed weights

HEADS swaps the checkpoint's attention head for an average of retrained heads (Phase 6) from an attached dataset.

The metric is ROC-AUC, so only the ranking of each column matters; probabilities are written as they are.
No study is ever dropped: a study that fails preprocessing or inference gets 0.5 everywhere, which is what
the sample submission holds.
"""
import gc
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

import raptor_core as rc  # replaced by the build script

VARIANT = os.environ.get("RAPTOR_VARIANT", "A62")     # the notebook runs with the defaults; env vars are for local tests
SOURCE = os.environ.get("RAPTOR_SOURCE", "test")      # "train" measures throughput on studies we already know
LIMIT = int(os.environ.get("RAPTOR_LIMIT", "0"))      # 0 = every study
HEADS = os.environ.get("RAPTOR_HEADS", "")            # "" = the checkpoint's own head; else a retrained-head prefix
INPUT = os.environ.get("RAPTOR_INPUT", "/kaggle/input")
OUT = os.environ.get("RAPTOR_OUT", "/kaggle/working")
CHUNK = 16
CHECKPOINTS = {"v10": "raptor_ft_coatnet_v10_full.pt",          # ours: dreaddevelopment/raptor-knee-native384dense
               "v5": "raptor_ft_coatnet_v5_full_swa.pt",        # dreaddevelopment/raptor-knee-maxspan
               "v8": "raptor_ft_coatnet_v8_full_swa.pt"}        # dreaddevelopment/raptor-knee-native384
SLOTS44 = (("Sagittal", 1, 12), ("Sagittal", 0, 10), ("Coronal", 1, 8), ("Coronal", 0, 6), ("Axial", -1, 8))
GEOMS = {"A": rc.NATIVE384_DENSE,
         "B": rc.Geometry(img=384, span_lo=0.06, span_hi=0.94),
         "g336": rc.Geometry(img=336),
         "g384_44": rc.Geometry(img=384, span_lo=0.06, span_hi=0.94, slots=SLOTS44)}
# Each arm: (checkpoint, geometry, windows, flip the 3 slice channels, weight). Arms are averaged as probabilities.
PLANS = {
    "A62": [("v10", "A", 62, False, 1.0)],
    "A24": [("v10", "A", 24, False, 1.0)],
    "AB62": [("v10", "A", 62, False, 1.0), ("v10", "B", 62, False, 1.0)],
    # the public ensemble's Raptor branch, weights fixed in advance (Phase 7)
    "RAPTOR4": [("v5", "g336", 62, False, 0.6), ("v5", "g336", 62, True, 0.1),
                ("v10", "A", 62, False, 0.1), ("v8", "g384_44", 42, False, 0.2)],
}
PLAN = PLANS[VARIANT]

torch.backends.cudnn.benchmark = True


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def find_root():
    for b in (f"{INPUT}/competitions/rsna-knee-abnormality-detection",
              f"{INPUT}/rsna-knee-abnormality-detection"):
        if os.path.exists(f"{b}/{SOURCE}.csv"):
            return b
    for d, dirs, fs in os.walk(INPUT):
        dirs[:] = [x for x in dirs if x not in ("train_series", "test_series", "train_images", "test_images")]
        if f"{SOURCE}.csv" in fs:
            return d
    raise SystemExit("competition data not attached")


def find_checkpoint(name="raptor_ft_coatnet_v10_full.pt"):
    for d, dirs, fs in os.walk(INPUT):
        dirs[:] = [x for x in dirs if "competition" not in x and x != "rsna-knee-abnormality-detection"]
        if name in fs:
            return os.path.join(d, name)
    raise SystemExit(f"{name} not attached")


def load_heads(prefix, dev):
    """Retrained attention heads (Phase 6) from an attached dataset: files <prefix>_seed<k>.pt.

    They have exactly the checkpoint head's parameter names, so each loads into a RaptorClassifier whose
    backbone is an identity; at inference their logits are averaged.
    """
    import glob
    paths = []
    for d, dirs, fs in os.walk(INPUT):
        dirs[:] = [x for x in dirs if "competition" not in x and x != "rsna-knee-abnormality-detection"]
        paths += glob.glob(os.path.join(d, f"{prefix}_seed*.pt"))
    if not paths:
        raise SystemExit(f"no heads named {prefix}_seed*.pt attached")
    heads = []
    for p in sorted(paths):
        h = rc.RaptorClassifier(torch.nn.Identity(), F_dim=1024)
        h.load_state_dict(torch.load(p, map_location="cpu"), strict=True)
        heads.append(h.eval().to(dev))
    log(f"loaded {len(heads)} retrained heads: {[os.path.basename(p) for p in sorted(paths)]}")
    return heads


@torch.no_grad()
def encode_study(model, vol, mask, k, dev, res=384, flip=False):
    """Backbone features [k, 1024] for one study, in half precision on GPU, falling back to full precision.

    flip reverses the three slices inside each window (c+1, c, c-1), the public ensemble's extra view.
    """
    x = rc.make_windows(vol, rc.eval_centers(mask, k), res=res)
    if flip:
        x = x.flip(1).contiguous()
    feats = []
    for i in range(0, len(x), CHUNK):
        xb = x[i:i + CHUNK].to(dev)
        try:
            with torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
                feats.append(model.backbone(xb).float().cpu())
        except RuntimeError as e:
            log(f"  half precision failed ({type(e).__name__}), retrying in full precision")
            if dev == "cuda":
                torch.cuda.empty_cache()
            feats.append(model.backbone(xb.float()).float().cpu())
    return torch.cat(feats)


def main():
    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    log(f"variant {VARIANT} | device {dev} | torch {torch.__version__}")

    root = find_root()
    series_root = f"{root}/{SOURCE}_series" if os.path.isdir(f"{root}/{SOURCE}_series") else f"{root}/{SOURCE}_images"
    studies = pd.read_csv(f"{root}/{SOURCE}.csv")
    studies["StudyInstanceUID"] = studies["StudyInstanceUID"].astype(str)
    ids = studies["StudyInstanceUID"].tolist()[:LIMIT or None]
    ser = pd.read_csv(f"{root}/{SOURCE}_series.csv")
    ser["StudyInstanceUID"] = ser["StudyInstanceUID"].astype(str)
    ser["SeriesInstanceUID"] = ser["SeriesInstanceUID"].astype(str)
    SER = {k: v.to_dict("records") for k, v in ser.groupby("StudyInstanceUID", sort=False)}
    columns = list(pd.read_csv(root + "/sample_submission.csv", nrows=1).columns)
    log(f"{len(ids)} {SOURCE} studies | {len(ser)} series | columns {columns}")

    models, res, heads = {}, {}, {}
    for name in dict.fromkeys(arm[0] for arm in PLAN):
        models[name], ck = rc.load_checkpoint(find_checkpoint(CHECKPOINTS[name]), dev)
        assert ck["lab"] == rc.LAB, f"{name}: label order differs from raptor_core.LAB"
        res[name] = int(ck["res"])
        heads[name] = load_heads(HEADS, dev) if (HEADS and name == "v10") else [models[name]]
    weights = np.array([arm[4] for arm in PLAN], np.float64)
    log(f"arms: {[(a[0], a[1], a[2], 'flip' if a[3] else 'normal', a[4]) for a in PLAN]}")

    probs = np.full((len(ids), len(rc.LAB)), 0.5, np.float32)
    empty_slices = np.zeros(len(ids), int)
    failed = []
    for i, sid in enumerate(ids):
        try:
            cache, stacks, arm_probs = {}, {}, []       # each DICOM file is read once per study
            for name, gkey, k, flip, _ in PLAN:
                if gkey not in stacks:
                    stacks[gkey] = rc.build_study(f"{series_root}/{sid}", SER.get(sid, []), GEOMS[gkey], cache=cache)[:2]
                vol, mask = stacks[gkey]
                if not mask.any():
                    # every slice blank: no series matched a slot, or none of the files could be read.
                    # Without this the model would happily score a stack of black images.
                    raise ValueError("no usable slices")
                empty_slices[i] = int((mask == 0).sum())
                f = encode_study(models[name], vol, mask, k, dev, res=res[name], flip=flip)[None].to(dev)
                with torch.no_grad():
                    logit = np.mean([h.head(f)[0].float().cpu().numpy() for h in heads[name]], axis=0)
                arm_probs.append(1 / (1 + np.exp(-logit)))
            probs[i] = np.tensordot(weights, np.stack(arm_probs), axes=1) / weights.sum()
            del cache, stacks
        except Exception as e:
            failed.append(sid)
            probs[i] = 0.5                       # the sample submission's value
            log(f"  study {i} {sid[:16]} FAILED ({type(e).__name__}: {e})")
        if (i + 1) % 25 == 0 or i + 1 == len(ids):
            done = time.time() - t0
            log(f"{i + 1}/{len(ids)} | {done:.0f}s | {done / (i + 1):.1f}s per study")
        if (i + 1) % 200 == 0:
            gc.collect()

    sub = pd.DataFrame(probs, columns=rc.LAB)
    sub.insert(0, "StudyInstanceUID", ids)
    sub = sub[columns]
    assert sub["StudyInstanceUID"].tolist() == ids, "row order drifted"
    assert list(sub.columns) == columns, "column order drifted"
    assert np.isfinite(sub[rc.LAB].values).all(), "non-finite predictions"
    sub.to_csv(f"{OUT}/submission.csv", index=False)
    log(f"wrote submission.csv: {len(sub)} rows, {len(failed)} fallback rows at 0.5 | "
        f"{(empty_slices > 0).sum()} studies missing at least one series | {time.time() - t0:.0f}s")
    log(sub.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
