"""Kaggle job: reproduce raptor_ft_coatnet_v10_full.pt on the 58 radiologist-labelled studies.

Runs where the DICOMs live (no 600 GB download) and writes everything the local analysis needs:
  gold_labels.csv            the 58 labelled rows of train.csv
  gold_series_meta.csv       one row per series in those studies (plane, sequence, scanner, size)
  gold_slots.json            which series filled each slot, and which slice indices were picked
  gold_vols_A.npy / masks    the 58 x 64 x 384 x 384 stacks for the checkpoint's own geometry
  feats_<run>.npy            backbone features, 58 x 62 x 1024 (float16), one per run
  preds_<run>.csv            study-level probabilities per run
  summary.json               per-finding AUC per run next to the AUCs stored in the checkpoint

Built into a single notebook script by scripts/build_kernel.py (raptor_core.py is pasted above).
"""
import gc
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

import raptor_core as rc  # replaced by the build script

INPUT = os.environ.get("RAPTOR_INPUT", "/kaggle/input")      # overridable for the local dry run
OUT = os.environ.get("RAPTOR_OUT", "/kaggle/working")
K_EVAL = 62
CHUNK = 16          # windows per forward pass; bounds T4 memory
RUNS = [            # (name, geometry key, input normalisation)
    ("A_imagenet", "A", "imagenet"),
    ("A_none", "A", "none"),
    ("B_imagenet", "B", "imagenet"),
]
GEOMS = {
    "A": rc.NATIVE384_DENSE,                                   # 2-98% span, stored at 384
    "B": rc.Geometry(img=384, span_lo=0.06, span_hi=0.94),     # older span, same size
}


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def find_competition_root():
    for b in (f"{INPUT}/competitions/rsna-knee-abnormality-detection",
              f"{INPUT}/rsna-knee-abnormality-detection"):
        if os.path.exists(b + "/train.csv"):
            return b
    for d, dirs, fs in os.walk(INPUT):
        dirs[:] = [x for x in dirs if x not in ("train_series", "test_series", "train_images", "test_images")]
        if "train.csv" in fs and "train_series.csv" in fs:
            return d
    raise SystemExit("competition data not attached")


def find_checkpoint(name="raptor_ft_coatnet_v10_full.pt"):
    for d, dirs, fs in os.walk(INPUT):
        # never descend into the competition's DICOM tree: hundreds of thousands of files
        dirs[:] = [x for x in dirs if "competition" not in x and x != "rsna-knee-abnormality-detection"]
        if name in fs:
            return os.path.join(d, name)
    raise SystemExit(f"{name} not attached")


def series_metadata(study_dir, sid, rows):
    """Header of the first file of every series, for describing what the gold studies contain."""
    import glob
    import pydicom

    out = []
    for r in rows:
        files = sorted(glob.glob(f"{study_dir}/{r['SeriesInstanceUID']}/*.dcm"))
        rec = dict(r, StudyInstanceUID=sid, n_files=len(files))
        if files:
            try:
                h = pydicom.dcmread(files[0], stop_before_pixels=True)
                for tag in ("SeriesDescription", "Manufacturer", "ManufacturerModelName",
                            "MagneticFieldStrength", "SliceThickness", "SpacingBetweenSlices",
                            "RepetitionTime", "EchoTime", "InversionTime", "Rows", "Columns",
                            "ScanningSequence", "SequenceName"):
                    v = getattr(h, tag, None)
                    rec[tag] = None if v is None else str(v)
                ps = getattr(h, "PixelSpacing", None)
                rec["PixelSpacing"] = None if ps is None else float(ps[0])
            except Exception as e:
                rec["header_error"] = f"{type(e).__name__}: {e}"
        out.append(rec)
    return out


@torch.no_grad()
def extract_features(model, vol, mask, norm, dev):
    centers = rc.eval_centers(mask, K_EVAL)
    x = rc.make_windows(vol, centers, res=vol.shape[-1], norm=norm)
    feats = []
    for i in range(0, len(x), CHUNK):
        xb = x[i:i + CHUNK].to(dev)
        try:
            with torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
                f = model.backbone(xb)
        except RuntimeError:
            torch.cuda.empty_cache()
            f = model.backbone(xb.float())
        feats.append(f.float().cpu())
    return torch.cat(feats).numpy(), centers


def main():
    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    log(f"device {dev} | torch {torch.__version__} | "
        f"{torch.cuda.get_device_name(0) if dev == 'cuda' else ''}")

    root = find_competition_root()
    series_root = root + "/train_series" if os.path.isdir(root + "/train_series") else root + "/train_images"
    tr = pd.read_csv(root + "/train.csv")
    ser = pd.read_csv(root + "/train_series.csv")
    log("root", root, "| files:", sorted(os.listdir(root)))
    log("train.csv", tr.shape, list(tr.columns))
    log("train_series.csv", ser.shape, list(ser.columns))
    log(ser.head(8).to_string())

    for df in (tr, ser):
        df["StudyInstanceUID"] = df["StudyInstanceUID"].astype(str)
    ser["SeriesInstanceUID"] = ser["SeriesInstanceUID"].astype(str)
    SER = {k: v.to_dict("records") for k, v in ser.groupby("StudyInstanceUID", sort=False)}

    # Same definition of the labelled set as the training script: every finding column filled in.
    gold = tr[tr[rc.LAB].notna().all(axis=1)].reset_index(drop=True)
    gold_ids = gold["StudyInstanceUID"].tolist()
    Y = gold[rc.LAB].values.astype(np.float32)
    log(f"gold studies {len(gold_ids)} | positives per finding:",
        dict(zip(rc.LAB, Y.astype(int).sum(0).tolist())))
    gold.to_csv(f"{OUT}/gold_labels.csv", index=False)

    meta = []
    for sid in gold_ids:
        meta += series_metadata(f"{series_root}/{sid}", sid, SER.get(sid, []))
    pd.DataFrame(meta).to_csv(f"{OUT}/gold_series_meta.csv", index=False)
    log(f"series metadata: {len(meta)} series | {time.time() - t0:.0f}s")

    ck_path = find_checkpoint()
    model, ck = rc.load_checkpoint(ck_path, dev)
    log("checkpoint", ck_path, {k: v for k, v in ck.items() if k != "aucs"})

    summary = {"checkpoint": {"gold_auc": ck["gold_auc"], "aucs": ck["aucs"], "epoch": ck["epoch"]},
               "runs": {}, "n_gold": len(gold_ids), "k_eval": K_EVAL}
    slots_out = {}
    for gkey, geom in GEOMS.items():
        runs = [r for r in RUNS if r[1] == gkey]
        vols = np.zeros((len(gold_ids), geom.n_slices, geom.img, geom.img), np.uint8)
        masks = np.zeros((len(gold_ids), geom.n_slices), np.uint8)
        for i, sid in enumerate(gold_ids):
            vols[i], masks[i], info = rc.build_study(f"{series_root}/{sid}", SER.get(sid, []), geom)
            slots_out.setdefault(gkey, {})[sid] = info
            if (i + 1) % 10 == 0:
                log(f"  geometry {gkey}: built {i + 1}/{len(gold_ids)} | {time.time() - t0:.0f}s")
        log(f"geometry {gkey}: empty slices per study min/median/max",
            (geom.n_slices - masks.sum(1)).min(), int(np.median(geom.n_slices - masks.sum(1))),
            (geom.n_slices - masks.sum(1)).max())
        if gkey == "A":
            np.save(f"{OUT}/gold_vols_A.npy", vols)
            np.save(f"{OUT}/gold_masks_A.npy", masks)

        for name, _, norm in runs:
            feats = np.zeros((len(gold_ids), K_EVAL, model.clsW.shape[1]), np.float16)
            probs = np.zeros((len(gold_ids), len(rc.LAB)), np.float32)
            centers_all = []
            for i in range(len(gold_ids)):
                f, centers = extract_features(model, vols[i], masks[i], norm, dev)
                feats[i] = f
                centers_all.append(centers)
                with torch.no_grad():
                    logits = model.head(torch.from_numpy(f)[None].to(dev))
                probs[i] = torch.sigmoid(logits)[0].cpu().numpy()
            mauc, aucs = rc.macro_auc(Y, probs)
            summary["runs"][name] = {"geometry": gkey, "norm": norm, "macro_auc": mauc, "aucs": aucs}
            np.save(f"{OUT}/feats_{name}.npy", feats)
            pd.DataFrame(probs, columns=rc.LAB).assign(StudyInstanceUID=gold_ids).to_csv(
                f"{OUT}/preds_{name}.csv", index=False)
            log(f"RUN {name}: macro-AUC {mauc:.4f} (checkpoint says {ck['gold_auc']:.4f}) "
                f"| {time.time() - t0:.0f}s")
            summary["runs"][name]["centers"] = centers_all[0]
        del vols, masks
        gc.collect()

    with open(f"{OUT}/gold_slots.json", "w") as fh:
        json.dump(slots_out, fh)
    with open(f"{OUT}/summary.json", "w") as fh:
        json.dump(summary, fh, indent=1)

    rows = [{"finding": k, "checkpoint": v, **{n: r["aucs"].get(k) for n, r in summary["runs"].items()}}
            for k, v in ck["aucs"].items()]
    log("\n" + pd.DataFrame(rows).round(4).to_string(index=False))
    log(f"DONE {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
