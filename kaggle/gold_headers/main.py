"""Kaggle job (CPU): slice geometry and laterality for every series of the 58 labelled studies.

Phase 3 needs to know, for each sagittal slot, whether increasing slice index moves towards the medial or
the lateral side of the knee. That depends on two things the pixel stacks do not carry:
  - the slice normal (from ImageOrientationPatient), whose x-component says whether index increases
    towards the patient's left or right, and
  - which knee was scanned (Laterality / ImageLaterality, or failing that the series description).

Writes gold_series_geometry.csv: one row per series with the header fields, the slice normal, and the
position (mm along the normal) of every slice in the same order raptor_core.order_series uses.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

import raptor_core as rc  # replaced by the build script

INPUT = os.environ.get("RAPTOR_INPUT", "/kaggle/input")
OUT = os.environ.get("RAPTOR_OUT", "/kaggle/working")
TAGS = ("Laterality", "ImageLaterality", "BodyPartExamined", "PatientPosition", "SeriesDescription",
        "ProtocolName", "StudyDescription", "Manufacturer", "MagneticFieldStrength")


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def find_competition_root():
    for b in (f"{INPUT}/competitions/rsna-knee-abnormality-detection",
              f"{INPUT}/rsna-knee-abnormality-detection"):
        if os.path.exists(b + "/train.csv"):
            return b
    raise SystemExit("competition data not attached")


def series_geometry(series_dir):
    import pydicom

    files, med_ps = rc.order_series(series_dir)
    rec = {"n_files": len(files), "pixel_spacing": med_ps}
    positions, ipps, normal = [], [], None
    for i, (f, _) in enumerate(files):
        h = pydicom.dcmread(f, stop_before_pixels=True)
        if i == 0:
            rec.update({t: (None if getattr(h, t, None) is None else str(getattr(h, t))) for t in TAGS})
            iop = getattr(h, "ImageOrientationPatient", None)
            if iop is not None and len(iop) == 6:
                rec["iop"] = json.dumps([float(v) for v in iop])
                normal = np.cross(np.array(iop[:3], float), np.array(iop[3:], float))
                rec.update(normal_x=normal[0], normal_y=normal[1], normal_z=normal[2])
        ipp = getattr(h, "ImagePositionPatient", None)
        if ipp is not None and normal is not None:
            ipps.append([float(v) for v in ipp])
            positions.append(float(np.dot(np.array(ipp, float), normal)))
    if positions:
        rec["positions"] = json.dumps([round(p, 3) for p in positions])
        rec["slice_step_mm"] = float(np.median(np.diff(positions))) if len(positions) > 1 else None
        rec["ipp_mean_x"], rec["ipp_mean_y"], rec["ipp_mean_z"] = np.mean(ipps, axis=0).tolist()
    return rec


def main():
    t0 = time.time()
    root = find_competition_root()
    tr = pd.read_csv(root + "/train.csv")
    ser = pd.read_csv(root + "/train_series.csv")
    for df in (tr, ser):
        df["StudyInstanceUID"] = df["StudyInstanceUID"].astype(str)
    ser["SeriesInstanceUID"] = ser["SeriesInstanceUID"].astype(str)
    gold_ids = tr[tr[rc.LAB].notna().all(axis=1)]["StudyInstanceUID"].tolist()
    series_root = root + "/train_series"

    rows = []
    for i, sid in enumerate(gold_ids):
        for r in ser[ser.StudyInstanceUID == sid].to_dict("records"):
            try:
                rows.append({**r, **series_geometry(f"{series_root}/{sid}/{r['SeriesInstanceUID']}")})
            except Exception as e:
                rows.append({**r, "error": f"{type(e).__name__}: {e}"})
        if (i + 1) % 10 == 0:
            log(f"{i + 1}/{len(gold_ids)} studies | {time.time() - t0:.0f}s")
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/gold_series_geometry.csv", index=False)
    log("series", len(df), "| errors", int(df.get("error", pd.Series(dtype=str)).notna().sum()))
    for t in ("Laterality", "ImageLaterality", "BodyPartExamined", "PatientPosition"):
        log(t, df[t].value_counts(dropna=False).head(6).to_dict())
    log(f"DONE {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
