# Knee MRI Raptor: reproduce and understand

A study of a public checkpoint for the Kaggle competition
[RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection):
[`raptor_ft_coatnet_v10_full.pt`](https://www.kaggle.com/datasets/dreaddevelopment/raptor-knee-native384dense)
by Dread Development. It is a CoAtNet-384 that scores twelve knee findings per MRI study, and it
reports a macro-AUC of 0.9174 on the 58 studies that carry radiologist labels.

The goal is not to beat it but to find out whether that number holds up and why the model works:
which slices it relies on for each finding, where it fails, and how much of the reported score
survives once its own model selection on those same 58 studies is accounted for.

## Plan

| Phase | Question | Where it runs |
|---|---|---|
| 0 | Can the stored per-finding AUCs be reproduced exactly, and which preprocessing produces them? | Kaggle T4 (`kaggle/gold_run`) |
| 1 | What is in the 58 labelled studies: prevalence, scanners, sequences, missing slots? | Mac |
| 2 | How uncertain is 0.917? Bootstrap intervals per finding, and the optimism from selecting the epoch on the same set | Mac |
| 3 | Where does the model look? Per-finding attention over the 62 windows and per-window contributions | Mac, from saved features |
| 4 | What does it depend on? Window count, slot ablation (drop a plane), span and normalisation | Mac + Kaggle |
| 5 | Error analysis: the worst false positives and negatives, viewed slice by slice | Mac |

## Results so far

### Phase 0: reproduction (done, 2026-09-16)

The checkpoint stores per-finding AUCs on the 58 labelled studies. Each AUC times (positives x
negatives) is an exact integer, so it fixes how many positive/negative pairs the model ranked wrongly
per finding: a fingerprint much stricter than a matching macro-AUC.

| Variant | Windows | Macro-AUC | Wrong pairs | Pairs from checkpoint fingerprint |
|---|---|---|---|---|
| Checkpoint as stored | | 0.9174 | 731 | |
| **2-98% span, ImageNet norm** | **24** | **0.9180** | **724** | **7 (6/12 findings exact)** |
| 2-98% span, ImageNet norm | 62 | 0.9171 | 726 | 71 |
| 2-98% span, no norm | 62 | 0.8786 | 1030 | |
| 6-94% span, ImageNet norm | 62 | 0.9219 | 685 | 102 |

- Preprocessing in `src/raptor_core.py` is confirmed, including ImageNet normalisation.
- The stored score was computed with **24** evaluation windows (the training script's default), not the
  62 the dataset description mentions. With 62 the score is essentially unchanged.
- Re-running in float32, bfloat16 and float16 moves findings by 1-2 pairs, the same size as the residual,
  so the remaining gap is numerical.
- Window count matters below 24: 0.843 at 8 windows, 0.890 at 12, 0.911 at 16.

## How the model sees a study

1. **Five fixed slots, 64 slices.** 18 sagittal (fluid-sensitive preferred), 14 sagittal (not fluid),
   12 coronal (fluid), 8 coronal (not fluid), 12 axial. Missing series leave zeros.
2. **Per series:** sort by physical position, take evenly spaced slices over 2-98% of the stack,
   window intensities to the series' 2nd-98th percentile, crop 140 mm around the centre, resize to 384.
3. **2.5D windows:** slices (c-1, c, c+1) become the RGB channels of one image; all 62 centres are used.
4. **Backbone:** `coatnet_rmlp_2_rw_384` turns each window into a 1024-d vector.
5. **Per-finding attention pooling:** each finding gets its own softmax over windows, so an ACL tear
   and patellofemoral OA can draw on different slices. The study logit is exactly the
   attention-weighted sum of per-window logits, which is what makes phase 3 possible.

## Layout

```
src/raptor_core.py        preprocessing, windows, model, metrics (single source of truth)
kaggle/gold_run/main.py   Kaggle job: gold-set reproduction + feature export
scripts/build_kernel.py   pastes raptor_core.py into a single-file Kaggle script
tests/                    synthetic-DICOM tests of the preprocessing
models/                   checkpoint (not committed)
outputs/kaggle/           downloaded Kaggle job outputs (not committed: competition data derivative)
```

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/kaggle datasets download dreaddevelopment/raptor-knee-native384dense -p models --unzip
.venv/bin/python -m pytest -q tests
```

Competition data requires accepting the competition rules on Kaggle first.

```bash
.venv/bin/python scripts/build_kernel.py gold_run
.venv/bin/kaggle kernels push -p kaggle/gold_run/build
.venv/bin/kaggle kernels output ibrahimdodo/raptor-knee-gold-reproduction -p outputs/kaggle/gold_run
```
