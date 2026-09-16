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
| 3 | Where does the model look? Per-finding attention over the 62 windows, medial/lateral anatomy, in-slice saliency | Mac + Kaggle (`kaggle/gold_headers`) |
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

### Phase 2: how much is 0.917 worth? (done, [notebook](notebooks/02_uncertainty.ipynb))

| | macro-AUC |
|---|---|
| Reproduced (A24) | 0.918 |
| 95% bootstrap interval, 58 studies | 0.894 to 0.940 |
| Epoch-selection optimism (best of 3 to 16 epochs, noise model) | -0.002 to -0.010 |
| Plausible score on fresh studies like these | about 0.91, ± 0.02 |

- Per-finding intervals are up to five times wider than the macro interval: Synovitis 0.67 to 0.91, PF OA 0.71 to 0.95.
  DeLong and bootstrap standard errors agree to within 0.001 per finding.
- The 6-94% span beats 2-98% in 95% of paired resamples (+0.005, CI -0.001 to +0.011): suggestive, not
  conclusive. No finding differs significantly. The checkpoint loses nothing on slices it was not trained on.
- Pairing matters: A62 and B62 correlate at 0.97 across resamples, so their difference has SD 0.003, against
  0.012 for either score alone. The smallest reliably detectable paired difference is about 0.012.
- Choosing the best of the 14 scoring configurations on this set inflates the score by about +0.0035.

### Phase 3: where does the model look? (done, `notebooks/03_where_it_looks.py`)

The executed notebook shows MRI slices from the competition data and is not committed; run it locally.

- Each study logit splits exactly into per-window contributions (attention x window logit, error 2e-6).
- The fluid-sensitive sagittal series takes 37-61% of attention for 11 of 12 findings. Fluid findings (effusion,
  synovitis, Baker's, contusion) give the non-fluid sagittal series 7-9%, while ACL and OA give it 23-28%. MCL and
  contusion draw a third of their attention from coronal fluid-sensitive slices. The second coronal series is
  nearly ignored (1.5-6%).
- Attention narrows when a finding is present, for all 12 findings (lateral meniscus: 11 effective windows).
- Medial and lateral meniscus attention sits at opposite ends of the sagittal stack in 58/58 studies. The medial
  end derived from DICOM geometry and laterality (tag, series text or scanner position) matches the model in
  57/58. The one mismatch is a study whose laterality tag contradicts its scanner position.
- Occlusion saliency lands on the intercondylar notch (ACL), the menisci and the Baker's cyst neck. Grad-CAM on the
  last stage is coarse, and on the last convolutional stage mostly noise. The model's final stages are
  transformers, so gradient maps need an interventional cross-check.

### Phase 4: what does the score depend on? (done, [notebook](notebooks/04_what_it_depends_on.ipynb))

- The attention head is permutation-invariant, so the medial/lateral anatomy from Phase 3 comes from pixels, not
  slice position.
- Removing one series costs at most 0.018 macro-AUC (fluid-sensitive sagittal), then 0.011 (fluid-sensitive coronal).
  The two non-fluid series together add nothing measurable (0.000).
- Without fluid-sensitive series, effusion, fracture, contusion, MCL and Baker's cyst lose 0.15-0.21 AUC. MCL needs
  coronal slices (-0.11 with sagittal only), and ACL needs sagittal (-0.16 with coronal only, 0.58 with axial only).
- Attention share predicts logit movement (Spearman +0.91) but only moderately predicts AUC damage (-0.45):
  influence is not necessity when series are redundant.
- Studies lacking the second coronal series are mostly acute trauma, but the model does not exploit missing series
  (at most 0.09 logit shift). The protection comes from low attention on blank windows, not from neutral blank logits.
- Two random windows per study give 0.77, sixteen 0.90. Contrast shifts are harmless. Zoom 1.25x keeps AUC but moves
  probabilities about 5 points. Zoom 1.5x costs 0.018, mostly MCL (-0.09).

### Phase 5: where and why it is wrong (done, [notebook](notebooks/05_errors.ipynb))

- Errors are concentrated: 25% of the 726 wrong pairs come from 17 of 696 study-finding cells.
- Nothing about site (report language), vendor, field strength, empty slots or magnified series predicts which studies
  are hard. Knees with more positive findings are slightly harder (Spearman +0.26, p = 0.05).
- Reports are in nine languages across the training set (English 1,736, Spanish 708, Turkish 546, Croatian 407,
  Greek 321, German 262, Bulgarian 220, Dutch 147, French 59); the 58 labelled studies cover eight.
- Synovitis scores track effusion (score correlation 0.98 against label correlation 0.40). Ranking synovitis by the
  effusion score gives the same AUC (0.800 vs 0.805). The OA compartments are partly merged (score correlations about
  0.8) but each own score still beats its best sibling by 0.07-0.10.
- The 14 worst errors, read in their original languages: 1 clear miss (undisplaced eminence fracture), 4 where label
  and report disagree (the model sides with the report each time), 4 focal cartilage lesions labelled OA, 3 effusion
  scored as synovitis, 2 borderline synovitis.
- Probabilities are shifted up by positive weighting and compressed by soft labels. An offset per finding plus one
  shared slope (1.77), fitted leave-one-out, cuts calibration error from 0.20 to 0.03.

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
kaggle/gold_headers/      Kaggle job (CPU): slice geometry and laterality per series
src/evaluation.py         bootstrap, DeLong, selection optimism
src/explain.py            window decomposition, Grad-CAM, occlusion, laterality
src/ablation.py           series removal (drop / blank), zoom and contrast perturbations
src/reports.py            report language detection
scripts/robustness_features.py   re-encode all windows under perturbations (MPS)
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
