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
- Checked against the author's training soft labels (`dreaddevelopment/rsna-knee-labels`, 4,349 non-gold studies):
  targets span 0.05-0.95 in 14 levels (hence the compressed logits); effusion and synovitis labels correlate at 0.88,
  so the synovitis entanglement was inherited, while the OA compartments' labels correlate at only 0.36-0.42, so that
  merging is the model's own. Subtracting log(pos_weight), recomputed from those labels with the training script's
  formula, cuts calibration error from 0.20 to 0.06 without any gold labels; one shared slope takes it to 0.04.

### Leaderboard check (2026-09-17)

Submitted through `kaggle/submit` (2-98% span, 62 windows): **public leaderboard 0.927**, against 0.918 on the 58
labelled studies and a 95% interval of 0.894-0.940. The gold-set estimate held up. For scale, 0.927 sits around rank
1,595 of 3,917 teams; 855 teams are at 0.94 or better. Throughput on a T4 was 2.3 s per study.

### Phase 6: retraining the head (done, [notebook](notebooks/06_retrain_head.ipynb))

Backbone features for all 4,407 studies were computed once on Kaggle (`kaggle/train_feats`, two T4 shards, no
failures); the 280 k-parameter attention head was then retrained locally in about 11 s per run, 6 recipes x 3 seeds,
selected on 652 held-out training studies and never on the 58 gold studies.

- Dropping the positive weighting fixes calibration at training time: ECE 0.19 to 0.05.
- Masking the "unsure" soft labels (0.3-0.5) untangles synovitis from effusion: their score correlation over 4,407
  studies falls from 0.97 to 0.77, below the training labels' own 0.88.
- In the 2 x 2 design the two choices do not interact: each moves only its own effect.
- No recipe beats the checkpoint's head on gold macro-AUC; all land about 0.01 lower (0.900-0.907), consistent
  with the checkpoint having been selected on those studies. The gold set cannot resolve it; the leaderboard can.
- **Leaderboard check (2026-09-18):** the R3 heads (3-seed mean, via the private dataset `ibrahimdodo/raptor-knee-heads`)
  scored **0.920** against the checkpoint's 0.927. The gold gap (-0.013) roughly halves on unseen data (-0.007): about
  half was selection optimism, the rest looks like the cost of a fresh head on features co-trained with another. For
  ranking, keep the checkpoint's head; for calibrated, untangled probabilities, R3 costs about 0.007 AUC.

### Phase 7: combining similar models (done, [notebook](notebooks/07_ensemble.ipynb))

The author's three public CoAtNet checkpoints (v5, v10 = ours, v8), each on its own slice layout, plus the public
ensemble's channel-flipped view. All three reproduce their stored AUCs to within 2-6 ranked pairs at 24 windows.

- The flipped view correlates 0.993 with its unflipped model and adds nothing.
- Different checkpoints correlate 0.94-0.96 and averaging pairs gains 0.3-0.9 points, more the more they disagree.
- The pre-registered four-arm ensemble (public weights 0.6/0.1/0.1/0.2) scores 0.926 on gold (+0.009 vs v10, paired CI
  -0.001 to +0.019) and **0.932 on the public leaderboard** (+0.005; about rank 1,575 of 4,008). About half the gold
  gain survives; the rest was selection, since v5 and v8 were built from gold-selected epochs.
- Same-family averaging explains about a third of the public stack's lead (+0.005 of 0.927 -> 0.941).

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
src/head_training.py      attention head on frozen features: recipes, masked loss, training
kaggle/train_feats/       Kaggle job: backbone features for all 4,407 training studies (two shards)
kaggle/arms_gold/         Kaggle job: all three public CoAtNet checkpoints on the 58 gold studies
scripts/train_heads.py    phase 6 recipes x seeds
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

Competition data requires accepting the competition rules on Kaggle first. The Phase 5 soft-label check needs
`.venv/bin/kaggle datasets download dreaddevelopment/rsna-knee-labels -f labels_llm_soft.csv -p data/soft_labels`.

```bash
.venv/bin/python scripts/build_kernel.py gold_run
.venv/bin/kaggle kernels push -p kaggle/gold_run/build
.venv/bin/kaggle kernels output ibrahimdodo/raptor-knee-gold-reproduction -p outputs/kaggle/gold_run
```
