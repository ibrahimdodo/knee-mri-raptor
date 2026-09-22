# Picking this up again

Written 2026-09-22, when the project was paused. The README has the findings; this file has the state.

## Where things stand

- Best submission: **0.932** on the public leaderboard (four CoAtNet checkpoints averaged), against 0.927 for the
  released checkpoint alone and 0.941 for the public fork stack. Competition closes **22 October 2026**.
- Phases 0-8 are finished and written up. The write-up doc is linked from the README.
- The repo is on GitHub (private): `git@github.com:ibrahimdodo/knee-mri-raptor.git`, account `ibrahimdodo`, SSH already
  works. Push with `git push`.

## What lives only on this Mac

Nothing below is in git. The sizes are what they cost to keep, and the notes are what they cost to rebuild.

| Path | Size | To rebuild |
|---|---|---|
| `outputs/kaggle/gold_run/` | 543 MB | Rerun `gold_run` on Kaggle, about 6 min of T4 time |
| `outputs/kaggle/train_feats/` | 536 MB | Rerun `train_feats` + `train_feats_b`, about 1.5 h each |
| `outputs/kaggle/dino_feats/` | 402 MB | Rerun `dino_feats` + `dino_feats_b`, about 50 min each |
| `outputs/kaggle/arms_gold/` | 41 MB | Rerun `arms_gold`, about 10 min |
| `outputs/local/` | 57 MB | Trained heads and cached ablations: `scripts/train_heads.py`, minutes |
| `models/raptor_ft_coatnet_v10_full.pt` | 279 MB | `kaggle datasets download dreaddevelopment/raptor-knee-native384dense -p models --unzip` |
| `data/raw/`, `data/soft_labels/` | 9 MB | Kaggle: competition CSVs (rules must be accepted) and `dreaddevelopment/rsna-knee-labels` |
| `.venv/` | 1.5 GB | `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt` |

Python 3.13, PyTorch on MPS. Before any Kaggle CLI call in a shell:
`export SSL_CERT_FILE=$(.venv/bin/python -c "import certifi;print(certifi.where())")`.

## Kaggle assets owned by this account

Private notebooks, built by `scripts/build_kernel.py <job> [KEY=VALUE ...]` and pushed with `kaggle kernels push -p
kaggle/<job>/build`:

| Notebook | What it does |
|---|---|
| `raptor-knee-gold-reproduction` | Phase 0: the 58 gold studies, three preprocessing variants, saves features |
| `raptor-knee-gold-headers` | Phase 3: DICOM geometry and laterality (CPU only) |
| `raptor-knee-train-features` (+ `-b`) | Phase 6: CoAtNet features for all 4,407 studies, two shards |
| `raptor-knee-arms-gold` | Phase 7: all three public checkpoints on the gold studies |
| `raptor-knee-dino-features` (+ `-b`) | Phase 8: frozen DINOv2 features, two shards |
| `raptor-knee-submission` | Submissions: `VARIANT=A62 / A24 / AB62 / RAPTOR4` |
| `raptor-knee-submission-r3` | Submission using the Phase 6 retrained heads |
| `raptor-knee-submission-raptor4` | The 0.932 submission (four arms, public weights) |

Also a private dataset, `ibrahimdodo/raptor-knee-heads`, holding the three R3 heads.

To submit again: build the notebook, push it, let it run, then
`kaggle competitions submit rsna-knee-abnormality-detection -k <notebook> -v <version> -f submission.csv -m "..."`.
Five submissions a day; the four-arm run takes about 4.4 hours on the hidden test set.

## The one open thread

Phase 8 showed an ensemble member must be **different and comparably good**. A frozen DINOv2 was different (score
correlation 0.65) but 0.15 AUC too weak, so it earned no weight. The untried step is to **fine-tune DINOv2-small on
this data** (unfreeze the last blocks, train on the soft labels), which is what the public stack's 20 DINOv2 members
are. That would test whether a strong, genuinely different member delivers the remaining +0.009 to 0.941.

Rough cost: 1.5-2 GPU-hours per fold on a T4, plus the training script; Kaggle allows 30 GPU-hours a week.
`src/head_training.py` already holds the recipe that Phase 6 settled on (no positive weighting, unsure labels masked),
and `kaggle/dino_feats/main.py` already builds the stacks and loads the mounted DINOv2 model.

## Things that cost time to rediscover

- Stored per-finding AUCs in a checkpoint are a fingerprint: AUC x positives x negatives is a whole number, and
  matching those counts catches configuration drift (it found that the stored scores used 24 windows, not 62).
- The 58 gold studies resolve nothing below about 0.012; use paired bootstrap, and choose on held-out training studies.
- Reports are in nine languages, not Spanish: `src/reports.py` detects them.
- A study with no usable images must fail loudly; otherwise the model scores a stack of black slices and looks confident.
