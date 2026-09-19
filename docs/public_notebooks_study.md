# What the top public notebooks do (2026-09-19)

Read from nine of the highest-scoring public notebooks in the competition. Their sources are in
`outputs/notebook_study/` (git-ignored: other people's code).

## The public frontier is one ensemble

- **568 teams sit at exactly 0.941**, and 184 are above it (of 3,994). A 0.941 fork ties anywhere from rank 185 to 752.
- Almost every top public notebook attaches the **same 12 data sources**: they are forks of Mattia Angeli's
  "Bend the Knee to the Dinosaurs", adjusted by one weight or one label at a time.
- Nothing public goes past 0.941. The 40 teams at 0.95 or more are not running a public recipe.

## Its ingredients

| Family | Members | Notes |
|---|---|---|
| DINOv2 "transformer" | 20 DINOv2-small checkpoints (5 folds) | Six series slots at 336 px, per-diagnosis attention queries over slots; knees flipped to one orientation first |
| DINOv3 | 5 fold checkpoints | Blended 0.45 with DINOv2 0.55 |
| RadImageNet | 10 attention heads on one shared ResNet-50 | Backbone pretrained on medical images; two slot layouts |
| Raptor (CoAtNet) | Dread Development's maxspan-v5, maxspan-v5 with slice order reversed, **native384dense-v10 (ours, weight 0.10)**, native384-v8, plus Mattia's gated CoAtNets | Rank-averaged inside the branch |
| Blend | Per-finding weights between branches | e.g. lateral meniscus 100% Raptor; ACL, lateral OA, fracture 75% Raptor |

Everything is combined as percentile ranks, never probabilities. Inference takes close to the 9-hour limit on
two T4s, and only after speed work (shared DINO prefixes, persistent GPU replicas, overlapped decoding).

## How it got to 0.941

| Step | Public score |
|---|---|
| 4-arm Raptor with slice-order reversal as a free extra view | 0.937 |
| Lateral meniscus routed 100% to Raptor | 0.940 |
| Gated CoAtNet added, per-finding routing for five findings | 0.941 |

Each step changed one thing and kept it if the public score rose by 0.001. The public leaderboard scores only
part of the test set, so a stack tuned this way carries the same selection optimism Phase 2 measured on the
58 gold studies. The private leaderboard will reshuffle the 568-way tie.

The pipeline also fails silently: without the DINOv2 model source attached, it drops 20 members, still finishes,
and scores 0.937.

## Differences from our checkpoint worth noting

- **Laterality normalisation.** Right knees are mirrored to match left ones (sagittal slice order reversed,
  coronal and axial flipped). Phase 3 found our model recognises medial from lateral from image content; these
  pipelines remove the need to.
- **Medical-image pretraining** (RadImageNet) and **self-supervised ViTs** (DINOv2/v3) alongside CoAtNet.
- **Different series handling**: six slots including a T1 slot, a 130 mm crop, and 1-99 percentile windowing.
- **Test-time views for free**: reversed slice order, horizontal flips.
- **Scale**: about 40 checkpoints against our one. The gain from a single CoAtNet (0.927) to the stack (0.941) is
  +0.014, most of it from combining families that fail differently.
