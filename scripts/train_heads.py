"""Phase 6: retrain the attention head under several label/loss recipes, 3 seeds each.

Trains on the 4,349 soft-labelled studies (85% train / 15% validation, same split for every run), keeps
the epoch with the best unweighted validation loss, and saves the 58 gold studies' logits for each run.
The gold studies are never used for training or selection.

Writes outputs/local/heads/<recipe>_seed<k>.npz (gold logits, history) and outputs/local/heads/runs.csv.
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import head_training as ht
import raptor_core as rc

# FEATURES=coatnet (Phase 6) retrains the head on the checkpoint's own frozen features and can warm-start from
# its head. FEATURES=dino (Phase 8) trains the same head on frozen DINOv2 features, which have no head to inherit,
# so only the recipe that Phase 6 settled on is run.
FEATURES = os.environ.get("RAPTOR_FEATURES", "coatnet")
RECIPES = {
    "coatnet": [
        ht.Recipe("R1_author", pos_weight=True),
        ht.Recipe("R2_no_posweight", pos_weight=False),
        ht.Recipe("R3_mask_unsure", pos_weight=False, mask_unsure=True),
        ht.Recipe("R4_mask_unsure_posweight", pos_weight=True, mask_unsure=True),
        ht.Recipe("R5_hard", pos_weight=False, hard_targets=True),
        ht.Recipe("R6_warm_mask_unsure", pos_weight=False, mask_unsure=True, warm_start=True, lr=3e-4),
    ],
    "dino": [ht.Recipe("D_mask_unsure", pos_weight=False, mask_unsure=True, epochs=20)],
}[FEATURES]
SEEDS = [0, 1, 2]
SHARDS = {"coatnet": ("outputs/kaggle/train_feats", "feats_shard{}.npy", "ids_shard{}.npy"),
          "dino": ("outputs/kaggle/dino_feats", "dino_feats_shard{}.npy", "dino_ids_shard{}.npy")}[FEATURES]


def load_features():
    folder, feat_name, id_name = SHARDS
    feats, ids = [], []
    for k in (0, 1):
        feats.append(np.load(ROOT / folder / feat_name.format(k)))
        ids.append(np.load(ROOT / folder / id_name.format(k)))
    return np.concatenate(feats), np.concatenate(ids).astype(str)


def main():
    out = ROOT / ("outputs/local/heads" if FEATURES == "coatnet" else f"outputs/local/heads_{FEATURES}")
    out.mkdir(parents=True, exist_ok=True)
    feats, ids = load_features()
    row = {s: i for i, s in enumerate(ids)}

    soft = pd.read_csv(ROOT / "data/soft_labels/labels_llm_soft.csv")
    soft["StudyInstanceUID"] = soft["StudyInstanceUID"].astype(str)
    gold = pd.read_csv(ROOT / "outputs/kaggle/gold_run/gold_labels.csv")
    gold["StudyInstanceUID"] = gold["StudyInstanceUID"].astype(str)
    assert not set(soft.StudyInstanceUID) & set(gold.StudyInstanceUID)

    train_rows = np.array([row[s] for s in soft.StudyInstanceUID])
    gold_rows = np.array([row[s] for s in gold.StudyInstanceUID])
    X, S = feats[train_rows], soft[rc.LAB].values
    Xg, Yg = feats[gold_rows], gold[rc.LAB].values.astype(int)

    split = np.random.default_rng(2646).permutation(len(X))
    n_val = int(0.15 * len(X))
    val_idx, train_idx = split[:n_val], split[n_val:]
    print(f"{len(train_idx)} train | {len(val_idx)} validation | {len(gold_rows)} gold (held out)", flush=True)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    checkpoint, _ = rc.load_checkpoint(str(ROOT / "models/raptor_ft_coatnet_v10_full.pt"), "cpu")
    np.save(out / "gold_labels.npy", Yg)
    if FEATURES == "coatnet":
        np.save(out / "checkpoint_gold_logits.npy", ht.predict(ht.AttentionHead.from_checkpoint(checkpoint).eval(), Xg))
    print(f"features: {FEATURES}, {X.shape[-1]}-d", flush=True)

    runs = []
    for recipe in RECIPES:
        for seed in SEEDS:
            recipe.seed = seed
            t0 = time.time()
            head, hist = ht.train_head(X, S, train_idx, val_idx, recipe,
                                       checkpoint_model=checkpoint if FEATURES == "coatnet" else None,
                                       device=device, log=lambda *a: None)
            head = head.cpu()
            gold_logits = ht.predict(head, Xg)
            best = min(hist, key=lambda h: h["val_loss"])
            np.savez(out / f"{recipe.name}_seed{seed}.npz", gold_logits=gold_logits, history=pd.DataFrame(hist).values)
            torch.save(head.state_dict(), out / f"{recipe.name}_seed{seed}.pt")
            macro = rc.macro_auc(Yg, gold_logits)[0]
            runs.append({"recipe": recipe.name, "seed": seed, "best_epoch": best["epoch"],
                         "val_loss": best["val_loss"], "gold_macro_auc": macro, "seconds": time.time() - t0})
            print(f"{recipe.name} seed {seed}: best epoch {best['epoch']} | val loss {best['val_loss']:.4f} | "
                  f"gold macro-AUC {macro:.4f} | {time.time() - t0:.0f}s", flush=True)
    pd.DataFrame(runs).to_csv(out / "runs.csv", index=False)


if __name__ == "__main__":
    main()
