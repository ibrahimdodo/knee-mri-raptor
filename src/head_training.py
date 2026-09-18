"""Retrain the attention head on frozen backbone features.

The backbone is 73 M parameters and its features are computed once on Kaggle; the head is 280 k parameters
and trains on a laptop in minutes. That makes the head's design choices cheap to test:

- the positive weighting in the loss, which Phase 5 showed inflates every probability;
- soft targets (0.05 to 0.95) against hard ones;
- masking "unsure" targets, the 0.35-0.45 band the labelling model used when a report gave it nothing to go
  on, which is where the synovitis labels collapsed into effusion labels.

The head has the same shape as the checkpoint's, so a trained head can be dropped into the released
checkpoint and scored by exactly the same pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


class AttentionHead(nn.Module):
    """Per-finding attention pooling over window features: the checkpoint's head, on its own."""

    def __init__(self, f_dim: int = 1024, n: int = 12, drop: float = 0.2):
        super().__init__()
        self.norm = nn.LayerNorm(f_dim)
        self.att = nn.Sequential(nn.Linear(f_dim, 256), nn.Tanh(), nn.Dropout(drop), nn.Linear(256, n))
        self.clsW = nn.Parameter(torch.zeros(n, f_dim))
        self.clsb = nn.Parameter(torch.zeros(n))
        nn.init.trunc_normal_(self.clsW, std=0.02)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        h = self.norm(feats)
        a = torch.softmax(self.att(h), dim=1)
        pooled = torch.einsum("bkn,bkf->bnf", a, h)
        return (pooled * self.clsW).sum(-1) + self.clsb

    @classmethod
    def from_checkpoint(cls, model) -> "AttentionHead":
        """A copy of a loaded RaptorClassifier's head, for warm starts and baselines."""
        head = cls(f_dim=model.clsW.shape[1], n=model.clsW.shape[0])
        head.norm.load_state_dict(model.norm.state_dict())
        head.att.load_state_dict(model.att.state_dict())
        with torch.no_grad():
            head.clsW.copy_(model.clsW)
            head.clsb.copy_(model.clsb)
        return head


@dataclass
class Recipe:
    """One set of label and loss choices to compare."""
    name: str
    hard_targets: bool = False       # round soft labels to 0/1 at 0.5
    mask_unsure: bool = False        # drop targets inside `unsure` from the loss entirely
    unsure: tuple = (0.3, 0.5)       # the band the labelling model used for "no idea"
    pos_weight: bool = True          # the training script's clip((1 - prev) / prev, 1, 10)
    warm_start: bool = False         # begin from the checkpoint's head instead of a fresh one
    epochs: int = 12
    lr: float = 1e-3
    weight_decay: float = 0.02
    batch_size: int = 64
    drop: float = 0.2
    seed: int = 0


def targets_and_mask(soft: np.ndarray, recipe: Recipe):
    """(targets, mask) as float32 [N, 12]; mask 0 means the finding is ignored for that study."""
    y = soft.astype(np.float32).copy()
    mask = np.ones_like(y)
    if recipe.mask_unsure:
        lo, hi = recipe.unsure
        mask[(y >= lo) & (y <= hi)] = 0.0
    if recipe.hard_targets:
        y = (y >= 0.5).astype(np.float32)
    return y, mask


def pos_weights(y: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """The training script's recipe: clip((1 - prevalence) / prevalence, 1, 10) over the training studies."""
    prev = np.clip((y * mask).sum(0) / np.maximum(mask.sum(0), 1), 0.03, 0.7)
    return np.clip((1 - prev) / prev, 1, 10).astype(np.float32)


def masked_bce(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, pw: torch.Tensor | None):
    """BCE over the unmasked entries only, with an optional per-finding weight on the positive term."""
    loss = nn.functional.binary_cross_entropy_with_logits(logits, y, pos_weight=pw, reduction="none")
    return (loss * mask).sum() / mask.sum().clamp(min=1)


def train_head(feats: np.ndarray, soft: np.ndarray, train_idx, val_idx, recipe: Recipe,
               checkpoint_model=None, device: str = "cpu", log=print):
    """Train one head and return (head, history). The epoch with the best validation loss is kept.

    Validation uses held-out training studies, never the 58 labelled ones: choosing on those is the
    selection optimism Phase 2 measured.
    """
    torch.manual_seed(recipe.seed)
    np.random.seed(recipe.seed)
    y_all, mask_all = targets_and_mask(soft, recipe)
    pw = torch.from_numpy(pos_weights(y_all[train_idx], mask_all[train_idx])).to(device) if recipe.pos_weight else None

    head = (AttentionHead.from_checkpoint(checkpoint_model) if recipe.warm_start
            else AttentionHead(f_dim=feats.shape[-1], n=soft.shape[1], drop=recipe.drop)).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=recipe.lr, weight_decay=recipe.weight_decay)
    steps = max(1, int(np.ceil(len(train_idx) / recipe.batch_size))) * recipe.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=recipe.lr, total_steps=steps, pct_start=0.2)

    def batches(idx, shuffle):
        order = np.random.permutation(idx) if shuffle else np.asarray(idx)
        for i in range(0, len(order), recipe.batch_size):
            sel = order[i:i + recipe.batch_size]
            yield (torch.from_numpy(feats[sel].astype(np.float32)).to(device),
                   torch.from_numpy(y_all[sel]).to(device),
                   torch.from_numpy(mask_all[sel]).to(device))

    best, best_state, history = np.inf, None, []
    for ep in range(recipe.epochs):
        head.train()
        total = 0.0
        for xb, yb, mb in batches(train_idx, True):
            opt.zero_grad()
            loss = masked_bce(head(xb), yb, mb, pw)
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 3.0)
            opt.step()
            sched.step()
            total += loss.item() * len(xb)
        head.eval()
        with torch.no_grad():
            # validation loss is unweighted, so recipes with and without pos_weight stay comparable
            val = sum(masked_bce(head(xb), yb, mb, None).item() * len(xb) for xb, yb, mb in batches(val_idx, False))
        train_loss, val_loss = total / len(train_idx), val / len(val_idx)
        history.append({"epoch": ep, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best:
            best, best_state = val_loss, {k: v.detach().clone() for k, v in head.state_dict().items()}
        log(f"  {recipe.name} epoch {ep}: train {train_loss:.4f} | val {val_loss:.4f}")
    head.load_state_dict(best_state)
    return head.eval(), history


@torch.no_grad()
def predict(head, feats: np.ndarray, device: str = "cpu", batch: int = 64) -> np.ndarray:
    out = [head(torch.from_numpy(feats[i:i + batch].astype(np.float32)).to(device)).cpu().numpy()
           for i in range(0, len(feats), batch)]
    return np.concatenate(out).astype(np.float64)
