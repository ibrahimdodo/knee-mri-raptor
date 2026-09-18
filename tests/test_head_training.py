import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import head_training as ht  # noqa: E402
import raptor_core as rc  # noqa: E402


def test_head_matches_the_checkpoint_head():
    torch.manual_seed(0)
    model = rc.RaptorClassifier(torch.nn.Identity(), F_dim=32).eval()
    head = ht.AttentionHead.from_checkpoint(model).eval()
    feats = torch.randn(3, 7, 32)
    np.testing.assert_allclose(head(feats).detach().numpy(), model.head(feats).detach().numpy(), atol=1e-6)


def test_targets_and_mask():
    soft = np.array([[0.05, 0.35, 0.8, 0.45]])
    y, m = ht.targets_and_mask(soft, ht.Recipe("plain"))
    np.testing.assert_allclose(y, soft); np.testing.assert_allclose(m, 1)
    y, m = ht.targets_and_mask(soft, ht.Recipe("masked", mask_unsure=True))
    np.testing.assert_allclose(m, [[1, 0, 1, 0]])
    y, _ = ht.targets_and_mask(soft, ht.Recipe("hard", hard_targets=True))
    np.testing.assert_allclose(y, [[0, 0, 1, 0]])


def test_pos_weights_follow_prevalence():
    y = np.zeros((100, 2), np.float32); y[:10, 0] = 1; y[:50, 1] = 1
    pw = ht.pos_weights(y, np.ones_like(y))
    assert pw[0] == pytest.approx(9.0, rel=0.01)      # 10% prevalence
    assert pw[1] == pytest.approx(1.0, rel=0.01)      # 50% prevalence, clipped at 1


def test_masked_bce_ignores_masked_entries():
    logits = torch.tensor([[0.0, 50.0]])
    y = torch.tensor([[1.0, 0.0]])
    both = ht.masked_bce(logits, y, torch.ones(1, 2), None)
    first_only = ht.masked_bce(logits, y, torch.tensor([[1.0, 0.0]]), None)
    assert float(first_only) == pytest.approx(0.6931, abs=1e-3)   # the wildly wrong second entry is ignored
    assert float(both) > 10


def test_training_learns_a_separable_signal():
    rng = np.random.default_rng(0)
    n, k, f = 900, 5, 16
    labels = (rng.random((n, 2)) < 0.4).astype(np.float32)
    feats = rng.normal(size=(n, k, f)).astype(np.float32)
    feats[:, 0, 0] += 4 * labels[:, 0]                 # finding 0 is written into one window
    feats[:, 1, 1] += 4 * labels[:, 1]
    idx = np.arange(n)
    head, hist = ht.train_head(feats, labels, idx[:700], idx[700:],
                               ht.Recipe("smoke", epochs=20, lr=3e-3, batch_size=32), log=lambda *a: None)
    p = ht.predict(head, feats[700:])
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(labels[700:, 0], p[:, 0]) > 0.9
    assert hist[-1]["val_loss"] < hist[0]["val_loss"]
