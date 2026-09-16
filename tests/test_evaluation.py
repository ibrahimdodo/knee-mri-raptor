import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import evaluation as ev  # noqa: E402


@pytest.fixture
def data():
    rng = np.random.default_rng(1)
    Y = (rng.random((58, 3)) < [0.15, 0.4, 0.6]).astype(int)
    P = np.clip(Y * 0.3 + rng.random((58, 3)), 0, 1)
    P[:5, 0] = 0.5                                 # some ties
    return Y, P


def test_auc_matches_sklearn_with_ties(data):
    Y, P = data
    for f in range(3):
        assert ev.auc(Y[:, f], P[:, f]) == pytest.approx(roc_auc_score(Y[:, f], P[:, f]))


def test_weighted_bootstrap_equals_explicit_resampling(data):
    Y, P = data
    W = ev.bootstrap_weights(58, 20, seed=3)
    fast = ev.bootstrap_aucs(Y, P, W)
    for b in range(20):
        idx = np.repeat(np.arange(58), W[b].astype(int))
        for f in range(3):
            y, p = Y[idx, f], P[idx, f]
            expected = roc_auc_score(y, p) if 0 < y.sum() < len(y) else np.nan
            np.testing.assert_allclose(fast[b, f], expected, equal_nan=True)


def test_delong_matches_bootstrap_se_roughly(data):
    Y, P = data
    W = ev.bootstrap_weights(58, 4000, seed=0)
    boot = ev.bootstrap_aucs(Y, P, W)
    for f in range(1, 3):
        _, se, _, _ = ev.delong_ci(Y[:, f], P[:, f])
        assert se == pytest.approx(np.nanstd(boot[:, f]), rel=0.2)


def test_delong_paired_identical_models_have_zero_difference(data):
    Y, P = data
    d, se, p = ev.delong_paired(Y[:, 1], P[:, 1], P[:, 1])
    assert d == 0 and p == pytest.approx(1.0)


def test_selection_optimism_is_positive_for_equal_noisy_candidates():
    rng = np.random.default_rng(0)
    B, V = 2000, 8
    shared = rng.normal(0, 0.02, (B, 1))
    boot = 0.9 + shared + rng.normal(0, 0.01, (B, V))
    orig = np.full(V, 0.9)
    opt, _, wins = ev.selection_optimism_bootstrap(orig, boot)
    # only the candidate-specific 0.01 noise creates optimism: 0.01 * E[max of 8] ~ 0.0142
    assert opt == pytest.approx(0.01 * ev.expected_max_std_normal(8), rel=0.1)
    assert wins.sum() == pytest.approx(1.0)


def test_expected_max_known_values():
    assert ev.expected_max_std_normal(1) == 0.0
    assert ev.expected_max_std_normal(2) == pytest.approx(1 / np.sqrt(np.pi), abs=0.01)
    assert ev.expected_max_std_normal(16) == pytest.approx(1.766, abs=0.01)
