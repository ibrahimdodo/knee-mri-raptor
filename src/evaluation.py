"""How much a score measured on a small labelled set can be trusted.

Three tools, all for ROC-AUC on binary labels:

- DeLong: an analytic standard error for one AUC, and the covariance between AUCs of several models
  scored on the same studies (so paired differences get the right, much smaller, variance).
- Study bootstrap: resample studies with replacement, recompute every AUC. Shared resamples across
  models make the comparison paired. Implemented with count weights so 10,000 resamples of 58 studies
  are a few matrix products rather than 10,000 calls to sklearn.
- Selection optimism: how far the best-looking candidate's score sits above what it would score on
  fresh data, either by bootstrap over an explicit candidate set, or from a noise model when the
  candidates (e.g. training epochs) are not available.
"""
from __future__ import annotations

import numpy as np


# ============================================================================
# Pairwise comparison matrix: the object every AUC here is built from
# ============================================================================
def psi(pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    """[m, n] matrix: 1 where the positive outranks the negative, 0.5 on ties, else 0.

    AUC is exactly the mean of this matrix: the probability that a random positive study scores
    above a random negative one.
    """
    return (pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :])


def auc(y: np.ndarray, p: np.ndarray) -> float:
    return float(psi(p[y == 1], p[y == 0]).mean())


# ============================================================================
# DeLong
# ============================================================================
def delong(y: np.ndarray, preds: np.ndarray):
    """AUCs of several models on the same studies, and their covariance matrix.

    preds: [k, N] scores from k models for N studies; y: [N] binary labels.
    Returns (aucs [k], cov [k, k]).

    Each positive study i gets a "placement" V10[i] = its mean psi against all negatives, and each
    negative j gets V01[j] = its mean psi against all positives. The AUC is the mean of either, and
    since positives and negatives are sampled independently,
        Var(AUC) = Var(V10) / m + Var(V01) / n,
    with the same formula giving covariances between models (DeLong, DeLong & Clarke-Pearson 1988).
    """
    preds = np.atleast_2d(preds)
    pos, neg = y == 1, y == 0
    m, n = pos.sum(), neg.sum()
    V10 = np.empty((len(preds), m))
    V01 = np.empty((len(preds), n))
    for k, p in enumerate(preds):
        P = psi(p[pos], p[neg])
        V10[k], V01[k] = P.mean(1), P.mean(0)
    aucs = V10.mean(1)
    cov = np.atleast_2d(np.cov(V10)) / m + np.atleast_2d(np.cov(V01)) / n
    return aucs, cov


def delong_ci(y: np.ndarray, p: np.ndarray, level: float = 0.95):
    """(auc, se, lo, hi) with a normal interval clipped to [0, 1]."""
    from scipy.stats import norm

    a, cov = delong(y, p[None])
    se = float(np.sqrt(cov[0, 0]))
    z = norm.ppf(0.5 + level / 2)
    return float(a[0]), se, max(0.0, a[0] - z * se), min(1.0, a[0] + z * se)


def delong_paired(y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray):
    """(auc_b - auc_a, se of the difference, two-sided p-value)."""
    from scipy.stats import norm

    a, cov = delong(y, np.stack([p_a, p_b]))
    d = a[1] - a[0]
    se = float(np.sqrt(max(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1], 1e-12)))
    return float(d), se, float(2 * norm.sf(abs(d) / se))


# ============================================================================
# Study bootstrap
# ============================================================================
def bootstrap_weights(n: int, B: int, seed: int = 0) -> np.ndarray:
    """[B, n] how many times each study appears in each resample (rows sum to n)."""
    rng = np.random.default_rng(seed)
    return rng.multinomial(n, np.full(n, 1.0 / n), size=B).astype(np.float64)


def bootstrap_aucs(Y: np.ndarray, P: np.ndarray, W: np.ndarray) -> np.ndarray:
    """AUC per resample and finding for one model.

    Y: [N, F] binary labels, P: [N, F] scores, W: [B, N] resample counts.
    Returns [B, F]; NaN where a resample lost every positive or every negative of a finding.

    A study drawn w times contributes w copies, so a positive/negative pair (i, j) counts w_i * w_j
    times: AUC_b = sum_ij w_i C_ij w_j / (sum_pos w * sum_neg w), where C is psi laid out on the
    full N x N grid (zero unless i is positive and j negative).
    """
    B, N = W.shape
    out = np.empty((B, Y.shape[1]))
    for f in range(Y.shape[1]):
        y = Y[:, f].astype(bool)
        C = np.zeros((N, N))
        C[np.ix_(y, ~y)] = psi(P[y, f], P[~y, f])
        num = np.einsum("bi,bi->b", W @ C, W)
        den = (W @ y) * (W @ ~y)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[:, f] = np.where(den > 0, num / den, np.nan)
    return out


def percentile_ci(samples: np.ndarray, level: float = 0.95, axis: int = 0):
    a = (1 - level) / 2
    return np.nanquantile(samples, a, axis=axis), np.nanquantile(samples, 1 - a, axis=axis)


# ============================================================================
# Selection optimism
# ============================================================================
def selection_optimism_bootstrap(macro_orig: np.ndarray, macro_boot: np.ndarray):
    """Optimism of picking the best of several candidates on the same labelled set.

    macro_orig: [V] score of each candidate on the real set; macro_boot: [B, V] on resamples.
    In each resample the resample plays "the set we selected on" and the real set plays "fresh data":
        optimism_b = macro_boot[b, best_b] - macro_orig[best_b].
    Returns (mean optimism, [B] optimism per resample, [V] how often each candidate won).
    """
    best = np.nanargmax(macro_boot, axis=1)
    opt = macro_boot[np.arange(len(best)), best] - macro_orig[best]
    wins = np.bincount(best, minlength=macro_boot.shape[1]) / len(best)
    return float(opt.mean()), opt, wins


def expected_max_std_normal(n: int, draws: int = 400_000, seed: int = 0) -> float:
    """E[max of n independent N(0, 1)]: 0 for n=1, ~1.03 for 3, ~1.77 for 16."""
    if n == 1:
        return 0.0
    rng = np.random.default_rng(seed)
    return float(rng.standard_normal((draws, n)).max(1).mean())


def epoch_selection_optimism(sd_diff: float, n: int) -> float:
    """Expected optimism of keeping the best of n equally good epochs by their score on one set.

    Model: score_e = true + shared + own_e, where `shared` is the luck every epoch gets from this
    particular set of studies and `own_e` is epoch-specific luck, independent across epochs. Only
    `own_e` can make one epoch look better than another, so only it biases the choice. Its SD is
    sd_diff / sqrt(2), where sd_diff is the SD of the score difference between two epochs across
    resamples, which is directly measurable for any two models scored on the same studies.
    """
    return sd_diff / np.sqrt(2) * expected_max_std_normal(n)
