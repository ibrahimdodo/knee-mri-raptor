import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import ablation as ab  # noqa: E402
import raptor_core as rc  # noqa: E402


class TinyBackbone(torch.nn.Module):
    """Feature = per-channel means of the window, so features are cheap and fully predictable."""
    num_features = 3

    def forward(self, x):
        return x.mean(dim=(2, 3))


@pytest.fixture
def setup():
    rng = np.random.default_rng(0)
    vol = rng.integers(1, 255, (64, 8, 8), dtype=np.uint8)
    mask = np.ones(64, np.uint8)
    model = rc.RaptorClassifier(TinyBackbone(), F_dim=3).eval()
    base = np.stack([model.backbone(rc.make_windows(vol, [c], res=8)).numpy()[0] for c in range(1, 63)])[None]
    return vol, mask, model, base


def test_unchanged_study_reuses_saved_features(setup):
    vol, mask, model, base = setup
    bank = ab.FeatureBank(model, base)
    f = bank.features(0, vol, mask, np.zeros(64, bool))
    np.testing.assert_allclose(f, base[0], atol=1e-6)


def test_blanking_matches_full_recompute(setup):
    vol, mask, model, base = setup
    bank = ab.FeatureBank(model, base)
    for ranges in ([(44, 52)], [(0, 18)], [(52, 64)]):          # middle slot, first slot, last slot
        v2, m2, changed = ab.blank_slots(vol, mask, ranges)
        fast = bank.features(0, v2, m2, changed)
        centres = rc.eval_centers(m2, 62)
        full = model.backbone(rc.make_windows(v2, centres, res=8)).numpy()
        np.testing.assert_allclose(fast, full, atol=1e-5)


def test_edge_blank_moves_window_centres(setup):
    vol, mask, _, _ = setup
    _, m2, _ = ab.blank_slots(vol, mask, [(0, 18)])
    assert min(rc.eval_centers(m2, 62)) == 19


def test_head_is_permutation_invariant(setup):
    _, _, model, base = setup
    perm = np.random.default_rng(1).permutation(62)
    np.testing.assert_allclose(ab.head_logits(model, base), ab.head_logits(model, base[:, perm]), atol=1e-5)


def test_perturbations_keep_black_and_shape():
    vol = np.zeros((2, 384, 384), np.uint8); vol[:, 100:284, 100:284] = 128
    for out in (ab.zoom(vol, 1.5), ab.gamma(vol, 1.5), ab.gamma(vol, 0.67)):
        assert out.shape == vol.shape and out.dtype == np.uint8 and out[:, 0, 0].max() == 0
    assert ab.gamma(vol, 1.5)[0, 192, 192] < 128 < ab.gamma(vol, 0.67)[0, 192, 192]
    assert (ab.zoom(vol, 1.5) > 0).mean() > (vol > 0).mean()     # the bright centre grows
