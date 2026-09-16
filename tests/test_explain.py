import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import explain as ex  # noqa: E402
import raptor_core as rc  # noqa: E402


def test_window_slots_and_straddling():
    own, straddle = ex.window_slots()
    assert len(own) == 62
    assert own[0] == 0 and own[-1] == 4
    # centre 17 = slices 16,17,18 crosses sagittal fluid -> other; centre 18 = 17,18,19 too
    assert straddle[ex.CENTRES == 17][0] and straddle[ex.CENTRES == 18][0]
    assert straddle.sum() == 8                     # two windows at each of 4 boundaries


def test_effective_windows_bounds():
    K = 10
    uniform = np.full((1, K, 2), 1 / K)
    peaked = np.zeros((1, K, 2)); peaked[0, 3] = 1
    assert ex.effective_windows(uniform) == pytest.approx(np.full((1, 2), K))
    assert ex.effective_windows(peaked) == pytest.approx(np.ones((1, 2)))


@pytest.mark.parametrize("texts,expected", [
    (("R",), "R"), ((None, "L"), "L"), ((float("nan"), "RT.Sag_T2W_TSE"), "R"),
    (("Rodilla izquierda",), "L"), (("KNEE",), None), (("pd_tse_fs_sag_320",), None),
])
def test_knee_side(texts, expected):
    assert ex.knee_side(*texts) == expected


def test_medial_direction():
    assert ex.medial_at_high_index(1.0, "R") is True      # index -> patient left -> medial of a right knee
    assert ex.medial_at_high_index(1.0, "L") is False
    assert ex.medial_at_high_index(-1.0, "L") is True
    assert ex.medial_at_high_index(0.1, "R") is None      # not a sagittal normal


def test_grad_cam_on_small_backbone():
    torch.manual_seed(0)
    bb = rc.build_backbone("coatnet_nano_rw_224")
    model = rc.RaptorClassifier(bb, F_dim=bb.num_features).eval()
    window = torch.randn(3, 224, 224)
    cam, logit = ex.grad_cam(model, window, finding=3)
    assert cam.shape == (224, 224) and cam.min() >= 0 and cam.max() <= 1 + 1e-6
    with torch.no_grad():
        _, _, wl = model.head_detailed(model.encode(window[None, None]))
    assert logit == pytest.approx(float(wl[0, 0, 3]), abs=1e-4)


def test_occlusion_on_small_backbone():
    torch.manual_seed(0)
    bb = rc.build_backbone("coatnet_nano_rw_224")
    model = rc.RaptorClassifier(bb, F_dim=bb.num_features).eval()
    window = torch.randn(3, 224, 224)
    drop, base = ex.occlusion(model, window, finding=0, patch=56, stride=56)
    assert drop.shape == (224, 224) and np.isfinite(drop).all()
    _, logit = ex.grad_cam(model, window, finding=0)
    assert base == pytest.approx(logit, abs=1e-4)
