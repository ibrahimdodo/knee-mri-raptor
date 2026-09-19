"""Checks on the preprocessing that the checkpoint depends on, using synthetic DICOM series."""
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import raptor_core as rc  # noqa: E402


def write_series(dirpath: Path, n: int, plane_normal=(0, 0, 1), spacing=0.5, size=400, photometric="MONOCHROME2"):
    """n slices whose brightness rises with physical slice index, written with shuffled file names.

    Each slice also carries a horizontal ramp: a perfectly flat slice sitting at the 2nd percentile
    would window to all zeros and be masked as empty, which real anatomy never does."""
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    dirpath.mkdir(parents=True)
    row = (1, 0, 0) if plane_normal == (0, 0, 1) else (0, 1, 0)
    col = tuple(np.cross(plane_normal, row).astype(int))
    order = list(range(n))
    random.Random(0).shuffle(order)
    for fname_i, s in enumerate(order):
        meta = FileMetaDataset()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.4"
        meta.MediaStorageSOPInstanceUID = generate_uid()
        ds = FileDataset(str(dirpath / f"{fname_i}.dcm"), {}, file_meta=meta, preamble=b"\0" * 128)
        ds.ImageOrientationPatient = list(row) + list(col)
        ds.ImagePositionPatient = list(np.array(plane_normal) * s * 3.0)
        ds.InstanceNumber = fname_i          # deliberately wrong order
        ds.PixelSpacing = [spacing, spacing]
        ds.Rows = ds.Columns = size
        ds.SamplesPerPixel, ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 1, 16, 16, 15, 0
        ds.PhotometricInterpretation = photometric
        ramp = np.tile(np.arange(size, dtype=np.uint16) // 4, (size, 1))
        ds.PixelData = (ramp + 10 * (s + 1)).astype(np.uint16).tobytes()
        ds.save_as(str(dirpath / f"{fname_i}.dcm"), enforce_file_format=True)


def test_order_series_uses_physical_position_not_filename(tmp_path):
    write_series(tmp_path / "s", 20)
    files, med_ps = rc.order_series(str(tmp_path / "s"))
    values = [rc.read_pixels(f)[0, 0] for f, _ in files]
    assert values == sorted(values)
    assert med_ps == 0.5


def test_crop_is_in_millimetres():
    geom = rc.Geometry(img=64)
    a = np.zeros((600, 600), np.float32)
    # 140 mm at 0.25 mm/px is 560 px: a 20 px border is cut away, the rest is resized to 64.
    a[:20, :] = 1.0
    out = rc.crop_mm_and_resize(a, 0.25, geom)
    assert out.shape == (64, 64) and out.max() == 0.0


def test_build_study_slots_span_and_empty_slot(tmp_path):
    geom = rc.Geometry(img=32, slots=(("Sagittal", 1, 5), ("Coronal", -1, 4), ("Axial", -1, 3)))
    write_series(tmp_path / "study" / "sagT2", 50, plane_normal=(1, 0, 0))
    write_series(tmp_path / "study" / "corPD", 10, plane_normal=(0, 1, 0))
    rows = [{"SeriesInstanceUID": "corPD", "Anatomical_Plane": "Coronal", "Fluid_Sensitive": 0},
            {"SeriesInstanceUID": "sagT2", "Anatomical_Plane": "Sagittal", "Fluid_Sensitive": 1}]
    vol, mask, info = rc.build_study(str(tmp_path / "study"), rows, geom)
    assert vol.shape == (12, 32, 32)
    assert mask.tolist() == [1] * 9 + [0] * 3                  # no axial series -> zeros, masked
    # 50 slices, span 2%..98%: lo = int(1.0) = 1, hi = int(49.0) - 1 = 48
    assert info[0]["picks"] == np.linspace(1, 48, 5).round().astype(int).tolist()
    assert info[2]["series"] is None
    # slices come out in increasing physical order within each slot
    assert all(np.diff(vol[:5, 16, 16].astype(int)) >= 0)


def test_eval_centers_uses_every_position_at_k62():
    mask = np.ones(64, np.uint8)
    assert rc.eval_centers(mask, 62) == list(range(1, 63))
    assert set(rc.eval_centers(mask, 42)) <= set(range(1, 63))


def test_windows_are_neighbouring_slices_as_channels():
    vol = np.stack([np.full((8, 8), i, np.uint8) for i in range(10)])
    w = rc.make_windows(vol, [4], res=8, norm="none")
    assert w.shape == (1, 3, 8, 8)
    assert torch.allclose(w[0, :, 0, 0] * 255, torch.tensor([3.0, 4.0, 5.0]))


@pytest.mark.parametrize("norm", ["imagenet", "none"])
def test_norm_modes(norm):
    vol = np.full((3, 4, 4), 255, np.uint8)
    w = rc.make_windows(vol, [1], res=4, norm=norm)
    expected = (1 - rc.IMAGENET_MEAN) / rc.IMAGENET_STD if norm == "imagenet" else torch.ones(3, 1, 1)
    assert torch.allclose(w[0, :, 0, 0], expected.flatten())


def test_build_study_cache_gives_identical_stacks(tmp_path):
    write_series(tmp_path / "study" / "sagT2", 40, plane_normal=(1, 0, 0))
    write_series(tmp_path / "study" / "corPD", 20, plane_normal=(0, 1, 0))
    rows = [{"SeriesInstanceUID": "sagT2", "Anatomical_Plane": "Sagittal", "Fluid_Sensitive": 1},
            {"SeriesInstanceUID": "corPD", "Anatomical_Plane": "Coronal", "Fluid_Sensitive": 0}]
    cache = {}
    for geom in (rc.Geometry(img=32), rc.Geometry(img=24, span_lo=0.06, span_hi=0.94)):
        plain = rc.build_study(str(tmp_path / "study"), rows, geom)
        cached = rc.build_study(str(tmp_path / "study"), rows, geom, cache=cache)
        np.testing.assert_array_equal(plain[0], cached[0])
        np.testing.assert_array_equal(plain[1], cached[1])
    assert any(k[0] == "px" for k in cache) and any(k[0] == "order" for k in cache)
