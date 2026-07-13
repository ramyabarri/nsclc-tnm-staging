"""Tests for the Vision Agent T-factor logic (size -> IASLC T category).

Uses synthetic masks with isotropic 1mm spacing so a ball of radius r has a
greatest dimension of ~2r mm.
"""

import numpy as np
import pytest

from agents.vision import VisionAgent, TFactorEvidence


def _agent():
    return VisionAgent({"spacing_zyx": [1.0, 1.0, 1.0]})


def _ball(radius: int) -> np.ndarray:
    r = int(radius)
    n = 2 * r + 6
    zz, yy, xx = np.indices((n, n, n))
    c = n // 2
    return (((zz - c) ** 2 + (yy - c) ** 2 + (xx - c) ** 2) <= r ** 2).astype(np.uint8)


def _classify(radius):
    a = _agent()
    mask = _ball(radius)
    feats = a.extract_radiomics(np.zeros_like(mask), mask)
    return a.classify_t_factor(feats, mask)


# --- size -> T mapping -----------------------------------------------------

@pytest.mark.parametrize("radius,expected", [
    (4,  "T1a"),   # ~8 mm
    (7,  "T1b"),   # ~14 mm
    (13, "T1c"),   # ~26 mm
    (17, "T2a"),   # ~34 mm
    (22, "T2b"),   # ~44 mm
    (30, "T3"),    # ~60 mm
    (40, "T4"),    # ~80 mm
])
def test_size_to_t_category(radius, expected):
    t, conf = _classify(radius)
    assert t == expected
    assert 0.5 <= conf <= 0.9


def test_size_thresholds_direct():
    a = _agent()
    assert a._size_to_t(9) == "T1a"
    assert a._size_to_t(10) == "T1a"
    assert a._size_to_t(10.5) == "T1b"
    assert a._size_to_t(30) == "T1c"
    assert a._size_to_t(30.5) == "T2a"
    assert a._size_to_t(70) == "T3"
    assert a._size_to_t(71) == "T4"


# --- diameter measurement --------------------------------------------------

def test_max_diameter_two_points():
    a = _agent()
    mask = np.zeros((1, 1, 31), dtype=np.uint8)
    mask[0, 0, 0] = 1
    mask[0, 0, 30] = 1
    assert a._max_diameter_mm(mask) == pytest.approx(30.0, abs=1e-6)


def test_anisotropic_spacing_scales_diameter():
    a = VisionAgent({"spacing_zyx": [3.0, 1.0, 1.0]})
    mask = np.zeros((11, 1, 1), dtype=np.uint8)
    mask[0, 0, 0] = 1
    mask[10, 0, 0] = 1
    assert a._max_diameter_mm(mask) == pytest.approx(30.0)  # 10 voxels * 3mm


# --- degenerate cases ------------------------------------------------------

def test_empty_mask_is_tx():
    a = _agent()
    mask = np.zeros((10, 10, 10), dtype=np.uint8)
    t, conf = a.classify_t_factor({}, mask)
    assert t == "TX"
    assert conf == 0.0


def test_tiny_mask_below_threshold_is_tx():
    a = _agent()
    mask = np.zeros((10, 10, 10), dtype=np.uint8)
    mask[5, 5, 5:8] = 1   # 3 voxels < min_tumour_voxels
    t, _ = a.classify_t_factor({}, mask)
    assert t == "TX"


# --- invasion upstaging ----------------------------------------------------

def test_invasion_mediastinal_forces_t4():
    a = _agent()
    mask = _ball(7)  # ~T1b by size
    feats = a.extract_radiomics(np.zeros_like(mask), mask)
    t, _ = a.classify_t_factor(feats, mask, invasion={"mediastinal": True})
    assert t == "T4"


def test_invasion_visceral_pleura_upstages_to_at_least_t2a():
    a = _agent()
    mask = _ball(7)  # ~T1b
    feats = a.extract_radiomics(np.zeros_like(mask), mask)
    t, _ = a.classify_t_factor(feats, mask, invasion={"visceral_pleura": True})
    assert t == "T2a"


def test_invasion_does_not_downstage():
    a = _agent()
    mask = _ball(30)  # ~T3 by size
    feats = a.extract_radiomics(np.zeros_like(mask), mask)
    t, _ = a.classify_t_factor(feats, mask, invasion={"visceral_pleura": True})
    assert t == "T3"   # visceral pleura floor is T2a, must not lower a T3


# --- radiomics & full run --------------------------------------------------

def test_extract_radiomics_fallback_features():
    a = _agent()
    mask = _ball(10)
    feats = a.extract_radiomics(np.zeros_like(mask), mask)
    assert feats["voxel_count"] > 0
    assert feats["volume_mm3"] > 0
    assert feats["max_diameter_mm"] > 0


def test_largest_component_ignores_satellite():
    """A small disconnected deposit must not enlarge the measured primary tumour."""
    a = _agent()
    mask = _ball(20)                       # primary ~40 mm
    # add a far-away 1-voxel satellite that would inflate a whole-mask diameter
    mask[0, 0, 0] = 1
    primary, n_comp, frac = a._primary_component(mask.astype(bool))
    assert n_comp == 2
    assert frac > 0.99
    feats = a.extract_radiomics(np.zeros_like(mask), mask)
    assert feats["n_components"] == 2
    # measured diameter reflects the primary (~40 mm), not the corner-to-ball span
    assert feats["max_diameter_mm"] < 50


def test_axial_diameter_below_3d_for_elongated_tumour():
    """Axial greatest dimension must not exceed the 3D diameter; for a z-elongated
    mask it should be strictly smaller (the fix that curbs T over-staging)."""
    a = VisionAgent({"spacing_zyx": [3.0, 1.0, 1.0]})
    # a column tall in z (30 slices) but small in-plane (~10 mm)
    mask = np.zeros((30, 20, 20), dtype=np.uint8)
    mask[:, 8:12, 8:12] = 1
    axial = a._max_axial_diameter_mm(mask.astype(bool))
    d3d = a._max_diameter_mm(mask.astype(bool))
    assert axial < d3d
    assert axial < 15   # in-plane extent only


def test_diameter3d_method_recovers_old_behaviour():
    a = VisionAgent({"spacing_zyx": [1.0, 1.0, 1.0], "diameter_method": "diameter3d"})
    mask = _ball(20)
    feats = a.extract_radiomics(np.zeros_like(mask), mask)
    assert feats["max_diameter_mm"] == pytest.approx(feats["max_diameter_3d_mm"])


def test_run_end_to_end(tmp_path):
    a = _agent()
    mask = _ball(22)  # ~T2b
    ct = np.zeros_like(mask, dtype=np.float32)
    ct_path = tmp_path / "case_ct.npy"
    mask_path = tmp_path / "case_mask.npy"
    np.save(ct_path, ct)
    np.save(mask_path, mask)

    ev = a.run(str(ct_path), mask_path=str(mask_path))
    assert isinstance(ev, TFactorEvidence)
    assert ev.t_category == "T2b"
    assert ev.tumour_size_mm > 40
    assert ev.tumour_volume_mm3 > 0
    assert ev.segmentation_path == str(mask_path)
