"""Tests for the multi-agent staging orchestrator.

LangGraph is optional; these exercise the sequential fallback (and whichever
backend is installed). Note text is synthetic — no MIMIC data.
"""

import numpy as np
import pytest

from orchestration import StagingOrchestrator
from agents.guideline_logic import StagingResult


def _ball(radius: int) -> np.ndarray:
    r = int(radius)
    n = 2 * r + 6
    zz, yy, xx = np.indices((n, n, n))
    c = n // 2
    return (((zz - c) ** 2 + (yy - c) ** 2 + (xx - c) ** 2) <= r ** 2).astype(np.uint8)


def _orch():
    # isotropic spacing so ball radius r -> ~2r mm greatest dimension
    return StagingOrchestrator({"vision": {"spacing_zyx": [1.0, 1.0, 1.0]}})


def _write_case(tmp_path, radius):
    mask = _ball(radius)
    ct_path = tmp_path / "ct.npy"
    mask_path = tmp_path / "mask.npy"
    np.save(ct_path, np.zeros_like(mask, dtype=np.float32))
    np.save(mask_path, mask)
    return str(ct_path), str(mask_path)


def test_end_to_end_stage(tmp_path):
    ct, mask = _write_case(tmp_path, radius=22)   # ~T2b
    notes = [{"note_id": "r1", "note_type": "radiology",
              "text": "Right hilar adenopathy. No metastatic disease."}]  # N1, M0
    res = _orch().run("P1", ct, notes=notes, mask_path=mask)

    assert isinstance(res, StagingResult)
    assert res.tnm_string == "T2bN1M0"
    assert res.overall_stage == "IIB"
    assert not res.metadata.get("errors")


def test_no_notes_defaults_to_n0_m0(tmp_path):
    ct, mask = _write_case(tmp_path, radius=22)   # ~T2b
    res = _orch().run("P2", ct, notes=[], mask_path=mask)
    assert res.tnm_string == "T2bN0M0"
    assert res.overall_stage == "IIA"


def test_graceful_degradation_on_bad_ct(tmp_path):
    # Vision fails (missing CT) but the pipeline still returns a result using N/M.
    notes = [{"note_id": "r1", "note_type": "radiology",
              "text": "Bulky mediastinal lymphadenopathy."}]  # N2
    res = _orch().run("P3", str(tmp_path / "does_not_exist.npy"),
                      notes=notes, mask_path=None)
    assert isinstance(res, StagingResult)
    assert res.metadata.get("errors")                 # vision error recorded
    assert any("vision" in e for e in res.metadata["errors"])
    assert res.rationale.final_t == "TX"              # T unavailable -> TX
    assert res.rationale.final_n == "N2"              # N still extracted


def test_metastasis_gives_stage_iv(tmp_path):
    ct, mask = _write_case(tmp_path, radius=7)        # small tumour ~T1b
    notes = [{"note_id": "r1", "note_type": "radiology",
              "text": "Hepatic and adrenal metastases."}]     # M1c
    res = _orch().run("P4", ct, notes=notes, mask_path=mask)
    assert res.overall_stage == "IVB"                 # M1c dominates

def test_repr_reports_backend():
    r = repr(_orch())
    assert "backend=" in r
