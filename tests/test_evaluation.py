"""Tests for the evaluation layer: guideline validation + staging harness helpers."""

import pytest

from scripts.evaluation.validate_guideline import (
    oracle_stage, validate, T_TOKENS, N_TOKENS, M_TOKENS,
)
from scripts.evaluation import stage_eval as se


# --- Guideline exhaustive validation ------------------------------------------

def test_guideline_matches_oracle_exhaustively():
    """The encoded IASLC table must match the independent oracle for every combo."""
    res = validate()
    assert res["total"] == len(T_TOKENS) * len(N_TOKENS) * len(M_TOKENS)
    assert res["failed"] == 0, f"mismatches: {res['mismatches']}"
    assert res["passed"] == res["total"]


@pytest.mark.parametrize("t,n,m,expected", [
    ("T1a", "N0", "M0", "IA1"),
    ("T2a", "N0", "M0", "IB"),
    ("T2b", "N0", "M0", "IIA"),
    ("T3", "N1", "M0", "IIIA"),
    ("T4", "N2", "M0", "IIIB"),
    ("T1a", "N3", "M0", "IIIB"),
    ("T3", "N3", "M0", "IIIC"),
    ("T2a", "any" if False else "N0", "M1a", "IVA"),
    ("T1a", "N0", "M1c", "IVB"),
    ("TX", "N0", "M0", "indeterminate"),
    ("T2a", "NX", "M0", "indeterminate"),
])
def test_oracle_known_points(t, n, m, expected):
    assert oracle_stage(t, n, m) == expected


# --- Staging harness label mapping --------------------------------------------

@pytest.mark.parametrize("cat,coarse", [
    ("T1a", 1), ("T1b", 1), ("T1c", 1), ("T1mi", 1), ("Tis", 1),
    ("T2a", 2), ("T2b", 2), ("T3", 3), ("T4", 4),
    ("TX", None), (None, None), ("T0", None),
])
def test_coarse_t_from_category(cat, coarse):
    assert se.coarse_t_from_category(cat) == coarse


@pytest.mark.parametrize("raw,tok", [(0, "N0"), (1, "N1"), (2, "N2"), (3, "N3"),
                                     (4, None), ("x", None), (None, None)])
def test_n_to_token(raw, tok):
    assert se.n_to_token(raw) == tok


@pytest.mark.parametrize("raw,tok", [(0, "M0"), (1, "M1b"), (3, None), (None, None)])
def test_m_to_token(raw, tok):
    assert se.m_to_token(raw) == tok


@pytest.mark.parametrize("gt,norm", [
    ("I", "I"), ("II", "II"), ("IIIa", "IIIa"), ("IIIb", "IIIb"),
    ("iiib", "IIIb"), ("IV", "IV"), (None, None), (1.0, None),
])
def test_normalize_gt_stage(gt, norm):
    assert se.normalize_gt_stage(gt) == norm


@pytest.mark.parametrize("stage,coarse", [
    ("IA1", "I"), ("IA3", "I"), ("IB", "I"), ("IIA", "II"), ("IIB", "II"),
    ("IIIA", "IIIa"), ("IIIB", "IIIb"), ("IIIC", "IIIb"),
    ("IVA", "IV"), ("IVB", "IV"), ("indeterminate", None), (None, None),
])
def test_agent_stage_to_coarse(stage, coarse):
    assert se.agent_stage_to_coarse(stage) == coarse


def test_score_block_perfect():
    out, cm = se.score_block("t", ["I", "II", "II"], ["I", "II", "II"])
    assert out["accuracy"] == 1.0
    assert out["n"] == 3 and out["n_skipped"] == 0
    assert cm is not None


def test_score_block_skips_none():
    out, _ = se.score_block("t", ["I", None, "II"], ["I", "II", None])
    assert out["n"] == 1 and out["n_skipped"] == 2


def test_end_to_end_stage_mapping_consistency():
    """Predicted-T + oracle N/M through the real guideline must land in the coarse
    scheme (a smoke test of run_guideline + agent_stage_to_coarse wiring)."""
    stage = se.agent_stage_to_coarse(se.run_guideline("T2a", "N0", "M0"))
    assert stage == "I"
    stage = se.agent_stage_to_coarse(se.run_guideline("T4", "N2", "M0"))
    assert stage == "IIIb"
