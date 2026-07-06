"""Tests for the Guideline Logic Agent (IASLC TNM stage adjudication)."""

import pytest

from agents.vision import TFactorEvidence
from agents.clinical_context import NMFactorEvidence
from agents.guideline_logic import GuidelineLogicAgent, ConflictResolution


def _agent():
    a = GuidelineLogicAgent({})
    a.load_rules()
    return a


def _stage(t, n, m, tconf=1.0, nmconf=1.0):
    a = _agent()
    te = TFactorEvidence(t_category=t, confidence=tconf)
    ne = NMFactorEvidence(n_category=n, m_category=m, confidence=nmconf)
    return a.run(te, ne, patient_id="TEST")


# --- Stage grouping (IASLC 8th-edition table) -----------------------------

@pytest.mark.parametrize("t,n,m,expected", [
    ("Tis", "N0", "M0", "0"),
    ("T1mi", "N0", "M0", "IA1"),
    ("T1a", "N0", "M0", "IA1"),
    ("T1b", "N0", "M0", "IA2"),
    ("T1c", "N0", "M0", "IA3"),
    ("T2a", "N0", "M0", "IB"),
    ("T2b", "N0", "M0", "IIA"),
    ("T3",  "N0", "M0", "IIB"),
    ("T4",  "N0", "M0", "IIIA"),
    ("T2a", "N1", "M0", "IIB"),
    ("T3",  "N1", "M0", "IIIA"),
    ("T4",  "N1", "M0", "IIIA"),
    ("T1a", "N2", "M0", "IIIA"),
    ("T2b", "N2", "M0", "IIIA"),
    ("T3",  "N2", "M0", "IIIB"),
    ("T4",  "N2", "M0", "IIIB"),
    ("T1a", "N3", "M0", "IIIB"),
    ("T2b", "N3", "M0", "IIIB"),
    ("T3",  "N3", "M0", "IIIC"),
    ("T4",  "N3", "M0", "IIIC"),
])
def test_stage_groupings(t, n, m, expected):
    result = _stage(t, n, m)
    assert result.overall_stage == expected
    assert result.tnm_string == f"{t}{n}{m}"


# --- M1 dominates regardless of T and N -----------------------------------

@pytest.mark.parametrize("m,expected", [("M1a", "IVA"), ("M1b", "IVA"), ("M1c", "IVB")])
def test_m1_dominates(m, expected):
    assert _stage("T1a", "N0", m).overall_stage == expected
    assert _stage("T4", "N3", m).overall_stage == expected


def test_m1_dominates_even_with_missing_tn():
    # No T/N evidence, but distant mets present -> stage IV regardless.
    result = _stage(None, None, "M1c")
    assert result.overall_stage == "IVB"


# --- Missing / default handling -------------------------------------------

def test_missing_m_defaults_to_m0():
    result = _stage("T2a", "N0", None)
    assert result.overall_stage == "IB"                    # T2aN0M0
    assert result.tnm_string == "T2aN0M0"
    assert any("M0" in w for w in result.rationale.warnings)
    assert any("M defaulted" in c for c in result.rationale.conflicts_detected)


def test_missing_t_is_indeterminate():
    result = _stage(None, "N0", "M0")
    assert result.overall_stage == "indeterminate"
    assert result.tnm_string == "TXN0M0"
    assert any("T category unavailable" in w for w in result.rationale.warnings)


def test_all_missing_is_indeterminate():
    result = _stage(None, None, None)
    assert result.overall_stage == "indeterminate"
    assert result.tnm_string == "TXNXM0"


# --- Confidence & rationale -----------------------------------------------

def test_confidence_is_min_of_agents():
    result = _stage("T2a", "N1", "M0", tconf=0.9, nmconf=0.4)
    assert result.confidence == pytest.approx(0.4)
    assert any("Low Clinical Context confidence" in w for w in result.rationale.warnings)


def test_rationale_records_raw_and_final():
    result = _stage("T3", "N2", "M0")
    r = result.rationale
    assert (r.final_t, r.final_n, r.final_m) == ("T3", "N2", "M0")
    assert (r.t_raw, r.n_raw, r.m_raw) == ("T3", "N2", "M0")
    assert r.overall_stage == "IIIB"
    assert r.rules_applied and "IIIB" in r.rules_applied[0]


def test_unrecognized_category_warns():
    result = _stage("T9", "N0", "M0")   # T9 is not a valid category
    assert any("Unrecognized T category" in w for w in result.rationale.warnings)


# --- Agent construction ----------------------------------------------------

def test_default_conflict_resolution_is_conservative():
    assert GuidelineLogicAgent({}).conflict_resolution == ConflictResolution.CONSERVATIVE


def test_lazy_load_on_run():
    # run() should auto-load rules without an explicit load_rules() call.
    a = GuidelineLogicAgent({})
    te = TFactorEvidence(t_category="T1a", confidence=1.0)
    ne = NMFactorEvidence(n_category="N0", m_category="M0", confidence=1.0)
    assert a.run(te, ne).overall_stage == "IA1"
