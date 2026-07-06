"""Tests for the Clinical Context Agent (retrieve-only N/M extraction).

All note text here is SYNTHETIC (author-written) — no MIMIC data in the repo.
"""

import pytest

from agents.clinical_context import ClinicalContextAgent


def _note(text, nid="n1", ntype="radiology"):
    return {"note_id": nid, "text": text, "note_type": ntype}


def _run(text):
    return ClinicalContextAgent({}).run([_note(text)])


# --- M category ------------------------------------------------------------

def test_m0_no_mention():
    ev = _run("CT chest performed. No acute findings in the lungs.")
    assert ev.m_category == "M0"
    assert ev.distant_metastasis is False


def test_m0_explicit_negation():
    ev = _run("No evidence of distant metastatic disease.")
    assert ev.m_category == "M0"
    assert ev.distant_metastasis is False
    assert any(s.label.startswith("metastasis") and "negated" in s.label
               for s in ev.supporting_spans)


def test_m1a_malignant_pleural_effusion():
    ev = _run("Large malignant pleural effusion on the right.")
    assert ev.m_category == "M1a"
    assert ev.distant_metastasis is True


def test_m1b_single_extrathoracic():
    ev = _run("Findings consistent with hepatic metastases.")
    assert ev.m_category == "M1b"
    assert "liver" in ev.metastasis_sites


def test_m1c_multiple_extrathoracic():
    ev = _run("Metastatic disease involving the brain and multiple osseous structures.")
    assert ev.m_category == "M1c"
    assert {"brain", "bone"} <= set(ev.metastasis_sites)


def test_organ_word_without_metastasis_context_is_ignored():
    # 'liver' with no metastasis trigger must NOT be read as a met site.
    ev = _run("Liver function tests within normal limits.")
    assert ev.m_category == "M0"


# --- N category ------------------------------------------------------------

def test_n0_negated_nodes():
    ev = _run("No mediastinal or hilar lymphadenopathy.")
    assert ev.n_category == "N0"
    assert ev.lymph_node_positive is False


def test_n1_hilar():
    ev = _run("There is right hilar adenopathy.")
    assert ev.n_category == "N1"
    assert ev.lymph_node_positive is True


def test_n2_mediastinal():
    ev = _run("Enlarged subcarinal and mediastinal lymph nodes are present.")
    assert ev.n_category == "N2"


def test_n3_supraclavicular():
    ev = _run("Bulky supraclavicular lymphadenopathy noted.")
    assert ev.n_category == "N3"


def test_highest_n_category_wins():
    ev = _run("Right hilar adenopathy. Also supraclavicular lymphadenopathy.")
    assert ev.n_category == "N3"


# --- provenance & structure ------------------------------------------------

def test_spans_have_provenance():
    ev = _run("Hepatic metastases present. Mediastinal lymphadenopathy noted.")
    assert ev.supporting_spans
    for s in ev.supporting_spans:
        assert s.source_note_id == "n1"
        assert s.end_char > s.start_char
        assert 0.0 <= s.confidence <= 1.0


def test_combined_case_end_to_end():
    text = (
        "CT chest and abdomen. Enlarged mediastinal lymph nodes measuring up to 2 cm. "
        "Hepatic metastases and an adrenal metastasis are identified. "
        "No supraclavicular adenopathy."
    )
    ev = _run(text)
    assert ev.n_category == "N2"
    assert ev.m_category == "M1c"          # liver + adrenal = 2 organ systems
    assert ev.distant_metastasis is True
    assert 0.0 < ev.confidence <= 1.0


# --- retrieval (keyword fallback works without ML stack) -------------------

def test_retrieve_context_keyword_fallback():
    agent = ClinicalContextAgent({})
    agent.index_notes([_note(
        "The weather is fine. Mediastinal lymphadenopathy is present. Patient stable."
    )])
    ctx = agent.retrieve_context("lymph node metastasis", n_results=2)
    assert any("lymphadenopathy" in c.lower() for c in ctx)


def test_repr_says_retrieve_only():
    assert "retrieve_only=True" in repr(ClinicalContextAgent({}))
