"""Longitudinal (budgeted) RAG ablation — RAG-select vs truncation under a fixed
context budget.

See reports/PREREGISTRATION_rag_longitudinal.md for the hypothesis, endpoints, and
decision rule (fixed BEFORE running).

Difference from rag_ablation.py: that script compares RAG against a baseline that
reads the *entire* record. Here both arms get the SAME budget of k sentences and
differ only in selection:
  - RAG arm         : the k sentences most relevant to the N/M query (semantic).
  - truncation arm  : the first k sentences of the concatenated record (naive).
This tests whether retrieval beats naive truncation when the record (median 19
notes) cannot fit the budget — the setting where RAG is actually meant to help.

Metrics mirror rag_ablation.py (accuracy, quadratic-weighted κ, paired Wilcoxon on
|N_pred − N_ref|). Primary endpoint: binary N0/N+ at k = 8.

NOTE: credentialed MIMIC data. Outputs (subject_ids) go to results/ablation/
(gitignored). Do not commit per-patient outputs.

Usage:
  python -m scripts.evaluation.rag_longitudinal                 # k=8 primary + {4,16}
  python -m scripts.evaluation.rag_longitudinal --budgets 8
  python -m scripts.evaluation.rag_longitudinal --refresh-cache
"""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path

import pandas as pd

from agents.clinical_context import ClinicalContextAgent
from scripts.evaluation.rag_ablation import (
    reference_n, load_cohort_notes, compute_stats, NOTE_DIR, COHORT_NOTES, OUT_DIR,
)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

NOTES_CACHE = OUT_DIR / "cohort_notes_cache.pkl"   # gitignored; avoids re-reading 1.9 GB


def get_notes(note_dir: Path, cohort_path: Path, refresh: bool) -> dict[int, str]:
    """Load (and cache) each subject's concatenated record text."""
    if NOTES_CACHE.exists() and not refresh:
        with open(NOTES_CACHE, "rb") as f:
            return pickle.load(f)
    subjects = pd.read_csv(cohort_path)["subject_id"].tolist()
    notes = load_cohort_notes(note_dir, subjects)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(NOTES_CACHE, "wb") as f:
        pickle.dump(notes, f)
    return notes


def first_k_sentences(agent: ClinicalContextAgent, text: str, k: int) -> str:
    """The naive truncation baseline: the first k sentences of the record."""
    sents = [s for s, _ in agent._sentences(text) if s.strip()]
    return " ".join(sents[:k])


def run_budgeted(notes_by_subject: dict[int, str], k: int) -> pd.DataFrame:
    """For each subject with a reference N, compare RAG-select vs first-k truncation
    under a k-sentence budget. Columns match rag_ablation for compute_stats reuse."""
    agent = ClinicalContextAgent({"rag_k": k, "use_semantic": True})
    from agents.clinical_context import _N_ORDER as _N_ORD

    rows = []
    for sid, text in notes_by_subject.items():
        ref = reference_n(text)
        if ref is None:
            continue
        full_note = [{"note_id": str(sid), "note_type": "mimic", "text": text}]
        trunc_note = [{"note_id": str(sid), "note_type": "mimic",
                       "text": first_k_sentences(agent, text, k)}]

        # RAG arm: retrieve the k most relevant sentences from the whole record.
        pred_rag = agent.run(full_note, patient_id=str(sid), use_rag=True).n_category
        # Truncation arm: extract from the first k sentences only (no retrieval).
        pred_trunc = agent.run(trunc_note, patient_id=str(sid), use_rag=False).n_category

        rows.append({
            "subject_id": sid, "ref_n": ref,
            "pred_rag": pred_rag, "pred_norag": pred_trunc,     # 'norag' = truncation here
            "err_rag": abs(_N_ORD[pred_rag] - _N_ORD[ref]),
            "err_norag": abs(_N_ORD[pred_trunc] - _N_ORD[ref]),
        })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="Budgeted RAG-vs-truncation N-stage ablation.")
    ap.add_argument("--budgets", type=int, nargs="+", default=[8, 4, 16],
                    help="k-sentence budgets; the FIRST is the primary endpoint (default 8).")
    ap.add_argument("--note-dir", type=Path, default=NOTE_DIR)
    ap.add_argument("--cohort", type=Path, default=COHORT_NOTES)
    ap.add_argument("--refresh-cache", action="store_true")
    args = ap.parse_args()

    notes = get_notes(args.note_dir, args.cohort, args.refresh_cache)
    print(f"Subjects with note text: {len(notes)}")

    all_stats = []
    for i, k in enumerate(args.budgets):
        df = run_budgeted(notes, k)
        if df.empty:
            print(f"k={k}: no patients with an extractable reference N.")
            continue
        df.to_csv(OUT_DIR / f"rag_longitudinal_k{k}_per_patient.csv", index=False)
        stats = compute_stats(df)
        stats["k_budget"] = k
        stats["is_primary"] = (i == 0)
        all_stats.append(stats)

        tag = "  <-- PRIMARY" if i == 0 else ""
        print(f"\n=== Budgeted RAG vs truncation, k={k} sentences ==={tag}")
        print(f"Eval patients (with reference N): {stats['n_patients']}")
        print(f"[binary N0/N+]   RAG: {stats['bin_acc_rag']:.3f}   truncation: {stats['bin_acc_norag']:.3f}")
        print(f"[4-class N0-N3]  RAG: {stats['accuracy_rag']:.3f}   truncation: {stats['accuracy_norag']:.3f}")
        print(f"[4-class κ(qw)]  RAG: {stats['kappa_rag']:.3f}   truncation: {stats['kappa_norag']:.3f}")
        print(f"Mean |err|       RAG: {stats['mean_err_rag']:.3f}   truncation: {stats['mean_err_norag']:.3f}")
        if stats.get("wilcoxon_p") is not None:
            print(f"Wilcoxon (err_trunc vs err_rag): W={stats['wilcoxon_stat']:.1f}, "
                  f"p={stats['wilcoxon_p']:.4g}  (RAG better on {stats['n_rag_better']}, "
                  f"worse on {stats['n_norag_better']})")
        else:
            print(stats.get("note"))

    if all_stats:
        pd.DataFrame(all_stats).to_csv(OUT_DIR / "rag_longitudinal_summary.csv", index=False)
        print(f"\nSaved per-k + summary to {OUT_DIR}/ (gitignored)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
