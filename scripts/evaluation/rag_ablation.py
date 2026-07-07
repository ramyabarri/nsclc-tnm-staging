"""RAG ablation — does retrieval improve N-stage extraction from clinical notes?

Compares the Clinical Context Agent WITH vs WITHOUT RAG retrieval on MIMIC-IV
notes. Reference N-stage is a high-precision SILVER standard extracted from
explicit staging statements (TNM strings / cN/pN mentions) in the note. Because
the agent infers N from anatomical *findings* (lymph-node locations) while the
reference comes from the explicit *stage statement*, the two signals are
distinct; and since this is a PAIRED comparison against the same reference, the
Wilcoxon test detects the RAG effect validly even if the reference is silver.

Metrics:
  - Accuracy (exact N match) with vs without RAG
  - Cohen's quadratic-weighted Kappa (prediction vs reference), each config
  - Wilcoxon signed-rank on per-patient ordinal error |N_pred - N_ref|

NOTE: operates on credentialed MIMIC data. Outputs (subject_ids) are written to
results/ablation/ which is gitignored. Do not commit per-patient outputs.

Usage:
  python -m scripts.evaluation.rag_ablation --limit 300
  python -m scripts.evaluation.rag_ablation --retrieval keyword   # force fallback
"""

from __future__ import annotations

import argparse
import logging
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from agents.clinical_context import ClinicalContextAgent

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

NOTE_DIR = Path("data/raw/mimic-iv-note-2.2/note")
COHORT_NOTES = Path("data/processed/cohort/cohort_notes_index.csv")
OUT_DIR = Path("results/ablation")

_N_ORD = {"N0": 0, "N1": 1, "N2": 2, "N3": 3}

# High-precision reference extraction from explicit staging statements.
# N adjacent to an M-stage (…N2M0) or a T-stage (T2N2…) — high precision.
_TNM = re.compile(
    r"\b(?:T[0-4][a-c]?\s*)?[cp]?N\s*([0-3])\s*M[0-1][a-c]?\b"   # N#M#, opt T/c/p
    r"|\bT[0-4][a-c]?\s*[cp]?N\s*([0-3])\b",                      # T#N#
    re.I,
)
# Standalone N-stage fallback (lower precision).
_NSTAGE = re.compile(r"\b[cp]?N\s*([0-3])\b", re.I)


def reference_n(text: str) -> str | None:
    """Silver-standard N-stage from explicit staging statements; None if absent."""
    hits = [int(m.group(1) or m.group(2)) for m in _TNM.finditer(text)]  # prefer T/M-anchored
    if not hits:
        hits = [int(m.group(1)) for m in _NSTAGE.finditer(text)]
    if not hits:
        return None
    return f"N{Counter(hits).most_common(1)[0][0]}"


def load_cohort_notes(note_dir, cohort_subjects, max_chars=40000):
    """Concatenate each cohort subject's note text (discharge + radiology)."""
    texts: dict[int, list[str]] = {}
    cohort = set(cohort_subjects)
    for name in ("discharge.csv.gz", "radiology.csv.gz"):
        path = note_dir / name
        if not path.exists():
            logger.warning("missing %s", path)
            continue
        for chunk in pd.read_csv(path, compression="gzip",
                                 usecols=["subject_id", "text"], chunksize=50_000):
            chunk = chunk[chunk["subject_id"].isin(cohort)]
            for sid, txt in zip(chunk["subject_id"], chunk["text"]):
                if isinstance(txt, str):
                    texts.setdefault(int(sid), []).append(txt)
    return {sid: "\n".join(t)[:max_chars] for sid, t in texts.items()}


def run_ablation(notes_by_subject, rag_k, retrieval):
    """Run WITH and WITHOUT RAG per subject that has a reference N. Returns a df."""
    # One agent reused across patients so the embedder loads only once.
    agent = ClinicalContextAgent({"rag_k": rag_k, "use_semantic": retrieval == "semantic"})

    rows = []
    for sid, text in notes_by_subject.items():
        ref = reference_n(text)
        if ref is None:
            continue
        note = [{"note_id": str(sid), "note_type": "mimic", "text": text}]
        pred_rag = agent.run(note, patient_id=str(sid), use_rag=True).n_category
        pred_nor = agent.run(note, patient_id=str(sid), use_rag=False).n_category
        rows.append({
            "subject_id": sid, "ref_n": ref,
            "pred_rag": pred_rag, "pred_norag": pred_nor,
            "err_rag": abs(_N_ORD[pred_rag] - _N_ORD[ref]),
            "err_norag": abs(_N_ORD[pred_nor] - _N_ORD[ref]),
        })
    return pd.DataFrame(rows)


def compute_stats(df):
    from sklearn.metrics import cohen_kappa_score
    from scipy.stats import wilcoxon

    refs = df["ref_n"].map(_N_ORD).to_numpy()
    rag = df["pred_rag"].map(_N_ORD).to_numpy()
    nor = df["pred_norag"].map(_N_ORD).to_numpy()

    # Binary node status (N0 vs N+) — a more robust primary endpoint than 4-class.
    ref_pos = (df["ref_n"] != "N0").to_numpy()
    rag_pos = (df["pred_rag"] != "N0").to_numpy()
    nor_pos = (df["pred_norag"] != "N0").to_numpy()

    stats = {
        "n_patients": len(df),
        # 4-class N (N0-N3)
        "accuracy_rag": float((df["pred_rag"] == df["ref_n"]).mean()),
        "accuracy_norag": float((df["pred_norag"] == df["ref_n"]).mean()),
        "kappa_rag": float(cohen_kappa_score(refs, rag, weights="quadratic")),
        "kappa_norag": float(cohen_kappa_score(refs, nor, weights="quadratic")),
        # binary node status (N0 vs N+)
        "bin_acc_rag": float((ref_pos == rag_pos).mean()),
        "bin_acc_norag": float((ref_pos == nor_pos).mean()),
        "bin_kappa_rag": float(cohen_kappa_score(ref_pos, rag_pos)),
        "bin_kappa_norag": float(cohen_kappa_score(ref_pos, nor_pos)),
        "mean_err_rag": float(df["err_rag"].mean()),
        "mean_err_norag": float(df["err_norag"].mean()),
    }
    diff = df["err_norag"] - df["err_rag"]          # positive => RAG better
    if (diff != 0).any():
        w = wilcoxon(df["err_norag"], df["err_rag"], zero_method="wilcox",
                     alternative="two-sided")
        stats["wilcoxon_stat"] = float(w.statistic)
        stats["wilcoxon_p"] = float(w.pvalue)
        stats["n_rag_better"] = int((diff > 0).sum())
        stats["n_norag_better"] = int((diff < 0).sum())
    else:
        stats["wilcoxon_stat"] = None
        stats["wilcoxon_p"] = None
        stats["note"] = "RAG and no-RAG produced identical N on every patient"
    return stats


def main():
    ap = argparse.ArgumentParser(description="RAG ablation for N-stage extraction.")
    ap.add_argument("--limit", type=int, default=0, help="cap eval-set size (0 = all)")
    ap.add_argument("--rag-k", type=int, default=8)
    ap.add_argument("--retrieval", choices=["semantic", "keyword"], default="semantic")
    ap.add_argument("--note-dir", type=Path, default=NOTE_DIR)
    ap.add_argument("--cohort", type=Path, default=COHORT_NOTES)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cohort_subjects = pd.read_csv(args.cohort)["subject_id"].tolist()
    print(f"Cohort subjects: {len(cohort_subjects)}")

    notes = load_cohort_notes(args.note_dir, cohort_subjects)
    print(f"Subjects with note text: {len(notes)}")
    if args.limit:
        notes = dict(list(notes.items())[: args.limit])

    df = run_ablation(notes, args.rag_k, args.retrieval)
    if df.empty:
        print("No patients with an extractable reference N-stage — cannot run ablation.")
        return

    df.to_csv(OUT_DIR / "rag_ablation_per_patient.csv", index=False)
    stats = compute_stats(df)

    print(f"\n=== RAG ablation (retrieval={args.retrieval}, rag_k={args.rag_k}) ===")
    print(f"Eval patients (with reference N): {stats['n_patients']}")
    print(f"[4-class N0-N3]  Accuracy WITH RAG: {stats['accuracy_rag']:.3f}   WITHOUT: {stats['accuracy_norag']:.3f}")
    print(f"[4-class N0-N3]  Kappa(qw) WITH RAG: {stats['kappa_rag']:.3f}   WITHOUT: {stats['kappa_norag']:.3f}")
    print(f"[binary N0/N+]   Accuracy WITH RAG: {stats['bin_acc_rag']:.3f}   WITHOUT: {stats['bin_acc_norag']:.3f}")
    print(f"[binary N0/N+]   Kappa    WITH RAG: {stats['bin_kappa_rag']:.3f}   WITHOUT: {stats['bin_kappa_norag']:.3f}")
    print(f"Mean |err| WITH RAG: {stats['mean_err_rag']:.3f}   WITHOUT RAG: {stats['mean_err_norag']:.3f}")
    if stats.get("wilcoxon_p") is not None:
        print(f"Wilcoxon signed-rank (err_norag vs err_rag): "
              f"W={stats['wilcoxon_stat']:.1f}, p={stats['wilcoxon_p']:.4g}  "
              f"(RAG better on {stats['n_rag_better']}, worse on {stats['n_norag_better']})")
    else:
        print(stats.get("note"))

    pd.DataFrame([stats]).to_csv(OUT_DIR / "rag_ablation_summary.csv", index=False)
    print(f"\nSaved per-patient + summary to {OUT_DIR}/ (gitignored)")


if __name__ == "__main__":
    main()
