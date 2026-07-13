"""End-to-end TNM staging evaluation on the NSCLC-Radiomics cohort.

This is the cohort-level evaluation the pipeline was missing. It isolates and
measures each stage so errors can be attributed to the right module
(supervisor's per-component + end-to-end table):

  1. Vision T-classification  — GTV ground-truth mask -> VisionAgent size->T
     mapping -> coarse T (1-4), compared to the clinical T stage. This decouples
     the T-*classification* logic from segmentation quality (Dice is measured
     separately by run_baselines.py), so the number here reflects the size->T
     rule alone.
  2. End-to-end stage         — predicted T (from imaging) + ground-truth N/M
     -> GuidelineLogicAgent -> AJCC stage, compared to the clinical overall
     stage. NSCLC-Radiomics has no free-text notes, so the Clinical Context
     agent's N/M is replaced here by the recorded clinical N/M (an oracle N/M),
     which is why this measures the Vision->Guideline path end-to-end. The
     Clinical Context / RAG path is evaluated separately on MIMIC.
  3. Guideline data-consistency — dataset ground-truth (T,N,M) -> guideline ->
     stage vs the dataset's own overall stage. A cross-check of the encoded
     table against the cohort's labelling (see caveats: coarse T labels and a
     possible edition mismatch).

Metrics: accuracy (+ bootstrap 95% CI), confusion matrix, quadratic-weighted
Cohen's kappa (ordinal), macro & weighted F1.

Ground-truth masks come from RTSTRUCT (GTV-1) — no nnU-Net needed, so this runs
on CPU. Results are cached per patient so re-runs are cheap and resumable.

Usage:
    python -m scripts.evaluation.stage_eval --sample 80
    python -m scripts.evaluation.stage_eval --all
    python -m scripts.evaluation.stage_eval --patients LUNG1-001 LUNG1-013
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd

from agents.vision import VisionAgent
from agents.guideline_logic import GuidelineLogicAgent
from agents.vision import TFactorEvidence
from agents.clinical_context import NMFactorEvidence
from scripts.preprocessing.prepare_nnunet_dataset import (
    index_series, load_ct_series, load_gtv_mask, _pick_ct_dir,
)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw/NSCLC-Radiomics")
CLINICAL_CSV = Path("data/raw/Lung1_clinical.csv")
OUT_DIR = Path("results/staging")
CACHE = OUT_DIR / "vision_t_cache.csv"

# ---- category / label mapping -------------------------------------------------

def coarse_t_from_category(cat: str | None) -> int | None:
    """Vision T category (T1a.., T4, TX) -> coarse T stage 1-4 (or None)."""
    if not cat or cat in ("TX", "T0"):
        return None
    if cat in ("Tis", "T1mi") or cat.startswith("T1"):
        return 1
    if cat.startswith("T2"):
        return 2
    if cat == "T3":
        return 3
    if cat == "T4":
        return 4
    return None


def coarse_t_to_token(t: int | None) -> str | None:
    """Ground-truth coarse T (1-4) -> a representative IASLC token to feed the
    guideline engine. Coarse labels don't resolve sub-categories; we use the
    canonical member of each group (T2 -> T2a). This ambiguity is reported."""
    return {1: "T1c", 2: "T2a", 3: "T3", 4: "T4"}.get(t)


def n_to_token(n) -> str | None:
    try:
        n = int(n)
    except (TypeError, ValueError):
        return None
    return {0: "N0", 1: "N1", 2: "N2", 3: "N3"}.get(n)   # 4 (3 cases) is invalid -> None


def m_to_token(m) -> str | None:
    try:
        m = int(m)
    except (TypeError, ValueError):
        return None
    if m == 0:
        return "M0"
    if m == 1:
        return "M1b"     # site unrecorded; M1 present
    return None          # value 3 (4 cases) is a data artefact -> None


def normalize_gt_stage(s) -> str | None:
    """Dataset Overall.Stage -> coarse scheme {I, II, IIIa, IIIb, IV}."""
    if not isinstance(s, str):
        return None
    s = s.strip()
    mapping = {"i": "I", "ii": "II", "iiia": "IIIa", "iiib": "IIIb",
               "iv": "IV", "iva": "IV", "ivb": "IV"}
    return mapping.get(s.lower())


def agent_stage_to_coarse(stage: str | None) -> str | None:
    """AJCC stage from the guideline engine -> the dataset's coarse scheme."""
    if not stage or stage == "indeterminate":
        return None
    if stage in ("0",):
        return "0"
    if stage.startswith("IA") or stage == "IB" or stage == "I":
        return "I"
    if stage.startswith("II") and not stage.startswith("III"):
        return "II"
    if stage == "IIIA":
        return "IIIa"
    if stage in ("IIIB", "IIIC"):
        return "IIIb"
    if stage.startswith("IV"):
        return "IV"
    return None


# ---- imaging: predict T from the GTV ground-truth mask ------------------------

def predict_t_for_patient(pid: str, raw_dir: Path) -> dict:
    """Rasterise the GTV-1 mask and run the Vision T-classifier on it (real
    agent code, native per-patient spacing). Returns a row dict."""
    patient_dir = raw_dir / pid
    idx = index_series(patient_dir)
    mods = idx.get(pid) or (next(iter(idx.values())) if idx else None)
    if not mods or "CT" not in mods or "RTSTRUCT" not in mods:
        return {"patient_id": pid, "error": "missing CT/RTSTRUCT"}

    ct_dir = _pick_ct_dir(mods["CT"])
    rt_file = sorted(mods["RTSTRUCT"][0].glob("*.dcm"))[0]
    ct_image = load_ct_series(ct_dir)
    mask, roi_label = load_gtv_mask(ct_dir, rt_file, ct_image)
    spacing_zyx = tuple(ct_image.GetSpacing()[::-1])   # (sx,sy,sz) -> (sz,sy,sx)

    agent = VisionAgent({"spacing_zyx": spacing_zyx})
    radiomics = agent.extract_radiomics(mask, mask)     # volume unused w/o pyradiomics
    t_cat, conf = agent.classify_t_factor(radiomics, mask)
    return {
        "patient_id": pid,
        "roi_label": roi_label,
        "diameter_mm": round(radiomics.get("max_diameter_mm", 0.0), 2),
        "volume_cm3": round(radiomics.get("volume_mm3", 0.0) / 1000.0, 2),
        "pred_t_category": t_cat,
        "pred_t_coarse": coarse_t_from_category(t_cat),
        "vision_confidence": conf,
        "error": "",
    }


# ---- metrics ------------------------------------------------------------------

def _bootstrap_acc_ci(y_true, y_pred, n=2000, seed=0):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    if len(y_true) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    accs = []
    idx = np.arange(len(y_true))
    for _ in range(n):
        s = rng.choice(idx, size=len(idx), replace=True)
        accs.append(float(np.mean(y_true[s] == y_pred[s])))
    return (round(float(np.percentile(accs, 2.5)), 3),
            round(float(np.percentile(accs, 97.5)), 3))


def _missing(x) -> bool:
    """True for None or NaN (pandas turns None columns into float NaN)."""
    if x is None:
        return True
    try:
        return bool(np.isnan(x))
    except (TypeError, ValueError):
        return False


def score_block(name, y_true, y_pred, ordinal=False):
    """Compute the metric bundle for one paired (true, pred) list."""
    from sklearn.metrics import (accuracy_score, cohen_kappa_score, f1_score,
                                 confusion_matrix)
    pairs = [(a, b) for a, b in zip(y_true, y_pred) if not _missing(a) and not _missing(b)]
    # Normalise numeric labels to int so 2.0 and 2 don't split classes.
    pairs = [(int(a) if isinstance(a, float) else a,
              int(b) if isinstance(b, float) else b) for a, b in pairs]
    yt = [a for a, _ in pairs]
    yp = [b for _, b in pairs]
    labels = sorted(set(yt) | set(yp), key=lambda x: (str(type(x)), x))
    out = {
        "name": name,
        "n": len(pairs),
        "n_skipped": len(y_true) - len(pairs),
        "labels": [str(x) for x in labels],
    }
    if not pairs:
        return out, None
    out["accuracy"] = round(accuracy_score(yt, yp), 3)
    lo, hi = _bootstrap_acc_ci(yt, yp)
    out["accuracy_ci95"] = [lo, hi]
    out["macro_f1"] = round(f1_score(yt, yp, average="macro", zero_division=0), 3)
    out["weighted_f1"] = round(f1_score(yt, yp, average="weighted", zero_division=0), 3)
    try:
        w = "quadratic" if ordinal else None
        out["cohen_kappa"] = round(cohen_kappa_score(yt, yp, weights=w), 3)
        out["kappa_weighting"] = w or "unweighted"
    except Exception as exc:
        out["cohen_kappa"] = None
        out["kappa_error"] = str(exc)
    cm = confusion_matrix(yt, yp, labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"true_{l}" for l in labels],
                         columns=[f"pred_{l}" for l in labels])
    return out, cm_df


# ---- driver -------------------------------------------------------------------

def select_patients(args, gt: pd.DataFrame) -> list[str]:
    if args.patients:
        return args.patients
    available = [p.name for p in RAW_DIR.iterdir() if p.is_dir() and p.name.startswith("LUNG1-")]
    available = [p for p in available if p in set(gt.index)]
    available.sort()
    if args.all:
        return available
    rng = random.Random(args.seed)
    return sorted(rng.sample(available, min(args.sample, len(available))))


def build_vision_table(patients, gt, use_cache=True) -> pd.DataFrame:
    cached = {}
    if use_cache and CACHE.exists():
        prev = pd.read_csv(CACHE)
        cached = {r["patient_id"]: r.to_dict() for _, r in prev.iterrows()}

    rows = []
    from tqdm.auto import tqdm
    for pid in tqdm(patients, desc="vision-T"):
        if pid in cached and not cached[pid].get("error"):
            rows.append(cached[pid]); continue
        try:
            rows.append(predict_t_for_patient(pid, RAW_DIR))
        except Exception as exc:
            logger.warning("Vision failed for %s: %s", pid, exc)
            rows.append({"patient_id": pid, "error": str(exc)})

    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Merge with any prior cache so the cache accumulates across sample runs.
    if use_cache and CACHE.exists():
        prev = pd.read_csv(CACHE)
        df = pd.concat([prev[~prev["patient_id"].isin(df["patient_id"])], df],
                       ignore_index=True)
    df.to_csv(CACHE, index=False)
    return df


def run_guideline(t_token, n_token, m_token) -> str | None:
    agent = _guideline_singleton()
    if t_token is None or n_token is None or m_token is None:
        return None
    res = agent.run(TFactorEvidence(t_category=t_token, confidence=1.0),
                    NMFactorEvidence(n_category=n_token, m_category=m_token, confidence=1.0))
    return res.overall_stage


_GUIDELINE = None
def _guideline_singleton():
    global _GUIDELINE
    if _GUIDELINE is None:
        _GUIDELINE = GuidelineLogicAgent()
        _GUIDELINE.load_rules()
    return _GUIDELINE


def main() -> int:
    ap = argparse.ArgumentParser(description="End-to-end TNM staging evaluation.")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--all", action="store_true", help="all locally-available patients")
    g.add_argument("--sample", type=int, default=80, help="random sample size")
    ap.add_argument("--patients", nargs="+", help="explicit patient ids")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    gt = pd.read_csv(CLINICAL_CSV).set_index("PatientID")
    patients = select_patients(args, gt)
    print(f"Evaluating {len(patients)} patients.")

    vision = build_vision_table(patients, gt, use_cache=not args.no_cache)
    vision = vision[vision["patient_id"].isin(patients)]

    # Assemble the paired evaluation records.
    rec = []
    for _, vr in vision.iterrows():
        pid = vr["patient_id"]
        if pid not in gt.index:
            continue
        grow = gt.loc[pid]
        gt_t = int(grow["clinical.T.Stage"]) if pd.notna(grow["clinical.T.Stage"]) and grow["clinical.T.Stage"] <= 4 else None
        gt_n_tok = n_to_token(grow["Clinical.N.Stage"])
        gt_m_tok = m_to_token(grow["Clinical.M.Stage"])
        gt_stage = normalize_gt_stage(grow["Overall.Stage"])
        pred_t_coarse = vr.get("pred_t_coarse")
        pred_t_coarse = int(pred_t_coarse) if pd.notna(pred_t_coarse) else None

        # end-to-end stage: predicted T (imaging) + oracle N/M
        e2e_stage = agent_stage_to_coarse(run_guideline(
            coarse_t_to_token(pred_t_coarse), gt_n_tok, gt_m_tok))
        # guideline data-consistency: ground-truth T + N + M
        cons_stage = agent_stage_to_coarse(run_guideline(
            coarse_t_to_token(gt_t), gt_n_tok, gt_m_tok))

        rec.append({
            "patient_id": pid, "error": vr.get("error", ""),
            "gt_t": gt_t, "pred_t": pred_t_coarse,
            "gt_stage": gt_stage, "e2e_stage": e2e_stage, "cons_stage": cons_stage,
        })
    recdf = pd.DataFrame(rec)
    recdf.to_csv(OUT_DIR / "staging_eval_records.csv", index=False)

    # Score the three dimensions.
    blocks, cms = [], {}
    b1, cm1 = score_block("Vision T-classification (coarse T1-4)",
                          recdf["gt_t"].tolist(), recdf["pred_t"].tolist(), ordinal=True)
    b2, cm2 = score_block("End-to-end stage (Vision T + oracle N/M)",
                          recdf["gt_stage"].tolist(), recdf["e2e_stage"].tolist(), ordinal=False)
    b3, cm3 = score_block("Guideline data-consistency (dataset T/N/M -> stage)",
                          recdf["gt_stage"].tolist(), recdf["cons_stage"].tolist(), ordinal=False)
    for b, cm, key in [(b1, cm1, "vision_t"), (b2, cm2, "end_to_end"), (b3, cm3, "guideline_consistency")]:
        blocks.append(b)
        if cm is not None:
            cms[key] = cm

    summary = {"n_patients": len(patients), "n_records": len(recdf), "blocks": blocks}
    (OUT_DIR / "staging_eval_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    _write_markdown(summary, cms, OUT_DIR / "staging_eval_summary.md")

    print("\n=== Staging evaluation ===")
    for b in blocks:
        acc = b.get("accuracy")
        ci = b.get("accuracy_ci95")
        line = f"  {b['name']}: n={b['n']}"
        if acc is not None:
            line += (f" | acc={acc} (95% CI {ci[0]}-{ci[1]})"
                     f" | κ={b.get('cohen_kappa')} ({b.get('kappa_weighting')})"
                     f" | macroF1={b.get('macro_f1')} | wF1={b.get('weighted_f1')}")
        if b.get("n_skipped"):
            line += f" | skipped={b['n_skipped']}"
        print(line)
    print(f"\nArtifacts in {OUT_DIR}/ (summary.md/json, records.csv, cache).")
    return 0


def _write_markdown(summary, cms, path):
    lines = ["# End-to-End TNM Staging Evaluation — NSCLC-Radiomics", "",
             f"_Patients evaluated:_ **{summary['n_records']}**", ""]
    lines += [
        "Vision T-classification uses the **GTV-1 ground-truth mask** as input, so it "
        "measures the size→T rule in isolation (segmentation Dice is reported separately). "
        "End-to-end stage combines the predicted T with the **recorded clinical N/M** "
        "(NSCLC-Radiomics has no free-text notes; the Clinical Context / RAG path is "
        "evaluated on MIMIC). Coarse ground-truth T labels (T1–T4) cannot resolve "
        "sub-categories, so T2 is fed to the guideline as T2a.", "",
        "| Component | n | Accuracy (95% CI) | Cohen κ | Macro F1 | Weighted F1 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for b in summary["blocks"]:
        if b.get("accuracy") is None:
            lines.append(f"| {b['name']} | {b['n']} | — | — | — | — |"); continue
        ci = b["accuracy_ci95"]
        lines.append(
            f"| {b['name']} | {b['n']} | {b['accuracy']} ({ci[0]}–{ci[1]}) | "
            f"{b.get('cohen_kappa')} ({b.get('kappa_weighting')}) | "
            f"{b.get('macro_f1')} | {b.get('weighted_f1')} |")
    lines.append("")
    for key, cm in cms.items():
        lines += [f"### Confusion matrix — {key}", "", "```",
                  cm.to_string(), "```", ""]
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
