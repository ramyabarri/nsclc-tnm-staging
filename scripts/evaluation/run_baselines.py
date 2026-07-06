# Runs the 2 baseline segmentation models on the 8-patient sanity sample
# and writes results to a CSV + markdown summary.
#
# Usage:
#   python -m scripts.evaluation.run_baselines
#   python -m scripts.evaluation.run_baselines --patients LUNG1-001 LUNG1-013

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from agents.vision.inference import (
    run_nnunet_inference,
    run_totalsegmentator_inference,
)
from agents.vision.metrics import compute_all_metrics

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

DEFAULT_PATIENTS = [
    "LUNG1-001",
    "LUNG1-013",
    "LUNG1-303",
    "LUNG1-378",
    "LUNG1-328",
    "LUNG1-380",
    "LUNG1-048",
    "LUNG1-280",
]
PROCESSED_DIR = Path("data/processed/radiomics")

# Path to the locally trained nnU-Net v2 model folder.
# Matches the trainer used in notebooks/experiments/02_nnunet_vastai_training.ipynb
# (nnUNetTrainer_250epochs). Layout after unzipping the downloaded model:
#   results/nnunet/Dataset001_NSCLCRadiomics/
#     nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres/fold_0/checkpoint_best.pth
# If you train with the full nnUNetTrainer (1000 epochs) instead, override with
# --nnunet-model pointing at the nnUNetTrainer__nnUNetPlans__3d_fullres folder.
NNUNET_MODEL_FOLDER = Path(
    "results/nnunet/Dataset001_NSCLCRadiomics"
    "/nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres"
)

DEFAULT_OUTPUT = Path("results/segmentation/week5_baselines.csv")

# (z, y, x) spacing in mm, matching the preprocessed (Z, Y, X) array layout.
SPACING_ZYX = (3.0, 1.0, 1.0)

MODELS = ["nnunet", "totalsegmentator"]


def run_model(model, patient_id, processed_dir, nnunet_model_folder):
    """Run one (model, patient) inference. Returns (mask, time_sec, notes)."""
    if model == "nnunet":
        mask, elapsed = run_nnunet_inference(patient_id, processed_dir, nnunet_model_folder)
        notes = "model folder not found - zero mask returned" if mask.sum() == 0 else ""
        return mask, elapsed, notes

    if model == "totalsegmentator":
        try:
            mask, elapsed = run_totalsegmentator_inference(patient_id, processed_dir)
            notes = "lung_nodules task - class 2 extracted as tumour candidate"
        except Exception as exc:
            logger.warning("TotalSegmentator failed for %s: %s", patient_id, exc)
            ct = np.load(processed_dir / f"{patient_id}_ct.npy")
            mask = np.zeros(ct.shape, dtype=np.uint8)
            elapsed = 0.0
            notes = f"TotalSegmentator failed ({exc}) - zero mask returned"
        return mask, elapsed, notes

    raise ValueError(f"Unknown model: {model}")


def write_markdown_summary(summary, path):
    """Write summary dataframe as a markdown table."""
    columns = list(summary.columns)
    lines = [
        "# Week 5 Vision Agent Baselines — Summary",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in summary.iterrows():
        cells = []
        for col in columns:
            value = row[col]
            cells.append(f"{value:.3f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run Vision Agent segmentation baselines.")
    parser.add_argument("--patients", nargs="+", default=DEFAULT_PATIENTS, help="Patient IDs to evaluate")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output CSV path")
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR, help="Directory with preprocessed .npy/.json")
    parser.add_argument("--nnunet-model", type=Path, default=NNUNET_MODEL_FOLDER, help="Path to locally trained nnU-Net model folder")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    pairs = [(patient_id, model) for patient_id in args.patients for model in MODELS]
    gt_cache: dict[str, np.ndarray] = {}

    rows = []
    for patient_id, model in tqdm(pairs, desc="patient x model"):
        if patient_id not in gt_cache:
            gt_cache[patient_id] = np.load(args.processed_dir / f"{patient_id}_mask.npy")
        gt = gt_cache[patient_id]

        mask, elapsed, notes = run_model(model, patient_id, args.processed_dir, args.nnunet_model)
        metrics = compute_all_metrics(mask, gt, spacing=SPACING_ZYX)

        rows.append({
            "patient_id": patient_id,
            "model": model,
            **metrics,
            "inference_time_sec": elapsed,
            "notes": notes,
        })

    results_df = pd.DataFrame(rows)
    results_df.to_csv(args.output, index=False)
    print(f"\nSaved {len(results_df)} rows to {args.output}")

    summary = results_df.groupby("model").agg(
        dice_mean=("dice", "mean"),
        dice_std=("dice", "std"),
        hd95_mean=("hd95_mm", "mean"),
        volume_error_mean=("volume_error_cm3", "mean"),
        inference_time_mean=("inference_time_sec", "mean"),
    ).reset_index()

    print("\nSummary (mean DSC +/- std, mean inference time):")
    for _, row in summary.iterrows():
        print(
            f"  {row['model']:18s} DSC = {row['dice_mean']:.3f} +/- {row['dice_std']:.3f}   "
            f"HD95 = {row['hd95_mean']:.1f} mm   "
            f"time = {row['inference_time_mean']:.1f} s"
        )

    summary_path = args.output.parent / "week5_baselines_summary.md"
    write_markdown_summary(summary, summary_path)
    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()
