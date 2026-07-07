# Runs the 2 baseline segmentation models on the sanity sample and writes a
# CSV + markdown summary.
#
# Models expect RAW HU, so the CT is reconstructed from the DICOM series (not the
# z-score-normalised .npy volumes) and the GTV ground truth is taken from the
# RTSTRUCT (GTV-1), matching how the nnU-Net model was trained.
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
from scripts.preprocessing.prepare_nnunet_dataset import (
    index_series, load_ct_series, load_gtv_mask, _pick_ct_dir,
)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

DEFAULT_PATIENTS = [
    "LUNG1-001", "LUNG1-013", "LUNG1-303", "LUNG1-378",
    "LUNG1-328", "LUNG1-380", "LUNG1-048", "LUNG1-280",
]
RAW_DIR = Path("data/raw/NSCLC-Radiomics")

# Trained nnU-Net model folder (matches the 250-epoch trainer). Override with
# --nnunet-model if you trained the full nnUNetTrainer (1000 epochs).
NNUNET_MODEL_FOLDER = Path(
    "results/nnunet/Dataset001_NSCLCRadiomics"
    "/nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres"
)

DEFAULT_OUTPUT = Path("results/segmentation/week5_baselines.csv")
MODELS = ["nnunet", "totalsegmentator"]


def load_raw_ct_gt(patient_id, raw_dir):
    """Reconstruct the raw-HU CT (SimpleITK image) and GTV-1 ground truth from
    the patient's DICOM. Returns (ct_image, gt_mask_zyx, spacing_zyx)."""
    patient_dir = Path(raw_dir) / patient_id
    idx = index_series(patient_dir)
    if not idx:
        raise FileNotFoundError(f"No DICOM series under {patient_dir}")
    mods = idx.get(patient_id) or next(iter(idx.values()))
    if "CT" not in mods or "RTSTRUCT" not in mods:
        raise ValueError(f"{patient_id}: missing CT or RTSTRUCT")
    ct_dir = _pick_ct_dir(mods["CT"])
    rt_file = sorted(mods["RTSTRUCT"][0].glob("*.dcm"))[0]
    ct_image = load_ct_series(ct_dir)
    gt, _label = load_gtv_mask(ct_dir, rt_file, ct_image)
    spacing_zyx = tuple(ct_image.GetSpacing()[::-1])  # (sx,sy,sz) -> (sz,sy,sx)
    return ct_image, gt, spacing_zyx


def run_model(model, ct_image, nnunet_model_folder):
    """Run one model on a raw-HU CT image. Returns (mask, time_sec, notes)."""
    if model == "nnunet":
        mask, elapsed = run_nnunet_inference(ct_image, nnunet_model_folder)
        notes = "trained nnU-Net (GTV)" if mask.sum() else "empty/zero prediction"
        return mask, elapsed, notes

    if model == "totalsegmentator":
        try:
            mask, elapsed = run_totalsegmentator_inference(ct_image)
            notes = "lung_nodules class 2 (nodule detection, not GTV)"
        except Exception as exc:
            logger.warning("TotalSegmentator failed: %s", exc)
            import SimpleITK as sitk
            mask = np.zeros(sitk.GetArrayFromImage(ct_image).shape, dtype=np.uint8)
            elapsed, notes = 0.0, f"TotalSegmentator failed ({exc})"
        return mask, elapsed, notes

    raise ValueError(f"Unknown model: {model}")


def write_markdown_summary(summary, path):
    columns = list(summary.columns)
    lines = [
        "# Week 5 Vision Agent Baselines — Summary",
        "",
        "_Raw-HU input reconstructed from DICOM; GTV-1 ground truth from RTSTRUCT._",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in summary.iterrows():
        cells = [f"{row[c]:.3f}" if isinstance(row[c], float) else str(row[c]) for c in columns]
        lines.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run Vision Agent segmentation baselines (raw HU).")
    parser.add_argument("--patients", nargs="+", default=DEFAULT_PATIENTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR, help="NSCLC-Radiomics DICOM root")
    parser.add_argument("--nnunet-model", type=Path, default=NNUNET_MODEL_FOLDER)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    for patient_id in tqdm(args.patients, desc="patients"):
        try:
            ct_image, gt, spacing = load_raw_ct_gt(patient_id, args.raw_dir)
        except Exception as exc:
            logger.warning("Skipping %s: %s", patient_id, exc)
            continue

        for model in MODELS:
            mask, elapsed, notes = run_model(model, ct_image, args.nnunet_model)
            metrics = compute_all_metrics(mask, gt, spacing=spacing)
            rows.append({
                "patient_id": patient_id, "model": model, **metrics,
                "inference_time_sec": elapsed, "notes": notes,
            })

    results_df = pd.DataFrame(rows)
    results_df.to_csv(args.output, index=False)
    print(f"\nSaved {len(results_df)} rows to {args.output}")

    if results_df.empty:
        print("No results (all patients skipped).")
        return

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
            f"HD95 = {row['hd95_mean']:.1f} mm   time = {row['inference_time_mean']:.1f} s"
        )

    summary_path = args.output.parent / "week5_baselines_summary.md"
    write_markdown_summary(summary, summary_path)
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
