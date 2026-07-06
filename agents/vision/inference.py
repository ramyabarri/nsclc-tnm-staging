# Inference wrappers for the 2 baseline segmentation models:
# 1. nnU-Net v2 (trained locally on NSCLC-Radiomics)
# 2. TotalSegmentator lung_nodules task

import logging
import tempfile
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk

logger = logging.getLogger(__name__)

# preprocessing pipeline uses (X, Y, Z) spacing = (1.0, 1.0, 3.0) mm
SPACING_XYZ = (1.0, 1.0, 3.0)

# nnU-Net v2 dataset name for the locally trained NSCLC-Radiomics model
NNUNET_DATASET_NAME = "Dataset001_NSCLCRadiomics"


def _array_to_nifti(volume, path):
    """Save a (Z, Y, X) numpy array as NIfTI with the pipeline spacing."""
    image = sitk.GetImageFromArray(volume)
    image.SetSpacing(SPACING_XYZ)
    sitk.WriteImage(image, str(path))


# --- 1. nnU-Net v2 (locally trained) ---

def run_nnunet_inference(patient_id, processed_dir, model_folder, device="cpu"):
    """Run locally trained nnU-Net v2 on one patient.

    Args:
        model_folder: Path to the nnU-Net results folder, e.g.
            results/nnunet/Dataset001_NSCLCRadiomics/nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres
        device: "cpu" or "cuda". Use "cuda" on the Vast.ai GPU box (seconds/patient).
    Returns:
        (mask, elapsed_sec) — mask is a binary (Z, Y, X) uint8 array.
        Returns a zero mask and logs a warning if inference fails.
    """
    ct = np.load(Path(processed_dir) / f"{patient_id}_ct.npy")
    model_folder = Path(model_folder)

    if not model_folder.exists():
        logger.warning(
            "nnU-Net model folder not found: %s — run nnUNet_plan_and_preprocess "
            "and nnUNet_train on the NSCLC-Radiomics cohort first.",
            model_folder,
        )
        return np.zeros(ct.shape, dtype=np.uint8), 0.0

    start = time.time()
    try:
        import torch
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_dir = tmp_path / "input"
            output_dir = tmp_path / "output"
            input_dir.mkdir()
            output_dir.mkdir()

            _array_to_nifti(ct, input_dir / f"{patient_id}_0000.nii.gz")

            predictor = nnUNetPredictor(
                tile_step_size=0.5,
                use_gaussian=True,
                use_mirroring=True,
                device=torch.device(device),
                verbose=False,
                allow_tqdm=False,
            )
            predictor.initialize_from_trained_model_folder(
                str(model_folder),
                use_folds=(0,),
                checkpoint_name="checkpoint_best.pth",
            )
            predictor.predict_from_files(
                str(input_dir), str(output_dir),
                save_probabilities=False, overwrite=True,
            )

            out_file = output_dir / f"{patient_id}.nii.gz"
            pred_image = sitk.ReadImage(str(out_file))
            pred_mask = (sitk.GetArrayFromImage(pred_image) > 0).astype(np.uint8)

    except Exception as exc:
        logger.warning("nnU-Net inference failed for %s: %s", patient_id, exc)
        pred_mask = np.zeros(ct.shape, dtype=np.uint8)

    return pred_mask, time.time() - start


# --- 2. TotalSegmentator (lung_nodules task) ---

def run_totalsegmentator_inference(patient_id, processed_dir):
    """Run TotalSegmentator lung_nodules task on one patient.

    Uses the task='lung_nodules' model which segments lung tissue (class 1)
    and lung nodules (class 2). We extract class 2 as the tumour candidate mask.
    Returns (mask, elapsed_sec).
    """
    ct = np.load(Path(processed_dir) / f"{patient_id}_ct.npy")

    from totalsegmentator.python_api import totalsegmentator

    start = time.time()
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        input_path = tmp_path / f"{patient_id}.nii.gz"
        output_dir = tmp_path / "segmentations"
        _array_to_nifti(ct, input_path)

        # The lung_nodules task has no "fast" variant — it rejects fast=True,
        # so we run the full-resolution model on CPU.
        totalsegmentator(
            str(input_path), str(output_dir),
            task="lung_nodules",
            fast=False, ml=False, device="cpu", quiet=True,
        )

        nodule_path = output_dir / "lung_nodules.nii.gz"
        if nodule_path.exists():
            pred_image = sitk.ReadImage(str(nodule_path))
            combined = (sitk.GetArrayFromImage(pred_image) > 0).astype(np.uint8)
        else:
            logger.warning("TotalSegmentator lung_nodules output missing for %s", patient_id)
            combined = np.zeros(ct.shape, dtype=np.uint8)

    return combined, time.time() - start
