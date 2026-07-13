# Inference wrappers for the 2 baseline segmentation models:
#   1. nnU-Net v2 (trained locally on NSCLC-Radiomics)
#   2. TotalSegmentator lung_nodules task
#
# Both models expect a RAW-HU CT (SimpleITK image) with correct geometry — they
# apply their own intensity normalisation internally. Do NOT pass the
# z-score-normalised pipeline .npy volumes here; that mismatch silently degrades
# predictions (nnU-Net/TotalSegmentator would see the wrong HU range).

import logging
import tempfile
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk

logger = logging.getLogger(__name__)

# nnU-Net v2 dataset name for the locally trained NSCLC-Radiomics model
NNUNET_DATASET_NAME = "Dataset001_NSCLCRadiomics"


# --- 1. nnU-Net v2 (locally trained) ---

def run_nnunet_inference(ct_image, model_folder, device="cpu"):
    """Run the trained nnU-Net v2 on a raw-HU CT.

    IMPORTANT: nnU-Net uses `spawn` multiprocessing for pre/post-processing. The
    calling script MUST be guarded by ``if __name__ == "__main__":`` — otherwise
    the worker processes re-execute the module top-level and the prediction
    silently degrades to an empty mask. (Notebooks are fine; standalone scripts
    are the hazard.)

    Args:
        ct_image: SimpleITK.Image in raw HU, with correct spacing/origin/direction
            (e.g. from load_ct_series on the DICOM series). nnU-Net resamples and
            normalises internally.
        model_folder: nnU-Net results folder, e.g.
            results/nnunet/Dataset001_NSCLCRadiomics/nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres
        device: "cpu" or "cuda".
    Returns:
        (mask, elapsed_sec) — binary (Z, Y, X) uint8 mask aligned to ct_image.
        Returns a zero mask and logs a warning if the model is missing or fails.
    """
    model_folder = Path(model_folder)
    ref_shape = sitk.GetArrayFromImage(ct_image).shape

    if not model_folder.exists():
        logger.warning(
            "nnU-Net model folder not found: %s — download/unzip the trained model "
            "into results/nnunet/ first.", model_folder,
        )
        return np.zeros(ref_shape, dtype=np.uint8), 0.0

    start = time.time()
    try:
        import torch
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            (tmp / "in").mkdir()
            (tmp / "out").mkdir()
            sitk.WriteImage(ct_image, str(tmp / "in" / "case_0000.nii.gz"))

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
                str(tmp / "in"), str(tmp / "out"),
                save_probabilities=False, overwrite=True,
            )
            pred_image = sitk.ReadImage(str(tmp / "out" / "case.nii.gz"))
            pred_mask = (sitk.GetArrayFromImage(pred_image) > 0).astype(np.uint8)

    except Exception as exc:
        # Log the full traceback, not just the message — a silent zero mask here
        # is indistinguishable from a genuine empty prediction and was masking
        # real failures (missing torch, checkpoint mismatch, multiprocessing).
        logger.warning("nnU-Net inference failed: %s", exc, exc_info=True)
        pred_mask = np.zeros(ref_shape, dtype=np.uint8)

    return pred_mask, time.time() - start


# --- 2. TotalSegmentator (lung_nodules task) ---

def run_totalsegmentator_inference(ct_image, device="cpu"):
    """Run TotalSegmentator lung_nodules on a raw-HU CT.

    The lung_nodules task segments lung tissue (class 1) and lung nodules
    (class 2); we extract the nodule mask. Returns (mask, elapsed_sec).
    """
    from totalsegmentator.python_api import totalsegmentator

    ref_shape = sitk.GetArrayFromImage(ct_image).shape
    ts_device = "gpu" if device in ("cuda", "gpu") else "cpu"

    start = time.time()
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        input_path = tmp / "input.nii.gz"
        output_dir = tmp / "segmentations"
        sitk.WriteImage(ct_image, str(input_path))

        # lung_nodules has no "fast" variant (it rejects fast=True).
        totalsegmentator(
            str(input_path), str(output_dir),
            task="lung_nodules",
            fast=False, ml=False, device=ts_device, quiet=True,
        )

        nodule_path = output_dir / "lung_nodules.nii.gz"
        if nodule_path.exists():
            pred_image = sitk.ReadImage(str(nodule_path))
            mask = (sitk.GetArrayFromImage(pred_image) > 0).astype(np.uint8)
        else:
            logger.warning("TotalSegmentator lung_nodules output missing")
            mask = np.zeros(ref_shape, dtype=np.uint8)

    return mask, time.time() - start
