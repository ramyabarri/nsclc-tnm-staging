"""
Convert NSCLC-Radiomics DICOM data to nnU-Net v2 dataset format.

Input: any directory tree containing the downloaded DICOMs. I don't assume a
particular folder layout (tcia_utils names its output differently across versions);
instead I walk the tree, read each series' DICOM header, and group the series by
PatientID and Modality, so the converter copes with whatever nesting the download
happened to produce.

Output:
    out_dir/Dataset001_NSCLCRadiomics/
        imagesTr/{case_id}_0000.nii.gz   <- CT in raw HU (nnU-Net normalises itself)
        labelsTr/{case_id}.nii.gz        <- binary GTV mask (1 = tumour)
        dataset.json

Usage:
    python prepare_nnunet_dataset.py \
        --raw-dir /workspace/data/raw \
        --out-dir /workspace/nnunet/raw \
        [--workers 4]
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pydicom
import SimpleITK as sitk
from rt_utils import RTStructBuilder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATASET_ID = "001"
DATASET_NAME = "NSCLCRadiomics"
DATASET_FOLDER = f"Dataset{DATASET_ID}_{DATASET_NAME}"


# DICOM discovery: group series by PatientID and Modality from the headers,
# independent of the on-disk folder layout.

def index_series(raw_dir: Path) -> dict[str, dict[str, list[Path]]]:
    """Walk raw_dir and return {patient_id: {modality: [series_dir, ...]}}.

    A "series directory" is any directory that directly contains .dcm files.
    PatientID and Modality are read from the first slice's header.
    """
    index: dict[str, dict[str, list[Path]]] = {}
    seen: set[Path] = set()

    for dcm in raw_dir.rglob("*.dcm"):
        series_dir = dcm.parent
        if series_dir in seen:
            continue
        seen.add(series_dir)
        try:
            ds = pydicom.dcmread(str(dcm), stop_before_pixels=True)
        except Exception:
            continue
        pid = str(getattr(ds, "PatientID", "")).strip()
        modality = str(getattr(ds, "Modality", "")).strip()
        if not pid or not modality:
            continue
        index.setdefault(pid, {}).setdefault(modality, []).append(series_dir)

    return index


def _pick_ct_dir(ct_dirs: list[Path]) -> Path:
    """Choose the CT series with the most slices (the axial acquisition)."""
    return max(ct_dirs, key=lambda d: len(list(d.glob("*.dcm"))))


def load_ct_series(ct_dir: Path) -> sitk.Image:
    reader = sitk.ImageSeriesReader()
    ids = reader.GetGDCMSeriesIDs(str(ct_dir))
    if not ids:
        raise FileNotFoundError(f"No DICOM series in {ct_dir}")
    reader.SetFileNames(reader.GetGDCMSeriesFileNames(str(ct_dir), ids[0]))
    return reader.Execute()


def _normalize_roi(name: str) -> str:
    """Uppercase and strip spaces/dashes/underscores: 'GTV-1' / 'gtv 1' -> 'GTV1'."""
    return re.sub(r"[\s_\-]+", "", name).upper()


def _gtv_roi_names(roi_names: list[str]) -> list[str]:
    """All ROI names whose normalized form starts with 'GTV'."""
    return [n for n in roi_names if _normalize_roi(n).startswith("GTV")]


def load_gtv_mask(ct_dir: Path, rt_file: Path, reference_ct: sitk.Image):
    """Load the primary GTV mask, returned as (mask (Z,Y,X) uint8, chosen_roi_label).

    Consistency policy (NSCLC-Radiomics): the primary gross tumour volume is
    named 'GTV-1'. To keep the ground truth meaning the SAME across patients we:
      1. select 'GTV-1' (normalized, so 'GTV-1'/'gtv 1'/'GTV1' all match); else
      2. fall back to the single largest GTV-prefixed ROI (most likely the
         primary tumour) so we never grab an arbitrary/secondary structure.
    The chosen label is returned so the caller can audit consistency across the
    cohort.
    """
    rtstruct = RTStructBuilder.create_from(
        dicom_series_path=str(ct_dir),
        rt_struct_path=str(rt_file),
    )
    roi_names = rtstruct.get_roi_names()
    gtv_names = _gtv_roi_names(roi_names)
    if not gtv_names:
        raise ValueError(f"No GTV ROI (found: {roi_names})")

    # 1. Prefer the documented primary, GTV-1.
    primary = next((n for n in gtv_names if _normalize_roi(n) == "GTV1"), None)

    if primary is not None:
        chosen = primary
        mask = rtstruct.get_roi_mask_by_name(chosen)
        label = "GTV-1"
    else:
        # 2. Fall back to the largest GTV ROI (deterministic, primary-tumour-like).
        best_name, best_mask, best_vox = None, None, -1
        for n in gtv_names:
            try:
                m = rtstruct.get_roi_mask_by_name(n)
            except Exception:
                continue
            vox = int(np.asarray(m, dtype=bool).sum())
            if vox > best_vox:
                best_name, best_mask, best_vox = n, m, vox
        if best_mask is None:
            raise ValueError(f"GTV ROIs present but none loadable: {gtv_names}")
        chosen, mask = best_name, best_mask
        label = f"largest-fallback ({chosen})"

    mask = np.transpose(mask, (2, 0, 1)).astype(np.uint8)  # (Y,X,Z) -> (Z,Y,X)
    expected = sitk.GetArrayFromImage(reference_ct).shape
    if mask.shape != expected:
        raise ValueError(f"Mask shape {mask.shape} != CT shape {expected}")
    if int(mask.sum()) == 0:
        raise ValueError(f"Chosen GTV ROI '{chosen}' is empty")
    return mask, label


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def convert_patient(
    patient_id: str,
    ct_dir: Path,
    rt_dir: Path,
    images_dir: Path,
    labels_dir: Path,
) -> tuple[str | None, str | None]:
    """Return (error_or_None, chosen_roi_label_or_None)."""
    case_id = patient_id.replace("-", "_")
    img_path = images_dir / f"{case_id}_0000.nii.gz"
    lbl_path = labels_dir / f"{case_id}.nii.gz"

    if img_path.exists() and lbl_path.exists():
        return None, "already-done"

    try:
        rt_files = sorted(rt_dir.glob("*.dcm"))
        if not rt_files:
            raise FileNotFoundError(f"No RTSTRUCT .dcm in {rt_dir}")

        ct_image = load_ct_series(ct_dir)
        mask_arr, roi_label = load_gtv_mask(ct_dir, rt_files[0], ct_image)

        # Save CT in raw HU; nnU-Net applies its own normalisation
        sitk.WriteImage(ct_image, str(img_path))

        mask_image = sitk.GetImageFromArray(mask_arr)
        mask_image.CopyInformation(ct_image)
        sitk.WriteImage(mask_image, str(lbl_path))
        return None, roi_label
    except Exception as exc:
        return str(exc), None


def _worker(args):
    err, label = convert_patient(*args)
    return args[0], err, label


def build_dataset(
    raw_dir: Path,
    out_dir: Path,
    workers: int = 4,
) -> None:
    dataset_dir = out_dir / DATASET_FOLDER
    images_dir = dataset_dir / "imagesTr"
    labels_dir = dataset_dir / "labelsTr"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Indexing DICOM series under %s ...", raw_dir)
    index = index_series(raw_dir)
    if not index:
        raise RuntimeError(f"No DICOM series found under {raw_dir}")

    jobs = []
    skipped: list[tuple[str, str]] = []
    for pid in sorted(index):
        modalities = index[pid]
        if "CT" not in modalities:
            skipped.append((pid, "no CT series"))
            continue
        if "RTSTRUCT" not in modalities:
            skipped.append((pid, "no RTSTRUCT series"))
            continue
        ct_dir = _pick_ct_dir(modalities["CT"])
        rt_dir = modalities["RTSTRUCT"][0]
        jobs.append((pid, ct_dir, rt_dir, images_dir, labels_dir))

    logger.info("Found %d patients with CT+RTSTRUCT (%d skipped)",
                len(jobs), len(skipped))
    for pid, reason in skipped:
        logger.warning("SKIP %s: %s", pid, reason)

    failed: list[tuple[str, str]] = list(skipped)
    roi_choices: Counter = Counter()

    with mp.Pool(workers) as pool:
        for i, (pid, err, label) in enumerate(pool.imap_unordered(_worker, jobs), 1):
            status = "OK" if err is None else f"FAILED: {err}"
            logger.info("[%d/%d] %s: %s%s", i, len(jobs), pid, status,
                        f" [{label}]" if label else "")
            if err:
                failed.append((pid, err))
            else:
                # Bucket the fallback cases together (the ROI name varies).
                roi_choices["GTV-1" if label == "GTV-1"
                            else "already-done" if label == "already-done"
                            else "largest-fallback"] += 1

    succeeded = len(jobs) - len([f for f in failed if f not in skipped])
    logger.info("Converted %d patients (%d failed/skipped)", succeeded, len(failed))

    # Label-consistency audit: how often we got the canonical GTV-1 vs a fallback.
    logger.info("GTV ROI selection across cohort: %s", dict(roi_choices))
    n_fallback = roi_choices.get("largest-fallback", 0)
    if n_fallback:
        logger.warning(
            "%d patient(s) had no 'GTV-1' ROI and used the largest-GTV fallback; "
            "inspect these for label consistency.", n_fallback)

    if failed:
        fail_path = dataset_dir / "_conversion_failures.json"
        with open(fail_path, "w") as f:
            json.dump(dict(failed), f, indent=2)
        logger.warning("%d failures logged to %s", len(failed), fail_path)

    # numTraining must match the actual image/label pairs on disk.
    num_training = len(list(labels_dir.glob("*.nii.gz")))

    # Write dataset.json (nnU-Net v2 minimal format)
    dataset_json = {
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "GTV": 1},
        "numTraining": num_training,
        "file_ending": ".nii.gz",
        "name": DATASET_NAME,
        "description": "NSCLC-Radiomics GTV segmentation (TCIA)",
        "reference": "https://wiki.cancerimagingarchive.net/display/Public/NSCLC-Radiomics",
    }
    with open(dataset_dir / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=2)
    logger.info("Wrote dataset.json (numTraining=%d) to %s", num_training, dataset_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Convert NSCLC-Radiomics DICOMs to nnU-Net v2 format."
    )
    parser.add_argument(
        "--raw-dir", type=Path, required=True,
        help="Directory tree containing the downloaded DICOMs (any layout)"
    )
    parser.add_argument(
        "--out-dir", type=Path, required=True,
        help="nnU-Net raw data directory (the nnUNet_raw base)"
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Parallel worker processes (default: 4)"
    )
    args = parser.parse_args()
    build_dataset(args.raw_dir, args.out_dir, args.workers)


if __name__ == "__main__":
    main()
