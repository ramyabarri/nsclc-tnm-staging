"""Load raw CT volumes and RTSTRUCT segmentation masks from NSCLC-Radiomics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pydicom
import SimpleITK as sitk
from rt_utils import RTStructBuilder


def _find_series_dirs(patient_dir: Path) -> dict[str, Path]:
    """Map each DICOM Modality found under ``patient_dir`` to its series directory.

    NSCLC-Radiomics layout is ``<patient_dir>/<study>/<series>/*.dcm``, with
    each series directory containing a single modality (CT, RTSTRUCT, ...).
    """
    series_by_modality: dict[str, Path] = {}
    for series_dir in sorted(patient_dir.glob("*/*")):
        if not series_dir.is_dir():
            continue
        dcm_files = sorted(series_dir.glob("*.dcm"))
        if not dcm_files:
            continue
        ds = pydicom.dcmread(dcm_files[0], stop_before_pixels=True)
        series_by_modality[ds.Modality] = series_dir
    return series_by_modality


def load_ct_series(patient_dir: Path) -> sitk.Image:
    """Load a patient's CT volume as a SimpleITK image.

    Parameters
    ----------
    patient_dir:
        Path to a patient folder, e.g. ``data/raw/NSCLC-Radiomics/LUNG1-001``.

    Returns
    -------
    sitk.Image
        The CT volume. ``sitk.GetArrayFromImage`` returns it with shape
        (Z, Y, X).
    """
    patient_dir = Path(patient_dir)
    series_by_modality = _find_series_dirs(patient_dir)
    if "CT" not in series_by_modality:
        raise FileNotFoundError(f"No CT series found under {patient_dir}")
    ct_dir = series_by_modality["CT"]

    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(str(ct_dir))
    if not series_ids:
        raise FileNotFoundError(f"No DICOM series found in {ct_dir}")
    file_names = reader.GetGDCMSeriesFileNames(str(ct_dir), series_ids[0])
    reader.SetFileNames(file_names)
    return reader.Execute()


def load_rtstruct_mask(patient_dir: Path, reference_ct: sitk.Image) -> np.ndarray:
    """Load the GTV-1 ROI from a patient's RTSTRUCT as a binary mask.

    Parameters
    ----------
    patient_dir:
        Path to a patient folder, e.g. ``data/raw/NSCLC-Radiomics/LUNG1-001``.
    reference_ct:
        The CT volume returned by :func:`load_ct_series`, which the RTSTRUCT
        contours are aligned to.

    Returns
    -------
    np.ndarray
        Binary (uint8) mask with shape (Z, Y, X), matching
        ``sitk.GetArrayFromImage(reference_ct).shape``.
    """
    patient_dir = Path(patient_dir)
    series_by_modality = _find_series_dirs(patient_dir)
    if "RTSTRUCT" not in series_by_modality:
        raise FileNotFoundError(f"No RTSTRUCT series found under {patient_dir}")
    if "CT" not in series_by_modality:
        raise FileNotFoundError(f"No CT series found under {patient_dir}")

    rtstruct_files = sorted(series_by_modality["RTSTRUCT"].glob("*.dcm"))
    rtstruct = RTStructBuilder.create_from(
        dicom_series_path=str(series_by_modality["CT"]),
        rt_struct_path=str(rtstruct_files[0]),
    )

    roi_names = rtstruct.get_roi_names()
    if "GTV-1" in roi_names:
        gtv_name = "GTV-1"
    else:
        gtv_name = next((name for name in roi_names if name.upper().startswith("GTV")), None)
    if gtv_name is None:
        raise ValueError(f"No GTV ROI found for {patient_dir} (available ROIs: {roi_names})")

    mask = rtstruct.get_roi_mask_by_name(gtv_name)  # shape (Y, X, Z)
    mask = np.transpose(mask, (2, 0, 1))  # -> (Z, Y, X), matching SimpleITK arrays

    expected_shape = sitk.GetArrayFromImage(reference_ct).shape
    if mask.shape != expected_shape:
        raise ValueError(
            f"RTSTRUCT mask shape {mask.shape} does not match CT shape "
            f"{expected_shape} for {patient_dir}"
        )

    return mask.astype(np.uint8)
