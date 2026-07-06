"""Resampling, intensity normalisation, and the per-patient preprocessing pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from scripts.preprocessing.dicom_loader import load_ct_series, load_rtstruct_mask


def resample_to_spacing(
    image: sitk.Image,
    target_spacing: tuple[float, float, float] = (1.0, 1.0, 3.0),
    interpolator: int = sitk.sitkBSpline,
) -> sitk.Image:
    """Resample a SimpleITK image to ``target_spacing``.

    Parameters
    ----------
    image:
        Image to resample (CT volume or mask).
    target_spacing:
        Desired (x, y, z) voxel spacing in mm.
    interpolator:
        SimpleITK interpolator. Use ``sitk.sitkBSpline`` for intensity
        volumes (CT) and ``sitk.sitkNearestNeighbor`` for label/mask volumes,
        so the output stays binary.

    Returns
    -------
    sitk.Image
        The resampled image, with the same origin and direction as ``image``.
    """
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    new_size = [
        int(round(osz * ospc / tspc))
        for osz, ospc, tspc in zip(original_size, original_spacing, target_spacing)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0)
    resampler.SetInterpolator(interpolator)

    return resampler.Execute(image)


def clip_and_normalise(
    volume: np.ndarray,
    hu_min: float = -1000.0,
    hu_max: float = 400.0,
) -> np.ndarray:
    """Clip a CT volume to an HU window and apply per-volume z-score normalisation.

    Parameters
    ----------
    volume:
        CT volume in Hounsfield units.
    hu_min, hu_max:
        Lower and upper bounds of the HU window.

    Returns
    -------
    np.ndarray
        The clipped and normalised volume as float32, with approximately
        zero mean and unit standard deviation.
    """
    clipped = np.clip(volume, hu_min, hu_max).astype(np.float32)
    return (clipped - clipped.mean()) / clipped.std()


def preprocess_patient(
    patient_id: str,
    raw_dir: Path,
    output_dir: Path,
    target_spacing: tuple[float, float, float] = (1.0, 1.0, 3.0),
    hu_min: float = -1000.0,
    hu_max: float = 400.0,
) -> dict:
    """Run the full CT preprocessing pipeline for a single patient.

    Loads the raw CT and its RTSTRUCT GTV-1 mask, resamples both to
    ``target_spacing`` (B-spline for the CT, nearest-neighbour for the mask),
    clips and normalises the CT intensities, and saves the results to
    ``output_dir`` as ``<patient_id>_ct.npy``, ``<patient_id>_mask.npy`` and
    ``<patient_id>_meta.json``.

    Parameters
    ----------
    patient_id:
        Patient identifier, e.g. ``"LUNG1-001"``.
    raw_dir:
        Root directory containing per-patient DICOM folders
        (``raw_dir / patient_id``).
    output_dir:
        Directory to write the processed outputs to.
    target_spacing:
        Desired (x, y, z) voxel spacing in mm.
    hu_min, hu_max:
        HU window for clipping.

    Returns
    -------
    dict
        Metadata describing the preprocessing run.
    """
    patient_dir = Path(raw_dir) / patient_id
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ct_image = load_ct_series(patient_dir)
    mask_array = load_rtstruct_mask(patient_dir, ct_image)

    original_spacing = ct_image.GetSpacing()
    original_shape = sitk.GetArrayFromImage(ct_image).shape  # (Z, Y, X)

    mask_image = sitk.GetImageFromArray(mask_array)
    mask_image.CopyInformation(ct_image)

    ct_resampled = resample_to_spacing(ct_image, target_spacing, interpolator=sitk.sitkBSpline)
    mask_resampled = resample_to_spacing(mask_image, target_spacing, interpolator=sitk.sitkNearestNeighbor)

    ct_array = sitk.GetArrayFromImage(ct_resampled).astype(np.float32)
    mask_resampled_array = (sitk.GetArrayFromImage(mask_resampled) > 0).astype(np.uint8)

    mean_before_clip = float(ct_array.mean())
    ct_normalised = clip_and_normalise(ct_array, hu_min=hu_min, hu_max=hu_max)

    np.save(output_dir / f"{patient_id}_ct.npy", ct_normalised)
    np.save(output_dir / f"{patient_id}_mask.npy", mask_resampled_array)

    metadata = {
        "patient_id": patient_id,
        "original_spacing": list(original_spacing),
        "original_shape": list(original_shape),
        "target_spacing": list(target_spacing),
        "resampled_shape": list(ct_array.shape),
        "hu_min": hu_min,
        "hu_max": hu_max,
        "mean_hu_before_clip": mean_before_clip,
        "mean_after_normalise": float(ct_normalised.mean()),
        "std_after_normalise": float(ct_normalised.std()),
        "mask_voxel_count_original": int(mask_array.sum()),
        "mask_voxel_count_resampled": int(mask_resampled_array.sum()),
    }

    with open(output_dir / f"{patient_id}_meta.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata
