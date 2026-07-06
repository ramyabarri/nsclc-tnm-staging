# Segmentation evaluation metrics: Dice, HD95, volume error

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt


def dice_score(pred, gt):
    """Compute Dice similarity coefficient between two binary masks."""
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    pred_sum = pred.sum()
    gt_sum = gt.sum()

    if pred_sum == 0 and gt_sum == 0:
        return 1.0
    if pred_sum == 0 or gt_sum == 0:
        return 0.0

    intersection = np.logical_and(pred, gt).sum()
    return float(2.0 * intersection / (pred_sum + gt_sum))


def hd95(pred, gt, spacing=(3.0, 1.0, 1.0)):
    """95th-percentile Hausdorff distance in mm. spacing is (z, y, x).
    Returns nan if either mask is empty.
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    if pred.sum() == 0 or gt.sum() == 0:
        return float("nan")

    pred_surface = pred & ~binary_erosion(pred)
    gt_surface = gt & ~binary_erosion(gt)

    dist_to_gt = distance_transform_edt(~gt, sampling=spacing)
    dist_to_pred = distance_transform_edt(~pred, sampling=spacing)

    distances = np.concatenate([dist_to_gt[pred_surface], dist_to_pred[gt_surface]])
    return float(np.percentile(distances, 95))


def volume_error_cm3(pred, gt, spacing=(3.0, 1.0, 1.0)):
    """Absolute volume difference in cm^3 between pred and gt masks."""
    voxel_vol = spacing[0] * spacing[1] * spacing[2]
    pred_voxels = int(np.asarray(pred, dtype=bool).sum())
    gt_voxels = int(np.asarray(gt, dtype=bool).sum())
    return abs(pred_voxels - gt_voxels) * voxel_vol / 1000.0


def compute_all_metrics(pred, gt, spacing):
    """Return a dict with dice, hd95_mm, volume_error_cm3, gt_volume_cm3, pred_volume_cm3."""
    voxel_vol_cm3 = (spacing[0] * spacing[1] * spacing[2]) / 1000.0
    pred_voxels = int(np.asarray(pred, dtype=bool).sum())
    gt_voxels = int(np.asarray(gt, dtype=bool).sum())

    return {
        "dice": dice_score(pred, gt),
        "hd95_mm": hd95(pred, gt, spacing=spacing),
        "volume_error_cm3": volume_error_cm3(pred, gt, spacing=spacing),
        "gt_volume_cm3": gt_voxels * voxel_vol_cm3,
        "pred_volume_cm3": pred_voxels * voxel_vol_cm3,
    }
