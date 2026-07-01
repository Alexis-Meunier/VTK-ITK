import itk
from typing import Any
import numpy as np


def compute_change_metrics(mask1, mask2_registered, reference_image) -> dict[str, Any]:
    """Compute volume and overlap metrics between two binary masks in
    the same physical space (after registration).

    Parameters
    ----------
    mask1, mask2_registered : itk.Image
    reference_image : used to read voxel spacing

    Returns
    -------
    dict with volumes (mm3 and cm3), absolute/relative change, and Dice score.
    """
    a1 = itk.GetArrayFromImage(mask1).astype(bool)
    a2 = itk.GetArrayFromImage(mask2_registered).astype(bool)

    spacing = reference_image.GetSpacing()
    voxel_volume = spacing[0] * spacing[1] * spacing[2]

    vol1 = a1.sum() * voxel_volume
    vol2 = a2.sum() * voxel_volume

    intersection = np.logical_and(a1, a2).sum()
    union = np.logical_or(a1, a2).sum()
    sum12 = (a1.sum() + a2.sum()) 
    dice = 2 * intersection / sum12 if sum12 > 0 else 0.0

    abs_change = vol2 - vol1
    rel_change = (abs_change / vol1 * 100) if vol1 > 0 else float("nan")

    iou = intersection / union if union > 0 else 0.0

    growth = np.logical_and(a2, np.logical_not(a1)).sum() * voxel_volume
    regressed = np.logical_and(a1, np.logical_not(a2)).sum() * voxel_volume

    if a1.any() and a2.any():
        c1 = np.argwhere(a1).mean(axis=0)         
        c2 = np.argwhere(a2).mean(axis=0)
        spacing_zyx = np.array(spacing)[::-1]
        centroid_shift = float(np.linalg.norm((c2 - c1) * spacing_zyx))
    else:
        centroid_shift = float("nan")

    return {
        "volume_scan1_mm3": float(vol1),
        "volume_scan2_mm3": float(vol2),
        "volume_scan1_cm3": float(vol1) / 1000,
        "volume_scan2_cm3": float(vol2) / 1000,
        "absolute_change_mm3": float(abs_change),
        "relative_change_percent": float(rel_change),
        "dice_overlap": float(dice),
        "intersection_voxels": int(intersection),
        "union_voxels": int(union),
        "mask1_array": a1,
        "mask2_array": a2,
        "iou": float(iou),
        "growth_mm3": float(growth),
        "regressed_mm3": float(regressed),
        "centroid_shift_mm": centroid_shift,
    }


def print_summary(metrics: dict[str, Any]) -> None:
    print("\n--- Tumor change analysis ---")
    print(f"{'Volume scan 1': <25}: {metrics['volume_scan1_cm3']:.2f} cm3")
    print(f"{'Volume scan 2 (registered)': <25}: {metrics['volume_scan2_cm3']:.2f} cm3")
    sign = "+" if metrics["absolute_change_mm3"] >= 0 else ""
    print(f"{'Absolute change': <25}: {sign}{metrics['absolute_change_mm3']/1000:.2f} cm3")
    print(f"{'Relative change': <25}: {sign}{metrics['relative_change_percent']:.1f} %")
    print(f"{'Dice overlap (spatial)': <25}: {metrics['dice_overlap']:.3f}")
    print(f"{'IoU': <25}: {metrics['iou']:.3f}")
    print(f"{'New growth': <25}: +{metrics['growth_mm3']/1000:.2f} cm3")
    print(f"{'Regressed tissue': <25}: -{metrics['regressed_mm3']/1000:.2f} cm3")
    print(f"{'Centroid shift': <25}: {metrics['centroid_shift_mm']:.2f} mm")
