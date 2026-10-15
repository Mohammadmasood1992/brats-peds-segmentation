"""Per-subject segmentation metrics (BraTS conventions) and summaries."""

import numpy as np
import pandas as pd
from scipy import ndimage

from .config import HD95_EMPTY, LABEL_STRUCTURES, REGION_STRUCTURES, STRUCTURES


def hd95(prediction, reference):
    """Symmetric 95th-percentile Hausdorff distance (voxel units = mm for 1 mm data) between two non-empty masks."""
    union = np.argwhere(prediction | reference)
    box = tuple(slice(max(a - 2, 0), b + 3) for a, b in zip(union.min(0), union.max(0)))
    p, r = prediction[box], reference[box]
    p_surface = p & ~ndimage.binary_erosion(p, border_value=0)
    r_surface = r & ~ndimage.binary_erosion(r, border_value=0)
    to_r = ndimage.distance_transform_edt(~r_surface)[p_surface]
    to_p = ndimage.distance_transform_edt(~p_surface)[r_surface]
    return float(max(np.percentile(to_r, 95), np.percentile(to_p, 95)))


def case_metrics(prediction, reference, with_hd95=True, voxel_ml=0.001):
    """Dice, HD95 and volumes for every label and region.

    BraTS conventions: both empty -> Dice 1, HD95 0; exactly one empty -> Dice 0, HD95 373.13.
    """
    row = {}
    for name, labels in STRUCTURES.items():
        p, r = np.isin(prediction, labels), np.isin(reference, labels)
        p_sum, r_sum = int(p.sum()), int(r.sum())
        if p_sum == 0 and r_sum == 0:
            dice, distance = 1.0, 0.0
        elif p_sum == 0 or r_sum == 0:
            dice, distance = 0.0, HD95_EMPTY
        else:
            dice = 2.0 * int((p & r).sum()) / (p_sum + r_sum)
            distance = hd95(p, r) if with_hd95 else np.nan
        row.update({f"dice_{name}": dice, f"hd95_{name}": distance, f"present_{name}": r_sum > 0,
                    f"volume_ml_ref_{name}": r_sum * voxel_ml, f"volume_ml_pred_{name}": p_sum * voxel_ml})
    row["mean_label_dice"] = float(np.mean([row[f"dice_{s}"] for s in LABEL_STRUCTURES]))
    row["mean_region_dice"] = float(np.mean([row[f"dice_{s}"] for s in REGION_STRUCTURES + ["ET"]]))
    return row


def summarize_metrics(metrics):
    """One row per structure: Dice (all subjects and subjects where the structure is present), HD95, volume error."""
    rows = []
    for name in STRUCTURES:
        present = metrics[metrics[f"present_{name}"]]
        rows.append({
            "structure": name, "subjects": len(metrics), "present": len(present),
            "dice_mean": metrics[f"dice_{name}"].mean(), "dice_median": metrics[f"dice_{name}"].median(),
            "dice_mean_present": present[f"dice_{name}"].mean(),
            "hd95_median": metrics[f"hd95_{name}"].median(), "hd95_mean": metrics[f"hd95_{name}"].mean(),
            "volume_error_ml_median": (metrics[f"volume_ml_pred_{name}"] - metrics[f"volume_ml_ref_{name}"]).median(),
        })
    return pd.DataFrame(rows)
