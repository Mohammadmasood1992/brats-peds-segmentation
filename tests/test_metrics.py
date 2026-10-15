"""BraTS metric conventions."""

import numpy as np

from brats_peds_seg.config import HD95_EMPTY
from brats_peds_seg.metrics import case_metrics, hd95


def test_perfect_prediction():
    reference = np.zeros((20, 20, 20), np.uint8)
    reference[5:10, 5:10, 5:10] = 2
    reference[6:8, 6:8, 6:8] = 1
    row = case_metrics(reference, reference)
    assert row["dice_NET"] == 1.0 and row["dice_ET"] == 1.0 and row["dice_WT"] == 1.0
    assert row["hd95_WT"] == 0.0


def test_empty_conventions():
    reference = np.zeros((10, 10, 10), np.uint8)
    reference[2:5, 2:5, 2:5] = 2
    row = case_metrics(reference, reference)
    assert row["dice_CC"] == 1.0 and row["hd95_CC"] == 0.0 and not row["present_CC"]  # both empty
    row = case_metrics(np.zeros_like(reference), reference)
    assert row["dice_NET"] == 0.0 and row["hd95_NET"] == HD95_EMPTY  # missed structure


def test_hd95_of_shifted_cube():
    a = np.zeros((30, 30, 30), bool)
    b = np.zeros_like(a)
    a[10:20, 10:20, 10:20] = True
    b[12:22, 10:20, 10:20] = True
    assert hd95(a, b) == 2.0
