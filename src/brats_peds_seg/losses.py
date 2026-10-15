"""Segmentation loss: region term (Dice / Tversky) + pixel term (cross-entropy / focal)."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceLoss, TverskyLoss

from .config import N_CLASSES
from .preprocessing import load_meta


class SegmentationLoss(nn.Module):
    """Region + pixel loss, summed over deep-supervision outputs with weights 1, 1/2, 1/4, ..."""

    def __init__(self, cfg, class_weights=None):
        super().__init__()
        common = dict(to_onehot_y=True, softmax=True, include_background=cfg["dice_include_background"], batch=cfg["dice_batch"])
        self.region = {None: None,
                       "dice": lambda: DiceLoss(**common),
                       "tversky": lambda: TverskyLoss(alpha=cfg["tversky_alpha"], beta=cfg["tversky_beta"], **common)}[cfg["loss_region"]]
        self.region = self.region() if self.region else None
        self.pixel, self.gamma = cfg["loss_pixel"], cfg["focal_gamma"]
        self.register_buffer("class_weights", None if class_weights is None else torch.as_tensor(class_weights, dtype=torch.float32))

    def single(self, logits, target):
        loss = 0.0
        if self.region is not None:
            loss = loss + self.region(logits, target)
        if self.pixel is not None:
            labels = target[:, 0].long()
            ce = F.cross_entropy(logits, labels, weight=self.class_weights, reduction="none")
            if self.pixel == "focal":
                pt = torch.exp(-F.cross_entropy(logits, labels, reduction="none"))
                ce = (1 - pt) ** self.gamma * ce
            loss = loss + ce.mean()
        return loss

    def forward(self, outputs, target):
        outputs = outputs if isinstance(outputs, list) else [outputs]
        weights = [0.5 ** i for i in range(len(outputs))]
        total = 0.0
        for weight, logits in zip(weights, outputs):
            t = target if logits.shape[2:] == target.shape[2:] else F.interpolate(target, size=logits.shape[2:], mode="nearest")
            total = total + weight * self.single(logits.float(), t)
        return total / sum(weights)


def compute_class_weights(cfg, cases, cache_dir):
    """None, the configured list, or inverse-sqrt-frequency weights from the training subjects of a fold."""
    if cfg["class_weights"] is None:
        return None
    if cfg["class_weights"] != "auto":
        return list(cfg["class_weights"])
    counts = np.zeros(N_CLASSES)
    for case in cases:
        counts += np.array([load_meta(cache_dir, case)["label_counts"][str(k)] for k in range(N_CLASSES)])
    weights = (counts.sum() / np.maximum(counts, 1)) ** 0.5
    return (weights / weights.mean()).tolist()
