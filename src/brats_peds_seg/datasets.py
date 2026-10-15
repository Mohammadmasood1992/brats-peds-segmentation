"""Training datasets and augmentation.

- 3D: each item loads one cached volume and returns `samples_per_volume` patches whose centres are drawn
  per class with `label_crop_ratios` (rare tissues are oversampled).
- 2D / 2.5D: each item is one slice; with probability `tumor_slice_fraction` the slice contains a tumor
  class chosen uniformly among the classes present in that subject, otherwise any brain slice.
"""

import math

import numpy as np
import torch
from monai import transforms as mt
from monai.data import DataLoader

from .config import N_CLASSES
from .preprocessing import load_meta


def build_train_transforms(cfg):
    dims = cfg["spatial_dims"]
    patch = tuple(cfg["patch_size_3d"] if dims == 3 else cfg["patch_size_2d"])
    keys = ["image", "label"]
    t = [mt.SpatialPadd(keys, spatial_size=patch, mode="constant")]
    if dims == 3:
        t.append(mt.RandCropByLabelClassesd(keys, label_key="label", spatial_size=patch, ratios=list(cfg["label_crop_ratios"]),
                                            num_classes=N_CLASSES, num_samples=cfg["samples_per_volume"], warn=False))
    else:
        t.append(mt.RandSpatialCropd(keys, roi_size=patch, random_size=False))
    aug = cfg["augmentation"]
    if aug["enabled"]:
        rotate = math.radians(aug["rotate_deg"])
        t.append(mt.RandAffined(keys, prob=aug["affine_prob"], rotate_range=(rotate,) * (3 if dims == 3 else 1),
                                scale_range=(aug["scale"],) * dims, mode=("bilinear", "nearest"), padding_mode="zeros"))
        for axis in (aug["flip_axes"] if aug["flip_axes"] is not None else range(dims)):
            t.append(mt.RandFlipd(keys, prob=aug["flip_prob"], spatial_axis=axis))
        t += [
            mt.RandGaussianNoised("image", prob=aug["noise_prob"], std=aug["noise_std"]),
            mt.RandScaleIntensityd("image", factors=aug["intensity_scale"], prob=aug["intensity_prob"]),
            mt.RandShiftIntensityd("image", offsets=aug["intensity_shift"], prob=aug["intensity_prob"]),
            mt.RandAdjustContrastd("image", prob=aug["gamma_prob"], gamma=tuple(aug["gamma_range"])),
        ]
    t.append(mt.EnsureTyped(keys, dtype=torch.float32, track_meta=False))
    return mt.Compose(t)


def extract_slices(volume, axis, index, context):
    """volume (C, X, Y, Z) -> (C * (2 * context + 1), H, W) around slice `index` along spatial `axis`."""
    n = volume.shape[axis + 1]
    idx = np.clip(np.arange(index - context, index + context + 1), 0, n - 1)
    stack = np.moveaxis(np.take(volume, idx, axis=axis + 1), axis + 1, 1)
    return stack.reshape(-1, *stack.shape[2:])


class TrainDataset3D(torch.utils.data.Dataset):
    def __init__(self, cases, cache_dir, cfg, length):
        self.cases, self.cache_dir, self.cfg, self.length = cases, cache_dir, cfg, length
        self.transform = build_train_transforms(cfg)

    def __len__(self):
        return self.length

    def __getitem__(self, _):
        case = self.cases[np.random.default_rng().integers(len(self.cases))]
        image = np.load(self.cache_dir / case / "image.npy")[self.cfg["channel_indices"]].astype(np.float32)
        label = np.load(self.cache_dir / case / "label.npy")[None].astype(np.float32)
        return self.transform({"image": image, "label": label})


class TrainDataset2D(torch.utils.data.Dataset):
    def __init__(self, cases, cache_dir, cfg, length):
        self.cases, self.cache_dir, self.cfg, self.length = cases, cache_dir, cfg, length
        self.axis = cfg["slice_axis"]
        self.slices = {c: load_meta(cache_dir, c)["slices"][self.axis] for c in cases}
        self.transform = build_train_transforms(cfg)

    def __len__(self):
        return self.length

    def __getitem__(self, _):
        rng = np.random.default_rng()
        case = self.cases[rng.integers(len(self.cases))]
        slices = self.slices[case]
        present = [k for k, s in slices["classes"].items() if s]
        if present and rng.random() < self.cfg["tumor_slice_fraction"]:
            candidates = slices["classes"][present[rng.integers(len(present))]]
        else:
            candidates = slices["brain"]
        index = candidates[rng.integers(len(candidates))]
        volume = np.load(self.cache_dir / case / "image.npy", mmap_mode="r")[self.cfg["channel_indices"]]
        image = extract_slices(volume, self.axis, index, self.cfg["context_slices"]).astype(np.float32)
        label = np.take(np.load(self.cache_dir / case / "label.npy", mmap_mode="r"), index, axis=self.axis)[None].astype(np.float32)
        return self.transform({"image": image, "label": label})


def build_train_loader(cfg, cases, cache_dir):
    if cfg["spatial_dims"] == 3:
        volumes_per_step = max(1, cfg["batch_size_3d"] // cfg["samples_per_volume"])
        dataset = TrainDataset3D(cases, cache_dir, cfg, cfg["iterations_per_epoch"] * volumes_per_step)
        batch_size = volumes_per_step
    else:
        dataset = TrainDataset2D(cases, cache_dir, cfg, cfg["iterations_per_epoch"] * cfg["batch_size_2d"])
        batch_size = cfg["batch_size_2d"]
    return DataLoader(dataset, batch_size=batch_size, num_workers=cfg["num_workers"], pin_memory=True, drop_last=True,
                      persistent_workers=cfg["num_workers"] > 0)
