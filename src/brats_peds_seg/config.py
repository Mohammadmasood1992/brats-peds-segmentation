"""Experiment configuration: defaults, YAML overrides and derived settings.

Every experiment is a flat dictionary. `DEFAULT_CONFIG` documents every option; experiment files in
`configs/*.yaml` only list the options they change. The resolved configuration is saved as
`config.json` in the run folder and is all that prediction needs to rebuild a model.
"""

import copy
from pathlib import Path

import yaml

ALL_MODALITIES = ["t1n", "t1c", "t2w", "t2f"]
LABEL_NAMES = {1: "ET", 2: "NET", 3: "CC", 4: "ED"}
STRUCTURES = {"ET": [1], "NET": [2], "CC": [3], "ED": [4], "TC": [1, 2, 3], "WT": [1, 2, 3, 4]}
LABEL_STRUCTURES = ["ET", "NET", "CC", "ED"]
REGION_STRUCTURES = ["TC", "WT"]
N_CLASSES = 5
HD95_EMPTY = 373.13  # BraTS convention when exactly one of prediction / reference is empty

DEFAULT_CONFIG = {
    "experiment_name": None,            # None -> generated from the main options
    "smoke_test": False,                # True -> tiny model/data, 2 epochs, fold 0 only (end-to-end check)

    # ---- paths (relative to the working directory, normally the repository root) ----
    "aligned_root": "data/BraTS-PEDs-v1_rigid_aligned",          # images, brain masks and labels
    "histmatched_root": "data/BraTS-PEDs-v1_rigid_aligned_histmatched",
    "metadata_file": "data/BraTS-PEDs_metadata.tsv",
    "cache_root": "cache",
    "runs_root": "runs",

    # ---- data ----
    "data_version": "histmatched",      # "histmatched" | "raw"
    "modalities": ["t1n", "t1c", "t2w", "t2f"],
    "normalization": "auto",            # "auto" | "zscore_subject" | "zscore_reference" | "none"
    "brain_mask": True,                 # raw only: zero voxels outside the SynthStrip brain mask
    "crop_margin": 8,                   # voxels kept around the brain (and tumor) bounding box
    "max_cases": None,                  # debug: limit training cases per fold

    # ---- dimensionality / sampling ----
    "dim": "3d",                        # "2d" | "3d"
    "slice_axis": 2,                    # 2D: 0 sagittal, 1 coronal, 2 axial
    "context_slices": 0,                # 2.5D: neighbouring slices per side added as channels
    "patch_size_3d": [128, 128, 128],
    "patch_size_2d": [192, 192],
    "batch_size_3d": 2,                 # patches per step
    "samples_per_volume": 2,            # 3D patches drawn per loaded volume
    "batch_size_2d": 32,
    "label_crop_ratios": [1, 1, 1, 1, 1],  # 3D patch-centre sampling ratios for classes 0..4 (oversamples rare tissues)
    "tumor_slice_fraction": 0.7,        # 2D: share of slices containing a (randomly chosen present) tumor class

    # ---- model ----
    "base_channels": 32,
    "max_channels": 320,
    "depth": 5,                         # resolution levels; inputs are padded to multiples of 2**(depth-1)
    "norm": "instance",                 # "instance" | "batch" | "group"
    "activation": "leakyrelu",          # "leakyrelu" | "relu" | "gelu"
    "residual": True,                   # residual conv blocks
    "dropout": 0.0,
    "downsample": "strided_conv",       # "strided_conv" | "maxpool"
    "upsample": "transpose",            # "transpose" | "interpolate"
    "block_attention": None,            # None | "se" | "cbam"  (inside every conv block)
    "skip_attention": False,            # attention gates on skip connections (Attention U-Net)
    "bottleneck": "conv",               # "conv" | "transformer" | "aspp"
    "transformer_layers": 4,
    "transformer_heads": 8,
    "transformer_mlp_ratio": 4.0,
    "aspp_rates": [1, 2, 3],
    "deep_supervision": True,
    "deep_supervision_levels": 3,       # number of decoder outputs in the loss (1 = off)

    # ---- loss ----
    "loss_region": "dice",              # "dice" | "tversky" | None
    "loss_pixel": "ce",                 # "ce" | "focal" | None
    "dice_include_background": False,
    "dice_batch": True,                 # Dice over the whole batch (stabler for rare classes)
    "tversky_alpha": 0.3,
    "tversky_beta": 0.7,
    "focal_gamma": 2.0,
    "class_weights": None,              # None | "auto" (inverse sqrt frequency) | list of 5 weights (pixel loss)

    # ---- optimisation ----
    "folds": [0, 1, 2, 3, 4],
    "n_folds": 5,
    "epochs": 200,
    "iterations_per_epoch": 250,        # one "epoch" = this many optimizer steps on random patches/slices
    "optimizer": "adamw",               # "adamw" | "sgd"
    "lr": 1e-3,                         # sgd: 1e-2 is typical
    "weight_decay": 1e-4,
    "scheduler": "cosine",              # "cosine" | "poly" | "constant"
    "warmup_epochs": 5,
    "amp": "bf16",                      # "bf16" | "fp16" | None
    "channels_last": True,              # channels-last memory layout; avoids huge cuDNN workspaces for half-precision 3D convs
    "grad_clip": 12.0,
    "val_every": 10,                    # epochs between validations
    "val_max_cases": None,              # subsample validation cases during training (None = all)
    "early_stopping_patience": None,    # number of validations without improvement (None = off)
    "selection_metric": "mean_label_dice",  # "mean_label_dice" | "mean_region_dice" | "WT" | "TC" | "ET" ...
    "resume": True,                     # continue a fold from last.pt if it exists

    # ---- augmentation ----
    "augmentation": {
        "enabled": True,
        "flip_prob": 0.5,
        "flip_axes": None,              # None = all spatial axes
        "affine_prob": 0.3,
        "rotate_deg": 15.0,
        "scale": 0.15,                  # scale factor in [1 - s, 1 + s]
        "noise_prob": 0.1,
        "noise_std": 0.1,
        "intensity_prob": 0.15,
        "intensity_scale": 0.1,
        "intensity_shift": 0.1,
        "gamma_prob": 0.15,
        "gamma_range": [0.7, 1.5],
    },

    # ---- inference ----
    "sw_overlap": 0.5,                  # 3D sliding-window overlap
    "sw_batch_size": 2,
    "val_tta": False,                   # flip TTA for the final out-of-fold predictions
    "predict_tta": True,                # flip TTA for new-data prediction
    "mask_predictions": False,          # zero predictions outside the brain mask

    # ---- system ----
    "num_workers": 8,
    "seed": 2026,
}

SMOKE_OVERRIDES = {
    "folds": [0], "epochs": 2, "iterations_per_epoch": 3, "val_every": 1, "max_cases": 8, "val_max_cases": 2,
    "base_channels": 8, "max_channels": 64, "patch_size_3d": [64, 64, 64], "patch_size_2d": [128, 128],
    "batch_size_2d": 4, "num_workers": 2, "warmup_epochs": 0, "resume": False,
}


def load_config(path=None, overrides=None):
    """DEFAULT_CONFIG updated with a YAML experiment file and then with `overrides` (nested dicts merge)."""
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    for update in [yaml.safe_load(Path(path).read_text()) if path else None, overrides]:
        for key, value in (update or {}).items():
            if key not in cfg:
                raise KeyError(f"Unknown configuration option: {key}")
            if isinstance(cfg[key], dict) and isinstance(value, dict):
                unknown = set(value) - set(cfg[key])
                if unknown:
                    raise KeyError(f"Unknown {key} options: {sorted(unknown)}")
                cfg[key].update(value)
            else:
                cfg[key] = value
    return cfg


def resolve_config(cfg):
    """Fill derived settings (normalization, experiment name, channel counts) and apply smoke-test overrides."""
    cfg = copy.deepcopy(cfg)
    if cfg["smoke_test"]:
        cfg.update(SMOKE_OVERRIDES)
    if cfg["normalization"] == "auto":
        cfg["normalization"] = "zscore_reference" if cfg["data_version"] == "histmatched" else "zscore_subject"
    if cfg["data_version"] == "histmatched":
        cfg["brain_mask"] = True  # the matched images are brain-masked by construction
    if cfg["experiment_name"] is None:
        parts = [cfg["dim"] + (f"ctx{cfg['context_slices']}" if cfg["dim"] == "2d" and cfg["context_slices"] else ""),
                 cfg["data_version"], cfg["bottleneck"]]
        if cfg["block_attention"]:
            parts.append(cfg["block_attention"])
        if cfg["skip_attention"]:
            parts.append("attgate")
        if sorted(cfg["modalities"]) != sorted(ALL_MODALITIES):
            parts.append("+".join(cfg["modalities"]))
        if cfg["smoke_test"]:
            parts.append("smoke")
        cfg["experiment_name"] = "_".join(parts)
    cfg["spatial_dims"] = 3 if cfg["dim"] == "3d" else 2
    cfg["channel_indices"] = [ALL_MODALITIES.index(m) for m in cfg["modalities"]]
    cfg["in_channels"] = len(cfg["modalities"]) * (2 * cfg["context_slices"] + 1 if cfg["dim"] == "2d" else 1)
    return cfg


def image_root(cfg):
    """Folder holding the network input images for the configured data version."""
    return Path(cfg["histmatched_root"] if cfg["data_version"] == "histmatched" else cfg["aligned_root"])


def list_cases(cfg, split="Training"):
    return sorted(d.name for d in (Path(cfg["aligned_root"]) / split).iterdir() if d.is_dir())
