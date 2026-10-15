"""Inference: sliding-window (3D) or slice-wise (2D / 2.5D) prediction, flip TTA, fold ensembles, NIfTI output."""

import itertools
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from monai.inferers import sliding_window_inference
from tqdm.auto import tqdm

from .config import N_CLASSES
from .model import DEVICE, autocast, build_model, memory_format
from .preprocessing import preprocess_case


def flip_sets(cfg):
    if cfg["spatial_dims"] == 3:
        axes = [2, 3, 4]
    else:
        axes = [a + 2 for a in range(3) if a != cfg["slice_axis"]]  # in-plane axes of (1, C, X, Y, Z)
    return [c for r in range(len(axes) + 1) for c in itertools.combinations(axes, r)]


def _predict_3d(model, x, cfg):
    with autocast(cfg):
        logits = sliding_window_inference(x, roi_size=tuple(cfg["patch_size_3d"]), sw_batch_size=cfg["sw_batch_size"],
                                          predictor=lambda patch: model(patch.contiguous(memory_format=memory_format(cfg))),
                                          overlap=cfg["sw_overlap"], mode="gaussian")
    return torch.softmax(logits.float(), 1)


def _predict_2d(model, x, cfg):
    axis, k = cfg["slice_axis"], cfg["context_slices"]
    volume = torch.movedim(x[0], axis + 1, 1)  # (C, S, H, W)
    channels, n_slices, height, width = volume.shape
    multiple = 2 ** (cfg["depth"] - 1)
    pad_h, pad_w = (-height) % multiple, (-width) % multiple
    probs = torch.empty((N_CLASSES, n_slices, height, width), device=x.device)
    batch_size = max(cfg["batch_size_2d"], 8)
    for start in range(0, n_slices, batch_size):
        idx = torch.arange(start, min(n_slices, start + batch_size), device=x.device)
        stack = torch.stack([volume[:, (idx + o).clamp(0, n_slices - 1)] for o in range(-k, k + 1)], 2)  # (C, b, 2k+1, H, W)
        batch = stack.permute(1, 0, 2, 3, 4).reshape(len(idx), channels * (2 * k + 1), height, width)
        batch = F.pad(batch, (0, pad_w, 0, pad_h)).contiguous(memory_format=memory_format(cfg))
        with autocast(cfg):
            logits = model(batch)
        probs[:, idx] = torch.softmax(logits.float()[:, :, :height, :width], 1).permute(1, 0, 2, 3)
    return torch.movedim(probs, 1, axis + 1)[None]


@torch.no_grad()
def predict_probabilities(models, image, cfg, tta=False):
    """image: float32 array (C, X, Y, Z) with the configured channels -> class probabilities (N_CLASSES, X, Y, Z)."""
    x = torch.from_numpy(np.ascontiguousarray(image))[None].to(DEVICE)
    total, count = None, 0
    for model in models:
        model.eval()
        for flips in (flip_sets(cfg) if tta else [()]):
            xf = torch.flip(x, flips) if flips else x
            p = _predict_3d(model, xf, cfg) if cfg["spatial_dims"] == 3 else _predict_2d(model, xf, cfg)
            p = torch.flip(p, flips) if flips else p
            total = p if total is None else total + p
            count += 1
    return (total / count)[0].cpu().numpy()


def uncrop(labels, bbox, shape):
    full = np.zeros(shape, dtype=np.uint8)
    full[tuple(slice(a, b) for a, b in bbox)] = labels
    return full


def save_segmentation(labels_full, reference_image, path):
    header = reference_image.header.copy()
    header.set_data_dtype(np.uint8)
    header.set_slope_inter(1, 0)
    image = nib.Nifti1Image(labels_full.astype(np.uint8), reference_image.affine, header)
    for getter, setter in [(reference_image.get_qform, image.set_qform), (reference_image.get_sform, image.set_sform)]:
        affine, code = getter(coded=True)
        if affine is not None:
            setter(affine, int(code))
    nib.save(image, str(path))


def load_models(run_dir, folds="all"):
    """Rebuild the fold models of a run from its config.json and best.pt checkpoints."""
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "config.json").read_text())
    fold_dirs = sorted(run_dir.glob("fold_*")) if folds == "all" else [run_dir / f"fold_{f}" for f in folds]
    models = []
    for fold_dir in fold_dirs:
        model = build_model(cfg)
        model.load_state_dict(torch.load(fold_dir / "best.pt", map_location=DEVICE, weights_only=False)["model"])
        models.append(model.eval())
    print(f"Loaded {len(models)} models from {run_dir.name}")
    return cfg, models


def predict_folder(run_dir, input_root, output_dir, folds="all", tta=None, cases=None):
    """Predict every subject folder in `input_root` with the fold ensemble; writes {case}.nii.gz in the original grid.

    Each subject folder must contain {case}-t1n/t1c/t2w/t2f.nii.gz and {case}-t1n_mask.nii.gz, prepared like
    the training data (rigid alignment, SynthStrip, and histogram matching for `histmatched` runs).
    """
    cfg, models = load_models(run_dir, folds)
    tta = cfg["predict_tta"] if tta is None else tta
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = cases or sorted(d.name for d in Path(input_root).iterdir() if d.is_dir())
    for case in tqdm(cases, desc="Predicting"):
        case_dir = Path(input_root) / case
        data = preprocess_case(case, case_dir, cfg)
        image = data["image"][cfg["channel_indices"]]
        labels = predict_probabilities(models, image, cfg, tta=tta).argmax(0).astype(np.uint8)
        if cfg["mask_predictions"]:
            labels[data["mask"] == 0] = 0
        save_segmentation(uncrop(labels, data["bbox"], data["shape"]), nib.load(case_dir / f"{case}-t1n.nii.gz"),
                          output_dir / f"{case}.nii.gz")
    print(f"Saved {len(cases)} predictions to {output_dir}")
    return output_dir
