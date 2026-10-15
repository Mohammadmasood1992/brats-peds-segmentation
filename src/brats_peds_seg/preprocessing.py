"""Network input preparation and the on-disk cache.

Preprocessing is identical for training, validation and new data: load the four sequences, apply the
brain mask, normalize, and crop to the brain bounding box (expanded to include all tumor labels) plus a
margin. Cached arrays are uncompressed `.npy` files (float16 images) so 2D slices can be memory-mapped.
"""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .config import ALL_MODALITIES, LABEL_NAMES, N_CLASSES, image_root


def cache_dir_for(cfg):
    masking = "brain" if cfg["brain_mask"] else "head"
    return Path(cfg["cache_root"]) / f"{cfg['data_version']}_{cfg['normalization']}_{masking}"


def reference_statistics(cfg):
    """Per-sequence mean/std of the histogram-matching reference subject (written by the matching step)."""
    stats = pd.read_csv(Path(cfg["histmatched_root"]) / "reports" / "reference_subject_statistics.csv").set_index("modality")
    return {m: (float(stats.loc[m, "mean"]), float(stats.loc[m, "std"])) for m in ALL_MODALITIES}


def preprocess_case(case, image_dir, cfg, mask_dir=None, label_dir=None):
    """Normalized, cropped arrays of one subject.

    image_dir / mask_dir / label_dir are folders containing {case}-{modality}.nii.gz,
    {case}-t1n_mask.nii.gz and {case}-seg.nii.gz.
    """
    mask_dir = mask_dir or image_dir
    reference = nib.load(image_dir / f"{case}-t1n.nii.gz")
    images = np.stack([nib.load(image_dir / f"{case}-{m}.nii.gz").get_fdata(dtype=np.float32) for m in ALL_MODALITIES])
    brain = np.asarray(nib.load(mask_dir / f"{case}-t1n_mask.nii.gz").dataobj) > 0
    label = None
    if label_dir is not None:
        label = np.asarray(nib.load(label_dir / f"{case}-seg.nii.gz").dataobj).astype(np.uint8)

    region = brain if cfg["brain_mask"] else np.any(images != 0, axis=0)
    ref_stats = reference_statistics(cfg) if cfg["normalization"] == "zscore_reference" else None
    for i, modality in enumerate(ALL_MODALITIES):
        image = images[i]
        if cfg["normalization"] == "zscore_subject":
            values = image[region]
            image -= values.mean()
            image /= max(values.std(), 1e-6)
        elif cfg["normalization"] == "zscore_reference":
            mean, std = ref_stats[modality]
            image -= mean
            image /= std
        if cfg["brain_mask"] or cfg["normalization"] != "none":
            image[~region] = 0.0

    keep = region | (label > 0) if label is not None else region
    coords = np.argwhere(keep)
    low = np.maximum(coords.min(0) - cfg["crop_margin"], 0)
    high = np.minimum(coords.max(0) + cfg["crop_margin"] + 1, keep.shape)
    box = tuple(slice(int(a), int(b)) for a, b in zip(low, high))
    return {
        "case": case,
        "image": images[(slice(None),) + box],
        "mask": brain[box].astype(np.uint8),
        "label": label[box] if label is not None else None,
        "bbox": [[int(a), int(b)] for a, b in zip(low, high)],
        "shape": list(brain.shape),
        "affine": reference.affine,
        "header": reference.header,
    }


def slice_index(label, mask):
    """Per slice axis: brain slices and slices containing each tumor class (for 2D sampling)."""
    index = {}
    for axis in range(3):
        other = tuple(a for a in range(3) if a != axis)
        index[axis] = {
            "brain": np.flatnonzero(mask.any(axis=other)).tolist(),
            "classes": {k: np.flatnonzero((label == k).any(axis=other)).tolist() for k in LABEL_NAMES},
        }
    return index


def cache_case(case, cfg, cache_dir):
    out = cache_dir / case
    if (out / "meta.json").exists():
        return
    out.mkdir(parents=True, exist_ok=True)
    case_dir = Path(cfg["aligned_root"]) / "Training" / case
    data = preprocess_case(case, image_root(cfg) / "Training" / case, cfg, mask_dir=case_dir, label_dir=case_dir)
    np.save(out / "image.npy", data["image"].astype(np.float16))
    np.save(out / "label.npy", data["label"])
    np.save(out / "mask.npy", data["mask"])
    meta = {
        "case": case, "bbox": data["bbox"], "shape": data["shape"],
        "label_counts": {str(k): int((data["label"] == k).sum()) for k in range(N_CLASSES)},
        "slices": slice_index(data["label"], data["mask"]),
    }
    (out / "meta.json").write_text(json.dumps(meta))  # written last: marks the case as complete


def build_cache(cfg, cases):
    """Cache the given Training subjects (skips subjects already cached) and return the cache folder."""
    cache_dir = cache_dir_for(cfg)
    missing = [c for c in cases if not (cache_dir / c / "meta.json").exists()]
    if missing:
        with ThreadPoolExecutor(8) as pool:
            list(tqdm(pool.map(lambda c: cache_case(c, cfg, cache_dir), missing), total=len(missing), desc=f"Caching {cache_dir.name}"))
    print(f"Cache ready: {cache_dir} ({len(cases)} subjects, {len(missing)} newly cached)")
    return cache_dir


def load_meta(cache_dir, case):
    meta = json.loads((cache_dir / case / "meta.json").read_text())
    meta["slices"] = {int(a): {"brain": v["brain"], "classes": {int(k): s for k, s in v["classes"].items()}} for a, v in meta["slices"].items()}
    return meta
