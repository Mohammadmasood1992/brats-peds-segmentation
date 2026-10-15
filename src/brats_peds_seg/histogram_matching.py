"""Brain-masked histogram matching.

Intensity distributions are restricted to each subject's SynthStrip brain mask ({case}-t1n_mask.nii.gz),
so zero background and skull/scalp voxels do not influence the matching.

- Voxels used: inside the brain mask AND finite AND nonzero.
- Reference: the Training subject closest to the robust median of per-sequence in-mask mean, std,
  skewness and excess kurtosis.
- Output: matched values inside the mask, 0 outside, float32, original geometry. Each subject's mask is
  copied next to its matched scans; segmentation files are not copied.
"""

import json
import multiprocessing
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import kurtosis, skew
from skimage.exposure import match_histograms
from tqdm.auto import tqdm

MRI_MODALITIES = ["t1n", "t1c", "t2w", "t2f"]
SPLITS = ["Training", "Validation"]
SEED = 2026
VOXEL_SAMPLE_PER_SCAN = 10_000
STATISTICS_TO_REPORT = ["mean", "std", "skewness", "excess_kurtosis", "p01", "median", "p99"]
REFERENCE_METRICS = ["mean", "std", "skewness", "excess_kurtosis"]


def summarize_intensities(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)),
        "skewness": float(skew(values, bias=False)),
        "excess_kurtosis": float(kurtosis(values, fisher=True, bias=False)),
        "p01": float(np.percentile(values, 1)),
        "median": float(np.median(values)),
        "p99": float(np.percentile(values, 99)),
    }


def mask_path_for(case_dir):
    return case_dir / f"{case_dir.name}-t1n_mask.nii.gz"


def load_masked(image_path, mask_path):
    """Return (image, volume, brain mask, voxel-selection mask)."""
    image = nib.load(str(image_path))
    volume = image.get_fdata(dtype=np.float32)
    mask_image = nib.load(str(mask_path))
    if mask_image.shape != image.shape or not np.allclose(mask_image.affine, image.affine, atol=1e-4):
        raise ValueError(f"Mask geometry does not match {image_path}")
    brain = np.asarray(mask_image.dataobj) > 0
    selection = brain & np.isfinite(volume) & (volume != 0)
    return image, volume, brain, selection


def build_scan_index(data_root):
    rows = []
    for split in SPLITS:
        if not (data_root / split).is_dir():
            continue
        for case_dir in sorted((data_root / split).iterdir()):
            if not case_dir.is_dir():
                continue
            mask_path = mask_path_for(case_dir)
            if not mask_path.exists():
                raise FileNotFoundError(f"Missing brain mask: {mask_path}")
            for modality in MRI_MODALITIES:
                path = case_dir / f"{case_dir.name}-{modality}.nii.gz"
                if path.exists():
                    rows.append({"split": split, "case_id": case_dir.name, "modality": modality,
                                 "path": path, "mask_path": mask_path})
    return pd.DataFrame(rows).sort_values(["split", "case_id", "modality"]).reset_index(drop=True)


def stats_worker(args):
    """Pass 1: sampled in-mask statistics of one Training scan (for reference selection)."""
    index, path, mask_path = args
    _, volume, _, selection = load_masked(path, mask_path)
    values = volume[selection]
    rng = np.random.default_rng([SEED, index])
    sample = rng.choice(values, size=min(VOXEL_SAMPLE_PER_SCAN, values.size), replace=False)
    return index, summarize_intensities(sample)


_reference_values = {}


def init_match_worker(reference_values):
    global _reference_values
    _reference_values = reference_values


def match_worker(args):
    """Pass 2: match one scan to the reference distribution of its sequence and write it."""
    index, split, case_id, modality, path, mask_path, output_path, keep_sample = args
    image, volume, brain, selection = load_masked(path, mask_path)
    source_values = volume[selection]
    matched_values = match_histograms(source_values, _reference_values[modality], channel_axis=None)

    rng = np.random.default_rng([SEED, 1, index])
    sample_indices = rng.choice(source_values.size, size=min(VOXEL_SAMPLE_PER_SCAN, source_values.size), replace=False)
    before_sample = source_values[sample_indices]
    after_sample = matched_values[sample_indices]

    matched_volume = np.zeros_like(volume, dtype=np.float32)
    matched_volume[selection] = matched_values.astype(np.float32, copy=False)
    header = image.header.copy()
    header.set_data_dtype(np.float32)
    header.set_slope_inter(1.0, 0.0)
    header["descrip"] = b"Brain-masked histogram match to Training reference"
    output_image = nib.Nifti1Image(matched_volume, image.affine, header)
    qform, qcode = image.get_qform(coded=True)
    sform, scode = image.get_sform(coded=True)
    if qform is not None:
        output_image.set_qform(qform, int(qcode))
    if sform is not None:
        output_image.set_sform(sform, int(scode))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(output_image, str(output_path))

    seg_path = path.parent / f"{case_id}-seg.nii.gz"
    tumor_outside = tumor_total = np.nan
    if seg_path.exists():
        tumor = np.asarray(nib.load(str(seg_path)).dataobj) > 0
        tumor_total = int(tumor.sum())
        tumor_outside = int((tumor & ~brain).sum())

    row = {
        "case_id": case_id, "split": split, "modality": modality,
        "sampled_voxels": int(sample_indices.size),
        "brain_mask_voxels": int(brain.sum()),
        "matched_voxels": int(source_values.size),
        "zero_or_nonfinite_in_mask": int(brain.sum() - source_values.size),
        "nonzero_outside_mask_zeroed": int((np.isfinite(volume) & (volume != 0) & ~brain).sum()),
        "tumor_voxels": tumor_total,
        "tumor_voxels_outside_mask": tumor_outside,
    }
    row.update({f"{k}_before": v for k, v in summarize_intensities(before_sample).items()})
    row.update({f"{k}_after": v for k, v in summarize_intensities(after_sample).items()})
    sample = (before_sample.copy(), after_sample.copy()) if keep_sample else None
    return index, row, sample


def select_reference(scan_index, workers):
    """Training subject closest to the robust (median / IQR-scaled) centre of the in-mask statistics."""
    training = scan_index[scan_index["split"].eq("Training")]
    jobs = [(i, r.path, r.mask_path) for i, r in training.iterrows()]
    stats = {}
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for index, summary in tqdm(pool.map(stats_worker, jobs, chunksize=4), total=len(jobs), desc="Pass 1: in-mask stats (Training)"):
            stats[index] = summary
    scan_stats = pd.concat([training[["case_id", "modality"]], pd.DataFrame.from_dict(stats, orient="index")], axis=1)
    feature_table = scan_stats.pivot(index="case_id", columns="modality", values=REFERENCE_METRICS)
    feature_iqr = feature_table.quantile(0.75) - feature_table.quantile(0.25)
    scaled = (feature_table - feature_table.median()) / feature_iqr.replace(0, np.nan)
    reference_distance = np.sqrt(scaled.fillna(0).pow(2).mean(axis=1))
    return reference_distance.idxmin(), reference_distance, scan_stats


def plot_statistics(before_after_stats, path):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for ax, statistic, title in zip(axes.flat, REFERENCE_METRICS, ["Mean", "Standard deviation", "Skewness", "Excess kurtosis"]):
        plot_data = before_after_stats.melt(id_vars=["modality"], value_vars=[f"{statistic}_before", f"{statistic}_after"],
                                            var_name="state", value_name="value")
        plot_data["state"] = plot_data["state"].map({f"{statistic}_before": "Before", f"{statistic}_after": "After"})
        sns.boxplot(data=plot_data, x="modality", y="value", hue="state", order=MRI_MODALITIES,
                    hue_order=["Before", "After"], showfliers=False, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("MRI sequence")
        ax.grid(axis="y", alpha=0.2)
        if ax is not axes.flat[0]:
            ax.get_legend().remove()
    axes.flat[0].legend(title="Processing")
    fig.suptitle("Per-scan brain-mask statistics before and after histogram matching")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_example_histograms(comparison_samples, reference_values, comparison_case, reference_case, rng, path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, modality in zip(axes.flat, MRI_MODALITIES):
        samples = dict(comparison_samples[modality])
        samples["reference"] = rng.choice(reference_values[modality], size=min(100_000, reference_values[modality].size), replace=False)
        low = min(np.percentile(v, 0.5) for v in samples.values())
        high = max(np.percentile(v, 99.5) for v in samples.values())
        for key, label, color in [("before", "Validation before", "#666666"), ("after", "Validation after", "#007f6e"),
                                  ("reference", "Training reference", "#d1495b")]:
            ax.hist(samples[key], bins=100, range=(low, high), density=True, histtype="step", linewidth=1.6, color=color, label=label)
        ax.set_title(modality.upper())
        ax.set_xlabel("Intensity inside brain mask (arbitrary units)")
        ax.set_ylabel("Density")
        ax.grid(alpha=0.2)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(f"Example brain-mask histograms: {comparison_case} matched to {reference_case}")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run_histogram_matching(data_root, output_root=None, workers=8):
    """Match every scan in data_root/{Training,Validation} and write images, masks and reports to output_root.

    output_root defaults to <data_root>_histogram_matched_masked_ref_<reference case>.
    """
    data_root = Path(data_root)
    scan_index = build_scan_index(data_root)
    print(f"Indexed {len(scan_index):,} MRI scans across {scan_index['case_id'].nunique():,} subjects")
    reference_case, reference_distance, scan_stats = select_reference(scan_index, workers)
    print(f"Reference subject (brain-masked): {reference_case}, distance {reference_distance[reference_case]:.3f}")

    output_root = Path(output_root) if output_root else data_root.parent / f"{data_root.name}_histogram_matched_masked_ref_{reference_case}"
    report_root = output_root / "reports"
    if output_root.exists():
        raise FileExistsError(f"Output already exists: {output_root}. Choose a new directory before rerunning.")
    report_root.mkdir(parents=True)
    scan_stats.to_csv(report_root / "training_inmask_statistics_for_reference_selection.csv", index=False)
    reference_distance.rename("distance_iqr_scaled").sort_values().to_csv(report_root / "reference_distance.csv")

    reference_values, reference_paths = {}, {}
    for modality in MRI_MODALITIES:
        row = scan_index[scan_index["case_id"].eq(reference_case) & scan_index["modality"].eq(modality)].iloc[0]
        _, volume, _, selection = load_masked(row.path, row.mask_path)
        reference_values[modality] = volume[selection].copy()
        reference_paths[modality] = str(row.path)

    rng = np.random.default_rng(SEED)
    reference_stat_rows = []
    for modality, values in reference_values.items():
        sample = rng.choice(values, size=min(VOXEL_SAMPLE_PER_SCAN, values.size), replace=False)
        row = {"modality": modality, "reference_case": reference_case, "sampled_voxels": sample.size, "brain_voxels": values.size}
        row.update(summarize_intensities(sample))
        reference_stat_rows.append(row)
    reference_stats = pd.DataFrame(reference_stat_rows)

    comparison_case = sorted(scan_index.loc[scan_index["split"].eq("Validation"), "case_id"].unique())[0]
    match_jobs = [(i, r.split, r.case_id, r.modality, r.path, r.mask_path, output_root / r.path.relative_to(data_root),
                   r.case_id == comparison_case) for i, r in scan_index.iterrows()]
    stat_rows, manifest_rows, comparison_samples = [], [], {}
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=init_match_worker, initargs=(reference_values,)) as pool:
        for index, row, sample in tqdm(pool.map(match_worker, match_jobs, chunksize=2), total=len(match_jobs), desc="Pass 2: matching"):
            stat_rows.append(row)
            job = match_jobs[index]
            manifest_rows.append({
                "case_id": row["case_id"], "split": row["split"], "modality": row["modality"],
                "source_path": job[4].as_posix(), "mask_path": job[5].as_posix(), "matched_path": job[6].as_posix(),
                "reference_case": reference_case, "reference_modality_path": reference_paths[row["modality"]],
            })
            if sample is not None:
                comparison_samples[row["modality"]] = {"before": sample[0], "after": sample[1]}

    for case_dir in sorted({Path(job[5]).parent for job in match_jobs}):
        shutil.copy2(mask_path_for(case_dir), output_root / case_dir.relative_to(data_root) / mask_path_for(case_dir).name)

    before_after_stats = pd.DataFrame(stat_rows)
    scan_manifest = pd.DataFrame(manifest_rows)
    summary_rows = []
    for modality in MRI_MODALITIES:
        modality_rows = before_after_stats[before_after_stats["modality"].eq(modality)]
        reference_row = reference_stats[reference_stats["modality"].eq(modality)].iloc[0]
        for statistic in STATISTICS_TO_REPORT:
            after_median = modality_rows[f"{statistic}_after"].median()
            summary_rows.append({
                "modality": modality, "statistic": statistic, "scans": len(modality_rows),
                "median_across_scans_before": modality_rows[f"{statistic}_before"].median(),
                "median_across_scans_after": after_median,
                "reference_subject_statistic": reference_row[statistic],
                "median_after_minus_reference": after_median - reference_row[statistic],
            })
    modality_summary = pd.DataFrame(summary_rows)

    before_after_stats.to_csv(report_root / "per_scan_before_after_statistics.csv", index=False)
    modality_summary.to_csv(report_root / "modality_before_after_summary.csv", index=False)
    reference_stats.to_csv(report_root / "reference_subject_statistics.csv", index=False)
    scan_manifest.to_csv(report_root / "matched_scan_manifest.csv", index=False)
    plot_statistics(before_after_stats, report_root / "before_after_statistics.png")
    plot_example_histograms(comparison_samples, reference_values, comparison_case, reference_case, rng,
                            report_root / "example_histograms_before_after.png")
    run_info = {
        "method": "skimage.exposure.match_histograms on brain-masked voxels",
        "brain_mask": "SynthStrip <case>-t1n_mask.nii.gz, applied to all four co-registered sequences",
        "reference_case": reference_case,
        "reference_split": "Training",
        "reference_selection": "Training subject closest to robust median across per-modality in-mask mean, std, "
                               "skewness, and excess kurtosis",
        "reference_distance_iqr_scaled": float(reference_distance[reference_case]),
        "modalities": MRI_MODALITIES,
        "matched_scans": int(len(scan_manifest)),
        "subjects": int(scan_manifest["case_id"].nunique()),
        "voxel_rule": "Match voxels inside brain mask that are finite and nonzero; set all other voxels to 0",
        "output_dtype": "float32",
        "statistics_sampling": f"{VOXEL_SAMPLE_PER_SCAN:,} in-mask voxels per scan, seeded per scan from {SEED}",
        "comparison_case": comparison_case,
        "masks": "Copied next to matched scans",
        "segmentation_files": "Not copied; labels are read from the aligned dataset",
    }
    (report_root / "run_info.json").write_text(json.dumps(run_info, indent=2))

    # Validate one output: geometry preserved and background outside the mask is zero.
    check = scan_manifest.iloc[0]
    source, output = nib.load(check["source_path"]), nib.load(check["matched_path"])
    assert source.shape == output.shape and np.allclose(source.affine, output.affine)
    brain = np.asarray(nib.load(check["mask_path"]).dataobj) > 0
    assert not np.any(output.get_fdata(dtype=np.float32)[~brain])

    tumor_cut = before_after_stats.drop_duplicates("case_id")
    tumor_cut = tumor_cut[tumor_cut["tumor_voxels_outside_mask"] > 0]
    print(f"Matched {len(scan_manifest):,} scans across {scan_manifest['case_id'].nunique():,} subjects into {output_root}")
    print(f"Subjects with tumor voxels outside the brain mask: {len(tumor_cut)}")
    return output_root, modality_summary
