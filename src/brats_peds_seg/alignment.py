"""Rigid orientation alignment of BraTS-PEDs subjects to a template subject.

All volumes share one header (LPS, 240x240x155, 1 mm), but in some subjects the head is rotated inside
that grid. The t1n of each subject is registered to the template t1n (6-DOF rigid: rotation + translation)
with Mattes mutual information on percentile-clipped intensities inside a head mask, multi-resolution.

Mutual information could not reliably distinguish true 90/180 degree rotations from upright heads in this
data (it repeatedly preferred wrong flips), so gross rotations are identified visually from an orientation
screen and passed as registration start rotations; all other subjects start from the identity.

Subjects rotated by more than a threshold are resampled (four sequences linear, `seg` nearest neighbour,
same grid); the others are hard-linked (or copied) unchanged.
"""

import multiprocessing
import os
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage
from tqdm.auto import tqdm

MRI_MODALITIES = ["t1n", "t1c", "t2w", "t2f"]
SPLITS = ["Training", "Validation"]
SEED = 2026


# ---------------------------------------------------------------- registration of one subject
def find_case_dir(data_root, case):
    for split in SPLITS:
        case_dir = Path(data_root) / split / case
        if case_dir.is_dir():
            return case_dir
    raise FileNotFoundError(case)


def head_mask(image):
    """Otsu foreground of the 99th-percentile-clipped image, largest component, closed, holes filled per slice."""
    array = sitk.GetArrayFromImage(image)
    clipped = sitk.GetImageFromArray(np.clip(array, 0, np.percentile(array[array > 0], 99)).astype(np.float32))
    clipped.CopyInformation(image)
    mask = sitk.BinaryMorphologicalOpening(sitk.OtsuThreshold(clipped, 0, 1), [2] * 3)
    components = sitk.RelabelComponent(sitk.ConnectedComponent(mask), sortByObjectSize=True)
    mask = sitk.GetArrayFromImage(sitk.BinaryMorphologicalClosing(sitk.Cast(components == 1, sitk.sitkUInt8), [3] * 3)) > 0
    for axis in range(3):  # the head is open at the FOV edge, so fill holes slice by slice along each axis
        mask = np.moveaxis(np.stack([ndimage.binary_fill_holes(s) for s in np.moveaxis(mask, axis, 0)]), 0, axis)
    out = sitk.GetImageFromArray(mask.astype(np.uint8))
    out.CopyInformation(image)
    return out


def normalize(image, mask):
    """Clip to the 0.5-99.5 percentile inside the head and rescale to [0, 1] so MI bins cover tissue contrast."""
    array = sitk.GetArrayFromImage(image)
    low, high = np.percentile(array[sitk.GetArrayFromImage(mask) > 0], [0.5, 99.5])
    out = sitk.GetImageFromArray(np.clip((array - low) / (high - low), 0, 1).astype(np.float32))
    out.CopyInformation(image)
    return out


def centroid(mask):
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(mask)
    return np.array(stats.GetCentroid(1))


def _register(fixed, fixed_mask, moving, initial, shrink, sigmas, iterations):
    method = sitk.ImageRegistrationMethod()
    method.SetMetricAsMattesMutualInformation(32)
    method.SetMetricSamplingStrategy(method.RANDOM)
    method.SetMetricSamplingPercentage(0.05, seed=SEED)
    method.SetMetricFixedMask(fixed_mask)
    method.SetInterpolator(sitk.sitkLinear)
    method.SetOptimizerAsRegularStepGradientDescent(1.0, 1e-4, iterations, relaxationFactor=0.6)
    method.SetOptimizerScalesFromPhysicalShift()
    method.SetShrinkFactorsPerLevel(shrink)
    method.SetSmoothingSigmasPerLevel(sigmas)
    method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    transform = sitk.Euler3DTransform(initial)
    method.SetInitialTransform(transform, inPlace=True)
    method.Execute(fixed, moving)
    return transform, method.GetMetricValue()


def rotation_angle_deg(matrix):
    matrix = np.asarray(matrix).reshape(3, 3)
    return float(np.degrees(np.arccos(np.clip((np.trace(matrix) - 1) / 2, -1, 1))))


def _evaluate(fixed, fixed_mask, moving, transform):
    """Mattes MI over every voxel of the fixed mask (no random sampling), comparable across starts."""
    method = sitk.ImageRegistrationMethod()
    method.SetMetricAsMattesMutualInformation(32)
    method.SetMetricSamplingStrategy(method.NONE)
    method.SetMetricFixedMask(fixed_mask)
    method.SetInterpolator(sitk.sitkLinear)
    method.SetInitialTransform(sitk.Euler3DTransform(transform))
    return method.MetricEvaluate(fixed, moving)


def _half_resolution(image, is_mask=False):
    if is_mask:
        return sitk.Shrink(image, [2] * 3)
    return sitk.Shrink(sitk.SmoothingRecursiveGaussian(image, 1.0), [2] * 3)


def _result_fields(prefix, transform):
    return {
        f"{prefix}rotation_deg": rotation_angle_deg(transform.GetMatrix()),
        f"{prefix}translation_mm": float(np.linalg.norm(transform.GetTranslation())),
        f"{prefix}matrix": list(transform.GetMatrix()),
        f"{prefix}center": list(transform.GetCenter()),
        f"{prefix}parameters": list(transform.GetParameters()),
    }


def register_case(data_root, case, template_case, starts_deg=((0, 0, 0),), threads=2):
    """Register case t1n to template t1n.

    Each start rotation (degrees about x, y, z) is optimized to half resolution and scored with full-sampling
    MI at half resolution; the best start (and the identity start, if different) is refined at full resolution.
    Returns plain parameters (picklable); rebuild transforms with transform_from_result.
    """
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(threads)
    fixed = sitk.ReadImage(str(find_case_dir(data_root, template_case) / f"{template_case}-t1n.nii.gz"), sitk.sitkFloat32)
    moving = sitk.ReadImage(str(find_case_dir(data_root, case) / f"{case}-t1n.nii.gz"), sitk.sitkFloat32)
    fixed_mask, moving_mask = head_mask(fixed), head_mask(moving)
    fixed, moving = normalize(fixed, fixed_mask), normalize(moving, moving_mask)
    fixed_center, moving_center = centroid(fixed_mask), centroid(moving_mask)
    fixed_half, mask_half, moving_half = _half_resolution(fixed), _half_resolution(fixed_mask, True), _half_resolution(moving)

    candidates = []
    for rotation in [tuple(np.radians(a) for a in start) for start in starts_deg]:
        initial = sitk.Euler3DTransform()
        initial.SetCenter(fixed_center.tolist())
        initial.SetRotation(*rotation)
        initial.SetTranslation((moving_center - fixed_center).tolist())
        transform, _ = _register(fixed, fixed_mask, moving, initial, [8, 4, 2], [4, 2, 1], 80)
        candidates.append((rotation, transform, _evaluate(fixed_half, mask_half, moving_half, transform)))

    best = min(candidates, key=lambda item: item[2])
    identity = next((c for c in candidates if not any(c[0])), best)
    best_final, _ = _register(fixed, fixed_mask, moving, best[1], [2, 1], [1, 0], 200)
    identity_final = best_final if best is identity else _register(fixed, fixed_mask, moving, identity[1], [2, 1], [1, 0], 200)[0]
    result = {
        "case_id": case,
        "template_case": template_case,
        "start_rotation_deg": [int(round(np.degrees(a))) for a in best[0]],
        "flip_chosen": best is not identity,
        "start_scores": {str([int(round(np.degrees(a))) for a in c[0]]): c[2] for c in candidates},
        "best_score": best[2],
        "identity_score": identity[2],
    }
    result.update(_result_fields("", best_final))
    result.update(_result_fields("identity_", identity_final))
    return result


def register_case_worker(args):
    return register_case(*args)


def transform_from_result(result, prefix=""):
    transform = sitk.Euler3DTransform()
    transform.SetCenter(result[f"{prefix}center"])
    transform.SetParameters(result[f"{prefix}parameters"])
    return transform


def apply_to_case(source_dir, output_dir, transform):
    """Resample the four sequences (linear) and seg (nearest) onto the case's own grid."""
    output_dir.mkdir(parents=True, exist_ok=True)
    case = source_dir.name
    written = []
    for name in MRI_MODALITIES + ["seg"]:
        source = source_dir / f"{case}-{name}.nii.gz"
        if not source.exists():
            continue
        if name == "seg":
            image = sitk.ReadImage(str(source))
            out = sitk.Resample(image, image, transform, sitk.sitkNearestNeighbor, 0, image.GetPixelID())
        else:
            image = sitk.ReadImage(str(source), sitk.sitkFloat32)
            out = sitk.Resample(image, image, transform, sitk.sitkLinear, 0.0, sitk.sitkFloat32)
        destination = output_dir / source.name
        sitk.WriteImage(out, str(destination), useCompression=True)
        written.append(destination)
    return written


def apply_case_worker(args):
    source_dir, output_dir, result = args
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(2)
    return [str(p) for p in apply_to_case(Path(source_dir), Path(output_dir), transform_from_result(result))]


def link_case(source_dir, output_dir, mode="hardlink"):
    """Unchanged subjects: hard-link (no extra disk space) or copy every NIfTI without resampling."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for source in sorted(source_dir.glob("*.nii.gz")):
        destination = output_dir / source.name
        if destination.exists():
            continue
        if mode == "hardlink":
            os.link(source, destination)
        else:
            shutil.copy2(source, destination)


# ---------------------------------------------------------------- dataset level
def list_case_dirs(data_root):
    return {d.name: d for split in SPLITS if (Path(data_root) / split).is_dir()
            for d in sorted((Path(data_root) / split).iterdir()) if d.is_dir()}


def register_all(data_root, template_case, starts_by_case=None, workers=16):
    """Register every subject's t1n to the template (identity start unless a start is given for the subject)."""
    case_dirs = list_case_dirs(data_root)
    starts_by_case = starts_by_case or {}
    jobs = [(str(data_root), case, template_case, starts_by_case.get(case, ((0, 0, 0),))) for case in case_dirs]
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        results = list(tqdm(pool.map(register_case_worker, jobs), total=len(jobs), desc=f"Rigid to {template_case}"))
    table = pd.DataFrame(results)
    table.insert(1, "split", table["case_id"].map({c: d.parent.name for c, d in case_dirs.items()}))
    return table


def mid_slices(volume):
    """nibabel array (x, y, z) in LPS -> axial, coronal, sagittal mid slices."""
    x, y, z = (s // 2 for s in volume.shape)
    return [volume[:, :, z].T, volume[:, y, :].T, volume[x, :, :].T]


def sharpness_ratio(t1n_path):
    """Weakest / strongest mean per-axis intensity gradient in tissue; thick-slice scans fall well below 0.8."""
    volume = nib.load(t1n_path).get_fdata(dtype=np.float32)
    tissue = volume > np.percentile(volume[volume > 0], 50)
    gradients = [np.mean(np.abs(np.diff(volume, axis=a))[tissue.take(range(volume.shape[a] - 1), axis=a)]) for a in range(3)]
    return min(gradients) / max(gradients)


def orientation_screen(data_root, output_dir, per_page=60):
    """Contact sheets of mid-sagittal + mid-axial t1n slices of every subject, for visual orientation review."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    case_dirs = list_case_dirs(data_root)
    cases = list(case_dirs)
    pages = []
    for page_start in range(0, len(cases), per_page):
        fig, axes = plt.subplots(6, 20, figsize=(30, 11))
        for k, ax in enumerate(axes.flat):
            ax.set_axis_off()
            index = page_start + k // 2
            if index >= len(cases):
                continue
            case = cases[index]
            volume = np.asarray(nib.load(case_dirs[case] / f"{case}-t1n.nii.gz").dataobj, dtype=np.float32)
            view = volume[volume.shape[0] // 2].T if k % 2 == 0 else volume[:, :, volume.shape[2] // 2].T
            ax.imshow(view, cmap="gray", origin="lower", vmin=0, vmax=np.percentile(volume[volume > 0], 99))
            if k % 2 == 0:
                ax.set_title(case[-9:-4], fontsize=9)
        fig.subplots_adjust(wspace=0.02, hspace=0.15, left=0.01, right=0.99, top=0.97, bottom=0.01)
        path = output_dir / f"orientation_screen_page{page_start // per_page + 1}.png"
        fig.savefig(path, dpi=55)
        plt.close(fig)
        pages.append(path)
    print(f"Saved {len(pages)} orientation screen pages to {output_dir}")
    return pages


def rank_template_candidates(data_root, report_dir, provisional_template, workers=16, n_candidates=10):
    """Template candidates: typical orientation, small tumor, tumor inside the brain mask, near-isotropic resolution.

    Every subject is first registered (identity start) to a provisional template; a subject's orientation is
    typical when its median rotation to all other subjects is small. Candidates are Training subjects whose
    tumor lies within the SynthStrip brain mask (<= 1 % outside) and is small (<= Training 25th percentile).
    """
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    provisional = register_all(data_root, provisional_template, workers=workers)
    provisional.to_json(report_dir / "pass1_provisional_registration.json", orient="records", indent=1)
    matrices = provisional.set_index("case_id")["matrix"].apply(lambda m: np.asarray(m).reshape(3, 3))
    median_rotation = {case: float(np.median([rotation_angle_deg(matrices[case].T @ other) for other in matrices]))
                       for case in matrices.index}
    case_dirs = list_case_dirs(data_root)
    rows = []
    for case in provisional.loc[provisional["split"].eq("Training"), "case_id"]:
        seg = np.asarray(nib.load(case_dirs[case] / f"{case}-seg.nii.gz").dataobj) > 0
        brain = np.asarray(nib.load(case_dirs[case] / f"{case}-t1n_mask.nii.gz").dataobj) > 0
        rows.append({"case_id": case, "median_rotation_to_others_deg": median_rotation[case],
                     "tumor_voxels": int(seg.sum()), "tumor_outside_mask_pct": 100 * (seg & ~brain).sum() / max(seg.sum(), 1)})
    candidates = pd.DataFrame(rows).set_index("case_id")
    candidates = candidates[(candidates["tumor_outside_mask_pct"] <= 1)
                            & (candidates["tumor_voxels"] <= candidates["tumor_voxels"].quantile(0.25))]
    candidates = candidates.sort_values("median_rotation_to_others_deg").head(n_candidates)
    candidates["sharpness_ratio"] = [sharpness_ratio(case_dirs[c] / f"{c}-t1n.nii.gz") for c in candidates.index]
    candidates.to_csv(report_dir / "template_candidates.csv")
    print(f"Ranked {len(candidates)} template candidates; saved to {report_dir / 'template_candidates.csv'}")
    return candidates


def plot_alignment_qc(data_root, alignment, template_case, output_dir, rows_per_page=6):
    """Before/after mid slices for every realigned subject with the template head outline."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    case_dirs = list_case_dirs(data_root)
    template_image = sitk.ReadImage(str(case_dirs[template_case] / f"{template_case}-t1n.nii.gz"), sitk.sitkFloat32)
    template_outline = mid_slices(np.transpose(sitk.GetArrayFromImage(head_mask(template_image)), (2, 1, 0)))
    flagged = alignment[alignment["realigned"]].sort_values("rotation_deg", ascending=False)
    pages = [flagged.iloc[i:i + rows_per_page] for i in range(0, len(flagged), rows_per_page)]
    paths = []
    for page_number, page in enumerate(pages, 1):
        fig, axes = plt.subplots(len(page), 6, figsize=(18, 3.1 * len(page)), constrained_layout=True, squeeze=False)
        for row_axes, row in zip(axes, page.to_dict("records")):
            case = row["case_id"]
            moving = sitk.ReadImage(str(case_dirs[case] / f"{case}-t1n.nii.gz"), sitk.sitkFloat32)
            aligned = sitk.Resample(moving, moving, transform_from_result(row), sitk.sitkLinear, 0.0)
            for offset, (image, label) in enumerate([(moving, "before"), (aligned, "after")]):
                volume = np.transpose(sitk.GetArrayFromImage(image), (2, 1, 0))
                high = np.percentile(volume[volume > 0], 99)
                for k, (view, outline, name) in enumerate(zip(mid_slices(volume), template_outline, ["axial", "coronal", "sagittal"])):
                    ax = row_axes[offset * 3 + k]
                    ax.imshow(view, cmap="gray", origin="lower", vmax=high)
                    ax.contour(outline, [0.5], colors="orange", linewidths=0.6)
                    ax.set_title(f"{case[-9:-4]} {label} {name} ({row['rotation_deg']:.0f}°)", fontsize=8)
                    ax.set_axis_off()
        fig.suptitle(f"Realigned subjects, page {page_number}/{len(pages)} (orange: template head outline)")
        path = output_dir / f"realigned_qc_page{page_number:02d}.png"
        fig.savefig(path, dpi=80)
        plt.close(fig)
        paths.append(path)
    print(f"Saved {len(paths)} QC pages to {output_dir}")
    return paths


def align_dataset(data_root, output_root, template_case, gross_rotation_starts=None, threshold_deg=10.0,
                  workers=16, link_mode="hardlink"):
    """Register all subjects to the template, save transforms/reports/QC, and write the aligned dataset."""
    data_root, output_root = Path(data_root), Path(output_root)
    report_dir = output_root / "reports"
    (report_dir / "transforms").mkdir(parents=True, exist_ok=True)
    print(f"Registering all subjects to {template_case}; gross rotation starts: {gross_rotation_starts or 'none'}")
    alignment = register_all(data_root, template_case, gross_rotation_starts, workers)
    alignment["template_case"] = template_case
    alignment["realigned"] = alignment["rotation_deg"] > threshold_deg
    alignment.to_json(report_dir / "alignment_results.json", orient="records", indent=1)
    alignment[["case_id", "split", "template_case", "start_rotation_deg", "rotation_deg", "translation_mm", "realigned"]].to_csv(
        report_dir / "alignment_results.csv", index=False)
    for row in alignment.to_dict("records"):
        sitk.WriteTransform(transform_from_result(row), str(report_dir / "transforms" / f"{row['case_id']}_to_{template_case}.tfm"))

    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.hist(alignment["rotation_deg"], bins=np.arange(0, 92.5, 1), color="#4c72b0")
    ax.axvline(threshold_deg, color="#d1495b", linestyle="--", label=f"threshold {threshold_deg:g}°")
    ax.set(xlabel="Rotation to template (degrees)", ylabel="Subjects", title=f"Rigid rotation to {template_case}")
    ax.legend()
    fig.savefig(report_dir / "rotation_histogram.png", dpi=130)
    plt.close(fig)
    plot_alignment_qc(data_root, alignment, template_case, report_dir / "qc")

    case_dirs = list_case_dirs(data_root)
    jobs = []
    for row in alignment.to_dict("records"):
        source_dir = case_dirs[row["case_id"]]
        output_dir = output_root / source_dir.relative_to(data_root)
        if row["realigned"]:
            jobs.append((str(source_dir), str(output_dir), row))
        else:
            link_case(source_dir, output_dir, link_mode)
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        list(tqdm(pool.map(apply_case_worker, jobs), total=len(jobs), desc="Resampling realigned subjects"))
    print(f"Resampled {len(jobs)} subjects (> {threshold_deg:g}°); {link_mode} {len(alignment) - len(jobs)} unchanged subjects into {output_root}")
    return alignment


def run_synthstrip(case_dirs, synthstrip_script):
    """New SynthStrip brain masks ({case}-t1n_mask.nii.gz) written next to each subject's t1n."""
    for case_dir in tqdm([Path(d) for d in case_dirs], desc="SynthStrip"):
        case = case_dir.name
        result = subprocess.run([str(synthstrip_script), "-in", str(case_dir / f"{case}-t1n.nii.gz"), "-out", str(case_dir)],
                                capture_output=True, text=True)
        if result.returncode != 0 or not (case_dir / f"{case}-t1n_mask.nii.gz").exists():
            raise RuntimeError(f"SynthStrip failed for {case}:\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}")
    print(f"New brain masks for {len(case_dirs)} subjects")


def tumor_mask_coverage(original_root, aligned_root):
    """Percentage of tumor voxels outside the brain mask, original vs aligned data (Training subjects)."""
    rows = []
    for case, case_dir in list_case_dirs(aligned_root).items():
        if case_dir.parent.name != "Training":
            continue
        row = {"case_id": case}
        for name, root in [("original", Path(original_root)), ("aligned", Path(aligned_root))]:
            d = root / "Training" / case
            seg = np.asarray(nib.load(d / f"{case}-seg.nii.gz").dataobj) > 0
            brain = np.asarray(nib.load(d / f"{case}-t1n_mask.nii.gz").dataobj) > 0
            row[f"tumor_outside_pct_{name}"] = 100 * (seg & ~brain).sum() / max(seg.sum(), 1)
        rows.append(row)
    return pd.DataFrame(rows)
