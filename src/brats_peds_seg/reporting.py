"""Run summaries, run comparison, plots and scoring of prediction folders."""

from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .config import LABEL_STRUCTURES, STRUCTURES
from .metrics import case_metrics, summarize_metrics


def run_tables(run_dir):
    """(per-structure summary, per-fold mean Dice) of a finished cross-validation run."""
    metrics = pd.read_csv(Path(run_dir) / "cv_metrics.csv")
    per_fold = metrics.groupby("fold")[[f"dice_{s}" for s in STRUCTURES] + ["mean_label_dice"]].mean()
    return summarize_metrics(metrics), per_fold


def plot_run(run_dir):
    """Out-of-fold Dice per subject and the training curves of every fold."""
    run_dir = Path(run_dir)
    metrics = pd.read_csv(run_dir / "cv_metrics.csv")
    fig, axes = plt.subplots(1, 2, figsize=(14, 4), constrained_layout=True)
    axes[0].boxplot([metrics[f"dice_{s}"] for s in STRUCTURES], tick_labels=list(STRUCTURES))
    axes[0].set(title="Out-of-fold Dice per subject", ylabel="Dice")
    for fold_dir in sorted(run_dir.glob("fold_*")):
        log = pd.read_csv(fold_dir / "log.csv")
        axes[1].plot(log["epoch"], log["train_loss"], label=f"{fold_dir.name} loss")
        if "val_mean_label_dice" in log:
            val = log.dropna(subset=["val_mean_label_dice"])
            axes[1].plot(val["epoch"], val["val_mean_label_dice"], "--", label=f"{fold_dir.name} val Dice")
    axes[1].set(title="Training curves", xlabel="epoch")
    axes[1].legend(fontsize=7)
    return fig


def compare_runs(runs_root="runs"):
    """One row per finished run, sorted by mean label Dice."""
    rows = []
    for summary_path in sorted(Path(runs_root).glob("*/cv_summary.csv")):
        summary = pd.read_csv(summary_path).set_index("structure")
        rows.append({"run": summary_path.parent.name, **{f"dice_{s}": summary.loc[s, "dice_mean"] for s in STRUCTURES},
                     **{f"hd95_{s}": summary.loc[s, "hd95_median"] for s in ["TC", "WT"]}})
    table = pd.DataFrame(rows)
    if len(table):
        table["mean_label_dice"] = table[[f"dice_{s}" for s in LABEL_STRUCTURES]].mean(1)
        table = table.sort_values("mean_label_dice", ascending=False)
    return table


def plot_prediction(run_dir, case, aligned_root, modality="t2f"):
    """Reference vs out-of-fold prediction on three orthogonal slices through the tumor centre."""
    run_dir, aligned_root = Path(run_dir), Path(aligned_root)
    prediction_path = next(run_dir.glob(f"fold_*/val_predictions/{case}.nii.gz"))
    image = nib.load(aligned_root / "Training" / case / f"{case}-{modality}.nii.gz").get_fdata(dtype=np.float32)
    reference = np.asarray(nib.load(aligned_root / "Training" / case / f"{case}-seg.nii.gz").dataobj)
    prediction = np.asarray(nib.load(prediction_path).dataobj)
    centre = np.round(np.argwhere(reference > 0).mean(0)).astype(int)
    colors = plt.matplotlib.colors.ListedColormap(["none", "#d62728", "#1f77b4", "#2ca02c", "#ff7f0e"])
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    for row, (labels, title) in enumerate([(reference, "reference"), (prediction, "prediction")]):
        for col, axis in enumerate([2, 1, 0]):
            take = lambda v: np.take(v, centre[axis], axis=axis).T
            ax = axes[row, col]
            ax.imshow(take(image), cmap="gray", origin="lower", vmax=np.percentile(image[image > 0], 99))
            ax.imshow(np.ma.masked_equal(take(labels), 0), cmap=colors, vmin=0, vmax=4, origin="lower", alpha=0.5, interpolation="nearest")
            ax.set_title(f"{title} ({['sagittal', 'coronal', 'axial'][axis]})")
            ax.set_axis_off()
    fig.suptitle(f"{case}: red ET, blue NET, green CC, orange ED")
    return fig


def evaluate_folder(prediction_dir, label_root, output_csv=None):
    """Score {case}.nii.gz predictions against label_root/{case}/{case}-seg.nii.gz (subjects without labels are skipped)."""
    rows = []
    for path in tqdm(sorted(Path(prediction_dir).glob("*.nii.gz")), desc="Evaluating"):
        case = path.name.removesuffix(".nii.gz")
        label_path = Path(label_root) / case / f"{case}-seg.nii.gz"
        if not label_path.exists():
            continue
        prediction = np.asarray(nib.load(path).dataobj).astype(np.uint8)
        reference = np.asarray(nib.load(label_path).dataobj).astype(np.uint8)
        rows.append({"case_id": case, **case_metrics(prediction, reference)})
    metrics = pd.DataFrame(rows)
    print(f"Evaluated {len(metrics)} subjects with labels")
    if len(metrics):
        metrics.to_csv(output_csv or Path(prediction_dir) / "metrics.csv", index=False)
    return metrics
