"""Cross-validation training.

Per fold: train on the other folds, validate every `val_every` epochs on this fold (selection by
`selection_metric`; best and last checkpoints saved), then predict the held-out subjects with the best
checkpoint and save NIfTI predictions and per-subject metrics.
"""

import json
import math
import os
import random
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # tensorboard imports tensorflow (if installed), which is verbose

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from .config import STRUCTURES, resolve_config
from .datasets import build_train_loader
from .folds import fold_cases, load_or_create_folds
from .inference import predict_probabilities, save_segmentation, uncrop
from .losses import SegmentationLoss, compute_class_weights
from .metrics import case_metrics, summarize_metrics
from .model import DEVICE, autocast, build_model, memory_format
from .preprocessing import build_cache, load_meta


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_optimizer(cfg, model):
    if cfg["optimizer"] == "sgd":
        return torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=0.99, nesterov=True, weight_decay=cfg["weight_decay"])
    return torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])


def build_scheduler(cfg, optimizer):
    epochs, warmup = cfg["epochs"], cfg["warmup_epochs"]

    def factor(epoch):
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(1, epochs - warmup)
        if cfg["scheduler"] == "cosine":
            return 0.5 * (1 + math.cos(math.pi * progress))
        if cfg["scheduler"] == "poly":
            return (1 - progress) ** 0.9
        return 1.0
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def predict_cached_case(models, cfg, cache_dir, case, tta=False):
    image = np.load(cache_dir / case / "image.npy")[cfg["channel_indices"]].astype(np.float32)
    labels = predict_probabilities(models, image, cfg, tta=tta).argmax(0).astype(np.uint8)
    if cfg["mask_predictions"]:
        labels[np.load(cache_dir / case / "mask.npy") == 0] = 0
    return labels


def validate(model, cfg, cache_dir, cases, with_hd95=False):
    rows = []
    for case in cases:
        labels = predict_cached_case([model], cfg, cache_dir, case)
        rows.append({"case_id": case, **case_metrics(labels, np.load(cache_dir / case / "label.npy"), with_hd95=with_hd95)})
    return pd.DataFrame(rows)


def train_fold(cfg, fold, folds, run_dir, cache_dir):
    fold_dir = run_dir / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    if (fold_dir / "val_metrics.csv").exists():
        print(f"fold {fold}: already finished, loading metrics")
        return pd.read_csv(fold_dir / "val_metrics.csv")
    train_cases, val_cases = fold_cases(cfg, folds, fold)
    seed_everything(cfg["seed"] + fold)
    model = build_model(cfg)
    class_weights = compute_class_weights(cfg, train_cases, cache_dir)
    loss_fn = SegmentationLoss(cfg, class_weights).to(DEVICE)
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)
    scaler = torch.amp.GradScaler(DEVICE.type, enabled=cfg["amp"] == "fp16")
    start_epoch, best_score, stale, history = 0, -np.inf, 0, []
    if cfg["resume"] and (fold_dir / "last.pt").exists():
        state = torch.load(fold_dir / "last.pt", map_location=DEVICE, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch, best_score, stale, history = state["epoch"] + 1, state["best_score"], state["stale"], state["history"]
        print(f"fold {fold}: resuming at epoch {start_epoch}")

    loader = build_train_loader(cfg, train_cases, cache_dir)
    writer = SummaryWriter(str(fold_dir / "tensorboard"))
    val_subset = val_cases[: cfg["val_max_cases"]] if cfg["val_max_cases"] else val_cases
    print(f"fold {fold}: {len(train_cases)} training / {len(val_cases)} validation subjects; class weights {class_weights}")

    for epoch in range(start_epoch, cfg["epochs"]):
        model.train()
        started, losses = time.time(), []
        for batch in loader:
            x = batch["image"].to(DEVICE, non_blocking=True).contiguous(memory_format=memory_format(cfg))
            y = batch["label"].to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(cfg):
                outputs = model(x)
            loss = loss_fn(outputs, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss.item())
        scheduler.step()
        record = {"epoch": epoch, "train_loss": float(np.mean(losses)), "lr": optimizer.param_groups[0]["lr"],
                  "seconds": time.time() - started}
        writer.add_scalar("train/loss", record["train_loss"], epoch)
        writer.add_scalar("train/lr", record["lr"], epoch)

        if (epoch + 1) % cfg["val_every"] == 0 or epoch + 1 == cfg["epochs"]:
            metrics = validate(model, cfg, cache_dir, val_subset)
            for column in [f"dice_{s}" for s in STRUCTURES] + ["mean_label_dice", "mean_region_dice"]:
                record[f"val_{column}"] = float(metrics[column].mean())
                writer.add_scalar(f"val/{column}", record[f"val_{column}"], epoch)
            key = cfg["selection_metric"] if cfg["selection_metric"].startswith("mean") else f"dice_{cfg['selection_metric']}"
            score = record[f"val_{key}"]
            if score > best_score:
                best_score, stale = score, 0
                torch.save({"model": model.state_dict(), "epoch": epoch, "score": score}, fold_dir / "best.pt")
            else:
                stale += 1
            print(f"fold {fold} epoch {epoch + 1}/{cfg['epochs']}: loss {record['train_loss']:.4f}, "
                  f"val {key} {score:.4f} (best {best_score:.4f}), {record['seconds']:.0f}s/epoch")
        history.append(record)
        pd.DataFrame(history).to_csv(fold_dir / "log.csv", index=False)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "epoch": epoch, "best_score": best_score, "stale": stale, "history": history}, fold_dir / "last.pt")
        if cfg["early_stopping_patience"] and stale >= cfg["early_stopping_patience"]:
            print(f"fold {fold}: early stopping at epoch {epoch + 1}")
            break
    writer.close()

    # Out-of-fold predictions with the best checkpoint (full grid NIfTI) and per-subject metrics incl. HD95.
    model.load_state_dict(torch.load(fold_dir / "best.pt", map_location=DEVICE, weights_only=False)["model"])
    prediction_dir = fold_dir / "val_predictions"
    prediction_dir.mkdir(exist_ok=True)
    aligned_root = Path(cfg["aligned_root"])
    rows = []
    for case in tqdm(val_cases, desc=f"fold {fold} out-of-fold prediction"):
        labels = predict_cached_case([model], cfg, cache_dir, case, tta=cfg["val_tta"])
        meta = load_meta(cache_dir, case)
        reference = nib.load(aligned_root / "Training" / case / f"{case}-t1n.nii.gz")
        save_segmentation(uncrop(labels, meta["bbox"], meta["shape"]), reference, prediction_dir / f"{case}.nii.gz")
        rows.append({"case_id": case, "fold": fold, **case_metrics(labels, np.load(cache_dir / case / "label.npy"))})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(fold_dir / "val_metrics.csv", index=False)
    print(f"fold {fold}: saved out-of-fold predictions and metrics to {fold_dir}")
    del model, optimizer, loader
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return metrics


def run_cross_validation(config):
    """Train the configured folds of one experiment; returns the run folder (runs_root/experiment_name)."""
    cfg = resolve_config(config)
    run_dir = Path(cfg["runs_root"]) / cfg["experiment_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    saved = run_dir / "config.json"
    serializable = json.loads(json.dumps(cfg, default=str))
    if saved.exists() and json.loads(saved.read_text()) != serializable:
        raise ValueError(f"{saved} exists with a different configuration; set a new experiment_name or delete the run")
    saved.write_text(json.dumps(serializable, indent=1))
    print(f"Experiment {cfg['experiment_name']}: {cfg['dim']} U-Net on {cfg['data_version']} data, folds {cfg['folds']}")
    folds = load_or_create_folds(cfg)
    needed = sorted({c for fold in cfg["folds"] for part in fold_cases(cfg, folds, fold) for c in part})
    cache_dir = build_cache(cfg, needed)
    all_metrics = pd.concat([train_fold(cfg, fold, folds, run_dir, cache_dir) for fold in cfg["folds"]], ignore_index=True)
    all_metrics.to_csv(run_dir / "cv_metrics.csv", index=False)
    summary = summarize_metrics(all_metrics)
    summary.to_csv(run_dir / "cv_summary.csv", index=False)
    print(f"Cross-validation finished. Run folder: {run_dir}")
    print(summary.round(3).to_string(index=False))
    return run_dir
