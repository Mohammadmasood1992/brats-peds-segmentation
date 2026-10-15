"""Subject-level k-fold split of the Training subjects.

Each subject contributes one multi-sequence scan, so a subject-level split has no leakage between
folds. Folds are stratified by data source (CBTN / DIPGr / DFCI) and by the presence of the rare labels
CC and ED, created once and saved so every experiment uses the same split.
"""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .config import LABEL_NAMES, list_cases


def label_presence_table(cfg, cases):
    aligned_root = Path(cfg["aligned_root"])

    def read(case):
        seg = np.asarray(nib.load(aligned_root / "Training" / case / f"{case}-seg.nii.gz").dataobj)
        return {"case_id": case, **{f"has_{name}": bool((seg == k).any()) for k, name in LABEL_NAMES.items()}}
    with ThreadPoolExecutor(8) as pool:
        return pd.DataFrame(list(pool.map(read, cases)))


def folds_path(cfg):
    return Path(cfg["cache_root"]) / f"folds_{cfg['n_folds']}_seed{cfg['seed']}.json"


def load_or_create_folds(cfg):
    """Return {fold: [case ids]}; created and saved (json + csv with stratification columns) on first use."""
    training_cases = list_cases(cfg, "Training")
    path = folds_path(cfg)
    if path.exists():
        folds = {int(k): v for k, v in json.loads(path.read_text()).items()}
        assert sorted(c for v in folds.values() for c in v) == training_cases, "fold file does not match the Training subjects"
        return folds
    print(f"Creating {cfg['n_folds']}-fold split of {len(training_cases)} Training subjects")
    path.parent.mkdir(parents=True, exist_ok=True)
    table = label_presence_table(cfg, training_cases)
    metadata = pd.read_csv(cfg["metadata_file"], sep="\t").drop_duplicates("BraTS-SubjectID").set_index("BraTS-SubjectID")
    table["source"] = table["case_id"].map(metadata["Source"]).fillna("unknown")
    table["stratum"] = table["source"] + "|CC" + table["has_CC"].astype(int).astype(str) + "|ED" + table["has_ED"].astype(int).astype(str)
    counts = table["stratum"].value_counts()
    table.loc[table["stratum"].map(counts) < cfg["n_folds"], "stratum"] = "rare"
    splitter = StratifiedKFold(cfg["n_folds"], shuffle=True, random_state=cfg["seed"])
    folds = {fold: sorted(table["case_id"].iloc[val_idx]) for fold, (_, val_idx) in enumerate(splitter.split(table, table["stratum"]))}
    path.write_text(json.dumps(folds, indent=1))
    table.assign(fold=table["case_id"].map({c: f for f, cs in folds.items() for c in cs})).to_csv(path.with_suffix(".csv"), index=False)
    print(f"Saved folds to {path}")
    return folds


def fold_summary(cfg):
    """Per-fold subject counts, rare-label presence and data sources."""
    table = pd.read_csv(folds_path(cfg).with_suffix(".csv"))
    return table.groupby("fold").agg(subjects=("case_id", "size"), has_ET=("has_ET", "sum"), has_CC=("has_CC", "sum"),
                                     has_ED=("has_ED", "sum"), CBTN=("source", lambda s: (s == "CBTN").sum()),
                                     DIPGr=("source", lambda s: (s == "DIPGr").sum()))


def fold_cases(cfg, folds, fold):
    """Training and validation subjects of one fold (truncated when `max_cases` is set)."""
    train = [c for f, cases in folds.items() if f != fold for c in cases]
    val = list(folds[fold])
    if cfg["max_cases"]:
        train, val = train[: cfg["max_cases"]], val[: max(2, cfg["max_cases"] // 4)]
    return train, val
