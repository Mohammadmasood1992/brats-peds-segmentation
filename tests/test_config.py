"""Configuration loading and derived settings."""

import pytest

from brats_peds_seg.config import load_config, resolve_config


def test_auto_normalization_and_name():
    cfg = resolve_config(load_config(overrides={"data_version": "raw", "dim": "2d", "context_slices": 1}))
    assert cfg["normalization"] == "zscore_subject"
    assert cfg["experiment_name"] == "2dctx1_raw_conv"
    cfg = resolve_config(load_config(overrides={"data_version": "histmatched", "brain_mask": False}))
    assert cfg["normalization"] == "zscore_reference" and cfg["brain_mask"] is True


def test_yaml_experiment_files_load(tmp_path):
    path = tmp_path / "exp.yaml"
    path.write_text("dim: 2d\naugmentation:\n  rotate_deg: 5\n")
    cfg = load_config(path)
    assert cfg["dim"] == "2d" and cfg["augmentation"]["rotate_deg"] == 5 and cfg["augmentation"]["flip_prob"] == 0.5


def test_unknown_options_are_rejected():
    with pytest.raises(KeyError):
        load_config(overrides={"learning_rate": 1e-3})
    with pytest.raises(KeyError):
        load_config(overrides={"augmentation": {"rotation": 5}})


def test_smoke_overrides():
    cfg = resolve_config(load_config(overrides={"smoke_test": True}))
    assert cfg["folds"] == [0] and cfg["epochs"] == 2 and cfg["experiment_name"].endswith("_smoke")
