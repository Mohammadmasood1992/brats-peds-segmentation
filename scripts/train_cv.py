"""Cross-validated training of one experiment.

Examples:
    python scripts/train_cv.py --config configs/3d_histmatched.yaml
    python scripts/train_cv.py --config configs/2p5d_histmatched.yaml --set epochs=100 --set folds=[0,1]
    python scripts/train_cv.py --config configs/3d_histmatched.yaml --smoke     # 2-minute end-to-end check
"""

import argparse

import yaml

from brats_peds_seg.config import load_config
from brats_peds_seg.training import run_cross_validation


def parse_override(text):
    key, value = text.split("=", 1)
    return key, yaml.safe_load(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="YAML experiment file (options not listed keep their defaults)")
    parser.add_argument("--set", action="append", default=[], type=parse_override, metavar="KEY=VALUE",
                        help="override one option; VALUE is parsed as YAML (repeatable)")
    parser.add_argument("--smoke", action="store_true", help="tiny model and data, 2 epochs, fold 0")
    args = parser.parse_args()
    overrides = dict(args.set)
    if args.smoke:
        overrides["smoke_test"] = True
    run_cross_validation(load_config(args.config, overrides))


if __name__ == "__main__":
    main()
