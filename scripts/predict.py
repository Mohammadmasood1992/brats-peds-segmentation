"""Predict new subjects with the fold ensemble of a trained run.

The input folder must hold one folder per subject with {case}-t1n/t1c/t2w/t2f.nii.gz and the SynthStrip mask
{case}-t1n_mask.nii.gz, prepared like the training data (rigid alignment, SynthStrip, and histogram matching
for runs trained on histogram-matched data).

Example:
    python scripts/predict.py --run-dir runs/3d_histmatched_conv \
        --input-root data/BraTS-PEDs-v1_rigid_aligned_histmatched/Validation --output-dir predictions/Validation
"""

import argparse
from pathlib import Path

from brats_peds_seg.inference import predict_folder


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, nargs="*", default=None, help="subset of folds (default: all)")
    parser.add_argument("--tta", choices=["config", "on", "off"], default="config", help="flip test-time augmentation")
    args = parser.parse_args()
    tta = {"config": None, "on": True, "off": False}[args.tta]
    predict_folder(args.run_dir, args.input_root, args.output_dir, folds=args.folds or "all", tta=tta)


if __name__ == "__main__":
    main()
