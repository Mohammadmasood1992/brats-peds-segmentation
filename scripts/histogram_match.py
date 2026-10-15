"""Brain-masked histogram matching of an (aligned) dataset to an automatically selected Training reference.

Example:
    python scripts/histogram_match.py --data-root data/BraTS-PEDs-v1_rigid_aligned \
        --output-root data/BraTS-PEDs-v1_rigid_aligned_histmatched
"""

import argparse
from pathlib import Path

from brats_peds_seg.histogram_matching import run_histogram_matching


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, required=True, help="dataset with {split}/{case}/{case}-t1n_mask.nii.gz")
    parser.add_argument("--output-root", type=Path, default=None,
                        help="default: <data-root>_histogram_matched_masked_ref_<reference case>")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    _, summary = run_histogram_matching(args.data_root, args.output_root, args.workers)
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
