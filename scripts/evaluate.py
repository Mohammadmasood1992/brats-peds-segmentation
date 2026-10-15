"""Score a folder of {case}.nii.gz predictions against {label-root}/{case}/{case}-seg.nii.gz.

Labels must be in the same (aligned) space as the predictions; for realigned subjects, apply the saved
rigid transform (<aligned root>/reports/transforms) to the labels first.

Example:
    python scripts/evaluate.py --predictions predictions/Validation --label-root data/Validation_labels_aligned
"""

import argparse
from pathlib import Path

from brats_peds_seg.metrics import summarize_metrics
from brats_peds_seg.reporting import evaluate_folder


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=None, help="default: <predictions>/metrics.csv")
    args = parser.parse_args()
    metrics = evaluate_folder(args.predictions, args.label_root, args.output_csv)
    if len(metrics):
        print(summarize_metrics(metrics).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
