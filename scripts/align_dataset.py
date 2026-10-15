"""Rigid orientation alignment of the dataset, in three steps.

1. screen          contact sheets of every subject for a visual check of gross (90/180 degree) rotations
2. rank-templates  rank template candidates (typical orientation, small tumor, sharp t1n)
3. align           register every subject to the chosen template, resample rotated subjects, re-run
                   SynthStrip on them and report tumor coverage by the brain mask

Example (values used for BraTS-PEDs-v1):
    python scripts/align_dataset.py screen --data-root data/BraTS-PEDs-v1
    python scripts/align_dataset.py rank-templates --data-root data/BraTS-PEDs-v1
    python scripts/align_dataset.py align --data-root data/BraTS-PEDs-v1 --output-root data/BraTS-PEDs-v1_rigid_aligned \
        --template BraTS-PED-00037-000 --gross-rotation BraTS-PED-00042-000:-90,0,0
"""

import argparse
from pathlib import Path

from brats_peds_seg import alignment

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_gross_rotation(value):
    case, angles = value.split(":")
    return case, (tuple(float(a) for a in angles.split(",")),)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    screen = sub.add_parser("screen", help="contact sheets for visual orientation review")
    screen.add_argument("--data-root", type=Path, required=True)
    screen.add_argument("--output-dir", type=Path, default=Path("reports/orientation_screen"))

    rank = sub.add_parser("rank-templates", help="rank template candidates")
    rank.add_argument("--data-root", type=Path, required=True)
    rank.add_argument("--report-dir", type=Path, default=Path("reports/template_selection"))
    rank.add_argument("--provisional-template", default="BraTS-PED-00190-000")
    rank.add_argument("--workers", type=int, default=16)

    align = sub.add_parser("align", help="register, resample and re-skull-strip")
    align.add_argument("--data-root", type=Path, required=True, help="original dataset with SynthStrip masks")
    align.add_argument("--output-root", type=Path, required=True)
    align.add_argument("--template", required=True)
    align.add_argument("--gross-rotation", action="append", default=[], type=parse_gross_rotation,
                       metavar="CASE:RX,RY,RZ", help="visually identified start rotation in degrees (repeatable)")
    align.add_argument("--threshold-deg", type=float, default=10.0, help="resample subjects rotated more than this")
    align.add_argument("--workers", type=int, default=16)
    align.add_argument("--link-mode", choices=["hardlink", "copy"], default="hardlink",
                       help="how unchanged subjects are placed in the output (hardlink needs the same filesystem)")
    align.add_argument("--skip-skullstrip", action="store_true")
    align.add_argument("--synthstrip-script", type=Path, default=REPO_ROOT / "docker" / "synthstrip" / "run_synthstrip.sh")
    args = parser.parse_args()

    if args.command == "screen":
        alignment.orientation_screen(args.data_root, args.output_dir)
    elif args.command == "rank-templates":
        candidates = alignment.rank_template_candidates(args.data_root, args.report_dir, args.provisional_template, args.workers)
        print(candidates.round(3).to_string())
        print("Prefer a candidate with sharpness_ratio >= 0.8 and no visible mass; inspect it before choosing.")
    else:
        result = alignment.align_dataset(args.data_root, args.output_root, args.template, dict(args.gross_rotation),
                                         args.threshold_deg, args.workers, args.link_mode)
        realigned = result.loc[result["realigned"]]
        print(realigned[["case_id", "split", "rotation_deg", "translation_mm", "start_rotation_deg"]].round(1).to_string(index=False))
        if not args.skip_skullstrip and len(realigned):
            case_dirs = [args.output_root / split / case for case, split in zip(realigned["case_id"], realigned["split"])]
            alignment.run_synthstrip(case_dirs, args.synthstrip_script)
            coverage = alignment.tumor_mask_coverage(args.data_root, args.output_root)
            coverage.to_csv(args.output_root / "reports" / "tumor_mask_coverage.csv", index=False)
            for name in ["original", "aligned"]:
                values = coverage[f"tumor_outside_pct_{name}"]
                print(f"{name:>8}: subjects with >1% tumor outside the brain mask: {(values > 1).sum()}, >5%: {(values > 5).sum()}, "
                      f"max {values.max():.1f}%")


if __name__ == "__main__":
    main()
