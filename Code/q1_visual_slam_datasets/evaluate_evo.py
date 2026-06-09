#!/usr/bin/env python3
"""
COMP0249 CW2 - Q1: Run EVO evaluation on all Q1 experiment trajectories.

Computes ATE (Absolute Trajectory Error) for each experiment, saves
individual plots and comparison plots for the report.

Usage:
    python evaluate_evo.py --all
    python evaluate_evo.py --dataset kitti07
    python evaluate_evo.py --dataset tum
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


RESULTS_ROOT = Path("results/q1")

# Ground truth paths and trajectory paths for each experiment.
# kitti07 uses TUM format (converted from KITTI matrix format).
# tum uses native TUM groundtruth.txt.
EXPERIMENTS = {
    "kitti07": {
        "gt": "dataset/poses/07_tum.txt",
        "fmt": "tum",
        "align_flags": ["--align", "--correct_scale"],  # Sim(3) for monocular
        "sequences": {
            "baseline":       "kitti07/baseline/KeyFrameTrajectory.txt",
            "feat_1000":      "kitti07/feat_1000/KeyFrameTrajectory.txt",
            "feat_500":       "kitti07/feat_500/KeyFrameTrajectory.txt",
            "feat_200":       "kitti07/feat_200/KeyFrameTrajectory.txt",
            "no_outlier":     "kitti07/no_outlier/KeyFrameTrajectory.txt",
            "no_loopclosure": "kitti07/no_loopclosure/KeyFrameTrajectory.txt",
        },
    },
    "tum": {
        "gt": "rgbd_dataset_freiburg3_long_office_household/rgbd_dataset_freiburg3_long_office_household/groundtruth.txt",
        "fmt": "tum",
        "align_flags": ["-va"],  # SE(3) alignment
        "sequences": {
            "baseline":       "tum/baseline/KeyFrameTrajectory.txt",
            "feat_1000":      "tum/feat_1000/KeyFrameTrajectory.txt",
            "feat_500":       "tum/feat_500/KeyFrameTrajectory.txt",
            "feat_200":       "tum/feat_200/KeyFrameTrajectory.txt",
            "no_outlier":     "tum/no_outlier/KeyFrameTrajectory.txt",
            "no_loopclosure": "tum/no_loopclosure/KeyFrameTrajectory.txt",
        },
    },
}

# Which experiments to group together for comparison plots
COMPARE_GROUPS = {
    "features":     ["baseline", "feat_1000", "feat_500", "feat_200"],
    "outlier":      ["baseline", "no_outlier"],
    "loopclosure":  ["baseline", "no_loopclosure"],
}


def run_evo_ape(gt, est, fmt, align_flags, out_prefix):
    """Run evo_ape and save the result zip and PDF plot."""
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
    zip_out  = out_prefix + ".zip"
    plot_out = out_prefix + "_ape.pdf"

    cmd = ["evo_ape", fmt, gt, est] + align_flags + [
        "--t_max_diff", "0.5",
        "--save_results", zip_out,
        "--save_plot", plot_out,
    ]
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  [ERROR] {result.stderr.strip()}")
        return None

    # Pull the RMSE out of stdout so we can print a summary table
    rmse = None
    for line in result.stdout.splitlines():
        if "rmse" in line.lower():
            try:
                rmse = float(line.split()[-1])
            except ValueError:
                pass

    print(f"  RMSE: {rmse:.4f} m" if rmse else "  RMSE: N/A")
    return {"zip": zip_out, "plot": plot_out, "rmse": rmse}


def make_comparison_plot(dataset_key, group_name, exp_names):
    """Use evo_res to overlay multiple experiment results on one plot."""
    zip_files = []
    for name in exp_names:
        p = RESULTS_ROOT / f"{dataset_key}_{name}.zip"
        if p.exists():
            zip_files.append(str(p))

    if len(zip_files) < 2:
        print(f"  [SKIP] {group_name}: not enough results to compare")
        return

    out = str(RESULTS_ROOT / f"{dataset_key}_{group_name}_comparison.pdf")
    cmd = ["evo_res"] + zip_files + ["--use_filenames", "--ignore_title",
                                      "--save_plot", out]
    print(f"  [compare] {group_name}: {' '.join(cmd)}")
    subprocess.run(cmd, input="y\n", text=True)


def evaluate_dataset(key):
    cfg = EXPERIMENTS[key]
    gt  = cfg["gt"]
    fmt = cfg["fmt"]
    align = cfg["align_flags"]
    summary = {}

    print(f"\n{'='*60}\n  {key}\n{'='*60}")

    for name, rel_path in cfg["sequences"].items():
        est = str(RESULTS_ROOT / rel_path)
        if not os.path.isfile(est):
            print(f"  [SKIP] {name}: no trajectory at {est}")
            continue
        if not os.path.isfile(gt):
            print(f"  [SKIP] {name}: ground truth not found at {gt}")
            continue

        print(f"\n  --- {name} ---")
        prefix = str(RESULTS_ROOT / f"{key}_{name}")
        res = run_evo_ape(gt, est, fmt, align, prefix)
        if res:
            summary[name] = res

    # Comparison plots for features / outlier / loop closure groups
    for group_name, exp_names in COMPARE_GROUPS.items():
        make_comparison_plot(key, group_name, exp_names)

    # Print summary table
    print(f"\n  {'Experiment':<20} {'RMSE ATE (m)':>12}")
    print(f"  {'-'*33}")
    for name, r in summary.items():
        val = f"{r['rmse']:.4f}" if r["rmse"] else "failed"
        print(f"  {name:<20} {val:>12}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=list(EXPERIMENTS.keys()))
    parser.add_argument('--all', action='store_true')
    args = parser.parse_args()

    # Make sure evo is installed
    if subprocess.run(["evo_ape", "--help"], capture_output=True).returncode not in [0, 2]:
        print("[ERROR] evo not found. Install: pip install evo --upgrade")
        sys.exit(1)

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    if args.all:
        for key in EXPERIMENTS:
            evaluate_dataset(key)
    elif args.dataset:
        evaluate_dataset(args.dataset)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
