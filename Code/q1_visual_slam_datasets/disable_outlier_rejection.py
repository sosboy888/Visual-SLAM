#!/usr/bin/env python3
"""
COMP0249 CW2 - Q1(c): Disable outlier rejection in ORB-SLAM2.

ORB-SLAM2 filters bad feature matches in two places:
  1. Initializer.cc — CheckHomography / CheckFundamental reject matches
     with high reprojection error during map initialisation.
  2. Optimizer.cc — PoseOptimization uses a Huber kernel and discards
     observations above a chi-squared threshold (~5.991 at 95% confidence).

To turn both off we either zero out the scores returned by the check
functions, or raise the chi2 thresholds to 1e9 so nothing ever gets rejected.

Usage:
    python disable_outlier_rejection.py --orbslam-src ~/ORB_SLAM2/src --disable
    python disable_outlier_rejection.py --orbslam-src ~/ORB_SLAM2/src --restore
"""

import argparse
import os
import re
import shutil


PATCHES = [
    # In FindHomography, the score from CheckHomography decides which matches
    # are inliers. Setting score=0 effectively accepts all matches.
    {
        "file": "Initializer.cc",
        "description": "Disable CheckHomography — accept all matches as inliers",
        "search": r"(score = CheckHomography\([^;]+;)",
        "replace": r"// [PATCH Q1c] CheckHomography disabled\n        score = 0.0f;",
        "target_var": "CheckHomography",
    },
    # Same for the fundamental matrix path
    {
        "file": "Initializer.cc",
        "description": "Disable CheckFundamental — accept all matches as inliers",
        "search": r"(score = CheckFundamental\([^;]+;)",
        "replace": r"// [PATCH Q1c] CheckFundamental disabled\n        score = 0.0f;",
        "target_var": "CheckFundamental",
    },
    # PoseOptimization uses chi2 thresholds to flag outlier observations.
    # Raising them to 1e9 means no observation is ever rejected.
    {
        "file": "Optimizer.cc",
        "description": "Raise monocular chi2 thresholds to disable outlier culling",
        "search": r"(const float chi2Mono\[4\] = \{)([^}]+)(\};)",
        "replace": r"\g<1>1e9f,1e9f,1e9f,1e9f\g<3>  // [PATCH Q1c] thresholds raised",
        "target_var": "chi2Mono",
    },
    {
        "file": "Optimizer.cc",
        "description": "Raise stereo chi2 thresholds (not used in mono but patched for completeness)",
        "search": r"(const float chi2Stereo\[4\] = \{)([^}]+)(\};)",
        "replace": r"\g<1>1e9f,1e9f,1e9f,1e9f\g<3>  // [PATCH Q1c] thresholds raised",
        "target_var": "chi2Stereo",
    },
]


def apply_patches(src_dir: str, restore: bool = False):
    for patch in PATCHES:
        filepath = os.path.join(src_dir, patch["file"])
        backup = filepath + ".outlier.bak"

        if not os.path.isfile(filepath):
            print(f"[SKIP] {patch['file']} not found at {filepath}")
            continue

        if restore:
            if os.path.isfile(backup):
                shutil.copy2(backup, filepath)
                print(f"[RESTORED] {patch['file']}")
            else:
                print(f"[WARN] No backup found for {patch['file']}")
            continue

        if not os.path.isfile(backup):
            shutil.copy2(filepath, backup)
            print(f"[BACKUP] {patch['file']} -> {backup}")

        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()

        new_content, count = re.subn(patch["search"], patch["replace"], content)

        if count == 0:
            print(f"[WARN] Pattern not matched in {patch['file']}: {patch['description']}")
            print(f"       May need to manually comment out '{patch['target_var']}'")
        else:
            with open(filepath, 'w') as f:
                f.write(new_content)
            print(f"[PATCHED] {patch['file']} ({count} match): {patch['description']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--orbslam-src', required=True,
                        help="Path to ORB_SLAM2/src directory")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--disable', action='store_true')
    group.add_argument('--restore', action='store_true')
    args = parser.parse_args()

    if not os.path.isdir(args.orbslam_src):
        print(f"[ERROR] Directory not found: {args.orbslam_src}")
        return

    action = "RESTORING" if args.restore else "DISABLING outlier rejection"
    print(f"\n=== {action} in {args.orbslam_src} ===\n")
    apply_patches(args.orbslam_src, restore=args.restore)
    print("\n[NEXT] Rebuild: cmake --build ~/ORB_SLAM2/build --parallel 12")


if __name__ == '__main__':
    main()
