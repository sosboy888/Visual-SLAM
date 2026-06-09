#!/usr/bin/env python3
"""
COMP0249 CW2 - Q1(d): Disable loop closure in ORB-SLAM2.

ORB-SLAM2 runs loop closure in a separate thread. The cleanest way to
turn it off is to make DetectLoop() always return false — the rest of
the system stays intact, we just never accept a loop candidate.

Usage:
    python disable_loop_closure.py --orbslam-src ~/ORB_SLAM2/src --disable
    python disable_loop_closure.py --orbslam-src ~/ORB_SLAM2/src --restore
"""

import argparse
import os
import re
import shutil


# We patch DetectLoop() in LoopClosing.cc to return false immediately.
# A .bak backup is created before touching the file so we can restore later.
PATCHES = [
    {
        "file": "LoopClosing.cc",
        "description": "Force DetectLoop() to always return false",
        "search": r"(bool LoopClosing::DetectLoop\(\)\s*\{)",
        "replace": r"\g<1>\n    // [PATCH Q1d] Loop closure disabled for experiment\n    return false;\n",
        "target_var": "DetectLoop",
    },
]


def apply_patches(src_dir: str, restore: bool = False):
    for patch in PATCHES:
        filepath = os.path.join(src_dir, patch["file"])
        backup = filepath + ".loopclosure.bak"

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
            # Regex didn't match — the function signature may differ slightly
            # between ORB-SLAM2 versions. Add return false; manually if needed.
            print(f"[WARN] Pattern not matched in {patch['file']}.")
            print(f"       Manually add 'return false;' at the top of LoopClosing::DetectLoop()")
        else:
            with open(filepath, 'w') as f:
                f.write(new_content)
            print(f"[PATCHED] {patch['file']}: {patch['description']}")


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

    action = "RESTORING" if args.restore else "DISABLING loop closure"
    print(f"\n=== {action} in {args.orbslam_src} ===\n")
    apply_patches(args.orbslam_src, restore=args.restore)
    print("\n[NEXT] Rebuild: cmake --build ~/ORB_SLAM2/build --parallel 12")


if __name__ == '__main__':
    main()
