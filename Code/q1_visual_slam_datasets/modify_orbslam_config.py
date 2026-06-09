#!/usr/bin/env python3
"""
COMP0249 CW2 - Q1(b): Generate ORB-SLAM2 YAML configs with different feature counts.

ORB-SLAM2 reads nFeatures from the YAML at startup — no recompile needed.
We generate one config per feature count so the experiment script can just
swap the YAML when calling mono_kitti / mono_tum.

Usage:
    python modify_orbslam_config.py --config KITTI04-12.yaml --features 1000 500 200
"""

import argparse
import os
import re


def modify_orb_features(config_path: str, n_features: int, output_path: str = None) -> str:
    """Write a copy of config_path with ORBextractor.nFeatures set to n_features."""
    if output_path is None:
        base, ext = os.path.splitext(config_path)
        output_path = f"{base}_feat{n_features}{ext}"

    with open(config_path, 'r') as f:
        content = f.read()

    # Replace the existing nFeatures value
    new_content = re.sub(
        r'(ORBextractor\.nFeatures\s*:\s*)\d+',
        rf'\g<1>{n_features}',
        content
    )

    if new_content == content:
        # Parameter wasn't in the file — append it
        print(f"[WARN] ORBextractor.nFeatures not found in {config_path}, appending.")
        new_content += f"\nORBextractor.nFeatures: {n_features}\n"

    with open(output_path, 'w') as f:
        f.write(new_content)

    print(f"[OK] {n_features} features -> {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True,
                        help="Base YAML config (e.g. KITTI04-12.yaml)")
    parser.add_argument('--features', type=int, nargs='+', default=[1000, 500, 200],
                        help="Feature counts to generate configs for")
    args = parser.parse_args()

    if not os.path.isfile(args.config):
        print(f"[ERROR] Config not found: {args.config}")
        return

    for n in args.features:
        modify_orb_features(args.config, n)


if __name__ == '__main__':
    main()
