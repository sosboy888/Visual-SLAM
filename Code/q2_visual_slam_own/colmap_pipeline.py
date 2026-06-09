#!/usr/bin/env python3
"""
COMP0249 CW2 - Q2: COLMAP Pipeline + Trajectory Extractor

Runs the full COLMAP SfM pipeline on a frame sequence and converts
the resulting camera poses to TUM format for comparison with ORB-SLAM2.

We use sequential matching (not exhaustive) because our input is a video —
consecutive frames share most of their features, so sequential overlap
of ~10 frames is plenty. Loop detection is disabled because it needs a
large vocabulary tree file we don't have.

Usage:
    python colmap_pipeline.py --images data/indoor/rgb --output colmap_out/indoor
    python colmap_pipeline.py --images data/indoor/rgb --output colmap_out/indoor --use-gpu
"""

import argparse
import os
import subprocess
import sys
import numpy as np
from pathlib import Path


def run_cmd(cmd):
    print(f"\n  $ {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[ERROR] Command failed (code {result.returncode})")
        sys.exit(1)


def run_colmap_pipeline(images_dir, output_dir, use_gpu=False):
    """Feature extraction → sequential matching → sparse mapping → TXT export."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    db     = os.path.join(output_dir, "database.db")
    sparse = os.path.join(output_dir, "sparse")
    Path(sparse).mkdir(exist_ok=True)
    gpu = "1" if use_gpu else "0"

    print("\n=== Step 1: Feature Extraction ===")
    run_cmd(["colmap", "feature_extractor",
             "--database_path", db,
             "--image_path", images_dir,
             "--ImageReader.single_camera", "1",
             "--SiftExtraction.use_gpu", gpu])

    print("\n=== Step 2: Sequential Matching ===")
    run_cmd(["colmap", "sequential_matcher",
             "--database_path", db,
             "--SiftMatching.use_gpu", gpu,
             "--SequentialMatching.overlap", "10",
             "--SequentialMatching.loop_detection", "0"])

    print("\n=== Step 3: Sparse Reconstruction ===")
    run_cmd(["colmap", "mapper",
             "--database_path", db,
             "--image_path", images_dir,
             "--output_path", sparse,
             "--Mapper.ba_global_max_num_iterations", "20",
             "--Mapper.ba_local_max_num_iterations", "10",
             "--Mapper.max_num_models", "1"])

    # Find the reconstruction — COLMAP outputs to sparse/0/
    model_path = os.path.join(sparse, "0")
    if not os.path.isdir(model_path):
        subdirs = sorted(d for d in os.listdir(sparse)
                         if os.path.isdir(os.path.join(sparse, d)))
        if not subdirs:
            print("[ERROR] No reconstruction in sparse/. Check COLMAP output.")
            return None
        model_path = os.path.join(sparse, subdirs[0])

    print("\n=== Step 4: Convert to TXT ===")
    txt_dir = os.path.join(output_dir, "sparse", "txt")
    Path(txt_dir).mkdir(exist_ok=True)
    run_cmd(["colmap", "model_converter",
             "--input_path", model_path,
             "--output_path", txt_dir,
             "--output_type", "TXT"])

    print(f"\n[OK] Done. TXT model at {txt_dir}")
    return txt_dir


def colmap_to_tum(images_txt, output_tum, fps=20.0):
    """
    Convert COLMAP images.txt to TUM trajectory format.

    COLMAP stores extrinsics as world-to-camera: given a point P_world,
    P_cam = R * P_world + t. To get the camera position in world space
    (what TUM format stores) we need P_world = R^T * (P_cam - t),
    so camera_pos_world = -R^T * t.
    """
    poses = []
    with open(images_txt) as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            parts = line.split()
            if len(parts) != 10:
                continue  # skip the points2D lines
            try:
                qw, qx, qy, qz = [float(parts[i]) for i in [1, 2, 3, 4]]
                tx, ty, tz      = [float(parts[i]) for i in [5, 6, 7]]
                name = parts[9]
                # Timestamp from sequential frame number in filename
                frame_num = int(os.path.splitext(name)[0])
                ts = frame_num / fps

                # Rotate world-to-camera to camera-to-world
                R = _quat_to_matrix(qw, qx, qy, qz)
                pos = -R.T @ np.array([tx, ty, tz])
                # Conjugate quaternion for inverse rotation
                q = np.array([-qx, -qy, -qz, qw])
                q /= np.linalg.norm(q)

                poses.append((ts, pos[0], pos[1], pos[2],
                              q[0], q[1], q[2], q[3]))
            except (ValueError, IndexError):
                pass

    poses.sort(key=lambda x: x[0])
    with open(output_tum, 'w') as f:
        for p in poses:
            f.write(f"{p[0]:.6f} {p[1]:.7f} {p[2]:.7f} {p[3]:.7f} "
                    f"{p[4]:.7f} {p[5]:.7f} {p[6]:.7f} {p[7]:.7f}\n")
    print(f"[OK] {len(poses)} COLMAP poses -> {output_tum}")
    return output_tum


def _quat_to_matrix(qw, qx, qy, qz):
    n = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
    qw, qx, qy, qz = qw/n, qx/n, qy/n, qz/n
    return np.array([
        [1-2*(qy**2+qz**2), 2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)  ],
        [2*(qx*qy+qz*qw),   1-2*(qx**2+qz**2),  2*(qy*qz-qx*qw) ],
        [2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw),    1-2*(qx**2+qy**2)],
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--use-gpu', action='store_true')
    parser.add_argument('--convert-only', metavar='IMAGES_TXT',
                        help="Skip COLMAP, just convert existing images.txt")
    parser.add_argument('--orbslam-traj',
                        help="ORB-SLAM2 trajectory for EVO comparison")
    parser.add_argument('--fps', type=float, default=20.0,
                        help="Frame rate used during extraction (for timestamp conversion)")
    args = parser.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)
    out_tum = os.path.join(args.output, "colmap_trajectory_tum.txt")

    if args.convert_only:
        colmap_to_tum(args.convert_only, out_tum, fps=args.fps)
    else:
        txt_dir = run_colmap_pipeline(args.images, args.output, args.use_gpu)
        if txt_dir:
            colmap_to_tum(os.path.join(txt_dir, "images.txt"), out_tum, fps=args.fps)

    if args.orbslam_traj and os.path.isfile(out_tum):
        prefix = os.path.join(args.output, "colmap_vs_orbslam")
        cmd = ["evo_ape", "tum", out_tum, args.orbslam_traj,
               "--align", "--correct_scale", "--t_max_diff", "2.0",
               "--save_plot", prefix + "_ape.pdf",
               "--save_results", prefix + "_ape.zip"]
        print(f"\n  $ {' '.join(cmd)}")
        subprocess.run(cmd)


if __name__ == '__main__':
    main()
