#!/usr/bin/env python3
"""
COMP0249 CW2 - Q2: Camera Calibration Helper

Two modes:
  1. Checkerboard — classic OpenCV calibration if you have a printed pattern.
  2. COLMAP — extract intrinsics from a COLMAP cameras.txt and write an
     ORB-SLAM2 YAML file. This is the mode we actually used.

Usage:
    python calibrate_camera.py --mode colmap \
        --colmap-cameras colmap_out/sparse_txt/cameras.txt \
        --output my_camera.yaml
"""

import argparse
import os
import glob
import numpy as np
from pathlib import Path


# ORB-SLAM2 expects this exact YAML structure — any missing field causes a crash
ORBSLAM2_YAML_TEMPLATE = """%YAML:1.0

Camera.type: "PinHole"

Camera.fx: {fx}
Camera.fy: {fy}
Camera.cx: {cx}
Camera.cy: {cy}

Camera.k1: {k1}
Camera.k2: {k2}
Camera.p1: {p1}
Camera.p2: {p2}

Camera.width: {width}
Camera.height: {height}
Camera.fps: {fps}
Camera.RGB: 1

ORBextractor.nFeatures: 2000
ORBextractor.scaleFactor: 1.2
ORBextractor.nLevels: 8
ORBextractor.iniThFAST: 20
ORBextractor.minThFAST: 7

Viewer.KeyFrameSize: 0.05
Viewer.KeyFrameLineWidth: 1
Viewer.GraphLineWidth: 0.9
Viewer.PointSize: 2
Viewer.CameraSize: 0.08
Viewer.CameraLineWidth: 3
Viewer.ViewpointX: 0
Viewer.ViewpointY: -0.7
Viewer.ViewpointZ: -1.8
Viewer.ViewpointF: 500
"""


def calibrate_checkerboard(images_dir, rows, cols, square_size, output):
    """Standard OpenCV checkerboard calibration."""
    try:
        import cv2
    except ImportError:
        print("[ERROR] pip install opencv-python")
        return

    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2) * square_size

    obj_points, img_points, img_size = [], [], None
    files = sorted(glob.glob(os.path.join(images_dir, "*.png")) +
                   glob.glob(os.path.join(images_dir, "*.jpg")))

    if not files:
        print(f"[ERROR] No images in {images_dir}")
        return

    for fpath in files:
        img = cv2.imread(fpath)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img_size is None:
            img_size = (gray.shape[1], gray.shape[0])
        ret, corners = cv2.findChessboardCorners(gray, (rows, cols), None)
        if ret:
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            obj_points.append(objp)
            img_points.append(corners)
            print(f"  [OK] {os.path.basename(fpath)}")
        else:
            print(f"  [SKIP] {os.path.basename(fpath)}")

    if len(obj_points) < 5:
        print(f"[ERROR] Only {len(obj_points)} valid frames — need at least 5")
        return

    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, img_size, None, None)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    k1, k2, p1, p2 = dist[0, 0], dist[0, 1], dist[0, 2], dist[0, 3]

    print(f"\nfx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
    write_orbslam_yaml(fx, fy, cx, cy, k1, k2, p1, p2, img_size[0], img_size[1], output)


def calibrate_from_colmap(cameras_txt, output):
    """
    Parse COLMAP cameras.txt and write ORB-SLAM2 YAML.
    COLMAP supports several camera models — we handle the common ones.
    We used SIMPLE_RADIAL (COLMAP's default for unknown cameras).
    """
    if not os.path.isfile(cameras_txt):
        print(f"[ERROR] Not found: {cameras_txt}")
        return

    with open(cameras_txt) as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if line.startswith('#') or not line:
            continue
        parts = line.split()
        model = parts[1]
        width, height = int(parts[2]), int(parts[3])
        params = [float(x) for x in parts[4:]]

        # Map COLMAP model params to fx, fy, cx, cy, distortion
        if model == "PINHOLE":
            fx, fy, cx, cy = params[:4]
            k1 = k2 = p1 = p2 = 0.0
        elif model in ("SIMPLE_RADIAL", "RADIAL"):
            fx = params[0]; cx, cy = params[1], params[2]
            fy = fx
            k1 = params[3] if len(params) > 3 else 0.0
            k2 = p1 = p2 = 0.0
        elif model == "OPENCV":
            fx, fy, cx, cy = params[:4]
            k1, k2, p1, p2 = params[4:8]
        else:
            # Unknown model — just take the first 4 params as fx,cx,cy
            fx = params[0]; cx, cy = params[1], params[2]
            fy = fx
            k1 = k2 = p1 = p2 = 0.0
            print(f"[WARN] Unknown model {model}, using first params")

        print(f"  Model: {model}, {width}x{height}")
        print(f"  fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
        write_orbslam_yaml(fx, fy, cx, cy, k1, k2, p1, p2, width, height, output)
        break  # use first camera only


def write_orbslam_yaml(fx, fy, cx, cy, k1, k2, p1, p2, width, height, output, fps=30):
    content = ORBSLAM2_YAML_TEMPLATE.format(
        fx=round(fx, 4), fy=round(fy, 4),
        cx=round(cx, 4), cy=round(cy, 4),
        k1=round(k1, 6), k2=round(k2, 6),
        p1=round(p1, 6), p2=round(p2, 6),
        width=width, height=height, fps=fps
    )
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w') as f:
        f.write(content)
    print(f"[OK] YAML written -> {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['checkerboard', 'colmap'], required=True)
    parser.add_argument('--images', help="Calibration images dir (checkerboard mode)")
    parser.add_argument('--rows', type=int, default=9)
    parser.add_argument('--cols', type=int, default=6)
    parser.add_argument('--square', type=float, default=0.025)
    parser.add_argument('--colmap-cameras', help="COLMAP cameras.txt path")
    parser.add_argument('--output', default='camera_orbslam.yaml')
    args = parser.parse_args()

    if args.mode == 'checkerboard':
        calibrate_checkerboard(args.images, args.rows, args.cols, args.square, args.output)
    else:
        calibrate_from_colmap(args.colmap_cameras, args.output)


if __name__ == '__main__':
    main()
