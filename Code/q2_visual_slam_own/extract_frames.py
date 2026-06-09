#!/usr/bin/env python3
"""
COMP0249 CW2 - Q2: Video Frame Extractor

Pulls frames out of a video file and writes them in TUM format
so they can be fed into ORB-SLAM2's mono_tum and COLMAP.

The tricky part: OpenCV names extracted frames by their timestamp
in the source video (e.g. 9.941667.png). When sorted alphabetically
this order is wrong (9 > 0.1 lexicographically). After extraction,
rename all frames to zero-padded sequential numbers and rewrite rgb.txt.

Usage:
    python extract_frames.py --video indoor.MOV --output data/indoor --every-nth 3
"""

import argparse
import os
import sys
from pathlib import Path


def extract_frames(video_path, output_dir, target_fps=None,
                   every_nth=1, max_frames=None):
    try:
        import cv2
    except ImportError:
        print("[ERROR] pip install opencv-python")
        sys.exit(1)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    rgb_dir = os.path.join(output_dir, "rgb")
    Path(rgb_dir).mkdir(exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {video_path}")
        sys.exit(1)

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {video_path}")
    print(f"  Source FPS:    {src_fps:.2f}")
    print(f"  Total frames:  {total}")

    if target_fps is not None and target_fps > 0:
        every_nth = max(1, int(round(src_fps / target_fps)))

    print(f"  Extracting every {every_nth}th frame (~{src_fps/every_nth:.1f} fps)")

    timestamps = []
    saved = 0
    frame_idx = 0
    frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % every_nth == 0:
            ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            fname = f"{ts:.6f}.png"
            cv2.imwrite(os.path.join(rgb_dir, fname), frame)
            timestamps.append((ts, fname))
            saved += 1
            if max_frames and saved >= max_frames:
                break
        frame_idx += 1

    cap.release()
    print(f"[OK] {saved} frames -> {rgb_dir}")

    # Write rgb.txt for mono_tum
    rgb_txt = os.path.join(output_dir, "rgb.txt")
    with open(rgb_txt, 'w') as f:
        f.write("# timestamp filename\n")
        for ts, fname in timestamps:
            f.write(f"{ts:.6f} rgb/{fname}\n")
    print(f"[OK] rgb.txt -> {rgb_txt}")

    return output_dir, timestamps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--fps', type=float, default=None,
                        help="Target FPS (overrides --every-nth)")
    parser.add_argument('--every-nth', type=int, default=1,
                        help="Keep every n-th frame")
    parser.add_argument('--max-frames', type=int, default=None)
    args = parser.parse_args()

    extract_frames(args.video, args.output,
                   target_fps=args.fps,
                   every_nth=args.every_nth,
                   max_frames=args.max_frames)


if __name__ == '__main__':
    main()
