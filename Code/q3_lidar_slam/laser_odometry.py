#!/usr/bin/env python3
"""
COMP0249 CW2 - Q3(b): Laser Odometry and Mapping

ICP-based 2D scan matching. For each consecutive pair of scans we find
the rigid body transform (rotation + translation) that best aligns them,
then compose these incremental transforms into a global trajectory.

The parameter sweep tests four knobs from the coursework spec:
  1. Maximum range       — filters out long-range noisy readings
  2. Angular resolution  — subsamples the scan to fewer beams
  3. Voxel grid size     — merges nearby points before matching
  4. Scan rate           — processes only every n-th scan

Usage:
    python laser_odometry.py --data indoor1.csv --output results/q3/indoor1
    python laser_odometry.py --data indoor1.csv --output results/q3 --sweep
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple
from dataclasses import dataclass

from lidar_loader import LidarLoader, LidarScan


# ---------------------------------------------------------------------------
# ICP (point-to-point, 2D)
# ---------------------------------------------------------------------------

def icp_2d(source, target, max_iter=50, tolerance=1e-4,
           max_correspondence_dist=1.0):
    """
    Iterative Closest Point for 2D point clouds.
    Alternates between finding nearest-neighbour correspondences and
    computing the optimal rigid transform via SVD until convergence.
    Returns (T_3x3, final_mean_error).
    """
    from scipy.spatial import KDTree
    src = source.copy()
    T_accum = np.eye(3)
    mean_dist = 1e9

    for _ in range(max_iter):
        dists, idx = KDTree(target).query(src, k=1)
        valid = dists < max_correspondence_dist
        if valid.sum() < 10:
            break

        T_step = _svd_transform(src[valid], target[idx[valid]])
        T_accum = T_step @ T_accum
        R, t = T_step[:2, :2], T_step[:2, 2]
        src = (R @ src.T).T + t

        mean_dist = dists[valid].mean()
        if mean_dist < tolerance:
            break

    return T_accum, mean_dist


def _svd_transform(src, tgt):
    """Compute the optimal 2D rigid transform from src→tgt via SVD."""
    mu_s, mu_t = src.mean(0), tgt.mean(0)
    H = (src - mu_s).T @ (tgt - mu_t)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    T = np.eye(3)
    T[:2, :2] = R
    T[:2, 2]  = mu_t - R @ mu_s
    return T


def apply_transform_2d(pts, T):
    h = np.ones((len(pts), 3)); h[:, :2] = pts
    return (T @ h.T).T[:, :2]


def voxel_downsample(points, voxel_size):
    """Merge points that fall in the same voxel cell — reduces density and noise."""
    if len(points) == 0:
        return points
    idx = np.unique(np.floor(points / voxel_size).astype(int),
                    axis=0, return_index=True)[1]
    return points[idx]


# ---------------------------------------------------------------------------
# Occupancy grid (log-odds update with Bresenham ray casting)
# ---------------------------------------------------------------------------

class OccupancyGrid:
    def __init__(self, resolution=0.05, size=60.0):
        self.res = resolution
        self.n   = int(size / resolution)
        self.origin = np.array([size / 2, size / 2])
        self.log_odds = np.zeros((self.n, self.n), dtype=np.float32)
        # Log-odds increments: occupied and free
        self.L_OCC, self.L_FREE = 0.9, -0.7
        self.L_MAX, self.L_MIN  = 5.0, -5.0

    def _to_grid(self, xy):
        return ((xy + self.origin) / self.res).astype(int)

    def update_scan(self, robot_pos, scan_pts):
        """For each ray, mark cells along it as free and the endpoint as occupied."""
        r0, c0 = self._to_grid(robot_pos)
        for pt in scan_pts:
            r1, c1 = self._to_grid(pt)
            for r, c in _bresenham(r0, c0, r1, c1)[:-1]:
                if 0 <= r < self.n and 0 <= c < self.n:
                    self.log_odds[r, c] = np.clip(
                        self.log_odds[r, c] + self.L_FREE, self.L_MIN, self.L_MAX)
            if 0 <= r1 < self.n and 0 <= c1 < self.n:
                self.log_odds[r1, c1] = np.clip(
                    self.log_odds[r1, c1] + self.L_OCC, self.L_MIN, self.L_MAX)

    def get_probability_map(self):
        return 1.0 / (1.0 + np.exp(-self.log_odds))

    def save(self, path):
        np.save(path, self.log_odds)
        print(f"[OK] Saved occupancy grid -> {path}")


def _bresenham(r0, c0, r1, c1):
    """Bresenham line — returns list of (row, col) cells between two points."""
    cells = []
    dr, dc = abs(r1-r0), abs(c1-c0)
    sr, sc = (1 if r0<r1 else -1), (1 if c0<c1 else -1)
    err = dr - dc
    r, c = r0, c0
    while True:
        cells.append((r, c))
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dc: err -= dc; r += sr
        if e2 < dr:  err += dr; c += sc
        if len(cells) > 2000: break
    return cells


# ---------------------------------------------------------------------------
# Odometry pipeline
# ---------------------------------------------------------------------------

@dataclass
class OdometryConfig:
    max_range:   float = 8.0
    angular_nth: int   = 1
    voxel_size:  float = 0.05
    scan_skip:   int   = 1
    icp_max_dist: float = 1.0
    icp_max_iter: int   = 50
    grid_resolution: float = 0.05
    grid_size:   float = 60.0
    label:       str   = "default"


def run_odometry(scans, cfg, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    trajectory, map_points = [], []
    T_world = np.eye(3)
    prev_pts = None
    grid = OccupancyGrid(cfg.grid_resolution, cfg.grid_size)

    processed = 0
    for i, scan in enumerate(scans):
        if i % cfg.scan_skip != 0:
            continue
        if cfg.angular_nth > 1:
            scan = scan.downsample(cfg.angular_nth)

        pts = scan.to_cartesian(max_range=cfg.max_range)
        if cfg.voxel_size > 0 and len(pts):
            pts = voxel_downsample(pts, cfg.voxel_size)
        if len(pts) < 20:
            continue

        if prev_pts is None:
            prev_pts = pts
            x, y = T_world[:2, 2]
            th = np.arctan2(T_world[1, 0], T_world[0, 0])
            trajectory.append((scan.timestamp, x, y, th))
            processed += 1
            continue

        T_step, _ = icp_2d(pts, prev_pts,
                            max_iter=cfg.icp_max_iter,
                            max_correspondence_dist=cfg.icp_max_dist)
        T_world = T_world @ np.linalg.inv(T_step)

        x, y = T_world[:2, 2]
        th = np.arctan2(T_world[1, 0], T_world[0, 0])
        trajectory.append((scan.timestamp, x, y, th))

        pts_w = apply_transform_2d(pts, T_world)
        map_points.append(pts_w)
        grid.update_scan(np.array([x, y]), pts_w)

        prev_pts = pts
        processed += 1
        if processed % 100 == 0:
            print(f"  Processed {processed}/{len(scans)}, pos=({x:.2f}, {y:.2f})")

    print(f"  Total scans processed: {processed}")
    traj = np.array(trajectory)

    # Save trajectory
    traj_path = os.path.join(output_dir, f"trajectory_{cfg.label}.txt")
    np.savetxt(traj_path, traj, header="timestamp x y theta", comments='# ')
    print(f"[OK] Trajectory saved -> {traj_path}")

    # Save point cloud
    if map_points:
        pts_all = np.vstack(map_points)
        pts_path = os.path.join(output_dir, f"pointcloud_{cfg.label}.npy")
        np.save(pts_path, pts_all)
        print(f"[OK] Point cloud saved -> {pts_path} ({len(pts_all)} points)")

    grid.save(os.path.join(output_dir, f"occupancy_{cfg.label}.npy"))
    _plot(traj, map_points, grid, cfg, output_dir)

    # Closure error
    if len(traj) > 1:
        err = np.linalg.norm(traj[-1, 1:3] - traj[0, 1:3])
        print(f"  Loop closure error (before optimisation): {err:.4f} m")

    return traj, map_points, grid


def _plot(traj, map_points, grid, cfg, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"LiDAR Odometry: {cfg.label}", fontsize=11)

    ax = axes[0]
    ax.plot(traj[:, 1], traj[:, 2], 'b-', lw=1)
    ax.plot(traj[0, 1], traj[0, 2], 'go', ms=10, label='Start')
    ax.plot(traj[-1, 1], traj[-1, 2], 'r^', ms=10, label='End')
    ax.set_title('Trajectory'); ax.set_aspect('equal')
    ax.grid(True); ax.legend()

    ax = axes[1]
    if map_points:
        pts = np.vstack(map_points)
        ax.scatter(pts[:, 0], pts[:, 1], s=0.5, c='k', alpha=0.3)
    ax.set_title('Point Cloud'); ax.set_aspect('equal'); ax.grid(True)

    ax = axes[2]
    prob = grid.get_probability_map()
    half = grid.n * grid.res / 2
    ax.imshow(prob, cmap='gray_r', origin='lower', vmin=0.3, vmax=0.7,
              extent=[-half, half, -half, half])
    ax.set_title('Occupancy Grid')

    plt.tight_layout()
    path = os.path.join(output_dir, f"results_{cfg.label}.pdf")
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"[OK] Plot saved -> {path}")


# ---------------------------------------------------------------------------
# Parameter sweep (Q3b)
# ---------------------------------------------------------------------------

def run_sweep(data_path, output_dir, topic="/scan"):
    print(f"\n{'='*60}\n PARAMETER SWEEP: {data_path}\n{'='*60}")
    experiments = [
        OdometryConfig(max_range=8.0,   label="maxrange_8m"),
        OdometryConfig(max_range=12.0,  label="maxrange_12m_sensor_max"),
        OdometryConfig(angular_nth=1,   label="ang_full"),
        OdometryConfig(angular_nth=2,   label="ang_every2nd"),
        OdometryConfig(angular_nth=3,   label="ang_every3rd"),
        OdometryConfig(voxel_size=0.05, label="voxel_5cm"),
        OdometryConfig(voxel_size=0.10, label="voxel_10cm"),
        OdometryConfig(scan_skip=1,     label="scanrate_full"),
        OdometryConfig(scan_skip=2,     label="scanrate_skip1in2"),
        OdometryConfig(scan_skip=3,     label="scanrate_skip2in3"),
    ]
    for cfg in experiments:
        print(f"\n--- Config: {cfg.label} ---")
        scans = list(LidarLoader(data_path, topic=topic))
        run_odometry(scans, cfg, os.path.join(output_dir, cfg.label))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--topic', default='/scan')
    parser.add_argument('--max-range', type=float, default=8.0)
    parser.add_argument('--angular-nth', type=int, default=1)
    parser.add_argument('--voxel', type=float, default=0.05)
    parser.add_argument('--scan-skip', type=int, default=1)
    parser.add_argument('--sweep', action='store_true',
                        help="Run all Q3(b) parameter experiments")
    args = parser.parse_args()

    if args.sweep:
        run_sweep(args.data, args.output, args.topic)
    else:
        scans = list(LidarLoader(args.data, topic=args.topic))
        cfg = OdometryConfig(max_range=args.max_range,
                             angular_nth=args.angular_nth,
                             voxel_size=args.voxel,
                             scan_skip=args.scan_skip,
                             label="default")
        run_odometry(scans, cfg, args.output)


if __name__ == '__main__':
    main()
