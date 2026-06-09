#!/usr/bin/env python3
"""
COMP0249 CW2 - Q3: Full LiDAR SLAM Pipeline

Runs the complete sequence: load scans → ICP odometry → loop closure
detection → factor graph optimisation → plots.

Usage:
    python slam_pipeline.py --data indoor1.csv --output results/q3/indoor1 --label indoor1
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from lidar_loader import LidarLoader
from laser_odometry import (icp_2d, apply_transform_2d, voxel_downsample,
                             OdometryConfig, OccupancyGrid)
from loop_closure import LoopClosureDetector
from factor_graph import PoseGraph, optimise_pose_graph, plot_before_after


def run_full_pipeline(data_path, output_dir, label,
                      topic="/scan", max_range=8.0,
                      angular_nth=1, voxel_size=0.05, scan_skip=1):

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}\n SLAM Pipeline: {label}\n{'='*60}")

    # ── 1. Load ──────────────────────────────────────────────────────────
    print("\n[1/5] Loading scans...")
    scans = list(LidarLoader(data_path, topic=topic))
    print(f"  Loaded {len(scans)} scans")

    # ── 2. ICP Odometry ──────────────────────────────────────────────────
    print("\n[2/5] Running laser odometry (ICP)...")
    trajectory, map_points = [], []
    T_world  = np.eye(3)
    prev_pts = None
    grid = OccupancyGrid(resolution=0.05, size=60.0)
    detector = LoopClosureDetector(
        min_travel=5.0, min_time_gap=10.0,
        desc_threshold=0.25, icp_error_threshold=0.15,
        keyframe_distance=0.5, voxel_size=voxel_size)

    for i, scan in enumerate(scans):
        if i % scan_skip != 0:
            continue
        if angular_nth > 1:
            scan = scan.downsample(angular_nth)

        pts = scan.to_cartesian(max_range=max_range)
        if voxel_size > 0 and len(pts):
            pts = voxel_downsample(pts, voxel_size)
        if len(pts) < 20:
            continue

        if prev_pts is None:
            prev_pts = pts
            x, y = T_world[:2, 2]
            th = np.arctan2(T_world[1, 0], T_world[0, 0])
            trajectory.append((scan.timestamp, x, y, th))
            continue

        T_step, _ = icp_2d(pts, prev_pts, max_iter=50, max_correspondence_dist=1.0)
        T_world = T_world @ np.linalg.inv(T_step)

        x, y = T_world[:2, 2]
        th = np.arctan2(T_world[1, 0], T_world[0, 0])
        trajectory.append((scan.timestamp, x, y, th))

        pts_w = apply_transform_2d(pts, T_world)
        map_points.append(pts_w)
        grid.update_scan(np.array([x, y]), pts_w)

        detector.check(len(trajectory)-1, pts, np.array([x, y, th]), scan.timestamp)
        prev_pts = pts

        if i % 200 == 0:
            print(f"  Scan {i}/{len(scans)}, pos=({x:.2f},{y:.2f}), "
                  f"loops={len(detector.detected_loops)}")

    traj = np.array(trajectory)
    print(f"  Trajectory: {len(traj)} poses")
    print(f"  Loop closures detected: {len(detector.detected_loops)}")

    # Save raw odometry results
    np.savetxt(os.path.join(output_dir, f"trajectory_raw_{label}.txt"),
               traj, header="timestamp x y theta", comments='# ')
    err_before = float(np.linalg.norm(traj[-1, 1:3] - traj[0, 1:3]))
    print(f"  Closure error (raw odometry): {err_before:.4f} m")
    grid.save(os.path.join(output_dir, f"occupancy_raw_{label}.npy"))

    # ── 3. Save loop closures ─────────────────────────────────────────────
    print("\n[3/5] Saving loop closures...")
    lc_path = os.path.join(output_dir, f"loop_closures_{label}.npz")
    if detector.detected_loops:
        loops_data = [{'query_idx': lc.query_idx, 'ref_idx': lc.ref_idx,
                       'query_ts': lc.query_ts, 'ref_ts': lc.ref_ts,
                       'T_relative': lc.T_relative, 'icp_error': lc.icp_error}
                      for lc in detector.detected_loops]
        np.savez_compressed(lc_path, loops=loops_data)
        print(f"  Saved {len(loops_data)} loop closures -> {lc_path}")
    detector.plot_loop_summary(traj,
        os.path.join(output_dir, f"loop_closures_{label}.pdf"))

    # ── 4. Factor graph ───────────────────────────────────────────────────
    print("\n[4/5] Factor graph optimisation...")
    graph = PoseGraph()
    graph.poses = traj[:, 1:4].copy()
    graph.timestamps = traj[:, 0]
    graph.build_odometry_edges()
    for lc in detector.detected_loops:
        graph.add_loop_closure(lc.query_idx, lc.ref_idx, lc.T_relative)

    before_poses = graph.poses.copy()
    graph.poses = optimise_pose_graph(graph)
    err_after = float(np.linalg.norm(graph.poses[-1, :2] - graph.poses[0, :2]))
    print(f"  Closure error BEFORE: {err_before:.4f} m")
    print(f"  Closure error AFTER:  {err_after:.4f} m")

    np.savetxt(os.path.join(output_dir, f"trajectory_optimised_{label}.txt"),
               np.column_stack([graph.timestamps, graph.poses]),
               header="timestamp x y theta", comments='# ')

    with open(os.path.join(output_dir, f"closure_errors_{label}.txt"), 'w') as f:
        f.write(f"label: {label}\nbefore_m: {err_before:.6f}\n"
                f"after_m: {err_after:.6f}\n"
                f"n_loop_closures: {len(detector.detected_loops)}\n")

    # ── 5. Plots ──────────────────────────────────────────────────────────
    print("\n[5/5] Generating plots...")
    plot_before_after(before_poses, graph.poses, graph.loop_edges,
        os.path.join(output_dir, f"before_after_{label}.pdf"))
    _plot_full_summary(traj, graph.poses, map_points, grid,
                       detector.detected_loops, label, output_dir,
                       err_before, err_after)

    print(f"\n[DONE] {label} pipeline complete. Results in {output_dir}")
    return {'label': label, 'closure_before': err_before,
            'closure_after': err_after, 'n_loops': len(detector.detected_loops)}


def _plot_full_summary(traj_raw, traj_opt, map_points, grid,
                       loops, label, out_dir, err_before, err_after):
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f"SLAM Results: {label}", fontsize=14)

    ax = axes[0, 0]
    ax.plot(traj_raw[:, 1], traj_raw[:, 2], 'b-', lw=1)
    ax.plot(traj_raw[0, 1], traj_raw[0, 2], 'go', ms=10)
    ax.plot(traj_raw[-1, 1], traj_raw[-1, 2], 'r^', ms=10)
    ax.set_title(f'Raw Odometry\nClosure error: {err_before:.3f}m')
    ax.set_aspect('equal'); ax.grid(True)

    ax = axes[0, 1]
    ax.plot(traj_opt[:, 0], traj_opt[:, 1], 'g-', lw=1)
    ax.plot(traj_opt[0, 0], traj_opt[0, 1], 'go', ms=10)
    ax.plot(traj_opt[-1, 0], traj_opt[-1, 1], 'r^', ms=10)
    for lc in loops:
        i = min(lc.query_idx, len(traj_opt)-1)
        j = min(lc.ref_idx, len(traj_opt)-1)
        ax.plot([traj_opt[i, 0], traj_opt[j, 0]],
                [traj_opt[i, 1], traj_opt[j, 1]], 'm-', lw=0.8, alpha=0.5)
    ax.set_title(f'After Optimisation ({len(loops)} loops)\nClosure error: {err_after:.3f}m')
    ax.set_aspect('equal'); ax.grid(True)

    ax = axes[0, 2]
    ax.set_rasterized(True)
    if map_points:
        pts = np.vstack(map_points)
        step = max(1, len(pts)//20000)
        ax.scatter(pts[::step, 0], pts[::step, 1], s=0.5, c='k', alpha=0.3,
                   rasterized=True)
    ax.set_title('Point Cloud Map'); ax.set_aspect('equal'); ax.grid(True)

    ax = axes[1, 0]
    prob = grid.get_probability_map()
    half = grid.n * grid.res / 2
    ax.imshow(prob, cmap='gray_r', origin='lower', vmin=0.3, vmax=0.7,
              extent=[-half, half, -half, half])
    ax.plot(traj_opt[:, 0], traj_opt[:, 1], 'b-', lw=0.8, alpha=0.7)
    ax.set_title('Occupancy Grid')

    ax = axes[1, 1]
    bars = ax.bar(['Before\nLoop Closure', 'After\nLoop Closure'],
                  [err_before, err_after], color=['tomato', 'steelblue'])
    for bar, v in zip(bars, [err_before, err_after]):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.05,
                f'{v:.3f}m', ha='center', fontsize=12, fontweight='bold')
    ax.set_ylabel('Closure Error (m)'); ax.set_title('Loop Closure Effect')
    ax.grid(True, axis='y')

    ax = axes[1, 2]
    ts = np.linspace(0, 1, len(traj_raw))
    ax.plot(ts, traj_raw[:, 1], 'b-', label='x raw', alpha=0.7)
    ax.plot(ts, traj_raw[:, 2], 'r-', label='y raw', alpha=0.7)
    ts2 = np.linspace(0, 1, len(traj_opt))
    ax.plot(ts2, traj_opt[:, 0], 'b--', label='x opt')
    ax.plot(ts2, traj_opt[:, 1], 'r--', label='y opt')
    ax.set_xlabel('Normalised Time'); ax.set_ylabel('Position (m)')
    ax.set_title('Position Over Time'); ax.legend(fontsize=8); ax.grid(True)

    plt.tight_layout()
    out = os.path.join(out_dir, f"full_summary_{label}.pdf")
    plt.savefig(out, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"[OK] Full summary plot -> {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help="LiDAR data file")
    parser.add_argument('--output', default='results/q3')
    parser.add_argument('--label', default='seq')
    parser.add_argument('--topic', default='/scan')
    parser.add_argument('--max-range', type=float, default=8.0)
    args = parser.parse_args()

    if not args.data:
        parser.print_help()
        return

    run_full_pipeline(args.data, args.output, args.label,
                      topic=args.topic, max_range=args.max_range)


if __name__ == '__main__':
    main()
