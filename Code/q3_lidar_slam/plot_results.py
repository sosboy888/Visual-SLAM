#!/usr/bin/env python3
"""
COMP0249 CW2 - Q3: Plot Generator

Reads saved trajectory and occupancy grid files and produces
the figures needed for the report. Kept separate from the pipeline
so you can regenerate plots without re-running the full SLAM.

Usage:
    python plot_results.py --results results/q3/indoor1 --label indoor1
    python plot_results.py --all --results results/q3
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def _load_traj(path):
    if not os.path.exists(path):
        return None
    data = np.loadtxt(path, comments='#')
    return data.reshape(-1, data.shape[-1]) if data.ndim == 1 else data


def _load_occ(path):
    return np.load(path) if os.path.exists(path) else None


def _plot_traj(ax, traj, color='b', label='', lw=1.5):
    if traj is None or not len(traj):
        return
    ax.plot(traj[:, 1], traj[:, 2], color=color, lw=lw, label=label)
    ax.plot(traj[0, 1],  traj[0, 2],  'go', ms=10, zorder=5)
    ax.plot(traj[-1, 1], traj[-1, 2], 'r^', ms=10, zorder=5)


def make_full_summary(results_dir, label, out_path):
    """6-panel summary: raw traj | optimised traj | point cloud |
    occupancy grid | closure error bar | position over time."""

    traj_raw = _load_traj(os.path.join(results_dir, f"trajectory_raw_{label}.txt"))
    traj_opt = _load_traj(os.path.join(results_dir, f"trajectory_optimised_{label}.txt"))
    occ_raw  = _load_occ(os.path.join(results_dir, f"occupancy_raw_{label}.npy"))
    pts_file = os.path.join(results_dir, f"pointcloud_{label}.npy")

    err_before = err_after = None
    err_file = os.path.join(results_dir, f"closure_errors_{label}.txt")
    if os.path.exists(err_file):
        with open(err_file) as f:
            for line in f:
                if 'before_m' in line:
                    err_before = float(line.split(':')[1])
                elif 'after_m' in line:
                    err_after = float(line.split(':')[1])

    n_loops = 0
    lc_file = os.path.join(results_dir, f"loop_closures_{label}.npz")
    if os.path.exists(lc_file):
        n_loops = len(np.load(lc_file, allow_pickle=True)['loops'])

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f"SLAM Results: {label}", fontsize=15, fontweight='bold')
    sp = mpatches.Patch(color='green', label='Start')
    ep = mpatches.Patch(color='red',   label='End')

    ax = axes[0, 0]
    _plot_traj(ax, traj_raw, color='steelblue')
    ax.set_title(f'Raw Odometry' + (f'\nClosure: {err_before:.3f}m' if err_before else ''))
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
    ax.legend(handles=[sp, ep], fontsize=8)

    ax = axes[0, 1]
    _plot_traj(ax, traj_opt, color='darkorange')
    ax.set_title(f'After Loop Closure ({n_loops} loops)' +
                 (f'\nClosure: {err_after:.3f}m' if err_after else ''))
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
    ax.legend(handles=[sp, ep], fontsize=8)

    # Rasterise the point cloud panel — without this a 50k-point scatter
    # produces a PDF with 50k individual vector objects that freezes viewers
    ax = axes[0, 2]
    ax.set_rasterized(True)
    if os.path.exists(pts_file):
        pts = np.load(pts_file)
        if len(pts):
            step = max(1, len(pts)//20000)
            ax.scatter(pts[::step, 0], pts[::step, 1],
                       s=0.5, c='k', alpha=0.4, rasterized=True)
    if traj_opt is not None:
        ax.plot(traj_opt[:, 1], traj_opt[:, 2], 'b-', lw=0.5, alpha=0.5)
    ax.set_title('Point Cloud Map'); ax.set_aspect('equal'); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    if occ_raw is not None:
        prob = 1.0 / (1.0 + np.exp(-occ_raw))
        half = occ_raw.shape[0] * 0.05 / 2
        ax.imshow(prob, cmap='gray_r', origin='lower', vmin=0.3, vmax=0.7,
                  extent=[-half, half, -half, half])
        if traj_opt is not None:
            ax.plot(traj_opt[:, 1], traj_opt[:, 2], 'b-', lw=0.8, alpha=0.7)
    ax.set_title('Occupancy Grid')

    ax = axes[1, 1]
    if err_before is not None and err_after is not None:
        bars = ax.bar(['Before\nLoop Closure', 'After\nLoop Closure'],
                      [err_before, err_after],
                      color=['tomato', 'steelblue'], edgecolor='black', lw=0.8)
        for bar, v in zip(bars, [err_before, err_after]):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.05,
                    f'{v:.3f}m', ha='center', fontsize=12, fontweight='bold')
        ax.set_ylabel('Closure Error (m)')
        ax.set_ylim(0, max(err_before, err_after) * 1.2)
        ax.grid(True, axis='y', alpha=0.3)
    ax.set_title('Loop Closure Effect')

    ax = axes[1, 2]
    if traj_raw is not None:
        t = np.linspace(0, 1, len(traj_raw))
        ax.plot(t, traj_raw[:, 1], 'b-', lw=1.2, label='x raw', alpha=0.8)
        ax.plot(t, traj_raw[:, 2], 'r-', lw=1.2, label='y raw', alpha=0.8)
    if traj_opt is not None:
        t = np.linspace(0, 1, len(traj_opt))
        ax.plot(t, traj_opt[:, 1], 'b--', lw=1.2, label='x opt')
        ax.plot(t, traj_opt[:, 2], 'r--', lw=1.2, label='y opt')
    ax.set_xlabel('Normalised Time'); ax.set_ylabel('Position (m)')
    ax.set_title('Position Over Time')
    ax.legend(fontsize=8, ncol=2); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"[OK] Full summary -> {out_path}")


def make_before_after(results_dir, label, out_path):
    """Side-by-side before/after trajectories — key figure for the report."""
    traj_raw = _load_traj(os.path.join(results_dir, f"trajectory_raw_{label}.txt"))
    traj_opt = _load_traj(os.path.join(results_dir, f"trajectory_optimised_{label}.txt"))

    err_before = err_after = None
    err_file = os.path.join(results_dir, f"closure_errors_{label}.txt")
    if os.path.exists(err_file):
        with open(err_file) as f:
            for line in f:
                if 'before_m' in line:
                    err_before = float(line.split(':')[1])
                elif 'after_m' in line:
                    err_after = float(line.split(':')[1])

    n_loops = 0
    lc_file = os.path.join(results_dir, f"loop_closures_{label}.npz")
    if os.path.exists(lc_file):
        n_loops = len(np.load(lc_file, allow_pickle=True)['loops'])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'Factor Graph Optimisation: Before vs After Loop Closure', fontsize=13)

    for ax, traj, title, color in zip(
            axes,
            [traj_raw, traj_opt],
            [f'Before Optimisation\nClosure error: {err_before:.3f}m' if err_before else 'Before',
             f'After Optimisation ({n_loops} loop constraints)\nClosure error: {err_after:.3f}m' if err_after else 'After'],
            ['steelblue', 'darkorange']):
        if traj is not None:
            ax.plot(traj[:, 1], traj[:, 2], color=color, lw=1.5)
            ax.plot(traj[0,  1], traj[0,  2], 'go', ms=12, zorder=5, label='Start')
            ax.plot(traj[-1, 1], traj[-1, 2], 'r^', ms=12, zorder=5, label='End')
            ax.plot([traj[0, 1], traj[-1, 1]],
                    [traj[0, 2], traj[-1, 2]], 'k--', lw=1, alpha=0.5, label='Closure gap')
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
        ax.set_aspect('equal'); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"[OK] Before/after -> {out_path}")


def process_sequence(results_dir, label):
    print(f"\nGenerating plots for: {label}")
    make_full_summary(results_dir, label,
                      os.path.join(results_dir, f"full_summary_{label}.pdf"))
    make_before_after(results_dir, label,
                      os.path.join(results_dir, f"before_after_{label}.pdf"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', required=True)
    parser.add_argument('--label', default=None)
    parser.add_argument('--all', action='store_true',
                        help="Process indoor1, indoor2, outdoor subdirectories")
    args = parser.parse_args()

    if args.all:
        for seq in ['indoor1', 'indoor2', 'outdoor']:
            d = os.path.join(args.results, seq)
            if os.path.isdir(d):
                process_sequence(d, seq)
    else:
        label = args.label or os.path.basename(args.results.rstrip('/'))
        process_sequence(args.results, label)


if __name__ == '__main__':
    main()
