#!/usr/bin/env python3
"""
COMP0249 CW2 - Q3(c): Loop Closure Detection

Two-stage detection:
  1. Descriptor matching  — fast histogram check to find candidates.
  2. ICP verification     — geometric check to reject false positives.

The histogram descriptor divides the scan into angular sectors and bins
the range readings per sector. It's rotation-sensitive enough to
distinguish different corridor orientations while being robust to
small changes in position.

To avoid false positives with recent frames, we gate by:
  - Minimum time gap between query and reference frame
  - Descriptor distance threshold
  - ICP error threshold
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Optional
from laser_odometry import icp_2d, voxel_downsample


@dataclass
class LoopCandidate:
    query_idx: int
    ref_idx: int
    query_ts: float
    ref_ts: float
    T_relative: np.ndarray    # 3x3 homogeneous relative transform from ICP
    icp_error: float
    descriptor_dist: float


class ScanDescriptor:
    """
    Histogram descriptor: divide 360° into N_SECTORS angular bins,
    compute a range histogram per sector, normalise.
    Fast and simple — works well in structured environments.
    Breaks down in open/featureless spaces (all sectors look similar).
    """
    N_SECTORS   = 36
    N_RANGE_BINS = 20
    RANGE_MAX   = 12.0

    @staticmethod
    def compute(scan_pts):
        if len(scan_pts) == 0:
            return np.zeros(ScanDescriptor.N_SECTORS * ScanDescriptor.N_RANGE_BINS)

        angles = np.arctan2(scan_pts[:, 1], scan_pts[:, 0])
        ranges = np.linalg.norm(scan_pts, axis=1)
        sector_idx = np.clip(
            ((angles + np.pi) / (2*np.pi) * ScanDescriptor.N_SECTORS).astype(int),
            0, ScanDescriptor.N_SECTORS - 1)

        desc = np.zeros(ScanDescriptor.N_SECTORS * ScanDescriptor.N_RANGE_BINS)
        for s in range(ScanDescriptor.N_SECTORS):
            mask = sector_idx == s
            if not mask.any():
                continue
            hist, _ = np.histogram(ranges[mask], bins=ScanDescriptor.N_RANGE_BINS,
                                   range=(0, ScanDescriptor.RANGE_MAX))
            desc[s*ScanDescriptor.N_RANGE_BINS:(s+1)*ScanDescriptor.N_RANGE_BINS] = \
                hist / (mask.sum() + 1e-6)

        norm = np.linalg.norm(desc)
        return desc / norm if norm > 0 else desc


class LoopClosureDetector:
    def __init__(self,
                 min_travel=5.0,
                 min_time_gap=10.0,
                 desc_threshold=0.25,
                 icp_error_threshold=0.15,
                 keyframe_distance=0.5,
                 voxel_size=0.05):
        self.min_travel         = min_travel
        self.min_time_gap       = min_time_gap
        self.desc_threshold     = desc_threshold
        self.icp_error_threshold = icp_error_threshold
        self.keyframe_distance  = keyframe_distance
        self.voxel_size         = voxel_size

        self.kf_descriptors: List[np.ndarray] = []
        self.kf_scans:       List[np.ndarray] = []
        self.kf_poses:       List[np.ndarray] = []
        self.kf_timestamps:  List[float]      = []
        self.kf_indices:     List[int]        = []

        self.last_kf_pos  = None
        self.total_travel = 0.0
        self.detected_loops: List[LoopCandidate] = []

    def _should_add_keyframe(self, pos):
        if self.last_kf_pos is None:
            return True
        return np.linalg.norm(pos - self.last_kf_pos) >= self.keyframe_distance

    def _add_keyframe(self, idx, scan_pts, pose, ts):
        pts_v = voxel_downsample(scan_pts, self.voxel_size) if self.voxel_size > 0 else scan_pts
        self.kf_descriptors.append(ScanDescriptor.compute(pts_v))
        self.kf_scans.append(pts_v)
        self.kf_poses.append(pose.copy())
        self.kf_timestamps.append(ts)
        self.kf_indices.append(idx)
        self.last_kf_pos = pose[:2].copy()

    def check(self, scan_idx, scan_pts, pose_xyt, timestamp) -> Optional[LoopCandidate]:
        """
        Main call per scan. Returns a LoopCandidate if a loop is confirmed,
        otherwise None.
        """
        pos = pose_xyt[:2]

        if self.last_kf_pos is not None:
            self.total_travel += np.linalg.norm(pos - self.last_kf_pos)

        if self._should_add_keyframe(pos):
            self._add_keyframe(scan_idx, scan_pts, pose_xyt, timestamp)
            if len(self.kf_indices) <= 1:
                return None

        if self.total_travel < self.min_travel:
            return None

        pts_v = voxel_downsample(scan_pts, self.voxel_size) if self.voxel_size > 0 else scan_pts
        query_desc = ScanDescriptor.compute(pts_v)

        # Find top-5 nearest keyframes by descriptor distance
        descs = np.array(self.kf_descriptors)
        dists = np.linalg.norm(descs - query_desc, axis=1)
        for kf_idx in np.argsort(dists)[:5]:
            desc_dist = dists[kf_idx]
            ref_ts = self.kf_timestamps[kf_idx]

            # Time and distance gating — avoid trivially close matches
            if abs(timestamp - ref_ts) < self.min_time_gap:
                continue
            if desc_dist > self.desc_threshold:
                continue

            # Geometric verification: ICP must agree with the descriptor match
            T_rel, icp_err = icp_2d(pts_v, self.kf_scans[kf_idx],
                                     max_iter=30, max_correspondence_dist=0.5)
            if icp_err < self.icp_error_threshold:
                lc = LoopCandidate(
                    query_idx=scan_idx,
                    ref_idx=self.kf_indices[kf_idx],
                    query_ts=timestamp,
                    ref_ts=ref_ts,
                    T_relative=T_rel,
                    icp_error=icp_err,
                    descriptor_dist=desc_dist,
                )
                self.detected_loops.append(lc)
                print(f"  [LOOP] scan {scan_idx} matches keyframe "
                      f"{self.kf_indices[kf_idx]} "
                      f"(desc_dist={desc_dist:.3f}, icp_err={icp_err:.4f})")
                return lc

        return None

    def plot_loop_summary(self, trajectory, output_path):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle("Loop Closure Detection Results")

        ax = axes[0]
        ax.plot(trajectory[:, 1], trajectory[:, 2], 'b-', lw=1, label='Trajectory')
        ax.plot(trajectory[0, 1], trajectory[0, 2], 'go', ms=10, label='Start')
        ax.plot(trajectory[-1, 1], trajectory[-1, 2], 'r^', ms=10, label='End')
        for lc in self.detected_loops:
            qi = min(lc.query_idx, len(trajectory)-1)
            ri = min(lc.ref_idx, len(trajectory)-1)
            ax.plot([trajectory[qi, 1], trajectory[ri, 1]],
                    [trajectory[qi, 2], trajectory[ri, 2]],
                    'm-', lw=1.5, alpha=0.7)
        ax.set_title(f"Trajectory + {len(self.detected_loops)} Loop Closures")
        ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
        ax.legend(); ax.set_aspect('equal'); ax.grid(True)

        ax = axes[1]
        if self.detected_loops:
            ax.scatter([lc.descriptor_dist for lc in self.detected_loops],
                       [lc.icp_error       for lc in self.detected_loops],
                       c='m', s=30)
            ax.axhline(self.icp_error_threshold,  color='r', ls='--',
                       label=f'ICP threshold ({self.icp_error_threshold})')
            ax.axvline(self.desc_threshold, color='b', ls='--',
                       label=f'Desc threshold ({self.desc_threshold})')
            ax.set_xlabel('Descriptor Distance')
            ax.set_ylabel('ICP Error (m)')
            ax.set_title('Loop Closure Quality')
            ax.legend(); ax.grid(True)

        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"[OK] Loop closure plot -> {output_path}")
