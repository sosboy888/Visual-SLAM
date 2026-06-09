#!/usr/bin/env python3
"""
COMP0249 CW2 - Q3: LiDAR Data Loader

Handles three formats:
  - ROS .bag  (needs rosbag installed)
  - .csv      (our RPLidar recording format — one scan per row)
  - .npz      (our compressed format for processed data)

The CSV format we use is:
  timestamp,angle_min,angle_max,angle_increment,range_min,range_max,r0,r1,...,rN
"""

import numpy as np
import os
from dataclasses import dataclass
from typing import Iterator, List, Optional


@dataclass
class LidarScan:
    timestamp: float
    ranges: np.ndarray      # metres, shape (N,)
    angles: np.ndarray      # radians, shape (N,)
    range_min: float = 0.15
    range_max: float = 12.0  # RPLidar A1 rated max

    def to_cartesian(self, max_range=None):
        """Convert polar readings to (x, y) Cartesian points, filtering invalid ranges."""
        r_max = max_range if max_range else self.range_max
        valid = (self.ranges >= self.range_min) & (self.ranges <= r_max)
        r = self.ranges[valid]
        a = self.angles[valid]
        return np.column_stack([r * np.cos(a), r * np.sin(a)])

    def downsample(self, nth):
        """Return a copy with every n-th beam kept — reduces angular resolution."""
        return LidarScan(
            timestamp=self.timestamp,
            ranges=self.ranges[::nth],
            angles=self.angles[::nth],
            range_min=self.range_min,
            range_max=self.range_max,
        )


class LidarLoader:
    """Unified loader — auto-detects format from file extension."""

    def __init__(self, path, topic="/scan", skip_nth=1,
                 max_range_override=None):
        self.path = path
        self.topic = topic
        self.skip_nth = skip_nth
        self.max_range_override = max_range_override
        _, ext = os.path.splitext(path)
        self._format = {'bag': 'bag', 'csv': 'csv',
                        'txt': 'csv', 'npz': 'npz'}.get(ext.lstrip('.').lower(), 'csv')

    def __iter__(self) -> Iterator[LidarScan]:
        if self._format == 'bag':
            yield from self._load_bag()
        elif self._format == 'csv':
            yield from self._load_csv()
        elif self._format == 'npz':
            yield from self._load_npz()

    def _load_bag(self):
        try:
            import rosbag
        except ImportError:
            raise ImportError("rosbag not found — use CSV export instead")
        bag = rosbag.Bag(self.path)
        for i, (_, msg, t) in enumerate(bag.read_messages(topics=[self.topic])):
            if self.skip_nth > 1 and i % self.skip_nth != 0:
                continue
            n = len(msg.ranges)
            angles = np.linspace(msg.angle_min, msg.angle_max, n)
            ranges = np.where(np.isfinite(msg.ranges), msg.ranges, 0.0).astype(np.float32)
            yield LidarScan(t.to_sec(), ranges, angles,
                            msg.range_min,
                            self.max_range_override or msg.range_max)
        bag.close()

    def _load_csv(self):
        """
        CSV format (one scan per row):
          timestamp, angle_min, angle_max, angle_increment,
          range_min, range_max, r0, r1, ..., rN
        The header line starts with '#' and is skipped.
        """
        with open(self.path) as f:
            lines = f.readlines()

        scan_idx = 0
        for line in lines:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            if self.skip_nth > 1 and scan_idx % self.skip_nth != 0:
                scan_idx += 1
                continue
            parts = line.split(',')
            try:
                ts        = float(parts[0])
                angle_min = float(parts[1])
                angle_max = float(parts[2])
                r_min     = float(parts[4])
                r_max     = self.max_range_override or float(parts[5])
                ranges    = np.array([float(x) for x in parts[6:]], dtype=np.float32)
                ranges    = np.where(np.isfinite(ranges), ranges, 0.0)
                angles    = np.linspace(angle_min, angle_max, len(ranges))
                yield LidarScan(ts, ranges, angles, r_min, r_max)
            except (ValueError, IndexError) as e:
                print(f"[WARN] Skipping malformed CSV line: {e}")
            scan_idx += 1

    def _load_npz(self):
        data = np.load(self.path, allow_pickle=True)
        timestamps  = data['timestamps']
        all_ranges  = data['ranges']
        all_angles  = data['angles']
        r_min = float(data.get('range_min', 0.15))
        r_max = self.max_range_override or float(data.get('range_max', 12.0))
        for i, (ts, ranges) in enumerate(zip(timestamps, all_ranges)):
            if self.skip_nth > 1 and i % self.skip_nth != 0:
                continue
            angles = all_angles[i] if all_angles.ndim == 2 else all_angles
            yield LidarScan(ts, ranges, angles, r_min, r_max)


def save_scans_npz(scans: List[LidarScan], output_path: str):
    """Compress a list of scans to .npz for fast reloading."""
    np.savez_compressed(
        output_path,
        timestamps=np.array([s.timestamp for s in scans]),
        ranges=np.array([s.ranges for s in scans]),
        angles=scans[0].angles if scans else np.array([]),
        range_min=scans[0].range_min if scans else 0.15,
        range_max=scans[0].range_max if scans else 12.0,
    )
    print(f"[OK] {len(scans)} scans -> {output_path}")
