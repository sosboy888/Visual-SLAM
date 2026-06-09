#!/usr/bin/env python3
"""
COMP0249 CW2 - Utility: ROS Bag → CSV Converter

Converts sensor_msgs/LaserScan messages from a ROS bag to a plain CSV
so the data can be used on machines without ROS installed.

CSV format (one scan per row):
  timestamp, angle_min, angle_max, angle_increment,
  range_min, range_max, r0, r1, ..., rN

Usage:
    python bag_to_csv.py --bag my_scan.bag --topic /scan --output my_scan.csv
"""

import argparse
import os
import sys


def convert_bag_to_csv(bag_path, topic, output_path, max_scans=None):
    try:
        import rosbag
    except ImportError:
        print("[ERROR] rosbag not available.")
        print("        Run this script on the robot PC or a machine with ROS sourced.")
        sys.exit(1)

    print(f"Converting {bag_path} -> {output_path}")
    bag = rosbag.Bag(bag_path)
    count = 0

    with open(output_path, 'w') as f:
        f.write("# timestamp,angle_min,angle_max,angle_increment,"
                "range_min,range_max,ranges...\n")
        for _, msg, t in bag.read_messages(topics=[topic]):
            ranges_str = ','.join(f'{r:.4f}' for r in msg.ranges)
            f.write(f"{t.to_sec():.6f},{msg.angle_min:.6f},{msg.angle_max:.6f},"
                    f"{msg.angle_increment:.8f},{msg.range_min:.4f},{msg.range_max:.4f},"
                    f"{ranges_str}\n")
            count += 1
            if count % 100 == 0:
                print(f"  {count} scans written...")
            if max_scans and count >= max_scans:
                break

    bag.close()
    print(f"[OK] {count} scans -> {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bag', required=True)
    parser.add_argument('--topic', default='/scan')
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-scans', type=int, default=None)
    args = parser.parse_args()

    convert_bag_to_csv(args.bag, args.topic, args.output, args.max_scans)


if __name__ == '__main__':
    main()
