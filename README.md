# Visual and LiDAR SLAM

---

## Overview

This project investigates two SLAM approaches:

1. **Visual SLAM** — running ORB-SLAM2 on the KITTI-07 and TUM fr3/long_office_household benchmark datasets, and on sequences we recorded ourselves at UCL, with a comparison against COLMAP.
2. **LiDAR SLAM** — a 2D SLAM pipeline built from scratch using an RPLidar A1, tested across three environments at UCL, with parameter sweeps, loop closure detection, and pose-graph optimisation.

---

## Results Summary

### Visual SLAM — Benchmark Datasets (ORB-SLAM2)

| Sequence | ATE RMSE (m) | Keyframes |
|---|---|---|
| KITTI-07 (baseline) | 2.86 | 516 / 1101 |
| TUM fr3/long (baseline) | 1.26 | 196 / 2585 |

**Key findings:**
- Loop closure is critical on KITTI-07: disabling it increased ATE from 2.86 m to 16.68 m (6x degradation)
- Loop closure had almost no effect on TUM fr3/long (1.26 m → 1.25 m) because it is a shorter, slower sequence with less accumulated drift
- Reducing ORB features below 2000 caused complete tracking failure on both sequences — 2000 features is close to the minimum for reliable operation
- Disabling outlier rejection paradoxically improved aligned ATE slightly (2.86 m → 2.33 m), but raw unaligned error was dramatically worse (82 m RMSE), confirming that outlier rejection is essential for geometric consistency

### Visual SLAM — Own Sequences (UCL Student Centre)

| Sequence | ORB-SLAM2 | COLMAP vs ORB-SLAM2 ATE (m) |
|---|---|---|
| Indoor | Tracked (208 keyframes) | 2.55 |
| Outdoor | Tracking failed | N/A |

- Indoor: calibrated using COLMAP SfM (fx = fy = 3469.78 px), tracked successfully throughout
- Outdoor: ORB-SLAM2 failed to initialise due to insufficient feature parallax; high focal length (fx = 3308 px) combined with flat sky/ground texture prevented bootstrapping
- COLMAP's global bundle adjustment gave a more geometrically consistent reconstruction than ORB-SLAM2's sequential tracking, as expected

### LiDAR SLAM — Parameter Sweep (Indoor 1, stairwell)

| Parameter | Configuration | Closure Error (m) |
|---|---|---|
| Max range | 8 m | 10.26 |
| Max range | 12 m | 11.92 |
| Angular res | Full | 10.26 |
| Angular res | Every 2nd beam | **0.53** |
| Angular res | Every 3rd beam | 2.65 |
| Voxel grid | 5 cm | 10.26 |
| Voxel grid | 10 cm | 14.78 |
| Scan rate | Full (813 scans) | 10.26 |
| Scan rate | 50% (407 scans) | 8.15 |
| Scan rate | 33% (271 scans) | 9.15 |

**Most significant finding:** downsampling to every 2nd beam reduced closure error from 10.26 m to 0.53 m — a 95% improvement. This is counterintuitive: fewer points gave better results because spurious stairwell readings (from the open gap to the floor below) were causing ICP to lock onto incorrect correspondences at full resolution. Downsampling reduced the influence of those noisy readings.

### LiDAR SLAM — Loop Closure and Pose-Graph Optimisation

| Sequence | Before Opt (m) | After Opt (m) | Loop Closures |
|---|---|---|---|
| Indoor 1 (stairwell) | 10.26 | 7.75 | 22 detected |
| Indoor 2 (hall) | 0.40 | 0.40 | 0 detected |
| Outdoor (quad) | 11.18 | 11.18 | 0 detected |

- Indoor 2 achieved only 0.40 m drift over a 20 m loop without any loop closure — clean flat walls gave ICP near-perfect conditions
- Outdoor was the worst result: pedestrians walking through the scan plane created dynamic point cloud noise that made ICP unreliable and prevented loop closure detection

---

## Repository Structure

```
├── data/
│   ├── kitti07/              # KITTI-07 sequence frames and ground truth
│   ├── tum_fr3/              # TUM fr3/long_office_household sequence
│   ├── ucl_indoor/           # UCL Student Centre indoor sequence
│   ├── ucl_outdoor/          # UCL Student Centre outdoor sequence
│   └── lidar/
│       ├── indoor1/          # Stairwell sequence (813 scans)
│       ├── indoor2/          # Main hall sequence (1679 scans)
│       └── outdoor/          # UCL quad sequence (543 scans)
├── config/
│   ├── kitti07.yaml          # ORB-SLAM2 config for KITTI
│   ├── tum_fr3.yaml          # ORB-SLAM2 config for TUM
│   └── ucl_indoor.yaml       # ORB-SLAM2 config with UCL calibration
├── src/
│   ├── lidar_slam/
│   │   ├── icp.py            # ICP scan matching implementation
│   │   ├── odometry.py       # Laser odometry pipeline
│   │   ├── loop_closure.py   # Histogram descriptor + ICP verification
│   │   ├── pose_graph.py     # Gauss-Newton pose-graph optimisation
│   │   └── occupancy.py      # Occupancy grid mapping
│   └── eval/
│       └── closure_error.py  # Start-to-end closure error computation
├── scripts/
│   ├── run_orbslam.sh        # ORB-SLAM2 experiment runner
│   ├── run_colmap.sh         # COLMAP reconstruction script
│   ├── parameter_sweep.sh    # LiDAR parameter sweep automation
│   └── evaluate.py           # EVO trajectory evaluation wrapper
├── results/
│   ├── q1/                   # Visual SLAM dataset experiments
│   ├── q2/                   # UCL own sequence experiments
│   └── q3/                   # LiDAR SLAM experiments
├── figures/                  # Generated figures for report
├── report/
│   └── report.tex            # Full LaTeX report source
└── README.md
```

---

## Dependencies

### Visual SLAM

```bash
# ORB-SLAM2
git clone https://github.com/raulmur/ORB_SLAM2
# Follow ORB-SLAM2 build instructions (requires OpenCV, Eigen3, Pangolin, g2o)

# COLMAP
sudo apt install colmap
# or build from source: https://colmap.github.io/install.html

# EVO trajectory evaluation
pip install evo
```

### LiDAR SLAM

```bash
pip install numpy scipy matplotlib open3d
# RPLidar driver (if recording new sequences)
pip install rplidar-roboticia
```

---

## Running the Experiments

### ORB-SLAM2 on KITTI-07

```bash
# Baseline
./scripts/run_orbslam.sh kitti07 config/kitti07.yaml data/kitti07/

# Evaluate with EVO
evo_ape kitti data/kitti07/groundtruth.txt results/q1/kitti07/baseline/trajectory.txt \
    --align --correct_scale --plot
```

### ORB-SLAM2 on TUM fr3/long

```bash
./scripts/run_orbslam.sh tum config/tum_fr3.yaml data/tum_fr3/

evo_ape tum data/tum_fr3/groundtruth.txt results/q1/tum/baseline/trajectory.txt \
    --align --plot
```

### COLMAP on UCL Indoor Sequence

```bash
# Extract calibration and reconstruct
./scripts/run_colmap.sh data/ucl_indoor/ results/q2/indoor/
```

### LiDAR SLAM Pipeline

```bash
# Run full pipeline on a sequence
python src/lidar_slam/odometry.py --sequence data/lidar/indoor1/ --max_range 8

# Parameter sweep
./scripts/parameter_sweep.sh data/lidar/indoor1/

# Evaluate closure error
python src/eval/closure_error.py results/q3/indoor1/trajectory.npy
```

---

## Calibration

Camera intrinsics for the UCL sequences were estimated using COLMAP SfM with sequential matching (10-frame overlap, no loop detection):

```
fx = fy = 3469.78 px
cx = 960, cy = 540
Distortion: ~0 (negligible)
```

These parameters are exported to `config/ucl_indoor.yaml` for use with ORB-SLAM2.

---

## Key Takeaways

**Visual SLAM:**
- Loop closure is essential for long sequences with large-scale drift (KITTI-07). It is less critical for short, slow sequences (TUM) where drift stays small
- ORB feature count has a hard threshold effect: above 2000 features tracking is stable, below it tracking fails entirely on these sequences
- Monocular SLAM has no absolute scale — all trajectory errors must be evaluated with Sim(3) alignment (scale + rotation + translation)

**LiDAR SLAM:**
- ICP quality is highly environment-dependent. Clean flat walls give excellent results; open spaces, noise sources, and dynamic objects (people) all degrade performance significantly
- More data is not always better: downsampling noisy sensor data can improve ICP by removing spurious correspondences
- Pose-graph optimisation can only correct drift where loop closures exist. Without loop closures there is nothing to constrain the optimisation

---

## References

- Mur-Artal & Tardós, *ORB-SLAM2*, IEEE Trans. Robotics, 2017
- Geiger et al., *Vision Meets Robotics: The KITTI Dataset*, IJRR, 2013
- Sturm et al., *A Benchmark for the Evaluation of RGB-D SLAM Systems*, IROS, 2012
- Grupp, *EVO: Python Package for the Evaluation of Odometry and SLAM*, 2017
- Schönberger & Frahm, *Structure-from-Motion Revisited*, CVPR, 2016
- Besl & McKay, *A Method for Registration of 3-D Shapes*, IEEE TPAMI, 1992
