#!/usr/bin/env bash
# COMP0249 CW2 - Q1: Run all ORB-SLAM2 experiments
# Usage: bash run_q1_experiments.sh
# Run from anywhere — paths are all absolute.

set -e

# ============================================================
# USER CONFIGURATION
# ============================================================
ORBSLAM2_DIR="$HOME/ORB_SLAM2"

# KITTI dataset (E:\UCL\Robot_CW2\dataset\ → /mnt/e/ in WSL2)
KITTI_SEQ_DIR="/mnt/e/UCL/Robot_CW2/dataset/sequences/07"
KITTI_GT="/mnt/e/UCL/Robot_CW2/dataset/poses/07.txt"

# TUM dataset — update this path once downloaded
TUM_SEQ_DIR="/mnt/e/UCL/Robot_CW2/rgbd_dataset_freiburg3_long_office_household/rgbd_dataset_freiburg3_long_office_household"
TUM_GT="$TUM_SEQ_DIR/groundtruth.txt"

# ORB-SLAM2 configs and vocab
KITTI_VOCAB="$ORBSLAM2_DIR/Vocabulary/ORBvoc.txt"
KITTI_CFG="$ORBSLAM2_DIR/Examples/Monocular/KITTI04-12.yaml"
TUM_VOCAB="$ORBSLAM2_DIR/Vocabulary/ORBvoc.txt"
TUM_CFG="$ORBSLAM2_DIR/Examples/Monocular/TUM3.yaml"

# Results stored inside the project directory
# Script is at: /mnt/e/UCL/Robot_CW2/q1_visual_slam_datasets/run_q1_experiments.sh
# Project is:   /mnt/e/UCL/Robot_CW2/
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="/mnt/e/UCL/Robot_CW2"
RESULTS_ROOT="$PROJECT_DIR/results/q1"
SCRIPTS_DIR="$SCRIPT_DIR"

MONO_KITTI="$ORBSLAM2_DIR/Examples/Monocular/mono_kitti"
MONO_TUM="$ORBSLAM2_DIR/Examples/Monocular/mono_tum"

# ============================================================
# Setup
# ============================================================
mkdir -p "$RESULTS_ROOT"/{kitti07,tum}/{baseline,feat_1000,feat_500,feat_200,no_outlier,no_loopclosure}
echo "Results will be saved to: $RESULTS_ROOT"
echo ""

# ============================================================
# Helpers
# ============================================================

# ORB-SLAM2 saves KeyFrameTrajectory.txt in the current working directory.
# We cd to ORBSLAM2_DIR before running so it lands there predictably.
# After running, search both ORBSLAM2_DIR and cwd just in case.
save_trajectory() {
    local out_dir="$1"
    local found=""

    # Search likely locations
    for loc in \
        "$ORBSLAM2_DIR/KeyFrameTrajectory.txt" \
        "$PWD/KeyFrameTrajectory.txt" \
        "$HOME/KeyFrameTrajectory.txt"; do
        if [ -f "$loc" ]; then
            found="$loc"
            break
        fi
    done

    if [ -n "$found" ]; then
        cp "$found" "$out_dir/KeyFrameTrajectory.txt"
        local n
        n=$(wc -l < "$out_dir/KeyFrameTrajectory.txt")
        echo "    [OK] Saved $n poses -> $out_dir/KeyFrameTrajectory.txt"
        # Remove so next run doesn't accidentally pick up stale file
        rm -f "$found"
    else
        echo "    [WARN] KeyFrameTrajectory.txt not found anywhere"
        echo "           Tracking may have failed, or EGL crash prevented save"
        echo "           Check: ls ~/ORB_SLAM2/KeyFrameTrajectory.txt"
    fi
}

run_kitti() {
    local cfg="$1"
    local out_dir="$2"
    echo ">>> Running KITTI | config: $(basename "$cfg") | output: $(basename "$out_dir")"
    # Remove any stale trajectory from previous run
    rm -f "$ORBSLAM2_DIR/KeyFrameTrajectory.txt"
    cd "$ORBSLAM2_DIR"
    # || true: EGL/viewer crash after sequence finishes should not abort script
    "$MONO_KITTI" "$KITTI_VOCAB" "$cfg" "$KITTI_SEQ_DIR" || true
    save_trajectory "$out_dir"
    cd "$PROJECT_DIR"
}

run_tum() {
    local cfg="$1"
    local out_dir="$2"
    if [ ! -d "$TUM_SEQ_DIR" ]; then
        echo "    [SKIP] TUM sequence not found at $TUM_SEQ_DIR"
        echo "           Download fr3_long_office_household from:"
        echo "           https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download"
        return
    fi
    echo ">>> Running TUM | config: $(basename "$cfg") | output: $(basename "$out_dir")"
    rm -f "$ORBSLAM2_DIR/KeyFrameTrajectory.txt"
    cd "$ORBSLAM2_DIR"
    "$MONO_TUM" "$TUM_VOCAB" "$cfg" "$TUM_SEQ_DIR" || true
    save_trajectory "$out_dir"
    cd "$PROJECT_DIR"
}

rebuild_orbslam() {
    echo "  Rebuilding ORB-SLAM2 (~1 min)..."
    cmake --build "$ORBSLAM2_DIR/build" --parallel "$(nproc)" 2>&1 | tail -3
    echo "  Rebuild done."
}

# ============================================================
# Q1(a) Baseline
# ============================================================
echo "========================================"
echo " Q1(a): Baseline"
echo "========================================"
run_kitti "$KITTI_CFG" "$RESULTS_ROOT/kitti07/baseline"
run_tum   "$TUM_CFG"   "$RESULTS_ROOT/tum/baseline"

# ============================================================
# Q1(b) ORB Feature count: 1000, 500, 200
# ============================================================
echo ""
echo "========================================"
echo " Q1(b): ORB Feature Count"
echo "========================================"
for N in 1000 500 200; do
    echo "--- nFeatures = $N ---"

    # Generate modified config
    python3 "$SCRIPTS_DIR/modify_orbslam_config.py" \
        --config "$KITTI_CFG" --features "$N" > /dev/null
    python3 "$SCRIPTS_DIR/modify_orbslam_config.py" \
        --config "$TUM_CFG" --features "$N" > /dev/null

    KITTI_CFG_FEAT="${KITTI_CFG%.yaml}_feat${N}.yaml"
    TUM_CFG_FEAT="${TUM_CFG%.yaml}_feat${N}.yaml"

    run_kitti "$KITTI_CFG_FEAT" "$RESULTS_ROOT/kitti07/feat_$N"
    run_tum   "$TUM_CFG_FEAT"   "$RESULTS_ROOT/tum/feat_$N"
done

# ============================================================
# Q1(c) Disable outlier rejection
# ============================================================
echo ""
echo "========================================"
echo " Q1(c): No Outlier Rejection"
echo "========================================"
python3 "$SCRIPTS_DIR/disable_outlier_rejection.py" \
    --orbslam-src "$ORBSLAM2_DIR/src" --disable
rebuild_orbslam

run_kitti "$KITTI_CFG" "$RESULTS_ROOT/kitti07/no_outlier"
run_tum   "$TUM_CFG"   "$RESULTS_ROOT/tum/no_outlier"

python3 "$SCRIPTS_DIR/disable_outlier_rejection.py" \
    --orbslam-src "$ORBSLAM2_DIR/src" --restore
rebuild_orbslam

# ============================================================
# Q1(d) Disable loop closure
# ============================================================
echo ""
echo "========================================"
echo " Q1(d): No Loop Closure"
echo "========================================"
python3 "$SCRIPTS_DIR/disable_loop_closure.py" \
    --orbslam-src "$ORBSLAM2_DIR/src" --disable
rebuild_orbslam

run_kitti "$KITTI_CFG" "$RESULTS_ROOT/kitti07/no_loopclosure"
run_tum   "$TUM_CFG"   "$RESULTS_ROOT/tum/no_loopclosure"

python3 "$SCRIPTS_DIR/disable_loop_closure.py" \
    --orbslam-src "$ORBSLAM2_DIR/src" --restore
rebuild_orbslam

# ============================================================
# EVO Evaluation
# ============================================================
echo ""
echo "========================================"
echo " Running EVO Evaluation"
echo "========================================"

run_evo() {
    local fmt="$1"
    local gt="$2"
    local est="$3"
    local out_prefix="$4"
    if [ ! -f "$est" ]; then
        echo "  [SKIP] No trajectory: $est"
        return
    fi
    if [ ! -f "$gt" ]; then
        echo "  [SKIP] No ground truth: $gt"
        return
    fi
    evo_ape "$fmt" "$gt" "$est" \
        -va --t_max_diff 0.5 \
        --save_results "${out_prefix}.zip" \
        --save_plot    "${out_prefix}_ape.pdf" \
        2>/dev/null \
        && echo "  [OK]   $(basename "$out_prefix")" \
        || echo "  [WARN] EVO failed for $(basename "$out_prefix")"
}

for EXP in baseline feat_1000 feat_500 feat_200 no_outlier no_loopclosure; do
    run_evo tum \
        "$KITTI_GT" \
        "$RESULTS_ROOT/kitti07/$EXP/KeyFrameTrajectory.txt" \
        "$RESULTS_ROOT/kitti07/$EXP/kitti07_$EXP"
    run_evo tum \
        "$TUM_GT" \
        "$RESULTS_ROOT/tum/$EXP/KeyFrameTrajectory.txt" \
        "$RESULTS_ROOT/tum/$EXP/tum_$EXP"
done

# Comparison plots
make_comparison() {
    local out="$1"; shift
    local zips=()
    for z in "$@"; do [ -f "$z" ] && zips+=("$z"); done
    [ ${#zips[@]} -lt 2 ] && { echo "  [SKIP] Not enough results for $(basename "$out")"; return; }
    evo_res "${zips[@]}" --save_plot "$out" -v 2>/dev/null \
        && echo "  [OK]   $(basename "$out")" || true
}

echo ""
echo "--- Comparison plots ---"
make_comparison "$RESULTS_ROOT/kitti07_features_comparison.pdf" \
    "$RESULTS_ROOT/kitti07/baseline/kitti07_baseline.zip" \
    "$RESULTS_ROOT/kitti07/feat_1000/kitti07_feat_1000.zip" \
    "$RESULTS_ROOT/kitti07/feat_500/kitti07_feat_500.zip" \
    "$RESULTS_ROOT/kitti07/feat_200/kitti07_feat_200.zip"

make_comparison "$RESULTS_ROOT/kitti07_outlier_comparison.pdf" \
    "$RESULTS_ROOT/kitti07/baseline/kitti07_baseline.zip" \
    "$RESULTS_ROOT/kitti07/no_outlier/kitti07_no_outlier.zip"

make_comparison "$RESULTS_ROOT/kitti07_loopclosure_comparison.pdf" \
    "$RESULTS_ROOT/kitti07/baseline/kitti07_baseline.zip" \
    "$RESULTS_ROOT/kitti07/no_loopclosure/kitti07_no_loopclosure.zip"

make_comparison "$RESULTS_ROOT/tum_features_comparison.pdf" \
    "$RESULTS_ROOT/tum/baseline/tum_baseline.zip" \
    "$RESULTS_ROOT/tum/feat_1000/tum_feat_1000.zip" \
    "$RESULTS_ROOT/tum/feat_500/tum_feat_500.zip" \
    "$RESULTS_ROOT/tum/feat_200/tum_feat_200.zip"

make_comparison "$RESULTS_ROOT/tum_outlier_comparison.pdf" \
    "$RESULTS_ROOT/tum/baseline/tum_baseline.zip" \
    "$RESULTS_ROOT/tum/no_outlier/tum_no_outlier.zip"

make_comparison "$RESULTS_ROOT/tum_loopclosure_comparison.pdf" \
    "$RESULTS_ROOT/tum/baseline/tum_baseline.zip" \
    "$RESULTS_ROOT/tum/no_loopclosure/tum_no_loopclosure.zip"

echo ""
echo "========================================"
echo " ALL Q1 EXPERIMENTS COMPLETE"
echo "========================================"
echo "Results: $RESULTS_ROOT"
echo ""
echo "Trajectories saved:"
find "$RESULTS_ROOT" -name "KeyFrameTrajectory.txt" | sort | \
    while read -r f; do printf "  %4d poses  %s\n" "$(wc -l < "$f")" "$f"; done
echo ""
echo "EVO plots:"
find "$RESULTS_ROOT" -name "*.pdf" | sort | while read -r f; do echo "  $f"; done


# ============================================================
# USER CONFIGURATION
# ============================================================
ORBSLAM2_DIR="$HOME/ORB_SLAM2"

# KITTI dataset (E:\UCL\Robot_CW2\dataset\ → /mnt/e/ in WSL2)
KITTI_SEQ_DIR="/mnt/e/UCL/Robot_CW2/dataset/sequences/07"
KITTI_GT="/mnt/e/UCL/Robot_CW2/dataset/poses/07.txt"

# TUM dataset — update this path once downloaded
TUM_SEQ_DIR="/mnt/e/UCL/Robot_CW2/rgbd_dataset_freiburg3_long_office_household/rgbd_dataset_freiburg3_long_office_household"
TUM_GT="$TUM_SEQ_DIR/groundtruth.txt"

# ORB-SLAM2 configs and vocab
KITTI_VOCAB="$ORBSLAM2_DIR/Vocabulary/ORBvoc.txt"
KITTI_CFG="$ORBSLAM2_DIR/Examples/Monocular/KITTI04-12.yaml"
TUM_VOCAB="$ORBSLAM2_DIR/Vocabulary/ORBvoc.txt"
TUM_CFG="$ORBSLAM2_DIR/Examples/Monocular/TUM3.yaml"

# Results stored inside the project directory (wherever this script lives)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_ROOT="$PROJECT_DIR/results/q1"
SCRIPTS_DIR="$SCRIPT_DIR"

MONO_KITTI="$ORBSLAM2_DIR/Examples/Monocular/mono_kitti"
MONO_TUM="$ORBSLAM2_DIR/Examples/Monocular/mono_tum"

# ============================================================
# Setup
# ============================================================
mkdir -p "$RESULTS_ROOT"/{kitti07,tum}/{baseline,feat_1000,feat_500,feat_200,no_outlier,no_loopclosure}
echo "Results will be saved to: $RESULTS_ROOT"
echo ""

# ============================================================
# Helpers
# ============================================================
run_kitti() {
    local cfg="$1"
    local out_dir="$2"
    echo ">>> Running KITTI | config: $(basename "$cfg") | output: $out_dir"
    cd "$ORBSLAM2_DIR"
    # || true so a Ctrl+C on the viewer doesn't abort the whole script
    "$MONO_KITTI" "$KITTI_VOCAB" "$cfg" "$KITTI_SEQ_DIR" || true
    if [ -f "$ORBSLAM2_DIR/KeyFrameTrajectory.txt" ]; then
        cp "$ORBSLAM2_DIR/KeyFrameTrajectory.txt" "$out_dir/"
        local n
        n=$(wc -l < "$out_dir/KeyFrameTrajectory.txt")
        echo "    [OK] Saved $n poses -> $out_dir/KeyFrameTrajectory.txt"
    else
        echo "    [WARN] KeyFrameTrajectory.txt not found — did tracking fail?"
    fi
    cd "$PROJECT_DIR"
}

run_tum() {
    local cfg="$1"
    local out_dir="$2"
    if [ ! -d "$TUM_SEQ_DIR" ]; then
        echo "    [SKIP] TUM sequence not found at $TUM_SEQ_DIR"
        echo "           Download it and update TUM_SEQ_DIR at the top of this script."
        return
    fi
    echo ">>> Running TUM | config: $(basename "$cfg") | output: $out_dir"
    cd "$ORBSLAM2_DIR"
    "$MONO_TUM" "$TUM_VOCAB" "$cfg" "$TUM_SEQ_DIR" || true
    if [ -f "$ORBSLAM2_DIR/KeyFrameTrajectory.txt" ]; then
        cp "$ORBSLAM2_DIR/KeyFrameTrajectory.txt" "$out_dir/"
        local n
        n=$(wc -l < "$out_dir/KeyFrameTrajectory.txt")
        echo "    [OK] Saved $n poses -> $out_dir/KeyFrameTrajectory.txt"
    else
        echo "    [WARN] KeyFrameTrajectory.txt not found — did tracking fail?"
    fi
    cd "$PROJECT_DIR"
}

rebuild_orbslam() {
    echo "  Rebuilding ORB-SLAM2 (this takes ~1 min)..."
    cmake --build "$ORBSLAM2_DIR/build" --parallel "$(nproc)" 2>&1 | tail -3
    echo "  Rebuild done."
}

# ============================================================
# Q1(a) Baseline
# ============================================================
echo "========================================"
echo " Q1(a): Baseline"
echo "========================================"
run_kitti "$KITTI_CFG" "$RESULTS_ROOT/kitti07/baseline"
run_tum   "$TUM_CFG"   "$RESULTS_ROOT/tum/baseline"

# ============================================================
# Q1(b) ORB Feature count: 1000, 500, 200
# ============================================================
echo ""
echo "========================================"
echo " Q1(b): ORB Feature Count"
echo "========================================"
for N in 1000 500 200; do
    echo "--- nFeatures = $N ---"

    KITTI_CFG_FEAT=$(python3 "$SCRIPTS_DIR/modify_orbslam_config.py" \
        --config "$KITTI_CFG" --features "$N" 2>&1 \
        | grep -o '/[^ ]*\.yaml' | tail -1)
    TUM_CFG_FEAT=$(python3 "$SCRIPTS_DIR/modify_orbslam_config.py" \
        --config "$TUM_CFG" --features "$N" 2>&1 \
        | grep -o '/[^ ]*\.yaml' | tail -1)

    [ -z "$KITTI_CFG_FEAT" ] && KITTI_CFG_FEAT="${KITTI_CFG%.yaml}_feat${N}.yaml"
    [ -z "$TUM_CFG_FEAT"   ] && TUM_CFG_FEAT="${TUM_CFG%.yaml}_feat${N}.yaml"

    run_kitti "$KITTI_CFG_FEAT" "$RESULTS_ROOT/kitti07/feat_$N"
    run_tum   "$TUM_CFG_FEAT"   "$RESULTS_ROOT/tum/feat_$N"
done

# ============================================================
# Q1(c) Disable outlier rejection
# ============================================================
echo ""
echo "========================================"
echo " Q1(c): No Outlier Rejection"
echo "========================================"
python3 "$SCRIPTS_DIR/disable_outlier_rejection.py" \
    --orbslam-src "$ORBSLAM2_DIR/src" --disable
rebuild_orbslam

run_kitti "$KITTI_CFG" "$RESULTS_ROOT/kitti07/no_outlier"
run_tum   "$TUM_CFG"   "$RESULTS_ROOT/tum/no_outlier"

python3 "$SCRIPTS_DIR/disable_outlier_rejection.py" \
    --orbslam-src "$ORBSLAM2_DIR/src" --restore
rebuild_orbslam

# ============================================================
# Q1(d) Disable loop closure
# ============================================================
echo ""
echo "========================================"
echo " Q1(d): No Loop Closure"
echo "========================================"
python3 "$SCRIPTS_DIR/disable_loop_closure.py" \
    --orbslam-src "$ORBSLAM2_DIR/src" --disable
rebuild_orbslam

run_kitti "$KITTI_CFG" "$RESULTS_ROOT/kitti07/no_loopclosure"
run_tum   "$TUM_CFG"   "$RESULTS_ROOT/tum/no_loopclosure"

python3 "$SCRIPTS_DIR/disable_loop_closure.py" \
    --orbslam-src "$ORBSLAM2_DIR/src" --restore
rebuild_orbslam

# ============================================================
# EVO Evaluation — generate all ATE plots
# ============================================================
echo ""
echo "========================================"
echo " Running EVO Evaluation"
echo "========================================"

run_evo() {
    local fmt="$1"   # kitti or tum
    local gt="$2"
    local est="$3"
    local out="$4"
    if [ ! -f "$est" ]; then
        echo "  [SKIP] $est not found"
        return
    fi
    mkdir -p "$(dirname "$out")"
    evo_ape "$fmt" "$gt" "$est" \
        -va --t_max_diff 0.5 \
        --save_results "${out}.zip" \
        --save_plot    "${out}_ape.pdf" \
        2>/dev/null && echo "  [OK] $(basename "$out")" \
        || echo "  [WARN] EVO failed for $(basename "$out")"
}

# KITTI experiments
for EXP in baseline feat_1000 feat_500 feat_200 no_outlier no_loopclosure; do
    run_evo tum \
        "$KITTI_GT" \
        "$RESULTS_ROOT/kitti07/$EXP/KeyFrameTrajectory.txt" \
        "$RESULTS_ROOT/kitti07/$EXP/kitti07_$EXP"
done

# TUM experiments
for EXP in baseline feat_1000 feat_500 feat_200 no_outlier no_loopclosure; do
    run_evo tum \
        "$TUM_GT" \
        "$RESULTS_ROOT/tum/$EXP/KeyFrameTrajectory.txt" \
        "$RESULTS_ROOT/tum/$EXP/tum_$EXP"
done

# Comparison plots (features, outlier, loop closure)
echo ""
echo "--- Generating comparison plots ---"

make_comparison() {
    local label="$1"; shift
    local zips=()
    for z in "$@"; do [ -f "$z" ] && zips+=("$z"); done
    [ ${#zips[@]} -lt 2 ] && return
    evo_res "${zips[@]}" \
        --save_plot "$RESULTS_ROOT/$label" -v 2>/dev/null \
        && echo "  [OK] $label" || true
}

make_comparison "kitti07_features_comparison.pdf" \
    "$RESULTS_ROOT/kitti07/baseline/kitti07_baseline.zip" \
    "$RESULTS_ROOT/kitti07/feat_1000/kitti07_feat_1000.zip" \
    "$RESULTS_ROOT/kitti07/feat_500/kitti07_feat_500.zip" \
    "$RESULTS_ROOT/kitti07/feat_200/kitti07_feat_200.zip"

make_comparison "kitti07_outlier_comparison.pdf" \
    "$RESULTS_ROOT/kitti07/baseline/kitti07_baseline.zip" \
    "$RESULTS_ROOT/kitti07/no_outlier/kitti07_no_outlier.zip"

make_comparison "kitti07_loopclosure_comparison.pdf" \
    "$RESULTS_ROOT/kitti07/baseline/kitti07_baseline.zip" \
    "$RESULTS_ROOT/kitti07/no_loopclosure/kitti07_no_loopclosure.zip"

make_comparison "tum_features_comparison.pdf" \
    "$RESULTS_ROOT/tum/baseline/tum_baseline.zip" \
    "$RESULTS_ROOT/tum/feat_1000/tum_feat_1000.zip" \
    "$RESULTS_ROOT/tum/feat_500/tum_feat_500.zip" \
    "$RESULTS_ROOT/tum/feat_200/tum_feat_200.zip"

make_comparison "tum_outlier_comparison.pdf" \
    "$RESULTS_ROOT/tum/baseline/tum_baseline.zip" \
    "$RESULTS_ROOT/tum/no_outlier/tum_no_outlier.zip"

make_comparison "tum_loopclosure_comparison.pdf" \
    "$RESULTS_ROOT/tum/baseline/tum_baseline.zip" \
    "$RESULTS_ROOT/tum/no_loopclosure/tum_no_loopclosure.zip"

echo ""
echo "========================================"
echo " ALL Q1 EXPERIMENTS COMPLETE"
echo "========================================"
echo "Results: $RESULTS_ROOT"
echo ""
echo "PDF plots:"
find "$RESULTS_ROOT" -name "*.pdf" | sort | while read -r f; do
    echo "  $f"
done
