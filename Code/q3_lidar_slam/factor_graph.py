#!/usr/bin/env python3
"""
COMP0249 CW2 - Q3(d): Factor Graph Optimisation

Combines odometry poses and loop closure constraints into a pose graph
and optimises using Gauss-Newton (our custom solver) or g2opy/GTSAM
if installed.

The idea: each odometry step is a soft constraint between consecutive
poses. Each loop closure adds another constraint between two non-adjacent
poses. We find the pose configuration that satisfies all constraints
as well as possible in a least-squares sense.

Loop closures are weighted higher than odometry because they've been
verified geometrically by ICP and are therefore more reliable.
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from pathlib import Path
from typing import Tuple


class PoseGraph:
    def __init__(self):
        self.poses:      np.ndarray = None   # (N, 3): x, y, theta
        self.timestamps: np.ndarray = None
        self.odom_edges: list = []           # [(i, j, dx, dy, dth, info_matrix)]
        self.loop_edges: list = []

    def load_trajectory(self, path):
        data = np.loadtxt(path, comments='#')
        self.timestamps = data[:, 0]
        self.poses = data[:, 1:4].copy()
        print(f"[OK] {len(self.poses)} poses from {path}")

    def build_odometry_edges(self, info_odom=None):
        """Build sequential edges from the trajectory. Higher information = tighter constraint."""
        if info_odom is None:
            info_odom = np.diag([100.0, 100.0, 50.0])
        self.odom_edges = []
        for i in range(len(self.poses) - 1):
            dx, dy, dth = _relative_pose(self.poses[i], self.poses[i+1])
            self.odom_edges.append((i, i+1, dx, dy, dth, info_odom))

    def add_loop_closure(self, i, j, T_rel, info_loop=None):
        """Add a loop closure edge. Higher information than odometry — ICP-verified."""
        if info_loop is None:
            info_loop = np.diag([500.0, 500.0, 200.0])
        dx, dy = T_rel[0, 2], T_rel[1, 2]
        dth = np.arctan2(T_rel[1, 0], T_rel[0, 0])
        self.loop_edges.append((i, j, dx, dy, dth, info_loop))

    def closure_error(self):
        if self.poses is None or len(self.poses) < 2:
            return 0.0
        return float(np.linalg.norm(self.poses[-1, :2] - self.poses[0, :2]))


def _relative_pose(p_i, p_j):
    """Compute relative pose in the frame of pose i."""
    xi, yi, thi = p_i
    xj, yj, thj = p_j
    c, s = np.cos(thi), np.sin(thi)
    return (c*(xj-xi) + s*(yj-yi),
           -s*(xj-xi) + c*(yj-yi),
            _wrap(thj - thi))


def _wrap(a):
    return (a + np.pi) % (2*np.pi) - np.pi


class GaussNewtonPoseGraph:
    """
    Custom 2D pose graph optimiser.
    Linearises the error function around the current estimate and
    solves the resulting linear system iteratively.
    This is the Gauss-Newton method — essentially Newton's method
    without the Hessian second-order term.
    """

    def __init__(self, graph):
        self.graph = graph
        self.poses = graph.poses.copy()
        self.N = len(self.poses)

    def _residual_jacobian(self, edge):
        i, j, dx_m, dy_m, dth_m, info = edge
        xi, yi, thi = self.poses[i]
        xj, yj, thj = self.poses[j]
        c, s = np.cos(thi), np.sin(thi)

        # Predicted relative pose
        dx_p =  c*(xj-xi) + s*(yj-yi)
        dy_p = -s*(xj-xi) + c*(yj-yi)
        dth_p = _wrap(thj - thi)

        r = np.array([dx_m-dx_p, dy_m-dy_p, _wrap(dth_m-dth_p)])

        # Jacobians w.r.t. pose i and pose j
        J_i = -np.array([
            [-c, -s, s*(xj-xi)-c*(yj-yi)],
            [ s, -c, c*(xj-xi)+s*(yj-yi)],
            [ 0,  0, -1],
        ])
        J_j = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
        return r, J_i, J_j, info

    def optimise(self, max_iter=50, tol=1e-5):
        all_edges = self.graph.odom_edges + self.graph.loop_edges
        for it in range(max_iter):
            n = self.N * 3
            H = np.zeros((n, n))
            b = np.zeros(n)
            for edge in all_edges:
                i, j = edge[0], edge[1]
                r, Ji, Jj, O = self._residual_jacobian(edge)
                si, sj = i*3, j*3
                H[si:si+3, si:si+3] += Ji.T @ O @ Ji
                H[si:si+3, sj:sj+3] += Ji.T @ O @ Jj
                H[sj:sj+3, si:si+3] += Jj.T @ O @ Ji
                H[sj:sj+3, sj:sj+3] += Jj.T @ O @ Jj
                b[si:si+3] += Ji.T @ O @ r
                b[sj:sj+3] += Jj.T @ O @ r

            # Fix first pose as anchor (prevents gauge freedom)
            H[:3, :] = H[:, :3] = 0
            H[:3, :3] = np.eye(3)
            b[:3] = 0

            try:
                dx = np.linalg.solve(H, b)
            except np.linalg.LinAlgError:
                dx = np.linalg.lstsq(H, b, rcond=None)[0]

            self.poses += dx.reshape(-1, 3)
            for k in range(self.N):
                self.poses[k, 2] = _wrap(self.poses[k, 2])

            if np.linalg.norm(dx) < tol:
                print(f"  Converged at iteration {it+1}")
                break

        return self.poses


def optimise_pose_graph(graph):
    """Try g2opy → GTSAM → fallback to our Gauss-Newton solver."""
    try:
        import g2o
        return _optimise_g2opy(graph)
    except ImportError:
        pass
    try:
        import gtsam
        return _optimise_gtsam(graph)
    except ImportError:
        pass
    print("[INFO] Using custom Gauss-Newton solver (g2opy/GTSAM not installed)")
    return GaussNewtonPoseGraph(graph).optimise()


def _optimise_g2opy(graph):
    import g2o
    opt = g2o.SparseOptimizer()
    solver = g2o.BlockSolverSE2(g2o.LinearSolverEigenSE2())
    opt.set_algorithm(g2o.OptimizationAlgorithmLevenberg(solver))
    for i, (x, y, th) in enumerate(graph.poses):
        v = g2o.VertexSE2(); v.set_id(i)
        v.set_estimate(g2o.SE2(x, y, th))
        if i == 0: v.set_fixed(True)
        opt.add_vertex(v)
    for i, j, dx, dy, dth, info in graph.odom_edges + graph.loop_edges:
        e = g2o.EdgeSE2()
        e.set_vertex(0, opt.vertex(i)); e.set_vertex(1, opt.vertex(j))
        e.set_measurement(g2o.SE2(dx, dy, dth)); e.set_information(info)
        opt.add_edge(e)
    opt.initialize_optimization(); opt.optimize(30)
    return np.array([[opt.vertex(i).estimate().translation()[0],
                      opt.vertex(i).estimate().translation()[1],
                      opt.vertex(i).estimate().rotation().angle()]
                     for i in range(len(graph.poses))])


def _optimise_gtsam(graph):
    import gtsam
    from gtsam import Pose2, NonlinearFactorGraph, Values, BetweenFactorPose2, noiseModel
    fg = NonlinearFactorGraph(); init = Values()
    prior = noiseModel.Diagonal.Sigmas(np.array([0.001, 0.001, 0.001]))
    fg.add(gtsam.PriorFactorPose2(0, Pose2(*graph.poses[0]), prior))
    for i, (x, y, th) in enumerate(graph.poses):
        init.insert(i, Pose2(x, y, th))
    for i, j, dx, dy, dth, info in graph.odom_edges + graph.loop_edges:
        sigma = 1.0 / np.sqrt(np.diag(info) + 1e-9)
        noise = noiseModel.Diagonal.Sigmas(sigma)
        fg.add(BetweenFactorPose2(i, j, Pose2(dx, dy, dth), noise))
    result = gtsam.LevenbergMarquardtOptimizer(fg, init).optimize()
    return np.array([[result.atPose2(i).x(), result.atPose2(i).y(),
                      result.atPose2(i).theta()] for i in range(len(graph.poses))])


def plot_before_after(before, after, loop_edges, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Factor Graph Optimisation: Before vs After Loop Closure")
    for ax, poses, title in zip(axes, [before, after],
                                ['Before Optimisation', 'After Optimisation']):
        ax.plot(poses[:, 0], poses[:, 1], 'b-', lw=1.5)
        ax.plot(poses[0, 0], poses[0, 1], 'go', ms=12, label='Start')
        ax.plot(poses[-1, 0], poses[-1, 1], 'r^', ms=12, label='End')
        for i, j, *_ in loop_edges:
            ax.plot([poses[i, 0], poses[j, 0]],
                    [poses[i, 1], poses[j, 1]], 'm-', lw=1, alpha=0.6)
        err = np.linalg.norm(poses[-1, :2] - poses[0, :2])
        ax.set_title(f"{title}\nClosure error: {err:.4f} m")
        ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
        ax.legend(); ax.set_aspect('equal'); ax.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"[OK] Before/after plot -> {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trajectory', required=True)
    parser.add_argument('--loop-closures', default=None)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)
    graph = PoseGraph()
    graph.load_trajectory(args.trajectory)
    graph.build_odometry_edges()

    print(f"Closure error BEFORE: {graph.closure_error():.4f} m")

    if args.loop_closures and os.path.isfile(args.loop_closures):
        lc_data = np.load(args.loop_closures, allow_pickle=True)
        for lc in lc_data['loops']:
            graph.add_loop_closure(int(lc['query_idx']), int(lc['ref_idx']),
                                   lc['T_relative'])
        print(f"Loaded {len(lc_data['loops'])} loop closures")

    before = graph.poses.copy()
    graph.poses = optimise_pose_graph(graph)
    print(f"Closure error AFTER:  {graph.closure_error():.4f} m")

    out_traj = os.path.join(args.output, "trajectory_optimised.txt")
    np.savetxt(out_traj, np.column_stack([graph.timestamps, graph.poses]),
               header="timestamp x y theta", comments='# ')
    print(f"[OK] Optimised trajectory -> {out_traj}")

    plot_before_after(before, graph.poses, graph.loop_edges,
                      os.path.join(args.output, "before_after.pdf"))


if __name__ == '__main__':
    main()
