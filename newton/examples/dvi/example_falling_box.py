# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Falling boxes example for debugging ground contact settling.

Multiple boxes are dropped from staggered heights with small lateral
and angular perturbation velocities. Each box should land on the
ground plane, slide/tumble briefly, and come to rest. Useful for
diagnosing contact solver issues where boxes keep rotating or
drifting along the ground instead of settling.

Run with:
    uv run -m newton.examples falling_box --num-boxes 5 --height 1.0
"""

from __future__ import annotations

from itertools import product

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverType


LOCAL_CORNERS_UNIT = np.array(
    [[sx, sy, sz] for sx, sy, sz in product((-1, 1), (-1, 1), (-1, 1))],
    dtype=np.float64,
)


def _quat_to_mat(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ]
    )


class Example:
    """Falling boxes contact settling debug example."""

    name = "Falling Box"

    def __init__(self, viewer, args):
        self.viewer = viewer
        iterations = getattr(args, "iterations", 200)
        self.device = getattr(args, "device", "cuda:0")

        self.num_boxes = getattr(args, "num_boxes", 20)
        self.height = getattr(args, "height", 1.0)
        self.max_lateral_vel = getattr(args, "max_lateral_vel", 5.0)
        self.max_angular_vel = getattr(args, "max_angular_vel", 10.0)
        self.box_half = getattr(args, "box_half", 0.1)
        self.gravity = -10.0

        builder = newton.ModelBuilder(gravity=self.gravity)
        builder.default_shape_cfg.mu = 0.5
        builder.default_shape_cfg.gap = 0.01
        builder.default_shape_cfg.margin = 0.002
        builder.add_ground_plane()

        # Lay out boxes on a grid so they don't interact with each other.
        cols = int(np.ceil(np.sqrt(self.num_boxes)))
        spacing = 6.0 * self.box_half
        for i in range(self.num_boxes):
            row, col = divmod(i, cols)
            x = (col - (cols - 1) / 2.0) * spacing
            y = (row - (cols - 1) / 2.0) * spacing
            z = self.height + (i % 5) * 0.2
            body = builder.add_body(
                xform=wp.transform([x, y, z], wp.quat_identity()),
            )
            builder.add_shape_box(body, hx=self.box_half, hy=self.box_half, hz=self.box_half)

        self.model = builder.finalize(device=self.device)
        self.viewer.set_model(self.model)

        # Assign varied initial velocities per box using a deterministic seed.
        rng = np.random.default_rng(42)
        self.state_0 = self.model.state()
        qd_np = self.state_0.body_qd.numpy()
        for i in range(self.num_boxes):
            # Random linear velocity in xy-plane.
            angle = rng.uniform(0, 2 * np.pi)
            speed = rng.uniform(0.5, self.max_lateral_vel)
            qd_np[i, 0] = speed * np.cos(angle)
            qd_np[i, 1] = speed * np.sin(angle)
            # Random angular velocity on a random axis.
            axis = rng.normal(size=3)
            axis /= np.linalg.norm(axis) + 1e-12
            wmag = rng.uniform(1.0, self.max_angular_vel)
            qd_np[i, 3:6] = axis * wmag
        self.state_0.body_qd.assign(qd_np)

        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.contacts = self.model.collide(self.state_0)

        contact_config = newton.solvers.NumericalSolverConfig(
            solver_type=SolverType.SPARSE_JACOBI,
            max_iterations=iterations,
            tolerance=1e-8,
            reg=1e-4,
            relax=0.9,
            omega=0.3,
            alpha=0.01,
            recovery_speed=1.0,
            backtrack_iterations=5,
        )

        self.solver = newton.solvers.SolverDVI(
            self.model,
            contact_solver=contact_config,
            enable_timers=True,
            angular_damping=1.0,
        )

        self.sim_dt = 0.005
        self.sim_time = 0.0
        self.frame = 0

        self._local_corners = LOCAL_CORNERS_UNIT * np.array([self.box_half, self.box_half, self.box_half])

        self.expected_z = self.box_half

        print(f"\nFalling Boxes Test")
        print(f"  Iterations: {iterations}")
        print(f"  Boxes: {self.num_boxes}, box half-extent: {self.box_half}")
        print(f"  Drop height: {self.height}")
        print(f"  Max lateral velocity: {self.max_lateral_vel} m/s")
        print(f"  Max angular velocity: {self.max_angular_vel} rad/s")
        print(f"  Expected resting z: ~{self.expected_z}")
        print()

    def _corner_z_min(self, body_idx: int) -> float:
        bq = self.state_0.body_q.numpy()
        pos = bq[body_idx, 0:3]
        rot = _quat_to_mat(bq[body_idx, 3:7])
        corners = (self._local_corners @ rot.T) + pos
        return float(corners[:, 2].min())

    def step(self):
        self.state_0.clear_forces()
        self.contacts = self.model.collide(self.state_0)
        self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
        self.state_0, self.state_1 = self.state_1, self.state_0

        self.sim_time += self.sim_dt
        self.frame += 1

        if self.frame % 50 == 0:
            self._print_status()

    def _print_status(self):
        bq = self.state_0.body_q.numpy()
        bd = self.state_0.body_qd.numpy()
        n_contacts = int(self.contacts.rigid_contact_count.numpy()[0]) if self.contacts else 0

        print(f"Frame {self.frame} (t={self.sim_time:.3f}s)  contacts={n_contacts}")
        for i in range(self.num_boxes):
            pos = bq[i, 0:3]
            vel_lin = bd[i, 0:3]
            vel_ang = bd[i, 3:6]
            speed_lin = float(np.linalg.norm(vel_lin))
            speed_ang = float(np.linalg.norm(vel_ang))
            cz_min = self._corner_z_min(i)
            settled = speed_lin < 0.01 and speed_ang < 0.05
            print(
                f"  box{i}: z={pos[2]:+.4f}  |v|={speed_lin:.4f}  "
                f"|w|={speed_ang:.4f}  corner_z_min={cz_min:+.5f}  "
                f"{'settled' if settled else 'MOVING'}"
            )
        print()

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        bq = self.state_0.body_q.numpy()
        bd = self.state_0.body_qd.numpy()

        no_nan = bool(np.all(np.isfinite(bq)) and np.all(np.isfinite(bd)))

        max_speed_lin = 0.0
        max_speed_ang = 0.0
        min_corner_z = float("inf")
        all_z_ok = True

        for i in range(self.num_boxes):
            vel_lin = bd[i, 0:3]
            vel_ang = bd[i, 3:6]
            speed_lin = float(np.linalg.norm(vel_lin))
            speed_ang = float(np.linalg.norm(vel_ang))
            max_speed_lin = max(max_speed_lin, speed_lin)
            max_speed_ang = max(max_speed_ang, speed_ang)
            min_corner_z = min(min_corner_z, self._corner_z_min(i))
            z_err = abs(bq[i, 2] - self.expected_z)
            if z_err > 0.05:
                all_z_ok = False

        no_penetration = min_corner_z > -0.01
        settled = max_speed_lin < 0.1 and max_speed_ang < 0.5

        print("\n" + "=" * 60)
        print("Final Test Results")
        print("=" * 60)
        print(f"  Max linear speed:  {max_speed_lin:.4f} m/s")
        print(f"  Max angular speed: {max_speed_ang:.4f} rad/s")
        print(f"  Min corner z:      {min_corner_z:.5f}")
        print(f"  No NaN:            {no_nan}")
        print(f"  No penetration:    {no_penetration}")
        print(f"  All z correct:     {all_z_ok}")
        print(f"  Settled:           {settled}")

        passed = no_nan and no_penetration and settled and all_z_ok
        print(f"\nTest: {'PASSED' if passed else 'FAILED'}")
        print("=" * 60)

        assert passed, (
            f"Falling boxes test failed: max_speed_lin={max_speed_lin:.4f}, "
            f"max_speed_ang={max_speed_ang:.4f}, min_corner_z={min_corner_z:.5f}"
        )


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--num-boxes", type=int, default=20, help="Number of boxes to drop")
    parser.add_argument("--height", type=float, default=1.0, help="Drop height [m]")
    parser.add_argument("--max-lateral-vel", type=float, default=5.0, help="Max initial lateral speed [m/s]")
    parser.add_argument("--max-angular-vel", type=float, default=10.0, help="Max initial angular speed [rad/s]")
    parser.add_argument("--box-half", type=float, default=0.1, help="Box half-extent [m]")
    parser.add_argument("--iterations", type=int, default=200, help="Max contact solver iterations")

    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
