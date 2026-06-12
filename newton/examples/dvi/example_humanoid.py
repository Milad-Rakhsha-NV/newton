# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

###########################################################################
# Example Humanoid (DVI Solver)
#
# Loads ``newton/examples/assets/nv_humanoid.xml`` and simulates one or more
# free-floating humanoids landing on a ground plane with SolverDVI. The
# joint constraint solver is selectable on the CLI so the example doubles
# as a benchmark fixture for comparing solvers:
#
#   - ``sparse_ldl``: Block-sparse tile LDL direct solver (default)
#   - ``sparse_jacobi``:    Matrix-free iterative Jacobi
#
# Contacts are always solved with ``SolverType.SPARSE_JACOBI`` (matches the other
# dvi examples).
#
# Usage::
#
#     # Run interactively with the tile LDL solver:
#     python -m newton.examples dvi_humanoid
#
#     # Pick a different joint solver:
#     python -m newton.examples dvi_humanoid --joint-solver sparse_jacobi
#
#     # Compare all viable joint solvers (no viewer):
#     python -m newton.examples dvi_humanoid --compare-solvers \
#         --num-frames 200 --num-worlds 4
#
###########################################################################

from __future__ import annotations

import time

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverType


JOINT_SOLVER_CHOICES = (
    "sparse_ldl",
    "sparse_jacobi",
)


_SOLVER_TYPE_MAP = {
    "sparse_ldl": SolverType.SPARSE_LDL,
    "sparse_jacobi": SolverType.SPARSE_JACOBI,
}


def _create_joint_solver(
    name: str,
    baumgarte_only: bool = False,
    position_correction: newton.solvers.NumericalSolverConfig | None = None,
) -> newton.solvers.NumericalSolverConfig:
    """Build a numerical solver config of the requested kind for the joint stage."""
    solver_type = _SOLVER_TYPE_MAP.get(name)
    if solver_type is None:
        raise ValueError(f"Unknown joint solver: {name}")

    alpha = 0.001
    recovery_speed = 0.05
    if name == "sparse_ldl":
        return newton.solvers.NumericalSolverConfig(
            solver_type=solver_type,
            max_iterations=50, omega=0.3, relax=0.8, alpha=alpha, recovery_speed=recovery_speed, reg=1e-6,
            position_correction=position_correction,
        )
    else:
        return newton.solvers.NumericalSolverConfig(
            solver_type=solver_type,
            max_iterations=20,
            omega=0.5,
            relax=0.9,
            alpha=alpha,
            recovery_speed=recovery_speed,
            reg=1e-4,
            tolerance=0.0,
            position_correction=position_correction,
        )


def _make_position_correction(name: str) -> newton.solvers.NumericalSolverConfig:
    """Build a position-correction solver config matching the joint solver kind."""
    solver_type = _SOLVER_TYPE_MAP.get(name)
    if solver_type is None:
        raise ValueError(f"Unknown joint solver: {name}")
    return newton.solvers.NumericalSolverConfig(solver_type=solver_type, reg=1e-6)


class Example:
    """Free-floating humanoid simulation with a configurable joint solver."""

    def __init__(self, viewer, args=None):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer
        self.num_worlds = getattr(args, "num_worlds", 4) if args else 4
        self.no_cuda_graph = getattr(args, "no_cuda_graph", False) if args else False
        self.joint_solver_name = getattr(args, "joint_solver", "sparse_ldl") if args else "sparse_ldl"
        self.baumgarte_only = getattr(args, "baumgarte_only", False) if args else False

        humanoid = newton.ModelBuilder()
        humanoid.default_joint_cfg = newton.ModelBuilder.JointDofConfig(limit_ke=1.0e3, limit_kd=1.0e1, friction=1e-5)
        humanoid.default_shape_cfg.ke = 5.0e4
        humanoid.default_shape_cfg.kd = 5.0e2
        humanoid.default_shape_cfg.kf = 1.0e3
        humanoid.default_shape_cfg.mu = 0.75
        humanoid.default_shape_cfg.gap = 0.01
        humanoid.default_shape_cfg.margin = 0.001

        mjcf_filename = newton.examples.get_asset("nv_humanoid.xml")
        humanoid.add_mjcf(
            mjcf_filename,
            ignore_names=["floor", "ground"],
            xform=wp.transform(wp.vec3(0, 0, 1.3)),
            parse_sites=False,
        )

        for i in range(len(humanoid.joint_target_ke)):
            humanoid.joint_target_ke[i] = 50
            humanoid.joint_target_kd[i] = 5

        builder = newton.ModelBuilder()
        builder.replicate(humanoid, self.num_worlds)
        builder.add_ground_plane()

        device = wp.get_device() if args is None else args.device
        self.model = builder.finalize(device=device)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # Run collision first to populate model.rigid_contact_max before creating solver
        self.contacts = self.model.collide(self.state_0)

        contact_config = newton.solvers.NumericalSolverConfig(
            solver_type=SolverType.SPARSE_JACOBI,
            max_iterations=5,
            omega=0.5,
            relax=0.9,
            alpha=0.001,
            recovery_speed=0.5,
            reg=1e-4,
            tolerance=1e-5,
        )

        # Baumgarte-only mode: no position correction, rely on velocity-level Baumgarte
        position_correction = None if self.baumgarte_only else _make_position_correction(self.joint_solver_name)

        self.solver = newton.solvers.SolverDVI(
            self.model,
            joint_solver=_create_joint_solver(
                self.joint_solver_name,
                self.baumgarte_only,
                position_correction=position_correction,
            ),
            contact_solver=contact_config,
            angular_damping=0.01,
            enable_contacts=True,
            enable_timers=False,
        )

        self.solver.finalize_for_capture(self.state_0)

        self.viewer.set_model(self.model)

        self.capture()

    def capture(self) -> None:
        self.graph = None
        if self.model.device.is_cuda and not self.no_cuda_graph:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self) -> None:
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.contacts = self.model.collide(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self) -> None:
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        if self.contacts is not None:
            self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_final(self) -> None:
        body_q = self.state_0.body_q.numpy()
        body_qd = self.state_0.body_qd.numpy()
        if not np.isfinite(body_q).all():
            raise AssertionError("body_q contains non-finite values")
        if not np.isfinite(body_qd).all():
            raise AssertionError("body_qd contains non-finite values")
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "all bodies are above the ground",
            lambda q, qd: q[2] > -0.05,
        )

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_broad_phase_arg(parser)
        parser.add_argument(
            "--num-worlds",
            type=int,
            default=4,
            help="Total number of simulated humanoid worlds.",
        )
        parser.add_argument(
            "--joint-solver",
            type=str,
            default="sparse_ldl",
            choices=JOINT_SOLVER_CHOICES,
            help="Numerical solver for joint constraints.",
        )
        parser.add_argument(
            "--no-cuda-graph",
            action="store_true",
            default=False,
            help="Disable CUDA graph capture (for debugging).",
        )
        parser.add_argument(
            "--compare-solvers",
            action="store_true",
            default=False,
            help=(
                "Skip the viewer and time every solver in JOINT_SOLVER_CHOICES "
                "for --num-frames steps each. Prints a comparison table."
            ),
        )
        parser.add_argument(
            "--bench-warmup",
            type=int,
            default=20,
            help="Warmup frames before timing in --compare-solvers mode.",
        )
        parser.add_argument(
            "--baumgarte-only",
            action="store_true",
            default=False,
            help="Use Baumgarte-only stabilization (no position correction).",
        )
        return parser


class _NullViewer:
    """Minimal stand-in for the Newton viewer used in --compare-solvers."""

    def set_model(self, model):
        pass

    def begin_frame(self, t):
        pass

    def log_state(self, state):
        pass

    def log_contacts(self, contacts, state):
        pass

    def end_frame(self):
        pass

    def is_running(self):
        return False

    def should_step(self):
        return False

    def apply_forces(self, state):
        pass

    def close(self):
        pass


def _measure_solver(*, joint_solver: str, args, num_frames: int, warmup: int) -> dict:
    """Run one solver for ``num_frames`` and return wall-clock timings."""
    args.joint_solver = joint_solver
    example = Example(viewer=_NullViewer(), args=args)

    for _ in range(warmup):
        example.step()
    if example.model.device.is_cuda:
        wp.synchronize_device(example.model.device)

    t0 = time.perf_counter()
    for _ in range(num_frames):
        example.step()
    if example.model.device.is_cuda:
        wp.synchronize_device(example.model.device)
    elapsed = time.perf_counter() - t0

    return {
        "solver": joint_solver,
        "frames": num_frames,
        "elapsed_s": elapsed,
        "ms_per_frame": 1000.0 * elapsed / num_frames,
        "fps": num_frames / elapsed,
        "joints": int(example.model.joint_count),
        "bodies": int(example.model.body_count),
    }


def _run_solver_comparison(args) -> None:
    """Time every entry in JOINT_SOLVER_CHOICES and print a comparison table."""
    num_frames = max(int(getattr(args, "num_frames", 200) or 200), 1)
    warmup = max(int(getattr(args, "bench_warmup", 20) or 0), 0)

    device = wp.get_device() if args.device is None else wp.get_device(args.device)

    print(
        f"\nHumanoid solver benchmark "
        f"(device={device}, num_worlds={args.num_worlds}, "
        f"frames={num_frames}, warmup={warmup})"
    )
    results = []
    for solver in JOINT_SOLVER_CHOICES:
        try:
            r = _measure_solver(joint_solver=solver, args=args, num_frames=num_frames, warmup=warmup)
        except Exception as exc:  # noqa: BLE001
            print(f"  {solver}: FAILED ({exc})")
            continue
        results.append(r)

    if not results:
        return

    print(f"\nModel: {results[0]['bodies']} bodies, {results[0]['joints']} joints")
    print(f"{'solver':<20s} {'ms/frame':>10s} {'fps':>10s} {'rel':>8s}")
    fastest = min(r["ms_per_frame"] for r in results)
    for r in results:
        rel = r["ms_per_frame"] / fastest
        print(f"{r['solver']:<20s} {r['ms_per_frame']:>10.3f} {r['fps']:>10.1f} {rel:>7.2f}x")


if __name__ == "__main__":
    parser = Example.create_parser()
    pre_args, _ = parser.parse_known_args()
    if getattr(pre_args, "compare_solvers", False):
        args = parser.parse_args()
        _run_solver_comparison(args)
    else:
        viewer, args = newton.examples.init(parser)
        newton.examples.run(Example(viewer, args), args)
