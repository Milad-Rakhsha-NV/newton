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
# Example Standing DR Legs (DVI Solver)
#
# Loads the Disney Research Legs robot and stands it on a ground plane with
# PD position targets holding the default pose. The DVI contact solver
# resolves foot-ground contact forces.
#
# Usage::
#
#     python -m newton.examples standing_drlegs
#
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.solvers.dvi.numerical_solver.base import NumericalSolverConfig
from newton.solvers import ActuatorIntegration, SolverType
from newton import JointTargetMode


GRAVITY = 9.81
PHYSICS_DT = 0.004
DRIVEN_KE = 5.0
DRIVEN_KD = 0.2

DR_LEGS_ACTUATED = [
    "j1_l_i", "j2_l_i", "j6_l_i", "j7_l_i", "j2_l_o", "j7_l_o",
    "j1_r_i", "j2_r_i", "j6_r_i", "j7_r_i", "j2_r_o", "j7_r_o",
]


def _short(label):
    return str(label).rsplit("/", 1)[-1]


def _build_model(device, mesh=True):
    asset_path = newton.utils.download_asset("disneyresearch")
    fname = "dr_legs_with_meshes_and_boxes.usda" if mesh else "dr_legs_with_boxes.usda"
    asset_file = str(asset_path / "dr_legs/usd" / fname)

    b = newton.ModelBuilder(up_axis=newton.Axis.Z)
    b.default_shape_cfg.margin = 1e-6
    b.default_shape_cfg.gap = 0.005
    b.add_usd(
        asset_file,
        joint_ordering=None,
        force_show_colliders=True,
        force_position_velocity_actuation=True,
        collapse_fixed_joints=False,
        enable_self_collisions=False,
        hide_collision_shapes=True,
        floating=True,
    )

    # Lift the assembly so the lowest foot starts above the contact gap
    # threshold (0.005m) for a clean visual drop onto the ground plane.
    zs = [b.body_q[i][2] for i in range(b.body_count)]
    lowest = min(zs)
    gap = 0.02
    b.add_ground_plane()
    lift = -lowest + gap
    for i in range(b.body_count):
        t = b.body_q[i]
        b.body_q[i] = wp.transform(
            wp.vec3(t[0], t[1], t[2] + lift), wp.quat(t[3], t[4], t[5], t[6])
        )

    model = b.finalize(skip_validation_joints=True, device=device)
    model.rigid_contact_max = 65536
    _apply_gains(model)
    return model


def _apply_gains(model):
    ke = model.joint_target_ke.numpy()
    kd = model.joint_target_kd.numpy()
    mode = model.joint_target_mode.numpy()
    qd_start = model.joint_qd_start.numpy()
    n = len(ke)
    for j in range(model.joint_count):
        d = int(qd_start[j])
        if d >= n:
            continue
        sl = _short(model.joint_label[j])
        mode[d] = int(JointTargetMode.POSITION)
        if sl in DR_LEGS_ACTUATED:
            ke[d] = DRIVEN_KE
            kd[d] = DRIVEN_KD
        else:
            ke[d] = 0.0
            kd[d] = 0.0
    model.joint_target_ke.assign(wp.array(ke, dtype=wp.float32, device=model.device))
    model.joint_target_kd.assign(wp.array(kd, dtype=wp.float32, device=model.device))
    model.joint_target_mode.assign(wp.array(mode, dtype=wp.int32, device=model.device))


def _create_solver(model):
    jc = NumericalSolverConfig(
        solver_type=SolverType.SPARSE_LDL,
        max_iterations=50,
        alpha=0.0,
        recovery_speed=100000.0,
        reg=1e-4,
        diagonal_precondition=True,
        iterative_refinement_steps=1,
    )
    contact_cfg = NumericalSolverConfig(
        solver_type=SolverType.SPARSE_JACOBI,
        max_iterations=20,
        alpha=0.0,
        omega=0.2,
        recovery_speed=0.1,
    )
    joint_limit_cfg = NumericalSolverConfig(
        solver_type=SolverType.SPARSE_JACOBI,
        max_iterations=20,
        alpha=0.0,
        recovery_speed=0.5,
        reg=1e-4,
        diagonal_precondition=True,
    )
    return newton.solvers.SolverDVI(
        model,
        joint_solver=jc,
        contact_solver=contact_cfg,
        angular_damping=0.0,
        enable_contacts=True,
        enable_timers=False,
        actuator_integration=ActuatorIntegration.SEMI_IMPLICIT,
        joint_limit_ke_scale=0.0,
        joint_limit_solver=joint_limit_cfg,
    )


class Example:
    """DR Legs standing on a ground plane with PD targets holding the default pose."""

    def __init__(self, viewer, args=None):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        substeps = getattr(args, "substeps", 4) if args else 4
        self.sim_substeps = substeps
        self.sim_dt = PHYSICS_DT / substeps

        self.viewer = viewer
        device = wp.get_device() if args is None else args.device

        self.model = _build_model(device, mesh=True)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.pipeline = newton.CollisionPipeline(self.model)
        self.contacts = self.model.contacts(collision_pipeline=self.pipeline)
        self.solver = _create_solver(self.model)

        self.viewer.set_model(self.model)

    def step(self) -> None:
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
        self.sim_time += self.frame_dt

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_final(self) -> None:
        body_q = self.state_0.body_q.numpy()
        body_qd = self.state_0.body_qd.numpy()
        if not np.isfinite(body_q).all():
            raise AssertionError("body_q contains non-finite values")
        if not np.isfinite(body_qd).all():
            raise AssertionError("body_qd contains non-finite values")
        # The pelvis (body 0) should remain above the ground.
        pelvis_z = body_q[0][2]
        if pelvis_z < 0.05:
            raise AssertionError(f"Pelvis fell to z={pelvis_z:.4f}, expected > 0.05")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--substeps", type=int, default=4, help="Physics substeps per frame.")
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
