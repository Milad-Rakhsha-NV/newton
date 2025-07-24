# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
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

from __future__ import annotations

import warp as wp

wp.config.enable_backward = False

import newton
import newton.examples
import newton.utils
from newton.utils.selection import ArticulationView

COLLAPSE_FIXED_JOINTS = False
VERBOSE = True
USE_CUDA_GRAPH = True


class Example:
    def __init__(self, stage_path, num_envs):
        up_axis = newton.Axis.Z
        env_builder = newton.ModelBuilder(up_axis=up_axis)
        # for i in range(num_envs):
        newton.utils.parse_usd(
            newton.examples.get_asset("anymal_c.usd"),
            env_builder,
            joint_drive_gains_scaling=1.0,
            collapse_fixed_joints=COLLAPSE_FIXED_JOINTS,
            enable_self_collisions=False,
            joint_ordering="dfs",
        )
        for i in range(12):
            env_builder.joint_target_ke[i + 6] = 50
            env_builder.joint_target_kd[i + 6] = 1
            env_builder.joint_armature[i + 6] = 0.1

        builder = newton.ModelBuilder()
        for i in range(num_envs):
            builder.add_builder(env_builder, xform=wp.transform([0, 2 * i, 1], wp.quat_identity()))

        builder.add_ground_plane()

        # finalize model
        self.model = builder.finalize()

        self.solver = newton.solvers.MuJoCoSolver(self.model)

        self.renderer = None
        if stage_path:
            self.renderer = newton.utils.SimRendererOpenGL(
                path=stage_path,
                model=self.model,
                scaling=2.0,
                up_axis=str(up_axis),
                screen_width=1280,
                screen_height=720,
                camera_pos=(0, 4, 30),
            )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.sim_time = 0.0
        fps = 200
        self.frame_dt = 1.0 / fps

        self.sim_substeps = 5
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.next_reset = 0.0
        self.step_count = 0

        # ===========================================================
        # create articulation views
        # ===========================================================

        self.use_cuda_graph = wp.get_device().is_cuda and USE_CUDA_GRAPH
        if self.use_cuda_graph:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            # explicit collisions needed without MuJoCo solver
            if not isinstance(self.solver, newton.solvers.MuJoCoSolver):
                contacts = self.model.collide(self.state_0)
            else:
                contacts = None

            self.solver.step(self.state_0, self.state_1, self.control, contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        # self.ants.set_dof_forces(self.control, dof_forces)
        if self.use_cuda_graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt
        self.step_count += 1

    def render(self):
        if self.renderer is None:
            return

        with wp.ScopedTimer("render", active=False):
            self.renderer.begin_frame(self.sim_time)
            self.renderer.render(self.state_0)
            self.renderer.end_frame()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--stage-path",
        type=lambda x: None if x == "None" else str(x),
        default="example_selection_articulations.usd",
        help="Path to the output USD file.",
    )
    parser.add_argument("--num-frames", type=int, default=2000, help="Total number of frames.")
    args = parser.parse_known_args()[0]
    example = Example("anymal.usd", 4)
    anymals = ArticulationView(
        example.model, "/anymal/base", verbose=VERBOSE, exclude_joint_types=[newton.JOINT_FREE, newton.JOINT_FIXED]
    )

    for _ in range(args.num_frames):
        example.step()
        example.render()

    print("Base heights:\n", anymals.get_root_transforms(example.state_0))
    if example.renderer:
        example.renderer.save()
