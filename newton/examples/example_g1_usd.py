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

###########################################################################
# Example G1
#
# Shows how to set up a simulation of a rigid-body humanoid articulation
# from a XML using the newton.ModelBuilder().
# Note this example does not include a trained policy.
#
# Users can pick bodies by right-clicking and dragging with the mouse.
#
###########################################################################

import warp as wp

wp.config.enable_backward = False

import newton
import newton.examples
import newton.utils


class Example:
    def __init__(self, stage_path="example_g1.usd", num_envs=8, use_cuda_graph=True):
        self.num_envs = num_envs
        self.usd_xpbd = False
        self.use_mujoco = False
        articulation_builder = newton.ModelBuilder()

        newton.utils.parse_usd(
            newton.examples.get_asset("g1_minimal.usd"),
            articulation_builder,
            joint_drive_gains_scaling=1.0,
            collapse_fixed_joints=False,
            enable_self_collisions=False,
            # collapse_fixed_joints=True,
        )
        articulation_builder.approximate_meshes("bounding_box")

        spacing = 3.0
        sqn = int(wp.ceil(wp.sqrt(float(self.num_envs))))

        builder = newton.ModelBuilder()
        for i in range(self.num_envs):
            pos = wp.vec3((i % sqn) * spacing, (i // sqn) * spacing, 1.0)
            builder.add_builder(articulation_builder, xform=wp.transform(pos, wp.quat_identity()))
        builder.add_ground_plane()
        builder.gravity = wp.vec3(0.0, 0.0, -9.81)

        self.sim_time = 0.0
        fps = 200
        self.frame_dt = 1.0 / fps
        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps
        # finalize model
        self.model = builder.finalize()

        self.control = self.model.control()
        # self.solver = newton.solvers.FeatherstoneSolver(self.model)
        # self.solver = newton.solvers.SemiImplicitSolver(self.model, joint_attach_kd=100, joint_attach_ke= 1000)
        if self.usd_xpbd:
            self.solver = newton.solvers.XPBDSolver(self.model, iterations=20)
        else:
            self.solver = newton.solvers.MuJoCoSolver(
                self.model,
                use_mujoco=self.use_mujoco,
                solver="newton",
                integrator="euler",
                iterations=20,
                ls_iterations=20,
                nefc_per_env=1000,
                ncon_per_env=1000,
                contact_stiffness_time_const=0.01,
                save_to_mjcf="mjwarp.xml",
            )
        self.renderer = None

        if stage_path:
            self.renderer = newton.utils.SimRendererOpenGL(
                path=stage_path,
                model=self.model,
                scaling=1.0,
                screen_width=1280,
                screen_height=720,
                camera_pos=(-2, 5, 6),
                camera_front=(0.0, -2, -2),
            )

        self.state_0, self.state_1 = self.model.state(), self.model.state()

        newton.sim.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self.contacts = None
        if not self.use_mujoco:
            self.contacts = self.model.collide(self.state_0)
        self.use_cuda_graph = (
            not getattr(self.solver, "use_mujoco", False) and wp.get_device().is_cuda and use_cuda_graph
        )

        if self.use_cuda_graph:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        if not self.use_mujoco:
            self.contacts = self.model.collide(self.state_0)
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            if self.renderer and hasattr(self.renderer, "apply_picking_force"):
                self.renderer.apply_picking_force(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        with wp.ScopedTimer("step", active=True):
            if self.use_cuda_graph:
                wp.capture_launch(self.graph)
            else:
                self.simulate()

        self.sim_time += self.frame_dt

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
        default="example_g1.usd",
        help="Path to the output USD file.",
    )
    parser.add_argument("--num-frames", type=int, default=12000, help="Total number of frames.")
    parser.add_argument("--num-envs", type=int, default=1, help="Total number of simulated environments.")
    parser.add_argument(
        "--show-mujoco-viewer",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Toggle MuJoCo viewer next to Newton renderer when MuJoCoSolver is active.",
    )
    parser.add_argument("--use-cuda-graph", default=True, action=argparse.BooleanOptionalAction)

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(stage_path=args.stage_path, num_envs=args.num_envs, use_cuda_graph=args.use_cuda_graph)

        show_mujoco_viewer = True  # example.use_mujoco
        if show_mujoco_viewer:
            import mujoco
            import mujoco.viewer
            import mujoco_warp

            mjm, mjd = example.solver.mj_model, example.solver.mj_data
            m, d = example.solver.mjw_model, example.solver.mjw_data
            viewer = mujoco.viewer.launch_passive(mjm, mjd)

        for _ in range(args.num_frames):
            example.step()
            example.render()

            if show_mujoco_viewer:
                if not example.solver.use_mujoco:
                    mujoco_warp.get_data_into(mjd, mjm, d)
                viewer.sync()

        if example.renderer:
            example.renderer.save()
