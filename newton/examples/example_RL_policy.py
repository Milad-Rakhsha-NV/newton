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
# Example Robot control via keyboard
#
# Shows how to control robot pretrained in IL via mjwarp.
#
###########################################################################

import torch
import warp as wp

import newton
import newton.examples
import newton.utils
from newton.examples.policy_utils import (
    RobotKeyboardController,
    compute_obs,
    find_physx_mjwarp_mapping,
    load_policy_and_setup_tensors,
)
from newton.examples.robot_configs import G1_23DOF, G1_29DOF, Anymal

wp.config.enable_backward = False


class Example:
    def __init__(self, asset, student_policy):
        self.student_policy = student_policy
        self.device = wp.get_device()
        # Convert Warp device to PyTorch device string
        self.torch_device = "cuda" if self.device.is_cuda else "cpu"
        self.use_mujoco = False

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
            armature=0.1,
            limit_ke=1.0e3,
            limit_kd=1.0e1,
        )
        builder.default_shape_cfg.ke = 5.0e4
        builder.default_shape_cfg.kd = 5.0e2
        builder.default_shape_cfg.kf = 1.0e3
        builder.default_shape_cfg.mu = 0.75

        if False:
            newton.utils.parse_mjcf(
                newton.examples.get_asset(asset),
                builder,
                collapse_fixed_joints=False,
                up_axis="Z",
                enable_self_collisions=False,
            )
        else:
            newton.utils.parse_usd(
                newton.examples.get_asset(asset),
                builder,
                joint_drive_gains_scaling=1.0,
                collapse_fixed_joints=False,
                enable_self_collisions=False,
                joint_ordering="dfs",
            )
            builder.approximate_meshes("convex_hull")

        builder.add_ground_plane()
        builder.gravity = wp.vec3(0.0, 0.0, -9.81)
        self.sim_time = 0.0
        self.sim_step = 0
        fps = 200
        self.frame_dt = 1.0e0 / fps

        self.sim_substeps = 1
        self.sim_dt = self.frame_dt / self.sim_substeps

        builder.joint_q[:3] = [0.0, 0.0, 0.76]
        builder.joint_q[3:7] = [0.0, 0.0, 0.7071, 0.7071]
        builder.joint_q[7:] = config.mjw_joint_pos

        for i in range(len(builder.joint_dof_mode)):
            builder.joint_dof_mode[i] = newton.JOINT_MODE_TARGET_POSITION

        for i in range(len(config.mjw_joint_stiffness)):
            builder.joint_target_ke[i + 6] = config.mjw_joint_stiffness[i]
            builder.joint_target_kd[i + 6] = config.mjw_joint_damping[i]
            builder.joint_armature[i + 6] = config.mjw_joint_armature[i]

        self.model = builder.finalize()
        self.solver = newton.solvers.MuJoCoSolver(
            self.model,
            use_mujoco=self.use_mujoco,
            solver="newton",
            ncon_per_env=30,
            contact_stiffness_time_const=0.01,
            save_to_mjcf="assets/robot.xml",
        )

        self.renderer = newton.utils.SimRendererOpenGL(self.model, "RL Policy Example")
        self.state_temp = self.model.state()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0)
        newton.sim.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

        # Pre-compute tensors that don't change during simulation
        self.physx_to_mjc_indices = torch.tensor(
            [physx_to_mjc[i] for i in range(len(physx_to_mjc))], device=self.torch_device
        )
        self.mjc_to_physx_indices = torch.tensor(
            [mjc_to_physx[i] for i in range(len(mjc_to_physx))], device=self.torch_device
        )
        self.gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=self.torch_device, dtype=torch.float32).unsqueeze(0)
        self.command = torch.zeros((1, 3), device=self.torch_device, dtype=torch.float32)

        self.use_cuda_graph = self.device.is_cuda and wp.is_mempool_enabled(wp.get_device()) and not self.use_mujoco
        if self.use_cuda_graph:
            torch_tensor = torch.zeros(config.num_dofs + 6, device=self.torch_device, dtype=torch.float32)
            self.control.joint_target = wp.from_torch(torch_tensor, dtype=wp.float32, requires_grad=False)
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        state_0_dict = self.state_0.__dict__
        state_1_dict = self.state_1.__dict__
        state_temp_dict = self.state_temp.__dict__
        self.contacts = self.model.collide(self.state_0)
        for i in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            if i < self.sim_substeps - 1 or not self.use_cuda_graph:
                # we can just swap the state references
                self.state_0, self.state_1 = self.state_1, self.state_0
            elif self.use_cuda_graph:
                # swap states by copying the state arrays for graph capture
                for key, value in state_0_dict.items():
                    if isinstance(value, wp.array):
                        if key not in state_temp_dict:
                            state_temp_dict[key] = wp.empty_like(value)
                        state_temp_dict[key].assign(value)
                        state_0_dict[key].assign(state_1_dict[key])
                        state_1_dict[key].assign(state_temp_dict[key])

    def step(self):
        with wp.ScopedTimer("step"):
            obs = compute_obs(
                self.student_policy,
                self.act,
                self.state_0,
                self.joint_pos_initial,
                self.torch_device,
                self.physx_to_mjc_indices,
                self.gravity_vec,
                self.command,
            )
            with torch.no_grad():
                self.act = self.policy(obs)
                self.rearranged_act = torch.index_select(self.act, 1, self.mjc_to_physx_indices)
                a = self.joint_pos_initial + 0.5 * self.rearranged_act
                a_with_zeros = torch.cat([torch.zeros(6, device=self.torch_device, dtype=torch.float32), a.squeeze(0)])
                a_wp = wp.from_torch(a_with_zeros, dtype=wp.float32, requires_grad=False)
                wp.copy(self.control.joint_target, a_wp)

            for _ in range(decimation):
                if self.use_cuda_graph:
                    wp.capture_launch(self.graph)
                else:
                    self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        if self.renderer is None:
            return

        with wp.ScopedTimer("render"):
            self.renderer.begin_frame(self.sim_time)
            self.renderer.render(self.state_0)
            self.renderer.end_frame()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument("--num-frames", type=int, default=100000, help="Total number of frames.")
    parser.add_argument("--robot", type=str, default="g1_29dof", help="Robot type: g1_29dof, g1_23dof, anymal")
    parser.add_argument("--physx", action=argparse.BooleanOptionalAction)
    parser.add_argument("--student", action=argparse.BooleanOptionalAction)

    args = parser.parse_known_args()[0]
    robots = {"g1_29dof": G1_29DOF, "g1_23dof": G1_23DOF, "anymal": Anymal}

    config = robots[args.robot]()
    mjc_to_physx = list(range(config.num_dofs))
    physx_to_mjc = list(range(config.num_dofs))
    decimation = 1

    with wp.ScopedDevice(args.device):
        if args.physx:
            policy_path = config.policy_path["physx"]
            mjc_to_physx, physx_to_mjc = find_physx_mjwarp_mapping()
        else:
            if args.student:
                policy_path = config.policy_path["mjw_student"]
            else:
                policy_path = config.policy_path["mjw"]

        example = Example(config.asset_path, args.student)

        # Use utility function to load policy and setup tensors
        load_policy_and_setup_tensors(example, policy_path, config.num_dofs, slice(7, None))

        # Initialize keyboard controller
        keyboard_controller = RobotKeyboardController(
            command_size=3,  # [forward, lateral, rotation]
            step_size=0.005,
            command_limits=(-1.0, 1.0),
            window_title="Robot Controller",
        )

        show_mujoco_viewer = False
        if show_mujoco_viewer:
            import mujoco
            import mujoco.viewer
            import mujoco_warp

            mjm, mjd = example.solver.mj_model, example.solver.mj_data
            m, d = example.solver.mjw_model, example.solver.mjw_data
            viewer = mujoco.viewer.launch_passive(mjm, mjd)

        running = True
        frame_count = 0
        for _ in range(args.num_frames):
            if not running:
                break

            # Handle keyboard input and check if we should continue
            running = keyboard_controller.update(verbose=False)

            # Update the robot's command from the keyboard controller
            cpu_command = keyboard_controller.get_command()
            example.command = cpu_command.to(example.torch_device)

            # Print current command values for debugging
            if frame_count % 180 == 0:  # Every second at 100 FPS
                cmd = example.command[0]
                kb_cmd = keyboard_controller.get_command()[0]
                print(
                    f"Frame {frame_count}: Robot cmd="
                    f"[{cmd[0].item():.3f}, {cmd[1].item():.3f}], "
                    f"KB cmd=[{kb_cmd[0].item():.3f}, "
                    f"{kb_cmd[1].item():.3f}]"
                )

            example.step()
            example.render()
            if show_mujoco_viewer:
                if not example.solver.use_mujoco:
                    mujoco_warp.get_data_into(mjd, mjm, d)
                viewer.sync()

            frame_count += 1

        if example.renderer:
            example.renderer.save()

        keyboard_controller.cleanup()
