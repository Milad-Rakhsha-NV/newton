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
# Example G1 Robot control via keyboard
#
# Shows how to control G1 robot pretrained in IL via mjwarp.
# Added keyboard control functionality:
#   - Arrow keys control movement direction (forward/backward/left/right)
#   - Command values range from -1.0 to 1.0
#   - Spacebar resets movement to zero
#   - Requires pygame for keyboard input
#
###########################################################################

import torch
import warp as wp
from robot_keyboard_controller import RobotKeyboardController

import newton
import newton.examples
import newton.utils
from newton.examples.policy_utils import compute_obs, load_policy_and_setup_tensors

wp.config.enable_backward = False
num_dofs = 37
mjc_to_physx = list(range(num_dofs))
physx_to_mjc = list(range(num_dofs))
mjwarp_joint_names = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "torso_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_pitch_joint",
    "left_elbow_roll_joint",
    "left_five_joint",
    "left_six_joint",
    "left_three_joint",
    "left_four_joint",
    "left_zero_joint",
    "left_one_joint",
    "left_two_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint",
    "right_elbow_roll_joint",
    "right_five_joint",
    "right_six_joint",
    "right_three_joint",
    "right_four_joint",
    "right_zero_joint",
    "right_one_joint",
    "right_two_joint",
]
physx_joint_names = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "torso_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_elbow_pitch_joint",
    "right_elbow_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_elbow_roll_joint",
    "right_elbow_roll_joint",
    "left_five_joint",
    "left_three_joint",
    "left_zero_joint",
    "right_five_joint",
    "right_three_joint",
    "right_zero_joint",
    "left_six_joint",
    "left_four_joint",
    "left_one_joint",
    "right_six_joint",
    "right_four_joint",
    "right_one_joint",
    "left_two_joint",
    "right_two_joint",
]
mjwarp_joint_pos = [
    -0.2,
    0.0,
    0.0,
    0.42,
    -0.23,
    0.0,
    -0.2,
    0.0,
    0.0,
    0.42,
    -0.23,
    0.0,
    0.0,
    0.35,
    0.16,
    0.0,
    0.87,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.52,
    0.35,
    -0.16,
    0.0,
    0.87,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    -1.0,
    -0.52,
]
mjwarp_joint_stiffness = [
    200.0,
    150.0,
    150.0,
    200.0,
    20.0,
    20.0,
    200.0,
    150.0,
    150.0,
    200.0,
    20.0,
    20.0,
    200.0,
    40.0,
    40.0,
    40.0,
    40.0,
    40.0,
    40.0,
    40.0,
    40.0,
    40.0,
    40.0,
    40.0,
    40.0,
    40.0,
    40.0,
    40.0,
    40.0,
    40.0,
    40.0,
    40.0,
    40.0,
    40.0,
    40.0,
    40.0,
    40.0,
]
mjwarp_joint_damping = [
    5.0,
    5.0,
    5.0,
    5.0,
    2.0,
    2.0,
    5.0,
    5.0,
    5.0,
    5.0,
    2.0,
    2.0,
    5.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
]

decimation = 1


def find_physx_mjwarp_mapping():
    """
    Finds the mapping between PhysX and MJWarp joint names.
    Returns a tuple of two lists: (mjc_to_physx, physx_to_mjc).
    """
    mjc_to_physx = []
    physx_to_mjc = []
    for j in mjwarp_joint_names:
        if j in physx_joint_names:
            mjc_to_physx.append(physx_joint_names.index(j))

    for j in physx_joint_names:
        if j in mjwarp_joint_names:
            physx_to_mjc.append(mjwarp_joint_names.index(j))

    print("Mapping from MJWarp to physx:", mjc_to_physx)
    print("Mapping from physx to MJWarp:", physx_to_mjc)
    return mjc_to_physx, physx_to_mjc


class Example:
    def __init__(self, stage_path=None, headless=False, parse_usd=True):
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

        # if False:
        # 	asset_path = newton.utils.download_asset("g1_description")
        # 	newton.utils.parse_mjcf(
        # 		str(asset_path / "g1_29dof_with_hand_rev_1_0.xml"),
        # 		builder,
        # 		collapse_fixed_joints=False,
        # 		up_axis="Z",
        # 		enable_self_collisions=False,
        # 	)
        # else:
        newton.utils.parse_usd(
            newton.examples.get_asset("g1_minimal_modified.usd"),
            builder,
            joint_drive_gains_scaling=1.0,
            collapse_fixed_joints=False,
            enable_self_collisions=False,
            joint_ordering="dfs",
        )
        builder.approximate_meshes("bounding_box")

        builder.add_ground_plane()
        builder.gravity = wp.vec3(0.0, 0.0, -9.81)
        self.sim_time = 0.0
        self.sim_step = 0
        fps = 200
        self.frame_dt = 1.0e0 / fps

        self.sim_substeps = 5
        self.sim_dt = self.frame_dt / self.sim_substeps

        builder.joint_q[:3] = [0.0, 0.0, 0.76]
        builder.joint_q[3:7] = [0.0, 0.0, 0.7071, 0.7071]
        builder.joint_q[7:] = mjwarp_joint_pos

        for i in range(len(builder.joint_dof_mode)):
            builder.joint_dof_mode[i] = newton.JOINT_MODE_TARGET_POSITION

        for i in range(len(mjwarp_joint_stiffness)):
            builder.joint_target_ke[i + 6] = mjwarp_joint_stiffness[i]
            builder.joint_target_kd[i + 6] = mjwarp_joint_damping[i]
            builder.joint_armature[i + 6] = 0.1
        self.model = builder.finalize()
        self.solver = newton.solvers.MuJoCoSolver(
            self.model,
            use_mujoco=self.use_mujoco,
            solver="newton",
            ncon_per_env=30,
            contact_stiffness_time_const=0.01,
            save_to_mjcf="g1_policy.xml",
        )

        self.renderer = None  # if headless else newton.utils.SimRendererOpenGL(self.model, stage_path)
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
            torch_tensor = torch.zeros(num_dofs + 6, device=self.torch_device, dtype=torch.float32)
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
    parser.add_argument(
        "--stage-path",
        type=lambda x: None if x == "None" else str(x),
        help="Path to the output URDF file.",
    )
    parser.add_argument("--num-frames", type=int, default=100000, help="Total number of frames.")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction)
    parser.add_argument("--policy-path", type=str, default="assets/physx_g1.pt", help="Path to the policy model.")
    parser.add_argument("--physx", action=argparse.BooleanOptionalAction)

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        if args.physx:
            policy_path = "assets/physx_g1.pt"
            mjc_to_physx, physx_to_mjc = find_physx_mjwarp_mapping()
            q0 = torch.tensor(mjwarp_joint_pos)
            physx_to_mjc_tensor = torch.tensor(physx_to_mjc, dtype=torch.int64)
            physx_init_joint_test = torch.index_select(q0, 0, physx_to_mjc_tensor)
            print("physx_init_joint_test:", physx_init_joint_test)
        else:
            policy_path = "assets/g1_policy.pt"

        example = Example(stage_path=args.stage_path, headless=args.headless)

        # Use utility function to load policy and setup tensors
        load_policy_and_setup_tensors(example, policy_path, num_dofs, slice(7, None))

        # Initialize keyboard controller
        keyboard_controller = RobotKeyboardController(
            command_size=3,  # [forward, lateral, rotation]
            step_size=0.05,
            command_limits=(-1.0, 1.0),
            window_title="G1 Robot Keyboard Controller",
        )

        show_mujoco_viewer = True
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
            if frame_count % 60 == 0:  # Every second at 60 FPS
                cmd = example.command[0]
                kb_cmd = keyboard_controller.get_command()[0]
                print(
                    f"Frame {frame_count}: Robot cmd="
                    f"[{cmd[0].item():.3f}, {cmd[1].item():.3f}], "
                    f"KB cmd=[{kb_cmd[0].item():.3f}, "
                    f"{kb_cmd[1].item():.3f}]"
                )

            example.step()
            # example.render()
            if show_mujoco_viewer:
                if not example.solver.use_mujoco:
                    mujoco_warp.get_data_into(mjd, mjm, d)
                viewer.sync()

            frame_count += 1

        if example.renderer:
            example.renderer.save()

        keyboard_controller.cleanup()
