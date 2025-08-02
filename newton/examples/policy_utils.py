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

"""
Policy utilities for robot control examples.

Common functions and classes used across different robot policy examples.
"""

from typing import Any

import torch

from newton.sim import State


@torch.jit.script
def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate a vector by the inverse of a quaternion.

    Args:
        q: The quaternion in (x, y, z, w). Shape is (..., 4).
        v: The vector in (x, y, z). Shape is (..., 3).

    Returns:
        The rotated vector in (x, y, z). Shape is (..., 3).
    """
    q_w = q[..., 3]  # w component is at index 3 for XYZW format
    q_vec = q[..., :3]  # xyz components are at indices 0, 1, 2
    a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    # for two-dimensional tensors, bmm is faster than einsum
    if q_vec.dim() == 2:
        c = q_vec * torch.bmm(q_vec.view(q.shape[0], 1, 3), v.view(q.shape[0], 3, 1)).squeeze(-1) * 2.0
    else:
        c = q_vec * torch.einsum("...i,...i->...", q_vec, v).unsqueeze(-1) * 2.0
    return a - b + c


def compute_obs(
    actions: torch.Tensor,
    state: State,
    joint_pos_initial: torch.Tensor,
    device: str,
    indices: torch.Tensor,
    gravity_vec: torch.Tensor,
    command: torch.Tensor,
) -> torch.Tensor:
    """Compute observation for robot policy.

    Args:
        actions: Previous actions tensor
        state: Current simulation state
        joint_pos_initial: Initial joint positions
        device: PyTorch device string
        indices: Index mapping for joint reordering
        gravity_vec: Gravity vector in world frame
        command: Command vector

    Returns:
        Observation tensor for policy input
    """
    # Extract state information with proper handling
    joint_q = state.joint_q if state.joint_q is not None else []
    joint_qd = state.joint_qd if state.joint_qd is not None else []

    root_quat_w = torch.tensor(joint_q[3:7], device=device, dtype=torch.float32).unsqueeze(0)
    root_lin_vel_w = torch.tensor(joint_qd[3:6], device=device, dtype=torch.float32).unsqueeze(0)
    root_ang_vel_w = torch.tensor(joint_qd[:3], device=device, dtype=torch.float32).unsqueeze(0)
    joint_pos_current = torch.tensor(joint_q[7:], device=device, dtype=torch.float32).unsqueeze(0)
    joint_vel_current = torch.tensor(joint_qd[6:], device=device, dtype=torch.float32).unsqueeze(0)

    vel_b = quat_rotate_inverse(root_quat_w, root_lin_vel_w)
    a_vel_b = quat_rotate_inverse(root_quat_w, root_ang_vel_w)
    grav = quat_rotate_inverse(root_quat_w, gravity_vec)
    joint_pos_rel = joint_pos_current - joint_pos_initial
    joint_vel_rel = joint_vel_current
    rearranged_joint_pos_rel = torch.index_select(joint_pos_rel, 1, indices)
    rearranged_joint_vel_rel = torch.index_select(joint_vel_rel, 1, indices)
    obs = torch.cat([vel_b, a_vel_b, grav, command, rearranged_joint_pos_rel, rearranged_joint_vel_rel, actions], dim=1)

    return obs


def load_policy_and_setup_tensors(example: Any, policy_path: str, num_dofs: int, joint_pos_slice: slice):
    """Load policy and setup initial tensors for robot control.

    Args:
        example: Robot example instance
        policy_path: Path to the policy file
        num_dofs: Number of degrees of freedom
        joint_pos_slice: Slice for extracting joint positions from state
    """
    device = example.torch_device
    print("[INFO] Loading policy from:", policy_path)
    example.policy = torch.jit.load(policy_path, map_location=device)

    # Handle potential None state
    joint_q = example.state_0.joint_q if example.state_0.joint_q is not None else []
    example.joint_pos_initial = torch.tensor(joint_q[joint_pos_slice], device=device, dtype=torch.float32).unsqueeze(0)
    example.act = torch.zeros(1, num_dofs, device=device, dtype=torch.float32)
    example.rearranged_act = torch.zeros(1, num_dofs, device=device, dtype=torch.float32)


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

    # print("Mapping from MJWarp to physx:", mjc_to_physx)
    # print("Mapping from physx to MJWarp:", physx_to_mjc)
    return mjc_to_physx, physx_to_mjc


"""
Robot Keyboard Controller

A simple keyboard control interface for robot command input.
"""

from typing import Any

import torch

try:
    import pygame  # type: ignore

    PYGAME_AVAILABLE = True
except ImportError:
    pygame = None  # type: ignore
    PYGAME_AVAILABLE = False


class RobotKeyboardController:
    """
    A simple keyboard controller for robot movement commands.
    """

    def __init__(
        self,
        command_size: int = 3,
        command_limits: tuple[float, float] = (-1.0, 1.0),
    ):
        """
        Initialize the keyboard controller.

        Args:
            command_size: Size of command tensor (default 3 for [forward, lateral, rotation])
            command_limits: Min and max values for commands
        """
        if not PYGAME_AVAILABLE:
            raise ImportError("pygame is required for RobotKeyboardController")

        self.command_size = command_size
        self.min_val, self.max_val = command_limits

        # Initialize command tensor
        self.command = torch.zeros((1, command_size), dtype=torch.float32)

        # Simple key mappings
        self.key_mappings = {
            pygame.K_w: (0, 1.0),  # forward
            pygame.K_s: (0, -1.0),  # backward
            pygame.K_a: (1, 0.5),  # left (reduced speed)
            pygame.K_d: (1, -0.5),  # right (reduced speed)
            pygame.K_q: (2, 1.0),  # rotate left
            pygame.K_e: (2, -1.0),  # rotate right
        }

        self._running = True
        pygame.init()
        pygame.font.init()

        # Create window for input and display
        self._screen = pygame.display.set_mode((400, 300))
        pygame.display.set_caption("Robot Control")

        # Initialize fonts
        self._font = pygame.font.Font(None, 28)
        self._small_font = pygame.font.Font(None, 24)

    def update(self, verbose: bool = False) -> bool:
        """
        Update the controller state based on keyboard input.

        Args:
            verbose: If True, print command changes to console

        Returns:
            False if user wants to quit, True otherwise
        """
        # Process events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._running = False
                return False

        # Reset commands
        self.command.fill_(0.0)

        # Check pressed keys
        keys = pygame.key.get_pressed()
        command_changed = False

        for key, (index, value) in self.key_mappings.items():
            if keys[key] and index < self.command_size:
                clamped_value = max(self.min_val, min(self.max_val, value))
                self.command[0, index] = clamped_value
                command_changed = True

        # Update display
        self._update_display()

        # Print feedback if requested
        if verbose and command_changed:
            cmd_str = ", ".join([f"{self.command[0, i].item():.3f}" for i in range(self.command_size)])
            print(f"Command: [{cmd_str}]")

        return self._running

    def _update_display(self):
        """Update the pygame window with current command values and instructions."""
        # Clear screen with dark background
        self._screen.fill((20, 30, 50))

        # Display current command values
        y_pos = 70

        instructions = [
            "Controls:",
            "W/S: Forward/Backward",
            "A/D: Left/Right",
            "Q/E: Rotate Left/Right",
            "Close window to exit",
        ]

        for instruction in instructions:
            color = (255, 255, 255) if instruction.endswith(":") else (200, 200, 200)
            inst_surface = self._small_font.render(instruction, True, color)
            self._screen.blit(inst_surface, (20, y_pos))
            y_pos += 25

        # Update display
        pygame.display.flip()

    def get_command(self) -> torch.Tensor:
        """Get the current command tensor."""
        return self.command.clone()

    def reset_commands(self):
        """Reset all commands to zero."""
        self.command.fill_(0.0)
        self._update_display()

    def cleanup(self):
        """Clean up pygame resources."""
        if PYGAME_AVAILABLE and pygame is not None and pygame.get_init():
            pygame.quit()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
