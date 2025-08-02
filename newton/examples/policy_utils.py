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
    student_policy: bool,
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
    joint_pos_rel = joint_pos_current.squeeze()[indices] - joint_pos_initial
    joint_vel_rel = joint_vel_current.squeeze()[indices]

    if student_policy:
        print("Using student policy observation format")
        obs = torch.cat(
            [a_vel_b.squeeze(), grav.squeeze(), command.squeeze(), joint_pos_rel, joint_vel_rel, actions.squeeze()],
            dim=0,
        )
    else:
        obs = torch.cat(
            [
                vel_b.squeeze(),
                a_vel_b.squeeze(),
                grav.squeeze(),
                command.squeeze(),
                joint_pos_rel,
                joint_vel_rel,
                actions.squeeze(),
            ],
            dim=0,
        )
    return obs


def load_policy_and_setup_tensors(example: Any, policy_path: str):
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


"""
Robot Keyboard Controller

A reusable keyboard control interface for robot command input using pygame.
Supports real-time command updates with visual feedback.
"""

from typing import Optional

import pygame
import torch


class RobotKeyboardController:
    """
    A generic keyboard controller for robot movement commands.

    Provides real-time keyboard input handling with visual feedback window.
    Commands are stored as torch tensors and can be customized for robots.
    """

    def __init__(
        self,
        command_size: int = 3,
        step_size: float = 0.01,
        command_limits: tuple[float, float] = (-1.0, 1.0),
        window_size: tuple[int, int] = (500, 400),
        window_title: str = "Robot Control - Use WASD Keys",
    ):
        """
        Initialize the keyboard controller.

        Args:
                        command_size: Size of command tensor
                                        (default 3 for [forward, lateral, rotation])
                        step_size: Increment/decrement step size for commands
                        command_limits: Min and max values for commands
                        window_size: pygame window dimensions
                        window_title: Window title
        """
        self.command_size = command_size
        self.step_size = step_size
        self.min_val, self.max_val = command_limits
        self.window_size = window_size

        # Initialize command tensor (always on CPU)
        self.command = torch.zeros((1, command_size), device="cpu", dtype=torch.float32)

        # Key mappings (can be customized)
        self.key_mappings = {
            pygame.K_w: ("forward", 0, 1),
            pygame.K_s: ("backward", 0, -1),
            pygame.K_a: ("left", 1, 1),
            pygame.K_d: ("right", 1, -1),
            pygame.K_q: ("rotate_left", 2, 1),
            pygame.K_e: ("rotate_right", 2, -1),
            pygame.K_SPACE: ("reset", -1, 0),
        }

        # Display settings
        self._control_screen: Optional[pygame.Surface] = None
        self._font: Optional[pygame.font.Font] = None
        self._small_font: Optional[pygame.font.Font] = None
        self._running = True

        # Initialize pygame
        self._init_pygame(window_title)

    def _init_pygame(self, title: str):
        """Initialize pygame and create the control window."""
        pygame.init()
        self._control_screen = pygame.display.set_mode(self.window_size)
        pygame.display.set_caption(title)

        # Initialize fonts
        pygame.font.init()
        self._font = pygame.font.Font(None, 48)
        self._small_font = pygame.font.Font(None, 28)

        # Ensure window has focus for keyboard input
        pygame.event.set_grab(False)
        pygame.key.set_repeat(1, 50)

        # Initial display update
        self._update_display()

    def update(self, verbose: bool = False) -> bool:
        """
        Update the controller state based on keyboard input.

        Args:
            verbose: If True, print command changes to console

        Returns:
            False if user wants to quit, True otherwise
        """
        # Check if we should continue (set by _update_display)
        if not self._running:
            return False

        # Process pygame events for reset command only (other events handled in _update_display)
        reset_pressed = False
        keys = pygame.key.get_pressed()

        # Check for reset key specifically
        if keys[pygame.K_SPACE]:
            reset_pressed = True

        # Reset all commands to zero at start (active control)
        self.command.fill_(0.0)

        # Handle reset command
        if reset_pressed:
            # Commands already reset above, just update display
            pass

        # Check which keys are currently pressed and set commands accordingly
        command_changed = False

        # Process currently pressed keys
        for key, (name, index, direction) in self.key_mappings.items():
            if keys[key] and name != "reset":
                if index < self.command_size and index >= 0:
                    # Use smaller range for lateral movement (left/right)
                    if index == 1:  # Lateral movement (left/right)
                        target_val = direction * 0.5  # Use 50% of max range
                    else:  # Forward/backward and rotation
                        target_val = direction * 1.0  # Use 80% of max range
                    target_val = torch.clamp(torch.tensor(target_val), self.min_val, self.max_val).item()
                    self.command[0, index] = target_val
                    command_changed = True

        # Update display every frame
        self._update_display()

        # Print feedback if requested
        if verbose and command_changed:
            cmd_str = ", ".join([f"{self.command[0, i].item():.3f}" for i in range(self.command_size)])
            print(f"Command: [{cmd_str}]")

        return self._running

    def _update_display(self):
        """Update the pygame window with current command values."""
        if self._control_screen is None:
            return

        # Process pygame events to ensure window responds and maintains focus
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._running = False

        # Additional event pump to ensure responsiveness
        pygame.event.pump()

        # Clear screen with dark blue background
        self._control_screen.fill((20, 30, 50))

        # Ensure fonts are initialized
        if self._font is None or self._small_font is None:
            try:
                pygame.font.init()
                self._font = pygame.font.Font(None, 48)
                self._small_font = pygame.font.Font(None, 28)
            except pygame.error:
                # If font initialization fails, skip rendering text
                pygame.display.flip()
                return

        # Get current command values directly from tensor (always on CPU)
        try:
            cmd_vals = [self.command[0, i].item() for i in range(self.command_size)]
        except (IndexError, RuntimeError):
            # If tensor access fails, use zeros
            cmd_vals = [0.0] * self.command_size

        # Create title
        try:
            if self._font is not None:
                title_text = self._font.render("Robot Control", True, (255, 255, 255))

                # Render command values with explicit formatting
                y_pos = 20
                self._control_screen.blit(title_text, (20, y_pos))
                y_pos += 60

                # Display command values with labels
                labels = ["Forward", "Lateral", "Rotation"]
                for i in range(min(len(cmd_vals), len(labels))):
                    val = cmd_vals[i]
                    label = labels[i]
                    # Color: green if non-zero, gray if zero
                    color = (100, 255, 100) if abs(val) > 0.01 else (150, 150, 150)

                    # Create text with explicit value formatting
                    text_str = f"{label}: {val:+.3f}"
                    cmd_text = self._font.render(text_str, True, color)

                    # Blit to screen
                    self._control_screen.blit(cmd_text, (20, y_pos))
                    y_pos += 45

                # Add some instructions at the bottom
                y_pos += 20
                instructions = [
                    "Controls:",
                    "WASD Keys: Move",
                    "QE Keys: Rotate",
                    "Space: Reset",
                    "Close window to exit",
                ]

                if self._small_font is not None:
                    for instruction in instructions:
                        color = (255, 255, 255) if instruction.endswith(":") else (200, 200, 200)
                        inst_surface = self._small_font.render(instruction, True, color)
                        self._control_screen.blit(inst_surface, (20, y_pos))
                        y_pos += 25

        except (pygame.error, TypeError):
            # If rendering fails, just clear the screen
            pass

        # Force display update with multiple methods to ensure visibility
        try:
            pygame.display.flip()
            pygame.display.update()
        except pygame.error:
            # If display update fails, continue anyway
            pass

    def get_command(self) -> torch.Tensor:
        """
        Get the current command tensor.

        Returns:
                        Current command as torch tensor of shape (1, command_size)
        """
        return self.command.clone()

    def set_command(self, command: torch.Tensor):
        """
        Set the command tensor directly.

        Args:
                        command: New command tensor
        """
        if command.shape != self.command.shape:
            raise ValueError(f"Command shape {command.shape} doesn't match expected {self.command.shape}")
        self.command = command.to("cpu")
        self._update_display()

    def reset_commands(self):
        """Reset all commands to zero."""
        self.command.fill_(0.0)
        self._update_display()

    def configure_key_mapping(self, key_mappings: dict[int, tuple[str, int, int]]):
        """
        Configure custom key mappings.

        Args:
                        key_mappings: dict mapping pygame keys to
                                        (name, command_index, direction)
        """
        self.key_mappings = key_mappings

    def cleanup(self):
        """Clean up pygame resources."""
        if pygame.get_init():
            pygame.quit()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
