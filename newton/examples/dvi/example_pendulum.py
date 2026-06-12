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
# Example Multiple Pendulums (DVI Solver)
#
# Creates multiple pendulums side by side, each with a different joint type:
#   - revolute: Hinge joint (rotation about Y axis)
#   - spherical: Ball-and-socket joint (free rotation)
#   - prismatic: Slider joint (translation along X axis)
#   - cylindrical: Combined hinge + slider (rotation about Y, translation along X)
#
# This example tests constraint enforcement for different joint types simultaneously.
#
# The capsules are thin for visual clarity but have artificially increased
# inertia to avoid numerical instability with anisotropic mass matrices.
#
# Command: python -m newton.examples dvi_pendulum
#
###########################################################################

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverType


# Joint configuration for creating pendulums
JOINT_CONFIGS = {
    "revolute": {
        "description": "Hinge joint - rotation about Y axis only",
        "num_constraints": 5,  # 3 position + 2 angular
        "allowed_motion": ["wy"],
    },
    "spherical": {
        "description": "Ball joint - rotation about all axes",
        "num_constraints": 3,  # 3 position only
        "allowed_motion": ["wx", "wy", "wz"],
    },
    "prismatic": {
        "description": "Slider joint - translation along X axis only",
        "num_constraints": 5,  # 2 position + 3 angular
        "allowed_motion": ["vx"],
    },
    "cylindrical": {
        "description": "Cylindrical joint - translation along X + rotation about Y",
        "num_constraints": 4,  # 2 position + 2 angular
        "allowed_motion": ["vx", "wy"],
    },
}

# Default joint types to create
DEFAULT_JOINT_TYPES = ["revolute", "spherical", "prismatic", "cylindrical"]
# DEFAULT_JOINT_TYPES = ["revolute"]

# Default actuation gains
# NOTE: Joint angle = 0 at initial configuration. Target positions are relative to this.
# With gravity, there will be a small steady-state error (expected for PD control).
# Higher gains reduce error but may cause oscillation.
DEFAULT_KE = 100.0  # Position gain (stiffness)
DEFAULT_KD = 30.0  # Velocity gain (damping)


def create_pendulum(
    builder: newton.ModelBuilder,
    joint_type: str,
    pivot_position: wp.vec3,
    link_length: float = 1.0,
    link_radius: float = 0.05,  # Thin for visual clarity
    initial_angle: float = math.pi / 4,
    body_key_prefix: str = "",
    inertia_scale: float = 50.0,  # High inertia for stability with spherical joints
    enable_actuation: bool = False,
    target_pos: float = 0.0,
) -> Tuple[int, int, Optional[int]]:
    """
    Create a single pendulum with the specified joint type.

    Args:
        builder: Newton model builder.
        joint_type: Type of joint ("revolute", "spherical", "prismatic", "cylindrical").
        pivot_position: World position of the pivot point.
        link_length: Length of the pendulum link.
        link_radius: Radius of the capsule (thin by default).
        initial_angle: Initial angle from vertical (radians).
        body_key_prefix: Prefix for body keys.
        inertia_scale: Factor to artificially increase inertia for stability.
        enable_actuation: Whether to enable PD control actuation.
        target_pos: Target joint angle for actuation (relative to initial config).
            NOTE: Joint angle = 0 at initial configuration because child_xform
            cancels the initial body rotation. Use small values (|target| < 0.5)
            to avoid large torques that cause instability.

    Returns:
        Tuple of (support_body_idx, link_body_idx, joint_idx).
    """
    rot_unit = wp.quat_identity()
    half_len = link_length / 2.0
    link_half_height = half_len - link_radius

    # For prismatic joints, keep vertical orientation
    if joint_type == "prismatic":
        theta = 0.0
    else:
        theta = initial_angle

    # Rotation about Y axis
    rot = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), theta)

    # Create support body (fixed anchor, mass=0 makes it fixed)
    # Support extends downward so pendulum can swing into it
    support_height = 0.5  # Large enough for pendulum to contact
    support_pos = wp.vec3(pivot_position[0], pivot_position[1], pivot_position[2])
    body_support = builder.add_body(
        xform=wp.transform(support_pos, rot_unit),
        mass=0.0,  # Fixed body
        label=f"{body_key_prefix}support",
    )
    # Support shape with density=0 to not add mass, but with collision enabled
    support_cfg = newton.ModelBuilder.ShapeConfig()
    support_cfg.density = 0.0
    builder.add_shape_box(body_support, hx=1.0, hy=0.4, hz=0.05, cfg=support_cfg)

    # Compute link center position
    link_pos = wp.vec3(
        pivot_position[0] - half_len * math.sin(theta),
        pivot_position[1],
        pivot_position[2] - half_len * math.cos(theta),
    )

    # Compute artificial isotropic inertia for numerical stability
    # This keeps thin visual but spherical-like inertia for the solver
    link_mass = 1.0
    if inertia_scale > 1.0:
        # Use inertia_scale to directly scale a base spherical inertia
        equiv_radius = (link_radius + link_length) / 2.0
        base_I = 0.4 * link_mass * equiv_radius * equiv_radius
        spherical_I = base_I * inertia_scale  # Scale up for stability
        link_inertia = wp.mat33(
            spherical_I,
            0.0,
            0.0,
            0.0,
            spherical_I,
            0.0,
            0.0,
            0.0,
            spherical_I,
        )
    else:
        link_inertia = None

    # Create pendulum link with thin capsule
    body_link = builder.add_body(
        xform=wp.transform(link_pos, rot),
        mass=link_mass,
        inertia=link_inertia,
        label=f"{body_key_prefix}link",
    )

    # Create shape config with collision enabled so pendulum can contact support
    link_cfg = newton.ModelBuilder.ShapeConfig()
    link_cfg.has_shape_collision = True  # Enable collision with support
    link_cfg.density = 0.0  # Use explicit mass/inertia from add_body(), not shape
    builder.add_shape_capsule(body_link, radius=link_radius, half_height=link_half_height, cfg=link_cfg)

    # Joint frame offsets
    joint_parent_offset = wp.vec3(0.0, 0.0, -0.05)  # Bottom of support
    joint_child_offset = wp.vec3(0.0, 0.0, half_len)  # Top of link

    # Actuation gains: use defaults when enabled, zero when disabled
    ke = DEFAULT_KE if enable_actuation else 0.0
    kd = DEFAULT_KD if enable_actuation else 0.0

    # Create joint based on type
    # Note: ke=0, kd=0 effectively disables actuation without changing API calls
    joint = None
    if joint_type == "revolute":
        joint = builder.add_joint_revolute(
            parent=body_support,
            child=body_link,
            parent_xform=wp.transform(joint_parent_offset, rot_unit),
            child_xform=wp.transform(joint_child_offset, wp.quat_inverse(rot)),
            axis=(0.0, 1.0, 0.0),
            target_pos=target_pos,
            target_ke=ke,
            target_kd=kd,
        )

    elif joint_type == "spherical":
        # Ball joints don't have simple 1D actuation in this implementation
        joint = builder.add_joint_ball(
            parent=body_support,
            child=body_link,
            parent_xform=wp.transform(joint_parent_offset, rot_unit),
            child_xform=wp.transform(joint_child_offset, wp.quat_inverse(rot)),
        )

    elif joint_type == "prismatic":
        # For prismatic, the child frame should be identity since we start vertical
        joint = builder.add_joint_prismatic(
            parent=body_support,
            child=body_link,
            parent_xform=wp.transform(joint_parent_offset, rot_unit),
            child_xform=wp.transform(joint_child_offset, rot_unit),
            axis=(1.0, 0.0, 0.0),  # Horizontal sliding
            target_pos=target_pos,
            target_ke=ke,
            target_kd=kd,
        )

    elif joint_type == "cylindrical":
        # D6 joint with 1 linear DOF (X) + 1 angular DOF (Y)
        joint = builder.add_joint_d6(
            parent=body_support,
            child=body_link,
            parent_xform=wp.transform(joint_parent_offset, rot_unit),
            child_xform=wp.transform(joint_child_offset, wp.quat_inverse(rot)),
            linear_axes=[
                newton.ModelBuilder.JointDofConfig(
                    axis=(1.0, 0.0, 0.0),  # Translation along X
                    target_pos=target_pos,
                    target_ke=ke,
                    target_kd=kd,
                )
            ],
            angular_axes=[
                newton.ModelBuilder.JointDofConfig(
                    axis=(0.0, 1.0, 0.0),  # Rotation about Y
                    target_pos=target_pos,
                    target_ke=ke,
                    target_kd=kd,
                )
            ],
        )

    # Add articulation
    if joint is not None:
        builder.add_articulation([joint])

    return body_support, body_link, joint


def apply_initial_velocity(
    state: "newton.State",
    body_idx: int,
    lin_vel: wp.vec3 = wp.vec3(0.0, 0.0, 0.0),
    ang_vel: wp.vec3 = wp.vec3(0.0, 0.0, 0.0),
):
    """Apply initial velocity to a body for testing constraint enforcement."""
    body_qd = state.body_qd.numpy()
    body_qd[body_idx] = [lin_vel[0], lin_vel[1], lin_vel[2], ang_vel[0], ang_vel[1], ang_vel[2]]
    state.body_qd.assign(body_qd)


class Example:
    """Example demonstrating multiple pendulums with different joint types."""

    def __init__(self, viewer, args):
        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 2
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer

        # Parse joint types from args
        joint_types_str = getattr(args, "joints", None)
        if joint_types_str:
            self.joint_types = [j.strip() for j in joint_types_str.split(",")]
        else:
            self.joint_types = DEFAULT_JOINT_TYPES

        # Validate joint types
        for jt in self.joint_types:
            if jt not in JOINT_CONFIGS:
                raise ValueError(f"Unknown joint type: {jt}. Valid types: {list(JOINT_CONFIGS.keys())}")

        # Get actuation settings from args
        self.enable_actuation = getattr(args, "actuate", False)
        self.target_pos = getattr(args, "target_pos", 0.0)

        # Build model with multiple pendulums
        builder = newton.ModelBuilder(gravity=-10.0)
        builder.rigid_contact_margin = 0.05
        builder.default_shape_cfg.ke = 1e4
        builder.default_shape_cfg.kd = 100.0
        builder.default_shape_cfg.mu = 0.5

        # Add ground plane
        builder.add_ground_plane()

        # Create pendulums side by side
        # Use larger spacing to avoid collisions when prismatic/cylindrical joints translate
        self.pendulum_info: Dict[str, Dict] = {}
        spacing = 2.5  # Increased spacing for prismatic/cylindrical translation
        start_x = -spacing * (len(self.joint_types) - 1) / 2.0

        for i, joint_type in enumerate(self.joint_types):
            pivot_y = start_x + i * spacing
            # Offset prismatic/cylindrical joints in Y direction to avoid collisions
            pivot_x = 0.0
            pivot_pos = wp.vec3(pivot_x, pivot_y, 2.5)

            support_idx, link_idx, joint_idx = create_pendulum(
                builder,
                joint_type=joint_type,
                pivot_position=pivot_pos,
                link_length=1.0,
                link_radius=0.05,  # Thin capsule
                initial_angle=math.pi / 4,
                body_key_prefix=f"{joint_type}_",
                inertia_scale=10.0,  # Artificial inertia for stability
                enable_actuation=self.enable_actuation,
                target_pos=self.target_pos,
            )

            self.pendulum_info[joint_type] = {
                "support_idx": support_idx,
                "link_idx": link_idx,
                "joint_idx": joint_idx,
                "pivot_position": pivot_pos,
            }

        # Finalize model
        device = "cpu" if args is None else args.device
        self.model = builder.finalize(device=device)

        # Create DVI solver with position correction for stability
        # Joint velocity solver (with commented-out position correction)
        joint_config = newton.solvers.NumericalSolverConfig(
            solver_type=SolverType.SPARSE_LDL,
            max_iterations=100,
            alpha=0.001,
            recovery_speed=0.1,
            omega=0.3,
            relax=0.9,
            reg=0.001,
            # position_correction=newton.solvers.NumericalSolverConfig(
            #     solver_type=SolverType.SPARSE_LDL,
            #     max_iterations=10,
            #     omega=0.5,
            #     relax=0.9,
            #     reg=0.001,
            # ),
        )
        # Contact solver
        contact_config = newton.solvers.NumericalSolverConfig(
            solver_type=SolverType.SPARSE_JACOBI,
            max_iterations=100,
            alpha=0.002,
            recovery_speed=-1.0,
            omega=0.98,
            relax=0.98,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        # Apply initial velocities to all pendulum links (only when not actuating)
        # When actuating, we want a clean start without perturbations
        # if not self.enable_actuation:
        for joint_type, info in self.pendulum_info.items():
            apply_initial_velocity(
                self.state_0,
                info["link_idx"],
                lin_vel=wp.vec3(0.3, 0.3, 0.3),
                ang_vel=wp.vec3(0.5, 0.3, 0.3),
            )

        # Run collision first to populate model.rigid_contact_max before creating solver
        self.contacts = self.model.collide(self.state_0)

        self.solver = newton.solvers.SolverDVI(
            self.model,
            joint_solver=joint_config,
            contact_solver=contact_config,
            angular_damping=0.01,
            enable_actuation=self.enable_actuation,
            enable_contacts=True,
        )

        self.viewer.set_model(self.model)

        # Capture CUDA graph (disabled for debugging)
        self.graph = None

    def capture(self):
        """Capture simulation loop into a CUDA graph for optimal GPU performance."""
        self.graph = None
        if self.model.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        self.contacts = self.model.collide(self.state_0)
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)

            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument(
        "--joints",
        type=str,
        default=None,
        help="Comma-separated list of joint types to create (e.g., 'revolute,spherical'). "
        f"Available: {', '.join(JOINT_CONFIGS.keys())}. Default: all types.",
    )
    parser.add_argument(
        "--actuate",
        action="store_true",
        help=f"Enable PD control actuation (ke={DEFAULT_KE}, kd={DEFAULT_KD}).",
    )
    parser.add_argument(
        "--target-pos",
        type=float,
        default=0.0,
        help="Target position/angle for actuation (default: 0.0).",
    )

    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
