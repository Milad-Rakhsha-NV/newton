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
# Example Multiple Double Pendulums (DVI Solver)
#
# Creates multiple double pendulums side by side, each with a different joint type:
#   - revolute: Hinge joint (rotation about Y axis)
#   - spherical: Ball-and-socket joint (free rotation)
#   - prismatic: Slider joint (translation along X axis)
#   - cylindrical: Combined hinge + slider
#   - universal: Universal joint (SPHERICAL + DP1 constraint)
#
# Each double pendulum consists of:
#   - Support body (fixed anchor)
#   - Link 1 (first pendulum segment)
#   - Link 2 (second pendulum segment)
#   - End sphere (attached with fixed joint)
#
# The capsules are thin for visual clarity but have artificially increased
# inertia to avoid numerical instability with anisotropic mass matrices.
#
# Command: python -m newton.examples dvi_double_pendulum
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
        "num_constraints": 5,
        "allowed_motion": ["wy"],
    },
    "spherical": {
        "description": "Ball joint - rotation about all axes",
        "num_constraints": 3,
        "allowed_motion": ["wx", "wy", "wz"],
    },
    "prismatic": {
        "description": "Slider joint - translation along X axis only",
        "num_constraints": 5,
        "allowed_motion": ["vx"],
    },
    "cylindrical": {
        "description": "Cylindrical joint - translation along X + rotation about Y",
        "num_constraints": 4,
        "allowed_motion": ["vx", "wy"],
    },
    "universal": {
        "description": "Universal joint - rotation about two perpendicular axes (X and Z)",
        "num_constraints": 4,  # 3 position + 1 angular (lock one rotation)
        "allowed_motion": ["wx", "wz"],
    },
}

# Default joint types to create
DEFAULT_JOINT_TYPES = ["revolute", "spherical", "cylindrical", "prismatic"]
# DEFAULT_JOINT_TYPES = ["revolute"]

# Default actuation gains
# NOTE: Joint angle = 0 at initial configuration. Target positions are relative to this.
# With gravity, there will be a small steady-state error (expected for PD control).
# Higher gains reduce error but may cause oscillation.
DEFAULT_KE = 100.0  # Position gain (stiffness)
DEFAULT_KD = 30.0  # Velocity gain (damping)


def create_joint(
    builder: newton.ModelBuilder,
    joint_type: str,
    parent: int,
    child: int,
    parent_xform: wp.transform,
    child_xform: wp.transform,
    enable_actuation: bool = False,
    target_pos: float = 0.0,
) -> int:
    """
    Create a joint between two bodies.

    Args:
        builder: Newton model builder.
        joint_type: Type of joint.
        parent: Parent body index.
        child: Child body index.
        parent_xform: Joint frame in parent body coordinates.
        child_xform: Joint frame in child body coordinates.
        enable_actuation: Whether to enable PD control actuation.
        target_pos: Target position/angle for actuation.

    Returns:
        Joint index.
    """
    # Actuation gains: use defaults when enabled, zero when disabled
    ke = DEFAULT_KE if enable_actuation else 0.0
    kd = DEFAULT_KD if enable_actuation else 0.0

    if joint_type == "revolute":
        return builder.add_joint_revolute(
            parent=parent,
            child=child,
            parent_xform=parent_xform,
            child_xform=child_xform,
            axis=(0.0, 1.0, 0.0),  # Rotation about Y
            target_pos=target_pos,
            target_ke=ke,
            target_kd=kd,
        )

    elif joint_type == "spherical":
        # Ball joints don't have simple 1D actuation
        return builder.add_joint_ball(
            parent=parent,
            child=child,
            parent_xform=parent_xform,
            child_xform=child_xform,
        )

    elif joint_type == "prismatic":
        return builder.add_joint_prismatic(
            parent=parent,
            child=child,
            parent_xform=parent_xform,
            child_xform=child_xform,
            axis=(1.0, 0.0, 0.0),  # Translation along X
            target_pos=target_pos,
            target_ke=ke,
            target_kd=kd,
        )

    elif joint_type == "cylindrical":
        return builder.add_joint_d6(
            parent=parent,
            child=child,
            parent_xform=parent_xform,
            child_xform=child_xform,
            linear_axes=[
                newton.ModelBuilder.JointDofConfig(
                    axis=(1.0, 0.0, 0.0),
                    target_pos=target_pos,
                    target_ke=ke,
                    target_kd=kd,
                )
            ],
            angular_axes=[
                newton.ModelBuilder.JointDofConfig(
                    axis=(0.0, 1.0, 0.0),
                    target_pos=target_pos,
                    target_ke=ke,
                    target_kd=kd,
                )
            ],
        )

    elif joint_type == "universal":
        # Universal joint: SPHERICAL + lock one rotation axis
        # Uses D6 joint with 0 linear DOFs and 2 angular DOFs
        return builder.add_joint_d6(
            parent=parent,
            child=child,
            parent_xform=parent_xform,
            child_xform=child_xform,
            linear_axes=[],  # No translation
            angular_axes=[
                newton.ModelBuilder.JointDofConfig(
                    axis=(1.0, 0.0, 0.0),
                    target_pos=target_pos,
                    target_ke=ke,
                    target_kd=kd,
                ),  # Rotation about X
                newton.ModelBuilder.JointDofConfig(
                    axis=(0.0, 0.0, 1.0),
                    target_pos=target_pos,
                    target_ke=ke,
                    target_kd=kd,
                ),  # Rotation about Z
            ],
        )

    else:
        raise ValueError(f"Unknown joint type: {joint_type}")


def create_double_pendulum(
    builder: newton.ModelBuilder,
    joint_type: str,
    pivot_position: wp.vec3,
    link_length: float = 0.8,
    link_radius: float = 0.05,
    initial_angle_1: float = math.pi / 3,
    initial_angle_2: float = math.pi / 4,
    body_key_prefix: str = "",
    inertia_scale: float = 50.0,  # High inertia for stability with spherical joints
    enable_actuation: bool = False,
    target_pos: float = 0.0,
) -> Dict:
    """
    Create a double pendulum with the specified joint type.

    Args:
        builder: Newton model builder.
        joint_type: Type of joint for both connections.
        pivot_position: World position of the first pivot point.
        link_length: Length of each link.
        link_radius: Radius of the capsule.
        initial_angle_1: Initial angle of first link from vertical.
        initial_angle_2: Initial angle of second link from vertical.
        body_key_prefix: Prefix for body keys.
        inertia_scale: Factor to artificially increase inertia.
        enable_actuation: Whether to enable PD control actuation.
        target_pos: Target joint angle for actuation (relative to initial config).
            NOTE: Joint angle = 0 at initial configuration because child_xform
            cancels the initial body rotation. Use small values (|target| < 0.5)
            to avoid large torques that cause instability.

    Returns:
        Dictionary with body and joint indices.
    """
    rot_unit = wp.quat_identity()
    half_len = link_length / 2.0
    link_half_height = half_len - link_radius
    sphere_radius = 0.12

    # For prismatic joints, keep vertical orientation
    if joint_type == "prismatic":
        theta_1 = 0.0
        theta_2 = 0.0
    else:
        theta_1 = initial_angle_1
        theta_2 = initial_angle_2

    # Rotations about Y axis
    rot_1 = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), theta_1)
    rot_2 = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), theta_2)

    # Create support body (fixed anchor, mass=0 makes it fixed)
    # Support extends downward so pendulum can swing into it
    support_height = 0.6  # Large enough for pendulum to contact
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

    # Compute positions
    p1_offset = wp.vec3(-half_len * math.sin(theta_1), 0.0, -half_len * math.cos(theta_1))
    p1_end = wp.vec3(-link_length * math.sin(theta_1), 0.0, -link_length * math.cos(theta_1))
    p2_offset = wp.vec3(-half_len * math.sin(theta_2), 0.0, -half_len * math.cos(theta_2))
    p2_end = wp.vec3(-link_length * math.sin(theta_2), 0.0, -link_length * math.cos(theta_2))

    # Compute artificial isotropic inertia for numerical stability
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

    # Link shape config with collision enabled so pendulum can contact support
    link_cfg = newton.ModelBuilder.ShapeConfig()
    link_cfg.has_shape_collision = False  # Enable collision with support
    link_cfg.density = 0.0  # Use explicit mass/inertia from add_body(), not shape

    # Create link 1
    link1_pos = wp.vec3(
        pivot_position[0] + p1_offset[0],
        pivot_position[1] + p1_offset[1],
        pivot_position[2] + p1_offset[2],
    )
    body_link1 = builder.add_body(
        xform=wp.transform(link1_pos, rot_1),
        mass=link_mass,
        inertia=link_inertia,
        label=f"{body_key_prefix}link1",
    )
    builder.add_shape_capsule(body_link1, radius=link_radius, half_height=link_half_height, cfg=link_cfg)

    # Create link 2
    link2_pos = wp.vec3(
        pivot_position[0] + p1_end[0] + p2_offset[0],
        pivot_position[1] + p1_end[1] + p2_offset[1],
        pivot_position[2] + p1_end[2] + p2_offset[2],
    )
    body_link2 = builder.add_body(
        xform=wp.transform(link2_pos, rot_2),
        mass=link_mass,
        inertia=link_inertia,
        label=f"{body_key_prefix}link2",
    )
    builder.add_shape_capsule(body_link2, radius=link_radius, half_height=link_half_height, cfg=link_cfg)

    # Create end sphere with explicit inertia
    end_pos = wp.vec3(
        pivot_position[0] + p1_end[0] + p2_end[0],
        pivot_position[1] + p1_end[1] + p2_end[1],
        pivot_position[2] + p1_end[2] + p2_end[2],
    )
    # Sphere inertia: I = (2/5) * m * r^2
    sphere_I = 0.4 * link_mass * sphere_radius * sphere_radius * inertia_scale
    sphere_inertia = wp.mat33(
        sphere_I,
        0.0,
        0.0,
        0.0,
        sphere_I,
        0.0,
        0.0,
        0.0,
        sphere_I,
    )
    body_end = builder.add_body(
        xform=wp.transform(end_pos, rot_unit),
        mass=link_mass,
        inertia=sphere_inertia,
        label=f"{body_key_prefix}end_sphere",
    )
    # Sphere shape config with collision enabled
    sphere_cfg = newton.ModelBuilder.ShapeConfig()
    sphere_cfg.density = 0.0
    sphere_cfg.has_shape_collision = True  # Enable collision with support
    builder.add_shape_sphere(body_end, radius=sphere_radius, cfg=sphere_cfg)

    # Actuation gains: use defaults when enabled, zero when disabled
    ke = DEFAULT_KE if enable_actuation else 0.0
    kd = DEFAULT_KD if enable_actuation else 0.0

    # Create joints
    joints = []

    # Joint 1: support -> link1
    # Offset from support center to bottom of support (where joint is)
    joint1_parent_offset = wp.vec3(0.0, 0.0, -0.05)
    joint1_child_offset = wp.vec3(0.0, 0.0, half_len)
    joint1 = create_joint(
        builder,
        joint_type,
        body_support,
        body_link1,
        parent_xform=wp.transform(joint1_parent_offset, rot_unit),
        child_xform=wp.transform(joint1_child_offset, wp.quat_inverse(rot_1)),
        enable_actuation=enable_actuation,
        target_pos=target_pos,
    )
    joints.append(joint1)

    # Joint 2: link1 -> link2 (always revolute for stability)
    joint2_parent_offset = wp.vec3(0.0, 0.0, -half_len)
    joint2_child_offset = wp.vec3(0.0, 0.0, half_len)
    joint2 = builder.add_joint_revolute(
        parent=body_link1,
        child=body_link2,
        parent_xform=wp.transform(joint2_parent_offset, wp.quat_inverse(rot_1)),
        child_xform=wp.transform(joint2_child_offset, wp.quat_inverse(rot_2)),
        axis=(0.0, 1.0, 0.0),  # Rotation about Y
        target_pos=target_pos,
        target_ke=ke,
        target_kd=kd,
    )
    joints.append(joint2)

    # Fixed joint: link2 -> end sphere
    joint_fixed = builder.add_joint_fixed(
        parent=body_link2,
        child=body_end,
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, -half_len), wp.quat_inverse(rot_2)),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), rot_unit),
    )
    joints.append(joint_fixed)

    # Add articulation
    builder.add_articulation(joints)

    return {
        "support_idx": body_support,
        "link1_idx": body_link1,
        "link2_idx": body_link2,
        "end_idx": body_end,
        "joints": joints,
        "pivot_position": pivot_position,
    }


def apply_initial_velocities(
    state: "newton.State",
    body_indices: List[int],
    lin_vel: wp.vec3 = wp.vec3(0.2, 0.2, 0.2),
    ang_vel: wp.vec3 = wp.vec3(0.3, 1.0, 0.3),
):
    """Apply initial velocities to multiple bodies."""
    body_qd = state.body_qd.numpy()
    for idx in body_indices:
        body_qd[idx] = [lin_vel[0], lin_vel[1], lin_vel[2], ang_vel[0], ang_vel[1], ang_vel[2]]
    state.body_qd.assign(body_qd)


class Example:
    """Example demonstrating multiple double pendulums with different joint types."""

    def __init__(self, viewer, args):
        self.fps = 200
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

        # Build model
        builder = newton.ModelBuilder(gravity=-10.0)
        builder.rigid_contact_margin = 0.05
        builder.default_shape_cfg.ke = 1e4
        builder.default_shape_cfg.kd = 100.0
        builder.default_shape_cfg.mu = 0.5

        # Add ground plane
        builder.add_ground_plane()

        # Create double pendulums side by side
        # Use larger spacing to avoid collisions when prismatic/cylindrical joints translate
        self.pendulum_info: Dict[str, Dict] = {}
        spacing = 3.0  # Increased from 2.0 for prismatic/cylindrical translation
        start_x = -spacing * (len(self.joint_types) - 1) / 2.0

        # Position prismatic/cylindrical joints with Y offset to avoid collisions
        for i, joint_type in enumerate(self.joint_types):
            pivot_x = start_x + i * spacing
            # Offset prismatic/cylindrical joints in Y direction
            pivot_y = 0.0
            pivot_pos = wp.vec3(pivot_x, pivot_y, 2.5)

            info = create_double_pendulum(
                builder,
                joint_type=joint_type,
                pivot_position=pivot_pos,
                link_length=0.8,
                link_radius=0.05,
                initial_angle_1=math.pi / 3,
                initial_angle_2=math.pi / 4,
                body_key_prefix=f"{joint_type}_",
                inertia_scale=10.0,  # Artificial inertia for stability
                enable_actuation=self.enable_actuation,
                target_pos=self.target_pos,
            )
            self.pendulum_info[joint_type] = info

        # Finalize model
        device = "cpu" if args is None else args.device
        self.model = builder.finalize(device=device)

        # Get solver type from args
        solver_type_str = getattr(args, "solver", "sparse_ldl")

        # Map CLI solver name to SolverType enum
        solver_type_map = {
            "sparse_ldl": SolverType.SPARSE_LDL,
            "sparse_jacobi": SolverType.SPARSE_JACOBI,
        }
        solver_type_enum = solver_type_map[solver_type_str]

        # Create DVI solver with position correction for stability
        if solver_type_str == "sparse_ldl":
            joint_config = newton.solvers.NumericalSolverConfig(
                solver_type=solver_type_enum,
                alpha=0.001,  # Disable Baumgarte (using position correction)
                reg=1e-6,
                recovery_speed=0.1,
                # position_correction=newton.solvers.NumericalSolverConfig(
                #     solver_type=solver_type_enum,
                #     reg=1e-6,
                # ),
            )
        else:
            # sparse_jacobi (iterative)
            joint_config = newton.solvers.NumericalSolverConfig(
                solver_type=solver_type_enum,
                max_iterations=10,
                alpha=1e6,
                omega=0.9,
                relax=0.9,
                reg=0.001,
                # position_correction=newton.solvers.NumericalSolverConfig(
                #     solver_type=solver_type_enum,
                #     max_iterations=10,
                #     omega=0.5,
                #     relax=0.9,
                #     reg=0.001,
                # ),
            )

        # Contact solver (always iterative)
        contact_config = newton.solvers.NumericalSolverConfig(
            solver_type=SolverType.SPARSE_JACOBI,
            max_iterations=10,
            alpha=0.002,
            recovery_speed=-1.0,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        # Apply initial velocities (only when not actuating)
        # When actuating, we want a clean start without perturbations
        if not self.enable_actuation:
            for joint_type, info in self.pendulum_info.items():
                apply_initial_velocities(
                    self.state_0,
                    [info["link1_idx"], info["link2_idx"]],
                    lin_vel=wp.vec3(0.2, 0.2, 0.2),
                    ang_vel=wp.vec3(0.3, 1.0, 0.3),
                )

        # Run collision first to populate model.rigid_contact_max before creating solver
        self.contacts = self.model.collide(self.state_0)

        self.solver = newton.solvers.SolverDVI(
            self.model,
            joint_solver=joint_config,
            contact_solver=contact_config,
            angular_damping=0.0,
            enable_actuation=self.enable_actuation,
            enable_contacts=True,
        )

        self.viewer.set_model(self.model)

        # CUDA graph (disabled for debugging)
        self.graph = None

    def capture(self):
        """Capture simulation loop into a CUDA graph."""
        self.graph = None
        if self.model.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)

            self.contacts = self.model.collide(self.state_0)
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
        help="Comma-separated list of joint types (e.g., 'revolute,spherical'). "
        f"Available: {', '.join(JOINT_CONFIGS.keys())}. Default: {', '.join(DEFAULT_JOINT_TYPES)}.",
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
    parser.add_argument(
        "--solver",
        type=str,
        default="sparse_ldl",
        choices=["sparse_ldl", "sparse_jacobi"],
        help="Joint solver: 'sparse_ldl' (direct) or 'sparse_jacobi' (iterative).",
    )

    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
