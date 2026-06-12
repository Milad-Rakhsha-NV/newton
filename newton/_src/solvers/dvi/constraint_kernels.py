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
Joint constraint kernels for the DVI solver.

This module implements kernels for bilateral joint constraints in maximal
coordinates. Joint constraints maintain kinematic relationships between
bodies.

Mathematical Formulation
------------------------

Joint constraints are formulated as:

    phi(q) = 0

where phi is the constraint function and q is the configuration.

At the velocity level:

    J * v = 0

where J = d(phi)/dq is the constraint Jacobian.

The constraint force (impulse) is:

    F_c = J^T * lambda

where lambda are the Lagrange multipliers (constraint impulses).

Constraint Stabilization (Baumgarte):

To prevent drift, we add position correction:

    J * v = -phi / (dt + alpha)  (alpha-damped, DVI-style)

This adds velocity correction proportional to position error.
The alpha parameter provides damping to prevent energy injection.

System to Solve:

The constraint impulses satisfy:

    N * lambda = b

where:
    - N = J * M_inv * J^T  (constraint-space system matrix)
    - b = -(J * v_pred + phi / (dt + alpha))  (right-hand side)
    - v_pred = v + dt * M_inv * f_ext  (predicted velocity)

Joint Types
-----------

- FIXED (type 3): 6 constraints (3 position + 3 orientation)
  Maintains constant relative pose between bodies.

- REVOLUTE (type 1): 5 constraints (3 position + 2 orientation)
  Allows rotation about one axis (hinge).

- PRISMATIC (type 0): 5 constraints (2 position + 3 orientation)
  Allows translation along one axis (slider).

- BALL (type 2): 3 constraints (3 position only)
  Allows free rotation (ball-and-socket).

- CYLINDRICAL (D6 with 1 lin + 1 ang): 4 constraints

- UNIVERSAL (D6 with 0 lin + 2 ang): 4 constraints

Jacobian Structure
------------------

For maximal coordinates, the Jacobian maps body velocities to constraint velocity:

    d(phi)/dt = J_a * V_a + J_b * V_b

For a position constraint phi = x_c - x_p (child - parent):
    J_a = [-I, -r_a x ...]  (parent contribution)
    J_b = [+I, +r_b x ...]  (child contribution)

For an orientation constraint along direction e:
    J_a = [0, -e]  (parent)
    J_b = [0, +e]  (child)
"""

import warp as wp

# =============================================================================
# Primitive Constraint Helper Functions
# =============================================================================


@wp.func
def fill_spherical_constraint(
    jac_row: int,
    c_idx: int,
    r_a: wp.vec3,
    r_b: wp.vec3,
    x_err: wp.vec3,
    joint_jac: wp.array2d[float],
    joint_violation: wp.array[float],
) -> int:
    """
    Fill SPHERICAL constraint (3 position constraints).

    Constrains the joint frames to coincide in position.

    Mathematical formulation:
        c = p_child - p_parent = 0

    Jacobian for each direction e (x, y, z):
        J_a = [-e, -r_a × e]
        J_b = [+e, +r_b × e]

    Args:
        jac_row: Starting row in Jacobian array for this joint.
        c_idx: Current constraint index within this joint.
        r_a: Lever arm from parent COM to joint frame.
        r_b: Lever arm from child COM to joint frame.
        x_err: Position error (child - parent).
        joint_jac: Jacobian array (output).
        joint_violation: Violation array (output).

    Returns:
        Updated constraint index (c_idx + 3).
    """
    for i in range(3):
        e = wp.vec3(0.0, 0.0, 0.0)
        if i == 0:
            e = wp.vec3(1.0, 0.0, 0.0)
        elif i == 1:
            e = wp.vec3(0.0, 1.0, 0.0)
        else:
            e = wp.vec3(0.0, 0.0, 1.0)

        # J_a (parent): -[e, r_a × e]
        joint_jac[jac_row + c_idx, 0] = -e[0]
        joint_jac[jac_row + c_idx, 1] = -e[1]
        joint_jac[jac_row + c_idx, 2] = -e[2]
        cross_a = wp.cross(r_a, e)
        joint_jac[jac_row + c_idx, 3] = -cross_a[0]
        joint_jac[jac_row + c_idx, 4] = -cross_a[1]
        joint_jac[jac_row + c_idx, 5] = -cross_a[2]

        # J_b (child): +[e, r_b × e]
        joint_jac[jac_row + c_idx, 6] = e[0]
        joint_jac[jac_row + c_idx, 7] = e[1]
        joint_jac[jac_row + c_idx, 8] = e[2]
        cross_b = wp.cross(r_b, e)
        joint_jac[jac_row + c_idx, 9] = cross_b[0]
        joint_jac[jac_row + c_idx, 10] = cross_b[1]
        joint_jac[jac_row + c_idx, 11] = cross_b[2]

        joint_violation[jac_row + c_idx] = x_err[i]
        c_idx = c_idx + 1

    return c_idx


@wp.func
def fill_parallel_plane_constraint(
    jac_row: int,
    c_idx: int,
    axis_b: wp.vec3,
    axis_a_1: wp.vec3,
    axis_a_2: wp.vec3,
    joint_jac: wp.array2d[float],
    joint_violation: wp.array[float],
) -> int:
    """
    Fill PARALLEL_PLANE (DP1) constraint (2 orientation constraints).

    Constrains an axis from body B to be perpendicular to two axes in body A.
    This is used to lock rotation about two axes.

    Mathematical formulation:
        c_1 = axis_b · axis_a_1 = 0
        c_2 = axis_b · axis_a_2 = 0

    The violation is the dot product (should be 0 when axes are perpendicular).

    Jacobian (angular only):
        J_ang_a = -(axis_b × axis_a_i)
        J_ang_b = -(axis_a_i × axis_b)

    Note: The cross product gives the direction of angular velocity that
    would cause the violation to change.

    Args:
        jac_row: Starting row in Jacobian array for this joint.
        c_idx: Current constraint index within this joint.
        axis_b: Axis from body B (in world frame).
        axis_a_1: First perpendicular axis from body A (in world frame).
        axis_a_2: Second perpendicular axis from body A (in world frame).
        joint_jac: Jacobian array (output).
        joint_violation: Violation array (output).

    Returns:
        Updated constraint index (c_idx + 2).
    """
    # Constraint 1: axis_b · axis_a_1 = 0
    cross_1 = wp.cross(axis_b, axis_a_1)
    joint_jac[jac_row + c_idx, 0] = 0.0
    joint_jac[jac_row + c_idx, 1] = 0.0
    joint_jac[jac_row + c_idx, 2] = 0.0
    joint_jac[jac_row + c_idx, 3] = -cross_1[0]
    joint_jac[jac_row + c_idx, 4] = -cross_1[1]
    joint_jac[jac_row + c_idx, 5] = -cross_1[2]
    joint_jac[jac_row + c_idx, 6] = 0.0
    joint_jac[jac_row + c_idx, 7] = 0.0
    joint_jac[jac_row + c_idx, 8] = 0.0
    cross_1_neg = wp.cross(axis_a_1, axis_b)
    joint_jac[jac_row + c_idx, 9] = -cross_1_neg[0]
    joint_jac[jac_row + c_idx, 10] = -cross_1_neg[1]
    joint_jac[jac_row + c_idx, 11] = -cross_1_neg[2]
    joint_violation[jac_row + c_idx] = wp.dot(axis_b, axis_a_1)
    c_idx = c_idx + 1

    # Constraint 2: axis_b · axis_a_2 = 0
    cross_2 = wp.cross(axis_b, axis_a_2)
    joint_jac[jac_row + c_idx, 0] = 0.0
    joint_jac[jac_row + c_idx, 1] = 0.0
    joint_jac[jac_row + c_idx, 2] = 0.0
    joint_jac[jac_row + c_idx, 3] = -cross_2[0]
    joint_jac[jac_row + c_idx, 4] = -cross_2[1]
    joint_jac[jac_row + c_idx, 5] = -cross_2[2]
    joint_jac[jac_row + c_idx, 6] = 0.0
    joint_jac[jac_row + c_idx, 7] = 0.0
    joint_jac[jac_row + c_idx, 8] = 0.0
    cross_2_neg = wp.cross(axis_a_2, axis_b)
    joint_jac[jac_row + c_idx, 9] = -cross_2_neg[0]
    joint_jac[jac_row + c_idx, 10] = -cross_2_neg[1]
    joint_jac[jac_row + c_idx, 11] = -cross_2_neg[2]
    joint_violation[jac_row + c_idx] = wp.dot(axis_b, axis_a_2)
    c_idx = c_idx + 1

    return c_idx


@wp.func
def fill_ball_on_axis_constraint(
    jac_row: int,
    c_idx: int,
    axis_a_1: wp.vec3,
    axis_a_2: wp.vec3,
    dp: wp.vec3,
    r_a: wp.vec3,
    r_b: wp.vec3,
    joint_jac: wp.array2d[float],
    joint_violation: wp.array[float],
) -> int:
    """
    Fill BALL_ON_AXIS (DP2) constraint (2 position constraints).

    Constrains the displacement vector between joint frames to be
    perpendicular to two axes in body A. This allows motion only along
    a specified axis.

    Mathematical formulation:
        c_1 = dp · axis_a_1 = 0
        c_2 = dp · axis_a_2 = 0

    where dp = p_child - p_parent is the displacement vector.

    Jacobian:
        J_lin_a = -axis_a_i
        J_ang_a = -axis_a_i × r_a + dp × axis_a_i (coupling with orientation)
        J_lin_b = +axis_a_i
        J_ang_b = +axis_a_i × r_b

    The angular part for body A includes coupling because rotating body A
    changes both the position of the joint frame AND the direction of the
    constraint axis.

    Args:
        jac_row: Starting row in Jacobian array for this joint.
        c_idx: Current constraint index within this joint.
        axis_a_1: First perpendicular axis from body A (in world frame).
        axis_a_2: Second perpendicular axis from body A (in world frame).
        dp: Displacement vector from parent to child joint frame.
        r_a: Lever arm from parent COM to joint frame.
        r_b: Lever arm from child COM to joint frame.
        joint_jac: Jacobian array (output).
        joint_violation: Violation array (output).

    Returns:
        Updated constraint index (c_idx + 2).
    """
    # Normalize dp for violation computation (avoid division by zero)
    dp_len = wp.length(dp)
    dp_dir = dp
    if dp_len > 1e-8:
        dp_dir = dp / dp_len

    # Constraint 1: dp · axis_a_1 = 0
    joint_jac[jac_row + c_idx, 0] = -axis_a_1[0]
    joint_jac[jac_row + c_idx, 1] = -axis_a_1[1]
    joint_jac[jac_row + c_idx, 2] = -axis_a_1[2]
    # Angular part for A: -axis_a_1 × r_a + dp × axis_a_1
    ang_a_1 = -wp.cross(axis_a_1, r_a) + wp.cross(dp, axis_a_1)
    joint_jac[jac_row + c_idx, 3] = ang_a_1[0]
    joint_jac[jac_row + c_idx, 4] = ang_a_1[1]
    joint_jac[jac_row + c_idx, 5] = ang_a_1[2]
    joint_jac[jac_row + c_idx, 6] = axis_a_1[0]
    joint_jac[jac_row + c_idx, 7] = axis_a_1[1]
    joint_jac[jac_row + c_idx, 8] = axis_a_1[2]
    ang_b_1 = wp.cross(axis_a_1, r_b)
    joint_jac[jac_row + c_idx, 9] = ang_b_1[0]
    joint_jac[jac_row + c_idx, 10] = ang_b_1[1]
    joint_jac[jac_row + c_idx, 11] = ang_b_1[2]
    joint_violation[jac_row + c_idx] = wp.dot(dp_dir, axis_a_1)
    c_idx = c_idx + 1

    # Constraint 2: dp · axis_a_2 = 0
    joint_jac[jac_row + c_idx, 0] = -axis_a_2[0]
    joint_jac[jac_row + c_idx, 1] = -axis_a_2[1]
    joint_jac[jac_row + c_idx, 2] = -axis_a_2[2]
    # Angular part for A: -axis_a_2 × r_a + dp × axis_a_2
    ang_a_2 = -wp.cross(axis_a_2, r_a) + wp.cross(dp, axis_a_2)
    joint_jac[jac_row + c_idx, 3] = ang_a_2[0]
    joint_jac[jac_row + c_idx, 4] = ang_a_2[1]
    joint_jac[jac_row + c_idx, 5] = ang_a_2[2]
    joint_jac[jac_row + c_idx, 6] = axis_a_2[0]
    joint_jac[jac_row + c_idx, 7] = axis_a_2[1]
    joint_jac[jac_row + c_idx, 8] = axis_a_2[2]
    ang_b_2 = wp.cross(axis_a_2, r_b)
    joint_jac[jac_row + c_idx, 9] = ang_b_2[0]
    joint_jac[jac_row + c_idx, 10] = ang_b_2[1]
    joint_jac[jac_row + c_idx, 11] = ang_b_2[2]
    joint_violation[jac_row + c_idx] = wp.dot(dp_dir, axis_a_2)
    c_idx = c_idx + 1

    return c_idx


@wp.func
def fill_perpendicular_rotation_constraint(
    jac_row: int,
    c_idx: int,
    axis_1: wp.vec3,
    axis_2: wp.vec3,
    swing_err: wp.vec3,
    joint_jac: wp.array2d[float],
    joint_violation: wp.array[float],
) -> int:
    """
    Fill rotation constraints perpendicular to an axis (2 constraints).

    Constrains rotation to be about a single axis by locking rotation
    in the two perpendicular directions.

    Mathematical formulation:
        c_1 = swing_err · axis_1 = 0
        c_2 = swing_err · axis_2 = 0

    where swing_err = axis_p × axis_c is the orientation error vector.

    Jacobian (angular only):
        J_ang_a = -axis_i
        J_ang_b = +axis_i
        (linear components are zero)

    Args:
        jac_row: Starting row in Jacobian array for this joint.
        c_idx: Current constraint index within this joint.
        axis_1: First perpendicular axis (in world frame).
        axis_2: Second perpendicular axis (in world frame).
        swing_err: Orientation error (swing) vector.
        joint_jac: Jacobian array (output).
        joint_violation: Violation array (output).

    Returns:
        Updated constraint index (c_idx + 2).
    """
    # Constraint 1: swing_err · axis_1 = 0
    joint_jac[jac_row + c_idx, 0] = 0.0
    joint_jac[jac_row + c_idx, 1] = 0.0
    joint_jac[jac_row + c_idx, 2] = 0.0
    joint_jac[jac_row + c_idx, 3] = -axis_1[0]
    joint_jac[jac_row + c_idx, 4] = -axis_1[1]
    joint_jac[jac_row + c_idx, 5] = -axis_1[2]
    joint_jac[jac_row + c_idx, 6] = 0.0
    joint_jac[jac_row + c_idx, 7] = 0.0
    joint_jac[jac_row + c_idx, 8] = 0.0
    joint_jac[jac_row + c_idx, 9] = axis_1[0]
    joint_jac[jac_row + c_idx, 10] = axis_1[1]
    joint_jac[jac_row + c_idx, 11] = axis_1[2]
    joint_violation[jac_row + c_idx] = wp.dot(swing_err, axis_1)
    c_idx = c_idx + 1

    # Constraint 2: swing_err · axis_2 = 0
    joint_jac[jac_row + c_idx, 0] = 0.0
    joint_jac[jac_row + c_idx, 1] = 0.0
    joint_jac[jac_row + c_idx, 2] = 0.0
    joint_jac[jac_row + c_idx, 3] = -axis_2[0]
    joint_jac[jac_row + c_idx, 4] = -axis_2[1]
    joint_jac[jac_row + c_idx, 5] = -axis_2[2]
    joint_jac[jac_row + c_idx, 6] = 0.0
    joint_jac[jac_row + c_idx, 7] = 0.0
    joint_jac[jac_row + c_idx, 8] = 0.0
    joint_jac[jac_row + c_idx, 9] = axis_2[0]
    joint_jac[jac_row + c_idx, 10] = axis_2[1]
    joint_jac[jac_row + c_idx, 11] = axis_2[2]
    joint_violation[jac_row + c_idx] = wp.dot(swing_err, axis_2)
    c_idx = c_idx + 1

    return c_idx


@wp.func
def fill_perpendicular_position_constraint(
    jac_row: int,
    c_idx: int,
    axis_1: wp.vec3,
    axis_2: wp.vec3,
    x_err: wp.vec3,
    r_a: wp.vec3,
    r_b: wp.vec3,
    joint_jac: wp.array2d[float],
    joint_violation: wp.array[float],
) -> int:
    """
    Fill position constraints perpendicular to an axis (2 constraints).

    Constrains the position error to be zero in the plane perpendicular to
    a given axis. This allows linear motion only along that axis.

    Mathematical formulation:
        c_1 = x_err · axis_1 = 0
        c_2 = x_err · axis_2 = 0

    where axis_1 and axis_2 are perpendicular to the sliding axis.

    Jacobian:
        J_lin_a = -axis_i
        J_ang_a = -axis_i × r_a
        J_lin_b = +axis_i
        J_ang_b = +axis_i × r_b

    Args:
        jac_row: Starting row in Jacobian array for this joint.
        c_idx: Current constraint index within this joint.
        axis_1: First perpendicular axis (in world frame).
        axis_2: Second perpendicular axis (in world frame).
        x_err: Position error vector.
        r_a: Lever arm from parent COM to joint frame.
        r_b: Lever arm from child COM to joint frame.
        joint_jac: Jacobian array (output).
        joint_violation: Violation array (output).

    Returns:
        Updated constraint index (c_idx + 2).
    """
    # Constraint 1: x_err · axis_1 = 0
    joint_jac[jac_row + c_idx, 0] = -axis_1[0]
    joint_jac[jac_row + c_idx, 1] = -axis_1[1]
    joint_jac[jac_row + c_idx, 2] = -axis_1[2]
    cross_a = wp.cross(r_a, axis_1)
    joint_jac[jac_row + c_idx, 3] = -cross_a[0]
    joint_jac[jac_row + c_idx, 4] = -cross_a[1]
    joint_jac[jac_row + c_idx, 5] = -cross_a[2]
    joint_jac[jac_row + c_idx, 6] = axis_1[0]
    joint_jac[jac_row + c_idx, 7] = axis_1[1]
    joint_jac[jac_row + c_idx, 8] = axis_1[2]
    cross_b = wp.cross(r_b, axis_1)
    joint_jac[jac_row + c_idx, 9] = cross_b[0]
    joint_jac[jac_row + c_idx, 10] = cross_b[1]
    joint_jac[jac_row + c_idx, 11] = cross_b[2]
    joint_violation[jac_row + c_idx] = wp.dot(x_err, axis_1)
    c_idx = c_idx + 1

    # Constraint 2: x_err · axis_2 = 0
    joint_jac[jac_row + c_idx, 0] = -axis_2[0]
    joint_jac[jac_row + c_idx, 1] = -axis_2[1]
    joint_jac[jac_row + c_idx, 2] = -axis_2[2]
    cross_a = wp.cross(r_a, axis_2)
    joint_jac[jac_row + c_idx, 3] = -cross_a[0]
    joint_jac[jac_row + c_idx, 4] = -cross_a[1]
    joint_jac[jac_row + c_idx, 5] = -cross_a[2]
    joint_jac[jac_row + c_idx, 6] = axis_2[0]
    joint_jac[jac_row + c_idx, 7] = axis_2[1]
    joint_jac[jac_row + c_idx, 8] = axis_2[2]
    cross_b = wp.cross(r_b, axis_2)
    joint_jac[jac_row + c_idx, 9] = cross_b[0]
    joint_jac[jac_row + c_idx, 10] = cross_b[1]
    joint_jac[jac_row + c_idx, 11] = cross_b[2]
    joint_violation[jac_row + c_idx] = wp.dot(x_err, axis_2)
    c_idx = c_idx + 1

    return c_idx


@wp.func
def fill_locked_orientation_constraint(
    jac_row: int,
    c_idx: int,
    ang_err: wp.vec3,
    joint_jac: wp.array2d[float],
    joint_violation: wp.array[float],
) -> int:
    """
    Fill LOCKED_ORIENTATION constraint (3 orientation constraints).

    Constrains all rotational DOFs, locking the relative orientation
    between parent and child bodies.

    Mathematical formulation:
        c_i = (ω_child - ω_parent) · e_i = 0  for i in {x, y, z}

    Jacobian for each direction e (x, y, z):
        J_ang_a = -e
        J_ang_b = +e
        (linear components are zero)

    Args:
        jac_row: Starting row in Jacobian array for this joint.
        c_idx: Current constraint index within this joint.
        ang_err: Angular error (axis-angle representation).
        joint_jac: Jacobian array (output).
        joint_violation: Violation array (output).

    Returns:
        Updated constraint index (c_idx + 3).
    """
    for i in range(3):
        e = wp.vec3(0.0, 0.0, 0.0)
        if i == 0:
            e = wp.vec3(1.0, 0.0, 0.0)
        elif i == 1:
            e = wp.vec3(0.0, 1.0, 0.0)
        else:
            e = wp.vec3(0.0, 0.0, 1.0)

        # Angular Jacobian: J_a = [0, -e], J_b = [0, +e]
        joint_jac[jac_row + c_idx, 0] = 0.0
        joint_jac[jac_row + c_idx, 1] = 0.0
        joint_jac[jac_row + c_idx, 2] = 0.0
        joint_jac[jac_row + c_idx, 3] = -e[0]
        joint_jac[jac_row + c_idx, 4] = -e[1]
        joint_jac[jac_row + c_idx, 5] = -e[2]

        joint_jac[jac_row + c_idx, 6] = 0.0
        joint_jac[jac_row + c_idx, 7] = 0.0
        joint_jac[jac_row + c_idx, 8] = 0.0
        joint_jac[jac_row + c_idx, 9] = e[0]
        joint_jac[jac_row + c_idx, 10] = e[1]
        joint_jac[jac_row + c_idx, 11] = e[2]

        joint_violation[jac_row + c_idx] = ang_err[i]
        c_idx = c_idx + 1

    return c_idx


@wp.func
def get_orthogonal_vectors(axis: wp.vec3) -> wp.vec3:
    """
    Get a vector orthogonal to the given axis.

    Args:
        axis: Input axis (should be normalized).

    Returns:
        A normalized vector perpendicular to axis.
    """
    # Find the component with smallest absolute value
    n_abs = wp.vec3(wp.abs(axis[0]), wp.abs(axis[1]), wp.abs(axis[2]))
    min_i = 0
    if n_abs[1] < n_abs[0]:
        min_i = 1
    if n_abs[2] < n_abs[0] and n_abs[2] < n_abs[1]:
        min_i = 2

    e_min = wp.vec3(0.0, 0.0, 0.0)
    if min_i == 0:
        e_min = wp.vec3(1.0, 0.0, 0.0)
    elif min_i == 1:
        e_min = wp.vec3(0.0, 1.0, 0.0)
    else:
        e_min = wp.vec3(0.0, 0.0, 1.0)

    return wp.normalize(wp.cross(axis, e_min))


# =============================================================================
# Joint Jacobian and Violation Computation
# =============================================================================


@wp.kernel
def compute_joint_jacobians_and_violation(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    joint_type: wp.array[int],
    joint_enabled: wp.array[bool],
    joint_parent: wp.array[int],
    joint_child: wp.array[int],
    joint_X_p: wp.array[wp.transform],
    joint_X_c: wp.array[wp.transform],
    joint_axis: wp.array[wp.vec3],
    joint_q_start: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_dof_dim: wp.array2d[int],  # (n_joints, 2): (n_linear, n_angular) per joint
    nc_offset: wp.array[wp.int32],  # Prefix-sum offset for each joint into compact arrays
    # outputs - Jacobian stored compactly with nc_offset indexing
    joint_jac: wp.array2d[float],  # shape: (total_nc, 12)
    joint_violation: wp.array[float],  # shape: (total_nc,)
    joint_body_a: wp.array[int],
    joint_body_b: wp.array[int],
):
    """
    Compute joint constraint Jacobians and violations.

    For each joint type, computes:
        1. Constraint violation c (position/orientation error)
        2. Constraint Jacobian J (maps velocities to constraint velocities)

    **Jacobian Layout:**

    For joint j with k constraints, the Jacobian is stored at:
        joint_jac[nc_offset[j] + c, 0:6]  = J_a (parent body Jacobian row)
        joint_jac[nc_offset[j] + c, 6:12] = J_b (child body Jacobian row)

    where c ∈ [0, k-1] is the constraint index within the joint.

    **Constraint Velocity:**

    The constraint velocity for constraint c is:
        ċ_c = J_a[c] · V_a + J_b[c] · V_b

    where V = [v_lin, v_ang]^T.

    **Position Constraint Jacobian:**

    For a constraint c = (x_child - x_parent) · e:
        J_a_lin = -e
        J_a_ang = -r_a × e
        J_b_lin = +e
        J_b_ang = +r_b × e

    **Orientation Constraint Jacobian:**

    For a constraint on angular velocity component:
        J_a_lin = 0
        J_a_ang = -e
        J_b_lin = 0
        J_b_ang = +e

    Args:
        body_q: Body transforms [position, quaternion].
        body_qd: Body spatial velocities.
        body_com: Body center of mass (local coordinates).
        joint_type: Joint type (0=prismatic, 1=revolute, 2=ball, 3=fixed).
        joint_enabled: Whether joint is active.
        joint_parent: Parent body index (-1 for world).
        joint_child: Child body index.
        joint_X_p: Joint frame in parent body coordinates.
        joint_X_c: Joint frame in child body coordinates.
        joint_axis: Joint axis (for revolute/prismatic).
        joint_q_start, joint_qd_start: Index offsets.
        nc_offset: Prefix-sum offset array for compact storage.

    Outputs:
        joint_jac: Constraint Jacobians (total_nc, 12) with compact storage.
        joint_violation: Constraint violations (total_nc,) with compact storage.
        joint_body_a, joint_body_b: Body indices.
    """
    tid = wp.tid()

    if not joint_enabled[tid]:
        return

    jtype = joint_type[tid]
    c_child = joint_child[tid]
    c_parent = joint_parent[tid]

    X_pj = joint_X_p[tid]
    X_cj = joint_X_c[tid]

    # Store body indices
    joint_body_a[tid] = c_parent
    joint_body_b[tid] = c_child

    # Parent joint frame in world coordinates
    X_wp = X_pj
    if c_parent >= 0:
        X_wp = body_q[c_parent] * X_wp

    # Child joint frame in world coordinates
    X_wc = body_q[c_child] * X_cj

    x_p = wp.transform_get_translation(X_wp)
    x_c = wp.transform_get_translation(X_wc)
    q_p = wp.transform_get_rotation(X_wp)
    q_c = wp.transform_get_rotation(X_wc)

    # Position error: c_pos = x_child - x_parent
    x_err = x_c - x_p

    # Orientation error as axis-angle
    # q_err = q_p^{-1} * q_c represents rotation from parent to child
    # This gives the axis-angle in the PARENT joint frame.
    # We must rotate it to world frame so it is consistent with the
    # world-frame Jacobian used by fill_locked_orientation_constraint.
    r_err = wp.quat_inverse(q_p) * q_c
    axis_angle = wp.vec3(r_err[0], r_err[1], r_err[2])
    angle = wp.acos(wp.clamp(r_err[3], -1.0, 1.0)) * 2.0
    axis_len = wp.length(axis_angle)
    if axis_len > 1e-10:
        ang_err_local = axis_angle / axis_len * angle
    else:
        ang_err_local = wp.vec3(0.0, 0.0, 0.0)
    # Rotate from parent joint frame to world frame
    ang_err = wp.quat_rotate(q_p, ang_err_local)

    # Lever arms from body COMs to joint frames
    r_a = wp.vec3(0.0, 0.0, 0.0)
    r_b = wp.vec3(0.0, 0.0, 0.0)

    if c_parent >= 0:
        X_wb_a = body_q[c_parent]
        com_a = body_com[c_parent]
        r_a = x_p - wp.transform_point(X_wb_a, com_a)

    X_wb_b = body_q[c_child]
    com_b = body_com[c_child]
    r_b = x_c - wp.transform_point(X_wb_b, com_b)

    # Jacobian row index from precomputed offset
    jac_row = nc_offset[tid]

    # =========================================================================
    # FIXED joint (type 3): 6 constraints - all DOFs locked
    # Composition: SPHERICAL (3) + LOCKED_ORIENTATION (3)
    # =========================================================================
    if jtype == 3:
        c_idx = 0
        # Position constraints (3): using SPHERICAL
        c_idx = fill_spherical_constraint(jac_row, c_idx, r_a, r_b, x_err, joint_jac, joint_violation)
        # Orientation constraints (3): using LOCKED_ORIENTATION
        c_idx = fill_locked_orientation_constraint(jac_row, c_idx, ang_err, joint_jac, joint_violation)

    # =========================================================================
    # BALL joint (type 2): 3 constraints - position locked, rotation free
    # Composition: SPHERICAL (3)
    # =========================================================================
    elif jtype == 2:
        c_idx = 0
        c_idx = fill_spherical_constraint(jac_row, c_idx, r_a, r_b, x_err, joint_jac, joint_violation)

    # =========================================================================
    # REVOLUTE joint (type 1): 5 constraints - rotation only about axis
    # Composition: SPHERICAL (3) + PERPENDICULAR_ROTATION (2)
    # =========================================================================
    elif jtype == 1:
        c_idx = 0
        axis = joint_axis[joint_qd_start[tid]]
        axis_p = wp.quat_rotate(q_p, axis)

        # Position constraints (3): using SPHERICAL
        c_idx = fill_spherical_constraint(jac_row, c_idx, r_a, r_b, x_err, joint_jac, joint_violation)

        # Find two vectors perpendicular to the rotation axis
        perp1 = get_orthogonal_vectors(axis_p)
        perp2 = wp.cross(axis_p, perp1)

        # Orientation constraints (2): lock rotation perpendicular to axis
        axis_c = wp.quat_rotate(q_c, axis)
        swing_err = wp.cross(axis_p, axis_c)
        c_idx = fill_perpendicular_rotation_constraint(
            jac_row, c_idx, perp1, perp2, swing_err, joint_jac, joint_violation
        )

    # =========================================================================
    # PRISMATIC joint (type 0): 5 constraints - translation only along axis
    # Composition: PERPENDICULAR_POSITION (2) + LOCKED_ORIENTATION (3)
    # =========================================================================
    elif jtype == 0:
        c_idx = 0
        axis = joint_axis[joint_qd_start[tid]]
        axis_p = wp.quat_rotate(q_p, axis)

        # Find two vectors perpendicular to the sliding axis
        perp1 = get_orthogonal_vectors(axis_p)
        perp2 = wp.cross(axis_p, perp1)

        # Position constraints (2): lock position perpendicular to axis
        c_idx = fill_perpendicular_position_constraint(
            jac_row, c_idx, perp1, perp2, x_err, r_a, r_b, joint_jac, joint_violation
        )

        # Orientation constraints (3): lock all rotation
        c_idx = fill_locked_orientation_constraint(jac_row, c_idx, ang_err, joint_jac, joint_violation)

    # =========================================================================
    # D6 joint (type 6): General 6-DOF joint with configurable axes
    # Number of constraints depends on configuration:
    # - n_linear==0: 3 position constraints (lock all, like ball joint position)
    # - n_linear==1 or 2: 2 position constraints (lock perpendicular to axis)
    # - n_linear==3: 0 position constraints (all linear DOFs free)
    # Same logic for angular DOFs
    # =========================================================================
    elif jtype == 6:
        qd_start = joint_qd_start[tid]
        n_linear = joint_dof_dim[tid, 0]
        n_angular = joint_dof_dim[tid, 1]

        c_idx = 0  # Current constraint index

        # Position constraints based on n_linear
        if n_linear == 0:
            # No linear DOFs -> lock all 3 position DOFs (SPHERICAL)
            c_idx = fill_spherical_constraint(jac_row, c_idx, r_a, r_b, x_err, joint_jac, joint_violation)
        elif n_linear < 3:
            # 1 or 2 linear DOFs -> lock position perpendicular to linear axis
            lin_axis = joint_axis[qd_start]
            lin_axis_p = wp.quat_rotate(q_p, lin_axis)
            lin_perp1 = get_orthogonal_vectors(lin_axis_p)
            lin_perp2 = wp.cross(lin_axis_p, lin_perp1)
            c_idx = fill_perpendicular_position_constraint(
                jac_row, c_idx, lin_perp1, lin_perp2, x_err, r_a, r_b, joint_jac, joint_violation
            )
        # else: n_linear == 3 -> all linear DOFs free, no position constraints

        # Angular constraints based on n_angular
        if n_angular == 0:
            # No angular DOFs -> lock all 3 rotation DOFs (LOCKED_ORIENTATION)
            c_idx = fill_locked_orientation_constraint(jac_row, c_idx, ang_err, joint_jac, joint_violation)
        elif n_angular < 3:
            # 1 or 2 angular DOFs -> lock rotation perpendicular to angular axis
            ang_axis = joint_axis[qd_start + n_linear]
            ang_axis_p = wp.quat_rotate(q_p, ang_axis)
            ang_perp1 = get_orthogonal_vectors(ang_axis_p)
            ang_perp2 = wp.cross(ang_axis_p, ang_perp1)
            axis_c = wp.quat_rotate(q_c, ang_axis)
            swing_err = wp.cross(ang_axis_p, axis_c)
            c_idx = fill_perpendicular_rotation_constraint(
                jac_row, c_idx, ang_perp1, ang_perp2, swing_err, joint_jac, joint_violation
            )
        # else: n_angular == 3 -> all angular DOFs free, no orientation constraints


# =============================================================================
# Joint Residual Computation
# =============================================================================


@wp.kernel
def compute_joint_residual(
    body_qd: wp.array[wp.spatial_vector],
    body_f: wp.array[wp.spatial_vector],
    body_inv_mass: wp.array[float],
    body_inv_inertia_world: wp.array[wp.mat33],
    joint_jac: wp.array2d[float],
    joint_violation: wp.array[float],
    joint_body_a: wp.array[int],
    joint_body_b: wp.array[int],
    joint_num_constraints: wp.array[int],
    nc_offset: wp.array[wp.int32],
    dt: float,
    alpha: float,
    recovery_speed: float,
    num_joints: int,
    # outputs
    joint_residual: wp.array[float],
):
    """
    Compute joint constraint residual (right-hand side).

    The residual is:
        b = -(J * v_pred + baumgarte_correction)

    where:
        - v_pred = v + dt * M_inv * f  (predicted velocity)
        - baumgarte_correction = clamp(phi / (dt + alpha), -recovery_speed, recovery_speed)
        - phi is the constraint violation
        - alpha is the Baumgarte damping parameter (DVI-style)
        - recovery_speed limits the Baumgarte correction velocity

    The clamping prevents overcorrection when violations are large,
    matching Project DVI's max_penetration_recovery_speed behavior.

    Args:
        body_qd: Body spatial velocities.
        body_f: Body spatial forces.
        body_inv_mass: Inverse masses.
        body_inv_inertia_world: Inverse inertia tensors in world frame.
        joint_jac: Joint Jacobians (num_joints x 6, 12).
        joint_violation: Constraint violations.
        joint_body_a, joint_body_b: Body indices.
        joint_num_constraints: Number of constraints per joint.
        dt: Time step.
        alpha: Baumgarte damping parameter.
        recovery_speed: Maximum Baumgarte correction velocity (m/s or rad/s).
            Set to -1 for unlimited. Typical value: 0.6 (DVI default).
        num_joints: Number of joints.

    Outputs:
        joint_residual: Constraint residuals.
    """
    tid = wp.tid()

    if tid >= num_joints:
        return

    num_c = joint_num_constraints[tid]
    if num_c == 0:
        return

    body_a = joint_body_a[tid]
    body_b = joint_body_b[tid]
    jac_row = nc_offset[tid]

    for c in range(num_c):
        # Extract Jacobian row
        J_a_lin = wp.vec3(joint_jac[jac_row + c, 0], joint_jac[jac_row + c, 1], joint_jac[jac_row + c, 2])
        J_a_ang = wp.vec3(joint_jac[jac_row + c, 3], joint_jac[jac_row + c, 4], joint_jac[jac_row + c, 5])
        J_b_lin = wp.vec3(joint_jac[jac_row + c, 6], joint_jac[jac_row + c, 7], joint_jac[jac_row + c, 8])
        J_b_ang = wp.vec3(joint_jac[jac_row + c, 9], joint_jac[jac_row + c, 10], joint_jac[jac_row + c, 11])

        res = 0.0

        # Parent body contribution: J_a · v_pred_a
        if body_a >= 0:
            v_a = body_qd[body_a]
            f_a = body_f[body_a]
            inv_m_a = body_inv_mass[body_a]
            inv_I_a = body_inv_inertia_world[body_a]

            # v_pred = v + dt · M^{-1} · f
            accel_lin_a = wp.spatial_top(f_a) * inv_m_a
            accel_ang_a = inv_I_a * wp.spatial_bottom(f_a)
            v_pred_lin_a = wp.spatial_top(v_a) + dt * accel_lin_a
            v_pred_ang_a = wp.spatial_bottom(v_a) + dt * accel_ang_a

            res = res + wp.dot(J_a_lin, v_pred_lin_a) + wp.dot(J_a_ang, v_pred_ang_a)

        # Child body contribution: J_b · v_pred_b
        v_b = body_qd[body_b]
        f_b = body_f[body_b]
        inv_m_b = body_inv_mass[body_b]
        inv_I_b = body_inv_inertia_world[body_b]

        accel_lin_b = wp.spatial_top(f_b) * inv_m_b
        accel_ang_b = inv_I_b * wp.spatial_bottom(f_b)
        v_pred_lin_b = wp.spatial_top(v_b) + dt * accel_lin_b
        v_pred_ang_b = wp.spatial_bottom(v_b) + dt * accel_ang_b

        res = res + wp.dot(J_b_lin, v_pred_lin_b) + wp.dot(J_b_ang, v_pred_ang_b)

        # Add Baumgarte position correction (alpha-damped, optionally clamped)
        phi = joint_violation[jac_row + c]
        baumgarte = phi / (dt + alpha)

        # Clamp Baumgarte correction if recovery speed is set (> 0)
        # This prevents overcorrection with large violations, matching DVI's behavior
        if recovery_speed > 0.0:
            baumgarte = wp.clamp(baumgarte, -recovery_speed, recovery_speed)

        res = res + baumgarte

        joint_residual[jac_row + c] = -res


# =============================================================================
# Joint Diagonal Computation
# =============================================================================


@wp.kernel
def compute_joint_diagonal(
    body_inv_mass: wp.array[float],
    body_inv_inertia_world: wp.array[wp.mat33],
    joint_jac: wp.array2d[float],
    joint_body_a: wp.array[int],
    joint_body_b: wp.array[int],
    joint_num_constraints: wp.array[int],
    nc_offset: wp.array[wp.int32],
    reg: float,
    num_joints: int,
    # outputs
    joint_diag: wp.array[float],
):
    """
    Compute diagonal of N = J * M_inv * J^T for joint constraints.

    For constraint c of joint i:
        N_cc = J_a * M_inv_a * J_a^T + J_b * M_inv_b * J_b^T

    The diagonal is used as preconditioner for iterative solvers.

    Args:
        body_inv_mass: Inverse masses.
        body_inv_inertia_world: Inverse inertia tensors in world frame.
        joint_jac: Joint Jacobians.
        joint_body_a, joint_body_b: Body indices.
        joint_num_constraints: Constraints per joint.
        nc_offset: Prefix-sum offset for each joint.
        reg: Regularization added to diagonal.
        num_joints: Number of joints.

    Outputs:
        joint_diag: Diagonal entries per constraint.
    """
    tid = wp.tid()

    if tid >= num_joints:
        return

    num_c = joint_num_constraints[tid]
    if num_c == 0:
        return

    body_a = joint_body_a[tid]
    body_b = joint_body_b[tid]
    jac_row = nc_offset[tid]

    for c in range(num_c):
        J_a_lin = wp.vec3(joint_jac[jac_row + c, 0], joint_jac[jac_row + c, 1], joint_jac[jac_row + c, 2])
        J_a_ang = wp.vec3(joint_jac[jac_row + c, 3], joint_jac[jac_row + c, 4], joint_jac[jac_row + c, 5])
        J_b_lin = wp.vec3(joint_jac[jac_row + c, 6], joint_jac[jac_row + c, 7], joint_jac[jac_row + c, 8])
        J_b_ang = wp.vec3(joint_jac[jac_row + c, 9], joint_jac[jac_row + c, 10], joint_jac[jac_row + c, 11])

        d = reg

        # Parent contribution
        if body_a >= 0:
            inv_m_a = body_inv_mass[body_a]
            inv_I_a = body_inv_inertia_world[body_a]
            d = d + wp.dot(J_a_lin, J_a_lin) * inv_m_a + wp.dot(J_a_ang, inv_I_a * J_a_ang)

        # Child contribution
        inv_m_b = body_inv_mass[body_b]
        inv_I_b = body_inv_inertia_world[body_b]
        d = d + wp.dot(J_b_lin, J_b_lin) * inv_m_b + wp.dot(J_b_ang, inv_I_b * J_b_ang)

        joint_diag[jac_row + c] = d


# =============================================================================
# Position Correction Kernels
# =============================================================================


@wp.kernel
def apply_position_correction(
    position_correction: wp.array[wp.spatial_vector],
    body_inv_mass: wp.array[float],
    # outputs
    body_q: wp.array[wp.transform],
):
    """
    Apply position correction to body transforms.

    The position update is:
        q = q + delta_q

    where delta_q is computed from:
        delta_q = M_inv * J^T * nu

    The correction is a 6D spatial vector:
        - Linear part (top 3): translation correction (world frame)
        - Angular part (bottom 3): rotation correction (world frame, axis-angle)

    Args:
        position_correction: 6D correction per body (linear, angular).
        body_inv_mass: Inverse mass (used to skip fixed bodies).

    Outputs:
        body_q: Body transforms (modified in place).
    """
    tid = wp.tid()

    # Skip fixed bodies (inv_mass = 0)
    if body_inv_mass[tid] <= 0.0:
        return

    # Get current transform
    q = body_q[tid]
    x0 = wp.transform_get_translation(q)
    r0 = wp.transform_get_rotation(q)

    # Get correction
    corr = position_correction[tid]
    delta_x = wp.spatial_top(corr)  # Linear correction (world frame)
    delta_w = wp.spatial_bottom(corr)  # Angular correction (world frame, axis-angle)

    # Apply linear correction
    x1 = x0 + delta_x

    # Apply angular correction using exponential map approximation
    # The angular correction is in world frame, use left multiplication
    # r_new = dq * r_old (world frame angular change)
    dq = wp.quat(0.5 * delta_w[0], 0.5 * delta_w[1], 0.5 * delta_w[2], 1.0)
    r1 = wp.normalize(dq * r0)

    body_q[tid] = wp.transform(x1, r1)


# =============================================================================
# Unified Constraint Kernels
# =============================================================================
@wp.kernel
def compute_delta_v(
    body_inv_mass: wp.array[float],
    body_inv_inertia_world: wp.array[wp.mat33],
    jacobian: wp.array2d[float],
    body_a: wp.array[int],
    body_b: wp.array[int],
    row_to_block: wp.array[wp.int32],
    lambda_: wp.array[float],
    active_row_count: wp.array[int],
    rows_per_block: int,
    # outputs
    delta_v: wp.array[wp.spatial_vector],
):
    """
    Compute velocity delta from constraint impulses.

    Launches one thread per constraint row for better GPU parallelism.
    Computes delta_v = M_inv * J^T * lambda for all constraint types.
    Uses atomic_add because multiple rows may affect the same body.

    Args:
        body_inv_mass: Inverse masses.
        body_inv_inertia_world: Inverse inertia tensors in world frame.
        jacobian: Jacobian matrix [total_nc, 12].
        body_a, body_b: Body indices per block.
        row_to_block: Precomputed block index per row (for joints).
        lambda_: Constraint impulses per row.
        active_row_count: GPU array [1] with active row count (read on GPU for graph compatibility).
        rows_per_block: Fixed rows per block (3 for contacts), or 0 to use row_to_block lookup.

    Outputs:
        delta_v: Velocity change per body (accumulated via atomic_add).
    """
    row = wp.tid()

    # Read active row count from GPU array (CUDA graph compatible)
    active_nc = active_row_count[0]

    # Skip rows beyond active constraints (prevents stale data processing)
    if row >= active_nc:
        return

    lam = lambda_[row]

    # Skip zero impulse (common for inactive constraints)
    if lam == 0.0:
        return

    # Find which block this row belongs to (O(1) lookup)
    if rows_per_block > 0:
        # Contacts: fixed 3 rows per block
        bid = row // rows_per_block
    else:
        # Joints: precomputed lookup
        bid = row_to_block[row]
    ba = body_a[bid]
    bb = body_b[bid]

    J_a_lin = wp.vec3(jacobian[row, 0], jacobian[row, 1], jacobian[row, 2])
    J_a_ang = wp.vec3(jacobian[row, 3], jacobian[row, 4], jacobian[row, 5])
    J_b_lin = wp.vec3(jacobian[row, 6], jacobian[row, 7], jacobian[row, 8])
    J_b_ang = wp.vec3(jacobian[row, 9], jacobian[row, 10], jacobian[row, 11])

    if ba >= 0:
        inv_m_a = body_inv_mass[ba]
        inv_I_a = body_inv_inertia_world[ba]
        dv_a_lin = J_a_lin * (inv_m_a * lam)
        dv_a_ang = inv_I_a * J_a_ang * lam
        wp.atomic_add(delta_v, ba, wp.spatial_vector(dv_a_lin, dv_a_ang))

    if bb >= 0:
        inv_m_b = body_inv_mass[bb]
        inv_I_b = body_inv_inertia_world[bb]
        dv_b_lin = J_b_lin * (inv_m_b * lam)
        dv_b_ang = inv_I_b * J_b_ang * lam
        wp.atomic_add(delta_v, bb, wp.spatial_vector(dv_b_lin, dv_b_ang))


@wp.kernel
def jacobi_iteration(
    jacobian: wp.array2d[float],
    residual: wp.array[float],
    body_a: wp.array[int],
    body_b: wp.array[int],
    row_to_block: wp.array[wp.int32],
    diag: wp.array[float],
    delta_v: wp.array[wp.spatial_vector],
    omega: float,
    relax: float,
    active_row_count: wp.array[int],
    rows_per_block: int,
    lambda_old: wp.array[float],
    # outputs
    lambda_new: wp.array[float],
):
    """
    Unified Jacobi iteration for all constraint types.

    Launches one thread per constraint row for better GPU parallelism.
    Solves N * lambda = b via matrix-free Jacobi iterations.
    Works for both joint and contact constraints.

    For each constraint row:
        1. Compute: g = J * delta_v - b
        2. Update: lambda_hat = lambda_old - omega * g / D_eff
        3. Relax: lambda_new = relax * lambda_hat + (1-relax) * lambda_old

    For contacts (rows_per_block == 3), uses trace-based preconditioning:
    D_eff = trace(diag_block) / 3, where trace = diag[3i] + diag[3i+1] + diag[3i+2].
    This captures the average coupling across normal/tangential directions.

    For joints, uses the per-row diagonal: D_eff = diag[row].

    Friction cone projection should be applied as a separate step.

    Args:
        jacobian: Jacobian matrix [total_nc, 12].
        residual: RHS vector (b) per row.
        body_a, body_b: Body indices per block.
        row_to_block: Precomputed block index per row (for joints).
        diag: Diagonal preconditioner per row.
        delta_v: Current M_inv * J^T * lambda.
        omega: Step size.
        relax: Solution mixing factor.
        active_row_count: GPU array [1] with active row count (read on GPU for graph compatibility).
        rows_per_block: Fixed rows per block (3 for contacts), or 0 to use row_to_block lookup.
        lambda_old: Input impulses.
        lambda_new: Output impulses.
    """
    row = wp.tid()

    # Read active row count from GPU array (CUDA graph compatible)
    active_nc = active_row_count[0]

    # Skip rows beyond active constraints (prevents stale data processing)
    if row >= active_nc:
        return

    # Jacobi update
    lam_old = lambda_old[row]

    # Compute effective diagonal preconditioner
    if rows_per_block > 0:
        # Contacts: use trace-based preconditioner = avg of 3 diagonal entries
        bid = row // rows_per_block
        row_base = bid * rows_per_block
        d = (diag[row_base] + diag[row_base + 1] + diag[row_base + 2]) / 3.0
    else:
        # Joints: per-row diagonal
        bid = row_to_block[row]
        d = diag[row]

    # Skip if diagonal is zero (inactive constraint)
    if d <= 0.0:
        lambda_new[row] = lam_old
        return
    ba = body_a[bid]
    bb = body_b[bid]

    J_a_lin = wp.vec3(jacobian[row, 0], jacobian[row, 1], jacobian[row, 2])
    J_a_ang = wp.vec3(jacobian[row, 3], jacobian[row, 4], jacobian[row, 5])
    J_b_lin = wp.vec3(jacobian[row, 6], jacobian[row, 7], jacobian[row, 8])
    J_b_ang = wp.vec3(jacobian[row, 9], jacobian[row, 10], jacobian[row, 11])

    # Compute J * delta_v
    J_dv = 0.0
    if ba >= 0:
        dv_a = delta_v[ba]
        J_dv = J_dv + wp.dot(J_a_lin, wp.spatial_top(dv_a)) + wp.dot(J_a_ang, wp.spatial_bottom(dv_a))

    if bb >= 0:
        dv_b = delta_v[bb]
        J_dv = J_dv + wp.dot(J_b_lin, wp.spatial_top(dv_b)) + wp.dot(J_b_ang, wp.spatial_bottom(dv_b))

    b = residual[row]
    grad = J_dv - b
    lam_hat = lam_old - omega * grad / d

    # Relaxation
    lam_new = relax * lam_hat + (1.0 - relax) * lam_old

    lambda_new[row] = lam_new


@wp.kernel
def jacobi_iteration_block3x3(
    jacobian: wp.array2d[float],
    residual: wp.array[float],
    body_a: wp.array[int],
    body_b: wp.array[int],
    diag_block_inv: wp.array[wp.mat33],
    delta_v: wp.array[wp.spatial_vector],
    omega: float,
    relax: float,
    active_contact_count: wp.array[int],
    lambda_old: wp.array[float],
    # outputs
    lambda_new: wp.array[float],
):
    """
    Block-3x3 Jacobi iteration for contact constraints.

    One thread per contact (not per row). Computes the 3-vector gradient
    and multiplies by the precomputed 3x3 inverse of the diagonal block
    of J M^{-1} J^T. This captures cross-coupling between normal and
    tangential directions.

    For each contact:
        1. Compute gradient: g_i = J_i * delta_v - b_i  (i=0,1,2)
        2. Apply block inverse: dlam = D_inv @ g
        3. Update: lambda_hat = lambda_old - omega * dlam
        4. Relax: lambda_new = relax * lambda_hat + (1-relax) * lambda_old
    """
    tid = wp.tid()

    active_count = active_contact_count[0]
    if tid >= active_count:
        return

    ba = body_a[tid]
    bb = body_b[tid]
    row_base = tid * 3
    D_inv = diag_block_inv[tid]

    # Check if block is valid (trace > 0)
    if D_inv[0, 0] == 0.0 and D_inv[1, 1] == 0.0 and D_inv[2, 2] == 0.0:
        lambda_new[row_base + 0] = lambda_old[row_base + 0]
        lambda_new[row_base + 1] = lambda_old[row_base + 1]
        lambda_new[row_base + 2] = lambda_old[row_base + 2]
        return

    # Compute gradient vector for all 3 rows
    grad = wp.vec3(0.0, 0.0, 0.0)
    for c in range(3):
        row = row_base + c
        J_a_lin = wp.vec3(jacobian[row, 0], jacobian[row, 1], jacobian[row, 2])
        J_a_ang = wp.vec3(jacobian[row, 3], jacobian[row, 4], jacobian[row, 5])
        J_b_lin = wp.vec3(jacobian[row, 6], jacobian[row, 7], jacobian[row, 8])
        J_b_ang = wp.vec3(jacobian[row, 9], jacobian[row, 10], jacobian[row, 11])

        J_dv = 0.0
        if ba >= 0:
            dv_a = delta_v[ba]
            J_dv = J_dv + wp.dot(J_a_lin, wp.spatial_top(dv_a)) + wp.dot(J_a_ang, wp.spatial_bottom(dv_a))
        if bb >= 0:
            dv_b = delta_v[bb]
            J_dv = J_dv + wp.dot(J_b_lin, wp.spatial_top(dv_b)) + wp.dot(J_b_ang, wp.spatial_bottom(dv_b))

        grad[c] = J_dv - residual[row]

    # Apply block-3x3 inverse preconditioner
    dlam = D_inv * grad

    # Update all 3 rows
    for c in range(3):
        row = row_base + c
        lam_old = lambda_old[row]
        lam_hat = lam_old - omega * dlam[c]
        lambda_new[row] = relax * lam_hat + (1.0 - relax) * lam_old


@wp.kernel
def apply_constraint_forces(
    jacobian: wp.array2d[float],
    body_a: wp.array[int],
    body_b: wp.array[int],
    row_to_block: wp.array[wp.int32],
    lambda_: wp.array[float],
    dt: float,
    total_nc: int,
    # outputs
    body_f: wp.array[wp.spatial_vector],
):
    """
    Apply constraint forces to bodies.

    Launches one thread per constraint row for better GPU parallelism.
    Computes F = J^T * lambda / dt for all constraint types.

    Args:
        jacobian: Jacobian matrix [total_nc, 12].
        body_a, body_b: Body indices per block.
        row_to_block: Precomputed block index per row.
        lambda_: Solved impulses.
        dt: Time step.
        total_nc: Total number of constraint rows.

    Outputs:
        body_f: Body forces (accumulated via atomic_add).
    """
    row = wp.tid()

    if row >= total_nc:
        return

    lam = lambda_[row]

    # Skip zero impulse (common for inactive constraints)
    if lam == 0.0:
        return

    # Find which block this row belongs to (O(1) lookup)
    bid = row_to_block[row]
    ba = body_a[bid]
    bb = body_b[bid]

    J_a_lin = wp.vec3(jacobian[row, 0], jacobian[row, 1], jacobian[row, 2])
    J_a_ang = wp.vec3(jacobian[row, 3], jacobian[row, 4], jacobian[row, 5])
    J_b_lin = wp.vec3(jacobian[row, 6], jacobian[row, 7], jacobian[row, 8])
    J_b_ang = wp.vec3(jacobian[row, 9], jacobian[row, 10], jacobian[row, 11])

    # F = J^T * lambda / dt
    if ba >= 0:
        f_a = wp.spatial_vector(J_a_lin * lam / dt, J_a_ang * lam / dt)
        wp.atomic_add(body_f, ba, f_a)

    if bb >= 0:
        f_b = wp.spatial_vector(J_b_lin * lam / dt, J_b_ang * lam / dt)
        wp.atomic_add(body_f, bb, f_b)


# ---------------------------------------------------------------------------
# Joint limit (unilateral) constraint kernels
#
# Joint limits are formulated as inequality constraints:
#   - Lower limit: q >= q_lo  →  C = q - q_lo >= 0
#   - Upper limit: q <= q_hi  →  C = q_hi - q >= 0
#
# Each violated limit produces one constraint row with Jacobian:
#   J = [0, ±a_w, 0, ∓a_w]  (angular, parent/child)
#
# The sign convention ensures λ >= 0 always (like contact normal impulse).
# ---------------------------------------------------------------------------


@wp.func
def _quat_twist_angle(q: wp.quat, axis: wp.vec3) -> float:
    """Extract twist angle from quaternion about a given axis.

    Uses twist-swing decomposition — identical to the version in
    ``actuation_kernels.quat_twist_angle`` so that the constraint
    solver measures the same angle as the penalty / implicit-PD path.
    """
    qv = wp.vec3(q[0], q[1], q[2])
    qw = q[3]
    twist_scalar = wp.dot(qv, axis)
    twist_len = wp.sqrt(twist_scalar * twist_scalar + qw * qw)
    if twist_len < 1.0e-8:
        return 0.0
    twist_x = wp.clamp(twist_scalar / twist_len, -1.0, 1.0)
    return 2.0 * wp.asin(twist_x)


@wp.func
def compute_revolute_angle(q_p: wp.quat, q_c: wp.quat, axis: wp.vec3) -> float:
    """Compute relative rotation angle for a revolute joint.

    Matches ``actuation_kernels.compute_relative_twist_angle``:
    hemisphere check + proper twist-swing decomposition.
    """
    # Hemisphere check (same as actuation_kernels)
    dot_quat = q_p[0] * q_c[0] + q_p[1] * q_c[1] + q_p[2] * q_c[2] + q_p[3] * q_c[3]
    if dot_quat < 0.0:
        q_c = wp.quat(-q_c[0], -q_c[1], -q_c[2], -q_c[3])
    q_rel = wp.quat_inverse(q_p) * q_c
    return _quat_twist_angle(q_rel, axis)


@wp.kernel
def compute_joint_limit_jacobians(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    joint_type: wp.array[int],
    joint_enabled: wp.array[bool],
    joint_parent: wp.array[int],
    joint_child: wp.array[int],
    joint_X_p: wp.array[wp.transform],
    joint_X_c: wp.array[wp.transform],
    joint_axis: wp.array[wp.vec3],
    joint_qd_start: wp.array[int],
    joint_dof_dim: wp.array2d[int],
    joint_limit_lower: wp.array[float],
    joint_limit_upper: wp.array[float],
    joint_limit_ke: wp.array[float],
    limit_nc_offset: wp.array[wp.int32],
    limit_max_nc: int,
    # outputs
    jacobian: wp.array2d[float],
    violation: wp.array[float],
    body_a: wp.array[int],
    body_b: wp.array[int],
    active_limit_count: wp.array[int],
):
    """Compute Jacobians and violations for active joint limits.

    One thread per joint. For each DOF with finite limits, checks if the
    current angle violates or nearly violates the limit. If so, writes
    a constraint row.

    The Jacobian convention ensures λ >= 0:
      - Lower limit: J pushes q upward (away from lower limit)
      - Upper limit: J pushes q downward (away from upper limit)
    """
    tid = wp.tid()

    if not joint_enabled[tid]:
        return

    jtype = joint_type[tid]
    # Skip FIXED, FREE, COMPOUND (no limits)
    if jtype == 3 or jtype == 4 or jtype == 5:
        return

    id_p = joint_parent[tid]
    id_c = joint_child[tid]
    qd_start = joint_qd_start[tid]
    row_base = limit_nc_offset[tid]

    # Compute parent joint frame in world
    if id_p < 0:
        X_wp = body_q[id_c]
    else:
        X_wp = body_q[id_p]

    X_wc = body_q[id_c]
    X_pj = joint_X_p[tid]
    X_cj = joint_X_c[tid]
    X_wj_p = wp.transform_multiply(X_wp, X_pj)
    X_wj_c = wp.transform_multiply(X_wc, X_cj)
    q_p = wp.transform_get_rotation(X_wj_p)
    q_c = wp.transform_get_rotation(X_wj_c)

    # Local row counter for this joint
    local_row = 0

    # REVOLUTE (type 1): 1 angular DOF
    if jtype == 1:
        lo = joint_limit_lower[qd_start]
        hi = joint_limit_upper[qd_start]
        ke = joint_limit_ke[qd_start]
        if ke > 0.0 and lo < hi:
            axis = joint_axis[qd_start]
            axis_w = wp.quat_rotate(q_p, axis)
            q_meas = compute_revolute_angle(q_p, q_c, axis)

            row = row_base + local_row
            if row < limit_max_nc:
                if q_meas < lo:
                    # Lower limit violated: C = q - q_lo, push q up
                    # J_parent_ang = -axis_w, J_child_ang = +axis_w
                    violation[row] = q_meas - lo  # negative when violated
                    for i in range(3):
                        jacobian[row, 3 + i] = -axis_w[i]  # parent angular
                        jacobian[row, 9 + i] = axis_w[i]   # child angular
                    body_a[local_row + limit_nc_offset[tid] // 1] = id_p  # handled below
                    wp.atomic_add(active_limit_count, 0, 1)
                elif q_meas > hi:
                    # Upper limit violated: C = q_hi - q, push q down
                    # J_parent_ang = +axis_w, J_child_ang = -axis_w
                    violation[row] = hi - q_meas  # negative when violated
                    for i in range(3):
                        jacobian[row, 3 + i] = axis_w[i]   # parent angular
                        jacobian[row, 9 + i] = -axis_w[i]  # child angular
                    wp.atomic_add(active_limit_count, 0, 1)
                else:
                    # Not violated — zero this row
                    violation[row] = 0.0

                body_a[row] = id_p
                body_b[row] = id_c
            local_row += 1

    # BALL (type 2): 3 angular DOFs — check each independently
    elif jtype == 2:
        # Hemisphere check for relative quaternion (same as actuation)
        dot_quat = q_p[0] * q_c[0] + q_p[1] * q_c[1] + q_p[2] * q_c[2] + q_p[3] * q_c[3]
        q_c_h = q_c
        if dot_quat < 0.0:
            q_c_h = wp.quat(-q_c[0], -q_c[1], -q_c[2], -q_c[3])
        q_rel = wp.quat_inverse(q_p) * q_c_h

        for dof in range(3):
            dof_idx = qd_start + dof
            lo = joint_limit_lower[dof_idx]
            hi = joint_limit_upper[dof_idx]
            ke = joint_limit_ke[dof_idx]
            if ke > 0.0 and lo < hi:
                axis = joint_axis[dof_idx]
                axis_w = wp.quat_rotate(q_p, axis)
                q_meas = _quat_twist_angle(q_rel, axis)

                row = row_base + local_row
                if row < limit_max_nc:
                    if q_meas < lo:
                        violation[row] = q_meas - lo
                        for i in range(3):
                            jacobian[row, 3 + i] = -axis_w[i]
                            jacobian[row, 9 + i] = axis_w[i]
                        wp.atomic_add(active_limit_count, 0, 1)
                    elif q_meas > hi:
                        violation[row] = hi - q_meas
                        for i in range(3):
                            jacobian[row, 3 + i] = axis_w[i]
                            jacobian[row, 9 + i] = -axis_w[i]
                        wp.atomic_add(active_limit_count, 0, 1)
                    else:
                        violation[row] = 0.0

                    body_a[row] = id_p
                    body_b[row] = id_c
                local_row += 1

    # D6 (type 6): Variable DOFs
    elif jtype == 6:
        n_linear = joint_dof_dim[tid, 0]
        n_angular = joint_dof_dim[tid, 1]

        # Hemisphere check for relative quaternion (same as actuation)
        dot_quat_d6 = q_p[0] * q_c[0] + q_p[1] * q_c[1] + q_p[2] * q_c[2] + q_p[3] * q_c[3]
        q_c_h_d6 = q_c
        if dot_quat_d6 < 0.0:
            q_c_h_d6 = wp.quat(-q_c[0], -q_c[1], -q_c[2], -q_c[3])
        q_rel_d6 = wp.quat_inverse(q_p) * q_c_h_d6

        # Angular DOFs (same logic as ball, offset by n_linear)
        for dof in range(n_angular):
            dof_idx = qd_start + n_linear + dof
            lo = joint_limit_lower[dof_idx]
            hi = joint_limit_upper[dof_idx]
            ke = joint_limit_ke[dof_idx]
            if ke > 0.0 and lo < hi:
                axis = joint_axis[dof_idx]
                axis_w = wp.quat_rotate(q_p, axis)
                q_meas = _quat_twist_angle(q_rel_d6, axis)

                row = row_base + local_row
                if row < limit_max_nc:
                    if q_meas < lo:
                        violation[row] = q_meas - lo
                        for i in range(3):
                            jacobian[row, 3 + i] = -axis_w[i]
                            jacobian[row, 9 + i] = axis_w[i]
                        wp.atomic_add(active_limit_count, 0, 1)
                    elif q_meas > hi:
                        violation[row] = hi - q_meas
                        for i in range(3):
                            jacobian[row, 3 + i] = axis_w[i]
                            jacobian[row, 9 + i] = -axis_w[i]
                        wp.atomic_add(active_limit_count, 0, 1)
                    else:
                        violation[row] = 0.0

                    body_a[row] = id_p
                    body_b[row] = id_c
                local_row += 1

    # PRISMATIC (type 0): 1 linear DOF
    elif jtype == 0:
        lo = joint_limit_lower[qd_start]
        hi = joint_limit_upper[qd_start]
        ke = joint_limit_ke[qd_start]
        if ke > 0.0 and lo < hi:
            axis = joint_axis[qd_start]
            axis_w = wp.quat_rotate(q_p, axis)

            # Compute linear displacement along axis
            p_p = wp.transform_get_translation(X_wj_p)
            p_c = wp.transform_get_translation(X_wj_c)
            q_meas = wp.dot(p_c - p_p, axis_w)

            row = row_base + local_row
            if row < limit_max_nc:
                if q_meas < lo:
                    # Lower limit: push along +axis
                    violation[row] = q_meas - lo
                    for i in range(3):
                        jacobian[row, 0 + i] = -axis_w[i]  # parent linear
                        jacobian[row, 6 + i] = axis_w[i]   # child linear
                    wp.atomic_add(active_limit_count, 0, 1)
                elif q_meas > hi:
                    # Upper limit: push along -axis
                    violation[row] = hi - q_meas
                    for i in range(3):
                        jacobian[row, 0 + i] = axis_w[i]
                        jacobian[row, 6 + i] = -axis_w[i]
                    wp.atomic_add(active_limit_count, 0, 1)
                else:
                    violation[row] = 0.0

                body_a[row] = id_p
                body_b[row] = id_c
            local_row += 1


@wp.kernel
def compute_joint_limit_residual(
    body_qd: wp.array[wp.spatial_vector],
    body_f: wp.array[wp.spatial_vector],
    body_inv_mass: wp.array[float],
    body_inv_inertia_world: wp.array[wp.mat33],
    jacobian: wp.array2d[float],
    violation: wp.array[float],
    body_a: wp.array[int],
    body_b: wp.array[int],
    dt: float,
    alpha: float,
    recovery_speed: float,
    limit_max_nc: int,
    # outputs
    residual: wp.array[float],
):
    """Compute residual (RHS) for joint limit constraints.

    b_i = -(J_i @ v_pred + phi_i / (dt + alpha))

    where v_pred = v + dt * M^{-1} * f is the predicted velocity and
    phi is the constraint violation (clamped by recovery_speed).
    """
    row = wp.tid()
    if row >= limit_max_nc:
        return

    phi = violation[row]

    ba = body_a[row]
    bb = body_b[row]

    J_a_lin = wp.vec3(jacobian[row, 0], jacobian[row, 1], jacobian[row, 2])
    J_a_ang = wp.vec3(jacobian[row, 3], jacobian[row, 4], jacobian[row, 5])
    J_b_lin = wp.vec3(jacobian[row, 6], jacobian[row, 7], jacobian[row, 8])
    J_b_ang = wp.vec3(jacobian[row, 9], jacobian[row, 10], jacobian[row, 11])

    # Compute J @ v_pred (v_pred = v + dt * M_inv * f)
    Jv = float(0.0)
    if ba >= 0:
        v_a = body_qd[ba]
        f_a = body_f[ba]
        inv_m_a = body_inv_mass[ba]
        inv_I_a = body_inv_inertia_world[ba]
        v_pred_lin = wp.spatial_top(v_a) + dt * inv_m_a * wp.spatial_top(f_a)
        v_pred_ang = wp.spatial_bottom(v_a) + dt * (inv_I_a * wp.spatial_bottom(f_a))
        Jv += wp.dot(J_a_lin, v_pred_lin) + wp.dot(J_a_ang, v_pred_ang)

    if bb >= 0:
        v_b = body_qd[bb]
        f_b = body_f[bb]
        inv_m_b = body_inv_mass[bb]
        inv_I_b = body_inv_inertia_world[bb]
        v_pred_lin = wp.spatial_top(v_b) + dt * inv_m_b * wp.spatial_top(f_b)
        v_pred_ang = wp.spatial_bottom(v_b) + dt * (inv_I_b * wp.spatial_bottom(f_b))
        Jv += wp.dot(J_b_lin, v_pred_lin) + wp.dot(J_b_ang, v_pred_ang)

    # Baumgarte stabilization with recovery speed clamp
    phi_corr = phi / (dt + alpha)
    if recovery_speed >= 0.0:
        phi_corr = wp.clamp(phi_corr, -recovery_speed, recovery_speed)

    residual[row] = -(Jv + phi_corr)


@wp.kernel
def compute_joint_limit_diagonal(
    body_inv_mass: wp.array[float],
    body_inv_inertia_world: wp.array[wp.mat33],
    jacobian: wp.array2d[float],
    body_a: wp.array[int],
    body_b: wp.array[int],
    reg: float,
    limit_max_nc: int,
    # outputs
    diag: wp.array[float],
):
    """Compute diagonal of N = J M^{-1} J^T for joint limit constraints."""
    row = wp.tid()
    if row >= limit_max_nc:
        return

    ba = body_a[row]
    bb = body_b[row]

    J_a_lin = wp.vec3(jacobian[row, 0], jacobian[row, 1], jacobian[row, 2])
    J_a_ang = wp.vec3(jacobian[row, 3], jacobian[row, 4], jacobian[row, 5])
    J_b_lin = wp.vec3(jacobian[row, 6], jacobian[row, 7], jacobian[row, 8])
    J_b_ang = wp.vec3(jacobian[row, 9], jacobian[row, 10], jacobian[row, 11])

    d = reg

    if ba >= 0:
        inv_m_a = body_inv_mass[ba]
        inv_I_a = body_inv_inertia_world[ba]
        d += wp.dot(J_a_lin, J_a_lin) * inv_m_a + wp.dot(J_a_ang, inv_I_a * J_a_ang)

    if bb >= 0:
        inv_m_b = body_inv_mass[bb]
        inv_I_b = body_inv_inertia_world[bb]
        d += wp.dot(J_b_lin, J_b_lin) * inv_m_b + wp.dot(J_b_ang, inv_I_b * J_b_ang)

    diag[row] = d


@wp.kernel
def project_joint_limits(
    limit_max_nc: int,
    # in/out
    lambda_: wp.array[float],
):
    """Project joint limit impulses: clamp λ >= 0.

    The Jacobian convention ensures that positive λ always pushes the joint
    away from the violated limit, so the unilateral constraint is simply λ >= 0.
    """
    row = wp.tid()
    if row >= limit_max_nc:
        return

    lam = lambda_[row]
    if lam < 0.0:
        lambda_[row] = 0.0
