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
Joint actuation kernels for the DVI solver.

This module implements PD control for joint DOFs (degrees of freedom).
For joints with free DOFs, compute actuation forces:
    tau = kp * (q_target - q) + kd * (qd_target - qd)

These forces are applied through the actuation Jacobian, which maps joint
space forces to body space forces for the free DOFs (complementary to
the constraint Jacobian which handles constrained DOFs).

**Important - Joint Angle Convention**:
    The joint angle (q) is measured relative to the initial configuration,
    NOT as an absolute world-frame angle. When you create a joint with
    `child_xform`, the rotation component defines the "zero angle" reference.

    For example, if you create a revolute joint with a body that has initial
    rotation of 45°, and set `child_xform` rotation to the inverse of that,
    then joint angle = 0 at the initial configuration.

    **target_pos** is the desired joint angle in this relative frame.
    Setting target_pos = 0 will try to return the joint to its initial config.
    Setting target_pos = 0.5 will try to rotate the joint 0.5 rad from initial.

**Angular Target Range**:
    For angular DOFs (revolute, ball, D6 angular), `target_pos` should be
    within [-π, π] radians (approximately [-3.14, 3.14]). This is because
    joint angles are measured using arctan2, which returns values in [-π, π].

    Values outside this range will be wrapped to [-π, π]. For example:
    - target_pos = -5.0 will be wrapped to approximately 1.28 rad
    - target_pos = 4.0 will be wrapped to approximately -2.28 rad

    If you need the joint to rotate more than ±180°, you must implement
    your own angle accumulation tracking.

**Continuous Angle Handling**:
    Angular position errors are computed using continuous angle representation
    to avoid discontinuities at ±π. The error is always computed as the
    shortest path between current and target angles, ensuring smooth control
    even when the joint angle crosses the ±180° boundary.
"""

import warp as wp


PI = 3.14159265358979323846
TWO_PI = 2.0 * PI


@wp.func
def wrap_angle_to_pi(angle: float) -> float:
    """
    Wrap an angle to the range [-π, π].

    This ensures continuous angle representation for PD control.
    For example, an angle of 359° becomes -1°, which is closer to 0°
    than the unwrapped value.

    Args:
        angle: Input angle in radians.

    Returns:
        Angle wrapped to [-π, π].
    """
    # Use floor-based wrapping for robustness with large angles
    return angle - TWO_PI * wp.floor((angle + PI) / TWO_PI)


@wp.func
def compute_angle_error(q_target: float, q_meas: float) -> float:
    """
    Compute the angular error between target and measured angles.

    This function returns the shortest angular path from q_meas to q_target,
    handling the wrap-around at ±π. The result is always in [-π, π].

    For example:
        - q_target=1°, q_meas=359° → error = 2° (not -358°)
        - q_target=-170°, q_meas=170° → error = 20° (not -340°)

    Args:
        q_target: Target angle in radians.
        q_meas: Measured angle in radians.

    Returns:
        Angular error (q_target - q_meas) wrapped to [-π, π].
    """
    error = q_target - q_meas
    return wrap_angle_to_pi(error)


@wp.func
def get_orthogonal_vector(axis: wp.vec3) -> wp.vec3:
    """
    Get an orthogonal vector to the given axis.

    This is used to construct auxiliary vectors for measuring joint angles.
    The orthogonal vector is deterministic based on the input axis.

    Args:
        axis: Unit axis vector.

    Returns:
        A unit vector orthogonal to the axis.
    """
    # Choose the axis component with the smallest magnitude to avoid numerical issues
    ax = wp.abs(axis[0])
    ay = wp.abs(axis[1])

    v1 = wp.vec3(0.0, 0.0, 0.0)
    if ax > ay:
        # Use (-z, 0, x) as orthogonal vector
        v1 = wp.vec3(-axis[2], 0.0, axis[0])
    else:
        # Use (0, z, -y) as orthogonal vector
        v1 = wp.vec3(0.0, axis[2], -axis[1])

    # Normalize
    v1_len = wp.length(v1)
    if v1_len > 1.0e-8:
        v1 = v1 / v1_len
    else:
        # Fallback to z-axis if axis is degenerate
        v1 = wp.vec3(0.0, 0.0, 1.0)

    return v1


@wp.func
def quat_twist_angle(q: wp.quat, axis: wp.vec3) -> float:
    """
    Extract the twist angle from a quaternion about a given axis.

    This uses the twist-swing decomposition similar to XPBD.
    The twist is the rotation component about the specified axis.

    For quaternion q representing rotation θ about axis n:
        q = (sin(θ/2)*n, cos(θ/2)) = (sin(θ/2)*nx, sin(θ/2)*ny, sin(θ/2)*nz, cos(θ/2))

    The twist angle is computed by projecting the quaternion's vector part onto the axis.

    Args:
        q: Input quaternion.
        axis: Unit axis to extract twist about (must be normalized).

    Returns:
        Twist angle in radians, in range [-π, π].
    """
    # Project quaternion vector part onto the axis
    # For a quaternion (x, y, z, w), the vector part is (x, y, z)
    qv = wp.vec3(q[0], q[1], q[2])
    qw = q[3]

    # The twist quaternion has its vector part parallel to the axis
    # twist = (dot(qv, axis) * axis, qw) normalized
    twist_scalar = wp.dot(qv, axis)

    # Normalize the twist quaternion
    twist_len = wp.sqrt(twist_scalar * twist_scalar + qw * qw)

    # Avoid division by zero
    if twist_len < 1.0e-8:
        return 0.0

    twist_x = twist_scalar / twist_len
    twist_w = qw / twist_len

    # Twist angle = 2 * atan2(sin(θ/2), cos(θ/2))
    # atan2 handles full [-π, π] range correctly, unlike asin which
    # only works for |θ| ≤ π/2
    angle = 2.0 * wp.atan2(twist_x, twist_w)

    return angle


@wp.func
def compute_relative_twist_angle(
    q_p: wp.quat,  # Parent joint frame rotation in world
    q_c: wp.quat,  # Child joint frame rotation in world
    axis: wp.vec3,  # Joint axis in local/parent frame
) -> float:
    """
    Compute the relative rotation angle about a specified axis using twist-swing decomposition.

    This approach is consistent with how XPBD handles angular joints.

    The computation:
    1. Compute relative quaternion: rel_q = inverse(q_p) * q_c
    2. Extract twist component about the joint axis
    3. Convert twist to angle

    At the initial configuration (q_p == q_c), the angle is 0.

    Args:
        q_p: Parent joint frame rotation in world.
        q_c: Child joint frame rotation in world.
        axis: Joint axis in parent local frame (unit vector).

    Returns:
        Relative rotation angle about the axis, in radians [-π, π].
    """
    # Ensure quaternions are in the same hemisphere for proper interpolation
    dot_quat = q_p[0] * q_c[0] + q_p[1] * q_c[1] + q_p[2] * q_c[2] + q_p[3] * q_c[3]
    if dot_quat < 0.0:
        q_c = wp.quat(-q_c[0], -q_c[1], -q_c[2], -q_c[3])

    # Compute relative quaternion: rotation from parent to child
    rel_q = wp.quat_inverse(q_p) * q_c

    # Extract twist angle about the specified axis
    angle = quat_twist_angle(rel_q, axis)

    return angle


@wp.func
def compute_revolute_angle(
    q_p: wp.quat,  # Parent joint frame rotation in world
    q_c: wp.quat,  # Child joint frame rotation in world
    axis: wp.vec3,  # Joint axis in local frame
) -> float:
    """
    Compute the revolute joint angle using twist-swing decomposition.

    This is consistent with how XPBD computes angular joint errors.
    The angle is the rotation of the child frame relative to the parent frame
    about the joint axis. At the initial configuration, the angle is 0.

    Args:
        q_p: Parent joint frame rotation (in world coordinates).
        q_c: Child joint frame rotation (in world coordinates).
        axis: Joint rotation axis in local frame.

    Returns:
        Joint angle in radians, in range [-π, π].
    """
    return compute_relative_twist_angle(q_p, q_c, axis)


@wp.func
def compute_cylindrical_angle(
    q_p: wp.quat,  # Parent joint frame rotation in world
    q_c: wp.quat,  # Child joint frame rotation in world
    axis: wp.vec3,  # Joint rotation axis in local frame
) -> float:
    """
    Compute the rotation angle for a cylindrical joint using twist-swing decomposition.

    Uses the same approach as revolute joint - extracts the twist component about
    the specified axis from the relative quaternion.

    Args:
        q_p: Parent joint frame rotation (in world coordinates).
        q_c: Child joint frame rotation (in world coordinates).
        axis: Joint rotation axis in local frame.

    Returns:
        Joint angle in radians, in range [-π, π].
    """
    return compute_relative_twist_angle(q_p, q_c, axis)


@wp.kernel
def compute_actuation_forces(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    joint_type: wp.array(dtype=int),
    joint_enabled: wp.array(dtype=bool),
    joint_parent: wp.array(dtype=int),
    joint_child: wp.array(dtype=int),
    joint_X_p: wp.array(dtype=wp.transform),
    joint_X_c: wp.array(dtype=wp.transform),
    joint_axis: wp.array(dtype=wp.vec3),
    joint_q_start: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_dof_dim: wp.array2d(dtype=int),  # shape (n_joints, 2): (n_linear, n_angular) per joint
    joint_target_pos: wp.array(dtype=float),
    joint_target_vel: wp.array(dtype=float),
    joint_target_ke: wp.array(dtype=float),
    joint_target_kd: wp.array(dtype=float),
    joint_f: wp.array(dtype=float),
    use_implicit_pd: int,
    # Joint limit arrays (penalty spring-damper when violated)
    joint_limit_lower: wp.array(dtype=float),
    joint_limit_upper: wp.array(dtype=float),
    joint_limit_ke: wp.array(dtype=float),
    joint_limit_kd: wp.array(dtype=float),
    has_joint_limits: int,
    joint_effort_limit: wp.array(dtype=float),
    has_effort_limits: int,
    # outputs
    body_f_act: wp.array(dtype=wp.spatial_vector),
):
    """
    Compute actuation forces for joint DOFs and apply to bodies.

    Combines PD control forces and direct joint forces (joint_f) in a single pass:
        tau = kp * (q_target - q) + kd * (qd_target - qd) + joint_f

    When ``use_implicit_pd == 1``, the PD terms (ke, kd) are **kept at full strength**
    in this kernel.  The mass-matrix augmentation in ``implicit_pd_kernels.py``
    provides implicit stability (prevents overshooting), but the explicit PD forces
    are still required as the driving terms that actually decelerate the system.
    This is NOT double-counting — see ``implicit-pd.md`` Section 9 for the proof.

    **Joint DOF counts:**
    - PRISMATIC (type 0): 1 DOF (translation along axis)
    - REVOLUTE (type 1): 1 DOF (rotation about axis)
    - BALL (type 2): 3 DOF (free rotation)
    - FIXED (type 3): 0 DOF (no actuation)
    - D6 (type 6): Variable DOFs based on axes

    Args:
        body_q: Body transforms.
        body_qd: Body velocities.
        body_com: Body centers of mass.
        joint_type: Joint types.
        joint_enabled: Joint enabled flags.
        joint_parent, joint_child: Body indices.
        joint_X_p, joint_X_c: Joint frames.
        joint_axis: Joint axes.
        joint_q_start, joint_qd_start: Joint DOF indices.
        joint_target_pos, joint_target_vel: Target positions/velocities.
        joint_target_ke, joint_target_kd: PD gains (always applied at full strength).
        joint_f: Direct joint forces/torques per DOF.
        use_implicit_pd: 1 to enable implicit PD (mass augmentation), 0 for normal.
            In both modes, ke and kd forces are applied identically.

    Outputs:
        body_f_act: Actuation forces accumulated per body (spatial forces).
    """
    tid = wp.tid()

    if not joint_enabled[tid]:
        return

    jtype = joint_type[tid]

    # Fixed joints and FREE joints have no actuation
    # FREE joints (type 4) are handled by the constraint solver, not actuation
    if jtype == 3 or jtype == 4:
        return

    id_p = joint_parent[tid]
    id_c = joint_child[tid]

    # Get body transforms and velocities
    # When parent is world (id_p < 0), use identity transform — NOT the child body.
    # Using body_q[id_c] here would make q_pc = inv(q_wc*q_pj) * (q_wc*q_cj),
    # cancelling the child rotation and returning only inv(q_pj)*q_cj (constant).
    X_wp = wp.transform_identity() if id_p < 0 else body_q[id_p]
    X_wc = body_q[id_c]

    v_p = wp.spatial_vector() if id_p < 0 else body_qd[id_p]
    v_c = body_qd[id_c]

    com_p = wp.vec3(0.0, 0.0, 0.0) if id_p < 0 else body_com[id_p]
    com_c = body_com[id_c]

    # Joint frames in world coordinates
    X_pj = joint_X_p[tid]
    X_cj = joint_X_c[tid]

    X_wj_p = wp.transform_multiply(X_wp, X_pj)
    X_wj_c = wp.transform_multiply(X_wc, X_cj)

    # Attachment points
    r_p = wp.transform_get_translation(X_wj_p) - (
        wp.transform_get_translation(X_wp) + wp.quat_rotate(wp.transform_get_rotation(X_wp), com_p)
    )
    r_c = wp.transform_get_translation(X_wj_c) - (
        wp.transform_get_translation(X_wc) + wp.quat_rotate(wp.transform_get_rotation(X_wc), com_c)
    )

    q_p = wp.transform_get_rotation(X_wj_p)
    q_c = wp.transform_get_rotation(X_wj_c)

    # Compute relative transform (q_p^{-1} * q_c)
    q_pc = wp.mul(wp.quat_inverse(q_p), q_c)

    # Get joint DOF indices
    q_start = joint_q_start[tid]
    qd_start = joint_qd_start[tid]

    # Actuation force (spatial wrench) to apply
    f_act_p = wp.spatial_vector()
    f_act_c = wp.spatial_vector()

    # Implicit PD treatment (use_implicit_pd == 1):
    #
    # The augmented mass matrix M̃ = M + h·C_eff·J^T·J makes the body "heavier"
    # along damped DOFs, providing implicit stability.  However, the explicit
    # damping force -kd·v MUST STILL appear in the RHS (body_f) as the driving
    # term.  This is NOT double-counting.
    #
    # Proof: The correct semi-implicit equation with implicit damping is:
    #   M̃ · v^{n+1} = M · v^n + h · f_ext + h · J^T · λ
    #   where M̃ = M + h·C
    #
    # Rearranging:
    #   v^{n+1} = v^n + h · M̃⁻¹ · (f_ext - C·v^n + J^T·λ)
    #
    # The "-C·v^n" term is exactly the explicit damping force.  Without it,
    # the augmented mass only resists acceleration but never actually decelerates
    # the existing velocity — the system stays wobbly.
    #
    # Both ke and kd remain in the actuation kernel at full strength:
    #   tau = ke*(q_target-q) + kd*(qd_target-qd) + joint_f
    kd_scale = 1.0

    # =========================================================================
    # PRISMATIC joint (type 0): 1 DOF - translation along axis
    # NOTE: joint_target_pos is indexed by qd (velocity DOF), not q (position DOF)
    # =========================================================================
    if jtype == 0:
        axis = joint_axis[qd_start]
        axis_p = wp.quat_rotate(q_p, axis)

        # Linear position error along axis
        x_err = wp.transform_get_translation(X_wj_c) - wp.transform_get_translation(X_wj_p)
        q_meas = wp.dot(x_err, axis_p)

        # Relative velocity along axis
        v_rel_lin = wp.spatial_top(v_c) - wp.spatial_top(v_p)
        v_rel_ang = wp.spatial_bottom(v_c) - wp.spatial_bottom(v_p)
        v_rel_lin_corrected = v_rel_lin - wp.cross(v_rel_ang, r_c) + wp.cross(v_rel_ang, r_p)
        qd_meas = wp.dot(v_rel_lin_corrected, axis_p)

        # PD control (NOTE: target arrays are indexed by qd, not q)
        q_target = joint_target_pos[qd_start]
        qd_target = joint_target_vel[qd_start]
        ke = joint_target_ke[qd_start]
        kd = joint_target_kd[qd_start] * kd_scale

        tau = ke * (q_target - q_meas) + kd * (qd_target - qd_meas) + joint_f[qd_start]

        # Clamp to effort limit
        if has_effort_limits == 1:
            eff_lim = joint_effort_limit[qd_start]
            if eff_lim > 0.0:
                tau = wp.clamp(tau, -eff_lim, eff_lim)

        # Apply force along axis
        f_lin = tau * axis_p
        f_act_p = wp.spatial_vector(-f_lin, -wp.cross(r_p, f_lin))
        f_act_c = wp.spatial_vector(f_lin, wp.cross(r_c, f_lin))

    # =========================================================================
    # REVOLUTE joint (type 1): 1 DOF - rotation about axis
    # Uses geometric arctan2 formula from reference implementation for proper
    # angle computation relative to initial configuration
    # Reference: python_DVI/joints.py lines 283-288
    #
    # IMPORTANT: q_meas is computed using arctan2, which gives values in [-π, π].
    # Therefore target_pos should also be within [-π, π] for correct behavior.
    # Values outside this range will be wrapped to [-π, π].
    # =========================================================================
    elif jtype == 1:
        axis = joint_axis[qd_start]
        axis_p = wp.quat_rotate(q_p, axis)

        # Angular position using geometric formula:
        # a = orthogonal vector to axis (from parent frame), in world
        # b = orthogonal vector to axis (from child frame), in world
        # n = rotation axis in world (from child frame)
        # angle = arctan2(dot(n, cross(a, b)), dot(a, b))
        # This gives the angle as deviation from initial configuration (q=0 at initial)
        # Result is in [-π, π]
        q_meas = compute_revolute_angle(q_p, q_c, axis)

        # Angular velocity about axis
        v_rel_ang = wp.spatial_bottom(v_c) - wp.spatial_bottom(v_p)
        qd_meas = wp.dot(v_rel_ang, axis_p)

        # PD control: error = target - measured
        # Wrap target to [-π, π] since q_meas is in that range
        q_target_raw = joint_target_pos[qd_start]
        q_target = wrap_angle_to_pi(q_target_raw)
        qd_target = joint_target_vel[qd_start]
        ke = joint_target_ke[qd_start]
        kd = joint_target_kd[qd_start] * kd_scale

        # Compute error directly (both values now in [-π, π])
        # Then wrap to handle the discontinuity at ±π
        angle_error = compute_angle_error(q_target, q_meas)
        tau = ke * angle_error + kd * (qd_target - qd_meas) + joint_f[qd_start]

        # Clamp to effort limit (before joint limit penalty — limits are separate)
        if has_effort_limits == 1:
            eff_lim = joint_effort_limit[qd_start]
            if eff_lim > 0.0:
                tau = wp.clamp(tau, -eff_lim, eff_lim)

        # Joint limit penalty: spring-damper that activates only when violated
        if has_joint_limits == 1:
            lim_lo = joint_limit_lower[qd_start]
            lim_hi = joint_limit_upper[qd_start]
            lim_ke = joint_limit_ke[qd_start]
            lim_kd = joint_limit_kd[qd_start]
            if lim_ke > 0.0 and lim_lo < lim_hi:
                if q_meas < lim_lo:
                    tau = tau + lim_ke * (lim_lo - q_meas) + lim_kd * (0.0 - qd_meas)
                elif q_meas > lim_hi:
                    tau = tau + lim_ke * (lim_hi - q_meas) + lim_kd * (0.0 - qd_meas)

        # Apply torque about axis
        f_tau = tau * axis_p
        f_act_p = wp.spatial_vector(wp.vec3(0.0, 0.0, 0.0), -f_tau)
        f_act_c = wp.spatial_vector(wp.vec3(0.0, 0.0, 0.0), f_tau)

    # =========================================================================
    # BALL joint (type 2): 3 DOF - free rotation (spherical)
    # Uses rotation vector representation for continuous angle handling
    # =========================================================================
    elif jtype == 2:
        # Convert relative quaternion to rotation vector (axis * angle)
        # This representation is continuous and avoids gimbal lock issues
        # The rotation vector components give the rotation about each axis

        # Get angle from quaternion (w = cos(angle/2))
        # Clamp to avoid numerical issues near identity
        w_clamped = wp.clamp(q_pc[3], -1.0, 1.0)
        half_angle = wp.acos(w_clamped)
        angle = 2.0 * half_angle

        # Compute sin(half_angle) for normalization
        sin_half = wp.sin(half_angle)

        # Angular velocity in world frame
        v_rel_ang = wp.spatial_bottom(v_c) - wp.spatial_bottom(v_p)

        # For small angles, use first-order approximation to avoid division by zero
        # For angle → 0: axis * angle ≈ 2 * (qx, qy, qz)
        # For larger angles: axis * angle = (qx, qy, qz) / sin(angle/2) * angle

        # Rotation vector = axis * angle (this is the scaled rotation axis)
        rot_vec = wp.vec3(0.0, 0.0, 0.0)
        if sin_half > 1.0e-6:
            # Normal case: extract rotation vector
            scale = angle / sin_half
            rot_vec = wp.vec3(q_pc[0] * scale, q_pc[1] * scale, q_pc[2] * scale)
        else:
            # Near identity: use small angle approximation
            rot_vec = wp.vec3(2.0 * q_pc[0], 2.0 * q_pc[1], 2.0 * q_pc[2])

        # DOF 0 (X axis) - wrap target to [-π, π]
        q_meas_0 = rot_vec[0]
        q_target_0 = wrap_angle_to_pi(joint_target_pos[qd_start + 0])
        qd_target_0 = joint_target_vel[qd_start + 0]
        ke_0 = joint_target_ke[qd_start + 0]
        kd_0 = joint_target_kd[qd_start + 0] * kd_scale
        angle_error_0 = compute_angle_error(q_target_0, q_meas_0)
        tau_0 = ke_0 * angle_error_0 + kd_0 * (qd_target_0 - v_rel_ang[0]) + joint_f[qd_start + 0]
        if has_effort_limits == 1:
            eff_lim_0 = joint_effort_limit[qd_start + 0]
            if eff_lim_0 > 0.0:
                tau_0 = wp.clamp(tau_0, -eff_lim_0, eff_lim_0)
        if has_joint_limits == 1:
            lim_lo_0 = joint_limit_lower[qd_start + 0]
            lim_hi_0 = joint_limit_upper[qd_start + 0]
            lim_ke_0 = joint_limit_ke[qd_start + 0]
            lim_kd_0 = joint_limit_kd[qd_start + 0]
            if lim_ke_0 > 0.0 and lim_lo_0 < lim_hi_0:
                if q_meas_0 < lim_lo_0:
                    tau_0 += lim_ke_0 * (lim_lo_0 - q_meas_0) + lim_kd_0 * (0.0 - v_rel_ang[0])
                elif q_meas_0 > lim_hi_0:
                    tau_0 += lim_ke_0 * (lim_hi_0 - q_meas_0) + lim_kd_0 * (0.0 - v_rel_ang[0])
        f_tau_0 = tau_0 * wp.vec3(1.0, 0.0, 0.0)
        f_act_p = f_act_p - wp.spatial_vector(wp.vec3(0.0, 0.0, 0.0), f_tau_0)
        f_act_c = f_act_c + wp.spatial_vector(wp.vec3(0.0, 0.0, 0.0), f_tau_0)

        # DOF 1 (Y axis) - wrap target to [-π, π]
        q_meas_1 = rot_vec[1]
        q_target_1 = wrap_angle_to_pi(joint_target_pos[qd_start + 1])
        qd_target_1 = joint_target_vel[qd_start + 1]
        ke_1 = joint_target_ke[qd_start + 1]
        kd_1 = joint_target_kd[qd_start + 1] * kd_scale
        angle_error_1 = compute_angle_error(q_target_1, q_meas_1)
        tau_1 = ke_1 * angle_error_1 + kd_1 * (qd_target_1 - v_rel_ang[1]) + joint_f[qd_start + 1]
        if has_effort_limits == 1:
            eff_lim_1 = joint_effort_limit[qd_start + 1]
            if eff_lim_1 > 0.0:
                tau_1 = wp.clamp(tau_1, -eff_lim_1, eff_lim_1)
        if has_joint_limits == 1:
            lim_lo_1 = joint_limit_lower[qd_start + 1]
            lim_hi_1 = joint_limit_upper[qd_start + 1]
            lim_ke_1 = joint_limit_ke[qd_start + 1]
            lim_kd_1 = joint_limit_kd[qd_start + 1]
            if lim_ke_1 > 0.0 and lim_lo_1 < lim_hi_1:
                if q_meas_1 < lim_lo_1:
                    tau_1 += lim_ke_1 * (lim_lo_1 - q_meas_1) + lim_kd_1 * (0.0 - v_rel_ang[1])
                elif q_meas_1 > lim_hi_1:
                    tau_1 += lim_ke_1 * (lim_hi_1 - q_meas_1) + lim_kd_1 * (0.0 - v_rel_ang[1])
        f_tau_1 = tau_1 * wp.vec3(0.0, 1.0, 0.0)
        f_act_p = f_act_p - wp.spatial_vector(wp.vec3(0.0, 0.0, 0.0), f_tau_1)
        f_act_c = f_act_c + wp.spatial_vector(wp.vec3(0.0, 0.0, 0.0), f_tau_1)

        # DOF 2 (Z axis) - wrap target to [-π, π]
        q_meas_2 = rot_vec[2]
        q_target_2 = wrap_angle_to_pi(joint_target_pos[qd_start + 2])
        qd_target_2 = joint_target_vel[qd_start + 2]
        ke_2 = joint_target_ke[qd_start + 2]
        kd_2 = joint_target_kd[qd_start + 2] * kd_scale
        angle_error_2 = compute_angle_error(q_target_2, q_meas_2)
        tau_2 = ke_2 * angle_error_2 + kd_2 * (qd_target_2 - v_rel_ang[2]) + joint_f[qd_start + 2]
        if has_effort_limits == 1:
            eff_lim_2 = joint_effort_limit[qd_start + 2]
            if eff_lim_2 > 0.0:
                tau_2 = wp.clamp(tau_2, -eff_lim_2, eff_lim_2)
        if has_joint_limits == 1:
            lim_lo_2 = joint_limit_lower[qd_start + 2]
            lim_hi_2 = joint_limit_upper[qd_start + 2]
            lim_ke_2 = joint_limit_ke[qd_start + 2]
            lim_kd_2 = joint_limit_kd[qd_start + 2]
            if lim_ke_2 > 0.0 and lim_lo_2 < lim_hi_2:
                if q_meas_2 < lim_lo_2:
                    tau_2 += lim_ke_2 * (lim_lo_2 - q_meas_2) + lim_kd_2 * (0.0 - v_rel_ang[2])
                elif q_meas_2 > lim_hi_2:
                    tau_2 += lim_ke_2 * (lim_hi_2 - q_meas_2) + lim_kd_2 * (0.0 - v_rel_ang[2])
        f_tau_2 = tau_2 * wp.vec3(0.0, 0.0, 1.0)
        f_act_p = f_act_p - wp.spatial_vector(wp.vec3(0.0, 0.0, 0.0), f_tau_2)
        f_act_c = f_act_c + wp.spatial_vector(wp.vec3(0.0, 0.0, 0.0), f_tau_2)

    # =========================================================================
    # D6 joint (type 6): Variable DOFs - linear and angular axes
    # Used for cylindrical joints (1 linear + 1 angular), etc.
    # DOFs are stored as: first all linear DOFs, then all angular DOFs
    # Max 3 linear + 3 angular DOFs supported (unrolled loops for Warp compatibility)
    # =========================================================================
    elif jtype == 6:
        n_linear = joint_dof_dim[tid, 0]
        n_angular = joint_dof_dim[tid, 1]

        # Position and velocity errors
        x_err = wp.transform_get_translation(X_wj_c) - wp.transform_get_translation(X_wj_p)
        v_rel_lin = wp.spatial_top(v_c) - wp.spatial_top(v_p)
        v_rel_ang = wp.spatial_bottom(v_c) - wp.spatial_bottom(v_p)

        # Linear DOFs (translation along axes) - unrolled for up to 3 DOFs
        # NOTE: joint_target_pos is indexed by qd (velocity DOF), not q (position DOF)
        if n_linear >= 1:
            i = 0
            axis = joint_axis[qd_start + i]
            axis_p = wp.quat_rotate(q_p, axis)
            q_meas = wp.dot(x_err, axis_p)
            qd_meas = wp.dot(v_rel_lin, axis_p)
            q_target = joint_target_pos[qd_start + i]
            qd_target = joint_target_vel[qd_start + i]
            ke = joint_target_ke[qd_start + i]
            kd = joint_target_kd[qd_start + i] * kd_scale
            tau = ke * (q_target - q_meas) + kd * (qd_target - qd_meas) + joint_f[qd_start + i]
            if has_effort_limits == 1:
                eff_lim = joint_effort_limit[qd_start + i]
                if eff_lim > 0.0:
                    tau = wp.clamp(tau, -eff_lim, eff_lim)
            f_lin = tau * axis_p
            f_act_p = f_act_p - wp.spatial_vector(f_lin, wp.cross(r_p, f_lin))
            f_act_c = f_act_c + wp.spatial_vector(f_lin, -wp.cross(r_c, f_lin))

        if n_linear >= 2:
            i = 1
            axis = joint_axis[qd_start + i]
            axis_p = wp.quat_rotate(q_p, axis)
            q_meas = wp.dot(x_err, axis_p)
            qd_meas = wp.dot(v_rel_lin, axis_p)
            q_target = joint_target_pos[qd_start + i]
            qd_target = joint_target_vel[qd_start + i]
            ke = joint_target_ke[qd_start + i]
            kd = joint_target_kd[qd_start + i] * kd_scale
            tau = ke * (q_target - q_meas) + kd * (qd_target - qd_meas) + joint_f[qd_start + i]
            if has_effort_limits == 1:
                eff_lim = joint_effort_limit[qd_start + i]
                if eff_lim > 0.0:
                    tau = wp.clamp(tau, -eff_lim, eff_lim)
            f_lin = tau * axis_p
            f_act_p = f_act_p - wp.spatial_vector(f_lin, wp.cross(r_p, f_lin))
            f_act_c = f_act_c + wp.spatial_vector(f_lin, -wp.cross(r_c, f_lin))

        if n_linear >= 3:
            i = 2
            axis = joint_axis[qd_start + i]
            axis_p = wp.quat_rotate(q_p, axis)
            q_meas = wp.dot(x_err, axis_p)
            qd_meas = wp.dot(v_rel_lin, axis_p)
            q_target = joint_target_pos[qd_start + i]
            qd_target = joint_target_vel[qd_start + i]
            ke = joint_target_ke[qd_start + i]
            kd = joint_target_kd[qd_start + i] * kd_scale
            tau = ke * (q_target - q_meas) + kd * (qd_target - qd_meas) + joint_f[qd_start + i]
            if has_effort_limits == 1:
                eff_lim = joint_effort_limit[qd_start + i]
                if eff_lim > 0.0:
                    tau = wp.clamp(tau, -eff_lim, eff_lim)
            f_lin = tau * axis_p
            f_act_p = f_act_p - wp.spatial_vector(f_lin, wp.cross(r_p, f_lin))
            f_act_c = f_act_c + wp.spatial_vector(f_lin, -wp.cross(r_c, f_lin))

        # Angular DOFs (rotation about axes) - unrolled for up to 3 DOFs
        # Uses geometric arctan2 formula for proper angle computation
        # Reference: python_DVI/joints.py lines 280-282 for cylindrical joints
        # Angular DOFs - wrap targets to [-π, π] since q_meas is in that range
        if n_angular >= 1:
            i = 0
            dof_idx = qd_start + n_linear + i
            axis = joint_axis[dof_idx]
            axis_p = wp.quat_rotate(q_p, axis)
            q_meas = compute_revolute_angle(q_p, q_c, axis)
            qd_meas = wp.dot(v_rel_ang, axis_p)
            q_target = wrap_angle_to_pi(joint_target_pos[dof_idx])
            qd_target = joint_target_vel[dof_idx]
            ke = joint_target_ke[dof_idx]
            kd = joint_target_kd[dof_idx] * kd_scale
            angle_error = compute_angle_error(q_target, q_meas)
            tau = ke * angle_error + kd * (qd_target - qd_meas) + joint_f[dof_idx]
            if has_effort_limits == 1:
                eff_lim = joint_effort_limit[dof_idx]
                if eff_lim > 0.0:
                    tau = wp.clamp(tau, -eff_lim, eff_lim)
            if has_joint_limits == 1:
                lim_lo = joint_limit_lower[dof_idx]
                lim_hi = joint_limit_upper[dof_idx]
                lim_ke = joint_limit_ke[dof_idx]
                lim_kd = joint_limit_kd[dof_idx]
                if lim_ke > 0.0 and lim_lo < lim_hi:
                    if q_meas < lim_lo:
                        tau += lim_ke * (lim_lo - q_meas) + lim_kd * (0.0 - qd_meas)
                    elif q_meas > lim_hi:
                        tau += lim_ke * (lim_hi - q_meas) + lim_kd * (0.0 - qd_meas)
            f_tau = tau * axis_p
            f_act_p = f_act_p - wp.spatial_vector(wp.vec3(0.0, 0.0, 0.0), f_tau)
            f_act_c = f_act_c + wp.spatial_vector(wp.vec3(0.0, 0.0, 0.0), f_tau)

        if n_angular >= 2:
            i = 1
            dof_idx = qd_start + n_linear + i
            axis = joint_axis[dof_idx]
            axis_p = wp.quat_rotate(q_p, axis)
            q_meas = compute_revolute_angle(q_p, q_c, axis)
            qd_meas = wp.dot(v_rel_ang, axis_p)
            q_target = wrap_angle_to_pi(joint_target_pos[dof_idx])
            qd_target = joint_target_vel[dof_idx]
            ke = joint_target_ke[dof_idx]
            kd = joint_target_kd[dof_idx] * kd_scale
            angle_error = compute_angle_error(q_target, q_meas)
            tau = ke * angle_error + kd * (qd_target - qd_meas) + joint_f[dof_idx]
            if has_effort_limits == 1:
                eff_lim = joint_effort_limit[dof_idx]
                if eff_lim > 0.0:
                    tau = wp.clamp(tau, -eff_lim, eff_lim)
            if has_joint_limits == 1:
                lim_lo = joint_limit_lower[dof_idx]
                lim_hi = joint_limit_upper[dof_idx]
                lim_ke = joint_limit_ke[dof_idx]
                lim_kd = joint_limit_kd[dof_idx]
                if lim_ke > 0.0 and lim_lo < lim_hi:
                    if q_meas < lim_lo:
                        tau += lim_ke * (lim_lo - q_meas) + lim_kd * (0.0 - qd_meas)
                    elif q_meas > lim_hi:
                        tau += lim_ke * (lim_hi - q_meas) + lim_kd * (0.0 - qd_meas)
            f_tau = tau * axis_p
            f_act_p = f_act_p - wp.spatial_vector(wp.vec3(0.0, 0.0, 0.0), f_tau)
            f_act_c = f_act_c + wp.spatial_vector(wp.vec3(0.0, 0.0, 0.0), f_tau)

        if n_angular >= 3:
            i = 2
            dof_idx = qd_start + n_linear + i
            axis = joint_axis[dof_idx]
            axis_p = wp.quat_rotate(q_p, axis)
            q_meas = compute_revolute_angle(q_p, q_c, axis)
            qd_meas = wp.dot(v_rel_ang, axis_p)
            q_target = wrap_angle_to_pi(joint_target_pos[dof_idx])
            qd_target = joint_target_vel[dof_idx]
            ke = joint_target_ke[dof_idx]
            kd = joint_target_kd[dof_idx] * kd_scale
            angle_error = compute_angle_error(q_target, q_meas)
            tau = ke * angle_error + kd * (qd_target - qd_meas) + joint_f[dof_idx]
            if has_effort_limits == 1:
                eff_lim = joint_effort_limit[dof_idx]
                if eff_lim > 0.0:
                    tau = wp.clamp(tau, -eff_lim, eff_lim)
            if has_joint_limits == 1:
                lim_lo = joint_limit_lower[dof_idx]
                lim_hi = joint_limit_upper[dof_idx]
                lim_ke = joint_limit_ke[dof_idx]
                lim_kd = joint_limit_kd[dof_idx]
                if lim_ke > 0.0 and lim_lo < lim_hi:
                    if q_meas < lim_lo:
                        tau += lim_ke * (lim_lo - q_meas) + lim_kd * (0.0 - qd_meas)
                    elif q_meas > lim_hi:
                        tau += lim_ke * (lim_hi - q_meas) + lim_kd * (0.0 - qd_meas)
            f_tau = tau * axis_p
            f_act_p = f_act_p - wp.spatial_vector(wp.vec3(0.0, 0.0, 0.0), f_tau)
            f_act_c = f_act_c + wp.spatial_vector(wp.vec3(0.0, 0.0, 0.0), f_tau)

    # Apply actuation forces to bodies (accumulated atomically)
    if id_p >= 0:
        wp.atomic_add(body_f_act, id_p, f_act_p)
    wp.atomic_add(body_f_act, id_c, f_act_c)


@wp.func
def compute_limit_penalty(
    q_meas: float,
    qd_meas: float,
    lo: float,
    hi: float,
    ke: float,
    kd: float,
) -> float:
    """Compute a clamped spring-damper penalty for joint limit violation.

    Returns 0 when inside limits. When violated, the restoring torque/force is:
        tau = -ke * violation - kd * velocity
    clamped to ±ke * (hi - lo) to prevent blow-up for large violations.
    """
    if ke <= 0.0 or lo >= hi:
        return 0.0

    violation = 0.0
    if q_meas < lo:
        violation = q_meas - lo  # negative
    elif q_meas > hi:
        violation = q_meas - hi  # positive
    else:
        return 0.0

    tau = -ke * violation - kd * qd_meas
    # Clamp to prevent explosive forces for large violations
    max_tau = ke * (hi - lo)
    tau = wp.clamp(tau, -max_tau, max_tau)
    return tau


@wp.kernel
def compute_joint_limit_forces(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    joint_type: wp.array(dtype=int),
    joint_enabled: wp.array(dtype=bool),
    joint_parent: wp.array(dtype=int),
    joint_child: wp.array(dtype=int),
    joint_X_p: wp.array(dtype=wp.transform),
    joint_X_c: wp.array(dtype=wp.transform),
    joint_axis: wp.array(dtype=wp.vec3),
    joint_q_start: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_dof_dim: wp.array2d(dtype=int),
    joint_limit_lower: wp.array(dtype=float),
    joint_limit_upper: wp.array(dtype=float),
    joint_limit_ke: wp.array(dtype=float),
    joint_limit_kd: wp.array(dtype=float),
    # outputs
    body_f_out: wp.array(dtype=wp.spatial_vector),
):
    """Apply penalty forces for joint limit violations.

    For each DOF with finite limits and positive limit_ke, checks if the current
    joint position exceeds [limit_lower, limit_upper] and applies a spring-damper
    restoring force:
        tau = -ke * violation - kd * velocity   (when outside limits)

    The force magnitude is clamped to ``ke * (limit_upper - limit_lower)`` to
    prevent explosive instability when the joint is far outside its range
    (e.g. initial configuration or after a missed collision).

    This is analogous to MuJoCo's joint limit enforcement via solref/solimp.

    Handles: PRISMATIC (linear), REVOLUTE (angular), BALL (3 angular),
             D6 (mixed linear + angular).
    """
    tid = wp.tid()

    if not joint_enabled[tid]:
        return

    jtype = joint_type[tid]

    # Fixed (3) and FREE (4) joints have no limits to enforce
    if jtype == 3 or jtype == 4:
        return

    id_p = joint_parent[tid]
    id_c = joint_child[tid]

    X_wp = body_q[id_c] if id_p < 0 else body_q[id_p]
    X_wc = body_q[id_c]

    v_p = wp.spatial_vector() if id_p < 0 else body_qd[id_p]
    v_c = body_qd[id_c]

    com_p = wp.vec3(0.0, 0.0, 0.0) if id_p < 0 else body_com[id_p]
    com_c = body_com[id_c]

    X_pj = joint_X_p[tid]
    X_cj = joint_X_c[tid]

    X_wj_p = wp.transform_multiply(X_wp, X_pj)
    X_wj_c = wp.transform_multiply(X_wc, X_cj)

    r_p = wp.transform_get_translation(X_wj_p) - (
        wp.transform_get_translation(X_wp) + wp.quat_rotate(wp.transform_get_rotation(X_wp), com_p)
    )
    r_c = wp.transform_get_translation(X_wj_c) - (
        wp.transform_get_translation(X_wc) + wp.quat_rotate(wp.transform_get_rotation(X_wc), com_c)
    )

    q_p = wp.transform_get_rotation(X_wj_p)
    q_c = wp.transform_get_rotation(X_wj_c)

    qd_start = joint_qd_start[tid]

    f_act_p = wp.spatial_vector()
    f_act_c = wp.spatial_vector()

    # PRISMATIC (type 0): 1 linear DOF
    if jtype == 0:
        axis = joint_axis[qd_start]
        axis_p = wp.quat_rotate(q_p, axis)
        x_err = wp.transform_get_translation(X_wj_c) - wp.transform_get_translation(X_wj_p)
        q_meas = wp.dot(x_err, axis_p)
        v_rel_lin = wp.spatial_top(v_c) - wp.spatial_top(v_p)
        qd_meas = wp.dot(v_rel_lin, axis_p)

        tau = compute_limit_penalty(
            q_meas, qd_meas,
            joint_limit_lower[qd_start], joint_limit_upper[qd_start],
            joint_limit_ke[qd_start], joint_limit_kd[qd_start],
        )
        if tau != 0.0:
            f_lin = tau * axis_p
            f_act_p = wp.spatial_vector(-f_lin, -wp.cross(r_p, f_lin))
            f_act_c = wp.spatial_vector(f_lin, wp.cross(r_c, f_lin))

    # REVOLUTE (type 1): 1 angular DOF
    elif jtype == 1:
        axis = joint_axis[qd_start]
        axis_p = wp.quat_rotate(q_p, axis)
        q_meas = compute_revolute_angle(q_p, q_c, axis)
        v_rel_ang = wp.spatial_bottom(v_c) - wp.spatial_bottom(v_p)
        qd_meas = wp.dot(v_rel_ang, axis_p)

        tau = compute_limit_penalty(
            q_meas, qd_meas,
            joint_limit_lower[qd_start], joint_limit_upper[qd_start],
            joint_limit_ke[qd_start], joint_limit_kd[qd_start],
        )
        if tau != 0.0:
            f_tau = tau * axis_p
            f_act_p = wp.spatial_vector(wp.vec3(0.0, 0.0, 0.0), -f_tau)
            f_act_c = wp.spatial_vector(wp.vec3(0.0, 0.0, 0.0), f_tau)

    # BALL (type 2): 3 angular DOFs
    elif jtype == 2:
        q_pc = wp.mul(wp.quat_inverse(q_p), q_c)
        w_clamped = wp.clamp(q_pc[3], -1.0, 1.0)
        half_angle = wp.acos(w_clamped)
        angle = 2.0 * half_angle
        sin_half = wp.sin(half_angle)

        rot_vec = wp.vec3(0.0, 0.0, 0.0)
        if sin_half > 1.0e-6:
            scale = angle / sin_half
            rot_vec = wp.vec3(q_pc[0] * scale, q_pc[1] * scale, q_pc[2] * scale)
        else:
            rot_vec = wp.vec3(2.0 * q_pc[0], 2.0 * q_pc[1], 2.0 * q_pc[2])

        v_rel_ang = wp.spatial_bottom(v_c) - wp.spatial_bottom(v_p)

        for i in range(3):
            tau = compute_limit_penalty(
                rot_vec[i], v_rel_ang[i],
                joint_limit_lower[qd_start + i], joint_limit_upper[qd_start + i],
                joint_limit_ke[qd_start + i], joint_limit_kd[qd_start + i],
            )
            if tau != 0.0:
                f_tau = wp.vec3(0.0, 0.0, 0.0)
                if i == 0:
                    f_tau = tau * wp.vec3(1.0, 0.0, 0.0)
                elif i == 1:
                    f_tau = tau * wp.vec3(0.0, 1.0, 0.0)
                else:
                    f_tau = tau * wp.vec3(0.0, 0.0, 1.0)
                f_act_p = f_act_p - wp.spatial_vector(wp.vec3(0.0, 0.0, 0.0), f_tau)
                f_act_c = f_act_c + wp.spatial_vector(wp.vec3(0.0, 0.0, 0.0), f_tau)

    # D6 (type 6): Variable linear + angular DOFs
    elif jtype == 6:
        n_linear = joint_dof_dim[tid, 0]
        n_angular = joint_dof_dim[tid, 1]

        x_err = wp.transform_get_translation(X_wj_c) - wp.transform_get_translation(X_wj_p)
        v_rel_lin = wp.spatial_top(v_c) - wp.spatial_top(v_p)
        v_rel_ang = wp.spatial_bottom(v_c) - wp.spatial_bottom(v_p)

        # Linear DOFs
        for i in range(3):
            if i < n_linear:
                axis = joint_axis[qd_start + i]
                axis_p = wp.quat_rotate(q_p, axis)
                q_meas = wp.dot(x_err, axis_p)
                qd_meas = wp.dot(v_rel_lin, axis_p)
                tau = compute_limit_penalty(
                    q_meas, qd_meas,
                    joint_limit_lower[qd_start + i], joint_limit_upper[qd_start + i],
                    joint_limit_ke[qd_start + i], joint_limit_kd[qd_start + i],
                )
                if tau != 0.0:
                    f_lin = tau * axis_p
                    f_act_p = f_act_p - wp.spatial_vector(f_lin, wp.cross(r_p, f_lin))
                    f_act_c = f_act_c + wp.spatial_vector(f_lin, -wp.cross(r_c, f_lin))

        # Angular DOFs
        for i in range(3):
            if i < n_angular:
                dof_idx = qd_start + n_linear + i
                axis = joint_axis[dof_idx]
                axis_p = wp.quat_rotate(q_p, axis)
                q_meas = compute_revolute_angle(q_p, q_c, axis)
                qd_meas = wp.dot(v_rel_ang, axis_p)
                tau = compute_limit_penalty(
                    q_meas, qd_meas,
                    joint_limit_lower[dof_idx], joint_limit_upper[dof_idx],
                    joint_limit_ke[dof_idx], joint_limit_kd[dof_idx],
                )
                if tau != 0.0:
                    f_tau = tau * axis_p
                    f_act_p = f_act_p - wp.spatial_vector(wp.vec3(0.0, 0.0, 0.0), f_tau)
                    f_act_c = f_act_c + wp.spatial_vector(wp.vec3(0.0, 0.0, 0.0), f_tau)

    # Apply limit forces to bodies
    if id_p >= 0:
        wp.atomic_add(body_f_out, id_p, f_act_p)
    wp.atomic_add(body_f_out, id_c, f_act_c)


@wp.kernel
def clamp_joint_limit_velocities(
    joint_type: wp.array(dtype=int),
    joint_enabled: wp.array(dtype=bool),
    joint_parent: wp.array(dtype=int),
    joint_child: wp.array(dtype=int),
    joint_X_p: wp.array(dtype=wp.transform),
    joint_X_c: wp.array(dtype=wp.transform),
    joint_axis: wp.array(dtype=wp.vec3),
    joint_qd_start: wp.array(dtype=int),
    joint_dof_dim: wp.array2d(dtype=int),
    joint_limit_lower: wp.array(dtype=float),
    joint_limit_upper: wp.array(dtype=float),
    joint_limit_ke: wp.array(dtype=float),
    dt: float,
    # in/out (both position and velocity may be corrected)
    body_q: wp.array(dtype=wp.transform),
    body_qd_out: wp.array(dtype=wp.spatial_vector),
):
    """Clamp body velocities to prevent joint angles from exceeding limits.

    After integration, if q + dt * qd would exceed a joint limit, the velocity
    component along that joint axis is reduced so the angle stays at the limit.
    This is a velocity-level position clamp, analogous to XPBD's position-level
    limit clamping but applied post-integration.

    Only processes DOFs where limit_ke > 0 and limit_lower < limit_upper.
    """
    tid = wp.tid()

    if not joint_enabled[tid]:
        return

    jtype = joint_type[tid]
    if jtype == 3 or jtype == 4:
        return

    id_p = joint_parent[tid]
    id_c = joint_child[tid]

    X_wp = body_q[id_c] if id_p < 0 else body_q[id_p]
    X_wc = body_q[id_c]

    X_pj = joint_X_p[tid]
    X_cj = joint_X_c[tid]
    X_wj_p = wp.transform_multiply(X_wp, X_pj)
    X_wj_c = wp.transform_multiply(X_wc, X_cj)
    q_p = wp.transform_get_rotation(X_wj_p)
    q_c = wp.transform_get_rotation(X_wj_c)

    qd_start = joint_qd_start[tid]

    v_p = wp.spatial_vector() if id_p < 0 else body_qd_out[id_p]
    v_c = body_qd_out[id_c]

    # REVOLUTE (type 1): 1 angular DOF
    if jtype == 1:
        lo = joint_limit_lower[qd_start]
        hi = joint_limit_upper[qd_start]
        ke = joint_limit_ke[qd_start]
        if ke > 0.0 and lo < hi:
            axis = joint_axis[qd_start]
            axis_p = wp.quat_rotate(q_p, axis)
            q_meas = compute_revolute_angle(q_p, q_c, axis)

            v_rel_ang = wp.spatial_bottom(v_c) - wp.spatial_bottom(v_p)
            qd_meas = wp.dot(v_rel_ang, axis_p)

            if q_meas < lo:
                # Already past lower limit: project position back and
                # zero ALL velocity along joint axis (prevents ratcheting
                # where the agent pushes against the rigid limit wall).
                delta_angle = lo - q_meas
                correction_quat = wp.quat_from_axis_angle(axis_p, delta_angle)
                q_c_body = wp.transform_get_rotation(body_q[id_c])
                body_q[id_c] = wp.transform(wp.transform_get_translation(body_q[id_c]),
                                            wp.mul(correction_quat, q_c_body))
                delta_qd = -qd_meas * axis_p
                body_qd_out[id_c] = v_c + wp.spatial_vector(wp.vec3(0.0, 0.0, 0.0), delta_qd)
            elif q_meas > hi:
                # Already past upper limit: same treatment.
                delta_angle = hi - q_meas
                correction_quat = wp.quat_from_axis_angle(axis_p, delta_angle)
                q_c_body = wp.transform_get_rotation(body_q[id_c])
                body_q[id_c] = wp.transform(wp.transform_get_translation(body_q[id_c]),
                                            wp.mul(correction_quat, q_c_body))
                delta_qd = -qd_meas * axis_p
                body_qd_out[id_c] = v_c + wp.spatial_vector(wp.vec3(0.0, 0.0, 0.0), delta_qd)
            else:
                # Within limits: check if next step would exceed.
                # Clamp velocity so that q + dt*qd = limit.
                q_next = q_meas + dt * qd_meas
                if q_next < lo:
                    qd_clamp = (lo - q_meas) / dt
                    delta_qd = (qd_clamp - qd_meas) * axis_p
                    body_qd_out[id_c] = v_c + wp.spatial_vector(wp.vec3(0.0, 0.0, 0.0), delta_qd)
                elif q_next > hi:
                    qd_clamp = (hi - q_meas) / dt
                    delta_qd = (qd_clamp - qd_meas) * axis_p
                    body_qd_out[id_c] = v_c + wp.spatial_vector(wp.vec3(0.0, 0.0, 0.0), delta_qd)

    # D6 (type 6): Variable angular DOFs
    elif jtype == 6:
        n_linear = joint_dof_dim[tid, 0]
        n_angular = joint_dof_dim[tid, 1]

        x_err = wp.transform_get_translation(X_wj_c) - wp.transform_get_translation(X_wj_p)
        v_rel_lin = wp.spatial_top(v_c) - wp.spatial_top(v_p)
        v_rel_ang = wp.spatial_bottom(v_c) - wp.spatial_bottom(v_p)

        delta_lin = wp.vec3(0.0, 0.0, 0.0)
        delta_ang = wp.vec3(0.0, 0.0, 0.0)
        correction_quat = wp.quat_identity()
        needs_pos_correction = False

        # Linear DOFs (position correction for linear is translation adjustment)
        for i in range(3):
            if i < n_linear:
                lo = joint_limit_lower[qd_start + i]
                hi = joint_limit_upper[qd_start + i]
                ke_i = joint_limit_ke[qd_start + i]
                if ke_i > 0.0 and lo < hi:
                    axis = joint_axis[qd_start + i]
                    axis_p_i = wp.quat_rotate(q_p, axis)
                    q_meas_i = wp.dot(x_err, axis_p_i)
                    qd_meas_i = wp.dot(v_rel_lin, axis_p_i)
                    if q_meas_i < lo:
                        delta_lin = delta_lin + (lo - q_meas_i) * axis_p_i
                        # Zero ALL velocity along this axis
                        delta_lin = delta_lin + (-qd_meas_i) * axis_p_i
                    elif q_meas_i > hi:
                        delta_lin = delta_lin + (hi - q_meas_i) * axis_p_i
                        delta_lin = delta_lin + (-qd_meas_i) * axis_p_i
                    else:
                        q_next_i = q_meas_i + dt * qd_meas_i
                        if q_next_i < lo:
                            qd_clamp = (lo - q_meas_i) / dt
                            delta_lin = delta_lin + (qd_clamp - qd_meas_i) * axis_p_i
                        elif q_next_i > hi:
                            qd_clamp = (hi - q_meas_i) / dt
                            delta_lin = delta_lin + (qd_clamp - qd_meas_i) * axis_p_i

        # Angular DOFs
        for i in range(3):
            if i < n_angular:
                dof_idx = qd_start + n_linear + i
                lo = joint_limit_lower[dof_idx]
                hi = joint_limit_upper[dof_idx]
                ke_i = joint_limit_ke[dof_idx]
                if ke_i > 0.0 and lo < hi:
                    axis = joint_axis[dof_idx]
                    axis_p_i = wp.quat_rotate(q_p, axis)
                    q_meas_i = compute_revolute_angle(q_p, q_c, axis)
                    qd_meas_i = wp.dot(v_rel_ang, axis_p_i)
                    if q_meas_i < lo:
                        delta_a = lo - q_meas_i
                        correction_quat = wp.mul(wp.quat_from_axis_angle(axis_p_i, delta_a), correction_quat)
                        needs_pos_correction = True
                        # Zero ALL velocity along this axis (prevent ratcheting)
                        delta_ang = delta_ang + (-qd_meas_i) * axis_p_i
                    elif q_meas_i > hi:
                        delta_a = hi - q_meas_i
                        correction_quat = wp.mul(wp.quat_from_axis_angle(axis_p_i, delta_a), correction_quat)
                        needs_pos_correction = True
                        delta_ang = delta_ang + (-qd_meas_i) * axis_p_i
                    else:
                        q_next_i = q_meas_i + dt * qd_meas_i
                        if q_next_i < lo:
                            qd_clamp = (lo - q_meas_i) / dt
                            delta_ang = delta_ang + (qd_clamp - qd_meas_i) * axis_p_i
                        elif q_next_i > hi:
                            qd_clamp = (hi - q_meas_i) / dt
                            delta_ang = delta_ang + (qd_clamp - qd_meas_i) * axis_p_i

        # Apply position and velocity corrections
        if needs_pos_correction:
            q_c_body = wp.transform_get_rotation(body_q[id_c])
            body_q[id_c] = wp.transform(wp.transform_get_translation(body_q[id_c]),
                                        wp.mul(correction_quat, q_c_body))
        if wp.length(delta_lin) > 0.0 or wp.length(delta_ang) > 0.0:
            body_qd_out[id_c] = v_c + wp.spatial_vector(delta_lin, delta_ang)
