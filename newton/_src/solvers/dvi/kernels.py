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
Warp kernels for the DVI (Differential Variational Inequality) solver.

This module contains integration, gravity, inertia, and utility kernels.
Individual constraint kernel modules:
    - contact_kernels.py: Contact constraint and friction cone projection
    - constraint_kernels.py: Joint/bilateral constraint handling
    - actuation_kernels.py: PD control for joint DOFs
"""

import warp as wp


# =============================================================================
# Inertia Frame Utilities
# =============================================================================


@wp.func
def rotate_inv_inertia(rot: wp.quat, inv_I_body: wp.mat33) -> wp.mat33:
    """Rotate a body-frame inverse inertia tensor to world frame.

    Returns ``R · inv_I_body · Rᵀ`` where ``R`` is the rotation matrix
    corresponding to *rot*.
    """
    R = wp.quat_to_matrix(rot)
    return R * inv_I_body * wp.transpose(R)


@wp.kernel
def compute_body_inv_inertia_world(
    body_q: wp.array(dtype=wp.transform),
    body_inv_inertia: wp.array(dtype=wp.mat33),
    body_inv_mass: wp.array(dtype=float),
    # outputs
    body_inv_inertia_world: wp.array(dtype=wp.mat33),
):
    """Pre-compute world-frame inverse inertia for all bodies.

    Writes ``R · I_body^{-1} · Rᵀ`` to *body_inv_inertia_world* for each
    dynamic body, or the zero matrix for fixed bodies (inv_mass == 0).
    Launched once per step so downstream kernels can read the result
    directly instead of repeating the rotation.

    Args:
        body_q: Body transforms [body_count].
        body_inv_inertia: Body-frame inverse inertia [body_count].
        body_inv_mass: Inverse masses [body_count].

    Outputs:
        body_inv_inertia_world: World-frame inverse inertia [body_count].
    """
    tid = wp.tid()
    if body_inv_mass[tid] == 0.0:
        body_inv_inertia_world[tid] = wp.mat33(0.0)
        return
    rot = wp.transform_get_rotation(body_q[tid])
    body_inv_inertia_world[tid] = rotate_inv_inertia(rot, body_inv_inertia[tid])


# =============================================================================
# Gravity Application Kernel
# =============================================================================


@wp.kernel
def apply_gravity_forces(
    body_mass: wp.array(dtype=float),
    body_inv_mass: wp.array(dtype=float),
    gravity: wp.array(dtype=wp.vec3),
    # outputs
    body_f: wp.array(dtype=wp.spatial_vector),
):
    """Apply gravitational force to body force buffer.

    F_gravity = m * g (only for dynamic bodies with inv_mass > 0)

    This should be called before constraint solving so the residual
    computation includes gravitational forces.
    """
    tid = wp.tid()

    inv_m = body_inv_mass[tid]
    if inv_m == 0.0:
        return

    m = body_mass[tid]
    g = gravity[0]

    # Add gravity force to linear component (top of spatial vector)
    f_grav = m * g
    f_old = body_f[tid]
    f_lin = wp.spatial_top(f_old) + f_grav
    tau = wp.spatial_bottom(f_old)

    body_f[tid] = wp.spatial_vector(f_lin, tau)


@wp.kernel
def apply_gyroscopic_forces(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_inertia: wp.array(dtype=wp.mat33),
    body_inv_mass: wp.array(dtype=float),
    # outputs
    body_f: wp.array(dtype=wp.spatial_vector),
):
    """Apply gyroscopic torque to body force buffer.

    The gyroscopic torque accounts for the rate of change of the inertia
    tensor in the world frame. Without this term, angular momentum is not
    conserved for bodies with non-spherical inertia.

    In the body frame, the Euler equation is:
        I_body * omega_dot = tau_ext - omega_body x (I_body * omega_body)

    Equivalently in the world frame:
        tau_gyro = -omega_world x (I_world * omega_world)

    We compute it in body frame for numerical accuracy:
        omega_local = R^T * omega_world
        gyro_local = omega_local x (I_body * omega_local)
        gyro_world = R * gyro_local

    This matches Project DVI's ComputeGyro() and IntLoadResidual_F().
    """
    tid = wp.tid()

    inv_m = body_inv_mass[tid]
    if inv_m == 0.0:
        return

    # Get current orientation and angular velocity
    q = body_q[tid]
    r = wp.transform_get_rotation(q)
    v = body_qd[tid]
    omega_world = wp.spatial_bottom(v)

    # Transform angular velocity to body frame
    omega_local = wp.quat_rotate_inv(r, omega_world)

    # Compute gyroscopic torque in body frame: omega x (I * omega)
    I_body = body_inertia[tid]
    I_omega_local = I_body * omega_local
    gyro_local = wp.cross(omega_local, I_omega_local)

    # Transform back to world frame
    gyro_world = wp.quat_rotate(r, gyro_local)

    # Subtract from torque (gyroscopic term opposes rotation)
    f_old = body_f[tid]
    f_lin = wp.spatial_top(f_old)
    tau = wp.spatial_bottom(f_old) - gyro_world

    body_f[tid] = wp.spatial_vector(f_lin, tau)


# =============================================================================
# Body Integration Kernel (Combined for efficiency)
# =============================================================================


@wp.kernel
def integrate_bodies_euler(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_f: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    body_mass: wp.array(dtype=float),
    body_inertia: wp.array(dtype=wp.mat33),
    body_inv_mass: wp.array(dtype=float),
    body_inv_inertia_world: wp.array(dtype=wp.mat33),
    gravity: wp.array(dtype=wp.vec3),
    angular_damping: float,
    dt: float,
    # outputs
    body_q_new: wp.array(dtype=wp.transform),
    body_qd_new: wp.array(dtype=wp.spatial_vector),
):
    """Semi-implicit Euler integration for rigid bodies.

    Combines velocity and position integration into a single kernel for
    efficiency. This avoids an extra kernel launch and memory traffic.

    **Algorithm:**

    1. Compute acceleration from forces:
           a_lin = f_lin / m + g
           a_ang = I_world^{-1} · τ

    2. Update velocity (semi-implicit Euler):
           v_lin^{n+1} = v_lin^n + dt · a_lin
           v_ang^{n+1} = v_ang^n + dt · a_ang

    3. Apply angular damping:
           v_ang^{n+1} *= (1 - angular_damping · dt)

    4. Update position using new velocity:
           x_com^{n+1} = x_com^n + dt · v_lin^{n+1}

    5. Update orientation using exponential map (world-frame ω):
           q^{n+1} = normalize([0.5·dt·ω, 1] ⊗ q^n)  (left multiplication)

    **Note:** Fixed bodies (inv_mass = 0) are not updated.

    Args:
        body_q: Current body transforms [position, quaternion].
        body_qd: Current body spatial velocities [v_lin, v_ang].
        body_f: Body spatial forces [f_lin, tau].
        body_com: Body center of mass (local coordinates).
        body_mass: Body masses [kg].
        body_inertia: Body inertia tensors (body frame) [kg*m^2].
        body_inv_mass: Inverse masses [1/kg].
        body_inv_inertia_world: World-frame inverse inertia (pre-computed) [1/(kg*m^2)].
        gravity: Gravity acceleration vector (array with 1 element) [m/s^2].
        angular_damping: Angular velocity damping coefficient.
        dt: Time step [s].

    Outputs:
        body_q_new: Updated transforms.
        body_qd_new: Updated velocities.
    """
    tid = wp.tid()

    q = body_q[tid]
    v = body_qd[tid]
    f = body_f[tid]
    inv_m = body_inv_mass[tid]
    com = body_com[tid]

    # Skip fixed bodies (inv_mass = 0)
    if inv_m == 0.0:
        body_q_new[tid] = q
        body_qd_new[tid] = v
        return

    # Get position and orientation
    x0 = wp.transform_get_translation(q)
    r0 = wp.transform_get_rotation(q)

    inv_I_world = body_inv_inertia_world[tid]

    # Compute COM position
    x_com = x0 + wp.quat_rotate(r0, com)

    # Extract velocity components
    v_lin = wp.spatial_top(v)
    v_ang = wp.spatial_bottom(v)

    # Extract force components
    f_lin = wp.spatial_top(f)
    tau = wp.spatial_bottom(f)

    # Semi-implicit Euler: update velocity first
    # v_lin += dt · f_lin/m
    v_lin_new = v_lin + dt * f_lin * inv_m
    # v_ang += dt · I_world^{-1} · τ
    v_ang_new = v_ang + dt * (inv_I_world * tau)
    # Apply angular damping for stability
    v_ang_new = v_ang_new * (1.0 - angular_damping * dt)

    # Update position using new velocity
    x_com_new = x_com + dt * v_lin_new

    # Update orientation using exponential map
    # v_ang_new is in world frame, so we use LEFT multiplication: q_new = dq_world * q_old
    dq = wp.quat(0.5 * dt * v_ang_new[0], 0.5 * dt * v_ang_new[1], 0.5 * dt * v_ang_new[2], 1.0)
    r1 = wp.normalize(dq * r0)

    # Compute origin from COM
    x1 = x_com_new - wp.quat_rotate(r1, com)

    body_q_new[tid] = wp.transform(x1, r1)
    body_qd_new[tid] = wp.spatial_vector(v_lin_new, v_ang_new)


@wp.kernel
def apply_velocity_correction_from_position(
    position_correction: wp.array(dtype=wp.spatial_vector),
    body_inv_mass: wp.array(dtype=float),
    dt: float,
    # outputs
    body_qd: wp.array(dtype=wp.spatial_vector),
):
    """Update velocity to be consistent with position correction.

    After position correction moves bodies by delta_x, the velocity must
    be updated by delta_x/dt to maintain position-velocity consistency.
    This prevents energy injection from position correction.

    Matches the XPBD approach: velocity is implicitly derived from
    position changes.

    Args:
        position_correction: 6D correction per body [delta_lin, delta_ang].
        body_inv_mass: Inverse mass (skip fixed bodies).
        dt: Time step.

    Outputs:
        body_qd: Body velocities (modified in place).
    """
    tid = wp.tid()

    # Skip fixed bodies
    if body_inv_mass[tid] <= 0.0:
        return

    corr = position_correction[tid]
    delta_lin = wp.spatial_top(corr)
    delta_ang = wp.spatial_bottom(corr)

    # Update velocity: v += correction / dt
    v_old = body_qd[tid]
    v_lin = wp.spatial_top(v_old) + delta_lin / dt
    v_ang = wp.spatial_bottom(v_old) + delta_ang / dt

    body_qd[tid] = wp.spatial_vector(v_lin, v_ang)


# =============================================================================
# State Copy Utilities
# =============================================================================


@wp.kernel
def swap_float_arrays(
    a: wp.array(dtype=float),
    count: int,
    # outputs
    b: wp.array(dtype=float),
):
    """Swap two float arrays in place."""
    tid = wp.tid()
    if tid >= count:
        return
    tmp = a[tid]
    a[tid] = b[tid]
    b[tid] = tmp


@wp.kernel
def init_friction_from_materials_kernel(
    contact_count: wp.array(dtype=int),
    contact_shape0: wp.array(dtype=int),
    contact_shape1: wp.array(dtype=int),
    shape_material_mu: wp.array(dtype=float),
    default_mu: float,
    max_contacts: int,
    # outputs
    contact_friction: wp.array(dtype=float),
):
    """Initialize friction coefficients from material properties on GPU."""
    tid = wp.tid()
    if tid >= max_contacts:
        return

    count = contact_count[0]
    if tid >= count:
        contact_friction[tid] = default_mu
        return

    s0 = contact_shape0[tid]
    s1 = contact_shape1[tid]

    mu = 0.0
    n = 0
    if s0 >= 0:
        mu = mu + shape_material_mu[s0]
        n = n + 1
    if s1 >= 0:
        mu = mu + shape_material_mu[s1]
        n = n + 1

    if n > 0:
        contact_friction[tid] = mu / float(n)
    else:
        contact_friction[tid] = default_mu


@wp.kernel
def negate_float_array(
    src: wp.array(dtype=float),
    n: int,
    # outputs
    dst: wp.array(dtype=float),
):
    """Negate a float array: dst = -src."""
    tid = wp.tid()
    if tid >= n:
        return
    dst[tid] = -src[tid]


@wp.kernel
def compute_contact_row_count(
    contact_count: wp.array(dtype=int),
    rows_per_contact: int,
    # outputs
    row_count: wp.array(dtype=int),
):
    """Compute active row count for contacts: row_count = contact_count * rows_per_contact.

    This kernel reads contact_count on GPU to avoid CPU readback during CUDA graph capture.
    """
    row_count[0] = contact_count[0] * rows_per_contact
