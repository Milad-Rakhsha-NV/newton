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
Contact dynamics kernels for the DVI solver.

This module implements shared kernels for frictional contact dynamics using cone
complementarity formulations. Solver-specific kernels are located in their
respective solver modules (sparse_jacobi.py, block_sparse_ldl_solver.py).

The contact problem is formulated as:

    Find lambda in K such that: (N*lambda + p)^T * (mu - lambda) >= 0 for all mu in K

where:
    - lambda is the contact impulse vector [lambda_n, lambda_t1, lambda_t2] per contact
    - N = J*W*J^T is the Delassus operator (contact-space mass matrix)
    - p = J*(v + dt*W*f_ext) is the contact-space predicted velocity
    - K is the friction cone: {lambda : ||lambda_t|| <= mu*lambda_n, lambda_n >= 0}
    - W = M^{-1} is the inverse mass matrix

The solver computes contact impulses that:
    1. Prevent interpenetration (normal impulse lambda_n >= 0)
    2. Satisfy Coulomb friction (||lambda_t|| <= mu*lambda_n)
    3. Achieve complementarity (either separating velocity or zero gap)

Mathematical Formulation
------------------------
The contact Jacobian J maps body velocities to contact-space velocities:
    v_rel = J_a * V_a + J_b * V_b

where V = [v_lin, v_ang]^T is the spatial velocity.

For a contact with normal n and contact point p:
    J_a = [n,    r_a x n   ]  (body A pushes in +n direction)
    J_b = [-n,  -r_b x n   ]  (body B pushes in -n direction)

where r_a, r_b are vectors from body COMs to the contact point.

References
----------
- M. Anitescu, F. A. Potra. "Formulating Dynamic Multi-Rigid-Body Contact
  Problems with Friction as Solvable Linear Complementarity Problems."
  Nonlinear Dynamics, 1997.
- A. Tasora, M. Anitescu. "A matrix-free cone complementarity approach for
  solving large-scale, nonsmooth, rigid body dynamics." Computer Methods
  in Applied Mechanics and Engineering, 2011.
"""

import warp as wp

# =============================================================================
# Friction Cone Projection
# =============================================================================


@wp.func
def project_friction_cone(f: wp.vec3, mu: float) -> wp.vec3:
    """
    Project a contact force onto the friction cone.

    The friction cone is defined as:
        K = {f : ||f_t|| <= mu*f_n, f_n >= 0}

    where f = [f_n, f_t1, f_t2] with f_n the normal component and
    f_t = [f_t1, f_t2] the tangential components.

    The projection minimizes ||f - f_proj||^2 subject to f_proj in K.

    Three cases:
        1. f inside cone: f_proj = f
        2. f below dual cone apex: f_proj = 0  (when ||f_t|| < -f_n/mu)
        3. f outside cone: project onto cone boundary

    For case 3, the projection onto the cone boundary is:
        f_n^proj = (f_n + mu*||f_t||) / (mu^2 + 1)
        f_t^proj = f_t * (mu*f_n^proj / ||f_t||)

    Args:
        f: Contact force in local contact frame [normal, tangent1, tangent2].
        mu: Friction coefficient.

    Returns:
        Projected force onto the friction cone.
    """
    fn = f[0]
    ft_mag = wp.sqrt(f[1] * f[1] + f[2] * f[2])

    # Case 1: Inside the cone - no projection needed
    if ft_mag <= mu * fn:
        return f

    # Case 2: Below dual cone apex - project to origin
    if mu != 0.0 and ft_mag < -fn / mu:
        return wp.vec3(0.0, 0.0, 0.0)

    # Zero friction coefficient special case
    if mu == 0.0 and fn < 0.0:
        return wp.vec3(0.0, f[1], f[2])

    # Case 3: Project onto cone boundary
    if mu * fn < ft_mag:
        fn_new = (fn + mu * ft_mag) / (mu * mu + 1.0)
        if fn_new < 0.0:
            return wp.vec3(0.0, 0.0, 0.0)
        scale = mu * fn_new / ft_mag
        return wp.vec3(fn_new, f[1] * scale, f[2] * scale)

    return f


@wp.func
def project_friction_cone_tangential(f: wp.vec3, mu: float) -> wp.vec3:
    """Project a contact force onto the friction cone (tangential-only clamp).

    Unlike the Anitescu-Tasora (minimum-norm) projection, this preserves the
    normal component and only clamps the tangential magnitude to satisfy
    ``||f_t|| <= mu * f_n``.  This avoids the normal-impulse inflation that
    the AT projection produces at converged fixed points when friction is
    saturated (factor ``(1 + 2*mu^2) / (1 + mu^2)``).

    Three cases:
        1. f inside cone: f_proj = f
        2. f_n < 0: f_proj = 0  (tensile normal not allowed)
        3. ||f_t|| > mu * f_n: clamp tangential, keep normal

    Args:
        f: Contact force in local contact frame [normal, tangent1, tangent2].
        mu: Friction coefficient.

    Returns:
        Projected force onto the friction cone.
    """
    fn = f[0]

    # Tensile normal: project to origin
    if fn <= 0.0:
        return wp.vec3(0.0, 0.0, 0.0)

    ft_mag = wp.sqrt(f[1] * f[1] + f[2] * f[2])

    # Inside cone: no projection needed
    if ft_mag <= mu * fn:
        return f

    # Clamp tangential, preserve normal
    if ft_mag > 0.0:
        scale = mu * fn / ft_mag
        return wp.vec3(fn, f[1] * scale, f[2] * scale)

    return f


# =============================================================================
# Contact Basis Computation
# =============================================================================


@wp.func
def compute_contact_basis(n: wp.vec3) -> wp.mat33:
    """
    Compute an orthonormal contact basis from the contact normal.

    Creates a right-handed coordinate system where:
        - Column 0: contact normal (n)
        - Column 1: first tangent direction (u)
        - Column 2: second tangent direction (w)

    The tangent vectors are computed using the minimum component method:
        1. Find the smallest absolute component of n
        2. Cross n with that axis unit vector to get u
        3. Cross n with u to get w

    Args:
        n: Contact normal vector (must be unit length, pointing from B to A).

    Returns:
        3x3 matrix where columns are [n, u, w].
    """
    # Find the axis with smallest component magnitude
    n_abs = wp.vec3(wp.abs(n[0]), wp.abs(n[1]), wp.abs(n[2]))
    min_i = 0
    if n_abs[1] < n_abs[0]:
        min_i = 1
    if n_abs[2] < n_abs[0] and n_abs[2] < n_abs[1]:
        min_i = 2

    # Get unit vector along minimum axis
    e_i = wp.vec3(0.0, 0.0, 0.0)
    if min_i == 0:
        e_i = wp.vec3(1.0, 0.0, 0.0)
    elif min_i == 1:
        e_i = wp.vec3(0.0, 1.0, 0.0)
    else:
        e_i = wp.vec3(0.0, 0.0, 1.0)

    # Compute tangent vectors: u = normalize(e_i x n), w = n x u
    u = wp.cross(e_i, n)
    u_len = wp.length(u)
    if u_len > 1e-10:
        u = u / u_len
    w = wp.cross(n, u)

    # Return as column-major matrix: [n | u | w]
    return wp.mat33(n[0], u[0], w[0], n[1], u[1], w[1], n[2], u[2], w[2])


# =============================================================================
# Velocity Delta Computation
# =============================================================================


@wp.kernel
def compute_delta_v_from_lambda(
    body_inv_mass: wp.array[float],
    body_inv_inertia_world: wp.array[wp.mat33],
    contact_count: wp.array[int],
    contact_body_a: wp.array[int],
    contact_body_b: wp.array[int],
    jac_n_a: wp.array[wp.spatial_vector],
    jac_n_b: wp.array[wp.spatial_vector],
    jac_t1_a: wp.array[wp.spatial_vector],
    jac_t1_b: wp.array[wp.spatial_vector],
    jac_t2_a: wp.array[wp.spatial_vector],
    jac_t2_b: wp.array[wp.spatial_vector],
    contact_lambda: wp.array[wp.vec3],
    contact_max: int,
    # outputs
    delta_v: wp.array[wp.spatial_vector],
):
    """
    Compute velocity delta from contact impulses.

    The velocity change due to contact impulses is:
        delta_v = W * J^T * lambda = M^{-1} * J^T * lambda

    For each body:
        delta_v_a = W_a * J_a^T * lambda
        delta_v_b = W_b * J_b^T * lambda

    Expanding for body A:
        delta_v_lin = (1/m_a) * (J_n_lin * lambda_n + J_t1_lin * lambda_t1 + J_t2_lin * lambda_t2)
        delta_v_ang = I_a^{-1} * (J_n_ang * lambda_n + J_t1_ang * lambda_t1 + J_t2_ang * lambda_t2)

    This kernel uses atomic_add because multiple contacts may affect the same body.
    """
    tid = wp.tid()

    count = wp.min(contact_max, contact_count[0])
    if tid >= count:
        return

    body_a = contact_body_a[tid]
    body_b = contact_body_b[tid]
    lam = contact_lambda[tid]

    J_n_a = jac_n_a[tid]
    J_t1_a = jac_t1_a[tid]
    J_t2_a = jac_t2_a[tid]
    J_n_b = jac_n_b[tid]
    J_t1_b = jac_t1_b[tid]
    J_t2_b = jac_t2_b[tid]

    # Body A: delta_v_a = W_a * J_a^T * lambda
    if body_a >= 0:
        inv_m_a = body_inv_mass[body_a]
        inv_I_a = body_inv_inertia_world[body_a]

        # J^T * lambda = sum over directions
        J_T_f_lin = wp.spatial_top(J_n_a) * lam[0] + wp.spatial_top(J_t1_a) * lam[1] + wp.spatial_top(J_t2_a) * lam[2]
        J_T_f_ang = (
            wp.spatial_bottom(J_n_a) * lam[0] + wp.spatial_bottom(J_t1_a) * lam[1] + wp.spatial_bottom(J_t2_a) * lam[2]
        )

        # W * J^T * lambda
        dv_a = wp.spatial_vector(J_T_f_lin * inv_m_a, inv_I_a * J_T_f_ang)
        wp.atomic_add(delta_v, body_a, dv_a)

    # Body B: delta_v_b = W_b * J_b^T * lambda
    if body_b >= 0:
        inv_m_b = body_inv_mass[body_b]
        inv_I_b = body_inv_inertia_world[body_b]

        J_T_f_lin = wp.spatial_top(J_n_b) * lam[0] + wp.spatial_top(J_t1_b) * lam[1] + wp.spatial_top(J_t2_b) * lam[2]
        J_T_f_ang = (
            wp.spatial_bottom(J_n_b) * lam[0] + wp.spatial_bottom(J_t1_b) * lam[1] + wp.spatial_bottom(J_t2_b) * lam[2]
        )

        dv_b = wp.spatial_vector(J_T_f_lin * inv_m_b, inv_I_b * J_T_f_ang)
        wp.atomic_add(delta_v, body_b, dv_b)


# =============================================================================
# Contact Kernels (using [nc, 12] Jacobian format)
# =============================================================================


@wp.kernel
def compute_contact_jacobians(
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    shape_body: wp.array[int],
    contact_count: wp.array[int],
    contact_point0: wp.array[wp.vec3],
    contact_point1: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    contact_shape0: wp.array[int],
    contact_shape1: wp.array[int],
    contact_thickness0: wp.array[float],
    contact_thickness1: wp.array[float],
    contact_max: int,
    # outputs
    jacobian: wp.array2d[float],
    body_a: wp.array[int],
    body_b: wp.array[int],
    violation: wp.array[float],
):
    """
    Compute contact Jacobians in unified [nc, 12] format.

    Each contact produces 3 rows in the Jacobian:
        - Row i*3 + 0: Normal direction
        - Row i*3 + 1: Tangent 1 direction
        - Row i*3 + 2: Tangent 2 direction

    Each row has 12 columns: [body_a_lin(3), body_a_ang(3), body_b_lin(3), body_b_ang(3)]
    """
    tid = wp.tid()

    count = wp.min(contact_max, contact_count[0])
    if tid >= count:
        return

    # Get body indices from shapes
    shape_a = contact_shape0[tid]
    shape_b = contact_shape1[tid]

    ba = -1
    bb = -1
    if shape_a >= 0:
        ba = shape_body[shape_a]
    if shape_b >= 0:
        bb = shape_body[shape_b]

    body_a[tid] = ba
    body_b[tid] = bb

    # Build contact frame
    n = contact_normal[tid]
    basis = compute_contact_basis(n)
    t1 = wp.vec3(basis[0, 1], basis[1, 1], basis[2, 1])
    t2 = wp.vec3(basis[0, 2], basis[1, 2], basis[2, 2])

    # Get contact points in world frame
    bx_a = contact_point0[tid]
    bx_b = contact_point1[tid]
    thickness_a = contact_thickness0[tid]
    thickness_b = contact_thickness1[tid]

    # Compute lever arms from body COMs to contact point
    r_a = wp.vec3(0.0)
    r_b = wp.vec3(0.0)

    if ba >= 0:
        X_wb_a = body_q[ba]
        com_a = body_com[ba]
        bx_a = wp.transform_point(X_wb_a, bx_a) + thickness_a * n
        r_a = bx_a - wp.transform_point(X_wb_a, com_a)

    if bb >= 0:
        X_wb_b = body_q[bb]
        com_b = body_com[bb]
        bx_b = wp.transform_point(X_wb_b, bx_b) - thickness_b * n
        r_b = bx_b - wp.transform_point(X_wb_b, com_b)

    # Compute penetration depth (violation)
    d = wp.dot(n, bx_b - bx_a)
    row_base = tid * 3

    violation[row_base + 0] = d
    violation[row_base + 1] = 0.0
    violation[row_base + 2] = 0.0

    # Compute Jacobians
    # v_rel = J_a * V_a + J_b * V_b, v_rel > 0 means separating
    # gap = (x_b - x_a) · n, d(gap)/dt = (v_b - v_a) · n
    # Body A: J_a = [-dir, dir x r_a]
    # Body B: J_b = [dir, r_b x dir]
    n_cross_r_a = wp.cross(n, r_a)
    t1_cross_r_a = wp.cross(t1, r_a)
    t2_cross_r_a = wp.cross(t2, r_a)
    r_b_cross_n = wp.cross(r_b, n)
    r_b_cross_t1 = wp.cross(r_b, t1)
    r_b_cross_t2 = wp.cross(r_b, t2)

    # Normal row
    jacobian[row_base + 0, 0] = -n[0]
    jacobian[row_base + 0, 1] = -n[1]
    jacobian[row_base + 0, 2] = -n[2]
    jacobian[row_base + 0, 3] = n_cross_r_a[0]
    jacobian[row_base + 0, 4] = n_cross_r_a[1]
    jacobian[row_base + 0, 5] = n_cross_r_a[2]
    jacobian[row_base + 0, 6] = n[0]
    jacobian[row_base + 0, 7] = n[1]
    jacobian[row_base + 0, 8] = n[2]
    jacobian[row_base + 0, 9] = r_b_cross_n[0]
    jacobian[row_base + 0, 10] = r_b_cross_n[1]
    jacobian[row_base + 0, 11] = r_b_cross_n[2]

    # Tangent 1 row
    jacobian[row_base + 1, 0] = -t1[0]
    jacobian[row_base + 1, 1] = -t1[1]
    jacobian[row_base + 1, 2] = -t1[2]
    jacobian[row_base + 1, 3] = t1_cross_r_a[0]
    jacobian[row_base + 1, 4] = t1_cross_r_a[1]
    jacobian[row_base + 1, 5] = t1_cross_r_a[2]
    jacobian[row_base + 1, 6] = t1[0]
    jacobian[row_base + 1, 7] = t1[1]
    jacobian[row_base + 1, 8] = t1[2]
    jacobian[row_base + 1, 9] = r_b_cross_t1[0]
    jacobian[row_base + 1, 10] = r_b_cross_t1[1]
    jacobian[row_base + 1, 11] = r_b_cross_t1[2]

    # Tangent 2 row
    jacobian[row_base + 2, 0] = -t2[0]
    jacobian[row_base + 2, 1] = -t2[1]
    jacobian[row_base + 2, 2] = -t2[2]
    jacobian[row_base + 2, 3] = t2_cross_r_a[0]
    jacobian[row_base + 2, 4] = t2_cross_r_a[1]
    jacobian[row_base + 2, 5] = t2_cross_r_a[2]
    jacobian[row_base + 2, 6] = t2[0]
    jacobian[row_base + 2, 7] = t2[1]
    jacobian[row_base + 2, 8] = t2[2]
    jacobian[row_base + 2, 9] = r_b_cross_t2[0]
    jacobian[row_base + 2, 10] = r_b_cross_t2[1]
    jacobian[row_base + 2, 11] = r_b_cross_t2[2]


@wp.kernel
def compute_contact_residual(
    body_qd: wp.array[wp.spatial_vector],
    body_f: wp.array[wp.spatial_vector],
    body_inv_mass: wp.array[float],
    body_inv_inertia_world: wp.array[wp.mat33],
    contact_count: wp.array[int],
    body_a: wp.array[int],
    body_b: wp.array[int],
    jacobian: wp.array2d[float],
    violation: wp.array[float],
    dt: float,
    alpha: float,
    recovery_speed: float,
    contact_max: int,
    # outputs
    residual: wp.array[float],
):
    """
    Compute contact residual using unified Jacobian format (DVI-style).

    Computes b_i = -(J * v_pred + phi / (dt + alpha)) for each constraint row.

    The Baumgarte term phi / (dt + alpha) is applied uniformly for all phi:
        - phi > 0 (gap): positive b_i drives lambda toward zero via
          unilateral projection (lambda_n >= 0) in the iterative solver.
        - phi < 0 (penetration): negative correction pushes bodies apart.

    recovery_speed limits the maximum penetration correction velocity.
    """
    tid = wp.tid()

    count = wp.min(contact_max, contact_count[0])
    if tid >= count:
        return

    ba = body_a[tid]
    bb = body_b[tid]
    row_base = tid * 3

    # Compute predicted velocities
    v_pred_a = wp.spatial_vector(wp.vec3(0.0), wp.vec3(0.0))
    v_pred_b = wp.spatial_vector(wp.vec3(0.0), wp.vec3(0.0))

    if ba >= 0:
        v_a = body_qd[ba]
        f_a = body_f[ba]
        inv_m_a = body_inv_mass[ba]
        inv_I_a = body_inv_inertia_world[ba]
        accel_lin_a = wp.spatial_top(f_a) * inv_m_a
        accel_ang_a = inv_I_a * wp.spatial_bottom(f_a)
        v_pred_a = wp.spatial_vector(
            wp.spatial_top(v_a) + dt * accel_lin_a,
            wp.spatial_bottom(v_a) + dt * accel_ang_a,
        )

    if bb >= 0:
        v_b = body_qd[bb]
        f_b = body_f[bb]
        inv_m_b = body_inv_mass[bb]
        inv_I_b = body_inv_inertia_world[bb]
        accel_lin_b = wp.spatial_top(f_b) * inv_m_b
        accel_ang_b = inv_I_b * wp.spatial_bottom(f_b)
        v_pred_b = wp.spatial_vector(
            wp.spatial_top(v_b) + dt * accel_lin_b,
            wp.spatial_bottom(v_b) + dt * accel_ang_b,
        )

    # Compute J * v_pred for each row
    for c in range(3):
        row = row_base + c
        J_a_lin = wp.vec3(jacobian[row, 0], jacobian[row, 1], jacobian[row, 2])
        J_a_ang = wp.vec3(jacobian[row, 3], jacobian[row, 4], jacobian[row, 5])
        J_b_lin = wp.vec3(jacobian[row, 6], jacobian[row, 7], jacobian[row, 8])
        J_b_ang = wp.vec3(jacobian[row, 9], jacobian[row, 10], jacobian[row, 11])

        J_v = 0.0
        if ba >= 0:
            J_v = J_v + wp.dot(J_a_lin, wp.spatial_top(v_pred_a))
            J_v = J_v + wp.dot(J_a_ang, wp.spatial_bottom(v_pred_a))
        if bb >= 0:
            J_v = J_v + wp.dot(J_b_lin, wp.spatial_top(v_pred_b))
            J_v = J_v + wp.dot(J_b_ang, wp.spatial_bottom(v_pred_b))

        phi = violation[row]

        # Normal constraint: b_i = -(J_v + phi / (dt + alpha))
        # Tangent constraints: b_i = -J_v (no position correction)
        baumgarte = 0.0
        if c == 0:
            baumgarte = phi / (dt + alpha)
            # Clamp penetration recovery speed (only for penetration, phi < 0)
            if recovery_speed > 0.0 and baumgarte < 0.0:
                baumgarte = wp.max(baumgarte, -recovery_speed)

        residual[row] = -(J_v + baumgarte)


@wp.kernel
def compute_contact_diagonal(
    body_inv_mass: wp.array[float],
    body_inv_inertia_world: wp.array[wp.mat33],
    contact_count: wp.array[int],
    body_a: wp.array[int],
    body_b: wp.array[int],
    jacobian: wp.array2d[float],
    reg: float,
    contact_max: int,
    # outputs
    diag: wp.array[float],
):
    """Compute diagonal preconditioner using unified Jacobian format.

    The diagonal entry for constraint row i is:

        d_i = J_i M^{-1} J_i^T + reg

    where ``reg`` is the regularization added to the diagonal for numerical
    stability.
    """
    tid = wp.tid()

    count = wp.min(contact_max, contact_count[0])
    if tid >= count:
        return

    ba = body_a[tid]
    bb = body_b[tid]
    row_base = tid * 3

    e = reg

    for c in range(3):
        row = row_base + c
        J_a_lin = wp.vec3(jacobian[row, 0], jacobian[row, 1], jacobian[row, 2])
        J_a_ang = wp.vec3(jacobian[row, 3], jacobian[row, 4], jacobian[row, 5])
        J_b_lin = wp.vec3(jacobian[row, 6], jacobian[row, 7], jacobian[row, 8])
        J_b_ang = wp.vec3(jacobian[row, 9], jacobian[row, 10], jacobian[row, 11])

        d = e

        if ba >= 0:
            inv_m_a = body_inv_mass[ba]
            inv_I_a = body_inv_inertia_world[ba]
            d = d + wp.dot(J_a_lin, J_a_lin) * inv_m_a
            d = d + wp.dot(J_a_ang, inv_I_a * J_a_ang)

        if bb >= 0:
            inv_m_b = body_inv_mass[bb]
            inv_I_b = body_inv_inertia_world[bb]
            d = d + wp.dot(J_b_lin, J_b_lin) * inv_m_b
            d = d + wp.dot(J_b_ang, inv_I_b * J_b_ang)

        # Handle contacts between fixed bodies
        if d < 1e-3:
            d = 1e10

        diag[row] = d


@wp.func
def _project_friction(f: wp.vec3, mu: float, use_tangential: int) -> wp.vec3:
    """Dispatch to cone or tangential projection based on flag."""
    if use_tangential != 0:
        return project_friction_cone_tangential(f, mu)
    return project_friction_cone(f, mu)


@wp.kernel
def project_friction_cones(
    contact_count: wp.array[int],
    friction: wp.array[float],
    contact_max: int,
    use_tangential: int,
    # inputs/outputs
    lambda_: wp.array[float],
):
    """Project contact impulses onto the friction cone.

    Uses cone (AT) or tangential-only projection based on ``use_tangential``.
    Operates on flat lambda array where each contact has 3 consecutive values.
    """
    tid = wp.tid()

    count = wp.min(contact_max, contact_count[0])
    if tid >= count:
        return

    row_base = tid * 3
    mu = friction[tid]

    f = wp.vec3(lambda_[row_base + 0], lambda_[row_base + 1], lambda_[row_base + 2])
    f_proj = _project_friction(f, mu, use_tangential)

    lambda_[row_base + 0] = f_proj[0]
    lambda_[row_base + 1] = f_proj[1]
    lambda_[row_base + 2] = f_proj[2]


@wp.kernel
def apply_contact_forces(
    contact_count: wp.array[int],
    body_a: wp.array[int],
    body_b: wp.array[int],
    jacobian: wp.array2d[float],
    lambda_: wp.array[float],
    dt: float,
    contact_max: int,
    # outputs
    body_f: wp.array[wp.spatial_vector],
):
    """
    Apply contact forces using unified Jacobian format.

    Computes F = J^T * lambda / dt for each contact.
    """
    tid = wp.tid()

    count = wp.min(contact_max, contact_count[0])
    if tid >= count:
        return

    ba = body_a[tid]
    bb = body_b[tid]
    row_base = tid * 3

    for c in range(3):
        row = row_base + c
        lam = lambda_[row]

        J_a_lin = wp.vec3(jacobian[row, 0], jacobian[row, 1], jacobian[row, 2])
        J_a_ang = wp.vec3(jacobian[row, 3], jacobian[row, 4], jacobian[row, 5])
        J_b_lin = wp.vec3(jacobian[row, 6], jacobian[row, 7], jacobian[row, 8])
        J_b_ang = wp.vec3(jacobian[row, 9], jacobian[row, 10], jacobian[row, 11])

        # Apply forces: F = J^T * lambda / dt
        # With corrected Jacobians (J_b = [n, ...] for body B), positive lambda
        # pushes body B in +n direction (away from A), which is correct for contacts
        if ba >= 0:
            f_a = wp.spatial_vector(J_a_lin * lam / dt, J_a_ang * lam / dt)
            wp.atomic_add(body_f, ba, f_a)

        if bb >= 0:
            f_b = wp.spatial_vector(J_b_lin * lam / dt, J_b_ang * lam / dt)
            wp.atomic_add(body_f, bb, f_b)


@wp.kernel
def write_contact_forces(
    contact_count: wp.array[int],
    contact_normal: wp.array[wp.vec3],
    lambda_: wp.array[float],
    dt: float,
    contact_max: int,
    # outputs
    contact_force: wp.array[wp.spatial_vector],
):
    """Write solved contact forces to contacts.force for sensor readout.

    Reconstructs the world-frame contact force from the contact-space impulses
    (lambda) and the contact normal/tangent basis.  The output is a spatial
    vector per contact where spatial_top = linear force [N] and
    spatial_bottom = 0 (torque unused by the contact sensor).

    Force convention: force exerted on shape0 (body A) by shape1 (body B),
    matching MuJoCo/XPBD convention used by ``SensorContact``.

    Args:
        contact_count: Number of active contacts.
        contact_normal: Per-contact normal (from B to A).
        lambda_: Solved contact impulses [lambda_n, lambda_t1, lambda_t2] per contact.
        dt: Time step (impulse / dt = force).
        contact_max: Maximum number of contacts (launch dimension).
        contact_force: Output spatial force per contact.
    """
    tid = wp.tid()

    count = wp.min(contact_max, contact_count[0])
    if tid >= count:
        return

    n = contact_normal[tid]
    basis = compute_contact_basis(n)
    t1 = wp.vec3(basis[0, 1], basis[1, 1], basis[2, 1])
    t2 = wp.vec3(basis[0, 2], basis[1, 2], basis[2, 2])

    row_base = tid * 3
    lambda_n = lambda_[row_base + 0]
    lambda_t1 = lambda_[row_base + 1]
    lambda_t2 = lambda_[row_base + 2]

    # Force on body A (shape0): J_a^T * lambda / dt
    # J_a_lin for normal row is -n, for t1 is -t1, for t2 is -t2
    # So force_on_A = (-n * lambda_n - t1 * lambda_t1 - t2 * lambda_t2) / dt
    force_lin = -(n * lambda_n + t1 * lambda_t1 + t2 * lambda_t2) / dt

    contact_force[tid] = wp.spatial_vector(force_lin, wp.vec3(0.0, 0.0, 0.0))


@wp.kernel
def warm_start_lambda(
    contact_count: wp.array[int],
    match_index: wp.array[wp.int32],
    prev_lambda: wp.array[float],
    contact_max: int,
    # outputs
    lambda_: wp.array[float],
):
    """Initialize lambda from previous frame's solved values.

    For each current-frame contact, ``match_index[tid]`` holds the index of
    the corresponding contact in the previous frame's sorted buffer (produced
    by :class:`~newton._src.geometry.contact_match.ContactMatcher`).  Matched
    contacts copy the previous normal + tangent impulses; unmatched contacts
    (``match_index < 0``) start from zero.

    Fully GPU-resident, graph-capturable, no CPU synchronization.
    """
    tid = wp.tid()

    count = wp.min(contact_max, contact_count[0])
    if tid >= count:
        return

    mi = match_index[tid]
    row_base = tid * 3

    if mi >= 0:
        prev_base = mi * 3
        lambda_[row_base + 0] = prev_lambda[prev_base + 0]
        lambda_[row_base + 1] = prev_lambda[prev_base + 1]
        lambda_[row_base + 2] = prev_lambda[prev_base + 2]
    else:
        lambda_[row_base + 0] = 0.0
        lambda_[row_base + 1] = 0.0
        lambda_[row_base + 2] = 0.0


@wp.kernel
def save_lambda(
    contact_count: wp.array[int],
    lambda_: wp.array[float],
    contact_max: int,
    # outputs
    prev_lambda: wp.array[float],
):
    """Save current frame's solved lambda for next-frame warm-starting.

    Copies the active portion of the lambda buffer; inactive slots are zeroed
    so stale values from a previous frame with more contacts don't leak.
    """
    tid = wp.tid()
    if tid >= contact_max:
        return

    count = wp.min(contact_max, contact_count[0])
    row_base = tid * 3

    if tid < count:
        prev_lambda[row_base + 0] = lambda_[row_base + 0]
        prev_lambda[row_base + 1] = lambda_[row_base + 1]
        prev_lambda[row_base + 2] = lambda_[row_base + 2]
    else:
        prev_lambda[row_base + 0] = 0.0
        prev_lambda[row_base + 1] = 0.0
        prev_lambda[row_base + 2] = 0.0


@wp.kernel
def compute_contact_diagonal_block3x3(
    body_inv_mass: wp.array[float],
    body_inv_inertia_world: wp.array[wp.mat33],
    contact_count: wp.array[int],
    body_a: wp.array[int],
    body_b: wp.array[int],
    jacobian: wp.array2d[float],
    reg: float,
    contact_max: int,
    # outputs
    diag_block_inv: wp.array[wp.mat33],
):
    """Compute inverse of 3x3 diagonal block of N = J M^{-1} J^T per contact.

    For contact k with rows [3k, 3k+1, 3k+2], computes:
        D[i,j] = J_i M^{-1} J_j^T   (i,j in {0,1,2})
    then inverts the 3x3 block.

    This captures the cross-coupling between normal and tangential directions
    that the scalar trace-based preconditioner misses.
    """
    tid = wp.tid()

    count = wp.min(contact_max, contact_count[0])
    if tid >= count:
        return

    ba = body_a[tid]
    bb = body_b[tid]
    row_base = tid * 3

    # Build 3x3 block: D[i][j] = sum over bodies of (J_i_lin * inv_m * J_j_lin + J_i_ang * inv_I * J_j_ang)
    D = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    for i in range(3):
        ri = row_base + i
        J_a_lin_i = wp.vec3(jacobian[ri, 0], jacobian[ri, 1], jacobian[ri, 2])
        J_a_ang_i = wp.vec3(jacobian[ri, 3], jacobian[ri, 4], jacobian[ri, 5])
        J_b_lin_i = wp.vec3(jacobian[ri, 6], jacobian[ri, 7], jacobian[ri, 8])
        J_b_ang_i = wp.vec3(jacobian[ri, 9], jacobian[ri, 10], jacobian[ri, 11])

        for j in range(3):
            rj = row_base + j
            J_a_lin_j = wp.vec3(jacobian[rj, 0], jacobian[rj, 1], jacobian[rj, 2])
            J_a_ang_j = wp.vec3(jacobian[rj, 3], jacobian[rj, 4], jacobian[rj, 5])
            J_b_lin_j = wp.vec3(jacobian[rj, 6], jacobian[rj, 7], jacobian[rj, 8])
            J_b_ang_j = wp.vec3(jacobian[rj, 9], jacobian[rj, 10], jacobian[rj, 11])

            d_ij = 0.0

            if ba >= 0:
                inv_m_a = body_inv_mass[ba]
                inv_I_a = body_inv_inertia_world[ba]
                d_ij += wp.dot(J_a_lin_i, J_a_lin_j) * inv_m_a
                d_ij += wp.dot(J_a_ang_i, inv_I_a * J_a_ang_j)

            if bb >= 0:
                inv_m_b = body_inv_mass[bb]
                inv_I_b = body_inv_inertia_world[bb]
                d_ij += wp.dot(J_b_lin_i, J_b_lin_j) * inv_m_b
                d_ij += wp.dot(J_b_ang_i, inv_I_b * J_b_ang_j)

            D[i, j] = d_ij

    # Add regularization to diagonal
    D[0, 0] = D[0, 0] + reg
    D[1, 1] = D[1, 1] + reg
    D[2, 2] = D[2, 2] + reg

    # Invert the 3x3 block
    diag_block_inv[tid] = wp.inverse(D)
