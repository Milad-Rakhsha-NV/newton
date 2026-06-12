# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Dense solver utilities for constraint systems.

This module provides shared functions for building and solving dense constraint
systems of the form:

    N * lambda = b

where:
    N = J * M_inv * J^T  (system matrix)
    M_inv = inverse mass matrix
    lambda = Lagrange multipliers (constraint impulses)
    b = right-hand side

These utilities are used by both velocity-level and position-level solvers.
"""

from __future__ import annotations

import numpy as np


def build_dense_jacobian(
    joint_jac: np.ndarray,
    joint_body_a: np.ndarray,
    joint_body_b: np.ndarray,
    joint_num_constraints: np.ndarray,
    nj: int,
    nb: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Build dense Jacobian matrix from per-joint sparse data.

    Args:
        joint_jac: Per-joint Jacobians in compact format [total_nc, 12].
            Uses compact storage where each joint's constraints are
            stored contiguously.
        joint_body_a: Parent body indices [nj].
        joint_body_b: Child body indices [nj].
        joint_num_constraints: Actual constraint count per joint [nj].
        nj: Number of joints.
        nb: Number of bodies.

    Returns:
        Bj: Dense Jacobian [total_nc, 6*nb].
        phi_indices: Mapping from flat constraint index to (joint, local_idx).
        total_nc: Total number of active constraints.
    """
    total_nc = sum(joint_num_constraints[j] for j in range(nj))
    if total_nc == 0:
        return np.zeros((0, 6 * nb)), np.zeros((0, 2), dtype=int), 0

    # Compute compact storage offsets
    nc_offset = np.zeros(nj + 1, dtype=np.int32)
    for j in range(nj):
        nc_offset[j + 1] = nc_offset[j] + joint_num_constraints[j]

    Bj = np.zeros((total_nc, 6 * nb))
    phi_indices = np.zeros((total_nc, 2), dtype=int)
    constraint_idx = 0

    for j in range(nj):
        nc = joint_num_constraints[j]
        if nc == 0:
            continue

        a_id = joint_body_a[j]
        b_id = joint_body_b[j]
        jac_row = nc_offset[j]

        # Parent body contribution
        if a_id >= 0:
            rows = slice(constraint_idx, constraint_idx + nc)
            cols_a = slice(6 * a_id, 6 * a_id + 6)
            Bj[rows, cols_a] = joint_jac[jac_row : jac_row + nc, 0:6]

        # Child body contribution
        rows = slice(constraint_idx, constraint_idx + nc)
        cols_b = slice(6 * b_id, 6 * b_id + 6)
        Bj[rows, cols_b] = joint_jac[jac_row : jac_row + nc, 6:12]

        # Store mapping
        for c in range(nc):
            phi_indices[constraint_idx + c] = [j, c]

        constraint_idx += nc

    return Bj, phi_indices, total_nc


def build_inverse_mass_matrix(
    inv_mass: np.ndarray,
    inv_inertia_world: np.ndarray,
    nb: int,
) -> np.ndarray:
    """Build block-diagonal inverse mass matrix.

    Args:
        inv_mass: Inverse masses [nb].
        inv_inertia_world: World-frame inverse inertia tensors [nb x 3 x 3].
        nb: Number of bodies.

    Returns:
        M_inv: Block-diagonal inverse mass matrix [6*nb x 6*nb].
    """
    M_inv = np.zeros((nb * 6, nb * 6))
    for i in range(nb):
        if inv_mass[i] > 0:
            M_inv[6 * i, 6 * i] = inv_mass[i]
            M_inv[6 * i + 1, 6 * i + 1] = inv_mass[i]
            M_inv[6 * i + 2, 6 * i + 2] = inv_mass[i]
            M_inv[6 * i + 3 : 6 * i + 6, 6 * i + 3 : 6 * i + 6] = inv_inertia_world[i]
    return M_inv


def build_system_matrix(
    Bj: np.ndarray,
    M_inv: np.ndarray,
    reg: float,
) -> np.ndarray:
    """
    Build constraint system matrix N = J * M_inv * J^T + reg*I.

    Args:
        Bj: Dense Jacobian [nc x 6*nb].
        M_inv: Inverse mass matrix [6*nb x 6*nb].
        reg: Regularization for numerical stability.

    Returns:
        N: System matrix [nc x nc].
    """
    nc = Bj.shape[0]
    if nc == 0:
        return np.zeros((0, 0))

    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        N = Bj @ M_inv @ Bj.T
    N = N + np.eye(nc) * reg
    return N


def extract_phi_vector(
    phi: np.ndarray,
    joint_num_constraints: np.ndarray,
    nj: int,
) -> np.ndarray:
    """
    Extract constraint violations into flat vector.

    Args:
        phi: Per-joint violations in compact format [total_nc].
        joint_num_constraints: Number of constraints per joint [nj].
        nj: Number of joints.

    Returns:
        phi_vec: Flat violation vector [total_nc].
    """
    total_nc = sum(joint_num_constraints[j] for j in range(nj))
    if total_nc == 0:
        return np.zeros(0)

    # Compute compact storage offsets
    nc_offset = np.zeros(nj + 1, dtype=np.int32)
    for j in range(nj):
        nc_offset[j + 1] = nc_offset[j] + joint_num_constraints[j]

    phi_vec = np.zeros(total_nc)
    constraint_idx = 0

    for j in range(nj):
        nc = joint_num_constraints[j]
        if nc == 0:
            continue
        jac_row = nc_offset[j]
        phi_vec[constraint_idx : constraint_idx + nc] = phi[jac_row : jac_row + nc]
        constraint_idx += nc

    return phi_vec


def compute_velocity_rhs(
    Bj: np.ndarray,
    M_inv: np.ndarray,
    v: np.ndarray,
    f_ext: np.ndarray,
    dt: float,
    alpha: float,
    phi_vec: np.ndarray,
) -> np.ndarray:
    """
    Compute right-hand side for velocity-level constraint solve.

    The velocity-level system solves:
        N * lambda = -b
    where:
        b = J * v_pred + phi / (dt + alpha)
        v_pred = v + dt * M_inv * f_ext

    Args:
        Bj: Dense Jacobian [nc x 6*nb].
        M_inv: Inverse mass matrix [6*nb x 6*nb].
        v: Current velocities [6*nb].
        f_ext: External forces [6*nb].
        dt: Time step.
        alpha: Baumgarte damping parameter (DVI-style).
        phi_vec: Constraint violations [nc].

    Returns:
        b: Right-hand side vector [nc].
    """
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        v_pred = v + dt * (M_inv @ f_ext)
    b = Bj @ v_pred + phi_vec / (dt + alpha)
    return b


def compute_position_rhs(phi_vec: np.ndarray) -> np.ndarray:
    """
    Compute right-hand side for position-level constraint solve.

    The position-level system solves:
        N * nu = -phi

    Args:
        phi_vec: Constraint violations [nc].

    Returns:
        b: Right-hand side vector [nc].
    """
    return phi_vec


def solve_direct(N: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """
    Solve constraint system using direct method.

    Solves: N * lambda = -rhs

    Args:
        N: System matrix [nc x nc].
        rhs: Right-hand side [nc].

    Returns:
        lam: Solution [nc].
    """
    if N.shape[0] == 0:
        return np.zeros(0)

    try:
        lam = np.linalg.solve(N, -rhs)
    except np.linalg.LinAlgError:
        lam = np.linalg.lstsq(N, -rhs, rcond=None)[0]

    return lam


def solve_jacobi(
    N: np.ndarray,
    rhs: np.ndarray,
    omega: float,
    relax: float,
    max_iter: int,
    tol: float,
) -> tuple[np.ndarray, int]:
    """
    Solve constraint system using Jacobi iterations.

    Solves: N * lambda = -rhs

    Args:
        N: System matrix [nc x nc].
        rhs: Right-hand side [nc].
        omega: Step size (iterative solvers only).
        relax: Solution mixing factor (iterative solvers only).
        max_iter: Maximum iterations.
        tol: Convergence tolerance.

    Returns:
        lam: Solution [nc].
        iterations: Number of iterations performed.
    """
    nc = N.shape[0]
    if nc == 0:
        return np.zeros(0), 0

    # Diagonal preconditioner
    diag = np.diag(N)
    D_inv = np.where(diag > 1e-10, 1.0 / diag, 1.0)

    lam = np.zeros(nc)
    for iteration in range(max_iter):
        lam_old = lam.copy()
        grad = N @ lam + rhs
        lam_new = lam - omega * D_inv * grad
        lam = relax * lam_new + (1.0 - relax) * lam

        if np.linalg.norm(lam - lam_old) < tol:
            return lam, iteration + 1

    return lam, max_iter


def solve_block_gs(
    joint_jac: np.ndarray,
    M_inv: np.ndarray,
    rhs_flat: np.ndarray,
    joint_body_a: np.ndarray,
    joint_body_b: np.ndarray,
    joint_num_constraints: np.ndarray,
    omega: float,
    relax: float,
    max_iter: int,
    tol: float,
    nj: int,
    nb: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Solve constraint system using block Gauss-Seidel.

    Solves per-joint blocks, immediately updating delta_v after each block.
    This is the Gauss-Seidel variant that operates on joint-sized blocks.

    Args:
        joint_jac: Per-joint Jacobians in compact format [total_nc, 12].
            Uses compact storage where each joint's constraints are
            stored contiguously.
        M_inv: Inverse mass matrix [6*nb, 6*nb].
        rhs_flat: Per-joint RHS in compact format [total_nc].
        joint_body_a: Parent body indices [nj].
        joint_body_b: Child body indices [nj].
        joint_num_constraints: Actual constraint count per joint [nj].
        omega: Step size (iterative solvers only).
        relax: Solution mixing factor (iterative solvers only).
        max_iter: Maximum iterations.
        tol: Convergence tolerance.
        nj: Number of joints.
        nb: Number of bodies.

    Returns:
        lambda_c: Per-joint multipliers in compact format [total_nc].
        delta_v: Velocity delta [nb*6].
        iterations: Number of iterations performed.
    """
    # Compute compact storage offsets
    nc_offset = np.zeros(nj + 1, dtype=np.int32)
    for j in range(nj):
        nc_offset[j + 1] = nc_offset[j] + joint_num_constraints[j]
    total_nc = nc_offset[nj]

    lambda_c = np.zeros(total_nc)
    delta_v = np.zeros(nb * 6)

    for iteration in range(max_iter):
        max_delta = 0.0

        for j in range(nj):
            nc = joint_num_constraints[j]
            if nc == 0:
                continue

            jac_row = nc_offset[j]
            a_id = joint_body_a[j]
            b_id = joint_body_b[j]

            J_a = joint_jac[jac_row : jac_row + nc, 0:6]
            J_b = joint_jac[jac_row : jac_row + nc, 6:12]

            # Compute JWJt for this joint block
            JWJt_i = np.zeros((nc, nc))
            if a_id >= 0:
                JWJt_i += J_a @ M_inv[6 * a_id : 6 * a_id + 6, 6 * a_id : 6 * a_id + 6] @ J_a.T
            JWJt_i += J_b @ M_inv[6 * b_id : 6 * b_id + 6, 6 * b_id : 6 * b_id + 6] @ J_b.T

            # Compute J * delta_v
            J_delta = np.zeros(nc)
            if a_id >= 0:
                J_delta += J_a @ delta_v[6 * a_id : 6 * a_id + 6]
            J_delta += J_b @ delta_v[6 * b_id : 6 * b_id + 6]

            # Solve: delta_f = solve(JWJt, rhs - J_delta) * omega
            rhs_j = rhs_flat[jac_row : jac_row + nc]
            residual = -rhs_j - J_delta
            try:
                delta_f = np.linalg.solve(JWJt_i, residual) * omega
            except np.linalg.LinAlgError:
                delta_f = residual / (np.diag(JWJt_i) + 1e-8) * omega

            # Update lambda with relaxation
            lambda_new = lambda_c[jac_row : jac_row + nc] + delta_f
            lam_slice = slice(jac_row, jac_row + nc)
            lambda_c[lam_slice] = relax * lambda_new + (1 - relax) * lambda_c[lam_slice]

            # Update delta_v (Gauss-Seidel: immediate update)
            if a_id >= 0:
                idx_a = slice(6 * a_id, 6 * a_id + 6)
                delta_v[idx_a] += M_inv[idx_a, idx_a] @ J_a.T @ delta_f
            idx_b = slice(6 * b_id, 6 * b_id + 6)
            delta_v[idx_b] += M_inv[idx_b, idx_b] @ J_b.T @ delta_f

            max_delta = max(max_delta, float(np.linalg.norm(delta_f)))

        if max_delta < tol:
            return lambda_c, delta_v, iteration + 1

    return lambda_c, delta_v, max_iter


def compute_delta_v(
    M_inv: np.ndarray,
    Bj: np.ndarray,
    lam: np.ndarray,
) -> np.ndarray:
    """
    Compute velocity delta from multipliers.

    delta_v = M_inv * J^T * lambda

    Args:
        M_inv: Inverse mass matrix [6*nb x 6*nb].
        Bj: Dense Jacobian [nc x 6*nb].
        lam: Multipliers [nc].

    Returns:
        delta_v: Velocity delta [6*nb].
    """
    if lam.size == 0:
        return np.zeros(M_inv.shape[0])

    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        delta_v = M_inv @ Bj.T @ lam

    return delta_v


def unpack_lambda_to_joints(
    lam_flat: np.ndarray,
    joint_num_constraints: np.ndarray,
    nj: int,
) -> np.ndarray:
    """
    Unpack flat lambda vector to per-joint compact format.

    Args:
        lam_flat: Flat multipliers [total_nc].
        joint_num_constraints: Number of constraints per joint [nj].
        nj: Number of joints.

    Returns:
        lambda_c: Per-joint multipliers in compact format [total_nc].
    """
    # Compute compact storage offsets
    nc_offset = np.zeros(nj + 1, dtype=np.int32)
    for j in range(nj):
        nc_offset[j + 1] = nc_offset[j] + joint_num_constraints[j]
    total_nc = nc_offset[nj]

    lambda_c = np.zeros(total_nc)
    constraint_idx = 0

    for j in range(nj):
        nc = joint_num_constraints[j]
        if nc == 0:
            continue
        jac_row = nc_offset[j]
        lambda_c[jac_row : jac_row + nc] = lam_flat[constraint_idx : constraint_idx + nc]
        constraint_idx += nc

    return lambda_c


# =============================================================================
# Contact System Building (Dense)
# =============================================================================


def build_contact_jacobian_dense(
    jac_n_a: np.ndarray,
    jac_n_b: np.ndarray,
    jac_t1_a: np.ndarray,
    jac_t1_b: np.ndarray,
    jac_t2_a: np.ndarray,
    jac_t2_b: np.ndarray,
    contact_body_a: np.ndarray,
    contact_body_b: np.ndarray,
    num_contacts: int,
    nb: int,
) -> np.ndarray:
    """
    Build dense contact Jacobian matrix.

    Each contact has 3 constraints (normal, tangent1, tangent2) and affects 2 bodies.
    The Jacobian maps body velocities to contact-space velocities.

    Args:
        jac_n_a, jac_n_b: Normal Jacobians [nc x 6] spatial vectors.
        jac_t1_a, jac_t1_b: Tangent1 Jacobians [nc x 6].
        jac_t2_a, jac_t2_b: Tangent2 Jacobians [nc x 6].
        contact_body_a, contact_body_b: Body indices [nc].
        num_contacts: Number of active contacts.
        nb: Number of bodies.

    Returns:
        Bc: Dense contact Jacobian [3*nc x 6*nb].
    """
    if num_contacts == 0:
        return np.zeros((0, 6 * nb))

    nc3 = num_contacts * 3
    Bc = np.zeros((nc3, 6 * nb))

    for i in range(num_contacts):
        row_n = i * 3
        row_t1 = i * 3 + 1
        row_t2 = i * 3 + 2

        body_a = contact_body_a[i]
        body_b = contact_body_b[i]

        # Body A contribution
        if body_a >= 0:
            cols = slice(6 * body_a, 6 * body_a + 6)
            Bc[row_n, cols] = jac_n_a[i]
            Bc[row_t1, cols] = jac_t1_a[i]
            Bc[row_t2, cols] = jac_t2_a[i]

        # Body B contribution
        if body_b >= 0:
            cols = slice(6 * body_b, 6 * body_b + 6)
            Bc[row_n, cols] = jac_n_b[i]
            Bc[row_t1, cols] = jac_t1_b[i]
            Bc[row_t2, cols] = jac_t2_b[i]

    return Bc


def build_contact_system_matrix(
    Bc: np.ndarray,
    M_inv: np.ndarray,
    reg: float,
) -> np.ndarray:
    """
    Build contact system matrix N = J * M_inv * J^T + reg*I.

    Args:
        Bc: Dense contact Jacobian [3*nc x 6*nb].
        M_inv: Inverse mass matrix [6*nb x 6*nb].
        reg: Regularization for numerical stability.

    Returns:
        N: Contact system matrix [3*nc x 3*nc].
    """
    nc3 = Bc.shape[0]
    if nc3 == 0:
        return np.zeros((0, 0))

    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        N = Bc @ M_inv @ Bc.T
    N = N + np.eye(nc3) * reg
    return N


def compute_contact_rhs_dense(
    Bc: np.ndarray,
    M_inv: np.ndarray,
    v: np.ndarray,
    f_ext: np.ndarray,
    contact_depth: np.ndarray,
    dt: float,
    alpha: float,
    recovery_speed: float,
    num_contacts: int,
) -> np.ndarray:
    """
    Compute right-hand side for contact velocity solve.

    The contact system solves:
        N * lambda = -rhs
    where:
        rhs = J * v_pred + baumgarte_correction
        v_pred = v + dt * M_inv * f_ext
        baumgarte_correction = phi / (dt + alpha) (only for normal, only for penetration)

    Args:
        Bc: Dense contact Jacobian [3*nc x 6*nb].
        M_inv: Inverse mass matrix [6*nb x 6*nb].
        v: Current velocities [6*nb].
        f_ext: External forces [6*nb].
        contact_depth: Penetration depths [nc].
        dt: Time step.
        alpha: Baumgarte damping parameter.
        recovery_speed: Maximum penetration correction speed (-1 for unlimited).
        num_contacts: Number of contacts.

    Returns:
        rhs: Right-hand side vector [3*nc].
    """
    if num_contacts == 0:
        return np.zeros(0)

    # Predicted velocity
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        v_pred = v + dt * (M_inv @ f_ext)

    # J * v_pred
    rhs = Bc @ v_pred

    # Add alpha-damped Baumgarte stabilization to normal components only
    for i in range(num_contacts):
        phi = contact_depth[i]
        if phi < 0:  # Only for penetration
            baumgarte = phi / (dt + alpha)
            # Apply contact recovery speed limit if set
            if recovery_speed > 0:
                baumgarte = max(baumgarte, -recovery_speed)
            rhs[i * 3] += baumgarte

    return rhs


def solve_contact_direct(
    N: np.ndarray,
    rhs: np.ndarray,
    friction: np.ndarray,
) -> np.ndarray:
    """
    Solve contact system using direct method with friction cone projection.

    Solves: N * lambda = -rhs, then projects onto friction cones.

    Args:
        N: System matrix [3*nc x 3*nc].
        rhs: Right-hand side [3*nc].
        friction: Friction coefficients [nc].

    Returns:
        lam: Solution [3*nc].
    """
    nc3 = N.shape[0]
    if nc3 == 0:
        return np.zeros(0)

    try:
        lam = np.linalg.solve(N, -rhs)
    except np.linalg.LinAlgError:
        lam = np.linalg.lstsq(N, -rhs, rcond=None)[0]

    # Project onto friction cones
    lam = project_contact_friction_cones(lam, friction)
    return lam


def solve_contact_jacobi(
    N: np.ndarray,
    rhs: np.ndarray,
    friction: np.ndarray,
    omega: float,
    relax: float,
    max_iter: int,
    tol: float,
) -> tuple[np.ndarray, int]:
    """
    Solve contact system using Jacobi iterations with friction projection.

    Solves: N * lambda = -rhs

    Args:
        N: System matrix [3*nc x 3*nc].
        rhs: Right-hand side [3*nc].
        friction: Friction coefficients [nc].
        omega: Step size.
        relax: Relaxation factor.
        max_iter: Maximum iterations.
        tol: Convergence tolerance.

    Returns:
        lam: Solution [3*nc].
        iterations: Number of iterations performed.
    """
    nc3 = N.shape[0]
    if nc3 == 0:
        return np.zeros(0), 0

    # Diagonal preconditioner
    diag = np.diag(N)
    D_inv = np.where(diag > 1e-10, 1.0 / diag, 1.0)

    lam = np.zeros(nc3)
    for iteration in range(max_iter):
        lam_old = lam.copy()
        grad = N @ lam + rhs
        lam_new = lam - omega * D_inv * grad

        # Relaxation
        lam = relax * lam_new + (1.0 - relax) * lam

        # Project onto friction cones
        lam = project_contact_friction_cones(lam, friction)

        if np.linalg.norm(lam - lam_old) < tol:
            return lam, iteration + 1

    return lam, max_iter



def project_contact_friction_cones(
    lam: np.ndarray,
    friction: np.ndarray,
) -> np.ndarray:
    """
    Project contact impulses onto friction cones.

    Each contact has 3 components: [fn, ft1, ft2].
    The friction cone constraint is: sqrt(ft1^2 + ft2^2) <= mu * fn.

    Args:
        lam: Contact impulses [3*nc].
        friction: Friction coefficients [nc].

    Returns:
        Projected impulses [3*nc].
    """
    lam_out = lam.copy()
    nc = len(friction)

    for i in range(nc):
        fn = lam_out[3 * i]
        ft1 = lam_out[3 * i + 1]
        ft2 = lam_out[3 * i + 2]
        ft = np.sqrt(ft1**2 + ft2**2)
        mu = friction[i]

        if ft <= mu * fn:
            # Inside cone, no projection needed
            continue

        if mu != 0.0 and ft < -fn / mu:
            # Below cone (pulling)
            lam_out[3 * i : 3 * i + 3] = 0.0
        elif mu == 0.0 and fn < 0:
            # Frictionless: clamp normal to non-negative
            lam_out[3 * i] = 0.0
        elif mu * fn < ft:
            # Project onto cone surface
            fn_new = (fn + mu * ft) / (mu**2 + 1)
            lam_out[3 * i] = fn_new
            if ft > 1e-10:
                scale = mu * fn_new / ft
                lam_out[3 * i + 1] = ft1 * scale
                lam_out[3 * i + 2] = ft2 * scale
            else:
                lam_out[3 * i + 1] = 0.0
                lam_out[3 * i + 2] = 0.0

    return lam_out


# =============================================================================
# SolveContext Integration
# =============================================================================


def build_dense_system_from_context(ctx) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build dense system matrix N from a SolveContext.

    Automatically detects whether the context contains joint or contact data
    and builds the appropriate system matrix.

    Args:
        ctx: SolveContext with J, body_a, body_b, joint_nc, nc_offset, M_inv_diag, M_inv_inertia.
            Contact vs joint is detected by ctx.contact_max > 0.

    Returns:
        Tuple of (J_full, M_inv, N) where:
        - J_full: Full Jacobian matrix
        - M_inv: Block-diagonal inverse mass matrix [6*nb, 6*nb]
        - N: System matrix J @ M_inv @ J^T

    Raises:
        ValueError: If required context fields are missing.
    """
    # Unified format: both joints and contacts use ctx.J with [nc, 12] format
    # Detect contacts by contact_max > 0
    is_contact = ctx.contact_max > 0 and ctx.contact_friction is not None

    if is_contact:
        return _build_contact_system_from_context(ctx)
    else:
        return _build_joint_system_from_context(ctx)


def _build_joint_system_from_context(ctx) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build dense system matrix from joint constraint context."""
    if ctx.J is None or ctx.M_inv_diag is None or ctx.M_inv_inertia is None:
        raise ValueError("Joint context requires ctx.J, ctx.M_inv_diag, ctx.M_inv_inertia")
    if ctx.body_a is None or ctx.body_b is None:
        raise ValueError("Joint context requires ctx.body_a, ctx.body_b")
    if ctx.joint_nc is None or ctx.nc_offset is None:
        raise ValueError("Joint context requires ctx.joint_nc, ctx.nc_offset")

    # Extract numpy arrays from warp arrays
    J = ctx.J.numpy()
    body_a = ctx.body_a.numpy()
    body_b = ctx.body_b.numpy()
    joint_nc = ctx.joint_nc.numpy()
    inv_mass = ctx.M_inv_diag.numpy()
    inv_inertia = ctx.M_inv_inertia.numpy()

    nb = ctx.nb
    nj = ctx.nj

    # Build dense Jacobian
    J_full, _, _ = build_dense_jacobian(J, body_a, body_b, joint_nc, nj, nb)

    # Build inverse mass matrix
    M_inv = build_inverse_mass_matrix(inv_mass, inv_inertia, nb)

    # Build system matrix (no reg here - caller adds it)
    N = build_system_matrix(J_full, M_inv, reg=0.0)

    return J_full, M_inv, N


def _build_contact_system_from_context(ctx) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build dense system matrix from contact constraint context.

    Uses unified [nc, 12] Jacobian format where each contact has 3 rows.
    """
    if ctx.J is None or ctx.M_inv_diag is None or ctx.M_inv_inertia is None:
        raise ValueError("Contact context requires ctx.J, ctx.M_inv_diag, ctx.M_inv_inertia")
    if ctx.body_a is None or ctx.body_b is None:
        raise ValueError("Contact context requires ctx.body_a, ctx.body_b")

    # Extract numpy arrays
    J = ctx.J.numpy()  # [nc, 12] unified format
    body_a = ctx.body_a.numpy()
    body_b = ctx.body_b.numpy()
    joint_nc = ctx.joint_nc.numpy()  # [3, 3, 3, ...] for contacts
    inv_mass = ctx.M_inv_diag.numpy()
    inv_inertia = ctx.M_inv_inertia.numpy()

    nb = ctx.nb
    nj = ctx.nj  # Number of contact blocks

    # Build dense Jacobian using same function as joints
    J_full, _, _ = build_dense_jacobian(J, body_a, body_b, joint_nc, nj, nb)

    # Build inverse mass matrix
    M_inv = build_inverse_mass_matrix(inv_mass, inv_inertia, nb)

    # Build system matrix (no reg here - caller adds it)
    N = build_system_matrix(J_full, M_inv, reg=0.0)

    return J_full, M_inv, N
