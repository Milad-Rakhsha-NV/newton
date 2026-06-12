# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
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

"""Sparse N assembly kernels for the block-sparse tile LDL solver.

Builds the constraint-space system matrix ``N = J M^{-1} J^T + reg I`` directly
into the block-sparse storage layout described by ``BlockSparseSymbolic``:

- ``N_diag[k]`` is the 6x6 diagonal block for pivot ``k`` (with regularization
  added on the live rows and identity padding on the rest so the factorization
  produces unit padding).
- ``N_off[s]`` is the 6x6 off-diagonal block at CSC slot ``s``, holding the
  contributions from the body shared by pivots ``I = N_off_row_idx[s]`` and
  ``J = column-of-slot-s``.

The kernels use a single-threaded loop per pivot/edge to keep the assembly
arithmetic readable; tile operations are reserved for the LDL factorization
where they pay off. The launch grid is ``num_pivots`` (or ``num_edges``), so
parallelism scales with topology size.
"""

from __future__ import annotations

import warp as wp


_BS = 6


@wp.kernel
def assemble_N_diag_kernel(
    joint_jac: wp.array(dtype=wp.float32, ndim=2),
    body_inv_mass: wp.array(dtype=wp.float32),
    body_inv_inertia_world: wp.array(dtype=wp.mat33),
    joint_body_a: wp.array(dtype=wp.int32),
    joint_body_b: wp.array(dtype=wp.int32),
    nc_offset: wp.array(dtype=wp.int32),
    pivot_order: wp.array(dtype=wp.int32),
    block_sizes: wp.array(dtype=wp.int32),
    reg: wp.float32,
    num_pivots: wp.int32,
    # output
    N_diag: wp.array(dtype=wp.float32, ndim=3),
):
    """Assemble each diagonal block ``N_diag[k] = J_k W J_k^T + reg I`` (live rows).

    Padded rows/cols (``r >= block_sizes[k]``) are filled with ``I`` on the
    diagonal so the downstream Cholesky pads to unity for inactive constraints.
    """
    k = wp.tid()
    if k >= num_pivots:
        return

    j_id = pivot_order[k]
    nc = block_sizes[k]
    row0 = nc_offset[j_id]
    body_a = joint_body_a[j_id]
    body_b = joint_body_b[j_id]

    has_a = wp.int32(0)
    inv_m_a = wp.float32(0.0)
    inv_I_a = wp.mat33(0.0)
    if body_a >= 0:
        has_a = wp.int32(1)
        inv_m_a = body_inv_mass[body_a]
        inv_I_a = body_inv_inertia_world[body_a]
    inv_m_b = body_inv_mass[body_b]
    inv_I_b = body_inv_inertia_world[body_b]

    for r in range(_BS):
        for c in range(_BS):
            N_diag[k, r, c] = wp.float32(0.0)

    for r in range(nc):
        Ja_lin_r = wp.vec3(joint_jac[row0 + r, 0], joint_jac[row0 + r, 1], joint_jac[row0 + r, 2])
        Ja_ang_r = wp.vec3(joint_jac[row0 + r, 3], joint_jac[row0 + r, 4], joint_jac[row0 + r, 5])
        Jb_lin_r = wp.vec3(joint_jac[row0 + r, 6], joint_jac[row0 + r, 7], joint_jac[row0 + r, 8])
        Jb_ang_r = wp.vec3(joint_jac[row0 + r, 9], joint_jac[row0 + r, 10], joint_jac[row0 + r, 11])

        for c in range(nc):
            Ja_lin_c = wp.vec3(joint_jac[row0 + c, 0], joint_jac[row0 + c, 1], joint_jac[row0 + c, 2])
            Ja_ang_c = wp.vec3(joint_jac[row0 + c, 3], joint_jac[row0 + c, 4], joint_jac[row0 + c, 5])
            Jb_lin_c = wp.vec3(joint_jac[row0 + c, 6], joint_jac[row0 + c, 7], joint_jac[row0 + c, 8])
            Jb_ang_c = wp.vec3(joint_jac[row0 + c, 9], joint_jac[row0 + c, 10], joint_jac[row0 + c, 11])

            v = wp.float32(0.0)
            if has_a == 1:
                v += inv_m_a * wp.dot(Ja_lin_r, Ja_lin_c)
                v += wp.dot(Ja_ang_r, inv_I_a * Ja_ang_c)
            v += inv_m_b * wp.dot(Jb_lin_r, Jb_lin_c)
            v += wp.dot(Jb_ang_r, inv_I_b * Jb_ang_c)

            if r == c:
                v += reg

            N_diag[k, r, c] = v

    for r in range(nc, _BS):
        N_diag[k, r, r] = wp.float32(1.0)


@wp.kernel
def assemble_N_off_kernel(
    joint_jac: wp.array(dtype=wp.float32, ndim=2),
    body_inv_mass: wp.array(dtype=wp.float32),
    body_inv_inertia_world: wp.array(dtype=wp.mat33),
    joint_body_a: wp.array(dtype=wp.int32),
    joint_body_b: wp.array(dtype=wp.int32),
    nc_offset: wp.array(dtype=wp.int32),
    pivot_order: wp.array(dtype=wp.int32),
    block_sizes: wp.array(dtype=wp.int32),
    N_off_row_idx: wp.array(dtype=wp.int32),
    N_off_col_idx: wp.array(dtype=wp.int32),
    nnz_N: wp.int32,
    # output
    N_off: wp.array(dtype=wp.float32, ndim=3),
):
    """Assemble each off-diagonal block ``N_off[s] = J_I W_shared J_J^T``.

    Determines the shared body by comparing the ``body_a/body_b`` pair of the
    two joints owning row pivot ``I`` and column pivot ``J``. There is exactly
    one such body for every (I, J) edge in the constraint graph (or two, in
    which case the contributions sum linearly).
    """
    s = wp.tid()
    if s >= nnz_N:
        return

    for r in range(_BS):
        for c in range(_BS):
            N_off[s, r, c] = wp.float32(0.0)

    j_pivot = N_off_col_idx[s]
    I = N_off_row_idx[s]

    j_id_I = pivot_order[I]
    j_id_J = pivot_order[j_pivot]

    nc_I = block_sizes[I]
    nc_J = block_sizes[j_pivot]
    row_I = nc_offset[j_id_I]
    row_J = nc_offset[j_id_J]

    a_I = joint_body_a[j_id_I]
    b_I = joint_body_b[j_id_I]
    a_J = joint_body_a[j_id_J]
    b_J = joint_body_b[j_id_J]

    for r in range(nc_I):
        Ja_lin_I = wp.vec3(joint_jac[row_I + r, 0], joint_jac[row_I + r, 1], joint_jac[row_I + r, 2])
        Ja_ang_I = wp.vec3(joint_jac[row_I + r, 3], joint_jac[row_I + r, 4], joint_jac[row_I + r, 5])
        Jb_lin_I = wp.vec3(joint_jac[row_I + r, 6], joint_jac[row_I + r, 7], joint_jac[row_I + r, 8])
        Jb_ang_I = wp.vec3(joint_jac[row_I + r, 9], joint_jac[row_I + r, 10], joint_jac[row_I + r, 11])

        for c in range(nc_J):
            Ja_lin_J = wp.vec3(joint_jac[row_J + c, 0], joint_jac[row_J + c, 1], joint_jac[row_J + c, 2])
            Ja_ang_J = wp.vec3(joint_jac[row_J + c, 3], joint_jac[row_J + c, 4], joint_jac[row_J + c, 5])
            Jb_lin_J = wp.vec3(joint_jac[row_J + c, 6], joint_jac[row_J + c, 7], joint_jac[row_J + c, 8])
            Jb_ang_J = wp.vec3(joint_jac[row_J + c, 9], joint_jac[row_J + c, 10], joint_jac[row_J + c, 11])

            v = wp.float32(0.0)

            if a_I >= 0 and a_I == a_J:
                inv_m = body_inv_mass[a_I]
                inv_I = body_inv_inertia_world[a_I]
                v += inv_m * wp.dot(Ja_lin_I, Ja_lin_J)
                v += wp.dot(Ja_ang_I, inv_I * Ja_ang_J)
            if a_I >= 0 and a_I == b_J:
                inv_m = body_inv_mass[a_I]
                inv_I = body_inv_inertia_world[a_I]
                v += inv_m * wp.dot(Ja_lin_I, Jb_lin_J)
                v += wp.dot(Ja_ang_I, inv_I * Jb_ang_J)
            if b_I == a_J and a_J >= 0:
                inv_m = body_inv_mass[b_I]
                inv_I = body_inv_inertia_world[b_I]
                v += inv_m * wp.dot(Jb_lin_I, Ja_lin_J)
                v += wp.dot(Jb_ang_I, inv_I * Ja_ang_J)
            if b_I == b_J:
                inv_m = body_inv_mass[b_I]
                inv_I = body_inv_inertia_world[b_I]
                v += inv_m * wp.dot(Jb_lin_I, Jb_lin_J)
                v += wp.dot(Jb_ang_I, inv_I * Jb_ang_J)

            N_off[s, r, c] = v


@wp.kernel
def zero_L_blocks_kernel(
    L_blocks: wp.array(dtype=wp.float32, ndim=3),
    nnz_L: wp.int32,
):
    """Zero ``L_blocks`` between solves so fill-only-by-pattern slots stay zero."""
    s = wp.tid()
    if s >= nnz_L:
        return
    for r in range(_BS):
        for c in range(_BS):
            L_blocks[s, r, c] = wp.float32(0.0)


@wp.kernel
def copy_N_off_to_L_kernel(
    N_off: wp.array(dtype=wp.float32, ndim=3),
    N_off_to_L: wp.array(dtype=wp.int32),
    nnz_N: wp.int32,
    # output
    L_blocks: wp.array(dtype=wp.float32, ndim=3),
):
    """Seed ``L_blocks`` with the off-diagonal entries of N for use as the
    starting point of the per-pivot update; pure-fill slots remain zero."""
    s = wp.tid()
    if s >= nnz_N:
        return
    dest = N_off_to_L[s]
    for r in range(_BS):
        for c in range(_BS):
            L_blocks[dest, r, c] = N_off[s, r, c]


# =============================================================================
# Diagonal preconditioning kernels
# =============================================================================
#
# Symmetric diagonal (Jacobi) preconditioning for the Schur complement
# N = J M⁻¹ Jᵀ.  High mass ratios across fixed joints produce diagonal
# entries that span many orders of magnitude, degrading the condition
# number of N and amplifying round-off during the Cholesky factorisation.
#
# The remedy is standard symmetric scaling:
#
#   S = diag(1 / sqrt(N_ii))       (per constraint row)
#   Ñ = S N S                       (unit diagonal, same spectrum up to scaling)
#   Ñ x̃ = S b                       (scaled RHS)
#   x  = S x̃                        (unscale solution)
#
# In block-sparse form the scaling vector is stored per-pivot as a 6-vector
# ``scale_blocks[k, r]``.  Three kernels apply the transformation:
#
#   1. ``compute_precond_scale_kernel`` — reads N_diag, writes scale_blocks.
#   2. ``apply_precond_scale_N_kernel`` — scales N_diag and N_off in place.
#   3. Scaled RHS / unscaled solution are handled by modified gather/scatter
#      kernels that accept the scale vector.


@wp.kernel
def compute_precond_scale_kernel(
    N_diag: wp.array(dtype=wp.float32, ndim=3),
    block_sizes: wp.array(dtype=wp.int32),
    num_pivots: wp.int32,
    # output
    scale_blocks: wp.array(dtype=wp.float32, ndim=2),
):
    """Compute ``scale_blocks[k, r] = 1 / sqrt(N_diag[k, r, r])`` for live rows.

    Padded rows (r >= block_sizes[k]) get scale = 1.0 (identity passthrough).
    A floor of 1e-30 prevents division by zero on degenerate rows.
    """
    k = wp.tid()
    if k >= num_pivots:
        return
    nc = block_sizes[k]
    for r in range(_BS):
        if r < nc:
            d = N_diag[k, r, r]
            # Floor to avoid sqrt(0) or negative values from round-off
            if d < 1.0e-30:
                d = 1.0e-30
            scale_blocks[k, r] = 1.0 / wp.sqrt(d)
        else:
            scale_blocks[k, r] = 1.0


@wp.kernel
def apply_precond_scale_N_diag_kernel(
    scale_blocks: wp.array(dtype=wp.float32, ndim=2),
    num_pivots: wp.int32,
    # input/output
    N_diag: wp.array(dtype=wp.float32, ndim=3),
):
    """Scale diagonal blocks in place: ``N_diag[k, r, c] *= s[k,r] * s[k,c]``."""
    k = wp.tid()
    if k >= num_pivots:
        return
    for r in range(_BS):
        sr = scale_blocks[k, r]
        for c in range(_BS):
            sc = scale_blocks[k, c]
            N_diag[k, r, c] = N_diag[k, r, c] * sr * sc


@wp.kernel
def apply_precond_scale_N_off_kernel(
    scale_blocks: wp.array(dtype=wp.float32, ndim=2),
    N_off_row_idx: wp.array(dtype=wp.int32),
    N_off_col_idx: wp.array(dtype=wp.int32),
    nnz_N: wp.int32,
    # input/output
    N_off: wp.array(dtype=wp.float32, ndim=3),
):
    """Scale off-diagonal blocks: ``N_off[s, r, c] *= s[I,r] * s[J,c]``.

    I = N_off_row_idx[s] (row pivot), J = N_off_col_idx[s] (column pivot).
    """
    s = wp.tid()
    if s >= nnz_N:
        return
    I = N_off_row_idx[s]
    J = N_off_col_idx[s]
    for r in range(_BS):
        sr = scale_blocks[I, r]
        for c in range(_BS):
            sc = scale_blocks[J, c]
            N_off[s, r, c] = N_off[s, r, c] * sr * sc


@wp.kernel
def apply_precond_scale_rhs_kernel(
    scale_blocks: wp.array(dtype=wp.float32, ndim=2),
    num_pivots: wp.int32,
    # input/output  (rhs_blocks already gathered in pivot order)
    rhs_blocks: wp.array(dtype=wp.float32, ndim=2),
):
    """Scale RHS in place: ``rhs_blocks[k, r] *= s[k, r]``."""
    k = wp.tid()
    if k >= num_pivots:
        return
    for r in range(_BS):
        rhs_blocks[k, r] = rhs_blocks[k, r] * scale_blocks[k, r]


@wp.kernel
def apply_precond_unscale_x_kernel(
    scale_blocks: wp.array(dtype=wp.float32, ndim=2),
    num_pivots: wp.int32,
    # input/output  (x_blocks from backward sub, before scatter)
    x_blocks: wp.array(dtype=wp.float32, ndim=2),
):
    """Unscale solution: ``x_blocks[k, r] *= s[k, r]``.

    Because ``S N S * x̃ = S b`` ⟹ ``x = S x̃``, the unscaling
    uses the same scale vector (symmetric scaling).
    """
    k = wp.tid()
    if k >= num_pivots:
        return
    for r in range(_BS):
        x_blocks[k, r] = x_blocks[k, r] * scale_blocks[k, r]


@wp.kernel
def apply_precond_add_reg_kernel(
    block_sizes: wp.array(dtype=wp.int32),
    reg: wp.float32,
    num_pivots: wp.int32,
    # input/output
    N_diag: wp.array(dtype=wp.float32, ndim=3),
):
    """Add regularization to the scaled diagonal: ``N_diag[k, r, r] += reg`` for live rows.

    After diagonal preconditioning scales all diagonal entries to ~1.0, the
    original ``reg`` (added during assembly relative to the unscaled system)
    has been scaled down to ``reg * s_k[r]^2`` which can be negligible.
    This kernel adds fresh regularization relative to the scaled system,
    improving pivot stability during LDL factorization.
    """
    k = wp.tid()
    if k >= num_pivots:
        return
    nc = block_sizes[k]
    for r in range(_BS):
        if r < nc:
            N_diag[k, r, r] = N_diag[k, r, r] + reg


__all__ = [
    "assemble_N_diag_kernel",
    "assemble_N_off_kernel",
    "zero_L_blocks_kernel",
    "copy_N_off_to_L_kernel",
    "compute_precond_scale_kernel",
    "apply_precond_scale_N_diag_kernel",
    "apply_precond_scale_N_off_kernel",
    "apply_precond_scale_rhs_kernel",
    "apply_precond_unscale_x_kernel",
    "apply_precond_add_reg_kernel",
]
