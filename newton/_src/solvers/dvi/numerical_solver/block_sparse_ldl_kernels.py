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

"""Block-sparse LDL/Cholesky factorization and triangular solves on Warp tiles.

For a positive-definite dvi constraint system ``N = J M^{-1} J^T + reg I``,
LDL with positive diagonal collapses to Cholesky. We exploit this and factor
``N = M M^T`` in block-sparse form, storing each diagonal block as its lower
triangular Cholesky factor in ``D_blocks`` and each off-diagonal block as
``M[I,J]`` in ``L_blocks``.

Left-looking factorization for pivot J:

    1. ``D[J] = N[J,J] - sum_K M[J,K] M[J,K]^T`` (over predecessor K, K < J)
       ``M[J,J] = chol(D[J])`` in place
    2. for each I in column J of L:
       ``M[I,J] = (N[I,J] - sum_K M[I,K] M[J,K]^T) @ M[J,J]^{-T}``

Pivots at the same elimination tree depth are independent. Within a depth
level, this kernel launches one tile-block per pivot.

All numeric block operations are 6x6 ``wp.tile_*`` ops. Live block sizes
``< 6`` are masked by identity-padding the diagonal block during N assembly so
the resulting Cholesky factor is unit on the padded rows and produces zero
off-diagonal contributions.
"""

from __future__ import annotations

import warp as wp

_BS = 6


@wp.kernel
def ldl_factor_level_kernel(
    L_col_ptr: wp.array[wp.int32],
    L_row_idx: wp.array[wp.int32],
    pred_diag_ptr: wp.array[wp.int32],
    pred_diag_slot: wp.array[wp.int32],
    pred_off_ptr: wp.array[wp.int32],
    pred_off_slot_IK: wp.array[wp.int32],
    pred_off_slot_JK: wp.array[wp.int32],
    level_pivots: wp.array[wp.int32],
    level_count: wp.int32,
    level_offset: wp.int32,
    N_diag: wp.array3d[wp.float32],
    # input/output (factor written in place)
    D_blocks: wp.array3d[wp.float32],
    L_blocks: wp.array3d[wp.float32],
):
    """Factor every pivot at this elimination tree level.

    Pivots within a single elimination tree depth share no write targets, so
    one ``wp.launch_tiled`` over ``level_count`` is race-free. Predecessor
    columns are at strictly higher depth and are already factored. ``ptr`` /
    ``slot`` indices are precomputed in symbolic so the kernel does no
    pattern lookups at runtime.
    """
    local = wp.tid()
    if local >= level_count:
        return
    J = level_pivots[level_offset + local]

    diag_3d = wp.tile_load(N_diag, shape=(1, _BS, _BS), offset=(J, 0, 0))
    diag = wp.tile_reshape(diag_3d, (_BS, _BS))

    pd_start = pred_diag_ptr[J]
    pd_end = pred_diag_ptr[J + 1]
    for p in range(pd_start, pd_end):
        slot = pred_diag_slot[p]
        M_JK_3d = wp.tile_load(L_blocks, shape=(1, _BS, _BS), offset=(slot, 0, 0))
        M_JK = wp.tile_reshape(M_JK_3d, (_BS, _BS))
        M_JK_T = wp.tile_transpose(M_JK)
        wp.tile_matmul(M_JK, M_JK_T, diag, alpha=-1.0)

    wp.tile_cholesky_inplace(diag)
    diag_out = wp.tile_reshape(diag, (1, _BS, _BS))
    wp.tile_store(D_blocks, diag_out, offset=(J, 0, 0))

    col_start = L_col_ptr[J]
    col_end = L_col_ptr[J + 1]
    for ptr in range(col_start, col_end):
        block_3d = wp.tile_load(L_blocks, shape=(1, _BS, _BS), offset=(ptr, 0, 0))
        block = wp.tile_reshape(block_3d, (_BS, _BS))

        po_start = pred_off_ptr[ptr]
        po_end = pred_off_ptr[ptr + 1]
        for p in range(po_start, po_end):
            slot_IK = pred_off_slot_IK[p]
            slot_JK = pred_off_slot_JK[p]
            M_IK_3d = wp.tile_load(L_blocks, shape=(1, _BS, _BS), offset=(slot_IK, 0, 0))
            M_IK = wp.tile_reshape(M_IK_3d, (_BS, _BS))
            M_JK_3d2 = wp.tile_load(L_blocks, shape=(1, _BS, _BS), offset=(slot_JK, 0, 0))
            M_JK2 = wp.tile_reshape(M_JK_3d2, (_BS, _BS))
            M_JK_T2 = wp.tile_transpose(M_JK2)
            wp.tile_matmul(M_IK, M_JK_T2, block, alpha=-1.0)

        block_T = wp.tile_transpose(block)
        wp.tile_lower_solve_inplace(diag, block_T)
        block_solved = wp.tile_transpose(block_T)
        block_out = wp.tile_reshape(block_solved, (1, _BS, _BS))
        wp.tile_store(L_blocks, block_out, offset=(ptr, 0, 0))


@wp.kernel
def gather_rhs_in_pivot_order_kernel(
    b: wp.array[wp.float32],
    nc_offset: wp.array[wp.int32],
    pivot_order: wp.array[wp.int32],
    block_sizes: wp.array[wp.int32],
    num_pivots: wp.int32,
    # output
    rhs_blocks: wp.array2d[wp.float32],
):
    """Gather ``ctx.b`` into per-pivot 6-vectors padded with zero."""
    k = wp.tid()
    if k >= num_pivots:
        return
    j_id = pivot_order[k]
    nc = block_sizes[k]
    row0 = nc_offset[j_id]
    for r in range(_BS):
        if r < nc:
            rhs_blocks[k, r] = b[row0 + r]
        else:
            rhs_blocks[k, r] = wp.float32(0.0)


@wp.kernel
def block_forward_sub_level_kernel(
    L_row_ptr: wp.array[wp.int32],
    L_col_idx: wp.array[wp.int32],
    L_csr_to_csc: wp.array[wp.int32],
    level_pivots: wp.array[wp.int32],
    level_count: wp.int32,
    level_offset: wp.int32,
    L_blocks: wp.array3d[wp.float32],
    D_blocks: wp.array3d[wp.float32],
    rhs_blocks: wp.array2d[wp.float32],
    # output
    y_blocks: wp.array2d[wp.float32],
):
    """Forward substitution at one level: solve ``M y = rhs`` per pivot.

    Per row I: ``y_I = M[I,I]^{-1} (rhs[I] - sum_{J<I} M[I,J] y[J])``.
    Same level scheduling as the factor: pivots at the same depth share no
    writes and only read from columns at greater depth (already solved).
    """
    local = wp.tid()
    if local >= level_count:
        return
    I = level_pivots[level_offset + local]

    rhs_3d = wp.tile_load(rhs_blocks, shape=(1, _BS), offset=(I, 0))
    rhs = wp.tile_reshape(rhs_3d, (_BS, 1))

    rs = L_row_ptr[I]
    re = L_row_ptr[I + 1]
    for ridx in range(rs, re):
        J = L_col_idx[ridx]
        slot = L_csr_to_csc[ridx]
        M_IJ_3d = wp.tile_load(L_blocks, shape=(1, _BS, _BS), offset=(slot, 0, 0))
        M_IJ = wp.tile_reshape(M_IJ_3d, (_BS, _BS))
        y_J_3d = wp.tile_load(y_blocks, shape=(1, _BS), offset=(J, 0))
        y_J = wp.tile_reshape(y_J_3d, (_BS, 1))
        wp.tile_matmul(M_IJ, y_J, rhs, alpha=-1.0)

    M_II_3d = wp.tile_load(D_blocks, shape=(1, _BS, _BS), offset=(I, 0, 0))
    M_II = wp.tile_reshape(M_II_3d, (_BS, _BS))
    wp.tile_lower_solve_inplace(M_II, rhs)

    rhs_out_2d = wp.tile_reshape(rhs, (1, _BS))
    wp.tile_store(y_blocks, rhs_out_2d, offset=(I, 0))


@wp.kernel
def block_backward_sub_level_kernel(
    L_col_ptr: wp.array[wp.int32],
    L_row_idx: wp.array[wp.int32],
    level_pivots: wp.array[wp.int32],
    level_count: wp.int32,
    level_offset: wp.int32,
    L_blocks: wp.array3d[wp.float32],
    D_blocks: wp.array3d[wp.float32],
    y_blocks: wp.array2d[wp.float32],
    # output
    x_blocks: wp.array2d[wp.float32],
):
    """Backward substitution at one level: solve ``M^T x = y`` per pivot.

    Per column J: ``x_J = M[J,J]^{-T} (y[J] - sum_{I>J} M[I,J]^T x[I])``.
    Run levels from depth 0 (roots, processed last in factor) back to highest
    depth (leaves) so the read dependencies on x[I] for I > J are satisfied.
    """
    local = wp.tid()
    if local >= level_count:
        return
    J = level_pivots[level_offset + local]

    y_3d = wp.tile_load(y_blocks, shape=(1, _BS), offset=(J, 0))
    y_J = wp.tile_reshape(y_3d, (_BS, 1))

    cs = L_col_ptr[J]
    ce = L_col_ptr[J + 1]
    for ptr in range(cs, ce):
        I = L_row_idx[ptr]
        M_IJ_3d = wp.tile_load(L_blocks, shape=(1, _BS, _BS), offset=(ptr, 0, 0))
        M_IJ = wp.tile_reshape(M_IJ_3d, (_BS, _BS))
        M_IJ_T = wp.tile_transpose(M_IJ)
        x_I_3d = wp.tile_load(x_blocks, shape=(1, _BS), offset=(I, 0))
        x_I = wp.tile_reshape(x_I_3d, (_BS, 1))
        wp.tile_matmul(M_IJ_T, x_I, y_J, alpha=-1.0)

    M_JJ_3d = wp.tile_load(D_blocks, shape=(1, _BS, _BS), offset=(J, 0, 0))
    M_JJ = wp.tile_reshape(M_JJ_3d, (_BS, _BS))
    M_JJ_T = wp.tile_transpose(M_JJ)
    wp.tile_upper_solve_inplace(M_JJ_T, y_J)

    out_2d = wp.tile_reshape(y_J, (1, _BS))
    wp.tile_store(x_blocks, out_2d, offset=(J, 0))


@wp.kernel
def scatter_x_kernel(
    x_blocks: wp.array2d[wp.float32],
    pivot_order: wp.array[wp.int32],
    nc_offset: wp.array[wp.int32],
    block_sizes: wp.array[wp.int32],
    num_pivots: wp.int32,
    # output
    out: wp.array[wp.float32],
):
    """Write per-pivot solution back into the row-major ``ctx.x`` array."""
    k = wp.tid()
    if k >= num_pivots:
        return
    j_id = pivot_order[k]
    row0 = nc_offset[j_id]
    nc = block_sizes[k]
    for r in range(nc):
        out[row0 + r] = x_blocks[k, r]


@wp.kernel
def compute_refinement_residual_kernel(
    joint_jac: wp.array2d[wp.float32],
    joint_body_a: wp.array[wp.int32],
    joint_body_b: wp.array[wp.int32],
    block_nc_per_joint: wp.array[wp.int32],
    nc_offset: wp.array[wp.int32],
    b: wp.array[wp.float32],
    delta_v: wp.array[wp.spatial_vector],
    num_joints: wp.int32,
    # output
    residual: wp.array[wp.float32],
):
    """Compute per-row residual ``r[c] = b[c] - J[c] @ delta_v`` for iterative refinement.

    This is a matrix-free residual: instead of explicitly forming N and
    computing ``b - N @ x``, we use the identity ``N @ x = J M⁻¹ Jᵀ x``
    and the pre-computed ``delta_v = M⁻¹ Jᵀ x`` to get ``J @ delta_v``.
    Then ``r = b - J @ delta_v``.

    One thread per joint; iterates over the joint's constraint rows.
    """
    j = wp.tid()
    if j >= num_joints:
        return
    nc = block_nc_per_joint[j]
    if nc == 0:
        return
    body_a = joint_body_a[j]
    body_b = joint_body_b[j]
    row0 = nc_offset[j]

    # Load delta_v for both bodies
    dv_a = wp.spatial_vector()
    if body_a >= 0:
        dv_a = delta_v[body_a]
    dv_b = delta_v[body_b]

    for c in range(nc):
        row = row0 + c
        # J @ delta_v for this row
        J_dv = wp.float32(0.0)
        if body_a >= 0:
            Ja_lin = wp.vec3(joint_jac[row, 0], joint_jac[row, 1], joint_jac[row, 2])
            Ja_ang = wp.vec3(joint_jac[row, 3], joint_jac[row, 4], joint_jac[row, 5])
            J_dv += wp.dot(Ja_lin, wp.spatial_top(dv_a)) + wp.dot(Ja_ang, wp.spatial_bottom(dv_a))
        Jb_lin = wp.vec3(joint_jac[row, 6], joint_jac[row, 7], joint_jac[row, 8])
        Jb_ang = wp.vec3(joint_jac[row, 9], joint_jac[row, 10], joint_jac[row, 11])
        J_dv += wp.dot(Jb_lin, wp.spatial_top(dv_b)) + wp.dot(Jb_ang, wp.spatial_bottom(dv_b))

        # r = b - J @ delta_v  (= b - N @ x)
        residual[row] = b[row] - J_dv


@wp.kernel
def scatter_x_add_kernel(
    x_blocks: wp.array2d[wp.float32],
    pivot_order: wp.array[wp.int32],
    nc_offset: wp.array[wp.int32],
    block_sizes: wp.array[wp.int32],
    num_pivots: wp.int32,
    # input/output
    out: wp.array[wp.float32],
):
    """Scatter-add per-pivot correction into the flat ``ctx.x`` array.

    Like ``scatter_x_kernel`` but *adds* the correction instead of
    overwriting: ``out[row] += x_blocks[k, r]``.  Used by iterative
    refinement to accumulate the correction onto the current solution.
    """
    k = wp.tid()
    if k >= num_pivots:
        return
    j_id = pivot_order[k]
    row0 = nc_offset[j_id]
    nc = block_sizes[k]
    for r in range(nc):
        out[row0 + r] = out[row0 + r] + x_blocks[k, r]


@wp.kernel
def compute_delta_v_from_lambda_kernel(
    body_inv_mass: wp.array[wp.float32],
    body_inv_inertia_world: wp.array[wp.mat33],
    joint_jac: wp.array2d[wp.float32],
    joint_body_a: wp.array[wp.int32],
    joint_body_b: wp.array[wp.int32],
    block_nc_per_joint: wp.array[wp.int32],
    nc_offset: wp.array[wp.int32],
    joint_lambda: wp.array[wp.float32],
    num_joints: wp.int32,
    # output
    delta_v: wp.array[wp.spatial_vector],
):
    """Compute ``delta_v = M^{-1} J^T lambda`` per joint, accumulated atomically."""
    j = wp.tid()
    if j >= num_joints:
        return
    nc = block_nc_per_joint[j]
    if nc == 0:
        return
    body_a = joint_body_a[j]
    body_b = joint_body_b[j]
    row0 = nc_offset[j]

    dv_a_lin = wp.vec3(0.0, 0.0, 0.0)
    dv_a_ang = wp.vec3(0.0, 0.0, 0.0)
    dv_b_lin = wp.vec3(0.0, 0.0, 0.0)
    dv_b_ang = wp.vec3(0.0, 0.0, 0.0)

    for c in range(nc):
        lam = joint_lambda[row0 + c]
        if body_a >= 0:
            Ja_lin = wp.vec3(joint_jac[row0 + c, 0], joint_jac[row0 + c, 1], joint_jac[row0 + c, 2])
            Ja_ang = wp.vec3(joint_jac[row0 + c, 3], joint_jac[row0 + c, 4], joint_jac[row0 + c, 5])
            dv_a_lin += Ja_lin * lam
            dv_a_ang += Ja_ang * lam
        Jb_lin = wp.vec3(joint_jac[row0 + c, 6], joint_jac[row0 + c, 7], joint_jac[row0 + c, 8])
        Jb_ang = wp.vec3(joint_jac[row0 + c, 9], joint_jac[row0 + c, 10], joint_jac[row0 + c, 11])
        dv_b_lin += Jb_lin * lam
        dv_b_ang += Jb_ang * lam

    if body_a >= 0:
        inv_m_a = body_inv_mass[body_a]
        inv_I_a = body_inv_inertia_world[body_a]
        dv_a = wp.spatial_vector(
            dv_a_lin * inv_m_a,
            inv_I_a * dv_a_ang,
        )
        wp.atomic_add(delta_v, body_a, dv_a)
    inv_m_b = body_inv_mass[body_b]
    inv_I_b = body_inv_inertia_world[body_b]
    dv_b = wp.spatial_vector(
        dv_b_lin * inv_m_b,
        inv_I_b * dv_b_ang,
    )
    wp.atomic_add(delta_v, body_b, dv_b)


__all__ = [
    "block_backward_sub_level_kernel",
    "block_forward_sub_level_kernel",
    "compute_delta_v_from_lambda_kernel",
    "compute_refinement_residual_kernel",
    "gather_rhs_in_pivot_order_kernel",
    "ldl_factor_level_kernel",
    "scatter_x_add_kernel",
    "scatter_x_kernel",
]
