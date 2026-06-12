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

"""Batched (multi-world) twins of the block-sparse tile LDL kernels.

Symbolic arrays are shared across worlds (single allocation in
``BatchedBlockSparseStorage``). Numeric and work arrays carry a leading world
dimension. Every kernel uses 2D ``wp.tid()`` over ``(world, item)`` so a
single ``wp.launch_tiled`` covers all worlds at one elimination tree depth.
"""

from __future__ import annotations

import warp as wp


_BS = 6


@wp.kernel
def batched_assemble_N_diag_kernel(
    joint_jac: wp.array(dtype=wp.float32, ndim=3),
    body_inv_mass: wp.array(dtype=wp.float32, ndim=2),
    body_inv_inertia_world: wp.array(dtype=wp.mat33, ndim=2),
    joint_body_a: wp.array(dtype=wp.int32),
    joint_body_b: wp.array(dtype=wp.int32),
    nc_offset: wp.array(dtype=wp.int32),
    pivot_order: wp.array(dtype=wp.int32),
    block_sizes: wp.array(dtype=wp.int32),
    reg: wp.float32,
    num_pivots: wp.int32,
    # output
    N_diag: wp.array(dtype=wp.float32, ndim=4),
):
    """Assemble per-world ``N_diag[w, k]``."""
    w, k = wp.tid()
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
        inv_m_a = body_inv_mass[w, body_a]
        inv_I_a = body_inv_inertia_world[w, body_a]
    inv_m_b = body_inv_mass[w, body_b]
    inv_I_b = body_inv_inertia_world[w, body_b]

    for r in range(_BS):
        for c in range(_BS):
            N_diag[w, k, r, c] = wp.float32(0.0)

    for r in range(nc):
        Ja_lin_r = wp.vec3(joint_jac[w, row0 + r, 0], joint_jac[w, row0 + r, 1], joint_jac[w, row0 + r, 2])
        Ja_ang_r = wp.vec3(joint_jac[w, row0 + r, 3], joint_jac[w, row0 + r, 4], joint_jac[w, row0 + r, 5])
        Jb_lin_r = wp.vec3(joint_jac[w, row0 + r, 6], joint_jac[w, row0 + r, 7], joint_jac[w, row0 + r, 8])
        Jb_ang_r = wp.vec3(joint_jac[w, row0 + r, 9], joint_jac[w, row0 + r, 10], joint_jac[w, row0 + r, 11])
        for c in range(nc):
            Ja_lin_c = wp.vec3(joint_jac[w, row0 + c, 0], joint_jac[w, row0 + c, 1], joint_jac[w, row0 + c, 2])
            Ja_ang_c = wp.vec3(joint_jac[w, row0 + c, 3], joint_jac[w, row0 + c, 4], joint_jac[w, row0 + c, 5])
            Jb_lin_c = wp.vec3(joint_jac[w, row0 + c, 6], joint_jac[w, row0 + c, 7], joint_jac[w, row0 + c, 8])
            Jb_ang_c = wp.vec3(joint_jac[w, row0 + c, 9], joint_jac[w, row0 + c, 10], joint_jac[w, row0 + c, 11])
            v = wp.float32(0.0)
            if has_a == 1:
                v += inv_m_a * wp.dot(Ja_lin_r, Ja_lin_c)
                v += wp.dot(Ja_ang_r, inv_I_a * Ja_ang_c)
            v += inv_m_b * wp.dot(Jb_lin_r, Jb_lin_c)
            v += wp.dot(Jb_ang_r, inv_I_b * Jb_ang_c)
            if r == c:
                v += reg
            N_diag[w, k, r, c] = v

    for r in range(nc, _BS):
        N_diag[w, k, r, r] = wp.float32(1.0)


@wp.kernel
def batched_assemble_N_off_kernel(
    joint_jac: wp.array(dtype=wp.float32, ndim=3),
    body_inv_mass: wp.array(dtype=wp.float32, ndim=2),
    body_inv_inertia_world: wp.array(dtype=wp.mat33, ndim=2),
    joint_body_a: wp.array(dtype=wp.int32),
    joint_body_b: wp.array(dtype=wp.int32),
    nc_offset: wp.array(dtype=wp.int32),
    pivot_order: wp.array(dtype=wp.int32),
    block_sizes: wp.array(dtype=wp.int32),
    N_off_row_idx: wp.array(dtype=wp.int32),
    N_off_col_idx: wp.array(dtype=wp.int32),
    nnz_N: wp.int32,
    # output
    N_off: wp.array(dtype=wp.float32, ndim=4),
):
    """Assemble per-world ``N_off[w, s]``."""
    w, s = wp.tid()
    if s >= nnz_N:
        return

    for r in range(_BS):
        for c in range(_BS):
            N_off[w, s, r, c] = wp.float32(0.0)

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
        Ja_lin_I = wp.vec3(joint_jac[w, row_I + r, 0], joint_jac[w, row_I + r, 1], joint_jac[w, row_I + r, 2])
        Ja_ang_I = wp.vec3(joint_jac[w, row_I + r, 3], joint_jac[w, row_I + r, 4], joint_jac[w, row_I + r, 5])
        Jb_lin_I = wp.vec3(joint_jac[w, row_I + r, 6], joint_jac[w, row_I + r, 7], joint_jac[w, row_I + r, 8])
        Jb_ang_I = wp.vec3(joint_jac[w, row_I + r, 9], joint_jac[w, row_I + r, 10], joint_jac[w, row_I + r, 11])
        for c in range(nc_J):
            Ja_lin_J = wp.vec3(joint_jac[w, row_J + c, 0], joint_jac[w, row_J + c, 1], joint_jac[w, row_J + c, 2])
            Ja_ang_J = wp.vec3(joint_jac[w, row_J + c, 3], joint_jac[w, row_J + c, 4], joint_jac[w, row_J + c, 5])
            Jb_lin_J = wp.vec3(joint_jac[w, row_J + c, 6], joint_jac[w, row_J + c, 7], joint_jac[w, row_J + c, 8])
            Jb_ang_J = wp.vec3(joint_jac[w, row_J + c, 9], joint_jac[w, row_J + c, 10], joint_jac[w, row_J + c, 11])
            v = wp.float32(0.0)
            if a_I >= 0 and a_I == a_J:
                inv_m = body_inv_mass[w, a_I]
                inv_I = body_inv_inertia_world[w, a_I]
                v += inv_m * wp.dot(Ja_lin_I, Ja_lin_J)
                v += wp.dot(Ja_ang_I, inv_I * Ja_ang_J)
            if a_I >= 0 and a_I == b_J:
                inv_m = body_inv_mass[w, a_I]
                inv_I = body_inv_inertia_world[w, a_I]
                v += inv_m * wp.dot(Ja_lin_I, Jb_lin_J)
                v += wp.dot(Ja_ang_I, inv_I * Jb_ang_J)
            if b_I == a_J and a_J >= 0:
                inv_m = body_inv_mass[w, b_I]
                inv_I = body_inv_inertia_world[w, b_I]
                v += inv_m * wp.dot(Jb_lin_I, Ja_lin_J)
                v += wp.dot(Jb_ang_I, inv_I * Ja_ang_J)
            if b_I == b_J:
                inv_m = body_inv_mass[w, b_I]
                inv_I = body_inv_inertia_world[w, b_I]
                v += inv_m * wp.dot(Jb_lin_I, Jb_lin_J)
                v += wp.dot(Jb_ang_I, inv_I * Jb_ang_J)
            N_off[w, s, r, c] = v


@wp.kernel
def batched_zero_L_blocks_kernel(
    L_blocks: wp.array(dtype=wp.float32, ndim=4),
    nnz_L: wp.int32,
):
    w, s = wp.tid()
    if s >= nnz_L:
        return
    for r in range(_BS):
        for c in range(_BS):
            L_blocks[w, s, r, c] = wp.float32(0.0)


@wp.kernel
def batched_copy_N_off_to_L_kernel(
    N_off: wp.array(dtype=wp.float32, ndim=4),
    N_off_to_L: wp.array(dtype=wp.int32),
    nnz_N: wp.int32,
    # output
    L_blocks: wp.array(dtype=wp.float32, ndim=4),
):
    w, s = wp.tid()
    if s >= nnz_N:
        return
    dest = N_off_to_L[s]
    for r in range(_BS):
        for c in range(_BS):
            L_blocks[w, dest, r, c] = N_off[w, s, r, c]


@wp.kernel
def batched_ldl_factor_level_kernel(
    L_col_ptr: wp.array(dtype=wp.int32),
    L_row_idx: wp.array(dtype=wp.int32),
    pred_diag_ptr: wp.array(dtype=wp.int32),
    pred_diag_slot: wp.array(dtype=wp.int32),
    pred_off_ptr: wp.array(dtype=wp.int32),
    pred_off_slot_IK: wp.array(dtype=wp.int32),
    pred_off_slot_JK: wp.array(dtype=wp.int32),
    level_pivots: wp.array(dtype=wp.int32),
    level_count: wp.int32,
    level_offset: wp.int32,
    N_diag: wp.array(dtype=wp.float32, ndim=4),
    D_blocks: wp.array(dtype=wp.float32, ndim=4),
    L_blocks: wp.array(dtype=wp.float32, ndim=4),
):
    w, local = wp.tid()
    if local >= level_count:
        return
    J = level_pivots[level_offset + local]

    diag_4d = wp.tile_load(N_diag, shape=(1, 1, _BS, _BS), offset=(w, J, 0, 0))
    diag = wp.tile_reshape(diag_4d, (_BS, _BS))

    pd_start = pred_diag_ptr[J]
    pd_end = pred_diag_ptr[J + 1]
    for p in range(pd_start, pd_end):
        slot = pred_diag_slot[p]
        M_JK_4d = wp.tile_load(L_blocks, shape=(1, 1, _BS, _BS), offset=(w, slot, 0, 0))
        M_JK = wp.tile_reshape(M_JK_4d, (_BS, _BS))
        M_JK_T = wp.tile_transpose(M_JK)
        wp.tile_matmul(M_JK, M_JK_T, diag, alpha=-1.0)

    wp.tile_cholesky_inplace(diag)
    diag_out = wp.tile_reshape(diag, (1, 1, _BS, _BS))
    wp.tile_store(D_blocks, diag_out, offset=(w, J, 0, 0))

    col_start = L_col_ptr[J]
    col_end = L_col_ptr[J + 1]
    for ptr in range(col_start, col_end):
        block_4d = wp.tile_load(L_blocks, shape=(1, 1, _BS, _BS), offset=(w, ptr, 0, 0))
        block = wp.tile_reshape(block_4d, (_BS, _BS))

        po_start = pred_off_ptr[ptr]
        po_end = pred_off_ptr[ptr + 1]
        for p in range(po_start, po_end):
            slot_IK = pred_off_slot_IK[p]
            slot_JK = pred_off_slot_JK[p]
            M_IK_4d = wp.tile_load(L_blocks, shape=(1, 1, _BS, _BS), offset=(w, slot_IK, 0, 0))
            M_IK = wp.tile_reshape(M_IK_4d, (_BS, _BS))
            M_JK_4d2 = wp.tile_load(L_blocks, shape=(1, 1, _BS, _BS), offset=(w, slot_JK, 0, 0))
            M_JK2 = wp.tile_reshape(M_JK_4d2, (_BS, _BS))
            M_JK_T2 = wp.tile_transpose(M_JK2)
            wp.tile_matmul(M_IK, M_JK_T2, block, alpha=-1.0)

        block_T = wp.tile_transpose(block)
        wp.tile_lower_solve_inplace(diag, block_T)
        block_solved = wp.tile_transpose(block_T)
        block_out = wp.tile_reshape(block_solved, (1, 1, _BS, _BS))
        wp.tile_store(L_blocks, block_out, offset=(w, ptr, 0, 0))


@wp.kernel
def batched_gather_rhs_kernel(
    b: wp.array(dtype=wp.float32, ndim=2),
    nc_offset: wp.array(dtype=wp.int32),
    pivot_order: wp.array(dtype=wp.int32),
    block_sizes: wp.array(dtype=wp.int32),
    num_pivots: wp.int32,
    rhs_blocks: wp.array(dtype=wp.float32, ndim=3),
):
    w, k = wp.tid()
    if k >= num_pivots:
        return
    j_id = pivot_order[k]
    nc = block_sizes[k]
    row0 = nc_offset[j_id]
    for r in range(_BS):
        if r < nc:
            rhs_blocks[w, k, r] = b[w, row0 + r]
        else:
            rhs_blocks[w, k, r] = wp.float32(0.0)


@wp.kernel
def batched_block_forward_sub_kernel(
    L_row_ptr: wp.array(dtype=wp.int32),
    L_col_idx: wp.array(dtype=wp.int32),
    L_csr_to_csc: wp.array(dtype=wp.int32),
    level_pivots: wp.array(dtype=wp.int32),
    level_count: wp.int32,
    level_offset: wp.int32,
    L_blocks: wp.array(dtype=wp.float32, ndim=4),
    D_blocks: wp.array(dtype=wp.float32, ndim=4),
    rhs_blocks: wp.array(dtype=wp.float32, ndim=3),
    y_blocks: wp.array(dtype=wp.float32, ndim=3),
):
    w, local = wp.tid()
    if local >= level_count:
        return
    I = level_pivots[level_offset + local]

    rhs_3d = wp.tile_load(rhs_blocks, shape=(1, 1, _BS), offset=(w, I, 0))
    rhs = wp.tile_reshape(rhs_3d, (_BS, 1))

    rs = L_row_ptr[I]
    re = L_row_ptr[I + 1]
    for ridx in range(rs, re):
        J = L_col_idx[ridx]
        slot = L_csr_to_csc[ridx]
        M_IJ_4d = wp.tile_load(L_blocks, shape=(1, 1, _BS, _BS), offset=(w, slot, 0, 0))
        M_IJ = wp.tile_reshape(M_IJ_4d, (_BS, _BS))
        y_J_3d = wp.tile_load(y_blocks, shape=(1, 1, _BS), offset=(w, J, 0))
        y_J = wp.tile_reshape(y_J_3d, (_BS, 1))
        wp.tile_matmul(M_IJ, y_J, rhs, alpha=-1.0)

    M_II_4d = wp.tile_load(D_blocks, shape=(1, 1, _BS, _BS), offset=(w, I, 0, 0))
    M_II = wp.tile_reshape(M_II_4d, (_BS, _BS))
    wp.tile_lower_solve_inplace(M_II, rhs)

    out = wp.tile_reshape(rhs, (1, 1, _BS))
    wp.tile_store(y_blocks, out, offset=(w, I, 0))


@wp.kernel
def batched_block_backward_sub_kernel(
    L_col_ptr: wp.array(dtype=wp.int32),
    L_row_idx: wp.array(dtype=wp.int32),
    level_pivots: wp.array(dtype=wp.int32),
    level_count: wp.int32,
    level_offset: wp.int32,
    L_blocks: wp.array(dtype=wp.float32, ndim=4),
    D_blocks: wp.array(dtype=wp.float32, ndim=4),
    y_blocks: wp.array(dtype=wp.float32, ndim=3),
    x_blocks: wp.array(dtype=wp.float32, ndim=3),
):
    w, local = wp.tid()
    if local >= level_count:
        return
    J = level_pivots[level_offset + local]

    y_3d = wp.tile_load(y_blocks, shape=(1, 1, _BS), offset=(w, J, 0))
    y_J = wp.tile_reshape(y_3d, (_BS, 1))

    cs = L_col_ptr[J]
    ce = L_col_ptr[J + 1]
    for ptr in range(cs, ce):
        I = L_row_idx[ptr]
        M_IJ_4d = wp.tile_load(L_blocks, shape=(1, 1, _BS, _BS), offset=(w, ptr, 0, 0))
        M_IJ = wp.tile_reshape(M_IJ_4d, (_BS, _BS))
        M_IJ_T = wp.tile_transpose(M_IJ)
        x_I_3d = wp.tile_load(x_blocks, shape=(1, 1, _BS), offset=(w, I, 0))
        x_I = wp.tile_reshape(x_I_3d, (_BS, 1))
        wp.tile_matmul(M_IJ_T, x_I, y_J, alpha=-1.0)

    M_JJ_4d = wp.tile_load(D_blocks, shape=(1, 1, _BS, _BS), offset=(w, J, 0, 0))
    M_JJ = wp.tile_reshape(M_JJ_4d, (_BS, _BS))
    M_JJ_T = wp.tile_transpose(M_JJ)
    wp.tile_upper_solve_inplace(M_JJ_T, y_J)

    out = wp.tile_reshape(y_J, (1, 1, _BS))
    wp.tile_store(x_blocks, out, offset=(w, J, 0))


@wp.kernel
def batched_scatter_x_kernel(
    x_blocks: wp.array(dtype=wp.float32, ndim=3),
    pivot_order: wp.array(dtype=wp.int32),
    nc_offset: wp.array(dtype=wp.int32),
    block_sizes: wp.array(dtype=wp.int32),
    num_pivots: wp.int32,
    out: wp.array(dtype=wp.float32, ndim=2),
):
    w, k = wp.tid()
    if k >= num_pivots:
        return
    j_id = pivot_order[k]
    row0 = nc_offset[j_id]
    nc = block_sizes[k]
    for r in range(nc):
        out[w, row0 + r] = x_blocks[w, k, r]


# =============================================================================
# Batched diagonal preconditioning kernels
# =============================================================================


@wp.kernel
def batched_compute_precond_scale_kernel(
    N_diag: wp.array(dtype=wp.float32, ndim=4),
    block_sizes: wp.array(dtype=wp.int32),
    num_pivots: wp.int32,
    # output
    scale_blocks: wp.array(dtype=wp.float32, ndim=3),
):
    """Batched: compute ``scale_blocks[w, k, r] = 1 / sqrt(N_diag[w, k, r, r])``."""
    w, k = wp.tid()
    if k >= num_pivots:
        return
    nc = block_sizes[k]
    for r in range(_BS):
        if r < nc:
            d = N_diag[w, k, r, r]
            if d < 1.0e-30:
                d = 1.0e-30
            scale_blocks[w, k, r] = 1.0 / wp.sqrt(d)
        else:
            scale_blocks[w, k, r] = 1.0


@wp.kernel
def batched_apply_precond_scale_N_diag_kernel(
    scale_blocks: wp.array(dtype=wp.float32, ndim=3),
    num_pivots: wp.int32,
    # input/output
    N_diag: wp.array(dtype=wp.float32, ndim=4),
):
    """Batched: scale diagonal blocks in place."""
    w, k = wp.tid()
    if k >= num_pivots:
        return
    for r in range(_BS):
        sr = scale_blocks[w, k, r]
        for c in range(_BS):
            sc = scale_blocks[w, k, c]
            N_diag[w, k, r, c] = N_diag[w, k, r, c] * sr * sc


@wp.kernel
def batched_apply_precond_scale_N_off_kernel(
    scale_blocks: wp.array(dtype=wp.float32, ndim=3),
    N_off_row_idx: wp.array(dtype=wp.int32),
    N_off_col_idx: wp.array(dtype=wp.int32),
    nnz_N: wp.int32,
    # input/output
    N_off: wp.array(dtype=wp.float32, ndim=4),
):
    """Batched: scale off-diagonal blocks in place."""
    w, s = wp.tid()
    if s >= nnz_N:
        return
    I = N_off_row_idx[s]
    J = N_off_col_idx[s]
    for r in range(_BS):
        sr = scale_blocks[w, I, r]
        for c in range(_BS):
            sc = scale_blocks[w, J, c]
            N_off[w, s, r, c] = N_off[w, s, r, c] * sr * sc


@wp.kernel
def batched_apply_precond_scale_rhs_kernel(
    scale_blocks: wp.array(dtype=wp.float32, ndim=3),
    num_pivots: wp.int32,
    # input/output
    rhs_blocks: wp.array(dtype=wp.float32, ndim=3),
):
    """Batched: scale RHS in place."""
    w, k = wp.tid()
    if k >= num_pivots:
        return
    for r in range(_BS):
        rhs_blocks[w, k, r] = rhs_blocks[w, k, r] * scale_blocks[w, k, r]


@wp.kernel
def batched_apply_precond_unscale_x_kernel(
    scale_blocks: wp.array(dtype=wp.float32, ndim=3),
    num_pivots: wp.int32,
    # input/output
    x_blocks: wp.array(dtype=wp.float32, ndim=3),
):
    """Batched: unscale solution in place."""
    w, k = wp.tid()
    if k >= num_pivots:
        return
    for r in range(_BS):
        x_blocks[w, k, r] = x_blocks[w, k, r] * scale_blocks[w, k, r]


@wp.kernel
def batched_apply_precond_add_reg_kernel(
    block_sizes: wp.array(dtype=wp.int32),
    reg: wp.float32,
    num_pivots: wp.int32,
    # input/output
    N_diag: wp.array(dtype=wp.float32, ndim=4),
):
    """Batched: add regularization to scaled diagonal blocks."""
    w, k = wp.tid()
    if k >= num_pivots:
        return
    nc = block_sizes[k]
    for r in range(_BS):
        if r < nc:
            N_diag[w, k, r, r] = N_diag[w, k, r, r] + reg


__all__ = [
    "batched_assemble_N_diag_kernel",
    "batched_assemble_N_off_kernel",
    "batched_zero_L_blocks_kernel",
    "batched_copy_N_off_to_L_kernel",
    "batched_ldl_factor_level_kernel",
    "batched_gather_rhs_kernel",
    "batched_block_forward_sub_kernel",
    "batched_block_backward_sub_kernel",
    "batched_scatter_x_kernel",
    "batched_compute_precond_scale_kernel",
    "batched_apply_precond_scale_N_diag_kernel",
    "batched_apply_precond_scale_N_off_kernel",
    "batched_apply_precond_scale_rhs_kernel",
    "batched_apply_precond_unscale_x_kernel",
    "batched_apply_precond_add_reg_kernel",
]
