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

"""Block-sparse storage layout for the tile LDL solver.

Holds per-step numeric arrays (``N``, ``L``, ``D``, work buffers) and the
device mirrors of the symbolic factorization. Allocated once during
``prepare_for_capture`` and reused for every solve, so the solve path issues
only kernel launches (no host-device sync, no allocations).
"""

from __future__ import annotations

from dataclasses import dataclass

import warp as wp

from .block_sparse_symbolic import BlockSparseSymbolic

_BLOCK_SIZE = 6


@dataclass
class BlockSparseStorage:
    """Numeric and symbolic device arrays for a single-world block-sparse LDL.

    Numeric arrays are zeroed on allocation. The N assembly kernel writes
    ``N_diag`` and ``N_off``; the factor kernel writes ``D_blocks`` and
    ``L_blocks`` in place; the substitution kernels write ``y_blocks``,
    ``z_blocks``, ``x_blocks``. ``rhs_blocks`` is the gather of ``ctx.b``
    in pivot order with zero padding beyond the live block size.
    """

    num_joints: int
    block_size: int
    nnz_L: int
    nnz_N: int

    N_diag: wp.array
    N_off: wp.array
    D_blocks: wp.array
    L_blocks: wp.array

    y_blocks: wp.array
    z_blocks: wp.array
    x_blocks: wp.array
    rhs_blocks: wp.array
    scale_blocks: wp.array  # Diagonal preconditioning scale [nj, 6]

    block_sizes_dev: wp.array
    pivot_order_dev: wp.array
    inv_pivot_order_dev: wp.array

    L_col_ptr_dev: wp.array
    L_row_idx_dev: wp.array
    L_row_ptr_dev: wp.array
    L_col_idx_dev: wp.array
    L_csr_to_csc_dev: wp.array

    N_off_col_ptr_dev: wp.array
    N_off_row_idx_dev: wp.array
    N_off_col_idx_dev: wp.array
    N_off_to_L_dev: wp.array

    parent_dev: wp.array
    pred_diag_ptr_dev: wp.array
    pred_diag_slot_dev: wp.array
    pred_off_ptr_dev: wp.array
    pred_off_slot_IK_dev: wp.array
    pred_off_slot_JK_dev: wp.array
    level_ptr_dev: wp.array
    level_pivots_dev: wp.array

    num_levels: int

    # Workspace for iterative refinement (flat residual in original row order).
    # Allocated only when iterative_refinement_steps > 0.
    _refinement_residual: wp.array | None = None


@dataclass
class BatchedBlockSparseStorage:
    """Multi-world block-sparse LDL storage.

    Symbolic device arrays are shared across worlds (single allocation).
    Numeric and work arrays gain a leading ``num_worlds`` dimension. All
    kernels using this storage launch with a leading ``world`` dimension.
    """

    num_worlds: int
    num_joints: int
    block_size: int
    nnz_L: int
    nnz_N: int

    N_diag: wp.array
    N_off: wp.array
    D_blocks: wp.array
    L_blocks: wp.array

    y_blocks: wp.array
    z_blocks: wp.array
    x_blocks: wp.array
    rhs_blocks: wp.array
    scale_blocks: wp.array  # Diagonal preconditioning scale [nw, nj, 6]

    block_sizes_dev: wp.array
    pivot_order_dev: wp.array
    inv_pivot_order_dev: wp.array

    L_col_ptr_dev: wp.array
    L_row_idx_dev: wp.array
    L_row_ptr_dev: wp.array
    L_col_idx_dev: wp.array
    L_csr_to_csc_dev: wp.array

    N_off_col_ptr_dev: wp.array
    N_off_row_idx_dev: wp.array
    N_off_col_idx_dev: wp.array
    N_off_to_L_dev: wp.array

    parent_dev: wp.array
    pred_diag_ptr_dev: wp.array
    pred_diag_slot_dev: wp.array
    pred_off_ptr_dev: wp.array
    pred_off_slot_IK_dev: wp.array
    pred_off_slot_JK_dev: wp.array
    level_ptr_dev: wp.array
    level_pivots_dev: wp.array

    num_levels: int


def _transfer_symbolic(symbolic: BlockSparseSymbolic, device) -> dict:
    """Push immutable symbolic arrays to the given device."""
    with wp.ScopedDevice(device):
        return {
            "block_sizes_dev": wp.array(symbolic.block_sizes, dtype=wp.int32, device=device),
            "pivot_order_dev": wp.array(symbolic.pivot_order, dtype=wp.int32, device=device),
            "inv_pivot_order_dev": wp.array(symbolic.inv_pivot_order, dtype=wp.int32, device=device),
            "L_col_ptr_dev": wp.array(symbolic.L_col_ptr, dtype=wp.int32, device=device),
            "L_row_idx_dev": wp.array(symbolic.L_row_idx, dtype=wp.int32, device=device),
            "L_row_ptr_dev": wp.array(symbolic.L_row_ptr, dtype=wp.int32, device=device),
            "L_col_idx_dev": wp.array(symbolic.L_col_idx, dtype=wp.int32, device=device),
            "L_csr_to_csc_dev": wp.array(symbolic.L_csr_to_csc, dtype=wp.int32, device=device),
            "N_off_col_ptr_dev": wp.array(symbolic.N_off_col_ptr, dtype=wp.int32, device=device),
            "N_off_row_idx_dev": wp.array(symbolic.N_off_row_idx, dtype=wp.int32, device=device),
            "N_off_col_idx_dev": wp.array(symbolic.N_off_col_idx, dtype=wp.int32, device=device),
            "N_off_to_L_dev": wp.array(symbolic.N_off_to_L, dtype=wp.int32, device=device),
            "parent_dev": wp.array(symbolic.parent, dtype=wp.int32, device=device),
            "pred_diag_ptr_dev": wp.array(symbolic.pred_diag_ptr, dtype=wp.int32, device=device),
            "pred_diag_slot_dev": wp.array(symbolic.pred_diag_slot, dtype=wp.int32, device=device),
            "pred_off_ptr_dev": wp.array(symbolic.pred_off_ptr, dtype=wp.int32, device=device),
            "pred_off_slot_IK_dev": wp.array(symbolic.pred_off_slot_IK, dtype=wp.int32, device=device),
            "pred_off_slot_JK_dev": wp.array(symbolic.pred_off_slot_JK, dtype=wp.int32, device=device),
            "level_ptr_dev": wp.array(symbolic.level_ptr, dtype=wp.int32, device=device),
            "level_pivots_dev": wp.array(symbolic.level_pivots, dtype=wp.int32, device=device),
        }


def allocate_storage(
    symbolic: BlockSparseSymbolic,
    device,
    *,
    total_nc: int = 0,
) -> BlockSparseStorage:
    """Allocate single-world block-sparse storage on ``device``.

    Args:
        symbolic: Symbolic factorization.
        device: Compute device.
        total_nc: Total number of constraint rows. When > 0 a flat residual
            buffer is allocated for iterative refinement.
    """
    nj = max(symbolic.num_joints, 1)
    nnz_L = max(symbolic.nnz_L, 1)
    nnz_N = max(symbolic.nnz_N, 1)
    bs = _BLOCK_SIZE

    sym_dev = _transfer_symbolic(symbolic, device)

    with wp.ScopedDevice(device):
        N_diag = wp.zeros((nj, bs, bs), dtype=wp.float32, device=device)
        N_off = wp.zeros((nnz_N, bs, bs), dtype=wp.float32, device=device)
        D_blocks = wp.zeros((nj, bs, bs), dtype=wp.float32, device=device)
        L_blocks = wp.zeros((nnz_L, bs, bs), dtype=wp.float32, device=device)

        y_blocks = wp.zeros((nj, bs), dtype=wp.float32, device=device)
        z_blocks = wp.zeros((nj, bs), dtype=wp.float32, device=device)
        x_blocks = wp.zeros((nj, bs), dtype=wp.float32, device=device)
        rhs_blocks = wp.zeros((nj, bs), dtype=wp.float32, device=device)
        scale_blocks = wp.zeros((nj, bs), dtype=wp.float32, device=device)

        refinement_residual = None
        if total_nc > 0:
            refinement_residual = wp.zeros(total_nc, dtype=wp.float32, device=device)

    return BlockSparseStorage(
        num_joints=symbolic.num_joints,
        block_size=bs,
        nnz_L=symbolic.nnz_L,
        nnz_N=symbolic.nnz_N,
        N_diag=N_diag,
        N_off=N_off,
        D_blocks=D_blocks,
        L_blocks=L_blocks,
        y_blocks=y_blocks,
        z_blocks=z_blocks,
        x_blocks=x_blocks,
        rhs_blocks=rhs_blocks,
        scale_blocks=scale_blocks,
        _refinement_residual=refinement_residual,
        num_levels=symbolic.num_levels,
        **sym_dev,
    )


def allocate_batched_storage(
    symbolic: BlockSparseSymbolic,
    device,
    num_worlds: int,
) -> BatchedBlockSparseStorage:
    """Allocate multi-world block-sparse storage on ``device``."""
    if num_worlds <= 0:
        raise ValueError("num_worlds must be positive")

    nj = max(symbolic.num_joints, 1)
    nnz_L = max(symbolic.nnz_L, 1)
    nnz_N = max(symbolic.nnz_N, 1)
    bs = _BLOCK_SIZE

    sym_dev = _transfer_symbolic(symbolic, device)

    with wp.ScopedDevice(device):
        N_diag = wp.zeros((num_worlds, nj, bs, bs), dtype=wp.float32, device=device)
        N_off = wp.zeros((num_worlds, nnz_N, bs, bs), dtype=wp.float32, device=device)
        D_blocks = wp.zeros((num_worlds, nj, bs, bs), dtype=wp.float32, device=device)
        L_blocks = wp.zeros((num_worlds, nnz_L, bs, bs), dtype=wp.float32, device=device)

        y_blocks = wp.zeros((num_worlds, nj, bs), dtype=wp.float32, device=device)
        z_blocks = wp.zeros((num_worlds, nj, bs), dtype=wp.float32, device=device)
        x_blocks = wp.zeros((num_worlds, nj, bs), dtype=wp.float32, device=device)
        rhs_blocks = wp.zeros((num_worlds, nj, bs), dtype=wp.float32, device=device)
        scale_blocks = wp.zeros((num_worlds, nj, bs), dtype=wp.float32, device=device)

    return BatchedBlockSparseStorage(
        num_worlds=num_worlds,
        num_joints=symbolic.num_joints,
        block_size=bs,
        nnz_L=symbolic.nnz_L,
        nnz_N=symbolic.nnz_N,
        N_diag=N_diag,
        N_off=N_off,
        D_blocks=D_blocks,
        L_blocks=L_blocks,
        y_blocks=y_blocks,
        z_blocks=z_blocks,
        x_blocks=x_blocks,
        rhs_blocks=rhs_blocks,
        scale_blocks=scale_blocks,
        num_levels=symbolic.num_levels,
        **sym_dev,
    )


__all__ = [
    "BatchedBlockSparseStorage",
    "BlockSparseStorage",
    "allocate_batched_storage",
    "allocate_storage",
]
