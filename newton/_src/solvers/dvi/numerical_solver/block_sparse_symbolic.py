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

"""Unified symbolic factorization for the block-sparse tile LDL solver.

Merges the CSC L pattern + elimination tree from ``BlockSymbolicFactorization``
with the MECA ordering + RSI / edge metadata from ``SparseBlockSymbolic`` and
adds level scheduling derived from the elimination tree.

The output ``BlockSparseSymbolic`` is consumed by the runtime numeric solver to
drive sparse N assembly, level-scheduled LDL factorization, and triangular
solves on Warp tiles.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .sparse_block_ldl_symbolic import (
    SparseBlockSymbolic,
    build_constraint_graph,
    compute_symbolic_factorization,
)


_BLOCK_SIZE = 6


@dataclass
class BlockSparseSymbolic:
    """Unified symbolic factorization for block-sparse tile LDL.

    All arrays are column-oriented in elimination order. Joint indices used in
    ``L_row_idx`` / ``L_col_idx`` / ``parent`` / ``level_pivots`` are *pivot
    indices* (positions in ``pivot_order``), not raw joint indices. This keeps
    the runtime kernels indexable by pivot directly.

    Attributes:
        num_joints: Number of joints (pivots).
        block_size: Padded block size (always 6).
        block_sizes: Live constraint count per pivot (in pivot order) [num_joints].
        block_offsets: Prefix-sum of ``block_sizes`` [num_joints + 1].
        total_nc: Sum of ``block_sizes``.
        pivot_order: Joint index for each pivot position [num_joints].
        inv_pivot_order: Pivot position for each joint index [num_joints].
        L_col_ptr: CSC column pointers for L's strict lower triangle [num_joints + 1].
        L_row_idx: CSC row indices for L (pivot indices) [nnz_L].
        L_row_ptr: CSR row pointers for L's strict lower triangle [num_joints + 1].
        L_col_idx: CSR column indices for L (pivot indices) [nnz_L].
        L_csr_to_csc: Permutation mapping CSR slot k to its CSC slot [nnz_L].
        N_off_col_ptr: CSC column pointers for N's strict lower triangle [num_joints + 1].
        N_off_row_idx: CSC row indices for N (pivot indices) [nnz_N].
        N_off_to_L: For each N off-diagonal slot, the corresponding L CSC slot [nnz_N].
        parent: Elimination tree parent (pivot index, -1 for roots) [num_joints].
        num_levels: Number of elimination tree depth levels.
        level_ptr: CSR-style start of each level into ``level_pivots`` [num_levels + 1].
        level_pivots: Pivots grouped by depth, leaves last [num_joints].
        rsi_ptr: Optional RSI pointer (forwarded from ``SparseBlockSymbolic``).
        rsi_e_row: Optional RSI row metadata.
        rsi_l_col: Optional RSI column metadata.
        rsi_target_pivot: Optional RSI scatter target pivot.
        rsi_target_is_diag: Optional RSI flag for diagonal scatter.
        rsi_target_local_row: Optional RSI local row.
        rsi_target_local_col: Optional RSI local column.
        nnz_L: Number of off-diagonal blocks in L.
        nnz_N: Number of off-diagonal blocks in N.
    """

    num_joints: int
    block_size: int

    block_sizes: np.ndarray
    block_offsets: np.ndarray
    total_nc: int

    pivot_order: np.ndarray
    inv_pivot_order: np.ndarray

    L_col_ptr: np.ndarray
    L_row_idx: np.ndarray
    L_row_ptr: np.ndarray
    L_col_idx: np.ndarray
    L_csr_to_csc: np.ndarray

    N_off_col_ptr: np.ndarray
    N_off_row_idx: np.ndarray
    N_off_col_idx: np.ndarray
    N_off_to_L: np.ndarray

    parent: np.ndarray

    pred_diag_ptr: np.ndarray
    pred_diag_slot: np.ndarray

    pred_off_ptr: np.ndarray
    pred_off_slot_IK: np.ndarray
    pred_off_slot_JK: np.ndarray

    num_levels: int
    level_ptr: np.ndarray
    level_pivots: np.ndarray

    rsi_ptr: np.ndarray
    rsi_e_row: np.ndarray
    rsi_l_col: np.ndarray
    rsi_target_pivot: np.ndarray
    rsi_target_is_diag: np.ndarray
    rsi_target_local_row: np.ndarray
    rsi_target_local_col: np.ndarray

    nnz_L: int
    nnz_N: int


def compute_block_sparse_symbolic(
    joint_body_a: np.ndarray,
    joint_body_b: np.ndarray,
    block_nc: np.ndarray,
    *,
    use_meca: bool = False,
) -> BlockSparseSymbolic:
    """Compute the unified symbolic factorization.

    Args:
        joint_body_a: Parent body index per joint, ``-1`` for world [joint_count].
        joint_body_b: Child body index per joint, ``-1`` for world [joint_count].
        block_nc: Live constraint count per joint, integers in ``{0, 3, 5, 6}``
            for the dvi solver [joint_count].
        use_meca: If True, use MECA ordering for fill-in reduction. If False,
            use natural joint order.

    Returns:
        ``BlockSparseSymbolic`` with all arrays expressed in pivot order.
    """
    joint_body_a = np.asarray(joint_body_a, dtype=np.int32)
    joint_body_b = np.asarray(joint_body_b, dtype=np.int32)
    block_nc = np.asarray(block_nc, dtype=np.int32)

    num_joints_total = int(joint_body_a.shape[0])

    active_joints = np.where(block_nc > 0)[0].astype(np.int32)
    num_joints = int(active_joints.shape[0])

    if num_joints == 0:
        return _empty_symbolic()

    body_a_active = joint_body_a[active_joints]
    body_b_active = joint_body_b[active_joints]
    block_nc_active = block_nc[active_joints]

    sub_symbolic = compute_symbolic_factorization(
        body_a_active,
        body_b_active,
        block_nc_active,
        use_meca=use_meca,
    )

    pivot_order_local = np.asarray(sub_symbolic.pivot_order, dtype=np.int32)
    pivot_order = active_joints[pivot_order_local].astype(np.int32)

    inv_pivot_order = np.full(num_joints_total, -1, dtype=np.int32)
    for k in range(num_joints):
        inv_pivot_order[pivot_order[k]] = k

    block_sizes = block_nc_active[pivot_order_local].astype(np.int32)
    block_offsets = np.zeros(num_joints + 1, dtype=np.int32)
    np.cumsum(block_sizes, out=block_offsets[1:])
    total_nc = int(block_offsets[-1])

    inv_local = np.full(num_joints, -1, dtype=np.int32)
    for k in range(num_joints):
        inv_local[pivot_order_local[k]] = k

    n_off_pairs_pivot: set[tuple[int, int]] = set()
    body_to_pivots: dict[int, list[int]] = defaultdict(list)
    for k in range(num_joints):
        ba = int(body_a_active[pivot_order_local[k]])
        bb = int(body_b_active[pivot_order_local[k]])
        if ba >= 0:
            body_to_pivots[ba].append(k)
        if bb >= 0:
            body_to_pivots[bb].append(k)
    for pivots in body_to_pivots.values():
        ps = sorted(pivots)
        for i, p1 in enumerate(ps):
            for p2 in ps[i + 1 :]:
                n_off_pairs_pivot.add((max(p1, p2), min(p1, p2)))

    n_off_cols: dict[int, list[int]] = defaultdict(list)
    for I, J in n_off_pairs_pivot:
        n_off_cols[J].append(I)
    for J in n_off_cols:
        n_off_cols[J].sort()

    nnz_N = sum(len(rows) for rows in n_off_cols.values())
    N_off_col_ptr = np.zeros(num_joints + 1, dtype=np.int32)
    N_off_row_idx = np.zeros(max(nnz_N, 1), dtype=np.int32)
    N_off_col_idx = np.zeros(max(nnz_N, 1), dtype=np.int32)
    ptr = 0
    for J in range(num_joints):
        N_off_col_ptr[J] = ptr
        if J in n_off_cols:
            for I in n_off_cols[J]:
                N_off_row_idx[ptr] = I
                N_off_col_idx[ptr] = J
                ptr += 1
    N_off_col_ptr[num_joints] = ptr

    L_pattern: set[tuple[int, int]] = set(n_off_pairs_pivot)
    parent = np.full(num_joints, -1, dtype=np.int32)
    for J in range(num_joints):
        rows_in_col = sorted(I for (I, K) in L_pattern if K == J and I > J)
        if rows_in_col:
            parent[J] = rows_in_col[0]
        for idx, I in enumerate(rows_in_col):
            for I2 in rows_in_col[idx + 1 :]:
                L_pattern.add((I2, I))

    L_cols: dict[int, list[int]] = defaultdict(list)
    for I, J in L_pattern:
        L_cols[J].append(I)
    for J in L_cols:
        L_cols[J].sort()

    nnz_L = sum(len(rows) for rows in L_cols.values())
    L_col_ptr = np.zeros(num_joints + 1, dtype=np.int32)
    L_row_idx = np.zeros(max(nnz_L, 1), dtype=np.int32)
    L_csc_lookup: dict[tuple[int, int], int] = {}
    ptr = 0
    for J in range(num_joints):
        L_col_ptr[J] = ptr
        if J in L_cols:
            for I in L_cols[J]:
                L_row_idx[ptr] = I
                L_csc_lookup[(I, J)] = ptr
                ptr += 1
    L_col_ptr[num_joints] = ptr

    L_rows: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for J in range(num_joints):
        for s in range(int(L_col_ptr[J]), int(L_col_ptr[J + 1])):
            I = int(L_row_idx[s])
            L_rows[I].append((J, s))
    for I in L_rows:
        L_rows[I].sort()

    L_row_ptr = np.zeros(num_joints + 1, dtype=np.int32)
    L_col_idx = np.zeros(max(nnz_L, 1), dtype=np.int32)
    L_csr_to_csc = np.zeros(max(nnz_L, 1), dtype=np.int32)
    ptr = 0
    for I in range(num_joints):
        L_row_ptr[I] = ptr
        if I in L_rows:
            for J, csc_slot in L_rows[I]:
                L_col_idx[ptr] = J
                L_csr_to_csc[ptr] = csc_slot
                ptr += 1
    L_row_ptr[num_joints] = ptr

    N_off_to_L = np.zeros(max(nnz_N, 1), dtype=np.int32)
    for J in range(num_joints):
        for s in range(int(N_off_col_ptr[J]), int(N_off_col_ptr[J + 1])):
            I = int(N_off_row_idx[s])
            csc_slot = L_csc_lookup.get((I, J))
            if csc_slot is None:
                raise RuntimeError(
                    f"N off-diagonal block ({I},{J}) missing from L pattern; fill-in computation is inconsistent."
                )
            N_off_to_L[s] = csc_slot

    pred_diag_ptr = np.zeros(num_joints + 1, dtype=np.int32)
    pred_diag_slots: list[int] = []
    for J in range(num_joints):
        pred_diag_ptr[J] = len(pred_diag_slots)
        for ridx in range(int(L_row_ptr[J]), int(L_row_ptr[J + 1])):
            pred_diag_slots.append(int(L_csr_to_csc[ridx]))
    pred_diag_ptr[num_joints] = len(pred_diag_slots)
    pred_diag_slot = np.array(pred_diag_slots, dtype=np.int32) if pred_diag_slots else np.zeros(1, dtype=np.int32)

    pred_off_ptr = np.zeros(nnz_L + 1, dtype=np.int32)
    pred_off_slot_IK_list: list[int] = []
    pred_off_slot_JK_list: list[int] = []
    row_csc_by_pivot: dict[int, dict[int, int]] = {}
    for I in range(num_joints):
        col_to_slot: dict[int, int] = {}
        for ridx in range(int(L_row_ptr[I]), int(L_row_ptr[I + 1])):
            col_to_slot[int(L_col_idx[ridx])] = int(L_csr_to_csc[ridx])
        row_csc_by_pivot[I] = col_to_slot

    for J in range(num_joints):
        col_J_predecessors: dict[int, int] = {}
        for ridx in range(int(L_row_ptr[J]), int(L_row_ptr[J + 1])):
            col_J_predecessors[int(L_col_idx[ridx])] = int(L_csr_to_csc[ridx])
        for ptr_idx in range(int(L_col_ptr[J]), int(L_col_ptr[J + 1])):
            I = int(L_row_idx[ptr_idx])
            pred_off_ptr[ptr_idx] = len(pred_off_slot_IK_list)
            row_I_cols = row_csc_by_pivot[I]
            for K, slot_JK in col_J_predecessors.items():
                slot_IK = row_I_cols.get(K, -1)
                if slot_IK >= 0:
                    pred_off_slot_IK_list.append(slot_IK)
                    pred_off_slot_JK_list.append(slot_JK)
    pred_off_ptr[nnz_L] = len(pred_off_slot_IK_list)

    pred_off_slot_IK = (
        np.array(pred_off_slot_IK_list, dtype=np.int32) if pred_off_slot_IK_list else np.zeros(1, dtype=np.int32)
    )
    pred_off_slot_JK = (
        np.array(pred_off_slot_JK_list, dtype=np.int32) if pred_off_slot_JK_list else np.zeros(1, dtype=np.int32)
    )

    depth = np.zeros(num_joints, dtype=np.int32)
    children = [[] for _ in range(num_joints)]
    roots = []
    for j in range(num_joints):
        p = int(parent[j])
        if p < 0:
            roots.append(j)
        else:
            children[p].append(j)

    order_topo: list[int] = []
    visited = np.zeros(num_joints, dtype=bool)
    stack = list(roots)
    while stack:
        node = stack.pop()
        if visited[node]:
            continue
        visited[node] = True
        order_topo.append(node)
        stack.extend(children[node])

    for node in order_topo:
        for child in children[node]:
            depth[child] = depth[node] + 1

    num_levels = int(depth.max()) + 1 if num_joints > 0 else 0
    level_pivots = np.zeros(num_joints, dtype=np.int32)
    level_counts = np.zeros(num_levels + 1, dtype=np.int32)
    for j in range(num_joints):
        level_counts[depth[j] + 1] += 1
    level_ptr = np.zeros(num_levels + 1, dtype=np.int32)
    np.cumsum(level_counts[1:], out=level_ptr[1:])
    cursors = level_ptr.copy()
    for j in range(num_joints):
        d = int(depth[j])
        slot = cursors[d]
        level_pivots[slot] = j
        cursors[d] = slot + 1

    rsi_ptr = np.asarray(sub_symbolic.rsi_ptr, dtype=np.int32)
    rsi_e_row = np.asarray(sub_symbolic.rsi_e_row, dtype=np.int32)
    rsi_l_col = np.asarray(sub_symbolic.rsi_l_col, dtype=np.int32)
    rsi_target_pivot = np.asarray(sub_symbolic.rsi_target_pivot, dtype=np.int32)
    rsi_target_is_diag = np.asarray(sub_symbolic.rsi_target_is_diag, dtype=np.int32)
    rsi_target_local_row = np.asarray(sub_symbolic.rsi_target_local_row, dtype=np.int32)
    rsi_target_local_col = np.asarray(sub_symbolic.rsi_target_local_col, dtype=np.int32)

    return BlockSparseSymbolic(
        num_joints=num_joints,
        block_size=_BLOCK_SIZE,
        block_sizes=block_sizes,
        block_offsets=block_offsets,
        total_nc=total_nc,
        pivot_order=pivot_order,
        inv_pivot_order=inv_pivot_order,
        L_col_ptr=L_col_ptr,
        L_row_idx=L_row_idx,
        L_row_ptr=L_row_ptr,
        L_col_idx=L_col_idx,
        L_csr_to_csc=L_csr_to_csc,
        N_off_col_ptr=N_off_col_ptr,
        N_off_row_idx=N_off_row_idx,
        N_off_col_idx=N_off_col_idx,
        N_off_to_L=N_off_to_L,
        parent=parent,
        pred_diag_ptr=pred_diag_ptr,
        pred_diag_slot=pred_diag_slot,
        pred_off_ptr=pred_off_ptr,
        pred_off_slot_IK=pred_off_slot_IK,
        pred_off_slot_JK=pred_off_slot_JK,
        num_levels=num_levels,
        level_ptr=level_ptr,
        level_pivots=level_pivots,
        rsi_ptr=rsi_ptr,
        rsi_e_row=rsi_e_row,
        rsi_l_col=rsi_l_col,
        rsi_target_pivot=rsi_target_pivot,
        rsi_target_is_diag=rsi_target_is_diag,
        rsi_target_local_row=rsi_target_local_row,
        rsi_target_local_col=rsi_target_local_col,
        nnz_L=nnz_L,
        nnz_N=nnz_N,
    )


def _empty_symbolic() -> BlockSparseSymbolic:
    """Return a symbolic factorization for the zero-active-joint case."""
    return BlockSparseSymbolic(
        num_joints=0,
        block_size=_BLOCK_SIZE,
        block_sizes=np.zeros(0, dtype=np.int32),
        block_offsets=np.zeros(1, dtype=np.int32),
        total_nc=0,
        pivot_order=np.zeros(0, dtype=np.int32),
        inv_pivot_order=np.zeros(0, dtype=np.int32),
        L_col_ptr=np.zeros(1, dtype=np.int32),
        L_row_idx=np.zeros(1, dtype=np.int32),
        L_row_ptr=np.zeros(1, dtype=np.int32),
        L_col_idx=np.zeros(1, dtype=np.int32),
        L_csr_to_csc=np.zeros(1, dtype=np.int32),
        N_off_col_ptr=np.zeros(1, dtype=np.int32),
        N_off_row_idx=np.zeros(1, dtype=np.int32),
        N_off_col_idx=np.zeros(1, dtype=np.int32),
        N_off_to_L=np.zeros(1, dtype=np.int32),
        parent=np.zeros(0, dtype=np.int32),
        pred_diag_ptr=np.zeros(1, dtype=np.int32),
        pred_diag_slot=np.zeros(1, dtype=np.int32),
        pred_off_ptr=np.zeros(1, dtype=np.int32),
        pred_off_slot_IK=np.zeros(1, dtype=np.int32),
        pred_off_slot_JK=np.zeros(1, dtype=np.int32),
        num_levels=0,
        level_ptr=np.zeros(1, dtype=np.int32),
        level_pivots=np.zeros(0, dtype=np.int32),
        rsi_ptr=np.zeros(1, dtype=np.int32),
        rsi_e_row=np.zeros(1, dtype=np.int32),
        rsi_l_col=np.zeros(1, dtype=np.int32),
        rsi_target_pivot=np.zeros(1, dtype=np.int32),
        rsi_target_is_diag=np.zeros(1, dtype=np.int32),
        rsi_target_local_row=np.zeros(1, dtype=np.int32),
        rsi_target_local_col=np.zeros(1, dtype=np.int32),
        nnz_L=0,
        nnz_N=0,
    )


def tile_symbolic_for_worlds(
    sym_one: BlockSparseSymbolic,
    num_worlds: int,
    joints_per_world: int,
) -> BlockSparseSymbolic:
    """Replicate a single-world symbolic factorization across identical worlds.

    Given a symbolic factorization for one world (``sym_one``), produce a
    combined symbolic that is block-diagonal: one independent copy of the
    original per world, with pivot / joint / L-slot indices offset so that
    the existing flat (non-batched) solve kernels work unchanged.

    This avoids the O(n²+) cost of recomputing the symbolic over all joints
    when every world has the same topology.

    Args:
        sym_one: Symbolic factorization for a single world.
        num_worlds: Number of identical worlds.
        joints_per_world: Number of raw (not just active) joints per world.
            Used to offset ``pivot_order`` and ``inv_pivot_order`` which index
            into the global joint arrays.

    Returns:
        A new ``BlockSparseSymbolic`` spanning all worlds.
    """
    if num_worlds <= 1:
        return sym_one

    W = num_worlds
    nj1 = sym_one.num_joints          # active pivots in one world
    nnzL1 = sym_one.nnz_L
    nnzN1 = sym_one.nnz_N
    tc1 = sym_one.total_nc
    nl1 = sym_one.num_levels

    nj = nj1 * W
    nnzL = nnzL1 * W
    nnzN = nnzN1 * W
    tc = tc1 * W

    # ── Trivial per-world tile of 1-D arrays with pivot-index offsets ──
    def _tile_pivot(arr):
        """Tile array and add w*nj1 offset to each world's copy."""
        parts = [arr + w * nj1 for w in range(W)]
        return np.concatenate(parts).astype(np.int32)

    def _tile_nnzL(arr):
        """Tile array and add w*nnzL1 offset."""
        parts = [arr + w * nnzL1 for w in range(W)]
        return np.concatenate(parts).astype(np.int32)

    def _tile_flat(arr):
        """Tile array without offset."""
        return np.tile(arr, W).astype(arr.dtype)

    # pivot_order: maps pivot position k -> global joint index
    # For world w, global_joint = w * joints_per_world + local_joint
    pivot_order = np.concatenate([
        sym_one.pivot_order + w * joints_per_world for w in range(W)
    ]).astype(np.int32)

    # inv_pivot_order: maps global joint index -> pivot position
    total_joints = joints_per_world * W
    inv_pivot_order = np.full(total_joints, -1, dtype=np.int32)
    for k in range(nj):
        inv_pivot_order[pivot_order[k]] = k

    # block_sizes: just tile
    block_sizes = _tile_flat(sym_one.block_sizes)

    # block_offsets: cumulative, offset by w*tc1
    bo1 = sym_one.block_offsets  # [nj1+1]
    block_offsets = np.zeros(nj + 1, dtype=np.int32)
    for w in range(W):
        s = w * nj1
        block_offsets[s:s + nj1 + 1] = bo1 + w * tc1
    # last entry
    block_offsets[nj] = tc

    # ── L CSC arrays ──
    # L_col_ptr: offset pointers by w*nnzL1
    lcp1 = sym_one.L_col_ptr  # [nj1+1]
    L_col_ptr = np.zeros(nj + 1, dtype=np.int32)
    for w in range(W):
        s = w * nj1
        L_col_ptr[s:s + nj1 + 1] = lcp1 + w * nnzL1
    L_col_ptr[nj] = nnzL

    # L_row_idx: pivot indices, offset by w*nj1
    L_row_idx = _tile_pivot(sym_one.L_row_idx[:nnzL1]) if nnzL1 > 0 else np.zeros(max(nnzL, 1), dtype=np.int32)

    # L CSR arrays
    lrp1 = sym_one.L_row_ptr  # [nj1+1]
    L_row_ptr = np.zeros(nj + 1, dtype=np.int32)
    for w in range(W):
        s = w * nj1
        L_row_ptr[s:s + nj1 + 1] = lrp1 + w * nnzL1
    L_row_ptr[nj] = nnzL

    L_col_idx = _tile_pivot(sym_one.L_col_idx[:nnzL1]) if nnzL1 > 0 else np.zeros(max(nnzL, 1), dtype=np.int32)
    L_csr_to_csc = _tile_nnzL(sym_one.L_csr_to_csc[:nnzL1]) if nnzL1 > 0 else np.zeros(max(nnzL, 1), dtype=np.int32)

    # ── N off-diagonal arrays ──
    ncp1 = sym_one.N_off_col_ptr  # [nj1+1]
    N_off_col_ptr = np.zeros(nj + 1, dtype=np.int32)
    for w in range(W):
        s = w * nj1
        N_off_col_ptr[s:s + nj1 + 1] = ncp1 + w * nnzN1
    N_off_col_ptr[nj] = nnzN

    N_off_row_idx = _tile_pivot(sym_one.N_off_row_idx[:nnzN1]) if nnzN1 > 0 else np.zeros(max(nnzN, 1), dtype=np.int32)
    N_off_col_idx = _tile_pivot(sym_one.N_off_col_idx[:nnzN1]) if nnzN1 > 0 else np.zeros(max(nnzN, 1), dtype=np.int32)
    N_off_to_L = _tile_nnzL(sym_one.N_off_to_L[:nnzN1]) if nnzN1 > 0 else np.zeros(max(nnzN, 1), dtype=np.int32)

    # ── Parent / elimination tree ──
    parent1 = sym_one.parent  # [nj1], pivot indices or -1
    parent = np.concatenate([
        np.where(parent1 >= 0, parent1 + w * nj1, -1) for w in range(W)
    ]).astype(np.int32)

    # ── Predecessor arrays (diag) ──
    pdp1 = sym_one.pred_diag_ptr  # [nj1+1]
    npd1 = int(pdp1[-1])  # total pred_diag entries for one world
    pred_diag_ptr = np.zeros(nj + 1, dtype=np.int32)
    for w in range(W):
        s = w * nj1
        pred_diag_ptr[s:s + nj1 + 1] = pdp1 + w * npd1
    pred_diag_ptr[nj] = npd1 * W

    pds1 = sym_one.pred_diag_slot[:npd1] if npd1 > 0 else np.zeros(0, dtype=np.int32)
    pred_diag_slot = _tile_nnzL(pds1) if npd1 > 0 else np.zeros(1, dtype=np.int32)

    # ── Predecessor arrays (off-diag) ──
    pop1 = sym_one.pred_off_ptr  # [nnzL1+1]
    npo1 = int(pop1[-1]) if nnzL1 > 0 else 0
    pred_off_ptr = np.zeros(nnzL + 1, dtype=np.int32)
    for w in range(W):
        s = w * nnzL1
        pred_off_ptr[s:s + nnzL1 + 1] = pop1 + w * npo1
    pred_off_ptr[nnzL] = npo1 * W

    poIK1 = sym_one.pred_off_slot_IK[:npo1] if npo1 > 0 else np.zeros(0, dtype=np.int32)
    poJK1 = sym_one.pred_off_slot_JK[:npo1] if npo1 > 0 else np.zeros(0, dtype=np.int32)
    pred_off_slot_IK = _tile_nnzL(poIK1) if npo1 > 0 else np.zeros(1, dtype=np.int32)
    pred_off_slot_JK = _tile_nnzL(poJK1) if npo1 > 0 else np.zeros(1, dtype=np.int32)

    # ── Level scheduling ──
    # Same number of levels, but each level has W times as many pivots.
    lp1 = sym_one.level_ptr  # [nl1+1]
    level_ptr = np.zeros(nl1 + 1, dtype=np.int32)
    for lvl in range(nl1):
        count1 = int(lp1[lvl + 1] - lp1[lvl])
        level_ptr[lvl + 1] = level_ptr[lvl] + count1 * W

    # level_pivots: for each level, concatenate all worlds' pivots at that level
    lv1 = sym_one.level_pivots  # [nj1]
    level_pivots = np.zeros(nj, dtype=np.int32)
    pos = 0
    for lvl in range(nl1):
        start1 = int(lp1[lvl])
        end1 = int(lp1[lvl + 1])
        pivots_at_level = lv1[start1:end1]  # pivots in this level for one world
        for w in range(W):
            n = len(pivots_at_level)
            level_pivots[pos:pos + n] = pivots_at_level + w * nj1
            pos += n

    # ── RSI arrays ──
    rp1 = sym_one.rsi_ptr  # [nj1+1]
    nrsi1 = int(rp1[-1]) if nj1 > 0 else 0
    rsi_ptr = np.zeros(nj + 1, dtype=np.int32)
    for w in range(W):
        s = w * nj1
        rsi_ptr[s:s + nj1 + 1] = rp1 + w * nrsi1
    rsi_ptr[nj] = nrsi1 * W

    def _tile_rsi_arr(arr, count, offset_fn):
        if count == 0:
            return np.zeros(1, dtype=np.int32)
        a = arr[:count]
        return np.concatenate([offset_fn(a, w) for w in range(W)]).astype(np.int32)

    rsi_e_row = _tile_rsi_arr(sym_one.rsi_e_row, nrsi1, lambda a, w: a)  # row-local, no offset
    rsi_l_col = _tile_rsi_arr(sym_one.rsi_l_col, nrsi1, lambda a, w: a + w * nnzL1)  # L-slot offset
    rsi_target_pivot = _tile_rsi_arr(sym_one.rsi_target_pivot, nrsi1, lambda a, w: a + w * nj1)
    rsi_target_is_diag = _tile_rsi_arr(sym_one.rsi_target_is_diag, nrsi1, lambda a, w: a)  # bool, no offset
    rsi_target_local_row = _tile_rsi_arr(sym_one.rsi_target_local_row, nrsi1, lambda a, w: a)
    rsi_target_local_col = _tile_rsi_arr(sym_one.rsi_target_local_col, nrsi1, lambda a, w: a)

    return BlockSparseSymbolic(
        num_joints=nj,
        block_size=_BLOCK_SIZE,
        block_sizes=block_sizes,
        block_offsets=block_offsets,
        total_nc=tc,
        pivot_order=pivot_order,
        inv_pivot_order=inv_pivot_order,
        L_col_ptr=L_col_ptr,
        L_row_idx=L_row_idx,
        L_row_ptr=L_row_ptr,
        L_col_idx=L_col_idx,
        L_csr_to_csc=L_csr_to_csc,
        N_off_col_ptr=N_off_col_ptr,
        N_off_row_idx=N_off_row_idx,
        N_off_col_idx=N_off_col_idx,
        N_off_to_L=N_off_to_L,
        parent=parent,
        pred_diag_ptr=pred_diag_ptr,
        pred_diag_slot=pred_diag_slot,
        pred_off_ptr=pred_off_ptr,
        pred_off_slot_IK=pred_off_slot_IK,
        pred_off_slot_JK=pred_off_slot_JK,
        num_levels=nl1,
        level_ptr=level_ptr,
        level_pivots=level_pivots,
        rsi_ptr=rsi_ptr,
        rsi_e_row=rsi_e_row,
        rsi_l_col=rsi_l_col,
        rsi_target_pivot=rsi_target_pivot,
        rsi_target_is_diag=rsi_target_is_diag,
        rsi_target_local_row=rsi_target_local_row,
        rsi_target_local_col=rsi_target_local_col,
        nnz_L=nnzL,
        nnz_N=nnzN,
    )


__all__ = [
    "BlockSparseSymbolic",
    "compute_block_sparse_symbolic",
    "tile_symbolic_for_worlds",
]
