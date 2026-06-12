# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Symbolic factorization for sparse block LDL solver.

This module implements the symbolic phase of sparse block LDL decomposition:
1. Constraint graph construction from joint topology
2. MECA (Minimum Edge Creation Algorithm) elimination ordering
3. Fill-in pattern computation
4. RSI (Reduction Scattering Indexation) array generation

The symbolic phase runs once per topology change and produces data structures
that the numeric phase uses for efficient GPU execution.

"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np


@dataclass
class ConstraintGraph:
    """Constraint graph representation.

    Nodes are joints (constraint blocks), edges connect joints sharing a body.
    This is the edge graph of the body graph.
    """

    num_nodes: int
    adjacency: dict[int, set[int]] = field(default_factory=dict)

    def add_edge(self, i: int, j: int):
        """Add undirected edge between nodes i and j."""
        if i == j:
            return
        if i not in self.adjacency:
            self.adjacency[i] = set()
        if j not in self.adjacency:
            self.adjacency[j] = set()
        self.adjacency[i].add(j)
        self.adjacency[j].add(i)

    def remove_node(self, node: int):
        """Remove node and all its edges."""
        if node in self.adjacency:
            for neighbor in self.adjacency[node]:
                if neighbor in self.adjacency:
                    self.adjacency[neighbor].discard(node)
            del self.adjacency[node]

    def neighbors(self, node: int) -> set[int]:
        """Get neighbors of a node."""
        return self.adjacency.get(node, set())

    def degree(self, node: int) -> int:
        """Get degree (number of neighbors) of a node."""
        return len(self.adjacency.get(node, set()))

    def copy(self) -> "ConstraintGraph":
        """Create a deep copy of the graph."""
        new_graph = ConstraintGraph(num_nodes=self.num_nodes)
        for node, neighbors in self.adjacency.items():
            new_graph.adjacency[node] = neighbors.copy()
        return new_graph


@dataclass
class PivotInfo:
    """Information about a single pivot in the elimination sequence."""

    joint_idx: int
    degree: int  # Number of constraints for this joint (1-6)
    height: int  # Number of edges to eliminate
    edge_start: int  # Start index in edge arrays
    rsi_start: int  # Start index in RSI arrays
    rsi_count: int  # Number of RSI entries for this pivot


@dataclass
class SparseBlockSymbolic:
    """Result of symbolic factorization for sparse block LDL.

    Contains all the data structures needed for the numeric phase.
    """

    # Basic dimensions
    num_joints: int
    total_nc: int  # Total number of scalar constraints

    # Pivot ordering
    pivot_order: np.ndarray  # [num_pivots] - joint indices in elim order
    pivot_degree: np.ndarray  # [num_pivots] - constraint count per pivot
    pivot_height: np.ndarray  # [num_pivots] - edge count per pivot

    # Block offsets (for converting between block and scalar indices)
    block_offsets: np.ndarray  # [num_joints+1] - cumulative constraint counts

    # Edge storage layout (column-major order, grouped by pivot)
    num_edges: int
    edge_pivot: np.ndarray  # [num_edges] - which pivot owns this edge
    edge_neighbor: np.ndarray  # [num_edges] - neighbor joint index
    edge_ptr: np.ndarray  # [num_pivots+1] - start of edges for each pivot

    # Fill-in edges (created during elimination)
    fill_edges: set  # Set of (i, j) pairs that are fill-in

    # RSI (Reduction Scattering Indexation)
    rsi_ptr: np.ndarray  # [num_pivots+1] - start of RSI for each pivot
    rsi_e_row: np.ndarray  # [total_rsi] - row index in E matrix
    rsi_l_col: np.ndarray  # [total_rsi] - column index in L^T matrix
    rsi_target_pivot: np.ndarray  # [total_rsi] - target pivot for scatter
    rsi_target_is_diag: np.ndarray  # [total_rsi] - is diagonal block
    rsi_target_local_row: np.ndarray  # [total_rsi] - local row in target
    rsi_target_local_col: np.ndarray  # [total_rsi] - local col in target

    # Memory layout for numeric arrays
    node_storage_size: int  # Total floats for node blocks
    node_ptr: np.ndarray  # [num_joints+1] - start of each node block
    edge_storage_size: int  # Total floats for edge blocks
    edge_block_ptr: np.ndarray  # [num_edges+1] - start of each edge block

    # Elimination tree
    parent: np.ndarray  # [num_joints] - parent in elimination tree


def build_constraint_graph(
    joint_body_a: np.ndarray,
    joint_body_b: np.ndarray,
) -> ConstraintGraph:
    """Build constraint graph from joint connectivity.

    The constraint graph has joints as nodes. Two joints are connected
    by an edge if they share a common body.

    Args:
        joint_body_a: Parent body index for each joint (-1 if world).
        joint_body_b: Child body index for each joint.

    Returns:
        ConstraintGraph representing joint connectivity.
    """
    nj = len(joint_body_a)
    graph = ConstraintGraph(num_nodes=nj)

    # Build body-to-joints mapping
    body_to_joints: dict[int, list[int]] = defaultdict(list)
    for j in range(nj):
        if joint_body_a[j] >= 0:
            body_to_joints[joint_body_a[j]].append(j)
        if joint_body_b[j] >= 0:
            body_to_joints[joint_body_b[j]].append(j)

    # Add edges between joints that share a body
    for joints in body_to_joints.values():
        for i, j1 in enumerate(joints):
            for j2 in joints[i + 1 :]:
                graph.add_edge(j1, j2)

    return graph


def meca_ordering(
    graph: ConstraintGraph,
) -> tuple[list[int], list[set[tuple[int, int]]]]:
    """Compute elimination ordering using MECA heuristic.

    MECA (Minimum Edge Creation Algorithm) chooses at each step the pivot
    that creates the minimum number of new fill-in edges.

    Args:
        graph: The constraint graph.

    Returns:
        pivot_order: List of node indices in elimination order.
        fill_edges_per_pivot: For each pivot, set of fill-in edges created.
    """
    remaining = set(range(graph.num_nodes))
    order: list[int] = []
    fill_edges_per_pivot: list[set[tuple[int, int]]] = []

    work_graph = graph.copy()

    while remaining:
        # Find node with minimum fill-in
        best_node = None
        best_fill_count = float("inf")
        best_fill_edges: set[tuple[int, int]] = set()

        for node in remaining:
            neighbors = work_graph.neighbors(node) & remaining
            # Count fill-in: edges that would be created between neighbors
            fill_edges: set[tuple[int, int]] = set()
            neighbors_list = sorted(neighbors)
            for i, n1 in enumerate(neighbors_list):
                for n2 in neighbors_list[i + 1 :]:
                    if n2 not in work_graph.neighbors(n1):
                        # This edge doesn't exist, would be fill-in
                        fill_edges.add((min(n1, n2), max(n1, n2)))

            fill_count = len(fill_edges)
            if fill_count < best_fill_count:
                best_fill_count = fill_count
                best_node = node
                best_fill_edges = fill_edges

        if best_node is None:
            break

        # Eliminate this node
        order.append(best_node)
        fill_edges_per_pivot.append(best_fill_edges)

        # Add fill-in edges to graph
        for n1, n2 in best_fill_edges:
            work_graph.add_edge(n1, n2)

        # Remove node from remaining set
        remaining.remove(best_node)

    return order, fill_edges_per_pivot


def compute_symbolic_factorization(
    joint_body_a: np.ndarray,
    joint_body_b: np.ndarray,
    joint_num_constraints: np.ndarray,
    use_meca: bool = True,
) -> SparseBlockSymbolic:
    """Compute symbolic factorization for sparse block LDL.

    This is the main entry point for the symbolic phase. It:
    1. Builds the constraint graph
    2. Computes elimination ordering (MECA or natural)
    3. Determines fill-in pattern
    4. Computes RSI arrays for the numeric phase
    5. Allocates storage layout

    Args:
        joint_body_a: Parent body index for each joint (-1 if world).
        joint_body_b: Child body index for each joint.
        joint_num_constraints: Number of constraints per joint [nj].
        use_meca: If True, use MECA ordering. If False, use natural order.

    Returns:
        SparseBlockSymbolic with all symbolic data.
    """
    nj = len(joint_body_a)

    if nj == 0:
        return _create_empty_symbolic()

    # Build constraint graph
    graph = build_constraint_graph(joint_body_a, joint_body_b)

    # Compute elimination ordering
    if use_meca:
        pivot_order, fill_edges_per_pivot = meca_ordering(graph)
    else:
        # Natural ordering (joint index order)
        pivot_order = list(range(nj))
        fill_edges_per_pivot = [set() for _ in range(nj)]
        # Compute fill-in for natural ordering
        work_graph = graph.copy()
        for i, pivot in enumerate(pivot_order):
            neighbors = work_graph.neighbors(pivot)
            neighbors_list = sorted(neighbors)
            for i1, n1 in enumerate(neighbors_list):
                for n2 in neighbors_list[i1 + 1 :]:
                    if n2 not in work_graph.neighbors(n1):
                        fill_edges_per_pivot[i].add((min(n1, n2), max(n1, n2)))
                        work_graph.add_edge(n1, n2)

    # Compute block offsets
    block_offsets = np.zeros(nj + 1, dtype=np.int32)
    for j in range(nj):
        block_offsets[j + 1] = block_offsets[j] + joint_num_constraints[j]
    total_nc = int(block_offsets[nj])

    # Compute pivot info and edges
    pivot_order_arr = np.array(pivot_order, dtype=np.int32)
    pivot_degree = joint_num_constraints[pivot_order_arr].astype(np.int32)

    # Track remaining neighbors at each elimination step
    work_graph = graph.copy()
    all_fill_edges: set[tuple[int, int]] = set()

    edges_list: list[tuple[int, int]] = []
    edge_ptr_list: list[int] = [0]
    pivot_height_list: list[int] = []

    eliminated: set[int] = set()
    for pivot_idx, pivot_joint in enumerate(pivot_order):
        # Get neighbors that haven't been eliminated yet
        neighbors = work_graph.neighbors(pivot_joint) - eliminated
        neighbors_sorted = sorted(neighbors)

        pivot_height_list.append(len(neighbors_sorted))

        # Record edges for this pivot
        for neighbor in neighbors_sorted:
            edges_list.append((pivot_idx, neighbor))

        edge_ptr_list.append(len(edges_list))

        # Add fill-in edges
        for n1, n2 in fill_edges_per_pivot[pivot_idx]:
            work_graph.add_edge(n1, n2)
            all_fill_edges.add((n1, n2))

        eliminated.add(pivot_joint)

    num_edges = len(edges_list)
    pivot_height = np.array(pivot_height_list, dtype=np.int32)
    edge_ptr = np.array(edge_ptr_list, dtype=np.int32)

    edge_pivot = np.zeros(max(num_edges, 1), dtype=np.int32)
    edge_neighbor = np.zeros(max(num_edges, 1), dtype=np.int32)
    for e_idx, (p_idx, neighbor) in enumerate(edges_list):
        edge_pivot[e_idx] = p_idx
        edge_neighbor[e_idx] = neighbor

    # Compute elimination tree (parent = first non-eliminated neighbor)
    parent = np.full(nj, -1, dtype=np.int32)
    for pivot_idx in range(nj):
        start = edge_ptr[pivot_idx]
        end = edge_ptr[pivot_idx + 1]
        if start < end:
            parent[pivot_order[pivot_idx]] = edge_neighbor[start]

    # Compute RSI arrays
    rsi_ptr_list: list[int] = [0]
    rsi_e_row_list: list[int] = []
    rsi_l_col_list: list[int] = []
    rsi_target_pivot_list: list[int] = []
    rsi_target_is_diag_list: list[int] = []
    rsi_target_local_row_list: list[int] = []
    rsi_target_local_col_list: list[int] = []

    for pivot_idx in range(nj):
        e_start = edge_ptr[pivot_idx]
        e_end = edge_ptr[pivot_idx + 1]
        height = e_end - e_start

        if height == 0:
            rsi_ptr_list.append(len(rsi_e_row_list))
            continue

        # Get neighbors for this pivot
        neighbors_list = [edge_neighbor[e] for e in range(e_start, e_end)]

        # Iterate over the lower triangle of the outer product
        for i in range(height):
            for j in range(i + 1):  # j <= i for lower triangle
                ni = neighbors_list[i]
                nj_neighbor = neighbors_list[j]

                rsi_e_row_list.append(i)
                rsi_l_col_list.append(j)

                if i == j:
                    # Diagonal: goes to N[ni, ni] diagonal block
                    rsi_target_pivot_list.append(ni)
                    rsi_target_is_diag_list.append(1)
                    rsi_target_local_row_list.append(0)
                    rsi_target_local_col_list.append(0)
                else:
                    # Off-diagonal
                    rsi_target_pivot_list.append(ni)
                    rsi_target_is_diag_list.append(0)
                    rsi_target_local_row_list.append(nj_neighbor)
                    rsi_target_local_col_list.append(0)

        rsi_ptr_list.append(len(rsi_e_row_list))

    rsi_ptr = np.array(rsi_ptr_list, dtype=np.int32)
    rsi_e_row = np.array(rsi_e_row_list if rsi_e_row_list else [0], dtype=np.int32)
    rsi_l_col = np.array(rsi_l_col_list if rsi_l_col_list else [0], dtype=np.int32)
    rsi_target_pivot = np.array(rsi_target_pivot_list if rsi_target_pivot_list else [0], dtype=np.int32)
    rsi_target_is_diag = np.array(rsi_target_is_diag_list if rsi_target_is_diag_list else [0], dtype=np.int32)
    rsi_target_local_row = np.array(rsi_target_local_row_list if rsi_target_local_row_list else [0], dtype=np.int32)
    rsi_target_local_col = np.array(rsi_target_local_col_list if rsi_target_local_col_list else [0], dtype=np.int32)

    # Compute storage layout
    node_ptr = np.zeros(nj + 1, dtype=np.int32)
    for j in range(nj):
        d = joint_num_constraints[j]
        node_ptr[j + 1] = node_ptr[j] + d * d
    node_storage_size = int(node_ptr[nj])

    # Edge blocks
    edge_block_ptr = np.zeros(max(num_edges + 1, 2), dtype=np.int32)
    for e_idx in range(num_edges):
        p_idx = edge_pivot[e_idx]
        neighbor_joint = edge_neighbor[e_idx]
        p_deg = joint_num_constraints[pivot_order[p_idx]]
        n_deg = joint_num_constraints[neighbor_joint]
        edge_block_ptr[e_idx + 1] = edge_block_ptr[e_idx] + n_deg * p_deg
    edge_storage_size = int(edge_block_ptr[num_edges]) if num_edges > 0 else 0

    return SparseBlockSymbolic(
        num_joints=nj,
        total_nc=total_nc,
        pivot_order=pivot_order_arr,
        pivot_degree=pivot_degree,
        pivot_height=pivot_height,
        block_offsets=block_offsets,
        num_edges=num_edges,
        edge_pivot=edge_pivot,
        edge_neighbor=edge_neighbor,
        edge_ptr=edge_ptr,
        fill_edges=all_fill_edges,
        rsi_ptr=rsi_ptr,
        rsi_e_row=rsi_e_row,
        rsi_l_col=rsi_l_col,
        rsi_target_pivot=rsi_target_pivot,
        rsi_target_is_diag=rsi_target_is_diag,
        rsi_target_local_row=rsi_target_local_row,
        rsi_target_local_col=rsi_target_local_col,
        node_storage_size=node_storage_size,
        node_ptr=node_ptr,
        edge_storage_size=edge_storage_size,
        edge_block_ptr=edge_block_ptr,
        parent=parent,
    )


def _create_empty_symbolic() -> SparseBlockSymbolic:
    """Create empty symbolic for zero-joint case."""
    return SparseBlockSymbolic(
        num_joints=0,
        total_nc=0,
        pivot_order=np.zeros(0, dtype=np.int32),
        pivot_degree=np.zeros(0, dtype=np.int32),
        pivot_height=np.zeros(0, dtype=np.int32),
        block_offsets=np.zeros(1, dtype=np.int32),
        num_edges=0,
        edge_pivot=np.zeros(1, dtype=np.int32),
        edge_neighbor=np.zeros(1, dtype=np.int32),
        edge_ptr=np.zeros(1, dtype=np.int32),
        fill_edges=set(),
        rsi_ptr=np.zeros(1, dtype=np.int32),
        rsi_e_row=np.zeros(1, dtype=np.int32),
        rsi_l_col=np.zeros(1, dtype=np.int32),
        rsi_target_pivot=np.zeros(1, dtype=np.int32),
        rsi_target_is_diag=np.zeros(1, dtype=np.int32),
        rsi_target_local_row=np.zeros(1, dtype=np.int32),
        rsi_target_local_col=np.zeros(1, dtype=np.int32),
        node_storage_size=0,
        node_ptr=np.zeros(1, dtype=np.int32),
        edge_storage_size=0,
        edge_block_ptr=np.zeros(2, dtype=np.int32),
        parent=np.zeros(0, dtype=np.int32),
    )
