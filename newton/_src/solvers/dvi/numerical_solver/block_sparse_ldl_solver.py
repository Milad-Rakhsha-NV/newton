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

"""Block-sparse tile LDL solver class.

Plugs into the dvi numerical solver framework via the
:class:`NumericalSolver` interface. Single-world by default; pass
``num_worlds > 1`` (or set it on the model) to use the batched code path.

The solver requires a one-time ``prepare_for_capture`` call to run symbolic
factorization and allocate device buffers; afterwards ``solve`` issues only
kernel launches and is safe to record into a CUDA graph.

Internally the algorithm is block Cholesky on the SPD matrix
``N = J M^{-1} J^T + reg I`` with ``reg > 0``, which is mathematically
identical to LDL up to a sign-free split; the public name retains "LDL" for
continuity with the rest of the dvi numerical solver lineup.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from .base import Device, NumericalSolver, NumericalSolverConfig, SolveContext
from .block_sparse_kernels import (
    apply_precond_add_reg_kernel,
    apply_precond_scale_N_diag_kernel,
    apply_precond_scale_N_off_kernel,
    apply_precond_scale_rhs_kernel,
    apply_precond_unscale_x_kernel,
    assemble_N_diag_kernel,
    assemble_N_off_kernel,
    compute_precond_scale_kernel,
    copy_N_off_to_L_kernel,
    zero_L_blocks_kernel,
)
from .block_sparse_ldl_batched import (
    batched_apply_precond_add_reg_kernel,
    batched_apply_precond_scale_N_diag_kernel,
    batched_apply_precond_scale_N_off_kernel,
    batched_apply_precond_scale_rhs_kernel,
    batched_apply_precond_unscale_x_kernel,
    batched_assemble_N_diag_kernel,
    batched_assemble_N_off_kernel,
    batched_block_backward_sub_kernel,
    batched_block_forward_sub_kernel,
    batched_compute_precond_scale_kernel,
    batched_copy_N_off_to_L_kernel,
    batched_gather_rhs_kernel,
    batched_ldl_factor_level_kernel,
    batched_scatter_x_kernel,
    batched_zero_L_blocks_kernel,
)
from .block_sparse_ldl_kernels import (
    block_backward_sub_level_kernel,
    block_forward_sub_level_kernel,
    compute_delta_v_from_lambda_kernel,
    compute_refinement_residual_kernel,
    gather_rhs_in_pivot_order_kernel,
    ldl_factor_level_kernel,
    scatter_x_add_kernel,
    scatter_x_kernel,
)
from .block_sparse_storage import (
    BatchedBlockSparseStorage,
    BlockSparseStorage,
    allocate_batched_storage,
    allocate_storage,
)
from .block_sparse_symbolic import BlockSparseSymbolic, compute_block_sparse_symbolic, tile_symbolic_for_worlds


def _block_nc_from_model(model) -> np.ndarray:
    """Compute live constraint count per joint from a dvi ``Model``.

    Mirrors the logic in ``JointConstraintInfo.from_model`` but returns a
    NumPy array on host so it can drive the symbolic phase. Used as a
    fallback when ``prepare_for_capture`` is called without a precomputed
    ``constraint_info``.
    """
    nj = model.joint_count if model.joint_count > 0 else 0
    if nj == 0:
        return np.zeros(0, dtype=np.int32)

    joint_type = model.joint_type.numpy()
    joint_enabled = model.joint_enabled.numpy()
    joint_dof_dim = model.joint_dof_dim.numpy()

    nc_per_joint = np.zeros(nj, dtype=np.int32)
    for j in range(nj):
        if not joint_enabled[j]:
            continue
        jtype = int(joint_type[j])
        n_lin, n_ang = int(joint_dof_dim[j, 0]), int(joint_dof_dim[j, 1])
        if jtype == 4 or jtype == 5:
            nc_per_joint[j] = 0
        elif jtype == 3:
            nc_per_joint[j] = 6
        elif jtype == 2:
            nc_per_joint[j] = 3
        elif jtype == 1 or jtype == 0:
            nc_per_joint[j] = 5
        elif jtype == 6:
            lin_nc = 3 if n_lin == 0 else (0 if n_lin == 3 else 2)
            ang_nc = 3 if n_ang == 0 else (0 if n_ang == 3 else 2)
            nc_per_joint[j] = lin_nc + ang_nc
    return nc_per_joint


class SparseLDLSolver(NumericalSolver):
    """Block-sparse tile-based direct solver for joint constraints.

    Satisfies the six-pillar contract:

    1. Block-sparse storage in CSC form for L and N's off-diagonal pattern.
    2. All numeric work runs in Warp kernels (no NumPy at runtime).
    3. The numeric ``solve`` path issues only kernel launches; symbolic and
       allocation happen in ``prepare_for_capture``.
    4. Numeric ``solve`` is CUDA-graph capturable: no host-device sync, no
       allocations, fixed launch grid sizes.
    5. Numeric kernels use ``wp.tile_*`` ops on 6x6 tiles (cooperative tile
       loads / matmul / Cholesky / triangular solves).
    6. Multi-world (cloned topology) reuses one symbolic factorization with
       per-world numeric arrays; one tiled launch covers all worlds at a
       given elimination tree depth.

    Args:
        config: Numerical solver configuration. Only ``reg`` is consumed.
        use_meca: If ``True``, use MECA fill-in-minimizing ordering. If
            ``False``, use natural joint order.
        num_worlds: Override for multi-world batching. ``None`` uses
            ``getattr(model, 'num_worlds', 1)`` at ``prepare_for_capture`` time.
    """

    supported_devices = {Device.CPU, Device.CUDA}
    default_params = {"reg": 1e-8}

    def __init__(
        self,
        config: NumericalSolverConfig,
        *,
        use_meca: bool = False,
        num_worlds: int | None = None,
    ):
        super().__init__(config)
        self._use_meca = use_meca
        self._num_worlds_override = num_worlds
        self._symbolic: BlockSparseSymbolic | None = None
        self._storage: BlockSparseStorage | BatchedBlockSparseStorage | None = None
        self._batched: bool = False
        self._device: str | None = None
        self._prepared: bool = False
        self._joint_body_a_dev: wp.array | None = None
        self._joint_body_b_dev: wp.array | None = None
        self._block_nc_dev: wp.array | None = None
        self._joint_count: int = 0

    @property
    def is_prepared(self) -> bool:
        """Whether ``prepare_for_capture`` has been called."""
        return self._prepared

    @property
    def symbolic(self) -> BlockSparseSymbolic | None:
        """The unified symbolic factorization (None until prepared)."""
        return self._symbolic

    @property
    def storage(self) -> BlockSparseStorage | BatchedBlockSparseStorage | None:
        """The numeric / device-mirror storage (None until prepared)."""
        return self._storage

    def prepare_for_capture(self, model, state, constraint_info=None, body_inv_inertia_world=None) -> None:
        """Run symbolic factorization and allocate device buffers.

        Reads joint topology from the model (one host-device transfer for the
        constraint counts if ``constraint_info`` is not provided), computes
        the block-sparse symbolic factorization on the host, and allocates
        all numeric arrays on the device. Idempotent.
        """
        if self._prepared:
            return

        device = model.device
        nj = int(model.joint_count) if model.joint_count > 0 else 0

        if nj == 0:
            self._symbolic = compute_block_sparse_symbolic(
                np.zeros(0, dtype=np.int32),
                np.zeros(0, dtype=np.int32),
                np.zeros(0, dtype=np.int32),
                use_meca=self._use_meca,
            )
            self._device = str(device)
            self._batched = False
            self._joint_count = 0
            self._prepared = True
            return

        joint_body_a_np = model.joint_parent.numpy().astype(np.int32)
        joint_body_b_np = model.joint_child.numpy().astype(np.int32)

        if constraint_info is not None and getattr(constraint_info, "joint_num_constraints", None) is not None:
            block_nc_np = constraint_info.joint_num_constraints.numpy().astype(np.int32)
        else:
            block_nc_np = _block_nc_from_model(model)

        # Detect multi-world with identical topology: compute symbolic
        # on world 0 only, then tile for all worlds.  This reduces the
        # O(n²+) Python symbolic phase from minutes to milliseconds at
        # high world counts.
        world_count = int(getattr(model, "world_count", 1))
        if world_count > 1 and hasattr(model, "joint_world_start") and hasattr(model, "body_world_start"):
            jws = model.joint_world_start.numpy()
            bws = model.body_world_start.numpy()
            jpw = int(jws[1] - jws[0])  # joints per world
            bpw = int(bws[1] - bws[0])  # bodies per world

            # Verify all worlds have the same size (homogeneous)
            sizes_ok = all(
                int(jws[w + 1] - jws[w]) == jpw and int(bws[w + 1] - bws[w]) == bpw for w in range(world_count)
            )

            if sizes_ok and jpw > 0:
                # Extract world 0 joints with world-local body indices
                j0s, j0e = int(jws[0]), int(jws[1])
                b0s = int(bws[0])
                a0 = joint_body_a_np[j0s:j0e].copy()
                b0 = joint_body_b_np[j0s:j0e].copy()
                nc0 = block_nc_np[j0s:j0e].copy()
                # Make body indices world-local
                a0[a0 >= 0] -= b0s
                b0 -= b0s

                sym_one = compute_block_sparse_symbolic(a0, b0, nc0, use_meca=self._use_meca)
                self._symbolic = tile_symbolic_for_worlds(sym_one, world_count, jpw)
            else:
                # Non-homogeneous worlds: fall back to full symbolic
                self._symbolic = compute_block_sparse_symbolic(
                    joint_body_a_np,
                    joint_body_b_np,
                    block_nc_np,
                    use_meca=self._use_meca,
                )
        else:
            self._symbolic = compute_block_sparse_symbolic(
                joint_body_a_np,
                joint_body_b_np,
                block_nc_np,
                use_meca=self._use_meca,
            )

        if self._num_worlds_override is not None:
            num_worlds = int(self._num_worlds_override)
        else:
            num_worlds = int(getattr(model, "num_worlds", 1))
        self._batched = num_worlds > 1

        # Compute total constraint rows for refinement buffer allocation
        total_nc = int(block_nc_np.sum()) if self.config.iterative_refinement_steps > 0 else 0

        if self._batched:
            self._storage = allocate_batched_storage(self._symbolic, device, num_worlds=num_worlds)
        else:
            self._storage = allocate_storage(self._symbolic, device, total_nc=total_nc)

        with wp.ScopedDevice(device):
            self._joint_body_a_dev = wp.array(joint_body_a_np, dtype=wp.int32, device=device)
            self._joint_body_b_dev = wp.array(joint_body_b_np, dtype=wp.int32, device=device)
            self._block_nc_dev = wp.array(block_nc_np, dtype=wp.int32, device=device)

        self._device = str(device)
        self._joint_count = nj
        self._prepared = True

    def solve(self, ctx: SolveContext) -> int:
        """Solve ``N x = b`` once.

        Issues kernels only; safe inside a CUDA graph capture region. No
        device-host transfers, no allocations. ``prepare_for_capture`` must
        have been called first; the orchestrator (``ConstraintSolver``,
        ``PositionSolver``) does this automatically on the first solve.
        """
        self.validate(ctx, require_matrix_free=True)

        if not self._prepared:
            raise RuntimeError("SparseLDLSolver.prepare_for_capture must be called before solve().")

        if self._symbolic is None or self._symbolic.num_joints == 0:
            return 1

        if self._batched:
            return self._solve_batched(ctx)
        return self._solve_single(ctx)

    def _solve_single(self, ctx: SolveContext) -> int:
        sym = self._symbolic
        st: BlockSparseStorage = self._storage  # type: ignore[assignment]
        device = ctx.device
        num_pivots = sym.num_joints
        nnz_L = max(sym.nnz_L, 1)
        nnz_N = max(sym.nnz_N, 1)
        joint_count = self._joint_count

        wp.launch(
            kernel=assemble_N_diag_kernel,
            dim=num_pivots,
            inputs=[
                ctx.J,
                ctx.M_inv_diag,
                ctx.M_inv_inertia,
                self._joint_body_a_dev,
                self._joint_body_b_dev,
                ctx.nc_offset,
                st.pivot_order_dev,
                st.block_sizes_dev,
                wp.float32(self.config.reg),
                num_pivots,
            ],
            outputs=[st.N_diag],
            device=device,
        )
        wp.launch(
            kernel=assemble_N_off_kernel,
            dim=nnz_N,
            inputs=[
                ctx.J,
                ctx.M_inv_diag,
                ctx.M_inv_inertia,
                self._joint_body_a_dev,
                self._joint_body_b_dev,
                ctx.nc_offset,
                st.pivot_order_dev,
                st.block_sizes_dev,
                st.N_off_row_idx_dev,
                st.N_off_col_idx_dev,
                nnz_N,
            ],
            outputs=[st.N_off],
            device=device,
        )

        # --- Diagonal preconditioning: S N S where S = diag(1/sqrt(N_ii)) ---
        use_precond = self.config.diagonal_precondition
        if use_precond:
            wp.launch(
                kernel=compute_precond_scale_kernel,
                dim=num_pivots,
                inputs=[st.N_diag, st.block_sizes_dev, num_pivots],
                outputs=[st.scale_blocks],
                device=device,
            )
            wp.launch(
                kernel=apply_precond_scale_N_diag_kernel,
                dim=num_pivots,
                inputs=[st.scale_blocks, num_pivots],
                outputs=[st.N_diag],
                device=device,
            )
            wp.launch(
                kernel=apply_precond_scale_N_off_kernel,
                dim=max(nnz_N, 1),
                inputs=[st.scale_blocks, st.N_off_row_idx_dev, st.N_off_col_idx_dev, nnz_N],
                outputs=[st.N_off],
                device=device,
            )
            # Add regularization in the scaled system to prevent near-zero pivots
            precond_reg = self.config.precond_reg
            if precond_reg > 0.0:
                wp.launch(
                    kernel=apply_precond_add_reg_kernel,
                    dim=num_pivots,
                    inputs=[st.block_sizes_dev, wp.float32(precond_reg), num_pivots],
                    outputs=[st.N_diag],
                    device=device,
                )

        wp.launch(
            kernel=zero_L_blocks_kernel,
            dim=nnz_L,
            inputs=[st.L_blocks, nnz_L],
            device=device,
        )
        wp.launch(
            kernel=copy_N_off_to_L_kernel,
            dim=nnz_N,
            inputs=[st.N_off, st.N_off_to_L_dev, nnz_N],
            outputs=[st.L_blocks],
            device=device,
        )

        for level in range(sym.num_levels - 1, -1, -1):
            count = int(sym.level_ptr[level + 1] - sym.level_ptr[level])
            if count == 0:
                continue
            wp.launch_tiled(
                kernel=ldl_factor_level_kernel,
                dim=count,
                inputs=[
                    st.L_col_ptr_dev,
                    st.L_row_idx_dev,
                    st.pred_diag_ptr_dev,
                    st.pred_diag_slot_dev,
                    st.pred_off_ptr_dev,
                    st.pred_off_slot_IK_dev,
                    st.pred_off_slot_JK_dev,
                    st.level_pivots_dev,
                    count,
                    int(sym.level_ptr[level]),
                    st.N_diag,
                ],
                outputs=[st.D_blocks, st.L_blocks],
                block_dim=128,
                device=device,
            )

        wp.launch(
            kernel=gather_rhs_in_pivot_order_kernel,
            dim=num_pivots,
            inputs=[
                ctx.b,
                ctx.nc_offset,
                st.pivot_order_dev,
                st.block_sizes_dev,
                num_pivots,
            ],
            outputs=[st.rhs_blocks],
            device=device,
        )

        # --- Scale RHS ---
        if use_precond:
            wp.launch(
                kernel=apply_precond_scale_rhs_kernel,
                dim=num_pivots,
                inputs=[st.scale_blocks, num_pivots],
                outputs=[st.rhs_blocks],
                device=device,
            )

        for level in range(sym.num_levels - 1, -1, -1):
            count = int(sym.level_ptr[level + 1] - sym.level_ptr[level])
            if count == 0:
                continue
            wp.launch_tiled(
                kernel=block_forward_sub_level_kernel,
                dim=count,
                inputs=[
                    st.L_row_ptr_dev,
                    st.L_col_idx_dev,
                    st.L_csr_to_csc_dev,
                    st.level_pivots_dev,
                    count,
                    int(sym.level_ptr[level]),
                    st.L_blocks,
                    st.D_blocks,
                    st.rhs_blocks,
                ],
                outputs=[st.y_blocks],
                block_dim=128,
                device=device,
            )

        for level in range(0, sym.num_levels):
            count = int(sym.level_ptr[level + 1] - sym.level_ptr[level])
            if count == 0:
                continue
            wp.launch_tiled(
                kernel=block_backward_sub_level_kernel,
                dim=count,
                inputs=[
                    st.L_col_ptr_dev,
                    st.L_row_idx_dev,
                    st.level_pivots_dev,
                    count,
                    int(sym.level_ptr[level]),
                    st.L_blocks,
                    st.D_blocks,
                    st.y_blocks,
                ],
                outputs=[st.x_blocks],
                block_dim=128,
                device=device,
            )

        # --- Unscale solution: x = S * x_tilde ---
        if use_precond:
            wp.launch(
                kernel=apply_precond_unscale_x_kernel,
                dim=num_pivots,
                inputs=[st.scale_blocks, num_pivots],
                outputs=[st.x_blocks],
                device=device,
            )

        wp.launch(
            kernel=scatter_x_kernel,
            dim=num_pivots,
            inputs=[
                st.x_blocks,
                st.pivot_order_dev,
                ctx.nc_offset,
                st.block_sizes_dev,
                num_pivots,
            ],
            outputs=[ctx.x],
            device=device,
        )

        if ctx.delta_v is not None:
            ctx.delta_v.zero_()
            wp.launch(
                kernel=compute_delta_v_from_lambda_kernel,
                dim=joint_count,
                inputs=[
                    ctx.M_inv_diag,
                    ctx.M_inv_inertia,
                    ctx.J,
                    self._joint_body_a_dev,
                    self._joint_body_b_dev,
                    self._block_nc_dev,
                    ctx.nc_offset,
                    ctx.x,
                    joint_count,
                ],
                outputs=[ctx.delta_v],
                device=device,
            )

        # =====================================================================
        # Iterative refinement: recover precision lost in float32 LDL.
        #
        # For each step:
        #   1. Compute residual r = b - N@x matrix-free (via delta_v)
        #   2. Gather r into pivot-order rhs_blocks
        #   3. Apply preconditioning scale (if enabled)
        #   4. Forward/backward substitution with existing L, D factors
        #   5. Unscale correction (if preconditioning)
        #   6. Scatter-add correction to ctx.x
        #   7. Recompute delta_v = M⁻¹ Jᵀ x
        # =====================================================================
        refinement_steps = self.config.iterative_refinement_steps
        if refinement_steps > 0 and ctx.delta_v is not None:
            for _ref_step in range(refinement_steps):
                # 1. Compute residual: r = b - J @ delta_v  (= b - N @ x)
                wp.launch(
                    kernel=compute_refinement_residual_kernel,
                    dim=joint_count,
                    inputs=[
                        ctx.J,
                        self._joint_body_a_dev,
                        self._joint_body_b_dev,
                        self._block_nc_dev,
                        ctx.nc_offset,
                        ctx.b,
                        ctx.delta_v,
                        joint_count,
                    ],
                    outputs=[st._refinement_residual],
                    device=device,
                )

                # 2. Gather residual into pivot-order rhs_blocks
                wp.launch(
                    kernel=gather_rhs_in_pivot_order_kernel,
                    dim=num_pivots,
                    inputs=[
                        st._refinement_residual,
                        ctx.nc_offset,
                        st.pivot_order_dev,
                        st.block_sizes_dev,
                        num_pivots,
                    ],
                    outputs=[st.rhs_blocks],
                    device=device,
                )

                # 3. Scale RHS (if preconditioning)
                if use_precond:
                    wp.launch(
                        kernel=apply_precond_scale_rhs_kernel,
                        dim=num_pivots,
                        inputs=[st.scale_blocks, num_pivots],
                        outputs=[st.rhs_blocks],
                        device=device,
                    )

                # 4a. Forward substitution with existing L, D
                for level in range(sym.num_levels - 1, -1, -1):
                    count = int(sym.level_ptr[level + 1] - sym.level_ptr[level])
                    if count == 0:
                        continue
                    wp.launch_tiled(
                        kernel=block_forward_sub_level_kernel,
                        dim=count,
                        inputs=[
                            st.L_row_ptr_dev,
                            st.L_col_idx_dev,
                            st.L_csr_to_csc_dev,
                            st.level_pivots_dev,
                            count,
                            int(sym.level_ptr[level]),
                            st.L_blocks,
                            st.D_blocks,
                            st.rhs_blocks,
                        ],
                        outputs=[st.y_blocks],
                        block_dim=128,
                        device=device,
                    )

                # 4b. Backward substitution
                for level in range(0, sym.num_levels):
                    count = int(sym.level_ptr[level + 1] - sym.level_ptr[level])
                    if count == 0:
                        continue
                    wp.launch_tiled(
                        kernel=block_backward_sub_level_kernel,
                        dim=count,
                        inputs=[
                            st.L_col_ptr_dev,
                            st.L_row_idx_dev,
                            st.level_pivots_dev,
                            count,
                            int(sym.level_ptr[level]),
                            st.L_blocks,
                            st.D_blocks,
                            st.y_blocks,
                        ],
                        outputs=[st.x_blocks],
                        block_dim=128,
                        device=device,
                    )

                # 5. Unscale correction (if preconditioning)
                if use_precond:
                    wp.launch(
                        kernel=apply_precond_unscale_x_kernel,
                        dim=num_pivots,
                        inputs=[st.scale_blocks, num_pivots],
                        outputs=[st.x_blocks],
                        device=device,
                    )

                # 6. Scatter-add correction to ctx.x
                wp.launch(
                    kernel=scatter_x_add_kernel,
                    dim=num_pivots,
                    inputs=[
                        st.x_blocks,
                        st.pivot_order_dev,
                        ctx.nc_offset,
                        st.block_sizes_dev,
                        num_pivots,
                    ],
                    outputs=[ctx.x],
                    device=device,
                )

                # 7. Recompute delta_v from updated x
                ctx.delta_v.zero_()
                wp.launch(
                    kernel=compute_delta_v_from_lambda_kernel,
                    dim=joint_count,
                    inputs=[
                        ctx.M_inv_diag,
                        ctx.M_inv_inertia,
                        ctx.J,
                        self._joint_body_a_dev,
                        self._joint_body_b_dev,
                        self._block_nc_dev,
                        ctx.nc_offset,
                        ctx.x,
                        joint_count,
                    ],
                    outputs=[ctx.delta_v],
                    device=device,
                )

        return 1

    def _solve_batched(self, ctx: SolveContext) -> int:
        sym = self._symbolic
        st: BatchedBlockSparseStorage = self._storage  # type: ignore[assignment]
        device = ctx.device
        num_pivots = sym.num_joints
        nnz_L = max(sym.nnz_L, 1)
        nnz_N = max(sym.nnz_N, 1)
        nw = st.num_worlds

        wp.launch(
            kernel=batched_assemble_N_diag_kernel,
            dim=(nw, num_pivots),
            inputs=[
                ctx.J,
                ctx.M_inv_diag,
                ctx.M_inv_inertia,
                self._joint_body_a_dev,
                self._joint_body_b_dev,
                ctx.nc_offset,
                st.pivot_order_dev,
                st.block_sizes_dev,
                wp.float32(self.config.reg),
                num_pivots,
            ],
            outputs=[st.N_diag],
            device=device,
        )
        wp.launch(
            kernel=batched_assemble_N_off_kernel,
            dim=(nw, nnz_N),
            inputs=[
                ctx.J,
                ctx.M_inv_diag,
                ctx.M_inv_inertia,
                self._joint_body_a_dev,
                self._joint_body_b_dev,
                ctx.nc_offset,
                st.pivot_order_dev,
                st.block_sizes_dev,
                st.N_off_row_idx_dev,
                st.N_off_col_idx_dev,
                nnz_N,
            ],
            outputs=[st.N_off],
            device=device,
        )

        # --- Batched diagonal preconditioning ---
        use_precond = self.config.diagonal_precondition
        if use_precond:
            wp.launch(
                kernel=batched_compute_precond_scale_kernel,
                dim=(nw, num_pivots),
                inputs=[st.N_diag, st.block_sizes_dev, num_pivots],
                outputs=[st.scale_blocks],
                device=device,
            )
            wp.launch(
                kernel=batched_apply_precond_scale_N_diag_kernel,
                dim=(nw, num_pivots),
                inputs=[st.scale_blocks, num_pivots],
                outputs=[st.N_diag],
                device=device,
            )
            wp.launch(
                kernel=batched_apply_precond_scale_N_off_kernel,
                dim=(nw, max(nnz_N, 1)),
                inputs=[st.scale_blocks, st.N_off_row_idx_dev, st.N_off_col_idx_dev, nnz_N],
                outputs=[st.N_off],
                device=device,
            )
            # Add regularization in the scaled system to prevent near-zero pivots
            precond_reg = self.config.precond_reg
            if precond_reg > 0.0:
                wp.launch(
                    kernel=batched_apply_precond_add_reg_kernel,
                    dim=(nw, num_pivots),
                    inputs=[st.block_sizes_dev, wp.float32(precond_reg), num_pivots],
                    outputs=[st.N_diag],
                    device=device,
                )

        wp.launch(
            kernel=batched_zero_L_blocks_kernel,
            dim=(nw, nnz_L),
            inputs=[st.L_blocks, nnz_L],
            device=device,
        )
        wp.launch(
            kernel=batched_copy_N_off_to_L_kernel,
            dim=(nw, nnz_N),
            inputs=[st.N_off, st.N_off_to_L_dev, nnz_N],
            outputs=[st.L_blocks],
            device=device,
        )

        for level in range(sym.num_levels - 1, -1, -1):
            count = int(sym.level_ptr[level + 1] - sym.level_ptr[level])
            if count == 0:
                continue
            wp.launch_tiled(
                kernel=batched_ldl_factor_level_kernel,
                dim=(nw, count),
                inputs=[
                    st.L_col_ptr_dev,
                    st.L_row_idx_dev,
                    st.pred_diag_ptr_dev,
                    st.pred_diag_slot_dev,
                    st.pred_off_ptr_dev,
                    st.pred_off_slot_IK_dev,
                    st.pred_off_slot_JK_dev,
                    st.level_pivots_dev,
                    count,
                    int(sym.level_ptr[level]),
                    st.N_diag,
                    st.D_blocks,
                    st.L_blocks,
                ],
                block_dim=128,
                device=device,
            )

        wp.launch(
            kernel=batched_gather_rhs_kernel,
            dim=(nw, num_pivots),
            inputs=[
                ctx.b,
                ctx.nc_offset,
                st.pivot_order_dev,
                st.block_sizes_dev,
                num_pivots,
                st.rhs_blocks,
            ],
            device=device,
        )

        # --- Scale RHS ---
        if use_precond:
            wp.launch(
                kernel=batched_apply_precond_scale_rhs_kernel,
                dim=(nw, num_pivots),
                inputs=[st.scale_blocks, num_pivots],
                outputs=[st.rhs_blocks],
                device=device,
            )

        for level in range(sym.num_levels - 1, -1, -1):
            count = int(sym.level_ptr[level + 1] - sym.level_ptr[level])
            if count == 0:
                continue
            wp.launch_tiled(
                kernel=batched_block_forward_sub_kernel,
                dim=(nw, count),
                inputs=[
                    st.L_row_ptr_dev,
                    st.L_col_idx_dev,
                    st.L_csr_to_csc_dev,
                    st.level_pivots_dev,
                    count,
                    int(sym.level_ptr[level]),
                    st.L_blocks,
                    st.D_blocks,
                    st.rhs_blocks,
                    st.y_blocks,
                ],
                block_dim=128,
                device=device,
            )

        for level in range(0, sym.num_levels):
            count = int(sym.level_ptr[level + 1] - sym.level_ptr[level])
            if count == 0:
                continue
            wp.launch_tiled(
                kernel=batched_block_backward_sub_kernel,
                dim=(nw, count),
                inputs=[
                    st.L_col_ptr_dev,
                    st.L_row_idx_dev,
                    st.level_pivots_dev,
                    count,
                    int(sym.level_ptr[level]),
                    st.L_blocks,
                    st.D_blocks,
                    st.y_blocks,
                    st.x_blocks,
                ],
                block_dim=128,
                device=device,
            )

        # --- Unscale solution ---
        if use_precond:
            wp.launch(
                kernel=batched_apply_precond_unscale_x_kernel,
                dim=(nw, num_pivots),
                inputs=[st.scale_blocks, num_pivots],
                outputs=[st.x_blocks],
                device=device,
            )

        wp.launch(
            kernel=batched_scatter_x_kernel,
            dim=(nw, num_pivots),
            inputs=[
                st.x_blocks,
                st.pivot_order_dev,
                ctx.nc_offset,
                st.block_sizes_dev,
                num_pivots,
                ctx.x,
            ],
            device=device,
        )

        return 1


__all__ = [
    "SparseLDLSolver",
]
