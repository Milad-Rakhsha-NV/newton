# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Sparse Jacobi solver using Warp kernels (matrix-free)."""

from __future__ import annotations

import warp as wp

from .base import Device, FrictionProjection, NumericalSolver, NumericalSolverConfig, SolveContext


class SparseJacobiSolver(NumericalSolver):
    """Matrix-free Jacobi solver using Warp kernels (CPU/GPU).

    This solver computes J @ delta_v directly without building the full N matrix.
    The iterations are performed using Warp kernels that are parallelizable on GPU.

    Uses the unified [nc, 12] Jacobian format for all constraint types.
    For contacts, friction cone projection is applied after each Jacobi iteration.
    """

    supported_devices = {Device.CPU, Device.CUDA}
    default_params = {
        "max_iterations": 100,
        "tolerance": 1e-6,
        "omega": 0.3,
        "relax": 0.5,
        "reg": 1e-8,
    }

    def __init__(self, config: NumericalSolverConfig):
        super().__init__(config)
        self._lambda_new = None
        self._allocated_size = 0
        self._contact_active_row_count = None  # GPU buffer for dynamic contact row count
        self._contact_active_count = None  # GPU buffer for contact count (for block-3x3 kernel)

    def _ensure_buffers(self, ctx: SolveContext, nc: int):
        """Ensure work buffers are allocated."""
        if self._lambda_new is None or self._allocated_size < nc:
            self._lambda_new = wp.zeros(max(nc, 1), dtype=wp.float32, device=ctx.device)
            self._allocated_size = nc

    def solve(self, ctx: SolveContext) -> int:
        """Matrix-free Jacobi solve using unified kernel.

        Args:
            ctx: Solve context with inputs and output buffers.

        Returns:
            Number of iterations performed.
        """
        from .. import constraint_kernels, contact_kernels, kernels

        if ctx.nj == 0 and ctx.contact_max == 0:
            return 0

        nc = ctx.nc
        is_contact = ctx.contact_max > 0 and ctx.contact_friction is not None
        device = ctx.device

        # For contacts: rows_per_block=3 (fixed), for joints: rows_per_block=0 (use row_to_block)
        rows_per_block = 3 if is_contact else 0

        # Get active row count array
        if is_contact:
            # Contacts: compute active_rows = contact_count * 3 on GPU (dynamic per frame)
            if self._contact_active_row_count is None or self._contact_active_row_count.device != device:
                self._contact_active_row_count = wp.zeros(1, dtype=wp.int32, device=device)
            wp.launch(
                kernel=kernels.compute_contact_row_count,
                dim=1,
                inputs=[ctx.contact_count_arr, 3],
                outputs=[self._contact_active_row_count],
                device=device,
            )
            active_row_count = self._contact_active_row_count
        else:
            # Joints: use precomputed value from constraint (fixed, computed once at init)
            active_row_count = ctx.active_row_count

        self._ensure_buffers(ctx, nc)

        # Initialize lambda to zero
        ctx.x.zero_()
        self._lambda_new.zero_()

        for _ in range(self.config.max_iterations):
            # Zero delta_v
            ctx.delta_v.zero_()

            # Compute delta_v = M_inv @ J.T @ lambda (one thread per constraint row)
            # Launch with max possible rows (nc), kernel skips inactive rows using GPU-side count
            wp.launch(
                kernel=constraint_kernels.compute_delta_v,
                dim=nc,
                inputs=[
                    ctx.M_inv_diag,
                    ctx.M_inv_inertia,
                    ctx.J,
                    ctx.body_a,
                    ctx.body_b,
                    ctx.row_to_block,
                    ctx.x,
                    active_row_count,
                    rows_per_block,
                ],
                outputs=[ctx.delta_v],
                device=device,
            )

            # Jacobi update
            if is_contact and ctx.block_precondition and ctx.diag_block_inv is not None:
                # Block-3x3 preconditioned Jacobi: one thread per contact
                wp.launch(
                    kernel=constraint_kernels.jacobi_iteration_block3x3,
                    dim=ctx.contact_max,
                    inputs=[
                        ctx.J,
                        ctx.b,
                        ctx.body_a,
                        ctx.body_b,
                        ctx.diag_block_inv,
                        ctx.delta_v,
                        ctx.omega,
                        ctx.relax,
                        ctx.contact_count_arr,
                        ctx.x,
                    ],
                    outputs=[self._lambda_new],
                    device=device,
                )
            else:
                # Scalar Jacobi: one thread per constraint row
                wp.launch(
                    kernel=constraint_kernels.jacobi_iteration,
                    dim=nc,
                    inputs=[
                        ctx.J,
                        ctx.b,
                        ctx.body_a,
                        ctx.body_b,
                        ctx.row_to_block,
                        ctx.diag,
                        ctx.delta_v,
                        ctx.omega,
                        ctx.relax,
                        active_row_count,
                        rows_per_block,
                        ctx.x,
                    ],
                    outputs=[self._lambda_new],
                    device=device,
                )

            # Swap buffers (for all allocated rows - kernel handles inactive)
            wp.launch(
                kernel=kernels.swap_float_arrays,
                dim=max(nc, 1),
                inputs=[ctx.x, nc],
                outputs=[self._lambda_new],
                device=device,
            )

            # Apply friction cone projection for contacts
            if is_contact:
                proj_flag = int(ctx.friction_projection == FrictionProjection.TANGENTIAL)
                wp.launch(
                    kernel=contact_kernels.project_friction_cones,
                    dim=ctx.contact_max,
                    inputs=[
                        ctx.contact_count_arr,
                        ctx.contact_friction,
                        ctx.contact_max,
                        proj_flag,
                    ],
                    outputs=[ctx.x],
                    device=device,
                )

        return self.config.max_iterations
