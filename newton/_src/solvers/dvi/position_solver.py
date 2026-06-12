# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Position correction solver for post-stabilization.

This module provides the PositionSolver class which handles position-level
constraint correction after velocity integration.

The position correction step solves:
    (J * M_inv * J^T) * nu = -phi
and applies:
    delta_q = M_inv * J^T * nu

where phi is the constraint violation after velocity integration.
"""

from __future__ import annotations

import warp as wp

from ...sim import Model, State
from . import constraint_kernels
from . import kernels
from .kernels import compute_body_inv_inertia_world
from .numerical_solver import (
    NumericalSolver,
    NumericalSolverConfig,
    SolveContext,
    SparseJacobiSolver,
    JointConstraint,
)


class PositionSolver:
    """Solver for position-level constraint correction.

    This class handles position correction (post-stabilization):
        1. Recompute Jacobians at current positions
        2. Compute RHS = -phi (negated constraint violation)
        3. Solve the position correction system
        4. Apply position corrections to bodies
    """

    def __init__(
        self,
        model: Model,
        solver: NumericalSolver | None = None,
        constraint: JointConstraint | None = None,
    ):
        """Initialize the position solver.

        Args:
            model: The simulation model.
            solver: Numerical solver for the position correction system.
            constraint: JointConstraint object (owned by SolverDVI).
        """
        self.model = model

        # Create default solver if not provided
        if solver is None:
            config = NumericalSolverConfig(
                max_iterations=10,
                tolerance=1e-6,
                omega=0.5,
                relax=0.9,
                alpha=1e6,  # Position solver doesn't use Baumgarte
                reg=1e-4,
            )
            solver = SparseJacobiSolver(config)

        self.numerical_solver = solver
        self._constraint = constraint

        # Allocate additional buffers needed for position solve
        self._allocate_buffers()

    def _allocate_buffers(self):
        """Allocate working buffers for position solve."""
        model = self.model
        device = model.device
        c = self._constraint

        nb = max(model.body_count, 1)
        nc = max(c.nc, 1)
        with wp.ScopedDevice(device):
            self.neg_violation = wp.zeros(nc, dtype=wp.float32)
            self.position_lambda = wp.zeros(nc, dtype=wp.float32)
            self.position_correction = wp.zeros(nb, dtype=wp.spatial_vector)
            self._body_inv_inertia_world = wp.zeros(nb, dtype=wp.mat33)

    def finalize_for_capture(self, state: State):
        """Initialize solver for CUDA graph capture.

        Call this method once before graph capture to pre-allocate buffers
        and compute topology-dependent data.

        Args:
            state: Initial simulation state.
        """
        self.numerical_solver.prepare_for_capture(self.model, state)

    def solve(self, state: State, body_inv_inertia_world: wp.array | None = None) -> None:
        """Solve position correction and apply to state.

        Args:
            state: Simulation state (body_q will be modified).
            body_inv_inertia_world: Pre-computed world-frame inverse inertia.
                When implicit PD is enabled, this should be the **augmented**
                inverse inertia from the velocity-level solve so that position
                corrections are consistent with the effective mass used during
                integration.  If None, recomputes from raw body inertia.
        """
        model = self.model

        if model.joint_count == 0:
            return

        if self._constraint.nc == 0:
            return

        device = model.device
        num_joints = model.joint_count
        c = self._constraint

        if body_inv_inertia_world is not None:
            # Use the augmented inverse inertia from the main solver.
            # This ensures position corrections use the same effective mass
            # as the velocity-level solve (critical for implicit PD consistency).
            inv_inertia = body_inv_inertia_world
        else:
            # Fallback: recompute from raw body inertia (no implicit PD).
            wp.launch(
                kernel=compute_body_inv_inertia_world,
                dim=model.body_count,
                inputs=[
                    state.body_q,
                    model.body_inv_inertia,
                    model.body_inv_mass,
                ],
                outputs=[self._body_inv_inertia_world],
                device=device,
            )
            inv_inertia = self._body_inv_inertia_world

        self.numerical_solver.prepare_for_capture(model, state, body_inv_inertia_world=inv_inertia)

        # Recompute Jacobians and violations at current positions
        wp.launch(
            kernel=constraint_kernels.compute_joint_jacobians_and_violation,
            dim=num_joints,
            inputs=[
                state.body_q,
                state.body_qd,
                model.body_com,
                model.joint_type,
                model.joint_enabled,
                model.joint_parent,
                model.joint_child,
                model.joint_X_p,
                model.joint_X_c,
                model.joint_axis,
                model.joint_q_start,
                model.joint_qd_start,
                model.joint_dof_dim,
                c.nc_offset,
            ],
            outputs=[
                c.jacobian,
                c.violation,
                c.body_a,
                c.body_b,
            ],
            device=device,
        )

        # Negate violation: rhs = -phi
        wp.launch(
            kernel=kernels.negate_float_array,
            dim=max(c.nc, 1),
            inputs=[c.violation, c.nc],
            outputs=[self.neg_violation],
            device=device,
        )

        # Compute diagonal preconditioner
        wp.launch(
            kernel=constraint_kernels.compute_joint_diagonal,
            dim=num_joints,
            inputs=[
                model.body_inv_mass,
                inv_inertia,
                c.jacobian,
                c.body_a,
                c.body_b,
                c.block_nc,
                c.nc_offset,
                self.numerical_solver.config.reg,
                num_joints,
            ],
            outputs=[c.diag],
            device=device,
        )

        # Zero lambda and position correction
        self.position_lambda.zero_()
        self.position_correction.zero_()

        # Build solve context
        ctx = SolveContext(
            b=self.neg_violation,
            nc=c.nc,
            nb=c.nb,
            device=device,
            J=c.jacobian,
            body_a=c.body_a,
            body_b=c.body_b,
            joint_nc=c.block_nc,
            nc_offset=c.nc_offset,
            row_to_block=c.row_to_block,
            active_row_count=c.active_row_count,
            M_inv_diag=model.body_inv_mass,
            M_inv_inertia=inv_inertia,
            diag=c.diag,
            nj=c.n_blocks,
            omega=self.numerical_solver.config.omega,
            relax=self.numerical_solver.config.relax,
            reg=self.numerical_solver.config.reg,
            x=self.position_lambda,
            delta_v=self.position_correction,
        )

        self.numerical_solver.solve(ctx)

        # Apply position correction
        wp.launch(
            kernel=constraint_kernels.apply_position_correction,
            dim=model.body_count,
            inputs=[
                self.position_correction,
                model.body_inv_mass,
            ],
            outputs=[state.body_q],
            device=device,
        )
