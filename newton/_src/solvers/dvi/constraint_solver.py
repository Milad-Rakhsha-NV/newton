# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
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

"""
Joint constraint solver for the DVI solver.

This module provides the ConstraintSolver class which handles bilateral joint
constraints using iterative methods (Jacobi or block Gauss-Seidel) or direct solves.

The constraint system is formulated as:

    N * lambda = b

where:
    N = J * M_inv * J^T  (constraint-space system matrix)
    b = -(J * v_pred + phi / (dt + alpha))  (alpha-damped Baumgarte)
    lambda = constraint impulses
    phi = constraint violation
    v_pred = v + dt * M_inv * f_ext  (predicted velocity)
    alpha = Baumgarte damping parameter (DVI-style)
"""

from __future__ import annotations

import warp as wp

from ...sim import Model, State
from . import constraint_kernels
from .numerical_solver import (
    NumericalSolver,
    NumericalSolverConfig,
    SolveContext,
    SparseJacobiSolver,
    JointConstraint,
)


class ConstraintSolver:
    """Solver for bilateral joint constraints.

    This class handles the solving of joint constraints using various methods:
    - SparseJacobiSolver: Kernel-based Jacobi iterations (GPU-friendly)
    - SparseLDLSolver: Block-sparse tile LDL direct solver (CPU/GPU)
    """

    def __init__(
        self,
        model: Model,
        solver: NumericalSolver | None = None,
        constraint: JointConstraint | None = None,
        enable_timers: bool = False,
    ):
        """Initialize the constraint solver.

        Args:
            model: The simulation model.
            solver: Numerical solver for the constraint system.
            constraint: JointConstraint object (owned by SolverDVI).
            enable_timers: Whether to enable performance timers.
        """
        self.model = model
        self.enable_timers = enable_timers

        # Create default solver if not provided
        if solver is None:
            config = NumericalSolverConfig(
                max_iterations=50,
                tolerance=1e-6,
                omega=0.3,
                relax=0.9,
                alpha=0.2,  # DVI-style damping
                reg=1e-8,
            )
            solver = SparseJacobiSolver(config)

        self.numerical_solver = solver
        self._constraint = constraint

    def finalize_for_capture(self, state: "State"):
        """Initialize solver for CUDA graph capture.

        Call this method once before graph capture to pre-allocate buffers
        and compute topology-dependent data.

        Args:
            state: Initial simulation state.
        """
        self.numerical_solver.prepare_for_capture(self.model, state)

    def solve(self, state: State, dt: float, body_inv_inertia_world: wp.array | None = None) -> int:
        """Solve joint constraints.

        Args:
            state: Current simulation state.
            dt: Time step size.
            body_inv_inertia_world: Pre-computed world-frame inverse inertia.

        Returns:
            Number of iterations performed.
        """
        model = self.model
        if model.joint_count == 0:
            return 0

        if self._constraint.nc == 0:
            return 0

        self.numerical_solver.prepare_for_capture(model, state, body_inv_inertia_world=body_inv_inertia_world)

        device = model.device
        num_joints = model.joint_count
        c = self._constraint

        with wp.ScopedTimer("DVI: Joint Jacobians", active=self.enable_timers):
            # Compute joint Jacobians and violations
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

        with wp.ScopedTimer("DVI: Joint residual", active=self.enable_timers):
            # Compute joint residual
            wp.launch(
                kernel=constraint_kernels.compute_joint_residual,
                dim=num_joints,
                inputs=[
                    state.body_qd,
                    state.body_f,
                    model.body_inv_mass,
                    body_inv_inertia_world,
                    c.jacobian,
                    c.violation,
                    c.body_a,
                    c.body_b,
                    c.block_nc,
                    c.nc_offset,
                    dt,
                    self.numerical_solver.config.alpha,
                    self.numerical_solver.config.recovery_speed,
                    num_joints,
                ],
                outputs=[c.residual],
                device=device,
            )

        with wp.ScopedTimer("DVI: Joint diagonal", active=self.enable_timers):
            # Compute diagonal
            wp.launch(
                kernel=constraint_kernels.compute_joint_diagonal,
                dim=num_joints,
                inputs=[
                    model.body_inv_mass,
                    body_inv_inertia_world,
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

        # Initialize lambda and delta_v.
        # When warm_start is enabled, reuse the previous frame's lambda as the
        # initial guess.  Joint constraint topology is fixed so the buffer is
        # the same size every frame — no remapping needed.
        if not self.numerical_solver.config.warm_start:
            c.lambda_.zero_()
        c.delta_v.zero_()

        with wp.ScopedTimer("DVI: Joint iterations", active=self.enable_timers):
            # Build solve context
            ctx = SolveContext(
                b=c.residual,
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
                M_inv_inertia=body_inv_inertia_world,
                diag=c.diag,
                nj=c.n_blocks,
                omega=self.numerical_solver.config.omega,
                relax=self.numerical_solver.config.relax,
                reg=self.numerical_solver.config.reg,
                x=c.lambda_,
                delta_v=c.delta_v,
            )

            return self.numerical_solver.solve(ctx)

    def apply_forces(self, state: State, dt: float):
        """Apply solved constraint forces to body force buffer.

        Args:
            state: Simulation state to update.
            dt: Time step size.
        """
        model = self.model
        if model.joint_count == 0:
            return

        c = self._constraint
        if c.nc == 0:
            return

        wp.launch(
            kernel=constraint_kernels.apply_constraint_forces,
            dim=c.nc,
            inputs=[
                c.jacobian,
                c.body_a,
                c.body_b,
                c.row_to_block,
                c.lambda_,
                dt,
                c.nc,
            ],
            outputs=[state.body_f],
            device=model.device,
        )
