# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Joint limit constraint solver.

Solves joint limits as unilateral constraints (inequality constraints)
using iterative solvers with λ >= 0 projection. This replaces the
penalty-based spring-damper approach with proper constraint impulses.

The solver runs before the bilateral joint constraint solve in the
DVI step pipeline, so that the bilateral LDL solver can account
for the limit impulses when enforcing joint constraints.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from ...sim import Model, State
from . import constraint_kernels, kernels
from .numerical_solver.base import (
    NumericalSolver,
    NumericalSolverConfig,
    SolverType,
)
from .numerical_solver.sparse_jacobi import SparseJacobiSolver


class JointLimitConstraintInfo:
    """Pre-computed joint limit constraint topology.

    Computes the maximum number of limit constraint rows and their
    offsets. Unlike contacts, the topology is fixed (one potential
    constraint per limited DOF), but the active set changes each step.

    Attributes:
        max_nc: Maximum constraint rows (one per limited DOF).
        limit_nc_per_joint: Max limit constraints per joint.
        limit_nc_offset: Prefix-sum offsets into Jacobian rows.
    """

    def __init__(
        self,
        max_nc: int,
        limit_nc_per_joint: wp.array,
        limit_nc_offset: wp.array,
    ):
        self.max_nc = max_nc
        self.limit_nc_per_joint = limit_nc_per_joint
        self.limit_nc_offset = limit_nc_offset

    @staticmethod
    def from_model(model: Model) -> JointLimitConstraintInfo:
        """Compute limit constraint topology from model."""
        device = model.device
        num_joints = model.joint_count if model.joint_count > 0 else 1

        if model.joint_count > 0 and model.joint_limit_ke is not None:
            joint_type = model.joint_type.numpy()
            joint_enabled = model.joint_enabled.numpy()
            joint_dof_dim = model.joint_dof_dim.numpy()
            limit_lower = model.joint_limit_lower.numpy()
            limit_upper = model.joint_limit_upper.numpy()
            limit_ke = model.joint_limit_ke.numpy()
            qd_start = model.joint_qd_start.numpy()

            nc_per_joint = np.zeros(num_joints, dtype=np.int32)
            for j in range(num_joints):
                if not joint_enabled[j]:
                    continue
                jtype = joint_type[j]
                qs = qd_start[j]
                n_lin, n_ang = joint_dof_dim[j]

                if jtype == 3 or jtype == 4 or jtype == 5:  # FIXED, FREE, COMPOUND
                    continue

                if jtype == 1:  # REVOLUTE: 1 DOF
                    if limit_ke[qs] > 0.0 and limit_lower[qs] < limit_upper[qs]:
                        nc_per_joint[j] = 1
                elif jtype == 0:  # PRISMATIC: 1 DOF
                    if limit_ke[qs] > 0.0 and limit_lower[qs] < limit_upper[qs]:
                        nc_per_joint[j] = 1
                elif jtype == 2:  # BALL: 3 angular DOFs
                    for dof in range(3):
                        idx = qs + dof
                        if limit_ke[idx] > 0.0 and limit_lower[idx] < limit_upper[idx]:
                            nc_per_joint[j] += 1
                elif jtype == 6:  # D6: variable DOFs
                    for dof in range(n_ang):
                        idx = qs + n_lin + dof
                        if limit_ke[idx] > 0.0 and limit_lower[idx] < limit_upper[idx]:
                            nc_per_joint[j] += 1

            nc_offset = np.zeros(num_joints, dtype=np.int32)
            offset = 0
            for j in range(num_joints):
                nc_offset[j] = offset
                offset += int(nc_per_joint[j])
            max_nc = int(offset)

            with wp.ScopedDevice(device):
                limit_nc_per_joint = wp.array(nc_per_joint, dtype=wp.int32, device=device)
                limit_nc_offset = wp.array(nc_offset, dtype=wp.int32, device=device)
        else:
            max_nc = 0
            with wp.ScopedDevice(device):
                limit_nc_per_joint = wp.zeros(num_joints, dtype=wp.int32)
                limit_nc_offset = wp.zeros(num_joints, dtype=wp.int32)

        return JointLimitConstraintInfo(
            max_nc=max_nc,
            limit_nc_per_joint=limit_nc_per_joint,
            limit_nc_offset=limit_nc_offset,
        )


class JointLimitSolver:
    """Solver for unilateral joint limit constraints.

    Formulates joint limits as inequality constraints and solves for
    Lagrange multipliers with λ >= 0 projection after each iteration.
    """

    def __init__(
        self,
        model: Model,
        solver: NumericalSolver | None = None,
        info: JointLimitConstraintInfo | None = None,
        enable_timers: bool = False,
    ):
        self.model = model
        self.enable_timers = enable_timers

        if info is None:
            info = JointLimitConstraintInfo.from_model(model)
        self._info = info

        if solver is None:
            config = NumericalSolverConfig(
                solver_type=SolverType.SPARSE_JACOBI,
                max_iterations=20,
                omega=0.3,
                relax=0.9,
                alpha=0.005,
                recovery_speed=1.0,
                reg=1e-8,
            )
            solver = SparseJacobiSolver(config)

        self.numerical_solver = solver
        self._max_nc = info.max_nc

        # Allocate buffers
        device = model.device
        nc = max(info.max_nc, 1)
        nb = max(model.body_count, 1)
        with wp.ScopedDevice(device):
            self._jacobian = wp.zeros((nc, 12), dtype=wp.float32)
            self._violation = wp.zeros(nc, dtype=wp.float32)
            self._residual = wp.zeros(nc, dtype=wp.float32)
            self._diag = wp.zeros(nc, dtype=wp.float32)
            self._lambda = wp.zeros(nc, dtype=wp.float32)
            self._delta_v = wp.zeros(nb, dtype=wp.spatial_vector)
            self._body_a = wp.zeros(nc, dtype=wp.int32)
            self._body_b = wp.zeros(nc, dtype=wp.int32)
            self._active_limit_count = wp.zeros(1, dtype=wp.int32)
            # row_to_block is identity for 1-row-per-block
            self._row_to_block = wp.array(np.arange(nc, dtype=np.int32), device=device)
            self._active_row_count = wp.array([info.max_nc], dtype=wp.int32, device=device)
            self._lambda_new = wp.zeros(nc, dtype=wp.float32, device=device)

    @property
    def max_nc(self) -> int:
        return self._max_nc

    def solve(
        self,
        state: State,
        dt: float,
        body_inv_inertia_world: wp.array,
    ) -> int:
        """Solve joint limit constraints.

        Args:
            state: Current simulation state.
            dt: Time step size.
            body_inv_inertia_world: Pre-computed world-frame inverse inertia.

        Returns:
            Number of iterations performed.
        """
        model = self.model
        if self._max_nc == 0:
            return 0

        device = model.device
        limit_max_nc = self._max_nc
        num_joints = model.joint_count

        # Zero arrays for this step
        self._jacobian.zero_()
        self._violation.zero_()
        self._active_limit_count.zero_()

        with wp.ScopedTimer("DVI: Limit Jacobians", active=self.enable_timers):
            wp.launch(
                kernel=constraint_kernels.compute_joint_limit_jacobians,
                dim=num_joints,
                inputs=[
                    state.body_q,
                    state.body_qd,
                    model.joint_type,
                    model.joint_enabled,
                    model.joint_parent,
                    model.joint_child,
                    model.joint_X_p,
                    model.joint_X_c,
                    model.joint_axis,
                    model.joint_qd_start,
                    model.joint_dof_dim,
                    model.joint_limit_lower,
                    model.joint_limit_upper,
                    model.joint_limit_ke,
                    self._info.limit_nc_offset,
                    limit_max_nc,
                ],
                outputs=[
                    self._jacobian,
                    self._violation,
                    self._body_a,
                    self._body_b,
                    self._active_limit_count,
                ],
                device=device,
            )

        with wp.ScopedTimer("DVI: Limit residual", active=self.enable_timers):
            wp.launch(
                kernel=constraint_kernels.compute_joint_limit_residual,
                dim=limit_max_nc,
                inputs=[
                    state.body_qd,
                    state.body_f,
                    model.body_inv_mass,
                    body_inv_inertia_world,
                    self._jacobian,
                    self._violation,
                    self._body_a,
                    self._body_b,
                    dt,
                    self.numerical_solver.config.alpha,
                    self.numerical_solver.config.recovery_speed,
                    limit_max_nc,
                ],
                outputs=[self._residual],
                device=device,
            )

        with wp.ScopedTimer("DVI: Limit diagonal", active=self.enable_timers):
            wp.launch(
                kernel=constraint_kernels.compute_joint_limit_diagonal,
                dim=limit_max_nc,
                inputs=[
                    model.body_inv_mass,
                    body_inv_inertia_world,
                    self._jacobian,
                    self._body_a,
                    self._body_b,
                    self.numerical_solver.config.reg,
                    limit_max_nc,
                ],
                outputs=[self._diag],
                device=device,
            )

        # Initialize lambda and delta_v
        self._lambda.zero_()
        self._delta_v.zero_()

        # Ensure work buffer for Jacobi swap
        if self._lambda_new is None or self._lambda_new.shape[0] < limit_max_nc:
            self._lambda_new = wp.zeros(max(limit_max_nc, 1), dtype=wp.float32, device=device)

        with wp.ScopedTimer("DVI: Limit iterations", active=self.enable_timers):
            # Manual iteration loop with λ≥0 projection after each step.
            # We cannot use SparseJacobiSolver.solve() directly because it
            # only applies friction cone projection for contacts, not the
            # unilateral λ≥0 projection needed for joint limits.
            config = self.numerical_solver.config
            rows_per_block = 0  # use row_to_block mapping (1 row per block)

            for _ in range(config.max_iterations):
                self._delta_v.zero_()

                # delta_v = M_inv @ J.T @ lambda
                wp.launch(
                    kernel=constraint_kernels.compute_delta_v,
                    dim=limit_max_nc,
                    inputs=[
                        model.body_inv_mass,
                        body_inv_inertia_world,
                        self._jacobian,
                        self._body_a,
                        self._body_b,
                        self._row_to_block,
                        self._lambda,
                        self._active_row_count,
                        rows_per_block,
                    ],
                    outputs=[self._delta_v],
                    device=device,
                )

                # Jacobi update
                wp.launch(
                    kernel=constraint_kernels.jacobi_iteration,
                    dim=limit_max_nc,
                    inputs=[
                        self._jacobian,
                        self._residual,
                        self._body_a,
                        self._body_b,
                        self._row_to_block,
                        self._diag,
                        self._delta_v,
                        config.omega,
                        config.relax,
                        self._active_row_count,
                        rows_per_block,
                        self._lambda,
                    ],
                    outputs=[self._lambda_new],
                    device=device,
                )

                # Swap buffers
                wp.launch(
                    kernel=kernels.swap_float_arrays,
                    dim=max(limit_max_nc, 1),
                    inputs=[self._lambda, limit_max_nc],
                    outputs=[self._lambda_new],
                    device=device,
                )

                # Project: clamp λ >= 0 (unilateral constraint)
                wp.launch(
                    kernel=constraint_kernels.project_joint_limits,
                    dim=limit_max_nc,
                    inputs=[limit_max_nc],
                    outputs=[self._lambda],
                    device=device,
                )

        return config.max_iterations

    def apply_forces(self, state: State, dt: float):
        """Apply solved joint limit forces to body force buffer."""
        if self._max_nc == 0:
            return

        wp.launch(
            kernel=constraint_kernels.apply_constraint_forces,
            dim=self._max_nc,
            inputs=[
                self._jacobian,
                self._body_a,
                self._body_b,
                self._row_to_block,
                self._lambda,
                dt,
                self._max_nc,
            ],
            outputs=[state.body_f],
            device=self.model.device,
        )
