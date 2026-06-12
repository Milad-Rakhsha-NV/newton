# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Contact position correction solver.

Fixes contact penetration at the position level after velocity integration.
Unlike Baumgarte stabilization (which acts at the velocity level and has
limited authority per step), this directly corrects body positions to
resolve penetration.

The position correction step solves (for active penetrating contacts):
    (J_n * M_inv * J_n^T) * nu = -phi
and applies:
    delta_q = M_inv * J_n^T * nu

where:
    phi = penetration depth (negative for penetration)
    J_n = contact normal Jacobian only (no friction tangents)

Only the normal direction is corrected — tangential sliding is a
velocity-level concern handled by the friction cone projection.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from ...sim import Contacts, Model, State
from . import contact_kernels
from .kernels import compute_body_inv_inertia_world
from .numerical_solver import (
    NumericalSolver,
    NumericalSolverConfig,
    SolveContext,
    SparseJacobiSolver,
)


# =============================================================================
# Kernels
# =============================================================================


@wp.kernel
def compute_contact_position_jacobians(
    body_q: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3),
    shape_body: wp.array(dtype=int),
    contact_count: wp.array(dtype=int),
    contact_point0: wp.array(dtype=wp.vec3),
    contact_point1: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    contact_shape0: wp.array(dtype=int),
    contact_shape1: wp.array(dtype=int),
    contact_thickness0: wp.array(dtype=float),
    contact_thickness1: wp.array(dtype=float),
    contact_max: int,
    # outputs
    jacobian: wp.array(dtype=float, ndim=2),
    body_a: wp.array(dtype=int),
    body_b: wp.array(dtype=int),
    violation: wp.array(dtype=float),
    active_mask: wp.array(dtype=int),
):
    """Compute contact normal Jacobians at current (post-integration) positions.

    Only the normal direction is needed for position correction.
    Each contact produces exactly 1 constraint row (normal only).

    active_mask[tid] = 1 if the contact is penetrating (phi < 0), else 0.
    """
    tid = wp.tid()

    count = wp.min(contact_max, contact_count[0])
    if tid >= count:
        # Zero out inactive slots
        body_a[tid] = -1
        body_b[tid] = -1
        violation[tid] = 0.0
        active_mask[tid] = 0
        return

    # Get body indices from shapes
    shape_a = contact_shape0[tid]
    shape_b = contact_shape1[tid]

    ba = -1
    bb = -1
    if shape_a >= 0:
        ba = shape_body[shape_a]
    if shape_b >= 0:
        bb = shape_body[shape_b]

    body_a[tid] = ba
    body_b[tid] = bb

    # Contact normal
    n = contact_normal[tid]

    # Get contact points in world frame at CURRENT positions
    bx_a = contact_point0[tid]
    bx_b = contact_point1[tid]
    thickness_a = contact_thickness0[tid]
    thickness_b = contact_thickness1[tid]

    # Compute lever arms from body COMs to contact point
    r_a = wp.vec3(0.0)
    r_b = wp.vec3(0.0)

    if ba >= 0:
        X_wb_a = body_q[ba]
        com_a = body_com[ba]
        bx_a = wp.transform_point(X_wb_a, bx_a) + thickness_a * n
        r_a = bx_a - wp.transform_point(X_wb_a, com_a)

    if bb >= 0:
        X_wb_b = body_q[bb]
        com_b = body_com[bb]
        bx_b = wp.transform_point(X_wb_b, bx_b) - thickness_b * n
        r_b = bx_b - wp.transform_point(X_wb_b, com_b)

    # Penetration depth
    d = wp.dot(n, bx_b - bx_a)

    violation[tid] = d

    # Only correct penetrating contacts (d < 0)
    if d >= 0.0:
        active_mask[tid] = 0
        # Zero Jacobian row so solver ignores this contact
        for c in range(12):
            jacobian[tid, c] = 0.0
        return

    active_mask[tid] = 1

    # Normal Jacobian: same as velocity-level but one row per contact
    # Body A: J_a = [-n, n x r_a]  (note: cross order for angular)
    # Body B: J_b = [n, r_b x n]
    n_cross_r_a = wp.cross(n, r_a)
    r_b_cross_n = wp.cross(r_b, n)

    jacobian[tid, 0] = -n[0]
    jacobian[tid, 1] = -n[1]
    jacobian[tid, 2] = -n[2]
    jacobian[tid, 3] = n_cross_r_a[0]
    jacobian[tid, 4] = n_cross_r_a[1]
    jacobian[tid, 5] = n_cross_r_a[2]
    jacobian[tid, 6] = n[0]
    jacobian[tid, 7] = n[1]
    jacobian[tid, 8] = n[2]
    jacobian[tid, 9] = r_b_cross_n[0]
    jacobian[tid, 10] = r_b_cross_n[1]
    jacobian[tid, 11] = r_b_cross_n[2]


@wp.kernel
def compute_contact_position_residual(
    violation: wp.array(dtype=float),
    active_mask: wp.array(dtype=int),
    contact_count: wp.array(dtype=int),
    contact_max: int,
    # outputs
    residual: wp.array(dtype=float),
):
    """Compute position correction RHS: b = -phi for penetrating contacts.

    For position correction, we want to fully resolve the penetration:
        (J M^-1 J^T) nu = -phi
    So b = -phi (which is positive when phi < 0, pushing bodies apart).
    """
    tid = wp.tid()

    count = wp.min(contact_max, contact_count[0])
    if tid >= count or active_mask[tid] == 0:
        residual[tid] = 0.0
        return

    residual[tid] = -violation[tid]


@wp.kernel
def compute_contact_position_diagonal(
    body_inv_mass: wp.array(dtype=float),
    body_inv_inertia_world: wp.array(dtype=wp.mat33),
    body_a: wp.array(dtype=int),
    body_b: wp.array(dtype=int),
    jacobian: wp.array(dtype=float, ndim=2),
    active_mask: wp.array(dtype=int),
    reg: float,
    contact_count: wp.array(dtype=int),
    contact_max: int,
    # outputs
    diag: wp.array(dtype=float),
):
    """Compute diagonal preconditioner for contact position correction."""
    tid = wp.tid()

    count = wp.min(contact_max, contact_count[0])
    if tid >= count or active_mask[tid] == 0:
        diag[tid] = 1e10  # Effectively infinite → zero correction
        return

    ba = body_a[tid]
    bb = body_b[tid]

    J_a_lin = wp.vec3(jacobian[tid, 0], jacobian[tid, 1], jacobian[tid, 2])
    J_a_ang = wp.vec3(jacobian[tid, 3], jacobian[tid, 4], jacobian[tid, 5])
    J_b_lin = wp.vec3(jacobian[tid, 6], jacobian[tid, 7], jacobian[tid, 8])
    J_b_ang = wp.vec3(jacobian[tid, 9], jacobian[tid, 10], jacobian[tid, 11])

    d = reg

    if ba >= 0:
        inv_m_a = body_inv_mass[ba]
        inv_I_a = body_inv_inertia_world[ba]
        d = d + wp.dot(J_a_lin, J_a_lin) * inv_m_a
        d = d + wp.dot(J_a_ang, inv_I_a * J_a_ang)

    if bb >= 0:
        inv_m_b = body_inv_mass[bb]
        inv_I_b = body_inv_inertia_world[bb]
        d = d + wp.dot(J_b_lin, J_b_lin) * inv_m_b
        d = d + wp.dot(J_b_ang, inv_I_b * J_b_ang)

    if d < 1e-3:
        d = 1e10

    diag[tid] = d


@wp.kernel
def project_contact_position_lambda(
    contact_count: wp.array(dtype=int),
    contact_max: int,
    # in/out
    lambda_: wp.array(dtype=float),
):
    """Project contact position lambda to be non-negative (unilateral constraint).

    For position correction, we only have normal forces and they must be >= 0
    (contacts can push apart but not pull together).
    """
    tid = wp.tid()

    count = wp.min(contact_max, contact_count[0])
    if tid >= count:
        return

    if lambda_[tid] < 0.0:
        lambda_[tid] = 0.0


# =============================================================================
# Contact Position Solver
# =============================================================================


class ContactPositionSolver:
    """Position-level contact penetration correction.

    Solves a normal-only contact system at the position level to directly
    fix penetration that Baumgarte stabilization can't fully resolve in
    a single velocity step.

    This uses the same iterative solver infrastructure as the velocity-level
    contact solver, but:
    - Only normal direction (1 row per contact, not 3)
    - RHS is -phi (penetration depth), not velocity residual
    - Lambda is projected to >= 0 (unilateral, no friction cone)
    - Applied as position correction (delta_q), not force
    """

    def __init__(
        self,
        model: Model,
        solver: NumericalSolver | None = None,
    ):
        self.model = model

        if solver is None:
            config = NumericalSolverConfig(
                max_iterations=20,
                tolerance=1e-6,
                omega=0.5,
                relax=0.9,
                reg=1e-4,
            )
            solver = SparseJacobiSolver(config)

        self.numerical_solver = solver

        # Allocate buffers sized to contact_max
        contact_max = model.rigid_contact_max if model.rigid_contact_max > 0 else 1000
        self._contact_max = contact_max
        device = model.device
        nb = max(model.body_count, 1)

        with wp.ScopedDevice(device):
            # One row per contact (normal only)
            self._jacobian = wp.zeros((contact_max, 12), dtype=wp.float32)
            self._body_a = wp.zeros(contact_max, dtype=wp.int32)
            self._body_b = wp.zeros(contact_max, dtype=wp.int32)
            self._violation = wp.zeros(contact_max, dtype=wp.float32)
            self._active_mask = wp.zeros(contact_max, dtype=wp.int32)
            self._residual = wp.zeros(contact_max, dtype=wp.float32)
            self._diag = wp.zeros(contact_max, dtype=wp.float32)
            self._lambda = wp.zeros(contact_max, dtype=wp.float32)
            self._delta_v = wp.zeros(nb, dtype=wp.spatial_vector)
            self._position_correction = wp.zeros(nb, dtype=wp.spatial_vector)
            self._body_inv_inertia_world = wp.zeros(nb, dtype=wp.mat33)

            # block_nc = 1 for each contact (1 normal row per contact)
            self._block_nc = wp.ones(contact_max, dtype=wp.int32)
            # nc_offset = identity (row i = contact i)
            nc_offset = np.arange(contact_max, dtype=np.int32)
            self._nc_offset = wp.array(nc_offset, dtype=wp.int32)
            # row_to_block = identity (row i belongs to block i)
            self._row_to_block = wp.array(nc_offset, dtype=wp.int32)
            # active_row_count (set dynamically)
            self._active_row_count = wp.zeros(1, dtype=wp.int32)

    def solve(
        self,
        state: State,
        contacts: Contacts,
        body_inv_inertia_world: wp.array | None = None,
    ) -> None:
        """Solve contact position correction and apply to state.

        Args:
            state: Simulation state (body_q will be modified in place).
            contacts: Contact data from collision detection.
            body_inv_inertia_world: Pre-computed inverse inertia. If None,
                recomputes from raw body inertia.
        """
        if contacts is None or contacts.rigid_contact_max == 0:
            return

        model = self.model
        device = model.device
        contact_max = self._contact_max

        # Compute inverse inertia if not provided
        if body_inv_inertia_world is not None:
            inv_inertia = body_inv_inertia_world
        else:
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

        # 1. Compute contact Jacobians at current (post-integration) positions
        wp.launch(
            kernel=compute_contact_position_jacobians,
            dim=contact_max,
            inputs=[
                state.body_q,
                model.body_com,
                model.shape_body,
                contacts.rigid_contact_count,
                contacts.rigid_contact_point0,
                contacts.rigid_contact_point1,
                contacts.rigid_contact_normal,
                contacts.rigid_contact_shape0,
                contacts.rigid_contact_shape1,
                contacts.rigid_contact_margin0,
                contacts.rigid_contact_margin1,
                contact_max,
            ],
            outputs=[
                self._jacobian,
                self._body_a,
                self._body_b,
                self._violation,
                self._active_mask,
            ],
            device=device,
        )

        # 2. Compute RHS = -phi
        wp.launch(
            kernel=compute_contact_position_residual,
            dim=contact_max,
            inputs=[
                self._violation,
                self._active_mask,
                contacts.rigid_contact_count,
                contact_max,
            ],
            outputs=[self._residual],
            device=device,
        )

        # 3. Compute diagonal preconditioner
        wp.launch(
            kernel=compute_contact_position_diagonal,
            dim=contact_max,
            inputs=[
                model.body_inv_mass,
                inv_inertia,
                self._body_a,
                self._body_b,
                self._jacobian,
                self._active_mask,
                self.numerical_solver.config.reg,
                contacts.rigid_contact_count,
                contact_max,
            ],
            outputs=[self._diag],
            device=device,
        )

        # Set active row count = contact_count (1 row per contact)
        # We use a kernel to keep it GPU-resident
        wp.launch(
            kernel=_copy_int_kernel,
            dim=1,
            inputs=[contacts.rigid_contact_count],
            outputs=[self._active_row_count],
            device=device,
        )

        # 4. Zero lambda and position correction
        self._lambda.zero_()
        self._delta_v.zero_()
        self._position_correction.zero_()

        # 5. Solve using SparseJacobi with unilateral projection
        # We manually iterate instead of calling solver.solve() so we can
        # inject the unilateral projection (lambda >= 0) after each iteration
        from .. dvi import constraint_kernels, kernels

        nc = contact_max  # over-allocated, kernel uses active_row_count

        # Ensure solver work buffers
        self.numerical_solver._ensure_buffers(
            SolveContext(
                b=self._residual, nc=nc, nb=model.body_count,
                device=device, x=self._lambda, delta_v=self._delta_v,
            ),
            nc,
        )

        rows_per_block = 1  # 1 normal row per contact block

        for _ in range(self.numerical_solver.config.max_iterations):
            self._delta_v.zero_()

            # delta_v = M_inv @ J.T @ lambda
            wp.launch(
                kernel=constraint_kernels.compute_delta_v,
                dim=nc,
                inputs=[
                    model.body_inv_mass,
                    inv_inertia,
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

            # Jacobi iteration
            wp.launch(
                kernel=constraint_kernels.jacobi_iteration,
                dim=nc,
                inputs=[
                    self._jacobian,
                    self._residual,
                    self._body_a,
                    self._body_b,
                    self._row_to_block,
                    self._diag,
                    self._delta_v,
                    self.numerical_solver.config.omega,
                    self.numerical_solver.config.relax,
                    self._active_row_count,
                    rows_per_block,
                    self._lambda,
                ],
                outputs=[self.numerical_solver._lambda_new],
                device=device,
            )

            # Swap
            wp.launch(
                kernel=kernels.swap_float_arrays,
                dim=max(nc, 1),
                inputs=[self._lambda, nc],
                outputs=[self.numerical_solver._lambda_new],
                device=device,
            )

            # Project: lambda >= 0 (unilateral contact)
            wp.launch(
                kernel=project_contact_position_lambda,
                dim=contact_max,
                inputs=[
                    contacts.rigid_contact_count,
                    contact_max,
                ],
                outputs=[self._lambda],
                device=device,
            )

        # 6. Compute final position correction: delta_q = M_inv @ J.T @ lambda
        self._position_correction.zero_()
        wp.launch(
            kernel=constraint_kernels.compute_delta_v,
            dim=nc,
            inputs=[
                model.body_inv_mass,
                inv_inertia,
                self._jacobian,
                self._body_a,
                self._body_b,
                self._row_to_block,
                self._lambda,
                self._active_row_count,
                rows_per_block,
            ],
            outputs=[self._position_correction],
            device=device,
        )

        # 7. Apply position correction to body_q
        wp.launch(
            kernel=constraint_kernels.apply_position_correction,
            dim=model.body_count,
            inputs=[
                self._position_correction,
                model.body_inv_mass,
            ],
            outputs=[state.body_q],
            device=device,
        )


@wp.kernel
def _copy_int_kernel(
    src: wp.array(dtype=int),
    # outputs
    dst: wp.array(dtype=int),
):
    """Copy a single int from one GPU array to another."""
    dst[0] = src[0]
