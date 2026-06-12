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
Contact dynamics solver for the DVI solver.

This module provides the ContactSolver class which handles frictional contact
dynamics using cone complementarity formulations.

Available solver backends:

    - SparseJacobiSolver: Matrix-free Jacobi iterations (default)
    - SparseLDLSolver: Block-sparse tile LDL direct solver
"""

from __future__ import annotations

import warp as wp

from ...sim import Contacts, Model, State
from . import contact_kernels, kernels
from .numerical_solver import (
    ContactConstraint,
    NumericalSolver,
    NumericalSolverConfig,
    SolveContext,
    SparseJacobiSolver,
)


class ContactSolver:
    """Solver for frictional contact dynamics.

    This class handles the iterative solving of contact impulses using
    either Jacobi iteration or LDL direct solve.

    The contact problem is formulated as finding lambda in K such that:
        (N*lambda + p)^T * (mu - lambda) >= 0  for all mu in K

    where K is the friction cone.
    """

    def __init__(
        self,
        model: Model,
        solver: NumericalSolver | None = None,
        constraint: ContactConstraint | None = None,
        enable_timers: bool = False,
    ):
        """Initialize the contact solver.

        Args:
            model: The simulation model.
            solver: Numerical solver for the contact system.
            constraint: ContactConstraint object (owned by SolverDVI).
            enable_timers: Whether to enable performance timers.
        """
        self.model = model
        self.enable_timers = enable_timers
        self._prev_lambda: wp.array | None = None  # persistent buffer for warm-starting
        self._last_dt: float = 0.005  # cached dt from last solve (for update_contacts)

        # Create default solver if not provided
        if solver is None:
            config = NumericalSolverConfig(
                max_iterations=100,
                tolerance=1e-6,
                omega=0.3,
                relax=0.9,
                alpha=0.2,  # DVI-style damping
                recovery_speed=0.6,  # m/s
                reg=1e-8,
            )
            solver = SparseJacobiSolver(config)

        self.numerical_solver = solver
        self._constraint = constraint

    def _init_friction_coefficients(self, contacts: Contacts):
        """Initialize per-contact friction coefficients from material properties."""
        model = self.model
        contact_max = self._constraint.contact_max
        device = model.device

        if model.shape_material_mu is None:
            self._constraint.friction.fill_(0.5)
        else:
            wp.launch(
                kernel=kernels.init_friction_from_materials_kernel,
                dim=contact_max,
                inputs=[
                    contacts.rigid_contact_count,
                    contacts.rigid_contact_shape0,
                    contacts.rigid_contact_shape1,
                    model.shape_material_mu,
                    0.5,  # default_mu
                    contact_max,
                ],
                outputs=[self._constraint.friction],
                device=device,
            )

    def solve(self, state: State, contacts: Contacts, dt: float, body_inv_inertia_world: wp.array | None = None) -> int:
        """Solve contact forces.

        Args:
            state: Current simulation state.
            contacts: Contact information from collision detection.
            dt: Time step size.
            body_inv_inertia_world: Pre-computed world-frame inverse inertia.

        Returns:
            Number of iterations performed.
        """
        self._last_dt = dt

        if contacts is None or contacts.rigid_contact_max == 0:
            return 0

        model = self.model
        device = model.device
        c = self._constraint
        contact_max = c.contact_max

        c.body_a.fill_(-1)
        c.body_b.fill_(-1)

        with wp.ScopedTimer("DVI: Contact Jacobians", active=self.enable_timers):
            wp.launch(
                kernel=contact_kernels.compute_contact_jacobians,
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
                    c.jacobian,
                    c.body_a,
                    c.body_b,
                    c.violation,
                ],
                device=device,
            )

        with wp.ScopedTimer("DVI: Init friction", active=self.enable_timers):
            self._init_friction_coefficients(contacts)

        with wp.ScopedTimer("DVI: Contact residual", active=self.enable_timers):
            wp.launch(
                kernel=contact_kernels.compute_contact_residual,
                dim=contact_max,
                inputs=[
                    state.body_qd,
                    state.body_f,
                    model.body_inv_mass,
                    body_inv_inertia_world,
                    contacts.rigid_contact_count,
                    c.body_a,
                    c.body_b,
                    c.jacobian,
                    c.violation,
                    dt,
                    self.numerical_solver.config.alpha,
                    self.numerical_solver.config.recovery_speed,
                    contact_max,
                ],
                outputs=[c.residual],
                device=device,
            )

        with wp.ScopedTimer("DVI: Contact diagonal", active=self.enable_timers):
            cfg = self.numerical_solver.config

            # Always compute scalar diagonal (used by non-block code paths)
            wp.launch(
                kernel=contact_kernels.compute_contact_diagonal,
                dim=contact_max,
                inputs=[
                    model.body_inv_mass,
                    body_inv_inertia_world,
                    contacts.rigid_contact_count,
                    c.body_a,
                    c.body_b,
                    c.jacobian,
                    cfg.reg,
                    contact_max,
                ],
                outputs=[c.diag],
                device=device,
            )

            # Compute block-3x3 diagonal inverse for block preconditioning
            if cfg.block_precondition:
                wp.launch(
                    kernel=contact_kernels.compute_contact_diagonal_block3x3,
                    dim=contact_max,
                    inputs=[
                        model.body_inv_mass,
                        body_inv_inertia_world,
                        contacts.rigid_contact_count,
                        c.body_a,
                        c.body_b,
                        c.jacobian,
                        cfg.reg,
                        contact_max,
                    ],
                    outputs=[c.diag_block_inv],
                    device=device,
                )

        # Initialize lambda: warm-start from previous frame or zero
        c.delta_v.zero_()
        warm_start = self.numerical_solver.config.warm_start
        if warm_start and self._prev_lambda is not None and self._has_match_index(contacts):
            wp.launch(
                kernel=contact_kernels.warm_start_lambda,
                dim=contact_max,
                inputs=[
                    contacts.rigid_contact_count,
                    contacts.rigid_contact_match_index,
                    self._prev_lambda,
                    contact_max,
                ],
                outputs=[c.lambda_],
                device=device,
            )
        else:
            c.lambda_.zero_()

        with wp.ScopedTimer("DVI: Contact iterations", active=self.enable_timers):
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
                diag_block_inv=c.diag_block_inv if cfg.block_precondition else None,
                block_precondition=cfg.block_precondition,
                nj=c.n_blocks,
                contact_max=contact_max,
                contact_count_arr=contacts.rigid_contact_count,
                contact_friction=c.friction,
                friction_projection=self.numerical_solver.config.friction_projection,
                omega=self.numerical_solver.config.omega,
                relax=self.numerical_solver.config.relax,
                reg=self.numerical_solver.config.reg,
                x=c.lambda_,
                delta_v=c.delta_v,
            )

            iters = self.numerical_solver.solve(ctx)

        # Save lambda for next-frame warm-starting
        if warm_start:
            self._save_lambda(contacts, contact_max, device)

        return iters

    def _has_match_index(self, contacts: Contacts) -> bool:
        """Check if contacts have match index data from ContactMatcher."""
        return (
            hasattr(contacts, "rigid_contact_match_index")
            and contacts.rigid_contact_match_index is not None
            and contacts.rigid_contact_match_index.shape[0] > 0
        )

    def _save_lambda(self, contacts: Contacts, contact_max: int, device: str):
        """Save current lambda for next-frame warm-starting."""
        c = self._constraint
        nc = contact_max * 3

        # Allocate prev_lambda buffer if needed
        if self._prev_lambda is None or self._prev_lambda.shape[0] < nc:
            self._prev_lambda = wp.zeros(max(nc, 1), dtype=wp.float32, device=device)

        wp.launch(
            kernel=contact_kernels.save_lambda,
            dim=contact_max,
            inputs=[
                contacts.rigid_contact_count,
                c.lambda_,
                contact_max,
            ],
            outputs=[self._prev_lambda],
            device=device,
        )

    def apply_forces(self, state: State, contacts: Contacts, dt: float):
        """Apply solved contact forces to body force buffer.

        Args:
            state: Simulation state to update.
            contacts: Contact information.
            dt: Time step size.
        """
        if contacts is None or contacts.rigid_contact_max == 0:
            return

        c = self._constraint

        wp.launch(
            kernel=contact_kernels.apply_contact_forces,
            dim=c.contact_max,
            inputs=[
                contacts.rigid_contact_count,
                c.body_a,
                c.body_b,
                c.jacobian,
                c.lambda_,
                dt,
                c.contact_max,
            ],
            outputs=[state.body_f],
            device=self.model.device,
        )

    def update_contacts(self, contacts: Contacts):
        """Update contacts object with solved forces.

        Writes both ``contacts.rigid_contact_force`` (normal-only, vec3) and
        ``contacts.force`` (full normal+friction spatial vector) when allocated.
        The latter is required by ``SensorContact`` for contact sensing.

        Args:
            contacts: The contacts object to update.
        """
        if contacts is None:
            return

        c = self._constraint
        contact_max = c.contact_max

        # Write rigid_contact_force (normal component only, legacy)
        contact_count = int(contacts.rigid_contact_count.numpy()[0])
        if contact_count > 0:
            lambda_np = c.lambda_.numpy()
            force_np = contacts.rigid_contact_force.numpy()
            normal_np = contacts.rigid_contact_normal.numpy()
            for i in range(contact_count):
                # Normal force is at index i*3
                force_np[i] = normal_np[i] * lambda_np[i * 3]
            contacts.rigid_contact_force.assign(force_np)

        # Write contacts.force (full normal + friction, spatial_vector)
        # Required by SensorContact for feet_air_time, undesired_contacts, etc.
        if contacts.force is not None and contact_count > 0:
            wp.launch(
                kernel=contact_kernels.write_contact_forces,
                dim=contact_max,
                inputs=[
                    contacts.rigid_contact_count,
                    contacts.rigid_contact_normal,
                    c.lambda_,
                    self._last_dt,
                    contact_max,
                ],
                outputs=[contacts.force],
                device=self.model.device,
            )
