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
DVI-style solver for contact dynamics using DVI (Differential Variational Inequality).

This solver implements a maximal coordinate formulation using cone complementarity
for frictional contact dynamics. Contacts and joint constraints are solved using
iterative methods (Jacobi) or direct methods (LDL).

References:
    - M. Anitescu, F. A. Potra. "Formulating Dynamic Multi-Rigid-Body Contact
      Problems with Friction as Solvable Linear Complementarity Problems."
      Nonlinear Dynamics, 1997.
    - A. Tasora, M. Anitescu. "A matrix-free cone complementarity approach for
      solving large-scale, nonsmooth, rigid body dynamics." Computer Methods
      in Applied Mechanics and Engineering, 2011.
"""

from __future__ import annotations

import warnings
from enum import Enum

import numpy as np
import warp as wp

try:
    from typing import override
except ImportError:
    try:
        from typing_extensions import override
    except ImportError:
        # Fallback no-op decorator if override is not available
        def override(func):
            return func


class ActuatorIntegration(Enum):
    """Actuator integration mode for PD control forces.

    Controls how joint stiffness (ke) and damping (kd) are treated in the solver.

    Attributes:
        EXPLICIT: Forces evaluated at current state, no mass augmentation.
            The PD forces ``ke * (q_target - q) + kd * (qdot_target - qdot)``
            are applied as explicit external forces. No inertia augmentation.
            Requires small timesteps for stiff springs.
        SEMI_IMPLICIT: Mass augmentation + explicit forces on RHS.
            PD forces are applied explicitly, but the effective body inertia
            is augmented with ``h * C_eff * (axis ⊗ axis)`` where
            ``C_eff = kd + h * ke``. This provides implicit stability while
            keeping the explicit driving forces. Allows larger timesteps.
        IMPLICIT: Exact implicit prediction with velocity correction.
            Same mass augmentation as SEMI_IMPLICIT, plus an additional
            velocity correction term ``-dt * ke * G^T * G * v`` in the
            prediction step. This matches the exact implicit integration
            of the linearised spring-damper system.
    """

    EXPLICIT = "explicit"
    SEMI_IMPLICIT = "semi_implicit"
    IMPLICIT = "implicit"


from ...sim import Contacts, Control, Model, State
from ...sim.articulation import eval_ik
from ..solver import SolverBase
from . import actuation_kernels, implicit_pd_kernels, kernels
from .constraint_solver import ConstraintSolver
from .contact_position_solver import ContactPositionSolver
from .contact_solver import ContactSolver
from .joint_limit_solver import JointLimitSolver
from .kernels import compute_body_inv_inertia_world
from .numerical_solver import (
    ContactConstraint,
    JointConstraint,
    NumericalSolverConfig,
    SolverType,
)
from .position_solver import PositionSolver
from .utils import DVITimer, print_solver_config, validate_solver_config


class JointConstraintInfo:
    """Pre-computed joint constraint topology (static, computed once).

    Attributes:
        num_joints: Number of joints in the model.
        joint_num_constraints: wp.array[int32] - constraint count per joint.
        joint_nc_offset: wp.array[int32] - prefix-sum offsets for compact storage.
        total_nc: Total number of joint constraints.
    """

    def __init__(
        self,
        num_joints: int,
        joint_num_constraints: wp.array,
        joint_nc_offset: wp.array,
        row_to_block: wp.array,
        total_nc: int,
    ):
        self.num_joints = num_joints
        self.joint_num_constraints = joint_num_constraints
        self.joint_nc_offset = joint_nc_offset
        self.row_to_block = row_to_block
        self.total_nc = total_nc

    @staticmethod
    def from_model(model: Model) -> JointConstraintInfo:
        """Compute constraint topology from model (CPU-based, one-time init)."""
        device = model.device
        num_joints = model.joint_count if model.joint_count > 0 else 1

        if model.joint_count > 0:
            joint_type = model.joint_type.numpy()
            joint_enabled = model.joint_enabled.numpy()
            joint_dof_dim = model.joint_dof_dim.numpy()

            # Constraint count per joint type (must match kernel logic exactly)
            nc_per_joint = np.zeros(num_joints, dtype=np.int32)
            for j in range(num_joints):
                if not joint_enabled[j]:
                    continue
                jtype = joint_type[j]
                n_lin, n_ang = joint_dof_dim[j]
                # FREE=4, FIXED=3, BALL=2, REVOLUTE=1, PRISMATIC=0, COMPOUND=5, D6=6
                if jtype == 4 or jtype == 5:  # FREE or COMPOUND
                    nc_per_joint[j] = 0
                elif jtype == 3:  # FIXED
                    nc_per_joint[j] = 6
                elif jtype == 2:  # BALL
                    nc_per_joint[j] = 3
                elif jtype == 1 or jtype == 0:  # REVOLUTE or PRISMATIC
                    nc_per_joint[j] = 5
                elif jtype == 6:  # D6
                    # Must match kernel logic exactly:
                    # - n_lin==0: 3 pos constraints, n_lin<3: 2, n_lin==3: 0
                    # - n_ang==0: 3 ang constraints, n_ang<3: 2, n_ang==3: 0
                    lin_nc = 3 if n_lin == 0 else (0 if n_lin == 3 else 2)
                    ang_nc = 3 if n_ang == 0 else (0 if n_ang == 3 else 2)
                    nc_per_joint[j] = lin_nc + ang_nc

            # Prefix-sum for compact storage offsets
            nc_offset = np.zeros(num_joints, dtype=np.int32)
            offset = 0
            for j in range(num_joints):
                nc_offset[j] = offset
                offset += int(nc_per_joint[j])
            total_nc = int(offset)

            # Precompute row_to_block mapping (O(1) lookup instead of binary search)
            row_to_block_np = np.zeros(max(total_nc, 1), dtype=np.int32)
            for j in range(num_joints):
                start = nc_offset[j]
                end = start + nc_per_joint[j]
                row_to_block_np[start:end] = j

            # Transfer to device
            with wp.ScopedDevice(device):
                joint_num_constraints = wp.array(nc_per_joint, dtype=wp.int32, device=device)
                joint_nc_offset = wp.array(nc_offset, dtype=wp.int32, device=device)
                joint_row_to_block = wp.array(row_to_block_np, dtype=wp.int32, device=device)
        else:
            total_nc = 0
            with wp.ScopedDevice(device):
                joint_num_constraints = wp.zeros(num_joints, dtype=wp.int32)
                joint_nc_offset = wp.zeros(num_joints, dtype=wp.int32)
                joint_row_to_block = wp.zeros(1, dtype=wp.int32)

        return JointConstraintInfo(
            num_joints=num_joints,
            joint_num_constraints=joint_num_constraints,
            joint_nc_offset=joint_nc_offset,
            row_to_block=joint_row_to_block,
            total_nc=total_nc,
        )


class SolverDVI(SolverBase):
    """A maximal coordinate solver using Differential Variational Inequalities (DVI).

    This solver implements the DVI formulation as described in the DVI engine,
    handling frictional contact dynamics using a cone complementarity approach.
    Joint constraints and contacts are solved using iterative methods.

    The solver separates constraint forces from contact forces and solves them
    in sequence within each time step.

    Solver Types
    ------------
    - **Jacobi**: GPU-friendly parallel solver. Reads from old values, writes to new.
      No race conditions. Good for GPU execution.
    - **LDL**: Block-sparse tile LDL direct solver. Provides exact solutions for
      bilateral constraints.

    Example
    -------

    .. code-block:: python

        solver = newton.solvers.SolverDVI(model)

        # simulation loop
        for i in range(100):
            contacts = model.collide(state_in)
            solver.step(state_in, state_out, control, contacts, dt)
            state_in, state_out = state_out, state_in

    """

    def __init__(
        self,
        model: Model,
        joint_solver: NumericalSolverConfig | None = None,
        contact_solver: NumericalSolverConfig | None = None,
        angular_damping: float = 0.05,
        enable_actuation: bool = True,
        enable_contacts: bool = True,
        enable_gyroscopic: bool = True,
        velocity_correction_after_position: bool = True,
        actuator_integration: ActuatorIntegration | str = ActuatorIntegration.SEMI_IMPLICIT,
        joint_limit_ke_scale: float = 1.0,
        joint_limit_solver: NumericalSolverConfig | None = None,
        enable_timers: bool = False,
        verbose: bool = False,
        # Deprecated — use actuator_integration instead
        use_implicit_pd: bool | None = None,
    ):
        """Initialize the DVI solver.

        Args:
            model: The model to be simulated.
            joint_solver: Config for joint constraint solver.  Set
                ``position_correction`` on the config to enable joint
                position correction.  If None, uses default SparseJacobi.
            contact_solver: Config for contact force solver.  Set
                ``position_correction`` on the config to enable contact
                position correction.  If None, uses default SparseJacobi.
            angular_damping: Angular velocity damping coefficient (0.0 = no damping).
            enable_actuation: Whether to enable PD control actuation forces.
            enable_contacts: Whether to enable contact force solving.
            enable_gyroscopic: Whether to include gyroscopic torque (-ω×(I·ω)).
                Essential for correct angular dynamics with non-spherical inertia.
                Matches Project DVI's ComputeGyro() + IntLoadResidual_F().
            actuator_integration: How PD actuator forces are integrated. Accepts an
                ``ActuatorIntegration`` enum value or one of the strings ``"explicit"``,
                ``"semi_implicit"``, ``"implicit"``.

                - **EXPLICIT**: Forces evaluated at current state, no mass augmentation.
                  Requires small timesteps for stiff springs.
                - **SEMI_IMPLICIT** (default): Mass augmentation via
                  ``I_eff = I_world + h * (kd + h*ke) * Σ(axis ⊗ axis)``
                  with explicit PD forces on the RHS.  Allows larger timesteps.
                - **IMPLICIT**: Same augmentation as SEMI_IMPLICIT, plus an additional
                  velocity correction ``-dt * ke * G^T * G * v`` for exact implicit
                  prediction of the linearised spring-damper.

                **Phase 1 limitation:** Only angular free DOFs are treated implicitly.
                Linear free DOFs (prismatic, D6-linear) remain explicit.
            joint_limit_ke_scale: Scale factor applied to joint limit stiffness (ke)
                and damping (kd) values.  USD/MJCF importers often produce very high
                limit stiffness (e.g. 10,000) that is tuned for implicit constraint
                solvers.  For penalty-based enforcement, lower values (e.g. 0.01–0.1)
                keep limit forces proportional to actuator effort.  Default 1.0 = no
                scaling.  Only used in penalty mode (when ``joint_limit_solver``
                is None).
            joint_limit_solver: Config for constraint-based joint limit solver.
                When provided, joint limits are enforced as unilateral constraints
                (λ ≥ 0) solved before the bilateral joint solve so that the
                LDL solver can account for limit impulses.
                When None (default), joint limits use penalty-based spring-damper
                forces in the actuation kernel.
            enable_timers: Whether to enable performance timers.
            verbose: Whether to print solver configuration on init.
            use_implicit_pd: **Deprecated.** Use ``actuator_integration`` instead.
                ``True`` maps to ``ActuatorIntegration.SEMI_IMPLICIT``,
                ``False`` maps to ``ActuatorIntegration.EXPLICIT``.

        Note:
            For contact-enabled simulations, call `model.collide()` at least once before
            creating the solver. This populates `model.rigid_contact_max` with the correct
            buffer size. If the solver is created before any collision, it defaults to
            1000 contacts which may be insufficient for large scenes.
        """
        super().__init__(model=model)

        # Handle deprecated use_implicit_pd parameter
        if use_implicit_pd is not None:
            warnings.warn(
                "use_implicit_pd is deprecated. Use actuator_integration instead: "
                "True → ActuatorIntegration.SEMI_IMPLICIT, "
                "False → ActuatorIntegration.EXPLICIT.",
                DeprecationWarning,
                stacklevel=2,
            )
            actuator_integration = (
                ActuatorIntegration.SEMI_IMPLICIT if use_implicit_pd else ActuatorIntegration.EXPLICIT
            )

        # Accept string values for convenience
        if isinstance(actuator_integration, str):
            actuator_integration = ActuatorIntegration(actuator_integration)

        self.angular_damping = angular_damping
        self.enable_actuation = enable_actuation
        self.enable_contacts = enable_contacts
        self.enable_gyroscopic = enable_gyroscopic
        self.velocity_correction_after_position = velocity_correction_after_position
        self.actuator_integration = actuator_integration
        self.joint_limit_ke_scale = joint_limit_ke_scale
        self.enable_timers = enable_timers
        self.verbose = verbose

        # Resolve configs → solver instances
        if joint_solver is None:
            joint_solver = NumericalSolverConfig(
                solver_type=SolverType.SPARSE_JACOBI,
                max_iterations=50,
                tolerance=1e-6,
                omega=0.3,
                relax=0.7,
                alpha=0.2,
                reg=1e-8,
            )
        if contact_solver is None:
            contact_solver = NumericalSolverConfig(
                solver_type=SolverType.SPARSE_JACOBI,
                max_iterations=100,
                tolerance=1e-6,
                omega=0.3,
                relax=0.7,
                alpha=0.2,
                recovery_speed=0.6,
                reg=1e-8,
            )

        joint_numerical_solver = joint_solver.create_solver()
        contact_numerical_solver = contact_solver.create_solver()

        # Position correction is enabled via nested config
        self.enable_position_correction = joint_solver.position_correction is not None
        self.enable_contact_position_correction = contact_solver.position_correction is not None

        position_numerical_solver = (
            joint_solver.position_correction.create_solver() if joint_solver.position_correction is not None else None
        )

        # Compute joint constraint topology once (shared by constraint and position solvers)
        self._joint_constraint_info = JointConstraintInfo.from_model(model)

        # Create JointConstraint object (owned by SolverDVI, shared by sub-solvers)
        self._joint_constraint = JointConstraint(
            nc=self._joint_constraint_info.total_nc,
            nb=model.body_count,
            n_blocks=self._joint_constraint_info.num_joints,
            device=model.device,
        )
        self._joint_constraint.allocate()
        # Copy topology info to the constraint object
        self._joint_constraint.block_nc = self._joint_constraint_info.joint_num_constraints
        self._joint_constraint.nc_offset = self._joint_constraint_info.joint_nc_offset
        self._joint_constraint.row_to_block = self._joint_constraint_info.row_to_block

        # Create ContactConstraint object (owned by SolverDVI, shared by sub-solvers)
        contact_max = model.rigid_contact_max if model.rigid_contact_max > 0 else 1000
        self._contact_constraint = ContactConstraint(
            nc=contact_max * 3,
            nb=model.body_count,
            n_blocks=contact_max,
            device=model.device,
            contact_max=contact_max,
        )
        self._contact_constraint.allocate()
        # Initialize nc_offset for contacts (each contact has 3 constraints)
        contact_nc_offset = np.arange(0, contact_max * 3, 3, dtype=np.int32)
        contact_block_nc = np.full(contact_max, 3, dtype=np.int32)
        with wp.ScopedDevice(model.device):
            self._contact_constraint.nc_offset = wp.array(contact_nc_offset, dtype=wp.int32, device=model.device)
            self._contact_constraint.block_nc = wp.array(contact_block_nc, dtype=wp.int32, device=model.device)

        # Create constraint and contact solvers
        # Note: We disable verbose timers in child solvers and use DVITimer instead
        self._constraint_solver = ConstraintSolver(
            model,
            solver=joint_numerical_solver,
            constraint=self._joint_constraint,
            enable_timers=False,  # Use DVITimer instead
        )

        self._contact_solver = ContactSolver(
            model,
            solver=contact_numerical_solver,
            constraint=self._contact_constraint,
            enable_timers=False,  # Use DVITimer instead
        )

        # Create joint limit solver if config provided (constraint-based limits)
        self._joint_limit_solver: JointLimitSolver | None = None
        if joint_limit_solver is not None and model.joint_count > 0 and model.joint_limit_ke is not None:
            self._joint_limit_solver = JointLimitSolver(
                model,
                solver=joint_limit_solver.create_solver(),
                enable_timers=False,
            )

        # Create position solver if numerical solver provided
        self._position_solver: PositionSolver | None = None
        if position_numerical_solver is not None:
            self._position_solver = PositionSolver(
                model,
                solver=position_numerical_solver,
                constraint=self._joint_constraint,
            )

        # Create contact position solver if nested config provided
        self._contact_position_solver: ContactPositionSolver | None = None
        if contact_solver.position_correction is not None:
            self._contact_position_solver = ContactPositionSolver(
                model,
                solver=contact_solver.position_correction.create_solver(),
            )

        # Pre-computed world-frame inverse inertia (reused every step)
        nb = model.body_count if model.body_count > 0 else 1
        self._body_inv_inertia_world = wp.zeros(nb, dtype=wp.mat33, device=model.device)

        # Per-body inertia augmentation buffer for implicit PD mode.
        # Stores h * C_eff * sum(axis ⊗ axis) accumulated from all connected joints.
        # Allocated regardless of actuator_integration mode so the buffer exists if
        # the mode is toggled at runtime.
        self._body_inertia_augment = wp.zeros(nb, dtype=wp.mat33, device=model.device)

        # Zero array for joint_f fallback when control.joint_f is None
        ndof = model.joint_dof_count if model.joint_dof_count > 0 else 1
        self._zero_joint_f = wp.zeros(ndof, dtype=float, device=model.device)

        # Pre-scale joint limit ke/kd if a non-unity scale is requested.
        # Stored as solver-owned arrays so the model object is not mutated.
        if joint_limit_ke_scale != 1.0 and model.joint_limit_ke is not None:
            ke_np = model.joint_limit_ke.numpy() * joint_limit_ke_scale
            kd_np = model.joint_limit_kd.numpy() * joint_limit_ke_scale
            with wp.ScopedDevice(model.device):
                self._scaled_limit_ke = wp.array(ke_np, dtype=float, device=model.device)
                self._scaled_limit_kd = wp.array(kd_np, dtype=float, device=model.device)
        else:
            self._scaled_limit_ke = None
            self._scaled_limit_kd = None

        # Create timing utility
        self._timer = DVITimer(enabled=enable_timers)

        # Store references for external access
        self.joint_numerical_solver = joint_numerical_solver
        self.contact_numerical_solver = contact_numerical_solver
        self.position_numerical_solver = position_numerical_solver
        contact_position_numerical_solver = (
            self._contact_position_solver.numerical_solver if self._contact_position_solver is not None else None
        )

        # Validate configuration
        self._validate_and_print_config(contact_position_numerical_solver)

    def _validate_and_print_config(self, contact_position_numerical_solver=None):
        """Validate solver configuration and print summary."""
        validate_solver_config(
            self.joint_numerical_solver,
            self.contact_numerical_solver,
            self.enable_position_correction,
            self.enable_contact_position_correction,
        )
        if self.verbose:
            print_solver_config(
                self.joint_numerical_solver,
                self.contact_numerical_solver,
                self.enable_position_correction,
                self.enable_contact_position_correction,
                position_solver=self.position_numerical_solver,
                contact_position_solver=contact_position_numerical_solver,
            )

    # =========================================================================
    # Public properties for accessing internal solver state
    # =========================================================================

    @property
    def joint_lambda(self):
        """Access joint constraint impulses."""
        return self._joint_constraint.lambda_

    @property
    def contact_lambda(self):
        """Access contact impulses."""
        return self._contact_constraint.lambda_

    @property
    def contact_body_a(self):
        """Access contact body A indices."""
        return self._contact_constraint.body_a

    @property
    def contact_body_b(self):
        """Access contact body B indices."""
        return self._contact_constraint.body_b

    def finalize_for_capture(self, state: State):
        """Initialize solver for CUDA graph capture.

        Call this method once before graph capture to pre-allocate buffers
        and compute topology-dependent data. This enables GPU-native execution
        during graph capture for LDL-based solvers.

        Args:
            state: Initial simulation state.
        """
        self._constraint_solver.finalize_for_capture(state)
        if self._position_solver is not None:
            self._position_solver.finalize_for_capture(state)

    # =========================================================================
    # Actuation
    # =========================================================================

    def update_actuation_forces(
        self,
        model: Model,
        state: State,
        control: Control,
    ):
        """Compute actuation forces (PD control + direct joint forces) and add to body forces.

        When ``self.actuator_integration`` is SEMI_IMPLICIT or IMPLICIT, the PD terms
        (ke * error + kd * vel_error) are kept at full strength.  The mass augmentation
        in implicit_pd_kernels provides implicit stability, but these explicit forces are
        still needed as the driving terms that actually decelerate the system
        (see implicit-pd.md Section 9).
        """
        if model.joint_count == 0:
            return

        joint_f = control.joint_f if control.joint_f is not None else self._zero_joint_f

        # When constraint-based limits are active, skip penalty-based limits in actuation
        _has_limits = model.joint_limit_ke is not None and self._joint_limit_solver is None
        # Provide dummy arrays when no limits exist (Warp kernels need valid arrays)
        _limit_lower = model.joint_limit_lower if _has_limits else self._zero_joint_f
        _limit_upper = model.joint_limit_upper if _has_limits else self._zero_joint_f
        _limit_ke = (self._scaled_limit_ke or model.joint_limit_ke) if _has_limits else self._zero_joint_f
        _limit_kd = (self._scaled_limit_kd or model.joint_limit_kd) if _has_limits else self._zero_joint_f

        wp.launch(
            kernel=actuation_kernels.compute_actuation_forces,
            dim=model.joint_count,
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
                control.joint_target_pos,
                control.joint_target_vel,
                model.joint_target_ke,
                model.joint_target_kd,
                joint_f,
                int(self.actuator_integration != ActuatorIntegration.EXPLICIT),
                _limit_lower,
                _limit_upper,
                _limit_ke,
                _limit_kd,
                int(_has_limits),
                model.joint_effort_limit if model.joint_effort_limit is not None else self._zero_joint_f,
                int(model.joint_effort_limit is not None),
            ],
            outputs=[state.body_f],
            device=model.device,
        )

    # =========================================================================
    # Main Step Function
    # =========================================================================

    @override
    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control | None,
        contacts: Contacts | None,
        dt: float,
    ):
        """Advance the simulation by one time step.

        The integration proceeds as follows:
        1. Clear forces
        2. Apply actuation forces (PD control)
        3. Solve joint constraints
        4. Apply joint constraint forces
        5. Solve contact forces
        6. Apply contact forces
        7. Integrate velocities and positions

        Args:
            state_in: Input state.
            state_out: Output state (will be overwritten).
            control: Control inputs (joint targets, forces).
            contacts: Contact information from collision detection.
            dt: Time step size.
        """
        model = self.model
        device = model.device
        timer = self._timer

        if control is None:
            control = model.control(clone_variables=False)

        timer.start("total")

        # 0. Pre-compute world-frame inverse inertia (used by all solvers)
        #    When actuator_integration is SEMI_IMPLICIT/IMPLICIT or joints have
        #    armature, this is deferred until after actuation so the augmentation
        #    can be accumulated.
        _needs_augmentation = self.actuator_integration != ActuatorIntegration.EXPLICIT or model.joint_count > 0
        if model.body_count and not _needs_augmentation:
            wp.launch(
                kernel=compute_body_inv_inertia_world,
                dim=model.body_count,
                inputs=[
                    state_in.body_q,
                    model.body_inv_inertia,
                    model.body_inv_mass,
                ],
                outputs=[self._body_inv_inertia_world],
                device=device,
            )

        # 1. Apply gravity to body_f (needed for constraint residual computation)
        if model.body_count:
            wp.launch(
                kernel=kernels.apply_gravity_forces,
                dim=model.body_count,
                inputs=[
                    model.body_mass,
                    model.body_inv_mass,
                    model.gravity,
                ],
                outputs=[state_in.body_f],
                device=device,
            )

        # 1b. Apply gyroscopic torque: -omega x (I * omega)
        # This is essential for correct angular dynamics with non-spherical inertia.
        # Matches Project DVI's ComputeGyro() + IntLoadResidual_F() behavior.
        if model.body_count and self.enable_gyroscopic:
            wp.launch(
                kernel=kernels.apply_gyroscopic_forces,
                dim=model.body_count,
                inputs=[
                    state_in.body_q,
                    state_in.body_qd,
                    model.body_inertia,
                    model.body_inv_mass,
                ],
                outputs=[state_in.body_f],
                device=device,
            )

        # 3. Apply actuation forces (PD control + direct joint forces)
        #    When actuator_integration is SEMI_IMPLICIT or IMPLICIT, ke/kd forces
        #    are KEPT at full strength.  The mass augmentation below provides implicit
        #    stability, but the explicit PD forces are still needed as the driving
        #    terms (see implicit-pd.md §9).
        if model.joint_count > 0 and self.enable_actuation:
            self.update_actuation_forces(model, state_in, control)

        # 3b. Compute augmented world-frame inverse inertia.
        #     This replaces the standard compute_body_inv_inertia_world (step 0)
        #     with an augmented version that includes:
        #       - Implicit PD stiffness/damping: h * C_eff * (axis ⊗ axis)
        #       - Joint armature: armature * (axis ⊗ axis)
        #
        #     Armature is a physical inertia addition (no dt scaling), matching
        #     Featherstone/MuJoCo where armature is added directly to the
        #     joint-space mass matrix diagonal.  It can exceed body inertia
        #     by 100x for small links — this is correct and stabilises PD control.
        #
        #     Pipeline:
        #       (a) Zero the augmentation buffer
        #       (b) Accumulate h * C_eff * (axis ⊗ axis) from implicit PD
        #       (b2) Accumulate armature * (axis ⊗ axis) from joint armature
        #       (c) Compute I_eff = R * I_body * R^T + augmentation, then invert
        if model.body_count and _needs_augmentation:
            # (a) Zero the augmentation buffer each substep
            self._body_inertia_augment.zero_()

            # (b) Accumulate angular damping augmentation from implicit PD
            if model.joint_count > 0 and self.actuator_integration in (
                    ActuatorIntegration.SEMI_IMPLICIT,
                    ActuatorIntegration.IMPLICIT,
                ):
                wp.launch(
                    kernel=implicit_pd_kernels.accumulate_angular_damping_augmentation,
                    dim=model.joint_count,
                    inputs=[
                        state_in.body_q,
                        model.joint_type,
                        model.joint_enabled,
                        model.joint_parent,
                        model.joint_child,
                        model.joint_X_p,
                        model.joint_axis,
                        model.joint_qd_start,
                        model.joint_dof_dim,
                        model.joint_target_ke,
                        model.joint_target_kd,
                        dt,
                    ],
                    outputs=[self._body_inertia_augment],
                    device=device,
                )

            # (b2) Accumulate armature augmentation
            if model.joint_count > 0:
                wp.launch(
                    kernel=implicit_pd_kernels.accumulate_armature_augmentation,
                    dim=model.joint_count,
                    inputs=[
                        state_in.body_q,
                        model.joint_type,
                        model.joint_enabled,
                        model.joint_parent,
                        model.joint_child,
                        model.joint_X_p,
                        model.joint_axis,
                        model.joint_qd_start,
                        model.joint_dof_dim,
                        model.joint_armature,
                        dt,
                    ],
                    outputs=[self._body_inertia_augment],
                    device=device,
                )

            # (b3) Accumulate conditional joint limit augmentation
            #      Only augments when the joint angle is outside [lower, upper].
            if model.joint_count > 0 and model.joint_limit_ke is not None and self._joint_limit_solver is None:
                wp.launch(
                    kernel=implicit_pd_kernels.accumulate_joint_limit_augmentation,
                    dim=model.joint_count,
                    inputs=[
                        state_in.body_q,
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
                        self._scaled_limit_ke or model.joint_limit_ke,
                        self._scaled_limit_kd or model.joint_limit_kd,
                        dt,
                    ],
                    outputs=[self._body_inertia_augment],
                    device=device,
                )

            # (c) Compute augmented inverse inertia: inv(I_world + augmentation)
            wp.launch(
                kernel=implicit_pd_kernels.compute_body_inv_inertia_world_augmented,
                dim=model.body_count,
                inputs=[
                    state_in.body_q,
                    model.body_inertia,
                    model.body_inv_mass,
                    self._body_inertia_augment,
                ],
                outputs=[self._body_inv_inertia_world],
                device=device,
            )

        # 3c. IMPLICIT mode: add velocity correction for exact prediction
        #     The existing actuation kernel provides f_PD = G^T*b_act/dt - kd*G^T*G*v.
        #     For IMPLICIT mode, the exact prediction also needs -dt*ke*G^T*G*v.
        #     This additional term makes the prediction match the fully implicit
        #     linearised spring-damper system.
        if (
            self.actuator_integration == ActuatorIntegration.IMPLICIT
            and model.joint_count > 0
            and self.enable_actuation
        ):
            wp.launch(
                kernel=implicit_pd_kernels.compute_implicit_velocity_correction,
                dim=model.joint_count,
                inputs=[
                    state_in.body_q,
                    state_in.body_qd,
                    model.joint_type,
                    model.joint_enabled,
                    model.joint_parent,
                    model.joint_child,
                    model.joint_X_p,
                    model.joint_axis,
                    model.joint_qd_start,
                    model.joint_dof_dim,
                    model.joint_target_ke,
                    dt,
                ],
                outputs=[state_in.body_f],
                device=device,
            )

        # 4a. Solve and apply joint limit constraints (unilateral, before bilateral)
        # Solved first so bilateral joint solver can account for limit forces.
        if self._joint_limit_solver is not None:
            timer.start("joint_limit_solve")
            self._joint_limit_solver.solve(state_in, dt, self._body_inv_inertia_world)
            timer.timings.joint_limit_solve = timer.stop("joint_limit_solve")
            self._joint_limit_solver.apply_forces(state_in, dt)

        # 4b-5. Solve and apply bilateral joint constraints
        if model.joint_count > 0:
            timer.start("joint_solve")
            self._constraint_solver.solve(state_in, dt, self._body_inv_inertia_world)
            timer.timings.joint_solve = timer.stop("joint_solve")
            self._constraint_solver.apply_forces(state_in, dt)

        # 6-7. Solve and apply contacts
        if contacts is not None and self.enable_contacts:
            timer.start("contact_solve")
            self._contact_solver.solve(state_in, contacts, dt, self._body_inv_inertia_world)
            timer.timings.contact_solve = timer.stop("contact_solve")
            self._contact_solver.apply_forces(state_in, contacts, dt)

        # 8. Integrate
        if model.body_count:
            timer.start("integration")
            wp.launch(
                kernel=kernels.integrate_bodies_euler,
                dim=model.body_count,
                inputs=[
                    state_in.body_q,
                    state_in.body_qd,
                    state_in.body_f,
                    model.body_com,
                    model.body_mass,
                    model.body_inertia,
                    model.body_inv_mass,
                    self._body_inv_inertia_world,
                    model.gravity,
                    self.angular_damping,
                    dt,
                ],
                outputs=[state_out.body_q, state_out.body_qd],
                device=device,
            )
            timer.timings.integration = timer.stop("integration")

        # 9. Position correction (post-stabilization)
        if self.enable_position_correction and model.joint_count > 0 and self._position_solver is not None:
            timer.start("position_solve")
            self._position_solver.solve(
                state_out,
                body_inv_inertia_world=self._body_inv_inertia_world if _needs_augmentation else None,
            )

            # 9b. Update velocity to be consistent with position correction.
            # Without this, position correction moves bodies but leaves velocities
            # inconsistent, causing energy injection. This matches XPBD's approach
            # where velocity is derived from position changes.
            if self.velocity_correction_after_position:
                wp.launch(
                    kernel=kernels.apply_velocity_correction_from_position,
                    dim=model.body_count,
                    inputs=[
                        self._position_solver.position_correction,
                        model.body_inv_mass,
                        dt,
                    ],
                    outputs=[state_out.body_qd],
                    device=device,
                )
            timer.timings.position_solve = timer.stop("position_solve")

        # 10. Contact position correction (fix penetration at position level)
        if (
            self.enable_contact_position_correction
            and contacts is not None
            and self._contact_position_solver is not None
        ):
            timer.start("contact_position_solve")
            self._contact_position_solver.solve(
                state_out,
                contacts,
                body_inv_inertia_world=self._body_inv_inertia_world if _needs_augmentation else None,
            )

            # Note: We intentionally do NOT apply velocity correction for
            # contact position correction. Unlike joint position correction
            # (where velocity must track position to prevent oscillation),
            # contact position correction only fixes geometric penetration.
            # The velocity-level contact solver already produced correct
            # separating velocities — converting the position fix into
            # delta_v/dt would inject energy and launch bodies upward.
            timer.timings.contact_position_solve = timer.stop("contact_position_solve")

        # 11. Joint limit velocity clamping — REMOVED.
        #     Previously clamped body velocities post-integration, but this
        #     provided a rigid wall that RL agents exploited (ratcheting).
        #     Joint limits are now handled as penalty forces in step 3
        #     (actuation) with implicit mass augmentation in step 3b.

        # 12. Inverse kinematics: project body_q/body_qd → joint_q/joint_qd.
        #     DVI solves in maximal coordinates, so state.joint_q/joint_qd
        #     would be stale without this. Needed for ArticulationView's
        #     get_dof_positions()/get_dof_velocities() to return correct values.
        eval_ik(model, state_out, state_out.joint_q, state_out.joint_qd)

        timer.timings.total = timer.stop("total")

        # Print timing and advance frame
        timer.print_timings()
        timer.next_frame()

        return state_out

    @override
    def update_contacts(self, contacts: Contacts, state: State | None = None) -> None:
        """Update contacts object with solved forces.

        Args:
            contacts: The contacts object to update.
            state: Optional simulation state (unused by DVI, kept for API compatibility).
        """
        self._contact_solver.update_contacts(contacts)
