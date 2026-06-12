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
Base classes and data structures for numerical solvers.

This module provides:
- SolveContext: Unified context object for all solvers
- NumericalSolverConfig: Configuration parameters
- NumericalSolver: Abstract base class
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING

import numpy as np
import warp as wp

if TYPE_CHECKING:
    pass


class Device(Enum):
    """Supported compute devices for solvers."""

    CPU = auto()
    CUDA = auto()


class FrictionProjection(Enum):
    """Friction cone projection method for contact solvers.

    Members
    -------
    CONE : str
        Anitescu-Tasora (AT) minimum-norm projection onto the friction cone.
        Projects both normal and tangential components.  This is the standard
        Euclidean projection used in Project DVI.
    TANGENTIAL : str
        Tangential-only clamp.  Preserves the normal component and only
        scales the tangential to satisfy ``|f_t| <= mu * f_n``.  Avoids
        the normal-impulse inflation that the cone projection produces
        at converged fixed points when friction is saturated.
    """

    CONE = "Cone"
    TANGENTIAL = "Tangential"


class SolverType(Enum):
    """Built-in numerical solver types.

    Used in ``NumericalSolverConfig.solver_type`` to select which solver
    implementation to instantiate.

    Members
    -------
    SPARSE_JACOBI : str
        Matrix-free Jacobi using Warp kernels (CPU/GPU).
    SPARSE_LDL : str
        Block-sparse tile LDL direct solver (CPU/GPU).
    """

    SPARSE_JACOBI = "SparseJacobi"
    SPARSE_LDL = "SparseLDL"


@dataclass
class NumericalSolverConfig:
    """Configuration for numerical solvers.

    Attributes:
        max_iterations: Maximum number of iterations (iterative solvers only).
        tolerance: Convergence tolerance (iterative solvers only).
        omega: Step size for gradient descent (iterative solvers only).
        relax: Solution mixing factor, 1.0 = fully use new value (iterative solvers only).
        alpha: Baumgarte damping parameter (DVI-style). The effective correction
            is phi / (dt + alpha). Higher alpha = more damping, less energy injection.
            For joints: alpha=0.2 matches DVI's default (or 1e6 to disable).
            For contacts: use alpha ~ dt (e.g., 0.001-0.005) for proper correction.
            The default 0.005 works well for contacts at typical timesteps.
        recovery_speed: Maximum Baumgarte recovery speed (m/s or rad/s).
            Limits how fast constraint violations are corrected.
            For joints this caps joint-drift correction velocity;
            for contacts it caps the penetration-recovery velocity.
            Matches DVI's ``max_penetration_recovery_speed``.
            Set to -1.0 for unlimited (default). Recommended: 0.6 (DVI default)
            for joints, 0.2-1.0 for contacts.
        reg: Regularization added to diagonal for numerical stability.
        warm_start: Whether to initialize lambda from the previous frame's
            converged values instead of zero.  For **joint constraints** the
            topology is fixed so the lambda buffer is simply reused.  For
            **contacts** the topology changes each frame, so warm-starting
            uses the ``ContactMatcher`` match indices to gather previous
            impulses for matched contacts (unmatched contacts start at zero).
            Requires the collision pipeline to be configured with
            ``contact_matching="latest"`` (or ``"sticky"``) for contacts.
            Fully GPU-resident, CUDA-graph-capturable.
        backtrack_iterations: Number of backtracking iterations.
        position_correction: Optional nested config for position-level correction.
            When set, enables post-stabilization that directly projects constraint
            violations after velocity integration.  For joint configs this fixes
            joint drift; for contact configs this fixes penetration.
            The nested config specifies solver type, iterations, etc. for the
            position solve.  ``None`` (default) disables position correction.
            Ignored on nested configs (no recursive nesting).
    """

    solver_type: SolverType = SolverType.SPARSE_JACOBI
    max_iterations: int = 50
    tolerance: float = 1e-8
    omega: float = 0.3
    relax: float = 0.9
    alpha: float = 0.005  # DVI-style damping: correction = phi / (dt + alpha)
    recovery_speed: float = -1.0  # -1.0 = unlimited, DVI default: 0.6
    reg: float = 1e-8
    warm_start: bool = False
    backtrack_iterations: int = 5
    diagonal_precondition: bool = False
    """Enable symmetric diagonal (Jacobi) preconditioning for the LDL solver.

    Scales the Schur complement N by S = diag(1/sqrt(N_ii)) so that the
    factored matrix has unit diagonal, dramatically improving numerical
    stability for systems with high mass ratios across fixed joints.
    Only used by SparseLDLSolver; ignored by iterative solvers."""
    precond_reg: float = 1e-4
    """Additional regularization added to the diagonal AFTER preconditioning.

    The original ``reg`` is absorbed into the scaling and becomes negligible
    in the scaled system. This parameter adds fresh regularization relative
    to the unit-diagonal scaled system, preventing near-zero pivots during
    LDL factorization. Only used when ``diagonal_precondition=True``.
    Typical values: 1e-4 to 1e-3."""
    friction_projection: FrictionProjection = FrictionProjection.CONE
    """Friction cone projection method for contact solvers.

    - ``CONE`` (default): Anitescu-Tasora minimum-norm projection onto the
      friction cone.  Matches Project DVI's default behaviour.
    - ``TANGENTIAL``: Tangential-only clamp that preserves the normal
      component.  Eliminates the normal-impulse inflation
      (factor ``(1 + 2μ²)/(1 + μ²)``) that the cone projection produces
      at converged fixed points when friction is saturated, especially
      with Baumgarte stabilisation (``alpha > 0``)."""
    iterative_refinement_steps: int = 0
    """Number of iterative refinement steps after LDL factorization.

    Each step computes the residual r = b - N@x (matrix-free via J and M⁻¹),
    re-solves N@dx = r using the existing L/D factors (forward/backward
    substitution only, no re-factorization), and updates x += dx.

    Recovers ~3-4 digits of precision per step in float32, which is
    valuable for systems with high mass ratios across joints where
    cancellation in the Schur complement degrades accuracy.

    Only used by SparseLDLSolver.  Typical values: 0 (disabled),
    1-2 (recommended for high mass-ratio systems)."""

    block_precondition: bool = False
    """Use block-3x3 inverse preconditioner for contact Jacobi/GS solvers.

    When ``True`` (default), computes the full 3x3 diagonal block of
    J M^{-1} J^T per contact and inverts it. This captures cross-coupling
    between normal and tangential directions.

    When ``False``, uses a scalar trace-based approximation:
    D_eff = trace(diag_block) / 3. Cheaper but less stable for some envs."""

    position_correction: NumericalSolverConfig | None = None

    def create_solver(self) -> NumericalSolver:
        """Instantiate the numerical solver selected by ``solver_type``.

        Returns:
            A ``NumericalSolver`` subclass instance configured with this config.

        Raises:
            ValueError: If ``solver_type`` is not a recognised ``SolverType``.
        """
        from .block_sparse_ldl_solver import SparseLDLSolver
        from .sparse_jacobi import SparseJacobiSolver

        _SOLVER_MAP = {
            SolverType.SPARSE_JACOBI: SparseJacobiSolver,
            SolverType.SPARSE_LDL: SparseLDLSolver,
        }

        cls = _SOLVER_MAP.get(self.solver_type)
        if cls is None:
            raise ValueError(
                f"Unknown solver_type={self.solver_type!r}. Valid options: {[t.value for t in SolverType]}"
            )
        return cls(self)


# Type alias for projection function (used for friction cone projection)
ProjectionFn = Callable[[np.ndarray], np.ndarray]


@dataclass
class SolveContext:
    """Unified context for numerical solvers.

    Contains all possible inputs/outputs for different solver types.
    Each solver extracts what it needs and validates required fields.

    The same context structure works for velocity-level, position-level,
    and contact constraint solves. The orchestrator (ConstraintSolver,
    PositionSolver, ContactSolver) is responsible for computing the
    correct RHS vector `b`:

    - Velocity-level: b = -(J @ v_pred + phi / (dt + alpha))  (alpha-damped Baumgarte)
    - Position-level: b = -phi
    - Contacts: includes project_fn for friction cone projection

    The numerical solver solves N @ x = b. Optimization-based solvers
    minimize 0.5 * x^T @ N @ x - b^T @ x (gradient = N @ x - b).
    """

    # Required fields
    b: wp.array  # RHS vector (computed by orchestrator)
    nc: int  # Number of constraints
    nb: int  # Number of bodies
    device: str  # Compute device

    # For dense solvers (pre-built N matrix)
    N: wp.array | None = None

    # For matrix-free joint solvers (to build N on the fly)
    #
    # Compact Jacobian format: [total_nc, 12]
    #   - Joint j is at rows nc_offset[j] to nc_offset[j] + joint_nc[j] - 1
    #   - total_nc = sum of joint_nc[0:nj] (actual constraint count)
    #   - Columns: [0:6] = body_a Jacobian, [6:12] = body_b Jacobian
    #
    # The compact storage uses prefix-sum offsets (nc_offset[j] = sum of joint_nc[0:j])
    # so memory usage scales with actual constraint count.
    J: wp.array | None = None
    body_a: wp.array | None = None  # Parent body indices [nj]
    body_b: wp.array | None = None  # Child body indices [nj]
    joint_nc: wp.array | None = None  # Constraint count per joint [nj]
    nc_offset: wp.array | None = None  # Prefix-sum offsets into compact arrays [nj]
    row_to_block: wp.array | None = None  # Precomputed block index per row [nc]
    active_row_count: wp.array | None = None  # GPU array [1] with active row count
    M_inv_diag: wp.array | None = None  # Inverse mass [nb]
    M_inv_inertia: wp.array | None = None  # Inverse inertia (world frame, pre-computed) [nb]
    diag: wp.array | None = None  # Diagonal preconditioner [max_nc]
    diag_block_inv: wp.array | None = None  # Inverse 3x3 diagonal block per contact [contact_max] mat33
    block_precondition: bool = False  # Use trace-based scalar (block-3x3 introduces ~6% friction error)
    nj: int = 0  # Number of joints

    # For contact solvers (spatial vector Jacobians)
    jac_n_a: wp.array | None = None
    jac_n_b: wp.array | None = None
    jac_t1_a: wp.array | None = None
    jac_t1_b: wp.array | None = None
    jac_t2_a: wp.array | None = None
    jac_t2_b: wp.array | None = None
    contact_count: int = 0
    contact_count_arr: wp.array | None = None
    contact_max: int = 0
    contact_friction: wp.array | None = None
    friction_projection: FrictionProjection = FrictionProjection.CONE

    # For inequality constraints (contacts)
    project_fn: ProjectionFn | None = None
    friction: np.ndarray | None = None

    # Solver config (for iterative solvers)
    omega: float = 0.3  # Relaxation parameter
    relax: float = 0.9  # Solution mixing
    reg: float = 1e-8  # Regularization

    # Output buffers
    x: wp.array | None = None  # Solution [nc]
    delta_v: wp.array | None = None  # Velocity/position change [nb, 6]

    # Work buffers for graph capture
    work_buffers: dict | None = field(default_factory=dict)


# =============================================================================
# Constraint Classes
# =============================================================================


@dataclass
class Constraint:
    """Base class for constraint data.

    Holds all data needed for constraint solving with a unified Jacobian format.
    Subclasses add constraint-type-specific attributes.

    Attributes:
        nc: Total number of constraint rows.
        nb: Number of bodies.
        n_blocks: Number of constraint blocks (joints or contacts).
        device: Compute device.
        jacobian: Jacobian matrix [nc, 12] where each row is
            [body_a_spatial(6), body_b_spatial(6)].
        body_a: Body A index per block.
        body_b: Body B index per block.
        block_nc: Constraint count per block.
        nc_offset: Prefix-sum offsets into Jacobian rows.
        violation: Constraint violation per row.
        residual: RHS vector (b) per row.
        diag: Diagonal preconditioner per row.
        lambda_: Constraint impulses per row.
        delta_v: Velocity correction per body.
    """

    nc: int
    nb: int
    n_blocks: int
    device: str

    jacobian: wp.array | None = None
    body_a: wp.array | None = None
    body_b: wp.array | None = None
    block_nc: wp.array | None = None
    nc_offset: wp.array | None = None
    row_to_block: wp.array | None = None  # Precomputed block index for each row
    active_row_count: wp.array | None = None  # GPU array [1] for active constraint rows
    violation: wp.array | None = None
    residual: wp.array | None = None
    diag: wp.array | None = None
    lambda_: wp.array | None = None
    delta_v: wp.array | None = None

    def allocate(self) -> None:
        """Allocate arrays on the device."""
        with wp.ScopedDevice(self.device):
            nc = max(self.nc, 1)
            self.jacobian = wp.zeros((nc, 12), dtype=wp.float32)
            self.body_a = wp.zeros(max(self.n_blocks, 1), dtype=wp.int32)
            self.body_b = wp.zeros(max(self.n_blocks, 1), dtype=wp.int32)
            self.block_nc = wp.zeros(max(self.n_blocks, 1), dtype=wp.int32)
            self.nc_offset = wp.zeros(max(self.n_blocks, 1), dtype=wp.int32)
            self.row_to_block = wp.zeros(nc, dtype=wp.int32)
            self.active_row_count = wp.array([self.nc], dtype=wp.int32)
            self.violation = wp.zeros(nc, dtype=wp.float32)
            self.residual = wp.zeros(nc, dtype=wp.float32)
            self.diag = wp.zeros(nc, dtype=wp.float32)
            self.lambda_ = wp.zeros(nc, dtype=wp.float32)
            self.delta_v = wp.zeros(max(self.nb, 1), dtype=wp.spatial_vector)

    def to_solve_context(
        self,
        config: NumericalSolverConfig,
        M_inv_diag: wp.array,
        M_inv_inertia: wp.array,
    ) -> SolveContext:
        """Convert to SolveContext for numerical solver.

        Args:
            config: Solver configuration.
            M_inv_diag: Inverse mass per body.
            M_inv_inertia: Inverse inertia per body.

        Returns:
            SolveContext populated with constraint data.
        """
        return SolveContext(
            b=self.residual,
            nc=self.nc,
            nb=self.nb,
            device=self.device,
            J=self.jacobian,
            body_a=self.body_a,
            body_b=self.body_b,
            joint_nc=self.block_nc,
            nc_offset=self.nc_offset,
            row_to_block=self.row_to_block,
            active_row_count=self.active_row_count,
            M_inv_diag=M_inv_diag,
            M_inv_inertia=M_inv_inertia,
            diag=self.diag,
            nj=self.n_blocks,
            omega=config.omega,
            relax=config.relax,
            reg=config.reg,
            x=self.lambda_,
            delta_v=self.delta_v,
        )


@dataclass
class JointConstraint(Constraint):
    """Constraint data for joint constraints.

    No additional attributes beyond the base class.
    """

    pass


@dataclass
class ContactConstraint(Constraint):
    """Constraint data for contact constraints.

    Attributes:
        contact_max: Maximum allocated contacts.
        friction: Friction coefficient per contact.
        diag_block_inv: Inverse of 3x3 diagonal block per contact for block preconditioning.
    """

    contact_max: int = 0
    friction: wp.array | None = None
    diag_block_inv: wp.array | None = None  # [contact_max] mat33

    def allocate(self) -> None:
        """Allocate arrays on the device."""
        super().allocate()
        with wp.ScopedDevice(self.device):
            self.friction = wp.zeros(max(self.contact_max, 1), dtype=wp.float32)
            self.diag_block_inv = wp.zeros(max(self.contact_max, 1), dtype=wp.mat33)


class NumericalSolver(ABC):
    """Abstract base class for numerical solvers.

    All solvers implement solve(ctx: SolveContext) which:
    1. Validates required context fields
    2. Solves the system N @ x = b (direct) or minimizes 0.5*x'Nx - b'x (iterative)
    3. Writes solution to ctx.x and optionally ctx.delta_v

    The same solver can be used for velocity-level, position-level,
    and contact solves - only the context differs.

    Attributes:
        config: Solver configuration parameters.
        supported_devices: Set of devices this solver can run on.
    """

    # Class attribute: which devices this solver supports
    supported_devices: set[Device] = {Device.CPU}

    # Class attribute: solver-specific default parameters
    # Subclasses override to declare which params they use and their defaults
    # Keys are param names, values are the solver's preferred defaults
    default_params: dict[str, float | int] = {"reg": 1e-8}

    def __init__(self, config: NumericalSolverConfig):
        """Initialize solver with configuration.

        Args:
            config: Solver configuration.
        """
        self.config = config

    @abstractmethod
    def solve(self, ctx: SolveContext) -> int:
        """Solve the linear system. Writes solution to ctx.x.

        Args:
            ctx: Solve context with inputs and output buffers.

        Returns:
            Number of iterations performed (1 for direct solvers).

        Raises:
            ValueError: If required context fields are missing.
        """
        pass

    def prepare_for_capture(self, model, state, constraint_info=None, body_inv_inertia_world=None) -> None:
        """Pre-allocate buffers for CUDA graph capture.

        Called once before graph capture to allocate work buffers and
        compute topology-dependent data. Default implementation does nothing.

        Solvers that need pre-allocation for graph capture should override this.

        Args:
            model: The simulation model.
            state: Initial simulation state.
            constraint_info: Optional JointConstraintInfo with pre-computed
                constraint counts. If provided, solvers should use this
                instead of recomputing.
            body_inv_inertia_world: Pre-computed world-frame inverse inertia
                tensors. When provided, used directly instead of rotating
                body-frame tensors.
        """
        pass

    def validate(
        self,
        ctx: SolveContext,
        require_N: bool = False,
        require_matrix_free: bool = False,
    ) -> None:
        """Validate that context has required fields.

        Args:
            ctx: Solve context to validate.
            require_N: If True, ctx.N must be provided.
            require_matrix_free: If True, ctx.J and ctx.M_inv_* must be provided.

        Raises:
            ValueError: If required fields are missing.
        """
        if ctx.b is None:
            raise ValueError("SolveContext.b is required")
        if ctx.nc <= 0:
            raise ValueError("SolveContext.nc must be positive")
        if ctx.x is None:
            raise ValueError("SolveContext.x (output buffer) is required")

        if require_N and ctx.N is None:
            raise ValueError(f"{self.__class__.__name__} requires ctx.N")

        if require_matrix_free:
            if ctx.J is None:
                raise ValueError(f"{self.__class__.__name__} requires ctx.J")
            if ctx.M_inv_diag is None:
                raise ValueError(f"{self.__class__.__name__} requires ctx.M_inv_diag")
            if ctx.M_inv_inertia is None:
                raise ValueError(f"{self.__class__.__name__} requires ctx.M_inv_inertia")

    def can_run_on(self, device: str) -> bool:
        """Check if solver can run on the given device.

        Args:
            device: Device string like "cuda:0" or "cpu".

        Returns:
            True if solver supports this device.
        """
        if "cuda" in device.lower():
            return Device.CUDA in self.supported_devices
        return Device.CPU in self.supported_devices


# =============================================================================
# Friction Cone Projection Utilities
# =============================================================================


def project_friction_cone(x: np.ndarray, friction: np.ndarray) -> np.ndarray:
    """Project contact impulses onto the friction cone.

    Args:
        x: Contact impulses [n*3] where each triple is (normal, tangent1, tangent2).
        friction: Friction coefficients per contact [n].

    Returns:
        Projected impulses satisfying the friction cone constraint.
    """
    x_out = x.copy()
    n_contacts = len(friction)

    for i in range(n_contacts):
        f = x_out[3 * i : 3 * i + 3]
        fn = f[0]
        ft = np.sqrt(f[1] ** 2 + f[2] ** 2)
        mu = friction[i]

        if ft <= mu * fn:
            continue

        if mu != 0.0 and ft < -fn / mu:
            f[0:3] = 0.0
        elif mu == 0.0 and fn < 0:
            f[0] = 0.0
        elif mu * fn < ft:
            f[0] = (fn + mu * ft) / (mu**2 + 1)
            if ft > 1e-10:
                scale = mu * f[0] / ft
                f[1] *= scale
                f[2] *= scale
            else:
                f[1] = 0.0
                f[2] = 0.0

    return x_out


def make_friction_projection(friction: np.ndarray) -> ProjectionFn:
    """Create a friction cone projection function for the given coefficients.

    Args:
        friction: Friction coefficients per contact [n].

    Returns:
        Projection function that can be passed to solvers.
    """

    def project(x: np.ndarray) -> np.ndarray:
        return project_friction_cone(x, friction)

    return project
