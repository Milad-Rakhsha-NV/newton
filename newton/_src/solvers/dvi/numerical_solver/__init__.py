# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Numerical solvers for constraint systems in the DVI solver.

This subpackage provides solver classes for linear systems of the form:

    N @ lambda = -b

where N = J @ M_inv @ J.T is the system matrix, and b is the right-hand side.

Available Solvers
-----------------

Available Solvers:
- SparseJacobiSolver: Matrix-free Jacobi using Warp kernels (CPU/GPU)
- SparseLDLSolver: Block-sparse tile LDL direct solver (CPU/GPU)

For contacts with friction, an optional projection function enforces the friction cone.
For bilateral joint constraints, no projection is needed.
"""

from ..contact_kernels import (
    project_friction_cone_tangential,
)
from .base import (
    Constraint,
    ContactConstraint,
    Device,
    FrictionProjection,
    JointConstraint,
    NumericalSolver,
    NumericalSolverConfig,
    ProjectionFn,
    SolveContext,
    SolverType,
    make_friction_projection,
    project_friction_cone,
)
from .block_sparse_ldl_solver import SparseLDLSolver
from .sparse_jacobi import SparseJacobiSolver

__all__ = [
    "Constraint",
    "ContactConstraint",
    "Device",
    "FrictionProjection",
    "JointConstraint",
    "NumericalSolver",
    "NumericalSolverConfig",
    "ProjectionFn",
    "SolveContext",
    "SolverType",
    "SparseJacobiSolver",
    "SparseLDLSolver",
    "make_friction_projection",
    "project_friction_cone",
    "project_friction_cone_tangential",
]
