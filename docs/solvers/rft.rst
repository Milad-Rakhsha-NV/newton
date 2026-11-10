.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

Granular Resistive Force Theory
===============================

.. experimental::

    :class:`newton.solvers.SolverRFT`, including its configuration and surface
    inputs, is experimental. Numerical verification and continuum comparisons
    do not establish accuracy for an arbitrary sand, wheel, foot, or robot.

Model and scope
---------------

``SolverRFT`` evaluates surface loads on explicit body-local triangular meshes
and composes them with a rigid-body integrator. It uses the cubic 3D-RFT fit
from Agarwal, Goldman, and Kamrin, *PNAS* 120, e2214017120 (2023),
`Supplementary Table S3 <https://arxiv.org/src/2205.14920>`_. Its nominal
material-strength scale is the supplementary Fig. S4 fit:

.. math::

    \xi_n = \rho_b |\mathbf g|
    (894\mu_{\mathrm{int}}^3 - 386\mu_{\mathrm{int}}^2 + 89\mu_{\mathrm{int}}).

``Config.strength`` can supply an independently calibrated value in N/m³.
This is a bulk-density law, with a stated internal-friction range of [0, 0.9].
The effective interface friction is at most the fully rough limit
``internal_friction``. The fitted angular response remains an approximation.

The published coefficients use the downward-oriented normal of an unoriented
plate. The outward body normal instead determines leading-surface activation
and the sign of compressive traction. These normals must be distinguished for
withdrawal and upper faces. A small dimensionless tolerance resolves the
horizontal-normal limit consistently under float32 rotations. The
implementation suppresses tensile polynomial-fit values and applies the
Coulomb cap to tangential traction.

Triangles are clipped to the occupied half-space before integration. Three
symmetric quadrature points on each resulting subtriangle integrate uniform
translation with depth-dependent force and its moment exactly. Rotating bodies
require mesh refinement because the local motion direction varies over a face.
``max_edge_length`` performs this refinement at construction.

The plane is gravity-aligned in each world. Its coordinate along the upward
gravity direction is ``surface_height``; gravity must be nonzero. Heights can
be supplied per world, including the global gravity entry when present.
Wrenches use world coordinates, with linear force first and torque about each
body's center of mass last.

Usage
-----

.. code-block:: python

    import newton
    import warp as wp
    from newton.solvers import SolverRFT

    builder = newton.ModelBuilder()
    body = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, -0.05), wp.quat_identity()),
        mass=1.0,
        inertia=wp.mat33(0.001, 0.0, 0.0, 0.0, 0.001, 0.0, 0.0, 0.0, 0.001),
    )
    model = builder.finalize()
    surface = newton.Mesh.create_box(0.04, 0.04, 0.01, compute_inertia=False)
    solver = SolverRFT(
        model,
        [(body, surface)],
        config=SolverRFT.Config(velocity_regularization=0.02),
        max_edge_length=0.01,
    )
    state_in, state_out = model.state(), model.state()
    state_in.body_qd.assign([[0.0, 0.0, -0.05, 0.0, 0.0, 0.0]])

    # Prescribed-trajectory force probe; leaves state_in unmodified.
    wrench = solver.evaluate(state_in).numpy()

    # Or integrate with the default SolverSemiImplicit.
    state_in.clear_forces()
    solver.step(state_in, state_out, None, None, 0.0001)

Provide ``rigid_solver=`` to use another integrator with the same model, or call
``apply_forces(state)`` before an externally managed integration step. Calling
both ``apply_forces`` and ``step`` applies the granular load twice. ``step``
restores its input external-force buffer after integration, preserving external
loads without accumulating stale granular loads across substeps. Do not create
a separate hard contact plane for the same bed interaction.

Surface meshes are explicitly supplied. They are not inferred from collision
shapes, and changes to collision geometry do not update the RFT surface.
Normals follow triangle winding. Closed meshes are required for loads on both
sides of an object. Static or kinematic intruders can be probed with
``evaluate`` even when their rigid integrator ignores external forces.

Near rest and dynamics
----------------------

At exactly zero local relative speed the model returns zero load. It does not
represent the set of admissible static granular stresses. A nonzero
``velocity_regularization`` scales the quasi-static amplitude by
:math:`v/\sqrt{v^2+v_\epsilon^2}` while retaining the actual motion direction.
This yields creep under a sustained load; it is not exact static support.
Explicit integration must resolve the resulting velocity-dependent stiffness.

``Config.inertial_coefficient`` adds the normal momentum-transfer contribution
:math:`-\lambda\rho_b \max(\mathbf n\cdot\mathbf v,0)^2\mathbf n` on occupied
leading surfaces. This extension of the 3D quasi-static law needs independent
dynamic calibration. It does not implement the complete dynamic-RFT wheel
model, which also requires a structural correction.

``surface_lowering`` is a GPU array containing an externally prescribed
effective lowering per triangle. It changes the clipping plane and pressure
depth; no evolution rule or material-volume conservation law is supplied.
Configuration is intentionally runtime-only in this first experimental slice;
no USD schema or importer behavior is introduced. Authored material schemas
require separate ownership and importer validation.

Material configuration scalars initialize the authoritative GPU ``material``
table described below. Update that table to change a material during execution.
Other Python inputs captured as kernel arguments, such as ``surface_velocity``,
require graph recapture after modification. Updating existing array contents
does not require recapture.

Limitations include bed compaction and history, concave-body plugging, coupled
intruders, moving free-surface evolution, cohesion, wet soils, grain-scale
fluctuations, and deep burial beyond the linear-depth regime. Differentiation
has not been validated. This solver evolves no granular particles.

Numerical investigations
------------------------

From the repository root:

.. code-block:: console

    uv run python -m unittest newton.tests.rft.test_rft
    uv run python -m scripts.benchmarks.benchmark_rft --device cpu
    uv run python -m scripts.benchmarks.benchmark_rft --device cuda:0

The first benchmark writes analytic plate sweeps, orientation and friction
bounds, similarity scaling, drilling mesh convergence, a regularized-penetration
comparison against scalar RK4, wheel-slip kinematic probes, and evaluation
throughput. The wheel probe and self-convergence study are not independent
physical references. Throughput is force evaluation only, excludes setup and
compilation, and reports whether CUDA graphs were used.

The companion Isaac Lab branch ``milad/go2-rft-mpm`` provides a direct Go2
training task, all-link collision-surface binding, and frozen-policy MPM
inference. Those finite-bed comparisons require reference convergence and
material calibration. Passing analytic checks or producing a policy rollout
does not establish physical sand accuracy.

Per-world materials and experimental bearing
--------------------------------------------

``material`` is a mutable GPU table with columns density, internal friction,
surface friction, strength (-1 selects the published fit), inertial coefficient,
velocity regularization, and bearing multiplier. There is one row per gravity
entry/world, including the global entry. Update existing array contents for
captured rollouts; changing the Python configuration does not update this table.

The default bearing multiplier is zero. A positive value blends the flowing law
with depth-dependent normal pressure on downward-facing patches near rest.
This is an experimental bearing extension, not the published granular law.
It supplies no exact tangential sticking or plastic stress history. Unloading,
work behavior and material-specific accuracy require independent validation.

Optional ``surface_shape_ids`` reject quadrature points inside another primitive
collider of the same body. This approximates an exterior union and requires
refinement at intersections; it is not an exact mesh Boolean operation and does
not resolve shielding between articulated links or interacting granular flows.

``body_impulse`` accumulates granular wrenches in float64. Differences divided
by an elapsed interval yield mean applied forces and torques without aliasing
substep loads. This diagnostic has no effect on rigid dynamics.
