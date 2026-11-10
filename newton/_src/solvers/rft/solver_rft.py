# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Reduced-order granular interaction composed with rigid-body time integration."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
import warp as wp

from ...geometry import Mesh
from ...sim import Contacts, Control, Model, ModelFlags, State, StateFlags
from ..semi_implicit import SolverSemiImplicit
from ..solver import SolverBase
from .kernels import accumulate_impulse, add_forces, integrate_surfaces


class SolverRFT(SolverBase):
    """Integrate rigid bodies interacting with a prescribed dry granular bed.

    .. experimental::

        This solver and its configuration may change without prior notice.
        It models homogeneous, cohesionless grains and a gravity-aligned plane.
        It does not evolve terrain, compaction, static supporting stresses, or
        interactions between overlapping intruder flow regions. Gradients have
        not been validated. With default bearing=0, exact rest produces zero granular force; optional
        velocity regularization introduces creep rather than exact static support.

    Surface tractions use the cubic three-dimensional RFT fit of Agarwal,
    Goldman, and Kamrin (2023), doi:10.1073/pnas.2214017120, SI Table S3.
    The published fit uses a downward **plate** normal, distinct from the outward
    body normal used to select leading surfaces. Tensile fitted tractions are
    suppressed and tangential stresses are capped by interface friction.

    Triangles are clipped at the bed surface and integrated with degree-two
    quadrature. For uniform translation, this integrates linear depth and its
    moment exactly on each planar face. Rotational flow needs spatial refinement.

    Args:
        model: Rigid-body simulation model.
        surfaces: Pairs of body index and :class:`~newton.Mesh`. Vertices are in
            body-local coordinates [m]; triangle winding must point outward.
            Multiple meshes may belong to one body. Collision shapes are not
            automatically included. Use closed meshes for bidirectional intrusion.
        config: Material and numerical parameters. A fresh default is used if omitted.
        rigid_solver: Integrator used after force evaluation. Defaults to
            :class:`~newton.solvers.SolverSemiImplicit` with zero angular damping.
            It must use the same model. Articulations require a suitable integrator.
        max_edge_length: Maximum local triangle edge length [m]. If set, triangles
            are subdivided at construction; no runtime remeshing occurs.
        surface_shape_ids: Optional collider id per surface, for primitive
            same-body internal-patch rejection. Triangle quadrature rejection
            approximates union boundaries and needs intersection refinement.
            Without this mapping all supplied surfaces are integrated independently.
        surface_height: Plane coordinate along upward gravity direction [m], as a
            scalar or one value per gravity entry/world. World translations along
            the upward direction require corresponding plane heights.

    Note:
        Use :meth:`evaluate` to obtain loads on prescribed trajectories without
        integrating dynamics. Use :meth:`apply_forces` with an externally managed
        integrator, or :meth:`step` for composition. Do not apply both in one step.
    """

    @dataclass
    class Config:
        """Material parameters and explicit near-zero-speed behavior."""

        density: float = 1600.0
        """Dense bulk bed density [kg/m³], not intrinsic grain density."""

        internal_friction: float = 0.4
        """Bulk friction coefficient; supported fit range is [0, 0.9]."""

        surface_friction: float = 0.4
        """Effective body-bed friction coefficient in [0, internal_friction]."""

        strength: float | None = None
        """Measured traction-per-depth scale [N/m³]; None uses SI Fig. S4's cubic fit."""

        inertial_coefficient: float = 0.0
        """Dimensionless normal momentum-flux coefficient; zero disables this correction."""

        velocity_regularization: float = 0.0
        """Speed scale [m/s] for smooth traction amplitude; zero keeps the ideal sliding law."""

        bearing: float = 0.0
        """Experimental near-rest normal-bearing multiplier; zero retains pure RFT.

        Blends depth-dependent compressive pressure on downward-facing surfaces
        near zero speed. It has no tangential sticking/history and is not the
        published constitutive law. Calibration and unloading validation are required.
        """

        surface_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
        """Prescribed bed velocity in world coordinates [m/s]."""

    def __init__(
        self,
        model: Model,
        surfaces: Sequence[tuple[int, Mesh]],
        *,
        config: Config | None = None,
        rigid_solver: SolverBase | None = None,
        max_edge_length: float | None = None,
        surface_height: float | Sequence[float] = 0.0,
        surface_shape_ids: Sequence[int] | None = None,
    ):
        super().__init__(model)
        self.config = replace(config) if config is not None else self.Config()
        cfg = self.config
        for name in (
            "density",
            "internal_friction",
            "surface_friction",
            "inertial_coefficient",
            "velocity_regularization",
            "bearing",
        ):
            value = getattr(cfg, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if cfg.density == 0.0:
            raise ValueError("density must be positive")
        if cfg.internal_friction > 0.9:
            raise ValueError("internal_friction exceeds the published fit range [0, 0.9]")
        if cfg.surface_friction > cfg.internal_friction:
            raise ValueError("surface_friction must not exceed internal_friction (fully rough limit)")
        if cfg.strength is not None and (not math.isfinite(cfg.strength) or cfg.strength < 0.0):
            raise ValueError("strength must be finite and nonnegative, or None")
        velocity = np.asarray(cfg.surface_velocity, dtype=np.float64)
        if velocity.shape != (3,) or not np.isfinite(velocity).all():
            raise ValueError("surface_velocity must have three finite components")
        if max_edge_length is not None and (not math.isfinite(max_edge_length) or max_edge_length <= 0.0):
            raise ValueError("max_edge_length must be finite and positive")
        gravity = model.gravity.numpy()
        if not np.isfinite(gravity).all() or np.any(np.linalg.norm(gravity, axis=1) == 0):
            raise ValueError("A gravity-loaded RFT bed requires nonzero, finite gravity in every world")
        height = np.asarray(surface_height, dtype=np.float64)
        if height.ndim == 0:
            height = np.full(len(gravity), float(height))
        if height.shape != (len(gravity),) or not np.isfinite(height).all():
            raise ValueError("surface_height must be finite and scalar or one value per gravity entry")
        triangles = []
        bodies = []
        triangle_shapes = []
        if surface_shape_ids is not None and len(surface_shape_ids) != len(surfaces):
            raise ValueError("surface_shape_ids must match surfaces")
        shape_bodies = model.shape_body.numpy() if surface_shape_ids is not None else None
        for surface_index, (body, mesh) in enumerate(surfaces):
            shape = -1 if surface_shape_ids is None else int(surface_shape_ids[surface_index])
            if shape >= 0 and (shape >= model.shape_count or shape_bodies[shape] != body):
                raise ValueError("Every surface shape must belong to its surface body")
            if not isinstance(body, (int, np.integer)) or not 0 <= body < model.body_count:
                raise ValueError("Every surface must belong to a valid rigid body")
            if not isinstance(mesh, Mesh):
                raise TypeError("Every surface geometry must be a newton.Mesh")
            vertices = np.asarray(mesh.vertices, dtype=np.float64)
            indices = np.asarray(mesh.indices)
            if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
                raise ValueError("Mesh vertices must have shape (N, 3) and be finite")
            if indices.size % 3 or not np.issubdtype(indices.dtype, np.integer):
                raise ValueError("Mesh indices must contain integer triangle triplets")
            if np.any(indices < 0) or np.any(indices >= len(vertices)):
                raise ValueError("Mesh indices are out of bounds")
            pending = list(vertices[indices.reshape(-1, 3)])
            while pending:
                triangle = pending.pop()
                if np.linalg.norm(np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])) == 0.0:
                    continue
                edges = triangle - np.roll(triangle, 1, axis=0)
                if max_edge_length is not None and np.max(np.linalg.norm(edges, axis=1)) > max_edge_length:
                    a, b, c = triangle
                    ab, bc, ca = (a + b) / 2, (b + c) / 2, (c + a) / 2
                    pending.extend(
                        (np.array([a, ab, ca]), np.array([ab, b, bc]), np.array([ca, bc, c]), np.array([ab, bc, ca]))
                    )
                else:
                    triangles.append(triangle)
                    bodies.append(body)
                    triangle_shapes.append(shape)
        if not triangles:
            raise ValueError("At least one nondegenerate surface triangle is required")
        with wp.ScopedDevice(model.device):
            values = [
                cfg.density,
                cfg.internal_friction,
                cfg.surface_friction,
                -1.0 if cfg.strength is None else cfg.strength,
                cfg.inertial_coefficient,
                cfg.velocity_regularization,
                cfg.bearing,
            ]
            self.material = wp.array(np.tile(values, (len(gravity), 1)), dtype=float)
            """Mutable per-world material table, shape [world_count + 1, 7].

            Columns: density [kg/m3], internal friction, surface friction,
            strength [N/m3] (-1 uses the density/friction fit), inertial
            coefficient, velocity regularization [m/s], and bearing multiplier.
            Reset writers must retain finite values and constructor bounds.
            Update contents in place for captured rollouts; global world uses -1.
            """
            self.triangle_shape = wp.array(triangle_shapes, dtype=int)
            adjacency = [[] for _ in range(model.body_count)]
            if surface_shape_ids is not None:
                for (body, _), shape in zip(surfaces, surface_shape_ids, strict=True):
                    if shape >= 0 and shape not in adjacency[body]:
                        adjacency[body].append(int(shape))
            self.body_shape_start = wp.array(np.cumsum([0] + [len(shapes) for shapes in adjacency]), dtype=int)
            self.body_shapes = wp.array([shape for shapes in adjacency for shape in shapes], dtype=int)
            self.triangle_body = wp.array(bodies, dtype=int)
            """Body index for each triangle, shape [triangle_count]."""
            self.triangle_vertices = wp.array(np.asarray(triangles), dtype=wp.vec3)
            """Body-local triangle vertices [m], shape [triangle_count, 3]."""
            self.surface_height = wp.array(height, dtype=float)
            """Plane coordinates along the upward gravity direction [m], per world."""
            self.surface_lowering = wp.zeros(len(triangles), dtype=float)
            """Externally prescribed effective surface lowering [m], per triangle.

            Positive values lower the clipping plane and reduce pressure depth.
            This is an input closure, not a terrain evolution or wheel wake model.
            """
            self.triangle_f = wp.zeros(len(triangles), dtype=wp.spatial_vector)
            """Last triangle wrenches [N, N·m], world-space force first, torque about body COM last."""
            self.body_f = wp.zeros(model.body_count, dtype=wp.spatial_vector)
            """Last integrated granular wrenches [N, N·m], world-space force first, torque about COM last."""
            self.body_impulse = wp.zeros((model.body_count, 6), dtype=wp.float64)
            """Cumulative granular impulse [N s, N m s], for interval-averaged diagnostics.

            Sample differences across an interval and divide by its duration.
            This accumulator does not affect integration or reset rigid state.
            """
            self._external_forces = wp.empty_like(self.body_f)
        if rigid_solver is None:
            rigid_solver = SolverSemiImplicit(model, angular_damping=0.0)
        if rigid_solver.model is not model:
            raise ValueError("rigid_solver and RFT solver must use the same model")
        self.rigid_solver = rigid_solver
        self.triangle_count = len(triangles)
        """Number of surface integration triangles."""

    def evaluate(self, state: State) -> wp.array[wp.spatial_vector]:
        """Evaluate granular loads without modifying the state or integrating dynamics.

        Args:
            state: Body poses and world-space COM velocities to evaluate.

        Returns:
            Reused :attr:`body_f` array of world-space COM wrenches [N, N·m].
        """
        cfg = self.config
        self.body_f.zero_()
        wp.launch(
            integrate_surfaces,
            dim=self.triangle_count,
            inputs=[
                self.triangle_body,
                self.triangle_vertices,
                state.body_q,
                state.body_qd,
                self.model.body_com,
                self.model.body_world,
                self.model.gravity,
                self.surface_height,
                self.surface_lowering,
                wp.vec3(cfg.surface_velocity),
                self.material,
                self.triangle_shape,
                self.body_shape_start,
                self.body_shapes,
                self.model.shape_type,
                self.model.shape_scale,
                self.model.shape_transform,
            ],
            outputs=[self.triangle_f, self.body_f],
            device=self.model.device,
        )
        return self.body_f

    def apply_forces(self, state: State) -> None:
        """Add evaluated granular loads to the state's external forces.

        Args:
            state: State receiving an additive contribution in ``body_f`` [N, N·m].
                Clear forces at the appropriate point in the integration loop.
        """
        self.evaluate(state)
        wp.launch(add_forces, dim=self.model.body_count, inputs=[self.body_f, state.body_f], device=self.model.device)

    def step(
        self, state_in: State, state_out: State, control: Control | None, contacts: Contacts | None, dt: float
    ) -> None:
        """Apply granular loads and advance the configured rigid-body integrator.

        The input external-force buffer is restored afterwards, so repeated
        substeps do not accumulate stale RFT loads. Diagnostics remain available
        in :attr:`body_f` and :attr:`triangle_f`.

        Args:
            state_in: Input state and external forces.
            state_out: Output simulation state.
            control: Control passed to the rigid-body solver.
            contacts: Contacts passed to the rigid-body solver.
            dt: Positive integration interval [s].
        """
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        wp.copy(self._external_forces, state_in.body_f)
        self.apply_forces(state_in)
        wp.launch(
            accumulate_impulse,
            dim=(self.model.body_count, 6),
            inputs=[self.body_f, dt, self.body_impulse],
            device=self.model.device,
        )
        try:
            self.rigid_solver.step(state_in, state_out, control, contacts, dt)
        finally:
            wp.copy(state_in.body_f, self._external_forces)

    def notify_model_changed(self, flags: ModelFlags | int) -> None:
        """Forward property updates to the rigid integrator; surface topology is immutable.

        Args:
            flags: Changed model properties. Reconstruct this solver after changing
                body count or supplied surface geometry.
        """
        self.rigid_solver.notify_model_changed(flags)

    def reset(self, state: State, world_mask: wp.array | None = None, flags: StateFlags | int | None = None) -> None:
        """Reset the rigid solver; the granular interaction has no history state.

        Args:
            state: State to reset.
            world_mask: Optional mask of worlds forwarded to the rigid solver.
            flags: Optional reset flags forwarded to the rigid solver.
        """
        self.rigid_solver.reset(state, world_mask, flags)
