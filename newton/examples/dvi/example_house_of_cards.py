# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
House-of-cards example for stress-testing contact solvers.

Builds a pyramid-shaped house of cards from equilateral-triangle tents.
Each tent is two cards leaning at 30 degrees from vertical (60 degrees
between them), with a horizontal floor card bridging adjacent tent apexes.
The base row has N tents; each higher story has one fewer tent, giving
N stories total.

Run with:
    uv run -m newton.examples house_of_cards --num-base-tents 4

Contact solver: sparse-jacobi
"""

from __future__ import annotations

import math

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverType


class Example:
    """Pyramid house-of-cards test for contact solvers."""

    name = "House of Cards"

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.num_base_tents = max(2, getattr(args, "num_base_tents", 4))
        self.num_stories = self.num_base_tents
        self.friction = getattr(args, "friction", 0.6)
        self.print_contacts = getattr(args, "print_contacts", False)
        iterations = getattr(args, "iterations", 200)
        self.device = getattr(args, "device", "cuda:0")

        # Card dimensions
        self.card_length = 0.5
        self.card_width = 0.30
        self.card_thickness = 0.005
        self.gravity = -10.0

        # Equilateral triangle geometry: 30 deg tilt from vertical.
        # Two cards + base form an equilateral triangle with side = card_length.
        tilt_rad = math.radians(30.0)
        self.tilt_rad = tilt_rad
        self.tent_height = self.card_length * math.cos(tilt_rad)  # L * sqrt(3)/2
        self.tent_half_base = self.card_length * math.sin(tilt_rad)  # L / 2
        # Tent base width = card_length (equilateral). Adjacent tents share
        # a foot position, so tent-to-tent spacing = tent base = card_length.
        self.tent_spacing = 2.0 * self.tent_half_base  # = card_length

        builder = newton.ModelBuilder(gravity=self.gravity)
        builder.default_shape_cfg.mu = self.friction
        builder.default_shape_cfg.gap = 0.001
        builder.default_shape_cfg.margin = 0.0005
        builder.add_ground_plane()

        self._build_house(builder)

        self.model = builder.finalize(device=self.device)
        self.viewer.set_model(self.model)

        contact_config = newton.solvers.NumericalSolverConfig(
            solver_type=SolverType.SPARSE_JACOBI,
            max_iterations=iterations,
            tolerance=1e-8,
            reg=1e-4,
            relax=0.5,
            omega=0.3,
            alpha=0.01,
            recovery_speed=0.01,
            backtrack_iterations=5,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.contacts = self.model.collide(self.state_0)

        self.solver = newton.solvers.SolverDVI(
            self.model,
            contact_solver=contact_config,
            enable_timers=True,
            angular_damping=0.05,
        )

        self.sim_dt = 0.005
        self.sim_time = 0.0
        self.frame = 0
        self.total_height = self.num_stories * (self.tent_height + self.card_thickness)

        # Position camera along +Y looking toward origin to see the A-frame profile.
        cam_distance = max(2.0, self.num_base_tents * self.tent_spacing)
        cam_z = 0.5 * self.total_height
        self.viewer.set_camera(wp.vec3(0.0, cam_distance, cam_z), 0.0, -90.0)

        mu_required = math.tan(tilt_rad)

        print("\nHouse of Cards Test (Pyramid)")
        print(f"  Solver: sparse-jacobi")
        print(f"  Base tents: {self.num_base_tents}")
        print(f"  Stories: {self.num_stories}")
        print(f"  Total cards: {self.num_cards}")
        print(f"  Tilt angle: 30.0 deg (equilateral, mu_required >= {mu_required:.3f})")
        print(f"  Friction (mu): {self.friction:.3f}")
        print(f"  Total height: {self.total_height:.4f} m")
        if self.friction < mu_required:
            print("  WARNING: friction below static-friction requirement; tents will slip.")
        print()

    def _build_house(self, builder) -> None:
        """Build a pyramid of equilateral-triangle tents with floor cards.

        Each tent: two cards tilted +-30 deg about Y, meeting at the apex.
        Floor cards lie flat at apex height, bridging adjacent tents.
        The geometry tiles perfectly with no gaps.
        """
        tilt_rad = self.tilt_rad
        rot_left = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), tilt_rad)
        rot_right = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), -tilt_rad)

        # Tent card box: thin along X, wide along Y, tall along Z.
        tent_hx = 0.5 * self.card_thickness
        tent_hy = 0.5 * self.card_width
        tent_hz = 0.5 * self.card_length

        # Floor card box: 10% longer than tent spacing to ensure stable contact.
        floor_hx = 0.5 * self.card_length * 1.1
        floor_hy = 0.5 * self.card_width
        floor_hz = 0.5 * self.card_thickness

        self.num_cards = 0
        for s in range(self.num_stories):
            tents_in_story = self.num_base_tents - s

            # Foot z: story 0 rests on ground, others on floor cards below.
            if s == 0:
                foot_z = 0.0
            else:
                foot_z = story_floor_top_z  # noqa: F821

            # Card center is at the geometric center of the tilted card.
            card_center_z = foot_z + 0.5 * self.tent_height

            # Center row about X = 0.
            row_span = (tents_in_story - 1) * self.tent_spacing
            x0 = -0.5 * row_span

            for t in range(tents_in_story):
                tent_x = x0 + t * self.tent_spacing

                # Left card leans toward +X: foot at tent_x - tent_half_base,
                # apex at tent_x. Center at midpoint of the tilted card.
                left_cx = tent_x - 0.5 * self.tent_half_base
                right_cx = tent_x + 0.5 * self.tent_half_base

                left_body = builder.add_body(
                    xform=wp.transform(wp.vec3(left_cx, 0.0, card_center_z), rot_left),
                    label=f"card_s{s}_t{t}_left",
                )
                builder.add_shape_box(left_body, hx=tent_hx, hy=tent_hy, hz=tent_hz)
                self.num_cards += 1

                right_body = builder.add_body(
                    xform=wp.transform(wp.vec3(right_cx, 0.0, card_center_z), rot_right),
                    label=f"card_s{s}_t{t}_right",
                )
                builder.add_shape_box(right_body, hx=tent_hx, hy=tent_hy, hz=tent_hz)
                self.num_cards += 1

            # Floor cards at apex height, bridging adjacent tents.
            if s < self.num_stories - 1:
                apex_z = foot_z + self.tent_height
                floor_center_z = apex_z + floor_hz

                for f in range(tents_in_story - 1):
                    left_tent_x = x0 + f * self.tent_spacing
                    right_tent_x = x0 + (f + 1) * self.tent_spacing
                    floor_x = 0.5 * (left_tent_x + right_tent_x)

                    floor_body = builder.add_body(
                        xform=wp.transform(
                            wp.vec3(floor_x, 0.0, floor_center_z), wp.quat_identity()
                        ),
                        label=f"card_s{s}_f{f}",
                    )
                    builder.add_shape_box(floor_body, hx=floor_hx, hy=floor_hy, hz=floor_hz)
                    self.num_cards += 1

                story_floor_top_z = floor_center_z + floor_hz

    def step(self):
        self.state_0.clear_forces()
        self.contacts = self.model.collide(self.state_0)
        self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
        self.state_0, self.state_1 = self.state_1, self.state_0

        self.sim_time += self.sim_dt
        self.frame += 1

        # if self.frame % 50 == 0:
        #     self._print_status()

        if self.print_contacts and self.frame % 20 == 0:
            self._print_contact_data()

    def _print_status(self):
        positions = self.state_0.body_q.numpy()
        velocities = self.state_0.body_qd.numpy()

        z_vals = positions[: self.num_cards, 2]
        lin_speed = np.linalg.norm(velocities[: self.num_cards, 3:6], axis=1)
        ang_speed = np.linalg.norm(velocities[: self.num_cards, 0:3], axis=1)

        contact_count = int(self.contacts.rigid_contact_count.numpy()[0]) if self.contacts else 0

        print(f"Frame {self.frame} (t={self.sim_time:.3f}s)")
        print(f"  z range: [{z_vals.min():.4f}, {z_vals.max():.4f}]")
        print(f"  max linear speed: {lin_speed.max():.4f} m/s")
        print(f"  max angular speed: {ang_speed.max():.4f} rad/s")
        print(f"  active contacts: {contact_count}")
        print()

    def _print_contact_data(self):
        if self.contacts is None:
            return

        contact_count = int(self.contacts.rigid_contact_count.numpy()[0])
        if contact_count == 0:
            print("  No active contacts")
            return

        contact_lambda = self.solver.contact_lambda.numpy()[:contact_count * 3]
        contact_body_a = self.solver.contact_body_a.numpy()[:contact_count]
        contact_body_b = self.solver.contact_body_b.numpy()[:contact_count]

        print(f"  Contact details ({contact_count} contacts):")
        for i in range(min(contact_count, 10)):
            fn = contact_lambda[i * 3]
            ft1 = contact_lambda[i * 3 + 1]
            ft2 = contact_lambda[i * 3 + 2]
            ft_mag = math.sqrt(ft1 * ft1 + ft2 * ft2)
            print(f"    [{i}] bodies ({contact_body_a[i]}, {contact_body_b[i]}): fn={fn:.4f}, ft={ft_mag:.4f}")
        if contact_count > 10:
            print(f"    ... and {contact_count - 10} more contacts")
        print()

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        """Sanity-check the final state.

        A house of cards is a deliberately fragile structure used to stress
        the contact solver. Multi-story configurations may collapse during the
        run, which is expected and informative. This check only catches
        outright solver failure (NaN, deep penetration, or unbounded explosion)
        rather than asserting the structure is still standing.
        """
        positions = self.state_0.body_q.numpy()
        velocities = self.state_0.body_qd.numpy()

        z_vals = positions[: self.num_cards, 2]
        lin_speed = np.linalg.norm(velocities[: self.num_cards, 3:6], axis=1)
        ang_speed = np.linalg.norm(velocities[: self.num_cards, 0:3], axis=1)

        no_nan = bool(np.all(np.isfinite(positions)) and np.all(np.isfinite(velocities)))
        no_penetration = bool(z_vals.min() > -0.05)
        no_explosion = bool(z_vals.max() < 5.0 * self.total_height + 1.0)
        max_lin = float(lin_speed.max())
        max_ang = float(ang_speed.max())

        gravity_speed_bound = abs(self.gravity) * self.sim_time + 5.0
        velocity_ok = max_lin < 5.0 * gravity_speed_bound and max_ang < 50.0

        print("\n" + "=" * 60)
        print("Final Test Results")
        print("=" * 60)
        print(f"  z range: [{z_vals.min():.4f}, {z_vals.max():.4f}]")
        print(f"  Max linear speed: {max_lin:.4f} m/s")
        print(f"  Max angular speed: {max_ang:.4f} rad/s")
        print(f"  No NaN: {no_nan}")
        print(f"  No penetration: {no_penetration}")
        print(f"  No explosion: {no_explosion}")
        print(f"  Velocity bounded: {velocity_ok}")

        passed = no_nan and no_penetration and no_explosion and velocity_ok
        print(f"\nTest: {'PASSED' if passed else 'FAILED'}")
        print("=" * 60)

        assert passed, (
            f"House of cards solver failure: nan={not no_nan}, "
            f"max_lin={max_lin:.4f}, max_ang={max_ang:.4f}, "
            f"z_min={z_vals.min():.4f}, z_max={z_vals.max():.4f}"
        )


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--num-base-tents", type=int, default=4, help="Tents in base row (>=2)")
    parser.add_argument("--friction", type=float, default=0.6, help="Coulomb friction coefficient")
    parser.add_argument("--iterations", type=int, default=200, help="Max contact solver iterations")
    parser.add_argument("--print-contacts", action="store_true", help="Print contact data")

    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
