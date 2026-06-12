# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Stacked boxes example for testing contact solvers.

Run with:
    uv run -m newton.examples stacked_boxes --num-boxes 3
"""

from __future__ import annotations

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverType


class Example:
    """Stacked boxes test for contact solvers."""

    name = "Stacked Boxes"

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.num_boxes = getattr(args, "num_boxes", 3)
        self.print_contacts = getattr(args, "print_contacts", False)
        iterations = getattr(args, "iterations", 500)
        self.device = getattr(args, "device", "cuda:0")

        # Box parameters
        self.box_half = 0.1
        self.box_size = 2 * self.box_half
        self.gravity = -10.0

        # Build model
        builder = newton.ModelBuilder(gravity=self.gravity)
        builder.default_shape_cfg.mu = 0.5
        builder.default_shape_cfg.gap = 0.01
        builder.default_shape_cfg.margin = 0.002
        # Add ground plane
        builder.add_ground_plane()

        # Stack of boxes
        for i in range(self.num_boxes):
            z = self.box_half + i * self.box_size + 0.01
            body = builder.add_body(xform=wp.transform([0.0, 0.0, z], wp.quat_identity()))
            builder.add_shape_box(body, hx=self.box_half, hy=self.box_half, hz=self.box_half)

        self.model = builder.finalize(device=self.device)
        self.viewer.set_model(self.model)

        contact_config = newton.solvers.NumericalSolverConfig(
            solver_type=SolverType.SPARSE_JACOBI,
            max_iterations=iterations,
            tolerance=1e-8,
            reg=1e-4,
            relax=0.5,
            omega=0.3,
            alpha=0.1,
            recovery_speed=1.0,
            backtrack_iterations=5,
        )

        # State
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        # Run collision first to populate model.rigid_contact_max before creating solver
        self.contacts = self.model.collide(self.state_0)

        self.solver = newton.solvers.SolverDVI(
            self.model,
            contact_solver=contact_config,
            enable_timers=True,
            angular_damping=0.5,
        )

        # Simulation parameters
        self.sim_dt = 0.005
        self.sim_time = 0.0
        self.frame = 0

        # Expected positions for verification (ground plane is body 0)
        self.expected_z = [self.box_half + i * self.box_size for i in range(self.num_boxes)]

        print(f"\nStacked Boxes Test")
        print(f"  Boxes: {self.num_boxes}")
        print(f"  Expected z positions: {[f'{z:.3f}' for z in self.expected_z]}")
        print()

    def step(self):
        """Advance simulation by one frame."""
        self.state_0.clear_forces()
        self.contacts = self.model.collide(self.state_0)
        self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
        self.state_0, self.state_1 = self.state_1, self.state_0

        self.sim_time += self.sim_dt
        self.frame += 1

        # Print status every 50 frames
        if self.frame % 50 == 0:
            self._print_status()

        # Print contact data if requested
        if self.print_contacts and self.frame % 20 == 0:
            self._print_contact_data()

    def _print_status(self):
        """Print current simulation status."""
        positions = self.state_0.body_q.numpy()
        velocities = self.state_0.body_qd.numpy()

        # Boxes are bodies 0 to num_boxes-1 (ground plane doesn't add a body)
        z_vals = [positions[i][2] for i in range(self.num_boxes)]
        vz_vals = [velocities[i][2] for i in range(self.num_boxes)]
        errors = [abs(z_vals[i] - self.expected_z[i]) for i in range(self.num_boxes)]

        print(f"Frame {self.frame} (t={self.sim_time:.3f}s)")
        print(f"  z positions: {[f'{z:.4f}' for z in z_vals]}")
        print(f"  z velocities: {[f'{v:.4f}' for v in vz_vals]}")
        print(f"  position errors: {[f'{e:.4f}' for e in errors]}")

        contact_count = int(self.contacts.rigid_contact_count.numpy()[0]) if self.contacts else 0
        print(f"  active contacts: {contact_count}")
        print()

    def _print_contact_data(self):
        """Print detailed contact information."""
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
            ft_mag = np.sqrt(ft1**2 + ft2**2)
            print(f"    [{i}] bodies ({contact_body_a[i]}, {contact_body_b[i]}): fn={fn:.4f}, ft={ft_mag:.4f}")
        if contact_count > 10:
            print(f"    ... and {contact_count - 10} more contacts")
        print()

    def render(self):
        """Render the scene."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        """Verify final state after simulation."""
        positions = self.state_0.body_q.numpy()
        velocities = self.state_0.body_qd.numpy()

        # Boxes are bodies 0 to num_boxes-1 (ground plane doesn't add a body)
        z_vals = [positions[i][2] for i in range(self.num_boxes)]
        vz_vals = [velocities[i][2] for i in range(self.num_boxes)]
        errors = [abs(z_vals[i] - self.expected_z[i]) for i in range(self.num_boxes)]

        max_error = max(errors)
        max_vel = max(abs(v) for v in vz_vals)
        no_penetration = all(z > 0.05 for z in z_vals)
        no_explosion = all(z < 2.0 for z in z_vals)

        print("\n" + "=" * 60)
        print("Final Test Results")
        print("=" * 60)
        print(f"  Expected z: {[f'{z:.3f}' for z in self.expected_z]}")
        print(f"  Actual z:   {[f'{z:.3f}' for z in z_vals]}")
        print(f"  Max position error: {max_error:.4f}")
        print(f"  Max z velocity: {max_vel:.4f}")
        print(f"  No penetration: {no_penetration}")
        print(f"  No explosion: {no_explosion}")

        # Pass criteria
        position_ok = max_error < 0.05
        velocity_ok = max_vel < 0.5

        passed = position_ok and velocity_ok and no_penetration and no_explosion
        print(f"\nTest: {'PASSED' if passed else 'FAILED'}")
        print("=" * 60)

        assert passed, f"Stacked boxes test failed: max_error={max_error:.4f}, max_vel={max_vel:.4f}"


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--num-boxes", type=int, default=3, help="Number of boxes to stack")
    parser.add_argument("--iterations", type=int, default=50, help="Max contact solver iterations")
    parser.add_argument("--print-contacts", action="store_true", help="Print contact data")

    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
