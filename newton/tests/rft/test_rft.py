# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Analytic surface-load, integration, and device contracts for granular RFT."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverRFT
from newton.tests.unittest_utils import add_function_test, get_test_devices


def _model(device, *, com=(0.0, 0.0, 0.0), gravity=(0.0, 0.0, -9.81)):
    builder = newton.ModelBuilder(gravity=gravity)
    builder.add_body(mass=1.0, com=wp.vec3(com), inertia=wp.mat33(1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0))
    return builder.finalize(device=device)


def _plate(width=0.2, length=0.2):
    vertices = np.array(
        [
            [-width / 2, -length / 2, 0],
            [-width / 2, length / 2, 0],
            [width / 2, length / 2, 0],
            [width / 2, -length / 2, 0],
        ]
    )
    return newton.Mesh(vertices, np.array([0, 1, 2, 0, 2, 3]), compute_inertia=False)


def _pose(state, position=(0.0, 0.0, -0.1), velocity=(0.0, 0.0, -0.1), rotation=None, omega=(0.0, 0.0, 0.0)):
    rotation = wp.quat_identity() if rotation is None else rotation
    state.body_q.assign([wp.transform(wp.vec3(position), rotation)])
    state.body_qd.assign(np.array([np.r_[velocity, omega]], dtype=np.float32))


def test_plate_loads(test, device):
    """Recover published-fit plate loads, rate invariance, and withdrawal direction."""
    model = _model(device)
    state = model.state()
    config = SolverRFT.Config(strength=1e5)
    solver = SolverRFT(model, [(0, _plate())], config=config)
    # SI Table S3: horizontal downward intrusion has alpha_z/xi_n = 0.87574.
    for speed in (0.01, 0.1, 1.0):
        _pose(state, velocity=(0, 0, -speed))
        np.testing.assert_allclose(solver.evaluate(state).numpy()[0], [0, 0, 350.296, 0, 0, 0], rtol=2e-6, atol=2e-5)
    # SI S3 fixture at beta=60, gamma=30, psi=60 degrees probes mixed polynomial terms.
    normal = np.array([np.sqrt(3) / 4, 0.75, -0.5])
    tangent = np.array([np.sqrt(3) / 2, -0.5, 0.0])
    bitangent = np.cross(normal, tangent)
    vertices = 0.01 * np.array([tangent, -0.5 * tangent + bitangent, -0.5 * tangent - bitangent])
    oblique = newton.Mesh(vertices, [0, 1, 2], compute_inertia=False)
    oblique_solver = SolverRFT(model, [(0, oblique)], config=config)
    _pose(state, velocity=0.1 * np.array([np.sqrt(3) / 2, 0, -0.5]))
    np.testing.assert_allclose(
        oblique_solver.evaluate(state).numpy()[0, :3], [-0.48720811, -0.35326587, 0.49414319], rtol=3e-6, atol=2e-6
    )
    # Withdrawal uses the upward-facing leading side of a thin plate.
    upward = newton.Mesh(_plate().vertices, _plate().indices.reshape(-1, 3)[:, ::-1].flatten(), compute_inertia=False)
    solver = SolverRFT(model, [(0, upward)], config=config)
    _pose(state, velocity=(0, 0, 0.1))
    np.testing.assert_allclose(solver.evaluate(state).numpy()[0], [0, 0, -31.584, 0, 0, 0], rtol=3e-6, atol=2e-5)
    _pose(state, position=(0, 0, 0.1))
    test.assertEqual(float(np.linalg.norm(solver.evaluate(state).numpy())), 0.0)


def test_clipped_triangle_moment(test, device):
    """Integrate an entering triangle's exact depth-weighted force and center of pressure."""
    del test
    model = _model(device, com=(0.04, 0.01, 0.02))
    state = model.state()
    length, depth = 0.2, 0.1
    # Leading normal is -Y. Only the bottom corner is buried.
    mesh = newton.Mesh(np.array([[0, 0, -depth], [length, 0, depth], [0, 0, depth]]), [0, 1, 2], compute_inertia=False)
    config = SolverRFT.Config(strength=1e5)
    solver = SolverRFT(model, [(0, mesh)], config=config)
    _pose(state, position=(0, 0, 0), velocity=(0, -0.1, 0))
    # Analytic buried area L*h/4, mean depth h/3, center of pressure (L/8, 0, -h/2).
    force = 1e5 * length * depth**2 / 12 * np.array([0.0, 0.22945, 0.08870])
    center_pressure = np.array([length / 8, 0, -depth / 2])
    expected = np.r_[force, np.cross(center_pressure - np.array([0.04, 0.01, 0.02]), force)]
    np.testing.assert_allclose(solver.evaluate(state).numpy()[0], expected, rtol=3e-6, atol=2e-6)
    refined = SolverRFT(model, [(0, mesh)], config=config, max_edge_length=0.04)
    np.testing.assert_allclose(refined.evaluate(state).numpy()[0], expected, rtol=3e-6, atol=2e-6)


def test_dynamic_and_surface_lowering(test, device):
    """Recover quadratic momentum flux and eliminate drag in unoccupied material."""
    model = _model(device)
    state = model.state()
    config = SolverRFT.Config(strength=0.0, inertial_coefficient=1.5)
    solver = SolverRFT(model, [(0, _plate())], config=config)
    for speed in (0.1, 0.5, 1.0):
        _pose(state, velocity=(0, 0, -speed))
        expected = 1.5 * 1600 * 0.04 * speed**2
        test.assertAlmostEqual(float(solver.evaluate(state).numpy()[0, 2]), expected, delta=expected * 2e-6)
    solver.surface_lowering.fill_(0.11)
    test.assertEqual(float(np.linalg.norm(solver.evaluate(state).numpy())), 0.0)
    solver = SolverRFT(model, [(0, _plate())], config=SolverRFT.Config(strength=1e5))
    _pose(state)
    force = solver.evaluate(state).numpy()[0, 2]
    solver.surface_lowering.fill_(0.04)
    test.assertAlmostEqual(float(solver.evaluate(state).numpy()[0, 2] / force), 0.6, delta=2e-6)


def test_finite_layer(test, device):
    """Integrate a triangle crossing both planes and reject material below the substrate."""
    model = _model(device)
    state = model.state()
    width, depth = 0.2, 0.1
    mesh = newton.Mesh(
        np.array([[-width / 2, 0, -2 * depth], [width / 2, 0, -2 * depth], [0, 0, depth]]),
        [0, 1, 2],
        compute_inertia=False,
    )
    solver = SolverRFT(model, [(0, mesh)], surface_bottom=-depth, config=SolverRFT.Config(strength=1e5))
    _pose(state, position=(0, 0, 0), velocity=(0, -0.1, 0))
    # Width at z is W*(h-z)/(3*h). Integrating depth times width gives 5*W*h²/18,
    # with pressure centroid z=-7*h/10. SI Table S3 supplies the traction direction.
    force = 1e5 * 5 * width * depth**2 / 18 * np.array([0, 0.22945, 0.08870])
    expected = np.r_[force, np.cross([0, 0, -0.7 * depth], force)]
    np.testing.assert_allclose(solver.evaluate(state).numpy()[0], expected, rtol=4e-6, atol=3e-6)
    graph = None
    if wp.get_device(device).is_cuda:
        with wp.ScopedCapture(device=device) as capture:
            solver.evaluate(state)
        graph = capture.graph
    solver.surface_bottom.fill_(0.0)
    if graph is not None:
        wp.capture_launch(graph)
        np.testing.assert_array_equal(solver.body_f.numpy(), 0)
    test.assertEqual(float(np.linalg.norm(solver.evaluate(state).numpy())), 0.0)
    solver.surface_bottom.fill_(-depth)
    if graph is not None:
        wp.capture_launch(graph)
        np.testing.assert_allclose(solver.body_f.numpy()[0], expected, rtol=4e-6, atol=3e-6)
    _pose(state, position=(0, 0, -0.5), velocity=(0, -0.1, 0))
    test.assertEqual(float(np.linalg.norm(solver.evaluate(state).numpy())), 0.0)


def test_covariance(test, device):
    """Rotate the complete system without changing its physical response."""
    del test
    mesh = newton.Mesh.create_box(0.08, 0.06, 0.04, compute_inertia=False)
    model = _model(device, com=(0.02, -0.01, 0.015))
    state = model.state()
    _pose(state, position=(0.01, 0.03, -0.12), velocity=(0.12, -0.06, -0.03), omega=(0.2, 0.1, -0.3))
    solver = SolverRFT(model, [(0, mesh)], max_edge_length=0.04)
    wrench = solver.evaluate(state).numpy()[0]
    # A fixed proper rotation transforms body, plane, gravity, velocity, and COM.
    rotation = wp.quat_from_axis_angle(wp.normalize(wp.vec3(0.3, 0.7, 0.2)), 1.1)
    matrix = np.array(wp.quat_to_matrix(rotation)).reshape(3, 3)
    rotated = _model(device, com=(0.02, -0.01, 0.015), gravity=matrix @ np.array([0, 0, -9.81]))
    rotated_state = rotated.state()
    _pose(
        rotated_state,
        position=matrix @ np.array([0.01, 0.03, -0.12]),
        velocity=matrix @ np.array([0.12, -0.06, -0.03]),
        rotation=rotation,
        omega=matrix @ np.array([0.2, 0.1, -0.3]),
    )
    rotated_solver = SolverRFT(rotated, [(0, mesh)], max_edge_length=0.04)
    np.testing.assert_allclose(
        rotated_solver.evaluate(rotated_state).numpy()[0],
        np.r_[matrix @ wrench[:3], matrix @ wrench[3:]],
        rtol=2e-5,
        atol=2e-5,
    )
    np.testing.assert_allclose(solver.triangle_f.numpy().sum(axis=0), wrench, rtol=3e-6, atol=1e-5)


def test_step_and_graph(test, device):
    """Compose forces without stale accumulation and support graph replay on CUDA."""
    model = _model(device)
    state, output = model.state(), model.state()
    _pose(state)
    external = np.array([[2, 0, -3, 0, 0, 0]], dtype=np.float32)
    state.body_f.assign(external)
    solver = SolverRFT(model, [(0, _plate())], config=SolverRFT.Config(strength=1e5))
    solver.step(state, output, None, None, 1e-4)
    np.testing.assert_array_equal(state.body_f.numpy(), external)
    expected_vz = -0.1 + 1e-4 * (350.296 - 3 - 9.81)
    test.assertAlmostEqual(float(output.body_qd.numpy()[0, 2]), expected_vz, delta=2e-7)
    if wp.get_device(device).is_cuda:
        expected = solver.evaluate(state).numpy()
        with wp.ScopedCapture(device=device) as capture:
            solver.evaluate(state)
        wp.capture_launch(capture.graph)
        np.testing.assert_array_equal(solver.body_f.numpy(), expected)
    _pose(state, velocity=(0, 0, 0))
    test.assertEqual(float(np.linalg.norm(solver.evaluate(state).numpy())), 0.0)


def test_world_planes(test, device):
    """Keep independent bed planes and gravity scales across replicated worlds."""
    child = newton.ModelBuilder()
    child.add_body(mass=1.0, inertia=wp.mat33(1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0))
    builder = newton.ModelBuilder()
    builder.add_world(child)
    builder.add_world(child)
    model = builder.finalize(device=device)
    state = model.state()
    state.body_q.assign(np.array([[0, 0, -0.1, 0, 0, 0, 1], [0, 0, 0.9, 0, 0, 0, 1]], dtype=np.float32))
    state.body_qd.assign(np.tile(np.array([0, 0, -0.1, 0, 0, 0], dtype=np.float32), (2, 1)))
    height = np.zeros(model.gravity.shape[0], dtype=np.float32)
    worlds = model.body_world.numpy()
    height[worlds[1]] = 1.0
    mesh = _plate()
    solver = SolverRFT(model, [(0, mesh), (1, mesh)], surface_height=height)
    forces = solver.evaluate(state).numpy()
    np.testing.assert_allclose(forces[0], forces[1], rtol=2e-6, atol=1e-5)
    state.body_f.zero_()
    solver.apply_forces(state)
    np.testing.assert_array_equal(state.body_f.numpy(), forces)
    solver.apply_forces(state)
    np.testing.assert_allclose(state.body_f.numpy(), 2 * forces, rtol=2e-6, atol=1e-5)
    test.assertEqual(solver.triangle_count, 4)


def test_material_capture(test, device):
    """Changing one persistent material row changes only its captured world load."""
    child = newton.ModelBuilder()
    child.add_body(mass=1.0, inertia=wp.mat33(1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0))
    builder = newton.ModelBuilder()
    builder.add_world(child)
    builder.add_world(child)
    model = builder.finalize(device=device)
    state = model.state()
    state.body_q.assign(np.array([[0, 0, -0.1, 0, 0, 0, 1]] * 2, dtype=np.float32))
    state.body_qd.assign(np.array([[0, 0, -0.1, 0, 0, 0]] * 2, dtype=np.float32))
    solver = SolverRFT(model, [(0, _plate()), (1, _plate())], config=SolverRFT.Config(strength=1e5))
    baseline = solver.evaluate(state).numpy().copy()
    graph = None
    if device.is_cuda:
        with wp.ScopedCapture(device=device) as capture:
            solver.evaluate(state)
        graph = capture.graph
    values = solver.material.numpy()
    values[model.body_world.numpy()[1], 3] = 2e5
    solver.material.assign(values)
    if graph is None:
        solver.evaluate(state)
    else:
        wp.capture_launch(graph)
    result = solver.body_f.numpy()
    np.testing.assert_allclose(result[0], baseline[0], rtol=2e-6, atol=1e-5)
    np.testing.assert_allclose(result[1], 2 * baseline[1], rtol=2e-6, atol=1e-5)


def test_bearing_and_internal_patches(test, device):
    """A nested collider contributes no internal pressure to a resting box."""
    builder = newton.ModelBuilder()
    body = builder.add_body(mass=1.0, inertia=wp.mat33(1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0))
    outer = builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    inner = builder.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05)
    model = builder.finalize(device=device)
    state = model.state()
    _pose(state, position=(0, 0, -0.3), velocity=(0, 0, 0))
    surfaces = [(body, newton.Mesh.create_box(h, h, h, compute_inertia=False)) for h in (0.1, 0.05)]
    solver = SolverRFT(
        model, surfaces, surface_shape_ids=[outer, inner], config=SolverRFT.Config(strength=1e5, bearing=1.0)
    )
    # Bottom-face area .2*.2, depth .4: static pressure = strength*depth.
    np.testing.assert_allclose(solver.evaluate(state).numpy()[body], [0, 0, 1600, 0, 0, 0], rtol=2e-6, atol=1e-4)
    np.testing.assert_allclose(solver.triangle_f.numpy()[12:], 0, atol=1e-5)
    _pose(state, position=(0, 0, 0.5), velocity=(0, 0, 0))
    np.testing.assert_array_equal(solver.evaluate(state).numpy(), 0)


class TestRFT(unittest.TestCase):
    """Validate the reduced-order granular interaction's numerical contracts."""

    def test_invalid_inputs(self):
        """Reject unsupported materials and invalid integration geometry."""
        model = _model("cpu")
        for config in (
            SolverRFT.Config(density=-1),
            SolverRFT.Config(internal_friction=1),
            SolverRFT.Config(surface_friction=0.5),
            SolverRFT.Config(strength=float("nan")),
        ):
            with self.assertRaises(ValueError):
                SolverRFT(model, [(0, _plate())], config=config)
        with self.assertRaises(ValueError):
            SolverRFT(model, [(1, _plate())])
        with self.assertRaises(ValueError):
            SolverRFT(model, [(0, _plate())], max_edge_length=0)
        with self.assertRaises(ValueError):
            SolverRFT(model, [])


for _device in get_test_devices():
    for _function in (
        test_plate_loads,
        test_clipped_triangle_moment,
        test_dynamic_and_surface_lowering,
        test_covariance,
        test_step_and_graph,
        test_world_planes,
        test_material_capture,
        test_bearing_and_internal_patches,
        test_finite_layer,
    ):
        add_function_test(TestRFT, _function.__name__, _function, devices=[_device])


if __name__ == "__main__":
    unittest.main(verbosity=2)
