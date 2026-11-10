# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run reproducible RFT verification and convergence investigations.

uv run python -m scripts.benchmarks.benchmark_rft --device cuda:0 --output newton/tests/outputs/rft

Analytic or manufactured references verify numerics, not physical sand accuracy.
Independent Go2/MPM comparisons are provided by the companion Isaac Lab integration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverRFT


def make_model(device, count=1):
    """Build isolated rigid bodies without collision or contact forces."""
    builder = newton.ModelBuilder()
    for _ in range(count):
        builder.add_body(mass=1.0, inertia=wp.mat33(1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0))
    return builder.finalize(device=device)


def plate(width=0.05):
    """Create an outward downward square surface."""
    v = np.array([[-1, -1, 0], [-1, 1, 0], [1, 1, 0], [1, -1, 0]]) * width / 2
    return newton.Mesh(v, [0, 1, 2, 0, 2, 3], compute_inertia=False)


def set_state(state, *, position=(0, 0, -0.1), velocity=(0, 0, -0.1), omega=(0, 0, 0), rotation=None):
    """Assign a single-body trajectory sample."""
    state.body_q.assign([wp.transform(wp.vec3(position), wp.quat_identity() if rotation is None else rotation)])
    state.body_qd.assign(np.r_[velocity, omega][None].astype(np.float32))


def relative_error(value, reference):
    """Measure norm error with an explicit near-zero reference floor."""
    return float(np.linalg.norm(np.asarray(value) - reference) / max(np.linalg.norm(reference), 1e-12))


def plate_sweeps(device):
    """Compare depth, rate, and inertial branches with closed-form plate references."""
    model = make_model(device)
    state = model.state()
    mesh = plate()
    solver = SolverRFT(model, [(0, mesh)], config=SolverRFT.Config(strength=1e5))
    rows = []
    for depth in (-0.02, 0, 0.005, 0.01, 0.02, 0.05, 0.1):
        for speed in (0.001, 0.01, 0.1, 1.0):
            set_state(state, position=(0, 0, -depth), velocity=(0, 0, -speed))
            force = float(solver.evaluate(state).numpy()[0, 2])
            expected = 1e5 * 0.87574 * 0.05**2 * max(depth, 0)
            rows.append(
                {
                    "depth": depth,
                    "speed": speed,
                    "force": force,
                    "analytic_force": expected,
                    "absolute_error": abs(force - expected),
                }
            )
    dynamic = SolverRFT(model, [(0, mesh)], config=SolverRFT.Config(strength=0, inertial_coefficient=1.0))
    dynamic_rows = []
    for speed in (0.01, 0.03, 0.1, 0.3, 1.0):
        set_state(state, velocity=(0, 0, -speed))
        force = float(dynamic.evaluate(state).numpy()[0, 2])
        expected = 1600 * 0.05**2 * speed**2
        dynamic_rows.append({"speed": speed, "force": force, "analytic_force": expected})
    return {
        "quasistatic": rows,
        "inertial": dynamic_rows,
        "max_absolute_error_N": max(row["absolute_error"] for row in rows),
    }


def orientation_sweep(device, count=240):
    """Check compression, Coulomb bounds, and rate invariance across spatial orientations."""
    rng = np.random.default_rng(20261109)
    normals = rng.normal(size=(count, 3))
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    velocities = rng.normal(size=(count, 3))
    velocities /= np.linalg.norm(velocities, axis=1)[:, None]
    velocities *= np.where(np.sum(normals * velocities, axis=1) >= 0, 1, -1)[:, None] * 0.05
    model = make_model(device, count)
    state = model.state()
    surfaces = []
    for i, normal in enumerate(normals):
        tangent = np.cross(normal, [1, 0, 0])
        tangent /= np.linalg.norm(tangent)
        bitangent = np.cross(normal, tangent)
        vertices = 0.01 * np.array([tangent, -0.5 * tangent + bitangent, -0.5 * tangent - bitangent])
        surfaces.append((i, newton.Mesh(vertices, [0, 1, 2], compute_inertia=False)))
    state.body_q.assign(np.tile(np.array([0, 0, -0.1, 0, 0, 0, 1], dtype=np.float32), (count, 1)))
    state.body_qd.assign(np.c_[velocities, np.zeros((count, 3))].astype(np.float32))
    solver = SolverRFT(model, surfaces, config=SolverRFT.Config(strength=1e5, surface_friction=0.15))
    wrench = solver.evaluate(state).numpy()
    normal_force = -np.sum(wrench[:, :3] * normals, axis=1)
    tangent_force = np.linalg.norm(wrench[:, :3] + normal_force[:, None] * normals, axis=1)
    ratios = tangent_force / np.maximum(normal_force, 1e-12)
    state.body_qd.assign(np.c_[velocities * 10, np.zeros((count, 3))].astype(np.float32))
    faster = solver.evaluate(state).numpy()
    return {
        "case_count": count,
        "minimum_compressive_force_N": float(normal_force.min()),
        "maximum_friction_ratio": float(ratios.max()),
        "surface_friction": 0.15,
        "rate_scaling_relative_error": relative_error(faster, wrench),
        "wrenches": wrench.tolist(),
    }


def similarity(device):
    """Check force and torque scaling under geometric similarity."""
    rows = []
    reference = None
    for scale in (1.0, 0.5, 2.0):
        model = make_model(device)
        state = model.state()
        mesh = newton.Mesh.create_box(0.04 * scale, 0.03 * scale, 0.02 * scale, compute_inertia=False)
        set_state(state, position=(0, 0, -0.08 * scale), velocity=(0.04, 0.02, -0.03))
        solver = SolverRFT(model, [(0, mesh)], max_edge_length=0.02 * scale)
        wrench = solver.evaluate(state).numpy()[0]
        if reference is None:
            reference = wrench
        expected = np.r_[reference[:3] * scale**3, reference[3:] * scale**4]
        rows.append(
            {
                "scale": scale,
                "wrench": wrench.tolist(),
                "force_error": relative_error(wrench[:3], expected[:3]),
                "torque_error": relative_error(wrench[3:], expected[3:]),
            }
        )
    return rows


def drilling_convergence(device):
    """Refine a rotating, translating submerged sphere and compare resultant wrenches."""
    rows = []
    for resolution in (8, 16, 32, 64, 128):
        model = make_model(device)
        state = model.state()
        mesh = newton.Mesh.create_sphere(
            0.04, num_latitudes=resolution, num_longitudes=2 * resolution, compute_inertia=False
        )
        solver = SolverRFT(model, [(0, mesh)])
        set_state(state, position=(0, 0, -0.08), velocity=(0.02, 0.005, -0.02), omega=(0, 0, 4))
        wrench = solver.evaluate(state).numpy()[0]
        rows.append({"resolution": resolution, "triangle_count": solver.triangle_count, "wrench": wrench.tolist()})
    reference = np.array(rows[-1]["wrench"])
    for row in rows:
        row["force_error"] = relative_error(np.array(row["wrench"][:3]), reference[:3])
        row["torque_error"] = relative_error(np.array(row["wrench"][3:]), reference[3:])
    return rows


def wheel_probe(device):
    """Record slip-dependent wheel wrenches without asserting experimental validation."""
    model = make_model(device)
    state = model.state()
    mesh = newton.Mesh.create_cylinder(0.05, 0.025, up_axis=newton.Axis.Y, segments=128, compute_inertia=False)
    solver = SolverRFT(model, [(0, mesh)], max_edge_length=0.01)
    rows = []
    for slip in (0, 0.2, 0.5, 0.8, 1.0):
        velocity = 0.05 * (1 - slip)
        set_state(state, position=(0, 0, 0.03), velocity=(velocity, 0, 0), omega=(0, 1, 0))
        wrench = solver.evaluate(state).numpy()[0]
        set_state(state, position=(0, 0, 0.03), velocity=(velocity * 10, 0, 0), omega=(0, 10, 0))
        faster = solver.evaluate(state).numpy()[0]
        rows.append({"slip": slip, "wrench": wrench.tolist(), "rate_error": relative_error(faster, wrench)})
    return {"status": "kinematic probe; no independent wheel reference supplied", "rows": rows}


def time_convergence(device):
    """Compare regularized plate penetration with an independent scalar RK4 reference."""
    width = 0.05
    area = width**2
    strength = 1e5
    epsilon = 0.02
    duration = 0.08
    coefficient = area * strength * 0.87574

    def rhs(y):
        z, v = y
        # Single downward-facing patch: only downward motion activates resistance.
        force = coefficient * max(-z, 0) * (-v) / np.sqrt(v * v + epsilon * epsilon) if v < 0 else 0.0
        return np.array([v, force - 9.81])

    reference = np.array([-0.1, -0.05])
    h = 1e-6
    for _ in range(round(duration / h)):
        k1 = rhs(reference)
        k2 = rhs(reference + h * k1 / 2)
        k3 = rhs(reference + h * k2 / 2)
        k4 = rhs(reference + h * k3)
        reference += h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    rows = []
    for dt in (1e-3, 5e-4, 2.5e-4):
        model = make_model(device)
        state, output = model.state(), model.state()
        set_state(state, velocity=(0, 0, -0.05))
        solver = SolverRFT(
            model, [(0, plate(width))], config=SolverRFT.Config(strength=strength, velocity_regularization=epsilon)
        )
        # A graph captures fixed batches; external forcing is zero and restored per step.
        if wp.get_device(device).is_cuda:
            solver.step(state, output, None, None, dt)
            set_state(state, velocity=(0, 0, -0.05))
            with wp.ScopedCapture(device=device) as capture:
                solver.step(state, output, None, None, dt)
                solver.step(output, state, None, None, dt)
            for _ in range(round(duration / dt) // 2):
                wp.capture_launch(capture.graph)
        else:
            for _ in range(round(duration / dt)):
                solver.step(state, output, None, None, dt)
                state, output = output, state
        z = float(state.body_q.numpy()[0, 2])
        v = float(state.body_qd.numpy()[0, 2])
        rows.append(
            {
                "dt": dt,
                "z": z,
                "v": v,
                "position_error_m": abs(z - reference[0]),
                "velocity_error_m_s": abs(v - reference[1]),
            }
        )
    return {
        "duration_s": duration,
        "regularization_m_s": epsilon,
        "scalar_rk4_reference": reference.tolist(),
        "rows": rows,
        "limitation": "Finite velocity regularization admits creep; this is not an exact static-support test.",
    }


def work_cycles(device):
    """Measure net work over closed prescribed cycles in the fixed-bed approximation."""
    rows = []
    for moving_rotation in (False, True):
        model = make_model(device)
        state = model.state()
        mesh = newton.Mesh.create_box(0.04, 0.03, 0.02, compute_inertia=False)
        solver = SolverRFT(model, [(0, mesh)], max_edge_length=0.02)
        work = 0.0
        samples = 256
        for i in range(samples):
            phase = 2 * np.pi * i / samples
            position = (0.02 * np.cos(phase), 0, -0.12 + 0.02 * np.sin(phase))
            velocity = (-0.02 * np.sin(phase), 0, 0.02 * np.cos(phase))
            angle = 0.3 * np.sin(phase) if moving_rotation else 0.0
            omega = (0, 0.3 * np.cos(phase), 0) if moving_rotation else (0, 0, 0)
            set_state(
                state,
                position=position,
                velocity=velocity,
                omega=omega,
                rotation=wp.quat_from_axis_angle(wp.vec3(0, 1, 0), float(angle)),
            )
            wrench = solver.evaluate(state).numpy()[0]
            work += float(np.dot(wrench[:3], velocity) + np.dot(wrench[3:], omega)) * 2 * np.pi / samples
        rows.append({"rotation": moving_rotation, "net_work_J": work, "samples": samples})
    return rows


def throughput(device):
    """Measure steady-state evaluation cost excluding model and kernel construction."""
    rows = []
    sphere = newton.Mesh.create_sphere(0.04, num_latitudes=32, num_longitudes=64, compute_inertia=False)
    cases = [
        (1, plate(), "plate"),
        (64, plate(), "plate"),
        (512, plate(), "plate"),
        (1, sphere, "sphere"),
        (32, sphere, "sphere"),
    ]
    for count, mesh, geometry in cases:
        model = make_model(device, count)
        state = model.state()
        state.body_q.assign(np.tile(np.array([0, 0, -0.1, 0, 0, 0, 1], dtype=np.float32), (count, 1)))
        state.body_qd.assign(np.tile(np.array([0, 0, -0.1, 0, 0, 0], dtype=np.float32), (count, 1)))
        solver = SolverRFT(model, [(i, mesh) for i in range(count)])
        solver.evaluate(state)
        if wp.get_device(device).is_cuda:
            with wp.ScopedCapture(device=device) as capture:
                solver.evaluate(state)

            def evaluate(graph=capture.graph):
                wp.capture_launch(graph)
        else:

            def evaluate(solver=solver, state=state):
                solver.evaluate(state)

        wp.synchronize_device(device)
        start = time.perf_counter()
        repetitions = 50 if solver.triangle_count > 10000 else 200
        for _ in range(repetitions):
            evaluate()
        wp.synchronize_device(device)
        elapsed = (time.perf_counter() - start) / repetitions
        rows.append(
            {
                "body_count": count,
                "geometry": geometry,
                "repetitions": repetitions,
                "triangle_count": solver.triangle_count,
                "evaluation_seconds": elapsed,
                "body_evaluations_per_second": count / elapsed,
                "cuda_graph": wp.get_device(device).is_cuda,
            }
        )
    return rows


def run(device):
    """Run each numerical investigation and report explicit acceptance gates."""
    result = {
        "device": str(device),
        "warp": wp.__version__,
        "numpy": np.__version__,
        "python": platform.python_version(),
        "reference": "Agarwal et al. 2023, SI Table S3 and Fig S4; analytic geometry and scalar RK4",
        "physical_validation": "not established by analytic/convergence checks",
        "source_sha256": {
            name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
            for name in ("newton/_src/solvers/rft/kernels.py", "newton/_src/solvers/rft/solver_rft.py")
        },
    }
    for name, function in [
        ("plate", plate_sweeps),
        ("orientations", orientation_sweep),
        ("similarity", similarity),
        ("drilling", drilling_convergence),
        ("wheel", wheel_probe),
        ("time", time_convergence),
        ("work_cycles", work_cycles),
        ("throughput", throughput),
    ]:
        print(f"Running {name} on {device}", flush=True)
        result[name] = function(device)
    result["gates"] = {
        "plate_analytic": result["plate"]["max_absolute_error_N"] < 1e-4,
        "closed_cycle_work": max(row["net_work_J"] for row in result["work_cycles"]) <= 1e-6,
        "compression": result["orientations"]["minimum_compressive_force_N"] >= -1e-7,
        "friction": result["orientations"]["maximum_friction_ratio"] <= 0.15001,
        "rate_independence": result["orientations"]["rate_scaling_relative_error"] < 1e-5,
        "similarity": max(row["force_error"] + row["torque_error"] for row in result["similarity"]) < 1e-4,
        "drilling_mesh": result["drilling"][-2]["force_error"] < 0.01 and result["drilling"][-2]["torque_error"] < 0.03,
        "timestep": result["time"]["rows"][-1]["position_error_m"] < 2e-5
        and result["time"]["rows"][-1]["velocity_error_m_s"] < 1e-4,
    }
    result["gates"] = {name: bool(value) for name, value in result["gates"].items()}
    return result


def main():
    """Write machine-readable numerical evidence and fail when a verification gate fails."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=Path("newton/tests/outputs/rft"))
    args = parser.parse_args()
    result = run(args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / f"rft-{args.device.replace(':', '-')}.json"
    path.write_text(json.dumps(result, indent=2))
    print(path)
    print(json.dumps(result["gates"], indent=2))
    if not all(result["gates"].values()):
        raise SystemExit("One or more numerical verification gates failed; inspect the report.")


if __name__ == "__main__":
    main()
