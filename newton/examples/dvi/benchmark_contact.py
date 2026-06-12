#!/usr/bin/env python3
"""
Contact dynamics comparison: MuJoCo vs DVI.

Simple contact scenarios to study friction behavior and normal force convergence:
  1. Box on ground   — static rest, check penetration depth and settling
  2. Sliding box     — initial horizontal velocity, friction decelerates to stop
  3. Box drop        — drop from height, check bounce and settling
  4. Tilted slide    — box on inclined plane, sliding under gravity

Analytical solutions available for validation:
  - Sliding box: stops at t = v0/(mu*g), distance = v0²/(2*mu*g)
  - Tilted slide: accelerates at a = g*(sin(theta) - mu*cos(theta))

Usage:
    python benchmark_contact.py                           # All scenarios
    python benchmark_contact.py --scenario sliding_box    # Single scenario
    python benchmark_contact.py --plot --fps 240          # With plots, higher fps
    python benchmark_contact.py --contact-iters 25,50,100 # Custom iteration sweep
"""

import argparse
import math
import os
import sys
import time

os.environ.setdefault("WARP_LOG_LEVEL", "warning")

import numpy as np
import warp as wp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import newton
from newton.solvers import ActuatorIntegration, SolverType

# ---------------------------------------------------------------------------
# Physical parameters
# ---------------------------------------------------------------------------
GRAVITY = -9.81
FPS = 60
SIM_TIME = 5.0

# Box geometry and material
BOX_HALF = 0.1                           # half-size (m)
BOX_DENSITY = 1000.0                     # kg/m³
BOX_MASS = BOX_DENSITY * (2*BOX_HALF)**3 # 8.0 kg
FRICTION = 0.5
RESTITUTION = 0.0                        # inelastic by default

# Sliding box
SLIDE_V0 = 2.0                           # initial x-velocity (m/s)
# Analytical: stop time = v0/(mu*g) ≈ 0.408 s, stop dist = v0²/(2*mu*g) ≈ 0.408 m

# Box drop
DROP_HEIGHT = 1.0                        # center-of-mass height (m)

# Tilted plane
TILT_ANGLE = math.radians(30)            # 30° incline
# With mu=0.5, cos30≈0.866, sin30=0.5 → mu*cos > sin → box should NOT slide
# Use mu=0.3 instead for tilted: mu*cos30 ≈ 0.26 < 0.5 → slides with a = g*(sin-mu*cos) ≈ 2.35 m/s²
TILT_FRICTION = 0.3


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------
def build_box_on_ground(init_height=None, init_vel=None, friction=FRICTION,
                        restitution=RESTITUTION, device="cuda:0"):
    """Box on infinite ground plane.

    Args:
        init_height: Height of box center. Default: BOX_HALF + small gap.
        init_vel: (vx, vy, vz) initial velocity. Default: zero.
        friction: Coulomb friction coefficient.
        restitution: Coefficient of restitution.
    """
    if init_height is None:
        init_height = BOX_HALF + 0.001
    if init_vel is None:
        init_vel = (0.0, 0.0, 0.0)

    builder = newton.ModelBuilder(gravity=GRAVITY)

    # Ground plane at z=0
    cfg_g = newton.ModelBuilder.ShapeConfig()
    cfg_g.friction = friction
    cfg_g.restitution = restitution
    builder.add_ground_plane(height=0.0, cfg=cfg_g)

    # Dynamic box with free joint
    box = builder.add_body(
        xform=wp.transform(wp.vec3(0, 0, init_height), wp.quat_identity()),
        mass=BOX_MASS, label="box",
    )
    cfg_b = newton.ModelBuilder.ShapeConfig()
    cfg_b.density = 0.0  # mass set explicitly
    cfg_b.friction = friction
    cfg_b.restitution = restitution
    builder.add_shape_box(box, hx=BOX_HALF, hy=BOX_HALF, hz=BOX_HALF, cfg=cfg_b)

    model = builder.finalize(device=device)

    # Initial velocity
    if any(v != 0 for v in init_vel):
        qd = model.joint_qd.numpy()
        qd[0], qd[1], qd[2] = init_vel
        model.joint_qd.assign(qd)

    return model


def build_tilted_box(angle=TILT_ANGLE, friction=TILT_FRICTION, device="cuda:0"):
    """Box on tilted plane. Plane normal = rotated z-axis.

    We tilt the ground by rotating the entire scene: gravity rotated so that
    the effective component along the plane causes sliding.
    Actually simpler: place box on a tilted large box acting as ramp.
    """
    builder = newton.ModelBuilder(gravity=GRAVITY)

    # Ground plane at z=0
    cfg_g = newton.ModelBuilder.ShapeConfig()
    cfg_g.friction = friction
    builder.add_ground_plane(height=0.0, cfg=cfg_g)

    # Ramp: large static box tilted at `angle`
    # Ramp surface at z ≈ 0.5 to keep box above ground
    ramp_rot = wp.quat_from_axis_angle(wp.vec3(0, 1, 0), angle)
    ramp = builder.add_body(
        xform=wp.transform(wp.vec3(0, 0, 0.3), ramp_rot),
        mass=0.0, label="ramp",  # static
    )
    cfg_r = newton.ModelBuilder.ShapeConfig()
    cfg_r.density = 0.0
    cfg_r.friction = friction
    builder.add_shape_box(ramp, hx=5.0, hy=5.0, hz=0.05, cfg=cfg_r)

    # Dynamic box placed on the ramp surface
    # Ramp top surface is at ramp_center_z + 0.05*cos(angle) along z
    ramp_surface_z = 0.3 + 0.05 * math.cos(angle)
    box_z = ramp_surface_z + BOX_HALF * math.cos(angle) + 0.01
    box_x = -BOX_HALF * math.sin(angle)
    box = builder.add_body(
        xform=wp.transform(wp.vec3(box_x, 0, box_z), ramp_rot),
        mass=BOX_MASS, label="box",
    )
    cfg_b = newton.ModelBuilder.ShapeConfig()
    cfg_b.density = 0.0
    cfg_b.friction = friction
    builder.add_shape_box(box, hx=BOX_HALF, hy=BOX_HALF, hz=BOX_HALF, cfg=cfg_b)

    model = builder.finalize(device=device)
    return model


# ---------------------------------------------------------------------------
# Solver factory
# ---------------------------------------------------------------------------
def create_solver(solver_name, model, contact_iters=50):
    if solver_name == "mujoco":
        return newton.solvers.SolverMuJoCo(
            model, solver="newton", integrator="euler",
            iterations=1, ls_iterations=4,
        )
    elif solver_name == "dvi":
        jc = newton.solvers.NumericalSolverConfig(
            solver_type=SolverType.SPARSE_LDL,
            max_iterations=50, omega=0.3, relax=0.8,
            alpha=0.0, recovery_speed=1e6, reg=1e-6,
        )
        cc = newton.solvers.NumericalSolverConfig(
            solver_type=SolverType.SPARSE_JACOBI,
            max_iterations=contact_iters, omega=0.5,
            alpha=0.0,
        )
        return newton.solvers.SolverDVI(
            model, joint_solver=jc, contact_solver=cc,
            angular_damping=0.0, enable_contacts=True,
            enable_timers=False, actuator_integration=ActuatorIntegration.EXPLICIT,
        )
    else:
        raise ValueError(f"Unknown solver: {solver_name}")


# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------
def simulate(model, solver_name, sim_time, fps, contact_iters=50, device="cuda:0"):
    """Run simulation, return (times, positions[N+1,3], velocities[N+1,3])."""
    solver = create_solver(solver_name, model, contact_iters=contact_iters)
    dt = 1.0 / fps
    nframes = int(sim_time * fps)

    state0 = model.state()
    state1 = model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state0)

    box_idx = model.body_count - 1  # last body is the dynamic box

    times = np.zeros(nframes + 1)
    positions = np.zeros((nframes + 1, 3))
    velocities = np.zeros((nframes + 1, 3))

    bq = state0.body_q.numpy()
    bqd = state0.body_qd.numpy()
    positions[0] = bq[box_idx][:3]
    velocities[0] = bqd[box_idx][:3]

    for frame in range(nframes):
        t = (frame + 1) * dt
        state0.clear_forces()
        contacts = model.collide(state0)
        solver.step(state0, state1, control, contacts, dt)
        state0, state1 = state1, state0

        bq = state0.body_q.numpy()
        bqd = state0.body_qd.numpy()
        positions[frame + 1] = bq[box_idx][:3]
        velocities[frame + 1] = bqd[box_idx][:3]
        times[frame + 1] = t

    return times, positions, velocities


# ---------------------------------------------------------------------------
# Analytical solutions
# ---------------------------------------------------------------------------
def analytical_sliding_box(times, v0, mu, g=9.81):
    """Sliding box on flat ground: constant deceleration until stop."""
    a = mu * g
    t_stop = v0 / a
    pos = np.where(times < t_stop,
                   v0 * times - 0.5 * a * times**2,
                   v0 * t_stop - 0.5 * a * t_stop**2)
    vel = np.where(times < t_stop, v0 - a * times, 0.0)
    return pos, vel


# ---------------------------------------------------------------------------
# Comparison runner
# ---------------------------------------------------------------------------
def compare(name, build_fn, build_kwargs, sim_time, fps, contact_iters_list,
            analytical_fn=None, analytical_kwargs=None):
    """Run MuJoCo + DVI at each contact iteration count, print comparison."""
    print(f"\n{'='*80}")
    print(f"  {name}")
    print(f"  Box: mass={BOX_MASS:.1f}kg, half={BOX_HALF}m, friction={build_kwargs.get('friction', FRICTION)}")
    print(f"  Sim: {sim_time}s @ {fps} fps, dt={1.0/fps:.6f}")
    print(f"{'='*80}")

    # MuJoCo reference
    model_mj = build_fn(**build_kwargs)
    t0 = time.perf_counter()
    t_mj, pos_mj, vel_mj = simulate(model_mj, "mujoco", sim_time, fps)
    wall_mj = time.perf_counter() - t0

    print(f"\n  MuJoCo ({wall_mj:.2f}s):")
    print(f"    Final pos: ({pos_mj[-1,0]:.6f}, {pos_mj[-1,1]:.6f}, {pos_mj[-1,2]:.6f})")
    print(f"    Final vel: ({vel_mj[-1,0]:.6f}, {vel_mj[-1,1]:.6f}, {vel_mj[-1,2]:.6f})")
    print(f"    Z range: [{np.min(pos_mj[:,2]):.6f}, {np.max(pos_mj[:,2]):.6f}]")

    results = {"mujoco": {"t": t_mj, "pos": pos_mj, "vel": vel_mj, "wall": wall_mj}}

    for ci in contact_iters_list:
        model_ch = build_fn(**build_kwargs)
        t0 = time.perf_counter()
        t_ch, pos_ch, vel_ch = simulate(model_ch, "dvi", sim_time, fps, contact_iters=ci)
        wall_ch = time.perf_counter() - t0

        n = min(len(t_mj), len(t_ch))
        dp = pos_mj[:n] - pos_ch[:n]
        dv = vel_mj[:n] - vel_ch[:n]

        print(f"\n  DVI iters={ci} ({wall_ch:.2f}s):")
        print(f"    Final pos: ({pos_ch[-1,0]:.6f}, {pos_ch[-1,1]:.6f}, {pos_ch[-1,2]:.6f})")
        print(f"    Final vel: ({vel_ch[-1,0]:.6f}, {vel_ch[-1,1]:.6f}, {vel_ch[-1,2]:.6f})")
        print(f"    Z range: [{np.min(pos_ch[:,2]):.6f}, {np.max(pos_ch[:,2]):.6f}]")
        print(f"    Max pos diff: x={np.max(np.abs(dp[:,0])):.6f}, y={np.max(np.abs(dp[:,1])):.6f}, z={np.max(np.abs(dp[:,2])):.6f}")
        print(f"    Max vel diff: vx={np.max(np.abs(dv[:,0])):.6f}, vy={np.max(np.abs(dv[:,1])):.6f}, vz={np.max(np.abs(dv[:,2])):.6f}")

        results[f"dvi_{ci}"] = {"t": t_ch, "pos": pos_ch, "vel": vel_ch, "wall": wall_ch}

    # Analytical comparison if available
    if analytical_fn is not None:
        ana_pos, ana_vel = analytical_fn(t_mj, **(analytical_kwargs or {}))
        print(f"\n  Analytical:")
        # Compare MuJoCo vs analytical
        dp_mj = pos_mj[:, 0] - ana_pos  # x-position
        dv_mj = vel_mj[:, 0] - ana_vel
        print(f"    MuJoCo vs analytical: max_pos_err={np.max(np.abs(dp_mj)):.6f}m, max_vel_err={np.max(np.abs(dv_mj)):.6f}m/s")
        # Compare each DVI vs analytical
        for ci in contact_iters_list:
            r = results[f"dvi_{ci}"]
            n = min(len(ana_pos), len(r["pos"]))
            dp_ch = r["pos"][:n, 0] - ana_pos[:n]
            dv_ch = r["vel"][:n, 0] - ana_vel[:n]
            print(f"    DVI i={ci} vs analytical: max_pos_err={np.max(np.abs(dp_ch)):.6f}m, max_vel_err={np.max(np.abs(dv_ch)):.6f}m/s")
        results["analytical"] = {"pos_x": ana_pos, "vel_x": ana_vel}

    # Trajectory at key timepoints
    ci_ref = contact_iters_list[len(contact_iters_list)//2]
    r_ch = results[f"dvi_{ci_ref}"]
    print(f"\n  {'t(s)':>6}  {'MJ_x':>10} {'CH_x':>10} {'Δx':>10}  {'MJ_vx':>10} {'CH_vx':>10} {'Δvx':>10}  {'MJ_z':>10} {'CH_z':>10} {'Δz':>10}")
    for idx in [0, 5, 10, 30, 60, 120, 180, 300]:
        if idx >= len(t_mj):
            idx = len(t_mj) - 1
        print(f"  {t_mj[idx]:6.3f}  {pos_mj[idx,0]:10.5f} {r_ch['pos'][idx,0]:10.5f} {pos_mj[idx,0]-r_ch['pos'][idx,0]:10.5f}"
              f"  {vel_mj[idx,0]:10.5f} {r_ch['vel'][idx,0]:10.5f} {vel_mj[idx,0]-r_ch['vel'][idx,0]:10.5f}"
              f"  {pos_mj[idx,2]:10.5f} {r_ch['pos'][idx,2]:10.5f} {pos_mj[idx,2]-r_ch['pos'][idx,2]:10.5f}")

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_all(all_results, output_dir=None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "contact_results")
    os.makedirs(output_dir, exist_ok=True)

    for scenario_name, results in all_results.items():
        r_mj = results["mujoco"]
        t_mj = r_mj["t"]

        fig, axes = plt.subplots(4, 1, figsize=(14, 14), squeeze=False)
        fig.suptitle(f"Contact Benchmark: {scenario_name}", fontsize=14)

        # Z position (penetration / settling)
        ax = axes[0, 0]
        ax.plot(t_mj, r_mj["pos"][:, 2], 'b-', label='MuJoCo', lw=2)
        for key in sorted(results.keys()):
            if key.startswith("dvi"):
                r = results[key]
                ax.plot(r["t"], r["pos"][:, 2], '--', label=key.replace("_", " i="), lw=1.2)
        ax.axhline(y=BOX_HALF, color='gray', ls=':', label=f'z={BOX_HALF} (resting)')
        ax.set_ylabel('Z position (m)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # X position
        ax = axes[1, 0]
        ax.plot(t_mj, r_mj["pos"][:, 0], 'b-', label='MuJoCo', lw=2)
        for key in sorted(results.keys()):
            if key.startswith("dvi"):
                r = results[key]
                ax.plot(r["t"], r["pos"][:, 0], '--', label=key.replace("_", " i="), lw=1.2)
        if "analytical" in results:
            ax.plot(t_mj, results["analytical"]["pos_x"], 'k:', label='Analytical', lw=1.5)
        ax.set_ylabel('X position (m)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # X velocity
        ax = axes[2, 0]
        ax.plot(t_mj, r_mj["vel"][:, 0], 'b-', label='MuJoCo', lw=2)
        for key in sorted(results.keys()):
            if key.startswith("dvi"):
                r = results[key]
                ax.plot(r["t"], r["vel"][:, 0], '--', label=key.replace("_", " i="), lw=1.2)
        if "analytical" in results:
            ax.plot(t_mj, results["analytical"]["vel_x"], 'k:', label='Analytical', lw=1.5)
        ax.set_ylabel('X velocity (m/s)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Z velocity
        ax = axes[3, 0]
        ax.plot(t_mj, r_mj["vel"][:, 2], 'b-', label='MuJoCo', lw=2)
        for key in sorted(results.keys()):
            if key.startswith("dvi"):
                r = results[key]
                ax.plot(r["t"], r["vel"][:, 2], '--', label=key.replace("_", " i="), lw=1.2)
        ax.set_ylabel('Z velocity (m/s)')
        ax.set_xlabel('Time (s)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fname = f"{scenario_name.replace(' ', '_').lower()}.png"
        fpath = os.path.join(output_dir, fname)
        plt.savefig(fpath, dpi=150)
        plt.close()
        print(f"  Plot: {fpath}")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def print_summary(all_results, contact_iters_list):
    print(f"\n{'='*80}")
    print(f"  CONTACT BENCHMARK SUMMARY")
    print(f"{'='*80}")
    print(f"  {'Scenario':<30} {'Solver':<16} {'Max Δz (m)':<14} {'Max Δvx (m/s)':<14} {'Final z (m)':<14}")
    print(f"  {'-'*88}")
    for scenario, results in all_results.items():
        r_mj = results["mujoco"]
        print(f"  {scenario:<30} {'mujoco':<16} {'---':<14} {'---':<14} {r_mj['pos'][-1,2]:<14.6f}")
        for ci in contact_iters_list:
            key = f"dvi_{ci}"
            if key not in results:
                continue
            r = results[key]
            n = min(len(r_mj["pos"]), len(r["pos"]))
            dz = np.max(np.abs(r_mj["pos"][:n, 2] - r["pos"][:n, 2]))
            dvx = np.max(np.abs(r_mj["vel"][:n, 0] - r["vel"][:n, 0]))
            print(f"  {'':<30} {f'dvi i={ci}':<16} {dz:<14.6f} {dvx:<14.6f} {r['pos'][-1,2]:<14.6f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Contact dynamics: MuJoCo vs DVI")
    parser.add_argument("--scenario", type=str, default=None,
                        choices=["box_on_ground", "sliding_box", "box_drop", "tilted_slide"])
    parser.add_argument("--sim-time", type=float, default=SIM_TIME)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--contact-iters", type=str, default="25,50,100",
                        help="Comma-separated list of contact iteration counts")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    wp.init()
    wp.set_device(args.device)

    ci_list = [int(x) for x in args.contact_iters.split(",")]
    scenarios = [args.scenario] if args.scenario else ["box_on_ground", "sliding_box", "box_drop", "tilted_slide"]

    all_results = {}

    if "box_on_ground" in scenarios:
        all_results["box_on_ground"] = compare(
            "Box on Ground (static rest)",
            build_box_on_ground, {"init_height": BOX_HALF + 0.001},
            args.sim_time, args.fps, ci_list,
        )

    if "sliding_box" in scenarios:
        all_results["sliding_box"] = compare(
            f"Sliding Box (v0={SLIDE_V0} m/s, mu={FRICTION})",
            build_box_on_ground, {"init_height": BOX_HALF + 0.001, "init_vel": (SLIDE_V0, 0, 0)},
            args.sim_time, args.fps, ci_list,
            analytical_fn=analytical_sliding_box,
            analytical_kwargs={"v0": SLIDE_V0, "mu": FRICTION},
        )

    if "box_drop" in scenarios:
        all_results["box_drop"] = compare(
            f"Box Drop (h={DROP_HEIGHT}m)",
            build_box_on_ground, {"init_height": DROP_HEIGHT},
            args.sim_time, args.fps, ci_list,
        )

    if "tilted_slide" in scenarios:
        all_results["tilted_slide"] = compare(
            f"Tilted Slide ({math.degrees(TILT_ANGLE):.0f}°, mu={TILT_FRICTION})",
            build_tilted_box, {"angle": TILT_ANGLE, "friction": TILT_FRICTION},
            args.sim_time, args.fps, ci_list,
        )

    print_summary(all_results, ci_list)

    if args.plot:
        plot_all(all_results)

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
