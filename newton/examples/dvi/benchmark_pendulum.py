#!/usr/bin/env python3
"""
Pendulum dynamics comparison: MuJoCo vs DVI.

Builds single and double pendulums with Ant-leg-like body properties
(capsule radius 0.08, density 5.0) and compares joint angle trajectories
between MuJoCo and DVI solvers.

Scenarios:
  1. Free swing  — pendulum released from 45° with joint damping
  2. Constant torque — fixed torque applied to each joint
  3. Sinusoidal torque — oscillating torque

Usage:
    python benchmark_pendulum.py                      # All scenarios
    python benchmark_pendulum.py --single-only --plot # Single + plots
    python benchmark_pendulum.py --double-only        # Double only
    python benchmark_pendulum.py --sim-time 5 --fps 120
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
# Ant leg geometry (from gymnasium ant.xml + IsaacLab overrides)
# ---------------------------------------------------------------------------
CAPSULE_RADIUS = 0.08
LEG_LENGTH = math.sqrt(0.2**2 + 0.2**2)    # ≈ 0.2828
FOOT_LENGTH = math.sqrt(0.4**2 + 0.4**2)   # ≈ 0.5657
DENSITY = 5.0

# MuJoCo MJCF defaults (not IsaacLab overrides)
JOINT_ARMATURE = 1.0
JOINT_DAMPING = 1.0
GRAVITY = -9.81

FPS = 60
SIM_TIME = 10.0

INIT_ANGLE_1 = math.radians(45.0)
INIT_ANGLE_2 = math.radians(30.0)

CONSTANT_TORQUE = 1.0
SINE_TORQUE_AMP = 2.0
SINE_TORQUE_FREQ = 1.0  # Hz


def capsule_mass_inertia(radius, length, density):
    """Mass and principal inertias for a capsule (cylinder + 2 hemispheres)."""
    r, L = radius, length
    cyl_vol = math.pi * r**2 * L
    sph_vol = (4.0 / 3.0) * math.pi * r**3
    mass = density * (cyl_vol + sph_vol)
    cyl_mass = density * cyl_vol
    sph_mass = density * sph_vol
    Ixx_cyl = cyl_mass * (3 * r**2 + L**2) / 12.0
    Izz_cyl = cyl_mass * r**2 / 2.0
    Ixx_sph = 2.0 * (2.0 / 5.0 * (sph_mass / 2.0) * r**2 + (sph_mass / 2.0) * (L / 2.0)**2)
    Izz_sph = 2.0 * (2.0 / 5.0 * (sph_mass / 2.0) * r**2)
    return mass, (Ixx_cyl + Ixx_sph, Ixx_cyl + Ixx_sph, Izz_cyl + Izz_sph)


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------
def build_single_pendulum(device="cuda:0"):
    """Single pendulum: fixed support + one capsule body on a revolute joint.

    Joint angle=0 means hanging straight down. Initial angle set via joint_q.
    """
    builder = newton.ModelBuilder(gravity=GRAVITY)
    mass, (Ixx, Iyy, Izz) = capsule_mass_inertia(CAPSULE_RADIUS, LEG_LENGTH, DENSITY)
    I = wp.mat33(Ixx, 0, 0, 0, Iyy, 0, 0, 0, Izz)
    half = LEG_LENGTH / 2.0

    rot = wp.quat_from_axis_angle(wp.vec3(0, 1, 0), INIT_ANGLE_1)

    # Fixed support at pivot
    sup = builder.add_link(
        xform=wp.transform(wp.vec3(0, 0, 2), wp.quat_identity()),
        mass=0.0, label="support",
    )

    # Link body — placed at initial tilted position
    link_pos = wp.vec3(
        -half * math.sin(INIT_ANGLE_1), 0.0,
        2.0 - half * math.cos(INIT_ANGLE_1),
    )
    lnk = builder.add_link(
        xform=wp.transform(link_pos, rot),
        mass=mass, inertia=I,
        com=wp.vec3(0, 0, 0),  # COM at capsule center
        lock_inertia=True, label="link",
    )

    cfg = newton.ModelBuilder.ShapeConfig()
    cfg.density = 0.0  # mass already set explicitly
    builder.add_shape_capsule(lnk, radius=CAPSULE_RADIUS, half_height=half, cfg=cfg)

    j0 = builder.add_joint_fixed(
        parent=-1, child=sup,
        parent_xform=wp.transform(wp.vec3(0, 0, 2), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0, 0, 0), wp.quat_identity()),
    )
    j1 = builder.add_joint_revolute(
        parent=sup, child=lnk,
        parent_xform=wp.transform(wp.vec3(0, 0, 0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0, 0, half), wp.quat_identity()),
        axis=(0.0, 1.0, 0.0),
        target_ke=0.0, target_kd=JOINT_DAMPING,
        armature=JOINT_ARMATURE,
    )
    builder.add_articulation([j0, j1])

    for k in range(len(builder.shape_collision_group)):
        builder.shape_collision_group[k] = -1

    model = builder.finalize(device=device)
    # Set initial angle
    jq = model.joint_q.numpy()
    jq[-1] = INIT_ANGLE_1
    model.joint_q.assign(jq)

    return model, builder


def build_double_pendulum(device="cuda:0"):
    """Double pendulum: fixed support + two capsules on revolute joints."""
    builder = newton.ModelBuilder(gravity=GRAVITY)

    mass1, (Ixx1, Iyy1, Izz1) = capsule_mass_inertia(CAPSULE_RADIUS, LEG_LENGTH, DENSITY)
    mass2, (Ixx2, Iyy2, Izz2) = capsule_mass_inertia(CAPSULE_RADIUS, FOOT_LENGTH, DENSITY)
    I1 = wp.mat33(Ixx1, 0, 0, 0, Iyy1, 0, 0, 0, Izz1)
    I2 = wp.mat33(Ixx2, 0, 0, 0, Iyy2, 0, 0, 0, Izz2)
    half1 = LEG_LENGTH / 2.0
    half2 = FOOT_LENGTH / 2.0

    rot1 = wp.quat_from_axis_angle(wp.vec3(0, 1, 0), INIT_ANGLE_1)
    rot2 = wp.quat_from_axis_angle(wp.vec3(0, 1, 0), INIT_ANGLE_2)

    pivot_z = 2.0

    # Support
    sup = builder.add_link(
        xform=wp.transform(wp.vec3(0, 0, pivot_z), wp.quat_identity()),
        mass=0.0, label="support",
    )

    # Link 1 (upper)
    l1_cx = -half1 * math.sin(INIT_ANGLE_1)
    l1_cz = pivot_z - half1 * math.cos(INIT_ANGLE_1)
    lnk1 = builder.add_link(
        xform=wp.transform(wp.vec3(l1_cx, 0, l1_cz), rot1),
        mass=mass1, inertia=I1, com=wp.vec3(0, 0, 0),
        lock_inertia=True, label="link1",
    )
    cfg = newton.ModelBuilder.ShapeConfig()
    cfg.density = 0.0
    builder.add_shape_capsule(lnk1, radius=CAPSULE_RADIUS, half_height=half1, cfg=cfg)

    # Link 2 (lower) — pivot2 at bottom of link1
    pivot2_x = -LEG_LENGTH * math.sin(INIT_ANGLE_1)
    pivot2_z = pivot_z - LEG_LENGTH * math.cos(INIT_ANGLE_1)
    l2_cx = pivot2_x - half2 * math.sin(INIT_ANGLE_2)
    l2_cz = pivot2_z - half2 * math.cos(INIT_ANGLE_2)
    lnk2 = builder.add_link(
        xform=wp.transform(wp.vec3(l2_cx, 0, l2_cz), rot2),
        mass=mass2, inertia=I2, com=wp.vec3(0, 0, 0),
        lock_inertia=True, label="link2",
    )
    builder.add_shape_capsule(lnk2, radius=CAPSULE_RADIUS, half_height=half2, cfg=cfg)

    # Joints
    j0 = builder.add_joint_fixed(
        parent=-1, child=sup,
        parent_xform=wp.transform(wp.vec3(0, 0, pivot_z), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0, 0, 0), wp.quat_identity()),
    )
    j1 = builder.add_joint_revolute(
        parent=sup, child=lnk1,
        parent_xform=wp.transform(wp.vec3(0, 0, 0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0, 0, half1), wp.quat_identity()),
        axis=(0, 1, 0), target_ke=0, target_kd=JOINT_DAMPING, armature=JOINT_ARMATURE,
    )
    j2 = builder.add_joint_revolute(
        parent=lnk1, child=lnk2,
        parent_xform=wp.transform(wp.vec3(0, 0, -half1), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0, 0, half2), wp.quat_identity()),
        axis=(0, 1, 0), target_ke=0, target_kd=JOINT_DAMPING, armature=JOINT_ARMATURE,
    )
    builder.add_articulation([j0, j1, j2])

    for k in range(len(builder.shape_collision_group)):
        builder.shape_collision_group[k] = -1

    model = builder.finalize(device=device)
    # Set initial angles (last 2 coords = revolute angles)
    jq = model.joint_q.numpy()
    jq[-2] = INIT_ANGLE_1
    jq[-1] = INIT_ANGLE_2
    model.joint_q.assign(jq)

    return model, builder


# ---------------------------------------------------------------------------
# Solver factory
# ---------------------------------------------------------------------------
def create_solver(solver_name, model):
    """Create solver. Returns (solver, substeps)."""
    if solver_name == "mujoco":
        solver = newton.solvers.SolverMuJoCo(
            model, solver="newton", integrator="euler",
            iterations=1, ls_iterations=4, disable_contacts=True,
        )
        return solver, 1
    elif solver_name == "dvi":
        jc = newton.solvers.NumericalSolverConfig(
            solver_type=SolverType.SPARSE_LDL,
            max_iterations=50, omega=0.3, relax=0.8,
            alpha=0.0, recovery_speed=1e6, reg=1e-6,
        )
        solver = newton.solvers.SolverDVI(
            model, joint_solver=jc, contact_solver=None,
            angular_damping=0.0, enable_contacts=False,
            enable_timers=False, actuator_integration=ActuatorIntegration.SEMI_IMPLICIT,
        )
        return solver, 1
    else:
        raise ValueError(f"Unknown solver: {solver_name}")


# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------
def simulate(model, solver_name, scenario, sim_time, fps, device="cuda:0"):
    """Run and return (times, joint_angles) arrays."""
    solver, substeps = create_solver(solver_name, model)
    dt = 1.0 / fps
    sim_dt = dt / substeps
    num_frames = int(sim_time * fps)

    # Find revolute joint DOF indices (skip the fixed joint's 0 DOFs)
    num_coords = model.joint_coord_count
    num_dofs = model.joint_dof_count

    state0 = model.state()
    state1 = model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state0)

    times = np.zeros(num_frames + 1)
    # Store all joint_q coords
    all_jq = np.zeros((num_frames + 1, num_coords))
    all_jq[0] = model.joint_q.numpy()

    for frame in range(num_frames):
        t = (frame + 1) * dt

        # Apply torques via joint_f (direct generalized forces per DOF)
        # Note: joint_act is a feedforward input for actuator classes, not used
        # directly by DVI or MuJoCo solvers. Use joint_f for direct torques.
        if scenario == "free_swing":
            pass
        elif scenario == "constant_torque":
            jf = control.joint_f.numpy()
            jf[:] = CONSTANT_TORQUE
            control.joint_f.assign(jf)
        elif scenario == "sine_torque":
            torque = SINE_TORQUE_AMP * math.sin(2 * math.pi * SINE_TORQUE_FREQ * t)
            jf = control.joint_f.numpy()
            jf[:] = torque
            control.joint_f.assign(jf)

        for _ in range(substeps):
            state0.clear_forces()
            solver.step(state0, state1, control, None, sim_dt)
            state0, state1 = state1, state0

        newton.eval_ik(model, state0, model.joint_q, model.joint_qd)
        all_jq[frame + 1] = model.joint_q.numpy()
        times[frame + 1] = t

    # Extract only the revolute joint coords (skip fixed joint coords which are always 0)
    # Fixed joints have 0 coords; revolute have 1 coord each.
    # The revolute coords are the last N entries where N = number of revolute joints.
    # For single pendulum: 1 coord (the angle)
    # For double pendulum: 2 coords (angle1, angle2)
    # We know the fixed joint has 0 coords, so all coords are revolute.
    return times, all_jq


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------
def compare_solvers(pendulum_type, scenario, sim_time, fps, device="cuda:0"):
    """Run both solvers, compare, return stats."""
    build_fn = build_single_pendulum if pendulum_type == "single" else build_double_pendulum
    model_mj, _ = build_fn(device=device)
    model_ch, _ = build_fn(device=device)

    num_coords = model_mj.joint_coord_count

    print(f"\n{'='*70}")
    print(f"  {pendulum_type.upper()} PENDULUM — {scenario}")
    print(f"  Sim: {sim_time}s @ {fps} fps, dt={1.0/fps:.6f}")
    print(f"  Bodies: {model_mj.body_count}, Joints: {model_mj.joint_count}, Coords: {num_coords}")
    mass_info = capsule_mass_inertia(CAPSULE_RADIUS, LEG_LENGTH, DENSITY)
    print(f"  Link mass: {mass_info[0]:.6f} kg, Armature: {JOINT_ARMATURE}, Damping: {JOINT_DAMPING}")
    print(f"{'='*70}")

    t0 = time.perf_counter()
    times_mj, jq_mj = simulate(model_mj, "mujoco", scenario, sim_time, fps, device)
    t_mj = time.perf_counter() - t0

    t0 = time.perf_counter()
    times_ch, jq_ch = simulate(model_ch, "dvi", scenario, sim_time, fps, device)
    t_ch = time.perf_counter() - t0

    diff = jq_mj - jq_ch
    max_diff = np.max(np.abs(diff))
    rms_diff = np.sqrt(np.mean(diff**2))
    mean_diff = np.mean(np.abs(diff))

    print(f"\n  MuJoCo time: {t_mj:.3f}s, DVI time: {t_ch:.3f}s")
    print(f"\n  {'Coord':<8} {'Max Diff (deg)':<16} {'RMS Diff (deg)':<16} {'Final MJ (deg)':<16} {'Final CH (deg)':<16}")
    print(f"  {'-'*72}")
    for c in range(num_coords):
        cd = diff[:, c]
        mx = np.max(np.abs(cd))
        rm = np.sqrt(np.mean(cd**2))
        print(f"  q[{c}]    {math.degrees(mx):>14.4f} {math.degrees(rm):>14.4f} {math.degrees(jq_mj[-1,c]):>14.4f} {math.degrees(jq_ch[-1,c]):>14.4f}")

    print(f"\n  Overall: max={math.degrees(max_diff):.4f}° rms={math.degrees(rms_diff):.4f}° mean={math.degrees(mean_diff):.4f}°")

    # Print trajectory at key timepoints
    print(f"\n  Time(s)  ", end="")
    for c in range(num_coords):
        print(f"  MJ_q{c}(°)    CH_q{c}(°)    Diff(°)", end="")
    print()
    for idx in [0, 30, 60, 120, 180, 300, 450, 600]:
        if idx >= len(times_mj):
            idx = len(times_mj) - 1
        print(f"  {times_mj[idx]:5.2f}  ", end="")
        for c in range(num_coords):
            print(f"  {math.degrees(jq_mj[idx,c]):9.4f}  {math.degrees(jq_ch[idx,c]):9.4f}  {math.degrees(diff[idx,c]):9.4f}", end="")
        print()

    return {
        "pendulum": pendulum_type, "scenario": scenario,
        "times": times_mj, "jq_mj": jq_mj, "jq_ch": jq_ch,
        "max_diff_deg": math.degrees(max_diff),
        "rms_diff_deg": math.degrees(rms_diff),
        "mean_diff_deg": math.degrees(mean_diff),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_results(results_list, output_dir=None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available")
        return

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "pendulum_results")
    os.makedirs(output_dir, exist_ok=True)

    for res in results_list:
        pend, scen = res["pendulum"], res["scenario"]
        times = res["times"]
        jq_mj, jq_ch = res["jq_mj"], res["jq_ch"]
        nc = jq_mj.shape[1]

        fig, axes = plt.subplots(nc + 1, 1, figsize=(12, 3 * (nc + 1)), squeeze=False)
        fig.suptitle(f"{pend.title()} Pendulum — {scen.replace('_',' ').title()}\n"
                     f"Max diff: {res['max_diff_deg']:.4f}°, RMS: {res['rms_diff_deg']:.4f}°", fontsize=13)

        for c in range(nc):
            ax = axes[c, 0]
            ax.plot(times, np.degrees(jq_mj[:, c]), 'b-', label='MuJoCo', lw=1.5)
            ax.plot(times, np.degrees(jq_ch[:, c]), 'r--', label='DVI', lw=1.5)
            ax.set_ylabel(f'q[{c}] (deg)')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Difference plot
        ax = axes[nc, 0]
        for c in range(nc):
            ax.plot(times, np.degrees(jq_mj[:, c] - jq_ch[:, c]), label=f'Δq[{c}]')
        ax.set_ylabel('Diff (deg)')
        ax.set_xlabel('Time (s)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fname = f"{pend}_{scen}.png"
        fpath = os.path.join(output_dir, fname)
        plt.savefig(fpath, dpi=150)
        plt.close()
        print(f"  Plot: {fpath}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Pendulum dynamics: MuJoCo vs DVI")
    parser.add_argument("--single-only", action="store_true")
    parser.add_argument("--double-only", action="store_true")
    parser.add_argument("--scenario", type=str, default=None,
                        choices=["free_swing", "constant_torque", "sine_torque"])
    parser.add_argument("--sim-time", type=float, default=SIM_TIME)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    wp.init()
    wp.set_device(args.device)

    ptypes = ["single"] if args.single_only else ["double"] if args.double_only else ["single", "double"]
    scenarios = [args.scenario] if args.scenario else ["free_swing", "constant_torque", "sine_torque"]

    results = []
    for pt in ptypes:
        for sc in scenarios:
            res = compare_solvers(pt, sc, args.sim_time, args.fps, device=args.device)
            results.append(res)

    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Pendulum':<12} {'Scenario':<18} {'Max Diff (°)':<14} {'RMS Diff (°)':<14}")
    print(f"  {'-'*58}")
    for r in results:
        print(f"  {r['pendulum']:<12} {r['scenario']:<18} {r['max_diff_deg']:>12.4f} {r['rms_diff_deg']:>12.4f}")

    if args.plot:
        plot_results(results)

    worst = max(r["max_diff_deg"] for r in results)
    if worst > 1.0:
        print(f"\n⚠️  Max difference {worst:.4f}° exceeds 1° — dynamics mismatch detected!")
        return 1
    else:
        print(f"\n✅ All within 1° (worst: {worst:.4f}°)")
        return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
