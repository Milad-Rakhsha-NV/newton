#!/usr/bin/env python3
"""
Benchmark humanoid(s) across MuJoCo, XPBD, and DVI solvers.

Two modes:
  1. **Hanging** (default): Each humanoid is attached to the world via a ball
     joint at the root and hangs under gravity with no contacts or floor.
     This isolates pure joint-constraint solver performance.

  2. **Falling** (``--contacts``): Each humanoid is spawned above a ground
     plane with a FREE root joint and falls under gravity.  Contacts with the
     floor are enabled, testing constraint + contact solver performance.

Usage:
    # Headless benchmark (hanging, no contacts)
    python benchmark_hanging_humanoid.py dvi --num-envs 64 --sim-time 10

    # Falling with contacts
    python benchmark_hanging_humanoid.py dvi_implicit --num-envs 64 --contacts

    # With live viewer
    python benchmark_hanging_humanoid.py xpbd --num-envs 4 --sim-time 5 --render

    # Compare all solvers (hanging)
    for s in mujoco xpbd dvi dvi_implicit; do
        python benchmark_hanging_humanoid.py $s --num-envs 256 --sim-time 10
    done

    # Compare all solvers (falling with contacts)
    for s in mujoco xpbd dvi dvi_implicit; do
        python benchmark_hanging_humanoid.py $s --num-envs 256 --sim-time 10 --contacts
    done
"""

import argparse
import math
import os
import sys
import time

os.environ.setdefault("WARP_LOG_LEVEL", "warning")

import numpy as np
import warp as wp

# Ensure newton is importable from the repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import newton
import newton.examples
from newton._src.sim.builder import Axis
from newton.solvers import ActuatorIntegration, SolverType


# -- Fair solver parameters (matched at ~1-2 mm constraint violation) ----------
SOLVER_PARAMS = {
    "mujoco":           {"substeps": 1, "iterations": 1, "ls_iterations": 4},
    "xpbd":             {"substeps": 4, "iterations": 2},
    "dvi":           {"substeps": 1, "position_correction": True, "actuator_integration": ActuatorIntegration.SEMI_IMPLICIT},
    "dvi_implicit":  {"substeps": 1, "position_correction": True, "actuator_integration": ActuatorIntegration.SEMI_IMPLICIT},
}

FPS = 60  # simulation frame rate


# -- Model builder -------------------------------------------------------------
def build_model(num_envs: int, device: str = "cuda:0", enable_contacts: bool = False):
    """Build a model with *num_envs* nv_humanoids.

    When *enable_contacts* is False (default), humanoids hang from ball joints
    with collisions disabled.  When True, humanoids have FREE root joints and
    are spawned above a ground plane with collisions enabled.

    Returns ``(model, builder)`` so the caller can extract custom attributes.
    """
    mjcf_file = newton.examples.get_asset("nv_humanoid.xml")
    ax_x = newton.ModelBuilder.JointDofConfig(axis=Axis.X)
    ax_y = newton.ModelBuilder.JointDofConfig(axis=Axis.Y)
    ax_z = newton.ModelBuilder.JointDofConfig(axis=Axis.Z)

    builder = newton.ModelBuilder(gravity=-10.0)
    cols = max(int(math.ceil(math.sqrt(num_envs))), 1)

    if enable_contacts:
        # Falling mode: FREE root joint, global ground plane, collisions on.
        # Model config matches example_humanoid.py: shape contact properties,
        # uniform joint PD (ke=50, kd=5), replicate pattern.
        spawn_height = 1.3

        humanoid = newton.ModelBuilder(gravity=-10.0)
        humanoid.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
            limit_ke=1.0e3, limit_kd=1.0e1, friction=1e-5)
        humanoid.default_shape_cfg.ke = 5.0e4
        humanoid.default_shape_cfg.kd = 5.0e2
        humanoid.default_shape_cfg.kf = 1.0e3
        humanoid.default_shape_cfg.mu = 0.75
        humanoid.default_shape_cfg.gap = 0.01
        humanoid.default_shape_cfg.margin = 0.001

        humanoid.add_mjcf(
            mjcf_file,
            ignore_names=["floor", "ground"],
            parse_sites=False,
            xform=wp.transform(wp.vec3(0, 0, spawn_height)),
        )
        for i in range(len(humanoid.joint_target_ke)):
            humanoid.joint_target_ke[i] = 50
            humanoid.joint_target_kd[i] = 5

        builder = newton.ModelBuilder()
        builder.replicate(humanoid, num_envs)
        builder.add_ground_plane()
    else:
        # Hanging mode: BALL root joint, no floor, collisions off.
        # Parse MJCF once, then replicate (17× faster than loop add_mjcf).
        spawn_height = 2.0
        humanoid = newton.ModelBuilder(gravity=-10.0)
        humanoid.add_mjcf(
            mjcf_file,
            ignore_names=["floor", "ground"],
            parse_sites=False,
            base_joint={
                "joint_type": newton.JointType.BALL,
                "angular_axes": [ax_x, ax_y, ax_z],
            },
            xform=wp.transform(wp.vec3(0, 0, spawn_height), wp.quat_identity()),
        )
        builder = newton.ModelBuilder(gravity=-10.0)
        builder.replicate(humanoid, num_envs)
        # Disable all collisions for hanging mode.
        for j in range(len(builder.shape_collision_group)):
            builder.shape_collision_group[j] = -1

    model = builder.finalize(device=device)
    return model, builder


def apply_passive_joint_properties(builder: newton.ModelBuilder, model: newton.Model):
    """Copy MJCF passive joint stiffness/damping into joint_target_ke/kd.

    The MJCF importer stores joint ``stiffness`` and ``damping`` as
    MuJoCo-specific custom attributes.  MuJoCo's solver reads these
    natively, but XPBD and DVI rely on :attr:`joint_target_ke` /
    :attr:`joint_target_kd` which default to zero.  This helper copies
    the values so the actuation kernels apply them.
    """
    stiff_attr = builder.custom_attributes.get("mujoco:dof_passive_stiffness")
    damp_attr = builder.custom_attributes.get("mujoco:dof_passive_damping")
    if stiff_attr is None and damp_attr is None:
        return

    ndof = model.joint_dof_count
    ke = model.joint_target_ke.numpy()
    kd = model.joint_target_kd.numpy()

    if stiff_attr is not None:
        for dof_idx, val in stiff_attr.values.items():
            if 0 <= dof_idx < ndof:
                ke[dof_idx] = float(val)

    if damp_attr is not None:
        for dof_idx, val in damp_attr.values.items():
            if 0 <= dof_idx < ndof:
                kd[dof_idx] = float(val)

    # For multi-env: custom attributes only store world-0 DOF indices.
    # Tile across all worlds.
    world_count = getattr(model, "world_count", 1)
    if world_count > 1:
        dof_per_world = ndof // world_count
        for w in range(1, world_count):
            ke[w * dof_per_world : (w + 1) * dof_per_world] = ke[:dof_per_world]
            kd[w * dof_per_world : (w + 1) * dof_per_world] = kd[:dof_per_world]

    model.joint_target_ke.assign(ke)
    model.joint_target_kd.assign(kd)


# -- Solver factory ------------------------------------------------------------
def create_solver(solver_name: str, model: newton.Model, enable_contacts: bool = False):
    """Return ``(solver, contacts, substeps)``."""
    params = SOLVER_PARAMS[solver_name]
    substeps = params["substeps"]

    if solver_name == "mujoco":
        solver = newton.solvers.SolverMuJoCo(
            model,
            solver="newton",
            integrator="euler",
            iterations=params["iterations"],
            ls_iterations=params["ls_iterations"],
            disable_contacts=not enable_contacts,
            # Generous constraint buffer for contacts (humanoid has many body
            # pairs that can touch the ground simultaneously).
            njmax=200 if enable_contacts else None,
            nconmax=100 if enable_contacts else None,
        )
        contacts = model.contacts()
    elif solver_name == "xpbd":
        solver = newton.solvers.SolverXPBD(
            model,
            iterations=params["iterations"],
            joint_linear_relaxation=0.7,
            joint_angular_relaxation=0.4,
        )
        contacts = model.contacts()
    elif solver_name in ("dvi", "dvi_implicit"):
        use_position_correction = params.get("position_correction", False)
        jc = newton.solvers.NumericalSolverConfig(
            solver_type=SolverType.SPARSE_LDL,
            max_iterations=50,
            omega=0.3,
            relax=0.8,
            alpha=0.0,
            recovery_speed=1000000.0,
            reg=1e-6,
            position_correction=None,
        )
        actuator_integration = params.get("actuator_integration", ActuatorIntegration.EXPLICIT)

        # For contacts, create a contact solver.
        # Parameters match example_humanoid.py (known-good config):
        #   - Few iterations (5) with high relax (0.9) to avoid over-correction
        #   - Bounded recovery speed (0.5) prevents aggressive penetration fix
        #   - alpha=0.001 for Baumgarte stabilization on contacts
        if enable_contacts:
            cc = newton.solvers.NumericalSolverConfig(
                solver_type=SolverType.SPARSE_JACOBI,
                max_iterations=20,
                omega=0.5,
                relax=0.9,
                alpha=0.0,
                recovery_speed=0.5,
                reg=1e-4,
            )
        else:
            cc = None

        solver = newton.solvers.SolverDVI(
            model,
            joint_solver=jc,
            contact_solver=cc,
            angular_damping=0.01,
            enable_contacts=enable_contacts,
            enable_timers=False,
            actuator_integration=actuator_integration,
        )
        contacts = model.contacts() if enable_contacts else None
    else:
        raise ValueError(f"Unknown solver: {solver_name}")

    return solver, contacts, substeps


# -- Main ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Benchmark nv_humanoid(s) with different solvers."
    )
    parser.add_argument(
        "solver",
        choices=["mujoco", "xpbd", "dvi", "dvi_implicit"],
        help="Solver to benchmark.",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Number of humanoid environments (default: 1).",
    )
    parser.add_argument(
        "--sim-time",
        type=float,
        default=10.0,
        help="Simulation duration in seconds (default: 10).",
    )
    parser.add_argument(
        "--contacts",
        action="store_true",
        help="Enable contacts: humanoids fall to a ground plane instead of hanging.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Open a live viewer instead of running headless.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Warp device (default: cuda:0).",
    )
    args = parser.parse_args()

    wp.init()
    wp.set_device(args.device)

    num_frames = int(args.sim_time * FPS)
    dt = 1.0 / FPS
    mode_str = "falling (contacts)" if args.contacts else "hanging (no contacts)"

    # -- Build model -----------------------------------------------------------
    print(f"Building model: {args.num_envs} humanoid(s), {mode_str} ...", flush=True)
    t0 = time.perf_counter()
    model, builder = build_model(args.num_envs, device=args.device, enable_contacts=args.contacts)
    t_build = time.perf_counter() - t0
    print(
        f"  Model ready: {model.body_count} bodies, "
        f"{model.joint_count} joints  ({t_build:.2f} s)",
        flush=True,
    )

    # For contacts mode, model already has ke=50/kd=5 from build_model.
    # For hanging mode, copy MJCF passive stiffness/damping for XPBD/DVI.
    if not args.contacts and args.solver in ("xpbd", "dvi", "dvi_implicit"):
        apply_passive_joint_properties(builder, model)

    # For contacts mode, run an initial collision to size contact buffers.
    if args.contacts:
        state_tmp = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_tmp)
        model.collide(state_tmp)

    # -- Create solver ---------------------------------------------------------
    print(f"Creating {args.solver} solver ...", flush=True)
    t0 = time.perf_counter()
    solver, contacts, substeps = create_solver(args.solver, model, enable_contacts=args.contacts)
    t_solver = time.perf_counter() - t0
    print(
        f"  Solver created  ({t_solver:.2f} s)  "
        f"params={SOLVER_PARAMS[args.solver]}",
        flush=True,
    )

    t_init = t_build + t_solver

    # -- Prepare states --------------------------------------------------------
    state0 = model.state()
    state1 = model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state0)

    sim_dt = dt / substeps

    # -- Headless benchmark ----------------------------------------------------
    if not args.render:
        # Warmup (10 frames, untimed)
        warmup = min(10, num_frames)
        for _ in range(warmup):
            for _ in range(substeps):
                state0.clear_forces()
                if contacts is not None:
                    model.collide(state0, contacts)
                solver.step(state0, state1, control, contacts, sim_dt)
                state0, state1 = state1, state0
        wp.synchronize()

        # Timed run
        print(
            f"Simulating {num_frames} frames "
            f"({args.sim_time:.1f} s @ {FPS} fps) ...",
            flush=True,
        )
        t0 = time.perf_counter()
        for _ in range(num_frames):
            for _ in range(substeps):
                state0.clear_forces()
                if contacts is not None:
                    model.collide(state0, contacts)
                solver.step(state0, state1, control, contacts, sim_dt)
                state0, state1 = state1, state0
        wp.synchronize()
        t_sim = time.perf_counter() - t0

        body_qd = state0.body_qd.numpy()
        max_vel = float(np.max(np.linalg.norm(body_qd[:, :3], axis=1)))
        stable = bool(np.isfinite(body_qd).all() and max_vel < 500)

        # Check positions for falling mode (are bodies above ground?)
        body_q = state0.body_q.numpy()
        min_z = float(np.min(body_q[:, 2]))  # z position (height)

        ms_per_frame = 1000.0 * t_sim / num_frames

        print()
        print("=" * 70)
        print(f"  Solver          : {args.solver}")
        print(f"  Mode            : {mode_str}")
        print(f"  Environments    : {args.num_envs}")
        print(f"  Bodies / joints : {model.body_count} / {model.joint_count}")
        print(f"  Sim time        : {args.sim_time:.1f} s  ({num_frames} frames)")
        print(f"  Init time       : {t_init:.2f} s  (build {t_build:.2f} + solver {t_solver:.2f})")
        print(f"  Simulation time : {t_sim:.2f} s")
        print(f"  ms / frame      : {ms_per_frame:.2f}")
        print(f"  Sim FPS         : {1000.0 / ms_per_frame:.1f}")
        print(f"  Stable          : {stable}  (max velocity = {max_vel:.2f})")
        if args.contacts:
            print(f"  Min Z height    : {min_z:.3f}")
        print("=" * 70)

    # -- Live viewer -----------------------------------------------------------
    else:
        try:
            viewer = newton.viewer.ViewerGL()
        except Exception as e:
            print(f"Error opening viewer: {e}", file=sys.stderr)
            print("Try running without --render for headless mode.", file=sys.stderr)
            sys.exit(1)

        viewer.set_model(model)

        frame = 0
        sim_time = 0.0
        while viewer.is_running() and frame < num_frames:
            if viewer.should_step():
                for _ in range(substeps):
                    state0.clear_forces()
                    if contacts is not None:
                        model.collide(state0, contacts)
                    solver.step(state0, state1, control, contacts, sim_dt)
                    state0, state1 = state1, state0
                sim_time += dt
                frame += 1

            viewer.begin_frame(sim_time)
            viewer.log_state(state0)
            viewer.end_frame()

        viewer.close()
        print(f"Viewer closed after {frame} frames.")


if __name__ == "__main__":
    main()
