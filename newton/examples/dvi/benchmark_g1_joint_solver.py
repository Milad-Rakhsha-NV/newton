#!/usr/bin/env python3
"""
Benchmark G1 robot with different DVI joint solver configurations.

Two DVI configs matching Isaac Lab newton_dvi preset, differing only in
joint_alpha and joint_position_correction:

  - dvi_poscorr:   alpha=1e6, position_correction=True  (Isaac Lab default)
  - dvi_baumgarte: alpha=0.0, position_correction=False

Plus MuJoCo as reference.

Usage:
    python benchmark_g1_joint_solver.py --sim-time 5 --video
    python benchmark_g1_joint_solver.py --configs mujoco dvi_poscorr --sim-time 3
"""

import argparse
import os
import subprocess
import sys
import time

os.environ.setdefault("WARP_LOG_LEVEL", "warning")

import numpy as np
import warp as wp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import newton
import newton.utils
from newton import CollisionPipeline, JointTargetMode, eval_fk, eval_ik
from newton.solvers import ActuatorIntegration, FrictionProjection, SolverType
from newton._src.solvers.dvi.numerical_solver.base import NumericalSolverConfig


# ── Simulation parameters (matching Isaac Lab newton_dvi) ─────────────────
PHYSICS_DT = 0.005    # Isaac Lab sim.dt = 0.005 (200 Hz physics step)
NUM_SUBSTEPS = 4      # Isaac Lab num_substeps=4
SIM_DT = PHYSICS_DT / NUM_SUBSTEPS  # 0.00125 s per substep (800 Hz)
SIM_FPS = 50          # Animation frame rate for video (= 1/PHYSICS_DT / decimation)

# ── Shared DVI solver parameters (everything except alpha / pos_corr) ─────
_DVI_COMMON = dict(
    # Joint solver (LDL)
    joint_solver_type=SolverType.SPARSE_LDL,
    joint_max_iterations=50,
    joint_recovery_speed=100000.0,
    joint_omega=0.3,
    joint_relax=0.8,
    joint_reg=1e-6,
    joint_iterative_refinement_steps=1,
    # Contact solver (Jacobi) — Isaac Lab defaults
    contact_solver_type=SolverType.SPARSE_JACOBI,
    contact_max_iterations=50,
    contact_alpha=0.0,
    contact_recovery_speed=1.0,
    contact_omega=0.3,
    contact_relax=0.9,
    contact_reg=1e-4,
    contact_position_correction=False,
    contact_block_precondition=False,
    contact_friction_projection=FrictionProjection.CONE,
    # Joint limit solver — constraint-based
    joint_limit_solver_type=SolverType.SPARSE_JACOBI,
    joint_limit_ke_scale=0.1,
    # Other
    angular_damping=0.01,
    actuator_integration=ActuatorIntegration.SEMI_IMPLICIT,
)

DVI_CONFIGS = {
    "dvi_poscorr": {
        "label": "DVI (alpha=1e6, pos_corr=True)",
        "joint_alpha": 1e6,
        "joint_position_correction": True,
    },
    "dvi_baumgarte": {
        "label": "DVI (alpha=0.0, pos_corr=False)",
        "joint_alpha": 0.0,
        "joint_position_correction": False,
    },
}

# Collision pipeline settings (matching Isaac Lab)
RIGID_CONTACT_MAX = 665536
SHAPE_GAP = 0.005


# ── Isaac Lab G1 actuator gains (from G1_CFG in isaaclab_assets) ─────────────
# Mapping from joint label substrings to (stiffness, damping, armature)
_ACTUATOR_MAP = [
    # Legs: hip/knee/torso
    ("hip_pitch_joint",    200.0, 5.0, 0.01),
    ("hip_roll_joint",     150.0, 5.0, 0.01),
    ("hip_yaw_joint",      150.0, 5.0, 0.01),
    ("knee_joint",         200.0, 5.0, 0.01),
    ("waist_yaw_joint",    200.0, 5.0, 0.01),   # torso_joint equivalent
    ("waist_roll_joint",   200.0, 5.0, 0.01),
    ("waist_pitch_joint",  200.0, 5.0, 0.01),
    # Feet
    ("ankle_pitch_joint",   20.0, 2.0, 0.01),
    ("ankle_roll_joint",    20.0, 2.0, 0.01),
    # Arms (shoulders, elbows, wrists)
    ("shoulder_pitch_joint", 40.0, 10.0, 0.01),
    ("shoulder_roll_joint",  40.0, 10.0, 0.01),
    ("shoulder_yaw_joint",   40.0, 10.0, 0.01),
    ("elbow_joint",          40.0, 10.0, 0.01),
    ("wrist_roll_joint",     40.0, 10.0, 0.01),
    ("wrist_pitch_joint",    40.0, 10.0, 0.01),
    ("wrist_yaw_joint",      40.0, 10.0, 0.01),
    # Fingers (all *_zero/one/two/three/four/five/six/index/middle/thumb joints)
    ("_hand_",               40.0, 10.0, 0.001),
]

# Isaac Lab G1 initial joint positions (from G1_CFG.init_state.joint_pos)
_INIT_JOINT_POS = {
    "hip_pitch_joint": -0.20,
    "knee_joint": 0.42,
    "ankle_pitch_joint": -0.23,
    "elbow_pitch_joint": 0.87,
    "left_shoulder_roll_joint": 0.16,
    "left_shoulder_pitch_joint": 0.35,
    "right_shoulder_roll_joint": -0.16,
    "right_shoulder_pitch_joint": 0.35,
    "left_one_joint": 1.0,
    "right_one_joint": -1.0,
    "left_two_joint": 0.52,
    "right_two_joint": -0.52,
}


def _apply_actuator_gains(builder):
    """Apply Isaac Lab per-joint stiffness, damping, armature to builder DOFs."""
    for j in range(builder.joint_count):
        qd_start = builder.joint_qd_start[j]
        if qd_start < 6:  # skip free joint (DOFs 0-5)
            continue
        label = builder.joint_label[j]
        matched = False
        for pattern, ke, kd, armature in _ACTUATOR_MAP:
            if pattern in label:
                builder.joint_target_ke[qd_start] = ke
                builder.joint_target_kd[qd_start] = kd
                builder.joint_armature[qd_start] = armature
                builder.joint_target_mode[qd_start] = int(JointTargetMode.POSITION)
                matched = True
                break
        if not matched:
            # Fallback: arms default
            builder.joint_target_ke[qd_start] = 40.0
            builder.joint_target_kd[qd_start] = 10.0
            builder.joint_armature[qd_start] = 0.001
            builder.joint_target_mode[qd_start] = int(JointTargetMode.POSITION)


def _apply_initial_pose(builder):
    """Apply Isaac Lab initial joint positions (joint_q is indexed by q_start)."""
    for j in range(builder.joint_count):
        q_start = builder.joint_q_start[j]
        label = builder.joint_label[j]
        for pattern, value in _INIT_JOINT_POS.items():
            if pattern in label:
                builder.joint_q[q_start] = value
                break


def _set_targets_from_initial_pose(model, control):
    """Copy initial joint_q values into control.joint_target_pos (DOF-indexed)."""
    jq = model.joint_q.numpy()
    tp = control.joint_target_pos.numpy()
    # For each joint, map q_start -> qd_start for revolute (1:1)
    # Skip the free joint (6 DOFs but 7 coords)
    joint_type = model.joint_type.numpy()
    joint_q_start = model.joint_q_start.numpy()
    joint_qd_start = model.joint_qd_start.numpy()
    for j in range(model.joint_count):
        qs = joint_q_start[j]
        qds = joint_qd_start[j]
        jt = joint_type[j]
        if jt == 0:  # JOINT_FREE — skip (6 dofs, 7 coords)
            continue
        elif jt == 1:  # JOINT_REVOLUTE — 1 coord, 1 dof
            tp[qds] = jq[qs]
        elif jt == 5:  # JOINT_D6 — 6 coords (quat+pos), 6 dofs
            pass  # unlikely for G1
    control.joint_target_pos.assign(wp.array(tp, dtype=wp.float32, device=model.device))


# ── Model builder ────────────────────────────────────────────────────────────
def build_model(num_envs: int, device: str = "cuda:0"):
    """Build G1 model from USD with collapse_fixed_joints, matching Isaac Lab."""
    g1 = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(g1)
    g1.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
        limit_ke=1.0e3, limit_kd=1.0e1, friction=1e-5
    )
    g1.default_shape_cfg.ke = 1.0e3
    g1.default_shape_cfg.kd = 2.0e2
    g1.default_shape_cfg.kf = 1.0e3
    g1.default_shape_cfg.mu = 0.75
    g1.default_shape_cfg.gap = SHAPE_GAP

    asset_path = newton.utils.download_asset("unitree_g1")
    g1.add_usd(
        str(asset_path / "usd_structured" / "g1_29dof_with_hand_rev_1_0.usda"),
        xform=wp.transform(wp.vec3(0, 0, 0)),  # USD already has correct vertical offset
        collapse_fixed_joints=True,
        enable_self_collisions=False,
        hide_collision_shapes=True,
        skip_mesh_approximation=True,
    )

    # Apply Isaac Lab per-joint actuator gains and armature
    _apply_actuator_gains(g1)
    _apply_initial_pose(g1)

    g1.approximate_meshes("bounding_box")

    builder = newton.ModelBuilder()
    builder.replicate(g1, num_envs)
    builder.default_shape_cfg.ke = 1.0e3
    builder.default_shape_cfg.kd = 2.0e2
    builder.add_ground_plane()

    model = builder.finalize(device=device)
    model.rigid_contact_max = RIGID_CONTACT_MAX

    # Apply Isaac Lab effort limits (300 for legs/arms, 20 for ankles)
    eff = model.joint_effort_limit.numpy()
    for j in range(model.joint_count):
        qd_start = int(model.joint_qd_start.numpy()[j])
        if qd_start < 6 or qd_start >= len(eff):
            continue
        label = str(model.joint_label[j])
        if 'ankle' in label:
            eff[qd_start] = 20.0
        else:
            eff[qd_start] = 300.0
    model.joint_effort_limit.assign(wp.array(eff, dtype=wp.float32, device=model.device))

    print(
        f"  Model: {model.body_count} bodies, {model.joint_count} joints, "
        f"{model.joint_dof_count} DOFs ({num_envs} envs)"
    )
    return model


# ── Solver factories ─────────────────────────────────────────────────────────
def create_mujoco_solver(model):
    solver = newton.solvers.SolverMuJoCo(
        model,
        solver="newton",
        integrator="implicitfast",
        njmax=300,
        nconmax=150,
        cone="elliptic",
        impratio=100,
        iterations=100,
        ls_iterations=50,
    )
    contacts = model.contacts()
    return solver, contacts


def create_dvi_solver(model, config_name):
    cfg = DVI_CONFIGS[config_name]
    c = _DVI_COMMON

    # Joint position correction sub-solver (if enabled)
    pos_corr = None
    if cfg["joint_position_correction"]:
        pos_corr = NumericalSolverConfig(
            solver_type=c["joint_solver_type"],
            max_iterations=max(c["joint_max_iterations"] // 2, 10),
            alpha=cfg["joint_alpha"],
            recovery_speed=c["joint_recovery_speed"],
        )

    # Joint solver
    jc = NumericalSolverConfig(
        solver_type=c["joint_solver_type"],
        max_iterations=c["joint_max_iterations"],
        omega=c["joint_omega"],
        relax=c["joint_relax"],
        alpha=cfg["joint_alpha"],
        recovery_speed=c["joint_recovery_speed"],
        reg=c["joint_reg"],
        position_correction=pos_corr,
        iterative_refinement_steps=c["joint_iterative_refinement_steps"],
        diagonal_precondition=True,   # Isaac Lab default
        precond_reg=1e-4,             # Isaac Lab default
    )

    # Contact solver
    contact_pos_corr = None  # contact_position_correction=False
    cc = NumericalSolverConfig(
        solver_type=c["contact_solver_type"],
        max_iterations=c["contact_max_iterations"],
        omega=c["contact_omega"],
        relax=c["contact_relax"],
        alpha=c["contact_alpha"],
        recovery_speed=c["contact_recovery_speed"],
        reg=c["contact_reg"],
        position_correction=contact_pos_corr,
        block_precondition=c["contact_block_precondition"],
        friction_projection=c["contact_friction_projection"],
    )

    # Joint limit solver (None = penalty-based, Isaac Lab default)
    jlc = None
    if c["joint_limit_solver_type"] is not None:
        jlc = NumericalSolverConfig(
            solver_type=c["joint_limit_solver_type"],
            max_iterations=10,
            omega=0.3,
            relax=0.9,
            reg=1e-8,
            alpha=0.0,
            recovery_speed=10.0,
        )

    solver = newton.solvers.SolverDVI(
        model,
        joint_solver=jc,
        contact_solver=cc,
        angular_damping=c["angular_damping"],
        enable_contacts=True,
        enable_timers=False,
        actuator_integration=c["actuator_integration"],
        joint_limit_ke_scale=c["joint_limit_ke_scale"],
        joint_limit_solver=jlc,
    )
    # Use CollisionPipeline like Isaac Lab
    pipeline = CollisionPipeline(model, broad_phase="explicit")
    contacts = model.contacts()
    return solver, contacts, pipeline


# ── Simulation ───────────────────────────────────────────────────────────────
def run_config(config_name, model, sim_time):
    """Run a single config and return metrics."""
    # Each frame = 1 physics step = NUM_SUBSTEPS solver substeps
    num_frames = int(sim_time / PHYSICS_DT)

    state0 = model.state()
    state1 = model.state()
    control = model.control()
    eval_fk(model, model.joint_q, model.joint_qd, state0)

    # Set PD targets to initial joint positions (matching Isaac Lab use_default_offset).
    # joint_target_pos is indexed by qd (DOFs), joint_q by coords.
    # For revolute joints: 1 coord = 1 DOF, so we copy q_start -> qd_start.
    _set_targets_from_initial_pose(model, control)

    pipeline = None
    if config_name == "mujoco":
        solver, contacts = create_mujoco_solver(model)
    else:
        solver, contacts, pipeline = create_dvi_solver(model, config_name)

    vel_history = []
    z_history = []

    t0 = time.perf_counter()
    for frame_i in range(num_frames):
        for sub in range(NUM_SUBSTEPS):
            state0.clear_forces()
            if pipeline is not None:
                pipeline.collide(state0, contacts)
            else:
                model.collide(state0, contacts)
            solver.step(state0, state1, control, contacts, SIM_DT)
            # eval_ik after each substep (matching Isaac Lab dvi_manager)
            if config_name != "mujoco":
                eval_ik(model, state1, state1.joint_q, state1.joint_qd)
            state0, state1 = state1, state0

        if frame_i % 10 == 0:
            wp.synchronize()
            body_q = state0.body_q.numpy()
            body_qd = state0.body_qd.numpy()
            max_vel = float(np.max(np.linalg.norm(body_qd[:, :3], axis=1)))
            min_z = float(np.min(body_q[:, 2]))
            stable = bool(np.isfinite(body_qd).all() and max_vel < 500)
            vel_history.append(max_vel)
            z_history.append(min_z)
            if not stable:
                print(f"    [{config_name}] UNSTABLE at frame {frame_i}!", flush=True)
                break

    wp.synchronize()
    t_sim = time.perf_counter() - t0

    body_q = state0.body_q.numpy()
    body_qd = state0.body_qd.numpy()
    max_vel = float(np.max(np.linalg.norm(body_qd[:, :3], axis=1)))
    mean_vel = float(np.mean(np.linalg.norm(body_qd[:, :3], axis=1)))
    max_ang_vel = float(np.max(np.linalg.norm(body_qd[:, 3:], axis=1)))
    min_z = float(np.min(body_q[:, 2]))
    stable = bool(np.isfinite(body_qd).all() and max_vel < 500)
    ms_per_frame = 1000.0 * t_sim / num_frames

    return {
        "config": config_name,
        "label": DVI_CONFIGS.get(config_name, {}).get("label", "MuJoCo (reference)"),
        "sim_time": t_sim,
        "ms_per_frame": ms_per_frame,
        "max_vel": max_vel,
        "mean_vel": mean_vel,
        "max_ang_vel": max_ang_vel,
        "min_z": min_z,
        "stable": stable,
    }


# ── Video recording ──────────────────────────────────────────────────────────
def record_video(config_name, model, sim_time, output_dir):
    """Record video using ViewerGL headless + ffmpeg pipe."""
    num_frames = int(sim_time / PHYSICS_DT)

    state0 = model.state()
    state1 = model.state()
    control = model.control()
    eval_fk(model, model.joint_q, model.joint_qd, state0)

    # Set PD targets to initial joint positions (matching Isaac Lab use_default_offset)
    _set_targets_from_initial_pose(model, control)

    pipeline = None
    if config_name == "mujoco":
        solver, contacts = create_mujoco_solver(model)
    else:
        solver, contacts, pipeline = create_dvi_solver(model, config_name)

    try:
        viewer = newton.viewer.ViewerGL(headless=True, width=1280, height=720)
    except Exception as e:
        print(f"  ViewerGL failed: {e}", flush=True)
        return None

    viewer.set_model(model)

    video_path = os.path.join(output_dir, f"g1_{config_name}.mp4")

    ffmpeg_proc = subprocess.Popen(
        [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", "1280x720", "-r", str(SIM_FPS),
            "-i", "-",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-crf", "18", video_path,
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    frame_buf = None
    sim_time_val = 0.0
    for frame_i in range(num_frames):
        for sub in range(NUM_SUBSTEPS):
            state0.clear_forces()
            if pipeline is not None:
                pipeline.collide(state0, contacts)
            else:
                model.collide(state0, contacts)
            solver.step(state0, state1, control, contacts, SIM_DT)
            if config_name != "mujoco":
                eval_ik(model, state1, state1.joint_q, state1.joint_qd)
            state0, state1 = state1, state0
        sim_time_val += PHYSICS_DT

        viewer.begin_frame(sim_time_val)
        viewer.log_state(state0)
        viewer.log_contacts(contacts, state0)
        viewer.end_frame()

        frame_buf = viewer.get_frame(target_image=frame_buf)
        ffmpeg_proc.stdin.write(frame_buf.numpy().tobytes())

        # Check stability
        if frame_i % 30 == 0 and frame_i > 0:
            wp.synchronize()
            body_qd = state0.body_qd.numpy()
            if not np.isfinite(body_qd).all():
                print(
                    f"    [{config_name}] NaN at frame {frame_i}, stopping",
                    flush=True,
                )
                break

    ffmpeg_proc.stdin.close()
    ffmpeg_proc.wait()
    viewer.close()

    return video_path if os.path.exists(video_path) else None


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="G1 joint solver benchmark")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--sim-time", type=float, default=5.0)
    parser.add_argument(
        "--configs", nargs="+",
        default=["mujoco", "dvi_poscorr", "dvi_baumgarte"],
    )
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--_record_video", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    wp.init()
    wp.set_device(args.device)

    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(__file__), "g1_joint_solver_results")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Building G1 model ({args.num_envs} envs) ...", flush=True)
    model = build_model(args.num_envs, args.device)

    # Subprocess video-only mode
    if args._record_video:
        cfg_name = args.configs[0]
        vp = record_video(cfg_name, model, args.sim_time, args.output_dir)
        if vp:
            print(f"VIDEO_PATH={vp}")
        return

    # ── Headless benchmark ───────────────────────────────────────────────
    results = []
    for cfg_name in args.configs:
        if cfg_name != "mujoco" and cfg_name not in DVI_CONFIGS:
            print(f"  Skipping unknown config: {cfg_name}")
            continue
        label = DVI_CONFIGS.get(cfg_name, {}).get("label", "MuJoCo (reference)")
        print(f"\n  Running {cfg_name} — {label} ...", flush=True)
        r = run_config(cfg_name, model, args.sim_time)
        results.append(r)
        print(f"    {r['ms_per_frame']:.1f} ms/frame | max_vel={r['max_vel']:.4f} | "
              f"min_z={r['min_z']:.3f} | stable={r['stable']}")

    # Summary table
    print(f"\n{'='*90}")
    print(f"  G1 Benchmark — envs={args.num_envs}, sim={args.sim_time}s, "
          f"dt={SIM_DT*1000:.1f}ms ({NUM_SUBSTEPS} sub @ {SIM_FPS} fps)")
    print(f"{'='*90}")
    print(f"{'Config':<22} {'ms/frm':>8} {'max_vel':>10} {'mean_vel':>10} "
          f"{'max_angv':>10} {'min_z':>8} {'ok':>4}")
    print("-" * 90)
    for r in results:
        ok = "✓" if r["stable"] else "✗"
        print(f"{r['config']:<22} {r['ms_per_frame']:>8.2f} {r['max_vel']:>10.4f} "
              f"{r['mean_vel']:>10.4f} {r['max_ang_vel']:>10.4f} "
              f"{r['min_z']:>8.3f} {ok:>4}")

    # ── Video recording ──────────────────────────────────────────────────
    if args.video:
        print(f"\nRecording videos ...", flush=True)
        for cfg_name in args.configs:
            if cfg_name != "mujoco" and cfg_name not in DVI_CONFIGS:
                continue
            print(f"  Recording {cfg_name} ...", flush=True)
            cmd = [
                sys.executable, os.path.abspath(__file__),
                "--num-envs", str(args.num_envs),
                "--sim-time", str(args.sim_time),
                "--configs", cfg_name,
                "--device", args.device,
                "--output-dir", args.output_dir,
                "--_record_video",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            video_path = os.path.join(args.output_dir, f"g1_{cfg_name}.mp4")
            if os.path.exists(video_path):
                size_kb = os.path.getsize(video_path) / 1024
                print(f"    ✓ {video_path} ({size_kb:.0f} KB)")
            else:
                print(f"    ✗ Failed")
                if result.stderr:
                    for l in result.stderr.strip().split("\n")[-5:]:
                        print(f"      {l}")


if __name__ == "__main__":
    main()
