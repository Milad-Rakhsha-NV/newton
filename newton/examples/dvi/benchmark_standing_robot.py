#!/usr/bin/env python3
"""
Benchmark a standing robot across MuJoCo and DVI solvers.

Each robot is spawned above a ground plane with PD control holding its initial
pose.  The benchmark measures stability (how long before falling), velocity
noise, and performance.

Supported robots:
  - g1:       Unitree G1 (29 DOF humanoid)
  - h1:       Unitree H1 (19 DOF humanoid)
  - anymal_c: ANYbotics ANYmal C (12 DOF quadruped)

Usage:
    # G1 benchmark (default)
    python benchmark_standing_robot.py --robot g1

    # H1 with video recording
    python benchmark_standing_robot.py --robot h1 --video

    # ANYmal C, DVI only, 16 envs
    python benchmark_standing_robot.py --robot anymal_c --configs dvi --num-envs 16

    # Compare all solvers for all robots
    for r in g1 h1 anymal_c; do
        python benchmark_standing_robot.py --robot $r --sim-time 3
    done
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
from newton.solvers import FrictionProjection, ActuatorIntegration, SolverType
from newton._src.solvers.dvi.numerical_solver.base import NumericalSolverConfig


# ═══════════════════════════════════════════════════════════════════════════════
# Simulation parameters (matching Isaac Lab newton_dvi preset)
# ═══════════════════════════════════════════════════════════════════════════════
PHYSICS_DT = 0.005  # 200 Hz physics step
NUM_SUBSTEPS = 4
SIM_DT = PHYSICS_DT / NUM_SUBSTEPS  # 0.00125 s (800 Hz solver)
SIM_FPS = 50  # video frame rate


# ═══════════════════════════════════════════════════════════════════════════════
# Robot configurations
# ═══════════════════════════════════════════════════════════════════════════════
class RobotConfig:
    """Per-robot asset and actuator configuration."""

    def __init__(
        self,
        name: str,
        asset_name: str,
        usd_subpath: str,
        actuator_map: list[tuple[str, float, float, float]],
        init_joint_pos: dict[str, float],
        effort_limits: list[tuple[str, float]],
        *,
        collapse_fixed_joints: bool = True,
        default_ke: float = 40.0,
        default_kd: float = 10.0,
        default_armature: float = 0.01,
        spawn_height: float = 0.0,
        use_urdf: bool = False,
    ):
        self.name = name
        self.asset_name = asset_name
        self.usd_subpath = usd_subpath
        self.actuator_map = actuator_map
        self.init_joint_pos = init_joint_pos
        self.effort_limits = effort_limits
        self.collapse_fixed_joints = collapse_fixed_joints
        self.default_ke = default_ke
        self.default_kd = default_kd
        self.default_armature = default_armature
        self.spawn_height = spawn_height
        self.use_urdf = use_urdf


# ── G1 ───────────────────────────────────────────────────────────────────────
ROBOT_G1 = RobotConfig(
    name="g1",
    asset_name="unitree_g1",
    usd_subpath="usd_structured/g1_29dof_with_hand_rev_1_0.usda",
    actuator_map=[
        # Legs
        ("hip_pitch_joint", 200.0, 5.0, 0.01),
        ("hip_roll_joint", 150.0, 5.0, 0.01),
        ("hip_yaw_joint", 150.0, 5.0, 0.01),
        ("knee_joint", 200.0, 5.0, 0.01),
        # Torso
        ("waist_yaw_joint", 200.0, 5.0, 0.01),
        ("waist_roll_joint", 200.0, 5.0, 0.01),
        ("waist_pitch_joint", 200.0, 5.0, 0.01),
        # Feet
        ("ankle_pitch_joint", 20.0, 2.0, 0.01),
        ("ankle_roll_joint", 20.0, 2.0, 0.01),
        # Arms
        ("shoulder_pitch_joint", 40.0, 10.0, 0.01),
        ("shoulder_roll_joint", 40.0, 10.0, 0.01),
        ("shoulder_yaw_joint", 40.0, 10.0, 0.01),
        ("elbow_joint", 40.0, 10.0, 0.01),
        ("wrist_roll_joint", 40.0, 10.0, 0.01),
        ("wrist_pitch_joint", 40.0, 10.0, 0.01),
        ("wrist_yaw_joint", 40.0, 10.0, 0.01),
        # Fingers
        ("_hand_", 40.0, 10.0, 0.001),
    ],
    init_joint_pos={
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
    },
    effort_limits=[
        ("ankle", 20.0),
        ("", 300.0),  # default for everything else
    ],
    default_ke=40.0,
    default_kd=10.0,
    default_armature=0.001,
)

# ── H1 ───────────────────────────────────────────────────────────────────────
ROBOT_H1 = RobotConfig(
    name="h1",
    asset_name="unitree_h1",
    usd_subpath="usd_structured/h1.usda",
    actuator_map=[
        # Legs
        ("hip_yaw_joint", 200.0, 5.0, 0.01),
        ("hip_roll_joint", 200.0, 5.0, 0.01),
        ("hip_pitch_joint", 200.0, 5.0, 0.01),
        ("knee_joint", 300.0, 8.0, 0.01),
        ("ankle_joint", 40.0, 2.0, 0.01),
        # Torso
        ("torso_joint", 200.0, 5.0, 0.01),
        # Arms
        ("shoulder_pitch_joint", 40.0, 10.0, 0.01),
        ("shoulder_roll_joint", 40.0, 10.0, 0.01),
        ("shoulder_yaw_joint", 40.0, 10.0, 0.01),
        ("elbow_joint", 40.0, 10.0, 0.01),
    ],
    init_joint_pos={
        "hip_pitch_joint": -0.28,
        "knee_joint": 0.58,
        "ankle_joint": -0.28,
        "shoulder_pitch_joint": 0.28,
    },
    effort_limits=[
        ("ankle", 40.0),
        ("", 300.0),
    ],
    collapse_fixed_joints=True,
    default_ke=40.0,
    default_kd=10.0,
    default_armature=0.01,
)

# ── ANYmal C ─────────────────────────────────────────────────────────────────
ROBOT_ANYMAL_C = RobotConfig(
    name="anymal_c",
    asset_name="anybotics_anymal_c",
    usd_subpath="urdf/anymal.urdf",
    actuator_map=[
        ("HAA", 80.0, 2.0, 0.01),  # hip abduction/adduction
        ("HFE", 80.0, 2.0, 0.01),  # hip flexion/extension
        ("KFE", 80.0, 2.0, 0.01),  # knee flexion/extension
    ],
    init_joint_pos={
        "LF_HAA": 0.0,
        "LF_HFE": 0.4,
        "LF_KFE": -0.8,
        "RF_HAA": 0.0,
        "RF_HFE": 0.4,
        "RF_KFE": -0.8,
        "LH_HAA": 0.0,
        "LH_HFE": -0.4,
        "LH_KFE": 0.8,
        "RH_HAA": 0.0,
        "RH_HFE": -0.4,
        "RH_KFE": 0.8,
    },
    effort_limits=[
        ("", 80.0),
    ],
    collapse_fixed_joints=True,
    default_ke=80.0,
    default_kd=2.0,
    default_armature=0.01,
    spawn_height=0.62,
    use_urdf=True,
)

ROBOT_REGISTRY: dict[str, RobotConfig] = {
    "g1": ROBOT_G1,
    "h1": ROBOT_H1,
    "anymal_c": ROBOT_ANYMAL_C,
}


# ═══════════════════════════════════════════════════════════════════════════════
# DVI solver configurations
# ═══════════════════════════════════════════════════════════════════════════════
DVI_CONFIGS = {
    "dvi": {
        "label": "DVI",
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# Model building
# ═══════════════════════════════════════════════════════════════════════════════
def _apply_actuator_gains(builder, robot: RobotConfig):
    """Apply per-joint stiffness, damping, armature from robot config."""
    n_dofs = len(builder.joint_target_ke)
    for j in range(builder.joint_count):
        qd_start = builder.joint_qd_start[j]
        if qd_start < 6 or qd_start >= n_dofs:  # skip free joint / out of range
            continue
        label = builder.joint_label[j]
        matched = False
        for pattern, ke, kd, armature in robot.actuator_map:
            if pattern in label:
                builder.joint_target_ke[qd_start] = ke
                builder.joint_target_kd[qd_start] = kd
                builder.joint_armature[qd_start] = armature
                builder.joint_target_mode[qd_start] = int(JointTargetMode.POSITION)
                matched = True
                break
        if not matched:
            builder.joint_target_ke[qd_start] = robot.default_ke
            builder.joint_target_kd[qd_start] = robot.default_kd
            builder.joint_armature[qd_start] = robot.default_armature
            builder.joint_target_mode[qd_start] = int(JointTargetMode.POSITION)


def _apply_initial_pose(builder, robot: RobotConfig):
    """Apply initial joint positions from robot config."""
    for j in range(builder.joint_count):
        q_start = builder.joint_q_start[j]
        label = builder.joint_label[j]
        for pattern, value in robot.init_joint_pos.items():
            if pattern in label:
                builder.joint_q[q_start] = value
                break


def _apply_effort_limits(model, robot: RobotConfig):
    """Apply effort limits from robot config. USD defaults are often 1e6 (unclamped)."""
    eff = model.joint_effort_limit.numpy()
    n_eff = len(eff)
    for j in range(model.joint_count):
        qd_start = int(model.joint_qd_start.numpy()[j])
        if qd_start < 6 or qd_start >= n_eff:
            continue
        label = str(model.joint_label[j])
        for pattern, limit in robot.effort_limits:
            if pattern == "" or pattern in label:
                eff[qd_start] = limit
                break
    model.joint_effort_limit.assign(
        wp.array(eff, dtype=wp.float32, device=model.device)
    )


def _set_targets_from_initial_pose(model, control):
    """Copy initial joint_q into control targets (DOF-indexed)."""
    jq = model.joint_q.numpy()
    tp = control.joint_target_pos.numpy()
    joint_type = model.joint_type.numpy()
    joint_q_start = model.joint_q_start.numpy()
    joint_qd_start = model.joint_qd_start.numpy()
    for j in range(model.joint_count):
        qs = joint_q_start[j]
        qds = joint_qd_start[j]
        jt = joint_type[j]
        if jt == 0:  # FREE
            continue
        elif jt == 1:  # REVOLUTE
            tp[qds] = jq[qs]
    control.joint_target_pos.assign(
        wp.array(tp, dtype=wp.float32, device=model.device)
    )


def build_model(robot: RobotConfig, num_envs: int, device: str = "cuda:0"):
    """Build robot model from USD."""
    single = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(single)
    single.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
        limit_ke=1.0e3, limit_kd=1.0e1, friction=1e-5
    )
    single.default_shape_cfg.ke = 1.0e3
    single.default_shape_cfg.kd = 2.0e2
    single.default_shape_cfg.kf = 1.0e3
    single.default_shape_cfg.mu = 0.75
    single.default_shape_cfg.gap = 0.005

    asset_path = newton.utils.download_asset(robot.asset_name)
    asset_file = str(asset_path / robot.usd_subpath)
    xform = wp.transform(wp.vec3(0, 0, robot.spawn_height))
    if robot.use_urdf:
        single.add_urdf(
            asset_file,
            xform=xform,
            floating=True,
            enable_self_collisions=False,
        )
    else:
        single.add_usd(
            asset_file,
            xform=xform,
            collapse_fixed_joints=robot.collapse_fixed_joints,
            enable_self_collisions=False,
            hide_collision_shapes=True,
            skip_mesh_approximation=True,
        )

    _apply_actuator_gains(single, robot)
    _apply_initial_pose(single, robot)
    single.approximate_meshes("bounding_box")

    builder = newton.ModelBuilder()
    builder.replicate(single, num_envs)
    builder.default_shape_cfg.ke = 1.0e3
    builder.default_shape_cfg.kd = 2.0e2
    builder.add_ground_plane()

    model = builder.finalize(device=device)
    model.rigid_contact_max = 665536

    _apply_effort_limits(model, robot)

    print(
        f"  {robot.name}: {model.body_count} bodies, {model.joint_count} joints, "
        f"{model.joint_dof_count} DOFs ({num_envs} envs)"
    )
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# Solver factories
# ═══════════════════════════════════════════════════════════════════════════════
def create_mujoco_solver(model):
    """Create MuJoCo solver (reference)."""
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


def create_dvi_solver(model, config_name: str):
    """Create DVI solver with the given config variant.

    Parameters match Isaac Lab DVISolverCfg defaults exactly.
    """
    # Joint solver — LDL direct with diagonal preconditioning
    jc = NumericalSolverConfig(
        solver_type=SolverType.SPARSE_LDL,
        max_iterations=50,
        omega=0.3,
        relax=0.8,
        alpha=0.0,
        recovery_speed=100000.0,
        reg=1e-6,
        iterative_refinement_steps=1,
        diagonal_precondition=True,
        precond_reg=1e-4,
    )

    # Contact solver — Jacobi (no block preconditioning)
    cc = NumericalSolverConfig(
        solver_type=SolverType.SPARSE_JACOBI,
        max_iterations=50,
        omega=0.3,
        relax=0.9,
        alpha=0.0,
        recovery_speed=1.0,
        reg=1e-4,
        block_precondition=False,
        friction_projection=FrictionProjection.TANGENTIAL,
    )

    solver = newton.solvers.SolverDVI(
        model,
        joint_solver=jc,
        contact_solver=cc,
        angular_damping=0.0,
        enable_contacts=True,
        enable_timers=False,
        actuator_integration=ActuatorIntegration.SEMI_IMPLICIT,
        joint_limit_ke_scale=1.0,
        # joint_limit_solver=None → penalty-based limits (Isaac Lab default)
    )
    pipeline = CollisionPipeline(model, broad_phase="explicit")
    contacts = model.contacts()
    return solver, contacts, pipeline


# ═══════════════════════════════════════════════════════════════════════════════
# Simulation
# ═══════════════════════════════════════════════════════════════════════════════
def run_config(config_name: str, model, sim_time: float):
    """Run a single solver config and return metrics."""
    num_frames = int(sim_time / PHYSICS_DT)
    is_mujoco = config_name == "mujoco"

    state0 = model.state()
    state1 = model.state()
    control = model.control()
    eval_fk(model, model.joint_q, model.joint_qd, state0)
    _set_targets_from_initial_pose(model, control)

    pipeline = None
    if is_mujoco:
        solver, contacts = create_mujoco_solver(model)
    else:
        solver, contacts, pipeline = create_dvi_solver(model, config_name)

    pelvis_z = []
    pelvis_ang_vel = []

    t0 = time.perf_counter()
    for frame_i in range(num_frames):
        for sub in range(NUM_SUBSTEPS):
            state0.clear_forces()
            if pipeline is not None:
                pipeline.collide(state0, contacts)
            else:
                model.collide(state0, contacts)
            solver.step(state0, state1, control, contacts, SIM_DT)
            if not is_mujoco:
                eval_ik(model, state1, state1.joint_q, state1.joint_qd)
            state0, state1 = state1, state0

        # Sample pelvis state every frame
        if frame_i % 5 == 0:
            wp.synchronize()
            bq = state0.body_q.numpy()
            bqd = state0.body_qd.numpy()
            z = float(bq[0, 2])
            w = bqd[0, 3:]
            pelvis_z.append(z)
            pelvis_ang_vel.append(float(np.linalg.norm(w)))

            if not np.isfinite(bqd).all() or z < 0.1:
                break

    wp.synchronize()
    t_sim = time.perf_counter() - t0
    ms_per_frame = 1000.0 * t_sim / max(num_frames, 1)

    bq = state0.body_q.numpy()
    bqd = state0.body_qd.numpy()
    final_z = float(bq[0, 2])
    max_vel = float(np.max(np.linalg.norm(bqd[:, :3], axis=1)))
    max_ang_vel = float(np.max(np.linalg.norm(bqd[:, 3:], axis=1)))
    stable = bool(np.isfinite(bqd).all() and max_vel < 500)

    # Compute standing-phase metrics (frames 20-60, ~0.1-0.3s — after landing)
    standing_w = pelvis_ang_vel[4:12] if len(pelvis_ang_vel) > 12 else pelvis_ang_vel
    mean_standing_w = float(np.mean(standing_w)) if standing_w else float("nan")

    label = "MuJoCo" if is_mujoco else DVI_CONFIGS[config_name]["label"]
    return {
        "config": config_name,
        "label": label,
        "ms_per_frame": ms_per_frame,
        "final_z": final_z,
        "max_vel": max_vel,
        "max_ang_vel": max_ang_vel,
        "mean_standing_w": mean_standing_w,
        "stable": stable,
        "pelvis_z": pelvis_z,
        "pelvis_ang_vel": pelvis_ang_vel,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Video recording
# ═══════════════════════════════════════════════════════════════════════════════
def record_video(config_name: str, model, sim_time: float, output_dir: str):
    """Record video via ViewerGL headless + ffmpeg pipe.

    Uses glReadPixels CPU readback instead of viewer.get_frame() PBO path
    to avoid CUDA/GL interop issues when a GPU training run is active.
    """
    from OpenGL import GL as gl

    W, H = 1280, 720
    num_frames = int(sim_time / PHYSICS_DT)
    is_mujoco = config_name == "mujoco"

    state0 = model.state()
    state1 = model.state()
    control = model.control()
    eval_fk(model, model.joint_q, model.joint_qd, state0)
    _set_targets_from_initial_pose(model, control)

    pipeline = None
    if is_mujoco:
        solver, contacts = create_mujoco_solver(model)
    else:
        solver, contacts, pipeline = create_dvi_solver(model, config_name)

    try:
        viewer = newton.viewer.ViewerGL(headless=True, width=W, height=H)
    except Exception as e:
        print(f"  ViewerGL failed: {e}", flush=True)
        return None

    viewer.set_model(model)

    video_path = os.path.join(output_dir, f"{config_name}.mp4")
    ffmpeg_proc = subprocess.Popen(
        [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{W}x{H}", "-r", str(SIM_FPS),
            "-i", "-",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-crf", "18", video_path,
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    sim_time_val = 0.0
    for frame_i in range(num_frames):
        for sub in range(NUM_SUBSTEPS):
            state0.clear_forces()
            if pipeline is not None:
                pipeline.collide(state0, contacts)
            else:
                model.collide(state0, contacts)
            solver.step(state0, state1, control, contacts, SIM_DT)
            if not is_mujoco:
                eval_ik(model, state1, state1.joint_q, state1.joint_qd)
            state0, state1 = state1, state0
        sim_time_val += PHYSICS_DT

        viewer.begin_frame(sim_time_val)
        viewer.log_state(state0)
        viewer.end_frame()

        # CPU readback (avoids CUDA/GL interop issues)
        fbo = viewer.renderer._frame_fbo
        gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, fbo)
        pixels = gl.glReadPixels(0, 0, W, H, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
        gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, 0)
        frame_np = np.frombuffer(pixels, dtype=np.uint8).reshape(H, W, 3)[::-1].copy()
        ffmpeg_proc.stdin.write(frame_np.tobytes())

        if frame_i % 30 == 0 and frame_i > 0:
            wp.synchronize()
            bqd = state0.body_qd.numpy()
            if not np.isfinite(bqd).all():
                print(f"    [{config_name}] NaN at frame {frame_i}", flush=True)
                break

    ffmpeg_proc.stdin.close()
    ffmpeg_proc.wait()
    viewer.close()
    return video_path if os.path.exists(video_path) else None


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Benchmark a standing robot with MuJoCo and DVI solvers"
    )
    parser.add_argument(
        "--robot", choices=list(ROBOT_REGISTRY.keys()), default="g1",
        help="Robot to benchmark (default: g1)",
    )
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--sim-time", type=float, default=3.0)
    parser.add_argument(
        "--configs", nargs="+",
        default=["mujoco", "dvi"],
        help="Solver configs to run (default: mujoco dvi)",
    )
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--_record_video", action="store_true", help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    wp.init()
    wp.set_device(args.device)

    robot = ROBOT_REGISTRY[args.robot]

    if args.output_dir is None:
        args.output_dir = os.path.join(
            os.path.dirname(__file__), "standing_results", robot.name
        )
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Building {robot.name} model ({args.num_envs} envs) ...", flush=True)
    model = build_model(robot, args.num_envs, args.device)

    # Subprocess video-only mode
    if args._record_video:
        cfg_name = args.configs[0]
        vp = record_video(cfg_name, model, args.sim_time, args.output_dir)
        if vp:
            print(f"VIDEO_PATH={vp}")
        return

    # ── Benchmark ────────────────────────────────────────────────────────
    all_configs = ["mujoco"] + list(DVI_CONFIGS.keys())
    results = []
    for cfg_name in args.configs:
        if cfg_name not in all_configs:
            print(f"  Skipping unknown config: {cfg_name}")
            continue
        label = "MuJoCo" if cfg_name == "mujoco" else DVI_CONFIGS[cfg_name]["label"]
        print(f"\n  Running {cfg_name} ({label}) ...", flush=True)
        r = run_config(cfg_name, model, args.sim_time)
        results.append(r)
        print(
            f"    {r['ms_per_frame']:.1f} ms/frame | final_z={r['final_z']:.3f} | "
            f"standing |w|={r['mean_standing_w']:.3f} | stable={r['stable']}"
        )

    # Summary
    print(f"\n{'='*85}")
    print(
        f"  {robot.name.upper()} Standing Benchmark — "
        f"envs={args.num_envs}, sim={args.sim_time}s, "
        f"dt={SIM_DT*1000:.2f}ms ({NUM_SUBSTEPS} sub)"
    )
    print(f"{'='*85}")
    print(
        f"{'Config':<18} {'ms/frm':>8} {'final_z':>8} "
        f"{'stand_w':>8} {'max_v':>8} {'max_w':>8} {'ok':>4}"
    )
    print("-" * 85)
    for r in results:
        ok = "✓" if r["stable"] else "✗"
        print(
            f"{r['config']:<18} {r['ms_per_frame']:>8.2f} {r['final_z']:>8.3f} "
            f"{r['mean_standing_w']:>8.3f} {r['max_vel']:>8.3f} "
            f"{r['max_ang_vel']:>8.3f} {ok:>4}"
        )

    # ── Video ────────────────────────────────────────────────────────────
    if args.video:
        print(f"\nRecording videos ...", flush=True)
        for cfg_name in args.configs:
            if cfg_name not in all_configs:
                continue
            print(f"  Recording {cfg_name} ...", flush=True)
            cmd = [
                sys.executable, os.path.abspath(__file__),
                "--robot", robot.name,
                "--num-envs", str(args.num_envs),
                "--sim-time", str(args.sim_time),
                "--configs", cfg_name,
                "--device", args.device,
                "--output-dir", args.output_dir,
                "--_record_video",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            video_path = os.path.join(args.output_dir, f"{cfg_name}.mp4")
            if os.path.exists(video_path):
                size_kb = os.path.getsize(video_path) / 1024
                print(f"    ✓ {video_path} ({size_kb:.0f} KB)")
            else:
                print(f"    ✗ Failed")
                if result.stderr:
                    for line in result.stderr.strip().split("\n")[-5:]:
                        print(f"      {line}")


if __name__ == "__main__":
    main()
