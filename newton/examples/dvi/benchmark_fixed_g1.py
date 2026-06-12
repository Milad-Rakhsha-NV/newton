#!/usr/bin/env python3
"""
Fixed-base G1 benchmark: MuJoCo vs DVI side-by-side.

The G1 humanoid is fixed to the world (no free joint) and driven with
Isaac Lab-style implicit PD actuation:

    target = raw_action * scale + default_joint_pos

Random sinusoidal actions are generated per joint, scaled identically to
the Isaac Lab G1 flat env, then applied as position targets.

Both solvers share the same model (built once) and see the same action
sequence, so any difference in motion is purely solver behavior.

Usage:
    python benchmark_fixed_g1.py                    # side-by-side video
    python benchmark_fixed_g1.py --no-video         # metrics only
    python benchmark_fixed_g1.py --sim-time 5       # 5 second sim
    python benchmark_fixed_g1.py --action-type step # step input instead of sinusoidal
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
# Simulation parameters (matching Isaac Lab)
# ═══════════════════════════════════════════════════════════════════════════════
PHYSICS_DT = 0.005  # 200 Hz physics step
NUM_SUBSTEPS = 4
SIM_DT = PHYSICS_DT / NUM_SUBSTEPS  # 0.00125 s
DVI_NUM_SUBSTEPS = 4  # Match IsaacLab num_substeps=4
DVI_SIM_DT = PHYSICS_DT / DVI_NUM_SUBSTEPS  # 0.00125 s
SIM_FPS = 50  # video frame rate

# ═══════════════════════════════════════════════════════════════════════════════
# Isaac Lab G1 actuator config (from isaaclab_assets/robots/unitree.py G1_CFG)
# ═══════════════════════════════════════════════════════════════════════════════
# Actuator config from G1_CFG (isaaclab_assets/robots/unitree.py)
# G1_MINIMAL_CFG = G1_CFG.copy() with g1_minimal.usd
# (pattern, ke, kd, armature)
ACTUATOR_MAP = [
    # Legs (ImplicitActuatorCfg)
    ("hip_pitch_joint", 200.0, 5.0, 0.01),
    ("hip_roll_joint", 150.0, 5.0, 0.01),
    ("hip_yaw_joint", 150.0, 5.0, 0.01),
    ("knee_joint", 200.0, 5.0, 0.01),
    ("torso_joint", 200.0, 5.0, 0.01),
    # Feet (ImplicitActuatorCfg)
    ("ankle_pitch_joint", 20.0, 2.0, 0.01),
    ("ankle_roll_joint", 20.0, 2.0, 0.01),
    # Arms (ImplicitActuatorCfg) — includes shoulders, elbows, AND fingers
    ("shoulder_pitch_joint", 40.0, 10.0, 0.01),
    ("shoulder_roll_joint", 40.0, 10.0, 0.01),
    ("shoulder_yaw_joint", 40.0, 10.0, 0.01),
    ("elbow_pitch_joint", 40.0, 10.0, 0.01),
    ("elbow_roll_joint", 40.0, 10.0, 0.01),
    # Fingers (part of "arms" actuator group in G1_CFG)
    ("_five_joint", 40.0, 10.0, 0.001),
    ("_three_joint", 40.0, 10.0, 0.001),
    ("_six_joint", 40.0, 10.0, 0.001),
    ("_four_joint", 40.0, 10.0, 0.001),
    ("_zero_joint", 40.0, 10.0, 0.001),
    ("_one_joint", 40.0, 10.0, 0.001),
    ("_two_joint", 40.0, 10.0, 0.001),
]

DEFAULT_KE = 40.0
DEFAULT_KD = 10.0
DEFAULT_ARMATURE = 0.01

# Isaac Lab G1_CFG initial joint positions
INIT_JOINT_POS = {
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

# Isaac Lab action scales (from G1FlatEnvCfg)
# Base scale=0.5, then per-joint overrides
ACTION_SCALE = {
    "hip_": 0.5,
    "knee_joint": 0.5,
    "torso_joint": 0.5,
    "ankle_": 0.5,
    "shoulder_": 0.5,
    "elbow_": 0.5,
    # Fingers get zero scale (not actuated by policy)
    "_zero_joint": 0.0,
    "_one_joint": 0.0,
    "_two_joint": 0.0,
    "_three_joint": 0.0,
    "_four_joint": 0.0,
    "_five_joint": 0.0,
    "_six_joint": 0.0,
}

# Effort limits from G1_CFG
EFFORT_LIMITS = [
    ("ankle", 20.0),
    ("", 300.0),
]

# DVI armature overrides from flat_env_cfg DVISolverCfg
DVI_ARMATURE_OVERRIDE = {
    "shoulder": 0.05,
    "elbow": 0.05,
    "wrist": 0.05,
    "hand": 0.05,
    "finger": 0.05,
}

SPAWN_HEIGHT = 0.74  # Isaac Lab default


# ═══════════════════════════════════════════════════════════════════════════════
# Model building
# ═══════════════════════════════════════════════════════════════════════════════
def _apply_actuator_gains(builder):
    """Apply per-joint stiffness, damping, armature from actuator map."""
    n_dofs = len(builder.joint_target_ke)
    for j in range(builder.joint_count):
        qd_start = builder.joint_qd_start[j]
        if qd_start >= n_dofs:
            continue
        label = builder.joint_label[j]
        jtype = builder.joint_type[j]
        if jtype == 0:  # FIXED — skip
            continue
        matched = False
        for pattern, ke, kd, armature in ACTUATOR_MAP:
            if pattern in label:
                builder.joint_target_ke[qd_start] = ke
                builder.joint_target_kd[qd_start] = kd
                builder.joint_armature[qd_start] = armature
                builder.joint_target_mode[qd_start] = int(JointTargetMode.POSITION)
                matched = True
                break
        if not matched:
            builder.joint_target_ke[qd_start] = DEFAULT_KE
            builder.joint_target_kd[qd_start] = DEFAULT_KD
            builder.joint_armature[qd_start] = DEFAULT_ARMATURE
            builder.joint_target_mode[qd_start] = int(JointTargetMode.POSITION)


def _apply_initial_pose(builder):
    """Apply initial joint positions."""
    for j in range(builder.joint_count):
        q_start = builder.joint_q_start[j]
        label = builder.joint_label[j]
        for pattern, value in INIT_JOINT_POS.items():
            if pattern in label:
                builder.joint_q[q_start] = value
                break


def _apply_effort_limits(model):
    """Apply effort limits from config."""
    eff = model.joint_effort_limit.numpy()
    n_eff = len(eff)
    for j in range(model.joint_count):
        qd_start = int(model.joint_qd_start.numpy()[j])
        if qd_start >= n_eff:
            continue
        label = str(model.joint_label[j])
        for pattern, limit in EFFORT_LIMITS:
            if pattern == "" or pattern in label:
                eff[qd_start] = limit
                break
    model.joint_effort_limit.assign(
        wp.array(eff, dtype=wp.float32, device=model.device)
    )


def _apply_armature_override(model, overrides: dict[str, float]):
    """Apply armature overrides to specific joints (for DVI stability)."""
    arm = model.joint_armature.numpy()
    n = len(arm)
    changed = 0
    for j in range(model.joint_count):
        qd_start = int(model.joint_qd_start.numpy()[j])
        if qd_start >= n:
            continue
        label = str(model.joint_label[j])
        for pattern, val in overrides.items():
            if pattern in label:
                arm[qd_start] = val
                changed += 1
                break
    model.joint_armature.assign(
        wp.array(arm, dtype=wp.float32, device=model.device)
    )
    print(f"  Armature override applied to {changed} joints")


def _get_default_targets(model):
    """Get default joint position targets (initial pose, DOF-indexed)."""
    jq = model.joint_q.numpy()
    joint_type = model.joint_type.numpy()
    joint_q_start = model.joint_q_start.numpy()
    joint_qd_start = model.joint_qd_start.numpy()
    n_dofs = model.joint_dof_count
    targets = np.zeros(n_dofs, dtype=np.float32)
    for j in range(model.joint_count):
        qs = joint_q_start[j]
        qds = joint_qd_start[j]
        jt = joint_type[j]
        if jt == 1:  # REVOLUTE
            targets[qds] = jq[qs]
    return targets


def _get_action_scales(model):
    """Get per-DOF action scale matching Isaac Lab G1FlatEnvCfg."""
    n_dofs = model.joint_dof_count
    scales = np.ones(n_dofs, dtype=np.float32) * 0.5  # default scale
    joint_type = model.joint_type.numpy()
    joint_qd_start = model.joint_qd_start.numpy()
    for j in range(model.joint_count):
        jt = joint_type[j]
        if jt == 0:  # FIXED
            continue
        qds = joint_qd_start[j]
        if qds >= n_dofs:
            continue
        label = str(model.joint_label[j])
        for pattern, scale_val in ACTION_SCALE.items():
            if pattern in label:
                scales[qds] = scale_val
                break
    return scales


def build_model(device: str = "cuda:0"):
    """Build fixed-base G1 model from USD."""
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

    asset_path = newton.utils.download_asset("unitree_g1")
    asset_file = str(
        asset_path / "usd/g1_minimal.usd"
    )
    xform = wp.transform(wp.vec3(0, 0, SPAWN_HEIGHT))
    single.add_usd(
        asset_file,
        xform=xform,
        floating=False,  # <-- FIXED base
        collapse_fixed_joints=True,
        enable_self_collisions=False,
        hide_collision_shapes=True,
        skip_mesh_approximation=True,
    )

    _apply_actuator_gains(single)
    _apply_initial_pose(single)
    single.approximate_meshes("bounding_box")

    builder = newton.ModelBuilder()
    builder.replicate(single, 1)
    builder.default_shape_cfg.ke = 1.0e3
    builder.default_shape_cfg.kd = 2.0e2
    builder.add_ground_plane()

    model = builder.finalize(device=device)
    model.rigid_contact_max = 65536

    _apply_effort_limits(model)

    print(
        f"  G1 fixed-base: {model.body_count} bodies, {model.joint_count} joints, "
        f"{model.joint_dof_count} DOFs"
    )
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# Action generation
# ═══════════════════════════════════════════════════════════════════════════════
def generate_actions_sinusoidal(n_frames, n_dofs, seed=42):
    """Generate sinusoidal actions in [-1, 1] per joint with random freq/phase."""
    rng = np.random.default_rng(seed)
    freqs = rng.uniform(0.3, 2.0, size=n_dofs)       # Hz
    phases = rng.uniform(0, 2 * np.pi, size=n_dofs)
    amplitudes = rng.uniform(0.3, 1.0, size=n_dofs)
    times = np.arange(n_frames) * PHYSICS_DT
    # shape: (n_frames, n_dofs)
    actions = amplitudes[None, :] * np.sin(
        2 * np.pi * freqs[None, :] * times[:, None] + phases[None, :]
    )
    return actions.astype(np.float32)


def generate_actions_step(n_frames, n_dofs, seed=42):
    """Generate step inputs: random target held for 0.5-1.0 s chunks."""
    rng = np.random.default_rng(seed)
    actions = np.zeros((n_frames, n_dofs), dtype=np.float32)
    i = 0
    while i < n_frames:
        chunk = rng.integers(int(0.5 / PHYSICS_DT), int(1.0 / PHYSICS_DT))
        end = min(i + chunk, n_frames)
        targets = rng.uniform(-0.8, 0.8, size=n_dofs).astype(np.float32)
        actions[i:end] = targets
        i = end
    return actions


# ═══════════════════════════════════════════════════════════════════════════════
# Solver factories
# ═══════════════════════════════════════════════════════════════════════════════
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


def create_dvi_solver(model):
    """Create DVI solver matching IsaacLab flat_env_cfg.DVISolverCfg exactly."""
    # Joint solver: sparse LDL with iterative refinement
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
    # Contact solver: sparse Jacobi
    cc = NumericalSolverConfig(
        solver_type=SolverType.SPARSE_JACOBI,
        max_iterations=40,
        omega=0.3,
        relax=0.9,
        alpha=0.0,
        recovery_speed=10000.0,
        reg=1e-4,
        block_precondition=False,
        friction_projection=FrictionProjection.TANGENTIAL,
    )
    # Joint limit solver: sparse Jacobi (IsaacLab defaults)
    jlc = NumericalSolverConfig(
        solver_type=SolverType.SPARSE_JACOBI,
        max_iterations=10,
        omega=0.3,
        relax=0.9,
        alpha=0.0,
        recovery_speed=10.0,
        reg=1e-8,
    )
    solver = newton.solvers.SolverDVI(
        model,
        joint_solver=jc,
        contact_solver=cc,
        angular_damping=0.0,
        enable_contacts=False,  # fixed base, no ground contact needed
        enable_timers=False,
        actuator_integration=ActuatorIntegration.SEMI_IMPLICIT,
        joint_limit_ke_scale=0.1,
        joint_limit_solver=jlc,
    )
    pipeline = CollisionPipeline(model, broad_phase="explicit")
    contacts = model.contacts()
    return solver, contacts, pipeline


# ═══════════════════════════════════════════════════════════════════════════════
# Simulation
# ═══════════════════════════════════════════════════════════════════════════════
def run_config(
    config_name: str,
    model,
    actions: np.ndarray,
    default_targets: np.ndarray,
    action_scales: np.ndarray,
    apply_armature_override: bool = False,
):
    """Run a single solver config and return per-frame joint positions."""
    n_frames = actions.shape[0]
    n_dofs = model.joint_dof_count
    is_mujoco = config_name == "mujoco"

    if apply_armature_override and not is_mujoco:
        _apply_armature_override(model, DVI_ARMATURE_OVERRIDE)

    state0 = model.state()
    state1 = model.state()
    control = model.control()
    eval_fk(model, model.joint_q, model.joint_qd, state0)

    # Set initial targets = default pose
    tp = control.joint_target_pos.numpy()
    tp[:n_dofs] = default_targets
    control.joint_target_pos.assign(
        wp.array(tp, dtype=wp.float32, device=model.device)
    )

    pipeline = None
    if is_mujoco:
        solver, contacts = create_mujoco_solver(model)
    else:
        solver, contacts, pipeline = create_dvi_solver(model)

    # Storage for trajectories
    joint_q_traj = []  # per-frame joint positions
    joint_qd_traj = []  # per-frame joint velocities

    t0 = time.perf_counter()
    for frame_i in range(n_frames):
        # ── Apply Isaac Lab-style action ──────────────────────────────
        # target = raw_action * scale + default_pos
        raw_action = actions[frame_i]
        target = raw_action * action_scales + default_targets
        tp = control.joint_target_pos.numpy()
        tp[:n_dofs] = target
        control.joint_target_pos.assign(
            wp.array(tp, dtype=wp.float32, device=model.device)
        )

        # ── Physics substeps ──────────────────────────────────────────
        n_subs = DVI_NUM_SUBSTEPS if not is_mujoco else NUM_SUBSTEPS
        dt_sub = DVI_SIM_DT if not is_mujoco else SIM_DT
        for sub in range(n_subs):
            state0.clear_forces()
            if pipeline is not None:
                pipeline.collide(state0, contacts)
            else:
                model.collide(state0, contacts)
            solver.step(state0, state1, control, contacts, dt_sub)
            if not is_mujoco:
                eval_ik(model, state1, state1.joint_q, state1.joint_qd)
            state0, state1 = state1, state0

        # Record joint state
        if frame_i % 2 == 0:  # every other frame to save memory
            wp.synchronize()
            jq = state0.joint_q.numpy()[:n_dofs].copy()
            jqd = state0.joint_qd.numpy()[:n_dofs].copy()
            joint_q_traj.append(jq)
            joint_qd_traj.append(jqd)

            if not np.isfinite(jq).all() or not np.isfinite(jqd).all():
                print(f"  [{config_name}] NaN at frame {frame_i}!")
                break

    wp.synchronize()
    t_sim = time.perf_counter() - t0

    return {
        "config": config_name,
        "joint_q": np.array(joint_q_traj),
        "joint_qd": np.array(joint_qd_traj),
        "sim_time_s": t_sim,
        "n_frames": n_frames,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Video recording (side-by-side)
# ═══════════════════════════════════════════════════════════════════════════════
def record_single_video(
    config_name: str,
    model,
    actions: np.ndarray,
    default_targets: np.ndarray,
    action_scales: np.ndarray,
    output_path: str,
    apply_armature_override: bool = False,
):
    """Record video for a single solver config."""
    from OpenGL import GL as gl

    W, H = 1280, 720
    n_frames = actions.shape[0]
    n_dofs = model.joint_dof_count
    is_mujoco = config_name == "mujoco"

    if apply_armature_override and not is_mujoco:
        _apply_armature_override(model, DVI_ARMATURE_OVERRIDE)

    state0 = model.state()
    state1 = model.state()
    control = model.control()
    eval_fk(model, model.joint_q, model.joint_qd, state0)

    tp = control.joint_target_pos.numpy()
    tp[:n_dofs] = default_targets
    control.joint_target_pos.assign(
        wp.array(tp, dtype=wp.float32, device=model.device)
    )

    pipeline = None
    if is_mujoco:
        solver, contacts = create_mujoco_solver(model)
    else:
        solver, contacts, pipeline = create_dvi_solver(model)

    try:
        viewer = newton.viewer.ViewerGL(headless=True, width=W, height=H)
    except Exception as e:
        print(f"  ViewerGL failed: {e}")
        return None

    viewer.set_model(model)
    # Zoom in on the robot (fixed base at origin, pelvis at z=0.74)
    from pyglet.math import Vec3 as PVec3
    viewer.camera.pos = PVec3(3.0, -2.0, 1.8)
    viewer.camera.look_at(PVec3(0.0, 0.0, 0.6))

    ffmpeg_proc = subprocess.Popen(
        [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{W}x{H}", "-r", str(SIM_FPS),
            "-i", "-",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-crf", "18", output_path,
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    sim_time_val = 0.0
    for frame_i in range(n_frames):
        raw_action = actions[frame_i]
        target = raw_action * action_scales + default_targets
        tp = control.joint_target_pos.numpy()
        tp[:n_dofs] = target
        control.joint_target_pos.assign(
            wp.array(tp, dtype=wp.float32, device=model.device)
        )

        n_subs = DVI_NUM_SUBSTEPS if not is_mujoco else NUM_SUBSTEPS
        dt_sub = DVI_SIM_DT if not is_mujoco else SIM_DT
        for sub in range(n_subs):
            state0.clear_forces()
            if pipeline is not None:
                pipeline.collide(state0, contacts)
            else:
                model.collide(state0, contacts)
            solver.step(state0, state1, control, contacts, dt_sub)
            if not is_mujoco:
                eval_ik(model, state1, state1.joint_q, state1.joint_qd)
            state0, state1 = state1, state0
        sim_time_val += PHYSICS_DT

        viewer.begin_frame(sim_time_val)
        viewer.log_state(state0)
        viewer.end_frame()

        fbo = viewer.renderer._frame_fbo
        gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, fbo)
        pixels = gl.glReadPixels(0, 0, W, H, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
        gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, 0)
        frame_np = np.frombuffer(pixels, dtype=np.uint8).reshape(H, W, 3)[::-1].copy()
        ffmpeg_proc.stdin.write(frame_np.tobytes())

        if frame_i % 50 == 0 and frame_i > 0:
            wp.synchronize()
            jq = state0.joint_q.numpy()
            if not np.isfinite(jq).all():
                print(f"  [{config_name}] NaN at frame {frame_i}")
                break

    ffmpeg_proc.stdin.close()
    ffmpeg_proc.wait()
    viewer.close()
    return output_path if os.path.exists(output_path) else None


def stitch_side_by_side(left_path: str, right_path: str, output_path: str,
                        left_label: str = "MuJoCo", right_label: str = "DVI"):
    """Stitch two videos side by side with labels using ffmpeg."""
    cmd = [
        "ffmpeg", "-y",
        "-i", left_path,
        "-i", right_path,
        "-filter_complex",
        f"[0:v]drawtext=text='{left_label}':fontsize=36:fontcolor=white:"
        f"borderw=2:bordercolor=black:x=20:y=20[left];"
        f"[1:v]drawtext=text='{right_label}':fontsize=36:fontcolor=white:"
        f"borderw=2:bordercolor=black:x=20:y=20[right];"
        f"[left][right]hstack=inputs=2",
        "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        print(f"  ffmpeg stitch failed: {result.stderr[-200:]}")
        return None
    return output_path


# ═══════════════════════════════════════════════════════════════════════════════
# Joint trajectory comparison metrics
# ═══════════════════════════════════════════════════════════════════════════════
def print_comparison(res_mj, res_dvi, model):
    """Print per-joint position error statistics between MuJoCo and DVI."""
    q_mj = res_mj["joint_q"]
    q_dvi = res_dvi["joint_q"]
    n = min(len(q_mj), len(q_dvi))
    q_mj = q_mj[:n]
    q_dvi = q_dvi[:n]

    diff = q_mj - q_dvi
    rms_per_joint = np.sqrt(np.mean(diff ** 2, axis=0))
    max_per_joint = np.max(np.abs(diff), axis=0)

    # Get joint labels
    joint_type = model.joint_type.numpy()
    joint_qd_start = model.joint_qd_start.numpy()
    n_dofs = model.joint_dof_count
    labels = [""] * n_dofs
    for j in range(model.joint_count):
        jt = joint_type[j]
        if jt == 0:  # FIXED
            continue
        qds = joint_qd_start[j]
        if qds < n_dofs:
            labels[qds] = str(model.joint_label[j])

    print(f"\n{'='*80}")
    print(f"  Joint Position Error: MuJoCo vs DVI ({n} frames)")
    print(f"{'='*80}")
    print(f"  {'Joint':<40} {'RMS (rad)':>10} {'Max (rad)':>10}")
    print(f"  {'-'*60}")

    total_rms = 0.0
    active = 0
    for i in range(n_dofs):
        if rms_per_joint[i] > 1e-10:
            flag = " ⚠️" if rms_per_joint[i] > 0.1 else ""
            name = labels[i] if labels[i] else f"dof_{i}"
            print(f"  {name:<40} {rms_per_joint[i]:>10.4f} {max_per_joint[i]:>10.4f}{flag}")
            total_rms += rms_per_joint[i] ** 2
            active += 1

    if active > 0:
        avg_rms = np.sqrt(total_rms / active)
        print(f"  {'-'*60}")
        print(f"  {'Average RMS':<40} {avg_rms:>10.4f}")
        print(f"  {'Max joint error':<40} {np.max(max_per_joint):>10.4f}")

    print(f"\n  MuJoCo sim time: {res_mj['sim_time_s']:.2f}s")
    print(f"  DVI sim time:    {res_dvi['sim_time_s']:.2f}s")
    print(f"  Speedup:         {res_mj['sim_time_s'] / max(res_dvi['sim_time_s'], 1e-6):.2f}x")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Fixed-base G1 benchmark: MuJoCo vs DVI side-by-side"
    )
    parser.add_argument("--sim-time", type=float, default=3.0)
    parser.add_argument(
        "--action-type", choices=["sinusoidal", "step"], default="sinusoidal",
        help="Action waveform type (default: sinusoidal)",
    )
    parser.add_argument("--no-video", action="store_true", help="Skip video recording")
    parser.add_argument("--armature-override", action="store_true",
                        help="Apply armature overrides on arm/hand joints for DVI")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    # Subprocess helpers (hidden)
    parser.add_argument("--_record", type=str, help=argparse.SUPPRESS)
    args = parser.parse_args()

    wp.init()
    wp.set_device(args.device)

    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(__file__), "fixed_g1_results")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Building fixed-base G1 model ...", flush=True)
    model = build_model(args.device)

    n_dofs = model.joint_dof_count
    default_targets = _get_default_targets(model)
    action_scales = _get_action_scales(model)

    # Print action config summary
    print(f"\n  Action config ({args.action_type}):")
    joint_type = model.joint_type.numpy()
    joint_qd_start = model.joint_qd_start.numpy()
    actuated = 0
    for i in range(n_dofs):
        if action_scales[i] > 0:
            actuated += 1
    print(f"    {actuated}/{n_dofs} actuated DOFs (fingers have scale=0)")

    # Generate actions
    n_frames = int(args.sim_time / PHYSICS_DT)
    if args.action_type == "sinusoidal":
        actions = generate_actions_sinusoidal(n_frames, n_dofs, args.seed)
    else:
        actions = generate_actions_step(n_frames, n_dofs, args.seed)

    # Use IsaacLab default joint positions as targets (use_default_offset=True)
    # default_targets already has the correct per-joint initial pose from _get_default_targets()
    # Actions are sinusoidal noise scaled by ACTION_SCALE
    print(f"  Using IsaacLab default targets: {default_targets}")

    # ── Subprocess video mode ────────────────────────────────────────
    if args._record:
        vp = record_single_video(
            args._record, model, actions, default_targets, action_scales,
            os.path.join(args.output_dir, f"{args._record}.mp4"),
            apply_armature_override=args.armature_override,
        )
        if vp:
            print(f"VIDEO_PATH={vp}")
        return

    # ── Benchmark both solvers ───────────────────────────────────────
    print(f"\n  Running MuJoCo ...", flush=True)
    res_mj = run_config("mujoco", model, actions, default_targets, action_scales)
    print(f"    Done in {res_mj['sim_time_s']:.2f}s")

    # Rebuild model for DVI (clean state)
    model = build_model(args.device)

    print(f"  Running DVI ...", flush=True)
    res_dvi = run_config(
        "dvi", model, actions, default_targets, action_scales,
        apply_armature_override=args.armature_override,
    )
    print(f"    Done in {res_dvi['sim_time_s']:.2f}s")

    # ── Comparison ───────────────────────────────────────────────────
    print_comparison(res_mj, res_dvi, model)

    # ── Video ────────────────────────────────────────────────────────
    if not args.no_video:
        print(f"\nRecording side-by-side video ...", flush=True)

        for cfg in ["mujoco", "dvi"]:
            print(f"  Recording {cfg} ...", flush=True)
            cmd = [
                sys.executable, os.path.abspath(__file__),
                "--sim-time", str(args.sim_time),
                "--action-type", args.action_type,
                "--device", args.device,
                "--output-dir", args.output_dir,
                "--seed", str(args.seed),
                "--_record", cfg,
            ]
            if args.armature_override:
                cmd.append("--armature-override")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            vp = os.path.join(args.output_dir, f"{cfg}.mp4")
            if os.path.exists(vp):
                size_kb = os.path.getsize(vp) / 1024
                print(f"    ✓ {vp} ({size_kb:.0f} KB)")
            else:
                print(f"    ✗ Failed")
                if result.stderr:
                    for line in result.stderr.strip().split("\n")[-5:]:
                        print(f"      {line}")

        # Stitch
        left = os.path.join(args.output_dir, "mujoco.mp4")
        right = os.path.join(args.output_dir, "dvi.mp4")
        combined = os.path.join(args.output_dir, "mujoco_vs_dvi.mp4")
        if os.path.exists(left) and os.path.exists(right):
            print(f"  Stitching side-by-side ...", flush=True)
            result = stitch_side_by_side(left, right, combined)
            if result:
                size_kb = os.path.getsize(combined) / 1024
                print(f"    ✓ {combined} ({size_kb:.0f} KB)")
            else:
                print(f"    ✗ Stitch failed")

    print("\nDone.")


if __name__ == "__main__":
    main()
