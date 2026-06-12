#!/usr/bin/env python3
"""
Fixed-base H1 benchmark: MuJoCo vs DVI side-by-side.

The H1 humanoid is fixed to the world (no free joint) and driven with
Isaac Lab-style implicit PD actuation:

    target = raw_action * scale + default_joint_pos

Random sinusoidal actions are generated per joint, scaled identically to
the Isaac Lab H1 flat env, then applied as position targets.

Both solvers share the same model (built once) and see the same action
sequence, so any difference in motion is purely solver behavior.

Usage:
    python benchmark_fixed_h1.py                    # side-by-side video
    python benchmark_fixed_h1.py --no-video         # metrics only
    python benchmark_fixed_h1.py --sim-time 5       # 5 second sim
    python benchmark_fixed_h1.py --action-type step # step input
    python benchmark_fixed_h1.py --num-envs 64      # multi-env NaN test
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
# Simulation parameters (matching Isaac Lab H1 DVI config)
# ═══════════════════════════════════════════════════════════════════════════════
PHYSICS_DT = 0.005  # 200 Hz physics step (Isaac Lab default)
NUM_SUBSTEPS_MUJOCO = 1
NUM_SUBSTEPS_DVI = 4  # Match H1 flat_env_cfg newton_dvi preset
SIM_FPS = 50  # video frame rate

# ═══════════════════════════════════════════════════════════════════════════════
# Isaac Lab H1 actuator config (from isaaclab_assets/robots/unitree.py H1_CFG)
# ═══════════════════════════════════════════════════════════════════════════════
# (pattern, ke, kd, armature)
ACTUATOR_MAP = [
    # Legs
    ("hip_pitch_joint", 200.0, 5.0, 0.1),
    ("hip_roll_joint", 150.0, 5.0, 0.1),
    ("hip_yaw_joint", 150.0, 5.0, 0.1),
    ("knee_joint", 200.0, 5.0, 0.1),
    ("torso_joint", 200.0, 5.0, 0.1),
    # Feet
    ("ankle_joint", 20.0, 4.0, 0.1),
    # Arms
    ("shoulder_pitch_joint", 40.0, 10.0, 0.1),
    ("shoulder_roll_joint", 40.0, 10.0, 0.1),
    ("shoulder_yaw_joint", 40.0, 10.0, 0.1),
    ("elbow_joint", 40.0, 10.0, 0.1),
]

DEFAULT_KE = 40.0
DEFAULT_KD = 10.0
DEFAULT_ARMATURE = 0.1

# Isaac Lab H1_CFG initial joint positions
INIT_JOINT_POS = {
    "hip_yaw_joint": 0.0,
    "hip_roll_joint": 0.0,
    "hip_pitch_joint": -0.28,
    "knee_joint": 0.79,
    "ankle_joint": -0.52,
    "torso_joint": 0.0,
    "shoulder_pitch_joint": 0.28,
    "shoulder_roll_joint": 0.0,
    "shoulder_yaw_joint": 0.0,
    "elbow_joint": 0.52,
}

# Isaac Lab action scale (all joints use 0.5)
ACTION_SCALE_DEFAULT = 0.5

# Joint limits from H1 MJCF (lower, upper) in radians
JOINT_LIMITS = {
    "left_hip_yaw_joint": (-0.43, 0.43),
    "left_hip_roll_joint": (-0.43, 0.43),
    "left_hip_pitch_joint": (-3.14, 2.53),
    "left_knee_joint": (-0.26, 2.05),
    "left_ankle_joint": (-0.87, 0.52),
    "right_hip_yaw_joint": (-0.43, 0.43),
    "right_hip_roll_joint": (-0.43, 0.43),
    "right_hip_pitch_joint": (-3.14, 2.53),
    "right_knee_joint": (-0.26, 2.05),
    "right_ankle_joint": (-0.87, 0.52),
    "torso_joint": (-2.35, 2.35),
    "left_shoulder_pitch_joint": (-2.87, 2.87),
    "left_shoulder_roll_joint": (-0.34, 3.11),
    "left_shoulder_yaw_joint": (-1.3, 4.45),
    "left_elbow_joint": (-1.25, 2.61),
    "right_shoulder_pitch_joint": (-2.87, 2.87),
    "right_shoulder_roll_joint": (-3.11, 0.34),
    "right_shoulder_yaw_joint": (-4.45, 1.3),
    "right_elbow_joint": (-1.25, 2.61),
}

# Effort limits from H1_CFG
EFFORT_LIMITS = [
    ("ankle", 100.0),
    ("", 300.0),  # default
]

SPAWN_HEIGHT = 1.05  # H1 standing height


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
        if jtype == 0:  # FIXED
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
    """Apply effort limits."""
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
    """Get per-DOF action scale. All H1 joints use 0.5."""
    n_dofs = model.joint_dof_count
    scales = np.ones(n_dofs, dtype=np.float32) * ACTION_SCALE_DEFAULT
    return scales


def _get_mid_joint_targets(model):
    """Get mid-joint-limit targets (midpoint of lower and upper limit)."""
    n_dofs = model.joint_dof_count
    targets = np.zeros(n_dofs, dtype=np.float32)
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
        for name, (lo, hi) in JOINT_LIMITS.items():
            if name in label:
                targets[qds] = (lo + hi) / 2.0
                break
    return targets


def build_model(num_envs: int = 1, device: str = "cuda:0",
                enable_contacts: bool = False):
    """Build fixed-base H1 model from MJCF."""
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

    asset_path = newton.utils.download_asset("unitree_h1")
    asset_file = str(asset_path / "mjcf" / "h1.xml")

    xform = wp.transform(wp.vec3(0, 0, SPAWN_HEIGHT))
    single.add_mjcf(
        asset_file,
        xform=xform,
        ignore_names=["floor", "ground"],
        parse_sites=False,
        base_joint={
            "joint_type": newton.JointType.FIXED,
        },
    )

    _apply_actuator_gains(single)
    _apply_initial_pose(single)

    builder = newton.ModelBuilder()
    builder.replicate(single, num_envs)

    if enable_contacts:
        builder.default_shape_cfg.ke = 1.0e3
        builder.default_shape_cfg.kd = 2.0e2
        builder.add_ground_plane()
    else:
        # Disable collisions
        for j in range(len(builder.shape_collision_group)):
            builder.shape_collision_group[j] = -1

    model = builder.finalize(device=device)
    model.rigid_contact_max = 65536

    _apply_effort_limits(model)

    print(
        f"  H1 fixed-base: {model.body_count} bodies, {model.joint_count} joints, "
        f"{model.joint_dof_count} DOFs, {num_envs} envs"
    )

    # Print body inertia summary
    mass = model.body_mass.numpy()
    inertia = model.body_inertia.numpy()
    bpe = model.body_count // max(num_envs, 1)
    for i in range(bpe):
        I = inertia[i]
        d = [I[0, 0], I[1, 1], I[2, 2]]
        min_I = min(d)
        cond = max(d) / (min_I + 1e-20)
        print(f"    body {i}: mass={mass[i]:.4f}  I_min={min_I:.6f}  cond={cond:.0f}")

    return model, builder


# ═══════════════════════════════════════════════════════════════════════════════
# Action generation
# ═══════════════════════════════════════════════════════════════════════════════
def generate_actions_sinusoidal(n_frames, n_dofs, num_envs=1, seed=42):
    """Generate sinusoidal actions in [-1, 1] per joint."""
    rng = np.random.default_rng(seed)
    freqs = rng.uniform(0.3, 2.0, size=n_dofs)
    phases = rng.uniform(0, 2 * np.pi, size=n_dofs)
    amplitudes = rng.uniform(0.3, 1.0, size=n_dofs)
    times = np.arange(n_frames) * PHYSICS_DT
    actions = amplitudes[None, :] * np.sin(
        2 * np.pi * freqs[None, :] * times[:, None] + phases[None, :]
    )
    return actions.astype(np.float32)


def generate_actions_random(n_frames, n_dofs, num_envs=1, seed=42):
    """Generate N(0,1) random actions (simulates untrained policy)."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_frames, n_dofs)).astype(np.float32)


def generate_actions_mid(n_frames, n_dofs, num_envs=1, seed=42):
    """Generate zero actions (mid-joint targets are set via default_targets)."""
    return np.zeros((n_frames, n_dofs), dtype=np.float32)


def generate_actions_step(n_frames, n_dofs, num_envs=1, seed=42):
    """Generate step inputs held for 0.5-1.0 s chunks."""
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
        iterations=1,
        ls_iterations=4,
        disable_contacts=True,
    )
    contacts = model.contacts()
    return solver, contacts, NUM_SUBSTEPS_MUJOCO


def create_dvi_solver(model):
    jc = NumericalSolverConfig(
        solver_type=SolverType.SPARSE_LDL,
        max_iterations=50,
        omega=0.3,
        relax=0.8,
        alpha=0.0,
        recovery_speed=100000.0,
        reg=1e-6,
    )
    solver = newton.solvers.SolverDVI(
        model,
        joint_solver=jc,
        contact_solver=None,
        angular_damping=0.0,
        enable_contacts=False,
        enable_timers=False,
        actuator_integration=ActuatorIntegration.SEMI_IMPLICIT,
    )
    return solver, None, NUM_SUBSTEPS_DVI


# ═══════════════════════════════════════════════════════════════════════════════
# Simulation
# ═══════════════════════════════════════════════════════════════════════════════
def run_config(solver_name, model, actions, default_targets, action_scales,
               num_envs=1):
    """Run simulation and return results dict."""
    if solver_name == "mujoco":
        solver, contacts, substeps = create_mujoco_solver(model)
    else:
        solver, contacts, substeps = create_dvi_solver(model)

    sim_dt = PHYSICS_DT / substeps
    n_frames = actions.shape[0]
    n_dofs_per_env = actions.shape[1]
    total_dofs = model.joint_dof_count

    state0 = model.state()
    state1 = model.state()
    control = model.control()
    eval_fk(model, model.joint_q, model.joint_qd, state0)

    # Storage
    joint_q_history = []
    joint_qd_history = []
    nan_frames = []

    # Warmup
    for _ in range(min(5, n_frames)):
        for _ in range(substeps):
            state0.clear_forces()
            if contacts is not None:
                model.collide(state0, contacts)
            solver.step(state0, state1, control, contacts, sim_dt)
            state0, state1 = state1, state0

    # Reset state
    state0 = model.state()
    state1 = model.state()
    eval_fk(model, model.joint_q, model.joint_qd, state0)

    wp.synchronize()
    t0 = time.perf_counter()

    for frame in range(n_frames):
        # Apply actions: target = action * scale + default
        raw = actions[frame]  # (n_dofs_per_env,)
        targets = raw * action_scales + default_targets

        # Tile targets across envs
        all_targets = np.tile(targets, num_envs)
        control.joint_target_pos.assign(
            wp.array(all_targets, dtype=wp.float32, device=model.device)
        )

        for _ in range(substeps):
            state0.clear_forces()
            if contacts is not None:
                model.collide(state0, contacts)
            solver.step(state0, state1, control, contacts, sim_dt)
            state0, state1 = state1, state0

        # Sync for DVI
        if solver_name == "dvi":
            eval_ik(model, state0, state0.joint_q, state0.joint_qd)

        jq = state0.joint_q.numpy().copy()
        jqd = state0.joint_qd.numpy().copy()
        joint_q_history.append(jq[:n_dofs_per_env])
        joint_qd_history.append(jqd[:n_dofs_per_env])

        # NaN check
        if np.isnan(jq).any() or np.isnan(jqd).any():
            nan_frames.append(frame)
            if len(nan_frames) <= 3:
                nan_dofs_q = np.where(np.isnan(jq[:n_dofs_per_env]))[0]
                nan_dofs_qd = np.where(np.isnan(jqd[:n_dofs_per_env]))[0]
                print(f"    *** NaN at frame {frame}: "
                      f"joint_q NaN dofs={nan_dofs_q.tolist()}, "
                      f"joint_qd NaN dofs={nan_dofs_qd.tolist()}")

    wp.synchronize()
    t_sim = time.perf_counter() - t0

    joint_q_arr = np.array(joint_q_history)  # (n_frames, n_dofs_per_env)
    joint_qd_arr = np.array(joint_qd_history)

    return {
        "solver": solver_name,
        "joint_q": joint_q_arr,
        "joint_qd": joint_qd_arr,
        "sim_time_s": t_sim,
        "nan_frames": nan_frames,
        "substeps": substeps,
    }


def print_comparison(res_mj, res_dvi, model):
    """Print comparison metrics."""
    print(f"\n{'='*70}")
    print(f"  Fixed-Base H1 — MuJoCo vs DVI Comparison")
    print(f"{'='*70}")

    n_dofs = res_mj["joint_q"].shape[1]
    n_frames = min(res_mj["joint_q"].shape[0], res_dvi["joint_q"].shape[0])

    q_mj = res_mj["joint_q"][:n_frames]
    q_dvi = res_dvi["joint_q"][:n_frames]
    qd_mj = res_mj["joint_qd"][:n_frames]
    qd_dvi = res_dvi["joint_qd"][:n_frames]

    # Mask NaN
    valid = ~np.isnan(q_mj) & ~np.isnan(q_dvi)
    if valid.any():
        q_diff = np.abs(q_mj - q_dvi)
        q_diff[~valid] = 0
        print(f"\n  Position error (mean per DOF):")
        for d in range(n_dofs):
            v = valid[:, d]
            if v.any():
                mean_err = np.mean(q_diff[:, d][v])
                max_err = np.max(q_diff[:, d][v])
                print(f"    DOF {d:2d}: mean={mean_err:.6f}  max={max_err:.6f}")

    valid_qd = ~np.isnan(qd_mj) & ~np.isnan(qd_dvi)
    if valid_qd.any():
        qd_diff = np.abs(qd_mj - qd_dvi)
        qd_diff[~valid_qd] = 0
        print(f"\n  Velocity error (mean per DOF):")
        for d in range(n_dofs):
            v = valid_qd[:, d]
            if v.any():
                mean_err = np.mean(qd_diff[:, d][v])
                max_err = np.max(qd_diff[:, d][v])
                print(f"    DOF {d:2d}: mean={mean_err:.4f}  max={max_err:.4f}")

    print(f"\n  Timing:")
    print(f"    MuJoCo: {res_mj['sim_time_s']:.3f}s ({res_mj['substeps']} substeps)")
    print(f"    DVI:    {res_dvi['sim_time_s']:.3f}s ({res_dvi['substeps']} substeps)")

    print(f"\n  NaN frames:")
    print(f"    MuJoCo: {len(res_mj['nan_frames'])} ({res_mj['nan_frames'][:5]})")
    print(f"    DVI:    {len(res_dvi['nan_frames'])} ({res_dvi['nan_frames'][:5]})")
    print(f"{'='*70}")


# ═══════════════════════════════════════════════════════════════════════════════
# Video recording
# ═══════════════════════════════════════════════════════════════════════════════
def record_single_video(solver_name, model, actions, default_targets,
                        action_scales, output_path, num_envs=1):
    """Record video using ViewerGL + OpenGL readback."""
    import OpenGL.GL as gl

    W, H = 1280, 720
    try:
        viewer = newton.viewer.ViewerGL(headless=True, width=W, height=H)
    except Exception as e:
        print(f"  ViewerGL failed: {e}")
        return None

    if solver_name == "mujoco":
        solver, contacts, substeps = create_mujoco_solver(model)
    else:
        solver, contacts, substeps = create_dvi_solver(model)

    sim_dt = PHYSICS_DT / substeps
    n_dofs_per_env = actions.shape[1]

    state0 = model.state()
    state1 = model.state()
    control = model.control()
    eval_fk(model, model.joint_q, model.joint_qd, state0)

    viewer.set_model(model)
    # Camera looking at H1 from the side
    from pyglet.math import Vec3 as PVec3
    viewer.camera.pos = PVec3(3.5, -2.5, 2.0)
    viewer.camera.look_at(PVec3(0.0, 0.0, 0.8))

    n_frames = actions.shape[0]
    video_dt = 1.0 / SIM_FPS
    steps_per_video_frame = max(1, int(video_dt / PHYSICS_DT))

    # Start ffmpeg
    cmd = [
        "ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{W}x{H}", "-pix_fmt", "rgb24", "-r", str(SIM_FPS),
        "-i", "pipe:", "-c:v", "libx264", "-preset", "fast",
        "-crf", "23", "-pix_fmt", "yuv420p", output_path,
    ]
    ffmpeg_proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    sim_time = 0.0
    for phys_frame in range(n_frames):
        raw = actions[phys_frame]
        targets = raw * action_scales + default_targets
        all_targets = np.tile(targets, num_envs)
        control.joint_target_pos.assign(
            wp.array(all_targets, dtype=wp.float32, device=model.device)
        )

        for _ in range(substeps):
            state0.clear_forces()
            if contacts is not None:
                model.collide(state0, contacts)
            solver.step(state0, state1, control, contacts, sim_dt)
            state0, state1 = state1, state0

        if solver_name == "dvi":
            eval_ik(model, state0, state0.joint_q, state0.joint_qd)

        sim_time += PHYSICS_DT

        if phys_frame % steps_per_video_frame == 0:
            viewer.begin_frame(sim_time)
            viewer.log_state(state0)
            viewer.end_frame()

            fbo = viewer.renderer._frame_fbo
            gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, fbo)
            pixels = gl.glReadPixels(0, 0, W, H, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
            gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, 0)
            frame_np = np.frombuffer(pixels, dtype=np.uint8).reshape(H, W, 3)[::-1].copy()
            ffmpeg_proc.stdin.write(frame_np.tobytes())

    ffmpeg_proc.stdin.close()
    ffmpeg_proc.wait()
    viewer.close()

    if os.path.exists(output_path):
        return output_path
    return None


def stitch_side_by_side(left_path, right_path, output_path):
    """Stitch two videos side-by-side."""
    cmd = [
        "ffmpeg", "-y", "-i", left_path, "-i", right_path,
        "-filter_complex",
        "[0:v]drawtext=text='MuJoCo':fontcolor=white:fontsize=24:x=10:y=10[left];"
        "[1:v]drawtext=text='DVI':fontcolor=white:fontsize=24:x=10:y=10[right];"
        "[left][right]hstack[v]",
        "-map", "[v]", "-c:v", "libx264", "-preset", "fast",
        "-crf", "23", "-pix_fmt", "yuv420p", output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    return os.path.exists(output_path)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Fixed-base H1 benchmark: MuJoCo vs DVI."
    )
    parser.add_argument("--sim-time", type=float, default=5.0)
    parser.add_argument("--action-type", choices=["sinusoidal", "step", "random", "mid"],
                        default="sinusoidal")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="fixed_h1_results")
    parser.add_argument("--_record", type=str, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    wp.init()
    wp.set_device(args.device)

    os.makedirs(args.output_dir, exist_ok=True)

    n_frames = int(args.sim_time / PHYSICS_DT)

    # ── Subprocess video mode ────────────────────────────────────────
    if args._record:
        print(f"\n  Recording {args._record} video ...", flush=True)
        model, builder = build_model(args.num_envs, args.device)
        n_dofs_per_env = model.joint_dof_count // max(args.num_envs, 1)
        if args.action_type == "mid":
            default_targets = _get_mid_joint_targets(model)[:n_dofs_per_env]
        else:
            default_targets = _get_default_targets(model)[:n_dofs_per_env]
        action_scales = _get_action_scales(model)[:n_dofs_per_env]

        if args.action_type == "sinusoidal":
            actions = generate_actions_sinusoidal(n_frames, n_dofs_per_env, args.num_envs, args.seed)
        elif args.action_type == "random":
            actions = generate_actions_random(n_frames, n_dofs_per_env, args.num_envs, args.seed)
        elif args.action_type == "mid":
            actions = generate_actions_mid(n_frames, n_dofs_per_env, args.num_envs, args.seed)
        else:
            actions = generate_actions_step(n_frames, n_dofs_per_env, args.num_envs, args.seed)

        vp = record_single_video(
            args._record, model, actions, default_targets, action_scales,
            os.path.join(args.output_dir, f"{args._record}.mp4"),
            num_envs=args.num_envs,
        )
        if vp:
            print(f"VIDEO_PATH={vp}")
        return

    # ── Main mode ────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  Fixed-Base H1 Benchmark")
    print(f"  sim_time={args.sim_time}s, actions={args.action_type}, "
          f"envs={args.num_envs}, seed={args.seed}")
    print(f"{'='*70}")

    # Build model
    print(f"\n  Building model ...", flush=True)
    model, builder = build_model(args.num_envs, args.device)
    n_dofs_per_env = model.joint_dof_count // max(args.num_envs, 1)
    if args.action_type == "mid":
        default_targets = _get_mid_joint_targets(model)[:n_dofs_per_env]
    else:
        default_targets = _get_default_targets(model)[:n_dofs_per_env]
    action_scales = _get_action_scales(model)[:n_dofs_per_env]

    print(f"  DOFs per env: {n_dofs_per_env}")
    print(f"  Default targets: {default_targets}")
    print(f"  Action scales: {action_scales}")

    # Generate actions
    if args.action_type == "sinusoidal":
        actions = generate_actions_sinusoidal(n_frames, n_dofs_per_env, args.num_envs, args.seed)
    elif args.action_type == "random":
        actions = generate_actions_random(n_frames, n_dofs_per_env, args.num_envs, args.seed)
    elif args.action_type == "mid":
        actions = generate_actions_mid(n_frames, n_dofs_per_env, args.num_envs, args.seed)
    else:
        actions = generate_actions_step(n_frames, n_dofs_per_env, args.num_envs, args.seed)

    # Run MuJoCo
    print(f"\n  Running MuJoCo ...", flush=True)
    res_mj = run_config("mujoco", model, actions, default_targets, action_scales, args.num_envs)
    print(f"    Done in {res_mj['sim_time_s']:.2f}s, NaN frames: {len(res_mj['nan_frames'])}")

    # Rebuild model for DVI
    model, builder = build_model(args.num_envs, args.device)

    # Run DVI
    print(f"\n  Running DVI ...", flush=True)
    res_dvi = run_config("dvi", model, actions, default_targets, action_scales, args.num_envs)
    print(f"    Done in {res_dvi['sim_time_s']:.2f}s, NaN frames: {len(res_dvi['nan_frames'])}")

    # Print comparison
    print_comparison(res_mj, res_dvi, model)

    # Record video
    if not args.no_video:
        print(f"\nRecording videos ...", flush=True)

        for cfg in ["mujoco", "dvi"]:
            print(f"  Recording {cfg} ...", flush=True)
            cmd = [
                sys.executable, os.path.abspath(__file__),
                "--sim-time", str(args.sim_time),
                "--action-type", args.action_type,
                "--num-envs", "1",  # always 1 for video
                "--device", args.device,
                "--output-dir", args.output_dir,
                "--seed", str(args.seed),
                "--_record", cfg,
            ]
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
            ok = stitch_side_by_side(left, right, combined)
            if ok:
                size_kb = os.path.getsize(combined) / 1024
                print(f"    ✓ {combined} ({size_kb:.0f} KB)")
            else:
                print(f"    ✗ Stitch failed")

    print("\nDone.")


if __name__ == "__main__":
    main()
