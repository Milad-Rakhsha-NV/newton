#!/usr/bin/env python3
"""
Hanging (fixed-base) DR Legs benchmark for the DVI solver.

The Disney DR Legs closed-loop biped is fixed to the world (pelvis pinned, no
free joint) and driven with Isaac Lab-style implicit PD actuation on its 12
servo joints:

    target = raw_action * scale + default_joint_pos

This mirrors the fixed-base G1/H1 examples (``benchmark_fixed_g1.py`` /
``benchmark_fixed_h1.py``), but DR Legs has 6 closed kinematic loops, so:

  * MuJoCo cannot load the model (no spanning tree -> "No root found in the
    joint graph"), so this benchmark is **DVI-only**.
  * The model is built with ``collapse_fixed_joints=False`` and
    ``skip_validation_joints=True`` (orphan joints, closed loops).
  * Contacts are disabled (fixed base, hanging in the air) so we isolate
    whether the DVI actuators actually move the driven joints while the 24
    passive linkage DOFs follow the loop constraints.

Robot, actuator and DVI-solver parameters are taken from the Isaac Lab DR Legs
walk env (``isaaclab_tasks/contrib/dr_legs``):

  * Actuated joints : DR_LEGS_ACTUATED_JOINTS (12 servos), ke=5.0, kd=0.2,
                      effort_limit=3.1
  * Passive joints  : 24 linkage DOFs, undriven (ke=kd=0)
  * Action          : JointPositionActionCfg, scale=0.3, use_default_offset=True
  * DVI solver      : sparse LDL joints (50 iters, alpha=0, recovery=1e5,
                      reg=1e-4), no position correction
  * dt = 0.004 s, num_substeps = 4

Usage:
    python benchmark_hanging_drlegs.py                 # DVI video + metrics
    python benchmark_hanging_drlegs.py --no-video      # metrics only
    python benchmark_hanging_drlegs.py --sim-time 5    # 5 second sim
    python benchmark_hanging_drlegs.py --action-type step
"""

import argparse
import os
import subprocess
import sys
import time

os.environ.setdefault("WARP_LOG_LEVEL", "warning")

import numpy as np
import warp as wp
import warp.config

warp.config.verbose = False

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import newton
import newton.utils
from newton import CollisionPipeline, JointTargetMode, eval_fk, eval_ik
from newton.solvers import ActuatorIntegration, SolverType
from newton._src.solvers.dvi.numerical_solver.base import NumericalSolverConfig


# ═══════════════════════════════════════════════════════════════════════════════
# Simulation parameters (matching Isaac Lab DR Legs walk env)
# ═══════════════════════════════════════════════════════════════════════════════
PHYSICS_DT = 0.004          # Isaac Lab sim dt
# DR Legs num_substeps (walk-env default 4). Overridable via DRLEGS_SUBSTEPS,
# matching the Isaac Lab env knob.
DVI_NUM_SUBSTEPS = 4        # DR Legs num_substeps (walk-env default); set via --substeps
SIM_FPS = 50                # video frame rate

SPAWN_HEIGHT = 0.28         # Isaac Lab DR Legs init pos z (assembled reference)

# ═══════════════════════════════════════════════════════════════════════════════
# Isaac Lab DR Legs robot config
# (from isaaclab_assets/robots/dr_legs.py + contrib/dr_legs walk env)
# ═══════════════════════════════════════════════════════════════════════════════
# Canonical 36-joint ordering.
DR_LEGS_JOINT_ORDER = [
    "j1_l_i", "j2_l_i", "j3_l_i", "j4_l_i", "j5_l_i", "j6_l_i", "j7_l_i", "j8_l_i", "j9_l_i",
    "j1_l_o", "j2_l_o", "j3_l_o", "j4_l_o", "j5_l_o", "j6_l_o", "j7_l_o", "j8_l_o", "j9_l_o",
    "j1_r_i", "j2_r_i", "j3_r_i", "j4_r_i", "j5_r_i", "j6_r_i", "j7_r_i", "j8_r_i", "j9_r_i",
    "j1_r_o", "j2_r_o", "j3_r_o", "j4_r_o", "j5_r_o", "j6_r_o", "j7_r_o", "j8_r_o", "j9_r_o",
]

# The 12 servo-driven joints (6 per leg).
DR_LEGS_ACTUATED_JOINTS = [
    "j1_l_i", "j2_l_i", "j6_l_i", "j7_l_i", "j2_l_o", "j7_l_o",
    "j1_r_i", "j2_r_i", "j6_r_i", "j7_r_i", "j2_r_o", "j7_r_o",
]

# The 24 undriven closed-loop linkage DOFs.
DR_LEGS_PASSIVE_JOINTS = [j for j in DR_LEGS_JOINT_ORDER if j not in DR_LEGS_ACTUATED_JOINTS]

# Implicit PD gains (ImplicitActuatorCfg on the driven joints).
DRIVEN_KE = 5.0
DRIVEN_KD = 0.2
DRIVEN_EFFORT = 3.1
DRIVEN_ARMATURE = 0.0

# Passive joints: undriven (zero stiffness/damping), large effort limit.
PASSIVE_EFFORT = 400.0

# Action config: JointPositionActionCfg(scale=0.3, use_default_offset=True).
ACTION_SCALE = 0.3

# Initial joint positions: all zero (assembled closed-loop reference pose).
INIT_JOINT_POS = 0.0


def _joint_short_label(full_label: str) -> str:
    """Return the trailing joint name (e.g. ``j1_l_i``) from a USD path label."""
    return full_label.rsplit("/", 1)[-1]


# ═══════════════════════════════════════════════════════════════════════════════
# Model building
# ═══════════════════════════════════════════════════════════════════════════════
def _apply_actuator_gains(model):
    """Apply per-joint stiffness/damping/armature/target-mode from the walk env.

    Driven joints get ke=5.0, kd=0.2 (position mode); passive linkage DOFs get
    zero gains so the DVI solver leaves them to the loop constraints.
    """
    ke = model.joint_target_ke.numpy()
    kd = model.joint_target_kd.numpy()
    arm = model.joint_armature.numpy()
    mode = model.joint_target_mode.numpy()
    qd_start = model.joint_qd_start.numpy()
    n = len(ke)
    actuated_dofs = []
    for j in range(model.joint_count):
        qds = int(qd_start[j])
        if qds >= n:
            continue
        name = _joint_short_label(str(model.joint_label[j]))
        if name in DR_LEGS_ACTUATED_JOINTS:
            ke[qds] = DRIVEN_KE
            kd[qds] = DRIVEN_KD
            arm[qds] = DRIVEN_ARMATURE
            mode[qds] = int(JointTargetMode.POSITION)
            actuated_dofs.append(qds)
        else:
            ke[qds] = 0.0
            kd[qds] = 0.0
            arm[qds] = 0.0
            mode[qds] = int(JointTargetMode.POSITION)
    model.joint_target_ke.assign(wp.array(ke, dtype=wp.float32, device=model.device))
    model.joint_target_kd.assign(wp.array(kd, dtype=wp.float32, device=model.device))
    model.joint_armature.assign(wp.array(arm, dtype=wp.float32, device=model.device))
    model.joint_target_mode.assign(wp.array(mode, dtype=wp.int32, device=model.device))
    return sorted(actuated_dofs)


def _apply_effort_limits(model):
    """Driven joints -> 3.1 N·m, passive linkage DOFs -> 400 N·m."""
    eff = model.joint_effort_limit.numpy()
    qd_start = model.joint_qd_start.numpy()
    n = len(eff)
    for j in range(model.joint_count):
        qds = int(qd_start[j])
        if qds >= n:
            continue
        name = _joint_short_label(str(model.joint_label[j]))
        eff[qds] = DRIVEN_EFFORT if name in DR_LEGS_ACTUATED_JOINTS else PASSIVE_EFFORT
    model.joint_effort_limit.assign(wp.array(eff, dtype=wp.float32, device=model.device))


def build_model(device: str = "cuda:0", mesh: bool = False):
    """Build fixed-base (hanging) DR Legs model from the Disney USD."""
    asset_path = newton.utils.download_asset("disneyresearch")
    fname = "dr_legs_with_meshes_and_boxes.usda" if mesh else "dr_legs_with_boxes.usda"
    asset_file = str(asset_path / "dr_legs/usd" / fname)

    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    builder.default_shape_cfg.margin = 1e-6
    builder.default_shape_cfg.gap = 0.005
    # Spawn at the ORIGIN (not SPAWN_HEIGHT): the closed-loop anchor frames
    # (joint_X_p / joint_X_c) are authored at the assembled reference pose, and
    # offsetting the whole robot via xform while pinning the pelvis there makes
    # the fixed-joint anchor inconsistent with the loop anchors, which blows the
    # loops open (verified: z=0.28 -> residual 1.9; origin -> 6e-8). Gravity +
    # the pin make it "hang"; the world height is cosmetic, so keep it at 0.
    xform = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())
    builder.add_usd(
        asset_file,
        xform=xform,
        joint_ordering=None,
        force_show_colliders=True,
        force_position_velocity_actuation=True,
        collapse_fixed_joints=False,   # closed loops: keep all joints
        enable_self_collisions=False,
        hide_collision_shapes=True,
        floating=True,                 # free base; pinned below with a fixed joint
    )
    # No ground plane: hanging in the air, contacts disabled.

    # TRUE HANGING (like the fixed-base G1/H1 examples): pin the pelvis to the
    # world with a fixed joint at its spawn pose, then leave gravity ON so the
    # legs hang and the actuators drive the joints against gravity. (A free base
    # in zero-g would tumble from actuation reaction torques and hide the joint
    # motion.) The earlier pinned-base explosion was caused by joint-limit penalty
    # forces, not pinning -- with joint_limit_ke_scale=0 + position correction the
    # pinned mechanism holds its loops at ~6e-8.
    pelvis_idx = next(
        (i for i in range(builder.body_count)
         if "pelvis" in str(builder.body_label[i]).lower()),
        0,
    )
    builder.add_joint_fixed(
        parent=-1,
        child=pelvis_idx,
        parent_xform=builder.body_q[pelvis_idx],
        label="world_to_pelvis_fixed",
    )
    print(f"  Pinned pelvis (body {pelvis_idx}) to world with a fixed joint.")

    model = builder.finalize(skip_validation_joints=True, device=device)
    model.rigid_contact_max = 65536

    actuated_dofs = _apply_actuator_gains(model)
    _apply_effort_limits(model)

    print(
        f"  DR Legs hanging: {model.body_count} bodies, {model.joint_count} joints, "
        f"{model.joint_dof_count} DOFs, {len(actuated_dofs)} actuated"
    )
    return model, actuated_dofs


def _get_default_targets(model):
    """Default joint-position targets (DOF-indexed). DR Legs default pose = all zero."""
    n_dofs = model.joint_dof_count
    return np.zeros(n_dofs, dtype=np.float32)


def _get_action_scales(model, actuated_dofs):
    """Per-DOF action scale: 0.3 on actuated DOFs, 0 on passive linkage DOFs."""
    n_dofs = model.joint_dof_count
    scales = np.zeros(n_dofs, dtype=np.float32)
    for d in actuated_dofs:
        scales[d] = ACTION_SCALE
    return scales


# ═══════════════════════════════════════════════════════════════════════════════
# Loop-closure residual diagnostic
# ═══════════════════════════════════════════════════════════════════════════════
def loop_residual(model, state):
    """Max world-space anchor gap across all joints (incl. loop closers)."""
    body_q = state.body_q.numpy()
    jp = model.joint_parent.numpy()
    jc = model.joint_child.numpy()
    Xp = model.joint_X_p.numpy()
    Xc = model.joint_X_c.numpy()

    def tf_point(qb, xa):
        p_b, w = qb[:3], qb[3:7]
        x, y, z, ww = w
        R = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * ww), 2 * (x * z + y * ww)],
            [2 * (x * y + z * ww), 1 - 2 * (x * x + z * z), 2 * (y * z - x * ww)],
            [2 * (x * z - y * ww), 2 * (y * z + x * ww), 1 - 2 * (x * x + y * y)],
        ])
        return p_b + R @ xa[:3]

    jtype = model.joint_type.numpy()
    free_t = int(newton.JointType.FREE)
    dist_t = int(newton.JointType.DISTANCE)
    res = []
    for j in range(len(jp)):
        # FREE/DISTANCE joints are not loop-closure constraints (the floating base
        # has 6 free DOFs); their "anchor gap" is meaningless, so skip them.
        if jtype[j] in (free_t, dist_t):
            continue
        wa = Xp[j][:3] if jp[j] < 0 else tf_point(body_q[jp[j]], Xp[j])
        wc = tf_point(body_q[jc[j]], Xc[j])
        res.append(np.linalg.norm(wa - wc))
    return np.array(res) if res else np.array([0.0])


# ═══════════════════════════════════════════════════════════════════════════════
# Action generation
# ═══════════════════════════════════════════════════════════════════════════════
def generate_actions_sinusoidal(n_frames, n_dofs, seed=42):
    """Sinusoidal policy actions in [-1, 1] per joint (full Isaac Lab action range).

    The Isaac Lab DR Legs walk env uses ``JointPositionActionCfg(scale=0.3,
    use_default_offset=True)``, so a policy action ``a`` in [-1, 1] maps to a
    joint target ``a * 0.3 + default_pos``. Driving each actuated joint with a
    full unit-amplitude sinusoid therefore sweeps the complete +/-0.3 rad target
    range the trained policy can command.
    """
    rng = np.random.default_rng(seed)
    freqs = rng.uniform(0.3, 1.5, size=n_dofs)
    phases = rng.uniform(0, 2 * np.pi, size=n_dofs)
    # Full unit amplitude => actions span the entire [-1, 1] policy range.
    amplitudes = np.ones(n_dofs)
    times = np.arange(n_frames) * PHYSICS_DT
    actions = amplitudes[None, :] * np.sin(
        2 * np.pi * freqs[None, :] * times[:, None] + phases[None, :]
    )
    return actions.astype(np.float32)


def generate_actions_step(n_frames, n_dofs, seed=42):
    """Step inputs: random target held for 0.5-1.0 s chunks."""
    rng = np.random.default_rng(seed)
    actions = np.zeros((n_frames, n_dofs), dtype=np.float32)
    i = 0
    while i < n_frames:
        chunk = rng.integers(int(0.5 / PHYSICS_DT), int(1.0 / PHYSICS_DT))
        end = min(i + chunk, n_frames)
        actions[i:end] = rng.uniform(-0.8, 0.8, size=n_dofs).astype(np.float32)
        i = end
    return actions


# ═══════════════════════════════════════════════════════════════════════════════
# DVI solver factory (matches walk env _dvi_solver_cfg)
# ═══════════════════════════════════════════════════════════════════════════════
def create_dvi_solver(model, use_poscorr: bool = False, use_joint_limits: bool = False):
    """Create DVI solver for the hanging DR Legs.

    Built on the walk-env DVISolverCfg. The 6 closed loops are stable WITHOUT
    position correction once the joint-LIMIT forces are handled correctly:
      * the original explosion was caused by PENALTY-based joint limits
        (ke springs) fighting the 24 passive closed-loop DOFs;
      * with joint_limit_ke_scale=0 (no penalty), the loops hold at ~1e-8;
      * the CONSTRAINT-based limit solver (sparse Jacobi, alpha=0,
        recovery_speed=0.5) is fully stable with the loops too.
    Position correction is therefore OFF by default; it only tightens the loop
    residual (e.g. 4e-6 -> 6e-8 at substeps=4) and is not required for stability.
    Pass use_poscorr=True to enable it.
    """
    # Optional position-correction sub-solve (geometric loop closure). Shares the
    # LDL factorization config; alpha defaults to 0 here (pure position projection).
    poscorr = NumericalSolverConfig(
        solver_type=SolverType.SPARSE_LDL,
        max_iterations=50,
        reg=1e-4,
        diagonal_precondition=True,
    ) if use_poscorr else None
    jc = NumericalSolverConfig(
        solver_type=SolverType.SPARSE_LDL,
        max_iterations=50,
        alpha=0.0,
        recovery_speed=100000.0,
        # Closed loops are rank-deficient: reg=1e-4 + diagonal precond for a
        # stable LDL factorization (validated on four-bar + DR Legs tests).
        reg=1e-4,
        diagonal_precondition=True,
        iterative_refinement_steps=1,
        # Optional (off by default): tightens loop residual but not required.
        position_correction=poscorr,
    )
    # Joint limits. Two modes:
    #  * use_joint_limits=False (default): penalty mode with ke_scale=0 => ZERO
    #    joint-limit forces. The penalty limit forces on the 24 passive closed-loop
    #    DOFs were the original source of instability; with them off the loops hold
    #    at ~1e-8 for all dt.
    #  * use_joint_limits=True: CONSTRAINT-BASED limit solver (sparse Jacobi,
    #    alpha=0, recovery_speed=0.5). Limits enforced as unilateral constraints
    #    (lambda >= 0); the penalty ke/kd path is auto-disabled when this is active.
    joint_limit_solver = None
    if use_joint_limits:
        joint_limit_solver = NumericalSolverConfig(
            solver_type=SolverType.SPARSE_JACOBI,
            max_iterations=20,
            alpha=0.0,
            recovery_speed=0.5,
            reg=1e-4,
            diagonal_precondition=True,
        )
    solver = newton.solvers.SolverDVI(
        model,
        joint_solver=jc,
        angular_damping=0.0,
        enable_contacts=False,          # hanging: no contacts
        enable_timers=False,
        actuator_integration=ActuatorIntegration.SEMI_IMPLICIT,
        joint_limit_ke_scale=0.0,
        joint_limit_solver=joint_limit_solver,
    )
    return solver


# ═══════════════════════════════════════════════════════════════════════════════
# Simulation
# ═══════════════════════════════════════════════════════════════════════════════
def run_dvi(model, actions, default_targets, action_scales, actuated_dofs,
            use_poscorr=False, use_joint_limits=False, substeps=DVI_NUM_SUBSTEPS):
    """Run the DVI solver on the hanging DR Legs and return joint trajectories."""
    n_frames = actions.shape[0]
    n_dofs = model.joint_dof_count
    sim_dt = PHYSICS_DT / substeps

    state0 = model.state()
    state1 = model.state()
    control = model.control()
    eval_fk(model, model.joint_q, model.joint_qd, state0)

    tp = control.joint_target_pos.numpy()
    tp[:n_dofs] = default_targets
    control.joint_target_pos.assign(wp.array(tp, dtype=wp.float32, device=model.device))

    solver = create_dvi_solver(model, use_poscorr=use_poscorr, use_joint_limits=use_joint_limits)

    joint_q_traj = []
    joint_qd_traj = []
    body_q_traj = []
    max_loop_res = 0.0

    # NOTE: ``eval_ik`` is a no-op for DR Legs (articulation_count == 0, orphan
    # joints), so ``joint_q`` stays at its initial zeros. The ground-truth motion
    # signal in maximal coordinates is BODY displacement, which we track here.
    body_q0 = state0.body_q.numpy()[:, :3].copy()

    r0 = loop_residual(model, state0)
    print(f"  Initial max loop residual: {r0.max():.3e}")

    t0 = time.perf_counter()
    diverged_at = None
    for frame_i in range(n_frames):
        raw_action = actions[frame_i]
        target = raw_action * action_scales + default_targets
        tp = control.joint_target_pos.numpy()
        tp[:n_dofs] = target
        control.joint_target_pos.assign(wp.array(tp, dtype=wp.float32, device=model.device))

        for _ in range(substeps):
            state0.clear_forces()
            solver.step(state0, state1, control, None, sim_dt)
            eval_ik(model, state1, state1.joint_q, state1.joint_qd)
            state0, state1 = state1, state0

        if frame_i % 2 == 0:
            wp.synchronize()
            jq = state0.joint_q.numpy()[:n_dofs].copy()
            jqd = state0.joint_qd.numpy()[:n_dofs].copy()
            bq = state0.body_q.numpy()[:, :3].copy()
            joint_q_traj.append(jq)
            joint_qd_traj.append(jqd)
            body_q_traj.append(bq)
            r = loop_residual(model, state0)
            max_loop_res = max(max_loop_res, float(r.max()))
            if not np.isfinite(bq).all() or not np.isfinite(r.max()):
                print(f"  [DVI] NaN/Inf at frame {frame_i}!")
                diverged_at = frame_i
                break

    wp.synchronize()
    t_sim = time.perf_counter() - t0

    return {
        "joint_q": np.array(joint_q_traj),
        "joint_qd": np.array(joint_qd_traj),
        "body_q": np.array(body_q_traj),
        "body_q0": body_q0,
        "sim_time_s": t_sim,
        "n_frames": n_frames,
        "max_loop_res": max_loop_res,
        "diverged_at": diverged_at,
        "actuated_dofs": actuated_dofs,
    }


def print_summary(res, model, default_targets, action_scales):
    """Report whether actuation moved the robot and loops stayed closed.

    Motion is measured in BODY space (per-body world displacement), because
    ``eval_ik`` is a no-op for this orphan-joint closed-loop model and leaves
    ``joint_q`` at zero. Body displacement is the ground-truth signal that the
    actuators are driving the mechanism.
    """
    bq = res["body_q"]
    if len(bq) == 0:
        print("  No frames recorded.")
        return

    bq0 = res["body_q0"]
    # per-body displacement from start, per recorded frame
    disp = np.linalg.norm(bq - bq0[None, :, :], axis=2)     # (frames, bodies)
    body_max_disp = disp.max(axis=0)                         # max disp per body
    body_rom = disp.max(axis=0) - disp.min(axis=0)

    labels = [_joint_short_label(str(model.body_label[i])) for i in range(model.body_count)]

    print(f"\n{'=' * 78}")
    print(f"  DR Legs hanging — DVI actuation summary ({len(bq)} frames)")
    print(f"{'=' * 78}")
    # show the 10 most-moved bodies
    order = np.argsort(-body_max_disp)[:10]
    print(f"  Most-actuated bodies (world displacement):")
    print(f"  {'body':<22}{'max_disp(m)':>13}{'ROM(m)':>11}")
    print(f"  {'-' * 46}")
    for i in order:
        print(f"  {labels[i]:<22}{body_max_disp[i]:>13.4f}{body_rom[i]:>11.4f}")

    print(f"  {'-' * 46}")
    print(f"  Body displacement   : max {body_max_disp.max():.4f} m, "
          f"mean {body_max_disp.mean():.4f} m")
    print(f"  Max loop residual   : {res['max_loop_res']:.3e}")
    print(f"  DVI sim time        : {res['sim_time_s']:.2f}s "
          f"({res['n_frames'] / max(res['sim_time_s'], 1e-6):.0f} steps/s)")
    print(f"  (eval_ik is a no-op for this closed-loop model; joint_q stays 0 — "
          f"motion measured in body space.)")

    # Pass/fail: bodies clearly moved (actuation works), loops stayed closed, no NaN.
    ok = (
        res["diverged_at"] is None
        and body_max_disp.max() > 5e-3
        and res["max_loop_res"] < 5e-2
    )
    if ok:
        print(f"\n  PASS: actuators drive the robot (max body disp "
              f"{body_max_disp.max():.3f} m) and loops stay closed "
              f"(max res {res['max_loop_res']:.2e}).")
    else:
        why = []
        if res["diverged_at"] is not None:
            why.append(f"diverged at frame {res['diverged_at']}")
        if body_max_disp.max() <= 5e-3:
            why.append(f"robot barely moved (max body disp {body_max_disp.max():.4f} m)")
        if res["max_loop_res"] >= 5e-2:
            why.append(f"loops opened (max res {res['max_loop_res']:.2e})")
        print(f"\n  FAIL: {'; '.join(why)}")


# ═══════════════════════════════════════════════════════════════════════════════
# Video recording (DVI only)
# ═══════════════════════════════════════════════════════════════════════════════
def record_video(model, actions, default_targets, action_scales, output_path,
                 use_poscorr=False, use_joint_limits=False, substeps=DVI_NUM_SUBSTEPS):
    """Record a DVI video of the hanging DR Legs."""
    from OpenGL import GL as gl

    W, H = 1280, 720
    n_frames = actions.shape[0]
    n_dofs = model.joint_dof_count
    sim_dt = PHYSICS_DT / substeps

    state0 = model.state()
    state1 = model.state()
    control = model.control()
    eval_fk(model, model.joint_q, model.joint_qd, state0)

    tp = control.joint_target_pos.numpy()
    tp[:n_dofs] = default_targets
    control.joint_target_pos.assign(wp.array(tp, dtype=wp.float32, device=model.device))

    solver = create_dvi_solver(model, use_poscorr=use_poscorr, use_joint_limits=use_joint_limits)

    try:
        viewer = newton.viewer.ViewerGL(headless=True, width=W, height=H)
    except Exception as e:
        print(f"  ViewerGL failed: {e}")
        return None

    viewer.set_model(model)
    from pyglet.math import Vec3 as PVec3
    # DR Legs pelvis pinned at z=0.28; legs hang below.
    # Pelvis pinned near the origin; legs hang below (negative z). Aim at the
    # robot's mid-body so the whole hanging mechanism is framed.
    viewer.camera.pos = PVec3(1.6, -1.4, -0.05)
    viewer.camera.look_at(PVec3(0.0, 0.0, -0.25))

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
        control.joint_target_pos.assign(wp.array(tp, dtype=wp.float32, device=model.device))

        for _ in range(substeps):
            state0.clear_forces()
            solver.step(state0, state1, control, None, sim_dt)
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
            if not np.isfinite(state0.joint_q.numpy()).all():
                print(f"  [DVI] NaN at frame {frame_i}")
                break

    ffmpeg_proc.stdin.close()
    ffmpeg_proc.wait()
    viewer.close()
    return output_path if os.path.exists(output_path) else None


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Hanging fixed-base DR Legs benchmark (DVI solver, closed-loop)"
    )
    parser.add_argument("--sim-time", type=float, default=4.0)
    parser.add_argument(
        "--action-type", choices=["sinusoidal", "step"], default="sinusoidal",
        help="Action waveform type (default: sinusoidal)",
    )
    parser.add_argument("--no-video", action="store_true", help="Skip video recording")
    parser.add_argument("--mesh", action="store_true", default=True,
                        help="Use meshes+boxes asset variant (default: on, for visible geometry)")
    parser.add_argument("--no-mesh", dest="mesh", action="store_false",
                        help="Use the collision-box-only asset variant")
    # Position correction is OFF by default (velocity-only solve): the constraint-
    # based joint-limit solver alone keeps the closed loops stable, so position
    # correction is not needed. Pass --poscorr to enable it (tightens loop residual).
    parser.add_argument("--poscorr", dest="poscorr", action="store_true", default=False,
                        help="Enable joint position correction (default: off)")
    parser.add_argument("--substeps", type=int, default=DVI_NUM_SUBSTEPS,
                        help=f"DVI solver substeps per sim step (default: {DVI_NUM_SUBSTEPS})")
    parser.add_argument("--joint-limits", dest="joint_limits", action="store_true", default=False,
                        help="Enable constraint-based joint-limit solver (sparse Jacobi, "
                             "alpha=0, recovery_speed=0.5)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--_record", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    wp.init()
    wp.set_device(args.device)

    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(__file__), "hanging_drlegs_results")
    os.makedirs(args.output_dir, exist_ok=True)

    print("Building hanging (fixed-base) DR Legs model ...", flush=True)
    model, actuated_dofs = build_model(args.device, mesh=args.mesh)

    n_dofs = model.joint_dof_count
    default_targets = _get_default_targets(model)
    action_scales = _get_action_scales(model, actuated_dofs)

    n_frames = int(args.sim_time / PHYSICS_DT)
    if args.action_type == "sinusoidal":
        actions = generate_actions_sinusoidal(n_frames, n_dofs, args.seed)
    else:
        actions = generate_actions_step(n_frames, n_dofs, args.seed)

    print(f"  Action: {args.action_type}, scale={ACTION_SCALE}, "
          f"{len(actuated_dofs)}/{n_dofs} actuated DOFs, {n_frames} frames")

    # ── Subprocess video mode ────────────────────────────────────────
    if args._record:
        vp = record_video(
            model, actions, default_targets, action_scales,
            os.path.join(args.output_dir, "dvi.mp4"),
            use_poscorr=args.poscorr,
            use_joint_limits=args.joint_limits,
            substeps=args.substeps,
        )
        if vp:
            print(f"VIDEO_PATH={vp}")
        return

    # ── Run DVI + summary ────────────────────────────────────────────
    print("\n  Running DVI ...", flush=True)
    print(f"  Position correction: {'ON' if args.poscorr else 'OFF'}, "
          f"joint-limit solver: {'sparse_jacobi(a=0,rec=0.5)' if args.joint_limits else 'OFF'}, "
          f"substeps={args.substeps}, asset={'meshes' if args.mesh else 'boxes'}")
    res = run_dvi(model, actions, default_targets, action_scales, actuated_dofs,
                  use_poscorr=args.poscorr, use_joint_limits=args.joint_limits,
                  substeps=args.substeps)
    print(f"    Done in {res['sim_time_s']:.2f}s")
    print_summary(res, model, default_targets, action_scales)

    # ── Video (subprocess for GL cleanup) ────────────────────────────
    if not args.no_video:
        print("\nRecording DVI video ...", flush=True)
        cmd = [
            sys.executable, os.path.abspath(__file__),
            "--sim-time", str(args.sim_time),
            "--action-type", args.action_type,
            "--device", args.device,
            "--output-dir", args.output_dir,
            "--seed", str(args.seed),
            "--_record",
        ]
        if args.mesh:
            cmd.append("--mesh")
        else:
            cmd.append("--no-mesh")
        if args.poscorr:
            cmd.append("--poscorr")
        if args.joint_limits:
            cmd.append("--joint-limits")
        cmd += ["--substeps", str(args.substeps)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        vp = os.path.join(args.output_dir, "dvi.mp4")
        if os.path.exists(vp):
            print(f"  ✓ {vp} ({os.path.getsize(vp) / 1024:.0f} KB)")
        else:
            print("  ✗ Video failed")
            if result.stderr:
                for line in result.stderr.strip().split("\n")[-6:]:
                    print(f"    {line}")

    print("\nDone.")


if __name__ == "__main__":
    main()
