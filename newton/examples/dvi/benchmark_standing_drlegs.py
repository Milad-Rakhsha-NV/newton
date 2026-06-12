"""Standing DR Legs contact-force probe (DVI solver, closed-loop).

Companion to ``benchmark_hanging_drlegs.py``. Instead of pinning the pelvis in the
air, this script stands DR Legs on its feet on a ground plane and reports the
per-foot contact forces measured exactly the way the Isaac Lab RL contact sensor
does (Newton's ``write_contact_forces`` kernel from solved DVI contact lambdas).

Goal: confirm the contact solver reports sane normal forces -- the two feet
together should support the body weight (~24.2 N for the 2.47 kg DR Legs) at
static equilibrium. If they do, the contact pipeline is healthy and the walk-in-
place jitter is a reward-shaping problem, not a physics one.

Solver config matches the validated walk-env DVI preset:
  * joints:  sparse_ldl, 50 iters, alpha=0, recovery 1e5, reg 1e-4, no poscorr
  * limits:  CONSTRAINT-based sparse_jacobi, alpha=0, recovery 0.5, reg 1e-4, ke_scale=0
  * contacts: sparse_jacobi, 20 iters, alpha=0, recovery 1.0 (UNCHANGED -- shown to work)
"""
import argparse
import os
import subprocess
import sys

import numpy as np
import warp as wp

os.environ.setdefault("WARP_LOG_LEVEL", "warning")

import newton  # noqa: E402
from newton._src.solvers.dvi.numerical_solver.base import NumericalSolverConfig  # noqa: E402
from newton.solvers import ActuatorIntegration, SolverType  # noqa: E402
from newton import JointTargetMode  # noqa: E402
from newton._src.solvers.dvi.contact_kernels import write_contact_forces  # noqa: E402

GRAVITY = 9.81
PHYSICS_DT = 0.004
DRIVEN_KE = 5.0
DRIVEN_KD = 0.2

DR_LEGS_ACTUATED = [
    "j1_l_i", "j2_l_i", "j6_l_i", "j7_l_i", "j2_l_o", "j7_l_o",
    "j1_r_i", "j2_r_i", "j6_r_i", "j7_r_i", "j2_r_o", "j7_r_o",
]


def _short(label):
    return str(label).rsplit("/", 1)[-1]


def build_model(device, mesh=True, ground_z=None, start_gap=None):
    asset_path = newton.utils.download_asset("disneyresearch")
    fname = "dr_legs_with_meshes_and_boxes.usda" if mesh else "dr_legs_with_boxes.usda"
    asset_file = str(asset_path / "dr_legs/usd" / fname)

    b = newton.ModelBuilder(up_axis=newton.Axis.Z)
    b.default_shape_cfg.margin = 1e-6
    b.default_shape_cfg.gap = 0.005
    # Spawn at the ORIGIN (closed-loop anchors are authored there; offsetting via
    # xform breaks the loops). The feet hang below the pelvis origin; we put the
    # ground plane right under the lowest foot point so the robot stands without
    # having to translate the assembly.
    b.add_usd(
        asset_file,
        joint_ordering=None,
        force_show_colliders=True,
        force_position_velocity_actuation=True,
        collapse_fixed_joints=False,
        enable_self_collisions=False,
        hide_collision_shapes=True,
        floating=True,
    )

    # Determine the lowest body z at spawn, then lift the whole assembly so the
    # lowest foot point starts a few mm BELOW the z=0 ground plane. A small initial
    # penetration guarantees the collision pipeline registers foot-ground contacts
    # immediately (placing the feet exactly at the contact margin can miss). The
    # robot then settles to static equilibrium in the first few steps.
    zs = [b.body_q[i][2] for i in range(b.body_count)]
    lowest = min(zs)
    # start_gap > 0 => feet start that far ABOVE the ground (drop, like the RL env
    # reset which spawns feet ~2cm above z=0). Default: small penetration so it is
    # already in contact for the static force probe.
    penetration = -start_gap if start_gap is not None else 0.004
    b.add_ground_plane()
    # Uniform rigid-body translation of the assembled reference keeps all loop
    # anchors consistent (unlike a per-body xform), so the closed loops stay closed.
    lift = -lowest - penetration  # lowest point lands at z = -penetration
    for i in range(b.body_count):
        t = b.body_q[i]
        b.body_q[i] = wp.transform(
            wp.vec3(t[0], t[1], t[2] + lift), wp.quat(t[3], t[4], t[5], t[6])
        )

    model = b.finalize(skip_validation_joints=True, device=device)
    model.rigid_contact_max = 65536

    # Apply walk-env driven-joint gains (ke/kd) so the actuators can hold pose.
    _apply_gains(model)

    foot_bodies = {}
    for i in range(model.body_count):
        sl = _short(model.body_label[i]).lower()
        if sl in ("foot_l", "foot_r"):
            foot_bodies[sl] = i
    total_mass = float(model.body_mass.numpy().sum())
    print(f"  DR Legs standing: {model.body_count} bodies, total mass {total_mass:.4f} kg, "
          f"weight {total_mass * GRAVITY:.3f} N")
    print(f"  feet bodies: {foot_bodies}")
    return model, foot_bodies, total_mass


def _apply_gains(model):
    ke = model.joint_target_ke.numpy()
    kd = model.joint_target_kd.numpy()
    mode = model.joint_target_mode.numpy()
    qd_start = model.joint_qd_start.numpy()
    n = len(ke)
    n_driven = 0
    for j in range(model.joint_count):
        d = int(qd_start[j])
        if d >= n:
            continue
        sl = _short(model.joint_label[j])
        mode[d] = int(JointTargetMode.POSITION)
        if sl in DR_LEGS_ACTUATED:
            ke[d] = DRIVEN_KE
            kd[d] = DRIVEN_KD
            n_driven += 1
        else:
            ke[d] = 0.0
            kd[d] = 0.0
    model.joint_target_ke.assign(wp.array(ke, dtype=wp.float32, device=model.device))
    model.joint_target_kd.assign(wp.array(kd, dtype=wp.float32, device=model.device))
    model.joint_target_mode.assign(wp.array(mode, dtype=wp.int32, device=model.device))
    print(f"  applied driven gains to {n_driven} joints (ke={DRIVEN_KE}, kd={DRIVEN_KD})")


def create_solver(model):
    jc = NumericalSolverConfig(
        solver_type=SolverType.SPARSE_LDL,
        max_iterations=50,
        alpha=0.0,
        recovery_speed=100000.0,
        reg=1e-4,
        diagonal_precondition=True,
        iterative_refinement_steps=1,
    )
    contact_cfg = NumericalSolverConfig(
        solver_type=SolverType.SPARSE_JACOBI,
        max_iterations=20,
        alpha=0.0,
        recovery_speed=1.0,
    )
    joint_limit_cfg = NumericalSolverConfig(
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
        contact_solver=contact_cfg,
        angular_damping=0.0,
        enable_contacts=True,
        enable_timers=False,
        actuator_integration=ActuatorIntegration.SEMI_IMPLICIT,
        joint_limit_ke_scale=0.0,
        joint_limit_solver=joint_limit_cfg,
    )
    return solver


def extract_foot_forces(model, contacts, solver, foot_bodies, dt):
    """Per-foot net contact force (N), using the same kernel as the RL sensor."""
    if contacts.force is None:
        return None
    constraint = solver._contact_solver._constraint
    wp.launch(
        write_contact_forces,
        dim=constraint.contact_max,
        inputs=[
            contacts.rigid_contact_count,
            contacts.rigid_contact_normal,
            constraint.lambda_,
            dt,
            constraint.contact_max,
        ],
        outputs=[contacts.force],
        device=contacts.device,
    )
    n = int(contacts.rigid_contact_count.numpy()[0])
    if n == 0:
        return {k: np.zeros(3) for k in foot_bodies}, 0
    forces = contacts.force.numpy()[:n, :3]        # spatial_vector -> linear force [N]
    shape0 = contacts.rigid_contact_shape0.numpy()[:n]
    shape1 = contacts.rigid_contact_shape1.numpy()[:n]
    shape_body = model.shape_body.numpy()
    per_foot = {k: np.zeros(3) for k in foot_bodies}
    body_to_foot = {v: k for k, v in foot_bodies.items()}
    for c in range(n):
        for sh in (shape0[c], shape1[c]):
            if sh < 0:
                continue
            bd = shape_body[sh]
            if bd in body_to_foot:
                per_foot[body_to_foot[bd]] += forces[c]
    return per_foot, n


def record_video(model, foot_bodies, output_path, substeps=4, n_frames=300, ground_z=0.0):
    """Render the standing DR Legs settling onto the ground (zero action, hold pose)."""
    from OpenGL import GL as gl

    W, H = 1280, 720
    sim_dt = PHYSICS_DT / substeps
    solver = create_solver(model)
    model.request_contact_attributes("force")
    s0, s1 = model.state(), model.state()
    ctrl = model.control()
    pipeline = newton.CollisionPipeline(model)
    contacts = model.contacts(collision_pipeline=pipeline)

    try:
        viewer = newton.viewer.ViewerGL(headless=True, width=W, height=H)
    except Exception as e:
        print(f"  ViewerGL failed: {e}")
        return None
    viewer.set_model(model)
    from pyglet.math import Vec3 as PVec3
    viewer.camera.pos = PVec3(1.6, -1.4, 0.45)
    viewer.camera.look_at(PVec3(0.0, 0.0, 0.12))

    SIM_FPS = int(round(1.0 / PHYSICS_DT))
    ffmpeg_proc = subprocess.Popen(
        ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", str(SIM_FPS), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", output_path],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    sim_time_val = 0.0
    for frame_i in range(n_frames):
        for _ in range(substeps):
            s0.clear_forces()
            pipeline.collide(s0, contacts)
            solver.step(s0, s1, ctrl, contacts, dt=sim_dt)
            s0, s1 = s1, s0
        sim_time_val += PHYSICS_DT
        viewer.begin_frame(sim_time_val)
        viewer.log_state(s0)
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
    return output_path if os.path.exists(output_path) else None


def main():
    ap = argparse.ArgumentParser(description="Standing DR Legs contact-force probe (DVI)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--mesh", action="store_true", default=True)
    ap.add_argument("--substeps", type=int, default=4)
    ap.add_argument("--video", default=None, help="output mp4 path; records settle then exits")
    args = ap.parse_args()

    if args.video:
        wp.init()
        with wp.ScopedDevice(args.device):
            model, foot_bodies, total_mass = build_model(args.device, mesh=args.mesh, start_gap=0.02)
            out = record_video(model, foot_bodies, args.video, substeps=args.substeps,
                               n_frames=args.steps)
            print(f"VIDEO_PATH={out}")
        return 0

    wp.init()
    with wp.ScopedDevice(args.device):
        model, foot_bodies, total_mass = build_model(args.device, mesh=args.mesh)
        weight = total_mass * GRAVITY
        solver = create_solver(model)

        s0, s1 = model.state(), model.state()
        ctrl = model.control()
        # Request the extended "force" attribute so the Contacts buffer allocates a
        # spatial-force array (same as the Isaac Lab RL contact sensor path). Without
        # this, contacts.force stays None and write_contact_forces has nowhere to go.
        model.request_contact_attributes("force")
        pipeline = newton.CollisionPipeline(model)
        contacts = model.contacts(collision_pipeline=pipeline)
        print(f"  contacts.force allocated: {contacts.force is not None}")

        sim_dt = PHYSICS_DT / args.substeps
        print(f"  substeps={args.substeps}, solver dt={sim_dt:.5f}, settling {args.steps} steps\n")

        history = []
        for i in range(args.steps):
            for _ in range(args.substeps):
                s0.clear_forces()
                pipeline.collide(s0, contacts)
                solver.step(s0, s1, ctrl, contacts, dt=sim_dt)
                s0, s1 = s1, s0
            res = extract_foot_forces(model, contacts, solver, foot_bodies, sim_dt)
            if res is None:
                continue
            per_foot, ncon = res
            fl_n = per_foot["foot_l"][2]
            fr_n = per_foot["foot_r"][2]
            total_n = fl_n + fr_n
            history.append((fl_n, fr_n, total_n, ncon))
            if (i + 1) % max(1, args.steps // 10) == 0:
                print(f"  step {i+1:4d}: contacts={ncon:3d}  "
                      f"Fz_l={fl_n:7.3f}  Fz_r={fr_n:7.3f}  Fz_total={total_n:7.3f} N  "
                      f"(weight={weight:.2f} N, ratio={total_n/weight:.3f})")

        # Report the settled (last 20%) average.
        hist = np.array([(a, b, c) for a, b, c, _ in history])
        if len(hist) == 0:
            print("\nNO CONTACTS DETECTED -- feet never touched the ground.")
            return 1
        tail = hist[int(0.8 * len(hist)):]
        avg_l, avg_r, avg_total = tail.mean(axis=0)
        print("\n" + "=" * 60)
        print(f"  SETTLED (last 20%) average vertical contact force:")
        print(f"    foot_l Fz = {avg_l:.3f} N")
        print(f"    foot_r Fz = {avg_r:.3f} N")
        print(f"    total  Fz = {avg_total:.3f} N")
        print(f"    body weight = {weight:.3f} N")
        print(f"    total / weight = {avg_total / weight:.3f}")
        print("=" * 60)
        # The kernel reports the reaction force on the foot shape (force_on_shape0 =
        # J^T*lambda/dt), whose vertical component is negative in this frame; its
        # MAGNITUDE is what must balance body weight at static equilibrium.
        ratio = abs(avg_total) / weight
        print(f"    |total| / weight = {ratio:.3f}")
        if 0.85 <= ratio <= 1.15:
            print("PASS: feet contact force balances body weight (contacts are healthy).")
            return 0
        print("WARN: contact force does NOT balance weight -- investigate contact pipeline.")
        return 2


if __name__ == "__main__":
    sys.exit(main())
