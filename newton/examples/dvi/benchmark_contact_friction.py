# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Benchmark: Contact Friction Accuracy
=====================================

Verifies DVI DVI contact solver against analytical Coulomb friction results.

Scenarios
---------
1. **Static box on ground** — no applied force.
   Expected: box rests at z = H + margin, zero velocity.

2. **Sub-threshold push** — horizontal force F < mu * m * g.
   Expected: box stays put (static friction absorbs the force).

3. **At-threshold push** — horizontal force F = mu * m * g.
   Expected: box at the verge of sliding, near-zero acceleration.

4. **Super-threshold push** — horizontal force F > mu * m * g.
   Expected: box slides with acceleration a = (F - mu*m*g) / m.

5. **Sliding box** — initial horizontal velocity, no external force.
   Expected: deceleration a = -mu * g until stop.

6. **Sliding box with push** — initial velocity + horizontal force F < mu*m*g.
   Expected: kinetic friction decelerates, F partially offsets it.

All scenarios use a flat ground plane with a single box (known mass,
inertia, margin, friction coefficient). Analytical values are compared
against simulation.

Usage::

    PYTHONPATH=. python3 newton/examples/dvi/benchmark_contact_friction.py
"""

import sys

import numpy as np
import warp as wp

import newton
from newton.solvers import FrictionProjection, ActuatorIntegration, SolverType

# ============================================================================
# Constants
# ============================================================================

GRAVITY = 9.81  # m/s² (positive; applied as -z)
BOX_HALF = 0.1  # half-extent of box (m)
BOX_MASS = 8.0  # kg (heavy enough for clean contact dynamics)
MU = 0.5  # friction coefficient
MARGIN = 0.0  # zero margin for simplest analytical comparison
GAP = 0.01  # gap for collision detection
DT = 1.0 / 500.0  # 500 Hz for accuracy
SIM_TIME = 2.0  # seconds
DEVICE = "cuda:0"
CONTACT_ITERS = 100
CONTACT_OMEGA = 0.5

# Derived
WEIGHT = BOX_MASS * GRAVITY  # N
FRICTION_FORCE = MU * WEIGHT  # N (max static / kinetic friction)


def _create_box_on_ground(
    mu: float = MU,
    margin: float = MARGIN,
    gap: float = GAP,
    mass: float = BOX_MASS,
    box_half: float = BOX_HALF,
    device: str = DEVICE,
):
    """Create a single free-body box on a ground plane."""
    builder = newton.ModelBuilder()
    builder.default_shape_cfg.mu = mu
    builder.default_shape_cfg.margin = margin
    builder.default_shape_cfg.gap = gap
    builder.default_shape_cfg.ke = 0.0
    builder.default_shape_cfg.kd = 0.0
    builder.default_shape_cfg.kf = 0.0
    builder.default_shape_cfg.density = 0.0  # Use explicit mass only

    inertia_val = 1.0 / 12.0 * mass * (2.0 * box_half) ** 2 * 2.0
    b = builder.add_body(
        xform=wp.transform(
            wp.vec3(0.0, 0.0, box_half + margin + 0.001),
            wp.quat_identity(),
        ),
        mass=mass,
        inertia=wp.mat33(
            inertia_val,
            0.0,
            0.0,
            0.0,
            inertia_val,
            0.0,
            0.0,
            0.0,
            inertia_val,
        ),
    )
    builder.add_shape_box(b, hx=box_half, hy=box_half, hz=box_half)
    builder.add_ground_plane()

    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, -GRAVITY))
    return model


def _create_solver(model):
    """Create a DVI solver tuned for contact accuracy."""
    jc = newton.solvers.NumericalSolverConfig(
        solver_type=SolverType.SPARSE_LDL,
        alpha=1e6,
        reg=1e-6,
    )
    cc = newton.solvers.NumericalSolverConfig(
        solver_type=SolverType.SPARSE_JACOBI,
        max_iterations=CONTACT_ITERS,
        alpha=0.0,
        recovery_speed=1e6,
        omega=CONTACT_OMEGA,
        reg=1e-8,
        friction_projection=FrictionProjection.TANGENTIAL,
    )
    return newton.solvers.SolverDVI(
        model,
        joint_solver=jc,
        contact_solver=cc,
        angular_damping=0.0,
        enable_contacts=True,
        actuator_integration=ActuatorIntegration.EXPLICIT,
    )


def _simulate(
    model,
    solver,
    applied_force_x: float = 0.0,
    initial_vx: float = 0.0,
    sim_time: float = SIM_TIME,
    dt: float = DT,
):
    """Run simulation and return time-series data.

    Returns:
        Dictionary with arrays: t, x, z, vx, vz
    """
    n_steps = int(sim_time / dt)
    state_0 = model.state()
    state_1 = model.state()

    # Set initial velocity
    if initial_vx != 0.0:
        qd = state_0.body_qd.numpy()
        qd[0][0] = initial_vx  # vx
        state_0.body_qd.assign(qd)

    ts = np.zeros(n_steps)
    xs = np.zeros(n_steps)
    zs = np.zeros(n_steps)
    vxs = np.zeros(n_steps)
    vzs = np.zeros(n_steps)

    for step in range(n_steps):
        state_0.clear_forces()

        # Apply horizontal force to body 0
        if applied_force_x != 0.0:
            bf = state_0.body_f.numpy()
            # body_f layout: [vx, vy, vz, wx, wy, wz] (spatial wrench)
            bf[0][0] = applied_force_x
            state_0.body_f.assign(bf)

        contacts = model.collide(state_0)
        solver.step(state_0, state_1, None, contacts, dt)
        state_0, state_1 = state_1, state_0

        q = state_0.body_q.numpy()
        qd = state_0.body_qd.numpy()
        ts[step] = (step + 1) * dt
        xs[step] = q[0][0]
        zs[step] = q[0][2]
        vxs[step] = qd[0][0]
        vzs[step] = qd[0][2]

    return {"t": ts, "x": xs, "z": zs, "vx": vxs, "vz": vzs}


# ============================================================================
# Scenario runners
# ============================================================================


def scenario_static():
    """Box sitting on ground, no force applied."""
    print("\n" + "=" * 70)
    print("SCENARIO 1: Static box on ground (F = 0)")
    print("=" * 70)

    model = _create_box_on_ground()
    solver = _create_solver(model)
    data = _simulate(model, solver, applied_force_x=0.0, sim_time=1.0)

    final_z = data["z"][-1]
    final_vx = data["vx"][-1]
    final_vz = data["vz"][-1]
    final_x = data["x"][-1]
    expected_z = BOX_HALF + MARGIN

    print(f"  Expected resting z:  {expected_z:.4f} m")
    print(f"  Actual resting z:    {final_z:.4f} m  (err = {abs(final_z - expected_z):.6f} m)")
    print(f"  Final x drift:       {final_x:.6f} m")
    print(f"  Final vx:            {final_vx:.6f} m/s")
    print(f"  Final vz:            {final_vz:.6f} m/s")

    ok = abs(final_z - expected_z) < 0.005 and abs(final_vx) < 0.01 and abs(final_x) < 0.001
    print(f"  Result: {'✅ PASS' if ok else '❌ FAIL'}")
    return ok


def scenario_sub_threshold():
    """Horizontal force F < mu * m * g — box should NOT slide."""
    print("\n" + "=" * 70)
    print(f"SCENARIO 2: Sub-threshold push (F = 0.5 * mu*m*g = {0.5 * FRICTION_FORCE:.2f} N)")
    print("=" * 70)

    F_applied = 0.5 * FRICTION_FORCE
    model = _create_box_on_ground()
    solver = _create_solver(model)
    data = _simulate(model, solver, applied_force_x=F_applied, sim_time=1.0)

    final_x = data["x"][-1]
    final_vx = data["vx"][-1]
    max_vx = np.max(np.abs(data["vx"]))

    print(f"  Applied force:       {F_applied:.2f} N")
    print(f"  Friction limit:      {FRICTION_FORCE:.2f} N")
    print(f"  Final x:             {final_x:.6f} m")
    print(f"  Final vx:            {final_vx:.6f} m/s")
    print(f"  Max |vx|:            {max_vx:.6f} m/s")

    # Box should not slide appreciably
    ok = abs(final_x) < 0.01 and abs(final_vx) < 0.05
    print(f"  Result: {'✅ PASS' if ok else '❌ FAIL'}")
    return ok


def scenario_at_threshold():
    """Horizontal force F = mu * m * g — at the sliding limit."""
    print("\n" + "=" * 70)
    print(f"SCENARIO 3: At-threshold push (F = mu*m*g = {FRICTION_FORCE:.2f} N)")
    print("=" * 70)

    F_applied = FRICTION_FORCE
    model = _create_box_on_ground()
    solver = _create_solver(model)
    data = _simulate(model, solver, applied_force_x=F_applied, sim_time=1.0)

    final_x = data["x"][-1]
    final_vx = data["vx"][-1]
    max_vx = np.max(np.abs(data["vx"]))

    # Net force is F - mu*mg = 0, so acceleration = 0
    # Due to numerical effects, there might be very small drift
    print(f"  Applied force:       {F_applied:.2f} N")
    print(f"  Friction limit:      {FRICTION_FORCE:.2f} N")
    print("  Net force (theory):  0.00 N")
    print(f"  Final x:             {final_x:.6f} m")
    print(f"  Final vx:            {final_vx:.6f} m/s")
    print(f"  Max |vx|:            {max_vx:.6f} m/s")

    # Small drift is acceptable; just shouldn't accelerate like zero-friction
    ok = abs(final_x) < 0.1 and abs(final_vx) < 0.5
    print(f"  Result: {'✅ PASS' if ok else '❌ FAIL'}")
    return ok


def scenario_super_threshold():
    """Horizontal force F > mu * m * g — box should slide.

    Analytical: a = (F - mu*m*g) / m
    """
    print("\n" + "=" * 70)
    print(f"SCENARIO 4: Super-threshold push (F = 2 * mu*m*g = {2 * FRICTION_FORCE:.2f} N)")
    print("=" * 70)

    F_applied = 2.0 * FRICTION_FORCE
    model = _create_box_on_ground()
    solver = _create_solver(model)
    data = _simulate(model, solver, applied_force_x=F_applied, sim_time=SIM_TIME)

    # Analytical
    a_analytical = (F_applied - FRICTION_FORCE) / BOX_MASS  # = mu*g
    expected_vx_final = a_analytical * SIM_TIME
    expected_x_final = 0.5 * a_analytical * SIM_TIME**2

    final_x = data["x"][-1]
    final_vx = data["vx"][-1]

    # Compute acceleration from velocity slope (linear fit in last half)
    half = len(data["t"]) // 2
    coeffs = np.polyfit(data["t"][half:], data["vx"][half:], 1)
    measured_accel = coeffs[0]

    print(f"  Applied force:       {F_applied:.2f} N")
    print(f"  Friction force:      {FRICTION_FORCE:.2f} N")
    print(f"  Net force:           {F_applied - FRICTION_FORCE:.2f} N")
    print(f"  Expected accel:      {a_analytical:.4f} m/s²")
    print(f"  Measured accel:      {measured_accel:.4f} m/s² (err = {abs(measured_accel - a_analytical):.4f})")
    print(f"  Expected final vx:   {expected_vx_final:.4f} m/s")
    print(f"  Actual final vx:     {final_vx:.4f} m/s (err = {abs(final_vx - expected_vx_final):.4f})")
    print(f"  Expected final x:    {expected_x_final:.4f} m")
    print(f"  Actual final x:      {final_x:.4f} m (err = {abs(final_x - expected_x_final):.4f})")

    accel_err_pct = abs(measured_accel - a_analytical) / a_analytical * 100
    vx_err_pct = abs(final_vx - expected_vx_final) / expected_vx_final * 100

    print(f"  Acceleration error:  {accel_err_pct:.1f}%")
    print(f"  Velocity error:      {vx_err_pct:.1f}%")

    ok = accel_err_pct < 10 and vx_err_pct < 10
    print(f"  Result: {'✅ PASS' if ok else '❌ FAIL'}")
    return ok


def scenario_sliding_deceleration():
    """Box with initial velocity, no external force. Should decelerate at mu*g.

    Analytical: a = -mu * g, stops at t_stop = v0 / (mu*g)
    """
    print("\n" + "=" * 70)
    print("SCENARIO 5: Sliding box (v0 = 5 m/s, F = 0)")
    print("=" * 70)

    v0 = 5.0
    model = _create_box_on_ground()
    solver = _create_solver(model)
    data = _simulate(model, solver, initial_vx=v0, sim_time=SIM_TIME)

    # Analytical
    a_friction = MU * GRAVITY  # deceleration magnitude
    t_stop = v0 / a_friction
    x_stop = v0 * t_stop - 0.5 * a_friction * t_stop**2

    # Find when box stops in simulation
    stop_idx = np.argmax(data["vx"] <= 0.01)
    if stop_idx == 0 and data["vx"][0] > 0.01:
        stop_idx = len(data["vx"]) - 1
    sim_t_stop = data["t"][stop_idx]
    sim_x_stop = data["x"][stop_idx]

    # Measure deceleration from the steady sliding phase (skip initial bounce).
    # The box starts slightly above ground and bounces for ~0.3s before settling
    # into steady contact sliding. Fit the linear velocity region after settling.
    settle_idx = int(0.4 / DT)  # skip first 0.4s of settling
    fit_end = max(settle_idx + 10, int(0.7 * stop_idx))
    if fit_end > stop_idx:
        fit_end = stop_idx
    if fit_end > settle_idx + 5:
        coeffs = np.polyfit(data["t"][settle_idx:fit_end], data["vx"][settle_idx:fit_end], 1)
        measured_decel = -coeffs[0]
    else:
        measured_decel = 0.0

    print(f"  Initial velocity:    {v0:.2f} m/s")
    print(f"  Expected decel:      {a_friction:.4f} m/s²")
    print(f"  Measured decel:      {measured_decel:.4f} m/s² (err = {abs(measured_decel - a_friction):.4f})")
    print(f"  Expected t_stop:     {t_stop:.4f} s")
    print(f"  Simulated t_stop:    {sim_t_stop:.4f} s (err = {abs(sim_t_stop - t_stop):.4f} s)")
    print(f"  Expected x_stop:     {x_stop:.4f} m")
    print(f"  Simulated x_stop:    {sim_x_stop:.4f} m (err = {abs(sim_x_stop - x_stop):.4f} m)")

    decel_err_pct = abs(measured_decel - a_friction) / a_friction * 100
    t_err_pct = abs(sim_t_stop - t_stop) / t_stop * 100
    x_err_pct = abs(sim_x_stop - x_stop) / x_stop * 100

    print(f"  Deceleration error:  {decel_err_pct:.1f}%")
    print(f"  Stop time error:     {t_err_pct:.1f}%")
    print(f"  Stop position error: {x_err_pct:.1f}%")

    ok = t_err_pct < 5 and x_err_pct < 5
    print(f"  Result: {'✅ PASS' if ok else '❌ FAIL'}")
    return ok


def scenario_sliding_with_push():
    """Sliding box with sub-threshold push opposing friction.

    v0 = 5 m/s in +x. Applied force F in +x, F < mu*m*g.
    Kinetic friction = mu*m*g in -x direction (opposing motion).
    Net decel = (mu*m*g - F) / m = mu*g - F/m
    """
    print("\n" + "=" * 70)
    F_applied = 0.3 * FRICTION_FORCE
    print(f"SCENARIO 6: Sliding box + push (v0=5, F={F_applied:.2f} N < {FRICTION_FORCE:.2f} N)")
    print("=" * 70)

    v0 = 5.0
    model = _create_box_on_ground()
    solver = _create_solver(model)
    data = _simulate(model, solver, initial_vx=v0, applied_force_x=F_applied, sim_time=SIM_TIME)

    # Analytical while sliding
    net_force = FRICTION_FORCE - F_applied  # net retarding force
    a_net = net_force / BOX_MASS  # deceleration magnitude
    t_stop = v0 / a_net
    x_stop = v0 * t_stop - 0.5 * a_net * t_stop**2

    # Find stop in simulation
    stop_idx = np.argmax(data["vx"] <= 0.01)
    if stop_idx == 0 and data["vx"][0] > 0.01:
        stop_idx = len(data["vx"]) - 1
    sim_t_stop = data["t"][stop_idx] if stop_idx < len(data["t"]) - 1 else data["t"][-1]
    sim_x_stop = data["x"][stop_idx]

    # Measure deceleration from steady sliding phase (skip bounce)
    settle_idx = int(0.4 / DT)
    fit_end = max(settle_idx + 10, int(0.7 * stop_idx))
    if fit_end > stop_idx:
        fit_end = stop_idx
    if fit_end > settle_idx + 5:
        coeffs = np.polyfit(data["t"][settle_idx:fit_end], data["vx"][settle_idx:fit_end], 1)
        measured_decel = -coeffs[0]
    else:
        measured_decel = 0.0

    print(f"  Friction force:      {FRICTION_FORCE:.2f} N (opposing motion)")
    print(f"  Applied force:       {F_applied:.2f} N (with motion)")
    print(f"  Net retarding force: {net_force:.2f} N")
    print(f"  Expected decel:      {a_net:.4f} m/s²")
    print(f"  Measured decel:      {measured_decel:.4f} m/s² (err = {abs(measured_decel - a_net):.4f})")
    print(f"  Expected t_stop:     {t_stop:.4f} s")
    print(f"  Simulated t_stop:    {sim_t_stop:.4f} s (err = {abs(sim_t_stop - t_stop):.4f} s)")
    print(f"  Expected x_stop:     {x_stop:.4f} m")
    print(f"  Simulated x_stop:    {sim_x_stop:.4f} m (err = {abs(sim_x_stop - x_stop):.4f} m)")

    # After stopping, the applied force (0.3 * mu*mg) is below threshold,
    # so box should remain stationary
    if t_stop < SIM_TIME and stop_idx < len(data["vx"]) - 10:
        vx_after_stop = np.mean(np.abs(data["vx"][stop_idx + 5 :]))
        print(f"  vx after stop:       {vx_after_stop:.6f} m/s (should be ~0)")
        stays_put = vx_after_stop < 0.05
    else:
        stays_put = True
        print("  (box still sliding at end of sim)")

    t_err_pct = abs(sim_t_stop - t_stop) / t_stop * 100
    print(f"  Stop time error:     {t_err_pct:.1f}%")

    ok = t_err_pct < 5 and stays_put
    print(f"  Result: {'✅ PASS' if ok else '❌ FAIL'}")
    return ok


def scenario_mu_sweep():
    """Sweep friction coefficient with super-threshold force.

    For each mu, apply F = 2 * mu * m * g (always 2x threshold).
    Expected acceleration a = mu * g (net = F - mu*mg = mu*mg, a = mu*g).
    """
    print("\n" + "=" * 70)
    print("SCENARIO 7: Friction coefficient sweep")
    print("=" * 70)

    all_ok = True
    print(
        f"  {'mu':>6s} | {'F_applied':>10s} | {'a_expected':>10s} | {'a_measured':>10s} | {'err%':>6s} | {'result':>6s}"
    )
    print(f"  {'-' * 6} | {'-' * 10} | {'-' * 10} | {'-' * 10} | {'-' * 6} | {'-' * 6}")

    for mu in [0.1, 0.3, 0.5, 0.7, 1.0]:
        friction = mu * BOX_MASS * GRAVITY
        F_applied = 2.0 * friction
        a_expected = mu * GRAVITY

        model = _create_box_on_ground(mu=mu)
        solver = _create_solver(model)
        data = _simulate(model, solver, applied_force_x=F_applied, sim_time=1.0, dt=DT)

        # Linear fit for acceleration
        half = len(data["t"]) // 2
        coeffs = np.polyfit(data["t"][half:], data["vx"][half:], 1)
        a_measured = coeffs[0]

        err_pct = abs(a_measured - a_expected) / a_expected * 100
        ok = err_pct < 10
        if not ok:
            all_ok = False

        status = "✅" if ok else "❌"
        print(f"  {mu:6.2f} | {F_applied:10.2f} | {a_expected:10.4f} | {a_measured:10.4f} | {err_pct:5.1f}% | {status}")

    print(f"  Overall: {'✅ PASS' if all_ok else '❌ FAIL'}")
    return all_ok


def scenario_mujoco_comparison():
    """Compare DVI vs MuJoCo for the super-threshold push.

    Uses a pushed box (not sliding) since MuJoCo Euler integrator handles
    initial contact sliding differently (implicit friction in Newton solver
    vs explicit Jacobi in DVI). The push scenario is more comparable.
    """
    print("\n" + "=" * 70)
    print("SCENARIO 8: MuJoCo comparison (pushed box, F=2*mu*m*g)")
    print("=" * 70)

    F_applied = 2.0 * FRICTION_FORCE
    results = {}

    for solver_name in ["dvi", "mujoco"]:
        model = _create_box_on_ground()

        if solver_name == "mujoco":
            try:
                solver = newton.solvers.SolverMuJoCo(
                    model,
                    solver="newton",
                    integrator="euler",
                    iterations=50,
                    ls_iterations=20,
                )
            except Exception as e:
                print(f"  MuJoCo solver unavailable: {e}")
                return True  # skip
        else:
            solver = _create_solver(model)

        # Let it settle first
        state_0 = model.state()
        state_1 = model.state()
        for _ in range(100):
            state_0.clear_forces()
            contacts = model.collide(state_0)
            solver.step(state_0, state_1, None, contacts, DT)
            state_0, state_1 = state_1, state_0

        # Now push
        n_steps = int(1.0 / DT)
        vxs = np.zeros(n_steps)
        ts = np.zeros(n_steps)

        for step in range(n_steps):
            state_0.clear_forces()
            bf = state_0.body_f.numpy()
            bf[0][0] = F_applied
            state_0.body_f.assign(bf)
            contacts = model.collide(state_0)
            solver.step(state_0, state_1, None, contacts, DT)
            state_0, state_1 = state_1, state_0
            ts[step] = (step + 1) * DT
            vxs[step] = state_0.body_qd.numpy()[0][0]

        results[solver_name] = {"t": ts, "vx": vxs}

    # Compare accelerations
    if "mujoco" in results and "dvi" in results:
        half = len(ts) // 2
        coeffs_mj = np.polyfit(results["mujoco"]["t"][half:], results["mujoco"]["vx"][half:], 1)
        coeffs_ch = np.polyfit(results["dvi"]["t"][half:], results["dvi"]["vx"][half:], 1)
        accel_mj = coeffs_mj[0]
        accel_ch = coeffs_ch[0]
        expected_accel = (F_applied - FRICTION_FORCE) / BOX_MASS

        print(f"  MuJoCo accel:        {accel_mj:.4f} m/s²")
        print(f"  DVI accel:        {accel_ch:.4f} m/s²")
        print(f"  Expected accel:      {expected_accel:.4f} m/s²")

        vx_diff = np.abs(results["mujoco"]["vx"] - results["dvi"]["vx"])
        max_diff = np.max(vx_diff)
        mean_diff = np.mean(vx_diff)
        print(f"  Max vx difference:   {max_diff:.6f} m/s")
        print(f"  Mean vx difference:  {mean_diff:.6f} m/s")

        ok = max_diff < 1.0 and abs(accel_mj - accel_ch) / expected_accel < 0.1
        print(f"  Result: {'✅ PASS' if ok else '❌ FAIL'}")
        return ok

    return True


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    wp.init()

    results = []
    results.append(("Static box", scenario_static()))
    results.append(("Sub-threshold push", scenario_sub_threshold()))
    results.append(("At-threshold push", scenario_at_threshold()))
    results.append(("Super-threshold push", scenario_super_threshold()))
    results.append(("Sliding deceleration", scenario_sliding_deceleration()))
    results.append(("Sliding with push", scenario_sliding_with_push()))
    results.append(("Mu sweep", scenario_mu_sweep()))
    results.append(("MuJoCo comparison", scenario_mujoco_comparison()))

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    all_pass = True
    for name, ok in results:
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"  {name:30s} {status}")
        if not ok:
            all_pass = False

    print(f"\nOverall: {'✅ ALL PASS' if all_pass else '❌ SOME FAILED'}")
    sys.exit(0 if all_pass else 1)
