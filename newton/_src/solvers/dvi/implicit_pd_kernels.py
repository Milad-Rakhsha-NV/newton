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
Implicit PD (stiffness/damping) treatment for the DVI DVI solver.

Overview
--------
Joint stiffness (ke) and damping (kd) are normally applied as explicit forces
in the actuation kernel.  Explicit treatment of stiff springs requires small
timesteps for stability (CFL-like condition), forcing the solver to use many
substeps.  For the nv_humanoid with K=20, substeps=4 is needed, making DVI
3.4x slower than MuJoCo/XPBD.

This module implements **implicit treatment** by folding the damping (and
optionally linearised stiffness) into the effective mass/inertia matrix:

    M_eff = M + h * C_eff * J_free^T * J_free

where C_eff = kd + h * ke  (damping + linearised stiffness contribution),
h is the timestep, and J_free is the free-DOF Jacobian for each joint.

Mathematical Derivation
-----------------------
Consider a joint DOF with spring-damper:

    tau = -ke * (q - q_target) - kd * (qdot - qdot_target)

The damping term  -kd * qdot  contributes  ∂tau/∂qdot = -kd  to the velocity
Jacobian.  To treat this implicitly, we absorb it into the mass matrix:

    (M + h * kd * J^T * J) * v^{n+1} = M * v^n + h * f_ext

For stiffness, a first-order linearisation about the current state gives:

    f_spring(q^{n+1}) ≈ f_spring(q^n) + ke * h * v^{n+1}

This looks exactly like a velocity-proportional damping with coefficient h*ke,
so the effective damping becomes:

    C_eff = kd + h * ke

And the augmented mass is:

    M_eff = M + h * C_eff * J_free^T * J_free
          = M + h * kd * J^T * J + h^2 * ke * J^T * J

This matches the diagonal blocks of DVI's HHT system matrix (M + h*C + h^2*K)
projected onto the body diagonal.

Cross-Body Coupling (Dropped Approximation)
--------------------------------------------
In maximal coordinates, joint damping couples two bodies:

    f_A = -C * (omega_A - omega_B) . e * e
    f_B = +C * (omega_A - omega_B) . e * e

The full augmented mass matrix has off-diagonal blocks:

    M_aug = [ M_A + h*C*(e⊗e)    -h*C*(e⊗e)     ]
            [ -h*C*(e⊗e)         M_B + h*C*(e⊗e)  ]

We keep only the diagonal blocks (per-body), dropping the cross-body terms.
This is the same approximation MuJoCo uses successfully.  The effect is a
slight over-damping: each body is treated as if the other is fixed.

Phase 1 vs Phase 2
-------------------
**Phase 1 (this implementation):** Angular free DOFs only.

For angular free DOFs (revolute, ball, D6-angular), the free-DOF Jacobian is:
    J = [0, 0, 0, ±e_x, ±e_y, ±e_z]    (pure angular)

So the augmentation is purely on the 3x3 rotational inertia:
    I_eff = I_world + h * sum_j(C_eff_j * e_j ⊗ e_j)

No translational mass change, no translation-rotation cross-coupling.
This covers ALL joints in the nv_humanoid (verified: 0 linear free DOFs).

**Phase 2 (future):** Linear free DOFs (prismatic, D6-linear).

For linear free DOFs, the Jacobian is:
    J = [±e_x, ±e_y, ±e_z, ±(r×e)_x, ±(r×e)_y, ±(r×e)_z]

This creates:
- Translational mass augmentation:  h*C * (e ⊗ e)  (3x3, mass becomes a tensor)
- Rotational inertia augmentation:  h*C * (r×e) ⊗ (r×e)  (3x3)
- Cross-coupling:  h*C * e ⊗ (r×e)  (breaks translation/rotation independence)

Implementing this requires switching from separate (scalar inv_mass, mat33 inv_I)
to a single 6x6 M_body_inv, touching ~90 kernel call sites.  Deferred.
"""

import warp as wp

from newton._src.solvers.dvi.actuation_kernels import compute_revolute_angle

# =============================================================================
# Kernel 1: Accumulate per-body angular inertia augmentation from joint damping
# =============================================================================


@wp.kernel
def accumulate_angular_damping_augmentation(
    body_q: wp.array[wp.transform],
    joint_type: wp.array[int],
    joint_enabled: wp.array[bool],
    joint_parent: wp.array[int],
    joint_child: wp.array[int],
    joint_X_p: wp.array[wp.transform],
    joint_axis: wp.array[wp.vec3],
    joint_qd_start: wp.array[int],
    joint_dof_dim: wp.array2d[int],
    joint_target_ke: wp.array[float],
    joint_target_kd: wp.array[float],
    dt: float,
    # outputs
    body_inertia_augment: wp.array[wp.mat33],
):
    """Accumulate implicit damping/stiffness augmentation per body (angular DOFs only).

    For each joint, computes the world-frame free-DOF axes and accumulates:

        body_inertia_augment[child] += h * C_eff * (axis_world ⊗ axis_world)

    where C_eff = kd + h * ke, and axis_world is the joint axis rotated to world frame.

    **Child-only augmentation:** The augmentation is applied only to the child
    body of each joint.  Augmenting both parent and child would double-count
    the inertia contribution for bodies that are children of one joint and
    parents of another (e.g., interior links in serial chains).  This caused
    multi-body chains to move ~4× too slowly with armature.

    **Phase 1 limitation:** Only handles angular free DOFs.  Linear free DOFs
    (prismatic, D6-linear) are skipped with zero augmentation — their implicit
    treatment requires full 6x6 per-body mass matrix (Phase 2).

    Joint type handling:
        - PRISMATIC (0):  1 linear DOF → skipped (Phase 2)
        - REVOLUTE (1):   1 angular DOF → augmented
        - BALL (2):       3 angular DOFs → all augmented
        - FIXED (3):      0 DOFs → skipped
        - FREE (4):       6 DOFs → skipped (handled by constraint solver)
        - COMPOUND (5):   skipped (not standard)
        - D6 (6):         n_linear skipped, n_angular augmented

    Args:
        body_q: Body transforms [body_count].
        joint_type: Joint types [joint_count].
        joint_enabled: Joint enabled flags [joint_count].
        joint_parent: Parent body index per joint (-1 for world) [joint_count].
        joint_child: Child body index per joint [joint_count].
        joint_X_p: Joint frame in parent body [joint_count].
        joint_axis: Joint axes in local frame [joint_dof_count, 3].
        joint_qd_start: Starting DOF index per joint [joint_count].
        joint_dof_dim: (n_linear, n_angular) per joint [joint_count, 2].
        joint_target_ke: Stiffness per DOF [joint_dof_count].
        joint_target_kd: Damping per DOF [joint_dof_count].
        dt: Timestep.

    Outputs:
        body_inertia_augment: Per-body 3x3 inertia augmentation (atomic-added) [body_count].
    """
    tid = wp.tid()

    if not joint_enabled[tid]:
        return

    jtype = joint_type[tid]

    # Fixed, FREE, and COMPOUND joints — no angular augmentation
    if jtype == 3 or jtype == 4 or jtype == 5:
        return

    id_p = joint_parent[tid]
    id_c = joint_child[tid]

    qd_start = joint_qd_start[tid]

    # Get parent orientation for rotating axes to world frame.
    # The actuation kernel uses q_p = rotation of the parent joint frame in world,
    # which is transform_get_rotation(X_wp * X_pj).  We replicate this exactly.
    if id_p < 0:
        X_wp = body_q[id_c]  # world body: use child (convention from actuation kernel)
    else:
        X_wp = body_q[id_p]

    X_pj = joint_X_p[tid]
    X_wj_p = wp.transform_multiply(X_wp, X_pj)
    q_p = wp.transform_get_rotation(X_wj_p)

    # -------------------------------------------------------------------------
    # REVOLUTE (type 1): 1 angular DOF
    # -------------------------------------------------------------------------
    if jtype == 1:
        axis = joint_axis[qd_start]
        axis_w = wp.quat_rotate(q_p, axis)

        ke = joint_target_ke[qd_start]
        kd = joint_target_kd[qd_start]
        c_eff = kd + dt * ke
        if c_eff > 0.0:
            aug = dt * c_eff * wp.outer(axis_w, axis_w)
            wp.atomic_add(body_inertia_augment, id_c, aug)

    # -------------------------------------------------------------------------
    # BALL (type 2): 3 angular DOFs
    # -------------------------------------------------------------------------
    elif jtype == 2:
        # Unrolled loop for 3 DOFs (Warp requires compile-time bounds)
        # DOF 0
        axis0 = joint_axis[qd_start + 0]
        axis_w0 = wp.quat_rotate(q_p, axis0)
        ke0 = joint_target_ke[qd_start + 0]
        kd0 = joint_target_kd[qd_start + 0]
        c_eff0 = kd0 + dt * ke0
        if c_eff0 > 0.0:
            aug0 = dt * c_eff0 * wp.outer(axis_w0, axis_w0)
            wp.atomic_add(body_inertia_augment, id_c, aug0)

        # DOF 1
        axis1 = joint_axis[qd_start + 1]
        axis_w1 = wp.quat_rotate(q_p, axis1)
        ke1 = joint_target_ke[qd_start + 1]
        kd1 = joint_target_kd[qd_start + 1]
        c_eff1 = kd1 + dt * ke1
        if c_eff1 > 0.0:
            aug1 = dt * c_eff1 * wp.outer(axis_w1, axis_w1)
            wp.atomic_add(body_inertia_augment, id_c, aug1)

        # DOF 2
        axis2 = joint_axis[qd_start + 2]
        axis_w2 = wp.quat_rotate(q_p, axis2)
        ke2 = joint_target_ke[qd_start + 2]
        kd2 = joint_target_kd[qd_start + 2]
        c_eff2 = kd2 + dt * ke2
        if c_eff2 > 0.0:
            aug2 = dt * c_eff2 * wp.outer(axis_w2, axis_w2)
            wp.atomic_add(body_inertia_augment, id_c, aug2)

    # -------------------------------------------------------------------------
    # PRISMATIC (type 0): 1 linear DOF — Phase 2 (skipped)
    # Linear free DOFs require full 6x6 body mass matrix for correct treatment.
    # The damping Jacobian has both linear and angular (lever arm) components:
    #     J = [±e, ±(r × e)]
    # which creates cross-coupling between translation and rotation.
    # -------------------------------------------------------------------------
    elif jtype == 0:
        pass  # Phase 2: needs 6x6 M_body

    # -------------------------------------------------------------------------
    # D6 (type 6): Variable DOFs — angular augmented, linear skipped (Phase 2)
    # -------------------------------------------------------------------------
    elif jtype == 6:
        n_linear = joint_dof_dim[tid, 0]
        n_angular = joint_dof_dim[tid, 1]

        # Skip linear DOFs (Phase 2)
        # Only process angular DOFs (offset by n_linear in the DOF array)

        # Angular DOF 0
        if n_angular >= 1:
            axis = joint_axis[qd_start + n_linear + 0]
            axis_w = wp.quat_rotate(q_p, axis)
            ke = joint_target_ke[qd_start + n_linear + 0]
            kd = joint_target_kd[qd_start + n_linear + 0]
            c_eff = kd + dt * ke
            if c_eff > 0.0:
                aug = dt * c_eff * wp.outer(axis_w, axis_w)
                wp.atomic_add(body_inertia_augment, id_c, aug)

        # Angular DOF 1
        if n_angular >= 2:
            axis = joint_axis[qd_start + n_linear + 1]
            axis_w = wp.quat_rotate(q_p, axis)
            ke = joint_target_ke[qd_start + n_linear + 1]
            kd = joint_target_kd[qd_start + n_linear + 1]
            c_eff = kd + dt * ke
            if c_eff > 0.0:
                aug = dt * c_eff * wp.outer(axis_w, axis_w)
                wp.atomic_add(body_inertia_augment, id_c, aug)

        # Angular DOF 2
        if n_angular >= 3:
            axis = joint_axis[qd_start + n_linear + 2]
            axis_w = wp.quat_rotate(q_p, axis)
            ke = joint_target_ke[qd_start + n_linear + 2]
            kd = joint_target_kd[qd_start + n_linear + 2]
            c_eff = kd + dt * ke
            if c_eff > 0.0:
                aug = dt * c_eff * wp.outer(axis_w, axis_w)
                wp.atomic_add(body_inertia_augment, id_c, aug)


# =============================================================================
# Kernel 2: Compute augmented world-frame inverse inertia
# =============================================================================


@wp.kernel
def compute_body_inv_inertia_world_augmented(
    body_q: wp.array[wp.transform],
    body_inertia: wp.array[wp.mat33],
    body_inv_mass: wp.array[float],
    body_inertia_augment: wp.array[wp.mat33],
    # outputs
    body_inv_inertia_world: wp.array[wp.mat33],
):
    """Compute world-frame inverse inertia with implicit PD augmentation.

    For each dynamic body:

        I_world = R * I_body * R^T              (standard rotation to world frame)
        I_eff   = I_world + body_inertia_augment (add accumulated implicit PD terms)
        result  = inverse(I_eff)                 (3x3 matrix inverse)

    The augmentation term ``body_inertia_augment`` was previously accumulated by
    ``accumulate_angular_damping_augmentation`` and contains:

        sum_j [ h * C_eff_j * (axis_world_j ⊗ axis_world_j) ]

    for all joints connected to this body.

    **Why we need I_body (not I_body^{-1}):**
    Normally ``compute_body_inv_inertia_world`` directly rotates ``I_body^{-1}``
    to world frame: ``R * I_body^{-1} * R^T``.  But with augmentation we need
    to ADD to I_world before inverting, which requires the non-inverted inertia.
    We use ``model.body_inertia`` (already stored for gyroscopic torque).

    Args:
        body_q: Body transforms [body_count].
        body_inertia: Body-frame inertia tensors (NOT inverse) [body_count].
        body_inv_mass: Inverse masses [body_count].
        body_inertia_augment: Per-body 3x3 augmentation in world frame [body_count].

    Outputs:
        body_inv_inertia_world: Augmented world-frame inverse inertia [body_count].
    """
    tid = wp.tid()

    if body_inv_mass[tid] == 0.0:
        # Fixed body — zero inverse inertia
        body_inv_inertia_world[tid] = wp.mat33(0.0)
        return

    rot = wp.transform_get_rotation(body_q[tid])
    R = wp.quat_to_matrix(rot)

    # Compute I_world = R * I_body * R^T
    I_body = body_inertia[tid]
    I_world = R * I_body * wp.transpose(R)

    # Add implicit PD augmentation (already in world frame)
    I_eff = I_world + body_inertia_augment[tid]

    # Invert the 3x3 effective inertia in double precision.
    # Armature and implicit PD augmentation add rank-1 outer products
    # (armature * axis ⊗ axis) that can exceed body inertia by 500,000×
    # for lightweight distal links (e.g. humanoid fingertips with
    # I_body ~ 1e-7 and armature ~ 0.05).  This creates condition
    # numbers ~ 1e6 in I_eff.  Float32 wp.inverse() produces ~0.5%
    # relative error at cond ~ 1e6, which snowballs through the
    # block-sparse LDL factorization and causes NaN.
    # Promoting to float64 for the 3x3 inversion eliminates this.
    I_eff_d = wp.mat33d(
        wp.float64(I_eff[0, 0]), wp.float64(I_eff[0, 1]), wp.float64(I_eff[0, 2]),
        wp.float64(I_eff[1, 0]), wp.float64(I_eff[1, 1]), wp.float64(I_eff[1, 2]),
        wp.float64(I_eff[2, 0]), wp.float64(I_eff[2, 1]), wp.float64(I_eff[2, 2]),
    )
    I_inv_d = wp.inverse(I_eff_d)
    body_inv_inertia_world[tid] = wp.mat33(
        wp.float32(I_inv_d[0, 0]), wp.float32(I_inv_d[0, 1]), wp.float32(I_inv_d[0, 2]),
        wp.float32(I_inv_d[1, 0]), wp.float32(I_inv_d[1, 1]), wp.float32(I_inv_d[1, 2]),
        wp.float32(I_inv_d[2, 0]), wp.float32(I_inv_d[2, 1]), wp.float32(I_inv_d[2, 2]),
    )


# =============================================================================
# Kernel 3: Accumulate joint armature augmentation per body
# =============================================================================


@wp.kernel
def accumulate_armature_augmentation(
    body_q: wp.array[wp.transform],
    joint_type: wp.array[int],
    joint_enabled: wp.array[bool],
    joint_parent: wp.array[int],
    joint_child: wp.array[int],
    joint_X_p: wp.array[wp.transform],
    joint_axis: wp.array[wp.vec3],
    joint_qd_start: wp.array[int],
    joint_dof_dim: wp.array2d[int],
    joint_armature: wp.array[float],
    dt: float,
    # outputs
    body_inertia_augment: wp.array[wp.mat33],
):
    """Accumulate **isotropic** armature augmentation per body (angular DOFs only).

    For each joint, accumulates:

        body_inertia_augment[child] += armature * I₃

    where ``I₃`` is the 3×3 identity matrix.  This adds the armature uniformly
    to all three rotational axes of the child body, regardless of the joint
    axis orientation.

    **Why isotropic?**
    In MuJoCo's joint-space (Featherstone) formulation, armature is added to
    the diagonal of the joint-space mass matrix: ``M_jj += armature``.  Through
    the Jacobian mapping ``M = J^T * I_spatial * J``, this effectively
    regularises the full spatial inertia of the child body, not just the
    component along the joint axis.

    The previous *directional* augmentation ``armature * (axis ⊗ axis)`` only
    added inertia along the joint's DoF axis.  For bodies with tiny moments of
    inertia on axes *perpendicular* to the joint axis (e.g. humanoid ankles
    with I_z = 0.000214 kg⋅m²), the perpendicular axes received zero
    augmentation, leading to inv(I) ~ 4600 on those axes.  Any disturbance
    torque on the unprotected axis produced ~500,000 rad/s² angular
    acceleration, causing immediate NaN blow-up in the LDL solver.

    Isotropic augmentation ensures that **all** rotational axes of the child
    body benefit from the armature regularisation, matching MuJoCo's effective
    behaviour and eliminating the instability.

    **No ``dt`` scaling** — armature is a constant inertia addition, not a
    semi-implicit damping term.  Scaling by ``dt`` made armature vanish at
    small timesteps, producing bodies ~50-500× lighter than MuJoCo and
    causing explicit PD instability.

    **Child-only augmentation:** The augmentation is applied only to the child
    body of each joint.  Augmenting both parent and child double-counts the
    effective inertia for interior links in serial chains, causing multi-body
    articulations (double pendulums, humanoid legs) to move too slowly.

    For multi-DOF joints (BALL, D6), the maximum armature across all angular
    DOFs is used for the isotropic augmentation, since ``arm * I₃`` already
    covers all axes.
    """
    tid = wp.tid()
    jtype = joint_type[tid]
    if not joint_enabled[tid]:
        return
    # Skip FREE and COMPOUND (no bilateral constraints = no joint DOFs)
    if jtype == 4 or jtype == 5:
        return

    id_c = joint_child[tid]
    qd_start = joint_qd_start[tid]

    # REVOLUTE (1 angular free DOF)
    if jtype == 1:
        arm = joint_armature[qd_start]
        if arm > 0.0:
            aug = arm * wp.identity(n=3, dtype=float)
            wp.atomic_add(body_inertia_augment, id_c, aug)

    # BALL (3 angular free DOFs) — use max armature across DOFs
    elif jtype == 2:
        arm = wp.max(joint_armature[qd_start + 0],
                     wp.max(joint_armature[qd_start + 1],
                            joint_armature[qd_start + 2]))
        if arm > 0.0:
            aug = arm * wp.identity(n=3, dtype=float)
            wp.atomic_add(body_inertia_augment, id_c, aug)

    # D6 (variable angular free DOFs) — use max armature across angular DOFs
    elif jtype == 6:
        n_linear = joint_dof_dim[tid, 0]
        n_angular = joint_dof_dim[tid, 1]

        arm = float(0.0)
        if n_angular >= 1:
            arm = wp.max(arm, joint_armature[qd_start + n_linear + 0])
        if n_angular >= 2:
            arm = wp.max(arm, joint_armature[qd_start + n_linear + 1])
        if n_angular >= 3:
            arm = wp.max(arm, joint_armature[qd_start + n_linear + 2])

        if arm > 0.0:
            aug = arm * wp.identity(n=3, dtype=float)
            wp.atomic_add(body_inertia_augment, id_c, aug)


# =============================================================================
# Kernel 4: Implicit velocity correction for exact prediction
# =============================================================================


@wp.kernel
def compute_implicit_velocity_correction(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    joint_type: wp.array[int],
    joint_enabled: wp.array[bool],
    joint_parent: wp.array[int],
    joint_child: wp.array[int],
    joint_X_p: wp.array[wp.transform],
    joint_axis: wp.array[wp.vec3],
    joint_qd_start: wp.array[int],
    joint_dof_dim: wp.array2d[int],
    joint_target_ke: wp.array[float],
    dt: float,
    # outputs
    body_f: wp.array[wp.spatial_vector],
):
    """Add implicit velocity correction for IMPLICIT actuator integration.

    For IMPLICIT mode, the exact prediction step requires:

        v_pred = v + dt * M_tilde_inv * (f_ext + G^T*b_act/dt - C_eff*G^T*G*v)

    The existing actuation kernel already provides:

        f_PD = G^T*b_act/dt - kd*G^T*G*v

    This kernel adds the missing term:

        -dt * ke * G^T * G * v   per actuated angular DOF

    For angular DOF j with world-frame axis e:

        qdot_j = dot(e, omega_child) - dot(e, omega_parent)
        correction = -dt * ke_j * qdot_j
        body_f[child].angular += correction * e
        body_f[parent].angular -= correction * e    (if parent exists)

    **Phase 1 limitation:** Only angular free DOFs are handled.  Linear free
    DOFs (prismatic, D6-linear) are skipped.

    Joint type handling (same as accumulate_angular_damping_augmentation):
        - PRISMATIC (0):  1 linear DOF -> skipped (Phase 2)
        - REVOLUTE (1):   1 angular DOF -> corrected
        - BALL (2):       3 angular DOFs -> all corrected
        - FIXED (3):      0 DOFs -> skipped
        - FREE (4):       6 DOFs -> skipped
        - COMPOUND (5):   skipped
        - D6 (6):         n_linear skipped, n_angular corrected

    Args:
        body_q: Body transforms [body_count].
        body_qd: Body spatial velocities [body_count].
        joint_type: Joint types [joint_count].
        joint_enabled: Joint enabled flags [joint_count].
        joint_parent: Parent body index per joint (-1 for world) [joint_count].
        joint_child: Child body index per joint [joint_count].
        joint_X_p: Joint frame in parent body [joint_count].
        joint_axis: Joint axes in local frame [joint_dof_count, 3].
        joint_qd_start: Starting DOF index per joint [joint_count].
        joint_dof_dim: (n_linear, n_angular) per joint [joint_count, 2].
        joint_target_ke: Stiffness per DOF [joint_dof_count].
        dt: Timestep.

    Outputs:
        body_f: Per-body spatial force (atomic-added) [body_count].
    """
    tid = wp.tid()

    if not joint_enabled[tid]:
        return

    jtype = joint_type[tid]

    # Fixed, FREE, and COMPOUND joints -- no angular correction
    if jtype == 3 or jtype == 4 or jtype == 5:
        return

    id_p = joint_parent[tid]
    id_c = joint_child[tid]

    qd_start = joint_qd_start[tid]

    # Get parent orientation for rotating axes to world frame (same as damping kernel)
    if id_p < 0:
        X_wp = body_q[id_c]
    else:
        X_wp = body_q[id_p]

    X_pj = joint_X_p[tid]
    X_wj_p = wp.transform_multiply(X_wp, X_pj)
    q_p = wp.transform_get_rotation(X_wj_p)

    # Get angular velocities
    omega_c = wp.spatial_bottom(body_qd[id_c])
    omega_p = wp.vec3(0.0, 0.0, 0.0)
    if id_p >= 0:
        omega_p = wp.spatial_bottom(body_qd[id_p])

    # -------------------------------------------------------------------------
    # REVOLUTE (type 1): 1 angular DOF
    # -------------------------------------------------------------------------
    if jtype == 1:
        axis = joint_axis[qd_start]
        axis_w = wp.quat_rotate(q_p, axis)
        ke = joint_target_ke[qd_start]
        if ke > 0.0:
            qdot = wp.dot(axis_w, omega_c) - wp.dot(axis_w, omega_p)
            correction = -dt * ke * qdot
            f_correction = correction * axis_w
            wp.atomic_add(body_f, id_c, wp.spatial_vector(wp.vec3(0.0, 0.0, 0.0), f_correction))
            if id_p >= 0:
                wp.atomic_add(body_f, id_p, wp.spatial_vector(wp.vec3(0.0, 0.0, 0.0), -f_correction))

    # -------------------------------------------------------------------------
    # BALL (type 2): 3 angular DOFs
    # -------------------------------------------------------------------------
    elif jtype == 2:
        # DOF 0
        axis0 = joint_axis[qd_start + 0]
        axis_w0 = wp.quat_rotate(q_p, axis0)
        ke0 = joint_target_ke[qd_start + 0]
        if ke0 > 0.0:
            qdot0 = wp.dot(axis_w0, omega_c) - wp.dot(axis_w0, omega_p)
            correction0 = -dt * ke0 * qdot0
            f0 = correction0 * axis_w0
            wp.atomic_add(body_f, id_c, wp.spatial_vector(wp.vec3(0.0, 0.0, 0.0), f0))
            if id_p >= 0:
                wp.atomic_add(body_f, id_p, wp.spatial_vector(wp.vec3(0.0, 0.0, 0.0), -f0))

        # DOF 1
        axis1 = joint_axis[qd_start + 1]
        axis_w1 = wp.quat_rotate(q_p, axis1)
        ke1 = joint_target_ke[qd_start + 1]
        if ke1 > 0.0:
            qdot1 = wp.dot(axis_w1, omega_c) - wp.dot(axis_w1, omega_p)
            correction1 = -dt * ke1 * qdot1
            f1 = correction1 * axis_w1
            wp.atomic_add(body_f, id_c, wp.spatial_vector(wp.vec3(0.0, 0.0, 0.0), f1))
            if id_p >= 0:
                wp.atomic_add(body_f, id_p, wp.spatial_vector(wp.vec3(0.0, 0.0, 0.0), -f1))

        # DOF 2
        axis2 = joint_axis[qd_start + 2]
        axis_w2 = wp.quat_rotate(q_p, axis2)
        ke2 = joint_target_ke[qd_start + 2]
        if ke2 > 0.0:
            qdot2 = wp.dot(axis_w2, omega_c) - wp.dot(axis_w2, omega_p)
            correction2 = -dt * ke2 * qdot2
            f2 = correction2 * axis_w2
            wp.atomic_add(body_f, id_c, wp.spatial_vector(wp.vec3(0.0, 0.0, 0.0), f2))
            if id_p >= 0:
                wp.atomic_add(body_f, id_p, wp.spatial_vector(wp.vec3(0.0, 0.0, 0.0), -f2))

    # -------------------------------------------------------------------------
    # PRISMATIC (type 0): 1 linear DOF -- Phase 2 (skipped)
    # -------------------------------------------------------------------------
    elif jtype == 0:
        pass  # Phase 2: needs 6x6 M_body

    # -------------------------------------------------------------------------
    # D6 (type 6): Variable DOFs -- angular corrected, linear skipped (Phase 2)
    # -------------------------------------------------------------------------
    elif jtype == 6:
        n_linear = joint_dof_dim[tid, 0]
        n_angular = joint_dof_dim[tid, 1]

        # Angular DOF 0
        if n_angular >= 1:
            axis = joint_axis[qd_start + n_linear + 0]
            axis_w = wp.quat_rotate(q_p, axis)
            ke = joint_target_ke[qd_start + n_linear + 0]
            if ke > 0.0:
                qdot = wp.dot(axis_w, omega_c) - wp.dot(axis_w, omega_p)
                correction = -dt * ke * qdot
                f_corr = correction * axis_w
                wp.atomic_add(body_f, id_c, wp.spatial_vector(wp.vec3(0.0, 0.0, 0.0), f_corr))
                if id_p >= 0:
                    wp.atomic_add(body_f, id_p, wp.spatial_vector(wp.vec3(0.0, 0.0, 0.0), -f_corr))

        # Angular DOF 1
        if n_angular >= 2:
            axis = joint_axis[qd_start + n_linear + 1]
            axis_w = wp.quat_rotate(q_p, axis)
            ke = joint_target_ke[qd_start + n_linear + 1]
            if ke > 0.0:
                qdot = wp.dot(axis_w, omega_c) - wp.dot(axis_w, omega_p)
                correction = -dt * ke * qdot
                f_corr = correction * axis_w
                wp.atomic_add(body_f, id_c, wp.spatial_vector(wp.vec3(0.0, 0.0, 0.0), f_corr))
                if id_p >= 0:
                    wp.atomic_add(body_f, id_p, wp.spatial_vector(wp.vec3(0.0, 0.0, 0.0), -f_corr))

        # Angular DOF 2
        if n_angular >= 3:
            axis = joint_axis[qd_start + n_linear + 2]
            axis_w = wp.quat_rotate(q_p, axis)
            ke = joint_target_ke[qd_start + n_linear + 2]
            if ke > 0.0:
                qdot = wp.dot(axis_w, omega_c) - wp.dot(axis_w, omega_p)
                correction = -dt * ke * qdot
                f_corr = correction * axis_w
                wp.atomic_add(body_f, id_c, wp.spatial_vector(wp.vec3(0.0, 0.0, 0.0), f_corr))
                if id_p >= 0:
                    wp.atomic_add(body_f, id_p, wp.spatial_vector(wp.vec3(0.0, 0.0, 0.0), -f_corr))


# =============================================================================
# Kernel 5: Conditional joint limit mass augmentation
# =============================================================================


@wp.kernel
def accumulate_joint_limit_augmentation(
    body_q: wp.array[wp.transform],
    joint_type: wp.array[int],
    joint_enabled: wp.array[bool],
    joint_parent: wp.array[int],
    joint_child: wp.array[int],
    joint_X_p: wp.array[wp.transform],
    joint_X_c: wp.array[wp.transform],
    joint_axis: wp.array[wp.vec3],
    joint_qd_start: wp.array[int],
    joint_dof_dim: wp.array2d[int],
    joint_limit_lower: wp.array[float],
    joint_limit_upper: wp.array[float],
    joint_limit_ke: wp.array[float],
    joint_limit_kd: wp.array[float],
    dt: float,
    # outputs
    body_inertia_augment: wp.array[wp.mat33],
):
    """Conditionally augment body inertia for joint limits (only when violated).

    For each joint DOF with finite limits, checks whether the current angle
    is outside [lower, upper].  Only when violated, accumulates:

        body_inertia_augment[child] += h * C_eff * (axis_world ⊗ axis_world)

    where C_eff = limit_kd + h * limit_ke.

    This provides implicit stability for the explicit penalty forces applied
    in the actuation kernel, matching the treatment used for PD stiffness/damping.
    The augmentation is conditional: when the limit is not violated, the body
    mass is unmodified — no spurious inertia increase during normal motion.
    """
    tid = wp.tid()

    if not joint_enabled[tid]:
        return

    jtype = joint_type[tid]
    # Skip FIXED, FREE, COMPOUND
    if jtype == 3 or jtype == 4 or jtype == 5:
        return

    id_p = joint_parent[tid]
    id_c = joint_child[tid]
    qd_start = joint_qd_start[tid]

    # Compute parent and child joint frames in world
    if id_p < 0:
        X_wp = body_q[id_c]
    else:
        X_wp = body_q[id_p]

    X_wc = body_q[id_c]
    X_pj = joint_X_p[tid]
    X_cj = joint_X_c[tid]
    X_wj_p = wp.transform_multiply(X_wp, X_pj)
    X_wj_c = wp.transform_multiply(X_wc, X_cj)
    q_p = wp.transform_get_rotation(X_wj_p)
    q_c = wp.transform_get_rotation(X_wj_c)

    # REVOLUTE (type 1): 1 angular DOF
    if jtype == 1:
        lo = joint_limit_lower[qd_start]
        hi = joint_limit_upper[qd_start]
        ke = joint_limit_ke[qd_start]
        kd = joint_limit_kd[qd_start]
        if ke > 0.0 and lo < hi:
            axis = joint_axis[qd_start]
            axis_w = wp.quat_rotate(q_p, axis)
            q_meas = compute_revolute_angle(q_p, q_c, axis)
            if q_meas < lo or q_meas > hi:
                c_eff = kd + dt * ke
                aug = dt * c_eff * wp.outer(axis_w, axis_w)
                wp.atomic_add(body_inertia_augment, id_c, aug)

    # D6 (type 6): Variable DOFs — angular only (Phase 1)
    elif jtype == 6:
        n_linear = joint_dof_dim[tid, 0]
        n_angular = joint_dof_dim[tid, 1]

        # Angular DOF 0
        if n_angular >= 1:
            dof_idx = qd_start + n_linear + 0
            lo = joint_limit_lower[dof_idx]
            hi = joint_limit_upper[dof_idx]
            ke = joint_limit_ke[dof_idx]
            kd = joint_limit_kd[dof_idx]
            if ke > 0.0 and lo < hi:
                axis = joint_axis[dof_idx]
                axis_w = wp.quat_rotate(q_p, axis)
                q_meas = compute_revolute_angle(q_p, q_c, axis)
                if q_meas < lo or q_meas > hi:
                    c_eff = kd + dt * ke
                    aug = dt * c_eff * wp.outer(axis_w, axis_w)
                    wp.atomic_add(body_inertia_augment, id_c, aug)

        # Angular DOF 1
        if n_angular >= 2:
            dof_idx = qd_start + n_linear + 1
            lo = joint_limit_lower[dof_idx]
            hi = joint_limit_upper[dof_idx]
            ke = joint_limit_ke[dof_idx]
            kd = joint_limit_kd[dof_idx]
            if ke > 0.0 and lo < hi:
                axis = joint_axis[dof_idx]
                axis_w = wp.quat_rotate(q_p, axis)
                q_meas = compute_revolute_angle(q_p, q_c, axis)
                if q_meas < lo or q_meas > hi:
                    c_eff = kd + dt * ke
                    aug = dt * c_eff * wp.outer(axis_w, axis_w)
                    wp.atomic_add(body_inertia_augment, id_c, aug)

        # Angular DOF 2
        if n_angular >= 3:
            dof_idx = qd_start + n_linear + 2
            lo = joint_limit_lower[dof_idx]
            hi = joint_limit_upper[dof_idx]
            ke = joint_limit_ke[dof_idx]
            kd = joint_limit_kd[dof_idx]
            if ke > 0.0 and lo < hi:
                axis = joint_axis[dof_idx]
                axis_w = wp.quat_rotate(q_p, axis)
                q_meas = compute_revolute_angle(q_p, q_c, axis)
                if q_meas < lo or q_meas > hi:
                    c_eff = kd + dt * ke
                    aug = dt * c_eff * wp.outer(axis_w, axis_w)
                    wp.atomic_add(body_inertia_augment, id_c, aug)
