# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Penalty-based contact force kernels for stable resting contacts."""

import warp as wp


@wp.kernel
def apply_penalty_contact_forces(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    shape_body: wp.array(dtype=int),
    contact_count: wp.array(dtype=int),
    contact_point0: wp.array(dtype=wp.vec3),
    contact_point1: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    contact_shape0: wp.array(dtype=int),
    contact_shape1: wp.array(dtype=int),
    contact_margin0: wp.array(dtype=float),
    contact_margin1: wp.array(dtype=float),
    ke: float,  # normal stiffness
    kd: float,  # normal damping
    kf: float,  # friction stiffness
    mu: float,  # friction coefficient
    contact_max: int,
    # outputs
    body_f: wp.array(dtype=wp.spatial_vector),
):
    """Apply penalty-based contact forces.
    
    Uses spring-damper model for normal forces and velocity-based friction.
    More stable for resting contacts than constraint-based approaches.
    """
    tid = wp.tid()
    
    count = wp.min(contact_max, contact_count[0])
    if tid >= count:
        return
    
    s0 = contact_shape0[tid]
    s1 = contact_shape1[tid]
    b0 = shape_body[s0]
    b1 = shape_body[s1]
    
    p0 = contact_point0[tid]
    p1 = contact_point1[tid]
    n = contact_normal[tid]
    
    # Penetration depth accounting for margins (negative when penetrating)
    d_raw = wp.dot(p1 - p0, n)
    m0 = contact_margin0[tid]
    m1 = contact_margin1[tid]
    d = d_raw - m0 - m1  # Actual gap (negative = penetration)
    
    if d > 0.0:
        return  # No contact
    
    # Get velocities at contact point
    v0 = wp.vec3(0.0)
    v1 = wp.vec3(0.0)
    
    if b0 >= 0:
        q0 = body_q[b0]
        qd0 = body_qd[b0]
        com0 = body_com[b0]
        pos0 = wp.transform_get_translation(q0)
        r0 = p0 - pos0 - com0
        v0 = wp.spatial_top(qd0) + wp.cross(wp.spatial_bottom(qd0), r0)
    
    if b1 >= 0:
        q1 = body_q[b1]
        qd1 = body_qd[b1]
        com1 = body_com[b1]
        pos1 = wp.transform_get_translation(q1)
        r1 = p1 - pos1 - com1
        v1 = wp.spatial_top(qd1) + wp.cross(wp.spatial_bottom(qd1), r1)
    
    # Relative velocity
    v_rel = v1 - v0
    vn = wp.dot(v_rel, n)
    vt = v_rel - vn * n
    
    # Normal force (spring + damper)
    fn = -ke * d - kd * vn
    if fn < 0.0:
        fn = 0.0
    
    # Friction force (clamped to friction cone)
    vt_mag = wp.length(vt)
    if vt_mag > 1e-6:
        ff_max = mu * fn
        ff_mag = wp.min(kf * vt_mag, ff_max)
        ff = -vt * (ff_mag / vt_mag)
    else:
        ff = wp.vec3(0.0)
    
    # Total contact force
    f_contact = fn * n + ff
    
    # Apply to bodies
    if b0 >= 0:
        q0 = body_q[b0]
        com0 = body_com[b0]
        pos0 = wp.transform_get_translation(q0)
        r0 = p0 - pos0 - com0
        tau0 = wp.cross(r0, -f_contact)
        wp.atomic_add(body_f, b0, wp.spatial_vector(-f_contact, tau0))
    
    if b1 >= 0:
        q1 = body_q[b1]
        com1 = body_com[b1]
        pos1 = wp.transform_get_translation(q1)
        r1 = p1 - pos1 - com1
        tau1 = wp.cross(r1, f_contact)
        wp.atomic_add(body_f, b1, wp.spatial_vector(f_contact, tau1))
