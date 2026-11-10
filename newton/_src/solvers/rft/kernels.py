# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Surface integration of the Agarwal et al. (2023) granular resistance law."""

import warp as wp

from ...geometry import GeoType

_BOX = wp.constant(int(GeoType.BOX))
_SPHERE = wp.constant(int(GeoType.SPHERE))
_CAPSULE = wp.constant(int(GeoType.CAPSULE))
_CYLINDER = wp.constant(int(GeoType.CYLINDER))
_ELLIPSOID = wp.constant(int(GeoType.ELLIPSOID))

_Terms = wp.types.vector(length=20, dtype=wp.float32)
_Polygon = wp.types.matrix(shape=(4, 3), dtype=wp.float32)

# Factual fit coefficients from arXiv:2205.14920v3, SI Table S3, in its monomial order.
# https://arxiv.org/src/2205.14920 (anc/SI-Mechanistic_framework_...pdf)
_C1 = wp.constant(
    _Terms(
        0.00212,
        -0.02320,
        -0.20890,
        -0.43083,
        -0.00259,
        0.48872,
        -0.00415,
        0.07204,
        -0.02750,
        -0.08772,
        0.01992,
        -0.45961,
        0.40799,
        -0.10107,
        -0.06576,
        0.05664,
        -0.09269,
        0.01892,
        0.01033,
        0.15120,
    )
)
_C2 = wp.constant(
    _Terms(
        -0.06796,
        -0.10941,
        0.04725,
        -0.06914,
        -0.05835,
        -0.65880,
        -0.11985,
        -0.25739,
        -0.26834,
        0.02692,
        -0.00736,
        0.63758,
        0.08997,
        0.21069,
        0.04748,
        0.20406,
        0.18519,
        0.04934,
        0.13527,
        -0.33207,
    )
)
_C3 = wp.constant(
    _Terms(
        -0.02634,
        -0.03436,
        0.45256,
        0.00835,
        0.02553,
        -1.31290,
        -0.05532,
        0.06790,
        -0.16404,
        0.02287,
        0.02927,
        0.95406,
        -0.00131,
        -0.11028,
        0.01487,
        -0.02730,
        0.10911,
        -0.04097,
        0.07881,
        -0.27519,
    )
)


@wp.func
def reference_resistance(normal: wp.vec3, direction: wp.vec3, gravity_direction: wp.vec3):
    # SI S7 parameterizes the *unoriented plate* by its downward normal.
    # The outward body normal is retained separately for activation and compression.
    plate_normal = normal
    # Resolve the horizontal-plate-normal limit consistently across rotated float32 frames.
    if wp.dot(normal, gravity_direction) <= 1.0e-6:
        plate_normal = -normal
    p1 = wp.dot(gravity_direction, direction)
    p2 = wp.dot(gravity_direction, plate_normal)
    p3 = wp.dot(plate_normal, direction)
    terms = _Terms(
        1.0,
        p1,
        p2,
        p3,
        p1 * p1,
        p2 * p2,
        p3 * p3,
        p1 * p2,
        p2 * p3,
        p3 * p1,
        p1 * p1 * p1,
        p2 * p2 * p2,
        p3 * p3 * p3,
        p1 * p2 * p2,
        p2 * p1 * p1,
        p2 * p3 * p3,
        p3 * p2 * p2,
        p3 * p1 * p1,
        p1 * p3 * p3,
        p1 * p2 * p3,
    )
    return wp.dot(_C1, terms) * plate_normal + wp.dot(_C2, terms) * direction + wp.dot(_C3, terms) * gravity_direction


@wp.func
def traction(
    normal: wp.vec3,
    velocity: wp.vec3,
    gravity_direction: wp.vec3,
    depth: float,
    strength: float,
    density: float,
    surface_friction: float,
    inertial_coefficient: float,
    velocity_regularization: float,
):
    speed = wp.length(velocity)
    result = wp.vec3(0.0)
    if depth > 0.0 and speed > 0.0 and wp.dot(normal, velocity) >= 0.0:
        direction = velocity / speed
        q = reference_resistance(normal, direction, gravity_direction)
        pressure = wp.max(0.0, -wp.dot(q, normal))
        tangent = q - wp.dot(q, normal) * normal
        tangent_length = wp.length(tangent)
        factor = float(1.0)
        if tangent_length > 0.0:
            factor = wp.min(1.0, surface_friction * pressure / tangent_length)
        amplitude = float(1.0)
        if velocity_regularization > 0.0:
            amplitude = speed / wp.sqrt(speed * speed + velocity_regularization * velocity_regularization)
        result = depth * strength * amplitude * (-pressure * normal + factor * tangent)
        normal_speed = wp.max(0.0, wp.dot(normal, velocity))
        result -= inertial_coefficient * density * normal_speed * normal_speed * normal
    return result


@wp.func
def inside_primitive(point: wp.vec3, kind: int, scale: wp.vec3):
    inside = False
    if kind == _BOX:
        inside = (
            wp.max(wp.max(wp.abs(point[0]) - scale[0], wp.abs(point[1]) - scale[1]), wp.abs(point[2]) - scale[2])
            < -1.0e-6
        )
    elif kind == _SPHERE:
        inside = wp.length(point) < scale[0] - 1.0e-6
    elif kind == _CYLINDER:
        inside = (
            wp.sqrt(point[0] * point[0] + point[1] * point[1]) < scale[0] - 1.0e-6
            and wp.abs(point[2]) < scale[1] - 1.0e-6
        )
    elif kind == _CAPSULE:
        radial = wp.vec3(point[0], point[1], wp.max(wp.abs(point[2]) - scale[1], 0.0))
        inside = wp.length(radial) < scale[0] - 1.0e-6
    elif kind == _ELLIPSOID:
        inside = wp.length(wp.cw_div(point, scale)) < 1.0 - 1.0e-5
    return inside


@wp.kernel
def integrate_surfaces(
    triangle_body: wp.array[int],
    triangle_vertices: wp.array2d[wp.vec3],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    body_world: wp.array[int],
    gravity: wp.array[wp.vec3],
    surface_height: wp.array[float],
    surface_lowering: wp.array[float],
    surface_velocity: wp.vec3,
    material: wp.array2d[float],
    triangle_shape: wp.array[int],
    body_shape_start: wp.array[int],
    body_shapes: wp.array[int],
    shape_type: wp.array[int],
    shape_scale: wp.array[wp.vec3],
    shape_transform: wp.array[wp.transform],
    triangle_f: wp.array[wp.spatial_vector],
    body_f: wp.array[wp.spatial_vector],
):
    tid = wp.tid()
    body = triangle_body[tid]
    world = body_world[body]
    density = material[world, 0]
    internal_friction = material[world, 1]
    surface_friction = material[world, 2]
    strength_override = material[world, 3]
    inertial_coefficient = material[world, 4]
    velocity_regularization = material[world, 5]
    bearing = material[world, 6]
    g = gravity[world]
    g_magnitude = wp.length(g)
    wrench = wp.spatial_vector()
    if g_magnitude > 0.0:
        down = g / g_magnitude
        up = -down
        height = surface_height[world] - surface_lowering[tid]
        pose = body_q[body]
        center = wp.transform_point(pose, body_com[body])
        twist = body_qd[body]
        a = wp.transform_point(pose, triangle_vertices[tid, 0])
        b = wp.transform_point(pose, triangle_vertices[tid, 1])
        c = wp.transform_point(pose, triangle_vertices[tid, 2])
        cross = wp.cross(b - a, c - a)
        area_twice = wp.length(cross)
        if area_twice > 0.0:
            normal = cross / area_twice
            vertices = wp.mat33(a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2])
            clipped = _Polygon()
            count = int(0)
            # Clip the triangle to the occupied half-space before quadrature.
            for i in range(3):
                previous = vertices[(i + 2) % 3]
                current = vertices[i]
                previous_depth = height - wp.dot(up, previous)
                current_depth = height - wp.dot(up, current)
                previous_inside = previous_depth > 0.0
                current_inside = current_depth > 0.0
                if previous_inside != current_inside:
                    t = previous_depth / (previous_depth - current_depth)
                    clipped[count] = previous + t * (current - previous)
                    count += 1
                if current_inside:
                    clipped[count] = current
                    count += 1
            strength = strength_override
            if strength < 0.0:
                mu = internal_friction
                strength = density * g_magnitude * (894.0 * mu * mu * mu - 386.0 * mu * mu + 89.0 * mu)
            for i in range(1, count - 1):
                p0 = clipped[0]
                p1 = clipped[i]
                p2 = clipped[i + 1]
                weight = wp.length(wp.cross(p1 - p0, p2 - p0)) / 6.0
                # Symmetric degree-two quadrature integrates depth-weighted moments exactly
                # when the plate orientation and translation direction are uniform.
                for j in range(3):
                    point = (p0 + p1 + p2) / 6.0
                    if j == 0:
                        point += p0 / 2.0
                    elif j == 1:
                        point += p1 / 2.0
                    else:
                        point += p2 / 2.0
                    # Remove patches inside another primitive of the same body.
                    # Quadrature rejection approximates intersection clipping;
                    # refine geometry at overlaps to check convergence.
                    covered = bool(False)
                    own_shape = triangle_shape[tid]
                    if own_shape >= 0:
                        local = wp.transform_point(wp.transform_inverse(pose), point)
                        for k in range(body_shape_start[body], body_shape_start[body + 1]):
                            other = body_shapes[k]
                            if other != own_shape:
                                other_point = wp.transform_point(wp.transform_inverse(shape_transform[other]), local)
                                if inside_primitive(other_point, shape_type[other], shape_scale[other]):
                                    covered = True
                    lever = point - center
                    velocity = wp.spatial_top(twist) + wp.cross(wp.spatial_bottom(twist), lever) - surface_velocity
                    depth = height - wp.dot(up, point)
                    stress = traction(
                        normal,
                        velocity,
                        down,
                        depth,
                        strength,
                        density,
                        surface_friction,
                        inertial_coefficient,
                        velocity_regularization,
                    )
                    # Experimental near-rest normal-bearing closure. This is not
                    # part of the published flowing RFT fit and supplies no exact
                    # tangential sticking state. Blend, rather than double-count,
                    # its pressure with the flowing law.
                    if bearing > 0.0:
                        speed2 = wp.dot(velocity, velocity)
                        eps = wp.max(velocity_regularization, 0.001)
                        blend = eps * eps / (speed2 + eps * eps)
                        pressure = bearing * strength * depth * wp.max(wp.dot(normal, down), 0.0)
                        tangent_velocity = velocity - wp.dot(normal, velocity) * normal
                        tangent_speed2 = wp.dot(tangent_velocity, tangent_velocity)
                        resting = -pressure * normal - surface_friction * pressure * tangent_velocity / wp.sqrt(
                            tangent_speed2 + eps * eps
                        )
                        stress = (1.0 - blend) * stress + blend * resting
                    force = weight * stress
                    if covered:
                        force = wp.vec3(0.0)
                    wrench += wp.spatial_vector(force, wp.cross(lever, force))
    triangle_f[tid] = wrench
    wp.atomic_add(body_f, body, wrench)


@wp.kernel
def add_forces(source: wp.array[wp.spatial_vector], destination: wp.array[wp.spatial_vector]):
    tid = wp.tid()
    destination[tid] += source[tid]


@wp.kernel
def accumulate_impulse(force: wp.array[wp.spatial_vector], dt: float, impulse: wp.array2d[wp.float64]):
    body, component = wp.tid()
    impulse[body, component] += wp.float64(dt) * wp.float64(force[body][component])
