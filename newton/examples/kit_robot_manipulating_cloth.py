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

import numpy as np
import warp as wp
from example_robot_manipulating_cloth import ExampleClothManipulation as Example
from omni.kit_app import KitApp
import newton
from typing import Any

@wp.kernel(enable_backward=False)
def deform_mesh_kernel(
    src_points: wp.array(dtype=wp.vec3),
    dst_points: Any,
):
    tid = wp.tid()
    dst_points[0, tid] = src_points[tid]

@wp.kernel
def _update_body_transforms(
    src: wp.array(dtype=wp.transform),
    dest_pos: wp.fabricarray(dtype=wp.vec3d),
    dest_rot: wp.fabricarray(dtype=wp.quatf),
    ordering: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    xform = src[ordering[i]]
    dest_pos[i] = wp.vec3d(wp.transform_get_translation(xform))
    dest_rot[i] = wp.quatf(wp.transform_get_rotation(xform))

def _render_bodies(usdrt_stage, state, fabric_ordering):
    import usdrt
    # Update bodies
    body_selection = usdrt_stage.SelectPrims(
        require_attrs=[
            (usdrt.Sdf.ValueTypeNames.Double3, "_worldPosition", usdrt.Usd.Access.Overwrite),
            (usdrt.Sdf.ValueTypeNames.Quatf, "_worldOrientation", usdrt.Usd.Access.Overwrite),
        ],
        device="cuda:0",
    )
    fpos = wp.fabricarray(data=body_selection, attrib="_worldPosition")
    frot = wp.fabricarray(data=body_selection, attrib="_worldOrientation")

    wp.launch(
        _update_body_transforms,
        dim=state.body_q.shape[0],
        inputs=[state.body_q, fpos, frot, fabric_ordering],
    )


def _render_cloth(usdrt_stage, state):
    import usdrt
    cloth_selection = usdrt_stage.SelectPrims(
        require_prim_type="Mesh",
        require_applied_schemas=["Deformable_mesh"],
        require_attrs=[
            (usdrt.Sdf.ValueTypeNames.Point3fArray, "points", usdrt.Usd.Access.Overwrite),
        ],
        device="cuda:0",
    )
    fpoints = wp.fabricarray(data=cloth_selection, attrib="points")
    wp.launch(
        deform_mesh_kernel,
        dim=state.particle_q.shape[0],
        inputs=[state.particle_q],
        outputs=[fpoints],
        device="cuda:0",
    )


def _create_cloth(usdrt_stage, usd_stage, model, usd_path, mesh_points, mesh_indices, vertexCounts):
    import usdrt
    from pxr import UsdGeom, Usd, Vt, Sdf, UsdShade
    # Create FSD-compatible UsdGeom.Mesh
    # if not usd_stage.ResolveIdentifierToEditTarget(usd_path):
    #     raise FileNotFoundError(f"USD file not found at path: '{usd_path}'.")
    # prim = UsdGeom.Xform.Define(usd_stage, Sdf.Path("/refCloth")).GetPrim()
    # # prim = usd_stage.DefinePrim("/refCloth", "Xform")
    # prim.GetReferences().AddReference(assetPath=usd_path, primPath="/root/shirt")
    # app.update()
    # mesh = UsdGeom.Mesh.Get(usd_stage, "/refCloth")
    # mesh_prim = usd_stage.GetPrimAtPath("/refCloth")
    # mesh_points = np.array(mesh.GetPointsAttr().Get())
    # mesh_indices = np.array(mesh.GetFaceVertexIndicesAttr().Get())
    # print(f"Mesh points: {mesh_points.shape}, indices: {mesh_indices.shape}")

    usd_geom = UsdGeom.Mesh.Define(usd_stage, "/cloth")
    vtarr_points = usdrt.Vt.Vec3fArray(mesh_points.reshape(-1, 3).tolist())
    vtarr_indices = usdrt.Vt.IntArray(mesh_indices.tolist())
    usd_geom.CreatePointsAttr(vtarr_points)
    usd_geom.CreateFaceVertexIndicesAttr(vtarr_indices)
    usd_geom.CreateFaceVertexCountsAttr(vertexCounts)
    app.update()
    
    # apply material
    mat_binding_api = UsdShade.MaterialBindingAPI.Apply(usd_geom.GetPrim())
    mtl = UsdShade.Material.Get(usd_stage, "/World/Looks/OmniSurface")
    mat_binding_api.Bind(mtl)

    usd_geom = usdrt_stage.DefinePrim("/cloth", "Mesh")
    usd_geom.CreateAttribute(
                "Deformable_mesh", usdrt.Sdf.ValueTypeNames.AppliedSchemaTypeTag, True)
    
def _create_xform_attrs(usdrt_stage, usd_stage, usd_render, app):
    import usdrt
    from pxr import UsdShade, Sdf

    for k, name in enumerate(usd_render.body_names):
        path = usd_render.root.GetPath().AppendChild(name)
        prim = usdrt_stage.GetPrimAtPath(str(path))

        # apply material
        usd_prim = usd_stage.GetPrimAtPath(str(path))
        mat_binding_api = UsdShade.MaterialBindingAPI.Apply(usd_prim)
        mtl = UsdShade.Material.Get(usd_stage, "/World/Looks/OmniPBR_02")
        mat_binding_api.Bind(mtl)

        prim.CreateAttribute("_worldPosition", usdrt.Sdf.ValueTypeNames.Double3, True).Set([k, k, k])
        prim.CreateAttribute("_worldOrientation", usdrt.Sdf.ValueTypeNames.Quatf, True).Set(usdrt.Gf.Quatf(1, 0, 0, 0))
        prim.CreateAttribute("_worldScale", usdrt.Sdf.ValueTypeNames.Float3, True).Set(usdrt.Gf.Vec3f(1, 1, 1))

    # Read back the fake transforms that we wrote above and use that to determine selection ordering
    # Why do we need to update here? no idea
    app.update()
    body_selection = usdrt_stage.SelectPrims(
        require_attrs=[
            (usdrt.Sdf.ValueTypeNames.Double3, "_worldPosition", usdrt.Usd.Access.ReadWrite),
            (usdrt.Sdf.ValueTypeNames.Quatf, "_worldOrientation", usdrt.Usd.Access.ReadWrite),
        ],
        device="cuda:0",
    )

    fpos = wp.fabricarray(data=body_selection, attrib="_worldPosition")
    fabric_ordering = wp.array(fpos.numpy()[:, 0], dtype=wp.int32, device="cuda:0")

    return fabric_ordering


def _create_stage(usd_context, options):
    import usdrt

    from omni.usd import Usd, UsdGeom
    usd_stage = Usd.Stage.Open(newton.examples.get_asset("unisex_shirt.usd"))
    mesh = UsdGeom.Mesh(usd_stage.GetPrimAtPath("/root/shirt"))
    mesh_points = np.array(mesh.GetPointsAttr().Get())
    mesh_indices = np.array(mesh.GetFaceVertexIndicesAttr().Get())
    vertexCounts= mesh.GetFaceVertexCountsAttr().Get()

    usd_context.open_stage(newton.examples.get_asset("scene_.usda"))
    # usd_stage = Usd.Stage.Open(newton.examples.get_asset("scene.usda"))
    # usd_context.new_stage()
    usd_stage = usd_context.get_stage()
    app.update()

    with wp.ScopedDevice(options.device):
        example = Example()

        # NOTE that changing the kit up axis seems problematic
        usd_render = newton.utils.SimRendererUsd(example.model, path=usd_stage, up_axis="Y")
        # usd_render.render_ground(size=5, plane=example.model.ground_plane_params + np.array([0.0, 0.0, 0.0, -0.0125]))

    stage_id = usd_context.get_stage_id()
    usdrt_stage = usdrt.Usd.Stage.Attach(stage_id)
    _create_cloth(usdrt_stage, usd_stage, example.model, newton.examples.get_asset("unisex_shirt.usd"), mesh_points, mesh_indices, vertexCounts)
    app.update()
    # _render_cloth(usdrt_stage, example.state_0)
    fabric_ordering = _create_xform_attrs(usdrt_stage, usd_stage, usd_render, app)
    _render_bodies(usdrt_stage, example.state_0, fabric_ordering)

    app.update()

    return example, usdrt_stage, usd_render, fabric_ordering, 



def _run(app, options):
    import omni.timeline
    import omni.usd
    usd_context = omni.usd.get_context()
    timeline = omni.timeline.get_timeline_interface()
    needs_stage_reset = True

    while app.is_running():
        if timeline.is_stopped() and needs_stage_reset:
            example, usdrt_stage, usd_render, fabric_ordering = _create_stage(usd_context, options)
            example.advance_frame()
            _render_bodies(usdrt_stage, example.state_0, fabric_ordering)
            _render_cloth(usdrt_stage, example.state_0)
            needs_stage_reset = False
        elif timeline.is_playing():
            with wp.ScopedDevice(options.device):
                with wp.ScopedTimer("step", synchronize=True):
                    example.advance_frame()

                _render_bodies(usdrt_stage, example.state_0, fabric_ordering)
                _render_cloth(usdrt_stage, example.state_0)

                needs_stage_reset = True

        app.update()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument("--usd_stage_path", type=str, default=None, help="Path to the USD stage to open.")
    options, kit_args = parser.parse_known_args()

    # KitApp is a wrapper over the omni.kit.app.IApp interface
    app = KitApp()

    # Startup kit and ask for the omni.ui extension. Kit will start it including all its dependencies
    # Add any extra command line arguments to the startup call, allowing the user to pass more to the script
    # fmt: off
    app.startup([
            "--enable", "omni.usd", 
            "--enable", "omni.usd.libs", 
            "--enable", "omni.kit.uiapp", 

            "--enable", "omni.kit.window.file",
            "--enable", "omni.kit.menu.utils",
            "--enable", "omni.kit.menu.file",
            "--enable", "omni.kit.menu.edit",
            "--enable", "omni.kit.menu.create",
            "--enable", "omni.kit.menu.common",
            "--enable", "omni.kit.context_menu",
            "--enable", "omni.kit.selection",
             "--enable", "omni.kit.window.stage", 
            "--enable", "omni.kit.window.property",
             "--enable", "omni.kit.viewport.bundle", 
             "--enable", "omni.kit.viewport.rtx",
             "--enable", "omni.hydra.usdrt_delegate",
             "--enable", "usdrt.scenegraph",
            "--enable", "omni.kit.window.status_bar",
            "--enable", "omni.stats",
            "--enable", "omni.rtx.settings.core",

            "--enable", "omni.kit.window.stats",
            "--enable", "omni.kit.window.script_editor",
            "--enable", "omni.kit.window.console",
            "--enable", "omni.kit.window.preferences",
            "--enable", "omni.kit.widget.viewport",
            
            "--enable", "omni.kit.window.toolbar",
            "--enable", "omni.timeline",

             "--/app/useFabricSceneDelegate=1",
             "--/renderer/multiGpu/autoEnable=false",
             ] + kit_args)

    # fmt: on

    _run(app, options)