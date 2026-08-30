from __future__ import print_function

import argparse
import json
import math
import os

import ovito
from ovito.io import import_file
from ovito.vis import ParticleDisplay, RenderSettings, TachyonRenderer, Viewport


def _camera_dir_from_angles(elev_deg, azim_deg):
    elev = math.radians(float(elev_deg))
    azim = math.radians(float(azim_deg))
    return (
        -math.cos(elev) * math.cos(azim),
        -math.cos(elev) * math.sin(azim),
        -math.sin(elev),
    )


def _clear_scene():
    for node in list(ovito.dataset.scene_nodes):
        try:
            node.remove_from_scene()
        except Exception:
            pass


def _apply_type_styles(node, styles):
    if not styles:
        return
    try:
        pprop = node.source.particle_properties.particle_type
    except Exception:
        return

    style_map = {}
    for item in styles:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        style_map[name] = item

    for particle_type in pprop.type_list:
        item = style_map.get(str(particle_type.name))
        if item is None:
            continue
        color = item.get("color")
        radius = item.get("radius")
        if color is not None and len(color) == 3:
            particle_type.color = (float(color[0]), float(color[1]), float(color[2]))
        if radius is not None:
            particle_type.radius = float(radius)


def _render_job(job, config):
    input_path = str(job["input_path"])
    output_path = str(job["output_path"])
    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    _clear_scene()
    node = import_file(input_path)
    if node not in ovito.dataset.scene_nodes:
        node.add_to_scene()

    try:
        node.source.particle_properties.position.display.shape = ParticleDisplay.Shape.Sphere
    except Exception:
        pass
    try:
        node.source.cell.display.render_cell = bool(config.get("show_cell", True))
        if bool(config.get("show_cell", True)):
            node.source.cell.display.line_width = float(job.get("cell_line_width", 0.18))
            cell_color = config.get("cell_color_rgb", [0.42, 0.49, 0.57])
            if len(cell_color) == 3:
                node.source.cell.display.rendering_color = (
                    float(cell_color[0]),
                    float(cell_color[1]),
                    float(cell_color[2]),
                )
    except Exception:
        pass

    _apply_type_styles(node, job.get("type_styles", []))
    node.compute()

    camera_cfg = dict(config.get("camera", {}))
    vp = Viewport()
    vp.type = Viewport.Type.PERSPECTIVE
    vp.camera_dir = _camera_dir_from_angles(camera_cfg.get("elev_deg", 10.0), camera_cfg.get("azim_deg", -78.0))
    vp.fov = math.radians(float(camera_cfg.get("fov_deg", 32.0)))
    vp.zoom_all()

    bg = config.get("background_rgb", [1.0, 1.0, 1.0])
    size_px = int(config.get("size_px", 640))
    rs = RenderSettings(
        size=(size_px, size_px),
        filename=output_path,
        background_color=(float(bg[0]), float(bg[1]), float(bg[2])),
        generate_alpha=bool(config.get("generate_alpha", True)),
    )

    renderer_cfg = dict(config.get("renderer", {}))
    rs.renderer = TachyonRenderer()
    rs.renderer.ambient_occlusion = bool(renderer_cfg.get("ambient_occlusion", True))
    rs.renderer.ambient_occlusion_samples = int(renderer_cfg.get("ambient_occlusion_samples", 64))
    rs.renderer.ambient_occlusion_brightness = float(renderer_cfg.get("ambient_occlusion_brightness", 0.8))
    rs.renderer.antialiasing = bool(renderer_cfg.get("antialiasing", True))
    rs.renderer.antialiasing_samples = int(renderer_cfg.get("antialiasing_samples", 64))
    rs.renderer.direct_light = bool(renderer_cfg.get("direct_light", True))
    rs.renderer.direct_light_intensity = float(renderer_cfg.get("direct_light_intensity", 0.9))
    rs.renderer.shadows = bool(renderer_cfg.get("shadows", True))
    rs.renderer.depth_of_field = bool(renderer_cfg.get("depth_of_field", False))
    if rs.renderer.depth_of_field:
        rs.renderer.focal_length = float(renderer_cfg.get("focal_length", 40.0))
        rs.renderer.aperture = float(renderer_cfg.get("aperture", 0.01))

    vp.render(rs)
    print(output_path)


def main():
    parser = argparse.ArgumentParser(description="Internal helper for OVITO Tachyon panel rendering.")
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    with open(args.manifest, "r") as handle:
        config = json.load(handle)

    jobs = list(config.get("jobs", []))
    if not jobs:
        raise RuntimeError("Manifest does not contain any render jobs.")

    for job in jobs:
        _render_job(job, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
