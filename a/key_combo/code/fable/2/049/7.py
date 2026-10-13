from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import bbox_coords, find_components, get_bg, paste, random_free_location


BG = 0
COLORS = tuple(range(1, 10))
CANVAS_SIZE = 30
MAX_GENERATION_ATTEMPTS = 400
INNER_WIDTH_CHOICES = (2, 3, 4, 5, 6)
INTERIOR_HEIGHT_RANGE = (3, 7)
OBJECT_COUNT_RANGE = (2, 4)
MAX_RECT_SIDE = 12


def _component_patch(comp: dict) -> np.ndarray:
    coords = np.asarray(comp["coords"], dtype=int)
    r0, c0, r1, c1 = map(int, comp["bbox"])
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    patch[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return patch


def _is_solid_rectangle(comp: dict) -> bool:
    patch = _component_patch(comp)
    return bool(patch.size > 0 and patch.all())


def _is_u_frame(comp: dict) -> bool:
    patch = _component_patch(comp)
    h, w = patch.shape
    if h < 2 or w < 3:
        return False
    expected = np.zeros_like(patch, dtype=bool)
    expected[:, 0] = True
    expected[:, -1] = True
    expected[-1, :] = True
    return bool(np.array_equal(patch, expected))


def _classify_objects(grid: np.ndarray) -> list[dict]:
    bg = get_bg(grid)
    comps = find_components(grid, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    assert comps, "expected at least one colored object"

    by_color: dict[int, list[dict]] = {}
    for comp in comps:
        by_color.setdefault(int(comp["color"]), []).append(comp)

    objects: list[dict] = []
    inner_widths: list[int] = []
    for color, color_comps in sorted(by_color.items()):
        assert len(color_comps) == 2, "each color must appear as exactly one frame and one rectangle"
        frames = [comp for comp in color_comps if _is_u_frame(comp)]
        rects = [comp for comp in color_comps if _is_solid_rectangle(comp)]
        assert len(frames) == 1 and len(rects) == 1, "each color must contain one U-frame and one solid rectangle"

        frame = frames[0]
        rect = rects[0]
        fr0, fc0, fr1, fc1 = map(int, frame["bbox"])
        inner_w = int(fc1 - fc0 - 1)
        interior_h = int(fr1 - fr0)
        assert inner_w >= 1 and interior_h >= 1

        area = int(rect["area"])
        assert area % inner_w == 0, "rectangle area must divide evenly by the frame interior width"
        fill_h = area // inner_w
        assert 1 <= fill_h <= interior_h, "liquid height must fit inside the frame interior"

        inner_widths.append(inner_w)
        objects.append(
            {
                "color": int(color),
                "frame_bbox": (fr0, fc0, fr1, fc1),
                "rect_bbox": tuple(map(int, rect["bbox"])),
                "area": area,
                "inner_w": inner_w,
                "interior_h": interior_h,
                "fill_h": fill_h,
                "rect_h": int(rect["bbox"][2] - rect["bbox"][0] + 1),
                "rect_w": int(rect["bbox"][3] - rect["bbox"][1] + 1),
            }
        )

    assert len(inner_widths) == len(set(inner_widths)), "frame interior widths must be distinct"
    return objects


def _apply_rule(grid: np.ndarray, objects: list[dict]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for obj in objects:
        rr0, cc0, rr1, cc1 = obj["rect_bbox"]
        out[rr0 : rr1 + 1, cc0 : cc1 + 1] = BG

        fr0, fc0, fr1, fc1 = obj["frame_bbox"]
        fill_h = int(obj["fill_h"])
        out[fr1 - fill_h : fr1, fc0 + 1 : fc1] = int(obj["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pour each solid color block into the same-colored U-frame and recompute the liquid level by area conservation.

    Input structure:
    - The background is black.
    - Several colors each appear twice: once as an upward-open 1-cell-thick U-frame and once as a detached solid rectangle.
    - The U-frames have distinct interior widths, and each rectangle's area is divisible by its frame's interior width.

    Rule:
    - Match every solid rectangle to the unique U-frame of the same color.
    - Erase the rectangle.
    - Fill the interior of that frame from the bottom upward with the same color, using exactly as many cells as the rectangle had.
    - The new liquid height is rectangle_area / frame_interior_width.

    Key insight:
    - The conserved quantity is area, not the rectangle's original width or height, so the liquid must be redistributed across the frame's own interior width.
    """
    g = np.asarray(grid, dtype=int)
    objects = _classify_objects(g)
    return _apply_rule(g, objects)


def _make_frame_patch(color: int, interior_h: int, inner_w: int) -> np.ndarray:
    patch = np.zeros((interior_h + 1, inner_w + 2), dtype=int)
    patch[:, 0] = color
    patch[:, -1] = color
    patch[-1, :] = color
    return patch


def _factor_pairs(area: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for h in range(1, area + 1):
        if area % h != 0:
            continue
        w = area // h
        if max(h, w) <= MAX_RECT_SIDE:
            pairs.append((int(h), int(w)))
    return pairs


def _sample_rectangle_shape(inner_w: int, interior_h: int, mode: str) -> tuple[int, int, int] | None:
    fill_heights = list(range(1, interior_h))
    random.shuffle(fill_heights)

    all_candidates: list[tuple[int, int, int]] = []
    nice_candidates: list[tuple[int, int, int]] = []
    for fill_h in fill_heights:
        area = inner_w * fill_h
        for rect_h, rect_w in _factor_pairs(area):
            if rect_w == inner_w or rect_w == fill_h:
                continue
            if mode == "wide" and rect_w <= inner_w:
                continue
            if mode == "narrow" and rect_w >= inner_w:
                continue
            cand = (int(fill_h), int(rect_h), int(rect_w))
            all_candidates.append(cand)
            if min(rect_h, rect_w) >= 2:
                nice_candidates.append(cand)

    if nice_candidates:
        return random.choice(nice_candidates)
    if all_candidates:
        return random.choice(all_candidates)
    return None


def _make_pair_spec(color: int, inner_w: int, mode: str) -> dict | None:
    heights = list(range(INTERIOR_HEIGHT_RANGE[0], INTERIOR_HEIGHT_RANGE[1] + 1))
    random.shuffle(heights)
    for interior_h in heights:
        sampled = _sample_rectangle_shape(inner_w, interior_h, mode)
        if sampled is None:
            continue
        fill_h, rect_h, rect_w = sampled
        return {
            "color": int(color),
            "inner_w": int(inner_w),
            "interior_h": int(interior_h),
            "fill_h": int(fill_h),
            "rect_h": int(rect_h),
            "rect_w": int(rect_w),
            "mode": mode,
            "frame_patch": _make_frame_patch(int(color), int(interior_h), int(inner_w)),
            "rect_patch": np.full((int(rect_h), int(rect_w)), int(color), dtype=int),
        }
    return None


def _sample_specs() -> list[dict] | None:
    count = random.randint(OBJECT_COUNT_RANGE[0], OBJECT_COUNT_RANGE[1])
    inner_ws = random.sample(INNER_WIDTH_CHOICES, count)
    colors = random.sample(COLORS, count)

    modes = ["wide", "narrow"]
    while len(modes) < count:
        modes.append(random.choice(("wide", "narrow")))
    random.shuffle(modes)

    specs: list[dict] = []
    for color, inner_w, mode in zip(colors, inner_ws, modes):
        spec = _make_pair_spec(color, inner_w, mode)
        if spec is None:
            return None
        specs.append(spec)

    fill_levels = {int(spec["fill_h"]) for spec in specs}
    if len(fill_levels) < 2:
        return None
    return specs


def _place_scene(specs: list[dict]) -> tuple[np.ndarray, list[dict]] | None:
    canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=int)
    reserved = np.zeros_like(canvas)
    scene = [dict(spec) for spec in specs]

    items: list[tuple[int, str, dict, np.ndarray]] = []
    for spec in scene:
        items.append((spec["frame_patch"].size, "frame", spec, spec["frame_patch"]))
        items.append((spec["rect_patch"].size, "rect", spec, spec["rect_patch"]))
    items.sort(key=lambda item: (item[0], max(item[3].shape)), reverse=True)

    for _size, kind, spec, patch in items:
        footprint = np.ones(patch.shape, dtype=int)
        try:
            r0, c0 = random_free_location(
                reserved,
                footprint,
                transparent=0,
                border=1,
                padding=1,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(reserved, footprint, r0, c0, transparent=0)
        paste(canvas, patch, r0, c0, transparent=0)
        bbox = (int(r0), int(c0), int(r0 + patch.shape[0] - 1), int(c0 + patch.shape[1] - 1))
        spec[f"{kind}_bbox"] = bbox

    return canvas, scene


def _crop_scene(inp: np.ndarray, out: np.ndarray, scene: list[dict]) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    coords = np.argwhere(inp != BG)
    r0, c0, r1, c1 = bbox_coords(coords)

    crop_r0 = random.randint(max(0, r0 - 4), r0 - 1)
    crop_c0 = random.randint(max(0, c0 - 4), c0 - 1)
    crop_r1 = random.randint(r1 + 1, min(inp.shape[0] - 1, r1 + 4))
    crop_c1 = random.randint(c1 + 1, min(inp.shape[1] - 1, c1 + 4))

    cropped_inp = inp[crop_r0 : crop_r1 + 1, crop_c0 : crop_c1 + 1].copy()
    cropped_out = out[crop_r0 : crop_r1 + 1, crop_c0 : crop_c1 + 1].copy()

    shifted_scene: list[dict] = []
    for spec in scene:
        shifted = dict(spec)
        for key in ("frame_bbox", "rect_bbox"):
            rr0, cc0, rr1, cc1 = spec[key]
            shifted[key] = (
                int(rr0 - crop_r0),
                int(cc0 - crop_c0),
                int(rr1 - crop_r0),
                int(cc1 - crop_c0),
            )
        shifted_scene.append(shifted)
    return cropped_inp, cropped_out, shifted_scene


def _height_shortcut_output(inp: np.ndarray, scene: list[dict], use_width: bool) -> np.ndarray:
    alt = inp.copy()
    for spec in scene:
        rr0, cc0, rr1, cc1 = spec["rect_bbox"]
        alt[rr0 : rr1 + 1, cc0 : cc1 + 1] = BG
        fr0, fc0, fr1, fc1 = spec["frame_bbox"]
        wrong_h = int(spec["rect_w"] if use_width else spec["rect_h"])
        wrong_h = max(1, min(int(spec["interior_h"]), wrong_h))
        alt[fr1 - wrong_h : fr1, fc0 + 1 : fc1] = int(spec["color"])
    return alt


def _scene_semantics_ok(inp: np.ndarray, out: np.ndarray, scene: list[dict]) -> bool:
    if np.array_equal(inp, out):
        return False

    if max(inp.shape) > 30 or max(out.shape) > 30:
        return False

    if not any(int(spec["rect_w"]) > int(spec["inner_w"]) for spec in scene):
        return False
    if not any(int(spec["rect_w"]) < int(spec["inner_w"]) for spec in scene):
        return False

    for spec in scene:
        area = int(spec["rect_h"]) * int(spec["rect_w"])
        if area != int(spec["inner_w"]) * int(spec["fill_h"]):
            return False
        if int(spec["fill_h"]) in (int(spec["rect_h"]), int(spec["rect_w"])):
            return False
        if int(spec["fill_h"]) >= int(spec["interior_h"]):
            return False

    if np.array_equal(_height_shortcut_output(inp, scene, use_width=False), out):
        return False
    if np.array_equal(_height_shortcut_output(inp, scene, use_width=True), out):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    specs = _sample_specs()
    if specs is None:
        return None

    placed = _place_scene(specs)
    if placed is None:
        return None
    inp_full, scene_full = placed
    out_full = _apply_rule(inp_full, scene_full)
    inp, out, scene = _crop_scene(inp_full, out_full, scene_full)

    if not _scene_semantics_ok(inp, out, scene):
        return None

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid area-conserved pouring input")
