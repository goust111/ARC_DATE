import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import bbox, components_mask, crop_bbox, find_components, get_bg, paste


BG = 0
PALETTE = tuple(range(1, 10))
GEN_ATTEMPTS = 400

_BASE_SHAPES = [
    ("111",),
    ("11", "10"),
    ("111", "010"),
    ("110", "011"),
    ("101", "111"),
    ("111", "100"),
]


def _mask_from_lines(lines: tuple[str, ...]) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in lines], dtype=bool)


def _normalize_coords(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    shifted = coords - np.array([r0, c0], dtype=int)
    return tuple(sorted((int(r), int(c)) for r, c in shifted))


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    return _normalize_coords(np.argwhere(mask))


def _build_shape_pool() -> list[np.ndarray]:
    seen = set()
    pool = []
    for spec in _BASE_SHAPES:
        base = _mask_from_lines(spec)
        for k in range(4):
            rotated = np.rot90(base, k=k)
            for variant in (rotated, np.fliplr(rotated)):
                if variant.shape[0] > 2:
                    continue
                key = _mask_key(variant)
                if key in seen:
                    continue
                seen.add(key)
                pool.append(variant.copy())
    return pool


SHAPE_POOL = _build_shape_pool()


def _intervals_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(int(a0), int(b0)) <= min(int(a1), int(b1))


def _template_info(grid: np.ndarray, comp: dict, bg: int) -> dict | None:
    color = int(comp["color"])
    r0, c0, r1, c1 = map(int, comp["bbox"])
    sprite = crop_bbox(grid, (r0, c0, r1, c1))
    color_mask = sprite == color
    if not (
        np.all(color_mask[0, :])
        and np.all(color_mask[-1, :])
        and np.all(color_mask[:, 0])
        and np.all(color_mask[:, -1])
    ):
        return None
    hole_mask = sprite == bg
    if not hole_mask.any():
        return None
    if hole_mask[0, :].any() or hole_mask[-1, :].any() or hole_mask[:, 0].any() or hole_mask[:, -1].any():
        return None
    if not np.all(color_mask | hole_mask):
        return None
    holes = components_mask(hole_mask, connectivity=4)
    if len(holes) != 1:
        return None
    hole_local = holes[0]
    if not (3 <= int(hole_local.shape[0]) <= 6):
        return None
    if int(comp["area"]) + int(hole_local.shape[0]) != int(sprite.size):
        return None
    hole_abs = hole_local + np.array([r0, c0], dtype=int)
    hole_bbox = bbox(hole_mask)
    assert hole_bbox is not None
    hr0, hc0, hr1, hc1 = map(int, hole_bbox)
    return {
        "kind": "template",
        "color": color,
        "bbox": (r0, c0, r1, c1),
        "hole_coords": hole_abs,
        "hole_bbox": (r0 + hr0, c0 + hc0, r0 + hr1, c0 + hc1),
        "shape": _normalize_coords(hole_local),
    }


def _object_info(comp: dict) -> dict:
    coords = comp["coords"]
    r0, c0, r1, c1 = map(int, comp["bbox"])
    return {
        "kind": "object",
        "color": int(comp["color"]),
        "bbox": (r0, c0, r1, c1),
        "coords": coords.copy(),
        "shape": _normalize_coords(coords),
    }


def _parse_scene(grid: np.ndarray) -> tuple[int, list[dict], list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    components = find_components(g, bg=bg, connectivity=4, monochromatic=True)
    templates = []
    objects = []
    for comp in components:
        template = _template_info(g, comp, bg)
        if template is not None:
            templates.append(template)
        else:
            objects.append(_object_info(comp))
    templates.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))
    objects.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))
    return bg, templates, objects


def _drop_assignments(objects: list[dict], templates: list[dict]) -> list[dict]:
    ordered = sorted(objects, key=lambda item: (item["bbox"][2], item["bbox"][0], item["bbox"][1]), reverse=True)
    assignments = []
    for obj in ordered:
        matched = None
        mismatches_before = 0
        for template in templates:
            if template["bbox"][0] <= obj["bbox"][2]:
                continue
            if not _intervals_overlap(obj["bbox"][1], obj["bbox"][3], template["bbox"][1], template["bbox"][3]):
                continue
            if obj["shape"] == template["shape"]:
                matched = template
                break
            mismatches_before += 1
        assignments.append(
            {
                "object": obj,
                "template": matched,
                "mismatches_before": mismatches_before,
            }
        )
    return assignments


def _apply_solution(grid: np.ndarray, assignments: list[dict], bg: int) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for item in assignments:
        coords = item["object"]["coords"]
        out[coords[:, 0], coords[:, 1]] = bg
    for item in assignments:
        template = item["template"]
        if template is None:
            continue
        coords = template["hole_coords"]
        out[coords[:, 0], coords[:, 1]] = int(item["object"]["color"])
    return out


def _solve_first_overlap_wrong(grid: np.ndarray) -> np.ndarray:
    bg, templates, objects = _parse_scene(grid)
    out = np.asarray(grid, dtype=int).copy()
    for obj in objects:
        coords = obj["coords"]
        out[coords[:, 0], coords[:, 1]] = bg
    ordered = sorted(objects, key=lambda item: (item["bbox"][2], item["bbox"][0], item["bbox"][1]), reverse=True)
    for obj in ordered:
        for template in templates:
            if template["bbox"][0] <= obj["bbox"][2]:
                continue
            if not _intervals_overlap(obj["bbox"][1], obj["bbox"][3], template["bbox"][1], template["bbox"][3]):
                continue
            coords = template["hole_coords"]
            out[coords[:, 0], coords[:, 1]] = int(obj["color"])
            break
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop each small object downward and fill only the first lower template gap whose normalized hole shape matches it.

    Input structure:
    - The background is black.
    - The left side contains 3-4 monochrome rectangular templates, each a solid rectangle with one enclosed 0-colored irregular hole.
    - The remaining non-background components are small 4-connected monochrome objects placed above or between templates.
    - Each template hole shape is unique, and at most one falling object shares that normalized shape.

    Rule:
    - Erase every small object from its original position.
    - Process the objects from lower initial position to higher initial position.
    - For each object, look downward through the templates whose column span overlaps the object's column span.
    - Ignore overlapping templates whose hole shape does not match the object's normalized shape; stop at the first lower overlapping template whose hole shape matches, and paint that hole with the object's color.
    - If no lower overlapping template matches, the object disappears.

    Key insight:
    - Templates are not ordinary obstacles: a falling object passes through mismatched rectangles and is captured only by the uniquely matching enclosed hole shape.
    """
    bg, templates, objects = _parse_scene(np.asarray(grid, dtype=int))
    assignments = _drop_assignments(objects, templates)
    return _apply_solution(grid, assignments, bg)


def _shape_sprite(mask: np.ndarray, color: int) -> np.ndarray:
    sprite = np.zeros(mask.shape, dtype=int)
    sprite[mask] = int(color)
    return sprite


def _can_place_sprite(grid: np.ndarray, sprite: np.ndarray, top: int, left: int) -> bool:
    h, w = sprite.shape
    if top < 0 or left < 0 or top + h > grid.shape[0] or left + w > grid.shape[1]:
        return False
    cells = np.argwhere(sprite != BG)
    occupied = {(int(top + r), int(left + c)) for r, c in cells}
    for rr, cc in occupied:
        if int(grid[rr, cc]) != BG:
            return False
    for rr, cc in occupied:
        for nr in range(rr - 1, rr + 2):
            for nc in range(cc - 1, cc + 2):
                if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                    if (nr, nc) not in occupied and int(grid[nr, nc]) != BG:
                        return False
    return True


def _paint_template(grid: np.ndarray, top: int, left: int, height: int, width: int, color: int, hole_mask: np.ndarray) -> dict:
    rect = np.full((height, width), int(color), dtype=int)
    hole_left = 1 if width == hole_mask.shape[1] + 2 else random.randint(1, width - hole_mask.shape[1] - 1)
    hole_top = 1
    rect[hole_top : hole_top + hole_mask.shape[0], hole_left : hole_left + hole_mask.shape[1]][hole_mask] = BG
    paste(grid, rect, top, left)
    hole_abs = np.argwhere(hole_mask) + np.array([top + hole_top, left + hole_left], dtype=int)
    return {
        "bbox": (top, left, top + height - 1, left + width - 1),
        "hole_coords": hole_abs,
        "shape": _mask_key(hole_mask),
    }


def _manual_expected(grid: np.ndarray, placements: list[dict], templates: list[dict]) -> np.ndarray:
    out = grid.copy()
    for placed in placements:
        coords = placed["coords"]
        out[coords[:, 0], coords[:, 1]] = BG
    for placed in placements:
        target = templates[placed["target_index"]]
        coords = target["hole_coords"]
        out[coords[:, 0], coords[:, 1]] = int(placed["color"])
    return out


def _plan_options(n_templates: int) -> list[list[int]]:
    if n_templates == 3:
        return [
            [0, 0, 1],
            [0, 1, 0],
        ]
    return [
        [0, 0, 1, 2],
        [0, 1, 0, 2],
        [0, 1, 1, 2],
    ]


def _generate_attempt() -> np.ndarray | None:
    n_templates = random.choice([3, 4])
    masks = [mask.copy() for mask in random.sample(SHAPE_POOL, n_templates)]
    template_heights = [int(mask.shape[0] + 2) for mask in masks]
    template_widths = [int(mask.shape[1] + 2 + random.randint(0, 1)) for mask in masks]
    top_margin = 5 if n_templates == 3 else 4
    gaps = [3] * (n_templates - 1)
    bottom_margin = 1
    height = top_margin + sum(template_heights) + sum(gaps) + bottom_margin
    if height > 30:
        return None
    right_edge = random.randint(8, 11)
    leftmost = min(right_edge - width + 1 for width in template_widths)
    if leftmost < 1:
        return None
    width = random.randint(max(18, right_edge + 6), min(30, right_edge + 12))
    if width > 30:
        return None

    colors = random.sample(PALETTE, 2 * n_templates)
    template_colors = colors[:n_templates]
    object_colors = colors[n_templates:]

    grid = np.zeros((height, width), dtype=int)
    template_specs = []
    row = top_margin
    for idx in range(n_templates):
        tw = template_widths[idx]
        th = template_heights[idx]
        left = right_edge - tw + 1
        spec = _paint_template(grid, row, left, th, tw, template_colors[idx], masks[idx])
        spec["index"] = idx
        template_specs.append(spec)
        row += th
        if idx < n_templates - 1:
            row += gaps[idx]

    regions = [(0, top_margin - 1)]
    for idx in range(1, n_templates):
        upper = template_specs[idx - 1]["bbox"][2] + 1
        lower = template_specs[idx]["bbox"][0] - 1
        regions.append((upper, lower))

    region_plan = random.choice(_plan_options(n_templates))
    target_order = list(range(n_templates))
    random.shuffle(target_order)
    target_order.sort(key=lambda idx: (region_plan[idx], -masks[idx].sum()))

    placements = []
    for target_index in target_order:
        mask = masks[target_index]
        sprite = _shape_sprite(mask, object_colors[target_index])
        region_index = region_plan[target_index]
        row0, row1 = regions[region_index]
        max_top = row1 - sprite.shape[0] + 1
        if row0 > max_top:
            return None
        min_overlap_left = min(template_specs[k]["bbox"][1] for k in range(region_index, target_index + 1))
        base_left = right_edge - sprite.shape[1] + 1
        candidate_cols = [c for c in range(max(min_overlap_left - sprite.shape[1] + 1, 0), base_left + 1)]
        candidate_cols = [c for c in candidate_cols if _intervals_overlap(c, c + sprite.shape[1] - 1, template_specs[target_index]["bbox"][1], template_specs[target_index]["bbox"][3])]
        random.shuffle(candidate_cols)
        candidate_rows = list(range(row0, max_top + 1))
        candidate_rows.sort(reverse=bool(random.getrandbits(1)))
        placed = None
        for top in candidate_rows:
            for left in candidate_cols:
                path_templates = template_specs[region_index : target_index + 1]
                if not all(_intervals_overlap(left, left + sprite.shape[1] - 1, t["bbox"][1], t["bbox"][3]) for t in path_templates):
                    continue
                if not _can_place_sprite(grid, sprite, top, left):
                    continue
                paste(grid, sprite, top, left, transparent=BG)
                placed = {
                    "target_index": target_index,
                    "color": object_colors[target_index],
                    "coords": np.argwhere(sprite != BG) + np.array([top, left], dtype=int),
                    "bbox": (top, left, top + sprite.shape[0] - 1, left + sprite.shape[1] - 1),
                    "shape": _mask_key(mask),
                }
                placements.append(placed)
                break
            if placed is not None:
                break
        if placed is None:
            return None

    expected = _manual_expected(grid, placements, template_specs)
    if not np.array_equal(main(grid), expected):
        return None

    bg, templates, objects = _parse_scene(grid)
    if bg != BG or len(templates) != n_templates or len(objects) != n_templates:
        return None
    assignments = _drop_assignments(objects, templates)
    if any(item["template"] is None for item in assignments):
        return None
    if len({template["shape"] for template in templates}) != n_templates:
        return None
    if len({obj["shape"] for obj in objects}) != n_templates:
        return None
    if not any(item["mismatches_before"] > 0 for item in assignments):
        return None
    if not any(obj["bbox"][0] > templates[0]["bbox"][0] for obj in objects):
        return None
    if np.array_equal(_solve_first_overlap_wrong(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
