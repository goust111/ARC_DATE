from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import bbox_w, find_components, get_bg, label_mask, paste, random_free_location


BG = 0
BLUE = 1
GREEN = 3
MAX_ATTEMPTS = 400


def _ring(height: int, width: int) -> np.ndarray:
    patch = np.ones((height, width), dtype=int)
    patch[1:-1, 1:-1] = 0
    return patch


HOLE_TEMPLATES = {
    3: [_ring(3, 3), _ring(4, 3)],
    4: [_ring(3, 4), _ring(4, 4)],
    5: [_ring(3, 5), _ring(4, 5)],
    6: [_ring(3, 6)],
}

SOLID_TEMPLATES = {
    2: [
        np.array([[1, 1], [1, 0], [1, 1]], dtype=int),
        np.array([[1, 1], [0, 1], [1, 1]], dtype=int),
    ],
    3: [
        np.array([[1, 1, 1], [1, 0, 0], [1, 1, 1]], dtype=int),
        np.array([[1, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int),
    ],
    4: [
        np.array([[1, 1, 1, 1], [0, 0, 1, 0], [0, 1, 1, 0]], dtype=int),
        np.array([[1, 1, 1, 0], [0, 1, 0, 0], [1, 1, 1, 1]], dtype=int),
    ],
    5: [
        np.array([[1, 1, 1, 1, 1], [0, 0, 1, 0, 0], [0, 1, 1, 1, 0]], dtype=int),
        np.array([[1, 1, 1, 1, 0], [0, 1, 0, 0, 0], [1, 1, 1, 0, 1]], dtype=int),
    ],
    6: [
        np.array([[1, 1, 1, 1, 1, 1], [0, 0, 0, 1, 0, 0], [0, 0, 1, 1, 1, 0]], dtype=int),
        np.array([[1, 1, 1, 1, 1, 0], [0, 0, 1, 0, 0, 0], [0, 1, 1, 1, 1, 1]], dtype=int),
    ],
}


def _flip_patch(patch: np.ndarray) -> np.ndarray:
    out = patch.copy()
    if random.random() < 0.5:
        out = np.flipud(out)
    if random.random() < 0.5:
        out = np.fliplr(out)
    return np.array(out, dtype=int)


def _object_hole_flags(grid: np.ndarray, objects: list[dict], bg: int) -> list[bool]:
    h, w = grid.shape
    obj_ids = np.zeros((h, w), dtype=int)
    for idx, obj in enumerate(objects, start=1):
        coords = obj["coords"]
        obj_ids[coords[:, 0], coords[:, 1]] = idx

    bg_labels, n_bg = label_mask(grid == bg, connectivity=4)
    boundary_labels = set(map(int, np.unique(np.concatenate(
        [bg_labels[0, :], bg_labels[-1, :], bg_labels[:, 0], bg_labels[:, -1]]
    ))))
    boundary_labels.discard(0)

    flags = [False] * len(objects)
    for label_id in range(1, n_bg + 1):
        if label_id in boundary_labels:
            continue

        coords = np.argwhere(bg_labels == label_id)
        owners = set()
        for r, c in coords:
            rr = int(r)
            cc = int(c)
            if rr > 0 and obj_ids[rr - 1, cc]:
                owners.add(int(obj_ids[rr - 1, cc]))
            if rr + 1 < h and obj_ids[rr + 1, cc]:
                owners.add(int(obj_ids[rr + 1, cc]))
            if cc > 0 and obj_ids[rr, cc - 1]:
                owners.add(int(obj_ids[rr, cc - 1]))
            if cc + 1 < w and obj_ids[rr, cc + 1]:
                owners.add(int(obj_ids[rr, cc + 1]))

        if len(owners) == 1:
            flags[next(iter(owners)) - 1] = True

    return flags


def _sorted_blue_objects(grid: np.ndarray) -> list[dict]:
    objects = find_components(
        grid,
        bg=BG,
        connectivity=4,
        monochromatic=True,
        colors=[BLUE],
        return_format="dict",
    )
    return sorted(objects, key=lambda obj: tuple(map(int, obj["bbox"])))


def _render_components(
    shape: tuple[int, int],
    components: list[dict],
    hole_flags: list[bool],
    *,
    hole_shift_mode: str,
    solid_shift_mode: str,
    recolor_holes: bool,
) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for obj, has_hole in zip(components, hole_flags):
        r0, c0, r1, c1 = map(int, obj["bbox"])
        local_h = r1 - r0 + 1
        local_w = c1 - c0 + 1
        mask = np.zeros((local_h, local_w), dtype=bool)
        rel = obj["coords"] - np.array([r0, c0])
        mask[rel[:, 0], rel[:, 1]] = True

        if has_hole:
            shift = local_w if hole_shift_mode == "width" else 1
            color = GREEN if recolor_holes else BLUE
        else:
            shift = local_w if solid_shift_mode == "width" else 1
            color = BLUE

        patch = np.full((local_h, local_w), color, dtype=int)
        paste(out, patch, r0, c0 + shift, mask=mask, clip=True)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Shift each blue object right using a hole-dependent rule, recoloring only the holed ones.

    Input structure:
    - The grid is a black background with 4-6 disconnected blue 4-connected objects.
    - Some blue objects enclose one or more black 4-connected background holes, while others are hole-free solid shapes.
    - Objects are separated from one another and have enough room on the right for their rule-specific motion.

    Rule:
    - Find every blue 4-connected object.
    - For each object, decide whether it contains any black 4-connected region that is not connected to the grid boundary.
    - If the object has a hole, recolor the whole object green and shift it right by its own bounding-box width.
    - If the object has no hole, keep it blue and shift it right by exactly 1 cell.
    - Erase all original object cells before painting the shifted copies, clipping any pixels that move past the right edge.

    Key insight:
    - The same topological test controls two different consequences at once: holed objects both change color and use a local width-based shift, while solid objects keep their color and use a fixed shift of 1.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert get_bg(g) == BG
    assert set(map(int, np.unique(g))).issubset({BG, BLUE})

    objects = _sorted_blue_objects(g)
    assert objects, "expected at least one blue object"
    hole_flags = _object_hole_flags(g, objects, BG)
    return _render_components(
        g.shape,
        objects,
        hole_flags,
        hole_shift_mode="width",
        solid_shift_mode="one",
        recolor_holes=True,
    )


def _build_specs() -> list[dict]:
    total = random.randint(4, 6)
    hole_count = random.randint(2, min(4, total - 2))
    solid_count = total - hole_count

    hole_widths = random.sample(sorted(HOLE_TEMPLATES), hole_count)
    shared_width = random.choice(hole_widths)

    solid_widths = [shared_width]
    if solid_count >= 2:
        second_choices = [w for w in sorted(SOLID_TEMPLATES) if w != shared_width]
        low_choices = [w for w in second_choices if w < max(hole_widths)]
        high_choices = [w for w in second_choices if w > min(hole_widths)]
        preferred = [w for w in second_choices if w in low_choices or w in high_choices]
        solid_widths.append(random.choice(preferred or second_choices))
    while len(solid_widths) < solid_count:
        solid_widths.append(random.choice(sorted(SOLID_TEMPLATES)))

    specs = []
    for width in hole_widths:
        patch = _flip_patch(random.choice(HOLE_TEMPLATES[width]))
        specs.append({"patch": patch, "has_hole": True, "width": int(patch.shape[1])})
    for width in solid_widths:
        patch = _flip_patch(random.choice(SOLID_TEMPLATES[width]))
        specs.append({"patch": patch, "has_hole": False, "width": int(patch.shape[1])})

    random.shuffle(specs)
    return specs


def _place_specs(specs: list[dict], height: int, width: int) -> tuple[np.ndarray, list[dict]] | None:
    grid = np.zeros((height, width), dtype=int)
    reserved = np.zeros((height, width), dtype=int)
    placed: list[dict] = []

    order = sorted(specs, key=lambda spec: (spec["patch"].shape[0] * (spec["patch"].shape[1] + (spec["width"] if spec["has_hole"] else 1))), reverse=True)
    for spec in order:
        patch = spec["patch"]
        shift = spec["width"] if spec["has_hole"] else 1
        reserve_patch = np.ones((patch.shape[0], patch.shape[1] + shift), dtype=int)
        try:
            r0, c0 = random_free_location(
                reserved,
                reserve_patch,
                transparent=BG,
                border=1,
                padding=1,
                padding_connectivity=8,
            )
        except ValueError:
            return None

        paste(reserved, reserve_patch, r0, c0, transparent=BG)
        paste(grid, patch, r0, c0, transparent=BG)
        placed.append(
            {
                "patch": patch.copy(),
                "has_hole": bool(spec["has_hole"]),
                "width": int(spec["width"]),
                "r": int(r0),
                "c": int(c0),
            }
        )

    return grid, sorted(placed, key=lambda item: (item["r"], item["c"]))


def _expected_from_plan(shape: tuple[int, int], plan: list[dict]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for item in plan:
        color = GREEN if item["has_hole"] else BLUE
        shift = item["width"] if item["has_hole"] else 1
        patch = np.full(item["patch"].shape, color, dtype=int)
        paste(out, patch, item["r"], item["c"] + shift, mask=item["patch"].astype(bool), clip=True)
    return out


def _validate_instance(inp: np.ndarray, plan: list[dict]) -> bool:
    if inp.ndim != 2 or max(inp.shape) > 30:
        return False
    if set(map(int, np.unique(inp))) - {BG, BLUE}:
        return False

    expected = _expected_from_plan(inp.shape, plan)
    try:
        actual = main(inp)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False

    objects = _sorted_blue_objects(inp)
    if len(objects) != len(plan) or not (4 <= len(objects) <= 6):
        return False

    hole_flags = _object_hole_flags(inp, objects, BG)
    hole_widths = [bbox_w(obj["bbox"]) for obj, flag in zip(objects, hole_flags) if flag]
    solid_widths = [bbox_w(obj["bbox"]) for obj, flag in zip(objects, hole_flags) if not flag]
    if len(hole_widths) < 2 or len(solid_widths) < 2:
        return False
    if len(set(hole_widths)) != len(hole_widths):
        return False
    if not (set(hole_widths) & set(solid_widths)):
        return False
    if len(set([1, *hole_widths])) < 2:
        return False
    if set(map(int, np.unique(actual))) != {BG, BLUE, GREEN}:
        return False

    wrong_all_one = _render_components(
        inp.shape,
        objects,
        hole_flags,
        hole_shift_mode="one",
        solid_shift_mode="one",
        recolor_holes=True,
    )
    if np.array_equal(wrong_all_one, expected):
        return False

    wrong_all_width = _render_components(
        inp.shape,
        objects,
        hole_flags,
        hole_shift_mode="width",
        solid_shift_mode="width",
        recolor_holes=True,
    )
    if np.array_equal(wrong_all_width, expected):
        return False

    first_hole_width = hole_widths[0]
    uniform_hole_shift = np.full(inp.shape, BG, dtype=int)
    for obj, flag in zip(objects, hole_flags):
        r0, c0, r1, c1 = map(int, obj["bbox"])
        local_h = r1 - r0 + 1
        local_w = c1 - c0 + 1
        mask = np.zeros((local_h, local_w), dtype=bool)
        rel = obj["coords"] - np.array([r0, c0])
        mask[rel[:, 0], rel[:, 1]] = True
        shift = first_hole_width if flag else 1
        color = GREEN if flag else BLUE
        patch = np.full((local_h, local_w), color, dtype=int)
        paste(uniform_hole_shift, patch, r0, c0 + shift, mask=mask, clip=True)
    if np.array_equal(uniform_hole_shift, expected):
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        specs = _build_specs()
        height = random.randint(18, 24)
        width = random.randint(18, 24)
        placed = _place_specs(specs, height, width)
        if placed is None:
            continue
        grid, plan = placed
        if _validate_instance(grid, plan):
            return grid
    raise RuntimeError("failed to generate a valid input grid")
