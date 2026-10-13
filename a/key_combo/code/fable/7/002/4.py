from __future__ import annotations

import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, find_components, get_bg


SHAPE_LIBRARY = [
    np.array([[1, 1], [1, 0]], dtype=int),
    np.array([[1, 1], [0, 1]], dtype=int),
    np.array([[1, 1], [1, 1]], dtype=int),
    np.array([[1, 1, 1]], dtype=int),
    np.array([[1], [1], [1]], dtype=int),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=int),
    np.array([[1, 0], [1, 1], [1, 0]], dtype=int),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=int),
    np.array([[0, 1, 1], [1, 1, 0]], dtype=int),
    np.array([[1, 1, 1], [1, 0, 0]], dtype=int),
    np.array([[1, 1, 1], [0, 0, 1]], dtype=int),
    np.array([[1, 1, 0], [1, 1, 1]], dtype=int),
]

WEIGHT_AREA_POOL = tuple(range(1, 9))
GENERATION_ATTEMPTS = 1200
BRANCH_ATTEMPTS = 160


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    return mask[rows][:, cols]


def _random_shape(rng: random.Random) -> np.ndarray:
    mask = np.array(rng.choice(SHAPE_LIBRARY), dtype=int)
    if rng.random() < 0.5:
        mask = np.fliplr(mask)
    mask = np.rot90(mask, rng.randrange(4))
    return _trim_mask(mask)


def _mask_cells(mask: np.ndarray, r0: int = 0, c0: int = 0) -> list[tuple[int, int]]:
    pts = np.argwhere(mask > 0)
    return [(int(r0 + r), int(c0 + c)) for r, c in pts]


def _cells_set(cells: list[tuple[int, int]]) -> set[tuple[int, int]]:
    return {(int(r), int(c)) for r, c in cells}


def _weight_dims(area: int, rng: random.Random) -> tuple[int, int]:
    options: list[tuple[int, int]] = []
    for h in range(1, area + 1):
        if area % h != 0:
            continue
        w = area // h
        if max(h, w) <= 8:
            options.append((h, w))
    options.sort(key=lambda hw: (max(hw), hw[1], hw[0]))
    best_span = max(options[0])
    compact = [hw for hw in options if max(hw) == best_span]
    return rng.choice(compact)


def _weight_layout(
    areas: list[int], colors: list[int], width: int, height: int, rng: random.Random
) -> list[dict[str, object]] | None:
    dims = [_weight_dims(area, rng) for area in areas]
    total_width = sum(w for _h, w in dims) + max(0, len(dims) - 1)
    if total_width > width:
        return None

    slack = width - total_width
    gaps = [0] * (len(dims) + 1)
    for _ in range(slack):
        gaps[rng.randrange(len(gaps))] += 1

    x = gaps[0]
    weights: list[dict[str, object]] = []
    for idx, (color, area, (h, w)) in enumerate(zip(colors, areas, dims)):
        r0 = height - h
        cells = [(r, c) for r in range(r0, height) for c in range(x, x + w)]
        weights.append(
            {
                "color": int(color),
                "area": int(area),
                "bbox": (int(r0), int(x), int(height - 1), int(x + w - 1)),
                "cells": cells,
            }
        )
        x += w
        if idx != len(dims) - 1:
            x += 1 + gaps[idx + 1]
    return weights


def _simulate_drop(
    coords: list[tuple[int, int]], quota: int, occupied: set[tuple[int, int]], height: int
) -> tuple[list[tuple[int, int]], int, str]:
    current = [(int(r), int(c)) for r, c in coords]
    moved = 0
    stop_reason = "quota"
    for step in range(1, int(quota) + 1):
        shifted = [(int(r + step), int(c)) for r, c in current]
        if any(r >= height for r, _c in shifted):
            stop_reason = "bottom"
            break
        if any((r, c) in occupied for r, c in shifted):
            stop_reason = "collision"
            break
        moved = step
    final = [(int(r + moved), int(c)) for r, c in current]
    return final, int(moved), stop_reason


def _parse_scene(grid: np.ndarray) -> tuple[int, list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    components = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    by_color: dict[int, list[dict[str, object]]] = defaultdict(list)
    for comp in components:
        by_color[int(comp["color"])].append(comp)

    objects: list[dict[str, object]] = []
    for color in sorted(by_color):
        parts = by_color[color]
        bottom_parts = []
        floating_parts = []
        for part in parts:
            coords = [(int(r), int(c)) for r, c in np.asarray(part["coords"], dtype=int)]
            if any(r == g.shape[0] - 1 for r, _c in coords):
                bottom_parts.append((part, coords))
            else:
                floating_parts.append((part, coords))

        assert len(bottom_parts) == 1, f"color {color} must have exactly one bottom weight"
        assert len(floating_parts) == 1, f"color {color} must have exactly one floating object"

        weight_part, weight_coords = bottom_parts[0]
        float_part, float_coords = floating_parts[0]
        weight_bbox = tuple(int(v) for v in weight_part["bbox"])
        wr0, wc0, wr1, wc1 = weight_bbox
        assert int(weight_part["area"]) == (wr1 - wr0 + 1) * (wc1 - wc0 + 1), "weight must be solid"

        objects.append(
            {
                "color": int(color),
                "quota": int(weight_part["area"]),
                "initial_coords": [(int(r), int(c)) for r, c in float_coords],
                "initial_bottom": max(int(r) for r, _c in float_coords),
                "weight_coords": weight_coords,
            }
        )

    quotas = [int(obj["quota"]) for obj in objects]
    assert len(quotas) == len(set(quotas)), "weight areas must be unique"
    return bg, objects


def _trace_solution(input_grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(input_grid, dtype=int)
    bg, objects = _parse_scene(g)

    out = np.full_like(g, bg)
    settled: set[tuple[int, int]] = set()
    trace_objects: list[dict[str, object]] = []
    for obj in sorted(objects, key=lambda item: int(item["initial_bottom"]), reverse=True):
        final_coords, moved, stop_reason = _simulate_drop(
            obj["initial_coords"], int(obj["quota"]), settled, g.shape[0]
        )
        color = int(obj["color"])
        for r, c in final_coords:
            out[r, c] = color
        settled.update(final_coords)
        trace_objects.append(
            {
                "color": color,
                "quota": int(obj["quota"]),
                "initial_coords": list(obj["initial_coords"]),
                "final_coords": final_coords,
                "actual_shift": int(moved),
                "stop_reason": stop_reason,
                "initial_bottom": int(obj["initial_bottom"]),
            }
        )

    return {"bg": bg, "objects": trace_objects, "output": out}


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Drop each floating object downward by the area of its same-colored bottom weight.

    Input structure:
    - Black background with several distinct-color monochrome 4-connected objects floating above the bottom.
    - The bottom edge contains one solid weight block per object; each weight shares its object's color and all weight areas are distinct.
    - Floating objects may overlap in column span but do not initially touch each other or the weights.

    Rule:
    - Remove all bottom weights.
    - Process floating objects from lower initial position to higher initial position.
    - For each object, let k be the area of its same-colored weight block; move the object straight down by at most k rows, stopping earlier if the next row would overlap a previously settled object or leave the grid.

    Key insight:
    - The falling distance is not encoded near the object itself: it is remotely provided by the area of a separate same-colored weight block, and the processing order changes later collisions.
    """
    return _trace_solution(np.asarray(input_grid, dtype=int))["output"]


def _paint_cells(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[int(r), int(c)] = int(color)


def _free_of_overlap(cells: list[tuple[int, int]], occupied: set[tuple[int, int]]) -> bool:
    return all((int(r), int(c)) not in occupied for r, c in cells)


def _overlap_columns(mask: np.ndarray, target_cols: set[int], width: int) -> list[int]:
    rel_cols = sorted({int(c) for _r, c in np.argwhere(mask > 0)})
    xs = []
    for c0 in range(width - mask.shape[1] + 1):
        shifted = {int(c0 + c) for c in rel_cols}
        if shifted & target_cols:
            xs.append(int(c0))
    return xs


def _generic_place(
    mask: np.ndarray,
    width: int,
    row_hi: int,
    occupied_initial: set[tuple[int, int]],
    quota: int,
    settled_below: set[tuple[int, int]],
    height: int,
    rng: random.Random,
    require_reason: str | None = None,
    require_midair_after_quota: bool = False,
) -> dict[str, object] | None:
    if row_hi < 0:
        return None
    xs = list(range(width - mask.shape[1] + 1))
    rows = list(range(row_hi + 1))
    rng.shuffle(xs)
    rng.shuffle(rows)
    for r0 in rows[:BRANCH_ATTEMPTS]:
        for c0 in xs[:BRANCH_ATTEMPTS]:
            cells = _mask_cells(mask, r0, c0)
            if not _free_of_overlap(cells, occupied_initial):
                continue
            final_cells, actual_shift, reason = _simulate_drop(cells, quota, settled_below, height)
            if actual_shift <= 0:
                continue
            if require_reason is not None and reason != require_reason:
                continue
            if require_midair_after_quota and not (
                reason == "quota" and max(r for r, _c in final_cells) < height - 1
            ):
                continue
            return {"mask": mask, "r0": int(r0), "c0": int(c0), "cells": cells}
    return None


def _build_attempt(rng: random.Random) -> np.ndarray | None:
    core_count = 3
    extra_count = rng.choice([0, 1])
    object_count = core_count + extra_count

    colors = rng.sample(list(Color.NOT_BLACK), object_count)
    k_midair = rng.choice([1, 2, 3])
    k_collision = rng.choice([k for k in WEIGHT_AREA_POOL if k >= 4 and k != k_midair])
    k_bottom = rng.choice([k for k in WEIGHT_AREA_POOL if k >= 4 and k not in {k_midair, k_collision}])
    remaining = [k for k in WEIGHT_AREA_POOL if k not in {k_midair, k_collision, k_bottom}]
    quotas = [k_bottom, k_collision, k_midair]
    if extra_count:
        quotas.append(rng.choice(remaining))

    paired = list(zip(colors, quotas))
    rng.shuffle(paired)
    lower_color, k_bottom = paired[0]
    collision_color, k_collision = paired[1]
    midair_color, k_midair = paired[2]
    extra_pair = paired[3] if extra_count else None

    all_areas = [k_bottom, k_collision, k_midair]
    all_colors = [lower_color, collision_color, midair_color]
    if extra_pair is not None:
        all_colors.append(extra_pair[0])
        all_areas.append(extra_pair[1])

    compact_width = sum(_weight_dims(area, rng)[1] for area in all_areas) + max(0, object_count - 1)
    width = rng.randint(max(18, compact_width + 1), 28)
    height = rng.randint(18, 26)

    weights = _weight_layout(all_areas, all_colors, width, height, rng)
    if weights is None:
        return None
    max_weight_h = max(int(w["bbox"][2] - w["bbox"][0] + 1) for w in weights)
    clear_row = height - max_weight_h - 2
    if clear_row < 7:
        return None

    lower_mask = _random_shape(rng)
    collision_mask = _random_shape(rng)
    midair_mask = _random_shape(rng)
    extra_mask = _random_shape(rng) if extra_pair is not None else None

    lower_h = lower_mask.shape[0]
    lower_cols = list(range(width - lower_mask.shape[1] + 1))
    rng.shuffle(lower_cols)
    lower_obj = None
    for c0 in lower_cols:
        max_r0 = clear_row - lower_h + 1
        min_r0 = max(4, height - k_bottom - lower_h)
        choices = list(range(min_r0, max_r0 + 1))
        rng.shuffle(choices)
        for r0 in choices[:BRANCH_ATTEMPTS]:
            cells = _mask_cells(lower_mask, r0, c0)
            final_cells, moved, reason = _simulate_drop(cells, k_bottom, set(), height)
            if reason != "bottom" or moved >= k_bottom:
                continue
            lower_obj = {"mask": lower_mask, "r0": int(r0), "c0": int(c0), "cells": cells, "final": final_cells}
            break
        if lower_obj is not None:
            break
    if lower_obj is None:
        return None

    occupied_initial = set(lower_obj["cells"])
    settled_after_lower = _cells_set(lower_obj["final"])
    lower_final_cols = {c for _r, c in lower_obj["final"]}

    collision_row_hi = lower_obj["r0"] - collision_mask.shape[0] - 2
    if collision_row_hi < 1:
        return None
    collision_xs = _overlap_columns(collision_mask, lower_final_cols, width)
    rng.shuffle(collision_xs)
    collision_obj = None
    for c0 in collision_xs[:BRANCH_ATTEMPTS]:
        rows = list(range(collision_row_hi + 1))
        rng.shuffle(rows)
        for r0 in rows[:BRANCH_ATTEMPTS]:
            cells = _mask_cells(collision_mask, r0, c0)
            if not _free_of_overlap(cells, occupied_initial):
                continue
            final_cells, moved, reason = _simulate_drop(cells, k_collision, settled_after_lower, height)
            if reason != "collision" or moved >= k_collision or moved <= 0:
                continue
            collision_obj = {
                "mask": collision_mask,
                "r0": int(r0),
                "c0": int(c0),
                "cells": cells,
                "final": final_cells,
            }
            break
        if collision_obj is not None:
            break
    if collision_obj is None:
        return None

    occupied_initial.update(collision_obj["cells"])
    settled_after_collision = settled_after_lower | _cells_set(collision_obj["final"])

    midair_row_hi = collision_obj["r0"] - midair_mask.shape[0] - 2
    if midair_row_hi < 0:
        return None
    midair_obj = _generic_place(
        midair_mask,
        width,
        midair_row_hi,
        occupied_initial,
        k_midair,
        settled_after_collision,
        height,
        rng,
        require_midair_after_quota=True,
    )
    if midair_obj is None:
        return None

    placed_objects = [
        (lower_color, k_bottom, lower_obj),
        (collision_color, k_collision, collision_obj),
        (midair_color, k_midair, midair_obj),
    ]
    occupied_initial.update(midair_obj["cells"])
    settled_after_midair = settled_after_collision | _cells_set(
        _simulate_drop(midair_obj["cells"], k_midair, settled_after_collision, height)[0]
    )

    if extra_pair is not None and extra_mask is not None:
        extra_color, extra_quota = extra_pair
        extra_row_hi = midair_obj["r0"] - extra_mask.shape[0] - 2
        if extra_row_hi < 0:
            return None
        extra_obj = _generic_place(
            extra_mask,
            width,
            extra_row_hi,
            occupied_initial,
            int(extra_quota),
            settled_after_midair,
            height,
            rng,
        )
        if extra_obj is None:
            return None
        placed_objects.append((extra_color, int(extra_quota), extra_obj))

    grid = np.full((height, width), Color.BLACK, dtype=int)
    for color, _quota, placed in placed_objects:
        _paint_cells(grid, placed["cells"], int(color))
    for weight in weights:
        _paint_cells(grid, weight["cells"], int(weight["color"]))

    trace = _trace_solution(grid)
    reasons = {obj["stop_reason"] for obj in trace["objects"]}
    if not {"bottom", "collision", "quota"}.issubset(reasons):
        return None
    if object_count != len(trace["objects"]):
        return None
    if len({obj["initial_bottom"] for obj in trace["objects"]}) != len(trace["objects"]):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        seed = random.randrange(10**9)
        grid = _build_attempt(random.Random(seed))
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input after many attempts")


def _validate_generated_case(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30
    trace = _trace_solution(g)
    out = np.asarray(trace["output"], dtype=int)
    assert out.ndim == 2
    assert max(out.shape) <= 30
    assert np.array_equal(out, main(g))

    bg, objects = _parse_scene(g)
    assert bg == Color.BLACK
    assert len(objects) in {3, 4}
    assert len({obj["quota"] for obj in objects}) == len(objects)

    reasons = [str(obj["stop_reason"]) for obj in trace["objects"]]
    assert "collision" in reasons
    assert "quota" in reasons
    assert "bottom" in reasons

    for obj in trace["objects"]:
        final_cells = obj["final_coords"]
        assert final_cells
        assert all(0 <= r < g.shape[0] and 0 <= c < g.shape[1] for r, c in final_cells)
        if obj["stop_reason"] == "quota":
            assert obj["actual_shift"] == obj["quota"]
            assert max(r for r, _c in final_cells) < g.shape[0] - 1
        else:
            assert obj["actual_shift"] < obj["quota"]

    return {
        "shape": tuple(int(v) for v in g.shape),
        "object_count": len(objects),
        "reasons": tuple(reasons),
        "output_shape": tuple(int(v) for v in out.shape),
    }
