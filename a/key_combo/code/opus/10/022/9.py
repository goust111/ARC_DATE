from __future__ import annotations

from collections import Counter
from itertools import combinations
import random

import numpy as np


BLACK = 0
PALETTE = tuple(range(1, 10))

GRID_H_RANGE = (12, 14)
GRID_W_RANGE = (11, 15)
OBJECT_COUNT_RANGE = (3, 5)
ROW_BANDS = (0, 2, 4)
MAX_GENERATION_ATTEMPTS = 500


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _normalize(cells: set[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def _translate(cells: tuple[tuple[int, int], ...], dr: int, dc: int) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((r + dr, c + dc) for r, c in cells))


def _bbox(cells: tuple[tuple[int, int], ...]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _center_col(cells: tuple[tuple[int, int], ...]) -> int:
    _r0, c0, _r1, c1 = _bbox(cells)
    return (c0 + c1) // 2


def _is_connected4(cells: tuple[tuple[int, int], ...]) -> bool:
    todo = [cells[0]]
    seen = {cells[0]}
    cell_set = set(cells)
    while todo:
        r, c = todo.pop()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in cell_set and (nr, nc) not in seen:
                seen.add((nr, nc))
                todo.append((nr, nc))
    return len(seen) == len(cells)


def _dihedral_variants(cells: tuple[tuple[int, int], ...]) -> set[tuple[tuple[int, int], ...]]:
    variants: set[tuple[tuple[int, int], ...]] = set()
    points = list(cells)
    for rotation in range(4):
        rotated = []
        for r, c in points:
            rr, cc = r, c
            for _ in range(rotation):
                rr, cc = cc, -rr
            rotated.append((rr, cc))
        for flip in (False, True):
            transformed = [(rr, -cc) if flip else (rr, cc) for rr, cc in rotated]
            variants.add(_normalize(set(transformed)))
    return variants


def _slide_cells(cells: tuple[tuple[int, int], ...]) -> tuple[tuple[tuple[int, int], ...], tuple[bool, ...]]:
    cell_set = set(cells)
    moved = []
    supported = []
    for r, c in cells:
        has_support = any(rr > r and cc > c for rr, cc in cell_set)
        supported.append(has_support)
        moved.append((r, c + 1) if has_support else (r, c))
    return tuple(sorted(moved)), tuple(supported)


def _all_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _analyze_objects(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    colors = sorted(int(color) for color in np.unique(g) if int(color) != BLACK)
    objects = []
    for color in colors:
        comps = _all_components(g == color)
        assert len(comps) == 1, "each color should form exactly one object"
        coords_arr = comps[0]
        cells = tuple(sorted((int(r), int(c)) for r, c in coords_arr))
        moved_cells, supported = _slide_cells(cells)
        objects.append(
            {
                "color": color,
                "cells": cells,
                "moved_cells": moved_cells,
                "supported": supported,
                "area": len(cells),
                "bbox": _bbox(cells),
                "moved_bbox": _bbox(moved_cells),
                "center_before": _center_col(cells),
                "center_after": _center_col(moved_cells),
            }
        )
    return objects


def _priority_key(obj: dict) -> tuple[int, int, int, int]:
    r0, c0, _r1, _c1 = obj["moved_bbox"]
    return (-int(obj["area"]), int(r0), int(c0), int(obj["color"]))


def _resolve_projection_columns(objects: list[dict]) -> dict[int, int]:
    grouped: dict[int, list[dict]] = {}
    for obj in objects:
        grouped.setdefault(int(obj["center_after"]), []).append(obj)

    desired: list[tuple[int, tuple[int, int, int, int], dict]] = []
    for center, group in grouped.items():
        for offset, obj in enumerate(sorted(group, key=_priority_key)):
            desired.append((center + offset, _priority_key(obj), obj))

    desired.sort(key=lambda item: (item[0], item[1]))
    used_cols: set[int] = set()
    projections: dict[int, int] = {}
    for desired_col, _priority, obj in desired:
        col = int(desired_col)
        while col in used_cols:
            col += 1
        used_cols.add(col)
        projections[int(obj["color"])] = col
    return projections


def _render_output(shape: tuple[int, int], objects: list[dict]) -> np.ndarray:
    h, w = shape
    out = np.zeros((h, w), dtype=int)
    for obj in objects:
        color = int(obj["color"])
        for r, c in obj["moved_cells"]:
            out[r, c] = color
    projections = _resolve_projection_columns(objects)
    for obj in objects:
        col = int(projections[int(obj["color"])])
        assert 0 <= col < w, "projection must stay inside the grid"
        out[h - 1, col] = int(obj["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each small object right where it has southeast support, then project the slid centers to the bottom row.

    Input structure:
    - Black background contains 3-5 small 4-connected monochrome objects in the upper half.
    - Each object uses a distinct non-black color and has some pixels with a same-object pixel strictly southeast of them.
    - The bottom row is empty black background before projection.

    Rule:
    - For each object independently, move every pixel one column right iff some pixel of the same object lies strictly below and strictly to its right, using the original object positions for that test.
    - After this one-round slide, compute each object's new bounding box and take its horizontal center column.
    - Mark that center on the bottom row with the object's color while keeping the slid objects visible.
    - If multiple objects want the same center column, the larger-area object keeps it; equal areas are broken by the object whose slid bbox starts higher. Lower-priority objects shift right by one additional column each.

    Key insight:
    - The projection is determined by the post-slide geometry, not the original shape, and collisions are resolved by object area before the lower-priority projection shifts right.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    objects = _analyze_objects(g)
    return _render_output(g.shape, objects)


def _build_shape_library() -> tuple[list[dict], list[dict]]:
    cells = [(r, c) for r in range(3) for c in range(4)]
    valid = []
    center_changed = []
    for area in range(3, 8):
        for combo in combinations(cells, area):
            if (0, 0) not in combo:
                continue
            shape = _normalize(set(combo))
            if not _is_connected4(shape):
                continue
            moved, supported = _slide_cells(shape)
            if len(set(moved)) != len(shape):
                continue
            if not any(supported) or all(supported):
                continue
            if len(_dihedral_variants(shape)) < 8:
                continue
            bbox = _bbox(shape)
            moved_bbox = _bbox(moved)
            entry = {
                "cells": shape,
                "area": len(shape),
                "bbox": bbox,
                "moved_cells": tuple(sorted(moved)),
                "moved_bbox": moved_bbox,
                "center_before": _center_col(shape),
                "center_after": _center_col(tuple(sorted(moved))),
            }
            valid.append(entry)
            if entry["center_before"] != entry["center_after"]:
                center_changed.append(entry)
    return valid, center_changed


SHAPE_LIBRARY, CENTER_CHANGED_LIBRARY = _build_shape_library()


def _shape_max_col(shape: dict) -> int:
    return max(max(c for _, c in shape["cells"]), shape["moved_bbox"][3])


def _cells_with_gap(existing: list[tuple[tuple[int, int], ...]], candidate: tuple[tuple[int, int], ...], gap: int) -> bool:
    for other in existing:
        for r, c in candidate:
            for rr, cc in other:
                if max(abs(r - rr), abs(c - cc)) <= gap:
                    return False
    return True


def _pick_collision_pair(branch_mode: str) -> list[tuple[dict, int]]:
    if branch_mode == "area":
        winner_area, loser_area = random.choice([(7, 5), (7, 6), (6, 5)])
        winner = random.choice([shape for shape in CENTER_CHANGED_LIBRARY if shape["area"] == winner_area])
        loser = random.choice([shape for shape in CENTER_CHANGED_LIBRARY if shape["area"] == loser_area])
        return [(winner, 4), (loser, 0)]

    shared_area = random.choice([5, 6, 7])
    a, b = random.sample([shape for shape in CENTER_CHANGED_LIBRARY if shape["area"] == shared_area], 2)
    top_row, bottom_row = sorted(random.sample(ROW_BANDS, 2))
    return [(a, top_row), (b, bottom_row)]


def _collision_center_range(width: int, pair: list[tuple[dict, int]]) -> tuple[int, int] | None:
    lower = 0
    upper = width - 2
    for shape, _row in pair:
        local_center = int(shape["center_after"])
        max_col = _shape_max_col(shape)
        lower = max(lower, local_center)
        upper = min(upper, width - 1 - max_col + local_center)
    if lower > upper:
        return None
    return lower, upper


def _build_scene() -> tuple[np.ndarray, str] | None:
    height = random.randint(*GRID_H_RANGE)
    width = random.randint(*GRID_W_RANGE)
    object_count = random.randint(*OBJECT_COUNT_RANGE)
    branch_mode = random.choice(["area", "tie"])
    pair = _pick_collision_pair(branch_mode)

    center_range = _collision_center_range(width, pair)
    if center_range is None:
        return None
    target_center = random.randint(center_range[0], center_range[1])

    objects = []
    input_placements: list[tuple[tuple[int, int], ...]] = []
    output_placements: list[tuple[tuple[int, int], ...]] = []
    reserved_centers = {target_center, target_center + 1}
    used_colors = random.sample(PALETTE, object_count)

    for color, (shape, row) in zip(used_colors[:2], pair):
        c0 = target_center - int(shape["center_after"])
        cells = _translate(shape["cells"], row, c0)
        moved_cells = _translate(shape["moved_cells"], row, c0)
        if max(r for r, _ in cells) > 6:
            return None
        if not _cells_with_gap(input_placements, cells, gap=1):
            return None
        if not _cells_with_gap(output_placements, moved_cells, gap=0):
            return None
        objects.append(
            {
                "color": color,
                "cells": cells,
                "shape": shape,
            }
        )
        input_placements.append(cells)
        output_placements.append(moved_cells)

    attempts = 0
    while len(objects) < object_count and attempts < 400:
        attempts += 1
        shape = random.choice(SHAPE_LIBRARY)
        row = random.choice(ROW_BANDS)
        max_col = _shape_max_col(shape)
        c0 = random.randint(0, width - 1 - max_col)
        center_after = c0 + int(shape["center_after"])
        if center_after in reserved_centers:
            continue
        cells = _translate(shape["cells"], row, c0)
        moved_cells = _translate(shape["moved_cells"], row, c0)
        if max(r for r, _ in cells) > 6:
            continue
        if not _cells_with_gap(input_placements, cells, gap=1):
            continue
        if not _cells_with_gap(output_placements, moved_cells, gap=0):
            continue
        objects.append(
            {
                "color": used_colors[len(objects)],
                "cells": cells,
                "shape": shape,
            }
        )
        input_placements.append(cells)
        output_placements.append(moved_cells)
        reserved_centers.add(center_after)

    if len(objects) != object_count:
        return None

    grid = np.zeros((height, width), dtype=int)
    for obj in objects:
        for r, c in obj["cells"]:
            grid[r, c] = int(obj["color"])
    return grid, branch_mode


def _projection_signature(objects: list[dict], *, use_slid_centers: bool) -> tuple[tuple[int, int], ...]:
    grouped: dict[int, list[dict]] = {}
    center_key = "center_after" if use_slid_centers else "center_before"
    for obj in objects:
        grouped.setdefault(int(obj[center_key]), []).append(obj)
    signature = []
    for center, group in grouped.items():
        for offset, obj in enumerate(sorted(group, key=_priority_key)):
            signature.append((int(obj["color"]), center + offset))
    return tuple(sorted(signature))


def _semantic_checks(grid: np.ndarray, branch_mode: str) -> bool:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or max(g.shape) > 30:
        return False
    if np.any(g[-1, :] != BLACK):
        return False

    objects = _analyze_objects(g)
    if not (3 <= len(objects) <= 5):
        return False
    if len({int(obj["color"]) for obj in objects}) != len(objects):
        return False

    object_cells = [obj["cells"] for obj in objects]
    moved_cells = [obj["moved_cells"] for obj in objects]
    for obj in objects:
        if not (3 <= int(obj["area"]) <= 7):
            return False
        if max(r for r, _ in obj["cells"]) > 6:
            return False
        if not _is_connected4(obj["cells"]):
            return False
        if len(_dihedral_variants(_normalize(set(obj["cells"])))) < 8:
            return False
        if len(set(obj["moved_cells"])) != len(obj["moved_cells"]):
            return False
        if not any(obj["supported"]) or all(obj["supported"]):
            return False

    for i, cells in enumerate(object_cells):
        if not _cells_with_gap(object_cells[:i] + object_cells[i + 1 :], cells, gap=1):
            return False
    for i, cells in enumerate(moved_cells):
        if not _cells_with_gap(moved_cells[:i] + moved_cells[i + 1 :], cells, gap=0):
            return False

    collision_groups: list[list[dict]] = []
    grouped: dict[int, list[dict]] = {}
    for obj in objects:
        grouped.setdefault(int(obj["center_after"]), []).append(obj)
    for group in grouped.values():
        if len(group) >= 2:
            collision_groups.append(group)
    if len(collision_groups) != 1 or len(collision_groups[0]) != 2:
        return False

    pair = sorted(collision_groups[0], key=_priority_key)
    shared_center = int(pair[0]["center_after"])
    projections = _resolve_projection_columns(objects)
    if {projections[int(obj["color"])] for obj in pair} != {shared_center, shared_center + 1}:
        return False
    if len(set(projections.values())) != len(projections):
        return False
    if max(projections.values()) >= g.shape[1]:
        return False

    if branch_mode == "area":
        if int(pair[0]["area"]) == int(pair[1]["area"]):
            return False
        winner = min(pair, key=lambda obj: projections[int(obj["color"])])
        loser = max(pair, key=lambda obj: projections[int(obj["color"])])
        if not (int(winner["area"]) > int(loser["area"])):
            return False
        if int(winner["moved_bbox"][0]) < int(loser["moved_bbox"][0]):
            return False
    else:
        if int(pair[0]["area"]) != int(pair[1]["area"]):
            return False
        if int(pair[0]["moved_bbox"][0]) == int(pair[1]["moved_bbox"][0]):
            return False

    if not all(int(obj["center_before"]) != int(obj["center_after"]) for obj in pair):
        return False
    if _projection_signature(objects, use_slid_centers=True) == _projection_signature(objects, use_slid_centers=False):
        return False

    out = main(g)
    if not isinstance(out, np.ndarray) or out.shape != g.shape:
        return False
    if not np.array_equal(out, _render_output(g.shape, objects)):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        built = _build_scene()
        if built is None:
            continue
        grid, branch_mode = built
        if _semantic_checks(grid, branch_mode):
            return grid
    raise RuntimeError("failed to generate a valid input for task 9")
