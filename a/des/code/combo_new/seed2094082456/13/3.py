from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6

DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

SYMMETRY_KINDS = ("both", "horizontal", "vertical", "none")
GRID_SIZE_RANGE = (18, 28)
OBJECT_COUNT_CHOICES = (2, 3, 3, 4, 4)
GEN_ATTEMPTS = 400


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _translate(coords: list[tuple[int, int]], dr: int, dc: int) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for r, c in coords]


def _shape_size(coords: list[tuple[int, int]]) -> tuple[int, int]:
    r0, c0, r1, c1 = _bbox(coords)
    return r1 - r0 + 1, c1 - c0 + 1


def _make_object(coords: list[tuple[int, int]]) -> dict[str, object]:
    pts = sorted((int(r), int(c)) for r, c in coords)
    r0, c0, r1, c1 = _bbox(pts)
    row_bounds: dict[int, list[int]] = {}
    col_bounds: dict[int, list[int]] = {}
    for r, c in pts:
        if r not in row_bounds:
            row_bounds[r] = [c, c]
        else:
            row_bounds[r][0] = min(row_bounds[r][0], c)
            row_bounds[r][1] = max(row_bounds[r][1], c)
        if c not in col_bounds:
            col_bounds[c] = [r, r]
        else:
            col_bounds[c][0] = min(col_bounds[c][0], r)
            col_bounds[c][1] = max(col_bounds[c][1], r)
    return {
        "coords": pts,
        "set": set(pts),
        "bbox": (r0, c0, r1, c1),
        "row_bounds": {r: (lo, hi) for r, (lo, hi) in row_bounds.items()},
        "col_bounds": {c: (lo, hi) for c, (lo, hi) in col_bounds.items()},
    }


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            seen[r, c] = True
            q: deque[tuple[int, int]] = deque([(r, c)])
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _marker_relation(
    obj: dict[str, object], marker: tuple[int, int], grey_mask: np.ndarray
) -> tuple[int, str] | None:
    mr, mc = marker
    row_bounds = obj["row_bounds"]
    col_bounds = obj["col_bounds"]

    if mc in col_bounds:
        top_r, bot_r = col_bounds[mc]
        if mr < top_r and not bool(grey_mask[mr + 1 : top_r, mc].any()):
            return top_r - mr - 1, "down"
        if mr > bot_r and not bool(grey_mask[bot_r + 1 : mr, mc].any()):
            return mr - bot_r - 1, "up"

    if mr in row_bounds:
        left_c, right_c = row_bounds[mr]
        if mc < left_c and not bool(grey_mask[mr, mc + 1 : left_c].any()):
            return left_c - mc - 1, "right"
        if mc > right_c and not bool(grey_mask[mr, right_c + 1 : mc].any()):
            return mc - right_c - 1, "left"

    return None


def _marker_assignments(
    objects: list[dict[str, object]], markers: list[tuple[int, int]], grey_mask: np.ndarray
) -> list[tuple[int, str]] | None:
    assignments: list[tuple[int, str]] = []
    for obj in objects:
        candidates: list[tuple[int, int, str, int, int]] = []
        for idx, marker in enumerate(markers):
            rel = _marker_relation(obj, marker, grey_mask)
            if rel is None:
                continue
            gap, move_dir = rel
            candidates.append((gap, idx, move_dir, marker[0], marker[1]))
        if not candidates:
            return None
        candidates.sort()
        if len(candidates) > 1 and candidates[1][0] == candidates[0][0]:
            return None
        _, idx, move_dir, _, _ = candidates[0]
        assignments.append((idx, move_dir))
    if len({idx for idx, _ in assignments}) != len(objects):
        return None
    return assignments


def _slide_coords(
    coords: list[tuple[int, int]],
    direction: str,
    shape: tuple[int, int],
    occupied: set[tuple[int, int]],
) -> tuple[list[tuple[int, int]], str, set[tuple[int, int]]]:
    h, w = shape
    dr, dc = DIRS[direction]
    current = list(coords)
    path_cells = set(current)
    while True:
        nxt = [(r + dr, c + dc) for r, c in current]
        blocked_by_edge = any(r < 0 or c < 0 or r >= h or c >= w for r, c in nxt)
        if blocked_by_edge:
            return current, "edge", path_cells
        if any((r, c) in occupied for r, c in nxt):
            return current, "block", path_cells
        current = nxt
        path_cells.update(current)


def _symmetry_color(coords: list[tuple[int, int]]) -> int:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    has_horizontal = np.array_equal(mask, mask[::-1, :])
    has_vertical = np.array_equal(mask, mask[:, ::-1])
    if has_horizontal and has_vertical:
        return YELLOW
    if has_horizontal:
        return BLUE
    if has_vertical:
        return GREEN
    return MAGENTA


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each grey object away from its aligned red marker, then recolor it by mirror symmetry.

    Input structure:
    - The background is black.
    - Foreground objects are 4-connected grey rectangle-like shapes that may contain a single-cell protrusion or missing corner.
    - Red single-pixel markers lie directly above, below, left, or right of the grey objects with a clear orthogonal corridor.

    Rule:
    - For each grey object, find the nearest aligned red marker by orthogonal gap from the object's boundary; the object moves one step at a time away from that marker.
    - Process objects in row-major order, letting each object slide until its next step would leave the grid or overlap an already stopped object; remove all red markers afterward.
    - Recolor each stopped object blue if it has horizontal reflection symmetry, green if it has vertical reflection symmetry, yellow if it has both, and magenta if it has neither.

    Key insight:
    - The red pixels act as repelling direction markers, while the final color depends only on the moved object's own reflection-symmetry class.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    objects = [_make_object(comp) for comp in _components_of_color(g, GREY)]
    objects = sorted(objects, key=lambda obj: (obj["bbox"][0], obj["bbox"][1]))
    markers = [(int(r), int(c)) for r, c in np.argwhere(g == RED)]

    assert objects
    assert len(markers) == len(objects)

    grey_mask = g == GREY
    assignments = _marker_assignments(objects, markers, grey_mask)
    assert assignments is not None

    occupied: set[tuple[int, int]] = set()
    stopped: list[list[tuple[int, int]]] = []
    for obj, (_marker_idx, move_dir) in zip(objects, assignments):
        final_coords, _reason, _path = _slide_coords(obj["coords"], move_dir, g.shape, occupied)
        stopped.append(final_coords)
        occupied.update(final_coords)

    out = np.full(g.shape, BLACK, dtype=int)
    for coords in stopped:
        color = _symmetry_color(coords)
        for r, c in coords:
            out[r, c] = color
    return out


def _rect(h: int, w: int) -> set[tuple[int, int]]:
    return {(r, c) for r in range(h) for c in range(w)}


def _normalize_shape(coords: set[tuple[int, int]]) -> list[tuple[int, int]]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, c0 = min(rows), min(cols)
    return sorted((r - r0, c - c0) for r, c in coords)


def _make_shape(kind: str, rng: random.Random) -> list[tuple[int, int]]:
    if kind == "both":
        h = rng.randint(2, 5)
        w = rng.randint(2, 5)
        return _normalize_shape(_rect(h, w))

    if kind == "horizontal":
        h = rng.choice((3, 5))
        w = rng.randint(2, 5)
        coords = _rect(h, w)
        mid_r = h // 2
        if rng.random() < 0.5:
            coords.add((mid_r, -1))
        else:
            coords.add((mid_r, w))
        return _normalize_shape(coords)

    if kind == "vertical":
        h = rng.randint(2, 5)
        w = rng.choice((3, 5))
        coords = _rect(h, w)
        mid_c = w // 2
        if rng.random() < 0.5:
            coords.add((-1, mid_c))
        else:
            coords.add((h, mid_c))
        return _normalize_shape(coords)

    h = rng.randint(3, 5)
    w = rng.randint(2, 5)
    coords = _rect(h, w)
    if rng.random() < 0.5:
        row_choices = [r for r in range(h) if r != h // 2]
        rr = rng.choice(row_choices)
        if rng.random() < 0.5:
            coords.add((rr, -1))
        else:
            coords.add((rr, w))
    else:
        col_choices = [c for c in range(w) if c != w // 2]
        cc = rng.choice(col_choices)
        if rng.random() < 0.5:
            coords.add((-1, cc))
        else:
            coords.add((h, cc))
    return _normalize_shape(coords)


def _contact_candidates(
    shape: list[tuple[int, int]],
    blocker_bbox: tuple[int, int, int, int],
    direction: str,
    grid_shape: tuple[int, int],
    rng: random.Random,
) -> tuple[int, int] | None:
    h, w = _shape_size(shape)
    gh, gw = grid_shape
    br0, bc0, br1, bc1 = blocker_bbox

    if direction == "up":
        final_r0 = br1 + 1
        if final_r0 + h > gh:
            return None
        lo = max(0, bc0 - w + 1)
        hi = min(gw - w, bc1)
        if lo > hi:
            return None
        return final_r0, rng.randint(lo, hi)

    if direction == "down":
        final_r0 = br0 - h
        if final_r0 < 0:
            return None
        lo = max(0, bc0 - w + 1)
        hi = min(gw - w, bc1)
        if lo > hi:
            return None
        return final_r0, rng.randint(lo, hi)

    if direction == "left":
        final_c0 = bc1 + 1
        if final_c0 + w > gw:
            return None
        lo = max(0, br0 - h + 1)
        hi = min(gh - h, br1)
        if lo > hi:
            return None
        return rng.randint(lo, hi), final_c0

    final_c0 = bc0 - w
    if final_c0 < 0:
        return None
    lo = max(0, br0 - h + 1)
    hi = min(gh - h, br1)
    if lo > hi:
        return None
    return rng.randint(lo, hi), final_c0


def _grid_shape_for_count(rng: random.Random, object_count: int) -> tuple[int, int]:
    lo, hi = GRID_SIZE_RANGE
    if object_count >= 4:
        lo = max(lo, 24)
    elif object_count == 3:
        lo = max(lo, 21)
    return rng.randint(lo, hi), rng.randint(lo, hi)


def _edge_anchor(shape: list[tuple[int, int]], direction: str, grid_shape: tuple[int, int], rng: random.Random) -> tuple[int, int]:
    h, w = _shape_size(shape)
    gh, gw = grid_shape
    if direction == "up":
        return 0, rng.randint(0, gw - w)
    if direction == "down":
        return gh - h, rng.randint(0, gw - w)
    if direction == "left":
        return rng.randint(0, gh - h), 0
    return rng.randint(0, gh - h), gw - w


def _max_reverse_shift(final_bbox: tuple[int, int, int, int], direction: str, grid_shape: tuple[int, int]) -> int:
    gh, gw = grid_shape
    r0, c0, r1, c1 = final_bbox
    if direction == "up":
        return gh - 2 - r1
    if direction == "down":
        return r0 - 1
    if direction == "left":
        return gw - 2 - c1
    return c0 - 1


def _place_object(
    rng: random.Random,
    shape: list[tuple[int, int]],
    direction: str,
    stop_mode: str,
    prev_key: tuple[int, int],
    grid_shape: tuple[int, int],
    initial_occupied: set[tuple[int, int]],
    path_reserved: set[tuple[int, int]],
    stopped_occupied: set[tuple[int, int]],
    final_objects: list[dict[str, object]],
) -> dict[str, object] | None:
    attempts = 240
    for _ in range(attempts):
        if stop_mode == "block":
            blocker = rng.choice(final_objects)
            anchor = _contact_candidates(shape, blocker["bbox"], direction, grid_shape, rng)
            if anchor is None:
                continue
            final_r0, final_c0 = anchor
        else:
            final_r0, final_c0 = _edge_anchor(shape, direction, grid_shape, rng)

        final_coords = _translate(shape, final_r0, final_c0)
        final_set = set(final_coords)
        if final_set & stopped_occupied:
            continue

        final_bbox = _bbox(final_coords)
        max_shift = _max_reverse_shift(final_bbox, direction, grid_shape)
        if max_shift < 1:
            continue

        shift_choices = list(range(1, max_shift + 1))
        rng.shuffle(shift_choices)
        for shift in shift_choices[:10]:
            dr, dc = DIRS[direction]
            initial_r0 = final_r0 - shift * dr
            initial_c0 = final_c0 - shift * dc
            initial_coords = _translate(shape, initial_r0, initial_c0)
            initial_set = set(initial_coords)
            initial_bbox = _bbox(initial_coords)
            if (initial_bbox[0], initial_bbox[1]) <= prev_key:
                continue
            if initial_set & initial_occupied:
                continue
            if initial_set & path_reserved:
                continue

            slid_coords, reason, path_cells = _slide_coords(initial_coords, direction, grid_shape, stopped_occupied)
            if set(slid_coords) != final_set:
                continue
            if reason != stop_mode:
                continue
            return {
                "direction": direction,
                "initial_coords": initial_coords,
                "initial_set": initial_set,
                "initial_bbox": initial_bbox,
                "final_coords": slid_coords,
                "final_set": final_set,
                "bbox": final_bbox,
                "path_cells": path_cells,
            }
    return None


def _marker_candidates_for_placement(obj: dict[str, object], move_dir: str) -> list[tuple[tuple[int, int], list[tuple[int, int]]]]:
    coords = obj["initial_coords"]
    made = _make_object(coords)
    candidates: list[tuple[tuple[int, int], list[tuple[int, int]]]] = []
    if move_dir == "up":
        for col, (_, bot_r) in made["col_bounds"].items():
            corridor = [(r, col) for r in range(bot_r + 1, bot_r + 4)]
            candidates.append(((bot_r, col), corridor))
    elif move_dir == "down":
        for col, (top_r, _) in made["col_bounds"].items():
            corridor = [(r, col) for r in range(top_r - 1, top_r - 4, -1)]
            candidates.append(((top_r, col), corridor))
    elif move_dir == "left":
        for row, (_, right_c) in made["row_bounds"].items():
            corridor = [(row, c) for c in range(right_c + 1, right_c + 4)]
            candidates.append(((row, right_c), corridor))
    else:
        for row, (left_c, _) in made["row_bounds"].items():
            corridor = [(row, c) for c in range(left_c - 1, left_c - 4, -1)]
            candidates.append(((row, left_c), corridor))
    return candidates


def _place_markers(
    rng: random.Random,
    objects: list[dict[str, object]],
    grid_shape: tuple[int, int],
) -> list[tuple[int, int]] | None:
    gh, gw = grid_shape
    grey_occupied = set().union(*(obj["initial_set"] for obj in objects))
    path_reserved = set().union(*(obj["path_cells"] for obj in objects))
    markers: list[tuple[int, int]] = []

    for obj in objects:
        options = _marker_candidates_for_placement(obj, obj["direction"])
        rng.shuffle(options)
        placed = None
        for boundary_cell, corridor in options:
            usable = []
            for cell in corridor:
                r, c = cell
                if not (0 <= r < gh and 0 <= c < gw):
                    break
                if cell in grey_occupied or cell in markers or cell in path_reserved:
                    break
                usable.append(cell)
            for marker in usable:
                if marker in grey_occupied or marker in markers or marker in path_reserved:
                    continue
                placed = marker
                break
            if placed is not None:
                break
        if placed is None:
            return None
        markers.append(placed)
    return markers


def _validate_grid(grid: np.ndarray, expected_objects: int) -> bool:
    grey_comps = [_make_object(comp) for comp in _components_of_color(grid, GREY)]
    grey_comps = sorted(grey_comps, key=lambda obj: (obj["bbox"][0], obj["bbox"][1]))
    markers = [(int(r), int(c)) for r, c in np.argwhere(grid == RED)]
    if len(grey_comps) != expected_objects or len(markers) != expected_objects:
        return False
    assignments = _marker_assignments(grey_comps, markers, grid == GREY)
    if assignments is None:
        return False
    out = main(grid)
    non_black = {int(v) for v in np.unique(out) if int(v) != BLACK}
    return 1 <= len(non_black) <= 4


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    object_count = rng.choice(OBJECT_COUNT_CHOICES)
    h, w = _grid_shape_for_count(rng, object_count)

    kinds = rng.sample(SYMMETRY_KINDS, object_count)
    directions = rng.sample(tuple(DIRS), object_count)
    stop_modes = ["edge"]
    for _ in range(1, object_count):
        stop_modes.append(rng.choice(("edge", "block")))
    if object_count >= 3 and all(mode == "edge" for mode in stop_modes[1:]):
        stop_modes[rng.randrange(1, object_count)] = "block"

    objects: list[dict[str, object]] = []
    initial_occupied: set[tuple[int, int]] = set()
    path_reserved: set[tuple[int, int]] = set()
    stopped_occupied: set[tuple[int, int]] = set()
    prev_key = (-10, -10)

    for idx in range(object_count):
        shape = _make_shape(kinds[idx], rng)
        direction = directions[idx]
        stop_mode = stop_modes[idx]
        if stop_mode == "block" and not objects:
            stop_mode = "edge"

        placed = _place_object(
            rng,
            shape,
            direction,
            stop_mode,
            prev_key,
            (h, w),
            initial_occupied,
            path_reserved,
            stopped_occupied,
            objects,
        )
        if placed is None:
            return None

        placed["symmetry_kind"] = kinds[idx]
        objects.append(placed)
        initial_occupied.update(placed["initial_set"])
        path_reserved.update(placed["path_cells"])
        stopped_occupied.update(placed["final_set"])
        prev_key = (placed["initial_bbox"][0], placed["initial_bbox"][1])

    markers = _place_markers(rng, objects, (h, w))
    if markers is None:
        return None

    grid = np.full((h, w), BLACK, dtype=int)
    for obj in objects:
        for r, c in obj["initial_coords"]:
            grid[r, c] = GREY
    for r, c in markers:
        if grid[r, c] != BLACK:
            return None
        grid[r, c] = RED

    if not _validate_grid(grid, object_count):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
