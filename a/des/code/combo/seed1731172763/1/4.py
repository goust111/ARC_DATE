import random
from typing import Iterable

import numpy as np


BG = 8
GREY = 5
YELLOW = 4
RED = 2

# The prompt leaves the cardinal mapping of the 2x2 missing corner implicit.
# We fix it visibly and consistently in clockwise order:
# missing TL -> up, missing TR -> right, missing BR -> down, missing BL -> left.
DIRS = ("up", "right", "down", "left")
DIR_TO_MISSING = {
    "up": (0, 0),
    "right": (0, 1),
    "down": (1, 1),
    "left": (1, 0),
}
MISSING_TO_DIR = {value: key for key, value in DIR_TO_MISSING.items()}

OBJECT_COLORS = (1, 3, 6, 7, 9)

GRID_H_RANGE = (16, 24)
GRID_W_RANGE = (16, 24)
OUTER_H_RANGE = (5, 8)
OUTER_W_RANGE = (5, 8)
MAX_GENERATE_ATTEMPTS = 400
PLACE_ATTEMPTS = 160
SEPARATION_PAD = 1


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    pts = list(points)
    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    return min(rows), min(cols), max(rows), max(cols)


def _is_connected(points: Iterable[tuple[int, int]]) -> bool:
    pts = set(points)
    if not pts:
        return False
    stack = [next(iter(pts))]
    seen = {stack[0]}
    while stack:
        r, c = stack.pop()
        for nbr in _neighbors4(r, c):
            if nbr in pts and nbr not in seen:
                seen.add(nbr)
                stack.append(nbr)
    return len(seen) == len(pts)


def _shift(points: Iterable[tuple[int, int]], dr: int, dc: int) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for r, c in points]


def _u_cells(height: int, width: int, open_dir: str) -> list[tuple[int, int]]:
    cells = set()
    for r in range(height):
        cells.add((r, 0))
        cells.add((r, width - 1))
    for c in range(width):
        cells.add((0, c))
        cells.add((height - 1, c))

    if open_dir == "up":
        for c in range(1, width - 1):
            cells.discard((0, c))
    elif open_dir == "down":
        for c in range(1, width - 1):
            cells.discard((height - 1, c))
    elif open_dir == "left":
        for r in range(1, height - 1):
            cells.discard((r, 0))
    else:
        assert open_dir == "right"
        for r in range(1, height - 1):
            cells.discard((r, width - 1))

    return sorted(cells)


def _l_cells(direction: str) -> list[tuple[int, int]]:
    missing = DIR_TO_MISSING[direction]
    return [(r, c) for r in range(2) for c in range(2) if (r, c) != missing]


def _container_open_dir(points: list[tuple[int, int]]) -> str:
    r0, c0, r1, c1 = _bbox(points)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    normalized = {(r - r0, c - c0) for r, c in points}
    matches = [direction for direction in DIRS if normalized == set(_u_cells(height, width, direction))]
    assert len(matches) == 1, "expected one exact U-shaped container orientation"
    return matches[0]


def _extract_instances(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    instances: list[dict] = []
    for container_pts in _components(g == GREY):
        open_dir = _container_open_dir(container_pts)
        r0, c0, r1, c1 = _bbox(container_pts)
        colored: list[tuple[int, int, int]] = []
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                value = int(g[r, c])
                if value not in (BG, GREY):
                    colored.append((r, c, value))

        assert len(colored) == 3, "each container must hold exactly one 3-cell object"
        object_colors = {value for _, _, value in colored}
        assert len(object_colors) == 1, "the L object must be monochrome"
        object_color = next(iter(object_colors))
        object_pts = [(r, c) for r, c, _ in colored]
        assert _is_connected(object_pts), "the object must be 4-connected"

        or0, oc0, or1, oc1 = _bbox(object_pts)
        assert (or1 - or0 + 1, oc1 - oc0 + 1) == (2, 2), "the object must occupy a 2x2 box"
        local = {(r - or0, c - oc0) for r, c in object_pts}
        missing = [(r, c) for r in range(2) for c in range(2) if (r, c) not in local]
        assert len(missing) == 1, "the object must be an L triomino"
        orientation = MISSING_TO_DIR[missing[0]]

        instances.append(
            {
                "container_bbox": (r0, c0, r1, c1),
                "open_dir": open_dir,
                "object_color": object_color,
                "object_pts": object_pts,
                "object_bbox": (or0, oc0, or1, oc1),
                "orientation": orientation,
            }
        )

    return instances


def _edge_shift(obj_bbox: tuple[int, int, int, int], shape: tuple[int, int], direction: str) -> tuple[int, int]:
    r0, c0, r1, c1 = obj_bbox
    h, w = shape
    if direction == "up":
        return -r0, 0
    if direction == "down":
        return h - 1 - r1, 0
    if direction == "left":
        return 0, -c0
    assert direction == "right"
    return 0, w - 1 - c1


def _exit_shift(
    obj_bbox: tuple[int, int, int, int],
    container_bbox: tuple[int, int, int, int],
    direction: str,
) -> tuple[int, int]:
    or0, oc0, or1, oc1 = obj_bbox
    cr0, cc0, cr1, cc1 = container_bbox
    if direction == "up":
        return cr0 - or0, 0
    if direction == "down":
        return cr1 - or1, 0
    if direction == "left":
        return 0, cc0 - oc0
    assert direction == "right"
    return 0, cc1 - oc1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each container's L object only when its missing-corner orientation matches the U opening.

    The clockwise missing-corner convention is:
    TL -> up, TR -> right, BR -> down, BL -> left.
    Matching objects leave a yellow copy at the opening and slide to the grid edge.
    Non-matching objects stay in place and turn red.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    out = g.copy()
    instances = _extract_instances(g)

    for instance in instances:
        open_dir = instance["open_dir"]
        object_pts = instance["object_pts"]
        object_color = instance["object_color"]
        if instance["orientation"] != open_dir:
            for r, c in object_pts:
                out[r, c] = RED
            continue

        for r, c in object_pts:
            out[r, c] = BG

        exit_dr, exit_dc = _exit_shift(instance["object_bbox"], instance["container_bbox"], open_dir)
        for r, c in _shift(object_pts, exit_dr, exit_dc):
            if out[r, c] != GREY:
                out[r, c] = YELLOW

        final_dr, final_dc = _edge_shift(instance["object_bbox"], g.shape, open_dir)
        for r, c in _shift(object_pts, final_dr, final_dc):
            if out[r, c] != GREY:
                out[r, c] = object_color

    return out


# --------------- generator ---------------


def _balanced_match_flags(count: int, rng: random.Random) -> list[bool]:
    if count == 1:
        return [rng.choice([False, True])]
    n_match = count // 2 if rng.random() < 0.5 else (count + 1) // 2
    flags = [True] * n_match + [False] * (count - n_match)
    rng.shuffle(flags)
    return flags


def _expand_cells(cells: Iterable[tuple[int, int]], shape: tuple[int, int], pad: int) -> set[tuple[int, int]]:
    h, w = shape
    expanded: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in range(-pad, pad + 1):
            for dc in range(-pad, pad + 1):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    expanded.add((nr, nc))
    return expanded


def _corridor_cells(
    obj_bbox: tuple[int, int, int, int],
    direction: str,
    shape: tuple[int, int],
) -> set[tuple[int, int]]:
    h, w = shape
    r0, c0, r1, c1 = obj_bbox
    cells: set[tuple[int, int]] = set()
    if direction == "up":
        row_range = range(0, r1 + 1)
        col_range = range(c0, c1 + 1)
    elif direction == "down":
        row_range = range(r0, h)
        col_range = range(c0, c1 + 1)
    elif direction == "left":
        row_range = range(r0, r1 + 1)
        col_range = range(0, c1 + 1)
    else:
        assert direction == "right"
        row_range = range(r0, r1 + 1)
        col_range = range(c0, w)

    for r in row_range:
        for c in col_range:
            cells.add((r, c))
    return cells


def _candidate_cells(spec: dict, top: int, left: int, shape: tuple[int, int]) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    container = set(_shift(_u_cells(spec["height"], spec["width"], spec["open_dir"]), top, left))
    object_pts = set(_shift(_l_cells(spec["orientation"]), top + spec["obj_r"], left + spec["obj_c"]))
    reserved = set(container) | set(object_pts)
    if spec["match"]:
        obj_bbox = (top + spec["obj_r"], left + spec["obj_c"], top + spec["obj_r"] + 1, left + spec["obj_c"] + 1)
        reserved |= _corridor_cells(obj_bbox, spec["open_dir"], shape)
    return container, reserved


def _random_spec(color: int, match: bool, rng: random.Random) -> dict:
    height = rng.randint(*OUTER_H_RANGE)
    width = rng.randint(*OUTER_W_RANGE)
    open_dir = rng.choice(DIRS)
    orientation = open_dir if match else rng.choice([direction for direction in DIRS if direction != open_dir])
    obj_r = rng.randint(1, height - 3)
    obj_c = rng.randint(1, width - 3)
    return {
        "height": height,
        "width": width,
        "open_dir": open_dir,
        "orientation": orientation,
        "obj_r": obj_r,
        "obj_c": obj_c,
        "color": color,
        "match": match,
    }


def _place_specs(specs: list[dict], shape: tuple[int, int], rng: random.Random) -> list[dict] | None:
    h, w = shape
    blocked: set[tuple[int, int]] = set()
    placed: list[dict] = []

    ordered = sorted(specs, key=lambda spec: (not spec["match"], -(spec["height"] * spec["width"])))
    for spec in ordered:
        row_min = 1
        row_max = h - spec["height"] - 1
        col_min = 1
        col_max = w - spec["width"] - 1
        if row_min > row_max or col_min > col_max:
            return None

        success = None
        for _ in range(PLACE_ATTEMPTS):
            top = rng.randint(row_min, row_max)
            left = rng.randint(col_min, col_max)
            container, reserved = _candidate_cells(spec, top, left, shape)
            padded = _expand_cells(reserved, shape, SEPARATION_PAD)
            if padded & blocked:
                continue
            success = {
                **spec,
                "top": top,
                "left": left,
                "container_cells": container,
                "reserved_cells": reserved,
            }
            blocked |= padded
            placed.append(success)
            break

        if success is None:
            return None

    return placed


def _draw_grid(placed_specs: list[dict], shape: tuple[int, int]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for spec in placed_specs:
        top = spec["top"]
        left = spec["left"]
        for r, c in spec["container_cells"]:
            grid[r, c] = GREY
        for dr, dc in _l_cells(spec["orientation"]):
            grid[top + spec["obj_r"] + dr, left + spec["obj_c"] + dc] = spec["color"]
    return grid


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    area = h * w
    max_count = min(4, max(1, area // 120 + 1))
    if max_count == 1:
        count = 1
    elif rng.random() < 0.65:
        count = rng.randint(2, max_count)
    else:
        count = 1

    match_flags = _balanced_match_flags(count, rng)
    colors = rng.sample(list(OBJECT_COLORS), count)
    specs = [_random_spec(color, match, rng) for color, match in zip(colors, match_flags)]
    placed_specs = _place_specs(specs, (h, w), rng)
    if placed_specs is None:
        return None

    grid = _draw_grid(placed_specs, (h, w))

    instances = _extract_instances(grid)
    if len(instances) != count:
        return None

    # Reject accidental degenerate scenes where the output would not visibly branch.
    out = main(grid)
    if count > 1:
        saw_moved = np.any(out == YELLOW)
        saw_locked = np.any(out == RED)
        if not (saw_moved and saw_locked):
            return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATE_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
