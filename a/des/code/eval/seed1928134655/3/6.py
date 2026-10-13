from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
OBJECT_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
DIR_TO_DELTA = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: set[tuple[int, int]] | list[tuple[int, int]]) -> tuple[int, int, int, int]:
    if not isinstance(cells, set):
        cells = set(cells)
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _rect_cells(r0: int, c0: int, h: int, w: int) -> set[tuple[int, int]]:
    return {(r, c) for r in range(r0, r0 + h) for c in range(c0, c0 + w)}


def _translate(cells: set[tuple[int, int]], dr: int, dc: int, steps: int) -> set[tuple[int, int]]:
    return {(r + dr * steps, c + dc * steps) for r, c in cells}


def _decode_arrow(
    rect_bbox: tuple[int, int, int, int], arrow_cells: set[tuple[int, int]]
) -> tuple[str, int] | None:
    if not arrow_cells:
        return None

    r0, c0, r1, c1 = rect_bbox
    distance = len(arrow_cells)
    shaft_len = distance - 2
    if not 2 <= shaft_len <= 5:
        return None

    if all(c > c1 for _r, c in arrow_cells):
        far_c = c1 + shaft_len
        for attach_r in range(r0, r1 + 1):
            expected = {(attach_r, c) for c in range(c1 + 1, far_c + 1)}
            expected.add((attach_r - 1, far_c))
            expected.add((attach_r + 1, far_c))
            if expected == arrow_cells:
                return "right", distance

    if all(c < c0 for _r, c in arrow_cells):
        far_c = c0 - shaft_len
        for attach_r in range(r0, r1 + 1):
            expected = {(attach_r, c) for c in range(far_c, c0)}
            expected.add((attach_r - 1, far_c))
            expected.add((attach_r + 1, far_c))
            if expected == arrow_cells:
                return "left", distance

    if all(r > r1 for r, _c in arrow_cells):
        far_r = r1 + shaft_len
        for attach_c in range(c0, c1 + 1):
            expected = {(r, attach_c) for r in range(r1 + 1, far_r + 1)}
            expected.add((far_r, attach_c - 1))
            expected.add((far_r, attach_c + 1))
            if expected == arrow_cells:
                return "down", distance

    if all(r < r0 for r, _c in arrow_cells):
        far_r = r0 - shaft_len
        for attach_c in range(c0, c1 + 1):
            expected = {(r, attach_c) for r in range(far_r, r0)}
            expected.add((far_r, attach_c - 1))
            expected.add((far_r, attach_c + 1))
            if expected == arrow_cells:
                return "up", distance

    return None


def _decode_component(comp: list[tuple[int, int]]) -> dict:
    comp_set = set(comp)
    min_r, min_c, max_r, max_c = _bbox(comp_set)
    candidates: list[tuple[int, tuple[int, int, int, int], str, int]] = []

    for h in (2, 3):
        for w in (2, 3):
            for r0 in range(min_r, max_r - h + 2):
                for c0 in range(min_c, max_c - w + 2):
                    rect = _rect_cells(r0, c0, h, w)
                    if not rect.issubset(comp_set):
                        continue
                    arrow = comp_set - rect
                    decoded = _decode_arrow((r0, c0, r0 + h - 1, c0 + w - 1), arrow)
                    if decoded is None:
                        continue
                    direction, distance = decoded
                    candidates.append((h * w, (r0, c0, r0 + h - 1, c0 + w - 1), direction, distance))

    assert candidates, "each colored component must contain one rectangle plus one T-arrow"
    best_area = max(area for area, _bbox0, _direction, _distance in candidates)
    best = [cand for cand in candidates if cand[0] == best_area]
    assert len(best) == 1, "object rectangle must be uniquely decodable"

    _area, (r0, c0, r1, c1), direction, distance = best[0]
    return {
        "bbox": (r0, c0, r1, c1),
        "cells": _rect_cells(r0, c0, r1 - r0 + 1, c1 - c0 + 1),
        "direction": direction,
        "distance": distance,
    }


def _can_place(cells: set[tuple[int, int]], wall_mask: np.ndarray) -> bool:
    h, w = wall_mask.shape
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if bool(wall_mask[r, c]):
            return False
    return True


def _actual_steps(
    cells: set[tuple[int, int]], direction: str, distance: int, wall_mask: np.ndarray
) -> int:
    dr, dc = DIR_TO_DELTA[direction]
    steps = 0
    for t in range(1, distance + 1):
        if not _can_place(_translate(cells, dr, dc, t), wall_mask):
            break
        steps = t
    return steps


def _render_scene(
    shape: tuple[int, int],
    wall_cells: set[tuple[int, int]],
    objects: list[dict],
) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for r, c in wall_cells:
        out[r, c] = GREY

    for obj in objects:
        dr, dc = DIR_TO_DELTA[obj["direction"]]
        cells = obj["cells"]
        steps = obj["actual_steps"]
        color = obj["color"]

        for t in range(steps):
            for r, c in _translate(cells, dr, dc, t):
                out[r, c] = GREY
        for r, c in _translate(cells, dr, dc, steps):
            out[r, c] = color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each colored rectangle along the direction encoded by its same-colored T-marker.

    Input structure
    - Background is black.
    - Each non-grey color forms exactly one filled 2x2..3x3 rectangle plus one adjacent T-shaped arrow.
    - Grey pixels are walls.

    Rule
    - The arrow lies on one side of the rectangle; that side is the movement direction.
    - Arrow pixel count equals the requested slide distance.
    - Walls and the grid boundary stop the moving rectangle one cell early.
    - The arrow disappears; every occupied intermediate rectangle position becomes grey trail.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    wall_mask = g == GREY
    wall_cells = {(int(r), int(c)) for r, c in np.argwhere(wall_mask)}

    objects: list[dict] = []
    for color in sorted(set(int(v) for v in np.unique(g)) - {BG, GREY}):
        for comp in _components_of_color(g, color):
            decoded = _decode_component(comp)
            actual_steps = _actual_steps(decoded["cells"], decoded["direction"], decoded["distance"], wall_mask)
            objects.append(
                {
                    "color": color,
                    "cells": decoded["cells"],
                    "direction": decoded["direction"],
                    "distance": decoded["distance"],
                    "actual_steps": actual_steps,
                }
            )

    out = _render_scene((h, w), wall_cells, objects)
    return out


GRID_H_RANGE = (15, 23)
GRID_W_RANGE = (15, 23)
OBJECT_COUNT_CHOICES = [2, 3, 4]
MAX_SCENE_ATTEMPTS = 400
MAX_OBJECT_ATTEMPTS = 300
MAX_WALL_ATTEMPTS = 120


def _sample_modes(count: int) -> list[str]:
    if count == 2:
        modes = list(random.choice([("free", "wall"), ("free", "boundary"), ("wall", "boundary")]))
    else:
        modes = ["free", "wall", "boundary"]
        while len(modes) < count:
            modes.append(random.choice(["free", "wall", "boundary"]))
    random.shuffle(modes)
    return modes


def _sample_directions(count: int) -> list[str]:
    dirs = ["up", "down", "left", "right"]
    random.shuffle(dirs)
    out = dirs[: min(count, 4)]
    while len(out) < count:
        out.append(random.choice(dirs))
    random.shuffle(out)
    return out


def _make_arrow(
    r0: int, c0: int, h: int, w: int, direction: str, attach: int, shaft_len: int
) -> set[tuple[int, int]]:
    r1 = r0 + h - 1
    c1 = c0 + w - 1

    if direction == "right":
        attach_r = r0 + attach
        far_c = c1 + shaft_len
        return {(attach_r, c) for c in range(c1 + 1, far_c + 1)} | {
            (attach_r - 1, far_c),
            (attach_r + 1, far_c),
        }
    if direction == "left":
        attach_r = r0 + attach
        far_c = c0 - shaft_len
        return {(attach_r, c) for c in range(far_c, c0)} | {
            (attach_r - 1, far_c),
            (attach_r + 1, far_c),
        }
    if direction == "down":
        attach_c = c0 + attach
        far_r = r1 + shaft_len
        return {(r, attach_c) for r in range(r1 + 1, far_r + 1)} | {
            (far_r, attach_c - 1),
            (far_r, attach_c + 1),
        }
    attach_c = c0 + attach
    far_r = r0 - shaft_len
    return {(r, attach_c) for r in range(far_r, r0)} | {
        (far_r, attach_c - 1),
        (far_r, attach_c + 1),
    }


def _position_ranges(
    H: int, W: int, h: int, w: int, direction: str, attach: int, mode: str, distance: int
) -> list[tuple[int, int]]:
    positions: list[tuple[int, int]] = []

    if direction in ("right", "left"):
        row_lo = max(0, 1 - attach)
        row_hi = min(H - h, H - 2 - attach)
        if row_lo > row_hi:
            return []

        if direction == "right":
            if mode == "boundary":
                for clearance in (distance - 2, distance - 1):
                    c0 = W - w - clearance
                    if 0 <= c0 <= W - w:
                        for r0 in range(row_lo, row_hi + 1):
                            positions.append((r0, c0))
            else:
                col_lo, col_hi = 0, W - w - distance
                if col_lo <= col_hi:
                    for r0 in range(row_lo, row_hi + 1):
                        for c0 in range(col_lo, col_hi + 1):
                            positions.append((r0, c0))
        else:
            if mode == "boundary":
                for clearance in (distance - 2, distance - 1):
                    c0 = clearance
                    if 0 <= c0 <= W - w:
                        for r0 in range(row_lo, row_hi + 1):
                            positions.append((r0, c0))
            else:
                col_lo, col_hi = distance, W - w
                if col_lo <= col_hi:
                    for r0 in range(row_lo, row_hi + 1):
                        for c0 in range(col_lo, col_hi + 1):
                            positions.append((r0, c0))
        return positions

    col_lo = max(0, 1 - attach)
    col_hi = min(W - w, W - 2 - attach)
    if col_lo > col_hi:
        return []

    if direction == "down":
        if mode == "boundary":
            for clearance in (distance - 2, distance - 1):
                r0 = H - h - clearance
                if 0 <= r0 <= H - h:
                    for c0 in range(col_lo, col_hi + 1):
                        positions.append((r0, c0))
        else:
            row_lo, row_hi = 0, H - h - distance
            if row_lo <= row_hi:
                for r0 in range(row_lo, row_hi + 1):
                    for c0 in range(col_lo, col_hi + 1):
                        positions.append((r0, c0))
    else:
        if mode == "boundary":
            for clearance in (distance - 2, distance - 1):
                r0 = clearance
                if 0 <= r0 <= H - h:
                    for c0 in range(col_lo, col_hi + 1):
                        positions.append((r0, c0))
        else:
            row_lo, row_hi = distance, H - h
            if row_lo <= row_hi:
                for r0 in range(row_lo, row_hi + 1):
                    for c0 in range(col_lo, col_hi + 1):
                        positions.append((r0, c0))
    return positions


def _wall_segment_for_stop(
    H: int,
    W: int,
    object_cells: set[tuple[int, int]],
    arrow_cells: set[tuple[int, int]],
    direction: str,
    stop_steps: int,
    forbidden_input: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
) -> set[tuple[int, int]] | None:
    r0, c0, r1, c1 = _bbox(object_cells)
    arrow_at = set(arrow_cells)

    if direction == "right":
        wall_col = c1 + stop_steps + 1
        block_coords = [(r, wall_col) for r in range(r0, r1 + 1) if (r, wall_col) not in arrow_at]
        random.shuffle(block_coords)
        for block_r, _wall_c in block_coords:
            lengths = list(range(2, min(6, H) + 1))
            random.shuffle(lengths)
            for length in lengths:
                starts = list(range(max(0, block_r - length + 1), min(block_r, H - length) + 1))
                random.shuffle(starts)
                for start in starts:
                    seg = {(rr, wall_col) for rr in range(start, start + length)}
                    if seg & forbidden_input or seg & reserved:
                        continue
                    return seg
        return None

    if direction == "left":
        wall_col = c0 - stop_steps - 1
        block_coords = [(r, wall_col) for r in range(r0, r1 + 1) if (r, wall_col) not in arrow_at]
        random.shuffle(block_coords)
        for block_r, _wall_c in block_coords:
            lengths = list(range(2, min(6, H) + 1))
            random.shuffle(lengths)
            for length in lengths:
                starts = list(range(max(0, block_r - length + 1), min(block_r, H - length) + 1))
                random.shuffle(starts)
                for start in starts:
                    seg = {(rr, wall_col) for rr in range(start, start + length)}
                    if seg & forbidden_input or seg & reserved:
                        continue
                    return seg
        return None

    if direction == "down":
        wall_row = r1 + stop_steps + 1
        block_coords = [(wall_row, c) for c in range(c0, c1 + 1) if (wall_row, c) not in arrow_at]
        random.shuffle(block_coords)
        for _wall_r, block_c in block_coords:
            lengths = list(range(2, min(6, W) + 1))
            random.shuffle(lengths)
            for length in lengths:
                starts = list(range(max(0, block_c - length + 1), min(block_c, W - length) + 1))
                random.shuffle(starts)
                for start in starts:
                    seg = {(wall_row, cc) for cc in range(start, start + length)}
                    if seg & forbidden_input or seg & reserved:
                        continue
                    return seg
        return None

    wall_row = r0 - stop_steps - 1
    block_coords = [(wall_row, c) for c in range(c0, c1 + 1) if (wall_row, c) not in arrow_at]
    random.shuffle(block_coords)
    for _wall_r, block_c in block_coords:
        lengths = list(range(2, min(6, W) + 1))
        random.shuffle(lengths)
        for length in lengths:
            starts = list(range(max(0, block_c - length + 1), min(block_c, W - length) + 1))
            random.shuffle(starts)
            for start in starts:
                seg = {(wall_row, cc) for cc in range(start, start + length)}
                if seg & forbidden_input or seg & reserved:
                    continue
                return seg
    return None


def _extra_wall_segment(
    H: int,
    W: int,
    forbidden_input: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
) -> set[tuple[int, int]] | None:
    for _ in range(MAX_WALL_ATTEMPTS):
        vertical = bool(random.getrandbits(1))
        length = random.randint(2, 6)
        if vertical:
            if H < length:
                continue
            c = random.randint(0, W - 1)
            r0 = random.randint(0, H - length)
            seg = {(r, c) for r in range(r0, r0 + length)}
        else:
            if W < length:
                continue
            r = random.randint(0, H - 1)
            c0 = random.randint(0, W - length)
            seg = {(r, c) for c in range(c0, c0 + length)}
        if seg & forbidden_input or seg & reserved:
            continue
        return seg
    return None


def _build_scene() -> tuple[np.ndarray, np.ndarray] | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)
    count = random.choice(OBJECT_COUNT_CHOICES)

    colors = random.sample(OBJECT_COLORS, count)
    modes = _sample_modes(count)
    directions = _sample_directions(count)

    wall_cells: set[tuple[int, int]] = set()
    occupied_input: set[tuple[int, int]] = set()
    reserved: set[tuple[int, int]] = set()
    objects: list[dict] = []

    for idx in range(count):
        color = colors[idx]
        mode = modes[idx]
        direction = directions[idx]
        placed = False

        for _ in range(MAX_OBJECT_ATTEMPTS):
            h = random.choice([2, 3])
            w = random.choice([2, 3])
            shaft_len = random.randint(2, 5)
            distance = shaft_len + 2
            attach = random.randrange(h if direction in ("left", "right") else w)

            positions = _position_ranges(H, W, h, w, direction, attach, mode, distance)
            if not positions:
                continue
            r0, c0 = random.choice(positions)
            object_cells = _rect_cells(r0, c0, h, w)
            arrow_cells = _make_arrow(r0, c0, h, w, direction, attach, shaft_len)

            candidate_input = object_cells | arrow_cells
            if candidate_input & occupied_input or candidate_input & reserved:
                continue

            wall_mask = np.zeros((H, W), dtype=bool)
            for wr, wc in wall_cells:
                wall_mask[wr, wc] = True

            if mode == "wall":
                free_steps = _actual_steps(object_cells, direction, distance, wall_mask)
                if free_steps != distance:
                    continue

                stops = list(range(1, distance))
                random.shuffle(stops)
                wall_seg = None
                actual_steps = None
                forbidden_input = occupied_input | candidate_input | wall_cells
                for stop_steps in stops:
                    candidate_wall = _wall_segment_for_stop(
                        H,
                        W,
                        object_cells,
                        arrow_cells,
                        direction,
                        stop_steps,
                        forbidden_input,
                        reserved,
                    )
                    if candidate_wall is None:
                        continue
                    trial_mask = wall_mask.copy()
                    for wr, wc in candidate_wall:
                        trial_mask[wr, wc] = True
                    trial_steps = _actual_steps(object_cells, direction, distance, trial_mask)
                    if trial_steps != stop_steps:
                        continue
                    wall_seg = candidate_wall
                    actual_steps = trial_steps
                    break
                if wall_seg is None or actual_steps is None:
                    continue
            else:
                wall_seg = set()
                actual_steps = _actual_steps(object_cells, direction, distance, wall_mask)
                if mode == "free" and actual_steps != distance:
                    continue
                if mode == "boundary" and actual_steps not in (distance - 2, distance - 1):
                    continue

            dr, dc = DIR_TO_DELTA[direction]
            dynamic_zone: set[tuple[int, int]] = set()
            for t in range(actual_steps + 1):
                dynamic_zone |= _translate(object_cells, dr, dc, t)

            if dynamic_zone & reserved:
                continue
            if wall_seg & occupied_input or wall_seg & reserved:
                continue

            wall_cells |= wall_seg
            occupied_input |= candidate_input | wall_seg
            reserved |= dynamic_zone | wall_seg
            objects.append(
                {
                    "color": color,
                    "cells": object_cells,
                    "arrow": arrow_cells,
                    "direction": direction,
                    "distance": distance,
                    "actual_steps": actual_steps,
                }
            )
            placed = True
            break

        if not placed:
            return None

    extra_walls = random.randint(0, 2)
    for _ in range(extra_walls):
        seg = _extra_wall_segment(H, W, occupied_input | wall_cells, reserved)
        if seg is None:
            continue
        wall_cells |= seg
        occupied_input |= seg
        reserved |= seg

    inp = np.full((H, W), BG, dtype=int)
    for r, c in wall_cells:
        inp[r, c] = GREY
    for obj in objects:
        for r, c in obj["cells"] | obj["arrow"]:
            inp[r, c] = obj["color"]

    expected = _render_scene((H, W), wall_cells, objects)
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_SCENE_ATTEMPTS):
        built = _build_scene()
        if built is None:
            continue
        inp, expected = built
        actual = main(inp)
        if not np.array_equal(actual, expected):
            continue

        colors = sorted(set(int(v) for v in np.unique(inp)) - {BG, GREY})
        if len(colors) < 2:
            continue
        if int(np.count_nonzero(inp == GREY)) == 0:
            continue
        return inp
    raise RuntimeError(f"generate_input failed after {MAX_SCENE_ATTEMPTS} attempts")
