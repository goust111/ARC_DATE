from __future__ import annotations

from collections import deque
from itertools import combinations
import random

import numpy as np


BG = 0
GREEN = 3
YELLOW = 4
GREY = 5
OBJECT_COLORS = (1, 2, 6, 7, 8, 9)
SPECIAL_COLORS = {BG, GREEN, YELLOW, GREY}

GRID_MIN = 14
GRID_MAX = 19
DIST_MIN = 8
DIST_MAX = 13
GEN_ATTEMPTS = 400


def _sign(delta: int) -> int:
    return int((delta > 0) - (delta < 0))


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _single_coord(grid: np.ndarray, color: int) -> tuple[int, int]:
    coords = np.argwhere(grid == int(color))
    assert coords.shape == (1, 2), f"expected exactly one marker of color {color}"
    return int(coords[0, 0]), int(coords[0, 1])


def _find_objects(grid: np.ndarray) -> tuple[list[dict], dict[tuple[int, int], int]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    objects: list[dict] = []
    cell_to_obj: dict[tuple[int, int], int] = {}

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color in SPECIAL_COLORS or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            colors = {color}
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if not _in_bounds(grid.shape, nr, nc) or seen[nr, nc]:
                        continue
                    ncolor = int(grid[nr, nc])
                    if ncolor in SPECIAL_COLORS:
                        continue
                    seen[nr, nc] = True
                    colors.add(ncolor)
                    q.append((nr, nc))
            assert len(colors) == 1, "foreground objects must be monochromatic"
            obj_id = len(objects)
            obj = {
                "color": color,
                "coords": tuple(coords),
                "area": len(coords),
            }
            objects.append(obj)
            for cell in coords:
                cell_to_obj[cell] = obj_id
    return objects, cell_to_obj


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide the green marker toward the yellow marker, picking up small objects and stopping before blockers.

    Input structure:
    - The grid uses black as background and contains exactly one green marker pixel and one yellow marker pixel.
    - The two markers lie on the same row, the same column, or a 45-degree diagonal.
    - Between them are separated 4-connected foreground objects in non-marker colors; each object is either small (area <= 3) or large (area >= 4).

    Rule:
    - Move the green marker one step at a time toward the yellow marker using the normalized row/column direction.
    - Entering background advances normally; entering a small object erases that whole object, records its color as cargo, and still advances; encountering a large object or the yellow marker stops the marker on the previous cell.
    - Every traversed cell becomes grey trail, then the stopped green marker is repainted and the picked objects are compressed into single pixels placed behind it in pickup order.

    Key insight:
    - Object area determines interaction role rather than motion: small components are absorbed and later redeployed, while large components behave as hard blockers that fix the final cargo layout.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    sr, sc = _single_coord(g, GREEN)
    yr, yc = _single_coord(g, YELLOW)
    dr = _sign(yr - sr)
    dc = _sign(yc - sc)
    assert (dr, dc) != (0, 0), "markers must be distinct"
    assert sr == yr or sc == yc or abs(yr - sr) == abs(yc - sc), "markers must be aligned"

    work = g.copy()
    objects, cell_to_obj = _find_objects(work)
    cargo: list[int] = []
    positions = [(sr, sc)]
    cur_r, cur_c = sr, sc

    while True:
        nr = cur_r + dr
        nc = cur_c + dc
        if not _in_bounds(work.shape, nr, nc):
            break
        if (nr, nc) == (yr, yc):
            break

        obj_id = cell_to_obj.get((nr, nc))
        if obj_id is not None:
            obj = objects[obj_id]
            if obj["area"] >= 4:
                break
            cargo.append(int(obj["color"]))
            for rr, cc in obj["coords"]:
                work[rr, cc] = BG
                cell_to_obj.pop((rr, cc), None)
        else:
            assert int(work[nr, nc]) == BG, "only background or objects may lie on the path"

        cur_r, cur_c = nr, nc
        positions.append((cur_r, cur_c))

    out = work.copy()
    for rr, cc in positions[:-1]:
        out[rr, cc] = GREY

    final_r, final_c = positions[-1]
    out[final_r, final_c] = GREEN
    out[yr, yc] = YELLOW

    for idx, color in enumerate(cargo, start=1):
        rr = final_r - dr * idx
        cc = final_c - dc * idx
        if _in_bounds(out.shape, rr, cc):
            out[rr, cc] = int(color)

    return out


def _line_cells(start: tuple[int, int], step: tuple[int, int], distance: int) -> list[tuple[int, int]]:
    sr, sc = start
    dr, dc = step
    return [(sr + dr * i, sc + dc * i) for i in range(distance + 1)]


def _orientation_kind(step: tuple[int, int]) -> str:
    dr, dc = step
    if dr == 0:
        return "horizontal"
    if dc == 0:
        return "vertical"
    return "diagonal"


def _small_templates(kind: str) -> list[tuple[tuple[int, int], ...]]:
    if kind == "horizontal":
        return [
            ((0, 0),),
            ((0, 0), (1, 0)),
            ((0, 0), (-1, 0)),
            ((0, 0), (1, 0), (-1, 0)),
        ]
    if kind == "vertical":
        return [
            ((0, 0),),
            ((0, 0), (0, 1)),
            ((0, 0), (0, -1)),
            ((0, 0), (0, 1), (0, -1)),
        ]
    return [
        ((0, 0),),
        ((0, 0), (1, 0)),
        ((0, 0), (-1, 0)),
        ((0, 0), (0, 1)),
        ((0, 0), (0, -1)),
        ((0, 0), (1, 0), (0, 1)),
        ((0, 0), (1, 0), (0, -1)),
        ((0, 0), (-1, 0), (0, 1)),
        ((0, 0), (-1, 0), (0, -1)),
    ]


def _big_templates(step: tuple[int, int]) -> list[tuple[tuple[int, int], ...]]:
    dr, dc = step
    kind = _orientation_kind(step)
    if kind == "horizontal":
        return [
            ((0, 0), (0, dc), (1, 0), (1, dc)),
            ((0, 0), (0, dc), (-1, 0), (-1, dc)),
            ((0, 0), (0, dc), (1, 0), (1, dc), (2, 0), (2, dc)),
            ((0, 0), (0, dc), (-1, 0), (-1, dc), (-2, 0), (-2, dc)),
        ]
    if kind == "vertical":
        return [
            ((0, 0), (dr, 0), (0, 1), (dr, 1)),
            ((0, 0), (dr, 0), (0, -1), (dr, -1)),
            ((0, 0), (dr, 0), (0, 1), (dr, 1), (0, 2), (dr, 2)),
            ((0, 0), (dr, 0), (0, -1), (dr, -1), (0, -2), (dr, -2)),
        ]
    return [
        ((0, 0), (dr, 0), (0, dc), (dr, dc)),
        ((0, 0), (dr, 0), (0, dc), (dr, dc), (2 * dr, dc)),
        ((0, 0), (dr, 0), (0, dc), (dr, dc), (dr, 2 * dc)),
    ]


def _abs_coords(anchor: tuple[int, int], offsets: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    ar, ac = anchor
    return tuple((ar + dr, ac + dc) for dr, dc in offsets)


def _cells_ok(
    shape: tuple[int, int],
    cells: tuple[tuple[int, int], ...],
    occupied: set[tuple[int, int]],
) -> bool:
    unique = set(cells)
    if len(unique) != len(cells):
        return False
    for r, c in cells:
        if not _in_bounds(shape, r, c) or (r, c) in occupied:
            return False
    return True


def _sample_positions(distance: int, small_count: int, has_blocker: bool) -> tuple[list[int], int | None]:
    if has_blocker:
        options: list[tuple[list[int], int]] = []
        for block_idx in range(4, distance):
            for picks in combinations(range(2, block_idx), small_count):
                if all(b - a >= 2 for a, b in zip(picks, picks[1:])) and block_idx - picks[-1] >= 2:
                    options.append((list(picks), block_idx))
        assert options, "no valid blocker layouts"
        smalls, blocker = random.choice(options)
        return smalls, blocker

    options = [
        list(picks)
        for picks in combinations(range(2, distance), small_count)
        if all(b - a >= 2 for a, b in zip(picks, picks[1:]))
    ]
    assert options, "no valid non-blocker layouts"
    return random.choice(options), None


def _sample_markers(
    shape: tuple[int, int],
    step: tuple[int, int],
    distance: int,
) -> tuple[tuple[int, int], tuple[int, int]]:
    h, w = shape
    dr, dc = step
    margin = 2

    if dr > 0:
        r0_lo, r0_hi = margin, h - margin - 1 - distance
    elif dr < 0:
        r0_lo, r0_hi = margin + distance, h - margin - 1
    else:
        r0_lo, r0_hi = margin, h - margin - 1

    if dc > 0:
        c0_lo, c0_hi = margin, w - margin - 1 - distance
    elif dc < 0:
        c0_lo, c0_hi = margin + distance, w - margin - 1
    else:
        c0_lo, c0_hi = margin, w - margin - 1

    assert r0_lo <= r0_hi and c0_lo <= c0_hi
    sr = random.randint(r0_lo, r0_hi)
    sc = random.randint(c0_lo, c0_hi)
    yr = sr + dr * distance
    yc = sc + dc * distance
    return (sr, sc), (yr, yc)


def _expected_from_meta(meta: dict) -> np.ndarray:
    grid = np.array(meta["input"], copy=True)
    sr, sc = meta["start"]
    yr, yc = meta["yellow"]
    dr, dc = meta["step"]
    final_idx = meta["blocker_index"] - 1 if meta["blocker_index"] is not None else meta["distance"] - 1
    line = _line_cells((sr, sc), (dr, dc), meta["distance"])
    out = grid.copy()

    for obj in meta["small_objects"]:
        for rr, cc in obj["cells"]:
            out[rr, cc] = BG

    for rr, cc in line[:final_idx]:
        out[rr, cc] = GREY

    final_r, final_c = line[final_idx]
    out[final_r, final_c] = GREEN
    out[yr, yc] = YELLOW

    for idx, obj in enumerate(meta["small_objects"], start=1):
        rr = final_r - dr * idx
        cc = final_c - dc * idx
        if _in_bounds(out.shape, rr, cc):
            out[rr, cc] = int(obj["color"])
    return out


def _build_attempt() -> tuple[np.ndarray, dict] | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    kind = random.choice(("horizontal", "vertical", "diagonal"))

    if kind == "horizontal":
        step = (0, random.choice((-1, 1)))
        max_dist = min(DIST_MAX, w - 5)
    elif kind == "vertical":
        step = (random.choice((-1, 1)), 0)
        max_dist = min(DIST_MAX, h - 5)
    else:
        step = (random.choice((-1, 1)), random.choice((-1, 1)))
        max_dist = min(DIST_MAX, min(h, w) - 5)

    if max_dist < DIST_MIN:
        return None

    has_blocker = random.random() < 0.68
    small_count = 1 if random.random() < 0.45 else 2
    min_dist = 8 if small_count == 1 else 9
    if has_blocker and small_count == 2:
        min_dist = 10
    if max_dist < min_dist:
        return None
    distance = random.randint(min_dist, max_dist)

    start, yellow = _sample_markers((h, w), step, distance)
    line = _line_cells(start, step, distance)
    small_indices, blocker_index = _sample_positions(distance, small_count, has_blocker)

    grid = np.zeros((h, w), dtype=int)
    grid[start] = GREEN
    grid[yellow] = YELLOW
    occupied = {start, yellow}
    colors = random.sample(list(OBJECT_COLORS), small_count + (1 if has_blocker else 0))

    small_objects: list[dict] = []
    for idx, path_idx in enumerate(small_indices):
        anchor = line[path_idx]
        templates = _small_templates(kind)[:]
        random.shuffle(templates)
        cells = None
        for template in templates:
            candidate = _abs_coords(anchor, template)
            if _cells_ok(grid.shape, candidate, occupied):
                cells = candidate
                break
        if cells is None:
            return None
        color = colors[idx]
        for rr, cc in cells:
            grid[rr, cc] = color
            occupied.add((rr, cc))
        small_objects.append({"index": path_idx, "color": color, "cells": cells})

    blocker = None
    if has_blocker:
        anchor = line[blocker_index]
        templates = _big_templates(step)[:]
        random.shuffle(templates)
        cells = None
        for template in templates:
            candidate = _abs_coords(anchor, template)
            if _cells_ok(grid.shape, candidate, occupied):
                cells = candidate
                break
        if cells is None:
            return None
        color = colors[-1]
        for rr, cc in cells:
            grid[rr, cc] = color
            occupied.add((rr, cc))
        blocker = {"index": blocker_index, "color": color, "cells": cells}

    meta = {
        "input": grid,
        "shape": grid.shape,
        "start": start,
        "yellow": yellow,
        "step": step,
        "distance": distance,
        "small_objects": sorted(small_objects, key=lambda obj: obj["index"]),
        "blocker_index": blocker_index,
        "blocker": blocker,
    }
    return grid, meta


def _validate_attempt(inp: np.ndarray, meta: dict) -> bool:
    expected = _expected_from_meta(meta)
    actual = main(inp)
    if not np.array_equal(actual, expected):
        return False

    if int(np.count_nonzero(inp == GREEN)) != 1 or int(np.count_nonzero(inp == YELLOW)) != 1:
        return False
    if int(np.count_nonzero(actual == GREEN)) != 1 or int(np.count_nonzero(actual == YELLOW)) != 1:
        return False

    objects, _ = _find_objects(inp)
    small_areas = sorted(obj["area"] for obj in objects if obj["area"] <= 3)
    big_areas = sorted(obj["area"] for obj in objects if obj["area"] >= 4)
    if len(small_areas) != len(meta["small_objects"]):
        return False
    if (meta["blocker"] is None) != (len(big_areas) == 0):
        return False
    if meta["blocker"] is not None and len(big_areas) != 1:
        return False

    sr, sc = meta["start"]
    dr, dc = meta["step"]
    yr, yc = meta["yellow"]
    final_idx = meta["blocker_index"] - 1 if meta["blocker_index"] is not None else meta["distance"] - 1
    final_r = sr + dr * final_idx
    final_c = sc + dc * final_idx
    if (int(np.argwhere(actual == GREEN)[0, 0]), int(np.argwhere(actual == GREEN)[0, 1])) != (final_r, final_c):
        return False
    if int(actual[yr, yc]) != YELLOW:
        return False
    if np.array_equal(actual, inp):
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        built = _build_attempt()
        if built is None:
            continue
        inp, meta = built
        if _validate_attempt(inp, meta):
            return inp
    raise RuntimeError("failed to generate a valid puzzle input")
