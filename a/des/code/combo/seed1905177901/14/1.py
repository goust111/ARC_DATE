from __future__ import annotations

import random
from collections import defaultdict, deque

import numpy as np


BG = 0
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
DIR_MAIN = (1, 1)
DIR_ANTI = (1, -1)


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _neighbors4(h: int, w: int, r: int, c: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(h, w, rr, cc):
                    if seen[nr, nc] or int(grid[nr, nc]) != int(color):
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: set[tuple[int, int]] | list[tuple[int, int]]) -> tuple[int, int, int, int]:
    pts = list(cells)
    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    return min(rows), min(cols), max(rows), max(cols)


def _infer_direction(cells: set[tuple[int, int]]) -> tuple[int, int]:
    r0, c0, r1, c1 = _bbox(cells)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if w > h:
        return DIR_MAIN
    if h > w:
        return DIR_ANTI

    rel = {(r - r0, c - c0) for r, c in cells}
    main_score = int((0, 0) in rel) + int((h - 1, w - 1) in rel)
    anti_score = int((0, w - 1) in rel) + int((h - 1, 0) in rel)
    return DIR_MAIN if main_score >= anti_score else DIR_ANTI


def _shift_cells(cells: set[tuple[int, int]], direction: tuple[int, int], steps: int) -> set[tuple[int, int]]:
    dr, dc = direction
    return {(r + dr * steps, c + dc * steps) for r, c in cells}


def _trail_cells(cells: set[tuple[int, int]], direction: tuple[int, int], steps: int) -> set[tuple[int, int]]:
    r0, c0, _r1, c1 = _bbox(cells)
    if direction == DIR_MAIN:
        start_r, start_c, dc = r0, c0, 1
    else:
        start_r, start_c, dc = r0, c1, -1
    return {(start_r + k, start_c + dc * k) for k in range(steps + 1)}


def _paint_cells(grid: np.ndarray, cells: set[tuple[int, int]], color: int) -> None:
    h, w = grid.shape
    for r, c in cells:
        if _in_bounds(h, w, r, c):
            grid[r, c] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each monochrome object diagonally by the number of same-colored edge markers.

    We interpret the prompt's "longest diagonal" using the object's bbox aspect ratio:
    wider objects move on the \ diagonal, taller objects move on the / diagonal.
    The visible trail is the corresponding top-corner diagonal segment from the
    original bbox to the shifted bbox.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    marker_counts: defaultdict[int, int] = defaultdict(int)
    objects: list[dict] = []

    for color in PALETTE:
        for comp in _components_of_color(g, color):
            pts = set(comp)
            if len(pts) == 1:
                r, c = comp[0]
                if r in (0, h - 1) or c in (0, w - 1):
                    marker_counts[int(color)] += 1
                    continue
            objects.append({"color": int(color), "cells": pts})

    out = np.full_like(g, BG)
    objects.sort(key=lambda obj: _bbox(obj["cells"])[:2])

    for obj in objects:
        color = int(obj["color"])
        cells = obj["cells"]
        direction = _infer_direction(cells)
        steps = int(marker_counts.get(color, 0))
        _paint_cells(out, _trail_cells(cells, direction, steps), color)
        _paint_cells(out, _shift_cells(cells, direction, steps), color)

    return out


def _shape_mask(direction: tuple[int, int]) -> np.ndarray:
    if direction == DIR_MAIN:
        h = random.randint(2, 4)
        w = random.randint(max(3, h + 1), min(7, h + 3))
    else:
        w = random.randint(2, 4)
        h = random.randint(max(3, w + 1), min(7, w + 3))

    mask = np.ones((h, w), dtype=bool)
    if h * w >= 6 and random.random() < 0.45:
        if direction == DIR_MAIN:
            rr, cc = random.choice([(0, w - 1), (h - 1, 0)])
        else:
            rr, cc = random.choice([(0, 0), (h - 1, w - 1)])
        mask[rr, cc] = False
    return mask


def _mask_to_cells(mask: np.ndarray, r0: int, c0: int) -> set[tuple[int, int]]:
    pts: set[tuple[int, int]] = set()
    for r, c in np.argwhere(mask):
        pts.add((int(r0 + r), int(c0 + c)))
    return pts


def _expand(cells: set[tuple[int, int]], h: int, w: int, radius: int = 1) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                rr = r + dr
                cc = c + dc
                if _in_bounds(h, w, rr, cc):
                    out.add((rr, cc))
    return out


def _boundary_cells(h: int, w: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for c in range(w):
        cells.append((0, c))
        if h > 1:
            cells.append((h - 1, c))
    for r in range(1, h - 1):
        cells.append((r, 0))
        if w > 1:
            cells.append((r, w - 1))
    return cells


def _render_input(specs: list[dict], h: int, w: int) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    for spec in specs:
        _paint_cells(grid, spec["cells"], spec["color"])
        _paint_cells(grid, set(spec["markers"]), spec["color"])
    return grid


def _render_output_from_specs(specs: list[dict], h: int, w: int) -> np.ndarray:
    out = np.full((h, w), BG, dtype=int)
    for spec in sorted(specs, key=lambda item: _bbox(item["cells"])[:2]):
        color = int(spec["color"])
        cells = set(spec["cells"])
        direction = spec["direction"]
        steps = int(spec["steps"])
        _paint_cells(out, _trail_cells(cells, direction, steps), color)
        _paint_cells(out, _shift_cells(cells, direction, steps), color)
    return out


def _place_object(
    h: int,
    w: int,
    direction: tuple[int, int],
    blocked: set[tuple[int, int]],
) -> dict | None:
    for _ in range(200):
        steps = random.randint(1, 4)
        mask = _shape_mask(direction)
        obj_h, obj_w = mask.shape

        min_r = 2
        max_r = h - steps - obj_h - 3
        if direction == DIR_MAIN:
            min_c = 2
            max_c = w - steps - obj_w - 3
        else:
            min_c = steps + 2
            max_c = w - obj_w - 3

        if max_r < min_r or max_c < min_c:
            continue

        r0 = random.randint(min_r, max_r)
        c0 = random.randint(min_c, max_c)
        cells = _mask_to_cells(mask, r0, c0)
        trail = _trail_cells(cells, direction, steps)
        shifted = _shift_cells(cells, direction, steps)
        footprint = cells | trail | shifted

        if any(not _in_bounds(h, w, r, c) for r, c in footprint):
            continue
        if footprint & blocked:
            continue

        return {
            "cells": cells,
            "direction": direction,
            "steps": steps,
            "footprint": footprint,
        }
    return None


def _assign_markers(specs: list[dict], h: int, w: int) -> bool:
    occupied = set()
    for spec in specs:
        occupied |= spec["cells"]

    used: set[tuple[int, int]] = set()
    boundary = _boundary_cells(h, w)
    random.shuffle(boundary)

    for spec in specs:
        chosen: list[tuple[int, int]] = []
        need = int(spec["steps"])
        for cell in boundary:
            if cell in used:
                continue
            if any(max(abs(cell[0] - rr), abs(cell[1] - cc)) <= 1 for rr, cc in used):
                continue
            if any(max(abs(cell[0] - rr), abs(cell[1] - cc)) <= 1 for rr, cc in occupied):
                continue
            chosen.append(cell)
            used.add(cell)
            if len(chosen) == need:
                break
        if len(chosen) != need:
            return False
        spec["markers"] = chosen
    return True


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(16, 24)
    w = random.randint(16, 24)
    num_objects = random.randint(2, 4)
    colors = random.sample(PALETTE, k=num_objects)

    directions = [DIR_MAIN, DIR_ANTI]
    while len(directions) < num_objects:
        directions.append(random.choice([DIR_MAIN, DIR_ANTI]))
    random.shuffle(directions)

    specs: list[dict] = []
    blocked: set[tuple[int, int]] = set()

    for color, direction in zip(colors, directions):
        placed = _place_object(h, w, direction, blocked)
        if placed is None:
            return None
        placed["color"] = int(color)
        placed["markers"] = []
        specs.append(placed)
        blocked |= _expand(placed["footprint"], h, w, radius=1)

    if not _assign_markers(specs, h, w):
        return None

    inp = _render_input(specs, h, w)
    expected = _render_output_from_specs(specs, h, w)
    actual = main(inp)

    if np.array_equal(inp, expected):
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input after 400 attempts")
