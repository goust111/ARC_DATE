from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
BLOCKER = 5
ARROW = 7
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PALETTE = [1, 2, 3, 4, 6, 8, 9]


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(int(cr), int(cc)):
                    if _in_bounds((h, w), int(nr), int(nc)) and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _line_component(comp: list[tuple[int, int]]) -> tuple[str, list[tuple[int, int]]] | None:
    if not (2 <= len(comp) <= 4):
        return None
    rows = sorted({r for r, _c in comp})
    cols = sorted({c for _r, c in comp})
    if len(rows) == 1:
        ordered = sorted(comp, key=lambda rc: rc[1])
        vals = [c for _r, c in ordered]
        if vals == list(range(vals[0], vals[-1] + 1)):
            return "h", ordered
    if len(cols) == 1:
        ordered = sorted(comp, key=lambda rc: rc[0])
        vals = [r for r, _c in ordered]
        if vals == list(range(vals[0], vals[-1] + 1)):
            return "v", ordered
    return None


def _extract_scene(grid: np.ndarray) -> tuple[list[dict], list[set[tuple[int, int]]], dict[tuple[int, int], int], set[tuple[int, int]]]:
    h, w = grid.shape
    bands: list[dict] = []
    rings: dict[tuple[int, int], int] = {}
    blockers = {(int(r), int(c)) for r, c in np.argwhere(grid == int(BLOCKER))}
    arrows = [set(comp) for comp in _components_of_color(grid, ARROW) if len(comp) == 3]

    seen = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            col = int(grid[r, c])
            if seen[r, c] or col in (BG, BLOCKER, ARROW):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(int(cr), int(cc)):
                    if _in_bounds((h, w), int(nr), int(nc)) and not seen[nr, nc] and int(grid[nr, nc]) == int(col):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))

            if len(comp) == 1:
                rings[comp[0]] = int(col)
                continue

            line = _line_component(comp)
            if line is None:
                continue
            orientation, ordered = line
            if orientation == "h":
                endpoints = [
                    {"cell": ordered[0], "outward": (0, -1)},
                    {"cell": ordered[-1], "outward": (0, 1)},
                ]
            else:
                endpoints = [
                    {"cell": ordered[0], "outward": (-1, 0)},
                    {"cell": ordered[-1], "outward": (1, 0)},
                ]
            bands.append(
                {
                    "color": int(col),
                    "orientation": orientation,
                    "cells": ordered,
                    "cell_set": set(ordered),
                    "endpoints": endpoints,
                }
            )

    for band in bands:
        band["dir"] = None

    for arrow in arrows:
        matches: list[tuple[int, tuple[int, int], tuple[int, int], tuple[int, int]]] = []
        for idx, band in enumerate(bands):
            for endpoint in band["endpoints"]:
                cell = endpoint["cell"]
                outward = endpoint["outward"]
                handle = (int(cell[0] + outward[0]), int(cell[1] + outward[1]))
                if handle in arrow:
                    matches.append((int(idx), cell, outward, handle))
        if len(matches) != 1:
            continue

        idx, _cell, outward, handle = matches[0]
        elbow = (int(handle[0] + outward[0]), int(handle[1] + outward[1]))
        if elbow not in arrow:
            continue
        branch = list(arrow - {handle, elbow})
        if len(branch) != 1:
            continue
        br = branch[0]
        direction = (int(br[0] - elbow[0]), int(br[1] - elbow[1]))
        if abs(int(direction[0])) + abs(int(direction[1])) != 1:
            continue
        band = bands[idx]
        if band["orientation"] == "h" and int(direction[1]) != 0:
            continue
        if band["orientation"] == "v" and int(direction[0]) != 0:
            continue
        band["dir"] = direction

    valid_bands = [band for band in bands if band["dir"] is not None]
    return valid_bands, arrows, rings, blockers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp each short band along its arrow direction, recoloring future copies
    whenever the moving band first passes through a color-ring pixel.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bands, arrows, rings, blockers = _extract_scene(g)

    out = g.copy()
    for arrow in arrows:
        for r, c in arrow:
            out[int(r), int(c)] = BG

    static_band_cells = set()
    for band in bands:
        static_band_cells |= band["cell_set"]

    for band in sorted(bands, key=lambda b: (min(r for r, _c in b["cells"]), min(c for _r, c in b["cells"]))):
        direction = band["dir"]
        assert direction is not None
        dr, dc = map(int, direction)
        current_color = int(band["color"])
        blocked = blockers | (static_band_cells - band["cell_set"])
        step = 1

        while True:
            shifted = [(int(r + dr * step), int(c + dc * step)) for r, c in band["cells"]]
            if any(not _in_bounds((h, w), int(r), int(c)) for r, c in shifted):
                break
            if any((int(r), int(c)) in blocked for r, c in shifted):
                break

            hit_color = None
            for cell in sorted(shifted):
                if cell in rings:
                    hit_color = int(rings[cell])
                    break
            if hit_color is not None:
                current_color = int(hit_color)

            for r, c in shifted:
                out[int(r), int(c)] = int(current_color)
            step += 1

    return out


def _shift_cells(cells: list[tuple[int, int]], direction: tuple[int, int], step: int) -> list[tuple[int, int]]:
    dr, dc = map(int, direction)
    return [(int(r + dr * step), int(c + dc * step)) for r, c in cells]


def _band_cells(start_r: int, start_c: int, length: int, direction: tuple[int, int]) -> list[tuple[int, int]]:
    if int(direction[0]) != 0:
        return [(int(start_r), int(start_c + k)) for k in range(int(length))]
    return [(int(start_r + k), int(start_c)) for k in range(int(length))]


def _arrow_cells(band_cells: list[tuple[int, int]], direction: tuple[int, int], arrow_end: int) -> list[tuple[int, int]]:
    if int(direction[0]) != 0:
        endpoint = band_cells[0] if int(arrow_end) == 0 else band_cells[-1]
        outward = (0, -1) if int(arrow_end) == 0 else (0, 1)
    else:
        endpoint = band_cells[0] if int(arrow_end) == 0 else band_cells[-1]
        outward = (-1, 0) if int(arrow_end) == 0 else (1, 0)

    handle = (int(endpoint[0] + outward[0]), int(endpoint[1] + outward[1]))
    elbow = (int(handle[0] + outward[0]), int(handle[1] + outward[1]))
    branch = (int(elbow[0] + direction[0]), int(elbow[1] + direction[1]))
    return [handle, elbow, branch]


def _max_steps_to_edge(shape: tuple[int, int], cells: list[tuple[int, int]], direction: tuple[int, int]) -> int:
    dr, dc = map(int, direction)
    vals = []
    for r, c in cells:
        if dr == -1:
            vals.append(int(r))
        elif dr == 1:
            vals.append(int(shape[0] - 1 - r))
        elif dc == -1:
            vals.append(int(c))
        else:
            vals.append(int(shape[1] - 1 - c))
    return min(vals)


def _choose_ring_count(max_count: int, min_count: int = 0) -> int:
    weights = {0: 2, 1: 3, 2: 2, 3: 1}
    options = list(range(int(min_count), int(max_count) + 1))
    return int(random.choices(options, weights=[weights[o] for o in options], k=1)[0])


def _can_place_actual(grid: np.ndarray, actual: dict[tuple[int, int], int]) -> bool:
    h, w = grid.shape
    for (r, c), color in actual.items():
        if not _in_bounds((h, w), int(r), int(c)):
            return False
        if int(grid[int(r), int(c)]) != BG:
            return False
        for nr, nc in _neighbors4(int(r), int(c)):
            if _in_bounds((h, w), int(nr), int(nc)) and int(grid[int(nr), int(nc)]) == int(color):
                return False
    return True


def _try_place_band(
    grid: np.ndarray,
    reserved: set[tuple[int, int]],
    band_color: int,
    ring_palette: list[int],
    stop_mode: str,
    min_rings: int,
) -> dict | None:
    h, w = grid.shape

    for _ in range(300):
        direction = random.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
        length = random.randint(2, 4)
        arrow_end = random.randint(0, 1)

        if int(direction[0]) != 0:
            start_r = random.randint(0, h - 1)
            start_c = random.randint(0, w - length)
        else:
            start_r = random.randint(0, h - length)
            start_c = random.randint(0, w - 1)

        band_cells = _band_cells(int(start_r), int(start_c), int(length), direction)
        arrow_cells = _arrow_cells(band_cells, direction, int(arrow_end))
        if any(not _in_bounds((h, w), int(r), int(c)) for r, c in band_cells + arrow_cells):
            continue

        max_steps = _max_steps_to_edge((h, w), band_cells, direction)
        if stop_mode == "blocker":
            if int(max_steps) < max(2, int(min_rings)):
                continue
            steps = random.randint(max(1, int(min_rings)), int(max_steps - 1))
        else:
            if int(max_steps) < max(1, int(min_rings)):
                continue
            steps = int(max_steps)

        ring_count = _choose_ring_count(min(3, int(steps)), min_count=int(min_rings))
        if int(ring_count) > len(ring_palette):
            continue

        ring_steps = sorted(random.sample(list(range(1, int(steps) + 1)), int(ring_count)))
        ring_colors = random.sample(ring_palette, int(ring_count))
        ring_cells: list[tuple[int, int]] = []
        for step in ring_steps:
            shifted = _shift_cells(band_cells, direction, int(step))
            ring_cells.append(random.choice(shifted))

        blocker_cell = None
        if stop_mode == "blocker":
            blocker_step = _shift_cells(band_cells, direction, int(steps + 1))
            blocker_cell = random.choice(blocker_step)

        actual: dict[tuple[int, int], int] = {}
        for cell in band_cells:
            actual[cell] = int(band_color)
        for cell in arrow_cells:
            actual[cell] = ARROW
        for cell, color in zip(ring_cells, ring_colors):
            actual[cell] = int(color)
        if blocker_cell is not None:
            actual[blocker_cell] = BLOCKER

        footprint = set(arrow_cells)
        for step in range(int(steps) + 1):
            footprint.update(_shift_cells(band_cells, direction, int(step)))
        if stop_mode == "blocker":
            footprint.update(_shift_cells(band_cells, direction, int(steps + 1)))

        if footprint & reserved:
            continue
        if not _can_place_actual(grid, actual):
            continue

        for (r, c), color in actual.items():
            grid[int(r), int(c)] = int(color)
        reserved.update(footprint)
        return {
            "direction": direction,
            "length": int(length),
            "stop_mode": stop_mode,
            "steps": int(steps),
            "ring_count": int(ring_count),
        }

    return None


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(12, 26)
    w = random.randint(12, 26)
    grid = np.full((h, w), BG, dtype=int)
    reserved: set[tuple[int, int]] = set()

    max_bands = 2
    if min(h, w) >= 15:
        max_bands = 3
    if min(h, w) >= 19 and h * w >= 360:
        max_bands = 4
    n_bands = random.randint(1, max_bands)

    band_colors = random.sample(PALETTE, int(n_bands))
    ring_palette = [c for c in PALETTE if c not in band_colors]

    if n_bands == 1:
        stop_modes = ["blocker"]
    else:
        stop_modes = ["blocker"] + [random.choice(["blocker", "edge"]) for _ in range(int(n_bands - 1))]
        if "edge" not in stop_modes and random.random() < 0.7:
            stop_modes[random.randrange(1, n_bands)] = "edge"
        random.shuffle(stop_modes)

    want_rings = random.random() < 0.9
    placed: list[dict] = []

    order = sorted(range(int(n_bands)), key=lambda idx: 0 if stop_modes[idx] == "edge" else 1)
    for pos, idx in enumerate(order):
        remaining = len(order) - pos
        have_rings = any(item["ring_count"] > 0 for item in placed)
        min_rings = 1 if want_rings and not have_rings and remaining == 1 else 0
        info = _try_place_band(
            grid,
            reserved,
            int(band_colors[idx]),
            ring_palette,
            stop_modes[idx],
            int(min_rings),
        )
        if info is None:
            return None
        placed.append(info)

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    if np.any(out == ARROW):
        return None
    if not np.any((grid == BG) & (out != BG)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC instance")
