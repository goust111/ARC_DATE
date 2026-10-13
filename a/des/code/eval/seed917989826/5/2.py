from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
import random

import numpy as np


Coord = tuple[int, int]

BG_COLOR = 0
RAIL_COLOR = 5
MARKER_COLOR = 9
BLOCK_COLOR_POOL = [1, 2, 3, 4, 6, 7, 8]
DIRS4: tuple[Coord, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))

GRID_H_RANGE = (12, 20)
GRID_W_RANGE = (12, 20)
RAIL_COUNT_CHOICES = [1, 2, 2, 3]
LINE_LEN_RANGE = (4, 9)
PLACEMENT_RETRIES = 200
GEN_RETRIES = 400


@dataclass(frozen=True)
class RailSpec:
    block_cells: tuple[Coord, ...]
    block_color: int
    touch_cell: Coord
    line_cells: tuple[Coord, ...]
    marker: Coord
    stations: dict[Coord, int]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _components_of_color(grid: np.ndarray, color: int) -> list[list[Coord]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[Coord]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            comp: list[Coord] = []
            q = deque([(r, c)])
            seen[r, c] = True
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if not _in_bounds((h, w), nr, nc):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            comps.append(comp)
    return comps


def _is_solid_2x2(comp: list[Coord]) -> bool:
    if len(comp) != 4:
        return False
    rows = sorted({r for r, _ in comp})
    cols = sorted({c for _, c in comp})
    if len(rows) != 2 or len(cols) != 2:
        return False
    return {(r, c) for r in rows for c in cols} == set(comp)


def _find_blocks(grid: np.ndarray, bg: int) -> list[dict]:
    blocks: list[dict] = []
    for color in map(int, np.unique(grid)):
        if color in {bg, RAIL_COLOR, MARKER_COLOR}:
            continue
        for comp in _components_of_color(grid, color):
            if _is_solid_2x2(comp):
                blocks.append(
                    {
                        "color": color,
                        "cells": tuple(sorted(comp)),
                    }
                )
    assert blocks, "expected at least one 2x2 block"
    return blocks


def _follow_line(
    grid: np.ndarray,
    start: Coord,
    direction: Coord,
    block_mask: np.ndarray,
    bg: int,
) -> tuple[tuple[Coord, ...], Coord] | None:
    h, w = grid.shape
    dr, dc = direction
    r, c = start
    line: list[Coord] = []

    while True:
        if not _in_bounds((h, w), r, c):
            return None
        if block_mask[r, c]:
            return None
        cell = int(grid[r, c])
        if cell in {bg, MARKER_COLOR}:
            return None
        line.append((r, c))

        nr, nc = r + dr, c + dc
        if not _in_bounds((h, w), nr, nc):
            return None
        nxt = int(grid[nr, nc])
        if nxt == MARKER_COLOR:
            return tuple(line), (nr, nc)
        if nxt == bg or block_mask[nr, nc]:
            return None
        r, c = nr, nc


def _trace_rail(grid: np.ndarray, block: dict, block_mask: np.ndarray, bg: int) -> RailSpec:
    block_cells = tuple(block["cells"])
    block_set = set(block_cells)
    candidates: list[RailSpec] = []

    for r, c in block_cells:
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if not _in_bounds(grid.shape, nr, nc):
                continue
            if (nr, nc) in block_set or block_mask[nr, nc]:
                continue
            if int(grid[nr, nc]) in {bg, MARKER_COLOR}:
                continue
            followed = _follow_line(grid, (nr, nc), (dr, dc), block_mask, bg)
            if followed is None:
                continue
            line_cells, marker = followed
            candidates.append(
                RailSpec(
                    block_cells=block_cells,
                    block_color=int(block["color"]),
                    touch_cell=(r, c),
                    line_cells=line_cells,
                    marker=marker,
                    stations={},
                )
            )

    unique: dict[tuple[Coord, Coord], RailSpec] = {}
    for spec in candidates:
        key = (spec.touch_cell, spec.marker)
        unique[key] = spec

    assert len(unique) == 1, "each block must attach to exactly one straight rail"
    spec = next(iter(unique.values()))
    stations = {
        cell: int(grid[cell])
        for cell in spec.line_cells
        if int(grid[cell]) != RAIL_COLOR
    }
    return RailSpec(
        block_cells=spec.block_cells,
        block_color=spec.block_color,
        touch_cell=spec.touch_cell,
        line_cells=spec.line_cells,
        marker=spec.marker,
        stations=stations,
    )


def _final_block_cells(spec: RailSpec) -> tuple[Coord, ...]:
    src_r, src_c = spec.touch_cell
    dst_r, dst_c = spec.line_cells[-1]
    dr = dst_r - src_r
    dc = dst_c - src_c
    return tuple((r + dr, c + dc) for r, c in spec.block_cells)


def _final_block_color(spec: RailSpec) -> int:
    color = spec.block_color
    for cell in spec.line_cells:
        if cell in spec.stations:
            color = spec.stations[cell]
    return color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each 2x2 block along its rail toward the maroon endpoint marker.

    The block recolors to the last station color it passes over; if there is no
    station on that rail, it keeps its original color. Rails stay in place,
    stations are restored to grey, and blocks move to the endpoint side.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)

    blocks = _find_blocks(g, bg)
    block_mask = np.zeros_like(g, dtype=bool)
    for block in blocks:
        for r, c in block["cells"]:
            block_mask[r, c] = True

    rails = [_trace_rail(g, block, block_mask, bg) for block in blocks]

    out = g.copy()
    for spec in rails:
        for r, c in spec.block_cells:
            out[r, c] = bg
    for spec in rails:
        for r, c in spec.line_cells:
            out[r, c] = RAIL_COLOR
    for spec in rails:
        final_color = _final_block_color(spec)
        for r, c in _final_block_cells(spec):
            out[r, c] = final_color
    return out


def _padded(cells: set[Coord], shape: tuple[int, int]) -> set[Coord]:
    h, w = shape
    out: set[Coord] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out.add((nr, nc))
    return out


def _station_mode_list(count: int) -> list[str]:
    if count == 1:
        return [random.choice(["none", "single", "multi"])]
    if count == 2:
        modes = ["none", "multi"]
        random.shuffle(modes)
        return modes
    modes = ["none", "single", "multi"]
    random.shuffle(modes)
    return modes


def _choose_station_cells(line_cells: tuple[Coord, ...], mode: str) -> list[Coord]:
    line_len = len(line_cells)
    max_k = max(0, min(4, line_len - 1))
    if mode == "none" or max_k == 0:
        return []
    if mode == "single":
        return [random.choice(list(line_cells))]
    count = random.randint(2, max_k)
    return random.sample(list(line_cells), count)


def _make_spec(
    h: int,
    w: int,
    block_color: int,
    station_mode: str,
) -> RailSpec | None:
    direction = random.choice(["E", "W", "S", "N"])
    if direction in {"E", "W"}:
        max_len = min(LINE_LEN_RANGE[1], w - 4)
    else:
        max_len = min(LINE_LEN_RANGE[1], h - 4)
    if max_len < LINE_LEN_RANGE[0]:
        return None
    line_len = random.randint(LINE_LEN_RANGE[0], max_len)

    if direction == "E":
        touch_on_top = bool(random.getrandbits(1))
        row_lo = 0 if touch_on_top else 1
        row_hi = h - 2 if touch_on_top else h - 1
        c0_lo, c0_hi = 2, w - line_len - 1
        if row_lo > row_hi or c0_lo > c0_hi:
            return None
        row = random.randint(row_lo, row_hi)
        c0 = random.randint(c0_lo, c0_hi)
        block_rows = (row, row + 1) if touch_on_top else (row - 1, row)
        block_cols = (c0 - 2, c0 - 1)
        touch_cell = (row, c0 - 1)
        line_cells = tuple((row, c) for c in range(c0, c0 + line_len))
        marker = (row, c0 + line_len)
    elif direction == "W":
        touch_on_top = bool(random.getrandbits(1))
        row_lo = 0 if touch_on_top else 1
        row_hi = h - 2 if touch_on_top else h - 1
        c0_lo, c0_hi = 1, w - line_len - 2
        if row_lo > row_hi or c0_lo > c0_hi:
            return None
        row = random.randint(row_lo, row_hi)
        c0 = random.randint(c0_lo, c0_hi)
        c1 = c0 + line_len - 1
        block_rows = (row, row + 1) if touch_on_top else (row - 1, row)
        block_cols = (c1 + 1, c1 + 2)
        touch_cell = (row, c1 + 1)
        line_cells = tuple((row, c) for c in range(c1, c0 - 1, -1))
        marker = (row, c0 - 1)
    elif direction == "S":
        touch_on_left = bool(random.getrandbits(1))
        col_lo = 0 if touch_on_left else 1
        col_hi = w - 2 if touch_on_left else w - 1
        r0_lo, r0_hi = 2, h - line_len - 1
        if col_lo > col_hi or r0_lo > r0_hi:
            return None
        col = random.randint(col_lo, col_hi)
        r0 = random.randint(r0_lo, r0_hi)
        block_rows = (r0 - 2, r0 - 1)
        block_cols = (col, col + 1) if touch_on_left else (col - 1, col)
        touch_cell = (r0 - 1, col)
        line_cells = tuple((r, col) for r in range(r0, r0 + line_len))
        marker = (r0 + line_len, col)
    else:
        touch_on_left = bool(random.getrandbits(1))
        col_lo = 0 if touch_on_left else 1
        col_hi = w - 2 if touch_on_left else w - 1
        r0_lo, r0_hi = 1, h - line_len - 2
        if col_lo > col_hi or r0_lo > r0_hi:
            return None
        col = random.randint(col_lo, col_hi)
        r0 = random.randint(r0_lo, r0_hi)
        r1 = r0 + line_len - 1
        block_rows = (r1 + 1, r1 + 2)
        block_cols = (col, col + 1) if touch_on_left else (col - 1, col)
        touch_cell = (r1 + 1, col)
        line_cells = tuple((r, col) for r in range(r1, r0 - 1, -1))
        marker = (r0 - 1, col)

    block_cells = tuple((r, c) for r in block_rows for c in block_cols)
    occupied = set(block_cells) | set(line_cells) | {marker}
    if len(occupied) != len(block_cells) + len(line_cells) + 1:
        return None

    station_cells = _choose_station_cells(line_cells, station_mode)
    station_palette = [c for c in BLOCK_COLOR_POOL if c != block_color]
    if len(station_cells) > len(station_palette):
        return None
    station_colors = random.sample(station_palette, len(station_cells))
    stations = {cell: color for cell, color in zip(station_cells, station_colors)}
    return RailSpec(
        block_cells=block_cells,
        block_color=block_color,
        touch_cell=touch_cell,
        line_cells=line_cells,
        marker=marker,
        stations=stations,
    )


def _spec_cells(spec: RailSpec) -> set[Coord]:
    return set(spec.block_cells) | set(spec.line_cells) | {spec.marker}


def _render_input(shape: tuple[int, int], specs: list[RailSpec]) -> np.ndarray:
    grid = np.full(shape, BG_COLOR, dtype=int)
    for spec in specs:
        for r, c in spec.line_cells:
            grid[r, c] = RAIL_COLOR
        for (r, c), color in spec.stations.items():
            grid[r, c] = color
        for r, c in spec.block_cells:
            grid[r, c] = spec.block_color
        mr, mc = spec.marker
        grid[mr, mc] = MARKER_COLOR
    return grid


def _render_expected(inp: np.ndarray, specs: list[RailSpec]) -> np.ndarray:
    out = inp.copy()
    for spec in specs:
        for r, c in spec.block_cells:
            out[r, c] = BG_COLOR
    for spec in specs:
        for r, c in spec.line_cells:
            out[r, c] = RAIL_COLOR
    for spec in specs:
        final_color = _final_block_color(spec)
        for r, c in _final_block_cells(spec):
            out[r, c] = final_color
    return out


def _attempt_generate() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    rail_count = random.choice(RAIL_COUNT_CHOICES)
    rail_count = min(rail_count, len(BLOCK_COLOR_POOL))
    block_colors = random.sample(BLOCK_COLOR_POOL, rail_count)
    station_modes = _station_mode_list(rail_count)

    specs: list[RailSpec] = []
    blocked: set[Coord] = set()
    for i in range(rail_count):
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            spec = _make_spec(h, w, block_colors[i], station_modes[i])
            if spec is None:
                continue
            cells = _spec_cells(spec)
            if any(cell in blocked for cell in cells):
                continue
            specs.append(spec)
            blocked |= _padded(cells, (h, w))
            placed = True
            break
        if not placed:
            return None

    inp = _render_input((h, w), specs)
    if Counter(inp.ravel()).most_common(1)[0][0] != BG_COLOR:
        return None

    expected = _render_expected(inp, specs)
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sliding-rail sample")
