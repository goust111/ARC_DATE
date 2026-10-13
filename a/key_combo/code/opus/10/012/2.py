from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
LEGEND_ROWS = 3
COLOR_POOL = [1, 2, 3, 4, 5, 6, 7, 8, 9]
MAX_GENERATION_ATTEMPTS = 500


def _inclusive_range(a: int, b: int) -> range:
    step = 1 if b >= a else -1
    return range(a, b + step, step)


def _legend_colors(grid: np.ndarray) -> list[int]:
    colors = [int(grid[r, 0]) for r in range(LEGEND_ROWS)]
    assert all(color != BACKGROUND for color in colors), "legend rows must be non-background"
    assert len(set(colors)) == LEGEND_ROWS, "legend colors must be distinct"
    return colors


def _endpoints_for_color(grid: np.ndarray, color: int) -> list[tuple[int, int]]:
    mask = grid == color
    mask[:LEGEND_ROWS, 0] = False
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    assert len(coords) == 2, "each legend color must have exactly two non-legend endpoints"
    return sorted(coords)


def _straight_cells(pair: list[tuple[int, int]]) -> list[tuple[int, int]]:
    (r0, c0), (r1, c1) = pair
    assert (r0 == r1) ^ (c0 == c1), "straight pair must share exactly one axis"
    if r0 == r1:
        return [(r0, c) for c in _inclusive_range(c0, c1)]
    return [(r, c0) for r in _inclusive_range(r0, r1)]


def _diagonal_cells(pair: list[tuple[int, int]]) -> list[tuple[int, int]]:
    (r0, c0), (r1, c1) = pair
    dr = r1 - r0
    dc = c1 - c0
    assert dr != 0 and abs(dr) == abs(dc), "diagonal pair must define a strict 45-degree line"
    step_r = 1 if dr > 0 else -1
    step_c = 1 if dc > 0 else -1
    return [(r0 + k * step_r, c0 + k * step_c) for k in range(abs(dr) + 1)]


def _l_cells(pair: list[tuple[int, int]]) -> list[tuple[int, int]]:
    (r0, c0), (r1, c1) = pair
    assert r0 != r1 and c0 != c1, "L-shape pair must not share a row or column"
    assert abs(r1 - r0) != abs(c1 - c0), "L-shape pair must not already form a diagonal"

    cells = [(r0, c) for c in _inclusive_range(c0, c1)]
    cells.extend((r, c1) for r in _inclusive_range(r0 + 1, r1))
    return cells


def _cells_for_mode(mode: int, pair: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if mode == 0:
        return _straight_cells(pair)
    if mode == 1:
        return _diagonal_cells(pair)
    assert mode == 2
    return _l_cells(pair)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each legend color's endpoint pair using the topology encoded by the legend row.

    Input structure:
    - The background is black.
    - Column 0 contains a three-row legend: row 0, row 1, and row 2 each hold one distinct non-black color.
    - Outside those three legend cells, each legend color appears exactly twice as that color's endpoint pair.

    Rule:
    - Clear the legend cells to black.
    - Connect the row-0 color by a straight horizontal or vertical segment.
    - Connect the row-1 color by a strict 45-degree diagonal segment.
    - Connect the row-2 color by the horizontal-then-vertical L-shaped path determined by the sorted endpoint pair.
    - Preserve all original endpoint pixels while filling every intermediate path cell in the same color.

    Key insight:
    - The legend does not map colors to new colors; its row positions encode which geometric connection rule each color must use.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= LEGEND_ROWS and g.shape[1] >= 2, "expected a 2D grid with a 3-row legend"

    legend = _legend_colors(g)
    out = g.copy()
    out[:LEGEND_ROWS, 0] = BACKGROUND

    for mode, color in enumerate(legend):
        pair = _endpoints_for_color(g, color)
        cells = _cells_for_mode(mode, pair)
        endpoints = set(pair)
        for r, c in cells:
            assert int(out[r, c]) in (BACKGROUND, color) or (r, c) in endpoints, "paths must stay non-overlapping"
            out[r, c] = color
    return out


def _halo(cells: list[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    halo: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    halo.add((nr, nc))
    return halo


def _candidate_ok(cells: list[tuple[int, int]], forbidden: set[tuple[int, int]]) -> bool:
    return len(cells) == len(set(cells)) and all(cell not in forbidden for cell in cells)


def _gen_straight(shape: tuple[int, int]) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    h, w = shape
    if random.random() < 0.5:
        max_dist = min(6, w - 4)
        if max_dist < 2:
            return None
        dist = random.randint(2, max_dist)
        r = random.randint(1, h - 2)
        c0 = random.randint(2, w - 2 - dist)
        pair = [(r, c0), (r, c0 + dist)]
    else:
        max_dist = min(6, h - 3)
        if max_dist < 2:
            return None
        dist = random.randint(2, max_dist)
        c = random.randint(2, w - 2)
        r0 = random.randint(1, h - 2 - dist)
        pair = [(r0, c), (r0 + dist, c)]
    return pair, _straight_cells(pair)


def _gen_diagonal(shape: tuple[int, int]) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    h, w = shape
    max_dist = min(6, h - 3, w - 4)
    if max_dist < 2:
        return None
    dist = random.randint(2, max_dist)
    step_c = random.choice((-1, 1))
    r0 = random.randint(1, h - 2 - dist)
    if step_c == 1:
        c0 = random.randint(2, w - 2 - dist)
    else:
        c0 = random.randint(2 + dist, w - 2)
    pair = [(r0, c0), (r0 + dist, c0 + step_c * dist)]
    return pair, _diagonal_cells(pair)


def _gen_l_shape(shape: tuple[int, int]) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    h, w = shape
    max_dr = min(6, h - 3)
    max_dc = min(6, w - 4)
    if max_dr < 2 or max_dc < 2:
        return None

    for _ in range(50):
        dr = random.randint(2, max_dr)
        dc = random.randint(2, max_dc)
        if dr == dc:
            continue
        step_c = random.choice((-1, 1))
        r0 = random.randint(1, h - 2 - dr)
        if step_c == 1:
            c0 = random.randint(2, w - 2 - dc)
        else:
            c0 = random.randint(2 + dc, w - 2)
        pair = [(r0, c0), (r0 + dr, c0 + step_c * dc)]
        pair = sorted(pair)
        return pair, _l_cells(pair)
    return None


def _mode_generator(mode: int):
    if mode == 0:
        return _gen_straight
    if mode == 1:
        return _gen_diagonal
    return _gen_l_shape


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(12, 18)
    w = random.randint(12, 18)
    legend = random.sample(COLOR_POOL, LEGEND_ROWS)
    shape = (h, w)

    specs: dict[int, dict[str, object]] = {}
    forbidden: set[tuple[int, int]] = set()
    for mode in (1, 2, 0):
        gen = _mode_generator(mode)
        placed = False
        for _ in range(200):
            candidate = gen(shape)
            if candidate is None:
                continue
            endpoints, cells = candidate
            if not _candidate_ok(cells, forbidden):
                continue
            specs[mode] = {"endpoints": endpoints, "cells": cells}
            forbidden.update(_halo(cells, shape))
            placed = True
            break
        if not placed:
            return None

    out = np.zeros(shape, dtype=int)
    inp = np.zeros(shape, dtype=int)
    for mode, color in enumerate(legend):
        spec = specs[mode]
        endpoints = spec["endpoints"]
        cells = spec["cells"]
        for r, c in cells:
            out[r, c] = color
        for r, c in endpoints:
            inp[r, c] = color
        inp[mode, 0] = color

    if np.count_nonzero(inp) != 9:
        return None

    try:
        if not np.array_equal(main(inp), out):
            return None
    except AssertionError:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 2")
