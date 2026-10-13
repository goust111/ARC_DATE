from __future__ import annotations

import itertools
import random

import numpy as np


BG = 0
RED = 2
YELLOW = 4
PAIR_COLORS = (1, 3, 7)

GEN_ATTEMPTS = 500


def _endpoint_cells(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int, int]]]:
    reds = [tuple(map(int, rc)) for rc in np.argwhere(grid == RED)]
    others: list[tuple[int, int, int]] = []
    for color in PAIR_COLORS:
        for r, c in np.argwhere(grid == color):
            others.append((int(r), int(c), int(color)))
    return reds, others


def _path_and_growth(
    shape: tuple[int, int],
    red: tuple[int, int],
    other: tuple[int, int],
    color: int,
) -> dict | None:
    h, w = map(int, shape)
    rr, rc = map(int, red)
    nr, nc = map(int, other)
    if rr == nr or rc == nc:
        return None

    left = min(rc, nc)
    right = max(rc, nc)
    span = right - left + 1
    grow_dir = 1 if nr < rr else -1

    path_cells: set[tuple[int, int]] = set()
    for c in range(left, right + 1):
        if (rr, c) != (rr, rc):
            path_cells.add((rr, c))
    row_step = 1 if nr > rr else -1
    for r in range(rr, nr + row_step, row_step):
        if (r, nc) != (nr, nc):
            path_cells.add((r, nc))
    corner = (rr, nc)

    growth_rows: list[dict] = []
    level = 1
    while True:
        row = rr + grow_dir * 2 * level
        if not (0 <= row < h):
            break
        row_left = left + level
        row_right = right - level
        if row_left > row_right:
            break
        cells = [(row, c) for c in range(row_left, row_right + 1)]
        growth_rows.append(
            {
                "row": int(row),
                "cells": cells,
                "color": int(color if level % 2 == 1 else YELLOW),
            }
        )
        level += 1

    footprint = set(path_cells)
    for row_info in growth_rows:
        footprint.update(row_info["cells"])

    return {
        "red": (rr, rc),
        "other": (nr, nc),
        "color": int(color),
        "corner": corner,
        "left": int(left),
        "right": int(right),
        "grow_dir": int(grow_dir),
        "path_cells": path_cells,
        "growth_rows": growth_rows,
        "levels": int(len(growth_rows)),
        "footprint": footprint,
    }


def _valid_matching(grid: np.ndarray, perm: tuple[tuple[int, int, int], ...]) -> list[dict] | None:
    reds, _ = _endpoint_cells(grid)
    endpoints = {(r, c) for r, c in reds} | {(r, c) for r, c, _ in perm}

    pairs: list[dict] = []
    occupied: set[tuple[int, int]] = set()
    for red, other in zip(reds, perm):
        spec = _path_and_growth(grid.shape, red, (other[0], other[1]), other[2])
        if spec is None:
            return None
        if spec["levels"] < 2:
            return None
        if spec["path_cells"] & (endpoints - {spec["red"], spec["other"]}):
            return None
        growth_cells = {cell for row_info in spec["growth_rows"] for cell in row_info["cells"]}
        if growth_cells & endpoints:
            return None
        if occupied & spec["footprint"]:
            return None
        occupied |= spec["footprint"]
        pairs.append(spec)
    return pairs


def _find_unique_pairs(grid: np.ndarray) -> list[dict]:
    reds, others = _endpoint_cells(grid)
    assert 2 <= len(reds) <= 3, "expected 2-3 red endpoints"
    assert len(reds) == len(others), "red and non-red endpoint counts must match"
    assert len({color for _, _, color in others}) == len(others), "non-red endpoint colors must be distinct"

    reds.sort()
    others = sorted(others, key=lambda x: (x[2], x[0], x[1]))
    valid: list[list[dict]] = []
    for perm in itertools.permutations(others):
        pairs = _valid_matching(grid, perm)
        if pairs is not None:
            valid.append(pairs)
    assert len(valid) == 1, f"expected a unique non-crossing matching, got {len(valid)}"
    return valid[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw each red-to-colored L path, then grow an alternating triangle away from the path's vertical arm.

    Input structure:
    - Black background with 2-3 red single-pixel endpoints and the same number of distinct non-red single-pixel endpoints.
    - Each colored endpoint belongs to one red endpoint; within a pair the two pixels are in different rows and columns.
    - The intended pairing is the unique one whose yellow L paths and derived growth regions stay disjoint.

    Rule:
    - For each matched pair, draw a yellow L path from the red pixel horizontally to the colored pixel's column,
      then vertically along that column to the colored pixel, keeping both endpoints unchanged.
    - The row containing the corner acts as the seed span. Grow away from the path's vertical arm, leaving one blank
      row between successive generated rows.
    - On generation k, place a solid row obtained by trimming k cells from each end of the seed span; odd generated
      rows use the pair's non-red color and even generated rows use yellow.
    - Stop when the next generated row would leave the grid or the trimmed span becomes empty.

    Key insight:
    - The hidden pairing must be recovered from the unique non-overlapping L geometry, and that same geometry determines
      the corner, the growth direction, and the alternating growth color.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    pairs = _find_unique_pairs(g)

    for spec in pairs:
        for r, c in spec["path_cells"]:
            if (r, c) not in {spec["red"], spec["other"]}:
                out[r, c] = YELLOW

    blocked = {(r, c) for r, c in np.argwhere(out != BG)}
    for spec in pairs:
        for row_info in spec["growth_rows"]:
            placed = False
            for cell in row_info["cells"]:
                if cell in blocked:
                    continue
                out[cell] = row_info["color"]
                blocked.add(cell)
                placed = True
            if not placed:
                break
    return out


def _render_input(pairs: list[dict], shape: tuple[int, int]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for spec in pairs:
        rr, rc = spec["red"]
        nr, nc = spec["other"]
        grid[rr, rc] = RED
        grid[nr, nc] = spec["color"]
    return grid


def _bands(width: int, count: int) -> list[tuple[int, int]] | None:
    cuts_needed = count - 1
    min_band = 7
    if width < count * min_band:
        return None
    cuts = sorted(random.sample(range(min_band, width - min_band + 1), cuts_needed))
    starts = [0] + cuts
    ends = cuts + [width]
    bands = [(s, e) for s, e in zip(starts, ends)]
    if min(e - s for s, e in bands) < min_band:
        return None
    return bands


def _candidate_pair(shape: tuple[int, int], band: tuple[int, int], color: int, grow_dir: int) -> dict | None:
    h, w = map(int, shape)
    c0, c1 = map(int, band)
    band_width = c1 - c0
    max_levels = min(4, (band_width - 3) // 2)
    if max_levels < 2:
        return None
    levels = random.randint(2, max_levels)
    span = random.randint(2 * levels + 1, min(band_width - 1, 9))
    left = random.randint(c0 + 1, c1 - span)
    right = left + span - 1

    red_on_left = bool(random.getrandbits(1))
    red_col = left if red_on_left else right
    other_col = right if red_on_left else left

    vertical_len = random.randint(2, 5)
    if grow_dir == 1:
        rr_min = vertical_len
        rr_max = h - 1 - 2 * levels
        if rr_min > rr_max:
            return None
        rr = random.randint(rr_min, rr_max)
        nr = rr - vertical_len
    else:
        rr_min = 2 * levels
        rr_max = h - 1 - vertical_len
        if rr_min > rr_max:
            return None
        rr = random.randint(rr_min, rr_max)
        nr = rr + vertical_len

    spec = _path_and_growth(shape, (rr, red_col), (nr, other_col), color)
    if spec is None or spec["levels"] < 2:
        return None
    return spec


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.choice((2, 3))
    h = random.randint(15, 22)
    w = random.randint(max(16, pair_count * 7 + 1), 26)
    bands = _bands(w, pair_count)
    if bands is None:
        return None

    colors = random.sample(list(PAIR_COLORS), pair_count)
    if pair_count == 2:
        grow_dirs = [1, -1]
    else:
        grow_dirs = [1, -1, random.choice((1, -1))]
        random.shuffle(grow_dirs)
    random.shuffle(colors)

    pairs: list[dict] = []
    occupied: set[tuple[int, int]] = set()
    for band, color, grow_dir in zip(bands, colors, grow_dirs):
        spec = None
        for _ in range(80):
            cand = _candidate_pair((h, w), band, color, grow_dir)
            if cand is None:
                continue
            if cand["footprint"] & occupied:
                continue
            spec = cand
            break
        if spec is None:
            return None
        occupied |= spec["footprint"]
        pairs.append(spec)

    grid = _render_input(pairs, (h, w))
    if not _validate_instance(grid):
        return None
    return grid


def _validate_instance(grid: np.ndarray) -> bool:
    try:
        pairs = _find_unique_pairs(grid)
        out = main(grid)
    except AssertionError:
        return False

    if grid.ndim != 2 or out.ndim != 2:
        return False
    if max(grid.shape) > 30 or max(out.shape) > 30:
        return False

    reds, others = _endpoint_cells(grid)
    if len(reds) != len(others) or len(reds) < 2:
        return False
    if any(spec["levels"] < 2 for spec in pairs):
        return False
    if len({spec["grow_dir"] for spec in pairs}) < 2:
        return False

    if not np.all((grid == BG) | (grid == RED) | np.isin(grid, PAIR_COLORS)):
        return False
    if np.count_nonzero(out == YELLOW) <= np.count_nonzero(grid == YELLOW):
        return False

    # Semantic check: removing one endpoint should destroy the unique interpretation.
    test = grid.copy()
    rr, rc = reds[0]
    test[rr, rc] = BG
    try:
        _find_unique_pairs(test)
    except AssertionError:
        return True
    return False


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
