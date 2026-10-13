from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG_COLOR = 7
LEGEND_COLOR = 5
THRESHOLD_WIDTH = 3
OBJECT_COLORS = (1, 2, 3, 4, 6, 8, 9)

GRID_HEIGHT_RANGE = (10, 18)
GRID_WIDTH_RANGE = (10, 18)
GENERATION_RETRIES = 400
PLACEMENT_RETRIES = 200


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _parse_legend(grid: np.ndarray, bg: int) -> tuple[int, set[tuple[int, int]], int]:
    h, _w = grid.shape
    candidates: list[tuple[int, set[tuple[int, int]], int]] = []

    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        coords = np.argwhere(grid == color)
        rows = sorted({int(r) for r, _c in coords})
        if len(rows) != 2 or rows[1] != rows[0] + 1:
            continue
        top_row, bottom_row = rows
        if top_row != 0 and bottom_row != h - 1:
            continue

        top_cols = sorted(int(c) for r, c in coords if int(r) == top_row)
        bottom_cols = sorted(int(c) for r, c in coords if int(r) == bottom_row)
        if len(top_cols) != 1 or len(bottom_cols) < 2:
            continue
        if bottom_cols != list(range(bottom_cols[0], bottom_cols[-1] + 1)):
            continue
        if not (bottom_cols[0] <= top_cols[0] <= bottom_cols[-1]):
            continue

        legend_cells = {tuple(map(int, coord)) for coord in coords}
        candidates.append((color, legend_cells, len(bottom_cols)))

    assert len(candidates) == 1, "expected exactly one 2-row edge legend"
    return candidates[0]


def _line_length(a: tuple[int, int], b: tuple[int, int]) -> int:
    if a[0] == b[0]:
        return abs(a[1] - b[1]) + 1
    return abs(a[0] - b[0]) + 1


def _line_footprint(
    a: tuple[int, int],
    b: tuple[int, int],
    thick: bool,
    shape: tuple[int, int],
) -> set[tuple[int, int]]:
    h, w = shape
    cells: set[tuple[int, int]] = set()

    if a[0] == b[0]:
        r = a[0]
        c0, c1 = sorted((a[1], b[1]))
        row_lo = max(0, r - (1 if thick else 0))
        row_hi = min(h - 1, r + (1 if thick else 0))
        for rr in range(row_lo, row_hi + 1):
            for cc in range(c0, c1 + 1):
                cells.add((rr, cc))
        return cells

    c = a[1]
    r0, r1 = sorted((a[0], b[0]))
    col_lo = max(0, c - (1 if thick else 0))
    col_hi = min(w - 1, c + (1 if thick else 0))
    for rr in range(r0, r1 + 1):
        for cc in range(col_lo, col_hi + 1):
            cells.add((rr, cc))
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    _legend_color, legend_cells, threshold = _parse_legend(g, bg)

    pairs: dict[int, list[tuple[int, int]]] = {}
    for r, c in np.argwhere(g != bg):
        coord = (int(r), int(c))
        if coord in legend_cells:
            continue
        color = int(g[r, c])
        pairs.setdefault(color, []).append(coord)

    assert pairs, "expected at least one colored pair"

    out = np.full_like(g, bg)
    for color, coords in pairs.items():
        assert len(coords) == 2, "each non-background color must appear exactly twice"
        a, b = coords
        assert a[0] == b[0] or a[1] == b[1], "paired markers must share a row or column"

        if a[0] == b[0]:
            row = a[0]
            c0, c1 = sorted((a[1], b[1]))
            if c1 - c0 > 1:
                assert np.all(g[row, c0 + 1 : c1] == bg), "horizontal markers must be separated by background"
        else:
            col = a[1]
            r0, r1 = sorted((a[0], b[0]))
            if r1 - r0 > 1:
                assert np.all(g[r0 + 1 : r1, col] == bg), "vertical markers must be separated by background"

        thick = _line_length(a, b) <= threshold
        for rr, cc in _line_footprint(a, b, thick=thick, shape=g.shape):
            assert out[rr, cc] in (bg, color), "line footprints must not conflict"
            out[rr, cc] = color

    for r, c in legend_cells:
        out[r, c] = bg
    return out


def _place_legend(grid: np.ndarray, side: str) -> set[tuple[int, int]]:
    h, w = grid.shape
    top_row = 0 if side == "top" else h - 2
    bar_start = random.randint(0, w - THRESHOLD_WIDTH)
    peak_col = random.randint(bar_start, bar_start + THRESHOLD_WIDTH - 1)

    grid[top_row, peak_col] = LEGEND_COLOR
    grid[top_row + 1, bar_start : bar_start + THRESHOLD_WIDTH] = LEGEND_COLOR
    return {(top_row, peak_col)} | {(top_row + 1, c) for c in range(bar_start, bar_start + THRESHOLD_WIDTH)}


def _body_row_bounds(height: int, legend_side: str) -> tuple[int, int]:
    if legend_side == "top":
        return 3, height - 1
    return 0, height - 4


def _sample_pair(
    shape: tuple[int, int],
    body_rows: tuple[int, int],
    legend_side: str,
    orientation: str,
    short: bool,
    force_edge_clip: bool,
    occupied: set[tuple[int, int]],
    legend_cells: set[tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    h, w = shape
    row_lo, row_hi = body_rows
    body_height = row_hi - row_lo + 1

    for _ in range(PLACEMENT_RETRIES):
        if orientation == "H":
            if short:
                length = THRESHOLD_WIDTH
            else:
                if w <= THRESHOLD_WIDTH:
                    return None
                length = random.randint(THRESHOLD_WIDTH + 1, w)

            row = (h - 1 if legend_side == "top" else 0) if force_edge_clip else random.randint(row_lo, row_hi)
            start_col = random.randint(0, w - length)
            a = (row, start_col)
            b = (row, start_col + length - 1)
        else:
            if short:
                length = THRESHOLD_WIDTH
            else:
                if body_height <= THRESHOLD_WIDTH:
                    return None
                length = random.randint(THRESHOLD_WIDTH + 1, body_height)

            col = random.choice((0, w - 1)) if force_edge_clip else random.randint(0, w - 1)
            start_row = random.randint(row_lo, row_hi - length + 1)
            a = (start_row, col)
            b = (start_row + length - 1, col)

        footprint = _line_footprint(a, b, thick=short, shape=shape)
        if footprint & occupied:
            continue
        if footprint & legend_cells:
            continue
        return a, b

    return None


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    legend_side = random.choice(("top", "bottom"))
    body_rows = _body_row_bounds(height, legend_side)

    if body_rows[1] - body_rows[0] + 1 <= THRESHOLD_WIDTH:
        return None

    grid = np.full((height, width), BG_COLOR, dtype=int)
    legend_cells = _place_legend(grid, legend_side)

    specs = [
        ("H", False, False),
        ("V", False, False),
        ("H", True, True),
        ("V", True, True),
    ]
    random.shuffle(specs)
    colors = random.sample(OBJECT_COLORS, len(specs))

    occupied: set[tuple[int, int]] = set()
    placements: list[tuple[int, tuple[int, int], tuple[int, int]]] = []
    for color, (orientation, short, force_edge_clip) in zip(colors, specs):
        pair = _sample_pair(
            shape=grid.shape,
            body_rows=body_rows,
            legend_side=legend_side,
            orientation=orientation,
            short=short,
            force_edge_clip=force_edge_clip,
            occupied=occupied,
            legend_cells=legend_cells,
        )
        if pair is None:
            return None
        a, b = pair
        occupied |= _line_footprint(a, b, thick=short, shape=grid.shape)
        placements.append((color, a, b))

    for color, a, b in placements:
        grid[a] = color
        grid[b] = color

    # The generated input should satisfy the solver's structural assumptions.
    _ = main(grid)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
