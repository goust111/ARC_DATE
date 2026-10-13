from __future__ import annotations

import random

import numpy as np


GREY = 5
YELLOW = 4
COLOR_POOL = (1, 2, 3, 6, 7, 8, 9)

COMPACT_HEIGHT_RANGE = (6, 10)
COMPACT_WIDTH_RANGE = (6, 10)
PAIR_COUNT_OPTIONS = ((2, 1), (1, 2), (2, 2), (3, 1), (1, 3))
MAX_GENERATION_ATTEMPTS = 400


def _compress_even_cells(grid: np.ndarray) -> np.ndarray:
    return np.asarray(grid, dtype=int)[::2, ::2].copy()


def _draw_axis_line(canvas: np.ndarray, start: tuple[int, int], end: tuple[int, int], color: int) -> np.ndarray:
    r0, c0 = map(int, start)
    r1, c1 = map(int, end)
    if r0 == r1:
        lo, hi = sorted((c0, c1))
        canvas[r0, lo : hi + 1] = int(color)
        rr = np.full(hi - lo + 1, r0, dtype=int)
        cc = np.arange(lo, hi + 1, dtype=int)
    elif c0 == c1:
        lo, hi = sorted((r0, r1))
        canvas[lo : hi + 1, c0] = int(color)
        rr = np.arange(lo, hi + 1, dtype=int)
        cc = np.full(hi - lo + 1, c0, dtype=int)
    else:
        raise AssertionError("paired cells must share one row or one column")
    return np.column_stack((rr, cc))


def _find_color_pairs(compact: np.ndarray) -> list[tuple[int, tuple[int, int], tuple[int, int]]]:
    pairs: list[tuple[int, tuple[int, int], tuple[int, int]]] = []
    for color in sorted(int(v) for v in np.unique(compact) if int(v) != GREY):
        coords = np.argwhere(compact == color)
        assert coords.shape[0] == 2, "each non-grey color must appear exactly twice"
        a = tuple(map(int, coords[0]))
        b = tuple(map(int, coords[1]))
        assert (a[0] == b[0]) ^ (a[1] == b[1]), "each pair must align horizontally or vertically"
        pairs.append((int(color), a, b))
    assert 3 <= len(pairs) <= 4, "expected three or four color pairs"
    return pairs


def _solve_compact(compact: np.ndarray) -> np.ndarray:
    out = compact.copy()
    touched: dict[tuple[int, int], set[int]] = {}

    for color, a, b in _find_color_pairs(compact):
        line_coords = _draw_axis_line(out, a, b, color)
        for r, c in line_coords:
            touched.setdefault((int(r), int(c)), set()).add(int(color))

    for (r, c), colors in touched.items():
        if len(colors) >= 2:
            out[int(r), int(c)] = YELLOW
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress away the grey separator lattice, connect each same-color pair, and mark line crossings yellow.

    Input structure:
    - The grid has a grey background, and every odd-index row and odd-index column is entirely grey.
    - Colored pixels appear only on even-row/even-column lattice points.
    - Each non-grey color appears exactly twice, forming one horizontal or vertical pair after compression.

    Rule:
    - Delete all odd-index rows and odd-index columns to obtain the compact lattice of even/even cells.
    - In the compact grid, connect each same-color pair with a straight horizontal or vertical segment, including endpoints.
    - Draw every pair's segment, then recolor any cell traversed by two or more different colors to yellow.

    Key insight:
    - The odd-index grey lattice is only a separator scaffold; the real objects live on the even/even subgrid, and the crossing logic is defined only after that compression.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape[0] % 2 == 0 and g.shape[1] % 2 == 0, "expected an even-sized separator grid"
    assert np.all(g[1::2, :] == GREY), "odd rows must be grey separators"
    assert np.all(g[:, 1::2] == GREY), "odd columns must be grey separators"
    assert np.all(g[::2, ::2] >= 0), "grid must contain valid color ids"
    assert np.all(g[::2, 1::2] == GREY) and np.all(g[1::2, ::2] == GREY)

    compact = _compress_even_cells(g)
    return _solve_compact(compact)


def _sample_dimensions(h_count: int, v_count: int) -> tuple[int, int]:
    min_h = max(COMPACT_HEIGHT_RANGE[0], h_count + 3)
    min_w = max(COMPACT_WIDTH_RANGE[0], v_count + 3)
    height = random.randint(min_h, COMPACT_HEIGHT_RANGE[1])
    width = random.randint(min_w, COMPACT_WIDTH_RANGE[1])
    return height, width


def _choose_rows_cols(height: int, width: int, h_count: int, v_count: int) -> tuple[list[int], list[int]]:
    rows = sorted(random.sample(range(1, height - 1), h_count))
    cols = sorted(random.sample(range(1, width - 1), v_count))
    return rows, cols


def _horizontal_endpoints(width: int, cols: list[int]) -> tuple[int, int]:
    left = random.randint(0, min(cols) - 1)
    right = random.randint(max(cols) + 1, width - 1)
    return left, right


def _vertical_endpoints(height: int, rows: list[int]) -> tuple[int, int]:
    top = random.randint(0, min(rows) - 1)
    bottom = random.randint(max(rows) + 1, height - 1)
    return top, bottom


def _build_compact_case() -> np.ndarray:
    h_count, v_count = random.choice(PAIR_COUNT_OPTIONS)
    height, width = _sample_dimensions(h_count, v_count)
    rows, cols = _choose_rows_cols(height, width, h_count, v_count)
    colors = random.sample(COLOR_POOL, h_count + v_count)

    compact = np.full((height, width), GREY, dtype=int)
    color_specs: list[tuple[int, tuple[int, int], tuple[int, int]]] = []

    for row, color in zip(rows, colors[:h_count]):
        left, right = _horizontal_endpoints(width, cols)
        compact[row, left] = color
        compact[row, right] = color
        color_specs.append((color, (row, left), (row, right)))

    for col, color in zip(cols, colors[h_count:]):
        top, bottom = _vertical_endpoints(height, rows)
        compact[top, col] = color
        compact[bottom, col] = color
        color_specs.append((color, (top, col), (bottom, col)))

    _validate_compact_case(compact, color_specs)
    return compact


def _validate_compact_case(compact: np.ndarray, specs: list[tuple[int, tuple[int, int], tuple[int, int]]]) -> None:
    height, width = map(int, compact.shape)
    assert 6 <= height <= 10 and 6 <= width <= 10
    assert np.count_nonzero(compact != GREY) == 2 * len(specs)

    h_specs = [spec for spec in specs if spec[1][0] == spec[2][0]]
    v_specs = [spec for spec in specs if spec[1][1] == spec[2][1]]
    assert h_specs and v_specs, "need both horizontal and vertical connections"

    crossings: set[tuple[int, int]] = set()
    for _h_color, (hr, hc0), (_, hc1) in h_specs:
        lo_c, hi_c = sorted((hc0, hc1))
        assert hi_c - lo_c >= 2, "horizontal lines need visible interior"
        for _v_color, (vr0, vc), (vr1, _) in v_specs:
            lo_r, hi_r = sorted((vr0, vr1))
            assert hi_r - lo_r >= 2, "vertical lines need visible interior"
            if lo_c < vc < hi_c and lo_r < hr < hi_r:
                crossings.add((hr, vc))
    assert len(crossings) >= 2, "need at least two compact-grid crossings"

    solved = _solve_compact(compact)
    yellow_cells = set(map(tuple, np.argwhere(solved == YELLOW)))
    assert yellow_cells == crossings, "crossings must be exactly the multi-color overlaps"

    for color, a, b in specs:
        coords = np.argwhere(compact == color)
        assert coords.shape[0] == 2
        got = {tuple(map(int, coords[0])), tuple(map(int, coords[1]))}
        assert got == {a, b}


def _expand_with_separators(compact: np.ndarray) -> np.ndarray:
    h, w = map(int, compact.shape)
    expanded = np.full((2 * h, 2 * w), GREY, dtype=int)
    expanded[::2, ::2] = compact
    return expanded


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        compact = _build_compact_case()
        expanded = _expand_with_separators(compact)
        expected = _solve_compact(compact)
        actual = main(expanded)
        if np.array_equal(actual, expected):
            return expanded
    raise RuntimeError("failed to generate a valid separator-compression line puzzle")
