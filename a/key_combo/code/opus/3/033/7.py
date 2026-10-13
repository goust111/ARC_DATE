from __future__ import annotations

import random

import numpy as np


BG_COLOR = 0
GRID_LINE_COLOR = 3
CORNER_MARKER_COLORS = {
    (0, 0): 2,
    (0, 1): 4,
    (1, 0): 1,
    (1, 1): 7,
}
WEAVE_COLOR_POOL = [5, 6, 8, 9]
GRID_COUNT_CHOICES = [3, 4]
CELL_SIZE_RANGE = (2, 5)
TOP_MARGIN_RANGE = (2, 4)
RIGHT_MARGIN_RANGE = (2, 4)
SIDE_MARGIN_RANGE = (0, 4)
GEN_RETRIES = 400


def _most_common_color(values: np.ndarray) -> int:
    flat = np.asarray(values, dtype=int).ravel()
    assert flat.size > 0
    counts = np.bincount(flat, minlength=10)
    return int(np.argmax(counts))


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _spans_between(separators: list[int]) -> list[tuple[int, int]]:
    spans = []
    for a, b in zip(separators, separators[1:]):
        assert b - a >= 2, "adjacent separator lines must leave a non-empty cell interior"
        spans.append((a + 1, b - 1))
    return spans


def _uniform_sizes(spans: list[tuple[int, int]]) -> int:
    sizes = {end - start + 1 for start, end in spans}
    assert len(sizes) == 1, "cell interiors must be uniform"
    return sizes.pop()


def _single_color(subgrid: np.ndarray, forbidden: set[int]) -> int:
    colors = {int(v) for v in subgrid.ravel() if int(v) not in forbidden}
    assert len(colors) == 1, "expected exactly one marker color in the region"
    return colors.pop()


def _parse_separator_grid(
    grid: np.ndarray,
) -> tuple[int, int, tuple[int, int, int, int], list[tuple[int, int]], list[tuple[int, int]]]:
    bg = _most_common_color(grid)
    non_bg = grid[grid != bg]
    assert non_bg.size > 0, "expected non-background content"
    line_color = _most_common_color(non_bg)

    box = _bbox(grid == line_color)
    assert box is not None, "separator grid must exist"
    r0, c0, r1, c1 = box

    sep_rows = [r for r in range(r0, r1 + 1) if np.all(grid[r, c0 : c1 + 1] == line_color)]
    sep_cols = [c for c in range(c0, c1 + 1) if np.all(grid[r0 : r1 + 1, c] == line_color)]
    assert len(sep_rows) >= 4 and len(sep_cols) >= 4, "need at least a 3x3 separator grid"
    assert sep_rows[0] == r0 and sep_rows[-1] == r1
    assert sep_cols[0] == c0 and sep_cols[-1] == c1

    row_spans = _spans_between(sep_rows)
    col_spans = _spans_between(sep_cols)
    _uniform_sizes(row_spans)
    _uniform_sizes(col_spans)
    return bg, line_color, box, row_spans, col_spans


def _read_corner_colors(
    grid: np.ndarray,
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
    bg: int,
    line_color: int,
) -> dict[tuple[int, int], int]:
    last_r = len(row_spans) - 1
    last_c = len(col_spans) - 1
    corners = {}
    for i, j in ((0, 0), (0, last_c), (last_r, 0), (last_r, last_c)):
        rs, re = row_spans[i]
        cs, ce = col_spans[j]
        corners[(i, j)] = _single_color(grid[rs : re + 1, cs : ce + 1], {bg, line_color})
    return corners


def _read_weave_colors(
    grid: np.ndarray,
    box: tuple[int, int, int, int],
    bg: int,
    line_color: int,
) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    coords = np.argwhere((grid != bg) & (grid != line_color))
    outside = [tuple(map(int, pt)) for pt in coords if pt[0] < r0 or pt[0] > r1 or pt[1] < c0 or pt[1] > c1]
    top = [pt for pt in outside if pt[0] < r0]
    right = [pt for pt in outside if pt[1] > c1]
    assert len(outside) == 2, "expected exactly two external marker pixels"
    assert len(top) == 1 and len(right) == 1, "expected one marker above and one marker to the right"
    return int(grid[top[0]]), int(grid[right[0]])


def _apply_role_fills(
    out: np.ndarray,
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
    bg: int,
    corner_colors: dict[tuple[int, int], int],
    weave_a: int,
    weave_b: int,
) -> np.ndarray:
    last_r = len(row_spans) - 1
    last_c = len(col_spans) - 1
    for i, (rs, re) in enumerate(row_spans):
        for j, (cs, ce) in enumerate(col_spans):
            if i in (0, last_r) and j in (0, last_c):
                color = int(corner_colors[(i, j)])
            elif i in (0, last_r) or j in (0, last_c):
                color = int(bg)
            else:
                color = int(weave_a if (i + j) % 2 == 0 else weave_b)
            out[rs : re + 1, cs : ce + 1] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill a separator grid by role: corners inherit their marker color and interior cells weave by parity.

    Input structure:
    - A black background surrounds a one-pixel separator grid drawn in a single dominant line color.
    - The separator rows and columns partition the grid into at least 3x3 uniform cells.
    - Each corner cell contains one colored marker pixel, all non-corner edge cells are empty, and two isolated colored pixels sit outside the grid: one above it and one to its right.

    Rule:
    - Preserve the separator lines and everything outside the cell interiors.
    - Fill each corner cell completely with the color of its existing corner marker.
    - Read weave color A from the external pixel above the grid and weave color B from the external pixel to the right of the grid.
    - Keep every non-corner edge cell as background.
    - For interior cells, use 0-based cell indices: fill with A when row_index + col_index is even, otherwise fill with B.

    Key insight:
    - Two constraints act at once: separator indices determine whether a cell is a corner, edge, or interior, while the interior weave colors are determined only by the spatial roles of the external markers, not by fixed color ids.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg, line_color, box, row_spans, col_spans = _parse_separator_grid(g)
    corner_colors = _read_corner_colors(g, row_spans, col_spans, bg, line_color)
    weave_a, weave_b = _read_weave_colors(g, box, bg, line_color)

    out = g.copy()
    _apply_role_fills(out, row_spans, col_spans, bg, corner_colors, weave_a, weave_b)
    return out


def _choose_weave_colors(rng: random.Random) -> tuple[int, int]:
    if rng.random() < 0.6:
        return (6, 8) if rng.random() < 0.5 else (8, 6)
    a, b = rng.sample(WEAVE_COLOR_POOL, 2)
    return int(a), int(b)


def _sample_layout(rng: random.Random) -> tuple[int, int, int, int, int, int, int] | None:
    cell_count = rng.choice(GRID_COUNT_CHOICES)
    cell_h = rng.randint(*CELL_SIZE_RANGE)
    cell_w = rng.randint(*CELL_SIZE_RANGE)

    grid_h = cell_count * cell_h + cell_count + 1
    grid_w = cell_count * cell_w + cell_count + 1
    extra_h = 30 - grid_h
    extra_w = 30 - grid_w
    if extra_h < TOP_MARGIN_RANGE[0] or extra_w < RIGHT_MARGIN_RANGE[0]:
        return None

    top = rng.randint(TOP_MARGIN_RANGE[0], min(TOP_MARGIN_RANGE[1], extra_h))
    bottom_max = min(SIDE_MARGIN_RANGE[1], extra_h - top)
    if bottom_max < SIDE_MARGIN_RANGE[0]:
        return None
    bottom = rng.randint(SIDE_MARGIN_RANGE[0], bottom_max)

    right = rng.randint(RIGHT_MARGIN_RANGE[0], min(RIGHT_MARGIN_RANGE[1], extra_w))
    left_max = min(SIDE_MARGIN_RANGE[1], extra_w - right)
    if left_max < SIDE_MARGIN_RANGE[0]:
        return None
    left = rng.randint(SIDE_MARGIN_RANGE[0], left_max)

    height = grid_h + top + bottom
    width = grid_w + left + right
    assert height <= 30 and width <= 30
    return cell_count, cell_h, cell_w, top, left, height, width


def _cell_spans_from_layout(cell_count: int, cell_h: int, cell_w: int, top: int, left: int):
    row_lines = [top + i * (cell_h + 1) for i in range(cell_count + 1)]
    col_lines = [left + j * (cell_w + 1) for j in range(cell_count + 1)]
    return _spans_between(row_lines), _spans_between(col_lines), row_lines, col_lines


def _random_cell_position(rng: random.Random, span_r: tuple[int, int], span_c: tuple[int, int]) -> tuple[int, int]:
    return rng.randint(span_r[0], span_r[1]), rng.randint(span_c[0], span_c[1])


def _build_example(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    layout = _sample_layout(rng)
    if layout is None:
        return None
    cell_count, cell_h, cell_w, top, left, height, width = layout
    weave_a, weave_b = _choose_weave_colors(rng)

    inp = np.full((height, width), BG_COLOR, dtype=int)
    row_spans, col_spans, row_lines, col_lines = _cell_spans_from_layout(cell_count, cell_h, cell_w, top, left)
    r0, r1 = row_lines[0], row_lines[-1]
    c0, c1 = col_lines[0], col_lines[-1]

    for r in row_lines:
        inp[r, c0 : c1 + 1] = GRID_LINE_COLOR
    for c in col_lines:
        inp[r0 : r1 + 1, c] = GRID_LINE_COLOR

    corner_colors = {
        (0, 0): CORNER_MARKER_COLORS[(0, 0)],
        (0, cell_count - 1): CORNER_MARKER_COLORS[(0, 1)],
        (cell_count - 1, 0): CORNER_MARKER_COLORS[(1, 0)],
        (cell_count - 1, cell_count - 1): CORNER_MARKER_COLORS[(1, 1)],
    }
    for (i, j), color in corner_colors.items():
        rr, cc = _random_cell_position(rng, row_spans[i], col_spans[j])
        inp[rr, cc] = color

    top_col_choices = [c for c in range(c0 + 1, c1) if c not in col_lines]
    right_row_choices = [r for r in range(r0 + 1, r1) if r not in row_lines]
    assert top_col_choices and right_row_choices
    top_marker_row = rng.randint(0, top - 2)
    top_marker_col = rng.choice(top_col_choices)
    right_marker_row = rng.choice(right_row_choices)
    right_marker_col = rng.randint(c1 + 2, width - 1)
    inp[top_marker_row, top_marker_col] = weave_a
    inp[right_marker_row, right_marker_col] = weave_b

    out = inp.copy()
    _apply_role_fills(out, row_spans, col_spans, BG_COLOR, corner_colors, weave_a, weave_b)
    return inp, out


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    built = _build_example(rng)
    if built is None:
        return None
    inp, expected = built

    if np.array_equal(inp, expected):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid examples after {GEN_RETRIES} attempts")
