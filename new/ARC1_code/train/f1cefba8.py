from __future__ import annotations

import numpy as np

from arc_utils import bbox, colors_present, get_bg, neighbors4
import random


def _solid_foreground_bbox(g: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    fg = g != bg
    b = bbox(fg)
    assert b is not None, "expected a foreground rectangle"
    r0, c0, r1, c1 = b
    assert np.all(fg[r0 : r1 + 1, c0 : c1 + 1]), "expected a solid foreground rectangle"
    return int(r0), int(c0), int(r1), int(c1)


def _outer_inner_colors(
    g: np.ndarray, bg: int, fg_bbox: tuple[int, int, int, int]
) -> tuple[int, int]:
    colors = colors_present(g, exclude={bg})
    assert len(colors) == 2, "expected exactly two foreground colors"

    r0, c0, r1, c1 = fg_bbox
    corner_colors = {int(g[r0, c0]), int(g[r0, c1]), int(g[r1, c0]), int(g[r1, c1])}
    assert len(corner_colors) == 1, "expected one outer color at all rectangle corners"
    outer = corner_colors.pop()
    assert outer in colors, "outer color must be one of the foreground colors"

    inner_colors = [color for color in colors if color != outer]
    assert len(inner_colors) == 1, "expected one inner color"
    return int(outer), int(inner_colors[0])


def _marker_axes(g: np.ndarray, inner_color: int) -> tuple[set[int], set[int]]:
    height, width = map(int, g.shape)
    rows: set[int] = set()
    cols: set[int] = set()
    marker_count = 0

    for r_raw, c_raw in np.argwhere(g == inner_color):
        r = int(r_raw)
        c = int(c_raw)
        same_neighbors = [
            (rr, cc)
            for rr, cc in neighbors4(r, c)
            if 0 <= rr < height and 0 <= cc < width and int(g[rr, cc]) == inner_color
        ]
        if len(same_neighbors) != 1:
            continue

        marker_count += 1
        rr, cc = same_neighbors[0]
        if rr != r:
            cols.add(c)
        else:
            assert cc != c, "marker neighbor must differ in row or column"
            rows.add(r)

    assert marker_count > 0, "expected at least one inner-color leaf marker"
    assert marker_count == len(rows) + len(cols), "expected each marker to select a unique axis"
    return rows, cols


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend the inner-color leaf markers into full row and column cuts through the foreground rectangle.

    Input structure
    - A black background contains one solid rectangular foreground region.
    - The rectangle uses exactly two colors: an outer color at all four corners and an inner color mass.
    - Single-cell leaf protrusions of the inner color stick into the outer-color margin.

    Rule
    - A leaf whose only inner-color neighbor is above or below selects its column; a leaf whose neighbor is left or right selects its row.
    - Selected rows and columns are recolored to the outer color inside the foreground rectangle.
    - The same selected rows and columns are extended through the background outside the rectangle using the inner color.

    Key insight
    - The protruding leaf pixels are not noise: each is a control marker naming one row or column to cut through the rectangle and continue outward.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    r0, c0, r1, c1 = _solid_foreground_bbox(g, bg)
    outer_color, inner_color = _outer_inner_colors(g, bg, (r0, c0, r1, c1))
    selected_rows, selected_cols = _marker_axes(g, inner_color)

    out = g.copy()
    for c in selected_cols:
        assert c0 <= c <= c1, "selected column must cross the foreground rectangle"
        out[:r0, c] = inner_color
        out[r0 : r1 + 1, c] = outer_color
        out[r1 + 1 :, c] = inner_color

    for r in selected_rows:
        assert r0 <= r <= r1, "selected row must cross the foreground rectangle"
        out[r, :c0] = inner_color
        out[r, c0 : c1 + 1] = outer_color
        out[r, c1 + 1 :] = inner_color

    return out

# --------------- generator (o5x) ---------------

# Canvas and rectangle geometry.
BACKGROUND_COLOR = 0
GRID_HEIGHT_RANGE, GRID_WIDTH_RANGE = (13, 19), (13, 18)
RECT_HEIGHT_RANGE, RECT_WIDTH_RANGE = (8, 13), (11, 14)
RECT_MARGIN, FRAME_WIDTH, MARKER_DEPTH = 1, 2, 1

# Marker structure.
ROW_MARKER_COUNT, COL_MARKER_COUNT_RANGE = 1, (1, 2)
ROW_MARKER_SIDES, COL_MARKER_SIDES = ["left", "right"], ["top", "bottom"]
OPPOSITE_COL_SIDE = {"top": "bottom", "bottom": "top"}
MIN_SAME_SIDE_COL_GAP = 3

# Palette and retry limits.
NON_BLACK_COLORS = range(1, 10)
GENERATION_RETRIES = 300


def _gen_feasible_dimensions(marker_count: int) -> tuple[int, int, int, int]:
    """Choose grid and rectangle dimensions that keep black as the background."""
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    choices: list[tuple[int, int]] = []

    for rect_h in range(RECT_HEIGHT_RANGE[0], RECT_HEIGHT_RANGE[1] + 1):
        for rect_w in range(RECT_WIDTH_RANGE[0], RECT_WIDTH_RANGE[1] + 1):
            if height < rect_h + 2 * RECT_MARGIN or width < rect_w + 2 * RECT_MARGIN:
                continue
            fill_area = (rect_h - 2 * FRAME_WIDTH) * (rect_w - 2 * FRAME_WIDTH)
            bg_count = height * width - rect_h * rect_w
            fill_count = fill_area + marker_count
            frame_count = rect_h * rect_w - fill_area - marker_count
            if bg_count > max(fill_count, frame_count):
                choices.append((rect_h, rect_w))

    assert choices, "every allowed grid size should support at least one rectangle"
    rect_h, rect_w = random.choice(choices)
    return height, width, rect_h, rect_w


def _gen_marker_axes(r0: int, c0: int, r1: int, c1: int, col_count: int) -> tuple[list[int], list[int]]:
    """Choose one inner-fill row and the requested number of inner-fill columns."""
    row_choices = list(range(r0 + FRAME_WIDTH, r1 - FRAME_WIDTH + 1))
    col_choices = list(range(c0 + FRAME_WIDTH, c1 - FRAME_WIDTH + 1))
    selected_rows = [random.choice(row_choices)]
    selected_cols = sorted(random.sample(col_choices, col_count))
    return selected_rows, selected_cols


def _gen_marker_sides(selected_rows: list[int], selected_cols: list[int]) -> tuple[dict[int, str], dict[int, str]]:
    """Assign each marker to a frame side, separating close column markers."""
    row_sides = {selected_rows[0]: random.choice(ROW_MARKER_SIDES)}
    col_sides = {col: random.choice(COL_MARKER_SIDES) for col in selected_cols}
    if len(selected_cols) == 2 and selected_cols[1] <= selected_cols[0] + MIN_SAME_SIDE_COL_GAP - 1:
        col_sides[selected_cols[1]] = OPPOSITE_COL_SIDE[col_sides[selected_cols[0]]]
    return row_sides, col_sides


def _gen_build_output(
    height: int,
    width: int,
    rect_h: int,
    rect_w: int,
    r0: int,
    c0: int,
    outer_color: int,
    inner_color: int,
    selected_rows: list[int],
    selected_cols: list[int],
) -> np.ndarray:
    r1 = r0 + rect_h - 1
    c1 = c0 + rect_w - 1
    out = np.zeros((height, width), dtype=int)

    out[r0 : r1 + 1, c0 : c1 + 1] = outer_color
    out[r0 + FRAME_WIDTH : r1 - FRAME_WIDTH + 1, c0 + FRAME_WIDTH : c1 - FRAME_WIDTH + 1] = inner_color

    for col in selected_cols:
        out[:r0, col] = inner_color
        out[r0 : r1 + 1, col] = outer_color
        out[r1 + 1 :, col] = inner_color

    for row in selected_rows:
        out[row, :c0] = inner_color
        out[row, c0 : c1 + 1] = outer_color
        out[row, c1 + 1 :] = inner_color

    return out


def _gen_input_from_output(
    out: np.ndarray,
    rect_h: int,
    rect_w: int,
    r0: int,
    c0: int,
    outer_color: int,
    inner_color: int,
    selected_rows: list[int],
    selected_cols: list[int],
    row_sides: dict[int, str],
    col_sides: dict[int, str],
) -> np.ndarray:
    height, width = map(int, out.shape)
    r1 = r0 + rect_h - 1
    c1 = c0 + rect_w - 1
    inp = out.copy()

    for col in selected_cols:
        inp[:r0, col] = BACKGROUND_COLOR
        inp[r1 + 1 :, col] = BACKGROUND_COLOR
    for row in selected_rows:
        inp[row, :c0] = BACKGROUND_COLOR
        inp[row, c1 + 1 :] = BACKGROUND_COLOR

    inp[r0 : r1 + 1, c0 : c1 + 1] = outer_color
    inp[r0 + FRAME_WIDTH : r1 - FRAME_WIDTH + 1, c0 + FRAME_WIDTH : c1 - FRAME_WIDTH + 1] = inner_color

    for col in selected_cols:
        if col_sides[col] == "top":
            inp[r0 + MARKER_DEPTH, col] = inner_color
        else:
            inp[r1 - MARKER_DEPTH, col] = inner_color

    for row in selected_rows:
        if row_sides[row] == "left":
            inp[row, c0 + MARKER_DEPTH] = inner_color
        else:
            inp[row, c1 - MARKER_DEPTH] = inner_color

    assert inp.shape == (height, width)
    return inp


def _gen_attempt() -> np.ndarray | None:
    col_count = random.randint(*COL_MARKER_COUNT_RANGE)
    marker_count = ROW_MARKER_COUNT + col_count
    height, width, rect_h, rect_w = _gen_feasible_dimensions(marker_count)

    r0 = random.randint(RECT_MARGIN, height - rect_h - RECT_MARGIN)
    c0 = random.randint(RECT_MARGIN, width - rect_w - RECT_MARGIN)
    r1 = r0 + rect_h - 1
    c1 = c0 + rect_w - 1

    outer_color, inner_color = random.sample(NON_BLACK_COLORS, 2)
    selected_rows, selected_cols = _gen_marker_axes(r0, c0, r1, c1, col_count)
    row_sides, col_sides = _gen_marker_sides(selected_rows, selected_cols)

    out = _gen_build_output(
        height,
        width,
        rect_h,
        rect_w,
        r0,
        c0,
        outer_color,
        inner_color,
        selected_rows,
        selected_cols,
    )
    inp = _gen_input_from_output(
        out,
        rect_h,
        rect_w,
        r0,
        c0,
        outer_color,
        inner_color,
        selected_rows,
        selected_cols,
        row_sides,
        col_sides,
    )

    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a valid f1cefba8 input")
