from __future__ import annotations

import numpy as np

from arc_utils import Color, colors_present, get_bg


def _bar_role(g: np.ndarray, color: int) -> tuple[str, int]:
    coords = np.argwhere(g == int(color))
    assert coords.shape[0] >= 2, "each foreground color must form a bar segment"

    rows = sorted(set(map(int, coords[:, 0])))
    cols = sorted(set(map(int, coords[:, 1])))
    if len(rows) == 1:
        assert len(cols) == coords.shape[0], "horizontal bar cannot branch or overlap"
        assert cols == list(range(cols[0], cols[-1] + 1)), "horizontal bar must be contiguous"
        return "horizontal", rows[0]

    if len(cols) == 1:
        assert len(rows) == coords.shape[0], "vertical bar cannot branch or overlap"
        assert rows == list(range(rows[0], rows[-1] + 1)), "vertical bar must be contiguous"
        return "vertical", cols[0]

    raise AssertionError("foreground color must form a straight horizontal or vertical bar")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend the two perpendicular colored bars across the grid and mark their crossing.

    Input structure
    - A single background color fills the canvas.
    - Exactly two foreground colors appear, each forming one contiguous 1-cell-thick bar segment.
    - One bar is horizontal and one is vertical, and their row/column intersection is background.

    Rule
    - Copy the input.
    - Fill the horizontal bar's entire row with its color and the vertical bar's entire column with its color.
    - Recolor the crossing cell yellow.

    Key insight
    - The short segments define the row and column of the completed cross; the only new color marks their intersection.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    fg_colors = colors_present(g, exclude={bg})
    assert len(fg_colors) == 2, "expected exactly two foreground bar colors"

    horizontal: list[tuple[int, int]] = []
    vertical: list[tuple[int, int]] = []
    for color in fg_colors:
        role, index = _bar_role(g, color)
        if role == "horizontal":
            horizontal.append((color, index))
        else:
            vertical.append((color, index))

    assert len(horizontal) == 1, "expected exactly one horizontal bar"
    assert len(vertical) == 1, "expected exactly one vertical bar"

    h_color, h_row = horizontal[0]
    v_color, v_col = vertical[0]
    assert int(g[h_row, v_col]) == int(bg), "bar intersection must start as background"

    out = g.copy()
    out[h_row, :] = int(h_color)
    out[:, v_col] = int(v_color)
    out[h_row, v_col] = Color.YELLOW
    return out

# --------------- generator (o5x) ---------------

# Canvas bounds: randint uses a high-exclusive upper bound.
GRID_SIZE_RANGE = (5, 9)
# Bar geometry: vertical is top-anchored; horizontal starts below it.
BAR_LENGTH, V_BAR_START_ROW = 2, 0
H_ROW_MIN = V_BAR_START_ROW + BAR_LENGTH
H_ANCHORS = ("left", "right")
# Color and retry controls: yellow is reserved for the output crossing.
EXCLUDED_COLOR, MAX_ATTEMPTS = Color.YELLOW, 300


def _gen_expected_output(inp: np.ndarray, h_row: int, v_col: int, h_color: int, v_color: int) -> np.ndarray:
    out = inp.copy()
    out[h_row, :] = int(h_color)
    out[:, v_col] = int(v_color)
    out[h_row, v_col] = Color.YELLOW
    return out


def _gen_attempt() -> np.ndarray | None:
    height = int(np.random.randint(*GRID_SIZE_RANGE))
    width = int(np.random.randint(*GRID_SIZE_RANGE))

    palette = [c for c in Color.ALL_COLORS if c != EXCLUDED_COLOR]
    bg_color, h_color, v_color = map(int, np.random.choice(palette, size=3, replace=False))

    h_row = int(np.random.randint(H_ROW_MIN, height))
    v_col = int(np.random.randint(0, width))

    anchors: list[str] = []
    if v_col not in range(BAR_LENGTH):
        anchors.append(H_ANCHORS[0])
    if v_col not in range(width - BAR_LENGTH, width):
        anchors.append(H_ANCHORS[1])
    if not anchors:
        return None

    h_anchor = str(np.random.choice(anchors))
    h_cols = (0, BAR_LENGTH - 1) if h_anchor == H_ANCHORS[0] else (width - BAR_LENGTH, width - 1)

    inp = np.full((height, width), bg_color, dtype=int)
    inp[V_BAR_START_ROW : V_BAR_START_ROW + BAR_LENGTH, v_col] = v_color
    inp[h_row, h_cols[0] : h_cols[1] + 1] = h_color

    expected = _gen_expected_output(inp, h_row, v_col, h_color, v_color)
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid bdad9b1f input")
