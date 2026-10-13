from __future__ import annotations

import numpy as np

from arc_utils import Color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Encode the number of distinct input colors as one of three grey lines on black.

    Input structure
    - The input is a 3x3 grid with one, two, or three distinct colors and no
      explicit background.

    Rule
    - One distinct color becomes a grey top row.
    - Two distinct colors become a grey main diagonal.
    - Three distinct colors become a grey anti-diagonal.
    - All non-line pixels become black.

    Key insight
    - The visible color identities do not matter; only the distinct-color count
      selects which 3-cell line to draw.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape == (3, 3), "expected the fixed 3x3 input grid"

    color_count = int(np.unique(g).size)
    assert 1 <= color_count <= 3, "expected one, two, or three distinct colors"

    out = np.full_like(g, Color.BLACK)
    if color_count == 1:
        out[0, :] = Color.GREY
    elif color_count == 2:
        np.fill_diagonal(out, Color.GREY)
    else:
        rows = np.arange(3)
        out[rows, 2 - rows] = Color.GREY
    return out

# --------------- generator (o5x) ---------------

# Input grid and palette constraints.
_GRID_SHAPE = (3, 3)
_CELL_COUNT = _GRID_SHAPE[0] * _GRID_SHAPE[1]
_INPUT_PALETTE = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)
# Distinct-color sampling and retry limits.
_COLOR_COUNT_MIN, _COLOR_COUNT_MAX = 1, 3
_GENERATION_ATTEMPTS = 300
# Output line encoding.
_OUTPUT_LINE_COLOR, _TOP_ROW_INDEX = 5, 0
_MAIN_DIAGONAL_COLOR_COUNT = 2


def _gen_expected_output(color_count: int) -> np.ndarray:
    out = np.zeros(_GRID_SHAPE, dtype=int)
    if color_count == _COLOR_COUNT_MIN:
        out[_TOP_ROW_INDEX, :] = _OUTPUT_LINE_COLOR
    elif color_count == _MAIN_DIAGONAL_COLOR_COUNT:
        np.fill_diagonal(out, _OUTPUT_LINE_COLOR)
    else:
        rows = np.arange(_GRID_SHAPE[0])
        out[rows, _GRID_SHAPE[1] - 1 - rows] = _OUTPUT_LINE_COLOR
    return out


def _gen_attempt() -> np.ndarray | None:
    color_count = int(np.random.randint(_COLOR_COUNT_MIN, _COLOR_COUNT_MAX + 1))
    colors = np.random.choice(_INPUT_PALETTE, size=color_count, replace=False)
    flat = np.empty(_CELL_COUNT, dtype=int)
    flat[:color_count] = colors
    flat[color_count:] = np.random.choice(colors, size=_CELL_COUNT - color_count, replace=True)
    np.random.shuffle(flat)
    return flat.reshape(_GRID_SHAPE)


def generate_input() -> np.ndarray:
    for _ in range(_GENERATION_ATTEMPTS):
        inp = _gen_attempt()
        if inp is None:
            continue
        color_count = int(np.unique(inp).size)
        expected = _gen_expected_output(color_count)
        if np.array_equal(main(inp), expected):
            return inp
    raise RuntimeError("failed to generate a valid 6e02f1e3 input")
