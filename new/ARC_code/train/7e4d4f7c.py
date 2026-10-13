from __future__ import annotations

import numpy as np

from arc_utils import Color, get_bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress the L-shaped cue to three rows and add a pink copy of the top-row foreground mask.

    Input structure
    The grid has a dominant background color, a two-color top row, and all
    foreground pixels below the top row lie in the first column as vertical markers.

    Rule
    Keep the input top row as output row 0 and the first marker row as output row 1.
    Output row 2 is background except at the top-row foreground columns, which are
    recolored pink.

    Key insight
    The lower repeated marker rows only identify the vertical cue; the generated
    row is a recolored mask of the horizontal cue in the top row.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert h >= 3, "expected room for a three-row summary"

    bg = get_bg(g)
    assert bg != Color.PINK, "pink is the generated marker color, not the background"

    top_colors = set(map(int, np.unique(g[0])))
    assert bg in top_colors, "top row must include background cells"
    top_fg_colors = top_colors - {bg}
    assert len(top_fg_colors) == 1, "top row must contain one foreground color"

    lower_fg = np.argwhere(g[1:] != bg)
    assert lower_fg.size > 0, "expected vertical marker pixels below the top row"
    assert np.all(lower_fg[:, 1] == 0), "lower foreground pixels must be in column 0"
    lower_marker_colors = set(map(int, g[1:][g[1:] != bg]))
    assert len(lower_marker_colors) == 1, "vertical marker cue must use one color"
    assert np.all(g[1, 1:] == bg), "first marker row must be empty except column 0"

    out = np.full((3, w), bg, dtype=int)
    out[0] = g[0]
    out[1] = g[1]
    out[2, g[0] != bg] = Color.PINK
    return out

# --------------- generator (att2) ---------------

_RNG = np.random.default_rng()

# Canvas and vertical cue bounds; RNG upper bounds are exclusive.
WIDTH_RANGE = (5, 14)
MARKER_COUNT_RANGE = (1, 7)
HEIGHT_RANGE = range(3, 13)

# Color, top-row sampling, and oracle retry.
PALETTE = tuple(range(10))
GENERATED_COLOR = Color.PINK
MARKER_MATCH_TOP_PROB = 0.5
TOP_FG_MIN_RATIO = 0.30
TOP_FG_MAX_RATIO = 0.65
MAX_GENERATION_ATTEMPTS = 300


def _colors_excluding(*blocked: int) -> list[int]:
    return [c for c in PALETTE if c not in blocked]


def _gen_attempt() -> np.ndarray:
    width = int(_RNG.integers(*WIDTH_RANGE))
    marker_count = int(_RNG.integers(*MARKER_COUNT_RANGE))
    height_options = [h for h in (2 * marker_count, 2 * marker_count + 1) if h in HEIGHT_RANGE]
    height = int(_RNG.choice(height_options))

    bg = int(_RNG.choice(_colors_excluding(GENERATED_COLOR)))
    top_fg = int(_RNG.choice(_colors_excluding(bg, GENERATED_COLOR)))
    if _RNG.random() < MARKER_MATCH_TOP_PROB:
        marker = top_fg
    else:
        marker = int(_RNG.choice(_colors_excluding(bg, top_fg)))

    min_fg = int(np.ceil(TOP_FG_MIN_RATIO * width))
    max_fg = int(np.floor(TOP_FG_MAX_RATIO * width))
    fg_count = int(_RNG.integers(min_fg, max_fg + 1))
    fg_cols = _RNG.choice(width, size=fg_count, replace=False)

    grid = np.full((height, width), bg, dtype=int)
    grid[0, fg_cols] = top_fg
    grid[1::2, 0] = marker
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        inp = _gen_attempt()
        bg = int(np.bincount(inp.ravel(), minlength=10).argmax())
        expected = np.full((3, inp.shape[1]), bg, dtype=int)
        expected[0] = inp[0]
        expected[1] = inp[1]
        expected[2, inp[0] != bg] = GENERATED_COLOR
        try:
            actual = main(inp)
        except AssertionError:
            continue
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError("failed to generate a valid 7e4d4f7c input")
