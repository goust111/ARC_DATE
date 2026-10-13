import numpy as np

from arc_utils import get_bg
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand the fill-color corner around a single marker into two large diagonal blocks.

    Input structure
    - The input is a 3x3 grid with one background color.
    - The two foreground colors are a single marker pixel and a three-pixel fill color.
    - The marker and fill pixels occupy a 2x2 square, with the marker at one corner.

    Rule
    - Find the diagonal direction from the marker to the opposite fill pixel.
    - Make a 9x9 background canvas.
    - Paint one 4x4 fill-color block in the matching output corner and another
      4x4 fill-color block one step inward along that diagonal.

    Key insight
    - The missing corner of the 2x2 fill square is the marker; its orientation
      chooses which diagonal staircase of two enlarged blocks is drawn.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape == (3, 3), "expected a 3x3 input grid"

    bg = get_bg(g)
    colors = [int(color) for color in np.unique(g) if int(color) != bg]
    assert len(colors) == 2, "expected exactly two foreground colors"

    counts = {color: int(np.count_nonzero(g == color)) for color in colors}
    fill_colors = [color for color, count in counts.items() if count == 3]
    marker_colors = [color for color, count in counts.items() if count == 1]
    assert len(fill_colors) == 1, "expected one three-pixel fill color"
    assert len(marker_colors) == 1, "expected one single-pixel marker color"
    fill = fill_colors[0]
    marker = marker_colors[0]

    marker_coords = np.argwhere(g == marker)
    assert marker_coords.shape == (1, 2), "expected a unique marker pixel"
    mr, mc = map(int, marker_coords[0])
    fill_coords = {tuple(map(int, coord)) for coord in np.argwhere(g == fill)}

    opposite = [(r, c) for r, c in fill_coords if r != mr and c != mc]
    assert len(opposite) == 1, "expected one fill pixel diagonally opposite the marker"
    orow, ocol = opposite[0]
    dr = 1 if orow > mr else -1
    dc = 1 if ocol > mc else -1

    expected_fill = {(mr + dr, mc), (mr, mc + dc), (mr + dr, mc + dc)}
    assert fill_coords == expected_fill, "fill pixels must form the other cells of the marker's 2x2 square"

    out = np.full((9, 9), bg, dtype=int)
    block_starts = [
        (0 if dr < 0 else 5, 0 if dc < 0 else 5),
        (4 if dr < 0 else 1, 4 if dc < 0 else 1),
    ]
    for r, c in block_starts:
        out[r : r + 4, c : c + 4] = fill
    return out

# --------------- generator (o5x) ---------------

# Palette sampling and retry policy.
COLOR_VALUES, ROLE_COLOR_COUNT = tuple(range(10)), 3
DIRECTION_STEPS, MAX_ATTEMPTS = (-1, 1), 300
# Grid geometry and role-count constraints.
INPUT_SHAPE, OUTPUT_SHAPE, CENTER = (3, 3), (9, 9), 1
FILL_CELLS, MARKER_CELLS, BLOCK_SIZE = 3, 1, 4
BLOCK_STARTS_BY_STEP = {-1: (0, 4), 1: (5, 1)}


def _gen_expected_output(inp: np.ndarray) -> np.ndarray:
    colors, counts = np.unique(inp, return_counts=True)
    bg = int(colors[np.argmax(counts)])
    fill = int(colors[np.where(counts == FILL_CELLS)[0][0]])
    marker = int(colors[np.where(counts == MARKER_CELLS)[0][0]])

    mr, mc = map(int, np.argwhere(inp == marker)[0])
    opposite = [(r, c) for r, c in np.argwhere(inp == fill) if r != mr and c != mc][0]
    dr = 1 if int(opposite[0]) > mr else -1
    dc = 1 if int(opposite[1]) > mc else -1

    out = np.full(OUTPUT_SHAPE, bg, dtype=int)
    row_starts = BLOCK_STARTS_BY_STEP[dr]
    col_starts = BLOCK_STARTS_BY_STEP[dc]
    for r, c in ((row_starts[0], col_starts[0]), (row_starts[1], col_starts[1])):
        out[r : r + BLOCK_SIZE, c : c + BLOCK_SIZE] = fill
    return out


def _gen_attempt() -> np.ndarray:
    bg, fill, marker = random.sample(COLOR_VALUES, ROLE_COLOR_COUNT)
    dr, dc = random.choice(DIRECTION_STEPS), random.choice(DIRECTION_STEPS)

    inp = np.full(INPUT_SHAPE, bg, dtype=int)
    inp[CENTER, CENTER] = marker
    inp[CENTER + dr, CENTER] = fill
    inp[CENTER, CENTER + dc] = fill
    inp[CENTER + dr, CENTER + dc] = fill
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        inp = _gen_attempt()
        expected = _gen_expected_output(inp)
        try:
            actual = main(inp)
        except AssertionError:
            continue
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError("failed to generate a valid 4522001f input")
