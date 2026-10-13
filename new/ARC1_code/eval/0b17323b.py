import numpy as np

from arc_utils import Color, colors_present, get_bg, in_bounds


def main(grid: np.ndarray) -> np.ndarray:
    """
    Continue the equally spaced marker line past its last blue marker, painting the continuation red.

    Input structure
    - The grid has a dominant background color.
    - Exactly one foreground color marks three pixels on a straight arithmetic line.
    - The three marker pixels are equally spaced.

    Rule
    - Order the markers along the line in row-major direction.
    - Use the vector between consecutive markers as the step.
    - Starting one step beyond the last marker, paint each in-bounds background cell red.
    - Stop before the next step would leave the grid.

    Key insight
    - The three blue pixels are not copied; they define the spacing and direction of a line continuation.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    marker_colors = colors_present(g, exclude={bg})
    assert len(marker_colors) == 1
    marker_color = marker_colors[0]
    assert marker_color != Color.RED

    coords = np.argwhere(g == marker_color)
    assert len(coords) == 3
    coords = coords[np.lexsort((coords[:, 1], coords[:, 0]))]

    deltas = np.diff(coords, axis=0)
    assert np.array_equal(deltas[0], deltas[1])
    dr, dc = map(int, deltas[0])
    assert dr != 0 or dc != 0

    out = g.copy()
    r, c = map(int, coords[-1] + deltas[0])
    while in_bounds(g.shape, r, c):
        assert out[r, c] == bg
        out[r, c] = Color.RED
        r += dr
        c += dc

    return out

# --------------- generator (o5x) ---------------

# Canvas and palette.
_GEN_GRID_SIZE = 15
_GEN_MARKER_COLORS = np.array([1, 3, 4, 5, 6, 7, 8, 9], dtype=int)

# Marker line geometry.
_GEN_MARKER_COUNT = 3
_GEN_CONTINUATION_LENGTHS = range(1, 5)
_GEN_DR_VALUES = range(0, 6)
_GEN_DC_VALUES = range(-5, 6)

# Generation retry budget.
_GEN_MAX_ATTEMPTS = 300

def _gen_continuation_len_0b17323b(r, c, dr, dc):
    steps = 0
    r += dr
    c += dc
    while 0 <= r < _GEN_GRID_SIZE and 0 <= c < _GEN_GRID_SIZE:
        steps += 1
        r += dr
        c += dc
    return steps

def _gen_line_choices_0b17323b():
    """Group valid `(dr, dc) -> starts` choices by continuation length."""
    choices = {length: {} for length in _GEN_CONTINUATION_LENGTHS}
    for dr in _GEN_DR_VALUES:
        for dc in _GEN_DC_VALUES:
            if dr == 0 and dc <= 0:
                continue
            for r0 in range(_GEN_GRID_SIZE):
                for c0 in range(_GEN_GRID_SIZE):
                    r2 = r0 + (_GEN_MARKER_COUNT - 1) * dr
                    c2 = c0 + (_GEN_MARKER_COUNT - 1) * dc
                    if not (0 <= r2 < _GEN_GRID_SIZE and 0 <= c2 < _GEN_GRID_SIZE):
                        continue
                    length = _gen_continuation_len_0b17323b(r2, c2, dr, dc)
                    if length in _GEN_CONTINUATION_LENGTHS:
                        choices[length].setdefault((dr, dc), []).append((r0, c0))
    return choices


_GEN_LINE_CHOICES_0B17323B = _gen_line_choices_0b17323b()

def _gen_pick_line_0b17323b():
    """Sample a continuation length, vector, and start using the original weights."""
    length = int(np.random.randint(_GEN_CONTINUATION_LENGTHS.start, _GEN_CONTINUATION_LENGTHS.stop))
    vectors = list(_GEN_LINE_CHOICES_0B17323B[length])
    if not vectors:
        return None

    dr, dc = vectors[int(np.random.randint(len(vectors)))]
    starts = _GEN_LINE_CHOICES_0B17323B[length][(dr, dc)]
    r0, c0 = starts[int(np.random.randint(len(starts)))]
    return length, r0, c0, dr, dc

def _gen_build_grids_0b17323b(r0, c0, dr, dc):
    """Return input, expected output, and continuation length for a sampled line."""
    inp = np.zeros((_GEN_GRID_SIZE, _GEN_GRID_SIZE), dtype=int)
    marker_color = int(_GEN_MARKER_COLORS[int(np.random.randint(len(_GEN_MARKER_COLORS)))])
    for k in range(_GEN_MARKER_COUNT):
        inp[r0 + k * dr, c0 + k * dc] = marker_color
    expected = inp.copy()
    r = r0 + _GEN_MARKER_COUNT * dr
    c = c0 + _GEN_MARKER_COUNT * dc
    painted = 0
    while 0 <= r < _GEN_GRID_SIZE and 0 <= c < _GEN_GRID_SIZE:
        expected[r, c] = Color.RED
        painted += 1
        r += dr
        c += dc
    return inp, expected, painted

def _gen_attempt():
    line = _gen_pick_line_0b17323b()
    if line is None:
        return None

    length, r0, c0, dr, dc = line
    inp, expected, painted = _gen_build_grids_0b17323b(r0, c0, dr, dc)
    if painted != length:
        return None
    try:
        actual = main(inp)
    except AssertionError:
        return None
    return inp if np.array_equal(actual, expected) else None

def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 0b17323b input")
