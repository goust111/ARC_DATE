import random

import numpy as np


BG = 5
PRIMITIVE = 1
MARKER_PALETTE = (2, 3, 4, 6, 7, 8)
MARKER_START_ROW = 4
PRIMITIVE_SIZE = 3
N_OPTIONS = (3, 6)
GRID_SIDE_MIN = 12
GRID_SIDE_MAX = 24
MAX_GENERATION_ATTEMPTS = 400


def _scale_mask(mask: np.ndarray, factor: int) -> np.ndarray:
    return np.kron(mask.astype(bool), np.ones((factor, factor), dtype=bool))


def _marker_colors(grid: np.ndarray, bg: int = BG) -> list[int]:
    colors: list[int] = []
    r = MARKER_START_ROW
    while r < grid.shape[0] and int(grid[r, 0]) != bg:
        colors.append(int(grid[r, 0]))
        r += 1
    if r < grid.shape[0]:
        assert np.all(grid[r:, 0] == bg), "marker run must be contiguous"
    assert colors, "expected at least one row marker"
    assert len(colors) % PRIMITIVE_SIZE == 0, "marker count must be a multiple of 3"
    return colors


def _spec_output(grid: np.ndarray, primitive_mask: np.ndarray, row_colors: list[int]) -> np.ndarray:
    n = len(row_colors)
    scale = n // PRIMITIVE_SIZE
    scaled_mask = _scale_mask(primitive_mask, scale)

    out = np.array(grid, dtype=int, copy=True)
    out[MARKER_START_ROW : MARKER_START_ROW + n, 0] = BG
    out[:n, :n] = BG

    for r, color in enumerate(row_colors):
        row_mask = scaled_mask[r]
        out[r, :n][row_mask] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Scale the corner primitive by the marker count, then recolor each enlarged row with the left-edge palette.

    Input structure:
    - The background is grey.
    - The top-left 3x3 area contains a blue primitive mask over the grey background.
    - Starting at row 4 in column 0, a contiguous vertical run of one-pixel row markers encodes the task.

    Rule:
    - Count the left-edge markers; if there are N markers then the scale factor is S = N / 3.
    - Read the blue/grey primitive mask from the top-left 3x3 area and scale it by S with pixel replication, producing an N x N mask.
    - Place that enlarged mask at the top-left corner and recolor row k of its foreground with the color of marker k.
    - Remove the original marker pixels back to grey and leave all other cells unchanged.

    Key insight:
    - The same marker run carries two coupled signals at once: its length determines the enlargement factor, and its per-row colors determine the recoloring of the enlarged rows.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    marker_colors = _marker_colors(g)
    n = len(marker_colors)
    scale = n // PRIMITIVE_SIZE
    assert scale >= 1
    assert n <= g.shape[0] and n <= g.shape[1], "scaled primitive must fit the grid"

    primitive = g[:PRIMITIVE_SIZE, :PRIMITIVE_SIZE]
    assert set(map(int, primitive.ravel())) <= {BG, PRIMITIVE}, "primitive must be blue on grey"
    primitive_mask = primitive == PRIMITIVE
    assert np.any(primitive_mask), "expected a non-empty primitive"

    return _spec_output(g, primitive_mask, marker_colors)


def _valid_primitive(mask: np.ndarray) -> bool:
    if mask.shape != (PRIMITIVE_SIZE, PRIMITIVE_SIZE):
        return False
    count = int(mask.sum())
    if not 3 <= count <= 7:
        return False
    if np.any(mask.sum(axis=1) == 0) or np.any(mask.sum(axis=0) == 0):
        return False
    if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
        return False
    if len({tuple(row.tolist()) for row in mask}) < 2:
        return False
    if len({tuple(col.tolist()) for col in mask.T}) < 2:
        return False
    if np.array_equal(mask, np.fliplr(mask)):
        return False
    if np.array_equal(mask, np.flipud(mask)):
        return False
    if np.array_equal(mask, np.rot90(mask, 2)):
        return False
    return True


def _random_primitive() -> np.ndarray:
    while True:
        mask = np.zeros((PRIMITIVE_SIZE, PRIMITIVE_SIZE), dtype=bool)
        on_count = random.randint(3, 7)
        for idx in random.sample(range(PRIMITIVE_SIZE * PRIMITIVE_SIZE), on_count):
            mask[idx // PRIMITIVE_SIZE, idx % PRIMITIVE_SIZE] = True
        if _valid_primitive(mask):
            return mask


def _generate_grid_shape(n: int) -> tuple[int, int]:
    min_rows = max(GRID_SIDE_MIN, MARKER_START_ROW + n)
    min_cols = max(GRID_SIDE_MIN, n)
    rows = random.randint(min_rows, GRID_SIDE_MAX)
    cols = random.randint(min_cols, GRID_SIDE_MAX)
    return rows, cols


def _build_input(primitive_mask: np.ndarray, row_colors: list[int], shape: tuple[int, int]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    grid[:PRIMITIVE_SIZE, :PRIMITIVE_SIZE][primitive_mask] = PRIMITIVE
    for offset, color in enumerate(row_colors):
        grid[MARKER_START_ROW + offset, 0] = int(color)
    return grid


def _validate_semantics(inp: np.ndarray, out: np.ndarray, primitive_mask: np.ndarray, row_colors: list[int]) -> None:
    n = len(row_colors)
    scale = n // PRIMITIVE_SIZE
    scaled_mask = _scale_mask(primitive_mask, scale)

    assert inp.ndim == 2 and out.ndim == 2
    assert inp.shape == out.shape
    assert max(inp.shape) <= 30 and max(out.shape) <= 30
    assert inp.shape[0] >= MARKER_START_ROW + n
    assert inp.shape[1] >= n
    assert len(set(row_colors)) == len(row_colors)
    assert set(row_colors) <= set(MARKER_PALETTE)

    primitive = inp[:PRIMITIVE_SIZE, :PRIMITIVE_SIZE]
    assert np.array_equal(primitive == PRIMITIVE, primitive_mask)
    assert np.all(primitive[~primitive_mask] == BG)
    assert np.all(inp[MARKER_START_ROW : MARKER_START_ROW + n, 0] == np.array(row_colors, dtype=int))
    assert int(inp[MARKER_START_ROW - 1, 0]) == BG
    assert np.all(inp[MARKER_START_ROW + n :, 0] == BG)
    assert scaled_mask.shape == (n, n)
    assert np.all(scaled_mask.sum(axis=1) > 0), "every marker row must recolor some foreground"

    expected = _spec_output(inp, primitive_mask, row_colors)
    assert np.array_equal(out, expected)

    untouched = np.ones(inp.shape, dtype=bool)
    untouched[:n, :n] = False
    untouched[MARKER_START_ROW : MARKER_START_ROW + n, 0] = False
    assert np.array_equal(inp[untouched], out[untouched]), "non-participating cells must stay unchanged"

    for r, color in enumerate(row_colors):
        assert np.all(out[r, :n][scaled_mask[r]] == color)
        assert np.all(out[r, :n][~scaled_mask[r]] == BG)


def _gen_attempt() -> np.ndarray | None:
    n = random.choice(N_OPTIONS)
    primitive_mask = _random_primitive()
    row_colors = random.sample(MARKER_PALETTE, n)
    shape = _generate_grid_shape(n)
    inp = _build_input(primitive_mask, row_colors, shape)
    out = main(inp)

    try:
        _validate_semantics(inp, out, primitive_mask, row_colors)
    except AssertionError:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
