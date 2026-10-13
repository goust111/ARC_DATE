import numpy as np

from arc_utils import Color, colors_present, coords_where, get_bg, in_bounds


def main(grid: np.ndarray) -> np.ndarray:
    """
    Turn each blue control pixel into a red-centered cross with green diagonal corner marks.

    Input structure:
    - The grid has one dominant background color and one foreground color.
    - The foreground pixels are blue control pixels.
    - The output has the same shape as the input.

    Rule:
    - Paint the full row and full column through every blue control pixel blue.
    - Paint the four in-bounds diagonal neighbors of each control pixel green.
    - Recolor each original control pixel red.

    Key insight:
    - The sparse blue pixels are centers of identical local-and-global marks: global
      row/column rays plus a local diagonal-neighbor decoration.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    foreground_colors = colors_present(g, exclude={bg})
    assert foreground_colors == [Color.BLUE], "expected only blue foreground control pixels"

    controls = [tuple(map(int, rc)) for rc in coords_where(g == Color.BLUE)]
    assert controls, "expected at least one blue control pixel"

    out = np.full_like(g, bg)
    for row, col in controls:
        out[row, :] = Color.BLUE
        out[:, col] = Color.BLUE

    for row, col in controls:
        for dr in (-1, 1):
            for dc in (-1, 1):
                rr = row + dr
                cc = col + dc
                if in_bounds(g.shape, rr, cc):
                    out[rr, cc] = Color.GREEN

    for row, col in controls:
        out[row, col] = Color.RED

    return out

# --------------- generator (o5x) ---------------

# Canvas and color choices.
GRID_SIZE_MIN, GRID_SIZE_MAX_EXCLUSIVE = 9, 15
BACKGROUND_COLORS = [
    c for c in Color.ALL_COLORS
    if c not in (Color.BLUE, Color.RED, Color.GREEN)
]

# Control-pixel structure.
CONTROL_COUNT_MIN, CONTROL_COUNT_MAX_EXCLUSIVE = 2, 5
CONTROL_MIN_SPACING = 2
DIAGONAL_OFFSETS = ((-1, -1), (-1, 1), (1, -1), (1, 1))

# Oracle validation.
GENERATION_RETRY_LIMIT = 300


def _gen_spaced_indices(n: int, k: int) -> list[int]:
    gap = CONTROL_MIN_SPACING - 1
    shifted_n = n - gap * (k - 1)
    shifted = np.random.choice(shifted_n, size=k, replace=False)
    shifted.sort()
    return [int(x) + i * gap for i, x in enumerate(shifted)]


def _sample_control_positions(n: int, count: int) -> list[tuple[int, int]]:
    rows = _gen_spaced_indices(n, count)
    cols = _gen_spaced_indices(n, count)
    np.random.shuffle(cols)
    return list(zip(rows, cols))


def _gen_expected_output(inp: np.ndarray) -> np.ndarray:
    bg = get_bg(inp)
    out = np.full_like(inp, bg)
    controls = np.argwhere(inp == Color.BLUE)

    rows, cols = controls[:, 0], controls[:, 1]
    out[rows, :] = Color.BLUE
    out[:, cols] = Color.BLUE

    h, w = inp.shape
    for row, col in controls:
        for dr, dc in DIAGONAL_OFFSETS:
            rr = row + dr
            cc = col + dc
            if 0 <= rr < h and 0 <= cc < w:
                out[rr, cc] = Color.GREEN

    out[rows, cols] = Color.RED

    return out


def _gen_attempt() -> np.ndarray | None:
    n = int(np.random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX_EXCLUSIVE))
    bg = int(np.random.choice(BACKGROUND_COLORS))
    num_controls = int(np.random.randint(CONTROL_COUNT_MIN, CONTROL_COUNT_MAX_EXCLUSIVE))
    grid = np.full((n, n), bg, dtype=int)
    for row, col in _sample_control_positions(n, num_controls):
        grid[row, col] = Color.BLUE
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRY_LIMIT):
        inp = _gen_attempt()
        if inp is None:
            continue
        try:
            expected = _gen_expected_output(inp)
            if np.array_equal(main(inp), expected):
                return inp
        except Exception:
            continue
    raise RuntimeError("failed to generate a valid 140c817e input")
