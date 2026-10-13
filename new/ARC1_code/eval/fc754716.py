import numpy as np

from arc_utils import get_bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the center marker color to draw a one-cell frame around the grid and erase the marker.

    Input structure
    A two-dimensional odd-height, odd-width grid with a single non-background
    marker pixel at the geometric center.

    Rule
    Create a background canvas of the same shape, paint the full outer border
    with the marker color, and leave all interior cells background.

    Key insight
    The center pixel is not kept as an object; it is a control pixel that supplies
    the frame color for the fixed outer-border region.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height % 2 == 1 and width % 2 == 1, "expected odd grid dimensions"
    assert height >= 3 and width >= 3, "expected a nonempty frame interior"

    bg = get_bg(g)
    marker_coords = np.argwhere(g != bg)
    assert len(marker_coords) == 1, "expected exactly one non-background marker"

    marker_row, marker_col = map(int, marker_coords[0])
    assert (marker_row, marker_col) == (
        height // 2,
        width // 2,
    ), "expected marker at the grid center"
    marker_color = int(g[marker_row, marker_col])

    out = np.full_like(g, bg)
    out[0, :] = marker_color
    out[-1, :] = marker_color
    out[:, 0] = marker_color
    out[:, -1] = marker_color
    return out

# --------------- generator (o5x) ---------------

# Canvas sizes, colors, and retry limit for the direct sampler.
GRID_SIZES = (3, 5, 7, 9)
MARKER_COLOR_RANGE = (1, 10)
BACKGROUND_COLOR = 0
MAX_GENERATION_ATTEMPTS = 300


def _gen_attempt() -> np.ndarray:
    rng = np.random.default_rng()
    height = int(rng.choice(GRID_SIZES))
    width = int(rng.choice(GRID_SIZES))
    marker_color = int(rng.integers(*MARKER_COLOR_RANGE))
    grid = np.full((height, width), BACKGROUND_COLOR, dtype=int)
    grid[height // 2, width // 2] = marker_color
    return grid


def _expected_output(inp: np.ndarray) -> np.ndarray:
    marker_color = int(inp[inp.shape[0] // 2, inp.shape[1] // 2])
    expected = np.full_like(inp, BACKGROUND_COLOR)
    expected[[0, -1], :] = marker_color
    expected[:, [0, -1]] = marker_color
    return expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        inp = _gen_attempt()
        if np.array_equal(main(inp), _expected_output(inp)):
            return inp

    raise RuntimeError("failed to generate valid fc754716 input")
