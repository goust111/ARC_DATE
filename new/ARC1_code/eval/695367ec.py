import numpy as np

from arc_utils import Color


def main(grid: np.ndarray) -> np.ndarray:
    """Draw a fixed 15x15 lattice whose separator spacing is set by the input square size.

    Input structure
    The input is a non-black solid square: every pixel has the same foreground
    color, and the side length of that square is the lattice cell size.

    Rule
    Create a black 15x15 output. Starting after one input-side-length block,
    paint full rows and columns in the input color every side_length + 1 cells.

    Key insight
    The input's content supplies only two parameters: the visible color and the
    spacing between lattice separator lines.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    height, width = g.shape
    assert height == width, "expected a square input"
    assert height > 0, "expected a nonempty input"

    colors = np.unique(g)
    assert colors.size == 1, "expected a solid single-color square"
    line_color = int(colors[0])
    assert line_color != Color.BLACK, "expected a non-black lattice color"

    output_size = 15
    assert height < output_size, "expected the input square to fit inside the 15x15 lattice period"

    out = np.full((output_size, output_size), Color.BLACK, dtype=int)
    separators = np.arange(height, output_size, height + 1)
    assert separators.size > 0, "expected at least one separator line"

    out[separators, :] = line_color
    out[:, separators] = line_color
    return out

# --------------- generator (o5x) ---------------

# Generator configuration: inclusive sampling bounds and retry guard.
INPUT_SIDE_MIN, INPUT_SIDE_MAX = 2, 6
COLOR_MIN, COLOR_MAX = 1, 9
OUTPUT_SIZE, MAX_ATTEMPTS = 15, 300

def _gen_expected_lattice(side, color):
    out = np.zeros((OUTPUT_SIZE, OUTPUT_SIZE), dtype=int)
    separators = np.arange(side, OUTPUT_SIZE, side + 1)
    out[separators, :] = color
    out[:, separators] = color
    return out


def _gen_attempt():
    side = int(np.random.randint(INPUT_SIDE_MIN, INPUT_SIDE_MAX + 1))
    color = int(np.random.randint(COLOR_MIN, COLOR_MAX + 1))
    inp = np.full((side, side), color, dtype=int)

    expected = _gen_expected_lattice(side, color)
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input():
    for _ in range(MAX_ATTEMPTS):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a valid input")
