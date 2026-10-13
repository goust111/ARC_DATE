import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    """Draw a blue lattice on the blank odd square grid, leaving only odd-row/odd-column intersections black.

    Input structure
    A square grid of odd side length is entirely background color 0.

    Rule
    Recolor every pixel whose row index or column index is even to color 1.
    Pixels where both row and column are odd remain background.

    Key insight
    The output is the period-2 grid-line pattern anchored at the top and left edges.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h == w and h % 2 == 1
    assert np.all(g == 0)

    rows, cols = np.indices(g.shape)
    out = np.zeros_like(g)
    out[(rows % 2 == 0) | (cols % 2 == 0)] = 1
    return out

# --------------- generator (o5x) ---------------

# Generator constants: side distribution, lattice phase/colors, and retry budget.
SIDE_CHOICES = np.arange(5, 30, 2)
LATTICE_PHASE, BACKGROUND_COLOR, LINE_COLOR = 0, 0, 1
MAX_GENERATION_ATTEMPTS = 300

def _gen_attempt() -> np.ndarray | None:
    n = int(np.random.choice(SIDE_CHOICES))
    return np.full((n, n), BACKGROUND_COLOR, dtype=int)


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        inp = _gen_attempt()
        if inp is None:
            continue
        rows, cols = np.indices(inp.shape)
        expected = np.full_like(inp, BACKGROUND_COLOR)
        expected[(rows % 2 == LATTICE_PHASE) | (cols % 2 == LATTICE_PHASE)] = LINE_COLOR
        if np.array_equal(main(inp), expected):
            return inp
    raise RuntimeError("failed to generate a valid input")
