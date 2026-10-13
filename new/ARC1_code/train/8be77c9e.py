import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build a vertically symmetric output by appending the input rows in reverse order.

    Input structure:
    - The input is a non-empty 2D ARC grid.
    - The whole input grid is the source pattern.

    Rule:
    - Keep the input grid as the top half.
    - Place a vertical reflection of the entire input directly below it.

    Key insight:
    The output is not recolored or edited cell-by-cell; it is the original pattern followed by the same rows read bottom-to-top.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "input must be a 2D grid"
    assert g.size > 0, "input grid must be non-empty"
    assert np.all((0 <= g) & (g <= 9)), "grid colors must be ARC color ids"

    return np.vstack((g, g[::-1]))

# --------------- generator (o5x) ---------------

# Canvas/object size bounds.
MIN_GRID_SIZE, MAX_GRID_SIZE = 3, 5
# Color/density sampling and retry limit.
MIN_FOREGROUND_COLOR, MAX_FOREGROUND_COLOR = 1, 9
MIN_FOREGROUND_DENSITY, MAX_FOREGROUND_DENSITY = 0.22, 0.56
GENERATION_RETRY_LIMIT = 300


def _gen_attempt() -> np.ndarray | None:
    h = np.random.randint(MIN_GRID_SIZE, MAX_GRID_SIZE + 1)
    w = np.random.randint(MIN_GRID_SIZE, MAX_GRID_SIZE + 1)
    color = np.random.randint(MIN_FOREGROUND_COLOR, MAX_FOREGROUND_COLOR + 1)

    n_cells = h * w
    min_fg = max(1, int(np.ceil(MIN_FOREGROUND_DENSITY * n_cells)))
    max_fg = max(min_fg, int(np.floor(MAX_FOREGROUND_DENSITY * n_cells)))
    fg_count = np.random.randint(min_fg, max_fg + 1)

    grid = np.zeros((h, w), dtype=int)

    top_r = np.random.randint(0, h // 2)
    asym_c = np.random.randint(0, w)
    grid[top_r, asym_c] = color
    blocked = (h - 1 - top_r, asym_c)

    candidates = [(r, c) for r in range(h) for c in range(w)
                  if (r, c) != (top_r, asym_c) and (r, c) != blocked]
    if fg_count - 1 > len(candidates):
        return None

    if fg_count > 1:
        chosen = np.random.choice(len(candidates), size=fg_count - 1, replace=False)
        for idx in np.atleast_1d(chosen):
            r, c = candidates[int(idx)]
            grid[r, c] = color

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRY_LIMIT):
        inp = _gen_attempt()
        if inp is None:
            continue
        out = np.vstack((inp, inp[::-1]))
        if np.array_equal(main(inp), out):
            return inp
    raise RuntimeError("failed to generate valid input")
