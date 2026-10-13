from __future__ import annotations

import numpy as np

from arc_utils import colors_present, get_bg


def _sign(x: int) -> int:
    return (x > 0) - (x < 0)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each same-colored endpoint pair by filling the diagonal segment between them.

    Input structure:
    - A dominant background surrounds several foreground colors.
    - Each foreground color appears exactly twice.
    - The two pixels of each foreground color are endpoints of a strict 45-degree diagonal segment.

    Rule:
    - Preserve the input pixels.
    - For every foreground color, paint that color on every cell along the diagonal segment connecting its two endpoints.

    Key insight:
    Each color supplies its own two endpoints; the missing cells are just the background cells on the straight diagonal between them.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be 2D"

    bg = get_bg(g)
    colors = colors_present(g, exclude={bg})
    assert colors, "expected at least one foreground endpoint pair"

    out = g.copy()
    for color in colors:
        coords = np.argwhere(g == color)
        assert len(coords) == 2, "each foreground color must have exactly two endpoints"
        (r0, c0), (r1, c1) = coords.astype(int).tolist()

        dr = r1 - r0
        dc = c1 - c0
        assert dr != 0 and abs(dr) == abs(dc), "endpoint pair must define a strict diagonal"

        step_r = _sign(dr)
        step_c = _sign(dc)
        for k in range(abs(dr) + 1):
            r = r0 + k * step_r
            c = c0 + k * step_c
            assert int(out[r, c]) in (bg, color), "diagonal paths must not cross other colors"
            out[r, c] = color

    return out

# --------------- generator (o5x) ---------------

# Canvas, colors, and endpoint-pair structure.
GRID_SIZE, BACKGROUND, FOREGROUND_COLORS = 10, 0, np.arange(1, 10)
MIN_PAIR_COUNT, MAX_PAIR_COUNT = 3, 5
MIN_DIAGONAL_LENGTH, MAX_DIAGONAL_LENGTH = 2, 6
DIAGONAL_STEP_CS = np.array([-1, 1])

# Rejection-sampling budget.
MAX_GENERATION_ATTEMPTS = 300


def _gen_sample_specs():
    n_pairs = int(np.random.randint(MIN_PAIR_COUNT, MAX_PAIR_COUNT + 1))
    colors = np.random.permutation(FOREGROUND_COLORS)[:n_pairs].astype(int).tolist()
    lengths = np.random.randint(
        MIN_DIAGONAL_LENGTH, MAX_DIAGONAL_LENGTH + 1, size=n_pairs
    ).astype(int).tolist()
    step_cs = np.random.choice(DIAGONAL_STEP_CS, size=n_pairs).astype(int).tolist()
    return sorted(zip(colors, lengths, step_cs), key=lambda x: x[1], reverse=True)


def _gen_available_diagonals(occupied, step_c, length):
    size = occupied.shape[0]
    choices = []
    for r0 in range(size - length):
        for c0 in range(size):
            c1 = c0 + step_c * length
            if not 0 <= c1 < size:
                continue
            cells = [(r0 + k, c0 + k * step_c) for k in range(length + 1)]
            if all(not occupied[r, c] for r, c in cells):
                choices.append(cells)
    return choices


def _gen_render_output(specs):
    out = np.full((GRID_SIZE, GRID_SIZE), BACKGROUND, dtype=int)
    occupied = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
    placed = []
    for color, length, step_c in specs:
        choices = _gen_available_diagonals(occupied, step_c, length)
        if not choices:
            return None
        cells = choices[int(np.random.randint(len(choices)))]
        rows, cols = np.array(cells, dtype=int).T
        out[rows, cols] = color
        occupied[rows, cols] = True
        placed.append((color, cells))
    return out, placed


def _gen_attempt() -> np.ndarray | None:
    rendered = _gen_render_output(_gen_sample_specs())
    if rendered is None:
        return None
    out, placed = rendered
    inp = out.copy()
    interiors = [cell for _, cells in placed for cell in cells[1:-1]]
    if interiors:
        rows, cols = np.array(interiors, dtype=int).T
        inp[rows, cols] = BACKGROUND
    try:
        return inp if np.array_equal(main(inp), out) else None
    except Exception:
        return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid 1f876c06 input")
