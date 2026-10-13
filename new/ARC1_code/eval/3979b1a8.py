from __future__ import annotations

import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand the three-color corner-and-cross motif into a 10x10 ordered pattern table.

    Input structure:
    - The input is a 5x5 grid with three colors.
    - The rarest color occupies exactly the four corners.
    - The middle-frequency color is the centered plus; the most common color is the background.

    Rule:
    - Keep the input as the top-left 5x5 block.
    - Form the role sequence [corner color, plus color, background color, corner color, plus color].
    - Copy that sequence across every row of the top-right block and down every row of the bottom-left block.
    - Fill the bottom-right block by taking the later row/column sequence entry off the diagonal, and the next color-role entry on the diagonal.

    Key insight:
    - The output pattern is fixed once the three colors are renamed by their motif roles.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape == (5, 5), "expected a 5x5 motif"

    colors, counts = np.unique(g, return_counts=True)
    assert colors.size == 3, "expected exactly three colors"
    assert len(set(map(int, counts))) == 3, "color frequencies must identify roles"
    by_frequency = np.argsort(counts)
    corner, plus, bg = [int(colors[i]) for i in by_frequency]

    corner_mask = np.zeros((5, 5), dtype=bool)
    corner_mask[0, 0] = True
    corner_mask[0, 4] = True
    corner_mask[4, 0] = True
    corner_mask[4, 4] = True

    plus_mask = np.zeros((5, 5), dtype=bool)
    plus_mask[1:4, 2] = True
    plus_mask[2, 1:4] = True

    assert np.array_equal(g == corner, corner_mask), "rarest color must be only the four corners"
    assert np.array_equal(g == plus, plus_mask), "middle-frequency color must be the centered plus"
    assert np.array_equal(g == bg, ~(corner_mask | plus_mask)), "most common color must fill the remaining cells"

    role_seq = np.array([corner, plus, bg, corner, plus], dtype=int)
    next_role = {corner: plus, plus: bg, bg: corner}

    out = np.empty((10, 10), dtype=int)
    out[:5, :5] = g
    out[:5, 5:] = role_seq
    out[5:, :5] = role_seq[:, None]

    for r in range(5):
        for c in range(5):
            if r == c:
                out[5 + r, 5 + c] = next_role[int(role_seq[r])]
            else:
                out[5 + r, 5 + c] = int(role_seq[max(r, c)])

    return out

# --------------- generator (o5x) ---------------

# CFG: sampling controls.
GEN_COLORS = np.arange(1, 10)
GEN_RETRY_LIMIT = 300
# CFG: fixed motif and output geometry.
GEN_N = 5
GEN_OUT_SHAPE = (10, 10)
GEN_ROLE_ORDER = (0, 1, 2, 0, 1)
GEN_CORNER_CELLS = ((0, 0, 4, 4), (0, 4, 0, 4))
GEN_PLUS_CELLS = ((1, 2, 2, 2, 3), (2, 1, 2, 3, 2))

def _gen_expected_output(inp: np.ndarray, corner: int, plus: int, bg: int) -> np.ndarray:
    roles = (corner, plus, bg)
    role_seq = np.array([roles[i] for i in GEN_ROLE_ORDER], dtype=int)
    next_role = {corner: plus, plus: bg, bg: corner}

    out = np.empty(GEN_OUT_SHAPE, dtype=int)
    out[:GEN_N, :GEN_N] = inp
    out[:GEN_N, GEN_N:] = role_seq
    out[GEN_N:, :GEN_N] = role_seq[:, None]

    block_idx = np.maximum.outer(np.arange(GEN_N), np.arange(GEN_N))
    block = role_seq[block_idx]
    block[np.diag_indices(GEN_N)] = [next_role[int(x)] for x in role_seq]
    out[GEN_N:, GEN_N:] = block
    return out


def _gen_attempt() -> np.ndarray | None:
    corner, plus, bg = [int(x) for x in np.random.choice(GEN_COLORS, size=3, replace=False)]
    inp = np.full((GEN_N, GEN_N), bg, dtype=int)
    inp[GEN_CORNER_CELLS] = corner
    inp[GEN_PLUS_CELLS] = plus

    expected = _gen_expected_output(inp, corner, plus, bg)
    return inp if np.array_equal(main(inp), expected) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
