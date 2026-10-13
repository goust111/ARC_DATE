from __future__ import annotations

import random

import numpy as np

from arc_utils import get_bg


BLUE = 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw a blue folded corner-to-corner route through the lone marker's column.

    Input structure:
    - A rectangular grid filled with one background color.
    - Exactly one non-background marker pixel; blue is absent from the input.

    Rule:
    - Preserve the marker and repaint background pixels in its column blue from top to bottom.
    - If the marker is in the unique center column, connect that column to the top-right and bottom-left corners.
    - Otherwise, connect the column to the top-left and bottom-right corners.

    Key insight:
    The marker's column is the route spine; its central vs off-center placement selects which opposite corner pair gets the horizontal edge segments.
    """
    g = np.asarray(grid, dtype=int)
    H, W = map(int, g.shape)
    assert H >= 2 and W >= 2
    assert not bool(np.any(g == BLUE)), "blue must be introduced by the output"

    bg = get_bg(g)
    marker_cells = np.argwhere(g != bg)
    assert marker_cells.shape == (1, 2), "input must contain exactly one marker pixel"
    r, c = map(int, marker_cells[0])
    marker_color = int(g[r, c])

    out = g.copy()
    out[:, c] = BLUE
    if W % 2 == 1 and c == W // 2:
        out[0, c:] = BLUE
        out[H - 1, : c + 1] = BLUE
    else:
        out[0, : c + 1] = BLUE
        out[H - 1, c:] = BLUE

    out[r, c] = marker_color
    return out

# --------------- generator (att2) ---------------

# Colors and palette constraints.
GEN_BLUE = 1
NON_BLUE_COLORS = [0, 2, 3, 4, 5, 6, 7, 8, 9]
# Branch, canvas, marker placement, and validation tunables.
BRANCH_OPTIONS = [False, True]
CENTER_BRANCH_SIZES = [5, 7, 9]
OFF_CENTER_SIZE_RANGE = (5, 10)
INTERIOR_MARGIN = 1
GENERATION_RETRIES = 300


def _expected_output(inp: np.ndarray) -> np.ndarray:
    out = inp.copy()
    n = out.shape[1]
    marker_cells = np.argwhere(out != out[0, 0])
    r, c = map(int, marker_cells[0])
    marker_color = int(out[r, c])

    out[:, c] = GEN_BLUE
    if n % 2 == 1 and c == n // 2:
        out[0, c:] = GEN_BLUE
        out[-1, : c + 1] = GEN_BLUE
    else:
        out[0, : c + 1] = GEN_BLUE
        out[-1, c:] = GEN_BLUE
    out[r, c] = marker_color
    return out


def _gen_attempt() -> np.ndarray | None:
    use_center_branch = random.choice(BRANCH_OPTIONS)
    if use_center_branch:
        n = random.choice(CENTER_BRANCH_SIZES)
    else:
        n = random.randint(*OFF_CENTER_SIZE_RANGE)

    bg = random.choice(NON_BLUE_COLORS)
    marker_color = random.choice([c for c in NON_BLUE_COLORS if c != bg])

    r = random.randint(INTERIOR_MARGIN, n - INTERIOR_MARGIN - 1)
    if use_center_branch:
        c = n // 2
    else:
        candidates = [
            c
            for c in range(INTERIOR_MARGIN, n - INTERIOR_MARGIN)
            if not (n % 2 == 1 and c == n // 2)
        ]
        c = random.choice(candidates)

    grid = np.full((n, n), bg, dtype=int)
    grid[r, c] = marker_color
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        inp = _gen_attempt()
        if inp is None:
            continue
        out = _expected_output(inp)
        try:
            got = main(inp)
        except Exception:
            continue
        if np.array_equal(got, out):
            return inp
    raise RuntimeError("failed to generate a valid input")
