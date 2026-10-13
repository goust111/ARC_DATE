import numpy as np

from arc_utils import Color, get_bg
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each red pixel into a clipped 3x3 blue halo while preserving existing foreground pixels.

    Input structure
    A black-background grid contains one or more red marker pixels and may contain other non-red foreground markers.
    Rule
    For every red pixel, recolor each black cell in its Chebyshev-distance-1 neighborhood to blue; do not overwrite any existing foreground pixel.
    Key insight
    The red pixels are centers of independent 3x3 stamps, clipped by the grid boundary, and the stamp only fills background cells.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = get_bg(g)
    assert bg == Color.BLACK

    red_pixels = np.argwhere(g == Color.RED)
    assert len(red_pixels) > 0

    out = g.copy()
    h, w = g.shape
    for r, c in red_pixels:
        r = int(r)
        c = int(c)
        for rr in range(max(0, r - 1), min(h, r + 2)):
            for cc in range(max(0, c - 1), min(w, c + 2)):
                if out[rr, cc] == bg:
                    out[rr, cc] = Color.BLUE

    return out

# --------------- generator (o5x) ---------------

# CFG: tunable generation parameters.
HEIGHT_RANGE, WIDTH_RANGE = (4, 10), (5, 10)  # Canvas bounds.
RED_COUNT_RANGE, DISTRACTOR_COUNT_RANGE = (1, 3), (0, 3)  # Object counts.
DISTRACTOR_COLORS = tuple(range(3, 10))  # Distractor palette.
RED_SEPARATION, DISTRACTOR_RED_SEPARATION, MAX_EDGE_REDS = 3, 2, 1  # Placement.
GENERATION_RETRIES = 300  # Retry limit.


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*HEIGHT_RANGE)
    w = random.randint(*WIDTH_RANGE)
    n_red = random.randint(*RED_COUNT_RANGE)
    n_distractor = random.randint(*DISTRACTOR_COUNT_RANGE)

    grid = np.zeros((h, w), dtype=int)
    reds: list[tuple[int, int]] = []

    edge_red_count = 0
    for _ in range(n_red):
        candidates = [
            (r, c)
            for r in range(h)
            for c in range(w)
            if all(max(abs(r - rr), abs(c - cc)) >= RED_SEPARATION for rr, cc in reds)
            and (edge_red_count < MAX_EDGE_REDS or not (r in (0, h - 1) or c in (0, w - 1)))
        ]
        if not candidates:
            return None
        r, c = random.choice(candidates)
        grid[r, c] = Color.RED
        reds.append((r, c))
        if r in (0, h - 1) or c in (0, w - 1):
            edge_red_count += 1

    distractor_candidates = [
        (r, c)
        for r in range(h)
        for c in range(w)
        if grid[r, c] == Color.BLACK
        and all(max(abs(r - rr), abs(c - cc)) >= DISTRACTOR_RED_SEPARATION for rr, cc in reds)
    ]
    if len(distractor_candidates) < n_distractor:
        return None

    random.shuffle(distractor_candidates)
    for r, c in distractor_candidates[:n_distractor]:
        grid[r, c] = random.choice(DISTRACTOR_COLORS)

    expected = grid.copy()
    for r, c in reds:
        for rr in range(max(0, r - 1), min(h, r + 2)):
            for cc in range(max(0, c - 1), min(w, c + 2)):
                if expected[rr, cc] == Color.BLACK:
                    expected[rr, cc] = Color.BLUE

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
