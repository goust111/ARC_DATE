import numpy as np

from arc_utils import Color, colors_present, coords_where, get_bg
import random


def main(grid: np.ndarray) -> np.ndarray:
    """Replace each isolated grey control pixel with a centered 3x3 grey-and-blue motif.

    Input structure
    - The grid background is the dominant color, and every foreground pixel is grey.
    - Each grey pixel is isolated with a full 3x3 background neighborhood around it.
    Rule
    - Erase each grey center to background.
    - Paint the four diagonal neighbors grey and the four orthogonal neighbors blue.
    Key insight
    - The sparse grey pixels are not final objects; they are centers for a fixed 3x3 stencil.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = get_bg(g)
    assert colors_present(g, exclude={bg}) == [Color.GREY]

    centers = coords_where(g == Color.GREY)
    assert len(centers) > 0

    h, w = g.shape
    out = np.full_like(g, bg)
    written = np.zeros(g.shape, dtype=bool)
    values = np.full(g.shape, bg, dtype=int)

    expected_center_window = np.full((3, 3), bg, dtype=int)
    expected_center_window[1, 1] = Color.GREY

    for r0, c0 in centers:
        r, c = int(r0), int(c0)
        assert 0 < r < h - 1 and 0 < c < w - 1
        assert np.array_equal(g[r - 1 : r + 2, c - 1 : c + 2], expected_center_window)

        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if dr == 0 and dc == 0:
                    value = bg
                elif abs(dr) == 1 and abs(dc) == 1:
                    value = Color.GREY
                else:
                    value = Color.BLUE

                if written[rr, cc]:
                    assert values[rr, cc] == value
                values[rr, cc] = value
                written[rr, cc] = True

    out[written] = values[written]
    return out

# --------------- generator (o5x) ---------------

# Canvas, center sampling, and retry constraints.
GRID_SHAPE = (9, 9)
CENTER_COORDS = range(1, 8)
CENTER_COUNT_RANGE = (3, 4)
MIN_CENTER_SEPARATION = 3
MAX_GENERATION_ATTEMPTS = 300
MOTIF = (
    (Color.GREY, Color.BLUE, Color.GREY),
    (Color.BLUE, Color.BLACK, Color.BLUE),
    (Color.GREY, Color.BLUE, Color.GREY),
)


def _select_centers(n_centers):
    candidates = [(r, c) for r in CENTER_COORDS for c in CENTER_COORDS]
    random.shuffle(candidates)
    centers = []
    for r, c in candidates:
        if all(max(abs(r - rr), abs(c - cc)) >= MIN_CENTER_SEPARATION for rr, cc in centers):
            centers.append((r, c))
            if len(centers) == n_centers:
                break
    return centers if len(centers) == n_centers else None


def _gen_attempt() -> np.ndarray | None:
    n_centers = random.randint(*CENTER_COUNT_RANGE)
    centers = _select_centers(n_centers)
    if centers is None:
        return None

    grid = np.zeros(GRID_SHAPE, dtype=int)
    for r, c in centers:
        grid[r, c] = Color.GREY
    out = np.zeros(GRID_SHAPE, dtype=int)
    for r, c in centers:
        out[r - 1 : r + 2, c - 1 : c + 2] = MOTIF

    if not np.array_equal(main(grid), out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid b60334d2 input")
