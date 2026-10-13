from __future__ import annotations

import numpy as np


BACKGROUND = 0
FRAME_COLOR = 3
FOREGROUND_CHOICES = (1, 2, 4, 6, 7, 8, 9)
SIDE_MIN = 3
SIDE_MAX = 5
GEN_ATTEMPTS = 400
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _foreground_color(grid: np.ndarray) -> int:
    colors = [int(color) for color in np.unique(grid) if int(color) != BACKGROUND]
    assert len(colors) == 1, "expected exactly one foreground color"
    return colors[0]


def _invert_binary(block: np.ndarray, fg: int) -> np.ndarray:
    return np.where(block == BACKGROUND, fg, BACKGROUND).astype(int)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble four rotated copies of the binary input, invert the off-diagonal quadrants, and wrap the result in a green frame.

    Input structure:
    - The input is a small square grid.
    - Cells use black background 0 and exactly one non-black foreground color.
    - The whole grid acts as the binary pattern; there are no extra markers or separators.

    Rule:
    - Use the input unchanged in the upper-left quadrant of a 2n x 2n core.
    - Place the 90-degree counterclockwise rotation in the upper-right and invert black/foreground there.
    - Place the 180-degree rotation in the lower-left and invert black/foreground there.
    - Place the 270-degree counterclockwise rotation in the lower-right without inversion, then draw a one-cell green frame around the whole core.

    Key insight:
    - The quadrant layout follows the standard quarter-turn arrangement, but only the checkerboard off-diagonal quadrants are color-inverted before the outer frame is added.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "input must be 2D"
    h, w = g.shape
    assert h == w, "input must be square"
    assert 1 <= h <= 30

    fg = _foreground_color(g)

    core = np.empty((2 * h, 2 * w), dtype=int)
    core[:h, :w] = g
    core[:h, w:] = _invert_binary(np.rot90(g, 1), fg)
    core[h:, :w] = _invert_binary(np.rot90(g, 2), fg)
    core[h:, w:] = np.rot90(g, 3)

    out = np.full((2 * h + 2, 2 * w + 2), FRAME_COLOR, dtype=int)
    out[1:-1, 1:-1] = core
    return out


def _has_quarter_turn_distinction(mask: np.ndarray) -> bool:
    return all(not np.array_equal(mask, np.rot90(mask, k)) for k in (1, 2, 3))


def _connected_frontier(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    frontier: set[tuple[int, int]] = set()
    for r, c in np.argwhere(mask):
        for dr, dc in NEIGHBORS4:
            nr, nc = int(r + dr), int(c + dc)
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                frontier.add((nr, nc))
    return list(frontier)


def _random_connected_mask(side: int, rng: np.random.Generator) -> np.ndarray:
    min_area = max(3, side)
    max_area = side * side - 2
    target_area = int(rng.integers(min_area, max_area + 1))

    mask = np.zeros((side, side), dtype=bool)
    start_r = int(rng.integers(0, side))
    start_c = int(rng.integers(0, side))
    mask[start_r, start_c] = True

    while int(mask.sum()) < target_area:
        frontier = _connected_frontier(mask)
        assert frontier, "frontier unexpectedly empty"
        add_r, add_c = frontier[int(rng.integers(0, len(frontier)))]
        mask[add_r, add_c] = True

    return mask


def _make_pattern(side: int, fg: int, rng: np.random.Generator) -> np.ndarray | None:
    mask = _random_connected_mask(side, rng)
    if not _has_quarter_turn_distinction(mask):
        return None

    if mask[0].all() and mask[-1].all() and mask[:, 0].all() and mask[:, -1].all():
        return None

    grid = np.zeros((side, side), dtype=int)
    grid[mask] = fg
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        side = int(rng.integers(SIDE_MIN, SIDE_MAX + 1))
        fg = int(rng.choice(FOREGROUND_CHOICES))
        grid = _make_pattern(side, fg, rng)
        if grid is None:
            continue
        out = main(grid)
        if grid.ndim == 2 and out.ndim == 2 and max(grid.shape) <= 30 and max(out.shape) <= 30:
            return grid
    raise RuntimeError("failed to generate valid input")
