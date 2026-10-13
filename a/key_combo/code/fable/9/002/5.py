from __future__ import annotations

import numpy as np


BACKGROUND = 0
FOREGROUND_COLORS = tuple(range(1, 10))
GENERATION_RETRY_LIMIT = 400


def _quadrant_rotation_steps(size: int, r: int, c: int) -> int:
    """Clockwise quarter-turn count determined by the pixel's quadrant."""
    half = size // 2
    if r < half:
        return 3 if c < half else 0
    return 2 if c < half else 1


def _unique_rotation_count(grid: np.ndarray) -> int:
    seen = {np.rot90(grid, k).tobytes() for k in range(4)}
    return len(seen)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp the seed into each foreground-selected output block, rotating each copy by its source quadrant.

    Input structure:
    - The input is an even-sized square seed grid with black background and exactly one non-black foreground color.
    - The seed itself is also the placement mask: every foreground pixel selects one output block.
    - The geometric center splits the seed into four equal quadrants, so every pixel belongs to exactly one quadrant.

    Rule:
    - Create a black output whose block layout mirrors the input, using one seed-sized block per input cell.
    - For each foreground input pixel, stamp the whole seed into the matching output block.
    - Use the pixel's quadrant to choose the stamp orientation: upper-right keeps the original seed, lower-right rotates it 90 degrees clockwise, lower-left rotates it 180 degrees, and upper-left rotates it 270 degrees clockwise.
    - Leave blocks for background input pixels entirely black.

    Key insight:
    - This is a block-stamping expansion, but the copied seed does not keep one global orientation; each stamped copy inherits a rotation from where its triggering pixel sits relative to the center.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    assert h == w and h % 2 == 0 and h > 0, "expected a non-empty even square seed"

    colors = sorted(int(v) for v in np.unique(g) if int(v) != BACKGROUND)
    assert len(colors) == 1, "expected exactly one foreground color"

    rotated = {k: np.rot90(g, -k) for k in range(4)}
    out = np.full((h * h, w * w), BACKGROUND, dtype=int)

    for r in range(h):
        for c in range(w):
            if int(g[r, c]) == BACKGROUND:
                continue
            steps = _quadrant_rotation_steps(h, r, c)
            out[r * h : (r + 1) * h, c * w : (c + 1) * w] = rotated[steps]

    return out


def _sample_size_2(color: int) -> np.ndarray:
    grid = np.full((2, 2), color, dtype=int)
    missing = int(np.random.randint(4))
    grid.flat[missing] = BACKGROUND
    return grid


def _sample_size_4(color: int) -> np.ndarray:
    grid = np.zeros((4, 4), dtype=int)
    quadrant_starts = ((0, 0), (0, 2), (2, 0), (2, 2))

    for r0, c0 in quadrant_starts:
        cells = np.array([(r0 + dr, c0 + dc) for dr in range(2) for dc in range(2)], dtype=int)
        count = int(np.random.randint(1, 4))
        chosen = cells[np.random.choice(len(cells), size=count, replace=False)]
        grid[chosen[:, 0], chosen[:, 1]] = color

    return grid


def _is_valid_input(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or g.shape[0] != g.shape[1] or g.shape[0] % 2 != 0 or g.shape[0] == 0:
        return False
    if max(g.shape) > 30:
        return False

    colors = sorted(int(v) for v in np.unique(g) if int(v) != BACKGROUND)
    if len(colors) != 1:
        return False
    if not np.any(g == BACKGROUND):
        return False
    if not np.any(g != BACKGROUND):
        return False
    if _unique_rotation_count(g) != 4:
        return False

    size = g.shape[0]
    if size == 4:
        half = 2
        quadrants = (
            g[:half, :half],
            g[:half, half:],
            g[half:, :half],
            g[half:, half:],
        )
        if any(not np.any(block != BACKGROUND) for block in quadrants):
            return False

    try:
        out = main(g)
    except AssertionError:
        return False
    return out.ndim == 2 and max(out.shape) <= 30


def _gen_attempt() -> np.ndarray | None:
    color = int(np.random.choice(FOREGROUND_COLORS))
    size = 2 if np.random.random() < 0.3 else 4
    grid = _sample_size_2(color) if size == 2 else _sample_size_4(color)
    if not _is_valid_input(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for self-spin stamp")
