from __future__ import annotations

import itertools
import random

import numpy as np


RED = 2
BACKGROUND = 0
NON_RED_COLOR_POOL = (1, 3, 4, 5)
GRID_SHAPE = (3, 3)
MAX_GENERATION_ATTEMPTS = 400


def _scale_sprite(sprite: np.ndarray, factor: int) -> np.ndarray:
    """Nearest-neighbor integer upscaling by pixel replication."""
    k = int(factor)
    if k <= 0:
        raise ValueError("scale factor must be positive")
    return np.kron(sprite, np.ones((k, k), dtype=int))


def _foreground_colors(grid: np.ndarray) -> list[int]:
    colors = sorted(int(color) for color in np.unique(grid) if int(color) != BACKGROUND)
    return colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp the K-scaled whole motif into every red-controlled macro-block.

    Input structure
    A small black-background motif contains 2-4 distinct non-black colors, one
    of which is red. The red pixels are control cells embedded among ordinary
    foreground pixels and background cells.

    Rule
    Let K be the number of distinct non-black colors. First build the exact
    K-times pixel-enlarged copy of the whole input. Then map each input pixel to
    one macro-block of that enlarged size: red pixels receive the full enlarged
    copy, while all other pixels receive a solid block of their own color.

    Key insight
    Palette diversity encodes the scale factor, and the red pixels use that same
    factor to recursively place whole-image copies instead of ordinary color
    enlargements.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    colors = _foreground_colors(g)
    assert len(colors) >= 2, "expected at least two foreground colors"
    assert RED in colors, "expected red control pixels"

    k = len(colors)
    scaled = _scale_sprite(g, k)
    block_h, block_w = scaled.shape

    h, w = g.shape
    out = np.empty((h * block_h, w * block_w), dtype=int)
    for r in range(h):
        for c in range(w):
            rr = slice(r * block_h, (r + 1) * block_h)
            cc = slice(c * block_w, (c + 1) * block_w)
            if int(g[r, c]) == RED:
                out[rr, cc] = scaled
            else:
                out[rr, cc] = int(g[r, c])
    return out


_VALID_RED_POSITION_SETS: list[tuple[int, ...]] = []
for count in (2, 3):
    for positions in itertools.combinations(range(9), count):
        rows = {pos // 3 for pos in positions}
        cols = {pos % 3 for pos in positions}
        if len(rows) >= 2 and len(cols) >= 2:
            _VALID_RED_POSITION_SETS.append(positions)


def _choose_red_positions(red_count: int) -> tuple[int, ...]:
    candidates = [positions for positions in _VALID_RED_POSITION_SETS if len(positions) == red_count]
    return random.choice(candidates)


def _render_grid(
    palette: list[int], red_positions: tuple[int, ...], extra_fg_positions: list[int]
) -> np.ndarray:
    grid = np.zeros(9, dtype=int)
    for pos in red_positions:
        grid[pos] = RED

    non_red_palette = [color for color in palette if color != RED]
    assigned = non_red_palette.copy()
    remaining = len(extra_fg_positions) - len(assigned)
    if remaining > 0:
        assigned.extend(random.choices(non_red_palette, k=remaining))
    random.shuffle(assigned)

    for pos, color in zip(extra_fg_positions, assigned):
        grid[pos] = color
    return grid.reshape(GRID_SHAPE)


def _is_valid_input(grid: np.ndarray) -> bool:
    colors = _foreground_colors(grid)
    if RED not in colors or len(colors) < 2:
        return False
    if grid.shape != GRID_SHAPE:
        return False

    red_coords = np.argwhere(grid == RED)
    if len(red_coords) < 2 or len(red_coords) > 3:
        return False
    if len({int(r) for r, _ in red_coords}) < 2:
        return False
    if len({int(c) for _, c in red_coords}) < 2:
        return False

    out = main(grid)
    if max(out.shape) > 30:
        return False

    non_red_fg = int(np.count_nonzero((grid != BACKGROUND) & (grid != RED)))
    if non_red_fg < len(colors) - 1:
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    k = random.choice((2, 3))
    palette = [RED] + random.sample(NON_RED_COLOR_POOL, k - 1)
    red_count = random.choice((2, 3))
    red_positions = _choose_red_positions(red_count)

    remaining_positions = [pos for pos in range(9) if pos not in red_positions]
    min_other_fg = k - 1
    max_other_fg = min(5, len(remaining_positions))
    other_fg_count = random.randint(min_other_fg, max_other_fg)
    extra_fg_positions = random.sample(remaining_positions, other_fg_count)

    grid = _render_grid(palette, red_positions, extra_fg_positions)
    return grid if _is_valid_input(grid) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
