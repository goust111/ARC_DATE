from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, get_bg, paste


TILE_SIZE = 3
STEP = 3
BG_COLOR = int(Color.ORANGE)
ODD_FILL_COLOR = int(Color.RED)
SECOND_FG_COLOR = int(Color.GREY)

GRID_TILE_COUNT_MIN = 4
GRID_TILE_COUNT_MAX = 8
SEED_BG_MIN = 2
SEED_BG_MAX = 4
SEED_MIN_PER_FG_COLOR = 2
GENERATION_RETRY_LIMIT = 400


def _rotate_clockwise(tile: np.ndarray, quarter_turns: int) -> np.ndarray:
    return np.rot90(tile, -int(quarter_turns) % 4).copy()


def _invert_occupancy(tile: np.ndarray, bg: int) -> np.ndarray:
    return np.where(tile == int(bg), ODD_FILL_COLOR, int(bg)).astype(int)


def _step_tile(seed: np.ndarray, step_index: int, bg: int) -> np.ndarray:
    tile = _rotate_clockwise(seed, int(step_index) % 4)
    if int(step_index) % 2 == 1:
        tile = _invert_occupancy(tile, bg)
    return tile


def _tile_signature(tile: np.ndarray) -> tuple[int, ...]:
    return tuple(int(v) for v in tile.reshape(-1))


def _visible_tile_count(length: int) -> int:
    return (int(length) + STEP - 1) // STEP


def _seed_is_structurally_clear(seed: np.ndarray, bg: int) -> bool:
    bg = int(bg)
    red_count = int(np.sum(seed == ODD_FILL_COLOR))
    grey_count = int(np.sum(seed == SECOND_FG_COLOR))
    bg_count = int(np.sum(seed == bg))

    if bg_count < SEED_BG_MIN or bg_count > SEED_BG_MAX:
        return False
    if red_count < SEED_MIN_PER_FG_COLOR or grey_count < SEED_MIN_PER_FG_COLOR:
        return False

    transformed = {_tile_signature(_step_tile(seed, k, bg)) for k in range(4)}
    if len(transformed) != 4:
        return False

    bg_mask = (seed == bg).astype(int)
    bg_rotations = {_tile_signature(_rotate_clockwise(bg_mask, k)) for k in range(4)}
    return len(bg_rotations) == 4


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate the top-left 3x3 seed down the main diagonal while coupling rotation with odd-step occupancy inversion.

    Input structure:
    - The canvas is dominated by orange background.
    - The top-left 3x3 tile is the only seed and mixes red, grey, and orange cells.
    - Everything outside that seed region is background.

    Rule:
    - Read the 3x3 seed from the top-left corner.
    - For the copy placed at offset (3k, 3k), rotate the seed clockwise by (k mod 4) quarter-turns.
    - After rotating, if k is odd, change every background cell in that 3x3 tile to red and erase every non-background cell to background.
    - Stamp each transformed tile with clipping at the canvas border until the next 3x3 start position lies completely outside the grid.

    Key insight:
    - The same diagonal step index k controls both geometry and occupancy, so odd steps are not just rotated copies: they are complements of the rotated tile written in red.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D ARC grid"
    assert g.shape[0] >= TILE_SIZE and g.shape[1] >= TILE_SIZE, "expected at least a 3x3 seed region"

    bg = int(get_bg(g))
    seed = g[:TILE_SIZE, :TILE_SIZE].copy()

    out = np.full_like(g, bg)
    step_index = 0
    while STEP * step_index < g.shape[0] and STEP * step_index < g.shape[1]:
        tile = _step_tile(seed, step_index, bg)
        paste(out, tile, STEP * step_index, STEP * step_index, clip=True)
        step_index += 1
    return out


def _sample_dimension() -> int:
    tile_count = random.randint(GRID_TILE_COUNT_MIN, GRID_TILE_COUNT_MAX)
    # Bias toward partial clipping so the border case appears often across examples.
    return STEP * (tile_count - 1) + random.choice((1, 1, 2, 2, 3))


def _sample_seed() -> np.ndarray:
    positions = list(range(TILE_SIZE * TILE_SIZE))
    while True:
        bg_count = random.randint(SEED_BG_MIN, SEED_BG_MAX)
        bg_positions = set(random.sample(positions, bg_count))
        fg_positions = [pos for pos in positions if pos not in bg_positions]

        red_count = random.randint(SEED_MIN_PER_FG_COLOR, len(fg_positions) - SEED_MIN_PER_FG_COLOR)
        red_positions = set(random.sample(fg_positions, red_count))

        seed = np.full((TILE_SIZE, TILE_SIZE), SECOND_FG_COLOR, dtype=int)
        for pos in bg_positions:
            seed.flat[pos] = BG_COLOR
        for pos in red_positions:
            seed.flat[pos] = ODD_FILL_COLOR

        if _seed_is_structurally_clear(seed, BG_COLOR):
            return seed


def _attempt_input() -> np.ndarray | None:
    height = _sample_dimension()
    width = _sample_dimension()
    grid = np.full((height, width), BG_COLOR, dtype=int)
    grid[:TILE_SIZE, :TILE_SIZE] = _sample_seed()

    out = main(grid)
    visible_tiles = min(_visible_tile_count(height), _visible_tile_count(width))
    if visible_tiles < GRID_TILE_COUNT_MIN:
        return None
    if np.array_equal(out, grid):
        return None
    if int(np.sum(out == SECOND_FG_COLOR)) == 0:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRY_LIMIT):
        grid = _attempt_input()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid diagonal rotate-invert ARC input")
