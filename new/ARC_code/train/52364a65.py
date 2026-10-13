from __future__ import annotations

import numpy as np

from arc_utils import find_components, get_bg, paste, random_free_location, random_sprite
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase the two leftmost columns of every foreground object, leaving the rest of each object in place.

    Input structure:
    - A grid has one dominant background color and several monochromatic 4-connected foreground objects.
    - Each foreground object spans at least two columns in its bounding box.

    Rule:
    - Find each same-color 4-connected foreground object.
    - Within each object's own bounding box, set foreground pixels in the two leftmost columns to the background color.
    - Leave all other pixels unchanged.

    Key insight:
    - The deletion is measured from each object's local left edge, not from the grid edge or from a shared global column.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    components = find_components(g, bg=bg, connectivity=4, monochromatic=True)
    assert components, "expected at least one foreground object"

    out = g.copy()
    for component in components:
        coords = component["coords"]
        _r0, c0, _r1, c1 = map(int, component["bbox"])
        assert c1 - c0 + 1 >= 2, "each object must span at least two columns"

        erase = coords[coords[:, 1] < c0 + 2]
        out[erase[:, 0], erase[:, 1]] = bg

    return out

# --------------- generator (att1, Claude) ---------------
# Canvas constraints.
BG = 8
GRID_SIZE_RANGE = (8, 11)
MIN_BACKGROUND_FRACTION = 0.50
# Object sampling constraints.
OBJECT_COUNT = 4
FOREGROUND_COLORS = [0, 1, 2, 3, 4, 5, 6, 7, 9]
SPRITE_WIDTH_RANGE = (2, 6)
SPRITE_HEIGHT_RANGE = (2, 5)
SPRITE_PIXEL_RANGE = (4, 14)
SPRITE_RETRIES = 30
SPRITE_CONNECTIVITY = 4
SPRITE_SYMMETRY = "not_symmetric"
# Placement and global retry policy.
PLACEMENT_PADDING = 1
PLACEMENT_CONNECTIVITY = 8
GENERATION_RETRIES = 300

def _sample_sprite(color: int) -> np.ndarray | None:
    """Return one connected sprite that lands inside the configured pixel-count range."""
    bw = random.randint(*SPRITE_WIDTH_RANGE)
    bh = random.randint(*SPRITE_HEIGHT_RANGE)
    max_px = bw * bh
    min_px = max(SPRITE_PIXEL_RANGE[0], bw + bh - 1)
    max_target_px = min(SPRITE_PIXEL_RANGE[1], max_px)
    if min_px > max_target_px:
        return None
    target_px = random.randint(min_px, max_target_px)
    density = target_px / max_px
    for _ in range(SPRITE_RETRIES):
        try:
            sprite = random_sprite(bh, bw, density=density, color_palette=[color],
                                   connectivity=SPRITE_CONNECTIVITY, background=BG,
                                   symmetry=SPRITE_SYMMETRY)
        except Exception:
            continue
        px = int(np.sum(sprite != BG))
        if SPRITE_PIXEL_RANGE[0] <= px <= SPRITE_PIXEL_RANGE[1]:
            return sprite
    return None

def _place_sprite(grid: np.ndarray, sprite: np.ndarray) -> bool:
    """Paste sprite with the required Chebyshev gap; return False if no slot exists."""
    try:
        r, c = random_free_location(grid, sprite, transparent=BG,
                                    padding=PLACEMENT_PADDING,
                                    padding_connectivity=PLACEMENT_CONNECTIVITY)
    except ValueError:
        return False
    paste(grid, sprite, r, c, transparent=BG)
    return True

def _solver_accepts(grid: np.ndarray) -> bool:
    """Return whether the solver accepts the sampled task instance."""
    try:
        out = main(grid)
    except Exception:
        return False
    return out is not None

def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_SIZE_RANGE)
    W = random.randint(*GRID_SIZE_RANGE)
    grid = np.full((H, W), BG, dtype=int)
    for color in random.sample(FOREGROUND_COLORS, OBJECT_COUNT):
        sprite = _sample_sprite(color)
        if sprite is None or not _place_sprite(grid, sprite):
            return None
    if np.sum(grid == BG) / grid.size < MIN_BACKGROUND_FRACTION:
        return None
    return grid if _solver_accepts(grid) else None

def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate input after {GENERATION_RETRIES} attempts")
