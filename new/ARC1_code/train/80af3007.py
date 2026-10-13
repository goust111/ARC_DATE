from __future__ import annotations

import numpy as np

from arc_utils import bbox, colors_present, get_bg
import itertools
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode a 3x3 mask made from solid 3x3 blocks and stamp that mask into each active output block.

    Input structure:
    - The input has one background color and one foreground color.
    - The foreground bounding box is a 9x9 square divided into nine uniform 3x3 tiles.
    - Each tile is either entirely foreground or entirely background, forming a 3x3 mask.

    Rule:
    - Compress the 9x9 source region to its 3x3 on/off mask.
    - Build a 9x9 output by replacing every on cell of that mask with a copy of the whole mask.
    - Replace every off cell of the mask with a blank 3x3 block.

    Key insight:
    - The input is a block-enlarged binary symbol, and the output is the Kronecker product of that symbol with itself.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    fg_colors = colors_present(g, exclude={bg})
    assert len(fg_colors) == 1, "expected exactly one foreground color"
    fg = fg_colors[0]

    box = bbox(g != bg)
    assert box is not None, "expected foreground pixels"
    r0, c0, r1, c1 = box
    crop = g[r0 : r1 + 1, c0 : c1 + 1]
    assert crop.shape == (9, 9), "expected a 9x9 block-encoded source mask"

    mask = np.zeros((3, 3), dtype=bool)
    for br in range(3):
        for bc in range(3):
            tile = crop[3 * br : 3 * br + 3, 3 * bc : 3 * bc + 3]
            is_on = bool(tile[0, 0] == fg)
            expected = fg if is_on else bg
            assert np.all(tile == expected), "each 3x3 source tile must be uniform"
            mask[br, bc] = is_on

    return np.where(np.kron(mask, mask), fg, bg).astype(int)

# --------------- generator (o5x) ---------------
# Fixed colors and block-encoding geometry.
BG_COLOR, FG_COLOR = 0, 5
MASK_SIDE, TILE_SIDE = 3, 3
REGION_SIDE = MASK_SIDE * TILE_SIDE
ON_CELL_RANGE = (5, 7)
# Canvas bounds, padding, and rejection guard.
ROW_RANGE, COL_RANGE = (16, 17), (18, 19)
PLACEMENT_PAD = 1
MAX_ATTEMPTS = 300

def _render_mask(mask: np.ndarray, pattern: np.ndarray) -> np.ndarray:
    return np.where(np.kron(mask, pattern), FG_COLOR, BG_COLOR).astype(int)

def _gen_valid_mask(on_count: int) -> np.ndarray:
    choices = []
    edge_indices = {0, MASK_SIDE - 1}
    for combo in itertools.combinations(range(MASK_SIDE * MASK_SIDE), on_count):
        rows = {i // MASK_SIDE for i in combo}
        cols = {i % MASK_SIDE for i in combo}
        if edge_indices <= rows and edge_indices <= cols:
            choices.append(combo)
    mask = np.zeros((MASK_SIDE, MASK_SIDE), dtype=bool)
    for idx in random.choice(choices):
        mask[idx // MASK_SIDE, idx % MASK_SIDE] = True
    return mask


def _gen_attempt() -> np.ndarray | None:
    mask = _gen_valid_mask(random.randint(*ON_CELL_RANGE))
    region = _render_mask(mask, np.ones((TILE_SIDE, TILE_SIDE), dtype=bool))

    h = random.randint(*ROW_RANGE)
    w = random.randint(*COL_RANGE)
    r0 = random.randint(PLACEMENT_PAD, h - REGION_SIDE - PLACEMENT_PAD)
    c0 = random.randint(PLACEMENT_PAD, w - REGION_SIDE - PLACEMENT_PAD)

    grid = np.full((h, w), BG_COLOR, dtype=int)
    grid[r0 : r0 + REGION_SIDE, c0 : c0 + REGION_SIDE] = region

    expected = _render_mask(mask, mask)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
