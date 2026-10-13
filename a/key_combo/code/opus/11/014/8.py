import random

import numpy as np


TL_COLOR = 2
TR_COLOR = 1
BL_COLOR = 3
BR_COLOR = 4
BG = 0

EVEN_SIZES = (6, 8, 10, 12, 14)

# Restrict to masks whose four mirror/rotation variants are all distinct.
ONE_CELL_MASKS = (
    np.array([[1, 0], [0, 0]], dtype=int),
    np.array([[0, 1], [0, 0]], dtype=int),
    np.array([[0, 0], [1, 0]], dtype=int),
    np.array([[0, 0], [0, 1]], dtype=int),
)
THREE_CELL_MASKS = tuple(1 - mask for mask in ONE_CELL_MASKS)


def _variant(mask: np.ndarray, tile_r: int, tile_c: int) -> np.ndarray:
    if tile_r % 2 == 0 and tile_c % 2 == 0:
        return mask
    if tile_r % 2 == 0 and tile_c % 2 == 1:
        return np.fliplr(mask)
    if tile_r % 2 == 1 and tile_c % 2 == 0:
        return np.flipud(mask)
    return np.rot90(mask, 2)


def _tile_quadrant(pattern: np.ndarray, height: int, width: int) -> np.ndarray:
    out = np.zeros((height, width), dtype=int)
    for tile_r in range(height // 2):
        for tile_c in range(width // 2):
            rr = 2 * tile_r
            cc = 2 * tile_c
            out[rr : rr + 2, cc : cc + 2] = _variant(pattern, tile_r, tile_c)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the four corner 2x2 motifs and fill the doubled canvas quadrants with alternating reflected tilings.

    Input structure:
    - The grid is a black background with even height and width.
    - Only the four 2x2 corner regions contain colored cells: top-left red, top-right blue, bottom-left green, bottom-right yellow.
    - Each corner region contains at least one cell of its corner color, and the interior away from the corners is black.

    Rule:
    - Extract the 2x2 block from each corner.
    - Create an output canvas with twice the input height and twice the input width, so each quadrant matches the original input size.
    - Fill each quadrant by repeating its corresponding corner block over 2x2 tiles, using original / left-right flip / up-down flip / 180-degree rotation according to the tile row and column parities.

    Key insight:
    - The input only stores one sparse 2x2 seed per corner; the output is produced by sending each seed to its matching quadrant and alternating the four reflection states in a checkerboard over tiles.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height % 2 == 0 and width % 2 == 0, "expected even input dimensions"
    assert height >= 4 and width >= 4, "expected enough room for corner crops and mirrored tiling"

    interior = g[2 : height - 2, 2 : width - 2]
    assert np.all(interior == BG), "expected the interior away from corners to be black"

    tl = g[:2, :2].copy()
    tr = g[:2, width - 2 : width].copy()
    bl = g[height - 2 : height, :2].copy()
    br = g[height - 2 : height, width - 2 : width].copy()

    assert np.any(tl == TL_COLOR), "expected the top-left corner motif"
    assert np.any(tr == TR_COLOR), "expected the top-right corner motif"
    assert np.any(bl == BL_COLOR), "expected the bottom-left corner motif"
    assert np.any(br == BR_COLOR), "expected the bottom-right corner motif"

    out = np.zeros((2 * height, 2 * width), dtype=int)
    out[:height, :width] = _tile_quadrant(tl, height, width)
    out[:height, width:] = _tile_quadrant(tr, height, width)
    out[height:, :width] = _tile_quadrant(bl, height, width)
    out[height:, width:] = _tile_quadrant(br, height, width)
    return out


def _render_corner(mask: np.ndarray, color: int) -> np.ndarray:
    return (mask * color).astype(int)


def _distinct_patterns() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    one_cell_choices = random.sample(list(ONE_CELL_MASKS), 2)
    three_cell_choices = random.sample(list(THREE_CELL_MASKS), 2)
    patterns = one_cell_choices + three_cell_choices
    random.shuffle(patterns)
    return tuple(pattern.copy() for pattern in patterns)


def _gen_attempt() -> np.ndarray:
    height = random.choice(EVEN_SIZES)
    width = random.choice(EVEN_SIZES)
    grid = np.zeros((height, width), dtype=int)

    tl_mask, tr_mask, bl_mask, br_mask = _distinct_patterns()
    corners = (
        (slice(0, 2), slice(0, 2), _render_corner(tl_mask, TL_COLOR)),
        (slice(0, 2), slice(width - 2, width), _render_corner(tr_mask, TR_COLOR)),
        (slice(height - 2, height), slice(0, 2), _render_corner(bl_mask, BL_COLOR)),
        (slice(height - 2, height), slice(width - 2, width), _render_corner(br_mask, BR_COLOR)),
    )
    for rs, cs, patch in corners:
        grid[rs, cs] = patch

    # Keep the example visually informative: all four seeds differ and each corner color appears.
    masks = (tl_mask, tr_mask, bl_mask, br_mask)
    assert len({tuple(mask.flatten()) for mask in masks}) == 4
    assert all(np.any(grid == color) for color in (TL_COLOR, TR_COLOR, BL_COLOR, BR_COLOR))

    out = main(grid)
    assert out.shape == (2 * height, 2 * width)
    assert max(out.shape) <= 30
    return grid


def generate_input() -> np.ndarray:
    for _ in range(200):
        try:
            return _gen_attempt()
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid corner-reflection tiling input")
