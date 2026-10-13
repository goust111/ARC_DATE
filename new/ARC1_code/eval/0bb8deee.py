from __future__ import annotations

import numpy as np

from arc_utils import bbox, colors_present, components_mask, crop_bbox, get_bg, paste
import itertools


def _find_separator_cross(g: np.ndarray, bg: int) -> tuple[int, int, int]:
    height, width = map(int, g.shape)
    fg_colors = colors_present(g, exclude={bg})
    assert len(fg_colors) == 5, "expected four sprite colors plus one separator color"

    solid_rows = [
        (row, color)
        for row in range(height)
        for color in fg_colors
        if np.all(g[row, :] == color)
    ]
    solid_cols = [
        (col, color)
        for col in range(width)
        for color in fg_colors
        if np.all(g[:, col] == color)
    ]
    crosses = [
        (row, col, row_color)
        for row, row_color in solid_rows
        for col, col_color in solid_cols
        if row_color == col_color
    ]
    assert len(crosses) == 1, "expected one same-color solid row/column separator cross"

    row, col, color = crosses[0]
    assert 0 < row < height - 1, "separator row must split the grid"
    assert 0 < col < width - 1, "separator column must split the grid"
    return int(row), int(col), int(color)


def _sprite_crop(region: np.ndarray, bg: int, separator: int) -> np.ndarray:
    assert region.size > 0, "quadrant region must be nonempty"
    sprite_colors = colors_present(region, exclude={bg})
    assert len(sprite_colors) == 1, "each quadrant must contain exactly one sprite color"
    assert sprite_colors[0] != separator, "separator color should not appear inside a quadrant sprite"

    box = bbox(region != bg)
    assert box is not None, "each quadrant must contain one sprite"
    crop = crop_bbox(region, box)
    assert crop.shape == (3, 3), "each sprite must crop to a 3x3 motif"
    return crop


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the four sprites separated by a solid cross and pack them into a 2x2 montage.

    Input structure
    - The background is the dominant color.
    - One non-background color forms a full solid row and a full solid column,
      splitting the grid into four nonempty quadrants.
    - Each quadrant contains exactly one single-color sprite on the background;
      each sprite has a 3x3 tight bounding box.

    Rule
    - Remove the separator row and column.
    - In each quadrant, crop the sprite to its tight bounding box, preserving
      background holes inside the crop.
    - Concatenate the four crops in quadrant order: upper-left, upper-right,
      lower-left, lower-right.

    Key insight
    - The cross is only a layout divider; the output is a compact atlas of the
      four local sprites in their original relative quadrant positions.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)

    bg = get_bg(g)
    cross_row, cross_col, separator = _find_separator_cross(g, bg)

    regions = [
        g[:cross_row, :cross_col],
        g[:cross_row, cross_col + 1 : width],
        g[cross_row + 1 : height, :cross_col],
        g[cross_row + 1 : height, cross_col + 1 : width],
    ]
    crops = [_sprite_crop(region, bg, separator) for region in regions]

    top = np.hstack(crops[:2])
    bottom = np.hstack(crops[2:])
    return np.vstack([top, bottom]).astype(int)

# --------------- generator (o5x) ---------------

# Tunables: sprite masks, canvas bounds, colors, and retry budget.
SPRITE_SIZE = 3
SPRITE_FILLED_RANGE = (4, 7)
SPRITE_MAX_COMPONENTS = 4
CANVAS_HEIGHT_RANGE = (12, 17)
CANVAS_WIDTH_RANGE = (12, 14)
BACKGROUND_COLOR = 0
FOREGROUND_COLORS = np.arange(1, 10)
SPRITE_COUNT = 4
GENERATION_RETRIES = 300


def _gen_full_bbox_masks() -> dict[int, tuple[np.ndarray, ...]]:
    cells = [(r, c) for r in range(SPRITE_SIZE) for c in range(SPRITE_SIZE)]
    full_extent = set(range(SPRITE_SIZE))
    masks_by_count: dict[int, tuple[np.ndarray, ...]] = {}
    for filled in range(*SPRITE_FILLED_RANGE):
        masks = []
        for coords in itertools.combinations(cells, filled):
            rows = {r for r, _ in coords}
            cols = {c for _, c in coords}
            if rows == full_extent and cols == full_extent:
                mask = np.zeros((SPRITE_SIZE, SPRITE_SIZE), dtype=int)
                for r, c in coords:
                    mask[r, c] = 1
                if len(components_mask(mask.astype(bool), connectivity=4)) <= SPRITE_MAX_COMPONENTS:
                    masks.append(mask)
        masks_by_count[filled] = tuple(masks)
    return masks_by_count


_GEN_MASKS_BY_COUNT = _gen_full_bbox_masks()


def _gen_sprite(color: int, rng: np.random.Generator) -> np.ndarray:
    filled = int(rng.integers(*SPRITE_FILLED_RANGE))
    masks = _GEN_MASKS_BY_COUNT[filled]
    mask = masks[int(rng.integers(0, len(masks)))]
    return (mask * int(color)).astype(int)


def _sample_palette(rng: np.random.Generator) -> tuple[int, list[int]]:
    colors = FOREGROUND_COLORS.copy()
    rng.shuffle(colors)
    return int(colors[0]), [int(c) for c in colors[1 : 1 + SPRITE_COUNT]]


def _make_canvas(separator: int, rng: np.random.Generator):
    """Sample the canvas and return it with quadrant boxes as r0/c0/h/w."""
    height = int(rng.integers(*CANVAS_HEIGHT_RANGE))
    width = int(rng.integers(*CANVAS_WIDTH_RANGE))
    cross_row = int(rng.integers(SPRITE_SIZE, height - SPRITE_SIZE))
    cross_col = int(rng.integers(SPRITE_SIZE, width - SPRITE_SIZE))

    grid = np.zeros((height, width), dtype=int)
    grid[cross_row, :] = separator
    grid[:, cross_col] = separator
    quadrants = [
        (0, 0, cross_row, cross_col),
        (0, cross_col + 1, cross_row, width - cross_col - 1),
        (cross_row + 1, 0, height - cross_row - 1, cross_col),
        (cross_row + 1, cross_col + 1, height - cross_row - 1, width - cross_col - 1),
    ]
    return grid, quadrants


def _place_sprites(grid, sprites, quadrants, rng) -> None:
    for sprite, (r0, c0, h, w) in zip(sprites, quadrants):
        rr = int(rng.integers(r0, r0 + h - SPRITE_SIZE + 1))
        cc = int(rng.integers(c0, c0 + w - SPRITE_SIZE + 1))
        paste(grid, sprite, rr, cc, transparent=BACKGROUND_COLOR)


def _passes_oracle(grid: np.ndarray, expected_output: np.ndarray) -> bool:
    try:
        return bool(np.array_equal(main(grid), expected_output))
    except AssertionError:
        return False


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    separator, sprite_colors = _sample_palette(rng)
    sprites = [_gen_sprite(color, rng) for color in sprite_colors]
    output = np.vstack([np.hstack(sprites[:2]), np.hstack(sprites[2:])])
    grid, quadrants = _make_canvas(separator, rng)
    _place_sprites(grid, sprites, quadrants, rng)
    return grid if _passes_oracle(grid, output) else None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        generated = _gen_attempt()
        if generated is not None:
            return generated
    raise RuntimeError("failed to generate a valid 0bb8deee input")
