from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5


_GLYPH_PATTERNS = {
    1: np.array(
        [
            [GREY, GREY, GREY, GREY],
            [GREY, GREY, GREY, GREY],
            [GREY, GREY, GREY, GREY],
            [GREY, GREY, GREY, GREY],
        ],
        dtype=int,
    ),
    2: np.array(
        [
            [GREY, GREY, GREY, GREY],
            [GREY, BLACK, BLACK, GREY],
            [GREY, BLACK, BLACK, GREY],
            [GREY, GREY, GREY, GREY],
        ],
        dtype=int,
    ),
    3: np.array(
        [
            [GREY, GREY, GREY, GREY],
            [GREY, GREY, GREY, GREY],
            [GREY, BLACK, BLACK, GREY],
            [GREY, BLACK, BLACK, GREY],
        ],
        dtype=int,
    ),
    4: np.array(
        [
            [GREY, GREY, GREY, GREY],
            [BLACK, GREY, GREY, BLACK],
            [BLACK, GREY, GREY, BLACK],
            [GREY, GREY, GREY, GREY],
        ],
        dtype=int,
    ),
}

_GLYPH_TO_QUADRANT = {
    tuple(tuple(int(x) for x in row) for row in glyph): quadrant
    for quadrant, glyph in _GLYPH_PATTERNS.items()
}

_SPRITES = {
    2: [
        np.array([[1, 0, 0], [0, 0, 0], [0, 0, 1]], dtype=int),
        np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]], dtype=int),
    ],
    3: [
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=int),
        np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=int),
        np.array([[1, 0, 1], [0, 0, 0], [0, 1, 0]], dtype=int),
    ],
    4: [
        np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], dtype=int),
        np.array([[1, 0, 0], [0, 1, 0], [1, 0, 1]], dtype=int),
        np.array([[1, 1, 0], [0, 0, 0], [0, 1, 1]], dtype=int),
    ],
}


def _scale_sprite(sprite: np.ndarray, factor: int) -> np.ndarray:
    return np.kron(sprite, np.ones((factor, factor), dtype=int))


def _find_sprite(grid: np.ndarray) -> np.ndarray:
    mask = (grid != BLACK) & (grid != GREY)
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected a colored sprite"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)

    center_r = int((r0 + r1) // 2)
    center_c = int((c0 + c1) // 2)
    top = min(max(center_r - 1, 0), grid.shape[0] - 3)
    left = min(max(center_c - 1, 0), grid.shape[1] - 3)
    sprite = grid[top : top + 3, left : left + 3]
    assert sprite.shape == (3, 3), "expected a 3x3 sprite"
    assert np.any(sprite != BLACK), "sprite must contain foreground"
    return sprite


def _decode_glyphs(grid: np.ndarray) -> list[int]:
    grey_cols = np.where(np.any(grid == GREY, axis=0))[0]
    assert grey_cols.size > 0, "expected at least one glyph"

    groups: list[tuple[int, int]] = []
    start = int(grey_cols[0])
    prev = int(grey_cols[0])
    for col in grey_cols[1:]:
        col = int(col)
        if col == prev + 1:
            prev = col
            continue
        groups.append((start, prev))
        start = prev = col
    groups.append((start, prev))

    quadrants: list[int] = []
    for c0, c1 in groups:
        block = grid[:, c0 : c1 + 1]
        grey_rows = np.where(np.any(block == GREY, axis=1))[0]
        r0 = int(grey_rows[0])
        r1 = int(grey_rows[-1])
        glyph = grid[r0 : r1 + 1, c0 : c1 + 1]
        assert glyph.shape == (4, 4), "expected 4x4 glyphs"
        key = tuple(tuple(int(x) for x in row) for row in glyph)
        assert key in _GLYPH_TO_QUADRANT, "unknown glyph pattern"
        quadrants.append(_GLYPH_TO_QUADRANT[key])
    return quadrants


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Scale a 3x3 sprite by its own foreground count and stamp it into glyph-coded output quadrants.

    Input structure:
    - The grid has a black background with one colored 3x3 sprite on the left and one or more 4x4 gray/black glyphs on the right.
    - The sprite uses a single foreground color and has 2, 3, or 4 non-black pixels.
    - Each glyph is one of four fixed 4x4 gray templates separated from the others by black columns.

    Rule:
    - Extract the 3x3 sprite and count its non-black pixels to get the scale factor N.
    - Enlarge the entire sprite by N using pixel replication, producing a 3N x 3N stamp.
    - Decode every glyph into a quadrant id: 1=upper-left, 2=upper-right, 3=lower-left, 4=lower-right.
    - Create a black 6N x 6N canvas and stamp the enlarged sprite into each decoded quadrant, treating black sprite pixels as transparent.

    Key insight:
    - Two independent codes must be combined: the sprite's foreground count determines the stamp size, while the glyph templates determine the stamp locations.
    """
    grid = np.asarray(input_grid, dtype=int)
    assert grid.ndim == 2, "expected a 2D grid"

    sprite = _find_sprite(grid)
    factor = int(np.count_nonzero(sprite != BLACK))
    assert 2 <= factor <= 4, "foreground count must be between 2 and 4"

    stamp = _scale_sprite(sprite, factor)
    out = np.zeros((6 * factor, 6 * factor), dtype=int)

    quadrant_offsets = {
        1: (0, 0),
        2: (0, 3 * factor),
        3: (3 * factor, 0),
        4: (3 * factor, 3 * factor),
    }
    stamp_mask = stamp != BLACK

    for quadrant in _decode_glyphs(grid):
        r0, c0 = quadrant_offsets[quadrant]
        target = out[r0 : r0 + 3 * factor, c0 : c0 + 3 * factor]
        target[stamp_mask] = stamp[stamp_mask]

    return out


def generate_input() -> np.ndarray:
    height = random.randint(7, 9)
    left_width = random.randint(4, 5)
    region_gap = random.randint(1, 2)

    factor = random.randint(2, 4)
    color = random.choice([BLUE, RED, GREEN])
    sprite = _SPRITES[factor][random.randrange(len(_SPRITES[factor]))] * color

    glyph_count = random.randint(1, 4)
    quadrants = random.sample([1, 2, 3, 4], k=glyph_count)
    random.shuffle(quadrants)

    glyph_gaps = [random.randint(1, 2) for _ in range(glyph_count - 1)]
    right_padding = random.randint(0, 1)
    width = left_width + region_gap + 4 * glyph_count + sum(glyph_gaps) + right_padding
    grid = np.zeros((height, width), dtype=int)

    sprite_top = random.randint(0, height - 3)
    sprite_left = random.randint(0, left_width - 3)
    grid[sprite_top : sprite_top + 3, sprite_left : sprite_left + 3] = sprite

    glyph_top = random.randint(0, height - 4)
    cursor = left_width + region_gap
    for index, quadrant in enumerate(quadrants):
        glyph = _GLYPH_PATTERNS[quadrant]
        grid[glyph_top : glyph_top + 4, cursor : cursor + 4] = glyph
        cursor += 4
        if index < len(glyph_gaps):
            cursor += glyph_gaps[index]

    return grid
