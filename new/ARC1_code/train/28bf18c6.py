from __future__ import annotations

import numpy as np

from arc_utils import bbox, colors_present, crop_bbox, get_bg, is_contiguous, paste
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the single foreground sprite to its bounding box and repeat it twice horizontally.

    Input structure
    - The input is an 8x8 grid with a dominant background and one non-background
      color forming a 3x3-bbox sprite.

    Rule
    - Crop the full foreground bounding box, preserving background holes inside it.
    - Concatenate two unchanged copies of that crop side by side.

    Key insight
    - The sprite's position in the 8x8 canvas is irrelevant; the output is the
      sprite-local 3x3 pattern duplicated into a 3x6 strip.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape == (8, 8), "expected the 8x8 task canvas"

    bg = get_bg(g)
    fg_colors = colors_present(g, exclude={bg})
    assert len(fg_colors) == 1, "expected exactly one foreground color"

    sprite_bbox = bbox(g != bg)
    assert sprite_bbox is not None, "expected a foreground sprite"
    sprite = crop_bbox(g, sprite_bbox)
    assert sprite.shape == (3, 3), "expected a 3x3 foreground bbox"

    return np.concatenate((sprite, sprite), axis=1)

# --------------- generator (o5x) ---------------

# Canvas and sprite geometry.
CANVAS_SHAPE_28BF18C6 = (8, 8)
SPRITE_SHAPE_28BF18C6 = (3, 3)
SPRITE_CELLS_28BF18C6 = SPRITE_SHAPE_28BF18C6[0] * SPRITE_SHAPE_28BF18C6[1]
ROW_RANGE_28BF18C6 = (0, CANVAS_SHAPE_28BF18C6[0] - SPRITE_SHAPE_28BF18C6[0])
COL_RANGE_28BF18C6 = (0, CANVAS_SHAPE_28BF18C6[1] - SPRITE_SHAPE_28BF18C6[1])

# Sprite sampling and validation.
FOREGROUND_COUNTS_28BF18C6 = tuple(range(4, 7))
COLOR_RANGE_28BF18C6 = (1, 9)
OUTPUT_REPEATS_28BF18C6 = 2
MAX_ATTEMPTS_28BF18C6 = 300


def _gen_valid_masks_28bf18c6() -> dict[int, tuple[np.ndarray, ...]]:
    masks: dict[int, list[np.ndarray]] = {count: [] for count in FOREGROUND_COUNTS_28BF18C6}
    for bits in range(1 << SPRITE_CELLS_28BF18C6):
        cell_count = bits.bit_count()
        if cell_count not in masks:
            continue

        mask = np.zeros(SPRITE_SHAPE_28BF18C6, dtype=int)
        for idx in range(SPRITE_CELLS_28BF18C6):
            if bits & (1 << idx):
                mask[idx // SPRITE_SHAPE_28BF18C6[1], idx % SPRITE_SHAPE_28BF18C6[1]] = 1

        edges = (mask[0, :], mask[-1, :], mask[:, 0], mask[:, -1])
        if not all(edge.any() for edge in edges):
            continue
        if not is_contiguous(mask, background=0, connectivity=8):
            continue
        masks[cell_count].append(mask.astype(bool))

    return {k: tuple(v) for k, v in masks.items()}


_GEN_MASKS_28BF18C6 = _gen_valid_masks_28bf18c6()


def _gen_attempt() -> np.ndarray | None:
    color = random.randint(*COLOR_RANGE_28BF18C6)
    cell_count = random.randint(FOREGROUND_COUNTS_28BF18C6[0], FOREGROUND_COUNTS_28BF18C6[-1])
    mask = random.choice(_GEN_MASKS_28BF18C6[cell_count])

    sprite = np.zeros(SPRITE_SHAPE_28BF18C6, dtype=int)
    sprite[mask] = color

    grid = np.zeros(CANVAS_SHAPE_28BF18C6, dtype=int)
    r0 = random.randint(*ROW_RANGE_28BF18C6)
    c0 = random.randint(*COL_RANGE_28BF18C6)
    paste(grid, sprite, r0, c0)

    expected = np.concatenate((sprite,) * OUTPUT_REPEATS_28BF18C6, axis=1)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS_28BF18C6):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for 28bf18c6")
