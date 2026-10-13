from __future__ import annotations

import numpy as np

from arc_utils import get_bg
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp the source sprite into each marked right-side slot while leaving the divider and empty slots unchanged.

    Input structure
    - A dominant background surrounds a full-height monochrome divider column.
    - The source sprite is the non-background panel immediately left of the divider.
    - The area right of the divider is partitioned into same-sized slots; marked slots contain one centered control pixel.

    Rule
    - Copy the left sprite into every right-side slot whose center contains the marker.
    - Leave unmarked slots, the divider, and the original source sprite unchanged.

    Key insight
    The marker does not supply a color; it selects a slot to be replaced by the left-side sprite template.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    h, w = g.shape
    bg = int(get_bg(g))

    divider_cols = []
    for c in range(w):
        col = g[:, c]
        if np.all(col == col[0]) and int(col[0]) != bg:
            divider_cols.append(c)
    assert len(divider_cols) == 1, "expected one full-height non-background divider"

    divider_col = divider_cols[0]
    assert 0 < divider_col < w - 1, "divider must separate source sprite from target slots"

    left_panel = g[:, :divider_col]
    sprite_rows = np.where(np.any(left_panel != bg, axis=1))[0]
    assert sprite_rows.size > 0, "expected a source sprite left of the divider"
    sprite_r0 = int(sprite_rows[0])
    sprite_r1 = int(sprite_rows[-1])
    assert sprite_r0 == 0, "source sprite must start at the top of the left panel"

    sprite = left_panel[: sprite_r1 + 1, :]
    slot_h, slot_w = sprite.shape
    assert slot_h % 2 == 1 and slot_w % 2 == 1, "marker slot needs a single center"
    assert slot_w == divider_col, "left panel width defines the slot width"
    assert h % slot_h == 0, "grid height must partition into sprite-height slots"
    assert (w - divider_col - 1) % slot_w == 0, "right area must partition into sprite-width slots"

    right_panel = g[:, divider_col + 1 :]
    marker_colors = set(map(int, np.unique(right_panel))) - {bg}
    assert len(marker_colors) == 1, "right panel should contain exactly one marker color"
    marker = marker_colors.pop()
    center = (slot_h // 2, slot_w // 2)

    out = g.copy()
    for r0 in range(0, h, slot_h):
        for c0 in range(divider_col + 1, w, slot_w):
            slot = g[r0 : r0 + slot_h, c0 : c0 + slot_w]
            marker_cells = np.argwhere(slot == marker)
            if marker_cells.size == 0:
                assert np.all(slot == bg), "unmarked slots must be empty background"
                continue

            assert marker_cells.shape == (1, 2), "marked slot should contain exactly one marker"
            marker_pos = (int(marker_cells[0, 0]), int(marker_cells[0, 1]))
            assert marker_pos == center, "marker must be centered in its slot"
            assert np.count_nonzero(slot != bg) == 1, "marked slot should otherwise be empty"
            out[r0 : r0 + slot_h, c0 : c0 + slot_w] = sprite

    return out

# --------------- generator (o5x) ---------------

# Geometry and fixed colors.
SPRITE_SIDE, SLOT_ROWS, SLOT_COLS = 3, 3, 3
BG_COLOR, DIVIDER_COLOR, MARKER_COLOR = 0, 5, 1
DIVIDER_COL = SPRITE_SIDE
CANVAS_SHAPE = (SPRITE_SIDE * SLOT_ROWS, SPRITE_SIDE * SLOT_COLS + SPRITE_SIDE + 1)
SLOT_COUNT = SLOT_ROWS * SLOT_COLS
SLOT_CENTER = SPRITE_SIDE // 2

# Sprite color sampling.
SPRITE_PALETTE = [2, 3, 4, 6, 7, 8, 9]
SPRITE_COLOR_COUNT_RANGE = (3, 5)

# Marker and retry sampling.
MARKED_SLOT_COUNT_RANGE = (3, 6)
MAX_GENERATION_ATTEMPTS = 300


def _build_sprite() -> np.ndarray:
    color_count = random.randint(*SPRITE_COLOR_COUNT_RANGE)
    sprite_colors = random.sample(SPRITE_PALETTE, color_count)
    extra_count = SPRITE_SIDE * SPRITE_SIDE - color_count
    cells = sprite_colors + [random.choice(sprite_colors) for _ in range(extra_count)]
    random.shuffle(cells)
    return np.array(cells, dtype=int).reshape(SPRITE_SIDE, SPRITE_SIDE)


def _place_markers(grid: np.ndarray, sprite: np.ndarray) -> np.ndarray:
    """Sample markers in-place and return the expected solver output."""
    expected = grid.copy()
    marked_count = random.randint(*MARKED_SLOT_COUNT_RANGE)
    for slot_i in random.sample(range(SLOT_COUNT), marked_count):
        sr, sc = divmod(slot_i, SLOT_COLS)
        r0 = sr * SPRITE_SIDE
        c0 = DIVIDER_COL + 1 + sc * SPRITE_SIDE
        grid[r0 + SLOT_CENTER, c0 + SLOT_CENTER] = MARKER_COLOR
        expected[r0 : r0 + SPRITE_SIDE, c0 : c0 + SPRITE_SIDE] = sprite
    return expected


def _gen_attempt() -> np.ndarray | None:
    sprite = _build_sprite()
    grid = np.full(CANVAS_SHAPE, BG_COLOR, dtype=int)
    grid[:SPRITE_SIDE, :SPRITE_SIDE] = sprite
    grid[:, DIVIDER_COL] = DIVIDER_COLOR
    expected = _place_markers(grid, sprite)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    return grid if np.array_equal(actual, expected) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
