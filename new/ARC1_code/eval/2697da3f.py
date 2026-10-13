from __future__ import annotations

import numpy as np

from arc_utils import bbox, colors_present, crop_bbox, get_bg, paste
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the foreground sprite and arrange four oriented copies around an empty center square.

    Input structure:
    - A mostly background grid contains exactly one foreground color.
    - The foreground mask has a nonempty bounding box; background holes inside
      that bbox are part of the sprite.

    Rule:
    - Let the cropped sprite have height h and width w.
    - Create a background square of side h + 2*w.
    - Paste the clockwise rotation above the center, the original sprite at the
      left, the horizontal mirror at the right, and the counterclockwise rotation
      below the center.

    Key insight:
    - The input sprite is one side of a four-way reflected/rotated ring around a
      blank h by h center.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    fg_colors = colors_present(g, exclude={bg})
    assert len(fg_colors) == 1, "expected exactly one foreground color"
    fg = fg_colors[0]

    sprite_bbox = bbox(g != bg)
    assert sprite_bbox is not None, "expected a nonempty foreground sprite"
    sprite = crop_bbox(g, sprite_bbox)
    assert np.any(sprite == fg), "sprite must contain foreground pixels"
    assert np.all((sprite == bg) | (sprite == fg)), "sprite may only contain bg and foreground"

    h, w = map(int, sprite.shape)
    out = np.full((h + 2 * w, h + 2 * w), bg, dtype=int)

    paste(out, np.rot90(sprite, -1), 0, w)
    paste(out, sprite, w, 0)
    paste(out, np.fliplr(sprite), w, w + h)
    paste(out, np.rot90(sprite, 1), w + h, w)
    return out

# --------------- generator (o5x) ---------------
# Sprite geometry and fill.
BG_COLOR, FG_COLOR = 0, 4
SPRITE_HEIGHTS = [3, 5]
SPRITE_WIDTH_MIN, SPRITE_WIDTH_MAX = 4, 6
MIN_FILL_DIVISOR, MAX_FILL_NUMERATOR, MAX_FILL_DENOMINATOR = 2, 7, 10
# Canvas size and placement slack.
GRID_SIDE_MIN, GRID_SIDE_MAX = 7, 10
GRID_EXTRA_MIN, GRID_EXTRA_MAX = 1, 5
GEN_ATTEMPTS = 300  # Retry limit.


def _gen_group_size(h: int, r: int) -> int:
    return 1 if r == h // 2 else 2


def _gen_symmetric_sprite() -> np.ndarray | None:
    h = random.choice(SPRITE_HEIGHTS)
    w = random.randint(max(SPRITE_WIDTH_MIN, h), SPRITE_WIDTH_MAX)
    area = h * w
    target_min = (area + MIN_FILL_DIVISOR - 1) // MIN_FILL_DIVISOR
    target = random.randint(target_min, (MAX_FILL_NUMERATOR * area) // MAX_FILL_DENOMINATOR)

    groups = [(r, c) for r in range(h // 2 + 1) for c in range(w)]
    selected = {
        (0, random.randrange(w)),
        (random.randrange(h // 2 + 1), 0),
        (random.randrange(h // 2 + 1), w - 1),
    }
    count = sum(_gen_group_size(h, r) for r, _ in selected)
    need = target - count
    if need < 0:
        return None

    remaining = [g for g in groups if g not in selected]
    singles = [g for g in remaining if _gen_group_size(h, g[0]) == 1]
    pairs = [g for g in remaining if _gen_group_size(h, g[0]) == 2]

    min_pairs = max(0, (need - len(singles) + 1) // 2)
    max_pairs = min(len(pairs), need // 2)
    if min_pairs > max_pairs:
        return None

    pair_count = random.randint(min_pairs, max_pairs)
    single_count = need - 2 * pair_count
    if single_count > len(singles):
        return None

    selected.update(random.sample(pairs, pair_count))
    selected.update(random.sample(singles, single_count))

    sprite = np.full((h, w), BG_COLOR, dtype=int)
    for r, c in selected:
        sprite[r, c] = FG_COLOR
        sprite[h - 1 - r, c] = FG_COLOR

    return sprite


def _gen_expected_output(sprite: np.ndarray) -> np.ndarray:
    h, w = map(int, sprite.shape)
    out = np.full((h + 2 * w, h + 2 * w), BG_COLOR, dtype=int)
    paste(out, np.rot90(sprite, -1), 0, w)
    paste(out, sprite, w, 0)
    paste(out, np.fliplr(sprite), w, w + h)
    paste(out, np.rot90(sprite, 1), w + h, w)
    return out


def _gen_embed_sprite(sprite: np.ndarray) -> np.ndarray:
    """Place the sprite on a random background canvas within configured bounds."""
    h, w = map(int, sprite.shape)
    rows = random.randint(max(GRID_SIDE_MIN, h + GRID_EXTRA_MIN), min(GRID_SIDE_MAX, h + GRID_EXTRA_MAX))
    cols = random.randint(max(GRID_SIDE_MIN, w + GRID_EXTRA_MIN), min(GRID_SIDE_MAX, w + GRID_EXTRA_MAX))
    grid = np.full((rows, cols), BG_COLOR, dtype=int)
    r0 = random.randint(0, rows - h)
    c0 = random.randint(0, cols - w)
    paste(grid, sprite, r0, c0)
    return grid


def _gen_attempt() -> np.ndarray | None:
    sprite = _gen_symmetric_sprite()
    if sprite is None:
        return None

    grid = _gen_embed_sprite(sprite)
    expected = _gen_expected_output(sprite)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 2697da3f input")
