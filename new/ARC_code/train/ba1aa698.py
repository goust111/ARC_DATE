import math
from collections import Counter
from fractions import Fraction, Fraction as _gen_Fraction

import numpy as np

from arc_utils import bbox, is_contiguous, paste
import itertools as _gen_itertools
import math as _gen_math
import random as _gen_random


def _unique_most_common(values) -> int:
    counts = Counter(map(int, values))
    ranked = counts.most_common()
    assert ranked and (len(ranked) == 1 or ranked[0][1] > ranked[1][1])
    return int(ranked[0][0])


def _round_half_up(value: Fraction) -> int:
    return math.floor(value + Fraction(1, 2))


def _extract_sprite(panel: np.ndarray, fill: int) -> tuple[np.ndarray, np.ndarray, int, int]:
    interior_mask = panel[1:-1, 1:-1] != fill
    box = bbox(interior_mask)
    assert box is not None
    r0, local_c0, r1, local_c1 = box
    r0 += 1
    r1 += 1
    local_c0 += 1
    local_c1 += 1

    sprite = panel[r0 : r1 + 1, local_c0 : local_c1 + 1].copy()
    sprite_mask = sprite != fill
    assert np.count_nonzero(sprite_mask) == np.count_nonzero(interior_mask)
    return sprite, sprite_mask, int(r0), int(local_c0)


def main(grid: np.ndarray) -> np.ndarray:
    """Extrapolate the moving sprite from a horizontal sequence of shared-border frames into the next standalone frame.

    Input structure
    - The grid is a row of equal-width framed panels sharing their vertical border columns.
    - Each panel has the same fill color and one identical interior sprite at the same local columns.
    - The sprite's top row changes by a constant amount from panel to panel.

    Rule
    - Build one output panel with the same height, frame color, fill color, and sprite.
    - Continue the sprite's diagonal motion one more panel to the right.
    - Because the output is a standalone panel, advance by the full panel width, not the shared-border spacing.

    Key insight
    - The panels are consecutive frames of motion; the output is the next frame after the sequence.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    frame = int(g[0, 0])
    assert h >= 3 and w >= 3
    assert np.all(g[0, :] == frame)
    assert np.all(g[-1, :] == frame)

    separators = [c for c in range(w) if np.all(g[:, c] == frame)]
    assert len(separators) >= 3
    assert separators[0] == 0 and separators[-1] == w - 1
    strides = [b - a for a, b in zip(separators, separators[1:])]
    assert len(set(strides)) == 1
    stride = strides[0]
    panel_w = stride + 1

    interior_values = []
    for c0, c1 in zip(separators, separators[1:]):
        interior_values.extend(g[1:-1, c0 + 1 : c1].ravel())
    fill = _unique_most_common(interior_values)

    panels = []
    for c0, c1 in zip(separators, separators[1:]):
        panel = g[:, c0 : c1 + 1]
        assert panel.shape == (h, panel_w)
        assert np.all(panel[0, :] == frame)
        assert np.all(panel[-1, :] == frame)
        assert np.all(panel[:, 0] == frame)
        assert np.all(panel[:, -1] == frame)
        panels.append(panel)

    template_sprite, template_mask, first_top, template_left = _extract_sprite(panels[0], fill)
    top_rows = [first_top]
    for panel in panels[1:]:
        sprite, sprite_mask, top, local_c0 = _extract_sprite(panel, fill)
        assert local_c0 == template_left
        assert sprite_mask.shape == template_mask.shape
        assert np.array_equal(sprite_mask, template_mask)
        assert np.array_equal(sprite[sprite_mask], template_sprite[template_mask])
        top_rows.append(top)

    deltas = [b - a for a, b in zip(top_rows, top_rows[1:])]
    assert len(deltas) >= 1 and len(set(deltas)) == 1
    next_top = _round_half_up(
        Fraction(top_rows[-1], 1) + Fraction(deltas[0] * panel_w, stride)
    )

    sprite_h, sprite_w = template_sprite.shape
    assert 1 <= next_top and next_top + sprite_h <= h - 1
    assert 1 <= template_left and template_left + sprite_w <= panel_w - 1

    out = np.full((h, panel_w), fill, dtype=int)
    out[0, :] = frame
    out[-1, :] = frame
    out[:, 0] = frame
    out[:, -1] = frame

    target = out[next_top : next_top + sprite_h, template_left : template_left + sprite_w]
    target[template_mask] = template_sprite[template_mask]
    return out

# --------------- generator (att2) ---------------
_GEN_HEIGHT, _GEN_LOCAL_LEFT = 16, 2
_GEN_INTERIOR_WIDTHS, _GEN_PANEL_COUNTS = (4, 5), (3, 4)
_GEN_COLORS = tuple(range(10))
_GEN_SPRITE_USES_FRAME_PROB = 0.3
_GEN_SPRITE_HEIGHTS, _GEN_SPRITE_WIDTHS = (1, 2), (2, 3)
_GEN_MIN_SPRITE_PIXELS, _GEN_MAX_SPRITE_PIXELS = 2, 5
_GEN_DELTAS = (-3, -2, -1, 1, 2, 3)
_GEN_MAX_ATTEMPTS = 300
_GEN_MASK_CACHE = {}

def _gen_round_half_up(value):
    return _gen_math.floor(value + _gen_Fraction(1, 2))

def _gen_contiguous_masks(sprite_h, sprite_w):
    key = (sprite_h, sprite_w)
    if key in _GEN_MASK_CACHE:
        return _GEN_MASK_CACHE[key]
    cells = [(r, c) for r in range(sprite_h) for c in range(sprite_w)]
    masks_by_count = {}
    max_pixels = min(_GEN_MAX_SPRITE_PIXELS, sprite_h * sprite_w)
    for pixel_count in range(_GEN_MIN_SPRITE_PIXELS, max_pixels + 1):
        masks = []
        for combo in _gen_itertools.combinations(cells, pixel_count):
            rows = {r for r, _ in combo}
            cols = {c for _, c in combo}
            if len(rows) != sprite_h or len(cols) != sprite_w:
                continue
            mask = np.zeros((sprite_h, sprite_w), dtype=bool)
            for r, c in combo:
                mask[r, c] = True
            if is_contiguous(mask.astype(int), background=0, connectivity=4):
                masks.append(mask)
        if masks:
            masks_by_count[pixel_count] = masks
    _GEN_MASK_CACHE[key] = masks_by_count
    return masks_by_count

def _gen_sprite(fill, sprite_color):
    sprite_h = _gen_random.choice(_GEN_SPRITE_HEIGHTS)
    sprite_w = _gen_random.choice(_GEN_SPRITE_WIDTHS)
    masks_by_count = _gen_contiguous_masks(sprite_h, sprite_w)

    pixel_count = _gen_random.choice(sorted(masks_by_count))
    mask = _gen_random.choice(masks_by_count[pixel_count])
    sprite = np.full((sprite_h, sprite_w), fill, dtype=int)
    sprite[mask] = sprite_color
    return sprite, mask

def _gen_make_panel(height, panel_w, frame, fill, sprite, top_row, local_left):
    panel = np.full((height, panel_w), fill, dtype=int)
    panel[0, :] = frame
    panel[-1, :] = frame
    panel[:, 0] = frame
    panel[:, -1] = frame
    paste(panel, sprite, top_row, local_left, transparent=fill)
    return panel

def _gen_choose_palette():
    frame, fill = _gen_random.sample(_GEN_COLORS, 2)
    if _gen_random.random() < _GEN_SPRITE_USES_FRAME_PROB:
        sprite_color = frame
    else:
        sprite_color = _gen_random.choice(
            [color for color in _GEN_COLORS if color not in (frame, fill)]
        )
    return frame, fill, sprite_color

def _gen_motion_choices(sprite_h, panel_w, num_panels):
    max_top = _GEN_HEIGHT - 1 - sprite_h
    stride = panel_w - 1
    choices = []
    for delta in _GEN_DELTAS:
        for start in range(1, max_top + 1):
            top_rows = [start + i * delta for i in range(num_panels)]
            if not all(1 <= top <= max_top for top in top_rows):
                continue
            next_top = _gen_round_half_up(
                _gen_Fraction(top_rows[-1], 1)
                + _gen_Fraction(delta * panel_w, stride)
            )
            if 1 <= next_top <= max_top:
                choices.append((delta, start, next_top))
    return choices

def _gen_make_input(num_panels, stride, frame, fill, sprite, start_top, delta):
    width = num_panels * stride + 1
    inp = np.full((_GEN_HEIGHT, width), fill, dtype=int)
    inp[0, :] = frame
    inp[-1, :] = frame
    for panel_i in range(num_panels + 1):
        inp[:, panel_i * stride] = frame
    for panel_i in range(num_panels):
        top = start_top + panel_i * delta
        paste(inp, sprite, top, panel_i * stride + _GEN_LOCAL_LEFT, transparent=fill)
    return inp

def _gen_attempt():
    interior_w = _gen_random.choice(_GEN_INTERIOR_WIDTHS)
    num_panels = _gen_random.choice(_GEN_PANEL_COUNTS)
    panel_w = interior_w + 2
    stride = panel_w - 1
    frame, fill, sprite_color = _gen_choose_palette()

    sprite, _ = _gen_sprite(fill, sprite_color)
    sprite_h, sprite_w = sprite.shape
    if _GEN_LOCAL_LEFT + sprite_w > panel_w - 1:
        return None

    choices = _gen_motion_choices(sprite_h, panel_w, num_panels)
    if not choices:
        return None
    delta, start_top, next_top = _gen_random.choice(choices)

    expected = _gen_make_panel(
        _GEN_HEIGHT, panel_w, frame, fill, sprite, next_top, _GEN_LOCAL_LEFT
    )
    inp = _gen_make_input(num_panels, stride, frame, fill, sprite, start_top, delta)
    if np.array_equal(main(inp), expected):
        return inp
    return None

def generate_input():
    for _ in range(_GEN_MAX_ATTEMPTS):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate valid ba1aa698 input")
