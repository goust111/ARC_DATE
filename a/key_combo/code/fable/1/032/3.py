from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import bbox, crop_bbox, find_components, get_bg, paste


MAX_ATTEMPTS = 400
DEPTH_CHOICES = (3, 3, 4, 4)


def _erode_mask(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask.astype(bool), 1, constant_values=False)
    return (
        padded[1:-1, 1:-1]
        & padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )


def _layer_depths(mask: np.ndarray) -> np.ndarray:
    depths = np.zeros(mask.shape, dtype=int)
    current = mask.astype(bool).copy()
    depth = 1
    while current.any():
        nxt = _erode_mask(current)
        depths[current & ~nxt] = depth
        current = nxt
        depth += 1
    return depths


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp the unique sprite repeatedly toward the side with more open space, eroding one layer per copy.

    Input structure:
    - The background is the dominant color.
    - Exactly one 8-connected multicolor sprite is present, and its colors form visible nested layers.
    - The sprite has much more empty space on either its left or right side, enough for at least two slots.

    Rule:
    - Keep the original sprite unchanged.
    - Compare the empty columns to the left and right of the sprite bbox and choose the larger side.
    - Place copy slots every sprite_width + 1 columns along that side; slot k contains the previous mask after removing all pixels that are 4-neighbor adjacent to background, with surviving pixels keeping their original colors.
    - Stop when the eroded mask becomes empty, clipping any out-of-bounds part of a slot.

    Key insight:
    - The repeated stamps are not identical tiles: each successive copy is one more 4-neighbor erosion of the original mask.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = int(get_bg(g))
    components = find_components(g, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    assert len(components) == 1, "expected exactly one multicolor sprite"

    comp = components[0]
    r0, c0, r1, c1 = map(int, comp["bbox"])
    sprite = crop_bbox(g, comp["bbox"])
    mask = sprite != bg
    assert mask.any(), "sprite must be nonempty"

    left_space = c0
    right_space = g.shape[1] - 1 - c1
    direction = 1 if right_space > left_space else -1
    step = sprite.shape[1] + 1

    out = g.copy()
    current = mask.copy()
    slot = 1
    while True:
        current = _erode_mask(current)
        if not current.any():
            break
        patch = np.full(sprite.shape, bg, dtype=int)
        patch[current] = sprite[current]
        paste(out, patch, r0, c0 + direction * slot * step, transparent=bg, clip=True)
        slot += 1

    return out


def _row_widths(height: int, width: int, style: str) -> list[int]:
    if style == "rect":
        return [width] * height

    widths = [width] * height
    mid_lo = (height - 1) // 2
    mid_hi = height // 2
    shrink_choices = [0, 1, 1, 2] if style == "rounded" else [0, 1, 2]

    for r in range(mid_lo - 1, -1, -1):
        widths[r] = max(3, widths[r + 1] - random.choice(shrink_choices))
    for r in range(mid_hi + 1, height):
        widths[r] = max(3, widths[r - 1] - random.choice(shrink_choices))

    if style != "rect" and width > 4 and all(w == width for w in widths):
        edge_r = random.choice([0, height - 1])
        widths[edge_r] = width - 1
    return widths


def _touching_left(width: int, ref_left: int, ref_width: int, total_width: int) -> int:
    lo = max(0, ref_left - width)
    hi = min(total_width - width, ref_left + ref_width)
    target = max(lo, min(hi, ref_left + random.choice([-1, 0, 1])))
    nearby = [left for left in range(lo, hi + 1) if abs(left - target) <= 1]
    return random.choice(nearby or list(range(lo, hi + 1)))


def _row_lefts(widths: list[int], total_width: int, drift: bool) -> list[int]:
    if not drift:
        return [(total_width - width) // 2 for width in widths]

    lefts = [0] * len(widths)
    mid_lo = (len(widths) - 1) // 2
    mid_hi = len(widths) // 2
    center_left = (total_width - widths[mid_lo]) // 2
    lefts[mid_lo] = center_left
    if mid_hi != mid_lo:
        lefts[mid_hi] = _touching_left(widths[mid_hi], lefts[mid_lo], widths[mid_lo], total_width)

    for r in range(mid_lo - 1, -1, -1):
        lefts[r] = _touching_left(widths[r], lefts[r + 1], widths[r + 1], total_width)
    for r in range(mid_hi + 1, len(widths)):
        lefts[r] = _touching_left(widths[r], lefts[r - 1], widths[r - 1], total_width)
    return lefts


def _candidate_mask(target_depth: int) -> tuple[np.ndarray, np.ndarray] | None:
    if target_depth == 3:
        height = random.randint(5, 8)
        width = random.randint(5, 7)
        styles = ["rect", "rounded", "drift"]
    else:
        height = random.randint(7, 9)
        width = random.randint(7, 8)
        styles = ["rect", "rect", "drift", "rounded"]

    style = random.choice(styles)
    widths = _row_widths(height, width, "rounded" if style == "rounded" else "drift" if style == "drift" else "rect")
    lefts = _row_lefts(widths, width, drift=(style == "drift"))

    mask = np.zeros((height, width), dtype=bool)
    for r, (row_width, left) in enumerate(zip(widths, lefts)):
        mask[r, left : left + row_width] = True

    box = bbox(mask)
    if box is None:
        return None
    mask = crop_bbox(mask, box).astype(bool)
    depths = _layer_depths(mask)
    if int(depths.max()) != target_depth:
        return None
    if np.count_nonzero(depths == target_depth) == 0:
        return None
    return mask, depths


def _layered_sprite(target_depth: int, bg: int) -> tuple[np.ndarray, np.ndarray, int]:
    for _ in range(MAX_ATTEMPTS):
        candidate = _candidate_mask(target_depth)
        if candidate is None:
            continue
        mask, depths = candidate
        max_depth = int(depths.max())
        colors = random.sample([color for color in range(10) if color != bg], k=max_depth)
        sprite = np.full(mask.shape, bg, dtype=int)
        for depth in range(1, max_depth + 1):
            sprite[depths == depth] = int(colors[depth - 1])
        return sprite, mask, max_depth
    raise RuntimeError("failed to generate a layered sprite")


def _place_sprite(sprite: np.ndarray, bg: int, open_left: bool) -> np.ndarray | None:
    height, width = map(int, sprite.shape)
    depth = int(_layer_depths(sprite != bg).max())
    closed_space = random.randint(0, 2)

    if depth == 3:
        min_open = 2 * (width + 1)
        max_open = min(30 - width - closed_space, min_open + 6)
    else:
        min_open = max(2 * (width + 1), 2 * width + 4)
        max_open = 30 - width - closed_space

    if max_open < min_open:
        return None

    open_space = random.randint(min_open, max_open)
    left_space = open_space if open_left else closed_space
    right_space = closed_space if open_left else open_space
    grid_h = height + random.randint(1, 5) + random.randint(1, 5)
    grid_w = left_space + width + right_space
    if grid_h > 30 or grid_w > 30:
        return None

    top_margin = random.randint(0, grid_h - height)
    grid = np.full((grid_h, grid_w), bg, dtype=int)
    paste(grid, sprite, top_margin, left_space, transparent=bg)
    return grid


def _generate_case() -> np.ndarray | None:
    bg = random.randrange(10)
    target_depth = random.choice(DEPTH_CHOICES)
    sprite, mask, depth = _layered_sprite(target_depth, bg)
    del mask

    open_left = bool(random.getrandbits(1))
    grid = _place_sprite(sprite, bg, open_left=open_left)
    if grid is None:
        return None

    bg = int(get_bg(grid))
    components = find_components(grid, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    if len(components) != 1:
        return None

    r0, c0, _r1, c1 = map(int, components[0]["bbox"])
    sprite_h, sprite_w = map(int, sprite.shape)
    if sprite_h != _r1 - r0 + 1 or sprite_w != c1 - c0 + 1:
        return None

    left_space = c0
    right_space = grid.shape[1] - 1 - c1
    if open_left:
        if left_space <= right_space:
            return None
    else:
        if right_space <= left_space:
            return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None

    step = sprite_w + 1
    direction = -1 if open_left else 1
    for k in (1, 2):
        slot_c = c0 + direction * k * step
        if slot_c < 0 or slot_c + sprite_w > grid.shape[1]:
            return None
        if not np.any(out[r0 : r0 + sprite_h, slot_c : slot_c + sprite_w] != bg):
            return None

    if depth == 4:
        slot_c = c0 + direction * 3 * step
        view_c0 = max(0, slot_c)
        view_c1 = min(grid.shape[1], slot_c + sprite_w)
        if view_c0 >= view_c1:
            return None
        if not np.any(out[r0 : r0 + sprite_h, view_c0:view_c1] != bg):
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _generate_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid erosion-stamping input")
