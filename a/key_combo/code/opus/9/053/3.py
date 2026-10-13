from __future__ import annotations

import hashlib
import random

import numpy as np


BACKGROUND = 0
MARKER_COLOR = 5
SPRITE_SIZE = 3
INPUT_SHAPE_RANGE = (14, 21)
GENERATION_RETRIES = 400
NONBLACK_PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
# The solver accepts any positive k parsed from the gray square.
# The generator only samples branch combinations that keep the derived
# output size 6 * k * C within the ARC 30x30 cap.
_SEEN_OUTPUT_SIGNATURES: set[str] = set()


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _extract_marker_square(grid: np.ndarray) -> tuple[int, tuple[int, int, int, int]]:
    marker_box = _bbox_from_mask(grid == MARKER_COLOR)
    assert marker_box is not None, "expected one gray marker square"
    r0, c0, r1, c1 = marker_box
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    assert height == width, "gray marker must be square"
    assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == MARKER_COLOR), "gray marker must be solid"
    assert int(np.count_nonzero(grid == MARKER_COLOR)) == height * width, "gray marker must be unique"
    return height, marker_box


def _extract_sprite(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    sprite_box = _bbox_from_mask((grid != BACKGROUND) & (grid != MARKER_COLOR))
    assert sprite_box is not None, "expected one colored sprite"
    r0, c0, r1, c1 = sprite_box
    assert (r1 - r0 + 1, c1 - c0 + 1) == (SPRITE_SIZE, SPRITE_SIZE), "sprite bbox must be 3x3"
    sprite = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
    return sprite, sprite_box


def _sprite_color_count(sprite: np.ndarray) -> int:
    colors = set(map(int, np.unique(sprite)))
    if BACKGROUND not in colors:
        return len(colors)
    return len(colors - {MARKER_COLOR})


def _scaled_pinwheel(sprite: np.ndarray, scale: int) -> np.ndarray:
    scaled = np.kron(sprite, np.ones((scale, scale), dtype=int))
    top = np.hstack((scaled, np.rot90(scaled, 1)))
    bottom = np.hstack((np.rot90(scaled, 2), np.rot90(scaled, -1)))
    return np.vstack((top, bottom))


def _build_output(sprite: np.ndarray, scale: int, repeat_count: int) -> np.ndarray:
    pinwheel = _scaled_pinwheel(sprite, scale)
    return np.tile(pinwheel, (repeat_count, repeat_count))


def _rotations_all_distinct(sprite: np.ndarray) -> bool:
    rots = [np.rot90(sprite, k) for k in range(4)]
    return all(not np.array_equal(rots[i], rots[j]) for i in range(4) for j in range(i + 1, 4))


def _boxes_touch_or_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + 1 < br0 or br1 + 1 < ar0 or ac1 + 1 < bc0 or bc1 + 1 < ac0)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Scale the 3x3 sprite by the gray square's side length, arrange the scaled copies into a rotation pinwheel, then tile that pinwheel by the sprite's color count.

    Input structure:
    - A black background contains one 3x3 colored sprite and one detached solid gray square marker.
    - The sprite may use black internally, plus two or three non-gray foreground colors, and its four quarter-turn rotations are all distinct.
    - The gray marker is a single solid square whose side length is the scale factor.

    Rule:
    - Read the gray square side length k.
    - Scale every sprite cell into a k x k block to obtain a larger sprite S.
    - Form a 2x2 pinwheel from S: upper-left original, upper-right 90-degree counterclockwise, lower-left 180-degree, lower-right 90-degree clockwise.
    - Count the distinct colors inside the original 3x3 sprite, including black only when black appears in the sprite, and tile the whole pinwheel that many times in both directions.

    Key insight:
    - Two independent visible controls drive the output size: the gray square gives the pixel scale, while the original sprite palette size gives the tiling count.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    scale, _marker_box = _extract_marker_square(g)
    sprite, _sprite_box = _extract_sprite(g)
    repeat_count = _sprite_color_count(sprite)
    return _build_output(sprite, scale, repeat_count)


def _choose_combo() -> tuple[int, int]:
    weighted = [(1, 2), (1, 3), (1, 3), (1, 4), (1, 4), (2, 2), (2, 2), (2, 2)]
    return random.choice(weighted)


def _sprite_palette_spec(color_count: int) -> tuple[list[int], bool]:
    if color_count == 2:
        n_fg, include_black = 2, False
    elif color_count == 3:
        n_fg, include_black = random.choice(((2, True), (3, False)))
    elif color_count == 4:
        n_fg, include_black = 3, True
    else:
        raise AssertionError(f"unsupported sprite color count: {color_count}")
    fg_colors = random.sample(NONBLACK_PALETTE, n_fg)
    return fg_colors, include_black


def _random_sprite(color_count: int) -> np.ndarray:
    fg_colors, include_black = _sprite_palette_spec(color_count)
    palette = list(fg_colors) + ([BACKGROUND] if include_black else [])
    min_nonblack = 5 if include_black else 6
    max_nonblack = 8 if include_black else 9

    for _ in range(500):
        cells = np.array([random.choice(palette) for _ in range(SPRITE_SIZE * SPRITE_SIZE)], dtype=int)
        sprite = cells.reshape(SPRITE_SIZE, SPRITE_SIZE)
        if any(color not in sprite for color in fg_colors):
            continue
        if include_black and BACKGROUND not in sprite:
            continue
        if not include_black and BACKGROUND in sprite:
            continue
        nonblack = int(np.count_nonzero(sprite != BACKGROUND))
        if not (min_nonblack <= nonblack <= max_nonblack):
            continue
        if any(np.all(sprite[r] == BACKGROUND) for r in range(SPRITE_SIZE)):
            continue
        if any(np.all(sprite[:, c] == BACKGROUND) for c in range(SPRITE_SIZE)):
            continue
        if len({int(x) for x in np.unique(sprite) if x != BACKGROUND}) not in (2, 3):
            continue
        if _sprite_color_count(sprite) != color_count:
            continue
        if not _rotations_all_distinct(sprite):
            continue
        return sprite
    raise RuntimeError("failed to generate a valid sprite")


def _place_objects(height: int, width: int, scale: int) -> tuple[tuple[int, int], tuple[int, int]]:
    for _ in range(200):
        sprite_r = random.randint(0, min(4, height - SPRITE_SIZE - scale - 4))
        sprite_c = random.randint(0, min(4, width - SPRITE_SIZE - scale - 4))
        marker_r = random.randint(max(sprite_r + SPRITE_SIZE + 3, height - scale - 5), height - scale)
        marker_c = random.randint(max(sprite_c + SPRITE_SIZE + 3, width - scale - 5), width - scale)
        sprite_box = (sprite_r, sprite_c, sprite_r + SPRITE_SIZE - 1, sprite_c + SPRITE_SIZE - 1)
        marker_box = (marker_r, marker_c, marker_r + scale - 1, marker_c + scale - 1)
        if not _boxes_touch_or_overlap(sprite_box, marker_box):
            return (sprite_r, sprite_c), (marker_r, marker_c)
    raise RuntimeError("failed to place sprite and marker")


def _semantic_checks(grid: np.ndarray, scale: int, color_count: int) -> bool:
    sprite, sprite_box = _extract_sprite(grid)
    parsed_scale, marker_box = _extract_marker_square(grid)
    if parsed_scale != scale or _sprite_color_count(sprite) != color_count:
        return False
    if _boxes_touch_or_overlap(sprite_box, marker_box):
        return False
    if not _rotations_all_distinct(sprite):
        return False
    out = main(grid)
    if max(out.shape) > 30:
        return False
    if out.shape != (6 * scale * color_count, 6 * scale * color_count):
        return False

    if scale != 1:
        alt_scale_out = _build_output(sprite, 1, color_count)
        if alt_scale_out.shape == out.shape and np.array_equal(alt_scale_out, out):
            return False
        unscaled_pinwheel = _scaled_pinwheel(sprite, 1)
        maybe_shortcut = np.tile(unscaled_pinwheel, (color_count, color_count))
        if maybe_shortcut.shape == out.shape and np.array_equal(maybe_shortcut, out):
            return False

    for alt_count in (2, 3, 4):
        if alt_count == color_count:
            continue
        alt = _build_output(sprite, scale, alt_count)
        if alt.shape == out.shape and np.array_equal(alt, out):
            return False

    return True


def _output_signature(grid: np.ndarray) -> str:
    out = main(grid)
    return hashlib.md5(out.tobytes() + str(out.shape).encode()).hexdigest()


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*INPUT_SHAPE_RANGE)
    width = random.randint(*INPUT_SHAPE_RANGE)
    scale, color_count = _choose_combo()
    sprite = _random_sprite(color_count)

    if 6 * scale * color_count > 30:
        return None

    grid = np.zeros((height, width), dtype=int)
    try:
        (sprite_r, sprite_c), (marker_r, marker_c) = _place_objects(height, width, scale)
    except RuntimeError:
        return None

    grid[sprite_r : sprite_r + SPRITE_SIZE, sprite_c : sprite_c + SPRITE_SIZE] = sprite
    grid[marker_r : marker_r + scale, marker_c : marker_c + scale] = MARKER_COLOR

    if not _semantic_checks(grid, scale, color_count):
        return None

    signature = _output_signature(grid)
    if signature in _SEEN_OUTPUT_SIGNATURES:
        return None
    _SEEN_OUTPUT_SIGNATURES.add(signature)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
