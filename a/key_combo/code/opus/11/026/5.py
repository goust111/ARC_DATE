from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BACKGROUND = 0
RED = 2
GREEN = 3
MARKER = 7

GEN_CANVAS_SIZE = 13
GEN_SPRITE_SIZE = 3
GEN_MARKER_DISTANCE = 3
GEN_ATTEMPTS = 400
GEN_SIDE_COUNT_WEIGHTS = (1, 3, 3, 2)

SIDES = ("up", "right", "down", "left")
ROTATION_BY_SIDE = {
    "up": 0,
    "right": -1,
    "down": 2,
    "left": 1,
}


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _rotation_signatures(sprite: np.ndarray) -> set[bytes]:
    return {np.rot90(sprite, k).tobytes() for k in range(4)}


def _sprite_is_valid(sprite: np.ndarray) -> bool:
    if sprite.shape != (GEN_SPRITE_SIZE, GEN_SPRITE_SIZE):
        return False
    if not set(map(int, np.unique(sprite))) <= {RED, GREEN}:
        return False
    red_count = int(np.count_nonzero(sprite == RED))
    green_count = int(np.count_nonzero(sprite == GREEN))
    if red_count == 0 or green_count == 0:
        return False
    if red_count in (1, 8):
        return False
    return len(_rotation_signatures(sprite)) == 4


def _extract_sprite_and_sides(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int], set[str]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    sprite_mask = (g != BACKGROUND) & (g != MARKER)
    box = _bbox(sprite_mask)
    assert box is not None, "expected one non-marker sprite"
    sprite = _crop_bbox(g, box)
    assert _sprite_is_valid(sprite), "expected a 3x3 red/green sprite with four distinct rotations"

    r0, c0, r1, c1 = box
    center_r = (r0 + r1) // 2
    center_c = (c0 + c1) // 2

    markers = np.argwhere(g == MARKER)
    assert 1 <= len(markers) <= 4, "expected 1..4 orange markers"

    sides: set[str] = set()
    for r, c in markers:
        r = int(r)
        c = int(c)
        if c == center_c and r < r0:
            assert r0 - r >= 2
            side = "up"
        elif r == center_r and c > c1:
            assert c - c1 >= 2
            side = "right"
        elif c == center_c and r > r1:
            assert r - r1 >= 2
            side = "down"
        elif r == center_r and c < c0:
            assert c0 - c >= 2
            side = "left"
        else:
            raise AssertionError("marker must lie on one cardinal extension line of the sprite bbox")
        assert side not in sides, "at most one marker per side"
        sides.add(side)

    return sprite, box, sides


def _render_output(sprite: np.ndarray, sides: Iterable[str]) -> np.ndarray:
    side_set = set(sides)
    h, w = sprite.shape
    top_extra = h if "up" in side_set else 0
    bottom_extra = h if "down" in side_set else 0
    left_extra = w if "left" in side_set else 0
    right_extra = w if "right" in side_set else 0

    out = np.zeros((top_extra + h + bottom_extra, left_extra + w + right_extra), dtype=int)
    center_r0 = top_extra
    center_c0 = left_extra
    out[center_r0 : center_r0 + h, center_c0 : center_c0 + w] = sprite

    for side in SIDES:
        if side not in side_set:
            continue
        arm = np.rot90(sprite, ROTATION_BY_SIDE[side])
        if side == "up":
            out[center_r0 - h : center_r0, center_c0 : center_c0 + w] = arm
        elif side == "right":
            out[center_r0 : center_r0 + h, center_c0 + w : center_c0 + 2 * w] = arm
        elif side == "down":
            out[center_r0 + h : center_r0 + 2 * h, center_c0 : center_c0 + w] = arm
        else:
            out[center_r0 : center_r0 + h, center_c0 - w : center_c0] = arm
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate the central sprite according to each scattered orange pixel and concatenate the rotated copies around it.

    Input structure:
    - A black background contains one central 3x3 sprite made only of red and green cells.
    - One to four orange single pixels lie outside the sprite on the cardinal extension lines through the sprite-bbox center.
    - Each side of the sprite bbox contributes at most one orange pixel, and every orange pixel is at least two cells away.

    Rule:
    - Find the minimal bbox containing all non-black, non-orange cells and treat that cropped region as the sprite.
    - Decode every orange pixel by its side relative to the sprite bbox center: up -> 0 degrees, right -> 90 degrees clockwise, down -> 180 degrees, left -> 270 degrees clockwise.
    - Rotate a copy of the sprite by that angle, place it flush against the corresponding side of the original sprite, then crop to the minimal rectangle containing the original sprite and all placed copies.
    - Remove the orange pixels; only the original sprite and the rotated side copies remain in the output.

    Key insight:
    - The marker position carries two coupled meanings at once: it chooses both the rotation angle and the side where that rotated copy must be concatenated.
    """
    sprite, _box, sides = _extract_sprite_and_sides(grid)
    return _render_output(sprite, sides)


def _random_sprite() -> np.ndarray:
    for _ in range(200):
        sprite = np.random.choice([RED, GREEN], size=(GEN_SPRITE_SIZE, GEN_SPRITE_SIZE)).astype(int)
        if _sprite_is_valid(sprite):
            return sprite
    raise RuntimeError("failed to sample a non-symmetric sprite")


def _sample_sides() -> set[str]:
    count = random.choices([1, 2, 3, 4], weights=GEN_SIDE_COUNT_WEIGHTS, k=1)[0]
    return set(random.sample(SIDES, count))


def _marker_positions() -> dict[str, tuple[int, int]]:
    center = GEN_CANVAS_SIZE // 2
    half = GEN_SPRITE_SIZE // 2
    top = center - half
    bottom = center + half
    left = center - half
    right = center + half
    return {
        "up": (top - GEN_MARKER_DISTANCE, center),
        "right": (center, right + GEN_MARKER_DISTANCE),
        "down": (bottom + GEN_MARKER_DISTANCE, center),
        "left": (center, left - GEN_MARKER_DISTANCE),
    }


def _build_input(sprite: np.ndarray, sides: set[str]) -> np.ndarray:
    grid = np.zeros((GEN_CANVAS_SIZE, GEN_CANVAS_SIZE), dtype=int)
    center = GEN_CANVAS_SIZE // 2
    half = GEN_SPRITE_SIZE // 2
    r0 = center - half
    c0 = center - half
    grid[r0 : r0 + GEN_SPRITE_SIZE, c0 : c0 + GEN_SPRITE_SIZE] = sprite
    for side, (r, c) in _marker_positions().items():
        if side in sides:
            grid[r, c] = MARKER
    return grid


def _render_with_constant_rotation(sprite: np.ndarray, sides: Iterable[str], rotation_k: int) -> np.ndarray:
    side_set = set(sides)
    h, w = sprite.shape
    top_extra = h if "up" in side_set else 0
    bottom_extra = h if "down" in side_set else 0
    left_extra = w if "left" in side_set else 0
    right_extra = w if "right" in side_set else 0

    out = np.zeros((top_extra + h + bottom_extra, left_extra + w + right_extra), dtype=int)
    center_r0 = top_extra
    center_c0 = left_extra
    out[center_r0 : center_r0 + h, center_c0 : center_c0 + w] = sprite

    arm = np.rot90(sprite, rotation_k)
    if "up" in side_set:
        out[center_r0 - h : center_r0, center_c0 : center_c0 + w] = arm
    if "right" in side_set:
        out[center_r0 : center_r0 + h, center_c0 + w : center_c0 + 2 * w] = arm
    if "down" in side_set:
        out[center_r0 + h : center_r0 + 2 * h, center_c0 : center_c0 + w] = arm
    if "left" in side_set:
        out[center_r0 : center_r0 + h, center_c0 - w : center_c0] = arm
    return out


def _has_side_dependent_rotation(sprite: np.ndarray, sides: set[str], out: np.ndarray) -> bool:
    if len(sides) < 2:
        return True
    for k in (0, -1, 2, 1):
        constant = _render_with_constant_rotation(sprite, sides, k)
        if np.array_equal(constant, out):
            return False
    return True


def _validate_generated(grid: np.ndarray) -> bool:
    try:
        sprite, _box, sides = _extract_sprite_and_sides(grid)
        out = main(grid)
    except Exception:
        return False

    expected = _render_output(sprite, sides)
    if not np.array_equal(out, expected):
        return False
    if max(grid.shape) > 30 or max(out.shape) > 30:
        return False
    if not _has_side_dependent_rotation(sprite, sides, out):
        return False

    for side in list(sides):
        reduced = grid.copy()
        rr, cc = _marker_positions()[side]
        reduced[rr, cc] = BACKGROUND
        if len(sides) == 1:
            reduced_out = _render_output(sprite, set())
        else:
            try:
                reduced_out = main(reduced)
            except Exception:
                return False
        if np.array_equal(reduced_out, out):
            return False

    if len(sides) > 1:
        for side in sides:
            singleton = _render_output(sprite, {side})
            if np.array_equal(singleton, out):
                return False

    return True


def _gen_attempt() -> np.ndarray | None:
    sprite = _random_sprite()
    sides = _sample_sides()
    grid = _build_input(sprite, sides)
    if not _validate_generated(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid scattered-pixel rotation input")
