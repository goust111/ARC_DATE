from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, crop_bbox, find_components, get_bg, paste


BG = int(Color.BLACK)
RED = int(Color.RED)
SOURCE_COLORS = [1, 3, 4, 5, 6, 7, 8, 9]
STEP_DIRS = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]
MAX_GEN_TRIES = 400


def _sprite_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.full(mask.shape, BG, dtype=int)
    patch[mask] = int(color)
    return patch


def _red_patch(sprite: np.ndarray) -> np.ndarray:
    return np.where(sprite != BG, RED, BG)


def _fits(shape: tuple[int, int], pos: tuple[int, int], size: int) -> bool:
    h, w = map(int, shape)
    r, c = map(int, pos)
    return 0 <= r and 0 <= c and r + size <= h and c + size <= w


def _sorted_components(grid: np.ndarray) -> tuple[list[dict], int]:
    bg = int(get_bg(grid))
    colors = sorted(set(map(int, grid.ravel())) - {bg})
    assert len(colors) == 1, "expected exactly one foreground color"
    assert colors[0] != RED, "input foreground must differ from the red output color"

    comps = find_components(grid, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    assert len(comps) == 3, "expected exactly three foreground components"
    comps = sorted(comps, key=lambda comp: tuple(map(int, comp["bbox"][:2])))
    return comps, bg


def _extract_sequence(grid: np.ndarray) -> tuple[list[tuple[int, int]], tuple[int, int], list[np.ndarray], int, int]:
    comps, bg = _sorted_components(grid)
    boxes = [tuple(map(int, comp["bbox"])) for comp in comps]
    sizes = {(r1 - r0 + 1, c1 - c0 + 1) for r0, c0, r1, c1 in boxes}
    assert len(sizes) == 1, "all sprites must share one bounding-box size"
    height, width = next(iter(sizes))
    assert height == width, "each sprite must occupy an n x n window"
    size = int(height)

    points = [(r0, c0) for r0, c0, _r1, _c1 in boxes]
    step1 = (points[1][0] - points[0][0], points[1][1] - points[0][1])
    step2 = (points[2][0] - points[1][0], points[2][1] - points[1][1])
    assert step1 == step2 and step1 != (0, 0), "sprite windows must be equally spaced on a line"

    sprites = [crop_bbox(grid, box) for box in boxes]
    areas = {int(np.count_nonzero(sprite != bg)) for sprite in sprites}
    assert len(areas) == 1, "all sprites must have the same shape"
    return points, step1, sprites, size, bg


def _rotation_delta(sprites: list[np.ndarray]) -> int:
    matches = [
        k
        for k in range(4)
        if np.array_equal(sprites[1], np.rot90(sprites[0], k))
        and np.array_equal(sprites[2], np.rot90(sprites[1], k))
    ]
    assert len(matches) == 1 and matches[0] in (1, 3), "expected a unique 90-degree step"
    return int(matches[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extrapolate a three-sprite pose sequence in both directions and stamp the future poses in red.

    Input structure:
    - The background is black.
    - Exactly three disconnected monochrome copies of one connected sprite appear on the canvas.
    - Each copy fills one n x n bounding box, and the three box top-left corners are equally spaced on one line.
    - The sprite is almost 4-fold rotationally symmetric, except for one offset pixel that reveals its orientation.

    Rule:
    - Sort the three sprite windows by their top-left positions and read the constant step vector between them.
    - Read the constant rotation increment by matching consecutive windows with 90-degree rotations.
    - Extend the sequence from both ends one step at a time while the next n x n window still fits in bounds.
    - Every extrapolated copy keeps the continuing orientation sequence but is recolored red; the original three keep their color.

    Key insight:
    - The copies only pretend to be pure translations: the single asymmetric pixel encodes a pose progression, so correct extrapolation must continue both the spacing and the rotation schedule.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30, "expected an ARC-sized 2D grid"

    points, step, sprites, size, bg = _extract_sequence(g)
    delta = _rotation_delta(sprites)

    out = g.copy()

    pos = points[0]
    sprite = sprites[0]
    while True:
        pos = (pos[0] - step[0], pos[1] - step[1])
        sprite = np.rot90(sprite, -delta)
        if not _fits(out.shape, pos, size):
            break
        patch = _red_patch(sprite)
        target = out[pos[0] : pos[0] + size, pos[1] : pos[1] + size]
        assert np.all(target[patch != BG] == bg), "left extrapolation must land on empty space"
        paste(out, patch, pos[0], pos[1], transparent=bg)

    pos = points[2]
    sprite = sprites[2]
    while True:
        pos = (pos[0] + step[0], pos[1] + step[1])
        sprite = np.rot90(sprite, delta)
        if not _fits(out.shape, pos, size):
            break
        patch = _red_patch(sprite)
        target = out[pos[0] : pos[0] + size, pos[1] : pos[1] + size]
        assert np.all(target[patch != BG] == bg), "right extrapolation must land on empty space"
        paste(out, patch, pos[0], pos[1], transparent=bg)

    return out


def _plus_mask(size: int) -> np.ndarray:
    mid = size // 2
    mask = np.zeros((size, size), dtype=bool)
    mask[mid, :] = True
    mask[:, mid] = True
    return mask


def _diamond_mask(size: int) -> np.ndarray:
    mid = size // 2
    rr, cc = np.indices((size, size))
    return (np.abs(rr - mid) + np.abs(cc - mid)) <= mid


def _frame_mask(size: int) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    mask[[0, -1], :] = True
    mask[:, [0, -1]] = True
    return mask


def _base_mask(size: int, family: str) -> np.ndarray:
    if family == "plus":
        return _plus_mask(size)
    if family == "diamond":
        return _diamond_mask(size)
    if family == "frame":
        return _frame_mask(size)
    raise ValueError(f"unknown sprite family: {family}")


def _oriented_mask(size: int, family: str, orientation: int) -> np.ndarray:
    mask = _base_mask(size, family).copy()
    mid = size // 2
    bump = {
        "plus": (0, mid + 1),
        "diamond": (0, mid + 1),
        "frame": (1, mid + 1),
    }[family]
    mask[bump] = True
    return np.rot90(mask, -int(orientation))


def _rotations_are_distinct(size: int, family: str) -> bool:
    masks = [_oriented_mask(size, family, k) for k in range(4)]
    seen = {mask.tobytes() for mask in masks}
    return len(seen) == 4


def _valid_start_range(limit: int, size: int, delta: int) -> tuple[int, int] | None:
    low = 0
    high = limit - size
    for i in range(3):
        low = max(low, -i * delta)
        high = min(high, limit - size - i * delta)
    if low > high:
        return None
    return int(low), int(high)


def _extension_count(shape: tuple[int, int], start: tuple[int, int], step: tuple[int, int], size: int) -> int:
    count = 0
    pos = (int(start[0]), int(start[1]))
    while True:
        pos = (pos[0] + step[0], pos[1] + step[1])
        if not _fits(shape, pos, size):
            return count
        count += 1


def _spec_output(
    grid: np.ndarray,
    positions: list[tuple[int, int]],
    step: tuple[int, int],
    family: str,
    size: int,
    start_orientation: int,
) -> np.ndarray:
    out = np.array(grid, copy=True)
    pos = positions[0]
    orientation = int(start_orientation)
    while True:
        pos = (pos[0] - step[0], pos[1] - step[1])
        orientation = (orientation - 1) % 4
        if not _fits(out.shape, pos, size):
            break
        patch = _sprite_patch(_oriented_mask(size, family, orientation), RED)
        paste(out, patch, pos[0], pos[1], transparent=BG)

    pos = positions[2]
    orientation = (int(start_orientation) + 2) % 4
    while True:
        pos = (pos[0] + step[0], pos[1] + step[1])
        orientation = (orientation + 1) % 4
        if not _fits(out.shape, pos, size):
            break
        patch = _sprite_patch(_oriented_mask(size, family, orientation), RED)
        paste(out, patch, pos[0], pos[1], transparent=BG)
    return out


def _wrong_translation_output(
    grid: np.ndarray,
    positions: list[tuple[int, int]],
    step: tuple[int, int],
    size: int,
    original_patch: np.ndarray,
) -> np.ndarray:
    out = np.array(grid, copy=True)
    red_patch = _red_patch(original_patch)

    pos = positions[0]
    while True:
        pos = (pos[0] - step[0], pos[1] - step[1])
        if not _fits(out.shape, pos, size):
            break
        paste(out, red_patch, pos[0], pos[1], transparent=BG)

    pos = positions[2]
    while True:
        pos = (pos[0] + step[0], pos[1] + step[1])
        if not _fits(out.shape, pos, size):
            break
        paste(out, red_patch, pos[0], pos[1], transparent=BG)
    return out


def _generate_case() -> dict:
    for _ in range(MAX_GEN_TRIES):
        size = random.choice([5, 7])
        family = random.choice(["plus", "diamond", "frame"])
        if not _rotations_are_distinct(size, family):
            continue

        side_mode = random.choice(["both", "left_only", "right_only"])
        dir_r, dir_c = random.choice(STEP_DIRS)
        step_hi = min(10, 30 - size, size + (3 if side_mode == "both" else 6))
        if step_hi < size + 1:
            continue
        step_scale = random.randint(size + 1, step_hi)
        step = (dir_r * step_scale, dir_c * step_scale)

        min_side = max(14, size + 8)
        if side_mode == "both":
            min_h = max(min_side, size + 4 * abs(step[0]) + 1)
            min_w = max(min_side, size + 4 * abs(step[1]) + 1)
        else:
            min_h = max(min_side, size + 3 * abs(step[0]) + 1)
            min_w = max(min_side, size + 3 * abs(step[1]) + 1)
        if min_h > 30 or min_w > 30:
            continue
        h = random.randint(min_h, 30)
        w = random.randint(min_w, 30)

        r_range = _valid_start_range(h, size, step[0])
        c_range = _valid_start_range(w, size, step[1])
        if r_range is None or c_range is None:
            continue

        candidates: list[tuple[tuple[int, int], list[tuple[int, int]], int, int]] = []
        for start_r in range(r_range[0], r_range[1] + 1):
            for start_c in range(c_range[0], c_range[1] + 1):
                start = (start_r, start_c)
                positions = [
                    start,
                    (start[0] + step[0], start[1] + step[1]),
                    (start[0] + 2 * step[0], start[1] + 2 * step[1]),
                ]
                if not all(_fits((h, w), pos, size) for pos in positions):
                    continue
                left_count = _extension_count((h, w), positions[0], (-step[0], -step[1]), size)
                right_count = _extension_count((h, w), positions[2], step, size)
                if left_count + right_count == 0:
                    continue
                if side_mode == "both" and not (left_count > 0 and right_count > 0):
                    continue
                if side_mode == "left_only" and not (left_count > 0 and right_count == 0):
                    continue
                if side_mode == "right_only" and not (left_count == 0 and right_count > 0):
                    continue
                candidates.append((start, positions, left_count, right_count))
        if not candidates:
            continue

        start, positions, left_count, right_count = random.choice(candidates)

        color = random.choice(SOURCE_COLORS)
        start_orientation = random.randint(0, 3)
        grid = np.full((h, w), BG, dtype=int)
        original_patches: list[np.ndarray] = []
        for idx, pos in enumerate(positions):
            patch = _sprite_patch(
                _oriented_mask(size, family, (start_orientation + idx) % 4),
                color,
            )
            original_patches.append(patch)
            paste(grid, patch, pos[0], pos[1], transparent=BG)

        expected = _spec_output(grid, positions, step, family, size, start_orientation)
        if np.array_equal(expected, grid):
            continue

        if np.array_equal(_wrong_translation_output(grid, positions, step, size, original_patches[0]), expected):
            continue

        solved = main(grid)
        if not np.array_equal(solved, expected):
            continue

        return {
            "input": grid,
            "output": expected,
            "positions": positions,
            "step": step,
            "size": size,
            "family": family,
            "start_orientation": start_orientation,
            "left_count": left_count,
            "right_count": right_count,
        }

    raise RuntimeError("failed to generate a valid input for puzzle 2")


def generate_input() -> np.ndarray:
    return _generate_case()["input"]
