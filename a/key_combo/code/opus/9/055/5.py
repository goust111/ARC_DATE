from __future__ import annotations

import itertools

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
ORANGE = 7

TRACK_STEP = 3
MAX_GENERATION_ATTEMPTS = 400
_RNG = np.random.default_rng()


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    stack = [start]
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _extract_single_red_sprite(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    red = grid == RED
    bbox = _bbox(red)
    assert bbox is not None, "expected one red sprite"
    assert _is_connected(red), "red sprite must be 4-connected"
    r0, c0, r1, c1 = bbox
    sprite = red[r0 : r1 + 1, c0 : c1 + 1].copy()
    return sprite, bbox


def _paste_mask(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    h, w = mask.shape
    grid[r0 : r0 + h, c0 : c0 + w][mask] = color


def _infer_track(grid: np.ndarray, sprite_bbox: tuple[int, int, int, int]) -> tuple[int, int, list[int]]:
    green_coords = np.argwhere(grid == GREEN)
    assert green_coords.size > 0, "expected green track dots"
    rows = np.unique(green_coords[:, 0])
    assert len(rows) == 1, "track must be horizontal"
    track_row = int(rows[0])

    green_cols = sorted(int(c) for c in green_coords[:, 1])
    assert len(green_cols) >= 2, "need multiple visible track dots"
    diffs = np.diff(green_cols)
    step = int(diffs.min())
    assert step > 0 and np.all(diffs == step), "track dots must be evenly spaced"

    start_col = int(sprite_bbox[1])
    assert start_col <= green_cols[0], "sprite must start at or before first visible dot"
    assert int(sprite_bbox[0] + (sprite_bbox[2] - sprite_bbox[0] + 1) // 2) == track_row

    full_track_cols = list(range(start_col, green_cols[-1] + 1, step))
    assert full_track_cols, "track cannot be empty"
    assert set(green_cols).issubset(full_track_cols), "visible track dots must lie on slots"
    return track_row, step, full_track_cols


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide the red sprite rightward by its own pixel count and stamp its bbox complement on passed slots.

    Input structure:
    - A black background contains one 4-connected red sprite and a horizontal row of evenly spaced green single-pixel track dots.
    - The red sprite sits on the leftmost track slot, with its tight bounding box reused as the slot footprint during motion.
    - The green dots define the slot spacing, while the sprite's tight box defines the local frame for every stamp.

    Rule:
    - Count the red sprite pixels; call this count P.
    - Reconstruct the whole track, including the hidden leftmost dot underneath the starting sprite.
    - Move the sprite right by P track slots.
    - On each intermediate slot 1..P-1, place an orange stamp equal to the complement of the sprite inside its tight bounding box.
    - On slot P, place the original red sprite; track dots remain unless covered by a stamp or the final sprite.

    Key insight:
    - The sprite's area is not just a shape property: it encodes the travel distance, while the sprite's own bbox complement encodes the trail left behind.
    """
    g = np.asarray(grid, dtype=int)
    sprite_mask, sprite_bbox = _extract_single_red_sprite(g)
    track_row, step, track_cols = _infer_track(g, sprite_bbox)

    p = int(sprite_mask.sum())
    r0, c0, _r1, _c1 = sprite_bbox
    complement_mask = ~sprite_mask
    assert np.any(complement_mask), "sprite bbox must have a non-empty complement"
    assert p < len(track_cols), "sprite must have enough slots to move"

    out = np.zeros_like(g)
    for col in track_cols:
        out[track_row, col] = GREEN

    for slot_idx in range(1, p):
        _paste_mask(out, complement_mask, r0, c0 + slot_idx * step, ORANGE)
    _paste_mask(out, sprite_mask, r0, c0 + p * step, RED)
    return out


def _enumerate_sprite_masks() -> dict[int, list[np.ndarray]]:
    by_area: dict[int, list[np.ndarray]] = {}
    for h, w in ((2, 2), (2, 3), (3, 2), (3, 3)):
        cells = [(r, c) for r in range(h) for c in range(w)]
        for area in range(3, min(6, h * w - 1) + 1):
            for chosen in itertools.combinations(cells, area):
                mask = np.zeros((h, w), dtype=bool)
                for r, c in chosen:
                    mask[r, c] = True
                if not _is_connected(mask):
                    continue
                if _bbox(mask) != (0, 0, h - 1, w - 1):
                    continue
                anchor_r = h // 2
                if not mask[anchor_r, 0]:
                    continue
                by_area.setdefault(area, []).append(mask)
    return by_area


_SPRITES_BY_AREA = _enumerate_sprite_masks()
assert _SPRITES_BY_AREA, "failed to build sprite catalog"


def _render_scene(
    shape: tuple[int, int],
    sprite_mask: np.ndarray,
    slot_count: int,
    top: int,
    left: int,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = shape
    p = int(sprite_mask.sum())
    track_row = top + sprite_mask.shape[0] // 2
    input_grid = np.zeros((h, w), dtype=int)
    output_grid = np.zeros((h, w), dtype=int)

    track_cols = [left + i * TRACK_STEP for i in range(slot_count)]
    for col in track_cols:
        input_grid[track_row, col] = GREEN
        output_grid[track_row, col] = GREEN

    _paste_mask(input_grid, sprite_mask, top, left, RED)

    complement_mask = ~sprite_mask
    for slot_idx in range(1, p):
        _paste_mask(output_grid, complement_mask, top, left + slot_idx * TRACK_STEP, ORANGE)
    _paste_mask(output_grid, sprite_mask, top, left + p * TRACK_STEP, RED)
    return input_grid, output_grid


def _semantic_checks(
    input_grid: np.ndarray,
    expected_output: np.ndarray,
    sprite_mask: np.ndarray,
    slot_count: int,
    top: int,
    left: int,
) -> bool:
    p = int(sprite_mask.sum())
    complement_size = int((~sprite_mask).sum())
    track_row = top + sprite_mask.shape[0] // 2
    visible_cols = [left + i * TRACK_STEP for i in range(1, slot_count)]

    if not np.array_equal(np.unique(input_grid), np.array([0, 2, 3])):
        return False
    if int((input_grid == RED).sum()) != p:
        return False
    if sorted(int(c) for c in np.argwhere(input_grid == GREEN)[:, 1]) != visible_cols:
        return False
    if len(np.unique(np.argwhere(input_grid == GREEN)[:, 0])) != 1:
        return False
    if input_grid[track_row, left] != RED:
        return False
    if int((expected_output == ORANGE).sum()) != (p - 1) * complement_size:
        return False
    if int((expected_output == RED).sum()) != p:
        return False
    if expected_output[track_row, left] != GREEN:
        return False
    if expected_output[track_row, left + p * TRACK_STEP] != RED:
        return False
    if np.array_equal(input_grid, expected_output):
        return False
    return True


def _attempt_generate() -> np.ndarray | None:
    areas = sorted(_SPRITES_BY_AREA)
    area = int(_RNG.choice(np.array(areas, dtype=int)))
    sprite_mask = _SPRITES_BY_AREA[area][int(_RNG.integers(len(_SPRITES_BY_AREA[area])))]
    p = int(sprite_mask.sum())

    min_slots = max(6, p + 1)
    slot_count = int(_RNG.integers(min_slots, 9))

    sprite_h, sprite_w = sprite_mask.shape
    top = int(_RNG.integers(1, 8))
    bottom = int(_RNG.integers(1, 9))
    left = int(_RNG.integers(1, 5))
    right = int(_RNG.integers(1, 5))

    height = top + sprite_h + bottom
    last_track_col = left + (slot_count - 1) * TRACK_STEP
    final_right = left + p * TRACK_STEP + sprite_w - 1
    width = max(last_track_col, final_right) + 1 + right
    if height > 30 or width > 30:
        return None

    input_grid, expected_output = _render_scene((height, width), sprite_mask, slot_count, top, left)
    if not _semantic_checks(input_grid, expected_output, sprite_mask, slot_count, top, left):
        return None

    try:
        actual_output = main(input_grid)
    except Exception:
        return None
    if not np.array_equal(actual_output, expected_output):
        return None
    return input_grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 055/5")
