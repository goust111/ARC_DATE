import random
from typing import Iterable

import numpy as np


BG = 0
GUIDE_COLOR = 6
SPRITE_COLORS = [1, 2, 3, 4, 5, 7, 8, 9]

GRID_H_RANGE = (15, 20)
GRID_W_RANGE = (15, 20)
SPRITE_COUNT_RANGE = (4, 6)
GUIDE_H_RANGE = (5, 9)
GUIDE_W_RANGE = (5, 9)
GUIDE_POINT_COUNT_RANGE = (3, 5)
SPRITE_GAP = 2
MAX_GENERATION_ATTEMPTS = 500


def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _expand_box(
    box: tuple[int, int, int, int], pad: int, shape: tuple[int, int] | None = None
) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = box
    r0 -= pad
    c0 -= pad
    r1 += pad
    c1 += pad
    if shape is not None:
        h, w = shape
        r0 = max(0, r0)
        c0 = max(0, c0)
        r1 = min(h - 1, r1)
        c1 = min(w - 1, c1)
    return int(r0), int(c0), int(r1), int(c1)


def _boxes_intersect(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _box_contains(
    outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]
) -> bool:
    return outer[0] <= inner[0] and outer[1] <= inner[1] and inner[2] <= outer[2] and inner[3] <= outer[3]


def _sprite_patch(border_color: int, center_color: int = BG) -> np.ndarray:
    patch = np.full((3, 3), BG, dtype=int)
    patch[0, :] = border_color
    patch[2, :] = border_color
    patch[:, 0] = border_color
    patch[:, 2] = border_color
    patch[1, 1] = center_color
    return patch


def _detect_sprite_window(grid: np.ndarray, r0: int, c0: int) -> dict | None:
    block = grid[r0 : r0 + 3, c0 : c0 + 3]
    if block.shape != (3, 3) or int(block[1, 1]) != BG:
        return None
    border = [
        int(block[0, 0]),
        int(block[0, 1]),
        int(block[0, 2]),
        int(block[1, 0]),
        int(block[1, 2]),
        int(block[2, 0]),
        int(block[2, 1]),
        int(block[2, 2]),
    ]
    color = border[0]
    if color in (BG, GUIDE_COLOR) or any(v != color for v in border):
        return None
    return {
        "bbox": (r0, c0, r0 + 2, c0 + 2),
        "color": color,
        "center": (r0 + 1, c0 + 1),
    }


def _find_sprites(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    sprites = []
    seen = set()
    for r0 in range(h - 2):
        for c0 in range(w - 2):
            sprite = _detect_sprite_window(grid, r0, c0)
            if sprite is None:
                continue
            if sprite["bbox"] in seen:
                continue
            seen.add(sprite["bbox"])
            sprites.append(sprite)
    return sprites


def _guide_bbox_points(
    box: tuple[int, int, int, int], point_count: int, rng: random.Random
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = box
    corners = [(r0, c0), (r0, c1), (r1, c0), (r1, c1)]
    templates = [
        [corners[0], corners[1], corners[2]],
        [corners[0], corners[1], corners[3]],
        [corners[0], corners[2], corners[3]],
        [corners[1], corners[2], corners[3]],
        [(r0, rng.randint(c0, c1)), (r1, rng.randint(c0, c1)), (rng.randint(r0, r1), c0), (rng.randint(r0, r1), c1)],
    ]
    points = set(rng.choice(templates))
    perimeter = []
    for c in range(c0, c1 + 1):
        perimeter.append((r0, c))
        perimeter.append((r1, c))
    for r in range(r0 + 1, r1):
        perimeter.append((r, c0))
        perimeter.append((r, c1))
    rng.shuffle(perimeter)
    for pt in perimeter:
        if len(points) >= point_count:
            break
        points.add(pt)
    return list(points)


def _filled_grid_and_selected(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    sprites = _find_sprites(grid)
    assert sprites, "expected at least one hollow sprite"

    ordered = sorted(sprites, key=lambda item: item["center"])
    cycle = [sprite["color"] for sprite in ordered]
    out = np.array(grid, dtype=int, copy=True)
    for sprite in sprites:
        r, c = sprite["center"]
        out[r, c] = cycle[(r + c) % len(cycle)]

    guide_box = bbox(grid == GUIDE_COLOR)
    assert guide_box is not None, "expected guide pixels"
    selected = [sprite for sprite in sprites if _box_contains(guide_box, sprite["bbox"])]
    assert len(selected) == 1, "expected exactly one sprite selected by guide bbox"
    return out, selected[0]


def main(grid: np.ndarray) -> np.ndarray:
    """Crop the guide-selected hollow sprite after filling every sprite center by a global diagonal cycle.

    Input structure:
    - The background is black.
    - The grid contains several non-overlapping 3x3 hollow square sprites with distinct border colors.
    - A sparse magenta guide mask does not overlap any sprite, and its bbox encloses exactly one sprite.

    Rule:
    - Detect every 3x3 hollow square sprite and sort them by center position in row-major order.
    - Build a color cycle from those sorted border colors, then fill each sprite center at (r, c) with the cycle color indexed by (r + c) mod len(cycle).
    - Compute the guide mask bbox and find the unique sprite whose bbox is fully contained inside it.
    - Return the 3x3 crop of that selected sprite after the center fill.

    Key insight:
    - The guide mask only chooses which sprite to output, while the center color of every sprite depends on the global arrangement of all sprite centers through a diagonal-residue cycle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    filled, selected = _filled_grid_and_selected(g)
    return crop_bbox(filled, selected["bbox"])


def _place_patch(grid: np.ndarray, patch: np.ndarray, top_left: tuple[int, int]) -> None:
    r0, c0 = top_left
    h, w = patch.shape
    grid[r0 : r0 + h, c0 : c0 + w] = patch


def _candidate_sprite_boxes(
    shape: tuple[int, int],
    existing_boxes: Iterable[tuple[int, int, int, int]],
    forbidden_boxes: Iterable[tuple[int, int, int, int]],
    rng: random.Random,
) -> list[tuple[int, int, int, int]]:
    h, w = shape
    candidates = [(r, c, r + 2, c + 2) for r in range(h - 2) for c in range(w - 2)]
    rng.shuffle(candidates)
    valid = []
    for box in candidates:
        if any(_boxes_intersect(box, _expand_box(other, SPRITE_GAP, shape)) for other in existing_boxes):
            continue
        if any(_boxes_intersect(box, forbidden) for forbidden in forbidden_boxes):
            continue
        valid.append(box)
    return valid


def _expected_selected_crop(
    sprite_infos: list[dict], selected_box: tuple[int, int, int, int]
) -> np.ndarray:
    ordered = sorted(sprite_infos, key=lambda item: item["center"])
    cycle = [sprite["color"] for sprite in ordered]
    selected = next(sprite for sprite in sprite_infos if sprite["bbox"] == selected_box)
    fill_color = cycle[(selected["center"][0] + selected["center"][1]) % len(cycle)]
    return _sprite_patch(selected["color"], fill_color)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    sprite_count = rng.randint(*SPRITE_COUNT_RANGE)
    if sprite_count > len(SPRITE_COLORS):
        return None

    guide_h = rng.randint(GUIDE_H_RANGE[0], min(GUIDE_H_RANGE[1], h))
    guide_w = rng.randint(GUIDE_W_RANGE[0], min(GUIDE_W_RANGE[1], w))
    if guide_h < 5 or guide_w < 5:
        return None

    guide_r0 = rng.randint(0, h - guide_h)
    guide_c0 = rng.randint(0, w - guide_w)
    guide_box = (guide_r0, guide_c0, guide_r0 + guide_h - 1, guide_c0 + guide_w - 1)

    sprite_r0 = rng.randint(guide_box[0] + 1, guide_box[2] - 3)
    sprite_c0 = rng.randint(guide_box[1] + 1, guide_box[3] - 3)
    selected_box = (sprite_r0, sprite_c0, sprite_r0 + 2, sprite_c0 + 2)

    colors = rng.sample(SPRITE_COLORS, sprite_count)
    selected_color = colors[0]
    grid = np.zeros((h, w), dtype=int)
    sprite_infos = [
        {"bbox": selected_box, "center": (sprite_r0 + 1, sprite_c0 + 1), "color": selected_color}
    ]
    _place_patch(grid, _sprite_patch(selected_color), (sprite_r0, sprite_c0))

    forbidden_for_others = [_expand_box(guide_box, 1, (h, w))]
    for color in colors[1:]:
        candidates = _candidate_sprite_boxes(
            (h, w),
            [info["bbox"] for info in sprite_infos],
            forbidden_for_others,
            rng,
        )
        if not candidates:
            return None
        box = candidates[0]
        r0, c0, _, _ = box
        _place_patch(grid, _sprite_patch(color), (r0, c0))
        sprite_infos.append({"bbox": box, "center": (r0 + 1, c0 + 1), "color": color})

    guide_points = _guide_bbox_points(guide_box, rng.randint(*GUIDE_POINT_COUNT_RANGE), rng)
    for r, c in guide_points:
        if grid[r, c] != BG:
            return None
        grid[r, c] = GUIDE_COLOR

    detected = _find_sprites(grid)
    if len(detected) != sprite_count:
        return None
    if bbox(grid == GUIDE_COLOR) != guide_box:
        return None

    contained = [sprite for sprite in detected if _box_contains(guide_box, sprite["bbox"])]
    if len(contained) != 1 or contained[0]["bbox"] != selected_box:
        return None

    expected = _expected_selected_crop(sprite_infos, selected_box)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
