from __future__ import annotations

import random
from typing import Iterable

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]

BG = 0
SPRITE_SIZE = 3
BODY_COLOR_POOL = (1, 3, 4, 5, 9)
ALL_NON_BG = tuple(range(1, 10))
GRID_MIN = 20
GRID_MAX = 26
RECT_SIDE_MIN = 2
RECT_SIDE_MAX = 5
COPY_COUNT_WEIGHTS = ((1, 3), (2, 4), (3, 3), (4, 2))
DIR_TO_COLOR = {
    (-1, 0): 2,  # up -> red
    (0, 1): 7,   # right -> orange
    (1, 0): 8,   # down -> teal
    (0, -1): 6,  # left -> pink
}

_DIRECTION_BAG: list[tuple[int, int]] = []


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox_from_coords(coords: list[Coord]) -> BBox:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _bbox_h(bbox: BBox) -> int:
    return int(bbox[2] - bbox[0] + 1)


def _bbox_w(bbox: BBox) -> int:
    return int(bbox[3] - bbox[1] + 1)


def _crop_bbox(grid: np.ndarray, bbox: BBox) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    dst[r0 : r0 + h, c0 : c0 + w] = patch


def _neighbors4(r: int, c: int) -> Iterable[Coord]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _find_components(
    grid: np.ndarray,
    *,
    bg: int,
    monochromatic: bool,
    colors: set[int] | None = None,
) -> list[dict]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            if visited[r, c]:
                continue
            color = int(grid[r, c])
            if color == bg:
                continue
            if colors is not None and color not in colors:
                continue

            coords: list[Coord] = []
            stack = [(r, c)]
            visited[r, c] = True
            seed = color

            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if visited[nr, nc]:
                        continue
                    value = int(grid[nr, nc])
                    if value == bg:
                        continue
                    if colors is not None and value not in colors:
                        continue
                    if monochromatic and value != seed:
                        continue
                    visited[nr, nc] = True
                    stack.append((nr, nc))

            components.append(
                {
                    "color": seed if monochromatic else None,
                    "coords": coords,
                    "bbox": _bbox_from_coords(coords),
                    "area": len(coords),
                }
            )

    return components


def _intersects(a: BBox, b: BBox) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _bbox_center(bbox: BBox) -> tuple[float, float]:
    r0, c0, r1, c1 = bbox
    return (r0 + r1) / 2.0, (c0 + c1) / 2.0


def _direction_from_bboxes(sprite_bbox: BBox, rect_bbox: BBox) -> tuple[int, int]:
    sr, sc = _bbox_center(sprite_bbox)
    rr, rc = _bbox_center(rect_bbox)
    dr = rr - sr
    dc = rc - sc
    if abs(dc) > abs(dr):
        return (0, 1) if dc > 0 else (0, -1)
    if abs(dr) > abs(dc):
        return (1, 0) if dr > 0 else (-1, 0)
    raise AssertionError("target direction is ambiguous")


def _perpendicular_offset(sprite_bbox: BBox, rect_bbox: BBox, direction: tuple[int, int]) -> float:
    sr, sc = _bbox_center(sprite_bbox)
    rr, rc = _bbox_center(rect_bbox)
    if direction[0] == 0:
        return rr - sr
    return rc - sc


def _find_sprites(grid: np.ndarray, bg: int) -> list[dict]:
    sprites: list[dict] = []
    for comp in _find_components(grid, bg=bg, monochromatic=False):
        bbox = comp["bbox"]
        if _bbox_h(bbox) != 3 or _bbox_w(bbox) != 3 or comp["area"] != 9:
            continue
        patch = _crop_bbox(grid, bbox)
        border_vals = [
            int(patch[r, c])
            for r in range(3)
            for c in range(3)
            if (r, c) != (1, 1)
        ]
        border_colors = set(border_vals)
        if len(border_colors) != 1:
            continue
        body_color = next(iter(border_colors))
        center_color = int(patch[1, 1])
        if body_color == bg or center_color == bg or body_color == center_color:
            continue
        sprites.append(
            {
                "bbox": bbox,
                "patch": patch,
                "body_color": body_color,
                "key_color": center_color,
            }
        )
    sprites.sort(key=lambda item: item["bbox"])
    return sprites


def _find_rectangles(grid: np.ndarray, bg: int) -> dict[int, dict]:
    rectangles: dict[int, dict] = {}
    for comp in _find_components(grid, bg=bg, monochromatic=True):
        bbox = comp["bbox"]
        h = _bbox_h(bbox)
        w = _bbox_w(bbox)
        if h < 2 or w < 2 or comp["area"] != h * w:
            continue
        color = int(comp["color"])
        if color in rectangles:
            raise AssertionError("rectangle colors must be unique")
        rectangles[color] = comp
    return rectangles


def _copy_positions(sprite_bbox: BBox, rect_bbox: BBox, direction: tuple[int, int], shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    dr, dc = direction
    r = sprite_bbox[0] + dr * SPRITE_SIZE
    c = sprite_bbox[1] + dc * SPRITE_SIZE
    positions: list[tuple[int, int]] = []
    limit = max(h, w) + 2

    for _ in range(limit):
        current_bbox = (r, c, r + 2, c + 2)
        if not (0 <= r and r + 3 <= h and 0 <= c and c + 3 <= w):
            raise AssertionError("stamp path left the grid before hitting the rectangle")
        positions.append((r, c))
        if _intersects(current_bbox, rect_bbox):
            return positions
        r += dr * SPRITE_SIZE
        c += dc * SPRITE_SIZE

    raise AssertionError("stamp path never overlaps the target rectangle")


def _analyze_scene(grid: np.ndarray) -> list[dict]:
    bg = _get_bg(grid)
    sprites = _find_sprites(grid, bg)
    rectangles = _find_rectangles(grid, bg)
    if not sprites:
        raise AssertionError("expected at least one sprite")

    scene: list[dict] = []
    key_colors = [int(sprite["key_color"]) for sprite in sprites]
    if len(set(key_colors)) != len(key_colors):
        raise AssertionError("sprite key colors must be unique")

    for sprite in sprites:
        key = int(sprite["key_color"])
        if key not in rectangles:
            raise AssertionError("missing matching rectangle")
        rect = rectangles[key]
        direction = _direction_from_bboxes(sprite["bbox"], rect["bbox"])
        copies = _copy_positions(sprite["bbox"], rect["bbox"], direction, grid.shape)
        scene.append(
            {
                **sprite,
                "target_bbox": rect["bbox"],
                "direction": direction,
                "copies": copies,
            }
        )

    body_colors = [int(item["body_color"]) for item in scene]
    if len(set(body_colors)) != len(body_colors):
        raise AssertionError("sprite body colors must be unique")

    return scene


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repeat each keyed 3x3 sprite toward its color-matched rectangle, recoloring every copied center by travel direction.

    Input structure:
    - The grid has a black background plus 2-4 full 3x3 sprites and 2-4 solid rectangles.
    - Each sprite has one uniform border color and a different center key color; each key color appears on exactly one separate solid rectangle.
    - Every matched rectangle lies in one clear basic direction from its sprite: up, right, down, or left, possibly with a small perpendicular offset.

    Rule:
    - For each sprite, find the unique solid rectangle whose color equals the sprite center.
    - Infer the sprite-to-rectangle basic direction from their relative placement, then copy the whole 3x3 sprite every 3 cells along that direction.
    - Stop after placing the first copy whose 3x3 bounding box overlaps the matched rectangle; copied pixels overwrite the rectangle where they land.
    - Keep the original sprite center unchanged, but recolor the center of every copy to the direction code color: up->2, right->7, down->8, left->6.

    Key insight:
    - The center pixel changes meaning across the process: in the original sprite it is a destination key, while in the repeated copies it becomes a direction marker.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise AssertionError("grid must be 2D")

    scene = _analyze_scene(g)
    out = g.copy()
    for item in scene:
        patch = np.array(item["patch"], copy=True)
        patch[1, 1] = DIR_TO_COLOR[item["direction"]]
        for r, c in item["copies"]:
            _paste(out, patch, r, c)
    return out


def _mark_bbox(mask: np.ndarray, bbox: BBox) -> None:
    r0, c0, r1, c1 = bbox
    mask[r0 : r1 + 1, c0 : c1 + 1] = True


def _expand_mask(mask: np.ndarray, pad: int = 1) -> np.ndarray:
    if pad <= 0:
        return mask.copy()
    h, w = mask.shape
    out = mask.copy()
    for r, c in np.argwhere(mask):
        r0 = max(0, int(r) - pad)
        r1 = min(h, int(r) + pad + 1)
        c0 = max(0, int(c) - pad)
        c1 = min(w, int(c) + pad + 1)
        out[r0:r1, c0:c1] = True
    return out


def _copy_count() -> int:
    counts, weights = zip(*COPY_COUNT_WEIGHTS)
    return int(random.choices(counts, weights=weights, k=1)[0])


def _take_directions(count: int) -> list[tuple[int, int]]:
    global _DIRECTION_BAG
    chosen: list[tuple[int, int]] = []
    while len(chosen) < count:
        if not _DIRECTION_BAG:
            _DIRECTION_BAG = list(DIR_TO_COLOR)
            random.shuffle(_DIRECTION_BAG)
        direction = _DIRECTION_BAG.pop()
        if direction in chosen:
            continue
        chosen.append(direction)
    random.shuffle(chosen)
    return chosen


def _choose_pair_count() -> int:
    return int(random.choices([2, 3, 4], weights=[2, 4, 3], k=1)[0])


def _overlap_choice(direction: tuple[int, int], copies: int) -> int:
    if direction in ((0, 1), (1, 0)):
        options = [0, 1, 2] if copies > 1 else [1, 2]
    else:
        options = [0, 1, 2] if copies > 1 else [0, 1]
    return int(random.choice(options))


def _pick_offset(options: list[int]) -> int:
    nonzero = [value for value in options if value != 0]
    if nonzero and random.random() < 0.7:
        return int(random.choice(nonzero))
    return int(random.choice(options))


def _build_pair_mask(shape: tuple[int, int], sprite_bbox: BBox, rect_bbox: BBox, copies: list[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    _mark_bbox(mask, sprite_bbox)
    _mark_bbox(mask, rect_bbox)
    for r, c in copies:
        _mark_bbox(mask, (r, c, r + 2, c + 2))
    return mask


def _candidate_pair(
    shape: tuple[int, int],
    direction: tuple[int, int],
    reserved: np.ndarray,
) -> dict | None:
    h, w = shape
    guarded = _expand_mask(reserved, pad=1)

    for _ in range(400):
        copies = _copy_count()
        rect_h = random.randint(RECT_SIDE_MIN, RECT_SIDE_MAX)
        rect_w = random.randint(RECT_SIDE_MIN, RECT_SIDE_MAX)
        overlap = _overlap_choice(direction, copies)

        if direction == (0, 1):
            sr = random.randint(0, h - SPRITE_SIZE)
            row_offsets = [d for d in range(-rect_h + 1, 3) if 0 <= sr + d <= h - rect_h]
            if not row_offsets:
                continue
            sc_max = w - rect_w - SPRITE_SIZE * copies - overlap
            if sc_max < 0:
                continue
            sc = random.randint(0, sc_max)
            rr0 = sr + _pick_offset(row_offsets)
            rc0 = sc + SPRITE_SIZE * copies + overlap
        elif direction == (0, -1):
            sr = random.randint(0, h - SPRITE_SIZE)
            row_offsets = [d for d in range(-rect_h + 1, 3) if 0 <= sr + d <= h - rect_h]
            if not row_offsets:
                continue
            sc_min = SPRITE_SIZE * copies - overlap + rect_w - 1
            if sc_min > w - SPRITE_SIZE:
                continue
            sc = random.randint(sc_min, w - SPRITE_SIZE)
            rr0 = sr + _pick_offset(row_offsets)
            rc0 = sc - SPRITE_SIZE * copies + overlap - rect_w + 1
        elif direction == (1, 0):
            sc = random.randint(0, w - SPRITE_SIZE)
            col_offsets = [d for d in range(-rect_w + 1, 3) if 0 <= sc + d <= w - rect_w]
            if not col_offsets:
                continue
            sr_max = h - rect_h - SPRITE_SIZE * copies - overlap
            if sr_max < 0:
                continue
            sr = random.randint(0, sr_max)
            rr0 = sr + SPRITE_SIZE * copies + overlap
            rc0 = sc + _pick_offset(col_offsets)
        else:
            sc = random.randint(0, w - SPRITE_SIZE)
            col_offsets = [d for d in range(-rect_w + 1, 3) if 0 <= sc + d <= w - rect_w]
            if not col_offsets:
                continue
            sr_min = SPRITE_SIZE * copies - overlap + rect_h - 1
            if sr_min > h - SPRITE_SIZE:
                continue
            sr = random.randint(sr_min, h - SPRITE_SIZE)
            rr0 = sr - SPRITE_SIZE * copies + overlap - rect_h + 1
            rc0 = sc + _pick_offset(col_offsets)

        sprite_bbox = (sr, sc, sr + 2, sc + 2)
        rect_bbox = (rr0, rc0, rr0 + rect_h - 1, rc0 + rect_w - 1)
        try:
            inferred = _direction_from_bboxes(sprite_bbox, rect_bbox)
        except AssertionError:
            continue
        if inferred != direction:
            continue

        try:
            copy_positions = _copy_positions(sprite_bbox, rect_bbox, direction, shape)
        except AssertionError:
            continue
        if len(copy_positions) != copies:
            continue

        actual = _build_pair_mask(shape, sprite_bbox, rect_bbox, copy_positions)
        if np.any(actual & guarded):
            continue

        return {
            "sprite_bbox": sprite_bbox,
            "rect_bbox": rect_bbox,
            "copies": copy_positions,
            "direction": direction,
        }

    return None


def _render_pair(grid: np.ndarray, sprite_bbox: BBox, rect_bbox: BBox, body_color: int, key_color: int) -> None:
    sr, sc, _, _ = sprite_bbox
    rr0, rc0, rr1, rc1 = rect_bbox
    patch = np.full((3, 3), body_color, dtype=int)
    patch[1, 1] = key_color
    _paste(grid, patch, sr, sc)
    grid[rr0 : rr1 + 1, rc0 : rc1 + 1] = key_color


def _generate_candidate() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    pair_count = _choose_pair_count()
    directions = _take_directions(pair_count)
    body_colors = random.sample(list(BODY_COLOR_POOL), pair_count)
    key_pool = [color for color in ALL_NON_BG if color not in body_colors]
    key_colors = random.sample(key_pool, pair_count)

    grid = np.full((h, w), BG, dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    for direction, body_color, key_color in zip(directions, body_colors, key_colors):
        placement = _candidate_pair((h, w), direction, reserved)
        if placement is None:
            return None
        _render_pair(grid, placement["sprite_bbox"], placement["rect_bbox"], body_color, key_color)
        reserved |= _build_pair_mask((h, w), placement["sprite_bbox"], placement["rect_bbox"], placement["copies"])

    return grid


def _validate_generated(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        return False
    if max(grid.shape) > 30:
        return False

    try:
        scene = _analyze_scene(grid)
        out = main(grid)
    except AssertionError:
        return False

    if not (2 <= len(scene) <= 4):
        return False
    if np.array_equal(out, grid):
        return False

    directions = {item["direction"] for item in scene}
    if len(directions) < 2:
        return False
    if not any(len(item["copies"]) > 1 for item in scene):
        return False
    if not any(abs(_perpendicular_offset(item["bbox"], item["target_bbox"], item["direction"])) > 0.0 for item in scene):
        return False

    pair_masks = []
    for item in scene:
        mask = _build_pair_mask(grid.shape, item["bbox"], item["target_bbox"], item["copies"])
        pair_masks.append(mask)

        sr, sc, _, _ = item["bbox"]
        if int(out[sr + 1, sc + 1]) != int(item["key_color"]):
            return False

        for idx, (r, c) in enumerate(item["copies"]):
            center = int(out[r + 1, c + 1])
            if center != DIR_TO_COLOR[item["direction"]]:
                return False
            patch = out[r : r + 3, c : c + 3]
            border = [
                int(patch[rr, cc])
                for rr in range(3)
                for cc in range(3)
                if (rr, cc) != (1, 1)
            ]
            if set(border) != {int(item["body_color"])}:
                return False

            input_patch = grid[r : r + 3, c : c + 3]
            allowed = {BG, int(item["key_color"])}
            if idx < len(item["copies"]) - 1:
                if set(map(int, input_patch.flatten())) != {BG}:
                    return False
            else:
                if not set(map(int, input_patch.flatten())).issubset(allowed):
                    return False

    for i in range(len(pair_masks)):
        for j in range(i + 1, len(pair_masks)):
            if np.any(pair_masks[i] & pair_masks[j]):
                return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(600):
        grid = _generate_candidate()
        if grid is not None and _validate_generated(grid):
            return grid
    raise RuntimeError("failed to generate a valid input grid")
