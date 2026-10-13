from __future__ import annotations

import random

import numpy as np


BG = 0
RAY_COLOR = 2
SPRITE_COLORS = (1, 3, 4, 7, 8, 9)
GRID_SIDES = (11, 13)
QUADRANT_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
MAX_GENERATION_ATTEMPTS = 600

_CASE_BAG: list[tuple[int, tuple[int, int]]] = []


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _find_sprites(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color in (BG, RAY_COLOR):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            arr = np.asarray(coords, dtype=int)
            r0 = int(arr[:, 0].min())
            c0 = int(arr[:, 1].min())
            r1 = int(arr[:, 0].max())
            c1 = int(arr[:, 1].max())
            components.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (r0, c0, r1, c1),
                    "area": int(arr.shape[0]),
                }
            )
    return components


def _quadrant_from_bbox(bbox: tuple[int, int, int, int], center: int) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    mid_r = r0 + r1
    mid_c = c0 + c1
    pivot = 2 * center
    assert mid_r != pivot and mid_c != pivot, "sprite bbox center must lie in one diagonal quadrant"
    dr = -1 if mid_r < pivot else 1
    dc = -1 if mid_c < pivot else 1
    return dr, dc


def _ray_output(side: int, direction: tuple[int, int]) -> np.ndarray:
    out = np.zeros((side, side), dtype=int)
    center = side // 2
    r = center
    c = center
    dr, dc = direction
    while 0 <= r < side and 0 <= c < side:
        out[r, c] = RAY_COLOR
        r += dr
        c += dc
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase every sprite and redraw a red diagonal ray whose direction is chosen by the unique largest sprite.

    Input structure:
    - The grid is an odd square on a black background with a single red center pixel.
    - Several separated monochrome 4-connected sprites lie strictly inside the four diagonal quadrants.
    - Every sprite uses a non-black non-red color, no sprite touches another sprite, and exactly one sprite has the largest area.

    Rule:
    - Find all 4-connected monochrome components excluding black background and the red center marker.
    - Select the unique largest component by pixel area.
    - Use the largest component's bounding-box center relative to the grid center to choose one diagonal quadrant.
    - Output a black grid containing only a red diagonal ray from the center to the corresponding edge.

    Key insight:
    - The visible colored sprites are all deleted; they matter only through area comparison, because the largest sprite's location indirectly determines the ray direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] == g.shape[1] and g.shape[0] % 2 == 1, "expected an odd square grid"

    side = int(g.shape[0])
    center = side // 2
    assert int(g[center, center]) == RAY_COLOR, "expected a red center pixel"

    sprites = _find_sprites(g)
    assert sprites, "expected at least one sprite"
    max_area = max(sprite["area"] for sprite in sprites)
    largest = [sprite for sprite in sprites if sprite["area"] == max_area]
    assert len(largest) == 1, "expected a unique largest sprite"

    direction = _quadrant_from_bbox(largest[0]["bbox"], center)
    return _ray_output(side, direction)


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    return out


def _dim_choices(area: int) -> list[tuple[int, int]]:
    dims: list[tuple[int, int]] = []
    for height in (2, 3):
        for width in (2, 3):
            if area > height * width:
                continue
            if area < height + width - 1:
                continue
            dims.append((height, width))
    return dims


def _connected_mask(height: int, width: int, area: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    if random.random() < 0.5:
        r, c = 0, random.choice([0, width - 1])
        end_r, end_c = height - 1, width - 1 - c
    else:
        r, c = random.choice([0, height - 1]), 0
        end_r, end_c = height - 1 - r, width - 1

    mask[r, c] = True
    while (r, c) != (end_r, end_c):
        moves = []
        if r != end_r:
            moves.append((r + (1 if end_r > r else -1), c))
        if c != end_c:
            moves.append((r, c + (1 if end_c > c else -1)))
        r, c = random.choice(moves)
        mask[r, c] = True

    while int(mask.sum()) < area:
        frontier: set[tuple[int, int]] = set()
        for rr, cc in np.argwhere(mask):
            for nr, nc in _neighbors4(int(rr), int(cc)):
                if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                    frontier.add((nr, nc))
        if not frontier:
            break
        rr, cc = random.choice(tuple(frontier))
        mask[rr, cc] = True

    return mask


def _make_patch(color: int, area: int, preferred_dims: tuple[int, int] | None = None, compact: bool = False) -> np.ndarray:
    dims = _dim_choices(area)
    if preferred_dims is not None and preferred_dims in dims:
        height, width = preferred_dims
    else:
        if compact:
            bbox_area = min(h * w for h, w in dims)
            dims = [dim for dim in dims if dim[0] * dim[1] == bbox_area]
        height, width = random.choice(dims)
    mask = _connected_mask(height, width, area)
    patch = np.zeros((height, width), dtype=int)
    patch[mask] = color
    return patch


def _quadrant_bounds(side: int, direction: tuple[int, int]) -> tuple[int, int, int, int]:
    center = side // 2
    dr, dc = direction
    if dr < 0:
        r0, r1 = 0, center - 1
    else:
        r0, r1 = center + 1, side - 1
    if dc < 0:
        c0, c1 = 0, center - 1
    else:
        c0, c1 = center + 1, side - 1
    return r0, r1, c0, c1


def _candidate_positions(
    forbidden: np.ndarray,
    patch: np.ndarray,
    direction: tuple[int, int],
) -> list[tuple[int, int, int]]:
    side = forbidden.shape[0]
    height, width = patch.shape
    patch_mask = patch != BG
    q_r0, q_r1, q_c0, q_c1 = _quadrant_bounds(side, direction)
    if height > q_r1 - q_r0 + 1 or width > q_c1 - q_c0 + 1:
        return []

    center = side // 2
    candidates: list[tuple[int, int, int]] = []
    for r0 in range(q_r0, q_r1 - height + 2):
        for c0 in range(q_c0, q_c1 - width + 2):
            if np.any(forbidden[r0 : r0 + height, c0 : c0 + width] & patch_mask):
                continue
            dist_key = abs(2 * center - (2 * r0 + height - 1)) + abs(2 * center - (2 * c0 + width - 1))
            candidates.append((r0, c0, dist_key))
    return candidates


def _pick_candidate(candidates: list[tuple[int, int, int]], mode: str) -> tuple[int, int] | None:
    if not candidates:
        return None
    ranked = sorted(candidates, key=lambda item: item[2])
    bucket = max(1, len(ranked) // 3)
    if mode == "near":
        pool = ranked[:bucket]
    elif mode == "far":
        pool = ranked[-bucket:]
    else:
        pool = ranked
    r0, c0, _ = random.choice(pool)
    return r0, c0


def _place_patch(grid: np.ndarray, forbidden: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    height, width = patch.shape
    mask = patch != BG
    grid[r0 : r0 + height, c0 : c0 + width][mask] = patch[mask]
    placed = np.zeros_like(forbidden)
    placed[r0 : r0 + height, c0 : c0 + width] = mask
    forbidden |= _dilate8(placed)


def _all_count_patterns(total: int, answer_idx: int) -> list[tuple[int, int, int, int]]:
    patterns: list[tuple[int, int, int, int]] = []
    for a in range(3):
        for b in range(3):
            for c in range(3):
                for d in range(3):
                    counts = (a, b, c, d)
                    if sum(counts) != total or counts[answer_idx] == 0:
                        continue
                    if sum(v > 0 for v in counts) < 3:
                        continue
                    patterns.append(counts)
    return patterns


def _next_case() -> tuple[int, tuple[int, int]]:
    global _CASE_BAG
    if not _CASE_BAG:
        _CASE_BAG = [(side, direction) for side in GRID_SIDES for direction in QUADRANT_DIRS]
        random.shuffle(_CASE_BAG)
    return _CASE_BAG.pop()


def _bbox_dims(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    return r1 - r0 + 1, c1 - c0 + 1


def _bbox_center_distance(bbox: tuple[int, int, int, int], center: int) -> int:
    r0, c0, r1, c1 = bbox
    return abs(2 * center - (r0 + r1)) + abs(2 * center - (c0 + c1))


def _valid_scene(
    grid: np.ndarray,
    expected_direction: tuple[int, int],
    require_near_contrast: bool,
    require_bbox_contrast: bool,
) -> bool:
    side = int(grid.shape[0])
    center = side // 2
    if grid.shape != (side, side) or side not in GRID_SIDES or int(grid[center, center]) != RAY_COLOR:
        return False
    if int(np.count_nonzero(grid == RAY_COLOR)) != 1:
        return False

    sprites = _find_sprites(grid)
    if not (4 <= len(sprites) <= 6):
        return False

    areas = [sprite["area"] for sprite in sprites]
    if len(set(areas)) != len(areas) or any(area < 3 or area > 9 for area in areas):
        return False

    colors = [sprite["color"] for sprite in sprites]
    if len(set(colors)) != len(colors):
        return False

    quadrant_counts = {direction: 0 for direction in QUADRANT_DIRS}
    occupied = np.zeros_like(grid, dtype=bool)
    moat = np.zeros_like(grid, dtype=bool)
    moat[center - 1 : center + 2, center - 1 : center + 2] = True
    occupied |= moat

    for sprite in sprites:
        bbox = sprite["bbox"]
        direction = _quadrant_from_bbox(bbox, center)
        quadrant_counts[direction] += 1
        if quadrant_counts[direction] > 2:
            return False
        coords = sprite["coords"]
        for r, c in coords:
            if occupied[int(r), int(c)]:
                return False
        sprite_mask = np.zeros_like(grid, dtype=bool)
        sprite_mask[coords[:, 0], coords[:, 1]] = True
        occupied |= _dilate8(sprite_mask)

    largest = max(sprites, key=lambda sprite: sprite["area"])
    if sum(sprite["area"] == largest["area"] for sprite in sprites) != 1:
        return False
    if _quadrant_from_bbox(largest["bbox"], center) != expected_direction:
        return False

    if require_near_contrast:
        largest_dist = _bbox_center_distance(largest["bbox"], center)
        if not any(
            sprite is not largest and _bbox_center_distance(sprite["bbox"], center) < largest_dist for sprite in sprites
        ):
            return False

    if require_bbox_contrast:
        largest_dims = _bbox_dims(largest["bbox"])
        if not any(
            sprite is not largest and _bbox_dims(sprite["bbox"]) == largest_dims and sprite["area"] < largest["area"]
            for sprite in sprites
        ):
            return False

    return np.array_equal(main(grid), _ray_output(side, expected_direction))


def _build_attempt() -> np.ndarray | None:
    side, answer_direction = _next_case()
    n_objects = random.randint(4, 6)
    answer_idx = QUADRANT_DIRS.index(answer_direction)
    counts = random.choice(_all_count_patterns(n_objects, answer_idx))

    areas = sorted(random.sample(range(3, 10), n_objects), reverse=True)
    colors = random.sample(list(SPRITE_COLORS), n_objects)
    require_near_contrast = random.random() < 0.75
    require_bbox_contrast = random.random() < 0.8

    largest_patch = _make_patch(colors[0], areas[0], compact=True)
    largest_dims = largest_patch.shape

    sprite_quadrants = [answer_direction]
    remaining_counts = list(counts)
    remaining_counts[answer_idx] -= 1
    tail_quadrants: list[tuple[int, int]] = []
    for idx, count in enumerate(remaining_counts):
        tail_quadrants.extend([QUADRANT_DIRS[idx]] * count)
    random.shuffle(tail_quadrants)
    sprite_quadrants.extend(tail_quadrants)

    sprites = [
        {
            "area": areas[0],
            "color": colors[0],
            "quadrant": answer_direction,
            "patch": largest_patch,
            "mode": "far" if require_near_contrast else "random",
        }
    ]

    mimic_index: int | None = None
    if require_bbox_contrast:
        candidates = [
            idx
            for idx, area in enumerate(areas[1:], start=1)
            if largest_dims in _dim_choices(area)
            and sprite_quadrants[idx] != answer_direction
        ]
        if not candidates:
            candidates = [
                idx for idx, area in enumerate(areas[1:], start=1) if largest_dims in _dim_choices(area)
            ]
        if not candidates:
            return None
        mimic_index = random.choice(candidates)

    near_index: int | None = None
    if require_near_contrast:
        near_candidates = [idx for idx, q in enumerate(sprite_quadrants[1:], start=1) if q != answer_direction]
        if not near_candidates:
            return None
        near_index = random.choice(near_candidates)

    for idx in range(1, n_objects):
        preferred_dims = largest_dims if idx == mimic_index else None
        patch = _make_patch(colors[idx], areas[idx], preferred_dims=preferred_dims)
        sprites.append(
            {
                "area": areas[idx],
                "color": colors[idx],
                "quadrant": sprite_quadrants[idx],
                "patch": patch,
                "mode": "near" if idx == near_index else "random",
            }
        )

    grid = np.zeros((side, side), dtype=int)
    center = side // 2
    grid[center, center] = RAY_COLOR

    forbidden = np.zeros_like(grid, dtype=bool)
    forbidden[center - 1 : center + 2, center - 1 : center + 2] = True

    order = []
    if near_index is not None:
        order.append(near_index)
    order.append(0)
    order.extend(
        idx
        for idx in sorted(range(1, len(sprites)), key=lambda i: sprites[i]["area"], reverse=True)
        if idx not in order
    )

    for idx in order:
        sprite = sprites[idx]
        candidates = _candidate_positions(forbidden, sprite["patch"], sprite["quadrant"])
        chosen = _pick_candidate(candidates, sprite["mode"])
        if chosen is None:
            return None
        _place_patch(grid, forbidden, sprite["patch"], chosen[0], chosen[1])

    if not _valid_scene(grid, answer_direction, require_near_contrast, require_bbox_contrast):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
