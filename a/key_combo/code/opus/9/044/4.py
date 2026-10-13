from __future__ import annotations

from collections import defaultdict, deque
from itertools import combinations
import random

import numpy as np


PALETTE = (1, 2, 3, 4, 5)
BACKGROUND = 0
SPRITE_SIZE = 3
MIN_TYPES = 3
MAX_TYPES = 5
MIN_SPRITES = 8
MAX_SPRITES = 15
GRID_HEIGHT_RANGE = (15, 18)
GRID_WIDTH_RANGE = (15, 18)
RETRY_LIMIT = 400


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in row) for row in mask.astype(int))


def _is_mask_connected(mask: np.ndarray) -> bool:
    points = np.argwhere(mask)
    if points.size == 0:
        return False
    start = tuple(int(v) for v in points[0])
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors8(r, c):
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc]:
                if (nr, nc) not in seen:
                    seen.add((nr, nc))
                    q.append((nr, nc))
    return len(seen) == len(points)


def _build_mask_pool() -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    for count in range(3, 8):
        for cells in combinations(range(SPRITE_SIZE * SPRITE_SIZE), count):
            mask = np.zeros((SPRITE_SIZE, SPRITE_SIZE), dtype=bool)
            for idx in cells:
                mask[idx // SPRITE_SIZE, idx % SPRITE_SIZE] = True
            if not mask[0].any() or not mask[-1].any() or not mask[:, 0].any() or not mask[:, -1].any():
                continue
            if _is_mask_connected(mask):
                masks.append(mask)
    return masks


MASK_POOL = _build_mask_pool()


def _extract_components(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    seen = np.zeros(g.shape, dtype=bool)
    components: list[dict] = []
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            color = int(g[r, c])
            if color == BACKGROUND or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors8(cr, cc):
                    if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1]:
                        if not seen[nr, nc] and int(g[nr, nc]) == color:
                            seen[nr, nc] = True
                            q.append((nr, nc))
            coord_arr = np.array(coords, dtype=int)
            r0, c0, r1, c1 = _bbox_from_coords(coord_arr)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            rel = coord_arr - np.array([r0, c0], dtype=int)
            mask[rel[:, 0], rel[:, 1]] = True
            components.append(
                {
                    "color": color,
                    "coords": coord_arr,
                    "bbox": (r0, c0, r1, c1),
                    "mask": mask,
                }
            )
    return components


def _normalize_type(component: dict) -> tuple[tuple[int, tuple[tuple[int, ...], ...]], np.ndarray]:
    r0, c0, r1, c1 = component["bbox"]
    assert (r1 - r0 + 1, c1 - c0 + 1) == (SPRITE_SIZE, SPRITE_SIZE), "every sprite bbox must be 3x3"
    mask = component["mask"]
    key = (int(component["color"]), _mask_key(mask))
    return key, mask


def _sprite_grid(mask: np.ndarray, color: int) -> np.ndarray:
    sprite = np.zeros((SPRITE_SIZE, SPRITE_SIZE), dtype=int)
    sprite[mask] = int(color)
    return sprite


def _pixel_upscale(grid: np.ndarray, factor: int) -> np.ndarray:
    return np.kron(np.asarray(grid, dtype=int), np.ones((factor, factor), dtype=int))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Select the uniquely most frequent colored 3x3 sprite type and enlarge it by the number of sprite types.

    Input structure:
    - The grid has black background and 8-15 scattered monochrome sprites.
    - Each sprite is one 8-connected component whose bounding box is exactly 3x3.
    - A sprite type is defined jointly by its 3x3 foreground mask and its color.

    Rule:
    - Extract every 8-connected monochrome sprite and normalize it to its 3x3 mask plus color.
    - Count how many distinct sprite types appear; call this number K.
    - Count the instances of each type and choose the unique type T with strictly highest frequency.
    - Take T's 3x3 sprite and pixel-upscale it by factor K, so each cell becomes a KxK block.

    Key insight:
    - The same sprite population is counted in two orthogonal ways: type diversity sets the scale, while type frequency chooses which sprite gets rendered.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    components = _extract_components(g)
    assert MIN_SPRITES <= len(components) <= MAX_SPRITES, "unexpected sprite count"

    groups: dict[tuple[int, tuple[tuple[int, ...], ...]], list[np.ndarray]] = defaultdict(list)
    for component in components:
        key, mask = _normalize_type(component)
        groups[key].append(mask)

    k = len(groups)
    assert MIN_TYPES <= k <= MAX_TYPES, "unexpected type count"

    max_count = max(len(items) for items in groups.values())
    winners = [key for key, items in groups.items() if len(items) == max_count]
    assert len(winners) == 1, "expected a unique most frequent type"

    winner_color, _ = winners[0]
    winner_mask = groups[winners[0]][0]
    return _pixel_upscale(_sprite_grid(winner_mask, winner_color), k)


def _sample_counts(k: int) -> list[int]:
    target_total = random.randint(MIN_SPRITES, MAX_SPRITES)
    solutions: list[list[int]] = []

    def rec(prefix: list[int], remaining: int, slots_left: int, cap: int, sink: list[list[int]]) -> None:
        if slots_left == 1:
            if 1 <= remaining <= cap:
                sink.append(prefix + [remaining])
            return
        min_rest = slots_left - 1
        max_rest = (slots_left - 1) * cap
        lo = max(1, remaining - max_rest)
        hi = min(cap, remaining - min_rest)
        for value in range(lo, hi + 1):
            rec(prefix + [value], remaining - value, slots_left - 1, cap, sink)

    for winner in range(3, min(6, target_total - (k - 1)) + 1):
        other_solutions: list[list[int]] = []
        rec([], target_total - winner, k - 1, winner - 1, other_solutions)
        solutions.extend([ [winner] + other_counts for other_counts in other_solutions ])

    if not solutions:
        raise RuntimeError("failed to sample sprite-type counts")

    near_tie = [counts for counts in solutions if max(counts[1:]) == counts[0] - 1]
    pool = near_tie if near_tie and random.random() < 0.7 else solutions
    return random.choice(pool)


def _choose_types(k: int) -> list[tuple[np.ndarray, int]]:
    mask_a, mask_b = random.sample(MASK_POOL, 2)
    color_x, color_y = random.sample(PALETTE, 2)

    types: list[tuple[np.ndarray, int]] = [
        (mask_a.copy(), color_x),
        (mask_a.copy(), color_y),
        (mask_b.copy(), color_x),
    ]
    used = {(color_x, _mask_key(mask_a)), (color_y, _mask_key(mask_a)), (color_x, _mask_key(mask_b))}

    if len(types) < k:
        all_options = [(mask, color) for mask in MASK_POOL for color in PALETTE]
        random.shuffle(all_options)
        for mask, color in all_options:
            key = (int(color), _mask_key(mask))
            if key in used:
                continue
            types.append((mask.copy(), int(color)))
            used.add(key)
            if len(types) == k:
                break

    assert len(types) == k
    shape_keys = {_mask_key(mask) for mask, _ in types}
    colors = {int(color) for _, color in types}
    assert len(shape_keys) < k and len(colors) < k
    return types


def _render_sprite(mask: np.ndarray, color: int) -> np.ndarray:
    sprite = np.zeros((SPRITE_SIZE, SPRITE_SIZE), dtype=int)
    sprite[mask] = int(color)
    return sprite


def _place_sprites(
    height: int,
    width: int,
    sprite_instances: list[tuple[np.ndarray, int]],
) -> np.ndarray | None:
    grid = np.zeros((height, width), dtype=int)
    offset_options: list[tuple[list[int], list[int]]] = []
    for row_offset in range(4):
        rows = list(range(row_offset, height - SPRITE_SIZE + 1, 4))
        if not rows:
            continue
        for col_offset in range(4):
            cols = list(range(col_offset, width - SPRITE_SIZE + 1, 4))
            if cols and len(rows) * len(cols) >= len(sprite_instances):
                offset_options.append((rows, cols))
    if not offset_options:
        return None

    rows, cols = random.choice(offset_options)
    sites = [(r, c) for r in rows for c in cols]
    chosen_sites = random.sample(sites, len(sprite_instances))

    for (mask, color), (r0, c0) in zip(sprite_instances, chosen_sites):
        grid[r0 : r0 + SPRITE_SIZE, c0 : c0 + SPRITE_SIZE] = _render_sprite(mask, color)
    return grid


def _analyze_grid(grid: np.ndarray) -> dict:
    components = _extract_components(grid)
    groups: dict[tuple[int, tuple[tuple[int, ...], ...]], list[dict]] = defaultdict(list)
    color_to_shapes: dict[int, set[tuple[tuple[int, ...], ...]]] = defaultdict(set)
    shape_to_colors: dict[tuple[tuple[int, ...], ...], set[int]] = defaultdict(set)

    for component in components:
        key, mask = _normalize_type(component)
        color, shape_key = key
        groups[key].append(component)
        color_to_shapes[color].add(shape_key)
        shape_to_colors[shape_key].add(color)

    counts = {key: len(items) for key, items in groups.items()}
    winner_key = max(counts, key=counts.get)
    return {
        "components": components,
        "groups": groups,
        "counts": counts,
        "winner_key": winner_key,
        "winner_mask": groups[winner_key][0]["mask"],
        "winner_color": int(winner_key[0]),
        "k": len(groups),
        "unique_colors": len(color_to_shapes),
        "unique_shapes": len(shape_to_colors),
        "winner_same_color_types": len(color_to_shapes[int(winner_key[0])]),
        "winner_same_shape_types": len(shape_to_colors[winner_key[1]]),
    }


def _semantic_checks(grid: np.ndarray, expected_counts: list[int]) -> bool:
    analysis = _analyze_grid(grid)
    counts = analysis["counts"]
    sorted_counts = sorted(counts.values(), reverse=True)
    expected_sorted = sorted(expected_counts, reverse=True)

    if len(analysis["components"]) != sum(expected_counts):
        return False
    if analysis["k"] != len(expected_counts):
        return False
    if sorted_counts != expected_sorted:
        return False
    if analysis["unique_colors"] >= analysis["k"]:
        return False
    if analysis["unique_shapes"] >= analysis["k"]:
        return False
    if analysis["winner_same_color_types"] < 2:
        return False
    if analysis["winner_same_shape_types"] < 2:
        return False
    winners = [key for key, count in counts.items() if count == max(counts.values())]
    if len(winners) != 1:
        return False

    expected = _pixel_upscale(_render_sprite(analysis["winner_mask"], analysis["winner_color"]), analysis["k"])
    if not np.array_equal(main(grid), expected):
        return False
    return True


def _attempt() -> np.ndarray | None:
    k = random.randint(MIN_TYPES, MAX_TYPES)
    counts = _sample_counts(k)
    types = _choose_types(k)

    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)

    instances = [types[idx] for idx, count in enumerate(counts) for _ in range(count)]
    random.shuffle(instances)
    grid = _place_sprites(height, width, instances)
    if grid is None:
        return None
    if not _semantic_checks(grid, counts):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(RETRY_LIMIT):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
