from __future__ import annotations

import random

import numpy as np


SEED_COLOR = 7
INNER_RING_COLORS = (5, 8, 9)
BASE_COLORS = (1, 2, 3, 4, 6)


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert len(coords) > 0, "expected a non-empty mask"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _extract_blocks(grid: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    blocks: list[tuple[int, int, int, int, int]] = []
    for value in np.unique(grid):
        color = int(value)
        if color == SEED_COLOR:
            continue
        r0, c0, r1, c1 = _bbox_from_mask(grid == color)
        region = grid[r0 : r1 + 1, c0 : c1 + 1]
        assert np.all(
            (region == color) | (region == SEED_COLOR)
        ), "each visible color must belong to one seeded rectangle"
        blocks.append((color, r0, c0, r1, c1))

    assert blocks, "expected at least one non-seed rectangle color"

    cover = np.zeros_like(grid, dtype=int)
    for _color, r0, c0, r1, c1 in blocks:
        cover[r0 : r1 + 1, c0 : c1 + 1] += 1
    assert np.all(cover == 1), "rectangles must tile the whole grid without overlap"
    return blocks


def _paint_block(
    out: np.ndarray,
    *,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    base_color: int,
    seed_count: int,
) -> None:
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    layer_count = min(seed_count, (min(height, width) + 1) // 2)
    palette = (base_color,) + INNER_RING_COLORS[: layer_count - 1]

    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            depth = min(r - r0, r1 - r, c - c0, c1 - c)
            out[r, c] = palette[min(depth, layer_count - 1)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repaint each tiled rectangle into concentric rings whose depth is set by its orange seed count.

    Input structure:
    - The whole grid is a tight tiling of solid rectangular blocks with no background.
    - Each block has its own original color, while orange singleton pixels appear inside the block as seeds.
    - Every visible non-orange color belongs to exactly one axis-aligned rectangle, and orange is used only for seeds.

    Rule:
    - Recover each rectangle from the bounding box of its visible original-color pixels.
    - Count the orange seed pixels inside that rectangle; this count N is between 1 and 4.
    - Repaint the rectangle as concentric 1-pixel rings colored from outside to inside as original color, then grey, then azure, then maroon.
    - If the rectangle is too small to realize all N rings, drop the innermost missing rings first and keep the outer rings.
    - Seed pixels are overwritten by the ring colors in the output.

    Key insight:
    - The orange pixels do not grow or move; they encode how many inward ring layers the surrounding rectangle should keep when it is peeled from the border toward the center.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    blocks = _extract_blocks(g)
    out = np.empty_like(g)

    for color, r0, c0, r1, c1 in blocks:
        region = g[r0 : r1 + 1, c0 : c1 + 1]
        seed_count = int(np.sum(region == SEED_COLOR))
        assert 1 <= seed_count <= 4, "each block must contain between 1 and 4 seeds"
        _paint_block(
            out,
            r0=r0,
            c0=c0,
            r1=r1,
            c1=c1,
            base_color=color,
            seed_count=seed_count,
        )

    return out


MIN_SIDE = 3
GRID_HEIGHT_RANGE = (10, 22)
GRID_WIDTH_RANGE = (10, 22)
BLOCK_COUNT_RANGE = (2, 5)
TRUNCATION_PROBABILITY = 0.5
MAX_GENERATION_ATTEMPTS = 400

Rect = tuple[int, int, int, int]


def _rect_height(rect: Rect) -> int:
    return rect[1] - rect[0]


def _rect_width(rect: Rect) -> int:
    return rect[3] - rect[2]


def _rect_area(rect: Rect) -> int:
    return _rect_height(rect) * _rect_width(rect)


def _ring_capacity(rect: Rect) -> int:
    return (min(_rect_height(rect), _rect_width(rect)) + 1) // 2


def _seed_capacity(rect: Rect) -> int:
    return max(0, (_rect_height(rect) - 2) * (_rect_width(rect) - 2))


def _split_options(rect: Rect) -> list[tuple[str, int]]:
    r0, r1, c0, c1 = rect
    options: list[tuple[str, int]] = []
    for split in range(r0 + MIN_SIDE, r1 - MIN_SIDE + 1):
        options.append(("h", split))
    for split in range(c0 + MIN_SIDE, c1 - MIN_SIDE + 1):
        options.append(("v", split))
    return options


def _partition_canvas(height: int, width: int, target_blocks: int) -> list[Rect] | None:
    rects: list[Rect] = [(0, height, 0, width)]
    while len(rects) < target_blocks:
        candidates: list[tuple[int, list[tuple[str, int]]]] = []
        weights: list[int] = []
        for index, rect in enumerate(rects):
            options = _split_options(rect)
            if options:
                candidates.append((index, options))
                weights.append(_rect_area(rect))
        if not candidates:
            return None

        chosen = random.choices(range(len(candidates)), weights=weights, k=1)[0]
        rect_index, options = candidates[chosen]
        axis, split = random.choice(options)
        r0, r1, c0, c1 = rects.pop(rect_index)
        if axis == "h":
            rects.extend([(r0, split, c0, c1), (split, r1, c0, c1)])
        else:
            rects.extend([(r0, r1, c0, split), (r0, r1, split, c1)])
    random.shuffle(rects)
    return rects


def _assign_seed_counts(rects: list[Rect]) -> list[int] | None:
    capacities = [_seed_capacity(rect) for rect in rects]
    ring_caps = [_ring_capacity(rect) for rect in rects]

    focus_n = random.randint(1, 4)
    if not any(capacity >= focus_n for capacity in capacities):
        return None

    counts: list[int | None] = [None] * len(rects)

    if random.random() < TRUNCATION_PROBABILITY:
        trunc_n = random.randint(2, 4)
        trunc_candidates = [
            index
            for index, (capacity, ring_cap) in enumerate(zip(capacities, ring_caps))
            if capacity >= trunc_n and ring_cap < trunc_n
        ]
        if not trunc_candidates:
            return None
        trunc_index = random.choice(trunc_candidates)
        counts[trunc_index] = trunc_n

    focus_candidates = [
        index
        for index, capacity in enumerate(capacities)
        if capacity >= focus_n and (counts[index] is None or counts[index] == focus_n)
    ]
    if not focus_candidates:
        return None
    focus_index = random.choice(focus_candidates)
    counts[focus_index] = focus_n

    for index, capacity in enumerate(capacities):
        if counts[index] is not None:
            continue
        options = list(range(1, min(4, capacity) + 1))
        if not options:
            return None
        trunc_options = [value for value in options if ring_caps[index] < value]
        if trunc_options and random.random() < 0.35:
            counts[index] = random.choice(trunc_options)
        else:
            counts[index] = random.choice(options)

    return [int(count) for count in counts]


def _interior_cells(rect: Rect) -> list[tuple[int, int]]:
    r0, r1, c0, c1 = rect
    return [(r, c) for r in range(r0 + 1, r1 - 1) for c in range(c0 + 1, c1 - 1)]


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    target_blocks = random.randint(*BLOCK_COUNT_RANGE)

    rects = _partition_canvas(height, width, target_blocks)
    if rects is None:
        return None
    if len(rects) > len(BASE_COLORS):
        return None

    counts = _assign_seed_counts(rects)
    if counts is None:
        return None

    grid = np.empty((height, width), dtype=int)
    colors = random.sample(list(BASE_COLORS), len(rects))

    for rect, color in zip(rects, colors):
        r0, r1, c0, c1 = rect
        grid[r0:r1, c0:c1] = color

    for rect, count in zip(rects, counts):
        cells = _interior_cells(rect)
        if len(cells) < count:
            return None
        for r, c in random.sample(cells, count):
            grid[r, c] = SEED_COLOR

    solved = main(grid)
    assert solved.shape == grid.shape
    assert np.all(solved != SEED_COLOR), "output should not preserve seed color"
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid seeded-rectangle mosaic")
