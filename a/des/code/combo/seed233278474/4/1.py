import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7

RING = (RED, ORANGE, YELLOW, GREEN, BLUE, MAGENTA)
RING_SET = set(RING)
RING_NEXT = {RING[i]: RING[(i + 1) % len(RING)] for i in range(len(RING))}
RING_PREV = {RING[(i + 1) % len(RING)]: RING[i] for i in range(len(RING))}
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


GEN_RETRIES = 400
GRID_H_RANGE = (11, 21)
GRID_W_RANGE = (11, 21)
REGION_COUNT_WEIGHTS = ((2, 2), (3, 3), (4, 3), (5, 2))
MIN_REGION_SIDE_CHOICES = (2, 3)


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _touches_edge(coords: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in coords)


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _region_center(region: dict) -> tuple[float, float]:
    r0, c0, r1, c1 = region["bbox"]
    return (r0 + r1) / 2.0, (c0 + c1) / 2.0


def _direction_between(src: dict, dst: dict) -> tuple[int, int]:
    sr, sc = _region_center(src)
    tr, tc = _region_center(dst)
    dr = tr - sr
    dc = tc - sc
    if abs(dr) > abs(dc):
        return (1, 0) if dr > 0 else (-1, 0)
    return (0, 1) if dc > 0 else (0, -1)


def _ordered_colors(present: set[int]) -> list[int]:
    starts = [color for color in present if RING_PREV[color] not in present]
    assert len(starts) == 1, "seed colors must form a single directed arc on the color ring"
    order: list[int] = []
    color = starts[0]
    while color in present:
        order.append(color)
        color = RING_NEXT[color]
        if color == starts[0]:
            break
    assert len(order) == len(present), "seed colors must not wrap into a full cycle"
    return order


def _analyze_regions(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    regions: list[dict] = []
    for coords in _components(g != GREY):
        if _touches_edge(coords, h, w):
            continue
        seed_cells = [(r, c, int(g[r, c])) for r, c in coords if int(g[r, c]) in RING_SET]
        assert len(seed_cells) == 1, "each enclosed region must contain exactly one ring-color seed"
        r0, c0, r1, c1 = _bbox(coords)
        cells = {(r, c) for r, c in coords}
        sr, sc, color = seed_cells[0]
        regions.append(
            {
                "bbox": (r0, c0, r1, c1),
                "cells": cells,
                "seed": (sr, sc),
                "color": color,
            }
        )
    assert regions, "expected at least one enclosed region"
    return regions


def _draw_arrow(
    out: np.ndarray,
    region_cells: set[tuple[int, int]],
    seed: tuple[int, int],
    direction: tuple[int, int],
    color: int,
) -> None:
    dr, dc = direction
    line = [seed]
    r, c = seed
    for _ in range(2):
        nr, nc = r + dr, c + dc
        if (nr, nc) not in region_cells:
            break
        line.append((nr, nc))
        r, c = nr, nc

    for rr, cc in line:
        out[rr, cc] = color

    tip_r, tip_c = line[-1]
    if dr != 0:
        wings = ((tip_r, tip_c - 1), (tip_r, tip_c + 1))
    else:
        wings = ((tip_r - 1, tip_c), (tip_r + 1, tip_c))
    for wr, wc in wings:
        if (wr, wc) in region_cells:
            out[wr, wc] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Chain enclosed seeded regions by visible color-ring adjacency, then draw a
    clipped T-shaped arrow in each region toward its successor.
    """
    g = np.asarray(grid, dtype=int)
    regions = _analyze_regions(g)
    by_color = {region["color"]: region for region in regions}
    assert len(by_color) == len(regions), "seed colors must be unique"

    order = _ordered_colors(set(by_color))
    out = g.copy()
    for color, next_color in zip(order, order[1:]):
        src = by_color[color]
        dst = by_color[next_color]
        direction = _direction_between(src, dst)
        _draw_arrow(out, src["cells"], src["seed"], direction, color)
    return out


def _choose_region_count(rng: random.Random) -> int:
    values = [count for count, _ in REGION_COUNT_WEIGHTS]
    weights = [weight for _, weight in REGION_COUNT_WEIGHTS]
    return rng.choices(values, weights=weights, k=1)[0]


def _rect_area(rect: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = rect
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _split_options(rect: tuple[int, int, int, int], min_side: int) -> list[str]:
    r0, c0, r1, c1 = rect
    options = []
    if r1 - r0 + 1 >= 2 * min_side + 1:
        options.append("h")
    if c1 - c0 + 1 >= 2 * min_side + 1:
        options.append("v")
    return options


def _partition_rectangles(
    h: int, w: int, n_regions: int, min_side: int, rng: random.Random
) -> tuple[list[tuple[int, int, int, int]], list[tuple[str, int, int, int]]] | None:
    rects = [(1, 1, h - 2, w - 2)]
    lines: list[tuple[str, int, int, int]] = []

    for _ in range(n_regions - 1):
        candidates = [
            (idx, rect, _split_options(rect, min_side))
            for idx, rect in enumerate(rects)
            if _split_options(rect, min_side)
        ]
        if not candidates:
            return None

        weights = [_rect_area(rect) for _, rect, _ in candidates]
        pick_idx = rng.choices(range(len(candidates)), weights=weights, k=1)[0]
        rect_index, rect, options = candidates[pick_idx]
        rects.pop(rect_index)
        r0, c0, r1, c1 = rect
        orientation = rng.choice(options)
        if orientation == "h":
            cut = rng.randint(r0 + min_side, r1 - min_side)
            rects.append((r0, c0, cut - 1, c1))
            rects.append((cut + 1, c0, r1, c1))
            lines.append(("h", cut, c0 - 1, c1 + 1))
        else:
            cut = rng.randint(c0 + min_side, c1 - min_side)
            rects.append((r0, c0, r1, cut - 1))
            rects.append((r0, cut + 1, r1, c1))
            lines.append(("v", cut, r0 - 1, r1 + 1))

    if len(rects) != n_regions:
        return None
    return rects, lines


def _forward_room(rect: tuple[int, int, int, int], seed: tuple[int, int], direction: tuple[int, int]) -> int:
    r0, c0, r1, c1 = rect
    r, c = seed
    dr, dc = direction
    if dr == 1:
        return r1 - r
    if dr == -1:
        return r - r0
    if dc == 1:
        return c1 - c
    return c - c0


def _seed_choices(
    rect: tuple[int, int, int, int], direction: tuple[int, int] | None
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]]:
    r0, c0, r1, c1 = rect
    all_cells = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
    if direction is None:
        return all_cells, [], []
    full = [cell for cell in all_cells if _forward_room(rect, cell, direction) >= 2]
    tight = [cell for cell in all_cells if _forward_room(rect, cell, direction) < 2]
    return all_cells, full, tight


def _pick_seed(
    rect: tuple[int, int, int, int], direction: tuple[int, int] | None, rng: random.Random
) -> tuple[int, int]:
    all_cells, full, tight = _seed_choices(rect, direction)
    if direction is None:
        return rng.choice(all_cells)

    branches: list[list[tuple[int, int]]] = []
    if full:
        branches.extend([full, full])
    if tight:
        branches.extend([tight, tight])
    branches.append(all_cells)
    branch = rng.choice(branches)
    return rng.choice(branch)


def _render_grid(
    h: int,
    w: int,
    rects: list[tuple[int, int, int, int]],
    lines: list[tuple[str, int, int, int]],
    color_by_rect: dict[tuple[int, int, int, int], int],
    seed_by_rect: dict[tuple[int, int, int, int], tuple[int, int]],
) -> np.ndarray:
    grid = np.full((h, w), BLACK, dtype=int)
    grid[0, :] = GREY
    grid[h - 1, :] = GREY
    grid[:, 0] = GREY
    grid[:, w - 1] = GREY
    for orientation, cut, a0, a1 in lines:
        if orientation == "h":
            grid[cut, a0 : a1 + 1] = GREY
        else:
            grid[a0 : a1 + 1, cut] = GREY
    for rect in rects:
        sr, sc = seed_by_rect[rect]
        grid[sr, sc] = color_by_rect[rect]
    return grid


def _direction_sequence(order_rects: list[tuple[int, int, int, int]]) -> list[tuple[int, int]]:
    regions = [{"bbox": rect} for rect in order_rects]
    return [_direction_between(src, dst) for src, dst in zip(regions, regions[1:])]


def _layout_ok(order_rects: list[tuple[int, int, int, int]], rng: random.Random) -> bool:
    if len(order_rects) < 3:
        return True
    dirs = _direction_sequence(order_rects)
    unique = len(set(dirs))
    if unique >= 2:
        return True
    return rng.random() < 0.25


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    n_regions = _choose_region_count(rng)
    min_side = rng.choice(MIN_REGION_SIDE_CHOICES)

    partition = _partition_rectangles(h, w, n_regions, min_side, rng)
    if partition is None:
        return None
    rects, lines = partition
    order_rects = rects[:]
    rng.shuffle(order_rects)
    if not _layout_ok(order_rects, rng):
        return None

    start_idx = rng.randrange(len(RING))
    color_order = [RING[(start_idx + i) % len(RING)] for i in range(n_regions)]
    color_by_rect = {rect: color for rect, color in zip(order_rects, color_order)}

    seed_by_rect: dict[tuple[int, int, int, int], tuple[int, int]] = {}
    for idx, rect in enumerate(order_rects):
        if idx + 1 < len(order_rects):
            direction = _direction_between({"bbox": rect}, {"bbox": order_rects[idx + 1]})
        else:
            direction = None
        seed_by_rect[rect] = _pick_seed(rect, direction, rng)

    grid = _render_grid(h, w, rects, lines, color_by_rect, seed_by_rect)

    try:
        regions = _analyze_regions(grid)
        if len(regions) != n_regions:
            return None
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None

    # Keep some samples with clipped arrows, but do not require it every time.
    if rng.random() < 0.5:
        extra_pixels = int((out != grid).sum())
        if extra_pixels < max(2, n_regions - 1):
            return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
