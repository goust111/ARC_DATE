from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7

SEED_COLORS = (BLUE, RED, GREEN, YELLOW, PINK)
GEN_RETRIES = 500


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _extract_regions(grid: np.ndarray) -> list[dict]:
    walkable = grid != GREY
    regions = []
    for comp in _components(walkable):
        seed_cells = [(r, c) for r, c in comp if int(grid[r, c]) != BLACK]
        assert len(seed_cells) == 1, "each region must contain exactly one seed"
        sr, sc = seed_cells[0]
        seed_color = int(grid[sr, sc])
        assert seed_color in SEED_COLORS, "unexpected seed color"
        r0, c0, r1, c1 = _bbox(comp)
        area = len(comp)
        assert area == (r1 - r0 + 1) * (c1 - c0 + 1), "regions must be rectangles"
        assert r1 - r0 + 1 >= 4 and c1 - c0 + 1 >= 4, "region smaller than 4x4"
        assert r0 < sr < r1 and c0 < sc < c1, "seed too close to a wall or boundary"
        regions.append(
            {
                "cells": comp,
                "bbox": (r0, c0, r1, c1),
                "area": area,
                "seed": (sr, sc),
                "color": seed_color,
            }
        )
    seed_colors = [region["color"] for region in regions]
    assert len(seed_colors) == len(set(seed_colors)), "seed colors must be unique"
    assert 3 <= len(regions) <= 5, "expected 3 to 5 regions"
    return regions


def _region_order(regions: list[dict]) -> list[dict]:
    return sorted(regions, key=lambda region: (region["area"], region["seed"][0], region["seed"][1]))


def _draw_shells(grid: np.ndarray, regions: list[dict]) -> np.ndarray:
    out = np.where(grid == GREY, GREY, BLACK).astype(int)
    depth_count = len(regions)
    h, w = grid.shape
    for region in _region_order(regions):
        sr, sc = region["seed"]
        seed_color = region["color"]
        for depth in range(depth_count):
            color = seed_color if depth % 2 == 0 else ORANGE
            if depth == 0:
                if out[sr, sc] == BLACK:
                    out[sr, sc] = color
                continue
            top = sr - depth
            bottom = sr + depth
            left = sc - depth
            right = sc + depth
            for c in range(left, right + 1):
                if _in_bounds((h, w), top, c) and out[top, c] == BLACK and grid[top, c] != GREY:
                    out[top, c] = color
                if _in_bounds((h, w), bottom, c) and out[bottom, c] == BLACK and grid[bottom, c] != GREY:
                    out[bottom, c] = color
            for r in range(top + 1, bottom):
                if _in_bounds((h, w), r, left) and out[r, left] == BLACK and grid[r, left] != GREY:
                    out[r, left] = color
                if _in_bounds((h, w), r, right) and out[r, right] == BLACK and grid[r, right] != GREY:
                    out[r, right] = color
    return out


def _reference_solve(grid: np.ndarray) -> np.ndarray:
    regions = _extract_regions(grid)
    out = np.where(grid == GREY, GREY, BLACK).astype(int)
    depth_count = len(regions)
    h, w = grid.shape
    for region in _region_order(regions):
        sr, sc = region["seed"]
        seed_color = region["color"]
        for r in range(max(0, sr - depth_count + 1), min(h, sr + depth_count)):
            for c in range(max(0, sc - depth_count + 1), min(w, sc + depth_count)):
                if grid[r, c] == GREY or out[r, c] != BLACK:
                    continue
                depth = max(abs(r - sr), abs(c - sc))
                if depth < depth_count:
                    out[r, c] = seed_color if depth % 2 == 0 else ORANGE
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Count wall-defined regions, then grow each seed into that many concentric square shells.

    Input structure:
    - The grid uses black background and one-cell-thick grey walls.
    - Grey walls partition the canvas into 3 to 5 rectangular regions.
    - Each region contains exactly one non-black seed pixel, and seed colors are unique.

    Rule:
    - Treat every non-grey connected area as one region and count the total number of regions N.
    - For each seed, draw concentric Chebyshev-distance square shells centered on that seed for depths 0 through N-1.
    - Even depths use the seed color, odd depths use orange, wall cells stay grey, out-of-bounds cells are ignored, and earlier seeds keep any overlapping pixels according to ascending region area then seed position.

    Key insight:
    - The shell depth is a global count derived from how many regions the grey walls create, while those same walls also clip the geometry of each seed's square growth.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    regions = _extract_regions(g)
    return _draw_shells(g, regions)


def _sample_parts(total: int, parts: int, minimum: int) -> list[int] | None:
    extra = total - parts * minimum
    if extra < 0:
        return None
    cuts = sorted(random.sample(range(extra + parts - 1), parts - 1)) if parts > 1 else []
    values = []
    prev = -1
    for cut in cuts + [extra + parts - 1]:
        values.append(cut - prev - 1)
        prev = cut
    return [value + minimum for value in values]


def _rand_from(values):
    return values[random.randrange(len(values))]


def _choose_dimensions(min_h: int, min_w: int, max_h: int = 20, max_w: int = 20) -> tuple[int, int]:
    return random.randint(min_h, max_h), random.randint(min_w, max_w)


def _pattern_vertical3() -> tuple[np.ndarray, list[tuple[int, int, int, int]]] | None:
    h, w = _choose_dimensions(13, 15)
    widths = _sample_parts(w - 2, 3, 4)
    if widths is None:
        return None
    grid = np.zeros((h, w), dtype=int)
    c1 = widths[0]
    c2 = widths[0] + 1 + widths[1]
    grid[:, c1] = GREY
    grid[:, c2] = GREY
    boxes = [
        (0, 0, h - 1, widths[0] - 1),
        (0, c1 + 1, h - 1, c2 - 1),
        (0, c2 + 1, h - 1, w - 1),
    ]
    return grid, boxes


def _pattern_horizontal3() -> tuple[np.ndarray, list[tuple[int, int, int, int]]] | None:
    h, w = _choose_dimensions(15, 13)
    heights = _sample_parts(h - 2, 3, 4)
    if heights is None:
        return None
    grid = np.zeros((h, w), dtype=int)
    r1 = heights[0]
    r2 = heights[0] + 1 + heights[1]
    grid[r1, :] = GREY
    grid[r2, :] = GREY
    boxes = [
        (0, 0, heights[0] - 1, w - 1),
        (r1 + 1, 0, r2 - 1, w - 1),
        (r2 + 1, 0, h - 1, w - 1),
    ]
    return grid, boxes


def _pattern_cross4() -> tuple[np.ndarray, list[tuple[int, int, int, int]]] | None:
    h, w = _choose_dimensions(13, 13)
    heights = _sample_parts(h - 1, 2, 4)
    widths = _sample_parts(w - 1, 2, 4)
    if heights is None or widths is None:
        return None
    grid = np.zeros((h, w), dtype=int)
    r = heights[0]
    c = widths[0]
    grid[r, :] = GREY
    grid[:, c] = GREY
    boxes = [
        (0, 0, r - 1, c - 1),
        (0, c + 1, r - 1, w - 1),
        (r + 1, 0, h - 1, c - 1),
        (r + 1, c + 1, h - 1, w - 1),
    ]
    return grid, boxes


def _pattern_t_vertical3() -> tuple[np.ndarray, list[tuple[int, int, int, int]]] | None:
    h, w = _choose_dimensions(13, 13)
    heights = _sample_parts(h - 1, 2, 4)
    widths = _sample_parts(w - 1, 2, 4)
    if heights is None or widths is None:
        return None
    split_left = random.choice((True, False))
    grid = np.zeros((h, w), dtype=int)
    r = heights[0]
    c = widths[0]
    grid[:, c] = GREY
    if split_left:
        grid[r, : c + 1] = GREY
        boxes = [
            (0, 0, r - 1, c - 1),
            (r + 1, 0, h - 1, c - 1),
            (0, c + 1, h - 1, w - 1),
        ]
    else:
        grid[r, c:] = GREY
        boxes = [
            (0, 0, h - 1, c - 1),
            (0, c + 1, r - 1, w - 1),
            (r + 1, c + 1, h - 1, w - 1),
        ]
    return grid, boxes


def _pattern_t_horizontal3() -> tuple[np.ndarray, list[tuple[int, int, int, int]]] | None:
    h, w = _choose_dimensions(13, 13)
    heights = _sample_parts(h - 1, 2, 4)
    widths = _sample_parts(w - 1, 2, 4)
    if heights is None or widths is None:
        return None
    split_top = random.choice((True, False))
    grid = np.zeros((h, w), dtype=int)
    r = heights[0]
    c = widths[0]
    grid[r, :] = GREY
    if split_top:
        grid[: r + 1, c] = GREY
        boxes = [
            (0, 0, r - 1, c - 1),
            (0, c + 1, r - 1, w - 1),
            (r + 1, 0, h - 1, w - 1),
        ]
    else:
        grid[r:, c] = GREY
        boxes = [
            (0, 0, r - 1, w - 1),
            (r + 1, 0, h - 1, c - 1),
            (r + 1, c + 1, h - 1, w - 1),
        ]
    return grid, boxes


def _pattern_cross5_vertical_extra() -> tuple[np.ndarray, list[tuple[int, int, int, int]]] | None:
    h, w = _choose_dimensions(13, 16)
    heights = _sample_parts(h - 1, 2, 4)
    widths = _sample_parts(w - 2, 3, 4)
    if heights is None or widths is None:
        return None
    top_extra = random.choice((True, False))
    grid = np.zeros((h, w), dtype=int)
    r = heights[0]
    c_extra = widths[0]
    c_main = widths[0] + 1 + widths[1]
    grid[r, :] = GREY
    grid[:, c_main] = GREY
    if top_extra:
        grid[: r + 1, c_extra] = GREY
        boxes = [
            (0, 0, r - 1, c_extra - 1),
            (0, c_extra + 1, r - 1, c_main - 1),
            (0, c_main + 1, r - 1, w - 1),
            (r + 1, 0, h - 1, c_main - 1),
            (r + 1, c_main + 1, h - 1, w - 1),
        ]
    else:
        grid[r:, c_extra] = GREY
        boxes = [
            (0, 0, r - 1, c_main - 1),
            (0, c_main + 1, r - 1, w - 1),
            (r + 1, 0, h - 1, c_extra - 1),
            (r + 1, c_extra + 1, h - 1, c_main - 1),
            (r + 1, c_main + 1, h - 1, w - 1),
        ]
    return grid, boxes


def _pattern_cross5_horizontal_extra() -> tuple[np.ndarray, list[tuple[int, int, int, int]]] | None:
    h, w = _choose_dimensions(16, 13)
    heights = _sample_parts(h - 2, 3, 4)
    widths = _sample_parts(w - 1, 2, 4)
    if heights is None or widths is None:
        return None
    left_extra = random.choice((True, False))
    grid = np.zeros((h, w), dtype=int)
    r_extra = heights[0]
    r_main = heights[0] + 1 + heights[1]
    c = widths[0]
    grid[:, c] = GREY
    grid[r_main, :] = GREY
    if left_extra:
        grid[r_extra, : c + 1] = GREY
        boxes = [
            (0, 0, r_extra - 1, c - 1),
            (r_extra + 1, 0, r_main - 1, c - 1),
            (0, c + 1, r_main - 1, w - 1),
            (r_main + 1, 0, h - 1, c - 1),
            (r_main + 1, c + 1, h - 1, w - 1),
        ]
    else:
        grid[r_extra, c:] = GREY
        boxes = [
            (0, 0, r_main - 1, c - 1),
            (0, c + 1, r_extra - 1, w - 1),
            (r_extra + 1, c + 1, r_main - 1, w - 1),
            (r_main + 1, 0, h - 1, c - 1),
            (r_main + 1, c + 1, h - 1, w - 1),
        ]
    return grid, boxes


PATTERN_BUILDERS = (
    _pattern_vertical3,
    _pattern_horizontal3,
    _pattern_cross4,
    _pattern_t_vertical3,
    _pattern_t_horizontal3,
    _pattern_cross5_vertical_extra,
    _pattern_cross5_horizontal_extra,
)


def _clearance(box: tuple[int, int, int, int], seed: tuple[int, int]) -> int:
    r0, c0, r1, c1 = box
    sr, sc = seed
    return min(sr - r0, r1 - sr, sc - c0, c1 - sc)


def _pick_seed(box: tuple[int, int, int, int], target_clearance: int | None = None) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    candidates = [
        (r, c)
        for r in range(r0 + 1, r1)
        for c in range(c0 + 1, c1)
        if _clearance(box, (r, c)) >= 1
    ]
    if not candidates:
        raise AssertionError("region has no legal seed position")
    if target_clearance is not None:
        good = [pos for pos in candidates if _clearance(box, pos) >= target_clearance]
        if good:
            candidates = good
    return random.choice(candidates)


def _place_seeds(grid: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    out = grid.copy()
    colors = random.sample(list(SEED_COLORS), len(boxes))
    showcase_index = max(
        range(len(boxes)),
        key=lambda idx: min(boxes[idx][2] - boxes[idx][0] + 1, boxes[idx][3] - boxes[idx][1] + 1),
    )
    target_depth = len(boxes) - 1
    placements: list[tuple[int, int]] = []
    for idx, box in enumerate(boxes):
        if idx == showcase_index:
            placements.append(_pick_seed(box, target_depth))
        else:
            prefer_edge = random.random() < 0.45
            if prefer_edge:
                r0, c0, r1, c1 = box
                edge_candidates = [
                    (r, c)
                    for r in range(r0 + 1, r1)
                    for c in range(c0 + 1, c1)
                    if _clearance(box, (r, c)) == 1
                ]
                placements.append(random.choice(edge_candidates) if edge_candidates else _pick_seed(box))
            else:
                placements.append(_pick_seed(box))
    for color, (r, c) in zip(colors, placements):
        out[r, c] = color
    return out


def _validate_input(grid: np.ndarray) -> None:
    assert isinstance(grid, np.ndarray)
    assert grid.ndim == 2 and max(grid.shape) <= 30
    regions = _extract_regions(grid)
    expected = _reference_solve(grid)
    actual = main(grid)
    assert np.array_equal(actual, expected), "solver mismatch"

    nonwall = np.count_nonzero(grid != GREY)
    assert nonwall > len(regions), "degenerate all-wall grid"

    orange_pixels = int(np.count_nonzero(actual == ORANGE))
    assert orange_pixels > 0, "need visible odd-depth shell pixels"

    for region in regions:
        sr, sc = region["seed"]
        for nr, nc in _neighbors4(sr, sc):
            assert _in_bounds(grid.shape, nr, nc)
            assert grid[nr, nc] != GREY


def _gen_attempt() -> np.ndarray | None:
    builder = _rand_from(PATTERN_BUILDERS)
    built = builder()
    if built is None:
        return None
    wall_grid, boxes = built
    random.shuffle(boxes)
    grid = _place_seeds(wall_grid, boxes)
    try:
        _validate_input(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
