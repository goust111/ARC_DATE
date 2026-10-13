from __future__ import annotations

import random

import numpy as np


BG = 0
COLOR_POOL = tuple(range(1, 10))
GRID_SIZE_RANGE = (9, 18)
PAIR_COUNT_RANGE = (2, 5)
RECT_SIDE_RANGE = (2, 7)
GEN_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 250


def _collect_pairs(grid: np.ndarray) -> list[tuple[int, list[tuple[int, int]]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    positions_by_color: dict[int, list[tuple[int, int]]] = {}
    for r, c in np.argwhere(g != BG):
        color = int(g[r, c])
        positions_by_color.setdefault(color, []).append((int(r), int(c)))

    assert positions_by_color, "expected at least one seed pair"

    ordered: list[tuple[int, list[tuple[int, int]]]] = []
    for color, positions in positions_by_color.items():
        assert len(positions) == 2, "every non-background color must appear exactly twice"
        (r0, c0), (r1, c1) = positions
        assert r0 != r1 and c0 != c1, "paired seeds must differ in both row and column"
        ordered.append((color, sorted(positions)))

    ordered.sort(key=lambda item: item[1][0])
    return ordered


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair same-color seeds and paint their bounding rectangles with a checkerboard.

    Each non-black color appears exactly twice. For every pair, use the axis-aligned
    rectangle defined by the two seed pixels, fill that rectangle with a checkerboard
    of the pair color and black, anchor the checkerboard so the rectangle's top-left
    cell is the pair color, then restore the two seed pixels. Pairs are processed in
    the scan order of each color's first occurrence; later rectangles overwrite earlier
    ones where they overlap.
    """
    g = np.asarray(grid, dtype=int)
    pairs = _collect_pairs(g)
    out = np.zeros_like(g)

    for color, positions in pairs:
        (ra, ca), (rb, cb) = positions
        r0, r1 = sorted((ra, rb))
        c0, c1 = sorted((ca, cb))

        region = out[r0 : r1 + 1, c0 : c1 + 1]
        rows = np.arange(r1 - r0 + 1)[:, None]
        cols = np.arange(c1 - c0 + 1)[None, :]
        mask = (rows + cols) % 2 == 0
        region[...] = BG
        region[mask] = color

        out[ra, ca] = color
        out[rb, cb] = color

    return out


def _intersection_area(
    rect_a: tuple[int, int, int, int], rect_b: tuple[int, int, int, int]
) -> int:
    ar0, ac0, ar1, ac1 = rect_a
    br0, bc0, br1, bc1 = rect_b
    r0 = max(ar0, br0)
    c0 = max(ac0, bc0)
    r1 = min(ar1, br1)
    c1 = min(ac1, bc1)
    if r0 > r1 or c0 > c1:
        return 0
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _area(rect: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = rect
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _corner_seeds(
    rect: tuple[int, int, int, int], orientation: int
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = rect
    if orientation == 0:
        return [(r0, c0), (r1, c1)]
    return [(r0, c1), (r1, c0)]


def _seeds_clear(
    seeds: list[tuple[int, int]], occupied: set[tuple[int, int]]
) -> bool:
    for r, c in seeds:
        if (r, c) in occupied:
            return False
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            if (r + dr, c + dc) in occupied:
                return False
    return True


def _sample_rect(height: int, width: int) -> tuple[int, int, int, int]:
    max_h = min(RECT_SIDE_RANGE[1], height)
    max_w = min(RECT_SIDE_RANGE[1], width)
    rect_h = random.randint(RECT_SIDE_RANGE[0], max_h)
    rect_w = random.randint(RECT_SIDE_RANGE[0], max_w)
    r0 = random.randint(0, height - rect_h)
    c0 = random.randint(0, width - rect_w)
    return (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1)


def _place_pairs(
    height: int, width: int, pair_count: int, overlap_mode: bool
) -> list[tuple[tuple[int, int, int, int], int]] | None:
    placed: list[tuple[tuple[int, int, int, int], int]] = []
    occupied_seeds: set[tuple[int, int]] = set()
    overlap_index = random.randint(1, pair_count - 1) if overlap_mode else -1

    for pair_idx in range(pair_count):
        must_overlap = pair_idx == overlap_index
        for _ in range(PLACEMENT_ATTEMPTS):
            rect = _sample_rect(height, width)
            orientation = random.randint(0, 1)
            seeds = _corner_seeds(rect, orientation)

            if not _seeds_clear(seeds, occupied_seeds):
                continue

            intersections = [_intersection_area(rect, old_rect) for old_rect, _ in placed]
            positive = [area for area in intersections if area > 0]
            partial = [
                area
                for area, (old_rect, _) in zip(intersections, placed)
                if 0 < area < min(_area(rect), _area(old_rect))
            ]

            if must_overlap:
                if len(positive) != 1 or len(partial) != 1:
                    continue
            elif positive:
                continue

            placed.append((rect, orientation))
            occupied_seeds.update(seeds)
            break
        else:
            return None

    return placed


def _build_grid(
    height: int,
    width: int,
    pairs: list[tuple[tuple[int, int, int, int], int]],
    colors: list[int],
) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    for color, (rect, orientation) in zip(colors, pairs):
        for r, c in _corner_seeds(rect, orientation):
            grid[r, c] = color
    return grid


def _has_overlap(pairs: list[tuple[tuple[int, int, int, int], int]]) -> bool:
    rects = [rect for rect, _ in pairs]
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            if _intersection_area(rects[i], rects[j]) > 0:
                return True
    return False


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.randint(*PAIR_COUNT_RANGE)
    height = random.randint(*GRID_SIZE_RANGE)
    width = random.randint(*GRID_SIZE_RANGE)
    overlap_mode = random.random() < 0.5

    pairs = _place_pairs(height, width, pair_count, overlap_mode)
    if pairs is None:
        return None

    if overlap_mode != _has_overlap(pairs):
        return None

    colors = random.sample(COLOR_POOL, pair_count)
    grid = _build_grid(height, width, pairs, colors)

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
