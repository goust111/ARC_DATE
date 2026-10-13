from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
GREY = 5


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected a nonempty mask"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _dominant_non_background_color(grid: np.ndarray) -> int:
    counts = Counter(int(v) for v in grid.ravel() if int(v) != BG)
    assert counts, "expected at least one non-background color"
    ranked = counts.most_common()
    assert len(ranked) == 1 or ranked[0][1] > ranked[1][1], "ring color must be uniquely dominant"
    return int(ranked[0][0])


def _ring_cycle(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        coords.append((r0, c))
    for r in range(r0 + 1, r1):
        coords.append((r, c1))
    for c in range(c1, c0 - 1, -1):
        coords.append((r1, c))
    for r in range(r1 - 1, r0, -1):
        coords.append((r, c0))
    return coords


def _count_gap_runs(values: list[int]) -> int:
    is_gap = [int(v) == BG for v in values]
    return sum(flag and not is_gap[i - 1] for i, flag in enumerate(is_gap))


def _extract_palette(grid: np.ndarray, ring_color: int) -> tuple[tuple[int, int, int, int], np.ndarray]:
    palette_mask = (grid != BG) & (grid != int(ring_color))
    bbox = _bbox_from_mask(palette_mask)
    r0, c0, r1, c1 = bbox
    palette = grid[r0 : r1 + 1, c0 : c1 + 1]
    assert bool(np.all(palette_mask[r0 : r1 + 1, c0 : c1 + 1])), "palette must be a solid rectangle"
    for c in range(palette.shape[1]):
        column = palette[:, c]
        assert len(set(map(int, column))) == 1, "each palette column must be uniform"
    return bbox, palette


def _detect_ring_cycles(
    grid: np.ndarray,
    palette_bbox: tuple[int, int, int, int],
    ring_color: int,
) -> list[list[tuple[int, int]]]:
    h, w = map(int, grid.shape)
    pr0, pc0, pr1, pc1 = palette_bbox
    layers: list[list[tuple[int, int]]] = []
    layer = 1

    while True:
        gap_r0 = pr0 - (2 * layer - 1)
        gap_c0 = pc0 - (2 * layer - 1)
        gap_r1 = pr1 + (2 * layer - 1)
        gap_c1 = pc1 + (2 * layer - 1)
        ring_r0 = pr0 - 2 * layer
        ring_c0 = pc0 - 2 * layer
        ring_r1 = pr1 + 2 * layer
        ring_c1 = pc1 + 2 * layer

        if ring_r0 < 0 or ring_c0 < 0 or ring_r1 >= h or ring_c1 >= w:
            break

        gap_vals = [int(grid[r, c]) for r, c in _ring_cycle(gap_r0, gap_c0, gap_r1, gap_c1)]
        assert all(v == BG for v in gap_vals), "adjacent layers must be separated by a black gap"

        ring_cycle = _ring_cycle(ring_r0, ring_c0, ring_r1, ring_c1)
        ring_vals = [int(grid[r, c]) for r, c in ring_cycle]
        assert all(v in (BG, int(ring_color)) for v in ring_vals), "ring cells must be grey or black gaps"
        if all(v == BG for v in ring_vals):
            break

        layers.append(ring_cycle)
        layer += 1

    assert layers, "expected at least one ring layer"
    return layers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each nested rectangular ring by the palette column indexed by its gap count.

    Input structure:
    - The background is black.
    - A centered nest of one-pixel rectangular rings surrounds an inner solid palette rectangle.
    - Adjacent rings are separated by one-pixel black gaps, and each ring is mostly grey with
      one or more black breaks along its perimeter.
    - The palette rectangle has uniform-color columns from left to right.

    Rule:
    - Identify the innermost palette rectangle and then scan outward through alternating black
      gap shells and grey ring shells.
    - For each ring, count how many separate black gap segments appear around that ring.
    - Use the color of the palette's K-th column from the left, where K is that ring's gap count.
    - Repaint every grey pixel of the ring to that color while keeping the black gap pixels and
      the palette itself unchanged.

    Key insight:
    The central colored block is a legend: each ring must first be isolated structurally, then its
    number of perimeter breaks maps to a palette column, which determines the ring's output color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    ring_color = _dominant_non_background_color(g)
    palette_bbox, palette = _extract_palette(g, ring_color)
    ring_cycles = _detect_ring_cycles(g, palette_bbox, ring_color)

    out = g.copy()
    palette_width = int(palette.shape[1])
    for ring_cycle in ring_cycles:
        ring_vals = [int(g[r, c]) for r, c in ring_cycle]
        gap_count = _count_gap_runs(ring_vals)
        assert 1 <= gap_count <= palette_width, "gap count must map to a palette column"
        target_color = int(palette[0, gap_count - 1])
        for r, c in ring_cycle:
            if int(g[r, c]) == int(ring_color):
                out[r, c] = target_color

    return out


# --------------- generator ---------------

PALETTE_COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
RING_COUNT_RANGE = (2, 5)
PALETTE_HEIGHT_RANGE = (2, 5)
PALETTE_WIDTH_RANGE = (3, 8)
GRID_LIMIT = 30
MIN_MARGIN = 1
GEN_ATTEMPTS = 400
GAP_POSITION_ATTEMPTS = 200


def _sample_gap_positions(cycle_len: int, gap_count: int) -> list[int] | None:
    assert gap_count >= 1
    if gap_count == 1:
        return [random.randrange(cycle_len)]

    for _ in range(GAP_POSITION_ATTEMPTS):
        picks = sorted(random.sample(range(cycle_len), gap_count))
        ok = True
        for i, cur in enumerate(picks):
            nxt = picks[(i + 1) % gap_count]
            if (nxt - cur) % cycle_len <= 1:
                ok = False
                break
        if ok:
            return picks
    return None


def _draw_ring(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    gap_count: int,
) -> tuple[list[tuple[int, int]], set[int]] | None:
    ring_cycle = _ring_cycle(*bbox)
    gap_positions = _sample_gap_positions(len(ring_cycle), gap_count)
    if gap_positions is None:
        return None

    gap_set = set(gap_positions)
    for idx, (r, c) in enumerate(ring_cycle):
        grid[r, c] = BG if idx in gap_set else GREY
    return ring_cycle, gap_set


def _grey_is_uniquely_dominant(grid: np.ndarray) -> bool:
    counts = Counter(int(v) for v in grid.ravel() if int(v) != BG)
    ranked = counts.most_common()
    return bool(ranked) and ranked[0][0] == GREY and (len(ranked) == 1 or ranked[0][1] > ranked[1][1])


def _build_case() -> tuple[np.ndarray, np.ndarray] | None:
    ring_count = random.randint(*RING_COUNT_RANGE)
    max_palette_h = min(PALETTE_HEIGHT_RANGE[1], GRID_LIMIT - 2 * MIN_MARGIN - 4 * ring_count)
    max_palette_w = min(PALETTE_WIDTH_RANGE[1], GRID_LIMIT - 2 * MIN_MARGIN - 4 * ring_count)
    min_palette_w = max(PALETTE_WIDTH_RANGE[0], ring_count)
    if max_palette_h < PALETTE_HEIGHT_RANGE[0] or max_palette_w < min_palette_w:
        return None

    palette_h = random.randint(PALETTE_HEIGHT_RANGE[0], max_palette_h)
    palette_w = random.randint(min_palette_w, max_palette_w)
    outer_h = palette_h + 4 * ring_count
    outer_w = palette_w + 4 * ring_count
    max_margin_h = (GRID_LIMIT - outer_h) // 2
    max_margin_w = (GRID_LIMIT - outer_w) // 2
    if max_margin_h < MIN_MARGIN or max_margin_w < MIN_MARGIN:
        return None

    margin_h = random.randint(MIN_MARGIN, max_margin_h)
    margin_w = random.randint(MIN_MARGIN, max_margin_w)
    height = outer_h + 2 * margin_h
    width = outer_w + 2 * margin_w

    grid = np.full((height, width), BG, dtype=int)
    expected = np.full((height, width), BG, dtype=int)

    palette_colors = np.array(random.sample(PALETTE_COLOR_POOL, palette_w), dtype=int)
    gap_counts = random.sample(range(1, palette_w + 1), ring_count)

    pr0 = margin_h + 2 * ring_count
    pc0 = margin_w + 2 * ring_count
    palette = np.tile(palette_colors, (palette_h, 1))
    grid[pr0 : pr0 + palette_h, pc0 : pc0 + palette_w] = palette
    expected[pr0 : pr0 + palette_h, pc0 : pc0 + palette_w] = palette

    for layer, gap_count in enumerate(gap_counts, start=1):
        bbox = (
            pr0 - 2 * layer,
            pc0 - 2 * layer,
            pr0 + palette_h - 1 + 2 * layer,
            pc0 + palette_w - 1 + 2 * layer,
        )
        ring = _draw_ring(grid, bbox, gap_count)
        if ring is None:
            return None
        ring_cycle, _gap_set = ring
        fill_color = int(palette_colors[gap_count - 1])
        for r, c in ring_cycle:
            if int(grid[r, c]) == GREY:
                expected[r, c] = fill_color

    if not _grey_is_uniquely_dominant(grid):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        built = _build_case()
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate a valid nested-ring palette puzzle")
