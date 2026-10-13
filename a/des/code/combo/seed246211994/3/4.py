from __future__ import annotations

from collections import deque
import random

import numpy as np


GREY = 5
GREEN = 3

# --------------- shared analysis ---------------


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _analyze_regions(grid: np.ndarray) -> tuple[int, np.ndarray, list[set[int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    h, w = g.shape
    bg = _dominant_color(g)
    bg_mask = g == bg
    labels = np.full((h, w), -1, dtype=int)
    region_count = 0

    for r in range(h):
        for c in range(w):
            if not bg_mask[r, c] or labels[r, c] != -1:
                continue
            q = deque([(r, c)])
            labels[r, c] = region_count
            while q:
                rr, cc = q.popleft()
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bg_mask[nr, nc] and labels[nr, nc] == -1:
                        labels[nr, nc] = region_count
                        q.append((nr, nc))
            region_count += 1

    assert region_count > 0, "expected at least one background region"

    adjacency = [set() for _ in range(region_count)]
    wall_mask = ~bg_mask

    # A region boundary is detectable as background-wall-background triples.
    for r in range(h):
        for c in range(w - 2):
            if bg_mask[r, c] and wall_mask[r, c + 1] and bg_mask[r, c + 2]:
                a = int(labels[r, c])
                b = int(labels[r, c + 2])
                if a != b:
                    adjacency[a].add(b)
                    adjacency[b].add(a)

    for r in range(h - 2):
        for c in range(w):
            if bg_mask[r, c] and wall_mask[r + 1, c] and bg_mask[r + 2, c]:
                a = int(labels[r, c])
                b = int(labels[r + 2, c])
                if a != b:
                    adjacency[a].add(b)
                    adjacency[b].add(a)

    return bg, labels, adjacency


def main(grid: np.ndarray) -> np.ndarray:
    """
    Identify wall-separated background regions and 2-color them grey/green.

    The region containing the bottom-left corner is grey. Crossing any wall
    flips the color, so adjacent regions alternate between grey and green.
    """
    g = np.asarray(grid, dtype=int)
    bg, labels, adjacency = _analyze_regions(g)
    h, _w = g.shape

    start_region = int(labels[h - 1, 0])
    assert start_region >= 0, "bottom-left cell must belong to the start region"

    parity = np.full(len(adjacency), -1, dtype=int)
    parity[start_region] = 0
    q = deque([start_region])

    while q:
        region = q.popleft()
        next_parity = 1 - int(parity[region])
        for nbr in adjacency[region]:
            if parity[nbr] == -1:
                parity[nbr] = next_parity
                q.append(nbr)
            else:
                assert int(parity[nbr]) == next_parity, "region adjacency must be bipartite"

    assert np.all(parity >= 0), "every region must be reachable from the bottom-left region"

    out = g.copy()
    for region_id, region_parity in enumerate(parity):
        out[labels == region_id] = GREY if int(region_parity) == 0 else GREEN
    return out


# --------------- generator ---------------

INPUT_COLOR_POOL = tuple(c for c in range(10) if c not in {GREY, GREEN})
GRID_HEIGHT_RANGE = (10, 24)
GRID_WIDTH_RANGE = (10, 24)
MAX_GENERATION_ATTEMPTS = 300

MIN_CHILD_SPAN = 2
MAX_SPLITS = 6

Rect = tuple[int, int, int, int]


def _rect_area(rect: Rect) -> int:
    r0, c0, r1, c1 = rect
    return int((r1 - r0 + 1) * (c1 - c0 + 1))


def _split_options(rect: Rect) -> list[str]:
    r0, c0, r1, c1 = rect
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    options: list[str] = []
    if width >= 2 * MIN_CHILD_SPAN + 1:
        options.append("v")
    if height >= 2 * MIN_CHILD_SPAN + 1:
        options.append("h")
    return options


def _apply_split(grid: np.ndarray, rect: Rect, orient: str, wall_color: int) -> tuple[Rect, Rect]:
    r0, c0, r1, c1 = rect
    if orient == "v":
        wall_c = random.randint(c0 + MIN_CHILD_SPAN, c1 - MIN_CHILD_SPAN)
        grid[r0 : r1 + 1, wall_c] = wall_color
        left = (r0, c0, r1, wall_c - 1)
        right = (r0, wall_c + 1, r1, c1)
        return left, right

    wall_r = random.randint(r0 + MIN_CHILD_SPAN, r1 - MIN_CHILD_SPAN)
    grid[wall_r, c0 : c1 + 1] = wall_color
    top = (r0, c0, wall_r - 1, c1)
    bottom = (wall_r + 1, c0, r1, c1)
    return top, bottom


def _is_connected_graph(adjacency: list[set[int]]) -> bool:
    seen = {0}
    q = deque([0])
    while q:
        node = q.popleft()
        for nbr in adjacency[node]:
            if nbr not in seen:
                seen.add(nbr)
                q.append(nbr)
    return len(seen) == len(adjacency)


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    bg = random.choice(INPUT_COLOR_POOL)
    wall = random.choice([c for c in INPUT_COLOR_POOL if c != bg])

    grid = np.full((height, width), bg, dtype=int)
    regions: list[Rect] = [(0, 0, height - 1, width - 1)]

    max_target = min(MAX_SPLITS, max(1, (height * width) // 36))
    target_choices = [1] + [k for k in range(2, max_target + 1) for _ in range(2)]
    target_splits = random.choice(target_choices)
    used_orientations: set[str] = set()
    split_count = 0

    for _ in range(target_splits):
        candidates: list[tuple[int, str]] = []
        weights: list[int] = []
        for idx, rect in enumerate(regions):
            for orient in _split_options(rect):
                candidates.append((idx, orient))
                weights.append(_rect_area(rect))
        if not candidates:
            break

        chosen_idx, orient = random.choices(candidates, weights=weights, k=1)[0]
        rect = regions.pop(chosen_idx)
        child_a, child_b = _apply_split(grid, rect, orient, wall)
        regions.extend([child_a, child_b])
        used_orientations.add(orient)
        split_count += 1

    if split_count == 0:
        return None
    if split_count >= 3 and len(used_orientations) == 1 and random.random() < 0.7:
        return None

    bg_pixels = int(np.count_nonzero(grid == bg))
    wall_pixels = int(grid.size - bg_pixels)
    if bg_pixels <= wall_pixels:
        return None

    try:
        analyzed_bg, labels, adjacency = _analyze_regions(grid)
        assert analyzed_bg == bg
        assert int(labels[-1, 0]) >= 0
        assert len(adjacency) == split_count + 1
        assert _is_connected_graph(adjacency)
        out = main(grid)
    except AssertionError:
        return None

    if not np.any(out == GREY) or not np.any(out == GREEN):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
