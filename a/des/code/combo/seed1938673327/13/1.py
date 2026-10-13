from __future__ import annotations

import random

import numpy as np


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _edge_distance(h: int, w: int, r: int, c: int) -> int:
    return min(r + 1, h - r, c + 1, w - c)


def _frame_cells(h: int, w: int, r: int, c: int, dist: int) -> set[tuple[int, int]]:
    radius = dist - 1
    top = max(0, r - radius)
    bottom = min(h - 1, r + radius)
    left = max(0, c - radius)
    right = min(w - 1, c + radius)

    cells = {(r, c)}
    for cc in range(left, right + 1):
        cells.add((top, cc))
        cells.add((bottom, cc))
    for rr in range(top, bottom + 1):
        cells.add((rr, left))
        cells.add((rr, right))
    return cells


def _paint_if_better(
    out: np.ndarray,
    priority: np.ndarray,
    r: int,
    c: int,
    dist: int,
    color: int,
) -> None:
    if dist > int(priority[r, c]):
        priority[r, c] = dist
        out[r, c] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each isolated marker with a square frame whose size is determined by its nearest edge distance.

    Input structure:
    - A black-like background with isolated single-pixel colored markers.
    - Markers are not adjacent to one another.

    Rule:
    - For every marker, compute D = min distance to the four grid edges, counting edge cells as distance 1.
    - Draw a 1-pixel-thick square frame centered on the marker with side length 2*D-1.
    - Keep the original marker as the colored center pixel.
    - If multiple frames overlap, the one with larger D wins.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    bg = _dominant_color(g)

    markers = [tuple(map(int, pos)) for pos in np.argwhere(g != bg)]
    assert markers, "expected at least one marker"
    for r, c in markers:
        for nr in range(max(0, r - 1), min(h, r + 2)):
            for nc in range(max(0, c - 1), min(w, c + 2)):
                if (nr, nc) != (r, c):
                    assert int(g[nr, nc]) == bg, "markers must be isolated single pixels"

    out = np.full((h, w), bg, dtype=int)
    priority = np.full((h, w), -1, dtype=int)

    for r, c in markers:
        color = int(g[r, c])
        dist = _edge_distance(h, w, r, c)
        radius = dist - 1
        top = max(0, r - radius)
        bottom = min(h - 1, r + radius)
        left = max(0, c - radius)
        right = min(w - 1, c + radius)

        for cc in range(left, right + 1):
            _paint_if_better(out, priority, top, cc, dist, color)
            _paint_if_better(out, priority, bottom, cc, dist, color)
        for rr in range(top, bottom + 1):
            _paint_if_better(out, priority, rr, left, dist, color)
            _paint_if_better(out, priority, rr, right, dist, color)
        _paint_if_better(out, priority, r, c, dist, color)

    return out


# --------------- generator ---------------

GEN_ATTEMPTS = 300
GEN_BG = 0
GEN_PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
GEN_HEIGHT_RANGE = (7, 18)
GEN_WIDTH_RANGE = (7, 18)
GEN_MAX_MARKERS = 5
GEN_MAX_FILL_FRACTION = 0.45
GEN_OVERLAP_PROB = 0.5


def _sample_distances(dmax: int, want_overlap: bool) -> list[int]:
    if want_overlap:
        max_markers = min(GEN_MAX_MARKERS, dmax, len(GEN_PALETTE))
        count = random.randint(2, max_markers)
        chosen = {dmax}
        near_big = [d for d in range(max(1, dmax - 2), dmax) if d not in chosen]
        if near_big and len(chosen) < count:
            chosen.add(random.choice(near_big))
        remaining = [d for d in range(1, dmax + 1) if d not in chosen]
        random.shuffle(remaining)
        for dist in remaining:
            if len(chosen) >= count:
                break
            chosen.add(dist)
    else:
        pool = list(range(1, dmax + 1))
        if dmax >= 5 and random.random() < 0.6:
            pool = pool[:-1]
        max_markers = min(4, len(pool), len(GEN_PALETTE))
        count = random.randint(2, max_markers)
        chosen = set(random.sample(pool, count))

    return sorted(chosen, reverse=True)


def _select_candidate(
    h: int,
    w: int,
    dist: int,
    markers: list[tuple[int, int, int]],
    coverage: np.ndarray,
    want_overlap: bool,
    already_overlapping: bool,
) -> tuple[int, int, set[tuple[int, int]], int] | None:
    centers = [(r, c) for r, c, _dist in markers]
    center_set = set(centers)
    options: list[tuple[int, int, set[tuple[int, int]], int, int]] = []

    for r in range(h):
        for c in range(w):
            if _edge_distance(h, w, r, c) != dist:
                continue
            if any(max(abs(r - r0), abs(c - c0)) <= 1 for r0, c0 in centers):
                continue
            if int(coverage[r, c]) > 0:
                continue

            footprint = _frame_cells(h, w, r, c, dist)
            if footprint & center_set:
                continue

            overlap_cells = sum(1 for rr, cc in footprint if int(coverage[rr, cc]) > 0)
            min_gap = min((abs(r - r0) + abs(c - c0) for r0, c0 in centers), default=h + w)
            options.append((r, c, footprint, overlap_cells, min_gap))

    if not options:
        return None

    random.shuffle(options)
    if not markers:
        options.sort(key=lambda item: item[4])
        r, c, footprint, overlap_cells, _gap = options[0]
        return r, c, footprint, overlap_cells

    if want_overlap and not already_overlapping:
        positive = [opt for opt in options if opt[3] > 0]
        if positive:
            positive.sort(key=lambda item: (item[3], -item[4]))
            r, c, footprint, overlap_cells, _gap = positive[0]
            return r, c, footprint, overlap_cells
        options.sort(key=lambda item: item[4])
        r, c, footprint, overlap_cells, _gap = options[0]
        return r, c, footprint, overlap_cells

    if want_overlap and already_overlapping:
        zero = [opt for opt in options if opt[3] == 0]
        if zero:
            zero.sort(key=lambda item: -item[4])
            r, c, footprint, overlap_cells, _gap = zero[0]
            return r, c, footprint, overlap_cells
        options.sort(key=lambda item: (item[3], -item[4]))
        r, c, footprint, overlap_cells, _gap = options[0]
        return r, c, footprint, overlap_cells

    zero = [opt for opt in options if opt[3] == 0]
    if not zero:
        return None
    zero.sort(key=lambda item: -item[4])
    r, c, footprint, overlap_cells, _gap = zero[0]
    return r, c, footprint, overlap_cells


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GEN_HEIGHT_RANGE)
    w = random.randint(*GEN_WIDTH_RANGE)
    dmax = min((h + 1) // 2, (w + 1) // 2)
    if dmax < 2:
        return None

    want_overlap = random.random() < GEN_OVERLAP_PROB
    distances = _sample_distances(dmax, want_overlap)

    markers: list[tuple[int, int, int]] = []
    coverage = np.zeros((h, w), dtype=int)
    has_overlap = False

    for dist in distances:
        choice = _select_candidate(h, w, dist, markers, coverage, want_overlap, has_overlap)
        if choice is None:
            return None
        r, c, footprint, overlap_cells = choice
        markers.append((r, c, dist))
        for rr, cc in footprint:
            coverage[rr, cc] += 1
        has_overlap = has_overlap or overlap_cells > 0

    overlap_count = int(np.count_nonzero(coverage >= 2))
    if want_overlap != (overlap_count > 0):
        return None

    colors = random.sample(GEN_PALETTE, len(markers))
    grid = np.full((h, w), GEN_BG, dtype=int)
    for (r, c, _dist), color in zip(markers, colors):
        grid[r, c] = color

    out = main(grid)
    if np.array_equal(grid, out):
        return None
    if np.count_nonzero(out != GEN_BG) > GEN_MAX_FILL_FRACTION * h * w:
        return None

    for (r, c, _dist), color in zip(markers, colors):
        if int(out[r, c]) != color:
            return None
        if np.count_nonzero(out == color) == 0:
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
