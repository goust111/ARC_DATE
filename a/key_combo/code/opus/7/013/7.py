from __future__ import annotations

import random
from collections import Counter

import numpy as np


_PALETTE = (1, 2, 3, 4, 5)
_K_CYCLE = (4, 5, 6)
_GEN_CALL_COUNT = 0
_GEN_RETRIES = 300


def _sorted_markers(grid: np.ndarray) -> list[tuple[int, int, int]]:
    coords = np.argwhere(grid != 0)
    markers = [(int(r), int(c), int(grid[r, c])) for r, c in coords]
    markers.sort(key=lambda item: (item[0], item[1]))
    return markers


def _stamp_bounds(r: int, c: int, k: int) -> tuple[int, int, int, int]:
    r0 = int(r - k // 2)
    c0 = int(c - k // 2)
    return r0, c0, r0 + k - 1, c0 + k - 1


def _solve_with_metadata(grid: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    h, w = g.shape
    k = int(np.unique(g).size)
    markers = _sorted_markers(g)
    assert markers, "expected at least one non-background marker"

    bounds: list[tuple[int, int, int, int]] = []
    for r, c, _color in markers:
        b = _stamp_bounds(r, c, k)
        r0, c0, r1, c1 = b
        assert 0 <= r0 <= r1 < h and 0 <= c0 <= c1 < w, "stamp must stay in bounds"
        bounds.append(b)

    out = np.zeros_like(g)
    overlap_cells: list[tuple[int, int]] = []
    majority_cells: list[tuple[int, int]] = []
    tie_cells: list[tuple[int, int]] = []
    coverage_map: dict[tuple[int, int], tuple[int, ...]] = {}

    for r in range(h):
        for c in range(w):
            covering: list[int] = []
            for idx, (r0, c0, r1, c1) in enumerate(bounds):
                if r0 <= r <= r1 and c0 <= c <= c1:
                    covering.append(idx)
            if not covering:
                continue

            if len(covering) >= 2:
                overlap_cells.append((r, c))
            coverage_map[(r, c)] = tuple(covering)

            votes: dict[int, int] = {}
            first_source: dict[int, tuple[int, int]] = {}
            for idx in covering:
                sr, sc, color = markers[idx]
                votes[color] = votes.get(color, 0) + 1
                source = (sr, sc)
                if color not in first_source or source < first_source[color]:
                    first_source[color] = source

            max_votes = max(votes.values())
            top_colors = [color for color, count in votes.items() if count == max_votes]
            if len(top_colors) == 1 and max_votes >= 2:
                majority_cells.append((r, c))
            if len(top_colors) >= 2:
                tie_cells.append((r, c))
            winner = min(top_colors, key=lambda color: first_source[color])
            out[r, c] = int(winner)

    return out, {
        "k": k,
        "markers": markers,
        "bounds": bounds,
        "overlap_cells": overlap_cells,
        "majority_cells": majority_cells,
        "tie_cells": tie_cells,
        "coverage_map": coverage_map,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each marker into a color-count-sized square stamp and resolve overlaps by per-cell color voting.

    Input structure:
    - The background is black (0), and every non-black cell is a single-pixel marker.
    - The number of distinct colors present in the whole grid, including black, is the
      hidden size code K for every stamp.
    - Some markers are close enough that their K x K stamps overlap, while others stay separate.

    Rule:
    - Compute K as the number of distinct colors present in the input grid, including black.
    - Replace every non-background marker by a same-color K x K square centered on that marker;
      when K is even, use the problem's left/up-biased center convention, which is equivalent
      to taking the marker at local index (K//2, K//2) inside the square.
    - For each output cell covered by one or more stamps, count the covering stamp colors and
      write the most frequent color. If multiple colors tie for the highest count, choose the
      color whose covering source marker has the smallest row, then smallest column.
    - Cells not covered by any stamp remain black.

    Key insight:
    - The marker colors do not only paint the result: their global color diversity sets K, and
      that single count changes every stamp footprint and therefore the entire overlap-voting pattern.
    """
    out, _meta = _solve_with_metadata(grid)
    return out


def _base_layout(k: int) -> list[tuple[int, int]]:
    rel = [(0, 0), (1, 1), (2, 0), (k + 1, 0)]
    for i in range(k - 4):
        rel.append((k + 3 + i, k + 2 + i * (k + 1)))
    return rel


def _transform_points(
    points: list[tuple[int, int]],
    transpose: bool,
    flip_r: bool,
    flip_c: bool,
) -> list[tuple[int, int]]:
    pts = points[:]
    h = max(r for r, _ in pts) + 1
    w = max(c for _, c in pts) + 1
    if transpose:
        pts = [(c, r) for r, c in pts]
        h, w = w, h
    if flip_r:
        pts = [(h - 1 - r, c) for r, c in pts]
    if flip_c:
        pts = [(r, w - 1 - c) for r, c in pts]
    return pts


def _choose_canvas(k: int, rel_points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rel_h = max(r for r, _ in rel_points) + 1
    rel_w = max(c for _, c in rel_points) + 1

    min_h = max(18, rel_h + k + 4)
    min_w = max(18, rel_w + k + 4)
    max_h = min(28, min_h + 6)
    max_w = min(28, min_w + 6)

    h = random.randint(min_h, max_h)
    w = random.randint(min_w, max_w)

    low_r = k // 2 + 1
    low_c = k // 2 + 1
    high_r = h - ((k + 1) // 2) - 1
    high_c = w - ((k + 1) // 2) - 1

    anchor_r = random.randint(low_r, high_r - rel_h + 1)
    anchor_c = random.randint(low_c, high_c - rel_w + 1)
    return h, w, anchor_r, anchor_c


def _color_roles(k: int) -> list[int]:
    distinct = random.sample(list(_PALETTE), k - 1)
    duplicate = random.choice(distinct)
    others = [color for color in distinct if color != duplicate]
    random.shuffle(others)
    ordered = [duplicate] + others
    return [ordered[0], ordered[0], *ordered[1:]]


def _pairwise_overlap(bounds: list[tuple[int, int, int, int]], i: int, j: int) -> bool:
    a = bounds[i]
    b = bounds[j]
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _semantic_checks(grid: np.ndarray) -> bool:
    markers = _sorted_markers(grid)
    if not markers:
        return False

    k = int(np.unique(grid).size)
    if len(markers) != k or not 4 <= k <= 6:
        return False

    counts = Counter(color for _r, _c, color in markers)
    if len(counts) != k - 1:
        return False
    if sorted(counts.values()) != [1] * (k - 2) + [2]:
        return False

    out, meta = _solve_with_metadata(grid)
    if not np.any(out != grid):
        return False
    if not meta["overlap_cells"] or not meta["majority_cells"] or not meta["tie_cells"]:
        return False
    if int(np.count_nonzero(out == 0)) == 0:
        return False

    bounds = meta["bounds"]
    if not _pairwise_overlap(bounds, 0, 1):
        return False
    if not _pairwise_overlap(bounds, 0, 2):
        return False
    if not _pairwise_overlap(bounds, 1, 2):
        return False
    if not _pairwise_overlap(bounds, 2, 3):
        return False
    if _pairwise_overlap(bounds, 0, 3) or _pairwise_overlap(bounds, 1, 3):
        return False

    return True


def _gen_attempt(k: int) -> np.ndarray | None:
    rel = _base_layout(k)
    rel = _transform_points(
        rel,
        transpose=bool(random.getrandbits(1)),
        flip_r=bool(random.getrandbits(1)),
        flip_c=bool(random.getrandbits(1)),
    )
    h, w, anchor_r, anchor_c = _choose_canvas(k, rel)

    centers = [(anchor_r + r, anchor_c + c) for r, c in rel]
    colors = _color_roles(k)
    tail = colors[2:]
    random.shuffle(tail)
    colors = colors[:2] + tail

    grid = np.zeros((h, w), dtype=int)
    for (r, c), color in zip(centers, colors):
        if grid[r, c] != 0:
            return None
        grid[r, c] = int(color)

    if not _semantic_checks(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT

    for _ in range(_GEN_RETRIES):
        k = _K_CYCLE[_GEN_CALL_COUNT % len(_K_CYCLE)]
        _GEN_CALL_COUNT += 1
        grid = _gen_attempt(k)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
