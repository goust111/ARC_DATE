from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
RED = 2
YELLOW = 4
ORANGE = 7
MARKER_PALETTE = (1, 3, 5, 6, 8, 9)
GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _parse_markers(grid: np.ndarray) -> list[tuple[int, int, int]]:
    coords = np.argwhere(grid != BG)
    markers: list[tuple[int, int, int]] = []
    seen_rows: set[int] = set()
    seen_cols: set[int] = set()
    seen_colors: set[int] = set()
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        color = int(grid[rr, cc])
        assert color != RED, "input markers must not use red"
        assert rr not in seen_rows, "markers must occupy distinct rows"
        assert cc not in seen_cols, "markers must occupy distinct columns"
        assert color not in seen_colors, "marker colors must be distinct"
        markers.append((rr, cc, color))
        seen_rows.add(rr)
        seen_cols.add(cc)
        seen_colors.add(color)
    assert markers, "expected at least one marker pixel"
    return markers


def _build_cross_grid(
    grid: np.ndarray,
    markers: list[tuple[int, int, int]],
) -> tuple[np.ndarray, list[list[set[int]]]]:
    h, w = grid.shape
    out = np.zeros((h, w), dtype=int)
    ownership = [[set() for _ in range(w)] for _ in range(h)]
    source_color: dict[int, int] = {}

    for source_id, (r, c, color) in enumerate(markers):
        out[r, c] = color
        ownership[r][c].add(source_id)
        source_color[source_id] = color

    for source_id in sorted(range(len(markers)), key=lambda idx: markers[idx][0]):
        r, c, color = markers[source_id]
        for cc in range(w):
            if cc == c:
                continue
            out[r, cc] = color
            ownership[r][cc].add(source_id)

    for source_id in sorted(range(len(markers)), key=lambda idx: markers[idx][1]):
        r, c, color = markers[source_id]
        for rr in range(h):
            if rr == r:
                continue
            out[rr, c] = color
            ownership[rr][c].add(source_id)

    for r in range(h):
        for c in range(w):
            owners = ownership[r][c]
            if not owners:
                continue
            if len(owners) >= 2:
                out[r, c] = RED
            else:
                source_id = next(iter(owners))
                out[r, c] = source_color[source_id]

    for _source_id, (r, c, color) in enumerate(markers):
        out[r, c] = color

    return out, ownership


def _closed_regions(
    wall_grid: np.ndarray,
    ownership: list[list[set[int]]],
) -> list[dict]:
    h, w = wall_grid.shape
    visited = np.zeros((h, w), dtype=bool)
    regions: list[dict] = []

    for r in range(h):
        for c in range(w):
            if visited[r, c] or int(wall_grid[r, c]) != BG:
                continue
            queue = deque([(r, c)])
            visited[r, c] = True
            cells: list[tuple[int, int]] = []
            touches_edge = False
            while queue:
                rr, cc = queue.popleft()
                cells.append((rr, cc))
                if rr == 0 or cc == 0 or rr == h - 1 or cc == w - 1:
                    touches_edge = True
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(wall_grid[nr, nc]) == BG:
                        visited[nr, nc] = True
                        queue.append((nr, nc))

            if touches_edge:
                continue

            boundary_sources: set[int] = set()
            for rr, cc in cells:
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and ownership[nr][nc]:
                        boundary_sources |= ownership[nr][nc]

            regions.append(
                {
                    "cells": cells,
                    "boundary_sources": boundary_sources,
                    "source_count": len(boundary_sources),
                }
            )

    return regions


def _solve(grid: np.ndarray, *, return_regions: bool = False):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    markers = _parse_markers(g)
    wall_grid, ownership = _build_cross_grid(g, markers)
    out = wall_grid.copy()
    regions = _closed_regions(wall_grid, ownership)

    for region in regions:
        count = int(region["source_count"])
        if count == 2:
            fill = YELLOW
        elif count >= 3:
            fill = ORANGE
        else:
            continue
        for r, c in region["cells"]:
            out[r, c] = fill

    if return_regions:
        return out, regions
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend isolated markers into full cross-lines, then flood-fill each enclosed black region by how many distinct marker sources bound it.

    Input structure:
    - A black background contains 3 to 5 isolated non-red marker pixels of distinct colors.
    - No two markers share a row or a column, so each marker defines one unique horizontal ray and one unique vertical ray.
    - The input contains only those marker pixels; all walls and enclosed regions are implicit.

    Rule:
    - For every marker, paint its entire row and entire column as that marker's cross, keeping the marker cell itself unchanged.
    - After drawing all row crosses and all column crosses, recolor every intersection between different markers' row/column sources to red.
    - Treat every non-black pixel in that completed cross-line grid as a wall, and flood-fill black cells with 4-connectivity.
    - Any black component that does not touch the grid boundary is enclosed: fill it yellow if exactly 2 distinct marker sources appear on its boundary walls, otherwise fill it orange if 3 or more distinct marker sources appear.

    Key insight:
    - The cross-lines are an intermediate wall construction rather than the final target, and each filled region's color depends on the combined set of marker sources around its boundary, not on any single nearby marker.
    """
    return _solve(grid)


def _sample_axis_positions(rng: np.random.Generator, count: int) -> tuple[list[int], int]:
    for _ in range(100):
        start_margin = int(rng.integers(1, 5))
        end_margin = int(rng.integers(1, 5))
        gaps = [int(x) for x in rng.integers(3, 6, size=count - 1)]
        positions = [start_margin]
        for gap in gaps:
            positions.append(positions[-1] + gap)
        size = positions[-1] + end_margin + 1
        if 10 <= size <= 30:
            return positions, size
    raise RuntimeError("failed to sample axis positions")


def _region_source_counts_from_perm(perm: tuple[int, ...]) -> list[int]:
    count = len(perm)
    inverse = [0] * count
    for row_idx, col_idx in enumerate(perm):
        inverse[col_idx] = row_idx
    counts: list[int] = []
    for row_idx in range(count - 1):
        for col_idx in range(count - 1):
            counts.append(len({row_idx, row_idx + 1, inverse[col_idx], inverse[col_idx + 1]}))
    return counts


def _sample_perm(rng: np.random.Generator, count: int) -> tuple[int, ...]:
    for _ in range(200):
        perm = tuple(int(x) for x in rng.permutation(count))
        counts = _region_source_counts_from_perm(perm)
        if 2 in counts and any(value >= 3 for value in counts):
            return perm
    raise RuntimeError("failed to sample a valid permutation")


def _validate_generated_case(grid: np.ndarray) -> None:
    g = np.asarray(grid, dtype=int)
    markers = _parse_markers(g)
    assert 3 <= len(markers) <= 5
    out, regions = _solve(g, return_regions=True)
    counts = [int(region["source_count"]) for region in regions]
    assert any(count == 2 for count in counts), "generator must include a 2-source enclosed region"
    assert any(count >= 3 for count in counts), "generator must include a 3+-source enclosed region"
    assert np.any(out == YELLOW), "yellow fill must appear"
    assert np.any(out == ORANGE), "orange fill must appear"

    for region in regions:
        cells = region["cells"]
        expected = YELLOW if region["source_count"] == 2 else ORANGE
        colors = {int(out[r, c]) for r, c in cells}
        assert colors == {expected}

    wall_grid, _ownership = _build_cross_grid(g, markers)
    assert np.any(wall_grid == RED), "cross intersections should be present and visible"
    assert np.array_equal(out[g != BG], wall_grid[g != BG]), "marker pixels must stay unchanged"


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        rng = np.random.default_rng()
        marker_count = int(rng.integers(3, 6))
        rows, height = _sample_axis_positions(rng, marker_count)
        cols, width = _sample_axis_positions(rng, marker_count)
        perm = _sample_perm(rng, marker_count)
        colors = list(rng.choice(MARKER_PALETTE, size=marker_count, replace=False))
        random.shuffle(colors)

        grid = np.zeros((height, width), dtype=int)
        for row_rank, col_rank in enumerate(perm):
            grid[rows[row_rank], cols[col_rank]] = int(colors[row_rank])

        try:
            _validate_generated_case(grid)
        except Exception:
            continue
        return grid
    raise RuntimeError("failed to generate a valid input for puzzle 1")
