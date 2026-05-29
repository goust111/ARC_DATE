# Variant of 28a6681f
# Dimension: D22 — 水源与盆地的空间关系 (meta)
# Change: 水源位于盆地上方或侧面，通过guide或直接的路径连接到盆地区域... → 多个水源对应多个盆地（映射关系）

from __future__ import annotations

from collections import defaultdict, deque
import random

import numpy as np


# Design analysis for D22/1
# -------------------------
# This variant turns the task from a single local water simulation into a visible
# source-to-basin mapping problem.  Each upper source has blue water resting on a
# colored shelf and below a same-colored tag.  The bottom half contains one U
# basin of each tag color.  Source positions are deliberately permuted relative
# to basin positions, so a solver must match by visible color equality and then
# place that source's water into the matching basin.
#
# ARC suitability:
# - The mapping cue is direct color equality, not numeric color ordering.
# - Only 2-3 source/basin colors appear in a sample.
# - All basins are clean U shapes; fill cells are visually obvious and filled
#   bottom-up, so the transformation is easy to check from 2-4 examples.
# - The generator validates every sampled grid with main() and forces at least
#   one non-vertical mapping, preventing degeneration into the original layout.


BG = 0
WATER = 1


def _components(points: set[tuple[int, int]]) -> list[set[tuple[int, int]]]:
    remaining = set(points)
    comps: list[set[tuple[int, int]]] = []
    while remaining:
        start = remaining.pop()
        comp = {start}
        q: deque[tuple[int, int]] = deque([start])
        while q:
            r, c = q.popleft()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nxt = (int(r + dr), int(c + dc))
                if nxt in remaining:
                    remaining.remove(nxt)
                    comp.add(nxt)
                    q.append(nxt)
        comps.append(comp)
    return comps


def _source_counts(grid: np.ndarray) -> dict[int, int]:
    H, _W = grid.shape
    counts: dict[int, int] = defaultdict(int)
    for r, c in np.argwhere(grid == WATER):
        rr = int(r)
        cc = int(c)
        if rr >= H // 2:
            continue
        support = int(grid[rr + 1, cc]) if rr + 1 < H else BG
        tag = int(grid[rr - 1, cc]) if rr - 1 >= 0 else BG
        candidates = [v for v in (support, tag) if v not in (BG, WATER)]
        assert candidates, "upper water must have a visible colored source tag"
        assert len(set(candidates)) == 1, "source tag and support must agree"
        counts[int(candidates[0])] += 1
    assert counts, "at least one upper source is required"
    return dict(counts)


def _basin_fill_cells(grid: np.ndarray, color: int) -> list[tuple[int, int]]:
    H, _W = grid.shape
    pts = {(int(r), int(c)) for r, c in np.argwhere(grid == int(color))}
    lower_comps = [comp for comp in _components(pts) if max(r for r, _c in comp) == H - 1]
    assert len(lower_comps) == 1, f"expected one bottom basin for color {color}"
    comp = lower_comps[0]
    rows = [r for r, _c in comp]
    cols = [c for _r, c in comp]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    assert r1 == H - 1 and c1 - c0 >= 2, "basin must be a bottom-anchored U"

    cells: list[tuple[int, int]] = []
    for r in range(r1 - 1, r0 - 1, -1):
        for c in range(c0 + 1, c1):
            if int(grid[r, c]) in (BG, WATER):
                cells.append((int(r), int(c)))
    assert cells, f"basin for color {color} has no interior"
    return cells


def main(grid: np.ndarray, check_input: bool = True, check_output: bool = True) -> np.ndarray:
    """
    Route several visible water sources into their matching colored basins.

    Input structure:
    - bg = 0. Water pixels = color 1.
    - The top half contains 2-3 separate blue source runs, each with a colored tag and support shelf.
    - The bottom half contains one bottom-anchored U basin for each source tag color.

    Rule:
    - Count the blue pixels in each tagged upper source.
    - Match that source to the bottom U basin with the same visible color.
    - Remove the upper source water and fill the matched basin bottom-up with the same number of blue pixels.

    Key insight: Source position no longer determines the destination; the colored tag gives a 1:1 source-to-basin mapping.
    """
    g = np.array(grid, dtype=int)
    assert g.ndim == 2
    out = g.copy()
    counts = _source_counts(g)

    if check_input:
        assert len(counts) in (2, 3), "variant expects 2 or 3 mapped sources"
        for color, count in counts.items():
            assert count >= 2, "each source should contain a visible run of at least two water pixels"
            assert len(_basin_fill_cells(g, int(color))) >= count, "matched basin capacity is too small"

    for r, c in np.argwhere(g == WATER):
        if int(r) < g.shape[0] // 2:
            out[int(r), int(c)] = BG

    for color, count in counts.items():
        cells = _basin_fill_cells(g, int(color))
        assert len(cells) >= count, f"not enough basin cells for color {color}"
        for r, c in cells[: int(count)]:
            out[int(r), int(c)] = WATER

    if check_output:
        assert int(np.sum(g == WATER)) == int(np.sum(out == WATER)), "water count must be conserved"
        assert not np.any(out[: g.shape[0] // 2] == WATER), "all source water must move into basins"
        for color, count in counts.items():
            filled = sum(1 for r, c in _basin_fill_cells(g, int(color)) if int(out[r, c]) == WATER)
            assert filled == count, f"basin {color} received the wrong amount of water"

    return out


def _draw_basin(grid: np.ndarray, color: int, c0: int, c1: int, top: int = 6) -> None:
    H, _W = grid.shape
    grid[top:H, c0] = int(color)
    grid[top:H, c1] = int(color)
    grid[H - 1, c0 : c1 + 1] = int(color)


def _draw_source(grid: np.ndarray, color: int, start: int, count: int) -> None:
    water_row = 1
    tag_row = 0
    support_row = 2
    grid[tag_row, start : start + count] = int(color)
    grid[water_row, start : start + count] = WATER
    grid[support_row, start : start + count] = int(color)


def _mapping_is_nontrivial(source_colors: list[int], basin_colors: list[int]) -> bool:
    return any(src != dst for src, dst in zip(source_colors, basin_colors))


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(200):
        n = rng.choice([2, 3])
        colors = rng.sample([2, 3, 4, 5, 6, 7, 8, 9], n)
        g = np.zeros((10, 10), dtype=int)

        if n == 2:
            basins = [(0, 3), (6, 9)]
            source_starts = [1, 6]
            counts_by_slot = [rng.randint(2, 3), rng.randint(2, 3)]
        else:
            basins = [(0, 2), (4, 6), (7, 9)]
            source_starts = [0, 4, 7]
            counts_by_slot = [rng.randint(2, 3), rng.randint(2, 3), rng.randint(2, 3)]

        for color, (c0, c1) in zip(colors, basins):
            _draw_basin(g, int(color), int(c0), int(c1))

        source_colors = list(colors)
        rng.shuffle(source_colors)
        if not _mapping_is_nontrivial(source_colors, colors):
            source_colors = source_colors[1:] + source_colors[:1]

        for color, start, count in zip(source_colors, source_starts, counts_by_slot):
            _draw_source(g, int(color), int(start), int(count))

        try:
            out = main(g)
        except AssertionError:
            continue

        source_counts = _source_counts(g)
        moved_to_lower = int(np.sum(out[5:] == WATER))
        if (
            len(source_counts) == n
            and moved_to_lower == int(np.sum(g == WATER))
            and _mapping_is_nontrivial(source_colors, colors)
        ):
            return g

    raise RuntimeError("failed to generate D22/1 mapped source sample")
