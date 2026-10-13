from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


def _full_width_green_bar_row(grid: np.ndarray) -> int:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == GREEN)]
    assert len(rows) == 1, "expected exactly one full-width green bar"
    return int(rows[0])


def _bbox_of(coords: list[Coord]) -> BBox:
    rs = [r for r, _c in coords]
    cs = [c for _r, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _color_components(grid: np.ndarray, color: int) -> list[list[Coord]]:
    h, w = map(int, grid.shape)
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[Coord]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[Coord] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(comp)
    return components


def _is_hollow_rectangle(grid: np.ndarray, coords: list[Coord]) -> bool:
    r0, c0, r1, c1 = _bbox_of(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if not (3 <= h <= 5 and 3 <= w <= 5):
        return False
    expected = set()
    for c in range(c0, c1 + 1):
        expected.add((r0, c))
        expected.add((r1, c))
    for r in range(r0, r1 + 1):
        expected.add((r, c0))
        expected.add((r, c1))
    if set(coords) != expected:
        return False
    return bool(np.all(grid[r0 + 1 : r1, c0 + 1 : c1] == BLACK))


def _draw_hollow_rectangle(grid: np.ndarray, bbox: BBox) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = BLUE
    grid[r1, c0 : c1 + 1] = BLUE
    grid[r0 : r1 + 1, c0] = BLUE
    grid[r0 : r1 + 1, c1] = BLUE


def _rectangle_cells(bbox: BBox) -> set[Coord]:
    r0, c0, r1, c1 = bbox
    cells = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _trace_diagonal(base: np.ndarray, start: Coord, dc: int) -> tuple[list[Coord], str]:
    h, w = map(int, base.shape)
    r, c = start
    rr = r - 1
    cc = c + dc
    cells: list[Coord] = []
    while 0 <= rr < h and 0 <= cc < w:
        if int(base[rr, cc]) != BLACK:
            return cells, "block"
        cells.append((rr, cc))
        rr -= 1
        cc += dc
    return cells, "edge"


def _simulate(
    grid: np.ndarray,
    *,
    splash_all_moved: bool = False,
    ignore_blue_blockers: bool = False,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    bar_row = _full_width_green_bar_row(g)

    out = g.copy()
    source_coords = [tuple(map(int, rc)) for rc in np.argwhere(g == RED)]
    out[g == RED] = BLACK

    moved: list[Coord] = []
    for r, c in sorted(source_coords):
        nr = r + 1
        assert 0 <= nr < h, "red pixel cannot shift out of bounds"
        assert int(out[nr, c]) == BLACK, "shift target must be empty"
        out[nr, c] = RED
        moved.append((nr, c))

    base = out.copy()
    if ignore_blue_blockers:
        base[base == BLUE] = BLACK

    traces: list[dict[str, object]] = []
    for nr, c in moved:
        should_splash = splash_all_moved or nr == bar_row - 1
        if not should_splash:
            continue
        for dc in (-1, 1):
            cells, stop_reason = _trace_diagonal(base, (nr, c), dc)
            for rr, cc in cells:
                out[rr, cc] = YELLOW
            traces.append(
                {
                    "start": (nr, c),
                    "dc": dc,
                    "cells": tuple(cells),
                    "stop_reason": stop_reason,
                }
            )
    return out, traces


def main(grid: np.ndarray) -> np.ndarray:
    """
    Shift red pixels down one row, then emit yellow diagonal splashes that are stopped by blue hollow rectangles.

    Input structure
    - The grid has black background, exactly one full-width green horizontal trigger bar, several red single pixels above it, and one or more blue hollow rectangles above the bar.
    - Red pixels are isolated moving markers; blue pixels form rectangle borders with black interiors.
    - Yellow is absent from the input and is reserved for the splash rays.

    Rule
    - Erase every red pixel from its original location and redraw it one row lower in the same column.
    - Any moved red pixel that lands exactly one row above the green bar emits two yellow rays: one up-left and one up-right.
    - Each ray advances one cell at a time diagonally upward and stops before the first non-black cell or the grid boundary, so blue rectangle borders remain unchanged.
    - The green bar, the blue hollow rectangles, and the moved red pixels stay visible; only previously black cells can become yellow.

    Key insight
    - The transformation has three dependent stages: shifting determines which red pixels become trigger-adjacent, trigger adjacency determines which pixels splash, and the fixed blue frames limit the visible splash length.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert YELLOW not in set(map(int, np.unique(g))), "yellow must be absent from the input"

    bar_row = _full_width_green_bar_row(g)
    assert 0 < bar_row < g.shape[0], "bar must leave room above it"

    blue_components = _color_components(g, BLUE)
    assert 1 <= len(blue_components) <= 2, "expected one or two blue hollow rectangles"
    assert all(_is_hollow_rectangle(g, comp) for comp in blue_components), "blue objects must be hollow rectangles"

    return _simulate(g)[0]


def _analyze_generated_input(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30
    assert YELLOW not in set(map(int, np.unique(g)))

    h, w = map(int, g.shape)
    bar_row = _full_width_green_bar_row(g)
    assert h >= 14 and w >= 13
    assert h <= 18 and w <= 18
    assert h // 2 + 1 <= bar_row <= h - 4

    red_sources = [tuple(map(int, rc)) for rc in np.argwhere(g == RED)]
    assert 2 <= len(red_sources) <= 4
    assert len({c for _r, c in red_sources}) == len(red_sources)
    assert all(0 <= r < bar_row - 1 for r, _c in red_sources)

    trigger_sources = [(r, c) for r, c in red_sources if r == bar_row - 2]
    nontrigger_sources = [(r, c) for r, c in red_sources if r != bar_row - 2]
    assert 1 <= len(trigger_sources) <= 2
    assert len(nontrigger_sources) >= 1

    blue_components = _color_components(g, BLUE)
    assert 1 <= len(blue_components) <= 2
    blue_bboxes = []
    for comp in blue_components:
        assert _is_hollow_rectangle(g, comp)
        bbox = _bbox_of(comp)
        r0, c0, r1, c1 = bbox
        assert r1 <= bar_row - 3
        blue_bboxes.append(bbox)
        for rr, cc in red_sources:
            assert not (r0 - 1 <= rr <= r1 + 1 and c0 - 1 <= cc <= c1 + 1), "rectangle too close to red source"

    actual, traces = _simulate(g)
    assert np.any(actual == YELLOW), "expected at least one visible splash"
    yellow_cells = {tuple(map(int, rc)) for rc in np.argwhere(actual == YELLOW)}

    moved_reds = [tuple(map(int, rc)) for rc in np.argwhere(actual == RED)]
    assert len(moved_reds) == len(red_sources)
    assert all(r + 1 < h for r, _c in red_sources)
    assert sorted((r + 1, c) for r, c in red_sources) == sorted(moved_reds)

    triggered_starts = {trace["start"] for trace in traces}
    assert triggered_starts == {(bar_row - 1, c) for _r, c in trigger_sources}

    blocked_traces = [trace for trace in traces if trace["stop_reason"] == "block"]
    edge_traces = [trace for trace in traces if trace["stop_reason"] == "edge"]
    assert blocked_traces, "at least one splash ray must be stopped by a blue frame"
    assert edge_traces, "at least one splash ray should still reach the boundary for contrast"
    assert all(len(trace["cells"]) >= 1 for trace in blocked_traces), "blocked ray must be visibly truncated"

    actual_base = _simulate(g)[0].copy()
    assert all(int(actual_base[r, c]) == YELLOW for r, c in yellow_cells)

    no_blockers, _ = _simulate(g, ignore_blue_blockers=True)
    assert not np.array_equal(no_blockers, actual), "blue blockers must affect the result"

    all_splash, _ = _simulate(g, splash_all_moved=True)
    assert not np.array_equal(all_splash, actual), "trigger adjacency must affect the result"

    for r, c in yellow_cells:
        assert int(g[r, c]) == BLACK, "yellow may only appear on original background"

    return {
        "bar_row": bar_row,
        "red_sources": tuple(sorted(red_sources)),
        "trigger_sources": tuple(sorted(trigger_sources)),
        "blue_bboxes": tuple(sorted(blue_bboxes)),
        "yellow_count": len(yellow_cells),
        "blocked_trace_count": len(blocked_traces),
        "edge_trace_count": len(edge_traces),
    }


def _point_near_bbox(point: Coord, bbox: BBox, margin: int = 1) -> bool:
    r, c = point
    r0, c0, r1, c1 = bbox
    return r0 - margin <= r <= r1 + margin and c0 - margin <= c <= c1 + margin


def _bbox_overlaps(a: BBox, b: BBox, margin: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + margin < br0 or br1 + margin < ar0 or ac1 + margin < bc0 or bc1 + margin < ac0)


def _all_rectangle_candidates(
    shape: tuple[int, int],
    bar_row: int,
    red_sources: list[Coord],
    moved_reds: list[Coord],
    existing: list[BBox],
) -> list[BBox]:
    h, w = map(int, shape)
    protected = red_sources + moved_reds
    candidates: list[BBox] = []
    for rect_h in range(3, 6):
        for rect_w in range(3, 6):
            max_r0 = bar_row - rect_h - 2
            if max_r0 < 0:
                continue
            for r0 in range(0, max_r0 + 1):
                r1 = r0 + rect_h - 1
                for c0 in range(0, w - rect_w + 1):
                    c1 = c0 + rect_w - 1
                    bbox = (r0, c0, r1, c1)
                    if any(_point_near_bbox(pt, bbox, margin=1) for pt in protected):
                        continue
                    if any(_bbox_overlaps(bbox, other, margin=1) for other in existing):
                        continue
                    candidates.append(bbox)
    random.shuffle(candidates)
    return candidates


def _build_base_grid(shape: tuple[int, int], bar_row: int, red_sources: list[Coord]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    grid[bar_row, :] = GREEN
    for r, c in red_sources:
        grid[r, c] = RED
    return grid


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(14, 18)
    w = random.randint(13, 18)
    min_bar = max(8, h // 2 + 1)
    max_bar = h - 4
    if min_bar > max_bar:
        return None
    bar_row = random.randint(min_bar, max_bar)

    trigger_count = random.randint(1, 2)
    total_red = random.randint(trigger_count + 1, 4)

    all_cols = list(range(1, w - 1))
    if len(all_cols) < total_red:
        return None
    random.shuffle(all_cols)
    trigger_cols = sorted(all_cols[:trigger_count])
    other_cols = all_cols[trigger_count:total_red]

    red_sources: list[Coord] = [(bar_row - 2, c) for c in trigger_cols]
    for c in other_cols:
        row_choices = [r for r in range(1, bar_row - 2) if r != bar_row - 2]
        if not row_choices:
            return None
        red_sources.append((random.choice(row_choices), c))
    red_sources.sort()

    moved_reds = [(r + 1, c) for r, c in red_sources]
    base = _build_base_grid((h, w), bar_row, red_sources)

    first_bbox_candidates: list[BBox] = []
    for bbox in _all_rectangle_candidates((h, w), bar_row, red_sources, moved_reds, []):
        grid = base.copy()
        _draw_hollow_rectangle(grid, bbox)
        try:
            _analyze_generated_input(grid)
        except AssertionError:
            continue
        first_bbox_candidates.append(bbox)
    if not first_bbox_candidates:
        return None

    first_bbox = random.choice(first_bbox_candidates)
    grid = base.copy()
    _draw_hollow_rectangle(grid, first_bbox)

    want_second = random.random() < 0.45
    if want_second:
        second_candidates: list[BBox] = []
        for bbox in _all_rectangle_candidates((h, w), bar_row, red_sources, moved_reds, [first_bbox]):
            trial = grid.copy()
            _draw_hollow_rectangle(trial, bbox)
            try:
                _analyze_generated_input(trial)
            except AssertionError:
                continue
            second_candidates.append(bbox)
        if second_candidates:
            _draw_hollow_rectangle(grid, random.choice(second_candidates))

    try:
        _analyze_generated_input(grid)
        assert np.array_equal(main(grid), _simulate(grid)[0])
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
