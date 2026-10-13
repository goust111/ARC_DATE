from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _segments(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    indices = sorted(int(i) for i in indices)
    spans: list[tuple[int, int]] = []
    start = end = indices[0]
    for idx in indices[1:]:
        if idx == end + 1:
            end = idx
        else:
            spans.append((start, end))
            start = end = idx
    spans.append((start, end))
    return spans


def _barrier_color(grid: np.ndarray, bg: int) -> int:
    counts = Counter(map(int, grid.ravel()))
    foreground = {color: count for color, count in counts.items() if color != int(bg)}
    assert foreground, "expected non-background pixels"
    largest = max(foreground.values())
    winners = [color for color, count in foreground.items() if count == largest]
    assert len(winners) == 1, "barrier color must be uniquely most frequent among non-background colors"
    return int(winners[0])


def _find_legend(grid: np.ndarray, bg: int, barrier: int) -> tuple[list[int], set[tuple[int, int]]]:
    h, w = grid.shape
    candidates: list[tuple[list[int], set[tuple[int, int]]]] = []
    for c in (0, w - 1):
        rows = [r for r in range(h) if int(grid[r, c]) not in (int(bg), int(barrier))]
        if len(rows) < 4:
            continue
        if rows != list(range(rows[0], rows[-1] + 1)):
            continue
        colors = [int(grid[r, c]) for r in rows]
        if len(set(colors)) != len(colors):
            continue
        cells = {(int(r), int(c)) for r in rows}
        candidates.append((colors, cells))
    assert len(candidates) == 1, "expected one edge legend strip"
    return candidates[0]


def _extract_markers(
    grid: np.ndarray, bg: int, barrier: int, legend_cells: set[tuple[int, int]], cycle_index: dict[int, int]
) -> dict[tuple[int, int], int]:
    h, w = grid.shape
    markers: dict[tuple[int, int], int] = {}
    for r in range(h):
        for c in range(w):
            value = int(grid[r, c])
            if value in (int(bg), int(barrier)) or (r, c) in legend_cells:
                continue
            assert value in cycle_index, "marker colors must come from the legend cycle"
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    assert int(grid[rr, cc]) == int(bg), "markers must be isolated single pixels"
            markers[(int(r), int(c))] = value
    return markers


def _advance(color: int, steps: int, cycle: list[int], cycle_index: dict[int, int]) -> int:
    return int(cycle[(cycle_index[int(color)] + int(steps)) % len(cycle)])


def _plan_line_moves(
    barrier_positions: list[int],
    marker_entries: list[tuple[int, int]],
    line_len: int,
    cycle: list[int],
    cycle_index: dict[int, int],
) -> list[tuple[int, int, int]]:
    if not barrier_positions:
        return []
    spans = _segments(barrier_positions)
    assert len(spans) <= 1, "each row and column may intersect at most one barrier segment"
    assert len(spans) == 1
    start, end = spans[0]
    left = [(idx, color) for idx, color in marker_entries if idx < start]
    right = [(idx, color) for idx, color in marker_entries if idx > end]
    if not left or not right:
        return []
    assert len(left) == 1 and len(right) == 1, "paired line must have exactly one marker on each side"
    assert 0 < start and end < line_len - 1, "paired line needs free cells outside the barrier"
    thickness = end - start + 1
    left_src, left_color = left[0]
    right_src, right_color = right[0]
    return [
        (left_src, end + 1, _advance(left_color, thickness, cycle, cycle_index)),
        (right_src, start - 1, _advance(right_color, thickness, cycle, cycle_index)),
    ]


def _analyze(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape

    bg = _get_bg(g)
    barrier = _barrier_color(g, bg)
    cycle, legend_cells = _find_legend(g, bg, barrier)
    cycle_index = {color: idx for idx, color in enumerate(cycle)}
    markers = _extract_markers(g, bg, barrier, legend_cells, cycle_index)

    clears: list[tuple[int, int]] = []
    paints: list[tuple[int, int, int]] = []
    metadata: list[tuple[str, int, int]] = []

    for r in range(h):
        barrier_positions = [c for c in range(w) if int(g[r, c]) == int(barrier)]
        marker_entries = [(c, color) for (rr, c), color in markers.items() if rr == r]
        line_moves = _plan_line_moves(barrier_positions, marker_entries, w, cycle, cycle_index)
        if not line_moves:
            continue
        thickness = len(barrier_positions)
        metadata.append(("row", int(r), int(thickness)))
        for src_c, dst_c, new_color in line_moves:
            clears.append((int(r), int(src_c)))
            paints.append((int(r), int(dst_c), int(new_color)))

    for c in range(w):
        barrier_positions = [r for r in range(h) if int(g[r, c]) == int(barrier)]
        marker_entries = [(r, color) for (r, cc), color in markers.items() if cc == c]
        line_moves = _plan_line_moves(barrier_positions, marker_entries, h, cycle, cycle_index)
        if not line_moves:
            continue
        thickness = len(barrier_positions)
        metadata.append(("col", int(c), int(thickness)))
        for src_r, dst_r, new_color in line_moves:
            clears.append((int(src_r), int(c)))
            paints.append((int(dst_r), int(c), int(new_color)))

    clear_set = set(clears)
    assert len(clear_set) == len(clears), "a marker may not move in two directions"
    destinations = [(r, c) for r, c, _ in paints]
    assert len(set(destinations)) == len(destinations), "move destinations must be unique"
    for r, c, _ in paints:
        assert (r, c) not in legend_cells and int(g[r, c]) != int(barrier), "moves must land outside the barrier and legend"
        assert int(g[r, c]) == int(bg), "destination cells must start as background"

    return {
        "bg": int(bg),
        "barrier": int(barrier),
        "cycle": cycle,
        "legend_cells": legend_cells,
        "markers": markers,
        "clears": clears,
        "paints": paints,
        "metadata": metadata,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap isolated marker pixels across a wall and recolor them by the crossed wall thickness.

    Input structure:
    - The dominant color is the background.
    - One unique most frequent non-background color forms solid wall segments, with each row and column crossing at most one contiguous wall segment.
    - One edge column contains a vertical single-pixel legend strip whose top-to-bottom colors define a cyclic order.
    - All remaining non-background cells are isolated marker pixels; some rows or columns have one marker on each side of a wall segment.

    Rule:
    - For every row or column that crosses exactly one wall segment and has exactly one marker on each side, erase those two source markers.
    - Move each marker to the first cell just outside the wall on the opposite side.
    - Recolor each moved marker by advancing its original color forward through the legend cycle by k steps, where k is the wall thickness on that row or column.
    - Rows or columns with markers on only one side stay unchanged, and the wall plus legend remain fixed.

    Key insight:
    - The wall simultaneously determines where paired markers swap to and how far their colors must advance around the edge legend cycle.
    """
    info = _analyze(grid)
    out = np.asarray(grid, dtype=int).copy()
    for r, c in info["clears"]:
        out[r, c] = info["bg"]
    for r, c, color in info["paints"]:
        out[r, c] = int(color)
    return out


_PALETTE = list(range(10))
_GRID_MIN = 18
_GRID_MAX = 26
_RETRIES = 400


def _choose_distinct(values: list[int], count: int) -> list[int]:
    return random.sample(values, count)


def _plan_instance() -> dict | None:
    h = random.randint(_GRID_MIN, _GRID_MAX)
    w = random.randint(_GRID_MIN, _GRID_MAX)
    legend_side = random.choice(["left", "right"])

    colors = _choose_distinct(_PALETTE, 8)
    bg = colors[0]
    barrier = colors[1]
    cycle_len = random.choice([5, 6])
    cycle = colors[2 : 2 + cycle_len]

    top_gap = 3
    bottom_gap = 3
    left_gap = 5 if legend_side == "left" else 3
    right_gap = 5 if legend_side == "right" else 3

    max_stem_h = min(11, h - top_gap - bottom_gap)
    max_cross_w = min(11, w - left_gap - right_gap)
    if max_stem_h < 6 or max_cross_w < 8:
        return None

    stem_h = random.randint(6, max_stem_h)
    cross_h = random.randint(1, min(3, stem_h - 2))
    cross_h = min(cross_h, stem_h - 2)
    stem_w = random.randint(1, 3)
    stem_w = min(stem_w, max_cross_w - 2)
    if stem_w < 1:
        return None
    cross_w = random.randint(max(8, stem_w + 3), max_cross_w)

    vr0 = random.randint(top_gap, h - bottom_gap - stem_h)
    vr1 = vr0 + stem_h - 1
    hr0 = random.randint(vr0 + 1, vr1 - cross_h)
    hr1 = hr0 + cross_h - 1
    hc0 = random.randint(left_gap, w - right_gap - cross_w)
    hc1 = hc0 + cross_w - 1
    vc0 = random.randint(hc0 + 1, hc1 - stem_w)
    vc1 = vc0 + stem_w - 1

    legend_col = 0 if legend_side == "left" else w - 1
    legend_len = len(cycle)
    legend_starts = []
    if vr0 - legend_len - 1 >= 1:
        legend_starts.extend(range(1, vr0 - legend_len))
    if vr1 + legend_len + 2 <= h - 1:
        legend_starts.extend(range(vr1 + 2, h - legend_len))
    if not legend_starts:
        return None
    legend_r0 = random.choice(legend_starts)

    rows_cross = list(range(hr0, hr1 + 1))
    rows_stem = [r for r in range(vr0, vr1 + 1) if not (hr0 <= r <= hr1)]
    cols_stem = list(range(vc0, vc1 + 1))
    cols_cross = [c for c in range(hc0, hc1 + 1) if not (vc0 <= c <= vc1)]
    if len(rows_cross) < 1 or len(rows_stem) < 2 or len(cols_stem) < 1 or len(cols_cross) < 2:
        return None

    pair_row_cross = random.choice(rows_cross)
    pair_row_stem = random.choice(rows_stem)
    extra_row = random.choice([r for r in rows_stem if r != pair_row_stem])
    pair_col_stem = random.choice(cols_stem)
    pair_col_cross = random.choice(cols_cross)
    extra_col = random.choice([c for c in cols_cross if c != pair_col_cross])

    left_min = 2 if legend_side == "left" else 1
    right_max = w - 3 if legend_side == "right" else w - 2
    left_pool = list(range(left_min, hc0 - 1))
    right_pool = list(range(hc1 + 2, right_max + 1))
    top_pool = list(range(1, vr0 - 1))
    bottom_pool = list(range(vr1 + 2, h - 1))
    if len(left_pool) < 3 or len(right_pool) < 2 or len(top_pool) < 2 or len(bottom_pool) < 3:
        return None

    left_cols = random.sample(left_pool, 3)
    right_cols = random.sample(right_pool, 2)
    top_rows = random.sample(top_pool, 2)
    bottom_rows = random.sample(bottom_pool, 3)

    a, b, c, d = cycle[:4]
    markers = [
        {"coord": (pair_row_cross, left_cols[0]), "color": a, "move_to": (pair_row_cross, hc1 + 1), "thickness": cross_w, "axis": "row"},
        {"coord": (pair_row_cross, right_cols[0]), "color": b, "move_to": (pair_row_cross, hc0 - 1), "thickness": cross_w, "axis": "row"},
        {"coord": (pair_row_stem, left_cols[1]), "color": a, "move_to": (pair_row_stem, vc1 + 1), "thickness": stem_w, "axis": "row"},
        {"coord": (pair_row_stem, right_cols[1]), "color": c, "move_to": (pair_row_stem, vc0 - 1), "thickness": stem_w, "axis": "row"},
        {"coord": (top_rows[0], pair_col_stem), "color": b, "move_to": (vr1 + 1, pair_col_stem), "thickness": stem_h, "axis": "col"},
        {"coord": (bottom_rows[0], pair_col_stem), "color": c, "move_to": (vr0 - 1, pair_col_stem), "thickness": stem_h, "axis": "col"},
        {"coord": (top_rows[1], pair_col_cross), "color": a, "move_to": (hr1 + 1, pair_col_cross), "thickness": cross_h, "axis": "col"},
        {"coord": (bottom_rows[1], pair_col_cross), "color": d, "move_to": (hr0 - 1, pair_col_cross), "thickness": cross_h, "axis": "col"},
        {"coord": (extra_row, left_cols[2]), "color": d, "move_to": None, "thickness": None, "axis": "row"},
        {"coord": (bottom_rows[2], extra_col), "color": c, "move_to": None, "thickness": None, "axis": "col"},
    ]

    return {
        "shape": (h, w),
        "bg": bg,
        "barrier": barrier,
        "cycle": cycle,
        "legend_col": legend_col,
        "legend_r0": legend_r0,
        "barrier_box": (vr0, vr1, vc0, vc1, hr0, hr1, hc0, hc1),
        "markers": markers,
    }


def _render_instance(spec: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    h, w = spec["shape"]
    bg = int(spec["bg"])
    barrier = int(spec["barrier"])
    cycle = list(spec["cycle"])
    legend_col = int(spec["legend_col"])
    legend_r0 = int(spec["legend_r0"])
    vr0, vr1, vc0, vc1, hr0, hr1, hc0, hc1 = spec["barrier_box"]

    grid = np.full((h, w), bg, dtype=int)
    grid[vr0 : vr1 + 1, vc0 : vc1 + 1] = barrier
    grid[hr0 : hr1 + 1, hc0 : hc1 + 1] = barrier
    for i, color in enumerate(cycle):
        grid[legend_r0 + i, legend_col] = int(color)

    occupied = {(r, legend_col) for r in range(legend_r0, legend_r0 + len(cycle))}
    moved: list[dict] = []
    for marker in spec["markers"]:
        r, c = marker["coord"]
        assert (r, c) not in occupied
        assert int(grid[r, c]) == bg
        grid[r, c] = int(marker["color"])
        occupied.add((int(r), int(c)))
        if marker["move_to"] is not None:
            moved.append(marker)

    cycle_index = {color: idx for idx, color in enumerate(cycle)}
    expected = grid.copy()
    for marker in moved:
        r, c = marker["coord"]
        expected[r, c] = bg
    for marker in moved:
        r, c = marker["move_to"]
        expected[r, c] = _advance(marker["color"], marker["thickness"], cycle, cycle_index)

    meta = {
        "moved": moved,
        "markers": spec["markers"],
        "cycle": cycle,
        "barrier_box": spec["barrier_box"],
    }
    return grid, expected, meta


def _semantic_ok(grid: np.ndarray, expected: np.ndarray, meta: dict) -> bool:
    try:
        analyzed = _analyze(grid)
    except AssertionError:
        return False

    if not np.array_equal(main(grid), expected):
        return False

    moved = list(meta["moved"])
    if len(moved) < 4:
        return False
    axes = {marker["axis"] for marker in moved}
    if axes != {"row", "col"}:
        return False
    thicknesses = {int(marker["thickness"]) for marker in moved}
    if len(thicknesses) < 3:
        return False

    cycle = meta["cycle"]
    cycle_index = {color: idx for idx, color in enumerate(cycle)}
    by_input_color: dict[int, list[int]] = defaultdict(list)
    for marker in moved:
        by_input_color[int(marker["color"])].append(
            _advance(marker["color"], marker["thickness"], cycle, cycle_index)
        )
    if not any(len(set(outputs)) >= 2 and len(outputs) >= 2 for outputs in by_input_color.values()):
        return False

    if len(analyzed["metadata"]) != len(moved) // 2:
        return False
    if {axis for axis, _, _ in analyzed["metadata"]} != {"row", "col"}:
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(_RETRIES):
        spec = _plan_instance()
        if spec is None:
            continue
        try:
            grid, expected, meta = _render_instance(spec)
        except AssertionError:
            continue
        if _semantic_ok(grid, expected, meta):
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
