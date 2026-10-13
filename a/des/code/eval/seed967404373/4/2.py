import random
from collections import deque

import numpy as np


BG = 0
GEN_RETRIES = 400
COLOR_POOL = list(range(1, 10))
MIN_LANE_GAP = 4


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _segment_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = a
    r1, c1 = b
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        return [(r0, c) for c in range(c0, c1 + step, step)]
    if c0 == c1:
        step = 1 if r1 >= r0 else -1
        return [(r, c0) for r in range(r0, r1 + step, step)]
    raise AssertionError("segments must be axis-aligned")


def _polyline_cells(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for i in range(len(points) - 1):
        seg = _segment_cells(points[i], points[i + 1])
        if i:
            seg = seg[1:]
        cells.extend(seg)
    return cells


def _choose_spaced(
    rng: random.Random, lo: int, hi: int, count: int, min_gap: int
) -> list[int] | None:
    if count == 0:
        return []
    if hi < lo:
        return None
    span = hi - lo + 1
    if span < 1 + (count - 1) * min_gap:
        return None
    values = list(range(lo, hi + 1))
    for _ in range(200):
        picks = sorted(rng.sample(values, count))
        if all(b - a >= min_gap for a, b in zip(picks, picks[1:])):
            return picks
    return None


def _legend_columns(grid: np.ndarray) -> tuple[int, int]:
    h, w = grid.shape
    c0 = None
    last_non_bg = None
    blank_run = 0
    for c in range(w):
        has_fg = bool(np.any(grid[:, c] != BG))
        if has_fg:
            if c0 is None:
                c0 = c
            last_non_bg = c
            blank_run = 0
        elif c0 is not None:
            blank_run += 1
            if blank_run >= 2 and last_non_bg is not None:
                return int(c0), int(last_non_bg)
    raise AssertionError("expected a left legend followed by at least two blank columns")


def _legend_priority(grid: np.ndarray) -> tuple[list[int], tuple[int, int]]:
    c0, c1 = _legend_columns(grid)
    sub = grid[:, c0 : c1 + 1]
    h, w = sub.shape
    seen = np.zeros((h, w), dtype=bool)
    entries: list[tuple[int, int, int]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(sub[r, c]) == BG:
                continue
            color = int(sub[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(sub[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            min_r = min(r0 for r0, _ in cells)
            min_c = min(c0_ for _, c0_ in cells)
            entries.append((min_r, min_c, color))

    entries.sort()
    priority = [color for _, _, color in entries]
    assert priority and len(priority) == len(set(priority)), "legend colors must be unique"
    return priority, (c0, c1)


def _find_crossings(grid: np.ndarray, legend_cols: tuple[int, int]) -> list[tuple[int, int, int, int]]:
    h, w = grid.shape
    c0, c1 = legend_cols
    crossings: list[tuple[int, int, int, int]] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if c0 <= c <= c1:
                continue
            center = int(grid[r, c])
            if center == BG:
                continue
            left = int(grid[r, c - 1])
            right = int(grid[r, c + 1])
            up = int(grid[r - 1, c])
            down = int(grid[r + 1, c])
            if left == BG or up == BG:
                continue
            if left == right and up == down and left != up and center in (left, up):
                crossings.append((r, c, left, up))
    return crossings


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor path crossings using the top-to-bottom priority legend on the left.

    At each crossing, the higher-priority path wins at the center pixel and also
    covers one extra pixel in both directions along the lower-priority path.
    """
    g = np.asarray(grid, dtype=int)
    priority, legend_cols = _legend_priority(g)
    rank = {color: i for i, color in enumerate(priority)}
    out = g.copy()

    for r, c, h_color, v_color in _find_crossings(g, legend_cols):
        assert h_color in rank and v_color in rank, "all path colors must appear in the legend"
        if rank[h_color] < rank[v_color]:
            high, low, low_axis = h_color, v_color, "v"
        else:
            high, low, low_axis = v_color, h_color, "h"

        out[r, c] = high
        if low_axis == "h":
            if int(g[r, c - 1]) == low:
                out[r, c - 1] = high
            if int(g[r, c + 1]) == low:
                out[r, c + 1] = high
        else:
            if int(g[r - 1, c]) == low:
                out[r - 1, c] = high
            if int(g[r + 1, c]) == low:
                out[r + 1, c] = high

    return out


def _random_corner_path(
    rng: random.Random, zone: tuple[int, int, int, int]
) -> list[tuple[int, int]] | None:
    r0, r1, c0, c1 = zone
    if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
        return None

    ra = rng.randint(r0, r1 - 2)
    rb = rng.randint(ra + 2, r1)
    ca = rng.randint(c0, c1 - 2)
    cb = rng.randint(ca + 2, c1)

    variants = [
        [(ra, ca), (ra, cb), (rb, cb)],
        [(rb, ca), (rb, cb), (ra, cb)],
        [(ra, cb), (ra, ca), (rb, ca)],
        [(rb, cb), (rb, ca), (ra, ca)],
    ]
    return _polyline_cells(rng.choice(variants))


def _build_expected(
    inp: np.ndarray,
    priority: list[int],
    intersections: list[tuple[int, int, int, int]],
) -> np.ndarray:
    out = inp.copy()
    rank = {color: i for i, color in enumerate(priority)}
    for r, c, h_color, v_color in intersections:
        if rank[h_color] < rank[v_color]:
            high, low, low_axis = h_color, v_color, "v"
        else:
            high, low, low_axis = v_color, h_color, "h"
        out[r, c] = high
        if low_axis == "h":
            out[r, c - 1] = high
            out[r, c + 1] = high
        else:
            out[r - 1, c] = high
            out[r + 1, c] = high
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_h = rng.randint(1, 3)
    n_v = rng.randint(1, 3)
    max_paths = n_h + n_v + 2

    legend_w = rng.choice([1, 2])
    block_h = rng.choice([1, 2])
    legend_gap_r = 1
    legend_h_max = max_paths * block_h + (max_paths - 1) * legend_gap_r

    min_h = max(legend_h_max + 1, 10 + (n_h - 1) * MIN_LANE_GAP)
    min_w = max(legend_w + 3 + 12, legend_w + 3 + 7 + (n_v - 1) * MIN_LANE_GAP)
    if min_h > 30 or min_w > 30:
        return None

    h = rng.randint(min_h, min(30, min_h + 6))
    w = rng.randint(min_w, min(30, min_w + 8))

    legend_c0 = rng.choice([0, 1])
    gap_cols = rng.choice([2, 3])
    main_c0 = legend_c0 + legend_w + gap_cols
    if main_c0 + 6 >= w:
        return None

    rows = _choose_spaced(rng, 2, h - 3, n_h, MIN_LANE_GAP)
    cols = _choose_spaced(rng, main_c0 + 2, w - 3, n_v, MIN_LANE_GAP)
    if rows is None or cols is None:
        return None

    min_row, max_row = min(rows), max(rows)
    min_col, max_col = min(cols), max(cols)
    if min_row <= 1 or min_col <= main_c0 + 1:
        return None
    if max_row >= h - 2 or max_col >= w - 2:
        return None

    h_paths = []
    for row in rows:
        c0 = rng.randint(main_c0, min_col - 1)
        c1 = rng.randint(max_col + 1, w - 2)
        h_paths.append({"axis": "h", "row": row, "c0": c0, "c1": c1})

    v_paths = []
    for col in cols:
        r0 = rng.randint(1, min_row - 1)
        r1 = rng.randint(max_row + 1, h - 2)
        v_paths.append({"axis": "v", "col": col, "r0": r0, "r1": r1})

    h_left = min(path["c0"] for path in h_paths)
    h_right = max(path["c1"] for path in h_paths)
    v_top = min(path["r0"] for path in v_paths)
    v_bottom = max(path["r1"] for path in v_paths)

    zones = []
    if v_top - 0 >= 3 and h_left - main_c0 >= 3:
        zones.append((0, v_top - 1, main_c0, h_left - 1))
    if v_top - 0 >= 3 and w - 1 - h_right >= 3:
        zones.append((0, v_top - 1, h_right + 1, w - 1))
    if h - 1 - v_bottom >= 3 and h_left - main_c0 >= 3:
        zones.append((v_bottom + 1, h - 1, main_c0, h_left - 1))
    if h - 1 - v_bottom >= 3 and w - 1 - h_right >= 3:
        zones.append((v_bottom + 1, h - 1, h_right + 1, w - 1))

    max_bent = min(2, len(zones))
    n_bent = rng.randint(0, max_bent)
    total_paths = n_h + n_v + n_bent

    legend_h = total_paths * block_h + (total_paths - 1) * legend_gap_r
    if legend_h > h:
        return None

    colors = rng.sample(COLOR_POOL, total_paths)
    priority = colors[:]
    rng.shuffle(priority)
    path_defs = []
    color_iter = iter(colors)

    for spec in h_paths:
        color = next(color_iter)
        cells = _segment_cells((spec["row"], spec["c0"]), (spec["row"], spec["c1"]))
        spec = dict(spec)
        spec["color"] = color
        spec["cells"] = cells
        path_defs.append(spec)

    for spec in v_paths:
        color = next(color_iter)
        cells = _segment_cells((spec["r0"], spec["col"]), (spec["r1"], spec["col"]))
        spec = dict(spec)
        spec["color"] = color
        spec["cells"] = cells
        path_defs.append(spec)

    rng.shuffle(zones)
    for zone in zones[:n_bent]:
        color = next(color_iter)
        cells = _random_corner_path(rng, zone)
        if cells is None:
            return None
        path_defs.append({"axis": "bend", "color": color, "cells": cells})

    grid = np.zeros((h, w), dtype=int)
    legend_r0 = rng.randint(0, h - legend_h)
    for i, color in enumerate(priority):
        r0 = legend_r0 + i * (block_h + legend_gap_r)
        grid[r0 : r0 + block_h, legend_c0 : legend_c0 + legend_w] = color

    occupancy: dict[tuple[int, int], list[int]] = {}
    for path in path_defs:
        for cell in path["cells"]:
            occupancy.setdefault(cell, []).append(int(path["color"]))

    intersections: list[tuple[int, int, int, int]] = []
    for h_path in [p for p in path_defs if p["axis"] == "h"]:
        for v_path in [p for p in path_defs if p["axis"] == "v"]:
            cell = (int(h_path["row"]), int(v_path["col"]))
            if cell not in occupancy or len(occupancy[cell]) != 2:
                return None
            intersections.append((cell[0], cell[1], int(h_path["color"]), int(v_path["color"])))

    if not intersections:
        return None

    for path in path_defs:
        for r, c in path["cells"]:
            owners = occupancy[(r, c)]
            if len(owners) == 1:
                if int(grid[r, c]) != BG:
                    return None
                grid[r, c] = int(path["color"])

    for r, c, h_color, v_color in intersections:
        visible = rng.choice([h_color, v_color])
        grid[r, c] = visible

    expected = _build_expected(grid, priority, intersections)

    if len(_find_crossings(grid, _legend_columns(grid))) != len(intersections):
        return None
    if not np.array_equal(main(grid), expected):
        return None

    for r, c, h_color, v_color in intersections:
        if int(grid[r, c - 1]) != h_color or int(grid[r, c + 1]) != h_color:
            return None
        if int(grid[r - 1, c]) != v_color or int(grid[r + 1, c]) != v_color:
            return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
