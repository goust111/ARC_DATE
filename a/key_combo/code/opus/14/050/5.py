from __future__ import annotations

import random
from collections import Counter

import numpy as np


BLACK = 0
PATH_COLORS = (1, 2, 3, 4)
SEPARATOR = 5
PINK = 6
MARKER_COLORS = (7, 8, 9)

DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
RETRY_LIMIT = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def _neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _is_connected(cells: set[tuple[int, int]]) -> bool:
    start = next(iter(cells))
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for nbr in _neighbors4(r, c):
            if nbr in cells and nbr not in seen:
                seen.add(nbr)
                stack.append(nbr)
    return seen == cells


def _endpoints(cells: set[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int]]:
    degrees = {cell: sum(nbr in cells for nbr in _neighbors4(*cell)) for cell in cells}
    assert all(degree in (1, 2) for degree in degrees.values()), "path must be an unbranched chain"
    endpoints = [cell for cell, degree in degrees.items() if degree == 1]
    assert len(endpoints) == 2, "path must have exactly two endpoints"
    return tuple(sorted(endpoints))


def _path_columns(cells: set[tuple[int, int]]) -> tuple[int, ...]:
    return tuple(sorted({c for _r, c in cells}))


def _analyze_input(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    bg = _get_bg(g)
    assert bg == BLACK, "background must be black"

    separator_rows = [r for r in range(h) if np.all(g[r, :] == SEPARATOR)]
    assert len(separator_rows) == 1, "expected one full-width gray separator row"
    sep = int(separator_rows[0])

    pink_coords = np.argwhere(g[:sep, :] == PINK)
    assert pink_coords.size > 0, "expected a pink marker run above the separator"
    pink_rows = sorted({int(r) for r, _c in pink_coords})
    assert len(pink_rows) == 1, "pink marker must occupy one row"
    pink_row = pink_rows[0]
    pink_cols = sorted(int(c) for _r, c in pink_coords)
    assert 2 <= len(pink_cols) <= 4, "pink marker run length must be 2..4"
    assert all(pink_cols[i + 1] == pink_cols[i] + 1 for i in range(len(pink_cols) - 1)), "pink marker must be contiguous"
    t = len(pink_cols)

    paths: list[dict] = []
    coord_to_path: dict[tuple[int, int], int] = {}
    for color in PATH_COLORS:
        coords = {(int(r), int(c)) for r, c in np.argwhere(g == color)}
        if not coords:
            continue
        assert all(r < sep for r, _c in coords), "paths must stay above the separator"
        assert 4 <= len(coords) <= 8, "path size must be 4..8"
        assert _is_connected(coords), "path color must form one connected object"
        endpoints = _endpoints(coords)
        columns = _path_columns(coords)
        path_index = len(paths)
        for cell in coords:
            assert cell not in coord_to_path, "paths must not overlap"
            coord_to_path[cell] = path_index
        paths.append(
            {
                "color": int(color),
                "cells": coords,
                "endpoints": endpoints,
                "columns": columns,
                "visible_endpoint": None,
                "gap": None,
                "marker_row": None,
                "marker_color": None,
                "marker_col": None,
            }
        )

    assert 2 <= len(paths) <= 4, "expected 2..4 path objects"

    marker_cells = [
        (r, c, int(g[r, c]))
        for r in range(sep + 1, h)
        for c in range(w)
        if int(g[r, c]) not in (BLACK, SEPARATOR)
    ]
    assert len(marker_cells) == len(paths), "expected one lower marker per path"
    assert all(color in MARKER_COLORS for _r, _c, color in marker_cells), "unexpected lower-marker color"

    for r, c, color in marker_cells:
        below_neighbors = [
            (nr, nc)
            for nr, nc in _neighbors4(r, c)
            if sep < nr < h and 0 <= nc < w and int(g[nr, nc]) not in (BLACK, SEPARATOR)
        ]
        assert not below_neighbors, "lower markers must be isolated pixels"

    matched_paths = set()
    markers = []
    for mr, mc, marker_color in marker_cells:
        hit = None
        gap = 0
        for r in range(mr - 1, -1, -1):
            value = int(g[r, mc])
            if value == BLACK:
                gap += 1
                continue
            if value == SEPARATOR:
                assert r == sep, "only the separator may lie on the sightline"
                continue
            hit = (r, mc)
            break

        assert hit is not None, "each lower marker must see a path endpoint"
        assert hit in coord_to_path, "sightline must hit a path cell"
        path_index = coord_to_path[hit]
        path = paths[path_index]
        assert hit in path["endpoints"], "sightline must hit an endpoint"
        assert path_index not in matched_paths, "each path must be matched by exactly one lower marker"
        matched_paths.add(path_index)

        path["visible_endpoint"] = hit
        path["gap"] = int(gap)
        path["marker_row"] = int(mr)
        path["marker_color"] = int(marker_color)
        path["marker_col"] = int(mc)
        markers.append(
            {
                "row": int(mr),
                "col": int(mc),
                "color": int(marker_color),
                "path_index": int(path_index),
                "hit": hit,
                "gap": int(gap),
            }
        )

    assert len(matched_paths) == len(paths), "marker/path matching must be one-to-one"
    selected = [i for i, path in enumerate(paths) if int(path["gap"]) == t]
    assert 1 <= len(selected) <= 2, "expected exactly 1 or 2 gated paths"

    return {
        "grid": g,
        "height": h,
        "width": w,
        "separator": sep,
        "pink_row": pink_row,
        "pink_cols": tuple(pink_cols),
        "t": t,
        "paths": paths,
        "markers": markers,
        "selected": tuple(selected),
    }


def _render_bars(grid: np.ndarray, analysis: dict, *, selected_paths: set[int] | None = None, endpoint_only: bool = False) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    sep = int(analysis["separator"])
    t = int(analysis["t"])
    out = g.copy()
    selected = set(analysis["selected"] if selected_paths is None else selected_paths)
    top = sep - t
    assert top >= 0, "bars must fit above the separator"

    for path_index in selected:
        path = analysis["paths"][path_index]
        cols = [int(path["visible_endpoint"][1])] if endpoint_only else list(path["columns"])
        for col in cols:
            out[top:sep, int(col)] = int(path["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw bars only for the path columns whose matched sightline gap equals the magenta run length.

    Input structure
    - The background is black, with one full-width gray separator row in the lower half.
    - Above the separator there is one contiguous magenta run and 2..4 one-pixel-wide 4-connected path objects in distinct path colors.
    - Below the separator there are the same number of isolated colored marker pixels, each sharing a column with exactly one path endpoint through a black sightline.

    Rule
    - Count the magenta run length T.
    - For each lower marker, scan upward in its column to the first non-black, non-gray pixel; this must be a path endpoint, and the number of black cells on that sightline is the gap G.
    - Keep only the matched paths whose gap satisfies G == T.
    - For each kept path, collect every column occupied by that path and fill the T cells immediately above the separator in those columns with the path color, overwriting any previous content there.

    Key insight
    - The magenta run length is both the gate and the bar height, while the lower marker-to-endpoint sightline supplies the measured gap that decides which path columns become histogram bars.
    """
    analysis = _analyze_input(grid)
    return _render_bars(np.asarray(grid, dtype=int), analysis)


def _allocate_intervals(width: int, count: int) -> list[tuple[int, int]] | None:
    for _ in range(80):
        spans = [random.randint(2, 4) for _ in range(count)]
        required = sum(spans) + (count - 1)
        if required > width:
            continue
        slack = width - required
        extras = [0] * (count + 1)
        for _ in range(slack):
            extras[random.randrange(count + 1)] += 1

        intervals = []
        cur = extras[0]
        for i, span in enumerate(spans):
            intervals.append((cur, cur + span - 1))
            cur += span
            if i < count - 1:
                cur += 1 + extras[i + 1]
        return intervals
    return None


def _build_path(visible_col: int, far_col: int, endpoint_row: int, color: int, min_rise: int) -> dict | None:
    hseg = abs(far_col - visible_col)
    rise_lo = max(1, 3 - hseg, min_rise)
    rise_hi = min(4, 7 - hseg, endpoint_row)
    if rise_lo > rise_hi:
        return None

    rise_total = random.randint(rise_lo, rise_hi)
    v1 = random.randint(1, rise_total)
    v2 = rise_total - v1
    sign = 1 if far_col > visible_col else -1

    cells = [(endpoint_row, visible_col)]
    for step in range(1, v1 + 1):
        cells.append((endpoint_row - step, visible_col))
    turn_row = endpoint_row - v1
    for col in range(visible_col + sign, far_col + sign, sign):
        cells.append((turn_row, col))
    for step in range(1, v2 + 1):
        cells.append((turn_row - step, far_col))

    cell_set = set(cells)
    if len(cell_set) != len(cells):
        return None
    if not _is_connected(cell_set):
        return None

    endpoints = _endpoints(cell_set)
    assert (endpoint_row, visible_col) in endpoints
    return {
        "color": int(color),
        "cells": tuple(cells),
        "cell_set": cell_set,
        "columns": tuple(sorted({c for _r, c in cells})),
        "endpoints": endpoints,
        "visible_endpoint": (int(endpoint_row), int(visible_col)),
        "far_endpoint": endpoints[0] if endpoints[1] == (endpoint_row, visible_col) else endpoints[1],
        "length": len(cells),
    }


def _choose_run_slot(
    h: int,
    w: int,
    sep: int,
    t: int,
    paths: list[dict],
    selected: set[int],
    forbidden_cols: set[int],
) -> tuple[int, int] | None:
    occupied = {(r, c) for path in paths for r, c in path["cells"]}
    rows = [sep - 1, sep - 2]
    random.shuffle(rows)
    candidates = []
    selected_cols = {c for i in selected for c in paths[i]["columns"]}
    for row in rows:
        if row < 0:
            continue
        for start in range(0, w - t + 1):
            segment = [(row, c) for c in range(start, start + t)]
            if any(cell in occupied for cell in segment):
                continue
            if any(c in forbidden_cols for _r, c in segment):
                continue
            overlap = sum(c in selected_cols for _r, c in segment)
            candidates.append((overlap, random.random(), row, start))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    _score, _rand, row, start = candidates[0]
    return int(row), int(start)


def _render_input_from_plan(h: int, w: int, sep: int, pink_row: int, pink_start: int, t: int, paths: list[dict], marker_specs: list[dict]) -> np.ndarray:
    g = np.zeros((h, w), dtype=int)
    g[sep, :] = SEPARATOR
    g[pink_row, pink_start : pink_start + t] = PINK
    for path in paths:
        for r, c in path["cells"]:
            g[r, c] = int(path["color"])
    for spec in marker_specs:
        g[int(spec["row"]), int(spec["col"])] = int(spec["color"])
    return g


def _shortcut_endpoint_row_only(grid: np.ndarray) -> np.ndarray:
    analysis = _analyze_input(grid)
    sep = int(analysis["separator"])
    t = int(analysis["t"])
    chosen = {
        i
        for i, path in enumerate(analysis["paths"])
        if int(sep - path["visible_endpoint"][0]) == t
    }
    return _render_bars(grid, analysis, selected_paths=chosen)


def _shortcut_endpoint_columns_only(grid: np.ndarray) -> np.ndarray:
    analysis = _analyze_input(grid)
    return _render_bars(grid, analysis, endpoint_only=True)


def _gen_attempt() -> np.ndarray | None:
    w = random.randint(16, 20)
    sep = random.randint(9, 12)
    h = random.randint(max(15, sep + 4), 19)
    t = random.randint(2, 4)

    selected_count = random.choice((1, 2))
    path_count = random.randint(selected_count + 1, 4)
    intervals = _allocate_intervals(w, path_count)
    if intervals is None:
        return None

    path_colors = random.sample(list(PATH_COLORS), path_count)
    selected_indices = set(random.sample(range(path_count), selected_count))
    nonselected_indices = [i for i in range(path_count) if i not in selected_indices]
    first_selected = next(iter(selected_indices))
    decoy = nonselected_indices[0]

    below_black = [None] * path_count
    gaps = [None] * path_count

    below_black[first_selected] = random.randint(1, 2)
    gaps[first_selected] = t

    shared_above = gaps[first_selected] - below_black[first_selected]
    below_black[decoy] = 3 - below_black[first_selected]
    gaps[decoy] = shared_above + below_black[decoy]

    for index in range(path_count):
        if below_black[index] is None:
            below_black[index] = random.randint(1, 2)
        if gaps[index] is None:
            choices = [g for g in range(below_black[index], 6) if g != t]
            if not choices:
                return None
            gaps[index] = random.choice(choices)

    paths = []
    marker_specs = []
    shared_endpoint_row = sep - shared_above - 1
    for index, interval in enumerate(intervals):
        left, right = interval
        direction = random.choice((-1, 1))
        visible_col = right if direction < 0 else left
        far_col = left if direction < 0 else right
        endpoint_row = sep + below_black[index] - gaps[index] - 1
        if endpoint_row < 2:
            return None

        min_rise = below_black[index] if index in selected_indices else 1
        path = _build_path(visible_col, far_col, endpoint_row, path_colors[index], min_rise)
        if path is None:
            return None
        if index in selected_indices and len(path["columns"]) < 2:
            return None

        paths.append(path)
        marker_specs.append(
            {
                "row": sep + below_black[index] + 1,
                "col": visible_col,
                "color": random.choice(MARKER_COLORS),
                "gap": gaps[index],
                "below_black": below_black[index],
                "endpoint_row": endpoint_row,
            }
        )

    forbidden_cols = {int(path["visible_endpoint"][1]) for path in paths}
    pink_slot = _choose_run_slot(h, w, sep, t, paths, selected_indices, forbidden_cols)
    if pink_slot is None:
        return None
    pink_row, pink_start = pink_slot

    grid = _render_input_from_plan(h, w, sep, pink_row, pink_start, t, paths, marker_specs)
    out = main(grid)
    analysis = _analyze_input(grid)

    if len(analysis["selected"]) != selected_count:
        return None
    if set(analysis["selected"]) != selected_indices:
        return None
    if np.array_equal(grid, out):
        return None

    colors_to_analysis = {path["color"]: path for path in analysis["paths"]}
    for planned, marker in zip(paths, marker_specs):
        actual = colors_to_analysis[int(planned["color"])]
        if tuple(actual["columns"]) != tuple(planned["columns"]):
            return None
        if tuple(sorted(actual["cells"])) != tuple(sorted(planned["cell_set"])):
            return None
        if int(actual["gap"]) != int(marker["gap"]):
            return None
        if int(actual["marker_row"]) != int(marker["row"]):
            return None

    actual_shared_rows = [
        path["visible_endpoint"][0]
        for i, path in enumerate(analysis["paths"])
        if i in {first_selected, decoy}
    ]
    if len(set(actual_shared_rows)) != 1 or actual_shared_rows[0] != shared_endpoint_row:
        return None

    if np.array_equal(_shortcut_endpoint_row_only(grid), out):
        return None
    if np.array_equal(_shortcut_endpoint_columns_only(grid), out):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
