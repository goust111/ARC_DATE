from __future__ import annotations

from collections import defaultdict
import numpy as np


BG = 0
CARDINALS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors(shape: tuple[int, int], r: int, c: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in CARDINALS:
        rr = r + dr
        cc = c + dc
        if _in_bounds(shape, rr, cc):
            out.append((rr, cc))
    return out


def _find_markers(grid: np.ndarray) -> list[dict[str, object]]:
    h, w = grid.shape
    seen: set[frozenset[tuple[int, int]]] = set()
    markers: list[dict[str, object]] = []

    for r in range(h - 1):
        for c in range(w - 1):
            color = int(grid[r, c])
            if color == BG:
                continue
            block = grid[r : r + 2, c : c + 2]
            if not np.all(block == color):
                continue

            cells = frozenset(((r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)))
            if cells in seen:
                continue

            adjacent_same = {
                nbr
                for rr, cc in cells
                for nbr in _neighbors(grid.shape, rr, cc)
                if nbr not in cells and int(grid[nbr]) == color
            }
            if len(adjacent_same) != 1:
                continue

            endpoint = next(iter(adjacent_same))
            seen.add(cells)
            markers.append(
                {
                    "color": color,
                    "cells": cells,
                    "endpoint": endpoint,
                }
            )

    assert markers, "expected at least one 2x2 source marker"
    return markers


def _trace_path(grid: np.ndarray, marker: dict[str, object]) -> dict[str, object]:
    marker_cells = set(marker["cells"])
    endpoint = tuple(marker["endpoint"])
    color = int(marker["color"])

    continuations = [
        nbr
        for nbr in _neighbors(grid.shape, endpoint[0], endpoint[1])
        if nbr not in marker_cells and int(grid[nbr]) != BG
    ]
    assert len(continuations) == 1, "source endpoint must have one outgoing continuation"

    nxt = continuations[0]
    dr = int(nxt[0] - endpoint[0])
    dc = int(nxt[1] - endpoint[1])
    assert abs(dr) + abs(dc) == 1, "path must be orthogonal"

    coords = [endpoint]
    r, c = endpoint
    while True:
        nr = r + dr
        nc = c + dc
        if not _in_bounds(grid.shape, nr, nc) or int(grid[nr, nc]) == BG:
            break
        coords.append((nr, nc))
        r, c = nr, nc

    assert len(coords) >= 2, "path must contain more than one cell"
    kind = "h" if dr == 0 else "v"
    return {
        "color": color,
        "coords": coords,
        "marker_cells": marker_cells,
        "kind": kind,
    }


def _reconstruct_paths(grid: np.ndarray) -> list[dict[str, object]]:
    markers = _find_markers(grid)
    paths = [_trace_path(grid, marker) for marker in markers]
    colors = [int(path["color"]) for path in paths]
    assert len(colors) == len(set(colors)), "generator expects distinct path colors"
    return paths


def _coord_to_paths(paths: list[dict[str, object]]) -> dict[tuple[int, int], list[int]]:
    coord_map: dict[tuple[int, int], list[int]] = defaultdict(list)
    for pid, path in enumerate(paths):
        for coord in path["coords"]:
            coord_map[coord].append(pid)
    return coord_map


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor directed line paths by the color of each crossed path and remove source markers.

    Input structure:
    - Background is black.
    - Each path is a single-pixel-wide horizontal or vertical straight segment.
    - Every path has one 2x2 same-color source marker attached to one endpoint.
    - Horizontal and vertical paths may cross on a shared pixel that currently shows
      one path's original color.

    Rule:
    - Traverse each path from its marked endpoint toward its far endpoint.
    - Keep the current path color until an intersection is reached.
    - At each intersection, downstream cells on that path switch to the crossed path's
      original color.
    - The shared intersection pixel itself flips from the visible path color to the
      other path's original color.
    - Remove all 2x2 source markers.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be 2D"

    paths = _reconstruct_paths(g)
    coord_map = _coord_to_paths(paths)
    intersections = {
        coord: ids for coord, ids in coord_map.items() if len(ids) == 2
    }

    out = np.full_like(g, BG)

    for pid, path in enumerate(paths):
        current_color = int(path["color"])
        for coord in path["coords"]:
            hit = coord_map[coord]
            if len(hit) == 1:
                out[coord] = current_color
                continue

            assert len(hit) == 2, "only pairwise crossings are supported"
            other = hit[0] if hit[1] == pid else hit[1]
            current_color = int(paths[other]["color"])

    for coord, hit in intersections.items():
        a, b = hit
        color_here = int(g[coord])
        color_a = int(paths[a]["color"])
        color_b = int(paths[b]["color"])
        assert color_here in {color_a, color_b}, "intersection pixel must show one crossing color"
        out[coord] = color_b if color_here == color_a else color_a

    return out


# --------------- generator ---------------

GRID_MIN = 14
GRID_MAX = 23
MAX_H = 3
MAX_V = 3
MIN_GAP = 3
MAX_GENERATION_ATTEMPTS = 400
SCENE_ATTEMPTS = 400
MIN_PATH_LEN = 5
MAX_PATHS = 5
MIN_INTERSECTIONS = 1
MAX_INTERSECTIONS = 8


def _pick_spaced_positions(
    rng: np.random.Generator, values: list[int], count: int, gap: int
) -> list[int] | None:
    if count > len(values):
        return None
    for _ in range(200):
        chosen = sorted(int(x) for x in rng.choice(values, size=count, replace=False))
        if all(b - a >= gap for a, b in zip(chosen, chosen[1:])):
            return chosen
    return None


def _ordered_cells_from_def(path: dict[str, object]) -> list[tuple[int, int]]:
    if path["kind"] == "h":
        row = int(path["fixed"])
        lo = int(path["lo"])
        hi = int(path["hi"])
        if path["source"] == "left":
            return [(row, c) for c in range(lo, hi + 1)]
        return [(row, c) for c in range(hi, lo - 1, -1)]

    col = int(path["fixed"])
    lo = int(path["lo"])
    hi = int(path["hi"])
    if path["source"] == "top":
        return [(r, col) for r in range(lo, hi + 1)]
    return [(r, col) for r in range(hi, lo - 1, -1)]


def _marker_cells_from_def(path: dict[str, object]) -> set[tuple[int, int]]:
    endpoint = _ordered_cells_from_def(path)[0]
    r, c = endpoint
    bias = str(path["bias"])

    if path["kind"] == "h":
        rows = [r - 1, r] if bias == "up" else [r, r + 1]
        cols = [c - 2, c - 1] if path["source"] == "left" else [c + 1, c + 2]
    else:
        rows = [r - 2, r - 1] if path["source"] == "top" else [r + 1, r + 2]
        cols = [c - 1, c] if bias == "left" else [c, c + 1]

    return {(rr, cc) for rr in rows for cc in cols}


def _intersection_map_from_defs(
    paths: list[dict[str, object]]
) -> dict[tuple[int, int], tuple[int, int]]:
    coord_map = _coord_to_paths(
        [{"coords": _ordered_cells_from_def(path)} for path in paths]
    )
    intersections: dict[tuple[int, int], tuple[int, int]] = {}
    for coord, ids in coord_map.items():
        if len(ids) == 2:
            intersections[coord] = (ids[0], ids[1])
        else:
            assert len(ids) == 1, "scene generation must avoid triple overlaps"
    return intersections


def _validate_scene(paths: list[dict[str, object]], shape: tuple[int, int]) -> bool:
    h, w = shape
    line_cells = [_ordered_cells_from_def(path) for path in paths]
    line_sets = [set(cells) for cells in line_cells]
    intersections = _intersection_map_from_defs(paths)

    if not (MIN_INTERSECTIONS <= len(intersections) <= MAX_INTERSECTIONS):
        return False

    for path, cells in zip(paths, line_cells):
        if len(cells) < MIN_PATH_LEN:
            return False
        if any(not _in_bounds(shape, r, c) for r, c in cells):
            return False
        if cells[0] in intersections or cells[-1] in intersections:
            return False
        fixed_vals = {r for r, _ in cells} if path["kind"] == "h" else {c for _, c in cells}
        if len(fixed_vals) != 1:
            return False

    all_line_cells = set().union(*line_sets)
    for pid, cells in enumerate(line_cells):
        cell_set = line_sets[pid]
        for idx, coord in enumerate(cells):
            degree = sum(nbr in all_line_cells for nbr in _neighbors(shape, coord[0], coord[1]))
            if coord in intersections:
                if degree != 4:
                    return False
            elif idx == 0 or idx == len(cells) - 1:
                if degree != 1:
                    return False
            elif degree != 2:
                return False

    marker_sets = [_marker_cells_from_def(path) for path in paths]
    all_marker_cells: set[tuple[int, int]] = set()
    for pid, (path, marker_cells, cells) in enumerate(zip(paths, marker_sets, line_cells)):
        endpoint = cells[0]
        if any(not _in_bounds(shape, r, c) for r, c in marker_cells):
            return False
        if marker_cells & all_line_cells:
            return False
        if marker_cells & all_marker_cells:
            return False

        touching_lines = {
            nbr
            for rr, cc in marker_cells
            for nbr in _neighbors(shape, rr, cc)
            if nbr in all_line_cells
        }
        if touching_lines != {endpoint}:
            return False

        all_marker_cells |= marker_cells

    return True


def _render_input(
    paths: list[dict[str, object]],
    intersections: dict[tuple[int, int], tuple[int, int]],
    visible_owner: dict[tuple[int, int], int],
    shape: tuple[int, int],
) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)

    for path in paths:
        color = int(path["color"])
        for r, c in _ordered_cells_from_def(path):
            grid[r, c] = color

    for coord, (a, b) in intersections.items():
        owner = int(visible_owner[coord])
        assert owner in (a, b)
        grid[coord] = int(paths[owner]["color"])

    for path in paths:
        color = int(path["color"])
        for r, c in _marker_cells_from_def(path):
            grid[r, c] = color

    return grid


def _expected_output_from_defs(
    paths: list[dict[str, object]],
    intersections: dict[tuple[int, int], tuple[int, int]],
    visible_owner: dict[tuple[int, int], int],
    shape: tuple[int, int],
) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    ordered = [_ordered_cells_from_def(path) for path in paths]

    for pid, cells in enumerate(ordered):
        current_color = int(paths[pid]["color"])
        for coord in cells:
            if coord not in intersections:
                out[coord] = current_color
                continue
            a, b = intersections[coord]
            other = b if a == pid else a
            current_color = int(paths[other]["color"])

    for coord, (a, b) in intersections.items():
        owner = int(visible_owner[coord])
        other = b if owner == a else a
        out[coord] = int(paths[other]["color"])

    return out


def _sample_horizontal_path(
    rng: np.random.Generator,
    row: int,
    columns: list[int],
    width: int,
    color: int,
) -> dict[str, object] | None:
    source = str(rng.choice(["left", "right"]))
    bias = str(rng.choice(["up", "down"]))
    a = int(rng.integers(0, len(columns)))
    b = int(rng.integers(a, len(columns)))

    c0_min = 2 if source == "left" else 1
    c1_max = width - 3 if source == "right" else width - 2
    if columns[a] - 1 < c0_min or columns[b] + 1 > c1_max:
        return None

    lo = int(rng.integers(c0_min, columns[a]))
    hi = int(rng.integers(columns[b] + 1, c1_max + 1))
    if hi - lo + 1 < MIN_PATH_LEN:
        return None

    return {
        "kind": "h",
        "fixed": row,
        "lo": lo,
        "hi": hi,
        "source": source,
        "bias": bias,
        "color": color,
    }


def _sample_vertical_path(
    rng: np.random.Generator,
    col: int,
    rows: list[int],
    height: int,
    color: int,
) -> dict[str, object] | None:
    source = str(rng.choice(["top", "bottom"]))
    bias = str(rng.choice(["left", "right"]))
    a = int(rng.integers(0, len(rows)))
    b = int(rng.integers(a, len(rows)))

    r0_min = 2 if source == "top" else 1
    r1_max = height - 3 if source == "bottom" else height - 2
    if rows[a] - 1 < r0_min or rows[b] + 1 > r1_max:
        return None

    lo = int(rng.integers(r0_min, rows[a]))
    hi = int(rng.integers(rows[b] + 1, r1_max + 1))
    if hi - lo + 1 < MIN_PATH_LEN:
        return None

    return {
        "kind": "v",
        "fixed": col,
        "lo": lo,
        "hi": hi,
        "source": source,
        "bias": bias,
        "color": color,
    }


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    height = int(rng.integers(GRID_MIN, GRID_MAX + 1))
    width = int(rng.integers(GRID_MIN, GRID_MAX + 1))

    row_values = list(range(3, height - 3))
    col_values = list(range(3, width - 3))
    if len(row_values) < 1 or len(col_values) < 1:
        return None

    max_h = min(MAX_H, len(row_values))
    max_v = min(MAX_V, len(col_values))
    if max_h == 0 or max_v == 0:
        return None

    for _ in range(SCENE_ATTEMPTS):
        n_h = int(rng.integers(1, max_h + 1))
        n_v = int(rng.integers(1, max_v + 1))
        if not (2 <= n_h + n_v <= MAX_PATHS):
            continue

        rows = _pick_spaced_positions(rng, row_values, n_h, MIN_GAP)
        cols = _pick_spaced_positions(rng, col_values, n_v, MIN_GAP)
        if rows is None or cols is None:
            continue

        colors = [int(x) for x in rng.choice(np.arange(1, 10), size=n_h + n_v, replace=False)]
        paths: list[dict[str, object]] = []

        ok = True
        for idx, row in enumerate(rows):
            path = _sample_horizontal_path(rng, row, cols, width, colors[idx])
            if path is None:
                ok = False
                break
            paths.append(path)
        if not ok:
            continue

        for idx, col in enumerate(cols):
            path = _sample_vertical_path(rng, col, rows, height, colors[n_h + idx])
            if path is None:
                ok = False
                break
            paths.append(path)
        if not ok:
            continue

        if not _validate_scene(paths, (height, width)):
            continue

        intersections = _intersection_map_from_defs(paths)
        if not intersections:
            continue

        visible_owner = {}
        for coord, pair in intersections.items():
            visible_owner[coord] = int(pair[int(rng.integers(0, 2))])

        inp = _render_input(paths, intersections, visible_owner, (height, width))
        expected = _expected_output_from_defs(paths, intersections, visible_owner, (height, width))
        try:
            actual = main(inp)
        except AssertionError:
            continue
        if np.array_equal(actual, expected):
            return inp

    return None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid path-relay input")
