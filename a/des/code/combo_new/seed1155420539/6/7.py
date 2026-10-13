from __future__ import annotations

import random
from typing import Callable

import numpy as np


BG = 0
GREY = 5
PATH_PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _is_edge(shape: tuple[int, int], r: int, c: int) -> bool:
    return r == 0 or c == 0 or r == shape[0] - 1 or c == shape[1] - 1


def _same_edge(shape: tuple[int, int], a: tuple[int, int], b: tuple[int, int]) -> bool:
    h, w = shape
    ar, ac = a
    br, bc = b
    return (
        (ac == 0 and bc == 0)
        or (ac == w - 1 and bc == w - 1)
        or (ar == 0 and br == 0)
        or (ar == h - 1 and br == h - 1)
    )


def _neighbors4(shape: tuple[int, int], r: int, c: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in DIRS4:
        nr, nc = r + dr, c + dc
        if _in_bounds(shape, nr, nc):
            out.append((nr, nc))
    return out


def _occupied(grid: np.ndarray, r: int, c: int) -> bool:
    return int(grid[r, c]) not in (BG, GREY)


def _path_neighbors(grid: np.ndarray, cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    return [(nr, nc) for nr, nc in _neighbors4(grid.shape, r, c) if _occupied(grid, nr, nc)]


def _trace_endpoint(grid: np.ndarray, marker: tuple[int, int]) -> tuple[tuple[int, int], int]:
    h, w = grid.shape
    mr, mc = marker
    assert _is_edge((h, w), mr, mc), "grey markers must lie on the grid edge"

    starts = [(nr, nc) for nr, nc in _neighbors4(grid.shape, mr, mc) if _occupied(grid, nr, nc)]
    assert len(starts) == 1, "each grey marker must touch exactly one path start"
    start = starts[0]
    sr, sc = start
    assert _is_edge((h, w), sr, sc), "path starts must touch the grid edge"
    color = int(grid[sr, sc])

    next_cells = _path_neighbors(grid, start)
    if not next_cells:
        return start, color
    assert len(next_cells) == 1, "a marked start cell must have one onward path neighbor"

    prev = start
    current = next_cells[0]
    direction = (current[0] - start[0], current[1] - start[1])

    for _ in range(grid.size * 2):
        cr, cc = current
        fr, fc = cr + direction[0], cc + direction[1]
        if _is_edge((h, w), cr, cc) and not _in_bounds((h, w), fr, fc):
            return current, color

        if _in_bounds((h, w), fr, fc) and _occupied(grid, fr, fc):
            prev, current = current, (fr, fc)
            continue

        options = [cell for cell in _path_neighbors(grid, current) if cell != prev]
        if not options:
            assert _is_edge((h, w), cr, cc), "only border cells may terminate a path"
            return current, color
        assert len(options) == 1, "after excluding straight-through, each step must be unique"
        nxt = options[0]
        direction = (nxt[0] - cr, nxt[1] - cc)
        prev, current = current, nxt

    raise AssertionError("path tracing exceeded the grid budget")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace each grey-labelled signal path through crossings and paint that path's far edge endpoint.

    Input structure:
    - The background is black, and several one-pixel-wide colored orthogonal paths
      run from one border cell to another border cell.
    - Different paths may share a crossing cell; the input can show only one of the
      two colors at such a crossing even though both paths pass through it.
    - Each path has one grey border marker adjacent to its start border cell, while
      the opposite border endpoint has no grey marker.

    Rule:
    - For each grey marker, enter the unique adjacent border path cell and take that
      visible color as the identity of the path being traced.
    - Follow 4-neighbor path geometry; if the cell straight ahead is occupied, keep
      going straight through the crossing, otherwise take the unique available turn.
    - When continuing straight would leave the grid, the current border cell is the
      path's terminal endpoint.
    - Remove all grey markers and recolor every traced endpoint with its path color,
      while leaving the rest of the grid unchanged.

    Key insight:
    - Crossing cells are geometrically shared by two paths, so color continuity alone
      is unreliable; the correct route is recovered by preserving travel direction at
      crossings and using the grey marker only to choose the starting path identity.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    markers = [tuple(map(int, cell)) for cell in np.argwhere(g == GREY)]
    assert markers, "expected at least one grey marker"
    out[g == GREY] = BG

    for marker in markers:
        endpoint, color = _trace_endpoint(g, marker)
        out[endpoint] = color

    return out


def _segment_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    ar, ac = a
    br, bc = b
    if ar == br:
        step = 1 if bc >= ac else -1
        return [(ar, c) for c in range(ac, bc + step, step)]
    if ac == bc:
        step = 1 if br >= ar else -1
        return [(r, ac) for r in range(ar, br + step, step)]
    raise AssertionError("polyline vertices must be axis-aligned")


def _vertices_to_cells(vertices: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for i in range(len(vertices) - 1):
        seg = _segment_cells(vertices[i], vertices[i + 1])
        if i:
            seg = seg[1:]
        cells.extend(seg)
    assert cells, "a path needs at least one cell"
    return cells


def _local_dirs(cells: list[tuple[int, int]]) -> dict[tuple[int, int], frozenset[tuple[int, int]]]:
    dirs: dict[tuple[int, int], set[tuple[int, int]]] = {cell: set() for cell in cells}
    for (r0, c0), (r1, c1) in zip(cells, cells[1:]):
        step = (r1 - r0, c1 - c0)
        dirs[(r0, c0)].add(step)
        dirs[(r1, c1)].add((-step[0], -step[1]))
    return {cell: frozenset(values) for cell, values in dirs.items()}


def _straight_axis(dirs: frozenset[tuple[int, int]]) -> str | None:
    if dirs == frozenset({(0, -1), (0, 1)}):
        return "h"
    if dirs == frozenset({(-1, 0), (1, 0)}):
        return "v"
    return None


def _hidden_endpoint_ok(
    shape: tuple[int, int],
    cell: tuple[int, int],
    straight_dirs: frozenset[tuple[int, int]],
    endpoint_dirs: frozenset[tuple[int, int]],
) -> bool:
    axis = _straight_axis(straight_dirs)
    if axis is None or len(endpoint_dirs) != 1 or not _is_edge(shape, cell[0], cell[1]):
        return False

    (dr, dc) = next(iter(endpoint_dirs))
    r, c = cell
    h, w = shape

    if r == 0:
        return axis == "h" and (dr, dc) == (1, 0)
    if r == h - 1:
        return axis == "h" and (dr, dc) == (-1, 0)
    if c == 0:
        return axis == "v" and (dr, dc) == (0, 1)
    assert c == w - 1
    return axis == "v" and (dr, dc) == (0, -1)


def _can_share_cell(
    shape: tuple[int, int],
    cell: tuple[int, int],
    existing_dirs: frozenset[tuple[int, int]],
    new_dirs: frozenset[tuple[int, int]],
) -> bool:
    existing_axis = _straight_axis(existing_dirs)
    new_axis = _straight_axis(new_dirs)
    if existing_axis is not None and new_axis is not None and existing_axis != new_axis:
        return True
    if _hidden_endpoint_ok(shape, cell, existing_dirs, new_dirs):
        return True
    if _hidden_endpoint_ok(shape, cell, new_dirs, existing_dirs):
        return True
    return False


def _marker_options(shape: tuple[int, int], start: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = start
    h, w = shape
    options: list[tuple[int, int]] = []
    if c == 0 or c == w - 1:
        for dr in (-1, 1):
            nr = r + dr
            if 0 <= nr < h:
                options.append((nr, c))
    elif r == 0 or r == h - 1:
        for dc in (-1, 1):
            nc = c + dc
            if 0 <= nc < w:
                options.append((r, nc))
    else:
        raise AssertionError("start must be on the border")
    return options


def _try_add_path(
    shape: tuple[int, int],
    paths: list[dict[str, object]],
    cell_info: dict[tuple[int, int], dict[str, object]],
    markers: set[tuple[int, int]],
    vertices: list[tuple[int, int]],
    color: int,
    marker: tuple[int, int],
) -> bool:
    cells = _vertices_to_cells(vertices)
    if len(cells) < 2 or len(set(cells)) != len(cells):
        return False

    local_dirs = _local_dirs(cells)
    start = cells[0]
    end = cells[-1]
    if not _is_edge(shape, start[0], start[1]) or not _is_edge(shape, end[0], end[1]):
        return False
    if marker in markers or marker in cell_info or marker in set(cells):
        return False
    if not _is_edge(shape, marker[0], marker[1]):
        return False
    if abs(marker[0] - start[0]) + abs(marker[1] - start[1]) != 1:
        return False
    if not _same_edge(shape, start, marker):
        return False
    if len(local_dirs[start]) != 1 or len(local_dirs[end]) != 1:
        return False

    for i, cell in enumerate(cells):
        if cell in markers:
            return False
        info = cell_info.get(cell)
        if info is None:
            continue
        if i == 0:
            return False
        if len(info["path_ids"]) != 1:
            return False
        if not _can_share_cell(shape, cell, info["dirs"], local_dirs[cell]):
            return False

    path_id = len(paths)
    paths.append(
        {
            "id": path_id,
            "color": color,
            "start": start,
            "end": end,
            "marker": marker,
            "cells": cells,
            "local_dirs": local_dirs,
        }
    )
    markers.add(marker)

    for cell in cells:
        info = cell_info.get(cell)
        if info is None:
            cell_info[cell] = {"dirs": local_dirs[cell], "path_ids": [path_id]}
        else:
            info["path_ids"].append(path_id)
            info["dirs"] = frozenset(set(info["dirs"]) | set(local_dirs[cell]))
    return True


def _choose_marker(
    shape: tuple[int, int],
    start: tuple[int, int],
    markers: set[tuple[int, int]],
    cell_info: dict[tuple[int, int], dict[str, object]],
) -> tuple[int, int] | None:
    options = [cell for cell in _marker_options(shape, start) if cell not in markers and cell not in cell_info]
    if not options:
        return None
    return random.choice(options)


def _build_core_paths(
    shape: tuple[int, int],
    paths: list[dict[str, object]],
    cell_info: dict[tuple[int, int], dict[str, object]],
    markers: set[tuple[int, int]],
    colors: list[int],
) -> tuple[int, int] | None:
    h, w = shape
    for _ in range(200):
        c_up = random.randint(2, max(2, w // 4))
        c_down_low = max(c_up + 5, w // 2)
        c_down_high = w - 5
        if c_down_low > c_down_high:
            continue
        c_down = random.randint(c_down_low, c_down_high)
        c_hidden = random.randint(c_up + 1, c_down - 1)

        ra0 = random.randint(2, max(2, h // 2 - 2))
        ra2_low = max(ra0 + 3, h // 2)
        ra2_high = h - 4
        if ra2_low > ra2_high:
            continue
        ra2 = random.randint(ra2_low, ra2_high)
        if ra2 - 2 < 2:
            continue
        rb_mid = random.randint(2, ra2 - 2)

        cb0_low = c_down + 2
        cb0_high = w - 2
        if cb0_low > cb0_high:
            continue
        cb0 = random.randint(cb0_low, cb0_high)

        a_vertices = [(ra0, 0), (ra0, c_up), (0, c_up), (0, c_down), (ra2, c_down), (ra2, w - 1)]
        b_vertices = [(h - 1, cb0), (rb_mid, cb0), (rb_mid, c_hidden), (0, c_hidden)]

        marker_a = _choose_marker(shape, a_vertices[0], markers, cell_info)
        if marker_a is None:
            continue
        if not _try_add_path(shape, paths, cell_info, markers, a_vertices, colors[0], marker_a):
            continue

        marker_b = _choose_marker(shape, b_vertices[0], markers, cell_info)
        if marker_b is None or not _try_add_path(shape, paths, cell_info, markers, b_vertices, colors[1], marker_b):
            paths.pop()
            markers.remove(marker_a)
            for cell in _vertices_to_cells(a_vertices):
                info = cell_info[cell]
                if info["path_ids"] == [0]:
                    del cell_info[cell]
            continue

        return (0, c_hidden)
    return None


def _sample_extra_vertices(shape: tuple[int, int]) -> list[tuple[int, int]] | None:
    h, w = shape
    family = random.choice(["lr2", "lr4", "tb2", "tb4"])
    if family == "lr2":
        r0 = random.randint(1, h - 2)
        r1 = random.randint(1, h - 2)
        if r0 == r1:
            return None
        c1 = random.randint(2, w - 3)
        return [(r0, 0), (r0, c1), (r1, c1), (r1, w - 1)]
    if family == "lr4":
        rows = random.sample(range(1, h - 1), 3)
        c1 = random.randint(2, w - 5)
        c2 = random.randint(c1 + 2, w - 2)
        return [(rows[0], 0), (rows[0], c1), (rows[1], c1), (rows[1], c2), (rows[2], c2), (rows[2], w - 1)]
    if family == "tb2":
        c0 = random.randint(1, w - 2)
        c1 = random.randint(1, w - 2)
        if c0 == c1:
            return None
        r1 = random.randint(2, h - 3)
        return [(0, c0), (r1, c0), (r1, c1), (h - 1, c1)]

    cols = random.sample(range(1, w - 1), 3)
    r1 = random.randint(2, h - 5)
    r2 = random.randint(r1 + 2, h - 3)
    return [(0, cols[0]), (r1, cols[0]), (r1, cols[1]), (r2, cols[1]), (r2, cols[2]), (h - 1, cols[2])]


def _try_add_extra_path(
    shape: tuple[int, int],
    paths: list[dict[str, object]],
    cell_info: dict[tuple[int, int], dict[str, object]],
    markers: set[tuple[int, int]],
    color: int,
) -> bool:
    for _ in range(120):
        vertices = _sample_extra_vertices(shape)
        if vertices is None:
            continue
        marker = _choose_marker(shape, vertices[0], markers, cell_info)
        if marker is None:
            continue
        if _try_add_path(shape, paths, cell_info, markers, vertices, color, marker):
            return True
    return False


def _render_case(
    shape: tuple[int, int],
    paths: list[dict[str, object]],
    cell_info: dict[tuple[int, int], dict[str, object]],
    hidden_cell: tuple[int, int] | None,
) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full(shape, BG, dtype=int)

    for cell, info in cell_info.items():
        path_ids = info["path_ids"]
        if hidden_cell is not None and cell == hidden_cell:
            color = int(paths[0]["color"])
        elif len(path_ids) == 1:
            color = int(paths[path_ids[0]]["color"])
        else:
            color = int(paths[random.choice(path_ids)]["color"])
        inp[cell] = color

    for path in paths:
        marker = path["marker"]
        inp[marker] = GREY

    out = inp.copy()
    out[inp == GREY] = BG
    for path in paths:
        end = path["end"]
        out[end] = int(path["color"])
    return inp, out


def _symmetry_ops() -> list[Callable[[np.ndarray], np.ndarray]]:
    return [
        lambda g: g.copy(),
        lambda g: np.rot90(g, 1).copy(),
        lambda g: np.rot90(g, 2).copy(),
        lambda g: np.rot90(g, 3).copy(),
        lambda g: np.fliplr(g).copy(),
        lambda g: np.flipud(g).copy(),
        lambda g: np.transpose(g).copy(),
        lambda g: np.fliplr(np.transpose(g)).copy(),
    ]


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(16, 24)
    w = random.randint(16, 24)
    shape = (h, w)

    paths: list[dict[str, object]] = []
    cell_info: dict[tuple[int, int], dict[str, object]] = {}
    markers: set[tuple[int, int]] = set()
    colors = random.sample(PATH_PALETTE, 4)

    hidden_cell = _build_core_paths(shape, paths, cell_info, markers, colors)
    if hidden_cell is None:
        return None

    extra_target = random.randint(0, 2)
    for extra_idx in range(extra_target):
        _try_add_extra_path(shape, paths, cell_info, markers, colors[2 + extra_idx])

    inp, expected = _render_case(shape, paths, cell_info, hidden_cell)
    transform = random.choice(_symmetry_ops())
    inp = transform(inp)
    expected = transform(expected)

    try:
        actual = main(inp)
    except Exception:
        return None

    if not np.array_equal(actual, expected):
        return None
    if np.count_nonzero(inp == GREY) != len(paths):
        return None
    if np.count_nonzero(actual != inp) < len(paths) + 1:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid signal-routing input")
