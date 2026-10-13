from __future__ import annotations

import random
from collections import Counter

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
TEAL = 8
MAROON = 9

EDGE_COLOR_POOL = (BLUE, RED, GREEN, PINK, ORANGE, TEAL, MAROON)
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

LEFT_PANEL_W = 7
SEP_W = 1
RIGHT_W_RANGE = (13, 17)
HEIGHT_RANGE = (16, 22)
WALL_COUNT_OPTIONS = (2, 3)
WALL_MARGIN = 3
WALL_MIN_GAP = 3
COL_MIN_GAP = 3
GEN_ATTEMPTS = 500


TEMPLATES = {
    "L3": {
        "vertices": [(0, 0), (0, 2), (2, 2)],
        "edges": [(0, 1), (1, 2)],
    },
    "T4": {
        "vertices": [(0, 2), (2, 0), (2, 2), (2, 4)],
        "edges": [(0, 2), (1, 2), (2, 3)],
    },
    "TVERT4": {
        "vertices": [(0, 2), (2, 0), (2, 2), (4, 2)],
        "edges": [(0, 2), (1, 2), (2, 3)],
    },
    "U4": {
        "vertices": [(0, 0), (2, 0), (2, 2), (0, 2)],
        "edges": [(0, 1), (1, 2), (3, 2)],
    },
}


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(arr: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return arr[r0 : r1 + 1, c0 : c1 + 1]


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _segment_cells(
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    include_start: bool,
    include_end: bool,
) -> list[tuple[int, int]]:
    r0, c0 = start
    r1, c1 = end
    if r0 != r1 and c0 != c1:
        raise AssertionError("segment must be orthogonal")
    cells: list[tuple[int, int]] = []
    if r0 == r1:
        step = 1 if c1 > c0 else -1
        for c in range(c0, c1 + step, step):
            cells.append((r0, c))
    else:
        step = 1 if r1 > r0 else -1
        for r in range(r0, r1 + step, step):
            cells.append((r, c0))
    if not include_start:
        cells = cells[1:]
    if not include_end:
        cells = cells[:-1]
    return cells


def _normalize_edge(endpoints: tuple[tuple[int, int], tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int]]:
    a, b = endpoints
    return (a, b) if a <= b else (b, a)


def _edge_sort_key(edge: dict) -> tuple[int, int, int, int]:
    cells = edge["cells"]
    top_left = min(cells)
    bottom_right = max(cells)
    return int(top_left[0]), int(top_left[1]), int(bottom_right[0]), int(bottom_right[1])


def _find_separator_col(grid: np.ndarray) -> int:
    cols = [c for c in range(grid.shape[1]) if bool(np.all(grid[:, c] == GREY))]
    assert len(cols) == 1, "expected exactly one full-height grey separator"
    return int(cols[0])


def _vertex_mask(proto: np.ndarray) -> np.ndarray:
    fg = proto != BLACK
    h, w = proto.shape
    out = np.zeros_like(fg, dtype=bool)
    for r, c in np.argwhere(fg):
        neighbors = []
        for dr, dc in DIRS4:
            rr, cc = int(r) + dr, int(c) + dc
            if 0 <= rr < h and 0 <= cc < w and fg[rr, cc]:
                neighbors.append((dr, dc))
        if len(neighbors) != 2:
            out[r, c] = True
            continue
        a, b = neighbors
        if (a[0] + b[0], a[1] + b[1]) != (0, 0):
            out[r, c] = True
    return out


def _extract_prototype(grid: np.ndarray, sep_c: int) -> tuple[np.ndarray, list[tuple[int, int]], list[dict]]:
    left = grid[:, :sep_c]
    proto_mask = (left != BLACK) & (left != GREY)
    box = _bbox(proto_mask)
    assert box is not None, "prototype panel missing"
    proto = _crop(left, box)
    assert not np.any((proto == GREY) | (proto == YELLOW))
    vertices_mask = _vertex_mask(proto)
    vertices = [tuple(map(int, rc)) for rc in np.argwhere(vertices_mask)]
    assert 3 <= len(vertices) <= 4, "expected 3-4 prototype vertices"

    fg = proto != BLACK
    vertex_set = set(vertices)
    seen: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    edges: list[dict] = []
    for vr, vc in sorted(vertices):
        for dr, dc in DIRS4:
            rr, cc = vr + dr, vc + dc
            if not _in_bounds(proto.shape, rr, cc) or not fg[rr, cc]:
                continue
            cells = [(vr, vc)]
            while _in_bounds(proto.shape, rr, cc) and fg[rr, cc]:
                cells.append((rr, cc))
                if (rr, cc) in vertex_set and (rr, cc) != (vr, vc):
                    endpoints = _normalize_edge(((vr, vc), (rr, cc)))
                    if endpoints not in seen:
                        seen.add(endpoints)
                        interior = cells[1:-1]
                        colors = {int(proto[r, c]) for r, c in interior}
                        assert len(colors) == 1, "each prototype edge must have a single visible interior color"
                        edges.append(
                            {
                                "endpoints": endpoints,
                                "cells": tuple(cells),
                                "color": int(next(iter(colors))),
                            }
                        )
                    break
                rr += dr
                cc += dc

    assert len(edges) >= 2, "prototype must contain at least two edges"
    edges.sort(key=_edge_sort_key)
    return proto, sorted(vertices), edges


def _extract_anchors(grid: np.ndarray, sep_c: int) -> list[tuple[int, int]]:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(grid[:, sep_c + 1 :] == YELLOW)]
    anchors = [(r, c + sep_c + 1) for r, c in coords]
    assert 3 <= len(anchors) <= 4, "expected 3-4 anchors"
    return sorted(anchors)


def _map_vertices_to_anchors(
    vertices: list[tuple[int, int]],
    anchors: list[tuple[int, int]],
) -> dict[tuple[int, int], tuple[int, int]]:
    vertex_rows = sorted({r for r, _ in vertices})
    vertex_cols = sorted({c for _, c in vertices})
    anchor_rows = sorted({r for r, _ in anchors})
    anchor_cols = sorted({c for _, c in anchors})
    assert len(vertex_rows) == len(anchor_rows)
    assert len(vertex_cols) == len(anchor_cols)

    vr_index = {r: i for i, r in enumerate(vertex_rows)}
    vc_index = {c: i for i, c in enumerate(vertex_cols)}
    ar_index = {r: i for i, r in enumerate(anchor_rows)}
    ac_index = {c: i for i, c in enumerate(anchor_cols)}

    expected = {(vr_index[r], vc_index[c]) for r, c in vertices}
    observed = {(ar_index[r], ac_index[c]) for r, c in anchors}
    assert expected == observed, "anchor pattern must match prototype vertices under independent row/col stretching"

    row_map = {r: anchor_rows[vr_index[r]] for r in vertex_rows}
    col_map = {c: anchor_cols[vc_index[c]] for c in vertex_cols}
    return {(r, c): (row_map[r], col_map[c]) for r, c in vertices}


def _extract_walls(grid: np.ndarray, sep_c: int) -> dict[int, int]:
    right = grid[:, sep_c + 1 :]
    width = right.shape[1]
    walls: dict[int, int] = {}
    for r in range(right.shape[0]):
        row = right[r]
        if int(np.sum(row == GREY)) != width - 3:
            continue
        door_cols = np.flatnonzero(row != GREY)
        if len(door_cols) != 3:
            continue
        if not np.array_equal(door_cols, np.arange(door_cols[0], door_cols[0] + 3)):
            continue
        if not bool(np.all(row[door_cols] == BLACK)):
            continue
        walls[int(r)] = int(sep_c + 1 + door_cols[1])
    assert 2 <= len(walls) <= 3, "expected 2-3 horizontal walls"
    return walls


def _route_cells(
    start: tuple[int, int],
    end: tuple[int, int],
    wall_doors: dict[int, int],
) -> list[tuple[int, int]]:
    sr, sc = start
    tr, tc = end
    route: list[tuple[int, int]] = []
    if sr == tr:
        return _segment_cells(start, end, include_start=False, include_end=False)

    step = 1 if tr > sr else -1
    crossed_rows = [
        wr
        for wr in sorted(wall_doors, reverse=step < 0)
        if min(sr, tr) < wr < max(sr, tr)
    ]

    cur = start
    for wr in crossed_rows:
        before_wall = (wr - step, cur[1])
        route.extend(_segment_cells(cur, before_wall, include_start=False, include_end=True))
        door_pos = (before_wall[0], wall_doors[wr])
        route.extend(_segment_cells(before_wall, door_pos, include_start=False, include_end=True))
        cross = (wr, wall_doors[wr])
        route.extend(_segment_cells(door_pos, cross, include_start=False, include_end=True))
        cur = cross

    vertical_target = (tr, cur[1])
    route.extend(
        _segment_cells(
            cur,
            vertical_target,
            include_start=False,
            include_end=(cur[1] != tc),
        )
    )
    if cur[1] != tc:
        route.extend(_segment_cells(vertical_target, end, include_start=False, include_end=False))
    return route


def _paint_route(out: np.ndarray, route: list[tuple[int, int]], color: int) -> None:
    for r, c in route:
        assert out[r, c] != GREY, "route may not paint walls"
        assert out[r, c] != YELLOW, "route may not overwrite anchors"
        out[r, c] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Map the compact prototype edges onto stretched anchors and route each edge through the maze doors.

    Input structure:
    - A black grid is split by one full-height grey separator column.
    - The left panel contains a compact orthogonal colored prototype with 3-4 graph vertices formed by endpoints, corners, or intersections.
    - The right panel contains 3-4 yellow anchor pixels and 2-3 full-width horizontal grey walls, each with a three-cell door gap.

    Rule:
    - Detect the prototype vertices and the colored straight edges between consecutive vertices.
    - Independently map sorted prototype vertex rows to sorted anchor rows, and sorted prototype vertex columns to sorted anchor columns; each prototype vertex maps to the yellow anchor at the paired stretched coordinates.
    - For every prototype edge, draw a path in the right panel between the corresponding anchors using the edge color. Move vertically toward the target until blocked by a wall, slide along that wall to the door-center pixel, cross there, and continue; within a corridor this yields the shortest orthogonal route.
    - If routed edges overlap, later edges overwrite earlier ones, where prototype edges are processed from the most top-left edge to the most bottom-right edge. Keep the separator, walls, door gaps, prototype, and yellow anchors unchanged.

    Key insight:
    - The prototype tells you which anchor pairs to connect and which colors to use, while the independently stretched anchor layout and the door-constrained walls determine the concrete maze paths.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert _most_common_color(g) == BLACK

    sep_c = _find_separator_col(g)
    _proto, vertices, edges = _extract_prototype(g, sep_c)
    anchors = _extract_anchors(g, sep_c)
    mapping = _map_vertices_to_anchors(vertices, anchors)
    wall_doors = _extract_walls(g, sep_c)

    out = g.copy()
    for edge in edges:
        a, b = edge["endpoints"]
        route = _route_cells(mapping[a], mapping[b], wall_doors)
        _paint_route(out, route, edge["color"])
    return out


def _sample_spaced_positions(
    n: int,
    lo: int,
    hi: int,
    min_gap: int,
) -> list[int] | None:
    if n == 0:
        return []
    needed = n + (n - 1) * min_gap
    span = hi - lo + 1
    if span < needed:
        return None
    compressed = span - (n - 1) * min_gap
    picks = sorted(random.sample(range(compressed), n))
    return [lo + picks[i] + i * min_gap for i in range(n)]


def _sample_wall_rows(height: int, wall_count: int) -> list[int] | None:
    return _sample_spaced_positions(
        wall_count,
        WALL_MARGIN,
        height - WALL_MARGIN - 1,
        WALL_MIN_GAP,
    )


def _corridors(height: int, wall_rows: list[int]) -> list[tuple[int, int]]:
    out = []
    start = 0
    for wr in wall_rows:
        out.append((start, wr - 1))
        start = wr + 1
    out.append((start, height - 1))
    return out


def _sample_row_from_interval(interval: tuple[int, int]) -> int:
    lo, hi = interval
    if hi - lo >= 2:
        lo += 1
        hi -= 1
    return random.randint(lo, hi)


def _choose_row_map(template_name: str, corridors: list[tuple[int, int]]) -> dict[int, int] | None:
    if template_name in ("L3", "T4", "U4"):
        chosen = [corridors[0], corridors[-1]]
        rows = sorted({r for r, _ in TEMPLATES[template_name]["vertices"]})
        sampled = [_sample_row_from_interval(interval) for interval in chosen]
        return {rows[i]: sampled[i] for i in range(len(rows))}

    assert template_name == "TVERT4"
    rows = sorted({r for r, _ in TEMPLATES[template_name]["vertices"]})
    if len(corridors) == 3:
        chosen = [corridors[0], corridors[1], corridors[2]]
    else:
        chosen = [corridors[0], random.choice([corridors[1], corridors[2]]), corridors[3]]
    sampled = [_sample_row_from_interval(interval) for interval in chosen]
    return {rows[i]: sampled[i] for i in range(len(rows))}


def _choose_col_map(template_name: str, right_w: int) -> dict[int, int] | None:
    cols = sorted({c for _, c in TEMPLATES[template_name]["vertices"]})
    sampled = _sample_spaced_positions(len(cols), 1, right_w - 2, COL_MIN_GAP)
    if sampled is None:
        return None
    return {cols[i]: sampled[i] for i in range(len(cols))}


def _edge_records(template_name: str, edge_colors: list[int]) -> list[dict]:
    vertices = TEMPLATES[template_name]["vertices"]
    records = []
    for (ai, bi), color in zip(TEMPLATES[template_name]["edges"], edge_colors):
        a = tuple(vertices[ai])
        b = tuple(vertices[bi])
        records.append(
            {
                "endpoints": _normalize_edge((a, b)),
                "color": int(color),
                "cells": tuple(_segment_cells(a, b, include_start=True, include_end=True)),
            }
        )
    records.sort(key=_edge_sort_key)
    return records


def _render_prototype(template_name: str, edge_records: list[dict]) -> np.ndarray:
    vertices = TEMPLATES[template_name]["vertices"]
    h = max(r for r, _ in vertices) + 1
    w = max(c for _, c in vertices) + 1
    proto = np.zeros((h, w), dtype=int)
    for edge in edge_records:
        for r, c in edge["cells"]:
            proto[r, c] = int(edge["color"])
    return proto


def _template_anchor_map(
    template_name: str,
    row_map: dict[int, int],
    col_map: dict[int, int],
    sep_c: int,
) -> dict[tuple[int, int], tuple[int, int]]:
    return {
        (r, c): (row_map[r], sep_c + 1 + col_map[c])
        for r, c in TEMPLATES[template_name]["vertices"]
    }


def _render_scene(
    template_name: str,
    proto: np.ndarray,
    right_h: int,
    right_w: int,
    proto_top: int,
    proto_left: int,
    wall_rows: list[int],
    door_centers_local: list[int],
    anchors_abs: list[tuple[int, int]],
) -> np.ndarray:
    total_h = right_h
    total_w = LEFT_PANEL_W + SEP_W + right_w
    grid = np.zeros((total_h, total_w), dtype=int)
    sep_c = LEFT_PANEL_W
    grid[:, sep_c] = GREY

    ph, pw = proto.shape
    grid[proto_top : proto_top + ph, proto_left : proto_left + pw] = proto

    for wr, dc_local in zip(wall_rows, door_centers_local):
        c0 = sep_c + 1
        c1 = total_w - 1
        grid[wr, c0 : c1 + 1] = GREY
        dc_abs = sep_c + 1 + dc_local
        grid[wr, dc_abs - 1 : dc_abs + 2] = BLACK

    for r, c in anchors_abs:
        grid[r, c] = YELLOW
    return grid


def _intended_output(
    inp: np.ndarray,
    template_name: str,
    edge_records: list[dict],
    mapping_abs: dict[tuple[int, int], tuple[int, int]],
    wall_doors_abs: dict[int, int],
) -> tuple[np.ndarray, list[list[tuple[int, int]]]]:
    out = inp.copy()
    routes: list[list[tuple[int, int]]] = []
    for edge in sorted(edge_records, key=_edge_sort_key):
        a, b = edge["endpoints"]
        route = _route_cells(mapping_abs[a], mapping_abs[b], wall_doors_abs)
        _paint_route(out, route, edge["color"])
        routes.append(route)
    return out, routes


def _edge_geometry_signature(edges: list[dict]) -> set[tuple[tuple[int, int], tuple[int, int], int]]:
    return {
        (edge["endpoints"][0], edge["endpoints"][1], int(edge["color"]))
        for edge in edges
    }


def _has_horizontal_slide(
    mapping_abs: dict[tuple[int, int], tuple[int, int]],
    edge_records: list[dict],
    wall_doors_abs: dict[int, int],
) -> bool:
    for edge in edge_records:
        a, b = edge["endpoints"]
        start = mapping_abs[a]
        end = mapping_abs[b]
        if start[0] == end[0]:
            continue
        step = 1 if end[0] > start[0] else -1
        cur_col = start[1]
        crossed_rows = [
            wr
            for wr in sorted(wall_doors_abs, reverse=step < 0)
            if min(start[0], end[0]) < wr < max(start[0], end[0])
        ]
        for wr in crossed_rows:
            if wall_doors_abs[wr] != cur_col:
                return True
            cur_col = wall_doors_abs[wr]
    return False


def _validate_scene(
    inp: np.ndarray,
    expected_out: np.ndarray,
    template_name: str,
    edge_records: list[dict],
    anchors_abs: list[tuple[int, int]],
    wall_doors_abs: dict[int, int],
    mapping_abs: dict[tuple[int, int], tuple[int, int]],
    routes: list[list[tuple[int, int]]],
    overlap_expected: bool,
) -> bool:
    try:
        produced = main(inp)
    except Exception:
        return False
    if not np.array_equal(produced, expected_out):
        return False

    sep_c = _find_separator_col(inp)
    _proto, parsed_vertices, parsed_edges = _extract_prototype(inp, sep_c)
    parsed_anchors = _extract_anchors(inp, sep_c)
    parsed_walls = _extract_walls(inp, sep_c)
    parsed_mapping = _map_vertices_to_anchors(parsed_vertices, parsed_anchors)

    if sorted(parsed_anchors) != sorted(anchors_abs):
        return False
    if parsed_walls != wall_doors_abs:
        return False
    if _edge_geometry_signature(parsed_edges) != _edge_geometry_signature(edge_records):
        return False
    if parsed_mapping != mapping_abs:
        return False

    if len(parsed_edges) != len(edge_records):
        return False
    if not _has_horizontal_slide(mapping_abs, edge_records, wall_doors_abs):
        return False
    if sum(1 for edge in edge_records if edge["endpoints"][0][0] == edge["endpoints"][1][0]) < 1:
        return False
    if sum(1 for edge in edge_records if edge["endpoints"][0][1] == edge["endpoints"][1][1]) < 1:
        return False

    overlap_counter = Counter(cell for route in routes for cell in route)
    overlap_cells = {cell for cell, count in overlap_counter.items() if count >= 2}
    if overlap_expected != bool(overlap_cells):
        return False

    unrelated_anchor_hits = set(anchors_abs)
    for edge, route in zip(edge_records, routes):
        a, b = edge["endpoints"]
        allowed = {mapping_abs[a], mapping_abs[b]}
        if any(cell in unrelated_anchor_hits - allowed for cell in route):
            return False

    painted = int(np.sum(expected_out != inp))
    if painted < 6:
        return False
    if painted >= (inp.shape[0] * (inp.shape[1] - LEFT_PANEL_W - 1)) // 2:
        return False

    sig = hashlib_like(expected_out)
    if sig == hashlib_like(inp):
        return False
    return True


def hashlib_like(grid: np.ndarray) -> tuple[int, tuple[int, int]]:
    return int(np.sum(grid * np.arange(1, grid.size + 1).reshape(grid.shape))), tuple(grid.shape)


def _generate_attempt() -> np.ndarray | None:
    overlap_expected = random.random() < 0.45
    if overlap_expected:
        template_name = "U4"
    else:
        template_name = random.choice(["L3", "T4", "TVERT4"])

    right_h = random.randint(*HEIGHT_RANGE)
    right_w = random.randint(*RIGHT_W_RANGE)
    wall_count = random.choice(WALL_COUNT_OPTIONS)
    wall_rows = _sample_wall_rows(right_h, wall_count)
    if wall_rows is None:
        return None
    corridors = _corridors(right_h, wall_rows)
    if any(lo > hi for lo, hi in corridors):
        return None

    if template_name == "TVERT4" and len(corridors) < 3:
        return None

    row_map = _choose_row_map(template_name, corridors)
    col_map = _choose_col_map(template_name, right_w)
    if row_map is None or col_map is None:
        return None

    edge_colors = random.sample(EDGE_COLOR_POOL, len(TEMPLATES[template_name]["edges"]))
    edge_records = _edge_records(template_name, edge_colors)
    proto = _render_prototype(template_name, edge_records)

    ph, pw = proto.shape
    if ph > right_h or pw > LEFT_PANEL_W:
        return None
    proto_top = random.randint(0, right_h - ph)
    proto_left = random.randint(0, LEFT_PANEL_W - pw)

    col_values = list(col_map.values())
    door_centers_local = []
    for _wr in wall_rows:
        door = random.randint(1, right_w - 2)
        door_centers_local.append(door)
    if not any(dc not in col_values for dc in door_centers_local):
        return None

    sep_c = LEFT_PANEL_W
    mapping_abs = _template_anchor_map(template_name, row_map, col_map, sep_c)
    anchors_abs = sorted(mapping_abs.values())
    wall_doors_abs = {wr: sep_c + 1 + dc for wr, dc in zip(wall_rows, door_centers_local)}

    inp = _render_scene(
        template_name,
        proto,
        right_h,
        right_w,
        proto_top,
        proto_left,
        wall_rows,
        door_centers_local,
        anchors_abs,
    )
    try:
        expected_out, routes = _intended_output(inp, template_name, edge_records, mapping_abs, wall_doors_abs)
    except AssertionError:
        return None

    if not _validate_scene(
        inp,
        expected_out,
        template_name,
        edge_records,
        anchors_abs,
        wall_doors_abs,
        mapping_abs,
        routes,
        overlap_expected,
    ):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
