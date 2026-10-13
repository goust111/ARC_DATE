from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ARC_UTILS = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS) not in sys.path:
    sys.path.append(str(ARC_UTILS))

from arc_utils import Color, find_components, get_bg


Frame = tuple[int, int, int, int]


def _rect_sort_key(rect: Frame) -> tuple[int, int, int, int]:
    return rect


def _is_hollow_frame(g: np.ndarray, color: int, rect: Frame) -> bool:
    r0, c0, r1, c1 = rect
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 5 or w < 5:
        return False
    patch = g[r0 : r1 + 1, c0 : c1 + 1]
    frame = np.zeros((h, w), dtype=bool)
    frame[0, :] = True
    frame[-1, :] = True
    frame[:, 0] = True
    frame[:, -1] = True
    return np.array_equal(patch == color, frame)


def _extract_objects(g: np.ndarray) -> tuple[int, int | None, list[Frame], list[set[tuple[int, int]]]]:
    bg = get_bg(g)
    colors = sorted(int(v) for v in np.unique(g) if int(v) != bg)
    assert len(colors) >= 1

    frame_candidates: list[tuple[int, list[Frame]]] = []
    for color in colors:
        comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, colors=[color])
        rects: list[Frame] = []
        ok = True
        for comp in comps:
            rect = tuple(int(x) for x in comp["bbox"])
            if not _is_hollow_frame(g, color, rect):
                ok = False
                break
            rects.append(rect)
        if ok and 3 <= len(rects) <= 5:
            frame_candidates.append((color, sorted(rects, key=_rect_sort_key)))

    assert len(frame_candidates) == 1
    frame_color, rects = frame_candidates[0]

    obstacle_colors = [c for c in colors if c != frame_color]
    if not obstacle_colors:
        return frame_color, None, rects, []
    assert len(obstacle_colors) == 1
    obstacle_color = obstacle_colors[0]
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, colors=[obstacle_color])
    obstacle_sets = [set((int(r), int(c)) for r, c in comp["coords"]) for comp in comps]
    return frame_color, obstacle_color, rects, obstacle_sets


def _edge_obstacles(
    obstacle_sets: list[set[tuple[int, int]]],
    r0: int,
    r1: int,
    c0: int,
    c1: int,
) -> list[set[tuple[int, int]]]:
    hits: list[set[tuple[int, int]]] = []
    for comp in obstacle_sets:
        if any(r0 <= r <= r1 and c0 <= c <= c1 for r, c in comp):
            if all(r0 <= r <= r1 and c0 <= c <= c1 for r, c in comp):
                hits.append(comp)
            else:
                raise AssertionError("obstacle may not partially overlap a candidate strip")
    return hits


def _candidate_edges(
    g: np.ndarray,
    bg: int,
    rects: list[Frame],
    obstacle_sets: list[set[tuple[int, int]]],
) -> list[dict]:
    edges: list[dict] = []
    h, w = g.shape
    for i, a in enumerate(rects):
        for j in range(i + 1, len(rects)):
            b = rects[j]

            if a[3] < b[1] or b[3] < a[1]:
                if a[3] < b[1]:
                    left_i, right_i, left, right = i, j, a, b
                else:
                    left_i, right_i, left, right = j, i, b, a
                r0 = max(left[0] + 1, right[0] + 1)
                r1 = min(left[2] - 1, right[2] - 1)
                c0 = left[3] + 1
                c1 = right[1] - 1
                if r0 <= r1 and c0 <= c1:
                    strip = g[r0 : r1 + 1, c0 : c1 + 1]
                    if np.all((strip == bg) | (strip == Color.RED)):
                        obs = _edge_obstacles(obstacle_sets, r0, r1, c0, c1)
                        if len(obs) <= 1:
                            edges.append(
                                {
                                    "kind": "h",
                                    "a": left_i,
                                    "b": right_i,
                                    "r0": r0,
                                    "r1": r1,
                                    "c0": c0,
                                    "c1": c1,
                                    "gap": c1 - c0 + 1,
                                    "overlap": r1 - r0 + 1,
                                    "obstacle": obs[0] if obs else None,
                                }
                            )

            if a[2] < b[0] or b[2] < a[0]:
                if a[2] < b[0]:
                    upper_i, lower_i, upper, lower = i, j, a, b
                else:
                    upper_i, lower_i, upper, lower = j, i, b, a
                c0 = max(upper[1] + 1, lower[1] + 1)
                c1 = min(upper[3] - 1, lower[3] - 1)
                r0 = upper[2] + 1
                r1 = lower[0] - 1
                if c0 <= c1 and r0 <= r1:
                    strip = g[r0 : r1 + 1, c0 : c1 + 1]
                    if np.all((strip == bg) | (strip == Color.RED)):
                        obs = _edge_obstacles(obstacle_sets, r0, r1, c0, c1)
                        if len(obs) <= 1:
                            edges.append(
                                {
                                    "kind": "v",
                                    "a": upper_i,
                                    "b": lower_i,
                                    "r0": r0,
                                    "r1": r1,
                                    "c0": c0,
                                    "c1": c1,
                                    "gap": r1 - r0 + 1,
                                    "overlap": c1 - c0 + 1,
                                    "obstacle": obs[0] if obs else None,
                                }
                            )
    return edges


def _push_path(path: list[tuple[int, int]], cells: list[tuple[int, int]]) -> None:
    for cell in cells:
        if not path or path[-1] != cell:
            path.append(cell)


def _horizontal_segment(row: int, c0: int, c1: int) -> list[tuple[int, int]]:
    step = 1 if c1 >= c0 else -1
    return [(row, c) for c in range(c0, c1 + step, step)]


def _vertical_segment(col: int, r0: int, r1: int) -> list[tuple[int, int]]:
    step = 1 if r1 >= r0 else -1
    return [(r, col) for r in range(r0, r1 + step, step)]


def _route_horizontal(
    row: int,
    c0: int,
    c1: int,
    obstacle: set[tuple[int, int]] | None,
    side: str | None,
) -> list[tuple[int, int]]:
    if obstacle is None or side is None:
        return _horizontal_segment(row, c0, c1)

    cols = sorted({c for _, c in obstacle})
    cmin, cmax = cols[0], cols[-1]
    assert c0 <= cmin - 1 and cmax + 1 <= c1

    top_profile = {c: min(r for r, cc in obstacle if cc == c) for c in cols}
    bottom_profile = {c: max(r for r, cc in obstacle if cc == c) for c in cols}

    path: list[tuple[int, int]] = []
    entry_col = cmin - 1
    exit_col = cmax + 1
    _push_path(path, _horizontal_segment(row, c0, entry_col))

    if side == "top":
        target = top_profile[cmin] - 1
        _push_path(path, _vertical_segment(entry_col, path[-1][0], target))
        cur_r, cur_c = path[-1]
        for c in range(cmin, cmax + 1):
            target = top_profile[c] - 1
            if target < cur_r:
                _push_path(path, _vertical_segment(cur_c, cur_r, target))
                cur_r = target
            _push_path(path, _horizontal_segment(cur_r, cur_c, c))
            cur_c = c
            if target > cur_r:
                _push_path(path, _vertical_segment(cur_c, cur_r, target))
                cur_r = target
    else:
        assert side == "bottom"
        target = bottom_profile[cmin] + 1
        _push_path(path, _vertical_segment(entry_col, path[-1][0], target))
        cur_r, cur_c = path[-1]
        for c in range(cmin, cmax + 1):
            target = bottom_profile[c] + 1
            if target > cur_r:
                _push_path(path, _vertical_segment(cur_c, cur_r, target))
                cur_r = target
            _push_path(path, _horizontal_segment(cur_r, cur_c, c))
            cur_c = c
            if target < cur_r:
                _push_path(path, _vertical_segment(cur_c, cur_r, target))
                cur_r = target

    _push_path(path, _horizontal_segment(path[-1][0], path[-1][1], exit_col))
    _push_path(path, _vertical_segment(exit_col, path[-1][0], row))
    _push_path(path, _horizontal_segment(row, exit_col, c1))
    return path


def _transpose_path(path: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return [(c, r) for r, c in path]


def _transpose_obstacle(obstacle: set[tuple[int, int]] | None) -> set[tuple[int, int]] | None:
    if obstacle is None:
        return None
    return {(c, r) for r, c in obstacle}


def _route_vertical(
    col: int,
    r0: int,
    r1: int,
    obstacle: set[tuple[int, int]] | None,
    side: str | None,
) -> list[tuple[int, int]]:
    side_map = {"left": "top", "right": "bottom", None: None}
    transposed = _route_horizontal(col, r0, r1, _transpose_obstacle(obstacle), side_map[side])
    return _transpose_path(transposed)


def _validate_route_cells(g: np.ndarray, frame_color: int, route: list[tuple[int, int]], obstacle: set[tuple[int, int]] | None) -> None:
    h, w = g.shape
    obs = obstacle or set()
    for r, c in route:
        assert 0 <= r < h and 0 <= c < w
        assert (r, c) not in obs
        assert int(g[r, c]) != frame_color


def _edge_tiebreak(rects: list[Frame], edge: dict) -> tuple:
    a = rects[edge["a"]]
    b = rects[edge["b"]]
    lo, hi = sorted((a, b), key=_rect_sort_key)
    return lo + hi


def _annotate_edge(g: np.ndarray, frame_color: int, rects: list[Frame], edge: dict) -> dict:
    annotated = dict(edge)
    if edge["kind"] == "h":
        top_row = edge["r0"]
        bottom_row = edge["r1"]
        if edge["obstacle"] is None:
            top_route = _route_horizontal(top_row, edge["c0"], edge["c1"], None, None)
            bottom_route = _route_horizontal(bottom_row, edge["c0"], edge["c1"], None, None)
            side = None
            effective = edge["gap"]
        else:
            top_route = _route_horizontal(top_row, edge["c0"], edge["c1"], edge["obstacle"], "top")
            bottom_route = _route_horizontal(bottom_row, edge["c0"], edge["c1"], edge["obstacle"], "bottom")
            if len(top_route) <= len(bottom_route):
                side = "top"
                effective = len(top_route)
            else:
                side = "bottom"
                effective = len(bottom_route)
            top_route = _route_horizontal(top_row, edge["c0"], edge["c1"], edge["obstacle"], side)
            bottom_route = _route_horizontal(bottom_row, edge["c0"], edge["c1"], edge["obstacle"], side)
        _validate_route_cells(g, frame_color, top_route, edge["obstacle"])
        _validate_route_cells(g, frame_color, bottom_route, edge["obstacle"])
        annotated["routes"] = [top_route] if top_row == bottom_row else [top_route, bottom_route]
        annotated["side"] = side
        annotated["effective"] = effective
    else:
        left_col = edge["c0"]
        right_col = edge["c1"]
        if edge["obstacle"] is None:
            left_route = _route_vertical(left_col, edge["r0"], edge["r1"], None, None)
            right_route = _route_vertical(right_col, edge["r0"], edge["r1"], None, None)
            side = None
            effective = edge["gap"]
        else:
            left_route = _route_vertical(left_col, edge["r0"], edge["r1"], edge["obstacle"], "left")
            right_route = _route_vertical(right_col, edge["r0"], edge["r1"], edge["obstacle"], "right")
            if len(left_route) <= len(right_route):
                side = "left"
                effective = len(left_route)
            else:
                side = "right"
                effective = len(right_route)
            left_route = _route_vertical(left_col, edge["r0"], edge["r1"], edge["obstacle"], side)
            right_route = _route_vertical(right_col, edge["r0"], edge["r1"], edge["obstacle"], side)
        _validate_route_cells(g, frame_color, left_route, edge["obstacle"])
        _validate_route_cells(g, frame_color, right_route, edge["obstacle"])
        annotated["routes"] = [left_route] if left_col == right_col else [left_route, right_route]
        annotated["side"] = side
        annotated["effective"] = effective
    annotated["tie"] = _edge_tiebreak(rects, edge)
    return annotated


def _spanning_edges(edges: list[dict], rects: list[Frame], *, effective: bool) -> list[dict]:
    parent = list(range(len(rects)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        assert ra != rb
        parent[rb] = ra

    chosen: list[dict] = []
    weight_key = "effective" if effective else "gap"
    for edge in sorted(edges, key=lambda e: (e[weight_key], e["gap"], e["tie"])):
        if find(edge["a"]) == find(edge["b"]):
            continue
        union(edge["a"], edge["b"])
        chosen.append(edge)
        if len(chosen) == len(rects) - 1:
            break

    assert len(chosen) == len(rects) - 1
    root = find(0)
    assert all(find(i) == root for i in range(len(rects)))
    return chosen


def _apply_edge(out: np.ndarray, frame_color: int, rects: list[Frame], edge: dict) -> None:
    for route in edge["routes"]:
        for r, c in route:
            if int(out[r, c]) != Color.RED:
                out[r, c] = frame_color

    if edge["kind"] == "h":
        left = rects[edge["a"]]
        right = rects[edge["b"]]
        if edge["r0"] + 1 <= edge["r1"] - 1:
            out[edge["r0"] + 1 : edge["r1"], left[3]] = Color.BLACK
            out[edge["r0"] + 1 : edge["r1"], right[1]] = Color.BLACK
    else:
        upper = rects[edge["a"]]
        lower = rects[edge["b"]]
        if edge["c0"] + 1 <= edge["c1"] - 1:
            out[upper[2], edge["c0"] + 1 : edge["c1"]] = Color.BLACK
            out[lower[0], edge["c0"] + 1 : edge["c1"]] = Color.BLACK


def main(grid: np.ndarray) -> np.ndarray:
    """Connect hollow blue frames by a minimum corridor tree whose edge weights include obstacle detours.

    Input structure:
    - The background is black, with 3 to 5 disconnected blue hollow rectangular frames and 1 to 3 red obstacle sprites.
    - A candidate gap is a horizontal or vertical strip between two frames whose interior spans overlap and whose strip contains only background or one red obstacle.

    Rule:
    - For every candidate gap, compute its straight wall-to-wall distance; if one obstacle lies in the strip, replace that with the shorter routed boundary-line length obtained by detouring above/below or left/right while hugging the obstacle outline.
    - Build a minimum spanning tree on the frames using routed effective length, then straight distance, then top-left frame position as tie-breakers.
    - For each selected edge, draw blue corridor boundary lines along the overlap-span extremes using the same routed side choice, erase the facing wall pixels strictly between those boundary lines, and keep red obstacle pixels unchanged.

    Key insight:
    - The obstacle changes both geometry and graph topology: the shortest visible gap is not always the cheapest tree edge once rerouting cost is included.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    assert bg == Color.BLACK
    frame_color, _obstacle_color, rects, obstacle_sets = _extract_objects(g)
    edges = [_annotate_edge(g, frame_color, rects, edge) for edge in _candidate_edges(g, bg, rects, obstacle_sets)]
    assert len(edges) >= len(rects)
    chosen = _spanning_edges(edges, rects, effective=True)

    out = g.copy()
    for edge in chosen:
        _apply_edge(out, frame_color, rects, edge)

    comps = find_components(out, bg=bg, connectivity=4, monochromatic=True, colors=[frame_color])
    assert len(comps) == 1
    if obstacle_sets:
        for obs in obstacle_sets:
            for r, c in obs:
                assert int(out[r, c]) == Color.RED
    return out


GEN_ROOM_MIN = 5
GEN_ROOM_MAX = 7
GEN_SMALL_GAP_MIN = 4
GEN_SMALL_GAP_MAX = 5
GEN_LARGE_GAP_MIN = 6
GEN_LARGE_GAP_MAX = 7
GEN_MARGIN_MIN = 1
GEN_MARGIN_MAX = 3
GEN_CANVAS_MAX = 30
GEN_RETRIES = 400


def _draw_frame(grid: np.ndarray, rect: Frame) -> None:
    r0, c0, r1, c1 = rect
    grid[r0, c0 : c1 + 1] = Color.BLUE
    grid[r1, c0 : c1 + 1] = Color.BLUE
    grid[r0 : r1 + 1, c0] = Color.BLUE
    grid[r0 : r1 + 1, c1] = Color.BLUE


def _shift_rect(rect: Frame, dr: int, dc: int) -> Frame:
    r0, c0, r1, c1 = rect
    return r0 + dr, c0 + dc, r1 + dr, c1 + dc


def _base_layout() -> tuple[list[Frame], str]:
    h_top = random.randint(GEN_ROOM_MIN, GEN_ROOM_MAX)
    h_bottom = random.randint(GEN_ROOM_MIN, GEN_ROOM_MAX)
    w_left = random.randint(GEN_ROOM_MIN, GEN_ROOM_MAX)
    w_right = random.randint(GEN_ROOM_MIN, GEN_ROOM_MAX)
    obstruct_horizontal = random.random() < 0.5
    if obstruct_horizontal:
        gap_h = random.randint(GEN_SMALL_GAP_MIN, GEN_SMALL_GAP_MAX)
        gap_v = random.randint(GEN_LARGE_GAP_MIN, GEN_LARGE_GAP_MAX)
        primary_kind = "h"
    else:
        gap_h = random.randint(GEN_LARGE_GAP_MIN, GEN_LARGE_GAP_MAX)
        gap_v = random.randint(GEN_SMALL_GAP_MIN, GEN_SMALL_GAP_MAX)
        primary_kind = "v"

    a = (0, 0, h_top - 1, w_left - 1)
    b = (0, w_left + gap_h, h_top - 1, w_left + gap_h + w_right - 1)
    c = (h_top + gap_v, 0, h_top + gap_v + h_bottom - 1, w_left - 1)
    d = (
        h_top + gap_v,
        w_left + gap_h,
        h_top + gap_v + h_bottom - 1,
        w_left + gap_h + w_right - 1,
    )
    return [a, b, c, d], primary_kind


def _attach_leaf(rects: list[Frame]) -> list[Frame] | None:
    base = list(rects)
    choices: list[Frame] = []
    a, b, c, d = base[:4]

    leaf_h = random.randint(GEN_ROOM_MIN, GEN_ROOM_MAX)
    leaf_w = random.randint(GEN_ROOM_MIN, GEN_ROOM_MAX)
    gap = random.randint(4, 6)

    # Above top row, aligned to A or B.
    for parent in (a, b):
        r0 = parent[0] - gap - leaf_h
        c0 = parent[1]
        choices.append((r0, c0, r0 + leaf_h - 1, c0 + leaf_w - 1))
    # Below bottom row, aligned to C or D.
    for parent in (c, d):
        r0 = parent[2] + gap + 1
        c0 = parent[1]
        choices.append((r0, c0, r0 + leaf_h - 1, c0 + leaf_w - 1))
    # Left of left column, aligned to A or C.
    for parent in (a, c):
        r0 = parent[0]
        c0 = parent[1] - gap - leaf_w
        choices.append((r0, c0, r0 + leaf_h - 1, c0 + leaf_w - 1))
    # Right of right column, aligned to B or D.
    for parent in (b, d):
        r0 = parent[0]
        c0 = parent[3] + gap + 1
        choices.append((r0, c0, r0 + leaf_h - 1, c0 + leaf_w - 1))

    random.shuffle(choices)
    for leaf in choices:
        good = True
        lr0, lc0, lr1, lc1 = leaf
        if lr0 < 0 or lc0 < 0:
            continue
        for rect in base:
            r0, c0, r1, c1 = rect
            rows_overlap = not (lr1 < r0 or r1 < lr0)
            cols_overlap = not (lc1 < c0 or c1 < lc0)
            if rows_overlap and cols_overlap:
                good = False
                break
        if good:
            return base + [leaf]
    return None


def _bounds(rects: list[Frame]) -> Frame:
    return (
        min(r0 for r0, _, _, _ in rects),
        min(c0 for _, c0, _, _ in rects),
        max(r1 for _, _, r1, _ in rects),
        max(c1 for _, _, _, c1 in rects),
    )


def _place_on_canvas(rects: list[Frame]) -> list[Frame] | None:
    min_r, min_c, max_r, max_c = _bounds(rects)
    height = max_r - min_r + 1
    width = max_c - min_c + 1
    if height > GEN_CANVAS_MAX or width > GEN_CANVAS_MAX:
        return None
    top_margin = random.randint(GEN_MARGIN_MIN, GEN_MARGIN_MAX)
    left_margin = random.randint(GEN_MARGIN_MIN, GEN_MARGIN_MAX)
    bottom_margin = random.randint(GEN_MARGIN_MIN, GEN_MARGIN_MAX)
    right_margin = random.randint(GEN_MARGIN_MIN, GEN_MARGIN_MAX)
    canvas_h = height + top_margin + bottom_margin
    canvas_w = width + left_margin + right_margin
    if canvas_h > GEN_CANVAS_MAX or canvas_w > GEN_CANVAS_MAX:
        return None
    dr = top_margin - min_r
    dc = left_margin - min_c
    return [_shift_rect(rect, dr, dc) for rect in rects]


def _edge_id(edge: dict) -> tuple[int, int, str]:
    return edge["a"], edge["b"], edge["kind"]


def _zigzag_horizontal(edge: dict) -> set[tuple[int, int]]:
    rows = list(range(edge["r0"], edge["r1"] + 1))
    col = random.randint(edge["c0"] + 1, edge["c1"] - 2)
    mirror = random.random() < 0.5
    coords: set[tuple[int, int]] = set()
    if len(rows) == 3:
        pattern = [(rows[0], col), (rows[1], col), (rows[1], col + 1), (rows[2], col + 1)]
    elif len(rows) == 4:
        pattern = [
            (rows[0], col),
            (rows[1], col),
            (rows[1], col + 1),
            (rows[2], col + 1),
            (rows[3], col + 1),
        ]
    else:
        pattern = [
            (rows[0], col),
            (rows[1], col),
            (rows[1], col + 1),
            (rows[2], col + 1),
            (rows[3], col + 1),
            (rows[4], col + 1),
        ]
    if mirror:
        pattern = [(r, edge["c0"] + edge["c1"] - c) for r, c in pattern]
    coords.update(pattern)
    return coords


def _zigzag_vertical(edge: dict) -> set[tuple[int, int]]:
    cols = list(range(edge["c0"], edge["c1"] + 1))
    row = random.randint(edge["r0"] + 1, edge["r1"] - 2)
    mirror = random.random() < 0.5
    coords: set[tuple[int, int]] = set()
    if len(cols) == 3:
        pattern = [(row, cols[0]), (row, cols[1]), (row + 1, cols[1]), (row + 1, cols[2])]
    elif len(cols) == 4:
        pattern = [
            (row, cols[0]),
            (row, cols[1]),
            (row + 1, cols[1]),
            (row + 1, cols[2]),
            (row + 1, cols[3]),
        ]
    else:
        pattern = [
            (row, cols[0]),
            (row, cols[1]),
            (row + 1, cols[1]),
            (row + 1, cols[2]),
            (row + 1, cols[3]),
            (row + 1, cols[4]),
        ]
    if mirror:
        pattern = [(edge["r0"] + edge["r1"] - r, c) for r, c in pattern]
    coords.update(pattern)
    return coords


def _place_obstacles(base_grid: np.ndarray, edges: list[dict], primary_kind: str) -> np.ndarray | None:
    grid = base_grid.copy()
    candidates = [edge for edge in edges if edge["kind"] == primary_kind and edge["overlap"] >= 3 and edge["gap"] >= 4]
    if not candidates:
        return None
    primary = random.choice(candidates)
    primary_obs = _zigzag_horizontal(primary) if primary["kind"] == "h" else _zigzag_vertical(primary)
    for r, c in primary_obs:
        grid[r, c] = Color.RED

    extra_pool = [edge for edge in edges if _edge_id(edge) != _edge_id(primary) and edge["overlap"] >= 3 and edge["gap"] >= 4]
    random.shuffle(extra_pool)
    extra_count = random.choices([0, 1, 2], weights=[2, 2, 1])[0]
    for edge in extra_pool[:extra_count]:
        obs = _zigzag_horizontal(edge) if edge["kind"] == "h" else _zigzag_vertical(edge)
        for r, c in obs:
            grid[r, c] = Color.RED
    return grid


def _semantic_ok(inp: np.ndarray) -> bool:
    try:
        bg = get_bg(inp)
        if bg != Color.BLACK:
            return False
        frame_color, obstacle_color, rects, obstacle_sets = _extract_objects(inp)
        if frame_color != Color.BLUE:
            return False
        if obstacle_color != Color.RED:
            return False
        if not (4 <= len(rects) <= 5):
            return False
        if not (1 <= len(obstacle_sets) <= 3):
            return False

        raw_edges = _candidate_edges(inp, bg, rects, obstacle_sets)
        edges = [_annotate_edge(inp, frame_color, rects, edge) for edge in raw_edges]
        if len(edges) < len(rects):
            return False
        if not any(edge["obstacle"] is not None and edge["effective"] > edge["gap"] for edge in edges):
            return False

        direct = _spanning_edges(edges, rects, effective=False)
        routed = _spanning_edges(edges, rects, effective=True)
        if {_edge_id(e) for e in direct} == {_edge_id(e) for e in routed}:
            return False

        usage = {idx: 0 for idx in range(len(obstacle_sets))}
        for edge in raw_edges:
            if edge["obstacle"] is not None:
                obs_idx = next(i for i, obs in enumerate(obstacle_sets) if obs == edge["obstacle"])
                usage[obs_idx] += 1
        if any(v != 1 for v in usage.values()):
            return False

        out = main(inp)
        if out.shape != inp.shape:
            return False
        if not np.any(out != inp):
            return False
    except AssertionError:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    rects, primary_kind = _base_layout()
    if random.random() < 0.45:
        maybe = _attach_leaf(rects)
        if maybe is not None:
            rects = maybe
    rects = _place_on_canvas(rects)
    if rects is None:
        return None

    max_r = max(r1 for _, _, r1, _ in rects)
    max_c = max(c1 for _, _, _, c1 in rects)
    grid = np.zeros((max_r + 2, max_c + 2), dtype=int)
    for rect in rects:
        _draw_frame(grid, rect)

    bg = get_bg(grid)
    raw_edges = _candidate_edges(grid, bg, sorted(rects, key=_rect_sort_key), [])
    if len(raw_edges) < len(rects):
        return None

    with_obstacles = _place_obstacles(grid, raw_edges, primary_kind)
    if with_obstacles is None:
        return None
    if not _semantic_ok(with_obstacles):
        return None
    return with_obstacles


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid obstacle-rerouted corridor puzzle")
