# Combo variant of 35ab12c3 (index 27)
# combo_spec: D2/3+D3/0+D9/1
# Name: 曼哈顿折线+最近邻贪心+干扰项障碍
#
# Design analysis
# a) Concrete behavior trace:
#    Original 35ab12c3 grouped same-color pixels, angle-sorted them, then drew only
#    H/V/45deg edges. Here each >=2-pixel color is traversed as an OPEN greedy path:
#    start at the seeded vertex (shadow-adjacent if present, else smallest (r,c)),
#    then repeatedly connect the nearest unvisited vertex. For a non-aligned pair
#    A=(1,1), B=(4,6), original logic would draw nothing; this combo instead tries
#    the Manhattan candidates A->(1,6)->B and A->(4,1)->B. Obstacles are singleton
#    distractor pixels. If the preferred candidate is blocked, the solver can either
#    switch to the other L candidate or detour one cell around the first blocker on a
#    straight subsegment; if no detour works it stops before the blocker. The carried
#    shadow follows the actually chosen routed subsegments, not an abstract ideal edge.
# b) Complexity calibration:
#    The generator uses 3-5 vertex open paths, not closed polygons, so D3 is visible.
#    At least one shape has a genuine non-aligned edge, so D2 is visible. Obstacles
#    are strongly favored and are placed specifically where they alter a Manhattan edge
#    route, so D9 is visible. Candidate routing is still low-branching: each edge has
#    at most two L candidates and each straight subsegment has at most two one-cell
#    sidesteps, which keeps the rule inferable from few examples.
# c) Hidden-assumption audit:
#    - Old code assumed centroid angle order and optional closing edge; that is gone.
#    - Old code assumed every edge is a single straight ray; now an edge is a routed
#      polyline assembled from straight subsegments.
#    - Old code assumed any different-color neighbor could be a shadow. That breaks
#      once distractors become functional. This combo only treats singleton colors
#      adjacent to the chosen start as shadows; obstacles are generated away from all
#      vertices so they cannot be mistaken for shadow anchors.
#    - D2 and D9 conflict with multi-pixel distractor groups: under Manhattan routing,
#      any 2-pixel distractor would look like a valid shape. The fusion therefore
#      restricts obstacles to singleton pixels even though D9 alone allowed 1-3.
# d) Placement legality:
#    Local patches are built on padded canvases, solved by the same `main()` used at
#    inference time, then cropped and placed with `random_free_location(..., padding=1)`.
#    This reserves room for one-cell obstacle detours and the translated shadow copy.
# e) Visual observability:
#    Humans can see the start vertex from the attached shadow anchor, then trace the
#    nearest-neighbor order, then notice that non-aligned pairs become L paths whose
#    corner/route changes in the presence of the obstacle pixel. No ordering uses color
#    ids or any invisible metadata.
# f) Visual cleanliness:
#    Shapes are monotone sparse paths with collision checks that reject self-crossing
#    routed cells. Obstacles are single pixels placed off all vertex neighborhoods, so
#    the bend/stop they induce remains locally legible.
# g) New assumptions introduced by the combo:
#    `main()` now assumes greedy-nearest decisions are unique enough to be learnable,
#    that shadow anchors are singleton colors adjacent to exactly one start vertex, and
#    that obstacle singletons are not vertex-adjacent. `generate_input()` explicitly
#    enforces all three conditions.

import numpy as np

from arc_utils import bbox_coords, colors_present, get_bg, neighbors8, paste, random_free_location

DIR8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _aligned_8(r0: int, c0: int, r1: int, c1: int) -> bool:
    dr = r1 - r0
    dc = c1 - c0
    return (dr == 0 and dc != 0) or (dc == 0 and dr != 0) or (dr != 0 and abs(dr) == abs(dc))


def _aligned(dr: int, dc: int) -> bool:
    return (dr == 0 and dc != 0) or (dc == 0 and dr != 0) or (dr != 0 and abs(dr) == abs(dc))


def _step_dir(r0: int, c0: int, r1: int, c1: int) -> tuple[int, int]:
    dr = r1 - r0
    dc = c1 - c0
    sr = 0 if dr == 0 else (1 if dr > 0 else -1)
    sc = 0 if dc == 0 else (1 if dc > 0 else -1)
    return sr, sc


def _segment_cells(r0: int, c0: int, r1: int, c1: int):
    sr, sc = _step_dir(r0, c0, r1, c1)
    steps = max(abs(r1 - r0), abs(c1 - c0))
    for k in range(steps + 1):
        yield r0 + sr * k, c0 + sc * k


def _segments_to_cells(
    segments: list[tuple[tuple[int, int], tuple[int, int]]],
) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for idx, (a, b) in enumerate(segments):
        seg = list(_segment_cells(a[0], a[1], b[0], b[1]))
        if idx:
            seg = seg[1:]
        cells.extend(seg)
    return cells


def _draw_path_bg_only(
    out: np.ndarray,
    segments: list[tuple[tuple[int, int], tuple[int, int]]],
    col: int,
    bg: int,
) -> None:
    for rr, cc in _segments_to_cells(segments):
        if out[rr, cc] == bg:
            out[rr, cc] = col


def _clear_segment_in_original(
    g: np.ndarray,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    bg: int,
    allow: set[int],
) -> bool:
    first = True
    for rr, cc in _segment_cells(r0, c0, r1, c1):
        if first:
            first = False
            continue
        if rr == r1 and cc == c1:
            break
        v = int(g[rr, cc])
        if v != bg and v not in allow:
            return False
    return True


def _color_counts(g: np.ndarray) -> dict[int, int]:
    vals, counts = np.unique(g, return_counts=True)
    return {int(v): int(c) for v, c in zip(vals, counts)}


def _dist2(a: tuple[int, int], b: tuple[int, int]) -> int:
    dr = a[0] - b[0]
    dc = a[1] - b[1]
    return dr * dr + dc * dc


def _greedy_path_trace(
    points_rc: list[tuple[int, int]],
    start: tuple[int, int],
) -> tuple[list[tuple[int, int]], bool]:
    path = [start]
    unique_nearest = True
    unvisited = {p for p in points_rc}
    unvisited.remove(start)
    current = start
    while unvisited:
        ranked = sorted(unvisited, key=lambda p: (_dist2(current, p), p[0], p[1]))
        if len(ranked) > 1 and _dist2(current, ranked[0]) == _dist2(current, ranked[1]):
            unique_nearest = False
        current = ranked[0]
        path.append(current)
        unvisited.remove(current)
    return path, unique_nearest


def _choose_start(
    g: np.ndarray,
    pts_rc: list[tuple[int, int]],
    col: int,
    bg: int,
    color_counts: dict[int, int],
) -> tuple[int, int]:
    h, w = g.shape
    seeded: list[tuple[int, int]] = []
    for r, c in pts_rc:
        for drn, dcn in DIR8:
            rr = r + drn
            cc = c + dcn
            if 0 <= rr < h and 0 <= cc < w:
                v = int(g[rr, cc])
                if v != bg and v != col and color_counts.get(v, 0) == 1:
                    seeded.append((r, c))
                    break
    return min(seeded) if seeded else min(pts_rc)


def _candidate_waypoints(a: tuple[int, int], b: tuple[int, int]) -> list[list[tuple[int, int]]]:
    if _aligned_8(a[0], a[1], b[0], b[1]):
        return [[a, b]]
    return [[a, (a[0], b[1]), b], [a, (b[0], a[1]), b]]


def _perp_offsets(dr: int, dc: int) -> list[tuple[int, int]]:
    return [(-dc, dr), (dc, -dr)]


def _cell_clear(g: np.ndarray, r: int, c: int, bg: int, allow: set[int]) -> bool:
    return int(g[r, c]) == bg or int(g[r, c]) in allow


def _first_blocker(
    g: np.ndarray,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    bg: int,
    allow: set[int],
) -> tuple[int, tuple[int, int]] | None:
    cells = list(_segment_cells(r0, c0, r1, c1))
    for idx, (rr, cc) in enumerate(cells[1:-1], start=1):
        if not _cell_clear(g, rr, cc, bg, allow):
            return idx, (rr, cc)
    return None


def _route_straight_segment(
    g: np.ndarray,
    a: tuple[int, int],
    b: tuple[int, int],
    bg: int,
    allow: set[int],
) -> dict:
    if not _aligned_8(a[0], a[1], b[0], b[1]):
        return {"segments": [], "cells": (), "completed": False, "detours": 0}

    blocker = _first_blocker(g, a[0], a[1], b[0], b[1], bg, allow)
    if blocker is None:
        segs = [(a, b)]
        return {"segments": segs, "cells": tuple(_segments_to_cells(segs)), "completed": True, "detours": 0}

    cells = list(_segment_cells(a[0], a[1], b[0], b[1]))
    block_idx, _ = blocker
    prev = cells[block_idx - 1]
    drs, dcs = _step_dir(a[0], a[1], b[0], b[1])

    for nr, nc in _perp_offsets(drs, dcs):
        mid0 = (prev[0] + nr, prev[1] + nc)
        mid1 = (b[0] + nr, b[1] + nc)
        if not (
            0 <= mid0[0] < g.shape[0]
            and 0 <= mid0[1] < g.shape[1]
            and 0 <= mid1[0] < g.shape[0]
            and 0 <= mid1[1] < g.shape[1]
        ):
            continue
        if int(g[mid0]) != bg or int(g[mid1]) != bg:
            continue
        if not _clear_segment_in_original(g, mid0[0], mid0[1], mid1[0], mid1[1], bg, allow=set()):
            continue
        segs: list[tuple[tuple[int, int], tuple[int, int]]] = []
        if a != prev:
            segs.append((a, prev))
        segs.append((prev, mid0))
        segs.append((mid0, mid1))
        segs.append((mid1, b))
        return {"segments": segs, "cells": tuple(_segments_to_cells(segs)), "completed": True, "detours": 1}

    if a != prev:
        segs = [(a, prev)]
        return {"segments": segs, "cells": tuple(_segments_to_cells(segs)), "completed": False, "detours": 0}
    return {"segments": [], "cells": (), "completed": False, "detours": 0}


def _route_candidate(
    g: np.ndarray,
    waypoints: list[tuple[int, int]],
    bg: int,
    allow: set[int],
) -> dict:
    all_segments: list[tuple[tuple[int, int], tuple[int, int]]] = []
    detours = 0
    completed = True
    for idx in range(len(waypoints) - 1):
        info = _route_straight_segment(g, waypoints[idx], waypoints[idx + 1], bg, allow)
        if info["segments"]:
            all_segments.extend(info["segments"])
        detours += int(info["detours"])
        if not info["completed"]:
            completed = False
            break
    cells = tuple(_segments_to_cells(all_segments))
    return {"segments": all_segments, "cells": cells, "completed": completed, "detours": detours}


def _route_edge(
    g: np.ndarray,
    a: tuple[int, int],
    b: tuple[int, int],
    bg: int,
    allow: set[int],
) -> dict:
    candidates = _candidate_waypoints(a, b)
    scored: list[tuple[tuple[int, int, int, int], int, dict]] = []
    for idx, waypoints in enumerate(candidates):
        info = _route_candidate(g, waypoints, bg, allow)
        if info["completed"]:
            score = (0, info["detours"], len(info["cells"]), idx)
        else:
            score = (1, -len(info["cells"]), info["detours"], idx)
        scored.append((score, idx, info))
    _, idx, info = min(scored, key=lambda x: x[0])
    out = dict(info)
    out["candidate_idx"] = idx
    out["waypoints"] = candidates[idx]
    return out


def _path_offset_compatible(
    segments: list[tuple[tuple[int, int], tuple[int, int]]],
    dr: int,
    dc: int,
) -> bool:
    for (r0, c0), (r1, c1) in segments:
        ds_r, ds_c = _step_dir(r0, c0, r1, c1)
        if ds_r * dc - ds_c * dr == 0:
            return False
    return True


def _shift_segments(
    segments: list[tuple[tuple[int, int], tuple[int, int]]],
    dr: int,
    dc: int,
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    return [((a[0] + dr, a[1] + dc), (b[0] + dr, b[1] + dc)) for a, b in segments]


def _route_infos_for_vertices(
    g: np.ndarray,
    vertices: list[tuple[int, int]],
    col: int,
    bg: int,
) -> list[dict]:
    return [_route_edge(g, vertices[i], vertices[i + 1], bg, allow={col}) for i in range(len(vertices) - 1)]


def _manhattan_edge_count(vertices: list[tuple[int, int]]) -> int:
    return sum(1 for i in range(len(vertices) - 1) if not _aligned_8(vertices[i][0], vertices[i][1], vertices[i + 1][0], vertices[i + 1][1]))


def _shape_route_cells(vertices: list[tuple[int, int]]) -> tuple[set[tuple[int, int]], list[tuple[tuple[int, int], tuple[int, int]]]] | None:
    h = max(r for r, _ in vertices) + 9
    w = max(c for _, c in vertices) + 9
    g = np.full((h, w), BG, dtype=int)
    shifted = [(r + 4, c + 4) for r, c in vertices]
    for r, c in shifted:
        g[r, c] = 1
    infos = _route_infos_for_vertices(g, shifted, 1, BG)
    all_segments: list[tuple[tuple[int, int], tuple[int, int]]] = []
    cells_seen: dict[tuple[int, int], int] = {}
    verts = set(shifted)
    for i, info in enumerate(infos):
        if not info["completed"]:
            return None
        if not _aligned_8(shifted[i][0], shifted[i][1], shifted[i + 1][0], shifted[i + 1][1]) and info["candidate_idx"] != 0:
            return None
        for rr, cc in info["cells"][1:-1]:
            if (rr, cc) in verts and (rr, cc) not in {shifted[i], shifted[i + 1]}:
                return None
        all_segments.extend(info["segments"])
    for rr, cc in _segments_to_cells(all_segments):
        cells_seen[(rr, cc)] = cells_seen.get((rr, cc), 0) + 1
    if any(count > 1 and cell not in verts for cell, count in cells_seen.items()):
        return None
    base_cells = {(r - 4, c - 4) for r, c in cells_seen}
    base_segments = [((a[0] - 4, a[1] - 4), (b[0] - 4, b[1] - 4)) for a, b in all_segments]
    return base_cells, base_segments


# Solver overview:
# Input structure:
#   - Each >=2-pixel non-background color is a shape's vertex set.
#   - Optional singleton colors adjacent to the chosen start vertex are shadow anchors.
#   - Optional singleton colors away from all vertices are obstacles and remain unchanged.
# Rule:
#   - Pick a start vertex, greedily visit the nearest unvisited same-color vertex, and
#     connect consecutive vertices.
#   - Straight-aligned pairs use one routed straight segment; non-aligned pairs try two
#     L candidates, preferring the lower-cost routed candidate under obstacle blocking.
#   - If a straight subsegment hits an obstacle, detour one cell around the first blocker
#     if possible, otherwise stop before it.
#   - Copy each start-anchor shadow along the actually routed main subsegments by the
#     same offset whenever the shifted subsegment is clear.
# Key insight:
#   This is no longer "find the polygon"; it is "read the start, then follow a greedy
#   path planner whose local routing can switch L corners or bend around a blocker."
def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = int(get_bg(g))
    out = g.copy()
    color_counts = _color_counts(g)

    colors = sorted(set(int(v) for v in np.unique(g)) - {bg})
    for col in colors:
        pts_rc = [tuple(map(int, p)) for p in np.argwhere(g == col)]
        if len(pts_rc) < 2:
            continue

        start = _choose_start(g, pts_rc, col, bg, color_counts)
        path, _ = _greedy_path_trace(pts_rc, start)

        carried: list[tuple[int, int, int]] = []
        sr0, sc0 = path[0]
        for drn, dcn in DIR8:
            rr = sr0 + drn
            cc = sc0 + dcn
            if 0 <= rr < h and 0 <= cc < w:
                v = int(g[rr, cc])
                if v != bg and v != col and color_counts.get(v, 0) == 1:
                    carried.append((drn, dcn, v))

        routed_edges: list[dict] = []
        for i in range(len(path) - 1):
            info = _route_edge(g, path[i], path[i + 1], bg, allow={col})
            if info["segments"]:
                routed_edges.append(info)

        for info in routed_edges:
            _draw_path_bg_only(out, info["segments"], col, bg)

        for info in routed_edges:
            for drn, dcn, ncol in carried:
                if not _path_offset_compatible(info["segments"], drn, dcn):
                    continue
                shifted = _shift_segments(info["segments"], drn, dcn)
                ok = True
                for (r0, c0), (r1, c1) in shifted:
                    if not (0 <= r0 < h and 0 <= c0 < w and 0 <= r1 < h and 0 <= c1 < w):
                        ok = False
                        break
                    if int(g[r0, c0]) not in {bg, ncol} or int(g[r1, c1]) not in {bg, ncol}:
                        ok = False
                        break
                    if not _clear_segment_in_original(g, r0, c0, r1, c1, bg, allow=set()):
                        ok = False
                        break
                if ok:
                    _draw_path_bg_only(out, shifted, ncol, bg)

    return out


# --------------- generator ---------------

BG = 0
COLORS = [1, 2, 3, 4, 6, 7, 8]
CANVAS_H_RANGE = (15, 23)
CANVAS_W_JITTER = (-3, 4)
CANVAS_W_CLIP = (15, 22)

SHAPE_COUNT_RANGE = (2, 5)
MAX_SHADOWS_CAP = 3
OBSTACLE_PROB = 1.0
OBSTACLE_EFFECT_MODES = ("switch", "detour")
PLACEMENT_PADDING = 1
PLACEMENT_PAD_CONNECTIVITY = 8

PATH_VERTICES_RANGE = (3, 6)
SEG_LEN_RANGE = (2, 5)
SKEW_DR_RANGE = (2, 5)
SKEW_DC_RANGE = (2, 5)
PATH_BUILD_RETRIES = 120
PATCH_MARGIN = 4

SHAPE_PLACEMENT_RETRIES = 40
PATCH_RETRIES = 80
GENERATE_RETRIES = 300


def _transform_coords(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    arr = np.array(coords, dtype=int)
    if np.random.rand() < 0.5:
        arr = arr[:, ::-1]
    arr -= arr.min(axis=0)
    return [tuple(map(int, p)) for p in arr]


def _build_vertices(require_manhattan: bool) -> list[tuple[int, int]] | None:
    for _ in range(PATH_BUILD_RETRIES):
        n_vertices = int(np.random.randint(*PATH_VERTICES_RANGE))
        kinds: list[str] = []
        vertices = [(0, 0)]
        used_skew = False
        for _seg in range(n_vertices - 1):
            choices = ["H", "V", "S"]
            if require_manhattan and not used_skew and len(kinds) == n_vertices - 2:
                kind = "S"
            else:
                kind = choices[int(np.random.randint(len(choices)))]
            if kind == "H":
                delta = (0, int(np.random.randint(*SEG_LEN_RANGE)))
            elif kind == "V":
                delta = (int(np.random.randint(*SEG_LEN_RANGE)), 0)
            else:
                dr = int(np.random.randint(*SKEW_DR_RANGE))
                dc = int(np.random.randint(*SKEW_DC_RANGE))
                if dr == dc:
                    dc += 1
                delta = (dr, dc)
                used_skew = True
            last = vertices[-1]
            vertices.append((last[0] + delta[0], last[1] + delta[1]))
            kinds.append(kind)

        if require_manhattan and not used_skew:
            continue
        if len(set(vertices)) != len(vertices):
            continue
        vertices = _transform_coords(vertices)
        if vertices[0] != min(vertices):
            continue
        greedy, unique = _greedy_path_trace(vertices, vertices[0])
        if greedy != vertices or not unique:
            continue
        shape_data = _shape_route_cells(vertices)
        if shape_data is None:
            continue
        return vertices
    return None


def _choose_shadow_offset(
    vertices: list[tuple[int, int]],
    shape_cells: set[tuple[int, int]],
    shape_segments: list[tuple[tuple[int, int], tuple[int, int]]],
) -> tuple[int, int] | None:
    anchor = vertices[0]
    candidates: list[tuple[int, int]] = []
    for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
        if not _path_offset_compatible(shape_segments, dr, dc):
            continue
        pos = (anchor[0] + dr, anchor[1] + dc)
        if pos in shape_cells:
            continue
        if sum(1 for vr, vc in vertices if max(abs(pos[0] - vr), abs(pos[1] - vc)) == 1) != 1:
            continue
        candidates.append((dr, dc))
    if not candidates:
        return None
    return candidates[int(np.random.randint(len(candidates)))]


def _crop_pair(inp: np.ndarray, out: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = (inp != BG) | (out != BG)
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = bbox_coords(coords)
    return inp[r0 : r1 + 1, c0 : c1 + 1].copy(), out[r0 : r1 + 1, c0 : c1 + 1].copy()


def _build_base_patch(
    shape_color: int,
    shadow_color: int | None,
    require_manhattan: bool,
) -> tuple[np.ndarray, list[tuple[int, int]]] | None:
    vertices = _build_vertices(require_manhattan=require_manhattan)
    if vertices is None:
        return None
    route_data = _shape_route_cells(vertices)
    if route_data is None:
        return None
    shape_cells, shape_segments = route_data

    h = max(r for r, _ in vertices) + 1 + 2 * PATCH_MARGIN
    w = max(c for _, c in vertices) + 1 + 2 * PATCH_MARGIN
    inp = np.full((h, w), BG, dtype=int)
    shifted_vertices = [(r + PATCH_MARGIN, c + PATCH_MARGIN) for r, c in vertices]
    shifted_shape_cells = {(r + PATCH_MARGIN, c + PATCH_MARGIN) for r, c in shape_cells}
    shifted_shape_segments = [
        ((a[0] + PATCH_MARGIN, a[1] + PATCH_MARGIN), (b[0] + PATCH_MARGIN, b[1] + PATCH_MARGIN))
        for a, b in shape_segments
    ]

    for r, c in shifted_vertices:
        inp[r, c] = int(shape_color)

    if shadow_color is not None:
        offset = _choose_shadow_offset(shifted_vertices, shifted_shape_cells, shifted_shape_segments)
        if offset is None:
            return None
        anchor = shifted_vertices[0]
        sa = (anchor[0] + offset[0], anchor[1] + offset[1])
        if not (0 <= sa[0] < h and 0 <= sa[1] < w) or inp[sa] != BG:
            return None
        inp[sa] = int(shadow_color)

    return inp, shifted_vertices


def _route_effect_stats(
    base_inp: np.ndarray,
    trial_inp: np.ndarray,
    vertices: list[tuple[int, int]],
    shape_color: int,
) -> dict[str, int]:
    base_infos = _route_infos_for_vertices(base_inp, vertices, int(shape_color), BG)
    trial_infos = _route_infos_for_vertices(trial_inp, vertices, int(shape_color), BG)
    diff_edges = 0
    diff_manhattan = 0
    completed_diff = 0
    candidate_switches = 0
    detour_diffs = 0
    stop_diffs = 0
    for i, (base, trial) in enumerate(zip(base_infos, trial_infos)):
        changed = base["cells"] != trial["cells"]
        if changed:
            diff_edges += 1
            if not _aligned_8(vertices[i][0], vertices[i][1], vertices[i + 1][0], vertices[i + 1][1]):
                diff_manhattan += 1
            if trial["completed"]:
                completed_diff += 1
            if base["candidate_idx"] != trial["candidate_idx"]:
                candidate_switches += 1
            if trial["detours"] > base["detours"]:
                detour_diffs += 1
            if base["completed"] and not trial["completed"]:
                stop_diffs += 1
    return {
        "diff_edges": diff_edges,
        "diff_manhattan": diff_manhattan,
        "completed_diff": completed_diff,
        "candidate_switches": candidate_switches,
        "detour_diffs": detour_diffs,
        "stop_diffs": stop_diffs,
    }


def _find_obstacle_cell(
    base_inp: np.ndarray,
    vertices: list[tuple[int, int]],
    shape_color: int,
    obstacle_color: int,
    effect_mode: str,
) -> tuple[int, int] | None:
    occupied = [tuple(map(int, p)) for p in np.argwhere(base_inp != BG)]
    best_cell = None
    best_score = None
    for r, c in np.argwhere(base_inp == BG):
        rr = int(r)
        cc = int(c)
        if any(max(abs(rr - vr), abs(cc - vc)) <= 1 for vr, vc in vertices):
            continue
        if any(max(abs(rr - ar), abs(cc - ac)) <= 1 for ar, ac in occupied if (ar, ac) not in vertices):
            continue
        trial = base_inp.copy()
        trial[rr, cc] = int(obstacle_color)
        stats = _route_effect_stats(base_inp, trial, vertices, int(shape_color))
        if effect_mode == "switch":
            if stats["candidate_switches"] == 0 or stats["diff_manhattan"] == 0 or stats["detour_diffs"] != 0:
                continue
            score = (
                stats["completed_diff"],
                stats["candidate_switches"],
                stats["diff_manhattan"],
                stats["diff_edges"],
                -abs(rr - vertices[0][0]) - abs(cc - vertices[0][1]),
            )
        elif effect_mode == "detour":
            if stats["detour_diffs"] == 0 or stats["candidate_switches"] != 0 or stats["stop_diffs"] != 0:
                continue
            score = (
                stats["completed_diff"],
                stats["detour_diffs"],
                stats["diff_edges"],
                -abs(rr - vertices[0][0]) - abs(cc - vertices[0][1]),
            )
        else:
            raise ValueError(f"unknown obstacle effect mode: {effect_mode}")
        if best_score is None or score > best_score:
            best_score = score
            best_cell = (rr, cc)
    return best_cell


def _make_shape_patch(
    shape_color: int,
    shadow_color: int | None,
    obstacle_color: int | None,
    require_manhattan: bool,
    require_obstacle_effect: bool,
    obstacle_effect_mode: str | None,
) -> tuple[np.ndarray, np.ndarray] | None:
    base = _build_base_patch(int(shape_color), shadow_color, require_manhattan=require_manhattan)
    if base is None:
        return None
    inp, vertices = base

    if obstacle_color is not None:
        if obstacle_effect_mode is None:
            return None
        obstacle_cell = _find_obstacle_cell(
            inp, vertices, int(shape_color), int(obstacle_color), obstacle_effect_mode
        )
        if obstacle_cell is None:
            return None
        inp[obstacle_cell] = int(obstacle_color)
        stats = _route_effect_stats(np.where(inp == obstacle_color, BG, inp), inp, vertices, int(shape_color))
        if require_obstacle_effect:
            if obstacle_effect_mode == "switch" and stats["candidate_switches"] == 0:
                return None
            if obstacle_effect_mode == "detour" and stats["detour_diffs"] == 0:
                return None
        if require_obstacle_effect and stats["diff_edges"] == 0:
            return None

    out = main(inp)
    if not np.array_equal(main(inp), out):
        return None

    if obstacle_color is not None and require_obstacle_effect:
        metrics = _variant_metrics(inp)
        if metrics["obstacle_route_diffs"] < 1 or metrics["manhattan_edges"] < 1:
            return None
        if obstacle_effect_mode == "switch" and metrics["obstacle_candidate_switches"] < 1:
            return None
        if obstacle_effect_mode == "detour" and metrics["obstacle_detours"] < 1:
            return None

    return _crop_pair(inp, out)


def _sample_palette(
    shape_count: int,
    has_obstacle: bool,
    min_shadows: int = 0,
) -> tuple[list[int], list[int], int | None] | None:
    max_shadows = min(MAX_SHADOWS_CAP, shape_count, len(COLORS) - shape_count - int(has_obstacle))
    if max_shadows < min_shadows:
        return None
    shadow_count = int(np.random.randint(min_shadows, max_shadows + 1))
    total = shape_count + shadow_count + int(has_obstacle)
    palette = list(np.random.permutation(COLORS)[:total])
    shape_colors = palette[:shape_count]
    shadow_colors = palette[shape_count : shape_count + shadow_count]
    obstacle_color = int(palette[-1]) if has_obstacle else None
    return shape_colors, shadow_colors, obstacle_color


def _assign_shadows(
    shape_count: int,
    shadow_colors: list[int],
    required_idx: int | None = None,
) -> dict[int, int]:
    if not shadow_colors:
        return {}
    targets: list[int]
    if required_idx is None:
        targets = np.random.choice(shape_count, size=len(shadow_colors), replace=False).tolist()
    else:
        others = [idx for idx in range(shape_count) if idx != required_idx]
        take = len(shadow_colors) - 1
        extra = np.random.choice(others, size=take, replace=False).tolist() if take > 0 else []
        targets = [required_idx] + extra
    return {int(idx): int(col) for idx, col in zip(sorted(targets), shadow_colors)}


def _try_place(inp: np.ndarray, out: np.ndarray, patch_in: np.ndarray, patch_out: np.ndarray) -> bool:
    try:
        loc = random_free_location(
            out,
            patch_out,
            transparent=BG,
            padding=PLACEMENT_PADDING,
            padding_connectivity=PLACEMENT_PAD_CONNECTIVITY,
        )
    except ValueError:
        return False
    paste(out, patch_out, loc[0], loc[1], transparent=BG)
    paste(inp, patch_in, loc[0], loc[1], transparent=BG)
    return True


def _sample_canvas_size() -> tuple[int, int]:
    h = int(np.random.randint(*CANVAS_H_RANGE))
    w = int(np.clip(h + np.random.randint(*CANVAS_W_JITTER), *CANVAS_W_CLIP))
    return h, w


def _obstacle_colors(g: np.ndarray, bg: int, color_counts: dict[int, int]) -> set[int]:
    h, w = g.shape
    obstacle_cols: set[int] = set()
    for col, count in color_counts.items():
        if col == bg or count != 1:
            continue
        r, c = map(int, np.argwhere(g == col)[0])
        anchored = False
        for rr, cc in neighbors8(r, c):
            if 0 <= rr < h and 0 <= cc < w:
                v = int(g[rr, cc])
                if v != bg and v != col and color_counts.get(v, 0) >= 2:
                    anchored = True
                    break
        if not anchored:
            obstacle_cols.add(col)
    return obstacle_cols


def _variant_metrics(grid: np.ndarray) -> dict[str, int]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    color_counts = _color_counts(g)
    obstacle_cols = _obstacle_colors(g, bg, color_counts)

    base = g.copy()
    for col in obstacle_cols:
        base[base == col] = bg

    manhattan_edges = 0
    obstacle_route_diffs = 0
    affected_shapes = 0
    shape_colors = 0
    obstacle_candidate_switches = 0
    obstacle_detours = 0

    for col in sorted(set(int(v) for v in np.unique(g)) - {bg} - obstacle_cols):
        pts = [tuple(map(int, p)) for p in np.argwhere(g == col)]
        if len(pts) < 2:
            continue
        start = _choose_start(base, pts, col, bg, _color_counts(base))
        path, _ = _greedy_path_trace(pts, start)
        shape_colors += 1
        manhattan_edges += _manhattan_edge_count(path)

        full_infos = _route_infos_for_vertices(g, path, col, bg)
        base_infos = _route_infos_for_vertices(base, path, col, bg)
        local_diff = 0
        local_switches = 0
        local_detours = 0
        for a, b in zip(full_infos, base_infos):
            if a["cells"] != b["cells"]:
                local_diff += 1
                local_switches += int(a["candidate_idx"] != b["candidate_idx"])
                local_detours += int(a["detours"] > b["detours"])
        obstacle_route_diffs += local_diff
        obstacle_candidate_switches += local_switches
        obstacle_detours += local_detours
        affected_shapes += int(local_diff > 0)

    return {
        "shape_colors": shape_colors,
        "manhattan_edges": manhattan_edges,
        "obstacle_route_diffs": obstacle_route_diffs,
        "affected_shapes": affected_shapes,
        "obstacle_colors": len(obstacle_cols),
        "obstacle_candidate_switches": obstacle_candidate_switches,
        "obstacle_detours": obstacle_detours,
    }


def _gen_attempt() -> np.ndarray | None:
    h, w = _sample_canvas_size()
    inp = np.full((h, w), BG, dtype=int)
    out = np.full((h, w), BG, dtype=int)

    has_obstacle = bool(np.random.rand() < OBSTACLE_PROB)
    shape_count = int(np.random.randint(*SHAPE_COUNT_RANGE))
    obstacle_target = int(np.random.randint(shape_count)) if has_obstacle else -1
    obstacle_effect_mode = (
        OBSTACLE_EFFECT_MODES[int(np.random.randint(len(OBSTACLE_EFFECT_MODES)))] if has_obstacle else None
    )

    palette = _sample_palette(shape_count, has_obstacle, min_shadows=1)
    if palette is None:
        return None
    shape_colors, shadow_colors, obstacle_color = palette
    shadow_map = _assign_shadows(shape_count, shadow_colors, required_idx=obstacle_target if has_obstacle else None)
    total_colors = shape_count + len(shadow_colors) + int(has_obstacle)

    for idx, shape_color in enumerate(shape_colors):
        placed = False
        for _ in range(SHAPE_PLACEMENT_RETRIES):
            patch = None
            for _ in range(PATCH_RETRIES):
                patch = _make_shape_patch(
                    int(shape_color),
                    shadow_map.get(idx),
                    obstacle_color if idx == obstacle_target else None,
                    require_manhattan=(idx == obstacle_target),
                    require_obstacle_effect=(idx == obstacle_target and has_obstacle),
                    obstacle_effect_mode=obstacle_effect_mode if idx == obstacle_target else None,
                )
                if patch is not None:
                    break
            if patch is None:
                continue
            if _try_place(inp, out, *patch):
                placed = True
                break
        if not placed:
            return None

    if get_bg(inp) != BG or get_bg(out) != BG:
        return None
    if len(colors_present(inp, exclude=[BG])) != total_colors:
        return None
    if not np.array_equal(main(inp), out):
        return None

    metrics = _variant_metrics(inp)
    if metrics["manhattan_edges"] < 1:
        return None
    if has_obstacle and (metrics["obstacle_colors"] != 1 or metrics["obstacle_route_diffs"] < 1):
        return None
    if obstacle_effect_mode == "switch" and metrics["obstacle_candidate_switches"] < 1:
        return None
    if obstacle_effect_mode == "detour" and metrics["obstacle_detours"] < 1:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"generate_input failed after {GENERATE_RETRIES} attempts")
