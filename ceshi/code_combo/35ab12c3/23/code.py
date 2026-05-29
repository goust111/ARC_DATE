# Combo variant of 35ab12c3 (index 23)
# combo_spec: D2/3+D15/2
# Name: 曼哈顿折线路径+路径绕行
#
# Design analysis
# - Concrete trace:
#   In the base task, two same-color vertices are connected only when they are
#   horizontal, vertical, or perfect-diagonal aligned, and any foreign-color
#   blocker cancels that edge outright. In this combo, a non-aligned pair such as
#   A=(2,2), B=(6,8) first tries the L path A->(2,8)->B; if that L is blocked it
#   also tries A->(6,2)->B; if both simple L paths are blocked it reroutes the
#   blocked leg(s) by replacing a segment S->T with
#   S->S+shift->T+shift->T. The visible result is a U/Z-shaped orthogonal route
#   instead of a missing edge. Shadow pixels still copy the chosen full waypoint
#   path in parallel.
# - Complexity calibration:
#   The route family is deliberately finite: straight segment, one-corner L, and
#   detoured versions created from 1-2 cell side shifts. This is richer than the
#   base task and visibly different from both single variants, but it avoids an
#   unconstrained BFS that would make examples noisy and hard to infer.
# - Hidden-assumption audit for solver changes:
#   1. "Edge == one segment" is no longer valid; every edge is treated as a
#      polyline with 2-7 waypoints.
#   2. A bend cell is now an interior occupancy check, so path validation must
#      inspect whole polyline cell traces rather than only per-segment interiors.
#   3. Shadow copying can no longer shift only endpoints; it must shift the exact
#      chosen waypoint list or the shadow could take a different corner / detour.
#   4. Multi-pixel distractors become unsafe because non-aligned same-color pairs
#      now form legal shapes. The generator therefore uses only single-pixel
#      blocker/distractor colors.
#   5. L-path ambiguity cannot be resolved by hidden color-id logic. The solver
#      first accepts any fully clear base path, then prefers the base orientation
#      with more already-clear legs before searching detours, and finally uses a
#      fixed stable order for true ties.
# - Placement legality:
#   The generator uses patch-level placement with a one-cell buffer between
#   patches. The only intentional close interaction happens inside one special
#   patch containing a non-aligned two-vertex shape plus two single-pixel blockers;
#   the output footprint is computed from the routed path itself before placement.
# - Visual observability / cleanliness:
#   Every generated grid contains at least one same-color non-aligned pair whose
#   output is a routed Manhattan polyline longer than a simple L, so both fused
#   dimensions are visible in a single object. Additional shapes remain sparse and
#   separated, keeping the output readable.
# - Generator guarantees for new assumptions:
#   1. The forced routed patch has exactly two vertices of its shape color, so the
#      solver's cycle order is unambiguous.
#   2. The two blockers use unique single-pixel colors, so they remain inert under
#      the new path rules and only serve as obstacles.
#   3. Shadows are attached only to ordinary patches, not the forced routed patch,
#      so rerouting and shadow-offset constraints do not interfere in the same
#      local construction.

import math

import numpy as np

from arc_utils import bbox_coords, colors_present, get_bg, neighbors8, paste, random_free_location
from arc_utils import get_bg as _bg_color

DIR8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
# Keep reroutes Manhattan-style: the blocked segment may shift sideways, but the
# offset lane itself is entered/exited only through orthogonal sidesteps.
DETOUR_BASES = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ROUTE_SHIFT_SCALES = (1, 2)


def _aligned_8(r0: int, c0: int, r1: int, c1: int) -> bool:
    dr = r1 - r0
    dc = c1 - c0
    if dr == 0 and dc != 0:
        return True
    if dc == 0 and dr != 0:
        return True
    if dr != 0 and abs(dr) == abs(dc):
        return True
    return False


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


def _polyline_cells(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for i in range(len(points) - 1):
        seg = list(_segment_cells(points[i][0], points[i][1], points[i + 1][0], points[i + 1][1]))
        if i:
            seg = seg[1:]
        cells.extend(seg)
    return cells


def _clear_segment_in_original(
    g: np.ndarray, r0: int, c0: int, r1: int, c1: int, bg: int, allow: set[int]
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


def _polyline_clear(g: np.ndarray, points: list[tuple[int, int]], bg: int, allow: set[int]) -> bool:
    h, w = g.shape
    if len(points) < 2:
        return False
    for rr, cc in points:
        if not (0 <= rr < h and 0 <= cc < w):
            return False
    for i in range(len(points) - 1):
        if not _aligned_8(points[i][0], points[i][1], points[i + 1][0], points[i + 1][1]):
            return False
    for rr, cc in _polyline_cells(points)[1:-1]:
        v = int(g[rr, cc])
        if v != bg and v not in allow:
            return False
    return True


def _draw_polyline_bg_only(out: np.ndarray, points: list[tuple[int, int]], col: int, bg: int) -> None:
    for rr, cc in _polyline_cells(points):
        if out[rr, cc] == bg:
            out[rr, cc] = col


def _shift_points(points: list[tuple[int, int]], dr: int, dc: int) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for r, c in points]


def _compress_points(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for p in points:
        if not out or p != out[-1]:
            out.append(p)
    return out


def _candidate_base_paths(a: tuple[int, int], b: tuple[int, int]) -> list[list[tuple[int, int]]]:
    r0, c0 = a
    r1, c1 = b
    if _aligned_8(r0, c0, r1, c1):
        return [[a, b]]
    return [[a, (r0, c1), b], [a, (r1, c0), b]]


def _count_clear_segments(g: np.ndarray, points: list[tuple[int, int]], bg: int, allow: set[int]) -> int:
    count = 0
    for i in range(len(points) - 1):
        if _clear_segment_in_original(g, points[i][0], points[i][1], points[i + 1][0], points[i + 1][1], bg, allow):
            count += 1
    return count


def _candidate_detour_shifts(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    drs, dcs = _step_dir(r0, c0, r1, c1)
    out: list[tuple[int, int]] = []
    for scale in ROUTE_SHIFT_SCALES:
        for br, bc in DETOUR_BASES:
            if drs * bc - dcs * br == 0:
                continue
            out.append((br * scale, bc * scale))
    return out


def _segment_route_options(
    g: np.ndarray, a: tuple[int, int], b: tuple[int, int], bg: int, allow: set[int]
) -> list[list[tuple[int, int]]]:
    if _clear_segment_in_original(g, a[0], a[1], b[0], b[1], bg, allow):
        return [[a, b]]
    routes: list[list[tuple[int, int]]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    for drs, dcs in _candidate_detour_shifts(a[0], a[1], b[0], b[1]):
        path = [a, (a[0] + drs, a[1] + dcs), (b[0] + drs, b[1] + dcs), b]
        path = _compress_points(path)
        key = tuple(path)
        if key in seen:
            continue
        seen.add(key)
        if _polyline_clear(g, path, bg, allow):
            routes.append(path)
    return routes


def _route_length(points: list[tuple[int, int]]) -> int:
    total = 0
    for i in range(len(points) - 1):
        total += max(abs(points[i + 1][0] - points[i][0]), abs(points[i + 1][1] - points[i][1]))
    return total


def _expand_path_routes(
    g: np.ndarray, base: list[tuple[int, int]], bg: int, allow: set[int]
) -> list[list[tuple[int, int]]]:
    paths: list[list[tuple[int, int]]] = [[base[0]]]
    for i in range(len(base) - 1):
        seg_routes = _segment_route_options(g, base[i], base[i + 1], bg, allow)
        if not seg_routes:
            return []
        next_paths: list[list[tuple[int, int]]] = []
        for prefix in paths:
            for seg in seg_routes:
                next_paths.append(_compress_points(prefix + seg[1:]))
        paths = next_paths
    valid: list[list[tuple[int, int]]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    for path in paths:
        key = tuple(path)
        if key in seen:
            continue
        seen.add(key)
        if _polyline_clear(g, path, bg, allow):
            valid.append(path)
    valid.sort(key=lambda pts: (_route_length(pts), len(pts), pts))
    return valid


def _find_path_in_original(
    g: np.ndarray, a: tuple[int, int], b: tuple[int, int], bg: int, allow: set[int]
) -> list[tuple[int, int]] | None:
    bases = _candidate_base_paths(a, b)
    for base in bases:
        if _polyline_clear(g, base, bg, allow):
            return base
    scored = sorted(
        enumerate(bases),
        key=lambda item: (-_count_clear_segments(g, item[1], bg, allow), item[0]),
    )
    for _, base in scored:
        routes = _expand_path_routes(g, base, bg, allow)
        if routes:
            return routes[0]
    return None


def _path_offset_compatible(points: list[tuple[int, int]], dr: int, dc: int) -> bool:
    for i in range(len(points) - 1):
        ds_r, ds_c = _step_dir(points[i][0], points[i][1], points[i + 1][0], points[i + 1][1])
        if ds_r * dc - ds_c * dr == 0:
            return False
    return True


def _angle_cycle(points_rc: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cr = sum(r for r, _ in points_rc) / len(points_rc)
    cc = sum(c for _, c in points_rc) / len(points_rc)

    def key(p: tuple[int, int]):
        r, c = p
        ang = math.atan2(r - cr, c - cc)
        rad = (r - cr) ** 2 + (c - cc) ** 2
        return (ang, rad, r, c)

    return sorted(points_rc, key=key)


def _ordered_cycle_and_carried(g: np.ndarray, col: int, bg: int):
    h, w = g.shape
    pts_rc = [tuple(map(int, p)) for p in np.argwhere(g == col)]
    if len(pts_rc) < 2:
        return None, None

    cycle = _angle_cycle(pts_rc)
    seed_idxs = []
    for i, (r, c) in enumerate(cycle):
        for drn, dcn in DIR8:
            rr = r + drn
            cc = c + dcn
            if 0 <= rr < h and 0 <= cc < w:
                v = int(g[rr, cc])
                if v != bg and v != col:
                    seed_idxs.append(i)
                    break

    if seed_idxs:
        start = min(seed_idxs, key=lambda i: (cycle[i][0], cycle[i][1]))
    else:
        start = min(range(len(cycle)), key=lambda i: (cycle[i][0], cycle[i][1]))
    cycle = cycle[start:] + cycle[:start]

    carried: list[tuple[int, int, int]] = []
    sr0, sc0 = cycle[0]
    for drn, dcn in DIR8:
        rr = sr0 + drn
        cc = sc0 + dcn
        if 0 <= rr < h and 0 <= cc < w:
            v = int(g[rr, cc])
            if v != bg and v != col:
                carried.append((drn, dcn, v))
    return cycle, carried


def _cycle_edges(cycle: list[tuple[int, int]]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    if len(cycle) == 2:
        return [(cycle[0], cycle[1])]
    return [(cycle[i], cycle[(i + 1) % len(cycle)]) for i in range(len(cycle))]


# Overview:
#   Rebuild each same-color sparse vertex set into a polyline/polygon. Non-aligned
#   pairs use Manhattan L paths; if the preferred simple path is blocked, the
#   blocked leg(s) are rerouted with short offset lanes, producing U/Z-like paths.
# Input structure:
#   The input still contains sparse same-color vertices for 2-4 shapes, optional
#   one-pixel shadow anchors near one start vertex, and optional one-pixel blocker /
#   distractor colors. Distinct groups remain separated by background padding.
# Rule:
#   For each shape color, order its vertices cyclically, connect consecutive pairs
#   by the first valid path among: clear straight path, clear L path, or detoured
#   versions of that straight/L path. Then copy any carried neighbor pixel along the
#   exact chosen path at the same offset, provided the shifted path is clear.
# Key insight:
#   The transformation is no longer "draw only aligned rays"; it is "choose the
#   simplest obstacle-free Manhattan-style route between the ordered vertices, then
#   shadow-copy that same route."
def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    h, w = g.shape
    out = g.copy()

    colors = sorted(set(int(v) for v in np.unique(g)) - {bg})
    for col in colors:
        cycle, carried = _ordered_cycle_and_carried(g, col, bg)
        if cycle is None or carried is None:
            continue

        drawable_paths: list[list[tuple[int, int]]] = []
        for a, b in _cycle_edges(cycle):
            path = _find_path_in_original(g, a, b, bg, allow={col})
            if path is not None:
                drawable_paths.append(path)

        for path in drawable_paths:
            _draw_polyline_bg_only(out, path, col, bg)

        for path in drawable_paths:
            for drn, dcn, ncol in carried:
                if not _path_offset_compatible(path, drn, dcn):
                    continue
                shifted = _shift_points(path, drn, dcn)
                if not all(0 <= rr < h and 0 <= cc < w for rr, cc in shifted):
                    continue
                if not _polyline_clear(g, shifted, bg, allow=set()):
                    continue
                _draw_polyline_bg_only(out, shifted, ncol, bg)

    return out


# --------------- generator ---------------

BG = 0
COLORS = [1, 2, 3, 4, 6, 7, 8]
CANVAS_H_RANGE = (15, 23)
CANVAS_W_JITTER = (-3, 4)
CANVAS_W_CLIP = (15, 22)

SHAPE_COUNT_RANGE = (2, 5)
MAX_SHADOWS_CAP = 1
PLACEMENT_PADDING = 1
PLACEMENT_PAD_CONNECTIVITY = 8

PAIR_H_RANGE = (2, 8)
PAIR_W_RANGE = (2, 8)
TRIANGLE_H_RANGE = (3, 8)
TRIANGLE_W_RANGE = (4, 9)
QUAD_H_RANGE = (4, 9)
QUAD_W_RANGE = (4, 9)
PENT_H_RANGE = (4, 8)
PENT_W_RANGE = (4, 8)
RECT_H_RANGE = (2, 8)
RECT_W_RANGE = (2, 8)
SHAPE_WEIGHTS = np.array([0.28, 0.22, 0.18, 0.16, 0.16], dtype=float)

ROUTED_PATCH_RETRIES = 120
SPECIAL_PAD = 4
SPECIAL_H_RANGE = (4, 8)
SPECIAL_W_RANGE = (5, 10)
SHAPE_PLACEMENT_RETRIES = 40
VERTEX_RETRIES = 80
GENERATE_RETRIES = 300


def _transform_coords(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    arr = np.array(coords, dtype=int)
    if np.random.rand() < 0.5:
        arr = arr[:, ::-1]
    if np.random.rand() < 0.5:
        arr[:, 0] *= -1
    if np.random.rand() < 0.5:
        arr[:, 1] *= -1
    arr -= arr.min(axis=0)
    return [tuple(map(int, p)) for p in arr]


def _build_l_pair() -> list[tuple[int, int]]:
    while True:
        h = int(np.random.randint(*PAIR_H_RANGE))
        w = int(np.random.randint(*PAIR_W_RANGE))
        if h != w:
            return _transform_coords([(0, 0), (h, w)])


def _build_skew_triangle() -> list[tuple[int, int]]:
    for _ in range(VERTEX_RETRIES):
        w = int(np.random.randint(*TRIANGLE_W_RANGE))
        h = int(np.random.randint(*TRIANGLE_H_RANGE))
        k = int(np.random.randint(1, w))
        if h == k or h == abs(w - k):
            continue
        vertices = _transform_coords([(0, 0), (0, w), (h, k)])
        if _shape_has_manhattan_edge(vertices):
            return vertices
    return _build_l_pair()


def _build_skew_quad() -> list[tuple[int, int]]:
    h = int(np.random.randint(*QUAD_H_RANGE))
    w = int(np.random.randint(*QUAD_W_RANGE))
    return _transform_coords([(0, 0), (1, w), (h, w - 1), (h - 1, 0)])


def _build_house() -> list[tuple[int, int]]:
    h = int(np.random.randint(*PENT_H_RANGE))
    w = int(np.random.randint(*PENT_W_RANGE))
    return _transform_coords([(0, 1), (1, 0), (h, 0), (h, w), (1, w - 1)])


def _build_rectangle() -> list[tuple[int, int]]:
    h = int(np.random.randint(*RECT_H_RANGE))
    w = int(np.random.randint(*RECT_W_RANGE))
    return _transform_coords([(0, 0), (0, w), (h, w), (h, 0)])


_SHAPE_BUILDERS = [_build_l_pair, _build_skew_triangle, _build_skew_quad, _build_house, _build_rectangle]


def _shape_cycle(vertices: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return _angle_cycle(vertices)


def _shape_edges(vertices: list[tuple[int, int]]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    cycle = _shape_cycle(vertices)
    if len(cycle) == 2:
        return [(cycle[0], cycle[1])]
    return [(cycle[i], cycle[(i + 1) % len(cycle)]) for i in range(len(cycle))]


def _shape_has_manhattan_edge(vertices: list[tuple[int, int]]) -> bool:
    return any(not _aligned_8(a[0], a[1], b[0], b[1]) for a, b in _shape_edges(vertices))


def _preferred_clear_path(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    return _candidate_base_paths(a, b)[0]


def _shape_paths(vertices: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    return [_preferred_clear_path(a, b) for a, b in _shape_edges(vertices)]


def _shape_cells(vertices: list[tuple[int, int]]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for points in _shape_paths(vertices):
        cells.update(_polyline_cells(points))
    return cells


def _shape_segment_dirs(vertices: list[tuple[int, int]]) -> list[tuple[int, int]]:
    dirs: list[tuple[int, int]] = []
    for points in _shape_paths(vertices):
        for i in range(len(points) - 1):
            dirs.append(_step_dir(points[i][0], points[i][1], points[i + 1][0], points[i + 1][1]))
    return dirs


def _make_vertices(require_manhattan: bool) -> list[tuple[int, int]]:
    weights = SHAPE_WEIGHTS / SHAPE_WEIGHTS.sum()
    for _ in range(VERTEX_RETRIES):
        idx = int(np.random.choice(len(_SHAPE_BUILDERS), p=weights))
        vertices = _SHAPE_BUILDERS[idx]()
        if len(set(vertices)) != len(vertices):
            continue
        if require_manhattan and not _shape_has_manhattan_edge(vertices):
            continue
        return vertices
    return _build_l_pair()


def _choose_shadow(
    vertices: list[tuple[int, int]], shape_cells: set[tuple[int, int]]
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    anchor = min(vertices)
    directions = _shape_segment_dirs(vertices)
    candidates: list[tuple[int, int]] = []
    for dr, dc in DIR8:
        if any(ds_r * dc - ds_c * dr == 0 for ds_r, ds_c in directions):
            continue
        pos = (anchor[0] + dr, anchor[1] + dc)
        if pos in shape_cells:
            continue
        if sum(1 for vr, vc in vertices if max(abs(pos[0] - vr), abs(pos[1] - vc)) == 1) != 1:
            continue
        candidates.append((dr, dc))
    if not candidates:
        return None
    return anchor, candidates[int(np.random.randint(len(candidates)))]


def _patch_bbox(
    shape_cells: set[tuple[int, int]],
    vertices: list[tuple[int, int]],
    shadow: tuple[tuple[int, int], tuple[int, int]] | None,
) -> tuple[int, int, int, int]:
    coords = list(shape_cells) + vertices
    if shadow is not None:
        (ar, ac), (dr, dc) = shadow
        coords.append((ar + dr, ac + dc))
        coords.extend((r + dr, c + dc) for r, c in shape_cells)
    r0, c0, r1, c1 = bbox_coords(np.array(coords, dtype=int))
    return int(r0), int(c0), int(r1), int(c1)


def _paint_shadow(out: np.ndarray, shape_color: int, shadow_color: int, offset: tuple[int, int]) -> None:
    dr, dc = offset
    h, w = out.shape
    src = out == shape_color
    shifted = np.zeros_like(src)
    r_src = slice(max(0, -dr), h - max(0, dr))
    c_src = slice(max(0, -dc), w - max(0, dc))
    r_dst = slice(max(0, dr), h - max(0, -dr))
    c_dst = slice(max(0, dc), w - max(0, -dc))
    shifted[r_dst, c_dst] = src[r_src, c_src]
    mask = shifted & (out == BG)
    out[mask] = shadow_color


def _crop_pair(inp: np.ndarray, out: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coords = np.argwhere((inp != BG) | (out != BG))
    r0, c0, r1, c1 = bbox_coords(coords)
    return inp[r0 : r1 + 1, c0 : c1 + 1].copy(), out[r0 : r1 + 1, c0 : c1 + 1].copy()


def _make_plain_shape_patch(shape_color: int, shadow_color: int | None) -> tuple[np.ndarray, np.ndarray] | None:
    vertices = _make_vertices(require_manhattan=False)
    cycle = _shape_cycle(vertices)
    cells = _shape_cells(cycle)

    shadow = None
    if shadow_color is not None:
        shadow = _choose_shadow(cycle, cells)
        if shadow is None:
            return None

    r0, c0, r1, c1 = _patch_bbox(cells, cycle, shadow)
    h, w = r1 - r0 + 1, c1 - c0 + 1
    out = np.full((h, w), BG, dtype=int)
    inp = np.full((h, w), BG, dtype=int)

    shifted_vertices = [(r - r0, c - c0) for r, c in cycle]
    for points in _shape_paths(shifted_vertices):
        _draw_polyline_bg_only(out, points, int(shape_color), BG)
    for r, c in shifted_vertices:
        inp[r, c] = int(shape_color)

    if shadow is not None:
        (ar, ac), offset = shadow
        anchor = (ar - r0, ac - c0)
        _paint_shadow(out, int(shape_color), int(shadow_color), offset)
        sa = (anchor[0] + offset[0], anchor[1] + offset[1])
        if not (0 <= sa[0] < h and 0 <= sa[1] < w) or inp[sa] != BG:
            return None
        inp[sa] = int(shadow_color)
        nb_hits = sum(
            1
            for rr, cc in neighbors8(anchor[0], anchor[1])
            if 0 <= rr < h and 0 <= cc < w and inp[rr, cc] == shadow_color
        )
        if nb_hits != 1:
            return None

    if not np.array_equal(main(inp), out):
        return None
    return inp, out


def _make_routed_manhattan_patch(
    shape_color: int, blocker_a: int, blocker_b: int
) -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(ROUTED_PATCH_RETRIES):
        dh = int(np.random.randint(*SPECIAL_H_RANGE))
        dw = int(np.random.randint(*SPECIAL_W_RANGE))
        if dh == dw:
            continue
        block_h = int(np.random.randint(2, dw))
        block_v = int(np.random.randint(2, dh))
        raw = [(0, 0), (dh, dw), (0, block_h), (block_v, 0)]
        pts = _transform_coords(raw)
        shifted = [(r + SPECIAL_PAD, c + SPECIAL_PAD) for r, c in pts]
        h = max(r for r, _ in shifted) + SPECIAL_PAD + 1
        w = max(c for _, c in shifted) + SPECIAL_PAD + 1

        inp = np.full((h, w), BG, dtype=int)
        a, b, x1, x2 = shifted
        inp[a] = int(shape_color)
        inp[b] = int(shape_color)
        inp[x1] = int(blocker_a)
        inp[x2] = int(blocker_b)

        metrics = _variant_metrics(inp)
        if metrics["routed_manhattan_edges"] < 1:
            continue
        out = main(inp)
        if np.array_equal(out, inp):
            continue
        return _crop_pair(inp, out)
    return None


def _sample_palette(shape_count: int) -> tuple[list[int], list[int], tuple[int, int]] | None:
    max_shadows = min(MAX_SHADOWS_CAP, shape_count - 1, len(COLORS) - shape_count - 2)
    if max_shadows < 0:
        return None
    shadow_count = int(np.random.randint(max_shadows + 1))
    total = shape_count + shadow_count + 2
    palette = list(np.random.permutation(COLORS)[:total])
    shape_colors = palette[:shape_count]
    shadow_colors = palette[shape_count : shape_count + shadow_count]
    blocker_colors = (int(palette[-2]), int(palette[-1]))
    return shape_colors, shadow_colors, blocker_colors


def _assign_shadows(shape_count: int, shadow_colors: list[int]) -> dict[int, int]:
    if not shadow_colors:
        return {}
    targets = np.random.choice(list(range(1, shape_count)), size=len(shadow_colors), replace=False).tolist()
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


def _variant_metrics(grid: np.ndarray) -> dict[str, int]:
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    manhattan_edges = 0
    routed_edges = 0
    routed_manhattan_edges = 0
    blocker_colors = 0
    for col in sorted(set(int(v) for v in np.unique(g)) - {bg}):
        cycle, _carried = _ordered_cycle_and_carried(g, col, bg)
        if cycle is None:
            blocker_colors += 1
            continue
        for a, b in _cycle_edges(cycle):
            path = _find_path_in_original(g, a, b, bg, allow={col})
            if path is None:
                continue
            base_len = 2 if _aligned_8(a[0], a[1], b[0], b[1]) else 3
            if base_len == 3:
                manhattan_edges += 1
            if len(path) > base_len:
                routed_edges += 1
                if base_len == 3:
                    routed_manhattan_edges += 1
    return {
        "manhattan_edges": manhattan_edges,
        "routed_edges": routed_edges,
        "routed_manhattan_edges": routed_manhattan_edges,
        "single_pixel_colors": blocker_colors,
    }


def _gen_attempt() -> np.ndarray | None:
    h, w = _sample_canvas_size()
    inp = np.full((h, w), BG, dtype=int)
    out = np.full((h, w), BG, dtype=int)

    shape_count = int(np.random.randint(*SHAPE_COUNT_RANGE))
    palette = _sample_palette(shape_count)
    if palette is None:
        return None
    shape_colors, shadow_colors, blocker_colors = palette
    shadow_map = _assign_shadows(shape_count, shadow_colors)
    total_colors = shape_count + len(shadow_colors) + 2

    routed_patch = _make_routed_manhattan_patch(int(shape_colors[0]), blocker_colors[0], blocker_colors[1])
    if routed_patch is None or not _try_place(inp, out, *routed_patch):
        return None

    for idx, shape_color in enumerate(shape_colors[1:], start=1):
        placed = False
        for _ in range(SHAPE_PLACEMENT_RETRIES):
            patch = _make_plain_shape_patch(int(shape_color), shadow_map.get(idx))
            if patch is None:
                continue
            if _try_place(inp, out, *patch):
                placed = True
                break
        if not placed:
            return None

    metrics = _variant_metrics(inp)
    if metrics["routed_manhattan_edges"] < 1:
        return None
    if get_bg(inp) != BG or get_bg(out) != BG:
        return None
    if len(colors_present(inp, exclude=[BG])) != total_colors:
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"generate_input failed after {GENERATE_RETRIES} attempts")
