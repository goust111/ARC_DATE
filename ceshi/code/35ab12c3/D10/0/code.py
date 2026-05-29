# Variant of 35ab12c3
# Dimension: D10 — 形状种类与拓扑 (meta)
# Change: 六种形状类型（折线/三角形/矩形/菱形/五边形/六边形），通过加权随机选择，包含开放折线和闭合多边形两种拓扑 → 仅开放折线（禁止闭合多边形）
#
# Design analysis for the D10 variant
# ----------------------------------
# a) Concrete behavior and trace
#    - Each non-background shape color now denotes exactly one open polyline: a simple
#      vertex chain with 2-6 vertices and 1-5 straight H/V/45deg segments.
#    - Input keeps only the vertices, plus an optional single 8-neighbor "shadow anchor"
#      pixel of another color adjacent to one endpoint.
#    - Output fills every segment of the open chain and copies the anchor color along a
#      parallel offset path, just like the original shadow behavior.
#    - Original critical step: `main()` sorted same-color points by polar angle and then
#      connected consecutive points cyclically, which implicitly closes any 3+ vertex set.
#      Example: vertices [(0,0),(0,4),(4,4)] become triangle edges AB, BC, CA.
#    - Variant step: build the visibility graph of same-color vertices where an edge
#      exists only if two vertices are H/V/diag aligned and the cells strictly between
#      them are background in the input. Valid shapes must form a simple path graph with
#      two degree-1 endpoints and all other vertices degree 2. The same example is now
#      interpreted as A-B-C only, so output draws AB and BC but never CA.
# b) Complexity calibration
#    - Restricting topology to open paths removes polygon classes, so the generator must
#      still produce visual variety. To keep samples informative, most shapes are bent
#      polylines with 3+ vertices; 2-point straight segments remain possible but are not
#      dominant. This keeps the topology change observable in 2-4 examples.
#    - Chord ambiguity is controlled: the generator rejects any non-consecutive pair of
#      vertices that would also be directly alignable through empty cells, so the solver's
#      path reconstruction stays unambiguous.
# c) Implicit-assumption audit of original `main()`
#    - `_angle_cycle(...)` assumed that a centroid-angle order is a meaningful contour
#      order. That is valid for convex-ish closed polygons, but it is wrong for open
#      paths because it invents a closing edge.
#    - `edges = [(cycle[i], cycle[(i + 1) % len(cycle)]) ...]` hard-coded closure for all
#      3+ point shapes. This is the exact D10 dependency that must be removed.
#    - Shadow carrying from `cycle[0]` is still valid only if the chosen start vertex is
#      an endpoint. The variant solver therefore selects the start among path endpoints,
#      preferring an endpoint that has a foreign-color neighbor.
#    - `_clear_in_original(..., allow={col})` was harmless for polygons because collinear
#      same-color interior points were not meant to split edges. In the open-path variant,
#      allowing same-color interior points would incorrectly skip over intermediate
#      vertices on a straight chain, so direct-path detection now requires strict
#      background-only interiors.
# d) Placement legality
#    - Shapes are still generated output-first as isolated local patches and placed with
#      `random_free_location(..., padding=1, connectivity=8)`, preserving the original
#      non-overlap / one-cell buffer behavior.
#    - Each patch is self-validated: no self-intersection, no direct non-consecutive
#      chord, optional shadow offset in-bounds, and oracle check `main(inp) == out`.
# e) Visual observability
#    - No hidden ordering by color id is used. All reconstruction relies only on visible
#      geometry: aligned vertices, empty cells between them, and endpoint-adjacent shadow
#      markers.
#    - The generator forces at least one bent polyline per grid, so the absence of closed
#      polygon completion is visible rather than degenerating into mostly straight lines.
# f) Visual cleanliness
#    - Distinct groups keep a one-cell gap after output completion, so shadow copies and
#      main lines do not visually merge with unrelated objects.
#    - Only one shadow layer per shape is used, matching the original attempt-2 code and
#      avoiding clutter from nested shadows.
# g) Color inference
#    - Colors retain the original semantics: each shape/distractor uses a unique color,
#      and an optional shadow uses another unique color. There is no color remapping.
# h) New solver assumptions introduced here
#    - Every shape color forms a simple path under the direct-visibility graph.
#    - If a shadow exists, its anchor pixel is adjacent to an endpoint rather than an
#      internal vertex.
#    - `generate_input()` guarantees these assumptions by construction and validates them
#      with the solver oracle before accepting a sample.

import numpy as np

from arc_utils import get_bg as _bg_color
from arc_utils import bbox_coords, colors_present, draw_line, get_bg, neighbors8, paste, random_free_location

DIR8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
BASE_DIRS = [(0, 1), (1, 0), (1, 1), (1, -1)]


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


def _aligned(dr: int, dc: int) -> bool:
    return (dr == 0 and dc != 0) or (dc == 0 and dr != 0) or (dr != 0 and abs(dr) == abs(dc))


def _step_dir(r0: int, c0: int, r1: int, c1: int) -> tuple[int, int]:
    dr = r1 - r0
    dc = c1 - c0
    sr = 0 if dr == 0 else (1 if dr > 0 else -1)
    sc = 0 if dc == 0 else (1 if dc > 0 else -1)
    return sr, sc


def _step(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return _step_dir(a[0], a[1], b[0], b[1])


def _segment_cells(r0: int, c0: int, r1: int, c1: int):
    sr, sc = _step_dir(r0, c0, r1, c1)
    steps = max(abs(r1 - r0), abs(c1 - c0))
    for k in range(steps + 1):
        yield r0 + sr * k, c0 + sc * k


def _gen_segment_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    return list(_segment_cells(a[0], a[1], b[0], b[1]))


def _clear_in_original(g: np.ndarray, r0: int, c0: int, r1: int, c1: int, bg: int, allow: set[int]) -> bool:
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


def _clear_bg_between(g: np.ndarray, r0: int, c0: int, r1: int, c1: int, bg: int) -> bool:
    first = True
    for rr, cc in _segment_cells(r0, c0, r1, c1):
        if first:
            first = False
            continue
        if rr == r1 and cc == c1:
            break
        if int(g[rr, cc]) != bg:
            return False
    return True


def _draw_line_bg_only(out: np.ndarray, r0: int, c0: int, r1: int, c1: int, col: int, bg: int) -> None:
    for rr, cc in _segment_cells(r0, c0, r1, c1):
        if out[rr, cc] == bg:
            out[rr, cc] = col


def _has_foreign_neighbor(g: np.ndarray, r: int, c: int, bg: int, col: int) -> bool:
    h, w = g.shape
    for dr, dc in DIR8:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            v = int(g[rr, cc])
            if v != bg and v != col:
                return True
    return False


def _build_path_graph(points_rc: list[tuple[int, int]], g: np.ndarray, bg: int):
    pts = sorted(points_rc)
    adj = {p: [] for p in pts}
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            (r0, c0), (r1, c1) = pts[i], pts[j]
            if not _aligned_8(r0, c0, r1, c1):
                continue
            if not _clear_bg_between(g, r0, c0, r1, c1, bg):
                continue
            adj[pts[i]].append(pts[j])
            adj[pts[j]].append(pts[i])

    degrees = [len(adj[p]) for p in pts]
    if len(pts) == 2:
        if degrees != [1, 1]:
            return None
    else:
        if degrees.count(1) != 2 or degrees.count(2) != len(pts) - 2:
            return None
    endpoints = sorted([p for p in pts if len(adj[p]) == 1])
    return adj, endpoints


def _traverse_path(adj: dict[tuple[int, int], list[tuple[int, int]]], start: tuple[int, int]) -> list[tuple[int, int]]:
    ordered = [start]
    prev = None
    cur = start
    while True:
        nxts = [p for p in adj[cur] if p != prev]
        if not nxts:
            break
        if len(nxts) != 1:
            return []
        nxt = nxts[0]
        ordered.append(nxt)
        prev, cur = cur, nxt
    return ordered


def _shape_info(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    info: list[dict] = []
    for col in sorted(set(int(v) for v in np.unique(g)) - {bg}):
        pts_rc = [tuple(map(int, p)) for p in np.argwhere(g == col)]
        if len(pts_rc) < 2:
            continue
        graph = _build_path_graph(pts_rc, g, bg)
        if graph is None:
            continue
        adj, endpoints = graph
        seed_endpoints = [p for p in endpoints if _has_foreign_neighbor(g, p[0], p[1], bg, col)]
        start = min(seed_endpoints or endpoints)
        ordered = _traverse_path(adj, start)
        if len(ordered) != len(pts_rc):
            continue
        dirs = [_step(ordered[i], ordered[i + 1]) for i in range(len(ordered) - 1)]
        info.append(
            {
                "color": col,
                "vertices": ordered,
                "endpoints": endpoints,
                "edge_dirs": dirs,
                "is_bent": len(set(dirs)) > 1,
            }
        )
    return info


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _bg_color(g)

    out = g.copy()

    for shape in _shape_info(g):
        col = int(shape["color"])
        ordered = shape["vertices"]
        start = ordered[0]

        carried: list[tuple[int, int, int]] = []
        sr0, sc0 = start
        for drn, dcn in DIR8:
            rr = sr0 + drn
            cc = sc0 + dcn
            if 0 <= rr < h and 0 <= cc < w:
                v = int(g[rr, cc])
                if v != bg and v != col:
                    carried.append((drn, dcn, v))

        drawable_edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
        for a, b in zip(ordered, ordered[1:]):
            if not _aligned_8(a[0], a[1], b[0], b[1]):
                continue
            if not _clear_bg_between(g, a[0], a[1], b[0], b[1], bg):
                continue
            drawable_edges.append((a, b))

        for (r0, c0), (r1, c1) in drawable_edges:
            _draw_line_bg_only(out, r0, c0, r1, c1, col, bg)

        for (r0, c0), (r1, c1) in drawable_edges:
            drs, dcs = _step_dir(r0, c0, r1, c1)
            for drn, dcn, ncol in carried:
                if drs * dcn - dcs * drn == 0:
                    continue
                rr0 = r0 + drn
                cc0 = c0 + dcn
                rr1 = r1 + drn
                cc1 = c1 + dcn
                if not (0 <= rr0 < h and 0 <= cc0 < w and 0 <= rr1 < h and 0 <= cc1 < w):
                    continue
                if not _aligned_8(rr0, cc0, rr1, cc1):
                    continue
                if not _clear_in_original(g, rr0, cc0, rr1, cc1, bg, allow=set()):
                    continue
                _draw_line_bg_only(out, rr0, cc0, rr1, cc1, ncol, bg)

    return out


# --------------- generator (D10 open-polyline variant) ---------------

# ---- CFG: canvas and palette ----
BG = 0
COLORS = [1, 2, 3, 4, 6, 7, 8]
CANVAS_H_RANGE = (15, 23)
CANVAS_W_JITTER = (-3, 4)
CANVAS_W_CLIP = (15, 22)

# ---- CFG: structural counts ----
SHAPE_COUNT_RANGE = (2, 5)
MAX_SHADOWS_CAP = 3
DISTRACTOR_PROB = 0.5
PLACEMENT_PADDING = 1
PLACEMENT_PAD_CONNECTIVITY = 8

# ---- CFG: open-polyline geometry ----
LINE_VERTICES = np.array([2, 3, 4, 5, 6], dtype=int)
LINE_VERTEX_WEIGHTS = np.array([0.10, 0.28, 0.28, 0.22, 0.12], dtype=float)
BENT_SHAPE_PROB = 0.8
POLYLINE_MIN_SEG_LEN = 2
POLYLINE_TOTAL_LENGTH_RANGE = (7, 15)   # total length 7-14
POLYLINE_BBOX_MAX = 12
POLYLINE_DIR_CHOICES = (2, 3)
POLYLINE_BUILD_RETRIES = 100

# ---- CFG: distractor geometry ----
DISTRACTOR_COUNT_RANGE = (1, 4)
DISTRACTOR_BBOX_RANGE = (2, 5)
DISTRACTOR_RETRIES = 50

# ---- CFG: retry budgets ----
SHAPE_PLACEMENT_RETRIES = 40
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


def _sample_lengths(n_segments: int) -> list[int] | None:
    low = max(POLYLINE_TOTAL_LENGTH_RANGE[0], POLYLINE_MIN_SEG_LEN * n_segments)
    high = POLYLINE_TOTAL_LENGTH_RANGE[1]
    if low >= high:
        return None
    total = int(np.random.randint(low, high))
    lengths: list[int] = []
    remaining = total
    for i in range(n_segments - 1):
        segs_left_after = n_segments - i - 1
        hi = remaining - POLYLINE_MIN_SEG_LEN * segs_left_after
        length = int(np.random.randint(POLYLINE_MIN_SEG_LEN, hi + 1))
        lengths.append(length)
        remaining -= length
    lengths.append(int(remaining))
    return lengths


def _has_vertex_inside_segment(a: tuple[int, int], b: tuple[int, int], vertices: list[tuple[int, int]]) -> bool:
    interior = set(_gen_segment_cells(a, b)[1:-1])
    if not interior:
        return False
    return any(v in interior for v in vertices)


def _has_visible_chord(vertices: list[tuple[int, int]]) -> bool:
    n = len(vertices)
    for i in range(n):
        for j in range(i + 1, n):
            if j == i + 1:
                continue
            a = vertices[i]
            b = vertices[j]
            if not _aligned_8(a[0], a[1], b[0], b[1]):
                continue
            others = [vertices[k] for k in range(n) if k not in (i, j)]
            if not _has_vertex_inside_segment(a, b, others):
                return True
    return False


def _self_intersects(vertices: list[tuple[int, int]]) -> bool:
    seen: set[tuple[int, int]] = set()
    for idx, (a, b) in enumerate(zip(vertices, vertices[1:])):
        seg = _gen_segment_cells(a, b)
        for k, cell in enumerate(seg):
            if idx > 0 and k == 0:
                continue
            if cell in seen:
                return True
            seen.add(cell)
    return False


def _valid_open_polyline(vertices: list[tuple[int, int]], require_turn: bool = False) -> bool:
    if len(vertices) < 2:
        return False
    if any(not _aligned_8(a[0], a[1], b[0], b[1]) for a, b in zip(vertices, vertices[1:])):
        return False
    if _self_intersects(vertices):
        return False
    if _has_visible_chord(vertices):
        return False
    dirs = [_step(a, b) for a, b in zip(vertices, vertices[1:])]
    if require_turn and len(set(dirs)) < 2:
        return False
    coords = np.array(vertices, dtype=int)
    r0, c0, r1, c1 = bbox_coords(coords)
    if r1 - r0 + 1 > POLYLINE_BBOX_MAX or c1 - c0 + 1 > POLYLINE_BBOX_MAX:
        return False
    return True


def _sample_vertex_count(require_turn: bool) -> int:
    if require_turn:
        return int(np.random.choice(LINE_VERTICES[1:], p=LINE_VERTEX_WEIGHTS[1:] / LINE_VERTEX_WEIGHTS[1:].sum()))
    return int(np.random.choice(LINE_VERTICES, p=LINE_VERTEX_WEIGHTS / LINE_VERTEX_WEIGHTS.sum()))


def _build_open_polyline(require_turn: bool = False) -> list[tuple[int, int]] | None:
    n_vertices = _sample_vertex_count(require_turn)
    if n_vertices == 2:
        length = _sample_lengths(1)
        if length is None:
            return None
        return _transform_coords([(0, 0), (0, length[0])])

    lengths = _sample_lengths(n_vertices - 1)
    if lengths is None:
        return None

    for _ in range(POLYLINE_BUILD_RETRIES):
        dir_count = int(np.random.randint(POLYLINE_DIR_CHOICES[0], POLYLINE_DIR_CHOICES[1] + 1))
        allowed_dirs = [BASE_DIRS[i] for i in np.random.permutation(len(BASE_DIRS))[:dir_count]]
        vertices = [(0, 0)]
        occupied = {(0, 0)}
        prev_dir = None
        ok = True

        for length in lengths:
            candidates = [d for d in allowed_dirs if d != prev_dir]
            if not candidates:
                ok = False
                break
            np.random.shuffle(candidates)
            placed = False
            for dr, dc in candidates:
                cur = vertices[-1]
                nxt = (cur[0] + dr * length, cur[1] + dc * length)
                seg = _gen_segment_cells(cur, nxt)
                if any(cell in occupied for cell in seg[1:]):
                    continue
                trial_vertices = vertices + [nxt]
                if _has_visible_chord(trial_vertices):
                    continue
                vertices.append(nxt)
                occupied.update(seg[1:])
                prev_dir = (dr, dc)
                placed = True
                break
            if not placed:
                ok = False
                break

        if not ok:
            continue

        transformed = _transform_coords(vertices)
        if _valid_open_polyline(transformed, require_turn=require_turn):
            return transformed
    return None


def _make_vertices(require_turn: bool = False) -> list[tuple[int, int]] | None:
    return _build_open_polyline(require_turn=require_turn)


def _shape_edges(vertices: list[tuple[int, int]]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    return list(zip(vertices, vertices[1:]))


def _shape_cells(vertices: list[tuple[int, int]]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for a, b in _shape_edges(vertices):
        cells.update(_gen_segment_cells(a, b))
    return cells


def _edge_dirs(vertices: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return [_step(a, b) for a, b in _shape_edges(vertices)]


def _choose_shadow(
    vertices: list[tuple[int, int]],
    shape_cells: set[tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    endpoints = sorted([vertices[0], vertices[-1]])
    directions = _edge_dirs(vertices)
    for anchor in endpoints:
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
        if candidates:
            return anchor, candidates[int(np.random.randint(len(candidates)))]
    return None


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


def _make_shape_patch(shape_color: int, shadow_color: int | None, require_turn: bool = False):
    vertices = _make_vertices(require_turn=require_turn)
    if vertices is None:
        return None
    cells = _shape_cells(vertices)

    shadow = None
    if shadow_color is not None:
        shadow = _choose_shadow(vertices, cells)
        if shadow is None:
            return None

    r0, c0, r1, c1 = _patch_bbox(cells, vertices, shadow)
    h, w = r1 - r0 + 1, c1 - c0 + 1
    out = np.full((h, w), BG, dtype=int)
    inp = np.full((h, w), BG, dtype=int)

    shifted_vertices = [(r - r0, c - c0) for r, c in vertices]
    for a, b in _shape_edges(shifted_vertices):
        draw_line(out, a[0], a[1], end_r=b[0], end_c=b[1], color=shape_color)
    for r, c in shifted_vertices:
        inp[r, c] = shape_color

    if shadow is not None:
        (ar, ac), offset = shadow
        anchor = (ar - r0, ac - c0)
        _paint_shadow(out, shape_color, int(shadow_color), offset)
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


def _sample_distractor_coords(h: int, w: int, count: int) -> list[tuple[int, int]] | None:
    coords: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    while len(coords) < count:
        cand = (int(np.random.randint(h)), int(np.random.randint(w)))
        if cand in seen:
            continue
        coords.append(cand)
        seen.add(cand)
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            if _aligned(coords[j][0] - coords[i][0], coords[j][1] - coords[i][1]):
                return None
    return coords


def _make_distractor_patch(color: int) -> tuple[np.ndarray, np.ndarray]:
    count = int(np.random.randint(*DISTRACTOR_COUNT_RANGE))
    if count == 1:
        patch = np.full((1, 1), color, dtype=int)
        return patch.copy(), patch.copy()
    for _ in range(DISTRACTOR_RETRIES):
        h = int(np.random.randint(*DISTRACTOR_BBOX_RANGE))
        w = int(np.random.randint(*DISTRACTOR_BBOX_RANGE))
        coords = _sample_distractor_coords(h, w, count)
        if coords is None:
            continue
        patch = np.full((h, w), BG, dtype=int)
        for r, c in coords:
            patch[r, c] = color
        return patch.copy(), patch.copy()
    patch = np.full((1, 1), color, dtype=int)
    return patch.copy(), patch.copy()


def _sample_palette(shape_count: int, has_distractor: bool):
    max_shadows = min(MAX_SHADOWS_CAP, shape_count, len(COLORS) - shape_count - int(has_distractor))
    if max_shadows < 0:
        return None
    shadow_count = int(np.random.randint(max_shadows + 1))
    total = shape_count + shadow_count + int(has_distractor)
    palette = list(np.random.permutation(COLORS)[:total])
    shape_colors = palette[:shape_count]
    shadow_colors = palette[shape_count : shape_count + shadow_count]
    distractor_color = int(palette[-1]) if has_distractor else None
    return shape_colors, shadow_colors, distractor_color


def _assign_shadows(shape_count: int, shadow_colors: list[int]) -> dict[int, int]:
    if not shadow_colors:
        return {}
    targets = np.random.choice(shape_count, size=len(shadow_colors), replace=False).tolist()
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
    H = int(np.random.randint(*CANVAS_H_RANGE))
    W = int(np.clip(H + np.random.randint(*CANVAS_W_JITTER), *CANVAS_W_CLIP))
    return H, W


def _gen_attempt() -> np.ndarray | None:
    H, W = _sample_canvas_size()
    out = np.full((H, W), BG, dtype=int)
    inp = np.full((H, W), BG, dtype=int)

    has_distractor = bool(np.random.rand() < DISTRACTOR_PROB)
    shape_count = int(np.random.randint(*SHAPE_COUNT_RANGE))

    palette = _sample_palette(shape_count, has_distractor)
    if palette is None:
        return None
    shape_colors, shadow_colors, distractor_color = palette
    shadow_map = _assign_shadows(shape_count, shadow_colors)
    forced_bent = {int(np.random.randint(shape_count))}

    total_colors = shape_count + len(shadow_colors) + int(has_distractor)

    for idx, shape_color in enumerate(shape_colors):
        require_turn = idx in forced_bent or bool(np.random.rand() < BENT_SHAPE_PROB)
        placed = False
        for _ in range(SHAPE_PLACEMENT_RETRIES):
            patch = _make_shape_patch(int(shape_color), shadow_map.get(idx), require_turn=require_turn)
            if patch is None:
                continue
            if _try_place(inp, out, *patch):
                placed = True
                break
        if not placed:
            return None

    if has_distractor and distractor_color is not None:
        if not _try_place(inp, out, *_make_distractor_patch(distractor_color)):
            return None

    if get_bg(inp) != BG or get_bg(out) != BG:
        return None
    if len(colors_present(inp, exclude=[BG])) != total_colors:
        return None
    shapes = _shape_info(inp)
    if len(shapes) != shape_count:
        return None
    if not any(shape["is_bent"] for shape in shapes):
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
