# Variant of 35ab12c3
# Dimension: D3 — 顶点排序与连接拓扑 (path)
# Change: 极角排序（angle_cycle）——所有同色顶点按围绕质心的极角排序，2个顶点连直线段，3+个顶点按顺序连成闭合多边形... → 最近邻贪心连接

import math

import numpy as np

from arc_utils import get_bg as _bg_color
from arc_utils import bbox_coords, colors_present, draw_line, get_bg, neighbors8, paste, random_free_location

# Design analysis for the D3 variant:
# a) Concrete behavior:
#    Each non-background color still denotes one vertex set. The chosen start vertex
#    is unchanged: prefer a vertex touching a different-color neighbor (shadow anchor),
#    else the smallest (r,c) vertex. The change is only the traversal/order:
#    from the current vertex, repeatedly connect to the nearest unvisited vertex
#    (Euclidean distance squared, ties by (r,c)). The output is now an OPEN path,
#    not a closed angle-sorted polygon. Example:
#      vertices {(0,0),(0,4),(4,4),(4,8)}, start=(0,0)
#      original: angle cycle closes a 4-edge loop
#      variant:  (0,0)->(0,4)->(4,4)->(4,8), no closing edge.
#    The carried shadow still comes from the same start vertex and is copied along
#    each drawable greedy segment.
# b) Complexity calibration:
#    If the generator kept using rectangles/diamonds, many samples would still look
#    like ordinary polygons and the D3 change would be visually weak. The generator
#    therefore now creates 3-5 vertex monotone turning paths. They are simple enough
#    to parse visually, but the old angle-cycle solver would add a wrong closing edge
#    or visit vertices in another order, so the difference is observable.
# c) Hidden-assumption audit in main():
#    - The old _angle_cycle() assumed the correct topology was a centroid-based cycle.
#      That assumption is removed; greedy order depends only on pairwise distances.
#    - The old "len(cycle)>2 => add closing edge" is invalid in the variant and is
#      replaced by consecutive-path edges only.
#    - Start-vertex shadow carryover remains valid only if the generator places any
#      shadow pixel next to the greedy start; the generator enforces that.
#    - _clear_in_original() still assumes intermediate cells are background or same
#      color. The generator now rejects any path whose consecutive segment would pass
#      through another non-endpoint vertex, avoiding ambiguous paths.
# d) Placement legality:
#    Patches are still built output-first, then placed with a 1-cell padded
#    random_free_location() call. Each patch contains only one isolated shape (plus
#    optional shadow), so solver segment tests see clean background between groups.
# e) Visual observability:
#    The rule never depends on color ids. Nearest-vertex decisions are made from
#    visible geometry only, and the generator rejects distance ties because humans
#    could not infer hidden tie-breaks from 2-4 examples.
# f) Visual cleanliness:
#    Paths are open and monotone, so they do not stack many overlaps in one patch.
#    Distinct objects remain separated by the existing placement padding.
# g) Color inference:
#    No new color mapping is introduced. Shape color and optional shadow color keep
#    the original meaning; only the connection topology changes.

DIR8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


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


def _draw_line_bg_only(out: np.ndarray, r0: int, c0: int, r1: int, c1: int, col: int, bg: int) -> None:
    for rr, cc in _segment_cells(r0, c0, r1, c1):
        if out[rr, cc] == bg:
            out[rr, cc] = col


def _angle_cycle(points_rc: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cr = sum(r for r, _ in points_rc) / len(points_rc)
    cc = sum(c for _, c in points_rc) / len(points_rc)

    def key(p: tuple[int, int]):
        r, c = p
        ang = math.atan2(r - cr, c - cc)
        rad = (r - cr) ** 2 + (c - cc) ** 2
        return (ang, rad, r, c)

    return sorted(points_rc, key=key)


def _dist2(a: tuple[int, int], b: tuple[int, int]) -> int:
    dr = a[0] - b[0]
    dc = a[1] - b[1]
    return dr * dr + dc * dc


def _greedy_path(points_rc: list[tuple[int, int]], start: tuple[int, int]) -> list[tuple[int, int]]:
    path = [start]
    unvisited = {p for p in points_rc}
    unvisited.remove(start)
    current = start
    while unvisited:
        nxt = min(unvisited, key=lambda p: (_dist2(current, p), p[0], p[1]))
        path.append(nxt)
        unvisited.remove(nxt)
        current = nxt
    return path


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


def _choose_start(g: np.ndarray, pts_rc: list[tuple[int, int]], col: int, bg: int) -> tuple[int, int]:
    h, w = g.shape
    seeded = []
    for r, c in pts_rc:
        for drn, dcn in DIR8:
            rr = r + drn
            cc = c + dcn
            if 0 <= rr < h and 0 <= cc < w:
                v = int(g[rr, cc])
                if v != bg and v != col:
                    seeded.append((r, c))
                    break
    return min(seeded) if seeded else min(pts_rc)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _bg_color(g)

    out = g.copy()

    colors = sorted(set(int(v) for v in np.unique(g)) - {bg})
    for col in colors:
        pts_rc = [tuple(map(int, p)) for p in np.argwhere(g == col)]
        if len(pts_rc) < 2:
            continue

        start = _choose_start(g, pts_rc, col, bg)
        path = _greedy_path(pts_rc, start)

        carried: list[tuple[int, int, int]] = []
        sr0, sc0 = path[0]
        for drn, dcn in DIR8:
            rr = sr0 + drn
            cc = sc0 + dcn
            if 0 <= rr < h and 0 <= cc < w:
                v = int(g[rr, cc])
                if v != bg and v != col:
                    carried.append((drn, dcn, v))

        edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]

        drawable_edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
        for (r0, c0), (r1, c1) in edges:
            if not _aligned_8(r0, c0, r1, c1):
                continue
            if not _clear_in_original(g, r0, c0, r1, c1, bg, allow={col}):
                continue
            drawable_edges.append(((r0, c0), (r1, c1)))

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


# --------------- generator ---------------

BG = 0
COLORS = [1, 2, 3, 4, 6, 7, 8]
CANVAS_H_RANGE = (15, 23)
CANVAS_W_JITTER = (-3, 4)
CANVAS_W_CLIP = (15, 22)

SHAPE_COUNT_RANGE = (2, 5)
MAX_SHADOWS_CAP = 3
DISTRACTOR_PROB = 0.5
PLACEMENT_PADDING = 1
PLACEMENT_PAD_CONNECTIVITY = 8

PATH_VERTICES_RANGE = (3, 6)      # 3-5 vertices: keeps D3 visible on every shape
PATH_TOTAL_LENGTH_MAX = 10
PATH_MIN_GAP = 2
PATH_BUILD_RETRIES = 80
PATH_DIRECTIONS = [(0, 1), (1, 0), (1, 1)]

DISTRACTOR_COUNT_RANGE = (1, 4)
DISTRACTOR_BBOX_RANGE = (2, 5)
DISTRACTOR_RETRIES = 50

SHAPE_PLACEMENT_RETRIES = 40
GENERATE_RETRIES = 300


def _aligned(dr: int, dc: int) -> bool:
    return (dr == 0 and dc != 0) or (dc == 0 and dr != 0) or (dr != 0 and abs(dr) == abs(dc))


def _step(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    dr = b[0] - a[0]
    dc = b[1] - a[1]
    return (0 if dr == 0 else (1 if dr > 0 else -1), 0 if dc == 0 else (1 if dc > 0 else -1))


def _gen_segment_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    drs, dcs = _step(a, b)
    steps = max(abs(b[0] - a[0]), abs(b[1] - a[1]))
    return [(a[0] + drs * k, a[1] + dcs * k) for k in range(steps + 1)]


def _path_edges(vertices: list[tuple[int, int]]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    return [(vertices[i], vertices[i + 1]) for i in range(len(vertices) - 1)]


def _edge_dirs(vertices: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return [_step(a, b) for a, b in _path_edges(vertices)]


def _shape_cells(vertices: list[tuple[int, int]]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for a, b in _path_edges(vertices):
        cells.update(_gen_segment_cells(a, b))
    return cells


def _edges_to_cells(edges: list[tuple[tuple[int, int], tuple[int, int]]]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for a, b in edges:
        cells.update(_gen_segment_cells(a, b))
    return cells


def _angle_edges(vertices: list[tuple[int, int]]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    cycle = _angle_cycle(vertices)
    if len(cycle) == 2:
        return [(cycle[0], cycle[1])]
    return [(cycle[i], cycle[(i + 1) % len(cycle)]) for i in range(len(cycle))]


def _segment_hits_other_vertex(
    a: tuple[int, int],
    b: tuple[int, int],
    vertices: list[tuple[int, int]],
) -> bool:
    verts = set(vertices)
    for rr, cc in _gen_segment_cells(a, b)[1:-1]:
        if (rr, cc) in verts:
            return True
    return False


def _sample_gaps(n_segments: int) -> list[int] | None:
    min_total = PATH_MIN_GAP * n_segments
    if min_total > PATH_TOTAL_LENGTH_MAX:
        return None
    total = int(np.random.randint(min_total, PATH_TOTAL_LENGTH_MAX + 1))
    remaining = total
    gaps = []
    for i in range(n_segments):
        steps_left = n_segments - 1 - i
        hi = remaining - PATH_MIN_GAP * steps_left
        gap = int(np.random.randint(PATH_MIN_GAP, hi + 1))
        gaps.append(gap)
        remaining -= gap
    return gaps


def _build_monotone_vertices() -> list[tuple[int, int]] | None:
    n_vertices = int(np.random.randint(*PATH_VERTICES_RANGE))
    n_segments = n_vertices - 1
    gaps = _sample_gaps(n_segments)
    if gaps is None:
        return None

    directions: list[tuple[int, int]] = []
    for i in range(n_segments):
        candidates = list(PATH_DIRECTIONS)
        if directions:
            prev = directions[-1]
            if len(candidates) > 1 and np.random.rand() < 0.75:
                candidates = [d for d in candidates if d != prev]
            if not candidates:
                candidates = list(PATH_DIRECTIONS)
            if i == n_segments - 1 and len(set(directions)) == 1:
                forced = [d for d in candidates if d != prev]
                if forced:
                    candidates = forced
        direction = candidates[int(np.random.randint(len(candidates)))]
        directions.append(direction)

    if len(set(directions)) < 2:
        return None

    vertices = [(0, 0)]
    for (dr, dc), gap in zip(directions, gaps):
        last = vertices[-1]
        vertices.append((last[0] + dr * gap, last[1] + dc * gap))

    if np.random.rand() < 0.5:
        vertices = [(c, r) for r, c in vertices]
    return vertices


def _is_variant_distinct(vertices: list[tuple[int, int]]) -> bool:
    return _edges_to_cells(_path_edges(vertices)) != _edges_to_cells(_angle_edges(vertices))


def _make_vertices() -> list[tuple[int, int]] | None:
    for _ in range(PATH_BUILD_RETRIES):
        vertices = _build_monotone_vertices()
        if vertices is None:
            continue
        if vertices[0] != min(vertices):
            continue
        greedy, unique_nearest = _greedy_path_trace(vertices, vertices[0])
        if greedy != vertices or not unique_nearest:
            continue
        if any(not _aligned_8(a[0], a[1], b[0], b[1]) for a, b in _path_edges(vertices)):
            continue
        if any(_segment_hits_other_vertex(a, b, vertices) for a, b in _path_edges(vertices)):
            continue
        if not _is_variant_distinct(vertices):
            continue
        return vertices
    return None


def _choose_shadow(
    vertices: list[tuple[int, int]],
    shape_cells: set[tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    anchor = vertices[0]
    directions = _edge_dirs(vertices)
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


def _make_shape_patch(shape_color: int, shadow_color: int | None) -> tuple[np.ndarray, np.ndarray] | None:
    vertices = _make_vertices()
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
    for a, b in _path_edges(shifted_vertices):
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


def _sample_palette(shape_count: int, has_distractor: bool) -> tuple[list[int], list[int], int | None] | None:
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

    total_colors = shape_count + len(shadow_colors) + int(has_distractor)

    for idx, shape_color in enumerate(shape_colors):
        placed = False
        for _ in range(SHAPE_PLACEMENT_RETRIES):
            patch = _make_shape_patch(int(shape_color), shadow_map.get(idx))
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
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"generate_input failed after {GENERATE_RETRIES} attempts")
