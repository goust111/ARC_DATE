# Combo variant of 35ab12c3 (index 35)
# combo_spec: D3/0+D11/0+D17/1
# Name: 高密度5-7组+颜色复用+最近邻贪心

import math

import numpy as np

from arc_utils import get_bg as _bg_color
from arc_utils import bbox_coords, colors_present, draw_line, get_bg, neighbors8, paste

# Design analysis for the fused combo variant
# -------------------------------------------
# a) Concrete behavior and trace:
#    Original 35ab12c3 treated every non-background color as one global group and,
#    for 3+ vertices, connected them by a closed angle-sorted cycle. In this combo,
#    the same shape color may appear in multiple distant local groups, so the solver
#    must first split one color into spatially separate "clear aligned" components.
#    Then each component is solved independently by a nearest-neighbor greedy path,
#    not by a centroid cycle. Example:
#      color 6 vertices = {(1,1),(1,5),(4,5)} and {(10,10),(13,10),(13,13)}
#      original logic would merge all six points into one bad cycle;
#      fused logic builds two components, then each component becomes an open greedy
#      path such as (1,1)->(1,5)->(4,5), with its own carried shadow offset.
# b) Complexity calibration:
#    To make the D3 change visually obvious under dense 5-7 group layouts, the
#    generator emits only compact monotone open paths whose greedy traversal is
#    unique and measurably different from an angle-cycle closure. Using rectangles
#    or diamonds here would often hide the D3 change inside shapes that still look
#    polygonal. Reused colors are limited to 1-2 surplus groups, which is enough to
#    force spatial grouping without turning the board into color noise.
# c) Hidden-assumption audit:
#    - "color == one group" is invalid once colors are reused, so main() must split
#      each color into local components before choosing a start vertex or drawing.
#    - "3+ vertices => close the cycle" is invalid once D3 switches to greedy path.
#    - The shadow-carry rule still assumes the chosen start vertex is unambiguous.
#      The generator enforces that only the greedy start has a different-colored
#      8-neighbor in the input patch.
#    - The component splitter assumes distinct same-color groups have no cross-group
#      clear aligned candidate edge. The generator enforces this during placement.
# d) Placement legality:
#    Patches are still built output-first and oracle-checked locally. Global dense
#    placement uses largest-first pasting with a 1-cell halo around final output
#    footprints, plus an extra constraint for same-color groups: across two reused
#    groups, no vertex pair may be H/V/45 aligned, otherwise the solver could merge
#    them into one component.
# e) Visual observability:
#    A human can infer the rule from examples because every changed decision uses
#    visible geometry only: local component separation, nearest visible vertex, and
#    repeated colors appearing in multiple distant clusters. No color-id ordering or
#    hidden tie-break beyond visible (distance, then row, then col) is needed.
# f) Visual cleanliness:
#    Dense samples use a larger canvas but smaller paths, low distractor frequency,
#    and at most two shadow colors. This keeps the board crowded but readable, with
#    clear separation between groups despite color reuse.
# g) Inferability of changed behavior:
#    The generator rejects path templates whose greedy open-path fill would equal the
#    original angle-cycle fill. It also requires at least one reused shape color in
#    every sample, so both D3 and D11 are structurally present rather than latent.

DIR8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _aligned_8(r0: int, c0: int, r1: int, c1: int) -> bool:
    dr = r1 - r0
    dc = c1 - c0
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
    unvisited = set(points_rc)
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
    unvisited = set(points_rc)
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


def _path_edges(vertices: list[tuple[int, int]]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    return [(vertices[i], vertices[i + 1]) for i in range(len(vertices) - 1)]


def _edges_to_cells(edges: list[tuple[tuple[int, int], tuple[int, int]]]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for a, b in edges:
        cells.update(_segment_cells(a[0], a[1], b[0], b[1]))
    return cells


def _choose_start(g: np.ndarray, pts_rc: list[tuple[int, int]], col: int, bg: int) -> tuple[int, int]:
    h, w = g.shape
    seeded: list[tuple[int, int]] = []
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


def _same_color_groups(g: np.ndarray, col: int, bg: int) -> list[list[tuple[int, int]]]:
    pts_rc = [tuple(map(int, p)) for p in np.argwhere(g == col)]
    n = len(pts_rc)
    if n == 0:
        return []
    if n == 1:
        return [[pts_rc[0]]]

    adj = [set() for _ in range(n)]
    for i in range(n):
        r0, c0 = pts_rc[i]
        for j in range(i + 1, n):
            r1, c1 = pts_rc[j]
            if not _aligned_8(r0, c0, r1, c1):
                continue
            if not _clear_in_original(g, r0, c0, r1, c1, bg, allow={col}):
                continue
            adj[i].add(j)
            adj[j].add(i)

    groups: list[list[tuple[int, int]]] = []
    seen = [False] * n
    for i in range(n):
        if seen[i]:
            continue
        stack = [i]
        seen[i] = True
        comp: list[tuple[int, int]] = []
        while stack:
            cur = stack.pop()
            comp.append(pts_rc[cur])
            for nxt in adj[cur]:
                if not seen[nxt]:
                    seen[nxt] = True
                    stack.append(nxt)
        groups.append(sorted(comp))

    groups.sort(key=min)
    return groups


def _drawable_edges(
    g: np.ndarray,
    edges: list[tuple[tuple[int, int], tuple[int, int]]],
    col: int,
    bg: int,
    allow: set[int],
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    drawable = []
    for (r0, c0), (r1, c1) in edges:
        if not _aligned_8(r0, c0, r1, c1):
            continue
        if not _clear_in_original(g, r0, c0, r1, c1, bg, allow=allow):
            continue
        drawable.append(((r0, c0), (r1, c1)))
    return drawable


def _greedy_cells_for_group(g: np.ndarray, pts_rc: list[tuple[int, int]], col: int, bg: int) -> set[tuple[int, int]]:
    start = _choose_start(g, pts_rc, col, bg)
    path = _greedy_path(pts_rc, start)
    return _edges_to_cells(_drawable_edges(g, _path_edges(path), col, bg, allow={col}))


def _angle_cells_for_group(g: np.ndarray, pts_rc: list[tuple[int, int]], col: int, bg: int) -> set[tuple[int, int]]:
    cycle = _angle_cycle(pts_rc)
    if len(cycle) < 2:
        return set()
    if len(cycle) == 2:
        edges = [(cycle[0], cycle[1])]
    else:
        edges = [(cycle[i], cycle[(i + 1) % len(cycle)]) for i in range(len(cycle))]
    return _edges_to_cells(_drawable_edges(g, edges, col, bg, allow={col}))


def _shape_group_stats(grid: np.ndarray) -> dict[str, int]:
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    groups_by_color: dict[int, list[list[tuple[int, int]]]] = {}
    greedy_distinct = 0

    for col in colors_present(g, exclude=[bg]):
        groups = [grp for grp in _same_color_groups(g, col, bg) if len(grp) >= 2]
        if groups:
            groups_by_color[int(col)] = groups
        for grp in groups:
            if _greedy_cells_for_group(g, grp, col, bg) != _angle_cells_for_group(g, grp, col, bg):
                greedy_distinct += 1

    shape_group_count = sum(len(groups) for groups in groups_by_color.values())
    reused_color_count = sum(1 for groups in groups_by_color.values() if len(groups) >= 2)
    reused_group_surplus = sum(len(groups) - 1 for groups in groups_by_color.values() if len(groups) >= 2)
    return {
        "shape_group_count": int(shape_group_count),
        "reused_color_count": int(reused_color_count),
        "reused_group_surplus": int(reused_group_surplus),
        "greedy_distinct_group_count": int(greedy_distinct),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Overview:
        Split each non-background color into local aligned-clear components, then solve
        each component independently as an open nearest-neighbor greedy path and copy
        the chosen start vertex's carried shadow along parallel offsets.

    Input structure:
        A 26-31ish grid with 5-7 shape groups. Shape colors may be reused across
        multiple distant groups. Each shape group is 3-5 same-colored vertex pixels
        whose intended greedy consecutive pairs are H/V/45-degree aligned. Some
        groups have a single different-colored shadow-anchor pixel next to the greedy
        start vertex. Optional distractors are 1-3 pixels that are pairwise non-aligned.

    Rule:
        For each color, first build local same-color components using only clear
        H/V/45 candidate connections in the input. For every component with at least
        two points, choose the start vertex: prefer a vertex touching a different-
        colored neighbor, otherwise the smallest (r,c). From that start, repeatedly
        connect the nearest unvisited vertex (tie by row, then col). Draw only the
        valid H/V/45 segments on background cells. Then copy every different-colored
        neighbor of the chosen start along each drawn segment as a parallel offset,
        skipping collinear offsets and blocked cells.

    Key insight:
        Color no longer identifies a whole object. The object is the local component,
        and its topology is the visible greedy nearest-neighbor traversal rather than
        a centroid-based polygon cycle.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _bg_color(g)
    out = g.copy()

    for col in colors_present(g, exclude=[bg]):
        for pts_rc in _same_color_groups(g, col, bg):
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

            drawable_edges = _drawable_edges(g, _path_edges(path), col, bg, allow={col})

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


BG = 0
# Keep the original task palette; this combo only changes reuse/density/path logic.
COLORS = [1, 2, 3, 4, 6, 7, 8]
CANVAS_H_RANGE = (26, 33)
CANVAS_W_JITTER = (-4, 5)
CANVAS_W_CLIP = (26, 32)

SHAPE_COUNT_RANGE = (5, 8)
MAX_SHADOWS_CAP = 2
PLACEMENT_PADDING = 1

PATH_VERTICES_RANGE = (3, 6)
PATH_TOTAL_LENGTH_MAX = 9
PATH_MIN_GAP = 2
PATH_BUILD_RETRIES = 120
PATH_DIRECTIONS = [(0, 1), (1, 0), (1, 1)]

DISTRACTOR_PROB_BY_SHAPES = {5: 0.25, 6: 0.15, 7: 0.05}
DISTRACTOR_COUNT_RANGE = (1, 4)
DISTRACTOR_BBOX_RANGE = (2, 5)
DISTRACTOR_RETRIES = 60

SHAPE_PATCH_RETRIES = 120
GENERATE_RETRIES = 900


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


def _shape_cells(vertices: list[tuple[int, int]]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for a, b in _path_edges(vertices):
        cells.update(_gen_segment_cells(a, b))
    return cells


def _edge_dirs(vertices: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return [_step(a, b) for a, b in _path_edges(vertices)]


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
    gaps = _sample_gaps(n_vertices - 1)
    if gaps is None:
        return None

    directions: list[tuple[int, int]] = []
    for i in range(n_vertices - 1):
        candidates = list(PATH_DIRECTIONS)
        if directions:
            prev = directions[-1]
            if len(candidates) > 1 and np.random.rand() < 0.75:
                candidates = [d for d in candidates if d != prev]
            if i == n_vertices - 2 and len(set(directions)) == 1:
                alt = [d for d in candidates if d != prev]
                if alt:
                    candidates = alt
        directions.append(candidates[int(np.random.randint(len(candidates)))])

    if len(set(directions)) < 2:
        return None

    vertices = [(0, 0)]
    for (dr, dc), gap in zip(directions, gaps):
        r, c = vertices[-1]
        vertices.append((r + dr * gap, c + dc * gap))

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
    candidates = []
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
    out[shifted & (out == BG)] = shadow_color


def _make_shape_patch(shape_color: int, shadow_color: int | None) -> dict[str, object] | None:
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
        sr, sc = anchor[0] + offset[0], anchor[1] + offset[1]
        if not (0 <= sr < h and 0 <= sc < w) or inp[sr, sc] != BG:
            return None
        inp[sr, sc] = int(shadow_color)
        if sum(1 for rr, cc in neighbors8(anchor[0], anchor[1]) if 0 <= rr < h and 0 <= cc < w and inp[rr, cc] == shadow_color) != 1:
            return None

    if not np.array_equal(main(inp), out):
        return None

    return {
        "patch_in": inp,
        "patch_out": out,
        "shape_color": int(shape_color),
        "vertices": shifted_vertices,
        "footprint": (h * w, int(np.count_nonzero(out))),
    }


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


def _make_distractor_patch(color: int) -> dict[str, object]:
    count = int(np.random.randint(*DISTRACTOR_COUNT_RANGE))
    if count == 1:
        patch = np.full((1, 1), color, dtype=int)
        return {"patch_in": patch.copy(), "patch_out": patch.copy(), "footprint": (1, 1)}
    for _ in range(DISTRACTOR_RETRIES):
        h = int(np.random.randint(*DISTRACTOR_BBOX_RANGE))
        w = int(np.random.randint(*DISTRACTOR_BBOX_RANGE))
        coords = _sample_distractor_coords(h, w, count)
        if coords is None:
            continue
        patch = np.full((h, w), BG, dtype=int)
        for r, c in coords:
            patch[r, c] = color
        return {"patch_in": patch.copy(), "patch_out": patch.copy(), "footprint": (h * w, int(np.count_nonzero(patch)))}
    patch = np.full((1, 1), color, dtype=int)
    return {"patch_in": patch.copy(), "patch_out": patch.copy(), "footprint": (1, 1)}


def _choose_has_distractor(shape_count: int) -> bool:
    return bool(np.random.rand() < DISTRACTOR_PROB_BY_SHAPES[shape_count])


def _sample_shadow_count(max_shadows: int) -> int:
    if max_shadows <= 0:
        return 0
    options = np.arange(max_shadows + 1)
    weights = np.array([1.0] + [2.5 + i for i in range(max_shadows)], dtype=float)
    weights /= weights.sum()
    return int(np.random.choice(options, p=weights))


def _sample_palette(shape_count: int, has_distractor: bool) -> tuple[list[int], list[int], int | None] | None:
    min_unique_shapes = max(3, shape_count - 2)
    max_unique_shapes = shape_count - 1
    unique_shape_count = int(np.random.randint(min_unique_shapes, max_unique_shapes + 1))

    shape_palette = list(map(int, np.random.permutation(COLORS)[:unique_shape_count]))
    shape_colors = shape_palette.copy()
    extra_count = shape_count - unique_shape_count
    if extra_count > 0:
        shape_colors.extend(int(v) for v in np.random.choice(shape_palette, size=extra_count, replace=True).tolist())
    np.random.shuffle(shape_colors)

    shadow_pool = [col for col in COLORS if col not in shape_palette]
    max_shadows = min(MAX_SHADOWS_CAP, shape_count, len(shadow_pool) - int(has_distractor))
    if max_shadows < 0:
        return None
    shadow_count = _sample_shadow_count(max_shadows)
    shadow_pool = list(map(int, np.random.permutation(shadow_pool)))
    shadow_colors = shadow_pool[:shadow_count]
    distractor_color = shadow_pool[shadow_count] if has_distractor else None
    return list(map(int, shape_colors)), list(map(int, shadow_colors)), None if distractor_color is None else int(distractor_color)


def _assign_shadows(shape_count: int, shadow_colors: list[int]) -> dict[int, int]:
    if not shadow_colors:
        return {}
    targets = np.random.choice(shape_count, size=len(shadow_colors), replace=False).tolist()
    return {int(idx): int(col) for idx, col in zip(sorted(targets), shadow_colors)}


def _sample_canvas_size(shape_count: int) -> tuple[int, int]:
    min_dim = 26 + max(0, shape_count - 6)
    h = int(np.random.randint(min_dim, CANVAS_H_RANGE[1]))
    w = int(np.clip(h + np.random.randint(*CANVAS_W_JITTER), min_dim, CANVAS_W_CLIP[1]))
    return h, w


def _dilate_8(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.copy()
    out = np.zeros_like(mask, dtype=bool)
    h, w = mask.shape
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            r_src = slice(max(0, -dr), h - max(0, dr))
            c_src = slice(max(0, -dc), w - max(0, dc))
            r_dst = slice(max(0, dr), h - max(0, -dr))
            c_dst = slice(max(0, dc), w - max(0, -dc))
            out[r_dst, c_dst] |= mask[r_src, c_src]
    return out


def _candidate_locations(canvas_out: np.ndarray, patch_out: np.ndarray) -> list[tuple[int, int]]:
    h, w = patch_out.shape
    H, W = canvas_out.shape
    if h > H or w > W:
        return []

    occupied = canvas_out != BG
    blocked = _dilate_8(occupied, PLACEMENT_PADDING)
    rr, cc = np.nonzero(patch_out != BG)
    if len(rr) == 0:
        return []

    candidates = []
    for r0 in range(H - h + 1):
        for c0 in range(W - w + 1):
            if blocked[r0 + rr, c0 + cc].any():
                continue
            candidates.append((int(r0), int(c0)))
    return candidates


def _cross_group_alignment_exists(
    new_vertices: list[tuple[int, int]],
    prior_vertices: list[tuple[int, int]],
) -> bool:
    for r0, c0 in new_vertices:
        for r1, c1 in prior_vertices:
            if _aligned_8(r0, c0, r1, c1):
                return True
    return False


def _place_shape_patch(
    inp: np.ndarray,
    out: np.ndarray,
    patch: dict[str, object],
    placed_shape_groups: list[dict[str, object]],
) -> bool:
    patch_in = patch["patch_in"]
    patch_out = patch["patch_out"]
    shape_color = int(patch["shape_color"])
    local_vertices = patch["vertices"]

    candidates = _candidate_locations(out, patch_out)
    valid = []
    for r0, c0 in candidates:
        global_vertices = [(r0 + r, c0 + c) for r, c in local_vertices]
        bad = False
        for info in placed_shape_groups:
            if int(info["shape_color"]) != shape_color:
                continue
            if _cross_group_alignment_exists(global_vertices, info["vertices"]):
                bad = True
                break
        if not bad:
            valid.append((r0, c0))

    if not valid:
        return False

    loc = valid[int(np.random.randint(len(valid)))]
    paste(out, patch_out, loc[0], loc[1], transparent=BG)
    paste(inp, patch_in, loc[0], loc[1], transparent=BG)
    placed_shape_groups.append(
        {
            "shape_color": shape_color,
            "vertices": [(loc[0] + r, loc[1] + c) for r, c in local_vertices],
        }
    )
    return True


def _place_generic_patch(inp: np.ndarray, out: np.ndarray, patch: dict[str, object]) -> bool:
    candidates = _candidate_locations(out, patch["patch_out"])
    if not candidates:
        return False
    loc = candidates[int(np.random.randint(len(candidates)))]
    paste(out, patch["patch_out"], loc[0], loc[1], transparent=BG)
    paste(inp, patch["patch_in"], loc[0], loc[1], transparent=BG)
    return True


def _build_patches(
    shape_colors: list[int],
    shadow_map: dict[int, int],
    distractor_color: int | None,
) -> list[dict[str, object]] | None:
    patches: list[dict[str, object]] = []
    for idx, shape_color in enumerate(shape_colors):
        patch = None
        for _ in range(SHAPE_PATCH_RETRIES):
            patch = _make_shape_patch(int(shape_color), shadow_map.get(idx))
            if patch is not None:
                break
        if patch is None:
            return None
        patches.append(patch)

    if distractor_color is not None:
        patches.append(_make_distractor_patch(int(distractor_color)))

    patches.sort(key=lambda p: p["footprint"], reverse=True)
    return patches


def _gen_attempt(shape_count: int) -> np.ndarray | None:
    has_distractor = _choose_has_distractor(shape_count)
    palette = _sample_palette(shape_count, has_distractor)
    if palette is None:
        return None
    shape_colors, shadow_colors, distractor_color = palette
    shadow_map = _assign_shadows(shape_count, shadow_colors)

    patches = _build_patches(shape_colors, shadow_map, distractor_color)
    if patches is None:
        return None

    h, w = _sample_canvas_size(shape_count)
    out = np.full((h, w), BG, dtype=int)
    inp = np.full((h, w), BG, dtype=int)
    placed_shape_groups: list[dict[str, object]] = []

    for patch in patches:
        if "shape_color" in patch:
            if not _place_shape_patch(inp, out, patch, placed_shape_groups):
                return None
        else:
            if not _place_generic_patch(inp, out, patch):
                return None

    present_colors = set(shape_colors) | set(shadow_colors)
    if distractor_color is not None:
        present_colors.add(int(distractor_color))

    stats = _shape_group_stats(inp)
    if stats["shape_group_count"] != shape_count:
        return None
    if stats["reused_color_count"] < 1 or stats["reused_group_surplus"] < 1:
        return None
    if stats["greedy_distinct_group_count"] != shape_count:
        return None
    if get_bg(inp) != BG or get_bg(out) != BG:
        return None
    if set(colors_present(inp, exclude=[BG])) != present_colors:
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    shape_counts = list(range(*SHAPE_COUNT_RANGE))
    target_shape_count = shape_counts[int(np.random.randint(len(shape_counts)))]
    for _ in range(GENERATE_RETRIES):
        result = _gen_attempt(target_shape_count)
        if result is not None:
            return result
    raise RuntimeError(
        f"generate_input failed after {GENERATE_RETRIES} attempts for shape_count={target_shape_count}"
    )
