# Variant of 35ab12c3
# Dimension: D2 — 路径方向类型 (path)
# Change: straight_ray + diagonal_path——顶点间通过水平、垂直或严格45度对角线连接，要求dr==0或... → 曼哈顿折线路径
#
# Design analysis
# - Source-task note:
#   The prompt names cb2d8a2c as pathfinding inspiration, but this workspace does
#   not contain a standalone cb2d8a2c solver source file. The available reference
#   artifact is the official ARC I/O JSON, whose visible baseline is "right-angle
#   turning". This variant therefore imports only that observable idea: when two
#   same-color vertices are not H/V/45deg aligned, connect them by one L-turn.
# - Concretization:
#   Original behavior on a non-aligned pair such as (1,1) and (4,6): no edge is
#   drawn at all because the pair is neither orthogonal nor diagonal. New behavior:
#   draw an L path (1,1)->(1,6)->(4,6). If the shape carries a shadow anchor with
#   offset (-1,0), copy the same polyline in parallel at rows shifted by -1.
# - Disambiguation:
#   Non-aligned pairs admit two L corners. The task rule is fixed and visible:
#   prefer horizontal-then-vertical; only if that exact polyline is blocked in the
#   original input do we fall back to vertical-then-horizontal. Generator samples
#   are built so the preferred branch is the one that actually appears, keeping the
#   task human-readable from few examples.
# - Complexity calibration:
#   Generator keeps the original "same-color vertices + optional shadow neighbor"
#   structure, but now forces at least one shape per grid to contain a non-aligned
#   edge so the D2 change is observable. Shapes are still sparse outlines rather
#   than dense fills, avoiding overlap-heavy pathfinding.
# - Implicit-assumption audit for main():
#   1. Old `_aligned_8(...)` gate assumed every drawable edge is a single straight
#      segment. That is false now, so the solver must reason over whole polylines.
#   2. Old `_clear_in_original(...)` skipped only segment endpoints. For L paths,
#      the bend cell is an interior cell and must also be checked; otherwise a path
#      could illegally turn through an occupied pixel.
#   3. Old shadow logic shifted only the two endpoints of each edge. For L paths we
#      must shift every waypoint, otherwise the copied shadow could choose a
#      different corner than the source path.
#   4. Old distractors could use multi-pixel non-aligned groups because those were
#      ignored by the solver. That is no longer safe: a non-aligned group can now
#      become a valid shape. Generator therefore restricts distractors to single
#      pixels.
# - Placement legality:
#   Output patches are constructed from the same preferred-path rule used by the
#   solver. Patches include the full shifted shadow footprint in their bbox and are
#   placed with a 1-cell buffer, so pasted shapes do not block one another's
#   polylines.
# - Visual observability / cleanliness:
#   Samples mix aligned and L-turn edges, but every grid contains at least one
#   genuine Manhattan edge. The horizontal-first corner is stable across samples,
#   making the rule visible without requiring color-id reasoning or hidden tie
#   breakers.

import math

import numpy as np

from arc_utils import bbox_coords, colors_present, get_bg, neighbors8, paste, random_free_location
from arc_utils import get_bg as _bg_color

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


def _candidate_paths(a: tuple[int, int], b: tuple[int, int]) -> list[list[tuple[int, int]]]:
    r0, c0 = a
    r1, c1 = b
    if _aligned_8(r0, c0, r1, c1):
        return [[a, b]]
    return [[a, (r0, c1), b], [a, (r1, c0), b]]


def _preferred_path(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    return _candidate_paths(a, b)[0]


def _polyline_cells(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for i in range(len(points) - 1):
        seg = list(_segment_cells(points[i][0], points[i][1], points[i + 1][0], points[i + 1][1]))
        if i:
            seg = seg[1:]
        cells.extend(seg)
    return cells


def _clear_path_in_original(g: np.ndarray, points: list[tuple[int, int]], bg: int, allow: set[int]) -> bool:
    cells = _polyline_cells(points)
    for rr, cc in cells[1:-1]:
        v = int(g[rr, cc])
        if v != bg and v not in allow:
            return False
    return True


def _draw_path_bg_only(out: np.ndarray, points: list[tuple[int, int]], col: int, bg: int) -> None:
    for rr, cc in _polyline_cells(points):
        if out[rr, cc] == bg:
            out[rr, cc] = col


def _shift_points(points: list[tuple[int, int]], dr: int, dc: int) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for r, c in points]


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


def _shape_cycle(vertices: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cycle = _angle_cycle(vertices)
    if len(cycle) == 2:
        return cycle
    return cycle


def _shape_edges(vertices: list[tuple[int, int]]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    cycle = _shape_cycle(vertices)
    if len(cycle) == 2:
        return [(cycle[0], cycle[1])]
    return [(cycle[i], cycle[(i + 1) % len(cycle)]) for i in range(len(cycle))]


def _shape_paths(vertices: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    return [_preferred_path(a, b) for a, b in _shape_edges(vertices)]


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


def _shape_has_manhattan_edge(vertices: list[tuple[int, int]]) -> bool:
    return any(not _aligned_8(a[0], a[1], b[0], b[1]) for a, b in _shape_edges(vertices))


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

        cycle = _angle_cycle(pts_rc)
        if len(cycle) < 2:
            continue

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

        if len(cycle) == 2:
            edges = [(cycle[0], cycle[1])]
        else:
            edges = [(cycle[i], cycle[(i + 1) % len(cycle)]) for i in range(len(cycle))]

        drawable_paths: list[list[tuple[int, int]]] = []
        for a, b in edges:
            chosen = None
            for points in _candidate_paths(a, b):
                if _clear_path_in_original(g, points, bg, allow={col}):
                    chosen = points
                    break
            if chosen is not None:
                drawable_paths.append(chosen)

        for points in drawable_paths:
            _draw_path_bg_only(out, points, col, bg)

        for points in drawable_paths:
            for drn, dcn, ncol in carried:
                shifted = _shift_points(points, drn, dcn)
                if not _path_offset_compatible(points, drn, dcn):
                    continue
                if not all(0 <= r < h and 0 <= c < w for r, c in shifted):
                    continue
                if not _clear_path_in_original(g, shifted, bg, allow=set()):
                    continue
                _draw_path_bg_only(out, shifted, ncol, bg)

    return out


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

SHAPE_WEIGHTS = np.array([0.24, 0.24, 0.20, 0.18, 0.14], dtype=float)

SHAPE_PLACEMENT_RETRIES = 40
GENERATE_RETRIES = 300
VERTEX_RETRIES = 80


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
    vertices: list[tuple[int, int]],
    shape_cells: set[tuple[int, int]],
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


def _make_shape_patch(shape_color: int, shadow_color: int | None, require_manhattan: bool) -> tuple[np.ndarray, np.ndarray] | None:
    vertices = _make_vertices(require_manhattan=require_manhattan)
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
        _draw_path_bg_only(out, points, int(shape_color), BG)
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

    if require_manhattan and not _shape_has_manhattan_edge(shifted_vertices):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp, out


def _make_distractor_patch(color: int) -> tuple[np.ndarray, np.ndarray]:
    patch = np.full((1, 1), int(color), dtype=int)
    return patch.copy(), patch.copy()


def _sample_palette(shape_count: int, has_distractor: bool) -> tuple[list[int], list[int], int | None] | None:
    max_shadows = min(MAX_SHADOWS_CAP, shape_count, len(COLORS) - shape_count - int(has_distractor))
    if max_shadows < 0:
        return None
    shadow_count = int(np.random.randint(max_shadows + 1))
    total = shape_count + shadow_count + int(has_distractor)
    palette = list(np.random.permutation(COLORS)[:total])
    shape_colors = palette[:shape_count]
    shadow_colors = palette[shape_count:shape_count + shadow_count]
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
    h = int(np.random.randint(*CANVAS_H_RANGE))
    w = int(np.clip(h + np.random.randint(*CANVAS_W_JITTER), *CANVAS_W_CLIP))
    return h, w


def _variant_metrics(grid: np.ndarray) -> dict[str, int]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    manhattan_edges = 0
    shape_colors = 0
    for col in sorted(set(int(v) for v in np.unique(g)) - {bg}):
        pts_rc = [tuple(map(int, p)) for p in np.argwhere(g == col)]
        if len(pts_rc) < 2:
            continue
        cycle = _angle_cycle(pts_rc)
        edges = [(cycle[0], cycle[1])] if len(cycle) == 2 else [(cycle[i], cycle[(i + 1) % len(cycle)]) for i in range(len(cycle))]
        local = 0
        for a, b in edges:
            if not _aligned_8(a[0], a[1], b[0], b[1]):
                local += 1
        if local:
            shape_colors += 1
            manhattan_edges += local
    return {"shape_colors_with_manhattan": shape_colors, "manhattan_edges": manhattan_edges}


def _gen_attempt() -> np.ndarray | None:
    h, w = _sample_canvas_size()
    out = np.full((h, w), BG, dtype=int)
    inp = np.full((h, w), BG, dtype=int)

    has_distractor = bool(np.random.rand() < DISTRACTOR_PROB)
    shape_count = int(np.random.randint(*SHAPE_COUNT_RANGE))
    required_idx = int(np.random.randint(shape_count))

    palette = _sample_palette(shape_count, has_distractor)
    if palette is None:
        return None
    shape_colors, shadow_colors, distractor_color = palette
    shadow_map = _assign_shadows(shape_count, shadow_colors)

    total_colors = shape_count + len(shadow_colors) + int(has_distractor)

    for idx, shape_color in enumerate(shape_colors):
        placed = False
        for _ in range(SHAPE_PLACEMENT_RETRIES):
            patch = _make_shape_patch(
                int(shape_color),
                shadow_map.get(idx),
                require_manhattan=idx == required_idx,
            )
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

    metrics = _variant_metrics(inp)
    if metrics["manhattan_edges"] == 0:
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
