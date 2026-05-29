# Variant of 35ab12c3
# Dimension: D9 — 干扰项设计 (obstacle)
# Change: 可选的干扰项组（0-1组，1-3个像素），其像素不能H/V/对角线对齐，在输出中保持不变... → 干扰项作为障碍物
#
# Design analysis for this variant
# --------------------------------
# spec:
#   core_change:
#     original:
#       Distractor pixels were merely static foreign colors. Any edge whose straight
#       segment crossed them was rejected wholesale by `_clear_in_original`, so the
#       distractor only removed the entire edge and had no structured geometric role.
#     variant:
#       Distractor pixels become obstacle cells. A shape edge still tries to connect
#       the same ordered vertices, but once the straight ray hits the first foreign
#       interior cell it cannot pass through it. The solver first tries a deterministic
#       one-cell sidestep around that blocker; if both sidesteps are unavailable it
#       stops before the blocker.
#   concrete_pixel_behavior:
#     - Obstacles are still an optional single color-group with 1-3 pixels.
#     - Their pixels stay unchanged in the output.
#     - Generator places obstacle pixels on interior cells of a would-be main edge,
#       never on a vertex and never 8-neighboring any vertex, so they are visually
#       read as blockers rather than shadow anchors.
#     - For an affected edge A->B, if the first blocker is X and the previous cell is P,
#       the preferred reroute is:
#         A ... P, then P -> P+n, then a parallel run to B+n, then B+n -> B
#       where n is one of the two unit normals to the original direction.
#     - If neither side corridor is open, only the prefix A ... P is drawn.
#     - Shadow copies are generated from the actually drawn routed subsegments, so
#       the shadow bends together with the main edge.
#   small_trace:
#     original_logic:
#       input edge wants to draw (4,2)->(4,8); distractor at (4,5).
#       `_clear_in_original(...)` returns False, so the whole edge is skipped.
#     variant_logic:
#       first blocker is (4,5), previous cell is (4,4). Try upward normal (-1,0):
#       route becomes (4,2)->(4,4)->(3,4)->(3,8)->(4,8) if that corridor is clear.
#       If both upward and downward corridors are blocked, keep only (4,2)->(4,4).
#     visible_difference:
#       the distractor now causes a local bend or early stop instead of deleting the
#       whole edge with no structure.
#   complexity_calibration:
#     - The sidestep amount is fixed to one cell. This keeps the rule learnable from
#       2-4 examples; allowing arbitrary shortest paths would be too unconstrained.
#     - Only the first blocker on each edge matters. With the original "1-3 pixels,
#       pairwise non-aligned" constraint, the obstacle group cannot form a long wall,
#       which keeps routing readable and avoids maze-like ambiguity.
#     - Generator strongly favors having an obstacle and requires at least one rerouted
#       edge, so the D9 change is usually visible rather than hidden.
#   implicit_assumptions_review:
#     - Original `main()` assumed every foreign interior pixel simply vetoed the whole
#       edge. That assumption is replaced by `_route_segment_in_original(...)`.
#     - Original seed selection treated any different-color neighbor as a carried
#       shadow. In this variant, only singleton-color neighbors are treated as shadow
#       anchors. This avoids multi-pixel obstacle groups being misread as shadows.
#     - This introduces a new assumption: true shadow colors are singleton groups.
#       Generator guarantees that.
#     - Another new assumption is that obstacle pixels are not 8-neighboring shape
#       vertices; otherwise a singleton obstacle could look like a shadow anchor.
#       Generator guarantees that too.
#   placement_legality:
#     - Local shape patches are built first, optionally with shadow anchors.
#     - Obstacle pixels are chosen from solver-relevant edge interiors, then the final
#       output patch is obtained by running `main()` on the patch itself.
#     - The patch bbox is expanded by a small margin so one-cell reroutes and their
#       translated shadows stay in-bounds.
#     - Global placement still uses output patches, so routed footprints reserve space
#       before shapes are pasted onto the canvas.
#   visual_observability_and_cleanliness:
#     - Obstacles use one distinct color and remain sparse (1-3 pixels).
#     - They sit inside a potential line corridor, so the detour is locally legible.
#     - Shapes remain separated by at least one-cell padding in their final outputs,
#       preventing routed segments from cluttering other objects.

import math

import numpy as np

from arc_utils import get_bg as _bg_color
from arc_utils import bbox_coords, colors_present, draw_line, get_bg, neighbors8, paste, random_free_location

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


def _cycle_edges(cycle: list[tuple[int, int]]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    if len(cycle) == 2:
        return [(cycle[0], cycle[1])]
    return [(cycle[i], cycle[(i + 1) % len(cycle)]) for i in range(len(cycle))]


def _color_counts(g: np.ndarray) -> dict[int, int]:
    vals, counts = np.unique(g, return_counts=True)
    return {int(v): int(c) for v, c in zip(vals, counts)}


def _ordered_cycle_for_color(
    g: np.ndarray,
    col: int,
    bg: int,
    pts_rc: list[tuple[int, int]],
    color_counts: dict[int, int],
) -> list[tuple[int, int]]:
    cycle = _angle_cycle(pts_rc)
    if len(cycle) < 2:
        return cycle

    seed_idxs = []
    h, w = g.shape
    for i, (r, c) in enumerate(cycle):
        for drn, dcn in DIR8:
            rr = r + drn
            cc = c + dcn
            if 0 <= rr < h and 0 <= cc < w:
                v = int(g[rr, cc])
                if v != bg and v != col and color_counts.get(v, 0) == 1:
                    seed_idxs.append(i)
                    break

    if seed_idxs:
        start = min(seed_idxs, key=lambda i: (cycle[i][0], cycle[i][1]))
    else:
        start = min(range(len(cycle)), key=lambda i: (cycle[i][0], cycle[i][1]))
    return cycle[start:] + cycle[:start]


def _perp_offsets(dr: int, dc: int) -> list[tuple[int, int]]:
    return [(-dc, dr), (dc, -dr)]


def _cell_is_clear(g: np.ndarray, r: int, c: int, bg: int, allow: set[int]) -> bool:
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
        if not _cell_is_clear(g, rr, cc, bg, allow):
            return idx, (rr, cc)
    return None


def _route_segment_in_original(
    g: np.ndarray,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    bg: int,
    allow: set[int],
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    if not _aligned_8(r0, c0, r1, c1):
        return []

    block = _first_blocker(g, r0, c0, r1, c1, bg, allow)
    if block is None:
        return [((r0, c0), (r1, c1))]

    cells = list(_segment_cells(r0, c0, r1, c1))
    block_idx, _ = block
    prev = cells[block_idx - 1]
    drs, dcs = _step_dir(r0, c0, r1, c1)

    for nr, nc in _perp_offsets(drs, dcs):
        mid0 = (prev[0] + nr, prev[1] + nc)
        mid1 = (r1 + nr, c1 + nc)
        if not (
            0 <= mid0[0] < g.shape[0]
            and 0 <= mid0[1] < g.shape[1]
            and 0 <= mid1[0] < g.shape[0]
            and 0 <= mid1[1] < g.shape[1]
        ):
            continue
        if int(g[mid0]) != bg or int(g[mid1]) != bg:
            continue
        if not _clear_in_original(g, mid0[0], mid0[1], mid1[0], mid1[1], bg, allow=set()):
            continue
        routed = []
        if (r0, c0) != prev:
            routed.append(((r0, c0), prev))
        routed.append((prev, mid0))
        routed.append((mid0, mid1))
        routed.append((mid1, (r1, c1)))
        return routed

    if (r0, c0) != prev:
        return [((r0, c0), prev)]
    return []


def _route_stats_for_color(g: np.ndarray, col: int, bg: int) -> tuple[int, int, int]:
    pts_rc = [tuple(map(int, p)) for p in np.argwhere(g == col)]
    if len(pts_rc) < 2:
        return 0, 0, 0
    color_counts = _color_counts(g)
    cycle = _ordered_cycle_for_color(g, col, bg, pts_rc, color_counts)
    blocked = 0
    detoured = 0
    trimmed = 0
    for (r0, c0), (r1, c1) in _cycle_edges(cycle):
        if not _aligned_8(r0, c0, r1, c1):
            continue
        if _first_blocker(g, r0, c0, r1, c1, bg, allow={col}) is None:
            continue
        blocked += 1
        route = _route_segment_in_original(g, r0, c0, r1, c1, bg, allow={col})
        if len(route) >= 2:
            detoured += 1
        else:
            trimmed += 1
    return blocked, detoured, trimmed


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _bg_color(g)
    out = g.copy()

    color_counts = _color_counts(g)
    colors = sorted(set(int(v) for v in np.unique(g)) - {bg})
    for col in colors:
        pts_rc = [tuple(map(int, p)) for p in np.argwhere(g == col)]
        if len(pts_rc) < 2:
            continue

        cycle = _ordered_cycle_for_color(g, col, bg, pts_rc, color_counts)
        if len(cycle) < 2:
            continue

        carried: list[tuple[int, int, int]] = []
        sr0, sc0 = cycle[0]
        for drn, dcn in DIR8:
            rr = sr0 + drn
            cc = sc0 + dcn
            if 0 <= rr < h and 0 <= cc < w:
                v = int(g[rr, cc])
                if v != bg and v != col and color_counts.get(v, 0) == 1:
                    carried.append((drn, dcn, v))

        routed_edges: list[list[tuple[tuple[int, int], tuple[int, int]]]] = []
        for (r0, c0), (r1, c1) in _cycle_edges(cycle):
            routed = _route_segment_in_original(g, r0, c0, r1, c1, bg, allow={col})
            if routed:
                routed_edges.append(routed)

        for routed in routed_edges:
            for (r0, c0), (r1, c1) in routed:
                _draw_line_bg_only(out, r0, c0, r1, c1, col, bg)

        for routed in routed_edges:
            for (r0, c0), (r1, c1) in routed:
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
                    if int(g[rr0, cc0]) not in {bg, ncol} or int(g[rr1, cc1]) not in {bg, ncol}:
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
OBSTACLE_PROB = 0.85
PLACEMENT_PADDING = 1
PLACEMENT_PAD_CONNECTIVITY = 8

LINE_VERTICES_RANGE = (2, 8)
LINE_TOTAL_LENGTH = 13
LINE_MIN_GAP = 2
TRIANGLE_SIDE_RANGE = (2, 8)
RECT_H_RANGE = (2, 9)
RECT_W_RANGE = (2, 9)
DIAMOND_ARM_RANGE = (2, 6)
PENTAGON_ROOF_RANGE = (2, 5)
PENTAGON_WALL_RANGE = (2, 7)
HEXAGON_SLANT_RANGE = (2, 5)
HEXAGON_WALL_RANGE = (2, 6)

SHAPE_WEIGHTS = np.array([0.28, 0.18, 0.18, 0.14, 0.12, 0.10], dtype=float)

OBSTACLE_COUNT_RANGE = (1, 4)
OBSTACLE_MARGIN = 2

SHAPE_PLACEMENT_RETRIES = 40
PATCH_RETRIES = 60
GENERATE_RETRIES = 300


def _step(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return _step_dir(a[0], a[1], b[0], b[1])


def _gen_segment_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    drs, dcs = _step(a, b)
    steps = max(abs(b[0] - a[0]), abs(b[1] - a[1]))
    return [(a[0] + drs * k, a[1] + dcs * k) for k in range(steps + 1)]


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


def _build_line() -> list[tuple[int, int]]:
    n_vertices = int(np.random.randint(*LINE_VERTICES_RANGE))
    remaining = LINE_TOTAL_LENGTH
    pos = [0]
    for i in range(n_vertices - 1):
        steps_left = n_vertices - 2 - i
        gap_hi = remaining - LINE_MIN_GAP * steps_left
        gap = int(np.random.randint(LINE_MIN_GAP, gap_hi + 1))
        pos.append(pos[-1] + gap)
        remaining -= gap
    return _transform_coords([(0, x) for x in pos])


def _build_triangle() -> list[tuple[int, int]]:
    side = int(np.random.randint(*TRIANGLE_SIDE_RANGE))
    return _transform_coords([(0, 0), (0, side), (side, side)])


def _build_rectangle() -> list[tuple[int, int]]:
    h = int(np.random.randint(*RECT_H_RANGE))
    w = int(np.random.randint(*RECT_W_RANGE))
    return _transform_coords([(0, 0), (0, w), (h, w), (h, 0)])


def _build_diamond() -> list[tuple[int, int]]:
    arm = int(np.random.randint(*DIAMOND_ARM_RANGE))
    return _transform_coords([(0, arm), (arm, 2 * arm), (2 * arm, arm), (arm, 0)])


def _build_pentagon() -> list[tuple[int, int]]:
    roof = int(np.random.randint(*PENTAGON_ROOF_RANGE))
    wall = int(np.random.randint(*PENTAGON_WALL_RANGE))
    h = roof + wall
    return _transform_coords([(0, roof), (roof, 0), (h, 0), (h, 2 * roof), (roof, 2 * roof)])


def _build_hexagon() -> list[tuple[int, int]]:
    slant = int(np.random.randint(*HEXAGON_SLANT_RANGE))
    wall = int(np.random.randint(*HEXAGON_WALL_RANGE))
    mid = slant + wall
    bot = 2 * slant + wall
    return _transform_coords([(0, slant), (slant, 0), (mid, 0), (bot, slant), (mid, 2 * slant), (slant, 2 * slant)])


_SHAPE_BUILDERS = [_build_line, _build_triangle, _build_rectangle, _build_diamond, _build_pentagon, _build_hexagon]


def _make_vertices() -> list[tuple[int, int]]:
    weights = SHAPE_WEIGHTS / SHAPE_WEIGHTS.sum()
    idx = int(np.random.choice(len(_SHAPE_BUILDERS), p=weights))
    return _SHAPE_BUILDERS[idx]()


def _shape_edges(vertices: list[tuple[int, int]]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    return _cycle_edges(vertices)


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
    anchor = min(vertices)
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
    margin: int = 0,
) -> tuple[int, int, int, int]:
    coords = list(shape_cells) + vertices
    if shadow is not None:
        (ar, ac), (dr, dc) = shadow
        coords.append((ar + dr, ac + dc))
        coords.extend((r + dr, c + dc) for r, c in shape_cells)
    r0, c0, r1, c1 = bbox_coords(np.array(coords, dtype=int))
    return int(r0 - margin), int(c0 - margin), int(r1 + margin), int(c1 + margin)


def _paint_shadow(out: np.ndarray, shape_color: int, shadow_color: int, offset: tuple[int, int]) -> None:
    dr, dc = offset
    h, w = out.shape
    src = (out == shape_color)
    shifted = np.zeros_like(src)
    r_src = slice(max(0, -dr), h - max(0, dr))
    c_src = slice(max(0, -dc), w - max(0, dc))
    r_dst = slice(max(0, dr), h - max(0, -dr))
    c_dst = slice(max(0, dc), w - max(0, -dc))
    shifted[r_dst, c_dst] = src[r_src, c_src]
    mask = shifted & (out == BG)
    out[mask] = shadow_color


def _base_shape_patch(
    shape_color: int,
    shadow_color: int | None,
    margin: int,
) -> tuple[np.ndarray, list[tuple[int, int]], tuple[int, int] | None] | None:
    vertices = _make_vertices()
    cells = _shape_cells(vertices)

    shadow = None
    if shadow_color is not None:
        shadow = _choose_shadow(vertices, cells)
        if shadow is None:
            return None

    r0, c0, r1, c1 = _patch_bbox(cells, vertices, shadow, margin=margin)
    h, w = r1 - r0 + 1, c1 - c0 + 1
    inp = np.full((h, w), BG, dtype=int)

    shifted_vertices = [(r - r0, c - c0) for r, c in vertices]
    for r, c in shifted_vertices:
        inp[r, c] = shape_color

    shifted_shadow_offset = None
    if shadow is not None:
        (ar, ac), offset = shadow
        anchor = (ar - r0, ac - c0)
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
        shifted_shadow_offset = offset

    return inp, shifted_vertices, shifted_shadow_offset


def _obstacle_candidates(base_inp: np.ndarray, shape_color: int) -> dict[int, list[tuple[int, int]]]:
    bg = BG
    pts_rc = [tuple(map(int, p)) for p in np.argwhere(base_inp == shape_color)]
    cycle = _ordered_cycle_for_color(base_inp, shape_color, bg, pts_rc, _color_counts(base_inp))
    anchors = [tuple(map(int, p)) for p in np.argwhere((base_inp != bg) & (base_inp != shape_color))]
    by_edge: dict[int, list[tuple[int, int]]] = {}
    for edge_idx, (a, b) in enumerate(_cycle_edges(cycle)):
        if not _aligned_8(a[0], a[1], b[0], b[1]):
            continue
        cells = _gen_segment_cells(a, b)
        if len(cells) < 5:
            continue
        for t in range(2, len(cells) - 2):
            r, c = cells[t]
            if any(max(abs(r - vr), abs(c - vc)) <= 1 for vr, vc in cycle):
                continue
            if any(max(abs(r - ar), abs(c - ac)) <= 1 for ar, ac in anchors):
                continue
            by_edge.setdefault(edge_idx, []).append((r, c))
    return by_edge


def _choose_obstacles(base_inp: np.ndarray, shape_color: int, obstacle_count: int) -> list[tuple[int, int]] | None:
    by_edge = _obstacle_candidates(base_inp, shape_color)
    if not by_edge:
        return None
    target = min(int(obstacle_count), len(by_edge))
    edge_ids = list(by_edge.keys())
    np.random.shuffle(edge_ids)
    chosen: list[tuple[int, int]] = []
    for edge_idx in edge_ids:
        options = list(by_edge[edge_idx])
        np.random.shuffle(options)
        for pos in options:
            if all(not _aligned(pos[0] - rr, pos[1] - cc) for rr, cc in chosen):
                chosen.append(pos)
                break
        if len(chosen) == target:
            return chosen
    return None


def _make_shape_patch(
    shape_color: int,
    shadow_color: int | None,
    obstacle_color: int | None,
    obstacle_count: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    margin = OBSTACLE_MARGIN if obstacle_color is not None and obstacle_count > 0 else 0
    base = _base_shape_patch(shape_color, shadow_color, margin=margin)
    if base is None:
        return None
    base_inp, shifted_vertices, _ = base
    base_out = main(base_inp)

    inp = base_inp.copy()
    if obstacle_color is not None and obstacle_count > 0:
        obstacle_cells = _choose_obstacles(inp, shape_color, obstacle_count)
        if obstacle_cells is None:
            return None
        for r, c in obstacle_cells:
            if inp[r, c] != BG:
                return None
            inp[r, c] = int(obstacle_color)
        out = main(inp)
        if np.count_nonzero(out != base_out) < 2:
            return None
        blocked, detoured, _ = _route_stats_for_color(inp, shape_color, BG)
        if blocked < 1 or detoured < 1:
            return None
        pts = [tuple(map(int, p)) for p in np.argwhere(inp == obstacle_color)]
        if not (1 <= len(pts) <= 3):
            return None
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                if _aligned(pts[j][0] - pts[i][0], pts[j][1] - pts[i][1]):
                    return None
                if any(max(abs(pts[i][0] - vr), abs(pts[i][1] - vc)) <= 1 for vr, vc in shifted_vertices):
                    return None
                if any(max(abs(pts[j][0] - vr), abs(pts[j][1] - vc)) <= 1 for vr, vc in shifted_vertices):
                    return None
        return inp, out

    if not np.array_equal(base_out, main(inp)):
        return None
    return inp, base_out


def _sample_palette(shape_count: int, has_obstacle: bool) -> tuple[list[int], list[int], int | None] | None:
    max_shadows = min(MAX_SHADOWS_CAP, shape_count, len(COLORS) - shape_count - int(has_obstacle))
    if max_shadows < 0:
        return None
    shadow_count = int(np.random.randint(max_shadows + 1))
    total = shape_count + shadow_count + int(has_obstacle)
    palette = list(np.random.permutation(COLORS)[:total])
    shape_colors = palette[:shape_count]
    shadow_colors = palette[shape_count : shape_count + shadow_count]
    obstacle_color = int(palette[-1]) if has_obstacle else None
    return shape_colors, shadow_colors, obstacle_color


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


def _gen_attempt() -> np.ndarray | None:
    h, w = _sample_canvas_size()
    out = np.full((h, w), BG, dtype=int)
    inp = np.full((h, w), BG, dtype=int)

    has_obstacle = bool(np.random.rand() < OBSTACLE_PROB)
    shape_count = int(np.random.randint(*SHAPE_COUNT_RANGE))

    palette = _sample_palette(shape_count, has_obstacle)
    if palette is None:
        return None
    shape_colors, shadow_colors, obstacle_color = palette
    shadow_map = _assign_shadows(shape_count, shadow_colors)
    obstacle_target = int(np.random.randint(shape_count)) if has_obstacle else -1
    obstacle_count = int(np.random.randint(*OBSTACLE_COUNT_RANGE)) if has_obstacle else 0

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
                    obstacle_count if idx == obstacle_target else 0,
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

    if has_obstacle and obstacle_color is not None:
        obstacle_pts = [tuple(map(int, p)) for p in np.argwhere(inp == obstacle_color)]
        if not (1 <= len(obstacle_pts) <= 3):
            return None
        for i in range(len(obstacle_pts)):
            for j in range(i + 1, len(obstacle_pts)):
                if _aligned(obstacle_pts[j][0] - obstacle_pts[i][0], obstacle_pts[j][1] - obstacle_pts[i][1]):
                    return None
        detoured_any = False
        for shape_color in shape_colors:
            _, detoured, _ = _route_stats_for_color(inp, int(shape_color), BG)
            detoured_any |= detoured > 0
        if not detoured_any:
            return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"generate_input failed after {GENERATE_RETRIES} attempts")
