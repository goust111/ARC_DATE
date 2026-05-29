# Combo variant of b9e38dc0 (index 24)
# combo_spec: D4/0+D11/0+D19/1
# Name: 边缘控制像素+颜色匹配关联+统一边框色

# Design analysis for the fused variant
# a) Concrete behavior and trace:
#    - Each input contains two same-colored wall sprites with the original half-enclosure geometry.
#    - Both sprites open toward the same grid edge, so border side alone cannot tell which control pixel
#      belongs to which sprite.
#    - Every sprite has two remote markers of one signal color:
#        1) a border control pixel on the opening-side edge,
#        2) an isolated singleton near that sprite's closed end.
#      The control pixel and singleton share the same color, and that color becomes the sprite's fill.
#    - Small trace:
#        original: internal seed blob -> nearest wall -> flood/radiate.
#        fused: blue/red border pixels are not physical seeds at all. The solver first color-matches each
#        border pixel to the same-colored isolated singleton, uses that singleton to choose the nearby wall,
#        reconstructs hidden interior start cells from the wall geometry, then floods/radiates with that color.
#      Because the two walls share one border color and face the same edge, this color pairing is the only
#      reliable association cue.
#    - D19 is integrated by moving the old "wall color -> ray width" parameter onto the paired signal color:
#      blue control+singleton pairs emit the original 1-pixel boundary rays; red pairs emit 2-pixel boundary
#      bands. Uniform wall color removes wall-color identity, so the paired signal color now carries both
#      object identity and the width parameter.
# b) Complexity calibration:
#    - Only two signal colors are used in one sample: blue and red. This keeps the mapping small and visible.
#    - The two wall sprites reuse the same local wall geometry and the same opening side, so the difference
#      in output beam width is attributable to signal color rather than to shape or direction changes.
#    - Control pixels are intentionally assigned to the two walls in reverse border order, so naive order or
#      nearest-edge pairing is wrong; color pairing is necessary but still human-readable because each wall has
#      a matching singleton tag of the same color near its closed end.
# c) Implicit assumption audit against the original solver:
#    - `wall = max(non_bg, key=count)` still works because all true walls share one color and dominate counts.
#    - "closest non-wall color to the wall is the fill seed" is false now: signal colors only appear as a
#      border pixel plus one isolated tag, never as internal seeds.
#    - "the seed chooses the wall by proximity" is false now: the isolated tag chooses the wall; the border
#      control chooses the color; both are joined by color equality.
#    - "wall color determines the width" is false now: all walls have the same color, so width must come from
#      the paired signal color instead.
#    - "one wall component is enough" is false now: the solver must process multiple same-colored wall
#      components separately.
# d) Placement legality:
#    - The generator builds one local wall and places two copies in disjoint halves of the grid.
#    - Markers are placed one cell beyond each closed end, away from the radiate zone.
#    - Border controls are placed on the opening-side edge and kept off the wall / boundary-ray pixels.
#      Obstacles avoid border cells, reconstructed seed cells, wall pixels, and tags.
# e) Visual observability:
#    - In every sample, both walls have the same border color and the same opening direction.
#    - Each wall has a nearby singleton whose color matches exactly one border control pixel.
#    - Blue and red outputs differ not only in hue but also in beam width, so the color->parameter mapping is
#      visible inside a single sample.
# f) Visual cleanliness:
#    - Only two sprites are used, with one tag per sprite and at most a few singleton obstacles.
#    - The sprites are split into separate halves so their radiation fans do not overlap.
#    - Obstacles remain sparse; tags sit near the closed ends instead of cluttering the shared opening edge.
# g) Change inferability:
#    - The only semantic colors are blue and red, reused consistently as control color, tag color, fill color,
#      and width selector. This repeated structure makes the mapping recoverable from 2-3 examples.
# 9) New solver assumptions and generator guarantees:
#    - Assumption: exactly two usable signal colors appear on the border and also as isolated interior tags.
#      Guarantee: generator uses one blue and one red pair and reserves those colors from obstacles/bg/walls.
#    - Assumption: each tag is closer to its own wall than to the other wall.
#      Guarantee: tags are placed immediately beyond their own closed ends.
#    - Assumption: order-only pairing would be wrong often enough to expose D11.
#      Guarantee: the two border controls are assigned in reverse border order relative to the wall order, and
#      generation rejects samples whose color-paired output matches the order-paired ablation.

from __future__ import annotations

import random

import numpy as np
from scipy.ndimage import distance_transform_cdt

from arc_utils import components_mask, get_bg, neighbors4, neighbors8


SIGNAL_RAY_WIDTH = {
    1: 1,  # blue signal -> thin rays
    2: 2,  # red signal -> thick rays
}
SIGNAL_COLORS = tuple(SIGNAL_RAY_WIDTH)
WALL_COMPONENT_MIN = 14


# -- Configuration constants --------------------------------------------------
GRID_H = (17, 18)
GRID_W = (19, 19)

WALL_PIXELS_MIN = 17
WALL_PIXELS_MAX = 25
DEPTH_VERT = (5, 10)
DEPTH_HORIZ = (6, 8)
GAP_VERT = (6, 7)
GAP_HORIZ = (5, 7)
TIP_12_PROB = 0.5
OPEN_END_MARGIN = 3

SEED_DEPTH_LOOKBACK = 3

MAX_OBSTACLES = 3
OBSTACLE_IN_FILL_PROB = 0.45
OBSTACLE_MIN_SEP = 2

MIN_NEW_FILL = 24
MIN_PAIR_DELTA = 10
MIN_WIDTH_DELTA = 6
GEN_RETRIES = 400


def _sgn(x: int) -> int:
    return (x > 0) - (x < 0)


def _degree1_pixels(points: set[tuple[int, int]]) -> list[tuple[int, int]]:
    pts = set(points)
    return [
        (r, c)
        for r, c in pts
        if sum((rr, cc) in pts for rr, cc in neighbors8(r, c)) == 1
    ]


def _comp_mask(shape: tuple[int, int], comp: set[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r, c in comp:
        mask[r, c] = True
    return mask


def _comp_neighbors(comp_mask: np.ndarray, r: int, c: int) -> list[tuple[int, int]]:
    h, w = comp_mask.shape
    out = []
    for rr, cc in neighbors8(r, c):
        if 0 <= rr < h and 0 <= cc < w and comp_mask[rr, cc]:
            out.append((rr, cc))
    return out


def _min_pair_dist(points: list[tuple[int, int]], comp: set[tuple[int, int]]) -> int:
    best = 10**9
    for r, c in points:
        for rr, cc in comp:
            d = abs(r - rr) + abs(c - cc)
            if d < best:
                best = d
                if best == 0:
                    return 0
    return best


def _component_geometry(comp: set[tuple[int, int]], shape: tuple[int, int]) -> dict[str, object]:
    comp_mask = _comp_mask(shape, comp)
    endpoints = [p for p in comp if len(_comp_neighbors(comp_mask, *p)) == 1]
    assert len(endpoints) == 2, "expected exactly two wall endpoints"

    def outward_dir(ep: tuple[int, int]) -> tuple[tuple[int, int], bool]:
        n1 = _comp_neighbors(comp_mask, *ep)[0]
        d1 = (_sgn(n1[0] - ep[0]), _sgn(n1[1] - ep[1]))
        d_use = d1
        is_12 = False
        n2s = [p for p in _comp_neighbors(comp_mask, *n1) if p != ep]
        if n2s:
            n2 = n2s[0]
            d2 = (_sgn(n2[0] - n1[0]), _sgn(n2[1] - n1[1]))
            if (d1[0] == 0) ^ (d1[1] == 0):
                if d2[0] != 0 and d2[1] != 0:
                    d_use = d2
                    is_12 = True
        return (-d_use[0], -d_use[1]), is_12

    out_info = [outward_dir(ep) for ep in endpoints]
    out_dirs = [d for d, _ in out_info]
    tip_12 = [flag for _, flag in out_info]
    rad = (_sgn(sum(d[0] for d in out_dirs)), _sgn(sum(d[1] for d in out_dirs)))
    return {
        "comp": comp,
        "comp_mask": comp_mask,
        "endpoints": endpoints,
        "out_dirs": out_dirs,
        "tip_12": tip_12,
        "rad": rad,
    }


def _inward_offset(
    ep: tuple[int, int],
    endpoints: list[tuple[int, int]],
    rad: tuple[int, int],
) -> tuple[int, int]:
    if rad[0] != 0:
        center = sum(c for _, c in endpoints) / len(endpoints)
        return (0, 1 if ep[1] < center else -1)
    center = sum(r for r, _ in endpoints) / len(endpoints)
    return (1 if ep[0] < center else -1, 0)


def _ray_points(
    start: tuple[int, int],
    d: tuple[int, int],
    rad: tuple[int, int],
    is_12: bool,
    shape: tuple[int, int],
) -> list[tuple[int, int]]:
    h, w = shape
    rr, cc = start
    pts = []
    step = 0
    while True:
        step += 1
        if not is_12:
            rr += d[0]
            cc += d[1]
        elif step % 2 == 1:
            rr += d[0]
            cc += d[1]
        elif rad[0] != 0:
            rr += d[0]
        else:
            cc += d[1]
        if not (0 <= rr < h and 0 <= cc < w):
            break
        pts.append((rr, cc))
    return pts


def _barrier_for_component(geom: dict[str, object], shape: tuple[int, int], width: int) -> np.ndarray:
    barrier = np.array(geom["comp_mask"], copy=True)
    endpoints = geom["endpoints"]
    out_dirs = geom["out_dirs"]
    tip_12 = geom["tip_12"]
    rad = geom["rad"]
    for ep, d, is_12 in zip(endpoints, out_dirs, tip_12):
        inward = _inward_offset(ep, endpoints, rad)
        for band in range(width):
            start = (ep[0] + inward[0] * band, ep[1] + inward[1] * band)
            if not (0 <= start[0] < shape[0] and 0 <= start[1] < shape[1]):
                continue
            if band > 0:
                barrier[start] = True
            for rr, cc in _ray_points(start, d, rad, is_12, shape):
                barrier[rr, cc] = True
    return barrier


def _anchor_from_comp(comp: set[tuple[int, int]], rad: tuple[int, int]) -> tuple[int, int]:
    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    if rad == (-1, 0):
        return min(rows), min(cols)
    if rad == (1, 0):
        return max(rows), min(cols)
    if rad == (0, -1):
        return min(rows), min(cols)
    return min(rows), max(cols)


def _to_local(comp: set[tuple[int, int]], rad: tuple[int, int]) -> set[tuple[int, int]]:
    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    r_min, r_max = min(rows), max(rows)
    c_min, c_max = min(cols), max(cols)
    local = set()
    for r, c in comp:
        if rad == (-1, 0):
            local.add((r - r_min, c - c_min))
        elif rad == (1, 0):
            local.add((r_max - r, c - c_min))
        elif rad == (0, -1):
            local.add((c - c_min, r - r_min))
        else:
            local.add((c_max - c, r - r_min))
    return local


def _local_interior(wall: set[tuple[int, int]], depth: int) -> list[tuple[int, int]]:
    by_row: dict[int, list[int]] = {}
    for u, v in wall:
        by_row.setdefault(u, []).append(v)
    interior = []
    for u in range(1, depth):
        if u not in by_row or len(by_row[u]) < 2:
            continue
        lo, hi = min(by_row[u]), max(by_row[u])
        for v in range(lo + 1, hi):
            if (u, v) not in wall:
                interior.append((u, v))
    return interior


def _infer_seed_positions(comp: set[tuple[int, int]], rad: tuple[int, int]) -> list[tuple[int, int]]:
    local_wall = _to_local(comp, rad)
    depth = max(u for u, _ in local_wall)
    interior = _local_interior(local_wall, depth)
    assert interior, "expected wall interior"

    wall_set = set(local_wall)
    adjacent = [
        p
        for p in interior
        if any((p[0] + dr, p[1] + dc) in wall_set for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)])
    ]
    pool = [p for p in adjacent if p[0] >= max(1, depth - SEED_DEPTH_LOOKBACK)]
    if not pool:
        pool = adjacent if adjacent else interior
    max_u = max(u for u, _ in pool)
    local_seed = [p for p in pool if p[0] == max_u]

    anchor = _anchor_from_comp(comp, rad)
    return [_map_local(u, v, rad, anchor) for u, v in local_seed]


def _flood_from_internal_seeds(
    g: np.ndarray,
    bg: int,
    barrier: np.ndarray,
    seed_positions: list[tuple[int, int]],
) -> np.ndarray:
    passable = (~barrier) & (g == bg)
    seen = np.zeros_like(passable, dtype=bool)
    q = []
    for r, c in seed_positions:
        if 0 <= r < g.shape[0] and 0 <= c < g.shape[1] and passable[r, c]:
            seen[r, c] = True
            q.append((r, c))
    assert q, "expected at least one background interior seed cell"
    while q:
        r, c = q.pop()
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1] and passable[rr, cc] and not seen[rr, cc]:
                seen[rr, cc] = True
                q.append((rr, cc))
    return seen


def _apply_shadows(
    out: np.ndarray,
    g: np.ndarray,
    filled_mask: np.ndarray,
    wall_color: int,
    fill_color: int,
    rad: tuple[int, int],
    bg: int,
) -> None:
    obstacles = (g != bg) & (g != fill_color) & (g != wall_color)
    h, w = g.shape

    def shadow_process(coords: list[tuple[int, int]]) -> None:
        started = False
        blocked = False
        for r, c in coords:
            if filled_mask[r, c]:
                started = True
            if started and obstacles[r, c]:
                blocked = True
                continue
            if blocked and filled_mask[r, c]:
                out[r, c] = bg
                filled_mask[r, c] = False

    if rad == (0, 0):
        return

    dr, dc = rad
    if dr != 0 and dc == 0:
        for c in range(w):
            rows = range(h) if dr > 0 else range(h - 1, -1, -1)
            shadow_process([(r, c) for r in rows])
    elif dc != 0 and dr == 0:
        for r in range(h):
            cols = range(w) if dc > 0 else range(w - 1, -1, -1)
            shadow_process([(r, c) for c in cols])
    elif dr == dc:
        for k in range(-(w - 1), h):
            coords = [(r, r - k) for r in range(h) if 0 <= r - k < w]
            coords.sort(key=lambda rc: rc[0], reverse=(dr < 0))
            shadow_process(coords)
    else:
        for k in range(h + w - 1):
            coords = [(r, k - r) for r in range(h) if 0 <= k - r < w]
            coords.sort(key=lambda rc: rc[0], reverse=(dr < 0))
            shadow_process(coords)


def _wall_components(g: np.ndarray, wall_color: int) -> list[dict[str, object]]:
    comps = []
    for pts in components_mask(g == wall_color, connectivity=8):
        comp = {(int(r), int(c)) for r, c in pts}
        if len(comp) < WALL_COMPONENT_MIN:
            continue
        if len(_degree1_pixels(comp)) != 2:
            continue
        comps.append({"points": comp, "geom": _component_geometry(comp, g.shape)})
    return comps


def _border_mask(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    mask = np.zeros(shape, dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _control_points_by_color(g: np.ndarray, bg: int, wall_color: int) -> dict[int, list[tuple[int, int]]]:
    mask = _border_mask(g.shape)
    out: dict[int, list[tuple[int, int]]] = {}
    for r, c in zip(*np.where(mask & (g != bg) & (g != wall_color))):
        color = int(g[int(r), int(c)])
        out.setdefault(color, []).append((int(r), int(c)))
    return out


def _marker_points_by_color(g: np.ndarray, bg: int, wall_color: int) -> dict[int, list[tuple[int, int]]]:
    mask = _border_mask(g.shape)
    out: dict[int, list[tuple[int, int]]] = {}
    for r, c in zip(*np.where((~mask) & (g != bg) & (g != wall_color))):
        color = int(g[int(r), int(c)])
        out.setdefault(color, []).append((int(r), int(c)))
    return out


def _border_side(ctrl_pt: tuple[int, int], shape: tuple[int, int]) -> tuple[int, int]:
    r, c = ctrl_pt
    h, w = shape
    if r == 0:
        return (-1, 0)
    if r == h - 1:
        return (1, 0)
    if c == 0:
        return (0, -1)
    if c == w - 1:
        return (0, 1)
    raise AssertionError("control pixel must lie on border")


def _component_sort_key(geom: dict[str, object]) -> float:
    comp = list(geom["comp"])
    rad = geom["rad"]
    if rad[0] != 0:
        return float(sum(c for _, c in comp) / len(comp))
    return float(sum(r for r, _ in comp) / len(comp))


def _assignment_by_color(
    wall_infos: list[dict[str, object]],
    controls: dict[int, list[tuple[int, int]]],
    markers: dict[int, list[tuple[int, int]]],
) -> list[tuple[int, int]]:
    shared = [color for color in SIGNAL_COLORS if color in controls and color in markers]
    assert len(shared) == 2, "expected exactly two matched signal colors"
    available = set(range(len(wall_infos)))
    pairs: list[tuple[int, int]] = []
    for color in sorted(shared):
        marker_pts = markers[color]
        assert marker_pts, "expected a marker for each signal color"
        idx = min(
            available or set(range(len(wall_infos))),
            key=lambda i: (_min_pair_dist(marker_pts, wall_infos[i]["points"]), -len(wall_infos[i]["points"])),
        )
        if idx in available:
            available.remove(idx)
        pairs.append((color, idx))
    return pairs


def _assignment_by_order(
    wall_infos: list[dict[str, object]],
    controls: dict[int, list[tuple[int, int]]],
) -> list[tuple[int, int]]:
    colors = [color for color in SIGNAL_COLORS if color in controls]
    assert len(colors) == 2, "expected exactly two border controls"
    control_items = []
    for color in colors:
        pt = controls[color][0]
        axis = pt[1] if pt[0] in (0, wall_infos[0]["geom"]["comp_mask"].shape[0] - 1) else pt[0]
        control_items.append((axis, color))
    control_items.sort()

    wall_order = sorted(range(len(wall_infos)), key=lambda i: _component_sort_key(wall_infos[i]["geom"]))
    return [(color, wall_idx) for (_, color), wall_idx in zip(control_items, wall_order)]


def _solve(
    grid: np.ndarray,
    ray_width_by_color: dict[int, int],
    *,
    pair_by_color: bool,
) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    bg = get_bg(g)
    non_bg = [int(v) for v in np.unique(g) if int(v) != bg]
    assert non_bg, "expected non-background colors"

    wall_color = max(non_bg, key=lambda c: int((g == c).sum()))
    wall_infos = _wall_components(g, wall_color)
    assert len(wall_infos) >= 2, "expected at least two wall components"

    controls = _control_points_by_color(g, bg, wall_color)
    markers = _marker_points_by_color(g, bg, wall_color)
    if pair_by_color:
        assignments = _assignment_by_color(wall_infos, controls, markers)
    else:
        assignments = _assignment_by_order(wall_infos, controls)

    for color, wall_idx in assignments:
        wall_info = wall_infos[wall_idx]
        geom = wall_info["geom"]
        width = ray_width_by_color.get(color, 1)
        barrier = _barrier_for_component(geom, g.shape, width)
        seeds = _infer_seed_positions(wall_info["points"], geom["rad"])
        filled_reachable = _flood_from_internal_seeds(g, bg, barrier, seeds)
        filled_mask = filled_reachable & (g == bg)
        out[filled_mask] = color
        _apply_shadows(out, g, filled_mask, wall_color, color, geom["rad"], bg)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill same-walled half-enclosures by pairing border control pixels with matching isolated tags.

    Input structure:
    - bg is the most common color. Two large 8-connected wall components share one wall color and open toward the same grid edge.
    - The only semantic non-wall colors are blue and red: each appears once as a border control pixel and once as an isolated singleton near one wall's closed end.
    - Any remaining non-bg singletons are obstacles; there is no internal fill seed blob.

    Rule:
    - Match each border control pixel to the isolated singleton of the same color, and use that singleton to choose the nearby wall sprite.
    - Reconstruct hidden interior start cells from that wall's geometry, flood the enclosed background, and continue radiating outward through the opening using the control color.
    - Blue signal pairs use 1-pixel boundary rays, while red signal pairs use 2-pixel boundary bands. Obstacles stay fixed and cast background shadows behind themselves along that wall's radiate direction.

    Key insight: The border pixel is only a remote trigger, and the shared wall color removes wall identity. Color equality between a border control and an isolated tag tells you which wall to activate, and that same paired color also selects the ray-width behavior.
    """
    return _solve(np.asarray(grid, dtype=int), SIGNAL_RAY_WIDTH, pair_by_color=True)


# --------------- generator ---------------

def _make_arm(depth: int, start_v: int, target_v: int, extra_steps: int, tip_12: bool = False) -> list[tuple[int, int]]:
    sign = 1 if target_v > start_v else -1
    n_side_steps = abs(target_v - start_v)
    chosen_rows = []
    if n_side_steps:
        if tip_12:
            chosen_rows = [2 * (i + 1) for i in range(n_side_steps)]
            extra_steps = 0
        else:
            chosen_rows = [1]
            if n_side_steps > 1:
                chosen_rows += random.sample(range(2, depth + 1), n_side_steps - 1)
            chosen_rows.sort()

    first_row = chosen_rows[0] if chosen_rows else 1
    extra_rows = [r for r in chosen_rows if r != first_row]
    random.shuffle(extra_rows)
    orthogonal_rows = set(extra_rows[:extra_steps])

    pts = [(0, start_v)]
    u, v = 0, start_v
    for row in range(1, depth + 1):
        u += 1
        if row in orthogonal_rows:
            pts.append((u, v))
            v += sign
        elif row in chosen_rows:
            v += sign
        pts.append((u, v))
    return pts


def _build_local_wall(rad: tuple[int, int], h: int, w: int) -> tuple[set[tuple[int, int]], int, int] | None:
    if rad[0]:
        max_depth = min(DEPTH_VERT[1], h - 6)
        if max_depth < DEPTH_VERT[0]:
            return None
        depth = random.randint(DEPTH_VERT[0], max_depth)
        gap = random.randint(*GAP_VERT)
    else:
        depth = random.randint(DEPTH_HORIZ[0], min(DEPTH_HORIZ[1], w - 6))
        gap = random.randint(GAP_HORIZ[0], min(GAP_HORIZ[1], h - 1))

    if gap == 5:
        left_bottom, right_bottom = 2, 3
    elif gap == 6:
        left_bottom, right_bottom = 2, 4
    else:
        left_bottom, right_bottom = random.choice([(2, 5), (3, 4)])

    bridge_extra = max(0, right_bottom - left_bottom - 1)
    base_count = 2 * (depth + 1) + bridge_extra
    max_extra_l = max(0, left_bottom - 1)
    max_extra_r = max(0, gap - right_bottom - 1)
    min_extra = max(0, WALL_PIXELS_MIN - base_count)
    max_extra = min(max_extra_l + max_extra_r, WALL_PIXELS_MAX - base_count)
    if min_extra > max_extra:
        return None

    extra = random.randint(min_extra, max_extra)
    extra_l = random.randint(max(0, extra - max_extra_r), min(max_extra_l, extra))
    extra_r = extra - extra_l

    max_n_side = max(left_bottom, gap - right_bottom)
    tip_12 = depth >= 2 * max_n_side and random.random() < TIP_12_PROB
    left = _make_arm(depth, 0, left_bottom, extra_l, tip_12=tip_12)
    right = _make_arm(depth, gap, right_bottom, extra_r, tip_12=tip_12)
    pts = set(left) | set(right)
    for v in range(left_bottom, right_bottom + 1):
        pts.add((depth, v))

    while len(pts) < WALL_PIXELS_MIN:
        candidates = [
            (u, v)
            for u in range(max(0, depth - 3), depth + 1)
            for v in range(gap + 1)
            if (u, v) not in pts
            and sum(
                (uu, vv) in pts
                for uu in range(u - 1, u + 2)
                for vv in range(v - 1, v + 2)
                if (uu, vv) != (u, v)
            ) >= 2
        ]
        random.shuffle(candidates)
        added = False
        for p in candidates:
            trial = pts | {p}
            if len(trial) <= WALL_PIXELS_MAX and len(_degree1_pixels(trial)) == 2:
                pts = trial
                added = True
                break
        if not added:
            return None

    if not (WALL_PIXELS_MIN <= len(pts) <= WALL_PIXELS_MAX) or len(_degree1_pixels(pts)) != 2:
        return None
    return pts, depth, gap


def _map_local(u: int, v: int, rad: tuple[int, int], anchor: tuple[int, int]) -> tuple[int, int]:
    r0, c0 = anchor
    if rad == (-1, 0):
        return r0 + u, c0 + v
    if rad == (1, 0):
        return r0 - u, c0 + v
    if rad == (0, -1):
        return r0 + v, c0 + u
    return r0 + v, c0 - u


def _sample_anchor_same_side(
    rad: tuple[int, int],
    depth: int,
    gap: int,
    h: int,
    w: int,
    slot: int,
) -> tuple[int, int] | None:
    if rad[0] != 0:
        if rad == (-1, 0):
            row_lo, row_hi = OPEN_END_MARGIN, h - depth - 3
        else:
            row_lo, row_hi = depth + 2, h - OPEN_END_MARGIN - 1
        if slot == 0:
            col_lo, col_hi = 0, w // 2 - gap - 2
        else:
            col_lo, col_hi = w // 2 + 1, w - gap - 1
    else:
        if rad == (0, -1):
            col_lo, col_hi = OPEN_END_MARGIN, w - depth - 3
        else:
            col_lo, col_hi = depth + 2, w - OPEN_END_MARGIN - 1
        if slot == 0:
            row_lo, row_hi = 0, h // 2 - gap - 1
        else:
            row_lo, row_hi = h // 2 + 1, h - gap - 1
    if row_lo > row_hi or col_lo > col_hi:
        return None
    return random.randint(row_lo, row_hi), random.randint(col_lo, col_hi)


def _render_wall(local_wall: set[tuple[int, int]], rad: tuple[int, int], anchor: tuple[int, int]) -> set[tuple[int, int]]:
    return {_map_local(u, v, rad, anchor) for u, v in local_wall}


def _marker_local(depth: int, gap: int) -> tuple[int, int]:
    return depth + 1, gap // 2


def _edge_positions(rad: tuple[int, int], h: int, w: int) -> list[tuple[int, int]]:
    if rad == (-1, 0):
        return [(0, c) for c in range(w)]
    if rad == (1, 0):
        return [(h - 1, c) for c in range(w)]
    if rad == (0, -1):
        return [(r, 0) for r in range(h)]
    return [(r, w - 1) for r in range(h)]


def _partition_value(p: tuple[int, int], rad: tuple[int, int]) -> int:
    return p[1] if rad[0] != 0 else p[0]


def _choose_control_pair(
    grid: np.ndarray,
    union_fill_zone: np.ndarray,
    union_barrier: np.ndarray,
    rad: tuple[int, int],
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    h, w = grid.shape
    edge = _edge_positions(rad, h, w)
    edge_zone = [p for p in edge if union_fill_zone[p[0], p[1]]]

    def edge_gap(p: tuple[int, int]) -> int:
        if not edge_zone:
            return 99
        if rad[0] != 0:
            return min(abs(p[1] - q[1]) for q in edge_zone)
        return min(abs(p[0] - q[0]) for q in edge_zone)

    candidates = [
        p
        for p in edge
        if grid[p[0], p[1]] == get_bg(grid)
        and not union_barrier[p[0], p[1]]
    ]
    preferred = [p for p in candidates if edge_gap(p) >= 2]
    pool = preferred if preferred else candidates
    if len(pool) < 2:
        return None
    pool = sorted(pool, key=lambda p: _partition_value(p, rad))
    best = None
    best_sep = -1
    for i in range(len(pool)):
        for j in range(i + 1, len(pool)):
            sep = _partition_value(pool[j], rad) - _partition_value(pool[i], rad)
            if sep > best_sep:
                best_sep = sep
                best = (pool[i], pool[j])
    return best


def _touches_border(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any())


def _obstacle_fits(pos: tuple[int, int], placed: list[tuple[int, int]]) -> bool:
    r, c = pos
    return all(max(abs(r - rr), abs(c - cc)) >= OBSTACLE_MIN_SEP for rr, cc in placed)


def _find_obstacle_candidates(
    grid: np.ndarray,
    bg: int,
    wall_mask: np.ndarray,
    nominal_fill_zone: np.ndarray,
    reserved_mask: np.ndarray,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    dist = distance_transform_cdt(~wall_mask, metric="taxicab")
    border = _border_mask(grid.shape)
    eligible = (grid == bg) & (dist > 1) & (~border) & (~reserved_mask)
    all_coords = [(int(r), int(c)) for r, c in zip(*np.where(eligible))]
    fill_coords = [p for p in all_coords if nominal_fill_zone[p[0], p[1]]]
    random.shuffle(all_coords)
    random.shuffle(fill_coords)
    return all_coords, fill_coords


def _place_obstacles(
    n_obstacles: int,
    candidates: list[tuple[int, int]],
    fill_candidates: list[tuple[int, int]],
    obstacle_colors: list[int],
    grid: np.ndarray,
) -> None:
    placed: list[tuple[int, int]] = []
    n_fill = random.randint(0, min(n_obstacles, len(fill_candidates)))
    if fill_candidates and random.random() < OBSTACLE_IN_FILL_PROB:
        n_fill = max(1, n_fill)
    for p in fill_candidates:
        if len(placed) >= n_fill:
            break
        if _obstacle_fits(p, placed):
            placed.append(p)
    placed_set = set(placed)
    for p in candidates:
        if len(placed) >= n_obstacles:
            break
        if p not in placed_set and _obstacle_fits(p, placed):
            placed.append(p)
            placed_set.add(p)
    for i, (r, c) in enumerate(placed):
        grid[r, c] = obstacle_colors[i % len(obstacle_colors)]


def _behavior_metrics(grid: np.ndarray) -> dict[str, int]:
    actual = main(grid)
    pairless = _solve(grid, SIGNAL_RAY_WIDTH, pair_by_color=False)
    widthless = _solve(grid, {1: 1, 2: 1}, pair_by_color=True)
    return {
        "pair_delta": int((actual != pairless).sum()),
        "width_delta": int((actual != widthless).sum()),
    }


def _build_two_sprite_grid() -> np.ndarray | None:
    h = random.randint(*GRID_H)
    w = random.randint(*GRID_W)
    rad = random.choice([(-1, 0), (1, 0), (0, -1), (0, 1)])

    built = _build_local_wall(rad, h, w)
    if built is None:
        return None
    local_wall, depth, gap = built

    anchors = []
    mapped_walls = []
    for slot in (0, 1):
        anchor = _sample_anchor_same_side(rad, depth, gap, h, w, slot)
        if anchor is None:
            return None
        anchors.append(anchor)
        mapped_walls.append(_render_wall(local_wall, rad, anchor))
    if mapped_walls[0] & mapped_walls[1]:
        return None

    bg_choices = [c for c in range(10) if c not in SIGNAL_COLORS]
    bg = random.choice(bg_choices)
    wall_choices = [c for c in bg_choices if c != bg]
    wall_color = random.choice(wall_choices)
    obstacle_pool = [c for c in range(10) if c not in (bg, wall_color, *SIGNAL_COLORS)]

    grid = np.full((h, w), bg, dtype=int)
    for wall_pts in mapped_walls:
        for r, c in wall_pts:
            grid[r, c] = wall_color

    signal_for_slot = list(SIGNAL_COLORS)
    random.shuffle(signal_for_slot)

    nominal_fill_zones = []
    reserved = np.zeros_like(grid, dtype=bool)
    union_barrier = np.zeros_like(grid, dtype=bool)

    for slot, wall_pts in enumerate(mapped_walls):
        geom = _component_geometry(wall_pts, grid.shape)
        barrier = _barrier_for_component(geom, grid.shape, SIGNAL_RAY_WIDTH[signal_for_slot[slot]])
        seeds = _infer_seed_positions(wall_pts, geom["rad"])
        nominal_fill = _flood_from_internal_seeds(grid, bg, barrier, seeds) & (grid == bg)
        nominal_fill_zones.append(nominal_fill)
        union_barrier |= barrier
        for r, c in seeds:
            reserved[r, c] = True

    union_fill = nominal_fill_zones[0] | nominal_fill_zones[1]

    for slot, anchor in enumerate(anchors):
        mu, mv = _marker_local(depth, gap)
        mr, mc = _map_local(mu, mv, rad, anchor)
        if not (0 <= mr < h and 0 <= mc < w):
            return None
        if grid[mr, mc] != bg or union_fill[mr, mc]:
            return None
        grid[mr, mc] = signal_for_slot[slot]
        reserved[mr, mc] = True

    control_pair = _choose_control_pair(grid, union_fill, union_barrier, rad)
    if control_pair is None:
        return None
    low_pos, high_pos = control_pair
    grid[high_pos[0], high_pos[1]] = signal_for_slot[0]
    grid[low_pos[0], low_pos[1]] = signal_for_slot[1]

    n_obstacles = random.randint(0, MAX_OBSTACLES)
    if n_obstacles:
        if not obstacle_pool:
            return None
        obstacle_colors = random.sample(obstacle_pool, min(len(obstacle_pool), max(1, n_obstacles)))
        wall_mask = grid == wall_color
        candidates, fill_candidates = _find_obstacle_candidates(grid, bg, wall_mask, union_fill, reserved)
        _place_obstacles(n_obstacles, candidates, fill_candidates, obstacle_colors, grid)

    try:
        out = main(grid)
    except Exception:
        return None

    new_fill = (grid == bg) & np.isin(out, SIGNAL_COLORS)
    if int(new_fill.sum()) < MIN_NEW_FILL or not _touches_border(new_fill):
        return None

    # Require both colors to participate so the pairing and width mapping are visible.
    for color in SIGNAL_COLORS:
        if int(((grid == bg) & (out == color)).sum()) < 8:
            return None

    metrics = _behavior_metrics(grid)
    if metrics["pair_delta"] < MIN_PAIR_DELTA or metrics["width_delta"] < MIN_WIDTH_DELTA:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _build_two_sprite_grid()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid b9e38dc0 combo input")
