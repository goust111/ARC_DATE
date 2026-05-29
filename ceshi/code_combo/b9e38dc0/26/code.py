# Combo variant of b9e38dc0 (index 26)
# combo_spec: D7/0+D10/1
# Name: 障碍偏转+串行处理

import heapq
import random

import numpy as np
from scipy.ndimage import distance_transform_cdt

from arc_utils import components_mask, get_bg, neighbors4, neighbors8

# Design analysis
# ===============
# a) Concrete behavior and trace
#    - The fused task contains 2 same-color funnel walls that open in the same cardinal
#      direction, plus one fill seed blob inside each wall and several extra singleton
#      pixels elsewhere. Those extra pixels are movers, not stationary obstacles.
#    - Original single-sprite logic would fill one funnel and leave extra pixels alone
#      (or in the D7 single variant, route them around one finished sprite). Original
#      D10 logic would serially fill multiple funnels but still stop at fill behavior.
#    - Here the order is: first solve all funnels serially from edge-near to edge-far,
#      then move every singleton toward that same border. Example with `rad = right`:
#      the rightmost funnel fills first and occupies a wide corridor near the right edge;
#      the left funnel fills second and is truncated by that earlier territory. Only after
#      this serial fill is frozen do the singleton movers travel rightward. A mover that
#      would have gone straight into the finished funnel mass now sidesteps up/down,
#      hugs the obstacle boundary, and reaches the right border in a shifted row.
#    - The visible difference from a naive "parallel funnels + straight movers" baseline
#      is twofold: later funnel fill is shorter, and mover landing rows depend on that
#      shortened serial obstacle shape.
# b) Complexity calibration
#    - The fusion keeps the original wall grammar, shared wall/fill colors, and one common
#      opening direction, so the puzzle still has a compact visual vocabulary.
#    - Complexity comes from exactly two coupled mechanisms that can be seen in 2-3
#      examples: near-edge funnel claims space first; singleton movers then detour around
#      the final claimed space. The generator restricts sprite count to 2 and mover count
#      to 2-4 so the board stays readable.
#    - Generator rejects weak cases. Each accepted sample must show both a substantial
#      serial-vs-parallel fill difference and at least one mover whose serial detour
#      endpoint differs from the endpoint it would get against the parallel obstacle.
# c) Implicit assumptions audited against the old solver
#    - Old code assumed one relevant wall component and one seed color blob. The fused
#      solver must collect all wall components of the wall color, collect all 4-connected
#      fill blobs of the fill color, and assign each seed blob to its nearest wall.
#    - Old flood-fill treated all fill-color cells as passable. That breaks seriality:
#      later funnels would pass through earlier filled territory. The new fill step allows
#      only background plus the current sprite's own seed cells.
#    - Old obstacle handling for D7 used a single finished sprite. The fused mover stage
#      must use the whole serially-produced union of wall+fill pixels as the obstacle
#      mask, otherwise D10 would not influence mover paths.
#    - The path finder assumes all funnels share one visible radiate direction. Generator
#      constructs both wall components with the same `rad` and keeps movers outside both
#      the serial and parallel fill zones so they do not perturb stage 1.
# d) Placement legality
#    - Wall components are rendered from local templates, kept 8-neighborhood separated,
#      and placed fully in-bounds with enough room toward the opening edge for radiation.
#    - Seed blobs are chosen from each wall interior near the closed end and checked not
#      to overlap walls or each other.
#    - Movers are placed only on cells that remain background under both serial and
#      parallel fill simulations. This guarantees they do not alter the fill geometry that
#      the generator validated before placing them.
# e) Visual observability
#    - Sorting is purely geometric: whichever wall opening is closer to the relevant grid
#      edge is processed first. No color id or hidden scalar is used.
#    - The same common `rad` governs both the fill order effect and the mover motion, so a
#      solver can infer one directional story from the examples instead of unrelated rules.
# f) Visual cleanliness
#    - There is one wall color, one fill color, and a small number of isolated movers.
#      No stationary obstacle clutter is added on top of the multi-sprite interaction.
#    - Movers are spaced apart and given unique landing cells, avoiding ambiguous overlaps.
# g) Change inferability
#    - Colors do not encode control flow. All wall components share the same wall color,
#      all seed blobs share the same fill color, and every extra singleton behaves as a
#      mover regardless of its specific color.
#    - Because at least one chosen mover changes endpoint between serial and parallel
#      obstacle geometry, the fusion is not a silent no-op of either constituent variant.
# h) New solver assumptions checked against generator
#    - `_collect_sprites` assumes each active wall has a nearby fill blob. Generator gives
#      each wall exactly one interior seed blob and keeps walls well separated.
#    - `_find_detour_path` assumes a forward/sideways monotone route exists from every
#      chosen mover to the target border around the serial obstacle union. Generator only
#      samples movers whose detour path exists and whose destination is unique.


def _sgn(x: int) -> int:
    return (x > 0) - (x < 0)


def _manhattan_min(a_pts: list[tuple[int, int]], b_pts: list[tuple[int, int]]) -> int:
    best = 10**9
    for ar, ac in a_pts:
        for br, bc in b_pts:
            d = abs(ar - br) + abs(ac - bc)
            if d < best:
                best = d
                if best == 0:
                    return 0
    return best


def _degree1_pixels(points: set[tuple[int, int]]) -> list[tuple[int, int]]:
    pts = set(points)
    return [
        (r, c)
        for r, c in pts
        if sum((rr, cc) in pts for rr, cc in neighbors8(r, c)) == 1
    ]


def _component_neighbors(comp: set[tuple[int, int]], r: int, c: int) -> list[tuple[int, int]]:
    out = []
    for rr, cc in neighbors8(r, c):
        if (rr, cc) in comp:
            out.append((rr, cc))
    return out


def _sprite_meta(comp: set[tuple[int, int]], shape: tuple[int, int]) -> dict:
    h, w = shape
    comp_mask = np.zeros(shape, dtype=bool)
    for r, c in comp:
        comp_mask[r, c] = True

    endpoints = [(r, c) for r, c in comp if len(_component_neighbors(comp, r, c)) == 1]
    assert len(endpoints) >= 2, "expected wall component with endpoints"

    def outward_dir(ep: tuple[int, int]) -> tuple[tuple[int, int], bool]:
        n1 = _component_neighbors(comp, *ep)[0]
        d1 = (_sgn(n1[0] - ep[0]), _sgn(n1[1] - ep[1]))
        d_use = d1
        is_12 = False
        n2s = [p for p in _component_neighbors(comp, *n1) if p != ep]
        if n2s:
            n2 = n2s[0]
            d2 = (_sgn(n2[0] - n1[0]), _sgn(n2[1] - n1[1]))
            if (d1[0] == 0) ^ (d1[1] == 0) and d2[0] != 0 and d2[1] != 0:
                d_use = d2
                is_12 = True
        return (-d_use[0], -d_use[1]), is_12

    out_info = [outward_dir(ep) for ep in endpoints]
    out_dirs = [d for d, _ in out_info]
    tip_12 = [flag for _, flag in out_info]
    rad = (_sgn(sum(d[0] for d in out_dirs)), _sgn(sum(d[1] for d in out_dirs)))

    barrier = comp_mask.copy()

    def add_ray(start: tuple[int, int], d: tuple[int, int]) -> None:
        rr, cc = start
        while True:
            rr += d[0]
            cc += d[1]
            if not (0 <= rr < h and 0 <= cc < w):
                break
            barrier[rr, cc] = True

    def add_ray_12(start: tuple[int, int], d: tuple[int, int]) -> None:
        rr, cc = start
        step = 0
        while True:
            step += 1
            if step % 2 == 1:
                rr += d[0]
                cc += d[1]
            else:
                if rad[0] != 0:
                    rr += d[0]
                else:
                    cc += d[1]
            if not (0 <= rr < h and 0 <= cc < w):
                break
            barrier[rr, cc] = True

    for ep, d, is_12 in zip(endpoints, out_dirs, tip_12):
        if is_12:
            add_ray_12(ep, d)
        else:
            add_ray(ep, d)

    if rad == (-1, 0):
        edge_dist = min(r for r, _ in endpoints)
    elif rad == (1, 0):
        edge_dist = (h - 1) - max(r for r, _ in endpoints)
    elif rad == (0, -1):
        edge_dist = min(c for _, c in endpoints)
    else:
        edge_dist = (w - 1) - max(c for _, c in endpoints)

    center = (
        sum(r for r, _ in comp) / len(comp),
        sum(c for _, c in comp) / len(comp),
    )
    return {
        "comp": comp,
        "comp_mask": comp_mask,
        "endpoints": endpoints,
        "rad": rad,
        "barrier": barrier,
        "edge_dist": int(edge_dist),
        "center": center,
    }


def _apply_shadows(filled_mask: np.ndarray, obstacles: np.ndarray, rad: tuple[int, int]) -> np.ndarray:
    out_mask = filled_mask.copy()
    h, w = out_mask.shape

    def shadow_process(coords: list[tuple[int, int]]) -> None:
        started = False
        blocked = False
        for r, c in coords:
            if out_mask[r, c]:
                started = True
            if started and obstacles[r, c]:
                blocked = True
                continue
            if blocked and out_mask[r, c]:
                out_mask[r, c] = False

    if rad == (0, 0):
        return out_mask

    dr, dc = rad
    if dr != 0 and dc == 0:
        for c in range(w):
            rows = range(h) if dr > 0 else range(h - 1, -1, -1)
            shadow_process([(r, c) for r in rows])
    elif dc != 0 and dr == 0:
        for r in range(h):
            cols = range(w) if dc > 0 else range(w - 1, -1, -1)
            shadow_process([(r, c) for c in cols])
    else:
        if dr == dc:
            for k in range(-(w - 1), h):
                coords = [(r, r - k) for r in range(h) if 0 <= r - k < w]
                coords.sort(key=lambda rc: rc[0], reverse=(dr < 0))
                shadow_process(coords)
        else:
            for k in range(h + w - 1):
                coords = [(r, k - r) for r in range(h) if 0 <= k - r < w]
                coords.sort(key=lambda rc: rc[0], reverse=(dr < 0))
                shadow_process(coords)
    return out_mask


def _infer_roles(g: np.ndarray) -> tuple[int, int, int] | None:
    vals = np.unique(g)
    bg = get_bg(g)
    non_bg = [int(v) for v in vals if int(v) != bg]
    if len(non_bg) < 2:
        return None

    wall = max(non_bg, key=lambda c: int((g == c).sum()))
    other_colors = [c for c in non_bg if c != wall]
    wall_points = [(int(r), int(c)) for r, c in zip(*np.where(g == wall))]

    def min_dist_to_wall(color: int) -> tuple[int, int]:
        pts = [(int(r), int(c)) for r, c in zip(*np.where(g == color))]
        return _manhattan_min(pts, wall_points), len(pts)

    fill = min(other_colors, key=lambda c: min_dist_to_wall(c))
    return bg, wall, fill


def _collect_sprites(g: np.ndarray, wall: int, fill: int) -> list[dict]:
    h, w = g.shape
    wall_comps = [
        set((int(r), int(c)) for r, c in pts)
        for pts in components_mask(g == wall, connectivity=8)
    ]
    fill_comps = [
        [(int(r), int(c)) for r, c in pts]
        for pts in components_mask(g == fill, connectivity=4)
    ]
    assert wall_comps, "expected wall components"
    assert fill_comps, "expected fill components"

    metas = [_sprite_meta(comp, (h, w)) for comp in wall_comps]
    assigned: dict[int, list[tuple[int, int]]] = {i: [] for i in range(len(metas))}
    for seed_comp in fill_comps:
        idx = min(
            range(len(metas)),
            key=lambda i: (
                _manhattan_min(seed_comp, list(metas[i]["comp"])),
                metas[i]["edge_dist"],
            ),
        )
        assigned[idx].extend(seed_comp)

    active = []
    for i, meta in enumerate(metas):
        if not assigned[i]:
            continue
        seed_mask = np.zeros_like(g, dtype=bool)
        for r, c in assigned[i]:
            seed_mask[r, c] = True
        active.append(
            {
                **meta,
                "seed_points": assigned[i],
                "seed_mask": seed_mask,
            }
        )

    active.sort(
        key=lambda m: (
            m["edge_dist"],
            round(m["center"][0], 3),
            round(m["center"][1], 3),
        )
    )
    return active


def _fill_for_sprite(occupancy: np.ndarray, bg: int, sprite: dict) -> np.ndarray:
    passable = (~sprite["barrier"]) & ((occupancy == bg) | sprite["seed_mask"])
    q = list(sprite["seed_points"])
    seen = np.zeros_like(passable, dtype=bool)
    for r, c in q:
        if passable[r, c]:
            seen[r, c] = True
    while q:
        r, c = q.pop()
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < occupancy.shape[0] and 0 <= cc < occupancy.shape[1]:
                if passable[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    q.append((rr, cc))

    filled_mask = seen & (occupancy == bg)
    obstacles = (occupancy != bg) & (~sprite["comp_mask"]) & (~sprite["seed_mask"])
    return _apply_shadows(filled_mask, obstacles, sprite["rad"])


def _solve_serial_fill_core(g: np.ndarray) -> dict | None:
    roles = _infer_roles(g)
    if roles is None:
        return None
    bg, wall, fill = roles
    sprites = _collect_sprites(g, wall, fill)
    if not sprites:
        return None
    common_rad = sprites[0]["rad"]
    if any(sprite["rad"] != common_rad for sprite in sprites):
        return None

    out = np.array(g, copy=True)
    per_sprite_masks = []
    for sprite in sprites:
        mask = _fill_for_sprite(out, bg, sprite)
        per_sprite_masks.append(mask)
        out[mask] = fill

    return {
        "out": out,
        "masks": per_sprite_masks,
        "sprites": sprites,
        "bg": bg,
        "wall": wall,
        "fill": fill,
        "rad": common_rad,
        "sprite_mask": (out == wall) | (out == fill),
    }


def _solve_parallel_fill_core(g: np.ndarray) -> dict | None:
    roles = _infer_roles(g)
    if roles is None:
        return None
    bg, wall, fill = roles
    sprites = _collect_sprites(g, wall, fill)
    if not sprites:
        return None
    common_rad = sprites[0]["rad"]
    if any(sprite["rad"] != common_rad for sprite in sprites):
        return None

    all_fill = np.zeros_like(g, dtype=bool)
    per_sprite_masks = []
    for sprite in sprites:
        mask = _fill_for_sprite(g, bg, sprite)
        per_sprite_masks.append(mask)
        all_fill |= mask

    out = np.array(g, copy=True)
    out[all_fill] = fill
    return {
        "out": out,
        "masks": per_sprite_masks,
        "sprites": sprites,
        "bg": bg,
        "wall": wall,
        "fill": fill,
        "rad": common_rad,
        "sprite_mask": (out == wall) | (out == fill),
    }


def _goal_reached(r: int, c: int, rad: tuple[int, int], h: int, w: int) -> bool:
    if rad == (-1, 0):
        return r == 0
    if rad == (1, 0):
        return r == h - 1
    if rad == (0, -1):
        return c == 0
    return c == w - 1


def _detour_moves(rad: tuple[int, int]) -> list[tuple[int, int]]:
    if rad[0] != 0:
        return [rad, (0, -1), (0, 1)]
    return [rad, (-1, 0), (1, 0)]


def _find_detour_path(
    start: tuple[int, int],
    sprite_mask: np.ndarray,
    rad: tuple[int, int],
) -> list[tuple[int, int]] | None:
    h, w = sprite_mask.shape
    sr, sc = start
    if sprite_mask[sr, sc]:
        return None

    moves = _detour_moves(rad)
    start_state = (sr, sc, 0)
    best: dict[tuple[int, int, int], tuple[int, int, int]] = {start_state: (0, 0, 0)}
    prev: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    heap: list[tuple[tuple[int, int, int], int, int, int]] = [((0, 0, 0), sr, sc, 0)]
    goal_state: tuple[int, int, int] | None = None

    while heap:
        cost, r, c, dprev = heapq.heappop(heap)
        if cost != best.get((r, c, dprev)):
            continue
        if _goal_reached(r, c, rad, h, w) and (r, c) != (sr, sc):
            goal_state = (r, c, dprev)
            break
        for nd, (dr, dc) in enumerate(moves):
            rr, cc = r + dr, c + dc
            if not (0 <= rr < h and 0 <= cc < w):
                continue
            if sprite_mask[rr, cc]:
                continue
            ncost = (
                cost[0] + (1 if nd != 0 else 0),
                cost[1] + 1,
                cost[2] + (0 if nd == dprev else 1),
            )
            key = (rr, cc, nd)
            if ncost < best.get(key, (10**9, 10**9, 10**9)):
                best[key] = ncost
                prev[key] = (r, c, dprev)
                heapq.heappush(heap, (ncost, rr, cc, nd))

    if goal_state is None:
        return None

    path = []
    cur = goal_state
    while True:
        r, c, _ = cur
        path.append((r, c))
        if (r, c) == (sr, sc):
            break
        cur = prev[cur]
    path.reverse()
    return path


def _path_side_steps(path: list[tuple[int, int]], rad: tuple[int, int]) -> int:
    if len(path) < 2:
        return 0
    return sum(1 for (r1, c1), (r2, c2) in zip(path, path[1:]) if (r2 - r1, c2 - c1) != rad)


def _ray_hits_sprite(start: tuple[int, int], sprite_mask: np.ndarray, rad: tuple[int, int]) -> bool:
    h, w = sprite_mask.shape
    r, c = start
    while True:
        r += rad[0]
        c += rad[1]
        if not (0 <= r < h and 0 <= c < w):
            return False
        if sprite_mask[r, c]:
            return True


def _extra_pixels(g: np.ndarray, bg: int, wall: int, fill: int) -> list[tuple[int, int, int]]:
    coords = np.argwhere((g != bg) & (g != wall) & (g != fill))
    return [(int(r), int(c), int(g[r, c])) for r, c in coords]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill multiple same-direction funnels in a visible near-edge order, then route the leftover singleton pixels around the finished obstacle mass.

    Input structure:
    - The most common color is background. The most frequent non-background color forms 2 separated funnel-like wall components that open in the same direction.
    - A blob of another color lies inside each wall near its closed end. Any remaining non-wall, non-seed pixels are isolated movers placed outside the eventual fill zones.

    Rule:
    - Infer each wall component's opening direction from its endpoints and sort the components by how close that opening is to the corresponding grid edge.
    - Process those wall+seed sprites serially in that order: each one flood-fills its own interior and radiates outward only through cells that are still background, so earlier filled territory blocks later sprites.
    - After the serial fill is complete, move every isolated pixel toward the same target border. If its straight route would hit the finished wall+fill union, it detours sideways along the obstacle boundary and then continues forward until it reaches the border.

    Key insight: The important object is the serially built obstacle union. The nearer funnel claims space first, reshaping the final barrier, and the singleton movers must be routed around that reshaped barrier rather than stopping in front of it.
    """
    g = np.array(grid, copy=True)
    solved = _solve_serial_fill_core(g)
    if solved is None:
        return g

    out = solved["out"].copy()
    bg = solved["bg"]
    wall = solved["wall"]
    fill = solved["fill"]
    rad = solved["rad"]
    sprite_mask = solved["sprite_mask"]

    movers = _extra_pixels(g, bg, wall, fill)
    placements: list[tuple[tuple[int, int], int]] = []
    for r, c, color in movers:
        path = _find_detour_path((r, c), sprite_mask, rad)
        placements.append((path[-1] if path is not None else (r, c), color))

    for r, c, _color in movers:
        out[r, c] = bg
    for (rr, cc), color in placements:
        out[rr, cc] = color
    return out


# --------------- generator ---------------

# Canvas size bounds.
GRID_H_VERT = (16, 22)
GRID_H_HORIZ = (12, 18)
GRID_W = (20, 24)

# Wall geometry.
WALL_PIXELS_MIN = 17
WALL_PIXELS_MAX = 24
DEPTH_VERT = (4, 6)
DEPTH_HORIZ = (5, 7)
GAP_VERT = (6, 7)
GAP_HORIZ = (5, 7)
TIP_12_PROB = 0.5

# Placement / sprite count.
SPRITE_COUNT = 2
OPEN_END_MARGIN = 3
AXIS_DISTANCE_DELTA = (6, 8)
CROSS_OFFSET_CHOICES = [-2, -1, 1, 2]

# Fill seed.
SEED_SIZE_WEIGHTS = [1, 1, 1, 2, 2, 3, 4, 5]
SEED_DEPTH_LOOKBACK = 3

# Movers.
MIN_MOVERS = 2
MAX_MOVERS = 4
MIN_CHANGED_MOVERS = 1
MOVER_MIN_SEP = 2
MOVER_END_SEP = 1
MOVER_WALL_DIST = 3

# Validation.
MIN_NEW_FILL = 20
MIN_SERIAL_BLOCKED = 8
PALETTE_SIZE = (4, 5)
GEN_RETRIES = 500
LOCAL_LAYOUT_RETRIES = 30
LOCAL_WALL_RETRIES = 12
SEED_PICK_RETRIES = 8


def _make_arm(depth, start_v, target_v, extra_steps, tip_12=False):
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


def _build_local_wall(rad, h, w):
    if rad[0]:
        max_depth = min(DEPTH_VERT[1], h - 8)
        if max_depth < DEPTH_VERT[0]:
            return None
        depth = random.randint(DEPTH_VERT[0], max_depth)
        gap = random.randint(*GAP_VERT)
    else:
        max_depth = min(DEPTH_HORIZ[1], w - 8)
        if max_depth < DEPTH_HORIZ[0]:
            return None
        depth = random.randint(DEPTH_HORIZ[0], max_depth)
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

    if not (WALL_PIXELS_MIN <= len(pts) <= WALL_PIXELS_MAX):
        return None
    if len(_degree1_pixels(pts)) != 2:
        return None
    return pts, depth, gap


def _map_local(u, v, rad, anchor):
    r0, c0 = anchor
    if rad == (-1, 0):
        return r0 + u, c0 + v
    if rad == (1, 0):
        return r0 - u, c0 + v
    if rad == (0, -1):
        return r0 + v, c0 + u
    return r0 + v, c0 - u


def _local_interior(wall, depth):
    by_row = {}
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


def _grow_seed_blob(seed, interior_set):
    blob = {seed}
    target_size = random.choice(SEED_SIZE_WEIGHTS)
    frontier = [seed]
    while len(blob) < target_size and frontier:
        u, v = frontier.pop(random.randrange(len(frontier)))
        nbrs = [(u - 1, v), (u + 1, v), (u, v - 1), (u, v + 1)]
        random.shuffle(nbrs)
        for q in nbrs:
            if q in interior_set and q not in blob:
                blob.add(q)
                frontier.append(q)
                if len(blob) >= target_size:
                    break
    return blob


def _touches_border(mask):
    return bool(mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any())


def _in_bounds_all(cells: list[tuple[int, int]], h: int, w: int) -> bool:
    return all(0 <= r < h and 0 <= c < w for r, c in cells)


def _touches_existing(cells: list[tuple[int, int]], occupied: set[tuple[int, int]]) -> bool:
    for r, c in cells:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if (rr, cc) in occupied:
                    return True
    return False


def _axis_limit(rad, h, w):
    return h if rad[0] else w


def _cross_limit(rad, h, w):
    return w if rad[0] else h


def _anchor_from_dist(rad, h, w, depth, gap, edge_dist, cross_start):
    if rad == (-1, 0):
        return edge_dist, cross_start
    if rad == (1, 0):
        return (h - 1) - edge_dist, cross_start
    if rad == (0, -1):
        return cross_start, edge_dist
    return cross_start, (w - 1) - edge_dist


def _render_wall(local_wall, rad, anchor):
    return [_map_local(u, v, rad, anchor) for u, v in local_wall]


def _place_sprites(rad, h, w, sprites_local):
    axis_cap = _axis_limit(rad, h, w)
    cross_cap = _cross_limit(rad, h, w)
    depths = [item["depth"] for item in sprites_local]
    max_far = axis_cap - max(depths) - 3
    if max_far < OPEN_END_MARGIN + AXIS_DISTANCE_DELTA[0]:
        return None

    for _ in range(120):
        near_dist = random.randint(OPEN_END_MARGIN, OPEN_END_MARGIN + 1)
        far_min = near_dist + AXIS_DISTANCE_DELTA[0]
        far_max = min(max_far, near_dist + AXIS_DISTANCE_DELTA[1])
        if far_min > far_max:
            continue
        edge_dists = [near_dist, random.randint(far_min, far_max)]

        occupied: set[tuple[int, int]] = set()
        placed = []
        common_center = random.randint(3, cross_cap - 4)
        offsets = [0, random.choice(CROSS_OFFSET_CHOICES)]

        ok = True
        for idx, item in enumerate(sprites_local):
            depth = item["depth"]
            gap = item["gap"]
            wall = item["wall"]

            candidates = [offsets[idx]]
            around = CROSS_OFFSET_CHOICES[:]
            random.shuffle(around)
            candidates += around

            accepted = None
            for offset in candidates:
                desired_center = common_center + offset
                cross_start = desired_center - (gap // 2)
                cross_start = max(0, min(cross_start, cross_cap - gap - 1))
                anchor = _anchor_from_dist(rad, h, w, depth, gap, edge_dists[idx], cross_start)
                cells = _render_wall(wall, rad, anchor)
                if not _in_bounds_all(cells, h, w):
                    continue
                if _touches_existing(cells, occupied):
                    continue
                accepted = (anchor, cells, cross_start)
                break

            if accepted is None:
                ok = False
                break

            anchor, cells, cross_start = accepted
            occupied.update(cells)
            placed.append(
                {
                    **item,
                    "anchor": anchor,
                    "cells": cells,
                    "edge_dist": edge_dists[idx],
                    "cross_start": cross_start,
                }
            )

        if ok:
            return placed
    return None


def _choose_seed_cells(sprite):
    wall = sprite["wall"]
    depth = sprite["depth"]
    interior = _local_interior(wall, depth)
    if not interior:
        return None
    wall_set = set(wall)
    adjacent = [
        p
        for p in interior
        if any((p[0] + dr, p[1] + dc) in wall_set for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)])
    ]
    seed_pool = [p for p in adjacent if p[0] >= max(1, depth - SEED_DEPTH_LOOKBACK)]
    if not seed_pool:
        return None
    blob = _grow_seed_blob(random.choice(seed_pool), set(interior))
    return [_map_local(u, v, sprite["rad"], sprite["anchor"]) for u, v in blob]


def _chebyshev_far_enough(pos: tuple[int, int], placed: list[tuple[int, int]], min_sep: int) -> bool:
    r, c = pos
    return all(max(abs(r - rr), abs(c - cc)) >= min_sep for rr, cc in placed)


def _candidate_movers(base_grid: np.ndarray, serial: dict, parallel: dict) -> list[dict]:
    bg = serial["bg"]
    wall = serial["wall"]
    rad = serial["rad"]
    serial_mask = serial["sprite_mask"]
    parallel_mask = parallel["sprite_mask"]
    h, w = base_grid.shape

    wall_mask = base_grid == wall
    wall_dist = distance_transform_cdt(~wall_mask, metric="taxicab")
    base_bg = (
        (base_grid == bg)
        & (serial["out"] == bg)
        & (parallel["out"] == bg)
        & (~serial_mask)
        & (~parallel_mask)
    )
    coords = [(int(r), int(c)) for r, c in zip(*np.where(base_bg & (wall_dist >= MOVER_WALL_DIST)))]
    random.shuffle(coords)

    records = []
    for r, c in coords:
        if _goal_reached(r, c, rad, h, w):
            continue
        if not _ray_hits_sprite((r, c), serial_mask, rad):
            continue
        serial_path = _find_detour_path((r, c), serial_mask, rad)
        if serial_path is None:
            continue
        serial_side_steps = _path_side_steps(serial_path, rad)
        if serial_side_steps == 0:
            continue

        parallel_path = _find_detour_path((r, c), parallel_mask, rad)
        parallel_end = None if parallel_path is None else parallel_path[-1]
        records.append(
            {
                "start": (r, c),
                "serial_path": serial_path,
                "serial_end": serial_path[-1],
                "serial_side_steps": serial_side_steps,
                "parallel_path": parallel_path,
                "parallel_end": parallel_end,
                "changed_end": parallel_end != serial_path[-1],
                "changed_path": parallel_path is None or parallel_path != serial_path,
            }
        )
    return records


def _pick_movers(records: list[dict]) -> list[dict]:
    if len(records) < MIN_MOVERS:
        return []

    changed = [rec for rec in records if rec["changed_end"]]
    stable = [rec for rec in records if not rec["changed_end"]]
    random.shuffle(changed)
    random.shuffle(stable)
    changed.sort(key=lambda rec: (-rec["serial_side_steps"], rec["serial_end"]))
    stable.sort(key=lambda rec: (-rec["serial_side_steps"], rec["serial_end"]))

    chosen = []
    used_starts: list[tuple[int, int]] = []
    used_ends: list[tuple[int, int]] = []
    blocked_cells: set[tuple[int, int]] = set()
    target_n = random.randint(MIN_MOVERS, min(MAX_MOVERS, len(records)))

    def try_add(rec: dict) -> bool:
        start = rec["start"]
        end = rec["serial_end"]
        if start in blocked_cells or end in blocked_cells:
            return False
        if not _chebyshev_far_enough(start, used_starts, MOVER_MIN_SEP):
            return False
        if not _chebyshev_far_enough(end, used_ends, MOVER_END_SEP):
            return False
        chosen.append(rec)
        used_starts.append(start)
        used_ends.append(end)
        blocked_cells.add(start)
        blocked_cells.add(end)
        return True

    for rec in changed:
        if sum(1 for item in chosen if item["changed_end"]) >= MIN_CHANGED_MOVERS:
            break
        try_add(rec)

    for rec in changed + stable:
        if len(chosen) >= target_n:
            break
        if rec in chosen:
            continue
        try_add(rec)

    if len(chosen) < MIN_MOVERS:
        return []
    if sum(1 for rec in chosen if rec["changed_end"]) < MIN_CHANGED_MOVERS:
        return []
    return chosen


def _variant_metrics(grid: np.ndarray) -> dict:
    serial = _solve_serial_fill_core(grid)
    parallel = _solve_parallel_fill_core(grid)
    if serial is None or parallel is None:
        return {
            "movers": 0,
            "rerouted": 0,
            "changed_end": 0,
            "avg_side_steps": 0.0,
            "serial_blocked": 0,
        }

    movers = _extra_pixels(grid, serial["bg"], serial["wall"], serial["fill"])
    rerouted = 0
    changed_end = 0
    side_steps = []
    for r, c, _color in movers:
        serial_path = _find_detour_path((r, c), serial["sprite_mask"], serial["rad"])
        parallel_path = _find_detour_path((r, c), parallel["sprite_mask"], parallel["rad"])
        if serial_path is None:
            continue
        steps = _path_side_steps(serial_path, serial["rad"])
        if steps > 0:
            rerouted += 1
            side_steps.append(steps)
        serial_end = serial_path[-1]
        parallel_end = None if parallel_path is None else parallel_path[-1]
        if parallel_end != serial_end:
            changed_end += 1

    serial_blocked = int(
        ((parallel["out"] == serial["fill"]) & (serial["out"] != serial["fill"]) & (grid == serial["bg"])).sum()
    )
    return {
        "movers": len(movers),
        "rerouted": rerouted,
        "changed_end": changed_end,
        "avg_side_steps": float(np.mean(side_steps)) if side_steps else 0.0,
        "serial_blocked": serial_blocked,
    }


def _gen_attempt():
    rad = random.choice([(-1, 0), (1, 0), (0, -1), (0, 1)])
    h = random.randint(*GRID_H_VERT) if rad[0] else random.randint(*GRID_H_HORIZ)
    w = random.randint(*GRID_W)

    for _ in range(LOCAL_LAYOUT_RETRIES):
        sprites_local = []
        ok = True
        for _sprite_idx in range(SPRITE_COUNT):
            built = None
            for _ in range(LOCAL_WALL_RETRIES):
                built = _build_local_wall(rad, h, w)
                if built is not None:
                    break
            if built is None:
                ok = False
                break
            wall, depth, gap = built
            sprites_local.append({"wall": wall, "depth": depth, "gap": gap, "rad": rad})
        if not ok:
            continue

        placed = _place_sprites(rad, h, w, sprites_local)
        if placed is None:
            continue

        palette = random.sample(range(10), random.randint(*PALETTE_SIZE))
        bg, wall_color, fill_color = palette[:3]
        mover_colors = palette[3:]
        if not mover_colors:
            continue

        grid = np.full((h, w), bg, dtype=int)
        bad = False
        for sprite in placed:
            for r, c in sprite["cells"]:
                if grid[r, c] != bg:
                    bad = True
                    break
            if bad:
                break
            for r, c in sprite["cells"]:
                grid[r, c] = wall_color
        if bad:
            continue

        all_seed_cells = []
        for sprite in placed:
            seed_cells = None
            for _ in range(SEED_PICK_RETRIES):
                seed_cells = _choose_seed_cells(sprite)
                if seed_cells is None:
                    continue
                if all(grid[r, c] == bg for r, c in seed_cells):
                    break
                seed_cells = None
            if seed_cells is None:
                bad = True
                break
            all_seed_cells.append(seed_cells)
        if bad:
            continue

        for seed_cells in all_seed_cells:
            for r, c in seed_cells:
                grid[r, c] = fill_color

        serial = _solve_serial_fill_core(grid)
        parallel = _solve_parallel_fill_core(grid)
        if serial is None or parallel is None:
            continue
        if len(serial["sprites"]) != SPRITE_COUNT:
            continue

        serial_blocked = int(
            ((parallel["out"] == fill_color) & (serial["out"] != fill_color) & (grid == bg)).sum()
        )
        if serial_blocked < MIN_SERIAL_BLOCKED:
            continue

        new_fill = (serial["out"] == fill_color) & (grid == bg)
        if int(new_fill.sum()) < MIN_NEW_FILL or not _touches_border(new_fill):
            continue

        records = _candidate_movers(grid, serial, parallel)
        chosen = _pick_movers(records)
        if len(chosen) < MIN_MOVERS:
            continue

        for i, rec in enumerate(chosen):
            r, c = rec["start"]
            grid[r, c] = mover_colors[i % len(mover_colors)]

        try:
            out = main(grid)
        except Exception:
            continue

        new_fill = (out == fill_color) & (grid == bg)
        if int(new_fill.sum()) < MIN_NEW_FILL or not _touches_border(new_fill):
            continue

        metrics = _variant_metrics(grid)
        if metrics["rerouted"] < MIN_MOVERS:
            continue
        if metrics["changed_end"] < MIN_CHANGED_MOVERS:
            continue
        if metrics["serial_blocked"] < MIN_SERIAL_BLOCKED:
            continue
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid b9e38dc0 combo variant input")
