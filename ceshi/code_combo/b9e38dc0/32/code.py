# Combo variant of b9e38dc0 (index 32)
# combo_spec: D3/1+D18/2
# Name: 加速移动+间歇涂色

from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

try:
    from arc_utils import components_mask, get_bg, neighbors4, neighbors8
except ImportError:
    sys.path.append(str(Path("/home/lixinyu/snap/new/solvers")))
    from arc_utils import components_mask, get_bg, neighbors4, neighbors8


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior trace
#    Keep the original scene grammar: a large wall component forms a half-enclosed
#    chamber and the closest other color is the fill seed. The fusion adds a second
#    role for the remaining non-wall colors:
#    - D3/1: the region that escapes through the opening is no longer painted as a
#      solid exterior wedge. After the ordinary fill + shadow logic runs, only every
#      k-th outward depth layer remains visible, where k is 2 for narrow/medium
#      openings and 3 for the widest opening.
#    - D18/2: every extra singleton pixel becomes a mover. It does not block the
#      flood-fill; instead, after the dashed fill is finalized, it travels along the
#      same radiate axis to the border. Its old landing cells revert to background
#      with increasing jump lengths 1, 2, 3, ... so the carved trail starts dense
#      and becomes sparse.
#    Example with downward radiate direction: if a mover starts on row 6 of a filled
#    column, the dense baseline motion would erase rows 6,7,8,9,10,... and stop with
#    its color on the bottom edge. The fused motion instead visits rows 6,7,9,12,...
#    then clamps to the border, so the background gaps grow from 0 skipped cells to
#    1 skipped cell to 2 skipped cells, while the surrounding fill keeps the
#    wall-controlled dashed rhythm from D3.
# b) Complexity calibration
#    Both new mechanisms stay low-branch:
#    - exterior dashes use only period 2 or 3, chosen from visible opening width;
#    - mover acceleration always follows the single sequence 1,2,3,...
#    The generator places 1-2 movers on distinct rows/columns of the radiate band so
#    a human can compare the uniform dashed fill against the non-uniform erased trail
#    without needing to infer color-dependent behavior.
# c) Implicit assumption audit
#    - The barrier rays must remain continuous. If they were dashed too, the BFS
#      would leak outside the intended wedge and destroy the original geometry.
#    - Shadow processing must run before exterior dash thinning; otherwise the
#      background gaps created by D3 would incorrectly terminate shadows.
#    - The original passable mask treated every non-fill, non-bg pixel as solid.
#      That breaks once singleton movers sit inside the would-be filled zone, so the
#      fused solver explicitly lets mover cells participate in reachability.
#    - Fill selection by "closest non-wall color" now assumes mover pixels are placed
#      farther from the wall than the seed blob. The generator enforces that by
#      inserting movers only after the seed is chosen and only on later fill cells.
#    - Classifying all remaining singleton components as movers is a new assumption.
#      The generator therefore avoids singleton obstacles entirely; any optional
#      obstacle-like pixels would have to be multi-cell components.
# d) Placement legality
#    The generator still builds the wall with the original geometry and validates it
#    through the solver. Mover start cells are chosen only from background cells that
#    the no-mover dashed solver would fill, are not on the border, have enough room
#    to show several accelerating jumps, and do not share the same orthogonal line.
# e) Visual observability
#    Across 2-4 examples a human sees two distinct sparse patterns:
#    - fill-colored exterior layers with constant cadence;
#    - background-colored mover cuts whose gaps increase over time and terminate in a
#      colored singleton on the border.
#    Because both use the same radiate direction, the difference is attributable to
#    spacing, not to hidden axis or color conventions.
# f) Visual cleanliness
#    No singleton blockers are generated, so extra lone pixels always mean movers.
#    Their trails only carve through the already-filled corridor, avoiding additional
#    overlapping object classes in the empty background.
# g) New solver assumptions mirrored by generation
#    `main()` assumes:
#    - wall is still the most frequent non-bg color;
#    - fill seed is still the non-wall color closest to the wall;
#    - every leftover singleton component is a mover;
#    - each mover has a clear straight radiate path to a border cell.
#    `generate_input()` constructs exactly that regime and rejects samples where the
#    D3 dash effect or the D18 acceleration effect is too weak to be visible.


def _sgn(x: int) -> int:
    return (x > 0) - (x < 0)


def _degree1_pixels(points):
    pts = set(points)
    return [(r, c) for r, c in pts if sum((rr, cc) in pts for rr, cc in neighbors8(r, c)) == 1]


def _dash_period(opening_span: int) -> int:
    return 2 if opening_span <= 7 else 3


def _dist_to_edge(pos: tuple[int, int], rad: tuple[int, int], h: int, w: int) -> int:
    r, c = pos
    dr, dc = rad
    if dr > 0:
        return h - 1 - r
    if dr < 0:
        return r
    if dc > 0:
        return w - 1 - c
    return c


def _edge_cell(pos: tuple[int, int], rad: tuple[int, int], h: int, w: int) -> tuple[int, int]:
    r, c = pos
    dr, dc = rad
    if dr > 0:
        return h - 1, c
    if dr < 0:
        return 0, c
    if dc > 0:
        return r, w - 1
    return r, 0


def _line_to_edge(start: tuple[int, int], rad: tuple[int, int], h: int, w: int) -> list[tuple[int, int]]:
    dr, dc = rad
    r, c = start
    coords = []
    while 0 <= r < h and 0 <= c < w:
        coords.append((int(r), int(c)))
        r += dr
        c += dc
    return coords


def _move_landings(
    start: tuple[int, int],
    rad: tuple[int, int],
    h: int,
    w: int,
    *,
    accelerating: bool,
) -> list[tuple[int, int]]:
    dr, dc = rad
    pos = (int(start[0]), int(start[1]))
    landings = [pos]
    step = 1
    while True:
        stride = step if accelerating else 1
        nxt = (pos[0] + dr * stride, pos[1] + dc * stride)
        if not (0 <= nxt[0] < h and 0 <= nxt[1] < w):
            edge = _edge_cell(pos, rad, h, w)
            if landings[-1] != edge:
                landings.append(edge)
            break
        landings.append((int(nxt[0]), int(nxt[1])))
        pos = (int(nxt[0]), int(nxt[1]))
        step += 1
    return landings


def _scene_signature(grid: np.ndarray) -> dict[str, object]:
    g = np.array(grid, copy=False)
    h, w = g.shape
    bg = int(get_bg(g))

    non_bg = [int(v) for v in np.unique(g) if int(v) != bg]
    assert non_bg, "expected at least one non-background color"

    wall = max(non_bg, key=lambda c: int((g == c).sum()))
    wall_mask = g == wall
    comps = [set((int(r), int(c)) for r, c in pts) for pts in components_mask(wall_mask, connectivity=8)]
    assert comps, "expected at least one wall connected component"

    other_colors = [c for c in non_bg if c != wall]
    assert other_colors, "expected at least one non-wall color"

    wall_points = [p for comp in comps for p in comp]

    def min_dist_to_wall(color: int) -> tuple[int, int]:
        pts = [(int(r), int(c)) for r, c in zip(*np.where(g == color))]
        best_d = 10**9
        for r, c in pts:
            for wr, wc in wall_points:
                d = abs(r - wr) + abs(c - wc)
                if d < best_d:
                    best_d = d
                    if best_d == 0:
                        return 0, len(pts)
        return best_d, len(pts)

    fill = min(other_colors, key=lambda c: (min_dist_to_wall(c)[0], min_dist_to_wall(c)[1]))
    fill_pos = [(int(r), int(c)) for r, c in zip(*np.where(g == fill))]
    if not fill_pos:
        return {
            "bg": bg,
            "wall": wall,
            "fill": fill,
            "fill_pos": [],
            "wall_comp": set(),
            "comp_mask": np.zeros_like(g, dtype=bool),
            "endpoints": [],
            "out_dirs": [],
            "tip_12": [],
            "rad": (0, 0),
            "open_proj": 0,
            "opening_span": 0,
            "movers": [],
            "mover_mask": np.zeros_like(g, dtype=bool),
            "obstacles": np.zeros_like(g, dtype=bool),
        }

    def comp_min_dist(comp: set[tuple[int, int]]) -> int:
        best_d = 10**9
        for r, c in fill_pos:
            for wr, wc in comp:
                d = abs(r - wr) + abs(c - wc)
                if d < best_d:
                    best_d = d
                    if best_d == 0:
                        return 0
        return best_d

    wall_comp = min(comps, key=lambda comp: (comp_min_dist(comp), -len(comp)))

    comp_mask = np.zeros_like(wall_mask, dtype=bool)
    for r, c in wall_comp:
        comp_mask[r, c] = True

    def comp_neighbors(r: int, c: int) -> list[tuple[int, int]]:
        out = []
        for rr, cc in neighbors8(r, c):
            if 0 <= rr < h and 0 <= cc < w and comp_mask[rr, cc]:
                out.append((rr, cc))
        return out

    endpoints = [(r, c) for r, c in wall_comp if len(comp_neighbors(r, c)) == 1]
    assert endpoints, "expected wall component endpoints"

    def outward_dir(ep: tuple[int, int]) -> tuple[tuple[int, int], bool]:
        n1 = comp_neighbors(*ep)[0]
        d1 = (_sgn(n1[0] - ep[0]), _sgn(n1[1] - ep[1]))
        d_use = d1
        is_12 = False
        n2s = [p for p in comp_neighbors(*n1) if p != ep]
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
    dr_sum = sum(d[0] for d in out_dirs)
    dc_sum = sum(d[1] for d in out_dirs)
    rad = (_sgn(dr_sum), _sgn(dc_sum))

    if rad != (0, 0):
        dr, dc = rad
        open_proj = max(r * dr + c * dc for r, c in endpoints)
        perp_dr, perp_dc = -dc, dr
        perp_vals = [r * perp_dr + c * perp_dc for r, c in endpoints]
        opening_span = max(perp_vals) - min(perp_vals) + 1
    else:
        open_proj = 0
        opening_span = 0

    remainder_mask = (g != bg) & (g != wall) & (g != fill)
    mover_mask = np.zeros_like(g, dtype=bool)
    obstacles = np.zeros_like(g, dtype=bool)
    movers: list[tuple[int, int, int]] = []
    for comp in components_mask(remainder_mask, connectivity=4):
        pts = [(int(r), int(c)) for r, c in comp]
        colors = {int(g[r, c]) for r, c in pts}
        assert len(colors) == 1, "each residual component should be monochrome"
        color = int(next(iter(colors)))
        if len(pts) == 1:
            r, c = pts[0]
            movers.append((r, c, color))
            mover_mask[r, c] = True
        else:
            for r, c in pts:
                obstacles[r, c] = True

    movers.sort(key=lambda item: (item[0], item[1], item[2]))
    return {
        "bg": bg,
        "wall": wall,
        "fill": fill,
        "fill_pos": fill_pos,
        "wall_comp": wall_comp,
        "comp_mask": comp_mask,
        "endpoints": endpoints,
        "out_dirs": out_dirs,
        "tip_12": tip_12,
        "rad": rad,
        "open_proj": int(open_proj),
        "opening_span": int(opening_span),
        "movers": movers,
        "mover_mask": mover_mask,
        "obstacles": obstacles,
    }


def _solve(
    grid: np.ndarray,
    *,
    dashed: bool,
    move_movers: bool,
    accelerating: bool = True,
) -> np.ndarray:
    g = np.array(grid, copy=True)
    h, w = g.shape
    sig = _scene_signature(g)
    bg = int(sig["bg"])
    wall = int(sig["wall"])
    fill = int(sig["fill"])
    fill_pos = list(sig["fill_pos"])
    if not fill_pos:
        return g

    comp_mask = np.array(sig["comp_mask"], copy=True)
    endpoints = list(sig["endpoints"])
    out_dirs = list(sig["out_dirs"])
    tip_12 = list(sig["tip_12"])
    rad = tuple(sig["rad"])
    mover_mask = np.array(sig["mover_mask"], copy=True)
    obstacles = np.array(sig["obstacles"], copy=True)

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

    passable = (~barrier) & ((g == bg) | (g == fill) | mover_mask)
    q = fill_pos[:]
    seen = np.zeros_like(passable, dtype=bool)
    for r, c in q:
        if passable[r, c]:
            seen[r, c] = True
    while q:
        r, c = q.pop()
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w and passable[rr, cc] and not seen[rr, cc]:
                seen[rr, cc] = True
                q.append((rr, cc))

    filled_mask = seen & (g == bg)
    out = g.copy()
    out[filled_mask] = fill

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

    if rad != (0, 0):
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

    if dashed and rad != (0, 0):
        dr, dc = rad
        rr, cc = np.indices(g.shape)
        depth = (rr * dr + cc * dc) - int(sig["open_proj"])
        new_fill_mask = (out == fill) & (g == bg)
        exterior = new_fill_mask & (depth > 0)
        period = _dash_period(int(sig["opening_span"]))
        keep = ((depth - 1) % period) == 0
        out[exterior & ~keep] = bg

    if move_movers and rad != (0, 0):
        for r, c, color in sig["movers"]:
            landings = _move_landings((r, c), rad, h, w, accelerating=accelerating)
            for rr, cc in landings[:-1]:
                out[rr, cc] = bg
            fr, fc = landings[-1]
            out[fr, fc] = int(color)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill a half-enclosed chamber, radiate it outward in dashed layers, and eject lone pixels with accelerating gaps.

    Input structure:
    - The most common color is the background, and the most frequent non-background color is a wall component with a single opening.
    - The nearest other color to that wall is the fill seed inside the chamber.
    - Any remaining singleton non-wall pixels are movers aligned with the same outward axis; larger residual components, if any, still behave as blockers.

    Rule:
    - Flood-fill the chamber and the opening-directed exterior exactly as in the base task, keeping the wall intact and applying blocker shadows along the radiate axis.
    - Thin only the exterior part of that fill into a regular dashed pattern whose period is determined by the visible opening width.
    - Then move each singleton mover straight toward the grid edge along the same axis: its previous landing cells revert to background, with jump lengths 1, 2, 3, ... so the erased trail becomes progressively sparser, and the mover finishes on the border.

    Key insight:
    The wall endpoints reveal both the outward direction and the opening width, which decode the uniform fill dashes; the extra singleton pixels follow that same direction but advertise a different rule because their background trail spaces out as the hop length increases.
    """
    return _solve(grid, dashed=True, move_movers=True, accelerating=True)


# --------------- generator (combo D3 intermittent radiation + D18 accelerating movers) ---------------

GRID_H_VERT = (11, 18)
GRID_H_HORIZ = (8, 18)
GRID_W = (15, 19)

WALL_PIXELS_MIN = 17
WALL_PIXELS_MAX = 25
DEPTH_VERT = (5, 10)
DEPTH_HORIZ = (6, 8)
GAP_VERT = (6, 7)
GAP_HORIZ = (5, 7)
TIP_12_PROB = 0.5

OPEN_END_MARGIN = 3

SEED_SIZE_WEIGHTS = [1, 1, 1, 1, 2, 2, 3, 4, 5, 6]
SEED_DEPTH_LOOKBACK = 3

MAX_MOVERS = 1
MIN_MOVER_EDGE_DIST = 4

MIN_NEW_FILL = 12
MIN_DASH_REMOVALS = 6
MIN_EXTERIOR_FILL = 8
MIN_ACCEL_DELTA = 2
MIN_BG_TRAIL = 1
PALETTE_SIZE = (4, 6)
GEN_RETRIES = 400


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
            )
            >= 2
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


def _map_local(u, v, rad, anchor):
    r0, c0 = anchor
    if rad == (-1, 0):
        return r0 + u, c0 + v
    if rad == (1, 0):
        return r0 - u, c0 + v
    if rad == (0, -1):
        return r0 + v, c0 + u
    return r0 + v, c0 - u


def _place_anchor(rad, h, w, depth, gap):
    if rad == (-1, 0):
        return random.randint(OPEN_END_MARGIN, h - depth - 3), random.randint(0, w - gap - 1)
    if rad == (1, 0):
        return random.randint(depth + 2, h - 4), random.randint(0, w - gap - 1)
    if rad == (0, -1):
        return random.randint(0, h - gap - 1), random.randint(OPEN_END_MARGIN, w - depth - 3)
    return random.randint(0, h - gap - 1), random.randint(depth + 2, w - 4)


def _local_interior(wall, depth, gap):
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
        p = frontier.pop(random.randrange(len(frontier)))
        u, v = p
        for q in random.sample([(u - 1, v), (u + 1, v), (u, v - 1), (u, v + 1)], 4):
            if q in interior_set and q not in blob:
                blob.add(q)
                frontier.append(q)
                if len(blob) >= target_size:
                    break
    return blob


def _touches_border(mask):
    return bool(mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any())


def _mover_candidate_info(grid: np.ndarray, base_out: np.ndarray) -> list[dict[str, object]]:
    sig = _scene_signature(grid)
    fill = int(sig["fill"])
    rad = tuple(sig["rad"])
    h, w = grid.shape
    if rad == (0, 0):
        return []

    candidates: list[dict[str, object]] = []
    fill_mask = (base_out == fill) & (grid == int(sig["bg"]))
    wall_points = list(sig["wall_comp"])
    for r, c in np.argwhere(fill_mask):
        start = (int(r), int(c))
        edge_dist = _dist_to_edge(start, rad, h, w)
        if edge_dist < MIN_MOVER_EDGE_DIST:
            continue
        wall_clearance = min(abs(start[0] - wr) + abs(start[1] - wc) for wr, wc in wall_points)
        if wall_clearance < 3:
            continue
        line = _line_to_edge(start, rad, h, w)
        fill_on_line = sum(1 for rr, cc in line if int(base_out[rr, cc]) == fill)
        if fill_on_line < MIN_BG_TRAIL + 1:
            continue
        accel_landings = _move_landings(start, rad, h, w, accelerating=True)
        dense_landings = _move_landings(start, rad, h, w, accelerating=False)
        accel_trail = max(0, len(accel_landings) - 1)
        delta = max(0, len(dense_landings) - len(accel_landings))
        if accel_trail < MIN_BG_TRAIL or delta < MIN_ACCEL_DELTA:
            continue
        ortho_key = start[1] if rad[0] != 0 else start[0]
        candidates.append(
            {
                "start": start,
                "edge_dist": int(edge_dist),
                "fill_on_line": int(fill_on_line),
                "delta": int(delta),
                "ortho_key": int(ortho_key),
            }
        )
    random.shuffle(candidates)
    candidates.sort(key=lambda item: (item["delta"], item["fill_on_line"], item["edge_dist"]), reverse=True)
    return candidates


def _combo_metrics(grid: np.ndarray) -> dict[str, int]:
    sig = _scene_signature(grid)
    bg = int(sig["bg"])
    fill = int(sig["fill"])
    variant_out = _solve(grid, dashed=True, move_movers=True, accelerating=True)
    dashed_nomove = _solve(grid, dashed=True, move_movers=False)
    continuous_nomove = _solve(grid, dashed=False, move_movers=False)
    dense_movers = _solve(grid, dashed=True, move_movers=True, accelerating=False)

    rr, cc = np.indices(grid.shape)
    dr, dc = sig["rad"]
    depth = (rr * dr + cc * dc) - int(sig["open_proj"])
    exterior = depth > 0

    variant_new_fill = (variant_out == fill) & (grid == bg)
    dashed_new_fill = (dashed_nomove == fill) & (grid == bg)
    continuous_new_fill = (continuous_nomove == fill) & (grid == bg)
    dense_new_fill = (dense_movers == fill) & (grid == bg)

    dash_removed = int((continuous_new_fill & ~dashed_new_fill).sum())
    exterior_visible = int((dashed_new_fill & exterior).sum())
    accel_delta = int((variant_out != dense_movers).sum())
    bg_trail = int((((dashed_nomove == fill) & (grid == bg)) & (variant_out == bg)).sum())
    mover_edge_hits = 0
    h, w = grid.shape
    for r, c, color in sig["movers"]:
        er, ec = _edge_cell((r, c), sig["rad"], h, w)
        mover_edge_hits += int(variant_out[er, ec] == color)

    return {
        "new_fill": int(variant_new_fill.sum()),
        "dash_removed": dash_removed,
        "exterior_visible": exterior_visible,
        "accel_delta": accel_delta,
        "bg_trail": bg_trail,
        "border_touch": int(_touches_border(variant_new_fill | (variant_out != grid))),
        "mover_count": len(sig["movers"]),
        "mover_edge_hits": int(mover_edge_hits),
        "opening_span": int(sig["opening_span"]),
        "period": int(_dash_period(int(sig["opening_span"]))) if sig["opening_span"] else 0,
    }


def _gen_attempt():
    rad = random.choice([(-1, 0), (1, 0), (0, -1), (0, 1)])
    h = random.randint(*GRID_H_VERT) if rad[0] else random.randint(*GRID_H_HORIZ)
    w = random.randint(*GRID_W)

    built = _build_local_wall(rad, h, w)
    if built is None:
        return None
    local_wall, depth, gap = built
    anchor = _place_anchor(rad, h, w, depth, gap)

    palette = random.sample(range(10), random.randint(*PALETTE_SIZE))
    bg, wall_color, fill_color = palette[:3]
    mover_colors = palette[3:]
    if not mover_colors:
        return None

    grid = np.full((h, w), bg, dtype=int)
    for u, v in local_wall:
        r, c = _map_local(u, v, rad, anchor)
        grid[r, c] = wall_color

    interior = _local_interior(local_wall, depth, gap)
    wall_set = set(local_wall)
    adjacent = [
        p
        for p in interior
        if any((p[0] + dr, p[1] + dc) in wall_set for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)])
    ]
    seed_pool = [p for p in adjacent if p[0] >= max(1, depth - SEED_DEPTH_LOOKBACK)]
    if not seed_pool:
        return None

    blob = _grow_seed_blob(random.choice(seed_pool), set(interior))
    for u, v in blob:
        r, c = _map_local(u, v, rad, anchor)
        if grid[r, c] != bg:
            return None
        grid[r, c] = fill_color

    try:
        base_out = _solve(grid, dashed=True, move_movers=False)
    except Exception:
        return None
    base_sig = _scene_signature(grid)
    if tuple(base_sig["rad"]) == (0, 0):
        return None

    base_new_fill = (base_out == fill_color) & (grid == bg)
    if int(base_new_fill.sum()) < MIN_NEW_FILL or not _touches_border(base_new_fill):
        return None

    candidates = _mover_candidate_info(grid, base_out)
    if not candidates:
        return None

    max_movers = min(MAX_MOVERS, len(mover_colors), len(candidates))
    if max_movers <= 0:
        return None
    target_movers = random.randint(1, max_movers)

    chosen = []
    used_ortho = set()
    for cand in candidates:
        if cand["ortho_key"] in used_ortho:
            continue
        chosen.append(cand)
        used_ortho.add(cand["ortho_key"])
        if len(chosen) >= target_movers:
            break
    if not chosen:
        return None

    for idx, cand in enumerate(chosen):
        r, c = cand["start"]
        grid[r, c] = mover_colors[idx]

    placed_sig = _scene_signature(grid)
    if len(placed_sig["movers"]) != len(chosen):
        return None

    try:
        out = main(grid)
        metrics = _combo_metrics(grid)
    except Exception:
        return None

    new_fill = (out == fill_color) & (grid == bg)
    if int(new_fill.sum()) < MIN_NEW_FILL:
        return None
    if metrics["dash_removed"] < MIN_DASH_REMOVALS:
        return None
    if metrics["exterior_visible"] < MIN_EXTERIOR_FILL:
        return None
    if metrics["bg_trail"] < MIN_BG_TRAIL:
        return None
    if metrics["accel_delta"] < MIN_ACCEL_DELTA:
        return None
    if metrics["mover_count"] == 0 or metrics["mover_edge_hits"] != metrics["mover_count"]:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid b9e38dc0 combo input")
