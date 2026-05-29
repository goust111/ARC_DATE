# Combo variant of b9e38dc0 (index 12)
# combo_spec: D1/3+D15/1
# Name: 同心收缩填充+棋盘格填充

import random

import numpy as np
from scipy.ndimage import distance_transform_cdt

from arc_utils import components_mask, get_bg, neighbors4, neighbors8


# ---------------------------------------------------------------------------
# Design analysis for the combo variant
#
# a) Concrete behavior and trace:
#    The original task computes one geometric region: a seed starts inside the
#    wall pocket, the wall plus the two endpoint rays define the corridor, and
#    obstacles remove straight "shadow" tails behind themselves along the
#    radiate direction. Originally every reachable background cell became fill.
#    In this combo, the geometry stays the same but the recoloring changes.
#    First we finish the full geometric region, including shadow trimming.
#    Then we measure each region cell's 8-neighbor boundary depth:
#      depth 0,2,4,...  -> solid fill
#      depth 1,3,5,...  -> checkerboard shell anchored to the seed parity
#    Example: if one local strip of the final region has depths
#    0,1,1,2,3,3 from outside to inside, the original output would be
#    F,F,F,F,F,F. This combo becomes F,(F/B),(F/B),F,(F/B),(F/B), where the
#    odd-depth cells alternate by global parity and the even-depth cells stay
#    solid fill. The visible result is a radiating region with concentric bands
#    whose odd bands are internally checkered.
#
# b) Complexity calibration:
#    The combo uses one stable geometric rule and one stable parity rule. It
#    does not introduce extra colors or case-by-case mappings. To keep the
#    checkerboard phase readable, seeds are singleton pixels and validation
#    rejects shallow regions. Four radiate directions and both wall arm modes
#    are retained so examples still have variety without becoming arbitrary.
#
# c) Implicit-assumption audit:
#    - The original `filled_mask = seen & (g == bg)` assumed all reached cells
#      receive the same output color. That is no longer true, so writeback is
#      split by boundary-depth parity plus checker parity.
#    - Shadow logic cannot depend on "cells that will end up fill-colored",
#      because odd-depth checker shells intentionally contain background cells.
#      Therefore shadows are applied to the geometric region mask before any
#      shell coloring.
#    - The "closest non-wall color = fill seed" heuristic still works only if
#      obstacles are kept farther from the wall than the seed. The generator
#      preserves that by using a singleton seed adjacent to the wall and placing
#      obstacles on background cells with wall distance > 1.
#    - The checkerboard phase needs one unambiguous anchor. The generator
#      therefore uses singleton seeds rather than blobs.
#
# d) Placement legality:
#    Wall construction, endpoint gap, and open-end margins remain those of the
#    original generator, so endpoint/radiate inference stays valid. Obstacles
#    are placed only on background cells, with Chebyshev separation, and never
#    on the wall or seed. The seed is chosen from interior cells adjacent to the
#    wall near the closed end, so it is guaranteed to lie in the reachable zone.
#
# e) Visual observability:
#    A solver looking only at example pairs can infer:
#    - the same semi-enclosed region is still the activated geometry,
#    - the activation now forms inward shells instead of a solid blob,
#    - every second shell contains a checkerboard rather than a flat color.
#    This is visible without using color ids because the only participating
#    colors are the seen seed color and the seen background color.
#
# f) Visual cleanliness:
#    The pattern is confined to the region that was already meaningful in the
#    original task. Walls and obstacles keep their input colors, while the new
#    shell/checker structure only affects previously fillable background cells.
#    Generator thresholds require both solid shells and checker shells to be
#    large enough to read rather than appearing as noise.
#
# g) Color inference:
#    No hidden color remapping is introduced. The combo only uses:
#      - existing fill color
#      - existing background color
#    The distinction comes from structure: even-depth shells are solid fill,
#    odd-depth shells expose a parity checkerboard.
#
# 9) New assumptions introduced by the modified solver and generator support:
#    - `_boundary_distance()` assumes a nontrivial region with visible depth.
#      Generator enforces a minimum max depth and minimum odd/even shell counts.
#    - The odd-depth checkerboard must contain both fill and bg cells; otherwise
#      the combo visually collapses toward the D1-only striped version.
#      Generator validates minimum counts for both checker outcomes.
#    - The radiating effect should still be externally visible, so generator
#      requires the visible fill output to reach a border, not just the hidden
#      geometric region.
# ---------------------------------------------------------------------------


def _sgn(x: int) -> int:
    return (x > 0) - (x < 0)


def _touches_border(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any())


def _flood_mask(passable: np.ndarray, starts: list[tuple[int, int]]) -> np.ndarray:
    """4-neighbor flood mask from all start cells over a boolean passable mask."""
    h, w = passable.shape
    q = starts[:]
    seen = np.zeros_like(passable, dtype=bool)
    for r, c in q:
        if 0 <= r < h and 0 <= c < w and passable[r, c]:
            seen[r, c] = True
    while q:
        r, c = q.pop()
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w and passable[rr, cc] and not seen[rr, cc]:
                seen[rr, cc] = True
                q.append((rr, cc))
    return seen


def _boundary_distance(mask: np.ndarray) -> np.ndarray:
    """8-neighbor distance from each True cell to the boundary of its region."""
    h, w = mask.shape
    dist = np.full((h, w), -1, dtype=int)
    q: list[tuple[int, int]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            boundary = False
            for rr, cc in neighbors8(r, c):
                if rr < 0 or rr >= h or cc < 0 or cc >= w or not mask[rr, cc]:
                    boundary = True
                    break
            if boundary:
                dist[r, c] = 0
                q.append((r, c))

    qi = 0
    while qi < len(q):
        r, c = q[qi]
        qi += 1
        d = dist[r, c]
        for rr, cc in neighbors8(r, c):
            if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and dist[rr, cc] == -1:
                dist[rr, cc] = d + 1
                q.append((rr, cc))
    return dist


def _compute_state(grid: np.ndarray) -> dict:
    """Shared geometric analysis for solver and generator validation."""
    g = np.array(grid, copy=True)
    h, w = g.shape

    vals = np.unique(g)
    bg = get_bg(g)
    non_bg = [int(v) for v in vals if int(v) != bg]
    assert non_bg, "expected at least one non-background color"

    wall = max(non_bg, key=lambda c: int((g == c).sum()))
    wall_mask = g == wall
    comps = [set((int(r), int(c)) for r, c in pts) for pts in components_mask(wall_mask, connectivity=8)]
    assert comps, "expected at least one wall connected component"

    other_colors = [c for c in non_bg if c != wall]
    assert other_colors, "expected at least one non-wall color (fill color)"

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
        empty = np.zeros_like(g, dtype=bool)
        return {
            "grid": g,
            "bg": bg,
            "wall": wall,
            "fill": fill,
            "fill_pos": [],
            "fill_seed_mask": empty,
            "region_mask": empty,
            "region_bg": empty,
            "dist": np.full_like(g, -1, dtype=int),
            "even_fill_mask": empty,
            "odd_shell_mask": empty,
            "odd_checker_fill_mask": empty,
            "odd_checker_bg_mask": empty,
            "rad": (0, 0),
            "tip_12_mode": False,
            "obstacles": empty,
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
    assert endpoints, "expected wall component to have endpoints"

    def outward_dir(ep: tuple[int, int]) -> tuple[tuple[int, int], bool]:
        n1 = comp_neighbors(*ep)[0]
        d1 = (_sgn(n1[0] - ep[0]), _sgn(n1[1] - ep[1]))
        d_use = d1
        is_12 = False
        n2s = [p for p in comp_neighbors(*n1) if p != ep]
        if n2s:
            n2 = n2s[0]
            d2 = (_sgn(n2[0] - n1[0]), _sgn(n2[1] - n1[1]))
            if (d1[0] == 0) ^ (d1[1] == 0) and d2[0] != 0 and d2[1] != 0:
                d_use = d2
                is_12 = True
        return (-d_use[0], -d_use[1]), is_12

    out_info = [outward_dir(ep) for ep in endpoints]
    out_dirs = [d for d, _ in out_info]
    tip_12_flags = [flag for _, flag in out_info]
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

    for ep, d, is_12 in zip(endpoints, out_dirs, tip_12_flags):
        if is_12:
            add_ray_12(ep, d)
        else:
            add_ray(ep, d)

    passable = (~barrier) & ((g == bg) | (g == fill))
    seen = _flood_mask(passable, fill_pos)

    obstacles = (g != bg) & (g != fill) & (g != wall)
    region_mask = seen.copy()

    def shadow_process(coords: list[tuple[int, int]]) -> None:
        started = False
        blocked = False
        for r, c in coords:
            if seen[r, c]:
                started = True
            if started and obstacles[r, c]:
                blocked = True
                continue
            if blocked and seen[r, c]:
                region_mask[r, c] = False

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

    dist = _boundary_distance(region_mask)
    region_bg = region_mask & (g == bg)
    odd_shell_mask = region_bg & ((dist % 2) == 1)
    even_fill_mask = region_bg & ((dist % 2) == 0)

    anchor = min(fill_pos)
    rows, cols = np.indices(g.shape)
    checker_phase = ((rows + cols) & 1) == ((anchor[0] + anchor[1]) & 1)
    odd_checker_fill_mask = odd_shell_mask & checker_phase
    odd_checker_bg_mask = odd_shell_mask & ~checker_phase

    return {
        "grid": g,
        "bg": bg,
        "wall": wall,
        "fill": fill,
        "fill_pos": fill_pos,
        "fill_seed_mask": g == fill,
        "region_mask": region_mask,
        "region_bg": region_bg,
        "dist": dist,
        "even_fill_mask": even_fill_mask,
        "odd_shell_mask": odd_shell_mask,
        "odd_checker_fill_mask": odd_checker_fill_mask,
        "odd_checker_bg_mask": odd_checker_bg_mask,
        "rad": rad,
        "tip_12_mode": any(tip_12_flags),
        "obstacles": obstacles,
    }


def _combo_metrics(grid: np.ndarray) -> dict:
    """Quantify whether generated examples exhibit the fused shell+checker behavior."""
    state = _compute_state(grid)
    out = main(grid)
    visible_fill = (out == state["fill"]) & (grid == state["bg"])
    odd_fill = state["odd_checker_fill_mask"] & (out == state["fill"])
    odd_bg = state["odd_checker_bg_mask"] & (out == state["bg"])
    even_fill = state["even_fill_mask"] & (out == state["fill"])
    max_depth = int(state["dist"][state["region_mask"]].max()) if state["region_mask"].any() else -1
    odd_total = int(state["odd_shell_mask"].sum())
    odd_ratio = float(odd_fill.sum()) / odd_total if odd_total else 0.0
    return {
        "region_bg_cells": int(state["region_bg"].sum()),
        "visible_fill_cells": int(visible_fill.sum()),
        "visible_fill_touches_border": _touches_border(visible_fill),
        "odd_shell_cells": odd_total,
        "odd_checker_fill_cells": int(odd_fill.sum()),
        "odd_checker_bg_cells": int(odd_bg.sum()),
        "odd_fill_ratio": odd_ratio,
        "even_fill_cells": int(even_fill.sum()),
        "max_depth": max_depth,
        "touches_border": _touches_border(state["region_mask"]),
        "tip_12_mode": state["tip_12_mode"],
        "obstacle_count": int(state["obstacles"].sum()),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the wall-bounded reachable region with concentric shells whose odd layers become checkerboard bands.

    Input structure:
    - The most common color is the background, and the most frequent non-background color is a partial wall with one open side.
    - A singleton non-wall seed sits inside the pocket near the closed end; any other non-background pixels act as fixed obstacles.

    Rule:
    - Compute the same reachable region as the original task: the wall and endpoint rays define the corridor, and obstacles cut away straight shadows along the radiate direction.
    - Recolor that final region by boundary depth. Background cells on even-depth shells become solid seed color; background cells on odd-depth shells use a seed-anchored checkerboard of seed color and background color.

    Key insight: Geometry is still solved exactly like the original puzzle, but coloring now depends on two overlaid structures at once: concentric boundary-depth shells and a global checker parity for every odd shell.
    """
    state = _compute_state(grid)
    g = state["grid"]
    if not state["fill_pos"]:
        return g

    out = g.copy()
    out[state["region_bg"]] = state["bg"]
    out[state["even_fill_mask"]] = state["fill"]
    out[state["odd_checker_fill_mask"]] = state["fill"]
    out[state["fill_seed_mask"]] = state["fill"]
    return out


# --------------- generator ---------------

# Canvas size bounds
GRID_H_VERT = (11, 18)
GRID_H_HORIZ = (8, 18)
GRID_W = (15, 19)

# Wall geometry
WALL_PIXELS_MIN = 17
WALL_PIXELS_MAX = 25
DEPTH_VERT = (5, 10)
DEPTH_HORIZ = (6, 8)
GAP_VERT = (6, 7)
GAP_HORIZ = (5, 7)
TIP_12_PROB = 0.5

# Placement margins
OPEN_END_MARGIN = 3

# Fill seed
SEED_DEPTH_LOOKBACK = 3

# Obstacles
MAX_OBSTACLES = 5
OBSTACLE_IN_FILL_PROB = 0.65
OBSTACLE_MIN_SEP = 2

# Validation
MIN_REGION_BG_CELLS = 24
MIN_VISIBLE_FILL = 12
MIN_EVEN_FILL = 10
MIN_ODD_SHELL_CELLS = 8
MIN_ODD_FILL = 4
MIN_ODD_BG = 4
MIN_STRIPE_DEPTH = 2
MIN_ODD_FILL_RATIO = 0.25
MAX_ODD_FILL_RATIO = 0.75
PALETTE_SIZE = (3, 5)
GEN_RETRIES = 400


def _degree1_pixels(points):
    """Return degree-1 pixels (exactly one 8-neighbor in the set)."""
    pts = set(points)
    return [(r, c) for r, c in pts if sum((rr, cc) in pts for rr, cc in neighbors8(r, c)) == 1]


def _make_arm(depth, start_v, target_v, extra_steps, tip_12=False):
    """Build one zigzag arm from (0, start_v) toward (depth, target_v)."""
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
    """Construct wall shape in local (u, v) coordinates. Returns (pts, depth, gap) or None."""
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


def _map_local(u, v, rad, anchor):
    """Map local (u, v) to grid (r, c) given radiate direction and anchor."""
    r0, c0 = anchor
    if rad == (-1, 0):
        return r0 + u, c0 + v
    if rad == (1, 0):
        return r0 - u, c0 + v
    if rad == (0, -1):
        return r0 + v, c0 + u
    return r0 + v, c0 - u


def _place_anchor(rad, h, w, depth, gap):
    """Choose anchor so the wall fits within margins."""
    if rad == (-1, 0):
        return random.randint(OPEN_END_MARGIN, h - depth - 3), random.randint(0, w - gap - 1)
    if rad == (1, 0):
        return random.randint(depth + 2, h - 4), random.randint(0, w - gap - 1)
    if rad == (0, -1):
        return random.randint(0, h - gap - 1), random.randint(OPEN_END_MARGIN, w - depth - 3)
    return random.randint(0, h - gap - 1), random.randint(depth + 2, w - 4)


def _local_interior(wall, depth, gap):
    """Return background cells strictly between the two wall arms in local coordinates."""
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


def _obstacle_fits(pos, placed):
    """Check that pos is at least Chebyshev OBSTACLE_MIN_SEP from all placed obstacles."""
    r, c = pos
    return all(max(abs(r - rr), abs(c - cc)) >= OBSTACLE_MIN_SEP for rr, cc in placed)


def _find_obstacle_candidates(grid, bg, wall_color, nominal_region):
    """Find obstacle candidates, preferring background cells inside the nominal active region."""
    wall_mask = grid == wall_color
    dist = distance_transform_cdt(~wall_mask, metric="taxicab")
    far_from_wall = dist > 1
    eligible = (grid == bg) & far_from_wall
    all_coords = list(zip(*np.where(eligible)))
    region_coords = [p for p in all_coords if nominal_region[p[0], p[1]]]
    random.shuffle(all_coords)
    random.shuffle(region_coords)
    return [(int(r), int(c)) for r, c in all_coords], [(int(r), int(c)) for r, c in region_coords]


def _place_obstacles(n_obstacles, candidates, region_candidates, obstacle_colors, grid):
    """Place up to n_obstacles on the grid, preferring some inside the active region."""
    placed = []
    n_region = random.randint(0, min(n_obstacles, len(region_candidates)))
    if region_candidates and random.random() < OBSTACLE_IN_FILL_PROB:
        n_region = max(1, n_region)
    for p in region_candidates:
        if len(placed) >= n_region:
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


def _gen_attempt():
    """Single attempt to generate a valid input grid for the fused shell-checker variant."""
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
    obstacle_colors = palette[3:]

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

    seed = random.choice(seed_pool)
    r, c = _map_local(seed[0], seed[1], rad, anchor)
    if grid[r, c] != bg:
        return None
    grid[r, c] = fill_color

    try:
        nominal_region = _compute_state(grid)["region_bg"]
    except Exception:
        return None

    n_obstacles = random.randint(0, MAX_OBSTACLES)
    if n_obstacles and not obstacle_colors:
        available = [c for c in range(10) if c not in (bg, wall_color, fill_color)]
        obstacle_colors = random.sample(available, random.randint(1, 2))
    if n_obstacles:
        candidates, region_candidates = _find_obstacle_candidates(grid, bg, wall_color, nominal_region)
        _place_obstacles(n_obstacles, candidates, region_candidates, obstacle_colors, grid)

    try:
        out = main(grid)
        metrics = _combo_metrics(grid)
    except Exception:
        return None

    if np.array_equal(grid, out):
        return None
    if not metrics["touches_border"]:
        return None
    if not metrics["visible_fill_touches_border"]:
        return None
    if metrics["region_bg_cells"] < MIN_REGION_BG_CELLS:
        return None
    if metrics["visible_fill_cells"] < MIN_VISIBLE_FILL:
        return None
    if metrics["even_fill_cells"] < MIN_EVEN_FILL:
        return None
    if metrics["odd_shell_cells"] < MIN_ODD_SHELL_CELLS:
        return None
    if metrics["odd_checker_fill_cells"] < MIN_ODD_FILL:
        return None
    if metrics["odd_checker_bg_cells"] < MIN_ODD_BG:
        return None
    if metrics["max_depth"] < MIN_STRIPE_DEPTH:
        return None
    if not (MIN_ODD_FILL_RATIO <= metrics["odd_fill_ratio"] <= MAX_ODD_FILL_RATIO):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid b9e38dc0 combo input")
