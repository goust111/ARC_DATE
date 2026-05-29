# Combo variant of 88e364bc (index 37)
# combo_spec: D3/0+D4/3
# Name: 重力流动+滑到中心

"""
Design analysis for the combo variant.

combo_variant:
  a_concretization_and_trace:
    - Preserved scene grammar: a 20x20 black background, 1-2 border legends, several colored
      wall loops, and yellow movers inside enclosed passable cavities.
    - Original behavior: a yellow in a region looked up the matching legend direction and then
      slid on one straight ray until it hit the far wall or a previously stopped yellow.
    - Fused behavior here: every region gets a center-seeking potential field. The yellow does
      not move on one ray anymore. Instead it tries to descend toward the region's geometric
      center; when the next inward step is blocked, it drifts along the current equal-distance
      contour to the nearest spill point that can continue inward; if the whole inner contour is
      already occupied, it pools on the nearest contour cell supported by that central pile.
    - Small trace:
      1. Suppose a region center is near (9.5, 9.5), one yellow starts at (5, 6), and another
         starts at (13, 12).
      2. Under the original rule with a right-pointing legend, both would try to move only east.
      3. Under this fused rule, both follow decreasing center-distance levels. Their paths can
         bend, because "down" now means "closer to the center field", not "farther along one axis".
      4. The first yellow claims the center-nearest free cell; the second then keeps flowing
         inward until the center is blocked, searches laterally on the same contour, and stops on
         the next ring around the center.
    - The visible difference is therefore twofold at once: the path is fluid/turning rather than
      rectilinear, and the final pile forms around the middle instead of against a wall.
  b_complexity_calibration:
    - Generator keeps 1-2 legend colors and 1-4 total regions so the wall-color-to-legend link
      remains learnable from a few examples.
    - Generated legends are cardinal only. That removes a distracting second degree of freedom and
      makes "contour flow toward center" visually legible.
    - Regions get 2-3 yellows and generator prefers distant starts from different center sectors,
      so the centripetal clustering effect is obvious instead of degenerating to a tiny local nudge.
  c_implicit_assumption_review_of_original_main:
    - `_legend_directions()` still holds: it only decodes a border rectangle into a direction key.
    - Original `_slide_points()` relied on one fixed ray and a single wall-collision stop rule.
      That assumption is broken by both combo parts and must be replaced.
    - Region extraction via passable connected components still holds because interiors remain bg/yellow.
    - Border color lookup via adjacent wall counts still holds because wall topology is unchanged.
    - Clearing movers before repainting still holds because yellow is still the only moving color.
    - New assumption introduced by the combo solver: the region is a connected passable cavity, so a
      BFS distance field from center-nearest cells reaches every passable cell. Generator enforces this.
  d_placement_legality:
    - Legends are placed first with a one-cell buffer, exactly as in the source task family.
    - Region interiors are grown only on free interior cells; their wall rings may not hit the grid border.
    - Generator paints yellows only on region-interior cells and validates enclosed background topology.
  e_visual_observability:
    - A human can infer the rule because the output yellows cluster near the middle of each cavity,
      and at least one particle per accepted sample changes its perpendicular coordinate relative to
      the legend axis, exposing non-ray motion.
    - Generator rejects grids where the combo output matches either the original straight-to-wall
      baseline or the single-variant "straight ray but stop near center" baseline.
  f_visual_cleanliness:
    - No extra markers are added. Only yellow positions change.
    - Region count and sizes stay close to the original task, avoiding clutter.
  g_color_inference:
    - Color semantics stay unchanged: legend fill color identifies the wall family; blue(1) is the
      legend center marker, red(2) is the indicator, yellow(4) is the moving particle.
  h_generator_checks_for_new_solver_assumptions:
    - Solver tie-breaks use the legend direction only on equal-quality inward moves. Generator therefore
      uses irregular cavities and multi-source placements so ties occur often enough to matter.
    - Solver assumes there is at least one border legend and at least one non-empty enclosed region.
      Generator guarantees both.
"""

from __future__ import annotations

from collections import deque
import random
import sys

import numpy as np

for _ARC_UTILS_DIR in (
    "/home/lixinyu/snap/new/solvers",
    "/home/lixinyu/snap/new/arc2_solver_select",
):
    if _ARC_UTILS_DIR not in sys.path:
        sys.path.append(_ARC_UTILS_DIR)

from arc_utils import bbox_coords, components_mask, get_bg, label_mask, neighbors4, paste


GRID_H, GRID_W = 20, 20
MOVER = 4
EPS = 1e-9

CARDINAL_DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
CORNERS = ["TL", "TR", "BL", "BR"]
WALL_COLOR_CHOICES = [3, 5, 6, 7, 8, 9]
LEGEND_BUFFER = 1

REGIONS_PER_COLOR = (1, 3)
MAX_TOTAL_REGIONS = 4
REGION_SIZE_BOUNDS = {
    1: (18, 92),
    2: (16, 58),
    3: (14, 40),
    4: (12, 28),
}
BLOB_GROW_RETRIES = 25

YELLOWS_PER_REGION = [2, 3]
SOURCE_LEVEL_BAND = 3
SOURCE_CANDIDATE_POOL = 16
GEN_RETRIES = 500


def _legend_directions(g: np.ndarray, bg: int) -> dict[int, tuple[int, int]]:
    """
    Detect border legends.

    The legend fill color is also the wall color it controls.
    The sign of centroid(indicator) - centroid(center) gives the legend direction.
    In this combo variant that direction is a contour-flow tie-break orientation.
    """
    h, w = g.shape
    non_bg = g != bg
    best: dict[int, tuple[int, tuple[int, int]]] = {}
    for pts in components_mask(non_bg, connectivity=4):
        if not any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue
        r0, c0, r1, c1 = bbox_coords(pts)
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        if area != len(pts):
            continue

        counts: dict[int, int] = {}
        for r, c in pts:
            v = int(g[r, c])
            counts[v] = counts.get(v, 0) + 1
        if len(counts) != 3:
            continue

        fill_col = max(counts, key=counts.get)
        others = [(cnt, col) for col, cnt in counts.items() if col != fill_col]
        if len(others) != 2:
            continue
        others.sort()
        if others[0][0] == others[1][0]:
            rc = np.array([(r0 + r1) / 2.0, (c0 + c1) / 2.0])
            tie_info = []
            for col in (others[0][1], others[1][1]):
                pts_col = np.array([(r, c) for r, c in pts if int(g[r, c]) == col], dtype=float)
                tie_info.append((np.sum((pts_col.mean(axis=0) - rc) ** 2), col))
            tie_info.sort()
            center_col = tie_info[0][1]
            indicator_col = tie_info[1][1]
        else:
            indicator_col = others[0][1]
            center_col = others[1][1]

        center = [(r, c) for r, c in pts if int(g[r, c]) == center_col]
        indic = [(r, c) for r, c in pts if int(g[r, c]) == indicator_col]
        if not center or not indic:
            continue

        cc = np.mean(np.array(center, dtype=float), axis=0)
        ci = np.mean(np.array(indic, dtype=float), axis=0)
        vr, vc = (ci - cc).tolist()
        dr = 0 if abs(vr) < 1e-6 else (1 if vr > 0 else -1)
        dc = 0 if abs(vc) < 1e-6 else (1 if vc > 0 else -1)
        if dr == 0 and dc == 0:
            continue

        prev = best.get(fill_col)
        if prev is None or len(pts) > prev[0]:
            best[fill_col] = (len(pts), (dr, dc))
    return {k: v for k, (_, v) in best.items()}


def _normalize_pref_dir(dr: int, dc: int) -> tuple[int, int]:
    dr = int(dr)
    dc = int(dc)
    if dr != 0 and dc != 0:
        return (1 if dr > 0 else -1), 0
    if dr != 0:
        return (1 if dr > 0 else -1), 0
    if dc != 0:
        return 0, (1 if dc > 0 else -1)
    raise ValueError("direction cannot be zero")


def _projection(p: tuple[int, int], dr: int, dc: int) -> int:
    return int(dr * p[0] + dc * p[1])


def _perp_coord(p: tuple[int, int], dr: int, dc: int) -> int:
    return int(p[1] if dr != 0 else p[0])


def _region_center(region_set: set[tuple[int, int]]) -> tuple[float, float]:
    pts = np.array(sorted(region_set), dtype=float)
    ctr = pts.mean(axis=0)
    return float(ctr[0]), float(ctr[1])


def _center_dist_sq(pos: tuple[int, int], center: tuple[float, float]) -> float:
    rr = float(pos[0]) - center[0]
    cc = float(pos[1]) - center[1]
    return rr * rr + cc * cc


def _cell_rank(
    pos: tuple[int, int],
    center: tuple[float, float],
    pref_dir: tuple[int, int],
) -> tuple[float, int, int, int]:
    return (
        _center_dist_sq(pos, center),
        -_projection(pos, pref_dir[0], pref_dir[1]),
        int(pos[0]),
        int(pos[1]),
    )


def _build_center_field(
    region_set: set[tuple[int, int]],
) -> tuple[tuple[float, float], list[tuple[int, int]], dict[tuple[int, int], int]]:
    center = _region_center(region_set)
    best_d2 = min(_center_dist_sq(p, center) for p in region_set)
    targets = [p for p in region_set if abs(_center_dist_sq(p, center) - best_d2) <= EPS]

    dist: dict[tuple[int, int], int] = {}
    queue = deque()
    for p in sorted(targets):
        dist[p] = 0
        queue.append(p)
    while queue:
        cur = queue.popleft()
        base = dist[cur]
        for nb in neighbors4(cur[0], cur[1]):
            if nb not in region_set or nb in dist:
                continue
            dist[nb] = base + 1
            queue.append(nb)
    return center, targets, dist


def _best_lower_move(
    region_set: set[tuple[int, int]],
    dist_map: dict[tuple[int, int], int],
    pos: tuple[int, int],
    occupied: set[tuple[int, int]],
    center: tuple[float, float],
    pref_dir: tuple[int, int],
) -> tuple[int, int] | None:
    level = dist_map[pos]
    if level <= 0:
        return None
    options = []
    for nb in neighbors4(pos[0], pos[1]):
        if nb not in region_set or nb in occupied:
            continue
        if dist_map[nb] == level - 1:
            options.append(nb)
    if not options:
        return None
    return min(options, key=lambda p: _cell_rank(p, center, pref_dir))


def _plateau_target(
    region_set: set[tuple[int, int]],
    dist_map: dict[tuple[int, int], int],
    start: tuple[int, int],
    occupied: set[tuple[int, int]],
    center: tuple[float, float],
    pref_dir: tuple[int, int],
) -> tuple[tuple[int, int], bool] | None:
    level = dist_map[start]
    queue = deque([(start, 0)])
    seen = {start}
    spill: list[tuple[int, tuple[float, int, int, int], tuple[int, int]]] = []
    support: list[tuple[int, tuple[float, int, int, int], tuple[int, int]]] = []

    while queue:
        pos, step_cost = queue.popleft()
        lower_open = False
        lower_any = False
        for nb in neighbors4(pos[0], pos[1]):
            if nb not in region_set:
                continue
            nb_level = dist_map[nb]
            if nb_level == level - 1:
                lower_any = True
                if nb not in occupied:
                    lower_open = True
        if lower_open:
            spill.append((step_cost, _cell_rank(pos, center, pref_dir), pos))
        elif lower_any:
            support.append((step_cost, _cell_rank(pos, center, pref_dir), pos))

        for nb in neighbors4(pos[0], pos[1]):
            if nb in seen or nb not in region_set or nb in occupied:
                continue
            if dist_map[nb] != level:
                continue
            seen.add(nb)
            queue.append((nb, step_cost + 1))

    if spill:
        _, _, pos = min(spill)
        return pos, True
    if support:
        _, _, pos = min(support)
        return pos, False
    return None


def _centripetal_flow_single(
    region_set: set[tuple[int, int]],
    start: tuple[int, int],
    occupied: set[tuple[int, int]],
    dist_map: dict[tuple[int, int], int],
    center: tuple[float, float],
    pref_dir: tuple[int, int],
) -> tuple[int, int]:
    pos = start
    for _ in range(len(region_set) * 4 + 4):
        nxt = _best_lower_move(region_set, dist_map, pos, occupied, center, pref_dir)
        if nxt is not None:
            pos = nxt
            continue
        plateau = _plateau_target(region_set, dist_map, pos, occupied, center, pref_dir)
        if plateau is None:
            break
        pos, has_spill = plateau
        if not has_spill:
            break
    return pos


def _centripetal_settle_points(
    region_set: set[tuple[int, int]],
    points: list[tuple[int, int]],
    pref_dir: tuple[int, int],
) -> list[tuple[int, int]]:
    if not points:
        return []
    pref_dir = _normalize_pref_dir(*pref_dir)
    center, _, dist_map = _build_center_field(region_set)
    indexed = list(enumerate([tuple(map(int, p)) for p in points]))
    indexed.sort(
        key=lambda item: (
            dist_map[item[1]],
            _center_dist_sq(item[1], center),
            _projection(item[1], pref_dir[0], pref_dir[1]),
            item[1][0],
            item[1][1],
        ),
        reverse=True,
    )
    occupied: set[tuple[int, int]] = set()
    final: list[tuple[int, int] | None] = [None] * len(points)
    for idx, start in indexed:
        stop = _centripetal_flow_single(region_set, start, occupied, dist_map, center, pref_dir)
        occupied.add(stop)
        final[idx] = stop
    return [p for p in final if p is not None]


def _center_ray_single(
    region_set: set[tuple[int, int]],
    start: tuple[int, int],
    dr: int,
    dc: int,
    blocked: set[tuple[int, int]],
    center: tuple[float, float],
) -> tuple[int, int]:
    r, c = start
    cur_d = _center_dist_sq((r, c), center)
    while True:
        nr, nc = r + dr, c + dc
        if (nr, nc) not in region_set or (nr, nc) in blocked:
            break
        next_d = _center_dist_sq((nr, nc), center)
        if next_d > cur_d + EPS:
            break
        r, c = nr, nc
        cur_d = next_d
    return (r, c)


def _center_ray_settle_points(
    region_set: set[tuple[int, int]],
    points: list[tuple[int, int]],
    pref_dir: tuple[int, int],
) -> list[tuple[int, int]]:
    if not points:
        return []
    dr, dc = _normalize_pref_dir(*pref_dir)
    center = _region_center(region_set)
    indexed = list(enumerate([tuple(map(int, p)) for p in points]))
    indexed.sort(
        key=lambda item: (_projection(item[1], dr, dc), item[1][0], item[1][1]),
        reverse=True,
    )
    occupied: set[tuple[int, int]] = set()
    final: list[tuple[int, int] | None] = [None] * len(points)
    for idx, start in indexed:
        stop = _center_ray_single(region_set, start, dr, dc, occupied, center)
        occupied.add(stop)
        final[idx] = stop
    return [p for p in final if p is not None]


def _boundary_slide_settle_points(
    region_set: set[tuple[int, int]],
    points: list[tuple[int, int]],
    pref_dir: tuple[int, int],
) -> list[tuple[int, int]]:
    if not points:
        return []
    dr, dc = _normalize_pref_dir(*pref_dir)
    indexed = list(enumerate([tuple(map(int, p)) for p in points]))
    indexed.sort(
        key=lambda item: (_projection(item[1], dr, dc), item[1][0], item[1][1]),
        reverse=True,
    )
    occupied: set[tuple[int, int]] = set()
    final: list[tuple[int, int] | None] = [None] * len(points)
    for idx, start in indexed:
        r, c = start
        while True:
            nr, nc = r + dr, c + dc
            if (nr, nc) not in region_set or (nr, nc) in occupied:
                break
            r, c = nr, nc
        stop = (r, c)
        occupied.add(stop)
        final[idx] = stop
    return [p for p in final if p is not None]


def _region_infos(g: np.ndarray, bg: int, mover: int) -> list[dict]:
    h, w = g.shape
    passable = (g == bg) | (g == mover)
    infos = []
    for pts in components_mask(passable, connectivity=4):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue
        movers = [(int(r), int(c)) for r, c in pts if int(g[r, c]) == mover]
        if not movers:
            continue

        pset = {(int(r), int(c)) for r, c in pts}
        border_counts: dict[int, int] = {}
        for r, c in pts:
            rr = int(r)
            cc = int(c)
            for nr, nc in neighbors4(rr, cc):
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if (nr, nc) in pset:
                    continue
                v = int(g[nr, nc])
                if v == bg or v == mover:
                    continue
                border_counts[v] = border_counts.get(v, 0) + 1
        if not border_counts:
            continue
        infos.append(
            {
                "pset": pset,
                "movers": movers,
                "border_col": max(border_counts, key=border_counts.get),
            }
        )
    return infos


def _solve_with_rule(grid: np.ndarray, mode: str) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    dirs = {col: _normalize_pref_dir(*vec) for col, vec in _legend_directions(g, bg).items()}
    assert dirs, "legend_directions must find at least one legend; grid should always have border legends"

    out = g.copy()
    for info in _region_infos(g, bg, MOVER):
        border_col = info["border_col"]
        if border_col not in dirs:
            continue
        pref_dir = dirs[border_col]
        movers = info["movers"]
        for r, c in movers:
            out[r, c] = bg

        if mode == "combo":
            new_pts = _centripetal_settle_points(info["pset"], movers, pref_dir)
        elif mode == "center_ray":
            new_pts = _center_ray_settle_points(info["pset"], movers, pref_dir)
        elif mode == "boundary":
            new_pts = _boundary_slide_settle_points(info["pset"], movers, pref_dir)
        else:
            raise ValueError(mode)

        for r, c in new_pts:
            out[r, c] = MOVER
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Flow yellow pixels inside each enclosed region toward the region center.

    Input structure:
    - The most common color is background; colored walls enclose passable cavities made of bg and yellow.
    - Border legends are solid corner rectangles whose fill color matches a wall color.
    - Each legend still contains a central marker and a red indicator; their offset gives that wall family's direction key.

    Rule:
    - For each enclosed region, compute the geometric center of its passable cells and build a discrete inward-distance field from the center-nearest cells.
    - Every yellow repeatedly tries to move to a neighboring cell with a strictly smaller inward distance. If that inward step is blocked, it searches along the current equal-distance contour for the nearest place that can continue inward; if no such spill point exists, it settles on the nearest contour cell supported by the existing central pile.
    - When several candidate moves are equally good, prefer the one farther along the region's legend direction. The final yellow cluster therefore occupies the nearest reachable cells around the center.

    Key insight:
    - The legend still links wall color to movement policy, but the policy is no longer "slide on one ray". The ray direction only resolves ties inside a center-seeking flow field, so both the path and the resting place are organized around the region center.
    """
    return _solve_with_rule(grid, mode="combo")


def _center_ray_main(grid: np.ndarray) -> np.ndarray:
    return _solve_with_rule(grid, mode="center_ray")


def _boundary_main(grid: np.ndarray) -> np.ndarray:
    return _solve_with_rule(grid, mode="boundary")


def _legend_5x5(fill: int, dr: int, dc: int) -> np.ndarray:
    g = np.full((5, 5), fill, dtype=int)
    g[1, 1] = 1
    g[1, 2] = 1
    g[1, 3] = 2
    g[3, 1] = 1
    g[3, 2] = 1
    g[3, 3] = 2
    rot = {(0, 1): 0, (-1, 0): 1, (0, -1): 2, (1, 0): 3}
    return np.rot90(g, rot[(dr, dc)])


def _corner_origin(corner: str, h: int, w: int, ph: int, pw: int) -> tuple[int, int]:
    if corner == "TL":
        return 0, 0
    if corner == "TR":
        return 0, w - pw
    if corner == "BL":
        return h - ph, 0
    if corner == "BR":
        return h - ph, w - pw
    raise ValueError(corner)


def _grow_blob(
    canvas_mask: np.ndarray,
    seed: tuple[int, int],
    target_size: int,
    rng: random.Random,
) -> set[tuple[int, int]] | None:
    h, w = canvas_mask.shape
    sr, sc = seed
    if not canvas_mask[sr, sc]:
        return None
    blob = {(sr, sc)}
    frontier = []

    def push_neighbors(r: int, c: int) -> None:
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and canvas_mask[nr, nc] and (nr, nc) not in blob:
                frontier.append((nr, nc))

    push_neighbors(sr, sc)
    while len(blob) < target_size and frontier:
        idx = rng.randrange(len(frontier))
        r, c = frontier.pop(idx)
        if (r, c) in blob or not canvas_mask[r, c]:
            continue
        blob.add((r, c))
        push_neighbors(r, c)
    if len(blob) < target_size:
        return None
    return blob


def _blob_wall_cells(blob: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    wall = set()
    for r, c in blob:
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in blob:
                wall.add((nr, nc))
    return wall


def _fill_holes(blob: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    walls = _blob_wall_cells(blob, h, w)
    barrier = np.zeros((h, w), dtype=bool)
    for r, c in blob:
        barrier[r, c] = True
    for r, c in walls:
        barrier[r, c] = True

    reachable = np.zeros((h, w), dtype=bool)
    stack = []
    for r in range(h):
        for c in (0, w - 1):
            if not barrier[r, c]:
                stack.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not barrier[r, c]:
                stack.append((r, c))
    while stack:
        r, c = stack.pop()
        if reachable[r, c] or barrier[r, c]:
            continue
        reachable[r, c] = True
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and not reachable[nr, nc] and not barrier[nr, nc]:
                stack.append((nr, nc))

    trapped = (~reachable) & (~barrier)
    if not trapped.any():
        return blob
    filled = set(blob)
    for r, c in np.argwhere(trapped):
        filled.add((int(r), int(c)))
    return filled


def _try_grow_valid_blob(
    rng: random.Random,
    canvas: np.ndarray,
    occupied: np.ndarray,
    target_size: int,
    h: int,
    w: int,
) -> set[tuple[int, int]] | None:
    for _ in range(BLOB_GROW_RETRIES):
        free = np.argwhere(canvas)
        if len(free) == 0:
            break
        sr, sc = free[rng.randrange(len(free))]
        cand = _grow_blob(canvas, (int(sr), int(sc)), target_size, rng)
        if cand is None:
            continue
        while True:
            filled = _fill_holes(cand, h, w)
            if filled == cand:
                break
            cand = filled
        if any(not canvas[r, c] for r, c in cand):
            continue
        walls = _blob_wall_cells(cand, h, w)
        if any(occupied[wr, wc] or wr in (0, h - 1) or wc in (0, w - 1) for wr, wc in walls):
            continue
        return cand
    return None


def _place_legends(
    rng: random.Random,
    grid: np.ndarray,
) -> tuple[np.ndarray, list[tuple[int, tuple[int, int]]]]:
    h, w = grid.shape
    n_legends = rng.choice([1, 2])
    corners = rng.sample(CORNERS, n_legends)
    wall_colors = rng.sample(WALL_COLOR_CHOICES, n_legends)
    legend_occupied = np.zeros((h, w), dtype=bool)
    legends = []

    for corner, wall_color in zip(corners, wall_colors):
        dr, dc = rng.choice(CARDINAL_DIRS)
        patch = _legend_5x5(wall_color, dr, dc)
        ph, pw = patch.shape
        r0, c0 = _corner_origin(corner, h, w, ph, pw)
        paste(grid, patch, r0, c0)
        r_lo, r_hi = max(0, r0 - LEGEND_BUFFER), min(h, r0 + ph + LEGEND_BUFFER)
        c_lo, c_hi = max(0, c0 - LEGEND_BUFFER), min(w, c0 + pw + LEGEND_BUFFER)
        legend_occupied[r_lo:r_hi, c_lo:c_hi] = True
        legends.append((wall_color, (dr, dc)))

    return legend_occupied, legends


def _grow_regions(
    rng: random.Random,
    legends: list[tuple[int, tuple[int, int]]],
    legend_occupied: np.ndarray,
) -> list[dict] | None:
    h, w = legend_occupied.shape
    color_dir = {wc: d for wc, d in legends}
    all_colors = [wc for wc, _ in legends]

    region_colors = []
    for col in all_colors:
        region_colors.extend([col] * rng.randint(*REGIONS_PER_COLOR))
    if len(region_colors) > MAX_TOTAL_REGIONS:
        region_colors = rng.sample(region_colors, MAX_TOTAL_REGIONS)
    rng.shuffle(region_colors)

    size_min, size_max = REGION_SIZE_BOUNDS[len(region_colors)]
    occupied = legend_occupied.copy()
    regions = []

    for region_color in region_colors:
        target_size = rng.randint(size_min, size_max)
        canvas = np.ones((h, w), dtype=bool)
        canvas[0, :] = False
        canvas[-1, :] = False
        canvas[:, 0] = False
        canvas[:, -1] = False
        canvas &= ~occupied
        if not canvas.any():
            continue

        blob = _try_grow_valid_blob(rng, canvas, occupied, target_size, h, w)
        if blob is None:
            continue
        walls = _blob_wall_cells(blob, h, w)
        regions.append(
            {
                "color": region_color,
                "dir": color_dir[region_color],
                "interior": blob,
                "walls": walls,
            }
        )

        reserve = blob | walls
        extra = set()
        for r, c in walls:
            for nr, nc in neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in reserve:
                    extra.add((nr, nc))
        for r, c in reserve | extra:
            occupied[r, c] = True

    if not regions:
        return None
    if {reg["color"] for reg in regions} != set(all_colors):
        return None
    return regions


def _validate_topology(grid: np.ndarray, regions: list[dict]) -> bool:
    h, w = grid.shape
    labeled, n_comp = label_mask(grid == 0, connectivity=4)
    bounded = []
    for comp_id in range(1, n_comp + 1):
        coords = np.argwhere(labeled == comp_id)
        if any(r in (0, h - 1) or c in (0, w - 1) for r, c in coords):
            continue
        bounded.append({(int(r), int(c)) for r, c in coords})
    region_sets = [reg["interior"] for reg in regions]
    if len(bounded) != len(region_sets):
        return False
    return all(comp in region_sets for comp in bounded)


def _sector_label(
    pos: tuple[int, int],
    center: tuple[float, float],
) -> tuple[int, int]:
    rr = 0 if abs(float(pos[0]) - center[0]) < 0.5 else (1 if float(pos[0]) > center[0] else -1)
    cc = 0 if abs(float(pos[1]) - center[1]) < 0.5 else (1 if float(pos[1]) > center[1] else -1)
    return rr, cc


def _choose_sources_for_region(
    rng: random.Random,
    blob: set[tuple[int, int]],
    pref_dir: tuple[int, int],
) -> list[tuple[int, int]] | None:
    center, _, dist_map = _build_center_field(blob)
    cells = list(blob)
    if len(cells) < min(YELLOWS_PER_REGION):
        return None
    max_level = max(dist_map[p] for p in cells)
    candidates = [p for p in cells if dist_map[p] >= max(2, max_level - SOURCE_LEVEL_BAND)]
    if len(candidates) < min(YELLOWS_PER_REGION):
        candidates = cells
    candidates = sorted(
        candidates,
        key=lambda p: (
            dist_map[p],
            _center_dist_sq(p, center),
            abs(_perp_coord(p, pref_dir[0], pref_dir[1])),
            _projection(p, pref_dir[0], pref_dir[1]),
            p[0],
            p[1],
        ),
        reverse=True,
    )
    candidates = candidates[: min(len(candidates), SOURCE_CANDIDATE_POOL)]

    target = 3 if len(cells) >= 28 and rng.random() < 0.55 else 2
    sectors: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for p in candidates:
        sectors.setdefault(_sector_label(p, center), []).append(p)

    chosen: list[tuple[int, int]] = []
    used = set()
    sector_order = sorted(
        sectors,
        key=lambda lab: (
            len(sectors[lab]),
            max(dist_map[p] for p in sectors[lab]),
            lab[0] != 0,
            lab[1] != 0,
        ),
        reverse=True,
    )
    for lab in sector_order:
        p = sectors[lab][0]
        if p not in used:
            chosen.append(p)
            used.add(p)
        if len(chosen) >= target:
            break

    for p in candidates:
        if len(chosen) >= target:
            break
        if p in used:
            continue
        chosen.append(p)
        used.add(p)

    if len(chosen) < 2:
        return None
    return chosen


def _place_yellows(
    rng: random.Random,
    regions: list[dict],
    grid: np.ndarray,
) -> bool:
    for reg in regions:
        sources = _choose_sources_for_region(rng, reg["interior"], reg["dir"])
        if sources is None:
            return False
        reg["sources"] = sources
        reg["center"], _, reg["dist_map"] = _build_center_field(reg["interior"])
        for r, c in sources:
            grid[r, c] = MOVER
    return True


def _oracle_check(input_grid: np.ndarray, regions: list[dict]) -> bool:
    h, w = input_grid.shape
    try:
        solver_out = main(input_grid)
    except Exception:
        return False

    struct_mask = (input_grid != MOVER) & (input_grid != 0)
    if not np.array_equal(solver_out[struct_mask], input_grid[struct_mask]):
        return False
    if int((solver_out == MOVER).sum()) != int((input_grid == MOVER).sum()):
        return False

    regions_mask = np.zeros((h, w), dtype=bool)
    for reg in regions:
        for r, c in reg["interior"]:
            regions_mask[r, c] = True
    return bool(np.all(regions_mask[solver_out == MOVER]))


def _variant_metrics(input_grid: np.ndarray, regions: list[dict]) -> dict[str, int]:
    combo_out = main(input_grid)
    center_ray_out = _center_ray_main(input_grid)
    boundary_out = _boundary_main(input_grid)

    total = 0
    moved = 0
    lateral_shift = 0
    multi_sector_regions = 0
    max_final_level = 0
    center_gain = 0

    for reg in regions:
        sources = reg["sources"]
        pref_dir = reg["dir"]
        finals = _centripetal_settle_points(reg["interior"], sources, pref_dir)
        center = reg["center"]
        dist_map = reg["dist_map"]
        sectors = {_sector_label(p, center) for p in sources}
        if len(sectors) >= 2:
            multi_sector_regions += 1
        for src, dst in zip(sources, finals):
            total += 1
            if src != dst:
                moved += 1
            if _perp_coord(src, pref_dir[0], pref_dir[1]) != _perp_coord(dst, pref_dir[0], pref_dir[1]):
                lateral_shift += 1
            max_final_level = max(max_final_level, dist_map[dst])
            center_gain += int(round((_center_dist_sq(src, center) - _center_dist_sq(dst, center)) * 1000))

    return {
        "total": total,
        "moved": moved,
        "lateral_shift": lateral_shift,
        "multi_sector_regions": multi_sector_regions,
        "max_final_level": max_final_level,
        "center_gain_milli": center_gain,
        "combo_vs_center_ray_diff": int(np.sum(combo_out != center_ray_out)),
        "combo_vs_boundary_diff": int(np.sum(combo_out != boundary_out)),
    }


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    grid = np.zeros((GRID_H, GRID_W), dtype=int)
    legend_occupied, legends = _place_legends(rng, grid)
    regions = _grow_regions(rng, legends, legend_occupied)
    if regions is None:
        return None

    for reg in regions:
        for r, c in reg["walls"]:
            grid[r, c] = reg["color"]
    if not _validate_topology(grid, regions):
        return None

    input_grid = grid.copy()
    if not _place_yellows(rng, regions, input_grid):
        return None
    if not _oracle_check(input_grid, regions):
        return None

    metrics = _variant_metrics(input_grid, regions)
    if metrics["moved"] != metrics["total"]:
        return None
    if metrics["lateral_shift"] <= 0:
        return None
    if metrics["multi_sector_regions"] <= 0:
        return None
    if metrics["max_final_level"] > 2:
        return None
    if metrics["center_gain_milli"] <= 0:
        return None
    if metrics["combo_vs_center_ray_diff"] <= 0:
        return None
    if metrics["combo_vs_boundary_diff"] <= 0:
        return None
    return input_grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        try:
            g = _gen_attempt(rng)
        except Exception:
            continue
        if g is not None:
            return g
    raise RuntimeError("generate_input: exceeded retries")
