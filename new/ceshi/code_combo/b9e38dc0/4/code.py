# Combo variant of b9e38dc0 (index 4)
# combo_spec: D3/0+D14/0
# Name: 周期性颜色+开口宽度射线

"""
Fusion design analysis
======================

a) Concrete behavior and trace:
   Start from the original enclosure task, but replace the outside wedge with a
   straight strip whose width equals the visible mouth width. Example: if a wall
   opens upward and the 4 mouth cells are at columns 5..8, the original solver
   would spread between two boundary rays and form a widening cone. In this
   combo, the interior is still flood-filled first, then those same 4 mouth
   cells are projected straight upward to the border as a 4-cell-wide strip.
   The strip is not monochrome: depth 0 at the mouth is the seed color, depth 1
   is the wall color, depth 2 returns to the seed color, and so on. Obstacles
   still erase everything behind them along the radiation direction, so the
   alternating strip can contain background gaps after shadowing.

b) Complexity calibration:
   The combo introduces exactly two coupled freedoms that are both visible in
   the geometry: opening width controls strip thickness, and depth controls a
   fixed two-color AB stripe cycle. The color rule uses only the seed color and
   the wall color already present in the input, so there is no extra arbitrary
   palette mapping.

c) Hidden-assumption audit:
   The original code assumed the outside region could be represented by two
   endpoint rays and that every painted cell shared one color. Both assumptions
   break here. The solver now:
   - closes the mouth temporarily so the BFS only captures the interior;
   - measures the mouth cells explicitly and extrudes them as a constant-width
     band;
   - keeps a boolean paint mask through the shadow phase, then assigns colors to
     surviving radiated cells by depth parity.
   New assumptions are limited and generator-backed: the selected wall component
   has two usable endpoints, the mouth cells lie on a straight row or column,
   and the mouth corridor is empty in the input.

d) Placement legality:
   The generator reuses the original wall builder so walls stay 8-connected,
   within bounds, and keep exactly two endpoints. Anchor placement reserves
   enough outward room to show several stripe alternations. Seeds are still
   placed inside the enclosure near the closed end. Obstacles are placed only on
   background cells, with spacing guards, and may land inside the strip.

e) Visual observability:
   A human can infer the rule from a few examples because two cues line up:
   the mouth width in the input matches the thickness of the outgoing band in
   the output, and the band shows repeated wall/fill alternation away from the
   original wall body. Those effects are visible without reading color ids.

f) Visual neatness:
   The output stays clean because the interior remains a solid fill region while
   the exterior is organized as parallel full-width bands. There are no extra
   markers, and the obstacle shadows appear as simple rectangular cut-outs along
   the travel direction.

g) Change inferability:
   The only recoloring rule is structural: outside-band cells alternate between
   the two semantically meaningful colors already in the scene. The generator
   rejects cases where one of the two colors disappears or where the strip is too
   shallow, so the periodic rule is repeated enough times to be learnable.
"""

import random
import sys

import numpy as np
from scipy.ndimage import distance_transform_cdt

sys.path.append("/home/lixinyu/snap/new/solvers")
from arc_utils import components_mask, get_bg, neighbors4, neighbors8


# -- Configuration constants --------------------------------------------------
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
PATTERN_REPEAT_DEPTH = 5  # need depths 0..5 to show several AB stripes

# Fill seed
SEED_SIZE_WEIGHTS = [1, 1, 1, 1, 2, 2, 3, 4, 5, 6]
SEED_DEPTH_LOOKBACK = 3

# Obstacles
MAX_OBSTACLES = 5
OBSTACLE_IN_FILL_PROB = 0.65
OBSTACLE_MIN_SEP = 2

# Validation
MIN_NEW_FILL = 20
MIN_RADIATE_PIXELS = 12
MIN_WALL_STRIPE_PIXELS = 6
PALETTE_SIZE = (3, 5)
GEN_RETRIES = 400


def _sgn(x: int) -> int:
    return (x > 0) - (x < 0)


def _projection(r: int, c: int, rad: tuple[int, int]) -> int:
    return int(r) * int(rad[0]) + int(c) * int(rad[1])


def _degree1_pixels(points):
    pts = set(points)
    return [
        (r, c)
        for r, c in pts
        if sum((rr, cc) in pts for rr, cc in neighbors8(r, c)) == 1
    ]


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


def _place_anchor(rad, h, w, depth, gap):
    if rad == (-1, 0):
        row_lo = max(OPEN_END_MARGIN, PATTERN_REPEAT_DEPTH)
        row_hi = h - depth - 3
        if row_lo > row_hi:
            return None
        return random.randint(row_lo, row_hi), random.randint(0, w - gap - 1)
    if rad == (1, 0):
        row_lo = depth + 2
        row_hi = h - 1 - PATTERN_REPEAT_DEPTH
        if row_lo > row_hi:
            return None
        return random.randint(row_lo, row_hi), random.randint(0, w - gap - 1)
    if rad == (0, -1):
        col_lo = max(OPEN_END_MARGIN, PATTERN_REPEAT_DEPTH)
        col_hi = w - depth - 3
        if col_lo > col_hi:
            return None
        return random.randint(0, h - gap - 1), random.randint(col_lo, col_hi)
    col_lo = depth + 2
    col_hi = w - 1 - PATTERN_REPEAT_DEPTH
    if col_lo > col_hi:
        return None
    return random.randint(0, h - gap - 1), random.randint(col_lo, col_hi)


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
        u, v = frontier.pop(random.randrange(len(frontier)))
        for q in random.sample([(u - 1, v), (u + 1, v), (u, v - 1), (u, v + 1)], 4):
            if q in interior_set and q not in blob:
                blob.add(q)
                frontier.append(q)
                if len(blob) >= target_size:
                    break
    return blob


def _obstacle_fits(pos, placed):
    r, c = pos
    return all(max(abs(r - rr), abs(c - cc)) >= OBSTACLE_MIN_SEP for rr, cc in placed)


def _find_obstacle_candidates(grid, bg, wall_color, nominal_fill_zone):
    wall_mask = grid == wall_color
    dist = distance_transform_cdt(~wall_mask, metric="taxicab")
    far_from_wall = dist > 1
    eligible = (grid == bg) & far_from_wall
    all_coords = list(zip(*np.where(eligible)))
    fill_coords = [p for p in all_coords if nominal_fill_zone[p[0], p[1]]]
    random.shuffle(all_coords)
    random.shuffle(fill_coords)
    return [(int(r), int(c)) for r, c in all_coords], [(int(r), int(c)) for r, c in fill_coords]


def _place_obstacles(n_obstacles, candidates, fill_candidates, obstacle_colors, grid):
    placed = []
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


def _touches_border(mask):
    return bool(mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any())


def _closest_wall_component(
    comps: list[set[tuple[int, int]]],
    fill_pos: list[tuple[int, int]],
) -> set[tuple[int, int]]:
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

    return min(comps, key=lambda comp: (comp_min_dist(comp), -len(comp)))


def _wall_geometry(
    shape: tuple[int, int],
    wall_comp: set[tuple[int, int]],
) -> tuple[np.ndarray, list[tuple[int, int]], tuple[int, int]]:
    h, w = shape
    comp_mask = np.zeros((h, w), dtype=bool)
    for r, c in wall_comp:
        comp_mask[r, c] = True

    def comp_neighbors(r: int, c: int) -> list[tuple[int, int]]:
        out = []
        for rr, cc in neighbors8(r, c):
            if 0 <= rr < h and 0 <= cc < w and comp_mask[rr, cc]:
                out.append((rr, cc))
        return out

    endpoints = [(r, c) for r, c in wall_comp if len(comp_neighbors(r, c)) == 1]
    assert len(endpoints) >= 2, "expected at least two wall endpoints"

    def outward_dir(ep: tuple[int, int]) -> tuple[int, int]:
        n1 = comp_neighbors(*ep)[0]
        d1 = (_sgn(n1[0] - ep[0]), _sgn(n1[1] - ep[1]))
        d_use = d1
        n2s = [p for p in comp_neighbors(*n1) if p != ep]
        if n2s:
            n2 = n2s[0]
            d2 = (_sgn(n2[0] - n1[0]), _sgn(n2[1] - n1[1]))
            if (d1[0] == 0) ^ (d1[1] == 0) and d2[0] != 0 and d2[1] != 0:
                d_use = d2
        return (-d_use[0], -d_use[1])

    out_dirs = [outward_dir(ep) for ep in endpoints]
    rad = (_sgn(sum(d[0] for d in out_dirs)), _sgn(sum(d[1] for d in out_dirs)))
    assert rad != (0, 0), "expected a cardinal radiation direction"
    assert (rad[0] == 0) ^ (rad[1] == 0), "expected cardinal radiation"
    return comp_mask, endpoints, rad


def _opening_cells(
    comp_mask: np.ndarray,
    endpoints: list[tuple[int, int]],
    rad: tuple[int, int],
) -> list[tuple[int, int]]:
    h, w = comp_mask.shape
    if rad[0] != 0:
        ordered = sorted(endpoints, key=lambda p: (p[1], p[0]))
        left, right = ordered[0], ordered[-1]
        row = left[0] if abs(left[0] - right[0]) <= 1 else round((left[0] + right[0]) / 2)
        assert 0 <= row < h
        gate = [(row, c) for c in range(left[1] + 1, right[1]) if not comp_mask[row, c]]
    else:
        ordered = sorted(endpoints, key=lambda p: (p[0], p[1]))
        top, bottom = ordered[0], ordered[-1]
        col = top[1] if abs(top[1] - bottom[1]) <= 1 else round((top[1] + bottom[1]) / 2)
        assert 0 <= col < w
        gate = [(r, col) for r in range(top[0] + 1, bottom[0]) if not comp_mask[r, col]]
    assert gate, "expected a non-empty mouth segment"
    return gate


def _apply_shadows(
    filled_mask: np.ndarray,
    obstacles: np.ndarray,
    rad: tuple[int, int],
) -> None:
    h, w = filled_mask.shape
    dr, dc = rad

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
                filled_mask[r, c] = False

    if dr != 0 and dc == 0:
        for c in range(w):
            rows = range(h) if dr > 0 else range(h - 1, -1, -1)
            shadow_process([(r, c) for r in rows])
    elif dc != 0 and dr == 0:
        for r in range(h):
            cols = range(w) if dc > 0 else range(w - 1, -1, -1)
            shadow_process([(r, c) for c in cols])
    else:
        raise AssertionError("expected cardinal radiation")


def _solve_core(grid: np.ndarray):
    g = np.array(grid, copy=True)
    h, w = g.shape

    bg = get_bg(g)
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
        meta = {
            "bg": int(bg),
            "wall": int(wall),
            "fill": int(fill),
            "rad": (0, 0),
            "opening_width": 0,
            "paint_mask": np.zeros_like(g, dtype=bool),
            "radiated_mask": np.zeros_like(g, dtype=bool),
            "wall_stripe_mask": np.zeros_like(g, dtype=bool),
            "depth_map": np.full_like(g, -1, dtype=int),
        }
        return g, meta

    wall_comp = _closest_wall_component(comps, fill_pos)
    comp_mask, endpoints, rad = _wall_geometry(g.shape, wall_comp)
    gate_cells = _opening_cells(comp_mask, endpoints, rad)

    gate_mask = comp_mask.copy()
    for r, c in gate_cells:
        gate_mask[r, c] = True

    passable = (~gate_mask) & ((g == bg) | (g == fill))
    seen = np.zeros_like(passable, dtype=bool)
    q = fill_pos[:]
    for r, c in q:
        if passable[r, c]:
            seen[r, c] = True
    while q:
        r, c = q.pop()
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w and passable[rr, cc] and not seen[rr, cc]:
                seen[rr, cc] = True
                q.append((rr, cc))

    band_mask = np.zeros((h, w), dtype=bool)
    dr, dc = rad
    for r, c in gate_cells:
        rr, cc = r, c
        while 0 <= rr < h and 0 <= cc < w:
            if g[rr, cc] == bg:
                band_mask[rr, cc] = True
            rr += dr
            cc += dc

    filled_mask = (seen | band_mask) & (g == bg)
    obstacles = (g != bg) & (g != fill) & (g != wall)
    _apply_shadows(filled_mask, obstacles, rad)

    out = g.copy()
    interior_mask = filled_mask & (~band_mask)
    radiated_mask = filled_mask & band_mask
    out[interior_mask] = fill

    depth_map = np.full((h, w), -1, dtype=int)
    wall_stripe_mask = np.zeros((h, w), dtype=bool)
    if gate_cells:
        open_proj = _projection(gate_cells[0][0], gate_cells[0][1], rad)
        rows, cols = np.indices((h, w))
        depth_map = rows * rad[0] + cols * rad[1] - open_proj
        wall_stripe_mask = radiated_mask & ((depth_map % 2) == 1)
        fill_stripe_mask = radiated_mask & (~wall_stripe_mask)
        out[fill_stripe_mask] = fill
        out[wall_stripe_mask] = wall

    meta = {
        "bg": int(bg),
        "wall": int(wall),
        "fill": int(fill),
        "rad": (int(rad[0]), int(rad[1])),
        "opening_width": int(len(gate_cells)),
        "paint_mask": filled_mask.copy(),
        "radiated_mask": radiated_mask,
        "wall_stripe_mask": wall_stripe_mask,
        "depth_map": depth_map,
    }
    return out, meta


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill a wall-bounded chamber, then emit a mouth-width striped beam through its opening.

    Input structure:
    - The background is the most common color. The dominant non-background color is a single wall
      component forming a partial enclosure with one visible mouth between its two endpoints.
    - A nearby non-wall color marks the seed blob inside the enclosure. Any other non-background
      pixels are preserved obstacles.

    Rule:
    - Temporarily treat the mouth as closed, flood-fill the interior chamber from the seed, then
      reopen the mouth and extend every mouth cell straight outward so the radiation width equals
      the mouth width.
    - Keep the interior solid in the seed color, but color the outward strip by depth bands that
      alternate between seed color and wall color. Obstacles still cast background shadows behind
      themselves along the same outward direction.

    Key insight:
    - The endpoints reveal both where the mouth is and which way the beam travels; once that mouth
      is measured, the output is determined by two visible structures at once: strip thickness comes
      from opening width, and strip color comes from periodic depth-based alternation.
    """
    return _solve_core(grid)[0]


def _gen_attempt():
    rad = random.choice([(-1, 0), (1, 0), (0, -1), (0, 1)])
    h = random.randint(*GRID_H_VERT) if rad[0] else random.randint(*GRID_H_HORIZ)
    w = random.randint(*GRID_W)

    built = _build_local_wall(rad, h, w)
    if built is None:
        return None
    local_wall, depth, gap = built
    anchor = _place_anchor(rad, h, w, depth, gap)
    if anchor is None:
        return None

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

    blob = _grow_seed_blob(random.choice(seed_pool), set(interior))
    for u, v in blob:
        r, c = _map_local(u, v, rad, anchor)
        if grid[r, c] != bg:
            return None
        grid[r, c] = fill_color

    try:
        _, nominal_meta = _solve_core(grid)
    except Exception:
        return None
    nominal_fill_zone = nominal_meta["paint_mask"] & (grid == bg)

    n_obstacles = random.randint(0, MAX_OBSTACLES)
    if n_obstacles and not obstacle_colors:
        available = [c for c in range(10) if c not in (bg, wall_color, fill_color)]
        obstacle_colors = random.sample(available, random.randint(1, 2))
    if n_obstacles:
        candidates, fill_candidates = _find_obstacle_candidates(grid, bg, wall_color, nominal_fill_zone)
        _place_obstacles(n_obstacles, candidates, fill_candidates, obstacle_colors, grid)

    try:
        out, meta = _solve_core(grid)
    except Exception:
        return None

    paint_mask = meta["paint_mask"] & (grid == bg)
    radiated_mask = meta["radiated_mask"] & paint_mask
    wall_stripe_mask = meta["wall_stripe_mask"] & paint_mask
    depth_map = meta["depth_map"]

    if int(paint_mask.sum()) < MIN_NEW_FILL or not _touches_border(paint_mask):
        return None
    if meta["opening_width"] < 3:
        return None
    if int(radiated_mask.sum()) < MIN_RADIATE_PIXELS:
        return None
    if int(wall_stripe_mask.sum()) < MIN_WALL_STRIPE_PIXELS:
        return None
    if not np.any(radiated_mask):
        return None
    if int(depth_map[radiated_mask].max()) < PATTERN_REPEAT_DEPTH:
        return None

    new_wall = (grid == bg) & (out == wall_color)
    new_fill = (grid == bg) & (out == fill_color)
    if not new_wall.any() or not new_fill.any():
        return None

    border_counts = [
        int(radiated_mask[0, :].sum()),
        int(radiated_mask[-1, :].sum()),
        int(radiated_mask[:, 0].sum()),
        int(radiated_mask[:, -1].sum()),
    ]
    if max(border_counts) != meta["opening_width"]:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid b9e38dc0 combo input")
