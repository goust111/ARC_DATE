# Variant of b9e38dc0
# Dimension: D13 — 开口方向的决定规则 (meta)
# Change: 开口方向由半包围边框的缺失面自然决定——边框哪一面不完整，射线就向那个方向辐射... → 开口方向由外部标记指示

import importlib.util
import random
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_cdt

from arc_utils import components_mask, get_bg, neighbors4, neighbors8


# Design analysis for the D13 variant
# a) Concrete change and behavior trace:
#    The wall is no longer visibly open. The generator takes the original funnel/U wall and adds a cap
#    across the former mouth, so the wall becomes a single closed sprite. A 2-pixel marker of a separate
#    color is then placed outside that capped side. The nearer marker pixel is the tail and the farther
#    pixel is the head, so the pair points away from the wall in one cardinal direction.
#    Example: an upward-radiating original wall had top endpoints and an open top gap. In this variant the
#    top gap is filled with wall pixels, and a 2-pixel vertical marker is placed above it. The old logic
#    would inspect wall endpoints to infer "up"; that now fails because the wall has no degree-1 opening.
#    The new logic first decodes the external marker, treats that side as a virtual opening, lets flood-fill
#    pass through the capped bar there, and then sends the two bounding rays outward from the capped side.
# b) Complexity calibration:
#    The marker has only 4 possible directions, matching the original four radiate directions, so the new
#    freedom stays small and repeated across samples. The wall, fill seed, flood-fill, endpoint-bounded rays,
#    and obstacle shadows all stay intact, which keeps the task recognizable rather than turning into a new
#    puzzle family.
# c) Hidden assumptions audit against the original main():
#    - Original endpoint detection assumed the selected wall component had degree-1 pixels. That is invalid
#      after capping, so this variant does not read radiate_dir from component endpoints at all.
#    - Original outward_dir() assumed local wall tangent determined the global opening side. That is replaced
#      by marker decoding. The 1:2-vs-1:1 arm mode is still read from wall geometry, but only after the side
#      has been selected by the marker.
#    - Original obstacles mask included every non-bg non-fill non-wall pixel. That would wrongly treat the
#      marker as an obstacle, so the new solver explicitly removes marker pixels from the shadow-casting set.
# d) Placement legality:
#    The cap is drawn exactly on the former mouth side, preserving one connected wall component. The marker
#    uses a one-cell background gap from the wall and stays within the original open-end margin budget.
#    Obstacles are only placed on background cells, so they never overlap wall, fill seed, or marker.
# e) Visual observability:
#    Humans can no longer infer the direction from a visible missing side because the mouth is sealed. The
#    only directional cue is the external 2-pixel arrow-like marker. Across samples, the same closed wall
#    family appears with markers on different sides, so D13 is directly visible from I/O pairs.
# f) Visual cleanliness:
#    The marker lives outside the wall with a background gap, avoiding overlap with the flood zone interior.
#    Obstacles remain sparse singleton pixels, and the cap is a clean bar on one side, so the grid stays easy
#    to parse visually.
# g) Color-change inferability:
#    No color remapping is introduced. The fill color is still the non-wall color nearest the wall; the marker
#    is recognized by its unique 2-pixel component shape and location, not by color id ordering.
# Post-change solver assumptions and generator guarantees:
#    - There is exactly one 2-pixel 4-connected non-wall component whose farther pixel is farther from the wall.
#      The generator enforces this by making the marker the only such component and avoiding 2-pixel fill blobs.
#    - The selected wall side is fully capped. The generator adds every cap pixel between the former endpoints.
#    - The chosen wall component is still the one nearest the fill seed. The generator keeps the seed inside it.
#    - The cap side still exposes enough space toward the grid border for the rays and marker. The original
#      open-end margin is kept, and the marker consumes exactly that exterior strip.


# -- Configuration constants --------------------------------------------------
# Canvas size bounds
GRID_H_VERT = (11, 18)
GRID_H_HORIZ = (8, 18)
GRID_W = (15, 19)

# Final wall size after capping the former mouth
WALL_PIXELS_MIN = 17
WALL_PIXELS_MAX = 25
DEPTH_VERT = (5, 8)
DEPTH_HORIZ = (5, 8)
GAP_VERT = (6, 7)
GAP_HORIZ = (5, 7)
TIP_12_PROB = 0.5

# Placement margins
OPEN_END_MARGIN = 3
MARKER_GAP = 1
MARKER_LEN = 2

# Fill seed
SEED_SIZE_WEIGHTS = [1, 1, 1, 3, 4, 5, 6]
SEED_DEPTH_LOOKBACK = 3

# Obstacles
MAX_OBSTACLES = 5
OBSTACLE_IN_FILL_PROB = 0.65
OBSTACLE_MIN_SEP = 2

# Validation
MIN_NEW_FILL = 20
PALETTE_SIZE = (4, 6)
GEN_RETRIES = 400
_ORIG_MAIN = None


def _sgn(x: int) -> int:
    return (x > 0) - (x < 0)


def _original_main(grid: np.ndarray) -> np.ndarray:
    global _ORIG_MAIN
    if _ORIG_MAIN is None:
        root = Path(__file__).resolve().parents[5]
        spec = importlib.util.spec_from_file_location("b9e38dc0_orig_d13", root / "arc2_solver_select" / "b9e38dc0.py")
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        _ORIG_MAIN = mod.main
    return _ORIG_MAIN(grid)


def _degree1_pixels(points):
    pts = set(points)
    return [
        (r, c)
        for r, c in pts
        if sum((rr, cc) in pts for rr, cc in neighbors8(r, c)) == 1
    ]


def _trim_dead_ends(points: set[tuple[int, int]]) -> set[tuple[int, int]]:
    pts = set(points)
    changed = True
    while changed:
        changed = False
        drop = {
            (r, c)
            for r, c in pts
            if sum((rr, cc) in pts for rr, cc in neighbors8(r, c)) <= 1
        }
        if drop and len(drop) < len(pts):
            pts -= drop
            changed = True
    return pts


def _seed_fill(mask: np.ndarray, starts: list[tuple[int, int]]) -> np.ndarray:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    q = []
    for r, c in starts:
        if 0 <= r < h and 0 <= c < w and mask[r, c] and not seen[r, c]:
            seen[r, c] = True
            q.append((r, c))
    while q:
        r, c = q.pop()
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                seen[rr, cc] = True
                q.append((rr, cc))
    return seen


def _pick_fill_color(g: np.ndarray, bg: int, wall: int, comps) -> int:
    non_bg = [int(v) for v in np.unique(g) if int(v) != int(bg)]
    wall_points = [p for comp in comps for p in comp]
    other_colors = []
    for color in non_bg:
        if color == wall:
            continue
        color_comps = [
            [(int(r), int(c)) for r, c in pts]
            for pts in components_mask(g == color, connectivity=4)
        ]
        if len(color_comps) == 1 and len(color_comps[0]) == 2:
            continue
        other_colors.append(color)
    assert other_colors, "expected at least one non-wall color"

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

    return min(other_colors, key=lambda c: (min_dist_to_wall(c)[0], min_dist_to_wall(c)[1]))


def _closest_wall_component(fill_pos, comps):
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


def _to_local(r: int, c: int, rad: tuple[int, int], side_value: int) -> tuple[int, int]:
    if rad == (-1, 0):
        return r - side_value, c
    if rad == (1, 0):
        return side_value - r, c
    if rad == (0, -1):
        return c - side_value, r
    return side_value - c, r


def _grid_dir_from_local(du: int, dv: int, rad: tuple[int, int]) -> tuple[int, int]:
    if rad == (-1, 0):
        return du, dv
    if rad == (1, 0):
        return -du, dv
    if rad == (0, -1):
        return dv, du
    return dv, -du


def _find_marker_direction(
    g: np.ndarray,
    bg: int,
    wall: int,
    fill: int,
    wall_comp: set[tuple[int, int]],
) -> tuple[tuple[int, int], set[tuple[int, int]]]:
    candidates = []
    for color in [int(v) for v in np.unique(g) if int(v) not in (bg, wall, fill)]:
        for pts in components_mask(g == color, connectivity=4):
            coords = [(int(r), int(c)) for r, c in pts]
            if len(coords) != 2:
                continue
            (r1, c1), (r2, c2) = coords
            if abs(r1 - r2) + abs(c1 - c2) != 1:
                continue
            d1 = min(abs(r1 - wr) + abs(c1 - wc) for wr, wc in wall_comp)
            d2 = min(abs(r2 - wr) + abs(c2 - wc) for wr, wc in wall_comp)
            if d1 == d2:
                continue
            tail, head = ((r1, c1), (r2, c2)) if d1 < d2 else ((r2, c2), (r1, c1))
            rad = (_sgn(head[0] - tail[0]), _sgn(head[1] - tail[1]))
            if abs(rad[0]) + abs(rad[1]) != 1:
                continue
            candidates.append((min(d1, d2), rad, {tail, head}))
    assert len(candidates) == 1, "expected exactly one directional marker"
    _, rad, marker_cells = candidates[0]
    return rad, marker_cells


def _wall_side_info(
    wall_comp: set[tuple[int, int]], rad: tuple[int, int]
) -> tuple[int, set[tuple[int, int]], int, int]:
    if rad == (-1, 0):
        side_value = min(r for r, _ in wall_comp)
    elif rad == (1, 0):
        side_value = max(r for r, _ in wall_comp)
    elif rad == (0, -1):
        side_value = min(c for _, c in wall_comp)
    else:
        side_value = max(c for _, c in wall_comp)

    local_pts = {_to_local(r, c, rad, side_value) for r, c in wall_comp}
    side_vs = sorted(v for u, v in local_pts if u == 0)
    assert side_vs, "expected capped side on marked face"
    v_lo, v_hi = side_vs[0], side_vs[-1]
    for v in range(v_lo, v_hi + 1):
        assert (0, v) in local_pts, "expected contiguous cap across marked side"
    return side_value, local_pts, v_lo, v_hi


def _build_breach_and_beam(
    shape: tuple[int, int], side_value: int, v_lo: int, v_hi: int, rad: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    h, w = shape
    breach = np.zeros(shape, dtype=bool)
    beam = np.zeros(shape, dtype=bool)
    span = range(v_lo + 1, v_hi)
    if rad == (-1, 0):
        for c in span:
            breach[side_value, c] = True
        beam[:side_value, v_lo + 1 : v_hi] = True
    elif rad == (1, 0):
        for c in span:
            breach[side_value, c] = True
        beam[side_value + 1 :, v_lo + 1 : v_hi] = True
    elif rad == (0, -1):
        for r in span:
            breach[r, side_value] = True
        beam[v_lo + 1 : v_hi, :side_value] = True
    else:
        for r in span:
            breach[r, side_value] = True
        beam[v_lo + 1 : v_hi, side_value + 1 :] = True
    return breach, beam


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill a capped enclosure and radiate outward through the side indicated by an external marker.

    Input structure:
    - bg = most common color. The wall is the most frequent non-bg color and forms a closed funnel/U sprite
      whose former open side has been capped with wall pixels.
    - A fill-color seed sits inside the enclosure. A separate 2-pixel marker lies outside the wall, aligned
      along one cardinal axis. Other sparse non-bg pixels may be obstacles.

    Rule:
    - Flood-fill the interior with the fill color, then continue outward through a virtual opening on the
      capped side pointed to by the marker.
    - The two boundary rays still follow the wall's arm-angle mode, and obstacle pixels cast shadows along
      the same marker-selected direction.

    Key insight: The wall geometry no longer reveals the opening because the mouth is sealed; the only
    directional cue is the external 2-pixel marker, whose nearer-to-farther ordering points to the radiate side.
    """
    g = np.array(grid, copy=True)
    h, w = g.shape

    bg = int(get_bg(g))
    non_bg = [int(v) for v in np.unique(g) if int(v) != bg]
    assert non_bg, "expected at least one non-background color"

    wall = max(non_bg, key=lambda c: int((g == c).sum()))
    wall_mask = g == wall
    comps = [set((int(r), int(c)) for r, c in pts) for pts in components_mask(wall_mask, connectivity=8)]
    assert comps, "expected at least one wall connected component"

    fill = _pick_fill_color(g, bg, wall, comps)
    fill_pos = [(int(r), int(c)) for r, c in zip(*np.where(g == fill))]
    if not fill_pos:
        return g

    wall_comp = _closest_wall_component(fill_pos, comps)
    wall_comp = _trim_dead_ends(wall_comp)
    assert wall_comp, "expected a capped core wall after trimming auxiliaries"
    comp_mask = np.zeros_like(wall_mask, dtype=bool)
    for r, c in wall_comp:
        comp_mask[r, c] = True

    rad, marker_cells = _find_marker_direction(g, bg, wall, fill, wall_comp)
    side_value, _, v_lo, v_hi = _wall_side_info(wall_comp, rad)
    interior_passable = (~comp_mask) & ((g == bg) | (g == fill))
    interior = _seed_fill(interior_passable, fill_pos)
    breach, beam = _build_breach_and_beam((h, w), side_value, v_lo, v_hi, rad)

    domain = interior | beam | breach
    passable = domain & ((g == bg) | (g == fill) | breach)
    seen = _seed_fill(passable, fill_pos)

    filled_mask = seen & ((g == bg) | breach)

    out = g.copy()
    out[filled_mask] = fill

    obstacles = (g != bg) & (g != fill) & (g != wall)
    for r, c in marker_cells:
        obstacles[r, c] = False

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

    dr, dc = rad
    if dr != 0 and dc == 0:
        for c in range(w):
            rows = range(h) if dr > 0 else range(h - 1, -1, -1)
            shadow_process([(r, c) for r in rows])
    elif dc != 0 and dr == 0:
        for r in range(h):
            cols = range(w) if dc > 0 else range(w - 1, -1, -1)
            shadow_process([(r, c) for c in cols])

    return out


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
        max_depth = min(DEPTH_HORIZ[1], w - 6)
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

    cap_extra = gap - 1
    base_wall_min = WALL_PIXELS_MIN - cap_extra
    base_wall_max = WALL_PIXELS_MAX - cap_extra
    if base_wall_max < 11:
        return None

    bridge_extra = max(0, right_bottom - left_bottom - 1)
    base_count = 2 * (depth + 1) + bridge_extra
    max_extra_l = max(0, left_bottom - 1)
    max_extra_r = max(0, gap - right_bottom - 1)
    min_extra = max(0, base_wall_min - base_count)
    max_extra = min(max_extra_l + max_extra_r, base_wall_max - base_count)
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

    while len(pts) < base_wall_min:
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
            if len(trial) <= base_wall_max and len(_degree1_pixels(trial)) == 2:
                pts = trial
                added = True
                break
        if not added:
            return None

    if not (base_wall_min <= len(pts) <= base_wall_max):
        return None
    if len(_degree1_pixels(pts)) != 2:
        return None
    return pts, depth, gap


def _cap_local_wall(wall, gap):
    pts = set(wall)
    for v in range(gap + 1):
        pts.add((0, v))
    spur_v = 1 if gap >= 2 else 0
    pts.add((-1, spur_v))
    pts.add((-2, spur_v))
    return pts


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


def _place_marker(grid, marker_color, rad, anchor, gap):
    mid_v = gap // 2
    wall_r, wall_c = _map_local(0, mid_v, rad, anchor)
    tail_r = wall_r + rad[0] * (MARKER_GAP + 1)
    tail_c = wall_c + rad[1] * (MARKER_GAP + 1)
    head_r = wall_r + rad[0] * (MARKER_GAP + MARKER_LEN)
    head_c = wall_c + rad[1] * (MARKER_GAP + MARKER_LEN)
    coords = [(tail_r, tail_c), (head_r, head_c)]
    h, w = grid.shape
    for r, c in coords:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if grid[r, c] != grid[0, 0]:
            return False
    grid[tail_r, tail_c] = marker_color
    grid[head_r, head_c] = marker_color
    return True


def _gen_attempt():
    rad = random.choice([(-1, 0), (1, 0), (0, -1), (0, 1)])
    h = random.randint(*GRID_H_VERT) if rad[0] else random.randint(*GRID_H_HORIZ)
    w = random.randint(*GRID_W)

    built = _build_local_wall(rad, h, w)
    if built is None:
        return None
    local_wall_open, depth, gap = built
    local_wall = _cap_local_wall(local_wall_open, gap)
    anchor = _place_anchor(rad, h, w, depth, gap)

    palette = random.sample(range(10), random.randint(*PALETTE_SIZE))
    bg, wall_color, fill_color, marker_color = palette[:4]
    obstacle_colors = palette[4:]

    grid = np.full((h, w), bg, dtype=int)
    for u, v in local_wall:
        r, c = _map_local(u, v, rad, anchor)
        grid[r, c] = wall_color

    if not _place_marker(grid, marker_color, rad, anchor, gap):
        return None

    interior = _local_interior(local_wall_open, depth, gap)
    wall_set = set(local_wall_open)
    adjacent = [
        p
        for p in interior
        if any((p[0] + dr, p[1] + dc) in wall_set for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)])
    ]
    seed_pool = [p for p in adjacent if p[0] >= max(1, depth - SEED_DEPTH_LOOKBACK)]
    if not seed_pool:
        return None

    blob = _grow_seed_blob(random.choice(seed_pool), set(interior))
    if len(blob) == 2:
        return None
    for u, v in blob:
        r, c = _map_local(u, v, rad, anchor)
        if grid[r, c] != bg:
            return None
        grid[r, c] = fill_color

    try:
        nominal_out = main(grid)
    except Exception:
        return None
    nominal_fill_zone = (nominal_out == fill_color) & (grid == bg)

    n_obstacles = random.randint(0, MAX_OBSTACLES)
    if n_obstacles and not obstacle_colors:
        available = [c for c in range(10) if c not in (bg, wall_color, fill_color, marker_color)]
        obstacle_colors = random.sample(available, random.randint(1, 2))
    if n_obstacles:
        candidates, fill_candidates = _find_obstacle_candidates(grid, bg, wall_color, nominal_fill_zone)
        _place_obstacles(n_obstacles, candidates, fill_candidates, obstacle_colors, grid)

    try:
        out = main(grid)
    except Exception:
        return None
    changed_bg = (out != grid) & (grid == bg)
    if int(changed_bg.sum()) < 8:
        return None
    try:
        orig_out = _original_main(grid)
    except Exception:
        return None
    if orig_out.shape != out.shape or np.array_equal(orig_out, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid b9e38dc0 D13 input")
