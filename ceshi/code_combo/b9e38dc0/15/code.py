# Combo variant of b9e38dc0 (index 15)
# combo_spec: D4/3+D13/2
# Name: 边框标记触发+外部标记指示方向

import importlib.util
import random
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_cdt

from arc_utils import components_mask, get_bg, neighbors4, neighbors8


# ---------------------------------------------------------------------------
# Design analysis for combo D4/3 + D13/2
#
# a) Concrete behavior trace:
#    Base task:
#      - There is an open funnel/U wall.
#      - A fill-color blob already sits inside the pocket.
#      - The wall endpoints reveal which side is open, so flood-fill starts from the
#        interior blob and then radiates out through that visible mouth.
#    Fused variant:
#      - The old mouth is capped, so the wall now looks closed on the radiate side.
#      - The old interior fill blob is removed entirely.
#      - A single trigger pixel of the fill color is embedded on the opposite bridge
#        (the closed end of the original open wall). This is the D4/3 source change.
#      - A separate 2-pixel marker is placed outside the capped side with a 1-cell bg
#        gap. The near pixel is the tail and the far pixel is the head, so the pair
#        points in one cardinal direction. This is the D13/2 direction change.
#      - Solver flow becomes:
#          1. Find the unique wall-touching non-wall pixel: this is the trigger color.
#          2. Decode the outside 2-pixel marker to obtain the radiate direction.
#          3. Treat the trigger pixel as part of the wall, recover the capped wall,
#             and carve a virtual breach through the capped side selected by the arrow.
#          4. Start BFS from the trigger's inner-side neighbor, not from the trigger
#             pixel itself, then let the fill pass through the virtual breach and
#             continue outward between the two endpoint rays.
#    Small local example in the wall's own coordinates:
#      - Open-wall original mouth endpoints were at local (0, 0) and (0, gap).
#      - Fusion adds cap pixels (0, 1..gap-1), places trigger at (depth, v_mark),
#        and places the outside arrow one gap away from the mouth side.
#      - Output recolors both the interior bg and those cap-interior pixels, so the
#        capped side behaves like a hidden opening rather than a visible one.
#
# b) Complexity calibration:
#    - The trigger is exactly one pixel and always wall-touching, so it is easy to
#      separate from the outside arrow and from sparse obstacles.
#    - The outside arrow has only four possible directions and always uses one 2-pixel
#      4-connected component, which gives repeated evidence across examples.
#    - No color remapping is introduced: trigger color is the fill color, wall keeps
#      its dominant non-bg color, and the direction marker is recognized by shape and
#      placement rather than by numeric color identity.
#    - The original wall family, obstacle shadows, and radiating fill remain intact,
#      so the task still reads as a variant of the same puzzle instead of a new one.
#
# c) Hidden-assumption audit against the original solver:
#    - Original `fill = closest non-wall color to wall` assumed a passable fill seed
#      exists off the wall. That fails here because the trigger sits on the border.
#      New solver reads the trigger directly and starts from its inner neighbor.
#    - Original endpoint detection assumed the wall component itself exposes degree-1
#      mouth endpoints. That fails after capping. New solver decodes direction from
#      the external marker, then reconstructs virtual endpoints on the marked face.
#    - Original passability assumed every wall-colored pixel is solid forever. That
#      fails on the capped side because some wall pixels now represent a hidden mouth.
#      New solver explicitly marks cap-interior cells as a virtual breach.
#    - Original obstacle logic treated every non-bg non-fill non-wall pixel as an
#      obstacle. That would incorrectly make the external arrow cast shadows, so the
#      solver removes those two cells from the obstacle mask.
#    - Combining D4/3 and D13/2 adds a new dependency: the trigger alone is not
#      enough to infer an interior start direction, so the solver must decode the
#      arrow first and then score the trigger's 4-neighbors by dot product with that
#      arrow direction.
#
# d) Placement legality:
#    - Generator first builds the original open wall in local coordinates, then caps
#      the mouth side in-place. This guarantees the virtual mouth lies exactly where
#      the original mouth used to be.
#    - The trigger is chosen only from bridge pixels at local row `depth` whose cell
#      immediately toward the mouth belongs to the interior. This guarantees a legal
#      inner-side BFS start exists and points along the decoded radiate direction.
#    - The outside marker uses a one-cell bg gap and length 2, and the original
#      open-end margin is kept so the marker and rays fit without leaving the grid.
#    - Obstacles are placed only on bg cells and are kept at chessboard distance > 1
#      from both the wall/trigger mass and the direction marker, preventing overlap
#      and reducing accidental visual ambiguity.
#
# e) Visual observability:
#    - Humans can see that the only fill-colored pixel in the input is stuck to the
#      wall border rather than floating inside the pocket.
#    - Humans can also see that the wall mouth is visibly sealed, so the only clue
#      for where the exterior radiation should go is the detached 2-pixel arrow.
#    - Across examples the same wall family appears with different arrow directions
#      and different trigger locations on the closed bridge, making both dimensions
#      observable from a few I/O pairs.
#
# f) Visual cleanliness:
#    - The two markers live in different zones: trigger on the wall, arrow outside
#      the wall with a bg gap. They do not overlap or share adjacency patterns.
#    - Obstacles remain sparse singleton pixels, so they do not form 2-pixel shapes
#      that could be mistaken for the outside arrow.
#    - The capped side is a clean contiguous bar, so the hidden-opening effect is
#      visually crisp in the output instead of noisy.
#
# g) Inferability of changes:
#    - The key state changes have structural cues: border-embedded fill color means
#      "source", detached 2-pixel aligned marker means "direction", and the capped
#      bar that becomes filled in the output reveals the hidden breach.
#    - Because there is no internal fill blob anymore, the solver cannot regress to
#      "pick the nearest interior color"; because the mouth is capped, it cannot
#      regress to "read the visible opening". Both old cues are intentionally removed.
#
# Post-change solver assumptions and generator guarantees:
#    - Exactly one wall-touching non-bg non-wall pixel exists; generator enforces a
#      single trigger and keeps obstacles away from the wall.
#    - Exactly one 2-pixel 4-connected component outside the wall has unequal wall
#      distances from its two pixels; generator makes that the only external arrow.
#    - The capped side is contiguous across the selected face; generator explicitly
#      adds every cap pixel between the former mouth endpoints.
#    - The trigger has at least one passable 4-neighbor in the arrow direction and
#      the best-scoring such neighbor is inside the enclosure; generator chooses the
#      trigger from bridge pixels whose inward local predecessor is interior.
# ---------------------------------------------------------------------------


# -- Configuration constants --------------------------------------------------
GRID_H_VERT = (11, 18)
GRID_H_HORIZ = (8, 18)
GRID_W = (15, 19)

WALL_PIXELS_MIN = 17
WALL_PIXELS_MAX = 25
DEPTH_VERT = (5, 8)
DEPTH_HORIZ = (5, 8)
GAP_VERT = (6, 7)
GAP_HORIZ = (5, 7)
TIP_12_PROB = 0.5

OPEN_END_MARGIN = 3
MARKER_GAP = 1
MARKER_LEN = 2

MAX_OBSTACLES = 5
OBSTACLE_IN_FILL_PROB = 0.65
OBSTACLE_MIN_SEP = 2

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
        spec = importlib.util.spec_from_file_location(
            "b9e38dc0_orig_combo15", root / "arc2_solver_select" / "b9e38dc0.py"
        )
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


def _local_neighbors(pt: tuple[int, int], pts: set[tuple[int, int]]) -> list[tuple[int, int]]:
    u, v = pt
    out = []
    for uu in range(u - 1, u + 2):
        for vv in range(v - 1, v + 2):
            if (uu, vv) != pt and (uu, vv) in pts:
                out.append((uu, vv))
    return out


def _find_trigger_marker(g: np.ndarray, bg: int, wall: int) -> tuple[int, int]:
    h, w = g.shape
    marker_candidates = []
    for r, c in zip(*np.where((g != bg) & (g != wall))):
        r = int(r)
        c = int(c)
        if any(0 <= rr < h and 0 <= cc < w and g[rr, cc] == wall for rr, cc in neighbors8(r, c)):
            marker_candidates.append((r, c))
    assert len(marker_candidates) == 1, "expected exactly one wall-touching trigger pixel"
    return marker_candidates[0]


def _find_direction_marker(
    g: np.ndarray,
    bg: int,
    wall: int,
    fill: int,
    wall_comp: set[tuple[int, int]],
) -> tuple[tuple[int, int], set[tuple[int, int]]]:
    candidates = []
    allowed = [int(v) for v in np.unique(g) if int(v) not in (bg, wall, fill)]
    for color in allowed:
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
    assert len(candidates) == 1, "expected exactly one external directional marker"
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
    assert side_vs, "expected capped side on the marked face"
    v_lo, v_hi = side_vs[0], side_vs[-1]
    for v in range(v_lo, v_hi + 1):
        assert (0, v) in local_pts, "expected contiguous cap across the marked side"
    assert v_hi - v_lo >= 2, "expected non-trivial cap span"
    return side_value, local_pts, v_lo, v_hi


def _virtual_endpoint_rays(
    local_pts: set[tuple[int, int]],
    rad: tuple[int, int],
    v_lo: int,
    v_hi: int,
) -> list[tuple[tuple[int, int], bool]]:
    out = []
    for ep in [(0, v_lo), (0, v_hi)]:
        arm_neighbors = [p for p in _local_neighbors(ep, local_pts) if p[0] > 0]
        assert arm_neighbors, "expected an inward arm neighbor at each virtual endpoint"
        n1 = min(arm_neighbors, key=lambda p: (p[0], abs(p[1] - ep[1])))
        d1 = (_sgn(n1[0] - ep[0]), _sgn(n1[1] - ep[1]))
        d_use = d1
        is_12 = False
        n2_candidates = [p for p in _local_neighbors(n1, local_pts) if p != ep and p[0] >= n1[0]]
        diag_candidates = [
            p
            for p in n2_candidates
            if abs(p[0] - n1[0]) == 1 and abs(p[1] - n1[1]) == 1
        ]
        if (d1[0] == 0) ^ (d1[1] == 0) and diag_candidates:
            n2 = min(diag_candidates, key=lambda p: (p[0], abs(p[1] - ep[1])))
            d_use = (_sgn(n2[0] - n1[0]), _sgn(n2[1] - n1[1]))
            is_12 = True
        out_local = (-d_use[0], -d_use[1])
        out.append((_grid_dir_from_local(out_local[0], out_local[1], rad), is_12))
    return out


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill a sealed wall pocket by reading a border trigger and an external direction marker.

    Input structure:
    - The most common color is background. The dominant non-background color is a capped funnel/U wall whose old mouth has been closed off.
    - A single fill-colored trigger pixel is embedded on the opposite wall bridge, while a detached 2-pixel marker outside the wall points along the radiate direction. Other sparse non-background pixels are obstacles.

    Rule:
    - Treat the trigger as part of the wall, decode the arrow to decide which capped side becomes a virtual opening, and start flood-fill from the trigger's inner-side neighbor.
    - Fill the interior and continue outward through that virtual opening, with endpoint rays inherited from the wall geometry and obstacles casting background shadows along the decoded direction.

    Key insight: Neither old cue survives on its own: the fill source is no longer inside the enclosure, and the opening is no longer visible in the wall shape, so the solver must combine the border trigger with the detached arrow.
    """
    g = np.array(grid, copy=True)
    h, w = g.shape

    bg = int(get_bg(g))
    non_bg = [int(v) for v in np.unique(g) if int(v) != bg]
    assert non_bg, "expected at least one non-background color"

    wall = max(non_bg, key=lambda c: int((g == c).sum()))
    wall_mask = g == wall

    trigger = _find_trigger_marker(g, bg, wall)
    fill = int(g[trigger])

    combined_wall = wall_mask.copy()
    combined_wall[trigger] = True
    comps = [
        set((int(r), int(c)) for r, c in pts)
        for pts in components_mask(combined_wall, connectivity=8)
    ]
    assert comps, "expected at least one wall connected component"
    wall_comp = next((comp for comp in comps if trigger in comp), None)
    assert wall_comp is not None, "expected trigger to belong to the wall component"

    comp_mask = np.zeros_like(combined_wall, dtype=bool)
    for r, c in wall_comp:
        comp_mask[r, c] = True

    rad, dir_marker_cells = _find_direction_marker(g, bg, wall, fill, wall_comp)
    side_value, local_pts, v_lo, v_hi = _wall_side_info(wall_comp, rad)
    ray_info = _virtual_endpoint_rays(local_pts, rad, v_lo, v_hi)

    breach = np.zeros_like(comp_mask, dtype=bool)
    if rad == (-1, 0):
        for c in range(v_lo + 1, v_hi):
            breach[side_value, c] = True
    elif rad == (1, 0):
        for c in range(v_lo + 1, v_hi):
            breach[side_value, c] = True
    elif rad == (0, -1):
        for r in range(v_lo + 1, v_hi):
            breach[r, side_value] = True
    else:
        for r in range(v_lo + 1, v_hi):
            breach[r, side_value] = True

    barrier = comp_mask.copy()
    barrier[breach] = False

    endpoint_cells = []
    if rad == (-1, 0):
        endpoint_cells = [(side_value, v_lo), (side_value, v_hi)]
    elif rad == (1, 0):
        endpoint_cells = [(side_value, v_lo), (side_value, v_hi)]
    elif rad == (0, -1):
        endpoint_cells = [(v_lo, side_value), (v_hi, side_value)]
    else:
        endpoint_cells = [(v_lo, side_value), (v_hi, side_value)]

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

    for ep, (d, is_12) in zip(endpoint_cells, ray_info):
        if is_12:
            add_ray_12(ep, d)
        else:
            add_ray(ep, d)

    passable = (~barrier) & ((g == bg) | (g == fill) | breach)

    seed_candidates = []
    for rr, cc in neighbors4(*trigger):
        if 0 <= rr < h and 0 <= cc < w and passable[rr, cc]:
            score = (rr - trigger[0]) * rad[0] + (cc - trigger[1]) * rad[1]
            seed_candidates.append((score, int(rr), int(cc)))
    assert seed_candidates, "expected an inner-side BFS start next to the trigger"
    best_score, seed_r, seed_c = max(seed_candidates)
    assert best_score > 0, "expected trigger interior to lie in the arrow-selected direction"

    seen = _seed_fill(passable, [(seed_r, seed_c)])
    filled_mask = seen & ((g == bg) | breach)

    out = g.copy()
    out[filled_mask] = fill

    obstacles = (g != bg) & (g != fill) & (g != wall)
    for r, c in dir_marker_cells:
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
    else:
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


def _build_local_open_wall(rad, h, w):
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


def _obstacle_fits(pos, placed):
    r, c = pos
    return all(max(abs(r - rr), abs(c - cc)) >= OBSTACLE_MIN_SEP for rr, cc in placed)


def _find_obstacle_candidates(grid, bg, solid_mask, dir_marker_mask, nominal_fill_zone):
    dist_solid = distance_transform_cdt(~solid_mask, metric="chessboard")
    dist_arrow = distance_transform_cdt(~dir_marker_mask, metric="chessboard")
    eligible = (grid == bg) & (dist_solid > 1) & (dist_arrow > 1)
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


def _place_direction_marker(grid, marker_color, rad, anchor, gap):
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
            return None
        if grid[r, c] != grid[0, 0]:
            return None
    for r, c in coords:
        grid[r, c] = marker_color
    return {(tail_r, tail_c), (head_r, head_c)}


def _cap_interior_cells(rad, anchor, gap):
    coords = []
    for v in range(1, gap):
        coords.append(_map_local(0, v, rad, anchor))
    return coords


def _gen_attempt():
    rad = random.choice([(-1, 0), (1, 0), (0, -1), (0, 1)])
    h = random.randint(*GRID_H_VERT) if rad[0] else random.randint(*GRID_H_HORIZ)
    w = random.randint(*GRID_W)

    built = _build_local_open_wall(rad, h, w)
    if built is None:
        return None
    local_wall_open, depth, gap = built
    local_wall = _cap_local_wall(local_wall_open, gap)
    anchor = _place_anchor(rad, h, w, depth, gap)

    palette = random.sample(range(10), random.randint(*PALETTE_SIZE))
    bg, wall_color, fill_color, marker_color = palette[:4]
    obstacle_colors = [c for c in palette[4:] if c not in (bg, wall_color, fill_color, marker_color)]

    grid = np.full((h, w), bg, dtype=int)
    for u, v in local_wall:
        r, c = _map_local(u, v, rad, anchor)
        grid[r, c] = wall_color

    dir_marker_cells = _place_direction_marker(grid, marker_color, rad, anchor, gap)
    if dir_marker_cells is None:
        return None

    interior = set(_local_interior(local_wall_open, depth))
    if not interior:
        return None

    bridge_candidates = [
        (u, v)
        for u, v in local_wall_open
        if u == depth and (u - 1, v) in interior
    ]
    if not bridge_candidates:
        return None

    trigger_u, trigger_v = random.choice(bridge_candidates)
    trigger_r, trigger_c = _map_local(trigger_u, trigger_v, rad, anchor)
    if grid[trigger_r, trigger_c] != wall_color:
        return None
    grid[trigger_r, trigger_c] = fill_color

    if int((grid == fill_color).sum()) != 1:
        return None

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
        solid_mask = (grid == wall_color) | (grid == fill_color)
        dir_marker_mask = np.zeros_like(grid, dtype=bool)
        for r, c in dir_marker_cells:
            dir_marker_mask[r, c] = True
        candidates, fill_candidates = _find_obstacle_candidates(
            grid, bg, solid_mask, dir_marker_mask, nominal_fill_zone
        )
        _place_obstacles(n_obstacles, candidates, fill_candidates, obstacle_colors, grid)

    try:
        out = main(grid)
    except Exception:
        return None

    new_fill = (out == fill_color) & (grid == bg)
    if int(new_fill.sum()) < MIN_NEW_FILL or not _touches_border(new_fill):
        return None

    cap_interior = _cap_interior_cells(rad, anchor, gap)
    breach_fill = sum(out[r, c] == fill_color for r, c in cap_interior)
    if breach_fill != len(cap_interior):
        return None

    try:
        orig_out = _original_main(grid)
    except Exception:
        orig_out = None
    if orig_out is not None and orig_out.shape == out.shape and np.array_equal(orig_out, out):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid b9e38dc0 combo input")
