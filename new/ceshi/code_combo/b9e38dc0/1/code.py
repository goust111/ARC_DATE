# Combo variant of b9e38dc0 (index 1)
# combo_spec: D1/0+D5/1
# Name: 仅辐射+距离编码终止

# Design analysis for the fused variant
# a) Concrete behavior:
#    Original examples flood-fill the whole interior and then keep expanding to the grid edge.
#    Here a 4-pixel source blob inside an upward-opening wall instead emits 4 short upward rays:
#    the cavity stays mostly hollow, and each ray stops after 4 steps instead of reaching the border.
# b) Complexity calibration:
#    The generator forces source blobs to have 3-4 pixels and to form a single contiguous strip
#    perpendicular to the opening. This avoids trivial 1-pixel cases and keeps the visible beam
#    length exactly equal to the source count instead of letting stacked rays blur the cue.
# c) Broken original assumptions:
#    The old `barrier -> passable -> filled_mask` chain assumed one flood region plus edge-reaching
#    radiation. That no longer holds. The new solver keeps only wall/opening detection from the wall,
#    then casts independent finite rays from the source pixels.
# d) Placement legality:
#    Every generated source pixel is chosen from interior cells with a clear straight path toward the
#    opening, so rays do not start inside walls. Walls, sources, and optional obstacle noise are
#    placed on disjoint cells inside bounds.
# e) Visual observability:
#    A human can now distinguish the task from the original because the interior is not broadly
#    painted, the output usually does not touch the border, and the beam length matches the count
#    of source pixels.
# f) Visual cleanliness:
#    Optional obstacle noise is kept off the nominal ray footprint, so the main wall/source/ray
#    relation remains readable instead of becoming a cluttered overlay.
# g) Color inference:
#    No invisible ordering is introduced. The fill color is still the non-wall color nearest the wall;
#    the new rule is expressed by geometry and source-count, not by numeric color ids.

import random

import numpy as np

from arc_utils import components_mask, get_bg, neighbors8


def _sgn(x: int) -> int:
    return (x > 0) - (x < 0)


def _scene_info(grid: np.ndarray) -> dict:
    g = np.array(grid, copy=True)
    h, w = g.shape

    bg = get_bg(g)
    vals = [int(v) for v in np.unique(g)]
    non_bg = [v for v in vals if v != bg]
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
    assert fill_pos, "expected at least one fill/source pixel"

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
    assert len(endpoints) >= 2, "expected two wall endpoints"

    def outward_dir(ep: tuple[int, int]) -> tuple[int, int]:
        n1 = comp_neighbors(*ep)[0]
        d1 = (_sgn(n1[0] - ep[0]), _sgn(n1[1] - ep[1]))  # inward direction
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
    assert rad != (0, 0), "expected a cardinal opening direction"

    return {
        "grid": g,
        "bg": bg,
        "wall": wall,
        "fill": fill,
        "fill_pos": fill_pos,
        "wall_comp": wall_comp,
        "comp_mask": comp_mask,
        "rad": rad,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Turn the interior source blob into a finite beam that exits through the wall opening.

    Input structure:
    - The background is the dominant color. The most common non-background color is an 8-connected
      wall component forming a partial enclosure with one open side.
    - A small fill-colored source blob sits inside that enclosure; any other non-background pixels
      are preserved as fixed blockers/noise.

    Rule:
    - Do not flood-fill the enclosed area. Instead, every source pixel casts a straight ray in the
      wall opening direction.
    - All rays share the same length: the number of source pixels in the blob. Rays stop early if
      they leave the grid or hit a non-background, non-fill pixel.

    Key insight:
    - The wall still tells you which way "out" is, but the stopping condition is now numerical:
      count the interior source pixels, then use that count as the beam length.
    """
    info = _scene_info(np.asarray(grid, dtype=int))
    g = info["grid"]
    h, w = g.shape
    bg = info["bg"]
    fill = info["fill"]
    rad = info["rad"]

    ray_len = len(info["fill_pos"])
    out = g.copy()

    for sr, sc in info["fill_pos"]:
        rr, cc = sr, sc
        for _ in range(ray_len):
            rr += rad[0]
            cc += rad[1]
            if not (0 <= rr < h and 0 <= cc < w):
                break
            val = int(out[rr, cc])
            if val == bg:
                out[rr, cc] = fill
            elif val == fill:
                continue
            else:
                break

    return out


# New solver assumptions guaranteed by the generator:
# - source pixels all lie inside the chosen wall component and have a clear straight path to the opening
# - the source blob spans at least two positions perpendicular to the ray direction, so the count-coded
#   stopping rule is visible as repeated parallel segments
# - optional obstacle noise stays away from the nominal beam, so the main D1/0 + D5/1 effect remains legible


# --------------- generator ---------------

# Canvas size
GRID_H_VERT = (11, 17)
GRID_H_HORIZ = (9, 17)
GRID_W = (15, 19)

# Wall geometry
WALL_PIXELS_MIN = 17
WALL_PIXELS_MAX = 25
DEPTH_VERT = (5, 6)
DEPTH_HORIZ = (5, 6)
GAP_VERT = (6, 7)
GAP_HORIZ = (5, 7)
TIP_12_PROB = 0.5

# Placement margins
OPEN_END_MARGIN = 3

# Source blob
SEED_DEPTH_LOOKBACK = 4
SEED_SIZE_CHOICES = [3, 3, 4, 4]
MIN_OUTSIDE_STEPS = 1

# Noise obstacles
MAX_OBSTACLES = 2
OBSTACLE_MIN_SEP = 2

# Validation
MIN_NEW_FILL = 8
MIN_OUTSIDE_FILL = 2
PALETTE_SIZE = (3, 4)
GEN_RETRIES = 400


def _degree1_pixels(points: set[tuple[int, int]]) -> list[tuple[int, int]]:
    pts = set(points)
    return [
        (r, c)
        for r, c in pts
        if sum((rr, cc) in pts for rr, cc in neighbors8(r, c)) == 1
    ]


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


def _build_local_wall(rad: tuple[int, int], h: int, w: int):
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


def _map_local(u: int, v: int, rad: tuple[int, int], anchor: tuple[int, int]) -> tuple[int, int]:
    r0, c0 = anchor
    if rad == (-1, 0):
        return r0 + u, c0 + v
    if rad == (1, 0):
        return r0 - u, c0 + v
    if rad == (0, -1):
        return r0 + v, c0 + u
    return r0 + v, c0 - u


def _place_anchor(rad: tuple[int, int], h: int, w: int, depth: int, gap: int) -> tuple[int, int]:
    if rad == (-1, 0):
        return random.randint(OPEN_END_MARGIN, h - depth - 3), random.randint(0, w - gap - 1)
    if rad == (1, 0):
        return random.randint(depth + 2, h - 4), random.randint(0, w - gap - 1)
    if rad == (0, -1):
        return random.randint(0, h - gap - 1), random.randint(OPEN_END_MARGIN, w - depth - 3)
    return random.randint(0, h - gap - 1), random.randint(depth + 2, w - 4)


def _local_interior(wall: set[tuple[int, int]], depth: int) -> list[tuple[int, int]]:
    by_row: dict[int, list[int]] = {}
    for u, v in wall:
        by_row.setdefault(u, []).append(v)
    interior = []
    for u in range(1, depth):
        row = by_row.get(u)
        if row is None or len(row) < 2:
            continue
        lo, hi = min(row), max(row)
        for v in range(lo + 1, hi):
            if (u, v) not in wall:
                interior.append((u, v))
    return interior


def _clear_to_open(wall: set[tuple[int, int]], u: int, v: int) -> bool:
    return all((uu, v) not in wall for uu in range(0, u + 1))


def _choose_source_blob(local_wall: set[tuple[int, int]], depth: int):
    by_row: dict[int, list[int]] = {}
    for u, v in _local_interior(local_wall, depth):
        if _clear_to_open(local_wall, u, v):
            by_row.setdefault(u, []).append(v)

    eligible_rows = [u for u in by_row if u >= max(1, depth - SEED_DEPTH_LOOKBACK)]
    if not eligible_rows:
        return None

    size_choices = SEED_SIZE_CHOICES[:]
    random.shuffle(size_choices)
    for target_size in size_choices:
        rows = [u for u in eligible_rows if u <= target_size - MIN_OUTSIDE_STEPS]
        random.shuffle(rows)
        for u in rows:
            cols = sorted(by_row[u])
            runs = []
            run = [cols[0]]
            for v in cols[1:]:
                if v == run[-1] + 1:
                    run.append(v)
                else:
                    runs.append(run)
                    run = [v]
            runs.append(run)
            random.shuffle(runs)
            for run in runs:
                if len(run) < target_size:
                    continue
                start_idx = random.randint(0, len(run) - target_size)
                chosen = run[start_idx : start_idx + target_size]
                return {(u, v) for v in chosen}
    return None


def _outside_region_mask(shape: tuple[int, int], rad: tuple[int, int], anchor: tuple[int, int]) -> np.ndarray:
    h, w = shape
    mask = np.zeros(shape, dtype=bool)
    r0, c0 = anchor
    if rad == (-1, 0):
        mask[:r0, :] = True
    elif rad == (1, 0):
        mask[r0 + 1 :, :] = True
    elif rad == (0, -1):
        mask[:, :c0] = True
    else:
        mask[:, c0 + 1 :] = True
    return mask


def _obstacle_fits(pos: tuple[int, int], placed: list[tuple[int, int]]) -> bool:
    r, c = pos
    return all(max(abs(r - rr), abs(c - cc)) >= OBSTACLE_MIN_SEP for rr, cc in placed)


def _place_noise_obstacles(
    grid: np.ndarray,
    bg: int,
    wall_color: int,
    nominal_fill_mask: np.ndarray,
    obstacle_colors: list[int],
) -> None:
    if not obstacle_colors:
        return

    h, w = grid.shape
    candidates = []
    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) != bg or nominal_fill_mask[r, c]:
                continue
            near_wall = False
            for rr in range(max(0, r - 1), min(h, r + 2)):
                for cc in range(max(0, c - 1), min(w, c + 2)):
                    if int(grid[rr, cc]) == wall_color:
                        near_wall = True
                        break
                if near_wall:
                    break
            if not near_wall:
                candidates.append((r, c))

    random.shuffle(candidates)
    placed: list[tuple[int, int]] = []
    n_obstacles = random.randint(0, MAX_OBSTACLES)
    for p in candidates:
        if len(placed) >= n_obstacles:
            break
        if _obstacle_fits(p, placed):
            placed.append(p)

    for i, (r, c) in enumerate(placed):
        grid[r, c] = obstacle_colors[i % len(obstacle_colors)]


def _gen_attempt():
    rad = random.choice([(-1, 0), (1, 0), (0, -1), (0, 1)])
    h = random.randint(*GRID_H_VERT) if rad[0] else random.randint(*GRID_H_HORIZ)
    w = random.randint(*GRID_W)

    built = _build_local_wall(rad, h, w)
    if built is None:
        return None
    local_wall, depth, gap = built
    anchor = _place_anchor(rad, h, w, depth, gap)

    blob = _choose_source_blob(local_wall, depth)
    if blob is None:
        return None

    palette = random.sample(range(10), random.randint(*PALETTE_SIZE))
    bg, wall_color, fill_color = palette[:3]
    obstacle_colors = palette[3:]

    grid = np.full((h, w), bg, dtype=int)
    for u, v in local_wall:
        r, c = _map_local(u, v, rad, anchor)
        grid[r, c] = wall_color

    for u, v in blob:
        r, c = _map_local(u, v, rad, anchor)
        if int(grid[r, c]) != bg:
            return None
        grid[r, c] = fill_color

    try:
        nominal_out = main(grid)
    except Exception:
        return None

    nominal_fill = (nominal_out == fill_color) & (grid == bg)
    _place_noise_obstacles(grid, bg, wall_color, nominal_fill, obstacle_colors)

    try:
        out = main(grid)
    except Exception:
        return None

    new_fill = (out == fill_color) & (grid == bg)
    outside_fill = new_fill & _outside_region_mask(grid.shape, rad, anchor)

    if int(new_fill.sum()) < MIN_NEW_FILL:
        return None
    if int(outside_fill.sum()) < MIN_OUTSIDE_FILL:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid b9e38dc0 combo input")
