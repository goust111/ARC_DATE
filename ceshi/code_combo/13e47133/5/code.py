# Combo variant of 13e47133 (index 5)
# combo_spec: D3/3+D8/2
# Name: 边缘控制像素+全局共享调色板

import random

import numpy as np

from arc_utils import components_mask


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace:
#    The wall geometry and the concentric-band output stay close to the source
#    task, but two rule stages change together.
#    Original:
#      - each region owns 0-3 interior diagonal markers
#      - each region builds its own palette from local marker depths
#    Combo:
#      - a clue region places 0-3 control pixels on one outer canvas side that it
#        touches; controls occupy consecutive border cells of that region
#      - after recovering those region-owned edge controls, all control colors are
#        pooled into one shared palette in global row-major order
#      - every region is then filled with the same shared palette by Chebyshev
#        distance to that region boundary
#    Example:
#      - top region controls at (0,4)=7, (0,5)=3
#      - left-bottom region control at (8,0)=4
#      - D3-only behavior would fill the first region from local [7,3] and the
#        second from local [4] (or [bg,4] if its first offset were >0)
#      - combo behavior first associates the controls to their regions, then pools
#        them by row-major position into shared palette [7,3,4], and both regions
#        cycle through [7,3,4]
#    The visual difference is therefore not only "markers moved to the border" but
#    also "different regions now repeat the same cross-region color order".
#
# b) Complexity calibration:
#    - regions remain rectangles from recursive full-row/full-column wall splits,
#      which keeps border ownership and boundary distance easy to perceive
#    - each active region uses exactly one hosting side and 1-3 controls, with
#      global total kept short (3-6) so the shared palette stays human-readable
#    - at least two regions contribute controls, and at least one contributing
#      region has multiple controls, so both the border-grouping cue and the
#      global-sharing cue appear in a single sample
#
# c) Hidden assumption audit against the original main():
#    - original `v != base` inside a region implied an interior marker whose own
#      boundary depth gave its palette index; that is false here because all clues
#      live on the outer border and usually have cell depth 0
#    - original `dist_to_color` was local per component; that is false here because
#      the fill palette is pooled across regions
#    - original `if ds[0] > 0: prepend base` was a local-palette rule; the shared
#      palette no longer carries per-region base prefixes
#    - the new solver assumes a region's controls, if any, lie on at most one outer
#      side; the generator enforces that, and avoids ambiguous corner-only singles
#      when a multi-side region can host an interior border position instead
#
# d) Placement legality:
#    - only regions touching the outer canvas may host controls
#    - controls occupy consecutive cells on one touched side, stay within that
#      region's border span, and never overwrite walls or other controls
#    - control counts are planned against per-region side capacity before colors are
#      assigned, so generation never relies on out-of-bounds retries
#
# e) Visual observability:
#    - edge controls are immediately visible because the only non-wall/non-base
#      input clues sit on the outer border
#    - the shared palette is inferable because the same band order reappears across
#      multiple separated regions, even when those regions contributed different
#      local control counts
#    - palette order depends only on visible row-major positions of controls, never
#      on numeric color ids
#
# f) Visual cleanliness:
#    - walls stay single-color separators and interiors remain uncluttered before
#      transformation
#    - the generator keeps the shared palette short and colors globally distinct,
#      preventing noisy repeated clue colors on the border
#
# g) Change inferability:
#    - humans can read "these border pixels belong to this region edge" and also
#      "all regions reuse one combined palette"
#    - the generator rejects samples where the combo output matches the D3-only
#      local-edge baseline, so the global-sharing half of the combo is always
#      behaviorally active rather than decorative


DIR8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
SIDE_PRIORITY = ("top", "bottom", "left", "right")


def _cc4(mask):
    return components_mask(mask, connectivity=4)


def _mode(arr):
    vals, counts = np.unique(np.asarray(arr), return_counts=True)
    return int(vals[counts.argmax()])


def _bg_wall(grid: np.ndarray) -> tuple[int, int]:
    bg = _mode(grid)
    other = grid[grid != bg]
    wall = bg if other.size == 0 else _mode(other)
    return int(bg), int(wall)


def _boundary_distance(component_mask: np.ndarray) -> np.ndarray:
    """
    Compute 8-neigh (Chebyshev) distance to boundary for True-cells in component_mask.
    Boundary cells are those that touch False or the grid edge in their 8-neighborhood.
    Returns dist array with -1 outside component.
    """
    h, w = component_mask.shape
    dist = np.full((h, w), -1, dtype=int)
    q: list[tuple[int, int]] = []

    for r in range(h):
        for c in range(w):
            if not component_mask[r, c]:
                continue
            is_boundary = False
            for dr, dc in DIR8:
                rr, cc = r + dr, c + dc
                if rr < 0 or rr >= h or cc < 0 or cc >= w or not component_mask[rr, cc]:
                    is_boundary = True
                    break
            if is_boundary:
                dist[r, c] = 0
                q.append((r, c))

    qi = 0
    while qi < len(q):
        r, c = q[qi]
        qi += 1
        d = dist[r, c]
        for dr, dc in DIR8:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and component_mask[rr, cc] and dist[rr, cc] == -1:
                dist[rr, cc] = d + 1
                q.append((rr, cc))
    return dist


def _edge_controls_for_component(g: np.ndarray, pts: np.ndarray, base: int):
    """
    Return one region's edge controls as [(r, c, color, offset_along_side), ...].

    Controls are non-base cells on the outer canvas border. The generator guarantees
    at most one hosting side per active region; the tie-breaker only handles the
    benign single-corner case robustly.
    """
    h, w = g.shape
    side_cells = {side: [] for side in SIDE_PRIORITY}
    side_controls = {side: [] for side in SIDE_PRIORITY}

    for r, c in pts:
        r = int(r)
        c = int(c)
        if r == 0:
            side_cells["top"].append((r, c))
            if int(g[r, c]) != int(base):
                side_controls["top"].append((r, c))
        if r == h - 1:
            side_cells["bottom"].append((r, c))
            if int(g[r, c]) != int(base):
                side_controls["bottom"].append((r, c))
        if c == 0:
            side_cells["left"].append((r, c))
            if int(g[r, c]) != int(base):
                side_controls["left"].append((r, c))
        if c == w - 1:
            side_cells["right"].append((r, c))
            if int(g[r, c]) != int(base):
                side_controls["right"].append((r, c))

    candidates = [(side, side_controls[side]) for side in SIDE_PRIORITY if side_controls[side]]
    if not candidates:
        return []

    best_len = max(len(cells) for _side, cells in candidates)
    chosen_side = next(side for side, cells in candidates if len(cells) == best_len)
    controls = side_controls[chosen_side]

    if chosen_side in ("top", "bottom"):
        span_start = min(c for _r, c in side_cells[chosen_side])
        raw = sorted((int(c - span_start), int(r), int(c), int(g[r, c])) for r, c in controls)
    else:
        span_start = min(r for r, _c in side_cells[chosen_side])
        raw = sorted((int(r - span_start), int(r), int(c), int(g[r, c])) for r, c in controls)

    ordered = []
    seen_offsets = set()
    for offset, r, c, color in raw:
        if offset in seen_offsets:
            continue
        seen_offsets.add(offset)
        ordered.append((int(r), int(c), int(color), int(offset)))
    return ordered


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repaint wall-separated regions with one shared palette assembled from border controls.

    Input structure:
    - A wall color partitions the grid into several non-wall regions.
    - Region interiors are mostly one base color.
    - The only clue pixels are non-base control pixels placed on outer border rows
      or columns touched by their corresponding regions.

    Rule:
    - For each region, recover its control pixels from the outer border side that
      hosts them.
    - Pool all recovered control colors into one global palette ordered by the
      controls' row-major positions in the grid.
    - Compute each region cell's Chebyshev distance to that region boundary and
      fill every region with the same cyclic palette via palette[distance mod n].

    Key insight: Border controls are still owned by individual regions, but they no
    longer define separate local palettes. Their visible global reading order
    determines a single palette that every region must share.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    bg, wall = _bg_wall(g)
    out = g.copy()
    free = g != wall
    comps = _cc4(free)

    region_data: list[tuple[np.ndarray, np.ndarray]] = []
    global_controls: list[tuple[int, int, int]] = []

    for pts in comps:
        comp_mask = np.zeros((h, w), dtype=bool)
        for r, c in pts:
            comp_mask[int(r), int(c)] = True

        vals = g[comp_mask]
        base = _mode(vals)
        dist = _boundary_distance(comp_mask)
        region_data.append((pts, dist))

        for r, c, color, _offset in _edge_controls_for_component(g, pts, base):
            global_controls.append((int(r), int(c), int(color)))

    global_controls.sort(key=lambda x: (x[0], x[1]))
    palette = [int(color) for _r, _c, color in global_controls]
    if not palette:
        palette = [int(bg)]

    m = len(palette)
    for pts, dist in region_data:
        for r, c in pts:
            out[int(r), int(c)] = palette[int(dist[int(r), int(c)]) % m]

    return out


def _main_local_edge_baseline(grid: np.ndarray) -> np.ndarray:
    """D3-only baseline: each region uses only its own edge controls."""
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    _bg, wall = _bg_wall(g)

    out = g.copy()
    free = g != wall
    comps = _cc4(free)

    for pts in comps:
        comp_mask = np.zeros((h, w), dtype=bool)
        for r, c in pts:
            comp_mask[int(r), int(c)] = True

        vals = g[comp_mask]
        base = _mode(vals)
        dist = _boundary_distance(comp_mask)
        controls = _edge_controls_for_component(g, pts, base)

        if controls:
            offsets = [offset for _r, _c, _color, offset in controls]
            palette = []
            if min(offsets) > 0:
                palette.append(int(base))
            palette.extend(int(color) for _r, _c, color, _offset in controls)
        else:
            palette = [int(base)]

        m = len(palette)
        for r, c in pts:
            out[int(r), int(c)] = palette[int(dist[int(r), int(c)]) % m]

    return out


# Generator-side guarantees for the combo solver:
# - only border-touching regions may host controls, so every clue is visibly on the
#   outer canvas edge rather than hidden inside a region
# - one side per active region avoids ambiguous region-to-control ownership
# - at least two regions contribute controls, and the combo output must differ from
#   the local-edge baseline, so global sharing is always active
# - control colors are globally distinct, making the row-major shared palette easy
#   to read directly from the input without numeric color assumptions

# ---- Canvas bounds ----
H_MIN, H_MAX = 10, 30
W_MIN, W_MAX = 13, 30

# ---- Wall subdivision ----
N_SPLITS_MIN, N_SPLITS_MAX = 2, 5
MIN_REGION_DIM = 3
SPLITTABLE_LEN = 2 * MIN_REGION_DIM + 1

# ---- Region count filter ----
N_REGIONS_MIN, N_REGIONS_MAX = 3, 6

# ---- Border controls ----
MARKERS_PER_REGION_MAX = 3
GLOBAL_CONTROLS_MIN, GLOBAL_CONTROLS_MAX = 3, 6
START_OFFSET_MAX = 1

# ---- Retry budget ----
GEN_MAX_ATTEMPTS = 400


def _place_walls(H: int, W: int, rng: random.Random):
    wall_mask = np.zeros((H, W), dtype=int)
    rects = [(0, 0, H - 1, W - 1)]
    n_splits = rng.randint(N_SPLITS_MIN, N_SPLITS_MAX)

    for _ in range(n_splits):
        candidates = []
        for i, (r0, c0, r1, c1) in enumerate(rects):
            can_h = (r1 - r0 + 1) >= SPLITTABLE_LEN
            can_v = (c1 - c0 + 1) >= SPLITTABLE_LEN
            if can_h or can_v:
                candidates.append((i, can_h, can_v))
        if not candidates:
            break

        i, can_h, can_v = rng.choice(candidates)
        r0, c0, r1, c1 = rects.pop(i)
        axes = ("h",) * can_h + ("v",) * can_v
        axis = rng.choice(axes)

        if axis == "h":
            rr = rng.randint(r0 + MIN_REGION_DIM, r1 - MIN_REGION_DIM)
            wall_mask[rr, c0:c1 + 1] = 1
            rects.append((r0, c0, rr - 1, c1))
            rects.append((rr + 1, c0, r1, c1))
        else:
            cc = rng.randint(c0 + MIN_REGION_DIM, c1 - MIN_REGION_DIM)
            wall_mask[r0:r1 + 1, cc] = 1
            rects.append((r0, c0, r1, cc - 1))
            rects.append((r0, cc + 1, r1, c1))

    return wall_mask, rects


def _pick_palette(rng: random.Random):
    colors = list(range(10))
    bg = rng.choice(colors)
    colors.remove(bg)
    wall = rng.choice(colors)
    colors.remove(wall)
    return bg, wall, colors


def _outer_sides(rect, H: int, W: int):
    r0, c0, r1, c1 = rect
    sides = []
    if r0 == 0:
        sides.append("top")
    if r1 == H - 1:
        sides.append("bottom")
    if c0 == 0:
        sides.append("left")
    if c1 == W - 1:
        sides.append("right")
    return sides


def _side_len(rect, side: str):
    r0, c0, r1, c1 = rect
    if side in ("top", "bottom"):
        return c1 - c0 + 1
    return r1 - r0 + 1


def _control_coord(rect, side: str, offset: int, H: int, W: int):
    r0, c0, r1, c1 = rect
    if side == "top":
        return 0, c0 + offset
    if side == "bottom":
        return H - 1, c0 + offset
    if side == "left":
        return r0 + offset, 0
    return r0 + offset, W - 1


def _region_capacity(rect, H: int, W: int):
    sides = _outer_sides(rect, H, W)
    if not sides:
        return 0
    return min(MARKERS_PER_REGION_MAX, max(_side_len(rect, side) for side in sides))


def _plan_control_counts(rects, H: int, W: int, rng: random.Random):
    capacities = [_region_capacity(rect, H, W) for rect in rects]
    candidates = [i for i, cap in enumerate(capacities) if cap >= 1]
    if len(candidates) < 2:
        return None

    max_total = min(GLOBAL_CONTROLS_MAX, sum(capacities[i] for i in candidates))
    if max_total < GLOBAL_CONTROLS_MIN:
        return None

    target_total = rng.randint(GLOBAL_CONTROLS_MIN, max_total)
    max_active = min(len(candidates), target_total - 1)
    if max_active < 2:
        return None

    # Sample the number of contributing regions and the global control count
    # independently enough that 3/4/5/6-control cases all appear regularly.
    active_n = rng.randint(2, max_active)
    active = None
    for _ in range(40):
        trial = rng.sample(candidates, active_n)
        if sum(capacities[idx] for idx in trial) >= target_total:
            active = trial
            break
    if active is None:
        return None

    counts = [0] * len(rects)
    for idx in active:
        counts[idx] = 1

    extra = target_total - active_n
    while extra > 0:
        expandable = [idx for idx in active if counts[idx] < capacities[idx]]
        if not expandable:
            return None
        idx = rng.choice(expandable)
        counts[idx] += 1
        extra -= 1

    return counts


def _place_controls_for_region(rect, count: int, H: int, W: int, rng: random.Random):
    if count == 0:
        return [], {"count": 0, "start": None, "side": "none"}

    sides = [side for side in _outer_sides(rect, H, W) if _side_len(rect, side) >= count]
    if not sides:
        return None

    side = rng.choice(sides)
    side_len = _side_len(rect, side)
    max_start = min(START_OFFSET_MAX, side_len - count)

    if count == 1 and len(sides) > 1:
        interior_starts = [d for d in range(max_start + 1) if 0 < d < side_len - 1]
        if interior_starts:
            start = rng.choice(interior_starts)
        else:
            start = rng.randint(0, max_start)
    else:
        start = rng.randint(0, max_start)

    placed = []
    for k in range(count):
        offset = start + k
        rr, cc = _control_coord(rect, side, offset, H, W)
        placed.append((int(rr), int(cc), int(offset), side))
    return placed, {"count": count, "start": start, "side": side}


def _stamp_controls(grid, rects, bg: int, marker_pool: list[int], rng: random.Random):
    H, W = grid.shape
    counts = _plan_control_counts(rects, H, W, rng)
    if counts is None:
        return None

    planned = []
    active_regions = 0
    multi_regions = 0
    for idx, rect in enumerate(rects):
        placed_meta = _place_controls_for_region(rect, counts[idx], H, W, rng)
        if placed_meta is None:
            return None
        placed, meta = placed_meta
        if meta["count"] > 0:
            active_regions += 1
        if meta["count"] > 1:
            multi_regions += 1
        for rr, cc, offset, side in placed:
            if grid[rr, cc] != bg:
                return None
            planned.append((rr, cc, idx, offset, side))

    if active_regions < 2 or multi_regions < 1:
        return None
    if not (GLOBAL_CONTROLS_MIN <= len(planned) <= min(GLOBAL_CONTROLS_MAX, len(marker_pool))):
        return None

    planned.sort(key=lambda x: (x[0], x[1]))
    colors = rng.sample(marker_pool, len(planned))

    used = set()
    region_counts = {}
    for color, (rr, cc, idx, _offset, _side) in zip(colors, planned):
        grid[rr, cc] = int(color)
        used.add(int(color))
        region_counts[idx] = region_counts.get(idx, 0) + 1

    if len(region_counts) < 2:
        return None

    return {
        "used_colors": used,
        "active_regions": active_regions,
        "multi_regions": multi_regions,
        "total_controls": len(planned),
    }


def _validate_color_order(grid, bg: int, wall: int):
    vals, counts = np.unique(grid, return_counts=True)
    order = np.argsort(-counts)
    if int(vals[order[0]]) != bg:
        return False
    return len(order) >= 2 and int(vals[order[1]]) == wall


def _gen_attempt(rng: random.Random):
    H = rng.randint(H_MIN, H_MAX)
    W = rng.randint(W_MIN, W_MAX)

    bg, wall, marker_pool = _pick_palette(rng)
    wall_mask, rects = _place_walls(H, W, rng)
    if not (N_REGIONS_MIN <= len(rects) <= N_REGIONS_MAX):
        return None

    grid = np.full((H, W), bg, dtype=int)
    grid[wall_mask == 1] = wall

    meta = _stamp_controls(grid, rects, bg, marker_pool, rng)
    if meta is None:
        return None
    if not _validate_color_order(grid, bg, wall):
        return None

    try:
        out = main(grid)
        baseline = _main_local_edge_baseline(grid)
    except Exception:
        return None

    if np.array_equal(grid, out):
        return None
    if np.array_equal(out, baseline):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_MAX_ATTEMPTS):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError(f"generate_input failed after {GEN_MAX_ATTEMPTS} attempts")
