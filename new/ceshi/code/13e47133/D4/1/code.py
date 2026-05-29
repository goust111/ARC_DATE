# Variant of 13e47133
# Dimension: D4 — 终止条件：填充的终点 (dst)
# Change: 填充直到整个区域被完全覆盖（无显式终止条件）... → 遇到其他颜色像素时停止

import random
import sys
from pathlib import Path

import numpy as np

_SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import components_mask


# Design analysis for the D4 variant
# ----------------------------------
# a) Concrete behavior and trace
#    The original task partitions the canvas with wall lines, then every non-wall
#    rectangle is filled completely by rectangular Chebyshev bands derived from
#    isolated marker pixels. The D4 variant keeps the same walls, same region
#    detection, and same palette extraction, but adds interior "blockers":
#    multi-cell solid patches whose color is not part of the palette. Filling now
#    grows inward from the four sides of a region. A side can paint a cell only if
#    the straight segment from that side to the cell contains no blocker. The cell
#    takes the palette color of the shallowest side that can still reach it. If
#    every side is blocked first, that cell stays base-colored. Small trace on a
#    7x7 region:
#    - Original: a cell at depth 2 always gets palette[2 % m].
#    - Variant: if a 2x2 blocker sits one row below the top edge, then cells under
#      that blocker are no longer reachable from the top. Some still get painted
#      from left/right/bottom; cells shadowed from all four sides remain base.
#    - Difference: palette reading is unchanged; only the propagation domain is
#      clipped by collisions.
# b) Complexity calibration
#    The variant must stay visually close to the seed task, so blockers are only a
#    small number of rectangular patches, not arbitrary noise. Markers remain
#    singleton pixels; blockers are multi-pixel monochrome components. This gives
#    the solver a stable structural distinction and makes the collision effect easy
#    to see in 2-4 examples.
# c) Implicit assumptions audit against the original main()
#    - `free = g != wall` still holds: blockers are inside free regions, not walls.
#    - Boundary-distance logic is still valid for extracting palette order, but it
#      must ignore blocker semantics; otherwise blockers would incorrectly become
#      extra palette entries.
#    - The original unconditional `out[r, c] = palette[d % m]` is no longer safe:
#      it would overwrite blocked shadows that should stay base-colored and would
#      erase blocker cells. The new code explicitly separates marker singletons
#      from blocker components and checks directional reachability before painting.
# d) Placement legality
#    Generator regions are still axis-aligned rectangles from recursive wall
#    splits. Blockers are placed strictly inside a region with at least one-cell
#    margin from the walls, never overlapping markers or each other. Markers are
#    placed first on inward diagonals from corners, so their depth ordering remains
#    well defined.
# e) Visual observability
#    The rule does not depend on color ids. Humans can spot that isolated pixels
#    encode the band cycle while solid patches interrupt nearby frames and cast
#    rectangular "shadows" of unfinished fill.
# f) Visual cleanliness
#    Each region contains at most one blocker patch. The patch is compact and the
#    region count remains moderate, so inputs do not become noisy.
# g) Color inference
#    Colors keep their original role: marker colors define the repeating palette by
#    depth order. The blocker color is a dedicated extra color whose only visible
#    effect is to stop growth; it never has to be mapped numerically.


DIR8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _cc4(mask: np.ndarray):
    return components_mask(mask, connectivity=4)


def _mode(arr):
    vals, counts = np.unique(np.asarray(arr), return_counts=True)
    return int(vals[counts.argmax()])


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
                if rr < 0 or rr >= h or cc < 0 or cc >= w:
                    is_boundary = True
                    break
                if not component_mask[rr, cc]:
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


def _classify_region_colors(g: np.ndarray, comp_mask: np.ndarray):
    base = _mode(g[comp_mask])
    marker_cells: list[tuple[int, int, int]] = []
    blocker_mask = np.zeros_like(comp_mask, dtype=bool)

    for color in np.unique(g[comp_mask]):
        color = int(color)
        if color == base:
            continue
        for pts in _cc4(comp_mask & (g == color)):
            if len(pts) == 1:
                r, c = map(int, pts[0])
                marker_cells.append((r, c, color))
            else:
                for r, c in pts:
                    blocker_mask[int(r), int(c)] = True
    return base, marker_cells, blocker_mask


def _build_palette(marker_cells: list[tuple[int, int, int]], base: int, dist: np.ndarray) -> list[int]:
    dist_to_color: dict[int, int] = {}
    for r, c, v in sorted(marker_cells, key=lambda x: (int(dist[x[0], x[1]]), x[0], x[1])):
        d = int(dist[r, c])
        if d >= 0 and d not in dist_to_color:
            dist_to_color[d] = int(v)

    if not dist_to_color:
        return [int(base)]

    ds = sorted(dist_to_color)
    palette: list[int] = []
    if ds[0] > 0:
        palette.append(int(base))
    for d in ds:
        palette.append(int(dist_to_color[d]))
    return palette


def _paint_region(
    out: np.ndarray,
    comp_mask: np.ndarray,
    base: int,
    palette: list[int],
    blocker_mask: np.ndarray,
    *,
    collision_stop: bool,
) -> None:
    rows, cols = np.where(comp_mask)
    r0, r1 = int(rows.min()), int(rows.max())
    c0, c1 = int(cols.min()), int(cols.max())

    local_comp = comp_mask[r0 : r1 + 1, c0 : c1 + 1]
    local_block = blocker_mask[r0 : r1 + 1, c0 : c1 + 1]
    solid = (~local_comp) | local_block

    bh, bw = solid.shape
    row_pref = np.zeros((bh, bw + 1), dtype=int)
    row_pref[:, 1:] = np.cumsum(solid.astype(int), axis=1)
    col_pref = np.zeros((bh + 1, bw), dtype=int)
    col_pref[1:, :] = np.cumsum(solid.astype(int), axis=0)

    m = len(palette)
    for r, c in zip(rows, cols):
        r = int(r)
        c = int(c)
        if blocker_mask[r, c]:
            continue

        lr, lc = r - r0, c - c0
        if not collision_stop:
            depth = min(lr, bh - 1 - lr, lc, bw - 1 - lc)
            out[r, c] = int(palette[depth % m])
            continue

        top_clear = col_pref[lr, lc] == 0
        bottom_clear = (col_pref[bh, lc] - col_pref[lr + 1, lc]) == 0
        left_clear = row_pref[lr, lc] == 0
        right_clear = (row_pref[lr, bw] - row_pref[lr, lc + 1]) == 0

        depths: list[int] = []
        if top_clear:
            depths.append(lr)
        if bottom_clear:
            depths.append(bh - 1 - lr)
        if left_clear:
            depths.append(lc)
        if right_clear:
            depths.append(bw - 1 - lc)

        if depths:
            out[r, c] = int(palette[min(depths) % m])
        else:
            out[r, c] = int(base)


def _solve(grid: np.ndarray, *, collision_stop: bool) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    bg = _mode(g)
    other = g[g != bg]
    wall = bg if other.size == 0 else _mode(other)

    out = g.copy()
    free = g != wall

    for pts in _cc4(free):
        comp_mask = np.zeros((h, w), dtype=bool)
        for r, c in pts:
            comp_mask[int(r), int(c)] = True

        base, marker_cells, blocker_mask = _classify_region_colors(g, comp_mask)
        dist = _boundary_distance(comp_mask)
        palette = _build_palette(marker_cells, base, dist)
        _paint_region(out, comp_mask, base, palette, blocker_mask, collision_stop=collision_stop)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill wall-partitioned regions with palette bands that stop when they hit blocker patches.

    Input structure:
    - The grid is split into rectangular free regions by a dominant wall color.
    - Inside each region, isolated single-pixel markers of non-base colors still encode a palette by boundary depth.
    - Some regions also contain compact multi-pixel obstacle patches of another color family.

    Rule:
    - For each region, sort the singleton marker colors by Chebyshev distance from that region's outer boundary.
    - If the shallowest marker is not on the boundary, prepend the base color to the palette as in the original task.
    - Repaint the region with the same rectangular band sequence, but only along straight inward growth from the four sides; once a side's growth line encounters a blocker patch, it cannot color cells behind that obstacle.
    - Cells shadowed from all four sides remain base-colored, and blocker patches themselves stay unchanged.

    Key insight: Marker pixels still define the cyclic color order, but multi-cell patches act as collision stops that locally truncate the otherwise full concentric frames.
    """
    return _solve(grid, collision_stop=True)


# --------------- generator ---------------

# ---- Canvas bounds ----
H_MIN, H_MAX = 10, 30
W_MIN, W_MAX = 13, 30

# ---- Wall subdivision ----
N_SPLITS_MIN, N_SPLITS_MAX = 2, 5
MIN_REGION_DIM = 3
SPLITTABLE_LEN = 2 * MIN_REGION_DIM + 1

# ---- Region count filter ----
N_REGIONS_MIN, N_REGIONS_MAX = 3, 6

# ---- Markers per region ----
MARKERS_PER_REGION_MAX = 3
MARKER_COUNT_WEIGHTS = [1, 2, 3, 3]
START_DIST_MAX = 1

# ---- Blockers ----
MIN_BLOCKER_REGIONS = 1
MAX_BLOCKER_REGIONS = 3
BLOCKER_SHAPES = [(2, 2), (2, 3), (3, 2), (2, 4), (4, 2), (3, 3)]

# ---- Distinct marker colors across grid ----
DISTINCT_MARKER_COLORS_MIN, DISTINCT_MARKER_COLORS_MAX = 3, 7

# ---- Retry budget ----
GEN_MAX_ATTEMPTS = 400

_CORNERS = [(0, 0, 1, 1), (0, 1, 1, -1), (1, 0, -1, 1), (1, 1, -1, -1)]


def _max_cheb_depth(h: int, w: int) -> int:
    return (min(h, w) - 1) // 2


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
            wall_mask[rr, c0 : c1 + 1] = 1
            rects.append((r0, c0, rr - 1, c1))
            rects.append((rr + 1, c0, r1, c1))
        else:
            cc = rng.randint(c0 + MIN_REGION_DIM, c1 - MIN_REGION_DIM)
            wall_mask[r0 : r1 + 1, cc] = 1
            rects.append((r0, c0, r1, cc - 1))
            rects.append((r0, cc + 1, r1, c1))

    return wall_mask, rects


def _place_markers(rect, marker_pool: list[int], rng: random.Random):
    r0, c0, r1, c1 = rect
    max_d = _max_cheb_depth(r1 - r0 + 1, c1 - c0 + 1)
    upper = min(MARKERS_PER_REGION_MAX, max_d)
    if upper <= 0:
        return []

    n = rng.choices(range(len(MARKER_COUNT_WEIGHTS)), weights=MARKER_COUNT_WEIGHTS)[0]
    n = min(n, upper)
    if n == 0:
        return []

    start_d = rng.randint(0, min(START_DIST_MAX, max_d - n))
    row_from, col_from, dr, dc = rng.choice(_CORNERS)
    br = r0 if row_from == 0 else r1
    bc = c0 if col_from == 0 else c1
    colors = rng.sample(marker_pool, n)
    return [(start_d + k, col, br + dr * (start_d + k), bc + dc * (start_d + k)) for k, col in enumerate(colors)]


def _pick_colors(rng: random.Random):
    colors = list(range(10))
    bg = rng.choice(colors)
    colors.remove(bg)
    wall = rng.choice(colors)
    colors.remove(wall)
    blocker = rng.choice(colors)
    colors.remove(blocker)
    return bg, wall, blocker, colors


def _stamp_markers(grid, rects, bg, marker_pool, rng):
    used = set()
    markers_by_rect: dict[tuple[int, int, int, int], list[tuple[int, int, int, int]]] = {}
    for rect in rects:
        markers = _place_markers(rect, marker_pool, rng)
        markers_by_rect[rect] = markers
        for _d, col, rr, cc in markers:
            if not (rect[0] <= rr <= rect[2] and rect[1] <= cc <= rect[3]):
                return None, None
            if grid[rr, cc] != bg:
                return None, None
            grid[rr, cc] = col
            used.add(col)
    return used, markers_by_rect


def _blocker_candidates(rect):
    r0, c0, r1, c1 = rect
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    out = []
    for bh, bw in BLOCKER_SHAPES:
        if bh >= h - 1 or bw >= w - 1:
            continue
        for anchor in ("top", "bottom", "left", "right"):
            if anchor == "top":
                rr_options = [r0 + 1, r0 + 2] if r0 + bh + 1 <= r1 else [r0 + 1]
                cc_lo, cc_hi = c0 + 1, c1 - bw
                if cc_lo > cc_hi:
                    continue
                for rr in rr_options:
                    if rr + bh - 1 >= r1:
                        continue
                    for cc in range(cc_lo, cc_hi + 1):
                        out.append((rr, cc, rr + bh - 1, cc + bw - 1))
            elif anchor == "bottom":
                rr_options = [r1 - bh, r1 - bh - 1] if r1 - bh - 1 >= r0 + 1 else [r1 - bh]
                cc_lo, cc_hi = c0 + 1, c1 - bw
                if cc_lo > cc_hi:
                    continue
                for rr in rr_options:
                    if rr < r0 + 1:
                        continue
                    for cc in range(cc_lo, cc_hi + 1):
                        out.append((rr, cc, rr + bh - 1, cc + bw - 1))
            elif anchor == "left":
                cc_options = [c0 + 1, c0 + 2] if c0 + bw + 1 <= c1 else [c0 + 1]
                rr_lo, rr_hi = r0 + 1, r1 - bh
                if rr_lo > rr_hi:
                    continue
                for cc in cc_options:
                    if cc + bw - 1 >= c1:
                        continue
                    for rr in range(rr_lo, rr_hi + 1):
                        out.append((rr, cc, rr + bh - 1, cc + bw - 1))
            else:
                cc_options = [c1 - bw, c1 - bw - 1] if c1 - bw - 1 >= c0 + 1 else [c1 - bw]
                rr_lo, rr_hi = r0 + 1, r1 - bh
                if rr_lo > rr_hi:
                    continue
                for cc in cc_options:
                    if cc < c0 + 1:
                        continue
                    for rr in range(rr_lo, rr_hi + 1):
                        out.append((rr, cc, rr + bh - 1, cc + bw - 1))
    return out


def _stamp_blockers(grid, rects, bg, blocker, markers_by_rect, rng):
    blocker_regions = [rect for rect in rects if (rect[2] - rect[0] + 1) >= 5 and (rect[3] - rect[1] + 1) >= 5]
    if not blocker_regions:
        return 0

    rng.shuffle(blocker_regions)
    target = rng.randint(MIN_BLOCKER_REGIONS, min(MAX_BLOCKER_REGIONS, len(blocker_regions)))
    placed = 0
    for rect in blocker_regions:
        if placed >= target:
            break
        candidates = _blocker_candidates(rect)
        rng.shuffle(candidates)
        markers = {(rr, cc) for _d, _col, rr, cc in markers_by_rect[rect]}
        for rr0, cc0, rr1, cc1 in candidates:
            ok = True
            for rr in range(rr0, rr1 + 1):
                for cc in range(cc0, cc1 + 1):
                    if grid[rr, cc] != bg or (rr, cc) in markers:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue

            # Keep blockers off the marker's direct corner diagonal so the palette
            # evidence remains visible near the boundary.
            too_close = False
            for _d, _col, mr, mc in markers_by_rect[rect]:
                if rr0 - 1 <= mr <= rr1 + 1 and cc0 - 1 <= mc <= cc1 + 1:
                    too_close = True
                    break
            if too_close:
                continue

            grid[rr0 : rr1 + 1, cc0 : cc1 + 1] = blocker
            placed += 1
            break
    return placed


def _validate_color_order(grid, bg, wall):
    vals, counts = np.unique(grid, return_counts=True)
    order = np.argsort(-counts)
    if int(vals[order[0]]) != bg:
        return False
    return len(order) >= 2 and int(vals[order[1]]) == wall


def _variant_metrics(grid: np.ndarray):
    out_stop = _solve(grid, collision_stop=True)
    out_full = _solve(grid, collision_stop=False)

    g = np.asarray(grid, dtype=int)
    bg = _mode(g)
    other = g[g != bg]
    wall = bg if other.size == 0 else _mode(other)

    free = g != wall
    h, w = g.shape
    diff = 0
    blocker_cells = 0
    active_regions = 0

    for pts in _cc4(free):
        comp_mask = np.zeros((h, w), dtype=bool)
        for r, c in pts:
            comp_mask[int(r), int(c)] = True
        _base, _markers, blocker_mask = _classify_region_colors(g, comp_mask)
        blocker_cells += int(blocker_mask.sum())
        region_diff = 0
        for r, c in pts:
            r = int(r)
            c = int(c)
            if blocker_mask[r, c]:
                continue
            if int(out_stop[r, c]) != int(out_full[r, c]):
                diff += 1
                region_diff += 1
        if blocker_mask.any() and region_diff > 0:
            active_regions += 1
    return {
        "stop_diff_cells": diff,
        "blocker_cells": blocker_cells,
        "active_regions": active_regions,
        "changed_from_input": int(np.count_nonzero(out_stop != g)),
    }


def _gen_attempt(rng: random.Random):
    H = rng.randint(H_MIN, H_MAX)
    W = rng.randint(W_MIN, W_MAX)

    bg, wall, blocker, marker_pool = _pick_colors(rng)
    wall_mask, rects = _place_walls(H, W, rng)
    if not (N_REGIONS_MIN <= len(rects) <= N_REGIONS_MAX):
        return None

    grid = np.full((H, W), bg, dtype=int)
    grid[wall_mask == 1] = wall

    used, markers_by_rect = _stamp_markers(grid, rects, bg, marker_pool, rng)
    if used is None:
        return None
    if not (DISTINCT_MARKER_COLORS_MIN <= len(used) <= DISTINCT_MARKER_COLORS_MAX):
        return None

    placed_blockers = _stamp_blockers(grid, rects, bg, blocker, markers_by_rect, rng)
    if placed_blockers < MIN_BLOCKER_REGIONS:
        return None
    if not _validate_color_order(grid, bg, wall):
        return None

    try:
        out = main(grid)
        metrics = _variant_metrics(grid)
    except Exception:
        return None

    if np.array_equal(grid, out):
        return None
    if metrics["blocker_cells"] < 4:
        return None
    if metrics["active_regions"] < 1:
        return None
    if metrics["stop_diff_cells"] < 8:
        return None
    if metrics["changed_from_input"] < 12:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_MAX_ATTEMPTS):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError(f"generate_input failed after {GEN_MAX_ATTEMPTS} attempts")
