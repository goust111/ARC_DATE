# Combo variant of 13e47133 (index 22)
# combo_spec: D2/3+D11/0
# Name: 颜色配对填充+统一标记数量

import random

import numpy as np

from arc_utils import components_mask

# Design analysis for the fused variant:
# - Concrete behavior:
#   In the original task, a region with two markers A@d=0 and B@d=1 uses palette
#   [A, B], so depths 0,1,2,3 become A,B,A,B.  Here the same two-marker region
#   uses color-paired chunks: marker A contributes [A,B] and marker B contributes
#   [B,A], hence palette [A,B,B,A] and depths 0,1,2,3 become A,B,B,A.  The third
#   ring is therefore the first unavoidable visual divergence from the source task.
# - Complexity calibration:
#   I fix every region to exactly two markers.  That keeps the pairing rule
#   unambiguous for humans: the complementary color of each marker is simply the
#   other marker in the same region.  Allowing 3+ markers would need a harder
#   pairing tie-break and would weaken learnability across 2-4 ARC examples.
# - Implicit assumptions audit:
#   main() still infers bg as the mode and wall as the second mode, so the
#   generator preserves those frequencies.  main() assumes marker distances can
#   be ordered and that the intended two-layer pairing is visible, so the
#   generator places exactly two markers at consecutive depths, using either
#   d=0,1 or d=1,2 for the whole puzzle, and forces every region to reach the
#   first depth where the paired palette diverges from the original cycle.  This
#   avoids degeneration back to the source behavior on shallow regions.
# - Placement legality:
#   Walls are built by recursive full-row/full-column cuts, which keeps regions
#   rectangular and 4-connected.  Markers are stamped on an inward diagonal from a
#   chosen corner, so the selected consecutive depths stay inside the owning
#   rectangle and never overlap walls or each other.
# - Visual observability / neatness:
#   Each region contains only base cells plus two isolated markers in the input,
#   and only the paired marker colors in the output.  Because every region repeats
#   the same "two markers -> four-step paired cycle" motif, the cross-region
#   consistency from D11/0 directly supports discovering the D2/3 pairing rule.

DIR8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _cc4(mask):
    return components_mask(mask, connectivity=4)


def _mode(arr):
    vals, counts = np.unique(np.asarray(arr), return_counts=True)
    return int(vals[counts.argmax()])


def _boundary_distance(component_mask: np.ndarray) -> np.ndarray:
    """Chebyshev distance to the component boundary for every True cell."""
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


def _ordered_markers(g: np.ndarray, pts, base: int, dist: np.ndarray) -> list[tuple[int, int]]:
    """Return at most one marker color per depth, ordered by (depth, row, col)."""
    raw: list[tuple[int, int, int, int]] = []
    for r, c in pts:
        color = int(g[r, c])
        if color == base:
            continue
        d = int(dist[r, c])
        if d < 0:
            continue
        raw.append((d, int(r), int(c), color))

    raw.sort()
    markers: list[tuple[int, int]] = []
    seen_depths = set()
    for d, _r, _c, color in raw:
        if d in seen_depths:
            continue
        seen_depths.add(d)
        markers.append((d, color))
    return markers


def _paired_palette(base: int, markers: list[tuple[int, int]]) -> list[int]:
    """Expand depth-ordered markers into paired two-layer chunks."""
    if not markers:
        return [base]

    palette: list[int] = []
    if markers[0][0] > 0:
        palette.append(base)

    colors = [color for _d, color in markers]
    if len(colors) == 1:
        complements = [colors[0]]
    elif len(colors) == 2:
        complements = [colors[1], colors[0]]
    else:
        # Fallback for unexpected inputs: pair with the nearest other marker in
        # depth order, then reading-order tie-breaks through the sorted list.
        depths = [d for d, _c in markers]
        complements = []
        for i, depth in enumerate(depths):
            best_j = min(
                (abs(depth - depths[j]), abs(i - j), j)
                for j in range(len(depths))
                if j != i
            )[2]
            complements.append(colors[best_j])

    for color, partner in zip(colors, complements):
        palette.extend([color, partner])
    return palette


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill wall-separated regions with paired concentric bands derived from exactly repeated local marker pairs.

    Input structure:
    - The grid is split by one wall color into several 4-connected non-wall regions.
    - Inside each region, one dominant base color covers almost all cells.
    - Every region contains the same number of isolated markers; in this fused generator that shared count is exactly two, placed at consecutive boundary depths.

    Rule:
    - For each region, sort its markers by Chebyshev distance from the region boundary.
    - Build a cyclic palette in two-layer chunks: each marker contributes its own color for one layer and the paired marker color for the next layer.
    - With two markers A then B, the palette chunk is [A, B, B, A]; if the first marker is not on the boundary, prepend the base color once.
    - Paint every non-wall cell by indexing that palette with the cell's boundary distance modulo the palette length.

    Key insight:
    The global "same marker count in every region" constraint makes the pairing rule stable across the whole grid: each region presents the same two-marker template, so the output reveals that every marker generates a self-color ring followed by a partner-color ring instead of the original one-color-per-depth cycle.
    """
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
            comp_mask[r, c] = True

        base = _mode(g[comp_mask])
        dist = _boundary_distance(comp_mask)
        markers = _ordered_markers(g, pts, base, dist)
        palette = _paired_palette(base, markers)
        period = len(palette)

        for r, c in pts:
            out[r, c] = palette[int(dist[r, c]) % period]

    return out


# --------------- generator ---------------

H_MIN, H_MAX = 16, 30
W_MIN, W_MAX = 16, 30

N_SPLITS_MIN, N_SPLITS_MAX = 2, 5
MIN_REGION_DIM_BASE = 5

N_REGIONS_MIN, N_REGIONS_MAX = 3, 6
MARKERS_PER_REGION = 2

DISTINCT_MARKER_COLORS_MIN, DISTINCT_MARKER_COLORS_MAX = 3, 7
ACTIVE_MARKER_COLORS_MIN, ACTIVE_MARKER_COLORS_MAX = 3, 5

GEN_MAX_ATTEMPTS = 400

_CORNERS = [(0, 0, 1, 1), (0, 1, 1, -1), (1, 0, -1, 1), (1, 1, -1, -1)]


def _max_cheb_depth(h: int, w: int) -> int:
    return (min(h, w) - 1) // 2


def _min_region_dim(start_depth: int) -> int:
    return MIN_REGION_DIM_BASE + 2 * start_depth


def _place_walls(H: int, W: int, min_region_dim: int, rng: random.Random):
    wall_mask = np.zeros((H, W), dtype=int)
    rects = [(0, 0, H - 1, W - 1)]
    n_splits = rng.randint(N_SPLITS_MIN, N_SPLITS_MAX)
    splittable_len = 2 * min_region_dim + 1

    for _ in range(n_splits):
        candidates = []
        for i, (r0, c0, r1, c1) in enumerate(rects):
            can_h = (r1 - r0 + 1) >= splittable_len
            can_v = (c1 - c0 + 1) >= splittable_len
            if can_h or can_v:
                candidates.append((i, can_h, can_v))
        if not candidates:
            break

        i, can_h, can_v = rng.choice(candidates)
        r0, c0, r1, c1 = rects.pop(i)
        axes = ('h',) * can_h + ('v',) * can_v
        axis = rng.choice(axes)

        if axis == 'h':
            rr = rng.randint(r0 + min_region_dim, r1 - min_region_dim)
            wall_mask[rr, c0:c1 + 1] = 1
            rects.append((r0, c0, rr - 1, c1))
            rects.append((rr + 1, c0, r1, c1))
        else:
            cc = rng.randint(c0 + min_region_dim, c1 - min_region_dim)
            wall_mask[r0:r1 + 1, cc] = 1
            rects.append((r0, c0, r1, cc - 1))
            rects.append((r0, cc + 1, r1, c1))

    return wall_mask, rects


def _pick_colors(rng: random.Random):
    colors = list(range(10))
    bg = rng.choice(colors)
    colors.remove(bg)
    wall = rng.choice(colors)
    colors.remove(wall)
    k = rng.randint(ACTIVE_MARKER_COLORS_MIN, ACTIVE_MARKER_COLORS_MAX)
    active = rng.sample(colors, k)
    return bg, wall, active


def _marker_positions(rect, start_depth: int, rng: random.Random):
    r0, c0, r1, c1 = rect
    needed_depth = start_depth + MARKERS_PER_REGION
    if _max_cheb_depth(r1 - r0 + 1, c1 - c0 + 1) < needed_depth:
        return None

    row_from, col_from, dr, dc = rng.choice(_CORNERS)
    br = r0 if row_from == 0 else r1
    bc = c0 if col_from == 0 else c1
    return [
        (start_depth + idx, br + (start_depth + idx) * dr, bc + (start_depth + idx) * dc)
        for idx in range(MARKERS_PER_REGION)
    ]


def _stamp_markers(
    grid,
    rects,
    bg: int,
    active_colors: list[int],
    start_depth: int,
    rng: random.Random,
):
    used = set()
    marker_counts = []

    for rect in rects:
        positions = _marker_positions(rect, start_depth, rng)
        if positions is None:
            return None

        colors = rng.sample(active_colors, MARKERS_PER_REGION)
        marker_counts.append(len(colors))

        for (_d, rr, cc), color in zip(positions, colors):
            if not (rect[0] <= rr <= rect[2] and rect[1] <= cc <= rect[3]):
                return None
            if grid[rr, cc] != bg:
                return None
            grid[rr, cc] = color
            used.add(color)

    return used, marker_counts


def _validate_color_order(grid: np.ndarray, bg: int, wall: int) -> bool:
    vals, counts = np.unique(grid, return_counts=True)
    order = np.argsort(-counts)
    if int(vals[order[0]]) != bg:
        return False
    return len(order) >= 2 and int(vals[order[1]]) == wall


def _gen_attempt(rng: random.Random):
    start_depth = rng.randint(0, 1)
    min_region_dim = _min_region_dim(start_depth)
    min_h = max(H_MIN, 21 if start_depth == 1 else H_MIN)
    min_w = max(W_MIN, 21 if start_depth == 1 else W_MIN)
    H = rng.randint(min_h, H_MAX)
    W = rng.randint(min_w, W_MAX)
    bg, wall, active_colors = _pick_colors(rng)

    wall_mask, rects = _place_walls(H, W, min_region_dim, rng)
    if not (N_REGIONS_MIN <= len(rects) <= N_REGIONS_MAX):
        return None

    visible_divergence_depth = start_depth + MARKERS_PER_REGION
    if any(
        _max_cheb_depth(r1 - r0 + 1, c1 - c0 + 1) < visible_divergence_depth
        for r0, c0, r1, c1 in rects
    ):
        return None

    grid = np.full((H, W), bg, dtype=int)
    grid[wall_mask == 1] = wall

    stamped = _stamp_markers(grid, rects, bg, active_colors, start_depth, rng)
    if stamped is None:
        return None
    used, marker_counts = stamped

    if len(set(marker_counts)) != 1 or marker_counts[0] != MARKERS_PER_REGION:
        return None
    if not (DISTINCT_MARKER_COLORS_MIN <= len(used) <= DISTINCT_MARKER_COLORS_MAX):
        return None
    if not _validate_color_order(grid, bg, wall):
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(grid, out):
        return None

    # Ensure the fusion is visibly different from the source behavior.  With
    # start_depth 0 the first divergence is at depth 2; with start_depth 1 the
    # prepended base ring pushes the first divergence to depth 3.
    free = grid != wall
    for pts in _cc4(free):
        comp_mask = np.zeros((H, W), dtype=bool)
        for r, c in pts:
            comp_mask[r, c] = True
        dist = _boundary_distance(comp_mask)
        if not np.any(dist == visible_divergence_depth):
            return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_MAX_ATTEMPTS} attempts")
