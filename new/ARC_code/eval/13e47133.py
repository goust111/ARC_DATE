import numpy as np

from arc_utils import components_mask
import random

DIR8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

def _cc4(mask):
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill partitioned regions with concentric colored bands derived from marker pixels.

    Input structure:
    - Grid partitioned into regions by wall-colored lines.
    - Each region has a dominant base color and a few isolated marker pixels of other colors at varying depths from the region boundary.

    Rule:
    - For each region, read marker colors ordered by their Chebyshev distance from the boundary to build a color palette.
    - If no marker sits on the boundary itself, the base color leads the palette.
    - Paint every cell with palette[distance mod palette_length], producing repeating concentric rectangular frames that radiate inward.

    Key insight: Each marker's depth from the boundary defines its position in the cyclic palette; the pattern then tiles inward automatically.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    # Global background and wall color (the main partition lines).
    bg = _mode(g)
    other = g[g != bg]
    wall = bg if other.size == 0 else _mode(other)

    out = g.copy()
    free = g != wall
    comps = _cc4(free)

    for pts in comps:
        # Build component mask
        comp_mask = np.zeros((h, w), dtype=bool)
        for r, c in pts:
            comp_mask[r, c] = True

        vals = g[comp_mask]
        base = _mode(vals)
        dist = _boundary_distance(comp_mask)

        # Markers are cells whose color differs from the region base.
        # Pick at most one marker color per distance (closest in reading order).
        dist_to_color: dict[int, int] = {}
        for r, c in pts:
            v = int(g[r, c])
            if v == base:
                continue
            d = int(dist[r, c])
            if d < 0:
                continue
            if d not in dist_to_color:
                dist_to_color[d] = v
            else:
                # stable tie-breaker: prefer earlier (r,c)
                pass

        palette: list[int] = []
        if dist_to_color:
            ds = sorted(dist_to_color.keys())
            if ds[0] > 0:
                palette.append(int(base))
            for d in ds:
                palette.append(int(dist_to_color[d]))
        else:
            palette = [int(base)]

        m = len(palette)
        for r, c in pts:
            d = int(dist[r, c])
            out[r, c] = palette[d % m]

    return out

# --------------- generator (attempt 1 (Claude)) ---------------

# ---- Canvas bounds ----
H_MIN, H_MAX = 10, 30                       # grid height range
W_MIN, W_MAX = 13, 30                       # grid width range

# ---- Wall subdivision ----
N_SPLITS_MIN, N_SPLITS_MAX = 2, 5           # recursive subdivision rounds
MIN_REGION_DIM = 3                          # each half must keep >= this min dim
# Threshold for splitting: need MIN_REGION_DIM + 1 (wall) + MIN_REGION_DIM
SPLITTABLE_LEN = 2 * MIN_REGION_DIM + 1     # 7

# ---- Region count filter ----
N_REGIONS_MIN, N_REGIONS_MAX = 3, 6

# ---- Markers per region ----
MARKERS_PER_REGION_MAX = 3                  # cap on markers placed per region
MARKER_COUNT_WEIGHTS = [1, 2, 3, 3]         # weights for choosing 0/1/2/3 markers
START_DIST_MAX = 1                          # start distance cap (0 or 1)

# ---- Distinct marker colors across grid ----
DISTINCT_MARKER_COLORS_MIN, DISTINCT_MARKER_COLORS_MAX = 3, 9

# ---- Retry budget ----
GEN_MAX_ATTEMPTS = 300

# Corner -> (diagonal step dr, dc, base row, base col) as a function of bbox.
# Encoded as (row_from, col_from, dr, dc): 0=r0, 1=r1 for row_from; similarly col.
_CORNERS = [(0, 0, 1, 1), (0, 1, 1, -1), (1, 0, -1, 1), (1, 1, -1, -1)]


def _max_cheb_depth(h: int, w: int) -> int:
    """Max Chebyshev depth from boundary in an h x w rectangle."""
    return (min(h, w) - 1) // 2


def _place_walls(H: int, W: int, rng: random.Random):
    """
    Build a wall mask by recursive rectangular subdivision.

    Returns (wall_mask, region_bboxes). wall_mask[r, c] == 1 on wall cells.
    Each bbox = (r0, c0, r1, c1) inclusive for a 4-connected free region.
    """
    wall_mask = np.zeros((H, W), dtype=int)
    rects = [(0, 0, H - 1, W - 1)]
    n_splits = rng.randint(N_SPLITS_MIN, N_SPLITS_MAX)

    for _ in range(n_splits):
        # Find rects large enough to split on at least one axis.
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
        axes = ('h',) * can_h + ('v',) * can_v
        axis = rng.choice(axes)

        if axis == 'h':
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


def _place_markers(rect, marker_pool: list, rng: random.Random):
    """
    Produce 0..MARKERS_PER_REGION_MAX markers along a diagonal from one bbox corner.
    Each marker = (dist, color, r, c) with consecutive Chebyshev distances.
    """
    r0, c0, r1, c1 = rect
    max_d = _max_cheb_depth(r1 - r0 + 1, c1 - c0 + 1)
    upper = min(MARKERS_PER_REGION_MAX, max_d)
    if upper <= 0:
        return []

    n = rng.choices(range(len(MARKER_COUNT_WEIGHTS)), weights=MARKER_COUNT_WEIGHTS)[0]
    n = min(n, upper)
    if n == 0:
        return []

    # start_d in [0, min(START_DIST_MAX, max_d - n)]; controls whether base leads palette.
    start_d = rng.randint(0, min(START_DIST_MAX, max_d - n))

    row_from, col_from, dr, dc = rng.choice(_CORNERS)
    br = r0 if row_from == 0 else r1
    bc = c0 if col_from == 0 else c1

    colors = rng.sample(marker_pool, n)
    return [(start_d + k, col, br + dr * (start_d + k), bc + dc * (start_d + k))
            for k, col in enumerate(colors)]


def _pick_palette(rng: random.Random):
    """Pick bg, wall, and the remaining 8-color marker pool (all distinct)."""
    colors = list(range(10))
    bg = rng.choice(colors)
    colors.remove(bg)
    wall = rng.choice(colors)
    colors.remove(wall)
    return bg, wall, colors


def _stamp_markers(grid, rects, bg, marker_pool, rng):
    """
    Place markers into grid (in place). Returns the set of colors used, or
    None if any marker would land outside its rect or overwrite non-bg.
    """
    used = set()
    for rect in rects:
        for _d, col, rr, cc in _place_markers(rect, marker_pool, rng):
            if not (rect[0] <= rr <= rect[2] and rect[1] <= cc <= rect[3]):
                return None
            if grid[rr, cc] != bg:
                return None
            grid[rr, cc] = col
            used.add(col)
    return used


def _validate_color_order(grid, bg, wall):
    """Confirm bg is mode and wall is second mode of the assembled grid."""
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

    used = _stamp_markers(grid, rects, bg, marker_pool, rng)
    if used is None:
        return None
    if not (DISTINCT_MARKER_COLORS_MIN <= len(used) <= DISTINCT_MARKER_COLORS_MAX):
        return None
    if not _validate_color_order(grid, bg, wall):
        return None

    # Final oracle: solver must produce a non-trivial transformation.
    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(grid, out):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_MAX_ATTEMPTS):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError(f"generate_input failed after {GEN_MAX_ATTEMPTS} attempts")
