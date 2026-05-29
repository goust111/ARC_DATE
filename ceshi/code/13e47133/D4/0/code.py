# Variant of 13e47133
# Dimension: D4 — 终止条件：填充的终点 (dst)
# Change: 填充直到整个区域被完全覆盖（无显式终止条件）... → 填充到指定深度后停止

"""
Design analysis for the D4 variant.

a) Concretization and behavior trace
The original task partitions the grid with one wall color, then treats every
non-wall component as a region whose marker pixels define a cyclic palette by
Chebyshev depth from the boundary. The old output paints every region all the
way to the center with palette[depth mod len(palette)].

This variant keeps the same objects and the same palette construction, but
changes only the stopping condition. If a region contains N marker pixels, then
only layers with boundary-distance d <= N are painted. Cells deeper than N are
reset to the region base color, so the region keeps an untouched inner core.

Concrete trace on a 9x9 rectangular region:
- Input: base = 1, markers at d=1 color 2 and d=2 color 3, so N=2.
- Original palette: [1,2,3], full fill gives depths 0..4 as 1,2,3,1,2.
- Variant palette is the same, but only depths 0,1,2 are painted; depths 3,4
  stay 1. The center no longer receives the repeated color 2.
- Difference from the original appears exactly at the inner layers beyond the
  marker-count cap.

b) Complexity calibration
The new hidden variable is a small integer count, not an arbitrary mapping.
Counts stay in 1..3 for marked regions so the rule is inferable, and multiple
regions per grid repeat these counts. The generator avoids trivial cases where
the cap would coincide with the full-fill result, so the visual effect remains
observable.

c) Implicit-assumption audit
Relevant original assumptions and required changes:
- `palette` construction still works: marker order is still derived from depth.
- The old unconditional `out[r,c] = palette[d % m]` relied on "fill forever".
  Under D4 this becomes wrong, because deeper cells must revert to base.
- `out = g.copy()` is no longer sufficient for untouched interior cells, since
  leaving them as copied input could preserve marker pixels by accident. The
  new solver assigns every region cell explicitly: palette color for d <= N,
  base otherwise.
- The old generator only needed markers to fit inside the region. The new
  generator must additionally ensure the cap is visible, especially when the
  palette begins with base. It now bounds marker count against region depth and
  rejects any sample whose output matches the old full-fill behavior.

d) Placement legality
Walls are still axis-aligned full-row/full-column splits, so every free region
remains a valid 4-connected rectangle. Markers are placed on one diagonal from
a corner at consecutive depths. The generator only accepts placements entirely
inside the owning rectangle, on background cells, with distinct marker colors
within that region.

e) Visual observability
The stop rule is visible because marked regions are generated only when there is
enough remaining depth for an inner base-colored hole to survive. The count is
read from visible marker pixels, not from color ids or any hidden ordering.

f) Visual cleanliness
There is still only one element type per region interior: isolated marker pixels
in the input and concentric bands in the output. The untouched center is simply
the original base color, so no extra overlay or clutter is introduced.

g) Color-change inferability
Colors keep their original role. Marker colors only define palette order by
visible depth, and the new stop condition depends only on how many markers are
present. No color-id arithmetic or invisible ranking is used.
"""

import random

import numpy as np

from arc_utils import components_mask


DIR8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
_CORNERS = [(0, 0, 1, 1), (0, 1, 1, -1), (1, 0, -1, 1), (1, 1, -1, -1)]


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


def _transform(grid: np.ndarray, *, capped: bool) -> np.ndarray:
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

        vals = g[comp_mask]
        base = _mode(vals)
        dist = _boundary_distance(comp_mask)

        dist_to_color: dict[int, int] = {}
        marker_count = 0
        for r, c in pts:
            v = int(g[r, c])
            if v == base:
                continue
            marker_count += 1
            d = int(dist[r, c])
            if d >= 0 and d not in dist_to_color:
                dist_to_color[d] = v

        if dist_to_color:
            ds = sorted(dist_to_color)
            palette: list[int] = []
            if ds[0] > 0:
                palette.append(int(base))
            for d in ds:
                palette.append(int(dist_to_color[d]))
        else:
            palette = [int(base)]

        fill_depth = marker_count
        m = len(palette)
        for r, c in pts:
            d = int(dist[r, c])
            if (not capped) or d <= fill_depth:
                out[r, c] = palette[d % m]
            else:
                out[r, c] = base

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill partitioned regions with concentric colored bands derived from marker pixels.

    Input structure:
    - Grid is split into non-wall regions by one dominant wall color.
    - Each region has one dominant base color and a small set of isolated marker pixels placed at increasing Chebyshev depths from the region boundary.

    Rule:
    - Read marker colors in boundary-distance order to build the same cyclic palette as in the original task, prepending the base color when the first marker is not on the boundary.
    - Let N be the number of marker pixels in that region. Paint only the cells with distance d <= N using palette[d mod palette_length]; any deeper cells stay the base color.

    Key insight: Marker depths still determine palette order, but marker count now sets the stopping depth, so the colored frames terminate early and leave a base-colored interior.
    """
    return _transform(grid, capped=True)


def _full_fill_main(grid: np.ndarray) -> np.ndarray:
    return _transform(grid, capped=False)


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
START_DIST_MAX = 1

# ---- Distinct marker colors across grid ----
DISTINCT_MARKER_COLORS_MIN, DISTINCT_MARKER_COLORS_MAX = 3, 9

# ---- Retry budget ----
GEN_MAX_ATTEMPTS = 400


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


def _pick_palette(rng: random.Random):
    colors = list(range(10))
    bg = rng.choice(colors)
    colors.remove(bg)
    wall = rng.choice(colors)
    colors.remove(wall)
    return bg, wall, colors


def _marker_options(max_d: int):
    options = []
    for start_d in range(min(START_DIST_MAX, max_d) + 1):
        if start_d == 0:
            max_n = min(MARKERS_PER_REGION_MAX, max_d - 1)
        else:
            max_n = min(MARKERS_PER_REGION_MAX, max_d - 2)
        for n in range(1, max_n + 1):
            if start_d + n <= max_d:
                options.append((start_d, n))
    return options


def _place_markers(rect, marker_pool: list[int], rng: random.Random):
    """
    Produce markers along one corner diagonal.

    Every returned option guarantees the capped solver differs from the old
    full-fill solver inside this rectangle: marker count stays strictly below
    the region depth, and base-led palettes get enough extra depth for a visible
    divergence.
    """
    r0, c0, r1, c1 = rect
    max_d = _max_cheb_depth(r1 - r0 + 1, c1 - c0 + 1)
    options = _marker_options(max_d)
    if not options:
        return []

    start_d, n = rng.choice(options)
    row_from, col_from, dr, dc = rng.choice(_CORNERS)
    br = r0 if row_from == 0 else r1
    bc = c0 if col_from == 0 else c1
    colors = rng.sample(marker_pool, n)
    return [(start_d + k, col, br + dr * (start_d + k), bc + dc * (start_d + k)) for k, col in enumerate(colors)]


def _stamp_markers(grid, rects, bg, marker_pool, rng):
    """
    Place markers into grid (in place).

    Returns (used_colors, marker_counts_per_region) or None if a placement would
    be invalid.
    """
    used = set()
    counts: list[int] = []
    for rect in rects:
        markers = _place_markers(rect, marker_pool, rng)
        counts.append(len(markers))
        for _d, col, rr, cc in markers:
            if not (rect[0] <= rr <= rect[2] and rect[1] <= cc <= rect[3]):
                return None
            if grid[rr, cc] != bg:
                return None
            grid[rr, cc] = col
            used.add(col)
    return used, counts


def _validate_color_order(grid, bg, wall):
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

    stamped = _stamp_markers(grid, rects, bg, marker_pool, rng)
    if stamped is None:
        return None
    used, counts = stamped

    marked_regions = [n for n in counts if n > 0]
    if len(marked_regions) < 2:
        return None
    if sum(marked_regions) < 3:
        return None
    if not any(n >= 2 for n in marked_regions):
        return None
    if len(set(marked_regions)) < min(2, len(marked_regions)):
        return None

    if not (DISTINCT_MARKER_COLORS_MIN <= len(used) <= DISTINCT_MARKER_COLORS_MAX):
        return None
    if not _validate_color_order(grid, bg, wall):
        return None

    try:
        out = main(grid)
        full = _full_fill_main(grid)
    except Exception:
        return None

    if np.array_equal(grid, out):
        return None
    if np.array_equal(out, full):
        return None
    if int(np.count_nonzero(out != full)) < 4:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_MAX_ATTEMPTS):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError(f"generate_input failed after {GEN_MAX_ATTEMPTS} attempts")
