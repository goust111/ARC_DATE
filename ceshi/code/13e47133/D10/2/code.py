# Variant of 13e47133
# Dimension: D10 — 距离度量方式 (meta)
# Change: 使用切比雪夫距离（8-邻域）计算到区域边界的距离... → 使用到标记像素的距离而非到边界的距离

import random

import numpy as np

from arc_utils import components_mask

DIR8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


# Design analysis for the D10 variant
# - Concrete behavior:
#   The old task read each region as a stack of rectangular shells measured from
#   the wall/boundary inward. This variant keeps the wall partitioning and keeps
#   marker colors as the palette source, but the geometry now starts from the
#   marker cells themselves. Marker cells are radius 0, their 8-neighborhood is
#   radius 1, and so on. To preserve multiple marker colors instead of collapsing
#   all radius-0 cells to one color, each cell uses its nearest marker as the
#   local source and advances cyclically from that marker's palette index.
# - Small trace:
#   Suppose a 7x7 region has three marker pixels A/B/C on an interior NW-SE
#   diagonal. Old logic: sort A/B/C by boundary depth, maybe prepend base, then
#   color every cell from its boundary shell number; output is rectangular frames.
#   New logic: order A/B/C by reading order, then for each cell pick the nearest
#   marker (tie -> earlier marker) and color by palette[(marker_index + distance)
#   mod 3]. Output becomes square-wave bands centered on the marker chain, not on
#   the outer box.
# - Complexity calibration:
#   A pure "distance to union of markers" rule would recolor every marker cell to
#   the same palette entry and make later marker colors visually weak. The chosen
#   nearest-marker rule keeps 2-3 distinct marker colors active in each marked
#   region while still making D10 the only real semantic change.
# - Implicit assumptions in the original main():
#   `dist = _boundary_distance(comp_mask)` assumed every region had one boundary-
#   based scalar field. That is the exact dimension being changed and must be
#   replaced. The old `dist_to_color` map also assumed markers were indexed by
#   boundary depth. In this variant, marker order must come directly from the
#   marker pixels themselves, so we sort marker coordinates in reading order and
#   do not bucket by depth.
# - Placement legality:
#   The generator still builds non-overlapping rectangular regions by recursive
#   wall subdivision. Marker chains are placed strictly inside a chosen rectangle:
#   for an n-marker NW-SE chain we require one-cell margin around it, so all
#   marker cells stay in-bounds, never touch walls, and leave room for visible
#   outward expansion.
# - Visual observability:
#   Markers are moved into region interiors and usually come in runs of 2 or 3.
#   That prevents degeneration back into boundary-like corner cues and makes the
#   new expansion visibly centered in the middle of a region. The generator also
#   rejects samples where the old boundary solver would produce the same output.
# - Cleanliness / color inference:
#   Regions remain simple rectangles separated by one wall color. Marker colors
#   are the only non-base colors inside a region, so the palette source is easy
#   to spot. No ordering depends on color ids; ordering is purely by visible
#   reading order of the marker chain.


def _cc4(mask):
    return components_mask(mask, connectivity=4)


def _mode(arr):
    vals, counts = np.unique(np.asarray(arr), return_counts=True)
    return int(vals[counts.argmax()])


def _boundary_distance(component_mask: np.ndarray) -> np.ndarray:
    """Original boundary-based field kept for generator-side regression checks."""
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


def _nearest_marker_index_and_distance(
    r: int, c: int, markers: list[tuple[int, int, int]]
) -> tuple[int, int]:
    best_idx = 0
    best_dist = None
    for idx, (mr, mc, _color) in enumerate(markers):
        d = max(abs(r - mr), abs(c - mc))
        if best_dist is None or d < best_dist or (d == best_dist and idx < best_idx):
            best_idx = idx
            best_dist = d
    return best_idx, int(best_dist)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repaint each wall-bounded region using waves that spread outward from the markers.

    Input structure:
    - The grid is split into orthogonally separated free regions by one wall color.
    - Inside each free region, one dominant base color fills most cells while a short
      list of isolated marker pixels supplies the non-base colors.

    Rule:
    - For every region, collect marker colors in reading order to form a cyclic palette.
    - Each cell finds the nearest marker by Chebyshev distance; ties go to the earlier
      marker in that order.
    - Color the cell with palette[(marker_index + distance_to_that_marker) mod palette_length].
      Regions with no markers stay in the base color.

    Key insight:
    The markers no longer describe how far inward a boundary shell lies; instead each
    marker becomes a local origin, and the palette advances as square bands expand away
    from those origins.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    bg = _mode(g)
    other = g[g != bg]
    wall = bg if other.size == 0 else _mode(other)

    out = g.copy()
    free = g != wall
    comps = _cc4(free)

    for pts in comps:
        comp_mask = np.zeros((h, w), dtype=bool)
        for r, c in pts:
            comp_mask[r, c] = True

        vals = g[comp_mask]
        base = _mode(vals)
        markers: list[tuple[int, int, int]] = []
        for r, c in pts:
            v = int(g[r, c])
            if v != base:
                markers.append((int(r), int(c), v))
        markers.sort(key=lambda item: (item[0], item[1]))

        if not markers:
            for r, c in pts:
                out[r, c] = base
            continue

        palette = [color for _r, _c, color in markers]
        m = len(palette)
        for r, c in pts:
            idx, d = _nearest_marker_index_and_distance(int(r), int(c), markers)
            out[r, c] = palette[(idx + d) % m]

    return out


# ---- Canvas bounds ----
H_MIN, H_MAX = 12, 30
W_MIN, W_MAX = 15, 30

# ---- Wall subdivision ----
N_SPLITS_MIN, N_SPLITS_MAX = 2, 5
MIN_REGION_DIM = 4
SPLITTABLE_LEN = 2 * MIN_REGION_DIM + 1

# ---- Region count filter ----
N_REGIONS_MIN, N_REGIONS_MAX = 3, 6

# ---- Markers per region ----
MARKER_CHOICES = [0, 2, 3]
MARKER_COUNT_WEIGHTS = [1, 4, 4]

# ---- Distinct marker colors across grid ----
DISTINCT_MARKER_COLORS_MIN, DISTINCT_MARKER_COLORS_MAX = 3, 9

# ---- Retry budget ----
GEN_MAX_ATTEMPTS = 500


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
    """
    Produce a strict-interior NW-SE diagonal chain of 0/2/3 markers.
    Each marker = (color, r, c).
    """
    r0, c0, r1, c1 = rect
    rh = r1 - r0 + 1
    rw = c1 - c0 + 1

    feasible_choices = [0]
    for n in (2, 3):
        if rh >= n + 2 and rw >= n + 2:
            feasible_choices.append(n)

    weights = [MARKER_COUNT_WEIGHTS[MARKER_CHOICES.index(n)] for n in feasible_choices]
    n = rng.choices(feasible_choices, weights=weights)[0]
    if n == 0:
        return []

    sr = rng.randint(r0 + 1, r1 - n)
    sc = rng.randint(c0 + 1, c1 - n)
    colors = rng.sample(marker_pool, n)
    return [(colors[k], sr + k, sc + k) for k in range(n)]


def _pick_palette(rng: random.Random):
    colors = list(range(10))
    bg = rng.choice(colors)
    colors.remove(bg)
    wall = rng.choice(colors)
    colors.remove(wall)
    return bg, wall, colors


def _stamp_markers(grid, rects, bg, marker_pool, rng):
    used = set()
    region_marker_counts = []
    interior_margin_sum = 0
    marked_regions = 0

    for rect in rects:
        markers = _place_markers(rect, marker_pool, rng)
        region_marker_counts.append(len(markers))
        if markers:
            marked_regions += 1
        r0, c0, r1, c1 = rect
        for col, rr, cc in markers:
            if not (r0 <= rr <= r1 and c0 <= cc <= c1):
                return None
            if grid[rr, cc] != bg:
                return None
            grid[rr, cc] = col
            used.add(col)
            interior_margin_sum += min(rr - r0, r1 - rr, cc - c0, c1 - cc)

    stats = {
        "region_marker_counts": region_marker_counts,
        "marked_regions": marked_regions,
        "interior_margin_sum": interior_margin_sum,
    }
    return used, stats


def _validate_color_order(grid, bg, wall):
    vals, counts = np.unique(grid, return_counts=True)
    order = np.argsort(-counts)
    if int(vals[order[0]]) != bg:
        return False
    return len(order) >= 2 and int(vals[order[1]]) == wall


def _boundary_style_main(grid: np.ndarray) -> np.ndarray:
    """Original solver logic kept only to reject degenerate generator samples."""
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    bg = _mode(g)
    other = g[g != bg]
    wall = bg if other.size == 0 else _mode(other)

    out = g.copy()
    free = g != wall
    comps = _cc4(free)

    for pts in comps:
        comp_mask = np.zeros((h, w), dtype=bool)
        for r, c in pts:
            comp_mask[r, c] = True

        vals = g[comp_mask]
        base = _mode(vals)
        dist = _boundary_distance(comp_mask)

        dist_to_color: dict[int, int] = {}
        for r, c in pts:
            v = int(g[r, c])
            if v == base:
                continue
            d = int(dist[r, c])
            if d >= 0 and d not in dist_to_color:
                dist_to_color[d] = v

        if dist_to_color:
            ds = sorted(dist_to_color.keys())
            palette = []
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
    used, stats = stamped

    if not (DISTINCT_MARKER_COLORS_MIN <= len(used) <= DISTINCT_MARKER_COLORS_MAX):
        return None
    if stats["marked_regions"] < 2:
        return None
    if stats["interior_margin_sum"] < stats["marked_regions"]:
        return None
    if not any(count >= 3 for count in stats["region_marker_counts"]):
        return None
    if not _validate_color_order(grid, bg, wall):
        return None

    try:
        out = main(grid)
        old_out = _boundary_style_main(grid)
    except Exception:
        return None

    if np.array_equal(grid, out):
        return None
    if np.array_equal(out, old_out):
        return None

    diff = np.count_nonzero(out != old_out)
    if diff < max(8, (H * W) // 20):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_MAX_ATTEMPTS):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError(f"generate_input failed after {GEN_MAX_ATTEMPTS} attempts")
