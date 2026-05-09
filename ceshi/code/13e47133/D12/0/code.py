# Variant of 13e47133
# Dimension: D12 — 墙壁在输出中的处理 (obstacle)
# Change: 墙壁像素在输出中保持不变... → 墙壁被相邻区域的最外层颜色覆盖

import random
from collections import deque

import numpy as np

from arc_utils import components_mask

DIR8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
DIR4 = [(-1, 0), (0, -1), (0, 1), (1, 0)]


"""
design_analysis:
  variant_goal:
    - Keep the original palette-construction rule inside each non-wall region.
    - Change only how wall-colored obstacle cells appear in the output: walls vanish,
      and every wall pixel is recolored by the outermost frame color of the nearest
      region.
  concretization_and_trace:
    - Original behavior: if two rectangles are separated by a wall column, the solver
      colors both rectangles with concentric frames but leaves the wall column unchanged.
    - Variant behavior: the same rectangles still get the same concentric frames, but
      the wall column is also repainted. The wall cells closest to the left rectangle
      take that rectangle's frame-0 color; the wall cells closest to the right
      rectangle take the right rectangle's frame-0 color.
    - Small example with a 2-column wall band:
      left outer color = red, right outer color = blue.
      Original output keeps [wall, wall].
      Variant output becomes [red, blue].
    - The difference is therefore localized to the final painting stage after each
      region's palette has been inferred.
  complexity_calibration:
    - A 1-pixel wall creates unavoidable left/right distance ties, which would force an
      arbitrary hidden tie-break in the solver and make the new rule hard to infer from
      few ARC examples.
    - To keep D12 observable, the generator uses 2-pixel-thick wall bands and keeps all
      walls parallel within one grid. Then each wall pixel has a unique nearest side,
      so the disappearance rule is visible and geometry-driven rather than scan-order
      driven.
    - Region count remains 3-6 and markers remain 0-3 per region, so the task still
      needs palette inference instead of collapsing into a trivial recoloring.
  implicit_assumption_audit:
    - The original `free = g != wall` and connected-component extraction still hold,
      because walls remain a single distinct color in the input.
    - The original palette derivation assumes wall cells never need distances. That is
      still fine for region cells, but no longer sufficient for the final output, so an
      added wall-filling pass is required.
    - `out = g.copy()` was harmless before because untouched wall cells were correct.
      Under the variant it would be wrong unless every wall pixel is explicitly
      overwritten after region painting.
    - Stable tie-breaking is still implemented in the wall BFS, but the generator is
      constructed so its valid samples do not rely on ties.
  placement_legality:
    - The generator partitions the canvas into axis-aligned strips separated by 2-cell
      wall bands. Regions never overlap, wall cells never overlap regions, and every
      region has width/height at least 3.
    - Marker placement still follows the original rule: consecutive Chebyshev depths on
      a diagonal from one region corner, with each marker checked to stay inside its
      region and to overwrite only base-colored cells.
  visual_observability:
    - The wall color is present in the input but absent from the output.
    - At least one wall band is required to have different nearest-side outer colors on
      its two columns/rows, so examples visibly show the wall split into two colors
      instead of merely being recolored uniformly.
  visual_cleanliness:
    - Parallel strip partitions avoid messy T/cross junctions, which would otherwise
      create visually noisy four-way competitions inside thick walls.
    - Markers remain isolated single pixels inside otherwise monochrome regions, so the
      palette cue is easy to locate.
  color_inference:
    - No hidden use of color ids is introduced. Colors only affect the task through
      visible marker positions and the resulting outermost frame colors.
    - The generator rejects samples where all neighboring regions induce the same wall
      coverage pattern, preventing regression to a nearly constant wall repaint.
  new_solver_assumptions_to_enforce:
    - `main()` assumes there is at least one non-wall region whenever wall cells exist.
      The generator always creates 3-6 regions, so this holds.
    - The wall-fill BFS assumes region ownership can be represented by a single
      outer-color label per region. This holds because frame-0 color is constant all
      along one region boundary.
"""


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


def _region_palette_and_fill(g: np.ndarray, wall: int):
    """Return the region-filled output and the frame-0 color attached to each free cell."""
    h, w = g.shape
    out = g.copy()
    free = g != wall
    comps = _cc4(free)
    outer = np.full((h, w), -1, dtype=int)

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

        palette: list[int] = []
        if dist_to_color:
            ds = sorted(dist_to_color)
            if ds[0] > 0:
                palette.append(int(base))
            for d in ds:
                palette.append(int(dist_to_color[d]))
        else:
            palette = [int(base)]

        outer_color = int(palette[0])
        m = len(palette)
        for r, c in pts:
            d = int(dist[r, c])
            out[r, c] = palette[d % m]
            outer[r, c] = outer_color

    return out, outer


def _paint_walls(out: np.ndarray, wall_mask: np.ndarray, outer: np.ndarray) -> None:
    """Flood each wall pixel from its nearest region, using the region's frame-0 color."""
    h, w = out.shape
    if not wall_mask.any():
        return

    dist = np.full((h, w), -1, dtype=int)
    color = np.full((h, w), -1, dtype=int)
    q: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in range(w):
            if outer[r, c] == -1:
                continue
            dist[r, c] = 0
            color[r, c] = int(outer[r, c])
            q.append((r, c))

    if not q:
        return

    while q:
        r, c = q.popleft()
        nd = dist[r, c] + 1
        col = color[r, c]
        for dr, dc in DIR4:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and dist[rr, cc] == -1:
                dist[rr, cc] = nd
                color[rr, cc] = col
                q.append((rr, cc))

    out[wall_mask] = color[wall_mask]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill partitioned regions with concentric bands inferred from marker depths, then let those bands swallow the walls.

    Input structure:
    - The grid is divided into several non-wall regions by a single wall color.
    - Each region has one dominant base color plus a few isolated marker pixels at specific Chebyshev depths from that region's boundary.

    Rule:
    - For every non-wall region, sort marker colors by boundary depth; if no depth-0 marker exists, prepend the region base color, then paint cyclic concentric frames with that palette.
    - After all regions are painted, repaint every wall pixel with the outermost frame color of the nearest region, so the wall color disappears from the output.

    Key insight: Marker depths still define each region's cyclic palette exactly as in the original task; the only extra step is that wall cells inherit frame-0 colors by nearest-region coverage.
    """
    g = np.asarray(grid, dtype=int)

    bg = _mode(g)
    other = g[g != bg]
    wall = bg if other.size == 0 else _mode(other)

    out, outer = _region_palette_and_fill(g, wall)
    wall_mask = g == wall
    _paint_walls(out, wall_mask, outer)
    return out


# ---- Canvas bounds ----
H_MIN, H_MAX = 10, 30
W_MIN, W_MAX = 13, 30

# ---- Wall and partition layout ----
WALL_THICKNESS = 2
N_REGIONS_MIN, N_REGIONS_MAX = 3, 6
MIN_REGION_DIM = 3

# ---- Markers per region ----
MARKERS_PER_REGION_MAX = 3
MARKER_COUNT_WEIGHTS = [1, 2, 3, 3]
START_DIST_MAX = 1

# ---- Distinct marker colors across grid ----
DISTINCT_MARKER_COLORS_MIN, DISTINCT_MARKER_COLORS_MAX = 3, 9

# ---- Retry budget ----
GEN_MAX_ATTEMPTS = 400

_CORNERS = [(0, 0, 1, 1), (0, 1, 1, -1), (1, 0, -1, 1), (1, 1, -1, -1)]


def _max_cheb_depth(h: int, w: int) -> int:
    """Max Chebyshev depth from boundary in an h x w rectangle."""
    return (min(h, w) - 1) // 2


def _feasible_region_counts(total_len: int) -> list[int]:
    out = []
    for n in range(N_REGIONS_MIN, N_REGIONS_MAX + 1):
        need = n * MIN_REGION_DIM + (n - 1) * WALL_THICKNESS
        if need <= total_len:
            out.append(n)
    return out


def _split_total(total: int, parts: int, min_part: int, rng: random.Random) -> list[int]:
    """Random composition of total into `parts` pieces, each at least `min_part`."""
    extra = total - parts * min_part
    cuts = sorted(rng.sample(range(extra + parts - 1), parts - 1)) if parts > 1 else []
    vals = []
    prev = -1
    for cut in cuts + [extra + parts - 1]:
        vals.append(cut - prev - 1)
        prev = cut
    return [min_part + v for v in vals]


def _place_walls(H: int, W: int, rng: random.Random):
    """
    Partition the grid into parallel strips separated by 2-cell wall bands.

    Returns (wall_mask, region_bboxes, orientation), where orientation is 'h' if walls
    are horizontal bands and 'v' if walls are vertical bands.
    """
    opts = []
    if _feasible_region_counts(W):
        opts.append("v")
    if _feasible_region_counts(H):
        opts.append("h")
    if not opts:
        return None

    orientation = rng.choice(opts)
    total_len = W if orientation == "v" else H
    n_regions = rng.choice(_feasible_region_counts(total_len))
    lengths = _split_total(total_len - (n_regions - 1) * WALL_THICKNESS, n_regions, MIN_REGION_DIM, rng)

    wall_mask = np.zeros((H, W), dtype=int)
    rects = []
    pos = 0
    for i, length in enumerate(lengths):
        if orientation == "v":
            rects.append((0, pos, H - 1, pos + length - 1))
            pos += length
            if i < n_regions - 1:
                wall_mask[:, pos:pos + WALL_THICKNESS] = 1
                pos += WALL_THICKNESS
        else:
            rects.append((pos, 0, pos + length - 1, W - 1))
            pos += length
            if i < n_regions - 1:
                wall_mask[pos:pos + WALL_THICKNESS, :] = 1
                pos += WALL_THICKNESS

    return wall_mask, rects, orientation


def _place_markers(rect, marker_pool: list[int], rng: random.Random):
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

    start_d = rng.randint(0, min(START_DIST_MAX, max_d - n))
    row_from, col_from, dr, dc = rng.choice(_CORNERS)
    br = r0 if row_from == 0 else r1
    bc = c0 if col_from == 0 else c1
    colors = rng.sample(marker_pool, n)

    return [(start_d + k, col, br + dr * (start_d + k), bc + dc * (start_d + k)) for k, col in enumerate(colors)]


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
    Place markers into grid in place.

    Returns:
      - used: set of marker colors used
      - region_markers: list aligned with rects, each entry a list of marker tuples
    """
    used = set()
    region_markers = []
    for rect in rects:
        markers = _place_markers(rect, marker_pool, rng)
        for _d, col, rr, cc in markers:
            if not (rect[0] <= rr <= rect[2] and rect[1] <= cc <= rect[3]):
                return None, None
            if grid[rr, cc] != bg:
                return None, None
            grid[rr, cc] = col
            used.add(col)
        region_markers.append(markers)
    return used, region_markers


def _validate_color_order(grid, bg, wall):
    """Confirm bg is mode and wall is second mode of the assembled grid."""
    vals, counts = np.unique(grid, return_counts=True)
    order = np.argsort(-counts)
    if int(vals[order[0]]) != bg:
        return False
    return len(order) >= 2 and int(vals[order[1]]) == wall


def _old_main_like(grid: np.ndarray) -> np.ndarray:
    """Reference the original behavior: regions repaint, walls stay unchanged."""
    g = np.asarray(grid, dtype=int)
    bg = _mode(g)
    other = g[g != bg]
    wall = bg if other.size == 0 else _mode(other)
    out, _outer = _region_palette_and_fill(g, wall)
    return out


def _outer_color_from_markers(markers, bg: int) -> int:
    if not markers:
        return int(bg)
    markers = sorted(markers)
    return int(bg) if markers[0][0] > 0 else int(markers[0][1])


def _has_visible_wall_split(rects, region_markers, bg: int) -> bool:
    outer_colors = [_outer_color_from_markers(markers, bg) for markers in region_markers]
    return any(outer_colors[i] != outer_colors[i + 1] for i in range(len(rects) - 1))


def _gen_attempt(rng: random.Random):
    H = rng.randint(H_MIN, H_MAX)
    W = rng.randint(W_MIN, W_MAX)

    layout = _place_walls(H, W, rng)
    if layout is None:
        return None
    wall_mask, rects, _orientation = layout

    bg, wall, marker_pool = _pick_palette(rng)
    grid = np.full((H, W), bg, dtype=int)
    grid[wall_mask == 1] = wall

    used, region_markers = _stamp_markers(grid, rects, bg, marker_pool, rng)
    if used is None:
        return None
    if not (DISTINCT_MARKER_COLORS_MIN <= len(used) <= DISTINCT_MARKER_COLORS_MAX):
        return None
    if not _validate_color_order(grid, bg, wall):
        return None
    if not _has_visible_wall_split(rects, region_markers, bg):
        return None

    try:
        out = main(grid)
        old_out = _old_main_like(grid)
    except Exception:
        return None

    if np.array_equal(grid, out):
        return None
    if np.array_equal(out, old_out):
        return None

    wall_changed = np.count_nonzero(out[wall_mask == 1] != wall)
    if wall_changed != int(wall_mask.sum()):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_MAX_ATTEMPTS):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError(f"generate_input failed after {GEN_MAX_ATTEMPTS} attempts")
