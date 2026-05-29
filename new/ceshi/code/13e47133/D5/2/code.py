# Variant of 13e47133
# Dimension: D5 — 变换：标记像素在输出中的处理 (transform)
# Change: 标记像素被同心框架的调色板颜色覆盖，不在输出中单独保留... → 标记像素消失且擦除所在框架层

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[5]
_SOLVER_DIR = _ROOT / "solvers"
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import components_mask

# Design analysis
# variant_spec:
#   core_change:
#     The original task uses each marker color as the paint assigned to its own
#     Chebyshev-depth layer. In this variant, the marker still defines a palette
#     slot, but the geometric ring at that exact absolute depth is erased back to
#     the region base/background color instead of being painted.
#   concrete_behavior:
#     Regions are still wall-separated free-space components. Markers still sit on
#     consecutive diagonal depths from a region corner, so a human can read depth
#     order directly from geometry. The output still contains concentric frames,
#     but any depth that hosts a marker becomes a blank ring. Deeper rings keep
#     following the cyclic palette, so the region shows visible "gaps" rather than
#     collapsing into a solid blank rectangle.
#   behavior_trace:
#     Example region with marker depths d=0:red and d=1:blue:
#       original palette = [red, blue], so depths 0,1,2,3 paint as red,blue,red,blue
#       variant output keeps the same palette construction, but absolute depths
#       0 and 1 are erased, so depths 0,1,2,3 become base,base,red,blue
#     Difference:
#       markers no longer survive as painted colors on their own rings; instead
#       they delete those rings while still determining the later cyclic colors.
#   complexity_check:
#     The rule is still single-mechanism and region-local: infer wall regions,
#     read marker depths, build palette, then blank marked depths. To keep the
#     effect observable, the generator rejects cases with too few erased cells or
#     too little surviving colored structure.
#   implicit_assumptions_review:
#     - Wall detection by global mode/second mode is preserved, so the generator
#       must still keep base as the most common color and wall as second-most.
#     - Marker detection still relies on "non-base cell inside a region", so the
#       generator must not introduce extra decorative colors.
#     - Palette construction assumes at most one meaningful marker color per
#       depth; the generator still places consecutive diagonal markers with unique
#       depths.
#     - Erasure is keyed by absolute boundary distance, not by color id or modulo,
#       which keeps the rule visible from geometry alone and avoids hidden color
#       ordering tricks.
#   placement_legality:
#     Regions remain axis-aligned rectangles from recursive wall splits. Marker
#     placement stays on an inward diagonal from one corner, with consecutive
#     Chebyshev depths that remain strictly inside the region and do not overlap
#     walls or each other.
#   visual_observability:
#     The generator requires multiple marked regions, several total markers, and a
#     minimum number of cells that differ from the legacy solver. This avoids
#     degenerating into samples where the new erase behavior is too subtle.

DIR8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _cc4(mask):
    return components_mask(mask, connectivity=4)


def _mode(arr):
    vals, counts = np.unique(np.asarray(arr), return_counts=True)
    return int(vals[counts.argmax()])


def _boundary_distance(component_mask: np.ndarray) -> np.ndarray:
    """Compute Chebyshev distance-to-boundary on a single free-space component."""
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


def _solve(grid: np.ndarray, erase_marker_layers: bool) -> np.ndarray:
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
            if d < 0:
                continue
            if d not in dist_to_color:
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

        erased_depths = set(dist_to_color) if erase_marker_layers else set()
        m = len(palette)
        for r, c in pts:
            d = int(dist[r, c])
            if d in erased_depths:
                out[r, c] = base
            else:
                out[r, c] = palette[d % m]

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill wall-partitioned regions with cyclic bands while marker layers become blank gaps.

    Input structure:
    - The grid is split into free-space regions by a single wall color.
    - Each region is mostly one base/background color, plus a few isolated marker pixels.
    - Marker pixels sit at specific Chebyshev depths from that region's boundary.

    Rule:
    - Read the marker colors in increasing boundary-distance order to assemble the same cyclic palette as in the original task, prepending the base color when the first marker is not on the boundary.
    - Paint the region by boundary distance modulo palette length.
    - However, if a boundary-distance layer contains a marker in the input, erase that entire absolute layer back to the region base color instead of painting it.

    Key insight: marker depth still chooses where a palette color would enter the inward band cycle, but the marker also punches out its own ring, leaving a blank concentric gap at that exact depth.
    """
    return _solve(grid, erase_marker_layers=True)


def _legacy_main(grid: np.ndarray) -> np.ndarray:
    return _solve(grid, erase_marker_layers=False)


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

# ---- Distinct marker colors across grid ----
DISTINCT_MARKER_COLORS_MIN, DISTINCT_MARKER_COLORS_MAX = 3, 9

# ---- Variant visibility filters ----
MIN_MARKED_REGIONS = 2
MIN_TOTAL_MARKERS = 3
MIN_ERASED_CELLS = 10
MIN_COLORED_OUTPUT_CELLS = 8

# ---- Retry budget ----
GEN_MAX_ATTEMPTS = 300

_CORNERS = [(0, 0, 1, 1), (0, 1, 1, -1), (1, 0, -1, 1), (1, 1, -1, -1)]


def _max_cheb_depth(h: int, w: int) -> int:
    return (min(h, w) - 1) // 2


def _place_walls(H: int, W: int, rng: random.Random):
    """Build a wall mask by recursive rectangular subdivision."""
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


def _place_markers(rect, marker_pool: list[int], rng: random.Random):
    """
    Produce 0..MARKERS_PER_REGION_MAX markers along a diagonal from one bbox corner.
    Marker depths are consecutive and stop before the deepest ring so the output
    still has at least one inner layer after the erased zone.
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
    return [
        (start_d + k, col, br + dr * (start_d + k), bc + dc * (start_d + k))
        for k, col in enumerate(colors)
    ]


def _pick_palette(rng: random.Random):
    colors = list(range(10))
    bg = rng.choice(colors)
    colors.remove(bg)
    wall = rng.choice(colors)
    colors.remove(wall)
    return bg, wall, colors


def _stamp_markers(grid, rects, bg, marker_pool, rng):
    """
    Place markers into grid (in place).
    Returns (used_colors, markers_by_region) or (None, None) on illegal placement.
    """
    used = set()
    markers_by_region: list[list[tuple[int, int, int, int]]] = []
    for rect in rects:
        markers = _place_markers(rect, marker_pool, rng)
        markers_by_region.append(markers)
        for _d, col, rr, cc in markers:
            if not (rect[0] <= rr <= rect[2] and rect[1] <= cc <= rect[3]):
                return None, None
            if grid[rr, cc] != bg:
                return None, None
            grid[rr, cc] = col
            used.add(col)
    return used, markers_by_region


def _validate_color_order(grid, bg, wall):
    vals, counts = np.unique(grid, return_counts=True)
    order = np.argsort(-counts)
    if int(vals[order[0]]) != bg:
        return False
    return len(order) >= 2 and int(vals[order[1]]) == wall


def _variant_metrics(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    bg = _mode(g)
    other = g[g != bg]
    wall = bg if other.size == 0 else _mode(other)
    variant = main(g)
    legacy = _legacy_main(g)
    free = g != wall
    erased = int(np.sum(free & (variant == bg) & (legacy != bg)))
    colored = int(np.sum(free & (variant != bg)))
    diff = int(np.sum(free & (variant != legacy)))
    return {
        "variant": variant,
        "erased_cells": erased,
        "colored_cells": colored,
        "diff_cells": diff,
    }


def _gen_attempt(rng: random.Random):
    H = rng.randint(H_MIN, H_MAX)
    W = rng.randint(W_MIN, W_MAX)

    bg, wall, marker_pool = _pick_palette(rng)

    wall_mask, rects = _place_walls(H, W, rng)
    if not (N_REGIONS_MIN <= len(rects) <= N_REGIONS_MAX):
        return None

    grid = np.full((H, W), bg, dtype=int)
    grid[wall_mask == 1] = wall

    used, markers_by_region = _stamp_markers(grid, rects, bg, marker_pool, rng)
    if used is None:
        return None

    total_markers = sum(len(markers) for markers in markers_by_region)
    marked_regions = sum(1 for markers in markers_by_region if markers)
    if total_markers < MIN_TOTAL_MARKERS or marked_regions < MIN_MARKED_REGIONS:
        return None

    if not (DISTINCT_MARKER_COLORS_MIN <= len(used) <= DISTINCT_MARKER_COLORS_MAX):
        return None
    if not _validate_color_order(grid, bg, wall):
        return None

    try:
        metrics = _variant_metrics(grid)
    except Exception:
        return None

    out = metrics["variant"]
    if np.array_equal(grid, out):
        return None
    if metrics["erased_cells"] < MIN_ERASED_CELLS:
        return None
    if metrics["colored_cells"] < MIN_COLORED_OUTPUT_CELLS:
        return None
    if metrics["diff_cells"] < MIN_ERASED_CELLS:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_MAX_ATTEMPTS):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError(f"generate_input failed after {GEN_MAX_ATTEMPTS} attempts")
