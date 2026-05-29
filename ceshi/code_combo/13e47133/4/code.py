# Combo variant of 13e47133 (index 4)
# combo_spec: D2/2+D5/2
# Name: 间断轨迹+标记擦除层

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_SOLVER_UTILS = Path("/home/lixinyu/snap/new/solvers")
if str(_SOLVER_UTILS) not in sys.path:
    sys.path.append(str(_SOLVER_UTILS))

from arc_utils import components_mask

# Design analysis for the combo variant
# -------------------------------------
# combo_variant_spec:
#   core_resolution:
#     The standalone D2 rule for this task is "use marker colors only on even layers, once
#     each, without cyclic repetition", while D5 erases every marker-hosting layer. A naive
#     intersection would erase every colored layer and collapse the task. The fused solver
#     therefore keeps both visible rules by consuming the ordered marker colors on the
#     surviving even layers only: marked layers are blank gaps, odd layers are blank gaps,
#     and the next unused marker color is assigned to the next unmarked even layer.
#   a_concretization:
#     input_view: >
#       The input still contains wall-separated rectangular regions, one dominant base color
#       per region, and a few isolated marker pixels placed on even Chebyshev depths
#       0, 2, 4, ... along one corner diagonal.
#     behavior_trace: >
#       Example region with markers at d=0 color 7 and d=2 color 3.
#       Original task: palette [7, 3], so depths 0..6 paint as 7,3,7,3,7,3,7.
#       Standalone D2/2 idea: only depths 0 and 2 would be colored, once each.
#       Fused combo: depths 0 and 2 are erased because they host markers, so the first two
#       surviving unmarked even layers inherit those colors instead, yielding
#       blank,blank,blank,blank,7,blank,3.
#       Difference from original: there is no cyclic repetition and odd rings are blank.
#       Difference from pure erase-only: the surviving color layers are sparse one-shot
#       frames rather than part of an infinite modulo cycle.
#   b_complexity_calibration:
#     choices:
#       - Markers are restricted to even depths so every marker participates in the parity
#         logic and humans can read the skip pattern directly from input geometry.
#       - Marker depths are consecutive even values starting at 0, and the generator keeps
#         at least as many later unmarked even layers as markers. This guarantees every
#         marker color can reappear exactly once after the erased gap sequence.
#       - Regions still use 0-3 markers and 3-9 distinct marker colors grid-wide, matching
#         the original task's scale while keeping the combo rule legible.
#   c_implicit_assumption_review:
#     - bg = mode(grid) and wall = second mode still hold because markers remain sparse and
#       the generator preserves large base-colored free space plus continuous walls.
#     - components of non-wall cells are still the correct regions; the combo changes paint,
#       not topology.
#     - base = mode(region_values) remains valid because markers are isolated.
#     - palette construction still assumes at most one marker depth per region layer.
#       The generator enforces unique even depths within each region.
#     - The combo replaces cyclic `palette[d % m]` with three assumptions:
#         1. odd absolute depths are always base-colored;
#         2. even depths containing markers are erased to base;
#         3. the ordered marker colors are consumed once each on the remaining even depths.
#   d_placement_legality:
#     - Recursive full-row/full-column wall splits keep regions rectangular and disjoint.
#     - A corner-diagonal marker at even depth d stays inside its rectangle whenever
#       d <= max Chebyshev depth.
#     - By capping marker count to half of all even layers in a region, the generator ensures
#       enough unmarked even depths remain for every marker color to appear once.
#     - The generator validates that markers never overwrite walls or one another.
#   e_visual_observability:
#     - Input markers appear every other inward step, hinting that parity matters.
#     - Output shows three simultaneous signals that humans can see:
#         1. all odd rings are blank;
#         2. even rings at marker depths are also blank;
#         3. the marker colors reappear once each on deeper even rings in the same order.
#     - Samples are rejected unless they contain multiple marked regions, enough erased cells,
#       enough surviving colored layers, and enough difference from both the skip-only and
#       erase-only ablations.
#   f_visual_cleanliness:
#     - Each region has only a few markers, so the diagonal cue remains readable.
#     - Walls remain unchanged and axis-aligned, keeping regions visually separated.
#     - Colored output is restricted to full rectangular rings, avoiding noisy local effects.
#   g_color_inference:
#     - Colors are copied from marker pixels with no hidden remapping.
#     - The only nontrivial inference is geometric: depth parity and whether a depth was
#       explicitly occupied by a marker.
#   main_to_generator_contract:
#     - main assumes marker depths are unique within a region; generate_input guarantees it.
#     - main expects enough surviving even layers to consume each marker color once;
#       generate_input enforces that per marked region.
#     - main would also erase odd-depth markers if they appeared, but generate_input never
#       emits them because they would add no visible evidence.


DIR8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _cc4(mask):
    return components_mask(mask, connectivity=4)


def _mode(arr):
    vals, counts = np.unique(np.asarray(arr), return_counts=True)
    return int(vals[counts.argmax()])


def _boundary_distance(component_mask: np.ndarray) -> np.ndarray:
    """Compute 8-neighborhood Chebyshev distance to the component boundary."""
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


def _region_palette(dist_to_color: dict[int, int], base: int) -> list[int]:
    if not dist_to_color:
        return [int(base)]
    ds = sorted(dist_to_color)
    palette: list[int] = []
    if ds[0] > 0:
        palette.append(int(base))
    for d in ds:
        palette.append(int(dist_to_color[d]))
    return palette


def _extract_regions(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _mode(g)
    other = g[g != bg]
    wall = bg if other.size == 0 else _mode(other)
    free = g != wall
    comps = _cc4(free)
    return g, h, w, bg, wall, comps


def _region_info(g: np.ndarray, pts, h: int, w: int):
    comp_mask = np.zeros((h, w), dtype=bool)
    for r, c in pts:
        comp_mask[r, c] = True

    base = _mode(g[comp_mask])
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

    return base, dist, dist_to_color


def _solve_cyclic(grid: np.ndarray, erase_marker_layers: bool) -> np.ndarray:
    g, h, w, _bg, wall, comps = _extract_regions(grid)
    out = g.copy()

    for pts in comps:
        base, dist, dist_to_color = _region_info(g, pts, h, w)
        erased_depths = set(dist_to_color) if erase_marker_layers else set()
        palette = _region_palette(dist_to_color, int(base))
        m = len(palette)

        for r, c in pts:
            d = int(dist[r, c])
            if d in erased_depths:
                out[r, c] = int(base)
            else:
                out[r, c] = int(palette[d % m])

    return out


def _solve_skip_only(grid: np.ndarray) -> np.ndarray:
    g, h, w, _bg, wall, comps = _extract_regions(grid)
    out = g.copy()

    for pts in comps:
        base, dist, dist_to_color = _region_info(g, pts, h, w)
        even_dist_to_color = {d: dist_to_color[d] for d in sorted(dist_to_color) if d % 2 == 0}

        for r, c in pts:
            d = int(dist[r, c])
            if d >= 0 and d % 2 == 0 and d in even_dist_to_color:
                out[r, c] = int(even_dist_to_color[d])
            else:
                out[r, c] = int(base)

    return out


def _solve_combo(grid: np.ndarray) -> np.ndarray:
    g, h, w, _bg, wall, comps = _extract_regions(grid)
    out = g.copy()

    for pts in comps:
        base, dist, dist_to_color = _region_info(g, pts, h, w)
        marker_depths = sorted(d for d in dist_to_color if d % 2 == 0)
        ordered_colors = [int(dist_to_color[d]) for d in marker_depths]

        max_depth = max(int(dist[r, c]) for r, c in pts)
        even_fill: dict[int, int] = {}
        next_color = 0
        marked_depths = set(marker_depths)
        for d in range(0, max_depth + 1, 2):
            if d in marked_depths:
                even_fill[d] = int(base)
            elif next_color < len(ordered_colors):
                even_fill[d] = ordered_colors[next_color]
                next_color += 1
            else:
                even_fill[d] = int(base)

        for r, c in pts:
            d = int(dist[r, c])
            if d % 2 == 1:
                out[r, c] = int(base)
            else:
                out[r, c] = int(even_fill[d])

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill wall-partitioned regions with sparse cyclic frames whose marked depths are erased.

    Input structure:
    - The grid is divided into free-space regions by one wall color.
    - Each region is mostly one base color, plus a few isolated marker pixels on even Chebyshev depths from the region boundary.

    Rule:
    - Read marker colors in increasing boundary-depth order.
    - Keep every odd boundary-distance layer at the region base color.
    - Among even layers, erase any layer that contains a marker in the input.
    - On the remaining even layers, use the ordered marker colors once each from outside to inside; after the colors are exhausted, leave later layers at the base color.

    Key insight: parity still creates the intermittent frame pattern, but marker-hosting layers turn into blank gaps and force the one-shot marker colors to resume only on later even rings.
    """
    return _solve_combo(grid)


def _legacy_main(grid: np.ndarray) -> np.ndarray:
    return _solve_cyclic(grid, erase_marker_layers=False)


def _skip_only_main(grid: np.ndarray) -> np.ndarray:
    return _solve_skip_only(grid)


def _erase_only_main(grid: np.ndarray) -> np.ndarray:
    return _solve_cyclic(grid, erase_marker_layers=True)


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
MARKER_COUNT_WEIGHTS = [1, 2, 4, 4]

# ---- Distinct marker colors across grid ----
DISTINCT_MARKER_COLORS_MIN, DISTINCT_MARKER_COLORS_MAX = 3, 9

# ---- Variant visibility filters ----
MIN_MARKED_REGIONS = 2
MIN_TOTAL_MARKERS = 3
MIN_ERASED_CELLS = 10
MIN_COLORED_OUTPUT_CELLS = 10
MIN_POST_MARKER_COLORED_LAYERS = 2
MIN_MULTI_MARKER_REGIONS = 1
MIN_DIFF_VS_SKIP_ONLY = 10
MIN_DIFF_VS_ERASE_ONLY = 10
MIN_DIFF_VS_LEGACY = 12

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
    Place 0..3 markers on consecutive even depths from one corner.
    The region always keeps at least as many later unmarked even layers as markers,
    so every marker color can reappear exactly once under the combo rule.
    """
    r0, c0, r1, c1 = rect
    max_d = _max_cheb_depth(r1 - r0 + 1, c1 - c0 + 1)
    total_even_layers = max_d // 2 + 1
    upper = min(MARKERS_PER_REGION_MAX, total_even_layers // 2)
    if upper <= 0:
        return []

    n = rng.choices(range(len(MARKER_COUNT_WEIGHTS)), weights=MARKER_COUNT_WEIGHTS)[0]
    n = min(n, upper)
    if n == 0:
        return []

    row_from, col_from, dr, dc = rng.choice(_CORNERS)
    br = r0 if row_from == 0 else r1
    bc = c0 if col_from == 0 else c1

    dists = [2 * k for k in range(n)]
    colors = rng.sample(marker_pool, n)
    return [(d, col, br + dr * d, bc + dc * d) for d, col in zip(dists, colors)]


def _pick_palette(rng: random.Random):
    colors = list(range(10))
    bg = rng.choice(colors)
    colors.remove(bg)
    wall = rng.choice(colors)
    colors.remove(wall)
    return bg, wall, colors


def _stamp_markers(grid, rects, bg, marker_pool, rng):
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


def _combo_metrics(grid: np.ndarray):
    g, _h, _w, bg, wall, comps = _extract_regions(grid)
    combo = main(g)
    skip_only = _skip_only_main(g)
    erase_only = _erase_only_main(g)
    legacy = _legacy_main(g)
    free = g != wall

    erased_cells = int(np.sum(free & (combo == bg) & (skip_only != bg)))
    colored_cells = int(np.sum(free & (combo != bg)))
    diff_vs_skip_only = int(np.sum(free & (combo != skip_only)))
    diff_vs_erase_only = int(np.sum(free & (combo != erase_only)))
    diff_vs_legacy = int(np.sum(free & (combo != legacy)))

    post_marker_colored_layers = 0
    combo_colored_layers = 0
    marked_regions = 0
    multi_marker_regions = 0
    total_markers = 0

    for pts in comps:
        comp_mask = np.zeros(g.shape, dtype=bool)
        for r, c in pts:
            comp_mask[r, c] = True

        base = _mode(g[comp_mask])
        dist = _boundary_distance(comp_mask)
        marker_depths = sorted(
            {
                int(dist[r, c])
                for r, c in pts
                if int(g[r, c]) != int(base) and int(dist[r, c]) >= 0
            }
        )
        if marker_depths:
            marked_regions += 1
        if len(marker_depths) >= 2:
            multi_marker_regions += 1
        total_markers += len(marker_depths)
        deepest_marker = marker_depths[-1] if marker_depths else -1

        max_comp_depth = max(int(dist[r, c]) for r, c in pts)
        for d in range(0, max_comp_depth + 1, 2):
            if d <= deepest_marker:
                continue
            layer_cells = [(r, c) for r, c in pts if int(dist[r, c]) == d]
            if not layer_cells:
                continue
            lr, lc = layer_cells[0]
            if int(combo[lr, lc]) != int(base):
                combo_colored_layers += 1
                post_marker_colored_layers += 1

    return {
        "combo": combo,
        "erased_cells": erased_cells,
        "colored_cells": colored_cells,
        "diff_vs_skip_only": diff_vs_skip_only,
        "diff_vs_erase_only": diff_vs_erase_only,
        "diff_vs_legacy": diff_vs_legacy,
        "combo_colored_layers": combo_colored_layers,
        "post_marker_colored_layers": post_marker_colored_layers,
        "marked_regions": marked_regions,
        "multi_marker_regions": multi_marker_regions,
        "total_markers": total_markers,
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
        metrics = _combo_metrics(grid)
    except Exception:
        return None

    out = metrics["combo"]
    if np.array_equal(grid, out):
        return None
    if metrics["erased_cells"] < MIN_ERASED_CELLS:
        return None
    if metrics["colored_cells"] < MIN_COLORED_OUTPUT_CELLS:
        return None
    if metrics["multi_marker_regions"] < MIN_MULTI_MARKER_REGIONS:
        return None
    if metrics["combo_colored_layers"] != metrics["total_markers"]:
        return None
    if metrics["post_marker_colored_layers"] < MIN_POST_MARKER_COLORED_LAYERS:
        return None
    if metrics["diff_vs_skip_only"] < MIN_DIFF_VS_SKIP_ONLY:
        return None
    if metrics["diff_vs_erase_only"] < MIN_DIFF_VS_ERASE_ONLY:
        return None
    if metrics["diff_vs_legacy"] < MIN_DIFF_VS_LEGACY:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_MAX_ATTEMPTS):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError(f"generate_input failed after {GEN_MAX_ATTEMPTS} attempts")
