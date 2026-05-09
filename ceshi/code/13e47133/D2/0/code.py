# Variant of 13e47133
# Dimension: D2 — 轨迹：颜色交替模式 (trajectory)
# Change: 调色板按距离循环：palette[dist % len]，标记颜色与背景色交替... → 周期性重复图案

"""
Design analysis
===============

variant_spec:
  source_task_baseline:
    - Reference task 221dfab4 replaces a simple 2-step alternation with a short
      periodic cycle [Y, Y, G]. The important lesson is that the input still
      gives a local cue for color order, while the output reveals that the
      trajectory through those colors is no longer one-step-per-layer.
    - This variant keeps 13e47133's region / marker semantics intact and only
      changes that trajectory: palette entries are consumed in 2-layer runs.

  concretization_and_trace:
    - Walls still partition the board into rectangular free regions.
    - Markers still sit at consecutive Chebyshev depths from a region boundary,
      and their depth order still defines the palette order.
    - Old behavior on palette [A, B] was:
        depth 0 -> A, 1 -> B, 2 -> A, 3 -> B, ...
      New behavior is:
        depth 0 -> A, 1 -> A, 2 -> B, 3 -> B, 4 -> A, 5 -> A, ...
    - If the first marker is deeper than depth 0, the base color is still
      prepended before applying the repeated-run rule.
    - Concrete trace:
        region palette = [base=7, marker1=3, marker2=8]
        old layers    = 7, 3, 8, 7, 3, 8, ...
        new layers    = 7, 7, 3, 3, 8, 8, 7, 7, ...
      The input construction does not change at the pixel level except that the
      generator now favors deeper / larger regions so this 2-layer beat is
      visible in the output.

  complexity_calibration:
    - A pure solver change would be too easy to miss if many regions were tiny
      or had only one marker; then old and new outputs can look nearly the same.
    - The generator therefore biases toward 2-3 markers per region and rejects
      instances where the new output differs from the legacy output in too few
      cells. This keeps the D2 change visible without changing the task's object
      vocabulary.

  implicit_assumptions_audit:
    - `bg = mode(grid)` and `wall = second mode` still assume markers remain
      sparse relative to region area. The generator preserves this.
    - `base = mode(region values)` assumes the region's dominant color is still
      the original background color. The generator keeps markers isolated so the
      assumption holds.
    - `dist_to_color[d]` keeps at most one color per depth. This was harmless in
      the original task because the generator already emitted exactly one marker
      per used depth. The variant keeps that invariant.
    - The only old assumption that breaks is `out[r, c] = palette[d % m]`,
      which encodes one palette advance per depth layer. The variant replaces it
      with a 2-layer run index `(d // 2) % m`.

  placement_legality:
    - Each region remains a rectangle produced by recursive full-row/full-column
      wall splits, so diagonal marker placement from a corner stays inside the
      region when `start_d + n - 1 <= max_depth`.
    - Markers are written only onto base-colored cells and never onto walls.
    - Validation runs the solver on every sampled input and also compares the
      result to the legacy 1-layer cycle to reject accidental degeneracy.

  visual_observability:
    - The new rule is human-visible only if at least some regions have depth >= 2
      and at least two palette entries. The generator narrows size ranges and
      requires a minimum output difference from the legacy solver.
    - The rule uses only visible geometry: boundary depth and marker order. No
      color-id arithmetic or hidden ordering is introduced.

  cleanliness_and_color_inference:
    - The grid still has only three visual element types: base fill, wall lines,
      and isolated marker pixels. No extra overlays are added.
    - Color inference remains structured because the same region markers still
      define the cyclic palette order; the only new fact to infer is that each
      color persists for two consecutive layers before the cycle advances.

  solver_assumptions_after_change:
    - Every processed component must still have a well-defined boundary-distance
      field and at most one marker color per distance.
    - `generate_input()` guarantees those conditions by using rectangular regions
      and consecutive, non-overlapping diagonal marker placements.
"""

import random

import numpy as np

from arc_utils import components_mask

DIR8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
BAND_RUN_LEN = 2


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


def _solve_with_band_run(grid: np.ndarray, band_run_len: int) -> np.ndarray:
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
            if d < 0 or d in dist_to_color:
                continue
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

        m = len(palette)
        for r, c in pts:
            d = int(dist[r, c])
            out[r, c] = palette[(d // band_run_len) % m]

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill wall-partitioned regions with concentric bands whose palette advances in repeated 2-layer runs.

    Input structure:
    - The grid is split into regions by a wall color.
    - Inside each free region, one dominant base color is interrupted by a few isolated marker pixels placed at different Chebyshev depths from the region boundary.

    Rule:
    - For each region, sort the marker colors by boundary depth to recover the palette order, prepending the base color if the nearest marker is not on the boundary.
    - Paint the region by boundary depth, but hold each palette entry for two consecutive layers before moving to the next one.
    - The palette then wraps around, so the inward pattern follows a repeated beat such as AABBAABB instead of a one-layer alternation.

    Key insight: Marker depths still tell you which colors come next; the only changed mechanic is the trajectory through that palette, which now advances once every two layers rather than once every layer.
    """
    return _solve_with_band_run(grid, BAND_RUN_LEN)


# --------------- generator ---------------

# ---- Canvas bounds ----
H_MIN, H_MAX = 14, 30
W_MIN, W_MAX = 16, 30

# ---- Wall subdivision ----
N_SPLITS_MIN, N_SPLITS_MAX = 2, 5
MIN_REGION_DIM = 5
SPLITTABLE_LEN = 2 * MIN_REGION_DIM + 1

# ---- Region count filter ----
N_REGIONS_MIN, N_REGIONS_MAX = 3, 6

# ---- Markers per region ----
MARKERS_PER_REGION_MAX = 3
START_DIST_MAX = 1

# ---- Distinct marker colors across grid ----
DISTINCT_MARKER_COLORS_MIN, DISTINCT_MARKER_COLORS_MAX = 3, 9

# ---- Variant visibility filters ----
MIN_DIFFERENCE_CELLS = 18
MIN_DEEP_REGIONS = 1
MIN_MARKED_REGIONS = 2

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


def _choose_marker_count(max_d: int, rng: random.Random) -> int:
    upper = min(MARKERS_PER_REGION_MAX, max_d + 1)
    if upper <= 0:
        return 0

    if upper == 1:
        return rng.choices([0, 1], weights=[1, 3])[0]

    options = [0, 1, 2]
    weights = [1, 2, 6]
    if upper >= 3:
        options.append(3)
        weights.append(7)

    options = [n for n in options if n <= upper]
    weights = weights[: len(options)]
    return int(rng.choices(options, weights=weights)[0])


def _place_markers(rect, marker_pool: list[int], rng: random.Random):
    r0, c0, r1, c1 = rect
    max_d = _max_cheb_depth(r1 - r0 + 1, c1 - c0 + 1)
    n = _choose_marker_count(max_d, rng)
    if n == 0:
        return []

    start_limit = min(START_DIST_MAX, max_d - n + 1)
    if start_limit < 0:
        return []
    start_d = rng.randint(0, start_limit)

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


def _stamp_markers(grid, rects, bg: int, marker_pool: list[int], rng: random.Random):
    used = set()
    infos = []
    for rect in rects:
        markers = _place_markers(rect, marker_pool, rng)
        for _d, col, rr, cc in markers:
            if not (rect[0] <= rr <= rect[2] and rect[1] <= cc <= rect[3]):
                return None, None
            if grid[rr, cc] != bg:
                return None, None
            grid[rr, cc] = col
            used.add(col)
        infos.append(
            {
                "rect": rect,
                "markers": markers,
                "max_depth": _max_cheb_depth(rect[2] - rect[0] + 1, rect[3] - rect[1] + 1),
            }
        )
    return used, infos


def _validate_color_order(grid, bg: int, wall: int) -> bool:
    vals, counts = np.unique(grid, return_counts=True)
    order = np.argsort(-counts)
    if int(vals[order[0]]) != bg:
        return False
    return len(order) >= 2 and int(vals[order[1]]) == wall


def _variant_metrics(grid: np.ndarray, infos) -> dict[str, int]:
    new_out = _solve_with_band_run(grid, BAND_RUN_LEN)
    legacy_out = _solve_with_band_run(grid, 1)
    diff_cells = int(np.count_nonzero(new_out != legacy_out))

    deep_regions = 0
    marked_regions = 0
    for info in infos:
        if info["markers"]:
            marked_regions += 1
            if info["max_depth"] >= 2 and len(info["markers"]) >= 2:
                deep_regions += 1

    return {
        "diff_cells": diff_cells,
        "deep_regions": deep_regions,
        "marked_regions": marked_regions,
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

    used, infos = _stamp_markers(grid, rects, bg, marker_pool, rng)
    if used is None or infos is None:
        return None
    if not (DISTINCT_MARKER_COLORS_MIN <= len(used) <= DISTINCT_MARKER_COLORS_MAX):
        return None
    if not _validate_color_order(grid, bg, wall):
        return None

    metrics = _variant_metrics(grid, infos)
    if metrics["diff_cells"] < MIN_DIFFERENCE_CELLS:
        return None
    if metrics["deep_regions"] < MIN_DEEP_REGIONS:
        return None
    if metrics["marked_regions"] < MIN_MARKED_REGIONS:
        return None

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
