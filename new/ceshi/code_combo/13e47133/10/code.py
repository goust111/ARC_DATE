# Combo variant of 13e47133 (index 10)
# combo_spec: D2/0+D10/0+D5/2
# Name: 周期性颜色+曼哈顿距离+标记擦除层

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np


def _ensure_arc_utils_on_path():
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        candidate = parent / "solvers" / "arc_utils.py"
        if candidate.exists():
            solvers_dir = str(candidate.parent)
            if solvers_dir not in sys.path:
                sys.path.append(solvers_dir)
            return
    raise ImportError("Could not locate solvers/arc_utils.py")


_ensure_arc_utils_on_path()

from arc_utils import components_mask

# Design analysis
# ---------------
# a) Concrete fused behavior and trace:
#    The original task read marker colors by Chebyshev depth and painted every
#    region layer with a one-step cyclic palette, so a two-color palette looked
#    like ABAB... on rectangular shells. The fused variant changes three linked
#    stages at once:
#    - depth is now Manhattan distance to the region boundary, so the shells are
#      taxicab diamonds and irregular regions expose 4-neighbor depth bends;
#    - the palette trajectory advances once every two layers, so a palette
#      [A, B, C] paints as A, A, B, B, C, C, A, A, ...;
#    - any absolute depth that contains a marker is erased back to the region
#      base color instead of being painted.
#    Small trace:
#      region palette cues by Manhattan depth: d=0 -> red, d=1 -> blue
#      original output by depth          : 0 red, 1 blue, 2 red, 3 blue, ...
#      fused output before erasing       : 0 red, 1 red, 2 blue, 3 blue, ...
#      fused output after erasing d=0,1  : 0 base, 1 base, 2 blue, 3 blue, ...
#    If the same region is L-shaped, the Manhattan depth-2 / depth-3 bands bend
#    around concave corners differently from the original Chebyshev rectangles.
#
# b) Complexity calibration:
#    A purely rectangular generator would hide the D10 change because Manhattan
#    and Chebyshev depths often agree there. A shallow generator would also hide
#    D2 and D5 because there would be no surviving layers after the erased ones.
#    The generator therefore forces at least one irregular component with
#    Manhattan-vs-Chebyshev gap cells, requires deep marked regions, and rejects
#    samples unless all three axes (metric, 2-layer cadence, erasure) produce a
#    visible cell-count difference.
#
# c) Implicit assumptions review:
#    - `bg = mode(grid)` and `wall = second mode` still require sparse markers and
#      relatively large regions; generation preserves and validates that.
#    - The old solver assumed 8-neighbor boundary detection and 8-neighbor BFS.
#      That assumption is invalid here, so both boundary seeding and expansion
#      must use `DIR4`.
#    - Palette extraction still assumes at most one meaningful marker per depth
#      after tie-breaking. The generator only places one marker on each used
#      Manhattan depth within a component.
#    - Erasure is keyed by absolute Manhattan depth, not by palette slot. This
#      means the solver must keep `dist_to_color` as a depth-indexed map rather
#      than erasing modulo positions.
#    - The new `(depth // 2)` trajectory assumes depth is a non-negative integer
#      boundary distance, which the BFS construction guarantees on every region.
#
# d) Placement legality:
#    Walls are drawn as connected axis-aligned segments. Marker placement samples
#    only cells already known to belong to a target component depth layer, so
#    markers never leave the region or overwrite walls. The featured region also
#    reserves at least two deeper layers after the last marker so the erased zone
#    does not consume the entire visible pattern.
#
# e) Visual observability:
#    A human can infer the rule because the same markers explain three visible
#    facts at once: their order still gives palette order, their depths carve
#    blank rings, and deeper surviving layers keep colors for two shells each.
#    Irregular regions make the Manhattan metric visible from shape alone; no
#    color-id arithmetic or hidden metadata is used.
#
# f) Visual cleanliness:
#    The scene still has only walls, base fill, and isolated markers. Region
#    count stays low, markers remain sparse, and only one wall network is drawn,
#    so the output is structured rather than cluttered.
#
# g) Color-change inferability:
#    Marker colors are never sorted by numeric value. They are read only by
#    boundary depth, and repeated use across regions gives several examples of
#    the same "two layers per palette step" cadence. Distinct marker colors stay
#    within the original 3-9 range so the palette remains learnable.
#
# h) Solver-generator contract after fusion:
#    `main()` assumes that every marked depth in a region corresponds to exactly
#    one intended marker color and that the dominant color in each free component
#    is its base/background. `generate_input()` guarantees those conditions by
#    using sparse single-cell markers, modal-color validation, and component-wise
#    depth-aware marker placement.

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]
BAND_RUN_LEN = 2


def _cc4(mask):
    return components_mask(mask, connectivity=4)


def _mode(arr):
    vals, counts = np.unique(np.asarray(arr), return_counts=True)
    return int(vals[counts.argmax()])


def _pts_to_mask(pts, h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    for r, c in pts:
        mask[int(r), int(c)] = True
    return mask


def _boundary_distance(component_mask: np.ndarray, boundary_dirs, step_dirs) -> np.ndarray:
    h, w = component_mask.shape
    dist = np.full((h, w), -1, dtype=int)
    q: list[tuple[int, int]] = []

    for r in range(h):
        for c in range(w):
            if not component_mask[r, c]:
                continue
            is_boundary = False
            for dr, dc in boundary_dirs:
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
        for dr, dc in step_dirs:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and component_mask[rr, cc] and dist[rr, cc] == -1:
                dist[rr, cc] = d + 1
                q.append((rr, cc))
    return dist


def _boundary_distance_manhattan(component_mask: np.ndarray) -> np.ndarray:
    return _boundary_distance(component_mask, DIR4, DIR4)


def _boundary_distance_chebyshev(component_mask: np.ndarray) -> np.ndarray:
    return _boundary_distance(component_mask, DIR8, DIR8)


def _solve(grid: np.ndarray, dist_fn, band_run_len: int, erase_marker_layers: bool) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    bg = _mode(g)
    other = g[g != bg]
    wall = bg if other.size == 0 else _mode(other)

    out = g.copy()
    free = g != wall

    for pts in _cc4(free):
        coords = sorted((int(r), int(c)) for r, c in pts)
        comp_mask = _pts_to_mask(coords, h, w)
        vals = g[comp_mask]
        base = _mode(vals)
        dist = dist_fn(comp_mask)

        dist_to_color: dict[int, int] = {}
        for r, c in coords:
            v = int(g[r, c])
            if v == base:
                continue
            d = int(dist[r, c])
            if d >= 0 and d not in dist_to_color:
                dist_to_color[d] = v

        if dist_to_color:
            palette: list[int] = []
            ordered_depths = sorted(dist_to_color)
            if ordered_depths[0] > 0:
                palette.append(int(base))
            for d in ordered_depths:
                palette.append(int(dist_to_color[d]))
        else:
            palette = [int(base)]

        erased_depths = set(dist_to_color) if erase_marker_layers else set()
        palette_len = len(palette)
        for r, c in coords:
            d = int(dist[r, c])
            if d in erased_depths:
                out[r, c] = base
            else:
                out[r, c] = palette[(d // band_run_len) % palette_len]

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repaint each wall-bounded region with Manhattan-depth diamond bands, two-layer palette runs, and blank marker layers.

    Input structure:
    - One wall color partitions the grid into several 4-connected non-wall regions.
    - Each region is mostly one base color and may contain a few isolated marker pixels of other colors.
    - Marker pixels sit on consecutive Manhattan-depth layers measured from that region's boundary.

    Rule:
    - For each region, compute Manhattan distance to the region boundary and read marker colors in increasing depth order to recover the palette, prepending the base color if the first marker is deeper than layer 0.
    - Paint surviving cells with `palette[(distance // 2) % len(palette)]`, so every palette entry lasts for two consecutive layers before the cycle advances.
    - If a depth layer contains a marker in the input, erase that entire absolute layer back to the base color instead of painting it; walls remain unchanged.

    Key insight:
    The markers simultaneously define palette order, reveal that depth is measured in taxicab shells rather than Chebyshev shells, and punch out their own layers, leaving repeated two-layer diamond bands interrupted by blank gaps.
    """
    return _solve(grid, _boundary_distance_manhattan, BAND_RUN_LEN, erase_marker_layers=True)


def _manhattan_single_run(grid: np.ndarray) -> np.ndarray:
    return _solve(grid, _boundary_distance_manhattan, 1, erase_marker_layers=True)


def _manhattan_no_erase(grid: np.ndarray) -> np.ndarray:
    return _solve(grid, _boundary_distance_manhattan, BAND_RUN_LEN, erase_marker_layers=False)


def _chebyshev_combo(grid: np.ndarray) -> np.ndarray:
    return _solve(grid, _boundary_distance_chebyshev, BAND_RUN_LEN, erase_marker_layers=True)


def _legacy_main(grid: np.ndarray) -> np.ndarray:
    return _solve(grid, _boundary_distance_chebyshev, 1, erase_marker_layers=False)


H_MIN, H_MAX = 16, 30
W_MIN, W_MAX = 16, 30
MIN_REGION_DIM = 3
MIN_REGION_CELLS = 22
N_REGIONS_MIN, N_REGIONS_MAX = 3, 6
MARKERS_PER_REGION_MAX = 3
MARKER_COUNT_WEIGHTS = [1, 2, 3, 4]
START_DIST_MAX = 1
DISTINCT_MARKER_COLORS_MIN, DISTINCT_MARKER_COLORS_MAX = 3, 9
GEN_MAX_ATTEMPTS = 500
EXTRA_SPLIT_PROB = 0.6

MIN_TOTAL_MARKERS = 4
MIN_MARKED_REGIONS = 2
MIN_COLORED_OUTPUT_CELLS = 12
MIN_BAND_DIFF_CELLS = 8
MIN_ERASE_DIFF_CELLS = 10
MIN_METRIC_DIFF_CELLS = 10
MIN_TOTAL_DIFF_CELLS = 18


def _draw_h(mask: np.ndarray, r: int, c0: int, c1: int):
    c0, c1 = sorted((int(c0), int(c1)))
    mask[int(r), c0:c1 + 1] = 1


def _draw_v(mask: np.ndarray, c: int, r0: int, r1: int):
    r0, r1 = sorted((int(r0), int(r1)))
    mask[r0:r1 + 1, int(c)] = 1


def _random_symmetry(mask: np.ndarray, rng: random.Random) -> np.ndarray:
    out = mask
    if rng.random() < 0.5:
        out = np.fliplr(out)
    if rng.random() < 0.5:
        out = np.flipud(out)
    if out.shape[0] == out.shape[1]:
        out = np.rot90(out, rng.randrange(4))
    else:
        out = np.rot90(out, rng.choice([0, 2]))
    return np.ascontiguousarray(out)


def _wall_has_no_dangling_pixels(mask: np.ndarray) -> bool:
    h, w = mask.shape
    for r, c in np.argwhere(mask == 1):
        if not any(0 <= r + dr < h and 0 <= c + dc < w and mask[r + dr, c + dc] == 1 for dr, dc in DIR4):
            return False
    return True


def _is_rect_component(pts) -> bool:
    rows = [int(r) for r, _ in pts]
    cols = [int(c) for _, c in pts]
    area = (max(rows) - min(rows) + 1) * (max(cols) - min(cols) + 1)
    return area == len(pts)


def _component_info(pts, h: int, w: int):
    coords = sorted((int(r), int(c)) for r, c in pts)
    mask = _pts_to_mask(coords, h, w)
    d4 = _boundary_distance_manhattan(mask)
    d8 = _boundary_distance_chebyshev(mask)

    cells_by_d: dict[int, list[tuple[int, int]]] = {}
    gap_cells = 0
    max_d = 0
    for r, c in coords:
        d = int(d4[r, c])
        cells_by_d.setdefault(d, []).append((r, c))
        max_d = max(max_d, d)
        if d != int(d8[r, c]):
            gap_cells += 1

    center_r = sum(r for r, _ in coords) / len(coords)
    center_c = sum(c for _, c in coords) / len(coords)
    return {
        "pts": coords,
        "d4": d4,
        "d8": d8,
        "cells_by_d": cells_by_d,
        "gap_cells": gap_cells,
        "max_d": max_d,
        "center_r": center_r,
        "center_c": center_c,
        "rect": _is_rect_component(coords),
    }


def _choose_marker_cells(info, depths, rng: random.Random, require_gap: bool):
    chosen: list[tuple[int, int]] = []
    used_gap = False
    d8 = info["d8"]

    for d in depths:
        candidates = info["cells_by_d"].get(d, [])
        if not candidates:
            return None

        separated = [
            p for p in candidates
            if all(max(abs(p[0] - q[0]), abs(p[1] - q[1])) > 1 for q in chosen)
        ]
        if separated:
            candidates = separated

        candidates = candidates[:]
        rng.shuffle(candidates)
        candidates.sort(
            key=lambda p: (
                0 if (require_gap and not used_gap and int(d8[p[0], p[1]]) != d) else 1,
                -abs(int(d8[p[0], p[1]]) - d),
                abs(p[0] - info["center_r"]) + abs(p[1] - info["center_c"]),
                p[0],
                p[1],
            )
        )

        pick = candidates[0]
        if int(d8[pick[0], pick[1]]) != d:
            used_gap = True
        chosen.append(pick)

    if require_gap and not used_gap:
        return None
    return chosen


def _pick_region_markers(info, rng: random.Random, marker_pool: list[int], force_feature: bool):
    max_d = int(info["max_d"])
    upper = min(MARKERS_PER_REGION_MAX, max_d + 1)
    if upper <= 0:
        return None if force_feature else []

    if force_feature:
        if max_d < 4 or upper < 2:
            return None
        counts = [3, 2] if upper >= 3 else [2]
        min_tail = 2
    else:
        counts = list(range(upper, -1, -1))
        rng.shuffle(counts)
        counts.sort(
            key=lambda n: (
                0 if n >= 2 and max_d >= n + 1 else 1,
                0 if n > 0 else 1,
                -n,
            )
        )
        min_tail = 1

    for n in counts:
        if n == 0:
            if not force_feature and rng.choices([0, 1], weights=[1, 4])[0] == 0:
                return []
            continue

        start_hi = min(START_DIST_MAX, max_d - (n - 1) - min_tail)
        if start_hi < 0:
            continue

        starts = list(range(start_hi + 1))
        if force_feature and 1 in starts:
            starts = [1, 0]
        elif rng.random() < 0.5:
            starts.reverse()

        for start_d in starts:
            depths = list(range(start_d, start_d + n))
            chosen = _choose_marker_cells(info, depths, rng, require_gap=force_feature)
            if chosen is None:
                continue
            colors = rng.sample(marker_pool, n)
            return [
                (depths[k], colors[k], chosen[k][0], chosen[k][1])
                for k in range(n)
            ]

    return None if force_feature else []


def _pick_palette(rng: random.Random):
    colors = list(range(10))
    bg = rng.choice(colors)
    colors.remove(bg)
    wall = rng.choice(colors)
    colors.remove(wall)
    return bg, wall, colors


def _validate_color_order(grid: np.ndarray, bg: int, wall: int) -> bool:
    vals, counts = np.unique(grid, return_counts=True)
    order = np.argsort(-counts)
    if int(vals[order[0]]) != bg:
        return False
    return len(order) >= 2 and int(vals[order[1]]) == wall


def _make_wall_layout(H: int, W: int, rng: random.Random):
    for _ in range(100):
        mask = np.zeros((H, W), dtype=int)

        rr = rng.randint(MIN_REGION_DIM, H - MIN_REGION_DIM - 1)
        cc = rng.randint(MIN_REGION_DIM, W - MIN_REGION_DIM - 1)
        c2_lo = cc + MIN_REGION_DIM + 1
        c2_hi = W - MIN_REGION_DIM - 1
        if c2_lo > c2_hi:
            continue

        _draw_h(mask, rr, 0, cc)
        _draw_v(mask, cc, 0, rr)
        _draw_v(mask, rng.randint(c2_lo, c2_hi), rr, H - 1)

        if rng.random() < EXTRA_SPLIT_PROB:
            split_cols = []
            for c1 in range(MIN_REGION_DIM, cc - MIN_REGION_DIM):
                left_width = c1
                middle_width = cc - c1 - 1
                if middle_width < MIN_REGION_DIM:
                    continue
                if rr * left_width < MIN_REGION_CELLS:
                    continue
                if rr * middle_width < MIN_REGION_CELLS:
                    continue
                split_cols.append(c1)
            if split_cols:
                _draw_v(mask, rng.choice(split_cols), 0, rr)

        mask = _random_symmetry(mask, rng)
        comps = _cc4(mask == 0)

        if not (N_REGIONS_MIN <= len(comps) <= N_REGIONS_MAX):
            continue
        if min(len(pts) for pts in comps) < MIN_REGION_CELLS:
            continue
        if not _wall_has_no_dangling_pixels(mask):
            continue

        infos = [_component_info(pts, *mask.shape) for pts in comps]
        if not any(not info["rect"] for info in infos):
            continue
        if not any(info["gap_cells"] > 0 and info["max_d"] >= 4 for info in infos):
            continue
        return mask, infos

    return None, None


def _stamp_markers(grid: np.ndarray, infos, bg: int, marker_pool: list[int], rng: random.Random):
    feature_candidates = [
        i for i, info in enumerate(infos)
        if info["gap_cells"] > 0 and info["max_d"] >= 4
    ]
    if not feature_candidates:
        return None, None
    feature_idx = rng.choice(feature_candidates)

    used_colors = set()
    markers_by_region: list[list[tuple[int, int, int, int]]] = []

    for i, info in enumerate(infos):
        markers = _pick_region_markers(info, rng, marker_pool, force_feature=(i == feature_idx))
        if markers is None:
            return None, None
        markers_by_region.append(markers)
        for _d, col, r, c in markers:
            if grid[r, c] != bg:
                return None, None
            grid[r, c] = col
            used_colors.add(col)

    featured_markers = markers_by_region[feature_idx]
    if not featured_markers:
        return None, None
    if not any(infos[feature_idx]["d4"][r, c] != infos[feature_idx]["d8"][r, c] for _d, _col, r, c in featured_markers):
        return None, None
    return used_colors, markers_by_region


def _variant_metrics(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    bg = _mode(g)
    other = g[g != bg]
    wall = bg if other.size == 0 else _mode(other)
    free = g != wall

    combo = main(g)
    run1 = _manhattan_single_run(g)
    no_erase = _manhattan_no_erase(g)
    cheb_combo = _chebyshev_combo(g)
    legacy = _legacy_main(g)

    return {
        "combo": combo,
        "band_diff_cells": int(np.count_nonzero(free & (combo != run1))),
        "erase_diff_cells": int(np.count_nonzero(free & (combo != no_erase))),
        "metric_diff_cells": int(np.count_nonzero(free & (combo != cheb_combo))),
        "total_diff_cells": int(np.count_nonzero(free & (combo != legacy))),
        "erased_cells": int(np.count_nonzero(free & (combo == bg) & (no_erase != bg))),
        "colored_cells": int(np.count_nonzero(free & (combo != bg))),
    }


def _gen_attempt(rng: random.Random):
    H = rng.randint(H_MIN, H_MAX)
    W = rng.randint(W_MIN, W_MAX)
    bg, wall, marker_pool = _pick_palette(rng)

    wall_mask, infos = _make_wall_layout(H, W, rng)
    if wall_mask is None or infos is None:
        return None

    grid = np.full(wall_mask.shape, bg, dtype=int)
    grid[wall_mask == 1] = wall

    used_colors, markers_by_region = _stamp_markers(grid, infos, bg, marker_pool, rng)
    if used_colors is None or markers_by_region is None:
        return None

    total_markers = sum(len(markers) for markers in markers_by_region)
    marked_regions = sum(1 for markers in markers_by_region if markers)
    if total_markers < MIN_TOTAL_MARKERS or marked_regions < MIN_MARKED_REGIONS:
        return None

    if not (DISTINCT_MARKER_COLORS_MIN <= len(used_colors) <= DISTINCT_MARKER_COLORS_MAX):
        return None
    if not _validate_color_order(grid, bg, wall):
        return None

    try:
        metrics = _variant_metrics(grid)
    except Exception:
        return None

    combo = metrics["combo"]
    if np.array_equal(grid, combo):
        return None
    if metrics["colored_cells"] < MIN_COLORED_OUTPUT_CELLS:
        return None
    if metrics["erased_cells"] < MIN_ERASE_DIFF_CELLS:
        return None
    if metrics["band_diff_cells"] < MIN_BAND_DIFF_CELLS:
        return None
    if metrics["erase_diff_cells"] < MIN_ERASE_DIFF_CELLS:
        return None
    if metrics["metric_diff_cells"] < MIN_METRIC_DIFF_CELLS:
        return None
    if metrics["total_diff_cells"] < MIN_TOTAL_DIFF_CELLS:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_MAX_ATTEMPTS} attempts")
