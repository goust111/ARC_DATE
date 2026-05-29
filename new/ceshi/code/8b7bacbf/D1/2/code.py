# Variant of 8b7bacbf
# Dimension: D1 — 路径追踪连通性 (path)
# Change: 8连通追踪（path_tracing）——从边缘种子像素出发，沿8连通（含对角线）的非背景像素追踪出完整线路组件... → 有限步数追踪
#
# Design analysis
# ---
# variant_spec:
#   visual_objects:
#     bg: most common color, as in the source task
#     seed: one unique-color pixel placed at a corner
#     line: one 8-connected 1px path leaving the seed into the grid interior
#     meter:
#       role: encodes the maximum tracing distance N
#       shape: a straight same-color bar on the border, attached to the seed on the border side not used by the line
#       decoding: N = bar length excluding the seed pixel itself
#       constraint: length is always 2-5, so it is visually obvious and distinguishable from the line stub
#     frame_loops:
#       role: same as the source task, single-color rings enclosing bg cavities
#       change: some loops attached to the same seed-line are intentionally farther than N steps from the seed
#   behavior_trace:
#     source_task:
#       - detect a unique border seed
#       - follow the entire 8-connected non-bg component from that seed
#       - fill every cavity whose ring is touched by that traced component and is adjacent to the seed's line
#     variant:
#       - detect the meter bar next to the seed and decode N from its length
#       - BFS over the same non-bg component, but stop expansion after N graph steps from the seed
#       - only fill cavities whose ring is touched inside this limited-radius traced set
#     concrete_example:
#       - seed at top-left, line enters downward, meter extends rightward for N=3
#       - loop A touches the line after 2 steps -> filled
#       - loop B touches the same line after 6 steps -> not filled
#       - under the original solver both A and B would fill; under the variant only A fills
#   complexity_calibration:
#     - the meter uses only a small discrete range 2-5, preventing overfitting to arbitrary distances
#     - examples contain both "near" and "far" attached loops so the distance limit is visible, not degenerate
#     - the source task's detached distractor loops are kept in reduced form, but the main new distractor is now "same line, too far"
#   implicit_assumptions_review:
#     original_main:
#       - assumption: the seed has exactly one meaningful non-bg neighbor color, namely the line color
#         status: false in the variant because the seed also touches a meter bar; solver must explicitly distinguish line vs meter
#       - assumption: tracing the full non-bg component is always correct
#         status: false in the variant; tracing must become shortest-path BFS capped at N steps
#       - assumption: any cavity ring hit anywhere on the connected component should fill
#         status: false in the variant; only rings touched inside the limited trace should fill
#     new_main_assumptions:
#       - each generated seed sits at a corner and has exactly two orthogonal non-bg branches: one line stub and one meter bar
#       - the meter bar is straight along the border and has length >= 2, while the line leaves the border immediately after its first stub pixel
#       - generator enforces enough path length to support at least one near and one far attached loop
#   placement_legality:
#     - meter and line occupy the two available corner-adjacent border directions, so they are unambiguous and non-overlapping
#     - line growth forbids 8-adjacency to foreign occupied pixels, avoiding accidental mergers between seed systems
#     - attached frames are still placed with empty cavity/ring cells and at least one 8-adjacency to the intended line
#   visual_observability:
#     - the border bar is easy to count by eye
#     - the output difference is local and repeated: early cavities fill, later cavities on the same path stay blank
#     - multi-seed examples may show different N values simultaneously, reinforcing the rule without using hidden color semantics
#   cleanliness:
#     - distractor networks are simplified to detached loops to keep the new signal legible
#     - meters stay on the border and never overlap the line or frames, so each role is spatially separated

from __future__ import annotations

from collections import deque
import os
import random
import sys

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
if _ROOT not in sys.path:
    sys.path.append(_ROOT)

from arc_utils import components_mask, get_bg, neighbors4, neighbors8


def _components_mask(mask: np.ndarray, conn8: bool) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8 if conn8 else 4)]


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _seed_pixels(g: np.ndarray, bg: int) -> list[tuple[int, int]]:
    h, w = g.shape
    vals, cnts = np.unique(g, return_counts=True)
    seeds: list[tuple[int, int]] = []
    for v, cnt in zip(vals, cnts):
        v = int(v)
        if v == bg or int(cnt) != 1:
            continue
        r, c = map(int, np.argwhere(g == v)[0])
        if r == 0 or c == 0 or r == h - 1 or c == w - 1:
            seeds.append((r, c))
    return seeds


def _border_run_length(g: np.ndarray, seed: tuple[int, int], nbr: tuple[int, int], bg: int) -> int:
    h, w = g.shape
    sr, sc = seed
    nr, nc = nbr
    col = int(g[nr, nc])
    if col == bg:
        return 0
    dr = nr - sr
    dc = nc - sc
    run = 0
    r, c = nr, nc
    while _in_bounds(h, w, r, c) and int(g[r, c]) == col and (r == 0 or c == 0 or r == h - 1 or c == w - 1):
        run += 1
        r += dr
        c += dc
    return run


def _seed_info(g: np.ndarray, seed: tuple[int, int], bg: int) -> dict[str, int | tuple[int, int] | None]:
    h, w = g.shape
    sr, sc = seed
    nonbg_neighbors = []
    for nr, nc in neighbors4(sr, sc):
        if _in_bounds(h, w, nr, nc) and int(g[nr, nc]) != bg:
            nonbg_neighbors.append((nr, nc))
    assert nonbg_neighbors, "seed pixel must have at least one non-bg neighbor"

    if len(nonbg_neighbors) == 1:
        line_pt = nonbg_neighbors[0]
        return {
            "seed_col": int(g[sr, sc]),
            "line_col": int(g[line_pt]),
            "line_pt": line_pt,
            "step_limit": None,
        }

    runs = [(_border_run_length(g, seed, nbr, bg), nbr) for nbr in nonbg_neighbors]
    runs.sort(key=lambda x: x[0], reverse=True)
    best_run, meter_pt = runs[0]
    if best_run < 2:
        line_pt = nonbg_neighbors[0]
        return {
            "seed_col": int(g[sr, sc]),
            "line_col": int(g[line_pt]),
            "line_pt": line_pt,
            "step_limit": None,
        }

    line_candidates = [nbr for nbr in nonbg_neighbors if nbr != meter_pt]
    assert line_candidates, "meter bar cannot be the seed's only branch"
    line_pt = line_candidates[0]
    return {
        "seed_col": int(g[sr, sc]),
        "line_col": int(g[line_pt]),
        "line_pt": line_pt,
        "step_limit": int(best_run),
    }


def _trace_nonbg_component8_from(
    g: np.ndarray, start: tuple[int, int], bg: int, step_limit: int | None
) -> set[tuple[int, int]]:
    h, w = g.shape
    q = deque([(start, 0)])
    dist = {start: 0}
    while q:
        (r, c), d = q.popleft()
        if step_limit is not None and d >= step_limit:
            continue
        for nr, nc in neighbors8(r, c):
            if not _in_bounds(h, w, nr, nc):
                continue
            if (nr, nc) in dist or int(g[nr, nc]) == bg:
                continue
            dist[(nr, nc)] = d + 1
            q.append(((nr, nc), d + 1))
    return set(dist)


def _cavities(g: np.ndarray, bg: int) -> list[tuple[set[tuple[int, int]], set[tuple[int, int]], set[int]]]:
    h, w = g.shape
    bg_comps = _components_mask(g == bg, conn8=False)
    regions = []
    for pts in bg_comps:
        region = set(pts)
        boundary = set()
        for r, c in region:
            for nr, nc in neighbors4(r, c):
                if _in_bounds(h, w, nr, nc) and (nr, nc) not in region and int(g[nr, nc]) != bg:
                    boundary.add((nr, nc))
        regions.append((region, boundary))

    corners = {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}
    outside_idxs = set()
    for i, (reg, _) in enumerate(regions):
        if any(p in reg for p in corners if int(g[p]) == bg):
            outside_idxs.add(i)

    cavities = []
    for i, (reg, bnd) in enumerate(regions):
        if i in outside_idxs or not bnd:
            continue
        bcols = {int(g[p]) for p in bnd}
        cavities.append((reg, bnd, bcols))
    return cavities


def _boundary_is_attached_to_line(
    g: np.ndarray, boundary: set[tuple[int, int]], traced: set[tuple[int, int]], line_col: int
) -> bool:
    h, w = g.shape
    boundary_in_traced = boundary & traced
    if not boundary_in_traced:
        return False
    for r, c in boundary_in_traced:
        for nr, nc in neighbors8(r, c):
            if _in_bounds(h, w, nr, nc) and int(g[nr, nc]) == line_col:
                return True
    return False


def _solve_core(grid: np.ndarray, use_step_limit: bool) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()
    cavities = _cavities(g, bg)

    for seed in _seed_pixels(g, bg):
        info = _seed_info(g, seed, bg)
        step_limit = info["step_limit"] if use_step_limit else None
        traced = _trace_nonbg_component8_from(g, seed, bg, step_limit)
        line_col = int(info["line_col"])
        seed_col = int(info["seed_col"])

        for cav, boundary, bcols in cavities:
            if len(bcols) != 1:
                continue
            if not _boundary_is_attached_to_line(g, boundary, traced, line_col):
                continue
            for r, c in cav:
                if int(out[r, c]) == bg:
                    out[r, c] = seed_col

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Variant rule:
    - Each unique border seed still owns one 8-connected line+frame structure.
    - A straight border bar next to the seed encodes N.
    - Trace only non-bg pixels within N BFS steps from the seed.
    - Fill only cavities whose frame is touched inside that limited trace and is adjacent to the seed's line.
    """
    return _solve_core(grid, use_step_limit=True)


# ========================== CFG ==========================
GRID_H_RANGE = (10, 30)
GRID_W_RANGE = (10, 30)
BG_COLORS = [0, 5]

SEED_COUNT_CHOICES = [1, 2, 2]
METER_LEN_RANGE = (2, 5)
LINE_LEN_RANGE = (10, 28)
MIN_LINE_LEN = 8

FRAME_COLORS_MAX = 3
CAVITY_SIZE_WEIGHTS = [2, 2, 2, 3]
DISTRACTOR_CAVITY_WEIGHTS = [2, 2, 3]
FRAME_ATTACH_TRIES = 60
DISTRACTOR_FRAME_TRIES = 40
FRAME_PLACEMENT_PAD = 1

DIAG_WEIGHT_ALIGNED = 5
DIAG_WEIGHT_OTHER = 1
ORTHO_WEIGHT_ALIGNED = 2
ORTHO_WEIGHT_OTHER = 1
BORDER_AVOIDANCE_PROB = 0.9
BIAS_FLIP_PROB = 0.3

DETACHED_DISTRACTOR_MAX = 2
GEN_RETRIES = 400
# =========================================================

DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
ORTHO_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _frame_ring_coords(r0: int, c0: int, ch: int, cw: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    h, w = ch + 2, cw + 2
    ring = []
    for dr in range(h):
        for dc in range(w):
            on_bdr = dr == 0 or dr == h - 1 or dc == 0 or dc == w - 1
            corner = dr in (0, h - 1) and dc in (0, w - 1)
            if on_bdr and not corner:
                ring.append((r0 + dr, c0 + dc))
    cavity = [(r0 + 1 + dr, c0 + 1 + dc) for dr in range(ch) for dc in range(cw)]
    return ring, cavity


def _validate_frame_placement(
    canvas: np.ndarray, ring: list[tuple[int, int]], cavity: list[tuple[int, int]], line_col: int, h: int, w: int
) -> bool:
    for rr, cc in ring:
        if canvas[rr, cc] != -1:
            return False
    for rr, cc in cavity:
        if canvas[rr, cc] != -1:
            return False

    for cr, cc in cavity:
        for nr, nc in neighbors8(cr, cc):
            if _in_bounds(h, w, nr, nc) and canvas[nr, nc] == line_col:
                return False

    ring_set = set(ring)
    for rr, cc in ring:
        for nr, nc in neighbors8(rr, cc):
            if not _in_bounds(h, w, nr, nc) or (nr, nc) in ring_set:
                continue
            v = canvas[nr, nc]
            if v != -1 and v != line_col:
                return False

    for rr, cc in ring:
        for nr, nc in neighbors8(rr, cc):
            if _in_bounds(h, w, nr, nc) and canvas[nr, nc] == line_col:
                return True
    return False


def _weighted_dirs(dr_bias: int, dc_bias: int) -> list[tuple[int, int]]:
    dirs = []
    for ddr, ddc in DIAG_DIRS:
        aligned = ddr == dr_bias or ddc == dc_bias
        w_d = DIAG_WEIGHT_ALIGNED if aligned else DIAG_WEIGHT_OTHER
        dirs.extend([(ddr, ddc)] * w_d)
    for ddr, ddc in ORTHO_DIRS:
        aligned = (ddr == dr_bias and ddc == 0) or (ddc == dc_bias and ddr == 0)
        w_d = ORTHO_WEIGHT_ALIGNED if aligned else ORTHO_WEIGHT_OTHER
        dirs.extend([(ddr, ddc)] * w_d)
    return dirs


def _grow_line_from_prefix(
    grid: np.ndarray, prefix: list[tuple[int, int]], line_col: int, target_len: int, rng: random.Random
) -> list[tuple[int, int]]:
    h, w = grid.shape
    path = list(prefix)
    cur = path[-1]
    path_set = set(path)
    dr_bias = 1 if cur[0] < h // 2 else -1
    dc_bias = 1 if cur[1] < w // 2 else -1

    while len(path) < target_len:
        dirs = _weighted_dirs(dr_bias, dc_bias)
        rng.shuffle(dirs)
        placed = False
        for ddr, ddc in dirs:
            nr, nc = cur[0] + ddr, cur[1] + ddc
            if not _in_bounds(h, w, nr, nc) or grid[nr, nc] != -1:
                continue
            if any((nnr, nnc) != cur and (nnr, nnc) in path_set for nnr, nnc in neighbors8(nr, nc)):
                continue
            foreign_touch = False
            for nnr, nnc in neighbors8(nr, nc):
                if not _in_bounds(h, w, nnr, nnc):
                    continue
                if (nnr, nnc) == cur:
                    continue
                if grid[nnr, nnc] != -1:
                    foreign_touch = True
                    break
            if foreign_touch:
                continue
            if (nr == 0 or nc == 0 or nr == h - 1 or nc == w - 1) and rng.random() < BORDER_AVOIDANCE_PROB:
                continue
            cur = (nr, nc)
            path.append(cur)
            path_set.add(cur)
            grid[nr, nc] = line_col
            placed = True
            break
        if not placed:
            if rng.random() < BIAS_FLIP_PROB:
                dr_bias *= -1
            if rng.random() < BIAS_FLIP_PROB:
                dc_bias *= -1
            break
    return path


def _corner_layout(
    corner_kind: str, h: int, w: int, meter_len: int, meter_horizontal: bool
) -> tuple[tuple[int, int], list[tuple[int, int]], list[tuple[int, int]]]:
    if corner_kind == "TL":
        seed = (0, 0)
        if meter_horizontal:
            meter = [(0, c) for c in range(1, meter_len + 1)]
            prefix = [(1, 0), (2, 1)]
        else:
            meter = [(r, 0) for r in range(1, meter_len + 1)]
            prefix = [(0, 1), (1, 2)]
    elif corner_kind == "TR":
        seed = (0, w - 1)
        if meter_horizontal:
            meter = [(0, c) for c in range(w - 2, w - 2 - meter_len, -1)]
            prefix = [(1, w - 1), (2, w - 2)]
        else:
            meter = [(r, w - 1) for r in range(1, meter_len + 1)]
            prefix = [(0, w - 2), (1, w - 3)]
    elif corner_kind == "BL":
        seed = (h - 1, 0)
        if meter_horizontal:
            meter = [(h - 1, c) for c in range(1, meter_len + 1)]
            prefix = [(h - 2, 0), (h - 3, 1)]
        else:
            meter = [(r, 0) for r in range(h - 2, h - 2 - meter_len, -1)]
            prefix = [(h - 1, 1), (h - 2, 2)]
    elif corner_kind == "BR":
        seed = (h - 1, w - 1)
        if meter_horizontal:
            meter = [(h - 1, c) for c in range(w - 2, w - 2 - meter_len, -1)]
            prefix = [(h - 2, w - 1), (h - 3, w - 2)]
        else:
            meter = [(r, w - 1) for r in range(h - 2, h - 2 - meter_len, -1)]
            prefix = [(h - 1, w - 2), (h - 2, w - 3)]
    else:
        raise ValueError(corner_kind)
    return seed, meter, prefix


def _halo_offsets(fh: int, fw: int) -> list[tuple[int, int]]:
    return [(dr, dc) for dr in range(-1, fh + 1) for dc in range(-1, fw + 1) if not (0 <= dr < fh and 0 <= dc < fw)]


def _try_attach_frame(
    canvas: np.ndarray,
    line_points: list[tuple[int, int]],
    line_col: int,
    frame_col: int,
    rng: random.Random,
    max_tries: int = FRAME_ATTACH_TRIES,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    if not line_points:
        return None
    h, w = canvas.shape
    pad = FRAME_PLACEMENT_PAD
    for _ in range(max_tries):
        ch = rng.choice(CAVITY_SIZE_WEIGHTS)
        cw = rng.choice(CAVITY_SIZE_WEIGHTS)
        lp = rng.choice(line_points)
        fh, fw = ch + 2, cw + 2
        offsets = _halo_offsets(fh, fw)
        rng.shuffle(offsets)
        for dr, dc in offsets:
            r0 = lp[0] - dr
            c0 = lp[1] - dc
            if r0 < pad or c0 < pad or r0 + fh > h - pad or c0 + fw > w - pad:
                continue
            ring, cavity = _frame_ring_coords(r0, c0, ch, cw)
            if not _validate_frame_placement(canvas, ring, cavity, line_col, h, w):
                continue
            for rr, cc in ring:
                canvas[rr, cc] = frame_col
            return ring, cavity
    return None


def _try_attach_distractor_frame(canvas: np.ndarray, frame_col: int, rng: random.Random) -> bool:
    h, w = canvas.shape
    for _ in range(DISTRACTOR_FRAME_TRIES):
        ch = rng.choice(DISTRACTOR_CAVITY_WEIGHTS)
        cw = rng.choice(DISTRACTOR_CAVITY_WEIGHTS)
        fh, fw = ch + 2, cw + 2
        r0 = rng.randint(1, h - fh - 1)
        c0 = rng.randint(1, w - fw - 1)
        ok = True
        for rr in range(r0 - 1, r0 + fh + 1):
            for cc in range(c0 - 1, c0 + fw + 1):
                if _in_bounds(h, w, rr, cc) and canvas[rr, cc] != -1:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue
        ring, _ = _frame_ring_coords(r0, c0, ch, cw)
        for rr, cc in ring:
            canvas[rr, cc] = frame_col
        return True
    return False


def _seed_variant_stats(grid: np.ndarray) -> list[dict[str, int]]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    cavities = _cavities(g, bg)
    stats = []
    for seed in _seed_pixels(g, bg):
        info = _seed_info(g, seed, bg)
        full_trace = _trace_nonbg_component8_from(g, seed, bg, None)
        limited_trace = _trace_nonbg_component8_from(g, seed, bg, int(info["step_limit"]))
        full_attached = 0
        limited_attached = 0
        for _cav, boundary, bcols in cavities:
            if len(bcols) != 1:
                continue
            if _boundary_is_attached_to_line(g, boundary, full_trace, int(info["line_col"])):
                full_attached += 1
                if _boundary_is_attached_to_line(g, boundary, limited_trace, int(info["line_col"])):
                    limited_attached += 1
        stats.append(
            {
                "limit": int(info["step_limit"]) if info["step_limit"] is not None else 10**9,
                "full_attached": full_attached,
                "limited_attached": limited_attached,
            }
        )
    return stats


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    bg = rng.choice(BG_COLORS)
    palette = [c for c in range(10) if c != bg]
    rng.shuffle(palette)

    num_seeds = rng.choice(SEED_COUNT_CHOICES)
    if 3 * num_seeds >= len(palette):
        return None

    canvas = -np.ones((h, w), dtype=int)

    seed_colors = [palette.pop() for _ in range(num_seeds)]
    line_colors = [palette.pop() for _ in range(num_seeds)]
    meter_colors = [palette.pop() for _ in range(num_seeds)]

    max_frame_cols = min(FRAME_COLORS_MAX, len(palette))
    if max_frame_cols <= 0:
        return None
    num_frame_cols = rng.randint(1, max_frame_cols)
    frame_colors_pool = [palette.pop() for _ in range(num_frame_cols)]

    corners = ["TL", "TR", "BL", "BR"]
    rng.shuffle(corners)
    corners_used = corners[:num_seeds]

    lines_info = []
    for i in range(num_seeds):
        meter_len = rng.randint(*METER_LEN_RANGE)
        seed_rc, meter_cells, prefix = _corner_layout(corners_used[i], h, w, meter_len, rng.choice([False, True]))
        all_cells = [seed_rc] + meter_cells + prefix
        if any(not _in_bounds(h, w, r, c) for r, c in all_cells):
            return None
        if any(canvas[r, c] != -1 for r, c in all_cells):
            return None

        canvas[seed_rc] = seed_colors[i]
        for r, c in meter_cells:
            canvas[r, c] = meter_colors[i]
        for r, c in prefix:
            canvas[r, c] = line_colors[i]

        lo = max(LINE_LEN_RANGE[0], meter_len + 7)
        hi = min(LINE_LEN_RANGE[1], h + w)
        if lo > hi:
            return None
        target_len = rng.randint(lo, hi)
        path = _grow_line_from_prefix(canvas, prefix, line_colors[i], target_len, rng)
        if len(path) < max(MIN_LINE_LEN, meter_len + 5):
            return None
        lines_info.append(
            {
                "seed_col": seed_colors[i],
                "seed_rc": seed_rc,
                "line_col": line_colors[i],
                "meter_col": meter_colors[i],
                "meter_len": meter_len,
                "path": path,
            }
        )

    for info in lines_info:
        path = info["path"]
        meter_len = info["meter_len"]
        early_points = path[1 : min(len(path), meter_len + 2)]
        late_start = meter_len + 4
        late_points = path[late_start:] if late_start < len(path) else []
        if not early_points or not late_points:
            return None

        near_fc = rng.choice(frame_colors_pool)
        far_fc = rng.choice(frame_colors_pool)
        if _try_attach_frame(canvas, early_points, info["line_col"], near_fc, rng) is None:
            return None
        if _try_attach_frame(canvas, late_points, info["line_col"], far_fc, rng) is None:
            return None

        extra_loops = rng.randint(0, 2)
        for _ in range(extra_loops):
            if rng.random() < 0.5:
                candidate_points = path[1 : min(len(path), meter_len + 3)]
            else:
                candidate_points = path[max(2, meter_len + 3) :]
            _try_attach_frame(canvas, candidate_points, info["line_col"], rng.choice(frame_colors_pool), rng)

    for _ in range(rng.randint(0, DETACHED_DISTRACTOR_MAX)):
        _try_attach_distractor_frame(canvas, rng.choice(frame_colors_pool), rng)

    grid = np.where(canvas == -1, bg, canvas).astype(int)

    vals, cnts = np.unique(grid, return_counts=True)
    if int(vals[np.argmax(cnts)]) != bg:
        return None
    for info in lines_info:
        if int((grid == info["seed_col"]).sum()) != 1:
            return None

    try:
        out = main(grid)
        full_out = _solve_core(grid, use_step_limit=False)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None
    if np.array_equal(out, full_out):
        return None

    stats = _seed_variant_stats(grid)
    if not stats:
        return None
    if any(s["limited_attached"] < 1 for s in stats):
        return None
    if sum(s["full_attached"] for s in stats) <= sum(s["limited_attached"] for s in stats):
        return None
    if max(s["full_attached"] - s["limited_attached"] for s in stats) < 1:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError("generate_input: exhausted attempts")
