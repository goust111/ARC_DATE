# Combo variant of 8b7bacbf (index 1)
# combo_spec: D1/0+D10/0
# Name: 4连通追踪+不规则腔体
#
# Design analysis
# a) Concrete behavior and trace:
#    - Original task: a border seed reaches a non-background drawing through 8-connectivity; the reachable
#      drawing includes chamfered rectangular loops, and the bg pocket inside each reachable loop is filled.
#    - Combo task: the seed now reaches only a 4-connected orthogonal network, and the fillable pockets are
#      irregular polyomino cavities instead of tiny rectangles.
#    - Small trace:
#      1. Original: S -> diagonal-friendly line -> rounded ring, so a corner-touch loop is still reachable.
#      2. Combo: S -> orthogonal line -> irregular loop. If the loop is reachable only through a diagonal-only
#         connector pixel, 4-connected tracing does not enter it, so its irregular cavity stays bg. If the loop
#         shares an orthogonal contact with the traced line, its cavity is filled with the seed color.
#      3. The visible difference is therefore twofold: the lines are orthogonal, and the filled regions are
#         non-rectangular cavities. A diagonal-only irregular probe loop makes the connectivity change observable.
# b) Complexity calibration:
#    - The cavity family is not fully random; it is sampled from a small template bank (L, T, zigzag, staircase,
#      hook, fat-L) with rotations and mirrors. This keeps the irregularity obvious but still learnable.
#    - Every accepted grid contains at least one fillable irregular cavity and one diagonal-only irregular probe
#      reached only through a single corner-connected connector pixel, so both changed dimensions have repeated
#      visible support.
# c) Implicit-assumption audit:
#    - Original `_nonbg_component8_from(...)` assumed every meaningful attachment may use diagonal contact.
#      That assumption is invalid here, so tracing is parameterized and `main()` uses 4-connectivity.
#    - Original line-to-loop witness used 8-neighbor adjacency. In the combo, a fillable loop must have an
#      orthogonal witness to the line color; diagonal witness alone is reserved for the probe distractor.
#    - D10's single-variant used a 4-neighbor shell around an irregular cavity. Under 4-connected tracing that
#      shell often breaks into multiple pieces, so it cannot be reused directly. The combo instead uses the
#      cavity's 8-neighbor shell as the frame ring; this keeps one enclosed cavity while making the ring itself
#      4-connected and therefore traceable.
#    - Outside detection is upgraded from "background corner component" to "any bg region touching the border",
#      because irregular rings can create exterior pockets away from corners.
# d) Placement legality:
#    - Ring cells and cavity cells must be empty, in-bounds, and separated from foreign colors by an 8-neighbor
#      buffer.
#    - No cavity cell may be 8-neighbor to the owning line color, so the cavity boundary is truly the frame.
#    - Fillable loops require at least one 4-neighbor ring-to-line contact. Probe loops require at least one
#      diagonal ring-to-line contact and zero orthogonal contacts.
# e) Visual observability:
#    - A solver looking at 2-4 examples can see that only orthogonally attached loops fill; corner-touch loops do
#      not. The filled regions are also visibly L/T/zigzag-like instead of rectangles.
# f) Visual cleanliness:
#    - The line is 1px wide and self-avoiding with 8-neighbor clearance except for the intentional diagonal-only
#      probe connector.
#    - Rings keep a buffer from unrelated structures, so the scene stays readable despite irregular shapes.
# g) Color inference:
#    - Color values carry no ordinal meaning. The only stable mapping is that each unique border seed recolors the
#      cavities of its own reachable orthogonal network.
# h) Anti-degeneration checks:
#    - The generator rejects grids where the new solver matches a legacy 8-connected solver, which guarantees an
#      active diagonal-only probe path.
#    - It also rejects grids where any fillable cavity is rectangular, which guarantees the irregular-cavity
#      control variable is active in the examples.
# i) New solver assumptions guaranteed by the generator:
#    - Each seed has at least one non-background 4-neighbor.
#    - Every intended fillable loop has a single-color 4-connected ring and a single enclosed 4-connected cavity.
#    - Distinct seed-owned networks remain 8-separated, preventing ambiguous ownership.

from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import components_mask, get_bg, neighbors4, neighbors8


def _components_mask(mask: np.ndarray, conn8: bool) -> list[list[tuple[int, int]]]:
    connectivity = 8 if conn8 else 4
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=connectivity)]


def _touches_border(pts: list[tuple[int, int]] | set[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts)


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


def _nonbg_component_from(
    g: np.ndarray,
    start: tuple[int, int],
    bg: int,
    conn8: bool,
) -> set[tuple[int, int]]:
    h, w = g.shape
    neigh_fn = neighbors8 if conn8 else neighbors4
    st = [start]
    vis = {start}
    while st:
        r, c = st.pop()
        for nr, nc in neigh_fn(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in vis and int(g[nr, nc]) != bg:
                vis.add((nr, nc))
                st.append((nr, nc))
    return vis


def _bg_regions_with_boundaries(
    g: np.ndarray,
    bg: int,
) -> list[tuple[set[tuple[int, int]], set[tuple[int, int]], bool]]:
    h, w = g.shape
    regions = []
    for pts in _components_mask(g == bg, conn8=False):
        region = set(pts)
        boundary = set()
        for r, c in region:
            for nr, nc in neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in region and int(g[nr, nc]) != bg:
                    boundary.add((nr, nc))
        regions.append((region, boundary, _touches_border(region, h, w)))
    return regions


def _line_color_for_seed(g: np.ndarray, sr: int, sc: int, bg: int) -> int:
    h, w = g.shape
    neigh_cols = []
    for nr, nc in neighbors4(sr, sc):
        if 0 <= nr < h and 0 <= nc < w:
            v = int(g[nr, nc])
            if v != bg:
                neigh_cols.append(v)
    assert neigh_cols, "seed pixel must have at least one non-bg 4-neighbor"
    return Counter(neigh_cols).most_common(1)[0][0]


def _fillable_region_keys(g: np.ndarray, bg: int, trace_conn8: bool, line_adj8: bool) -> set[frozenset[tuple[int, int]]]:
    h, w = g.shape
    neigh_line = neighbors8 if line_adj8 else neighbors4
    cavities = []
    for region, boundary, touches_border in _bg_regions_with_boundaries(g, bg):
        if touches_border or not boundary:
            continue
        if len({int(g[p]) for p in boundary}) != 1:
            continue
        cavities.append((region, boundary))

    filled: set[frozenset[tuple[int, int]]] = set()
    for sr, sc in _seed_pixels(g, bg):
        traced = _nonbg_component_from(g, (sr, sc), bg, conn8=trace_conn8)
        line_col = _line_color_for_seed(g, sr, sc, bg)
        for region, boundary in cavities:
            boundary_in_traced = boundary & traced
            if not boundary_in_traced:
                continue
            ok = False
            for r, c in boundary_in_traced:
                for nr, nc in neigh_line(r, c):
                    if 0 <= nr < h and 0 <= nc < w and int(g[nr, nc]) == line_col:
                        ok = True
                        break
                if ok:
                    break
            if ok:
                filled.add(frozenset(region))
    return filled


def _solve_with_trace_mode(grid: np.ndarray, trace_conn8: bool, line_adj8: bool) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    out = g.copy()
    neigh_line = neighbors8 if line_adj8 else neighbors4
    cavities = []
    for region, boundary, touches_border in _bg_regions_with_boundaries(g, bg):
        if touches_border or not boundary:
            continue
        if len({int(g[p]) for p in boundary}) != 1:
            continue
        cavities.append((region, boundary))

    for sr, sc in _seed_pixels(g, bg):
        seed_col = int(g[sr, sc])
        traced = _nonbg_component_from(g, (sr, sc), bg, conn8=trace_conn8)
        line_col = _line_color_for_seed(g, sr, sc, bg)
        for region, boundary in cavities:
            boundary_in_traced = boundary & traced
            if not boundary_in_traced:
                continue
            ok = False
            for r, c in boundary_in_traced:
                for nr, nc in neigh_line(r, c):
                    if 0 <= nr < h and 0 <= nc < w and int(g[nr, nc]) == line_col:
                        ok = True
                        break
                if ok:
                    break
            if not ok:
                continue
            for r, c in region:
                if int(out[r, c]) == bg:
                    out[r, c] = seed_col
    return out


def _region_is_rectangle(region: set[tuple[int, int]] | frozenset[tuple[int, int]]) -> bool:
    rows = [r for r, _ in region]
    cols = [c for _, c in region]
    return len(region) == (max(rows) - min(rows) + 1) * (max(cols) - min(cols) + 1)


def combo_metrics(grid: np.ndarray) -> dict[str, int]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    new_regions = _fillable_region_keys(g, bg, trace_conn8=False, line_adj8=False)
    legacy_regions = _fillable_region_keys(g, bg, trace_conn8=True, line_adj8=True)
    legacy_only = legacy_regions - new_regions
    new_out = _solve_with_trace_mode(g, trace_conn8=False, line_adj8=False)
    legacy_out = _solve_with_trace_mode(g, trace_conn8=True, line_adj8=True)
    return {
        "fillable_irregular": sum(1 for reg in new_regions if not _region_is_rectangle(reg)),
        "fillable_rectangular": sum(1 for reg in new_regions if _region_is_rectangle(reg)),
        "legacy_only_regions": len(legacy_only),
        "legacy_only_cells": int(np.count_nonzero(new_out != legacy_out)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the interiors of irregular frame loops that are orthogonally reachable from unique border seeds.

    Input structure:
    - The background is the most frequent color.
    - One or two unique single-pixel seed colors sit on the grid border, typically at corners.
    - Each seed touches an orthogonal 1px line, and some single-color frame loops enclose irregular bg cavities.

    Rule:
    - Trace each seed's non-background network using 4-connectivity only.
    - A cavity is fillable only when its boundary is a single frame color, the cavity itself is a closed 4-connected
      bg region, and the traced seed network reaches that frame through an orthogonal ring-to-line contact.
    - Fill every such cavity with the corresponding seed color; regions reachable only through a diagonal-only
      connector do not count.

    Key insight:
    The seed color is not matched to a frame color directly. Instead, it labels whichever irregular cavities belong
    to the same 4-connected drawing, so the critical distinction is orthogonal attachment versus corner-only touch.
    """
    return _solve_with_trace_mode(grid, trace_conn8=False, line_adj8=False)


# ========================== CFG ==========================
GRID_H_RANGE = (8, 30)
GRID_W_RANGE = (8, 30)
BG_COLORS = [0, 5]

SEED_COUNT_WEIGHTS = [1, 2, 2]
MIN_LINE_LEN = 4
LINE_LEN_RANGE = (8, 28)

FRAME_COLORS_MAX = 4
TOTAL_LOOPS_MAX = 9
FRAME_PLACEMENT_PAD = 1
FRAME_ATTACH_TRIES = 90
DISTRACTOR_FRAME_TRIES = 80
DIAG_PROBE_TRIES = 120

STRAIGHT_WEIGHT = 4
TURN_WEIGHT = 3
INWARD_WEIGHT = 2
BORDER_AVOIDANCE_PROB = 0.85
BIAS_FLIP_PROB = 0.3

DISTRACTOR_NETWORK_PROB = 0.45
DISTRACTOR_NETWORK_TRIES = 60
DISTRACTOR_LINE_LEN = (5, 12)
DISTRACTOR_SOLO_MAX = 1
DISTRACTOR_NETWORK_FRAMES = (1, 2)

GEN_RETRIES = 600
LINE_GROW_RETRIES = 8
# =========================================================

ORTHO_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _is_orth_neighbor(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1


def _parse_mask(rows: list[str]) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


CAVITY_TEMPLATE_BANK = [
    _parse_mask(["110", "100", "111"]),      # L
    _parse_mask(["111", "010", "010"]),      # T
    _parse_mask(["011", "110", "100"]),      # zigzag
    _parse_mask(["1100", "0110", "0011"]),   # staircase
    _parse_mask(["1110", "0010", "0110"]),   # hook
    _parse_mask(["1100", "1000", "1110"]),   # fat L
]
CAVITY_TEMPLATE_WEIGHTS = [3, 3, 2, 2, 2, 2]


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _irregular_frame_coords(
    r0: int,
    c0: int,
    cavity_mask: np.ndarray,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """
    Build a 4-connected irregular frame as the cavity's 8-neighbor shell.

    Using the Moore shell rather than the 4-neighbor shell is the key combo adjustment:
    it preserves one enclosed irregular cavity while keeping the ring itself 4-connected.
    """
    mh, mw = cavity_mask.shape
    cavity = []
    ring = set()
    for rr in range(mh):
        for cc in range(mw):
            if not cavity_mask[rr, cc]:
                continue
            cavity.append((r0 + 1 + rr, c0 + 1 + cc))
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    ring.add((r0 + 1 + rr + dr, c0 + 1 + cc + dc))
    ring -= set(cavity)
    return sorted(ring), cavity


def _mask_has_single_cavity_and_ring4(cavity_mask: np.ndarray) -> bool:
    ring, cavity = _irregular_frame_coords(0, 0, cavity_mask)
    local_h, local_w = cavity_mask.shape[0] + 2, cavity_mask.shape[1] + 2
    local = np.zeros((local_h, local_w), dtype=int)
    for rr, cc in ring:
        local[rr, cc] = 1
    ring_components = _components_mask(local == 1, conn8=False)
    if len(ring_components) != 1:
        return False
    inner = []
    for pts in _components_mask(local == 0, conn8=False):
        if not _touches_border(pts, local_h, local_w):
            inner.append(set(pts))
    return len(inner) == 1 and inner[0] == set(cavity)


def _random_cavity_mask(rng: random.Random) -> np.ndarray:
    for _ in range(48):
        base = rng.choices(CAVITY_TEMPLATE_BANK, weights=CAVITY_TEMPLATE_WEIGHTS, k=1)[0].copy()
        if rng.random() < 0.5:
            base = np.fliplr(base)
        rot = rng.randrange(4)
        if rot:
            base = np.rot90(base, rot)
        base = _trim_mask(base)
        if not _region_is_rectangle(set(map(tuple, np.argwhere(base)))):
            if _mask_has_single_cavity_and_ring4(base):
                return base
    raise RuntimeError("failed to sample valid cavity mask")


def _grow_line(
    grid: np.ndarray,
    start: tuple[int, int],
    line_col: int,
    target_len: int,
    rng: random.Random,
) -> list[tuple[int, int]]:
    """Grow a meandering 4-connected orthogonal line with 8-neighbor self-clearance."""
    h, w = grid.shape
    path = [start]
    dr_bias = 1 if start[0] < h // 2 else -1
    dc_bias = 1 if start[1] < w // 2 else -1
    cur = start
    path_set = {start}
    last_dir = None

    while len(path) < target_len:
        weighted_dirs = []
        for ddr, ddc in ORTHO_DIRS:
            if last_dir is not None and (ddr, ddc) == (-last_dir[0], -last_dir[1]):
                continue
            nr, nc = cur[0] + ddr, cur[1] + ddc
            if not _in_bounds(h, w, nr, nc) or grid[nr, nc] != -1:
                continue
            if any((nnr, nnc) != cur and (nnr, nnc) in path_set for nnr, nnc in neighbors8(nr, nc)):
                continue
            if (nr == 0 or nc == 0 or nr == h - 1 or nc == w - 1) and rng.random() < BORDER_AVOIDANCE_PROB:
                continue

            weight = TURN_WEIGHT
            if last_dir == (ddr, ddc):
                weight += STRAIGHT_WEIGHT
            if ddr == dr_bias or ddc == dc_bias:
                weight += INWARD_WEIGHT
            weighted_dirs.extend([(ddr, ddc)] * weight)

        if not weighted_dirs:
            if rng.random() < BIAS_FLIP_PROB:
                dr_bias *= -1
            if rng.random() < BIAS_FLIP_PROB:
                dc_bias *= -1
            break

        ddr, ddc = rng.choice(weighted_dirs)
        cur = (cur[0] + ddr, cur[1] + ddc)
        path.append(cur)
        path_set.add(cur)
        grid[cur] = line_col
        last_dir = (ddr, ddc)

    return path


def _halo_offsets(fh: int, fw: int) -> list[tuple[int, int]]:
    return [
        (dr, dc)
        for dr in range(-1, fh + 1)
        for dc in range(-1, fw + 1)
        if not (0 <= dr < fh and 0 <= dc < fw)
    ]


def _line_cells_by_room(line_path: list[tuple[int, int]], h: int, w: int) -> list[tuple[int, int]]:
    uniq = list(dict.fromkeys(line_path))
    return sorted(
        uniq,
        key=lambda p: min(p[0], h - 1 - p[0], p[1], w - 1 - p[1]),
        reverse=True,
    )


def _validate_frame_placement(
    canvas: np.ndarray,
    ring: list[tuple[int, int]],
    cavity: list[tuple[int, int]],
    line_col: int,
    h: int,
    w: int,
) -> bool:
    for rr, cc in ring:
        if not _in_bounds(h, w, rr, cc) or canvas[rr, cc] != -1:
            return False
    for rr, cc in cavity:
        if not _in_bounds(h, w, rr, cc) or canvas[rr, cc] != -1:
            return False

    for cr, cc in cavity:
        for nr, nc in neighbors8(cr, cc):
            if _in_bounds(h, w, nr, nc) and canvas[nr, nc] == line_col:
                return False

    ring_set = set(ring)
    has_orth_touch = False
    for rr, cc in ring:
        for nr, nc in neighbors8(rr, cc):
            if not _in_bounds(h, w, nr, nc) or (nr, nc) in ring_set:
                continue
            v = canvas[nr, nc]
            if v == -1:
                continue
            if v == line_col:
                if _is_orth_neighbor((rr, cc), (nr, nc)):
                    has_orth_touch = True
                continue
            return False
    return has_orth_touch


def _validate_diagonal_probe_frame(
    canvas: np.ndarray,
    ring: list[tuple[int, int]],
    cavity: list[tuple[int, int]],
    line_col: int,
    h: int,
    w: int,
) -> bool:
    for rr, cc in ring:
        if not _in_bounds(h, w, rr, cc) or canvas[rr, cc] != -1:
            return False
    for rr, cc in cavity:
        if not _in_bounds(h, w, rr, cc) or canvas[rr, cc] != -1:
            return False

    for cr, cc in cavity:
        for nr, nc in neighbors8(cr, cc):
            if _in_bounds(h, w, nr, nc) and canvas[nr, nc] == line_col:
                return False

    ring_set = set(ring)
    has_diag_touch = False
    for rr, cc in ring:
        for nr, nc in neighbors8(rr, cc):
            if not _in_bounds(h, w, nr, nc) or (nr, nc) in ring_set:
                continue
            v = canvas[nr, nc]
            if v == -1:
                continue
            if v == line_col:
                if _is_orth_neighbor((rr, cc), (nr, nc)):
                    return False
                has_diag_touch = True
                continue
            return False
    return has_diag_touch


def _try_attach_frame(
    canvas: np.ndarray,
    line_path: list[tuple[int, int]],
    line_col: int,
    frame_col: int,
    rng: random.Random,
    max_tries: int = FRAME_ATTACH_TRIES,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    h, w = canvas.shape
    pad = FRAME_PLACEMENT_PAD
    line_cells = _line_cells_by_room(line_path, h, w)
    for _ in range(max_tries):
        cavity_mask = _random_cavity_mask(rng)
        ring_local, _ = _irregular_frame_coords(0, 0, cavity_mask)
        local_ring = ring_local[:]
        rng.shuffle(local_ring)
        dirs = ORTHO_DIRS[:]
        rng.shuffle(dirs)
        cells = line_cells[:]
        rng.shuffle(cells)
        tried_origins = set()
        for lp in cells:
            for rr, cc in local_ring:
                for dr, dc in dirs:
                    target_r, target_c = lp[0] + dr, lp[1] + dc
                    r0 = target_r - rr
                    c0 = target_c - cc
                    if (r0, c0) in tried_origins:
                        continue
                    tried_origins.add((r0, c0))
                    fh, fw = cavity_mask.shape[0] + 2, cavity_mask.shape[1] + 2
                    if r0 < pad or c0 < pad or r0 + fh > h - pad or c0 + fw > w - pad:
                        continue
                    ring, cavity = _irregular_frame_coords(r0, c0, cavity_mask)
                    if not _validate_frame_placement(canvas, ring, cavity, line_col, h, w):
                        continue
                    for pr, pc in ring:
                        canvas[pr, pc] = frame_col
                    return ring, cavity
    return None


def _try_place_diagonal_probe_frame(
    canvas: np.ndarray,
    line_path: list[tuple[int, int]],
    line_col: int,
    frame_col: int,
    rng: random.Random,
) -> bool:
    h, w = canvas.shape
    pad = FRAME_PLACEMENT_PAD
    line_cells = _line_cells_by_room(line_path, h, w)
    for _ in range(DIAG_PROBE_TRIES):
        cavity_mask = _random_cavity_mask(rng)
        ring_local, _ = _irregular_frame_coords(0, 0, cavity_mask)
        cells = line_cells[:]
        rng.shuffle(cells)
        for lp in cells:
            connector_dirs = DIAG_DIRS[:]
            rng.shuffle(connector_dirs)
            for cdr, cdc in connector_dirs:
                kr, kc = lp[0] + cdr, lp[1] + cdc
                if not _in_bounds(h, w, kr, kc) or canvas[kr, kc] != -1:
                    continue
                if any(_in_bounds(h, w, nr, nc) and canvas[nr, nc] == line_col for nr, nc in neighbors4(kr, kc)):
                    continue
                connector_ok = True
                for nr, nc in neighbors8(kr, kc):
                    if not _in_bounds(h, w, nr, nc):
                        continue
                    v = canvas[nr, nc]
                    if v != -1 and v != line_col:
                        connector_ok = False
                        break
                if not connector_ok:
                    continue

                local_ring = ring_local[:]
                rng.shuffle(local_ring)
                ring_touch_dirs = ORTHO_DIRS + DIAG_DIRS
                rng.shuffle(ring_touch_dirs)
                tried_origins = set()
                for rr, cc in local_ring:
                    for rdr, rdc in ring_touch_dirs:
                        target_r, target_c = kr + rdr, kc + rdc
                        r0 = target_r - rr
                        c0 = target_c - cc
                        if (r0, c0) in tried_origins:
                            continue
                        tried_origins.add((r0, c0))
                        fh, fw = cavity_mask.shape[0] + 2, cavity_mask.shape[1] + 2
                        if r0 < pad or c0 < pad or r0 + fh > h - pad or c0 + fw > w - pad:
                            continue
                        ring, cavity = _irregular_frame_coords(r0, c0, cavity_mask)

                        ok = True
                        ring_set = set(ring)
                        if (kr, kc) in ring_set or (kr, kc) in cavity:
                            continue
                        for rr2, cc2 in ring:
                            if not _in_bounds(h, w, rr2, cc2) or canvas[rr2, cc2] != -1:
                                ok = False
                                break
                            for nr, nc in neighbors8(rr2, cc2):
                                if not _in_bounds(h, w, nr, nc) or (nr, nc) == (kr, kc) or (nr, nc) in ring_set:
                                    continue
                                if canvas[nr, nc] != -1:
                                    ok = False
                                    break
                            if not ok:
                                break
                        if not ok:
                            continue
                        for rr2, cc2 in cavity:
                            if not _in_bounds(h, w, rr2, cc2) or canvas[rr2, cc2] != -1:
                                ok = False
                                break
                            for nr, nc in neighbors8(rr2, cc2):
                                if _in_bounds(h, w, nr, nc) and canvas[nr, nc] == line_col:
                                    ok = False
                                    break
                            if not ok:
                                break
                        if not ok:
                            continue
                        if not any(abs(rr2 - kr) <= 1 and abs(cc2 - kc) <= 1 for rr2, cc2 in ring):
                            continue

                        for pr, pc in ring:
                            canvas[pr, pc] = frame_col
                        canvas[kr, kc] = line_col
                        return True
    return False


def _try_attach_distractor_frame(
    canvas: np.ndarray,
    frame_col: int,
    rng: random.Random,
    max_tries: int = DISTRACTOR_FRAME_TRIES,
) -> bool:
    h, w = canvas.shape
    for _ in range(max_tries):
        cavity_mask = _random_cavity_mask(rng)
        fh, fw = cavity_mask.shape[0] + 2, cavity_mask.shape[1] + 2
        if h - fh - 1 < 1 or w - fw - 1 < 1:
            return False
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
        ring, _ = _irregular_frame_coords(r0, c0, cavity_mask)
        for rr, cc in ring:
            canvas[rr, cc] = frame_col
        return True
    return False


def _try_place_distractor_network(
    canvas: np.ndarray,
    line_col: int,
    frame_colors_pool: list[int],
    rng: random.Random,
) -> bool:
    h, w = canvas.shape
    for _ in range(DISTRACTOR_NETWORK_TRIES):
        r = rng.randint(2, h - 3)
        c = rng.randint(2, w - 3)
        if any(canvas[r + dr, c + dc] != -1 for dr in range(-1, 2) for dc in range(-1, 2)):
            continue
        canvas[r, c] = line_col
        target_len = rng.randint(*DISTRACTOR_LINE_LEN)
        path = _grow_line(canvas, (r, c), line_col, target_len, rng)
        if len(path) < 3:
            for pr, pc in path:
                canvas[pr, pc] = -1
            continue
        attached = 0
        for _ in range(rng.randint(*DISTRACTOR_NETWORK_FRAMES)):
            fc = rng.choice(frame_colors_pool)
            if _try_attach_frame(canvas, path, line_col, fc, rng) is not None:
                attached += 1
        if attached > 0:
            return True
        for pr, pc in path:
            canvas[pr, pc] = -1
    return False


def _corner_config(corner_kind: str, h: int, w: int, rng: random.Random) -> tuple[tuple[int, int], tuple[int, int]]:
    if corner_kind == "TL":
        seed = (0, 0)
        first = rng.choice([(0, 1), (1, 0)])
    elif corner_kind == "TR":
        seed = (0, w - 1)
        first = rng.choice([(0, w - 2), (1, w - 1)])
    elif corner_kind == "BL":
        seed = (h - 1, 0)
        first = rng.choice([(h - 1, 1), (h - 2, 0)])
    elif corner_kind == "BR":
        seed = (h - 1, w - 1)
        first = rng.choice([(h - 1, w - 2), (h - 2, w - 1)])
    else:
        raise ValueError(corner_kind)
    return seed, first


def _distribute_loop_budget(num_seeds: int, num_attached_target: int, rng: random.Random) -> list[int]:
    seed_order = list(range(num_seeds))
    rng.shuffle(seed_order)
    per_seed = [0] * num_seeds
    remaining = num_attached_target
    while remaining > 0 and any(per_seed[i] == 0 for i in range(num_seeds)):
        for i in seed_order:
            if per_seed[i] == 0 and remaining > 0:
                per_seed[i] += 1
                remaining -= 1
    while remaining > 0:
        per_seed[rng.choice(seed_order)] += 1
        remaining -= 1
    return per_seed


def _seed_components_are_disjoint(
    grid: np.ndarray,
    lines_info: list[tuple[int, tuple[int, int], int, list[tuple[int, int]]]],
    bg: int,
) -> bool:
    traces = []
    seed_positions = [seed_rc for _, seed_rc, _, _ in lines_info]
    for _, seed_rc, _, _ in lines_info:
        traced = _nonbg_component_from(grid, seed_rc, bg, conn8=True)
        for other_seed in seed_positions:
            if other_seed != seed_rc and other_seed in traced:
                return False
        traces.append(traced)
    for i in range(len(traces)):
        for j in range(i + 1, len(traces)):
            if traces[i] & traces[j]:
                return False
    return True


def _fillable_cavity_profile(grid: np.ndarray, bg: int) -> tuple[int, int]:
    fillable = _fillable_region_keys(grid, bg, trace_conn8=False, line_adj8=False)
    irregular = sum(1 for reg in fillable if not _region_is_rectangle(reg))
    rectangular = sum(1 for reg in fillable if _region_is_rectangle(reg))
    return irregular, rectangular


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    bg = rng.choice(BG_COLORS)
    palette = [c for c in range(10) if c != bg]
    rng.shuffle(palette)

    num_seeds = rng.choice(SEED_COUNT_WEIGHTS)
    if num_seeds * 2 > len(palette):
        return None

    canvas = -np.ones((h, w), dtype=int)
    seed_colors = [palette.pop() for _ in range(num_seeds)]
    line_colors = [palette.pop() for _ in range(num_seeds)]

    all_corners = ["TL", "TR", "BL", "BR"]
    rng.shuffle(all_corners)
    corners_used = all_corners[:num_seeds]

    num_frame_cols = rng.randint(1, min(FRAME_COLORS_MAX, len(palette)))
    frame_colors_pool = [palette.pop() for _ in range(num_frame_cols)]

    lines_info = []
    for i in range(num_seeds):
        seed_rc, first_rc = _corner_config(corners_used[i], h, w, rng)
        if canvas[seed_rc] != -1 or canvas[first_rc] != -1:
            return None
        canvas[seed_rc] = seed_colors[i]
        path = None
        for _ in range(LINE_GROW_RETRIES):
            canvas[first_rc] = line_colors[i]
            max_len = min(LINE_LEN_RANGE[1], max(LINE_LEN_RANGE[0], (h * w) // 5))
            target_len = rng.randint(LINE_LEN_RANGE[0], max_len)
            candidate = _grow_line(canvas, first_rc, line_colors[i], target_len, rng)
            if len(candidate) >= MIN_LINE_LEN:
                path = candidate
                break
            for pr, pc in candidate:
                if (pr, pc) != seed_rc:
                    canvas[pr, pc] = -1
        if path is None:
            return None
        lines_info.append((seed_colors[i], seed_rc, line_colors[i], path))

    per_seed = [1] * num_seeds
    extra_attached = rng.randint(0, min(2, TOTAL_LOOPS_MAX - num_seeds - 1))
    for _ in range(extra_attached):
        per_seed[rng.randrange(num_seeds)] += 1

    for i, (_, _, line_col, path) in enumerate(lines_info):
        attached = 0
        for _ in range(per_seed[i]):
            fc = rng.choice(frame_colors_pool)
            if _try_attach_frame(canvas, path, line_col, fc, rng) is not None:
                attached += 1
        if attached == 0:
            return None

    probe_owner = rng.randrange(num_seeds)
    probe_ok = _try_place_diagonal_probe_frame(
        canvas,
        lines_info[probe_owner][3],
        lines_info[probe_owner][2],
        rng.choice(frame_colors_pool),
        rng,
    )
    if not probe_ok:
        return None

    if rng.random() < 0.35:
        _try_attach_distractor_frame(canvas, rng.choice(frame_colors_pool), rng)

    grid = np.where(canvas == -1, bg, canvas).astype(int)

    vals, cnts = np.unique(grid, return_counts=True)
    if int(vals[np.argmax(cnts)]) != bg:
        return None

    for seed_col, _, _, _ in lines_info:
        if int((grid == seed_col).sum()) != 1:
            return None

    if not _seed_components_are_disjoint(grid, lines_info, bg):
        return None

    irregular_fillable, rectangular_fillable = _fillable_cavity_profile(grid, bg)
    if irregular_fillable == 0 or rectangular_fillable != 0:
        return None

    try:
        out = main(grid)
        legacy_out = _solve_with_trace_mode(grid, trace_conn8=True, line_adj8=True)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None
    if np.array_equal(out, legacy_out):
        return None

    metrics = combo_metrics(grid)
    if metrics["legacy_only_regions"] == 0 or metrics["legacy_only_cells"] == 0:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError("generate_input: exhausted attempts")
