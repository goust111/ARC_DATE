# Combo variant of 8b7bacbf (index 25)
# combo_spec: D15/1+D16/0
# Name: 角落指定背景+边缘接触排除
#
# Design analysis for the fused variant
# a) Concrete behavior and execution trace
#    - Original task:
#      * background = most common color;
#      * a unique border seed traces one 8-connected non-background drawing;
#      * every 4-connected background cavity whose monochrome boundary belongs to that drawing is filled;
#      * "outside" means the background component containing a background corner.
#    - Fused task:
#      * background is inferred from the four corner pixels by majority, so it may be a minority color overall;
#      * outside is any background component that reaches any grid edge, even if it never reaches a corner.
#    - Small trace:
#      * Suppose the actual background is grey 5. Three corners are 5 and the remaining corner is a seed 7.
#      * A large orange/red drawing covers most cells, so the most common color is orange 2, not 5.
#      * One closed loop on the seed line encloses a true interior 5-cavity: both old and new "outside" rules
#        treat it as fillable, but only the new background rule finds it because 5 is not the mode color.
#      * The same seed line also touches a U-shaped pocket whose 5-region reaches the top edge through a narrow
#        opening segment away from every corner. Under the old corner-based outside rule that pocket looks like an
#        interior cavity and would be filled; under the fused rule it is exterior and must stay background.
#      * The visible difference is therefore twofold: corner majority, not frequency, identifies background; and
#        edge reachability, not corner membership, decides which background regions are excluded.
# b) Complexity calibration
#    - The object vocabulary stays close to the source task: singleton border seed, thin meandering line, small
#      single-color loop frames, background cavities, plus one disconnected solid distractor panel whose only job
#      is to break the old "most frequent color = background" heuristic cleanly.
#    - To make D16 observable without adding arbitrary symbols, every sample contains:
#      * at least one normal closed loop that really fills;
#      * at least one line-attached edge pocket with the same monochrome-boundary grammar that does not fill.
#    - To make D15 observable, every sample forces three corners to the real background color while a non-background
#      line/frame color becomes globally most common. This keeps the new rule visible in 2-4 examples.
# c) Hidden-assumption audit against the original main()
#    - Original `bg = get_bg(g)` assumes the most common color is background. That is invalid here; replace it with
#      a corner-majority rule. The generator guarantees a unique majority by using exactly one non-background corner.
#    - Original outside detection assumes any fillable cavity is simply "not the background component containing a
#      background corner". That fails for side-touching pockets disconnected from corner background; replace it with
#      `touches_border`.
#    - Original `_seed_pixels()` still works because the seed remains a globally unique singleton on the border.
#    - Original line-color inference from the seed's 4-neighbors still works because the seed keeps one orthogonal
#      line contact.
# d) Placement legality
#    - The generator reserves one corner for the seed and leaves the other three corners as background.
#    - A border-touching pocket is built as a clean U-shaped monochrome boundary whose background interior reaches
#      exactly one non-corner edge segment while remaining 4-disconnected from all corner background.
#    - Closed loops, the distractor panel, and distractor networks use empty-cell and halo checks so objects do not
#      overlap or merge.
#    - Pocket cavity cells are reserved during line growth, preventing the line from entering the would-be outside
#      region and accidentally breaking the intended boundary grammar.
# e) Visual observability
#    - Humans can see that the corners agree on the background color even when that color is not the most frequent.
#    - They can also compare two line-attached regions with the same local grammar: one fully enclosed loop fills,
#      while one region that clearly reaches an edge does not.
#    - No rule depends on hidden color ids; both changes are topological and visible.
# f) Visual cleanliness
#    - The palette stays small: one background color, one seed color, one line color, one distractor-panel color,
#      and a small frame-color pool.
#    - The edge pocket is the only open shape and is spatially separated from closed loops, so the sample still reads
#      as a tidy "seed -> line -> loop/pocket" composition instead of a cluttered collage.
# g) Change inferability
#    - D15 is repeated structurally by the corners themselves: three matching corners reveal the background even when
#      that color is globally rare.
#    - D16 is repeated structurally by the required edge pocket: a region that looks loop-like locally still remains
#      unfilled because it visibly reaches the border.
# 9) New solver assumptions and generator guarantees
#    - Assumption: among the four corners, the true background color has strict majority. Guarantee: exactly one corner
#      carries the seed and all other corners remain background.
#    - Assumption: at least one fillable closed cavity exists. Guarantee: the generator requires at least one attached
#      closed loop whose cavity the fused solver fills.
#    - Assumption: at least one edge-touching non-corner background region is line-attached and monochrome-bounded.
#      Guarantee: the generator always places a line-attached border pocket and rejects samples where the profile check
#      does not find it.
#    - Assumption: the fused behavior is not degenerate. Guarantee: the generator rejects any sample whose fused output
#      matches either (i) the most-common-background solver or (ii) the corner-majority + corner-outside solver.

from __future__ import annotations

from collections import Counter, deque
import random
import sys

import numpy as np

SOLVER_UTILS = "/home/lixinyu/snap/new/solvers"
if SOLVER_UTILS not in sys.path:
    sys.path.insert(0, SOLVER_UTILS)

from arc_utils import components_mask, get_bg, neighbors4, neighbors8


EMPTY = -1
RESERVED = -2

GRID_H_RANGE = (11, 13)
GRID_W_RANGE = (11, 13)
BG_COLORS = [0, 5]

LINE_LEN_RANGE = (16, 26)
MIN_TOTAL_LINE_LEN = 12

ATTACHED_LOOP_COUNT = (1, 3)
DISTRACTOR_NETWORK_PROB = 1.0
DISTRACTOR_NETWORK_TRIES = 50
DISTRACTOR_LINE_LEN = (10, 18)
DISTRACTOR_FRAMES = (2, 3)
DISTRACTOR_SOLO_MAX = 2

FRAME_ATTACH_TRIES = 120
DISTRACTOR_FRAME_TRIES = 50
FRAME_PLACEMENT_PAD = 1
FRAME_COLORS_MAX = 3
CAVITY_SIZE_CHOICES = [2, 2, 2, 2, 2, 3]
POCKET_SIZE_CHOICES = [2, 3]
PANEL_H_CHOICES = [5, 6]
PANEL_W_CHOICES = [6, 7]

DIAG_WEIGHT_ALIGNED = 5
DIAG_WEIGHT_OTHER = 1
ORTHO_WEIGHT_ALIGNED = 2
ORTHO_WEIGHT_OTHER = 1
BORDER_AVOIDANCE_PROB = 0.85
BIAS_FLIP_PROB = 0.3

GEN_RETRIES = 1200

DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
ORTHO_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
CORNERS = ("TL", "TR", "BL", "BR")


def _components_mask(mask: np.ndarray, conn8: bool) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8 if conn8 else 4)]


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _touches_border(pts: list[tuple[int, int]] | set[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts)


def _corner_background(g: np.ndarray) -> int:
    h, w = g.shape
    corner_vals = [int(g[0, 0]), int(g[0, w - 1]), int(g[h - 1, 0]), int(g[h - 1, w - 1])]
    return Counter(corner_vals).most_common(1)[0][0]


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
    seeds.sort()
    return seeds


def _nonbg_component8_from(g: np.ndarray, start: tuple[int, int], bg: int) -> set[tuple[int, int]]:
    h, w = g.shape
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for nr, nc in neighbors8(r, c):
            if _in_bounds(h, w, nr, nc) and (nr, nc) not in seen and int(g[nr, nc]) != bg:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return seen


def _line_color_for_seed(g: np.ndarray, sr: int, sc: int, bg: int) -> int:
    h, w = g.shape
    neigh_cols = []
    for nr, nc in neighbors4(sr, sc):
        if _in_bounds(h, w, nr, nc):
            v = int(g[nr, nc])
            if v != bg:
                neigh_cols.append(v)
    assert neigh_cols, "seed pixel must have at least one non-background 4-neighbor"
    return Counter(neigh_cols).most_common(1)[0][0]


def _bg_regions_with_boundaries(
    g: np.ndarray,
    bg: int,
) -> list[tuple[set[tuple[int, int]], set[tuple[int, int]], bool, bool]]:
    h, w = g.shape
    bg_corner_pts = {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}
    out = []
    for pts in _components_mask(g == bg, conn8=False):
        region = set(pts)
        boundary = set()
        for r, c in region:
            for nr, nc in neighbors4(r, c):
                if _in_bounds(h, w, nr, nc) and (nr, nc) not in region and int(g[nr, nc]) != bg:
                    boundary.add((nr, nc))
        touches = _touches_border(region, h, w)
        touches_bg_corner = any(p in bg_corner_pts and int(g[p]) == bg for p in region)
        out.append((region, boundary, touches, touches_bg_corner))
    return out


def _solve_with_rules(grid: np.ndarray, bg: int, *, outside_by_edge: bool) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    cavities = []
    for region, boundary, touches_border, touches_bg_corner in _bg_regions_with_boundaries(g, bg):
        outside = touches_border if outside_by_edge else touches_bg_corner
        if outside or not boundary:
            continue
        cavities.append((region, boundary, {int(g[p]) for p in boundary}))

    for sr, sc in _seed_pixels(g, bg):
        seed_col = int(g[sr, sc])
        line_col = _line_color_for_seed(g, sr, sc, bg)
        traced = _nonbg_component8_from(g, (sr, sc), bg)

        for region, boundary, bcols in cavities:
            if len(bcols) != 1:
                continue
            boundary_in_traced = boundary & traced
            if not boundary_in_traced:
                continue

            ok = False
            for r, c in boundary_in_traced:
                for nr, nc in neighbors8(r, c):
                    if _in_bounds(h, w, nr, nc) and int(g[nr, nc]) == line_col:
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill enclosed cavities on a seeded line drawing, with background read from the corners
    and exterior excluded by edge reachability.

    Input structure:
    - One border singleton seed starts a thin 8-connected line network that reaches several
      single-color frame loops.
    - Three of the four corners share the true background color; the remaining corner may be
      occupied by the seed, so background is not necessarily the most frequent color.
    - Besides ordinary closed loops, the drawing also includes a monochrome edge pocket whose
      background interior reaches one non-corner border segment.

    Rule:
    - Infer the background color from the majority color among the four corners.
    - Trace each unique border seed's full non-background 8-connected component and recover its
      line color from the seed's orthogonal non-background neighbor.
    - Enumerate 4-connected regions of the inferred background color, but treat any region that
      can reach any grid edge as exterior.
    - For every remaining monochrome-bounded region touched by the traced drawing, flood-fill
      the region with the seed color.

    Key insight:
    The two changed meta-rules interact: first identify which color is truly "background" from
    the corners rather than from frequency, then exclude all background regions that visibly
    connect to the edge, even if they are separated from the corner background.
    """
    g = np.asarray(grid, dtype=int)
    bg = _corner_background(g)
    return _solve_with_rules(g, bg, outside_by_edge=True)


def _frame_ring_coords(r0: int, c0: int, ch: int, cw: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    ring = []
    for dr in range(ch + 2):
        for dc in range(cw + 2):
            on_border = dr == 0 or dc == 0 or dr == ch + 1 or dc == cw + 1
            corner = dr in (0, ch + 1) and dc in (0, cw + 1)
            if on_border and not corner:
                ring.append((r0 + dr, c0 + dc))
    cavity = [(r0 + 1 + dr, c0 + 1 + dc) for dr in range(ch) for dc in range(cw)]
    return ring, cavity


def _validate_frame_placement(
    canvas: np.ndarray,
    ring: list[tuple[int, int]],
    cavity: list[tuple[int, int]],
    line_col: int,
    h: int,
    w: int,
) -> bool:
    for rr, cc in ring:
        if not _in_bounds(h, w, rr, cc) or int(canvas[rr, cc]) != EMPTY:
            return False
    for rr, cc in cavity:
        if not _in_bounds(h, w, rr, cc) or int(canvas[rr, cc]) != EMPTY:
            return False

    ring_set = set(ring)
    for cr, cc in cavity:
        for nr, nc in neighbors8(cr, cc):
            if _in_bounds(h, w, nr, nc) and int(canvas[nr, nc]) == line_col:
                return False

    for rr, cc in ring:
        for nr, nc in neighbors8(rr, cc):
            if not _in_bounds(h, w, nr, nc) or (nr, nc) in ring_set:
                continue
            v = int(canvas[nr, nc])
            if v not in (EMPTY, line_col):
                return False

    for rr, cc in ring:
        for nr, nc in neighbors8(rr, cc):
            if _in_bounds(h, w, nr, nc) and int(canvas[nr, nc]) == line_col:
                return True
    return False


def _halo_offsets(fh: int, fw: int) -> list[tuple[int, int]]:
    return [(dr, dc) for dr in range(-1, fh + 1) for dc in range(-1, fw + 1) if not (0 <= dr < fh and 0 <= dc < fw)]


def _grow_line(
    grid: np.ndarray,
    start: tuple[int, int],
    line_col: int,
    target_len: int,
    rng: random.Random,
) -> list[tuple[int, int]]:
    h, w = grid.shape
    path = [start]
    path_set = {start}
    cur = start
    dr_bias = 1 if start[0] < h // 2 else -1
    dc_bias = 1 if start[1] < w // 2 else -1

    while len(path) < target_len:
        dirs = []
        for ddr, ddc in DIAG_DIRS:
            aligned = ddr == dr_bias or ddc == dc_bias
            weight = DIAG_WEIGHT_ALIGNED if aligned else DIAG_WEIGHT_OTHER
            dirs.extend([(ddr, ddc)] * weight)
        for ddr, ddc in ORTHO_DIRS:
            aligned = (ddr == dr_bias and ddc == 0) or (ddc == dc_bias and ddr == 0)
            weight = ORTHO_WEIGHT_ALIGNED if aligned else ORTHO_WEIGHT_OTHER
            dirs.extend([(ddr, ddc)] * weight)
        rng.shuffle(dirs)

        placed = False
        for ddr, ddc in dirs:
            nr, nc = cur[0] + ddr, cur[1] + ddc
            if not _in_bounds(h, w, nr, nc) or int(grid[nr, nc]) != EMPTY:
                continue
            if any((nnr, nnc) != cur and (nnr, nnc) in path_set for nnr, nnc in neighbors8(nr, nc)):
                continue
            if (nr == 0 or nc == 0 or nr == h - 1 or nc == w - 1) and rng.random() < BORDER_AVOIDANCE_PROB:
                continue
            grid[nr, nc] = line_col
            cur = (nr, nc)
            path.append(cur)
            path_set.add(cur)
            placed = True
            break

        if not placed:
            if rng.random() < BIAS_FLIP_PROB:
                dr_bias *= -1
            if rng.random() < BIAS_FLIP_PROB:
                dc_bias *= -1
            break

    return path


def _try_attach_frame(
    canvas: np.ndarray,
    line_path: list[tuple[int, int]],
    line_col: int,
    frame_col: int,
    rng: random.Random,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    h, w = canvas.shape
    for _ in range(FRAME_ATTACH_TRIES):
        ch = rng.choice(CAVITY_SIZE_CHOICES)
        cw = rng.choice(CAVITY_SIZE_CHOICES)
        fh, fw = ch + 2, cw + 2
        lp = rng.choice(line_path)
        offsets = _halo_offsets(fh, fw)
        rng.shuffle(offsets)
        for dr, dc in offsets:
            r0 = lp[0] - dr
            c0 = lp[1] - dc
            if r0 < FRAME_PLACEMENT_PAD or c0 < FRAME_PLACEMENT_PAD:
                continue
            if r0 + fh > h - FRAME_PLACEMENT_PAD or c0 + fw > w - FRAME_PLACEMENT_PAD:
                continue
            ring, cavity = _frame_ring_coords(r0, c0, ch, cw)
            if not _validate_frame_placement(canvas, ring, cavity, line_col, h, w):
                continue
            for rr, cc in ring:
                canvas[rr, cc] = frame_col
            return ring, cavity
    return None


def _try_attach_frame_exhaustive(
    canvas: np.ndarray,
    line_path: list[tuple[int, int]],
    line_col: int,
    frame_col: int,
    rng: random.Random,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    h, w = canvas.shape
    path_order = list(line_path)
    rng.shuffle(path_order)
    size_order = [(2, 2), (2, 3), (3, 2), (3, 3)]
    rng.shuffle(size_order)
    for ch, cw in size_order:
        fh, fw = ch + 2, cw + 2
        offsets = _halo_offsets(fh, fw)
        rng.shuffle(offsets)
        for lp in path_order:
            for dr, dc in offsets:
                r0 = lp[0] - dr
                c0 = lp[1] - dc
                if r0 < FRAME_PLACEMENT_PAD or c0 < FRAME_PLACEMENT_PAD:
                    continue
                if r0 + fh > h - FRAME_PLACEMENT_PAD or c0 + fw > w - FRAME_PLACEMENT_PAD:
                    continue
                ring, cavity = _frame_ring_coords(r0, c0, ch, cw)
                if not _validate_frame_placement(canvas, ring, cavity, line_col, h, w):
                    continue
                for rr, cc in ring:
                    canvas[rr, cc] = frame_col
                return ring, cavity
    return None


def _try_place_panel(canvas: np.ndarray, panel_col: int, rng: random.Random) -> bool:
    h, w = canvas.shape
    corners = {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}
    choices = [(ph, pw) for ph in PANEL_H_CHOICES for pw in PANEL_W_CHOICES]
    rng.shuffle(choices)
    for ph, pw in choices:
        if ph > h or pw > w:
            continue
        rows = list(range(0, h - ph + 1))
        cols = list(range(0, w - pw + 1))
        rng.shuffle(rows)
        rng.shuffle(cols)
        for r0 in rows:
            for c0 in cols:
                cells = {(rr, cc) for rr in range(r0, r0 + ph) for cc in range(c0, c0 + pw)}
                if cells & corners:
                    continue
                ok = True
                for rr in range(r0 - 1, r0 + ph + 1):
                    for cc in range(c0 - 1, c0 + pw + 1):
                        if _in_bounds(h, w, rr, cc) and int(canvas[rr, cc]) != EMPTY:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    continue
                canvas[r0 : r0 + ph, c0 : c0 + pw] = panel_col
                return True
    return False


def _place_panel_far(canvas: np.ndarray, panel_col: int, corner: str, rng: random.Random) -> bool:
    h, w = canvas.shape
    choices = [(ph, pw) for ph in PANEL_H_CHOICES for pw in PANEL_W_CHOICES]
    rng.shuffle(choices)
    if corner in ("TL", "BL"):
        preferred_edges = ["right", "bottom"]
    else:
        preferred_edges = ["left", "bottom"]
    if corner in ("BL", "BR"):
        preferred_edges = preferred_edges[::-1]

    for ph, pw in choices:
        for edge in preferred_edges:
            candidates: list[tuple[int, int]] = []
            if edge == "right" and ph <= h - 2 and pw <= w:
                c0 = w - pw
                for r0 in [1, max(1, (h - ph) // 2), max(1, h - ph - 1)]:
                    if 0 <= r0 <= h - ph:
                        candidates.append((r0, c0))
            elif edge == "left" and ph <= h - 2 and pw <= w:
                c0 = 0
                for r0 in [1, max(1, (h - ph) // 2), max(1, h - ph - 1)]:
                    if 0 <= r0 <= h - ph:
                        candidates.append((r0, c0))
            elif edge == "bottom" and pw <= w - 2 and ph <= h:
                r0 = h - ph
                for c0 in [1, max(1, (w - pw) // 2), max(1, w - pw - 1)]:
                    if 0 <= c0 <= w - pw:
                        candidates.append((r0, c0))
            rng.shuffle(candidates)
            for r0, c0 in candidates:
                ok = True
                cells = {(rr, cc) for rr in range(r0, r0 + ph) for cc in range(c0, c0 + pw)}
                corners = {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}
                if cells & corners:
                    ok = False
                if not ok:
                    continue
                for rr in range(r0 - 1, r0 + ph + 1):
                    for cc in range(c0 - 1, c0 + pw + 1):
                        if _in_bounds(h, w, rr, cc) and int(canvas[rr, cc]) != EMPTY:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    continue
                canvas[r0 : r0 + ph, c0 : c0 + pw] = panel_col
                return True
    return False


def _place_frequency_band(canvas: np.ndarray, color: int, corner: str) -> bool:
    h, w = canvas.shape
    thickness = 3 if min(h, w) >= 12 else 2
    candidates: list[list[tuple[int, int]]] = []
    if corner in ("TL", "TR"):
        candidates.append([(r, c) for r in range(h - thickness, h) for c in range(1, w - 1)])
    else:
        candidates.append([(r, c) for r in range(0, thickness) for c in range(1, w - 1)])
    if corner in ("TL", "BL"):
        candidates.append([(r, c) for r in range(1, h - 1) for c in range(w - thickness, w)])
    else:
        candidates.append([(r, c) for r in range(1, h - 1) for c in range(0, thickness)])

    placed = False
    for cells in candidates:
        if all(int(canvas[r, c]) == EMPTY for r, c in cells):
            for r, c in cells:
                canvas[r, c] = color
            placed = True
    return placed


def _place_mode_breaker_slab(canvas: np.ndarray, color: int, corner: str) -> set[tuple[int, int]] | None:
    h, w = canvas.shape
    candidates: list[list[tuple[int, int]]] = []
    if corner in ("TL", "BL"):
        candidates.append([(r, c) for r in range(1, h - 1) for c in range(max(2, w // 2), w)])
        candidates.append([(r, c) for r in range(1, h - 1) for c in range(max(1, w // 2 - 1), w - 1)])
    else:
        candidates.append([(r, c) for r in range(1, h - 1) for c in range(0, min(w - 2, w // 2))])
        candidates.append([(r, c) for r in range(1, h - 1) for c in range(1, min(w - 1, w // 2 + 1))])
    if corner in ("TL", "TR"):
        candidates.append([(r, c) for r in range(max(2, h // 2), h) for c in range(1, w - 1)])
    else:
        candidates.append([(r, c) for r in range(0, min(h - 2, h // 2)) for c in range(1, w - 1)])

    for cells in candidates:
        if cells and all(int(canvas[r, c]) == EMPTY for r, c in cells):
            for r, c in cells:
                canvas[r, c] = color
            return set(cells)
    return None

def _place_big_panel_opposite(canvas: np.ndarray, color: int, corner: str) -> bool:
    h, w = canvas.shape
    sizes = [(4, 5), (4, 4), (3, 5), (3, 4)]
    placed = False
    for ph, pw in sizes:
        if ph > h - 2 or pw > w - 2:
            continue
        if corner == "TL":
            row_range = range(max(1, h // 2 - 1), h - ph + 1)
            col_range = range(max(1, w // 2 - 1), w - pw + 1)
        elif corner == "TR":
            row_range = range(max(1, h // 2 - 1), h - ph + 1)
            col_range = range(0, max(1, w // 2 - pw + 1))
        elif corner == "BL":
            row_range = range(0, max(1, h // 2 - ph + 1))
            col_range = range(max(1, w // 2 - 1), w - pw + 1)
        else:
            row_range = range(0, max(1, h // 2 - ph + 1))
            col_range = range(0, max(1, w // 2 - pw + 1))
        positions = [(r0, c0) for r0 in row_range for c0 in col_range]
        random.Random(0).shuffle(positions)
        for r0, c0 in positions:
            if r0 < 0 or c0 < 0 or r0 + ph > h or c0 + pw > w:
                continue
            cells = {(rr, cc) for rr in range(r0, r0 + ph) for cc in range(c0, c0 + pw)}
            corners = {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}
            if cells & corners:
                continue
            if any(int(canvas[rr, cc]) != EMPTY for rr, cc in cells):
                continue
            for rr, cc in cells:
                canvas[rr, cc] = color
            placed = True
            break
    return placed


def _try_attach_distractor_frame(canvas: np.ndarray, frame_col: int, rng: random.Random) -> bool:
    h, w = canvas.shape
    for _ in range(DISTRACTOR_FRAME_TRIES):
        ch = rng.choice(CAVITY_SIZE_CHOICES)
        cw = rng.choice(CAVITY_SIZE_CHOICES)
        fh, fw = ch + 2, cw + 2
        if h - fh - 1 < 1 or w - fw - 1 < 1:
            return False
        r0 = rng.randint(1, h - fh - 1)
        c0 = rng.randint(1, w - fw - 1)
        ok = True
        for rr in range(r0 - 1, r0 + fh + 1):
            for cc in range(c0 - 1, c0 + fw + 1):
                if _in_bounds(h, w, rr, cc) and int(canvas[rr, cc]) != EMPTY:
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
        if any(int(canvas[r + dr, c + dc]) != EMPTY for dr in range(-1, 2) for dc in range(-1, 2)):
            continue
        canvas[r, c] = line_col
        target_len = rng.randint(*DISTRACTOR_LINE_LEN)
        path = _grow_line(canvas, (r, c), line_col, target_len, rng)
        if len(path) < 4:
            for pr, pc in path:
                canvas[pr, pc] = EMPTY
            continue
        attached = 0
        for _ in range(rng.randint(*DISTRACTOR_FRAMES)):
            frame_col = rng.choice(frame_colors_pool)
            if _try_attach_frame(canvas, path, line_col, frame_col, rng) is not None:
                attached += 1
        if attached > 0:
            return True
        for pr, pc in path:
            canvas[pr, pc] = EMPTY
    return False


def _seed_corner_rc(corner: str, h: int, w: int) -> tuple[int, int]:
    if corner == "TL":
        return 0, 0
    if corner == "TR":
        return 0, w - 1
    if corner == "BL":
        return h - 1, 0
    if corner == "BR":
        return h - 1, w - 1
    raise ValueError(corner)


def _adjacent_sides(corner: str) -> tuple[str, str]:
    if corner == "TL":
        return "top", "left"
    if corner == "TR":
        return "top", "right"
    if corner == "BL":
        return "bottom", "left"
    if corner == "BR":
        return "bottom", "right"
    raise ValueError(corner)


def _first_step_for(corner: str, side: str, h: int, w: int) -> tuple[int, int]:
    if corner == "TL" and side == "top":
        return 0, 1
    if corner == "TL" and side == "left":
        return 1, 0
    if corner == "TR" and side == "top":
        return 0, w - 2
    if corner == "TR" and side == "right":
        return 1, w - 1
    if corner == "BL" and side == "bottom":
        return h - 1, 1
    if corner == "BL" and side == "left":
        return h - 2, 0
    if corner == "BR" and side == "bottom":
        return h - 1, w - 2
    if corner == "BR" and side == "right":
        return h - 2, w - 1
    raise ValueError((corner, side))


def _edge_pocket_shape(
    side: str,
    offset: int,
    ch: int,
    cw: int,
    h: int,
    w: int,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], tuple[int, int]] | None:
    boundary: set[tuple[int, int]] = set()
    cavity: set[tuple[int, int]] = set()

    if side == "left":
        r0 = offset
        if r0 < 1 or r0 + ch >= h - 1 or cw + 1 >= w:
            return None
        for r in range(r0, r0 + ch):
            for c in range(cw):
                cavity.add((r, c))
        for c in range(cw + 1):
            boundary.add((r0 - 1, c))
            boundary.add((r0 + ch, c))
        for r in range(r0, r0 + ch):
            boundary.add((r, cw))
        anchor = (r0 + ch // 2, cw + 1)
    elif side == "right":
        r0 = offset
        if r0 < 1 or r0 + ch >= h - 1 or w - cw - 2 < 0:
            return None
        for r in range(r0, r0 + ch):
            for c in range(w - cw, w):
                cavity.add((r, c))
        for c in range(w - cw - 1, w):
            boundary.add((r0 - 1, c))
            boundary.add((r0 + ch, c))
        for r in range(r0, r0 + ch):
            boundary.add((r, w - cw - 1))
        anchor = (r0 + ch // 2, w - cw - 2)
    elif side == "top":
        c0 = offset
        if c0 < 1 or c0 + cw >= w - 1 or ch + 1 >= h:
            return None
        for r in range(ch):
            for c in range(c0, c0 + cw):
                cavity.add((r, c))
        for r in range(ch):
            boundary.add((r, c0 - 1))
            boundary.add((r, c0 + cw))
        for c in range(c0 - 1, c0 + cw + 1):
            boundary.add((ch, c))
        anchor = (ch + 1, c0 + cw // 2)
    elif side == "bottom":
        c0 = offset
        if c0 < 1 or c0 + cw >= w - 1 or h - ch - 2 < 0:
            return None
        for r in range(h - ch, h):
            for c in range(c0, c0 + cw):
                cavity.add((r, c))
        for r in range(h - ch, h):
            boundary.add((r, c0 - 1))
            boundary.add((r, c0 + cw))
        for c in range(c0 - 1, c0 + cw + 1):
            boundary.add((h - ch - 1, c))
        anchor = (h - ch - 2, c0 + cw // 2)
    else:
        raise ValueError(side)

    if not all(_in_bounds(h, w, r, c) for r, c in boundary | cavity | {anchor}):
        return None
    return boundary, cavity, anchor


def _find_path8(
    start: tuple[int, int],
    goal: tuple[int, int],
    blocked: set[tuple[int, int]],
    h: int,
    w: int,
) -> list[tuple[int, int]] | None:
    if start == goal:
        return [start]
    q = deque([start])
    prev = {start: None}
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors8(r, c):
            if not _in_bounds(h, w, nr, nc):
                continue
            nxt = (nr, nc)
            if nxt in blocked or nxt in prev:
                continue
            prev[nxt] = (r, c)
            if nxt == goal:
                q.clear()
                break
            q.append(nxt)
    if goal not in prev:
        return None
    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path


def _place_seed_and_pocket(
    canvas: np.ndarray,
    seed_col: int,
    line_col: int,
    frame_col: int,
    corner: str,
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    h, w = canvas.shape
    seed = _seed_corner_rc(corner, h, w)
    if int(canvas[seed]) != EMPTY:
        return None

    side_options = list(_adjacent_sides(corner))
    rng.shuffle(side_options)
    for side in side_options:
        first = _first_step_for(corner, side, h, w)
        ch = rng.choice(POCKET_SIZE_CHOICES)
        cw = rng.choice(POCKET_SIZE_CHOICES)
        if side in ("left", "right"):
            offsets = list(range(2, h - ch - 1))
        else:
            offsets = list(range(2, w - cw - 1))
        rng.shuffle(offsets)
        for offset in offsets:
            shape = _edge_pocket_shape(side, offset, ch, cw, h, w)
            if shape is None:
                continue
            boundary, cavity, anchor = shape
            blocked = boundary | cavity | {seed}
            if first in blocked or anchor in blocked:
                continue
            route = _find_path8(first, anchor, blocked, h, w)
            if route is None:
                continue
            path_cells = route
            path_set = set(path_cells)

            ok = True
            for r, c in boundary:
                if int(canvas[r, c]) != EMPTY:
                    ok = False
                    break
            if not ok:
                continue
            for r, c in cavity:
                if int(canvas[r, c]) != EMPTY:
                    ok = False
                    break
            if not ok:
                continue
            for r, c in path_cells:
                if int(canvas[r, c]) != EMPTY:
                    ok = False
                    break
            if not ok:
                continue
            for cr, cc in cavity:
                for nr, nc in neighbors8(cr, cc):
                    if _in_bounds(h, w, nr, nc) and (nr, nc) in path_set:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue
            if not any(any((nr, nc) in path_set for nr, nc in neighbors8(br, bc)) for br, bc in boundary):
                continue

            canvas[seed] = seed_col
            for r, c in path_cells:
                canvas[r, c] = line_col
            for r, c in boundary:
                canvas[r, c] = frame_col
            for r, c in cavity:
                canvas[r, c] = RESERVED
            return path_cells
    return None


def _fillable_profile(grid: np.ndarray, bg: int) -> tuple[int, int]:
    seed_infos = []
    h, w = grid.shape
    for sr, sc in _seed_pixels(grid, bg):
        seed_infos.append((_nonbg_component8_from(grid, (sr, sc), bg), _line_color_for_seed(grid, sr, sc, bg)))

    closed_fillable = 0
    edge_touch_fillable = 0
    for region, boundary, touches_border, touches_bg_corner in _bg_regions_with_boundaries(grid, bg):
        if not boundary or len({int(grid[p]) for p in boundary}) != 1:
            continue
        fillable = False
        for traced, line_col in seed_infos:
            boundary_in_traced = boundary & traced
            if not boundary_in_traced:
                continue
            for r, c in boundary_in_traced:
                if any(_in_bounds(h, w, nr, nc) and int(grid[nr, nc]) == line_col for nr, nc in neighbors8(r, c)):
                    fillable = True
                    break
            if fillable:
                break
        if not fillable:
            continue
        if touches_border and not touches_bg_corner:
            edge_touch_fillable += 1
        if not touches_border:
            closed_fillable += 1
    return closed_fillable, edge_touch_fillable


def _behavior_profile(grid: np.ndarray) -> dict[str, int]:
    grid = np.asarray(grid, dtype=int)
    corner_bg = _corner_background(grid)
    mode_bg = int(get_bg(grid))
    new_out = _solve_with_rules(grid, corner_bg, outside_by_edge=True)
    closed_fillable, edge_touch_fillable = _fillable_profile(grid, corner_bg)
    try:
        wrong_bg_out = _solve_with_rules(grid, mode_bg, outside_by_edge=True)
        diff_vs_mode_bg = int(not np.array_equal(new_out, wrong_bg_out))
    except Exception:
        diff_vs_mode_bg = 1
    try:
        wrong_outside_out = _solve_with_rules(grid, corner_bg, outside_by_edge=False)
        diff_vs_corner_outside = int(not np.array_equal(new_out, wrong_outside_out))
    except Exception:
        diff_vs_corner_outside = 1
    return {
        "corner_bg": corner_bg,
        "mode_bg": mode_bg,
        "bg_disagree": int(corner_bg != mode_bg),
        "closed_fillable": closed_fillable,
        "edge_touch_fillable": edge_touch_fillable,
        "diff_vs_mode_bg": diff_vs_mode_bg,
        "diff_vs_corner_outside": diff_vs_corner_outside,
    }


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    bg = rng.choice(BG_COLORS)
    palette = [c for c in range(10) if c != bg]
    rng.shuffle(palette)

    seed_col = palette.pop()
    line_col = palette.pop()
    num_frame_cols = rng.randint(1, min(FRAME_COLORS_MAX, len(palette)))
    frame_colors_pool = [palette.pop() for _ in range(num_frame_cols)]
    distractor_line_col = line_col
    panel_col = palette.pop()

    canvas = np.full((h, w), EMPTY, dtype=int)

    corner = rng.choice(CORNERS)
    lead_frame_col = rng.choice(frame_colors_pool)
    prefix_path = _place_seed_and_pocket(canvas, seed_col, line_col, lead_frame_col, corner, rng)
    if prefix_path is None:
        return None
    if _place_mode_breaker_slab(canvas, panel_col, corner) is None:
        return None

    anchor = prefix_path[-1]
    extra_target = rng.randint(*LINE_LEN_RANGE)
    grown = _grow_line(canvas, anchor, line_col, extra_target, rng)
    line_path = prefix_path[:-1] + grown
    if len(line_path) < MIN_TOTAL_LINE_LEN:
        return None

    if _try_attach_frame_exhaustive(canvas, line_path, line_col, rng.choice(frame_colors_pool), rng) is None:
        return None
    for _ in range(rng.randint(*ATTACHED_LOOP_COUNT)):
        _try_attach_frame(canvas, line_path, line_col, rng.choice(frame_colors_pool), rng)

    if rng.random() < DISTRACTOR_NETWORK_PROB:
        _try_place_distractor_network(canvas, distractor_line_col, frame_colors_pool, rng)
    for _ in range(rng.randint(1, DISTRACTOR_SOLO_MAX)):
        _try_attach_distractor_frame(canvas, rng.choice(frame_colors_pool), rng)

    grid = np.where(canvas >= 0, canvas, bg).astype(int)

    if _corner_background(grid) != bg:
        return None
    if int(get_bg(grid)) == bg:
        return None
    if int((grid == seed_col).sum()) != 1:
        return None
    if any(int(grid[r, c]) != bg for r, c in ((_seed_corner_rc(c, h, w)) for c in CORNERS) if (r, c) != _seed_corner_rc(corner, h, w)):
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None

    profile = _behavior_profile(grid)
    if profile["bg_disagree"] != 1:
        return None
    if profile["closed_fillable"] < 1 or profile["edge_touch_fillable"] < 1:
        return None
    if profile["diff_vs_mode_bg"] != 1 or profile["diff_vs_corner_outside"] != 1:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted attempts")
