# Combo variant of 8b7bacbf (index 5)
# combo_spec: D3/1+D10/0
# Name: 交替色填充+不规则腔体

import random
from collections import Counter

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4, neighbors8

# -----------------------------------------------------------------------------
# Design analysis for the combo variant
# -----------------------------------------------------------------------------
# a) Concrete behavior and trace
# - Base task objects stay recognizable: unique border seed pixels, one thin
#   8-connected line per seed, and single-color loops that enclose background.
# - The two changes are coupled at the cavity stage:
#   1. cavities are no longer mostly tiny rectangles; they are irregular
#      4-connected polyominoes such as L/T/stair/hook shapes.
#   2. filling is no longer monochrome; each filled cavity becomes a local
#      checkerboard made from the seed color and the background color.
# - Example trace:
#   original: a border seed of color 7 reaches a loop, solver finds the inside
#   bg region, and writes all cavity cells to 7.
#   combo: the same tracing logic finds an irregular L-shaped cavity whose ring
#   is color 3. Let the cavity cells be {(2,2),(2,3),(3,2),(4,2),(4,3)} and
#   let the background color be 0.
#   The lexicographically top-left cavity cell (2,2) is anchored to seed color
#   7, so output becomes:
#       7 0
#       0
#       7 0
#   Difference from the original is therefore visible in both geometry and the
#   painted interior.
#
# b) Complexity calibration
# - Cavity shapes are sampled from a small template bank with rotation/mirror,
#   not from unrestricted random blobs. This keeps repeated visual evidence
#   across training examples while still being clearly non-rectangular.
# - The alternating fill always uses exactly two recoverable colors:
#   seed color and the already-visible background color. No hidden color
#   mapping is added.
# - The parity anchor is local to each cavity (its top-left occupied cell),
#   which is easier to infer from 2-4 ARC examples than a less observable
#   dependency on path direction or color id.
#
# c) Implicit-assumption audit of the original solver
# - Seed detection via "unique non-bg border pixel" still holds.
# - Inferring the line color from non-bg 4-neighbors of the seed still holds.
# - Tracing the seed's full non-bg 8-connected component still holds.
# - What changes:
#   * Rectangular geometry can no longer be assumed anywhere.
#   * "Outside bg" should be detected by border contact, not only by containing
#     a background corner, because irregular rings can create non-corner outer
#     pockets.
#   * `len(bcols) == 1` remains important so a human can still recognize one
#     coherent enclosing ring around each cavity.
#   * Writing every cavity pixel to the seed color is no longer valid; painting
#     must depend on parity relative to a cavity anchor.
#
# d) Placement legality
# - Generator constructs each irregular loop by taking a cavity mask and adding
#   a 4-neighbor shell around its exposed edges; this guarantees a closed
#   single-color ring around exactly one cavity region.
# - Ring cells and cavity cells must be empty, in-bounds, and separated from
#   foreign non-bg objects.
# - Cavity cells cannot be 8-adjacent to the line color, so the line touches
#   the loop only from outside and never becomes part of the cavity boundary.
# - Distractor loops use the same irregular construction but are placed with a
#   margin so they do not accidentally connect to any seeded network.
#
# e) Visual observability
# - The fusion remains visually learnable: the solver still follows obvious
#   seed-to-line-to-loop connectivity, and the only new fill colors come from
#   objects already visible around the cavity.
# - Every fillable cavity is irregular and has area at least 5, so the result
#   cannot visually collapse into the original 2x2 solid-fill regime.
# - The generator additionally rejects samples where the alternating rule would
#   be trivial or where a rectangular fillable cavity sneaks back in.
#
# f) Visual cleanliness
# - No extra markers are added. Variation is confined to the same semantic
#   regions as the source task: loop interiors.
# - The palette is sampled without reusing seed colors as line/frame colors, so
#   the seed-colored parity cells stay legible against the surrounding line
#   network and the preserved background cells.
# - Requiring disjoint seeded components avoids ambiguous ownership of a cavity.
#
# g) Change inferability
# - The second fill color is not arbitrary: it is the pre-existing background
#   color already visible outside every object.
# - The alternation rule is not numeric; it is spatial and anchored by the
#   cavity's visible top-left occupied cell.
# - Because all target cavities are irregular but still 4-connected, a human can
#   infer "fill the whole enclosed shape, but in an alternating pattern" rather
#   than memorizing a list of hand-authored shapes.
#
# 9) New assumptions introduced by the combo variant
# - Each fillable cavity must have a single-color boundary.
# - Each fillable cavity must be 4-connected, not touch the grid border, and be
#   uniquely associated with one seed network.
# - Seed color must differ from the background color so the alternating pattern
#   is observable.
# - The lexicographically smallest cavity cell is treated as the local anchor;
#   the generator therefore keeps cavities simply connected and visually clear
#   enough that this anchor is not ambiguous.


def _components_mask(mask: np.ndarray, conn8: bool) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8 if conn8 else 4)]


def _touches_border(pts: list[tuple[int, int]], h: int, w: int) -> bool:
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


def _nonbg_component8_from(g: np.ndarray, start: tuple[int, int], bg: int) -> set[tuple[int, int]]:
    h, w = g.shape
    st = [start]
    vis = {start}
    while st:
        r, c = st.pop()
        for nr, nc in neighbors8(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in vis and int(g[nr, nc]) != bg:
                vis.add((nr, nc))
                st.append((nr, nc))
    return vis


def _bg_regions_with_boundaries(g: np.ndarray, bg: int) -> list[tuple[set[tuple[int, int]], set[tuple[int, int]], bool]]:
    h, w = g.shape
    regions = []
    for pts in _components_mask(g == bg, conn8=False):
        region = set(pts)
        boundary = set()
        for r, c in region:
            for nr, nc in neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in region and int(g[nr, nc]) != bg:
                    boundary.add((nr, nc))
        regions.append((region, boundary, _touches_border(pts, h, w)))
    return regions


def _line_color_for_seed(g: np.ndarray, sr: int, sc: int, bg: int) -> int:
    h, w = g.shape
    neigh_cols = []
    for nr, nc in neighbors4(sr, sc):
        if 0 <= nr < h and 0 <= nc < w:
            v = int(g[nr, nc])
            if v != bg:
                neigh_cols.append(v)
    assert neigh_cols, "seed pixel must have at least one non-bg neighbor"
    return Counter(neigh_cols).most_common(1)[0][0]


def _paint_alternating_cavity(
    out: np.ndarray,
    cavity: set[tuple[int, int]],
    seed_col: int,
    bg: int,
) -> None:
    anchor_r, anchor_c = min(cavity)
    for r, c in cavity:
        parity = (r - anchor_r + c - anchor_c) % 2
        out[r, c] = seed_col if parity == 0 else bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace border-seeded line networks to closed loops and repaint their irregular
    interiors with a seed/background alternating pattern.

    Input structure:
    - The most common color is background.
    - One or two unique single-pixel seed colors lie on the grid border.
    - Each seed touches a thin 8-connected line of another color.
    - That line is part of a larger non-background component containing one or
      more single-color closed rings whose enclosed background regions may be
      irregular L/T/stair-like cavities rather than rectangles.

    Rule:
    - For each seed, trace its full non-background 8-connected component.
    - Find every 4-connected background cavity that does not touch the grid
      border, has a single-color non-background boundary, and whose boundary is
      touched by the traced component from the line side.
    - Repaint each such cavity with an alternating checkerboard that uses the
      seed color and the background color, anchored at the cavity's
      lexicographically top-left cell.

    Key insight:
    - The difficult part is no longer "fill a rectangle"; it is recognizing that
      arbitrary enclosed background shapes still belong to a seed's line network,
      then keeping every other cavity cell as background under a local parity
      pattern.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    cavities = []
    for region, boundary, touches_border in _bg_regions_with_boundaries(g, bg):
        if touches_border or not boundary:
            continue
        bcols = {int(g[p]) for p in boundary}
        cavities.append((region, boundary, bcols))

    for sr, sc in _seed_pixels(g, bg):
        seed_col = int(g[sr, sc])
        line_col = _line_color_for_seed(g, sr, sc, bg)
        traced = _nonbg_component8_from(g, (sr, sc), bg)

        for cav, boundary, bcols in cavities:
            if len(bcols) != 1:
                continue
            boundary_in_traced = boundary & traced
            if not boundary_in_traced:
                continue

            ok = False
            for r, c in boundary_in_traced:
                for nr, nc in neighbors8(r, c):
                    if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1] and int(g[nr, nc]) == line_col:
                        ok = True
                        break
                if ok:
                    break
            if not ok:
                continue

            _paint_alternating_cavity(out, cav, seed_col, bg)

    return out


# ========================== CFG ==========================
GRID_H_RANGE = (8, 30)
GRID_W_RANGE = (8, 30)
BG_COLORS = [0, 5]

SEED_COUNT_WEIGHTS = [1, 2, 2]
MIN_LINE_LEN = 4
LINE_LEN_RANGE = (8, 35)

FRAME_COLORS_MAX = 4
TOTAL_LOOPS_MAX = 9
FRAME_PLACEMENT_PAD = 1
FRAME_ATTACH_TRIES = 80
DISTRACTOR_FRAME_TRIES = 80

DIAG_WEIGHT_ALIGNED = 5
DIAG_WEIGHT_OTHER = 1
ORTHO_WEIGHT_ALIGNED = 2
ORTHO_WEIGHT_OTHER = 1
BORDER_AVOIDANCE_PROB = 0.85
BIAS_FLIP_PROB = 0.3

DISTRACTOR_NETWORK_PROB = 0.6
DISTRACTOR_NETWORK_TRIES = 50
DISTRACTOR_LINE_LEN = (5, 15)
DISTRACTOR_SOLO_MAX = 2
DISTRACTOR_NETWORK_FRAMES = (1, 3)

GEN_RETRIES = 500
# =========================================================

DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
ORTHO_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


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


def _random_cavity_mask(rng: random.Random) -> np.ndarray:
    for _ in range(32):
        base = rng.choices(CAVITY_TEMPLATE_BANK, weights=CAVITY_TEMPLATE_WEIGHTS, k=1)[0].copy()
        if rng.random() < 0.5:
            base = np.fliplr(base)
        rot = rng.randrange(4)
        if rot:
            base = np.rot90(base, rot)
        base = _trim_mask(base)
        if _mask_has_single_cavity(base):
            return base
    raise RuntimeError("failed to sample valid irregular cavity mask")


def _region_is_rectangle(region: set[tuple[int, int]]) -> bool:
    rows = [r for r, _ in region]
    cols = [c for _, c in region]
    return len(region) == (max(rows) - min(rows) + 1) * (max(cols) - min(cols) + 1)


def _irregular_frame_coords(r0: int, c0: int, cavity_mask: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    mh, mw = cavity_mask.shape
    ring = set()
    cavity = []
    for rr in range(mh):
        for cc in range(mw):
            if not cavity_mask[rr, cc]:
                continue
            cavity.append((r0 + 1 + rr, c0 + 1 + cc))
            for nr, nc in neighbors4(rr + 1, cc + 1):
                if 1 <= nr < mh + 1 and 1 <= nc < mw + 1 and cavity_mask[nr - 1, nc - 1]:
                    continue
                ring.add((r0 + nr, c0 + nc))
    return sorted(ring), cavity


def _mask_has_single_cavity(cavity_mask: np.ndarray) -> bool:
    ring, cavity = _irregular_frame_coords(0, 0, cavity_mask)
    local_h, local_w = cavity_mask.shape[0] + 2, cavity_mask.shape[1] + 2
    local = np.zeros((local_h, local_w), dtype=int)
    for rr, cc in ring:
        local[rr, cc] = 1
    inner = []
    for pts in _components_mask(local == 0, conn8=False):
        if not _touches_border(pts, local_h, local_w):
            inner.append(set(pts))
    return len(inner) == 1 and inner[0] == set(cavity)


def _validate_frame_placement(
    canvas: np.ndarray,
    ring: list[tuple[int, int]],
    cavity: list[tuple[int, int]],
    line_col: int,
    h: int,
    w: int,
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


def _grow_line(grid: np.ndarray, start: tuple[int, int], line_col: int, target_len: int, rng: random.Random) -> list[tuple[int, int]]:
    h, w = grid.shape
    path = [start]
    dr_bias = 1 if start[0] < h // 2 else -1
    dc_bias = 1 if start[1] < w // 2 else -1
    cur = start
    path_set = {start}
    while len(path) < target_len:
        dirs = []
        for ddr, ddc in DIAG_DIRS:
            aligned = ddr == dr_bias or ddc == dc_bias
            w_d = DIAG_WEIGHT_ALIGNED if aligned else DIAG_WEIGHT_OTHER
            dirs.extend([(ddr, ddc)] * w_d)
        for ddr, ddc in ORTHO_DIRS:
            aligned = (ddr == dr_bias and ddc == 0) or (ddc == dc_bias and ddr == 0)
            w_d = ORTHO_WEIGHT_ALIGNED if aligned else ORTHO_WEIGHT_OTHER
            dirs.extend([(ddr, ddc)] * w_d)
        rng.shuffle(dirs)
        placed = False
        for ddr, ddc in dirs:
            nr, nc = cur[0] + ddr, cur[1] + ddc
            if not _in_bounds(h, w, nr, nc) or grid[nr, nc] != -1:
                continue
            if any((nnr, nnc) != cur and (nnr, nnc) in path_set for nnr, nnc in neighbors8(nr, nc)):
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


def _halo_offsets(fh: int, fw: int) -> list[tuple[int, int]]:
    return [(dr, dc) for dr in range(-1, fh + 1) for dc in range(-1, fw + 1) if not (0 <= dr < fh and 0 <= dc < fw)]


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
    for _ in range(max_tries):
        cavity_mask = _random_cavity_mask(rng)
        fh, fw = cavity_mask.shape[0] + 2, cavity_mask.shape[1] + 2
        lp = rng.choice(line_path)
        offsets = _halo_offsets(fh, fw)
        rng.shuffle(offsets)
        for dr, dc in offsets:
            r0 = lp[0] - dr
            c0 = lp[1] - dc
            if r0 < pad or c0 < pad or r0 + fh > h - pad or c0 + fw > w - pad:
                continue
            ring, cavity = _irregular_frame_coords(r0, c0, cavity_mask)
            if not _validate_frame_placement(canvas, ring, cavity, line_col, h, w):
                continue
            for rr, cc in ring:
                canvas[rr, cc] = frame_col
            return ring, cavity
    return None


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


def _seed_components_are_disjoint(grid: np.ndarray, lines_info: list[tuple[int, tuple[int, int], int, list[tuple[int, int]]]], bg: int) -> bool:
    traces = []
    seed_positions = [seed_rc for _, seed_rc, _, _ in lines_info]
    for _, seed_rc, _, _ in lines_info:
        traced = _nonbg_component8_from(grid, seed_rc, bg)
        for other_seed in seed_positions:
            if other_seed != seed_rc and other_seed in traced:
                return False
        traces.append(traced)
    for i in range(len(traces)):
        for j in range(i + 1, len(traces)):
            if traces[i] & traces[j]:
                return False
    return True


def _fillable_cavity_profile(grid: np.ndarray, bg: int) -> tuple[int, int, int, int, int]:
    seed_infos = []
    for sr, sc in _seed_pixels(grid, bg):
        traced = _nonbg_component8_from(grid, (sr, sc), bg)
        line_col = _line_color_for_seed(grid, sr, sc, bg)
        seed_infos.append((traced, line_col))

    fillable = 0
    irregular = 0
    rectangular = 0
    max_area = 0
    alt_cells_total = 0

    for region, boundary, touches_border in _bg_regions_with_boundaries(grid, bg):
        if touches_border or not boundary:
            continue
        bcols = {int(grid[p]) for p in boundary}
        if len(bcols) != 1:
            continue
        for traced, line_col in seed_infos:
            boundary_in_traced = boundary & traced
            if not boundary_in_traced:
                continue
            ok = False
            for r, c in boundary_in_traced:
                if any(
                    0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and int(grid[nr, nc]) == line_col
                    for nr, nc in neighbors8(r, c)
                ):
                    ok = True
                    break
            if not ok:
                continue
            fillable += 1
            if _region_is_rectangle(region):
                rectangular += 1
            else:
                irregular += 1
            max_area = max(max_area, len(region))
            anchor_r, anchor_c = min(region)
            alt_cells_total += sum((r - anchor_r + c - anchor_c) % 2 == 1 for r, c in region)
            break
    return fillable, irregular, rectangular, max_area, alt_cells_total


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
        canvas[first_rc] = line_colors[i]
        target_len = rng.randint(LINE_LEN_RANGE[0], min(LINE_LEN_RANGE[1], h + w))
        path = _grow_line(canvas, first_rc, line_colors[i], target_len, rng)
        if len(path) < MIN_LINE_LEN:
            return None
        lines_info.append((seed_colors[i], seed_rc, line_colors[i], path))

    total_loops_target = rng.randint(num_seeds, TOTAL_LOOPS_MAX)
    num_distractors_target = rng.randint(0, max(0, total_loops_target - num_seeds))
    num_attached_target = total_loops_target - num_distractors_target
    per_seed = _distribute_loop_budget(num_seeds, num_attached_target, rng)

    for i, (_, _, line_col, path) in enumerate(lines_info):
        attached = 0
        for _ in range(per_seed[i]):
            fc = rng.choice(frame_colors_pool)
            if _try_attach_frame(canvas, path, line_col, fc, rng) is not None:
                attached += 1
        if attached == 0:
            return None

    if rng.random() < DISTRACTOR_NETWORK_PROB:
        _try_place_distractor_network(canvas, rng.choice(line_colors), frame_colors_pool, rng)
    else:
        for _ in range(rng.randint(0, DISTRACTOR_SOLO_MAX)):
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

    fillable, irregular_fillable, rectangular_fillable, max_area, alt_cells_total = _fillable_cavity_profile(grid, bg)
    if fillable == 0:
        return None
    if irregular_fillable == 0 or rectangular_fillable != 0:
        return None
    if max_area < 5 or alt_cells_total == 0:
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError("generate_input: exhausted attempts")
