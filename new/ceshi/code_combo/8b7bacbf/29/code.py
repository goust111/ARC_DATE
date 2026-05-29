# Combo variant of 8b7bacbf (index 29)
# combo_spec: D3/1+D5/1+D10/0
# Name: 交替色填充+多色边界+不规则腔体

import random
from collections import Counter

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4, neighbors8


# -----------------------------------------------------------------------------
# Design analysis for the combo variant
# -----------------------------------------------------------------------------
# a) Concrete behavior and trace
# - Original task:
#   a border singleton seed identifies one 8-connected non-bg drawing; any
#   enclosed bg cavity whose boundary belongs to that drawing is flood-filled
#   with the seed color.
# - Combo task:
#   1. The cavity is no longer a near-rectangular 2x2/3x3 box; it is an
#      irregular 4-connected polyomino such as L/T/zigzag/stair/hook.
#   2. Its surrounding ring is no longer single-color; exactly two boundary
#      colors appear on the ring.
#   3. The line touches only one visible boundary-color group of that ring.
#   4. The interior is not solid-filled. Instead, the cavity is painted by a
#      local checkerboard anchored at the cavity's top-left cell:
#        seed color / touched boundary color / seed color / ...
# - Trace example:
#   a seed of color 7 reaches an irregular ring whose touched cap is color 3
#   and whose remaining boundary is color 8. The original solver would skip
#   this cavity because `len(bcols) != 1`. The combo solver instead detects
#   that the traced line touches only boundary color 3, then paints the
#   irregular cavity as a 7/3 checkerboard.
#
# b) Complexity calibration
# - The multicolor rule is limited to exactly 2 boundary colors. This keeps the
#   condition visible without creating a large hidden mapping table.
# - Irregular cavities come from a small template bank with rotation/mirroring,
#   so examples repeat a recognizable object family instead of arbitrary noise.
# - The alternating fill is always checkerboard with the same local anchor rule;
#   no per-example stripe direction or global phase is introduced.
#
# c) Implicit-assumption audit of the original `main()`
# - Seed detection as a unique non-bg border pixel still holds and is preserved.
# - Inferring `line_col` from the seed's 4-neighbors still holds because the
#   generator still places a single orthogonal line pixel next to the seed.
# - Tracing one 8-connected non-bg component from the seed still holds.
# - Background-region extraction still works, but "outside" must now be defined
#   as "touches any border cell", not only "contains a bg corner". Irregular
#   rings make non-corner outside pockets more plausible.
# - The original `len(bcols) == 1` assumption is broken by D5 and must become
#   `len(bcols) == 2`.
# - The original "any traced boundary pixel near line color" test is too weak
#   once the ring is multicolor; we must isolate which boundary color is
#   actually touched by external traced line pixels.
# - The original solid write `out[r, c] = seed_col` is broken by D3 and must be
#   replaced by a parity-based two-color paint using the touched boundary color.
#
# d) Placement legality
# - Every cavity mask is 4-connected and validated to yield exactly one sealed
#   inner bg region when converted into a shell ring.
# - Ring cells and cavity cells must be empty before placement.
# - Cavity pixels may not be 8-adjacent to the attached line color, ensuring
#   the ring, not the line, forms the enclosure.
# - The line may touch only a designated extremal side-group of the ring; if
#   contact spills into another side-group, that placement is rejected.
# - Distractor frames require a one-cell empty margin so they do not attach to
#   any existing network accidentally.
#
# e) Visual observability
# - A human can see three consistent cues across examples:
#   1. only seed-connected irregular multicolor rings change;
#   2. inside each changed ring, one fill color is the seed color;
#   3. the second fill color is copied from the boundary color at the side where
#      the line physically touches the ring.
# - This is visible from geometry and local adjacency; no hidden color ordering
#   or numeric ID convention is used.
#
# f) Visual cleanliness
# - The generator keeps objects separated by the original margin logic, so the
#   line, ring, cavity and fill pattern stay visually separable.
# - Boundary colors are only two-valued per ring, and the interior uses only
#   one of those two plus the seed color. That keeps the grids busy but legible.
#
# g) Color-change inferability
# - The fill colors are structurally grounded:
#   one is the border singleton seed, the other is the unique boundary color
#   touched by that seed's own line.
# - No arbitrary recoloring table is introduced; the second color is always
#   recoverable from the visible contact geometry.
#
# 9) New assumptions introduced by the combo solver
# - Fillable cavities must have exactly 2 boundary colors.
# - Among those 2 colors, traced external line contact must identify exactly 1
#   touched boundary color.
# - The cavity parity anchor is the lexicographically top-left cavity cell.
# - Seeded connected components must stay disjoint so one cavity is not claimed
#   by multiple seeds. The generator checks all of these conditions explicitly.


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


def _boundary_contact_colors(
    g: np.ndarray,
    boundary: set[tuple[int, int]],
    traced: set[tuple[int, int]],
    line_col: int,
) -> set[int]:
    h, w = g.shape
    touched: set[int] = set()
    for r, c in boundary:
        for nr, nc in neighbors8(r, c):
            if (
                0 <= nr < h
                and 0 <= nc < w
                and (nr, nc) in traced
                and (nr, nc) not in boundary
                and int(g[nr, nc]) == line_col
            ):
                touched.add(int(g[r, c]))
                break
    return touched


def _paint_alternating_cavity(
    out: np.ndarray,
    cavity: set[tuple[int, int]],
    seed_col: int,
    alt_col: int,
    bg: int,
) -> None:
    anchor_r, anchor_c = min(cavity)
    for r, c in cavity:
        if int(out[r, c]) != bg:
            continue
        parity = (r - anchor_r + c - anchor_c) % 2
        out[r, c] = seed_col if parity == 0 else alt_col


def main(grid: np.ndarray) -> np.ndarray:
    """
    Follow each unique border seed into its attached drawing and repaint only
    the enclosed irregular cavities that match the fused boundary rule.

    Input structure:
    - Background is the most common color.
    - Each actionable drawing has a single-pixel unique seed on the border and
      an 8-connected meandering line extending inward.
    - Some lines touch closed irregular frame rings whose boundary uses exactly
      two colors and encloses a background cavity.

    Rule:
    - For every seed, trace its full 8-connected non-background component.
    - Among non-border-touching background cavities, keep only those whose
      boundary has exactly two colors and whose traced line touches exactly one
      of those boundary colors from outside the ring.
    - Fill that cavity with a local checkerboard alternating between the seed
      color and the uniquely touched boundary color.

    Key insight:
    - The seed chooses the active network, while the line-ring contact geometry
      chooses the second fill color; the interior pattern is then a fixed local
      parity rule rather than a solid flood-fill.
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

        for cavity, boundary, bcols in cavities:
            if len(bcols) != 2:
                continue
            if not (boundary & traced):
                continue
            contact_cols = _boundary_contact_colors(g, boundary, traced, line_col)
            if len(contact_cols) != 1:
                continue
            alt_col = next(iter(contact_cols))
            _paint_alternating_cavity(out, cavity, seed_col, alt_col, bg)

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
MIN_FILLABLE_CAVITIES = 1

DIAG_WEIGHT_ALIGNED = 5
DIAG_WEIGHT_OTHER = 1
ORTHO_WEIGHT_ALIGNED = 2
ORTHO_WEIGHT_OTHER = 1
BORDER_AVOIDANCE_PROB = 0.85
BIAS_FLIP_PROB = 0.3

DISTRACTOR_NETWORK_PROB = 0.55
DISTRACTOR_NETWORK_TRIES = 50
DISTRACTOR_LINE_LEN = (5, 15)
DISTRACTOR_NETWORK_FRAMES = (1, 2)
DISTRACTOR_SOLO_MAX = 2

GEN_RETRIES = 500
# =========================================================

DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
ORTHO_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDES = ["top", "right", "bottom", "left"]


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _parse_mask(rows: list[str]) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


CAVITY_TEMPLATE_BANK = [
    _parse_mask(["110", "100", "111"]),
    _parse_mask(["111", "010", "010"]),
    _parse_mask(["011", "110", "100"]),
    _parse_mask(["1100", "0110", "0011"]),
    _parse_mask(["1110", "0010", "0110"]),
    _parse_mask(["1100", "1000", "1110"]),
    _parse_mask(["010", "111", "010"]),
]
CAVITY_TEMPLATE_WEIGHTS = [3, 3, 2, 2, 2, 2, 2]


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


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
    local_h = cavity_mask.shape[0] + 2
    local_w = cavity_mask.shape[1] + 2
    local = np.zeros((local_h, local_w), dtype=int)
    for rr, cc in ring:
        local[rr, cc] = 1
    inner = []
    for pts in _components_mask(local == 0, conn8=False):
        if not _touches_border(pts, local_h, local_w):
            inner.append(set(pts))
    return len(inner) == 1 and inner[0] == set(cavity)


def _random_cavity_mask(rng: random.Random) -> np.ndarray:
    for _ in range(32):
        idx = rng.choices(range(len(CAVITY_TEMPLATE_BANK)), weights=CAVITY_TEMPLATE_WEIGHTS, k=1)[0]
        mask = CAVITY_TEMPLATE_BANK[idx].copy()
        if rng.random() < 0.5:
            mask = np.fliplr(mask)
        rot = rng.randrange(4)
        if rot:
            mask = np.rot90(mask, rot)
        mask = _trim_mask(mask)
        if _mask_has_single_cavity(mask):
            return mask
    raise RuntimeError("failed to sample irregular cavity mask")


def _region_is_rectangle(region: set[tuple[int, int]]) -> bool:
    rows = [r for r, _ in region]
    cols = [c for _, c in region]
    return len(region) == (max(rows) - min(rows) + 1) * (max(cols) - min(cols) + 1)


def _local_side_groups(local_ring: list[tuple[int, int]]) -> dict[str, list[tuple[int, int]]]:
    rows = [r for r, _ in local_ring]
    cols = [c for _, c in local_ring]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)
    return {
        "top": [pt for pt in local_ring if pt[0] == min_r],
        "bottom": [pt for pt in local_ring if pt[0] == max_r],
        "left": [pt for pt in local_ring if pt[1] == min_c],
        "right": [pt for pt in local_ring if pt[1] == max_c],
    }


def _side_touch_candidates(
    side_group: list[tuple[int, int]],
    side: str,
    fh: int,
    fw: int,
) -> list[tuple[int, int]]:
    if side in ("top", "bottom"):
        mids = [pt for pt in side_group if 0 < pt[1] < fw - 1]
    else:
        mids = [pt for pt in side_group if 0 < pt[0] < fh - 1]
    return mids or side_group


def _translate_points(pts: list[tuple[int, int]], r0: int, c0: int) -> list[tuple[int, int]]:
    return [(r0 + r, c0 + c) for r, c in pts]


def _origin_from_line_touch(
    lp: tuple[int, int],
    local_touch: tuple[int, int],
    side: str,
) -> tuple[int, int]:
    lr, lc = local_touch
    if side == "top":
        return lp[0] + 1 - lr, lp[1] - lc
    if side == "bottom":
        return lp[0] - 1 - lr, lp[1] - lc
    if side == "left":
        return lp[0] - lr, lp[1] + 1 - lc
    if side == "right":
        return lp[0] - lr, lp[1] - 1 - lc
    raise ValueError(side)


def _validate_frame_placement(
    canvas: np.ndarray,
    ring: list[tuple[int, int]],
    cavity: list[tuple[int, int]],
    line_col: int | None,
    allowed_touch: set[tuple[int, int]] | None,
) -> bool:
    h, w = canvas.shape
    ring_set = set(ring)

    for rr, cc in ring:
        if canvas[rr, cc] != -1:
            return False
    for rr, cc in cavity:
        if canvas[rr, cc] != -1:
            return False

    if line_col is not None:
        for cr, cc in cavity:
            for nr, nc in neighbors8(cr, cc):
                if _in_bounds(h, w, nr, nc) and canvas[nr, nc] == line_col:
                    return False

    touched = set()
    for rr, cc in ring:
        for nr, nc in neighbors8(rr, cc):
            if not _in_bounds(h, w, nr, nc) or (nr, nc) in ring_set:
                continue
            v = int(canvas[nr, nc])
            if v == -1:
                continue
            if line_col is not None and v == line_col:
                touched.add((rr, cc))
                continue
            return False

    if line_col is None:
        return True
    if not touched:
        return False
    if allowed_touch is not None and not touched <= allowed_touch:
        return False
    return True


def _grow_line(
    grid: np.ndarray,
    start: tuple[int, int],
    line_col: int,
    target_len: int,
    rng: random.Random,
) -> list[tuple[int, int]]:
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
            dirs.extend([(ddr, ddc)] * (DIAG_WEIGHT_ALIGNED if aligned else DIAG_WEIGHT_OTHER))
        for ddr, ddc in ORTHO_DIRS:
            aligned = (ddr == dr_bias and ddc == 0) or (ddc == dc_bias and ddr == 0)
            dirs.extend([(ddr, ddc)] * (ORTHO_WEIGHT_ALIGNED if aligned else ORTHO_WEIGHT_OTHER))
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


def _choose_two_distinct(colors: list[int], rng: random.Random) -> tuple[int, int]:
    a, b = rng.sample(colors, 2)
    return int(a), int(b)


def _paint_bicolor_ring(
    canvas: np.ndarray,
    ring: list[tuple[int, int]],
    contact_group: set[tuple[int, int]],
    contact_col: int,
    other_col: int,
) -> None:
    for rr, cc in ring:
        canvas[rr, cc] = contact_col if (rr, cc) in contact_group else other_col


def _try_attach_bicolor_irregular_frame(
    canvas: np.ndarray,
    line_path: list[tuple[int, int]],
    line_col: int,
    frame_colors_pool: list[int],
    rng: random.Random,
    max_tries: int = FRAME_ATTACH_TRIES,
):
    h, w = canvas.shape
    pad = FRAME_PLACEMENT_PAD
    for _ in range(max_tries):
        cavity_mask = _random_cavity_mask(rng)
        local_ring, local_cavity = _irregular_frame_coords(0, 0, cavity_mask)
        fh = cavity_mask.shape[0] + 2
        fw = cavity_mask.shape[1] + 2
        side_groups = _local_side_groups(local_ring)
        contact_side = rng.choice(SIDES)
        touch_candidates = _side_touch_candidates(side_groups[contact_side], contact_side, fh, fw)
        if not touch_candidates:
            continue
        lp = rng.choice(line_path)
        local_touch = rng.choice(touch_candidates)
        r0, c0 = _origin_from_line_touch(lp, local_touch, contact_side)
        if r0 < pad or c0 < pad or r0 + fh > h - pad or c0 + fw > w - pad:
            continue
        ring = _translate_points(local_ring, r0, c0)
        cavity = _translate_points(local_cavity, r0, c0)
        contact_group = set(_translate_points(side_groups[contact_side], r0, c0))
        if not _validate_frame_placement(canvas, ring, cavity, line_col, contact_group):
            continue
        contact_col, other_col = _choose_two_distinct(frame_colors_pool, rng)
        _paint_bicolor_ring(canvas, ring, contact_group, contact_col, other_col)
        return {
            "ring": ring,
            "cavity": cavity,
            "area": len(cavity),
            "contact_color": contact_col,
            "other_color": other_col,
        }
    return None


def _try_place_solo_bicolor_irregular_frame(
    canvas: np.ndarray,
    frame_colors_pool: list[int],
    rng: random.Random,
    max_tries: int = DISTRACTOR_FRAME_TRIES,
) -> bool:
    h, w = canvas.shape
    for _ in range(max_tries):
        cavity_mask = _random_cavity_mask(rng)
        local_ring, local_cavity = _irregular_frame_coords(0, 0, cavity_mask)
        fh = cavity_mask.shape[0] + 2
        fw = cavity_mask.shape[1] + 2
        if h - fh - 1 < 1 or w - fw - 1 < 1:
            return False
        r0 = rng.randint(1, h - fh - 1)
        c0 = rng.randint(1, w - fw - 1)
        ring = _translate_points(local_ring, r0, c0)
        cavity = _translate_points(local_cavity, r0, c0)
        if not _validate_frame_placement(canvas, ring, cavity, None, None):
            continue
        side_groups = _local_side_groups(local_ring)
        contact_side = rng.choice(SIDES)
        contact_group = set(_translate_points(side_groups[contact_side], r0, c0))
        contact_col, other_col = _choose_two_distinct(frame_colors_pool, rng)
        _paint_bicolor_ring(canvas, ring, contact_group, contact_col, other_col)
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
            if _try_attach_bicolor_irregular_frame(canvas, path, line_col, frame_colors_pool, rng) is not None:
                attached += 1
        if attached > 0:
            return True
        for pr, pc in path:
            canvas[pr, pc] = -1
    return False


def _corner_config(corner_kind: str, h: int, w: int, rng: random.Random) -> tuple[tuple[int, int], tuple[int, int]]:
    if corner_kind == "TL":
        return (0, 0), rng.choice([(0, 1), (1, 0)])
    if corner_kind == "TR":
        return (0, w - 1), rng.choice([(0, w - 2), (1, w - 1)])
    if corner_kind == "BL":
        return (h - 1, 0), rng.choice([(h - 1, 1), (h - 2, 0)])
    if corner_kind == "BR":
        return (h - 1, w - 1), rng.choice([(h - 1, w - 2), (h - 2, w - 1)])
    raise ValueError(corner_kind)


def _distribute_loop_budget(num_seeds: int, total: int, rng: random.Random) -> list[int]:
    order = list(range(num_seeds))
    rng.shuffle(order)
    out = [0] * num_seeds
    remaining = total
    while remaining > 0 and any(out[i] == 0 for i in range(num_seeds)):
        for i in order:
            if out[i] == 0 and remaining > 0:
                out[i] += 1
                remaining -= 1
    while remaining > 0:
        out[rng.choice(order)] += 1
        remaining -= 1
    return out


def _seed_components_are_disjoint(
    grid: np.ndarray,
    lines_info: list[tuple[int, tuple[int, int], int, list[tuple[int, int]]]],
    bg: int,
) -> bool:
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


def _fillable_cavity_records(grid: np.ndarray, bg: int) -> list[dict[str, int | bool]]:
    cavities = []
    for region, boundary, touches_border in _bg_regions_with_boundaries(grid, bg):
        if touches_border or not boundary:
            continue
        cavities.append((region, boundary, {int(grid[p]) for p in boundary}))

    records = []
    used = set()
    for sr, sc in _seed_pixels(grid, bg):
        seed_col = int(grid[sr, sc])
        line_col = _line_color_for_seed(grid, sr, sc, bg)
        traced = _nonbg_component8_from(grid, (sr, sc), bg)
        for idx, (region, boundary, bcols) in enumerate(cavities):
            if idx in used or len(bcols) != 2 or not (boundary & traced):
                continue
            contact_cols = _boundary_contact_colors(grid, boundary, traced, line_col)
            if len(contact_cols) != 1:
                continue
            used.add(idx)
            records.append(
                {
                    "seed_color": seed_col,
                    "area": len(region),
                    "boundary_color_count": len(bcols),
                    "is_rectangle": _region_is_rectangle(region),
                    "alt_color": next(iter(contact_cols)),
                }
            )
    return records


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    bg = rng.choice(BG_COLORS)
    palette = [c for c in range(10) if c != bg]
    rng.shuffle(palette)

    num_seeds = rng.choice(SEED_COUNT_WEIGHTS)
    if len(palette) < num_seeds * 2 + 2:
        return None

    canvas = -np.ones((h, w), dtype=int)

    seed_colors = [palette.pop() for _ in range(num_seeds)]
    line_colors = [palette.pop() for _ in range(num_seeds)]

    all_corners = ["TL", "TR", "BL", "BR"]
    rng.shuffle(all_corners)
    corners_used = all_corners[:num_seeds]

    num_frame_cols = rng.randint(2, min(FRAME_COLORS_MAX, len(palette)))
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
    num_solo_distractors = rng.randint(0, min(DISTRACTOR_SOLO_MAX, max(0, total_loops_target - num_seeds)))
    attached_fillable_total = total_loops_target - num_solo_distractors
    per_seed = _distribute_loop_budget(num_seeds, attached_fillable_total, rng)

    placed_fillable_areas = []
    for i, (_, _, line_col, path) in enumerate(lines_info):
        attached = 0
        for _ in range(per_seed[i]):
            placed = _try_attach_bicolor_irregular_frame(canvas, path, line_col, frame_colors_pool, rng)
            if placed is not None:
                attached += 1
                placed_fillable_areas.append(int(placed["area"]))
        if attached == 0:
            return None

    if rng.random() < DISTRACTOR_NETWORK_PROB:
        _try_place_distractor_network(canvas, rng.choice(line_colors), frame_colors_pool, rng)
    else:
        for _ in range(num_solo_distractors):
            _try_place_solo_bicolor_irregular_frame(canvas, frame_colors_pool, rng)

    grid = np.where(canvas == -1, bg, canvas).astype(int)

    vals, cnts = np.unique(grid, return_counts=True)
    if int(vals[np.argmax(cnts)]) != bg:
        return None

    for seed_col, _, _, _ in lines_info:
        if int((grid == seed_col).sum()) != 1:
            return None

    if not _seed_components_are_disjoint(grid, lines_info, bg):
        return None

    fillable = _fillable_cavity_records(grid, bg)
    if len(fillable) < max(MIN_FILLABLE_CAVITIES, num_seeds):
        return None
    if any(rec["boundary_color_count"] != 2 or rec["is_rectangle"] for rec in fillable):
        return None
    if not placed_fillable_areas or max(placed_fillable_areas) < 5:
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None

    filled_nonseed = {
        int(out[r, c])
        for r, c in zip(*np.where((grid == bg) & (out != bg)))
        if int(out[r, c]) not in seed_colors
    }
    if not filled_nonseed:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError("generate_input: exhausted attempts")
