# Variant of 8b7bacbf
# Dimension: D3 — 腔体填充方式 (trajectory)
# Change: flood-fill——将连通腔体内部所有背景像素用种子颜色完全填满... → 交替色填充

import random
from collections import Counter

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4, neighbors8

# -----------------------------------------------------------------------------
# Design analysis for the D3 variant
# -----------------------------------------------------------------------------
# Source-task note:
# - The requested reference task `13e47133` exists in the allowed resources only
#   as `13e47133.json`; no local solver file is available to inspect. The
#   alternating-pattern baseline below is therefore derived from its visible
#   I/O behavior and from the explicit mechanism tag text:
#   "paint s.color ... alternate with region background color ...".
#
# a) Concrete behavior and trace
# - Input objects remain the same as the original task: unique border seed
#   pixels, one 8-connected line per seed, and single-color frame loops whose
#   cavities are background-colored in the input.
# - Output changes only at the last painting stage. Instead of filling every
#   cavity cell with the seed color, each filled cavity becomes a local
#   checkerboard using exactly two colors:
#   1. seed color = the unique border pixel color for that traced component
#   2. alternate color = the cavity's boundary/ring color
# - The checkerboard is anchored locally per cavity: the lexicographically
#   top-left cavity cell gets the seed color, then colors alternate by
#   Manhattan parity within that cavity.
# - Example trace:
#   Original: seed color 7 reaches a frame of color 3 enclosing a 3x2 cavity.
#   Original solver writes all 6 cavity cells to 7.
#   Variant solver writes:
#       7 3
#       3 7
#       7 3
#   The upstream detection pipeline is unchanged; only the write operation is
#   different, and the second color is recoverable from the visible frame.
#
# b) Complexity calibration
# - The rule uses exactly two colors per filled cavity, so it is visibly
#   different from flood-fill without exploding the hypothesis space.
# - The second color is not arbitrary: it is re-used from the frame boundary,
#   which gives the solver and the human a local cue.
# - Checkerboard is preferable to a global stripe direction because it avoids
#   ambiguity about whether pattern direction depends on seed position, path
#   direction, or global axes.
#
# c) Implicit-assumption audit of the original `main()`
# - Seed detection via "unique non-bg border pixel" still holds.
# - Line-color inference from 4-neighbors of the seed still holds.
# - Tracing the seed's 8-connected non-bg component still holds.
# - Cavity extraction as 4-connected background regions with non-empty non-bg
#   boundary still holds.
# - `len(bcols) == 1` mattered less before because every filled pixel received
#   the seed color anyway. In the variant it matters more, because the boundary
#   color is now part of the output pattern. We keep this guard.
# - The original unconditional write `out[r, c] = seed_col` would destroy the
#   variant because every cell would collapse to one color. It must be replaced
#   by a parity-dependent write using both seed and boundary colors.
#
# d) Placement legality
# - Generator still places frames only when ring and cavity cells are empty,
#   cavities are not 8-adjacent to the attached line, and rings touch the line
#   from outside. Those constraints remain sufficient for the solver to recover
#   the same cavity set.
# - Because the output now uses the ring color inside cavities too, the cavity
#   must stay separated from unrelated objects; the original emptiness and
#   margin checks already provide that.
#
# e) Visual observability
# - A 2x2 cavity already shows both colors, so the new behavior is visible even
#   on the smallest allowed cavities. The generator nevertheless biases cavity
#   sizes toward 3 and 4 to make the alternation easier to see in a few-shot
#   ARC setting.
# - The pattern rule is constant across examples: top-left cavity cell is seed
#   color, adjacent cavity cells alternate.
#
# f) Visual cleanliness
# - No new marker objects are introduced. The only new visual content is inside
#   regions that were already designated as fill targets in the original task.
# - Using the frame color as the second fill color ties the new pixels to the
#   surrounding frame, so the result stays interpretable rather than noisy.
#
# g) Color-change inferability
# - Colors are structurally grounded:
#   seed color comes from the border singleton;
#   alternate color comes from the enclosing frame ring.
# - This avoids hidden numeric conventions and keeps the mapping local.
#
# 9) New assumptions introduced by the variant
# - For each fillable cavity, the solver assumes the boundary color set has size
#   exactly 1, because that unique boundary color becomes the second color of
#   the checkerboard. The generator enforces single-color rings.
# - The solver anchors parity using the cavity's top-left coordinate. The
#   generator uses rectangular cavities, so that anchor is visually obvious.
# - The solver assumes seed color and frame color differ. The generator draws
#   them from a no-reuse palette, so this is guaranteed.


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
    Fill enclosed regions along a line seeded by unique border pixels.

    Variant rule:
    - Detect the same fillable cavities as in the original task.
    - For each cavity attached to a seed's traced component, paint a local
      checkerboard using the seed color and the cavity boundary color.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    out = g.copy()

    bg_comps = _components_mask(g == bg, conn8=False)
    regions = []
    for pts in bg_comps:
        region = set(pts)
        touches = _touches_border(pts, h, w)
        boundary = set()
        for r, c in region:
            for nr, nc in neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in region and int(g[nr, nc]) != bg:
                    boundary.add((nr, nc))
        regions.append((region, boundary, touches, len(region)))

    corners = {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}
    outside_idxs = set()
    for i, (reg, _, _, _) in enumerate(regions):
        if any(p in reg for p in corners if int(g[p]) == bg):
            outside_idxs.add(i)

    cavities = []
    for i, (reg, bnd, _, _) in enumerate(regions):
        if i in outside_idxs or not bnd:
            continue
        bcols = {int(g[p]) for p in bnd}
        cavities.append((reg, bnd, bcols))

    for sr, sc in _seed_pixels(g, bg):
        seed_col = int(g[sr, sc])

        neigh_cols = []
        for nr, nc in neighbors4(sr, sc):
            if 0 <= nr < h and 0 <= nc < w:
                v = int(g[nr, nc])
                if v != bg:
                    neigh_cols.append(v)
        assert neigh_cols, "seed pixel must have at least one non-bg neighbor"
        line_col = Counter(neigh_cols).most_common(1)[0][0]

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
                    if 0 <= nr < h and 0 <= nc < w and int(g[nr, nc]) == line_col:
                        ok = True
                        break
                if ok:
                    break
            if not ok:
                continue

            frame_col = next(iter(bcols))
            _paint_alternating_cavity(out, cav, seed_col, frame_col, bg)

    return out


# --------------- generator (attempt_1_distractor_network) ---------------

# ========================== CFG ==========================
GRID_H_RANGE = (8, 30)
GRID_W_RANGE = (8, 30)
BG_COLORS = [0, 5]

SEED_COUNT_WEIGHTS = [1, 2, 2]
MIN_LINE_LEN = 4
LINE_LEN_RANGE = (8, 35)

FRAME_COLORS_MAX = 4
TOTAL_LOOPS_MAX = 9
# Slightly larger cavities than the original make the alternating pattern
# easier to observe while keeping the same object family.
CAVITY_SIZE_WEIGHTS = [2, 2, 3, 3, 4]
DISTRACTOR_CAVITY_WEIGHTS = [2, 2, 3, 4]
FRAME_PLACEMENT_PAD = 1
FRAME_ATTACH_TRIES = 40
DISTRACTOR_FRAME_TRIES = 40

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

GEN_RETRIES = 300
# =========================================================

DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
ORTHO_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _frame_ring_coords(r0: int, c0: int, ch: int, cw: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    H, W = ch + 2, cw + 2
    ring = []
    for dr in range(H):
        for dc in range(W):
            on_bdr = dr == 0 or dr == H - 1 or dc == 0 or dc == W - 1
            corner = (dr in (0, H - 1)) and (dc in (0, W - 1))
            if on_bdr and not corner:
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
    return [
        (dr, dc)
        for dr in range(-1, fh + 1)
        for dc in range(-1, fw + 1)
        if not (0 <= dr < fh and 0 <= dc < fw)
    ]


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
        ch = rng.choice(CAVITY_SIZE_WEIGHTS)
        cw = rng.choice(CAVITY_SIZE_WEIGHTS)
        lp = rng.choice(line_path)
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


def _try_attach_distractor_frame(
    canvas: np.ndarray,
    frame_col: int,
    rng: random.Random,
    max_tries: int = DISTRACTOR_FRAME_TRIES,
) -> bool | None:
    h, w = canvas.shape
    for _ in range(max_tries):
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
    return None


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
    attached_cavity_areas = []
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
            placed = _try_attach_frame(canvas, path, line_col, fc, rng)
            if placed is not None:
                attached += 1
                _, cavity = placed
                attached_cavity_areas.append(len(cavity))
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

    # Ensure the new behavior is visually active on the sample: every attached
    # cavity is at least 2x2 already, and at least one cavity should be larger
    # than 2 cells to expose multiple alternations rather than a trivial swap.
    if not attached_cavity_areas or max(attached_cavity_areas) < 6:
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
