# Variant of 8b7bacbf
# Dimension: D14 — 腔体与路径的连接判定 (meta)
# Change: 腔体的边界像素必须与追踪到的路径连通分量有交集，且交集部分8邻域内必须有线路颜色像素... → 仅要求边界与路径4连通

import random
from collections import Counter

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4, neighbors8

"""
Design analysis for the D14 variant
===================================

spec_delta:
  original_connection_test:
    - A cavity was eligible if its frame boundary was in the seed-traced 8-connected
      non-background component and some traced boundary pixel had a line-color pixel
      in its 8-neighborhood.
    - Consequence: a frame that only touches the path at one corner still counts.
  new_connection_test:
    - Keep the same seed tracing and cavity extraction.
    - Change only the final attachment predicate: the cavity boundary must have a
      4-neighbor that is a traced line-color pixel.
    - Consequence: only edge-sharing line/frame contacts fill; corner-only contacts do not.

concrete_behavior:
  - A seed line can snake diagonally and a frame ring can sit one corner away from it.
  - In the old task that diagonal touch joined the ring to the traced 8-component and the
    8-neighborhood test accepted it, so the cavity filled.
  - In this variant the same ring is still part of the traced 8-component, but it has no
    orthogonal line contact, so it stays background.
  - Therefore samples must show both:
      1. orthogonally attached rings that fill
      2. diagonal-only attached rings that visibly touch the line at a corner but do not fill

complexity_calibration:
  - Keep 1-2 seeds, 8-connected meandering lines, single-color frame rings, and optional
    isolated distractor loops from the base task.
  - Add at least one diagonal-only attached ring per grid so the stricter rule is visible.
  - Also keep at least one orthogonally attached ring per seed so the positive case remains.
  - This preserves the original object vocabulary while making the new branch observable.

implicit_assumption_audit:
  - `_nonbg_component8_from(...)` is still valid: the task still traces the whole drawing
    with 8-connectivity, including diagonally attached rings.
  - The old `boundary & traced` check remains harmless but is no longer sufficient to decide
    fillability, because diagonal-only rings also satisfy it.
  - The old `neighbors8(...) == line_col` predicate is now wrong: it encodes the loosened
    corner-touch acceptance. Replace it with a 4-neighbor check against traced line-color pixels.
  - Generator-side placement must guarantee that diagonal-only rings truly have zero 4-neighbor
    line contacts anywhere on the ring; otherwise the intended distractors would accidentally fill.

placement_legality:
  - Ring and cavity cells must be empty before placement.
  - No cavity cell may be 8-adjacent to the line, so the line never directly borders the cavity.
  - Ring cells may not be 8-adjacent to foreign non-background colors.
  - For `ortho` frames: require at least one ring cell with a 4-neighbor line pixel.
  - For `diag_only` frames: require at least one ring cell with an 8-neighbor line pixel and
    require that no ring cell has a 4-neighbor line pixel.

visual_observability:
  - Humans can infer the rule from a few examples because some loops that look "connected"
    by a corner stay empty while side-touching loops fill with the seed color.
  - The generator avoids relying on color id ordering or hidden metadata; only geometry changes.

main_new_assumptions_checked_against_generator:
  - `main()` now assumes that each fillable cavity has at least one boundary pixel with a traced
    orthogonal line-color neighbor. `_validate_frame_placement(..., "ortho")` guarantees this.
  - `main()` also assumes diagonal-only distractors may still be in the traced 8-component but have
    no such orthogonal line contact. `_validate_frame_placement(..., "diag_only")` guarantees this.
  - Single-color frame boundaries remain required; the generator still draws each ring in one color.
"""


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


def _extract_cavities(g: np.ndarray, bg: int) -> list[tuple[set[tuple[int, int]], set[tuple[int, int]], set[int]]]:
    h, w = g.shape
    bg_comps = _components_mask(g == bg, conn8=False)
    regions = []
    for pts in bg_comps:
        region = set(pts)
        boundary = set()
        for r, c in region:
            for nr, nc in neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in region and int(g[nr, nc]) != bg:
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


def _line_contact_mode(
    g: np.ndarray,
    boundary: set[tuple[int, int]],
    traced: set[tuple[int, int]],
    line_col: int,
) -> str | None:
    h, w = g.shape
    has_diag_or_ortho = False
    has_ortho = False
    for r, c in boundary:
        for nr, nc in neighbors8(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) in traced and int(g[nr, nc]) == line_col:
                has_diag_or_ortho = True
                break
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) in traced and int(g[nr, nc]) == line_col:
                has_ortho = True
                break
    if has_ortho:
        return "ortho"
    if has_diag_or_ortho:
        return "diag_only"
    return None


def _attachment_stats(g: np.ndarray) -> dict[str, int]:
    bg = get_bg(g)
    cavities = _extract_cavities(g, bg)
    stats = {"ortho": 0, "diag_only": 0, "none": 0}
    for sr, sc in _seed_pixels(g, bg):
        neigh_cols = []
        for nr, nc in neighbors4(sr, sc):
            if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1]:
                v = int(g[nr, nc])
                if v != bg:
                    neigh_cols.append(v)
        if not neigh_cols:
            continue
        line_col = Counter(neigh_cols).most_common(1)[0][0]
        traced = _nonbg_component8_from(g, (sr, sc), bg)
        for _, boundary, bcols in cavities:
            if len(bcols) != 1:
                continue
            mode = _line_contact_mode(g, boundary, traced, line_col)
            if mode is None:
                continue
            stats[mode] += 1
    return stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill enclosed regions along a line seeded by unique border pixels.

    D14 change:
    - Only cavities whose frame boundary has a traced orthogonal line-color contact fill.
    - Corner-only line/frame touches stay empty even though the whole drawing is still
      traced with 8-connectivity from the seed.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()
    cavities = _extract_cavities(g, bg)

    for sr, sc in _seed_pixels(g, bg):
        seed_col = int(g[sr, sc])

        neigh_cols = []
        for nr, nc in neighbors4(sr, sc):
            if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1]:
                v = int(g[nr, nc])
                if v != bg:
                    neigh_cols.append(v)
        assert neigh_cols, "seed pixel must have at least one non-bg neighbor"
        line_col = Counter(neigh_cols).most_common(1)[0][0]
        traced = _nonbg_component8_from(g, (sr, sc), bg)

        for cav, boundary, bcols in cavities:
            if len(bcols) != 1:
                continue
            if _line_contact_mode(g, boundary, traced, line_col) != "ortho":
                continue
            for r, c in cav:
                if int(out[r, c]) == bg:
                    out[r, c] = seed_col

    return out


# --------------- generator ---------------

# ========================== CFG ==========================
GRID_H_RANGE = (8, 30)
GRID_W_RANGE = (8, 30)
BG_COLORS = [0, 5]

SEED_COUNT_WEIGHTS = [1, 2, 2]
MIN_LINE_LEN = 4
LINE_LEN_RANGE = (8, 35)

FRAME_COLORS_MAX = 4
TOTAL_LOOPS_MAX = 9
CAVITY_SIZE_WEIGHTS = [2, 2, 2, 2, 2, 3]
DISTRACTOR_CAVITY_WEIGHTS = [2, 2, 2, 3]
FRAME_PLACEMENT_PAD = 1
FRAME_ATTACH_TRIES = 40
DIAG_ATTACH_TRIES = 90
DISTRACTOR_FRAME_TRIES = 40
MAX_DIAG_ATTACHED = 2

DIAG_WEIGHT_ALIGNED = 5
DIAG_WEIGHT_OTHER = 1
ORTHO_WEIGHT_ALIGNED = 2
ORTHO_WEIGHT_OTHER = 1
BORDER_AVOIDANCE_PROB = 0.85
BIAS_FLIP_PROB = 0.3

DISTRACTOR_SOLO_MAX = 1
GEN_RETRIES = 300
# =========================================================

DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
ORTHO_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _in_bounds(h, w, r, c):
    return 0 <= r < h and 0 <= c < w


def _frame_ring_coords(r0, c0, ch, cw):
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


def _ring_contact_mode(canvas, ring, line_col, h, w):
    has8 = False
    has4 = False
    for rr, cc in ring:
        for nr, nc in neighbors8(rr, cc):
            if _in_bounds(h, w, nr, nc) and canvas[nr, nc] == line_col:
                has8 = True
                break
        for nr, nc in neighbors4(rr, cc):
            if _in_bounds(h, w, nr, nc) and canvas[nr, nc] == line_col:
                has4 = True
                break
    if has4:
        return "ortho"
    if has8:
        return "diag_only"
    return None


def _validate_frame_placement(canvas, ring, cavity, line_col, h, w, require_contact):
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

    mode = _ring_contact_mode(canvas, ring, line_col, h, w)
    if require_contact == "ortho":
        return mode == "ortho"
    if require_contact == "diag_only":
        return mode == "diag_only"
    if require_contact == "any8":
        return mode in {"ortho", "diag_only"}
    raise ValueError(require_contact)


def _grow_line(grid, start, line_col, target_len, rng):
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


def _halo_offsets(fh, fw):
    return [(dr, dc) for dr in range(-1, fh + 1) for dc in range(-1, fw + 1) if not (0 <= dr < fh and 0 <= dc < fw)]


def _try_attach_frame(canvas, line_path, line_col, frame_col, rng, require_contact="ortho", max_tries=FRAME_ATTACH_TRIES):
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
            if not _validate_frame_placement(canvas, ring, cavity, line_col, h, w, require_contact):
                continue
            for rr, cc in ring:
                canvas[rr, cc] = frame_col
            return ring, cavity
    return None


def _try_attach_distractor_frame(canvas, frame_col, rng, max_tries=DISTRACTOR_FRAME_TRIES):
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


def _corner_config(corner_kind, h, w, rng):
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


def _distribute_loop_budget(num_seeds, num_attached_target, rng):
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

    total_loops_target = rng.randint(num_seeds + 1, TOTAL_LOOPS_MAX)
    max_diag = min(MAX_DIAG_ATTACHED, total_loops_target - num_seeds)
    num_diag_target = rng.randint(1, max_diag)
    max_isolated = min(DISTRACTOR_SOLO_MAX, total_loops_target - num_diag_target - num_seeds)
    num_isolated_target = rng.randint(0, max_isolated)
    num_ortho_target = total_loops_target - num_diag_target - num_isolated_target

    per_seed_ortho = _distribute_loop_budget(num_seeds, num_ortho_target, rng)
    per_seed_diag = [0] * num_seeds
    for _ in range(num_diag_target):
        per_seed_diag[rng.randrange(num_seeds)] += 1

    for i, (_, _, line_col, path) in enumerate(lines_info):
        attached = 0
        for _ in range(per_seed_ortho[i]):
            fc = rng.choice(frame_colors_pool)
            placed = _try_attach_frame(canvas, path, line_col, fc, rng, require_contact="ortho", max_tries=FRAME_ATTACH_TRIES)
            if placed is not None:
                attached += 1
        if attached < per_seed_ortho[i]:
            return None

    diag_success = 0
    for i, (_, _, line_col, path) in enumerate(lines_info):
        for _ in range(per_seed_diag[i]):
            fc = rng.choice(frame_colors_pool)
            placed = _try_attach_frame(
                canvas,
                path,
                line_col,
                fc,
                rng,
                require_contact="diag_only",
                max_tries=DIAG_ATTACH_TRIES,
            )
            if placed is None:
                return None
            diag_success += 1
    if diag_success != num_diag_target:
        return None

    for _ in range(num_isolated_target):
        _try_attach_distractor_frame(canvas, rng.choice(frame_colors_pool), rng)

    grid = np.where(canvas == -1, bg, canvas).astype(int)

    vals, cnts = np.unique(grid, return_counts=True)
    if int(vals[np.argmax(cnts)]) != bg:
        return None

    for seed_col, _, _, _ in lines_info:
        if int((grid == seed_col).sum()) != 1:
            return None

    stats = _attachment_stats(grid)
    if stats["ortho"] < num_seeds or stats["diag_only"] < 1:
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
