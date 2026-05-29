# Variant of 8b7bacbf
# Dimension: D9 — 腔体边界颜色约束 (obstacle)
# Change: 单色边界约束——只有边界完全由一种颜色组成的封闭区域才被视为有效腔体... → 部分开口的框架环
#
# Design analysis
# ===============
# variant_spec:
#   concretization:
#     - Keep the original seed/line/frame vocabulary:
#       border seed pixel -> 8-connected line -> monochrome rectangular frame ring.
#     - The old frame ring is the same "rounded rectangle" as before:
#       bbox border without the four bbox corners.
#     - The D9 change is local and visible: attached frames are now usually missing
#       1-2 contiguous pixels on exactly one side, so the cavity is no longer truly
#       4-closed in the input, but it should still be filled in the output.
#     - The output still flood-fills the whole inner rectangle with the seed color.
#   behavior_trace:
#     - Old logic on a broken frame:
#       a 1px gap connects the inner bg cells to the outer bg through 4-neighbors,
#       so `_components_mask(g == bg, conn8=False)` merges cavity and outside into
#       one background component; the cavity disappears and nothing is filled.
#     - New logic:
#       detect a near-complete monochrome rounded-rectangle frame component whose
#       bbox perimeter is missing at most 2 contiguous side pixels, and fill its
#       inner rectangle if that frame belongs to the traced seed network.
#     - Difference:
#       cavity recognition no longer depends on the cavity already being a closed
#       bg component; it depends on inferred closure of the frame itself.
#   complexity_calibration:
#     - To avoid an over-simplified "fill every open rectangle" dataset, the task
#       still includes: 1-2 seeds, multiple attached frames, optional distractors,
#       repeated frame colors, and unchanged seed-to-line attachment logic.
#     - Openings are restricted to 1-2 contiguous side pixels. This is enough to
#       make the visual change obvious, but avoids too many ambiguous gap shapes.
#   implicit_assumption_audit:
#     - Old `bg_comps = _components_mask(g == bg, conn8=False)` depended on
#       "valid cavity is already 4-closed". That fails immediately for this variant.
#     - Old `outside_idxs` also depended on the same closure invariant; with a gap,
#       the intended cavity is absorbed into the corner-connected outside component.
#     - Old `len(bcols) == 1` remains valid: the remaining frame pixels are still a
#       single border color. The changed part is closure, not border chromaticity.
#     - Old traced-component logic remains valid and is preserved.
#   placement_legality:
#     - The generator places an ideal closed frame first, then removes a short
#       contiguous side segment to create the opening.
#     - Gap cells are kept away from line-contact cells so the line still reads as
#       touching the outer frame, not entering the cavity through the gap.
#     - Frames keep a 1-cell margin from the canvas border and do not overlap or
#       8-touch foreign colors, so the solver's bbox-based frame inference stays
#       unambiguous.
#   observability:
#     - A human can see that filled cavities may sit inside visibly broken frames.
#       This is the central cue distinguishing the variant from the original task.
#     - The generator guarantees at least one attached broken frame per sample, so
#       the new rule is not hidden behind rare branches.
#   visual_cleanliness:
#     - Only one frame side may contain the opening, and it is only 1-2 pixels.
#       This keeps the frame recognizable as a nearly closed ring instead of noise.
#   color_inference:
#     - No invisible color ordering is used. Colors retain their original roles:
#       seed color = fill color, line color = attachment carrier, frame color =
#       obstacle/border color.
# new_solver_assumptions:
#   - Valid frames are rounded rectangles whose visible frame component equals the
#     ideal ring minus 0-2 contiguous cells on one side.
#   - The inner rectangle is background in the input.
#   - The frame belongs to the seed's traced non-bg 8-component and has at least
#     one pixel 8-adjacent to the seed's line color.
#   - The generator explicitly guarantees those assumptions.

import random
from collections import Counter

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4, neighbors8


def _components_mask(mask: np.ndarray, conn8: bool) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8 if conn8 else 4)]


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


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _frame_ring_coords(r0: int, c0: int, ch: int, cw: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Return (ring, cavity) for the original rounded-rectangle frame shape."""
    h, w = ch + 2, cw + 2
    ring = []
    for dr in range(h):
        for dc in range(w):
            on_bdr = dr == 0 or dc == 0 or dr == h - 1 or dc == w - 1
            corner = (dr in (0, h - 1)) and (dc in (0, w - 1))
            if on_bdr and not corner:
                ring.append((r0 + dr, c0 + dc))
    cavity = [(r0 + 1 + dr, c0 + 1 + dc) for dr in range(ch) for dc in range(cw)]
    return ring, cavity


def _frame_sides(r0: int, c0: int, ch: int, cw: int) -> list[list[tuple[int, int]]]:
    return [
        [(r0, c0 + dc) for dc in range(1, cw + 1)],
        [(r0 + ch + 1, c0 + dc) for dc in range(1, cw + 1)],
        [(r0 + dr, c0) for dr in range(1, ch + 1)],
        [(r0 + dr, c0 + cw + 1) for dr in range(1, ch + 1)],
    ]


def _is_single_short_opening(missing: set[tuple[int, int]], r0: int, c0: int, ch: int, cw: int) -> bool:
    if not missing:
        return True
    if len(missing) > 2:
        return False
    for side in _frame_sides(r0, c0, ch, cw):
        idxs = [i for i, p in enumerate(side) if p in missing]
        if len(idxs) != len(missing):
            continue
        if idxs and idxs[-1] - idxs[0] + 1 == len(idxs):
            return True
    return False


def _component_has_line_contact(g: np.ndarray, pts: set[tuple[int, int]], line_col: int) -> bool:
    h, w = g.shape
    for r, c in pts:
        for nr, nc in neighbors8(r, c):
            if _in_bounds(h, w, nr, nc) and int(g[nr, nc]) == line_col:
                return True
    return False


def _frame_candidate_from_component(
    g: np.ndarray,
    pts: list[tuple[int, int]],
    bg: int,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], int] | None:
    pts_set = set(pts)
    if len(pts_set) < 6:
        return None

    rows = [r for r, _ in pts_set]
    cols = [c for _, c in pts_set]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    ch = r1 - r0 - 1
    cw = c1 - c0 - 1
    if ch < 2 or cw < 2:
        return None

    ring, cavity = _frame_ring_coords(r0, c0, ch, cw)
    ring_set = set(ring)
    if not pts_set <= ring_set:
        return None

    missing = ring_set - pts_set
    if not _is_single_short_opening(missing, r0, c0, ch, cw):
        return None

    for r, c in missing:
        if int(g[r, c]) != bg:
            return None
    for r, c in cavity:
        if int(g[r, c]) != bg:
            return None

    return set(cavity), pts_set, len(missing)


def _all_frame_candidates(g: np.ndarray, bg: int) -> list[tuple[int, set[tuple[int, int]], set[tuple[int, int]], int]]:
    candidates = []
    for col in sorted(set(map(int, g.flatten())) - {int(bg)}):
        for comp in _components_mask(g == int(col), conn8=True):
            found = _frame_candidate_from_component(g, comp, bg)
            if found is None:
                continue
            cavity, frame_pts, missing_count = found
            candidates.append((int(col), cavity, frame_pts, int(missing_count)))
    return candidates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill seed-attached frame cavities, including near-closed frames with 1-2px openings.

    Relative to the original task, the only semantic change is cavity validity:
    a single-color rounded-rectangle frame may be missing a short contiguous side
    segment and is still treated as enclosing its inner rectangle.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = int(get_bg(g))
    out = g.copy()

    frame_candidates = _all_frame_candidates(g, bg)

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

        for _frame_col, cavity, frame_pts, _missing_count in frame_candidates:
            if not frame_pts <= traced:
                continue
            if not _component_has_line_contact(g, frame_pts, line_col):
                continue
            for r, c in cavity:
                if int(out[r, c]) == bg:
                    out[r, c] = seed_col

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
CAVITY_SIZE_WEIGHTS = [2, 2, 2, 3, 3, 4]
DISTRACTOR_CAVITY_WEIGHTS = [2, 2, 3, 3, 4]
FRAME_PLACEMENT_PAD = 1
FRAME_ATTACH_TRIES = 40
DISTRACTOR_FRAME_TRIES = 40
ATTACHED_GAP_WEIGHTS = [(1, 4), (2, 3)]
DISTRACTOR_GAP_WEIGHTS = [(0, 2), (1, 3), (2, 2)]

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


def _adjacent_to_color(canvas: np.ndarray, cell: tuple[int, int], color: int) -> bool:
    h, w = canvas.shape
    r, c = cell
    for nr, nc in neighbors8(r, c):
        if _in_bounds(h, w, nr, nc) and int(canvas[nr, nc]) == int(color):
            return True
    return False


def _weighted_choice(weighted_values: list[tuple[int, int]], rng: random.Random) -> int:
    bag = []
    for value, weight in weighted_values:
        bag.extend([int(value)] * int(weight))
    return int(rng.choice(bag))


def _grow_line(grid: np.ndarray, start: tuple[int, int], line_col: int, target_len: int, rng: random.Random) -> list[tuple[int, int]]:
    """Grow a meandering 8-connected 1px-wide path of line_col from start."""
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


def _pick_gap_cells(
    canvas: np.ndarray,
    ring: list[tuple[int, int]],
    r0: int,
    c0: int,
    ch: int,
    cw: int,
    line_col: int,
    weighted_lengths: list[tuple[int, int]],
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    contact_cells = {cell for cell in ring if _adjacent_to_color(canvas, cell, line_col)}
    if not contact_cells:
        return None

    lengths = []
    for gap_len, weight in weighted_lengths:
        lengths.extend([int(gap_len)] * int(weight))
    rng.shuffle(lengths)

    sides = _frame_sides(r0, c0, ch, cw)
    for gap_len in lengths:
        segments = []
        for side in sides:
            if len(side) < gap_len:
                continue
            for i in range(len(side) - gap_len + 1):
                seg = side[i : i + gap_len]
                if any(_adjacent_to_color(canvas, cell, line_col) for cell in seg):
                    continue
                if not any(cell not in seg for cell in contact_cells):
                    continue
                segments.append(seg)
        if segments:
            return list(rng.choice(segments))
    return None


def _try_attach_frame(
    canvas: np.ndarray,
    line_path: list[tuple[int, int]],
    line_col: int,
    frame_col: int,
    rng: random.Random,
    *,
    gap_weights: list[tuple[int, int]],
    max_tries: int = FRAME_ATTACH_TRIES,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]] | None:
    """Place an attached frame and optionally carve a short opening."""
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
            gap_cells = _pick_gap_cells(canvas, ring, r0, c0, ch, cw, line_col, gap_weights, rng)
            if gap_cells is None:
                continue
            kept_ring = [cell for cell in ring if cell not in set(gap_cells)]
            for rr, cc in kept_ring:
                canvas[rr, cc] = frame_col
            return kept_ring, cavity, gap_cells
    return None


def _try_attach_distractor_frame(
    canvas: np.ndarray,
    frame_col: int,
    rng: random.Random,
    *,
    gap_weights: list[tuple[int, int]],
    max_tries: int = DISTRACTOR_FRAME_TRIES,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]] | None:
    """Place a frame in empty space, optionally with a short opening."""
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

        ring, cavity = _frame_ring_coords(r0, c0, ch, cw)
        gap_len = _weighted_choice(gap_weights, rng)
        gap_cells: list[tuple[int, int]] = []
        if gap_len > 0:
            segments = []
            for side in _frame_sides(r0, c0, ch, cw):
                if len(side) < gap_len:
                    continue
                for i in range(len(side) - gap_len + 1):
                    segments.append(side[i : i + gap_len])
            if not segments:
                continue
            gap_cells = list(rng.choice(segments))

        kept_ring = [cell for cell in ring if cell not in set(gap_cells)]
        for rr, cc in kept_ring:
            canvas[rr, cc] = frame_col
        return kept_ring, cavity, gap_cells
    return None


def _try_place_distractor_network(canvas: np.ndarray, line_col: int, frame_colors_pool: list[int], rng: random.Random) -> bool:
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
            if _try_attach_frame(canvas, path, line_col, fc, rng, gap_weights=DISTRACTOR_GAP_WEIGHTS) is not None:
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

    attached_open_frames = 0
    for i, (_, _, line_col, path) in enumerate(lines_info):
        attached = 0
        for _ in range(per_seed[i]):
            fc = rng.choice(frame_colors_pool)
            placed = _try_attach_frame(canvas, path, line_col, fc, rng, gap_weights=ATTACHED_GAP_WEIGHTS)
            if placed is not None:
                attached += 1
                _ring, _cavity, gap_cells = placed
                if gap_cells:
                    attached_open_frames += 1
        if attached == 0:
            return None

    if attached_open_frames == 0:
        return None

    if rng.random() < DISTRACTOR_NETWORK_PROB:
        _try_place_distractor_network(canvas, rng.choice(line_colors), frame_colors_pool, rng)
    else:
        for _ in range(rng.randint(0, DISTRACTOR_SOLO_MAX)):
            _try_attach_distractor_frame(canvas, rng.choice(frame_colors_pool), rng, gap_weights=DISTRACTOR_GAP_WEIGHTS)

    grid = np.where(canvas == -1, bg, canvas).astype(int)

    vals, cnts = np.unique(grid, return_counts=True)
    if int(vals[np.argmax(cnts)]) != bg:
        return None

    for seed_col, _, _, _ in lines_info:
        if int((grid == seed_col).sum()) != 1:
            return None

    # Post-check the exact assumptions used by the new solver:
    # every seed must own at least one recognized frame candidate, and at least
    # one of those recognized frames across the sample must be genuinely open.
    frame_candidates = _all_frame_candidates(grid, bg)
    recognized_open = 0
    for _seed_col, seed_rc, _line_col, _path in lines_info:
        sr, sc = seed_rc
        neigh_cols = []
        for nr, nc in neighbors4(sr, sc):
            if _in_bounds(h, w, nr, nc):
                v = int(grid[nr, nc])
                if v != bg:
                    neigh_cols.append(v)
        if not neigh_cols:
            return None
        line_col = Counter(neigh_cols).most_common(1)[0][0]
        traced = _nonbg_component8_from(grid, seed_rc, bg)

        recognized = 0
        for _frame_col, _cavity, frame_pts, missing_count in frame_candidates:
            if not frame_pts <= traced:
                continue
            if not _component_has_line_contact(grid, frame_pts, line_col):
                continue
            recognized += 1
            if missing_count > 0:
                recognized_open += 1
        if recognized == 0:
            return None

    if recognized_open == 0:
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
