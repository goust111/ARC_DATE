# Variant of 8b7bacbf
# Dimension: D5 — 终点/停止条件 (dst)
# Change: 闭合框架腔体（cavity_fill）——封闭的单色框架环所围成的背景区域作为填充目标... → 多色边界腔体

import random
from collections import Counter

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4, neighbors8


# Design analysis for the D5 variant
# ----------------------------------
# spec_variant:
#   core_change:
#     old: only cavities with a single-color boundary are fill targets, and they are
#       filled by the unique border seed color of the connected line component.
#     new: only cavities whose boundary uses 2 distinct colors are fill targets.
#       The seed still chooses which 8-connected drawing to trace, but the fill color
#       is now copied from the boundary color directly touched by that seed's own line.
#   pixel_realization:
#     - Keep the original 1px meandering seed-to-line structure.
#     - Keep the original rounded-rect ring geometry (bbox boundary minus corners),
#       so D5 remains the only meaningful innovation.
#     - Replace fillable single-color rings with 2-color rings:
#       one pair of opposite sides uses color A, the other pair uses color B.
#     - The external line touches exactly one side-pair, so the touched boundary color
#       is unambiguous and visually visible.
#   behavior_trace_example:
#     - Old solver on a 2-color ring: cavity is detected, but bcols={A,B} so
#       `len(bcols)!=1` causes the cavity to be skipped.
#     - New solver on the same ring: the cavity is detected, boundary has 2 colors,
#       and among boundary pixels exactly one boundary color has neighbors that belong
#       to the traced seed line outside the ring. The cavity is filled with that
#       touched boundary color.
#     - Difference: the stop/target condition changes from "single-color enclosure"
#       to "multi-color enclosure with a unique touched side color".
#   complexity_calibration:
#     - Limit multicolor boundaries to exactly 2 colors. More colors would make the
#       mapping hard to infer from 2-4 ARC examples.
#     - Keep the side-based coloring repeated across examples, so the rule has support.
#     - Add occasional attached single-color loops as negative examples; they are
#       visibly attached to a traced line but remain unfilled, proving that the
#       multicolor condition matters.
#   implicit_assumption_audit_main:
#     - `_seed_pixels`: still assumes each seed color appears exactly once on border.
#       Generator preserves this.
#     - `line_col = Counter(neigh_cols).most_common(1)[0][0]`: assumes seed has at
#       least one non-bg orthogonal neighbor and that this neighbor belongs to the
#       intended line. Generator places exactly one orthogonal first-line pixel.
#     - Background cavity extraction via 4-connected bg components is still valid,
#       because ring geometry still encloses bg areas exactly as before.
#     - Old `if len(bcols) != 1: continue` is invalid under the variant. It is
#       replaced by `if len(bcols) < 2: continue`.
#     - Old "any boundary pixel near line_color" was harmless because all boundary
#       pixels shared one color. In the variant that would over-accept ambiguous
#       2-color rings, so the new solver isolates boundary pixels touched by external
#       traced line pixels that are not themselves part of the cavity boundary and
#       requires the touched boundary-color set to have size exactly 1.
#     - Filling can no longer blindly write `seed_col`; it must write the unique
#       touched boundary color.
#   placement_legality:
#     - Frames stay at least 1 cell away from the canvas edge, preserving outside-bg
#       corner detection.
#     - Ring cells and cavity cells must be empty before placement.
#     - Cavity cells cannot be 8-adjacent to the line, preserving a real enclosure.
#     - Only the designated contact side of a multicolor ring may be adjacent to
#       external line pixels; this guarantees an unambiguous fill color.
#   visual_observability:
#     - Humans can see: attached 2-color loops get filled with the color on the side
#       where the traced line touches; attached 1-color loops do not.
#     - This does not depend on hidden color ids or ordering.
#   visual_cleanliness:
#     - Reusing the original ring shape avoids clutter.
#     - Two side-colors per ring are easy to segment visually.
#     - Other objects remain spaced apart by the original no-overlap / no-foreign-
#       adjacency checks.
#   color_inference:
#     - The output color is not an arbitrary lookup table.
#     - It is a direct visible copy from the touched side color of the boundary,
#       with the seed selecting which connected drawing is active.
#   new_solver_assumptions_checked_against_generator:
#     - Every fillable ring is 2-colored.
#     - Its traced line touches only one side-color.
#     - Different seeded drawings remain disconnected in 8-connectivity.
#     - Single-color attached rings, when present, are distractors and are meant to
#       stay unchanged.


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


def _boundary_contact_colors(
    g: np.ndarray,
    boundary: set[tuple[int, int]],
    traced: set[tuple[int, int]],
    line_col: int,
) -> set[int]:
    """Boundary colors whose pixels are touched by external traced line pixels."""
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Variant rule:
    - Trace each unique border seed into its 8-connected non-bg drawing.
    - Consider only enclosed background cavities whose boundary uses 2 colors.
    - If the traced line touches exactly one boundary color of such a cavity, flood
      the cavity with that touched boundary color.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    out = g.copy()

    bg_comps = _components_mask(g == bg, conn8=False)
    regions = []
    for pts in bg_comps:
        region = set(pts)
        boundary = set()
        for r, c in region:
            for nr, nc in neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in region and int(g[nr, nc]) != bg:
                    boundary.add((nr, nc))
        regions.append((region, boundary, _touches_border(pts, h, w)))

    corners = {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}
    outside_idxs = set()
    for i, (reg, _, _) in enumerate(regions):
        if any(p in reg for p in corners if int(g[p]) == bg):
            outside_idxs.add(i)

    cavities = []
    for i, (reg, bnd, _) in enumerate(regions):
        if i in outside_idxs or not bnd:
            continue
        bcols = {int(g[p]) for p in bnd}
        cavities.append((reg, bnd, bcols))

    for sr, sc in _seed_pixels(g, bg):
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
            if len(bcols) < 2:
                continue
            if not (boundary & traced):
                continue
            contact_cols = _boundary_contact_colors(g, boundary, traced, line_col)
            if len(contact_cols) != 1:
                continue
            fill_col = next(iter(contact_cols))
            for r, c in cav:
                if int(out[r, c]) == bg:
                    out[r, c] = fill_col

    return out


# --------------- generator ---------------

# Canvas / grid
GRID_H_RANGE = (8, 30)
GRID_W_RANGE = (8, 30)
BG_COLORS = [0, 5]

# Seeds / lines
SEED_COUNT_WEIGHTS = [1, 2, 2]
MIN_LINE_LEN = 4
LINE_LEN_RANGE = (8, 35)

# Frames / loops
FRAME_COLORS_MAX = 4
TOTAL_LOOPS_MAX = 9
MULTI_CAVITY_SIZE_WEIGHTS = [3, 3, 4]
SINGLE_CAVITY_SIZE_WEIGHTS = [2, 2, 3]
FRAME_PLACEMENT_PAD = 1
FRAME_ATTACH_TRIES = 50
DISTRACTOR_FRAME_TRIES = 40
ATTACHED_SINGLE_DISTRACTOR_PROB = 0.55

# Line growth
DIAG_WEIGHT_ALIGNED = 5
DIAG_WEIGHT_OTHER = 1
ORTHO_WEIGHT_ALIGNED = 2
ORTHO_WEIGHT_OTHER = 1
BORDER_AVOIDANCE_PROB = 0.85
BIAS_FLIP_PROB = 0.3

# Distractor network
DISTRACTOR_NETWORK_PROB = 0.55
DISTRACTOR_NETWORK_TRIES = 50
DISTRACTOR_LINE_LEN = (5, 15)
DISTRACTOR_NETWORK_FRAMES = (1, 2)
DISTRACTOR_SOLO_MAX = 2

# Generation
GEN_RETRIES = 400

DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
ORTHO_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDES = ["top", "right", "bottom", "left"]
OPPOSITE = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _frame_ring_layout(r0: int, c0: int, ch: int, cw: int):
    sides = {
        "top": [(r0, c0 + dc) for dc in range(1, cw + 1)],
        "bottom": [(r0 + ch + 1, c0 + dc) for dc in range(1, cw + 1)],
        "left": [(r0 + dr, c0) for dr in range(1, ch + 1)],
        "right": [(r0 + dr, c0 + cw + 1) for dr in range(1, ch + 1)],
    }
    ring = [pt for side in SIDES for pt in sides[side]]
    cavity = [(r0 + 1 + dr, c0 + 1 + dc) for dr in range(ch) for dc in range(cw)]
    return sides, ring, cavity


def _side_slots(ch: int, cw: int, side: str, strict_middle: bool) -> list[int]:
    if side in ("top", "bottom"):
        if strict_middle:
            return list(range(2, cw))
        return list(range(1, cw + 1))
    if strict_middle:
        return list(range(2, ch))
    return list(range(1, ch + 1))


def _placement_from_touch(lp: tuple[int, int], ch: int, cw: int, side: str, slot: int) -> tuple[int, int]:
    if side == "top":
        return lp[0] + 1, lp[1] - slot
    if side == "bottom":
        return lp[0] - (ch + 2), lp[1] - slot
    if side == "left":
        return lp[0] - slot, lp[1] + 1
    if side == "right":
        return lp[0] - slot, lp[1] - (cw + 2)
    raise ValueError(side)


def _paint_frame(canvas: np.ndarray, sides: dict[str, list[tuple[int, int]]], side_colors: dict[str, int]) -> None:
    for side, pts in sides.items():
        col = side_colors[side]
        for rr, cc in pts:
            canvas[rr, cc] = col


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


def _grow_line(grid: np.ndarray, start: tuple[int, int], line_col: int, target_len: int, rng: random.Random):
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


def _try_attach_pattern_frame(
    canvas: np.ndarray,
    line_path: list[tuple[int, int]],
    line_col: int,
    side_colors: dict[str, int],
    contact_side: str,
    cavity_weights: list[int],
    strict_middle: bool,
    rng: random.Random,
    max_tries: int = FRAME_ATTACH_TRIES,
):
    h, w = canvas.shape
    pad = FRAME_PLACEMENT_PAD
    for _ in range(max_tries):
        ch = rng.choice(cavity_weights)
        cw = rng.choice(cavity_weights)
        slots = _side_slots(ch, cw, contact_side, strict_middle)
        if not slots:
            continue
        lp = rng.choice(line_path)
        slot = rng.choice(slots)
        r0, c0 = _placement_from_touch(lp, ch, cw, contact_side, slot)
        fh, fw = ch + 2, cw + 2
        if r0 < pad or c0 < pad or r0 + fh > h - pad or c0 + fw > w - pad:
            continue
        sides, ring, cavity = _frame_ring_layout(r0, c0, ch, cw)
        if not _validate_frame_placement(canvas, ring, cavity, line_col, set(sides[contact_side])):
            continue
        _paint_frame(canvas, sides, side_colors)
        return {
            "ring": ring,
            "cavity": cavity,
            "contact_color": int(side_colors[contact_side]),
            "boundary_colors": {int(side_colors[s]) for s in SIDES},
        }
    return None


def _try_attach_bicolor_frame(
    canvas: np.ndarray,
    line_path: list[tuple[int, int]],
    line_col: int,
    frame_colors_pool: list[int],
    rng: random.Random,
):
    contact_side = rng.choice(SIDES)
    primary, secondary = _choose_two_distinct(frame_colors_pool, rng)
    side_colors = {}
    for side in SIDES:
        side_colors[side] = primary if side in (contact_side, OPPOSITE[contact_side]) else secondary
    return _try_attach_pattern_frame(
        canvas,
        line_path,
        line_col,
        side_colors,
        contact_side,
        MULTI_CAVITY_SIZE_WEIGHTS,
        True,
        rng,
    )


def _try_attach_single_frame(
    canvas: np.ndarray,
    line_path: list[tuple[int, int]],
    line_col: int,
    frame_col: int,
    rng: random.Random,
):
    contact_side = rng.choice(SIDES)
    side_colors = {side: frame_col for side in SIDES}
    return _try_attach_pattern_frame(
        canvas,
        line_path,
        line_col,
        side_colors,
        contact_side,
        SINGLE_CAVITY_SIZE_WEIGHTS,
        False,
        rng,
    )


def _try_place_solo_frame(canvas: np.ndarray, side_colors: dict[str, int], cavity_weights: list[int], rng: random.Random):
    h, w = canvas.shape
    for _ in range(DISTRACTOR_FRAME_TRIES):
        ch = rng.choice(cavity_weights)
        cw = rng.choice(cavity_weights)
        fh, fw = ch + 2, cw + 2
        if h - fh - 2 < 1 or w - fw - 2 < 1:
            continue
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
        sides, ring, cavity = _frame_ring_layout(r0, c0, ch, cw)
        if not _validate_frame_placement(canvas, ring, cavity, None, None):
            continue
        _paint_frame(canvas, sides, side_colors)
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
            if len(frame_colors_pool) >= 2 and rng.random() < 0.7:
                if _try_attach_bicolor_frame(canvas, path, line_col, frame_colors_pool, rng) is not None:
                    attached += 1
            else:
                fc = rng.choice(frame_colors_pool)
                if _try_attach_single_frame(canvas, path, line_col, fc, rng) is not None:
                    attached += 1
        if attached > 0:
            return True
        for pr, pc in path:
            canvas[pr, pc] = -1
    return False


def _corner_config(corner_kind: str, h: int, w: int, rng: random.Random):
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
        lines_info.append((seed_colors[i], line_colors[i], path))

    total_loops_target = rng.randint(num_seeds, TOTAL_LOOPS_MAX)
    num_solo_distractors = rng.randint(0, min(DISTRACTOR_SOLO_MAX, max(0, total_loops_target - num_seeds)))
    attached_fillable_total = total_loops_target - num_solo_distractors
    per_seed_fillable = _distribute_loop_budget(num_seeds, attached_fillable_total, rng)

    any_single_attached = False
    for i, (_, line_col, path) in enumerate(lines_info):
        attached = 0
        for _ in range(per_seed_fillable[i]):
            if _try_attach_bicolor_frame(canvas, path, line_col, frame_colors_pool, rng) is not None:
                attached += 1
        if attached == 0:
            return None
        if rng.random() < ATTACHED_SINGLE_DISTRACTOR_PROB:
            fc = rng.choice(frame_colors_pool)
            if _try_attach_single_frame(canvas, path, line_col, fc, rng) is not None:
                any_single_attached = True

    if rng.random() < DISTRACTOR_NETWORK_PROB:
        _try_place_distractor_network(canvas, rng.choice(line_colors), frame_colors_pool, rng)
    else:
        for _ in range(num_solo_distractors):
            if rng.random() < 0.5:
                fc = rng.choice(frame_colors_pool)
                _try_place_solo_frame(canvas, {side: fc for side in SIDES}, SINGLE_CAVITY_SIZE_WEIGHTS, rng)
            else:
                a, b = _choose_two_distinct(frame_colors_pool, rng)
                contact_side = rng.choice(SIDES)
                side_colors = {side: (a if side in (contact_side, OPPOSITE[contact_side]) else b) for side in SIDES}
                _try_place_solo_frame(canvas, side_colors, MULTI_CAVITY_SIZE_WEIGHTS, rng)

    grid = np.where(canvas == -1, bg, canvas).astype(int)

    vals, cnts = np.unique(grid, return_counts=True)
    if int(vals[np.argmax(cnts)]) != bg:
        return None

    for seed_col, _, _ in lines_info:
        if int((grid == seed_col).sum()) != 1:
            return None

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None

    # Reject accidental degeneration back to the original task:
    # at least one filled cell must use a non-seed color.
    fill_colors = {int(v) for v in np.unique(out[grid == bg]) if v != bg}
    if fill_colors and fill_colors <= set(seed_colors):
        return None

    # Encourage but do not require attached single-color distractors across attempts.
    if num_seeds > 1 and not any_single_attached and rng.random() < 0.35:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError("generate_input: exhausted attempts")
