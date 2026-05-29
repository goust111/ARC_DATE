# Combo variant of 8b7bacbf (index 23)
# combo_spec: D6/1+D9/3
# Name: 填充+擦除路径+开口框架

import random
from collections import Counter

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4, neighbors8

# Design analysis
# ===============
# fused_behavior:
#   concretization:
#     - Keep the base vocabulary: 1-2 unique border seed pixels, one thin 8-connected line per
#       seed, and several rounded-rectangle frame loops enclosing rectangular bg cavities.
#     - The frame is no longer fully present as a monochrome ring. Instead, 1-2 contiguous cells
#       on exactly one frame side are missing from the frame color, and those missing side cells are
#       occupied by the seed's line color. In the input, the line literally plugs the opening.
#     - Because the line occupies the missing side cells, the cavity is truly 4-closed in the input
#       even though the visible frame color itself is open. After solving, the cavity stays filled
#       but the line is erased, so the opening becomes visible again.
#   trace_example:
#     - Suppose a 2x3 cavity is surrounded by frame color 4, except the top side has a 2-cell gap.
#       Two path pixels of line color 8 lie exactly in those gap positions, making the boundary
#       around the cavity [4,4,8,8,4,...]. Seed color is 7.
#     - Base task logic would reject this if it insisted on a single-color frame boundary. D9-only
#       logic would also misread it because the missing side cells are not bg; they are path pixels.
#     - Fused logic should detect a rectangular bg cavity whose ideal ring is fully occupied by
#       non-bg cells, with exactly 1-2 contiguous side cells in line_color and all remaining ring
#       cells in one frame_color. Then fill the cavity with seed color and erase the whole line.
#     - Final output therefore shows: seed pixel still on the border, the path removed to bg, the
#       frame reopened at the former plug, and the cavity permanently filled in seed color.
# complexity_calibration:
#   - All attached fillable frames use the same repeated structural cue: rectangular cavity, one
#     side plug of length 1-2, and a line that disappears after the fill. This gives the solver
#     multiple consistent examples per grid instead of a one-off special case.
#   - The generator still keeps 1-2 seeds, several attached frames, optional distractor frames, and
#     a shared frame-color pool. This avoids the over-simplified regime where every non-bg object
#     obviously participates in the same transform.
#   - Gap lengths are limited to {1, 2}. That is visually noticeable, but still constrained enough
#     for humans to infer from a few examples.
# implicit_assumption_review:
#   - Original `len(bcols) == 1` is broken by the fusion, because a valid filled cavity now has a
#     mixed boundary: one frame color plus the line color used as a temporary plug.
#   - D9-only frame-component inference is also insufficient here, because the opening is not a bg
#     gap. The "missing" ring cells are occupied by line pixels, so the visible frame-color
#     component alone does not encode the closure; the full cavity boundary does.
#   - We therefore switch detection to rectangular bg components first, then reconstruct the ideal
#     ring around that bg rectangle and test the ring-color pattern against the seed's traced line.
#   - The transform-side assumption changes too: D6/1 erased the frame ring, but the fused spec
#     says erase the path after the fill. The solver must therefore preserve frame-color pixels and
#     only rewrite traced `line_color` pixels to bg.
# placement_legality:
#   - Attached frames are placed around existing line pixels. The chosen 1-2 side cells are already
#     occupied by a straight path segment; all other ring cells and cavity cells must be empty.
#   - No line pixel outside the chosen plug is allowed to touch the cavity interior, preventing
#     accidental second gaps or visually confusing intrusions.
#   - Frames keep a margin from the border, do not overlap, and do not 8-touch foreign non-line
#     colors, so the intended ring/cavity geometry remains readable.
# visual_observability:
#   - The distinctive cue is visible directly in I/O pairs: an open-looking frame becomes filled
#     even though the opening is visible in the output. The missing explanation is that the input
#     path had temporarily sealed that opening.
#   - Because the entire seed line is erased, the input/output difference is not subtle. Humans can
#     see both halves of the fused rule: fill occurs, and the path that made fill possible vanishes.
# visual_cleanliness_and_inference:
#   - Attached frames are rectangular and spaced apart; the line is thin and does not snake through
#     cavities. This keeps the grid legible despite combining two transformations.
#   - Colors keep their structural roles: seed color is the fill color, line color is the temporary
#     closure that disappears, and frame color is the persistent border that remains open afterward.
# new_solver_assumptions:
#   - Valid target cavities are rectangular 4-connected bg components not touching the grid border.
#   - The ideal ring around such a cavity is fully occupied by traced non-bg pixels in the input.
#   - Along that ideal ring, exactly 1-2 contiguous cells on one side are `line_color`, and every
#     other ring cell shares one non-bg frame color.
#   - Each seed that should transform has at least one such matched cavity; the generator enforces
#     that and the post-check verifies it before accepting a sample.


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


def _line_color_from_seed(g: np.ndarray, seed_rc: tuple[int, int], bg: int) -> int:
    h, w = g.shape
    sr, sc = seed_rc
    neigh_cols = []
    for nr, nc in neighbors4(sr, sc):
        if 0 <= nr < h and 0 <= nc < w:
            v = int(g[nr, nc])
            if v != bg:
                neigh_cols.append(v)
    assert neigh_cols, "seed pixel must have at least one non-bg neighbor"
    return int(Counter(neigh_cols).most_common(1)[0][0])


def _nonbg_component8_from(g: np.ndarray, start: tuple[int, int], bg: int) -> set[tuple[int, int]]:
    h, w = g.shape
    stack = [start]
    vis = {start}
    while stack:
        r, c = stack.pop()
        for nr, nc in neighbors8(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in vis and int(g[nr, nc]) != bg:
                vis.add((nr, nc))
                stack.append((nr, nc))
    return vis


def _frame_ring_coords(r0: int, c0: int, ch: int, cw: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
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


def _is_contiguous_side_segment(cells: set[tuple[int, int]], r0: int, c0: int, ch: int, cw: int) -> bool:
    if not (1 <= len(cells) <= 2):
        return False
    for side in _frame_sides(r0, c0, ch, cw):
        idxs = [i for i, cell in enumerate(side) if cell in cells]
        if len(idxs) != len(cells):
            continue
        if idxs[-1] - idxs[0] + 1 == len(idxs):
            return True
    return False


def _find_rectangular_cavities(
    g: np.ndarray,
    bg: int,
) -> list[tuple[set[tuple[int, int]], set[tuple[int, int]], int, int, int, int]]:
    h, w = g.shape
    cavities = []
    for pts in _components_mask(g == bg, conn8=False):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue
        region = set(pts)
        rows = [r for r, _ in pts]
        cols = [c for _, c in pts]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if len(region) != (r1 - r0 + 1) * (c1 - c0 + 1):
            continue
        ch = r1 - r0 + 1
        cw = c1 - c0 + 1
        outer_r0 = r0 - 1
        outer_c0 = c0 - 1
        if outer_r0 < 0 or outer_c0 < 0 or r1 + 1 >= h or c1 + 1 >= w:
            continue
        ring, _ = _frame_ring_coords(outer_r0, outer_c0, ch, cw)
        ring_set = set(ring)
        if any(int(g[r, c]) == bg for r, c in ring_set):
            continue
        cavities.append((region, ring_set, outer_r0, outer_c0, ch, cw))
    return cavities


def _match_plugged_frame(
    g: np.ndarray,
    traced: set[tuple[int, int]],
    line_col: int,
    cavity: set[tuple[int, int]],
    ring_set: set[tuple[int, int]],
    outer_r0: int,
    outer_c0: int,
    ch: int,
    cw: int,
    bg: int,
) -> bool:
    del cavity
    if not ring_set <= traced:
        return False

    gap_cells = {cell for cell in ring_set if int(g[cell]) == line_col}
    if not _is_contiguous_side_segment(gap_cells, outer_r0, outer_c0, ch, cw):
        return False

    frame_cells = ring_set - gap_cells
    frame_cols = {int(g[cell]) for cell in frame_cells}
    if len(frame_cols) != 1:
        return False
    frame_col = next(iter(frame_cols))
    if frame_col == bg or frame_col == line_col:
        return False
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill cavities that are temporarily sealed by a seed's path, then erase that path.

    Input structure:
    - The background is the most common color.
    - Each seed is a unique single pixel on the border, attached to one thin 8-connected line.
    - The line reaches several rounded-rectangle frames whose cavities are bg rectangles.
    - Some attached frames are visibly open in frame color, but the missing 1-2 side cells are
      occupied by the line color, so the cavity is actually closed in the input.

    Rule:
    - For each seed, trace its full non-bg 8-component.
    - Find rectangular bg cavities enclosed by an ideal frame ring whose boundary is fully occupied
      by traced pixels, with exactly 1-2 contiguous side cells in the seed's line color and all
      remaining ring cells in one frame color.
    - Flood-fill each matched cavity with the seed color.
    - After processing that seed's cavities, erase the traced line-color pixels to background.

    Key insight:
    - The line is not just attached to the frame; it temporarily closes the frame's opening. The
      fill must be inferred from that temporary closure, and the final output reveals the opening
      again because the path that plugged it disappears.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    out = g.copy()

    cavities = _find_rectangular_cavities(g, bg)

    for seed_rc in _seed_pixels(g, bg):
        seed_col = int(g[seed_rc])
        line_col = _line_color_from_seed(g, seed_rc, bg)
        traced = _nonbg_component8_from(g, seed_rc, bg)

        matched_any = False
        for cavity, ring_set, outer_r0, outer_c0, ch, cw in cavities:
            if not _match_plugged_frame(g, traced, line_col, cavity, ring_set, outer_r0, outer_c0, ch, cw, bg):
                continue
            matched_any = True
            for r, c in cavity:
                out[r, c] = seed_col

        if matched_any:
            for r, c in traced:
                if int(g[r, c]) == line_col:
                    out[r, c] = bg

    return out


# --------------- generator ---------------

# ========================== CFG ==========================
GRID_H_RANGE = (8, 30)
GRID_W_RANGE = (8, 30)
BG_COLORS = [0, 5]

SEED_COUNT_WEIGHTS = [1, 2, 2]
MIN_LINE_LEN = 5
LINE_LEN_RANGE = (10, 38)

FRAME_COLORS_MAX = 4
ATTACHED_FRAMES_PER_SEED = (1, 3)
DISTRACTOR_SOLO_MAX = 2

CAVITY_SIZE_WEIGHTS = [2, 2, 3, 3, 4]
DISTRACTOR_CAVITY_WEIGHTS = [2, 2, 3, 3, 4]
FRAME_PLACEMENT_PAD = 1
FRAME_ATTACH_TRIES = 100
DISTRACTOR_FRAME_TRIES = 60

ATTACHED_GAP_WEIGHTS = [(1, 5), (2, 3)]
DISTRACTOR_GAP_WEIGHTS = [(0, 3), (1, 2), (2, 1)]

DIAG_WEIGHT_ALIGNED = 5
DIAG_WEIGHT_OTHER = 1
ORTHO_WEIGHT_ALIGNED = 2
ORTHO_WEIGHT_OTHER = 1
BORDER_AVOIDANCE_PROB = 0.85
BIAS_FLIP_PROB = 0.3

GEN_RETRIES = 400
# =========================================================

DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
ORTHO_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _weighted_choice(weighted_values: list[tuple[int, int]], rng: random.Random) -> int:
    bag = []
    for value, weight in weighted_values:
        bag.extend([int(value)] * int(weight))
    return int(rng.choice(bag))


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
            if not _in_bounds(h, w, nr, nc) or int(grid[nr, nc]) != -1:
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


def _candidate_gap_segments(line_path: list[tuple[int, int]]) -> dict[int, list[tuple[tuple[int, int], ...]]]:
    segs: dict[int, list[tuple[tuple[int, int], ...]]] = {1: [], 2: []}
    seen = {1: set(), 2: set()}

    for i, cell in enumerate(line_path):
        if i > 0 or i + 1 < len(line_path):
            seg = (cell,)
            if seg not in seen[1]:
                seen[1].add(seg)
                segs[1].append(seg)

    for i in range(len(line_path) - 1):
        a = line_path[i]
        b = line_path[i + 1]
        if abs(a[0] - b[0]) + abs(a[1] - b[1]) != 1:
            continue
        seg = tuple(sorted((a, b)))
        if seg not in seen[2]:
            seen[2].add(seg)
            segs[2].append(seg)

    return segs


def _segment_sides(segment: tuple[tuple[int, int], ...]) -> list[str]:
    if len(segment) == 1:
        return ["T", "B", "L", "R"]
    a, b = segment
    if a[0] == b[0]:
        return ["T", "B"]
    if a[1] == b[1]:
        return ["L", "R"]
    return []


def _segment_origins(
    segment: tuple[tuple[int, int], ...],
    side: str,
    ch: int,
    cw: int,
) -> list[tuple[int, int]]:
    length = len(segment)
    origins = []
    if side in ("T", "B"):
        if length == 2 and segment[0][0] != segment[1][0]:
            return origins
        row = segment[0][0]
        cmin = min(c for _, c in segment)
        for start_idx in range(cw - length + 1):
            c0 = cmin - 1 - start_idx
            r0 = row if side == "T" else row - ch - 1
            origins.append((r0, c0))
        return origins

    if length == 2 and segment[0][1] != segment[1][1]:
        return origins
    col = segment[0][1]
    rmin = min(r for r, _ in segment)
    for start_idx in range(ch - length + 1):
        r0 = rmin - 1 - start_idx
        c0 = col if side == "L" else col - cw - 1
        origins.append((r0, c0))
    return origins


def _validate_attached_frame(
    canvas: np.ndarray,
    gap_segment: tuple[tuple[int, int], ...],
    line_col: int,
    ring: list[tuple[int, int]],
    cavity: list[tuple[int, int]],
    r0: int,
    c0: int,
    ch: int,
    cw: int,
) -> bool:
    h, w = canvas.shape
    gap_set = set(gap_segment)
    ring_set = set(ring)
    corners = {(r0, c0), (r0, c0 + cw + 1), (r0 + ch + 1, c0), (r0 + ch + 1, c0 + cw + 1)}

    for rr, cc in ring:
        if (rr, cc) in gap_set:
            if int(canvas[rr, cc]) != line_col:
                return False
            continue
        if int(canvas[rr, cc]) != -1:
            return False

    for rr, cc in cavity:
        if int(canvas[rr, cc]) != -1:
            return False
        for nr, nc in neighbors8(rr, cc):
            if not _in_bounds(h, w, nr, nc):
                continue
            if int(canvas[nr, nc]) == line_col and (nr, nc) not in gap_set:
                return False

    for rr, cc in corners:
        if int(canvas[rr, cc]) != -1:
            return False

    for rr, cc in ring:
        if (rr, cc) in gap_set:
            continue
        for nr, nc in neighbors8(rr, cc):
            if not _in_bounds(h, w, nr, nc) or (nr, nc) in ring_set or (nr, nc) in corners:
                continue
            v = int(canvas[nr, nc])
            if v != -1 and v != line_col:
                return False

    has_external_line = False
    for rr, cc in ring:
        for nr, nc in neighbors8(rr, cc):
            if not _in_bounds(h, w, nr, nc):
                continue
            if (nr, nc) in ring_set or (nr, nc) in corners or (nr, nc) in gap_set:
                continue
            if int(canvas[nr, nc]) == line_col:
                has_external_line = True
                break
        if has_external_line:
            break
    return has_external_line


def _try_attach_frame(
    canvas: np.ndarray,
    line_path: list[tuple[int, int]],
    line_col: int,
    frame_col: int,
    rng: random.Random,
    *,
    max_tries: int = FRAME_ATTACH_TRIES,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]] | None:
    h, w = canvas.shape
    gap_segments = _candidate_gap_segments(line_path)
    if not gap_segments[1] and not gap_segments[2]:
        return None

    for _ in range(max_tries):
        ch = int(rng.choice(CAVITY_SIZE_WEIGHTS))
        cw = int(rng.choice(CAVITY_SIZE_WEIGHTS))
        gap_len = _weighted_choice(ATTACHED_GAP_WEIGHTS, rng)
        segments = list(gap_segments.get(gap_len, []))
        rng.shuffle(segments)
        for segment in segments:
            sides = _segment_sides(segment)
            rng.shuffle(sides)
            for side in sides:
                origins = _segment_origins(segment, side, ch, cw)
                rng.shuffle(origins)
                for r0, c0 in origins:
                    if r0 < FRAME_PLACEMENT_PAD or c0 < FRAME_PLACEMENT_PAD:
                        continue
                    if r0 + ch + 2 > h - FRAME_PLACEMENT_PAD or c0 + cw + 2 > w - FRAME_PLACEMENT_PAD:
                        continue
                    ring, cavity = _frame_ring_coords(r0, c0, ch, cw)
                    if not _validate_attached_frame(canvas, segment, line_col, ring, cavity, r0, c0, ch, cw):
                        continue
                    gap_set = set(segment)
                    for rr, cc in ring:
                        if (rr, cc) not in gap_set:
                            canvas[rr, cc] = frame_col
                    return set(ring), set(cavity)
    return None


def _try_place_distractor_frame(
    canvas: np.ndarray,
    frame_col: int,
    rng: random.Random,
    *,
    max_tries: int = DISTRACTOR_FRAME_TRIES,
) -> bool:
    h, w = canvas.shape
    for _ in range(max_tries):
        ch = int(rng.choice(DISTRACTOR_CAVITY_WEIGHTS))
        cw = int(rng.choice(DISTRACTOR_CAVITY_WEIGHTS))
        fh, fw = ch + 2, cw + 2
        r0 = rng.randint(1, h - fh - 1)
        c0 = rng.randint(1, w - fw - 1)

        ok = True
        for rr in range(r0 - 1, r0 + fh + 1):
            for cc in range(c0 - 1, c0 + fw + 1):
                if _in_bounds(h, w, rr, cc) and int(canvas[rr, cc]) != -1:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue

        ring, _ = _frame_ring_coords(r0, c0, ch, cw)
        gap_len = _weighted_choice(DISTRACTOR_GAP_WEIGHTS, rng)
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

        gap_set = set(gap_cells)
        for rr, cc in ring:
            if (rr, cc) not in gap_set:
                canvas[rr, cc] = frame_col
        return True
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


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    bg = int(rng.choice(BG_COLORS))

    palette = [c for c in range(10) if c != bg]
    rng.shuffle(palette)

    num_seeds = int(rng.choice(SEED_COUNT_WEIGHTS))
    if num_seeds * 2 > len(palette):
        return None

    canvas = -np.ones((h, w), dtype=int)

    seed_colors = [int(palette.pop()) for _ in range(num_seeds)]
    line_colors = [int(palette.pop()) for _ in range(num_seeds)]

    all_corners = ["TL", "TR", "BL", "BR"]
    rng.shuffle(all_corners)
    corners_used = all_corners[:num_seeds]

    num_frame_cols = rng.randint(1, min(FRAME_COLORS_MAX, len(palette)))
    frame_colors_pool = [int(palette.pop()) for _ in range(num_frame_cols)]

    lines_info: list[tuple[int, tuple[int, int], int, list[tuple[int, int]]]] = []
    for i in range(num_seeds):
        seed_rc, first_rc = _corner_config(corners_used[i], h, w, rng)
        if int(canvas[seed_rc]) != -1 or int(canvas[first_rc]) != -1:
            return None
        canvas[seed_rc] = seed_colors[i]
        canvas[first_rc] = line_colors[i]
        target_len = rng.randint(LINE_LEN_RANGE[0], min(LINE_LEN_RANGE[1], h + w + 10))
        path = _grow_line(canvas, first_rc, line_colors[i], target_len, rng)
        if len(path) < MIN_LINE_LEN:
            return None
        lines_info.append((seed_colors[i], seed_rc, line_colors[i], path))

    attached_total = 0
    for _seed_col, _seed_rc, line_col, path in lines_info:
        target_frames = rng.randint(*ATTACHED_FRAMES_PER_SEED)
        attached = 0
        for _ in range(target_frames):
            frame_col = int(rng.choice(frame_colors_pool))
            if _try_attach_frame(canvas, path, line_col, frame_col, rng) is not None:
                attached += 1
        if attached == 0:
            return None
        attached_total += attached

    for _ in range(rng.randint(0, DISTRACTOR_SOLO_MAX)):
        _try_place_distractor_frame(canvas, int(rng.choice(frame_colors_pool)), rng)

    grid = np.where(canvas == -1, bg, canvas).astype(int)

    vals, cnts = np.unique(grid, return_counts=True)
    if int(vals[np.argmax(cnts)]) != bg:
        return None

    for seed_col, _seed_rc, _line_col, _path in lines_info:
        if int((grid == seed_col).sum()) != 1:
            return None

    seed_traces: list[set[tuple[int, int]]] = []
    for _seed_col, seed_rc, _line_col, _path in lines_info:
        traced = _nonbg_component8_from(grid, seed_rc, bg)
        if any(traced & other for other in seed_traces):
            return None
        seed_traces.append(traced)

    cavities = _find_rectangular_cavities(grid, bg)
    recognized_total = 0
    for (_seed_col, seed_rc, _line_col, _path), traced in zip(lines_info, seed_traces):
        line_col = _line_color_from_seed(grid, seed_rc, bg)
        recognized = 0
        for cavity, ring_set, outer_r0, outer_c0, ch, cw in cavities:
            if _match_plugged_frame(grid, traced, line_col, cavity, ring_set, outer_r0, outer_c0, ch, cw, bg):
                recognized += 1
        if recognized == 0:
            return None
        recognized_total += recognized

    if recognized_total != attached_total:
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None

    filled_cells = int(((grid == bg) & (out != bg)).sum())
    erased_cells = int(((grid != bg) & (out == bg)).sum())
    if filled_cells <= 0 or erased_cells <= 0:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted attempts")
