# Combo variant of 8b7bacbf (index 14)
# combo_spec: D12/0+D13/1
# Name: 单种子多线路+多线路共色

import random

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4, neighbors8


# ---------------------------------------------------------------------------
# Design analysis for the combo variant
# ---------------------------------------------------------------------------
#
# combo_variant_spec:
#   input:
#     bg:
#       most common color, still drawn from {0, 5}
#     seed:
#       count: exactly 1
#       shape: one unique-color border pixel
#       placement: non-corner edge cell so it can expose two 4-neighbor line starts
#     active_lines:
#       count: exactly 2 visible branches in every generated sample
#       color: both branches use the same shared line color
#       relation: each branch starts from a different 4-neighbor of the single seed
#       geometry: 8-connected meandering 1px paths; branches stay separated except
#         for meeting at the seed itself
#     frame_loops:
#       count: 1-3 per branch, at least 1 on each branch
#       attachment: each loop is 8-adjacent to exactly one branch path
#       boundary_color: independent from the fill color; ring colors may repeat
#     distractors:
#       optional isolated frame loops not connected to the seeded network
#   output:
#     every cavity attached to either seeded branch is flood-filled with the
#     shared line color
#     the seed color remains a single pixel and never becomes the fill color
#
# behavior_trace:
#   Original task:
#     seed S touches one line color L. The solver traces the seeded non-bg
#     component and fills every attached cavity with S.
#   D12/0 alone:
#     one seed touches multiple branches; each branch can map to a different
#     line color, so the fill source moves from seed color to branch color.
#   This combo:
#     there is still one seed with multiple branches, but the branches now share
#     one line color L. The solver no longer needs branch-by-branch color lookup;
#     it must gather the whole seeded component and fill every reachable cavity
#     with the common line color L rather than with S.
#   Concrete miniature:
#     a top-edge seed of color 3 touches two starts, both color 6. The left and
#     right branches each reach a separate ring. The output fills both cavities
#     with 6, while the seed stays 3. Using seed color 3 would collapse the combo
#     back toward the original task and remove the D12 color-source change.
#
# complexity_calibration:
#   - Branch count is fixed at 2 so the "single seed, multiple same-color starts"
#     is always visible from a few examples.
#   - All filled cavities use the same line color, which keeps color inference
#     simple enough for ARC while still making the seed-vs-line distinction clear.
#   - Ring colors can repeat across branches, so the observable rule is driven by
#     connectivity and the shared line color rather than by a frame-color table.
#
# old_main_assumption_audit:
#   - `_seed_pixels(...)` remains valid, but the combo expects exactly one seed.
#   - "one seed -> one line start" is no longer valid; the seed must have multiple
#     non-bg 4-neighbors and they must all share one line color.
#   - Tracing the whole seeded non-bg component is still correct and is now the
#     intended abstraction, because both branches belong to one seeded network.
#   - `out[cavity] = seed_col` is invalid in this combo. Keeping that line would
#     make the generated samples behave like the original solver, so the fill must
#     come from the shared line color instead.
#
# placement_legality:
#   - The generator places the seed on a non-corner edge position with exactly
#     two reserved start cells.
#   - Branch growth rejects any occupied 8-neighbor other than the seed/current
#     predecessor, so the two branches do not merge away from the seed.
#   - Frame placement requires empty ring+cavity cells, forbids any occupied
#     neighbor around the cavity, and allows ring contact only with the hosting
#     branch path.
#
# visual_observability:
#   - Every sample shows one unique seed color that does not spread in the output,
#     plus one line color that visibly expands into multiple cavities.
#   - The two same-color starts on opposite sides of the seed make the multi-line
#     nature visible even though both branches share the same color.
#   - Optional isolated loops provide negative examples without cluttering the
#     main seeded network.
#
# cleanliness_and_inference:
#   - The generator keeps cavities separated from unrelated structures by an
#     empty halo, avoiding ambiguous "almost touching" shapes.
#   - Color usage is intentionally small: background, one seed color, one shared
#     line/fill color, and a small frame-color pool.
#
# new_main_assumption_audit:
#   - `main()` now assumes there is exactly one seed, that all of its non-bg
#     4-neighbors have the same color, and that there are at least two such
#     neighbors. `generate_input()` constructs exactly that.
#   - `main()` also assumes fillable cavities are closed 4-connected bg regions
#     whose boundary has a single ring color and touches the seeded component via
#     at least one shared-line pixel. Frame placement guarantees this geometry.


def _components_mask(mask: np.ndarray, conn8: bool) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8 if conn8 else 4)]


def _touches_border(pts: set[tuple[int, int]], h: int, w: int) -> bool:
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
    stack = [start]
    vis = {start}
    while stack:
        r, c = stack.pop()
        for nr, nc in neighbors8(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in vis and int(g[nr, nc]) != bg:
                vis.add((nr, nc))
                stack.append((nr, nc))
    return vis


def _seed_line_info(g: np.ndarray, seed_rc: tuple[int, int], bg: int) -> tuple[int, list[tuple[int, int]]] | None:
    h, w = g.shape
    starts: list[tuple[int, int]] = []
    colors = set()
    for nr, nc in neighbors4(*seed_rc):
        if 0 <= nr < h and 0 <= nc < w:
            v = int(g[nr, nc])
            if v != bg:
                starts.append((nr, nc))
                colors.add(v)
    if len(starts) < 2 or len(colors) != 1:
        return None
    return int(next(iter(colors))), starts


def _cavities(g: np.ndarray, bg: int) -> list[tuple[set[tuple[int, int]], set[tuple[int, int]], set[int]]]:
    h, w = g.shape
    bg_comps = _components_mask(g == bg, conn8=False)
    cavities: list[tuple[set[tuple[int, int]], set[tuple[int, int]], set[int]]] = []
    for pts in bg_comps:
        region = set(pts)
        if _touches_border(region, h, w):
            continue
        boundary = set()
        for r, c in region:
            for nr, nc in neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in region and int(g[nr, nc]) != bg:
                    boundary.add((nr, nc))
        if not boundary:
            continue
        boundary_colors = {int(g[p]) for p in boundary}
        cavities.append((region, boundary, boundary_colors))
    return cavities


def _boundary_touches_line(boundary: set[tuple[int, int]], line_component: set[tuple[int, int]], h: int, w: int) -> bool:
    for r, c in boundary:
        for nr, nc in neighbors8(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) in line_component:
                return True
    return False


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill every frame cavity reachable from a single border seed that fans into multiple same-color branches.

    Input structure:
    - The background is the most frequent color.
    - Exactly one unique-color seed sits on the border.
    - That seed touches multiple 4-neighbor line starts, and all of those branches share one line color.
    - Closed single-color frame rings are attached alongside the branches; some other isolated rings may exist as distractors.

    Rule:
    - Start from the seed and trace the entire 8-connected non-background component reachable from it.
    - Among that traced component, identify the shared line color adjacent to the seed.
    - For each enclosed background cavity whose boundary belongs to the traced component and is 8-adjacent to at least one traced line pixel, flood-fill the cavity with the shared line color.

    Key insight:
    - The seed is only the entry point. It does not supply the fill color.
    - All seeded branches use the same line color, so the solver should collect the whole seeded network and paint every reachable cavity with that common branch color.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    out = g.copy()

    seeds = _seed_pixels(g, bg)
    if len(seeds) != 1:
        return out
    seed_rc = seeds[0]

    seed_line = _seed_line_info(g, seed_rc, bg)
    if seed_line is None:
        return out
    line_col, _ = seed_line

    traced = _nonbg_component8_from(g, seed_rc, bg)
    traced_line = {p for p in traced if int(g[p]) == line_col}
    if not traced_line:
        return out

    for cavity, boundary, boundary_colors in _cavities(g, bg):
        if len(boundary_colors) != 1:
            continue
        if not (boundary & traced):
            continue
        if not _boundary_touches_line(boundary, traced_line, h, w):
            continue
        for r, c in cavity:
            if int(out[r, c]) == bg:
                out[r, c] = line_col

    return out


# ---------------------------------------------------------------------------
# Generator config
# ---------------------------------------------------------------------------

GRID_H_RANGE = (10, 30)
GRID_W_RANGE = (10, 30)
BG_COLORS = [0, 5]

ACTIVE_BRANCH_COUNT = 2
MIN_LINE_LEN = 5
LINE_LEN_RANGE = (8, 32)

FRAME_COLORS_MAX = 4
ATTACHED_LOOPS_PER_BRANCH = (1, 3)
TOTAL_LOOPS_MAX = 8
CAVITY_SIZE_WEIGHTS = [2, 2, 2, 3]
DISTRACTOR_CAVITY_WEIGHTS = [2, 2, 3]
FRAME_PLACEMENT_PAD = 1
FRAME_ATTACH_TRIES = 45
DISTRACTOR_FRAME_TRIES = 40

DIAG_WEIGHT_ALIGNED = 5
DIAG_WEIGHT_OTHER = 1
ORTHO_WEIGHT_ALIGNED = 2
ORTHO_WEIGHT_OTHER = 1
BORDER_AVOIDANCE_PROB = 0.85
BIAS_FLIP_PROB = 0.3

DISTRACTOR_SOLO_MAX = 2
GEN_RETRIES = 400

DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
ORTHO_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _start_cell_clear(canvas: np.ndarray, first_rc: tuple[int, int], allowed_neighbor: tuple[int, int]) -> bool:
    h, w = canvas.shape
    r, c = first_rc
    if canvas[r, c] != -1:
        return False
    for nr, nc in neighbors8(r, c):
        if not _in_bounds(h, w, nr, nc):
            continue
        if canvas[nr, nc] != -1 and (nr, nc) != allowed_neighbor:
            return False
    return True


def _frame_ring_coords(r0: int, c0: int, ch: int, cw: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    fh, fw = ch + 2, cw + 2
    ring = []
    for dr in range(fh):
        for dc in range(fw):
            on_border = dr == 0 or dc == 0 or dr == fh - 1 or dc == fw - 1
            corner = (dr in (0, fh - 1)) and (dc in (0, fw - 1))
            if on_border and not corner:
                ring.append((r0 + dr, c0 + dc))
    cavity = [(r0 + 1 + dr, c0 + 1 + dc) for dr in range(ch) for dc in range(cw)]
    return ring, cavity


def _validate_frame_placement(
    canvas: np.ndarray,
    ring: list[tuple[int, int]],
    cavity: list[tuple[int, int]],
    line_path: list[tuple[int, int]],
    h: int,
    w: int,
) -> bool:
    ring_set = set(ring)
    line_path_set = set(line_path)

    for rr, cc in ring:
        if canvas[rr, cc] != -1:
            return False
    for rr, cc in cavity:
        if canvas[rr, cc] != -1:
            return False

    for cr, cc in cavity:
        for nr, nc in neighbors8(cr, cc):
            if _in_bounds(h, w, nr, nc) and canvas[nr, nc] != -1:
                return False

    touches_line = False
    for rr, cc in ring:
        for nr, nc in neighbors8(rr, cc):
            if not _in_bounds(h, w, nr, nc) or (nr, nc) in ring_set:
                continue
            if canvas[nr, nc] == -1:
                continue
            if (nr, nc) not in line_path_set:
                return False
            touches_line = True
    return touches_line


def _weighted_dirs(dr_bias: int, dc_bias: int) -> list[tuple[int, int]]:
    dirs = []
    for ddr, ddc in DIAG_DIRS:
        aligned = ddr == dr_bias or ddc == dc_bias
        dirs.extend([(ddr, ddc)] * (DIAG_WEIGHT_ALIGNED if aligned else DIAG_WEIGHT_OTHER))
    for ddr, ddc in ORTHO_DIRS:
        aligned = (ddr == dr_bias and ddc == 0) or (ddc == dc_bias and ddr == 0)
        dirs.extend([(ddr, ddc)] * (ORTHO_WEIGHT_ALIGNED if aligned else ORTHO_WEIGHT_OTHER))
    return dirs


def _grow_branch(
    grid: np.ndarray,
    start: tuple[int, int],
    line_col: int,
    target_len: int,
    rng: random.Random,
    dr_bias: int,
    dc_bias: int,
    seed_rc: tuple[int, int],
) -> list[tuple[int, int]]:
    h, w = grid.shape
    path = [start]
    path_set = {start}
    cur = start
    while len(path) < target_len:
        dirs = _weighted_dirs(dr_bias, dc_bias)
        rng.shuffle(dirs)
        placed = False
        for ddr, ddc in dirs:
            nr, nc = cur[0] + ddr, cur[1] + ddc
            if not _in_bounds(h, w, nr, nc) or grid[nr, nc] != -1:
                continue
            if (nr == 0 or nc == 0 or nr == h - 1 or nc == w - 1) and rng.random() < BORDER_AVOIDANCE_PROB:
                continue

            blocked = False
            for ar, ac in neighbors8(nr, nc):
                if not _in_bounds(h, w, ar, ac):
                    continue
                if (ar, ac) == cur or (ar, ac) == seed_rc:
                    continue
                if grid[ar, ac] != -1:
                    blocked = True
                    break
            if blocked:
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
            if not _validate_frame_placement(canvas, ring, cavity, line_path, h, w):
                continue
            for rr, cc in ring:
                canvas[rr, cc] = frame_col
            return ring, cavity
    return None


def _try_attach_distractor_frame(canvas: np.ndarray, frame_col: int, rng: random.Random, max_tries: int = DISTRACTOR_FRAME_TRIES) -> bool:
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
    return False


def _edge_seed_config(
    h: int, w: int, rng: random.Random
) -> tuple[tuple[int, int], list[tuple[int, int]], list[tuple[int, int]]]:
    edges = []
    c = rng.randint(3, w - 4)
    edges.append(((0, c), [(0, c - 1), (0, c + 1)], [(1, -1), (1, 1)]))
    edges.append(((h - 1, c), [(h - 1, c - 1), (h - 1, c + 1)], [(-1, -1), (-1, 1)]))
    r = rng.randint(3, h - 4)
    edges.append(((r, 0), [(r - 1, 0), (r + 1, 0)], [(-1, 1), (1, 1)]))
    edges.append(((r, w - 1), [(r - 1, w - 1), (r + 1, w - 1)], [(-1, -1), (1, -1)]))
    return rng.choice(edges)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    bg = rng.choice(BG_COLORS)
    palette = [c for c in range(10) if c != bg]
    rng.shuffle(palette)

    if len(palette) < 4:
        return None

    canvas = -np.ones((h, w), dtype=int)
    seed_col = palette.pop()
    shared_line_col = palette.pop()

    num_frame_cols = rng.randint(1, min(FRAME_COLORS_MAX, len(palette)))
    frame_colors_pool = [palette.pop() for _ in range(num_frame_cols)]
    if not frame_colors_pool:
        return None

    seed_rc, starts, biases = _edge_seed_config(h, w, rng)
    if canvas[seed_rc] != -1 or any(not _start_cell_clear(canvas, start, seed_rc) for start in starts):
        return None
    canvas[seed_rc] = seed_col

    branch_paths: list[list[tuple[int, int]]] = []
    for start, (dr_bias, dc_bias) in zip(starts, biases):
        if not _start_cell_clear(canvas, start, seed_rc):
            return None
        canvas[start] = shared_line_col
        target_len = rng.randint(LINE_LEN_RANGE[0], min(LINE_LEN_RANGE[1], h + w))
        path = _grow_branch(canvas, start, shared_line_col, target_len, rng, dr_bias, dc_bias, seed_rc)
        if len(path) < MIN_LINE_LEN:
            return None
        branch_paths.append(path)

    total_loops = 0
    for path in branch_paths:
        attached_target = rng.randint(*ATTACHED_LOOPS_PER_BRANCH)
        attached = 0
        for _ in range(attached_target):
            frame_col = rng.choice(frame_colors_pool)
            if _try_attach_frame(canvas, path, frame_col, rng) is not None:
                attached += 1
        if attached == 0:
            return None
        total_loops += attached
    if total_loops > TOTAL_LOOPS_MAX:
        return None

    for _ in range(rng.randint(0, DISTRACTOR_SOLO_MAX)):
        _try_attach_distractor_frame(canvas, rng.choice(frame_colors_pool), rng)

    grid = np.where(canvas == -1, bg, canvas).astype(int)

    vals, cnts = np.unique(grid, return_counts=True)
    if int(vals[np.argmax(cnts)]) != bg:
        return None
    if int((grid == seed_col).sum()) != 1:
        return None

    seeds = _seed_pixels(grid, bg)
    if len(seeds) != 1 or seeds[0] != seed_rc:
        return None
    seed_line = _seed_line_info(grid, seed_rc, bg)
    if seed_line is None:
        return None
    line_col, line_starts = seed_line
    if line_col != shared_line_col or len(line_starts) != ACTIVE_BRANCH_COUNT:
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None

    if int((out == seed_col).sum()) != 1:
        return None
    if int((out == shared_line_col).sum()) <= int((grid == shared_line_col).sum()):
        return None
    if np.any((out != grid) & (out != shared_line_col)):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted attempts")
