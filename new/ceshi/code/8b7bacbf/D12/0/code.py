# Variant of 8b7bacbf
# Dimension: D12 — 种子数量与基数关系 (meta)
# Change: 1-2个种子，每个种子独立处理，多个种子之间不交互（多:多的独立映射）... → 单种子多线路

import random
from collections import deque

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4, neighbors8


# ---------------------------------------------------------------------------
# Design analysis for the D12 variant
# ---------------------------------------------------------------------------
#
# spec_variant:
#   input:
#     bg: most common color, still 0 or 5
#     seed:
#       count: exactly 1
#       shape: one unique-color pixel on the border
#       position: non-corner border cell so it can expose two distinct line starts
#     active_lines:
#       count: exactly 2
#       relation: both lines 4-touch the single seed on two different sides
#       colors: distinct from each other and from the seed color
#       geometry: 8-connected 1px meandering paths, separated from each other
#     frame_loops:
#       count: 1-3 per line, at least 1 for each line
#       attachment: each loop is 8-adjacent to exactly one active line
#       boundary_color: independent of the eventual fill color; frame colors may repeat
#     distractors:
#       optional solo loops and optional disconnected line+loop network
#       these never touch either active line
#   output:
#     each cavity attached to line A is filled with line-color A
#     each cavity attached to line B is filled with line-color B
#     the seed color is never used for filling
#
# Behavior trace on the key changed step:
#   Original:
#     seed S has one neighboring line color L. The solver traces the whole non-bg
#     component reachable from S and fills every attached cavity with S.
#   Variant:
#     seed S has two neighboring line colors L1 and L2. The solver must separate
#     the two same-seed branches, find which cavity boundary is adjacent to which
#     branch, and fill with L1 or L2 respectively.
#   Concrete miniature:
#     S on the top edge touches a left branch in color 2 and a right branch in
#     color 4. A red frame and a green frame can both use the same ring color 7.
#     The left cavity becomes 2, the right cavity becomes 4. Using seed color S
#     would collapse the two outputs and lose the new 1:many mapping.
#
# Complexity calibration:
#   - The variant always has two active line colors, so the 1:many behavior is
#     visible in every generated sample.
#   - Frame colors stay separate from fill colors; repeated frame colors ensure
#     the solver cannot rely on frame-color -> fill-color lookup.
#   - The number of active branches is fixed at 2 to keep the rule legible from
#     2-4 examples and to avoid edge-placement ambiguity from 3-way branching.
#
# Old-main assumption audit:
#   - `_seed_pixels(...)`: still valid, but the variant expects exactly one seed.
#   - `line_col = most common non-bg seed neighbor`: invalid now because the seed
#     intentionally touches multiple different line colors.
#   - `traced = _nonbg_component8_from(seed)`: too coarse now because both lines
#     and all attached frames are in one large non-bg component; we must instead
#     isolate per-color line components grown from the seed's neighbors.
#   - `out[cavity] = seed_col`: invalid, because the fill color must now come from
#     the matched branch line color, not from the seed.
#
# Placement legality and generator guarantees:
#   - The seed is placed on a non-corner border cell and starts exactly two lines.
#   - Line growth rejects candidates that would become 8-adjacent to another line
#     (except the shared seed), so the branches stay visually separate.
#   - Frame placement requires an empty ring+cavity area, forbids cavity adjacency
#     to the hosting line, and forbids ring adjacency to foreign non-bg colors.
#   - Each active line must successfully host at least one loop; oracle checks then
#     verify both line colors actually gain filled cavity area in the output.
#
# Visual clarity:
#   - One seed, two branch colors, and two corresponding fill colors are directly
#     visible. Reusing ring colors across branches makes the new mapping observable.
#   - Distractors remain disconnected and sparse, so they add counterexamples
#     without obscuring the branch-to-cavity correspondences.


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


def _touches_border(pts: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts)


def _same_color_component8_from(g: np.ndarray, start: tuple[int, int], color: int) -> set[tuple[int, int]]:
    h, w = g.shape
    q = deque([start])
    vis = {start}
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors8(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in vis and int(g[nr, nc]) == color:
                vis.add((nr, nc))
                q.append((nr, nc))
    return vis


def _seed_line_components(g: np.ndarray, seed_rc: tuple[int, int], bg: int) -> list[tuple[int, set[tuple[int, int]]]]:
    h, w = g.shape
    by_color: dict[int, tuple[int, int]] = {}
    for nr, nc in neighbors4(*seed_rc):
        if 0 <= nr < h and 0 <= nc < w:
            v = int(g[nr, nc])
            if v != bg:
                by_color.setdefault(v, (nr, nc))
    items = []
    for line_col, start in by_color.items():
        items.append((line_col, _same_color_component8_from(g, start, line_col)))
    return items


def _cavities(g: np.ndarray, bg: int) -> list[tuple[set[tuple[int, int]], set[tuple[int, int]], set[int]]]:
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
        if any(p in reg for p in corners if int(g[p]) == bg) or _touches_border(list(reg), h, w):
            outside_idxs.add(i)

    cavities = []
    for i, (reg, boundary) in enumerate(regions):
        if i in outside_idxs or not boundary:
            continue
        boundary_colors = {int(g[p]) for p in boundary}
        cavities.append((reg, boundary, boundary_colors))
    return cavities


def _boundary_touches_line(boundary: set[tuple[int, int]], line_component: set[tuple[int, int]], h: int, w: int) -> bool:
    for r, c in boundary:
        for nr, nc in neighbors8(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) in line_component:
                return True
    return False


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    out = g.copy()

    seeds = _seed_pixels(g, bg)
    if len(seeds) != 1:
        return out
    seed_rc = seeds[0]

    line_infos = _seed_line_components(g, seed_rc, bg)
    if len(line_infos) < 2:
        return out

    cavities = _cavities(g, bg)
    for cav, boundary, boundary_colors in cavities:
        if len(boundary_colors) != 1:
            continue
        touching_lines = []
        for line_col, line_component in line_infos:
            if _boundary_touches_line(boundary, line_component, h, w):
                touching_lines.append(line_col)
        if len(touching_lines) != 1:
            continue
        fill_col = touching_lines[0]
        for r, c in cav:
            if int(out[r, c]) == bg:
                out[r, c] = fill_col
    return out


# ---------------------------------------------------------------------------
# Generator config
# ---------------------------------------------------------------------------

GRID_H_RANGE = (10, 30)
GRID_W_RANGE = (10, 30)
BG_COLORS = [0, 5]

ACTIVE_LINE_COUNT = 2
MIN_LINE_LEN = 5
LINE_LEN_RANGE = (8, 32)

FRAME_COLORS_MAX = 4
ATTACHED_LOOPS_PER_LINE = (1, 3)
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

DISTRACTOR_NETWORK_PROB = 0.45
DISTRACTOR_NETWORK_TRIES = 40
DISTRACTOR_LINE_LEN = (5, 12)
DISTRACTOR_NETWORK_FRAMES = (1, 2)
DISTRACTOR_SOLO_MAX = 2

GEN_RETRIES = 400

DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
ORTHO_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _frame_ring_coords(r0: int, c0: int, ch: int, cw: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    h, w = ch + 2, cw + 2
    ring = []
    for dr in range(h):
        for dc in range(w):
            on_border = dr == 0 or dc == 0 or dr == h - 1 or dc == w - 1
            corner = (dr in (0, h - 1)) and (dc in (0, w - 1))
            if on_border and not corner:
                ring.append((r0 + dr, c0 + dc))
    cavity = [(r0 + 1 + dr, c0 + 1 + dc) for dr in range(ch) for dc in range(cw)]
    return ring, cavity


def _validate_frame_placement(canvas: np.ndarray, ring: list[tuple[int, int]], cavity: list[tuple[int, int]], line_col: int, h: int, w: int) -> bool:
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
            v = int(canvas[nr, nc])
            if v != -1 and v != line_col:
                return False

    for rr, cc in ring:
        for nr, nc in neighbors8(rr, cc):
            if _in_bounds(h, w, nr, nc) and canvas[nr, nc] == line_col:
                return True
    return False


def _weighted_dirs(dr_bias: int, dc_bias: int) -> list[tuple[int, int]]:
    dirs = []
    for ddr, ddc in DIAG_DIRS:
        aligned = ddr == dr_bias or ddc == dc_bias
        dirs.extend([(ddr, ddc)] * (DIAG_WEIGHT_ALIGNED if aligned else DIAG_WEIGHT_OTHER))
    for ddr, ddc in ORTHO_DIRS:
        aligned = (ddr == dr_bias and ddc == 0) or (ddc == dc_bias and ddr == 0)
        dirs.extend([(ddr, ddc)] * (ORTHO_WEIGHT_ALIGNED if aligned else ORTHO_WEIGHT_OTHER))
    return dirs


def _grow_line(grid: np.ndarray, start: tuple[int, int], line_col: int, target_len: int, rng: random.Random, dr_bias: int, dc_bias: int, seed_rc: tuple[int, int] | None = None) -> list[tuple[int, int]]:
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
                if not _in_bounds(h, w, ar, ac) or (ar, ac) == cur:
                    continue
                if seed_rc is not None and (ar, ac) == seed_rc:
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


def _try_attach_frame(canvas: np.ndarray, line_path: list[tuple[int, int]], line_col: int, frame_col: int, rng: random.Random, max_tries: int = FRAME_ATTACH_TRIES) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
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


def _try_place_distractor_network(canvas: np.ndarray, line_col: int, frame_colors_pool: list[int], rng: random.Random) -> bool:
    h, w = canvas.shape
    for _ in range(DISTRACTOR_NETWORK_TRIES):
        r = rng.randint(2, h - 3)
        c = rng.randint(2, w - 3)
        if any(canvas[r + dr, c + dc] != -1 for dr in range(-1, 2) for dc in range(-1, 2)):
            continue
        canvas[r, c] = line_col
        dr_bias = rng.choice([-1, 1])
        dc_bias = rng.choice([-1, 1])
        path = _grow_line(canvas, (r, c), line_col, rng.randint(*DISTRACTOR_LINE_LEN), rng, dr_bias, dc_bias)
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


def _edge_seed_config(h: int, w: int, rng: random.Random) -> tuple[tuple[int, int], list[tuple[int, int]], list[tuple[int, int]]]:
    edges = []
    if w >= 9:
        c = rng.randint(3, w - 4)
        edges.append(((0, c), [(0, c - 1), (0, c + 1)], [(1, -1), (1, 1)]))
        edges.append(((h - 1, c), [(h - 1, c - 1), (h - 1, c + 1)], [(-1, -1), (-1, 1)]))
    if h >= 9:
        r = rng.randint(3, h - 4)
        edges.append(((r, 0), [(r - 1, 0), (r + 1, 0)], [(-1, 1), (1, 1)]))
        edges.append(((r, w - 1), [(r - 1, w - 1), (r + 1, w - 1)], [(-1, -1), (1, -1)]))
    seed_rc, starts, biases = rng.choice(edges)
    return seed_rc, starts, biases


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    bg = rng.choice(BG_COLORS)
    palette = [c for c in range(10) if c != bg]
    rng.shuffle(palette)

    if len(palette) < 6:
        return None

    canvas = -np.ones((h, w), dtype=int)
    seed_col = palette.pop()
    line_colors = [palette.pop() for _ in range(ACTIVE_LINE_COUNT)]

    num_frame_cols = rng.randint(1, min(FRAME_COLORS_MAX, len(palette) - 1))
    frame_colors_pool = [palette.pop() for _ in range(num_frame_cols)]
    distractor_line_colors = palette[:]

    seed_rc, starts, biases = _edge_seed_config(h, w, rng)
    if canvas[seed_rc] != -1 or any(canvas[p] != -1 for p in starts):
        return None
    canvas[seed_rc] = seed_col

    lines_info = []
    for i, start in enumerate(starts):
        if canvas[start] != -1:
            return None
        line_col = line_colors[i]
        canvas[start] = line_col
        target_len = rng.randint(LINE_LEN_RANGE[0], min(LINE_LEN_RANGE[1], h + w))
        dr_bias, dc_bias = biases[i]
        path = _grow_line(canvas, start, line_col, target_len, rng, dr_bias, dc_bias, seed_rc=seed_rc)
        if len(path) < MIN_LINE_LEN:
            return None
        lines_info.append((line_col, path))

    total_loops = 0
    for line_col, path in lines_info:
        attached_target = rng.randint(*ATTACHED_LOOPS_PER_LINE)
        attached = 0
        for _ in range(attached_target):
            frame_col = rng.choice(frame_colors_pool)
            if _try_attach_frame(canvas, path, line_col, frame_col, rng) is not None:
                attached += 1
        if attached == 0:
            return None
        total_loops += attached
    if total_loops > TOTAL_LOOPS_MAX:
        return None

    if frame_colors_pool and rng.random() < DISTRACTOR_NETWORK_PROB and distractor_line_colors:
        _try_place_distractor_network(canvas, rng.choice(distractor_line_colors), frame_colors_pool, rng)
    else:
        for _ in range(rng.randint(0, DISTRACTOR_SOLO_MAX)):
            _try_attach_distractor_frame(canvas, rng.choice(frame_colors_pool), rng)

    grid = np.where(canvas == -1, bg, canvas).astype(int)

    vals, cnts = np.unique(grid, return_counts=True)
    if int(vals[np.argmax(cnts)]) != bg:
        return None
    if int((grid == seed_col).sum()) != 1:
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None

    for line_col, _ in lines_info:
        if int((out == line_col).sum()) <= int((grid == line_col).sum()):
            return None
    if len({int(v) for v in np.unique(out[grid == bg]) if v != bg}) < ACTIVE_LINE_COUNT:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted attempts")
