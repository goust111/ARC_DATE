import numpy as np
from collections import Counter, deque

from arc_utils import components_mask, get_bg, neighbors4, neighbors8
import random
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill enclosed regions along a line seeded by unique border pixels.

    Input structure:
    - bg = most common color.
    - Colored lines/drawings on the grid.
    - Unique single-pixel colors on the grid border mark seed endpoints of each line.

    Rule:
    - Each border seed pixel connects (8-connected) to a line drawing.
    - Find closed bg cavities whose boundary touches that line.
    - Flood-fill each such cavity with the seed pixel's color.

    Key insight: Only cavities reachable through the seed's own line get filled; the seed's unique color acts as a per-line fill label.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    out = g.copy()

    # Precompute background regions (4-connected). Exclude those containing background corners as "outside".
    bg_comps = _components_mask(g == bg, conn8=False)
    regions = []  # list of (set_pts, boundary_set, touches_border, size)
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

        # Trace the connected "drawing" reachable from the seed (8-connected, non-background).
        traced = _nonbg_component8_from(g, (sr, sc), bg)

        for cav, boundary, bcols in cavities:
            if len(bcols) != 1:
                continue
            boundary_in_traced = boundary & traced
            if not boundary_in_traced:
                continue

            # Require that the traced boundary is adjacent (8-neigh) to at least one pixel of the line color.
            # This prevents filling regions that are only connected through other structures, not the line itself.
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

            for r, c in cav:
                if int(out[r, c]) == bg:
                    out[r, c] = seed_col

    return out

# --------------- generator (attempt_1_distractor_network) ---------------

# ========================== CFG ==========================
# Canvas / grid
GRID_H_RANGE = (8, 30)             # grid height bounds
GRID_W_RANGE = (8, 30)             # grid width bounds
BG_COLORS = [0, 5]                 # background color choices

# Seeds
SEED_COUNT_WEIGHTS = [1, 2, 2]     # bias toward 2 (2-seed attempts fail more)
MIN_LINE_LEN = 4                   # discard attempt if line shorter than this
LINE_LEN_RANGE = (8, 35)           # target line length bounds (clamped by h+w)

# Frame loops
FRAME_COLORS_MAX = 4               # max distinct frame colors
TOTAL_LOOPS_MAX = 9                # max total loops per grid
CAVITY_SIZE_WEIGHTS = [2, 2, 2, 2, 2, 3]       # cavity h/w choices (attached)
DISTRACTOR_CAVITY_WEIGHTS = [2, 2, 2, 3]        # cavity h/w choices (distractor)
FRAME_PLACEMENT_PAD = 1            # min distance from frame bbox to grid edge
FRAME_ATTACH_TRIES = 40            # retries per frame attachment attempt
DISTRACTOR_FRAME_TRIES = 40        # retries per distractor frame placement

# Line growth
DIAG_WEIGHT_ALIGNED = 5            # weight for bias-aligned diagonal steps
DIAG_WEIGHT_OTHER = 1              # weight for non-aligned diagonal steps
ORTHO_WEIGHT_ALIGNED = 2           # weight for bias-aligned orthogonal steps
ORTHO_WEIGHT_OTHER = 1             # weight for non-aligned orthogonal steps
BORDER_AVOIDANCE_PROB = 0.85       # probability of skipping border cells
BIAS_FLIP_PROB = 0.3               # probability of flipping bias when stuck

# Distractor network
DISTRACTOR_NETWORK_PROB = 0.6      # chance of line+frame network vs solo frames
DISTRACTOR_NETWORK_TRIES = 50      # retries for placing distractor network
DISTRACTOR_LINE_LEN = (5, 15)      # distractor line length range
DISTRACTOR_SOLO_MAX = 2            # max solo distractor frames
DISTRACTOR_NETWORK_FRAMES = (1, 3) # frames per distractor network

# Generation
GEN_RETRIES = 300                  # outer retry budget in generate_input()
# =========================================================

DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
ORTHO_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _in_bounds(h, w, r, c):
    return 0 <= r < h and 0 <= c < w


# ---------------------------------------------------------------------------
# Frame geometry
# ---------------------------------------------------------------------------

def _frame_ring_coords(r0, c0, ch, cw):
    """Return (ring_coords, cavity_coords) for a rounded-rect frame.

    Frame bbox is [r0, r0+ch+1] x [c0, c0+cw+1] (inclusive).
    Ring = bbox boundary minus 4 corners.
    """
    H, W = ch + 2, cw + 2
    ring = []
    for dr in range(H):
        for dc in range(W):
            on_bdr = (dr == 0 or dr == H - 1 or dc == 0 or dc == W - 1)
            corner = (dr in (0, H - 1)) and (dc in (0, W - 1))
            if on_bdr and not corner:
                ring.append((r0 + dr, c0 + dc))
    cavity = [(r0 + 1 + dr, c0 + 1 + dc) for dr in range(ch) for dc in range(cw)]
    return ring, cavity


def _validate_frame_placement(canvas, ring, cavity, line_col, h, w):
    """Check that ring+cavity can be placed on canvas without conflicts.

    Returns True if: ring and cavity cells are empty, no cavity cell is
    8-adjacent to line_col, ring is not 8-adjacent to foreign colors, and
    at least one ring cell is 8-adjacent to a line_col pixel.
    """
    # Ring and cavity cells must all be empty
    for rr, cc in ring:
        if canvas[rr, cc] != -1:
            return False
    for rr, cc in cavity:
        if canvas[rr, cc] != -1:
            return False

    # No cavity cell may be 8-adjacent to any line_col pixel
    for cr, cc in cavity:
        for nr, nc in neighbors8(cr, cc):
            if _in_bounds(h, w, nr, nc) and canvas[nr, nc] == line_col:
                return False

    # Ring must not be 8-adjacent to any non-bg, non-line_col pixel
    ring_set = set(ring)
    for rr, cc in ring:
        for nr, nc in neighbors8(rr, cc):
            if not _in_bounds(h, w, nr, nc) or (nr, nc) in ring_set:
                continue
            v = canvas[nr, nc]
            if v != -1 and v != line_col:
                return False

    # At least one ring cell must be 8-adjacent to a line_col pixel
    for rr, cc in ring:
        for nr, nc in neighbors8(rr, cc):
            if _in_bounds(h, w, nr, nc) and canvas[nr, nc] == line_col:
                return True
    return False


# ---------------------------------------------------------------------------
# Line growth
# ---------------------------------------------------------------------------

def _grow_line(grid, start, line_col, target_len, rng):
    """Grow a meandering 8-connected 1px-wide path of line_col from start."""
    h, w = grid.shape
    path = [start]
    dr_bias = 1 if start[0] < h // 2 else -1
    dc_bias = 1 if start[1] < w // 2 else -1
    cur = start
    path_set = {start}
    while len(path) < target_len:
        # Build weighted direction list: diagonals preferred, bias-aligned boosted
        dirs = []
        for ddr, ddc in DIAG_DIRS:
            aligned = (ddr == dr_bias or ddc == dc_bias)
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
            # Keep path thin: reject if 8-adjacent to existing path (except cur)
            if any((nnr, nnc) != cur and (nnr, nnc) in path_set
                   for nnr, nnc in neighbors8(nr, nc)):
                continue
            # Strongly avoid grid border
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


# ---------------------------------------------------------------------------
# Frame attachment
# ---------------------------------------------------------------------------

def _halo_offsets(fh, fw):
    """Compute offsets around a frame bbox that lie outside it (the halo)."""
    return [(dr, dc) for dr in range(-1, fh + 1) for dc in range(-1, fw + 1)
            if not (0 <= dr < fh and 0 <= dc < fw)]


def _try_attach_frame(canvas, line_path, line_col, frame_col, rng, max_tries=FRAME_ATTACH_TRIES):
    """Try to place a frame loop 8-adjacent to some line pixel.

    canvas is modified in place; returns (ring, cavity) or None on failure.
    """
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


def _try_attach_distractor_frame(canvas, frame_col, rng, max_tries=DISTRACTOR_FRAME_TRIES):
    """Place a loop in empty space, not touching any non-bg pixel."""
    h, w = canvas.shape
    for _ in range(max_tries):
        ch = rng.choice(DISTRACTOR_CAVITY_WEIGHTS)
        cw = rng.choice(DISTRACTOR_CAVITY_WEIGHTS)
        fh, fw = ch + 2, cw + 2
        r0 = rng.randint(1, h - fh - 1)
        c0 = rng.randint(1, w - fw - 1)
        # Check the full bbox + 1-cell margin is empty
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


def _try_place_distractor_network(canvas, line_col, frame_colors_pool, rng):
    """Place a disconnected line+frame distractor (no seed)."""
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


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def _corner_config(corner_kind, h, w, rng):
    """Return (seed_rc, first_line_rc) for corner_kind in {TL,TR,BL,BR}."""
    if corner_kind == 'TL':
        seed = (0, 0)
        first = rng.choice([(0, 1), (1, 0)])
    elif corner_kind == 'TR':
        seed = (0, w - 1)
        first = rng.choice([(0, w - 2), (1, w - 1)])
    elif corner_kind == 'BL':
        seed = (h - 1, 0)
        first = rng.choice([(h - 1, 1), (h - 2, 0)])
    elif corner_kind == 'BR':
        seed = (h - 1, w - 1)
        first = rng.choice([(h - 1, w - 2), (h - 2, w - 1)])
    else:
        raise ValueError(corner_kind)
    return seed, first


def _distribute_loop_budget(num_seeds, num_attached_target, rng):
    """Distribute attached-loop budget across seeds, guaranteeing >= 1 each."""
    seed_order = list(range(num_seeds))
    rng.shuffle(seed_order)
    per_seed = [0] * num_seeds
    remaining = num_attached_target
    # Guarantee at least 1 per seed
    while remaining > 0 and any(per_seed[i] == 0 for i in range(num_seeds)):
        for i in seed_order:
            if per_seed[i] == 0 and remaining > 0:
                per_seed[i] += 1
                remaining -= 1
    # Distribute remainder randomly
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

    all_corners = ['TL', 'TR', 'BL', 'BR']
    rng.shuffle(all_corners)
    corners_used = all_corners[:num_seeds]

    num_frame_cols = rng.randint(1, min(FRAME_COLORS_MAX, len(palette)))
    frame_colors_pool = [palette.pop() for _ in range(num_frame_cols)]

    # Grow seed lines
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

    # Decide loop counts and distribute across seeds
    total_loops_target = rng.randint(num_seeds, TOTAL_LOOPS_MAX)
    num_distractors_target = rng.randint(0, max(0, total_loops_target - num_seeds))
    num_attached_target = total_loops_target - num_distractors_target
    per_seed = _distribute_loop_budget(num_seeds, num_attached_target, rng)

    # Attach frames to each seed's line
    for i, (_, _, line_col, path) in enumerate(lines_info):
        attached = 0
        for _ in range(per_seed[i]):
            fc = rng.choice(frame_colors_pool)
            if _try_attach_frame(canvas, path, line_col, fc, rng) is not None:
                attached += 1
        if attached == 0:
            return None

    # Place distractors
    if rng.random() < DISTRACTOR_NETWORK_PROB:
        _try_place_distractor_network(canvas, rng.choice(line_colors), frame_colors_pool, rng)
    else:
        for _ in range(rng.randint(0, DISTRACTOR_SOLO_MAX)):
            _try_attach_distractor_frame(canvas, rng.choice(frame_colors_pool), rng)

    # Finalize grid
    grid = np.where(canvas == -1, bg, canvas).astype(int)

    # Validate bg is most common color
    vals, cnts = np.unique(grid, return_counts=True)
    if int(vals[np.argmax(cnts)]) != bg:
        return None

    # Validate each seed color appears exactly once
    for seed_col, _, _, _ in lines_info:
        if int((grid == seed_col).sum()) != 1:
            return None

    # Oracle check: solver must change at least one cell
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
