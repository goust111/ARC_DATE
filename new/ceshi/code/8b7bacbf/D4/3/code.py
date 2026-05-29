# Variant of 8b7bacbf
# Dimension: D4 — 种子源对象定义 (src)
# Change: 网格边缘的唯一色单像素——在网格最外圈、颜色在整个网格中只出现一次的像素作为种子... → 模板查找型种子
#
# Design analysis for this D4 variant
# a) Concrete behavior and flow impact
#    - Original task: a seed is a single unique border pixel. Its 4-neighbor line color identifies the
#      8-connected drawing component; cavities touched by that component get flood-filled with the seed color.
#    - Variant: a seed is a border-touching 2x2 monochrome block. The solver must first detect that template.
#      The fill color is the block color. The attached line does NOT start via a 4-neighbor contact from the
#      border anymore: it starts from a single pixel that is only diagonally 8-adjacent to one inward-facing
#      corner of the 2x2 block. This makes the seed lookup materially different and visible in the input.
#    - Small trace:
#      Original: color 7 appears once at (0,0); line color 3 sits at (0,1); solver reads the unique pixel,
#      inspects its 4-neighbors, gets line color 3, then traces the non-bg component.
#      Variant: color 7 appears exactly four times as {(0,4),(0,5),(1,4),(1,5)}; the first line pixel is at
#      (2,6), so there is no 4-neighbor contact at all. The solver must find the 2x2 border template, inspect
#      its 8-neighborhood, infer line color 3 from the diagonal contact, then trace the component from the block.
# b) Complexity calibration
#    - The seed template is kept simple and repeated: always a monochrome 2x2 block touching one grid edge.
#    - The template color appears exactly four times, which avoids ambiguity with frame loops and keeps lookup
#      local and visually readable.
#    - Up to two seeds are used, with distinct colors and distinct edges, so the new behavior is repeatedly
#      observable within a sample set without exploding the case space.
# c) Hidden assumptions audit against the original main()
#    - _seed_pixels(...) assumed a seed color appears once and sits on the outer ring. This is invalid now;
#      replaced with _seed_blocks(...) that requires count==4, exact 2x2 geometry, and border contact.
#    - The original line-color inference read the seed's 4-neighbors. That fails here because the line is only
#      diagonally attached. The new _seed_line_color(...) reads the 8-neighborhood around the whole 2x2 block.
#    - _nonbg_component8_from(...) still works because the block, line, and frames remain one 8-connected
#      component.
#    - The cavity logic remains valid because D4 changes only the seed-source object, not the ring/cavity rule.
# d) Placement legality
#    - Every seed block is flush with exactly one border side and inset from corners enough to leave room for a
#      diagonal-only first line pixel.
#    - A 4-neighbor halo around each seed block is forbidden to that seed's line growth, so the line cannot later
#      form an accidental orthogonal attachment that would blur the intended D4 distinction.
#    - Distinct seed blocks use distinct sides and are checked for overlap with each other and with their first
#      line pixels. Frame placement logic is otherwise unchanged.
# e) Visual observability
#    - The solver-relevant cue is now a visible 2x2 border stamp rather than a lone border dot.
#    - Because the stamp color is reused as the eventual fill color, the object-level correspondence is easy to
#      infer from 2-4 demonstrations.
#    - The diagonal-only connection is also visible: the line begins one step inward and offset, not directly
#      adjacent to the seed block.
# f) Visual cleanliness
#    - Seed blocks are placed on different sides and separated from frame loops by the existing padding rules.
#    - The line is still 1px wide and the loops remain simple monochrome rounded-rect rings, so the new template
#      does not create extra clutter inside the main puzzle area.
# g) Color reasoning
#    - No hidden numeric ordering is used. The seed-block color itself is the fill color, exactly as in the
#      original task, just transferred from a single-pixel token to a 2x2 template token.
# 9) New assumptions introduced by the modified solver and how the generator guarantees them
#    - Assumption: each true seed color appears exactly four times and forms one border-touching 2x2 block.
#      Guarantee: generator reserves a distinct color per seed block and validates _seed_blocks(grid, bg).
#    - Assumption: at least one non-bg 8-neighbor around a seed block belongs to its line.
#      Guarantee: generator always places a diagonal first line pixel for every seed.
#    - Assumption: that local neighbor color is representative of the attached line.
#      Guarantee: generator forbids 4-neighbor halo cells around the block and keeps other structures away from
#      the seed area, so the only non-bg color next to the block is the line color.

import numpy as np
from collections import Counter
import random

from arc_utils import components_mask, get_bg, neighbors4, neighbors8


def _components_mask(mask: np.ndarray, conn8: bool) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8 if conn8 else 4)]


def _touches_border(pts: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts)


def _seed_blocks(g: np.ndarray, bg: int) -> list[list[tuple[int, int]]]:
    h, w = g.shape
    vals, cnts = np.unique(g, return_counts=True)
    blocks: list[tuple[int, int, list[tuple[int, int]]]] = []
    for v, cnt in zip(vals, cnts):
        v = int(v)
        if v == bg or int(cnt) != 4:
            continue
        pts = [tuple(map(int, rc)) for rc in np.argwhere(g == v)]
        rs = sorted({r for r, _ in pts})
        cs = sorted({c for _, c in pts})
        if len(rs) != 2 or len(cs) != 2:
            continue
        expected = {(rs[0], cs[0]), (rs[0], cs[1]), (rs[1], cs[0]), (rs[1], cs[1])}
        if set(pts) != expected:
            continue
        if not _touches_border(pts, h, w):
            continue
        blocks.append((rs[0], cs[0], sorted(pts)))
    blocks.sort(key=lambda x: (x[0], x[1]))
    return [pts for _, _, pts in blocks]


def _seed_line_color(g: np.ndarray, seed_pts: list[tuple[int, int]], bg: int) -> int:
    h, w = g.shape
    seed_set = set(seed_pts)
    seed_col = int(g[seed_pts[0]])
    neigh_cols: list[int] = []
    for r, c in seed_pts:
        for nr, nc in neighbors8(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in seed_set:
                v = int(g[nr, nc])
                if v != bg and v != seed_col:
                    neigh_cols.append(v)
    assert neigh_cols, "seed block must have at least one non-bg 8-neighbor"
    return Counter(neigh_cols).most_common(1)[0][0]


def _is_diag_only_seed_block(g: np.ndarray, seed_pts: list[tuple[int, int]], bg: int) -> bool:
    h, w = g.shape
    seed_set = set(seed_pts)
    seed_col = int(g[seed_pts[0]])
    has_8_nonseed = False
    for r, c in seed_pts:
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in seed_set:
                if int(g[nr, nc]) not in (bg, seed_col):
                    return False
        for nr, nc in neighbors8(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in seed_set:
                if int(g[nr, nc]) not in (bg, seed_col):
                    has_8_nonseed = True
    return has_8_nonseed


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
    Fill enclosed regions along a line seeded by border-touching 2x2 color blocks.

    Input structure:
    - bg = most common color.
    - Colored lines/drawings on the grid.
    - One or two monochrome 2x2 blocks touch the border; each block color is the fill color for one line.

    Rule:
    - Each seed block connects to its line only through diagonal 8-connectivity.
    - Find closed bg cavities whose boundary touches that seed's traced 8-connected drawing.
    - Flood-fill each such cavity with the seed block color.
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

    for seed_pts in _seed_blocks(g, bg):
        seed_col = int(g[seed_pts[0]])
        line_col = _seed_line_color(g, seed_pts, bg)

        traced = _nonbg_component8_from(g, seed_pts[0], bg)

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

            for r, c in cav:
                if int(out[r, c]) == bg:
                    out[r, c] = seed_col

    return out


# --------------- generator (D4 seed-template variant) ---------------

# ========================== CFG ==========================
GRID_H_RANGE = (8, 30)
GRID_W_RANGE = (8, 30)
BG_COLORS = [0, 5]

SEED_COUNT_WEIGHTS = [1, 2, 2]
SEED_LAYOUT_TRIES = 40
MIN_LINE_LEN = 4
LINE_LEN_RANGE = (8, 35)

FRAME_COLORS_MAX = 4
TOTAL_LOOPS_MAX = 9
CAVITY_SIZE_WEIGHTS = [2, 2, 2, 2, 2, 3]
DISTRACTOR_CAVITY_WEIGHTS = [2, 2, 2, 3]
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


def _in_bounds(h, w, r, c):
    return 0 <= r < h and 0 <= c < w


def _frame_ring_coords(r0, c0, ch, cw):
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


def _grow_line(grid, start, line_col, target_len, rng, forbidden: set[tuple[int, int]] | None = None):
    h, w = grid.shape
    forbidden = set() if forbidden is None else set(forbidden)
    path = [start]
    dr_bias = 1 if start[0] < h // 2 else -1
    dc_bias = 1 if start[1] < w // 2 else -1
    cur = start
    path_set = {start}
    while len(path) < target_len:
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
            if not _in_bounds(h, w, nr, nc) or grid[nr, nc] != -1 or (nr, nc) in forbidden:
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


def _try_attach_frame(canvas, line_path, line_col, frame_col, rng, max_tries=FRAME_ATTACH_TRIES):
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


def _try_place_distractor_network(canvas, line_col, frame_colors_pool, rng):
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


def _seed_block_forbidden(seed_pts: list[tuple[int, int]]) -> set[tuple[int, int]]:
    seed_set = set(seed_pts)
    forbidden = set()
    for r, c in seed_pts:
        for nr, nc in neighbors4(r, c):
            if (nr, nc) not in seed_set:
                forbidden.add((nr, nc))
    return forbidden


def _sample_seed_block(side: str, h: int, w: int, rng: random.Random):
    if side == 'T':
        c0 = rng.randint(2, w - 4)
        seed_pts = [(0, c0), (0, c0 + 1), (1, c0), (1, c0 + 1)]
        first = rng.choice([(2, c0 - 1), (2, c0 + 2)])
    elif side == 'B':
        c0 = rng.randint(2, w - 4)
        seed_pts = [(h - 2, c0), (h - 2, c0 + 1), (h - 1, c0), (h - 1, c0 + 1)]
        first = rng.choice([(h - 3, c0 - 1), (h - 3, c0 + 2)])
    elif side == 'L':
        r0 = rng.randint(2, h - 4)
        seed_pts = [(r0, 0), (r0, 1), (r0 + 1, 0), (r0 + 1, 1)]
        first = rng.choice([(r0 - 1, 2), (r0 + 2, 2)])
    elif side == 'R':
        r0 = rng.randint(2, h - 4)
        seed_pts = [(r0, w - 2), (r0, w - 1), (r0 + 1, w - 2), (r0 + 1, w - 1)]
        first = rng.choice([(r0 - 1, w - 3), (r0 + 2, w - 3)])
    else:
        raise ValueError(side)
    return seed_pts, first, _seed_block_forbidden(seed_pts)


def _sample_seed_layout(h: int, w: int, num_seeds: int, rng: random.Random):
    sides = ['T', 'B', 'L', 'R']
    for _ in range(SEED_LAYOUT_TRIES):
        chosen_sides = rng.sample(sides, num_seeds)
        layout = []
        occupied = set()
        ok = True
        for side in chosen_sides:
            placed = False
            for _ in range(SEED_LAYOUT_TRIES):
                seed_pts, first, forbidden = _sample_seed_block(side, h, w, rng)
                cells = set(seed_pts) | {first}
                if occupied & cells:
                    continue
                layout.append((seed_pts, first, forbidden))
                occupied |= cells
                placed = True
                break
            if not placed:
                ok = False
                break
        if ok:
            return layout
    return None


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

    seed_layout = _sample_seed_layout(h, w, num_seeds, rng)
    if seed_layout is None:
        return None

    canvas = -np.ones((h, w), dtype=int)

    seed_colors = [palette.pop() for _ in range(num_seeds)]
    line_colors = [palette.pop() for _ in range(num_seeds)]

    num_frame_cols = rng.randint(1, min(FRAME_COLORS_MAX, len(palette)))
    frame_colors_pool = [palette.pop() for _ in range(num_frame_cols)]

    lines_info = []
    for i, (seed_pts, first_rc, forbidden) in enumerate(seed_layout):
        for sr, sc in seed_pts:
            if canvas[sr, sc] != -1:
                return None
            canvas[sr, sc] = seed_colors[i]
        if canvas[first_rc] != -1:
            return None
        canvas[first_rc] = line_colors[i]
        target_len = rng.randint(LINE_LEN_RANGE[0], min(LINE_LEN_RANGE[1], h + w))
        path = _grow_line(canvas, first_rc, line_colors[i], target_len, rng, forbidden=forbidden)
        if len(path) < MIN_LINE_LEN:
            return None
        lines_info.append((seed_colors[i], seed_pts, line_colors[i], path))

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

    blocks = _seed_blocks(grid, bg)
    if len(blocks) != num_seeds:
        return None
    if not all(_is_diag_only_seed_block(grid, block, bg) for block in blocks):
        return None

    for seed_col, seed_pts, _, _ in lines_info:
        if int((grid == seed_col).sum()) != 4:
            return None
        if set(seed_pts) not in [set(block) for block in blocks]:
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
