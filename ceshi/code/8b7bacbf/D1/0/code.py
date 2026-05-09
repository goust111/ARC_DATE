# Variant of 8b7bacbf
# Dimension: D1 — 路径追踪连通性 (path)
# Change: 8连通追踪（path_tracing）——从边缘种子像素出发，沿8连通（含对角线）的非背景像素追踪出完整线路组件... → 4连通追踪
#
# Design analysis
# a) Concrete behavior:
#    - Original samples use diagonal-friendly line drawings and "rounded rectangle" rings whose corners are
#      removed, so both the line and the ring rely on 8-connectivity.
#    - In this variant the traced structure is orthogonal: seed -> 4-connected line -> 4-connected rectangular
#      frame ring. The output still flood-fills the enclosed cavity with the seed color.
#    - Small trace example:
#      Original: seed S touches line L; L can reach a frame F through a diagonal touch, so 8-connected tracing
#      includes F and its cavity gets filled.
#      Variant: the same diagonal touch no longer connects. Only if S reaches F through up/down/left/right
#      contacts does F belong to the traced component and only then is its cavity filled.
#    - To make this visible, the generator always adds at least one "diagonal probe" frame: it touches a seed
#      line only at a corner. Legacy 8-connected tracing would fill it; the new 4-connected tracing must not.
# b) Complexity calibration:
#    - The task still has 1-2 seeds, multiple frame colors, attached loops, and optional distractor structures.
#    - The key free variable is small and visible: orthogonal contact vs diagonal-only contact.
#    - Repeated rectangular loops provide enough support for the solver to infer "closed cavity fill" while the
#      diagonal probe demonstrates that connectivity, not color identity, controls filling.
# c) Implicit-assumption audit against the original main():
#    - `_nonbg_component8_from(...)`: depended on every meaningful connection being allowed diagonally.
#      Invalid now; replaced by configurable connectivity and used as 4-connected in `main()`.
#    - Original loop boundary witness used `neighbors8(...) == line_col`: that treated diagonal touch as valid
#      line attachment. In this variant the intended attached loops are orthogonally attached, so the solver now
#      requires a 4-neighbor witness as well.
#    - Original generator used rings without corners. Those are not 4-connected loops, so keeping them would
#      silently break the new tracing model. The ring geometry is therefore changed to full rectangles including
#      corners; this is a necessary compatibility edit, not a new puzzle idea.
# d) Placement legality:
#    - Attached rings are placed only when their ring and cavity cells are empty, their cavity is not adjacent
#      to the line, and at least one ring cell is 4-adjacent to the owning line color.
#    - Diagonal probe rings are placed only when they have at least one diagonal contact to the owning line but
#      zero 4-neighbor contacts to any non-background pixel.
#    - Foreign-color 8-adjacency around a placed ring is forbidden to avoid accidental merges between networks.
# e) Visual observability:
#    - The line style changes from diagonal meanders to orthogonal bends.
#    - The rings change from chamfered/rounded loops to square rectangles.
#    - Each generated grid contains at least one visible corner-touch distractor that remains empty in the output.
# f) Visual cleanliness:
#    - The line is kept 1px wide and self-avoiding with 8-neighbor clearance, preventing visual tangles.
#    - Rings keep a 1-cell buffer from unrelated structures.
# g) Color reasoning:
#    - No color ordering or numeric property matters; only the unique border seed color labels the fill.
#    - A small number of reused frame colors keeps the scene readable.
# h) Generator anti-degeneration check:
#    - After generation, the grid is solved twice: once with the new 4-connected tracing rule and once with the
#      legacy 8-connected rule. The sample is accepted only if the outputs differ, which guarantees that the D1
#      control variable is active on the generated input.
# i) New main() assumptions and generator guarantees:
#    - Assumption: each seed has at least one 4-neighbor non-background pixel. Guaranteed by corner seeding.
#    - Assumption: any fillable ring is a single-color 4-connected rectangle. Guaranteed by ring construction.
#    - Assumption: a diagonal-only frame should stay unfilled. Guaranteed by explicit diagonal-probe placement.

from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import components_mask, get_bg, neighbors4, neighbors8


def _components_mask(mask: np.ndarray, conn8: bool) -> list[list[tuple[int, int]]]:
    connectivity = 8 if conn8 else 4
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=connectivity)]


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


def _nonbg_component_from(
    g: np.ndarray,
    start: tuple[int, int],
    bg: int,
    conn8: bool,
) -> set[tuple[int, int]]:
    h, w = g.shape
    neigh_fn = neighbors8 if conn8 else neighbors4
    st = [start]
    vis = {start}
    while st:
        r, c = st.pop()
        for nr, nc in neigh_fn(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in vis and int(g[nr, nc]) != bg:
                vis.add((nr, nc))
                st.append((nr, nc))
    return vis


def _solve_with_trace_mode(grid: np.ndarray, trace_conn8: bool, line_adj8: bool) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    neigh_line = neighbors8 if line_adj8 else neighbors4
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
        assert neigh_cols, "seed pixel must have at least one non-bg 4-neighbor"
        line_col = Counter(neigh_cols).most_common(1)[0][0]

        traced = _nonbg_component_from(g, (sr, sc), bg, conn8=trace_conn8)

        for cav, boundary, bcols in cavities:
            if len(bcols) != 1:
                continue
            boundary_in_traced = boundary & traced
            if not boundary_in_traced:
                continue

            ok = False
            for r, c in boundary_in_traced:
                for nr, nc in neigh_line(r, c):
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Variant rule:
    - border seed pixels still identify which cavities should be filled
    - but tracing from the seed is strictly 4-connected
    - and an attached ring must have a 4-neighbor witness to the line color
    """
    return _solve_with_trace_mode(grid, trace_conn8=False, line_adj8=False)


# --------------- generator ---------------

# ========================== CFG ==========================
GRID_H_RANGE = (8, 30)
GRID_W_RANGE = (8, 30)
BG_COLORS = [0, 5]

SEED_COUNT_WEIGHTS = [1, 2, 2]
MIN_LINE_LEN = 4
LINE_LEN_RANGE = (8, 28)

FRAME_COLORS_MAX = 4
TOTAL_LOOPS_MAX = 9
CAVITY_SIZE_WEIGHTS = [2, 2, 2, 3, 3, 4]
DISTRACTOR_CAVITY_WEIGHTS = [2, 2, 3, 4]
FRAME_PLACEMENT_PAD = 1
FRAME_ATTACH_TRIES = 50
DISTRACTOR_FRAME_TRIES = 40
DIAG_PROBE_TRIES = 60

STRAIGHT_WEIGHT = 4
TURN_WEIGHT = 3
INWARD_WEIGHT = 2
BORDER_AVOIDANCE_PROB = 0.85
BIAS_FLIP_PROB = 0.3

DISTRACTOR_NETWORK_PROB = 0.5
DISTRACTOR_NETWORK_TRIES = 50
DISTRACTOR_LINE_LEN = (5, 12)
DISTRACTOR_SOLO_MAX = 1
DISTRACTOR_NETWORK_FRAMES = (1, 2)

GEN_RETRIES = 400
# =========================================================

ORTHO_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _is_orth_neighbor(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1


def _frame_ring_coords(r0: int, c0: int, ch: int, cw: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Return (ring_coords, cavity_coords) for a full rectangular ring."""
    H, W = ch + 2, cw + 2
    ring = []
    for dr in range(H):
        for dc in range(W):
            if dr == 0 or dr == H - 1 or dc == 0 or dc == W - 1:
                ring.append((r0 + dr, c0 + dc))
    cavity = [(r0 + 1 + dr, c0 + 1 + dc) for dr in range(ch) for dc in range(cw)]
    return ring, cavity


def _validate_frame_placement(canvas, ring, cavity, line_col, h, w) -> bool:
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
    has_orth_touch = False
    for rr, cc in ring:
        for nr, nc in neighbors8(rr, cc):
            if not _in_bounds(h, w, nr, nc) or (nr, nc) in ring_set:
                continue
            v = canvas[nr, nc]
            if v == -1:
                continue
            if v == line_col:
                if _is_orth_neighbor((rr, cc), (nr, nc)):
                    has_orth_touch = True
                continue
            return False
    return has_orth_touch


def _validate_diagonal_probe_frame(canvas, ring, cavity, line_col, h, w) -> bool:
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
    has_diag_touch = False
    for rr, cc in ring:
        for nr, nc in neighbors8(rr, cc):
            if not _in_bounds(h, w, nr, nc) or (nr, nc) in ring_set:
                continue
            v = canvas[nr, nc]
            if v == -1:
                continue
            if v == line_col:
                if _is_orth_neighbor((rr, cc), (nr, nc)):
                    return False
                has_diag_touch = True
                continue
            return False
    return has_diag_touch


def _grow_line(grid, start, line_col, target_len, rng):
    """Grow a meandering 4-connected 1px-wide orthogonal path."""
    h, w = grid.shape
    path = [start]
    dr_bias = 1 if start[0] < h // 2 else -1
    dc_bias = 1 if start[1] < w // 2 else -1
    cur = start
    path_set = {start}
    last_dir = None

    while len(path) < target_len:
        weighted_dirs = []
        for ddr, ddc in ORTHO_DIRS:
            if last_dir is not None and (ddr, ddc) == (-last_dir[0], -last_dir[1]):
                continue
            nr, nc = cur[0] + ddr, cur[1] + ddc
            if not _in_bounds(h, w, nr, nc) or grid[nr, nc] != -1:
                continue
            if any((nnr, nnc) != cur and (nnr, nnc) in path_set for nnr, nnc in neighbors8(nr, nc)):
                continue
            if (nr == 0 or nc == 0 or nr == h - 1 or nc == w - 1) and rng.random() < BORDER_AVOIDANCE_PROB:
                continue

            weight = TURN_WEIGHT
            if last_dir == (ddr, ddc):
                weight += STRAIGHT_WEIGHT
            if ddr == dr_bias or ddc == dc_bias:
                weight += INWARD_WEIGHT
            weighted_dirs.extend([(ddr, ddc)] * weight)

        if not weighted_dirs:
            if rng.random() < BIAS_FLIP_PROB:
                dr_bias *= -1
            if rng.random() < BIAS_FLIP_PROB:
                dc_bias *= -1
            break

        ddr, ddc = rng.choice(weighted_dirs)
        cur = (cur[0] + ddr, cur[1] + ddc)
        path.append(cur)
        path_set.add(cur)
        grid[cur] = line_col
        last_dir = (ddr, ddc)

    return path


def _halo_offsets(fh: int, fw: int) -> list[tuple[int, int]]:
    return [
        (dr, dc)
        for dr in range(-1, fh + 1)
        for dc in range(-1, fw + 1)
        if not (0 <= dr < fh and 0 <= dc < fw)
    ]


def _try_attach_frame(canvas, line_path, line_col, frame_col, rng, max_tries=FRAME_ATTACH_TRIES):
    """Place a ring that is 4-neighbor attached to the owning line."""
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


def _try_place_diagonal_probe_frame(canvas, line_path, line_col, frame_col, rng) -> bool:
    """
    Place a loop that only corner-touches the line.

    This is the key D1 witness: legacy 8-connected tracing would merge it into the
    seed component, while the new 4-connected tracing must keep it separate.
    """
    h, w = canvas.shape
    pad = FRAME_PLACEMENT_PAD
    for _ in range(DIAG_PROBE_TRIES):
        ch = rng.choice(DISTRACTOR_CAVITY_WEIGHTS)
        cw = rng.choice(DISTRACTOR_CAVITY_WEIGHTS)
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
            if not _validate_diagonal_probe_frame(canvas, ring, cavity, line_col, h, w):
                continue
            for rr, cc in ring:
                canvas[rr, cc] = frame_col
            return True
    return False


def _try_attach_distractor_frame(canvas, frame_col, rng, max_tries=DISTRACTOR_FRAME_TRIES):
    """Place a loop in empty space, not touching any non-bg pixel."""
    h, w = canvas.shape
    for _ in range(max_tries):
        ch = rng.choice(DISTRACTOR_CAVITY_WEIGHTS)
        cw = rng.choice(DISTRACTOR_CAVITY_WEIGHTS)
        fh, fw = ch + 2, cw + 2
        if h - fh - 1 <= 1 or w - fw - 1 <= 1:
            return None
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
    """Place a disconnected orthogonal line+frame distractor with no seed."""
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
        max_len = min(LINE_LEN_RANGE[1], max(LINE_LEN_RANGE[0], (h * w) // 5))
        target_len = rng.randint(LINE_LEN_RANGE[0], max_len)
        path = _grow_line(canvas, first_rc, line_colors[i], target_len, rng)
        if len(path) < MIN_LINE_LEN:
            return None
        lines_info.append((seed_colors[i], seed_rc, line_colors[i], path))

    total_loops_target = rng.randint(max(num_seeds + 1, 2), TOTAL_LOOPS_MAX)
    max_distractors = max(1, total_loops_target - num_seeds)
    num_distractors_target = rng.randint(1, max_distractors)
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

    probe_owner = rng.randrange(num_seeds)
    probe_ok = _try_place_diagonal_probe_frame(
        canvas,
        lines_info[probe_owner][3],
        lines_info[probe_owner][2],
        rng.choice(frame_colors_pool),
        rng,
    )
    if not probe_ok:
        return None

    extra_distractors = max(0, num_distractors_target - 1)
    if extra_distractors > 0:
        if rng.random() < DISTRACTOR_NETWORK_PROB:
            _try_place_distractor_network(canvas, rng.choice(line_colors), frame_colors_pool, rng)
            extra_distractors -= 1
        for _ in range(min(extra_distractors, DISTRACTOR_SOLO_MAX)):
            _try_attach_distractor_frame(canvas, rng.choice(frame_colors_pool), rng)

    grid = np.where(canvas == -1, bg, canvas).astype(int)

    vals, cnts = np.unique(grid, return_counts=True)
    if int(vals[np.argmax(cnts)]) != bg:
        return None

    for seed_col, _, _, _ in lines_info:
        if int((grid == seed_col).sum()) != 1:
            return None

    try:
        out = main(grid)
        legacy_out = _solve_with_trace_mode(grid, trace_conn8=True, line_adj8=True)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None
    if np.array_equal(out, legacy_out):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError("generate_input: exhausted attempts")
