# Variant of 8b7bacbf
# Dimension: D11 — 干扰项策略 (obstacle)
# Change: 0-5个未连接到种子路径的干扰框架环，可能包含独立的线路+框架网络... → 视觉相似的干扰种子

"""
Design analysis for the D11 variant
===================================

a) Concrete behavior and flow impact
------------------------------------
- Original visible pattern:
  - Real seeds are globally unique single pixels on the grid border.
  - Each real seed touches a meandering non-bg line.
  - That line reaches one or more single-color frame loops; their bg cavities are
    filled with the seed color in the output.
  - Some original samples also had disconnected distractor loops/networks.
- Variant visible pattern:
  - Real border seeds stay unchanged.
  - The old disconnected loop/network distractors are removed.
  - New distractors are interior "fake seeds": single pixels of colors not used
    as real seeds, placed away from the border.
  - Each fake seed is attached as a 1-pixel leaf to a real line, so visually it
    resembles the true seed-line relation, except it is not on the border.
- Concrete miniature trace:
  - Original logic on a component:
    border seed S touches line L -> trace the whole non-bg component from S ->
    find cavities whose ring boundary lies on that traced component -> fill with S.
  - Variant input adds an interior singleton F touching the same line L.
  - Correct variant logic:
    F must be ignored because it is not on the border; only S may label cavities.
  - Relaxed / incorrect logic without the border rule:
    F would also be treated as a seed and could color the same cavities wrongly.
  - Therefore the changed reasoning point is seed identification, not cavity
    detection or frame geometry.

b) Complexity calibration
-------------------------
- The new distractor must be obvious enough to matter, so generation requires at
  least one fake seed when colors/space allow.
- Fake seeds are not arbitrary dots: they are attached to real lines with exactly
  one non-bg 8-neighbor, making them visually seed-like rather than generic noise.
- To keep the puzzle readable, fake-seed count is capped at 5 and frame geometry
  otherwise stays close to the original task.

c) Implicit assumption audit of main()
--------------------------------------
- Background detection via most-common color still holds.
- Background-region extraction and single-color cavity boundary checks still hold;
  D11 does not change cavity semantics.
- The original `_seed_pixels()` already relied on two hidden assumptions:
  1. seed color appears exactly once globally
  2. seed pixel lies on the border
  In the original generator there were no interior singleton distractions, so the
  second condition was "quietly harmless". In this variant it becomes the key
  discriminant and must stay explicit.
- `line_col = most_common(non-bg 4-neighbor color around the seed)` still holds
  because only real border seeds are examined, and the generator guarantees each
  real seed touches at least one line pixel.
- The traced 8-connected component can now include fake-seed leaves. This is safe
  because filling still additionally requires cavity boundaries to touch the real
  line color, and fake seeds are generated away from cavity boundaries.

d) Placement legality
---------------------
- Real seeds, lines, and attached frames keep the original non-overlap rules.
- A fake seed may be placed only if:
  - it is strictly interior (not on the border),
  - its cell is empty,
  - it has exactly one non-bg 4-neighbor,
  - that sole neighbor is a real line pixel,
  - it has no other non-bg 8-neighbors.
- This keeps the fake seed a clean 1-pixel leaf, avoids touching frame rings, and
  avoids creating ambiguous merged objects.

e) Visual observability
-----------------------
- Across 2-4 ARC examples, a human can now notice:
  - some singleton colored pixels are on the edge and drive the fill,
  - other singleton colored pixels sit inside the grid and do nothing.
- The only consistent separator is border placement, which is directly visible.
- The generator rejects samples where removing the border rule would not change
  behavior, preventing silent regression to the original task.

f) Visual cleanliness
---------------------
- Fake seeds are spaced by the local 8-neighborhood emptiness rule, so they do
  not cluster into noisy blobs.
- Since old disconnected distractor loops/networks are removed, the grid is often
  cleaner than the original D11 while still harder at the seed-recognition step.

g) Color inferability
---------------------
- No hidden color ordering is used by the task rule itself.
- Real and fake seeds both use solid colors as local labels, but the solver rule
  depends only on visible border placement.
- The number of non-bg colors stays modest because frame-color diversity is kept
  near the original range and fake-seed count is capped by remaining palette size.
"""

from collections import Counter
import random
import sys

import numpy as np

SOLVER_UTILS = "/home/lixinyu/snap/new/solvers"
if SOLVER_UTILS not in sys.path:
    sys.path.append(SOLVER_UTILS)

from arc_utils import components_mask, get_bg, neighbors4, neighbors8


def _components_mask(mask: np.ndarray, conn8: bool) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8 if conn8 else 4)]


def _touches_border(pts: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts)


def _on_border(r: int, c: int, h: int, w: int) -> bool:
    return r == 0 or c == 0 or r == h - 1 or c == w - 1


def _singleton_pixels(g: np.ndarray, bg: int) -> list[tuple[int, int, int]]:
    singletons: list[tuple[int, int, int]] = []
    vals, cnts = np.unique(g, return_counts=True)
    for v, cnt in zip(vals, cnts):
        v = int(v)
        if v == bg or int(cnt) != 1:
            continue
        r, c = map(int, np.argwhere(g == v)[0])
        singletons.append((r, c, v))
    return singletons


def _seed_pixels(g: np.ndarray, bg: int, require_border_seed: bool) -> list[tuple[int, int]]:
    h, w = g.shape
    seeds: list[tuple[int, int]] = []
    for r, c, _v in _singleton_pixels(g, bg):
        if require_border_seed and not _on_border(r, c, h, w):
            continue
        seeds.append((r, c))
    if not require_border_seed:
        seeds.sort(key=lambda rc: (_on_border(rc[0], rc[1], h, w), rc[0], rc[1]))
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


def _solve(grid: np.ndarray, *, require_border_seed: bool) -> np.ndarray:
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

    for sr, sc in _seed_pixels(g, bg, require_border_seed=require_border_seed):
        seed_col = int(g[sr, sc])

        neigh_cols = []
        for nr, nc in neighbors4(sr, sc):
            if 0 <= nr < h and 0 <= nc < w:
                v = int(g[nr, nc])
                if v != bg:
                    neigh_cols.append(v)
        if not neigh_cols:
            continue
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

            for r, c in cav:
                if int(out[r, c]) == bg:
                    out[r, c] = seed_col

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Variant rule:
    - Only globally unique single pixels on the grid border are true seeds.
    - Interior singleton pixels are fake seeds and must be ignored.
    - As in the original task, each true seed labels cavities on its traced line
      component.
    """
    return _solve(grid, require_border_seed=True)


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
FRAME_PLACEMENT_PAD = 1
FRAME_ATTACH_TRIES = 40

DIAG_WEIGHT_ALIGNED = 5
DIAG_WEIGHT_OTHER = 1
ORTHO_WEIGHT_ALIGNED = 2
ORTHO_WEIGHT_OTHER = 1
BORDER_AVOIDANCE_PROB = 0.85
BIAS_FLIP_PROB = 0.3

FAKE_SEED_MAX = 5
FAKE_SEED_COUNT_CHOICES = [1, 1, 2, 2, 3, 3, 4, 5]
FAKE_SEED_PLACE_TRIES = 120

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
            corner = dr in (0, H - 1) and dc in (0, W - 1)
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
            v = int(canvas[nr, nc])
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
            if _on_border(nr, nc, h, w) and rng.random() < BORDER_AVOIDANCE_PROB:
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


def _distribute_loop_budget(num_seeds: int, total_loops: int, rng: random.Random) -> list[int]:
    seed_order = list(range(num_seeds))
    rng.shuffle(seed_order)
    per_seed = [0] * num_seeds
    remaining = total_loops
    while remaining > 0 and any(per_seed[i] == 0 for i in range(num_seeds)):
        for i in seed_order:
            if per_seed[i] == 0 and remaining > 0:
                per_seed[i] += 1
                remaining -= 1
    while remaining > 0:
        per_seed[rng.choice(seed_order)] += 1
        remaining -= 1
    return per_seed


def _can_place_fake_seed(canvas: np.ndarray, r: int, c: int, line_col: int) -> bool:
    h, w = canvas.shape
    if not _in_bounds(h, w, r, c) or _on_border(r, c, h, w) or canvas[r, c] != -1:
        return False

    nonbg4 = []
    for nr, nc in neighbors4(r, c):
        if _in_bounds(h, w, nr, nc) and canvas[nr, nc] != -1:
            nonbg4.append((nr, nc, int(canvas[nr, nc])))
    if len(nonbg4) != 1 or nonbg4[0][2] != line_col:
        return False

    only_neighbor = (nonbg4[0][0], nonbg4[0][1])
    for nr, nc in neighbors8(r, c):
        if _in_bounds(h, w, nr, nc) and canvas[nr, nc] != -1 and (nr, nc) != only_neighbor:
            return False
    return True


def _fake_seed_candidates(
    canvas: np.ndarray,
    lines_info: list[tuple[int, tuple[int, int], int, list[tuple[int, int]]]],
) -> list[tuple[int, int]]:
    candidates: set[tuple[int, int]] = set()
    for _seed_col, _seed_rc, line_col, path in lines_info:
        for pr, pc in path:
            for nr, nc in neighbors4(pr, pc):
                if _can_place_fake_seed(canvas, nr, nc, line_col):
                    candidates.add((nr, nc))
    return list(candidates)


def _place_fake_seeds(
    canvas: np.ndarray,
    lines_info: list[tuple[int, tuple[int, int], int, list[tuple[int, int]]]],
    fake_colors: list[int],
    target_count: int,
    rng: random.Random,
) -> list[tuple[int, int, int]]:
    colors = fake_colors[:]
    rng.shuffle(colors)
    placed: list[tuple[int, int, int]] = []
    attempts = 0
    while colors and len(placed) < target_count and attempts < FAKE_SEED_PLACE_TRIES:
        attempts += 1
        cands = _fake_seed_candidates(canvas, lines_info)
        if not cands:
            break
        r, c = rng.choice(cands)
        col = colors.pop()
        canvas[r, c] = col
        placed.append((r, c, col))
    return placed


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

    max_frame_cols = min(FRAME_COLORS_MAX, max(1, len(palette) - 1))
    num_frame_cols = rng.randint(1, max_frame_cols)
    frame_colors_pool = [palette.pop() for _ in range(num_frame_cols)]
    fake_color_pool = palette[:]
    if not fake_color_pool:
        return None

    lines_info: list[tuple[int, tuple[int, int], int, list[tuple[int, int]]]] = []
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
    per_seed = _distribute_loop_budget(num_seeds, total_loops_target, rng)

    for i, (_seed_col, _seed_rc, line_col, path) in enumerate(lines_info):
        attached = 0
        for _ in range(per_seed[i]):
            fc = rng.choice(frame_colors_pool)
            if _try_attach_frame(canvas, path, line_col, fc, rng) is not None:
                attached += 1
        if attached == 0:
            return None

    target_fake = min(rng.choice(FAKE_SEED_COUNT_CHOICES), FAKE_SEED_MAX, len(fake_color_pool))
    if target_fake <= 0:
        return None
    fake_seeds = _place_fake_seeds(canvas, lines_info, fake_color_pool, target_fake, rng)
    if len(fake_seeds) == 0:
        return None

    grid = np.where(canvas == -1, bg, canvas).astype(int)

    vals, cnts = np.unique(grid, return_counts=True)
    if int(vals[np.argmax(cnts)]) != bg:
        return None

    for seed_col, _seed_rc, _line_col, _path in lines_info:
        if int((grid == seed_col).sum()) != 1:
            return None
    for _r, _c, fake_col in fake_seeds:
        if int((grid == fake_col).sum()) != 1:
            return None

    true_out = main(grid)
    if np.array_equal(true_out, grid):
        return None

    relaxed_out = _solve(grid, require_border_seed=False)
    if np.array_equal(relaxed_out, true_out):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError("generate_input: exhausted attempts")
