# Variant of 8b7bacbf
# Dimension: D6 — 变换操作类型 (transform)
# Change: 腔体内部flood-fill为种子颜色——cavity.interior.color = seed.color... → 框架环重涂色
#
# Design analysis
# a) 具体化与行为追踪
#    原题输入里，每个 seed 在边界上，并且 8 连到一条细线；细线再 8 连到若干单色闭环框架。
#    原题输出把这些闭环围住的 bg 腔体整块填成 seed 色。这个变体不再改腔体内部，而是保留
#    腔体为空白，仅把与该 seed 路径相连的“单色环边界像素”整体改涂成 seed 色。
#    例子：若一个 2 色线条连到一个 3 色框架，原题会把框架内部的 0 区域涂成 7；变体则保留
#    内部 0 不动，把框架那一圈 3 改成 7。变化发生在 solver 的最后写回阶段；输入构造仍需
#    生成“seed -> 线 -> 闭环框架”的关系，只是输出目标从 cavity 切换成 ring。
# b) 复杂度校准
#    不增加新的匹配机制，仍沿用“4 连通背景腔体 + 8 连通非背景追踪 + 单色边界筛选”。
#    这样复杂度与原题同级，但视觉效果显著不同：原题是空洞被填满，变体是边框变色、内部留空。
#    为避免样例太弱，生成器继续放未连接的干扰环，使“哪些环会变色”有对照。
# c) 隐式假设审查
#    - `_seed_pixels`: 依赖 seed 颜色全局唯一且在边界上；变体仍成立。
#    - `_nonbg_component8_from`: 依赖线和框架都属于同一 8 连通非背景组件；变体仍成立。
#    - 背景腔体枚举: 依赖 cavity 是 4 连通 bg 区域、边界全为非 bg；变体仍成立。
#    - `len(bcols) == 1`: 原题用它保证 cavity 的边界是单色框架，避免把混合结构误当目标；
#      变体更需要它，因为现在真正要重涂的是这个边界本身。
#    - 原题最后 `for r,c in cav: out[r,c]=seed_col` 依赖“目标是腔体内部”。变体下这行会错，
#      必须改成对 `boundary` 上色，否则会退化回原规则。
#    - 新风险是两个 seed 若通过 8 邻接意外连成同一非背景组件，同一框架可能被两个 seed 争抢。
#      原题里这也含糊，但只填内部时不容易肉眼察觉；变体会直接出现边框被后写 seed 覆盖。
#      因此生成器新增“不同 seed 的非背景组件必须彼此 8 断开”的校验。
# d) 放置合法性
#    框架仍使用原来的 rounded-rect ring，环与腔体都占空白格；腔体不能 8 邻接到 line 色，
#    环不能 8 邻接外来颜色，且至少有一个环像素 8 邻接 line 色。这样 solver 的 cavity 检测、
#    单色边界判断和“该环属于哪条 seed 路径”都保持可解。
# e) 视觉可观察性
#    人类在 2-4 个样例里能直接看到：与 seed 连通的环边框改成 seed 色，但环内空白不变；
#    没连通的环维持原色。这个规律完全基于可见的连接关系和颜色变化，不用任何数值排序。
# f) 可视化整洁度
#    仍保留 1px 线、单层框架、空白腔体，避免多层叠色。组件断开校验还能减少不同 seed 网络
#    彼此贴近造成的视觉混杂。
# g) 颜色变化可推断性
#    seed 色在整图唯一出现，重涂后的环直接继承该色；frame 色数量仍限制在少数几种。
#    输出中的颜色变化由“是否通过线连接到某个 seed”决定，结构线索充足。

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


def _single_color_cavities(g: np.ndarray, bg: int) -> list[tuple[set[tuple[int, int]], set[tuple[int, int]], int]]:
    """Return enclosed bg regions together with their non-bg single-color boundaries."""
    h, w = g.shape
    bg_comps = _components_mask(g == bg, conn8=False)
    corners = {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}

    regions: list[tuple[set[tuple[int, int]], set[tuple[int, int]]]] = []
    outside_idxs = set()
    for i, pts in enumerate(bg_comps):
        region = set(pts)
        boundary = set()
        for r, c in region:
            for nr, nc in neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in region and int(g[nr, nc]) != bg:
                    boundary.add((nr, nc))
        regions.append((region, boundary))
        if any(p in region for p in corners if int(g[p]) == bg):
            outside_idxs.add(i)

    cavities = []
    for i, (region, boundary) in enumerate(regions):
        if i in outside_idxs or not boundary:
            continue
        bcols = {int(g[p]) for p in boundary}
        if len(bcols) != 1:
            continue
        cavities.append((region, boundary, next(iter(bcols))))
    return cavities


def _line_color_at_seed(g: np.ndarray, sr: int, sc: int, bg: int) -> int:
    neigh_cols = []
    h, w = g.shape
    for nr, nc in neighbors4(sr, sc):
        if 0 <= nr < h and 0 <= nc < w:
            v = int(g[nr, nc])
            if v != bg:
                neigh_cols.append(v)
    assert neigh_cols, "seed pixel must have at least one non-bg neighbor"
    return int(Counter(neigh_cols).most_common(1)[0][0])


def _recolorable_ring_count(g: np.ndarray) -> int:
    """Diagnostic metric: how many cavity-boundary rings the variant will recolor."""
    g = np.asarray(g, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    cavities = _single_color_cavities(g, bg)
    count = 0

    for sr, sc in _seed_pixels(g, bg):
        line_col = _line_color_at_seed(g, sr, sc, bg)
        traced = _nonbg_component8_from(g, (sr, sc), bg)
        for _cav, boundary, _frame_col in cavities:
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
            if ok:
                count += 1
    return count


def _seed_components_are_disjoint(g: np.ndarray) -> bool:
    """Generator-side guard against two seeds sharing one 8-connected drawing."""
    bg = get_bg(g)
    comps = [_nonbg_component8_from(g, seed, bg) for seed in _seed_pixels(g, bg)]
    for i in range(len(comps)):
        for j in range(i + 1, len(comps)):
            if comps[i] & comps[j]:
                return False
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor frame loops along a line seeded by unique border pixels.

    Input structure:
    - bg = most common color.
    - Colored lines/drawings on the grid.
    - Unique single-pixel colors on the grid border mark seed endpoints of each line.

    Rule:
    - Each border seed pixel connects (8-connected) to a line drawing.
    - Find closed bg cavities whose single-color boundary touches that line network.
    - Do not fill the cavity interior.
    - Instead recolor the boundary ring itself to the seed pixel's color.

    Key insight: The seed is still a per-line label, but the transformed target is the ring,
    not the interior of the enclosed cavity.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    out = g.copy()

    cavities = _single_color_cavities(g, bg)

    for sr, sc in _seed_pixels(g, bg):
        seed_col = int(g[sr, sc])
        line_col = _line_color_at_seed(g, sr, sc, bg)

        # Trace the connected drawing reachable from the seed (8-connected, non-background).
        traced = _nonbg_component8_from(g, (sr, sc), bg)

        for _cav, boundary, _frame_col in cavities:
            boundary_in_traced = boundary & traced
            if not boundary_in_traced:
                continue

            # Require that the traced boundary is adjacent (8-neigh) to at least one pixel of the line color.
            # This keeps the frame assignment tied to the seed's own path rather than arbitrary contact elsewhere.
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

            for r, c in boundary:
                out[r, c] = seed_col

    return out


# --------------- generator (attempt_1_distractor_network) ---------------

# ========================== CFG ==========================
# Canvas / grid
GRID_H_RANGE = (8, 30)
GRID_W_RANGE = (8, 30)
BG_COLORS = [0, 5]

# Seeds
SEED_COUNT_WEIGHTS = [1, 2, 2]
MIN_LINE_LEN = 4
LINE_LEN_RANGE = (8, 35)

# Frame loops
FRAME_COLORS_MAX = 4
TOTAL_LOOPS_MAX = 9
CAVITY_SIZE_WEIGHTS = [2, 2, 2, 2, 2, 3]
DISTRACTOR_CAVITY_WEIGHTS = [2, 2, 2, 3]
FRAME_PLACEMENT_PAD = 1
FRAME_ATTACH_TRIES = 40
DISTRACTOR_FRAME_TRIES = 40

# Line growth
DIAG_WEIGHT_ALIGNED = 5
DIAG_WEIGHT_OTHER = 1
ORTHO_WEIGHT_ALIGNED = 2
ORTHO_WEIGHT_OTHER = 1
BORDER_AVOIDANCE_PROB = 0.85
BIAS_FLIP_PROB = 0.3

# Distractor network
DISTRACTOR_NETWORK_PROB = 0.6
DISTRACTOR_NETWORK_TRIES = 50
DISTRACTOR_LINE_LEN = (5, 15)
DISTRACTOR_SOLO_MAX = 2
DISTRACTOR_NETWORK_FRAMES = (1, 3)

# Generation
GEN_RETRIES = 300
# =========================================================

DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
ORTHO_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _frame_ring_coords(r0: int, c0: int, ch: int, cw: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Return (ring_coords, cavity_coords) for a rounded-rect frame."""
    fh, fw = ch + 2, cw + 2
    ring = []
    for dr in range(fh):
        for dc in range(fw):
            on_bdr = dr == 0 or dr == fh - 1 or dc == 0 or dc == fw - 1
            corner = (dr in (0, fh - 1)) and (dc in (0, fw - 1))
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
    """Check that ring+cavity can be placed on canvas without conflicts."""
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
    """Compute offsets around a frame bbox that lie outside it (the halo)."""
    return [(dr, dc) for dr in range(-1, fh + 1) for dc in range(-1, fw + 1) if not (0 <= dr < fh and 0 <= dc < fw)]


def _try_attach_frame(
    canvas: np.ndarray,
    line_path: list[tuple[int, int]],
    line_col: int,
    frame_col: int,
    rng: random.Random,
    max_tries: int = FRAME_ATTACH_TRIES,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    """Try to place a frame loop 8-adjacent to some line pixel."""
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


def _try_attach_distractor_frame(
    canvas: np.ndarray,
    frame_col: int,
    rng: random.Random,
    max_tries: int = DISTRACTOR_FRAME_TRIES,
) -> bool:
    """Place a loop in empty space, not touching any non-bg pixel."""
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


def _try_place_distractor_network(
    canvas: np.ndarray,
    line_col: int,
    frame_colors_pool: list[int],
    rng: random.Random,
) -> bool:
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


def _corner_config(corner_kind: str, h: int, w: int, rng: random.Random) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return (seed_rc, first_line_rc) for corner_kind in {TL,TR,BL,BR}."""
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
    """Distribute attached-loop budget across seeds, guaranteeing >= 1 each."""
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

    for i, (_seed_col, _seed_rc, line_col, path) in enumerate(lines_info):
        attached = 0
        for _ in range(per_seed[i]):
            fc = rng.choice(frame_colors_pool)
            if _try_attach_frame(canvas, path, line_col, fc, rng) is not None:
                attached += 1
        if attached == 0:
            return None

    placed_distractor = False
    if num_distractors_target > 0:
        if rng.random() < DISTRACTOR_NETWORK_PROB:
            placed_distractor = _try_place_distractor_network(canvas, rng.choice(line_colors), frame_colors_pool, rng)
        if not placed_distractor:
            for _ in range(max(1, min(num_distractors_target, DISTRACTOR_SOLO_MAX))):
                if _try_attach_distractor_frame(canvas, rng.choice(frame_colors_pool), rng):
                    placed_distractor = True
        if not placed_distractor:
            return None

    grid = np.where(canvas == -1, bg, canvas).astype(int)

    vals, cnts = np.unique(grid, return_counts=True)
    if int(vals[np.argmax(cnts)]) != bg:
        return None

    for seed_col, _seed_rc, _line_col, _path in lines_info:
        if int((grid == seed_col).sum()) != 1:
            return None

    if not _seed_components_are_disjoint(grid):
        return None

    if _recolorable_ring_count(grid) == 0:
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
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted attempts")
