# Variant of 8b7bacbf
# Dimension: D13 — 线路颜色与种子颜色的关系 (meta)
# Change: 每个种子有独立的线路颜色，种子色唯一出现且与线路颜色不同... → 线路颜色与种子颜色相同
#
# Design analysis
# ---------------
# source_task: None
#
# concretization_and_trace:
#   input:
#     - 每个真实种子仍然放在网格角点, 但它的颜色不再是“只出现一次”的标签色。
#     - 种子角点与其线路共享同一种颜色, 线路从角点的对角内侧像素起步, 以 8 连通方式向内延展。
#     - 线路再 8 邻接若干单色框环; 框环内部是背景腔体。
#   output:
#     - 与原题一样, 只填充那些边界属于被某个角点 seed 所连接 drawing 的腔体。
#     - 填充值就是该角点 seed 的颜色, 也就是该线路自身的颜色。
#   local_trace_example:
#     - 原逻辑: 角点有单像素颜色 7, 相邻线路是颜色 1。`_seed_pixels()` 通过“7 只出现一次”找到 seed,
#       再从 seed 的 4 邻推断 line_color=1, 最终把接到这条线路的腔体填成 7。
#     - 变体逻辑: 角点是 7, 线路也全是 7。此时颜色 7 会在整条线路上多次出现, 原 `_seed_pixels()`
#       会完全漏掉这个 seed, 原 solver 通常不会填任何东西。新逻辑直接从四个角读取非背景像素作为 seed,
#       并令 line_color=seed_color=7, 之后仍按“腔体边界是否接到 traced drawing 且贴近线路”来决定填充。
#     - 差异:
#       1. seed 识别从“颜色计数”改成“边缘/角点位置”。
#       2. line_color 不再从邻居统计推断, 而是直接等于 seed_color。
#
# complexity_calibration:
#   - 仍保留 1-2 个 seed、1-9 个框环、1-4 个框颜色, 不引入新的对象类型。
#   - 每个 seed 颜色仍与其他 seed 颜色不同, 避免把 D13 以外的“跨 seed 颜色匹配”问题混进来。
#   - 线路与种子同色, 且至少包含 1 个非 seed 像素, 保证每个 seed 颜色在全图出现次数 > 1。
#   - 角点 seed 是固定少量类别, 人类可以在 2-4 个样例中稳定观察到“同色角点+同色线路”的变化。
#
# implicit_assumption_review_of_original_main:
#   - `_seed_pixels()`:
#       旧前提: seed 颜色全图只出现一次。
#       变体: 不成立, 因为 seed 颜色会沿线路重复出现。
#       处理: 改成 `_corner_seed_pixels()` 只看四角的非背景像素。
#   - `seed_col = int(g[sr, sc])`:
#       旧前提: seed 自身颜色就是填充值。
#       变体: 仍成立。
#   - 统计 `neigh_cols` 并用 `Counter(...).most_common(1)` 求 `line_col`:
#       旧前提: 线路颜色与 seed 颜色不同, 且可从邻居反推。
#       变体: 不成立也没必要; 直接令 `line_col = seed_col`。
#   - `_nonbg_component8_from(g, (sr, sc), bg)`:
#       旧前提: 从 seed 出发的 8 连通 non-bg drawing 正好覆盖该 seed 的线路和其附着框环。
#       变体: 仍成立, 因为 seed 与线路仍是同一 non-bg 组件。
#   - `len(bcols) != 1` 的单色边界判定:
#       与 D13 无关, 继续保留。
#   - “traced boundary 至少 8 邻接到一个 line_color 像素”:
#       旧前提: 需要确保被填的是贴着线路的腔体, 不是仅通过其它结构偶然连上的区域。
#       变体: 仍成立; 此时 line_color 就是 seed/线路的共享颜色。
#
# placement_legality:
#   - 只有真实 seed 会占据角点; 其它结构保持离边至少 1 格, 不会制造伪 seed。
#   - 线路从角点的对角内侧起步, 与 seed 保持 8 连通但尽量不沿边扩展, 减少边缘杂讯。
#   - 框环与腔体都必须落在空白处; 腔体不能贴线; 框环不能 8 邻接外来颜色; 至少一个框环贴到对应线路。
#   - generator 在完成后调用新 `main()` 作为 oracle, 并额外检查每个 seed 颜色出现次数 > 1。
#
# visual_observability:
#   - 人类在输入里能直接看到: 角上的彩色像素与深入网格的线路是同色的, 不再有“角点是独立标签色”的现象。
#   - 输出则把相连框环的内部也涂成这条线路本身的颜色。这个视觉差异在 2-3 个样例中就能稳定观察到。
#
# visual_tidiness:
#   - 角点 seed 固定在四角, 线路从内侧起步, 框环保持离边 1 格, 可减少角落处多层对象重叠。
#   - 仍沿用原来的细线、单色框环和空腔体设计, 视觉元素层次没有增加。
#
# color_inference:
#   - 颜色总数仍受控: seed/line 颜色 1-2 种, frame 颜色 1-4 种。
#   - 颜色变化规则只依赖清晰的结构线索: “哪一个角点发出的同色连通 drawing 触到该框环”。
#   - 不依赖任何颜色 id 顺序或数值语义。
#
# post_edit_main_assumptions:
#   - solver 新假设 1: 非背景角点就是 seed。generator 通过角点专用 seed 放置和边缘留白保证这一点。
#   - solver 新假设 2: 每个 seed 至少连接一条同色线路。generator 通过对角 first pixel 和最短长度检查保证。
#   - solver 新假设 3: 被填腔体的边界会在 traced drawing 中, 且其边界附近能看到 seed 同色线路像素。
#     `_try_attach_frame()` 的贴线约束保证这一点。

import random

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4, neighbors8


def _components_mask(mask: np.ndarray, conn8: bool) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8 if conn8 else 4)]


def _touches_border(pts: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts)


def _corner_seed_pixels(g: np.ndarray, bg: int) -> list[tuple[int, int]]:
    h, w = g.shape
    corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
    return [(r, c) for r, c in corners if int(g[r, c]) != bg]


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
    Fill enclosed regions along a line seeded by border-corner pixels.

    Variant D13:
    - seed and its line share the same color
    - seeds are identified by border position (corners), not by global color uniqueness
    """
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

    for sr, sc in _corner_seed_pixels(g, bg):
        seed_col = int(g[sr, sc])
        line_col = seed_col
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


# --------------- generator (attempt_1_distractor_network, D13 variant) ---------------

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
    """Return (ring_coords, cavity_coords) for a rounded-rect frame."""
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
            v = canvas[nr, nc]
            if v != -1 and v != line_col:
                return False

    for rr, cc in ring:
        for nr, nc in neighbors8(rr, cc):
            if _in_bounds(h, w, nr, nc) and canvas[nr, nc] == line_col:
                return True
    return False


def _grow_line(grid, start, line_col, target_len, rng):
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


def _halo_offsets(fh, fw):
    """Compute offsets around a frame bbox that lie outside it (the halo)."""
    return [(dr, dc) for dr in range(-1, fh + 1) for dc in range(-1, fw + 1) if not (0 <= dr < fh and 0 <= dc < fw)]


def _try_attach_frame(canvas, line_path, line_col, frame_col, rng, max_tries=FRAME_ATTACH_TRIES):
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


def _try_attach_distractor_frame(canvas, frame_col, rng, max_tries=DISTRACTOR_FRAME_TRIES):
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


def _corner_config(corner_kind, h, w, rng):
    """Return (seed_rc, first_line_rc) for corner_kind in {TL,TR,BL,BR}."""
    if corner_kind == "TL":
        seed = (0, 0)
        first = (1, 1)
    elif corner_kind == "TR":
        seed = (0, w - 1)
        first = (1, w - 2)
    elif corner_kind == "BL":
        seed = (h - 1, 0)
        first = (h - 2, 1)
    elif corner_kind == "BR":
        seed = (h - 1, w - 1)
        first = (h - 2, w - 2)
    else:
        raise ValueError(corner_kind)
    return seed, first


def _distribute_loop_budget(num_seeds, num_attached_target, rng):
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
    if num_seeds > len(palette):
        return None

    canvas = -np.ones((h, w), dtype=int)

    seed_colors = [palette.pop() for _ in range(num_seeds)]
    line_colors = list(seed_colors)

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

    for i, (_, _, line_col, path) in enumerate(lines_info):
        attached = 0
        attach_path = path[1:] if len(path) > 1 else path
        for _ in range(per_seed[i]):
            fc = rng.choice(frame_colors_pool)
            if _try_attach_frame(canvas, attach_path, line_col, fc, rng) is not None:
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

    for seed_col, seed_rc, _, _ in lines_info:
        if int((grid == seed_col).sum()) <= 1:
            return None
        if int(grid[seed_rc]) != seed_col:
            return None

    if len(_corner_seed_pixels(grid, bg)) != num_seeds:
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
