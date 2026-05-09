# Combo variant of 8b7bacbf (index 24)
# combo_spec: D4/1+D18/0
# Name: 边缘色块种子+边缘中点定位
#
# Design analysis
# ---
# fused_rule:
#   summary:
#     - "seed 从单像素角落点改成 2-3 像素的边缘短段。"
#     - "seed 不再靠近角落，而是放在某一条边的中点带。"
#     - "seed 所在的边编码追踪方向：上边->向下， 下边->向上， 左边->向右， 右边->向左。"
#   concrete_shape:
#     - "top/bottom seed 是水平短段；left/right seed 是竖直短段。"
#     - "若该边长度为偶数，seed 长度固定为 2，正好跨过几何中心。"
#     - "若该边长度为奇数，seed 长度固定为 3，正好以中心格为中点。"
#   behavior_trace:
#     original_flow:
#       - "找全局只出现 1 次的边缘像素。"
#       - "从该单点四邻域估计 line_color。"
#       - "从 seed 所在的非背景连通体找接触的闭环并填腔体。"
#     fused_flow:
#       - "找全局只出现 2-3 次、且恰好组成边中央短段的颜色块。"
#       - "由 seed 位于哪一条边推断 inward 方向。"
#       - "只查看 seed 朝 inward 方向的一排相邻格来确定 line_color 与入口。"
#       - "再沿 8 连通结构追踪并填充其接触的闭环腔体。"
#     local_example:
#       - "例如 11x12 网格中，top seed 只能是 [(0,5),(0,6)]；solver 因此知道线必须从第 1 行往下接出。"
#       - "例如 13x13 网格中，right seed 只能是 [(5,12),(6,12),(7,12)]；solver 因此只看这些像素左侧的非背景邻居。"
#   complexity_calibration:
#     - "中心 seed 只允许四条边这 4 类位置，方向编码类别少且样例间可重复。"
#     - "长度只保留 2/3 两档，视觉上明显区别于原题单点 seed，但不会引入过多自由度。"
#     - "线条仍沿原题的单像素 8 连通蜿蜒路径，闭环与填色逻辑保持一致。"
#   implicit_assumption_audit_main:
#     broken_old_assumptions:
#       - "_seed_pixels() 依赖颜色计数为 1；融合后必须改成识别 2-3 像素边中央色块。"
#       - "从单个 seed 像素四邻域取 line_color 的假设不再成立；现在要聚合整段 seed 并按 inward 方向过滤。"
#       - "原题默认 seed 位置本身不携带信息；融合后 seed 所在边必须参与推理。"
#     new_solver_assumptions:
#       - "每个 seed color 只出现一次，且全部像素构成一个 4 连通短段。"
#       - "短段的所有像素都在同一条边上，并且精确落在该边中点模板。"
#       - "至少一个 seed 像素在 inward 方向上 4 邻接到对应 line_color。"
#     generator_guarantees:
#       - "generator 只生成上述四类中点模板。"
#       - "line 起点始终放在 inward 一侧，且 seed 周围其余 4 邻不引入其他非背景颜色。"
#       - "frame/distractor 的旧约束继续保留，避免 cavity 判定退化。"
#   placement_legality:
#     - "seed、line、frame 全部写入前都检查空位，禁止重叠。"
#     - "line 起点固定在 seed 内侧一格，且不是边界格。"
#     - "attached frame 仍要求 ring/cavity 为空、cavity 不贴 line、ring 至少 8 邻接一处 line。"
#   visual_observability:
#     - "解题者从少量样例可直接看到：角落 seed 消失，改为边中央短段；线条总是从短段朝网格内部发出。"
#     - "边别与 inward 方向是一一对应的，不依赖颜色编号或不可视排序。"
#   cleanliness:
#     - "frame 仍然附着在线的外侧，且生成时尽量避开 seed 起始几步，避免中点提示被遮挡。"
#   nondegeneracy_check_plan:
#     - "生成后验证 solver 检出的 seed 必须全部满足 midpoint 模板。"
#     - "统计样本中四条边的使用分布与 2/3 长度分布，确认中点定位和边缘色块两个控制变量都实际出现。"

import random
from collections import Counter

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4, neighbors8


def _components_mask(mask: np.ndarray, conn8: bool) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8 if conn8 else 4)]


def _touches_border(pts: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts)


def _is_border_cell(r: int, c: int, h: int, w: int) -> bool:
    return r == 0 or c == 0 or r == h - 1 or c == w - 1


def _seed_side(pts: list[tuple[int, int]], h: int, w: int) -> str | None:
    if all(r == 0 for r, _ in pts):
        return "T"
    if all(r == h - 1 for r, _ in pts):
        return "B"
    if all(c == 0 for _, c in pts):
        return "L"
    if all(c == w - 1 for _, c in pts):
        return "R"
    return None


def _midpoint_seed_template(side: str, h: int, w: int) -> list[tuple[int, int]]:
    if side in ("T", "B"):
        if w % 2 == 0:
            cols = [w // 2 - 1, w // 2]
        else:
            mid = w // 2
            cols = [mid - 1, mid, mid + 1]
        row = 0 if side == "T" else h - 1
        return [(row, c) for c in cols]
    if h % 2 == 0:
        rows = [h // 2 - 1, h // 2]
    else:
        mid = h // 2
        rows = [mid - 1, mid, mid + 1]
    col = 0 if side == "L" else w - 1
    return [(r, col) for r in rows]


def _forward_delta(side: str) -> tuple[int, int]:
    return {
        "T": (1, 0),
        "B": (-1, 0),
        "L": (0, 1),
        "R": (0, -1),
    }[side]


def _seed_records(g: np.ndarray, bg: int) -> list[tuple[list[tuple[int, int]], str, tuple[int, int]]]:
    h, w = g.shape
    vals, cnts = np.unique(g, return_counts=True)
    seeds: list[tuple[list[tuple[int, int]], str, tuple[int, int]]] = []
    for v, cnt in zip(vals, cnts):
        v = int(v)
        cnt = int(cnt)
        if v == bg or cnt not in (2, 3):
            continue
        comps = _components_mask(g == v, conn8=False)
        if len(comps) != 1:
            continue
        pts = sorted(comps[0])
        if len(pts) != cnt or not all(_is_border_cell(r, c, h, w) for r, c in pts):
            continue
        side = _seed_side(pts, h, w)
        if side is None:
            continue
        if pts != sorted(_midpoint_seed_template(side, h, w)):
            continue
        seeds.append((pts, side, _forward_delta(side)))
    return seeds


def _forward_entries(
    g: np.ndarray,
    seed_pts: list[tuple[int, int]],
    delta: tuple[int, int],
    seed_col: int,
    bg: int,
) -> tuple[int, list[tuple[int, int]]]:
    h, w = g.shape
    dr, dc = delta
    front_positions: list[tuple[int, int]] = []
    front_colors: list[int] = []
    for r, c in seed_pts:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            v = int(g[nr, nc])
            if v != bg and v != seed_col:
                front_positions.append((nr, nc))
                front_colors.append(v)
    assert front_colors, "midpoint seed must expose its line on the inward side"
    line_col = Counter(front_colors).most_common(1)[0][0]
    starts = sorted({pos for pos in front_positions if int(g[pos]) == line_col})
    assert starts, "line entry detection failed"
    return line_col, starts


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
    Fill frame cavities using unique border blocks placed at edge midpoints.

    Input structure:
    - bg is the most frequent color.
    - Each actionable seed is a unique-color 2-3 pixel block sitting exactly on the midpoint of one border side.
    - A non-background 8-connected drawing extends inward from that seed and touches one or more single-color frame loops.

    Rule:
    - Determine which side each seed block occupies; that side tells you the inward tracing direction toward the opposite edge.
    - Read the line color only from the seed's inward-facing neighbors, then trace the whole connected drawing.
    - Every enclosed bg cavity whose single-color boundary is touched by that traced drawing gets flood-filled with the seed color.

    Key insight:
    - The seed is no longer a lone border pixel: it is a centered border segment whose position carries directional meaning.
    - The side of the grid disambiguates where the line begins, while the rest of the task remains "follow that drawing and fill the loops it reaches".
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

    for seed_pts, side, delta in _seed_records(g, bg):
        del side  # direction has already been materialized in delta
        seed_col = int(g[seed_pts[0]])
        line_col, starts = _forward_entries(g, seed_pts, delta, seed_col, bg)
        traced = _nonbg_component8_from(g, starts[0], bg)

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


# --------------- generator (attempt_1_distractor_network) ---------------

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
FRAME_ATTACH_SKIP = 2

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

GEN_RETRIES = 400
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
            corner = (dr in (0, H - 1)) and (dc in (0, W - 1))
            if on_bdr and not corner:
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
            v = canvas[nr, nc]
            if v != -1 and v != line_col:
                return False

    for rr, cc in ring:
        for nr, nc in neighbors8(rr, cc):
            if _in_bounds(h, w, nr, nc) and canvas[nr, nc] == line_col:
                return True
    return False


def _default_biases(start: tuple[int, int], h: int, w: int) -> tuple[int, int]:
    return (1 if start[0] < h // 2 else -1), (1 if start[1] < w // 2 else -1)


def _grow_line(
    grid: np.ndarray,
    start: tuple[int, int],
    line_col: int,
    target_len: int,
    rng: random.Random,
    dr_bias: int | None = None,
    dc_bias: int | None = None,
) -> list[tuple[int, int]]:
    h, w = grid.shape
    path = [start]
    if dr_bias is None or dc_bias is None:
        dr_bias, dc_bias = _default_biases(start, h, w)
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
    anchors = line_path[FRAME_ATTACH_SKIP:] if len(line_path) > FRAME_ATTACH_SKIP else line_path
    for _ in range(max_tries):
        ch = rng.choice(CAVITY_SIZE_WEIGHTS)
        cw = rng.choice(CAVITY_SIZE_WEIGHTS)
        lp = rng.choice(anchors)
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


def _midpoint_seed_block(side: str, h: int, w: int, rng: random.Random) -> tuple[list[tuple[int, int]], tuple[int, int], int, int]:
    seed = _midpoint_seed_template(side, h, w)
    dr, dc = _forward_delta(side)
    if side in ("T", "B"):
        attach_c = rng.choice([c for _, c in seed])
        first = (1, attach_c) if side == "T" else (h - 2, attach_c)
        side_bias = rng.choice([-1, 1])
        return seed, first, dr, side_bias
    attach_r = rng.choice([r for r, _ in seed])
    first = (attach_r, 1) if side == "L" else (attach_r, w - 2)
    side_bias = rng.choice([-1, 1])
    return seed, first, side_bias, dc


def _choose_seed_sides(num_seeds: int, rng: random.Random) -> list[str]:
    if num_seeds == 1:
        return [rng.choice(["T", "B", "L", "R"])]
    sides = [rng.choice(["T", "B"]), rng.choice(["L", "R"])]
    rng.shuffle(sides)
    return sides


def _distribute_loop_budget(num_seeds: int, num_attached_target: int, rng: random.Random) -> list[int]:
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


def _fusion_metrics(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    seeds = _seed_records(g, bg)
    sides = [side for _, side, _ in seeds]
    lengths = [len(pts) for pts, _, _ in seeds]
    inward_neighbor_counts = []
    for pts, _, delta in seeds:
        seed_col = int(g[pts[0]])
        _, starts = _forward_entries(g, pts, delta, seed_col, bg)
        inward_neighbor_counts.append(len(starts))
    return {
        "num_seeds": len(seeds),
        "sides": sides,
        "lengths": lengths,
        "inward_entries": inward_neighbor_counts,
    }


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
    sides_used = _choose_seed_sides(num_seeds, rng)

    num_frame_cols = rng.randint(1, min(FRAME_COLORS_MAX, len(palette)))
    frame_colors_pool = [palette.pop() for _ in range(num_frame_cols)]

    lines_info = []
    for i in range(num_seeds):
        seed_pts, first_rc, dr_bias, dc_bias = _midpoint_seed_block(sides_used[i], h, w, rng)
        if any(canvas[r, c] != -1 for r, c in seed_pts) or canvas[first_rc] != -1:
            return None
        for r, c in seed_pts:
            canvas[r, c] = seed_colors[i]
        canvas[first_rc] = line_colors[i]
        target_len = rng.randint(LINE_LEN_RANGE[0], min(LINE_LEN_RANGE[1], h + w))
        path = _grow_line(canvas, first_rc, line_colors[i], target_len, rng, dr_bias, dc_bias)
        if len(path) < MIN_LINE_LEN:
            return None
        lines_info.append((seed_colors[i], seed_pts, line_colors[i], path, sides_used[i]))

    total_loops_target = rng.randint(num_seeds, TOTAL_LOOPS_MAX)
    num_distractors_target = rng.randint(0, max(0, total_loops_target - num_seeds))
    num_attached_target = total_loops_target - num_distractors_target
    per_seed = _distribute_loop_budget(num_seeds, num_attached_target, rng)

    for i, (_, _, line_col, path, _) in enumerate(lines_info):
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

    seed_records = _seed_records(grid, bg)
    if len(seed_records) != num_seeds:
        return None
    seen_seed_cols = {int(grid[pts[0]]) for pts, _, _ in seed_records}
    if seen_seed_cols != set(seed_colors):
        return None

    for seed_col, seed_pts, _, _, side in lines_info:
        if int((grid == seed_col).sum()) != len(seed_pts):
            return None
        pts = sorted(seed_pts)
        if pts != sorted(_midpoint_seed_template(side, h, w)):
            return None
        _, _, delta = next(rec for rec in seed_records if int(grid[rec[0][0]]) == seed_col)
        line_col, starts = _forward_entries(grid, pts, delta, seed_col, bg)
        if line_col not in line_colors or len(starts) != 1:
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
