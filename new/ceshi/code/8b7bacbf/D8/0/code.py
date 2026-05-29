# Variant of 8b7bacbf
# Dimension: D8 — 多步处理顺序 (multi_step)
# Change: 三步流程：(1)识别边缘种子像素→(2)追踪每个种子连接的线路→(3)填充沿线的闭合腔体... → 迭代传播填充

"""
design_analysis:
  concretization:
    original_objects:
      - 边缘唯一色种子像素
      - 与种子相连的 8 连通单像素线
      - 由单色框围出的背景腔体
    variant_objects:
      - 保持同样的种子/线路/框结构
      - 但至少有一串“共享墙体”的相邻腔体链：
        第一腔体的外框贴着线；
        第二腔体不贴线，只与第一腔体共用一段框墙；
        第三腔体再与第二腔体共墙，依此类推
    pixel_level_behavior:
      - 第 1 轮：只有边界框像素 8 邻接当前线前沿的腔体被填色
      - 第 2+ 轮：上一轮刚被填的腔体区域加入前沿；与这些已填区域接墙/接边的下一个腔体再被填色
      - 终止：没有新的腔体与当前前沿接触
    trace_example:
      - 原题：若 A/B/C 三个框都在同一非背景网络里，并且都接到线的同一条框链，solver 会一轮全填
      - 变体：只让 A 直接贴线，B 只贴 A 的共享墙，C 只贴 B 的共享墙
      - 结果：round1 填 A；round2 填 B；round3 填 C；若某个 D 从未贴到这条传播前沿，则保持空腔
  complexity_calibration:
    - 不引入新的颜色语义；传播只由可见的几何接触决定
    - 链长限制在 2-4，避免“只有一步”而退化，也避免链过深导致拥挤
    - 每个样例只需要少量重复结构：同色框的共墙链足够让人观察到扩散规律
  implicit_assumptions_review:
    original_solver_dependencies:
      - "_nonbg_component8_from(seed)" 假设“属于该线网络”就应立即填充；变体中这个假设失效
      - "boundary & traced" 在原题里足够；变体里它只能判定“属于该网络”，不能判定“本轮是否可激活”
      - 原题最终直接写入 out，不关心顺序；变体必须显式维护 frontier 并迭代到稳定
      - 原题用 line_color 仅过滤“真沿线”接触；变体里仍保留 line 的识别，但传播前沿需要从 line 扩展到已填腔体
    new_solver_assumptions:
      - 每个种子相邻的非背景主色就是其 line_color
      - 需要传播的后继腔体与前驱腔体之间存在可见的边/墙接触，因此 boundary 对 active frontier 的 8 邻接可检测到
      - 不同种子网络彼此分离；generator 负责保证
  placement_legality:
    - 生成器只放置轴对齐矩形框，框角仍为空，保持原题视觉风格
    - 链式相邻腔体通过共享一整段框墙连接，腔体内部互不重叠
    - 只有链首框允许贴线；后续框若再次贴线则判失败，避免退化成原题一步填充
    - 干扰框只放在远离所有线和链的位置，避免意外传播
  visual_observability:
    - 输入里可以直接看到“第一框碰线、第二框只碰第一框、第三框只碰第二框”的层次
    - 输出里会出现沿链向外扩散的同色填充，而不是所有相关框同时上色
  cleanliness:
    - 放置时要求链整体周围留空，避免多个网络缠在一起
    - 干扰项只用孤立单框，不再引入额外的无种子线路网络，减少视觉噪声
  color_reasoning:
    - 颜色规则不变：种子色就是最终填充色
    - 框色不承载额外编码；同链可复用同一框色，仅服务于“看得出它们是一串相连框”
"""

from __future__ import annotations

import sys
import random
from collections import Counter

import numpy as np

sys.path.append("/home/lixinyu/snap/new/solvers")
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


def _same_color_component8_from(g: np.ndarray, starts: list[tuple[int, int]], color: int) -> set[tuple[int, int]]:
    h, w = g.shape
    vis = {p for p in starts if 0 <= p[0] < h and 0 <= p[1] < w and int(g[p]) == color}
    st = list(vis)
    while st:
        r, c = st.pop()
        for nr, nc in neighbors8(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in vis and int(g[nr, nc]) == color:
                vis.add((nr, nc))
                st.append((nr, nc))
    return vis


def _frame_ring_coords(r0: int, c0: int, ch: int, cw: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    h, w = ch + 2, cw + 2
    ring = []
    for dr in range(h):
        for dc in range(w):
            on_border = dr == 0 or dc == 0 or dr == h - 1 or dc == w - 1
            corner = dr in (0, h - 1) and dc in (0, w - 1)
            if on_border and not corner:
                ring.append((r0 + dr, c0 + dc))
    cavity = [(r0 + 1 + dr, c0 + 1 + dc) for dr in range(ch) for dc in range(cw)]
    return ring, cavity


def _compute_cavities(g: np.ndarray, bg: int) -> list[dict]:
    h, w = g.shape
    bg_comps = _components_mask(g == bg, conn8=False)
    corners = {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}
    cavities: list[dict] = []
    for pts in bg_comps:
        region = set(pts)
        if any(p in region for p in corners if int(g[p]) == bg):
            continue
        boundary = set()
        for r, c in region:
            for nr, nc in neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in region and int(g[nr, nc]) != bg:
                    boundary.add((nr, nc))
        if not boundary:
            continue
        cavities.append(
            {
                "region": region,
                "boundary": boundary,
                "bcols": {int(g[p]) for p in boundary},
            }
        )
    return cavities


def _seed_line_info(g: np.ndarray, seed: tuple[int, int], bg: int) -> tuple[int, set[tuple[int, int]], set[tuple[int, int]]]:
    h, w = g.shape
    sr, sc = seed
    neigh_cols = []
    for nr, nc in neighbors4(sr, sc):
        if 0 <= nr < h and 0 <= nc < w and int(g[nr, nc]) != bg:
            neigh_cols.append(int(g[nr, nc]))
    assert neigh_cols, "seed pixel must have at least one non-bg neighbor"
    line_col = Counter(neigh_cols).most_common(1)[0][0]
    starts = [(nr, nc) for nr, nc in neighbors4(sr, sc) if 0 <= nr < h and 0 <= nc < w and int(g[nr, nc]) == line_col]
    line_pixels = _same_color_component8_from(g, starts, line_col)
    traced = _nonbg_component8_from(g, seed, bg)
    return int(line_col), line_pixels, traced


def _boundary_touches_active(boundary: set[tuple[int, int]], active: set[tuple[int, int]]) -> bool:
    for r, c in boundary:
        for nr, nc in neighbors8(r, c):
            if (nr, nc) in active:
                return True
    return False


def _seed_activation_rounds(g: np.ndarray, cavities: list[dict], seed: tuple[int, int], bg: int) -> dict[int, int]:
    _line_col, line_pixels, traced = _seed_line_info(g, seed, bg)
    active = set(line_pixels)
    rounds: dict[int, int] = {}
    eligible = [i for i, cav in enumerate(cavities) if len(cav["bcols"]) == 1 and bool(cav["boundary"] & traced)]

    round_id = 0
    while True:
        new_ids = []
        for idx in eligible:
            if idx in rounds:
                continue
            if _boundary_touches_active(cavities[idx]["boundary"], active):
                new_ids.append(idx)
        if not new_ids:
            break
        round_id += 1
        for idx in new_ids:
            rounds[idx] = round_id
        for idx in new_ids:
            active.update(cavities[idx]["region"])
    return rounds


def _one_step_fill(g: np.ndarray) -> np.ndarray:
    bg = get_bg(g)
    out = g.copy()
    cavities = _compute_cavities(g, bg)
    for seed in _seed_pixels(g, bg):
        seed_col = int(g[seed])
        line_col, line_pixels, traced = _seed_line_info(g, seed, bg)
        for cav in cavities:
            if len(cav["bcols"]) != 1 or not (cav["boundary"] & traced):
                continue
            ok = False
            for r, c in cav["boundary"]:
                for nr, nc in neighbors8(r, c):
                    if (nr, nc) in line_pixels or (0 <= nr < g.shape[0] and 0 <= nc < g.shape[1] and int(g[nr, nc]) == line_col and (nr, nc) in traced):
                        ok = True
                        break
                if ok:
                    break
            if ok:
                for r, c in cav["region"]:
                    if int(out[r, c]) == bg:
                        out[r, c] = seed_col
    return out


def _activation_profile(g: np.ndarray) -> dict:
    bg = get_bg(g)
    cavities = _compute_cavities(g, bg)
    per_seed = []
    max_round = 0
    propagated = 0
    for seed in _seed_pixels(g, bg):
        rounds = _seed_activation_rounds(g, cavities, seed, bg)
        if rounds:
            seed_max = max(rounds.values())
            seed_prop = sum(v >= 2 for v in rounds.values())
        else:
            seed_max = 0
            seed_prop = 0
        per_seed.append({"seed": seed, "rounds": rounds, "max_round": seed_max, "propagated": seed_prop})
        max_round = max(max_round, seed_max)
        propagated += seed_prop
    return {"max_round": max_round, "propagated_cavities": propagated, "per_seed": per_seed}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Variant rule:
    - trace each seed's own line network as before
    - but do not fill every reachable cavity at once
    - instead, use iterative frontier expansion:
      line pixels activate round-1 cavities; newly filled cavities become new frontier;
      continue until no further cavity boundary touches the frontier
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    cavities = _compute_cavities(g, bg)
    out = g.copy()

    for seed in _seed_pixels(g, bg):
        seed_col = int(g[seed])
        rounds = _seed_activation_rounds(g, cavities, seed, bg)
        for idx in rounds:
            for r, c in cavities[idx]["region"]:
                if int(out[r, c]) == bg:
                    out[r, c] = seed_col
    return out


# ----------------------------- generator -----------------------------

GRID_H_RANGE = (8, 30)
GRID_W_RANGE = (8, 30)
BG_COLORS = [0, 5]

SEED_COUNT_WEIGHTS = [1, 2]
MIN_LINE_LEN = 5
LINE_LEN_RANGE = (9, 34)

FRAME_COLORS_MAX = 4
CHAIN_LEN_CHOICES = [2, 2, 3, 3, 4]
EXTRA_SINGLE_LOOPS_MAX = 2
DISTRACTOR_SOLO_MAX = 2
FRAME_PLACEMENT_PAD = 1
CHAIN_PLACEMENT_TRIES = 80
SINGLE_FRAME_TRIES = 40
DISTRACTOR_FRAME_TRIES = 40

CHAIN_CAVITY_SIZES = [2, 2, 3]
SINGLE_CAVITY_SIZES = [2, 2, 2, 3]

DIAG_WEIGHT_ALIGNED = 5
DIAG_WEIGHT_OTHER = 1
ORTHO_WEIGHT_ALIGNED = 2
ORTHO_WEIGHT_OTHER = 1
BORDER_AVOIDANCE_PROB = 0.85
BIAS_FLIP_PROB = 0.3

GEN_RETRIES = 400

DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
ORTHO_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
CARDINALS = {
    "R": (0, 1),
    "L": (0, -1),
    "D": (1, 0),
    "U": (-1, 0),
}


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _grow_line(grid: np.ndarray, start: tuple[int, int], line_col: int, target_len: int, rng: random.Random) -> list[tuple[int, int]]:
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


def _corner_config(corner_kind: str, h: int, w: int, rng: random.Random) -> tuple[tuple[int, int], tuple[int, int]]:
    if corner_kind == "TL":
        return (0, 0), rng.choice([(0, 1), (1, 0)])
    if corner_kind == "TR":
        return (0, w - 1), rng.choice([(0, w - 2), (1, w - 1)])
    if corner_kind == "BL":
        return (h - 1, 0), rng.choice([(h - 1, 1), (h - 2, 0)])
    if corner_kind == "BR":
        return (h - 1, w - 1), rng.choice([(h - 1, w - 2), (h - 2, w - 1)])
    raise ValueError(corner_kind)


def _distribute_loop_budget(num_seeds: int, total: int, rng: random.Random) -> list[int]:
    order = list(range(num_seeds))
    rng.shuffle(order)
    per_seed = [0] * num_seeds
    remaining = total
    while remaining > 0 and any(v == 0 for v in per_seed):
        for idx in order:
            if remaining <= 0:
                break
            if per_seed[idx] == 0:
                per_seed[idx] += 1
                remaining -= 1
    while remaining > 0:
        per_seed[rng.choice(order)] += 1
        remaining -= 1
    return per_seed


def _chain_layout(r0: int, c0: int, ch: int, cw: int, length: int, direction: str) -> tuple[list[list[tuple[int, int]]], list[list[tuple[int, int]]]]:
    rings = []
    cavities = []
    if direction in ("R", "L"):
        step = cw + 1
        sign = 1 if direction == "R" else -1
        for i in range(length):
            ring, cavity = _frame_ring_coords(r0, c0 + sign * i * step, ch, cw)
            rings.append(ring)
            cavities.append(cavity)
    else:
        step = ch + 1
        sign = 1 if direction == "D" else -1
        for i in range(length):
            ring, cavity = _frame_ring_coords(r0 + sign * i * step, c0, ch, cw)
            rings.append(ring)
            cavities.append(cavity)
    return rings, cavities


def _validate_chain_placement(
    canvas: np.ndarray,
    rings: list[list[tuple[int, int]]],
    cavities: list[list[tuple[int, int]]],
    line_pixels: set[tuple[int, int]],
    line_col: int,
) -> bool:
    h, w = canvas.shape
    ring_sets = [set(r) for r in rings]
    cavity_sets = [set(c) for c in cavities]
    all_ring = set().union(*ring_sets)
    all_cavity = set().union(*cavity_sets)

    for rr, cc in all_ring | all_cavity:
        if not _in_bounds(h, w, rr, cc):
            return False
        if canvas[rr, cc] != -1:
            return False

    for rr, cc in all_cavity:
        for nr, nc in neighbors8(rr, cc):
            if (nr, nc) in line_pixels:
                return False

    for rr, cc in all_ring:
        for nr, nc in neighbors8(rr, cc):
            if not _in_bounds(h, w, nr, nc):
                continue
            if (nr, nc) in all_ring or (nr, nc) in all_cavity:
                continue
            v = int(canvas[nr, nc])
            if v != -1 and v != line_col:
                return False

    first_touches = any(any((nr, nc) in line_pixels for nr, nc in neighbors8(rr, cc)) for rr, cc in ring_sets[0])
    if not first_touches:
        return False

    for later in ring_sets[1:]:
        if any(any((nr, nc) in line_pixels for nr, nc in neighbors8(rr, cc)) for rr, cc in later):
            return False
    return True


def _place_chain(
    canvas: np.ndarray,
    line_path: list[tuple[int, int]],
    line_col: int,
    frame_col: int,
    length: int,
    rng: random.Random,
) -> list[list[tuple[int, int]]] | None:
    h, w = canvas.shape
    line_pixels = set(line_path)
    for _ in range(CHAIN_PLACEMENT_TRIES):
        ch = rng.choice(CHAIN_CAVITY_SIZES)
        cw = rng.choice(CHAIN_CAVITY_SIZES)
        fh, fw = ch + 2, cw + 2
        lp = rng.choice(line_path)
        offsets = _halo_offsets(fh, fw)
        rng.shuffle(offsets)
        dirs = list(CARDINALS)
        rng.shuffle(dirs)
        for dr, dc in offsets:
            r0 = lp[0] - dr
            c0 = lp[1] - dc
            for direction in dirs:
                rings, cavities = _chain_layout(r0, c0, ch, cw, length, direction)
                all_pts = set().union(*[set(x) for x in rings + cavities])
                if not all_pts:
                    continue
                rs = [r for r, _ in all_pts]
                cs = [c for _, c in all_pts]
                if min(rs) < FRAME_PLACEMENT_PAD or min(cs) < FRAME_PLACEMENT_PAD:
                    continue
                if max(rs) > h - 1 - FRAME_PLACEMENT_PAD or max(cs) > w - 1 - FRAME_PLACEMENT_PAD:
                    continue
                if not _validate_chain_placement(canvas, rings, cavities, line_pixels, line_col):
                    continue
                for rr, cc in set().union(*[set(r) for r in rings]):
                    canvas[rr, cc] = frame_col
                return cavities
    return None


def _validate_single_frame(
    canvas: np.ndarray,
    ring: list[tuple[int, int]],
    cavity: list[tuple[int, int]],
    line_pixels: set[tuple[int, int]],
    line_col: int,
) -> bool:
    h, w = canvas.shape
    ring_set = set(ring)
    cavity_set = set(cavity)
    for rr, cc in ring_set | cavity_set:
        if not _in_bounds(h, w, rr, cc) or canvas[rr, cc] != -1:
            return False
    for rr, cc in cavity_set:
        for nr, nc in neighbors8(rr, cc):
            if (nr, nc) in line_pixels:
                return False
    for rr, cc in ring_set:
        for nr, nc in neighbors8(rr, cc):
            if not _in_bounds(h, w, nr, nc) or (nr, nc) in ring_set or (nr, nc) in cavity_set:
                continue
            v = int(canvas[nr, nc])
            if v != -1 and v != line_col:
                return False
    return any(any((nr, nc) in line_pixels for nr, nc in neighbors8(rr, cc)) for rr, cc in ring_set)


def _place_single_attached_frame(
    canvas: np.ndarray,
    line_path: list[tuple[int, int]],
    line_col: int,
    frame_col: int,
    rng: random.Random,
) -> bool:
    h, w = canvas.shape
    line_pixels = set(line_path)
    for _ in range(SINGLE_FRAME_TRIES):
        ch = rng.choice(SINGLE_CAVITY_SIZES)
        cw = rng.choice(SINGLE_CAVITY_SIZES)
        fh, fw = ch + 2, cw + 2
        lp = rng.choice(line_path)
        offsets = _halo_offsets(fh, fw)
        rng.shuffle(offsets)
        for dr, dc in offsets:
            r0 = lp[0] - dr
            c0 = lp[1] - dc
            ring, cavity = _frame_ring_coords(r0, c0, ch, cw)
            rs = [r for r, _ in ring + cavity]
            cs = [c for _, c in ring + cavity]
            if min(rs) < FRAME_PLACEMENT_PAD or min(cs) < FRAME_PLACEMENT_PAD:
                continue
            if max(rs) > h - 1 - FRAME_PLACEMENT_PAD or max(cs) > w - 1 - FRAME_PLACEMENT_PAD:
                continue
            if not _validate_single_frame(canvas, ring, cavity, line_pixels, line_col):
                continue
            for rr, cc in ring:
                canvas[rr, cc] = frame_col
            return True
    return False


def _place_distractor_frame(canvas: np.ndarray, frame_col: int, rng: random.Random) -> bool:
    h, w = canvas.shape
    for _ in range(DISTRACTOR_FRAME_TRIES):
        ch = rng.choice(SINGLE_CAVITY_SIZES)
        cw = rng.choice(SINGLE_CAVITY_SIZES)
        fh, fw = ch + 2, cw + 2
        if h - fh - 1 <= 1 or w - fw - 1 <= 1:
            return False
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
        ring, _cavity = _frame_ring_coords(r0, c0, ch, cw)
        for rr, cc in ring:
            canvas[rr, cc] = frame_col
        return True
    return False


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
    num_frame_cols = rng.randint(1, min(FRAME_COLORS_MAX, len(palette)))
    frame_colors_pool = [palette.pop() for _ in range(num_frame_cols)]

    corners = ["TL", "TR", "BL", "BR"]
    rng.shuffle(corners)
    corners_used = corners[:num_seeds]

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

    total_loops_target = rng.randint(num_seeds + 1, num_seeds + 5)
    per_seed = _distribute_loop_budget(num_seeds, total_loops_target, rng)
    chain_seed = rng.randrange(num_seeds)
    chain_lengths = [1] * num_seeds
    chain_lengths[chain_seed] = rng.choice(CHAIN_LEN_CHOICES)
    if num_seeds == 2 and rng.random() < 0.5:
        other = 1 - chain_seed
        chain_lengths[other] = rng.choice([2, 2, 3])

    at_least_one_propagating_chain = False
    for i, (_seed_col, _seed_rc, line_col, path) in enumerate(lines_info):
        loops_for_seed = max(per_seed[i], chain_lengths[i])
        frame_col = rng.choice(frame_colors_pool)
        cavities = _place_chain(canvas, path, line_col, frame_col, chain_lengths[i], rng)
        if cavities is None:
            return None
        if chain_lengths[i] >= 2:
            at_least_one_propagating_chain = True
        extra_needed = max(0, loops_for_seed - chain_lengths[i])
        extra_cap = min(extra_needed, EXTRA_SINGLE_LOOPS_MAX)
        for _ in range(extra_cap):
            _place_single_attached_frame(canvas, path, line_col, rng.choice(frame_colors_pool), rng)

    if not at_least_one_propagating_chain:
        return None

    for _ in range(rng.randint(0, DISTRACTOR_SOLO_MAX)):
        _place_distractor_frame(canvas, rng.choice(frame_colors_pool), rng)

    grid = np.where(canvas == -1, bg, canvas).astype(int)

    vals, cnts = np.unique(grid, return_counts=True)
    if int(vals[np.argmax(cnts)]) != bg:
        return None
    for seed_col, _seed_rc, _line_col, _path in lines_info:
        if int((grid == seed_col).sum()) != 1:
            return None

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None

    profile = _activation_profile(grid)
    if profile["max_round"] < 2 or profile["propagated_cavities"] < 1:
        return None
    if np.array_equal(out, _one_step_fill(grid)):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError("generate_input: exhausted attempts")
