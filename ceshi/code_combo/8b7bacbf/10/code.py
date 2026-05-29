# Combo variant of 8b7bacbf (index 10)
# combo_spec: D6/1+D8/0
# Name: 填充+擦除路径+迭代传播

from __future__ import annotations

import random
import sys
from collections import Counter

import numpy as np

sys.path.append("/home/lixinyu/snap/new/solvers")
from arc_utils import components_mask, get_bg, neighbors4, neighbors8

"""
design_analysis:
  concretization_and_trace:
    - 保留原题的对象词汇：边缘唯一色种子、每个种子对应的一条 8 连通细线、若干单色框环以及框内背景腔体。
    - 融合后的关键样例应包含至少一条“链式框串”：
      第一框直接贴线；第二框不贴线，只与第一框共墙；第三框再与第二框共墙。
    - 原题在这类结构上的行为：若整个框串都属于种子可追踪的非背景网络，solver 会把所有可识别腔体一次性填完。
    - D6 单变体只改写结果：被激活腔体填色，同时其边框被擦成背景。
    - 本融合版进一步改写时序：round1 只激活直接贴当前前沿的腔体；该腔体的内部被填成种子色，环墙被擦成背景；
      被填内部与被擦环墙一起并入“活动前沿”，round2 再去碰下一个腔体；如此迭代直到没有新腔体。
    - 具体追踪：
      seed=7，line=3，三个并排共享墙体的框 A/B/C，框色都为 4。
      原题：A/B/C 的腔体都会直接变成 7，环 4 保留。
      仅 D6：只有直接可激活的框会被填成 7 且其环被擦掉。
      本融合版：round1 填 A、擦 A 的环；A 的内部和被擦掉的共享墙进入前沿。
      round2 前沿首次接触 B，于是填 B、擦 B 的环。
      round3 同理激活 C。输出表现为沿框链逐步扩散，同时已激活框的轮廓消失。
  complexity_calibration:
    - 若所有框都直接贴线，融合版会退化成“一轮全做完”，人眼看不出 D8 的存在；因此生成器必须强制至少一条链长 >= 2。
    - 若链过长或附加大量无种子网络，网格会拥挤且不利于从 2-4 个例子归纳；因此链长限制在 2-4，
      并只加少量额外单框/孤立干扰框。
    - 颜色语义保持极简：种子色=最终填充色，线色只负责指出起始前沿，框色只负责围出腔体并在激活后消失。
  implicit_assumption_review:
    - 原 solver 里“boundary & traced 即可立即填充”的前提失效；在融合版中，这只能说明某腔体属于同一网络，
      不能说明它在当前轮已暴露。
    - 原 solver 的 cavity 列表可以继续静态复用，但激活条件必须改成“边界是否触达当前活动前沿”，而不是“是否属于 traced”。
    - 单纯把新填腔体内部加入前沿还不够；共享墙体会在 D6 擦除后变成新的通道，因此被擦掉的 boundary 也必须并入前沿。
    - 原来的“单色边界”假设仍仅用于识别输入里的原始候选腔体；传播阶段不重新分割网格，以避免擦环后出现多色边界而漏检。
    - 新假设：
      每个种子的正交邻居里存在至少一个非背景像素，从而可恢复 line_color。
      不同种子网络彼此分离，不会竞争同一腔体。
      传播链的后继腔体通过共墙或近邻方式接触前一轮前沿。
  placement_legality:
    - 生成器使用轴对齐矩形环，角仍留空，保证与原题视觉一致。
    - 链式相邻框通过共享整段墙体摆放；腔体内部不重叠，整体不越界。
    - 只有链首允许接触 line_pixels；后续框若也贴线，则会退化成一轮激活，因此显式禁止。
    - 干扰单框放在留白区域，至少隔开一圈空白，避免被任何线或链意外触发。
  visual_observability:
    - 输入中可以直接看到“线只碰第一个框，其余框只顺着共墙串下去”的层次。
    - 输出中除了种子色扩散，还能看到已激活框的环被擦掉，未激活/干扰框保持原样。
    - 生成器会拒绝所有与“单步填充+擦环”输出完全相同的样本，确保 D8 的多轮特征确实可见。
  cleanliness_and_inference:
    - 每个样例只保留 1-2 条种子线、1-2 条短链和少量孤立框，避免局部堆叠过密。
    - 颜色变化完全由可见结构驱动，不依赖颜色编号大小或其他不可视属性。
  solver_assumption_audit:
    - _seed_pixels 假设种子色在全图只出现一次且落在边缘；generator 保证。
    - _seed_line_info 假设种子正交邻接线像素而非直接邻接框；generator 从角点先放 line 首像素保证。
    - _seed_activation_rounds 假设可传播的后继腔体都在初始 traced 网络内；链式共享墙保证这一点。
    - main 假设不同种子不会改写同一 cavity/boundary；generator 保证不同网络分离。
"""


def _components_mask(mask: np.ndarray, conn8: bool) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8 if conn8 else 4)]


def _seed_pixels(g: np.ndarray, bg: int) -> list[tuple[int, int]]:
    h, w = g.shape
    vals, cnts = np.unique(g, return_counts=True)
    seeds: list[tuple[int, int]] = []
    for v, cnt in zip(vals, cnts):
        value = int(v)
        if value == bg or int(cnt) != 1:
            continue
        r, c = map(int, np.argwhere(g == value)[0])
        if r == 0 or c == 0 or r == h - 1 or c == w - 1:
            seeds.append((r, c))
    return seeds


def _nonbg_component8_from(g: np.ndarray, start: tuple[int, int], bg: int) -> set[tuple[int, int]]:
    h, w = g.shape
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for nr, nc in neighbors8(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in seen and int(g[nr, nc]) != bg:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return seen


def _same_color_component8_from(g: np.ndarray, starts: list[tuple[int, int]], color: int) -> set[tuple[int, int]]:
    h, w = g.shape
    seen = {p for p in starts if 0 <= p[0] < h and 0 <= p[1] < w and int(g[p]) == color}
    stack = list(seen)
    while stack:
        r, c = stack.pop()
        for nr, nc in neighbors8(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in seen and int(g[nr, nc]) == color:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return seen


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
    line_pixels = _same_color_component8_from(g, starts, int(line_col))
    traced = _nonbg_component8_from(g, seed, bg)
    return int(line_col), line_pixels, traced


def _boundary_reaches_active(boundary: set[tuple[int, int]], active: set[tuple[int, int]]) -> bool:
    if boundary & active:
        return True
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
            if _boundary_reaches_active(cavities[idx]["boundary"], active):
                new_ids.append(idx)
        if not new_ids:
            break
        round_id += 1
        for idx in new_ids:
            rounds[idx] = round_id
        for idx in new_ids:
            active.update(cavities[idx]["region"])
            active.update(cavities[idx]["boundary"])
    return rounds


def _one_step_fill_erase(g: np.ndarray) -> np.ndarray:
    bg = get_bg(g)
    out = g.copy()
    cavities = _compute_cavities(g, bg)
    for seed in _seed_pixels(g, bg):
        seed_col = int(g[seed])
        _line_col, line_pixels, traced = _seed_line_info(g, seed, bg)
        for cav in cavities:
            if len(cav["bcols"]) != 1 or not (cav["boundary"] & traced):
                continue
            if not _boundary_reaches_active(cav["boundary"], line_pixels):
                continue
            for r, c in cav["region"]:
                if int(out[r, c]) == bg:
                    out[r, c] = seed_col
            for r, c in cav["boundary"]:
                out[r, c] = bg
    return out


def _fusion_profile(g: np.ndarray) -> dict:
    bg = get_bg(g)
    cavities = _compute_cavities(g, bg)
    by_seed = []
    max_round = 0
    propagated = 0
    propagated_boundary_cells = 0
    for seed in _seed_pixels(g, bg):
        rounds = _seed_activation_rounds(g, cavities, seed, bg)
        seed_prop = [idx for idx, rid in rounds.items() if rid >= 2]
        if rounds:
            max_round = max(max_round, max(rounds.values()))
        propagated += len(seed_prop)
        propagated_boundary_cells += sum(len(cavities[idx]["boundary"]) for idx in seed_prop)
        by_seed.append({"seed": seed, "rounds": rounds})
    return {
        "max_round": max_round,
        "propagated_cavities": propagated,
        "propagated_boundary_cells": propagated_boundary_cells,
        "by_seed": by_seed,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Iteratively fill line-linked cavities, erasing each activated ring as the front advances.

    Input structure:
    - The grid has a dominant background color, 1-2 unique seed pixels on the border, one thin 8-connected line per seed,
      and several single-color frame loops enclosing background cavities.
    - Some loops are arranged in short shared-wall chains: only the first loop touches the seed's line directly,
      while later loops touch earlier loops instead.

    Rule:
    - For each seed, recover its line network and treat the line as the initial active front.
    - Repeatedly find still-unfilled cavities from that traced network whose boundary now touches the active front.
      Fill each such cavity with the seed color and erase its boundary ring to background.
    - The newly filled interior and the erased ring both extend the front for the next round. Stop when no new cavity is exposed.

    Key insight:
    - Membership in the seed's overall non-background network is not enough to trigger filling immediately.
      A cavity only activates when the current front reaches it, and erasing one ring can expose the next cavity in the chain.
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
            for r, c in cavities[idx]["boundary"]:
                out[r, c] = bg
    return out


# ----------------------------- generator -----------------------------

GRID_H_RANGE = (8, 30)
GRID_W_RANGE = (8, 30)
BG_COLORS = [0, 5]

SEED_COUNT_WEIGHTS = [1, 1, 2]
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

GEN_RETRIES = 450

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


def _chain_layout(
    r0: int,
    c0: int,
    ch: int,
    cw: int,
    length: int,
    direction: str,
) -> tuple[list[list[tuple[int, int]]], list[list[tuple[int, int]]]]:
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
    ring_sets = [set(ring) for ring in rings]
    cavity_sets = [set(cavity) for cavity in cavities]
    all_ring = set().union(*ring_sets)
    all_cavity = set().union(*cavity_sets)

    for rr, cc in all_ring | all_cavity:
        if not _in_bounds(h, w, rr, cc) or canvas[rr, cc] != -1:
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
            value = int(canvas[nr, nc])
            if value != -1 and value != line_col:
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
) -> bool:
    h, w = canvas.shape
    line_pixels = set(line_path)
    for _ in range(CHAIN_PLACEMENT_TRIES):
        ch = rng.choice(CHAIN_CAVITY_SIZES)
        cw = rng.choice(CHAIN_CAVITY_SIZES)
        fh, fw = ch + 2, cw + 2
        lp = rng.choice(line_path)
        offsets = _halo_offsets(fh, fw)
        rng.shuffle(offsets)
        directions = list(CARDINALS)
        rng.shuffle(directions)
        for dr, dc in offsets:
            r0 = lp[0] - dr
            c0 = lp[1] - dc
            for direction in directions:
                rings, cavities = _chain_layout(r0, c0, ch, cw, length, direction)
                all_pts = set().union(*[set(item) for item in rings + cavities])
                rs = [r for r, _ in all_pts]
                cs = [c for _, c in all_pts]
                if min(rs) < FRAME_PLACEMENT_PAD or min(cs) < FRAME_PLACEMENT_PAD:
                    continue
                if max(rs) > h - 1 - FRAME_PLACEMENT_PAD or max(cs) > w - 1 - FRAME_PLACEMENT_PAD:
                    continue
                if not _validate_chain_placement(canvas, rings, cavities, line_pixels, line_col):
                    continue
                for rr, cc in set().union(*[set(ring) for ring in rings]):
                    canvas[rr, cc] = frame_col
                return True
    return False


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
            value = int(canvas[nr, nc])
            if value != -1 and value != line_col:
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
        ring, _ = _frame_ring_coords(r0, c0, ch, cw)
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
    frame_color_count = rng.randint(1, min(FRAME_COLORS_MAX, len(palette)))
    frame_colors_pool = [palette.pop() for _ in range(frame_color_count)]

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
    if num_seeds == 2 and rng.random() < 0.45:
        other = 1 - chain_seed
        chain_lengths[other] = rng.choice([2, 2, 3])

    at_least_one_propagating_chain = False
    for i, (_seed_col, _seed_rc, line_col, path) in enumerate(lines_info):
        loops_for_seed = max(per_seed[i], chain_lengths[i])
        if not _place_chain(canvas, path, line_col, rng.choice(frame_colors_pool), chain_lengths[i], rng):
            return None
        if chain_lengths[i] >= 2:
            at_least_one_propagating_chain = True
        extra_cap = min(max(0, loops_for_seed - chain_lengths[i]), EXTRA_SINGLE_LOOPS_MAX)
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

    profile = _fusion_profile(grid)
    if profile["max_round"] < 2 or profile["propagated_cavities"] < 1:
        return None
    if profile["propagated_boundary_cells"] <= 0:
        return None
    if np.array_equal(out, _one_step_fill_erase(grid)):
        return None

    filled_cells = int(((grid == bg) & (out != bg)).sum())
    erased_cells = int(((grid != bg) & (out == bg)).sum())
    if filled_cells <= 0 or erased_cells <= 0:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted attempts")
