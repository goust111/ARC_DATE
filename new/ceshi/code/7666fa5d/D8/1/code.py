# Variant of 7666fa5d
# Dimension: D8 — 多步处理：填充顺序与迭代 (multi_step)
# Change: 单步处理——对每条主对角线独立扫描，收集活跃线段后一次性填充所有区间... → 先连接后填充（两步处理）

import random

import numpy as np

from arc_utils import components_mask, get_bg as _bg_color

"""
Design analysis for the D8 variant
source_task: none

a) 具体化与行为追踪
   - 仍然是 16x16、azure 背景、单一线段颜色、红色填充。
   - 线段仍然沿反对角线 k=r+c 放置，但输入里至少有一条“中间层”反对角线
     会被故意打断成两个 8-不连通片段；它们共线、同色、同 k，只是中间留 1-2 个背景格。
   - 输出分两步形成：
     1. 先把同一 k 上的片段间空隙补成线段本色，得到完整连续段。
     2. 再把这些“补完后的完整段”之间的平行四边形区域填成红色。
   - 小例子：若 k=8、12、16 三条线都覆盖某条主对角线 b=2，而 k=12 在输入里缺少
     b=2 对应的中间格，则原始单步逻辑会在该格处把 k=8 与 k=16 之间整段视为可填背景，
     该格会变红；本变体先补上 k=12 的线色，该格最终保留为线色，不会被红色覆盖。

b) 复杂度校准
   - 只引入一种新的可见机制：同一反对角线上的断裂先被修复。
   - 仍保留 4-5 条反对角线、连续长度与原题接近，避免把任务简化成“单条线补洞”。
   - 断裂固定出现在内部 k（不是最外侧边界线），这样它同时参与左右两侧关系，
     使“先连接后填充”和原始单步逻辑产生稳定差异。

c) 原始 main() 的隐式假设审查
   - 原代码 `components_mask(g == line)` 后，把每个 8 连通块都当作一条独立完整线段。
     这依赖“同一 k 上不存在逻辑上属于同一条线的断裂片段”。变体中该前提失效。
   - 原代码按组件直接生成 `(k, b_min, b_max)`；若同一 k 有两个片段，则它们会变成两条
     分开的短段，缺口处的 b 不再被视为该 k 活跃。
   - 原代码 `out = g.copy()` 后只会把背景改成红色，永远不会新增线段颜色；
     这在原题无害，但在本变体中会漏掉“先连接”的第一步。
   - 因此新 solver 必须：
     1. 先按 k 合并所有线色像素，逻辑段取该 k 上最小/最大 b 的整段。
     2. 先把这些整段画回输出，再做原题式的红色填充。

d) 放置合法性
   - generator 先构造“输出态”的连续整段，再反推出“输入态”的断裂段。
   - 所有 k 共用一个主对角线重叠核心区间 core，保证多条线在同一批 b 上同时活跃，
     从而一定能形成填充，也保证断裂发生在真正参与交互的位置。
   - 断裂只发生在内部 k，且缺口完全落在 core 内；这样补线与填充两步都可观察。
   - 所有段仍然位于单一 k 上，所有格坐标通过 `(r,c)=((k+b)/2,(k-b)/2)` 计算并受边界约束。

e) 视觉可观察性
   - 解题者会看到：输入中有一条斜线明显“断了一截”，输出先把这截补回原色，
     红色区域再以补完后的线为边界展开。
   - 若没有这一步，缺口位置会出现红色或更大的红区；因此 2-3 组样例就能观察到
     “断线要先补，再做填充”这一顺序差异。

f) 可视化整洁度
   - 仍然只有三类像素：背景、线色、红填充；没有额外标记层。
   - 断裂只出现在 1 条内部反对角线上，避免整图充满碎片。
   - core 让各段的交互集中在同一斜向带状区域，输出图形仍然是清晰的平行四边形簇。

g) 颜色变化可推断性
   - 颜色规则没有额外映射：断裂补线使用原线色，区域填充始终使用红色 2。
   - 颜色变化完全由结构角色决定：共线缺口 -> 线色；线段之间的背景 -> 红色。

9) 新 solver 的隐式假设与 generator 保证
   - 假设输入只有一种主要非背景线色；generator 始终只放一种线色。
   - 假设同一 k 上的多个片段应视为一条逻辑段；generator 只在这种情况下制造断裂。
   - 假设所有线色像素都落在合法反对角线上；generator 用统一坐标公式放置并验证。
   - 假设断裂不会与别的对象重叠；generator 先生成连续输出段，再只删除该段内部的若干格。
"""


def _cc8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8)]


def _b_bounds_for_k(k: int) -> tuple[int, int]:
    b_lo = max(-k, k - 2 * (W - 1))
    b_hi = min(2 * (H - 1) - k, k)
    return b_lo, b_hi


def _align_bounds(b_lo: int, b_hi: int, parity: int) -> tuple[int, int]:
    if b_lo % 2 != parity:
        b_lo += 1
    if b_hi % 2 != parity:
        b_hi -= 1
    return b_lo, b_hi


def _seg_cells(k: int, b_min: int, b_max: int) -> list[tuple[int, int]]:
    return [((k + b) // 2, (k - b) // 2) for b in range(b_min, b_max + 1, 2)]


def _extract_merged_segments(grid: np.ndarray, line: int) -> list[tuple[int, int, int]]:
    by_k: dict[int, list[int]] = {}
    for r, c in np.argwhere(grid == line):
        rr = int(r)
        cc = int(c)
        k = rr + cc
        b = rr - cc
        by_k.setdefault(k, []).append(b)
    return sorted((k, min(bs), max(bs)) for k, bs in by_k.items())


def _draw_completed_segments(grid: np.ndarray, segs: list[tuple[int, int, int]], line: int, bg: int) -> np.ndarray:
    out = grid.copy()
    for k, b_min, b_max in segs:
        for r, c in _seg_cells(k, b_min, b_max):
            if out[r, c] == bg:
                out[r, c] = line
    return out


def _apply_fill(grid: np.ndarray, segs: list[tuple[int, int, int]], bg: int, fill: int) -> np.ndarray:
    out = grid.copy()
    for b in range(-(W - 1), H):
        active = sorted([s for s in segs if s[1] <= b <= s[2]], key=lambda t: t[0])
        if len(active) < 2:
            continue
        r_min = max(0, b)
        r_max = min(H - 1, (W - 1) + b)
        for (k_left, _a0, _a1), (k_right, _b0, _b1) in zip(active, active[1:]):
            for r in range(r_min, r_max + 1):
                c = r - b
                k = r + c
                if k_left < k < k_right and out[r, c] == bg:
                    out[r, c] = fill
    return out


def _legacy_segments(grid: np.ndarray, line: int) -> list[tuple[int, int, int]]:
    segs: list[tuple[int, int, int]] = []
    for pts in _cc8(grid == line):
        ks = {r + c for r, c in pts}
        if len(ks) != 1:
            continue
        k = int(next(iter(ks)))
        bs = [r - c for r, c in pts]
        segs.append((k, int(min(bs)), int(max(bs))))
    return segs


def _legacy_main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    non_bg = [c for c in sorted(set(map(int, g.ravel()))) if c != bg]
    assert non_bg
    line = max(non_bg, key=lambda c: int(np.sum(g == c)))
    return _apply_fill(g.copy(), _legacy_segments(g, line), bg, FILL)


def _variant_metrics(grid: np.ndarray) -> dict[str, int]:
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    non_bg = [c for c in sorted(set(map(int, g.ravel()))) if c != bg]
    assert non_bg
    line = max(non_bg, key=lambda c: int(np.sum(g == c)))
    merged = _extract_merged_segments(g, line)
    legacy = _legacy_segments(g, line)
    bridge_cells = sum((b1 - b0) // 2 + 1 for _k, b0, b1 in merged) - sum((b1 - b0) // 2 + 1 for _k, b0, b1 in legacy)
    variant_out = main(g)
    legacy_out = _legacy_main(g)
    return {
        "bridge_cells": int(bridge_cells),
        "delta_vs_legacy": int(np.sum(variant_out != legacy_out)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    D8 variant:
    1. Merge broken same-k anti-diagonal fragments into completed line segments.
    2. Fill red between adjacent completed segments along each main diagonal.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert (h, w) == (H, W), "grid must be 16x16"

    bg = _bg_color(g)
    non_bg = [c for c in sorted(set(map(int, g.ravel()))) if c != bg]
    assert non_bg, "grid must contain at least one non-background color"

    line = max(non_bg, key=lambda c: int(np.sum(g == c)))
    segs = _extract_merged_segments(g, line)

    out = _draw_completed_segments(g, segs, line, bg)
    out = _apply_fill(out, segs, bg, FILL)
    return out


# --------------- generator ---------------

H = W = 16
BG = 8
FILL = 2
LINE_COLORS = [1, 3, 4, 5, 6, 7, 9]

K_LO, K_HI = 4, 26
GAP_VALUES = (4, 6, 8)
GAP_TARGET_SPANS = (14, 16)
NUM_KS_CHOICES = [4, 5]
NUM_KS_WEIGHTS = [1, 1]

SEG_LEN_MIN, SEG_LEN_MAX = 5, 9
CORE_LEN_MIN, CORE_LEN_MAX = 3, 5

GEN_RETRY = 400
BUILD_RETRY = 60
MIN_FILL_CELLS = 6


def _choose_k_values(parity: int) -> list[int] | None:
    num_ks = random.choices(NUM_KS_CHOICES, weights=NUM_KS_WEIGHTS)[0]
    if num_ks == 4:
        valid_combos = [(a, b, c) for a in GAP_VALUES for b in GAP_VALUES for c in GAP_VALUES if a + b + c in GAP_TARGET_SPANS]
        gaps = list(random.choice(valid_combos))
    else:
        gaps = [4, 4, 4, 4]

    total_span = sum(gaps)
    starts = [k for k in range(K_LO, K_HI - total_span + 1) if k % 2 == parity]
    if not starts:
        return None
    k0 = random.choice(starts)
    ks = [k0]
    for gap in gaps:
        ks.append(ks[-1] + gap)
    return ks


def _choose_core_interval(ks: list[int]) -> tuple[int, int] | None:
    parity = ks[0] % 2
    inter_lo = max(_align_bounds(*_b_bounds_for_k(k), parity)[0] for k in ks)
    inter_hi = min(_align_bounds(*_b_bounds_for_k(k), parity)[1] for k in ks)
    inter_lo, inter_hi = _align_bounds(inter_lo, inter_hi, parity)
    if inter_lo > inter_hi:
        return None

    cell_options: list[int] = []
    for cells in range(CORE_LEN_MIN, CORE_LEN_MAX + 1):
        span = 2 * (cells - 1)
        if inter_lo + span <= inter_hi:
            cell_options.append(cells)
    if not cell_options:
        return None

    core_cells = random.choice(cell_options)
    span = 2 * (core_cells - 1)
    starts = list(range(inter_lo, inter_hi - span + 1, 2))
    if not starts:
        return None
    core_lo = random.choice(starts)
    return core_lo, core_lo + span


def _choose_covering_segment(k: int, core_lo: int, core_hi: int, min_len: int = SEG_LEN_MIN) -> tuple[int, int] | None:
    parity = k % 2
    b_lo, b_hi = _align_bounds(*_b_bounds_for_k(k), parity)
    for _ in range(BUILD_RETRY):
        length = random.randint(min_len, SEG_LEN_MAX)
        span = 2 * (length - 1)
        start_lo = max(b_lo, core_hi - span)
        start_hi = min(core_lo, b_hi - span)
        start_lo, start_hi = _align_bounds(start_lo, start_hi, parity)
        if start_lo > start_hi:
            continue
        b_min = random.choice(list(range(start_lo, start_hi + 1, 2)))
        b_max = b_min + span
        if b_min <= core_lo and core_hi <= b_max:
            return b_min, b_max
    return None


def _choose_split_plan(seg: tuple[int, int, int], core_lo: int, core_hi: int) -> tuple[tuple[int, int], tuple[int, int], list[int]] | None:
    _k, b_min, b_max = seg
    for gap_cells in (2, 1):
        gap_span = 2 * (gap_cells - 1)
        starts = []
        for gap_start in range(core_lo, core_hi - gap_span + 1, 2):
            gap_end = gap_start + gap_span
            left_cells = (gap_start - b_min) // 2
            right_cells = (b_max - gap_end) // 2
            if left_cells >= 2 and right_cells >= 2:
                starts.append(gap_start)
        if not starts:
            continue
        gap_start = random.choice(starts)
        gap_end = gap_start + gap_span
        piece1 = (b_min, gap_start - 2)
        piece2 = (gap_end + 2, b_max)
        missing_bs = list(range(gap_start, gap_end + 1, 2))
        return piece1, piece2, missing_bs
    return None


def _build_output_grid(line: int, segs: list[tuple[int, int, int]]) -> np.ndarray:
    grid = np.full((H, W), BG, dtype=int)
    for k, b_min, b_max in segs:
        for r, c in _seg_cells(k, b_min, b_max):
            grid[r, c] = line
    return _apply_fill(grid, segs, BG, FILL)


def _build_input_grid(line: int, segs: list[tuple[int, int, int]], split_idx: int, split_plan: tuple[tuple[int, int], tuple[int, int], list[int]]) -> np.ndarray:
    grid = np.full((H, W), BG, dtype=int)
    piece1, piece2, _missing = split_plan
    for idx, (k, b_min, b_max) in enumerate(segs):
        ranges = [(b_min, b_max)]
        if idx == split_idx:
            ranges = [piece1, piece2]
        for rb_min, rb_max in ranges:
            for r, c in _seg_cells(k, rb_min, rb_max):
                grid[r, c] = line
    return grid


def _gen_attempt() -> np.ndarray | None:
    line = random.choice(LINE_COLORS)
    parity = random.randint(0, 1)
    ks = _choose_k_values(parity)
    if ks is None or len(ks) < 4:
        return None

    core = _choose_core_interval(ks)
    if core is None:
        return None
    core_lo, core_hi = core

    split_idx = random.randrange(1, len(ks) - 1)
    segs: list[tuple[int, int, int]] = []
    for idx, k in enumerate(ks):
        min_len = SEG_LEN_MIN + 1 if idx == split_idx else SEG_LEN_MIN
        seg = _choose_covering_segment(k, core_lo, core_hi, min_len=min_len)
        if seg is None:
            return None
        segs.append((k, seg[0], seg[1]))

    split_plan = _choose_split_plan(segs[split_idx], core_lo, core_hi)
    if split_plan is None:
        return None

    output = _build_output_grid(line, segs)
    if int(np.sum(output == FILL)) < MIN_FILL_CELLS:
        return None

    inp = _build_input_grid(line, segs, split_idx, split_plan)
    variant_metrics = _variant_metrics(inp)
    if variant_metrics["bridge_cells"] <= 0 or variant_metrics["delta_vs_legacy"] <= 0:
        return None

    if not np.array_equal(main(inp), output):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY):
        res = _gen_attempt()
        if res is not None:
            return res
    raise RuntimeError("Failed to generate input")
