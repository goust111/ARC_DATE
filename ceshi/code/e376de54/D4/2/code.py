# Variant of e376de54
# Dimension: D4 — 变换：延伸或裁剪线段 (transform)
# Change: 短于参考的线段延伸像素到参考长度，长于参考的线段裁剪多余像素... → 延伸时颜色渐变标记

import random

import numpy as np

from arc_utils import get_bg


# Design analysis
# ---
# variant_spec:
#   task_core:
#     keep:
#       - 仍然只有一组共线方向一致的平行线段
#       - 仍然用垂直于线段方向的中位线作为参考线
#       - 仍然把所有线段的起点对齐到参考线起点，再统一成参考长度
#       - 长于参考的线段仍然只做裁剪，不改颜色
#     change:
#       - 只有“被补出来的新像素”改成统一标记色 `_EXT_MARK`
#       - 输入中不出现该标记色；输出里它只出现在短线段被补长后的尾部
#   concretization_and_trace:
#     input_picture:
#       - 16x16 背景 7
#       - 3 或 5 条同方向线段，颜色来自普通调色板，不包含标记色
#       - 每条线段仍是纯色，且共享同一端点模式（同起点或同终点）
#     output_picture:
#       - 所有线段仍然被平移到同一起点，并统一到参考长度
#       - 若某线原长 L < 参考长 T，则输出该线前 L 格保持原色，后 T-L 格改成标记色
#       - 若 L >= T，则输出前 T 格全部保持原色；超过部分被裁掉
#     trace_example:
#       - 水平参考长 T=5，短线原长 L=3，输入三格是蓝色
#       - 原题输出：五格全部蓝色
#       - 本变体输出：前 3 格蓝色，新增的后 2 格是黄色标记
#       - 差异只落在“新增尾段的颜色”，几何对齐方式完全不变
#   complexity_calibration:
#     - 不引入按颜色 id 计算的映射，避免不可见规则
#     - 标记色固定且全局唯一，2~4 个样例内人类可直接观察到“只有延伸部分会变色”
#     - 仍保留至少一个短线和一个长线，保证同一题里同时出现“延伸+裁剪”两种分支
#   implicit_assumption_audit:
#     original_solver_dependencies:
#       - 逐色扫描连续像素，默认输入线段是纯色: 继续成立，因为输入保持纯色
#       - 清空旧线段后整段重画为原色: 不再成立，必须改为“原有长度部分原色 + 新增部分标记色”
#       - 参考段长度由参考线自身给出: 继续成立
#       - 线段按起点对齐后写入不会互相覆盖: 继续成立，因为垂直坐标仍互异
#     new_solver_dependencies:
#       - 输入不含标记色线段，否则输出中“原色 vs 新增色”区分会变弱
#       - 每个短线段的新增像素总是出现在对齐后线段的尾部
#     generator_support:
#       - 颜色池排除 `_EXT_MARK`
#       - 仍强制至少一短一长，保证变体行为真实发生而非退化回原题
#   placement_legality:
#     - 几何放置沿用原题的 output-first 搜索，先保证参考长度版本整体可放入网格
#     - 输入端只改变各线长度和锚定端，不改变垂直偏移，因此不会新增交叉
#     - 对角线仍要求垂直间距 >= 4，避免同色或异色线段在 8 邻域里粘连成歧义结构
#   visual_observability:
#     - 输入没有标记色，输出中标记色只出现在被补长的尾段，位置规律稳定
#     - 长线裁剪后仍保持原色，和短线延伸后的双色尾巴形成直接对照
#   cleanliness:
#     - 同一条线上最多两种颜色，且变化只发生在尾端，视觉上整洁
#     - 不叠加额外标记、边框或噪声，避免掩盖主规则
#   color_inference:
#     - 标记色是全局固定功能色，不由原色数值计算而来
#     - 样例间可重复出现相同标记色，足以支持人类归纳“黄色 = 新增段”

_DIR_CANDIDATES = [(0, 1), (1, 0), (1, 1), (1, -1)]  # horiz, vert, diag \, diag /

# Unified extension marker: generator reserves it so it only appears in output.
_EXT_MARK = 4


def _pick_main_dir(g: np.ndarray, bg: int) -> tuple[int, int]:
    counts = []
    h, w = g.shape
    for dr, dc in _DIR_CANDIDATES:
        cnt = 0
        for r in range(h):
            for c in range(w):
                v = int(g[r, c])
                if v == bg:
                    continue
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and int(g[rr, cc]) == v:
                    cnt += 1
        counts.append(((dr, dc), cnt))

    best_cnt = max(cnt for _, cnt in counts)
    assert best_cnt > 0, "main direction must be detectable"
    best_dirs = [d for d, cnt in counts if cnt == best_cnt]
    assert len(best_dirs) == 1, "main direction must be unique"
    return best_dirs[0]


def _dot(p: tuple[int, int], d: tuple[int, int]) -> int:
    return p[0] * d[0] + p[1] * d[1]


def _perp_dot(p: tuple[int, int], d: tuple[int, int]) -> int:
    # perp to (dr,dc) can be (dc,-dr); dot(p, perp) = r*dc - c*dr
    return p[0] * d[1] - p[1] * d[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Align parallel line segments to a reference span, but tint newly extended cells with a marker color.

    Input structure:
    - The background is the most frequent color.
    - Several monochrome line segments lie on distinct parallel tracks, all sharing one dominant direction.
    - Each segment is at least 2 cells long; stray single pixels, if any, are not part of the transformation.

    Rule:
    - Detect the shared direction and choose the segment at the median perpendicular offset as the reference.
    - Translate every segment so its start matches the reference start, then force the reference length.
    - Cells that come from the original shifted segment keep their source color; only extra cells added to lengthen a short segment become the marker color.
    - Segments longer than the reference are simply truncated, and the grid size stays unchanged.

    Key insight: Geometry is still normalized exactly as in the base task; the new signal is that extension is visible because appended tail cells are recolored.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    d = _pick_main_dir(g, bg)
    dr, dc = d
    delta_t = dr * dr + dc * dc

    segments = []
    colors = [int(c) for c in np.unique(g) if int(c) != bg]
    for col in colors:
        mask = g == col
        for r in range(h):
            for c in range(w):
                if not mask[r, c]:
                    continue
                pr, pc = r - dr, c - dc
                if 0 <= pr < h and 0 <= pc < w and mask[pr, pc]:
                    continue
                cells = []
                rr, cc = r, c
                while 0 <= rr < h and 0 <= cc < w and mask[rr, cc]:
                    cells.append((rr, cc))
                    rr += dr
                    cc += dc
                if len(cells) < 2:
                    continue
                start = cells[0]
                end = cells[-1]
                segments.append(
                    {
                        "color": col,
                        "cells": cells,
                        "start": start,
                        "end": end,
                        "t_start": _dot(start, d),
                        "t_end": _dot(end, d),
                        "s": _perp_dot(start, d),
                        "len": len(cells),
                    }
                )

    assert segments, "at least one multi-cell segment must exist"

    s_vals = sorted(s["s"] for s in segments)
    if delta_t == 2:
        for i in range(len(s_vals) - 1):
            assert s_vals[i + 1] - s_vals[i] >= 4
    assert len(s_vals) % 2 == 1, "segment count must have a single median"
    assert len(s_vals) == len(set(s_vals)), "segment line coordinates must be unique"
    s_ref = s_vals[len(s_vals) // 2]
    ref_cands = [s for s in segments if s["s"] == s_ref]
    assert len(ref_cands) == 1, "median reference segment must be unique"
    ref = ref_cands[0]
    ref_t0, ref_t1 = ref["t_start"], ref["t_end"]
    assert ref_t1 >= ref_t0, "traced segment coordinates must increase"
    assert (ref_t1 - ref_t0) % delta_t == 0, "reference segment length must align with delta_t"
    target_len = (ref_t1 - ref_t0) // delta_t + 1

    out = g.copy()

    for s in segments:
        for r, c in s["cells"]:
            out[r, c] = bg

    for s in segments:
        col = s["color"]
        start = s["start"]
        dt = ref_t0 - s["t_start"]
        assert dt % delta_t == 0, "segment shift must be aligned to delta_t"
        k = dt // delta_t
        r0 = start[0] + k * dr
        c0 = start[1] + k * dc
        original_keep = min(s["len"], target_len)
        rr, cc = r0, c0
        for idx in range(target_len):
            assert 0 <= rr < h and 0 <= cc < w, "shifted segment must stay in bounds"
            out[rr, cc] = col if idx < original_keep else _EXT_MARK
            rr += dr
            cc += dc

    return out


# --------------- generator (o5x) ---------------

# --- Canvas ---
_H, _W = 16, 16
_BG = 7

# --- Directions and colors ---
_DIRS = [(0, 1), (1, 0), (1, 1), (1, -1)]
_COLORS = [1, 2, 3, 5, 6, 8, 9]  # reserve _EXT_MARK so it is output-only

# --- Segment counts and reference length ---
_SEGMENT_COUNTS = [3, 5]
_REF_LEN_RANGE = (3, 8)
_MAX_SEG_LEN = 12

# --- Perpendicular gap constraints ---
_PERP_GAP_ORTHO = [2, 3, 4, 5, 6]
_PERP_GAP_DIAG = [4, 6]

# --- Color assignment ---
_MAX_DISTINCT_COLORS = 3
_SAME_LEN_AVOID_PROB = 0.75

# --- Search / retry ---
_T_SCAN_RANGE = (-30, 31)
_GEN_RETRIES = 300


def _step(d):
    """Dot product of direction with itself (1 for ortho, 2 for diag)."""
    dr, dc = d
    return dr * dr + dc * dc


def _cell_from_ts(d, t, s):
    """Convert (t, s) coordinates back to (row, col) for direction d. Returns None if out of bounds."""
    if d == (0, 1):
        r, c = s, t
    elif d == (1, 0):
        r, c = t, -s
    elif d == (1, 1):
        if (t + s) % 2 or (t - s) % 2:
            return None
        r, c = (t + s) // 2, (t - s) // 2
    else:
        if (t - s) % 2 or (-s - t) % 2:
            return None
        r, c = (t - s) // 2, (-s - t) // 2
    if 0 <= r < _H and 0 <= c < _W:
        return r, c
    return None


def _segment_cells(d, t0, s, length):
    """Return list of (r, c) for a segment starting at t0 with given perpendicular offset s, or None if any cell is OOB."""
    delta = _step(d)
    cells = []
    for k in range(length):
        cell = _cell_from_ts(d, t0 + k * delta, s)
        if cell is None:
            return None
        cells.append(cell)
    return cells


def _max_len_from(d, t0, s):
    """Max segment length extending forward from t0 at perpendicular offset s."""
    delta = _step(d)
    length = 0
    while _cell_from_ts(d, t0 + length * delta, s) is not None:
        length += 1
    return length


def _max_len_to(d, t1, s):
    """Max segment length extending backward from t1 at perpendicular offset s."""
    delta = _step(d)
    length = 0
    while _cell_from_ts(d, t1 - length * delta, s) is not None:
        length += 1
    return length


def _s_limits(d):
    """Valid range (lo, hi) inclusive for perpendicular offset s given direction d."""
    if d == (0, 1):
        return 0, 15
    if d == (1, 0):
        return -15, 0
    if d == (1, 1):
        return -15, 15
    return -30, 0


def _gap_patterns(d, n):
    """Enumerate all valid perpendicular gap sequences of length n-1, ensuring irregular spacing."""
    allowed = _PERP_GAP_DIAG if d in [(1, 1), (1, -1)] else _PERP_GAP_ORTHO
    lo, hi = _s_limits(d)
    max_span = hi - lo
    patterns = [[]]
    for _ in range(n - 1):
        patterns = [p + [g] for p in patterns for g in allowed]
    return [
        p
        for p in patterns
        if sum(p) <= max_span and (n == 3 and p[0] != p[1] or n == 5 and len(set(p)) > 1)
    ]


def _choose_offsets(d, n):
    """Pick n distinct perpendicular offsets with irregular gaps."""
    lo, hi = _s_limits(d)
    gaps = random.choice(_gap_patterns(d, n))
    s0 = random.randint(lo, hi - sum(gaps))
    s_vals = [s0]
    for gap in gaps:
        s_vals.append(s_vals[-1] + gap)
    return tuple(s_vals)


def _feasible_placements(d, s_vals, ref_len):
    """Find all (t0, anchor_mode, capacity_list) where segments can be placed with room for length variation."""
    ref_idx = len(s_vals) // 2
    placements = []
    for t0 in range(*_T_SCAN_RANGE):
        if any(_segment_cells(d, t0, s, ref_len) is None for s in s_vals):
            continue
        t1 = t0 + (ref_len - 1) * _step(d)
        start_caps = [_max_len_from(d, t0, s) for s in s_vals]
        if any(start_caps[i] > ref_len for i in range(len(s_vals)) if i != ref_idx):
            placements.append((t0, "start", start_caps))
        end_caps = [_max_len_to(d, t1, s) for s in s_vals]
        if any(end_caps[i] > ref_len for i in range(len(s_vals)) if i != ref_idx):
            placements.append((t0, "end", end_caps))
    return placements


def _choose_colors(s_vals):
    """Assign 1-3 distinct non-bg, non-marker colors across segments, ensuring all chosen colors appear."""
    n_colors = random.randint(1, min(_MAX_DISTINCT_COLORS, len(s_vals)))
    palette = random.sample(_COLORS, n_colors)
    colors = [random.choice(palette) for _ in s_vals]
    while len(set(colors)) < n_colors:
        colors[random.randrange(len(colors))] = random.choice(palette)
    return tuple(colors)


def _choose_lengths(ref_len, caps, n):
    """Choose input lengths with at least one shorter and one longer line. Returns None if infeasible."""
    ref_idx = n // 2
    lengths = [ref_len] * n
    nonrefs = [i for i in range(n) if i != ref_idx]
    longable = [i for i in nonrefs if caps[i] > ref_len]
    if not longable:
        return None

    long_i = random.choice(longable)
    short_i = random.choice([i for i in nonrefs if i != long_i])
    lengths[long_i] = random.randint(ref_len + 1, min(_MAX_SEG_LEN, caps[long_i]))

    short_hi = ref_len - 1
    if ref_len >= 4 and random.random() < 0.8:
        short_hi = ref_len - 2
    short_hi = max(2, short_hi)
    lengths[short_i] = random.randint(2, short_hi)

    for i in nonrefs:
        if i in (long_i, short_i):
            continue
        options = list(range(2, min(_MAX_SEG_LEN, caps[i]) + 1))
        if len(options) > 1 and ref_len in options and random.random() < _SAME_LEN_AVOID_PROB:
            options.remove(ref_len)
        lengths[i] = random.choice(options)
    return lengths


def _paint(grid, d, s_vals, starts, lengths, colors):
    """Render monochrome input segments onto grid. Returns False if any segment cell is OOB."""
    for s, t0, length, color in zip(s_vals, starts, lengths, colors):
        cells = _segment_cells(d, t0, s, length)
        if cells is None:
            return False
        for r, c in cells:
            grid[r, c] = color
    return True


def _paint_variant_output(grid, d, s_vals, t0, ref_len, lengths, colors):
    """Render the expected output with original-color prefix + marker-color extension suffix."""
    for s, orig_len, color in zip(s_vals, lengths, colors):
        cells = _segment_cells(d, t0, s, ref_len)
        if cells is None:
            return False
        original_keep = min(orig_len, ref_len)
        for idx, (r, c) in enumerate(cells):
            grid[r, c] = color if idx < original_keep else _EXT_MARK
    return True


def _gen_attempt():
    """Build one candidate (input, output) pair; returns input grid or None on failure."""
    d = random.choice(_DIRS)
    n = random.choice(_SEGMENT_COUNTS)
    ref_len = random.randint(*_REF_LEN_RANGE)
    s_vals = _choose_offsets(d, n)
    placements = _feasible_placements(d, s_vals, ref_len)
    if not placements:
        return None

    t0, anchor, caps = random.choice(placements)
    colors = _choose_colors(s_vals)
    lengths = _choose_lengths(ref_len, caps, n)
    if lengths is None:
        return None
    if not any(length < ref_len for length in lengths):
        return None

    out = np.full((_H, _W), _BG, dtype=int)
    if not _paint_variant_output(out, d, s_vals, t0, ref_len, lengths, colors):
        return None

    delta = _step(d)
    t1 = t0 + (ref_len - 1) * delta
    if anchor == "start":
        starts = [t0] * n
    else:
        starts = [t1 - (length - 1) * delta for length in lengths]

    inp = np.full((_H, _W), _BG, dtype=int)
    if not _paint(inp, d, s_vals, starts, lengths, colors):
        return None

    try:
        if np.array_equal(main(inp), out):
            return inp
    except Exception:
        return None
    return None


def generate_input():
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
