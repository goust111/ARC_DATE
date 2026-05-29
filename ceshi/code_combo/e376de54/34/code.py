# Combo variant of e376de54 (index 34)
# combo_spec: D4/2+D14/1
# Name: 延伸渐变+裁剪标记

import random

import numpy as np

from arc_utils import Color, get_bg


"""
design_analysis:
  fused_rule:
    keep:
      - 原题的对象结构不变: 16x16 背景上有 3 或 5 条同方向平行线段
      - 参考线仍由垂直偏移 s 的中位数决定
      - 所有线段仍统一到参考线的起点与参考长度
    change:
      - 对齐后保留下来的原有部分维持原色
      - 短线被补出来的新增尾段改成统一扩展标记色 yellow
      - 长线被删掉的原始多余像素不消失，而是在原坐标改成统一警示色 red
      - 输出同时显示“保留 / 延伸 / 裁剪”三类信息，允许反推完整变换过程
  a_concretization_and_trace:
    example:
      - 设共享方向为水平，参考长度 T=5，参考起点列为 4
      - 一条短线输入长 3，若对齐后覆盖输出列 4..6，则列 4..6 保持原色，列 7..8 变黄
      - 一条长线输入长 7，若原始占据列 2..8，而最终目标是列 4..8，则列 4..8 保持原色，列 2..3 变红
      - 原题里黄段不存在、红段也不存在；融合后这两类痕迹同时可见
    consequence:
      - 对 shared-start 输入，裁剪痕迹会留在旧尾部，延伸标记出现在新尾部
      - 对 shared-end 输入，裁剪痕迹会留在旧前缀，延伸标记仍出现在对齐后线段的尾部
  b_complexity_calibration:
    - 颜色功能只增加两个全局固定语义色，不引入按原色计算的隐藏映射
    - 输入继续只含普通线段色，功能色仅出现在输出，便于人类从少量样例归纳
    - 生成器强制至少一短一长，因此每题都同时出现黄段与红段，不退化为单变体
  c_implicit_assumption_audit:
    - 原 solver 逐色追踪连续像素，依赖输入线段是纯色: 仍成立，生成器禁止输入含功能色
    - 原 solver 清空旧段后整段重画为原色: 已失效；现在必须分三步处理:
      1. 清空原段
      2. 把被裁掉的 original-final 差集改成 red
      3. 在目标区间内写入“原色前缀 + yellow 扩展尾段”
    - 原 solver 默认输出只需表达最终几何: 已失效；现在输出还要编码历史轨迹
    - 新增假设是 red 与 yellow 都必须和背景、普通线段色区分，生成器据此保留这两色为 output-only
  d_placement_legality:
    - 延续 output-first: 先找能容纳所有目标线段的参考区间，再从它反推输入
    - 线段始终位于不同的垂直偏移 s 上，因此最终段与裁剪轨迹都只落在各自轨道内
    - trail = original_cells - final_cells，天然不越界；extension 只发生在已验证可放置的目标区间内
  e_visual_observability:
    - 人类能直接看到: 原色表示保留，黄段只在短线目标尾部出现，红段只在长线旧位置残留
    - 同一题里同时出现 short 和 long，可横向对照“哪部分新增、哪部分被切掉”
    - 规则不依赖颜色编号大小、扫描顺序或其他不可视因素
  f_visual_cleanliness:
    - 继续使用原题的 offset gap 约束，尤其对角线要求间距 >= 4，避免红黄标记粘连
    - 每条轨道最多出现三种语义颜色，但它们在线性顺序上稳定: red 只在旧多余部分，yellow 只在新尾部
  g_inference_of_change:
    - yellow 与 red 的含义由结构位置共同提示，而不是孤立颜色替换
    - 黄色总贴着对齐后线段的尾部，红色总留在输入长线被舍弃的位置，2-4 个样例足以归纳
  new_main_assumptions_to_generator:
    - 输入不得使用 red/yellow 作为普通线段色
    - 必须存在唯一主方向、唯一中位参考线、且参考长度可在网格内完整重画
    - 至少一条线 short 于参考、至少一条线 long 于参考，从而让两种新增语义都真正触发
"""

_DIR_CANDIDATES = [(0, 1), (1, 0), (1, 1), (1, -1)]  # horiz, vert, diag \, diag /
_EXT_MARK = int(Color.YELLOW)
_TRAIL_COLOR = int(Color.RED)


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
    return p[0] * d[1] - p[1] * d[0]


def _aligned_start(start: tuple[int, int], t_start: int, ref_t0: int, d: tuple[int, int]) -> tuple[int, int]:
    dr, dc = d
    delta_t = dr * dr + dc * dc
    dt = ref_t0 - t_start
    assert dt % delta_t == 0, "segment shift must be aligned to delta_t"
    k = dt // delta_t
    return start[0] + k * dr, start[1] + k * dc


def _trace_segments(g: np.ndarray, bg: int, d: tuple[int, int]) -> list[dict]:
    h, w = g.shape
    dr, dc = d
    segments = []
    colors = [int(c) for c in np.unique(g) if int(c) != bg]
    assert _EXT_MARK not in colors, "input must reserve yellow for extension markings"
    assert _TRAIL_COLOR not in colors, "input must reserve red for trim markings"
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
    return segments


def _reference_segment(segments: list[dict], delta_t: int) -> tuple[dict, int]:
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
    return ref, target_len


def main(grid: np.ndarray) -> np.ndarray:
    """
    Align all parallel segments to the reference span while exposing both extension and trimming as explicit colored evidence.

    Input structure:
    - The most frequent color is the background, and every transformable object is a monochrome multi-cell segment.
    - All such segments lie on distinct parallel tracks with one shared detectable direction.
    - Yellow and red are reserved as output-only functional colors, so they do not appear in the input segments.

    Rule:
    - Detect the common direction and choose the segment at the median perpendicular offset as the reference.
    - Shift each segment so its start matches the reference start, then force the reference length.
    - Within that target interval, cells covered by the segment's original shifted span keep the original color, while newly added tail cells turn yellow.
    - Any original cells that do not survive because the segment was too long remain visible at their old positions in red; isolated single pixels stay untouched.

    Key insight: The geometry still normalizes every line to the median reference segment, but the output now records the whole edit history: original-color cells were kept, yellow cells were appended, and red cells were trimmed away.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = int(get_bg(g))
    assert _EXT_MARK != bg and _TRAIL_COLOR != bg, "marker colors must differ from background"

    d = _pick_main_dir(g, bg)
    dr, dc = d
    delta_t = dr * dr + dc * dc

    segments = _trace_segments(g, bg, d)
    ref, target_len = _reference_segment(segments, delta_t)
    ref_t0 = ref["t_start"]

    out = g.copy()
    for s in segments:
        for r, c in s["cells"]:
            out[r, c] = bg

    for s in segments:
        r0, c0 = _aligned_start(s["start"], s["t_start"], ref_t0, d)

        final_cells = []
        rr, cc = r0, c0
        for _ in range(target_len):
            assert 0 <= rr < h and 0 <= cc < w, "shifted segment must stay in bounds"
            final_cells.append((rr, cc))
            rr += dr
            cc += dc

        final_set = set(final_cells)
        for cell in s["cells"]:
            if cell not in final_set:
                out[cell] = _TRAIL_COLOR

        original_keep = min(s["len"], target_len)
        for idx, (r, c) in enumerate(final_cells):
            out[r, c] = s["color"] if idx < original_keep else _EXT_MARK

    return out


# --------------- generator (combo-34) ---------------

_H, _W = 16, 16
_BG = 7

_DIRS = [(0, 1), (1, 0), (1, 1), (1, -1)]
_COLORS = [1, 3, 5, 6, 8, 9]

_SEGMENT_COUNTS = [3, 5]
_REF_LEN_RANGE = (3, 8)
_MAX_SEG_LEN = 12

_PERP_GAP_ORTHO = [2, 3, 4, 5, 6]
_PERP_GAP_DIAG = [4, 6]

_MAX_DISTINCT_COLORS = 3
_SAME_LEN_AVOID_PROB = 0.75

_T_SCAN_RANGE = (-30, 31)
_GEN_RETRIES = 300


def _step(d):
    dr, dc = d
    return dr * dr + dc * dc


def _cell_from_ts(d, t, s):
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
    delta = _step(d)
    cells = []
    for k in range(length):
        cell = _cell_from_ts(d, t0 + k * delta, s)
        if cell is None:
            return None
        cells.append(cell)
    return cells


def _max_len_from(d, t0, s):
    delta = _step(d)
    length = 0
    while _cell_from_ts(d, t0 + length * delta, s) is not None:
        length += 1
    return length


def _max_len_to(d, t1, s):
    delta = _step(d)
    length = 0
    while _cell_from_ts(d, t1 - length * delta, s) is not None:
        length += 1
    return length


def _s_limits(d):
    if d == (0, 1):
        return 0, 15
    if d == (1, 0):
        return -15, 0
    if d == (1, 1):
        return -15, 15
    return -30, 0


def _gap_patterns(d, n):
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
    lo, hi = _s_limits(d)
    gaps = random.choice(_gap_patterns(d, n))
    s0 = random.randint(lo, hi - sum(gaps))
    s_vals = [s0]
    for gap in gaps:
        s_vals.append(s_vals[-1] + gap)
    return tuple(s_vals)


def _feasible_placements(d, s_vals, ref_len):
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
    n_colors = random.randint(1, min(_MAX_DISTINCT_COLORS, len(s_vals)))
    palette = random.sample(_COLORS, n_colors)
    colors = [random.choice(palette) for _ in s_vals]
    while len(set(colors)) < n_colors:
        colors[random.randrange(len(colors))] = random.choice(palette)
    return tuple(colors)


def _choose_lengths(ref_len, caps, n):
    ref_idx = n // 2
    lengths = [ref_len] * n
    nonrefs = [i for i in range(n) if i != ref_idx]
    longable = [i for i in nonrefs if caps[i] > ref_len]
    if not longable or ref_len <= 2:
        return None

    long_i = random.choice(longable)
    short_i = random.choice([i for i in nonrefs if i != long_i])

    long_lo = ref_len + 1
    if caps[long_i] >= ref_len + 2 and random.random() < 0.8:
        long_lo = ref_len + 2
    lengths[long_i] = random.randint(long_lo, min(_MAX_SEG_LEN, caps[long_i]))

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

    if not any(length < ref_len for length in lengths):
        return None
    if not any(length > ref_len for length in lengths):
        return None
    return lengths


def _paint(grid, d, s_vals, starts, lengths, colors):
    for s, t0, length, color in zip(s_vals, starts, lengths, colors):
        cells = _segment_cells(d, t0, s, length)
        if cells is None:
            return False
        for r, c in cells:
            grid[r, c] = color
    return True


def _build_output(d, s_vals, starts, lengths, colors, ref_t0, ref_len):
    out = np.full((_H, _W), _BG, dtype=int)
    delta = _step(d)
    for s, start, length, color in zip(s_vals, starts, lengths, colors):
        original = _segment_cells(d, start, s, length)
        final = _segment_cells(d, ref_t0, s, ref_len)
        if original is None or final is None:
            return None
        final_set = set(final)
        for r, c in original:
            if (r, c) not in final_set:
                out[r, c] = _TRAIL_COLOR
        original_keep = min(length, ref_len)
        for idx, (r, c) in enumerate(final):
            out[r, c] = color if idx < original_keep else _EXT_MARK
        assert len(final) == ref_len
        assert delta in (1, 2)
    return out


def _gen_attempt():
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

    delta = _step(d)
    t1 = t0 + (ref_len - 1) * delta
    if anchor == "start":
        starts = [t0] * n
    else:
        starts = [t1 - (length - 1) * delta for length in lengths]

    inp = np.full((_H, _W), _BG, dtype=int)
    if not _paint(inp, d, s_vals, starts, lengths, colors):
        return None

    out = _build_output(d, s_vals, starts, lengths, colors, t0, ref_len)
    if out is None:
        return None
    if int(np.count_nonzero(out == _EXT_MARK)) == 0:
        return None
    if int(np.count_nonzero(out == _TRAIL_COLOR)) == 0:
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
