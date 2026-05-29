# Combo variant of e376de54 (index 7)
# combo_spec: D4/2+D14/2
# Name: 延伸渐变+延伸标记

import random

import numpy as np

from arc_utils import get_bg


# Design analysis
# ---
# a) 具体化与行为追踪:
#    原题先找出一组同方向平行线段，再用“垂直偏移的中位线段”作为参考，
#    把所有线段规范到同一个目标区间。原题在输出里把目标区间整段都画成线段原色，
#    所以人眼看不出哪些像素是补出来的。本融合变体保留相同几何规范，但把
#    “目标区间内原本就属于该线段的部分”保留原色，把“为了补到参考长度而新增
#    的部分”统一改成标记色 `_MARKER`。被裁掉的多余部分直接消失。
#    具体例子: 水平目标区间长度 T=6。若某短线输入为 `A A A A` 且与参考共起点，
#    输出应为 `A A A A M M`；若某短线输入为 `A A A A` 且与参考共终点，
#    输出应为 `M M A A A A`。这正是两维融合的协同点:
#    D4/2 要求“延伸像素必须显式异色”，D14/2 要求“原始区间与延伸区间能从
#    位置上区分出来”，于是输出同一条线会分成连续两段: 原色段 + 标记段。
# b) 复杂度校准:
#    标记色使用一个全局固定颜色，不按原色做隐藏映射，避免不可见规则。
#    输入仍只含 3 或 5 条线、1 到 3 种普通线色，保持原题认知负担。
#    生成器强制至少一条线变短、一条线变长，并要求输出出现至少 2 个标记像素，
#    否则样本会退化成几何题或原题行为。
# c) 隐式假设审查:
#    - 方向检测依赖“同色相邻像素重复出现”: 仍成立，因为输入线段保持纯色，
#      标记色只出现在输出，不会污染输入检测。
#    - 逐色追踪连续段默认每条输入线是单色: 仍成立，生成器保证输入不含双色线。
#    - 原 solver 清空后整段重画为原色: 已不成立，必须改成“目标区间内按是否
#      落在原始 t 区间里分色”。
#    - 原 solver 只根据起点对齐重画，所以不会表达“原始段究竟在前还是在后”。
#      融合后必须保留输入线段在目标区间中的实际占据区间，因此需要新增假设:
#      每条线段都与参考共享一个端点（同起点或同终点），这样原始区间在目标
#      区间内必然是一整段连续前缀或后缀，而不是中间悬空。
# d) 放置合法性:
#    生成器仍采用 output-first。先确定参考目标区间 `(t0, ref_len)` 和每条平行轨道，
#    再从这个合法目标状态反推输入长度。不同线段的垂直偏移互异，故不会重叠；
#    对角线仍保持至少 4 的偏移间隔，避免 8 邻域粘连成歧义形状。容量检查保证
#    无论选择共起点还是共终点，输入和输出都不会越界。
# e) 视觉可观察性:
#    2 到 4 个样例里，人类可直接看到: 中位那条线给出目标跨度；长线被裁掉；
#    短线被补齐；且补上的一侧总是统一标记色。更重要的是，若输入是共终点短线，
#    标记会出现在前端；若输入是共起点短线，标记会出现在后端。这样“延伸方向”
#    也是可见的，而不是仅仅看到换了颜色。
# f) 可视化整洁度:
#    每条线最多只有两种颜色，而且双色只出现在同一直线的连续两段中；不存在
#    覆盖、边框、噪声或多层叠加。背景、原色段、标记段三类元素边界清楚。
# g) 变化可推断性:
#    标记色是全局功能色而不是对象身份色，所以多条短线会重复使用它，为人类
#    提供足够支持去归纳“这个颜色表示延伸部分”。原色段与标记段还伴随稳定的
#    结构线索: 它们总共线、总相邻，且只在达到参考跨度时切换。
# h) 新 solver 假设与 generator 对应:
#    solver 现在断言每条输入线必须满足 `t_start == ref_t0` 或 `t_end == ref_t1`，
#    也就是输入必须共享未来的一个端点。generator 通过 `anchor in {start,end}`
#    的构造严格保证这一点，并保留参考线本身长度不变，使目标区间可唯一恢复。

_DIR_CANDIDATES = [(0, 1), (1, 0), (1, 1), (1, -1)]  # horiz, vert, diag \, diag /
_MARKER = 4


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


def _step(d: tuple[int, int]) -> int:
    dr, dc = d
    return dr * dr + dc * dc


def _cell_from_ts(
    d: tuple[int, int], t: int, s: int, shape: tuple[int, int]
) -> tuple[int, int] | None:
    h, w = shape
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
    if 0 <= r < h and 0 <= c < w:
        return r, c
    return None


def _extract_segments(
    g: np.ndarray, d: tuple[int, int], bg: int
) -> list[dict[str, int | list[tuple[int, int]] | tuple[int, int]]]:
    h, w = g.shape
    dr, dc = d
    segments = []
    colors = [int(c) for c in np.unique(g) if int(c) not in (bg, _MARKER)]
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
    return segments


def main(grid: np.ndarray) -> np.ndarray:
    """
    Normalize a family of parallel segments to the reference span while making every added extension visibly distinct.

    Input structure:
    - The background is the most frequent color.
    - Several monochrome multi-cell segments lie on distinct parallel tracks and share one dominant direction.
    - Every segment already shares either the future reference start or the future reference end, so its original occupied interval is a contiguous prefix or suffix of the target span.

    Rule:
    - Detect the common direction and choose the segment at the median perpendicular offset as the reference.
    - Use the reference span as the target interval for every segment on its own parallel line.
    - Within that target interval, cells that belonged to the input segment stay in the segment's original color, cells needed only to extend a short segment become marker color 4, and any cells outside the target interval disappear.

    Key insight: The geometry is still defined by the median-offset reference segment, but the output no longer hides extension: each shortened line becomes a two-part stripe whose marker-colored side reveals which portion had to be added.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    d = _pick_main_dir(g, bg)
    delta_t = _step(d)
    segments = _extract_segments(g, d, bg)
    assert segments, "at least one multi-cell segment must exist"

    s_vals = sorted(int(s["s"]) for s in segments)
    if delta_t == 2:
        for i in range(len(s_vals) - 1):
            assert s_vals[i + 1] - s_vals[i] >= 4
    assert len(s_vals) % 2 == 1, "segment count must have a single median"
    assert len(s_vals) == len(set(s_vals)), "segment line coordinates must be unique"

    s_ref = s_vals[len(s_vals) // 2]
    ref_cands = [s for s in segments if s["s"] == s_ref]
    assert len(ref_cands) == 1, "median reference segment must be unique"
    ref = ref_cands[0]
    ref_t0 = int(ref["t_start"])
    ref_t1 = int(ref["t_end"])
    assert ref_t1 >= ref_t0, "traced segment coordinates must increase"
    assert (ref_t1 - ref_t0) % delta_t == 0, "reference segment length must align with delta_t"

    out = g.copy()
    for s in segments:
        for r, c in s["cells"]:
            out[r, c] = bg

    for s in segments:
        seg_t0 = int(s["t_start"])
        seg_t1 = int(s["t_end"])
        assert seg_t0 == ref_t0 or seg_t1 == ref_t1, "segment must share one anchored endpoint"
        for t in range(ref_t0, ref_t1 + 1, delta_t):
            cell = _cell_from_ts(d, t, int(s["s"]), (h, w))
            assert cell is not None, "target segment cell must stay in bounds"
            r, c = cell
            out[r, c] = int(s["color"]) if seg_t0 <= t <= seg_t1 else _MARKER

    return out


# --------------- generator ---------------

_H, _W = 16, 16
_BG = 7

_DIRS = [(0, 1), (1, 0), (1, 1), (1, -1)]
_COLORS = [1, 2, 3, 5, 6, 8, 9]  # reserve _MARKER as output-only

_SEGMENT_COUNTS = [3, 5]
_REF_LEN_RANGE = (4, 8)
_MAX_SEG_LEN = 12

_PERP_GAP_ORTHO = [2, 3, 4, 5, 6]
_PERP_GAP_DIAG = [4, 6]

_MAX_DISTINCT_COLORS = 3
_SAME_LEN_AVOID_PROB = 0.75

_T_SCAN_RANGE = (-30, 31)
_GEN_RETRIES = 300


def _segment_cells(
    d: tuple[int, int], t0: int, s: int, length: int
) -> list[tuple[int, int]] | None:
    delta = _step(d)
    cells = []
    for k in range(length):
        cell = _cell_from_ts(d, t0 + k * delta, s, (_H, _W))
        if cell is None:
            return None
        cells.append(cell)
    return cells


def _max_len_from(d: tuple[int, int], t0: int, s: int) -> int:
    delta = _step(d)
    length = 0
    while _cell_from_ts(d, t0 + length * delta, s, (_H, _W)) is not None:
        length += 1
    return length


def _max_len_to(d: tuple[int, int], t1: int, s: int) -> int:
    delta = _step(d)
    length = 0
    while _cell_from_ts(d, t1 - length * delta, s, (_H, _W)) is not None:
        length += 1
    return length


def _s_limits(d: tuple[int, int]) -> tuple[int, int]:
    if d == (0, 1):
        return 0, 15
    if d == (1, 0):
        return -15, 0
    if d == (1, 1):
        return -15, 15
    return -30, 0


def _gap_patterns(d: tuple[int, int], n: int) -> list[list[int]]:
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


def _choose_offsets(d: tuple[int, int], n: int) -> tuple[int, ...]:
    lo, hi = _s_limits(d)
    gaps = random.choice(_gap_patterns(d, n))
    s0 = random.randint(lo, hi - sum(gaps))
    s_vals = [s0]
    for gap in gaps:
        s_vals.append(s_vals[-1] + gap)
    return tuple(s_vals)


def _feasible_placements(
    d: tuple[int, int], s_vals: tuple[int, ...], ref_len: int
) -> list[tuple[int, str, list[int]]]:
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


def _choose_colors(n: int) -> tuple[int, ...]:
    n_colors = random.randint(1, min(_MAX_DISTINCT_COLORS, n))
    palette = random.sample(_COLORS, n_colors)
    colors = [random.choice(palette) for _ in range(n)]
    while len(set(colors)) < n_colors:
        colors[random.randrange(n)] = random.choice(palette)
    return tuple(colors)


def _choose_lengths(ref_len: int, caps: list[int], n: int) -> list[int] | None:
    ref_idx = n // 2
    lengths = [ref_len] * n
    nonrefs = [i for i in range(n) if i != ref_idx]
    longable = [i for i in nonrefs if caps[i] > ref_len]
    if not longable:
        return None

    long_i = random.choice(longable)
    short_i = random.choice([i for i in nonrefs if i != long_i])
    lengths[long_i] = random.randint(ref_len + 1, min(_MAX_SEG_LEN, caps[long_i]))

    short_hi = max(2, ref_len - 2)
    lengths[short_i] = random.randint(2, short_hi)

    for i in nonrefs:
        if i in (long_i, short_i):
            continue
        options = list(range(2, min(_MAX_SEG_LEN, caps[i]) + 1))
        if len(options) > 1 and ref_len in options and random.random() < _SAME_LEN_AVOID_PROB:
            options.remove(ref_len)
        lengths[i] = random.choice(options)
    return lengths


def _paint_input(
    grid: np.ndarray,
    d: tuple[int, int],
    s_vals: tuple[int, ...],
    starts: list[int],
    lengths: list[int],
    colors: tuple[int, ...],
) -> bool:
    for s, t0, length, color in zip(s_vals, starts, lengths, colors):
        cells = _segment_cells(d, t0, s, length)
        if cells is None:
            return False
        for r, c in cells:
            if grid[r, c] != _BG:
                return False
            grid[r, c] = color
    return True


def _build_output(
    d: tuple[int, int],
    s_vals: tuple[int, ...],
    t0: int,
    ref_len: int,
    starts: list[int],
    lengths: list[int],
    colors: tuple[int, ...],
) -> np.ndarray | None:
    out = np.full((_H, _W), _BG, dtype=int)
    delta = _step(d)
    target_ts = [t0 + k * delta for k in range(ref_len)]
    for s, start, length, color in zip(s_vals, starts, lengths, colors):
        t_end = start + (length - 1) * delta
        for t in target_ts:
            cell = _cell_from_ts(d, t, s, (_H, _W))
            if cell is None:
                return None
            r, c = cell
            out[r, c] = color if start <= t <= t_end else _MARKER
    return out


def _gen_attempt() -> np.ndarray | None:
    d = random.choice(_DIRS)
    n = random.choice(_SEGMENT_COUNTS)
    ref_len = random.randint(*_REF_LEN_RANGE)
    s_vals = _choose_offsets(d, n)
    placements = _feasible_placements(d, s_vals, ref_len)
    if not placements:
        return None

    t0, anchor, caps = random.choice(placements)
    colors = _choose_colors(n)
    lengths = _choose_lengths(ref_len, caps, n)
    if lengths is None:
        return None
    if not any(length < ref_len for length in lengths):
        return None
    if not any(length > ref_len for length in lengths):
        return None

    delta = _step(d)
    t1 = t0 + (ref_len - 1) * delta
    if anchor == "start":
        starts = [t0] * n
    else:
        starts = [t1 - (length - 1) * delta for length in lengths]

    inp = np.full((_H, _W), _BG, dtype=int)
    if not _paint_input(inp, d, s_vals, starts, lengths, colors):
        return None

    out = _build_output(d, s_vals, t0, ref_len, starts, lengths, colors)
    if out is None:
        return None
    if int(np.sum(out == _MARKER)) < 2:
        return None

    try:
        if np.array_equal(main(inp), out):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
