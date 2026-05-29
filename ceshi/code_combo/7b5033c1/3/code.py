# Combo variant of 7b5033c1 (index 3)
# combo_spec: D1/2+D2/1
# Name: 颜色段长度起点+游程编码
#
# Design analysis
# a) 具体化与行为追踪:
#    输入仍是一条 4 连通、无分叉的多色蛇形路径，颜色沿路径以连续段出现。
#    原题把路径从“左上端点”开始逐像素拉直成单列。
#    本融合版先改起点，再改输出形式:
#    - 先比较两个端点各自所在颜色段的长度，从较短段对应的端点开始遍历；
#    - 再把遍历到的颜色序列做标准 run-length encoding，输出 Kx2 矩阵 [颜色, 连续次数]。
#    具体小例子:
#    若从左上端点到另一端的路径颜色是
#    [7,7,7,7,1,1,8,8,8,6,6]
#    原题输出是 11x1 单列:
#    [7,7,7,7,1,1,8,8,8,6,6]^T
#    本融合版先看两端段长: 左端是 4，右端是 2，所以要从右端开始。
#    反向遍历颜色序列变成
#    [6,6,8,8,8,1,1,7,7,7,7]
#    再做 RLE，输出:
#    [[6,2],
#     [8,3],
#     [1,2],
#     [7,4]]
#    差异不是简单“反过来”或简单“计数”，而是方向规则和编码方式同时生效。
# b) 复杂度校准:
#    为了让人类从少量样例看出第二列是“段长”而不是任意颜色，生成器把所有段长限制在 2/3/4/5 四档，
#    这样右列只会反复出现少数几种颜色。与此同时，输入路径颜色只从 {0,1,6,7,8,9} 中选，
#    刻意避开 2/3/4/5，使输出右列形成稳定的小字典，视觉上更容易被识别为计数列。
#    另外，每个样例至少包含 3 种不同段长，并且至少一种段长重复出现，避免“每行都不重复”而难以归纳。
# c) 隐式假设审查:
#    - `get_bg(g)` 依赖背景色是全图众数；路径很稀疏，生成器继续保证这一点。
#    - `g != bg` 依赖所有非背景像素都属于同一条蛇；生成器只放置这一条对象。
#    - 端点检测依赖对象是简单链、没有分叉和环；路径生成时禁止额外 4 邻接。
#    - 从任一端点出发都能唯一追踪全路径；简单链前提保持成立。
#    - “端点段长”不再能由几何位置决定，而必须由路径顺序上的前缀/后缀同色 run 决定；
#      因此 solver 不能再直接 `start = min(ends)`。
#    - RLE 输出把第二列解释为段长数字，这隐含要求段长落在 ARC 颜色域 0..9 内；
#      生成器把所有段长限制到 2..5 来满足该新前提。
#    - 若两个端点段长刚好相等，融合规则本身会有方向歧义；solver 用“左上端点优先”作为稳定平局规则，
#      生成器则尽量避免产生平局样本。
# d) 放置合法性:
#    路径仍在 16x16 网格内部逐段延伸。每个新增像素都检查:
#    - 不越界；
#    - 不与已有路径重叠；
#    - 除前驱外，不与已有路径形成额外 4 邻接。
#    因此最终对象始终是一条可被 solver 唯一追踪的简单蛇形路径。
#    着色只覆盖已生成的路径坐标，不会引入新的几何冲突。
# e) 视觉可观察性:
#    观察样例时，人类能同时看到两件事:
#    - 输出不再是高单列，而是更短的双列表；
#    - 输出首行对应的不是左上端点那一侧，而是颜色块更短的那一端。
#    生成器强制“较短端点 != 左上端点”，并要求融合输出与“旧方向 + RLE”输出不同，
#    从而避免训练样例退化成只看出其中一个变化。
# f) 可视化整洁度:
#    输入里仍只有一条蛇，没有叠加标记、辅助符号或多对象竞争同一区域，视觉负担很低。
#    输出采用标准两列矩阵，左列表示颜色、右列表示计数，列语义稳定且不混叠。
# g) 颜色变化可推断性:
#    这里没有复杂颜色映射。左列直接继承原路径颜色；右列直接等于段长。
#    为了让“右列是计数”可归纳，生成器限制段长档位少、并把输入色板与计数色板分开，
#    使第二列呈现出明显的结构性重复。
# h) 融合行为量化指标:
#    关键指标不是 `main(input)` 是否成功，而是“融合方向规则是否真的改变了 RLE 输出”。
#    下面生成器内置两层过滤:
#    - `short_endpoint_is_not_top_left == True`
#    - `combo_output != old_top_left_rle_output`
#    只有同时满足，样本才被接受。
# i) 对新 `main()` 的假设审查:
#    - `_trace_path` 假设每个中间点恰有一个“除前驱外的后继”；生成器用简单链保证。
#    - `_endpoint_run_lengths` 假设路径颜色按连续色段着色，且相邻色段颜色不同；生成器构造时强制相邻段异色。
#    - `_encode_runs` 假设路径长度有限，且 run 可顺序累积；这里没有分支、回环或多对象，因此成立。

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

for helper_dir in (
    Path("/home/lixinyu/snap/new/arc2_solver_select"),
    Path("/home/lixinyu/snap/new/solvers"),
):
    if helper_dir.exists():
        sys.path.insert(0, str(helper_dir))

from arc_utils import get_bg


ALLOWED_RUN_LENGTHS = (2, 3, 4, 5)
COUNT_COLOR_SET = set(ALLOWED_RUN_LENGTHS)
INPUT_COLOR_POOL = tuple(c for c in range(10) if c not in COUNT_COLOR_SET)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _trace_path(
    pset: set[tuple[int, int]],
    start: tuple[int, int],
    total: int,
) -> list[tuple[int, int]]:
    """Trace the unique simple chain from one endpoint."""
    order: list[tuple[int, int]] = []
    prev: tuple[int, int] | None = None
    cur = start

    for _ in range(total):
        order.append(cur)
        nbrs = [q for q in _neighbors4(*cur) if q in pset and q != prev]
        if not nbrs:
            break
        assert len(nbrs) == 1, "Path must be a simple chain"
        prev, cur = cur, nbrs[0]

    assert len(order) == total, "All path cells must be visited exactly once"
    return order


def _endpoint_run_lengths(colors: list[int]) -> tuple[int, int]:
    """Return the lengths of the first and last color runs."""
    if not colors:
        return 0, 0

    prefix = 1
    while prefix < len(colors) and colors[prefix] == colors[0]:
        prefix += 1

    suffix = 1
    while suffix < len(colors) and colors[-1 - suffix] == colors[-1]:
        suffix += 1

    return prefix, suffix


def _ordered_colors_for_combo(g: np.ndarray, bg: int) -> list[int]:
    """
    Return path colors in the fused traversal order:
    shorter endpoint color-segment first, top-left as tie-break.
    """
    pts = [(int(r), int(c)) for r, c in np.argwhere(g != bg)]
    if not pts:
        return []
    if len(pts) == 1:
        r, c = pts[0]
        return [int(g[r, c])]

    pset = set(pts)
    deg = {p: sum(q in pset for q in _neighbors4(*p)) for p in pts}
    ends = [p for p in pts if deg[p] == 1]
    assert len(ends) == 2, "Expected exactly two endpoints"

    order = _trace_path(pset, ends[0], len(pts))
    colors = [int(g[r, c]) for r, c in order]
    prefix_len, suffix_len = _endpoint_run_lengths(colors)

    if suffix_len < prefix_len or (suffix_len == prefix_len and order[0] != min(ends)):
        colors.reverse()
    return colors


def _encode_runs(colors: list[int]) -> np.ndarray:
    """Encode a color sequence into rows [color, count]."""
    if not colors:
        return np.zeros((0, 2), dtype=int)

    rows: list[list[int]] = []
    cur_color = int(colors[0])
    cur_len = 1
    for color in colors[1:]:
        color = int(color)
        if color == cur_color:
            cur_len += 1
        else:
            rows.append([cur_color, cur_len])
            cur_color = color
            cur_len = 1
    rows.append([cur_color, cur_len])
    return np.array(rows, dtype=int)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fused rule:
    1. Find the unique snake path.
    2. Compare the color-run lengths at the two endpoints.
    3. Start from the endpoint whose contiguous endpoint segment is shorter.
    4. Output the traversed color sequence as run-length rows [color, count].

    If the two endpoint segments tie in length, use the original top-left endpoint tie-break.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    colors = _ordered_colors_for_combo(g, bg)
    return _encode_runs(colors)


def _try_extend(
    path_set: set[tuple[int, int]],
    last: tuple[int, int],
    dr: int,
    dc: int,
    count: int,
    h: int,
    w: int,
) -> list[tuple[int, int]]:
    """Try to extend the path by `count` steps; stop before any illegal placement."""
    segment: list[tuple[int, int]] = []
    r, c = last
    for i in range(1, count + 1):
        nr, nc = r + dr * i, c + dc * i
        if nr < 1 or nr >= h - 1 or nc < 1 or nc >= w - 1:
            break
        if (nr, nc) in path_set:
            break

        pred = (r + dr * (i - 1), c + dc * (i - 1))
        bad = False
        for nb in _neighbors4(nr, nc):
            if nb != pred and nb in path_set:
                bad = True
                break
        if bad:
            break

        segment.append((nr, nc))
        path_set.add((nr, nc))
    return segment


def _sample_choice_partition(total: int, num_parts: int, choices: tuple[int, ...]) -> list[int] | None:
    """Sample an exact partition of `total` into `num_parts` values chosen from `choices`."""
    min_len = min(choices)
    max_len = max(choices)

    def dfs(remaining: int, parts_left: int) -> list[int] | None:
        if parts_left == 0:
            return [] if remaining == 0 else None

        candidates = [
            value
            for value in choices
            if (parts_left - 1) * min_len <= remaining - value <= (parts_left - 1) * max_len
        ]
        random.shuffle(candidates)
        for value in candidates:
            tail = dfs(remaining - value, parts_left - 1)
            if tail is not None:
                return [value, *tail]
        return None

    if num_parts == 0:
        return [] if total == 0 else None
    if not (num_parts * min_len <= total <= num_parts * max_len):
        return None
    return dfs(total, num_parts)


def _build_segment_colors(palette: list[int], num_segments: int) -> list[int]:
    """Adjacent-distinct segment colors with at least one non-adjacent repeat."""
    assert num_segments > len(palette)

    seg_colors = list(palette)
    random.shuffle(seg_colors)
    while len(seg_colors) < num_segments:
        choices = [c for c in palette if c != seg_colors[-1]]
        seg_colors.append(random.choice(choices))

    for i in range(1, num_segments):
        if seg_colors[i] == seg_colors[i - 1]:
            choices = [c for c in palette if c != seg_colors[i - 1]]
            seg_colors[i] = random.choice(choices)

    assert all(seg_colors[i] != seg_colors[i - 1] for i in range(1, num_segments))
    assert any(seg_colors.count(color) >= 2 for color in palette)
    return seg_colors


def _sample_segment_lengths(total: int, num_segments: int, top_left_is_first: bool) -> list[int] | None:
    """
    Build segment lengths with three guarantees:
    - endpoint run lengths are unequal and differ clearly;
    - the shorter endpoint is opposite the old top-left start;
    - the sample exhibits several repeated length values for visible RLE structure.
    """
    if num_segments < 5:
        return None

    for _ in range(300):
        short_len = random.choice((2, 3))
        long_choices = tuple(v for v in ALLOWED_RUN_LENGTHS if v >= short_len + 2)
        if not long_choices:
            continue
        long_len = random.choice(long_choices)

        interior_total = total - short_len - long_len
        interior_parts = num_segments - 2
        interior = _sample_choice_partition(interior_total, interior_parts, ALLOWED_RUN_LENGTHS)
        if interior is None:
            continue

        seg_lengths = [long_len, *interior, short_len] if top_left_is_first else [short_len, *interior, long_len]
        counts = {value: seg_lengths.count(value) for value in set(seg_lengths)}
        if len(counts) < 3:
            continue
        if max(counts.values()) < 2:
            continue
        return seg_lengths
    return None


def _expected_combo_output(path_colors: list[int], path: list[tuple[int, int]]) -> np.ndarray:
    """Expected fused output from the construction order of `path`."""
    prefix_len, suffix_len = _endpoint_run_lengths(path_colors)
    top_left_is_first = path[0] == min(path[0], path[-1])

    ordered = list(path_colors)
    if suffix_len < prefix_len or (suffix_len == prefix_len and not top_left_is_first):
        ordered.reverse()
    return _encode_runs(ordered)


def _old_top_left_rle_output(path_colors: list[int], path: list[tuple[int, int]]) -> np.ndarray:
    """Reference output if we kept the old top-left start but still applied RLE."""
    ordered = list(path_colors)
    if path[0] != min(path[0], path[-1]):
        ordered.reverse()
    return _encode_runs(ordered)


def _gen_attempt() -> np.ndarray | None:
    h, w = 16, 16
    bg = random.choice(INPUT_COLOR_POOL)

    non_bg_count = random.randint(3, 4)
    palette = random.sample([c for c in INPUT_COLOR_POOL if c != bg], non_bg_count)
    target_len = random.randint(22, 28)

    start_r = random.randint(1, 4)
    start_c = random.randint(1, w - 2)
    path = [(start_r, start_c)]
    path_set = {(start_r, start_c)}
    turns = 0
    is_vertical = True
    h_dir = random.choice((-1, 1))

    while len(path) < target_len:
        run_len = min(random.randint(1, 5), target_len - len(path))
        if run_len <= 0:
            break

        last = path[-1]
        if is_vertical:
            v_dir = 1 if random.random() < 0.85 else -1
            segment = _try_extend(path_set, last, v_dir, 0, run_len, h, w)
            if not segment:
                segment = _try_extend(path_set, last, -v_dir, 0, run_len, h, w)
        else:
            segment = _try_extend(path_set, last, 0, h_dir, run_len, h, w)
            if not segment:
                segment = _try_extend(path_set, last, 0, -h_dir, run_len, h, w)
                if segment:
                    h_dir = -h_dir
            h_dir = -h_dir

        if not segment:
            break

        path.extend(segment)
        turns += 1
        is_vertical = not is_vertical

    total = len(path)
    if not (22 <= total <= 28):
        return None
    if not (7 <= turns <= 11):
        return None

    max_segments = min(7, total // 2)
    min_segments = max(5, non_bg_count + 2)
    if min_segments > max_segments:
        return None
    num_segments = random.randint(min_segments, max_segments)

    top_left_is_first = path[0] == min(path[0], path[-1])
    seg_lengths = _sample_segment_lengths(total, num_segments, top_left_is_first)
    if seg_lengths is None:
        return None

    # Generator-side fusion guarantee:
    # top-left endpoint must be the longer endpoint, so the new direction differs from the old one.
    if top_left_is_first:
        if not (seg_lengths[0] > seg_lengths[-1]):
            return None
    else:
        if not (seg_lengths[0] < seg_lengths[-1]):
            return None

    seg_colors = _build_segment_colors(palette, num_segments)
    colors: list[int] = []
    for seg_len, color in zip(seg_lengths, seg_colors):
        colors.extend([color] * seg_len)

    grid = np.full((h, w), bg, dtype=int)
    for (r, c), color in zip(path, colors):
        grid[r, c] = color

    expected_out = _expected_combo_output(colors, path)
    actual_out = main(grid)
    if not np.array_equal(actual_out, expected_out):
        return None

    old_out = _old_top_left_rle_output(colors, path)
    if np.array_equal(actual_out, old_out):
        return None

    if actual_out.shape[0] >= total:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate input after 500 attempts")
