# Combo variant of 7b5033c1 (index 4)
# combo_spec: D1/2+D4/2
# Name: 回文色段+游程编码
#
# Design analysis
# a) 具体化与行为追踪:
#    - 原任务: 输入是一条 4-连通、无分叉的多色蛇。solver 从左上优先的端点出发，
#      逐像素追踪整条路径，输出 N x 1 单列颜色序列。
#    - 本融合: 路径仍是一条蛇，但颜色不再是任意分段，而是“颜色段回文 + 长度也镜像”。
#      例如沿路径的逐像素颜色流为
#      [7,7, 1,1,1, 8,8,8,8, 1,1,1, 7,7]，
#      它来自 5 个颜色段:
#      颜色段 = [7, 1, 8, 1, 7]
#      段长度 = [2, 3, 4, 3, 2]
#      原任务会输出 14 x 1 单列。
#      本融合先保留同样的路径追踪顺序，再做 run-length encoding，输出
#      [[7, 2],
#       [1, 3],
#       [8, 4],
#       [1, 3],
#       [7, 2]]
#      即 K x 2，两列分别是“颜色”和“该颜色连续出现次数”。
#    - 差异点不在路径检测，而在“路径线性序列 -> 输出网格”的最后一步。
#      同时，因为颜色段和长度都回文，从任一端读出的 RLE 行序都相同，
#      所以解题者既能沿原规则追踪，也能用回文性复核答案。
# b) 复杂度校准:
#    - 若长度取值过多，第二列会像任意数字噪声。这里把段长限制在 2..6，
#      并让每个样例只出现 5 段或 7 段，保持规则紧凑可见。
#    - 输入色板限制为 {1,7,8,9} 中的 3 或 4 种，尽量与第二列计数值 2..6 分离，
#      让人更容易看出“左列是颜色，右列是长度”。
#    - 段数固定为奇数，是为了在“相邻段颜色不同”的同时保持回文。
#      偶数段回文会在中间产生相同颜色相邻，视觉上会退化成更少的段。
# c) 隐式假设审查:
#    - 原 `mask = g != bg` 假设所有非背景像素都属于同一对象。本融合仍成立，
#      因为生成器继续只放一条蛇，没有额外标记物。
#    - 原 `ends = [p for p in pts if deg[p] == 1]` 假设对象是简单链而非环/分叉。
#      本融合不改变这一点，生成器仍显式防止自交和额外 4 邻接。
#    - 原 `start = min(ends)` 用来固定方向；本融合虽然回文使两端 RLE 一致，
#      但 solver 仍可沿用这个起点规则，不会与新性质冲突。
#    - 原 `cols = [g[r,c] ...]` 默认逐像素直接输出。这里必须替换成“按追踪顺序聚合同色 run”，
#      否则会错误回退到原始任务。
#    - 新增前提: 输出第二列要能直接容纳段长度，因此段长必须落在 ARC 颜色编号 0..9 内。
#      生成器把段长限制在 2..6，保证这个前提。
# d) 放置合法性:
#    - 生成器按原任务思路逐段延伸路径，并在每一步检查:
#      1. 不越界；
#      2. 不重叠；
#      3. 新格子除前驱外，不能与已有路径形成额外 4 邻接。
#    - 颜色只是在合法路径上事后赋值，不会改变几何合法性。
#    - 回文约束只作用于沿路径的颜色段和段长，不引入任何额外空间对象，因此不存在多对象碰撞问题。
# e) 视觉可观察性:
#    - 输入侧: 解题者会看到蛇上出现左右镜像的色段模式，如 A-B-C-B-A 或 A-B-C-D-C-B-A。
#    - 输出侧: 不再是高单列，而是短很多的两列矩阵；左列颜色回文，右列计数也回文。
#    - 这种“输入是对称分段，输出是对应的对称 RLE 表”比单纯压缩更容易被肉眼识别。
# f) 可视化整洁度:
#    - 输入只有一个对象，没有在蛇旁叠加符号、框线或额外标记。
#    - 颜色段数控制在 5 或 7，网格仍然干净，不会出现密集碎片。
#    - 输出固定两列，结构极规整，不会像散乱编码那样难读。
# g) 颜色变化可推断性:
#    - 没有引入任意颜色映射。输入中的蛇颜色原样进入输出第一列。
#    - 第二列不是新的神秘编码，而是直接写实际段长；结合回文重复，人类可从少量样例归纳。
# 9) 新 solver 假设与 generator 保证:
#    - 假设: 非背景像素组成一条简单 4-连通链。保证: `_try_extend` 阻止自交与分叉。
#    - 假设: 段长 <= 9，能作为 ARC 颜色值放入输出。保证: 段长只从 2..6 采样。
#    - 假设: 至少存在 3 个颜色段，RLE 才能明显区别于原任务。保证: 只生成 5 段或 7 段回文。
#    - 假设: 回文不是假象而是真正影响了生成分布。保证: 生成器显式镜像颜色段与段长，
#      并用反向追踪的 RLE 等价性做校验。

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


DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
BG_COLOR = 0
INPUT_COLOR_POOL = (1, 7, 8, 9)
COUNT_POOL = (2, 3, 4, 5, 6)
GRID_SHAPE = (16, 16)


def _trace_path(g: np.ndarray, bg: int) -> list[tuple[int, int]]:
    """Return the snake cells from the top-left endpoint to the other endpoint."""
    pts = [(int(r), int(c)) for r, c in np.argwhere(g != bg)]
    if not pts:
        return []
    if len(pts) == 1:
        return pts

    pset = set(pts)
    deg: dict[tuple[int, int], int] = {}
    for r, c in pts:
        d = 0
        for dr, dc in DIR4:
            if (r + dr, c + dc) in pset:
                d += 1
        deg[(r, c)] = d

    ends = [p for p in pts if deg[p] == 1]
    assert len(ends) == 2
    assert all(d <= 2 for d in deg.values())
    start = min(ends)

    order: list[tuple[int, int]] = []
    prev: tuple[int, int] | None = None
    cur = start
    for _ in range(len(pts)):
        order.append(cur)
        r, c = cur
        nbrs = []
        for dr, dc in DIR4:
            nxt = (r + dr, c + dc)
            if nxt in pset and nxt != prev:
                nbrs.append(nxt)
        if not nbrs:
            break
        assert len(nbrs) == 1
        prev, cur = cur, nbrs[0]

    assert len(order) == len(pts)
    return order


def _rle_rows(colors: list[int]) -> np.ndarray:
    """Convert a traced color stream into rows of [color, run_length]."""
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
            assert 0 <= cur_len <= 9
            rows.append([cur_color, cur_len])
            cur_color = color
            cur_len = 1
    assert 0 <= cur_len <= 9
    rows.append([cur_color, cur_len])
    return np.array(rows, dtype=int)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace the snake from the top-left endpoint, then output one RLE row per color run.

    Output format:
    - shape = (num_segments, 2)
    - col 0 = segment color
    - col 1 = literal segment length
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    order = _trace_path(g, bg)
    colors = [int(g[r, c]) for r, c in order]
    return _rle_rows(colors)


def _try_extend(
    path_set: set[tuple[int, int]],
    last: tuple[int, int],
    dr: int,
    dc: int,
    count: int,
    h: int,
    w: int,
) -> list[tuple[int, int]]:
    """Try to extend a simple path; stop before any illegal placement."""
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
        for ddr, ddc in DIR4:
            nb = (nr + ddr, nc + ddc)
            if nb != pred and nb in path_set:
                bad = True
                break
        if bad:
            break

        segment.append((nr, nc))
        path_set.add((nr, nc))
    return segment


def _is_palindrome(values: list[int]) -> bool:
    return list(values) == list(values[::-1])


def _build_palindrome_spec() -> tuple[list[int], list[int]] | None:
    """
    Create mirrored segment colors and mirrored segment lengths.

    The visible patterns are:
    - 5 segments: A B C B A
    - 7 segments: A B C D C B A
    """
    for _ in range(200):
        unique_color_count = random.choice((3, 4))
        half_colors = random.sample(INPUT_COLOR_POOL, unique_color_count)
        seg_colors = half_colors + half_colors[-2::-1]

        left_lengths = [random.choice(COUNT_POOL) for _ in range(unique_color_count - 1)]
        center_length = random.choice(COUNT_POOL)
        seg_lengths = left_lengths + [center_length] + left_lengths[::-1]

        total = sum(seg_lengths)
        if not (18 <= total <= 28):
            continue
        if len(set(seg_lengths)) < 2:
            continue
        assert _is_palindrome(seg_colors)
        assert _is_palindrome(seg_lengths)
        assert all(seg_colors[i] != seg_colors[i - 1] for i in range(1, len(seg_colors)))
        return list(seg_colors), list(seg_lengths)
    return None


def _generate_path(target_len: int) -> list[tuple[int, int]] | None:
    """Generate one simple snake of exactly `target_len` cells."""
    h, w = GRID_SHAPE
    for _ in range(160):
        start_r = random.randint(1, 4)
        start_c = random.randint(1, w - 2)
        path = [(start_r, start_c)]
        path_set = {(start_r, start_c)}
        turns = 0
        is_vertical = True
        h_dir = random.choice((-1, 1))

        while len(path) < target_len:
            step_budget = min(random.randint(1, 5), target_len - len(path))
            if step_budget <= 0:
                break

            last = path[-1]
            if is_vertical:
                v_dir = 1 if random.random() < 0.85 else -1
                segment = _try_extend(path_set, last, v_dir, 0, step_budget, h, w)
                if not segment:
                    segment = _try_extend(path_set, last, -v_dir, 0, step_budget, h, w)
            else:
                segment = _try_extend(path_set, last, 0, h_dir, step_budget, h, w)
                if not segment:
                    segment = _try_extend(path_set, last, 0, -h_dir, step_budget, h, w)
                    if segment:
                        h_dir = -h_dir
                h_dir = -h_dir

            if not segment:
                break

            path.extend(segment)
            turns += 1
            is_vertical = not is_vertical

        if len(path) != target_len:
            continue
        if not (7 <= turns <= 12):
            continue
        return path
    return None


def _expand_segments(seg_colors: list[int], seg_lengths: list[int]) -> list[int]:
    colors: list[int] = []
    for color, seg_len in zip(seg_colors, seg_lengths):
        colors.extend([int(color)] * int(seg_len))
    return colors


def _gen_attempt() -> np.ndarray | None:
    spec = _build_palindrome_spec()
    if spec is None:
        return None
    seg_colors, seg_lengths = spec
    colors = _expand_segments(seg_colors, seg_lengths)
    target_len = len(colors)

    path = _generate_path(target_len)
    if path is None:
        return None

    grid = np.full(GRID_SHAPE, BG_COLOR, dtype=int)
    for (r, c), color in zip(path, colors):
        grid[r, c] = color

    expected = np.array([[color, seg_len] for color, seg_len in zip(seg_colors, seg_lengths)], dtype=int)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None

    if not _is_palindrome(seg_colors):
        return None
    if not _is_palindrome(seg_lengths):
        return None

    reverse_actual = _rle_rows(list(reversed(colors)))
    if not np.array_equal(reverse_actual, expected):
        return None

    if expected.shape[0] >= target_len:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(600):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate input after 600 attempts")
