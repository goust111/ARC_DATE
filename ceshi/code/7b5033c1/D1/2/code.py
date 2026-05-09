# Variant of 7b5033c1
# Dimension: D1 — 变换方式：线性化拉直 (transform)
# Change: 将蛇形4-连通路径拉直为 Nx1 单列，按遍历顺序输出每个像素的颜色
#   → 按颜色分段统计替代逐像素输出，输出每个颜色段的 [颜色, 长度编码色]。
#
# Design analysis
# a) 具体化与行为追踪:
#    输入仍然是一条 4-连通、无分叉的多色蛇形路径；颜色沿路径以连续段出现。
#    原任务把路径按端点到端点展开成单列，例如路径颜色
#    [1, 1, 1, 3, 3, 2, 2, 2] -> 输出 8x1 列 [1, 1, 1, 3, 3, 2, 2, 2]^T。
#    本变体改为先沿同一路径遍历，再做 run-length encoding，
#    但不直接把长度数字写进网格，而是用固定颜色编码少数几档长度:
#    长度 2 -> 4, 长度 3 -> 6, 长度 4 -> 8。
#    [1, 1, 1, 3, 3, 2, 2, 2] -> 输出 3x2:
#    [[1, 6],
#     [3, 4],
#     [2, 6]]
#    因此变化只发生在“把 traversal order 转成输出网格”的环节；
#    路径检测、端点选择、遍历方向都保持不变。
# b) 复杂度校准:
#    如果长度档位过多，人类从 2-3 个样例里很难稳定归纳“右列在表示长度”。
#    生成器因此强制:
#    - 每段长度只允许取 2/3/4 三档；
#    - 每个训练样例都出现这三档，保证右列的三种编码色反复出现；
#    - 段数严格多于调色板颜色数，保证至少有一种颜色非相邻重复出现，
#      这样解题者能看出这是“按遍历顺序分段”，不是简单做颜色直方图。
# c) 隐式假设审查:
#    - 原 `mask = g != bg` 依赖“所有非背景像素都属于同一条蛇”；本变体仍成立。
#    - 原 `ends = [p for p in pts if deg[p] == 1]` 依赖“对象是简单链而非环/分叉”；本变体仍成立，
#      生成器继续保证只有两个端点。
#    - 原 `cols = [g[r, c] for ...]` 默认逐像素直接输出；这里不能再逐像素写出，
#      必须改成顺序遍历后累计同色 run，否则会错误地回到原任务。
#    - 新增假设: 输出第二列不是原始长度数字，而是固定的长度编码色；
#      solver 和 generator 必须共享同一张映射表。
# d) 放置合法性:
#    生成器仍使用逐段延伸路径的方式构造输入，并在放置时检查:
#    - 不越界；
#    - 不与已有路径重叠；
#    - 新单元除前驱外不能与路径产生额外 4 邻接。
#    这些条件保证输入始终是一条 solver 可以稳定追踪的简单路径。
# e) 视觉可观察性:
#    人类看到样例时，最明显的差异会是输出从“高单列”变成“更短的双列”，
#    且右列只出现 4/6/8 三种稳定颜色，并分别对应短/中/长三档。
#    这比直接出现大量不同长度数字更容易从少量样例里归纳。
#
# Reference-task note:
# - 题面要求先阅读 source_task `8f215267` 的 solver。
# - 在当前工作区搜索后，仅找到 `8f215267.json`，没有可访问的 solver 代码。
# - 因此这里只能参考其“counting”思路，把计数显式编码进输出第二列。

from __future__ import annotations

from pathlib import Path
import sys
import random

import numpy as np

for helper_dir in (
    Path("/home/lixinyu/snap/new/arc2_solver_select"),
    Path("/home/lixinyu/snap/new/solvers"),
):
    if helper_dir.exists():
        sys.path.insert(0, str(helper_dir))

from arc_utils import get_bg


def _trace_path(g: np.ndarray, bg: int) -> list[tuple[int, int]]:
    """Return snake cells in endpoint-to-endpoint order, starting at the top-left endpoint."""
    pts = [(int(r), int(c)) for r, c in np.argwhere(g != bg)]
    if not pts:
        return []
    if len(pts) == 1:
        return pts

    pset = set(pts)
    deg: dict[tuple[int, int], int] = {}
    for r, c in pts:
        d = 0
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            if (r + dr, c + dc) in pset:
                d += 1
        deg[(r, c)] = d

    ends = [p for p in pts if deg[p] == 1]
    assert len(ends) == 2
    start = min(ends)

    order: list[tuple[int, int]] = []
    prev: tuple[int, int] | None = None
    cur = start
    for _ in range(len(pts)):
        order.append(cur)

        r, c = cur
        nbrs = []
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nxt = (r + dr, c + dc)
            if nxt in pset and nxt != prev:
                nbrs.append(nxt)
        if not nbrs:
            break

        assert len(nbrs) == 1
        prev, cur = cur, nbrs[0]

    assert len(order) == len(pts)
    return order


LENGTH_TO_CODE = {2: 4, 3: 6, 4: 8}
ALLOWED_LENGTHS = tuple(LENGTH_TO_CODE)
LENGTH_CODE_SET = set(LENGTH_TO_CODE.values())


def _encode_runs(colors: list[int]) -> np.ndarray:
    """Encode a color sequence as rows of [color, length_code_color]."""
    if not colors:
        return np.zeros((0, 2), dtype=int)

    runs: list[list[int]] = []
    cur_color = int(colors[0])
    cur_len = 1
    for color in colors[1:]:
        color = int(color)
        if color == cur_color:
            cur_len += 1
        else:
            runs.append([cur_color, cur_len])
            cur_color = color
            cur_len = 1
    runs.append([cur_color, cur_len])

    encoded = []
    for color, run_len in runs:
        assert run_len in LENGTH_TO_CODE
        encoded.append([color, LENGTH_TO_CODE[run_len]])
    return np.array(encoded, dtype=int)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Straighten a colored snake path into run-length rows [color, length_code_color].

    Input structure:
    - bg = most common color.
    - Non-bg pixels form a single 4-connected path with exactly two endpoints.
    - Colors appear in contiguous segments along that path.

    Rule:
    - Find the top-left-most endpoint and trace the path in order.
    - Compress consecutive equal colors into runs.
    - Output a Kx2 grid whose rows are [segment_color, segment_length_code].
    - The length code is fixed: 2 -> 4, 3 -> 6, 4 -> 8.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    order = _trace_path(g, bg)
    colors = [int(g[r, c]) for r, c in order]
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
    """Try to extend the path by `count` steps; stop before illegal placements."""
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
        for ddr, ddc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nb = (nr + ddr, nc + ddc)
            if nb != pred and nb in path_set:
                bad = True
                break
        if bad:
            break

        segment.append((nr, nc))
        path_set.add((nr, nc))
    return segment


def _random_partition(total: int, num_parts: int, min_len: int, max_len: int) -> list[int] | None:
    """Partition total into num_parts values each in [min_len, max_len]."""
    if num_parts * min_len > total or num_parts * max_len < total:
        return None

    parts = [min_len] * num_parts
    remaining = total - sum(parts)
    for _ in range(remaining):
        eligible = [i for i, part in enumerate(parts) if part < max_len]
        if not eligible:
            return None
        parts[random.choice(eligible)] += 1
    random.shuffle(parts)
    return parts


def _build_segment_colors(palette: list[int], num_segments: int) -> list[int]:
    """Create adjacent-distinct segment colors and force at least one non-adjacent repeat."""
    assert num_segments > len(palette)

    colors = list(palette)
    random.shuffle(colors)
    while len(colors) < num_segments:
        choices = [c for c in palette if c != colors[-1]]
        colors.append(random.choice(choices))

    for i in range(1, len(colors)):
        if colors[i] == colors[i - 1]:
            choices = [c for c in palette if c != colors[i - 1]]
            colors[i] = random.choice(choices)

    assert any(colors.count(c) >= 2 for c in palette)
    assert all(colors[i] != colors[i - 1] for i in range(1, len(colors)))
    return colors


def _sample_segment_lengths(total: int, num_segments: int) -> list[int] | None:
    """Sample segment lengths from the small visible set {2,3,4}."""
    if num_segments < len(ALLOWED_LENGTHS):
        return None
    if not (2 * num_segments <= total <= 4 * num_segments):
        return None

    for _ in range(200):
        lengths = [2, 3, 4]
        while len(lengths) < num_segments:
            lengths.append(random.choice(ALLOWED_LENGTHS))
        random.shuffle(lengths)
        if sum(lengths) == total:
            return lengths
    return None


def _gen_attempt() -> np.ndarray | None:
    h, w = 16, 16
    available_for_input = [c for c in range(10) if c not in LENGTH_CODE_SET]
    bg = random.choice(available_for_input)

    non_bg_count = random.randint(3, 4)
    palette = random.sample([c for c in available_for_input if c != bg], non_bg_count)
    target_len = random.randint(20, 24)

    start_r = random.randint(1, 4)
    start_c = random.randint(1, w - 2)
    path = [(start_r, start_c)]
    path_set = {(start_r, start_c)}
    turns = 0
    is_vertical = True
    h_dir = random.choice([-1, 1])

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
    if total < 20 or total > 30:
        return None
    if turns < 7 or turns > 11:
        return None

    max_segments = min(6, total // 2)
    min_segments = max(4, non_bg_count + 1)
    if min_segments > max_segments:
        return None
    num_segments = random.randint(min_segments, max_segments)

    seg_lengths = _sample_segment_lengths(total, num_segments)
    if seg_lengths is None:
        return None
    if set(seg_lengths) != set(ALLOWED_LENGTHS):
        return None

    seg_colors = _build_segment_colors(palette, num_segments)
    colors: list[int] = []
    for seg_len, color in zip(seg_lengths, seg_colors):
        colors.extend([color] * seg_len)

    grid = np.full((h, w), bg, dtype=int)
    for (r, c), color in zip(path, colors):
        grid[r, c] = color

    expected_out = np.array([[color, LENGTH_TO_CODE[seg_len]] for color, seg_len in zip(seg_colors, seg_lengths)], dtype=int)
    actual_out = main(grid)
    if not np.array_equal(actual_out, expected_out):
        return None
    if actual_out.shape[0] >= total:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate input after 400 attempts")
