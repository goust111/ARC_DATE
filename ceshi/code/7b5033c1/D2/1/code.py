# Variant of 7b5033c1
# Dimension: D2 — 遍历起点选择规则 (transform)
# Change: 从两个端点中选择左上角优先的端点（min(row, col)）开始遍历... → 从颜色较少的端点开始

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path("/home/lixinyu/snap/new/arc2_solver_select")))
from arc_utils import get_bg


# Design analysis for the D2 variant:
# a) 具体化与行为追踪:
#    输入仍是一条 4 连通、无分叉的蛇形路径，颜色沿路径按连续色段分布。输出仍是把整条蛇拉直成单列。
#    原题小例子: 端点 A 在左上、其端点色段长度为 7；端点 B 在右下、其端点色段长度为 3。
#    原逻辑会因为 A 更靠左上而从 A 开始，输出首色来自长度 7 的长段。
#    本变体要求从 B 开始，因为 B 所在色段更短，输出首色变成长度 3 的短段颜色。
#    因此真正变化只发生在“确定遍历方向”这一环节，路径识别和逐格跟踪保持原样。
# b) 复杂度校准:
#    若两端色段常常等长，或较短端点恰好也是左上端点，则样本会退化回原题、视觉差异不明显。
#    生成器因此强制两端色段长度至少相差 3，并让“较短端点”与“左上端点”相反。
# c) 隐式假设审查:
#    - 端点检测依赖路径是简单链且恰有两个度为 1 的点；该前提保持不变。
#    - 原代码的 start = min(ends) 直接绑定旧属性，在变体中失效，必须替换。
#    - 跟踪路径时依赖“非背景像素只有一条链”，因此从任一端点出发都能唯一遍历全体像素；该前提保持不变。
#    - 新逻辑额外依赖“端点色段长度可由路径两端的连续同色 run 读出”；生成器通过连续色段着色来保证。
# d) 放置合法性:
#    路径仍在 16x16 网格内部逐段延伸，新增像素前检查越界、重叠和非法 4 邻接，保证整条链无自交、无支路。
#    色段只覆盖已有路径像素，不改变几何结构，因此 solver 的路径提取仍然成立。
# e) 视觉可观察性:
#    若观察几组样例，人类会看到输出首色并不总对应左上端点，而是对应端点中颜色块更短的一边。
#    生成器固定制造“短段端点 != 左上端点”的样本，使这条规则在 2-4 组样例中可见。


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _trace_path(pset: set[tuple[int, int]], start: tuple[int, int], total: int) -> list[tuple[int, int]]:
    """Trace the unique simple path from one endpoint."""
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
    """Lengths of the first and last contiguous color runs along a traced path."""
    prefix = 1
    while prefix < len(colors) and colors[prefix] == colors[0]:
        prefix += 1

    suffix = 1
    while suffix < len(colors) and colors[-1 - suffix] == colors[-1]:
        suffix += 1

    return prefix, suffix


def main(grid: np.ndarray) -> np.ndarray:
    """
    Straighten the snake into a single column.

    Variant rule:
    - Find the two endpoints of the path.
    - Compare the length of each endpoint's contiguous color segment along the path.
    - Start from the endpoint with the shorter segment.
    - If both endpoint segments have the same length, fall back to the old top-left tie-break.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    pts = [(int(r), int(c)) for r, c in np.argwhere(g != bg)]
    if not pts:
        return np.zeros((0, 1), dtype=int)

    pset = set(pts)
    deg: dict[tuple[int, int], int] = {}
    for r, c in pts:
        deg[(r, c)] = sum((nr, nc) in pset for nr, nc in _neighbors4(r, c))

    ends = [p for p in pts if deg[p] == 1]
    assert len(ends) == 2, "Expected exactly two endpoints"

    # Trace from an arbitrary endpoint, then decide whether that direction is correct.
    order = _trace_path(pset, ends[0], len(pts))
    colors = [int(g[r, c]) for r, c in order]
    prefix_len, suffix_len = _endpoint_run_lengths(colors)

    # New assumption audit:
    # - prefix/suffix runs are exactly the endpoint color segments because tracing visits
    #   cells in path order and the generator colors the path in contiguous segments.
    # - Equal lengths are ambiguous under the new rule, so we keep the original min(ends)
    #   tie-break for determinism.
    if suffix_len < prefix_len or (suffix_len == prefix_len and order[0] != min(ends)):
        colors.reverse()

    return np.array(colors, dtype=int).reshape((-1, 1))


def _try_extend(
    path_set: set[tuple[int, int]],
    last: tuple[int, int],
    dr: int,
    dc: int,
    count: int,
    H: int,
    W: int,
) -> list[tuple[int, int]]:
    """Try to extend the path by `count` steps in direction (dr, dc)."""
    segment: list[tuple[int, int]] = []
    r, c = last
    for i in range(1, count + 1):
        nr, nc = r + dr * i, c + dc * i
        if nr < 1 or nr >= H - 1 or nc < 1 or nc >= W - 1:
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


def _random_partition(total: int, num_parts: int, min_len: int, max_len: int) -> list[int] | None:
    """Partition total into num_parts values each in [min_len, max_len]."""
    if num_parts * min_len > total or num_parts * max_len < total:
        return None

    parts = [min_len] * num_parts
    remaining = total - sum(parts)
    for _ in range(remaining):
        eligible = [i for i in range(num_parts) if parts[i] < max_len]
        if not eligible:
            return None
        parts[random.choice(eligible)] += 1
    random.shuffle(parts)
    return parts


def _build_segment_colors(palette: list[int], num_segments: int) -> list[int]:
    """Create adjacent-distinct segment colors while ensuring all palette colors appear."""
    assert num_segments >= len(palette)
    shuffled_palette = list(palette)
    random.shuffle(shuffled_palette)

    seg_colors = list(shuffled_palette)
    while len(seg_colors) < num_segments:
        choices = [c for c in palette if c != seg_colors[-1]]
        seg_colors.append(random.choice(choices))
    return seg_colors


def _expected_colors(path_colors: list[int], path: list[tuple[int, int]]) -> list[int]:
    """Return the correct output colors under the variant rule."""
    prefix_len, suffix_len = _endpoint_run_lengths(path_colors)
    if prefix_len < suffix_len:
        return list(path_colors)
    if suffix_len < prefix_len:
        return list(reversed(path_colors))
    return list(path_colors) if path[0] == min(path[0], path[-1]) else list(reversed(path_colors))


def _gen_attempt() -> np.ndarray | None:
    H, W = 16, 16
    bg = random.randint(0, 9)
    non_bg_count = random.randint(3, 5)
    palette = random.sample([c for c in range(10) if c != bg], non_bg_count)

    target_len = random.randint(20, 30)
    start_r = random.randint(1, 4)
    start_c = random.randint(1, W - 2)

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
            segment = _try_extend(path_set, last, v_dir, 0, run_len, H, W)
            if not segment:
                segment = _try_extend(path_set, last, -v_dir, 0, run_len, H, W)
        else:
            segment = _try_extend(path_set, last, 0, h_dir, run_len, H, W)
            if not segment:
                segment = _try_extend(path_set, last, 0, -h_dir, run_len, H, W)
                if segment:
                    h_dir = -h_dir
            h_dir = -h_dir

        if not segment:
            break

        path.extend(segment)
        turns += 1
        is_vertical = not is_vertical

    total = len(path)
    if not (20 <= total <= 30):
        return None
    if not (7 <= turns <= 11):
        return None

    # Generator-side legality + visibility guarantees:
    # - Endpoint color runs differ clearly (at least 3 cells).
    # - The shorter-run endpoint is the opposite of the old top-left endpoint,
    #   so generated samples express the new rule instead of silently matching the old one.
    short_len = random.randint(2, 4)
    long_len = random.randint(short_len + 3, 9)
    interior_total = total - short_len - long_len
    min_interior_segments = max(2, non_bg_count - 2)
    max_interior_segments = min(4, interior_total // 2)
    if min_interior_segments > max_interior_segments:
        return None

    interior_segments = random.randint(min_interior_segments, max_interior_segments)
    middle_lengths = _random_partition(interior_total, interior_segments, min_len=2, max_len=7)
    if middle_lengths is None:
        return None

    top_left_end = min(path[0], path[-1])
    if path[0] == top_left_end:
        seg_lengths = [long_len, *middle_lengths, short_len]
    else:
        seg_lengths = [short_len, *middle_lengths, long_len]

    num_segments = len(seg_lengths)
    seg_colors = _build_segment_colors(palette, num_segments)
    if any(seg_colors[i] == seg_colors[i - 1] for i in range(1, num_segments)):
        return None

    colors: list[int] = []
    for seg_len, color in zip(seg_lengths, seg_colors):
        colors.extend([color] * seg_len)

    expected_out = _expected_colors(colors, path)
    if path[0] == top_left_end:
        if not (_endpoint_run_lengths(colors)[0] > _endpoint_run_lengths(colors)[1]):
            return None
    else:
        if not (_endpoint_run_lengths(colors)[0] < _endpoint_run_lengths(colors)[1]):
            return None

    grid = np.full((H, W), bg, dtype=int)
    for (r, c), color in zip(path, colors):
        grid[r, c] = color

    actual_out = main(grid).reshape(-1).tolist()
    if actual_out != expected_out:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate input after 300 attempts")
