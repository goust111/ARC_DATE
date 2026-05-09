# Variant of 7b5033c1
# Dimension: D9 — 源-输出基数关系 (meta)
# Change: 1:1 — 一条路径产生一个色列输出... → 1:1 但含条件过滤

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path("/home/lixinyu/snap/new/arc2_solver_select")))
from arc_utils import get_bg


# Design analysis for the D9 variant:
# - Reference-task note:
#   The prompt requires reading the solver of source task 8b9c3697 first, but this workspace
#   does not contain an 8b9c3697 solver file after direct code search. I therefore used the
#   official JSON examples at /ARC-data/ARC-AGI-2/data/evaluation/8b9c3697.json as the only
#   available baseline for the "conditional_deletion" idea: some source content survives,
#   some disappears according to an inferred condition.
# - a) 具体化与行为追踪:
#   输入仍然是一条无分叉的 4 连通蛇形路径，路径像素带有多段连续颜色。原题会先按左上端点
#   开始追踪整条路径，再把所有颜色依次写成单列。
#   本变体只改输出阶段: 先得到线性颜色序列，再按“相邻同色的连续段”分段，只保留长度 >= 3
#   的色段，长度 1 或 2 的短段直接删除，输出变成更短的单列。
#   具体小例子:
#   路径线性化后若是 [2,2,2, 5, 7,7, 3,3,3,3, 6]，
#   原题输出同一列；本变体输出 [2,2,2, 3,3,3,3]。
#   差异只发生在“线性序列转输出”这一步，路径识别、端点选择、遍历顺序全部保持原样。
# - b) 复杂度校准:
#   若所有色段都 >= 3，本变体会退化成原题；若所有色段都 < 3，输出几乎为空，也不利于观察。
#   生成器因此强制同时出现至少两个短段和两个长段，并保证删除像素数显著大于 0。
# - c) 隐式假设审查:
#   1. bg = get_bg(g): 依赖背景仍是全局最多颜色。变体没有改变这一点。
#   2. mask / pts: 依赖非背景像素仍只构成一条链。变体没有改变几何结构，这点仍成立。
#   3. ends = [deg==1]: 依赖链恰有两个端点。生成器继续显式保证简单路径，因此仍成立。
#   4. start = min(ends): 原题以左上端点固定遍历方向；变体不改变这个方向规则，因此保留。
#   5. order tracing: 依赖每个非端点只有一个“除 prev 外”的后继。对简单链仍成立。
#   6. 原题最后直接 return 全部 cols。这里是旧属性绑定点，在变体中必须替换成“按连续段过滤”。
#   7. 新逻辑额外假设“连续段”是沿路径遍历顺序定义，而不是空间上的几何连块；这正是本题的
#      线性化后过滤语义。生成器通过按路径顺序着色来保证样例可读。
# - d) 放置合法性:
#   生成器沿原题方式在 16x16 网格内逐步延展路径。每次加入新格前检查:
#   不越界、不重叠、且不会与前驱之外的已有路径 4 邻接，从而避免自交和分叉。
#   着色只发生在已合法生成的路径像素上，不会破坏 solver 所需的链结构。
# - e) 视觉可观察性:
#   人类看 2-4 组样例时，会发现输出列的颜色顺序仍遵循输入蛇形路径，但所有很短的插入色段都
#   系统性消失，只剩下长段。因为输出仍是多色、而不是只剩一种颜色，这条规则比“多数颜色保留”
#   更容易从少量样例中观察出来。
# - main() 的新增假设审查:
#   新增的每个循环只假设路径线性化序列可被按相邻同色分段；generate_input() 通过
#   “相邻段颜色不同、长短段混合”来保证这一结构稳定且可见。


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _trace_path(pset: set[tuple[int, int]], start: tuple[int, int], total: int) -> list[tuple[int, int]]:
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

    assert len(order) == total, "All path cells must be visited"
    return order


def _filter_runs(colors: list[int], min_len: int = 3) -> list[int]:
    kept: list[int] = []
    i = 0
    while i < len(colors):
        j = i + 1
        while j < len(colors) and colors[j] == colors[i]:
            j += 1
        if j - i >= min_len:
            kept.extend(colors[i:j])
        i = j
    return kept


def main(grid: np.ndarray) -> np.ndarray:
    """
    Straighten the snake path into a single-column strip, but delete short color runs.

    Variant rule:
    - Trace the unique path from the top-left-most endpoint exactly as in the original task.
    - Read the colors in traversal order.
    - Split that 1D color sequence into maximal adjacent-equal runs.
    - Keep only runs of length >= 3; runs of length 1 or 2 are removed from the output.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    mask = g != bg
    pts = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    if not pts:
        return np.zeros((0, 1), dtype=int)

    pset = set(pts)
    deg: dict[tuple[int, int], int] = {}
    for r, c in pts:
        deg[(r, c)] = sum((nr, nc) in pset for nr, nc in _neighbors4(r, c))

    ends = [p for p in pts if deg[p] == 1]
    assert len(ends) == 2, "Expected exactly two endpoints"
    start = min(ends)

    order = _trace_path(pset, start, len(pts))
    cols = [int(g[r, c]) for r, c in order]
    filtered = _filter_runs(cols, min_len=3)
    return np.array(filtered, dtype=int).reshape((-1, 1))


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


def _random_typed_partition(total: int) -> list[int] | None:
    """
    Build a mixed segment-length list summing to total.

    Guarantees:
    - at least 2 short runs (length 1 or 2)
    - at least 2 long runs (length >= 3)
    - total deleted pixels >= 4
    """
    for _ in range(300):
        num_segments = random.randint(6, 8)
        short_count = random.randint(2, min(4, num_segments - 2))
        short_idx = set(random.sample(range(num_segments), short_count))

        parts = [1 if i in short_idx else 3 for i in range(num_segments)]
        caps = [2 if i in short_idx else 8 for i in range(num_segments)]
        remaining = total - sum(parts)
        if remaining < 0:
            continue

        while remaining > 0:
            eligible = [i for i in range(num_segments) if parts[i] < caps[i]]
            if not eligible:
                break
            pick = random.choice(eligible)
            parts[pick] += 1
            remaining -= 1
        if remaining != 0:
            continue

        short_pixels = sum(parts[i] for i in short_idx)
        long_runs = [parts[i] for i in range(num_segments) if i not in short_idx]
        if short_pixels < 4:
            continue
        if len(long_runs) < 2 or sum(long_runs) < 8:
            continue
        return parts

    return None


def _build_segment_colors(palette: list[int], num_segments: int) -> list[int]:
    shuffled = list(palette)
    random.shuffle(shuffled)

    seg_colors: list[int] = []
    for i in range(num_segments):
        if i < len(shuffled):
            candidate = shuffled[i]
        else:
            choices = [c for c in palette if c != seg_colors[-1]]
            candidate = random.choice(choices)
        if seg_colors and candidate == seg_colors[-1]:
            choices = [c for c in palette if c != seg_colors[-1]]
            candidate = random.choice(choices)
        seg_colors.append(candidate)
    return seg_colors


def _expected_output(path: list[tuple[int, int]], colors: list[int]) -> np.ndarray:
    oriented = list(colors) if path[0] == min(path[0], path[-1]) else list(reversed(colors))
    filtered = _filter_runs(oriented, min_len=3)
    return np.array(filtered, dtype=int).reshape((-1, 1))


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
        segment: list[tuple[int, int]] = []

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

    seg_lengths = _random_typed_partition(total)
    if seg_lengths is None:
        return None
    if sum(1 for x in seg_lengths if x < 3) < 2:
        return None
    if sum(1 for x in seg_lengths if x >= 3) < 2:
        return None

    seg_colors = _build_segment_colors(palette, len(seg_lengths))

    colors: list[int] = []
    for seg_len, col in zip(seg_lengths, seg_colors):
        colors.extend([col] * seg_len)

    grid = np.full((H, W), bg, dtype=int)
    for (r, c), col in zip(path, colors):
        grid[r, c] = col

    expected_out = _expected_output(path, colors)
    actual_out = main(grid)
    if not np.array_equal(actual_out, expected_out):
        return None
    if actual_out.shape[0] >= total:
        return None
    if actual_out.shape[0] < 8:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate input after 400 attempts")
