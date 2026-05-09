# Combo variant of 7b5033c1 (index 31)
# combo_spec: D1/2+D8/1
# Name: 游程编码+段重排
#
# Design analysis
# a) 具体化与行为追踪:
#    输入仍是一条 4-连通、无分叉的多色蛇形路径，端点读取规则不变。
#    原任务示例:
#    traversal colors = [7,7, 1,1,1,1,1,1, 9,9,9,9,9, 8,8,8,8, 7,7,7]
#    原输出会直接是 20x1 单列。
#    本融合先做 RLE:
#    runs = [(7,2), (1,6), (9,5), (8,4), (7,3)]
#    再按段长降序重排:
#    sorted runs = [(1,6), (9,5), (8,4), (7,3), (7,2)]
#    最终输出是 5x2:
#    [[1,6],
#     [9,5],
#     [8,4],
#     [7,3],
#     [7,2]]
#    因此改动不是“把路径换个方向画出来”，而是把路径线性化后的中间序列先压缩成段，
#    再把这些段作为整体按长度重新排序。
# b) 复杂度校准:
#    若段长太多或排序存在大量并列，人类很难在 2-4 个样例里稳定归纳规则。
#    这里把生成器约束为每个样例恰好 5 段，且段长固定为 {2,3,4,5,6} 各一次。
#    这样第二列是少量可见数字，排序目标明确，且没有“长度并列时怎么打破平局”的歧义。
#    同时强制至少一个颜色非相邻重复，避免任务退化成简单颜色集合排序。
# c) 隐式假设审查:
#    - 原 `g != bg` 假设所有非背景像素都属于同一对象；融合后仍需成立。
#    - 原 `deg == 1` 假设对象是简单链而非环/分叉；生成器继续保证仅两个端点。
#    - 原 `cols = [g[r, c] ...]` 默认最终答案就是逐像素序列；这在融合后失效，
#      该序列现在只是中间产物，必须继续做 run-length segmentation。
#    - 新 `sorted(runs, key=...)` 隐含“排序单元是颜色段而不是颜色种类”；生成器通过
#      邻接不同色的分段着色来保证这一点。
#    - 新输出把第二列当作真实段长数字使用，因此生成器必须把段长限制在 ARC 颜色值 0-9 内。
# d) 放置合法性:
#    生成器沿用逐段延伸的蛇构造法，并检查:
#    - 新单元不越界；
#    - 不与已有路径重叠；
#    - 除前驱外不与已有路径产生额外 4 邻接。
#    颜色是在 canonical path 求出后再赋给路径，所以不会因为起始生成方向不同而错位。
# e) 视觉可观察性:
#    人类会同时看到两个明显变化:
#    - 输出从高单列变成短得多的双列表格，提示“做了分段统计”；
#    - 双列表格按第二列从大到小排列，且不是输入的自然遍历顺序，提示“段被重排了”。
# f) 可视化整洁度:
#    融合不在输入网格叠加额外标记，仍只保留一条蛇，因此视觉上干净。
#    输出只用两列，左列是颜色、右列是长度，信息分区明确，不会和输入对象混叠。
# g) 颜色变化可推断性:
#    右列不是任意编码色，而是直接使用小整数段长 2..6；这让“计数”语义直接可见。
#    同时生成器禁止输入颜色使用 2..6，使左右两列在视觉上更容易区分:
#    左列来自输入色带，右列来自固定的小长度值集合。
# h) main() 的新隐式假设审查:
#    - `_segmentize` 假设相邻同色像素应合并成单段；生成器保证相邻段颜色不同。
#    - `_reorder_runs` 采用 `(-length, color)` 作为总排序键；但生成器进一步保证长度全异，
#      因而实际样本中不会暴露颜色平局规则，也不会依赖它。
#    - 输出 shape 为 `(段数, 2)`；生成器固定为 5 段，确保双列格式稳定可见。

from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path("/home/lixinyu/snap/new")
for helper_dir in (ROOT / "arc2_solver_select", ROOT / "solvers"):
    if str(helper_dir) not in sys.path and helper_dir.exists():
        sys.path.insert(0, str(helper_dir))

from arc_utils import get_bg


ALLOWED_LENGTHS = (2, 3, 4, 5, 6)
FORBIDDEN_INPUT_COLORS = set(ALLOWED_LENGTHS)
TARGET_SEGMENT_COUNT = len(ALLOWED_LENGTHS)
TARGET_PATH_LEN = sum(ALLOWED_LENGTHS)


def _trace_path_order(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not points:
        return []
    if len(points) == 1:
        r, c = points[0]
        return [(int(r), int(c))]

    pts = [(int(r), int(c)) for r, c in points]
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


def _trace_path_colors(grid: np.ndarray) -> list[int]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    points = [(int(r), int(c)) for r, c in np.argwhere(g != bg)]
    order = _trace_path_order(points)
    return [int(g[r, c]) for r, c in order]


def _segmentize(colors: list[int]) -> list[tuple[int, int]]:
    if not colors:
        return []

    runs: list[tuple[int, int]] = []
    cur_color = int(colors[0])
    cur_len = 1
    for color in colors[1:]:
        color = int(color)
        if color == cur_color:
            cur_len += 1
        else:
            runs.append((cur_color, cur_len))
            cur_color = color
            cur_len = 1
    runs.append((cur_color, cur_len))
    return runs


def _reorder_runs(runs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return sorted(runs, key=lambda run: (-run[1], run[0]))


def _has_nonadjacent_repeat(colors: list[int]) -> bool:
    for i, color in enumerate(colors):
        for j in range(i + 2, len(colors)):
            if colors[j] == color:
                return True
    return False


def _variant_metrics(colors: list[int]) -> dict[str, int]:
    runs = _segmentize(colors)
    seg_lengths = [run_len for _color, run_len in runs]
    seg_colors = [color for color, _run_len in runs]
    reordered = _reorder_runs(runs)
    moved = sum(int(src != dst) for src, dst in zip(runs, reordered))
    return {
        "segment_count": len(runs),
        "distinct_lengths": len(set(seg_lengths)),
        "nonadjacent_repeat": int(_has_nonadjacent_repeat(seg_colors)),
        "changed_after_sort": int(runs != reordered),
        "moved_segments": moved,
    }


def _is_visible_variant(colors: list[int]) -> bool:
    metrics = _variant_metrics(colors)
    runs = _segmentize(colors)
    lengths = {run_len for _color, run_len in runs}
    return (
        metrics["segment_count"] == TARGET_SEGMENT_COUNT
        and lengths == set(ALLOWED_LENGTHS)
        and metrics["nonadjacent_repeat"] == 1
        and metrics["changed_after_sort"] == 1
        and metrics["moved_segments"] >= 2
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace the snake, compress consecutive same-color cells into (color, length)
    rows, then reorder those rows by descending run length.
    """
    colors = _trace_path_colors(np.array(grid, dtype=int))
    runs = _segmentize(colors)
    if not runs:
        return np.zeros((0, 2), dtype=int)
    return np.array(_reorder_runs(runs), dtype=int)


def _try_extend(
    path_set: set[tuple[int, int]],
    last: tuple[int, int],
    dr: int,
    dc: int,
    count: int,
    h: int,
    w: int,
) -> list[tuple[int, int]]:
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


def _build_segment_colors(palette: list[int]) -> list[int] | None:
    templates = [
        (0, 1, 2, 0, 1),
        (0, 1, 0, 2, 1),
        (0, 1, 2, 1, 0),
        (0, 2, 1, 0, 2),
    ]
    for _ in range(100):
        chosen = random.sample(palette, 3)
        template = list(random.choice(templates))
        seg_colors = [chosen[idx] for idx in template]
        if all(seg_colors[i] != seg_colors[i - 1] for i in range(1, len(seg_colors))):
            return seg_colors
    return None


def _gen_path(target_len: int, h: int, w: int) -> tuple[list[tuple[int, int]], int] | tuple[None, None]:
    start_r = random.randint(1, 4)
    start_c = random.randint(1, w - 2)
    path = [(start_r, start_c)]
    path_set = {(start_r, start_c)}
    turns = 0
    is_vertical = True
    h_dir = random.choice([-1, 1])

    while len(path) < target_len:
        step_cap = min(random.randint(1, 4), target_len - len(path))
        if step_cap <= 0:
            break

        last = path[-1]
        if is_vertical:
            v_dir = 1 if random.random() < 0.85 else -1
            segment = _try_extend(path_set, last, v_dir, 0, step_cap, h, w)
            if not segment:
                segment = _try_extend(path_set, last, -v_dir, 0, step_cap, h, w)
        else:
            segment = _try_extend(path_set, last, 0, h_dir, step_cap, h, w)
            if not segment:
                segment = _try_extend(path_set, last, 0, -h_dir, step_cap, h, w)
                if segment:
                    h_dir = -h_dir
            h_dir = -h_dir

        if not segment:
            break

        path.extend(segment)
        turns += 1
        is_vertical = not is_vertical

    if len(path) != target_len:
        return None, None
    return path, turns


def _gen_attempt() -> np.ndarray | None:
    h, w = 16, 16
    available_for_input = [c for c in range(10) if c not in FORBIDDEN_INPUT_COLORS]
    bg = random.choice(available_for_input)
    palette = random.sample([c for c in available_for_input if c != bg], 3)

    path, turns = _gen_path(TARGET_PATH_LEN, h, w)
    if path is None or turns is None:
        return None
    if not (7 <= turns <= 12):
        return None

    canonical_path = _trace_path_order(path)
    seg_lengths = list(ALLOWED_LENGTHS)
    random.shuffle(seg_lengths)
    if seg_lengths == sorted(seg_lengths, reverse=True):
        return None

    seg_colors = _build_segment_colors(palette)
    if seg_colors is None:
        return None

    colors: list[int] = []
    for seg_len, color in zip(seg_lengths, seg_colors):
        colors.extend([color] * seg_len)
    if len(colors) != len(canonical_path):
        return None
    if not _is_visible_variant(colors):
        return None

    grid = np.full((h, w), bg, dtype=int)
    for (r, c), color in zip(canonical_path, colors):
        grid[r, c] = color

    expected_out = np.array(_reorder_runs(list(zip(seg_colors, seg_lengths))), dtype=int)
    actual_out = main(grid)
    if not np.array_equal(actual_out, expected_out):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate input after 500 attempts")
