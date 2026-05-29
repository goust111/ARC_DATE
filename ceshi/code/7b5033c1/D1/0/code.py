# Variant of 7b5033c1
# Dimension: D1 — 变换方式：线性化拉直 (transform)
# Change: 将蛇形4-连通路径拉直为 Nx1 单列，按遍历顺序输出每个像素的颜色... → 折叠为多列矩阵
#
# Design analysis
# a) Concrete behavior:
#    The input is still one 4-connected non-background snake with two endpoints and
#    piecewise-constant colors along the path. The original task traced the snake
#    from the top-left endpoint and emitted the colors as an N x 1 column.
#    In this variant we keep the same trace order, but fold that 1-D color stream
#    into a rectangle whose width K equals the number of distinct non-background
#    colors in the snake. Rows are filled in boustrophedon order:
#      seq = [a,b,c,d,e,f], K = 3
#      original output: [[a],[b],[c],[d],[e],[f]]
#      variant output:  [[a,b,c],
#                        [f,e,d]]
#    The changed step is only the final "linear colors -> output grid" stage.
# b) Complexity calibration:
#    K varies between 3 and 5 across generated tasks, and path length is always a
#    multiple of K with at least 4 rows. This avoids visually weak cases such as a
#    single column, a single row, or an incomplete last row where the folding rule
#    would be harder to infer from few examples.
# c) Implicit-assumption audit of the original solver:
#    - `mask = g != bg`, endpoint detection, and path traversal still rely on the
#      old assumption that all non-bg cells form one simple chain. That remains true.
#    - `start = min(ends)` still works because the reading direction is unchanged.
#    - The old `reshape((-1, 1))` assumed output width was always 1. This is the
#      only assumption intentionally broken by the variant; it is replaced by
#      `width = number of distinct snake colors` plus snake-row folding.
# d) Placement legality:
#    The generator still builds one non-self-touching path inside the border, so
#    cells never overlap and the solver sees exactly one chain. It also enforces
#    that every palette color appears at least once and total_len % K == 0, which
#    is required by the new rectangular fold.
# e) Visual observability:
#    Across examples, the output width changes with the number of colors in the
#    input path, while the path itself stays a single serpentine object. That makes
#    the D1 change visible: the solver must infer both the traversal order and the
#    fold width rule rather than just flatten to one column.
# 9) New solver assumptions and generator guarantees:
#    - Assumption: the snake has exactly two endpoints and no branches. Guaranteed
#      by `_try_extend`, which rejects any new cell that would touch older path
#      cells other than its predecessor.
#    - Assumption: the number of traced cells is divisible by K. Guaranteed by the
#      generator choosing `target_len` from multiples of K only.
#    - Assumption: K >= 2 so the task is not visually identical to the source
#      task. Guaranteed because the generator always uses 3..5 colors.

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/lixinyu/snap/new/arc2_solver_select")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arc_utils import get_bg


def _trace_path(g: np.ndarray, bg: int) -> list[tuple[int, int]]:
    mask = g != bg
    pts = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    if not pts:
        return []

    pset = set(pts)
    deg: dict[tuple[int, int], int] = {}
    for r, c in pts:
        d = 0
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            if (r + dr, c + dc) in pset:
                d += 1
        deg[(r, c)] = d

    assert all(d <= 2 for d in deg.values())
    ends = [p for p in pts if deg[p] == 1]
    assert len(ends) == 2
    start = min(ends)

    order: list[tuple[int, int]] = []
    prev = None
    cur = start
    for _ in range(len(pts)):
        order.append(cur)
        r, c = cur
        nbrs = []
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            q = (r + dr, c + dc)
            if q in pset and q != prev:
                nbrs.append(q)
        if not nbrs:
            break
        assert len(nbrs) == 1
        prev, cur = cur, nbrs[0]

    assert len(order) == len(pts)
    return order


def _fold_colors(colors: list[int], width: int) -> np.ndarray:
    assert width > 0
    assert len(colors) % width == 0
    height = len(colors) // width
    out = np.zeros((height, width), dtype=int)
    idx = 0
    for r in range(height):
        chunk = colors[idx : idx + width]
        idx += width
        out[r] = chunk if r % 2 == 0 else chunk[::-1]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace the snake from the top-left endpoint, then fold its color stream into a
    boustrophedon matrix whose width equals the number of distinct snake colors.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    order = _trace_path(g, bg)
    if not order:
        return np.zeros((0, 0), dtype=int)

    colors = [int(g[r, c]) for r, c in order]
    fold_width = len({c for c in colors})
    return _fold_colors(colors, fold_width)


def _try_extend(
    path_set: set[tuple[int, int]],
    last: tuple[int, int],
    dr: int,
    dc: int,
    count: int,
    H: int,
    W: int,
) -> list[tuple[int, int]]:
    """Try to extend the path by `count` steps in direction (dr,dc)."""
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
        eligible = [i for i in range(num_parts) if parts[i] < max_len]
        if not eligible:
            return None
        parts[random.choice(eligible)] += 1
    random.shuffle(parts)
    return parts


def _build_color_stream(total: int, palette: list[int]) -> list[int] | None:
    color_count = len(palette)
    min_segments = max(color_count + 1, 6)
    max_segments = min(9, total // 2)
    if min_segments > max_segments:
        return None

    num_segments = random.randint(min_segments, max_segments)
    seg_lengths = _random_partition(total, num_segments, min_len=2, max_len=7)
    if seg_lengths is None:
        return None

    seg_colors = palette[:]
    random.shuffle(seg_colors)
    while len(seg_colors) < num_segments:
        prev = seg_colors[-1]
        choices = [c for c in palette if c != prev]
        seg_colors.append(random.choice(choices))
    for i in range(1, len(seg_colors)):
        if seg_colors[i] == seg_colors[i - 1]:
            choices = [c for c in palette if c != seg_colors[i - 1]]
            seg_colors[i] = random.choice(choices)

    colors: list[int] = []
    for seg_len, col in zip(seg_lengths, seg_colors):
        colors.extend([col] * seg_len)
    return colors


def _gen_attempt() -> np.ndarray | None:
    H, W = 16, 16
    bg = random.randint(0, 9)
    color_count = random.randint(3, 5)
    palette = random.sample([c for c in range(10) if c != bg], color_count)

    lengths = [n for n in range(18, 31) if n % color_count == 0]
    target_len = random.choice(lengths)

    start_r = random.randint(1, 4)
    start_c = random.randint(1, W - 2)

    path = [(start_r, start_c)]
    path_set = {(start_r, start_c)}
    turns = 0
    is_vertical = True
    h_dir = random.choice([-1, 1])

    while len(path) < target_len:
        run_len = random.randint(1, 5)
        run_len = min(run_len, target_len - len(path))
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
    if total != target_len:
        return None
    if turns < 6 or turns > 12:
        return None

    colors = _build_color_stream(total, palette)
    if colors is None:
        return None

    grid = np.full((H, W), bg, dtype=int)
    for (r, c), col in zip(path, colors):
        grid[r, c] = col

    expected_out = _fold_colors(colors, color_count)
    actual_out = main(grid)
    if not np.array_equal(actual_out, expected_out):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate input after 400 attempts")
