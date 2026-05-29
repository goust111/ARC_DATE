from __future__ import annotations

# Variant of 7b5033c1
# Dimension: D1 — 变换方式：线性化拉直 (transform)
# Change: 将蛇形4-连通路径拉直为 Nx1 单列，按遍历顺序输出每个像素的颜色... → 重新着色路径而非提取
#
# Reference-task calibration:
# - The source task 2b83f449 is useful as a baseline because it keeps the original geometry
#   and changes colors according to a structural role inferred from the input.
# - This variant follows the same complexity level: topology is read first, then recoloring
#   is applied in-place. The new structural role is "which traversal run-length bucket does
#   this snake segment belong to?" rather than "where should a token fall?".
#
# Design analysis required by the task:
# a) Concrete behavior and trace:
#    The input is still one 4-connected snake on a background. The old solver traced the snake
#    from the top-left endpoint and exported the visited colors as an N x 1 strip.
#    The variant keeps the trace step but changes the last step:
#    - old: path colors [8,8,8,3,3,7,7,7,7,7] -> output that exact color list as a column
#    - new: split that list into same-color runs [8x3, 3x2, 7x5], keep those cells in place,
#      recolor them by run length bucket: 3 -> color 2, 2 -> color 2, 5 -> color 4.
#    So the path geometry is unchanged, but the output now highlights run lengths rather than
#    extracting raw colors.
# b) Complexity calibration:
#    A plain global color substitution would be too weak and visually close to a lookup table.
#    Recoloring traversal runs preserves the original path-reading dependency, so the task still
#    requires endpoint detection and ordered traversal. The generator forces short, medium and
#    long runs to coexist, making the D1 change visually obvious.
# c) Hidden assumptions in the original main():
#    - `bg = most common color` still holds.
#    - `mask = g != bg` still isolates the snake because only the transform changed.
#    - Endpoint detection and single-path traversal are still valid and are reused unchanged.
#    - The original `return np.array(cols).reshape((-1, 1))` is no longer valid: the output must
#      stay on the full canvas.
#    - Reading each cell color is not the end result anymore; colors are only used to detect
#      maximal same-color runs along traversal order.
# d) Placement legality:
#    The generator still builds one non-self-touching snake inside a 16x16 grid. Adjacent runs
#    always use different input colors so run boundaries are recoverable. The solver assumes one
#    simple path with exactly two endpoints, and the generator verifies every produced sample with
#    `main(input)`.
# e) Visual observability:
#    Across examples the same three output colors always mean short / medium / long runs, while
#    the input colors vary independently. Because the snake stays in place, a human can compare
#    the run lengths directly on the input and see the recolor in the output.
# 9) New assumptions introduced by the new main():
#    - Maximal runs are detected by color changes along the traced path, so adjacent runs must
#      have different input colors. The generator enforces that.
#    - For clean visual contrast, generator inputs avoid output colors {2,4,6} and background
#      colors also avoid them. The solver itself does not require this, but sample construction
#      guarantees it.

from collections import Counter, defaultdict
import random

import numpy as np


SHORT_COLOR = 2
MEDIUM_COLOR = 4
LONG_COLOR = 6
OUTPUT_COLORS = {SHORT_COLOR, MEDIUM_COLOR, LONG_COLOR}
RUN_LENGTH_BUCKETS = {
    "short": [2, 3],
    "medium": [4, 5],
    "long": [6, 7, 8],
}
BG_CHOICES = [0, 1, 3, 5, 7, 8, 9]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def _trace_path(mask: np.ndarray) -> list[tuple[int, int]]:
    pts = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    if not pts:
        return []
    if len(pts) == 1:
        return [pts[0]]

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
    prev: tuple[int, int] | None = None
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


def _bucket_color(run_len: int) -> int:
    if run_len <= 3:
        return SHORT_COLOR
    if run_len <= 5:
        return MEDIUM_COLOR
    return LONG_COLOR


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor a snake in place according to same-color run lengths along traversal order.

    Input structure:
    - Background is the most common color.
    - Non-background cells form a single simple 4-connected path with two endpoints.
    - Along the path, colors appear in contiguous runs.

    Rule:
    - Start from the top-left endpoint and trace the whole snake.
    - Split the traversal into maximal same-color runs.
    - Recolor each run in place by its length:
      * 2-3 cells -> color 2
      * 4-5 cells -> color 4
      * 6+ cells  -> color 6
    - Return the full grid, keeping the snake shape unchanged.
    """
    g = np.array(grid, dtype=int)
    bg = _get_bg(g)
    mask = g != bg
    order = _trace_path(mask)
    if not order:
        return g.copy()

    out = g.copy()
    path_colors = [int(g[r, c]) for r, c in order]

    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and path_colors[j] == path_colors[i]:
            j += 1
        new_color = _bucket_color(j - i)
        for r, c in order[i:j]:
            out[r, c] = new_color
        i = j

    return out


def _try_extend(
    path_set: set[tuple[int, int]],
    last: tuple[int, int],
    dr: int,
    dc: int,
    count: int,
    height: int,
    width: int,
) -> list[tuple[int, int]]:
    segment: list[tuple[int, int]] = []
    r, c = last
    for i in range(1, count + 1):
        nr, nc = r + dr * i, c + dc * i
        if nr < 1 or nr >= height - 1 or nc < 1 or nc >= width - 1:
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


def _sample_run_lengths() -> list[int] | None:
    labels = list(RUN_LENGTH_BUCKETS)
    for _ in range(200):
        num_segments = random.randint(5, 6)
        categories = labels + [random.choice(labels) for _ in range(num_segments - len(labels))]
        random.shuffle(categories)
        lengths = [random.choice(RUN_LENGTH_BUCKETS[label]) for label in categories]
        if 20 <= sum(lengths) <= 30:
            return lengths
    return None


def _choose_input_colors(lengths: list[int], bg: int) -> list[int] | None:
    buckets = [_bucket_color(length) for length in lengths]
    available = [c for c in range(10) if c != bg and c not in OUTPUT_COLORS]
    for _ in range(200):
        palette = random.sample(available, 3)
        colors: list[int] = []
        prev = None
        for _ in lengths:
            choices = [c for c in palette if c != prev]
            col = random.choice(choices)
            colors.append(col)
            prev = col

        color_to_buckets: dict[int, set[int]] = defaultdict(set)
        bucket_to_colors: dict[int, set[int]] = defaultdict(set)
        for col, bucket in zip(colors, buckets):
            color_to_buckets[col].add(bucket)
            bucket_to_colors[bucket].add(col)

        reused_across_buckets = any(len(v) >= 2 for v in color_to_buckets.values())
        mixed_bucket = any(len(v) >= 2 for v in bucket_to_colors.values())
        if reused_across_buckets and mixed_bucket:
            return colors
    return None


def _build_snake(target_len: int, height: int, width: int) -> list[tuple[int, int]] | None:
    for _ in range(200):
        start_r = random.randint(1, 4)
        start_c = random.randint(1, width - 2)
        path = [(start_r, start_c)]
        path_set = {(start_r, start_c)}
        turns = 0
        is_vertical = True
        h_dir = random.choice([-1, 1])

        while len(path) < target_len:
            run_len = min(random.randint(1, 5), target_len - len(path))
            last = path[-1]

            if is_vertical:
                v_dir = 1 if random.random() < 0.85 else -1
                segment = _try_extend(path_set, last, v_dir, 0, run_len, height, width)
                if not segment:
                    segment = _try_extend(path_set, last, -v_dir, 0, run_len, height, width)
            else:
                segment = _try_extend(path_set, last, 0, h_dir, run_len, height, width)
                if not segment:
                    segment = _try_extend(path_set, last, 0, -h_dir, run_len, height, width)
                    if segment:
                        h_dir = -h_dir
                h_dir = -h_dir

            if not segment:
                break

            path.extend(segment)
            turns += 1
            is_vertical = not is_vertical

        if len(path) != target_len or not (7 <= turns <= 11):
            continue

        mask = np.zeros((height, width), dtype=bool)
        for r, c in path:
            mask[r, c] = True
        order = _trace_path(mask)
        if len(order) == target_len:
            return order

    return None


def _paint_runs(
    shape: tuple[int, int],
    order: list[tuple[int, int]],
    lengths: list[int],
    colors: list[int],
    bg: int,
) -> np.ndarray:
    grid = np.full(shape, bg, dtype=int)
    idx = 0
    for seg_len, col in zip(lengths, colors):
        for r, c in order[idx : idx + seg_len]:
            grid[r, c] = col
        idx += seg_len
    assert idx == len(order)
    return grid


def _gen_attempt() -> np.ndarray | None:
    height, width = 16, 16
    bg = random.choice(BG_CHOICES)

    lengths = _sample_run_lengths()
    if lengths is None:
        return None

    order = _build_snake(sum(lengths), height, width)
    if order is None:
        return None

    input_colors = _choose_input_colors(lengths, bg)
    if input_colors is None:
        return None

    inp = _paint_runs((height, width), order, lengths, input_colors, bg)
    expected = _paint_runs(
        (height, width),
        order,
        lengths,
        [_bucket_color(length) for length in lengths],
        bg,
    )

    # Prevent silent degeneration: every path cell should actually change color.
    if np.any((inp != bg) & (inp == expected)):
        return None
    if not np.array_equal(main(inp), expected):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("Failed to generate input after 300 attempts")
