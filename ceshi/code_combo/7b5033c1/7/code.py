# Combo variant of 7b5033c1 (index 7)
# combo_spec: D2/1+D7/2
# Name: 最长路径选择+段长度起点

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

try:
    from arc_utils import get_bg
except ModuleNotFoundError:
    sys.path.insert(0, str(Path("/home/lixinyu/snap/new/arc2_solver_select")))
    from arc_utils import get_bg


# Design analysis for the combo variant:
# a) 具体化与行为追踪:
#    输入不再只有一条蛇，而是 2-3 条互不接触的 4 连通彩色蛇。只有最长的那条需要输出。
#    原题流程是: 把所有非背景像素视为同一条链 -> 找全局左上端点 -> 沿链输出颜色。
#    融合后流程变成两级筛选:
#    1) 先按 4 连通分量拆开所有蛇，比较长度，只保留唯一最长蛇。
#    2) 在这条最长蛇内部，再比较两个端点所在连续同色段的长度；从较短色段那一端开始遍历。
#    具体例子: 左上角有一条长度 8 的干扰蛇，右下区域有一条长度 23 的目标蛇。目标蛇左上端点
#    所在色段长度为 7，另一端色段长度为 3。原题代码会被左上短蛇吸走，只输出 8 个像素；
#    只做 D7 变体时会改为选长度 23 的蛇，但仍从它的左上端点起步；本融合版还要继续翻转方向，
#    因为真正的起点应是色段长度 3 的那一端。差异因此同时体现在“选哪条蛇”和“该蛇从哪端读”。
# b) 复杂度校准:
#    如果最长蛇碰巧也是全局最左上蛇，或者其较短端点恰好也是左上端点，样本会部分退化。
#    生成器因此强制:
#    - 至少有一条更短的蛇位于更靠左上的区域，迫使 D7 的“先选最长”可见。
#    - 最长蛇的左上端点色段更长、另一端更短，迫使 D2 的“从短段端点起步”可见。
#    - 最长与次长之间至少拉开 6 格，避免“看起来都差不多长”的歧义。
# c) 隐式假设审查:
#    原 main() 的 `pts = argwhere(mask)` 默认所有非背景像素属于同一链；融合后失效，必须先分量化。
#    原 `deg` 和 `ends` 默认全局只有两个端点；融合后每条蛇各有两个端点，只能在选中分量内计算。
#    原 `start = min(ends)` 绑定“左上端点即起点”；融合后这只适合做色段比较时的基准方向，不再是最终起点。
#    原跟踪循环依赖“从当前像素出发未访问邻居唯一”；该前提在每个单独分量上仍成立，生成器继续保证简单链。
#    新增前提是“端点色段长度可从沿路径顺序的前缀/后缀 run 读出”；这要求着色按连续色段落在路径顺序上，
#    生成器会显式按段给路径上色。
# d) 放置合法性:
#    所有蛇都在 16x16 内部生长；新增像素前检查越界、重叠，以及除前驱外不存在额外 4 邻接。
#    每放完一条蛇就预留 8 邻域缓冲区，防止不同蛇接触或视觉粘连，保证分量分离且容易辨认。
# e) 视觉可观察性:
#    人类看到训练样例时，应能稳定观察到两件事:
#    - 左上角那条并不一定是答案，答案总是更长的蛇。
#    - 即使锁定了最长蛇，输出首色也来自它较短的端点色段，而不是几何上更左上的端点。
#    这两个现象在 2-4 个样例中都会重复出现，因此不是偶然排布。
# f) 可视化整洁度:
#    生成器把短干扰蛇放在左上，目标蛇放在离左上更远的位置，可选第三条短蛇放在右下。
#    再加 8 邻域空隙，避免多条折线互相贴边，看图时每条蛇都能单独分辨。
# g) 颜色变化可推断性:
#    颜色只承担“连续色段”这一结构作用，不引入额外颜色映射规则。每条蛇只用少量重复颜色，
#    相邻色段颜色不同，端点 run 长短由连续块长度直接体现，人类可以从几何+色段长度一起归纳规则。


Coord = tuple[int, int]


def _neighbors4(r: int, c: int) -> tuple[Coord, Coord, Coord, Coord]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _find_components(mask: np.ndarray) -> list[list[Coord]]:
    remaining = {(int(r), int(c)) for r, c in np.argwhere(mask)}
    components: list[list[Coord]] = []
    while remaining:
        start = remaining.pop()
        stack = [start]
        comp = [start]
        while stack:
            cur = stack.pop()
            for nb in _neighbors4(*cur):
                if nb in remaining:
                    remaining.remove(nb)
                    stack.append(nb)
                    comp.append(nb)
        components.append(comp)
    return components


def _component_degrees(component: list[Coord]) -> dict[Coord, int]:
    pset = set(component)
    return {p: sum(nb in pset for nb in _neighbors4(*p)) for p in component}


def _trace_path(component: list[Coord], start: Coord | None = None) -> list[Coord]:
    pset = set(component)
    deg = _component_degrees(component)
    ends = sorted(p for p in component if deg[p] == 1)
    assert len(ends) == 2, "Each snake must be a simple chain with two endpoints"

    cur = min(ends) if start is None else start
    prev: Coord | None = None
    order: list[Coord] = []
    for _ in range(len(component)):
        order.append(cur)
        nbrs = [nb for nb in _neighbors4(*cur) if nb in pset and nb != prev]
        if not nbrs:
            break
        assert len(nbrs) == 1, "Simple chain expected during tracing"
        prev, cur = cur, nbrs[0]

    assert len(order) == len(component), "Trace must visit every pixel exactly once"
    return order


def _endpoint_run_lengths(colors: list[int]) -> tuple[int, int]:
    prefix = 1
    while prefix < len(colors) and colors[prefix] == colors[0]:
        prefix += 1

    suffix = 1
    while suffix < len(colors) and colors[-1 - suffix] == colors[-1]:
        suffix += 1

    return prefix, suffix


def _select_longest_component(components: list[list[Coord]]) -> list[Coord]:
    return max(components, key=lambda comp: (len(comp), -min(comp)[0], -min(comp)[1]))


def _solve_selected_component(g: np.ndarray, component: list[Coord]) -> list[int]:
    # We intentionally trace from the component's top-left endpoint first, because that
    # reproduces the old direction inside the chosen snake and makes the D2 comparison local.
    order = _trace_path(component)
    colors = [int(g[r, c]) for r, c in order]
    prefix_len, suffix_len = _endpoint_run_lengths(colors)

    # If the opposite endpoint has the shorter run, reverse. Equal runs are ambiguous,
    # so we keep the old top-left direction as a deterministic tie-break.
    if suffix_len < prefix_len:
        colors.reverse()
    return colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combo rule:
    1. Split all non-background pixels into 2-3 snake components.
    2. Select the unique longest component.
    3. Trace that component, but start from the endpoint whose contiguous color run is shorter.
    4. Output the traced colors as an (N x 1) column.

    Post-change assumption audit for the solver:
    - `components = _find_components(mask)` assumes different snakes stay 4-disconnected.
      The generator enforces this with an 8-neighbor moat.
    - `_select_longest_component(...)` assumes the intended target is uniquely longest.
      The generator enforces a length gap of at least 6.
    - `_trace_path(component)` assumes the selected component is a simple chain with exactly
      two degree-1 endpoints. The path grower enforces that.
    - `_endpoint_run_lengths(colors)` assumes endpoint colors form contiguous path-order runs.
      The generator colors each snake in explicit contiguous segments.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    mask = g != bg
    if not np.any(mask):
        return np.zeros((0, 1), dtype=int)

    components = _find_components(mask)
    best = _select_longest_component(components)
    colors = _solve_selected_component(g, best)
    return np.array(colors, dtype=int).reshape((-1, 1))


def _random_partition(total: int, num_parts: int, min_len: int, max_len: int) -> list[int] | None:
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


def _buffer8(cells: list[Coord], H: int, W: int) -> set[Coord]:
    out: set[Coord] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W:
                    out.add((nr, nc))
    return out


def _try_extend(
    path_set: set[Coord],
    blocked: set[Coord],
    last: Coord,
    dr: int,
    dc: int,
    count: int,
    H: int,
    W: int,
) -> list[Coord]:
    segment: list[Coord] = []
    r, c = last
    for i in range(1, count + 1):
        nr, nc = r + dr * i, c + dc * i
        if nr < 1 or nr >= H - 1 or nc < 1 or nc >= W - 1:
            break
        if (nr, nc) in blocked or (nr, nc) in path_set:
            break

        pred = (r + dr * (i - 1), c + dc * (i - 1))
        if any(nb != pred and nb in path_set for nb in _neighbors4(nr, nc)):
            break

        segment.append((nr, nc))
        path_set.add((nr, nc))
    return segment


def _grow_snake(
    H: int,
    W: int,
    target_len: int,
    blocked: set[Coord],
    min_turns: int,
    preferred_cells: list[Coord] | None = None,
) -> list[Coord] | None:
    interior = [(r, c) for r in range(1, H - 1) for c in range(1, W - 1)]
    for _ in range(300):
        free = [p for p in interior if p not in blocked]
        if preferred_cells is not None:
            preferred_free = [p for p in preferred_cells if p not in blocked]
            start_pool = preferred_free if preferred_free else free
        else:
            start_pool = free
        if not start_pool:
            return None

        start = random.choice(start_pool)
        path = [start]
        path_set = {start}
        turns = 0
        is_vertical = random.choice([True, False])
        h_dir = random.choice([-1, 1])

        for _step in range(target_len * 5):
            if len(path) >= target_len:
                break

            run_len = min(random.randint(1, 5), target_len - len(path))
            last = path[-1]
            segment: list[Coord] = []

            if is_vertical:
                dirs = [1, -1]
                random.shuffle(dirs)
                for v_dir in dirs:
                    segment = _try_extend(path_set, blocked, last, v_dir, 0, run_len, H, W)
                    if segment:
                        break
            else:
                used_dir: int | None = None
                for cand in (h_dir, -h_dir):
                    segment = _try_extend(path_set, blocked, last, 0, cand, run_len, H, W)
                    if segment:
                        used_dir = cand
                        break
                h_dir = -used_dir if used_dir is not None else -h_dir

            if not segment:
                break

            path.extend(segment)
            turns += 1
            is_vertical = not is_vertical

        if len(path) == target_len and turns >= min_turns:
            return path
    return None


def _build_segment_colors(bg: int, num_segments: int, palette_size: int) -> list[int]:
    usable = [c for c in range(10) if c != bg]
    palette = random.sample(usable, palette_size)
    seg_colors = list(palette)
    random.shuffle(seg_colors)
    while len(seg_colors) < num_segments:
        choices = [c for c in palette if c != seg_colors[-1]]
        seg_colors.append(random.choice(choices))
    for i in range(1, len(seg_colors)):
        if seg_colors[i] == seg_colors[i - 1]:
            choices = [c for c in palette if c != seg_colors[i - 1]]
            seg_colors[i] = random.choice(choices)
    return seg_colors


def _make_target_colors(total: int, bg: int, path: list[Coord]) -> list[int] | None:
    # We want the top-left endpoint of the target to have the longer endpoint run, so the
    # D2 rule visibly flips direction after D7 has already chosen the longest component.
    short_len = random.randint(2, 4)
    long_len = random.randint(short_len + 3, 9)
    interior_total = total - short_len - long_len
    if interior_total < 6:
        return None

    min_interior_segments = 2
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
    palette_size = random.randint(3, min(5, num_segments, 5))
    seg_colors = _build_segment_colors(bg, num_segments, palette_size)
    if any(seg_colors[i] == seg_colors[i - 1] for i in range(1, num_segments)):
        return None

    colors: list[int] = []
    for seg_len, color in zip(seg_lengths, seg_colors):
        colors.extend([color] * seg_len)
    if len(colors) != total:
        return None
    return colors


def _make_distractor_colors(total: int, bg: int) -> list[int] | None:
    min_seg_len = 2 if total >= 10 else 1
    min_segments = 3 if total >= 10 else 2
    max_segments = min(5, total)
    if min_segments > max_segments:
        return None

    for _ in range(60):
        num_segments = random.randint(min_segments, max_segments)
        seg_lengths = _random_partition(total, num_segments, min_seg_len, max(5, total))
        if seg_lengths is None:
            continue

        palette_size = random.randint(2, min(4, num_segments, 4))
        seg_colors = _build_segment_colors(bg, num_segments, palette_size)

        colors: list[int] = []
        for seg_len, color in zip(seg_lengths, seg_colors):
            colors.extend([color] * seg_len)
        if len(colors) == total:
            return colors
    return None


def _path_colors_from_top_left(path: list[Coord], colors: list[int]) -> list[int]:
    return list(colors) if path[0] == min(path[0], path[-1]) else list(reversed(colors))


def _fusion_expected_colors(path: list[Coord], colors: list[int]) -> list[int]:
    top_left_colors = _path_colors_from_top_left(path, colors)
    prefix_len, suffix_len = _endpoint_run_lengths(top_left_colors)
    if suffix_len < prefix_len:
        return list(reversed(top_left_colors))
    return top_left_colors


def _paint_paths(bg: int, H: int, W: int, paths: list[list[Coord]], colors_per_path: list[list[int]]) -> np.ndarray:
    grid = np.full((H, W), bg, dtype=int)
    for path, colors in zip(paths, colors_per_path):
        for (r, c), color in zip(path, colors):
            grid[r, c] = color
    return grid


def _gen_attempt() -> np.ndarray | None:
    H, W = 16, 16
    bg = random.randint(0, 9)
    num_paths = random.choice([2, 3])

    longest_len = random.randint(20, 25)
    short1_len = random.randint(6, 10)
    lengths = [short1_len, longest_len]
    if num_paths == 3:
        short2_len = random.randint(5, min(11, longest_len - 6))
        lengths.append(short2_len)

    upper_left_cells = [(r, c) for r in range(1, 6) for c in range(1, 6)]
    away_from_corner = [(r, c) for r in range(1, H - 1) for c in range(1, W - 1) if r >= 4 or c >= 4]
    lower_right_cells = [(r, c) for r in range(7, H - 1) for c in range(7, W - 1)]

    blocked: set[Coord] = set()

    short1 = _grow_snake(H, W, lengths[0], blocked, min_turns=2, preferred_cells=upper_left_cells)
    if short1 is None:
        return None
    blocked |= _buffer8(short1, H, W)

    target = _grow_snake(H, W, lengths[1], blocked, min_turns=6, preferred_cells=away_from_corner)
    if target is None:
        return None
    blocked |= _buffer8(target, H, W)

    paths = [short1, target]
    if num_paths == 3:
        short2 = _grow_snake(H, W, lengths[2], blocked, min_turns=2, preferred_cells=lower_right_cells)
        if short2 is None:
            return None
        paths.append(short2)

    if min(min(path) for path in paths) == min(target):
        return None

    comp_lengths = sorted((len(path) for path in paths), reverse=True)
    if len(comp_lengths) < 2 or comp_lengths[0] - comp_lengths[1] < 6:
        return None

    target_colors = _make_target_colors(len(target), bg, target)
    if target_colors is None:
        return None

    colors_per_path: list[list[int]] = []
    for i, path in enumerate(paths):
        if path is target:
            colors_per_path.append(target_colors)
        else:
            colors = _make_distractor_colors(len(path), bg)
            if colors is None:
                return None
            colors_per_path.append(colors)

    grid = _paint_paths(bg, H, W, paths, colors_per_path)

    # Generator-side guards against both kinds of degeneration.
    target_top_left_colors = _path_colors_from_top_left(target, target_colors)
    prefix_len, suffix_len = _endpoint_run_lengths(target_top_left_colors)
    if prefix_len <= suffix_len:
        return None

    actual_out = main(grid).reshape(-1).tolist()
    expected_out = _fusion_expected_colors(target, target_colors)
    if actual_out != expected_out:
        return None

    # D7 must matter: the globally top-left component is not the target.
    components = _find_components(grid != bg)
    chosen = _select_longest_component(components)
    if set(chosen) != set(target):
        return None

    # D2 must matter inside the chosen target: the output differs from taking the target
    # from its top-left endpoint directly.
    if expected_out == target_top_left_colors:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(600):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate input after 600 attempts")


def _original_like_output(grid: np.ndarray) -> list[int]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    pts = [(int(r), int(c)) for r, c in np.argwhere(g != bg)]
    if not pts:
        return []

    pset = set(pts)
    deg = {p: sum(nb in pset for nb in _neighbors4(*p)) for p in pts}
    ends = [p for p in pts if deg[p] == 1]
    start = min(ends)

    order: list[Coord] = []
    prev: Coord | None = None
    cur = start
    for _ in range(len(pts)):
        order.append(cur)
        nbrs = [nb for nb in _neighbors4(*cur) if nb in pset and nb != prev]
        if not nbrs:
            break
        prev, cur = cur, min(nbrs)
    return [int(g[r, c]) for r, c in order]


def fusion_metrics(grid: np.ndarray) -> dict[str, int]:
    """
    Quantify whether generated samples really activate both control variables.

    Returned metrics:
    - longest_gap: longest component length minus second-longest length.
    - d7_active: 1 iff the globally top-left component is not the chosen target.
    - d2_active: 1 iff the target's top-left orientation is not the final output orientation.
    - original_len: output length produced by the original single-chain solver logic.
    - combo_len: output length under the combo solver.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    components = _find_components(g != bg)
    lengths = sorted((len(comp) for comp in components), reverse=True)
    best = _select_longest_component(components)
    best_order = _trace_path(best)
    best_top_left_colors = [int(g[r, c]) for r, c in best_order]
    combo_colors = _solve_selected_component(g, best)

    longest_gap = lengths[0] - lengths[1] if len(lengths) > 1 else lengths[0]
    global_top_left_component = min(components, key=lambda comp: min(comp))
    d7_active = int(set(global_top_left_component) != set(best))
    d2_active = int(combo_colors != best_top_left_colors)
    return {
        "longest_gap": int(longest_gap),
        "d7_active": d7_active,
        "d2_active": d2_active,
        "original_len": len(_original_like_output(g)),
        "combo_len": len(combo_colors),
    }

