# Combo variant of 16de56c4 (index 9)
# combo_spec: D3/1+D4/3
# Name: 周期性图案重复+色块分组

from __future__ import annotations

from collections import defaultdict
import importlib.util
import random

import numpy as np

from arc_utils import get_bg


# Design analysis
# spec:
#   fused_rule_summary:
#     - 保留原题“唯一射击边 -> 按行/按列传播”的大框架。
#     - 不再用两个单像素 marker 定义 beat；每条活跃线改为两个相同的连通小条块。
#     - 条块长度 = step。条块内部的颜色序列就是可见模板，模板后的等长背景段一起构成周期。
#     - 输出时从射击边开始，按 [条块模板, 等长背景] 的 2*step 周期整线平铺。
#   concrete_trace:
#     - 例：left edge 某行 step=3，边缘条块为 [2, 5, 2]，输入前 9 格可写成
#       [2,5,2,0,0,0,2,5,2]。
#     - 原题会从两个同色点推出“每隔 step 打一个点”，输出类似 [2,0,0,2,0,0,...]。
#     - 本融合版读取的是长度为 3 的整段模板 [2,5,2]，并把 [2,5,2,0,0,0]
#       视为一个完整周期，输出变成 [2,5,2,0,0,0,2,5,2,0,0,0,...]。
#     - 差异在于：step 仍然来自结构尺寸，但被复制的是小色块模板，而不是单色 beat 点列。
#   complexity_calibration:
#     - 条块必须是多色模板，且模板内部至少有一个颜色重复出现；否则原题 solver 可能把各颜色拆成独立 beat 而退化。
#     - generator 只取 left/right 射线，并把 step 收紧到 3-4；这是因为原始高度上限 12 会把 top/bottom 样例压回
#       只能使用 step=2 的退化区间。
#     - 继续要求 line_len > 4*step，确保输出至少会在第二个样本条块之外继续生成。
#     - 可选 bystander 只允许落在周期背景段的内部相位，避免与模板连成一体或被输出覆盖。
#     - 每个网格使用 3-6 色的小调色板，保证不同线之间既有重复支撑，又不会因颜色过多而杂乱。
#   implicit_assumptions_in_main:
#     - 原 solver 假设“按颜色计数能找到两个 marker 点”；这里改为“按连续非背景段扫描能找到两个相同模板段”。
#     - 原 solver 用两个点间距直接当 step；这里改为用边缘条块长度当 step，并要求第二条块与之同长同序且隔着等长背景。
#     - 原 solver 的写入单位是单个 beat 位置；这里改为整段模板复制，所以 line 内部的相邻非背景像素不再表示冲突。
#     - 新 solver 默认每条活跃线恰有一个边缘模板候选；generator 通过固定布局和禁止贴邻 bystander 保证这一点。
#   placement_legality:
#     - 线内布局固定为“边缘模板 / 等长背景 / 第二模板”，所以需要 line_len >= 3*step 才能放下输入，
#       且 line_len > 4*step 才能保证输出真的超出输入。
#     - bystander 若存在，只能放在更远处背景半周期的内部格位，既不越界，也不与模板段相邻。
#     - 活跃线索引继续保持至少 1 格间隔，减少平行线之间的视觉串扰。
#     - 仍检查非射击边上的边界计数，确保射击边唯一可见。
#   observability_and_cleanliness:
#     - 人类能直接看到“一个小色块在边上出现一次，又隔着同宽空隙重复一次”，因此容易读出模板与周期。
#     - 输出只做一种动作：把这个块-空隙节奏继续延伸。没有 companion 改色、没有多套条件分支，归纳负担更小。
#     - 不使用颜色值大小、排序等不可视信息；只依赖条块长度、相等模板、与边的相对位置。
#   anti_degeneration_check:
#     - 关键控制变量是“模板内部多色序列”和“等长背景半周期”。若内部不是多色，或新输出与原题 solver
#       在同一输入上经常一致，就说明设计退化。下面的 variant_metrics 会统计这些量。


_ROWS_RANGE = (7, 12)
_COLS_RANGE = (9, 21)
_ACTIVE_LINES_RANGE = (4, 9)
_STEP_RANGE = (3, 4)
_ALL_COLORS = list(range(1, 10))
_EDGES = ("left", "right")

_PALETTE_SIZE_RANGE = (3, 6)
_P_BYSTANDER = 0.45

_MAX_RETRIES = 400


def _edge_info(g: np.ndarray, bg: int) -> tuple[str, bool, int, int]:
    H, W = g.shape
    edge_counts = {
        "top": sum(1 for c in range(W) if g[0, c] != bg),
        "bottom": sum(1 for c in range(W) if g[H - 1, c] != bg),
        "left": sum(1 for r in range(H) if g[r, 0] != bg),
        "right": sum(1 for r in range(H) if g[r, W - 1] != bg),
    }
    max_count = max(edge_counts.values())
    assert max_count > 0, "No non-bg cells on any edge"
    winners = [edge for edge, count in edge_counts.items() if count == max_count]
    assert len(winners) == 1, f"Ambiguous shooting edge: {winners} (counts={edge_counts})"
    edge = winners[0]
    by_row = edge in ("left", "right")
    num_lines = H if by_row else W
    line_len = W if by_row else H
    return edge, by_row, num_lines, line_len


def _choose_dims(edge: str, n_active: int) -> tuple[int, int, int, int] | None:
    min_num_lines = 2 * n_active - 1
    by_row = edge in ("left", "right")
    if by_row:
        row_lo = max(_ROWS_RANGE[0], min_num_lines)
        if row_lo > _ROWS_RANGE[1]:
            return None
        H = random.randint(row_lo, _ROWS_RANGE[1])
        W = random.randint(max(_COLS_RANGE[0], 10), _COLS_RANGE[1])
        return H, W, H, W

    col_lo = max(_COLS_RANGE[0], min_num_lines)
    if col_lo > _COLS_RANGE[1]:
        return None
    W = random.randint(col_lo, _COLS_RANGE[1])
    H = random.randint(max(_ROWS_RANGE[0], 9), _ROWS_RANGE[1])
    return H, W, W, H


def _spaced_indices(n: int, total: int) -> list[int]:
    available = total - (n - 1)
    if available < n:
        return []
    picks = sorted(random.sample(range(available), n))
    return [picks[i] + i for i in range(n)]


def _scan_segments(vals: list[int], bg: int) -> list[tuple[int, int, tuple[int, ...]]]:
    segments: list[tuple[int, int, tuple[int, ...]]] = []
    p = 0
    n = len(vals)
    while p < n:
        if vals[p] == bg:
            p += 1
            continue
        start = p
        seq = [vals[p]]
        p += 1
        while p < n and vals[p] != bg:
            seq.append(vals[p])
            p += 1
        segments.append((start, p - 1, tuple(seq)))
    return segments


def _count_other_edge_dots(
    dots_rc: list[tuple[int, int]], edge: str, H: int, W: int
) -> dict[str, int]:
    boundary = {
        "top": lambda r, c: r == 0,
        "bottom": lambda r, c: r == H - 1,
        "left": lambda r, c: c == 0,
        "right": lambda r, c: c == W - 1,
    }
    counts = {}
    for other in ("top", "bottom", "left", "right"):
        if other == edge:
            continue
        check = boundary[other]
        counts[other] = sum(1 for r, c in dots_rc if check(r, c))
    return counts


def _sample_motif(step: int, palette: list[int]) -> tuple[int, ...]:
    n_colors = random.randint(2, min(3, step, len(palette)))
    motif_colors = random.sample(palette, n_colors)
    motif = [motif_colors[0]]
    for _ in range(1, step):
        motif.append(random.choice(motif_colors))
    if len(set(motif)) == 1:
        motif[-1] = motif_colors[1]
    if len(set(motif)) == 1 or max(motif.count(color) for color in set(motif)) < 2:
        return _sample_motif(step, palette)
    return tuple(motif)


def _build_line_segments(
    step: int, motif: tuple[int, ...], line_len: int
) -> tuple[list[tuple[int, int]], str]:
    cells = [(dist, color) for dist, color in enumerate(motif)]
    cells.extend((2 * step + dist, color) for dist, color in enumerate(motif))

    line_kind = "plain"
    if step >= 3 and random.random() < _P_BYSTANDER:
        candidates = [
            dist
            for dist in range(3 * step, line_len)
            if step < (dist % (2 * step)) < (2 * step - 1)
        ]
        used = {dist for dist, _ in cells}
        candidates = [dist for dist in candidates if dist not in used and dist - 1 not in used and dist + 1 not in used]
        if candidates:
            line_kind = "bystander"
            cells.append((random.choice(candidates), random.choice([c for c in _ALL_COLORS if c not in motif])))
    return cells, line_kind


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repeat a grouped edge template as a periodic block-and-gap ray from one dominant border.

    Input structure:
    - Background is the most common color. Non-background cells appear on a sparse set of
      rows or columns launched from a unique border edge.
    - On each active line, the informative source is not a single pixel but a short connected
      bar touching the shooting edge. A second identical bar appears farther inward.
    - The bar length is the step. Reading the bar from the edge inward gives a short color
      template, and the empty gap between the two copies has the same length as the bar.
    - Any extra non-background cell is a bystander placed inside a phase that the repeated
      template leaves as background.

    Rule:
    - Identify the unique shooting edge by counting boundary non-background cells; left/right
      propagate along rows, top/bottom along columns.
    - For each active line, read positions from the shooting edge inward and find the unique
      edge-touching non-background segment whose sequence reappears once after an equally long
      background gap.
    - Let that segment have length step and color sequence motif. Build a period of length
      2*step: first the motif, then step background cells.
    - Tile that period across the whole line, anchored at the shooting edge. Background phases
      are left untouched, so bystanders on those phases remain in place.

    Key insight: the grouped source bar now does two jobs at once. Its size encodes the rhythm,
    and its internal color order supplies the repeat template; solving the task means extending
    that block-plus-gap cadence outward from the dominant edge.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))
    edge, by_row, num_lines, line_len = _edge_info(g, bg)

    def cell(line: int, pos: int) -> tuple[int, int]:
        return (line, pos) if by_row else (pos, line)

    def pos_from_dist(dist: int) -> int:
        if edge in ("left", "top"):
            return dist
        return line_len - 1 - dist

    for li in range(num_lines):
        vals = [int(g[cell(li, pos_from_dist(dist))]) for dist in range(line_len)]
        if not any(v != bg for v in vals):
            continue

        segments = _scan_segments(vals, bg)
        candidates: list[tuple[int, tuple[int, ...]]] = []
        for start, end, seq in segments:
            if start != 0:
                continue
            step = end - start + 1
            expected_start = 2 * step
            expected_end = 3 * step - 1
            if expected_end >= line_len:
                continue
            match = [seg for seg in segments if seg[0] == expected_start and seg[1] == expected_end and seg[2] == seq]
            if match:
                candidates.append((step, seq))

        assert len(candidates) == 1, (
            f"Line {li} ({'row' if by_row else 'col'}) expected one grouped template, "
            f"got {candidates} from segments={segments}"
        )
        step, motif = candidates[0]
        period = 2 * step

        for dist in range(line_len):
            phase = dist % period
            if phase >= step:
                continue
            color = int(motif[phase])
            if color == bg:
                continue
            out[cell(li, pos_from_dist(dist))] = color

    return out


def _gen_attempt() -> tuple[np.ndarray, dict[str, int]] | None:
    n_active = random.randint(*_ACTIVE_LINES_RANGE)
    edge = random.choice(_EDGES)
    dims = _choose_dims(edge, n_active)
    if dims is None:
        return None
    H, W, num_lines, line_len = dims
    by_row = edge in ("left", "right")

    max_step = min(_STEP_RANGE[1], (line_len - 1) // 4)
    if max_step < _STEP_RANGE[0]:
        return None

    active_lines = _spaced_indices(n_active, num_lines)
    if not active_lines:
        return None

    palette_size = random.randint(_PALETTE_SIZE_RANGE[0], min(_PALETTE_SIZE_RANGE[1], len(_ALL_COLORS)))
    palette = random.sample(_ALL_COLORS, palette_size)

    def cell(line: int, pos: int) -> tuple[int, int]:
        return (line, pos) if by_row else (pos, line)

    def pos_from_dist(dist: int) -> int:
        if edge in ("left", "top"):
            return dist
        return line_len - 1 - dist

    inp = np.zeros((H, W), dtype=int)
    all_rc: list[tuple[int, int]] = []
    colorful_lines = 0
    bystander_lines = 0
    extension_gain = 0
    steps_used: list[int] = []

    for li in active_lines:
        step = random.randint(_STEP_RANGE[0], max_step)
        motif = _sample_motif(step, palette)
        cells, line_kind = _build_line_segments(step, motif, line_len)

        for dist, color in cells:
            r, c = cell(li, pos_from_dist(dist))
            if inp[r, c] != 0:
                return None
            inp[r, c] = color
            all_rc.append((r, c))

        steps_used.append(step)
        colorful_lines += int(len(set(motif)) >= 2)
        bystander_lines += int(line_kind == "bystander")
        extension_gain += sum(1 for dist in range(4 * step, line_len) if (dist % (2 * step)) < step)

    other_counts = _count_other_edge_dots(all_rc, edge, H, W)
    shooting_edge_dots = len(active_lines)
    if any(count >= shooting_edge_dots for count in other_counts.values()):
        return None

    try:
        out = main(inp)
    except Exception:
        return None
    if np.array_equal(inp, out):
        return None

    metrics = {
        "active_lines": n_active,
        "colorful_lines": colorful_lines,
        "bystander_lines": bystander_lines,
        "extension_gain": extension_gain,
        "max_step": max(steps_used),
    }

    if colorful_lines != n_active:
        return None
    if extension_gain < n_active:
        return None

    return inp, metrics


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result[0]
    raise RuntimeError(f"Failed to generate valid input after {_MAX_RETRIES} attempts")


def variant_metrics(samples: int = 64) -> dict[str, float]:
    totals = defaultdict(int)
    for _ in range(samples):
        result = None
        while result is None:
            result = _gen_attempt()
        _, metrics = result
        for key, value in metrics.items():
            totals[key] += value

    return {
        "avg_active_lines": totals["active_lines"] / samples,
        "avg_colorful_lines": totals["colorful_lines"] / samples,
        "avg_bystander_lines": totals["bystander_lines"] / samples,
        "avg_extension_gain": totals["extension_gain"] / samples,
        "avg_max_step": totals["max_step"] / samples,
    }


def compare_with_original(samples: int = 32) -> dict[str, float]:
    path = "/home/lixinyu/snap/new/arc2_solver_select/16de56c4.py"
    spec = importlib.util.spec_from_file_location("orig_16de56c4", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    same = 0
    for _ in range(samples):
        inp = generate_input()
        try:
            original_out = mod.main(inp)
        except Exception:
            original_out = None
        if original_out is not None and np.array_equal(main(inp), original_out):
            same += 1

    return {
        "same_output_rate": same / samples,
        "different_output_rate": (samples - same) / samples,
    }
