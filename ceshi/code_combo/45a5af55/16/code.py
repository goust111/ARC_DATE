# Combo variant of 45a5af55 (index 16)
# combo_spec: D11/0+D13/0
# Name: 分隔行划分多组并排嵌套

from __future__ import annotations

import numpy as np

BG = 0
INNER_GAP_ROWS = 1
GROUP_GAP_ROWS = 2
PACK_GAP = 1
MAX_SIDE = 30

# Design analysis for the combo variant
# a) 具体化与行为追踪
#    - 原题把整张图视为一个连续色条堆栈，自底向上读取所有满宽色条，生成一个同心方框。
#    - 单独套 D11 时，可以把若干“原题子实例”用背景行切开，再把每组的结果并排输出。
#    - 单独套 D13 时，色条之间会出现背景分隔行，因此解析步骤必须跳过这些格式行。
#    - 融合后如果仍把“所有背景行都只当噪声跳过”，那 D11 的分组会丢失；如果把“所有背景行都当组边界”，
#      那 D13 的组内分隔又退化掉。这里显式使用两档背景分隔高度：1 行表示组内相邻色条的格式分隔，
#      2 行表示组与组之间的边界。例：上到下若看到
#      [A][0][B][00][C][0][D]，则原题会误把 0 当色条或把全部条纹混成一组；融合后应解析为两组：
#      group1=(A,B)，group2=(C,D)。每组内部仍按“自底向上、厚度=高度”的原规则生成方框，
#      输出为 frame(group1) 0 frame(group2)，并且两个方框底边对齐。
# b) 复杂度校准
#    - 只使用 1 行和 2 行两种背景间隔，档位少、差异清晰，人眼能从少量样例归纳。
#    - 组数限制为 2-4；每组色条数限制为 2-4。这样既能稳定出现“多组并排”，又能稳定出现“组内分隔”。
#    - 色条高度限制为 1-3，足以制造不同框厚，但避免输出尺寸失控。
# c) 隐式假设审查
#    - 原代码 `while r >= 0` 默认整图只有一组连续色条；融合后不成立，必须先按背景行高度切组。
#    - 原代码 `col = g[r, 0]` 默认每行都是有效色条；融合后背景行必须被识别为格式行。
#    - 原代码按“同色连续行”测高度，仍然有效，但前提变成：每条非背景行是满宽纯色行。
#    - 原代码把所有 runs 塞进一个 `cur`；融合后需要 `groups -> runs -> frames -> packed output`。
#    - 新冲突点是“哪些背景行只是组内分隔，哪些是组间边界”；这里由分隔行高度 1 vs 2 消解。
# d) 放置合法性
#    - 生成器只放置满宽纯色条和满宽背景分隔行，因此输入不存在重叠。
#    - 输出先独立构造每组方框，再写入统一背景画布，采用固定 1 列空隙并底边对齐，互不覆盖。
#    - 生成器同时约束输入高、输出宽和输出高均不超过 30。
# e) 视觉可观察性
#    - 输入中总能看到至少两组条纹堆栈，且组内是“1 行空白断开”，组间是“2 行空白断开”。
#    - 输出中总能看到多个彼此分离的同心方框，从而支持解题者推断“先分组，再逐组套原规则，最后横向拼接”。
# f) 可视化整洁度
#    - 背景只承担一种职责：垂直分隔。没有额外噪点、标记或局部遮挡。
#    - 输出使用 1 列背景间距，防止相邻外框在视觉上粘连成一个更大的结构。
# g) 变化的可推断性
#    - 颜色完全沿用原题语义，不做映射；0 只作背景分隔色。
#    - 真正新增的规则只依赖可见结构：背景分隔高度和分组后的并排布局。
# h) 新 solver 假设与 generator 保证
#    - solver 假设所有非背景行都是满宽纯色条；generator 强制满足。
#    - solver 假设组内分隔恰为 1 行背景、组间分隔至少 2 行背景；generator 固定为 1 和 2。
#    - generator 额外拒绝“把全部色条并成单组时输出恰好相同”的退化样本，确保融合行为确实起作用。


def _parse_row_runs(grid: np.ndarray) -> list[tuple[int, int]]:
    """Compress the grid into top-to-bottom runs of uniform full-width rows."""
    g = np.array(grid, dtype=int)
    h = int(g.shape[0])
    runs: list[tuple[int, int]] = []
    r = 0
    while r < h:
        col = int(g[r, 0])
        assert bool((g[r] == col).all())
        run_h = 1
        rr = r + 1
        while rr < h and int(g[rr, 0]) == col:
            assert bool((g[rr] == col).all())
            run_h += 1
            rr += 1
        runs.append((col, run_h))
        r = rr
    return runs


def _parse_groups(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    """
    Parse the fused input into groups of stripes.

    Returns groups in top-to-bottom order; each group's stripe runs are returned
    in bottom-to-top order to match the original nesting logic.
    """
    groups: list[list[tuple[int, int]]] = []
    current_top_down: list[tuple[int, int]] = []

    for col, run_h in _parse_row_runs(grid):
        if col == BG:
            if run_h >= GROUP_GAP_ROWS:
                if current_top_down:
                    groups.append(list(reversed(current_top_down)))
                    current_top_down = []
            else:
                assert run_h == INNER_GAP_ROWS
                assert current_top_down
            continue
        current_top_down.append((int(col), int(run_h)))

    if current_top_down:
        groups.append(list(reversed(current_top_down)))

    assert groups
    return groups


def _build_frame(runs: list[tuple[int, int]]) -> np.ndarray:
    """Apply the original many:1 stripe-to-nested-square rule to one group."""
    assert runs
    c0, h0 = runs[0]
    cur = np.full((h0, h0), c0, dtype=int)
    for col, thickness in runs[1:]:
        new_n = int(cur.shape[0] + 2 * thickness)
        nxt = np.full((new_n, new_n), col, dtype=int)
        nxt[thickness : thickness + cur.shape[0], thickness : thickness + cur.shape[1]] = cur
        cur = nxt
    return cur


def _pack_frames(frames: list[np.ndarray]) -> np.ndarray:
    """Pack frames left-to-right, bottom aligned, with a 1-pixel background gap."""
    assert frames
    out_h = max(int(frame.shape[0]) for frame in frames)
    out_w = sum(int(frame.shape[1]) for frame in frames) + PACK_GAP * (len(frames) - 1)
    out = np.full((out_h, out_w), BG, dtype=int)
    c = 0
    for frame in frames:
        fh, fw = frame.shape
        out[out_h - fh :, c : c + fw] = frame
        c += fw + PACK_GAP
    return out


def _single_group_output_from_all_runs(groups: list[list[tuple[int, int]]]) -> np.ndarray:
    """
    Reference output for the degenerate interpretation that ignores group
    boundaries and nests every stripe into one giant frame.
    """
    flattened: list[tuple[int, int]] = []
    for runs in reversed(groups):
        flattened.extend(runs)
    return _build_frame(flattened)


def _fusion_signature(grid: np.ndarray) -> dict[str, int]:
    """
    Quantify whether a sample really exercises the combo rule.

    - n_groups: how many independently nested frames should be produced.
    - n_inner_gaps: count of 1-row separators inside groups.
    - n_group_gaps: count of 2+-row separators between groups.
    """
    row_runs = _parse_row_runs(grid)
    n_inner_gaps = sum(1 for col, h in row_runs if col == BG and h == INNER_GAP_ROWS)
    n_group_gaps = sum(1 for col, h in row_runs if col == BG and h >= GROUP_GAP_ROWS)
    return {
        "n_groups": n_group_gaps + 1,
        "n_inner_gaps": n_inner_gaps,
        "n_group_gaps": n_group_gaps,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combo rule:
    - Full-width non-zero stripes are arranged in multiple vertical groups.
    - A 1-row background band separates adjacent stripes within the same group.
    - A 2-row background band separates one group from the next group.
    - Each group independently builds one nested square frame from bottom to top.
    - The resulting frames are packed left-to-right in input group order.
    """
    g = np.array(grid, dtype=int)
    groups = _parse_groups(g)
    frames = [_build_frame(runs) for runs in groups]
    return _pack_frames(frames)


def _sample_group() -> list[tuple[int, int]]:
    n_runs = int(np.random.randint(2, 5))
    heights = [int(np.random.randint(1, 4)) for _ in range(n_runs)]
    colors: list[int] = []
    prev = None
    for _ in range(n_runs):
        c = int(np.random.randint(1, 10))
        while c == BG or c == prev:
            c = int(np.random.randint(1, 10))
        colors.append(c)
        prev = c
    return list(zip(colors, heights))


def _frame_size(runs: list[tuple[int, int]]) -> int:
    center = runs[0][1]
    border = sum(height for _color, height in runs[1:])
    return int(center + 2 * border)


def _render_group_input(runs: list[tuple[int, int]], width: int) -> np.ndarray:
    """Render one group's stripes top-to-bottom with 1-row internal separators."""
    parts: list[np.ndarray] = []
    top_down = list(reversed(runs))
    for idx, (color, height) in enumerate(top_down):
        parts.append(np.full((height, width), color, dtype=int))
        if idx != len(top_down) - 1:
            parts.append(np.full((INNER_GAP_ROWS, width), BG, dtype=int))
    return np.vstack(parts)


def _gen_attempt() -> np.ndarray | None:
    n_groups = int(np.random.randint(2, 5))
    width = int(np.random.randint(6, 16))

    groups = [_sample_group() for _ in range(n_groups)]
    frame_sizes = [_frame_size(runs) for runs in groups]
    group_blocks = [_render_group_input(runs, width) for runs in groups]

    total_input_h = sum(int(block.shape[0]) for block in group_blocks) + GROUP_GAP_ROWS * (n_groups - 1)
    total_output_w = sum(frame_sizes) + PACK_GAP * (n_groups - 1)
    total_output_h = max(frame_sizes)

    if width > MAX_SIDE or total_input_h > MAX_SIDE:
        return None
    if total_output_w > MAX_SIDE or total_output_h > MAX_SIDE:
        return None

    # Keep the visual grouping obvious across examples.
    if len(set(frame_sizes)) == 1 and np.random.rand() < 0.6:
        return None
    if len({len(runs) for runs in groups}) == 1 and np.random.rand() < 0.35:
        return None

    parts: list[np.ndarray] = []
    for idx, block in enumerate(group_blocks):
        parts.append(block)
        if idx != len(group_blocks) - 1:
            parts.append(np.full((GROUP_GAP_ROWS, width), BG, dtype=int))
    grid = np.vstack(parts)

    parsed_groups = _parse_groups(grid)
    expected = _pack_frames([_build_frame(runs) for runs in parsed_groups])
    actual = main(grid)
    if not np.array_equal(actual, expected):
        raise RuntimeError("generator produced an invalid sample")

    # Reject samples that accidentally collapse to the single-frame interpretation.
    if np.array_equal(actual, _single_group_output_from_all_runs(parsed_groups)):
        return None

    sig = _fusion_signature(grid)
    if sig["n_groups"] < 2 or sig["n_inner_gaps"] < n_groups or sig["n_group_gaps"] != n_groups - 1:
        raise RuntimeError("generator produced a non-fusion sample")

    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 500 attempts")
