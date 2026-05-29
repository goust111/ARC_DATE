# Variant of 45a5af55
# Dimension: D11 — 源对象基数与嵌套关系 (meta)
# Change: N个色条 → 1个嵌套框架结构（many:1 基数关系）... → 多组色条 → 多个嵌套框架并排输出（many:many）

from __future__ import annotations

import numpy as np

BG = 0
GAP = 1
MAX_SIDE = 30

# Design analysis for the D11 variant
# a) 具体化与行为追踪
#    - 原题输入是单组、满宽、无背景间隔的水平色条，整张图只会生成一个同心方框。
#    - 本变体把输入改成“多组色条堆栈按竖直方向串联”，组与组之间插入 1-2 行纯背景色 0。
#    - 每组内部仍保留原题规则：自底向上读取色条，最底层变中心实心方块，后续各层按自身高度向外包裹。
#    - 输出不再只有一个结果，而是把每组各自生成的方框按输入中从上到下的组顺序，改为从左到右并排放置，
#      所有方框底边对齐，组间留 1 列背景分隔。例：若输入有 A/B 两组，则原逻辑会把 0 行误当作新色条继续包裹；
#      新逻辑应先切成 A 与 B 两个子问题，各自构框，再横向拼接为 [frame(A)] 0 [frame(B)]。
# b) 复杂度校准
#    - 组数限制为 2-4，既确保 many:many 明显可见，又避免输出尺寸失控。
#    - 每组色条数限制为 2-4，高度为 1-3，这样方框尺寸有变化，但不会大到挤压其他组。
#    - 输出布局固定为横向并排，避免再引入额外朝向分支，保证 D11 是主变化而不是新任务。
# c) 隐式假设审查（针对原 main 的关键前提）
#    - 原代码 `while r >= 0` 默认整图只有一个连续堆栈；变体中不成立，因为 0 行会切断组。
#    - 原代码 `col = g[r, 0]` 后直接向上合并，默认每一行都属于某个有效色条；变体中需先跳过背景行。
#    - 原代码把所有 run 统一塞进一个 `runs`，默认 many:1；变体必须先得到 `groups -> runs per group`。
#    - 原代码输出只维护一个 `cur`；变体需要 `frames[]`，最后再进行安全拼接。
# d) 放置合法性
#    - 输入端：每组仍是满宽纯色条，组间只有纯 0 行，因此不存在重叠或歧义。
#    - 输出端：各方框单独生成后写入背景画布，使用固定 1 列空隙并做底边对齐，不会越界或互相覆盖。
#    - 生成器同时约束输入总高和输出总宽/总高均不超过 30。
# e) 视觉可观察性
#    - 人类可直接看到：输入里有多个被 0 行隔开的“原题子实例”，输出里出现多个彼此分离的同心方框。
#    - 从 2-3 个样例即可归纳“先分组、再逐组应用原规则、最后横向并排”的流程。
# f) 可视化整洁度
#    - 只使用整行背景做分隔，不引入额外标记。
#    - 输出使用背景列隔开多个方框，防止外层颜色接触后视觉粘连。
# g) 颜色变化可推断性
#    - 颜色语义完全继承原题：颜色不重映射，只表示该层色条对应的方框层颜色。
#    - 变体只改变对象基数和组合关系，不增加额外的颜色编码负担。


def _parse_groups(grid: np.ndarray) -> list[np.ndarray]:
    """Split the input into non-background row blocks."""
    groups: list[np.ndarray] = []
    h = int(grid.shape[0])
    r = 0
    while r < h:
        while r < h and bool((grid[r] == BG).all()):
            r += 1
        if r >= h:
            break
        start = r
        while r < h and not bool((grid[r] == BG).all()):
            r += 1
        groups.append(grid[start:r])
    return groups


def _parse_runs(block: np.ndarray) -> list[tuple[int, int]]:
    """Parse one group into (color, height) runs from bottom to top."""
    h = int(block.shape[0])
    runs: list[tuple[int, int]] = []
    r = h - 1
    while r >= 0:
        col = int(block[r, 0])
        run_h = 1
        rr = r - 1
        while rr >= 0 and bool((block[rr] == col).all()):
            run_h += 1
            rr -= 1
        runs.append((col, run_h))
        r = rr
    return runs


def _build_frame(runs: list[tuple[int, int]]) -> np.ndarray:
    """Apply the original many:1 nesting rule to a single group."""
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
    """Place independently generated frames side by side with a background gap."""
    assert frames
    out_h = max(int(frame.shape[0]) for frame in frames)
    out_w = sum(int(frame.shape[1]) for frame in frames) + GAP * (len(frames) - 1)
    out = np.full((out_h, out_w), BG, dtype=int)
    c = 0
    for frame in frames:
        fh, fw = frame.shape
        out[out_h - fh :, c : c + fw] = frame
        c += fw + GAP
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Variant rule:
    - Input contains multiple full-width stripe stacks separated by pure background rows.
    - Each stack independently follows the original 45a5af55 nesting rule.
    - The resulting square frames are laid out left-to-right in top-group order, bottom aligned.
    """
    g = np.array(grid, dtype=int)
    groups = _parse_groups(g)
    assert groups
    frames = [_build_frame(_parse_runs(block)) for block in groups]
    return _pack_frames(frames)


def _sample_runs(n_runs: int) -> list[tuple[int, int]]:
    heights = [int(np.random.randint(1, 4)) for _ in range(n_runs)]
    colors: list[int] = []
    prev = None
    for _ in range(n_runs):
        c = int(np.random.randint(1, 10))
        while c == prev:
            c = int(np.random.randint(1, 10))
        colors.append(c)
        prev = c
    return list(zip(colors, heights))


def _render_group_input(runs: list[tuple[int, int]], width: int) -> np.ndarray:
    stripes = [np.full((height, width), color, dtype=int) for color, height in reversed(runs)]
    return np.vstack(stripes)


def _gen_attempt() -> np.ndarray | None:
    n_groups = int(np.random.randint(2, 5))
    width = int(np.random.randint(7, 16))
    sep_heights = [int(np.random.randint(1, 3)) for _ in range(n_groups - 1)]

    groups: list[list[tuple[int, int]]] = []
    frame_sizes: list[int] = []
    input_heights: list[int] = []

    for _ in range(n_groups):
        runs = _sample_runs(int(np.random.randint(2, 5)))
        frame_size = runs[0][1] + 2 * sum(height for _color, height in runs[1:])
        groups.append(runs)
        frame_sizes.append(frame_size)
        input_heights.append(sum(height for _color, height in runs))

    total_input_h = sum(input_heights) + sum(sep_heights)
    total_output_w = sum(frame_sizes) + GAP * (n_groups - 1)
    total_output_h = max(frame_sizes)
    if total_input_h > MAX_SIDE or width > MAX_SIDE:
        return None
    if total_output_w > MAX_SIDE or total_output_h > MAX_SIDE:
        return None

    # Keep the variant visually obvious: always produce at least two groups and
    # avoid a totally uniform set of frame sizes too often.
    if len(set(frame_sizes)) == 1 and np.random.rand() < 0.5:
        return None

    rendered_groups = [_render_group_input(runs, width) for runs in groups]
    parts: list[np.ndarray] = []
    for idx, block in enumerate(rendered_groups):
        parts.append(block)
        if idx != len(rendered_groups) - 1:
            parts.append(np.full((sep_heights[idx], width), BG, dtype=int))
    grid = np.vstack(parts)

    expected = _pack_frames([_build_frame(runs) for runs in groups])
    actual = main(grid)
    if not np.array_equal(actual, expected):
        raise RuntimeError("generator produced an invalid sample")
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 400 attempts")
