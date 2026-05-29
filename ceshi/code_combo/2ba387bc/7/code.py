# Combo variant of 2ba387bc (index 7)
# combo_spec: D2/0+D14/0
# Name: 按列排序+反向堆叠
#
# Design analysis
# a) 具体化与行为追踪
# - 输入对象保持原题不变：背景上散落若干个 4x4 物体，分成两类。
#   1) frame: 单色边框 + 中间 2x2 为背景
#   2) solid: 单色实心 4x4
# - 原题流程：
#   先按 frame/solid 分类；每组按 `(row, col)` 排序；frame 从左列顶端向下堆叠，
#   solid 从右列顶端向下堆叠。
# - 融合后流程：
#   仍先按 frame/solid 分类；但每组改按 `(col, row)` 排序；frame 仍从左列顶端向下堆叠，
#   solid 改为在右列从底部向上堆叠。
# - 具体小例子：
#   frame A 在 `(r=2,c=11)`，frame B 在 `(r=10,c=4)`。
#   原题按 row 排：A 在上、B 在下；融合后按 col 排：B 在上、A 在下。
#   solid C 在 `(r=3,c=5)`，solid D 在 `(r=8,c=13)`。
#   它们按 col 排后是 C 再 D，但输出时 C 要落在右列最底部 4 行，D 在它上方 4 行。
#   也就是说，frame 组和 solid 组同时共享“按列排序”，但两组的堆叠方向不同。
# b) 复杂度校准
# - 若生成器允许“按 row 排”和“按 col 排”给出同一顺序，则 D2 会退化，不易肉眼看出。
# - 若 solid 只有 1 个，则右列自底向上和自顶向下完全等价，D14 会退化。
# - 因此生成器必须至少保证：
#   1) `col-sort` 与 `row-sort` 的输出不同；
#   2) `solid` 组在 `col-sort` 下的 top-down 与 bottom-up 输出不同。
# - 这两个约束让少量 few-shot 样例里就能看出“排序依据变了”且“右列方向反了”。
# c) 隐式假设审查
# - `find_components(..., monochromatic=False)` 假设对象彼此不接触；生成器继续用 `padding=1`
#   保证 4 邻接不粘连。
# - `bbox_h == bbox_w == 4` 假设所有目标对象都是 4x4；融合不改这一点。
# - `len(border_colors) == 1 and interior == bg` 假设 frame 仅靠空心结构识别；融合不改形状。
# - 原 `sort(key=(row,col))` 是旧规则的核心假设；这里必须统一改成 `sort(key=(col,row))`。
# - 原右列 `paste(..., i*4, 4)` 假设 solid 也从顶部向下放；融合后这行必须改成按
#   `out_h - 4*(i+1)` 计算目标行。
# - 多维组合没有新冲突：排序发生在抽取后的对象列表上，堆叠方向发生在渲染阶段，两个改动
#   作用于不同步骤，但会共同影响 solid 在输出中的最终上下顺序。
# d) 放置合法性
# - 输入生成继续使用 `random_free_location(..., border=1, padding=1, padding_connectivity=4)`，
#   保证对象不重叠、不越界、不接触。
# - 每个对象使用唯一非背景色，避免不同对象因相同颜色接触后被误并成一个连通块。
# - 接受样本前执行 `main(g)` 与期望输出一致性检查，避免生成器产出 solver 无法处理的输入。
# e) 视觉可观察性
# - 排序只依赖可见的空间位置，不依赖颜色数值等人类不可见因素。
# - D2 通过“更靠左的块在输出中更靠上”体现；D14 通过“右列首个 solid 落在底部”体现。
# - 当样例里至少有两个 solid，且至少一组对象的 row/col 次序不一致时，人类能从 2-4 个样例中
#   推出融合规则。
# f) 可视化整洁度
# - 输入仍是稀疏散落的 4x4 方块，没有额外噪声或覆盖结构。
# - 输出仍保持 8 列宽，只是右列从底部开始填充；空白仅表现为背景填充，不会制造歧义。
# g) 变化的可推断性
# - 颜色完全保留原物体颜色，不承担控制语义；真正决定输出的是形状类别和输入中的水平位置。
# - 颜色数量仍限制在 ARC 常见的 1..9，既能区分对象，也不会引入额外映射负担。
# 9) 新隐式假设审查
# - 新 solver 假设 right-column reverse stacking 只有在 solid 至少 2 个时才可观察；
#   生成器用 `reverse_signal` 强制这一点。
# - 新 solver 仍假设每个被识别对象都能完整裁成 4x4 patch；生成器构造严格满足。

from __future__ import annotations

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, paste, random_free_location


Obj = tuple[int, int, np.ndarray]


def _key_by_row(item: Obj) -> tuple[int, int]:
    return int(item[0]), int(item[1])


def _key_by_col(item: Obj) -> tuple[int, int]:
    return int(item[1]), int(item[0])


def _collect_objects(g: np.ndarray, bg: int) -> tuple[list[Obj], list[Obj]]:
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    frames: list[Obj] = []
    solids: list[Obj] = []

    for comp in comps:
        b = comp["bbox"]
        if bbox_h(b) != 4 or bbox_w(b) != 4:
            continue
        r0, c0, _r1, _c1 = b
        sub = crop_bbox(g, b)

        border = np.concatenate([sub[0, :], sub[-1, :], sub[1:-1, 0], sub[1:-1, -1]])
        border_colors = set(map(int, border.flatten()))
        is_frame = len(border_colors) == 1 and bool((sub[1:3, 1:3] == int(bg)).all())

        if is_frame:
            frames.append((int(r0), int(c0), sub))
        else:
            solids.append((int(r0), int(c0), sub))

    return frames, solids


def _render_output(
    frames: list[Obj],
    solids: list[Obj],
    bg: int,
    *,
    sort_key,
    reverse_solids: bool,
) -> np.ndarray:
    frames_sorted = sorted(frames, key=sort_key)
    solids_sorted = sorted(solids, key=sort_key)

    out_h = 4 * max(int(len(frames_sorted)), int(len(solids_sorted)))
    out = np.full((out_h, 8), int(bg), dtype=int)

    for i, (_r0, _c0, sub) in enumerate(frames_sorted):
        paste(out, sub, int(i * 4), 0)

    for i, (_r0, _c0, sub) in enumerate(solids_sorted):
        dst_r = int(out_h - 4 * (i + 1)) if reverse_solids else int(i * 4)
        paste(out, sub, dst_r, 4)

    return out


def _signals(frames: list[Obj], solids: list[Obj], bg: int) -> tuple[int, int, int]:
    """
    Return three observability signals:
    - order_signal: row-sort vs col-sort differs.
    - reverse_signal: solid top-down vs bottom-up differs under col-sort.
    - combo_signal: original task output vs combo output differs.
    """
    row_top = _render_output(frames, solids, bg, sort_key=_key_by_row, reverse_solids=False)
    col_top = _render_output(frames, solids, bg, sort_key=_key_by_col, reverse_solids=False)
    col_rev = _render_output(frames, solids, bg, sort_key=_key_by_col, reverse_solids=True)
    order_signal = int(not np.array_equal(row_top, col_top))
    reverse_signal = int(not np.array_equal(col_top, col_rev))
    combo_signal = int(not np.array_equal(row_top, col_rev))
    return order_signal, reverse_signal, combo_signal


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combo rule:
    - Detect 4x4 frames and 4x4 solids.
    - Sort each group by input column (left-to-right), breaking ties by row.
    - Stack frames in the left column from top to bottom.
    - Stack solids in the right column from bottom to top.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    frames, solids = _collect_objects(g, bg)
    return _render_output(frames, solids, bg, sort_key=_key_by_col, reverse_solids=True)


def _make_frame(col: int, bg: int = 0) -> np.ndarray:
    a = np.full((4, 4), col, dtype=int)
    a[1:3, 1:3] = bg
    return a


def _make_solid(col: int) -> np.ndarray:
    return np.full((4, 4), col, dtype=int)


def _gen_attempt() -> np.ndarray | None:
    bg = 0
    h = int(np.random.randint(20, 31))
    w = int(np.random.randint(20, 31))
    g = np.full((h, w), bg, dtype=int)

    n_frames = int(np.random.randint(1, 5))
    n_solids = int(np.random.randint(2, 5))
    colors = np.random.permutation(np.arange(1, 10))[: n_frames + n_solids]
    frame_colors = [int(x) for x in colors[:n_frames]]
    solid_colors = [int(x) for x in colors[n_frames:]]

    frames: list[Obj] = []
    solids: list[Obj] = []

    try:
        for col in frame_colors:
            sub = _make_frame(col, bg)
            r0, c0 = random_free_location(
                g,
                sub,
                transparent=bg,
                border=1,
                padding=1,
                padding_connectivity=4,
            )
            paste(g, sub, r0, c0, transparent=bg)
            frames.append((int(r0), int(c0), sub))

        for col in solid_colors:
            sub = _make_solid(col)
            r0, c0 = random_free_location(
                g,
                sub,
                transparent=bg,
                border=1,
                padding=1,
                padding_connectivity=4,
            )
            paste(g, sub, r0, c0, transparent=bg)
            solids.append((int(r0), int(c0), sub))

        order_signal, reverse_signal, combo_signal = _signals(frames, solids, bg)
        if order_signal == 0 or reverse_signal == 0 or combo_signal == 0:
            return None

        expected = _render_output(frames, solids, bg, sort_key=_key_by_col, reverse_solids=True)
        if not np.array_equal(main(g), expected):
            return None

        detected_frames, detected_solids = _collect_objects(g, bg)
        if len(detected_frames) != len(frames) or len(detected_solids) != len(solids):
            return None

        return g
    except Exception:
        return None


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 400 attempts")
