# Variant of 2ba387bc
# Dimension: D2 — 排序规则——组内排列顺序 (match_cond)
# Change: 每组（框架组和实心组）内部按输入中的行位置(row)从上到下排序，行相同时按列位置(col)排序
# ... → 按列位置排序（左到右）
#
# Design analysis
# - 原任务对象不变：背景上散落多个 4x4 物体，分为两类。
#   1) frame: 单一颜色边框 + 2x2 背景色内孔
#   2) solid: 单一颜色的 4x4 实心块
#   输出仍然是 8 列宽，左列堆叠 frame，右列堆叠 solid。
# - 这次只改 D2：组内堆叠顺序从“按输入中的 row 再 col”改为“按输入中的 col 再 row”。
#   也就是 frame 组、solid 组都按更靠左的对象先放到输出上方。
# - 具体行为追踪：
#   假设 frame A 在 (r=2,c=12)，frame B 在 (r=10,c=4)。
#   原规则按 row 排：A 在上，B 在下。
#   新规则按 col 排：B 在上，A 在下。
#   物体形状、颜色、左右分列方式都不变，只有同组内部的纵向顺序改变。
# - 复杂度校准：
#   如果生成器经常让“按 row 排”和“按 col 排”得到同一顺序，那么变体会退化，肉眼看不出变化。
#   因此生成器必须拒绝这类样本，至少保证有一组的列序和行序不同。
# - 对 main() 的隐式假设逐项审查：
#   1) `find_components(..., monochromatic=False)` 依赖对象之间不接触；变体不改这一点，生成器继续保证 1 格间隔。
#   2) `bbox_h == bbox_w == 4` 依赖所有目标对象都是 4x4；变体不改尺寸。
#   3) `len(bcols) == 1 and interior == bg` 依赖 frame 的边框单色且中心挖空；变体不改分类特征。
#   4) 原 `sort(key=(row,col))` 依赖旧排序规则；这里必须改成 `sort(key=(col,row))`。
#   5) 输出宽 8、左右两列固定，依赖“按类型分左右列”的旧规则；变体不改。
# - 放置合法性：
#   生成器继续使用 `random_free_location(..., border=1, padding=1)`，确保对象不重叠、不越界、不同对象不会 4 邻接粘连。
#   这让 `find_components` 的检测假设继续成立。
# - 视觉可观察性：
#   排序不能依赖颜色值等不可见因素，所以生成器用“输入中的水平位置”制造差异。
#   只要至少一组出现“更上方的块反而更靠右”，解题者就能从输出看出排序看的是左右而不是上下。
# - 可视化整洁度：
#   保留原任务的稀疏散落布局，不叠加额外标记；对象形状本身已经足够区分。
# - 颜色可推断性：
#   颜色只用来区分不同对象，不参与排序规则；仍控制在 ARC 常用的 1..9 非背景色内。
# - source_task: 本变体没有提供 source_task，按要求跳过。

from __future__ import annotations

import numpy as np
from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, paste, random_free_location


def _key_by_row(item: tuple[int, int, np.ndarray]) -> tuple[int, int]:
    return int(item[0]), int(item[1])


def _key_by_col(item: tuple[int, int, np.ndarray]) -> tuple[int, int]:
    return int(item[1]), int(item[0])


def _collect_objects(g: np.ndarray, bg: int) -> tuple[list[tuple[int, int, np.ndarray]], list[tuple[int, int, np.ndarray]]]:
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    frames: list[tuple[int, int, np.ndarray]] = []
    solids: list[tuple[int, int, np.ndarray]] = []

    for comp in comps:
        b = comp["bbox"]
        if bbox_h(b) != 4 or bbox_w(b) != 4:
            continue
        r0, c0, _r1, _c1 = b
        sub = crop_bbox(g, b)

        border = np.concatenate([sub[0, :], sub[-1, :], sub[1:-1, 0], sub[1:-1, -1]])
        bcols = set(map(int, border.flatten()))

        if len(bcols) == 1 and bool((sub[1:3, 1:3] == int(bg)).all()):
            frames.append((int(r0), int(c0), sub))
        else:
            solids.append((int(r0), int(c0), sub))

    return frames, solids


def _render_output(
    frames: list[tuple[int, int, np.ndarray]],
    solids: list[tuple[int, int, np.ndarray]],
    bg: int,
    *,
    sort_key,
) -> np.ndarray:
    frames_sorted = sorted(frames, key=sort_key)
    solids_sorted = sorted(solids, key=sort_key)

    out_h = 4 * max(int(len(frames_sorted)), int(len(solids_sorted)))
    out = np.full((out_h, 8), int(bg), dtype=int)

    for i, (_r0, _c0, sub) in enumerate(frames_sorted):
        paste(out, sub, int(i * 4), 0)
    for i, (_r0, _c0, sub) in enumerate(solids_sorted):
        paste(out, sub, int(i * 4), 4)

    return out


def _variant_signal(
    frames: list[tuple[int, int, np.ndarray]], solids: list[tuple[int, int, np.ndarray]], bg: int
) -> int:
    """
    Quantify whether D2 is active on a sample.
    Returns 1 iff the old row-based output differs from the new col-based output.
    """
    row_out = _render_output(frames, solids, bg, sort_key=_key_by_row)
    col_out = _render_output(frames, solids, bg, sort_key=_key_by_col)
    return int(not np.array_equal(row_out, col_out))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort scattered 4x4 squares into two columns by type: frames left, solids right.

    Variant rule:
    - Classify each 4x4 object as frame or solid by its interior.
    - Sort each group left-to-right by input column.
    - Break column ties by row for determinism.
    - Stack frames in left column (cols 0-3), solids in right (cols 4-7).
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    frames, solids = _collect_objects(g, bg)
    return _render_output(frames, solids, bg, sort_key=_key_by_col)


def _make_frame(col: int, bg: int = 0) -> np.ndarray:
    a = np.full((4, 4), col, dtype=int)
    a[1:3, 1:3] = bg
    return a


def _make_solid(col: int) -> np.ndarray:
    return np.full((4, 4), col, dtype=int)


def _gen_attempt() -> np.ndarray | None:
    bg = 0
    h = int(np.random.randint(18, 30))
    w = int(np.random.randint(18, 30))
    g = np.full((h, w), bg, dtype=int)

    n_frames = int(np.random.randint(1, 5))
    n_solids = int(np.random.randint(1, 5))
    colors = np.random.permutation(np.arange(1, 10))[: n_frames + n_solids]
    frame_colors = [int(x) for x in colors[:n_frames]]
    solid_colors = [int(x) for x in colors[n_frames:]]

    frames: list[tuple[int, int, np.ndarray]] = []
    solids: list[tuple[int, int, np.ndarray]] = []

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

        if _variant_signal(frames, solids, bg) == 0:
            return None

        expected = _render_output(frames, solids, bg, sort_key=_key_by_col)
        if not np.array_equal(main(g), expected):
            return None
        return g
    except Exception:
        return None


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 300 attempts")
