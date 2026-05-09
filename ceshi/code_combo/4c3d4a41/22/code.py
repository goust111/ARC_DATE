# Combo variant of 4c3d4a41 (index 22)
# combo_spec: D3/2+D11/0
# Name: 开口位置可变+右墙开口
#
# Design analysis
# ===============
# fused_variant:
#   summary:
#     - 画布仍是固定 8x20，但整体左右镜像：框架移到左侧，开口改到右墙，横杆放在框架右侧。
#     - 融合后的真正新规则不是“仅仅镜像”，而是右墙 2 格开口的行位置可变，并且这个开口位置直接决定横杆插入后的垂直落点。
#     - 横杆仍需按 4 个凸起列与 4 条彩色列做列对齐；只是对齐完成后，它不再总是落在原题那条固定底座行上。
#   concrete_behavior_trace:
#     - 原题：左墙开口固定在 rows 4-5，杆从左侧进入，底座固定落在 row 5，颜色再堆到各凸起顶部上方。
#     - 本组合：若右墙开口在 rows 3-4，则杆从右侧进入后底座落在 row 4；若开口在 rows 5-6，则底座落在 row 6。
#     - 例：某列原本有 3 个彩色像素、对应凸起高度为 2。原题固定底座 row 5 时可见数是 3；本组合若开口把底座抬到 row 4，可见数变成 2；若压到 row 6，可见数变成 4 上限里的 3。
#     - 差异因此来自两层协同：`D11/0` 让解题者必须适应左右反转，`D3/2` 再让他们从右墙开口的高低读出新的垂直定位，而不是继续假设“总是贴近底部”。
#   complexity_calibration:
#     - 开口只允许 3 个档位 `(3,4)`, `(4,5)`, `(5,6)`，对应底座行 `4,5,6`。档位少，肉眼可归纳；又足够明显，不会退化成固定位置。
#     - 仍保留 4 根目标列、4 个凸起、固定奇数间距，不引入新的匹配自由度，避免同时变化过多导致任务失焦。
#     - 槽列颜色仍是一列一种色，不叠加额外颜色映射，保证人类能把注意力集中在“开口高度 -> 杆的垂直落点”上。
#   implicit_assumptions_review:
#     - 原 solver 的 `target_base_row = sr1 - 2` 依赖“杆总是落在靠近底部的固定高度”，融合后失效，必须改成从框架右墙缺口恢复目标底座行。
#     - 原 solver 默认框架在右、杆在左；融合后虽然“最大灰块=框架、第二大灰块=杆”仍成立，但 `dx` 将从正值变成负值，不能硬编码方向。
#     - 原 solver 从框架 bbox 拷贝输出区域这一点仍成立，因为输出依旧只保留框架区域；generate_input 会继续保证杆与框架在输入中分离、不接触。
#     - 新 solver 额外假设：右墙缺口恰好是 2 个连续背景格，且底座行等于缺口的较大行号。generator 用固定模板严格保证这点。
#   placement_legality:
#     - 框架固定在 cols 0..10，杆输入固定在 cols 12..18，中间留 1 列间隔，因此输入里两个灰组件绝不 4 邻接。
#     - 杆输出固定平移到 cols 2..8，完整落在框架内部 cols 1..9 的中间区域，不会碰到左右墙。
#     - 3 个候选底座行都保证 4 高横杆整体留在 rows 1..6 内；即便某个凸起高度为 4，顶部也只到 row 1，不越界。
#   visual_observability:
#     - 单看输入就能看到两个稳定线索：右墙开口在不同高度、杆位于右侧等待从右进入。
#     - 对比输出时，杆整体上下位置会随开口变化，而彩色列保留下来的高度也随之变化；这比“只是镜像”更容易从 2-3 个样例中读出。
#     - 变化只依赖可见几何位置，不依赖颜色编号、面积排序或其他人类不可见因素。
#   cleanliness:
#     - 画面仍只有三类元素：灰框架、灰横杆、框架内 4 条彩色列；没有额外标记层。
#     - 开口、杆、彩色列分属不同区域，插入后颜色只出现在 4 根目标列里，避免视觉混杂。
#   generator_guarantees_for_new_main:
#     - 每个样例都随机选 3 档开口之一，因而 `dy` 不是常量。
#     - 右墙缺口总在框架 bbox 内唯一出现，solver 的缺口检测不会与背景区域混淆。
#     - `main(input)` 必须等于按同一缺口档位构造的 expected；同时统计中会验证不同档位确实都出现，避免退化成固定底座行。

from __future__ import annotations

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg as _bg_color
from arc_utils import colors_present, get_bg, paste


def _bbox(pts):
    arr = pts if isinstance(pts, np.ndarray) else np.asarray(list(pts), dtype=int)
    return bbox_coords(arr)


def _cc4(mask):
    return components_mask(mask, connectivity=4)


def _most_common(arr):
    vals, counts = np.unique(np.asarray(arr), return_counts=True)
    return int(vals[counts.argmax()])


def _sorted_components(mask: np.ndarray):
    comps = [(len(pts), _bbox(pts), pts) for pts in _cc4(mask)]
    comps.sort(key=lambda item: (-item[0], item[1]))
    return comps


def _opening_rows(g: np.ndarray, slot_bbox: tuple[int, int, int, int], bg: int) -> list[int]:
    sr0, _sc0, sr1, sc1 = slot_bbox
    return [r for r in range(sr0, sr1 + 1) if int(g[r, sc1]) == bg]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Dock the grey bar into a left-side frame by matching the colored columns, with the right-wall gap deciding the final height.

    Input structure:
    - Grey forms two disconnected objects: a large frame on the left whose right wall has a 2-cell opening, and a smaller bar on the right with four upward projections.
    - Inside the frame are four colored columns, each occupying one target column and separated by blank columns.
    - The wall opening can appear at different row pairs, and that vertical offset is part of the puzzle data.

    Rule:
    - Slide the bar leftward into the frame so its four projection columns align with the four colored columns.
    - Do not place the bar at a fixed floor height; instead, use the right-wall opening to choose the bar's final base row.
    - After the bar is inserted, restack each column's original colored pixels immediately above the bar in that column, clipped to the available space.

    Key insight: The colored columns still determine horizontal alignment, but the variable right-wall opening is now the cue for vertical placement, so both axes must be inferred from the input geometry.
    """
    g = np.asarray(grid, dtype=int)
    _h, w = g.shape

    bg = _bg_color(g)
    other = g[g != bg]
    if other.size == 0:
        return g.copy()
    grey = _most_common(other)

    grey_comps = _sorted_components(g == grey)
    if len(grey_comps) < 2:
        return g.copy()

    (_, slot_bbox, _slot_pts), (_, _bar_bbox, bar_pts) = grey_comps[0], grey_comps[1]
    sr0, sc0, sr1, sc1 = slot_bbox

    out = np.full_like(g, bg)
    out[sr0 : sr1 + 1, sc0 : sc1 + 1] = g[sr0 : sr1 + 1, sc0 : sc1 + 1]
    slot_region = out[sr0 : sr1 + 1, sc0 : sc1 + 1]
    slot_region[(slot_region != bg) & (slot_region != grey)] = bg

    open_rows = _opening_rows(g, slot_bbox, bg)
    if len(open_rows) != 2 or open_rows[1] != open_rows[0] + 1:
        return g.copy()

    bar_base_row = max(int(r) for r, _ in bar_pts)
    target_base_row = int(max(open_rows))
    dy = target_base_row - bar_base_row

    col_min_row: dict[int, int] = {}
    for r, c in bar_pts:
        r = int(r)
        c = int(c)
        col_min_row[c] = min(col_min_row.get(c, r), r)
    src_proj_cols = sorted(c for c, rmin in col_min_row.items() if rmin < bar_base_row)

    dst_cols = []
    for col in range(sc0, sc1 + 1):
        has_color = any(int(g[r, col]) != bg and int(g[r, col]) != grey for r in range(sr0, sr1 + 1))
        if has_color:
            dst_cols.append(col)

    dx = None
    if src_proj_cols and len(src_proj_cols) == len(dst_cols):
        deltas = [dst - src for src, dst in zip(src_proj_cols, dst_cols)]
        if len(set(deltas)) == 1:
            dx = deltas[0]
    if dx is None:
        best = (-1, 0)
        dst_set = set(dst_cols)
        for cand_dx in range(-w, w + 1):
            score = sum((c + cand_dx) in dst_set for c in src_proj_cols)
            if score > best[0]:
                best = (score, cand_dx)
        dx = best[1]

    placed_by_col: dict[int, list[int]] = {}
    for r, c in bar_pts:
        rr = int(r) + dy
        cc = int(c) + dx
        if sr0 <= rr <= sr1 and sc0 <= cc <= sc1:
            out[rr, cc] = grey
            placed_by_col.setdefault(cc, []).append(rr)

    interior_top = sr0 + 1
    interior_bottom = sr1 - 1

    for col in dst_cols:
        if col not in placed_by_col:
            continue
        bar_top = min(placed_by_col[col])
        desired_bottom = bar_top - 1
        if desired_bottom < interior_top:
            continue

        rows = [r for r in range(sr0, sr1 + 1) if int(g[r, col]) != bg and int(g[r, col]) != grey]
        if not rows:
            continue
        colors = [int(g[r, col]) for r in rows]

        max_len = desired_bottom - interior_top + 1
        if max_len <= 0:
            continue
        if len(colors) > max_len:
            colors = colors[-max_len:]

        start_row = max(interior_top, desired_bottom - len(colors) + 1)
        for i, color in enumerate(colors):
            rr = start_row + i
            if interior_top <= rr <= interior_bottom:
                out[rr, col] = color

    return out


BG = 0
GREY = 5
PALETTE = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)

H, W = 8, 20

FRAME_R0, FRAME_C0 = 0, 0
FRAME_H, FRAME_W = 8, 11
OPEN_ROW_CHOICES = [(3, 4), (4, 5), (5, 6)]

SLOT_COLS = [2, 4, 6, 8]
SLOT_LENS = [3, 4, 4, 3]
SLOT_TOP_ROW = 1

BAR_INPUT_R0 = 2
BAR_INPUT_C0 = 12
BAR_OUTPUT_C0 = 2
BAR_H, BAR_W = 4, 7
BAR_LOCAL_PROJ_COLS = [0, 2, 4, 6]

HEIGHT0_RANGE = (2, 5)
HEIGHT_REST_RANGE = (1, 5)
MAX_ATTEMPTS = 400


def _frame_patch(open_rows: tuple[int, int]) -> np.ndarray:
    patch = np.zeros((FRAME_H, FRAME_W), dtype=int)
    patch[[0, -1], :] = GREY
    patch[:, 0] = GREY
    patch[:, -1] = GREY
    patch[list(open_rows), -1] = BG
    return patch


def _bar_patch(heights: list[int]) -> np.ndarray:
    patch = np.zeros((BAR_H, BAR_W), dtype=int)
    patch[-1, :] = GREY
    for local_c, height in zip(BAR_LOCAL_PROJ_COLS, heights):
        patch[BAR_H - height : BAR_H, local_c] = GREY
    return patch


def _paint_slot_caps(grid: np.ndarray, col: int, length: int, count: int) -> None:
    cap = int(length - count)
    if cap > 0:
        grid[SLOT_TOP_ROW : SLOT_TOP_ROW + cap, col] = GREY


def _paint_slot_input_column(grid: np.ndarray, col: int, length: int, color: int, count: int) -> None:
    _paint_slot_caps(grid, col, length, count)
    grid[SLOT_TOP_ROW + length - count : SLOT_TOP_ROW + length, col] = color


def _paint_slot_output_column(
    grid: np.ndarray,
    col: int,
    color: int,
    count: int,
    height: int,
    target_base_row: int,
) -> None:
    desired_bottom = int(target_base_row - height)
    visible = min(int(count), max(0, desired_bottom - SLOT_TOP_ROW + 1))
    if visible <= 0:
        return
    start_row = desired_bottom - visible + 1
    grid[start_row : desired_bottom + 1, col] = color


def _output_bar_r0(open_rows: tuple[int, int]) -> int:
    return int(max(open_rows) - (BAR_H - 1))


def _sample_params(rng: np.random.Generator) -> tuple[list[int], list[int], list[int], tuple[int, int]]:
    colors = [int(v) for v in rng.choice(PALETTE, size=len(SLOT_COLS), replace=False)]
    colored_counts = [int(rng.integers(1, length + 1)) for length in SLOT_LENS]
    heights = [int(rng.integers(*HEIGHT0_RANGE))]
    heights.extend(int(rng.integers(*HEIGHT_REST_RANGE)) for _ in range(len(SLOT_COLS) - 1))
    open_rows = OPEN_ROW_CHOICES[int(rng.integers(0, len(OPEN_ROW_CHOICES)))]
    return heights, colors, colored_counts, open_rows


def _build_input(
    heights: list[int],
    colors: list[int],
    colored_counts: list[int],
    open_rows: tuple[int, int],
) -> np.ndarray:
    grid = np.zeros((H, W), dtype=int)
    paste(grid, _frame_patch(open_rows), FRAME_R0, FRAME_C0)
    paste(grid, _bar_patch(heights), BAR_INPUT_R0, BAR_INPUT_C0)
    for col, length, color, count in zip(SLOT_COLS, SLOT_LENS, colors, colored_counts):
        _paint_slot_input_column(grid, col, length, color, count)
    return grid


def _build_expected(
    heights: list[int],
    colors: list[int],
    colored_counts: list[int],
    open_rows: tuple[int, int],
) -> np.ndarray:
    grid = np.zeros((H, W), dtype=int)
    paste(grid, _frame_patch(open_rows), FRAME_R0, FRAME_C0)
    paste(grid, _bar_patch(heights), _output_bar_r0(open_rows), BAR_OUTPUT_C0)
    target_base_row = int(max(open_rows))
    for col, length, color, count, height in zip(SLOT_COLS, SLOT_LENS, colors, colored_counts, heights):
        _paint_slot_caps(grid, col, length, count)
        _paint_slot_output_column(grid, col, color, count, height, target_base_row)
    return grid


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    heights, colors, colored_counts, open_rows = _sample_params(rng)
    grid = _build_input(heights, colors, colored_counts, open_rows)
    expected = _build_expected(heights, colors, colored_counts, open_rows)

    if get_bg(grid) != BG:
        return None
    if sorted(colors_present(grid, exclude=[BG, GREY])) != sorted(colors):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {MAX_ATTEMPTS} attempts")
