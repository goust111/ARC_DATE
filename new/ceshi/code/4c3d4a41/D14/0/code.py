# Variant of 4c3d4a41
# Dimension: D14 — 对接后灰色填充规则 (transform)
# Change: 杆对接后，彩色像素上方到凸起顶部之间无额外灰色填充；灰色帽保持原位... → 凸起顶部到彩色像素之间填充灰色

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg as _bg_color
from arc_utils import colors_present, get_bg, paste

# design_analysis:
#   concrete_behavior:
#     input_geometry:
#       - 画布、框架、横杆、4 个目标列的几何关系保持原题固定规格。
#       - 槽内每个目标列仍然是“顶部灰帽 + 底部单色块”的连续竖列。
#     transform_change:
#       - 横杆移入框架、按列对齐的规则不变。
#       - 原实现把颜色块压到可见空间的下端，保留顶部灰帽；如果可见空间高于原始列长，
#         会在灰帽和颜色块之间留下背景空隙。
#       - D14 要求把这段空隙改涂成灰色，使该列从顶部灰帽一直延续到颜色块，不再断成“灰 + 背景 + 颜色”。
#     traced_example:
#       - 以短列 L=3、输入颜色数 n=1、对应凸起高度 h=1 为例。
#       - 原逻辑输出列(行 1..6)可写成 [5, 5, 0, C, 5, 0]：
#         顶部 2 格灰帽保留，颜色块仍落在第 4 行，第 3 行留成背景。
#       - D14 变体改成 [5, 5, 5, C, 5, 0]：
#         唯一新增的是第 3 行被补成灰色。
#   complexity_calibration:
#     - 这类“补灰”只会出现在长度为 3 的短列且对应凸起高度为 1 时。
#     - 若生成器不主动制造这种组合，很多样例会与原行为完全相同，D14 不可观察。
#     - 左端短列虽然也是长度 3，但它对应的是横杆最左凸起；若高度降到 1，
#       该凸起不再高于 base，solver 的投影列检测会少一列，属于额外维度变化。
#     - 因此生成器保留左端原约束(h0 >= 2)，固定让右端短列(列 17)取 h=1，
#       用它稳定承载 D14 的新增补灰效果。
#   implicit_assumption_review:
#     - `grey` 仍可由“非背景中出现最多的颜色”可靠识别，因为结构色只有框架和横杆。
#     - 最大灰色连通块仍是框架、次大灰色连通块仍是横杆；D14 不改变对象数量或连通性。
#     - 原代码默认“清掉颜色后留下的背景空隙可以不管”；这正是 D14 失效点，
#       所以必须显式计算灰帽结束行与颜色重放起始行之间的区间并补灰。
#     - `src_proj_cols` 只统计“在 base 之上仍有灰像素”的列，所以最左凸起必须继续满足 h>=2；
#       generator 按此约束采样，避免 bar 对齐机制被 D14 之外的变化干扰。
#     - 目标列依旧满足：每列只有一种彩色、且从顶部向下连续出现灰帽后再出现颜色。
#       solver 通过扫描顶部连续灰帽来定位补灰起点；generator 保证这一前提成立。
#   placement_legality:
#     - 仍使用固定 8x20 几何，不新增对象，不改 bar/frame 的 bbox。
#     - 横杆输出位置仍是 cols 11..17、base row 5；所有补灰都发生在框架内列 11/13/15/17 上，
#       不会越界，也不会覆盖边框。
#   visual_observability:
#     - 至少一个短列会出现“原本是黑色背景、现在变成灰色”的 1 格差异。
#     - 该差异正好落在彩色块与顶部灰帽之间，肉眼可直接看到是“把空格补成支柱”。
#   visual_cleanliness:
#     - 补灰只发生在已有竖列内部，不会引入新颜色，也不会让对象交叠关系变复杂。
#   color_inference:
#     - 颜色集合与原题一致，D14 不引入任何颜色映射；颜色仍只表示 4 个独立目标列。


def _bbox(pts):
    return bbox_coords(pts if isinstance(pts, np.ndarray) else np.asarray(list(pts), dtype=int))


def _cc4(mask):
    return components_mask(mask, connectivity=4)


def _most_common(arr):
    vals, counts = np.unique(np.asarray(arr), return_counts=True)
    return int(vals[counts.argmax()])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Dock a grey toothed bar into a grey frame, align it to the colored columns, and rebuild each column with grey gap fill.

    Input structure:
    - Background plus a grey structural color forming two disconnected pieces: a large open frame on the right and a smaller horizontal bar with upward teeth on the left.
    - Inside the frame, the target columns contain a top grey cap followed by a stack of same-colored pixels marking the columns that the teeth must match.

    Rule:
    - Translate the bar into the frame so its base sits one cell above the frame floor and its tooth columns line up with the colored target columns.
    - Re-stack each column's colored pixels within the available space above the matched tooth, trimming overflow if needed.
    - Under the D14 variant, any background gap between the preserved top grey cap and the relocated colored pixels is turned grey, creating a continuous grey shaft before the color stack.

    Key insight:
    - The transform is still driven by matching tooth columns to colored columns; the only new behavior is that the column interior is rebuilt as cap, then added grey filler, then colors, instead of leaving a background hole in the middle.
    """
    g = np.asarray(grid, dtype=int)
    _, w = g.shape

    bg = _bg_color(g)
    other = g[g != bg]
    if other.size == 0:
        return g.copy()
    grey = _most_common(other)

    def comps_of_color(color: int):
        comps = [(len(pts), _bbox(pts), pts) for pts in _cc4(g == color)]
        comps.sort(key=lambda t: (-t[0], t[1]))
        return comps

    grey_comps = comps_of_color(grey)
    if len(grey_comps) < 2:
        return g.copy()

    (_, slot_bbox, _), (_, _, bar_pts) = grey_comps[0], grey_comps[1]
    sr0, sc0, sr1, sc1 = slot_bbox

    out = np.full_like(g, bg)
    out[sr0 : sr1 + 1, sc0 : sc1 + 1] = g[sr0 : sr1 + 1, sc0 : sc1 + 1]
    slot_region = out[sr0 : sr1 + 1, sc0 : sc1 + 1]
    slot_region[(slot_region != bg) & (slot_region != grey)] = bg

    bar_base_row = max(r for r, _ in bar_pts)
    slot_interior_bottom = sr1 - 1
    target_base_row = slot_interior_bottom - 1
    dy = target_base_row - bar_base_row

    col_min_row = {}
    for r, c in bar_pts:
        col_min_row[c] = min(col_min_row.get(c, r), r)
    src_proj_cols = sorted(c for c, rmin in col_min_row.items() if rmin < bar_base_row)

    dst_cols_set = set()
    for r, c in np.argwhere((g != bg) & (g != grey)):
        r = int(r)
        c = int(c)
        if sr0 <= r <= sr1 and sc0 <= c <= sc1:
            dst_cols_set.add(c)
    dst_cols = sorted(dst_cols_set)

    dx = None
    if src_proj_cols and len(src_proj_cols) == len(dst_cols):
        deltas = [d - s for s, d in zip(src_proj_cols, dst_cols)]
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

    placed = []
    for r, c in bar_pts:
        rr = r + dy
        cc = c + dx
        if sr0 <= rr <= sr1 and sc0 <= cc <= sc1:
            out[rr, cc] = grey
            placed.append((rr, cc))

    placed_by_col = {}
    for rr, cc in placed:
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

        rows = [r for r in range(sr0, sr1 + 1) if g[r, col] != bg and g[r, col] != grey]
        if not rows:
            continue
        colors = [int(g[r, col]) for r in rows]

        max_len = desired_bottom - interior_top + 1
        if max_len <= 0:
            continue
        if len(colors) > max_len:
            colors = colors[-max_len:]

        start_row = desired_bottom - len(colors) + 1
        start_row = max(start_row, interior_top)

        cap_end = interior_top - 1
        for r in range(interior_top, min(interior_bottom, desired_bottom) + 1):
            if g[r, col] == grey:
                cap_end = r
                continue
            break

        fill_start = cap_end + 1
        if fill_start < start_row:
            out[fill_start:start_row, col] = grey

        for i, colv in enumerate(colors):
            rr = start_row + i
            if interior_top <= rr <= interior_bottom:
                out[rr, col] = colv

    return out


BG = 0
GREY = 5
PALETTE = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)

H, W = 8, 20

FRAME_R0, FRAME_C0 = 0, 9
FRAME_H, FRAME_W = 8, 11
LEFT_WALL_OPEN_ROWS = (4, 5)

SLOT_COLS = [11, 13, 15, 17]
SLOT_LENS = [3, 4, 4, 3]
SLOT_TOP_ROW = 1
GAP_SLOT_IDX = 3

BAR_INPUT_C0 = 1
BAR_OUTPUT_C0 = 11
BAR_R0 = 2
BAR_H, BAR_W = 4, 7
BAR_LOCAL_PROJ_COLS = [0, 2, 4, 6]
BAR_BASE_ABS_ROW = BAR_R0 + BAR_H - 1

HEIGHT0_RANGE = (2, 5)
HEIGHT_REST_RANGE = (1, 5)
MAX_ATTEMPTS = 300


def _frame_patch() -> np.ndarray:
    patch = np.zeros((FRAME_H, FRAME_W), dtype=int)
    patch[[0, -1], :] = GREY
    patch[:, -1] = GREY
    patch[:, 0] = GREY
    patch[list(LEFT_WALL_OPEN_ROWS), 0] = BG
    return patch


def _bar_patch(heights: list[int]) -> np.ndarray:
    patch = np.zeros((BAR_H, BAR_W), dtype=int)
    patch[-1, :] = GREY
    for local_c, height in zip(BAR_LOCAL_PROJ_COLS, heights):
        patch[BAR_H - height : BAR_H, local_c] = GREY
    return patch


def _paint_input_slot_column(grid: np.ndarray, col: int, length: int, color: int, count: int) -> None:
    cap = length - count
    grid[SLOT_TOP_ROW : SLOT_TOP_ROW + cap, col] = GREY
    grid[SLOT_TOP_ROW + cap : SLOT_TOP_ROW + length, col] = color


def _paint_output_slot_column(
    grid: np.ndarray, col: int, length: int, color: int, count: int, height: int
) -> None:
    cap = length - count
    desired_bottom = BAR_BASE_ABS_ROW - height
    visible = min(count, desired_bottom - SLOT_TOP_ROW + 1)

    grid[SLOT_TOP_ROW : SLOT_TOP_ROW + cap, col] = GREY
    if visible <= 0:
        return

    start_row = desired_bottom - visible + 1
    fill_start = SLOT_TOP_ROW + cap
    if fill_start < start_row:
        grid[fill_start:start_row, col] = GREY
    grid[start_row : start_row + visible, col] = color


def _build_grid(
    heights: list[int], colors: list[int], colored_counts: list[int], bar_c0: int, output_mode: bool
) -> np.ndarray:
    grid = np.zeros((H, W), dtype=int)
    paste(grid, _frame_patch(), FRAME_R0, FRAME_C0)
    paste(grid, _bar_patch(heights), BAR_R0, bar_c0)
    for col, length, color, count, height in zip(SLOT_COLS, SLOT_LENS, colors, colored_counts, heights):
        if output_mode:
            _paint_output_slot_column(grid, col, length, color, count, height)
        else:
            _paint_input_slot_column(grid, col, length, color, count)
    return grid


def _variant_gap_cells(heights: list[int]) -> int:
    length = SLOT_LENS[GAP_SLOT_IDX]
    capacity = BAR_BASE_ABS_ROW - heights[GAP_SLOT_IDX]
    return max(0, capacity - length)


def _sample_params(rng: np.random.Generator) -> tuple[list[int], list[int], list[int]]:
    colors = [int(v) for v in rng.choice(PALETTE, size=len(SLOT_COLS), replace=False)]
    colored_counts = [int(rng.integers(1, length + 1)) for length in SLOT_LENS]

    heights = [int(rng.integers(*HEIGHT0_RANGE))]
    heights.extend(int(rng.integers(*HEIGHT_REST_RANGE)) for _ in range(len(SLOT_COLS) - 1))
    heights[GAP_SLOT_IDX] = 1
    return heights, colors, colored_counts


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    heights, colors, colored_counts = _sample_params(rng)
    if _variant_gap_cells(heights) <= 0:
        return None

    grid = _build_grid(heights, colors, colored_counts, BAR_INPUT_C0, output_mode=False)
    expected = _build_grid(heights, colors, colored_counts, BAR_OUTPUT_C0, output_mode=True)

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
