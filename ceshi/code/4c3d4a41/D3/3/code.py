# Variant of 4c3d4a41
# Dimension: D3 — 目标对象：框架结构 (dst)
# Change: 右侧固定位置的灰色矩形框架（8x11），左墙有2格开口（行4-5），内部有4条彩色列（金字塔形长度3,4,4,3）... → 框架带内部隔板

import numpy as np

from arc_utils import bbox_coords, colors_present, components_mask, get_bg as _bg_color
from arc_utils import get_bg, paste

# Design analysis
# ===============
# spec_variant:
#   grid: 8 x 20, bg=0, grey=5
#   preserved_from_original:
#     - 右侧仍是固定 8x11 框架，左墙仍在行 4-5 有开口
#     - 左侧仍有一根 7 格底边、4 个奇数列凸起的灰色横杆
#     - 横杆仍通过水平平移与 4 条彩色目标列对齐，然后移入框架
#   dst_change_core:
#     - 框架内部新增一条灰色水平隔板，位于绝对行 3
#     - 隔板把框架内部切成上下两个腔体
#     - 4 个目标列仍在绝对列 11,13,15,17
#     - 外侧两列 11 和 17 在隔板处有孔，因此它们的目标腔体是上腔 rows 1..3
#     - 内侧两列 13 和 15 没有隔板孔，因此它们的目标腔体是下腔 rows 4..6
#   bar_change_needed_for_visibility:
#     - 横杆整体下移一格，输入时位于 rows 3..6，底边在 row 6
#     - 输出时底边贴住框架内地板 row 6
#     - 为了让“进入上腔/停在下腔”在视觉上明确：
#       - 外侧两根凸起高度固定为 4，它们穿过隔板孔，顶部落在 row 3
#       - 内侧两根凸起高度在 {1,2} 中采样，只停在下腔
#   column_encoding:
#     - 每个目标列只在自己的腔体内编码，长度都固定为 3
#     - 输入里仍沿用“上部灰帽 + 下部单色块”的编码方式
#     - 上腔列使用 rows 1..3；下腔列使用 rows 4..6
#   output_rule:
#     - 清掉原横杆左侧位置，只保留右框 bbox 内内容
#     - 把横杆平移到框架内，对齐 4 个目标列
#     - 对每列，在对应腔体内取原单色块，按原顺序保留颜色；
#       只把能够位于该凸起顶部之上的那一段留在输出中，多余部分消失
#
# behavior_trace:
#   - 原题：所有颜色都从统一的 interior_top=row1 向下堆，solver 只需知道每列 bar_top
#   - 变体：列 11/17 只能在上腔 rows 1..3 活动，列 13/15 只能在下腔 rows 4..6 活动
#   - 例子：
#       col 11 若输入为 [grey, color, color] 且对应凸起高 4，
#       输出时 row 3 被凸起占据，只剩 rows 1..2 可见，所以该列颜色留在 rows 1..2
#       col 13 若输入为 [grey, color, color] 且对应凸起高 2，
#       下腔中 row 5..6 被凸起占据，只剩 row 4 可见，所以只保留最底部那 1 格颜色
#   - 差异：
#       原 solver 的 “所有列共享同一个 interior_top / interior_bottom” 在这里失效，
#       因为是否能放色取决于列属于上腔还是下腔
#
# complexity_calibration:
#   - 不引入额外对象数量；仍是 1 个框架 + 1 个横杆 + 4 个彩色列
#   - 上/下腔分配固定为 2 列 + 2 列，样例间反复出现，肉眼可总结
#   - 内侧凸起高度只有 1 或 2 两档，避免过多自由度；外侧固定为 4，形成稳定的“穿隔板”视觉锚点
#   - 每列颜色仍单色且 4 列颜色互异，避免颜色映射难题
#
# hidden_assumptions_review:
#   - 原代码 `target_base_row = slot_interior_bottom - 1` 依赖“底边上方还要留一行”；变体中为了给下腔留空间，底边直接贴到 row 6
#   - 原代码 `interior_top = sr0 + 1` 假设所有列共享同一顶部；现在必须按列判定上腔或下腔
#   - 原代码从整列提取颜色，会把隔板另一侧的区域也算进去；现在必须只在本列所属腔体内提取
#   - 原代码可在整个 interior 中覆写颜色；现在如果不按腔体裁剪，就会把颜色错误地写穿隔板
#
# placement_legality:
#   - 框架 bbox 固定 rows 0..7, cols 9..19；横杆输入 bbox 固定 rows 3..6, cols 1..7
#   - 输出平移后横杆 bbox 为 rows 3..6, cols 11..17，完全落在框架 bbox 内
#   - 外侧高凸起只在隔板孔列 11/17 穿过 row 3；内侧低凸起不会碰上腔
#   - 生成器只在各自腔体的 3 格竖井内放灰帽和颜色，不越界、不与框架墙体冲突
#
# visual_observability:
#   - 输入已能看见隔板把内部切成两层，且外侧目标列在上层、内侧目标列在下层
#   - 输出中外侧两根凸起穿过隔板，内侧两根停在隔板下方，规则可从 2-3 个样例直接归纳
#   - 颜色不参与排序、不参与映射，只承担“这一列的内容被保留多少”的可视化作用

BG = 0
GREY = 5
PALETTE = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)

H, W = 8, 20

FRAME_R0, FRAME_C0 = 0, 9
FRAME_H, FRAME_W = 8, 11
LEFT_WALL_OPEN_ROWS = (4, 5)
PARTITION_ROW = 3

SLOT_COLS = [11, 13, 15, 17]
UPPER_COLS = {11, 17}
LOWER_COLS = {13, 15}
SHAFT_ROWS = {
    11: [1, 2, 3],
    13: [4, 5, 6],
    15: [4, 5, 6],
    17: [1, 2, 3],
}

BAR_INPUT_C0 = 1
BAR_OUTPUT_C0 = 11
BAR_R0 = 3
BAR_H, BAR_W = 4, 7
BAR_LOCAL_PROJ_COLS = [0, 2, 4, 6]
BAR_TARGET_HEIGHTS_OUTER = 4
BAR_TARGET_HEIGHTS_INNER = (1, 2)

MAX_ATTEMPTS = 300


def _bbox(pts):
    return bbox_coords(pts if isinstance(pts, np.ndarray) else np.asarray(list(pts), dtype=int))


def _cc4(mask):
    return components_mask(mask, connectivity=4)


def _most_common(arr):
    vals, counts = np.unique(np.asarray(arr), return_counts=True)
    return int(vals[counts.argmax()])


def _frame_patch() -> np.ndarray:
    patch = np.zeros((FRAME_H, FRAME_W), dtype=int)
    patch[[0, -1], :] = GREY
    patch[:, -1] = GREY
    patch[:, 0] = GREY
    patch[list(LEFT_WALL_OPEN_ROWS), 0] = BG
    patch[PARTITION_ROW, :] = GREY
    patch[PARTITION_ROW, 2] = BG   # abs col 11
    patch[PARTITION_ROW, 8] = BG   # abs col 17
    return patch


def _bar_patch(heights: list[int]) -> np.ndarray:
    patch = np.zeros((BAR_H, BAR_W), dtype=int)
    patch[-1, :] = GREY
    for local_c, height in zip(BAR_LOCAL_PROJ_COLS, heights):
        patch[BAR_H - height : BAR_H, local_c] = GREY
    return patch


def _paint_input_column(grid: np.ndarray, col: int, color: int, count: int) -> None:
    rows = SHAFT_ROWS[col]
    cap = len(rows) - count
    for i, r in enumerate(rows):
        grid[r, col] = GREY if i < cap else color


def _paint_output_column(grid: np.ndarray, col: int, color: int, count: int, height: int) -> None:
    rows = SHAFT_ROWS[col]
    cap = len(rows) - count
    for i in range(cap):
        grid[rows[i], col] = GREY

    bar_rows = [BAR_R0 + BAR_H - height + k for k in range(height)]
    visible_rows = [r for r in rows if r < min(r for r in bar_rows if r in rows)]
    visible = min(count, len(visible_rows))
    if visible <= 0:
        return
    for r in visible_rows[-visible:]:
        grid[r, col] = color


def _build_grid(heights: list[int], colors: list[int], counts: list[int], bar_c0: int, output_mode: bool) -> np.ndarray:
    grid = np.zeros((H, W), dtype=int)
    paste(grid, _frame_patch(), FRAME_R0, FRAME_C0)
    paste(grid, _bar_patch(heights), BAR_R0, bar_c0, transparent=BG)
    for col, color, count, height in zip(SLOT_COLS, colors, counts, heights):
        if output_mode:
            _paint_output_column(grid, col, color, count, height)
        else:
            _paint_input_column(grid, col, color, count)
    return grid


def _sample_params(rng: np.random.Generator) -> tuple[list[int], list[int], list[int]]:
    colors = [int(v) for v in rng.choice(PALETTE, size=len(SLOT_COLS), replace=False)]
    counts = [int(rng.integers(1, 4)) for _ in SLOT_COLS]
    h13 = int(rng.choice(BAR_TARGET_HEIGHTS_INNER))
    h15 = int(rng.choice(BAR_TARGET_HEIGHTS_INNER))
    heights = [BAR_TARGET_HEIGHTS_OUTER, h13, h15, BAR_TARGET_HEIGHTS_OUTER]
    return heights, colors, counts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide a grey bar into a partitioned frame, sending some projections into the upper chamber and others into the lower one.

    Input structure:
    - The background is dominant, and grey is the main structural color. Grey forms two components: a large frame on the right and a separate bar on the left.
    - The frame keeps the original outer shell and left-wall opening, but now also contains a horizontal internal divider. Two target columns pierce that divider; two stay entirely below it.
    - Colored cells appear only inside the four target columns, mixed with grey caps that mark unused cells inside each chamber.

    Rule:
    - Move the bar horizontally into the frame so its four projection columns align with the four colored target columns, with the bar base resting on the interior floor.
    - Detect which target columns belong to the upper chamber by checking where the divider row is open; the remaining target columns belong to the lower chamber.
    - For each column, keep its original color order but only within that column's chamber, stacking the surviving colored cells immediately above the inserted projection in that same chamber.

    Key insight:
    - The new task is no longer a single open cavity. The divider splits the destination into two separate vertical shafts, so each column must be interpreted relative to its own chamber before colors are restacked.
    """
    g = np.asarray(grid, dtype=int)
    _h, w = g.shape

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

    (_, slot_bbox, _), (_, _bar_bbox, bar_pts) = grey_comps[0], grey_comps[1]
    sr0, sc0, sr1, sc1 = slot_bbox

    out = np.full_like(g, bg)
    out[sr0 : sr1 + 1, sc0 : sc1 + 1] = g[sr0 : sr1 + 1, sc0 : sc1 + 1]
    slot_region = out[sr0 : sr1 + 1, sc0 : sc1 + 1]
    slot_region[(slot_region != bg) & (slot_region != grey)] = bg

    bar_base_row = max(r for r, _ in bar_pts)
    target_base_row = sr1 - 1
    dy = target_base_row - bar_base_row

    col_min_row = {}
    for r, c in bar_pts:
        col_min_row[c] = min(col_min_row.get(c, r), r)
    src_proj_cols = sorted([c for c, rmin in col_min_row.items() if rmin < bar_base_row])

    dst_cols = sorted(
        {
            int(c)
            for r, c in np.argwhere((g != bg) & (g != grey))
            if sr0 <= int(r) <= sr1 and sc0 <= int(c) <= sc1
        }
    )

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

    interior_cols = list(range(sc0 + 1, sc1))
    divider_row = max(
        range(sr0 + 1, sr1),
        key=lambda r: sum(g[r, c] == grey for c in interior_cols),
    )

    for col in dst_cols:
        if col not in placed_by_col:
            continue

        shaft_rows = (
            list(range(sr0 + 1, divider_row + 1))
            if g[divider_row, col] != grey
            else list(range(divider_row + 1, sr1))
        )
        if not shaft_rows:
            continue

        bar_rows_here = sorted(r for r in placed_by_col[col] if r in shaft_rows)
        if not bar_rows_here:
            continue
        bar_top = bar_rows_here[0]
        allowed_rows = [r for r in shaft_rows if r < bar_top]
        if not allowed_rows:
            continue

        colors = [int(g[r, col]) for r in shaft_rows if g[r, col] != bg and g[r, col] != grey]
        if not colors:
            continue
        if len(colors) > len(allowed_rows):
            colors = colors[-len(allowed_rows) :]

        dest_rows = allowed_rows[-len(colors) :]
        for rr, colv in zip(dest_rows, colors):
            out[rr, col] = colv

    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    heights, colors, counts = _sample_params(rng)
    grid = _build_grid(heights, colors, counts, BAR_INPUT_C0, output_mode=False)
    expected = _build_grid(heights, colors, counts, BAR_OUTPUT_C0, output_mode=True)

    if get_bg(grid) != BG:
        return None
    if sorted(colors_present(grid, exclude=[BG, GREY])) != sorted(colors):
        return None
    if sorted(colors_present(expected, exclude=[BG, GREY])) != sorted(colors):
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
