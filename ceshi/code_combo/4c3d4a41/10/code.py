# Combo variant of 4c3d4a41 (index 10)
# combo_spec: D6/0+D14/0
# Name: 紧凑裁剪+灰色填充

import numpy as np

from arc_utils import bbox_coords, colors_present, components_mask, get_bg, get_bg as _bg_color, paste

# design_analysis:
#   concrete_behavior:
#     input_geometry:
#       - 输入仍是原题固定 8x20 画布：右侧有一个 8x11 的灰色半开放框架，左侧有一个独立灰色横杆。
#       - 框架内部 4 个奇数列各自包含“顶部连续灰帽 + 底部单色块”的竖列。
#     fused_output:
#       - D6/0 先把输出限制为框架 bbox，本质上只保留 cols 9..19，对应一个紧凑 8x11 结果。
#       - D14/0 再重建每个目标列：横杆凸起与彩色列对齐后，彩色块仍贴着凸起顶部放置，
#         但灰帽与彩色块之间若出现空洞，整段都补成灰色，形成连续灰柱。
#     traced_example:
#       - 以右端短列为例：原框架内该列长度 L=3，输入只有 1 个彩色像素，因此输入列(行 1..6)像
#         [5, 5, C, 0, 0, 0]。
#       - 若对应凸起高度 h=1，则 bar_top=5，颜色可放到 desired_bottom=4。
#       - 原题输出该列会变成 [5, 5, 0, C, 5, 0]，因为颜色被压到凸起上方，但中间空 1 格不管。
#       - 本融合输出改成 [5, 5, 5, C, 5, 0]，并且整幅图还会裁成只剩框架的 8x11。
#   complexity_calibration:
#     - 仅靠裁剪就能区分 D6，但 D14 可能退化为不可见，所以生成器必须稳定制造至少一列补灰。
#     - 右端短列(col 17)天然最适合承载 D14：它列长只有 3，而 h=1 时可见容量是 4，必然产生 1 格补灰。
#     - 左端短列若取 h=1，会让最左凸起不再高于 base，破坏 solver 用“高于底边的灰像素列”检测凸起列的假设。
#       因此保留 h0>=2，并把右端短列固定为 h=1，使变化显著但自由度仍可控。
#   implicit_assumption_review:
#     - `grey` 仍可由“非背景中最常见颜色”识别；融合没有引入第二种结构色。
#     - 最大灰色 4 连通块仍是框架，次大灰色块仍是横杆；裁剪发生在输出阶段，不改变输入连通性。
#     - `src_proj_cols` 依赖“真正凸起的列在 base 之上至少还有 1 格灰像素”，所以 generator 保证最左凸起高度 >= 2。
#     - 目标列仍然是一列一种颜色、顶部灰帽连续，solver 才能通过扫描输入顶部灰帽长度来决定补灰起点。
#     - 裁剪后输出局部坐标改变，但内部几何不变，因此 solver 应先在全局坐标完成对接，再按框架 bbox 裁切。
#   placement_legality:
#     - 输入始终使用原题合法布局：bar 在 cols 1..7，frame 在 cols 9..19，中间留空，不重叠。
#     - 输出裁剪后保留的 8x11 区域正好是 frame bbox；横杆对接后占 local cols 2..8，仍完全在框架内部。
#     - 补灰只发生在目标列内部、且只写到凸起顶部之上，不会覆盖边框或越出 crop。
#   visual_observability:
#     - 每个样例都会同时呈现两个稳定信号：输出尺寸从 8x20 缩成 8x11，以及至少一列出现新增灰柱。
#     - 右端短列的补灰直接出现在彩色块下方、凸起上方，人眼可明显看出“空洞被灰色托住”。
#   visual_cleanliness:
#     - 裁剪减少左侧大块背景，让注意力集中到框架内部，反而增强了 D14 的可见性。
#     - 颜色仍只在四个竖列中出现，灰色仍只担当框架/横杆/柱托，层次清晰。
#   color_inference:
#     - 不引入颜色映射；4 种彩色仍只表示 4 个独立目标列。
#     - 规则变化完全由空间关系驱动，解题者无需依赖颜色编号本身。


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
FILL_SLOT_IDX = 3

BAR_INPUT_C0 = 1
BAR_OUTPUT_C0 = 11
BAR_R0 = 2
BAR_H, BAR_W = 4, 7
BAR_LOCAL_PROJ_COLS = [0, 2, 4, 6]
BAR_BASE_ABS_ROW = BAR_R0 + BAR_H - 1

HEIGHT0_RANGE = (2, 5)
HEIGHT_REST_RANGE = (1, 5)
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


def _paint_output_slot_column(grid: np.ndarray, col: int, length: int, color: int, count: int, height: int) -> None:
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


def _build_input_grid(heights: list[int], colors: list[int], colored_counts: list[int]) -> np.ndarray:
    grid = np.zeros((H, W), dtype=int)
    paste(grid, _frame_patch(), FRAME_R0, FRAME_C0)
    paste(grid, _bar_patch(heights), BAR_R0, BAR_INPUT_C0)
    for col, length, color, count in zip(SLOT_COLS, SLOT_LENS, colors, colored_counts):
        _paint_input_slot_column(grid, col, length, color, count)
    return grid


def _build_expected_output(heights: list[int], colors: list[int], colored_counts: list[int]) -> np.ndarray:
    out = np.zeros((FRAME_H, FRAME_W), dtype=int)
    paste(out, _frame_patch(), 0, 0)
    paste(out, _bar_patch(heights), BAR_R0 - FRAME_R0, BAR_OUTPUT_C0 - FRAME_C0)
    for col, length, color, count, height in zip(SLOT_COLS, SLOT_LENS, colors, colored_counts, heights):
        _paint_output_slot_column(out, col - FRAME_C0, length, color, count, height)
    return out


def _variant_fill_cells(heights: list[int]) -> int:
    length = SLOT_LENS[FILL_SLOT_IDX]
    height = heights[FILL_SLOT_IDX]
    return max(0, (BAR_BASE_ABS_ROW - height) - length)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Dock the grey toothed bar into the grey frame, rebuild the target columns, then crop to the frame alone.

    Input structure:
    - The background dominates the 8x20 canvas, while grey forms two disconnected structures: an open frame on the right and a toothed horizontal bar on the left.
    - Inside the frame, four colored columns sit under optional grey caps; each column uses one non-grey color and marks the columns the bar teeth must match.

    Rule:
    - Move the bar into the frame so its base sits one row above the frame floor and its tooth columns align with the colored target columns.
    - Re-stack each column's colored pixels immediately above the matched tooth, trimming overflow if the available height is smaller than the input stack.
    - If there is any gap between the preserved top grey cap and the relocated color stack, fill that gap with grey, then return only the frame bbox as a compact 8x11 output.

    Key insight:
    - The colored columns still identify where each tooth must land, but the final object is no longer a sparse full-canvas scene; it becomes a tight crop of the docked frame whose interior columns are rebuilt into continuous grey supports topped by color.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    other = g[g != bg]
    if other.size == 0:
        return g.copy()
    grey = _most_common(other)
    _, w = g.shape

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
    target_base_row = sr1 - 2
    dy = target_base_row - bar_base_row

    col_min_row = {}
    for r, c in bar_pts:
        col_min_row[c] = min(col_min_row.get(c, r), r)
    src_proj_cols = sorted(c for c, rmin in col_min_row.items() if rmin < bar_base_row)

    dst_cols = sorted(
        {
            int(c)
            for r, c in np.argwhere((g != bg) & (g != grey))
            if sr0 <= int(r) <= sr1 and sc0 <= int(c) <= sc1
        }
    )

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

        start_row = max(interior_top, desired_bottom - len(colors) + 1)

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

    return out[sr0 : sr1 + 1, sc0 : sc1 + 1]


def _sample_params(rng: np.random.Generator) -> tuple[list[int], list[int], list[int]]:
    colors = [int(v) for v in rng.choice(PALETTE, size=len(SLOT_COLS), replace=False)]
    colored_counts = [int(rng.integers(1, length + 1)) for length in SLOT_LENS]

    heights = [int(rng.integers(*HEIGHT0_RANGE))]
    heights.extend(int(rng.integers(*HEIGHT_REST_RANGE)) for _ in range(len(SLOT_COLS) - 1))
    heights[FILL_SLOT_IDX] = 1
    return heights, colors, colored_counts


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    heights, colors, colored_counts = _sample_params(rng)
    if _variant_fill_cells(heights) <= 0:
        return None

    grid = _build_input_grid(heights, colors, colored_counts)
    expected = _build_expected_output(heights, colors, colored_counts)

    if get_bg(grid) != BG:
        return None
    if sorted(colors_present(grid, exclude=[BG, GREY])) != sorted(colors):
        return None
    if main(grid).shape != (FRAME_H, FRAME_W):
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
