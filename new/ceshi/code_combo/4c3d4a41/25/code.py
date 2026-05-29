# Combo variant of 4c3d4a41 (index 25)
# combo_spec: D4/2+D9/0
# Name: 凸起着色+多色交替列
#
# Design analysis
# ===============
# fused_variant:
#   summary:
#     - 保留原题的固定几何：右侧半开放灰框、内部 4 条目标列、左侧 7 宽横杆以及 4 根与目标列间距匹配的凸起。
#     - D9/0 作用在输入端：槽内每条彩色段不再是单色，而是用同一对颜色做纵向交替。
#     - D4/2 作用在输出端：横杆插入后，列内“保留下来的彩色像素数量”仍由几何可见空间决定，但这些像素全部改刷成对应凸起自己的颜色，覆盖掉原交替模式。
#   concrete_example:
#     - 设某列输入彩色数为 4，交替色对为 [2,8]，该列自上而下彩色段是 [2,8,2,8]。
#     - 若对齐后的凸起高度为 3，则其上方只剩 2 格可见；原题会保留末 2 个输入色并继续显示 [2,8]。
#     - 融合后仍只保留 2 格，但这 2 格全部重涂成该凸起颜色，例如 [7,7]。
#     - 差异因此同时可见：输入里能看见“多色交替列”，输出里能看见“列色来自横杆凸起而非原列”。
#   complexity_calibration:
#     - 所有槽列共用同一对交替色，避免颜色种类过多，同时让“交替”在 2-4 个样例里足够重复。
#     - 4 个凸起颜色互不相同，且与交替色对完全不重合，避免输出看起来像没发生 recolor。
#     - 生成器要求至少 3 条槽列有 >=2 个彩色像素，且至少 1 条有 >=3 个彩色像素，保证人眼能稳定观察到交替。
#     - 生成器同时要求样例里既有“被截短”的列也有“完整保留”的列，避免几何机制退化。
#   implicit_assumptions_review:
#     - 原 solver 的“第二大灰色连通块就是横杆”已失效，因为凸起露出部分改成彩色；新 solver 改为先找最大灰色块作为框架，再找位于其左侧的最大非背景连通块作为横杆。
#     - 原 solver 在输出阶段复制输入列的逐像素颜色序列；融合后只能复用“彩色像素数量”，颜色来源必须切换为凸起颜色。
#     - `proj_color_by_src` 依赖“每根凸起在底座上方至少露出 1 个彩色像素”；生成器把高度限制为 2..4 来保证这一点。
#     - `dst_cols` 依赖“框架内只有 4 条含彩色像素的目标列”；生成器固定列位于 11,13,15,17 并保持其余内部列为空。
#   placement_legality:
#     - 框架固定在 cols 9..19，输入横杆固定在 cols 1..7，平移后横杆落在 cols 11..17，不越界也不与框架外区域重叠。
#     - 凸起彩色只出现在底座上方，同列仍与灰色底座 4 连通，因此横杆始终是单个连通物体。
#     - 槽列只占用框架内部奇数列，偶数列保持背景，视觉分区清晰。
#   observability_and_cleanliness:
#     - 输入样例里同一对颜色会在多条列中反复交替出现，解题者容易识别“列内不是单色”。
#     - 输出样例里这些交替色全部消失，改为 4 个来自横杆凸起的单色列，且颜色与输入交替色对不重合，变化线索明确。
#     - 彩色区仍只位于凸起顶部上方，不会与插入后的横杆互相覆盖，画面保持整洁。

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
    comps.sort(key=lambda t: (-t[0], t[1]))
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide a colored-projection bar into the frame, then replace each surviving alternating slot segment with the aligned projection color.

    Input structure:
    - A large grey half-open frame sits on the right, with four target columns inside it.
    - Each target column has a grey cap and a colored segment made from two alternating colors rather than one solid color.
    - A separate bar sits to the left: its base is grey, while each upward projection carries its own non-grey color above the base.

    Rule:
    - Move the bar into the frame so the base rests one row above the interior floor and the four projections line up with the four colored slot columns.
    - For each matched column, keep only as many colored cells as still fit above the inserted projection, using the same visibility/truncation geometry as the original task.
    - Repaint every surviving cell in that column to the matched projection's color, so the original alternating pattern is completely overwritten.

    Key insight: The slot columns still determine where and how many pixels survive, but their internal colors no longer matter after docking; the final color of each surviving stack must be read from the aligned projection on the bar.
    """
    g = np.asarray(grid, dtype=int)
    _h, w = g.shape

    bg = _bg_color(g)
    other = g[g != bg]
    if other.size == 0:
        return g.copy()
    grey = _most_common(other)

    grey_comps = _sorted_components(g == grey)
    if not grey_comps:
        return g.copy()
    _, slot_bbox, _slot_pts = grey_comps[0]
    sr0, sc0, sr1, sc1 = slot_bbox

    fg_comps = _sorted_components(g != bg)
    bar_entry = None
    for entry in fg_comps:
        _size, bbox, _pts = entry
        _br0, bc0, _br1, bc1 = bbox
        if bc1 < sc0:
            bar_entry = entry
            break
    if bar_entry is None:
        return g.copy()
    _, _bar_bbox, bar_pts = bar_entry

    out = np.full_like(g, bg)
    out[sr0 : sr1 + 1, sc0 : sc1 + 1] = g[sr0 : sr1 + 1, sc0 : sc1 + 1]
    slot_region = out[sr0 : sr1 + 1, sc0 : sc1 + 1]
    slot_region[(slot_region != bg) & (slot_region != grey)] = bg

    bar_base_row = max(int(r) for r, _c in bar_pts)
    slot_interior_bottom = sr1 - 1
    target_base_row = slot_interior_bottom - 1
    dy = target_base_row - bar_base_row

    col_min_row: dict[int, int] = {}
    proj_color_by_src: dict[int, int] = {}
    for r, c in bar_pts:
        r = int(r)
        c = int(c)
        col_min_row[c] = min(col_min_row.get(c, r), r)
        val = int(g[r, c])
        if r < bar_base_row and val != grey and val != bg:
            proj_color_by_src.setdefault(c, val)

    src_proj_cols = sorted(c for c, rmin in col_min_row.items() if rmin < bar_base_row)
    if not src_proj_cols:
        return g.copy()

    dst_cols: list[int] = []
    colored_count_by_col: dict[int, int] = {}
    for col in range(sc0, sc1 + 1):
        rows = [
            r
            for r in range(sr0, sr1 + 1)
            if int(g[r, col]) != bg and int(g[r, col]) != grey
        ]
        if rows:
            dst_cols.append(col)
            colored_count_by_col[col] = len(rows)

    dx = None
    if len(src_proj_cols) == len(dst_cols):
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

    placed_by_col: dict[int, list[int]] = {}
    proj_color_by_dst: dict[int, int] = {}
    for r, c in bar_pts:
        rr = int(r) + dy
        cc = int(c) + dx
        if sr0 <= rr <= sr1 and sc0 <= cc <= sc1:
            out[rr, cc] = int(g[int(r), int(c)])
            placed_by_col.setdefault(cc, []).append(rr)
    for src_col, color in proj_color_by_src.items():
        dst_col = src_col + dx
        if sc0 <= dst_col <= sc1:
            proj_color_by_dst[dst_col] = color

    interior_top = sr0 + 1
    interior_bottom = sr1 - 1

    for col in dst_cols:
        if col not in placed_by_col or col not in proj_color_by_dst:
            continue
        bar_top = min(placed_by_col[col])
        desired_bottom = bar_top - 1
        if desired_bottom < interior_top:
            continue

        count = colored_count_by_col[col]
        visible = min(count, desired_bottom - interior_top + 1)
        if visible <= 0:
            continue

        start_row = desired_bottom - visible + 1
        for rr in range(start_row, desired_bottom + 1):
            if interior_top <= rr <= interior_bottom:
                out[rr, col] = proj_color_by_dst[col]

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

BAR_INPUT_C0 = 1
BAR_OUTPUT_C0 = 11
BAR_R0 = 2
BAR_H, BAR_W = 4, 7
BAR_LOCAL_PROJ_COLS = [0, 2, 4, 6]
BAR_BASE_ABS_ROW = BAR_R0 + BAR_H - 1

HEIGHT_RANGE = (2, 5)  # {2,3,4}; every projection exposes at least one colored cell
MAX_ATTEMPTS = 400


def _frame_patch() -> np.ndarray:
    patch = np.zeros((FRAME_H, FRAME_W), dtype=int)
    patch[[0, -1], :] = GREY
    patch[:, -1] = GREY
    patch[:, 0] = GREY
    patch[list(LEFT_WALL_OPEN_ROWS), 0] = BG
    return patch


def _bar_patch(heights: list[int], proj_colors: list[int]) -> np.ndarray:
    patch = np.zeros((BAR_H, BAR_W), dtype=int)
    patch[-1, :] = GREY
    for local_c, height, color in zip(BAR_LOCAL_PROJ_COLS, heights, proj_colors):
        patch[BAR_H - height : BAR_H - 1, local_c] = color
    return patch


def _paint_input_slot_column(
    grid: np.ndarray,
    col: int,
    length: int,
    alt_colors: list[int],
    count: int,
    phase: int,
) -> None:
    cap = length - count
    grid[SLOT_TOP_ROW : SLOT_TOP_ROW + cap, col] = GREY
    for idx in range(count):
        rr = SLOT_TOP_ROW + cap + idx
        grid[rr, col] = alt_colors[(phase + idx) % 2]


def _paint_output_slot_column(
    grid: np.ndarray,
    col: int,
    length: int,
    count: int,
    proj_color: int,
    height: int,
) -> None:
    cap = length - count
    grid[SLOT_TOP_ROW : SLOT_TOP_ROW + cap, col] = GREY
    visible = min(count, BAR_BASE_ABS_ROW - height)
    if visible <= 0:
        return
    row_end = BAR_BASE_ABS_ROW - height
    row_start = row_end - visible + 1
    grid[row_start : row_end + 1, col] = proj_color


def _build_grid(
    heights: list[int],
    alt_colors: list[int],
    proj_colors: list[int],
    colored_counts: list[int],
    bar_c0: int,
    output_mode: bool,
) -> np.ndarray:
    grid = np.zeros((H, W), dtype=int)
    paste(grid, _frame_patch(), FRAME_R0, FRAME_C0)
    paste(grid, _bar_patch(heights, proj_colors), BAR_R0, bar_c0)
    for idx, (col, length, proj_color, count, height) in enumerate(
        zip(SLOT_COLS, SLOT_LENS, proj_colors, colored_counts, heights)
    ):
        if output_mode:
            _paint_output_slot_column(grid, col, length, count, proj_color, height)
        else:
            _paint_input_slot_column(grid, col, length, alt_colors, count, phase=idx % 2)
    return grid


def _sample_params(
    rng: np.random.Generator,
) -> tuple[list[int], list[int], list[int], list[int]]:
    while True:
        perm = [int(v) for v in rng.permutation(PALETTE)]
        alt_colors = perm[:2]
        proj_colors = perm[2:6]
        colored_counts = [int(rng.integers(1, length + 1)) for length in SLOT_LENS]
        heights = [int(rng.integers(*HEIGHT_RANGE)) for _ in SLOT_COLS]
        visible_counts = [min(count, BAR_BASE_ABS_ROW - height) for count, height in zip(colored_counts, heights)]

        if sum(count >= 2 for count in colored_counts) < 3:
            continue
        if max(colored_counts) < 3:
            continue
        if len(set(heights)) < 2:
            continue
        if not any(vis < count for vis, count in zip(visible_counts, colored_counts)):
            continue
        if not any(vis == count for vis, count in zip(visible_counts, colored_counts)):
            continue
        if len(set(proj_colors)) < 4:
            continue
        return heights, alt_colors, proj_colors, colored_counts


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    heights, alt_colors, proj_colors, colored_counts = _sample_params(rng)
    grid = _build_grid(
        heights,
        alt_colors,
        proj_colors,
        colored_counts,
        BAR_INPUT_C0,
        output_mode=False,
    )
    expected = _build_grid(
        heights,
        alt_colors,
        proj_colors,
        colored_counts,
        BAR_OUTPUT_C0,
        output_mode=True,
    )

    if get_bg(grid) != BG:
        return None
    if sorted(colors_present(grid, exclude=[BG, GREY])) != sorted(alt_colors + proj_colors):
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
