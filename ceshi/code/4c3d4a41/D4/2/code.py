# Variant of 4c3d4a41
# Dimension: D4 — 变换：彩色像素的重排规则 (transform)
# Change: 彩色像素在杆进入后堆叠在对应凸起顶部上方，从框架内顶部开始向下排列，数量受可用空间限制... → 颜色重新着色为凸起对应色
#
# Design analysis
# ===============
# variant_spec:
#   core_change:
#     - 保留原题的框架、横杆平移、按列对齐、以及“柱子上方可见空间决定可保留多少彩色像素”的几何机制。
#     - 只替换 D4：输出里每一列可见的彩色像素不再沿用输入列本来的颜色，而是统一改成对接到该列的凸起颜色。
#   concrete_objects:
#     - 框架仍为固定的灰色半开放矩形，内部 4 个目标列仍在 cols 11,13,15,17，列长仍为 [3,4,4,3]。
#     - 输入中的槽列仍由“上方若干灰帽 + 下方若干同色彩色像素”组成，每列至少 1 个彩色像素。
#     - 左侧横杆仍有 4 个向上凸起，但现在“底座保持灰色，凸起高出底座的部分各自带一种非灰颜色”；4 个凸起颜色互不相同。
#   behavior_trace:
#     - 原逻辑：列 i 的输出彩色区长度 = min(输入该列彩色数, 凸起上方可用高度)，颜色保持输入列颜色。
#     - 新逻辑：列 i 的输出彩色区长度计算完全不变，但这段可见彩色区全部改刷成“与该列对齐的凸起颜色”。
#     - 例：某列输入有 3 个蓝色像素，凸起插入后上方只剩 2 格；原题输出 2 个蓝色像素，新变体输出 2 个“该凸起自己的颜色”像素。
#   complexity_calibration:
#     - 若凸起仍是灰色，则 recolor 规则不可见，因此必须给凸起显式着色。
#     - 若允许凸起高度为 1，则该凸起不会在底座上方露出颜色，人类看不到映射线索；因此生成器把高度限制在 [2,4]。
#     - 若槽列颜色与凸起颜色有大量重合，输出可能看起来像没变；因此生成器强制 4 个槽列色与 4 个凸起色两两不重合。
#   implicit_assumptions_review:
#     - 原 solver 用“第二大灰色连通块就是横杆”，这个前提在变体中失效，因为凸起改成彩色后横杆不再是纯灰块。
#     - 新 solver 改为：框架仍由最大灰色连通块识别；横杆改由“位于框架左侧的最大非背景连通块”识别。
#     - 原 solver 在输出阶段把颜色逐像素从输入列复制到输出；变体下这会错误保留旧色，因此改为只保留数量信息，颜色由对齐后的凸起提供。
#   placement_legality:
#     - 横杆仍完全位于框架左侧输入区，bbox 在 cols 1..7；框架 bbox 固定在 cols 9..19，因此平移后不会越界。
#     - 凸起颜色只出现在底座上方，底座整行保持灰色，保证横杆仍是单一 4 连通物体且结构清晰。
#   visual_observability:
#     - 每个样例中同时展示 4 个不同槽列颜色和 4 个不同凸起颜色；输出时 4 列都会改成另一组颜色，映射可直接肉眼比对。
#     - 高度仍控制可见彩色长度，样例中既能看出“哪一列对哪一根凸起”，也能看出“颜色来源已经换成横杆而不是原列”。
#   cleanliness:
#     - 输出中的彩色像素仍只出现在凸起上方可见区域，不与插入后的横杆重叠，因此画面不会混乱。

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
    Dock a colored-projection bar into the frame, then repaint the surviving slot pixels with the matched projection colors.

    Input structure:
    - Background plus a large grey half-open frame on the right containing four grey-capped colored columns.
    - A separate bar on the left with a grey base and four upward projections whose visible parts are non-grey and distinct.

    Rule:
    - Translate the bar into the frame so its base sits just above the frame floor and its projections align with the colored slot columns.
    - For each aligned column, keep as many colored cells as can still fit above the inserted projection, exactly as in the original docking geometry.
    - Recolor those surviving cells to the color carried by the matching projection instead of preserving the slot column's old color.

    Key insight: The alignment problem is unchanged, but the informative payload now comes from the bar's projection colors, so the solver must infer a column-to-projection color mapping before painting the result.
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
        br0, bc0, br1, bc1 = bbox
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

    dst_cols = []
    colored_rows_by_col: dict[int, list[int]] = {}
    for col in range(sc0, sc1 + 1):
        rows = [
            r
            for r in range(sr0, sr1 + 1)
            if int(g[r, col]) != bg and int(g[r, col]) != grey
        ]
        if rows:
            dst_cols.append(col)
            colored_rows_by_col[col] = rows

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

        count = len(colored_rows_by_col[col])
        max_len = desired_bottom - interior_top + 1
        visible = min(count, max_len)
        if visible <= 0:
            continue

        start_row = desired_bottom - visible + 1
        for rr in range(start_row, desired_bottom + 1):
            if interior_top <= rr <= interior_bottom:
                out[rr, col] = proj_color_by_dst[col]

    return out


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

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

HEIGHT_RANGE = (2, 5)  # {2,3,4}; every projection shows at least one colored cell above the base
MAX_ATTEMPTS = 300


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


def _paint_slot_column(
    grid: np.ndarray,
    col: int,
    length: int,
    color: int,
    count: int,
    height: int | None,
) -> None:
    cap = length - count
    grid[SLOT_TOP_ROW : SLOT_TOP_ROW + cap, col] = GREY
    if height is None:
        grid[SLOT_TOP_ROW + cap : SLOT_TOP_ROW + length, col] = color
        return

    visible = min(count, BAR_BASE_ABS_ROW - height)
    if visible <= 0:
        return
    row_end = BAR_BASE_ABS_ROW - height
    grid[row_end - visible + 1 : row_end + 1, col] = color


def _build_grid(
    heights: list[int],
    slot_colors: list[int],
    proj_colors: list[int],
    colored_counts: list[int],
    bar_c0: int,
    output_mode: bool,
) -> np.ndarray:
    grid = np.zeros((H, W), dtype=int)
    paste(grid, _frame_patch(), FRAME_R0, FRAME_C0)
    paste(grid, _bar_patch(heights, proj_colors), BAR_R0, bar_c0)
    for col, length, slot_color, proj_color, count, height in zip(
        SLOT_COLS,
        SLOT_LENS,
        slot_colors,
        proj_colors,
        colored_counts,
        heights,
    ):
        paint_color = proj_color if output_mode else slot_color
        _paint_slot_column(grid, col, length, paint_color, count, height if output_mode else None)
    return grid


def _sample_params(
    rng: np.random.Generator,
) -> tuple[list[int], list[int], list[int], list[int]]:
    while True:
        perm = [int(v) for v in rng.permutation(PALETTE)]
        slot_colors = perm[: len(SLOT_COLS)]
        proj_colors = perm[len(SLOT_COLS) : 2 * len(SLOT_COLS)]
        colored_counts = [int(rng.integers(1, L + 1)) for L in SLOT_LENS]
        heights = [int(rng.integers(*HEIGHT_RANGE)) for _ in SLOT_COLS]
        if len(set(heights)) < 2:
            continue
        if sum(count > 1 for count in colored_counts) < 2:
            continue
        return heights, slot_colors, proj_colors, colored_counts


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    heights, slot_colors, proj_colors, colored_counts = _sample_params(rng)
    grid = _build_grid(
        heights,
        slot_colors,
        proj_colors,
        colored_counts,
        BAR_INPUT_C0,
        output_mode=False,
    )
    expected = _build_grid(
        heights,
        slot_colors,
        proj_colors,
        colored_counts,
        BAR_OUTPUT_C0,
        output_mode=True,
    )

    if get_bg(grid) != BG:
        return None
    if sorted(colors_present(grid, exclude=[BG, GREY])) != sorted(slot_colors + proj_colors):
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
