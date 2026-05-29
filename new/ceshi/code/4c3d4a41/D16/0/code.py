# Variant of 4c3d4a41
# Dimension: D16 — 杆移动的水平偏移计算 (match_cond)
# Change: 水平偏移(dx)通过凸起列与彩色列的列号差计算，所有凸起的偏移量相同（刚性平移）... → 凸起列号排序匹配

from __future__ import annotations

import numpy as np

from arc_utils import bbox_coords, colors_present, components_mask, get_bg, paste

# Design analysis:
# variant_spec:
#   concrete_behavior:
#     - 画面仍是 8x20，右侧是灰色半开放框架，左侧是带 4 个向上凸起的灰色横杆。
#     - 框架内部仍有 4 根竖直彩色列；每列由“若干灰色帽檐 + 若干同色像素”组成。
#     - 横杆移入框架后，横向位置仍是刚性平移到 4 个内部目标列之下；但“哪一根彩色列堆到哪一个凸起上”不再按原列号对应，
#       而是按排序对应：凸起按高度从高到低排序，彩色列按有色像素数量从多到少排序，二者一一配对。
#     - 因此输出里颜色可能换列，但仍只落在 4 个可见目标列上，且都堆在各自配对凸起的顶部上方。
#   behavior_trace:
#     - 原逻辑：输出列 11/13/15/17 上的颜色，直接来自输入中同列的颜色列，横向关系完全是“位置对位置”。
#     - 变体逻辑：假设 4 个凸起高度从高到低对应输出列 [15,11,17,13]，而输入中彩色数量从多到少对应列 [11,17,13,15]。
#       那么输出里：输入 11 列的颜色会去 15 列，输入 17 列的颜色会去 11 列，依此类推。
#     - 差异：原题看列号；变体看相对大小顺序。相同的 4 根彩色列会被重新分配到不同凸起上。
#   complexity_calibration:
#     - 只保留 4 个凸起与 4 根彩色列，避免组合过多。
#     - 生成器强制使用互不相同的高度 {1,2,3,4} 与互不相同的着色数量 {1,2,3,4}，避免排序并列导致歧义。
#     - 同时强制“高度排序列序 != 数量排序列序”，保证样例里确实发生跨列重分配，而不是退化为原题。
#   implicit_assumption_review:
#     - 原 solver 中 `dst_cols = sorted(...)` + 逐列搬运，默认“输入第 k 个彩色列最后仍去第 k 个目标列”；该前提在变体中失效。
#     - 原 solver 保留框架内部的灰色帽檐，这在变体中仍然可用，因为 4 个输出列集合不变；每一列都会收到新的颜色覆盖，不会留下孤立彩色像素。
#     - 新 solver 额外依赖：横杆仍是固定 7 宽、凸起落在每隔一列的位置，因此即使高度为 1（只剩底座格）也能被识别为“最矮凸起”。
#       生成器保持该模板，并用全异高度和全异数量保证排序稳定。
#   placement_legality:
#     - 框架内部 9 列宽，4 根目标列始终取等距列块 `start,start+2,start+4,start+6`，其中 `start ∈ {10,11,12}`，所以横杆宽 7 总能完整放入。
#     - 输入横杆左上角列限制在 {0,1,2}，与右侧框架严格分离，不重叠不越界。
#   visual_observability:
#     - 人眼可直接比较“更高的凸起”与“颜色更多的列”，看到颜色堆叠被重新分发到不同列。
#     - 排序只依赖可见结构属性：凸起高度、彩色像素个数；不依赖颜色 id。
#   visual_cleanliness:
#     - 仍只在 4 根内部列上发生变化，没有额外标记层；框架、横杆、颜色堆叠三层关系清楚。
#   color_inference:
#     - 颜色本身不被变换，只是整列颜色堆叠被搬到别的凸起上；解题者只需推断“搬运依据”而不是颜色映射。


def _bbox(pts):
    arr = pts if isinstance(pts, np.ndarray) else np.asarray(list(pts), dtype=int)
    return bbox_coords(arr)


def _cc4(mask):
    return components_mask(mask, connectivity=4)


def _most_common(arr):
    vals, counts = np.unique(np.asarray(arr), return_counts=True)
    return int(vals[counts.argmax()])


def _projection_infos(bar_pts: np.ndarray) -> tuple[int, list[dict[str, int]]]:
    bar_bbox = _bbox(bar_pts)
    bar_base_row = max(int(r) for r, _ in bar_pts)
    col_min_row: dict[int, int] = {}
    for r, c in bar_pts:
        r = int(r)
        c = int(c)
        col_min_row[c] = min(col_min_row.get(c, r), r)

    infos: list[dict[str, int]] = []
    _r0, c0, _r1, c1 = bar_bbox
    for col in range(int(c0), int(c1) + 1, 2):
        top = int(col_min_row.get(col, bar_base_row))
        infos.append(
            {
                "src_col": int(col),
                "top_row": int(top),
                "height": int(bar_base_row - top + 1),
            }
        )
    return int(bar_base_row), infos


def _slot_infos(g: np.ndarray, bg: int, grey: int, sr0: int, sc0: int, sr1: int, sc1: int) -> list[dict[str, object]]:
    infos: list[dict[str, object]] = []
    for col in range(sc0, sc1 + 1):
        rows = [r for r in range(sr0, sr1 + 1) if g[r, col] != bg and g[r, col] != grey]
        if not rows:
            continue
        colors = [int(g[r, col]) for r in rows]
        infos.append(
            {
                "col": int(col),
                "count": int(len(colors)),
                "colors": colors,
            }
        )
    return infos


def _rigid_dx(src_proj_cols: list[int], dst_cols: list[int], w: int) -> int:
    dx = None
    if src_proj_cols and len(src_proj_cols) == len(dst_cols):
        deltas = [d - s for s, d in zip(src_proj_cols, dst_cols)]
        if len(set(deltas)) == 1:
            dx = deltas[0]
    if dx is not None:
        return int(dx)

    best = (-1, 0)
    dst_set = set(dst_cols)
    for cand_dx in range(-w, w + 1):
        score = sum((c + cand_dx) in dst_set for c in src_proj_cols)
        if score > best[0]:
            best = (score, cand_dx)
    return int(best[1])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fit a grey bar into a grey frame, then redistribute colored stacks by rank instead of by x-position.

    Input structure:
    - The background is plain, while grey forms two disconnected structures: a right-hand frame and a left-hand bar with four upward projections.
    - Inside the frame, four vertical columns contain colored pixels below optional grey caps; the number of colored cells per column is meaningful.

    Rule:
    - Move the bar upward into the frame so its base sits one row above the frame's interior floor.
    - Place the bar horizontally as one rigid piece beneath the four interior target columns.
    - Rank the bar projections by height (tallest first) and rank the frame's colored columns by colored-cell count (largest first).
    - Transfer each original colored stack to the output column of the projection with the same rank, stacking it immediately above that projection.

    Key insight: The destination of each colored stack is decided by an order statistic match between visible sizes, not by preserving the original column index.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    bg = get_bg(g)
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

    bar_base_row, proj_infos = _projection_infos(bar_pts)
    if not proj_infos:
        return out

    slot_interior_bottom = sr1 - 1
    target_base_row = slot_interior_bottom - 1
    dy = int(target_base_row - bar_base_row)

    slot_infos = _slot_infos(g, bg, grey, sr0, sc0, sr1, sc1)
    dst_cols = sorted(int(info["col"]) for info in slot_infos)
    src_proj_cols = sorted(int(info["src_col"]) for info in proj_infos)
    dx = _rigid_dx(src_proj_cols, dst_cols, w)

    placed = []
    placed_by_src_col: dict[int, list[tuple[int, int]]] = {}
    for r, c in bar_pts:
        rr = int(r) + dy
        cc = int(c) + dx
        if sr0 <= rr <= sr1 and sc0 <= cc <= sc1:
            out[rr, cc] = grey
            placed.append((rr, cc))
            placed_by_src_col.setdefault(int(c), []).append((rr, cc))

    placed_top_by_src_col = {src_col: min(rr for rr, _ in cells) for src_col, cells in placed_by_src_col.items()}
    proj_rank = sorted(proj_infos, key=lambda info: (-int(info["height"]), int(info["src_col"])))
    slot_rank = sorted(slot_infos, key=lambda info: (-int(info["count"]), int(info["col"])))

    interior_top = sr0 + 1
    interior_bottom = sr1 - 1

    for proj_info, slot_info in zip(proj_rank, slot_rank):
        src_col = int(proj_info["src_col"])
        if src_col not in placed_top_by_src_col:
            continue

        out_col = src_col + dx
        bar_top = int(placed_top_by_src_col[src_col])
        desired_bottom = bar_top - 1
        if not (interior_top <= desired_bottom <= interior_bottom):
            continue

        colors = list(slot_info["colors"])
        max_len = desired_bottom - interior_top + 1
        if max_len <= 0:
            continue
        if len(colors) > max_len:
            colors = colors[-max_len:]

        start_row = max(interior_top, desired_bottom - len(colors) + 1)
        for i, colv in enumerate(colors):
            rr = start_row + i
            if interior_top <= rr <= interior_bottom:
                out[rr, out_col] = int(colv)

    return out


# --------------- generator (D16 rank-match variant) ---------------

BG = 0
GREY = 5
PALETTE = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)

H, W = 8, 20

FRAME_R0, FRAME_C0 = 0, 9
FRAME_H, FRAME_W = 8, 11
LEFT_WALL_OPEN_ROWS = (4, 5)

SLOT_TOP_ROW = 1
SLOT_LENS = [3, 4, 4, 3]
SLOT_START_CHOICES = [10, 11, 12]

BAR_R0 = 2
BAR_H, BAR_W = 4, 7
BAR_LOCAL_PROJ_COLS = [0, 2, 4, 6]
BAR_BASE_ABS_ROW = BAR_R0 + BAR_H - 1
BAR_INPUT_C0_CHOICES = [0, 1]

MAX_ATTEMPTS = 400


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


def _slot_cols(slot_start: int) -> list[int]:
    return [int(slot_start + 2 * i) for i in range(4)]


def _paint_input_slot_column(grid: np.ndarray, col: int, length: int, color: int, count: int) -> None:
    cap = int(length - count)
    grid[SLOT_TOP_ROW : SLOT_TOP_ROW + cap, col] = GREY
    grid[SLOT_TOP_ROW + cap : SLOT_TOP_ROW + length, col] = int(color)


def _rank_cols_by_heights(slot_start: int, heights: list[int]) -> list[int]:
    pairs = [(int(height), int(slot_start + local_c)) for local_c, height in zip(BAR_LOCAL_PROJ_COLS, heights)]
    pairs.sort(key=lambda t: (-t[0], t[1]))
    return [col for _height, col in pairs]


def _rank_cols_by_counts(slot_cols: list[int], counts: list[int]) -> list[int]:
    pairs = [(int(count), int(col)) for col, count in zip(slot_cols, counts)]
    pairs.sort(key=lambda t: (-t[0], t[1]))
    return [col for _count, col in pairs]


def _build_input(slot_start: int, bar_input_c0: int, heights: list[int], colors: list[int], counts: list[int]) -> np.ndarray:
    grid = np.zeros((H, W), dtype=int)
    paste(grid, _frame_patch(), FRAME_R0, FRAME_C0)
    paste(grid, _bar_patch(heights), BAR_R0, bar_input_c0)
    for col, length, color, count in zip(_slot_cols(slot_start), SLOT_LENS, colors, counts):
        _paint_input_slot_column(grid, col, length, int(color), int(count))
    return grid


def _build_expected(inp: np.ndarray, slot_start: int, heights: list[int], colors: list[int], counts: list[int]) -> np.ndarray:
    slot_cols = _slot_cols(slot_start)
    out = np.full((H, W), BG, dtype=int)
    out[FRAME_R0 : FRAME_R0 + FRAME_H, FRAME_C0 : FRAME_C0 + FRAME_W] = inp[
        FRAME_R0 : FRAME_R0 + FRAME_H, FRAME_C0 : FRAME_C0 + FRAME_W
    ]
    slot_region = out[FRAME_R0 : FRAME_R0 + FRAME_H, FRAME_C0 : FRAME_C0 + FRAME_W]
    slot_region[(slot_region != BG) & (slot_region != GREY)] = BG

    bar_patch = _bar_patch(heights)
    for rr in range(BAR_H):
        for cc in range(BAR_W):
            if bar_patch[rr, cc] == GREY:
                out[BAR_R0 + rr, slot_start + cc] = GREY

    proj_rank = sorted(
        [(int(height), int(slot_start + local_c)) for local_c, height in zip(BAR_LOCAL_PROJ_COLS, heights)],
        key=lambda t: (-t[0], t[1]),
    )
    slot_rank = sorted(
        [(int(count), int(col), int(color)) for col, count, color in zip(slot_cols, counts, colors)],
        key=lambda t: (-t[0], t[1]),
    )

    for (height, out_col), (count, _src_col, color) in zip(proj_rank, slot_rank):
        desired_bottom = BAR_BASE_ABS_ROW - height
        visible = min(int(count), int(desired_bottom - SLOT_TOP_ROW + 1))
        if visible <= 0:
            continue
        start_row = desired_bottom - visible + 1
        out[start_row : desired_bottom + 1, out_col] = int(color)

    return out


def _sample_counts(rng: np.random.Generator) -> list[int]:
    counts = [None, None, None, None]
    pos4 = int(rng.choice([1, 2]))
    counts[pos4] = 4
    remaining_vals = [1, 2, 3]
    rng.shuffle(remaining_vals)
    for idx in range(4):
        if counts[idx] is None:
            counts[idx] = int(remaining_vals.pop())
    return [int(v) for v in counts]


def _sample_params(rng: np.random.Generator) -> tuple[int, int, list[int], list[int], list[int]]:
    slot_start = int(rng.choice(SLOT_START_CHOICES))
    bar_input_c0 = int(rng.choice(BAR_INPUT_C0_CHOICES))
    heights = [int(v) for v in rng.permutation([1, 2, 3, 4])]
    counts = _sample_counts(rng)
    colors = [int(v) for v in rng.choice(PALETTE, size=4, replace=False)]
    return slot_start, bar_input_c0, heights, colors, counts


def _variant_visible(slot_start: int, heights: list[int], counts: list[int]) -> bool:
    slot_cols = _slot_cols(slot_start)
    return _rank_cols_by_heights(slot_start, heights) != _rank_cols_by_counts(slot_cols, counts)


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    slot_start, bar_input_c0, heights, colors, counts = _sample_params(rng)
    if not _variant_visible(slot_start, heights, counts):
        return None

    inp = _build_input(slot_start, bar_input_c0, heights, colors, counts)
    expected = _build_expected(inp, slot_start, heights, colors, counts)

    if get_bg(inp) != BG:
        return None
    if sorted(colors_present(inp, exclude=[BG, GREY])) != sorted(colors):
        return None
    if np.array_equal(inp, expected):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {MAX_ATTEMPTS} attempts")
