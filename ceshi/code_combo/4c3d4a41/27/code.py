# Combo variant of 4c3d4a41 (index 27)
# combo_spec: D10/1+D16/0
# Name: 高度递增+排序匹配

from __future__ import annotations

import numpy as np

from arc_utils import bbox_coords, colors_present, components_mask, get_bg, paste

# Design analysis:
# variant_spec:
#   concrete_behavior:
#     - 画面仍是 8x20：右侧固定是灰色半开放框架，左侧是带 4 个凸起的灰色横杆，框架内部有 4 根彩色列。
#     - 组合改动 1（D10/1）把 4 个凸起高度限制为从左到右严格递增。由于本题高度只能取 1..4，4 根凸起的唯一合法序列就是 [1,2,3,4]。
#     - 组合改动 2（D16/0）不再按列号直接对应，而是把“凸起高度降序”与“彩色像素数量降序”做排序匹配。
#     - 两项叠加后，目标列顺序变成确定的：最高凸起固定在最右，次高在右二，依次直到最左。因此输入中“颜色最多的列”会被送到最右侧凸起上方。
#   behavior_trace:
#     - 原题：若输入 4 根彩色列分别在 [10,12,14,16]，每列颜色数是 [2,4,1,3]，则输出仍把这 4 列各自放回 [10,12,14,16] 上方。
#     - 本变体：高度固定为 [1,2,3,4]，所以凸起高度降序对应输出列 [16,14,12,10]。
#       若颜色数仍是 [2,4,1,3]，则彩色列数量降序对应输入列 [12,16,10,14]。
#       匹配后得到：输入 12->输出 16，输入 16->输出 14，输入 10->输出 12，输入 14->输出 10。
#     - 差异：原题只看“哪一列对哪一列”；本变体改为“哪一列更高 / 哪一列颜色更多”，于是颜色堆会跨列重排。
#   complexity_calibration:
#     - 保持原题的单一框架、单一横杆、4 根目标列，不引入额外对象，避免视觉噪声。
#     - 生成器强制颜色数使用 4 个互不相同的值，并让它们满足各列长度约束，消除排序并列，保证匹配唯一。
#     - 同时拒绝退化样本：若“数量降序列序”恰好等于“高度降序列序”，则排序匹配会退化为原题按列保留，生成器直接重采样。
#   implicit_assumption_review:
#     - 原 solver 默认 `dst_cols` 与 `src_proj_cols` 一一按排序后的 x 位置对应；在组合后，这个前提失效，必须改成先刚性平移横杆，再单独计算“颜色列 -> 目标凸起”的排序配对。
#     - 原 solver 通过 `col_min_row` 找到每根凸起的顶端。这里即使高度为 1（只有底座格），该列也仍必须被识别为一个合法凸起，所以要按 bar bbox 的隔列模板补全 4 根凸起信息。
#     - 原 solver 清除框架内彩色像素但保留灰帽檐。组合后这个做法仍成立：变化的是彩色堆的去向，不是灰色框架或帽檐本身。
#     - 新 solver 依赖“高度排序可区分、颜色数量排序可区分”；生成器用固定严格递增高度和互异数量保证该假设。
#   placement_legality:
#     - 框架仍固定在 cols 9..19，内部可容纳 7 宽横杆；目标列取 `start,start+2,start+4,start+6`，其中 `start ∈ {10,11,12}`，不会越界。
#     - 输入横杆左上角列只取 `{0,1}`；若取 2，则最高凸起会与框架左壁相接，灰色两组件合并，破坏原题可观察结构。
#     - 彩色列长度仍用原题的 [3,4,4,3]；数量 4 只允许落在两个中间长列上，数量 1/2/3 可落在任意合法列，始终满足列高约束。
#   visual_observability:
#     - 人眼可直接看到“左低右高”的凸起序列，也能数出每根彩色列中有几个有色格，因此排序依据完全可见。
#     - 由于最高凸起总在最右，而颜色数量顺序每题都变化，解题者能从少量样例中观察到“颜色最多的列被送到更高凸起”这一稳定规律。
#   visual_cleanliness:
#     - 变化只发生在 4 根内部目标列，框架和横杆模板保持简单，避免多层标记叠加。
#     - 颜色本身不变，只是整段同色堆被搬到别的凸起上方；因此输出仍然整洁，且每列结构清晰可辨。


def _bbox(pts):
    arr = pts if isinstance(pts, np.ndarray) else np.asarray(list(pts), dtype=int)
    return bbox_coords(arr)


def _cc4(mask):
    return components_mask(mask, connectivity=4)


def _most_common(arr):
    vals, counts = np.unique(np.asarray(arr), return_counts=True)
    return int(vals[counts.argmax()])


def _projection_infos(bar_pts: np.ndarray) -> tuple[int, list[dict[str, int]]]:
    bar_base_row = max(int(r) for r, _ in bar_pts)
    _br0, bc0, _br1, bc1 = _bbox(bar_pts)

    col_min_row: dict[int, int] = {}
    for r, c in bar_pts:
        r = int(r)
        c = int(c)
        col_min_row[c] = min(col_min_row.get(c, r), r)

    infos: list[dict[str, int]] = []
    for col in range(int(bc0), int(bc1) + 1, 2):
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
        infos.append(
            {
                "col": int(col),
                "count": int(len(rows)),
                "colors": [int(g[r, col]) for r in rows],
            }
        )
    return infos


def _rigid_dx(src_proj_cols: list[int], dst_cols: list[int], w: int) -> int:
    if src_proj_cols and len(src_proj_cols) == len(dst_cols):
        deltas = [int(d) - int(s) for s, d in zip(src_proj_cols, dst_cols)]
        if len(set(deltas)) == 1:
            return int(deltas[0])

    best = (-1, 0)
    dst_set = set(int(c) for c in dst_cols)
    for cand_dx in range(-w, w + 1):
        score = sum((int(c) + cand_dx) in dst_set for c in src_proj_cols)
        if score > best[0]:
            best = (score, cand_dx)
    return int(best[1])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Insert the grey bar into the frame, then reassign color stacks by size rank under a fixed increasing-height profile.

    Input structure:
    - Grey forms two disconnected objects: a right-hand half-open frame and a left-hand bar with four upward projections.
    - The four bar projections rise strictly from left to right, so the rightmost projection is always the tallest.
    - Inside the frame, four vertical columns contain one color each, possibly beneath grey caps; the colored-cell counts are the sortable signal.

    Rule:
    - Move the entire bar upward into the frame so its base sits one row above the frame interior floor, keeping the bar rigid.
    - Align the translated bar horizontally beneath the four frame target columns.
    - Rank the bar projections by height from tallest to shortest and rank the original colored columns by colored-cell count from largest to smallest.
    - For each rank pair, place that source color stack immediately above the matched projection's top, truncating only if the available space is shorter than the stack.

    Key insight: Because the bar heights already form a visible strict left-to-right increase, height ranking becomes a deterministic right-to-left destination order, so the puzzle reduces to matching "more colored cells" with "taller projection."
    """
    g = np.asarray(grid, dtype=int)
    _h, w = g.shape

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

    slot_infos = _slot_infos(g, bg, grey, sr0, sc0, sr1, sc1)
    if not slot_infos:
        return out

    slot_interior_bottom = sr1 - 1
    target_base_row = slot_interior_bottom - 1
    dy = int(target_base_row - bar_base_row)

    dst_cols = sorted(int(info["col"]) for info in slot_infos)
    src_proj_cols = sorted(int(info["src_col"]) for info in proj_infos)
    dx = _rigid_dx(src_proj_cols, dst_cols, w)

    placed_by_src_col: dict[int, list[int]] = {}
    for r, c in bar_pts:
        rr = int(r) + dy
        cc = int(c) + dx
        if sr0 <= rr <= sr1 and sc0 <= cc <= sc1:
            out[rr, cc] = grey
            placed_by_src_col.setdefault(int(c), []).append(int(rr))

    placed_top_by_src_col = {src_col: min(rows) for src_col, rows in placed_by_src_col.items()}
    proj_rank = sorted(proj_infos, key=lambda info: (-int(info["height"]), int(info["src_col"])))
    slot_rank = sorted(slot_infos, key=lambda info: (-int(info["count"]), int(info["col"])))

    interior_top = sr0 + 1
    interior_bottom = sr1 - 1

    for proj_info, slot_info in zip(proj_rank, slot_rank):
        src_col = int(proj_info["src_col"])
        if src_col not in placed_top_by_src_col:
            continue

        out_col = int(src_col + dx)
        bar_top = int(placed_top_by_src_col[src_col])
        desired_bottom = int(bar_top - 1)
        if not (interior_top <= desired_bottom <= interior_bottom):
            continue

        colors = list(slot_info["colors"])
        max_len = int(desired_bottom - interior_top + 1)
        if max_len <= 0:
            continue
        if len(colors) > max_len:
            colors = colors[-max_len:]

        start_row = max(interior_top, desired_bottom - len(colors) + 1)
        for i, color in enumerate(colors):
            rr = start_row + i
            if interior_top <= rr <= interior_bottom:
                out[rr, out_col] = int(color)

    return out


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
BAR_HEIGHTS = [1, 2, 3, 4]
# Keep the bar as a separate grey component in the input.
# `2` would make the tallest rightmost projection touch the frame wall.
BAR_INPUT_C0_CHOICES = [0, 1]

MAX_ATTEMPTS = 500


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


def _build_input(slot_start: int, bar_input_c0: int, heights: list[int], colors: list[int], counts: list[int]) -> np.ndarray:
    grid = np.zeros((H, W), dtype=int)
    paste(grid, _frame_patch(), FRAME_R0, FRAME_C0)
    paste(grid, _bar_patch(heights), BAR_R0, bar_input_c0)
    for col, length, color, count in zip(_slot_cols(slot_start), SLOT_LENS, colors, counts):
        _paint_input_slot_column(grid, col, length, int(color), int(count))
    return grid


def _build_expected(
    inp: np.ndarray,
    slot_start: int,
    heights: list[int],
    colors: list[int],
    counts: list[int],
    *,
    rank_match: bool,
) -> np.ndarray:
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

    if rank_match:
        slot_pairs = sorted(
            [(int(count), int(col), int(color)) for col, count, color in zip(slot_cols, counts, colors)],
            key=lambda t: (-t[0], t[1]),
        )
    else:
        slot_pairs = [(int(count), int(col), int(color)) for col, count, color in zip(slot_cols, counts, colors)]

    for (height, out_col), (count, _src_col, color) in zip(proj_rank if rank_match else proj_rank[::-1], slot_pairs):
        desired_bottom = BAR_BASE_ABS_ROW - int(height)
        visible = min(int(count), int(desired_bottom - SLOT_TOP_ROW + 1))
        if visible <= 0:
            continue
        start_row = desired_bottom - visible + 1
        out[start_row : desired_bottom + 1, out_col] = int(color)

    return out


def _sample_counts(rng: np.random.Generator) -> list[int]:
    counts: list[int | None] = [None, None, None, None]
    pos4 = int(rng.choice([1, 2]))
    counts[pos4] = 4
    remaining = [1, 2, 3]
    rng.shuffle(remaining)
    for idx in range(4):
        if counts[idx] is None:
            counts[idx] = int(remaining.pop())
    return [int(v) for v in counts]


def _rank_cols_by_counts(slot_cols: list[int], counts: list[int]) -> list[int]:
    pairs = [(int(count), int(col)) for col, count in zip(slot_cols, counts)]
    pairs.sort(key=lambda t: (-t[0], t[1]))
    return [col for _count, col in pairs]


def _rank_cols_by_heights(slot_start: int, heights: list[int]) -> list[int]:
    pairs = [(int(height), int(slot_start + local_c)) for local_c, height in zip(BAR_LOCAL_PROJ_COLS, heights)]
    pairs.sort(key=lambda t: (-t[0], t[1]))
    return [col for _height, col in pairs]


def _variant_visible(slot_start: int, heights: list[int], counts: list[int]) -> bool:
    return _rank_cols_by_counts(_slot_cols(slot_start), counts) != _rank_cols_by_heights(slot_start, heights)


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    slot_start = int(rng.choice(SLOT_START_CHOICES))
    bar_input_c0 = int(rng.choice(BAR_INPUT_C0_CHOICES))
    heights = list(BAR_HEIGHTS)
    counts = _sample_counts(rng)
    colors = [int(v) for v in rng.choice(PALETTE, size=4, replace=False)]

    if not _variant_visible(slot_start, heights, counts):
        return None

    inp = _build_input(slot_start, bar_input_c0, heights, colors, counts)
    expected = _build_expected(inp, slot_start, heights, colors, counts, rank_match=True)

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
