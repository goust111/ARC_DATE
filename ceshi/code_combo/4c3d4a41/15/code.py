# Combo variant of 4c3d4a41 (index 15)
# combo_spec: D8/0+D16/0
# Name: 可变凸起数量+排序匹配

from __future__ import annotations

import numpy as np

from arc_utils import bbox_coords, colors_present, components_mask, get_bg, paste

# Design analysis:
# variant_spec:
#   concrete_behavior:
#     - 右侧仍是灰色半开放框架，左侧仍是灰色横杆，但横杆的凸起数与框架内彩色列数不再固定为 4，而是在 2..5 之间同步变化。
#     - 横杆宽度随凸起数变成 `2*n-1`，凸起仍落在每隔一列的位置；框架内的彩色列也仍保持每隔一列的等距排布。
#     - 输出时横杆作为刚体水平移入框架，但彩色列不再按原列号对应，而是按“有色数降序”与“凸起高度降序”做排序配对。
#   behavior_trace:
#     - 例：若输入有 5 个凸起，高度从左到右为 [2,5,1,4,3]，框架内 5 根彩色列的有色数为 [4,1,5,2,3]。
#     - 原题会把第 1 根彩色列继续放到第 1 根输出凸起上、第 2 根继续放到第 2 根上，完全按横向位置直连。
#     - 本融合版先把凸起按高到低排成 [第2,第4,第5,第1,第3]，再把彩色列按有色数排成 [第3,第1,第5,第4,第2]，
#       于是“第 3 根彩色列的整段颜色”会去最高凸起上，“第 1 根彩色列”去第二高凸起上，依此类推。
#     - 因而输出既体现 D8 的“数量可变”，也体现 D16 的“排序重分配”，而不是原题那种固定四列的一一同位搬运。
#   complexity_calibration:
#     - `n` 只放在 2..5，避免自由度过大；横杆与框架内部仍沿单一等距模板生成，人眼容易先数数量再比高低/长短。
#     - 生成器强制高度与有色数都互不相同：高度使用 `1..n` 的一个排列，有色数使用 `1..n` 的一个排列，避免排序并列歧义。
#     - 同时强制“高度排序得到的目标列序”和“有色数排序得到的源列序”不完全相同，保证样本里确实发生跨列重分配。
#   implicit_assumption_review:
#     - 原 solver 默认“凸起数=彩色列数=4”，并且默认用 `sorted(dst_cols)` 与 `sorted(src_proj_cols)` 做按位置配对；这两点在融合后都失效。
#     - 新 solver 改为从 bar bbox 动态推断宽度与凸起数，从框架内动态提取全部有色列，再用排序配对；因此不再依赖固定列号 11/13/15/17。
#     - 新 solver 额外依赖：横杆仍是奇偶交替模板，左端就是第一个凸起列；彩色列仍是单色连续竖段；高度与有色数排序可稳定比较。
#     - 生成器通过固定“每隔一列的模板、单色连续竖段、唯一高度、唯一有色数”来满足这些新前提。
#   placement_legality:
#     - 框架内部只有 9 列，因此 `n=5` 时横杆宽度恰为 9，只能从内部最左列开始；`n<5` 时允许有限横向平移，但始终完整落在框架内。
#     - 输入横杆也必须保持在框架左侧，所以其左上角列被限制为 `0 .. 10-2*n`；这样宽度 `2*n-1` 的横杆永不越界，也不会碰到框架。
#     - 彩色列数量始终与凸起数量一致，因此排序配对总是一一对应，不会出现多列争抢同一凸起或有凸起闲置的非法情形。
#   visual_observability:
#     - 解题者能直接数出“有几根凸起 / 有几根彩色列”，再比较“哪根更高 / 哪根颜色更多”；排序依据完全来自可见形状，不依赖颜色编号。
#     - 当 `n` 在样例间变化时，固定四列的原题假设会被明显打破；当排序结果打乱列号时，颜色会横向换列，肉眼能看到不是原位堆叠。
#   visual_cleanliness:
#     - 仍只有三层对象：框架、横杆、彩色堆叠。输出里颜色只出现在凸起正上方，不会引入新的标记层或遮挡关系。
#     - 目标列始终隔一列留白，即使 `n=5` 也只是占满内部 9 列模板，不会和边框重叠。
#   color_inference:
#     - 颜色本身不做数值映射，只是整段单色堆叠被搬到别的凸起上；人类需要推断的是“哪一段去哪里”，不是“颜色如何改写”。


def _bbox(pts: np.ndarray | list[tuple[int, int]]) -> tuple[int, int, int, int]:
    arr = pts if isinstance(pts, np.ndarray) else np.asarray(list(pts), dtype=int)
    return bbox_coords(arr)


def _cc4(mask: np.ndarray) -> list[np.ndarray]:
    return components_mask(mask, connectivity=4)


def _most_common(arr: np.ndarray) -> int:
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


def _slot_infos(
    g: np.ndarray, bg: int, grey: int, sr0: int, sc0: int, sr1: int, sc1: int
) -> list[dict[str, object]]:
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
        deltas = [int(d) - int(s) for s, d in zip(src_proj_cols, dst_cols)]
        if len(set(deltas)) == 1:
            dx = int(deltas[0])
    if dx is not None:
        return int(dx)

    best = (-1, 0)
    dst_set = set(int(c) for c in dst_cols)
    for cand_dx in range(-w, w + 1):
        score = sum((int(c) + cand_dx) in dst_set for c in src_proj_cols)
        if score > best[0]:
            best = (score, cand_dx)
    return int(best[1])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide a grey bar with a variable number of projections into the frame, then rematch colored stacks by rank.

    Input structure:
    - Background fills most cells, while grey forms two disconnected structures: a right-hand half-open frame and a left-hand bar.
    - The bar has 2 to 5 upward projections on an every-other-column template, and the frame interior contains the same number of colored vertical stacks.
    - Each colored stack is a single-color run beneath optional grey cap cells, so its visible colored-cell count can be measured directly.

    Rule:
    - Move the whole grey bar into the frame so its base sits one row above the frame's interior bottom, using one rigid horizontal shift.
    - Detect every projection column of the moved bar and rank those projections from tallest to shortest.
    - Detect every colored column in the input frame and rank those columns from most colored cells to fewest.
    - Pair the two rankings position by position, and place each source color stack immediately above the matched projection's top cell.

    Key insight: The puzzle is no longer "same x-column goes to same x-column"; you must first count how many structures exist, then match by ordered size cues.
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
    slot_infos = _slot_infos(g, bg, grey, sr0, sc0, sr1, sc1)
    if not proj_infos or len(proj_infos) != len(slot_infos):
        return out

    slot_interior_bottom = sr1 - 1
    target_base_row = slot_interior_bottom - 1
    dy = int(target_base_row - bar_base_row)

    dst_cols = sorted(int(info["col"]) for info in slot_infos)
    src_proj_cols = sorted(int(info["src_col"]) for info in proj_infos)
    dx = _rigid_dx(src_proj_cols, dst_cols, w)

    placed_by_src_col: dict[int, list[tuple[int, int]]] = {}
    for r, c in bar_pts:
        rr = int(r) + dy
        cc = int(c) + dx
        if sr0 <= rr <= sr1 and sc0 <= cc <= sc1:
            out[rr, cc] = grey
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
        for i, colv in enumerate(colors):
            rr = start_row + i
            if interior_top <= rr <= interior_bottom:
                out[rr, out_col] = int(colv)

    return out


BG = 0
GREY = 5
PALETTE = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)

H, W = 8, 21

FRAME_R0, FRAME_C0 = 0, 10
FRAME_H, FRAME_W = 8, 11
LEFT_WALL_OPEN_ROWS = (4, 5)

SLOT_TOP_ROW = 1
SLOT_LEN = 5

BAR_R0 = 1
BAR_H = 5
BAR_BASE_ABS_ROW = BAR_R0 + BAR_H - 1

MIN_PROJS = 2
MAX_PROJS = 5
MAX_ATTEMPTS = 600


def _frame_patch() -> np.ndarray:
    patch = np.zeros((FRAME_H, FRAME_W), dtype=int)
    patch[[0, -1], :] = GREY
    patch[:, -1] = GREY
    patch[:, 0] = GREY
    patch[list(LEFT_WALL_OPEN_ROWS), 0] = BG
    return patch


def _bar_local_proj_cols(n_projs: int) -> list[int]:
    return [2 * i for i in range(int(n_projs))]


def _bar_patch(heights: list[int]) -> np.ndarray:
    width = 2 * len(heights) - 1
    patch = np.zeros((BAR_H, width), dtype=int)
    patch[-1, :] = GREY
    for local_c, height in zip(_bar_local_proj_cols(len(heights)), heights):
        patch[BAR_H - int(height) : BAR_H, int(local_c)] = GREY
    return patch


def _slot_cols(slot_start: int, n_projs: int) -> list[int]:
    return [int(slot_start + 2 * i) for i in range(int(n_projs))]


def _slot_start_choices(n_projs: int) -> list[int]:
    interior_left = FRAME_C0 + 1
    interior_right = FRAME_C0 + FRAME_W - 2
    max_start = interior_right - 2 * (int(n_projs) - 1)
    return [int(v) for v in range(interior_left, max_start + 1)]


def _bar_input_c0_choices(n_projs: int) -> list[int]:
    max_c0 = 10 - 2 * int(n_projs)
    return [int(v) for v in range(0, max_c0 + 1)]


def _paint_input_slot_column(grid: np.ndarray, col: int, color: int, count: int) -> None:
    cap = int(SLOT_LEN - count)
    grid[SLOT_TOP_ROW : SLOT_TOP_ROW + cap, col] = GREY
    grid[SLOT_TOP_ROW + cap : SLOT_TOP_ROW + SLOT_LEN, col] = int(color)


def _rank_proj_out_cols(slot_start: int, heights: list[int]) -> list[int]:
    pairs = [(int(height), int(slot_start + local_c)) for local_c, height in zip(_bar_local_proj_cols(len(heights)), heights)]
    pairs.sort(key=lambda t: (-t[0], t[1]))
    return [int(col) for _height, col in pairs]


def _rank_slot_src_cols(slot_cols: list[int], counts: list[int]) -> list[int]:
    pairs = [(int(count), int(col)) for col, count in zip(slot_cols, counts)]
    pairs.sort(key=lambda t: (-t[0], t[1]))
    return [int(col) for _count, col in pairs]


def _build_input(slot_start: int, bar_input_c0: int, heights: list[int], colors: list[int], counts: list[int]) -> np.ndarray:
    grid = np.zeros((H, W), dtype=int)
    paste(grid, _frame_patch(), FRAME_R0, FRAME_C0)
    paste(grid, _bar_patch(heights), BAR_R0, bar_input_c0)
    for col, color, count in zip(_slot_cols(slot_start, len(heights)), colors, counts):
        _paint_input_slot_column(grid, int(col), int(color), int(count))
    return grid


def _build_expected(inp: np.ndarray, slot_start: int, heights: list[int], colors: list[int], counts: list[int]) -> np.ndarray:
    out = np.full((H, W), BG, dtype=int)
    out[FRAME_R0 : FRAME_R0 + FRAME_H, FRAME_C0 : FRAME_C0 + FRAME_W] = inp[
        FRAME_R0 : FRAME_R0 + FRAME_H, FRAME_C0 : FRAME_C0 + FRAME_W
    ]
    slot_region = out[FRAME_R0 : FRAME_R0 + FRAME_H, FRAME_C0 : FRAME_C0 + FRAME_W]
    slot_region[(slot_region != BG) & (slot_region != GREY)] = BG

    paste(out, _bar_patch(heights), BAR_R0, slot_start, transparent=BG)

    proj_rank = sorted(
        [(int(height), int(slot_start + local_c)) for local_c, height in zip(_bar_local_proj_cols(len(heights)), heights)],
        key=lambda t: (-t[0], t[1]),
    )
    slot_rank = sorted(
        [(int(count), int(col), int(color)) for col, count, color in zip(_slot_cols(slot_start, len(heights)), counts, colors)],
        key=lambda t: (-t[0], t[1]),
    )

    for (height, out_col), (count, _src_col, color) in zip(proj_rank, slot_rank):
        desired_bottom = BAR_BASE_ABS_ROW - int(height)
        visible = min(int(count), int(desired_bottom - SLOT_TOP_ROW + 1))
        if visible <= 0:
            continue
        start_row = desired_bottom - visible + 1
        out[start_row : desired_bottom + 1, int(out_col)] = int(color)

    return out


def _sample_params(
    rng: np.random.Generator,
    n_projs: int,
) -> tuple[int, int, list[int], list[int], list[int]]:
    slot_start = int(rng.choice(_slot_start_choices(n_projs)))
    bar_input_c0 = int(rng.choice(_bar_input_c0_choices(n_projs)))
    heights = [int(v) for v in rng.permutation(np.arange(1, n_projs + 1))]
    counts = [int(v) for v in rng.permutation(np.arange(1, n_projs + 1))]
    while not _variant_visible(slot_start, heights, counts):
        counts = [int(v) for v in rng.permutation(np.arange(1, n_projs + 1))]
    colors = [int(v) for v in rng.choice(PALETTE, size=n_projs, replace=False)]
    return slot_start, bar_input_c0, heights, colors, counts


def _variant_visible(slot_start: int, heights: list[int], counts: list[int]) -> bool:
    slot_cols = _slot_cols(slot_start, len(heights))
    return _rank_proj_out_cols(slot_start, heights) != _rank_slot_src_cols(slot_cols, counts)


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    n_projs = int(rng.integers(MIN_PROJS, MAX_PROJS + 1))
    slot_start, bar_input_c0, heights, colors, counts = _sample_params(rng, n_projs)

    inp = _build_input(slot_start, bar_input_c0, heights, colors, counts)
    expected = _build_expected(inp, slot_start, heights, colors, counts)

    if n_projs != len(_slot_cols(slot_start, len(heights))):
        return None
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
