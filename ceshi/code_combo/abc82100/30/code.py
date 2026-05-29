# Combo variant of abc82100 (index 30)
# combo_spec: D2/0+D9/0+D10/1
# Name: 图例区域编码+单一规则+边缘分布
#
# Design analysis
# a) 具体化与行为追踪
#    - 原题在全图中散布若干“source -> target + teal 模板”局部示例，solver 需要逐处搜规则，再按 source 色分别盖章。
#    - 融合后只保留一条规则，并把它集中编码到顶部边缘的一条图例项里。图例项仍是 3x6 小块:
#        * 中间行左两格是一个示例 source 色和唯一 target 色。
#        * 右侧 3x4 小窗是唯一的 teal 模板，固定锚点在局部 (1,1)。
#      这个图例项整块贴在顶边，因此“规则区在边缘、工作区在中央”的分区非常明显。
#    - 工作区中央只放孤立的彩色单像素，而且至少出现两种不同颜色；它们全部都触发同一模板和同一 target 色。
#    - 小例子追踪:
#        * 原逻辑: 红旁边黄，再沿方向接到某个 teal 形状，得到“红 -> 黄 + 该形状”；蓝要靠另一条示例学到。
#        * 融合逻辑: 顶边只给出一次“某示例源色 -> 统一 target 色 + 模板”，中央却同时出现红、蓝、灰等孤立像素；
#          输出里这些不同颜色的点全部都扩成同一个 target 色模板副本。
#        * 差异: 颜色不再决定选哪条规则；solver 先识别边缘图例，再领会“任意孤立彩色源都用同一条规则”。
# b) 复杂度校准
#    - 若把中央源像素全做成同色，题目会退化成“找唯一 source 色再盖章”，看不出 D9/0 的变化。
#      因此生成器强制中央至少 2 种 source 颜色，并让图例示例色只是其中之一。
#    - 若允许多个模板或多个图例项，又会退回普通 legend lookup；因此这里只保留一个模板、一个 target 色、一个图例项。
#    - 模板仍保留 3x4 二维形状而不是一维编码，避免视觉上过于贫乏，也保留“平移盖章”的核心难点。
# c) 隐式假设审查
#    - 原 `_discover_rules()` 依赖“source/target 对 + 邻近 teal”散布在全图任意位置；融合后这个前提完全失效。
#      新 solver 直接通过唯一 teal 连通块定位顶部图例窗口，再反推左侧颜色对。
#    - 原 `main()` 按 source 颜色分组应用不同规则；融合后 source 颜色本身不再决定规则。
#      新逻辑改成: 图例外所有孤立的非背景非 teal 像素都视为源，统一使用唯一模板/target。
#    - 新 solver 的关键前提:
#        * 唯一 teal 连通块位于顶边图例中，且其 bbox 恰好就是 3x4 模板窗。
#        * 模板包含固定锚点，并触及模板窗四边，这样 teal bbox 能唯一锁定窗口位置。
#        * 工作区源像素彼此 4 邻隔离，否则“所有孤立彩点都是源”会变得含混。
#      generator 会显式构造并验证这些前提。
# d) 放置合法性
#    - 图例固定贴顶边；下方保留至少一行空白，再把源像素限制在中央矩形，避免图例区和工作区混杂。
#    - 源像素彼此不重叠、不过 4 邻，并且其输出盖章足迹互不重叠，保证输入输出都清爽可读。
#    - 模板完整落在网格内，中央源的位置也会保证完整盖章留在网格内；这比原题更整洁，也更突出“中央复制”。
# e) 视觉可观察性
#    - 人类会先看到顶边唯一一条“颜色对 + teal 模板”的图例，再看到中央多种颜色的孤立点。
#    - 输出中这些不同颜色的点统一消失并变成同色同形的模板副本，足以从少量样例推出“source 色无关，只有是否为孤立源有关”。
# f) 可视化整洁度
#    - 输入只有两层语义区域: 顶边图例、中央散点；没有原题那种多个局部示例和多个 teal 簇混在场景中的噪声。
#    - 输出只留下统一颜色的多个模板副本，避免不同规则互相覆盖造成的阅读负担。
# g) 变化的可推断性
#    - 颜色变化被强约束成“所有中央孤立彩点 -> 同一个 target 色”，并由边缘图例显式给出 target 色与模板。
#    - 由于每题至少出现两种 source 色，解题者能从样例直接观察到: 决定变换的不是源颜色身份，而是它属于中央孤立源集合。
# h) 对新 `main()` 假设的逐项回查
#    - teal bbox 解码假设: 需要唯一 teal 连通块、且 bbox=3x4。生成器只在图例放 teal，并选触四边的连通模板。
#    - 图例左右相对位置假设: source/target 位于模板窗左侧两格。生成器固定按该格式绘制图例。
#    - “孤立源”判定假设: 工作区彩点没有 4 邻彩点。生成器用 reserved 邻域强制隔离。
#    - “中央分布”假设: 这些源不靠边、不贴图例。生成器只在中央 bbox 内采样，并要求至少一行分隔。

from __future__ import annotations

import random
from functools import lru_cache
from itertools import combinations

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, label_mask


PATTERN_COLOR = 8
NON_BG_NON_TEAL_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]

GRID_DIM_MAX = 20
GRID_HEIGHT_RANGE = (11, 20)
GRID_WIDTH_RANGE = (12, 20)

LEGEND_HEIGHT = 3
LEGEND_WIDTH = 6
LEGEND_SEPARATOR = 1
TEMPLATE_H = 3
TEMPLATE_W = 4
TEMPLATE_COL0 = 2
ANCHOR_LOCAL = (1, 1)

STENCIL_AREA_RANGE = (4, 7)
SCATTERED_COUNT_RANGE = (4, 7)
SCATTERED_COLOR_COUNT_RANGE = (2, 4)
SCATTERED_RETRIES_PER_PIXEL = 120
OUTER_RETRIES = 300

_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _reserve_4neighbors(reserved: set[tuple[int, int]], cells: list[tuple[int, int]]) -> None:
    for r, c in cells:
        for dr, dc in _DIRS4:
            reserved.add((int(r) + dr, int(c) + dc))


def _stamp_cells(sr: int, sc: int, rel: np.ndarray, shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = map(int, shape)
    cells: set[tuple[int, int]] = set()
    for rr, cc in np.asarray(rel, dtype=int):
        nr = int(sr) + int(rr)
        nc = int(sc) + int(cc)
        if 0 <= nr < h and 0 <= nc < w:
            cells.add((nr, nc))
    return cells


@lru_cache(maxsize=None)
def _valid_stencil_masks() -> tuple[np.ndarray, ...]:
    """Connected 3x4 masks that include the anchor and touch all four sides."""
    masks: list[np.ndarray] = []
    anchor_index = ANCHOR_LOCAL[0] * TEMPLATE_W + ANCHOR_LOCAL[1]
    for area in range(STENCIL_AREA_RANGE[0], STENCIL_AREA_RANGE[1] + 1):
        for cells in combinations(range(TEMPLATE_H * TEMPLATE_W), area):
            if anchor_index not in cells:
                continue
            mask = np.zeros((TEMPLATE_H, TEMPLATE_W), dtype=bool)
            for cell in cells:
                mask[cell // TEMPLATE_W, cell % TEMPLATE_W] = True
            spans_edges = (
                mask[0, :].any()
                and mask[-1, :].any()
                and mask[:, 0].any()
                and mask[:, -1].any()
            )
            if not spans_edges:
                continue
            _labels, count = label_mask(mask, connectivity=8)
            if int(count) == 1:
                masks.append(mask)
    return tuple(masks)


def _central_bounds(shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = map(int, shape)
    row_lo = max(LEGEND_HEIGHT + LEGEND_SEPARATOR + 1, h // 3)
    row_hi = min(h - 2, h - max(2, h // 3) - 1)
    col_lo = max(2, w // 4)
    col_hi = min(w - 3, w - max(2, w // 4) - 1)
    return int(row_lo), int(row_hi), int(col_lo), int(col_hi)


def _legend_patch(rule: dict) -> np.ndarray:
    legend = np.zeros((LEGEND_HEIGHT, LEGEND_WIDTH), dtype=int)
    legend[1, 0] = int(rule["example_source"])
    legend[1, 1] = int(rule["target"])
    window = legend[:, TEMPLATE_COL0 : TEMPLATE_COL0 + TEMPLATE_W]
    window[np.asarray(rule["mask"], dtype=bool)] = PATTERN_COLOR
    return legend


def _decode_single_legend(grid: np.ndarray, bg: int, pattern_color: int = PATTERN_COLOR) -> tuple[dict, tuple[int, int, int, int]]:
    teal_components = components_mask(grid == int(pattern_color), connectivity=8)
    if len(teal_components) != 1:
        raise ValueError("expected exactly one teal legend component")

    pattern = np.asarray(teal_components[0], dtype=int)
    r0, c0, r1, c1 = map(int, bbox_coords(pattern))
    if r0 != 0 or (r1 - r0 + 1) != TEMPLATE_H or (c1 - c0 + 1) != TEMPLATE_W:
        raise ValueError("teal component must be a top-edge 3x4 legend window")

    anchor = (r0 + ANCHOR_LOCAL[0], c0 + ANCHOR_LOCAL[1])
    legend_left = c0 - TEMPLATE_COL0
    legend_right = legend_left + LEGEND_WIDTH - 1
    if legend_left < 0 or legend_right >= grid.shape[1]:
        raise ValueError("legend color pair would be out of bounds")

    src = int(grid[anchor[0], legend_left])
    tgt = int(grid[anchor[0], legend_left + 1])
    if src in (int(bg), int(pattern_color)) or tgt in (int(bg), int(pattern_color)) or src == tgt:
        raise ValueError("invalid legend color pair")

    rel = pattern - np.array(anchor, dtype=int)
    return (
        {
            "example_source": src,
            "target_color": tgt,
            "rel": np.asarray(rel, dtype=int),
        },
        (0, legend_left, LEGEND_HEIGHT - 1, legend_right),
    )


def _is_isolated_colored_pixel(
    grid: np.ndarray,
    r: int,
    c: int,
    legend_rect: tuple[int, int, int, int],
    bg: int,
    pattern_color: int = PATTERN_COLOR,
) -> bool:
    r0, c0, r1, c1 = map(int, legend_rect)
    if r0 <= int(r) <= r1 and c0 <= int(c) <= c1:
        return False
    color = int(grid[int(r), int(c)])
    if color in (int(bg), int(pattern_color)):
        return False
    h, w = grid.shape
    for dr, dc in _DIRS4:
        nr = int(r) + dr
        nc = int(c) + dc
        if 0 <= nr < h and 0 <= nc < w:
            if int(grid[nr, nc]) not in (int(bg), int(pattern_color)):
                return False
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read a single edge legend and expand every isolated central source pixel with one shared stencil.

    Input structure:
    - Black background with exactly one teal template window embedded in a top-edge legend block.
    - The legend block has one row-entry: a sample source color, one target color, and a 3x4 teal stencil.
    - Away from the edge, the central field contains isolated colored single pixels, often in multiple colors.

    Rule:
    - Decode the top legend into one template and one target color.
    - Treat every isolated non-background, non-teal pixel outside the legend as a source, regardless of its own color.
    - Translate the shared stencil so its fixed anchor lands on each source pixel, recolor the copied shape to the legend target color, and erase the legend and source pixels.

    Key insight: The edge legend is not a color-by-color lookup table here; it is a single exemplar describing one reusable action, and the real grouping cue is spatial/isolation-based rather than source-color identity.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    h, w = g.shape

    rule, legend_rect = _decode_single_legend(g, bg=bg, pattern_color=PATTERN_COLOR)
    rel = np.asarray(rule["rel"], dtype=int)

    out = np.full_like(g, int(bg))

    for r, c in map(tuple, np.argwhere((g != int(bg)) & (g != int(PATTERN_COLOR)))):
        r = int(r)
        c = int(c)
        if _is_isolated_colored_pixel(g, r, c, legend_rect, bg=bg, pattern_color=PATTERN_COLOR):
            rows = r + rel[:, 0]
            cols = c + rel[:, 1]
            ok = (0 <= rows) & (rows < h) & (0 <= cols) & (cols < w)
            out[rows[ok], cols[ok]] = int(rule["target_color"])
        else:
            lr0, lc0, lr1, lc1 = map(int, legend_rect)
            if not (lr0 <= r <= lr1 and lc0 <= c <= lc1):
                out[r, c] = int(g[r, c])

    return out


def _build_rule() -> dict:
    colors = NON_BG_NON_TEAL_COLORS[:]
    random.shuffle(colors)
    example_source = int(colors[0])
    target = int(colors[1])

    max_sc_colors = min(SCATTERED_COLOR_COUNT_RANGE[1], len(colors) - 1)
    n_sc_colors = random.randint(SCATTERED_COLOR_COUNT_RANGE[0], max_sc_colors)
    extra_sources = random.sample([c for c in colors[2:] if c != target], n_sc_colors - 1)
    scattered_colors = [example_source] + [int(c) for c in extra_sources]
    random.shuffle(scattered_colors)

    mask = random.choice(_valid_stencil_masks())
    rel = np.argwhere(mask).astype(int) - np.array(ANCHOR_LOCAL, dtype=int)
    return {
        "example_source": example_source,
        "target": target,
        "mask": mask.copy(),
        "rel": np.asarray(rel, dtype=int),
        "scattered_colors": [int(c) for c in scattered_colors],
    }


def _place_scattered(rule: dict, shape: tuple[int, int]) -> list[tuple[int, int, int]] | None:
    h, w = map(int, shape)
    row_lo, row_hi, col_lo, col_hi = _central_bounds(shape)
    if row_lo > row_hi or col_lo > col_hi:
        return None

    rel = np.asarray(rule["rel"], dtype=int)
    rr_min = int(rel[:, 0].min())
    rr_max = int(rel[:, 0].max())
    cc_min = int(rel[:, 1].min())
    cc_max = int(rel[:, 1].max())
    row_lo = max(row_lo, -rr_min)
    row_hi = min(row_hi, h - 1 - rr_max)
    col_lo = max(col_lo, -cc_min)
    col_hi = min(col_hi, w - 1 - cc_max)
    if row_lo > row_hi or col_lo > col_hi:
        return None

    max_count = min(
        SCATTERED_COUNT_RANGE[1],
        max(SCATTERED_COUNT_RANGE[0], ((row_hi - row_lo + 1) * (col_hi - col_lo + 1)) // 4),
    )
    total_sources = random.randint(SCATTERED_COUNT_RANGE[0], max_count)

    base_colors = list(rule["scattered_colors"])
    assigned_colors = base_colors[:]
    for _ in range(total_sources - len(base_colors)):
        assigned_colors.append(int(random.choice(base_colors)))
    random.shuffle(assigned_colors)

    occupied: set[tuple[int, int]] = set()
    reserved: set[tuple[int, int]] = set()
    footprints: set[tuple[int, int]] = set()
    placed: list[tuple[int, int, int]] = []

    for color in assigned_colors:
        success = False
        for _ in range(SCATTERED_RETRIES_PER_PIXEL):
            r = random.randint(row_lo, row_hi)
            c = random.randint(col_lo, col_hi)
            if (r, c) in occupied or (r, c) in reserved:
                continue
            stamp = _stamp_cells(r, c, rel, shape)
            if stamp & footprints:
                continue
            placed.append((int(r), int(c), int(color)))
            occupied.add((int(r), int(c)))
            _reserve_4neighbors(reserved, [(int(r), int(c))])
            footprints |= stamp
            success = True
            break
        if not success:
            return None

    return placed


def _build_input(rule: dict, scattered: list[tuple[int, int, int]], shape: tuple[int, int]) -> np.ndarray:
    inp = np.zeros(shape, dtype=int)
    legend = _legend_patch(rule)
    legend_left = random.randint(0, shape[1] - LEGEND_WIDTH)
    inp[:LEGEND_HEIGHT, legend_left : legend_left + LEGEND_WIDTH] = legend
    rule["legend_left"] = int(legend_left)

    for r, c, color in scattered:
        inp[int(r), int(c)] = int(color)
    return inp


def _build_output(rule: dict, scattered: list[tuple[int, int, int]], shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    rel = np.asarray(rule["rel"], dtype=int)
    for r, c, _color in scattered:
        for nr, nc in _stamp_cells(int(r), int(c), rel, shape):
            out[int(nr), int(nc)] = int(rule["target"])
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_HEIGHT_RANGE)
    w = random.randint(*GRID_WIDTH_RANGE)
    shape = (h, w)

    rule = _build_rule()
    scattered = _place_scattered(rule, shape)
    if scattered is None:
        return None

    inp = _build_input(rule, scattered, shape)
    expected = _build_output(rule, scattered, shape)

    distinct_colors = {int(color) for _r, _c, color in scattered}
    if len(distinct_colors) < 2:
        return None

    try:
        if np.array_equal(main(inp), expected):
            return inp
    except Exception:
        pass
    return None


def generate_input() -> np.ndarray:
    for _ in range(OUTER_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {OUTER_RETRIES} attempts")
