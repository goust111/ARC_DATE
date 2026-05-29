# Combo variant of d59b0160 (index 25)
# combo_spec: D5/3+D18/0
# Name: 颜色频率映射+颜色对应关系

import random
from collections import Counter

import numpy as np

from arc_utils import components_mask, paste


# Design analysis for the combo variant
# =====================================
# a) 具体化与行为追踪
#    - 保留原题的大框架: 16x16 网格、背景 7、左上角 4x4 图例、其余位置放若干 4-连通组件。
#    - 但图例的 3x3 内区不再表示“待收集的颜色集合”。现在固定为:
#        row 0: 3 个图例 token
#        row 1: 与上面同列对应的 3 个图例 token
#        row 2: 保持背景 7
#      同列两格仍定义 3 组配对关系, 这是 D5/3 的部分。
#    - 同时这些 token 也不再直接拿去和组件中的颜色 id 比较。对每个组件, 先统计其内部所有非背景、
#      非脚手架色 0 的彩色像素频率, 按从高到低得到 6 个“频率槽位”。图例 row-major 的第 1 个 token
#      对应最高频颜色, 第 2 个 token 对应第二高频颜色, ..., 第 6 个 token 对应第六高频颜色。
#      这是 D18/0 的部分。
#    - 两者叠加后的最终判定是:
#        先做 token -> 频率排序后的真实组件颜色 的映射,
#        再按图例列配对检查组件里是否存在对应颜色对的 4 邻接接触。
#      只有 3 组映射后的颜色对都出现, 该组件才删除。
#    - 小例子:
#      图例若是
#        [9 1 6]
#        [2 8 4]
#      则三列 token 对分别是 (9,2), (1,8), (6,4), 但这只是“槽位配对说明”。
#      某组件里 6 种彩色像素的频率从高到低若是:
#        red > teal > yellow > blue > maroon > grey
#      那么 9->red, 1->teal, 6->yellow, 2->blue, 8->maroon, 4->grey,
#      实际需要检查的相邻对变成:
#        (red, blue), (teal, maroon), (yellow, grey)。
#      原题会直接看图例颜色是否都出现; D5 单变体会直接看 (9,2)/(1,8)/(6,4) 是否相邻;
#      融合后必须先“频率翻译”再检查相邻对。
#
# b) 复杂度校准
#    - 图例 token 固定 6 个、配对固定 3 组, 频率槽位也固定 6 档, 自由度受控。
#    - 每个组件都恰好含 6 种彩色标记, 且频率严格为 6/5/4/3/2/1, 没有平票, 人眼可从重复样例中
#      观察到“最长/最多的那种色块先对应图例第一个 token, 其次对应第二个 token ...”。
#    - 组件内部还固定出现 3 组接触簇, 因此样例中既反复支撑“要数频率”, 也反复支撑“要看配对接触”。
#    - generator 强制至少有 1 个组件满足融合规则但不满足“按图例原色直接配对”的旧思路,
#      防止退化成只看颜色值的 D5 单变体。
#
# c) 隐式假设审查
#    - 原 solver 的 `_find_legend()` 把图例内区当作无序集合; 现在位置是核心信息, 必须按前两行读取。
#    - 原 solver 遍历 `g != bg` 的所有连通分量。现在图例前两行本身含 4 邻接 token, 若不排除 `[:sz, :sz]`
#      整个图例区, 可能把图例误判成待删组件, 所以要显式跳过图例方块。
#    - 原 solver 用“颜色出现过”就够了; 现在必须额外假设:
#      1) 组件中存在至少 6 种非 0 彩色像素, 否则无法完成 6 个 token 的映射。
#      2) 这 6 种颜色的频率在前 6 名上没有平票, 否则“第 k 高频”不唯一。
#      `main()` 遇到不满足这两个条件的组件时一律保留。generator 则构造性地保证条件成立。
#    - 由于映射完全基于可见频率, 不能把脚手架色 0 算进排序, 否则会退化成“黑底永远第一”这种
#      人类不可视的捷径。solver 明确只统计 `not in {bg, 0}` 的颜色。
#
# d) 放置合法性
#    - 不再随机搜尺寸不一的矩形, 改为固定三块 6x6 组件, 放在 top-right / bottom-left / bottom-right
#      三个互不接触的锚点, 与图例和彼此之间至少隔一圈背景。
#    - 每个 6x6 patch 先用 0 填满形成一个整体连通矩形, 再在预留槽位上放 6 个彩色簇。
#      因此组件始终是单个 4-连通分量, 且不会与其他组件粘连或越界。
#    - 彩色簇的槽位图是固定的, 不同簇之间只在设计好的 3 处发生接触, 所以不会意外生成额外颜色对。
#
# e) 视觉可观察性
#    - 图例中“第一行/第二行同列配对”是直接可见的。
#    - 组件内部 6 种彩色簇大小严格递减, 人眼可以比较哪种颜色最多、第二多、第三多。
#    - 输入输出对里被删除的组件不是“含图例原色配对”就删, 而是“最大色对应第 1 个 token、第四大色
#      对应第 4 个 token ...”后形成的 3 对都接触才删。多组样例足以暴露这个频率映射。
#
# f) 可视化整洁度
#    - 每个组件都是规整的 6x6 黑色矩形底板, 上面覆盖 6 个紧凑彩色簇; 图例单独放在左上角。
#    - 三个组件固定落在三个象限区, 没有对象重叠, 没有高密度散点噪声, 画面比随机撒点更整洁。
#    - 组件内部的三对接触簇彼此隔开, 解题者容易局部检查“哪两种颜色真的贴在一起”。
#
# g) 变化的可推断性
#    - 颜色变化的关键不是颜色 id 本身, 而是“在该组件里排第几大”。这通过 6/5/4/3/2/1 的重复尺寸
#      结构提供了稳定线索。
#    - 图例 token 使用 6 个互异颜色, 只是为了让 6 个槽位可区分; 真正决定匹配的是 row-major 次序
#      与组件中的频率排序共同诱导出的映射。
#    - 为防止样例退化, generator 会拒绝那些“融合规则删除集合”和“按图例原色直接配对删除集合”
#      完全相同的构造。


GRID_H, GRID_W = 16, 16
BG = 7
BORDER = 3
SCAFFOLD = 0
LEGEND_SIZE = 4
PALETTE = [1, 2, 4, 5, 6, 8, 9]
NUM_PAIRS = 3
NUM_LEGEND_TOKENS = 2 * NUM_PAIRS
PATCH_H, PATCH_W = 6, 6
PATCH_ANCHORS = [(1, 9), (9, 1), (9, 9)]
MAX_RETRIES = 400


SLOT_ORDER = ["s6", "s5", "s4", "s3", "s2", "s1"]
ROLE_SLOT_CELLS = {
    "fusion": {
        "s6": [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)],
        "s5": [(2, 3), (2, 4), (3, 3), (3, 4), (3, 5)],
        "s4": [(3, 0), (3, 1), (4, 0), (4, 1)],
        "s3": [(0, 3), (0, 4), (0, 5)],
        "s2": [(4, 3), (5, 3)],
        "s1": [(5, 0)],
    },
    "wrong": {
        "s6": [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)],
        "s5": [(2, 3), (2, 4), (3, 3), (3, 4), (3, 5)],
        "s4": [(3, 0), (3, 1), (4, 0), (4, 1)],
        "s3": [(5, 0), (5, 1), (5, 2)],
        "s2": [(0, 3), (0, 4)],
        # Move s1 away from s4 so this layout fails one required pair.
        "s1": [(5, 5)],
    },
    "mixed": {
        "s6": [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)],
        "s5": [(2, 3), (2, 4), (3, 3), (3, 4), (3, 5)],
        "s4": [(3, 0), (3, 1), (4, 0), (4, 1)],
        "s3": [(0, 3), (0, 4), (0, 5)],
        # Move s2 away from s5 so this layout fails a different required pair.
        "s2": [(5, 4), (5, 5)],
        "s1": [(2, 5)],
    },
}


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color along the 1-cell outer border."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _find_legend(g: np.ndarray, bg: int) -> tuple[int, int, list[int], list[int]]:
    """
    Detect the top-left legend framed by a uniform bottom edge and right edge.

    The first two interior rows encode 3 column-wise token pairs.
    """
    h, w = g.shape
    for sz in range(3, min(h, w) + 1):
        x = int(g[sz - 1, sz - 1])
        if x == bg:
            continue
        if np.all(g[sz - 1, 0:sz] == x) and np.all(g[0:sz, sz - 1] == x):
            inside = g[0 : sz - 1, 0 : sz - 1]
            if inside.shape != (3, 3):
                continue
            top = [int(v) for v in inside[0, :]]
            bottom = [int(v) for v in inside[1, :]]
            if any(v in (bg, x) for v in top + bottom):
                continue
            if len(set(top + bottom)) != NUM_LEGEND_TOKENS:
                continue
            return int(sz), int(x), top, bottom
    assert False, "no compatible legend found in grid"


def _component_has_pair(g: np.ndarray, pts_set: set[tuple[int, int]], a: int, b: int) -> bool:
    """Return True iff this component contains a 4-neighbor adjacency between colors a and b."""
    for r, c in pts_set:
        if int(g[r, c]) != a:
            continue
        if (r - 1, c) in pts_set and int(g[r - 1, c]) == b:
            return True
        if (r + 1, c) in pts_set and int(g[r + 1, c]) == b:
            return True
        if (r, c - 1) in pts_set and int(g[r, c - 1]) == b:
            return True
        if (r, c + 1) in pts_set and int(g[r, c + 1]) == b:
            return True
    return False


def _ranked_component_colors(g: np.ndarray, pts: np.ndarray, bg: int) -> list[int] | None:
    """
    Rank the component's visible accent colors by frequency.

    Scaffold color 0 is excluded from ranking on purpose: the combo rule is meant to be
    inferable from visible accent-color multiplicities, not from the black rectangle base.
    """
    counts: Counter[int] = Counter()
    for r, c in pts:
        v = int(g[r, c])
        if v in (bg, SCAFFOLD):
            continue
        counts[v] += 1

    if len(counts) < NUM_LEGEND_TOKENS:
        return None

    ranked = counts.most_common(NUM_LEGEND_TOKENS)
    if len(ranked) < NUM_LEGEND_TOKENS:
        return None

    freqs = [cnt for _col, cnt in ranked]
    if len(set(freqs)) != NUM_LEGEND_TOKENS:
        return None
    if freqs != sorted(freqs, reverse=True):
        return None

    return [int(col) for col, _cnt in ranked]


def _mapped_pairs_for_component(
    g: np.ndarray,
    pts: np.ndarray,
    legend_top: list[int],
    legend_bottom: list[int],
    bg: int,
) -> list[tuple[int, int]] | None:
    """Map legend tokens to this component's frequency-ranked colors, then return the 3 required color pairs."""
    ranked_colors = _ranked_component_colors(g, pts, bg)
    if ranked_colors is None:
        return None

    row_major_tokens = legend_top + legend_bottom
    token_to_color = {token: ranked_colors[idx] for idx, token in enumerate(row_major_tokens)}
    return [(token_to_color[src], token_to_color[dst]) for src, dst in zip(legend_top, legend_bottom)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Delete connected components whose frequency-ranked colors realize every legend pair.

    Input structure:
    - Background (bg) fills most of the grid.
    - Top-left corner: a small square legend bordered on its bottom and right edges by one color.
    - Inside the legend, the first two interior rows contain 6 distinct token colors; same-column entries form 3 token pairs.
    - Elsewhere: 4-connected components on the bg, each with a black scaffold and 6 colored accent blobs of different sizes.

    Rule:
    - Read the 3 legend token pairs column-wise, but do not compare those token colors directly to component colors.
    - For each non-legend component, rank its non-bg, non-black accent colors by frequency from most common to least common.
    - Map the legend's row-major token order onto that ranked list, then convert each legend column into a real component-color pair.
    - Erase (fill with bg) every component whose interior contains all 3 mapped pairs as 4-neighbor color contacts; leave all others unchanged.

    Key insight: The legend specifies pair structure, but the actual colors to test are obtained indirectly from each component's own frequency ordering.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)

    sz, _border, legend_top, legend_bottom = _find_legend(g, bg)

    out = g.copy()
    obj_mask = (g != bg).copy()
    obj_mask[:sz, :sz] = False

    comps = components_mask(obj_mask, connectivity=4)
    for pts in comps:
        mapped_pairs = _mapped_pairs_for_component(g, pts, legend_top, legend_bottom, bg)
        if mapped_pairs is None:
            continue
        pts_set = set(map(tuple, pts))
        if all(_component_has_pair(g, pts_set, a, b) or _component_has_pair(g, pts_set, b, a)
               for a, b in mapped_pairs):
            for r, c in pts:
                out[r, c] = bg
    return out


def _place_legend(inp: np.ndarray, out: np.ndarray, legend_top: list[int], legend_bottom: list[int]) -> None:
    """Draw the 4x4 legend box and encode 3 column-wise token pairs."""
    s = LEGEND_SIZE
    for g in (inp, out):
        g[:s, :s] = BG
        g[s - 1, :s] = BORDER
        g[:s, s - 1] = BORDER
        g[0, :NUM_PAIRS] = legend_top
        g[1, :NUM_PAIRS] = legend_bottom


def _build_patch(pattern: str, rank_colors: list[int]) -> np.ndarray:
    """Build one 6x6 component patch for the requested pair-layout pattern."""
    patch = np.full((PATCH_H, PATCH_W), SCAFFOLD, dtype=int)
    slot_cells = ROLE_SLOT_CELLS[pattern]
    for idx, slot in enumerate(SLOT_ORDER):
        color = int(rank_colors[idx])
        cells = slot_cells[slot]
        for r, c in cells:
            patch[r, c] = color
    return patch


def _direct_identity_pairs(legend_top: list[int], legend_bottom: list[int]) -> set[tuple[int, int]]:
    """Concrete legend-color pairs used by the direct D5-style interpretation."""
    return {tuple(sorted((int(a), int(b)))) for a, b in zip(legend_top, legend_bottom)}


def _fusion_pairs_from_rank_colors(rank_colors: list[int]) -> set[tuple[int, int]]:
    """Concrete color pairs implied by the combo rule once rank slots are instantiated."""
    return {
        tuple(sorted((int(rank_colors[0]), int(rank_colors[3])))),
        tuple(sorted((int(rank_colors[1]), int(rank_colors[4])))),
        tuple(sorted((int(rank_colors[2]), int(rank_colors[5])))),
    }


def _identity_pair_deletion_mask(
    grid: np.ndarray,
    legend_top: list[int],
    legend_bottom: list[int],
    bg: int,
) -> tuple[bool, ...]:
    """Baseline: which components would be deleted if one skipped the frequency mapping and used direct legend-color pairs."""
    want = _direct_identity_pairs(legend_top, legend_bottom)
    obj_mask = (grid != bg).copy()
    obj_mask[:LEGEND_SIZE, :LEGEND_SIZE] = False
    mask: list[bool] = []
    for pts in components_mask(obj_mask, connectivity=4):
        pts_set = set(map(tuple, pts))
        seen: set[tuple[int, int]] = set()
        for r, c in pts_set:
            col = int(grid[r, c])
            for nr, nc in ((r + 1, c), (r, c + 1)):
                if (nr, nc) in pts_set:
                    pair = tuple(sorted((col, int(grid[nr, nc]))))
                    if pair[0] != pair[1]:
                        seen.add(pair)
        mask.append(want.issubset(seen))
    return tuple(mask)


def _sample_rank_colors(legend_top: list[int], legend_bottom: list[int]) -> list[int]:
    """
    Choose a rank permutation that is visibly different from the direct legend-color interpretation.

    We reject the row-major legend order and any permutation whose fusion pair set coincides with the
    direct identity-pair set, because such samples would hide the D18 contribution.
    """
    row_major = legend_top + legend_bottom
    direct = _direct_identity_pairs(legend_top, legend_bottom)
    for _ in range(100):
        rank_colors = random.sample(row_major, len(row_major))
        if rank_colors == row_major:
            continue
        if _fusion_pairs_from_rank_colors(rank_colors) == direct:
            continue
        return rank_colors
    raise RuntimeError("failed to sample a non-degenerate rank permutation")


def _gen_attempt() -> np.ndarray | None:
    """Construct one valid input grid for the combo rule."""
    inp = np.full((GRID_H, GRID_W), BG, dtype=int)
    out = inp.copy()

    legend_tokens = random.sample(PALETTE, NUM_LEGEND_TOKENS)
    legend_top = legend_tokens[:NUM_PAIRS]
    legend_bottom = legend_tokens[NUM_PAIRS:]
    _place_legend(inp, out, legend_top, legend_bottom)

    rank_orders = [_sample_rank_colors(legend_top, legend_bottom) for _ in range(3)]
    patterns = ["fusion", "wrong", "mixed"]
    components = [_build_patch(pattern, rank_orders[idx]) for idx, pattern in enumerate(patterns)]

    placements = list(zip(PATCH_ANCHORS, components, patterns))
    random.shuffle(placements)
    for (r, c), patch, pattern in placements:
        paste(inp, patch, r, c)
        if pattern != "fusion":
            paste(out, patch, r, c)

    try:
        combo_out = main(inp)
    except Exception:
        return None
    if not np.array_equal(combo_out, out):
        return None

    # Non-degeneracy check: the fused rule must disagree with direct color-pair matching.
    bg = _bg_boundary(inp)
    combo_deleted = tuple(
        not np.array_equal(inp[r : r + PATCH_H, c : c + PATCH_W], combo_out[r : r + PATCH_H, c : c + PATCH_W])
        for r, c in PATCH_ANCHORS
    )
    direct_deleted = _identity_pair_deletion_mask(inp, legend_top, legend_bottom, bg)
    if any(direct_deleted):
        return None
    if combo_deleted == direct_deleted:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
