# Combo variant of d59b0160 (index 41)
# combo_spec: D3/3+D5/3
# Name: 嵌套矩形框架图例+颜色对应关系

import math
import random

import numpy as np

from arc_utils import components_mask, paste, random_free_location


# Design analysis for the combo variant
# =====================================
# a) 具体化与行为追踪
#    - 保留原题的大框架: 16x16, 背景 7, 左上角是图例, 其余区域放若干由颜色 0 连成的矩形组件.
#    - 但图例不再是 4x4 的 L 形边框. 现在左上角是一个 7x7 的嵌套方框:
#        depth 0: 最外层 7x7 方框颜色 a
#        depth 1: 里面 5x5 方框颜色 b
#        depth 2: 再里面 3x3 方框颜色 c
#        depth 3: 中心 1x1 颜色 d
#      从外到内得到颜色链 [a,b,c,d].
#    - D3/3 的层级信息和 D5/3 的配对信息在这里合并:
#      相邻层自动形成 3 个配对: (a,b), (b,c), (c,d).
#    - 原题删除条件是“组件颜色集合包含全部图例色”.
#      融合后删除条件改成“组件内部必须出现全部 3 个相邻层颜色对的 4 邻接接触”.
#    - 具体例子:
#      若图例从外到内是 [1,4,6,8], 则所需配对是 (1,4), (4,6), (6,8).
#      一个组件里即使同时含有 1/4/6/8 四种颜色, 但若内部只有相邻对 (1,6) 和 (4,8),
#      缺少真正的 (1,4)/(4,6)/(6,8), 则新规则保留, 原规则会误删.
#
# b) 复杂度校准
#    - 采用固定 4 层 => 固定 3 个配对, 不把自由度放得过大; 两三组样例足够让人看到“读嵌套顺序”.
#    - 图例尺寸固定为 7x7, 层数稳定, 视觉上明显不同于原题的 4x4 L 形角标.
#    - generator 每个样例强制至少放 1 个“含全部 4 个图例色但缺少正确相邻配对”的保留组件,
#      从而稳定打破原题的颜色集合逻辑.
#
# c) 隐式假设审查
#    - 原 `_find_legend()` 假设图例是“底边+右边单色边框围出的区域”; 这个前提失效, 必须改成解析同心方框层.
#    - 原 solver 对图例内部取 unique 集合即可; 现在不行, 因为颜色顺序决定配对关系.
#    - 原 solver 直接对全部 `g != bg` 做连通分量. 新图例本身是一个多色大连通块, 若不显式跳过 `[:sz,:sz]`,
#      图例会被当成候选组件, 甚至天然含有全部相邻层接触. 所以主逻辑必须排除整个图例区.
#    - 新 `main()` 还隐含一个前提: 组件里的“正确配对”来自 generator 的刻意摆放而不是偶然碰撞.
#      生成器通过组间缓冲区阻止额外彩色邻接, 保证这个前提成立.
#
# d) 放置合法性
#    - 图例完全占据左上角 7x7, 组件通过 `random_free_location(..., padding=1)` 放置,
#      与图例及彼此之间至少隔 1 圈背景, 因而不会粘连成同一连通分量.
#    - 组件内部的彩色证据只使用“单点”和“相邻二元组”; 每放一组就封锁其 4 邻域,
#      防止不同组之间意外形成新的颜色对.
#    - 矩形 patch 本身用颜色 0 填满, 所以每个 patch 始终是单个 4-连通组件.
#
# e) 视觉可观察性
#    - 解题者可以直接看到图例是同心方框, 并自然读出“外层到内层”的顺序.
#    - 组件内的关键证据仍然是“两个彩色点是否贴边相邻”, 这是纯视觉可见的, 不依赖颜色数值比较.
#    - 每个样例都包含“颜色全齐但配对不对”的保留对象, 能强迫观察者放弃原题的集合思路.
#
# f) 可视化整洁度
#    - 图例区域是规整的 7x7 同心方框, 和下方/右侧对象有 1 格空隙, 不会混在一起.
#    - 组件内部只放少量配对证据和少量干扰单点, 保持底色矩形清晰, 避免密集噪声.
#    - 配对证据被刻意分散, 同一组件中不同二元组不会首尾相接形成杂乱链条.
#
# g) 变化的可推断性
#    - 颜色变化并非任意映射, 而是由“嵌套顺序”这一强结构提示出来: 邻接层才配对.
#    - 图例只用 4 个颜色、3 个配对, 并在每个样例中重复支撑, 人类能从少量例子归纳规则.


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color on the 1-cell outer border."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _ring_values(block: np.ndarray, depth: int) -> np.ndarray:
    """Values on one square ring of a top-left nested-frame block."""
    sz = int(block.shape[0])
    sub = block[depth : sz - depth, depth : sz - depth]
    if sub.shape == (1, 1):
        return sub.reshape(-1)
    return np.concatenate([sub[0, :], sub[-1, :], sub[1:-1, 0], sub[1:-1, -1]])


def _find_legend(g: np.ndarray, bg: int) -> tuple[int, list[int]]:
    """
    Detect a top-left odd-sized square made of one-cell-thick nested square frames.

    Returns:
      - legend size
      - colors from outermost layer to innermost layer
    """
    h, w = g.shape
    max_sz = min(h, w)
    if max_sz % 2 == 0:
        max_sz -= 1

    for sz in range(max_sz, 2, -2):
        block = g[:sz, :sz]
        if np.any(block == bg):
            continue
        if sz < w and not np.all(g[:sz, sz] == bg):
            continue
        if sz < h and not np.all(g[sz, :sz] == bg):
            continue

        colors: list[int] = []
        ok = True
        for depth in range((sz + 1) // 2):
            vals = _ring_values(block, depth)
            col = int(vals[0])
            if col == bg or not np.all(vals == col):
                ok = False
                break
            if colors and col == colors[-1]:
                ok = False
                break
            colors.append(col)

        if ok and len(colors) >= 2 and len(set(colors)) == len(colors):
            return int(sz), colors
    raise AssertionError("no nested-frame legend found in grid")


def _component_present_pairs(
    g: np.ndarray,
    pts_set: set[tuple[int, int]],
    tracked_colors: set[int],
) -> set[tuple[int, int]]:
    """All unordered 4-neighbor color pairs inside one component, restricted to tracked colors."""
    pairs: set[tuple[int, int]] = set()
    for r, c in pts_set:
        a = int(g[r, c])
        if a not in tracked_colors:
            continue
        for nr, nc in ((r + 1, c), (r, c + 1)):
            if (nr, nc) not in pts_set:
                continue
            b = int(g[nr, nc])
            if b not in tracked_colors or a == b:
                continue
            pairs.add(tuple(sorted((a, b))))
    return pairs


def _required_pairs(legend_colors: list[int]) -> set[tuple[int, int]]:
    """Consecutive nested-frame colors define the required unordered adjacency pairs."""
    return {tuple(sorted((a, b))) for a, b in zip(legend_colors, legend_colors[1:])}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Delete connected components whose local touching color pairs realize the nested-frame legend chain.

    Input structure:
    - Background (bg) fills most of the 16x16 grid.
    - Top-left corner: a compact odd-sized square made of concentric one-cell-thick rectangular frames.
    - Reading those frame colors from outside to inside yields an ordered legend color chain.
    - Elsewhere: rectangular 4-connected objects built on color 0, with sparse colored markers embedded inside.

    Rule:
    - Parse the nested legend and convert consecutive frame colors into required color pairs.
    - Ignore the legend square itself when collecting components; it is only an instruction area.
    - For every other 4-connected component, inspect which legend colors touch by a 4-neighbor edge inside that component.
    - Erase the component iff it realizes every required pair from the legend chain; otherwise keep it unchanged.

    Key insight: The nested frames encode both hierarchy and pairing at once: adjacency in legend depth becomes the adjacency pattern that must appear inside a component.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)

    sz, legend_colors = _find_legend(g, bg)
    assert len(legend_colors) >= 2, "legend must contain at least two nested colors"
    need_pairs = _required_pairs(legend_colors)
    tracked = set(legend_colors)

    out = g.copy()
    obj_mask = (g != bg).copy()
    obj_mask[:sz, :sz] = False

    for pts in components_mask(obj_mask, connectivity=4):
        pts_set = set(map(tuple, pts))
        if need_pairs.issubset(_component_present_pairs(g, pts_set, tracked)):
            for r, c in pts:
                out[r, c] = bg
    return out


# --------------- generator (combo D3/3 + D5/3) ---------------

GRID_H, GRID_W = 16, 16
BG = 7
BASE = 0
LEGEND_SIZE = 7
PALETTE = [1, 2, 4, 5, 6, 8, 9]
NUM_PAIRS = 3
NUM_KEPT = 3

SHAPE_WEIGHTS = [4, 3, 3]  # medium, tall, wide
TALL_H = (5, 8)
TALL_W = (3, 4)
WIDE_H = (3, 4)
WIDE_W = (5, 8)
MED_H = (4, 6)
MED_W = (4, 6)

DENSITY_LO = 0.08
DENSITY_HI = 0.32

COMP_COUNTS = [4, 5]
COMP_WEIGHTS = [5, 2]

PATCH_LAYOUT_RETRIES = 80
MAX_RETRIES = 500


def _draw_legend(g: np.ndarray, legend_colors: list[int]) -> None:
    """Render the 7x7 nested square-frame legend in the top-left corner."""
    s = LEGEND_SIZE
    g[:s, :s] = BG
    for depth, col in enumerate(legend_colors):
        r0 = depth
        r1 = s - depth
        g[r0:r1, r0] = col
        g[r0:r1, r1 - 1] = col
        g[r0, r0:r1] = col
        g[r1 - 1, r0:r1] = col


def _place_legend(inp: np.ndarray, out: np.ndarray, legend_colors: list[int]) -> None:
    for g in (inp, out):
        _draw_legend(g, legend_colors)


def _component_dims() -> tuple[int, int]:
    """Sample component dimensions that fit several separated markers."""
    mode = random.choices(["medium", "tall", "wide"], weights=SHAPE_WEIGHTS)[0]
    if mode == "tall":
        return random.randint(*TALL_H), random.randint(*TALL_W)
    if mode == "wide":
        return random.randint(*WIDE_H), random.randint(*WIDE_W)
    return random.randint(*MED_H), random.randint(*MED_W)


def _group_cells(cell_group: tuple[tuple[int, int], ...]) -> set[tuple[int, int]]:
    """Cells in one marker plus a 4-neighbor buffer around them."""
    blocked: set[tuple[int, int]] = set()
    for r, c in cell_group:
        blocked.add((r, c))
        blocked.add((r - 1, c))
        blocked.add((r + 1, c))
        blocked.add((r, c - 1))
        blocked.add((r, c + 1))
    return blocked


def _candidate_groups(h: int, w: int, size: int) -> list[tuple[tuple[int, int], ...]]:
    """Candidate singleton/domino placements inside one patch."""
    groups: list[tuple[tuple[int, int], ...]] = []
    if size == 1:
        for r in range(h):
            for c in range(w):
                groups.append(((r, c),))
        return groups

    for r in range(h):
        for c in range(w - 1):
            groups.append(((r, c), (r, c + 1)))
    for r in range(h - 1):
        for c in range(w):
            groups.append(((r, c), (r + 1, c)))
    return groups


def _layout_groups(h: int, w: int, lengths: list[int]) -> list[tuple[tuple[int, int], ...]] | None:
    """Pick non-interfering marker locations; groups only touch when intentionally placed as a domino."""
    for _ in range(PATCH_LAYOUT_RETRIES):
        blocked: set[tuple[int, int]] = set()
        groups: list[tuple[tuple[int, int], ...]] = []
        ok = True
        for size in lengths:
            candidates = _candidate_groups(h, w, size)
            random.shuffle(candidates)
            chosen = None
            for group in candidates:
                if all((r, c) not in blocked for r, c in group):
                    chosen = group
                    groups.append(group)
                    blocked |= _group_cells(group)
                    break
            if chosen is None:
                ok = False
                break
        if ok:
            return groups
    return None


def _present_pairs_in_patch(patch: np.ndarray, tracked_colors: set[int]) -> set[tuple[int, int]]:
    """Legend-color adjacency pairs present in a component patch."""
    h, w = patch.shape
    pairs: set[tuple[int, int]] = set()
    for r in range(h):
        for c in range(w):
            a = int(patch[r, c])
            if a not in tracked_colors:
                continue
            for nr, nc in ((r + 1, c), (r, c + 1)):
                if nr >= h or nc >= w:
                    continue
                b = int(patch[nr, nc])
                if b not in tracked_colors or a == b:
                    continue
                pairs.add(tuple(sorted((a, b))))
    return pairs


def _build_component_patch(h: int, w: int, legend_colors: list[int], role: str) -> np.ndarray | None:
    """
    Build one rectangular component.

    Roles:
      - delete_all_pairs: all consecutive legend pairs appear.
      - keep_broken_all_colors: all legend colors appear, but only wrong pairings do.
      - keep_partial: some correct pairs may appear, but at least one required pair is missing.
    """
    area = h * w
    min_colored = max(1, math.ceil(area * DENSITY_LO))
    max_colored = max(min_colored, math.floor(area * DENSITY_HI))
    tracked = set(legend_colors)
    req_pairs = _required_pairs(legend_colors)
    spare_colors = [c for c in PALETTE if c not in tracked]

    colored_groups: list[tuple[list[int], int]]
    if role == "delete_all_pairs":
        colored_groups = [([a, b], 2) for a, b in zip(legend_colors, legend_colors[1:])]
    elif role == "keep_broken_all_colors":
        colored_groups = [
            ([legend_colors[0], legend_colors[2]], 2),
            ([legend_colors[1], legend_colors[3]], 2),
        ]
    else:
        kept_count = random.randint(0, NUM_PAIRS - 1)
        kept_idx = set(random.sample(range(NUM_PAIRS), kept_count))
        colored_groups = []
        for i, (a, b) in enumerate(zip(legend_colors, legend_colors[1:])):
            if i in kept_idx:
                colored_groups.append(([a, b], 2))
            else:
                colored_groups.append(([a], 1))
                colored_groups.append(([b], 1))

    colored_cells = sum(size for _, size in colored_groups)
    filler_palette = spare_colors if spare_colors else PALETTE
    filler_budget = max(0, max_colored - colored_cells)
    filler_count = random.randint(0, min(2, filler_budget))
    for _ in range(filler_count):
        colored_groups.append(([random.choice(filler_palette)], 1))
    while sum(size for _, size in colored_groups) < min_colored:
        colored_groups.append(([random.choice(filler_palette)], 1))

    groups = _layout_groups(h, w, [size for _, size in colored_groups])
    if groups is None:
        return None

    patch = np.full((h, w), BASE, dtype=int)
    for (colors, size), cells in zip(colored_groups, groups):
        if size == 1:
            (r, c), = cells
            patch[r, c] = colors[0]
        else:
            (r0, c0), (r1, c1) = cells
            patch[r0, c0] = colors[0]
            patch[r1, c1] = colors[1]

    present_pairs = _present_pairs_in_patch(patch, tracked)
    if role == "delete_all_pairs":
        if not req_pairs.issubset(present_pairs):
            return None
    else:
        if req_pairs.issubset(present_pairs):
            return None
        if role == "keep_broken_all_colors":
            present_colors = {int(v) for v in patch.flatten() if int(v) in tracked}
            if present_colors != tracked:
                return None

    return patch


def _component_stats(g: np.ndarray, sz: int, legend_colors: list[int]) -> list[dict]:
    """Summaries used by the generator to prevent degeneration to the old color-set rule."""
    tracked = set(legend_colors)
    req_pairs = _required_pairs(legend_colors)
    obj_mask = (g != BG).copy()
    obj_mask[:sz, :sz] = False

    stats: list[dict] = []
    for pts in components_mask(obj_mask, connectivity=4):
        pts_set = set(map(tuple, pts))
        colors = {int(g[r, c]) for r, c in pts if int(g[r, c]) in tracked}
        pairs = _component_present_pairs(g, pts_set, tracked)
        stats.append(
            {
                "colors": colors,
                "pairs": pairs,
                "matches_new": req_pairs.issubset(pairs),
                "matches_old": tracked.issubset(colors),
            }
        )
    return stats


def _gen_attempt() -> np.ndarray | None:
    """Construct one valid input grid for the combined nested-frame/pair-logic variant."""
    inp = np.full((GRID_H, GRID_W), BG, dtype=int)
    out = inp.copy()

    legend_colors = random.sample(PALETTE, NUM_PAIRS + 1)
    _place_legend(inp, out, legend_colors)

    component_count = random.choices(COMP_COUNTS, weights=COMP_WEIGHTS)[0]
    deleted_count = component_count - NUM_KEPT
    roles = ["delete_all_pairs"] * deleted_count + ["keep_broken_all_colors", "keep_partial", "keep_partial"]
    random.shuffle(roles)

    for role in roles:
        h, w = _component_dims()
        patch = _build_component_patch(h, w, legend_colors, role)
        if patch is None:
            return None
        try:
            r, c = random_free_location(
                inp, patch, transparent=BG, border=1, padding=1, padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(inp, patch, r, c)
        if role != "delete_all_pairs":
            paste(out, patch, r, c)

    if not np.array_equal(main(inp), out):
        return None

    # Anti-degeneration check: every sample must contain at least one component that
    # the old "contains all legend colors" rule would delete but the new pair rule keeps.
    stats = _component_stats(inp, LEGEND_SIZE, legend_colors)
    old_not_new = sum(int(s["matches_old"] and not s["matches_new"]) for s in stats)
    if old_not_new < 1:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
