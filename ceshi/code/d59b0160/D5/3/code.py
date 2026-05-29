# Variant of d59b0160
# Dimension: D5 — 图例颜色集合的提取方式 (legend_lookup)
# Change: 图例区域内所有非背景(7)、非边框(3)颜色的集合作为目标颜色集... → 图例定义颜色对应关系而非颜色集合

import math
import random

import numpy as np

from arc_utils import components_mask, paste, random_free_location


# Design analysis for the D5 variant
# ==================================
# a) 具体化与行为追踪
#    - 保留原题的大框架: 16x16 网格, 背景 7, 左上角 4x4 图例, 其它位置放置若干 4-连通矩形组件.
#    - 图例仍由底边和右边的绿色边框(3)围出, 但 3x3 内部不再是“3 个散落目标色”.
#      现在固定为:
#        row 0: 3 个源色
#        row 1: 与上面同列对应的 3 个目标色
#        row 2: 全部保持背景 7 作为视觉分隔
#      例如内区若是
#        [1 4 6]
#        [2 5 8]
#        [7 7 7]
#      就表示三组配对: (1,2), (4,5), (6,8).
#    - 组件仍是由颜色 0 填满的矩形, 其内嵌入若干彩色像素. 与原题不同的是, 彩色像素会刻意组成
#      “相邻二元组”或“分离单点”, 使“是否存在某个颜色对的相邻出现”在视觉上可见.
#    - 原题在一个组件上的判定是:
#        组件颜色集合是否包含图例给出的全部 3 个颜色
#      变体后的判定是:
#        对图例中的每一列颜色对, 组件中是否至少出现一次这两个颜色的 4 邻接接触.
#      只有当 3 组相邻色对都出现时才删除该组件.
#    - 行为追踪小例子:
#      原逻辑: 图例颜色集 {1,2,4}; 某组件里虽然 1,2,4 都出现, 就会被删.
#      新逻辑: 图例对是 (1,2), (4,5), (6,8); 某组件即使含有 1,2,4,5,6,8 六种颜色,
#      但若内部只出现了相邻对 (1,5), (2,6), (4,8), 缺少真正的 (1,2)/(4,5)/(6,8),
#      则必须保留. 这正是 D5 的核心差异.
#
# b) 复杂度校准
#    - 颜色对数量固定为 3 组, 不再增加额外自由度. 这样两三组样例足以让人观察到“看的是配对, 不是全色集”.
#    - 为防止视觉上退化回原题, generator 每个样例都至少制造 1 个“含全部 6 个图例颜色但缺少正确配对”
#      的保留组件. 这样旧规则会删, 新规则会留, 差异稳定可见.
#    - 同时保留 1-2 个真正应删除的组件, 让“完整满足 3 组配对就删”得到重复支撑.
#
# c) 隐式假设审查
#    - 原 solver 的 `_find_legend()` 默认“图例内部只是颜色集合, 不关心位置”. 现在这个前提失效,
#      因而必须按列读取前两行, 不能再取 unique set.
#    - 原 solver 把所有 `g != bg` 的连通分量都拿来判定. 原题图例像素彼此 4-不连通, 因而无害;
#      现在图例第一行、第二行本身有相邻颜色, 若不排除整个图例方块, solver 可能把图例自己当成候选组件.
#      所以新的 `main()` 必须显式跳过 `[:sz, :sz]` 图例区域.
#    - 原 solver 只检查“颜色是否出现过”, 与颜色相对位置无关. 变体里这不再成立, 必须转为逐组件检查
#      4 邻接边上是否出现指定颜色对.
#
# d) 放置合法性
#    - 每个组件仍通过 `random_free_location(..., padding=1)` 放置, 保证与图例及其它组件至少隔一圈背景,
#      不重叠、不断边、不把两个组件粘成一个大连通块.
#    - 组件内部的彩色标记通过一个“带缓冲区”的小型布局器生成: 每放一个单点或一个二元相邻对,
#      就把它自身及其 4 邻域加入禁放区. 这样除了刻意设置的二元组外, 不会再产生额外相邻关系.
#    - solver 的新假设是“组件中出现的 legend pair, 真的是 generator 有意放进去的配对”.
#      上述禁放策略正是在构造上保证这一点.
#
# e) 视觉可观察性
#    - 图例的两行同列对应是直接可见的, 不依赖颜色 id 排序.
#    - 组件内的关键证据是“两个彩色像素是否贴在一起”, 这是纯视觉信息, 两三组样例即可归纳.
#    - 保留组件中会经常出现“颜色都在, 但正确相邻搭配不全”的情况, 促使解题者放弃原题的集合思路.
#
# f) 可视化整洁度
#    - 组件仍是规整矩形, 背景与图例区域保持分离.
#    - 图例第三行留空, 避免 3x3 内部显得过密, 也让“上排 source / 下排 target”更清楚.
#    - 组件内部彩色点数量控制在较低密度, 且配对之间留空, 避免杂乱.
#
# g) 颜色变化可推断性
#    - 本变体不做颜色替换, 只是把“看颜色集合”改为“看颜色配对”.
#    - 每个样例只用 3 组配对、共 6 个图例颜色, 结构信息集中且反复出现, 人类可从少量样例推断.


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color along the 1-cell outer border."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _find_legend(g: np.ndarray, bg: int) -> tuple[int, int, list[tuple[int, int]]]:
    """
    Detect the top-left legend framed by a uniform bottom edge and right edge.

    In this variant, the legend interior encodes column-wise color correspondences:
      - first interior row: source colors
      - second interior row: target colors
      - same column => one required pair
    """
    h, w = g.shape
    for sz in range(3, min(h, w) + 1):
        x = int(g[sz - 1, sz - 1])
        if x == bg:
            continue
        if np.all(g[sz - 1, 0:sz] == x) and np.all(g[0:sz, sz - 1] == x):
            inside = g[0 : sz - 1, 0 : sz - 1]
            if inside.shape[0] < 2:
                continue
            pairs: list[tuple[int, int]] = []
            for c in range(inside.shape[1]):
                src = int(inside[0, c])
                dst = int(inside[1, c])
                if src in (bg, x) or dst in (bg, x):
                    continue
                pairs.append((src, dst))
            if pairs:
                return int(sz), int(x), pairs
    assert False, "no legend border found in grid"


def _component_has_pair(g: np.ndarray, pts_set: set[tuple[int, int]], a: int, b: int) -> bool:
    """Return True iff this component contains a 4-neighbor adjacency between colors a and b."""
    for r, c in pts_set:
        col = int(g[r, c])
        if col != a:
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Delete connected components whose local adjacent color pairs realize every legend mapping.

    Input structure:
    - Background (bg) fills most of the grid.
    - Top-left corner: a small square legend bordered on its bottom and right edges by one color.
    - Inside that legend, the first interior row lists source colors and the second interior row lists matching target colors; same-column entries form the required pairs.
    - Elsewhere: multicolored 4-connected objects built on the bg.

    Rule:
    - Read the legend column pairs from the first two interior rows.
    - Ignore the legend square itself when scanning components; the legend is only an instruction area.
    - For each other connected component, test every legend pair and ask whether those two colors ever touch by a 4-neighbor edge inside that component.
    - Erase (fill with bg) every component that realizes all legend pairs; leave all others unchanged.

    Key insight: The legend no longer gives a palette to collect globally; it specifies which colors must show up as touching pairs inside the same object.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)

    sz, _border, legend_pairs = _find_legend(g, bg)
    assert legend_pairs, "legend contains no color pairs"

    out = g.copy()
    obj_mask = (g != bg).copy()
    obj_mask[:sz, :sz] = False

    comps = components_mask(obj_mask, connectivity=4)
    for pts in comps:
        pts_set = set(map(tuple, pts))
        if all(_component_has_pair(g, pts_set, src, dst) or _component_has_pair(g, pts_set, dst, src)
               for src, dst in legend_pairs):
            for r, c in pts:
                out[r, c] = bg
    return out


# --------------- generator (D5 pair-legend variant) ---------------

GRID_H, GRID_W = 16, 16
BG = 7
BORDER = 3
LEGEND_SIZE = 4
PALETTE = [1, 2, 4, 5, 6, 8, 9]
NUM_PAIRS = 3
NUM_KEPT = 3

SHAPE_WEIGHTS = [4, 3, 3]  # medium, tall, wide
TALL_H = (6, 10)
TALL_W = (3, 5)
WIDE_H = (3, 5)
WIDE_W = (6, 9)
MED_H = (4, 7)
MED_W = (4, 7)

DENSITY_LO = 0.10
DENSITY_HI = 0.45

COMP_COUNTS = [4, 5]
COMP_WEIGHTS = [4, 1]

PATCH_LAYOUT_RETRIES = 80
MAX_RETRIES = 400


def _place_legend(inp: np.ndarray, out: np.ndarray, pairs: list[tuple[int, int]]) -> None:
    """Draw the 4x4 legend box and encode the 3 column-wise source->target pairs."""
    s = LEGEND_SIZE
    for g in (inp, out):
        g[:s, :s] = BG
        g[s - 1, :s] = BORDER
        g[:s, s - 1] = BORDER
        for idx, (src, dst) in enumerate(pairs):
            g[0, idx] = src
            g[1, idx] = dst


def _component_dims() -> tuple[int, int]:
    """Sample component dimensions adapted to hold several separated pair markers."""
    mode = random.choices(["medium", "tall", "wide"], weights=SHAPE_WEIGHTS)[0]
    if mode == "tall":
        return random.randint(*TALL_H), random.randint(*TALL_W)
    if mode == "wide":
        return random.randint(*WIDE_H), random.randint(*WIDE_W)
    return random.randint(*MED_H), random.randint(*MED_W)


def _group_cells(cell_group: tuple[tuple[int, int], ...]) -> set[tuple[int, int]]:
    """Cells in a group plus their 4-neighbor buffer, used to prevent accidental pair creation."""
    blocked: set[tuple[int, int]] = set()
    for r, c in cell_group:
        blocked.add((r, c))
        blocked.add((r - 1, c))
        blocked.add((r + 1, c))
        blocked.add((r, c - 1))
        blocked.add((r, c + 1))
    return blocked


def _candidate_groups(h: int, w: int, size: int) -> list[tuple[tuple[int, int], ...]]:
    """Candidate placement groups for one marker: singleton or adjacent domino."""
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
    """Choose non-interfering singleton/domino locations inside a component patch."""
    for _ in range(PATCH_LAYOUT_RETRIES):
        blocked: set[tuple[int, int]] = set()
        groups: list[tuple[tuple[int, int], ...]] = []
        ok = True
        for size in lengths:
            candidates = _candidate_groups(h, w, size)
            random.shuffle(candidates)
            placed = None
            for group in candidates:
                if all((r, c) not in blocked for r, c in group):
                    placed = group
                    groups.append(group)
                    blocked |= _group_cells(group)
                    break
            if placed is None:
                ok = False
                break
        if ok:
            return groups
    return None


def _max_colored_cells(area: int) -> int:
    hi = max(1, math.floor(area * DENSITY_HI))
    return hi


def _build_component_patch(h: int, w: int, pairs: list[tuple[int, int]], role: str) -> np.ndarray | None:
    """
    Build a rectangular color-0 component whose colored markers enforce the intended pair logic.

    Roles:
      - delete_all_pairs: contains all 3 correct adjacent pairs
      - keep_broken_all_colors: contains all 6 legend colors but only wrong adjacent pairings
      - keep_partial: contains at most 2 correct pairs and misses at least 1 legend pair
    """
    area = h * w
    min_colored = max(1, math.ceil(area * DENSITY_LO))
    max_colored = _max_colored_cells(area)
    pair_set = {tuple(sorted(p)) for p in pairs}

    if role == "delete_all_pairs":
        colored_groups: list[tuple[list[int], int]] = [([src, dst], 2) for src, dst in pairs]
    elif role == "keep_broken_all_colors":
        targets = [dst for _, dst in pairs]
        rotated = targets[1:] + targets[:1]
        colored_groups = [([src, wrong_dst], 2) for (src, _), wrong_dst in zip(pairs, rotated)]
    else:
        kept_count = random.randint(0, NUM_PAIRS - 1)
        kept_idx = set(random.sample(range(NUM_PAIRS), kept_count))
        colored_groups = []
        for i, (src, dst) in enumerate(pairs):
            if i in kept_idx:
                colored_groups.append(([src, dst], 2))
            else:
                # Separate singletons make the missing pair visually present-but-unpaired.
                colored_groups.append(([src], 1))
                colored_groups.append(([dst], 1))

    colored_cells = sum(size for _, size in colored_groups)
    filler_budget = max(0, max_colored - colored_cells)
    filler_count = random.randint(0, min(3, filler_budget))
    spare_colors = [c for c in PALETTE if c not in {x for p in pairs for x in p}]
    filler_palette = spare_colors if spare_colors else PALETTE
    for _ in range(filler_count):
        colored_groups.append(([random.choice(filler_palette)], 1))

    if sum(size for _, size in colored_groups) < min_colored:
        deficit = min_colored - sum(size for _, size in colored_groups)
        for _ in range(deficit):
            colored_groups.append(([random.choice(filler_palette)], 1))

    groups = _layout_groups(h, w, [size for _, size in colored_groups])
    if groups is None:
        return None

    patch = np.zeros((h, w), dtype=int)
    for (colors, size), cells in zip(colored_groups, groups):
        if size == 1:
            (r, c), = cells
            patch[r, c] = colors[0]
        else:
            (r0, c0), (r1, c1) = cells
            patch[r0, c0] = colors[0]
            patch[r1, c1] = colors[1]

    # Validate the intended semantics before returning.
    present_pairs = {
        tuple(sorted((int(patch[r, c]), int(patch[nr, nc]))))
        for r in range(h)
        for c in range(w)
        for nr, nc in ((r + 1, c), (r, c + 1))
        if nr < h and nc < w and patch[r, c] != 0 and patch[nr, nc] != 0
    }

    if role == "delete_all_pairs":
        if not pair_set.issubset(present_pairs):
            return None
    else:
        if pair_set.issubset(present_pairs):
            return None
        if role == "keep_broken_all_colors":
            flat_colors = {int(v) for v in patch.flatten() if v not in (0, BG)}
            if flat_colors != {x for p in pairs for x in p}:
                return None

    return patch


def _gen_attempt() -> np.ndarray | None:
    """Construct one valid input grid for the pair-based legend variant."""
    inp = np.full((GRID_H, GRID_W), BG, dtype=int)
    out = inp.copy()

    legend_colors = random.sample(PALETTE, 2 * NUM_PAIRS)
    sources = legend_colors[:NUM_PAIRS]
    targets = legend_colors[NUM_PAIRS:]
    pairs = list(zip(sources, targets))
    _place_legend(inp, out, pairs)

    component_count = random.choices(COMP_COUNTS, weights=COMP_WEIGHTS)[0]
    deleted_count = component_count - NUM_KEPT
    roles = ["delete_all_pairs"] * deleted_count + ["keep_broken_all_colors", "keep_partial", "keep_partial"]
    random.shuffle(roles)

    for role in roles:
        h, w = _component_dims()
        patch = _build_component_patch(h, w, pairs, role)
        if patch is None:
            return None
        try:
            r, c = random_free_location(
                inp, patch, transparent=BG, padding=1, padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(inp, patch, r, c)
        if not role.startswith("delete"):
            paste(out, patch, r, c)

    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
