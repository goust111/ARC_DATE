# Combo variant of abc82100 (index 5)
# combo_spec: D2/0+D10/1
# Name: 图例区域集中编码+边缘规则分布
#
# Design analysis
# a) 具体化与行为追踪
#    - 原题把每条规则编码成“局部相邻 source/target 色对 + 附近 teal 模板”，solver 在整张图上
#      找任何满足这种局部几何关系的片段，再把对应模板平移到同色散点上。
#    - 本融合变体把规则改成固定槽位的边缘图例: 四条边各有一个 3x6 的 legend 槽位，槽内第 2 行
#      左侧是 source 色块、隔一个黑格后是 target 色块，右侧 3x3 小窗显示 teal 模板。
#      规则最多 4 条，因此每条规则占用一个不同边缘槽位；散落 source 像素只会出现在中央工作区。
#    - 小例子追踪:
#        * 原逻辑: 在场景里看到 [红][黄] 相邻，并在其延长线上接近 teal 连通块，于是学到
#          “红像素 -> 黄模板”。
#        * 融合后: 先看边缘四个槽位，假设上边槽位的第 2 行出现 [红][黑][黄]，右侧 3x3 小窗里
#          有一个 teal 形状；再看中心区域若干红色单像素，输出就是把那个 3x3 形状平移到每个红点，
#          并改涂成黄色。
#        * 差异: 规则不再来自场景中的局部邻接证据，而是来自“边缘查表”；源像素与规则示例在空间上
#          明确分区，形成“周边图例 + 中心数据”的组织。
# b) 复杂度校准
#    - 保留 1-4 条规则、每条规则 3-5 个散落 source 像素，避免退化成单规则单目标。
#    - 模板用 3x3 窗口而不是原题任意大小 teal 连通块，这是为了让四边固定槽位紧凑可放，同时仍保留
#      足够多的 8 连通形状变化。额外要求模板包含固定锚点、同时触及顶行和底行，保证肉眼可区分。
#    - 颜色对保持全局唯一，且旁观像素颜色避开所有 src/tgt 色，减少“这个像素是规则源还是普通点”的歧义。
# c) 隐式假设审查
#    - 原 `_discover_rules()` 依赖“source/target 4 邻相接且沿延长线接近 teal 模板”。融合后该前提失效，
#      因此 solver 改成只检查四个固定边缘槽位。
#    - 原 `main()` 用 `legend_pixels` 排除局部示例像素；这里 legend 变成若干矩形槽位，所以排除逻辑改成
#      “位于已占用的边缘槽位内”。
#    - 新 solver 新增假设:
#        * 规则图例只会出现在 top/bottom/left/right 四个固定槽位之一。
#        * 每个槽位的编码格式固定为 [src][bg][tgt] + 右侧 3x3 teal 小窗。
#        * 所有真实 source 像素都位于中央工作区，不会塞进图例槽位。
#      generator 会显式构造并验证这些条件。
#    - 为避免退化回原题，图例内 source 与 target 中间故意隔一个黑格，旧版“相邻色对 + teal”发现器
#      不会把这些边缘图例重新识别成原题规则。
# d) 放置合法性
#    - 每条规则占用一个不同边缘槽位，最多 4 条正好对应四条边；槽位彼此不重叠。
#    - 散落 source 像素和旁观像素只放中央工作区，彼此保持 4 邻隔离；旁观像素还避开所有输出盖章足迹，
#      保证在输出中稳定保留。
#    - teal 只存在于边缘图例槽位，不会出现在中心工作区，因此 solver 的“先读图例再应用”流程具有
#      唯一入口。
# e) 视觉可观察性
#    - 人类会先看到四周边缘的若干小牌子，每张牌子明确给出 source 色、target 色以及模板缩略图；
#      再看到中心区域多枚同色单像素。输出把这些单像素替换成相应颜色的模板。
#    - 规则读取依赖可见的空间分区和模板形状，而不是颜色编号排序或不可见元数据。
# f) 可视化整洁度
#    - 图例只待在边缘固定槽位，源像素和旁观像素只待在中央工作区，避免对象互相覆盖。
#    - 四个槽位使用同一编码格式，便于跨样例快速对照；模板仅 3x3，边缘区域不会拥挤成杂乱拼贴。
# g) 变化的可推断性
#    - 颜色变化始终由同一张图例牌子显式展示: source 色块、target 色块和模板缩略图一起出现。
#    - 一题最多 4 条规则，查表负担可控；每条规则有多个散落 source 像素提供重复证据，便于从少量样例
#      归纳“同色单点都扩展成哪种目标色模板”。

from __future__ import annotations

import random
from functools import lru_cache
from itertools import combinations

import numpy as np

from arc_utils import get_bg, label_mask


PATTERN_COLOR = 8
NON_BG_NON_TEAL_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]

RULE_COUNT_WEIGHTS = [1, 1, 2, 2, 3, 3, 4]

GRID_DIM_MAX = 20
MIN_HEIGHT_BY_RULES = {1: 11, 2: 13, 3: 16, 4: 19}
MIN_WIDTH_BY_RULES = {1: 13, 2: 16, 3: 18, 4: 20}

ENTRY_H = 3
ENTRY_W = 6
ENTRY_SOURCE = (1, 0)
ENTRY_TARGET = (1, 2)
TEMPLATE_H = 3
TEMPLATE_W = 3
TEMPLATE_COL0 = 3
ANCHOR_LOCAL = (1, 1)

SCATTERED_MIN = 3
SCATTERED_MAX_CAP = 5
BYSTANDER_MAX = 5

SCATTERED_RETRIES_PER_PIXEL = 80
BYSTANDER_RETRIES = 80
OUTER_RETRIES = 400

EDGES = ("top", "right", "bottom", "left")
_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _reserve_4neighbors(reserved: set[tuple[int, int]], cells: list[tuple[int, int]]) -> None:
    for r, c in cells:
        for dr, dc in _DIRS4:
            reserved.add((int(r) + dr, int(c) + dc))


@lru_cache(maxsize=None)
def _valid_stencil_masks() -> tuple[np.ndarray, ...]:
    """All connected 3x3 masks that contain the fixed anchor and touch top/bottom rows."""
    masks: list[np.ndarray] = []
    anchor_index = ANCHOR_LOCAL[0] * TEMPLATE_W + ANCHOR_LOCAL[1]
    for area in range(3, 7):
        for cells in combinations(range(TEMPLATE_H * TEMPLATE_W), area):
            if anchor_index not in cells:
                continue
            mask = np.zeros((TEMPLATE_H, TEMPLATE_W), dtype=bool)
            for cell in cells:
                mask[cell // TEMPLATE_W, cell % TEMPLATE_W] = True
            if not mask[0, :].any() or not mask[-1, :].any():
                continue
            _labels, count = label_mask(mask, connectivity=8)
            if int(count) == 1:
                masks.append(mask)
    return tuple(masks)


def _edge_slots(shape: tuple[int, int]) -> dict[str, tuple[int, int]]:
    h, w = map(int, shape)
    return {
        "top": (0, (w - ENTRY_W) // 2),
        "bottom": (h - ENTRY_H, (w - ENTRY_W) // 2),
        "left": ((h - ENTRY_H) // 2, 0),
        "right": ((h - ENTRY_H) // 2, w - ENTRY_W),
    }


def _source_box(shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = map(int, shape)
    return ENTRY_H, h - ENTRY_H, ENTRY_W, w - ENTRY_W


def _slot_cells(edge: str, shape: tuple[int, int]) -> set[tuple[int, int]]:
    r0, c0 = _edge_slots(shape)[edge]
    return {
        (int(r0) + dr, int(c0) + dc)
        for dr in range(ENTRY_H)
        for dc in range(ENTRY_W)
    }


def _legend_entry(rule: dict) -> np.ndarray:
    patch = np.zeros((ENTRY_H, ENTRY_W), dtype=int)
    patch[ENTRY_SOURCE] = int(rule["src"])
    patch[ENTRY_TARGET] = int(rule["tgt"])
    window = patch[:, TEMPLATE_COL0 : TEMPLATE_COL0 + TEMPLATE_W]
    window[np.asarray(rule["mask"], dtype=bool)] = PATTERN_COLOR
    return patch


def _build_rules(num_rules: int) -> list[dict]:
    colors = NON_BG_NON_TEAL_COLORS[:]
    random.shuffle(colors)
    edges = random.sample(list(EDGES), num_rules)
    masks = list(_valid_stencil_masks())
    rules: list[dict] = []
    for i, edge in enumerate(edges):
        src = colors[2 * i]
        tgt = colors[2 * i + 1]
        mask = random.choice(masks)
        rel = np.argwhere(mask).astype(int) - np.array(ANCHOR_LOCAL, dtype=int)
        rules.append(
            {
                "src": int(src),
                "tgt": int(tgt),
                "edge": edge,
                "mask": mask.copy(),
                "rel": np.asarray(rel, dtype=int),
            }
        )
    return rules


def _decode_slot(patch: np.ndarray, bg: int, edge: str) -> dict | None:
    patch = np.asarray(patch, dtype=int)
    if not np.any(patch != int(bg)):
        return None

    allowed = np.zeros_like(patch, dtype=bool)
    allowed[ENTRY_SOURCE] = True
    allowed[ENTRY_TARGET] = True
    allowed[:, TEMPLATE_COL0 : TEMPLATE_COL0 + TEMPLATE_W] = True
    if np.any((patch != int(bg)) & (~allowed)):
        raise ValueError(f"unexpected pixels in {edge} legend slot")

    src = int(patch[ENTRY_SOURCE])
    tgt = int(patch[ENTRY_TARGET])
    if src in (int(bg), int(PATTERN_COLOR)) or tgt in (int(bg), int(PATTERN_COLOR)) or src == tgt:
        raise ValueError(f"invalid color pair in {edge} legend slot")

    window = patch[:, TEMPLATE_COL0 : TEMPLATE_COL0 + TEMPLATE_W]
    if np.any((window != int(bg)) & (window != int(PATTERN_COLOR))):
        raise ValueError(f"non-teal noise in {edge} template window")
    mask = window == int(PATTERN_COLOR)
    if not bool(mask[ANCHOR_LOCAL]):
        raise ValueError(f"missing anchor cell in {edge} legend slot")
    _labels, count = label_mask(mask, connectivity=8)
    if int(count) != 1:
        raise ValueError(f"template in {edge} legend slot must be 8-connected")

    rel = np.argwhere(mask).astype(int) - np.array(ANCHOR_LOCAL, dtype=int)
    return {
        "source_color": src,
        "target_color": tgt,
        "rel": np.asarray(rel, dtype=int),
        "edge": edge,
    }


def _decode_legend_rules(grid: np.ndarray, bg: int) -> tuple[list[dict], set[tuple[int, int]]]:
    slots = _edge_slots(grid.shape)
    rules: list[dict] = []
    legend_cells: set[tuple[int, int]] = set()
    seen_edges: set[str] = set()
    seen_sources: set[int] = set()
    seen_targets: set[int] = set()

    for edge in EDGES:
        r0, c0 = slots[edge]
        patch = grid[r0 : r0 + ENTRY_H, c0 : c0 + ENTRY_W]
        rule = _decode_slot(patch, bg=bg, edge=edge)
        if rule is None:
            continue
        src = int(rule["source_color"])
        tgt = int(rule["target_color"])
        if edge in seen_edges or src in seen_sources or tgt in seen_targets:
            raise ValueError("duplicate legend encoding")
        seen_edges.add(edge)
        seen_sources.add(src)
        seen_targets.add(tgt)
        rules.append(rule)
        legend_cells |= _slot_cells(edge, grid.shape)

    if not rules:
        raise ValueError("legend not found on edges")
    return rules, legend_cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read edge legends as a lookup table, then expand matching source pixels in the center.

    Input structure:
    - Black background with up to four fixed legend slots attached to the grid edges.
    - Each occupied slot is a 3x6 plaque: source color, one black spacer, target color,
      and a 3x3 teal miniature stencil on the right.
    - Colored source pixels to be transformed are scattered only in the central workspace,
      away from the edge legend plaques.

    Rule:
    - Decode every occupied edge slot into a source_color -> (target_color, stencil) rule.
    - For each central pixel whose color matches a legend source color, place that slot's
      stencil so its fixed anchor lands on the source pixel, then recolor the stencil to
      the corresponding target color.
    - Remove all legend plaques and the source pixels they consume; preserve unrelated
      bystander pixels unchanged.

    Key insight: The rule evidence is no longer hidden in local scene fragments. The border
    itself is the legend area, so the solver must first parse the edge lookup plaques and only
    then apply the listed stencil/color substitutions to the central data region.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    h, w = g.shape

    rules, legend_cells = _decode_legend_rules(g, bg=bg)
    source_colors = {int(rule["source_color"]) for rule in rules}

    out = np.full_like(g, int(bg))

    for r, c in map(tuple, np.argwhere((g != int(bg)) & (g != int(PATTERN_COLOR)))):
        r = int(r)
        c = int(c)
        color = int(g[r, c])
        if (r, c) in legend_cells:
            continue
        if color in source_colors:
            continue
        out[r, c] = color

    for rule in rules:
        rel = np.asarray(rule["rel"], dtype=int)
        for sr, sc in map(tuple, np.argwhere(g == int(rule["source_color"]))):
            sr = int(sr)
            sc = int(sc)
            if (sr, sc) in legend_cells:
                continue
            rows = sr + rel[:, 0]
            cols = sc + rel[:, 1]
            ok = (0 <= rows) & (rows < h) & (0 <= cols) & (cols < w)
            out[rows[ok], cols[ok]] = int(rule["target_color"])

    return out


def _stamp_cells(sr: int, sc: int, rel: np.ndarray, shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = map(int, shape)
    cells: set[tuple[int, int]] = set()
    for rr, cc in np.asarray(rel, dtype=int):
        nr = int(sr) + int(rr)
        nc = int(sc) + int(cc)
        if 0 <= nr < h and 0 <= nc < w:
            cells.add((nr, nc))
    return cells


def _place_scattered(rules: list[dict], shape: tuple[int, int]) -> bool:
    r0, r1, c0, c1 = _source_box(shape)
    if r1 <= r0 or c1 <= c0:
        return False

    box_area = (r1 - r0) * (c1 - c0)
    sc_max = min(SCATTERED_MAX_CAP, max(SCATTERED_MIN, box_area // max(8, 5 * len(rules))))
    occupied: set[tuple[int, int]] = set()
    reserved: set[tuple[int, int]] = set()

    for rule in rules:
        n_sc = random.randint(SCATTERED_MIN, sc_max)
        positions: list[tuple[int, int]] = []
        for _ in range(n_sc * SCATTERED_RETRIES_PER_PIXEL):
            if len(positions) >= n_sc:
                break
            r = random.randint(r0, r1 - 1)
            c = random.randint(c0, c1 - 1)
            if (r, c) in occupied or (r, c) in reserved:
                continue
            positions.append((r, c))
            occupied.add((r, c))
            _reserve_4neighbors(reserved, [(r, c)])
        if len(positions) < SCATTERED_MIN:
            return False
        rule["scattered"] = positions

    return True


def _stamp_footprint(rules: list[dict], shape: tuple[int, int]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for rule in rules:
        for sr, sc in rule["scattered"]:
            cells |= _stamp_cells(sr, sc, rule["rel"], shape)
    return cells


def _place_bystanders(rules: list[dict], shape: tuple[int, int]) -> list[tuple[int, int, int]]:
    r0, r1, c0, c1 = _source_box(shape)
    used_colors = {int(rule["src"]) for rule in rules} | {int(rule["tgt"]) for rule in rules}
    bystander_colors = [c for c in NON_BG_NON_TEAL_COLORS if c not in used_colors]
    if not bystander_colors or r1 <= r0 or c1 <= c0:
        return []

    occupied = {tuple(map(int, pos)) for rule in rules for pos in rule["scattered"]}
    reserved: set[tuple[int, int]] = set()
    _reserve_4neighbors(reserved, list(occupied))
    forbidden = _stamp_footprint(rules, shape)

    n = random.randint(0, min(BYSTANDER_MAX, len(bystander_colors) * 2))
    placed: list[tuple[int, int, int]] = []
    for _ in range(n):
        for _ in range(BYSTANDER_RETRIES):
            r = random.randint(r0, r1 - 1)
            c = random.randint(c0, c1 - 1)
            if (r, c) in occupied or (r, c) in reserved or (r, c) in forbidden:
                continue
            color = int(random.choice(bystander_colors))
            placed.append((r, c, color))
            occupied.add((r, c))
            _reserve_4neighbors(reserved, [(r, c)])
            break
    return placed


def _build_output(rules: list[dict], bystanders: list[tuple[int, int, int]], shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for r, c, color in bystanders:
        out[int(r), int(c)] = int(color)
    for rule in rules:
        for sr, sc in rule["scattered"]:
            for nr, nc in _stamp_cells(sr, sc, rule["rel"], shape):
                out[int(nr), int(nc)] = int(rule["tgt"])
    return out


def _build_input(rules: list[dict], bystanders: list[tuple[int, int, int]], shape: tuple[int, int]) -> np.ndarray:
    inp = np.zeros(shape, dtype=int)
    slots = _edge_slots(shape)
    for rule in rules:
        r0, c0 = slots[rule["edge"]]
        inp[r0 : r0 + ENTRY_H, c0 : c0 + ENTRY_W] = _legend_entry(rule)
    for rule in rules:
        for r, c in rule["scattered"]:
            inp[int(r), int(c)] = int(rule["src"])
    for r, c, color in bystanders:
        inp[int(r), int(c)] = int(color)
    return inp


def _gen_attempt() -> np.ndarray | None:
    num_rules = random.choice(RULE_COUNT_WEIGHTS)
    h = random.randint(MIN_HEIGHT_BY_RULES[num_rules], GRID_DIM_MAX)
    w = random.randint(MIN_WIDTH_BY_RULES[num_rules], GRID_DIM_MAX)
    shape = (h, w)

    rules = _build_rules(num_rules)
    if not _place_scattered(rules, shape):
        return None
    bystanders = _place_bystanders(rules, shape)

    inp = _build_input(rules, bystanders, shape)
    expected = _build_output(rules, bystanders, shape)
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
