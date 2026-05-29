# Combo variant of 28a6681f (index 28)
# combo_spec: D1/2+D7/1+D9/1
# Name: 反向重力+嵌套盆地+连通器均衡

from __future__ import annotations

import random

import numpy as np


# Design analysis
# a) 具体化与行为追踪
#    - 画面固定为 10x10，背景 0，水 1，静态障碍使用 2 种颜色。
#    - 结构改成“顶部锚定的大外壳 + 中央更高顶的小内盆 + 两侧更低顶的外层侧腔”。
#      这仍然是“盆中盆”，只是整体上下翻转：水源不在上方下落，而是在底部中央竖井里向上涌。
#    - 关键几何不是任意连通器，而是一个有层级顶高的连通腔体：
#      1) 内层小盆的顶部最高，只有它占据第 2-5 行的中央两列；
#      2) 外层两侧腔体的顶部更低，只从第 6 行开始出现；
#      3) 三个区域在更低处连成一个底部连通的开腔，所以一旦水量超过内盆容量，额外水量会在整组连通器里按共同水位均衡。
#    - 一个具体追踪：
#      若总水量为 16，原题的重力逻辑会把水往下压到最低处；本变体则把同样的 16 格水重新分布到该腔体的最高可占据行。
#      先占满内盆 8 格 (r=2..5, c=4..5)，剩余 8 格再占满下一层整行 (r=6, c=1..8)，于是两侧外层腔体同时出现水。
#      若总水量为 24，则继续把 r=7, c=1..8 也整行占满，表现为“先内后外、且外层左右同步均衡”。
#    - 差异总结：
#      原题依赖逐像素下落、对角滑动和水平找落点；本变体把方向翻成向上，并把核心从局部路径模拟改成
#      “底部连通、顶部封闭腔体中的体积-截面水位计算”。
#
# b) 复杂度校准
#    - 如果外层侧腔一开始就和内盆有相同的顶高，那么“先填内层”就不明显；如果侧腔只剩 1 格高，又会让连通器均衡太弱。
#    - 这里把内盆独占的高位区固定为 4 层 * 2 列 = 8 格；外层均衡区固定为两层、每层 8 格。
#      这样视觉上一定能看到：
#      1) 小内盆先被顶满；
#      2) 额外水量出现时左右两侧一起长出来，而不是只偏向一边。
#    - 水量只取 16 或 24。
#      这两个档位都严格对应“完整水位层”，避免半行水面带来的歧义，也都显式包含外层均衡，因此不会退化成只看反向重力或只看嵌套形状。
#
# c) 隐式假设审查（围绕新的 main()）
#    - 原 solver 的输入断言“水必须从下方受支撑”不再成立：现在水源就是底部上涌，输入可以是一团底部水库，不需要逐格有下方支撑。
#    - 原 solver 的 `settle()` 假定移动是单步、局部、单调向下的；连通器均衡需要按整体容积重排，尤其允许水最终出现在当前输入上方远处，
#      因此必须整体替换为连通腔体回填。
#    - 原 solver 的输出断言要求每个水格左/右/下都非背景，这对应“卡在盆底的水”；本变体应改成“上方非背景或触顶”，
#      因为稳定水是贴着顶部悬挂的。
#    - D7 的嵌套结构与 D9 的连通器机制叠加后，引入了一个新前提：决定先后顺序的不是颜色，而是连通腔体各行的可用截面。
#      也就是说，求解器默认“含水开腔与底边连通，但不接触顶/左/右边界”，生成器必须构造出这样的合法容器。
#
# d) 放置合法性
#    - 外壳固定为顶边整排屋顶、左右边墙、底部只留中央竖井开口；因此内部开腔与外部背景完全隔离，只通过底边中央 2 列接触网格底部。
#    - 内盆固定悬挂在中央，两侧外层腔通过更低的天花板形成；所有障碍都画在互不冲突的矩形带上，没有重叠与越界风险。
#    - 输入水只放在该开腔内部，绝不覆盖障碍。
#
# e) 视觉可观察性
#    - 人类看 2-4 组样例时，会直接看到：
#      1) 输入水都堆在下方中央来源处；
#      2) 输出水却跑到顶部，说明是反向重力；
#      3) 中央小盆上方总是先满；
#      4) 再多的水会同时出现在左右外层侧腔，说明不是单侧溢流，而是连通器均衡。
#    - 这里只有一种结构家族和两个水量层级，样例之间会重复支撑同一规律，不需要靠颜色编号或隐藏排序来推理。
#
# f) 可视化整洁度
#    - 外壳、内盆、水源三者分层清楚：顶部是静态壳体，中部是嵌套腔体，下部是中央水源。
#    - 没有额外导流件、斜坡或散落障碍，避免把注意力从“三层融合规则”上移开。
#    - 颜色只区分外壳与内盆，水保持蓝色 1，画面简洁。
#
# g) 变化的可推断性
#    - 颜色不编码任何顺序、优先级或匹配关系；几何位置本身就是规则线索。
#    - 真正的决定因素是：
#      1) 哪些行只属于内盆；
#      2) 哪些行属于整个连通器截面；
#      3) 总水量恰好覆盖了多少完整截面层。
#      这些都能从输入输出的空间结构直接观察出来。
# New solver assumptions audited against the generator:
#    - 每个含水开腔都是 4 连通、与底边相连、但不接触顶/左/右边界的单一容器分量。
#    - 生成器只采样完整水位层水量，因此 `main(check_input=True)` 可以要求没有“半层最终水面”。
#    - 该结构家族保证中央高顶区容量固定为 8，外层均衡区按整行扩张；这正是“先内后外”的唯一依据，求解器不依赖颜色大小。


BG = 0
WATER = 1


def _collect_component(
    open_mask: np.ndarray, start: tuple[int, int], seen: set[tuple[int, int]]
) -> list[tuple[int, int]]:
    h, w = open_mask.shape
    stack = [start]
    seen.add(start)
    comp: list[tuple[int, int]] = []
    while stack:
        r, c = stack.pop()
        comp.append((r, c))
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr = int(r + dr)
            cc = int(c + dc)
            nxt = (rr, cc)
            if 0 <= rr < h and 0 <= cc < w and bool(open_mask[rr, cc]) and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return comp


def _group_rows(component: list[tuple[int, int]]) -> dict[int, list[tuple[int, int]]]:
    by_row: dict[int, list[tuple[int, int]]] = {}
    for r, c in component:
        by_row.setdefault(int(r), []).append((int(r), int(c)))
    for cells in by_row.values():
        cells.sort()
    return by_row


def _fill_highest_rows(
    component: list[tuple[int, int]], water_count: int, require_exact_level: bool
) -> set[tuple[int, int]]:
    by_row = _group_rows(component)
    filled: set[tuple[int, int]] = set()
    remaining = int(water_count)
    for r in sorted(by_row):
        row_cells = by_row[r]
        if remaining >= len(row_cells):
            filled.update(row_cells)
            remaining -= len(row_cells)
            continue
        if remaining == 0:
            break
        if require_exact_level:
            raise AssertionError("water volume does not match a complete inverted shared waterline")
        filled.update(row_cells[:remaining])
        remaining = 0
        break
    if remaining != 0:
        raise AssertionError("water volume exceeds inverted vessel capacity")
    return filled


def _assert_top_packed(component: list[tuple[int, int]], filled: set[tuple[int, int]]) -> None:
    by_row = _group_rows(component)
    seen_air = False
    for r in sorted(by_row):
        row_cells = by_row[r]
        row_has_water = [cell in filled for cell in row_cells]
        if any(row_has_water) and not all(row_has_water):
            raise AssertionError("inverted communicating-vessel row cannot be partially filled")
        if all(row_has_water):
            assert not seen_air, "water must occupy the highest rows only"
        else:
            seen_air = True


def _component_source_order(component: list[tuple[int, int]]) -> list[tuple[int, int]]:
    center = 4.5
    return sorted(
        component,
        key=lambda p: (-int(p[0]), abs(float(p[1]) - center), int(p[1])),
    )


def _draw_case(outer_color: int, inner_color: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    g = np.zeros((10, 10), dtype=int)

    # Outer inverted shell.
    g[0, :] = int(outer_color)
    g[0:8, 0] = int(outer_color)
    g[0:8, 9] = int(outer_color)

    # Lower the ceiling of the outer side chambers so the inner basin owns the highest rows.
    g[1:6, 1:3] = int(outer_color)
    g[1:6, 7:9] = int(outer_color)

    # Leave only a central bottom shaft as the source opening.
    g[8:10, 0:4] = int(outer_color)
    g[8:10, 6:10] = int(outer_color)

    # Inner inverted basin nested inside the outer shell.
    g[1, 3:7] = int(inner_color)
    g[1:6, 3] = int(inner_color)
    g[1:6, 6] = int(inner_color)

    open_mask = np.logical_or(g == BG, g == WATER)
    component = _collect_component(open_mask, (9, 4), set())
    return g, component


def _make_output(base: np.ndarray, component: list[tuple[int, int]], water_count: int) -> np.ndarray:
    out = np.array(base, dtype=int).copy()
    for r, c in _fill_highest_rows(component, water_count, require_exact_level=True):
        out[r, c] = WATER
    return out


def _make_input(base: np.ndarray, component: list[tuple[int, int]], water_count: int) -> np.ndarray:
    inp = np.array(base, dtype=int).copy()
    for r, c in _component_source_order(component)[:water_count]:
        inp[r, c] = WATER
    return inp


def _layout_metrics(inp: np.ndarray, out: np.ndarray) -> dict[str, int]:
    inner_cells = [(r, c) for r in range(2, 6) for c in (4, 5)]
    outer_side_cells = [(r, c) for r in (6, 7) for c in range(1, 9) if c not in (4, 5)]
    full_equalization_rows = [(r, c) for r in (6, 7) for c in range(1, 9)]
    source_cells = [(r, c) for r in range(8, 10) for c in (4, 5)]

    inner_fill = sum(1 for cell in inner_cells if int(out[cell]) == WATER)
    outer_fill = sum(1 for cell in outer_side_cells if int(out[cell]) == WATER)
    equal_rows_fill = sum(1 for cell in full_equalization_rows if int(out[cell]) == WATER)
    source_input_fill = sum(1 for cell in source_cells if int(inp[cell]) == WATER)
    source_output_fill = sum(1 for cell in source_cells if int(out[cell]) == WATER)
    return {
        "water_total": int(np.count_nonzero(out == WATER)),
        "inner_fill": int(inner_fill),
        "inner_capacity": len(inner_cells),
        "outer_fill": int(outer_fill),
        "equal_rows_fill": int(equal_rows_fill),
        "source_input_fill": int(source_input_fill),
        "source_output_fill": int(source_output_fill),
        "moved": int(np.any(inp != out)),
    }


def main(grid: np.ndarray, check_input: bool = True, check_output: bool = True) -> np.ndarray:
    """
    Rebalance bottom-source water upward into an inverted nested vessel until a shared upper waterline is reached.

    Input structure:
    - bg = 0 and water = 1.
    - Other colors draw a large top-anchored outer shell and a smaller inverted basin nested inside it.
    - Water begins as a bottom-centered source mass inside the same enclosed cavity instead of hanging near the top.

    Rule:
    - For each open cavity containing water, treat background and water cells as one 4-connected vessel component.
    - The relevant component must touch the bottom edge, because this variant reverses gravity and the source rises upward into the container.
    - Count the water pixels in that component, clear them, and refill the same component from the highest rows downward.
    - Because the inner nested chamber has the highest ceiling, it fills first; once that capacity is exhausted, the lower connected outer chambers fill in equal-height rows like communicating vessels.

    Key insight: This is not local falling anymore. The answer comes from total water volume versus the row-by-row cross-section of one bottom-connected inverted vessel, so reverse gravity, nested topology, and equal-level redistribution are solved together.
    """
    g = np.array(grid, dtype=int).copy()
    h, w = g.shape
    initial_water = int(np.count_nonzero(g == WATER))

    if check_input:
        assert g.shape == (10, 10), "generator and solver both assume a 10x10 canvas"
        assert initial_water > 0, "input must contain water"

    open_mask = np.logical_or(g == BG, g == WATER)
    seen: set[tuple[int, int]] = set()
    for r, c in np.argwhere(g == WATER):
        start = (int(r), int(c))
        if start in seen:
            continue
        component = _collect_component(open_mask, start, seen)
        touches_bottom = any(rr == h - 1 for rr, _cc in component)
        touches_top_or_side = any(rr == 0 or cc in (0, w - 1) for rr, cc in component)
        if check_input:
            assert touches_bottom, "reverse-gravity vessel must connect to the bottom source"
            assert not touches_top_or_side, "inverted vessel cavity must stay closed on top and sides"
        if not touches_bottom or touches_top_or_side:
            continue

        water_count = sum(1 for rr, cc in component if int(g[rr, cc]) == WATER)
        filled = _fill_highest_rows(component, water_count, require_exact_level=check_input)
        for rr, cc in component:
            g[rr, cc] = BG
        for rr, cc in filled:
            g[rr, cc] = WATER

    if check_output:
        assert int(np.count_nonzero(g == WATER)) == initial_water, "water count must be conserved"
        open_mask = np.logical_or(g == BG, g == WATER)
        seen = set()
        for r, c in np.argwhere(g == WATER):
            rr = int(r)
            cc = int(c)
            assert rr == 0 or int(g[rr - 1, cc]) != BG, "output water must be supported from above"
            start = (rr, cc)
            if start in seen:
                continue
            component = _collect_component(open_mask, start, seen)
            touches_bottom = any(r0 == h - 1 for r0, _c0 in component)
            touches_top_or_side = any(r0 == 0 or c0 in (0, w - 1) for r0, c0 in component)
            assert touches_bottom, "output water disconnected from the bottom source cavity"
            assert not touches_top_or_side, "output water leaked out of the inverted vessel"
            filled = {(r0, c0) for r0, c0 in component if int(g[r0, c0]) == WATER}
            _assert_top_packed(component, filled)

    return g


def generate_input() -> np.ndarray:
    water_levels = [16, 24]
    solid_colors = [2, 3, 4, 5, 6, 7, 8, 9]

    for _ in range(100):
        outer_color, inner_color = random.sample(solid_colors, 2)
        base, component = _draw_case(outer_color=outer_color, inner_color=inner_color)
        water_count = random.choice(water_levels)
        target = _make_output(base, component, water_count)
        inp = _make_input(base, component, water_count)
        out = main(inp)
        metrics = _layout_metrics(inp, out)

        if not np.array_equal(out, target):
            continue
        if metrics["moved"] != 1:
            continue
        if metrics["inner_fill"] != metrics["inner_capacity"]:
            continue
        if metrics["outer_fill"] <= 0:
            continue
        if metrics["source_input_fill"] < 4:
            continue
        if metrics["source_output_fill"] != 0:
            continue

        return inp

    raise AssertionError("failed to generate a valid inverted nested communicating-vessel sample")
