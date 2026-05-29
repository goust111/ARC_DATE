# Variant of 28a6681f
# Dimension: D4 — 水像素的颜色 (trajectory)
# Change: 所有水像素统一为蓝色（color 1），流动过程中颜色不变... → 多色水源

from __future__ import annotations

import random

import numpy as np

BG = 0
WATER_COLORS = (1, 2)
SOLID_COLORS = (3, 4, 5, 6, 7, 8, 9)
GRID_SHAPE = (10, 10)

# Design analysis
# ---------------
# a) 具体化与行为追踪
# - 仍然使用 10x10、底部锚定的连通盆地链；不改重力、对角滑落、水平寻坠口的物理。
# - 变体只改“水是什么颜色”：输入中的可移动水现在固定使用两种颜色 1 和 2。
# - 生成器只造一种高可观察性场景：边缘水库里上下两层是不同颜色，它们沿同一条路径进入共享盆地。
# - 原始逻辑：所有水都被当成 1，先到与后到没有颜色差异，最后只看到一团蓝水。
# - 变体逻辑：下层水因为更早接触到可达低位，会先占据深槽；上层另一色随后流入时会被这些已落位像素挡住，
#   因而停在更高或更外侧的位置。输出中同一连通水体里会同时出现两种颜色，并且深浅层次可见。
# - 具体小例子：右侧 2 列水库中，底部 4 行为红色、顶部 3 行为蓝色。红色先翻过分隔墙进入中间盆地并继续下沉；
#   当蓝色后来沿相同路径到达时，红色已经占住最深可达单元，所以蓝色只能停在红色上方或邻侧。
#
# b) 复杂度校准
# - 水色只用 2 种，避免把 D4 变成“颜色太多导致无法归纳”的任务。
# - 结构仍保持 3 个相邻盆地（源盆地 + 2 个接续盆地），这样多色竞争会反复发生，但自由度仍低。
# - 颜色先后不由颜色 id 决定，而由输入里“哪一色在更低的源层”决定；生成器会随机交换上下两色，避免隐藏规则。
#
# c) 原 main() 中依赖旧属性的代码审查
# - `water = 1`：旧版假设所有可动像素都是 1。变体下必须改成 `WATER_COLORS = (1, 2)`。
# - `np.argwhere(g == water)`：旧版只会移动蓝水；变体下要改成“扫描所有水色”。
# - `g[rr, cc] = water`：旧版落位时直接写回 1；变体下若保留会抹掉颜色身份，必须写回原像素自己的颜色。
# - `set(... ) - {bg, water}`：旧版从上部颜色集合里排除单一水色；变体下要排除全部水色，否则会把红水误判成 guide。
# - 输出稳定性检查同样要覆盖全部水色，而不是只看 1。
# - 生成器也必须避开把 2 用作静态结构色，否则 solver 无法区分“红水”和“红墙”。
#
# d) 放置合法性
# - 结构固定先在标准朝向下搭建，再整体镜像，以保证左右两种源向都合法。
# - 右侧源模板使用贴边且自下而上的连续水片，每个水像素都有下方支撑，且不会左右同时邻接背景。
# - 盆地分隔墙都自底部生长，不与水源重叠；镜像后这些合法性仍保持。
#
# e) 视觉可观察性
# - 每个样例都要求输出里两色水仍同时存在，并落在同一个 4 连通水体中。
# - 同时要求至少有一对正交相邻的异色水像素，这样“共享盆地且先来后到”能直接肉眼看到。
# - 生成器还要求至少若干水像素离开源盆地，防止退化成“只是换了水色但根本没发生竞争”。
#
# f) 可视化整洁度
# - 静态结构只保留底边、竖墙和少量颜色分段，不加上部 guide，避免和两色水叠得过乱。
# - 水源只放在单个边缘水库里，人的注意力可以集中在“颜色分层如何变成盆地分层”这一件事。
#
# g) 颜色变化可推断性
# - 没有颜色映射，也没有按颜色 id 排序；颜色只作为“水身份”被保留下来。
# - 同一颜色在不同样例里既可能先到也可能后到，因为上下两层的颜色会随机对调。
# - 因此人类能从少量样例归纳出：不是“蓝水总在下”，而是“先到的那层水占深位，而且保持原色”。


def main(grid: np.ndarray, check_input: bool = True, check_output: bool = True) -> np.ndarray:
    """
    Simulate multi-colored water pixels settling into basin chains under the original gravity rules.

    Input structure:
    - bg = 0. Movable water uses colors 1 and 2, while all other colors are static basin walls or floors.
    - Water starts as supported edge-packed sheets or short runs that can spill from one basin into the next.
    - The basin geometry is unchanged from the original task family: bottom-anchored dividers create neighboring cups.

    Rule:
    - Each water pixel keeps its own color while following the same movement priorities as before: fall vertically, otherwise use the unique open diagonal, otherwise slide along a supported row toward the nearest reachable drop.
    - Empty background cells are the only traversable cells; settled water of any color blocks later arrivals just like a wall.
    - All water pixels are repeatedly relocated bottom-up until no pixel can improve its position.

    Key insight: The shape logic is identical to the single-color task, but color now records arrival order inside the shared basins because earlier water claims the deepest reachable cells and later colors stack around it.
    """
    g = np.array(grid, dtype=int).copy()
    h, w = g.shape
    water_colors = set(WATER_COLORS)

    def is_water_value(value: int) -> bool:
        return int(value) in water_colors

    def water_points(arr: np.ndarray) -> list[tuple[int, int]]:
        return [(int(r), int(c)) for r, c in np.argwhere(np.isin(arr, WATER_COLORS))]

    if check_input:
        for r, c in water_points(g):
            rr = int(r)
            cc = int(c)
            assert rr == h - 1 or int(g[rr + 1, cc]) != BG, "input water must be supported"
            left_bg = cc > 0 and int(g[rr, cc - 1]) == BG
            right_bg = cc + 1 < w and int(g[rr, cc + 1]) == BG
            assert not (left_bg and right_bg), "input water cannot have bg on both horizontal sides"

    def component_count(points: set[tuple[int, int]], use_diagonal: bool) -> int:
        remaining = set(points)
        if use_diagonal:
            neighbors = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
        else:
            neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        count = 0
        while remaining:
            count += 1
            stack = [remaining.pop()]
            while stack:
                r0, c0 = stack.pop()
                for dr, dc in neighbors:
                    nxt = (int(r0 + dr), int(c0 + dc))
                    if nxt in remaining:
                        remaining.remove(nxt)
                        stack.append(nxt)
        return count

    bottom_colors = set(map(int, g[h - 1]))
    guide_candidates = sorted(set(map(int, g[:7].flatten())) - {BG} - water_colors - bottom_colors)
    for color in guide_candidates:
        pts = {(int(r), int(c)) for r, c in np.argwhere(g[:7] == int(color))}
        diag_links = sum(1 for r, c in pts if (r + 1, c - 1) in pts or (r + 1, c + 1) in pts)
        if diag_links >= 3:
            assert any(c in (0, w - 1) for _r, c in pts), "guide must enter from a grid edge"
            assert component_count(pts, False) == component_count(pts, True), "guide must be 4-connected"

    def settle(r: int, c: int) -> tuple[int, int]:
        rr = int(r)
        cc = int(c)
        for _ in range(h * w):
            if rr + 1 < h and int(g[rr + 1, cc]) == BG:
                rr += 1
                continue

            opts: list[tuple[int, int]] = []
            if (
                rr + 1 < h
                and cc - 1 >= 0
                and int(g[rr, cc - 1]) == BG
                and int(g[rr + 1, cc - 1]) == BG
            ):
                opts.append((int(rr + 1), int(cc - 1)))
            if (
                rr + 1 < h
                and cc + 1 < w
                and int(g[rr, cc + 1]) == BG
                and int(g[rr + 1, cc + 1]) == BG
            ):
                opts.append((int(rr + 1), int(cc + 1)))
            if opts:
                assert len(opts) == 1, f"ambiguous diagonal moves: {opts}"
                rr, cc = opts[0]
                continue

            if rr + 1 < h and int(g[rr + 1, cc]) != BG:
                left = int(cc)
                while left - 1 >= 0 and int(g[rr, left - 1]) == BG:
                    left -= 1
                right = int(cc)
                while right + 1 < w and int(g[rr, right + 1]) == BG:
                    right += 1

                drops = [int(x) for x in range(int(left), int(right) + 1) if int(g[rr + 1, int(x)]) == BG]
                if not drops:
                    break

                min_dist = min(abs(int(x - cc)) for x in drops)
                nearest = [int(x) for x in drops if abs(int(x - cc)) == int(min_dist)]
                assert len(nearest) == 1, f"ambiguous nearest drop columns: {nearest}"
                best = nearest[0]
                assert best != int(cc)
                step = -1 if best < int(cc) else 1
                nx = int(cc + step)
                assert int(g[rr, nx]) == BG
                cc = nx
                continue

            break
        else:
            raise AssertionError("water pixel did not settle")
        return int(rr), int(cc)

    for _pass in range(h * w):
        moved = False
        pts = sorted(
            [(int(r), int(c), int(g[int(r), int(c)])) for r, c in np.argwhere(np.isin(g, WATER_COLORS))],
            reverse=True,
        )
        for r, c, color in pts:
            if int(g[int(r), int(c)]) != int(color) or not is_water_value(int(color)):
                continue
            g[int(r), int(c)] = BG
            rr, cc = settle(int(r), int(c))
            moved = bool(moved or ((int(rr), int(cc)) != (int(r), int(c))))
            assert int(g[int(rr), int(cc)]) == BG
            g[int(rr), int(cc)] = int(color)
        if not moved:
            break
    else:
        raise AssertionError("water simulation did not converge")

    if check_output:
        for r, c in water_points(g):
            rr = int(r)
            cc = int(c)
            assert cc == 0 or int(g[rr, cc - 1]) != BG
            assert rr == h - 1 or int(g[rr + 1, cc]) != BG
            assert cc == w - 1 or int(g[rr, cc + 1]) != BG

    return g


def _build_canonical_candidate(rng: random.Random) -> tuple[np.ndarray, dict[str, object]]:
    grid = np.zeros(GRID_SHAPE, dtype=int)
    left_wall, mid_wall, outer_wall = 1, 4, 7
    left_color, mid_color, outer_color = rng.sample(SOLID_COLORS, 3)
    left_height = rng.randint(2, 4)
    mid_height = rng.randint(2, 4)
    outer_height = rng.randint(3, 5)

    for c in range(left_wall, mid_wall + 1):
        grid[9, c] = left_color
    for c in range(mid_wall, outer_wall + 1):
        grid[9, c] = mid_color

    for r in range(10 - left_height, 10):
        grid[r, left_wall] = left_color
    for r in range(10 - mid_height, 10):
        grid[r, mid_wall] = mid_color
    for r in range(10 - outer_height, 10):
        grid[r, outer_wall] = outer_color

    source_top = rng.randint(0, 4)
    narrow_rows = rng.randint(0, min(2, 9 - source_top))
    split_row = rng.randint(source_top + 2, 8)
    early_color, late_color = rng.sample(WATER_COLORS, 2)

    for r in range(source_top, 10):
        cols = [9] if r < source_top + narrow_rows else [8, 9]
        water_color = late_color if r < split_row else early_color
        for c in cols:
            assert int(grid[r, c]) == BG
            grid[r, c] = water_color

    mirror = bool(rng.randint(0, 1))
    if mirror:
        grid = np.fliplr(grid)

    source_cols = {0, 1} if mirror else {8, 9}
    return grid, {"source_cols": source_cols}


def _water_components(grid: np.ndarray) -> list[set[tuple[int, int]]]:
    points = {(int(r), int(c)) for r, c in np.argwhere(np.isin(grid, WATER_COLORS))}
    components: list[set[tuple[int, int]]] = []
    while points:
        start = points.pop()
        comp = {start}
        stack = [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nxt = (int(r + dr), int(c + dc))
                if nxt in points:
                    points.remove(nxt)
                    comp.add(nxt)
                    stack.append(nxt)
        components.append(comp)
    return components


def _cross_color_contacts(grid: np.ndarray) -> int:
    contacts = 0
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            color = int(grid[r, c])
            if color not in WATER_COLORS:
                continue
            for dr, dc in [(1, 0), (0, 1)]:
                rr = int(r + dr)
                cc = int(c + dc)
                if rr >= grid.shape[0] or cc >= grid.shape[1]:
                    continue
                other = int(grid[rr, cc])
                if other in WATER_COLORS and other != color:
                    contacts += 1
    return contacts


def _positions_by_color(grid: np.ndarray) -> dict[int, set[tuple[int, int]]]:
    return {
        color: {(int(r), int(c)) for r, c in np.argwhere(grid == int(color))}
        for color in WATER_COLORS
    }


def _variant_visible(inp: np.ndarray, out: np.ndarray, source_cols: set[int]) -> bool:
    in_pos = _positions_by_color(inp)
    out_pos = _positions_by_color(out)
    if not all(out_pos[color] for color in WATER_COLORS):
        return False

    moved = sum(len(in_pos[color] - out_pos[color]) for color in WATER_COLORS)
    if moved < 4:
        return False

    spilled = sum(1 for r, c in np.argwhere(np.isin(out, WATER_COLORS)) if int(c) not in source_cols)
    if spilled < 3:
        return False

    components = _water_components(out)
    shared_component = any({int(out[r, c]) for r, c in comp} == set(WATER_COLORS) for comp in components)
    if not shared_component:
        return False

    if _cross_color_contacts(out) < 1:
        return False

    return True


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(400):
        grid, meta = _build_canonical_candidate(rng)
        try:
            out = main(grid)
        except AssertionError:
            continue
        if _variant_visible(grid, out, set(meta["source_cols"])):
            return grid
    raise AssertionError("failed to generate a valid multi-color sample")
