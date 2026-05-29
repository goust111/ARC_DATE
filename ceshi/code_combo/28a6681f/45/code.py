# Combo variant of 28a6681f (index 45)
# combo_spec: D9/1+D10/2
# Name: 水花溅射标记+盆地壁染色

from __future__ import annotations

import random

import numpy as np


# Design analysis
# ---------------
# a) 具体化与行为追踪
# - 画面仍是 10x10 的“蓝水流入底部相邻盆地”家族：0 是背景，1 是水，2-3 种其他颜色构成
#   底部台阶、盆壁和隔板。输入里的蓝水仍放在左或右侧高台上方的受支撑水源区。
# - 融合后输出里有两种新增视觉结果：
#   1. 壁染色：最终与稳定水体左右接触的结构像素会被染成蓝色。
#   2. 水花标记：水在运动途中或最终落稳时，一旦“下方被支撑面挡住”，就在该碰撞点正上方
#      留下固定颜色 8 的水花；若最终与盆壁接触，则在该壁接触点上方也补一个同色水花。
# - 小例子：假设一滴水从左侧高台落入 2 列宽盆地。原题输出只会在盆底留下蓝水。
#   现在它在碰到底面那一层时，会在落点上方留下 8；若最后某一行水贴着右壁，那一格右壁本身
#   会变成 1，且其上方若是空白，也会出现一个 8。于是输出同时包含“蓝色真实水/染蓝壁”和
#   “8 色碰撞标记”，解题者需要区分二者角色。
#
# b) 复杂度校准
# - 物理规则仍保持原题单一局部规则，不新增第二套运动学，只把可视化结果扩展成两层：
#   蓝色壁染 + 8 色水花。这样变化显著，但推理核心仍围绕同一套落水过程。
# - 生成器只使用原 spec 中更稳定的“无 guide 的边缘高台 + 底部相邻盆地”子族，避免上方斜导轨
#   也被染色而削弱“盆壁染色”的主线。
# - 盆地数量保持 2 或 3，宽度只取 1-2 列，水量取少量离散值，并反复出现 overflow / 非 overflow
#   两类情形，让几组样例就能看出“蓝色壁线跟水位走，8 色水花跟碰撞点走”。
#
# c) 对原 main() 隐式假设的审查
# - 原 solver 默认“所有蓝色都是可继续运动的水”。融合后这不再成立，因为一部分蓝色是染蓝盆壁。
#   所以必须把流程拆成：先算真实稳定水位置，再做壁染色与水花渲染。
# - 原 solver 的输出校验会遍历所有蓝色。现在也不成立，因为染蓝壁不满足“左下右都非背景”的水稳定性。
#   因此校验只能针对单独跟踪的真实水位置。
# - 轨迹标记不能写回物理网格，否则会把 8 当成障碍，改变后续水路。故水花只在所有水稳定后渲染。
# - 原 `settle()` 只返回终点。这里需要额外记录“曾经被下方支撑挡住的格子”，否则无法从运动过程
#   生成水花痕迹。
#
# d) 放置合法性
# - 结构仍按底部连通列栈构造，所有实心列都锚定在底边，天然不越界。
# - 左侧高台的源水只放在受支撑格上，并保持至少一侧贴墙，因此满足原题输入合法性。
# - 盆宽、隔板宽、最右边保留空隙的采样范围与 D10 参考一致，保证列不重叠且总宽适配 10x10。
# - 颜色 8 仅用于输出水花，结构色显式排除 8，避免“输入里已经有水花色”造成歧义。
#
# e) 视觉可观察性
# - 染蓝壁像素沿着最终水位形成竖向或折线状蓝痕，这是“哪些墙被水真正贴到”的线索。
# - 8 色水花通常悬在碰撞点上方，既可能出现在盆底水面上方，也可能出现在被撞到的壁上方，
#   因而不是简单的“再多一层蓝水”，肉眼可见其与壁染色的几何关系不同。
# - 生成器拒绝掉只出现壁染色或只出现极少量水花的样本，确保 2-4 个例子里两种效果都反复出现。
#
# f) 可视化整洁度
# - 输出语义色保持很少：0 背景，1 水/染蓝壁，8 水花，外加 2-3 个普通结构色。
# - 水花永远只画在最终仍为空白的格子里，不覆盖真实水，也不覆盖结构，因此不会产生三层叠色。
# - 源区在上半部、盆地区在下半部，虽然中间可能有少量路径水花，但主视觉仍集中在底部盆地。
#
# g) 颜色变化的可推断性
# - 颜色 1 仍是“水相关”，但现在分成真实水和被染色的壁；其结构线索来自是否占据原先实心盆壁位置。
# - 颜色 8 只出现在原本空白处，且总贴着碰撞位置上方，样例中能归纳出“这是水花标记，不是结构改色”。
# - 不使用颜色大小、稀有度或排序规则，全部依据几何接触关系。
#
# h) 退化检查指标
# - 用 `(splash_count, dyed_wall_count, lower_splash_count)` 量化融合行为。
# - 生成器要求 `splash_count >= 3`、`dyed_wall_count >= 2`、`lower_splash_count >= 2`，并且
#   `main(input)` 必须不同于纯原题稳定水输出，避免退化成原题或仅 D10/2。
#
# i) main() 新增假设与 generator 保证
# - `_settle_water()` 假设输入里不存在预先画好的 8 色水花；generator 不会放置 8。
# - `_add_splashes()` 假设要渲染的水花目标格在最终图里仍是背景；该函数只在背景上画点。
# - `main()` 假设真正需要校验稳定性的只有 tracked water positions；generator 构造的输出始终满足。
# - 生成器还保证至少一部分水离开源高台并进入底盆，否则这两个新增通道都不够明显。


BG = 0
WATER = 1
SPLASH = 8
STRUCTURE_PALETTE = [2, 3, 4, 5, 6, 7, 9]
GRID_SIZE = 10


def _component_count(points: set[tuple[int, int]], use_diagonal: bool) -> int:
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
            r, c = stack.pop()
            for dr, dc in neighbors:
                nxt = (int(r + dr), int(c + dc))
                if nxt in remaining:
                    remaining.remove(nxt)
                    stack.append(nxt)
    return count


def _validate_input_and_guides(g: np.ndarray) -> None:
    h, w = g.shape
    assert not np.any(g == int(SPLASH)), "input must not already contain splash markers"

    for r, c in np.argwhere(g == int(WATER)):
        rr = int(r)
        cc = int(c)
        assert rr == h - 1 or int(g[rr + 1, cc]) != int(BG), "input water must be supported"
        left_bg = cc > 0 and int(g[rr, cc - 1]) == int(BG)
        right_bg = cc + 1 < w and int(g[rr, cc + 1]) == int(BG)
        assert not (left_bg and right_bg), "input water cannot have bg on both horizontal sides"

    bottom_colors = set(map(int, g[h - 1]))
    for color in sorted(set(map(int, g[:7].flatten())) - {BG, WATER} - bottom_colors):
        pts = {(int(r), int(c)) for r, c in np.argwhere(g[:7] == int(color))}
        diag_links = sum(1 for r, c in pts if (r + 1, c - 1) in pts or (r + 1, c + 1) in pts)
        if diag_links >= 3:
            assert any(c in (0, w - 1) for _r, c in pts), "guide must enter from a grid edge"
            assert _component_count(pts, False) == _component_count(pts, True), "guide must be 4-connected"


def _settle_water(
    grid: np.ndarray,
    check_input: bool = True,
    check_output: bool = True,
) -> tuple[np.ndarray, set[tuple[int, int]], set[tuple[int, int]]]:
    g = np.array(grid, dtype=int).copy()
    h, w = g.shape

    if check_input:
        _validate_input_and_guides(g)

    def settle(r: int, c: int) -> tuple[int, int, set[tuple[int, int]]]:
        rr = int(r)
        cc = int(c)
        impact_cells: set[tuple[int, int]] = set()
        moved_from_source = False

        for _ in range(h * w):
            if rr + 1 < h and int(g[rr + 1, cc]) == int(BG):
                rr += 1
                moved_from_source = True
                continue

            if moved_from_source:
                impact_cells.add((int(rr), int(cc)))

            opts = []
            if (
                rr + 1 < h
                and cc - 1 >= 0
                and int(g[rr, cc - 1]) == int(BG)
                and int(g[rr + 1, cc - 1]) == int(BG)
            ):
                opts.append((int(rr + 1), int(cc - 1)))
            if (
                rr + 1 < h
                and cc + 1 < w
                and int(g[rr, cc + 1]) == int(BG)
                and int(g[rr + 1, cc + 1]) == int(BG)
            ):
                opts.append((int(rr + 1), int(cc + 1)))
            if opts:
                assert len(opts) == 1, f"ambiguous diagonal moves: {opts}"
                rr, cc = opts[0]
                moved_from_source = True
                continue

            if rr + 1 < h and int(g[rr + 1, cc]) != int(BG):
                left = int(cc)
                while left - 1 >= 0 and int(g[rr, left - 1]) == int(BG):
                    left -= 1
                right = int(cc)
                while right + 1 < w and int(g[rr, right + 1]) == int(BG):
                    right += 1

                drops = [int(x) for x in range(int(left), int(right) + 1) if int(g[rr + 1, int(x)]) == int(BG)]
                if not drops:
                    break

                min_dist = min(abs(int(x - cc)) for x in drops)
                nearest = [int(x) for x in drops if abs(int(x - cc)) == int(min_dist)]
                assert len(nearest) == 1, f"ambiguous nearest drop columns: {nearest}"
                best = nearest[0]
                assert best != int(cc)
                step = -1 if best < int(cc) else 1
                nx = int(cc + step)
                assert int(g[rr, nx]) == int(BG)
                cc = nx
                moved_from_source = True
                continue

            break
        else:
            raise AssertionError("water pixel did not settle")

        return int(rr), int(cc), impact_cells

    all_impacts: set[tuple[int, int]] = set()
    for _pass in range(h * w):
        moved = False
        pts = sorted([(int(r), int(c)) for r, c in np.argwhere(g == int(WATER))], reverse=True)
        for r, c in pts:
            if int(g[r, c]) != int(WATER):
                continue
            g[r, c] = int(BG)
            rr, cc, impacts = settle(int(r), int(c))
            all_impacts.update(impacts)
            moved = bool(moved or ((int(rr), int(cc)) != (int(r), int(c))))
            g[rr, cc] = int(WATER)
        if not moved:
            break
    else:
        raise AssertionError("water simulation did not converge")

    water_positions = {(int(r), int(c)) for r, c in np.argwhere(g == int(WATER))}
    if check_output:
        for rr, cc in water_positions:
            assert cc == 0 or int(g[rr, cc - 1]) != int(BG)
            assert rr == h - 1 or int(g[rr + 1, cc]) != int(BG)
            assert cc == w - 1 or int(g[rr, cc + 1]) != int(BG)

    return g, water_positions, all_impacts


def _dye_contact_walls(
    settled: np.ndarray,
    water_positions: set[tuple[int, int]],
) -> tuple[np.ndarray, set[tuple[int, int]]]:
    out = np.array(settled, dtype=int).copy()
    h, w = out.shape
    dyed_walls: set[tuple[int, int]] = set()

    for rr, cc in water_positions:
        for dc in (-1, 1):
            nc = int(cc + dc)
            if 0 <= nc < w and int(out[rr, nc]) not in (int(BG), int(WATER), int(SPLASH)):
                out[rr, nc] = int(WATER)
                dyed_walls.add((int(rr), int(nc)))

    return out, dyed_walls


def _add_splashes(
    rendered: np.ndarray,
    impact_cells: set[tuple[int, int]],
    dyed_walls: set[tuple[int, int]],
) -> tuple[np.ndarray, set[tuple[int, int]]]:
    out = np.array(rendered, dtype=int).copy()
    h, _w = out.shape
    splash_cells: set[tuple[int, int]] = set()

    for rr, cc in impact_cells:
        sr = int(rr - 1)
        if 0 <= sr < h and int(out[sr, cc]) == int(BG):
            splash_cells.add((int(sr), int(cc)))

    for rr, cc in dyed_walls:
        sr = int(rr - 1)
        if 0 <= sr < h and int(out[sr, cc]) == int(BG):
            splash_cells.add((int(sr), int(cc)))

    for r, c in splash_cells:
        out[int(r), int(c)] = int(SPLASH)

    return out, splash_cells


def _apply_combo_effects(
    settled: np.ndarray,
    water_positions: set[tuple[int, int]],
    impact_cells: set[tuple[int, int]],
) -> tuple[np.ndarray, set[tuple[int, int]], set[tuple[int, int]]]:
    dyed, dyed_walls = _dye_contact_walls(settled, water_positions)
    out, splash_cells = _add_splashes(dyed, impact_cells, dyed_walls)
    return out, dyed_walls, splash_cells


def _fusion_signature(grid: np.ndarray) -> dict[str, int]:
    settled, water_positions, impact_cells = _settle_water(grid, check_input=True, check_output=True)
    out, dyed_walls, splash_cells = _apply_combo_effects(settled, water_positions, impact_cells)
    lower_splash_count = sum(1 for r, _c in splash_cells if int(r) >= 5)
    return {
        "water_count": int(len(water_positions)),
        "impact_count": int(len(impact_cells)),
        "dyed_wall_count": int(len(dyed_walls)),
        "splash_count": int(len(splash_cells)),
        "lower_splash_count": int(lower_splash_count),
        "delta_vs_original": int(np.sum(out != settled)),
    }


def main(grid: np.ndarray, check_input: bool = True, check_output: bool = True) -> np.ndarray:
    """
    Settle blue water into the lower basins, then render both impact splashes and wall staining.

    Input structure:
    - Background is 0 and moving water is color 1. The remaining 2-3 colors form bottom-anchored
      basin walls, dividers, shelves, and floors.
    - Water begins in a supported edge reservoir above the basin chain, so it can drain and sometimes
      overflow into neighboring cups.
    - The input contains only real water and solid structure; splash markers appear only in the output.

    Rule:
    - Water obeys the original local physics: fall straight through background, slip diagonally through
      the unique open side, and slide horizontally along supported rows toward the nearest unique drop.
    - While simulating, remember every moved water cell whose downward motion gets blocked by a support.
      After all water settles, place a splash marker (color 8) one cell above each remembered collision
      site whenever that cell is still background.
    - Also after settlement, recolor any solid wall pixel that touches water from the left or right to blue,
      and place another splash marker above that dyed wall contact when possible.

    Key insight: First recover the same stable water arrangement as in the original task; then separate two
    extra collision traces in the output: blue cells on former walls mark true wall contact, while color-8
    pixels hovering above impacts mark where the flow struck a support or wall.
    """
    settled, water_positions, impact_cells = _settle_water(
        grid,
        check_input=check_input,
        check_output=check_output,
    )
    out, _dyed_walls, _splash_cells = _apply_combo_effects(settled, water_positions, impact_cells)
    return out


def _build_left_structure() -> tuple[np.ndarray, dict[str, object]] | None:
    n_basins = random.choice([2, 2, 3])
    plateau_w = random.choice([2, 3, 3])
    basin_ws = [random.randint(1, 2) for _ in range(n_basins)]
    used = int(plateau_w + sum(basin_ws) + (n_basins - 1) + 1)
    tail_gap = GRID_SIZE - used
    if tail_gap < 0 or tail_gap > 2:
        return None

    plateau_top = random.randint(4, 6)
    outer_top = random.randint(4, 6)
    divider_tops = [random.randint(5, 7) for _ in range(n_basins - 1)]

    basin_floor_tops: list[list[int]] = []
    left_wall_top = int(plateau_top)
    for idx, width in enumerate(basin_ws):
        right_wall_top = int(divider_tops[idx] if idx < n_basins - 1 else outer_top)
        min_floor_top = int(max(left_wall_top, right_wall_top) + 1)
        if min_floor_top > 9:
            return None
        floor_top = random.randint(max(min_floor_top, 7), 9)
        tops = [int(floor_top) for _ in range(width)]
        if width == 2 and floor_top >= min_floor_top + 1 and random.random() < 0.6:
            tops[random.randint(0, 1)] -= 1
        basin_floor_tops.append(tops)
        left_wall_top = right_wall_top

    palette = random.sample(STRUCTURE_PALETTE, k=random.randint(2, 3))
    top_by_col = [GRID_SIZE] * GRID_SIZE
    color_by_col = [BG] * GRID_SIZE
    basins: list[dict[str, object]] = []
    col = 0

    for c in range(col, col + plateau_w):
        top_by_col[c] = int(plateau_top)
        color_by_col[c] = int(palette[0])
    col += plateau_w

    left_wall_top = int(plateau_top)
    color_idx = 1
    for idx, width in enumerate(basin_ws):
        right_wall_top = int(divider_tops[idx] if idx < n_basins - 1 else outer_top)
        basin_cols = list(range(col, col + width))
        for off, c in enumerate(basin_cols):
            top_by_col[c] = int(basin_floor_tops[idx][off])
            color_by_col[c] = int(palette[color_idx % len(palette)])
        capacity = sum(max(0, int(top_by_col[c] - max(left_wall_top, right_wall_top))) for c in basin_cols)
        basins.append({"cols": basin_cols, "capacity": int(capacity)})
        col += width
        color_idx += 1
        if idx < n_basins - 1:
            top_by_col[col] = int(divider_tops[idx])
            color_by_col[col] = int(palette[color_idx % len(palette)])
            col += 1
            color_idx += 1
            left_wall_top = right_wall_top

    top_by_col[col] = int(outer_top)
    color_by_col[col] = int(palette[color_idx % len(palette)])

    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    for c in range(GRID_SIZE):
        if top_by_col[c] < GRID_SIZE:
            grid[int(top_by_col[c]) :, c] = int(color_by_col[c])

    return grid, {
        "plateau_top": int(plateau_top),
        "plateau_w": int(plateau_w),
        "tail_gap": int(tail_gap),
        "basins": basins,
    }


def _place_source_water(grid: np.ndarray, plateau_top: int, source_w: int, water_n: int) -> np.ndarray | None:
    cells: list[tuple[int, int]] = []
    for r in range(int(plateau_top - 1), -1, -1):
        for c in range(int(source_w)):
            cells.append((int(r), int(c)))
    if water_n > len(cells):
        return None
    out = np.array(grid, dtype=int).copy()
    for r, c in cells[:water_n]:
        out[r, c] = int(WATER)
    return out


def _gen_attempt() -> tuple[np.ndarray, dict[str, int | bool]] | None:
    built = _build_left_structure()
    if built is None:
        return None
    structure, meta = built

    basin_caps = [int(basin["capacity"]) for basin in meta["basins"]]
    if not basin_caps or basin_caps[0] < 2:
        return None

    source_w = int(meta["plateau_w"])
    source_capacity = int(meta["plateau_top"] * source_w)
    total_capacity = int(sum(basin_caps))
    if total_capacity < 4:
        return None

    want_overflow = bool(len(basin_caps) >= 2 and basin_caps[1] > 0 and random.random() < 0.55)
    if want_overflow:
        low = max(4, basin_caps[0] + 1)
        high = min(source_capacity, total_capacity)
    else:
        low = 3
        high = min(source_capacity, basin_caps[0])
    if low > high:
        return None

    water_n = random.randint(int(low), int(high))
    grid = _place_source_water(structure, int(meta["plateau_top"]), source_w, int(water_n))
    if grid is None:
        return None

    try:
        settled, water_positions, impact_cells = _settle_water(grid, check_input=True, check_output=True)
    except AssertionError:
        return None

    output, dyed_walls, splash_cells = _apply_combo_effects(settled, water_positions, impact_cells)
    signature = {
        "dyed_wall_count": int(len(dyed_walls)),
        "splash_count": int(len(splash_cells)),
        "lower_splash_count": int(sum(1 for r, _c in splash_cells if int(r) >= 5)),
        "delta_vs_original": int(np.sum(output != settled)),
    }

    input_water = {(int(r), int(c)) for r, c in np.argwhere(grid == int(WATER))}
    output_cols = {int(c) for _r, c in water_positions}
    overflow_observed = any(any(int(c) in output_cols for c in basin["cols"]) for basin in meta["basins"][1:])

    if want_overflow != overflow_observed:
        return None
    if signature["dyed_wall_count"] < 2:
        return None
    if signature["splash_count"] < 3:
        return None
    if signature["lower_splash_count"] < 2:
        return None
    if signature["delta_vs_original"] < 5:
        return None
    if input_water == water_positions:
        return None
    if min(r for r, _c in water_positions) < 3:
        return None
    if np.array_equal(output, settled):
        return None

    try:
        if not np.array_equal(main(grid), output):
            return None
    except AssertionError:
        return None

    source_side = random.choice(["left", "right"])
    if source_side == "right":
        grid = np.fliplr(grid)

    return grid, {
        "source_side": bool(source_side == "right"),
        "want_overflow": bool(want_overflow),
        "overflow_observed": bool(overflow_observed),
        "dyed_wall_count": int(signature["dyed_wall_count"]),
        "splash_count": int(signature["splash_count"]),
        "lower_splash_count": int(signature["lower_splash_count"]),
    }


def generate_input() -> np.ndarray:
    for _ in range(500):
        attempt = _gen_attempt()
        if attempt is not None:
            grid, _meta = attempt
            return grid
    raise RuntimeError("Failed to generate valid D9/1+D10/2 combo variant after 500 attempts")
