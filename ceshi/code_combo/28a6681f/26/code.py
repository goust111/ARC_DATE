# Combo variant of 28a6681f (index 26)
# combo_spec: D4/1+D15/2
# Name: 已有水和新水颜色区分+多色水源

from __future__ import annotations

import random

import numpy as np


# Design analysis
# ---------------
# a) 具体化与行为追踪
# - 仍保持 10x10、底部锚定的相邻盆地链，以及原题同一套局部重力/斜滑/水平寻坠口物理。
# - 但输入中的“水”被拆成三类可见语义：
#   1. 深蓝 `1`：已经稳定、不会再移动的旧水；
#   2. 红色 `2`：第一批新水；
#   3. 绿色 `3`：第二批新水。
# - 关键不是“把 2 和 3 都当成普通水一起跑”，而是“先分别求出红水和绿水各自想落到哪里，再按固定颜色先后
#   叠加到同一个共享盆地上”。旧蓝水从一开始就占据盆底可达位置；红水 `2` 先认领它独立可达的低位；绿水 `3`
#   再求自己的目标格，但与红水冲突的目标格会被红水保留，绿水余量则停留在自己的源区。输出时三类水全部统一改成深蓝。
# - 小例子：左盆底部已有一层蓝水。若红源和绿源在“只看自己时”都会把最低两格当成落点，那么红源先认领
#   这两格后，绿源对应的两滴就不能再占它们，而会保留在自己的源区低位。若把左右源区的红/绿对调，最终哪一侧
#   被迫滞留也会跟着对调。原题里若把所有新水视作同一批蓝水同时流动，这种颜色驱动的先后差异会消失。
#
# b) 复杂度校准
# - 新水颜色只固定为两种，旧水只固定为一种，避免颜色角色太多而失去可归纳性。
# - 盆地仍只保留原题家族中最稳定的 2 个相邻盆地；变化集中在“旧水固定 + 两批新水顺序叠加”。
# - 为了让 D4 不退化，红 `2` 与绿 `3` 的先后语义在所有样例中保持固定，但 generator 会随机交换它们的左右位置。
#   只有当“只交换红绿、几何保持不变”时输出会随之改变，样本才会被接受；这样多色水源不是装饰，而会真实改变最终蓝水占位。
#
# c) 原 main() 隐式假设审查
# - `water = 1` 不再成立。现在 `1` 是不动旧水，`2/3` 才是待模拟的新水批次。
# - 原版 `np.argwhere(g == water)` 默认所有水都会移动；融合后必须只移动当前批次颜色。
# - 原版把非背景非水都当静态结构；融合后旧水虽然不动，但语义上仍是“已占据水位”，必须像障碍一样
#   支撑后来的新水。
# - 原版所有水一次性共同迭代；融合后若仍这样做，`2/3` 在输出统一染蓝后会退化成“只是输入换色”。
#   因此 solver 改成“两阶段”：先让每种新水在“结构 + 旧水”背景上各自独立求稳，再按固定颜色顺序认领共享低位。
# - 原版输出检查只需确认所有 `1` 稳定。融合后也只检查最终统一为 `1` 的蓝水，而不是中间批次色。
# - guide 候选色集合原来只需排除 `{0,1}`；现在必须排除全部水色 `{0,1,2,3}`，否则会把红/绿源水误判成 guide。
#
# d) 放置合法性
# - generator 只构造底部连通、列宽固定的合法盆地模板：三根底锚定立壁形成两个相邻盆地，不越界也不重叠。
# - 旧蓝水只放在盆地内部已稳定的整层位置，并且层数不超过相邻盆壁高度允许的上界。
# - 红/绿源区分别贴在左右外侧边缘，均是“每格下方有支撑、且不会左右两侧同时邻接背景”的合法原题水源。
# - 由于 solver 假设两批源区互不重叠，generator 只使用左右分离的两个源模块。
#
# e) 视觉可观察性
# - 输入里蓝/红/绿三类水同时出现且语义不同：蓝色已经在盆里，红绿还在源区。
# - 输出里所有新旧水统一成蓝色，所以解题者必须看懂：蓝色旧水不动，红绿两批各自按同一物理进入盆地。
# - generator 强制最终蓝水同时覆盖两侧盆地，并要求“分批顺序版输出”与“红绿合并同流版输出”不同，
#   这样从 2-4 个样例中能看出多色源区并不是只起到计数作用。
#
# f) 可视化整洁度
# - 语义色只有 4 类：背景 0、旧水/最终水 1、红新水 2、绿新水 3，其余是普通结构色。
# - 旧水集中在底部盆地，红绿集中在左右源区，区域职责清晰，不靠额外标记提示。
# - 盆地宽度固定为 2 列，旧水与新水叠层关系会形成清晰的矩形蓝水位，不会出现过细碎的噪声。
#
# g) 变化的可推断性
# - 红、绿两色的语义在全任务中保持固定：红 `2` 是先到批次，绿 `3` 是后到批次。generator 会在左右两侧
#   交换这两个批次的放置位置，使解题者能从多组样例里看出“不是左边总先，而是红色总先”。
# - 输出统一染蓝不是任意映射，而是“所有水最终都变成同一种稳定水色”；这一点会在每个样例里重复出现。
# - 同时保留旧蓝水不动、红绿消失并转蓝、最终蓝水位随两批叠加顺序变化，这三条线索一起支撑规则归纳。
#
# h) 退化检查指标
# - `priority_gap`：在同一输入上只把“红先绿后”改成“绿先红后”，输出蓝水占位改变了多少格；这个指标作为诊断保留，
#   但由于输出最终统一染蓝，它并不是每个合法样本都必须显式可见。
# - `stack_count`：最终新到达蓝水中，直接压在输入旧蓝水或前一批新水上的格子数。
# - generator 的可见性筛选要求两种新水都发生移动、旧蓝水确实支撑到至少一部分新蓝水、且最终蓝水同时出现在两个相邻盆区，
#   以保证融合后的三类水角色都在样本里反复出现。
#
# i) main() 新假设与 generator 保证
# - `_batch_order()` 假设颜色语义固定为 `2` 先于 `3`；generator 会随机交换这两个颜色的左右源区位置来提供样例支撑。
# - `_solve_sequential_batches()` 假设左右两批的源区彼此分离，且保留余量时仍能在各自源区底部稳定堆叠；
#   generator 用左右边缘竖向水库保证这一点。
# - `_validate_old_water()` 假设输入旧蓝水已经完全稳定；generator 只在完整盆层里预填旧水。
# - `_priority_reversed_output()` 只是退化检测辅助，不参与正式输出；generator 仅用它筛掉“改变批次先后也不影响结果”的样本。


BG = 0
OLD_WATER = 1
LEFT_NEW_WATER = 2
RIGHT_NEW_WATER = 3
NEW_WATER_COLORS = (LEFT_NEW_WATER, RIGHT_NEW_WATER)
ALL_WATER_COLORS = (OLD_WATER,) + NEW_WATER_COLORS
SOLID_COLORS = (4, 5, 6, 7, 8, 9)
GRID_SHAPE = (10, 10)
MAX_GENERATION_ATTEMPTS = 700

LEFT_WALL = 2
MID_WALL = 5
RIGHT_WALL = 7
LEFT_BASIN_COLS = (3, 4)
RIGHT_BASIN_COLS = (6,)


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


def _water_points(arr: np.ndarray, colors: tuple[int, ...]) -> list[tuple[int, int]]:
    return [(int(r), int(c)) for r, c in np.argwhere(np.isin(arr, colors))]


def _validate_old_water(g: np.ndarray) -> None:
    h, w = g.shape
    for r, c in np.argwhere(g == int(OLD_WATER)):
        rr = int(r)
        cc = int(c)
        assert cc == 0 or int(g[rr, cc - 1]) != int(BG), "old water must already be left-supported"
        assert rr == h - 1 or int(g[rr + 1, cc]) != int(BG), "old water must already be floor-supported"
        assert cc == w - 1 or int(g[rr, cc + 1]) != int(BG), "old water must already be right-supported"


def _validate_guides(g: np.ndarray) -> None:
    h, w = g.shape
    bottom_colors = set(map(int, g[h - 1]))
    guide_candidates = sorted(set(map(int, g[:7].flatten())) - {BG} - set(ALL_WATER_COLORS) - bottom_colors)
    for color in guide_candidates:
        pts = {(int(r), int(c)) for r, c in np.argwhere(g[:7] == int(color))}
        diag_links = sum(1 for r, c in pts if (r + 1, c - 1) in pts or (r + 1, c + 1) in pts)
        if diag_links >= 3:
            assert any(c in (0, w - 1) for _r, c in pts), "guide must enter from a grid edge"
            assert _component_count(pts, False) == _component_count(pts, True), "guide must be 4-connected"


def _validate_new_water_sources(g: np.ndarray, colors: tuple[int, ...]) -> None:
    h, w = g.shape
    for r, c in _water_points(g, colors):
        rr = int(r)
        cc = int(c)
        assert rr == h - 1 or int(g[rr + 1, cc]) != int(BG), "input water must be supported"
        left_bg = cc > 0 and int(g[rr, cc - 1]) == int(BG)
        right_bg = cc + 1 < w and int(g[rr, cc + 1]) == int(BG)
        assert not (left_bg and right_bg), "input water cannot have bg on both horizontal sides"


def _settle_only_color(g: np.ndarray, moving_color: int) -> np.ndarray:
    h, w = g.shape

    def settle(r: int, c: int) -> tuple[int, int]:
        rr = int(r)
        cc = int(c)
        for _ in range(h * w):
            if rr + 1 < h and int(g[rr + 1, cc]) == int(BG):
                rr += 1
                continue

            opts: list[tuple[int, int]] = []
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
                continue

            break
        else:
            raise AssertionError("water pixel did not settle")
        return int(rr), int(cc)

    for _pass in range(h * w):
        moved = False
        pts = sorted([(int(r), int(c)) for r, c in np.argwhere(g == int(moving_color))], reverse=True)
        for r, c in pts:
            if int(g[r, c]) != int(moving_color):
                continue
            g[r, c] = int(BG)
            rr, cc = settle(int(r), int(c))
            moved = bool(moved or ((int(rr), int(cc)) != (int(r), int(c))))
            assert int(g[rr, cc]) == int(BG)
            g[rr, cc] = int(moving_color)
        if not moved:
            break
    else:
        raise AssertionError("water simulation did not converge")

    return g


def _batch_order(g: np.ndarray) -> list[int]:
    return [int(color) for color in NEW_WATER_COLORS if np.any(g == int(color))]


def _assert_final_stability(g: np.ndarray) -> None:
    h, w = g.shape
    for r, c in np.argwhere(g == int(OLD_WATER)):
        rr = int(r)
        cc = int(c)
        assert rr == h - 1 or int(g[rr + 1, cc]) != int(BG)
        left_bg = cc > 0 and int(g[rr, cc - 1]) == int(BG)
        right_bg = cc + 1 < w and int(g[rr, cc + 1]) == int(BG)
        assert not (left_bg and right_bg), "final water cannot expose both horizontal sides"


def _solve_sequential_batches(
    grid: np.ndarray,
    *,
    check_input: bool,
    check_output: bool,
    order: list[int] | None = None,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    g = np.array(grid, dtype=int).copy()
    base = np.where(np.isin(g, NEW_WATER_COLORS), BG, g)
    occupied = base.copy()
    settled_masks: dict[int, np.ndarray] = {}

    if check_input:
        assert g.shape == GRID_SHAPE, f"expected a {GRID_SHAPE[0]}x{GRID_SHAPE[1]} grid"
        _validate_old_water(g)
        _validate_new_water_sources(g, NEW_WATER_COLORS)
        _validate_guides(g)

    batch_order = _batch_order(g) if order is None else [int(color) for color in order if np.any(g == int(color))]

    for color in batch_order:
        source_mask = g == int(color)
        batch_grid = base.copy()
        assert not np.any(source_mask & (batch_grid != BG)), "fresh sources must start in empty cells"
        batch_grid[source_mask] = int(color)
        batch_grid = _settle_only_color(batch_grid, int(color))
        solo_mask = batch_grid == int(color)
        final_mask = solo_mask & (occupied == BG)
        settled_masks[int(color)] = final_mask
        occupied[final_mask] = int(OLD_WATER)

    if check_output:
        _assert_final_stability(occupied)

    return occupied, settled_masks


def _collapsed_new_water_output(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int).copy()
    base = np.where(np.isin(g, NEW_WATER_COLORS), BG, g)
    batch_grid = base.copy()
    batch_grid[np.isin(g, NEW_WATER_COLORS)] = int(LEFT_NEW_WATER)
    batch_grid = _settle_only_color(batch_grid, int(LEFT_NEW_WATER))
    out = base.copy()
    out[batch_grid == int(LEFT_NEW_WATER)] = int(OLD_WATER)
    return out


def _priority_reversed_output(grid: np.ndarray) -> np.ndarray:
    out, _ = _solve_sequential_batches(
        np.asarray(grid, dtype=int),
        check_input=False,
        check_output=False,
        order=[int(RIGHT_NEW_WATER), int(LEFT_NEW_WATER)],
    )
    return out


def _moved_count(source_mask: np.ndarray, final_mask: np.ndarray) -> int:
    return int(np.count_nonzero(source_mask ^ final_mask) // 2)


def _stack_count(grid: np.ndarray, settled_masks: dict[int, np.ndarray]) -> int:
    old_mask = grid == int(OLD_WATER)
    left_mask = settled_masks.get(int(LEFT_NEW_WATER), np.zeros_like(grid, dtype=bool))
    right_mask = settled_masks.get(int(RIGHT_NEW_WATER), np.zeros_like(grid, dtype=bool))
    total = 0

    for r, c in np.argwhere(left_mask):
        rr = int(r)
        cc = int(c)
        if rr + 1 < grid.shape[0] and old_mask[rr + 1, cc]:
            total += 1

    support_mask = old_mask | left_mask
    for r, c in np.argwhere(right_mask):
        rr = int(r)
        cc = int(c)
        if rr + 1 < grid.shape[0] and support_mask[rr + 1, cc]:
            total += 1

    return int(total)


def _sequential_signature(grid: np.ndarray) -> dict[str, int]:
    out, settled_masks = _solve_sequential_batches(grid, check_input=False, check_output=False)
    try:
        reversed_out = _priority_reversed_output(grid)
        priority_gap = int(np.count_nonzero((out == int(OLD_WATER)) ^ (reversed_out == int(OLD_WATER))))
    except AssertionError:
        priority_gap = 0
    left_moved = _moved_count(grid == int(LEFT_NEW_WATER), settled_masks.get(int(LEFT_NEW_WATER), np.zeros_like(grid, dtype=bool)))
    right_moved = _moved_count(grid == int(RIGHT_NEW_WATER), settled_masks.get(int(RIGHT_NEW_WATER), np.zeros_like(grid, dtype=bool)))
    left_basin_fill = int(np.count_nonzero((out == int(OLD_WATER))[:, LEFT_BASIN_COLS]))
    right_basin_fill = int(np.count_nonzero((out == int(OLD_WATER))[:, RIGHT_BASIN_COLS]))
    old_support_count = 0
    old_mask = grid == int(OLD_WATER)
    for mask in settled_masks.values():
        for r, c in np.argwhere(mask):
            rr = int(r)
            cc = int(c)
            if rr + 1 < grid.shape[0] and old_mask[rr + 1, cc]:
                old_support_count += 1
    return {
        "priority_gap": int(priority_gap),
        "left_moved": int(left_moved),
        "right_moved": int(right_moved),
        "stack_count": int(_stack_count(grid, settled_masks)),
        "old_support_count": int(old_support_count),
        "left_basin_fill": int(left_basin_fill),
        "right_basin_fill": int(right_basin_fill),
    }


def main(grid: np.ndarray, check_input: bool = True, check_output: bool = True) -> np.ndarray:
    """
    Settle two visible batches of new water onto an already stable blue water level inside shared basins.

    Input structure:
    - Background is 0 on a 10x10 grid.
    - Deep blue `1` marks old water that is already stable and must stay in place.
    - Red `2` and green `3` are two fresh source batches, while every other non-zero color is a fixed wall, floor, or guide.
    - The lower half still contains bottom-anchored neighboring basins; the fresh colors begin in supported source regions above or beside them.

    Rule:
    - Keep the deep-blue old water fixed for the whole transformation and treat it as occupied support.
    - Read the fresh colors as separate pours in color order: red `2` arrives before green `3`. For each batch, first solve its own preferred resting cells against the static structure plus old water using the original local physics:
      fall straight down through background, otherwise use the unique open diagonal, otherwise slide horizontally toward the nearest reachable drop.
    - Then let earlier batches claim shared resting cells first. Any later cells whose preferred targets were already claimed stay behind in that batch's supported source region. After both batches are resolved, recolor every water cell to deep blue.

    Key insight: The task is no longer "move all water-colored pixels together." The solver must distinguish old blue water from new colored water, and must also keep the red and green pours separate so red claims shared deep cells first while green is visibly blocked and remains higher or closer to its source.
    """
    out, _settled_masks = _solve_sequential_batches(
        np.asarray(grid, dtype=int),
        check_input=bool(check_input),
        check_output=bool(check_output),
    )
    return out


def _draw_basin_chain(rng: random.Random) -> tuple[np.ndarray, dict[str, int]]:
    g = np.zeros(GRID_SHAPE, dtype=int)
    left_color, mid_color, right_color = rng.sample(SOLID_COLORS, 3)
    left_height = int(rng.randint(2, 4))
    mid_height = int(rng.randint(2, 4))
    right_height = int(rng.randint(3, 5))

    for c in range(LEFT_WALL, MID_WALL + 1):
        g[9, c] = int(left_color)
    for c in range(MID_WALL, RIGHT_WALL + 1):
        g[9, c] = int(mid_color)

    for r in range(10 - left_height, 10):
        g[r, LEFT_WALL] = int(left_color)
    for r in range(10 - mid_height, 10):
        g[r, MID_WALL] = int(mid_color)
    for r in range(10 - right_height, 10):
        g[r, RIGHT_WALL] = int(right_color)

    return g, {"left_height": left_height, "mid_height": mid_height, "right_height": right_height}


def _fill_old_water(
    g: np.ndarray,
    *,
    basin: str,
    depth: int,
) -> None:
    cols = LEFT_BASIN_COLS if basin == "left" else RIGHT_BASIN_COLS
    for level in range(int(depth)):
        row = 8 - int(level)
        for c in cols:
            assert int(g[row, c]) == int(BG), "old water must be placed into empty basin cells"
            g[row, c] = int(OLD_WATER)


def _add_left_source(g: np.ndarray, top_row: int, color: int) -> None:
    for r in range(int(top_row), 10):
        for c in [0, 1]:
            assert int(g[r, c]) == int(BG), "left source overlaps an occupied cell"
            g[r, c] = int(color)


def _add_right_source(g: np.ndarray, top_row: int, narrow_rows: int, color: int) -> None:
    for r in range(int(top_row), 10):
        for c in [8, 9]:
            assert int(g[r, c]) == int(BG), "right source overlaps an occupied cell"
            g[r, c] = int(color)


def _pick_old_water_config(rng: random.Random, heights: dict[str, int]) -> tuple[str, int]:
    choices: list[tuple[str, int]] = []
    left_max = min(int(heights["left_height"]), int(heights["mid_height"])) - 1
    right_max = min(int(heights["mid_height"]), int(heights["right_height"])) - 1
    for basin, max_depth in [("left", left_max), ("right", right_max)]:
        for depth in range(1, min(2, max_depth) + 1):
            choices.append((basin, int(depth)))
    if not choices:
        raise AssertionError("no stable old-water configuration fits the basin heights")
    return rng.choice(choices)


def _candidate_visible(sig: dict[str, int]) -> bool:
    if sig["left_moved"] < 2 or sig["right_moved"] < 2:
        return False
    if sig["stack_count"] < 2:
        return False
    if sig["old_support_count"] < 1:
        return False
    if sig["left_basin_fill"] == 0 or sig["right_basin_fill"] == 0:
        return False
    return True


def generate_input() -> np.ndarray:
    rng = random.Random()

    for _ in range(MAX_GENERATION_ATTEMPTS):
        g, heights = _draw_basin_chain(rng)
        basin, depth = _pick_old_water_config(rng, heights)
        _fill_old_water(g, basin=basin, depth=depth)

        left_wall_top = 10 - int(heights["left_height"])
        right_wall_top = 10 - int(heights["right_height"])
        left_top = int(rng.randint(0, max(0, left_wall_top - 1)))
        right_top = int(rng.randint(0, max(0, right_wall_top - 2)))
        narrow_rows = 0

        if rng.random() < 0.5:
            left_color, right_color = LEFT_NEW_WATER, RIGHT_NEW_WATER
        else:
            left_color, right_color = RIGHT_NEW_WATER, LEFT_NEW_WATER

        _add_left_source(g, left_top, left_color)
        _add_right_source(g, right_top, narrow_rows, right_color)

        try:
            out = main(g)
        except AssertionError:
            continue

        sig = _sequential_signature(g)
        if not _candidate_visible(sig):
            continue

        return g

    raise AssertionError("failed to generate a valid fused old-water/multi-source sample")
