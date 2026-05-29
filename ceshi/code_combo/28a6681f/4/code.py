# Combo variant of 28a6681f (index 4)
# combo_spec: D4/1+D11/1
# Name: 多色水源+按颜色分批处理

from __future__ import annotations

import random

import numpy as np

BG = 0
WATER_ORDER = (1, 2)  # 1 = blue, 2 = red
WATER_SET = frozenset(WATER_ORDER)
SOLID_COLORS = (3, 4, 5, 6, 7, 8, 9)
GRID_SHAPE = (10, 10)

# Design analysis
# ---------------
# a) 具体化与行为追踪
# - 保留原题的 10x10、底部锚定盆地链、下落/斜滑/水平寻坠口三段式水动力学。
# - D4 让可移动水变成两种固定颜色 1 和 2；D11 再规定它们不是“同时动”，而是按颜色批处理。
# - 本融合版固定顺序为蓝水 1 先处理，再处理红水 2。处理蓝水时，红水暂时不在世界里；等蓝水完全稳定后，
#   再把红水放回输入中的源位并单独沉降。这样蓝水会像“先灌进去的一层”那样占据深位，红水随后被蓝水挡住。
# - 具体小例子：右侧高台上有 4 列连续水源，靠坠口的两列是红水，远离坠口的两列是蓝水。
#   原始单色/同步多色逻辑会让靠近坠口的红水更早入盆，常常先占下层；融合逻辑则先删除红水、只让远侧蓝水
#   跨过红水原来的位置去下落，等蓝水铺到盆底后，再把红水放回高台，因此最终常见“蓝在下、红在上”的分层。
#
# b) 复杂度校准
# - 只用两种水色，避免把规则变成难以从少量样例归纳的多类排序题。
# - 地形仍是 2-3 个相邻盆地共享溢流的原题风格，能重复出现“先到者堵住后到者”的局部现象。
# - 水源模板固定为同一高台上的两段并排色块：这样颜色顺序有重复支撑，且输入中的两色都足够显眼。
#
# c) 隐式假设审查
# - 原版 `water = 1` 不成立；所有识别、排除、计数都必须覆盖 `WATER_ORDER`。
# - 原版“所有水一起迭代直到稳定”的外层循环不再正确；若直接保留，就退化成 D4 的同步多色版本。
# - 批处理要求“后续颜色在自己回合之前不占空间”，所以主求解必须先抽走所有水，再按颜色把源像素重新放回。
# - Guide 检查中原版只排除一个水色；融合后若不排除两种水色，会把红水误判成上部装饰/导流件。
# - 新增的隐式假设是：后处理颜色的输入源位在前一批稳定后仍然空着；否则无法把后一批重新放回。生成器会
#   专门筛掉这类退化样本，并把源区设计成高台出口，让先处理的蓝水通常离开源位。
#
# d) 放置合法性
# - 先画底部轮廓，再在边缘加一条与底色连通的高台；所有静态像素都在网格内且与水源不重叠。
# - 水源是连续矩形，所有水像素下方不是同色水就是高台支撑，因此满足原题的 supported 约束。
# - 左右朝向通过整体镜像得到，镜像后“蓝远、红近、共享同一坠口”的关系仍然成立。
#
# e) 视觉可观察性
# - 输入能直接看到两种水共处在同一高台源区，输出能直接看到同盆或相邻盆里的颜色分层。
# - 生成器只接受“批处理输出”和“同步多色输出”确实不同的样本，避免视觉上退化成普通多色水。
# - 还要求输出中两色相邻或位于同一连通水体里，这样人类能把“蓝先红后”的阻挡关系读出来。
#
# f) 可视化整洁度
# - 结构只保留底部盆地和单条边缘高台，不引入额外 guide/Funnel，避免多色水和上部装饰缠在一起。
# - 两种水在输入中只占同一条 4 列源带；观察者的注意力会集中在“谁先沉到底”而不是找对象。
#
# g) 变化的可推断性
# - 颜色没有重映射：蓝仍是 1，红仍是 2；变化只体现在批处理顺序与最终相对层次。
# - 样例里蓝总是先处理，且被放在离坠口更远的位置；如果仍然频繁出现在更低处，人类就能归纳出
#   规则不是“近者先落”，而是“按颜色分批，蓝先于红进入共享盆地”。


def _water_mask(arr: np.ndarray) -> np.ndarray:
    return np.isin(arr, tuple(WATER_ORDER))


def _water_points(arr: np.ndarray, colors: tuple[int, ...] | None = None) -> list[tuple[int, int]]:
    if colors is None:
        mask = _water_mask(arr)
    else:
        mask = np.isin(arr, tuple(int(color) for color in colors))
    return [(int(r), int(c)) for r, c in np.argwhere(mask)]


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
            r0, c0 = stack.pop()
            for dr, dc in neighbors:
                nxt = (int(r0 + dr), int(c0 + dc))
                if nxt in remaining:
                    remaining.remove(nxt)
                    stack.append(nxt)
    return count


def _check_guide_constraints(g: np.ndarray) -> None:
    h, w = g.shape
    bottom_colors = set(map(int, g[h - 1]))
    guide_candidates = sorted(set(map(int, g[:7].flatten())) - {BG} - WATER_SET - bottom_colors)
    for color in guide_candidates:
        pts = {(int(r), int(c)) for r, c in np.argwhere(g[:7] == int(color))}
        diag_links = sum(1 for r, c in pts if (r + 1, c - 1) in pts or (r + 1, c + 1) in pts)
        if diag_links >= 3:
            assert any(c in (0, w - 1) for _r, c in pts), "guide must enter from a grid edge"
            assert _component_count(pts, False) == _component_count(pts, True), "guide must be 4-connected"


def _validate_full_input(g: np.ndarray) -> None:
    h, w = g.shape
    for r, c in _water_points(g):
        rr = int(r)
        cc = int(c)
        assert rr == h - 1 or int(g[rr + 1, cc]) != BG, "input water must be supported"
        left_bg = cc > 0 and int(g[rr, cc - 1]) == BG
        right_bg = cc + 1 < w and int(g[rr, cc + 1]) == BG
        assert not (left_bg and right_bg), "input water cannot have bg on both horizontal sides"
    _check_guide_constraints(g)


def _check_stable_output(g: np.ndarray) -> None:
    h, w = g.shape
    for r, c in _water_points(g):
        rr = int(r)
        cc = int(c)
        assert cc == 0 or int(g[rr, cc - 1]) != BG
        assert rr == h - 1 or int(g[rr + 1, cc]) != BG
        assert cc == w - 1 or int(g[rr, cc + 1]) != BG


def _settle_cell(g: np.ndarray, r: int, c: int) -> tuple[int, int]:
    h, w = g.shape
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


def _simulate_colors_together(grid: np.ndarray, active_colors: tuple[int, ...]) -> np.ndarray:
    g = np.array(grid, dtype=int).copy()
    h, w = g.shape
    if not active_colors:
        return g

    for _pass in range(h * w):
        moved = False
        pts = sorted(
            [(int(r), int(c), int(g[int(r), int(c)])) for r, c in np.argwhere(np.isin(g, active_colors))],
            reverse=True,
        )
        for r, c, color in pts:
            if int(g[int(r), int(c)]) != int(color):
                continue
            g[int(r), int(c)] = BG
            rr, cc = _settle_cell(g, int(r), int(c))
            moved = bool(moved or ((int(rr), int(cc)) != (int(r), int(c))))
            assert int(g[int(rr), int(cc)]) == BG
            g[int(rr), int(cc)] = int(color)
        if not moved:
            break
    else:
        raise AssertionError("water simulation did not converge")

    return g


def _simulate_batched(grid: np.ndarray, check_input: bool = True, check_output: bool = True) -> np.ndarray:
    g = np.array(grid, dtype=int).copy()
    if check_input:
        _validate_full_input(g)

    source_points = {
        int(color): [(int(r), int(c)) for r, c in np.argwhere(g == int(color))]
        for color in WATER_ORDER
    }

    world = g.copy()
    world[_water_mask(world)] = BG

    for color in WATER_ORDER:
        pts = source_points[int(color)]
        if not pts:
            continue
        for r, c in pts:
            assert int(world[int(r), int(c)]) == BG, "later water source must stay empty until its batch"
            world[int(r), int(c)] = int(color)
        world = _simulate_colors_together(world, (int(color),))

    if check_output:
        _check_stable_output(world)
        for color in WATER_ORDER:
            assert int(np.count_nonzero(g == int(color))) == int(np.count_nonzero(world == int(color)))

    return world


def _simulate_interleaved(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int).copy()
    _validate_full_input(g)
    out = _simulate_colors_together(g, tuple(color for color in WATER_ORDER if np.any(g == int(color))))
    _check_stable_output(out)
    return out


def _build_profile(rng: random.Random) -> list[int | None]:
    outer = rng.randint(4, 5)
    separator = rng.randint(outer + 1, 7)
    left_base = rng.randint(separator + 1, 8)
    right_base = rng.randint(separator + 1, 8)

    left_cols = [left_base, max(separator + 1, left_base - rng.randint(0, 1))]
    right_cols = [right_base, max(separator + 1, right_base - rng.randint(0, 1))]
    if rng.random() < 0.5:
        left_cols.reverse()
    if rng.random() < 0.5:
        right_cols.reverse()

    return [None, outer, left_cols[0], left_cols[1], separator, separator, right_cols[0], right_cols[1], outer, None]


def _paint_profile(grid: np.ndarray, profile: list[int | None], palette: list[int]) -> None:
    for c, top in enumerate(profile):
        if top is None:
            continue
        color = int(palette[c % len(palette)])
        grid[int(top) :, int(c)] = color


def _add_batched_source(grid: np.ndarray, profile: list[int | None], palette: list[int], side: str, rng: random.Random) -> dict[str, object]:
    shelf_color = int(palette[0])
    if side == "right":
        outer_top = int(profile[8])
        source_cols = [6, 7, 8, 9]
        near_cols = {6, 7}
        far_cols = {8, 9}
    else:
        outer_top = int(profile[1])
        source_cols = [0, 1, 2, 3]
        near_cols = {2, 3}
        far_cols = {0, 1}

    source_bottom = rng.randint(max(2, outer_top - 2), outer_top - 1)
    support_row = int(source_bottom + 1)
    height = rng.randint(2, min(4, source_bottom + 1))
    top = int(source_bottom - height + 1)

    for c in source_cols:
        grid[support_row, c] = shelf_color

    for r in range(top, source_bottom + 1):
        for c in source_cols:
            assert int(grid[r, c]) == BG
            grid[r, c] = 1 if c in far_cols else 2

    return {
        "side": side,
        "source_cols": set(source_cols),
        "near_cols": set(near_cols),
        "far_cols": set(far_cols),
        "source_rows": set(range(top, source_bottom + 1)),
    }


def _water_components(grid: np.ndarray) -> list[set[tuple[int, int]]]:
    remaining = {(int(r), int(c)) for r, c in np.argwhere(_water_mask(grid))}
    components: list[set[tuple[int, int]]] = []
    while remaining:
        start = remaining.pop()
        comp = {start}
        stack = [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nxt = (int(r + dr), int(c + dc))
                if nxt in remaining:
                    remaining.remove(nxt)
                    comp.add(nxt)
                    stack.append(nxt)
        components.append(comp)
    return components


def _cross_color_contacts(grid: np.ndarray) -> int:
    contacts = 0
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color not in WATER_SET:
                continue
            for dr, dc in [(1, 0), (0, 1)]:
                rr = int(r + dr)
                cc = int(c + dc)
                if rr >= h or cc >= w:
                    continue
                other = int(grid[rr, cc])
                if other in WATER_SET and other != color:
                    contacts += 1
    return contacts


def _mean_row(grid: np.ndarray, color: int) -> float:
    pts = np.argwhere(grid == int(color))
    assert len(pts) > 0
    return float(np.mean(pts[:, 0]))


def _shared_component_has_both_colors(grid: np.ndarray) -> bool:
    for comp in _water_components(grid):
        colors = {int(grid[r, c]) for r, c in comp}
        if colors == WATER_SET:
            return True
    return False


def _effect_metrics(inp: np.ndarray, batched: np.ndarray, interleaved: np.ndarray, meta: dict[str, object]) -> dict[str, float | int]:
    source_cols = set(meta["source_cols"])
    input_water = int(np.count_nonzero(_water_mask(inp)))
    moved_outside_source = sum(1 for r, c in _water_points(batched) if int(c) not in source_cols)
    diff_cells = int(np.count_nonzero(batched != interleaved))
    blue_lower_gap = _mean_row(batched, 1) - _mean_row(batched, 2)
    blue_outside_source = int(sum(1 for r, c in np.argwhere(batched == 1) if int(c) not in source_cols))
    red_outside_source = int(sum(1 for r, c in np.argwhere(batched == 2) if int(c) not in source_cols))
    return {
        "input_water": input_water,
        "moved_outside_source": moved_outside_source,
        "diff_cells": diff_cells,
        "blue_lower_gap": blue_lower_gap,
        "blue_outside_source": blue_outside_source,
        "red_outside_source": red_outside_source,
        "cross_contacts": _cross_color_contacts(batched),
        "shared_component": int(_shared_component_has_both_colors(batched)),
    }


def _interesting_candidate(inp: np.ndarray, batched: np.ndarray, interleaved: np.ndarray, meta: dict[str, object]) -> bool:
    metrics = _effect_metrics(inp, batched, interleaved, meta)
    if metrics["diff_cells"] < 4:
        return False
    if metrics["moved_outside_source"] < max(4, metrics["input_water"] // 2):
        return False
    if metrics["blue_lower_gap"] < 0.35:
        return False
    if metrics["blue_outside_source"] < 3 or metrics["red_outside_source"] < 2:
        return False
    if not metrics["shared_component"]:
        return False
    if metrics["cross_contacts"] < 1:
        return False
    return True


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    grid = np.zeros(GRID_SHAPE, dtype=int)
    palette = list(rng.sample(SOLID_COLORS, k=rng.choice([2, 3])))
    profile = _build_profile(rng)
    _paint_profile(grid, profile, palette)
    meta = _add_batched_source(grid, profile, palette, rng.choice(["left", "right"]), rng)

    try:
        batched = _simulate_batched(grid)
        interleaved = _simulate_interleaved(grid)
    except AssertionError:
        return None

    if not _interesting_candidate(grid, batched, interleaved, meta):
        return None
    return grid


def main(grid: np.ndarray, check_input: bool = True, check_output: bool = True) -> np.ndarray:
    """
    Simulate two-color water in the original basin world, but process the colors as separate pours.

    Input structure:
    - bg = 0. Blue water is color 1 and red water is color 2; every other color is a static wall, floor, shelf, or basin divider.
    - The terrain still forms bottom-anchored neighboring basins of the original task family, often with an edge source shelf above them.
    - Both water colors can appear together in the same supported source region before either one has fully settled.

    Rule:
    - Ignore later batches while simulating earlier ones: first remove all unsimulated water, then place only blue water and let it settle with the original local rules.
    - After blue reaches a fixed point, keep that settled blue in place as a new obstacle, reinsert the red source pixels at their original positions, and settle red under the same physics.
    - Within each batch, movement priorities stay unchanged: fall vertically through bg, otherwise use the unique open diagonal, otherwise slide along a supported row toward the nearest unique drop.

    Key insight: This is no longer just "multi-colored water"; the colors encode a pouring schedule. Blue can claim deep cells before red even exists in the simulation, so the final layering depends on batch order rather than only on geometric proximity to the basin.
    """
    return _simulate_batched(grid, check_input=check_input, check_output=check_output)


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(800):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise AssertionError("failed to generate a visible batched multi-color sample")
