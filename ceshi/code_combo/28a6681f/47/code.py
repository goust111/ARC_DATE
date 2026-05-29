# Combo variant of 28a6681f (index 47)
# combo_spec: D13/2+D15/0
# Name: 按距离盆底排序+无支撑约束

from __future__ import annotations

import random

import numpy as np


# Design analysis for combo D13/2 + D15/0
# ----------------------------------------
# a) 具体化与行为追踪
# - 画面仍是 10x10、0 背景、1 水、其余颜色为静态盆壁。
# - 保留原题的局部物理：水仍然只会直落、唯一对角滑落、以及在有支撑时沿当前行朝最近缺口水平滑动。
# - 真正改动的地方有两个：
#   1. 输入水不再要求“下方有支撑”。上半区可以直接出现悬空水滴团。
#   2. 每一轮不再按坐标降序处理，而是按“到最近盆底候选格的曼哈顿距离”从近到远处理。
# - 这里把“盆底候选格”具体定义为：下半区中、下方已被墙/边界挡住，且同一行左右两侧都能看见至少一个固体的背景格。
#   这样它们是人眼可见的“槽底/台底”，不会把完全敞开的外侧地面误当成盆底。
# - 小例子：模板里 `(6,9)` 这颗悬空水离右侧高台更近，而 `(0,7)` 等更高的水离底部更远。
#   原始 bottom-right 顺序会先处理右下团块，让它们先把低位口袋占满；现在改成距离排序后，最接近槽底的悬空水先落位，
#   较远水后处理，于是至少有 1-2 格最终停在更高一层，输出发生可见变化。
#
# b) 复杂度校准
# - 只保留一个小族模板：同一类右侧悬空水 + 下方窄口袋地形。这样人类能从 2-4 个样例里反复看到同一个排序规律。
# - 距离度量固定为曼哈顿距离，不依赖颜色 id，也不依赖不可视的对象编号。
# - 输入里至少有 4 个真正“下方是背景”的悬空水像素，且水像素会落在 4 个以上不同距离带上，避免退化成“只是换了个 tie-break”。
#
# c) 对原 main() 的隐式假设审查
# - 原 `check_input` 的 `input water must be supported` 在这里失效，必须删除；否则合法悬空源会被直接拒绝。
# - 原主循环默认按 `(row,col)` 降序扫描，这个前提正是 D13/2 要替换的对象。现在每轮都要重新依据当前坐标计算距盆底距离。
# - 原输出断言要求左右下都被挡住。距离排序后最终水仍应是稳定的，但稳定不一定能仅靠局部三向阻挡表达；这里改成
#   “删掉任意一滴后再次调用 settle，它仍回到原位”的动态稳定性检查。
# - 原 guide 检查与局部物理并不冲突，可以保留；本生成器只用下半区固体，不会触发额外歧义。
#
# d) 放置合法性
# - 所有模板都是字面量 10x10 位图，没有越界风险。
# - 水像素只放在模板预留的背景格中；镜像和重着色都不会造成水/墙重叠。
# - 结构只有 2 个固体标签，重着色时只替换这些标签，不会碰水和背景。
#
# e) 视觉可观察性
# - 输入里会明显看到“悬空水团”而不是原题那种贴墙支撑水段，因此 D15/0 在输入层面直接可见。
# - 输出再与原始 bottom-right 求解器做对比时会产生至少 2 格差异，说明排序真的在起作用，而不是空改输入约束。
# - 距离近的水通常更靠近下方槽底或高台口袋，人类可以从例子里看出“近处先占位、远处后落下”的规律。
#
# f) 可视化整洁度
# - 画面只分成两层：上部悬空水团、下部窄槽盆地。没有额外标记或轨迹色。
# - 固体颜色随机但语义固定，背景大面积留空，输出仍然干净。
#
# g) 变化的可推断性
# - 不使用颜色映射和颜色排序。唯一决定顺序的量是几何距离，这在像素网格上可见。
# - 生成器会统计：悬空水数量、距离带数量、以及新旧 solver 输出差异；若这些指标不足，样本直接丢弃。
#
# h) 量化退化检查目标
# - `unsupported_count >= 4`：保证“无支撑约束”真的出现在输入里。
# - `distance_bands >= 4`：保证排序不是只在一两个等距像素上靠 tie-break 生效。
# - `order_gap >= 2`：保证本 solver 与原始 bottom-right solver 在同一输入上有可见分叉。
#
# i) 新 solver 假设与 generator 保证
# - solver 假设下半区存在至少一个可识别的盆底候选格；模板几何保证这一点。
# - solver 假设局部对角/水平移动始终唯一；模板是从可运行原型里筛出来的，不会触发二义性断言。
# - solver 假设颜色只代表固体，不编码顺序；生成器每次随机重着色以防颜色值泄漏。


BG = 0
WATER = 1
H = 10
W = 10
SOLID_POOL = (2, 3, 4, 5, 6, 7, 8, 9)


_BASE_TEMPLATES: tuple[np.ndarray, ...] = (
    np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 3, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 3, 0, 0, 0],
            [0, 0, 3, 0, 0, 0, 4, 0, 0, 0],
            [0, 0, 3, 0, 0, 4, 3, 0, 0, 0],
        ],
        dtype=int,
    ),
    np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 3, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 3, 0, 0, 0],
            [0, 0, 3, 0, 0, 0, 4, 0, 0, 0],
            [0, 0, 3, 0, 0, 4, 3, 0, 0, 0],
        ],
        dtype=int,
    ),
    np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 3, 1, 0, 1],
            [0, 0, 0, 0, 0, 0, 3, 0, 0, 0],
            [0, 0, 3, 0, 0, 0, 4, 0, 0, 0],
            [0, 0, 3, 0, 0, 4, 3, 0, 0, 0],
        ],
        dtype=int,
    ),
    np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 3, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 3, 0, 0, 0],
            [0, 0, 3, 0, 0, 0, 4, 0, 0, 0],
            [0, 0, 3, 0, 0, 4, 3, 0, 0, 0],
        ],
        dtype=int,
    ),
)


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


def _check_optional_guides(g: np.ndarray) -> None:
    bottom_colors = set(map(int, g[H - 1]))
    for color in sorted(set(map(int, g[:7].flatten())) - {BG, WATER} - bottom_colors):
        pts = {(int(r), int(c)) for r, c in np.argwhere(g[:7] == int(color))}
        diag_links = sum(1 for r, c in pts if (r + 1, c - 1) in pts or (r + 1, c + 1) in pts)
        if diag_links >= 3:
            assert any(c in (0, W - 1) for _r, c in pts), "guide must enter from a grid edge"
            assert _component_count(pts, False) == _component_count(pts, True), "guide must be 4-connected"


def _row_has_solid_left(structure: np.ndarray, r: int, c: int) -> bool:
    return any(int(structure[r, x]) != int(BG) for x in range(0, int(c)))


def _row_has_solid_right(structure: np.ndarray, r: int, c: int) -> bool:
    return any(int(structure[r, x]) != int(BG) for x in range(int(c) + 1, W))


def _basin_bottom_cells(structure: np.ndarray) -> list[tuple[int, int]]:
    """
    Infer visible basin-floor cells from geometry alone.

    Preferred candidates are lower-half bg cells whose below side is blocked and whose
    row contains solid material on both the left and the right, so they read as floor
    cells inside a trough rather than as open exterior ground.
    """
    preferred: list[tuple[int, int]] = []
    fallback: list[tuple[int, int]] = []
    for r in range(H // 2, H):
        for c in range(W):
            if int(structure[r, c]) != int(BG):
                continue
            below_blocked = bool(r == H - 1 or int(structure[r + 1, c]) != int(BG))
            if not below_blocked:
                continue
            fallback.append((int(r), int(c)))
            if _row_has_solid_left(structure, int(r), int(c)) and _row_has_solid_right(structure, int(r), int(c)):
                preferred.append((int(r), int(c)))
    bottoms = preferred if preferred else fallback
    assert bottoms, "structure must expose at least one basin-bottom candidate"
    return sorted(set(bottoms))


def _distance_to_bottom(cell: tuple[int, int], bottoms: list[tuple[int, int]]) -> int:
    r, c = cell
    return min(abs(int(r - br)) + abs(int(c - bc)) for br, bc in bottoms)


def _settle(g: np.ndarray, r: int, c: int) -> tuple[int, int]:
    rr = int(r)
    cc = int(c)
    for _ in range(H * W):
        if rr + 1 < H and int(g[rr + 1, cc]) == int(BG):
            rr += 1
            continue

        opts: list[tuple[int, int]] = []
        if (
            rr + 1 < H
            and cc - 1 >= 0
            and int(g[rr, cc - 1]) == int(BG)
            and int(g[rr + 1, cc - 1]) == int(BG)
        ):
            opts.append((int(rr + 1), int(cc - 1)))
        if (
            rr + 1 < H
            and cc + 1 < W
            and int(g[rr, cc + 1]) == int(BG)
            and int(g[rr + 1, cc + 1]) == int(BG)
        ):
            opts.append((int(rr + 1), int(cc + 1)))
        if opts:
            assert len(opts) == 1, f"ambiguous diagonal moves: {opts}"
            rr, cc = opts[0]
            continue

        if rr + 1 < H and int(g[rr + 1, cc]) != int(BG):
            left = int(cc)
            while left - 1 >= 0 and int(g[rr, left - 1]) == int(BG):
                left -= 1
            right = int(cc)
            while right + 1 < W and int(g[rr, right + 1]) == int(BG):
                right += 1

            drops = [int(x) for x in range(left, right + 1) if int(g[rr + 1, int(x)]) == int(BG)]
            if not drops:
                break

            min_dist = min(abs(int(x - cc)) for x in drops)
            nearest = [int(x) for x in drops if abs(int(x - cc)) == int(min_dist)]
            assert len(nearest) == 1, f"ambiguous nearest drop columns: {nearest}"
            best = nearest[0]
            assert best != int(cc)
            cc += -1 if best < int(cc) else 1
            continue

        break
    else:
        raise AssertionError("water pixel did not settle")
    return int(rr), int(cc)


def _check_output_water(g: np.ndarray) -> None:
    for r, c in np.argwhere(g == int(WATER)):
        rr = int(r)
        cc = int(c)
        tmp = g.copy()
        tmp[rr, cc] = int(BG)
        nr, nc = _settle(tmp, rr, cc)
        assert (nr, nc) == (rr, cc), "output water must be dynamically stable"


def _simulate(
    grid: np.ndarray,
    *,
    use_distance_order: bool,
    check_input: bool,
    check_output: bool,
) -> np.ndarray:
    g = np.array(grid, dtype=int).copy()
    assert g.shape == (H, W), f"expected 10x10 grid, got {g.shape}"

    if check_input:
        assert np.any(g == int(WATER)), "input must contain water"
        _check_optional_guides(g)

    structure = np.where(g == int(WATER), int(BG), g)
    bottoms = _basin_bottom_cells(structure)

    for _pass in range(H * W):
        moved = False
        pts = [(int(r), int(c)) for r, c in np.argwhere(g == int(WATER))]
        if use_distance_order:
            pts = sorted(pts, key=lambda rc: (_distance_to_bottom(rc, bottoms), rc[0], rc[1]))
        else:
            pts = sorted(pts, reverse=True)

        for r, c in pts:
            if int(g[r, c]) != int(WATER):
                continue
            g[r, c] = int(BG)
            rr, cc = _settle(g, int(r), int(c))
            moved = bool(moved or ((rr, cc) != (r, c)))
            assert int(g[rr, cc]) == int(BG), "water paths should not overlap"
            g[rr, cc] = int(WATER)
        if not moved:
            break
    else:
        raise AssertionError("water simulation did not converge")

    if check_output:
        _check_output_water(g)
    return g


def _count_unsupported(grid: np.ndarray) -> int:
    count = 0
    for r, c in np.argwhere(grid == int(WATER)):
        rr = int(r)
        cc = int(c)
        if rr + 1 < H and int(grid[rr + 1, cc]) == int(BG):
            count += 1
    return int(count)


def _recolor_solids(template: np.ndarray, rng: random.Random) -> np.ndarray:
    g = np.array(template, dtype=int).copy()
    labels = sorted(set(map(int, g.flatten())) - {BG, WATER})
    colors = rng.sample(list(SOLID_POOL), len(labels))
    mapping = {label: color for label, color in zip(labels, colors)}
    for src, dst in mapping.items():
        g[g == int(src)] = int(dst)
    return g


def _behavior_metrics(grid: np.ndarray) -> dict[str, int]:
    structure = np.where(grid == int(WATER), int(BG), grid)
    bottoms = _basin_bottom_cells(structure)
    distances = [_distance_to_bottom((int(r), int(c)), bottoms) for r, c in np.argwhere(grid == int(WATER))]
    combo = _simulate(grid, use_distance_order=True, check_input=True, check_output=True)
    original = _simulate(grid, use_distance_order=False, check_input=True, check_output=True)
    return {
        "unsupported_count": int(_count_unsupported(grid)),
        "distance_bands": int(len(set(map(int, distances)))),
        "order_gap": int(np.count_nonzero(combo != original)),
        "moved_water": int(np.count_nonzero((grid == int(WATER)) != (combo == int(WATER))) // 2),
    }


def main(grid: np.ndarray, check_input: bool = True, check_output: bool = True) -> np.ndarray:
    """
    Simulate basin-filling water with unsupported sources, processing the closest droplets first.

    Input structure:
    - Background is 0 and water is 1.
    - Other colors are rigid basin walls and shelves concentrated in the lower half.
    - Source water may hang freely in open air; it no longer needs direct support from below or side packing.

    Rule:
    - Infer the visible basin-floor cells from the static structure alone, using lower-half background cells
      whose below side is blocked and that sit between solids on the same row as the preferred basin bottoms.
    - In each simulation pass, sort all current water pixels by Manhattan distance to the nearest such basin-bottom
      cell, from near to far. Then settle each selected pixel with the original local motion: straight fall,
      unique diagonal slip, and supported horizontal sliding toward the unique nearest drop.
    - Repeat passes until no water moves. Structure stays fixed and water count is conserved.

    Key insight:
    - The physics of one droplet is unchanged, but the multi-particle interaction is now governed by geometric
      proximity to the basin floor. Allowing suspended sources makes that distance ordering visually meaningful:
      closer floating water claims low slots before farther water gets its turn.
    """
    return _simulate(
        grid,
        use_distance_order=True,
        check_input=bool(check_input),
        check_output=bool(check_output),
    )


def generate_input() -> np.ndarray:
    """
    Return a legal sample that visibly exercises both combo dimensions.

    Acceptance criteria:
    - at least 4 input water pixels are unsupported,
    - source pixels occupy at least 4 distinct basin-distance bands,
    - distance ordering changes the output versus the original bottom-right scan.
    """
    rng = random.Random()
    for _ in range(300):
        inp = np.array(rng.choice(_BASE_TEMPLATES), dtype=int)
        if rng.random() < 0.5:
            inp = np.fliplr(inp).copy()
        inp = _recolor_solids(inp, rng)

        try:
            metrics = _behavior_metrics(inp)
        except AssertionError:
            continue

        if (
            metrics["unsupported_count"] >= 4
            and metrics["distance_bands"] >= 4
            and metrics["order_gap"] >= 2
            and metrics["moved_water"] >= 6
        ):
            return inp

    raise AssertionError("failed to generate a visible D13/2+D15/0 sample")
