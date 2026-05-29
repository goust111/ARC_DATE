# Combo variant of 28a6681f (index 1)
# combo_spec: D1/0+D3/0
# Name: 纯垂直下落+滑向最远落点

from __future__ import annotations

import random

import numpy as np


# Design analysis for combo D1/0 + D3/0
# --------------------------------------
# a) 具体化与行为追踪:
#    - 保留原题的 10x10、蓝色水=1、下方多色实心盆地/井道的骨架，但把可观察重点收敛到
#      一个上方支撑水源棚面和两个下方竖井。源水始终在 row 0..2、col 3..4 的 2x3 块
#      上；棚面下方 row 3 是实心屋顶，只在两个井口列开洞。
#    - 关键局部状态固定成:
#        * 源水左侧最近井口在 col 2；
#        * 远侧井口在 col 8；
#        * 对于左列源水 (row 2, col 3)，原题中的左下对角 (3,2) 明确开放。
#    - 同一像素在不同规则下的追踪:
#        * 原题 / 仅 D3 改最远:
#          水到达 (2,3) 时，下方被挡，但左下对角开放，于是立即对角滑入 col 2 井道；
#          这一步发生在水平滑动之前，所以“最远落点”来不及介入该像素的第一次转向。
#        * 仅 D1 改纯垂直:
#          同样到达 (2,3) 时，对角滑动被禁用，于是它只能留在棚面上考察水平落口；
#          在棚面连续背景段里，最近井口还是 col 2，所以它继续偏向近侧井。
#        * 融合 combo:
#          对角被禁用后，像素仍停在棚面；但此时水平规则也改成“选最远落点”，于是它
#          会跳过更近的 col 2，直接横移到 col 8 上方再下落。
#    - 差异因此不是两个互不相干的修改，而是串在同一决策链里:
#      “先取消对角绕障”把像素强制留在棚面，“再改成最远落点”决定它会跨棚奔向远井。
#
# b) 复杂度校准:
#    - 只使用一个受控模板家族，再通过镜像、井深和固体重着色产生变化。这样既不会退化成
#      死板的单一位图，也不会引入太多自由度导致路径歧义。
#    - 两个井口的横向距离固定成明显的“近/远”关系，且 2x3 水源会重复触发同一决策，
#      让解题者能从 2-4 个样例里看到稳定的远井优先现象。
#    - 井深只在少数档位里变化 ((3,5), (3,6), (4,5), (4,6))，足够让最终水位有变化，
#      但不改变“左井更近、右井更远、对角只通向近井”的核心几何。
#
# c) 隐式假设审查:
#    - 原 solver 的 diagonal 分支默认“被挡后先看对角，再看水平”。combo 中必须删掉
#      对角分支，否则水永远会提前钻进近井，D3 的最远策略无法生效。
#    - 原 solver 的最近落点按 1 格 1 格重算是安全的；改成最远落点后如果还每步重算，
#      棚面上会左右来回摇摆。因此这里必须对“当前棚面接触”一次性承诺唯一最远井口。
#    - 原 output 检查把“左右都不是背景”当成稳定性的充分条件；对 combo 来说这不是一般
#      性真理，因为理论上可能存在两侧开阔但没有任何可落口的平棚。这里改成更直接的稳定性
#      验证: 临时移除每个输出水像素，再调用 settle()，它必须回到原位。
#    - 原代码里与上方 guide 相关的检测不再是必要逻辑。此融合版完全不依赖 guide；生成器
#      也保证上半区没有额外非底色语义物体需要解析。
#
# d) 放置合法性:
#    - 模板是手工控制的 10x10 结构，井口、固体、水源都在界内且互不重叠。
#    - 镜像仅在整张图上做 fliplr，不会改变合法性。
#    - 固体重着色只替换实心区域颜色，不会碰背景或水。
#
# e) 视觉可观察性:
#    - 每个样本都让同一块上方水源面对“左近右远”的双井口，因此解题者能直接看到:
#        * 原题风格会把更多水送进近井；
#        * combo 会让更多水跨棚送进远井；
#        * 且近井旁本来可走的对角入口，在 combo 中不会被使用。
#    - 生成器内部会把 combo 输出与三个对照版本比较:
#      原题、仅禁对角、仅改最远。只有当 combo 与三者都明显不同才接受样本。
#
# f) 可视化整洁度:
#    - 画面只包含三类区域: 上方矩形水源、中下部三色固体体块、体块里的两个竖井。
#      没有额外标记，也没有多层符号叠加。
#    - 两个井口在屋顶行上是唯一的两个空洞，远近关系一眼可见。
#
# g) 变化可推断性:
#    - 颜色没有任何隐藏排序语义；2-3 种固体颜色只是静态地形着色。
#    - 真正决定行为的是几何位置: 哪个井口更近、哪个更远、哪个能被对角直达。
#
# 9) 新 solver 假设与 generator 保证:
#    - 输入水总是被下方支撑，且每个水像素不会同时拥有左右两个背景邻居；2 列宽源水模板
#      专门保证了这一点。
#    - 屋顶行恰好只有两个井口，因此棚面上的候选落点集合始终可控，最远井口唯一。
#    - 至少有一个源水像素在原题 / 仅 D3 版本中会用到近井对角入口；至少有一个源水像素在
#      combo 中会改为跨棚去远井。生成器通过行为指标过滤，防止退化回任何单变体。


BG = 0
WATER = 1
GRID_SHAPE = (10, 10)
ROOF_ROW = 3
LEFT_BLOCK_END = 4
RIGHT_BLOCK_START = 6
LEFT_SHAFT_COL = 2
RIGHT_SHAFT_COL = 8
SOLID_PALETTE = (2, 3, 4, 5, 6, 7, 8, 9)
CAPACITY_OPTIONS = ((3, 5), (3, 6), (4, 5), (4, 6))
MAX_GENERATION_ATTEMPTS = 200


def _simulate(
    grid: np.ndarray,
    *,
    allow_diagonal: bool,
    prefer_farthest: bool,
    check_input: bool = True,
    check_output: bool = True,
) -> np.ndarray:
    g = np.array(grid, dtype=int).copy()
    h, w = g.shape
    input_water_count = int(np.count_nonzero(g == int(WATER)))

    if check_input:
        assert g.shape == GRID_SHAPE, f"expected grid shape {GRID_SHAPE}, got {g.shape}"
        assert np.all((g >= 0) & (g <= 9)), "grid colors must stay within ARC palette"
        for r, c in np.argwhere(g == int(WATER)):
            rr = int(r)
            cc = int(c)
            assert rr == h - 1 or int(g[rr + 1, cc]) != int(BG), "input water must be supported"
            left_bg = cc > 0 and int(g[rr, cc - 1]) == int(BG)
            right_bg = cc + 1 < w and int(g[rr, cc + 1]) == int(BG)
            assert not (left_bg and right_bg), "input water cannot have bg on both horizontal sides"

    def settle(r: int, c: int) -> tuple[int, int]:
        rr = int(r)
        cc = int(c)
        for _ in range(h * w):
            if rr + 1 < h and int(g[rr + 1, cc]) == int(BG):
                rr += 1
                continue

            if allow_diagonal:
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

                if prefer_farthest:
                    best_dist = max(abs(int(x - cc)) for x in drops)
                    best_cols = [int(x) for x in drops if abs(int(x - cc)) == int(best_dist)]
                    assert len(best_cols) == 1, f"ambiguous farthest drop columns: {best_cols}"
                    best = int(best_cols[0])
                    assert best != int(cc)
                    for x in range(min(int(cc), best) + 1, max(int(cc), best) + 1):
                        assert int(g[rr, int(x)]) == int(BG)
                    cc = int(best)
                else:
                    best_dist = min(abs(int(x - cc)) for x in drops)
                    best_cols = [int(x) for x in drops if abs(int(x - cc)) == int(best_dist)]
                    assert len(best_cols) == 1, f"ambiguous nearest drop columns: {best_cols}"
                    best = int(best_cols[0])
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
        pts = sorted([(int(r), int(c)) for r, c in np.argwhere(g == int(WATER))], reverse=True)
        for r, c in pts:
            if int(g[int(r), int(c)]) != int(WATER):
                continue
            g[int(r), int(c)] = int(BG)
            rr, cc = settle(int(r), int(c))
            moved = bool(moved or ((int(rr), int(cc)) != (int(r), int(c))))
            assert int(g[int(rr), int(cc)]) == int(BG), "water paths should not overlap"
            g[int(rr), int(cc)] = int(WATER)
        if not moved:
            break
    else:
        raise AssertionError("water simulation did not converge")

    if check_output:
        final_pts = [(int(r), int(c)) for r, c in np.argwhere(g == int(WATER))]
        assert len(final_pts) == input_water_count, "water count must be conserved"
        for r, c in final_pts:
            g[int(r), int(c)] = int(BG)
            rr, cc = settle(int(r), int(c))
            g[int(r), int(c)] = int(WATER)
            assert (int(rr), int(cc)) == (int(r), int(c)), f"output water at {(r, c)} is not stable"

    return g


def main(grid: np.ndarray, check_input: bool = True, check_output: bool = True) -> np.ndarray:
    """
    Simulate shelf-supported water that may only fall vertically and that commits to the farthest drop on a shelf.

    Input structure:
    - Background is 0 and water is color 1.
    - All other colors are fixed bottom-anchored solids forming a roof-like shelf and two vertical basin shafts.
    - Input water starts as a supported upper source block above the basin area rather than already inside a shaft.

    Rule:
    - Water falls straight down through background cells whenever the cell below is empty.
    - If downward motion is blocked, diagonal sidesteps are forbidden; the pixel stays on its current supported row.
    - On that supported row, scan the full contiguous background run containing the pixel, find the unique farthest
      drop column in the run, slide horizontally to that column, and then resume vertical falling.
    - Repeat until no water pixel can move.

    Key insight: Removing diagonal escape forces water to remain on a shelf long enough for the horizontal policy to
    matter, and the reversed "pick the farthest drop" policy then sends water past nearer shafts into more remote ones.
    """
    return _simulate(
        grid,
        allow_diagonal=False,
        prefer_farthest=True,
        check_input=check_input,
        check_output=check_output,
    )


def _build_case(
    *,
    left_color: int,
    center_color: int,
    right_color: int,
    left_capacity: int,
    right_capacity: int,
    mirror: bool,
) -> np.ndarray:
    g = np.zeros(GRID_SHAPE, dtype=int)

    g[ROOF_ROW:, :] = int(center_color)
    g[ROOF_ROW:, :LEFT_BLOCK_END] = int(left_color)
    g[ROOF_ROW:, RIGHT_BLOCK_START:] = int(right_color)

    for r in range(ROOF_ROW, min(GRID_SHAPE[0], ROOF_ROW + int(left_capacity))):
        g[int(r), LEFT_SHAFT_COL] = int(BG)
    for r in range(ROOF_ROW, min(GRID_SHAPE[0], ROOF_ROW + int(right_capacity))):
        g[int(r), RIGHT_SHAFT_COL] = int(BG)

    g[:ROOF_ROW, 3:5] = int(WATER)

    if mirror:
        g = np.fliplr(g)

    return g


def _shaft_columns(grid: np.ndarray) -> tuple[int, int]:
    cols = [int(c) for c in np.where(np.array(grid, dtype=int)[ROOF_ROW] == int(BG))[0]]
    assert len(cols) == 2, f"expected exactly two shaft mouths, found {cols}"
    return int(cols[0]), int(cols[1])


def _near_and_far_shaft(grid: np.ndarray) -> tuple[int, int]:
    left_col, right_col = _shaft_columns(grid)
    source_cols = np.argwhere(np.array(grid, dtype=int) == int(WATER))[:, 1]
    source_center = float(source_cols.mean())
    left_dist = abs(float(left_col) - source_center)
    right_dist = abs(float(right_col) - source_center)
    assert left_dist != right_dist, "near/far shaft relationship must be unique"
    if left_dist < right_dist:
        return int(left_col), int(right_col)
    return int(right_col), int(left_col)


def _behavior_metrics(grid: np.ndarray) -> dict[str, int]:
    original = _simulate(grid, allow_diagonal=True, prefer_farthest=False, check_input=True, check_output=True)
    vertical_nearest = _simulate(grid, allow_diagonal=False, prefer_farthest=False, check_input=True, check_output=True)
    diagonal_farthest = _simulate(grid, allow_diagonal=True, prefer_farthest=True, check_input=True, check_output=True)
    combo = main(grid, check_input=True, check_output=True)

    near_col, far_col = _near_and_far_shaft(grid)

    def count_in_col(arr: np.ndarray, col: int) -> int:
        return int(np.sum(np.array(arr, dtype=int)[:, int(col)] == int(WATER)))

    return {
        "diff_vs_original": int(np.sum(combo != original)),
        "diff_vs_vertical_nearest": int(np.sum(combo != vertical_nearest)),
        "diff_vs_diagonal_farthest": int(np.sum(combo != diagonal_farthest)),
        "original_near": count_in_col(original, near_col),
        "original_far": count_in_col(original, far_col),
        "vertical_nearest_near": count_in_col(vertical_nearest, near_col),
        "vertical_nearest_far": count_in_col(vertical_nearest, far_col),
        "diagonal_farthest_near": count_in_col(diagonal_farthest, near_col),
        "diagonal_farthest_far": count_in_col(diagonal_farthest, far_col),
        "combo_near": count_in_col(combo, near_col),
        "combo_far": count_in_col(combo, far_col),
    }


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        left_color, center_color, right_color = rng.sample(SOLID_PALETTE, 3)
        left_capacity, right_capacity = rng.choice(CAPACITY_OPTIONS)
        mirror = bool(rng.getrandbits(1))

        grid = _build_case(
            left_color=int(left_color),
            center_color=int(center_color),
            right_color=int(right_color),
            left_capacity=int(left_capacity),
            right_capacity=int(right_capacity),
            mirror=mirror,
        )

        metrics = _behavior_metrics(grid)
        if (
            metrics["diff_vs_original"] >= 4
            and metrics["diff_vs_vertical_nearest"] >= 4
            and metrics["diff_vs_diagonal_farthest"] >= 4
            and metrics["combo_far"]
            >= max(
                metrics["original_far"],
                metrics["vertical_nearest_far"],
                metrics["diagonal_farthest_far"],
            )
            + 2
            and metrics["combo_near"]
            <= min(
                metrics["original_near"],
                metrics["vertical_nearest_near"],
                metrics["diagonal_farthest_near"],
            )
            - 2
        ):
            return grid

    raise AssertionError("generator could not build a visible D1/0+D3/0 combo sample")
