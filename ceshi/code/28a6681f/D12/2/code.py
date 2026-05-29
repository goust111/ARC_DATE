# Variant of 28a6681f
# Dimension: D12 — 水量守恒规则 (transform)
# Change: 结构像素不变，水总量严格守恒——输入中有N个水像素，输出中恰好也有N个水像素... → 壁面被水侵蚀消失

from __future__ import annotations

import numpy as np


# Design analysis
# ---------------
# spec:
#   visual:
#     - 仍然使用 10x10 网格，背景为 0，水为 1，结构色固定为 2/3/4。
#     - 保留原题“底部盆地 + 邻接槽位”的视觉母题，但把关键隔墙做成单格厚的内壁。
#     - 生成器使用左右两槽结构：一侧是较浅的平台盆，另一侧是更深的盆。
#       输入水先稳定堆在浅盆中，输出中隔墙被侵蚀后，水会从破口落入更深的邻盆。
#   behavior_trace:
#     - 小例子：左侧浅盆在第 8 行有一条平台，中央隔墙位于第 4 列，右侧是更深的空盆。
#     - 原逻辑：水在平台上方已被支撑住；因为平台右侧隔墙完整，同行没有“可继续下落的唯一缺口”，
#       所以水不会跨墙，只会停在左槽内。
#     - 变体逻辑：当水位堆到与隔墙同高时，满足“墙左是水、墙右是背景”的那些墙像素被擦除。
#       破口右侧正好对应更深的空腔，水于是沿原题的水平找落点规则滑到破口并下落。
#     - 差异：原题只移动水；本变体先改结构，再让原有流体规则利用新通道继续流动。
#   complexity:
#     - 不引入颜色语义，侵蚀完全由可见的局部接触关系决定，避免“看不出为什么只有某色会消失”。
#     - 侵蚀对象限制为内侧单格厚竖墙，且目标盆更深，保证结构变化会引发明显重分布，
#       而不是消掉几格后结果仍与原题相同。
#     - 分支只保留少量可观察自由度：水源在左/右、隔墙列为 4/5、平台高度为 7/8、隔墙高度为 2/3/4。
#   audit_of_original_main:
#     - 原 `settle()` 假设结构静止：这里保留其局部运动规则，但外层改成“每轮水体安定后再扫描侵蚀”。
#     - 原全局循环只关心水是否移动：这里若本轮有墙被擦除，也继续迭代。
#     - 原输出默认结构不变：这里不再检查结构守恒，只检查最终水像素本身仍满足稳定性。
#     - 原题允许相邻盆通过低边自然溢流：这里生成器把隔墙高度设在自然溢流线之上，
#       用“无侵蚀版输出不同”作为验收，避免退化回原题。
#   placement_legality:
#     - 外墙固定在左右边界，避免它们也满足“水一侧 / 背景一侧”的侵蚀条件。
#     - 底行不参与侵蚀，防止把盆底打穿。
#     - 浅盆平台始终高于深盆底，保证破墙后存在真实落差，原 `settle()` 才能把水导向新盆。
#   observability:
#     - 输入中目标盆为空，输出中目标盆出现水且中央墙少掉若干像素，人眼可直接归纳“水把墙磨掉了”。
#     - 同一规则同时出现左右镜像和不同侵蚀深度，样例不会被误读成固定模板拷贝。
#   cleanliness:
#     - 结构只分三种颜色，水只一种颜色，且水与墙不会在同一格叠加显示。
#     - 输出中的变化集中在中央隔墙和深盆内部，不会出现全图随机散落像素。


BG = 0
WATER = 1
LEFT_COLOR = 2
MID_COLOR = 3
RIGHT_COLOR = 4
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


def _simulate(
    grid: np.ndarray,
    *,
    allow_erosion: bool,
    check_input: bool,
    check_output: bool,
) -> np.ndarray:
    g = np.array(grid, dtype=int).copy()
    H, W = g.shape
    bg = BG
    water = WATER

    if check_input:
        for r, c in np.argwhere(g == int(water)):
            rr = int(r)
            cc = int(c)
            assert rr == H - 1 or int(g[rr + 1, cc]) != int(bg), "input water must be supported"
            left_bg = cc > 0 and int(g[rr, cc - 1]) == int(bg)
            right_bg = cc + 1 < W and int(g[rr, cc + 1]) == int(bg)
            assert not (left_bg and right_bg), "input water cannot have bg on both horizontal sides"

    bottom_colors = set(map(int, g[H - 1]))
    for color in sorted(set(map(int, g[:7].flatten())) - {bg, water} - bottom_colors):
        pts = {(int(r), int(c)) for r, c in np.argwhere(g[:7] == int(color))}
        diag_links = sum(1 for r, c in pts if (r + 1, c - 1) in pts or (r + 1, c + 1) in pts)
        if diag_links >= 3:
            assert any(c in (0, W - 1) for _r, c in pts), "guide must enter from a grid edge"
            assert _component_count(pts, False) == _component_count(pts, True), "guide must be 4-connected"

    def settle(r: int, c: int) -> tuple[int, int]:
        rr = int(r)
        cc = int(c)
        for _ in range(H * W):
            if rr + 1 < H and int(g[rr + 1, cc]) == int(bg):
                rr += 1
                continue

            opts = []
            if (
                rr + 1 < H
                and cc - 1 >= 0
                and int(g[rr, cc - 1]) == int(bg)
                and int(g[rr + 1, cc - 1]) == int(bg)
            ):
                opts.append((int(rr + 1), int(cc - 1)))
            if (
                rr + 1 < H
                and cc + 1 < W
                and int(g[rr, cc + 1]) == int(bg)
                and int(g[rr + 1, cc + 1]) == int(bg)
            ):
                opts.append((int(rr + 1), int(cc + 1)))
            if opts:
                assert len(opts) == 1, f"ambiguous diagonal moves: {opts}"
                rr, cc = opts[0]
                continue

            if rr + 1 < H and int(g[rr + 1, cc]) != int(bg):
                left = int(cc)
                while left - 1 >= 0 and int(g[rr, left - 1]) == int(bg):
                    left -= 1
                right = int(cc)
                while right + 1 < W and int(g[rr, right + 1]) == int(bg):
                    right += 1

                drops = [int(x) for x in range(left, right + 1) if int(g[rr + 1, int(x)]) == int(bg)]
                if not drops:
                    break

                min_dist = min(abs(int(x - cc)) for x in drops)
                nearest = [int(x) for x in drops if abs(int(x - cc)) == int(min_dist)]
                assert len(nearest) == 1, f"ambiguous nearest drop columns: {nearest}"
                best = nearest[0]
                assert best != int(cc)
                step = -1 if best < int(cc) else 1
                nx = int(cc + step)
                assert int(g[rr, nx]) == int(bg)
                cc = nx
                continue

            break
        else:
            raise AssertionError("water pixel did not settle")
        return int(rr), int(cc)

    def erode_cells() -> list[tuple[int, int]]:
        if not allow_erosion:
            return []
        cells: list[tuple[int, int]] = []
        for r in range(H - 1):
            for c in range(1, W - 1):
                v = int(g[r, c])
                if v in (bg, water):
                    continue
                left = int(g[r, c - 1])
                right = int(g[r, c + 1])
                if (left == water and right == bg) or (left == bg and right == water):
                    cells.append((int(r), int(c)))
        return cells

    for _pass in range(4 * H * W):
        moved = False
        pts = sorted([(int(r), int(c)) for r, c in np.argwhere(g == int(water))], reverse=True)
        for r, c in pts:
            if int(g[int(r), int(c)]) != int(water):
                continue
            g[int(r), int(c)] = int(bg)
            rr, cc = settle(int(r), int(c))
            moved = bool(moved or ((int(rr), int(cc)) != (int(r), int(c))))
            g[int(rr), int(cc)] = int(water)

        eroded = erode_cells()
        if eroded:
            for r, c in eroded:
                g[int(r), int(c)] = int(bg)

        if not moved and not eroded:
            break
    else:
        raise AssertionError("water simulation did not converge")

    if check_output:
        # Under erosion, a stable pixel can legitimately have background on one side
        # (for example on a stranded shelf). The right notion of "finished" is that
        # removing any single water pixel and re-settling it lands it back in place.
        for r, c in np.argwhere(g == int(water)):
            rr0 = int(r)
            cc0 = int(c)
            g[rr0, cc0] = int(bg)
            rr1, cc1 = settle(rr0, cc0)
            g[rr0, cc0] = int(water)
            assert (int(rr1), int(cc1)) == (rr0, cc0), "output water must already be locally stable"

    return g


def main(grid: np.ndarray, check_input: bool = True, check_output: bool = True) -> np.ndarray:
    """
    Simulate water settling into stepped basins whose thin interior walls can be worn away.

    Input structure:
    - bg = 0 and water = 1 as before; the other colors are solid basin pixels.
    - Bottom-anchored walls create neighboring basins, with one side often shallower than the other.
    - Water starts as a supported pile inside one basin, already pressed against a thin interior wall.

    Rule:
    - Water still uses the original local motion: fall straight down, otherwise take the unique open
      diagonal, otherwise slide horizontally toward the nearest place where it can drop again.
    - After each full settling sweep, any non-edge wall pixel that has water on one horizontal side
      and background on the other is eroded and becomes background.
    - The simulation repeats until neither water nor walls change.

    Key insight: Water no longer waits for a static overflow rim; by piling up against a thin divider,
    it carves a breach that creates a new lower drop route into a neighboring basin.
    """
    return _simulate(
        grid,
        allow_erosion=True,
        check_input=check_input,
        check_output=check_output,
    )


def _solid_count(grid: np.ndarray) -> int:
    return int(np.count_nonzero((grid != BG) & (grid != WATER)))


def _removed_solid_count(inp: np.ndarray, out: np.ndarray) -> int:
    solid_in = (inp != BG) & (inp != WATER)
    solid_out = (out != BG) & (out != WATER)
    return int(np.count_nonzero(solid_in & ~solid_out))


def _target_water_count(out: np.ndarray, partition_col: int, source_side: str) -> int:
    if source_side == "L":
        return int(np.count_nonzero(out[:, partition_col + 1:9] == WATER))
    return int(np.count_nonzero(out[:, 1:partition_col] == WATER))


def _build_case(rng: np.random.Generator) -> tuple[np.ndarray, dict[str, int | str]]:
    g = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)

    partition_col = int(rng.choice([4, 5]))
    source_side = str(rng.choice(["L", "R"]))
    shelf_row = int(rng.choice([7, 8]))
    partition_top = int(rng.integers(2, min(5, shelf_row - 1)))
    water_top = int(rng.integers(partition_top, shelf_row - 1))

    if source_side == "L":
        source_color = LEFT_COLOR
        target_color = RIGHT_COLOR
        source_cols = list(range(1, partition_col))
        target_cols = list(range(partition_col + 1, 9))
        source_outer_col = 0
        target_outer_col = 9
        source_fill_order = list(range(partition_col - 1, 0, -1))
    else:
        source_color = RIGHT_COLOR
        target_color = LEFT_COLOR
        source_cols = list(range(partition_col + 1, 9))
        target_cols = list(range(1, partition_col))
        source_outer_col = 9
        target_outer_col = 0
        source_fill_order = list(range(partition_col + 1, 9))

    assert source_cols and target_cols

    source_wall_top = int(rng.integers(1, water_top + 1))
    target_wall_top = int(rng.integers(2, 7))

    g[9, :partition_col] = LEFT_COLOR
    g[9, partition_col] = MID_COLOR
    g[9, partition_col + 1:] = RIGHT_COLOR
    g[partition_top:9, partition_col] = MID_COLOR
    g[source_wall_top:10, source_outer_col] = source_color
    g[target_wall_top:10, target_outer_col] = target_color
    g[shelf_row, source_cols] = source_color

    rows_touching_wall = shelf_row - water_top
    max_top_count = len(source_cols)
    top_count = int(rng.integers(1, max_top_count + 1))
    water_count = rows_touching_wall * len(source_cols) - (len(source_cols) - top_count)

    remaining = water_count
    for r in range(shelf_row - 1, -1, -1):
        if remaining <= 0:
            break
        take = min(len(source_cols), remaining)
        for c in source_fill_order[:take]:
            g[r, c] = WATER
        remaining -= take

    assert remaining == 0

    return g, {
        "partition_col": partition_col,
        "source_side": source_side,
        "shelf_row": shelf_row,
        "partition_top": partition_top,
    }


def _attempt_case(rng: np.random.Generator) -> np.ndarray | None:
    inp, meta = _build_case(rng)
    try:
        out = main(inp)
        out_static = _simulate(inp, allow_erosion=False, check_input=True, check_output=True)
    except Exception:
        return None

    removed = _removed_solid_count(inp, out)
    transferred = _target_water_count(out, int(meta["partition_col"]), str(meta["source_side"]))

    if removed < 2:
        return None
    if transferred < 2:
        return None
    if np.array_equal(out, out_static):
        return None
    if _solid_count(out) >= _solid_count(inp):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(400):
        result = _attempt_case(rng)
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed to construct a valid erosion case")
