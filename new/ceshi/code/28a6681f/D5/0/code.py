# Variant of 28a6681f
# Dimension: D5 — 终点/停止条件 (dst)
# Change: 水像素流到'稳定'位置即停止——稳定定义为：左右和下方均为非背景或网格边界（不再有运动空间）... → 固定水位线终点

from __future__ import annotations

import random

import numpy as np


# Design analysis for the D5 variant
# ----------------------------------
# a) 具体化与行为追踪
# - 网格仍为 10x10，水仍是 1，背景仍是 0。
# - 下半区只生成 2-3 个相邻的竖直盆地：每个盆地由两根底部锚定的墙柱围出，
#   盆地内部是从底边向上的空腔。这样保留了原题“相邻盆地/溢流链”的视觉骨架。
# - 新增“水位线标记色” 8：每个盆地左右两侧墙柱上各有一个 8，且两点同一行。
#   这两个 8 所在的行就是该盆地的目标水位线。输出中，这个盆地只填充到该行及以下，
#   即使该盆地上方仍有更多空腔也不继续装水；多余的水转去下一个相邻盆地。
# - 小例子：若左盆地宽 2、标记在第 7 行，则容量固定为 2*(10-7)=6。即使几何上该盆地
#   的墙一直高到第 5 行，原始逻辑会继续把更多水留在左盆地；现在第 7 行以下一旦填满，
#   额外的水就去右盆地。因此同一输入在旧逻辑和新逻辑下会出现不同输出。
#
# b) 复杂度校准
# - 只保留一个新的自由度：每个盆地的水位线行号。颜色和几何都尽量简化，避免把变化
#   扩散成“识别任意复杂容器”。
# - 盆地数量限制为 2 或 3，且都是矩形竖槽；这样视觉上很清楚，解题者能在 2-4 个样例里
#   看出“先读标记，再按水量依次填盆地”的规律。
# - 标记色固定为 8，不参与排序，不借助颜色 id 的大小关系做任何决策。
#
# c) 对原 main() 的隐式假设审查
# - 原 `settle()` 逐像素把水推到“最低稳定位置”。这依赖旧假设：终点只由局部几何决定。
#   D5 变体里终点改为“每盆地的外部标记容量”，所以这个局部推进前提不成立，必须改为
#   “统计总水量 + 读出盆地容量 + 顺序分配”的求解方式。
# - 原 `check_output` 断言每个水像素左右下方都被挡住。若允许“只填到标记线”，这条断言
#   仍然可以保留在本变体里，但前提是生成器只产生整层整层的矩形水块，不产生悬空或缺口。
#   因此生成器会只生成完整填满到水位线的盆地，不生成任意局部残量。
# - 原代码中“水 count 守恒、结构像素不变”的假设仍然成立，所以主流程继续以计数守恒为核心。
#
# d) 放置合法性
# - 盆地由底行的非背景列唯一确定；生成器保证这些“墙列”彼此不重叠，且相邻墙之间至少
#   留 1 列作为盆地内部。
# - 墙柱顶部都在第 5-6 行，顶部第 0-3 行专门留给源水仓和支撑平台，避免重叠。
# - 水位标记行必须不高于两侧墙柱的顶部之上，因此约束为 `line_row >= max(top_left, top_right) + 1`。
# - 输入源水放在上侧边缘仓中，底部有实体平台支撑；部分行若未填满，也从边界向内连续铺开，
#   保证不会出现“左右两侧都是背景”的孤立输入水。
#
# e) 视觉可观察性
# - 同一个盆地里，标记色 8 明确夹住一条水平线，输出水面恰好与这条线对齐，而不是和墙顶对齐。
# - 为了让 D5 变化可见，生成器故意让至少一个已填满的盆地在水位线上方还留有空腔。
#   这使得“不是填到几何允许的最高处，而是填到标记线”能被直接看见。
# - 同时约 70% 的样本让水填满至少两个相邻盆地，确保“满了就溢到下一盆地”的分支经常出现。
#
# f) 可视化整洁度
# - 结构只用两种墙色加一种标记色。标记总是嵌在墙柱里，不和水重叠。
# - 上方源水仓是规则矩形/台阶状堆积，不和下方盆地交叉，画面层次清楚。
#
# g) 颜色变化可推断性
# - 颜色语义固定：0 背景，1 水，2/3 墙，8 水位线标记。不存在颜色置换题意。
#
# h) 量化退化检查目标
# - 退化判据不是 `main(input)` 是否报错，而是：在生成样本上，新输出和原始“最低稳定”
#   逻辑应当经常不同。测试时会统计“至少一个已填满盆地在其墙顶下方留下空腔”的样本数，
#   以及“新旧 solver 输出不同”的比例。
#
# i) main() 新增假设与 generator 保证
# - `main()` 假设底行所有非背景非水列就是盆地墙列；生成器保证上方支撑平台不触底。
# - `main()` 假设每个盆地的左右边墙上各有且仅有一个公共标记行；生成器严格这样放置。
# - `main()` 假设水从左或右单侧源区流入，因此只需要根据输入水的重心判断灌水方向；
#   生成器只产生左源样本或其水平镜像，满足该假设。


BG = 0
WATER = 1
MARKER = 8
SOLID_COLORS = (2, 3)
H = 10
W = 10


def _find_wall_columns(grid: np.ndarray) -> list[int]:
    return [int(c) for c in range(grid.shape[1]) if int(grid[grid.shape[0] - 1, c]) not in (BG, WATER)]


def _parse_basins(grid: np.ndarray) -> list[dict[str, int | list[int]]]:
    wall_cols = _find_wall_columns(grid)
    assert 3 <= len(wall_cols) <= 4, f"expected 2-3 basins, got wall columns {wall_cols}"
    basins: list[dict[str, int | list[int]]] = []
    for left, right in zip(wall_cols, wall_cols[1:]):
        assert right - left >= 2, "basin must contain at least one interior column"
        left_rows = {int(r) for r in np.where(grid[:, left] == int(MARKER))[0]}
        right_rows = {int(r) for r in np.where(grid[:, right] == int(MARKER))[0]}
        common_rows = sorted(left_rows & right_rows)
        assert len(common_rows) == 1, f"basin between {left} and {right} needs one marker row"
        line_row = int(common_rows[0])
        cols = [int(c) for c in range(left + 1, right)]
        basins.append(
            {
                "left": int(left),
                "right": int(right),
                "line": line_row,
                "width": int(len(cols)),
                "capacity": int(len(cols) * (grid.shape[0] - line_row)),
            }
        )
    assert 2 <= len(basins) <= 3
    return basins


def _fill_basin(grid: np.ndarray, basin: dict[str, int | list[int]], amount: int, source_side: str) -> int:
    cols = list(range(int(basin["left"]) + 1, int(basin["right"])))
    row_cols = cols if source_side == "left" else list(reversed(cols))
    placed = 0
    for r in range(grid.shape[0] - 1, int(basin["line"]) - 1, -1):
        for c in row_cols:
            if placed >= amount:
                return placed
            assert int(grid[r, c]) == BG
            grid[r, c] = WATER
            placed += 1
    return placed


def main(grid: np.ndarray, check_input: bool = True, check_output: bool = True) -> np.ndarray:
    """
    Route water into neighboring basins, but stop each basin at its colored waterline rather than at the deepest stable resting cells.

    Input structure:
    - Background is 0 and water is 1.
    - Bottom-anchored wall columns form 2-3 adjacent vertical basins in the lower half of the grid.
    - Each basin has a pair of special marker pixels (color 8), one on each side wall at the same row, defining that basin's target fill level.
    - Input water starts as a supported source block near the left or right upper side.

    Rule:
    - Count all water pixels, remove them from the grid, and read the basin intervals from the bottom wall columns.
    - For each basin, convert its paired marker row into a fixed capacity: every cell from that row down to the bottom inside the basin may be filled.
    - Starting from the source side, fill basins in order. Once a basin reaches its marked level, any remaining water is sent to the next basin instead of rising higher in the same one.
    - Wall and marker pixels stay unchanged.

    Key insight:
    - The final destination is controlled by visible waterline markers, not by purely local geometric stability. Extra water can bypass empty headroom in an earlier basin and appear in a later basin.
    """
    g = np.array(grid, dtype=int).copy()
    total_water = int(np.sum(g == WATER))
    assert g.shape == (H, W), f"expected 10x10 grid, got {g.shape}"

    if check_input:
        assert total_water > 0, "input needs source water"
        for r, c in np.argwhere(g == WATER):
            rr = int(r)
            cc = int(c)
            assert rr == H - 1 or int(g[rr + 1, cc]) != BG, "input water must be supported"
            left_bg = cc > 0 and int(g[rr, cc - 1]) == BG
            right_bg = cc + 1 < W and int(g[rr, cc + 1]) == BG
            assert not (left_bg and right_bg), "input water cannot float as a 1-cell horizontal island"

    basins = _parse_basins(g)
    assert int(np.sum(int(b["capacity"]) for b in basins)) >= total_water, "not enough marked basin capacity"

    water_cols = np.argwhere(g == WATER)[:, 1]
    source_side = "left" if float(np.mean(water_cols)) < (W - 1) / 2 else "right"

    g[g == WATER] = BG
    ordered = basins if source_side == "left" else list(reversed(basins))

    remaining = total_water
    for basin in ordered:
        if remaining <= 0:
            break
        amount = min(remaining, int(basin["capacity"]))
        placed = _fill_basin(g, basin, amount, source_side)
        assert placed == amount
        remaining -= amount
    assert remaining == 0, "water should fit into the marked basins"

    if check_output:
        for basin in basins:
            left = int(basin["left"])
            right = int(basin["right"])
            line = int(basin["line"])
            for c in range(left + 1, right):
                assert np.all(g[:line, c] != WATER), "no output water may rise above a basin's marker row"
        for r, c in np.argwhere(g == WATER):
            rr = int(r)
            cc = int(c)
            assert cc == 0 or int(g[rr, cc - 1]) != BG
            assert rr == H - 1 or int(g[rr + 1, cc]) != BG
            assert cc == W - 1 or int(g[rr, cc + 1]) != BG

    return g


def _draw_walls(grid: np.ndarray, wall_cols: list[int], wall_tops: list[int]) -> None:
    for i, (c, top) in enumerate(zip(wall_cols, wall_tops)):
        color = int(SOLID_COLORS[i % len(SOLID_COLORS)])
        grid[top:, c] = color


def _add_markers(grid: np.ndarray, basins: list[dict[str, int]]) -> None:
    for basin in basins:
        row = int(basin["line"])
        grid[row, int(basin["left"])] = MARKER
        grid[row, int(basin["right"])] = MARKER


def _layout_options() -> list[list[int]]:
    return [
        [1, 4, 8],      # widths 2 and 3, one spare column on the right
        [0, 3, 6, 9],   # three width-2 basins
    ]


def _build_canonical_output(rng: random.Random) -> tuple[np.ndarray, list[dict[str, int]], int]:
    grid = np.zeros((H, W), dtype=int)
    grid[4, :5] = SOLID_COLORS[0]
    wall_cols = list(rng.choice(_layout_options()))
    if len(wall_cols) == 4:
        wall_tops = [5 for _ in wall_cols]
    else:
        wall_tops = [int(rng.choice((5, 6))) for _ in wall_cols]
    _draw_walls(grid, wall_cols, wall_tops)

    basins: list[dict[str, int]] = []
    used_lines: set[int] = set()
    for left, right in zip(wall_cols, wall_cols[1:]):
        min_line = max(wall_tops[wall_cols.index(left)], wall_tops[wall_cols.index(right)]) + 1
        max_line = 8
        assert min_line <= max_line, "need at least one row of empty headroom above the marked water"
        choices = [int(x) for x in range(min_line, max_line + 1) if x not in used_lines]
        assert choices, "shared walls require globally distinct marker rows"
        line = int(rng.choice(choices))
        basins.append({"left": int(left), "right": int(right), "line": line})
        used_lines.add(line)

    _add_markers(grid, basins)

    basin_caps = [int((b["right"] - b["left"] - 1) * (H - b["line"])) for b in basins]
    if len(basins) == 2:
        fill_count = 2 if rng.random() < 0.7 else 1
    else:
        fill_count = rng.choice((2, 2, 3, 1))
    total_water = int(sum(basin_caps[:fill_count]))
    assert total_water > 0

    for basin in basins[:fill_count]:
        left = int(basin["left"])
        right = int(basin["right"])
        line = int(basin["line"])
        grid[line:, left + 1 : right] = WATER

    return grid, basins, total_water


def _place_source_block(grid: np.ndarray, count: int) -> None:
    chamber_cols = list(range(5))
    shelf_row = 4
    support_color = SOLID_COLORS[0]
    grid[shelf_row, chamber_cols] = support_color

    cells: list[tuple[int, int]] = []
    for r in range(shelf_row - 1, -1, -1):
        for c in chamber_cols:
            cells.append((r, c))
    assert count <= len(cells), f"source chamber too small for {count} water cells"
    for r, c in cells[:count]:
        grid[r, c] = WATER


def _make_input_from_output(output_grid: np.ndarray, total_water: int) -> np.ndarray:
    inp = np.array(output_grid, dtype=int).copy()
    inp[inp == WATER] = BG
    _place_source_block(inp, total_water)
    return inp


def _flip_pair_if_needed(inp: np.ndarray, out: np.ndarray, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    if rng.random() < 0.5:
        return np.fliplr(inp), np.fliplr(out)
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(200):
        rng = random.Random(random.randrange(1 << 30))
        output_grid, basins, total_water = _build_canonical_output(rng)
        assert any(int(b["line"]) > min(np.where(output_grid[:, int(b["left"])] != BG)[0][0], np.where(output_grid[:, int(b["right"])] != BG)[0][0]) for b in basins), "need visible headroom above at least one filled basin"
        inp = _make_input_from_output(output_grid, total_water)
        inp, out = _flip_pair_if_needed(inp, output_grid, rng)
        solved = main(inp)
        if np.array_equal(solved, out):
            return inp
    raise RuntimeError("failed to generate a valid D5 variant sample")
