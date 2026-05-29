# Combo variant of 28a6681f (index 6)
# combo_spec: D5/0+D6/2
# Name: 矩形水块+堆叠高度编码

from __future__ import annotations

import random

import numpy as np


# Design analysis for the combo variant
# -------------------------------------
# a) 具体化与行为追踪
# - 保留原题的 10x10、蓝色水=1、底部相邻盆地链的视觉骨架，但把输入水改成一个紧凑矩形。
# - 这里的矩形源块总是贴左边或右边，放在上半区的一条短支撑台上；其“高度”h 才是有效信息，
#   宽度只负责让它看起来像一个整体大水块，不再对应输出水量。
# - 输出不再按原题那样逐像素找“最深稳定点”。改为：读出源块高度 h，然后把每个盆地内部列
#   的最深 h 个背景格都涂成水。若中间分隔墙较矮，h 层水会跨过它连成一片，形成自然的溢流感。
# - 例子：若源块是 3x2，则输入只有 6 个水像素；但若底部盆地内部共有 5 列，输出会出现
#   5*3=15 个水像素，即 5 列都堆 3 层。原始 solver 必然守恒 6 个像素，而融合后答案是 15 个，
#   这正是“高度编码”覆盖“数量守恒”的核心差异。
#
# b) 复杂度校准
# - 只让一个显著属性参与编码：源矩形的行数 h，取值限制为 2..4，方便在少量样例里反复出现。
# - 源宽度只在 2 或 3 之间变化，既能显出“矩形块”而不是细线，又不会引入太多无关组合。
# - 盆地只做成 2 或 3 个相邻竖槽，底部总内部宽度固定在 4 或 5 列，因此输出模式很整齐，
#   人眼容易看出“每一列都重复了同样的高度”。
#
# c) 对原 main() 隐式假设的审查
# - 原 solver 的核心前提是“终点由局部重力/滑动稳定性唯一决定”。D6/2 明确打破这一点，
#   所以不能再保留逐像素 `settle()` 模拟，必须改为“读编码属性 -> 直接构造目标层数”。
# - 原 solver 默认水量守恒；本组合明确允许不守恒，因此任何输入/输出水像素计数相等的断言
#   都必须删除。
# - 原输出断言要求每个水像素左右下方都被堵住；但这里的正确输出可能是多列整齐的 h 层堆叠，
#   顶层两侧完全可能暴露在背景中，所以这个旧断言也不再成立。
# - 原代码允许任意分散水体；本组合新增了“唯一、连通、填满 bbox 的边贴矩形源块”假设，
#   需要由 `main()` 检查并由 generator 严格保证。
#
# d) 放置合法性
# - 支撑台固定在第 4 行，源块放在其上方，因此高度 2..4 时都不会越界。
# - 盆地墙只从第 5..9 行开始，保证不会和源块或支撑台重叠。
# - 底行所有非背景列都保留给盆地墙，方便 solver 通过底行唯一解析盆地区间。
# - generator 只选那些外侧墙足够高的布局，保证 h 层水不会从最左或最右边外泄到背景里。
#
# e) 视觉可观察性
# - 输入里最显眼的是矩形水块的高度；输出里最显眼的是每个盆地列都出现同样的堆叠层数。
# - 源宽度在不同样例中会变化，但输出高度只跟源高度同步，这能直接提示“宽度/总像素数不重要”。
# - 某些样例里内部墙会比 h 更矮，于是输出跨墙连通；另一些样例里内部墙更高，输出分成多个槽。
#   这两类都遵循同一个规则，能帮助解题者看到“高度编码 + 盆地链”的联合模式。
#
# f) 可视化整洁度
# - 结构只用 3 个非水颜色：1 个给源支撑台，2 个交替给墙。没有额外标记、符号或噪声。
# - 支撑台始终待在上半区，墙始终锚定底部，源块和目标区上下分层，读图时不需要猜覆盖关系。
#
# g) 变化的可推断性
# - 没有任何基于颜色编号大小的逻辑；颜色只区分“水 / 背景 / 静态结构”。
# - 真正决定输出的是可见的几何属性“源块高度”，并且这一属性在输入里是直接可数的。
#
# h) 量化退化检查目标
# - 生成器故意让源宽度 2..3，而底部目标内部宽度为 4 或 5，因此大多数样本都会产生显式的
#   水量不守恒；这保证新规则与原始守恒重力模拟不能退化为同一输出。
# - 另外保留 `_classic_main()` 作为旧行为参考，后续统计时可直接比较新旧输出是否不同。
#
# i) main() 的新增假设与 generator 保证
# - `main()` 假设所有输入水像素恰好组成一个贴单侧边缘的实心矩形；generator 始终这样放。
# - `main()` 假设底行所有非背景列就是盆地墙列，且相邻墙之间至少留 1 列内部空间；
#   generator 只采样满足这一点的墙布局。
# - `main()` 假设目标盆地内部列没有别的固体障碍；generator 不在这些列里放额外结构，
#   这样“每列取最深 h 个背景格”就是清晰、稳定且可见的规则。


BG = 0
WATER = 1
H = 10
W = 10
SHELF_ROW = 4

LAYOUTS: tuple[tuple[int, ...], ...] = (
    (2, 5, 9),      # widths 2 and 3 -> total target width 5
    (3, 6, 9),      # widths 2 and 2 -> total target width 4
    (2, 4, 7, 9),   # widths 1, 2, 1 -> total target width 4
)


def _component_count(points: set[tuple[int, int]]) -> int:
    remaining = set(points)
    count = 0
    while remaining:
        count += 1
        stack = [remaining.pop()]
        while stack:
            r, c = stack.pop()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nxt = (int(r + dr), int(c + dc))
                if nxt in remaining:
                    remaining.remove(nxt)
                    stack.append(nxt)
    return count


def _source_box(grid: np.ndarray) -> tuple[int, int, int, int]:
    pts = np.argwhere(grid == int(WATER))
    assert pts.size > 0, "expected a non-empty water source"
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _source_info(grid: np.ndarray) -> tuple[int, int, int, int, int, int, str]:
    r0, c0, r1, c1 = _source_box(grid)
    height = int(r1 - r0 + 1)
    width = int(c1 - c0 + 1)
    if c0 == 0:
        side = "left"
    elif c1 == W - 1:
        side = "right"
    else:
        raise AssertionError("water source must touch exactly one side edge")
    assert not (c0 == 0 and c1 == W - 1), "source cannot touch both side edges"
    return height, width, r0, c0, r1, c1, side


def _find_wall_columns(grid: np.ndarray) -> list[int]:
    return [int(c) for c in range(W) if int(grid[H - 1, c]) not in (BG, WATER)]


def _parse_basins(grid: np.ndarray) -> list[dict[str, int]]:
    wall_cols = _find_wall_columns(grid)
    assert 3 <= len(wall_cols) <= 4, f"expected 2-3 basins, got wall columns {wall_cols}"
    basins: list[dict[str, int]] = []
    for left, right in zip(wall_cols, wall_cols[1:]):
        assert right - left >= 2, "each basin needs at least one interior column"
        basins.append(
            {
                "left": int(left),
                "right": int(right),
                "width": int(right - left - 1),
            }
        )
    assert 2 <= len(basins) <= 3
    return basins


def _target_columns(basins: list[dict[str, int]]) -> list[int]:
    cols: list[int] = []
    for basin in basins:
        cols.extend(int(c) for c in range(int(basin["left"]) + 1, int(basin["right"])))
    return cols


def _stack_height_output(base: np.ndarray, height: int, basins: list[dict[str, int]]) -> np.ndarray:
    out = np.array(base, dtype=int).copy()
    for c in _target_columns(basins):
        empties = [int(r) for r in range(H - 1, -1, -1) if int(out[r, c]) == int(BG)]
        assert len(empties) >= int(height), "not enough empty depth for encoded stack height"
        for r in empties[:height]:
            out[int(r), int(c)] = int(WATER)
    return out


def main(grid: np.ndarray, check_input: bool = True, check_output: bool = True) -> np.ndarray:
    """
    Rebuild basin water from a rectangular source block, using source height as the destination stack depth.

    Input structure:
    - Background is 0 and water is 1.
    - Static non-water colors form 2 or 3 adjacent bottom-anchored basins in the lower half of the grid.
    - The only input water is one filled rectangle touching exactly one side edge and resting on an upper shelf.
    - The rectangle's row count is the code that matters; its width is not the target water amount.

    Rule:
    - Identify the source rectangle and read its height h.
    - Remove the source water, parse the basin intervals from the non-background columns on the bottom row,
      and treat every interior basin column independently.
    - In each interior basin column, fill the deepest h background cells with water.
    - Static structure colors stay unchanged, but total water pixel count is allowed to change.

    Key insight:
    - The solver must ignore literal water quantity and instead use a visible source attribute: the rectangle's
      height. The output is an h-layer restacking across the whole basin chain, even when that creates or deletes
      water compared with the input.
    """
    g = np.array(grid, dtype=int).copy()
    assert g.shape == (H, W), f"expected 10x10 grid, got {g.shape}"

    height, width, r0, c0, r1, c1, _side = _source_info(g)
    basins = _parse_basins(g)

    if check_input:
        source = {(int(r), int(c)) for r, c in np.argwhere(g == int(WATER))}
        assert _component_count(source) == 1, "water source must be one 4-connected component"
        assert np.all(g[r0 : r1 + 1, c0 : c1 + 1] == int(WATER)), "source must be a filled rectangle"
        assert int(np.sum(g == int(WATER))) == int(height * width), "there must be no water outside the source rectangle"
        assert 2 <= int(height) <= 4, "generator uses source heights 2..4"
        assert 2 <= int(width) <= 3, "generator uses compact rectangle widths 2..3"
        assert int(r1) == SHELF_ROW - 1, "source must sit directly on the upper shelf"
        assert int(g[SHELF_ROW, c0:c1 + 1].min()) != int(BG), "source shelf must support the whole rectangle"
        for r, c in source:
            assert r == H - 1 or int(g[r + 1, c]) != int(BG), "input water must be supported"
            left_bg = c > 0 and int(g[r, c - 1]) == int(BG)
            right_bg = c + 1 < W and int(g[r, c + 1]) == int(BG)
            assert not (left_bg and right_bg), "input water cannot have bg on both horizontal sides"

        target_cols = _target_columns(basins)
        assert target_cols, "need at least one basin interior column"
        for c in target_cols:
            assert np.all(g[SHELF_ROW:, c] == int(BG)), "target basin columns must be free of extra solids"

    base = g.copy()
    base[base == int(WATER)] = int(BG)
    out = _stack_height_output(base, height, basins)

    if check_output:
        target_cols = set(_target_columns(basins))
        solid_mask = (g != int(WATER)) & (g != int(BG))
        assert np.array_equal(out[solid_mask], g[solid_mask]), "structure colors must stay unchanged"
        for c in range(W):
            col_water = int(np.sum(out[:, c] == int(WATER)))
            if c in target_cols:
                assert col_water == int(height), "each basin column must receive exactly source-height water"
            else:
                assert col_water == 0, "water must appear only inside the basin interiors"

    return out


def _classic_main(grid: np.ndarray) -> np.ndarray:
    """Original deepest-fill behavior, kept only for anti-degeneracy checks."""
    g = np.array(grid, dtype=int).copy()

    def settle(r: int, c: int) -> tuple[int, int]:
        rr = int(r)
        cc = int(c)
        for _ in range(H * W):
            if rr + 1 < H and int(g[rr + 1, cc]) == int(BG):
                rr += 1
                continue

            opts = []
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
                if len(opts) != 1:
                    raise AssertionError(f"ambiguous diagonal moves: {opts}")
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
                if len(nearest) != 1:
                    raise AssertionError(f"ambiguous nearest drop columns: {nearest}")
                best = nearest[0]
                if best == int(cc):
                    raise AssertionError("classic settle found a zero-step horizontal move")
                step = -1 if best < int(cc) else 1
                nx = int(cc + step)
                if int(g[rr, nx]) != int(BG):
                    raise AssertionError("classic horizontal slide hit a blocked cell")
                cc = nx
                continue

            break
        else:
            raise AssertionError("classic water pixel did not settle")
        return int(rr), int(cc)

    for _pass in range(H * W):
        moved = False
        pts = sorted([(int(r), int(c)) for r, c in np.argwhere(g == int(WATER))], reverse=True)
        for r, c in pts:
            if int(g[r, c]) != int(WATER):
                continue
            g[r, c] = int(BG)
            rr, cc = settle(r, c)
            moved = bool(moved or ((rr, cc) != (r, c)))
            g[rr, cc] = int(WATER)
        if not moved:
            break
    else:
        raise AssertionError("classic water simulation did not converge")

    return g


def _draw_walls(grid: np.ndarray, wall_cols: tuple[int, ...], wall_tops: list[int], colors: tuple[int, int]) -> None:
    for idx, (c, top) in enumerate(zip(wall_cols, wall_tops)):
        grid[int(top) :, int(c)] = int(colors[idx % len(colors)])


def _sample_wall_tops(rng: random.Random, wall_cols: tuple[int, ...], height: int) -> list[int]:
    water_top = int(H - height)
    outer_choices = [int(r) for r in range(5, water_top + 1)]
    left_outer = int(rng.choice(outer_choices))
    right_outer = int(rng.choice(outer_choices))

    tops = {int(wall_cols[0]): left_outer, int(wall_cols[-1]): right_outer}
    for c in wall_cols[1:-1]:
        if rng.random() < 0.7 and water_top + 1 <= 9:
            short_low = max(6, water_top + 1)
            top = int(rng.randint(short_low, 9))
        else:
            tall_low = max(6, water_top - 1)
            tall_high = min(9, water_top)
            top = int(rng.randint(tall_low, tall_high))
        tops[int(c)] = top
    return [int(tops[int(c)]) for c in wall_cols]


def _build_output_case(rng: random.Random) -> tuple[np.ndarray, int, int, int]:
    height = int(rng.randint(2, 4))
    wall_cols = tuple(rng.choice(LAYOUTS))
    wall_tops = _sample_wall_tops(rng, wall_cols, height)
    colors = rng.sample(list(range(2, 10)), 3)
    shelf_color = int(colors[0])
    wall_colors = (int(colors[1]), int(colors[2]))

    base = np.zeros((H, W), dtype=int)
    _draw_walls(base, wall_cols, wall_tops, wall_colors)
    basins = _parse_basins(base)
    out = _stack_height_output(base, height, basins)
    target_width = int(len(_target_columns(basins)))
    return out, height, target_width, shelf_color


def _place_source_rectangle(grid: np.ndarray, height: int, width: int, shelf_color: int) -> None:
    assert 2 <= int(height) <= 4
    assert 2 <= int(width) <= 3
    top = int(SHELF_ROW - height)
    grid[SHELF_ROW, :width] = int(shelf_color)
    grid[top:SHELF_ROW, :width] = int(WATER)


def _flip_pair_if_needed(inp: np.ndarray, out: np.ndarray, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    if rng.random() < 0.5:
        return np.fliplr(inp).copy(), np.fliplr(out).copy()
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(400):
        rng = random.Random(random.randrange(1 << 30))
        out, height, target_width, shelf_color = _build_output_case(rng)
        src_width = int(rng.choice([2, 3]))
        assert src_width != target_width, "generator must preserve count mismatch against the target width"

        expected = np.array(out, dtype=int).copy()
        expected[SHELF_ROW, :src_width] = int(shelf_color)
        inp = np.array(expected, dtype=int).copy()
        inp[inp == int(WATER)] = int(BG)
        _place_source_rectangle(inp, height, src_width, shelf_color)
        inp, expected = _flip_pair_if_needed(inp, expected, rng)

        try:
            solved = main(inp)
            classic = _classic_main(inp)
        except Exception:
            continue

        if np.array_equal(solved, expected) and not np.array_equal(solved, classic):
            return inp

    raise RuntimeError("failed to generate a valid combo variant sample")
