# Combo variant of 28a6681f (index 2)
# combo_spec: D1/2+D7/1
# Name: 反向重力+嵌套盆地

from __future__ import annotations

import random

import numpy as np


# Design analysis
# a) 具体化与行为追踪
#    - 网格保持 10x10，背景为 0，水为 1，静态障碍用 2 种其他颜色。
#    - D7/1 的“外盆套内盆”继续保留，但整体上下翻转：外层结构是锚定在顶边的大倒 U，
#      内层结构是悬挂在其内部的小倒 U，中间仍然只有 2 列宽的中心竖井。
#    - D1/2 把重力方向从“向下沉”改成“向上浮”。因此输入水源不再放在盆口上方，而是放在中心竖井
#      下部并接到底边，看起来像从底部向上顶入倒置容器。
#    - 一个具体样例：若外壁在 c=1,8，内壁在 c=3,6，内盆下沿在 r=4，则输入会在 c=4,5 放置一根
#      连接到底边的 2 列水柱。原任务里这些水会下落进底部盆；融合后它们会先向上塞满内层倒盆
#      (r=2..4, c=4..5)，再绕过内壁下沿溢到外层侧槽 (r=1..4, c=2 与 c=7)。
#      差异不只是“上下翻面”，而是“方向翻转后嵌套拓扑的填充顺序也反了”。
#
# b) 复杂度校准
#    - 只做反向重力而不做嵌套，会退化成普通倒置单盆；只做嵌套而不翻转，视觉上又太像 D7/1。
#      这里固定两层容器同时出现，确保两个维度都可见。
#    - 外盆总宽固定 8，内盆总宽固定 4，内腔宽固定 2。这样“盆中盆”和“先内后外”的层级关系足够清楚，
#      又不会因为太多自由度引入歧义。
#    - 内盆下沿只取 r=3 或 4，对应总水量 10 或 14。两档都恰好产生完整的稳定层：
#      内盆全满，外层侧槽也形成完整连续层，避免肉眼难解释的半层残局。
#
# c) 隐式假设审查
#    - 原 solver 的局部运动规则默认“重力朝 +row 方向”。融合后这个前提失效。实现上不重写第二套物理，
#      而是把整张图上下翻转，复用原始向下模拟，再翻回来。这保证了对角滑动和“滑向最近落点”也一起被正确翻转。
#    - 原输入断言要求“每个水格下方非背景”，这和底部水源冲突。新 main() 改为检查：
#      水必须位于内盆中心竖井内、形成一块连接到底边的 2 列矩形源区。这样显式对应新生成器，而不是偷偷依赖旧断言。
#    - 原输出断言要求左/右/下都非背景；反向重力下应镜像为左/右/上都非背景。否则会把合法的“向上稳定水层”误判掉。
#    - 原 guide 检查针对上半区斜坡导流件；本融合体只使用水平/垂直墙，不再生成 guide，因此相关隐式前提直接移除。
#    - 由于 solver 通过 _find_inverted_layout() 识别倒置双层盆地，main() 隐含要求：
#      顶边有唯一外层顶梁，第二行有唯一内层顶梁，且两者宽度分别为 8 和 4。generate_input() 明确按此构造。
#
# d) 放置合法性
#    - 外盆左边界只取 0/1/2，因此宽 8 的外盆总能完整落在 10 列内。
#    - 外盆顶梁固定在 r=0，内盆顶梁固定在 r=1；外壁下沿随机取到 5..7，内壁下沿取 3..4，
#      保证内盆完全嵌在外盆内部，且外壁至少比内壁深 2 格。
#    - 底部水柱只放在中心竖井两列中，不会覆盖任何障碍，也不会越出网格。
#
# e) 视觉可观察性
#    - 人类从 2-4 个样例中应直接看到三件事：结构在顶部、存在明显“大盆包小盆”、水从底部上升后先占据中间再占据两侧。
#    - 生成器强制输出满足“内盆全满且外层有水”，因此不会出现看不出溢流先后的弱样例。
#    - 额外还检查“同一输入若按原向下重力求解，输出必须不同”，避免样本退化成方向翻转无实际作用的伪组合。
#
# f) 可视化整洁度
#    - 上半区只放倒置双层盆地，不再混入 guide 或散碎支撑块；下半区只保留一根规整的 2 列水柱。
#    - 外盆和内盆始终用不同颜色，角色单一，便于快速看出包裹关系。
#    - 水的可移动区域基本集中在中轴附近和两侧槽道，不会在整幅图中到处撒点。
#
# g) 颜色变化可推断性
#    - 颜色只区分“外层静态壁”“内层静态壁”“水”“背景”四类语义，不承载任何隐藏排序。
#    - 不依赖颜色数值大小；只要是非 0/1 的两种不同颜色即可。
#
# h) main() 的新隐式假设复查
#    - _find_inverted_layout() 假设倒置嵌套结构的几何模板固定；生成器通过固定宽度和顶锚定位置保证这一点。
#    - main() 的输入检查假设水源占满中心两列且接到底边；生成器用规则矩形水柱保证这一点。
#    - 反向物理通过上下翻转实现，因此要求结构与水都完全落在单张 10x10 网格内；生成器无越界放置。


BG = 0
WATER = 1


def _simulate_downward(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int).copy()
    height, width = g.shape

    def settle(r: int, c: int) -> tuple[int, int]:
        rr = int(r)
        cc = int(c)
        for _ in range(height * width):
            if rr + 1 < height and int(g[rr + 1, cc]) == BG:
                rr += 1
                continue

            opts: list[tuple[int, int]] = []
            if (
                rr + 1 < height
                and cc - 1 >= 0
                and int(g[rr, cc - 1]) == BG
                and int(g[rr + 1, cc - 1]) == BG
            ):
                opts.append((int(rr + 1), int(cc - 1)))
            if (
                rr + 1 < height
                and cc + 1 < width
                and int(g[rr, cc + 1]) == BG
                and int(g[rr + 1, cc + 1]) == BG
            ):
                opts.append((int(rr + 1), int(cc + 1)))
            if opts:
                assert len(opts) == 1, f"ambiguous diagonal moves: {opts}"
                rr, cc = opts[0]
                continue

            if rr + 1 < height and int(g[rr + 1, cc]) != BG:
                left = int(cc)
                while left - 1 >= 0 and int(g[rr, left - 1]) == BG:
                    left -= 1
                right = int(cc)
                while right + 1 < width and int(g[rr, right + 1]) == BG:
                    right += 1

                drops = [int(x) for x in range(int(left), int(right) + 1) if int(g[rr + 1, int(x)]) == BG]
                if not drops:
                    break

                min_dist = min(abs(int(x - cc)) for x in drops)
                nearest = [int(x) for x in drops if abs(int(x - cc)) == int(min_dist)]
                assert len(nearest) == 1, f"ambiguous nearest drop columns: {nearest}"
                best = nearest[0]
                step = -1 if best < int(cc) else 1
                nx = int(cc + step)
                assert int(g[rr, nx]) == BG
                cc = nx
                continue

            break
        else:
            raise AssertionError("water pixel did not settle")
        return int(rr), int(cc)

    for _pass in range(height * width):
        moved = False
        pts = sorted([(int(r), int(c)) for r, c in np.argwhere(g == WATER)], reverse=True)
        for r, c in pts:
            if int(g[r, c]) != WATER:
                continue
            g[r, c] = BG
            rr, cc = settle(int(r), int(c))
            moved = bool(moved or ((int(rr), int(cc)) != (int(r), int(c))))
            g[int(rr), int(cc)] = WATER
        if not moved:
            break
    else:
        raise AssertionError("water simulation did not converge")

    return g


def _find_inverted_layout(grid: np.ndarray) -> dict[str, int]:
    solids = grid > 1
    assert solids.shape == (10, 10), "this combo keeps the original 10x10 canvas"

    outer_cols = [int(c) for c in np.where(solids[0])[0]]
    assert outer_cols, "missing outer roof"
    outer_left = min(outer_cols)
    outer_right = max(outer_cols)

    inner_cols = [int(c) for c in np.where(solids[1])[0] if outer_left < int(c) < outer_right]
    assert inner_cols, "missing inner roof"
    inner_left = min(inner_cols)
    inner_right = max(inner_cols)

    outer_rows_left = [int(r) for r in np.where(solids[:, outer_left])[0]]
    outer_rows_right = [int(r) for r in np.where(solids[:, outer_right])[0]]
    inner_rows_left = [int(r) for r in np.where(solids[:, inner_left])[0]]
    inner_rows_right = [int(r) for r in np.where(solids[:, inner_right])[0]]
    assert outer_rows_left and outer_rows_right and inner_rows_left and inner_rows_right

    return {
        "outer_left": int(outer_left),
        "outer_right": int(outer_right),
        "outer_bottom": int(max(max(outer_rows_left), max(outer_rows_right))),
        "inner_left": int(inner_left),
        "inner_right": int(inner_right),
        "inner_bottom": int(max(max(inner_rows_left), max(inner_rows_right))),
    }


def _inner_cells(layout: dict[str, int]) -> list[tuple[int, int]]:
    return [
        (int(r), int(c))
        for r in range(2, int(layout["inner_bottom"]) + 1)
        for c in range(int(layout["inner_left"]) + 1, int(layout["inner_right"]))
    ]


def _annulus_cells(layout: dict[str, int]) -> list[tuple[int, int]]:
    outer_left = int(layout["outer_left"])
    outer_right = int(layout["outer_right"])
    inner_left = int(layout["inner_left"])
    inner_right = int(layout["inner_right"])
    inner_bottom = int(layout["inner_bottom"])
    return [
        (int(r), int(c))
        for r in range(1, inner_bottom + 1)
        for c in range(outer_left + 1, outer_right)
        if int(c) < inner_left or int(c) > inner_right
    ]


def _fusion_metrics(inp: np.ndarray, out: np.ndarray) -> dict[str, int]:
    layout = _find_inverted_layout(inp)
    inner_cells = _inner_cells(layout)
    annulus_cells = _annulus_cells(layout)
    input_rows = [int(r) for r, _c in np.argwhere(inp == WATER)]
    output_rows = [int(r) for r, _c in np.argwhere(out == WATER)]
    downward_baseline = _simulate_downward(inp)

    return {
        "water_total": int(np.sum(out == WATER)),
        "inner_fill": int(sum(1 for r, c in inner_cells if int(out[r, c]) == WATER)),
        "inner_capacity": int(len(inner_cells)),
        "annulus_fill": int(sum(1 for r, c in annulus_cells if int(out[r, c]) == WATER)),
        "annulus_capacity": int(len(annulus_cells)),
        "moved": int(np.any(inp != out)),
        "input_bottom_touch": int(max(input_rows) == inp.shape[0] - 1),
        "input_top": int(min(input_rows)),
        "output_top": int(min(output_rows)),
        "differs_from_downward": int(not np.array_equal(out, downward_baseline)),
    }


def _build_inverted_nested_basin(
    outer_left: int,
    outer_bottom: int,
    inner_bottom: int,
    outer_color: int,
    inner_color: int,
) -> np.ndarray:
    assert 0 <= outer_left <= 2
    assert inner_bottom in (3, 4)
    assert outer_bottom >= inner_bottom + 2
    grid = np.zeros((10, 10), dtype=int)

    outer_right = outer_left + 7
    inner_left = outer_left + 2
    inner_right = inner_left + 3

    for c in range(outer_left, outer_right + 1):
        grid[0, c] = int(outer_color)
    for r in range(0, outer_bottom + 1):
        grid[r, outer_left] = int(outer_color)
        grid[r, outer_right] = int(outer_color)

    for c in range(inner_left, inner_right + 1):
        grid[1, c] = int(inner_color)
    for r in range(1, inner_bottom + 1):
        grid[r, inner_left] = int(inner_color)
        grid[r, inner_right] = int(inner_color)

    return grid


def _place_bottom_source(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int).copy()
    layout = _find_inverted_layout(g)
    source_cols = [int(layout["inner_left"]) + 1, int(layout["inner_right"]) - 1]

    # Full inner cup + full outer side annulus gives a clean two-stage target level.
    water_count = int(len(_inner_cells(layout)) + len(_annulus_cells(layout)))
    height = water_count // 2
    start_row = g.shape[0] - height
    assert start_row >= 0

    for r in range(start_row, g.shape[0]):
        for c in source_cols:
            assert int(g[r, c]) == BG
            g[r, c] = WATER
    return g


def main(grid: np.ndarray, check_input: bool = True, check_output: bool = True) -> np.ndarray:
    """
    Simulate reverse-gravity water rising from a bottom source into nested inverted basins.

    Input structure:
    - Background is 0 and water is color 1.
    - Two other colors draw static walls: a large top-anchored inverted U and a smaller inverted U nested inside it.
    - Water begins as a bottom-connected 2-column source inside the center shaft below the inner roof, not as a top pour.

    Rule:
    - Water moves upward through background cells instead of downward.
    - When its upward path is blocked, it uses the same mirrored local physics as the original task:
      unique diagonal slip around an opening, then horizontal sliding toward the nearest column where it can keep rising.
    - Repeating that motion fills the inner inverted cup first; only after the inner cup is saturated does water spill into the outer side annulus.

    Key insight: The obstacle topology is still "nested basins", but the whole physical direction is vertically inverted. Predicting the output requires combining the mirrored motion law with the inner-first, outer-second overflow order.
    """
    g = np.array(grid, dtype=int).copy()
    height, width = g.shape

    if check_input:
        assert g.shape == (10, 10), "generator and solver both assume the original 10x10 canvas"
        layout = _find_inverted_layout(g)
        assert int(layout["outer_right"] - layout["outer_left"]) == 7, "outer inverted basin must be width 8"
        assert int(layout["inner_right"] - layout["inner_left"]) == 3, "inner inverted basin must be width 4"
        assert int(layout["outer_bottom"]) >= int(layout["inner_bottom"]) + 2, "outer basin must extend below inner basin"

        water_cells = [(int(r), int(c)) for r, c in np.argwhere(g == WATER)]
        assert water_cells, "input must contain water"
        source_cols = [int(layout["inner_left"]) + 1, int(layout["inner_right"]) - 1]
        assert sorted(set(int(c) for _r, c in water_cells)) == source_cols, "source water must occupy the center shaft"
        assert max(int(r) for r, _c in water_cells) == height - 1, "reverse-gravity source must touch the bottom edge"

        top_row = min(int(r) for r, _c in water_cells)
        expected = {(int(r), int(c)) for r in range(top_row, height) for c in source_cols}
        assert set(water_cells) == expected, "source water must form a solid 2-column rectangle"

    mirrored = np.flipud(g)
    settled = _simulate_downward(mirrored)
    out = np.flipud(settled)

    if check_output:
        assert int(np.sum(out == WATER)) == int(np.sum(g == WATER)), "water count must be conserved"
        for r, c in np.argwhere(out == WATER):
            rr = int(r)
            cc = int(c)
            assert rr == 0 or int(out[rr - 1, cc]) != BG
            assert cc == 0 or int(out[rr, cc - 1]) != BG
            assert cc == width - 1 or int(out[rr, cc + 1]) != BG

    return out


def generate_input() -> np.ndarray:
    solid_colors = [2, 3, 4, 5, 6, 7, 8, 9]

    for _ in range(200):
        inner_bottom = random.choice([3, 4])
        outer_bottom = random.randint(inner_bottom + 2, 7)
        outer_left = random.choice([0, 1, 2])
        outer_color, inner_color = random.sample(solid_colors, 2)

        base = _build_inverted_nested_basin(
            outer_left=outer_left,
            outer_bottom=outer_bottom,
            inner_bottom=inner_bottom,
            outer_color=outer_color,
            inner_color=inner_color,
        )
        inp = _place_bottom_source(base)
        out = main(inp)
        metrics = _fusion_metrics(inp, out)

        if metrics["moved"] != 1:
            continue
        if metrics["inner_fill"] != metrics["inner_capacity"]:
            continue
        if metrics["annulus_fill"] != metrics["annulus_capacity"]:
            continue
        if metrics["input_bottom_touch"] != 1:
            continue
        if metrics["output_top"] >= metrics["input_top"]:
            continue
        if metrics["differs_from_downward"] != 1:
            continue

        return inp

    raise AssertionError("failed to generate a reverse-gravity nested-basin sample")
