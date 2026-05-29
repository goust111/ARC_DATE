# Variant of 28a6681f
# Dimension: D1 — 流动路径规则 (path)
# Change: gravity + flow——水像素优先垂直下落，被阻挡时通过唯一开放侧的对角线滑动，支撑后沿水平方向滑向最近可落点... → BFS洪水填充式扩散

from __future__ import annotations

import random

import numpy as np


DESIGN_ANALYSIS = """
Variant design analysis
=======================

a) Concrete behavior and trace
- 保留原题的核心可见对象：`0` 是背景，`1` 是水，其余 2-3 种颜色仍是静态导流/盆地结构；底部依旧是相邻、可互相溢连的盆地。
- 为了让 BFS 扩散与原重力沉降产生稳定可见差异，生成器把水源做成贴边的“已装满小水仓”，只留一个侧向出口进入盆地区域。
- 新 solver 不再把每个蓝像素当作一个单独掉落的小球，而是把所有输入水像素只解释成：
  1. 水的总量；
  2. BFS 的发源前沿。
- 具体追踪一个小例子：
  原逻辑：右上角水仓的水会逐像素从出口掉入右盆地底部，填高后再越过较矮隔墙流入左侧盆地。
  变体逻辑：先把输入水仓清空，再从出口开始做 4 邻域、整层同步的 BFS 扩散；离出口最近的上层走廊、侧袋和盆口会先被占据，只有水量足够时才向更深处扩展。
  差异：原题输出集中在低处、且全部“有支撑”；变体输出会保留明显的高位水块和悬空水块，形成更均匀的扩散前沿。

b) Complexity calibration
- 只改 D1：静态结构还是底部锚定盆地与上方导流件，不引入新的颜色语义，也不引入额外标记。
- 为避免 BFS 退化成“把整片天空都染满”，生成器提高了上方导流/封顶结构的使用频率，把可达背景限制在盆地网络附近。
- 水量限制在 8 或 10 个像素，既能跨过 2-3 层 BFS 前沿，看出“同步扩散”，又不会把整个可达腔体全部填满。

c) Hidden assumptions in the original main() that must change
- 原 `settle()` 逐像素模拟“下落/对角滑动/水平找落点”，隐含前提是每个水像素总能被视为独立粒子；变体中这一前提失效。
- 原外层多轮迭代依赖“局部移动直到全局稳定”；BFS 版本不再追求局部稳定，而是一次性按距离层数分配全部水量。
- 原输出检查要求每个水像素左/下/右都有支撑；这正是旧重力行为的结果。变体中高位和悬空水是正常现象，所以该检查必须删除。
- 原输入检查中的“水必须被支撑”仍然保留，因为生成器仍把输入画成靠边水仓/导流件上的稳定水源，这有助于维持原题视觉语义。

d) Placement legality
- 生成器固定使用 10x10 画布，并只在网格内部放置导流顶棚、隔墙、底部基座和水仓支撑。
- 右侧模板先构造一个 2x(4 或 5) 的满水仓，出口左侧只留单格开口；随后在下方放置 2 或 3 个底部相邻盆地。
- 再以 50% 概率做左右镜像，得到左侧来水版本；镜像不会破坏“唯一出口”“底部锚定”“相邻盆地”的合法性。
- 生成后会同时运行新 solver 和旧重力基线。若新 solver 水量不守恒、旧基线在该布局上报错、或场景越界/重叠，样本直接丢弃重采。

e) Visual observability
- 解题者能直接看到：输入里蓝色块贴在边缘导流仓内；输出里蓝色优先铺开在离出口最近的上层通道和盆口，而不是全部沉到最底部。
- 为了避免靠不可见 tie-break 学规则，生成器要求样本至少有 3 层 BFS 前沿，并且新旧输出差异足够大；这样即使最后一层偶尔是部分填充，人眼仍能从主体形状看出“按距离扩散”。

f) Visual cleanliness
- 结构层数保持简单：背景、蓝水、上方导流色、底部盆地色，偶尔再加一个强调色给某根隔墙。
- 盆地和导流件始终分布在固定的大区域：水仓在一侧上方，盆地在下半部，避免对象相互遮挡导致画面杂乱。

g) Color inference
- 颜色不参与排序或映射推理；只有可见几何关系起作用。
- 所有非 0/1 颜色都只是“不可穿透固体”，解题关键完全是 BFS 距离，而不是颜色 id。

h) Quantified anti-degeneration checks used by the generator
- `delta_vs_gravity`: 新 BFS 输出与旧重力基线输出不同的像素数，要求至少为 6。
- `unsupported_water`: 新输出中下方仍是背景的水像素数，要求至少为 2；旧重力版理论上该值为 0。
- `height_gap`: 旧重力输出的平均行号减去新输出的平均行号，要求至少为 1.0，确保 BFS 水整体更“高”。
- `layer_count`: 真实被填充的 BFS 层数，要求至少为 3。

i) New solver assumptions and generator guarantees
- 新 solver 假设输入水只是“总量 + 发源位置”，最终输出水只会落在原背景格，不会保留在输入水仓里；生成器始终按这个语义构造样本。
- 新 solver 假设至少有一个背景出口与水源 4 邻接；生成器通过单出口水仓保证这一点。
- 新 solver 假设可达背景格数量不少于输入水量；生成器运行后验验证，若不满足就重采。
"""


BG = 0
WATER = 1
GRID_SHAPE = (10, 10)
GEN_RETRIES = 400


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _water_positions(grid: np.ndarray) -> list[tuple[int, int]]:
    return [(int(r), int(c)) for r, c in np.argwhere(np.asarray(grid) == WATER)]


def _diffusion_plan(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[list[tuple[int, int]]], bool, int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    sources = _water_positions(g)
    remaining = len(sources)

    seen = np.zeros((h, w), dtype=bool)
    frontier = sorted(sources)
    for r, c in frontier:
        seen[r, c] = True

    filled: list[tuple[int, int]] = []
    layers: list[list[tuple[int, int]]] = []
    partial_last_layer = False

    while frontier and remaining > 0:
        layer: set[tuple[int, int]] = set()
        for r, c in frontier:
            for rr, cc in _neighbors4(r, c, h, w):
                if seen[rr, cc]:
                    continue
                if int(g[rr, cc]) != BG:
                    continue
                seen[rr, cc] = True
                layer.add((rr, cc))

        if not layer:
            break

        ordered = sorted(layer)
        if remaining >= len(ordered):
            chosen = ordered
            frontier = ordered
            remaining -= len(ordered)
        else:
            chosen = ordered[:remaining]
            frontier = []
            remaining = 0
            partial_last_layer = True

        layers.append(chosen)
        filled.extend(chosen)

    return filled, layers, partial_last_layer, remaining


def main(grid: np.ndarray, check_input: bool = True, check_output: bool = True) -> np.ndarray:
    """
    Redistribute source water by breadth-first diffusion through the basin network.

    Input structure:
    - bg = 0. Water pixels = color 1 (blue).
    - Other colors remain static solid walls, basin rims, and optional upper guides.
    - Water starts as a supported source block against an edge or guide-adjacent pocket above neighboring basins.

    Rule:
    - Treat every input water pixel as one unit of available water and as a BFS source location.
    - Remove the source block, then expand a 4-neighbor wave through background cells only.
    - Each full BFS layer fills simultaneously; if the last layer is only partly needed, use reading order within that layer to keep the result deterministic.

    Key insight: Water no longer hunts for the deepest stable resting place. The output is the nearest reachable background prefix around the source opening, so upper shelves and side pockets fill before distant lower cells.
    """
    g = np.array(grid, dtype=int).copy()
    h, w = g.shape
    assert (h, w) == GRID_SHAPE, f"expected {GRID_SHAPE}, got {(h, w)}"

    sources = _water_positions(g)
    if check_input:
        assert sources, "expected at least one water source"
        for r, c in sources:
            assert r == h - 1 or int(g[r + 1, c]) != BG, "input water must be supported"
        assert any(
            int(g[rr, cc]) == BG
            for r, c in sources
            for rr, cc in _neighbors4(r, c, h, w)
        ), "source water must touch background through at least one side"

    source_count = len(sources)
    out = g.copy()
    out[out == WATER] = BG

    filled, _layers, _partial_last_layer, remaining = _diffusion_plan(g)
    assert remaining == 0, "not enough reachable background cells for BFS diffusion"

    for r, c in filled:
        out[r, c] = WATER

    if check_output:
        assert int(np.sum(out == WATER)) == source_count, "water count must be conserved"
        solid_mask = (g != BG) & (g != WATER)
        assert np.array_equal(out[solid_mask], g[solid_mask]), "solid structure must stay unchanged"
        for r, c in np.argwhere(out == WATER):
            assert int(g[int(r), int(c)]) == BG, "output water must occupy former background cells"

    return out


def _gravity_baseline(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int).copy()
    h, w = g.shape

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
                opts.append((rr + 1, cc - 1))
            if (
                rr + 1 < h
                and cc + 1 < w
                and int(g[rr, cc + 1]) == BG
                and int(g[rr + 1, cc + 1]) == BG
            ):
                opts.append((rr + 1, cc + 1))
            if opts:
                if len(opts) != 1:
                    raise AssertionError(f"ambiguous diagonal moves: {opts}")
                rr, cc = opts[0]
                continue

            if rr + 1 < h and int(g[rr + 1, cc]) != BG:
                left = cc
                while left - 1 >= 0 and int(g[rr, left - 1]) == BG:
                    left -= 1
                right = cc
                while right + 1 < w and int(g[rr, right + 1]) == BG:
                    right += 1

                drops = [x for x in range(left, right + 1) if int(g[rr + 1, x]) == BG]
                if not drops:
                    break

                min_dist = min(abs(x - cc) for x in drops)
                nearest = [x for x in drops if abs(x - cc) == min_dist]
                if len(nearest) != 1:
                    raise AssertionError(f"ambiguous nearest drop columns: {nearest}")
                best = nearest[0]
                step = -1 if best < cc else 1
                nx = cc + step
                if int(g[rr, nx]) != BG:
                    raise AssertionError("horizontal slide blocked")
                cc = nx
                continue

            break
        else:
            raise AssertionError("water pixel did not settle")
        return rr, cc

    for _pass in range(h * w):
        moved = False
        pts = sorted(_water_positions(g), reverse=True)
        for r, c in pts:
            if int(g[r, c]) != WATER:
                continue
            g[r, c] = BG
            rr, cc = settle(r, c)
            moved = bool(moved or ((rr, cc) != (r, c)))
            g[rr, cc] = WATER
        if not moved:
            break
    else:
        raise AssertionError("water simulation did not converge")

    return g


def _build_candidate() -> np.ndarray:
    g = np.zeros(GRID_SHAPE, dtype=int)
    palette = random.sample([2, 3, 4, 5, 6, 7, 8, 9], random.choice([2, 3]))
    guide_color = palette[0]
    wall_color = palette[1]
    accent_color = palette[2] if len(palette) == 3 else wall_color

    entry_r = random.choice([3, 4])
    left_wall = random.choice([1, 2])
    config = random.choice(["two", "three"])
    right_wall = 8

    if config == "two":
        inner_cols = [random.choice([4, 5])]
        inner_tops = [random.choice([entry_r + 2, entry_r + 3])]
    else:
        left_wall = 1
        inner_cols = [4, 6]
        inner_tops = [random.choice([entry_r + 2, entry_r + 3]), random.choice([entry_r + 1, entry_r + 2])]

    # Upper guide/roof: keeps the diffusion local so the D1 change is visually readable.
    g[entry_r - 1, left_wall:8] = guide_color
    g[:entry_r, 7] = guide_color
    if entry_r == 4 and random.random() < 0.5:
        g[0:2, 6] = guide_color

    # Filled side reservoir. It is fully occupied in the input, so the new solver can use it
    # as water amount + source frontier without immediately refilling the reservoir cells.
    g[: entry_r + 1, 8:10] = WATER
    g[entry_r + 1, 8:10] = guide_color

    # Bottom-anchored basin network.
    g[9, left_wall : right_wall + 1] = wall_color
    g[entry_r:10, left_wall] = wall_color
    g[entry_r + 1 : 10, right_wall] = accent_color

    for idx, col in enumerate(inner_cols):
        color = wall_color if idx == 0 else accent_color
        g[inner_tops[idx] : 10, col] = color

    # Optional small support nub for organic variation without changing the core path rule.
    if config == "two" and random.random() < 0.5:
        nub_col = inner_cols[0] - 1 if inner_cols[0] - left_wall >= 2 else inner_cols[0] + 1
        if left_wall < nub_col < right_wall:
            g[8, nub_col] = accent_color

    if random.random() < 0.5:
        g = np.fliplr(g)

    return g


def _count_unsupported(out: np.ndarray) -> int:
    h, _w = out.shape
    count = 0
    for r, c in _water_positions(out):
        if r + 1 < h and int(out[r + 1, c]) == BG:
            count += 1
    return count


def _behavior_metrics(grid: np.ndarray) -> dict[str, float | int | bool]:
    new_out = main(grid)
    old_out = _gravity_baseline(grid)
    filled, layers, partial_last_layer, remaining = _diffusion_plan(grid)
    assert remaining == 0

    new_rows = [r for r, _c in _water_positions(new_out)]
    old_rows = [r for r, _c in _water_positions(old_out)]

    return {
        "delta_vs_gravity": int(np.sum(new_out != old_out)),
        "unsupported_water": int(_count_unsupported(new_out)),
        "height_gap": float(np.mean(old_rows) - np.mean(new_rows)),
        "layer_count": int(len(layers)),
        "partial_last_layer": bool(partial_last_layer),
        "filled_cells": int(len(filled)),
    }


def generate_input() -> np.ndarray:
    last_error: Exception | None = None
    for _ in range(GEN_RETRIES):
        candidate = _build_candidate()
        try:
            metrics = _behavior_metrics(candidate)
        except Exception as exc:
            last_error = exc
            continue

        if metrics["delta_vs_gravity"] < 6:
            continue
        if metrics["unsupported_water"] < 2:
            continue
        if metrics["height_gap"] < 1.0:
            continue
        if metrics["layer_count"] < 3:
            continue
        if metrics["partial_last_layer"]:
            continue

        return candidate

    if last_error is not None:
        raise RuntimeError(f"failed to generate a valid BFS-diffusion sample: {last_error}")
    raise RuntimeError("failed to generate a valid BFS-diffusion sample")
