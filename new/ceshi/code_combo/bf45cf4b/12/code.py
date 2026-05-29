# Combo variant of bf45cf4b (index 12)
# combo_spec: D8/0+D11/2+D2/1
# Name: 面积区分双瓦片组合放大

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

try:
    from arc_utils import colors_present, crop_bbox, find_components, get_bg, paste, random_free_location
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[4] / "solvers"))
    from arc_utils import colors_present, crop_bbox, find_components, get_bg, paste, random_free_location

"""
融合设计分析

a) 具体化与行为追踪
   原题输入只有 2 个前景分量: 1 个单色控制精灵 + 1 个多色方块。输出是把控制精灵的每个实心像素
   等比放大成同一个方块。
   现在输入固定有 3 个前景分量:
   - 最大面积分量: 控制精灵, 单色, 且矩形 bbox 内有空洞
   - 另外两个较小分量: 同尺寸的多色正方形瓦片
   小例子: 控制精灵 bbox 为 5x4, 两个 3x3 瓦片分别是 A、B。若控制精灵第 1 列某格为实心, 就在输出
   的对应 3x3 区域贴 A; 第 2 列实心则贴 B; 第 3 列再贴 A; 第 4 列贴 B。空洞位置保持整块背景。
   差异不再是“选哪个单一瓦片”, 而是“两个瓦片都参与, 按列奇偶交替拼接”。

b) 复杂度校准
   这里真正新增了两个推理步骤:
   - 先按面积而不是颜色数识别角色: 最大的是控制精灵, 另外两个是瓦片
   - 再把控制精灵的空间分区(列奇偶)映射到两个瓦片
   为避免退化:
   - 两个瓦片强制同尺寸, 都至少双色, 且图案不同
   - 两个瓦片保持不含背景孔洞, 让“较小面积对象”这个线索稳定可见, 不被内部挖空干扰
   - 控制精灵强制在奇数列和偶数列上都至少有一个实心像素
   - 生成时拒绝任何“交替输出”和“单瓦片输出”完全相同的样本

c) 隐式假设审查
   原 main() 的关键前提有三条, 在融合后都失效:
   - `len(comps) == 2`: 现在必须是 3
   - 通过颜色数把“多色对象 = block、单色对象 = ctrl”直接二分: 现在有两个多色 block
   - 输出只需要一个 block: 现在需要 A/B 两个 block 同时参与
   新 main() 的每段逻辑依赖的新前提如下:
   - 组件分类: 恰有 3 个前景分量, 且最大面积唯一
   - 控制精灵: 最大面积分量是单色, 且 bbox 内有背景空洞
   - 瓦片: 其余两个分量是同尺寸正方形, 且都至少双色
   - A/B 编号: 按输入中的阅读顺序(top-left 更靠前者为 A)固定编号, 避免“两个小瓦片谁是 A”
     的欠规格问题
   生成器会逐条构造并校验这些前提。

d) 放置合法性
   生成器使用 `random_free_location(..., border=1, padding=1, padding_connectivity=8)` 随机放置 3 个对象,
   保证:
   - 不重叠
   - 不越界
   - 8 连通下彼此至少隔 1 个背景像素
   控制精灵用透明粘贴保留 bbox 内空洞; 瓦片是不透明正方形, 不会被打碎成多个组件。

e) 视觉可观察性
   人看到几组样例时, 稳定可见的线索是:
   - 输入里最大的对象总是单色带洞矩形, 因此可被识别为控制精灵
   - 另外两个较小对象都是同尺寸彩色方块, 因此可被识别为两类瓦片
   - 输出在控制精灵的相邻列之间会出现 A/B/A/B 的纵向交替节奏, 而不是整图只用一种瓦片
   - A 与 B 的编号由输入里的阅读顺序固定, 不依赖颜色编号等不可视信息

f) 可视化整洁度
   输入只含 3 个分离对象, 没有覆盖层。控制精灵和瓦片尺寸都控制在中等范围, 留有边框与间隔,
   样本不会显得拥挤。输出也仍是规则的瓦片拼接矩形, 没有附加标记。

g) 变化的可推断性
   本变体不引入抽象的颜色重映射; 输出直接复制两个输入瓦片。颜色信息只承担“识别两个不同瓦片图案”
   的职责, 核心规则由空间结构承担: 最大面积对象给出掩码, 列奇偶决定贴哪一块瓦片。

9) 新 solver 假设与 generator 保证
   Solver 假设:
   - 恰有 3 个前景组件
   - 最大面积组件唯一且为单色控制精灵
   - 另外两个组件是同面积同尺寸正方形且至少双色
   - 两瓦片按阅读顺序可稳定编号为 A/B
   Generator 额外保证:
   - 控制精灵在奇偶两类列上都存在实心像素
   - 交替输出同时不同于“全用 A”和“全用 B”的退化输出
   Generator 会显式验证这些条件; 否则丢弃样本重生。
"""

GRID_MIN = 12
GRID_MAX = 18
TILE_SIZES = (2, 3, 4)
CTRL_DIM_RANGE = range(5, 8)


def _reading_order_key(info: dict) -> tuple[int, int]:
    r0, c0, _r1, _c1 = info["bbox"]
    return int(r0), int(c0)


def _component_infos(grid: np.ndarray, bg: int) -> list[dict]:
    comps = find_components(grid, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    infos: list[dict] = []
    for comp in comps:
        patch = crop_bbox(grid, comp["bbox"])
        infos.append(
            {
                "bbox": comp["bbox"],
                "patch": patch,
                "cols": colors_present(patch, exclude={bg}),
                "shape": tuple(int(x) for x in patch.shape),
                "area": int(comp["area"]),
            }
        )
    return infos


def _classify_objects(grid: np.ndarray, bg: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    infos = _component_infos(grid, bg)
    assert len(infos) == 3

    by_area = sorted(infos, key=lambda info: int(info["area"]), reverse=True)
    ctrl_info = by_area[0]
    tile_infos = by_area[1:]

    assert int(ctrl_info["area"]) > int(tile_infos[0]["area"])
    assert int(ctrl_info["area"]) > int(tile_infos[1]["area"])
    assert len(ctrl_info["cols"]) == 1

    ctrl = ctrl_info["patch"]
    assert np.any(ctrl == int(bg))

    side = int(tile_infos[0]["shape"][0])
    tile_area = int(tile_infos[0]["area"])
    assert side == int(tile_infos[0]["shape"][1])
    assert all(info["shape"] == (side, side) for info in tile_infos)
    assert all(int(info["area"]) == tile_area for info in tile_infos)
    assert all(len(info["cols"]) >= 2 for info in tile_infos)

    tile_infos = sorted(tile_infos, key=_reading_order_key)
    tile_a = tile_infos[0]["patch"]
    tile_b = tile_infos[1]["patch"]
    return ctrl, tile_a, tile_b


def _compose_output(tile_a: np.ndarray, tile_b: np.ndarray, ctrl: np.ndarray, bg: int) -> np.ndarray:
    th, tw = tile_a.shape
    assert tile_b.shape == (th, tw)
    ch, cw = ctrl.shape
    out = np.full((ch * th, cw * tw), int(bg), dtype=int)
    for r in range(ch):
        for c in range(cw):
            if int(ctrl[r, c]) == int(bg):
                continue
            tile = tile_a if (c % 2 == 0) else tile_b
            paste(out, tile, r * th, c * tw)
    return out


def _compose_single_tile_output(tile: np.ndarray, ctrl: np.ndarray, bg: int) -> np.ndarray:
    th, tw = tile.shape
    ch, cw = ctrl.shape
    out = np.full((ch * th, cw * tw), int(bg), dtype=int)
    for r in range(ch):
        for c in range(cw):
            if int(ctrl[r, c]) != int(bg):
                paste(out, tile, r * th, c * tw)
    return out


def _tile_difference_score(tile1: np.ndarray, tile2: np.ndarray) -> int:
    assert tile1.shape == tile2.shape
    return int(np.sum(tile1 != tile2))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Largest-area component is the control sprite. The two smaller same-size square components are
    tile A and tile B, ordered by reading order in the input. Filled pixels in 1-based odd control
    columns receive tile A; filled pixels in 1-based even control columns receive tile B.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    ctrl, tile_a, tile_b = _classify_objects(g, bg)
    return _compose_output(tile_a, tile_b, ctrl, bg)


def _neighbors8(h: int, w: int, r: int, c: int) -> list[tuple[int, int]]:
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr = r + dr
            cc = c + dc
            if 0 <= rr < h and 0 <= cc < w:
                out.append((rr, cc))
    return out


def _make_tile(bg: int, fg_colors: list[int], size: int, rng: np.random.Generator) -> np.ndarray:
    n_colors = int(rng.integers(2, min(4, len(fg_colors)) + 1))
    palette = [int(x) for x in rng.choice(fg_colors, size=n_colors, replace=False)]
    tile = np.full((size, size), palette[0], dtype=int)
    stride = max(1, size // 2)
    pattern = int(rng.integers(5))

    for r in range(size):
        for c in range(size):
            if pattern == 0:
                idx = (r + c) % n_colors
            elif pattern == 1:
                idx = c % n_colors
            elif pattern == 2:
                idx = r % n_colors
            elif pattern == 3:
                idx = min(r, c, size - 1 - r, size - 1 - c) % n_colors
            else:
                idx = ((r // stride) + (c // stride)) % n_colors
            tile[r, c] = palette[idx]

    if len(colors_present(tile, exclude={bg})) < 2:
        tile[0, -1] = palette[1]
    return tile


def _make_ctrl(bg: int, color: int, h: int, w: int, min_fill: int, rng: np.random.Generator) -> np.ndarray:
    area = h * w
    lower = max(h + w - 1, min_fill, int(np.ceil(0.45 * area)))
    upper = min(area - 1, int(np.floor(0.72 * area)))
    assert lower <= upper

    for _ in range(200):
        target = int(rng.integers(lower, upper + 1))
        ar = int(rng.integers(0, h))
        ac = int(rng.integers(0, w))
        mask = np.zeros((h, w), dtype=bool)
        mask[:, ac] = True
        mask[ar, :] = True

        while int(mask.sum()) < target:
            frontier = []
            for r in range(h):
                for c in range(w):
                    if mask[r, c]:
                        continue
                    if any(mask[rr, cc] for rr, cc in _neighbors8(h, w, r, c)):
                        frontier.append((r, c))
            if not frontier:
                break
            pick = frontier[int(rng.integers(len(frontier)))]
            mask[pick] = True

        if int(mask.sum()) != target:
            continue
        if not np.any(mask[:, ::2]):
            continue
        if not np.any(mask[:, 1::2]):
            continue
        if bool(mask.all()):
            continue

        ctrl = np.full((h, w), int(bg), dtype=int)
        ctrl[mask] = int(color)
        return ctrl

    raise RuntimeError("failed to make control sprite")


def _make_input(
    tile1: np.ndarray,
    tile2: np.ndarray,
    ctrl: np.ndarray,
    bg: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, tuple[str, str]]:
    objects = [("tile1", tile1), ("tile2", tile2), ("ctrl", ctrl)]
    max_h = max(int(obj.shape[0]) for _name, obj in objects)
    max_w = max(int(obj.shape[1]) for _name, obj in objects)
    h_min = max(GRID_MIN, max_h + 2)
    w_min = max(GRID_MIN, max_w + 2)

    for _ in range(100):
        H = int(rng.integers(h_min, GRID_MAX + 1))
        W = int(rng.integers(w_min, GRID_MAX + 1))
        grid = np.full((H, W), int(bg), dtype=int)
        positions: dict[str, tuple[int, int]] = {}
        order = [0, 1, 2]
        rng.shuffle(order)
        ok = True

        for idx in order:
            name, patch = objects[idx]
            try:
                pos = random_free_location(
                    grid,
                    patch,
                    transparent=bg,
                    border=1,
                    padding=1,
                    padding_connectivity=8,
                )
            except ValueError:
                ok = False
                break
            paste(grid, patch, pos[0], pos[1], transparent=bg)
            positions[name] = pos

        if not ok:
            continue

        ordered = tuple(
            name
            for name, _pos in sorted(
                ((name, positions[name]) for name in ("tile1", "tile2")),
                key=lambda item: item[1],
            )
        )
        return grid, ordered

    raise RuntimeError("placement failed")


def _fusion_metric(grid: np.ndarray) -> dict[str, int]:
    bg = int(get_bg(grid))
    ctrl, tile_a, tile_b = _classify_objects(grid, bg)
    mask = (ctrl != int(bg))
    odd_count = int(mask[:, ::2].sum())
    even_count = int(mask[:, 1::2].sum())
    mixed = _compose_output(tile_a, tile_b, ctrl, bg)
    only_a = _compose_single_tile_output(tile_a, ctrl, bg)
    only_b = _compose_single_tile_output(tile_b, ctrl, bg)
    return {
        "odd_count": odd_count,
        "even_count": even_count,
        "mixed_diff_from_a": int(not np.array_equal(mixed, only_a)),
        "mixed_diff_from_b": int(not np.array_equal(mixed, only_b)),
        "tile_diff_cells": _tile_difference_score(tile_a, tile_b),
    }


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    palette = [int(x) for x in rng.choice(np.arange(10, dtype=int), size=7, replace=False)]
    bg = int(palette[0])
    ctrl_color = int(palette[1])
    tile_palette = palette[2:]

    side = int(rng.choice(TILE_SIZES))
    ctrl_shapes = [(h, w) for h in CTRL_DIM_RANGE for w in CTRL_DIM_RANGE if w >= 4]
    ch, cw = ctrl_shapes[int(rng.integers(len(ctrl_shapes)))]

    tile1 = _make_tile(bg, tile_palette, side, rng)
    tile2 = _make_tile(bg, tile_palette, side, rng)
    if np.array_equal(tile1, tile2):
        return None
    if _tile_difference_score(tile1, tile2) < max(2, side):
        return None

    ctrl = _make_ctrl(bg, ctrl_color, ch, cw, side * side + 1, rng)
    inp, ordered_tiles = _make_input(tile1, tile2, ctrl, bg, rng)
    tile_map = {"tile1": tile1, "tile2": tile2}
    tile_a = tile_map[ordered_tiles[0]]
    tile_b = tile_map[ordered_tiles[1]]

    expected = _compose_output(tile_a, tile_b, ctrl, bg)
    only_a = _compose_single_tile_output(tile_a, ctrl, bg)
    only_b = _compose_single_tile_output(tile_b, ctrl, bg)

    infos = _component_infos(inp, bg)
    if len(infos) != 3:
        return None

    by_area = sorted(infos, key=lambda info: int(info["area"]), reverse=True)
    if not (
        int(by_area[0]["area"]) > int(by_area[1]["area"])
        and int(by_area[0]["area"]) > int(by_area[2]["area"])
    ):
        return None
    if len(by_area[0]["cols"]) != 1:
        return None

    tile_infos = sorted(by_area[1:], key=_reading_order_key)
    if int(tile_infos[0]["area"]) != int(tile_infos[1]["area"]):
        return None
    if any(info["shape"] != (side, side) for info in tile_infos):
        return None
    if any(len(info["cols"]) < 2 for info in tile_infos):
        return None

    metric = _fusion_metric(inp)
    if metric["odd_count"] == 0 or metric["even_count"] == 0:
        return None
    if metric["mixed_diff_from_a"] == 0 or metric["mixed_diff_from_b"] == 0:
        return None

    if np.array_equal(expected, only_a) or np.array_equal(expected, only_b):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(400):
        inp = _gen_attempt(rng)
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate valid input")


def audit_variant_behavior(trials: int = 64) -> dict[str, object]:
    """
    Quantify that generated samples actually exercise the fused rule:
    both column parities are active, the alternating output differs from
    the all-A baseline, the all-B baseline, and the two tiles are visibly distinct.
    """
    rng = np.random.default_rng()
    metrics: list[dict[str, int]] = []
    for _ in range(int(trials)):
        inp = _gen_attempt(rng)
        if inp is None:
            continue
        metrics.append(_fusion_metric(inp))

    if not metrics:
        raise RuntimeError("audit failed to collect any valid samples")

    odd_counts = [m["odd_count"] for m in metrics]
    even_counts = [m["even_count"] for m in metrics]
    tile_diffs = [m["tile_diff_cells"] for m in metrics]
    diff_from_a = sum(m["mixed_diff_from_a"] for m in metrics)
    diff_from_b = sum(m["mixed_diff_from_b"] for m in metrics)
    return {
        "samples": len(metrics),
        "odd_count_range": [min(odd_counts), max(odd_counts)],
        "even_count_range": [min(even_counts), max(even_counts)],
        "tile_diff_range": [min(tile_diffs), max(tile_diffs)],
        "diff_from_a_rate": diff_from_a / len(metrics),
        "diff_from_b_rate": diff_from_b / len(metrics),
    }
