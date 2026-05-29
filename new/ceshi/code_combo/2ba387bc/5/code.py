# Combo variant of 2ba387bc (index 5)
# combo_spec: D1/0+D12/1
# Name: 边框分类+框架内部第三色填充
#
# Design analysis
# a) 具体化与行为追踪
#    - 原题对象有两类：
#        1. frame = 单色 1px 边框 + 背景色 2x2 内芯
#        2. solid = 4x4 全同色
#      原 solver 用 `inner == bg` 判断 frame。
#    - 这个融合版必须同时体现两件事：
#        1. D12/1: 左类对象的 2x2 内芯不再是背景，而是非背景色。
#        2. D1/0: 分类不再看内芯，而只看 12 个边框像素是否全同色。
#    - 若只把原 frame 改成“单色边框 + 彩色内芯”，但仍保留原 solid 为全同色，
#      那么边框一致性分类会退化：全同色 solid 的边框也同色，无法和 frame 分开。
#      所以这里做必要的配套调整：右类对象也改成有彩色 2x2 内芯，但边框故意破成双色。
#    - 因此两类对象具体长相为：
#      左类（uniform-border）：
#          a a a a
#          a b b a
#          a b b a
#          a a a a
#        边框 12 格全是 a，内芯 2x2 全是 b，且 a != b。
#      右类（broken-border）：
#          b b b b
#          a b b a
#          a b b a
#          a a a a
#        或其上下左右四种朝向之一。内芯仍是 b，但边框同时含 a、b 两色。
#    - 小例子追踪：
#      原逻辑看到左类对象时，`sub[1:3,1:3] != bg`，会把它误判为 solid。
#      新逻辑读取边框颜色集合：左类集合大小为 1，所以分到左列；右类集合大小为 2，
#      所以分到右列。两类对象都带非背景内芯，说明“看中心是否为背景”已完全失效。
# b) 复杂度校准
#    - 两类对象都只使用 2 种非背景色，避免出现“左列是双色、右列是三色”的简化捷径。
#    - 两类对象都带统一 2x2 内芯，避免解题者仅凭“有没有内芯”分类。
#    - 生成器强制至少一对左右对象共享同一组颜色 `(a, b)`，所以“颜色对本身”也不足以分类，
#      只能看边框是否一致。
# c) 隐式假设审查（main 逐段）
#    - `find_components(..., monochromatic=False)` 依赖对象跨颜色时仍保持 4 连通。
#      这里所有 4x4 patch 都被完全填满，所以该假设仍成立。
#    - `bbox_h == bbox_w == 4` 依赖所有有效对象仍是 4x4。生成器只放置 4x4 patch，成立。
#    - 原代码的 `inner == bg` 依赖“frame 的内芯就是背景”。融合后这个前提彻底失效，必须删除。
#    - 新代码改为 `len(border_colors) == 1`。这带来新的输入假设：
#        1. 左类对象边框必须严格单色。
#        2. 右类对象边框必须至少含两色。
#      生成器分别用 `_make_uniform_border_patch` 和 `_make_broken_border_patch` 保证。
#    - 原代码把“非 frame 的 4x4 对象都归为 solid”。这里仍沿用这个兜底，
#      但 generator 只产生上述两类合法对象，所以不会混入第三类 4x4 干扰图案。
# d) 放置合法性
#    - 每个对象都是无背景孔洞的完整 4x4 patch。
#    - `random_free_location(..., border=1, padding=1, padding_connectivity=4)` 保证对象之间
#      不重叠、不越界、且不会 4 邻接粘连，solver 能稳定提取每个 4x4 bbox。
# e) 视觉可观察性
#    - 看到 2-4 组样例时，人类会发现：左列对象的边框总是完整单色环，右列对象的边框总有一侧
#      与其余边不同；而两列对象都拥有彩色中心，因此“中心是否空洞”不再是线索。
#    - 排序仍按输入行坐标，从上到下堆叠；没有使用颜色值大小等不可视信息。
# f) 可视化整洁度
#    - 仍是单层散落 4x4 对象，没有额外标记、覆盖或跨对象连接。
#    - 右类对象只破坏一整条边，模式清楚，不会因为随机噪声导致视觉杂乱。
# g) 变化可推断性
#    - 所有对象都恰好 2 色，且左右两类复用同一小调色板，避免“颜色多少/绝对颜色”成为捷径。
#    - 真正稳定的结构线索只有“边框是否全同色”，这正是融合规则要暴露的核心。
# h) 量化退化检查思路
#    - 指标 1：`legacy_diff`，比较本 solver 与原始 solver 在同一输入上的输出是否不同。
#      这里要求应接近 100%，否则说明样本退化回旧规则。
#    - 指标 2：`mixed_with_uniform_center`，统计右类对象中“边框多色但 2x2 内芯单色”的数量。
#      该值高说明右类仍带 frame-like 内芯，分类必须看边框。
#    - 指标 3：`shared_color_overlap`，统计左右两类所用颜色集合交集大小。该值至少为 2，
#      说明颜色身份不能直接决定类别。
# 9) main 的新隐式假设与 generator 对应保证
#    - 假设每个有效对象是单个 4 连通 4x4 patch：generator 保证。
#    - 假设左类边框单色、右类边框非单色：generator 保证。
#    - 假设对象按 row/col 排序即可复现目标顺序：generator 不引入需要别的排序键的信号。
#    - 假设背景仍是最常见颜色：大画布 + 少量 4x4 小块保证背景占优。

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SOLVER_UTILS = Path("/home/lixinyu/snap/new/solvers")
if str(SOLVER_UTILS) not in sys.path:
    sys.path.append(str(SOLVER_UTILS))

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, paste, random_free_location


SIDES = ("top", "bottom", "left", "right")


def _border_values(sub: np.ndarray) -> np.ndarray:
    return np.concatenate((sub[0, :], sub[-1, :], sub[1:-1, 0], sub[1:-1, -1]))


def _border_color_count(sub: np.ndarray) -> int:
    return len(set(map(int, _border_values(sub).flatten())))


def _extract_objects(grid: np.ndarray) -> tuple[int, list[tuple[int, int, np.ndarray]], list[tuple[int, int, np.ndarray]]]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")

    uniform_border: list[tuple[int, int, np.ndarray]] = []
    broken_border: list[tuple[int, int, np.ndarray]] = []

    for comp in comps:
        b = comp["bbox"]
        if bbox_h(b) != 4 or bbox_w(b) != 4:
            continue
        r0, c0, _r1, _c1 = b
        sub = crop_bbox(g, b)
        if _border_color_count(sub) == 1:
            uniform_border.append((int(r0), int(c0), sub))
        else:
            broken_border.append((int(r0), int(c0), sub))

    uniform_border.sort(key=lambda t: (int(t[0]), int(t[1])))
    broken_border.sort(key=lambda t: (int(t[0]), int(t[1])))
    return bg, uniform_border, broken_border


def _build_output(
    bg: int,
    uniform_border: list[tuple[int, int, np.ndarray]],
    broken_border: list[tuple[int, int, np.ndarray]],
) -> np.ndarray:
    out_h = 4 * max(int(len(uniform_border)), int(len(broken_border)))
    out = np.full((out_h, 8), int(bg), dtype=int)

    for i, (_r0, _c0, sub) in enumerate(uniform_border):
        paste(out, sub, int(i * 4), 0)
    for i, (_r0, _c0, sub) in enumerate(broken_border):
        paste(out, sub, int(i * 4), 4)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort scattered 4x4 objects into two columns by border consistency.

    Left column: objects whose 12 border cells are all the same color.
    Right column: objects whose border contains multiple colors.

    Both classes may have a colored 2x2 interior, so the original "center == bg" test is invalid.
    """
    bg, uniform_border, broken_border = _extract_objects(np.array(grid, dtype=int))
    return _build_output(bg, uniform_border, broken_border)


def _legacy_main(grid: np.ndarray) -> np.ndarray:
    """Original 2ba387bc rule, kept only for generator-side degeneration checks."""
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")

    frames: list[tuple[int, int, np.ndarray]] = []
    solids: list[tuple[int, int, np.ndarray]] = []
    for comp in comps:
        b = comp["bbox"]
        if bbox_h(b) != 4 or bbox_w(b) != 4:
            continue
        r0, c0, _r1, _c1 = b
        sub = crop_bbox(g, b)
        if _border_color_count(sub) == 1 and bool((sub[1:3, 1:3] == int(bg)).all()):
            frames.append((int(r0), int(c0), sub))
        else:
            solids.append((int(r0), int(c0), sub))

    frames.sort(key=lambda t: (int(t[0]), int(t[1])))
    solids.sort(key=lambda t: (int(t[0]), int(t[1])))
    return _build_output(bg, frames, solids)


def _make_uniform_border_patch(border_col: int, inner_col: int) -> np.ndarray:
    patch = np.full((4, 4), int(border_col), dtype=int)
    patch[1:3, 1:3] = int(inner_col)
    return patch


def _make_broken_border_patch(border_col: int, inner_col: int, side: str) -> np.ndarray:
    patch = _make_uniform_border_patch(border_col, inner_col)
    if side == "top":
        patch[0, :] = int(inner_col)
    elif side == "bottom":
        patch[-1, :] = int(inner_col)
    elif side == "left":
        patch[:, 0] = int(inner_col)
    elif side == "right":
        patch[:, -1] = int(inner_col)
    else:
        raise ValueError(f"unknown side: {side}")
    return patch


def _random_pair(palette: list[int]) -> tuple[int, int]:
    cols = np.random.choice(np.array(palette, dtype=int), size=2, replace=False)
    return int(cols[0]), int(cols[1])


def _random_side() -> str:
    return str(np.random.choice(np.array(SIDES, dtype=object)))


def _sample_metrics(grid: np.ndarray) -> dict[str, int]:
    bg, uniform_border, broken_border = _extract_objects(grid)
    left_colors = {int(v) for _r, _c, sub in uniform_border for v in np.unique(sub)}
    right_colors = {int(v) for _r, _c, sub in broken_border for v in np.unique(sub)}
    mixed_with_uniform_center = sum(
        1
        for _r, _c, sub in broken_border
        if len(set(map(int, sub[1:3, 1:3].flatten()))) == 1
    )
    all_two_color_objects = sum(
        1
        for _r, _c, sub in (*uniform_border, *broken_border)
        if len(set(map(int, sub.flatten())) - {int(bg)}) == 2
    )
    legacy_diff = int(not np.array_equal(main(grid), _legacy_main(grid)))
    return {
        "uniform_border_count": int(len(uniform_border)),
        "broken_border_count": int(len(broken_border)),
        "mixed_with_uniform_center": int(mixed_with_uniform_center),
        "shared_color_overlap": int(len(left_colors & right_colors)),
        "all_two_color_objects": int(all_two_color_objects),
        "legacy_diff": int(legacy_diff),
    }


def _gen_attempt() -> np.ndarray | None:
    bg = 0
    h = int(np.random.randint(18, 30))
    w = int(np.random.randint(18, 30))
    g = np.full((h, w), bg, dtype=int)

    n_uniform = int(np.random.randint(1, 5))
    n_broken = int(np.random.randint(1, 5))
    palette_size = int(np.random.randint(3, 6))
    palette = list(map(int, np.random.choice(np.arange(1, 10), size=palette_size, replace=False)))

    try:
        shared_border, shared_inner = _random_pair(palette)

        uniform_specs: list[tuple[int, int]] = [(shared_border, shared_inner)]
        broken_specs: list[tuple[int, int, str]] = [(shared_border, shared_inner, _random_side())]

        for _ in range(n_uniform - 1):
            uniform_specs.append(_random_pair(palette))
        for _ in range(n_broken - 1):
            border_col, inner_col = _random_pair(palette)
            broken_specs.append((border_col, inner_col, _random_side()))

        uniform_border: list[tuple[int, int, np.ndarray]] = []
        broken_border: list[tuple[int, int, np.ndarray]] = []

        for border_col, inner_col in uniform_specs:
            patch = _make_uniform_border_patch(border_col, inner_col)
            r0, c0 = random_free_location(
                g,
                patch,
                transparent=bg,
                border=1,
                padding=1,
                padding_connectivity=4,
            )
            paste(g, patch, r0, c0)
            uniform_border.append((int(r0), int(c0), patch))

        for border_col, inner_col, side in broken_specs:
            patch = _make_broken_border_patch(border_col, inner_col, side)
            r0, c0 = random_free_location(
                g,
                patch,
                transparent=bg,
                border=1,
                padding=1,
                padding_connectivity=4,
            )
            paste(g, patch, r0, c0)
            broken_border.append((int(r0), int(c0), patch))

        expected = _build_output(
            bg,
            sorted(uniform_border, key=lambda t: (int(t[0]), int(t[1]))),
            sorted(broken_border, key=lambda t: (int(t[0]), int(t[1]))),
        )
        actual = main(g)
        metrics = _sample_metrics(g)

        if actual.shape != expected.shape or not np.array_equal(actual, expected):
            return None
        if metrics["legacy_diff"] != 1:
            return None
        if metrics["mixed_with_uniform_center"] != n_broken:
            return None
        if metrics["shared_color_overlap"] < 2:
            return None
        if metrics["all_two_color_objects"] != n_uniform + n_broken:
            return None
        return g
    except Exception:
        return None


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 400 attempts")
