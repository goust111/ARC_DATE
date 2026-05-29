# Combo variant of 2ba387bc (index 2)
# combo_spec: D1/0+D4/0
# Name: 边框一致性分类+彩色内部框架
#
# Design analysis
# a) 具体化与行为追踪
#    - 原题对象只有两类：单色边框+背景内芯的 frame，和整块单色的 solid。
#      原 solver 用 `inner == bg` 把二者分开。
#    - 融合后，输入中的每个对象都保持 4x4、都带 2x2 彩色内芯；区别只在外边框：
#        A 类（一致边框）:
#          b b b b
#          b i i b
#          b i i b
#          b b b b
#          其中 b != i，且中心不再是背景色。
#        B 类（多色边框）:
#          t t t t
#          s i i s
#          s i i s
#          t t t t
#          其中 t != s，且 i 与边框色都不同。
#    - 小例子：
#      一个 A 类对象若是“红框蓝芯”，原逻辑会因为中心不是背景，把它误判到右列；
#      新逻辑只看边框颜色集合大小，发现外圈全红，所以放左列。
#      一个 B 类对象若是“上下一色、左右另一色、中心第三色”，新逻辑看到外圈含两色，
#      因而放右列；中心颜色在两类中都可能出现，所以它只能当干扰项，不能当判据。
# b) 复杂度校准
#    - 没有增加对象尺寸、旋转或额外标记，复杂度集中在“观察边框颜色关系而忽略中心”。
#    - B 类边框固定为两色框，而不是任意噪声边框；这样视觉上明显区别于 A 类，同时避免
#      过多自由度导致样本难归纳。
#    - generator 会复用部分颜色，让某些中心色也出现在别的对象边框中，防止颜色角色被硬编码。
# c) 隐式假设审查（main 逐段）
#    - `find_components(..., monochromatic=False)` 假设每个对象仍是单个 4 邻接连通块。
#      现在 A/B 两类对象都完全由非背景像素构成，且 4x4 连通，这个前提成立。
#    - `bbox_h == bbox_w == 4` 假设有效对象都恰好是 4x4；generator 只放置 4x4 patch 保证此点。
#    - 原来的 `inner == bg` 依赖“frame 中心挖空”的旧属性，在融合后失效，必须删除。
#    - 新逻辑 `len(border_colors) == 1` 假设分类只由外边框是否单色决定；generator 恰好只生成
#      “单色边框对象”和“两色边框对象”，不存在第三类干扰模式。
# d) 放置合法性
#    - 所有对象仍通过 `random_free_location(..., padding=1, connectivity=4)` 放置，确保不重叠、
#      不接触、不越界，因此 solver 提取到的连通域与单个 4x4 对象一一对应。
# e) 视觉可观察性
#    - 人类能直接看到：左列对象外圈颜色完全一致，右列对象外圈上下与左右颜色不同。
#      两类对象的中心都着色，所以“看中心是不是背景”在样例里会系统性失败，融合特征可见。
# f) 可视化整洁度
#    - 输入仍是稀疏散落的大块对象，没有额外覆盖层；B 类也只使用规则化两色边框，画面整洁。
# g) 变化的可推断性
#    - 颜色仍限制在 ARC 常见的 1..9。颜色复用的结构线索来自“边框位置 vs 中心位置”，而不是
#      任意颜色映射，因此 2-4 个样例足以支持归纳。
# 8) 融合行为量化指标
#    - `colored_center_count`: 中心 2x2 非背景的对象数。这里应等于对象总数，证明输入没有退化回原题。
#    - `uniform_border_count` / `mixed_border_count`: 两类对象都必须为正，证明分类变量真正起作用。
#    - `old_rule_disagreement_count`: 按原题逻辑（中心是否背景）会把多少对象放错类。这里应至少等于
#      `uniform_border_count`，因为所有 A 类对象在旧逻辑下都会被误判。
# 9) main 的新隐式假设与 generator 对应保证
#    - 假设每个对象中心 2x2 都是单色非背景：`_make_uniform_border_frame` 与
#      `_make_mixed_border_frame` 保证。
#    - 假设 B 类边框至少含两色：generator 生成时强制 `top_bottom_col != side_col`。
#    - 假设 A/B 两类都出现，否则新旧规则在布局上可能不充分区分：generator 强制两类数量都 >= 1。

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SOLVER_UTILS = Path("/home/lixinyu/snap/new/solvers")
if str(SOLVER_UTILS) not in sys.path:
    sys.path.append(str(SOLVER_UTILS))

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, paste, random_free_location


def _border_pixels(sub: np.ndarray) -> np.ndarray:
    return np.concatenate([sub[0, :], sub[-1, :], sub[1:-1, 0], sub[1:-1, -1]])


def _is_uniform_border(sub: np.ndarray) -> bool:
    return len(set(map(int, _border_pixels(sub).flatten()))) == 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort scattered 4x4 colored-core squares into two columns by border consistency.

    Left column: outer border is a single color.
    Right column: outer border contains multiple colors.
    The 2x2 center is always colored and must be ignored as a classification cue.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    uniform_border: list[tuple[int, int, np.ndarray]] = []
    mixed_border: list[tuple[int, int, np.ndarray]] = []

    for comp in comps:
        b = comp["bbox"]
        if bbox_h(b) != 4 or bbox_w(b) != 4:
            continue
        r0, c0, _r1, _c1 = b
        sub = crop_bbox(g, b)
        if _is_uniform_border(sub):
            uniform_border.append((int(r0), int(c0), sub))
        else:
            mixed_border.append((int(r0), int(c0), sub))

    uniform_border.sort(key=lambda t: (int(t[0]), int(t[1])))
    mixed_border.sort(key=lambda t: (int(t[0]), int(t[1])))

    out_h = 4 * max(int(len(uniform_border)), int(len(mixed_border)))
    out = np.full((out_h, 8), int(bg), dtype=int)

    for i, (_r0, _c0, sub) in enumerate(uniform_border):
        paste(out, sub, int(i * 4), 0)
    for i, (_r0, _c0, sub) in enumerate(mixed_border):
        paste(out, sub, int(i * 4), 4)

    return out


def _make_uniform_border_frame(border_col: int, inner_col: int) -> np.ndarray:
    patch = np.full((4, 4), int(border_col), dtype=int)
    patch[1:3, 1:3] = int(inner_col)
    return patch


def _make_mixed_border_frame(top_bottom_col: int, side_col: int, inner_col: int) -> np.ndarray:
    patch = np.full((4, 4), int(side_col), dtype=int)
    patch[0, :] = int(top_bottom_col)
    patch[-1, :] = int(top_bottom_col)
    patch[1:3, 1:3] = int(inner_col)
    return patch


def _stack_expected(
    uniform_border: list[tuple[int, int, np.ndarray]],
    mixed_border: list[tuple[int, int, np.ndarray]],
    bg: int,
) -> np.ndarray:
    left = sorted(uniform_border, key=lambda t: (int(t[0]), int(t[1])))
    right = sorted(mixed_border, key=lambda t: (int(t[0]), int(t[1])))
    out_h = 4 * max(int(len(left)), int(len(right)))
    out = np.full((out_h, 8), int(bg), dtype=int)
    for i, (_r0, _c0, sub) in enumerate(left):
        paste(out, sub, int(i * 4), 0)
    for i, (_r0, _c0, sub) in enumerate(right):
        paste(out, sub, int(i * 4), 4)
    return out


def _rand_color(exclude: set[int]) -> int:
    choices = [c for c in range(1, 10) if c not in exclude]
    if not choices:
        raise ValueError("no colors available")
    return int(np.random.choice(choices))


def _choose_inner_color(border_colors: set[int], shared_pool: list[int]) -> int:
    reusable = [c for c in shared_pool if c not in border_colors]
    if reusable and np.random.rand() < 0.8:
        return int(np.random.choice(reusable))
    return _rand_color(border_colors)


def _old_rule_would_call_frame(sub: np.ndarray, bg: int) -> bool:
    border_colors = set(map(int, _border_pixels(sub).flatten()))
    return len(border_colors) == 1 and bool((sub[1:3, 1:3] == int(bg)).all())


def _fusion_metrics(grid: np.ndarray) -> dict[str, int]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")

    colored_center_count = 0
    uniform_border_count = 0
    mixed_border_count = 0
    old_rule_disagreement_count = 0

    for comp in comps:
        b = comp["bbox"]
        if bbox_h(b) != 4 or bbox_w(b) != 4:
            continue
        sub = crop_bbox(g, b)
        if bool((sub[1:3, 1:3] != int(bg)).all()):
            colored_center_count += 1
        if _is_uniform_border(sub):
            uniform_border_count += 1
        else:
            mixed_border_count += 1
        old_is_left = _old_rule_would_call_frame(sub, bg)
        new_is_left = _is_uniform_border(sub)
        if old_is_left != new_is_left:
            old_rule_disagreement_count += 1

    return {
        "colored_center_count": int(colored_center_count),
        "uniform_border_count": int(uniform_border_count),
        "mixed_border_count": int(mixed_border_count),
        "old_rule_disagreement_count": int(old_rule_disagreement_count),
    }


def _gen_attempt() -> np.ndarray | None:
    bg = 0
    h = int(np.random.randint(18, 30))
    w = int(np.random.randint(18, 30))
    g = np.full((h, w), bg, dtype=int)

    n_uniform = int(np.random.randint(1, 5))
    n_mixed = int(np.random.randint(1, 5))

    try:
        uniform_borders = [_rand_color(set()) for _ in range(n_uniform)]
        mixed_border_specs = []
        for _ in range(n_mixed):
            tb_col = _rand_color(set())
            side_col = _rand_color({tb_col})
            mixed_border_specs.append((tb_col, side_col))

        shared_pool = list(uniform_borders)
        for tb_col, side_col in mixed_border_specs:
            shared_pool.extend([tb_col, side_col])

        uniform_specs: list[tuple[int, int]] = []
        mixed_specs: list[tuple[int, int, int]] = []

        for border_col in uniform_borders:
            inner_col = _choose_inner_color({border_col}, shared_pool)
            uniform_specs.append((border_col, inner_col))

        for tb_col, side_col in mixed_border_specs:
            inner_col = _choose_inner_color({tb_col, side_col}, shared_pool)
            mixed_specs.append((tb_col, side_col, inner_col))

        if uniform_specs and mixed_border_specs:
            tb_col, side_col = mixed_border_specs[0]
            forced_inner = tb_col if tb_col != uniform_specs[0][0] else side_col
            if forced_inner != uniform_specs[0][0]:
                uniform_specs[0] = (uniform_specs[0][0], forced_inner)

        if mixed_specs and uniform_borders:
            candidate = uniform_borders[0]
            tb_col, side_col, inner_col = mixed_specs[0]
            if candidate not in {tb_col, side_col}:
                mixed_specs[0] = (tb_col, side_col, candidate)

        uniform_objs: list[tuple[int, int, np.ndarray]] = []
        mixed_objs: list[tuple[int, int, np.ndarray]] = []

        for border_col, inner_col in uniform_specs:
            patch = _make_uniform_border_frame(border_col, inner_col)
            r0, c0 = random_free_location(
                g,
                patch,
                transparent=bg,
                border=1,
                padding=1,
                padding_connectivity=4,
            )
            paste(g, patch, r0, c0)
            uniform_objs.append((int(r0), int(c0), patch))

        for tb_col, side_col, inner_col in mixed_specs:
            patch = _make_mixed_border_frame(tb_col, side_col, inner_col)
            r0, c0 = random_free_location(
                g,
                patch,
                transparent=bg,
                border=1,
                padding=1,
                padding_connectivity=4,
            )
            paste(g, patch, r0, c0)
            mixed_objs.append((int(r0), int(c0), patch))

        expected = _stack_expected(uniform_objs, mixed_objs, bg)
        actual = main(g)
        metrics = _fusion_metrics(g)

        if actual.shape != expected.shape or not np.array_equal(actual, expected):
            return None
        if metrics["colored_center_count"] != n_uniform + n_mixed:
            return None
        if metrics["uniform_border_count"] != n_uniform or metrics["mixed_border_count"] != n_mixed:
            return None
        if metrics["old_rule_disagreement_count"] < 1:
            return None
        return g
    except Exception:
        return None


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 300 attempts")
