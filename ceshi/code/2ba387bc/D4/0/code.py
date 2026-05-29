# Variant of 2ba387bc
# Dimension: D4 — 源对象的类型标识物 (src)
# Change: 框架(frame)通过1像素彩色边框+背景色内部识别；实心(solid)通过全部同色填充识别
#   → 带颜色分组的框架：框架内部填充非背景色，分类依据改为“内部颜色是否与边框颜色不同”。
#
# Design analysis
# a) 具体化与行为追踪
#    - 输入仍由若干分散的 4x4 对象组成，输出仍把 frame 堆到左列、solid 堆到右列。
#    - 变体后的 frame 具体长相：
#        b b b b
#        b i i b
#        b i i b
#        b b b b
#      其中边框色 b 与内部色 i 都非背景色，且 b != i。
#      solid 仍是 4x4 全同色 s。
#    - 小例子：
#      原逻辑里，若 sub[1:3,1:3] 全是 bg，则视为 frame；否则一律视为 solid。
#      变体里，一个红框蓝芯对象的中心不是 bg，原逻辑会误判成 solid；
#      新逻辑应读取“边框单色、中心单色、且两者不同”，据此判成 frame。
# b) 复杂度校准
#    - 不改对象尺寸、排序规则、输出布局，复杂度只集中在“类型识别从背景洞改为颜色关系”。
#    - 生成器会复用部分颜色：至少一个 solid 的颜色会出现在某个 frame 的边框或内芯中，
#      避免“某些颜色天然只属于某一类对象”的过度简化。
# c) 隐式假设审查（main 相关）
#    - find_components(..., monochromatic=False) 假设每个对象是 4 邻接的单个连通块。
#      变体 frame 因为内芯非背景，面积从 12 变成 16，但仍是单个 4x4 连通块，这个假设成立。
#    - 原来的 `inner == bg` 直接依赖旧属性，在变体中已失效，必须替换为 border/inner 颜色关系判断。
#    - “其余都是 solid” 仍可保留，因为生成器只产生两类对象：双色 frame 与单色 solid。
# d) 放置合法性
#    - 所有对象都以完整 4x4 patch 放置；`random_free_location(..., padding=1, connectivity=4)`
#      保证对象之间不接触、不重叠、不越界，solver 能稳定得到每个 4x4 bbox。
# e) 视觉可观察性
#    - 人类可直接看到：左列输出的是“外圈一种颜色、中心另一种颜色”的双色块，
#      右列输出的是纯色块；规则差异不依赖数值大小或不可见元数据。
# f) 可视化整洁度
#    - 仍采用单层散落对象；不叠加额外标记，不引入旋转/镜像，视觉负担低。
# g) 颜色变化可推断性
#    - 颜色种类限制在 ARC 常见 1..9；frame 的区分线索来自结构位置（边框 vs 中心），
#      而不是任意颜色映射。少量颜色复用反而强化“看关系，不看绝对颜色”的可推断性。
# 参考任务说明
#    - 题面引用了 b5ca7ac4 的 color_grouping 思路。当前 workspace 中不存在它的 solver 文件，
#      只有 `/home/lixinyu/snap/new/ARC-data/ARC-AGI-2/data/evaluation/b5ca7ac4.json` 可读，
#      因此这里只借用“通过内部颜色关系形成可见分组线索”的复杂度基线，不引入其他机制。
# 9) main 的新隐式假设与 generator 对应保证
#    - 假设每个有效对象 bbox 恰为 4x4：generator 只放置 4x4 patch。
#    - 假设 frame 的 12 个边框格同色、中心 2x2 同色：generator 的 `_make_frame` 保证。
#    - 假设 solid 是全同色：generator 的 `_make_solid` 保证。
#    - 假设不存在第三类 4x4 图案：generator 不生成其他模式。

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SOLVER_UTILS = Path("/home/lixinyu/snap/new/solvers")
if str(SOLVER_UTILS) not in sys.path:
    sys.path.append(str(SOLVER_UTILS))

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, paste, random_free_location


def _is_frame_patch(sub: np.ndarray) -> bool:
    border = np.concatenate([sub[0, :], sub[-1, :], sub[1:-1, 0], sub[1:-1, -1]])
    inner = sub[1:3, 1:3]
    border_colors = set(map(int, border.flatten()))
    inner_colors = set(map(int, inner.flatten()))
    return len(border_colors) == 1 and len(inner_colors) == 1 and next(iter(border_colors)) != next(iter(inner_colors))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort scattered 4x4 squares into two columns by type: colored-core frames left, solids right.

    Frame identification is no longer based on background holes. A frame is any 4x4 patch whose
    border is uniformly one color and whose 2x2 center is uniformly a different color.
    """
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
        if _is_frame_patch(sub):
            frames.append((int(r0), int(c0), sub))
        else:
            solids.append((int(r0), int(c0), sub))

    frames.sort(key=lambda t: (int(t[0]), int(t[1])))
    solids.sort(key=lambda t: (int(t[0]), int(t[1])))

    out_h = 4 * max(int(len(frames)), int(len(solids)))
    out = np.full((out_h, 8), int(bg), dtype=int)

    for i, (_r0, _c0, sub) in enumerate(frames):
        paste(out, sub, int(i * 4), 0)
    for i, (_r0, _c0, sub) in enumerate(solids):
        paste(out, sub, int(i * 4), 4)

    return out


def _make_frame(border_col: int, inner_col: int) -> np.ndarray:
    a = np.full((4, 4), int(border_col), dtype=int)
    a[1:3, 1:3] = int(inner_col)
    return a


def _make_solid(col: int) -> np.ndarray:
    return np.full((4, 4), int(col), dtype=int)


def _stack_expected(
    frames: list[tuple[int, int, np.ndarray]],
    solids: list[tuple[int, int, np.ndarray]],
    bg: int,
) -> np.ndarray:
    frames_sorted = sorted(frames, key=lambda t: (int(t[0]), int(t[1])))
    solids_sorted = sorted(solids, key=lambda t: (int(t[0]), int(t[1])))
    out_h = 4 * max(int(len(frames_sorted)), int(len(solids_sorted)))
    out = np.full((out_h, 8), int(bg), dtype=int)
    for i, (_r0, _c0, sub) in enumerate(frames_sorted):
        paste(out, sub, int(i * 4), 0)
    for i, (_r0, _c0, sub) in enumerate(solids_sorted):
        paste(out, sub, int(i * 4), 4)
    return out


def _rand_color(exclude: set[int]) -> int:
    choices = [c for c in range(1, 10) if c not in exclude]
    if not choices:
        choices = list(range(1, 10))
    return int(np.random.choice(choices))


def _gen_attempt() -> np.ndarray | None:
    bg = 0
    h = int(np.random.randint(18, 30))
    w = int(np.random.randint(18, 30))
    g = np.full((h, w), bg, dtype=int)

    n_frames = int(np.random.randint(1, 5))
    n_solids = int(np.random.randint(1, 5))

    try:
        frame_specs: list[tuple[int, int]] = []
        frame_role_colors: list[int] = []
        for _ in range(n_frames):
            border_col = _rand_color(set())
            inner_col = _rand_color({border_col})
            frame_specs.append((border_col, inner_col))
            frame_role_colors.extend([border_col, inner_col])

        solid_specs: list[int] = []
        linked_color = int(np.random.choice(frame_role_colors))
        solid_specs.append(linked_color)
        for _ in range(n_solids - 1):
            solid_specs.append(_rand_color(set()))

        frames: list[tuple[int, int, np.ndarray]] = []
        solids: list[tuple[int, int, np.ndarray]] = []

        for border_col, inner_col in frame_specs:
            patch = _make_frame(border_col, inner_col)
            r0, c0 = random_free_location(
                g,
                patch,
                transparent=bg,
                border=1,
                padding=1,
                padding_connectivity=4,
            )
            paste(g, patch, r0, c0)
            frames.append((int(r0), int(c0), patch))

        for col in solid_specs:
            patch = _make_solid(col)
            r0, c0 = random_free_location(
                g,
                patch,
                transparent=bg,
                border=1,
                padding=1,
                padding_connectivity=4,
            )
            paste(g, patch, r0, c0)
            solids.append((int(r0), int(c0), patch))

        expected = _stack_expected(frames, solids, bg)
        actual = main(g)
        if actual.shape != expected.shape or not np.array_equal(actual, expected):
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
