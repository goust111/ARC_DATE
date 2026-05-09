# Combo variant of fc7cae8d (index 27)
# combo_spec: D4/1+D9/0
# Name: 颜色互换+颜色图例

from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/arc2_solver_select")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import crop_bbox, find_components, get_bg, paste, random_free_location


"""
设计分析
========
a) 具体化与行为追踪
- 保留原题主体: 外框上有一条水平控制边和一条垂直控制边, 内部有一个被 D4 打乱朝向的带引导边矩形。
- 融合后额外在某个角落放一个小 legend, 形式固定为 2x2:
    A B
    B A
  它表示颜色 A 与颜色 B 在最终输出中互换。
- 原题流程:
  1. 读边框控制边颜色。
  2. 枚举旋转/镜像, 让主体两条引导边与对应控制边对齐。
  3. 裁掉 1px 引导边, 输出核心 core。
- 融合后流程:
  1. 同样先做 D4 对齐与裁剪得到 core。
  2. 再去角落找到 legend, 解码出要互换的颜色对 (A,B)。
  3. 仅在 core 中交换 A 与 B, 其余颜色不变。
- 小例子:
  对齐裁剪后的 core =
    1 2 4 1
    3 4 2 1
    1 3 4 2
  角落 legend =
    2 4
    4 2
  原题输出仍是上面的 core。
  融合后输出变为
    1 4 2 1
    3 2 4 1
    1 3 2 4
  第三个图案色 3 保持不动, 所以这不是普通的“所有非填充色互换”, 而是“查 legend 后定点交换”。

b) 复杂度校准
- 若核心里只有两种图案色, 融合版会退化成单独 D4/1 的“直接交换两色”。因此生成器强制核心恰好有 3 种图案色:
  两种被 legend 指定并交换, 第三种保留不变。
- 为了让交换肉眼可见, 两个被交换颜色都要求至少出现 4 次, 保留色至少出现 3 次。
- 为了让 D4 对齐本身仍然必要, 生成器筛掉具有非平凡 D4 对称的核心, 避免“怎么看都一样”的主体。
- legend 只用一个小 2x2 组件, 既明显不同于原题, 又不会引入过多自由度或视觉噪声。

c) 隐式假设审查
- `_detect_controls_frame()` 依赖: 边框仍然只有两条近乎纯色的控制边。融合没有改这点, generator 保持原规则。
- `_extract_center_object()` 依赖: 最大的内部连通块就是主体。新增 legend 只有 4 格, 且固定在角落小区域, 所以该前提仍成立。
- `_guide_edges()` 依赖: 主体填充色是主体内多数色, 引导边近乎整条纯色。generator 控制总非填充比例不超过约 36%, 因此前提仍成立。
- `_align_and_crop()` 依赖: 存在唯一能同时匹配两条控制边的 D4 变换。generator 使用互异控制色并拒绝 D4 对称核心来支撑该假设。
- `_extract_swap_pair()` 新增依赖: 存在且仅存在一个小型 2x2 角落 legend, 其图案是对角重复的
    A B
    B A
  generator 固定 legend 尺寸、形状、离边距离和角落区域来保证这一点。
- 最关键的新冲突是: 单独 D4/1 会假设“核心里只有两种图案色可直接交换”。融合版故意打破这个假设,
  所以 solver 不能再从核心自身推出交换对, 必须读取 legend。

d) 放置合法性
- 先画边框, 再把 legend 固定贴到四角之一的内缩位置 `(2,2)/(2,n-4)/(n-4,2)/(n-4,n-4)`。
  legend 四周都保留背景缓冲, 不会与边框连成一个组件。
- 主体随后使用 `random_free_location(..., border=1, padding=1)` 放置, 因此不会越界, 不会和 legend 重叠,
  也不会因为贴得太近而误连通。

e) 视觉可观察性
- 人类从样例能同时看到三件事: 边框控制边、被打乱朝向的主体、以及角落小 legend。
- 输出相较原题的显著变化是: 只有 legend 中那两种颜色交换, 第三种图案色不动。
- 这迫使解题者先完成原题对齐, 再理解 legend 的语义, 两阶段都可被样例直接观察。

f) 可视化整洁度
- legend 只有一个紧凑 2x2 小块, 且位于角落的独立留白区, 不会与主体叠加。
- 主体仍是单个大矩形, 视觉主次清晰。
- 外框噪声仍只出现在非控制边上, 不会跑到 legend 区或主体上。

g) 颜色变化可推断性
- 每个样例只出现 1 个交换规则, 由 2x2 legend 明确给出, 颜色集合被限制在较小 ARC 调色板内。
- 核心恰好有 3 种图案色: 两种交换, 一种保留。这样输入/输出对比会稳定呈现“只有 legend 那两色互换”。
- 量化退化检查中, 我们要求:
  1. 相对原题裁剪结果必须真正改变若干格。
  2. 相对单独 D4/1 的“只有两图案色时才交换”基线必须不同。
  3. 核心确实有 3 种图案色, 且第三种颜色在输出中保持原位。

主函数新增的逐段假设
--------------------
- `for comp in find_components(...)`: 假设 legend 与主体都不触边, 且主体面积远大于 legend。
- `for k in range(4)` / `for flip in (False, True)`: 假设主体的正确朝向可以仅由两条引导边颜色唯一确定。
- `if ctrl_key` 这种分支在本题未新增, 因为 legend 不依赖控制色查表; 新分支变成 `_extract_swap_pair()`。
- `_swap_selected_colors()` 假设 legend 指定的两色确实出现在裁剪核心中; generator 强制这两色都重复出现。
"""


def _transform(arr: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.rot90(arr, int(k))
    return np.fliplr(out) if flip else out


def _inverse_transform(arr: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.fliplr(arr) if flip else arr
    return np.rot90(out, -int(k))


def _counts_by_color(arr: np.ndarray) -> dict[int, int]:
    return {int(color): int(count) for color, count in Counter(map(int, arr.flatten())).items()}


def _d4_variants(arr: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    for k in range(4):
        base = np.rot90(arr, int(k))
        variants.append(base)
        variants.append(np.fliplr(base))
    return variants


def _is_d4_distinct(arr: np.ndarray) -> bool:
    seen: list[np.ndarray] = []
    for variant in _d4_variants(arr):
        if any(np.array_equal(variant, prev) for prev in seen):
            return False
        seen.append(variant)
    return True


def _detect_controls_frame(arr: np.ndarray, bg: int) -> list[tuple[str, int]]:
    h, w = arr.shape
    sides = {
        "top": arr[0, 1 : w - 1],
        "bottom": arr[h - 1, 1 : w - 1],
        "left": arr[1 : h - 1, 0],
        "right": arr[1 : h - 1, w - 1],
    }
    cand: list[tuple[str, int, int, int]] = []
    for side, vals in sides.items():
        values = list(map(int, vals))
        non_bg = [x for x in values if x != int(bg)]
        if not non_bg:
            continue
        col, count = Counter(non_bg).most_common(1)[0]
        if int(count) >= len(values) - 1:
            cand.append((side, int(col), int(count), int(len(values))))
    cand.sort(key=lambda t: (t[2] / t[3], t[2]), reverse=True)
    return [(side, col) for side, col, _count, _length in cand[:2]]


def _extract_center_object(arr: np.ndarray, bg: int) -> np.ndarray:
    h, w = arr.shape
    comps = find_components(arr, bg=int(bg), connectivity=4, monochromatic=False, return_format="dict")
    best: tuple[int, tuple[int, int, int, int]] | None = None
    for comp in comps:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        if r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1:
            continue
        area = int(comp["area"])
        if best is None or area > best[0]:
            best = (area, (r0, c0, r1, c1))
    if best is None:
        return arr.copy()
    return crop_bbox(arr, best[1])


def _guide_edges(arr: np.ndarray) -> dict[str, int]:
    fill = int(Counter(map(int, arr.flatten())).most_common(1)[0][0])
    edges = {
        "top": arr[0, :],
        "bottom": arr[-1, :],
        "left": arr[:, 0],
        "right": arr[:, -1],
    }
    out: dict[str, int] = {}
    for side, vals in edges.items():
        values = list(map(int, vals))
        non_fill = [x for x in values if x != int(fill)]
        if not non_fill:
            continue
        col, count = Counter(non_fill).most_common(1)[0]
        if int(count) >= len(values) - 1:
            out[side] = int(col)
    return out


def _align_and_crop(obj: np.ndarray, desired: dict[str, int]) -> np.ndarray:
    for k in range(4):
        base = np.rot90(obj, int(k))
        for flip in (False, True):
            t = np.fliplr(base) if flip else base
            gedges = _guide_edges(t)
            if all(side in gedges and int(gedges[side]) == int(col) for side, col in desired.items()):
                top = 1 if "top" in desired else 0
                bottom = int(t.shape[0]) - (1 if "bottom" in desired else 0)
                left = 1 if "left" in desired else 0
                right = int(t.shape[1]) - (1 if "right" in desired else 0)
                return t[int(top) : int(bottom), int(left) : int(right)]
    return obj.copy()


def _corner_anchors(shape: tuple[int, int]) -> dict[str, tuple[int, int]]:
    h, w = map(int, shape)
    return {
        "tl": (2, 2),
        "tr": (2, w - 4),
        "bl": (h - 4, 2),
        "br": (h - 4, w - 4),
    }


def _extract_swap_pair(arr: np.ndarray, bg: int) -> tuple[int, int] | None:
    anchors = _corner_anchors(arr.shape)
    comps = find_components(arr, bg=int(bg), connectivity=4, monochromatic=False, return_format="dict")
    best: tuple[int, tuple[int, int]] | None = None
    for comp in comps:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        if r0 == 0 or c0 == 0 or r1 == arr.shape[0] - 1 or c1 == arr.shape[1] - 1:
            continue
        if int(comp["area"]) != 4:
            continue
        patch = crop_bbox(arr, (r0, c0, r1, c1))
        if patch.shape != (2, 2):
            continue
        if int(np.any(patch == int(bg))):
            continue
        if not (int(patch[0, 0]) == int(patch[1, 1]) and int(patch[0, 1]) == int(patch[1, 0])):
            continue
        colors = sorted({int(x) for x in patch.flatten()})
        if len(colors) != 2:
            continue
        dist = min(abs(r0 - ar) + abs(c0 - ac) for ar, ac in anchors.values())
        candidate = (dist, (colors[0], colors[1]))
        if best is None or candidate[0] < best[0]:
            best = candidate
    return None if best is None else best[1]


def _swap_selected_colors(arr: np.ndarray, a: int, b: int) -> np.ndarray:
    out = np.array(arr, copy=True)
    mask_a = arr == int(a)
    mask_b = arr == int(b)
    out[mask_a] = int(b)
    out[mask_b] = int(a)
    return out


def _swap_all_if_two_pattern_colors(arr: np.ndarray) -> np.ndarray:
    counts = Counter(map(int, arr.flatten()))
    fill = int(counts.most_common(1)[0][0])
    pattern_colors = sorted(color for color in counts if int(color) != fill)
    if len(pattern_colors) != 2:
        return arr.copy()
    return _swap_selected_colors(arr, int(pattern_colors[0]), int(pattern_colors[1]))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Phase 1: align the guided rectangle to the frame-edge controls and crop the guides.
    Phase 2: read the corner legend and swap exactly that color pair inside the cropped core.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    ctrls = _detect_controls_frame(g, bg)
    desired = {side: col for side, col in ctrls}
    obj = _extract_center_object(g, bg)
    core = _align_and_crop(obj, desired)

    pair = _extract_swap_pair(g, bg)
    if pair is None:
        return core
    a, b = pair
    return _swap_selected_colors(core, int(a), int(b))


def _build_core(h: int, w: int, fill: int, pattern_colors: tuple[int, int, int]) -> np.ndarray | None:
    area = h * w
    min_counts = {
        int(pattern_colors[0]): max(4, area // 22),
        int(pattern_colors[1]): max(4, area // 22),
        int(pattern_colors[2]): max(3, area // 26),
    }
    max_nonfill = max(sum(min_counts.values()) + 3, int(area * 0.36))

    for _ in range(140):
        core = np.full((h, w), int(fill), dtype=int)
        extra = random.randint(0, max_nonfill - sum(min_counts.values()))
        targets = {color: count for color, count in min_counts.items()}

        while extra > 0:
            color = int(random.choice(pattern_colors))
            targets[color] += 1
            extra -= 1

        for color in pattern_colors:
            painted = 0
            attempts = 0
            target = int(targets[int(color)])
            while painted < target and attempts < area * 12:
                attempts += 1
                shape_kind = random.choice(["dot", "line", "rect", "el"])
                if shape_kind == "dot":
                    cells = [(random.randrange(h), random.randrange(w))]
                elif shape_kind == "line":
                    line_len = random.randint(2, min(6, max(h, w)))
                    if random.random() < 0.5:
                        rr = random.randrange(h)
                        c0 = random.randrange(w - line_len + 1)
                        cells = [(rr, c0 + dc) for dc in range(line_len)]
                    else:
                        r0 = random.randrange(h - line_len + 1)
                        cc = random.randrange(w)
                        cells = [(r0 + dr, cc) for dr in range(line_len)]
                elif shape_kind == "rect":
                    rh = random.randint(1, min(4, h))
                    rw = random.randint(1, min(4, w))
                    r0 = random.randrange(h - rh + 1)
                    c0 = random.randrange(w - rw + 1)
                    cells = [(r0 + dr, c0 + dc) for dr in range(rh) for dc in range(rw)]
                else:
                    rh = random.randint(2, min(4, h))
                    rw = random.randint(2, min(4, w))
                    r0 = random.randrange(h - rh + 1)
                    c0 = random.randrange(w - rw + 1)
                    cells = [(r0 + dr, c0) for dr in range(rh)] + [(r0 + rh - 1, c0 + dc) for dc in range(rw)]

                random.shuffle(cells)
                for r, c in cells:
                    if core[r, c] != int(fill):
                        continue
                    core[r, c] = int(color)
                    painted += 1
                    if painted >= target:
                        break
            if painted < target:
                break
        else:
            # Add a corner-biased marker to further discourage accidental D4 symmetry.
            marker_color = int(random.choice(pattern_colors))
            marker_corner = random.choice(
                [
                    (0, 0),
                    (0, max(0, w - 2)),
                    (max(0, h - 2), 0),
                    (max(0, h - 2), max(0, w - 2)),
                ]
            )
            mr, mc = marker_corner
            marker_cells = [(mr, mc), (min(h - 1, mr + 1), mc), (min(h - 1, mr + 1), min(w - 1, mc + 1))]
            for r, c in marker_cells:
                core[r, c] = int(marker_color)

            counts = _counts_by_color(core)
            if counts.get(int(fill), 0) <= max(counts.get(int(c), 0) for c in pattern_colors):
                continue
            if any(counts.get(int(c), 0) < min_counts[int(c)] for c in pattern_colors):
                continue
            if not _is_d4_distinct(core):
                continue
            return core

    return None


def _attach_guides(core: np.ndarray, guide_sides: tuple[str, str], guide_colors: dict[str, int]) -> np.ndarray:
    h, w = core.shape
    has_top = "top" in guide_sides
    has_bottom = "bottom" in guide_sides
    has_left = "left" in guide_sides
    has_right = "right" in guide_sides
    out = np.zeros((h + has_top + has_bottom, w + has_left + has_right), dtype=int)
    r0 = 1 if has_top else 0
    c0 = 1 if has_left else 0
    paste(out, core, r0, c0)

    if has_top:
        out[0, c0 : c0 + w] = int(guide_colors["top"])
    if has_bottom:
        out[-1, c0 : c0 + w] = int(guide_colors["bottom"])
    if has_left:
        out[r0 : r0 + h, 0] = int(guide_colors["left"])
    if has_right:
        out[r0 : r0 + h, -1] = int(guide_colors["right"])

    if has_top and has_left:
        out[0, 0] = 0
    if has_top and has_right:
        out[0, -1] = 0
    if has_bottom and has_left:
        out[-1, 0] = 0
    if has_bottom and has_right:
        out[-1, -1] = 0
    return out


def _paint_frame(grid: np.ndarray, horiz_side: str, vert_side: str, ctrl_h: int, ctrl_v: int) -> None:
    n = grid.shape[0]
    if horiz_side == "top":
        grid[0, :] = int(ctrl_h)
    else:
        grid[-1, :] = int(ctrl_h)
    if vert_side == "left":
        grid[:, 0] = int(ctrl_v)
    else:
        grid[:, -1] = int(ctrl_v)

    for side in ("top", "bottom", "left", "right"):
        if side in (horiz_side, vert_side):
            continue
        if side in ("top", "bottom"):
            r = 0 if side == "top" else n - 1
            coords = [(r, c) for c in range(1, n - 1)]
        else:
            c = 0 if side == "left" else n - 1
            coords = [(r, c) for r in range(1, n - 1)]
        min_count = int(np.ceil(0.35 * len(coords)))
        max_count = int(np.floor(0.60 * len(coords)))
        noise_count = random.randint(min_count, max_count)
        for r, c in random.sample(coords, noise_count):
            grid[r, c] = 5


def _legend_patch(a: int, b: int) -> np.ndarray:
    return np.array([[int(a), int(b)], [int(b), int(a)]], dtype=int)


def _paste_corner_legend(grid: np.ndarray, a: int, b: int, corner: str) -> None:
    patch = _legend_patch(a, b)
    anchors = _corner_anchors(grid.shape)
    r0, c0 = anchors[corner]
    paste(grid, patch, int(r0), int(c0), transparent=0)


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(24, 30)
    horiz_side, vert_side = random.choice(
        [("top", "left"), ("top", "right"), ("bottom", "left"), ("bottom", "right")]
    )

    semantic_colors = [1, 2, 3, 4, 6, 7, 8, 9]
    ctrl_h, ctrl_v = random.sample(semantic_colors, 2)
    remaining = [c for c in semantic_colors if c not in {ctrl_h, ctrl_v}]
    fill = random.choice(remaining)
    remaining = [c for c in remaining if c != fill]
    pattern_colors = tuple(random.sample(remaining, 3))
    swap_pair = tuple(random.sample(pattern_colors, 2))
    keep_color = next(color for color in pattern_colors if color not in swap_pair)

    max_core_h = min(17, n - 8)
    max_core_w = min(17, n - 8)
    if max_core_h < 7 or max_core_w < 7:
        return None

    core_h = random.randint(7, max_core_h)
    core_w = random.randint(7, max_core_w)
    core = _build_core(core_h, core_w, fill, pattern_colors)
    if core is None:
        return None

    counts = _counts_by_color(core)
    if counts.get(int(swap_pair[0]), 0) < 4 or counts.get(int(swap_pair[1]), 0) < 4:
        return None
    if counts.get(int(keep_color), 0) < 3:
        return None

    expected = _swap_selected_colors(core, int(swap_pair[0]), int(swap_pair[1]))
    if np.array_equal(expected, core):
        return None
    if int(np.sum(expected == int(keep_color))) != int(np.sum(core == int(keep_color))):
        return None

    guide_colors = {horiz_side: ctrl_h, vert_side: ctrl_v}
    aligned_object = _attach_guides(core, (horiz_side, vert_side), guide_colors)
    k = random.randrange(4)
    flip = bool(random.randrange(2))
    input_object = _inverse_transform(aligned_object, k, flip)

    grid = np.zeros((n, n), dtype=int)
    _paint_frame(grid, horiz_side, vert_side, ctrl_h, ctrl_v)
    legend_corner = random.choice(["tl", "tr", "bl", "br"])
    _paste_corner_legend(grid, int(swap_pair[0]), int(swap_pair[1]), legend_corner)

    try:
        r0, c0 = random_free_location(grid, input_object, transparent=0, border=1, padding=1)
        paste(grid, input_object, r0, c0, transparent=0)
    except ValueError:
        return None

    metrics = variant_metrics(grid)
    if metrics["pattern_colors"] != 3:
        return None
    if metrics["changed_cells"] < 8:
        return None
    if metrics["same_as_original"]:
        return None
    if metrics["same_as_plain_d4"]:
        return None
    if metrics["kept_color_cells"] < 3:
        return None
    if metrics["legend_found"] != 1:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate input for fc7cae8d combo variant 27")


def variant_metrics(grid: np.ndarray) -> dict[str, int]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    ctrls = _detect_controls_frame(g, bg)
    desired = {side: col for side, col in ctrls}
    obj = _extract_center_object(g, bg)
    phase1 = _align_and_crop(obj, desired)
    pair = _extract_swap_pair(g, bg)
    out = phase1.copy() if pair is None else _swap_selected_colors(phase1, int(pair[0]), int(pair[1]))

    counts = Counter(map(int, phase1.flatten()))
    fill = int(counts.most_common(1)[0][0])
    pattern_colors = sorted(color for color in counts if int(color) != fill)
    kept_color_cells = 0
    if pair is not None:
        remaining = [color for color in pattern_colors if color not in set(map(int, pair))]
        if remaining:
            keep_color = int(remaining[0])
            kept_color_cells = int(np.sum(phase1 == keep_color))
            if not np.array_equal(phase1 == keep_color, out == keep_color):
                kept_color_cells = -kept_color_cells

    plain_d4 = _swap_all_if_two_pattern_colors(phase1)
    return {
        "legend_found": int(pair is not None),
        "pattern_colors": int(len(pattern_colors)),
        "changed_cells": int(np.sum(phase1 != out)),
        "same_as_original": int(np.array_equal(phase1, out)),
        "same_as_plain_d4": int(np.array_equal(plain_d4, out)),
        "kept_color_cells": int(max(kept_color_cells, 0)),
    }
