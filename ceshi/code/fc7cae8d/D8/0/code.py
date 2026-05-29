# Variant of fc7cae8d
# Dimension: D8 — 多步序列处理 (meta)
# Change: 单步——一次对齐+裁剪完成全部逻辑... → 两步变换：先对齐后图案替换

from __future__ import annotations

from collections import Counter
import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/arc2_solver_select")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import crop_bbox, find_components, get_bg, paste, random_free_location

"""
设计分析
========
参考任务说明:
- 用户要求先读 source_task=d35bdbdc 的 solver 代码再设计。
- 当前仓库里找不到 d35bdbdc 的 solver 文件, 只有 evaluation JSON 与机制摘要:
  "Step 1: identify ... ; Step 2: ... use color/property match to transfer attributes."
- 因此这里把该摘要当作多步基线: 第一阶段先定位/整理主对象, 第二阶段再做基于颜色键的属性替换。

a) 具体化与行为追踪
- 输入仍保留原题的主体: 边框上两条控制边 + 内部带两条引导边的矩形对象。
- 新增一个 legend 区, 由若干条 1x4 彩色条带组成, 每条含义为:
  [水平控制色, 垂直控制色, 待替换图案色, 目标颜色]。
- 第一步与原题完全一致: 先根据引导边颜色和边框控制边颜色做 D8 对齐, 再裁掉引导边。
- 第二步才使用 legend: 读取当前输入边框的 (水平控制色, 垂直控制色), 在 legend 中找到唯一匹配条带,
  将第一步得到的核心图中所有 src 色替换成 dst 色。
- 小例子:
  原题: 控制边颜色=(2,3), 对齐裁剪后得到 core=[[4,7],[7,1]], 输出仍是 [[4,7],[7,1]]。
  变体: 输入里额外有 legend 条带 [2,3,7,8], [6,9,1,4], ...。
  第一步后仍得到 [[4,7],[7,1]], 第二步查到 (2,3)->(7->8), 输出变成 [[4,8],[8,1]]。
  差异就在于 "裁剪结果并不是最终答案", 解题者必须继续执行颜色替换。

b) 复杂度校准
- 只放一条 legend 会退化成直接附带答案, 因此生成器放 3-5 条条带, 只有一条 key 匹配当前控制色对。
- 如果被替换颜色只出现 1 次, 视觉上不明显, 因此生成器强制 src 在核心里至少出现 3 次。
- 如果核心图对某些旋转/翻转不变, 第一步的 D8 变化肉眼不明显, 因此生成器筛掉对非恒等 D8 变换不变的核心。

c) 隐式假设审查
- 原 main() 中 `return cropped_core` 依赖 "裁剪即答案"; 变体下这个前提失效, 必须增加第二阶段替换。
- 原 `extract_center_object()` 依赖 "内部最大连通块就是主体"; 变体新增 legend 后仍保留该假设,
  但生成器用很小的 1x4 条带并与主体留空, 保证主体仍是最大的内部连通块。
- 原 `guide_edges()` 依赖 "主体填充色占多数, 引导边近乎整条纯色"; 生成器继续保证 fill 为主体多数色,
  且 guide 边是完整单色边, 所以前提仍成立。
- 新增 `legend` 解析依赖 "legend 条带是独立的 1x4 连通块, 且控制色对唯一匹配";
  生成器显式保证条带尺寸固定、相互分离、key 唯一。

d) 放置合法性
- 主体对象与 legend 都通过 `random_free_location(..., padding=1, border=1)` 放置,
  保证不重叠、不越界、彼此不 4 邻接/8 邻接到误连成同一组件。
- 主体总是离边框至少 1 格, 因而不会触碰 frame, solver 仍可把 touching-border 的组件排除掉。

e) 视觉可观察性
- 从样例上看, 输入里能同时看到: 边框控制色、被打乱朝向的主体、以及独立 legend 条带。
- 输出相较原题会多一个明显变化: 某种图案色会统一改成另一种颜色, 且这组颜色由 legend 决定。
- 因为 legend 有多条而非单条, 且 key 用的是边框控制色对, 人类解题者能观察到这是 "先对齐裁剪, 再查表替色"。
"""


def _transform(arr: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.rot90(arr, int(k))
    return np.fliplr(out) if flip else out


def _inverse_transform(arr: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.fliplr(arr) if flip else arr
    return np.rot90(out, -int(k))


def _detect_controls_frame(arr: np.ndarray, bg: int) -> list[tuple[str, int]]:
    H, W = arr.shape
    sides = {
        "top": arr[0, 1 : W - 1],
        "bottom": arr[H - 1, 1 : W - 1],
        "left": arr[1 : H - 1, 0],
        "right": arr[1 : H - 1, W - 1],
    }
    cand: list[tuple[str, int, int, int]] = []
    for side, vals in sides.items():
        v = list(map(int, vals))
        v_nb = [x for x in v if x != int(bg)]
        if not v_nb:
            continue
        col, count = Counter(v_nb).most_common(1)[0]
        if int(count) >= len(v) - 1:
            cand.append((side, int(col), int(count), int(len(v))))
    cand.sort(key=lambda t: (t[2] / t[3], t[2]), reverse=True)
    return [(side, col) for side, col, _count, _len in cand[:2]]


def _extract_center_object(arr: np.ndarray, bg: int) -> np.ndarray:
    H, W = arr.shape
    comps = find_components(arr, bg=int(bg), connectivity=4, monochromatic=False, return_format="dict")
    best: tuple[int, tuple[int, int, int, int]] | None = None
    for comp in comps:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        if r0 == 0 or c0 == 0 or r1 == H - 1 or c1 == W - 1:
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
        v = list(map(int, vals))
        v_nf = [x for x in v if x != int(fill)]
        if not v_nf:
            continue
        col, count = Counter(v_nf).most_common(1)[0]
        if int(count) >= len(v) - 1:
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


def _extract_legend_rows(arr: np.ndarray, bg: int) -> list[tuple[int, int, int, int]]:
    H, W = arr.shape
    comps = find_components(arr, bg=int(bg), connectivity=4, monochromatic=False, return_format="dict")
    rows: list[tuple[int, int, int, int]] = []
    for comp in comps:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        if r0 == 0 or c0 == 0 or r1 == H - 1 or c1 == W - 1:
            continue
        if int(comp["area"]) != 4:
            continue
        patch = crop_bbox(arr, (r0, c0, r1, c1))
        if patch.shape != (1, 4):
            continue
        rows.append(tuple(map(int, patch[0])))
    return rows


def _legend_map(rows: list[tuple[int, int, int, int]]) -> dict[tuple[int, int], tuple[int, int]]:
    out: dict[tuple[int, int], tuple[int, int]] = {}
    for h_ctrl, v_ctrl, src, dst in rows:
        out[(int(h_ctrl), int(v_ctrl))] = (int(src), int(dst))
    return out


def _control_key(ctrls: list[tuple[str, int]]) -> tuple[int, int] | None:
    horiz = [int(col) for side, col in ctrls if side in ("top", "bottom")]
    vert = [int(col) for side, col in ctrls if side in ("left", "right")]
    if not horiz or not vert:
        return None
    return int(horiz[0]), int(vert[0])


def _replace_color(arr: np.ndarray, src: int, dst: int) -> np.ndarray:
    out = np.array(arr, copy=True)
    out[out == int(src)] = int(dst)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Phase 1: align the guided rectangle to the frame-edge controls and crop the guides.
    Phase 2: use the frame control colors as a key into the legend, then recolor one pattern color.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    ctrls = _detect_controls_frame(g, bg)
    desired = {side: col for side, col in ctrls}
    obj = _extract_center_object(g, bg)
    core = _align_and_crop(obj, desired)

    ctrl_key = _control_key(ctrls)
    if ctrl_key is None:
        return core

    legends = _extract_legend_rows(g, bg)
    mapping = _legend_map(legends)
    if ctrl_key not in mapping:
        return core

    src, dst = mapping[ctrl_key]
    return _replace_color(core, src, dst)


def _build_core(h: int, w: int, fill: int, pattern_colors: list[int], src_color: int) -> np.ndarray:
    area = h * w
    max_nonfill = max(len(pattern_colors) + 6, int(area * 0.35))
    min_nonfill = max(len(pattern_colors) + 4, area // 8)
    min_src = max(3, area // 14)

    for _ in range(120):
        core = np.full((h, w), int(fill), dtype=int)
        target_nonfill = random.randint(min_nonfill, max_nonfill)
        painted = 0

        for color in pattern_colors:
            r = random.randrange(h)
            c = random.randrange(w)
            if core[r, c] == int(fill):
                painted += 1
            core[r, c] = int(color)

        while painted < target_nonfill:
            color = random.choice(pattern_colors)
            shape_kind = random.choice(["dot", "line", "rect"])
            if shape_kind == "dot":
                cells = [(random.randrange(h), random.randrange(w))]
            elif shape_kind == "line":
                line_len = random.randint(2, min(5, max(h, w)))
                if random.random() < 0.5:
                    rr = random.randrange(h)
                    c0 = random.randrange(max(1, w - line_len + 1))
                    cells = [(rr, c0 + dc) for dc in range(line_len)]
                else:
                    r0 = random.randrange(max(1, h - line_len + 1))
                    cc = random.randrange(w)
                    cells = [(r0 + dr, cc) for dr in range(line_len)]
            else:
                mh = random.randint(1, min(3, h))
                mw = random.randint(1, min(3, w))
                r0 = random.randrange(h - mh + 1)
                c0 = random.randrange(w - mw + 1)
                cells = [(r0 + dr, c0 + dc) for dr in range(mh) for dc in range(mw)]

            remaining = target_nonfill - painted
            fill_cells = [(r, c) for r, c in cells if core[r, c] == int(fill)]
            if remaining <= 0:
                break
            if len(fill_cells) > remaining:
                fill_cells = random.sample(fill_cells, remaining)
            for r, c in fill_cells:
                core[r, c] = int(color)
            painted += len(fill_cells)

        # Add a small corner-biased marker to suppress D8 symmetry.
        marker_color = random.choice(pattern_colors)
        corner = random.choice(
            [
                (0, 0),
                (0, max(0, w - 2)),
                (max(0, h - 2), 0),
                (max(0, h - 2), max(0, w - 2)),
            ]
        )
        mr, mc = corner
        marker_cells = [(mr, mc), (min(h - 1, mr + 1), mc), (min(h - 1, mr + 1), min(w - 1, mc + 1))]
        for r, c in marker_cells:
            core[r, c] = int(marker_color)

        src_count = int(np.sum(core == int(src_color)))
        if src_count < min_src:
            fill_cells = [tuple(map(int, rc)) for rc in np.argwhere(core == int(fill))]
            random.shuffle(fill_cells)
            for r, c in fill_cells[: min_src - src_count]:
                core[r, c] = int(src_color)

        if _is_visibly_asymmetric(core):
            return core

    raise RuntimeError("failed to build asymmetric core")


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


def _is_visibly_asymmetric(arr: np.ndarray) -> bool:
    for k in range(4):
        for flip in (False, True):
            if k == 0 and not flip:
                continue
            t = _transform(arr, k, flip)
            if t.shape == arr.shape and np.array_equal(t, arr):
                return False
    return True


def _build_legend_rows(
    actual_row: tuple[int, int, int, int], palette: list[int], n_rows: int
) -> list[tuple[int, int, int, int]]:
    rows = [tuple(map(int, actual_row))]
    used_keys = {tuple(map(int, actual_row[:2]))}
    while len(rows) < int(n_rows):
        h_ctrl = random.choice(palette)
        v_ctrl = random.choice(palette)
        if h_ctrl == v_ctrl:
            continue
        key = (int(h_ctrl), int(v_ctrl))
        if key in used_keys:
            continue
        src = random.choice(palette)
        dst_choices = [c for c in palette if c != int(src)]
        dst = random.choice(dst_choices)
        rows.append((int(h_ctrl), int(v_ctrl), int(src), int(dst)))
        used_keys.add(key)
    random.shuffle(rows)
    return rows


def _legend_block(rows: list[tuple[int, int, int, int]]) -> np.ndarray:
    block = np.zeros((2 * len(rows) - 1, 4), dtype=int)
    for i, row in enumerate(rows):
        block[2 * i, :] = np.array(row, dtype=int)
    return block


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(26, 32)
    horiz_side, vert_side = random.choice(
        [("top", "left"), ("top", "right"), ("bottom", "left"), ("bottom", "right")]
    )

    semantic_colors = [1, 2, 3, 4, 6, 7, 8, 9]
    ctrl_h, ctrl_v = random.sample(semantic_colors, 2)

    remaining = [c for c in semantic_colors if c not in {ctrl_h, ctrl_v}]
    fill = random.choice(remaining)
    remaining = [c for c in remaining if c != fill]
    dst = random.choice(remaining)
    remaining = [c for c in remaining if c != dst]

    n_pattern_colors = random.randint(2, min(4, len(remaining)))
    pattern_colors = random.sample(remaining, n_pattern_colors)
    src = random.choice(pattern_colors)

    max_core_h = min(17, n - 8)
    max_core_w = min(17, n - 8)
    if max_core_h < 6 or max_core_w < 6:
        return None

    core_h = random.randint(6, max_core_h)
    core_w = random.randint(6, max_core_w)
    core_before = _build_core(core_h, core_w, fill, pattern_colors, src)
    if int(np.sum(core_before == int(src))) < 3:
        return None
    if int(np.sum(core_before == int(dst))) > 0:
        return None

    output = _replace_color(core_before, src, dst)
    if np.array_equal(output, core_before):
        return None

    guide_colors = {horiz_side: ctrl_h, vert_side: ctrl_v}
    aligned_object = _attach_guides(core_before, (horiz_side, vert_side), guide_colors)

    k = random.randrange(4)
    flip = bool(random.randrange(2))
    input_object = _inverse_transform(aligned_object, k, flip)

    legend_rows = _build_legend_rows((ctrl_h, ctrl_v, src, dst), semantic_colors, random.randint(3, 5))
    legend_block = _legend_block(legend_rows)

    grid = np.zeros((n, n), dtype=int)
    _paint_frame(grid, horiz_side, vert_side, ctrl_h, ctrl_v)

    try:
        r0, c0 = random_free_location(grid, input_object, transparent=0, border=1, padding=1)
        paste(grid, input_object, r0, c0, transparent=0)
        lr, lc = random_free_location(grid, legend_block, transparent=0, border=1, padding=1)
        paste(grid, legend_block, lr, lc, transparent=0)
    except ValueError:
        return None

    metrics = variant_metrics(grid)
    if metrics["matched_key_count"] != 1:
        return None
    if metrics["changed_cells"] < 3:
        return None
    if metrics["legend_rows"] < 3:
        return None
    if not np.array_equal(main(grid), output):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate input for fc7cae8d D8 variant")


def variant_metrics(grid: np.ndarray) -> dict[str, int]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    ctrls = _detect_controls_frame(g, bg)
    desired = {side: col for side, col in ctrls}
    obj = _extract_center_object(g, bg)
    phase1 = _align_and_crop(obj, desired)

    legend_rows = _extract_legend_rows(g, bg)
    ctrl_key = _control_key(ctrls)
    matched_key_count = 0
    recolored_cells = 0
    changed_cells = 0

    if ctrl_key is not None:
        matched = [row for row in legend_rows if tuple(row[:2]) == ctrl_key]
        matched_key_count = len(matched)
        if matched:
            _h, _v, src, dst = matched[0]
            recolored_cells = int(np.sum(phase1 == int(src)))
            changed_cells = recolored_cells if int(src) != int(dst) else 0

    return {
        "legend_rows": int(len(legend_rows)),
        "matched_key_count": int(matched_key_count),
        "recolored_cells": int(recolored_cells),
        "changed_cells": int(changed_cells),
    }
