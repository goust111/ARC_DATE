# Combo variant of fc7cae8d (index 39)
# combo_spec: D8/0+D9/0+D10/0
# Name: 两步变换+图例+干扰

from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import crop_bbox, find_components, get_bg, paste, random_free_location

"""
设计分析
========
a) 具体化与行为追踪
- 原题只有一个内部对象, 引导边颜色直接等于边框控制色, 所以 solver 可以:
  1. 取最大内部连通块。
  2. 试 D8 变换, 让该对象的两条引导边直接对齐边框控制边。
  3. 裁掉两条引导边, 输出核心。
- 融合后输入含三个可见层次:
  1. 一个真正源矩形: 内部核心外包两条引导边, 但引导色 != 边框控制色。
  2. 1-2 个无引导边的干扰矩形。
  3. 一个角落 legend 区域: 上半部分若干 1x2 色条表示 `guide -> control`,
     下半部分若干 1x4 色条表示 `(h_ctrl, v_ctrl) -> (src, dst)`。
- 小例子:
  边框控制边是 `top=2,left=6`; 真正对象暴露的引导边是 `right=8,bottom=4`;
  legend 里有 `8->2, 4->6`, 以及规则行 `[2,6,7,3]`。
  第一步不能再直接拿 `8/4` 去匹配 `2/6`, 必须先查 legend, 认出该对象经 D8 变换后
  会把 `8` 对齐到 top、`4` 对齐到 left, 再裁掉这两条引导边得到 core。
  第二步再根据控制色对 `(2,6)` 查到 `7->3`, 把 core 里所有 `7` 变成 `3`。
- 差异有三层:
  1. “最大内部块就是目标”不成立, 因为有干扰矩形。
  2. “引导色直接等于控制色”不成立, 因为需要 `guide -> control` 解码。
  3. “裁剪结果就是最终答案”不成立, 因为还要做一次 legend 驱动的改色。

b) 复杂度校准
- legend 不只放真实键, 而是放 3-4 条 guide 映射和 3-4 条 recolor 规则, 否则会退化成
  “输入里直接附一条答案说明”。真实键只是其中一条。
- 被替换的 `src` 在核心里至少出现 3 次, 且 `dst` 预先不在核心中, 确保第二步肉眼可见。
- 干扰矩形始终存在, 且首个干扰通常与源对象接近或更大, 明确破坏原题“最大块”捷径。
- 核心保留 fill 外框, 图案只画在内部, 这样真正对象的引导边与普通矩形边界在视觉上更可区分,
  也避免干扰矩形偶然长出伪 guide。

c) 隐式假设审查
- 原 `extract_center_object()` 假设“最大内部连通块就是目标”; 融合后失效, 必须改成枚举全部内部组件。
- 原 `guide_edges()` 假设“非 fill 的整条边就是 guide”; 这仍可用, 但前提变成:
  真对象核心外圈保持 fill, 干扰矩形也不把图案画到最外圈, 否则可能伪造 guide。
- 原 `main()` 假设“边框控制色与 guide 色直接相等”; 融合后必须先解析 1x2 legend 行得到 `guide_to_control`。
- 新增 recolor 阶段假设“存在唯一 `(h_ctrl,v_ctrl)` 规则行”; generator 显式保证 pair key 唯一。
- legend 解析本身依赖“1x2/1x4 行与其他对象分离”; generator 用角落布局和空行分隔保证这一点。

d) 放置合法性
- legend 固定放在内侧角落 `(1,1)` / `(1,n-1-w)` / `(n-1-h,1)` / `(n-1-h,n-1-w)` 之一,
  不触边框控制线, 也不越界。
- 真对象与干扰对象使用 `random_free_location(..., border=1, padding=1)` 放置,
  保证不重叠、不断开边框、彼此保持空隙。
- 干扰矩形最小 5x5, legend 行最小组件只有 1x2 / 1x4, 组件尺寸上也容易区分。

e) 视觉可观察性
- 人类能看到: 边框给出两个控制边; 内部有多个矩形但只有一个矩形带两条彩色引导边;
  角落 legend 同时给出 guide 解码表和改色规则表; 输出则是该矩形被对齐裁掉 guide 后,
  其内部某一种图案色被统一替换。
- 多个样例里只要控制边色对变化, 对应的 guide 色对和 recolor 规则也随 legend 变化,
  人类可归纳出“先解码 guide 再对齐, 最后按控制色对改色”的顺序关系。

f) 可视化整洁度
- legend 被集中在单一角落区域, 干扰矩形只放在主体活动区, 不与 legend 叠加。
- guide 解码行固定宽 2, recolor 行固定宽 4, 并用空行分节, 视觉上比散落的小组件更整洁。
- 边框只保留两条纯控制边, 不再叠加原题的灰噪声边, 避免和 legend / 干扰共同造成拥挤。

g) 颜色变化可推断性
- 第一阶段的颜色映射不是任意全局置换, 只是少量离散的 `guide -> control` 成对说明。
- 第二阶段的颜色变化也不是大规模重绘, 只对一个 `src` 颜色做全局替换, 且通过
  `(h_ctrl, v_ctrl, src, dst)` 结构化行明示条件与结果。
- generator 控制总色数在 ARC 常见范围内, 并让 `src` 重复出现, 否则图例虽然存在但人类很难确认
  “究竟哪种颜色被规则触发了”。

新增 solver 假设与 generator 对应保证
------------------------------------
- 角落 legend 中所有 guide 映射颜色键唯一, 所有 recolor 控制键唯一。
- 恰有一个内部组件在某个 D8 变换下, 其两条 guide 经过 `guide_to_control` 映射后匹配边框控制边。
- 真对象和干扰矩形的最外圈都是 fill, 所以只有真对象会被 `_guide_edges()` 识别出两条引导边。
- 真对象 guide 色与控制色始终不同, 第二步实际改色单元数始终 >= 3, 防止退化回原题或仅 D10/D9 子题。
"""


def _transform(arr: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.rot90(arr, int(k))
    return np.fliplr(out) if flip else out


def _inverse_transform(arr: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.fliplr(arr) if flip else arr
    return np.rot90(out, -int(k))


def _detect_controls_frame(arr: np.ndarray, bg: int) -> list[tuple[str, int]]:
    h, w = arr.shape
    sides = {
        "top": arr[0, 1 : w - 1],
        "bottom": arr[h - 1, 1 : w - 1],
        "left": arr[1 : h - 1, 0],
        "right": arr[1 : h - 1, w - 1],
    }
    cand: list[tuple[str, int, int, int]] = []
    for side, values in sides.items():
        vals = list(map(int, values))
        non_bg = [x for x in vals if x != int(bg)]
        if not non_bg:
            continue
        col, count = Counter(non_bg).most_common(1)[0]
        if int(count) >= len(vals) - 1:
            cand.append((side, int(col), int(count), int(len(vals))))
    cand.sort(key=lambda t: (t[2] / t[3], t[2]), reverse=True)
    return [(side, col) for side, col, _count, _length in cand[:2]]


def _control_key(ctrls: list[tuple[str, int]]) -> tuple[int, int] | None:
    horiz = [int(col) for side, col in ctrls if side in ("top", "bottom")]
    vert = [int(col) for side, col in ctrls if side in ("left", "right")]
    if not horiz or not vert:
        return None
    return int(horiz[0]), int(vert[0])


def _guide_edges(arr: np.ndarray) -> dict[str, int]:
    fill = int(Counter(map(int, arr.flatten())).most_common(1)[0][0])
    edges = {
        "top": arr[0, :],
        "bottom": arr[-1, :],
        "left": arr[:, 0],
        "right": arr[:, -1],
    }
    out: dict[str, int] = {}
    for side, values in edges.items():
        vals = list(map(int, values))
        non_fill = [x for x in vals if x != int(fill)]
        if not non_fill:
            continue
        col, count = Counter(non_fill).most_common(1)[0]
        if int(count) >= len(vals) - 1:
            out[side] = int(col)
    return out


def _crop_guides(arr: np.ndarray, desired: dict[str, int]) -> np.ndarray:
    top = 1 if "top" in desired else 0
    bottom = int(arr.shape[0]) - (1 if "bottom" in desired else 0)
    left = 1 if "left" in desired else 0
    right = int(arr.shape[1]) - (1 if "right" in desired else 0)
    return arr[int(top) : int(bottom), int(left) : int(right)]


def _interior_components(arr: np.ndarray, bg: int) -> list[dict]:
    h, w = arr.shape
    comps = find_components(arr, bg=int(bg), connectivity=4, monochromatic=False, return_format="dict")
    out: list[dict] = []
    for comp in comps:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        if r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1:
            continue
        out.append(comp)
    return out


def _extract_legend_pairs(arr: np.ndarray, bg: int) -> list[tuple[int, int]]:
    rows: list[tuple[int, int]] = []
    for comp in _interior_components(arr, bg):
        if int(comp["area"]) != 2:
            continue
        patch = crop_bbox(arr, tuple(map(int, comp["bbox"])))
        if patch.shape != (1, 2):
            continue
        rows.append((int(patch[0, 0]), int(patch[0, 1])))
    return rows


def _extract_legend_rules(arr: np.ndarray, bg: int) -> list[tuple[int, int, int, int]]:
    rows: list[tuple[int, int, int, int]] = []
    for comp in _interior_components(arr, bg):
        if int(comp["area"]) != 4:
            continue
        patch = crop_bbox(arr, tuple(map(int, comp["bbox"])))
        if patch.shape != (1, 4):
            continue
        rows.append(tuple(map(int, patch[0])))
    return rows


def _guide_map(rows: list[tuple[int, int]]) -> dict[int, int]:
    return {int(guide): int(ctrl) for guide, ctrl in rows}


def _rule_map(rows: list[tuple[int, int, int, int]]) -> dict[tuple[int, int], tuple[int, int]]:
    return {(int(h), int(v)): (int(src), int(dst)) for h, v, src, dst in rows}


def _find_target_matches(
    arr: np.ndarray, bg: int, desired: dict[str, int], guide_to_control: dict[int, int]
) -> list[dict]:
    matches: list[dict] = []
    for comp in _interior_components(arr, bg):
        bbox = tuple(map(int, comp["bbox"]))
        obj = crop_bbox(arr, bbox)
        valid: list[dict] = []
        for k in range(4):
            base = np.rot90(obj, int(k))
            for flip in (False, True):
                transformed = np.fliplr(base) if flip else base
                guide_edges = _guide_edges(transformed)
                if set(guide_edges) != set(desired):
                    continue
                mapped = {side: guide_to_control.get(int(col), -1) for side, col in guide_edges.items()}
                if any(int(mapped[side]) != int(desired[side]) for side in desired):
                    continue
                core = _crop_guides(transformed, desired)
                if core.shape[0] < 3 or core.shape[1] < 3:
                    continue
                valid.append(
                    {
                        "bbox": bbox,
                        "area": int(comp["area"]),
                        "core": core,
                        "guide_edges": guide_edges,
                        "k": int(k),
                        "flip": bool(flip),
                    }
                )
        if valid:
            chosen = max(valid, key=lambda item: (item["core"].shape[0] * item["core"].shape[1], item["area"]))
            chosen["transform_count"] = int(len(valid))
            matches.append(chosen)
    return matches


def _replace_color(arr: np.ndarray, src: int, dst: int) -> np.ndarray:
    out = np.array(arr, copy=True)
    out[out == int(src)] = int(dst)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    1. Read the legend's guide->control rows.
    2. Find the unique interior rectangle whose guide edges can align to the frame controls under D8.
    3. Crop those aligned guide edges.
    4. Use the frame control-color pair to select a recolor rule from the legend and recolor the core.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    ctrls = _detect_controls_frame(g, bg)
    desired = {side: col for side, col in ctrls}
    guide_to_control = _guide_map(_extract_legend_pairs(g, bg))
    matches = _find_target_matches(g, bg, desired, guide_to_control)
    if not matches:
        raise ValueError("no guided interior component matches the frame controls after legend decoding")

    target = max(matches, key=lambda item: (item["area"], -item["transform_count"]))
    core = np.array(target["core"], copy=True)

    ctrl_key = _control_key(ctrls)
    if ctrl_key is None:
        return core
    rules = _rule_map(_extract_legend_rules(g, bg))
    if ctrl_key not in rules:
        return core
    src, dst = rules[ctrl_key]
    return _replace_color(core, src, dst)


def _legacy_main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    ctrls = _detect_controls_frame(g, bg)
    desired = {side: col for side, col in ctrls}

    best: tuple[int, tuple[int, int, int, int]] | None = None
    for comp in _interior_components(g, bg):
        area = int(comp["area"])
        bbox = tuple(map(int, comp["bbox"]))
        if best is None or area > best[0]:
            best = (area, bbox)
    if best is None:
        return g.copy()

    obj = crop_bbox(g, best[1])
    for k in range(4):
        base = np.rot90(obj, int(k))
        for flip in (False, True):
            transformed = np.fliplr(base) if flip else base
            guide_edges = _guide_edges(transformed)
            if all(side in guide_edges and int(guide_edges[side]) == int(col) for side, col in desired.items()):
                return _crop_guides(transformed, desired)
    return obj


def _is_visibly_asymmetric(arr: np.ndarray) -> bool:
    for k in range(4):
        for flip in (False, True):
            if k == 0 and not flip:
                continue
            transformed = _transform(arr, k, flip)
            if transformed.shape == arr.shape and np.array_equal(transformed, arr):
                return False
    return True


def _build_core(h: int, w: int, fill: int, pattern_colors: list[int], src_color: int) -> np.ndarray:
    interior_h = h - 2
    interior_w = w - 2
    min_src = max(3, (h * w) // 16)

    for _ in range(160):
        core = np.full((h, w), int(fill), dtype=int)
        painted = 0
        target_nonfill = random.randint(max(len(pattern_colors) + 4, (interior_h * interior_w) // 6), max(8, (interior_h * interior_w) // 3))

        for color in pattern_colors:
            r = random.randrange(1, h - 1)
            c = random.randrange(1, w - 1)
            if core[r, c] == int(fill):
                painted += 1
            core[r, c] = int(color)

        while painted < target_nonfill:
            color = random.choice(pattern_colors)
            shape_kind = random.choice(["dot", "line", "rect"])
            if shape_kind == "dot":
                cells = [(random.randrange(1, h - 1), random.randrange(1, w - 1))]
            elif shape_kind == "line":
                if random.random() < 0.5:
                    length = random.randint(2, min(5, interior_w))
                    rr = random.randrange(1, h - 1)
                    c0 = random.randrange(1, w - length)
                    cells = [(rr, c0 + dc) for dc in range(length)]
                else:
                    length = random.randint(2, min(5, interior_h))
                    r0 = random.randrange(1, h - length)
                    cc = random.randrange(1, w - 1)
                    cells = [(r0 + dr, cc) for dr in range(length)]
            else:
                mh = random.randint(1, min(3, interior_h))
                mw = random.randint(1, min(3, interior_w))
                r0 = random.randrange(1, h - mh)
                c0 = random.randrange(1, w - mw)
                cells = [(r0 + dr, c0 + dc) for dr in range(mh) for dc in range(mw)]

            remaining = target_nonfill - painted
            if remaining <= 0:
                break
            fill_cells = [(r, c) for r, c in cells if core[r, c] == int(fill)]
            if len(fill_cells) > remaining:
                fill_cells = random.sample(fill_cells, remaining)
            for r, c in fill_cells:
                core[r, c] = int(color)
            painted += len(fill_cells)

        marker_color = random.choice(pattern_colors)
        mr, mc = random.choice([(1, 1), (1, w - 3), (h - 3, 1), (h - 3, w - 3)])
        marker_cells = [(mr, mc), (mr + 1, mc), (mr + 1, mc + 1)]
        for r, c in marker_cells:
            core[r, c] = int(marker_color)

        src_count = int(np.sum(core == int(src_color)))
        if src_count < min_src:
            fill_cells = [tuple(map(int, rc)) for rc in np.argwhere(core[1:-1, 1:-1] == int(fill))]
            fill_cells = [(r + 1, c + 1) for r, c in fill_cells]
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


def _build_pair_rows(actual_rows: list[tuple[int, int]], palette: list[int], n_rows: int) -> list[tuple[int, int]]:
    rows = [tuple(map(int, row)) for row in actual_rows]
    used_guides = {int(g) for g, _c in rows}
    used_ctrls = {int(c) for _g, c in rows}
    while len(rows) < int(n_rows):
        guide_choices = [c for c in palette if c not in used_guides]
        ctrl_choices = [c for c in palette if c not in used_ctrls]
        if not guide_choices or not ctrl_choices:
            break
        guide = random.choice(guide_choices)
        ctrl = random.choice(ctrl_choices)
        rows.append((int(guide), int(ctrl)))
        used_guides.add(int(guide))
        used_ctrls.add(int(ctrl))
    random.shuffle(rows)
    return rows


def _build_rule_rows(
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


def _legend_block(pair_rows: list[tuple[int, int]], rule_rows: list[tuple[int, int, int, int]]) -> np.ndarray:
    height = 2 * len(pair_rows) + 2 * len(rule_rows)
    block = np.zeros((height, 4), dtype=int)
    r = 0
    for guide, ctrl in pair_rows:
        block[r, 0] = int(guide)
        block[r, 1] = int(ctrl)
        r += 2
    for h_ctrl, v_ctrl, src, dst in rule_rows:
        block[r, :] = np.array([h_ctrl, v_ctrl, src, dst], dtype=int)
        r += 2
    return block


def _place_legend_corner(grid: np.ndarray, legend: np.ndarray) -> None:
    n = grid.shape[0]
    h, w = legend.shape
    corners = [(2, 2), (2, n - 2 - w), (n - 2 - h, 2), (n - 2 - h, n - 2 - w)]
    random.shuffle(corners)
    for r0, c0 in corners:
        if r0 < 1 or c0 < 1:
            continue
        if r0 + h > n - 1 or c0 + w > n - 1:
            continue
        region = grid[r0 : r0 + h, c0 : c0 + w]
        if not np.any((region != 0) & (legend != 0)):
            paste(grid, legend, r0, c0, transparent=0)
            return
    raise ValueError("failed to place legend in a corner")


def _choose_core_dims(n: int) -> tuple[int, int] | None:
    max_h = min(12, n - 16)
    max_w = min(12, n - 16)
    if max_h < 6 or max_w < 6:
        return None
    return random.randint(6, max_h), random.randint(6, max_w)


def _choose_distractor_dims(n: int, source_h: int, source_w: int, want_large: bool, idx: int) -> tuple[int, int] | None:
    max_h = min(14, n - 10)
    max_w = min(14, n - 10)
    if want_large and idx == 0:
        min_h = min(max_h, max(5, source_h - 1))
        min_w = min(max_w, max(5, source_w - 1))
        max_h = max(min_h, min(max_h, source_h + 2))
        max_w = max(min_w, min(max_w, source_w + 2))
    else:
        min_h = 5
        min_w = 5
    if min_h > max_h or min_w > max_w:
        return None
    return random.randint(min_h, max_h), random.randint(min_w, max_w)


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(30, 36)
    horiz_side, vert_side = random.choice(
        [("top", "left"), ("top", "right"), ("bottom", "left"), ("bottom", "right")]
    )

    palette = list(range(1, 10))
    ctrl_h, ctrl_v = random.sample(palette, 2)
    remaining = [c for c in palette if c not in {ctrl_h, ctrl_v}]
    guide_h, guide_v = random.sample(remaining, 2)
    remaining = [c for c in remaining if c not in {guide_h, guide_v}]
    fill = random.choice(remaining)
    remaining = [c for c in remaining if c != fill]
    dst = random.choice(remaining)
    remaining = [c for c in remaining if c != dst]
    n_pattern = random.randint(2, min(3, len(remaining)))
    pattern_colors = random.sample(remaining, n_pattern)
    src = random.choice(pattern_colors)

    dims = _choose_core_dims(n)
    if dims is None:
        return None
    core_h, core_w = dims
    core_before = _build_core(core_h, core_w, fill, pattern_colors, src)
    if int(np.sum(core_before == int(src))) < 3:
        return None
    if int(np.sum(core_before == int(dst))) > 0:
        return None

    output = _replace_color(core_before, src, dst)
    if np.array_equal(output, core_before):
        return None

    guide_colors = {horiz_side: guide_h, vert_side: guide_v}
    aligned_object = _attach_guides(core_before, (horiz_side, vert_side), guide_colors)
    k = random.randrange(4)
    flip = bool(random.randrange(2))
    input_object = _inverse_transform(aligned_object, k, flip)

    pair_rows = _build_pair_rows([(guide_h, ctrl_h), (guide_v, ctrl_v)], palette, random.randint(3, 4))
    rule_rows = _build_rule_rows((ctrl_h, ctrl_v, src, dst), palette, random.randint(3, 4))
    legend_block = _legend_block(pair_rows, rule_rows)

    grid = np.zeros((n, n), dtype=int)
    _paint_frame(grid, horiz_side, vert_side, ctrl_h, ctrl_v)
    try:
        _place_legend_corner(grid, legend_block)
        r0, c0 = random_free_location(grid, input_object, transparent=0, border=1, padding=1)
        paste(grid, input_object, r0, c0, transparent=0)
    except ValueError:
        return None

    n_distractors = random.randint(1, 2)
    want_large = random.random() < 0.75
    source_h, source_w = input_object.shape
    distractor_palette = [c for c in palette if c not in {ctrl_h, ctrl_v, guide_h, guide_v}]

    for idx in range(n_distractors):
        dims = _choose_distractor_dims(n, source_h, source_w, want_large, idx)
        if dims is None:
            return None
        dh, dw = dims
        d_fill = random.choice(distractor_palette)
        d_pattern_pool = [c for c in distractor_palette if c != d_fill]
        d_pattern_count = random.randint(2, min(3, len(d_pattern_pool)))
        distractor = _build_core(dh, dw, d_fill, random.sample(d_pattern_pool, d_pattern_count), random.choice(d_pattern_pool))
        try:
            dr, dc = random_free_location(grid, distractor, transparent=0, border=1, padding=1)
        except ValueError:
            return None
        paste(grid, distractor, dr, dc, transparent=0)

    metrics = variant_metrics(grid)
    if metrics["pair_rows"] < 3:
        return None
    if metrics["rule_rows"] < 3:
        return None
    if metrics["candidate_components"] != 1:
        return None
    if metrics["distractor_count"] < 1:
        return None
    if metrics["indirect_guides"] != 2:
        return None
    if metrics["changed_cells"] < 3:
        return None
    if metrics["phase1_equal"] != 0:
        return None
    if metrics["legacy_equal"] != 0:
        return None
    if not np.array_equal(main(grid), output):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate input for fc7cae8d combo 39")


def variant_metrics(grid: np.ndarray) -> dict[str, int]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    ctrls = _detect_controls_frame(g, bg)
    desired = {side: col for side, col in ctrls}
    ctrl_key = _control_key(ctrls)
    pair_rows = _extract_legend_pairs(g, bg)
    rule_rows = _extract_legend_rules(g, bg)
    guide_to_control = _guide_map(pair_rows)
    matches = _find_target_matches(g, bg, desired, guide_to_control)

    changed_cells = 0
    indirect_guides = 0
    phase1_equal = 1
    distractor_count = 0
    transform_count = 0

    final = main(g)
    if matches:
        target = max(matches, key=lambda item: (item["area"], -item["transform_count"]))
        core = target["core"]
        transform_count = int(target["transform_count"])
        indirect_guides = sum(int(target["guide_edges"][side]) != int(desired[side]) for side in desired)
        phase1_equal = int(np.array_equal(core, final))

        rules = _rule_map(rule_rows)
        if ctrl_key is not None and ctrl_key in rules:
            src, dst = rules[ctrl_key]
            changed_cells = int(np.sum(core == int(src))) if int(src) != int(dst) else 0

        target_bbox = tuple(map(int, target["bbox"]))
        for comp in _interior_components(g, bg):
            bbox = tuple(map(int, comp["bbox"]))
            patch = crop_bbox(g, bbox)
            is_pair = int(comp["area"]) == 2 and patch.shape == (1, 2)
            is_rule = int(comp["area"]) == 4 and patch.shape == (1, 4)
            if is_pair or is_rule:
                continue
            if bbox != target_bbox:
                distractor_count += 1

    legacy_equal = int(np.array_equal(_legacy_main(g), final))
    return {
        "pair_rows": int(len(pair_rows)),
        "rule_rows": int(len(rule_rows)),
        "candidate_components": int(len(matches)),
        "target_transform_count": int(transform_count),
        "distractor_count": int(distractor_count),
        "indirect_guides": int(indirect_guides),
        "changed_cells": int(changed_cells),
        "phase1_equal": int(phase1_equal),
        "legacy_equal": int(legacy_equal),
    }
