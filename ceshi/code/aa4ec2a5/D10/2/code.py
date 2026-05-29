# Variant of aa4ec2a5
# Dimension: D10 — 洞的封闭性定义 (meta)
# Change: 洞定义为被蓝色对象像素完全4连通封闭的背景色区域（即该背景区域不与网格外部背景相连）... → 凹陷也视为洞
#
# Design analysis:
# spec:
#   core_change:
#     old: 只有 bbox 内完全不接触 bbox 边界的 4 连通背景区域会触发青色/品红色重着色
#     new: 除了旧式封闭洞，还把对象边界上的深凹陷视作洞
#   deep_indentation_definition:
#     shape: 从对象 tight bbox 的某一边向内延伸的矩形背景槽
#     opening_width: 1-2
#     depth: >= 3
#     side_walls: 槽的两侧必须整段贴着前景像素
#     closure: 槽的最深一行/列之后必须有一整段前景像素封底
#     non_examples:
#       - 宽度 >= 3 的凹口
#       - 深度 <= 2 的浅凹口
#       - 贴着 bbox 角点、缺少双侧墙体的普通外部背景
# a) 具体化与行为追踪:
#   - 输入仍是黄色背景上的多个蓝色 4 连通对象。
#   - 输出仍给每个对象画 1 像素 8 邻域红色轮廓。
#   - 旧题里只有真正封闭的孔洞对象会变成青色、洞填品红；现在一个顶部/底部/左右的窄深槽也会触发同样操作。
#   - 例子: 一个 8x8 实心矩形，从顶边向下挖去 1x4 的槽。旧逻辑把该背景视为外部，不会重着色；新逻辑把这 4 个背景格当作洞，整块前景变青、槽填品红，外轮廓按“已填实”后的形状计算。
# b) 复杂度校准:
#   - 不把所有凹口都算洞，只保留少量离散档位: 宽度 1-2 且深度 >= 3。
#   - 生成器稳定产出至少一个深凹陷对象，并额外放入一个浅/宽凹口负例，避免规则退化成“只要有凹口就变色”。
# c) 原 main() 的隐式假设审查:
#   - `_interior_holes` 依赖“洞一定不碰 bbox 边”；该前提在变体中失效，所以必须新增边界槽扫描。
#   - `if holes:` 过去只由封闭洞触发；现在要改成“任何特征洞区域”触发。
#   - `solid[hole] = True` 过去只填封闭洞；现在也必须把深凹陷填入 solid，否则轮廓仍会沿凹槽内部出现。
#   - `out[outline & (out == bg)] = 2` 仍然安全，因为 special 区域在画轮廓前已被改成 6，不会被红色覆盖。
# d) 放置合法性:
#   - 仍保持对象之间 Chebyshev 距离 >= 3、离画布边界 >= 1，避免红色轮廓互相覆盖。
#   - 深凹陷对象在 tight bbox 内构造，mouth 不贴 bbox 角，保证 solver 的边界扫描定义可被稳定触发。
# e) 视觉可观察性:
#   - 每个样例至少有一个深凹陷对象，且其输出与原题不同: 凹槽被填成品红，整块变青，红轮廓跨过原凹口。
#   - 同时保留至少一个浅/宽凹口对象，输出保持蓝色并保留凹口内部红边，便于人类比较“深/浅、窄/宽”的差异。
# f) 可视化整洁度:
#   - 凹槽只做单个矩形，宽度很小、深度适中，避免一个对象内出现过多重叠标记。
#   - 特征对象数量仍控制在 1-2 个，其余对象保持原题风格的普通实心块。
# g) 颜色变化可推断性:
#   - 颜色规则没有新增颜色，只复用原题的蓝/青/品红/红四种语义。
#   - 颜色变化由可见的几何结构驱动: 封闭洞或深窄凹陷 -> 青色主体 + 品红区域。
# h) 变体行为指标:
#   - 关键指标 1: 每个生成样例的 deep_indentation 组件数 >= 1。
#   - 关键指标 2: 同一输入上，新 solver 与原 solver 的输出应因深凹陷填充而出现差异。
# i) 新 solver 的输入假设:
#   - 深凹陷检测只识别轴对齐、矩形、mouth 宽 1-2 的 cul-de-sac。
#   - `generate_input()` 显式构造这种凹槽，并过滤会意外触发/误触发的新旧形状，确保这些假设始终成立。

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, find_components, get_bg, is_contiguous


def _coords_mask(shape: tuple[int, int], coords: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[coords[:, 0], coords[:, 1]] = True
    return mask


def _edge_runs(line: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    i = 0
    n = len(line)
    while i < n:
        if not bool(line[i]):
            i += 1
            continue
        j = i
        while j + 1 < n and bool(line[j + 1]):
            j += 1
        runs.append((i, j))
        i = j + 1
    return runs


def _interior_holes_in_sub(sub: np.ndarray) -> list[np.ndarray]:
    holes: list[np.ndarray] = []
    h, w = sub.shape
    for hole in components_mask(~sub, connectivity=4):
        touches_edge = np.any(
            (hole[:, 0] == 0)
            | (hole[:, 1] == 0)
            | (hole[:, 0] == h - 1)
            | (hole[:, 1] == w - 1)
        )
        if not bool(touches_edge):
            holes.append(hole)
    return holes


def _deep_notch_regions_in_sub(sub: np.ndarray) -> list[np.ndarray]:
    h, w = sub.shape
    regions: list[np.ndarray] = []

    for c0, c1 in _edge_runs(~sub[0]):
        width = c1 - c0 + 1
        if width > 2 or c0 == 0 or c1 == w - 1:
            continue
        depth = 0
        while depth < h:
            if sub[depth, c0 : c1 + 1].any():
                break
            if not bool(sub[depth, c0 - 1]) or not bool(sub[depth, c1 + 1]):
                break
            depth += 1
        if depth >= 3 and depth < h and sub[depth, c0 : c1 + 1].all():
            regions.append(
                np.array(
                    [(r, c) for r in range(depth) for c in range(c0, c1 + 1)],
                    dtype=int,
                )
            )

    for c0, c1 in _edge_runs(~sub[h - 1]):
        width = c1 - c0 + 1
        if width > 2 or c0 == 0 or c1 == w - 1:
            continue
        depth = 0
        while depth < h:
            r = h - 1 - depth
            if sub[r, c0 : c1 + 1].any():
                break
            if not bool(sub[r, c0 - 1]) or not bool(sub[r, c1 + 1]):
                break
            depth += 1
        close_r = h - 1 - depth
        if depth >= 3 and close_r >= 0 and sub[close_r, c0 : c1 + 1].all():
            regions.append(
                np.array(
                    [(r, c) for r in range(h - depth, h) for c in range(c0, c1 + 1)],
                    dtype=int,
                )
            )

    for r0, r1 in _edge_runs(~sub[:, 0]):
        width = r1 - r0 + 1
        if width > 2 or r0 == 0 or r1 == h - 1:
            continue
        depth = 0
        while depth < w:
            if sub[r0 : r1 + 1, depth].any():
                break
            if not bool(sub[r0 - 1, depth]) or not bool(sub[r1 + 1, depth]):
                break
            depth += 1
        if depth >= 3 and depth < w and sub[r0 : r1 + 1, depth].all():
            regions.append(
                np.array(
                    [(r, c) for r in range(r0, r1 + 1) for c in range(depth)],
                    dtype=int,
                )
            )

    for r0, r1 in _edge_runs(~sub[:, w - 1]):
        width = r1 - r0 + 1
        if width > 2 or r0 == 0 or r1 == h - 1:
            continue
        depth = 0
        while depth < w:
            c = w - 1 - depth
            if sub[r0 : r1 + 1, c].any():
                break
            if not bool(sub[r0 - 1, c]) or not bool(sub[r1 + 1, c]):
                break
            depth += 1
        close_c = w - 1 - depth
        if depth >= 3 and close_c >= 0 and sub[r0 : r1 + 1, close_c].all():
            regions.append(
                np.array(
                    [(r, c) for r in range(r0, r1 + 1) for c in range(w - depth, w)],
                    dtype=int,
                )
            )

    return regions


def _shift_regions(regions: list[np.ndarray], r0: int, c0: int) -> list[np.ndarray]:
    shifted_regions: list[np.ndarray] = []
    for region in regions:
        shifted = region.copy()
        shifted[:, 0] += int(r0)
        shifted[:, 1] += int(c0)
        shifted_regions.append(shifted)
    return shifted_regions


def _interior_holes(mask: np.ndarray, coords: np.ndarray) -> list[np.ndarray]:
    r0, c0, r1, c1 = bbox_coords(coords)
    sub = mask[r0 : r1 + 1, c0 : c1 + 1]
    return _shift_regions(_interior_holes_in_sub(sub), r0, c0)


def _deep_notch_regions(mask: np.ndarray, coords: np.ndarray) -> list[np.ndarray]:
    r0, c0, r1, c1 = bbox_coords(coords)
    sub = mask[r0 : r1 + 1, c0 : c1 + 1]
    return _shift_regions(_deep_notch_regions_in_sub(sub), r0, c0)


def _outline_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    outline = np.zeros_like(mask, dtype=bool)
    rows, cols = np.where(mask)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr = rows + dr
            cc = cols + dc
            ok = (0 <= rr) & (rr < h) & (0 <= cc) & (cc < w)
            outline[rr[ok], cc[ok]] = True
    outline &= ~mask
    return outline


def main(grid: np.ndarray) -> np.ndarray:
    """
    Add red outlines around objects, and also treat deep narrow indentations as holes.

    Input structure:
    - A mostly uniform background contains one or more 4-connected blue foreground objects.
    - Some objects have true enclosed holes; others may instead have a narrow deep notch that opens to the outside from one side of the object's tight bounding box.

    Rule:
    - For every foreground component, draw a one-cell 8-neighborhood outline in red (2).
    - If a component has either enclosed holes or deep narrow boundary indentations, recolor the component to teal (8).
    - Fill every such hole-like background region with magenta (6) before drawing the outline.

    Key insight:
    - The output is computed from a "filled" version of each component: classical closed holes and qualifying deep indentations both become part of the solid mask used for recoloring and outlining, while shallow or wide concavities remain ordinary background.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True):
        coords = comp["coords"]
        mask = _coords_mask(g.shape, coords)
        enclosed = _interior_holes(mask, coords)
        deep_notches = _deep_notch_regions(mask, coords)
        special_regions = enclosed + deep_notches

        solid = mask.copy()
        if special_regions:
            out[mask] = 8
            for region in special_regions:
                out[region[:, 0], region[:, 1]] = 6
                solid[region[:, 0], region[:, 1]] = True

        outline = _outline_mask(solid)
        out[outline & (out == bg)] = 2

    return out


# --------------- generator (c) ---------------

# Canvas / grid bounds
GRID_H_RANGE = (15, 28)
GRID_W_RANGE = (15, 27)
BG_COLOR = 4
FG_COLOR = 1

# Component counts
COMP_COUNT_RANGE = (3, 4)
SPECIAL_COUNT_RANGE = (1, 2)  # components that should trigger teal/magenta
SECOND_SPECIAL_HOLE_PROB = 0.55

# Polyomino construction (plain solids)
POLY_NUM_RECTS_RANGE = (2, 3)
POLY_RECT_DIM_RANGE = (2, 7)
POLY_CANVAS_PAD = 4
POLY_PLACEMENT_RETRIES = 50
POLY_GEN_RETRIES = 30

# Small solid shapes
SMALL_SHAPE_PROB = 0.2
SMALL_SHAPE_SIZE_RANGE = (1, 4)

# Classical enclosed-hole construction
HOLE_VALID_AREAS = [2, 3, 4, 6, 8, 9, 12, 16]
HOLE_DIM_MAX = 4
HOLLOW_PAD_RANGE = (1, 3)
HOLLOW_MAX_NOTCHES = 2
HOLLOW_EXTENSION_PROB = 0.5
HOLLOW_EXT_H_RANGE = (1, 4)
HOLLOW_EXT_SIDE_W_RANGE = (1, 3)
HOLLOW_HOLE_SEARCH_RETRIES = 200
HOLLOW_GEN_RETRIES = 50
HOLLOW_MIN_AREA = 10

# Boundary-notch construction
NOTCH_BASE_H_RANGE = (6, 10)
NOTCH_BASE_W_RANGE = (6, 10)
DEEP_NOTCH_DEPTH_RANGE = (3, 5)
SHALLOW_NOTCH_DEPTH_RANGE = (1, 2)
NOTCH_GEN_RETRIES = 40
NOTCH_COMPONENT_MIN_AREA = 12

# Placement
CHEBYSHEV_MIN = 3
BORDER_MIN = 1
PLACEMENT_RETRIES = 300

# Top-level generation
GEN_OUTER_RETRIES = 300


def _crop_mask(mask: np.ndarray) -> np.ndarray | None:
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return None
    return mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]


def _feature_counts(mask: np.ndarray) -> tuple[int, int]:
    enclosed = len(_interior_holes_in_sub(mask))
    deep = len(_deep_notch_regions_in_sub(mask))
    return enclosed, deep


def _has_special_region(mask: np.ndarray) -> bool:
    enclosed, deep = _feature_counts(mask)
    return enclosed > 0 or deep > 0


def _rect_dims(area: int) -> tuple[int, int] | None:
    factors = [
        (i, area // i)
        for i in range(1, int(area**0.5) + 1)
        if area % i == 0 and i <= HOLE_DIM_MAX and area // i <= HOLE_DIM_MAX
    ]
    if not factors:
        return None
    h, w = random.choice(factors)
    if random.random() < 0.5:
        h, w = w, h
    return h, w


def _chebyshev_conflict(rows_a: np.ndarray, cols_a: np.ndarray, pr: int, pc: int, pm: np.ndarray) -> bool:
    rows_b, cols_b = np.where(pm)
    rows_b, cols_b = rows_b + pr, cols_b + pc
    for ra, ca in zip(rows_a, cols_a):
        if np.min(np.maximum(np.abs(rows_b - ra), np.abs(cols_b - ca))) < CHEBYSHEV_MIN:
            return True
    return False


def _make_polyomino() -> np.ndarray | None:
    num_rects = random.randint(*POLY_NUM_RECTS_RANGE)
    rects = [
        (random.randint(*POLY_RECT_DIM_RANGE), random.randint(*POLY_RECT_DIM_RANGE))
        for _ in range(num_rects)
    ]

    canvas_h = sum(r[0] for r in rects) + POLY_CANVAS_PAD
    canvas_w = sum(r[1] for r in rects) + POLY_CANVAS_PAD
    mask = np.zeros((canvas_h, canvas_w), dtype=bool)
    r0, c0 = 2, 2
    mask[r0 : r0 + rects[0][0], c0 : c0 + rects[0][1]] = True

    for i in range(1, num_rects):
        rh, rw = rects[i]
        for _ in range(POLY_PLACEMENT_RETRIES):
            nr = random.randint(max(0, r0 - rh + 1), r0 + rects[0][0] + 2)
            nc = random.randint(max(0, c0 - rw + 1), c0 + rects[0][1] + 2)
            if nr + rh > canvas_h or nc + rw > canvas_w or nr < 0 or nc < 0:
                continue
            if mask[nr : nr + rh, nc : nc + rw].any():
                mask[nr : nr + rh, nc : nc + rw] = True
                break

    mask = _crop_mask(mask)
    if mask is None or not is_contiguous(mask.astype(int), background=0, connectivity=4):
        return None
    if _has_special_region(mask):
        return None
    return mask


def _extend_mask(mask: np.ndarray, base_h: int, base_w: int) -> np.ndarray:
    if random.random() >= HOLLOW_EXTENSION_PROB:
        return mask

    side = random.choice(["top", "bottom", "left", "right"])
    ext_h = random.randint(*HOLLOW_EXT_H_RANGE)

    if side in ("top", "bottom"):
        ext_w = min(random.randint(2, max(2, base_w)), base_w)
        offset = random.randint(0, base_w - ext_w)
        new_mask = np.zeros((base_h + ext_h, base_w), dtype=bool)
        if side == "top":
            new_mask[ext_h:, :] = mask
            new_mask[0:ext_h, offset : offset + ext_w] = True
        else:
            new_mask[:base_h, :] = mask
            new_mask[base_h : base_h + ext_h, offset : offset + ext_w] = True
    else:
        ext_h_actual = min(ext_h, base_h)
        ext_w_actual = random.randint(*HOLLOW_EXT_SIDE_W_RANGE)
        offset = random.randint(0, base_h - ext_h_actual)
        new_mask = np.zeros((base_h, base_w + ext_w_actual), dtype=bool)
        if side == "left":
            new_mask[:, ext_w_actual:] = mask
            new_mask[offset : offset + ext_h_actual, 0:ext_w_actual] = True
        else:
            new_mask[:, :base_w] = mask
            new_mask[offset : offset + ext_h_actual, base_w : base_w + ext_w_actual] = True

    return new_mask


def _carve_hole(mask: np.ndarray, hh: int, hw: int) -> np.ndarray | None:
    mh, mw = mask.shape
    for _ in range(HOLLOW_HOLE_SEARCH_RETRIES):
        hr = random.randint(1, mh - hh - 1)
        hc = random.randint(1, mw - hw - 1)
        if not mask[hr : hr + hh, hc : hc + hw].all():
            continue
        if (
            not mask[hr - 1, hc : hc + hw].any()
            or not mask[hr + hh, hc : hc + hw].any()
            or not mask[hr : hr + hh, hc - 1].any()
            or not mask[hr : hr + hh, hc + hw].any()
        ):
            continue

        carved = mask.copy()
        carved[hr : hr + hh, hc : hc + hw] = False
        if not is_contiguous(carved.astype(int), background=0, connectivity=4):
            continue
        cropped = _crop_mask(carved)
        if cropped is None:
            continue
        enclosed, deep = _feature_counts(cropped)
        if enclosed == 1 and deep == 0:
            return cropped
    return None


def _make_hollow_component(target_hole_area: int) -> np.ndarray | None:
    dims = _rect_dims(target_hole_area)
    if dims is None:
        return None
    hh, hw = dims

    pad_top = random.randint(*HOLLOW_PAD_RANGE)
    pad_bot = random.randint(*HOLLOW_PAD_RANGE)
    pad_left = random.randint(*HOLLOW_PAD_RANGE)
    pad_right = random.randint(*HOLLOW_PAD_RANGE)

    base_h = hh + pad_top + pad_bot
    base_w = hw + pad_left + pad_right
    mask = np.ones((base_h, base_w), dtype=bool)

    num_notches = random.randint(0, HOLLOW_MAX_NOTCHES)
    for _ in range(num_notches):
        corner = random.choice(["tl", "tr", "bl", "br"])
        nh = random.randint(
            1,
            max(1, pad_top - 1) if corner[0] == "t" else max(1, pad_bot - 1),
        )
        nw = random.randint(
            1,
            max(1, pad_left - 1) if corner[1] == "l" else max(1, pad_right - 1),
        )
        if corner == "tl":
            mask[0:nh, 0:nw] = False
        elif corner == "tr":
            mask[0:nh, base_w - nw : base_w] = False
        elif corner == "bl":
            mask[base_h - nh : base_h, 0:nw] = False
        else:
            mask[base_h - nh : base_h, base_w - nw : base_w] = False

    mask = _extend_mask(mask, base_h, base_w)
    return _carve_hole(mask, hh, hw)


def _carve_edge_notch(mask: np.ndarray, *, side: str, width: int, depth: int, offset: int) -> np.ndarray:
    carved = mask.copy()
    if side == "top":
        carved[0:depth, offset : offset + width] = False
    elif side == "bottom":
        carved[-depth:, offset : offset + width] = False
    elif side == "left":
        carved[offset : offset + width, 0:depth] = False
    else:
        carved[offset : offset + width, -depth:] = False
    return carved


def _make_deep_notch_component() -> np.ndarray | None:
    for _ in range(NOTCH_GEN_RETRIES):
        base_h = random.randint(*NOTCH_BASE_H_RANGE)
        base_w = random.randint(*NOTCH_BASE_W_RANGE)
        mask = np.ones((base_h, base_w), dtype=bool)
        side = random.choice(["top", "bottom", "left", "right"])

        if side in ("top", "bottom"):
            width = random.randint(1, 2)
            max_depth = min(DEEP_NOTCH_DEPTH_RANGE[1], base_h - 2)
            if max_depth < DEEP_NOTCH_DEPTH_RANGE[0] or base_w - width - 2 < 0:
                continue
            depth = random.randint(DEEP_NOTCH_DEPTH_RANGE[0], max_depth)
            offset = random.randint(1, base_w - width - 1)
        else:
            width = random.randint(1, 2)
            max_depth = min(DEEP_NOTCH_DEPTH_RANGE[1], base_w - 2)
            if max_depth < DEEP_NOTCH_DEPTH_RANGE[0] or base_h - width - 2 < 0:
                continue
            depth = random.randint(DEEP_NOTCH_DEPTH_RANGE[0], max_depth)
            offset = random.randint(1, base_h - width - 1)

        carved = _carve_edge_notch(mask, side=side, width=width, depth=depth, offset=offset)
        if not is_contiguous(carved.astype(int), background=0, connectivity=4):
            continue
        enclosed, deep = _feature_counts(carved)
        if enclosed == 0 and deep == 1 and int(carved.sum()) >= NOTCH_COMPONENT_MIN_AREA:
            return carved
    return None


def _make_shallow_notch_component() -> np.ndarray | None:
    for _ in range(NOTCH_GEN_RETRIES):
        base_h = random.randint(*NOTCH_BASE_H_RANGE)
        base_w = random.randint(*NOTCH_BASE_W_RANGE)
        mask = np.ones((base_h, base_w), dtype=bool)
        side = random.choice(["top", "bottom", "left", "right"])
        use_shallow_depth = random.random() < 0.5

        if side in ("top", "bottom"):
            if use_shallow_depth:
                width = random.randint(1, 2)
                max_depth = min(SHALLOW_NOTCH_DEPTH_RANGE[1], base_h - 2)
                if max_depth < SHALLOW_NOTCH_DEPTH_RANGE[0] or base_w - width - 2 < 0:
                    continue
                depth = random.randint(SHALLOW_NOTCH_DEPTH_RANGE[0], max_depth)
            else:
                width = 3
                max_depth = min(DEEP_NOTCH_DEPTH_RANGE[1], base_h - 2)
                if base_w - width - 2 < 0 or max_depth < DEEP_NOTCH_DEPTH_RANGE[0]:
                    continue
                depth = random.randint(DEEP_NOTCH_DEPTH_RANGE[0], max_depth)
            offset = random.randint(1, base_w - width - 1)
        else:
            if use_shallow_depth:
                width = random.randint(1, 2)
                max_depth = min(SHALLOW_NOTCH_DEPTH_RANGE[1], base_w - 2)
                if max_depth < SHALLOW_NOTCH_DEPTH_RANGE[0] or base_h - width - 2 < 0:
                    continue
                depth = random.randint(SHALLOW_NOTCH_DEPTH_RANGE[0], max_depth)
            else:
                width = 3
                max_depth = min(DEEP_NOTCH_DEPTH_RANGE[1], base_w - 2)
                if base_h - width - 2 < 0 or max_depth < DEEP_NOTCH_DEPTH_RANGE[0]:
                    continue
                depth = random.randint(DEEP_NOTCH_DEPTH_RANGE[0], max_depth)
            offset = random.randint(1, base_h - width - 1)

        carved = _carve_edge_notch(mask, side=side, width=width, depth=depth, offset=offset)
        if not is_contiguous(carved.astype(int), background=0, connectivity=4):
            continue
        enclosed, deep = _feature_counts(carved)
        if enclosed == 0 and deep == 0 and int(carved.sum()) >= NOTCH_COMPONENT_MIN_AREA:
            return carved
    return None


def _make_small_shape() -> np.ndarray:
    size = random.randint(*SMALL_SHAPE_SIZE_RANGE)
    if size == 1:
        return np.ones((1, 1), dtype=bool)
    if size == 2:
        return np.ones((1, 2), dtype=bool) if random.random() < 0.5 else np.ones((2, 1), dtype=bool)
    if size == 3:
        return random.choice(
            [
                np.array([[1, 1], [1, 0]], dtype=bool),
                np.array([[1, 0], [1, 1]], dtype=bool),
                np.array([[1, 1, 1]], dtype=bool),
                np.array([[1], [1], [1]], dtype=bool),
            ]
        )
    return np.ones((2, 2), dtype=bool)


def _make_clean_solid() -> np.ndarray | None:
    if random.random() < SMALL_SHAPE_PROB:
        mask = _make_small_shape()
        return mask if not _has_special_region(mask) else None
    for _ in range(POLY_GEN_RETRIES):
        mask = _make_polyomino()
        if mask is not None:
            return mask
    return None


def _make_component(kind: str) -> np.ndarray | None:
    if kind == "deep":
        return _make_deep_notch_component()
    if kind == "hole":
        for _ in range(HOLLOW_GEN_RETRIES):
            mask = _make_hollow_component(random.choice(HOLE_VALID_AREAS))
            if mask is not None and int(mask.sum()) >= HOLLOW_MIN_AREA:
                return mask
        return None
    if kind == "shallow":
        return _make_shallow_notch_component()
    for _ in range(POLY_GEN_RETRIES):
        mask = _make_clean_solid()
        if mask is not None:
            return mask
    return None


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)

    num_components = random.randint(*COMP_COUNT_RANGE)
    num_special = random.randint(SPECIAL_COUNT_RANGE[0], min(SPECIAL_COUNT_RANGE[1], num_components - 1))

    kinds: list[str] = ["deep"]
    while len(kinds) < num_special:
        kinds.append("hole" if random.random() < SECOND_SPECIAL_HOLE_PROB else "deep")

    remaining = num_components - num_special
    if remaining > 0:
        kinds.append("shallow")
    while len(kinds) < num_components:
        kinds.append("solid")
    random.shuffle(kinds)

    components: list[np.ndarray] = []
    for kind in kinds:
        mask = _make_component(kind)
        if mask is None:
            return None
        components.append(mask)

    grid = np.full((H, W), BG_COLOR, dtype=int)
    placed: list[tuple[int, int, np.ndarray]] = []

    order = sorted(range(num_components), key=lambda i: -int(components[i].sum()))
    for idx in order:
        mask = components[idx]
        mh, mw = mask.shape
        rows_a, cols_a = np.where(mask)

        placed_ok = False
        for _ in range(PLACEMENT_RETRIES):
            r = random.randint(BORDER_MIN, max(BORDER_MIN, H - mh - BORDER_MIN))
            c = random.randint(BORDER_MIN, max(BORDER_MIN, W - mw - BORDER_MIN))
            if r + mh > H - BORDER_MIN or c + mw > W - BORDER_MIN:
                continue

            rows_here, cols_here = rows_a + r, cols_a + c
            conflict = False
            for pr, pc, pm in placed:
                row_gap = max(0, max(r - (pr + pm.shape[0] - 1), pr - (r + mh - 1)))
                col_gap = max(0, max(c - (pc + pm.shape[1] - 1), pc - (c + mw - 1)))
                if max(row_gap, col_gap) >= CHEBYSHEV_MIN:
                    continue
                if _chebyshev_conflict(rows_here, cols_here, pr, pc, pm):
                    conflict = True
                    break

            if not conflict:
                placed.append((r, c, mask))
                placed_ok = True
                break

        if not placed_ok:
            return None

    deep_components = 0
    for r, c, mask in placed:
        rows, cols = np.where(mask)
        grid[rows + r, cols + c] = FG_COLOR
        enclosed, deep = _feature_counts(mask)
        if deep > 0:
            deep_components += 1
        if enclosed > 0 and deep > 0:
            return None

    if deep_components == 0:
        return None

    try:
        _ = main(grid)
    except Exception:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_OUTER_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_OUTER_RETRIES} attempts")
