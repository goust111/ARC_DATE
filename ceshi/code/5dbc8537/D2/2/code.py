# Variant of 5dbc8537
# Dimension: D2 — 精灵来源结构与属性 (src)
# Change: 散落在模板外接矩形外部的全填充矩形精灵，每个精灵是完整矩形（bbox内无背景色），含1-2种非背景色，部分为1x1黑色标... → 对角线标记精灵

from __future__ import annotations

import math
import random

import numpy as np

from arc_utils import bbox_coords, colors_present, components_mask, crop_bbox, get_bg, paste, random_free_location

# Design analysis:
# variant_spec:
#   a_concretization_and_trace:
#     input:
#       - 模板区域保持原题结构: 一个触碰三边的 base 色长条框架, 内部有若干矩形洞.
#       - 变化只发生在模板外部的 source: 不再摆放完整矩形, 而是每个精灵只显示两个同色像素.
#       - 这两个像素位于某个轴对齐矩形 bbox 的一条对角线上; bbox 尺寸就是要恢复的真实精灵尺寸.
#       - 为了让规则清楚可见, 本变体把填充块限制为纯色实心矩形, 且至少有一边 >= 3; 这样 marker 对不会在 8 邻接下重新变成单个连通块.
#     trace_example:
#       - 原题里, 一个 3x4 红色 piece 在模板外直接画成 3x4 实心矩形, solver 用连通分量读出 patch.
#       - 本变体里, 同一个 piece 只画成两枚红点, 例如 bbox 左上和右下角; solver 不能再依赖连通分量面积, 必须先用这两点恢复 3x4 隐含矩形.
#       - 模板内候选洞仍按矩形尺寸匹配; 差异只在 source 读取阶段.
#   b_complexity_calibration:
#     - 若允许同色出现多个 marker 对, pairing 会退化成组合匹配, 对少量样例不友好.
#     - 因此每个 source rectangle 使用唯一颜色, 使“颜色 -> 一对对角点 -> 一个隐含矩形”稳定成立.
#     - 若保留 1x1 点块, 对角标记会退化成单点; 若保留 2x2, 两个对角点又会在 8 邻接下连通. 因此改为所有矩形至少一边 >= 3.
#     - 若保留原来的双色内部纹理, 两个角点不足以恢复纹理内容, 会引入额外未说明规则; 所以输出块统一为纯色实心矩形.
#   c_implicit_assumption_audit_for_original_main:
#     - 原代码 `find_components(masked, ...)` 假设每个 source 是独立连通块; 变体下失效, 因为两个角点互不连通.
#     - 原代码 `patch = crop_bbox(g, b)` 默认 bbox 内真实像素就是 sprite 内容; 变体下 bbox 内大多是背景, 直接裁剪会得到错误 patch.
#     - 原代码 `mask` 记录 bbox 内非背景形状; 变体下若沿用它, 只会把两个点贴回洞里, 而不是恢复整个矩形.
#     - 原代码 exact-cover / hole candidate / base bbox 推断仍然成立, 因为模板与洞的几何约束没有改变.
#   d_placement_legality:
#     - 生成器对模板外 source 采用“bbox 预留 + 实际只画两点”的方式: 用完整 bbox 占位确保隐含矩形彼此不重叠, 再在该 bbox 两角放 marker.
#     - 这样 source 虽不连通, 但每个隐含 bbox 都独立、无越界、与模板保持至少 1 格背景间距.
#     - 模板内洞仍由不重叠矩形精确铺满, 且总洞面积等于所有隐含矩形面积之和.
#   e_visual_observability:
#     - 从 2-3 个样例里, 人类能明显看到: 模板外每种颜色只出现两个孤立点, 输出里则变成对应尺寸的整块矩形填回模板.
#     - 不使用颜色 id 参与排序或变换; 颜色只是对象身份与输出填色本身.
#   f_visual_cleanliness:
#     - source 的隐含 bbox 彼此分离, 不会出现多组 marker 交叠导致“哪个点配哪个点”看不清.
#     - 模板内仍是单层结构: base 框架 + 矩形洞, 没有额外覆盖层.
#   g_color_inference:
#     - 不引入隐藏颜色映射. marker 的颜色就是恢复后整块矩形的颜色.
#   solver_assumptions_added_by_variant:
#     - 模板外每个非背景且非 base 的颜色恰好出现 2 次.
#     - 这两个像素不共行也不共列, 且对应 bbox 不为 2x2, 因而 source 不会重新退化成 8 连通对象.
#     - 每种 bbox 尺寸对应的洞覆盖在 exact-cover 中是可还原的; 生成器用 oracle 校验保证这一点.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct hidden rectangles from diagonal marker pairs and fill matching cavities in the template.

    Input structure:
    - bg is the most common color. A dominant base-color strip forms the template and contains rectangular holes.
    - Outside that template there are no solid source blocks; each source is encoded by exactly two same-colored pixels at opposite corners of its bbox.

    Rule:
    - Detect the template bbox from the largest non-bg base-color component.
    - For each outside color, use its two pixels to infer the full axis-aligned rectangle spanning them.
    - Match those recovered rectangle sizes to all-hole rectangles inside the template and place solid fills into the holes.

    Key insight: The source pieces are implicit rather than connected — the diagonal marker pair determines the missing rectangle that must be pasted back into the template.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    base_candidates: list[tuple[int, int]] = []
    for col in colors_present(g, exclude={bg}):
        max_area = 0
        for coords in components_mask(g == int(col), connectivity=8):
            max_area = max(int(max_area), int(coords.shape[0]))
        base_candidates.append((int(max_area), int(col)))

    assert base_candidates
    base_candidates.sort(reverse=True)
    if len(base_candidates) > 1:
        assert base_candidates[0][0] > base_candidates[1][0]
    base_col = int(base_candidates[0][1])

    base_coords = np.argwhere(g == int(base_col))
    assert not (base_coords.size == 0)
    r0, c0, r1, c1 = bbox_coords(base_coords)
    dst_bbox = (int(r0), int(c0), int(r1), int(c1))

    dst = crop_bbox(g, dst_bbox)
    dh, dw = map(int, dst.shape)
    hole_mask = dst != int(base_col)
    out = np.full((dh, dw), int(base_col), dtype=int)

    masked = g.copy()
    masked[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = int(bg)

    sprites: list[dict[str, np.ndarray | int]] = []
    for col in colors_present(masked, exclude={bg}):
        coords = np.argwhere(masked == int(col))
        assert int(coords.shape[0]) == 2
        sr0, sc0, sr1, sc1 = bbox_coords(coords)
        h = int(sr1 - sr0 + 1)
        w = int(sc1 - sc0 + 1)
        assert h >= 2 and w >= 2 and (h > 2 or w > 2)
        patch = np.full((h, w), int(col), dtype=int)
        sprites.append({"h": h, "w": w, "patch": patch})

    sprites.sort(key=lambda s: (-int(s["h"]) * int(s["w"]), -int(s["h"]), -int(s["w"])))

    candidates: list[list[tuple[int, int]]] = []
    areas = [int(s["h"]) * int(s["w"]) for s in sprites]
    assert sum(areas) == int(hole_mask.sum())
    for sp in sprites:
        h, w = int(sp["h"]), int(sp["w"])
        pos: list[tuple[int, int]] = []
        if h <= dh and w <= dw:
            for rr0 in range(int(dh - h + 1)):
                for cc0 in range(int(dw - w + 1)):
                    if bool(hole_mask[rr0 : rr0 + h, cc0 : cc0 + w].all()):
                        pos.append((int(rr0), int(cc0)))
        candidates.append(pos)

    assert all(candidates)
    used = ~hole_mask
    placement: list[tuple[int, int] | None] = [None] * len(sprites)

    def dfs(i: int) -> bool:
        if i == len(sprites):
            return True

        h, w = int(sprites[i]["h"]), int(sprites[i]["w"])
        for rr0, cc0 in candidates[i]:
            if bool(used[rr0 : rr0 + h, cc0 : cc0 + w].any()):
                continue
            used[rr0 : rr0 + h, cc0 : cc0 + w] = True
            placement[i] = (int(rr0), int(cc0))
            if dfs(i + 1):
                return True
            placement[i] = None
            used[rr0 : rr0 + h, cc0 : cc0 + w] = False
        return False

    assert dfs(0)

    for i, pos in enumerate(placement):
        assert pos is not None
        rr0, cc0 = map(int, pos)
        paste(out, np.asarray(sprites[i]["patch"]), rr0, cc0, overwrite=True)

    return out


# --------------- generator (D2 diagonal markers) ---------------

GRID_MIN, GRID_MAX = 15, 25
THICK_MIN, THICK_MAX = 7, 10
OUTER_GAP = 8
SPRITE_COUNT_MIN, SPRITE_COUNT_MAX = 4, 6
HOLE_FRAC_LO, HOLE_FRAC_HI = 0.39, 0.50
CORRIDOR_COVERAGE = 0.78
MAX_RECT_LEN = 8
MAX_RECT_WIDTH = 6
WIDTH_TOLERANCE = 2
WIDTH_RETRIES = 300
MAX_RETRIES = 300


def _split_lengths(total: int, count: int) -> list[int] | None:
    """Partition *total* into *count* integers in [2, MAX_RECT_LEN]."""
    if count <= 0 or 2 * count > total or count * MAX_RECT_LEN < total:
        return None
    lengths = [2] * count
    remaining = total - 2 * count
    while remaining:
        choices = [i for i, v in enumerate(lengths) if v < MAX_RECT_LEN]
        if not choices:
            return None
        i = random.choice(choices)
        add = random.randint(1, min(remaining, MAX_RECT_LEN - lengths[i]))
        lengths[i] += add
        remaining -= add
    random.shuffle(lengths)
    return lengths


def _choose_widths(lengths: list[int], thick: int, full_area: int) -> list[int] | None:
    """Pick widths in [2, max_w] so areas hit the target band and (h,w) pairs stay unique."""
    max_w = min(MAX_RECT_WIDTH, thick - 2)
    if max_w < 2:
        return None
    lo = math.ceil(HOLE_FRAC_LO * full_area)
    hi = max(lo, math.floor(HOLE_FRAC_HI * full_area))
    target = random.randint(lo, hi)
    best: tuple[int, list[int]] | None = None
    opts = list(range(2, max_w + 1))

    for _ in range(WIDTH_RETRIES):
        widths = [random.choice(opts) for _ in lengths]
        if len(set(zip(lengths, widths))) != len(lengths):
            continue
        if any(h == 2 and w == 2 for h, w in zip(lengths, widths)):
            continue
        area = sum(h * w for h, w in zip(lengths, widths))
        if not (lo <= area <= hi):
            continue
        score = abs(area - target)
        if best is None or score < best[0]:
            best = (score, widths[:])
            if score <= WIDTH_TOLERANCE:
                break
    return None if best is None else best[1]


def _corridor_rects(long: int, thick: int, count: int) -> list[tuple[int, int, int, int]] | None:
    """Generate connected stacked rectangles along the long axis."""
    min_len = max(2 * count + 1, round(CORRIDOR_COVERAGE * long))
    if min_len > long - 1:
        min_len = long - 1
    if min_len < 2 * count:
        return None
    hole_len = random.randint(min_len, long - 1)
    lengths = _split_lengths(hole_len, count)
    if lengths is None:
        return None
    widths = _choose_widths(lengths, thick, long * thick)
    if widths is None:
        return None

    start = random.randint(0, long - hole_len)
    r = start
    prev_c = prev_w = None
    rects: list[tuple[int, int, int, int]] = []
    for h, w in zip(lengths, widths):
        c_hi = thick - w - 1
        if c_hi < 1:
            return None
        if prev_c is None:
            c = random.randint(1, c_hi)
        else:
            c_lo = max(1, prev_c - w + 1)
            c_hi = min(c_hi, prev_c + prev_w - 1)
            if c_lo > c_hi:
                return None
            c = random.randint(c_lo, c_hi)
        rects.append((r, c, h, w))
        r += h
        prev_c, prev_w = c, w

    if len({(h, w) for _, _, h, w in rects}) != len(rects):
        return None
    if any(h == 2 and w == 2 for _, _, h, w in rects):
        return None
    return rects


def _make_pieces(rects: list[tuple[int, int, int, int]], bg: int, base: int) -> dict[int, np.ndarray] | None:
    """Assign a unique color to each rectangle and build solid patches."""
    available = [c for c in range(10) if c not in (bg, base)]
    if len(available) < len(rects):
        return None
    chosen = random.sample(available, len(rects))
    pieces: dict[int, np.ndarray] = {}
    for i, (_, _, h, w) in enumerate(rects):
        pieces[i] = np.full((h, w), int(chosen[i]), dtype=int)
    return pieces


def _render_output(dh: int, dw: int, base: int,
                   rects: list[tuple[int, int, int, int]],
                   pieces: dict[int, np.ndarray]) -> np.ndarray:
    out = np.full((dh, dw), base, dtype=int)
    for i, (r, c, _, _,) in enumerate(rects):
        paste(out, pieces[i], r, c)
    return out


def _marker_patch(h: int, w: int, color: int, bg: int) -> np.ndarray:
    """Create a sparse source patch with a same-colored diagonal marker pair."""
    patch = np.full((h, w), bg, dtype=int)
    if random.random() < 0.5:
        patch[0, 0] = color
        patch[h - 1, w - 1] = color
    else:
        patch[0, w - 1] = color
        patch[h - 1, 0] = color
    return patch


def _scatter_markers(H: int, W: int, bg: int, base: int,
                     tr0: int, tc0: int, dh: int, dw: int,
                     rects: list[tuple[int, int, int, int]],
                     pieces: dict[int, np.ndarray]) -> np.ndarray:
    """Build the input grid: template frame with holes + scattered diagonal marker pairs outside."""
    inp = np.full((H, W), bg, dtype=int)
    inp[tr0 : tr0 + dh, tc0 : tc0 + dw] = base
    for r, c, h, w in rects:
        inp[tr0 + r : tr0 + r + h, tc0 + c : tc0 + c + w] = bg

    placement = np.full((H, W), bg, dtype=int)
    placement[tr0 : tr0 + dh, tc0 : tc0 + dw] = base
    order = list(range(len(rects)))
    random.shuffle(order)
    order.sort(key=lambda i: pieces[i].size, reverse=True)
    for i in order:
        h, w = pieces[i].shape
        reserve = np.full((h, w), base, dtype=int)
        rr, cc = random_free_location(placement, reserve, transparent=bg, padding=1, padding_connectivity=8)
        color = int(pieces[i][0, 0])
        marker = _marker_patch(h, w, color, bg)
        paste(inp, marker, rr, cc, transparent=bg)
        paste(placement, reserve, rr, cc)
    return inp


def _gen_attempt() -> np.ndarray | None:
    vertical = random.random() < 0.5
    thick = random.randint(THICK_MIN, THICK_MAX)
    if vertical:
        H = random.randint(GRID_MIN, GRID_MAX)
        W = random.randint(max(GRID_MIN, thick + OUTER_GAP), GRID_MAX)
        dh, dw = H, thick
        tr0, tc0 = 0, (0 if random.random() < 0.5 else W - thick)
    else:
        H = random.randint(max(GRID_MIN, thick + OUTER_GAP), GRID_MAX)
        W = random.randint(GRID_MIN, GRID_MAX)
        dh, dw = thick, W
        tr0, tc0 = (0 if random.random() < 0.5 else H - thick), 0
    long = dh if vertical else dw

    max_count = min(SPRITE_COUNT_MAX, (long - 1) // 2)
    if max_count < SPRITE_COUNT_MIN:
        return None
    total = random.randint(SPRITE_COUNT_MIN, max_count)

    canon = _corridor_rects(long, thick, total)
    if canon is None:
        return None
    rects = canon if vertical else [(c, r, w, h) for r, c, h, w in canon]

    bg, base = random.sample(range(10), 2)
    pieces = _make_pieces(rects, bg, base)
    if pieces is None:
        return None

    out = _render_output(dh, dw, base, rects, pieces)
    try:
        inp = _scatter_markers(H, W, bg, base, tr0, tc0, dh, dw, rects, pieces)
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid 5dbc8537 D2 input")
