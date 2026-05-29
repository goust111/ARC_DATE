# Combo variant of 5dbc8537 (index 4)
# combo_spec: D2/2+D4/1
# Name: 对角线标记精灵旋转填入

from __future__ import annotations

import math
import random

import numpy as np

from arc_utils import bbox_coords, colors_present, components_mask, crop_bbox, get_bg, paste, random_free_location

# Design analysis:
# combo_variant:
#   a_concretization_and_trace:
#     input:
#       - 仍然有一个触碰三边的 base 色长条模板, 模板内的孔区仍由若干轴对齐矩形精确铺满.
#       - 外部 source 不再是实心矩形, 而是“每种颜色恰好两枚像素”的对角线标记; 这两点位于该矩形 bbox 的一条对角线上.
#       - 由于 D4/1 叠加了旋转, 外部 marker 的 bbox 对应的是“填回前”的矩形朝向: 若目标洞位需要 90/270 度旋转, 则外部 bbox 的高宽与洞高宽交换.
#     trace_example:
#       - 例: 模板上三分之一区域有一个 2x5 洞. 输出里应填入一块 2x5 的纯色矩形.
#       - 原题会在模板外直接给一个 2x5 实心块; 这里只给一个同色的 5x2 稀疏 patch, 其中只有两个对角角点着色.
#       - solver 先从这两个点恢复出隐含的 5x2 矩形, 再依据“上段 -> rot90(k=1)”把它转成 2x5, 最后才能与该洞匹配.
#       - 差异不只是 source 编码变了, 也不只是 paste 前多转一下; 必须先恢复隐含尺寸, 再让尺寸在旋转后对上洞位.
#   b_complexity_calibration:
#     - 若允许一个颜色对应多组 marker, pairing 会退化成组合搜索; 因此每个 piece 使用唯一颜色, 让“颜色 -> 一对点 -> 一个矩形”稳定可见.
#     - 若允许 1x1 或 2x2, 对角标记会退化为单点或 8 邻接连通块; 因此所有矩形都要求两维 >= 2 且至少一维 >= 3.
#     - 为了让旋转真正起作用, 生成器避免正方形, 并强制样例至少包含一个上/左段 piece 和一个下/右段 piece; 这两类都会发生高宽交换.
#     - 中段继续保留 180 度分支, 但不把可观察性完全寄托在它身上, 以免“纯色矩形做 180 度看不出变化”削弱题意.
#   c_implicit_assumption_audit_for_original_main:
#     - 原代码 `find_components(...)` 假设每个 source 是独立连通块; 融合后失效, 因为每个 source 只有两枚互不连通的角点.
#     - 原代码 `crop_bbox(g, b)` 默认 bbox 内像素就是完整 patch; 融合后 bbox 内大多是背景, 只能把 bbox 当作隐含矩形尺寸而不是现成纹理.
#     - 原代码候选洞只按 source 的原始 `(h, w)` 枚举; 融合后必须对每个 source 额外枚举 `90/180/270` 旋转, 因为 90/270 会交换尺寸.
#     - 原代码最终直接 paste 原 patch; 融合后必须 paste 旋转后的隐含矩形.
#   d_placement_legality:
#     - 模板内孔区仍由不重叠矩形组成, 并通过长走廊式构造保持整体连通.
#     - 模板外散布 marker 时, 先用完整 source bbox 作为占位体做 `random_free_location`, 再只在该 bbox 的两角画点; 这样隐含 bbox 彼此不接触也不越界.
#     - 生成器额外拒绝“洞尺寸唯一但逆旋转后的 source 尺寸重复”的情况, 避免 exact-cover 候选过度对称.
#   e_visual_observability:
#     - 人类能直接看到两件事: 外部每种颜色只有两个角点, 说明需要先补全矩形; 同时许多外部 bbox 的长宽与对应洞位长宽互换, 说明还需要旋转.
#     - 旋转规则只依赖洞位沿条带长轴的三分区位置, 不是颜色 id、扫描顺序或黑色特权等不可视因素.
#   f_visual_cleanliness:
#     - source 区域只有稀疏 marker, 模板区域只有 base 框架和矩形洞, 没有叠层符号或额外装饰.
#     - 唯一颜色约束让每对 marker 的归属清楚; bbox 预留和 1 格间距避免不同 source 在视觉上纠缠.
#   g_color_inference:
#     - 不引入隐藏颜色映射. marker 的颜色就是补全后矩形的填充色.
#     - 颜色只承担“对象身份 + 输出填色”两件事, 不承担旋转类别编码, 使可推断性集中在几何关系上.
#   solver_assumptions_added_by_variant:
#     - 模板外每个非背景颜色恰好出现两次, 且两点不共行不共列.
#     - 每个恢复出的隐含矩形都不是 2x2, 并且所有 piece 都只允许使用 90/180/270 三种旋转之一.
#     - 至少存在一个 rot=1 和一个 rot=3 的 piece, 因而“先恢复矩形再旋转”不是可有可无的伪规则.


def _rotation_for_rect(rr0: int, cc0: int, h: int, w: int, dh: int, dw: int) -> int:
    """Required `np.rot90(..., k)` count for a hole rectangle at this location."""
    if dh >= dw:
        coord = float(rr0) + float(h - 1) / 2.0
        span = float(dh)
    else:
        coord = float(cc0) + float(w - 1) / 2.0
        span = float(dw)
    cut1 = span / 3.0
    cut2 = 2.0 * span / 3.0
    if coord < cut1:
        return 1
    if coord >= cut2:
        return 3
    return 2


def _zone_index(rr0: int, cc0: int, h: int, w: int, dh: int, dw: int) -> int:
    return _rotation_for_rect(rr0, cc0, h, w, dh, dw) - 1


def _source_shape(h: int, w: int, rot: int) -> tuple[int, int]:
    """Shape of the inverse-rotated source rectangle that will be encoded by marker pairs."""
    return (int(w), int(h)) if int(rot) % 2 == 1 else (int(h), int(w))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover hidden rectangles from diagonal marker pairs, rotate them by hole position, and rebuild the strip template.

    Input structure:
    - The most common color is the background; one large base-colored strip forms the template and contains a connected region of rectangular holes.
    - Outside the template, each source sprite is implicit: exactly two same-colored pixels mark opposite corners of the source rectangle's bbox rather than showing the full block.

    Rule:
    - Locate the template from the largest non-background connected component color, then crop to its bbox.
    - For each outside color, infer the full solid rectangle spanning its two markers.
    - Match those recovered rectangles to hole subrectangles, but only after applying the location-based rotation rule: first third of the strip's long axis -> 90 degrees CCW, middle third -> 180 degrees, last third -> 270 degrees.

    Key insight: The source geometry is encoded indirectly, so the solver must first reconstruct each missing rectangle from its diagonal pair and only then test which rotated version can legally cover a hole rectangle.
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

    sprites: list[dict[str, object]] = []
    for col in colors_present(masked, exclude={bg}):
        coords = np.argwhere(masked == int(col))
        assert int(coords.shape[0]) == 2
        sr0, sc0, sr1, sc1 = bbox_coords(coords)
        h = int(sr1 - sr0 + 1)
        w = int(sc1 - sc0 + 1)
        assert h >= 2 and w >= 2 and (h > 2 or w > 2)
        patch = np.full((h, w), int(col), dtype=int)
        rots = {k: np.rot90(patch, int(k)) for k in (1, 2, 3)}
        sprites.append({"area": int(h * w), "rots": rots})

    areas = [int(sp["area"]) for sp in sprites]
    assert sum(areas) == int(hole_mask.sum())

    candidates: list[list[tuple[int, int, int, int, int]]] = []
    for sp in sprites:
        pos: list[tuple[int, int, int, int, int]] = []
        rots = sp["rots"]
        assert isinstance(rots, dict)
        for rot in (1, 2, 3):
            patch = np.asarray(rots[rot])
            h, w = map(int, patch.shape)
            if h > dh or w > dw:
                continue
            for rr0 in range(int(dh - h + 1)):
                for cc0 in range(int(dw - w + 1)):
                    if _rotation_for_rect(rr0, cc0, h, w, dh, dw) != int(rot):
                        continue
                    if bool(hole_mask[rr0 : rr0 + h, cc0 : cc0 + w].all()):
                        pos.append((int(rr0), int(cc0), int(rot), int(h), int(w)))
        candidates.append(pos)

    assert all(candidates)

    order = sorted(range(len(sprites)), key=lambda i: (len(candidates[i]), -int(sprites[i]["area"])))
    sprites = [sprites[i] for i in order]
    candidates = [candidates[i] for i in order]

    used = ~hole_mask
    placement: list[tuple[int, int, int, int, int] | None] = [None] * len(sprites)

    def dfs(i: int) -> bool:
        if i == len(sprites):
            return True

        for rr0, cc0, rot, h, w in candidates[i]:
            if bool(used[rr0 : rr0 + h, cc0 : cc0 + w].any()):
                continue
            used[rr0 : rr0 + h, cc0 : cc0 + w] = True
            placement[i] = (int(rr0), int(cc0), int(rot), int(h), int(w))
            if dfs(i + 1):
                return True
            placement[i] = None
            used[rr0 : rr0 + h, cc0 : cc0 + w] = False
        return False

    assert dfs(0)

    for i, pos in enumerate(placement):
        assert pos is not None
        rr0, cc0, rot, _h, _w = map(int, pos)
        rots = sprites[i]["rots"]
        assert isinstance(rots, dict)
        paste(out, np.asarray(rots[rot]), rr0, cc0, overwrite=True)

    return out


# --------------- generator (combo D2/2 + D4/1) ---------------

GRID_MIN, GRID_MAX = 15, 25
THICK_MIN, THICK_MAX = 7, 10
OUTER_GAP = 8
SPRITE_COUNT_MIN, SPRITE_COUNT_MAX = 5, 7
HOLE_FRAC_LO, HOLE_FRAC_HI = 0.39, 0.50
CORRIDOR_COVERAGE = 0.78
MAX_RECT_LEN = 8
MAX_RECT_WIDTH = 6
WIDTH_TOLERANCE = 2
WIDTH_RETRIES = 400
MAX_RETRIES = 500


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
    """Pick widths in [2, max_w] so areas hit the target band and output shapes stay unique/non-square."""
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
        shapes = list(zip(lengths, widths))
        if len(set(shapes)) != len(shapes):
            continue
        if any(h == w for h, w in shapes):
            continue
        if any(h == 2 and w == 2 for h, w in shapes):
            continue
        area = sum(h * w for h, w in shapes)
        if not (lo <= area <= hi):
            continue
        score = abs(area - target)
        if best is None or score < best[0]:
            best = (score, widths[:])
            if score <= WIDTH_TOLERANCE:
                break
    return None if best is None else best[1]


def _corridor_rects(long: int, thick: int, count: int) -> list[tuple[int, int, int, int]] | None:
    """Generate connected stacked rectangles along the strip's long axis."""
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

    shapes = {(h, w) for _, _, h, w in rects}
    if len(shapes) != len(rects):
        return None
    if any(h == w or (h == 2 and w == 2) for _, _, h, w in rects):
        return None
    return rects


def _make_pieces(rects: list[tuple[int, int, int, int]], bg: int, base: int) -> dict[int, np.ndarray] | None:
    """Assign a unique color to each output rectangle and build solid patches."""
    available = [c for c in range(10) if c not in (bg, base)]
    if len(available) < len(rects):
        return None
    chosen = random.sample(available, len(rects))
    pieces: dict[int, np.ndarray] = {}
    for i, (_, _, h, w) in enumerate(rects):
        pieces[i] = np.full((h, w), int(chosen[i]), dtype=int)
    return pieces


def _render_output(
    dh: int,
    dw: int,
    base: int,
    rects: list[tuple[int, int, int, int]],
    pieces: dict[int, np.ndarray],
) -> np.ndarray:
    out = np.full((dh, dw), base, dtype=int)
    for i, (r, c, _, _) in enumerate(rects):
        paste(out, pieces[i], r, c)
    return out


def _marker_patch(h: int, w: int, color: int, bg: int) -> np.ndarray:
    """Create a sparse source patch containing only a same-colored diagonal marker pair."""
    patch = np.full((h, w), bg, dtype=int)
    if random.random() < 0.5:
        patch[0, 0] = color
        patch[h - 1, w - 1] = color
    else:
        patch[0, w - 1] = color
        patch[h - 1, 0] = color
    return patch


def _scatter_markers(
    H: int,
    W: int,
    bg: int,
    base: int,
    tr0: int,
    tc0: int,
    dh: int,
    dw: int,
    rects: list[tuple[int, int, int, int]],
    pieces: dict[int, np.ndarray],
) -> np.ndarray:
    """Build the input grid: template frame with holes + inverse-rotated diagonal marker sources."""
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
        r, c, h, w = rects[i]
        rot = _rotation_for_rect(r, c, h, w, dh, dw)
        sh, sw = _source_shape(h, w, rot)
        reserve = np.full((sh, sw), base, dtype=int)
        rr, cc = random_free_location(placement, reserve, transparent=bg, padding=1, padding_connectivity=8)
        color = int(pieces[i][0, 0])
        marker = _marker_patch(sh, sw, color, bg)
        paste(inp, marker, rr, cc, transparent=bg)
        paste(placement, reserve, rr, cc)
    return inp


def _zone_coverage_ok(rects: list[tuple[int, int, int, int]], dh: int, dw: int) -> bool:
    seen: set[int] = set()
    for r, c, h, w in rects:
        seen.add(_zone_index(r, c, h, w, dh, dw))
    return seen == {0, 1, 2}


def _rotation_visibility_ok(rects: list[tuple[int, int, int, int]], dh: int, dw: int) -> bool:
    """Require at least one top/left and one bottom/right piece, both causing a shape swap."""
    seen_swap_rots: set[int] = set()
    source_shapes: list[tuple[int, int]] = []
    for r, c, h, w in rects:
        rot = _rotation_for_rect(r, c, h, w, dh, dw)
        sh, sw = _source_shape(h, w, rot)
        source_shapes.append((sh, sw))
        if rot in (1, 3) and (sh, sw) != (h, w):
            seen_swap_rots.add(rot)
    return seen_swap_rots == {1, 3} and len(set(source_shapes)) == len(source_shapes)


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

    max_count = min(SPRITE_COUNT_MAX, (long - 1) // 2, 8)
    if max_count < SPRITE_COUNT_MIN:
        return None
    total = random.randint(SPRITE_COUNT_MIN, max_count)

    canon = _corridor_rects(long, thick, total)
    if canon is None:
        return None
    rects = canon if vertical else [(c, r, w, h) for r, c, h, w in canon]
    if not _zone_coverage_ok(rects, dh, dw):
        return None
    if not _rotation_visibility_ok(rects, dh, dw):
        return None

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
    raise RuntimeError("failed to generate valid 5dbc8537 combo input")
