# Variant of 5dbc8537
# Dimension: D4 — 填充变换方式 (transform)
# Change: 将精灵原样粘贴到匹配的洞中（精灵的背景色像素被基底色替换），输出为紧裁后的模板外接矩形... → 旋转后填入

from __future__ import annotations

import math
import random

import numpy as np

from arc_utils import (
    bbox_coords,
    colors_present,
    components_mask,
    crop_bbox,
    find_components,
    get_bg,
    paste,
    random_free_location,
)

# Design analysis (YAML-style notes used to drive this variant implementation)
# design:
#   variant_core:
#     original_behavior:
#       - 外部精灵的 bbox 与模板孔区中的某个矩形洞 bbox 直接同尺寸匹配。
#       - solver 只需要做矩形精确覆盖，再把原 patch 原样贴回去。
#     new_behavior:
#       - 仍然是同一块条带模板、同一块连通孔区、同样的矩形精确覆盖。
#       - 变化只落在“填回去之前的空间变换”：矩形洞的位置沿模板长轴分成三段，
#         上/左段对应 90° 逆时针，中段对应 180°，下/右段对应 270°。
#       - 生成器先构造最终输出中的洞内 patch，再把每个 patch 逆旋转后散落到模板外，
#         因此输入中的精灵方向与输出中的方向系统性不同。
#   concrete_trace:
#     - 例：竖直条带中，某个 2x3 洞位于上三分之一。输出里该洞放的是朝“左上角钩子”方向的 2x3 图案。
#     - 原题做法会把外部 2x3 精灵直接贴进去；本变体则把输入精灵先做逆旋转，外部看到的是 3x2 版本。
#     - solver 在 DFS 里枚举候选矩形时，只有当“该候选位置所属分区要求的角度 == 当前尝试的旋转角”时，
#       才允许这个 sprite 占据该矩形。
#   complexity_calibration:
#     - 只使用 3 个离散角度 90/180/270，避免过多自由度。
#     - 每个样例要求非点状矩形在三段里都至少出现一次，让人能从少量样例看到重复支撑。
#     - 非点状 patch 统一采用双颜色、带固定朝向的角标 L 形图案，避免纯色或旋转对称导致“转了也看不出来”。
#   implicit_assumptions_review:
#     - 原 solver 的 `sum(h*w) == hole_area` 假设了精灵 bbox 全填满；该任务原 spec 就是如此，本变体继续保持。
#     - 原 solver 的候选列表只按 `(h,w)` 枚举；变体下这个前提失效，因为 90/270 会交换维度，
#       所以候选必须升级为 `(位置, 旋转角, 旋转后尺寸)`。
#     - 原 solver 最终 `paste(out, patch, ...)` 默认 patch 不变；变体下必须贴 `rot90(patch, k)`。
#     - 原题里即使 patch 内部旋转对称也无害；变体下这会让 D4 不可观察，因此生成器显式避免这种退化。
#   placement_legality:
#     - 模板内孔区的构造仍沿用“长走廊矩形串接 + 点标记”方式，保证孔区连通且矩形 exact cover 可解。
#     - 外部散布时使用逆旋转后的 patch，并继续通过 `random_free_location(..., padding=1)` 保证互不接触。
#     - solver 与 generator 共用同一分区旋转函数，避免规则漂移。
#   visual_observability:
#     - 规则不依赖颜色 id、排序或隐藏元数据，只依赖肉眼可见的洞位相对条带中心的区段。
#     - 角度分支只有 3 类，并且在每个样例里重复出现；输入和输出都能直接看到 L 形角标在转向。
#   visual_cleanliness:
#     - patch 图案采用简洁的角标 L 形，不引入额外叠层或符号。
#     - 保留原题的条带模板和外部散落布局，整体仍然整洁。
#   color_inference:
#     - 本变体不引入颜色映射；颜色只用于区分精灵。
#     - 非点状精灵统一双色，颜色数量保持适中，避免“颜色规则”喧宾夺主。
#   generator_guarantees_for_new_main:
#     - 每个非点状 sprite 的最终 patch 对 90/180/270 都不自同构，因此任意非零旋转都可观察。
#     - 至少一个上/左段、一个中段、一个下/右段非点状洞，支撑三种旋转分支。
#     - source sprite 始终是目标 patch 的逆旋转版本，所以新 solver 的旋转逻辑是必要的，不是可有可无。


def _rotation_for_rect(rr0: int, cc0: int, h: int, w: int, dh: int, dw: int) -> int:
    """Required `np.rot90(..., k)` quarter-turn count for a hole at this location."""
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild a strip template by rotating the scattered rectangles before they fill the holes.

    Input structure:
    - The most common color is the background; one large base-colored strip acts as the template and contains a connected region of rectangular cavities.
    - Several colored rectangular sprites lie outside that strip, but many of them only match a cavity after a non-zero rotation.

    Rule:
    - Use the largest non-background connected component color to locate the base strip and exact-cover its cavity region with the outside sprite rectangles.
    - The cavity position along the strip's long axis dictates the rotation: first third -> 90 degrees CCW, middle third -> 180 degrees, last third -> 270 degrees.
    - Rotate each matched sprite by that location-based angle, place it into the chosen cavity rectangle, and return the tight template bbox.

    Key insight: The puzzle is still a rectangular exact-cover task, but each candidate rectangle now carries a mandatory rotation label derived from where that rectangle sits inside the template.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    # --- pick base color by largest CC size (excluding bg) ---
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

    # --- destination bbox = bbox of all base-color pixels ---
    base_coords = np.argwhere(g == int(base_col))
    assert not (base_coords.size == 0)
    r0, c0, r1, c1 = bbox_coords(base_coords)
    dst_bbox = (int(r0), int(c0), int(r1), int(c1))

    dst = crop_bbox(g, dst_bbox)
    dh, dw = map(int, dst.shape)
    hole_mask = dst != int(base_col)

    out = np.full((dh, dw), int(base_col), dtype=int)

    # --- source sprites = connected components of non-bg pixels outside dst_bbox ---
    masked = g.copy()
    masked[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = int(bg)
    sprite_comps = find_components(masked, bg=bg, connectivity=8, monochromatic=False, return_format="dict")

    sprites: list[dict] = []
    for comp in sprite_comps:
        b = tuple(map(int, comp["bbox"]))
        pr0, pc0, pr1, pc1 = b
        patch = crop_bbox(g, b)
        ph, pw = map(int, patch.shape)
        m = np.zeros((ph, pw), dtype=bool)
        coords = comp["coords"]
        m[coords[:, 0] - pr0, coords[:, 1] - pc0] = True
        rots = {k: np.rot90(patch, int(k)) for k in (1, 2, 3)}
        rot_masks = {k: np.rot90(m, int(k)) for k in (1, 2, 3)}
        sprites.append(
            {
                "h": int(ph),
                "w": int(pw),
                "area": int(ph * pw),
                "patch": patch,
                "mask": m,
                "rots": rots,
                "rot_masks": rot_masks,
            }
        )

    candidates: list[list[tuple[int, int, int, int, int]]] = []
    areas = [int(s["area"]) for s in sprites]
    assert sum(areas) == int(hole_mask.sum())
    for sp in sprites:
        pos: list[tuple[int, int, int, int, int]] = []
        for rot in (1, 2, 3):
            patch = sp["rots"][rot]
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

    used = ~hole_mask  # base pixels are treated as already occupied
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

    # --- paste according to the exact cover placement, using the required rotation ---
    for i, pos in enumerate(placement):
        assert pos is not None
        rr0, cc0, rot, _h, _w = map(int, pos)
        sp = sprites[i]
        paste(out, sp["rots"][rot], rr0, cc0, mask=sp["rot_masks"][rot], overwrite=True)

    return out


# --------------- generator (att2, D4 rotation variant) ---------------

# ── Configuration ────────────────────────────────────────────────────────
GRID_MIN, GRID_MAX = 15, 25
THICK_MIN, THICK_MAX = 7, 10
OUTER_GAP = 8
SPRITE_COUNT_MIN, SPRITE_COUNT_MAX = 7, 15
DOT_MIN, DOT_MAX = 1, 2
DOT_NON_DOT_BUFFER = 6
DOT_LONG_BUFFER = 4
HOLE_FRAC_LO, HOLE_FRAC_HI = 0.39, 0.50
CORRIDOR_COVERAGE = 0.78
MAX_RECT_LEN = 8
MAX_RECT_WIDTH = 6
MIN_SHAPE_GROUPS = 3
MIN_PALETTE_SIZE = 5
WIDTH_TOLERANCE = 2
MAX_RETRIES = 400


def _split_lengths(total: int, count: int) -> list[int] | None:
    """Partition *total* into *count* positive integers, each <= MAX_RECT_LEN."""
    if count <= 0 or count > total or count * MAX_RECT_LEN < total:
        return None
    lengths = [1] * count
    remaining = total - count
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


def _choose_widths(lengths: list[int], thick: int, dots: int, full_area: int) -> list[int] | None:
    """Pick a width for each length so total rect area falls within hole-fraction bounds."""
    max_w = min(MAX_RECT_WIDTH, thick - 2)
    lo = max(0, math.ceil(HOLE_FRAC_LO * full_area) - dots)
    hi = max(lo, math.floor(HOLE_FRAC_HI * full_area) - dots)
    paths: dict[int, list[int]] = {0: []}
    for h in lengths:
        opts = list(range(1, max_w + 1))
        if h == 1 and max_w >= 2:
            opts = list(range(2, max_w + 1))
        random.shuffle(opts)
        items = list(paths.items())
        random.shuffle(items)
        nxt: dict[int, list[int]] = {}
        for cur, path in items:
            for w in opts:
                s = cur + h * w
                if s <= hi and s not in nxt:
                    nxt[s] = path + [w]
        paths = nxt
        if not paths:
            return None
    valid = [(s, p) for s, p in paths.items() if lo <= s <= hi]
    if not valid:
        return None
    target = random.randint(lo, hi)
    best = min(abs(s - target) for s, _ in valid)
    near = [p for s, p in valid if abs(s - target) <= best + WIDTH_TOLERANCE]
    return random.choice(near)


def _place_dot_markers(
    rects: list[tuple[int, int, int, int, bool]],
    occupied: np.ndarray,
    thick: int,
    dots: int,
) -> list[tuple[int, int, int, int, bool]] | None:
    """Attach *dots* 1x1 markers adjacent to existing rects. Returns None on failure."""
    for _ in range(dots):
        choices: list[tuple[int, int]] = []
        for r, c, h, w, _ in rects:
            for rr in range(r, r + h):
                for cc in (c - 1, c + w):
                    if 0 <= cc < thick and not occupied[rr, cc]:
                        choices.append((rr, cc))
        if not choices:
            return None
        rr, cc = random.choice(choices)
        occupied[rr, cc] = True
        rects.append((rr, cc, 1, 1, True))
    return rects


def _corridor_rects(long: int, thick: int, count: int, dots: int) -> list[tuple[int, int, int, int, bool]] | None:
    """Generate connected stacked rectangles along the long axis, plus dot markers."""
    min_len = max(count + 2, round(CORRIDOR_COVERAGE * long))
    if min_len > long - 1:
        min_len = long - 1
    hole_len = random.randint(min_len, long - 1)
    lengths = _split_lengths(hole_len, count)
    if lengths is None:
        return None
    widths = _choose_widths(lengths, thick, dots, long * thick)
    if widths is None:
        return None

    start = random.randint(0, long - hole_len)
    r = start
    prev_c = prev_w = None
    rects: list[tuple[int, int, int, int, bool]] = []
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
        rects.append((r, c, h, w, False))
        r += h
        prev_c, prev_w = c, w

    occupied = np.zeros((long, thick), dtype=bool)
    for r0, c0, h, w, _ in rects:
        occupied[r0 : r0 + h, c0 : c0 + w] = True

    return _place_dot_markers(rects, occupied, thick, dots)


def _make_patch(h: int, w: int, cols: list[int]) -> np.ndarray:
    """
    Create a directionally-asymmetric patch.

    Non-dot pieces use a simple corner-L marker so every non-zero rotation is visibly different.
    """
    out = np.full((h, w), cols[0], dtype=int)
    if h == 1 and w == 1:
        return out
    if len(cols) == 1:
        return out

    accent = cols[1]
    if h == 1:
        out[0, 0] = accent
        if w >= 4:
            out[0, 1] = accent
        return out
    if w == 1:
        out[0, 0] = accent
        if h >= 4:
            out[1, 0] = accent
        return out

    arm_w = 3 if w >= 5 else 2
    arm_h = 3 if h >= 5 else 2
    out[0, :arm_w] = accent
    out[:arm_h, 0] = accent

    if h >= 4 and w >= 4:
        out[1, 1] = accent
    return out


def _make_pieces(
    rects: list[tuple[int, int, int, int, bool]],
    bg: int,
    base: int,
) -> dict[int, np.ndarray] | None:
    """Assign colors and build final-orientation patch arrays for each rect."""
    groups: dict[tuple[int, int], list[int]] = {}
    for i, (_, _, h, w, is_dot) in enumerate(rects):
        if not is_dot:
            groups.setdefault((h, w), []).append(i)
    if len(groups) < MIN_SHAPE_GROUPS:
        return None

    available = [c for c in range(1, 10) if c not in (bg, base)]
    if len(available) < MIN_PALETTE_SIZE:
        return None
    palette = random.sample(available, random.randint(MIN_PALETTE_SIZE, len(available)))

    keys = list(groups)
    random.shuffle(keys)
    group_cols: dict[tuple[int, int], list[int]] = {}
    for idx, key in enumerate(keys):
        primary = palette[idx % len(palette)]
        secondary = palette[(idx + 1 + random.randrange(len(palette) - 1)) % len(palette)]
        if secondary == primary:
            secondary = palette[(idx + 1) % len(palette)]
        group_cols[key] = [primary, secondary]

    pieces: dict[int, np.ndarray] = {}
    for i, (_, _, h, w, is_dot) in enumerate(rects):
        if is_dot:
            pieces[i] = np.array([[0]], dtype=int)
            continue
        patch = _make_patch(h, w, group_cols[(h, w)])
        assert all(not np.array_equal(patch, np.rot90(patch, k)) for k in (1, 2, 3))
        pieces[i] = patch
    return pieces


def _render_output(
    dh: int,
    dw: int,
    base: int,
    rects: list[tuple[int, int, int, int, bool]],
    pieces: dict[int, np.ndarray],
) -> np.ndarray:
    """Build the expected output: base-colored template with rotated-into-place pieces."""
    out = np.full((dh, dw), base, dtype=int)
    for i, (r, c, _, _, _) in enumerate(rects):
        paste(out, pieces[i], r, c)
    return out


def _scatter_sprites(
    H: int,
    W: int,
    bg: int,
    base: int,
    tr0: int,
    tc0: int,
    dh: int,
    dw: int,
    rects: list[tuple[int, int, int, int, bool]],
    pieces: dict[int, np.ndarray],
) -> np.ndarray:
    """Build the input grid: template frame with holes + inverse-rotated scattered sprites outside."""
    inp = np.full((H, W), bg, dtype=int)
    inp[tr0 : tr0 + dh, tc0 : tc0 + dw] = base
    for r, c, h, w, _ in rects:
        inp[tr0 + r : tr0 + r + h, tc0 + c : tc0 + c + w] = bg

    placement = np.full((H, W), bg, dtype=int)
    placement[tr0 : tr0 + dh, tc0 : tc0 + dw] = base

    order = list(range(len(rects)))
    random.shuffle(order)
    order.sort(key=lambda i: pieces[i].size, reverse=True)
    for i in order:
        r, c, h, w, _is_dot = rects[i]
        rot = _rotation_for_rect(r, c, h, w, dh, dw)
        patch = np.rot90(pieces[i], 4 - int(rot))
        rr, cc = random_free_location(placement, patch, transparent=bg, padding=1, padding_connectivity=8)
        paste(inp, patch, rr, cc)
        paste(placement, patch, rr, cc)
    return inp


def _zone_coverage_ok(rects: list[tuple[int, int, int, int, bool]], dh: int, dw: int) -> bool:
    seen: set[int] = set()
    for r, c, h, w, is_dot in rects:
        if is_dot:
            continue
        seen.add(_zone_index(r, c, h, w, dh, dw))
    return seen == {0, 1, 2}


def _gen_attempt() -> np.ndarray | None:
    """Single attempt: sample layout, build pieces, scatter inverse rotations, oracle-verify."""
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

    total = random.randint(SPRITE_COUNT_MIN, SPRITE_COUNT_MAX)
    min_dots = max(DOT_MIN, total - (long - DOT_LONG_BUFFER))
    max_dots = min(DOT_MAX, total - DOT_NON_DOT_BUFFER)
    if min_dots > max_dots:
        return None
    dots = random.randint(min_dots, max_dots)

    canon = _corridor_rects(long, thick, total - dots, dots)
    if canon is None:
        return None
    rects = canon if vertical else [(c, r, w, h, d) for r, c, h, w, d in canon]
    if not _zone_coverage_ok(rects, dh, dw):
        return None

    bg, base = random.sample(range(1, 10), 2)
    pieces = _make_pieces(rects, bg, base)
    if pieces is None:
        return None

    out = _render_output(dh, dw, base, rects, pieces)
    try:
        inp = _scatter_sprites(H, W, bg, base, tr0, tc0, dh, dw, rects, pieces)
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
    raise RuntimeError("failed to generate valid 5dbc8537 D4 input")
