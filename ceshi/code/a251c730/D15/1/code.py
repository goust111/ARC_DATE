# Variant of a251c730
# Dimension: D15 — 符号编码与图例 (symbol_instructions)
# Change: 无符号编码——精灵和标记的对应关系直接由颜色相等表达，不需要额外图例解读
#         → 标记形状编码变换：目标区的彩色标记改为小符号，颜色仍匹配精灵类型，
#           但符号形状额外指示印刻前要做的变换。

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, paste, random_free_location

# Design analysis
# ---------------
# variant_spec:
#   source_task:
#     original_behavior:
#       - 源框里放 1-2 类多色精灵，每类由一个稀有锚点色唯一标识。
#       - 目标框里只有同色单像素 marker；solver 逐个按颜色匹配，再把精灵平移到 marker 上。
#     new_behavior:
#       - 颜色仍只负责“这是哪一类精灵”。
#       - 单像素 marker 改成小型单色符号；符号的形状负责“印刻前做什么变换”。
#       - 采用 3 档固定图例，避免图例过多导致样例中无法归纳：
#         - 2 像素水平线 `HH` -> identity
#         - 2 像素竖线 `V/V` -> rot90
#         - 3 像素 L 形 `L./LL` -> flip_h
#       - 三种符号都把包围盒左上角那个已着色像素作为公共定位点；变换后的精灵锚点对齐到这里。
#   concretization_and_trace:
#     sprites:
#       - 继续使用 2 色精灵：主体色 + 唯一锚点色。
#       - 为了让旋转/翻转可见，只使用对 rot90 和 flip_h 都不对称的形状。
#     marker_example:
#       - 原题：红色单点 -> 找红锚点精灵 -> 原样平移。
#       - 变体：红色竖线 -> 找红锚点精灵 -> 先旋转 90 度 -> 再把“旋转后的锚点”对齐到竖线左上格。
#       - 差异不在匹配对象，而在匹配后多了一步由符号控制的几何变换。
#   complexity_calibration:
#     - 只保留 3 个符号档位，避免大量旋转/镜像分支让 2-4 个样例不足以总结规则。
#     - 每个生成样例固定放 3 个 marker，并且 3 种符号各出现一次，保证 D15 行为一定可观察。
#     - 仍然只用颜色匹配类型，不引入额外颜色映射或排序规则。
#   hidden_assumption_audit:
#     old_main_dependencies:
#       - 旧版 `dst` 识别依赖 “目标框组件最大尺寸 == 1”；变体后 marker 是 2-3 像素，必须改成按小符号形状识别。
#       - 旧版逐个扫描 `inner != dst_bg` 的像素；变体后一个 marker 含多个像素，必须改成按连通组件解析。
#       - 旧版默认 marker 像素本身就是对齐点；变体后要先从符号形状恢复操作码，再取其公共定位点。
#       - 旧版可直接使用源精灵原 patch；变体后必须同步变换 patch、mask 和 anchor 坐标。
#     retained_assumptions:
#       - 源框仍然通过“存在较大多色组件”识别。
#       - 锚点仍是每个源精灵里最稀有的颜色；生成器保证每个精灵恰有一个锚点像素。
#       - patch 非前景部分仍用 0 作为内部哨兵，所以生成器保留颜色 0 不参与精灵/边框/背景，避免与真实内容冲突。
#   placement_legality:
#     - 输出优先：先在目标输出内放置“变换后的精灵”，再反推输入里的符号位置。
#     - 设输出内精灵左上角为 `(top,left)`，变换后锚点为 `(ar,ac)`，则符号定位点固定为 `(top+ar,left+ac)`。
#     - 只有当该定位点放入对应符号后仍完全位于目标内、且符号之间留出 1 格空白时，此位置才合法。
#     - 输出内精灵彼此不重叠，输入内符号彼此不重叠且不相邻，因此 solver 组件分割稳定。
#   visual_observability:
#     - 3 个 marker 同时出现，解题者能直接看到“同色但不同形状”导致输出朝向不同。
#     - 源精灵特意选非对称形状，旋转/翻转后的差异肉眼可见，不会退化成原题。
#   cleanliness:
#     - 目标区输入只含小符号，不再叠加整只精灵，视觉层次仍然是“源模板区 + 目标指令区”两层。
#     - 符号统一是单色小组件，和源区多色精灵在外观上容易区分。
#   color_inference:
#     - 颜色只承担类型匹配，和原题一致；新增信息只由形状传达，因此不存在额外颜色置换推理负担。
#   new_solver_assumptions_checked_against_generator:
#     - `main()` 假设目标框内每个组件都是单色且形状来自 3 个固定符号之一；生成器严格按该库放置。
#     - `main()` 假设每种锚点色在源框中至少出现一只精灵；生成器对 source/marker 计数都保证每类 >= 1。
#     - `main()` 假设所有 marker 的定位点就是其组件 bbox 左上角；生成器只使用左上角着色的符号模板。


def _normalize_mask(mask: np.ndarray) -> frozenset[tuple[int, int]]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return frozenset()
    r0, c0 = coords.min(axis=0)
    return frozenset((int(r - r0), int(c - c0)) for r, c in coords)


MARKER_SHAPES: dict[str, tuple[str, ...]] = {
    "identity": ("MM",),
    "rot90": ("M", "M"),
    "flip_h": ("M.", "MM"),
}
MARKER_SIGNATURE_TO_OP = {
    frozenset((r, c) for r, row in enumerate(rows) for c, ch in enumerate(row) if ch == "M"): op
    for op, rows in MARKER_SHAPES.items()
}
MARKER_COMPONENT_SIZES = {len(sig) for sig in MARKER_SIGNATURE_TO_OP}
TRANSFORM_OPS = ("identity", "rot90", "flip_h")


def _render_shape(rows: tuple[str, ...], fill: int, bg: int, key: str) -> np.ndarray:
    return np.array([[fill if ch == key else bg for ch in row] for row in rows], dtype=int)


def _render_marker(op: str, color: int, bg: int) -> np.ndarray:
    return _render_shape(MARKER_SHAPES[op], color, bg, "M")


def _transform_array(arr: np.ndarray, op: str) -> np.ndarray:
    if op == "identity":
        return arr.copy()
    if op == "rot90":
        return np.rot90(arr, -1).copy()
    if op == "flip_h":
        return np.fliplr(arr).copy()
    raise ValueError(f"unknown transform op: {op}")


def _apply_transform(
    patch: np.ndarray, mask: np.ndarray, anchor_rc: tuple[int, int], op: str
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    patch_t = _transform_array(patch, op)
    mask_t = _transform_array(mask.astype(int), op).astype(bool)
    anchor_mask = np.zeros(mask.shape, dtype=int)
    anchor_mask[int(anchor_rc[0]), int(anchor_rc[1])] = 1
    anchor_t_mask = _transform_array(anchor_mask, op)
    anchor_pts = np.argwhere(anchor_t_mask == 1)
    assert anchor_pts.shape[0] == 1
    ar, ac = map(int, anchor_pts[0])
    return patch_t, mask_t, (int(ar), int(ac))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp source sprites into a destination frame via color-matched symbol markers.

    Input structure:
    - Two bordered rectangles: a source rectangle with multi-cell sprites and a destination rectangle with tiny colored symbols.
    - Each source sprite has a unique anchor color; each destination symbol uses that same color and also has a shape from a fixed symbol legend.

    Rule:
    - Match each destination symbol to the source sprite whose rare anchor color is the same.
    - Decode the symbol shape into a transform, apply it to the sprite, then align the transformed anchor to the symbol's top-left locator cell.
    - Output is the destination rectangle with its interior cleared and the transformed sprites stamped in.

    Key insight: Color still selects the sprite type, but marker geometry now encodes a second instruction: how the matched sprite must be transformed before placement.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape

    def interior_bg(rect: tuple[int, int, int, int]) -> int:
        r0, c0, r1, c1 = map(int, rect)
        inner = g[int(r0) + 1 : int(r1), int(c0) + 1 : int(c1)]
        return int(Counter(map(int, inner.flatten())).most_common(1)[0][0])

    def rect_components(rect: tuple[int, int, int, int]) -> tuple[int, list[dict]]:
        r0, c0, r1, c1 = map(int, rect)
        inner = g[int(r0) + 1 : int(r1), int(c0) + 1 : int(c1)]
        ibg = int(Counter(map(int, inner.flatten())).most_common(1)[0][0])
        comps = []

        for coords in components_mask(inner != int(ibg), connectivity=4):
            if coords.size == 0:
                continue
            rr0, cc0, rr1, cc1 = bbox_coords(coords)
            mask = np.zeros((int(rr1 - rr0 + 1), int(cc1 - cc0 + 1)), dtype=bool)
            mask[coords[:, 0] - int(rr0), coords[:, 1] - int(cc0)] = True
            patch = inner[int(rr0) : int(rr1) + 1, int(cc0) : int(cc1) + 1].copy()
            patch[~mask] = 0

            vals = [int(inner[int(r), int(c)]) for r, c in coords]
            cols = Counter(vals)
            comps.append(
                {
                    "size": int(coords.shape[0]),
                    "colors": dict(cols),
                    "bbox": (int(rr0), int(cc0), int(rr1), int(cc1)),
                    "mask": mask,
                    "patch": patch,
                    "signature": _normalize_mask(mask),
                }
            )

        comps.sort(key=lambda d: -int(d["size"]))
        return int(ibg), comps

    rects: list[tuple[int, tuple[int, int, int, int], int, int]] = []
    min_h = 6
    min_w = 6
    for r0 in range(int(H)):
        for c0 in range(int(W)):
            bc = int(g[int(r0), int(c0)])
            for r1 in range(int(r0 + min_h - 1), int(H)):
                for c1 in range(int(c0 + min_w - 1), int(W)):
                    if (
                        int(g[int(r0), int(c1)]) != int(bc)
                        or int(g[int(r1), int(c0)]) != int(bc)
                        or int(g[int(r1), int(c1)]) != int(bc)
                    ):
                        continue
                    if not bool((g[int(r0), int(c0) : int(c1) + 1] == int(bc)).all()):
                        continue
                    if not bool((g[int(r1), int(c0) : int(c1) + 1] == int(bc)).all()):
                        continue
                    if not bool((g[int(r0) : int(r1) + 1, int(c0)] == int(bc)).all()):
                        continue
                    if not bool((g[int(r0) : int(r1) + 1, int(c1)] == int(bc)).all()):
                        continue
                    ibg = interior_bg((int(r0), int(c0), int(r1), int(c1)))
                    if int(ibg) == int(bc):
                        continue
                    area = int((int(r1) - int(r0) + 1) * (int(c1) - int(c0) + 1))
                    rects.append((int(area), (int(r0), int(c0), int(r1), int(c1)), int(bc), int(ibg)))

    assert rects

    rects.sort(reverse=True, key=lambda x: int(x[0]))
    rects = rects[:6]

    src = None
    dst = None
    for _area, rect, bc, _ibg in rects:
        ibg2, comps = rect_components(rect)
        if not comps:
            continue
        max_sz = max(int(c["size"]) for c in comps)
        is_symbol_rect = (
            len(comps) >= 2
            and all(
                int(comp["size"]) in MARKER_COMPONENT_SIZES
                and len([k for k in comp["colors"] if int(k) != 0]) == 1
                and comp["signature"] in MARKER_SIGNATURE_TO_OP
                for comp in comps
            )
        )
        if max_sz >= 5 and src is None:
            src = (rect, bc, ibg2, comps)
        if is_symbol_rect and dst is None:
            dst = (rect, bc, ibg2, comps)

    assert src is not None and dst is not None, "must find both source and destination rectangles"

    (_sr, _sc, _sr1, _sc1), _src_bc, _src_bg, src_comps = src
    (_dr, _dc, dr1, dc1), dst_bc, dst_bg, dst_comps = dst

    templates: dict[int, tuple[np.ndarray, np.ndarray, tuple[int, int]]] = {}
    for comp in src_comps:
        cols = {int(k): int(v) for k, v in comp["colors"].items() if int(k) != 0}
        if len(cols) < 2:
            continue
        anchor_col = int(min(cols.items(), key=lambda kv: (int(kv[1]), int(kv[0])))[0])
        anchors = np.argwhere((comp["patch"] == int(anchor_col)) & comp["mask"])
        if anchors.size == 0:
            continue
        ar, ac = map(int, anchors[0])
        if int(anchor_col) not in templates or int(comp["mask"].sum()) > int(templates[int(anchor_col)][1].sum()):
            templates[int(anchor_col)] = (comp["patch"].copy(), comp["mask"].copy(), (int(ar), int(ac)))

    out_h = int(dr1 - _dr + 1)
    out_w = int(dc1 - _dc + 1)
    out = np.full((out_h, out_w), int(dst_bg), dtype=int)
    out[0, :] = int(dst_bc)
    out[-1, :] = int(dst_bc)
    out[:, 0] = int(dst_bc)
    out[:, -1] = int(dst_bc)

    marker_records: list[tuple[int, int, int, str]] = []
    for comp in dst_comps:
        colors = [int(k) for k in comp["colors"] if int(k) != 0]
        if len(colors) != 1:
            continue
        op = MARKER_SIGNATURE_TO_OP.get(comp["signature"])
        if op is None:
            continue
        rr0, cc0, _rr1, _cc1 = map(int, comp["bbox"])
        marker_records.append((int(rr0), int(cc0), int(colors[0]), op))

    marker_records.sort()
    for rr0, cc0, mcol, op in marker_records:
        if int(mcol) not in templates:
            continue
        patch, mask, anchor_rc = templates[int(mcol)]
        patch_t, mask_t, (ar, ac) = _apply_transform(patch, mask, anchor_rc, op)
        top = int(1 + int(rr0) - int(ar))
        left = int(1 + int(cc0) - int(ac))
        paste(out, patch_t, int(top), int(left), mask=mask_t, overwrite=True, clip=True)

    return out


# --------------- generator (O2) ---------------

# === CFG: Canvas & layout ===
GRID_SIZE = 30
WALLPAPER_COLORS = (4, 5)
TOTAL_COLORS_TARGET = (7, 8, 9)
GAP_RANGE = (1, 4)

# === CFG: Rectangle size bounds ===
SRC_H_RANGE = (9, 19)
SRC_W_RANGE = (8, 18)
DST_H_RANGE = (9, 19)
DST_W_RANGE = (8, 20)
DST_W_SIDE_MAX = 20

# === CFG: Sprite & placement ===
SPRITE_TYPE_CHOICES = (1, 2)
MARKER_COUNT = 3
SOURCE_SPRITE_RANGE = (2, 4)
SOURCE_SPRITE_PADDING = 1
MARKER_PADDING = 1

# === CFG: Generation retries ===
MAX_GEN_ATTEMPTS = 400

SPRITE_SHAPES = [
    (("AB.", "BBB"), (0, 0)),
    (("ABB", "BBB"), (0, 0)),
    ((".A.", "BBB", ".BB"), (0, 1)),
    (("AB.", "BBB", ".BB"), (0, 0)),
    (("ABB", "BBB", "..B"), (0, 0)),
    ((".AB", "BBB", "BB."), (0, 1)),
    (("B.B", "BBB", ".A."), (2, 1)),
    (("AB.", "BBB", ".B."), (0, 0)),
    (("BAB", "BBB"), (0, 1)),
]


def _render_sprite(shape_rows: tuple[str, ...], body: int, anchor: int, bg: int) -> np.ndarray:
    char_map = {"B": body, "A": anchor}
    return np.array([[char_map.get(ch, bg) for ch in row] for row in shape_rows], dtype=int)


def _make_wallpaper(direction: str, palette: list[int]) -> np.ndarray:
    period = len(palette)
    stripe = np.array(palette * ((GRID_SIZE + period - 1) // period), dtype=int)[:GRID_SIZE]
    if direction == "vertical":
        return np.tile(stripe, (GRID_SIZE, 1))
    return np.tile(stripe.reshape(-1, 1), (1, GRID_SIZE))


def _build_frame(h: int, w: int, border: int, bg: int) -> np.ndarray:
    rect = np.full((h, w), bg, dtype=int)
    rect[0, :] = rect[-1, :] = border
    rect[:, 0] = rect[:, -1] = border
    return rect


def _choose_counts(n_total: int, n_types: int) -> list[int]:
    counts = [1] * n_types
    for _ in range(n_total - n_types):
        counts[random.randrange(n_types)] += 1
    return counts


def _choose_dimensions(layout: str) -> tuple[int, int, int, int]:
    gap = random.randint(*GAP_RANGE)
    if layout == "side_by_side":
        src_w = random.randint(*SRC_W_RANGE)
        dst_w_max = min(DST_W_SIDE_MAX, GRID_SIZE - gap - src_w)
        if dst_w_max < DST_W_RANGE[0]:
            raise ValueError("no side-by-side width fit")
        dst_w = random.randint(DST_W_RANGE[0], dst_w_max)
        src_h = random.randint(*SRC_H_RANGE)
        dst_h = random.randint(*DST_H_RANGE)
    else:
        src_h = random.randint(*SRC_H_RANGE)
        dst_h_max = min(DST_H_RANGE[1], GRID_SIZE - gap - src_h)
        if dst_h_max < DST_H_RANGE[0]:
            raise ValueError("no stacked height fit")
        dst_h = random.randint(DST_H_RANGE[0], dst_h_max)
        src_w = random.randint(*SRC_W_RANGE)
        dst_w = random.randint(*DST_W_RANGE)
    return src_h, src_w, dst_h, dst_w


def _place_rectangles(
    layout: str, src_h: int, src_w: int, dst_h: int, dst_w: int
) -> tuple[tuple[int, int], tuple[int, int]]:
    if layout == "side_by_side":
        gap = random.randint(1, GRID_SIZE - src_w - dst_w)
        slack = GRID_SIZE - src_w - dst_w - gap
        left_margin = random.randint(0, slack)
        src_r = random.randint(0, GRID_SIZE - src_h)
        dst_r = random.randint(0, GRID_SIZE - dst_h)
        if random.random() < 0.5:
            src_c, dst_c = left_margin, left_margin + src_w + gap
        else:
            dst_c, src_c = left_margin, left_margin + dst_w + gap
    else:
        gap = random.randint(1, GRID_SIZE - src_h - dst_h)
        slack = GRID_SIZE - src_h - dst_h - gap
        top_margin = random.randint(0, slack)
        src_c = random.randint(0, GRID_SIZE - src_w)
        dst_c = random.randint(0, GRID_SIZE - dst_w)
        if random.random() < 0.5:
            src_r, dst_r = top_margin, top_margin + src_h + gap
        else:
            dst_r, src_r = top_margin, top_margin + dst_h + gap
    return (src_r, src_c), (dst_r, dst_c)


def _assign_palette(n_types: int) -> tuple[list[int], int, int, list[int], list[int], int | None] | None:
    wallpaper_n = 5 if n_types == 1 else random.choice(list(WALLPAPER_COLORS))
    used_total_target = random.choice(list(TOTAL_COLORS_TARGET))
    extras_needed = used_total_target - (wallpaper_n + 1 + n_types)
    if extras_needed < 0 or extras_needed > 2:
        return None

    all_colors = list(range(1, 10))
    wallpaper = random.sample(all_colors, wallpaper_n)
    remaining = [c for c in all_colors if c not in wallpaper]
    random.shuffle(remaining)

    src_border = remaining.pop()
    anchors = [remaining.pop() for _ in range(n_types)]

    dst_border_extra = extras_needed >= 2 or (extras_needed == 1 and random.random() < 0.5)
    body_extra = extras_needed >= 2 or (extras_needed >= 1 and not dst_border_extra)

    if dst_border_extra:
        if not remaining:
            return None
        dst_border = remaining.pop()
    else:
        border_choices = [c for c in wallpaper if c != src_border]
        if not border_choices:
            return None
        dst_border = random.choice(border_choices)

    body_pool = wallpaper[:]
    forced_body: int | None = None
    if body_extra:
        if not remaining:
            return None
        forced_body = remaining.pop()
        body_pool.append(forced_body)

    return wallpaper, src_border, dst_border, anchors, body_pool, forced_body


def _shape_is_variant_visible(shape_rows: tuple[str, ...], anchor_rc: tuple[int, int]) -> bool:
    patch = _render_sprite(shape_rows, 1, 2, 0)
    mask = patch != 0
    variants = []
    for op in TRANSFORM_OPS:
        patch_t, mask_t, anchor_t = _apply_transform(patch, mask, anchor_rc, op)
        variants.append((patch_t.shape, tuple(map(tuple, patch_t.tolist())), tuple(map(tuple, mask_t.astype(int).tolist())), anchor_t))
    return len(set(variants)) == len(variants)


ASYM_SPRITE_SHAPES = [spec for spec in SPRITE_SHAPES if _shape_is_variant_visible(*spec)]


def _occupied_overlap(canvas: np.ndarray, patch: np.ndarray, top: int, left: int, bg: int) -> bool:
    h, w = patch.shape
    return bool(((canvas[int(top) : int(top) + h, int(left) : int(left) + w] != int(bg)) & (patch != int(bg))).any())


def _bbox_has_padding(canvas: np.ndarray, top: int, left: int, h: int, w: int, bg: int, padding: int) -> bool:
    rr0 = max(0, int(top) - int(padding))
    cc0 = max(0, int(left) - int(padding))
    rr1 = min(canvas.shape[0], int(top) + int(h) + int(padding))
    cc1 = min(canvas.shape[1], int(left) + int(w) + int(padding))
    return not bool((canvas[rr0:rr1, cc0:cc1] != int(bg)).any())


def _place_markers(
    sprite_types: list[dict], marker_specs: list[dict], dst_h: int, dst_w: int, dst_bg: int
) -> tuple[np.ndarray, np.ndarray, list[dict]] | None:
    inner_h = dst_h - 2
    inner_w = dst_w - 2
    out_inner = np.full((inner_h, inner_w), dst_bg, dtype=int)
    marker_inner = np.full((inner_h, inner_w), dst_bg, dtype=int)
    placed: list[dict] = []

    for spec in marker_specs:
        sprite = sprite_types[int(spec["type_id"])]
        base_patch = _render_sprite(sprite["shape_rows"], sprite["body"], sprite["anchor"], dst_bg)
        base_mask = base_patch != dst_bg
        patch_t, mask_t, anchor_t = _apply_transform(base_patch, base_mask, sprite["anchor_rc"], str(spec["op"]))
        marker_patch = _render_marker(str(spec["op"]), sprite["anchor"], dst_bg)
        marker_h, marker_w = marker_patch.shape
        ph, pw = patch_t.shape

        candidates: list[tuple[int, int, int, int]] = []
        for top in range(inner_h - ph + 1):
            for left in range(inner_w - pw + 1):
                if _occupied_overlap(out_inner, patch_t, top, left, dst_bg):
                    continue
                locator_r = int(top + int(anchor_t[0]))
                locator_c = int(left + int(anchor_t[1]))
                if locator_r + marker_h > inner_h or locator_c + marker_w > inner_w:
                    continue
                if _occupied_overlap(marker_inner, marker_patch, locator_r, locator_c, dst_bg):
                    continue
                if not _bbox_has_padding(marker_inner, locator_r, locator_c, marker_h, marker_w, dst_bg, MARKER_PADDING):
                    continue
                candidates.append((int(top), int(left), int(locator_r), int(locator_c)))

        if not candidates:
            return None

        top, left, locator_r, locator_c = random.choice(candidates)
        paste(out_inner, patch_t, top, left, transparent=dst_bg)
        paste(marker_inner, marker_patch, locator_r, locator_c, transparent=dst_bg)
        placed.append(
            {
                "type_id": int(spec["type_id"]),
                "op": str(spec["op"]),
                "locator": (int(locator_r), int(locator_c)),
            }
        )

    return out_inner, marker_inner, placed


def _place_sources(sprite_types: list[dict], source_type_ids: list[int], src_h: int, src_w: int, src_bg: int) -> np.ndarray | None:
    inner = np.full((src_h - 2, src_w - 2), src_bg, dtype=int)
    for type_id in source_type_ids:
        sprite = sprite_types[int(type_id)]
        patch = _render_sprite(sprite["shape_rows"], sprite["body"], sprite["anchor"], src_bg)
        try:
            r0, c0 = random_free_location(
                inner,
                patch,
                transparent=src_bg,
                border=SOURCE_SPRITE_PADDING,
                padding=SOURCE_SPRITE_PADDING,
            )
        except ValueError:
            return None
        paste(inner, patch, r0, c0, transparent=src_bg)
    return inner


def _gen_attempt() -> np.ndarray | None:
    n_types = random.choice(list(SPRITE_TYPE_CHOICES))

    palette_result = _assign_palette(n_types)
    if palette_result is None:
        return None
    wallpaper, src_border, dst_border, anchors, body_pool, forced_body = palette_result

    src_bg = random.choice(wallpaper)
    dst_bg_choices = [c for c in wallpaper if c != dst_border]
    if not dst_bg_choices:
        return None
    dst_bg = random.choice(dst_bg_choices)

    if len(ASYM_SPRITE_SHAPES) < n_types:
        return None
    sprite_specs = random.sample(ASYM_SPRITE_SHAPES, n_types)

    body_choices = [c for c in body_pool if c not in anchors and c != src_bg and c != dst_bg]
    if len(set(body_choices)) < n_types:
        return None
    if forced_body is not None and forced_body not in body_choices:
        return None

    bodies: list[int] = []
    remaining_bodies = body_choices[:]
    random.shuffle(remaining_bodies)
    if forced_body is not None:
        bodies.append(int(forced_body))
        remaining_bodies = [c for c in remaining_bodies if c != forced_body]
    while len(bodies) < n_types:
        pick = remaining_bodies.pop()
        if pick not in bodies:
            bodies.append(int(pick))

    sprite_types = [
        {
            "shape_rows": shape_rows,
            "anchor_rc": anchor_rc,
            "body": int(body),
            "anchor": int(anchor),
            "h": len(shape_rows),
            "w": len(shape_rows[0]),
        }
        for (shape_rows, anchor_rc), body, anchor in zip(sprite_specs, bodies, anchors)
    ]

    source_total = random.randint(max(2, n_types), SOURCE_SPRITE_RANGE[1])
    source_type_ids = [i for i, ct in enumerate(_choose_counts(source_total, n_types)) for _ in range(ct)]
    random.shuffle(source_type_ids)

    marker_type_ids = [i for i, ct in enumerate(_choose_counts(MARKER_COUNT, n_types)) for _ in range(ct)]
    random.shuffle(marker_type_ids)
    marker_ops = list(TRANSFORM_OPS)
    random.shuffle(marker_ops)
    marker_specs = [{"type_id": int(type_id), "op": str(op)} for type_id, op in zip(marker_type_ids, marker_ops)]

    layout = random.choice(["side_by_side", "stacked"])
    try:
        src_h, src_w, dst_h, dst_w = _choose_dimensions(layout)
    except ValueError:
        return None
    if src_h * src_w <= dst_h * dst_w:
        return None

    marker_result = _place_markers(sprite_types, marker_specs, dst_h, dst_w, dst_bg)
    if marker_result is None:
        return None
    out_inner, marker_inner, _marker_records = marker_result

    output = _build_frame(dst_h, dst_w, dst_border, dst_bg)
    paste(output[1:-1, 1:-1], out_inner, 0, 0)

    source_inner = _place_sources(sprite_types, source_type_ids, src_h, src_w, src_bg)
    if source_inner is None:
        return None

    source_rect = _build_frame(src_h, src_w, src_border, src_bg)
    paste(source_rect[1:-1, 1:-1], source_inner, 0, 0)

    dest_rect = _build_frame(dst_h, dst_w, dst_border, dst_bg)
    paste(dest_rect[1:-1, 1:-1], marker_inner, 0, 0)

    try:
        src_pos, dst_pos = _place_rectangles(layout, src_h, src_w, dst_h, dst_w)
    except ValueError:
        return None

    grid = _make_wallpaper(random.choice(["vertical", "horizontal"]), wallpaper)
    paste(grid, source_rect, src_pos[0], src_pos[1])
    paste(grid, dest_rect, dst_pos[0], dst_pos[1])

    if not np.array_equal(main(grid), output):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a251c730 D15 input after {MAX_GEN_ATTEMPTS} attempts")
