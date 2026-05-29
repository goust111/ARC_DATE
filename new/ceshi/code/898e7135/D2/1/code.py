# Variant of 898e7135
# Dimension: D2 — 源对象：精灵的结构与属性 (src)
# Change: 精灵由 k×k 实心块组成（k 为缩放因子），每个精灵单色，非单元精灵颜色互不相同，单元精灵（k×k 实心方块）可共享颜色 → 精灵含内部洞（非实心块）

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import (
    bbox,
    bbox_coords,
    components_mask,
    crop_bbox,
    find_components,
    get_bg,
    paste,
    random_free_location,
)


"""
design_analysis:
  variant_goal:
    keep_constant:
      - 最大前景连通块仍是单色框架
      - 框架内是洞/墙布局，输出仍为按 k 放大的裁剪框架
      - 外部精灵仍按 D4 旋转/翻转后与框架内对应洞区域匹配
      - 颜色机制基本不变：非单元精灵颜色互异，单元精灵可共享颜色，干扰点仍是单像素
    changed_core:
      - 非单元精灵不再是“块级全实心 polyomino”，而是“块级掩码里含缺块的连通形”
      - 框架中的一个 slot 也因此不只是普通洞连通块；其 bbox 内会有被包围的 frame-color 墙岛
      - 这些墙岛在输入里表现为框架内部的二阶结构，在输出里表现为放大后仍保留的内部 frame-color 洞

  concrete_pixelization:
    layout_level:
      - slot 仍由 layout 上的 bg 单元组成，4 连通
      - 但某些 slot 的 tight bbox 内存在 False 单元且这些 False 不接触 bbox 边界；它们是 slot 的内部墙
    sprite_level:
      - 将 slot 的 bool mask 直接做 kron 放大得到单色精灵
      - mask=False 的块在输入精灵 bbox 内保留为背景 0，因此精灵本身出现可见空洞
    example_trace:
      original:
        - 3x3 slot mask 若是 L 形，放大后外部精灵是实心色块拼成的 L
        - solver 下采样时，每个 kxk 块只会是“全满/全空”，恢复 L 形并匹配
      variant:
        - 3x3 slot mask 可以是 111/101/111，中心 0 是框架内部墙岛
        - 放大后外部精灵是一个单色方环，中心保持背景色
        - solver 下采样后得到同一个 3x3 环形 mask，再匹配到框架中的环形 slot
      difference:
        - 原题只需识别一阶外轮廓；变体必须保留 bbox 内部缺块，处理“洞里的墙”

  complexity_calibration:
    anti_simplification:
      - 每个样例固定保留 1 个 unit slot 用于稳健推断 k
      - 其余 2-4 个非单元 slot 全部来自“带内部洞”的模板库，确保 D2 变化每题可见
      - 非单元 slot 仍要求 D4 规范形唯一，避免不同颜色但同形导致匹配歧义
    controlled_freedom:
      - 洞结构只来自少量模板，防止自由度过大导致成功率下降
      - 模板包含对称和非对称环形，既能让人看出内部洞，也保留旋转/翻转的可观察性

  implicit_assumption_audit:
    relevant_main_lines:
      - frame_comp = max(comps, key=area):
          assumption: 框架仍是最大前景连通块
          status: generator 继续保证成立
      - hole = sub == bg ; layout = tight-cropped hole mask:
          assumption: slot 可以只靠 bool mask 表达，不需要“填满 bbox”
          status: 仍成立，且正好保留内部 False 墙岛
      - components_mask(layout):
          assumption: 一个 slot 只需外层连通；bbox 内出现内部 False 不会破坏组件提取
          status: 成立，环形 slot 仍是单个 4 连通组件
      - piece mask pm built from component coords:
          assumption: 外部精灵 bbox 内允许存在背景孔洞
          status: 成立，coords 只记录前景，内部洞自然留为 False
      - infer k from smallest solid square piece:
          assumption: 至少有一个 unit piece
          status: 新 generator 固定生成 1 个 unit slot 来保证
      - downsample_blocks:
          assumption: 每个 kxk 子块必须全满或全空，不能半块
          status: 仍成立；变体改变的是“哪些块存在”，不是块内像素结构
      - dihedral variant dedup by raw bytes:
          old_risk: 不同 shape 若元素数相同，单看 bytes 可能碰撞
          fix: 改为使用 (shape, bytes) 去重
      - random_free_location on transparent holes:
          old_generator_risk: 原题无洞时无害；有洞后会允许别的对象落进精灵内部空洞
          fix: 生成器改为用整 bbox 的 probe patch 先找空位，再按 mask 粘贴真精灵

  placement_legality:
    layout:
      - 在工作布局上放置 slot 时，只检查 mask=True 的格子，并对其做 8 邻域禁放
      - 内部墙格因为被环形 slot 包围，天然不会再被别的 slot 占用
      - 最终对所有洞的联合 bbox 做裁剪，使 frame 内部布局与 solver 的 tight crop 完全一致
    canvas:
      - 框架先放入
      - 每个精灵用“整 bbox 不得覆盖已有非背景”的规则找位置，防止别的对象出现在精灵洞内
      - 干扰点最后散布，只落在周围 8 邻域全空的位置，因此也不会钻进洞里

  visual_observability:
    - 每题至少两个非单元环形精灵，肉眼能直接看到“精灵里有洞”
    - 框架内部对应位置也会有被包围的小墙岛，输出放大后这些墙岛仍保留
    - 颜色不参与排序或变换，只承担“哪块精灵填哪种形状”的可见区分

  cleanliness:
    - 不叠放对象，不让干扰点进入精灵洞内，不让精灵与框架 bbox 相交
    - 模板尺寸受控，画面仍是少量清晰对象，而不是密集噪声

  new_main_assumptions_and_generator_guarantees:
    - assumption: 至少一个 unit piece 存在
      guarantee: n_units 固定为 1
    - assumption: 非单元 slot 的 D4 规范形两两不同
      guarantee: 模板采样时按 canonical key 去重
    - assumption: 精灵块级结构严格对齐到 kxk
      guarantee: 所有精灵都由 kron(slot_mask, ones(k,k)) 生成
    - assumption: 背景 0 仍是全局最多色
      guarantee: 画布面积预算保守，生成后再次检查
"""


# --- tunables ----------------------------------------------------------------

K_CHOICES = [2, 3]
WORK_LAYOUT_H_RANGE = (6, 12)
WORK_LAYOUT_W_RANGE = (6, 9)
N_SLOTS_RANGE = (3, 5)  # total slots, including the single unit slot
N_UNITS = 1

HOLE_DENSITY_RANGE = (0.20, 0.55)
TEMPLATE_SLOT_RETRIES = 40
UNIT_SLOT_RETRIES = 40

CANVAS_DIM_RANGE = (18, 30)
AREA_BUFFER_MULT = 5
AREA_BUFFER_CONST = 60

DISTRACTOR_RANGE = (8, 13)
DISTRACTOR_RETRY_MULT = 10
MIN_DISTRACTORS = 8

GENERATE_MAX_RETRIES = 400


def _mask_from_rows(rows: list[str]) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


NON_UNIT_TEMPLATE_ROWS = [
    ["111", "101", "111"],
    ["1111", "1001", "1111"],
    ["1110", "1011", "1111", "0110"],
    ["0111", "1101", "1111", "0111"],
    ["0110", "1111", "1011", "1110"],
    ["01110", "11111", "11011", "01110"],
]

NON_UNIT_TEMPLATES = [_mask_from_rows(rows) for rows in NON_UNIT_TEMPLATE_ROWS]


def _count_internal_holes(mask: np.ndarray) -> int:
    """Count enclosed False-regions inside a tight bool mask."""
    hh, ww = mask.shape
    void = ~mask
    total = 0
    for coords in components_mask(void, connectivity=4):
        if coords.size == 0:
            continue
        rs = coords[:, 0]
        cs = coords[:, 1]
        if np.any(rs == 0) or np.any(rs == hh - 1) or np.any(cs == 0) or np.any(cs == ww - 1):
            continue
        total += 1
    return int(total)


def _canonical_shape_key(mask: np.ndarray) -> tuple:
    """Canonical key under D4, including shape to avoid bytes-only collisions."""
    variants = []
    for kk in range(4):
        r = np.rot90(mask, kk)
        variants.append((r.shape, r.tobytes()))
        fr = np.fliplr(r)
        variants.append((fr.shape, fr.tobytes()))
    return min(variants)


def _dihedral_variants(mask: np.ndarray) -> list[np.ndarray]:
    uniq: list[np.ndarray] = []
    seen: set[tuple] = set()
    for kk in range(4):
        rot = np.rot90(mask, kk)
        for variant in (rot, np.fliplr(rot)):
            key = (variant.shape, variant.tobytes())
            if key in seen:
                continue
            seen.add(key)
            uniq.append(variant)
    return uniq


def _random_dihedral(mask: np.ndarray) -> np.ndarray:
    return random.choice(_dihedral_variants(mask))


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[:-1, :] |= mask[1:, :]
    out[1:, :] |= mask[:-1, :]
    out[:, :-1] |= mask[:, 1:]
    out[:, 1:] |= mask[:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[:-1, 1:] |= mask[1:, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    return out


def _forbid_coords(forbidden: np.ndarray, coords: np.ndarray) -> None:
    cell_mask = np.zeros_like(forbidden)
    cell_mask[coords[:, 0], coords[:, 1]] = True
    forbidden |= _dilate8(cell_mask)


def _place_mask(forbidden: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    """Place a tight bool mask on the work layout without touching other slots."""
    H, W = forbidden.shape
    h, w = mask.shape
    if h > H or w > W:
        return None
    candidates = []
    for top in range(H - h + 1):
        for left in range(W - w + 1):
            region = forbidden[top : top + h, left : left + w]
            if bool((region & mask).any()):
                continue
            candidates.append((int(top), int(left)))
    if not candidates:
        return None
    top, left = random.choice(candidates)
    coords = np.argwhere(mask)
    coords = coords + np.array([top, left], dtype=int)
    return coords


def _crop_slots_to_union_bbox(slots: list[dict]) -> tuple[list[dict], np.ndarray] | tuple[None, None]:
    if not slots:
        return None, None
    layout_mask = None
    for slot in slots:
        coords = slot["coords"]
        if layout_mask is None:
            r1 = int(coords[:, 0].max()) + 1
            c1 = int(coords[:, 1].max()) + 1
            layout_mask = np.zeros((r1, c1), dtype=bool)
        else:
            nr = max(layout_mask.shape[0], int(coords[:, 0].max()) + 1)
            nc = max(layout_mask.shape[1], int(coords[:, 1].max()) + 1)
            if (nr, nc) != layout_mask.shape:
                grown = np.zeros((nr, nc), dtype=bool)
                grown[: layout_mask.shape[0], : layout_mask.shape[1]] = layout_mask
                layout_mask = grown
        layout_mask[coords[:, 0], coords[:, 1]] = True

    hole_box = bbox(layout_mask)
    if hole_box is None:
        return None, None
    lr0, lc0, lr1, lc1 = map(int, hole_box)
    cropped = layout_mask[lr0 : lr1 + 1, lc0 : lc1 + 1]

    new_slots: list[dict] = []
    for slot in slots:
        coords = slot["coords"] - np.array([lr0, lc0], dtype=int)
        rr0, cc0, rr1, cc1 = bbox_coords(coords)
        sm = np.zeros((int(rr1 - rr0 + 1), int(cc1 - cc0 + 1)), dtype=bool)
        sm[coords[:, 0] - rr0, coords[:, 1] - cc0] = True
        new_slots.append(
            {
                "coords": coords,
                "area": int(sm.sum()),
                "mask": sm,
                "shape_key": slot.get("shape_key"),
                "internal_holes": int(_count_internal_holes(sm)),
            }
        )
    return new_slots, cropped


def _gen_layout(work_h: int, work_w: int, n_slots: int, n_units: int) -> tuple[list[dict], np.ndarray] | tuple[None, None]:
    """Generate 1 unit slot plus 2-4 unique non-unit slots, all non-unit ones holed."""
    n_non_units = int(n_slots - n_units)
    if n_non_units <= 0:
        return None, None

    # Sample distinct template families under D4.
    template_pool = []
    seen_keys: set[tuple] = set()
    for base in NON_UNIT_TEMPLATES:
        key = _canonical_shape_key(base)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        template_pool.append({"mask": base, "shape_key": key})
    if len(template_pool) < n_non_units:
        return None, None

    forbidden = np.zeros((work_h, work_w), dtype=bool)
    slots: list[dict] = []

    chosen_templates = random.sample(template_pool, n_non_units)
    chosen_templates.sort(key=lambda t: -(t["mask"].shape[0] * t["mask"].shape[1]))
    for templ in chosen_templates:
        placed = None
        for _ in range(TEMPLATE_SLOT_RETRIES):
            oriented = _random_dihedral(templ["mask"])
            coords = _place_mask(forbidden, oriented)
            if coords is None:
                continue
            placed = {"coords": coords, "shape_key": templ["shape_key"]}
            _forbid_coords(forbidden, coords)
            break
        if placed is None:
            return None, None
        slots.append(placed)

    # One unit slot keeps k directly inferable.
    for _ in range(UNIT_SLOT_RETRIES):
        coords = _place_mask(forbidden, np.ones((1, 1), dtype=bool))
        if coords is None:
            continue
        slots.append({"coords": coords, "shape_key": None})
        _forbid_coords(forbidden, coords)
        break
    else:
        return None, None

    slots, layout_mask = _crop_slots_to_union_bbox(slots)
    if slots is None or layout_mask is None:
        return None, None

    density = sum(int(s["area"]) for s in slots) / int(layout_mask.shape[0] * layout_mask.shape[1])
    lo, hi = HOLE_DENSITY_RANGE
    if not (lo <= density <= hi):
        return None, None

    # Variant must be visible: all non-unit slots must contain at least one internal hole.
    non_unit_slots = [s for s in slots if int(s["area"]) > 1]
    if len(non_unit_slots) != n_non_units:
        return None, None
    if not all(int(s["internal_holes"]) >= 1 for s in non_unit_slots):
        return None, None

    return slots, layout_mask


def _make_piece(slot: dict, color: int, k: int, apply_dihedral: bool) -> dict:
    """Build output/input patches directly from the slot's holed block mask."""
    if slot["area"] == 1:
        mask = np.ones((k, k), dtype=bool)
        patch = np.full((k, k), color, dtype=int)
        return {
            "color": int(color),
            "slot": slot,
            "out_patch": patch,
            "out_mask": mask,
            "in_patch": patch,
            "in_mask": mask,
        }

    out_mask = np.kron(slot["mask"], np.ones((k, k), dtype=bool))
    out_patch = np.where(out_mask, int(color), 0).astype(int)

    if apply_dihedral:
        rot = random.randint(0, 3)
        in_mask = np.rot90(out_mask, rot)
        in_patch = np.rot90(out_patch, rot)
        if random.random() < 0.5:
            in_mask = np.fliplr(in_mask)
            in_patch = np.fliplr(in_patch)
    else:
        in_mask = out_mask
        in_patch = out_patch

    return {
        "color": int(color),
        "slot": slot,
        "out_patch": out_patch,
        "out_mask": out_mask,
        "in_patch": in_patch,
        "in_mask": in_mask,
    }


def _build_expected_output(
    pieces: list[dict],
    layout_h: int,
    layout_w: int,
    k: int,
    frame_color: int,
) -> np.ndarray:
    out = np.full(((layout_h + 2) * k, (layout_w + 2) * k), int(frame_color), dtype=int)
    for p in pieces:
        coords = p["slot"]["coords"]
        top = (int(coords[:, 0].min()) + 1) * int(k)
        left = (int(coords[:, 1].min()) + 1) * int(k)
        paste(out, p["out_patch"], top, left, mask=p["out_mask"], overwrite=True)
    return out


def _build_input_canvas(
    pieces: list[dict],
    layout_mask: np.ndarray,
    k: int,
    frame_color: int,
    distractor_color: int,
) -> np.ndarray | None:
    frame_h = int(layout_mask.shape[0]) + 2
    frame_w = int(layout_mask.shape[1]) + 2
    total_piece_area = sum(int(p["in_mask"].sum()) for p in pieces)
    needed_cells = frame_h * frame_w + AREA_BUFFER_MULT * total_piece_area + AREA_BUFFER_CONST

    min_h = max(CANVAS_DIM_RANGE[0], frame_h + 2)
    min_w = max(CANVAS_DIM_RANGE[0], frame_w + 2)
    canvas_h = random.randint(min_h, CANVAS_DIM_RANGE[1])
    canvas_w = random.randint(min_w, CANVAS_DIM_RANGE[1])
    while canvas_h * canvas_w < needed_cells and (canvas_h < CANVAS_DIM_RANGE[1] or canvas_w < CANVAS_DIM_RANGE[1]):
        if canvas_h <= canvas_w and canvas_h < CANVAS_DIM_RANGE[1]:
            canvas_h += 1
        elif canvas_w < CANVAS_DIM_RANGE[1]:
            canvas_w += 1
        else:
            break
    if canvas_h * canvas_w < needed_cells:
        return None

    canvas = np.zeros((canvas_h, canvas_w), dtype=int)

    frame_img = np.full((frame_h, frame_w), int(frame_color), dtype=int)
    frame_img[1:-1, 1:-1] = np.where(layout_mask, 0, int(frame_color))
    fr_max = canvas_h - frame_h - 1
    fc_max = canvas_w - frame_w - 1
    if fr_max < 1 or fc_max < 1:
        return None
    paste(canvas, frame_img, random.randint(1, fr_max), random.randint(1, fc_max))

    # With holed pieces, placement must reserve the full bbox, not only foreground cells.
    for p in sorted(pieces, key=lambda item: -(item["in_patch"].shape[0] * item["in_patch"].shape[1])):
        probe = np.ones_like(p["in_patch"], dtype=int)
        try:
            pr, pc = random_free_location(
                canvas,
                probe,
                transparent=0,
                border=0,
                padding=1,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(canvas, p["in_patch"], pr, pc, mask=p["in_mask"], overwrite=True)

    n_distractors = random.randint(*DISTRACTOR_RANGE)
    placed = 0
    for _ in range(n_distractors * DISTRACTOR_RETRY_MULT):
        if placed >= n_distractors:
            break
        rr = random.randint(0, canvas_h - 1)
        cc = random.randint(0, canvas_w - 1)
        if canvas[rr, cc] != 0:
            continue
        r_lo, r_hi = max(0, rr - 1), min(canvas_h, rr + 2)
        c_lo, c_hi = max(0, cc - 1), min(canvas_w, cc + 2)
        if np.any(canvas[r_lo:r_hi, c_lo:c_hi] != 0):
            continue
        canvas[rr, cc] = int(distractor_color)
        placed += 1
    if placed < MIN_DISTRACTORS:
        return None

    vals, counts = np.unique(canvas, return_counts=True)
    if int(vals[np.argmax(counts)]) != 0:
        return None

    return canvas


def main(grid: np.ndarray) -> np.ndarray:
    """
    Inflate a frame's hole-layout and fill each slot with its matching outside piece.

    Variant-specific detail:
    non-unit pieces are still built from k×k-aligned blocks, but some blocks may be
    absent inside the piece bbox, creating enclosed holes. After block downsampling,
    those holes must still match the frame slot mask exactly.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    assert comps, "input must contain at least one foreground component"
    frame_comp = max(comps, key=lambda d: int(d["area"]))
    fr0, fc0, fr1, fc1 = map(int, frame_comp["bbox"])

    frame_cols = [int(g[int(r), int(c)]) for r, c in frame_comp["coords"]]
    bgc = int(Counter(frame_cols).most_common(1)[0][0])

    sub = crop_bbox(g, (fr0, fc0, fr1, fc1))
    hole = sub == bg
    hole_box = bbox(hole)
    assert hole_box is not None, "frame must enclose at least one bg hole"
    lr0, lc0, lr1, lc1 = map(int, hole_box)
    layout = hole[lr0 : lr1 + 1, lc0 : lc1 + 1]

    slots = []
    for coords in components_mask(layout, connectivity=4):
        rr0, cc0, rr1, cc1 = bbox_coords(coords)
        rr0, cc0, rr1, cc1 = map(int, (rr0, cc0, rr1, cc1))
        sm = np.zeros((int(rr1 - rr0 + 1), int(cc1 - cc0 + 1)), dtype=bool)
        sm[coords[:, 0] - rr0, coords[:, 1] - cc0] = True
        slots.append({"area": int(sm.sum()), "r0": rr0, "c0": cc0, "mask": sm})
    slots.sort(key=lambda s: (-int(s["area"]), int(s["r0"]), int(s["c0"])))
    assert slots, "frame must contain at least one slot"

    pieces = []
    for comp in comps:
        if comp is frame_comp:
            continue
        coords = comp["coords"]
        if coords.size == 0 or int(coords.shape[0]) <= 1:
            continue
        pr0, pc0, pr1, pc1 = map(int, comp["bbox"])
        patch = crop_bbox(g, (pr0, pc0, pr1, pc1))
        pm = np.zeros(patch.shape, dtype=bool)
        pm[coords[:, 0] - pr0, coords[:, 1] - pc0] = True
        pieces.append({"area": int(pm.sum()), "mask": pm, "patch": patch})
    assert pieces, "at least one non-frame piece must exist"

    k = None
    for p in pieces:
        h, w = map(int, p["mask"].shape)
        if h == w and int(p["mask"].sum()) == int(h * w):
            k = int(h) if k is None else min(int(k), int(h))
    assert k is not None and int(k) >= 1, "a unit piece is required to infer k"
    k = int(k)

    def downsample_blocks(mask: np.ndarray) -> np.ndarray | None:
        hh, ww = mask.shape
        if hh % k != 0 or ww % k != 0:
            return None
        out_h = int(hh // k)
        out_w = int(ww // k)
        out = np.zeros((out_h, out_w), dtype=bool)
        for r in range(out_h):
            for c in range(out_w):
                block = mask[r * k : (r + 1) * k, c * k : (c + 1) * k]
                if bool(block.all()):
                    out[r, c] = True
                elif bool(block.any()):
                    return None
        return out

    def dihedral_variants(patch: np.ndarray, mask: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        uniq: list[tuple[np.ndarray, np.ndarray]] = []
        seen: set[tuple] = set()
        for kk in range(4):
            rot_patch = np.rot90(patch, kk)
            rot_mask = np.rot90(mask, kk)
            for vp, vm in ((rot_patch, rot_mask), (np.fliplr(rot_patch), np.fliplr(rot_mask))):
                key = (vm.shape, vm.tobytes())
                if key in seen:
                    continue
                seen.add(key)
                uniq.append((vp, vm))
        return uniq

    candidates = []
    for pi, p in enumerate(pieces):
        for vp, vm in dihedral_variants(p["patch"], p["mask"]):
            um = downsample_blocks(vm)
            if um is None:
                continue
            candidates.append({"pi": int(pi), "area": int(p["area"]), "um": um, "vp": vp, "vm": vm})

    used_piece: set[int] = set()
    placements: list[tuple[dict, dict]] = []
    for s in slots:
        need_area = int(s["area"]) * int(k) * int(k)
        found = None
        for cand in candidates:
            if int(cand["pi"]) in used_piece:
                continue
            if int(cand["area"]) != int(need_area):
                continue
            if cand["um"].shape != s["mask"].shape:
                continue
            if np.array_equal(cand["um"], s["mask"]):
                found = cand
                break
        assert found is not None, "every slot must match a piece"
        used_piece.add(int(found["pi"]))
        placements.append((s, found))

    out_h = (int(layout.shape[0]) + 2) * int(k)
    out_w = (int(layout.shape[1]) + 2) * int(k)
    out = np.full((out_h, out_w), int(bgc), dtype=int)

    for s, cand in placements:
        top = (int(s["r0"]) + 1) * int(k)
        left = (int(s["c0"]) + 1) * int(k)
        paste(out, cand["vp"], top, left, mask=cand["vm"], overwrite=True)

    return out


def _gen_attempt() -> np.ndarray | None:
    k = random.choice(K_CHOICES)
    work_h = random.randint(*WORK_LAYOUT_H_RANGE)
    work_w = random.randint(*WORK_LAYOUT_W_RANGE)
    n_slots = random.randint(*N_SLOTS_RANGE)

    palette = list(range(1, 10))
    random.shuffle(palette)
    frame_color = palette.pop()
    distractor_color = palette.pop()
    needed_piece_colors = (n_slots - N_UNITS) + 1  # +1 shared unit color
    if len(palette) < needed_piece_colors:
        return None
    non_unit_colors = palette[: n_slots - N_UNITS]
    unit_color = palette[n_slots - N_UNITS]

    slots, layout_mask = _gen_layout(work_h, work_w, n_slots, N_UNITS)
    if slots is None or layout_mask is None:
        return None

    non_unit_slots = [s for s in slots if int(s["area"]) > 1]
    unit_slots = [s for s in slots if int(s["area"]) == 1]
    if len(unit_slots) != N_UNITS or len(non_unit_slots) != n_slots - N_UNITS:
        return None

    pieces = [_make_piece(slot, non_unit_colors[i], k, apply_dihedral=True) for i, slot in enumerate(non_unit_slots)]
    pieces += [_make_piece(slot, unit_color, k, apply_dihedral=False) for slot in unit_slots]

    expected_out = _build_expected_output(pieces, int(layout_mask.shape[0]), int(layout_mask.shape[1]), k, frame_color)
    canvas = _build_input_canvas(pieces, layout_mask, k, frame_color, distractor_color)
    if canvas is None:
        return None

    try:
        produced = main(canvas)
    except Exception:
        return None
    if not np.array_equal(produced, expected_out):
        return None

    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_MAX_RETRIES):
        res = _gen_attempt()
        if res is not None:
            return res
    raise RuntimeError(f"generate_input: failed to generate after {GENERATE_MAX_RETRIES} attempts")
