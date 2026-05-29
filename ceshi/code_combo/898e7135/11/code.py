# Combo variant of 898e7135 (index 11)
# combo_spec: D2/1+D8/0
# Name: 嵌套洞无直观k线索

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
  fused_goal:
    keep_constant:
      - 最大前景连通块仍是单色矩形框架
      - 框内背景洞仍按 4 连通拆成 slots，外部 piece 仍与 slot 一一对应
      - 输出仍是把框架 tight-crop 后按统一 k 放大，并把匹配 piece 填回对应位置
    changed_core:
      - D2/1: slot / piece 不再只是实心 polyomino，而是块级掩码中含内部洞的 holed mask
      - D8/0: 样本中完全移除 unit slot，k 不再能从最小 kxk 实心方块直接读出
      - 融合后 solver 必须同时处理二阶洞结构和纯面积比推断 k

  concrete_trace:
    original:
      - 假设某个 slot 是 3 格 L 形，k=3
      - 外部 piece 是 9 像素厚的实心 L；solver 可从某个 unit piece 直接读出 k=3
      - 下采样后只需匹配一阶外轮廓
    fused_variant:
      - 一个 slot 可以是 111/101/111，中心 0 是被 hole 包围的 frame-color 墙岛
      - 外部 piece 是同形状的单色方环，中心保留背景 0；旋转/翻转后散落在框外
      - 因为没有 area=1 的 slot，也就没有 kxk 的直接 clue block；solver 只能先用
        piece_area = slot_area * k^2 在全部 piece/slot 的面积集合上反推唯一 k
      - 得到 k 后，再把每个 piece 按 kxk 块下采样，恢复含内部洞的 slot mask 去匹配
    visible_difference:
      - 原题只需识别“放大实心块形状”；融合版要求识别“放大环形/嵌套洞形状”，且 k 线索消失

  complexity_calibration:
    anti_ambiguity:
      - 非 unit slot 全部来自少量 holed 模板库，模板在 D4 规范形下互异，避免同形多色歧义
      - k 只取 {2, 3}，但所有 piece 都带内部洞，因此旧的最小实心方块启发式恒失效
      - slot 数量固定为 3，保证每个样例都有多块可对照，同时不把 30x30 画布挤满
    anti_simplification:
      - 每个样例 0 个 unit slot
      - 每个真实 piece 至少 1 个内部洞
      - 生成后显式检查 old_square_heuristic_k != true_k，防止退化回原题

  implicit_assumption_audit:
    solver_lines:
      - frame_comp = max(comps, key=area)
        assumption: 框架仍是最大前景连通块
        guarantee: 生成器只接受 frame_area > max(piece_area) 的样本
      - hole = sub == bg ; layout = tight-cropped hole mask
        assumption: slot 可以由 bool mask 表达，bbox 内允许存在被包围的 False 墙岛
        guarantee: holed 模板正是通过这个表示进入框内布局
      - components_mask(layout, connectivity=4)
        assumption: slot 的外层 True 区域保持 4 连通；内部 False 洞不会把 slot 切断
        guarantee: 模板库只使用 4 连通且含内部洞的 mask
      - piece mask 从 component coords 构建
        assumption: 外部 piece bbox 内允许背景孔洞
        guarantee: piece 直接由 kron(slot_mask, ones(k,k)) 生成，孔洞天然保留；
                   solver 只收集 frame bbox 外的非单像素组件，因此不会把框内墙岛误判成 piece
      - len(pieces) == len(slots)
        assumption: 除 frame 外的非单像素前景分量全都是真实 piece
        guarantee: 干扰仍只使用单像素，真实 piece 全部面积 > 1，框内 wall islands 被 bbox 规则排除
      - area-ratio inference for k
        assumption: 所有 piece 共用一个 k，且 piece_area = slot_area * k^2
        guarantee: 所有 piece 都由同一 k 的 block upscale 构造
      - downsample_blocks
        assumption: 每个 kxk 小块非全满即全空，不会出现半块
        guarantee: piece 由整块 kron 扩张得到
      - dihedral matching
        assumption: D4 去重不能只看 bytes，要连同 shape 一起看
        guarantee: solver / generator 都使用 (shape, bytes) 规范 key

  placement_legality:
    layout_side:
      - 在 work layout 上放置 tight holed mask，只对 True 单元做 8 邻域禁放
      - 不同 slots 因 8 分离而不会接触或合并；内部墙岛不参与占位
      - 最终裁成所有 holes 的联合 bbox，与 solver 的 tight-crop 规则一致
    canvas_side:
      - 先放框架，再放 pieces
      - holed piece 的摆放用整 bbox 的 probe patch 预占位，避免别的对象落进它的洞里
      - 单像素干扰点只放在 8 邻域全空位置，因此不会钻进 piece 洞或贴着框架

  observability_and_cleanliness:
    observability:
      - 人眼能直接看到外部 piece 是“带洞的单色环/嵌套形”，框内对应位置也有被包围的小墙岛
      - 又因为没有 unit slot，看不到任何直接暴露 k 的最小方块，只能从整体尺度关系推断
    cleanliness:
      - 颜色角色仍简单：frame、distractor、piece colors
      - 对象不重叠，piece 之间与框架之间至少隔 1 个背景格，视觉上可分辨

  quantitative_fusion_checks:
    metrics:
      - unit_slot_count == 0
      - min(piece_internal_holes) >= 1
      - old_square_heuristic_k == 1 and true_k in {2, 3}
    meaning:
      - 这三项同时成立时，样本确实触发了“嵌套洞 + 无直观 k 线索”的融合行为，而不是退化回原题

  new_main_assumptions_and_generator_guarantees:
    - assumption: slot 的 D4 规范形两两可区分
      guarantee: 模板采样按 canonical key 去重
    - assumption: frame 内部至少有一个 slot，且所有 slot 都是非 unit
      guarantee: n_slots == 3，模板库全部面积 > 1
    - assumption: 背景 0 仍是全局最多色
      guarantee: 画布面积预算保守，生成后再次检查
"""


K_CHOICES = [2, 3]
WORK_LAYOUT_H_RANGE = (6, 12)
WORK_LAYOUT_W_RANGE = (6, 9)
N_SLOTS_RANGE = (3, 3)

HOLE_DENSITY_RANGE = (0.20, 0.55)
TEMPLATE_SLOT_RETRIES = 40

CANVAS_DIM_RANGE = (18, 30)
AREA_BUFFER_MULT = 2
AREA_BUFFER_CONST = 60
FRAME_INNER_PAD_RANGE = (0, 4)
FRAME_BUILD_RETRIES = 24

DISTRACTOR_RANGE = (8, 13)
DISTRACTOR_RETRY_MULT = 10
MIN_DISTRACTORS = 8

GENERATE_MAX_RETRIES = 1500


def _mask_from_rows(rows: list[str]) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


NON_UNIT_TEMPLATE_ROWS = [
    ["111", "101", "111"],
    ["1111", "1001", "1111"],
    ["111", "101", "111", "010"],
    ["1110", "1011", "1110"],
    ["1110", "1011", "1111", "0110"],
    ["0111", "1101", "1111", "0111"],
    ["0110", "1111", "1011", "1110"],
    ["01110", "11111", "11011", "01110"],
]

NON_UNIT_TEMPLATES = [_mask_from_rows(rows) for rows in NON_UNIT_TEMPLATE_ROWS]


def _count_internal_holes(mask: np.ndarray) -> int:
    hh, ww = map(int, mask.shape)
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


def _canonical_shape_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    variants = []
    for kk in range(4):
        rot = np.rot90(mask, kk)
        variants.append((rot.shape, rot.tobytes()))
        flip = np.fliplr(rot)
        variants.append((flip.shape, flip.tobytes()))
    return min(variants)


def _dihedral_variants(mask: np.ndarray) -> list[np.ndarray]:
    uniq: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
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
    H, W = map(int, forbidden.shape)
    h, w = map(int, mask.shape)
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
    return np.argwhere(mask) + np.array([top, left], dtype=int)


def _crop_slots_to_union_bbox(slots: list[dict]) -> tuple[list[dict], np.ndarray] | tuple[None, None]:
    if not slots:
        return None, None

    layout_mask = None
    for slot in slots:
        coords = slot["coords"]
        if layout_mask is None:
            nr = int(coords[:, 0].max()) + 1
            nc = int(coords[:, 1].max()) + 1
            layout_mask = np.zeros((nr, nc), dtype=bool)
        else:
            nr = max(int(layout_mask.shape[0]), int(coords[:, 0].max()) + 1)
            nc = max(int(layout_mask.shape[1]), int(coords[:, 1].max()) + 1)
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
        rr0, cc0, rr1, cc1 = map(int, bbox_coords(coords))
        sm = np.zeros((rr1 - rr0 + 1, cc1 - cc0 + 1), dtype=bool)
        sm[coords[:, 0] - rr0, coords[:, 1] - cc0] = True
        new_slots.append(
            {
                "coords": coords,
                "area": int(sm.sum()),
                "mask": sm,
                "shape_key": slot["shape_key"],
                "internal_holes": int(_count_internal_holes(sm)),
            }
        )
    return new_slots, cropped


def _gen_layout(
    work_h: int,
    work_w: int,
    n_slots: int,
    templates: list[np.ndarray] | None = None,
) -> tuple[list[dict], np.ndarray] | tuple[None, None]:
    source_templates = NON_UNIT_TEMPLATES if templates is None else templates
    if n_slots <= 0 or n_slots > len(source_templates):
        return None, None

    template_pool = []
    seen_keys: set[tuple[tuple[int, int], bytes]] = set()
    for base in source_templates:
        key = _canonical_shape_key(base)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        template_pool.append({"mask": base, "shape_key": key})
    if len(template_pool) < n_slots:
        return None, None

    forbidden = np.zeros((work_h, work_w), dtype=bool)
    slots: list[dict] = []

    chosen_templates = random.sample(template_pool, n_slots)
    chosen_templates.sort(key=lambda item: -(item["mask"].shape[0] * item["mask"].shape[1]))
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

    slots, layout_mask = _crop_slots_to_union_bbox(slots)
    if slots is None or layout_mask is None:
        return None, None

    density = sum(int(slot["area"]) for slot in slots) / int(layout_mask.shape[0] * layout_mask.shape[1])
    lo, hi = HOLE_DENSITY_RANGE
    if not (lo <= density <= hi):
        return None, None
    if any(int(slot["area"]) <= 1 for slot in slots):
        return None, None
    if not all(int(slot["internal_holes"]) >= 1 for slot in slots):
        return None, None

    return slots, layout_mask


def _make_piece(slot: dict, color: int, k: int) -> dict:
    out_mask = np.kron(slot["mask"], np.ones((k, k), dtype=bool))
    out_patch = np.where(out_mask, int(color), 0).astype(int)

    in_mask = out_mask
    in_patch = out_patch
    rot = random.randint(0, 3)
    in_mask = np.rot90(in_mask, rot)
    in_patch = np.rot90(in_patch, rot)
    if random.random() < 0.5:
        in_mask = np.fliplr(in_mask)
        in_patch = np.fliplr(in_patch)

    return {
        "color": int(color),
        "slot": slot,
        "out_patch": out_patch,
        "out_mask": out_mask,
        "in_patch": in_patch,
        "in_mask": in_mask,
        "internal_holes": int(_count_internal_holes(slot["mask"])),
    }


def _build_expected_output(pieces: list[dict], layout_h: int, layout_w: int, k: int, frame_color: int) -> np.ndarray:
    out = np.full(((layout_h + 2) * k, (layout_w + 2) * k), int(frame_color), dtype=int)
    for piece in pieces:
        coords = piece["slot"]["coords"]
        top = (int(coords[:, 0].min()) + 1) * int(k)
        left = (int(coords[:, 1].min()) + 1) * int(k)
        paste(out, piece["out_patch"], top, left, mask=piece["out_mask"], overwrite=True)
    return out


def _build_input_canvas(
    pieces: list[dict],
    layout_mask: np.ndarray,
    frame_color: int,
    distractor_color: int,
) -> np.ndarray | None:
    total_piece_area = sum(int(piece["in_mask"].sum()) for piece in pieces)
    max_piece_area = max(int(piece["in_mask"].sum()) for piece in pieces)
    layout_h = int(layout_mask.shape[0])
    layout_w = int(layout_mask.shape[1])

    for _ in range(FRAME_BUILD_RETRIES):
        pad_top = random.randint(*FRAME_INNER_PAD_RANGE)
        pad_bottom = random.randint(*FRAME_INNER_PAD_RANGE)
        pad_left = random.randint(*FRAME_INNER_PAD_RANGE)
        pad_right = random.randint(*FRAME_INNER_PAD_RANGE)

        frame_h = layout_h + pad_top + pad_bottom + 2
        frame_w = layout_w + pad_left + pad_right + 2
        frame_area = frame_h * frame_w - int(layout_mask.sum())
        if frame_area <= max_piece_area:
            continue

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
            continue

        canvas = np.zeros((canvas_h, canvas_w), dtype=int)

        frame_img = np.full((frame_h, frame_w), int(frame_color), dtype=int)
        r0 = 1 + pad_top
        c0 = 1 + pad_left
        frame_img[r0 : r0 + layout_h, c0 : c0 + layout_w] = np.where(layout_mask, 0, int(frame_color))
        fr_max = canvas_h - frame_h - 1
        fc_max = canvas_w - frame_w - 1
        if fr_max < 1 or fc_max < 1:
            continue
        paste(canvas, frame_img, random.randint(1, fr_max), random.randint(1, fc_max))

        failed = False
        for piece in sorted(pieces, key=lambda item: -(item["in_patch"].shape[0] * item["in_patch"].shape[1])):
            probe = np.ones_like(piece["in_patch"], dtype=int)
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
                failed = True
                break
            paste(canvas, piece["in_patch"], pr, pc, mask=piece["in_mask"], overwrite=True)
        if failed:
            continue

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
            continue

        vals, counts = np.unique(canvas, return_counts=True)
        if int(vals[np.argmax(counts)]) != 0:
            continue

        return canvas

    return None


def _old_square_heuristic_k_from_pieces(pieces: list[dict]) -> int:
    guess = None
    for piece in pieces:
        h, w = map(int, piece["in_mask"].shape)
        if h == w and int(piece["in_mask"].sum()) == h * w:
            guess = h if guess is None else min(int(guess), h)
    return int(guess or 1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Scale a frame's holed slot layout and refill it with the matching exterior sprites.

    Input structure:
    - The most common color is background. The largest foreground component is a monochrome frame.
    - Inside that frame, background cells form 4-connected slots; some slot bounding boxes contain enclosed frame-color wall islands.
    - Outside the frame, each real piece is a monochrome k-by-k block expansion of one slot, possibly rotated/flipped, and pieces may keep visible background holes inside their own bounding boxes.

    Rule:
    - Tight-crop the frame's interior hole layout, extract each slot mask including its nested wall pattern, and collect all non-singleton exterior pieces.
    - Infer the shared scale factor k from the global area relation piece_area = slot_area * k^2, since no unit slot gives a direct clue.
    - Downsample each piece in k-by-k blocks, match its block mask to a slot under dihedral symmetry, then paste the matched oriented piece into the k-scaled frame.

    Key insight:
    - The internal holes matter at the block level: a piece's empty blocks correspond to frame-colored wall islands inside the matching slot.
    - Because the smallest-square clue is absent, k must be recovered from the whole slot/piece area multiset before the holed masks can be compared.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    assert comps, "expected frame plus pieces"
    frame_comp = max(comps, key=lambda comp: int(comp["area"]))
    fr0, fc0, fr1, fc1 = map(int, frame_comp["bbox"])

    frame_cols = [int(g[int(r), int(c)]) for r, c in frame_comp["coords"]]
    frame_color = int(Counter(frame_cols).most_common(1)[0][0])

    sub = crop_bbox(g, (fr0, fc0, fr1, fc1))
    hole = sub == bg
    hole_box = bbox(hole)
    assert hole_box is not None, "frame must enclose holes"
    lr0, lc0, lr1, lc1 = map(int, hole_box)
    layout = hole[lr0 : lr1 + 1, lc0 : lc1 + 1]

    slots = []
    for coords in components_mask(layout, connectivity=4):
        rr0, cc0, rr1, cc1 = map(int, bbox_coords(coords))
        sm = np.zeros((rr1 - rr0 + 1, cc1 - cc0 + 1), dtype=bool)
        sm[coords[:, 0] - rr0, coords[:, 1] - cc0] = True
        slots.append({"area": int(sm.sum()), "r0": rr0, "c0": cc0, "mask": sm})
    slots.sort(key=lambda slot: (-int(slot["area"]), int(slot["r0"]), int(slot["c0"])))
    assert slots, "frame must contain at least one slot"

    pieces = []
    for comp in comps:
        if comp is frame_comp:
            continue
        coords = comp["coords"]
        if coords.size == 0 or int(coords.shape[0]) <= 1:
            continue
        pr0, pc0, pr1, pc1 = map(int, comp["bbox"])
        # Holed slots create disconnected frame-color wall islands inside the frame bbox.
        # They belong to the frame layout, not to the external piece set.
        if fr0 <= pr0 <= pr1 <= fr1 and fc0 <= pc0 <= pc1 <= fc1:
            continue
        patch = crop_bbox(g, (pr0, pc0, pr1, pc1))
        pm = np.zeros(patch.shape, dtype=bool)
        pm[coords[:, 0] - pr0, coords[:, 1] - pc0] = True
        pieces.append({"area": int(pm.sum()), "mask": pm, "patch": patch})

    assert pieces, "at least one non-frame piece must exist"
    assert len(pieces) == len(slots), "piece/slot count mismatch"

    slot_areas = sorted(int(slot["area"]) for slot in slots)
    piece_areas = sorted(int(piece["area"]) for piece in pieces)
    max_piece_span = max(max(int(piece["mask"].shape[0]), int(piece["mask"].shape[1])) for piece in pieces)
    k_candidates = []
    for k in range(2, max_piece_span + 1):
        if sorted(int(area) * int(k) * int(k) for area in slot_areas) == piece_areas:
            k_candidates.append(int(k))
    assert len(k_candidates) == 1, "k must be uniquely determined by area ratios"
    k = int(k_candidates[0])

    def downsample_blocks(mask: np.ndarray) -> np.ndarray | None:
        hh, ww = map(int, mask.shape)
        if hh % k != 0 or ww % k != 0:
            return None
        out = np.zeros((hh // k, ww // k), dtype=bool)
        for r in range(hh // k):
            for c in range(ww // k):
                block = mask[r * k : (r + 1) * k, c * k : (c + 1) * k]
                if bool(block.all()):
                    out[r, c] = True
                elif bool(block.any()):
                    return None
        return out

    def dihedral_variants(patch: np.ndarray, mask: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        uniq: list[tuple[np.ndarray, np.ndarray]] = []
        seen: set[tuple[tuple[int, int], bytes]] = set()
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
    for piece_index, piece in enumerate(pieces):
        for vp, vm in dihedral_variants(piece["patch"], piece["mask"]):
            um = downsample_blocks(vm)
            if um is None:
                continue
            candidates.append(
                {"pi": int(piece_index), "area": int(piece["area"]), "um": um, "vp": vp, "vm": vm}
            )

    used_piece: set[int] = set()
    placements: list[tuple[dict, dict]] = []
    for slot in slots:
        need_area = int(slot["area"]) * int(k) * int(k)
        found = None
        for cand in candidates:
            if int(cand["pi"]) in used_piece:
                continue
            if int(cand["area"]) != int(need_area):
                continue
            if cand["um"].shape != slot["mask"].shape:
                continue
            if np.array_equal(cand["um"], slot["mask"]):
                found = cand
                break
        assert found is not None, "every slot must match a piece"
        used_piece.add(int(found["pi"]))
        placements.append((slot, found))

    out = np.full(((layout.shape[0] + 2) * k, (layout.shape[1] + 2) * k), int(frame_color), dtype=int)
    for slot, cand in placements:
        top = (int(slot["r0"]) + 1) * int(k)
        left = (int(slot["c0"]) + 1) * int(k)
        paste(out, cand["vp"], top, left, mask=cand["vm"], overwrite=True)
    return out


def _gen_attempt() -> np.ndarray | None:
    k = random.choice(K_CHOICES)
    work_h = random.randint(*WORK_LAYOUT_H_RANGE)
    work_w = random.randint(*WORK_LAYOUT_W_RANGE)
    n_slots = random.randint(*N_SLOTS_RANGE)
    candidate_templates = NON_UNIT_TEMPLATES
    if k == 3:
        n_slots = 3
        candidate_templates = [
            template
            for template in NON_UNIT_TEMPLATES
            if max(map(int, template.shape)) <= 4 and int(template.sum()) <= 10
        ]

    palette = list(range(1, 10))
    random.shuffle(palette)
    frame_color = palette.pop()
    distractor_color = palette.pop()
    if len(palette) < n_slots:
        return None
    piece_colors = palette[:n_slots]

    slots, layout_mask = _gen_layout(work_h, work_w, n_slots, templates=candidate_templates)
    if slots is None or layout_mask is None:
        return None

    pieces = [_make_piece(slot, piece_colors[i], k) for i, slot in enumerate(slots)]
    if any(int(piece["internal_holes"]) < 1 for piece in pieces):
        return None
    if _old_square_heuristic_k_from_pieces(pieces) == int(k):
        return None

    expected_out = _build_expected_output(
        pieces,
        int(layout_mask.shape[0]),
        int(layout_mask.shape[1]),
        k,
        frame_color,
    )
    canvas = _build_input_canvas(pieces, layout_mask, frame_color, distractor_color)
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
