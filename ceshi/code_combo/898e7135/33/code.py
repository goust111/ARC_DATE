# Combo variant of 898e7135 (index 33)
# combo_spec: D1/1+D2/1
# Name: 含洞旋转匹配

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
    preserved_backbone:
      - 输入仍是一个最大前景连通块作为单色框架，框架外散落若干可匹配精灵
      - 输出仍是将框架内布局按 k 放大，并把对应精灵放回各自洞位
      - k 仍由至少一个 unit piece 的 kxk 实心方块直接暴露出来
    fused_changes:
      - D2/1: 非单元 slot / piece 的块级掩码允许内部缺块，因此 tight bbox 内会出现被包围的 frame-color 墙岛或 sprite 空洞
      - D1/1: 匹配时只允许 4 个旋转，不允许翻转；镜像手性必须保留

  concrete_behavior_trace:
    example:
      slot_A_mask:
        - "1110"
        - "1011"
        - "0110"
      slot_B_mask:
        - slot_A 的水平镜像
      k: 2
    original_task_behavior:
      - 若精灵是实心块，solver 只看外轮廓，slot_A 与 slot_B 在允许翻转时会被视作同一个二面体类
      - 旧 generator 也可能完全不生成镜像对，因此“禁翻转”在样例里不一定可观察
    fused_behavior:
      - 框架内两个 slot 的 tight bbox 中都保留一个内部墙岛
      - 框架外两个精灵都是带洞的放大版，一个对应 slot_A，一个对应 slot_B
      - solver 先按 kxk 下采样恢复完整块级掩码，再只在旋转群内比较
      - 因为 slot_A 与 slot_B 互为镜像但不旋转等价，所以必须区分左右手性
    visible_difference:
      - 观察者会同时看到“精灵本身有洞”和“镜像形状不能互换”这两层信息

  complexity_calibration:
    anti_degeneration:
      - 每题固定 1 个 unit slot，避免 k 推断退化
      - 每个非单元 slot 都要求至少 1 个内部洞，避免退化回原始实心块任务
      - 每题强制放入 1 组 chiral mirror pair，避免“仅旋转”只停留在代码层而不出现在样例里
    ambiguity_control:
      - 所有非单元 slot 的 rotation canonical key 两两不同
      - 强制镜像对只在旋转意义下不同，不再引入额外颜色规则
      - 非单元颜色互异，unit piece 单独共享色需求被简化为仅 1 个 unit piece
    bounded_freedom:
      - 含洞形状来自小型模板库并只做旋转，成功率更稳定
      - 总 slot 数限制在 3-5，既能展示融合行为，又不会让画面过密

  implicit_assumption_audit_for_main:
    - find_components(..., monochromatic=False):
        assumption: 框架仍是最大前景连通块
        generator_guarantee: frame wall area is checked to exceed every piece foreground area, and objects are 8-separated
    - hole = sub == bg ; layout = tight crop of hole union:
        assumption: slot 的信息可以只靠背景布尔掩码表达，即使 bbox 内有内部墙岛也不需要额外对象类型
        status: 成立；内部洞恰好表现为 slot mask 内的 False 岛
    - components_mask(layout, connectivity=4):
        assumption: 含洞 slot 只要外层仍 4 连通，就会被当作单个组件
        status: 成立；内部墙岛不会把 ring-like slot 切成多个组件
    - piece mask built from foreground coords inside bbox:
        assumption: 外部精灵 bbox 内允许保留背景孔洞
        status: 成立；coords 只记录前景像素，孔洞自然留空
    - infer k from smallest solid square piece:
        assumption: 至少存在一个 unit piece，且非单元精灵都不是实心方块
        generator_guarantee: exactly one unit piece is always generated, and every non-unit mask has internal holes
    - downsample_blocks:
        assumption: 每个 kxk 子块必须全满或全空，不能出现半块
        generator_guarantee: every piece is kron(mask, ones(k,k)) followed only by rotation
    - transform enumeration:
        old_assumption: flips are legal matching transforms
        fused_status: 不成立；solver 必须只枚举 4 个旋转
    - first exact unused match wins:
        assumption: 每个 slot 在未使用 piece 中有唯一 rotation-class 对应项
        generator_guarantee: non-unit rotation keys are unique, and the forced mirror pair remains distinct because flips are forbidden

  placement_legality:
    layout:
      - slot 在工作布局上仅按 mask=True 的格子占位
      - 每次放置后对真实洞格做 8 邻域禁放，保证不同 slot 之间不接触
      - 最后裁剪到所有洞格联合 bbox，确保 solver 看到的 tight crop 与生成布局一致
    canvas:
      - 先放 frame，再放 pieces，再放单像素干扰点
      - 含洞 piece 用整 bbox probe 选位，防止别的对象钻入精灵孔洞
      - distractor 只落在周围 8 邻域全空处，不与 frame / pieces 接触

  visual_observability:
    - 每个非单元精灵肉眼都可见内部空洞，对应 frame 中也能看到内部墙岛
    - 至少一对非单元 slot 互为镜像但不旋转等价，因此“禁止翻转”有直接像素证据
    - 颜色只承担对象身份区分，不参与不可见排序

  cleanliness:
    - 不叠放对象，不让干扰点进入精灵洞内，不让精灵与框架相交
    - slot 数量、模板尺寸和画布缓冲都受控，避免高密度杂乱布局

  new_main_assumptions_and_generator_guarantees:
    - assumption: 非单元 piece 的 bbox 可包含背景孔洞
      guarantee: all non-unit pieces are created from holed block masks
    - assumption: 至少有一组镜像但非旋转等价的 slot
      guarantee: layout generation always places one explicit chiral pair
    - assumption: 旧版允许翻转的 solver 在这些样例上会失去区分力
      guarantee: the chiral pair is built from a holed base whose mirror has a different rotation key
"""


RotKey = tuple[tuple[int, int], bytes]


# --- tunables ----------------------------------------------------------------

K_CHOICES = [2, 3]
WORK_LAYOUT_H_RANGE = (6, 12)
WORK_LAYOUT_W_RANGE = (6, 9)
N_SLOTS_RANGE = (3, 5)  # includes the single unit slot
N_UNITS = 1

HOLE_DENSITY_RANGE = (0.20, 0.55)
TEMPLATE_SLOT_RETRIES = 50
UNIT_SLOT_RETRIES = 50

CANVAS_DIM_RANGE = (18, 30)
AREA_BUFFER_MULT = 5
AREA_BUFFER_CONST = 60

DISTRACTOR_RANGE = (8, 13)
DISTRACTOR_RETRY_MULT = 10
MIN_DISTRACTORS = 8

GENERATE_MAX_RETRIES = 500


def _mask_from_rows(rows: list[str]) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


def _rotation_shape_key(mask: np.ndarray) -> RotKey:
    variants: list[RotKey] = []
    for kk in range(4):
        rot = np.rot90(mask, int(kk))
        variants.append((tuple(map(int, rot.shape)), rot.tobytes()))
    return min(variants)


def _count_internal_holes(mask: np.ndarray) -> int:
    hh, ww = map(int, mask.shape)
    total = 0
    for coords in components_mask(~mask, connectivity=4):
        if coords.size == 0:
            continue
        rs = coords[:, 0]
        cs = coords[:, 1]
        if np.any(rs == 0) or np.any(rs == hh - 1) or np.any(cs == 0) or np.any(cs == ww - 1):
            continue
        total += 1
    return int(total)


def _random_rotation(mask: np.ndarray) -> np.ndarray:
    return np.rot90(mask, random.randint(0, 3))


def _tight_mask_from_coords(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = map(int, bbox_coords(coords))
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    out[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return out


def _slot_from_coords(coords: np.ndarray) -> dict:
    mask = _tight_mask_from_coords(coords)
    r0, c0, _r1, _c1 = map(int, bbox_coords(coords))
    return {
        "coords": coords,
        "area": int(coords.shape[0]),
        "mask": mask,
        "r0": int(r0),
        "c0": int(c0),
        "rot_key": _rotation_shape_key(mask),
        "internal_holes": int(_count_internal_holes(mask)),
    }


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
    """Place a tight mask on the work layout, ignoring its internal holes."""
    H, W = map(int, forbidden.shape)
    h, w = map(int, mask.shape)
    if h > H or w > W:
        return None

    candidates: list[tuple[int, int]] = []
    base_coords = np.argwhere(mask)
    for top in range(H - h + 1):
        for left in range(W - w + 1):
            coords = base_coords + np.array([top, left], dtype=int)
            if bool(forbidden[coords[:, 0], coords[:, 1]].any()):
                continue
            candidates.append((int(top), int(left)))

    if not candidates:
        return None
    top, left = random.choice(candidates)
    return base_coords + np.array([top, left], dtype=int)


def _crop_slots_to_union_bbox(slots: list[dict]) -> tuple[list[dict], np.ndarray] | tuple[None, None]:
    if not slots:
        return None, None

    layout_mask = None
    for slot in slots:
        coords = slot["coords"]
        if layout_mask is None:
            layout_mask = np.zeros((int(coords[:, 0].max()) + 1, int(coords[:, 1].max()) + 1), dtype=bool)
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

    cropped_slots = []
    for slot in slots:
        coords = slot["coords"] - np.array([lr0, lc0], dtype=int)
        cropped_slots.append(_slot_from_coords(coords))

    return cropped_slots, cropped


def _count_rotation_mirror_pairs(slots: list[dict]) -> int:
    count = 0
    for i in range(len(slots)):
        mirror_key = _rotation_shape_key(np.fliplr(slots[i]["mask"]))
        if mirror_key == slots[i]["rot_key"]:
            continue
        for j in range(i + 1, len(slots)):
            if slots[j]["rot_key"] == mirror_key:
                count += 1
    return int(count)


CHIRAL_PAIR_TEMPLATE_ROWS = [
    ["1110", "1011", "0110"],
    ["1111", "1010", "0110"],
]

EXTRA_TEMPLATE_ROWS = [
    ["111", "101", "111"],
    ["1111", "1001", "1111"],
    ["01110", "11111", "11011", "01110"],
    ["0110", "1111", "1011", "1110"],
]


def _build_template_masks() -> tuple[list[tuple[np.ndarray, np.ndarray]], list[np.ndarray]]:
    pair_masks: list[tuple[np.ndarray, np.ndarray]] = []
    for rows in CHIRAL_PAIR_TEMPLATE_ROWS:
        base = _mask_from_rows(rows)
        mirror = np.fliplr(base)
        assert _count_internal_holes(base) >= 1
        assert _count_internal_holes(mirror) >= 1
        assert _rotation_shape_key(base) != _rotation_shape_key(mirror)
        pair_masks.append((base, mirror))

    extras: list[np.ndarray] = []
    seen: set[RotKey] = set()
    for rows in EXTRA_TEMPLATE_ROWS:
        mask = _mask_from_rows(rows)
        assert _count_internal_holes(mask) >= 1
        key = _rotation_shape_key(mask)
        if key in seen:
            continue
        seen.add(key)
        extras.append(mask)

    assert len(extras) >= 2
    return pair_masks, extras


CHIRAL_PAIR_MASKS, EXTRA_TEMPLATE_MASKS = _build_template_masks()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Inflate a frame's holed slot-layout and refill it with the matching outside sprites.

    Input structure:
    - Background is the most common color. The largest foreground component is a monochrome rectangular frame.
    - Inside that frame, each slot is a 4-connected background region whose tight bbox may contain enclosed frame-colored wall islands.
    - Outside the frame lie monochrome sprites made of kxk-aligned blocks; non-unit sprites may also contain enclosed background holes. A solid kxk unit sprite is present so k can be inferred.

    Rule:
    - Tight-crop the frame's background layout, split it into slot components, and keep each slot's full block mask including any internal holes.
    - Recover each outside sprite's block mask by downsampling its pixels in kxk chunks.
    - Match a sprite to a slot only if some rotation of that sprite matches the slot mask exactly; flips are forbidden.
    - Output the cropped frame layout scaled by k, using the frame color as the background and pasting each matched rotated sprite into its slot.

    Key insight:
    The solver must preserve two kinds of structure at once: internal holes inside each block mask and the handedness of chiral masks. Looking only at outer silhouettes or allowing mirror flips loses necessary information.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    assert comps, "input must contain a frame component"
    frame_comp = max(comps, key=lambda d: int(d["area"]))
    fr0, fc0, fr1, fc1 = map(int, frame_comp["bbox"])

    frame_cols = [int(g[int(r), int(c)]) for r, c in frame_comp["coords"]]
    frame_color = int(Counter(frame_cols).most_common(1)[0][0])

    sub = crop_bbox(g, (fr0, fc0, fr1, fc1))
    hole = sub == int(bg)
    hole_box = bbox(hole)
    assert hole_box is not None, "frame must enclose at least one background hole"
    lr0, lc0, lr1, lc1 = map(int, hole_box)
    layout = hole[lr0 : lr1 + 1, lc0 : lc1 + 1]

    slots = []
    for coords in components_mask(layout, connectivity=4):
        rr0, cc0, rr1, cc1 = map(int, bbox_coords(coords))
        sm = np.zeros((rr1 - rr0 + 1, cc1 - cc0 + 1), dtype=bool)
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
    for piece in pieces:
        h, w = map(int, piece["mask"].shape)
        if h == w and int(piece["mask"].sum()) == int(h * w):
            k = int(h) if k is None else min(int(k), int(h))
    assert k is not None and int(k) >= 1, "a unit piece is required to infer k"
    k = int(k)

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

    def rotation_variants(patch: np.ndarray, mask: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        uniq: list[tuple[np.ndarray, np.ndarray]] = []
        seen: set[RotKey] = set()
        for kk in range(4):
            vp = np.rot90(patch, int(kk))
            vm = np.rot90(mask, int(kk))
            key = (tuple(map(int, vm.shape)), vm.tobytes())
            if key in seen:
                continue
            seen.add(key)
            uniq.append((vp, vm))
        return uniq

    candidates = []
    for pi, piece in enumerate(pieces):
        for vp, vm in rotation_variants(piece["patch"], piece["mask"]):
            um = downsample_blocks(vm)
            if um is None:
                continue
            candidates.append({"pi": int(pi), "area": int(piece["area"]), "um": um, "vp": vp, "vm": vm})

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
        assert found is not None, "every slot must match a piece under rotations only"
        used_piece.add(int(found["pi"]))
        placements.append((slot, found))

    out_h = (int(layout.shape[0]) + 2) * int(k)
    out_w = (int(layout.shape[1]) + 2) * int(k)
    out = np.full((out_h, out_w), int(frame_color), dtype=int)

    for slot, cand in placements:
        top = (int(slot["r0"]) + 1) * int(k)
        left = (int(slot["c0"]) + 1) * int(k)
        paste(out, cand["vp"], int(top), int(left), mask=cand["vm"], overwrite=True)

    return out


def _gen_layout(work_h: int, work_w: int, n_slots: int) -> tuple[list[dict], np.ndarray] | tuple[None, None]:
    n_non_units = int(n_slots - N_UNITS)
    if n_non_units < 2:
        return None, None

    forbidden = np.zeros((work_h, work_w), dtype=bool)
    raw_slots: list[dict] = []
    used_rot_keys: set[RotKey] = set()

    pair_base, pair_mirror = random.choice(CHIRAL_PAIR_MASKS)
    pair_oriented = [_random_rotation(pair_base), _random_rotation(pair_mirror)]
    pair_oriented.sort(key=lambda mask: -(int(mask.shape[0]) * int(mask.shape[1])))
    for mask in pair_oriented:
        placed = None
        for _ in range(TEMPLATE_SLOT_RETRIES):
            coords = _place_mask(forbidden, mask)
            if coords is None:
                continue
            placed = _slot_from_coords(coords)
            if placed["rot_key"] in used_rot_keys:
                placed = None
                continue
            _forbid_coords(forbidden, coords)
            raw_slots.append(placed)
            used_rot_keys.add(placed["rot_key"])
            break
        if placed is None:
            return None, None

    extra_masks = EXTRA_TEMPLATE_MASKS[:]
    random.shuffle(extra_masks)
    for mask in extra_masks:
        if len(raw_slots) >= n_non_units:
            break
        key = _rotation_shape_key(mask)
        if key in used_rot_keys:
            continue
        placed = None
        for _ in range(TEMPLATE_SLOT_RETRIES):
            oriented = _random_rotation(mask)
            coords = _place_mask(forbidden, oriented)
            if coords is None:
                continue
            placed = _slot_from_coords(coords)
            if placed["rot_key"] in used_rot_keys:
                placed = None
                continue
            _forbid_coords(forbidden, coords)
            raw_slots.append(placed)
            used_rot_keys.add(placed["rot_key"])
            break
        if placed is None:
            continue
    if len(raw_slots) != n_non_units:
        return None, None

    for _ in range(UNIT_SLOT_RETRIES):
        coords = _place_mask(forbidden, np.ones((1, 1), dtype=bool))
        if coords is None:
            continue
        _forbid_coords(forbidden, coords)
        raw_slots.append(_slot_from_coords(coords))
        break
    else:
        return None, None

    slots, layout_mask = _crop_slots_to_union_bbox(raw_slots)
    if slots is None or layout_mask is None:
        return None, None

    density = float(sum(int(slot["area"]) for slot in slots)) / float(layout_mask.shape[0] * layout_mask.shape[1])
    lo, hi = HOLE_DENSITY_RANGE
    if not (lo <= density <= hi):
        return None, None

    non_unit_slots = [slot for slot in slots if int(slot["area"]) > 1]
    if len(non_unit_slots) != n_non_units:
        return None, None
    if not all(int(slot["internal_holes"]) >= 1 for slot in non_unit_slots):
        return None, None
    if len({slot["rot_key"] for slot in non_unit_slots}) != len(non_unit_slots):
        return None, None
    if _count_rotation_mirror_pairs(non_unit_slots) < 1:
        return None, None

    return slots, layout_mask


def _make_piece(slot: dict, color: int, k: int, apply_rotation: bool) -> dict:
    if int(slot["area"]) == 1:
        mask = np.ones((k, k), dtype=bool)
        patch = np.full((k, k), int(color), dtype=int)
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

    if apply_rotation:
        rot = random.randint(0, 3)
        in_mask = np.rot90(out_mask, int(rot))
        in_patch = np.rot90(out_patch, int(rot))
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
    frame_h = int(layout_mask.shape[0]) + 2
    frame_w = int(layout_mask.shape[1]) + 2
    total_piece_area = sum(int(piece["in_mask"].sum()) for piece in pieces)
    needed_cells = frame_h * frame_w + AREA_BUFFER_MULT * total_piece_area + AREA_BUFFER_CONST

    min_h = max(CANVAS_DIM_RANGE[0], frame_h + 2)
    min_w = max(CANVAS_DIM_RANGE[0], frame_w + 2)
    canvas_h = random.randint(min_h, CANVAS_DIM_RANGE[1])
    canvas_w = random.randint(min_w, CANVAS_DIM_RANGE[1])
    while canvas_h * canvas_w < needed_cells and (
        canvas_h < CANVAS_DIM_RANGE[1] or canvas_w < CANVAS_DIM_RANGE[1]
    ):
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
            return None
        paste(canvas, piece["in_patch"], pr, pc, mask=piece["in_mask"], overwrite=True)

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


def _gen_attempt() -> np.ndarray | None:
    k = random.choice(K_CHOICES)
    work_h = random.randint(*WORK_LAYOUT_H_RANGE)
    work_w = random.randint(*WORK_LAYOUT_W_RANGE)
    n_slots = random.randint(*N_SLOTS_RANGE)

    palette = list(range(1, 10))
    random.shuffle(palette)
    frame_color = palette.pop()
    distractor_color = palette.pop()
    needed_piece_colors = (n_slots - N_UNITS) + 1
    if len(palette) < needed_piece_colors:
        return None
    non_unit_colors = palette[: n_slots - N_UNITS]
    unit_color = palette[n_slots - N_UNITS]

    slots, layout_mask = _gen_layout(work_h, work_w, n_slots)
    if slots is None or layout_mask is None:
        return None

    non_unit_slots = [slot for slot in slots if int(slot["area"]) > 1]
    unit_slots = [slot for slot in slots if int(slot["area"]) == 1]
    if len(unit_slots) != N_UNITS or len(non_unit_slots) != n_slots - N_UNITS:
        return None

    frame_area = int((layout_mask.shape[0] + 2) * (layout_mask.shape[1] + 2) - layout_mask.sum())
    max_piece_area = int(max(slot["area"] * k * k for slot in non_unit_slots + unit_slots))
    if frame_area <= max_piece_area:
        return None

    pieces = [_make_piece(slot, non_unit_colors[i], k, apply_rotation=True) for i, slot in enumerate(non_unit_slots)]
    pieces += [_make_piece(slot, unit_color, k, apply_rotation=False) for slot in unit_slots]

    expected = _build_expected_output(
        pieces,
        int(layout_mask.shape[0]),
        int(layout_mask.shape[1]),
        int(k),
        int(frame_color),
    )
    canvas = _build_input_canvas(pieces, layout_mask, int(frame_color), int(distractor_color))
    if canvas is None:
        return None

    try:
        produced = main(canvas)
    except Exception:
        return None
    if not np.array_equal(produced, expected):
        return None

    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_MAX_RETRIES):
        res = _gen_attempt()
        if res is not None:
            return res
    raise RuntimeError(f"generate_input: failed after {GENERATE_MAX_RETRIES} attempts")
