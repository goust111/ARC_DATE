# Combo variant of 898e7135 (index 5)
# combo_spec: D1/1+D9/2
# Name: 大形状旋转匹配

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
Design analysis for combo D1/1 + D9/2
=====================================

combo_summary:
  preserved_core:
    - 输入仍是一个单色矩形框架，框内若干背景色洞，框外散落与洞一一对应的单色块状精灵。
    - 每个精灵仍是某个洞按 k×k 实心块放大的版本；k 仍由最小 unit-slot 对应的实心方块推断。
    - 输出仍是把洞布局 tight-crop 后整体按 k 放大，并用匹配精灵填回洞位。

  fused_change:
    - 匹配只允许 4 个旋转，不允许镜像翻转。
    - 非单元洞面积扩展到 2-12，并且每题强制包含一对 area>=8 的大洞：
      它们互为镜像，但不属于同一个旋转类。
    - 这让“只允许旋转”成为真正必要条件，因为若允许翻转，这对大洞会在块级形状上产生额外可匹配项。

concrete_behavior_trace:
  small_example:
    - 设 k=2，框内有两个 9 格大洞：
      A = 4x3 bbox 中缺掉左上角的形状，B = A 的水平镜像。
    - 输入中的两个对应 piece 都是各自洞放大后的单色块图，只随机旋转，不做翻转。
  original_task:
    - solver 会把 piece 下采样到块级掩码，再在“旋转+翻转”下比较，
      所以 A-piece 与 B-slot 也会被视为同形。
  fused_task:
    - solver 只能枚举 4 个旋转，A-piece 只能回到 A-slot，不能填到 B-slot。
    - 因为这对对象面积大、bbox 更复杂，解题者必须看出手性，而不是把它们当成“翻一下也一样”。
  visible_difference:
    - 样例里会稳定出现至少一对大面积左右手形状；如果忽略镜像限制，就会把这对对象视为可互换。

complexity_calibration:
  why_not_just_expand_area:
    - 只放一个大洞，或者只放旋转等价的大洞，仍可能退化成原题。
  why_not_just_remove_flips:
    - 若 generator 不主动构造镜像但非旋转等价的大洞，2-4 组样例里很难肉眼看出区别。
  final_control:
    - 每题至少 1 对 large chiral mirror pair，pair 的 area 取 8-12 范围中的少量档位。
    - 其余 non-unit 洞仍要求 rotation-class 唯一，避免匹配歧义。
    - k=3 时收紧 slot 数和总面积预算，避免大精灵挤爆画布。

implicit_assumption_audit_for_main:
  - find_components(..., monochromatic=False):
    assumption: 框架必须仍是最大的前景连通块。
    generator_guard: 画框与所有 piece 保持至少 1 格背景间隔，并检查 frame_area > max_piece_area。
  - hole tight crop:
    assumption: 取框内背景洞的 tight bbox 后，就是待放大的布局。
    generator_guard: 先在 12x8 工作布局里放洞，再按 hole bbox 裁成最终 layout，保持该定义不变。
  - slot extraction by 4-connectivity:
    assumption: 每个洞内部 4 连通，不同洞之间 8-隔离。
    generator_guard: 放置时对每个洞做 8 邻域禁放。
  - infer k from smallest solid square piece:
    assumption: 至少有一个 unit slot，对应 k×k 实心方块。
    generator_guard: 每题固定保留 1-2 个 area=1 洞。
  - rotation-only candidate enumeration:
    assumption: piece 的合法朝向只有旋转，没有翻转。
    generator_guard: non-unit piece 的输入朝向只做随机旋转。
  - first-match placement:
    assumption: 对每个 slot，未使用 piece 中恰有一个 rotation-class 精确匹配。
    generator_guard: 所有 non-unit slot 的 rotation key 全部唯一；镜像配对只在 flip 下等价，不在 rotation 下重合。

placement_legality:
  - 在固定 12x8 工作网格中放置 tight mask，任何已占单元及其 8 邻域都会被标成 forbidden。
  - 放完后再 tight-crop 到最终 layout，检查尺寸 4-12 x 4-8、hole density 20%-50%。
  - 画布容量按 frame + piece 总面积预算；piece 使用 random_free_location(..., padding=1, padding_connectivity=8) 散布。
  - 若 frame 不是最大组件、背景 0 不是众数、或 oracle 校验失败，则整例重采。

visual_observability:
  - 关键可观察信号不是颜色，而是至少一对 area>=8 的左右手大轮廓。
  - 两个大 shape 颜色不同，但颜色只用于对象身份；真正可归纳的规则是“镜像不再等价，只能旋转对齐”。
  - 额外小洞只作配套，不会淹没这对大形状，因为它们面积更大、bbox 更复杂、视觉上更显眼。

visual_cleanliness_and_inference:
  - 仍保持单色 frame、单色 piece、单色 singleton 干扰点，避免区域叠层。
  - piece 彼此与 frame 之间至少隔 1 格背景，干扰点独立散布，不接触有效对象。
  - 不使用任何人类不可见因素进行排序或匹配；全部规则都能从形状、连通性、位置与缩放关系中观察到。
"""


RotKey = tuple[tuple[int, int], bytes]


def _rotation_shape_key(mask: np.ndarray) -> RotKey:
    """Canonical key for a tight boolean mask under rotations only."""
    variants = []
    for kk in range(4):
        rot = np.rot90(mask, int(kk))
        variants.append((tuple(map(int, rot.shape)), rot.tobytes()))
    return min(variants)


def _tight_mask_from_coords(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    r0, c0, r1, c1 = map(int, bbox_coords(coords))
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask, (r0, c0)


def _slot_from_coords(coords: np.ndarray) -> dict:
    mask, (r0, c0) = _tight_mask_from_coords(coords)
    return {
        "coords": coords,
        "area": int(coords.shape[0]),
        "mask": mask,
        "r0": int(r0),
        "c0": int(c0),
        "rot_key": _rotation_shape_key(mask),
    }


def _is_chiral(mask: np.ndarray) -> bool:
    return _rotation_shape_key(mask) != _rotation_shape_key(np.fliplr(mask))


def _is_irregular_large_shape(mask: np.ndarray) -> bool:
    h, w = map(int, mask.shape)
    area = int(mask.sum())
    return area >= 8 and h >= 3 and w >= 3 and area < h * w


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


def _random_shape_mask(
    target_size: int,
    *,
    require_large_chiral: bool,
    used_rot_keys: set[RotKey],
) -> np.ndarray | None:
    """Grow one connected tight mask, optionally forcing a large irregular chiral shape."""
    max_side = min(6, max(3, int(target_size)))
    min_h = 3 if require_large_chiral else 2
    min_w = 3 if require_large_chiral else 2

    for _ in range(SHAPE_RETRIES):
        box_h = random.randint(min_h, max_side)
        box_w = random.randint(min_w, max_side)
        if box_h * box_w < int(target_size):
            continue

        start = (random.randint(0, box_h - 1), random.randint(0, box_w - 1))
        chosen = {start}
        frontier = {
            (start[0] - 1, start[1]),
            (start[0] + 1, start[1]),
            (start[0], start[1] - 1),
            (start[0], start[1] + 1),
        }

        while len(chosen) < int(target_size):
            cands = []
            for nr, nc in frontier:
                if not (0 <= nr < box_h and 0 <= nc < box_w):
                    continue
                if (nr, nc) in chosen:
                    continue
                freedom = 0
                for ar, ac in ((nr - 1, nc), (nr + 1, nc), (nr, nc - 1), (nr, nc + 1)):
                    if 0 <= ar < box_h and 0 <= ac < box_w and (ar, ac) not in chosen:
                        freedom += 1
                cands.append((freedom, nr, nc))
            if not cands:
                break
            cands.sort(reverse=True)
            top = cands[: min(5, len(cands))]
            _, nr, nc = random.choice(top)
            chosen.add((nr, nc))
            frontier.discard((nr, nc))
            frontier.update({(nr - 1, nc), (nr + 1, nc), (nr, nc - 1), (nr, nc + 1)})

        if len(chosen) != int(target_size):
            continue

        coords = np.array(sorted(chosen), dtype=int)
        mask, _ = _tight_mask_from_coords(coords)
        rot_key = _rotation_shape_key(mask)
        if rot_key in used_rot_keys:
            continue
        if require_large_chiral:
            if not _is_irregular_large_shape(mask):
                continue
            if not _is_chiral(mask):
                continue
        return mask
    return None


def _place_mask_randomly(forbidden: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    """Place a tight mask into the work layout while respecting 8-neighbor separation."""
    h, w = map(int, mask.shape)
    H, W = map(int, forbidden.shape)
    if h > H or w > W:
        return None

    base_coords = np.argwhere(mask)
    candidates: list[tuple[int, int]] = []
    for top in range(H - h + 1):
        for left in range(W - w + 1):
            coords = base_coords + np.array([top, left], dtype=int)
            if not bool(forbidden[coords[:, 0], coords[:, 1]].any()):
                candidates.append((int(top), int(left)))
    if not candidates:
        return None

    top, left = random.choice(candidates)
    coords = base_coords + np.array([top, left], dtype=int)
    _forbid_coords(forbidden, coords)
    return coords


def _count_large_chiral_mirror_pairs(non_unit_slots: list[dict]) -> int:
    count = 0
    for i in range(len(non_unit_slots)):
        slot_i = non_unit_slots[i]
        if int(slot_i["area"]) < 8 or not _is_chiral(slot_i["mask"]):
            continue
        mirror_key = _rotation_shape_key(np.fliplr(slot_i["mask"]))
        if mirror_key == slot_i["rot_key"]:
            continue
        for j in range(i + 1, len(non_unit_slots)):
            slot_j = non_unit_slots[j]
            if int(slot_j["area"]) < 8:
                continue
            if slot_j["rot_key"] == mirror_key:
                count += 1
    return count


def _extract_layout_slots(layout: np.ndarray) -> list[dict]:
    slots = []
    for coords in components_mask(layout, connectivity=4):
        rr0, cc0, rr1, cc1 = map(int, bbox_coords(coords))
        mask = np.zeros((rr1 - rr0 + 1, cc1 - cc0 + 1), dtype=bool)
        mask[coords[:, 0] - rr0, coords[:, 1] - cc0] = True
        slots.append(
            {
                "area": int(mask.sum()),
                "r0": int(rr0),
                "c0": int(cc0),
                "mask": mask,
                "rot_key": _rotation_shape_key(mask),
            }
        )
    slots.sort(key=lambda slot: (-int(slot["area"]), int(slot["r0"]), int(slot["c0"])))
    return slots


def _extract_problem_structure(grid: np.ndarray) -> dict:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    assert comps, "input must contain at least one foreground component"
    frame_comp = max(comps, key=lambda comp: int(comp["area"]))
    fr0, fc0, fr1, fc1 = map(int, frame_comp["bbox"])
    sub = crop_bbox(g, (fr0, fc0, fr1, fc1))
    hole = sub == int(bg)
    hole_box = bbox(hole)
    assert hole_box is not None, "frame must contain holes"
    lr0, lc0, lr1, lc1 = map(int, hole_box)
    layout = hole[lr0 : lr1 + 1, lc0 : lc1 + 1]

    pieces = []
    for comp in comps:
        if comp is frame_comp:
            continue
        coords = comp["coords"]
        if coords.size == 0 or int(coords.shape[0]) <= 1:
            continue
        pr0, pc0, pr1, pc1 = map(int, comp["bbox"])
        patch = crop_bbox(g, (pr0, pc0, pr1, pc1))
        mask = np.zeros(patch.shape, dtype=bool)
        mask[coords[:, 0] - pr0, coords[:, 1] - pc0] = True
        pieces.append({"area": int(mask.sum()), "patch": patch, "mask": mask})

    k = None
    for piece in pieces:
        h, w = map(int, piece["mask"].shape)
        if h == w and int(piece["mask"].sum()) == int(h * w):
            k = int(h) if k is None else min(int(k), int(h))

    return {
        "grid": g,
        "bg": bg,
        "layout": layout,
        "slots": _extract_layout_slots(layout),
        "pieces": pieces,
        "k": int(k or 1),
    }


def fusion_diagnostics(grid: np.ndarray) -> dict:
    """Expose a small structural metric to confirm the combo behavior appears in samples."""
    info = _extract_problem_structure(grid)
    non_unit_slots = [slot for slot in info["slots"] if int(slot["area"]) > 1]
    large_slots = [slot for slot in non_unit_slots if int(slot["area"]) >= 8]
    return {
        "k": int(info["k"]),
        "slot_count": int(len(info["slots"])),
        "large_slot_areas": sorted(int(slot["area"]) for slot in large_slots),
        "large_chiral_mirror_pairs": int(_count_large_chiral_mirror_pairs(non_unit_slots)),
        "max_slot_area": int(max((slot["area"] for slot in info["slots"]), default=0)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand the frame layout by its block size and refill each hole with the matching sprite.

    Input structure:
    - The most common color is the background; the largest foreground component is a monochrome frame.
    - Inside that frame, background-colored holes form 4-connected slots, including at least one unit slot.
    - Outside the frame sit separated monochrome pieces made of solid k×k blocks; some large pieces are chiral.

    Rule:
    - Tight-crop the hole layout from inside the frame and split it into slot masks.
    - Infer k from the smallest solid square piece, then downsample each piece to block level.
    - Match every slot with one unused piece using exact block-mask equality under rotations only; mirror flips are illegal.
    - Return the cropped layout enlarged by k, with frame color as background and the matched rotated piece pasted into each slot.

    Key insight:
    Each outside sprite is a rotated k×k-block expansion of one hole shape, but large mirror-image shapes are no longer interchangeable. The solver must compare downsampled masks under the 4 rotations only and preserve chirality.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    assert comps, "input must contain a frame component"
    frame_comp = max(comps, key=lambda comp: int(comp["area"]))
    fr0, fc0, fr1, fc1 = map(int, frame_comp["bbox"])

    frame_cols = [int(g[int(r), int(c)]) for r, c in frame_comp["coords"]]
    frame_color = int(Counter(frame_cols).most_common(1)[0][0])

    sub = crop_bbox(g, (fr0, fc0, fr1, fc1))
    hole = sub == int(bg)
    hole_box = bbox(hole)
    assert hole_box is not None, "frame must enclose at least one background hole"
    lr0, lc0, lr1, lc1 = map(int, hole_box)
    layout = hole[lr0 : lr1 + 1, lc0 : lc1 + 1]

    slots = _extract_layout_slots(layout)
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
        mask = np.zeros(patch.shape, dtype=bool)
        mask[coords[:, 0] - pr0, coords[:, 1] - pc0] = True
        pieces.append({"area": int(mask.sum()), "mask": mask, "patch": patch})
    assert pieces, "at least one non-frame piece must exist"

    k = None
    for piece in pieces:
        h, w = map(int, piece["mask"].shape)
        if h == w and int(piece["mask"].sum()) == int(h * w):
            k = int(h) if k is None else min(int(k), int(h))
    k = int(k or 1)

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
        uniq = []
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
            candidates.append(
                {
                    "pi": int(pi),
                    "area": int(piece["area"]),
                    "um": um,
                    "vp": vp,
                    "vm": vm,
                }
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


K_CHOICES = [2, 3]
WORK_LAYOUT_SHAPE = (12, 8)
FINAL_LAYOUT_H_RANGE = (4, 12)
FINAL_LAYOUT_W_RANGE = (4, 8)
N_SLOTS_RANGE_K2 = (3, 5)
N_SLOTS_RANGE_K3 = (3, 4)
N_UNITS_MAX = 2

PAIR_SIZE_CHOICES_K2 = [8, 8, 9, 9, 10, 10, 11, 12]
PAIR_SIZE_CHOICES_K3 = [8, 8, 9, 9, 10]
SMALL_NON_UNIT_CHOICES = [2, 3, 4, 5, 6, 7]
TOTAL_SLOT_AREA_BUDGET = {2: 30, 3: 22}

HOLE_DENSITY_RANGE = (0.20, 0.50)
SHAPE_RETRIES = 140
LAYOUT_RETRIES = 220

CANVAS_DIM_RANGE = (15, 30)
AREA_BUFFER_MULT = 4
AREA_BUFFER_CONST = 60

DISTRACTOR_RANGE = (8, 13)
DISTRACTOR_RETRY_MULT = 12
MIN_DISTRACTORS = 8

GENERATE_MAX_RETRIES = 700


def _choose_slot_sizes(k: int) -> tuple[int, list[int]] | None:
    n_slots = random.randint(*(N_SLOTS_RANGE_K2 if k == 2 else N_SLOTS_RANGE_K3))
    n_units = random.randint(1, min(N_UNITS_MAX, n_slots - 2))
    non_unit_count = int(n_slots - n_units)
    if non_unit_count < 2:
        return None

    pair_size = random.choice(PAIR_SIZE_CHOICES_K2 if k == 2 else PAIR_SIZE_CHOICES_K3)
    sizes = [int(pair_size), int(pair_size)]
    budget = int(TOTAL_SLOT_AREA_BUDGET[int(k)])

    while len(sizes) < non_unit_count:
        remaining = non_unit_count - len(sizes)
        max_allowed = budget - n_units - sum(sizes) - 2 * (remaining - 1)
        if max_allowed < 2:
            return None
        choices = [size for size in SMALL_NON_UNIT_CHOICES if size <= max_allowed]
        if not choices:
            return None
        sizes.append(int(random.choice(choices)))

    random.shuffle(sizes)
    return int(n_units), sizes


def _crop_layout(slots: list[dict]) -> tuple[list[dict], np.ndarray] | None:
    full_mask = np.zeros(WORK_LAYOUT_SHAPE, dtype=bool)
    for slot in slots:
        full_mask[slot["coords"][:, 0], slot["coords"][:, 1]] = True

    hole_box = bbox(full_mask)
    if hole_box is None:
        return None
    r0, c0, r1, c1 = map(int, hole_box)
    layout_mask = full_mask[r0 : r1 + 1, c0 : c1 + 1]
    h, w = map(int, layout_mask.shape)
    if not (FINAL_LAYOUT_H_RANGE[0] <= h <= FINAL_LAYOUT_H_RANGE[1]):
        return None
    if not (FINAL_LAYOUT_W_RANGE[0] <= w <= FINAL_LAYOUT_W_RANGE[1]):
        return None
    density = float(layout_mask.sum()) / float(h * w)
    if not (HOLE_DENSITY_RANGE[0] <= density <= HOLE_DENSITY_RANGE[1]):
        return None

    cropped_slots = []
    for slot in slots:
        coords = slot["coords"].copy()
        coords[:, 0] -= int(r0)
        coords[:, 1] -= int(c0)
        cropped_slots.append(_slot_from_coords(coords))
    return cropped_slots, layout_mask


def _generate_slots(slot_sizes: list[int], n_units: int) -> tuple[list[dict], np.ndarray] | None:
    for _ in range(LAYOUT_RETRIES):
        forbidden = np.zeros(WORK_LAYOUT_SHAPE, dtype=bool)
        unit_slots: list[dict] = []
        non_unit_slots: list[dict] = []
        used_rot_keys: set[RotKey] = set()

        pair_size = max(slot_sizes)
        extra_sizes = list(slot_sizes)
        extra_sizes.remove(pair_size)
        extra_sizes.remove(pair_size)
        large_pair_mask = _random_shape_mask(
            int(pair_size),
            require_large_chiral=True,
            used_rot_keys=used_rot_keys,
        )
        if large_pair_mask is None:
            continue
        mirror_mask = np.fliplr(large_pair_mask)
        mirror_key = _rotation_shape_key(mirror_mask)
        base_key = _rotation_shape_key(large_pair_mask)
        if mirror_key == base_key:
            continue

        ok = True
        for mask in random.sample([large_pair_mask, mirror_mask], 2):
            coords = _place_mask_randomly(forbidden, mask)
            if coords is None:
                ok = False
                break
            slot = _slot_from_coords(coords)
            used_rot_keys.add(slot["rot_key"])
            non_unit_slots.append(slot)
        if not ok:
            continue

        for size in sorted(extra_sizes, reverse=True):
            mask = _random_shape_mask(int(size), require_large_chiral=False, used_rot_keys=used_rot_keys)
            if mask is None:
                ok = False
                break
            coords = _place_mask_randomly(forbidden, mask)
            if coords is None:
                ok = False
                break
            slot = _slot_from_coords(coords)
            used_rot_keys.add(slot["rot_key"])
            non_unit_slots.append(slot)
        if not ok:
            continue

        for _ in range(n_units):
            coords = _place_mask_randomly(forbidden, np.ones((1, 1), dtype=bool))
            if coords is None:
                ok = False
                break
            unit_slots.append(_slot_from_coords(coords))
        if not ok:
            continue

        cropped = _crop_layout(unit_slots + non_unit_slots)
        if cropped is None:
            continue
        cropped_slots, layout_mask = cropped
        cropped_units = [slot for slot in cropped_slots if int(slot["area"]) == 1]
        cropped_non_units = [slot for slot in cropped_slots if int(slot["area"]) > 1]
        if len(cropped_units) != int(n_units):
            continue
        if not any(int(slot["area"]) >= 8 for slot in cropped_non_units):
            continue
        if _count_large_chiral_mirror_pairs(cropped_non_units) < 1:
            continue
        return cropped_slots, layout_mask

    return None


def _make_piece(slot: dict, color: int, k: int, *, rotate_only: bool) -> dict:
    if int(slot["area"]) == 1:
        patch = np.full((k, k), int(color), dtype=int)
        mask = np.ones((k, k), dtype=bool)
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
    if rotate_only:
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
    pieces: list[dict], layout_h: int, layout_w: int, k: int, frame_color: int
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
    layout_h: int,
    layout_w: int,
    frame_color: int,
    distractor_color: int,
) -> np.ndarray | None:
    frame_h, frame_w = int(layout_h + 2), int(layout_w + 2)
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

    for piece in sorted(pieces, key=lambda item: -int(item["in_mask"].sum())):
        try:
            pr, pc = random_free_location(
                canvas,
                piece["in_patch"],
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
    size_plan = _choose_slot_sizes(int(k))
    if size_plan is None:
        return None
    n_units, non_unit_sizes = size_plan

    palette = list(range(1, 10))
    random.shuffle(palette)
    frame_color = palette.pop()
    distractor_color = palette.pop()
    needed_piece_colors = len(non_unit_sizes) + 1
    if len(palette) < needed_piece_colors:
        return None
    non_unit_colors = palette[: len(non_unit_sizes)]
    unit_color = palette[len(non_unit_sizes)]

    layout_res = _generate_slots(non_unit_sizes, int(n_units))
    if layout_res is None:
        return None
    slots, layout_mask = layout_res

    unit_slots = [slot for slot in slots if int(slot["area"]) == 1]
    non_unit_slots = [slot for slot in slots if int(slot["area"]) > 1]
    if not unit_slots or len(non_unit_slots) < 2:
        return None
    if _count_large_chiral_mirror_pairs(non_unit_slots) < 1:
        return None

    layout_h, layout_w = map(int, layout_mask.shape)
    frame_area = int((layout_h + 2) * (layout_w + 2) - layout_mask.sum())
    max_piece_area = int(
        max((int(slot["area"]) * int(k) * int(k) for slot in non_unit_slots + unit_slots), default=0)
    )
    if frame_area <= max_piece_area:
        return None

    pieces = [
        _make_piece(slot, non_unit_colors[i], int(k), rotate_only=True)
        for i, slot in enumerate(non_unit_slots)
    ]
    pieces += [_make_piece(slot, unit_color, int(k), rotate_only=False) for slot in unit_slots]

    expected_out = _build_expected_output(pieces, layout_h, layout_w, int(k), int(frame_color))
    canvas = _build_input_canvas(
        pieces,
        layout_mask,
        layout_h,
        layout_w,
        int(frame_color),
        int(distractor_color),
    )
    if canvas is None:
        return None

    try:
        produced = main(canvas)
    except Exception:
        return None
    if not np.array_equal(produced, expected_out):
        return None

    diag = fusion_diagnostics(canvas)
    if int(diag["large_chiral_mirror_pairs"]) < 1:
        return None
    if int(diag["max_slot_area"]) < 8:
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_MAX_RETRIES):
        res = _gen_attempt()
        if res is not None:
            return res
    raise RuntimeError(f"generate_input: failed after {GENERATE_MAX_RETRIES} attempts")
