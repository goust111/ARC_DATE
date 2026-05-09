# Combo variant of 898e7135 (index 35)
# combo_spec: D1/4+D6/3
# Name: 排序排除干扰

from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np

from arc_utils import bbox, bbox_coords, components_mask, crop_bbox, find_components, get_bg, paste

"""
Design analysis for combo variant D1/4 + D6/3
==============================================

spec_combo:
  preserved_core:
    - 背景仍是 0，且必须是输入里出现次数最多的颜色。
    - 最大前景连通块仍是单色 frame；frame 内的 bg 连通域仍是待填充 slot。
    - 每个真实 piece 仍由 k×k 实心块组成；至少一个真实 unit slot 仍暴露出真实的 k。
    - 输出仍是把紧裁后的 frame 内布局按 k 放大，再把选中的 piece 贴到 slot 左上锚点。

  fused_change:
    - D1/4: 不再做形状匹配，真实 piece 与 slot 的对应关系改为“按面积从大到小排序后逐一配对”。
    - D6/3: 输入额外加入 1-2 个多像素伪精灵，它们在视觉上仍像外部 piece，但面积不属于真实排序序列。
    - 融合后的 solver 必须先识别并排除面积异常的伪精灵，再对剩余真实 piece 做面积排序配对。

  concrete_trace:
    - 例子：slot 面积依次是 [5, 5, 3, 1]，k=2，所以真实 piece 像素面积是 [20, 20, 12, 4]。
    - 两个面积 5 的 slot 形状不同，左上的是真 L 形，右下的是横条。外部真实 piece 的摆放顺序被反过来，
      因此按面积排序后，同面积 tie-break 会把“横条 piece”先贴到“L 形 slot”的位置。
    - 同时再加入两个伪精灵，面积为 16 和 8。输入里的所有多像素连通块面积排序后会变成 [20, 20, 16, 12, 8, 4]。
    - 原题逻辑会尝试把所有 piece 都配洞，或者用形状匹配保留原对应关系；融合后逻辑则要删除 16 和 8 这两个异常项，
      用 [20, 20, 12, 4] 去对应 [5, 5, 3, 1]。
    - 差异有两层：一是确实有额外 piece 被忽略；二是保留下来的真实 piece 也不是按原本形状回洞，而是按面积排序后粘贴。

  complexity_calibration:
    - 不能让所有非 unit slot 面积互异，否则“面积排序”与“形状回洞”经常得到同样输出，D1 改动会退化。
    - 生成器因此强制至少有一组“相同面积但 D4 形状不同”的非 unit slot，并反转这组真实 piece 的输入顺序。
    - 伪精灵的数量固定在 1-2 个，避免视觉过乱；它们的面积明确避开真实 slot 面积集合，从而能被面积异常规则识别。
    - 当 k=3 时额外加入一个 2x2 实心假方块，使“最小实心方块就是 k”的旧启发式失效；这让 D6 改动真正参与推理链。

  implicit_assumption_audit_of_original_main:
    - “所有非单像素前景连通块都是真 piece”：
      融合后失效。solver 需要允许 1-2 个 leftover 多像素连通块存在。
    - “最小实心方块边长就是 k”：
      融合后失效，因为 k=3 时会额外出现 2x2 假方块。solver 需要枚举所有实心方块边长候选，并验证哪一个 k 能解释真实面积序列。
    - “piece 面积多重集与 slot 面积多重集完全一致”：
      融合后失效。正确表述应是：删掉 1-2 个面积异常连通块后，剩余 piece 面积多重集与 slot 面积乘 k^2 后一致。
    - “逐个 greedy 匹配即可”：
      对面积排序规则而言，不再需要形状回溯；但需要先完成 subset 选择，否则直接 zip 会被伪精灵破坏长度和面积序列。
    - “同面积 piece 的相对顺序无关”：
      融合后不成立。因为 D1 变体恰恰依赖可见的 bbox 左上角顺序来决定同面积 piece 的去向，生成器必须稳定控制这一点。

  placement_legality:
    - frame 固定放在 30×30 画布右下，给外部 piece 留出左上区域，便于稳定控制同面积 piece 的输入顺序。
    - 所有外部对象都按真实 mask 做 8 邻域间隔检测，保证 piece、伪精灵、frame 彼此不接触、不重叠、不越界。
    - 额外检查 frame 的前景面积大于任一外部对象，保证“最大前景连通块是 frame”的 solver 前提持续成立。

  visual_observability:
    - 每个样例都至少包含一个重复非 unit 面积组，所以解题者会看到“这些 shape 明明不对应洞，却因为面积和位置顺序被交换”。
    - 每个样例都至少包含 1 个额外多像素对象，且它的面积插在真实面积序列中间或至少不属于真实集合，
      使“先删异常面积，再排序”可以从 2-4 个样例中归纳出来。
    - 排序 tie-break 只依赖可见的输入空间位置 `(bbox top, bbox left)`，不依赖颜色编号等不可视因素。

  cleanliness:
    - 没有覆盖标记、文字提示或额外编码系统；只有 frame、真实 piece、伪精灵三类对象。
    - 伪精灵和真实 piece 共享同一“框外散落对象”语义区域，场景自然，不会引入第二套注释图层。

  post-change solver assumptions and generator guarantees:
    - solver 假设至少存在一个真实 unit slot，因此真实 k×k 方块一定在真实 piece 中；generator 保证这一点。
    - solver 假设恰有一个候选 k 能在删去少量异常面积后解释全部 slot；generator 通过面积设计与 k=3 的 2x2 假方块保证这一点。
    - solver 假设伪精灵面积不与真实需求面积冲突；generator 显式从未使用的 slot 面积档位里选择伪精灵基面积。
"""


K_CHOICES = [2, 3]
LAYOUT_H_RANGE = (4, 12)
LAYOUT_W_RANGE = (4, 8)
N_SLOTS_RANGE = (3, 6)
N_UNITS_MAX = 2
NON_UNIT_SIZE_CHOICES = [2, 3, 4, 5, 6, 7]
DUPLICATE_AREA_CHOICES = [2, 3, 4, 5]

HOLE_DENSITY_RANGE = (0.20, 0.50)
UNIT_SLOT_RETRIES = 20
NON_UNIT_SLOT_RETRIES = 30
GROW_SEED_TRIES = 8
DECOY_MASK_RETRIES = 80

CANVAS_SIZE = 30
N_DECOYS_FOR_K2 = [1, 2]

GENERATE_MAX_RETRIES = 400


def _slot_sort_key(slot: dict) -> tuple[int, int, int]:
    return (-int(slot["area"]), int(slot["r0"]), int(slot["c0"]))


def _piece_sort_key(piece: dict) -> tuple[int, int, int]:
    return (-int(piece["area"]), int(piece["r0"]), int(piece["c0"]))


def _canonical_shape_key(mask: np.ndarray) -> bytes:
    variants = []
    for kk in range(4):
        rot = np.rot90(mask, kk)
        variants.append(rot.tobytes() + bytes([rot.shape[0], rot.shape[1]]))
        flip = np.fliplr(rot)
        variants.append(flip.tobytes() + bytes([flip.shape[0], flip.shape[1]]))
    return min(variants)


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


def _grow_slot(forbidden: np.ndarray, target_size: int) -> np.ndarray | None:
    h, w = forbidden.shape
    free_cells = np.argwhere(~forbidden)
    if free_cells.size == 0:
        return None
    random.shuffle(free_cells)
    for start in free_cells[:GROW_SEED_TRIES]:
        r0, c0 = int(start[0]), int(start[1])
        chosen = {(r0, c0)}
        frontier = {(r0 - 1, c0), (r0 + 1, c0), (r0, c0 - 1), (r0, c0 + 1)}
        while len(chosen) < int(target_size) and frontier:
            cands = [
                pos
                for pos in frontier
                if 0 <= pos[0] < h and 0 <= pos[1] < w and not forbidden[pos[0], pos[1]] and pos not in chosen
            ]
            if not cands:
                break
            nr, nc = random.choice(cands)
            chosen.add((int(nr), int(nc)))
            frontier.discard((nr, nc))
            frontier.update({(nr - 1, nc), (nr + 1, nc), (nr, nc - 1), (nr, nc + 1)})
        if len(chosen) == int(target_size):
            return np.array(sorted(chosen), dtype=int)
    return None


def _mask_from_coords(coords: np.ndarray) -> tuple[int, int, np.ndarray]:
    r0, c0, r1, c1 = map(int, bbox_coords(coords))
    mask = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return int(r0), int(c0), mask


def _choose_non_unit_sizes(n_non_unit: int) -> list[int]:
    assert int(n_non_unit) >= 2
    dup_area = int(random.choice(DUPLICATE_AREA_CHOICES))
    sizes = [dup_area, dup_area]
    while len(sizes) < int(n_non_unit):
        sizes.append(int(random.choice(NON_UNIT_SIZE_CHOICES)))
    random.shuffle(sizes)
    return sizes


def _gen_layout(h: int, w: int, n_slots: int, n_units: int) -> list[dict] | None:
    forbidden = np.zeros((h, w), dtype=bool)
    slots: list[dict] = []
    used_shape_keys: set[bytes] = set()

    for _ in range(int(n_units)):
        for _ in range(UNIT_SLOT_RETRIES):
            coords = _grow_slot(forbidden, 1)
            if coords is None:
                continue
            r0, c0, mask = _mask_from_coords(coords)
            slots.append(
                {
                    "coords": coords,
                    "area": 1,
                    "r0": int(r0),
                    "c0": int(c0),
                    "mask": mask,
                    "shape_key": mask.tobytes(),
                }
            )
            _forbid_coords(forbidden, coords)
            break
        else:
            return None

    for size in _choose_non_unit_sizes(int(n_slots - n_units)):
        for _ in range(NON_UNIT_SLOT_RETRIES):
            coords = _grow_slot(forbidden, int(size))
            if coords is None:
                continue
            r0, c0, mask = _mask_from_coords(coords)
            key = _canonical_shape_key(mask)
            if key in used_shape_keys:
                continue
            used_shape_keys.add(key)
            slots.append(
                {
                    "coords": coords,
                    "area": int(size),
                    "r0": int(r0),
                    "c0": int(c0),
                    "mask": mask,
                    "shape_key": key,
                }
            )
            _forbid_coords(forbidden, coords)
            break
        else:
            return None

    density = sum(int(slot["area"]) for slot in slots) / float(h * w)
    lo, hi = HOLE_DENSITY_RANGE
    if not (float(lo) <= density <= float(hi)):
        return None

    groups = defaultdict(list)
    for slot in slots:
        if int(slot["area"]) > 1:
            groups[int(slot["area"])].append(slot)
    if not any(len(group) >= 2 for group in groups.values()):
        return None

    return slots


def _scaled_patch_from_mask(mask: np.ndarray, color: int, k: int, apply_dihedral: bool) -> tuple[np.ndarray, np.ndarray]:
    full_mask = np.kron(mask, np.ones((k, k), dtype=bool))
    patch = np.where(full_mask, int(color), 0).astype(int)
    if apply_dihedral:
        rot = random.randint(0, 3)
        full_mask = np.rot90(full_mask, rot)
        patch = np.rot90(patch, rot)
        if random.random() < 0.5:
            full_mask = np.fliplr(full_mask)
            patch = np.fliplr(patch)
    return patch, full_mask


def _make_real_piece(slot: dict, color: int, k: int) -> dict:
    out_mask = np.kron(slot["mask"], np.ones((k, k), dtype=bool))
    out_patch = np.where(out_mask, int(color), 0).astype(int)
    in_patch, in_mask = _scaled_patch_from_mask(slot["mask"], int(color), int(k), apply_dihedral=int(slot["area"]) > 1)
    return {
        "slot": slot,
        "color": int(color),
        "out_patch": out_patch,
        "out_mask": out_mask,
        "in_patch": in_patch,
        "in_mask": in_mask,
        "area": int(in_mask.sum()),
    }


def _desired_real_piece_order(pieces: list[dict]) -> list[dict]:
    by_area = defaultdict(list)
    for piece in pieces:
        by_area[int(piece["slot"]["area"])].append(piece)

    ordered: list[dict] = []
    for area in sorted(by_area.keys(), reverse=True):
        group = sorted(by_area[area], key=lambda p: (int(p["slot"]["r0"]), int(p["slot"]["c0"])))
        if int(area) > 1 and len(group) >= 2:
            group = list(reversed(group))
        ordered.extend(group)
    return ordered


def _random_non_square_mask(area: int) -> np.ndarray | None:
    for _ in range(DECOY_MASK_RETRIES):
        coords = _grow_slot(np.zeros((5, 5), dtype=bool), int(area))
        if coords is None:
            continue
        _, _, mask = _mask_from_coords(coords)
        if mask.shape[0] == mask.shape[1] and int(mask.sum()) == int(mask.shape[0] * mask.shape[1]):
            continue
        return mask
    return None


def _choose_block_decoy_base_sizes(slot_areas: list[int], n_needed: int) -> list[int] | None:
    used = set(int(a) for a in slot_areas)
    middle = [a for a in range(2, 8) if a not in used and min(slot_areas) < a < max(slot_areas)]
    outer = [a for a in range(2, 8) if a not in used and a not in middle]
    choices = middle + outer
    if len(choices) < int(n_needed):
        return None
    return choices[: int(n_needed)]


def _make_block_decoy(base_area: int, color: int, k: int) -> dict | None:
    base_mask = _random_non_square_mask(int(base_area))
    if base_mask is None:
        return None
    patch, mask = _scaled_patch_from_mask(base_mask, int(color), int(k), apply_dihedral=True)
    return {
        "kind": "block_decoy",
        "base_area": int(base_area),
        "color": int(color),
        "in_patch": patch,
        "in_mask": mask,
        "area": int(mask.sum()),
    }


def _make_small_square_decoy(color: int) -> dict:
    patch = np.full((2, 2), int(color), dtype=int)
    mask = np.ones((2, 2), dtype=bool)
    return {
        "kind": "small_square_decoy",
        "base_area": 4,
        "color": int(color),
        "in_patch": patch,
        "in_mask": mask,
        "area": 4,
    }


def _can_place_with_gap(occupied: np.ndarray, mask: np.ndarray, r0: int, c0: int) -> bool:
    h, w = map(int, mask.shape)
    hh, ww = occupied.shape
    if r0 < 0 or c0 < 0 or r0 + h > hh or c0 + w > ww:
        return False
    probe = np.zeros_like(occupied)
    probe[r0 : r0 + h, c0 : c0 + w] = mask
    return not bool(np.any(_dilate8(probe) & occupied))


def _build_input_canvas(
    real_pieces_in_order: list[dict],
    decoys: list[dict],
    layout_mask: np.ndarray,
    layout_h: int,
    layout_w: int,
    frame_color: int,
) -> np.ndarray | None:
    canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=int)
    frame_h, frame_w = layout_h + 2, layout_w + 2
    frame_top = CANVAS_SIZE - frame_h - 1
    frame_left = CANVAS_SIZE - frame_w - 1
    if frame_top < 1 or frame_left < 1:
        return None

    frame_img = np.full((frame_h, frame_w), int(frame_color), dtype=int)
    frame_img[1:-1, 1:-1] = np.where(layout_mask, 0, int(frame_color))
    frame_area = int(np.count_nonzero(frame_img))
    outside_objects = list(real_pieces_in_order) + list(decoys)
    if any(int(obj["in_mask"].sum()) >= frame_area for obj in outside_objects):
        return None

    paste(canvas, frame_img, int(frame_top), int(frame_left))
    occupied = canvas != 0

    start_idx = 0
    for piece in real_pieces_in_order:
        placed = False
        for idx in range(start_idx, CANVAS_SIZE * CANVAS_SIZE):
            rr = idx // CANVAS_SIZE
            cc = idx % CANVAS_SIZE
            if not _can_place_with_gap(occupied, piece["in_mask"], int(rr), int(cc)):
                continue
            paste(canvas, piece["in_patch"], int(rr), int(cc), mask=piece["in_mask"], overwrite=True)
            occupied[rr : rr + piece["in_mask"].shape[0], cc : cc + piece["in_mask"].shape[1]] |= piece["in_mask"]
            piece["r0"] = int(rr)
            piece["c0"] = int(cc)
            start_idx = int(idx + 1)
            placed = True
            break
        if not placed:
            return None

    for decoy in decoys:
        placed = False
        for idx in range(CANVAS_SIZE * CANVAS_SIZE):
            rr = idx // CANVAS_SIZE
            cc = idx % CANVAS_SIZE
            if not _can_place_with_gap(occupied, decoy["in_mask"], int(rr), int(cc)):
                continue
            paste(canvas, decoy["in_patch"], int(rr), int(cc), mask=decoy["in_mask"], overwrite=True)
            occupied[rr : rr + decoy["in_mask"].shape[0], cc : cc + decoy["in_mask"].shape[1]] |= decoy["in_mask"]
            decoy["r0"] = int(rr)
            decoy["c0"] = int(cc)
            placed = True
            break
        if not placed:
            return None

    vals, counts = np.unique(canvas, return_counts=True)
    if int(vals[np.argmax(counts)]) != 0:
        return None
    return canvas


def _build_combo_output(slot_order: list[dict], piece_order: list[dict], layout_h: int, layout_w: int, k: int, frame_color: int) -> np.ndarray:
    out = np.full(((layout_h + 2) * k, (layout_w + 2) * k), int(frame_color), dtype=int)
    for slot, piece in zip(slot_order, piece_order):
        top = (int(slot["r0"]) + 1) * int(k)
        left = (int(slot["c0"]) + 1) * int(k)
        paste(out, piece["in_patch"], int(top), int(left), mask=piece["in_mask"], overwrite=True, clip=True)
    return out


def _build_shape_reference_output(real_pieces: list[dict], layout_h: int, layout_w: int, k: int, frame_color: int) -> np.ndarray:
    out = np.full(((layout_h + 2) * k, (layout_w + 2) * k), int(frame_color), dtype=int)
    for piece in real_pieces:
        top = (int(piece["slot"]["r0"]) + 1) * int(k)
        left = (int(piece["slot"]["c0"]) + 1) * int(k)
        paste(out, piece["out_patch"], int(top), int(left), mask=piece["out_mask"], overwrite=True)
    return out


def _swap_count(slot_order: list[dict], piece_order: list[dict]) -> int:
    count = 0
    by_area_slots = defaultdict(list)
    by_area_pieces = defaultdict(list)
    for slot in slot_order:
        by_area_slots[int(slot["area"])].append(slot)
    for piece in piece_order:
        by_area_pieces[int(piece["slot"]["area"])].append(piece)

    for area, slots in by_area_slots.items():
        if int(area) <= 1 or len(slots) < 2:
            continue
        slot_sources = [id(slot) for slot in slots]
        piece_sources = [id(piece["slot"]) for piece in by_area_pieces[int(area)]]
        if piece_sources != slot_sources:
            count += 1
    return int(count)


def _extract_scene(grid: np.ndarray) -> tuple[int, np.ndarray, list[dict], list[dict]]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    assert comps, "input must contain foreground components"
    frame_comp = max(comps, key=lambda d: int(d["area"]))
    fr0, fc0, fr1, fc1 = map(int, frame_comp["bbox"])

    frame_cols = [int(g[int(r), int(c)]) for r, c in frame_comp["coords"]]
    frame_color = int(Counter(frame_cols).most_common(1)[0][0])

    sub = crop_bbox(g, (fr0, fc0, fr1, fc1))
    hole = sub == int(bg)
    hole_box = bbox(hole)
    assert hole_box is not None, "frame must enclose at least one bg hole"
    lr0, lc0, lr1, lc1 = map(int, hole_box)
    layout = hole[lr0 : lr1 + 1, lc0 : lc1 + 1]

    slots: list[dict] = []
    for coords in components_mask(layout, connectivity=4):
        rr0, cc0, rr1, cc1 = map(int, bbox_coords(coords))
        mask = np.zeros((int(rr1 - rr0 + 1), int(cc1 - cc0 + 1)), dtype=bool)
        mask[coords[:, 0] - rr0, coords[:, 1] - cc0] = True
        slots.append({"area": int(mask.sum()), "r0": int(rr0), "c0": int(cc0), "mask": mask})
    slots.sort(key=_slot_sort_key)
    assert slots, "frame must contain at least one slot"

    pieces: list[dict] = []
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
        pieces.append(
            {
                "area": int(mask.sum()),
                "r0": int(pr0),
                "c0": int(pc0),
                "patch": patch,
                "mask": mask,
            }
        )
    assert pieces, "at least one non-frame multi-pixel piece must exist"
    return frame_color, layout, slots, pieces


def _square_k_candidates(pieces: list[dict]) -> list[int]:
    return sorted(
        {
            int(piece["mask"].shape[0])
            for piece in pieces
            if piece["mask"].shape[0] == piece["mask"].shape[1]
            and int(piece["mask"].sum()) == int(piece["mask"].shape[0] * piece["mask"].shape[1])
            and int(piece["mask"].shape[0]) > 1
        }
    )


def _select_real_pieces_by_area(slots: list[dict], pieces: list[dict], k: int) -> list[dict] | None:
    need = Counter(int(slot["area"]) * int(k) * int(k) for slot in slots)
    by_area: dict[int, list[dict]] = defaultdict(list)
    for piece in pieces:
        by_area[int(piece["area"])].append(piece)

    chosen: list[dict] = []
    for area, count in need.items():
        group = sorted(by_area.get(int(area), []), key=_piece_sort_key)
        if len(group) < int(count):
            return None
        chosen.extend(group[: int(count)])

    if len(chosen) != len(slots):
        return None

    chosen.sort(key=_piece_sort_key)
    chosen_areas = [int(piece["area"]) for piece in chosen]
    need_areas = sorted((int(slot["area"]) * int(k) * int(k) for slot in slots), reverse=True)
    if sorted(chosen_areas, reverse=True) != need_areas:
        return None
    return chosen


def _infer_k_and_real_pieces(slots: list[dict], pieces: list[dict]) -> tuple[int, list[dict]]:
    candidates = _square_k_candidates(pieces)
    assert candidates, "need at least one solid square to infer k candidates"

    valid: list[tuple[int, list[dict]]] = []
    for k in candidates:
        chosen = _select_real_pieces_by_area(slots, pieces, int(k))
        if chosen is not None:
            valid.append((int(k), chosen))

    assert valid, "no candidate k yields a complete area-matched subset"
    assert len(valid) == 1, "ambiguous k: more than one subset/scale explains the scene"
    return valid[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Inflate the frame layout and fill its slots after removing anomalous pseudo-pieces.

    Input structure:
    - The largest foreground component is a monochrome frame around bg holes.
    - Outside the frame are k×k-block pieces plus 1-2 extra multi-pixel pseudo-pieces.
    - At least one true unit slot still contributes a real k×k square, but a smaller fake square may also appear.

    Rule:
    - Read the hole components inside the frame as slots and sort them by area descending, then by top-left position.
    - Among all outside multi-pixel components, choose the unique subset whose pixel areas equal the slot areas multiplied by k^2 for exactly one square-derived k.
    - Ignore leftover components as pseudo-pieces, sort the selected real pieces by area descending and top-left position, then paste the i-th piece into the i-th slot.

    Key insight:
    - The fused task is no longer geometric matching; it is subset recovery plus numeric ordering.
    - Real pieces are exactly the area sequence implied by the slots at one valid scale k, while pseudo-pieces are the area outliers that must be discarded first.
    """
    frame_color, layout, slots, pieces = _extract_scene(np.array(grid, dtype=int))
    k, real_pieces = _infer_k_and_real_pieces(slots, pieces)

    out_h = (int(layout.shape[0]) + 2) * int(k)
    out_w = (int(layout.shape[1]) + 2) * int(k)
    out = np.full((out_h, out_w), int(frame_color), dtype=int)

    for slot, piece in zip(slots, real_pieces):
        top = (int(slot["r0"]) + 1) * int(k)
        left = (int(slot["c0"]) + 1) * int(k)
        paste(out, piece["patch"], int(top), int(left), mask=piece["mask"], overwrite=True, clip=True)

    return out


def _gen_attempt(return_meta: bool = False):
    k = int(random.choice(K_CHOICES))
    layout_h = int(random.randint(*LAYOUT_H_RANGE))
    layout_w = int(random.randint(*LAYOUT_W_RANGE))
    n_slots = int(random.randint(*N_SLOTS_RANGE))
    n_units = int(random.randint(1, min(N_UNITS_MAX, n_slots - 2)))

    n_decoys = 2 if k == 3 else int(random.choice(N_DECOYS_FOR_K2))
    palette = list(range(1, 10))
    random.shuffle(palette)
    needed_piece_colors = (n_slots - n_units) + 1
    needed_colors = 1 + needed_piece_colors + n_decoys
    if len(palette) < needed_colors:
        return None

    frame_color = int(palette.pop())
    piece_palette = [int(palette.pop()) for _ in range(needed_piece_colors)]
    decoy_palette = [int(palette.pop()) for _ in range(n_decoys)]
    non_unit_colors = piece_palette[: n_slots - n_units]
    unit_color = int(piece_palette[-1])

    slots = _gen_layout(layout_h, layout_w, n_slots, n_units)
    if slots is None:
        return None

    layout_mask = np.zeros((layout_h, layout_w), dtype=bool)
    for slot in slots:
        layout_mask[slot["coords"][:, 0], slot["coords"][:, 1]] = True
    if not (layout_mask[0].any() and layout_mask[-1].any() and layout_mask[:, 0].any() and layout_mask[:, -1].any()):
        return None

    unit_slots = [slot for slot in slots if int(slot["area"]) == 1]
    non_unit_slots = [slot for slot in slots if int(slot["area"]) > 1]
    if len(non_unit_slots) < 2:
        return None

    real_pieces = [_make_real_piece(slot, non_unit_colors[i], k) for i, slot in enumerate(non_unit_slots)]
    real_pieces += [_make_real_piece(slot, unit_color, k) for slot in unit_slots]

    slot_areas = [int(slot["area"]) for slot in slots]
    n_block_decoys = n_decoys - (1 if k == 3 else 0)
    base_sizes = _choose_block_decoy_base_sizes(slot_areas, n_block_decoys)
    if base_sizes is None:
        return None

    decoys: list[dict] = []
    decoy_color_idx = 0
    if k == 3:
        decoys.append(_make_small_square_decoy(decoy_palette[decoy_color_idx]))
        decoy_color_idx += 1
    for base_area in base_sizes:
        decoy = _make_block_decoy(int(base_area), decoy_palette[decoy_color_idx], k)
        if decoy is None:
            return None
        decoys.append(decoy)
        decoy_color_idx += 1

    real_area_set = {int(slot["area"]) * int(k) * int(k) for slot in slots}
    if any(int(decoy["area"]) in real_area_set for decoy in decoys):
        return None

    intended_real_order = _desired_real_piece_order(real_pieces)
    slot_order = sorted(slots, key=_slot_sort_key)
    if _swap_count(slot_order, intended_real_order) == 0:
        return None

    canvas = _build_input_canvas(
        real_pieces_in_order=intended_real_order,
        decoys=decoys,
        layout_mask=layout_mask,
        layout_h=layout_h,
        layout_w=layout_w,
        frame_color=frame_color,
    )
    if canvas is None:
        return None

    actual_real_order = sorted(real_pieces, key=_piece_sort_key)
    if _swap_count(slot_order, actual_real_order) == 0:
        return None

    combo_expected = _build_combo_output(slot_order, actual_real_order, layout_h, layout_w, k, frame_color)
    shape_reference = _build_shape_reference_output(real_pieces, layout_h, layout_w, k, frame_color)
    shape_diff_pixels = int(np.count_nonzero(combo_expected != shape_reference))
    if shape_diff_pixels == 0:
        return None

    try:
        produced = main(canvas)
    except Exception:
        return None
    if not np.array_equal(produced, combo_expected):
        return None

    if return_meta:
        candidate_ks = _square_k_candidates(
            [
                {"mask": piece["in_mask"], "area": int(piece["area"]), "r0": int(piece["r0"]), "c0": int(piece["c0"])}
                for piece in real_pieces
            ]
            + [
                {"mask": decoy["in_mask"], "area": int(decoy["area"]), "r0": int(decoy["r0"]), "c0": int(decoy["c0"])}
                for decoy in decoys
            ]
        )
        return canvas, {
            "k": int(k),
            "n_decoys": int(len(decoys)),
            "decoy_areas": [int(decoy["area"]) for decoy in decoys],
            "real_areas": [int(piece["area"]) for piece in actual_real_order],
            "swap_count": int(_swap_count(slot_order, actual_real_order)),
            "shape_diff_pixels": int(shape_diff_pixels),
            "candidate_ks": candidate_ks,
            "interior_decoy_count": int(
                sum(
                    min(int(piece["area"]) for piece in real_pieces)
                    < int(decoy["area"])
                    < max(int(piece["area"]) for piece in real_pieces)
                    for decoy in decoys
                )
            ),
        }
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_MAX_RETRIES):
        result = _gen_attempt(return_meta=False)
        if result is not None:
            return result
    raise RuntimeError(f"generate_input: failed after {GENERATE_MAX_RETRIES} attempts")
