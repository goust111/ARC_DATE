# Variant of 898e7135
# Dimension: D1 — 匹配条件：块级形状匹配（含旋转/翻转） (match_cond)
# Change: shape_match + rotation_flip — 精灵下采样为块级掩码后，与洞的形状在二面体群（4次旋转 × 水平翻转 = 8种变换）下比较，形状一致即配对
#         → 面积排序匹配：精灵按面积从大到小排序，洞按面积从大到小排序，第 i 大精灵对应第 i 大洞，不再做形状比较

from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np

from arc_utils import bbox, bbox_coords, components_mask, crop_bbox, find_components, get_bg, paste

"""
Design analysis for the D1 variant
=================================

spec:
  preserved_structure:
    - 背景色仍为 0，且必须是输入里出现次数最多的颜色。
    - 最大前景连通块仍是单色 frame；frame 内部仍包含若干 4-连通 hole slot。
    - 外部 piece 仍是由 k×k 实心块拼成的彩色 sprite；k 仍由最小实心方块 piece 推出。
    - 输出仍是把 frame 的内部布局按 k 放大，并把外部 piece 贴到对应 slot 的左上锚点。
  changed_dimension:
    - 原题通过“同面积 + 下采样后形状在 D4 下相等”决定 piece/slot 对应关系。
    - 本变体改为“只看面积排序”；slot 按 (-area, r0, c0) 排序，piece 按 (-pixel_area, input_bbox_top, input_bbox_left) 排序，然后逐一配对。
    - 一旦 piece 被选中，输出直接粘贴它在输入中观察到的朝向；不再额外搜索旋转/翻转。否则会把几何匹配逻辑重新带回 solver。

  concrete_trace:
    - 例子：frame 内有两个面积都为 4 的洞，左边是 L 形，右边是 1×4 横条。
    - 输入外部也有这两个 sprite，但散放顺序被反过来：横条的 bbox 更靠左上，L 形更靠右下。
    - 原逻辑：solver 会把横条旋到与横条洞匹配，把 L 形放回 L 形洞。
    - 新逻辑：solver 不看形状，只看面积排序，因此“更靠左上的面积 4 piece”会进“更靠左上的面积 4 slot”。
      如果这个更靠左上的 piece 恰好是横条，那么它会被贴到 L 形 slot 的左上角，视觉上会覆盖部分原本的 frame 墙体。
    - 由于新配对不再保证 bbox 与洞区域兼容，piece 也可能伸出放大后 frame 的右/下边界；输出本来就是裁剪后的放大 frame，
      所以实现上对 paste 使用 clip=True，只保留落在输出裁剪框内的部分。
    - 差异因此直接可见：同面积但不同形的 piece 被重排了，输出不再是“每个洞恰好被对应形状填满”。

  complexity_calibration:
    - 不能把任务简化成“只要面积互异即可”，否则面积排序与原题按形状匹配在效果上经常一致，D1 变化不可见。
    - 生成器因此强制至少出现一组“同面积、不同 D4 形状”的非 unit slot，并反转这组 piece 的输入顺序。
    - tie-break 只使用可见的空间位置（bbox 左上角），不使用颜色编号等不可见规则。
    - 仍保留 1-2 个 unit slot，用于稳定地暴露 k；它们不承担变体的核心信息。

  implicit_assumption_audit:
    - find largest component as frame:
      仍依赖 frame 是最大前景 4-连通块，生成器保证 piece 与 frame、piece 与 piece 之间至少隔 1 个背景格。
    - tight-crop holes inside frame bbox:
      仍依赖至少有一个 hole，且所有 hole 的并集触到内布局的四边，这样 tight crop 才能恢复原 layout 尺寸；生成器继续强制该条件。
    - infer k from smallest solid square piece:
      仍依赖至少一个 area=1 slot，对应一个 k×k 实心方块 piece；生成器保证这一点。
    - sort pieces by observed bbox:
      新增假设：同面积 piece 的输入位置必须形成稳定顺序。生成器把 frame 固定在右下，并按目标顺序单调扫描放置 piece，保证 solver 的排序键与设计一致。
    - zip(sorted_slots, sorted_pieces):
      新增假设：piece 面积多重集与 slot 面积多重集在乘以 k^2 后完全一致；生成器按 slot 直接生成 piece，因此恒成立。

  placement_legality:
    - frame 放在 30×30 画布右下角，留出左上大块空区给 piece 排列。
    - piece 使用真实 mask 做 8 邻域间隔检测，保证不重叠、不接触 frame、也不彼此接触。
    - distractor 仍是孤立单像素，颜色独立，且不会碰触其它对象。

  visual_observability:
    - 至少一组重复面积的非 unit slot 保证“排序”真的在起作用。
    - 这些重复面积 slot 形状在 D4 下互不等价，因此如果 piece 被交换，人眼能看到输出里的覆盖/错位。
    - 颜色数量仍然受控：frame 1 色，distractor 1 色，非 unit piece 各自不同色，unit piece 共享 1 色。

  cleanliness:
    - 不叠加多层标记；只有 frame、piece、单像素噪声三类对象。
    - 变体的视觉信号来自“同面积不同形 piece 被重排”，不是来自额外符号或颜色编码。
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

CANVAS_SIZE = 30

DISTRACTOR_RANGE = (8, 13)
DISTRACTOR_RETRY_MULT = 10
MIN_DISTRACTORS = 8

GENERATE_MAX_RETRIES = 300


def _slot_sort_key(slot: dict) -> tuple[int, int, int]:
    return (-int(slot["area"]), int(slot["r0"]), int(slot["c0"]))


def _piece_sort_key(piece: dict) -> tuple[int, int, int]:
    return (-int(piece["area"]), int(piece["place_r"]), int(piece["place_c"]))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Inflate the frame layout and fill slots by size ordering instead of shape matching.

    Rule change:
    - Slots are ranked by area descending.
    - Pieces are ranked by pixel area descending.
    - The i-th ranked piece is pasted into the i-th ranked slot at that slot's scaled top-left anchor.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    assert comps, "input must contain at least the frame"
    frame_comp = max(comps, key=lambda d: int(d["area"]))
    fr0, fc0, fr1, fc1 = map(int, frame_comp["bbox"])

    frame_cols = [int(g[int(r), int(c)]) for r, c in frame_comp["coords"]]
    bgc = int(Counter(frame_cols).most_common(1)[0][0])

    sub = crop_bbox(g, (fr0, fc0, fr1, fc1))
    hole = sub == int(bg)
    hole_box = bbox(hole)
    assert hole_box is not None, "frame must enclose at least one bg hole"
    lr0, lc0, lr1, lc1 = map(int, hole_box)
    layout = hole[lr0 : lr1 + 1, lc0 : lc1 + 1]

    slots = []
    for coords in components_mask(layout, connectivity=4):
        rr0, cc0, rr1, cc1 = bbox_coords(coords)
        rr0, cc0, rr1, cc1 = map(int, (rr0, cc0, rr1, cc1))
        h = int(rr1 - rr0 + 1)
        w = int(cc1 - cc0 + 1)
        sm = np.zeros((h, w), dtype=bool)
        sm[coords[:, 0] - rr0, coords[:, 1] - cc0] = True
        slots.append({"area": int(sm.sum()), "r0": int(rr0), "c0": int(cc0), "mask": sm})
    slots.sort(key=_slot_sort_key)
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
        ph, pw = map(int, patch.shape)
        pm = np.zeros((ph, pw), dtype=bool)
        pm[coords[:, 0] - pr0, coords[:, 1] - pc0] = True
        pieces.append(
            {
                "area": int(pm.sum()),
                "r0": int(pr0),
                "c0": int(pc0),
                "patch": patch,
                "mask": pm,
            }
        )
    assert pieces, "at least one non-frame piece must exist"

    k = None
    for p in pieces:
        h, w = map(int, p["mask"].shape)
        if h == w and int(p["mask"].sum()) == int(h * w):
            k = int(h) if k is None else min(int(k), int(h))
    k = int(k or 1)

    need_areas = sorted(int(s["area"]) * int(k) * int(k) for s in slots)
    have_areas = sorted(int(p["area"]) for p in pieces)
    assert need_areas == have_areas, "piece areas must match slot areas up to k^2 scaling"

    pieces.sort(key=lambda p: (-int(p["area"]), int(p["r0"]), int(p["c0"])))
    assert len(pieces) == len(slots), "every slot must have exactly one piece"

    out_h = (int(layout.shape[0]) + 2) * int(k)
    out_w = (int(layout.shape[1]) + 2) * int(k)
    out = np.full((out_h, out_w), int(bgc), dtype=int)

    for slot, piece in zip(slots, pieces):
        top = (int(slot["r0"]) + 1) * int(k)
        left = (int(slot["c0"]) + 1) * int(k)
        paste(out, piece["patch"], int(top), int(left), mask=piece["mask"], overwrite=True, clip=True)

    return out


def _canonical_shape_key(mask: np.ndarray) -> bytes:
    variants = []
    for kk in range(4):
        r = np.rot90(mask, kk)
        variants.append(r.tobytes() + bytes([r.shape[0], r.shape[1]]))
        fr = np.fliplr(r)
        variants.append(fr.tobytes() + bytes([fr.shape[0], fr.shape[1]]))
    return min(variants)


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
        while len(chosen) < target_size and frontier:
            cands = [
                p
                for p in frontier
                if 0 <= p[0] < h and 0 <= p[1] < w and not forbidden[p[0], p[1]] and p not in chosen
            ]
            if not cands:
                break
            nr, nc = random.choice(cands)
            chosen.add((nr, nc))
            frontier.discard((nr, nc))
            frontier.update({(nr - 1, nc), (nr + 1, nc), (nr, nc - 1), (nr, nc + 1)})
        if len(chosen) == target_size:
            return np.array(sorted(chosen))
    return None


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


def _mask_from_coords(coords: np.ndarray) -> tuple[int, int, np.ndarray]:
    r0, c0, r1, c1 = map(int, bbox_coords(coords))
    sm = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
    sm[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return int(r0), int(c0), sm


def _choose_non_unit_sizes(n_non_unit: int) -> list[int]:
    assert n_non_unit >= 2
    dup_area = random.choice(DUPLICATE_AREA_CHOICES)
    sizes = [int(dup_area), int(dup_area)]
    while len(sizes) < n_non_unit:
        sizes.append(random.choice(NON_UNIT_SIZE_CHOICES))
    random.shuffle(sizes)
    return sizes


def _gen_layout(h: int, w: int, n_slots: int, n_units: int) -> list[dict] | None:
    forbidden = np.zeros((h, w), dtype=bool)
    slots: list[dict] = []
    used_shape_keys: set[bytes] = set()

    for _ in range(n_units):
        for _ in range(UNIT_SLOT_RETRIES):
            coords = _grow_slot(forbidden, 1)
            if coords is None:
                continue
            r0, c0, sm = _mask_from_coords(coords)
            slots.append(
                {
                    "coords": coords,
                    "area": 1,
                    "r0": int(r0),
                    "c0": int(c0),
                    "mask": sm,
                    "shape_key": sm.tobytes(),
                }
            )
            _forbid_coords(forbidden, coords)
            break
        else:
            return None

    non_unit_sizes = _choose_non_unit_sizes(n_slots - n_units)
    for size in non_unit_sizes:
        for _ in range(NON_UNIT_SLOT_RETRIES):
            coords = _grow_slot(forbidden, int(size))
            if coords is None:
                continue
            r0, c0, sm = _mask_from_coords(coords)
            key = _canonical_shape_key(sm)
            if key in used_shape_keys:
                continue
            used_shape_keys.add(key)
            slots.append(
                {
                    "coords": coords,
                    "area": int(size),
                    "r0": int(r0),
                    "c0": int(c0),
                    "mask": sm,
                    "shape_key": key,
                }
            )
            _forbid_coords(forbidden, coords)
            break
        else:
            return None

    density = sum(int(s["area"]) for s in slots) / float(h * w)
    lo, hi = HOLE_DENSITY_RANGE
    if not (float(lo) <= density <= float(hi)):
        return None

    non_unit_slots = [s for s in slots if int(s["area"]) > 1]
    groups = defaultdict(list)
    for slot in non_unit_slots:
        groups[int(slot["area"])].append(slot)
    if not any(len(group) >= 2 for group in groups.values()):
        return None

    return slots


def _make_piece(slot: dict, color: int, k: int) -> dict:
    out_mask = np.kron(slot["mask"], np.ones((k, k), dtype=bool))
    patch = np.where(out_mask, int(color), 0).astype(int)
    mask = out_mask

    if int(slot["area"]) > 1:
        rot = random.randint(0, 3)
        patch = np.rot90(patch, rot)
        mask = np.rot90(mask, rot)
        if random.random() < 0.5:
            patch = np.fliplr(patch)
            mask = np.fliplr(mask)

    return {
        "color": int(color),
        "slot": slot,
        "patch": patch,
        "mask": mask,
        "area": int(mask.sum()),
    }


def _desired_piece_order(pieces: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for piece in pieces:
        groups[int(piece["slot"]["area"])].append(piece)

    ordered: list[dict] = []
    for area in sorted(groups.keys(), reverse=True):
        group = sorted(groups[area], key=lambda p: (int(p["slot"]["r0"]), int(p["slot"]["c0"])))
        if int(area) > 1 and len(group) >= 2:
            group = list(reversed(group))
        ordered.extend(group)
    return ordered


def _can_place_with_gap(occupied: np.ndarray, mask: np.ndarray, r0: int, c0: int) -> bool:
    h, w = map(int, mask.shape)
    hh, ww = occupied.shape
    if r0 < 0 or c0 < 0 or r0 + h > hh or c0 + w > ww:
        return False
    probe = np.zeros_like(occupied)
    probe[r0 : r0 + h, c0 : c0 + w] = mask
    return not bool(np.any(_dilate8(probe) & occupied))


def _build_input_canvas(
    pieces_in_order: list[dict],
    layout_mask: np.ndarray,
    layout_h: int,
    layout_w: int,
    frame_color: int,
    distractor_color: int,
) -> np.ndarray | None:
    canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=int)
    frame_h, frame_w = layout_h + 2, layout_w + 2
    frame_top = CANVAS_SIZE - frame_h - 1
    frame_left = CANVAS_SIZE - frame_w - 1
    if frame_top < 1 or frame_left < 1:
        return None

    frame_img = np.full((frame_h, frame_w), int(frame_color), dtype=int)
    frame_img[1:-1, 1:-1] = np.where(layout_mask, 0, int(frame_color))
    paste(canvas, frame_img, int(frame_top), int(frame_left))

    occupied = canvas != 0
    start_idx = 0
    for piece in pieces_in_order:
        placed = False
        for idx in range(start_idx, CANVAS_SIZE * CANVAS_SIZE):
            rr = idx // CANVAS_SIZE
            cc = idx % CANVAS_SIZE
            if not _can_place_with_gap(occupied, piece["mask"], int(rr), int(cc)):
                continue
            paste(canvas, piece["patch"], int(rr), int(cc), mask=piece["mask"], overwrite=True)
            occupied[rr : rr + piece["mask"].shape[0], cc : cc + piece["mask"].shape[1]] |= piece["mask"]
            piece["place_r"] = int(rr)
            piece["place_c"] = int(cc)
            start_idx = int(idx + 1)
            placed = True
            break
        if not placed:
            return None

    n_distractors = random.randint(*DISTRACTOR_RANGE)
    placed = 0
    for _ in range(n_distractors * DISTRACTOR_RETRY_MULT):
        if placed >= n_distractors:
            break
        rr = random.randint(0, CANVAS_SIZE - 1)
        cc = random.randint(0, CANVAS_SIZE - 1)
        if canvas[rr, cc] != 0:
            continue
        r_lo, r_hi = max(0, rr - 1), min(CANVAS_SIZE, rr + 2)
        c_lo, c_hi = max(0, cc - 1), min(CANVAS_SIZE, cc + 2)
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


def _build_expected_output(
    slot_order: list[dict],
    piece_order: list[dict],
    layout_h: int,
    layout_w: int,
    k: int,
    frame_color: int,
) -> np.ndarray:
    out = np.full(((layout_h + 2) * k, (layout_w + 2) * k), int(frame_color), dtype=int)
    for slot, piece in zip(slot_order, piece_order):
        top = (int(slot["r0"]) + 1) * int(k)
        left = (int(slot["c0"]) + 1) * int(k)
        paste(out, piece["patch"], int(top), int(left), mask=piece["mask"], overwrite=True, clip=True)
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
        piece_sources = [id(piece["slot"]) for piece in by_area_pieces[int(area)]]
        slot_sources = [id(slot) for slot in slots]
        if piece_sources != slot_sources:
            count += 1
    return int(count)


def _gen_attempt() -> np.ndarray | None:
    k = random.choice(K_CHOICES)
    layout_h = random.randint(*LAYOUT_H_RANGE)
    layout_w = random.randint(*LAYOUT_W_RANGE)
    n_slots = random.randint(*N_SLOTS_RANGE)
    n_units = random.randint(1, min(N_UNITS_MAX, n_slots - 2))

    palette = list(range(1, 10))
    random.shuffle(palette)
    frame_color = int(palette.pop())
    distractor_color = int(palette.pop())
    needed_piece_colors = (n_slots - n_units) + 1
    if len(palette) < needed_piece_colors:
        return None
    non_unit_colors = [int(c) for c in palette[: n_slots - n_units]]
    unit_color = int(palette[n_slots - n_units])

    slots = _gen_layout(layout_h, layout_w, n_slots, n_units)
    if slots is None:
        return None

    layout_mask = np.zeros((layout_h, layout_w), dtype=bool)
    for slot in slots:
        layout_mask[slot["coords"][:, 0], slot["coords"][:, 1]] = True
    if not (layout_mask[0].any() and layout_mask[-1].any() and layout_mask[:, 0].any() and layout_mask[:, -1].any()):
        return None

    unit_slots = [s for s in slots if int(s["area"]) == 1]
    non_unit_slots = [s for s in slots if int(s["area"]) > 1]

    pieces = [_make_piece(slot, non_unit_colors[i], k) for i, slot in enumerate(non_unit_slots)]
    pieces += [_make_piece(slot, unit_color, k) for slot in unit_slots]

    intended_piece_order = _desired_piece_order(pieces)
    slot_order = sorted(slots, key=_slot_sort_key)
    if _swap_count(slot_order, intended_piece_order) == 0:
        return None

    canvas = _build_input_canvas(
        pieces_in_order=intended_piece_order,
        layout_mask=layout_mask,
        layout_h=layout_h,
        layout_w=layout_w,
        frame_color=frame_color,
        distractor_color=distractor_color,
    )
    if canvas is None:
        return None

    actual_piece_order = sorted(pieces, key=_piece_sort_key)
    if _swap_count(slot_order, actual_piece_order) == 0:
        return None

    expected_out = _build_expected_output(slot_order, actual_piece_order, layout_h, layout_w, k, frame_color)

    try:
        produced = main(canvas)
    except Exception:
        return None
    if not np.array_equal(produced, expected_out):
        return None

    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_MAX_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"generate_input: failed after {GENERATE_MAX_RETRIES} attempts")
