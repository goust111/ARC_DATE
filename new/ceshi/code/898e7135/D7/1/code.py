from __future__ import annotations

# Variant of 898e7135
# Dimension: D7 — 精灵-洞基数关系 (meta)
# Change: 严格 1:1 — 每个精灵恰好对应一个洞，每个洞恰好对应一个精灵，总数相等... → 一精灵对多洞（1:many）
#
# Design analysis
# ----------------
# spec_variant:
#   invariants_kept:
#     - 背景仍为 0，最大前景连通块仍是单色框架。
#     - 框内仍是由背景色洞区域组成的 layout，输出仍是 layout 按 k 放大后再加 1 格框边。
#     - 框外精灵仍由 k*k 实心块拼出；精灵与洞仍按块级形状在 D4（旋转/镜像）下匹配。
#     - 噪声仍是单像素干扰色，不进入输出。
#   D7 core change:
#     - 原题：每个洞消费一个不同的物理精灵，`used_piece` 保证 1:1。
#     - 变体：同一 canonical shape class 只提供一个物理精灵，但框内可以出现 2-3 个同类洞；
#       solver 必须按“形状类”找精灵，并允许在多个洞上重复使用该精灵。
#   concrete behavior trace:
#     - 例：框内有 3 个 L 形洞和 1 个单位洞，框外只有 1 个 L 形精灵和 1 个单位方块。
#     - 原逻辑：第一个 L 洞匹配后把该精灵加入 `used_piece`，第二个 L 洞会因“精灵已用过”而失败。
#     - 新逻辑：第一个 L 洞和第二个/第三个 L 洞都映射到同一个 canonical L-shape class，
#       每个洞各自选取该精灵的一个 D4 朝向并粘贴到输出，因此同色 L 形会在输出里出现多次。
#   complexity calibration:
#     - 为了让 1:many 肉眼可见，生成器强制至少一个非单位形状重复 2-3 次。
#     - 仍保留 1 个单位洞和若干唯一非单位洞，避免所有洞都一样导致样例过于单薄。
#     - 每个物理精灵对应唯一 canonical shape class，避免“两个不同颜色精灵同形”带来歧义。
#   old-main assumption audit:
#     - `used_piece` 依赖“每个形状只会用一次”；变体中不成立，必须删除。
#     - 按面积和 exact mask 顺序扫描候选在原题无害，因为总会剩下唯一未用精灵；
#       变体中如果不先按 canonical shape class 分组，重复洞可能错误抢占无关候选。
#     - `k` 仍由最小实心方块推断，因此生成器继续保证存在一个单位洞对应的 k*k 方块精灵。
#   placement legality:
#     - 洞在 layout 中仍要求彼此 8-分离；生成器按具体 mask 放置并维护 forbidden=dilate8(occupied)。
#     - frame 必须仍是最大的前景连通块，因此生成器拒绝任何“某块精灵面积 >= frame 面积”的样例。
#   visual observability:
#     - 输入里稳定出现“洞数 > 精灵数”，且重复洞是非单位形状；输出里同色同形状会出现多次。
#     - 重复洞可采用不同朝向，保证“同一精灵可多次旋转/翻转后填洞”的规则能在少量样例中被看到。
#   new-main assumptions introduced:
#     - 每个 canonical shape class 至多对应一个物理精灵。
#     - 每个重复洞都真的是某个物理精灵的 D4 变体，而不是仅面积相同。
#     - 生成器显式保证以上条件，因此 solver 的按类复用不会发生歧义。

from collections import Counter
import random

import numpy as np

from arc_utils import bbox, bbox_coords, components_mask, crop_bbox, find_components, get_bg, paste, random_free_location


# --- shared helpers ----------------------------------------------------------

def _canonical_shape_key(mask: np.ndarray) -> tuple[int, int, bytes]:
    """Canonical key for a tight bool mask under D4."""
    variants: list[tuple[int, int, bytes]] = []
    for kk in range(4):
        rot = np.rot90(mask, kk)
        variants.append((int(rot.shape[0]), int(rot.shape[1]), rot.tobytes()))
        flip = np.fliplr(rot)
        variants.append((int(flip.shape[0]), int(flip.shape[1]), flip.tobytes()))
    return min(variants)


def _dihedral_variants(patch: np.ndarray, mask: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Unique D4 transforms for a patch/mask pair."""
    raw: list[tuple[np.ndarray, np.ndarray]] = []
    for kk in range(4):
        rp = np.rot90(patch, kk)
        rm = np.rot90(mask, kk)
        raw.append((rp, rm))
        raw.append((np.fliplr(rp), np.fliplr(rm)))
    uniq: list[tuple[np.ndarray, np.ndarray]] = []
    seen: set[tuple[int, int, bytes]] = set()
    for patch_v, mask_v in raw:
        key = (int(mask_v.shape[0]), int(mask_v.shape[1]), mask_v.tobytes())
        if key in seen:
            continue
        seen.add(key)
        uniq.append((patch_v, mask_v))
    return uniq


def _dihedral_masks(mask: np.ndarray) -> list[np.ndarray]:
    zero_patch = np.zeros(mask.shape, dtype=int)
    return [mv for _, mv in _dihedral_variants(zero_patch, mask)]


def _downsample_blocks(mask: np.ndarray, k: int) -> np.ndarray | None:
    hh, ww = map(int, mask.shape)
    if hh % int(k) != 0 or ww % int(k) != 0:
        return None
    out = np.zeros((hh // int(k), ww // int(k)), dtype=bool)
    for r in range(out.shape[0]):
        for c in range(out.shape[1]):
            block = mask[r * int(k) : (r + 1) * int(k), c * int(k) : (c + 1) * int(k)]
            if bool(block.all()):
                out[r, c] = True
            elif bool(block.any()):
                return None
    return out


def _tight_mask_from_coords(coords: np.ndarray) -> tuple[np.ndarray, int, int]:
    rr0, cc0, rr1, cc1 = bbox_coords(coords)
    rr0, cc0, rr1, cc1 = map(int, (rr0, cc0, rr1, cc1))
    tight = np.zeros((rr1 - rr0 + 1, cc1 - cc0 + 1), dtype=bool)
    tight[coords[:, 0] - rr0, coords[:, 1] - cc0] = True
    return tight, int(rr0), int(cc0)


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


def _parse_grid(grid: np.ndarray) -> tuple[np.ndarray, int, np.ndarray, list[dict], list[dict], int]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    assert comps, "must contain at least one foreground component"
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
        sm, rr0, cc0 = _tight_mask_from_coords(coords)
        slots.append(
            {
                "area": int(sm.sum()),
                "r0": int(rr0),
                "c0": int(cc0),
                "mask": sm,
                "shape_key": _canonical_shape_key(sm),
            }
        )
    slots.sort(key=lambda s: (-int(s["area"]), int(s["r0"]), int(s["c0"])))
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
        pm = np.zeros(patch.shape, dtype=bool)
        pm[coords[:, 0] - pr0, coords[:, 1] - pc0] = True
        pieces.append({"area": int(pm.sum()), "mask": pm, "patch": patch})
    assert pieces, "at least one non-frame piece must exist"

    k = None
    for p in pieces:
        h, w = map(int, p["mask"].shape)
        if h == w and int(p["mask"].sum()) == int(h * w):
            k = int(h) if k is None else min(int(k), int(h))
    k = int(k or 1)
    return g, int(frame_color), layout, slots, pieces, int(k)


def _variant_metrics(grid: np.ndarray) -> dict[str, int]:
    """Small diagnostic used after implementation to verify 1:many behavior."""
    _g, _frame_color, _layout, slots, pieces, k = _parse_grid(grid)
    slot_counts = Counter(s["shape_key"] for s in slots)
    piece_counts = Counter()
    for p in pieces:
        um = _downsample_blocks(p["mask"], k)
        assert um is not None
        piece_counts[_canonical_shape_key(um)] += 1
    return {
        "slot_count": int(len(slots)),
        "piece_count": int(len(pieces)),
        "reuse_gap": int(len(slots) - len(pieces)),
        "duplicate_shape_classes": int(sum(v > 1 for v in slot_counts.values())),
        "max_slot_multiplicity": int(max(slot_counts.values())),
        "max_piece_class_count": int(max(piece_counts.values())),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Inflate the frame's hole-layout and fill each hole with its matching piece class.

    Variant behavior:
    - Slots are matched by canonical block-shape class under D4.
    - A single physical piece may be reused for multiple slots of the same class.
    """
    _g, frame_color, layout, slots, pieces, k = _parse_grid(grid)

    piece_classes: dict[tuple[int, int, bytes], dict] = {}
    for pi, p in enumerate(pieces):
        variants = []
        class_key = None
        for vp, vm in _dihedral_variants(p["patch"], p["mask"]):
            um = _downsample_blocks(vm, k)
            if um is None:
                continue
            key = _canonical_shape_key(um)
            if class_key is None:
                class_key = key
            else:
                assert key == class_key, "all D4 variants of a piece must share a canonical shape class"
            variants.append({"pi": int(pi), "vp": vp, "vm": vm, "um": um})
        assert class_key is not None, "piece must be expressible as k*k blocks"
        assert class_key not in piece_classes, "variant assumes at most one physical piece per shape class"
        piece_classes[class_key] = {"area": int(p["area"]), "variants": variants}

    placements: list[tuple[dict, dict]] = []
    for s in slots:
        found_class = piece_classes.get(s["shape_key"])
        assert found_class is not None, "every slot shape class must have a matching piece"
        need_area = int(s["area"]) * int(k) * int(k)
        assert int(found_class["area"]) == int(need_area), "piece area must match slot area after scaling"
        found_variant = None
        for cand in found_class["variants"]:
            if cand["um"].shape != s["mask"].shape:
                continue
            if np.array_equal(cand["um"], s["mask"]):
                found_variant = cand
                break
        assert found_variant is not None, "slot must match some D4 orientation of its class piece"
        placements.append((s, found_variant))

    out_h = (int(layout.shape[0]) + 2) * int(k)
    out_w = (int(layout.shape[1]) + 2) * int(k)
    out = np.full((out_h, out_w), int(frame_color), dtype=int)

    for s, cand in placements:
        top = (int(s["r0"]) + 1) * int(k)
        left = (int(s["c0"]) + 1) * int(k)
        paste(out, cand["vp"], int(top), int(left), mask=cand["vm"], overwrite=True)

    return out


# --------------- generator ---------------------------------------------------

K_CHOICES = [2, 3]
BOARD_H_RANGE = (4, 12)
BOARD_W_RANGE = (4, 8)
REPEAT_MULT_CHOICES = [2, 3]
UNIQUE_NON_UNIT_BY_REPEAT = {2: [1, 2], 3: [1]}
NON_UNIT_SIZE_CHOICES = [2, 3, 4, 5, 6, 7]

HOLE_DENSITY_RANGE = (0.20, 0.50)
GROW_SEED_TRIES = 8
SHAPE_RETRIES = 40
PLACEMENT_RETRIES = 40

CANVAS_DIM_RANGE = (15, 30)
AREA_BUFFER_MULT = 4
AREA_BUFFER_CONST = 50

DISTRACTOR_RANGE = (8, 13)
DISTRACTOR_RETRY_MULT = 10
MIN_DISTRACTORS = 8

GENERATE_MAX_RETRIES = 400


def _grow_slot(forbidden: np.ndarray, target_size: int) -> np.ndarray | None:
    """Grow a 4-connected shape of the requested size in a free board."""
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
            chosen.add((int(nr), int(nc)))
            frontier.discard((int(nr), int(nc)))
            frontier.update({(nr - 1, nc), (nr + 1, nc), (nr, nc - 1), (nr, nc + 1)})
        if len(chosen) == target_size:
            return np.array(sorted(chosen), dtype=int)
    return None


def _gen_shape_mask(target_size: int, used_keys: set[tuple[int, int, bytes]], *, require_multi_variant: bool) -> np.ndarray | None:
    side = max(4, int(target_size) + 2)
    free = np.zeros((side, side), dtype=bool)
    for _ in range(SHAPE_RETRIES):
        coords = _grow_slot(free, int(target_size))
        if coords is None:
            continue
        mask, _r0, _c0 = _tight_mask_from_coords(coords)
        key = _canonical_shape_key(mask)
        if key in used_keys:
            continue
        if require_multi_variant and len(_dihedral_masks(mask)) < 2:
            continue
        return mask
    return None


def _place_slot_instances(board_h: int, board_w: int, instances: list[dict]) -> tuple[list[dict], np.ndarray] | None:
    order = list(range(len(instances)))
    order.sort(key=lambda idx: (-int(instances[idx]["area"]), random.random()))

    placed: list[dict | None] = [None] * len(instances)
    occupied = np.zeros((board_h, board_w), dtype=bool)
    forbidden = np.zeros((board_h, board_w), dtype=bool)

    for idx in order:
        mask = instances[idx]["mask"]
        mh, mw = map(int, mask.shape)
        candidates = []
        for r0 in range(board_h - mh + 1):
            for c0 in range(board_w - mw + 1):
                trial = np.zeros((board_h, board_w), dtype=bool)
                trial[r0 : r0 + mh, c0 : c0 + mw] = mask
                if bool((trial & forbidden).any()):
                    continue
                candidates.append((int(r0), int(c0), trial))
        if not candidates:
            return None
        r0, c0, trial = random.choice(candidates)
        occupied |= trial
        forbidden |= _dilate8(trial)
        coords = np.argwhere(trial)
        placed[idx] = {
            "class_id": instances[idx]["class_id"],
            "area": int(instances[idx]["area"]),
            "shape_key": instances[idx]["shape_key"],
            "coords": coords,
            "mask": mask,
            "r0": int(r0),
            "c0": int(c0),
        }

    full_box = bbox(occupied)
    assert full_box is not None
    br0, bc0, br1, bc1 = map(int, full_box)
    layout_mask = occupied[br0 : br1 + 1, bc0 : bc1 + 1]
    layout_h, layout_w = map(int, layout_mask.shape)
    if not (BOARD_H_RANGE[0] <= layout_h <= BOARD_H_RANGE[1]):
        return None
    if not (BOARD_W_RANGE[0] <= layout_w <= BOARD_W_RANGE[1]):
        return None

    slots: list[dict] = []
    for item in placed:
        assert item is not None
        shifted = item["coords"] - np.array([br0, bc0], dtype=int)
        tight_mask, rr0, cc0 = _tight_mask_from_coords(shifted)
        slots.append(
            {
                "class_id": item["class_id"],
                "area": int(item["area"]),
                "coords": shifted,
                "mask": tight_mask,
                "r0": int(rr0),
                "c0": int(cc0),
                "shape_key": _canonical_shape_key(tight_mask),
            }
        )

    density = sum(int(s["area"]) for s in slots) / float(layout_h * layout_w)
    if not (HOLE_DENSITY_RANGE[0] <= density <= HOLE_DENSITY_RANGE[1]):
        return None
    return slots, layout_mask


def _make_piece(base_mask: np.ndarray, color: int, k: int, *, apply_dihedral: bool) -> dict:
    if int(base_mask.sum()) == 1:
        patch = np.full((k, k), int(color), dtype=int)
        mask = np.ones((k, k), dtype=bool)
        return {"color": int(color), "in_patch": patch, "in_mask": mask}

    out_mask = np.kron(base_mask, np.ones((k, k), dtype=bool))
    out_patch = np.where(out_mask, int(color), 0).astype(int)
    if not apply_dihedral:
        return {"color": int(color), "in_patch": out_patch, "in_mask": out_mask}

    variants = _dihedral_variants(out_patch, out_mask)
    in_patch, in_mask = random.choice(variants)
    return {"color": int(color), "in_patch": in_patch, "in_mask": in_mask}


def _build_expected_output(slots: list[dict], layout_h: int, layout_w: int, k: int, frame_color: int, color_by_class: dict[str, int]) -> np.ndarray:
    out = np.full(((layout_h + 2) * k, (layout_w + 2) * k), int(frame_color), dtype=int)
    for slot in slots:
        color = int(color_by_class[slot["class_id"]])
        slot_mask = np.kron(slot["mask"], np.ones((k, k), dtype=bool))
        slot_patch = np.where(slot_mask, color, 0).astype(int)
        top = (int(slot["r0"]) + 1) * int(k)
        left = (int(slot["c0"]) + 1) * int(k)
        paste(out, slot_patch, int(top), int(left), mask=slot_mask, overwrite=True)
    return out


def _build_input_canvas(pieces: list[dict], layout_mask: np.ndarray, frame_color: int, distractor_color: int) -> np.ndarray | None:
    layout_h, layout_w = map(int, layout_mask.shape)
    frame_h, frame_w = layout_h + 2, layout_w + 2
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

    fr_max, fc_max = canvas_h - frame_h - 1, canvas_w - frame_w - 1
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
        paste(canvas, piece["in_patch"], int(pr), int(pc), mask=piece["in_mask"], overwrite=True)

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
    repeat_mult = random.choice(REPEAT_MULT_CHOICES)
    n_unique_non_unit = random.choice(UNIQUE_NON_UNIT_BY_REPEAT[repeat_mult])
    board_h = random.randint(*BOARD_H_RANGE)
    board_w = random.randint(*BOARD_W_RANGE)

    palette = list(range(1, 10))
    random.shuffle(palette)
    frame_color = palette.pop()
    distractor_color = palette.pop()

    used_keys: set[tuple[int, int, bytes]] = set()
    classes: list[dict] = []

    repeat_size = random.choice([2, 3, 4, 5, 6])
    repeat_mask = _gen_shape_mask(repeat_size, used_keys, require_multi_variant=True)
    if repeat_mask is None:
        return None
    repeat_key = _canonical_shape_key(repeat_mask)
    used_keys.add(repeat_key)
    repeat_color = palette.pop()
    classes.append({"class_id": "repeat", "base_mask": repeat_mask, "color": int(repeat_color), "shape_key": repeat_key})

    for idx in range(n_unique_non_unit):
        size = random.choice(NON_UNIT_SIZE_CHOICES)
        mask = _gen_shape_mask(size, used_keys, require_multi_variant=False)
        if mask is None:
            return None
        key = _canonical_shape_key(mask)
        used_keys.add(key)
        classes.append({"class_id": f"uniq_{idx}", "base_mask": mask, "color": int(palette.pop()), "shape_key": key})

    unit_mask = np.ones((1, 1), dtype=bool)
    classes.append({"class_id": "unit", "base_mask": unit_mask, "color": int(palette.pop()), "shape_key": _canonical_shape_key(unit_mask)})

    instances: list[dict] = []
    repeat_variants = _dihedral_masks(repeat_mask)
    for _ in range(repeat_mult):
        chosen = random.choice(repeat_variants)
        instances.append({"class_id": "repeat", "mask": chosen, "area": int(chosen.sum()), "shape_key": repeat_key})
    for cls in classes:
        if not str(cls["class_id"]).startswith("uniq_"):
            continue
        chosen = random.choice(_dihedral_masks(cls["base_mask"]))
        instances.append({"class_id": str(cls["class_id"]), "mask": chosen, "area": int(chosen.sum()), "shape_key": cls["shape_key"]})
    instances.append({"class_id": "unit", "mask": unit_mask, "area": 1, "shape_key": _canonical_shape_key(unit_mask)})
    random.shuffle(instances)

    placed = None
    for _ in range(PLACEMENT_RETRIES):
        placed = _place_slot_instances(board_h, board_w, instances)
        if placed is not None:
            break
    if placed is None:
        return None
    slots, layout_mask = placed
    layout_h, layout_w = map(int, layout_mask.shape)

    color_by_class = {str(cls["class_id"]): int(cls["color"]) for cls in classes}
    pieces: list[dict] = []
    for cls in classes:
        apply_dihedral = str(cls["class_id"]) != "unit"
        piece = _make_piece(cls["base_mask"], int(cls["color"]), int(k), apply_dihedral=apply_dihedral)
        piece["class_id"] = str(cls["class_id"])
        pieces.append(piece)

    frame_area = int((layout_h + 2) * (layout_w + 2) - sum(int(s["area"]) for s in slots))
    if frame_area <= max(int(p["in_mask"].sum()) for p in pieces):
        return None

    metrics = Counter(s["class_id"] for s in slots)
    if int(metrics["repeat"]) < 2:
        return None
    if len(pieces) >= len(slots):
        return None

    expected = _build_expected_output(slots, layout_h, layout_w, k, frame_color, color_by_class)
    canvas = _build_input_canvas(pieces, layout_mask, frame_color, distractor_color)
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
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GENERATE_MAX_RETRIES} attempts")
