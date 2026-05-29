# Variant of 898e7135
# Dimension: D2 — 源对象：精灵的结构与属性 (src)
# Change: 精灵由 k×k 实心块组成（k 为缩放因子），每个精灵单色，非单元精灵颜色互不相同，单元精灵（k×k 实心方块）可共享颜色 → 精灵作为空心轮廓（仅边缘像素）

from __future__ import annotations

from collections import Counter
from math import gcd
import random

import numpy as np
from scipy import ndimage

from arc_utils import bbox, bbox_coords, components_mask, crop_bbox, find_components, get_bg, paste, random_free_location


# Design analysis for the D2 hollow-outline variant
# --------------------------------------------------
# spec_like_analysis:
#   core_change:
#     original: 每个外部 piece 是由 k×k 实心块拼出的整块精灵；solver 直接按整块降采样。
#     variant: 每个外部 piece 先由同样的块级形状放大到像素级，再只保留整体外轮廓的 1 像素边界；
#       output 仍然恢复为实心放大形状。frame、slot、颜色分配、D4 匹配、裁剪输出都保持原题。
#   pixel_form:
#     frame: 仍是 1 像素厚单色矩形框，框内是 layout 的 hole/wall 布局。
#     slots: 仍是 frame 内的 4 连通 hole 区域，8 分离，面积 1-7。
#     pieces:
#       - 输入端 piece 的 bbox 仍等于对应实心放大形状的 bbox。
#       - piece 只有最外层边界像素着色，内部像素回到背景 0。
#       - 非单元 piece 仍然各自单色且颜色互不相同；单元 piece 仍可共享一种颜色。
#       - piece 仍散落在 frame 外。
#     output:
#       - 与原题相同：将对应 slot 按 k 放大后，用 piece 的颜色填成实心。
#   behavior_trace:
#     original_small_example:
#       slot_mask: 一个 block 级 L 形，k=3。
#       input_piece: 3 倍放大后得到 6x6/9x6 一类实心区域，solver 可直接把每个 3x3 block 看成全满。
#     variant_small_example:
#       input_piece: 同一个实心放大区域先抽出外轮廓，只剩单像素边框；内部 0 需要先被恢复。
#       solver_flow:
#         1. 取该 component 的轮廓 mask。
#         2. 对轮廓做 hole fill，恢复原实心放大形状。
#         3. 再按 k×k block 做整块判定，得到 block 级形状，与 slot 做 D4 匹配。
#       difference:
#         原题的 `downsample_blocks()` 直接吃输入 mask；本变体必须先恢复 filled mask，否则所有 block 都会“部分填充”而失败。
#   complexity_calibration:
#     kept:
#       - slot 生成、颜色使用、frame 提取、D4 匹配逻辑不变。
#       - 仍保留 1-2 个 unit slot，维持可推断的 k。
#     added:
#       - 轮廓恢复这一步是唯一新增推理。
#     anti_degeneracy:
#       - 生成器要求每个样例至少有一个 piece 满足 `outline_area < solid_area`，避免整例退化成原题视觉。
#       - 非单元 slot 仍强制 D4 形状唯一，避免恢复后匹配歧义。
#   implicit_assumption_audit:
#     broken_original_assumptions:
#       - `k` 不能再靠“最小实心正方形 piece 的面积等于 h*w”判断；改为 piece bbox 尺寸的 gcd。
#       - `downsample_blocks()` 不能直接对原 piece mask 操作；需先 fill holes。
#       - 输出粘贴不能再用输入轮廓 patch；需改为恢复后的实心 patch。
#     new_assumptions:
#       - 每个 piece 的 bbox 高宽都是 k 的倍数；generator 从 block 级 shape 放大后再取轮廓，保证成立。
#       - piece 表示的是无洞 polyomino 的外轮廓；原 spec 的 slot 面积 <= 7，不可能形成最小 8 环，因此 fill_holes 唯一。
#       - 存在 unit slot，因此所有 piece bbox 尺寸 gcd 正好等于 k；generator 保持原 spec 的 1-2 个 unit slot。
#   placement_legality:
#     risk: 空心 piece 的内部是 0，如果沿用原来的透明放置，其他对象可能落进轮廓内部。
#     fix:
#       - 单独维护 `reserve_canvas`。
#       - frame 在 reserve 上占用整个外接矩形，确保 piece 真正位于 frame bbox 外。
#       - piece 在 reserve 上按“恢复后的实心 mask”占位，再把视觉上的空心 patch 贴到实际 canvas。
#       - distractor 同时避开 reserve，不能落到 piece 腹腔或 frame 内部。
#   visual_observability:
#     - 人类能直接看到：输入外部精灵是空心边框，输出对应位置却是实心填充。
#     - 颜色仍只承担 object identity，不参与排序或数值规则。
#     - 每例都保留多个 piece，其中至少一个真正空心，避免仅靠单个 2x2 unit square 这种视觉上不变的退化情况。
#   cleanliness:
#     - 对象层次仍然只有 frame、pieces、singleton distractors 三类。
#     - 使用 reserve 避免轮廓内部混入其他元素，保证外观清爽。


def _dihedral_masks(mask: np.ndarray) -> list[np.ndarray]:
    """Unique D4 variants of a boolean mask."""
    base = []
    for kk in range(4):
        r = np.rot90(mask, kk)
        base.append(r)
        base.append(np.fliplr(r))
    uniq: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for m in base:
        key = (tuple(map(int, m.shape)), m.tobytes())
        if key in seen:
            continue
        seen.add(key)
        uniq.append(m)
    return uniq


def _outline_mask(filled: np.ndarray) -> np.ndarray:
    """1-pixel outer contour of a solid orthogonal shape."""
    boundary = filled.copy()
    if filled.shape[0] >= 3 and filled.shape[1] >= 3:
        interior = (
            filled[1:-1, 1:-1]
            & filled[:-2, 1:-1]
            & filled[2:, 1:-1]
            & filled[1:-1, :-2]
            & filled[1:-1, 2:]
        )
        boundary[1:-1, 1:-1] &= ~interior
    return boundary


def _recover_filled(mask: np.ndarray) -> np.ndarray:
    """Recover the solid shape from a hollow outline."""
    return ndimage.binary_fill_holes(mask.astype(bool)).astype(bool)


def _downsample_blocks(mask: np.ndarray, k: int) -> np.ndarray | None:
    """Convert a solid k-scaled pixel mask back to block level."""
    hh, ww = map(int, mask.shape)
    if hh % int(k) != 0 or ww % int(k) != 0:
        return None
    out_h, out_w = hh // int(k), ww // int(k)
    out = np.zeros((out_h, out_w), dtype=bool)
    for r in range(out_h):
        for c in range(out_w):
            block = mask[r * int(k):(r + 1) * int(k), c * int(k):(c + 1) * int(k)]
            if bool(block.all()):
                out[r, c] = True
            elif bool(block.any()):
                return None
    return out


def _random_reserved_location(reserve: np.ndarray, solid_mask: np.ndarray) -> tuple[int, int]:
    """Find a random location where the solid occupancy fits with 1-cell 8-neighborhood padding."""
    patch = solid_mask.astype(int)
    return random_free_location(
        reserve,
        patch,
        transparent=0,
        border=0,
        padding=1,
        padding_connectivity=8,
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Inflate a frame's hole-layout and fill each hole with its matching piece.

    Variant rule:
    - Outside pieces are hollow 1-pixel outlines of k-scaled slot shapes.
    - Recover each piece's solid mask, then match its block-level shape under D4.
    - Output remains the solid filled scaled layout.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    assert comps, "expected at least one foreground component"
    frame_comp = max(comps, key=lambda d: int(d["area"]))
    fr0, fc0, fr1, fc1 = map(int, frame_comp["bbox"])

    frame_cols = [int(g[int(r), int(c)]) for r, c in frame_comp["coords"]]
    bgc = int(Counter(frame_cols).most_common(1)[0][0])

    sub = crop_bbox(g, (fr0, fc0, fr1, fc1))
    hole = sub == int(bg)
    hole_box = bbox(hole)
    assert hole_box is not None, "frame must enclose at least one bg hole"
    lr0, lc0, lr1, lc1 = map(int, hole_box)
    layout = hole[lr0:lr1 + 1, lc0:lc1 + 1]

    slots = []
    for coords in components_mask(layout, connectivity=4):
        rr0, cc0, rr1, cc1 = map(int, bbox_coords(coords))
        sm = np.zeros((rr1 - rr0 + 1, cc1 - cc0 + 1), dtype=bool)
        sm[coords[:, 0] - rr0, coords[:, 1] - cc0] = True
        slots.append({"area": int(sm.sum()), "r0": int(rr0), "c0": int(cc0), "mask": sm})
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
        colors = patch[pm]
        color = int(Counter(map(int, colors.tolist())).most_common(1)[0][0])
        pieces.append({"mask": pm, "color": color})
    assert pieces, "at least one non-frame piece must exist"

    dims = [int(v) for p in pieces for v in p["mask"].shape]
    k = dims[0]
    for v in dims[1:]:
        k = gcd(int(k), int(v))
    assert k >= 2, "outline variant expects scale factor >= 2"

    candidates = []
    for pi, p in enumerate(pieces):
        for vm in _dihedral_masks(p["mask"]):
            filled = _recover_filled(vm)
            um = _downsample_blocks(filled, k)
            if um is None:
                continue
            vp = np.where(filled, int(p["color"]), 0).astype(int)
            candidates.append(
                {
                    "pi": int(pi),
                    "solid_area": int(filled.sum()),
                    "um": um,
                    "vp": vp,
                    "vm": filled,
                }
            )

    used_piece: set[int] = set()
    placements: list[tuple[dict, dict]] = []
    for s in slots:
        need_area = int(s["area"]) * int(k) * int(k)
        found = None
        for cand in candidates:
            if int(cand["pi"]) in used_piece:
                continue
            if int(cand["solid_area"]) != int(need_area):
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
        paste(out, cand["vp"], int(top), int(left), mask=cand["vm"], overwrite=True)
    return out


# --------------- generator (hollow-outline variant) ---------------

# --- tunables ----------------------------------------------------------------

K_CHOICES = [2, 3]
LAYOUT_H_RANGE = (4, 12)
LAYOUT_W_RANGE = (4, 8)
N_SLOTS_RANGE = (3, 6)
N_UNITS_MAX = 2
NON_UNIT_SIZE_CHOICES = [2, 3, 4, 5, 6, 7]

HOLE_DENSITY_RANGE = (0.20, 0.50)
UNIT_SLOT_RETRIES = 20
NON_UNIT_SLOT_RETRIES = 30
GROW_SEED_TRIES = 8

CANVAS_DIM_RANGE = (15, 30)
AREA_BUFFER_MULT = 4
AREA_BUFFER_CONST = 50

DISTRACTOR_RANGE = (8, 13)
DISTRACTOR_RETRY_MULT = 10
MIN_DISTRACTORS = 8

GENERATE_MAX_RETRIES = 300


def _canonical_shape_key(mask: np.ndarray) -> bytes:
    """Canonical bytes representation under D4 for a bool mask (tight-cropped)."""
    variants = []
    for kk in range(4):
        r = np.rot90(mask, kk)
        variants.append(r.tobytes() + bytes([r.shape[0], r.shape[1]]))
        fr = np.fliplr(r)
        variants.append(fr.tobytes() + bytes([fr.shape[0], fr.shape[1]]))
    return min(variants)


def _grow_slot(forbidden: np.ndarray, target_size: int) -> np.ndarray | None:
    """Grow a 4-connected slot of a given size starting from a random free cell."""
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
                p for p in frontier
                if 0 <= p[0] < h and 0 <= p[1] < w and not forbidden[p[0], p[1]] and p not in chosen
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


def _gen_layout(h: int, w: int, n_slots: int, n_units: int) -> list[dict] | None:
    """Generate layout slots; unit slots are first in the returned list."""
    forbidden = np.zeros((h, w), dtype=bool)
    slots: list[dict] = []
    used_shape_keys: set[bytes] = set()

    for _ in range(int(n_units)):
        for _ in range(UNIT_SLOT_RETRIES):
            coords = _grow_slot(forbidden, 1)
            if coords is None:
                continue
            slots.append({"coords": coords, "area": 1, "mask": np.ones((1, 1), dtype=bool)})
            _forbid_coords(forbidden, coords)
            break
        else:
            return None

    for _ in range(int(n_slots - n_units)):
        size = random.choice(NON_UNIT_SIZE_CHOICES)
        for _ in range(NON_UNIT_SLOT_RETRIES):
            coords = _grow_slot(forbidden, int(size))
            if coords is None:
                continue
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            sm = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
            sm[coords[:, 0] - r0, coords[:, 1] - c0] = True
            key = _canonical_shape_key(sm)
            if key in used_shape_keys:
                continue
            used_shape_keys.add(key)
            slots.append({"coords": coords, "area": int(size), "mask": sm, "shape_key": key})
            _forbid_coords(forbidden, coords)
            break
        else:
            return None

    density = sum(int(s["area"]) for s in slots) / float(h * w)
    lo, hi = HOLE_DENSITY_RANGE
    if not (lo <= density <= hi):
        return None
    return slots


def _make_piece(slot: dict, color: int, k: int, apply_dihedral: bool) -> dict:
    """Build solid output patch and hollow input patch for one slot."""
    solid_mask = np.kron(slot["mask"], np.ones((k, k), dtype=bool))
    in_solid_mask = solid_mask.copy()
    if apply_dihedral:
        rot = random.randint(0, 3)
        in_solid_mask = np.rot90(in_solid_mask, rot)
        if random.random() < 0.5:
            in_solid_mask = np.fliplr(in_solid_mask)
    outline_mask = _outline_mask(in_solid_mask)
    out_patch = np.where(solid_mask, int(color), 0).astype(int)
    in_patch = np.where(outline_mask, int(color), 0).astype(int)
    return {
        "color": int(color),
        "slot": slot,
        "out_patch": out_patch,
        "out_mask": solid_mask,
        "in_patch": in_patch,
        "in_mask": outline_mask,
        "reserve_mask": in_solid_mask,
        "hollow_gap": int(in_solid_mask.sum()) - int(outline_mask.sum()),
    }


def _build_expected_output(pieces: list[dict], layout_h: int, layout_w: int, k: int, frame_color: int) -> np.ndarray:
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
    layout_h: int,
    layout_w: int,
    frame_color: int,
    distractor_color: int,
) -> np.ndarray | None:
    """Build the input grid with a separate reserve mask for hollow shapes."""
    frame_h, frame_w = int(layout_h + 2), int(layout_w + 2)
    total_piece_area = sum(int(p["reserve_mask"].sum()) for p in pieces)
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
    reserve = np.zeros((canvas_h, canvas_w), dtype=int)

    frame_img = np.full((frame_h, frame_w), int(frame_color), dtype=int)
    frame_img[1:-1, 1:-1] = np.where(layout_mask, 0, int(frame_color))
    fr_max, fc_max = canvas_h - frame_h - 1, canvas_w - frame_w - 1
    if fr_max < 1 or fc_max < 1:
        return None
    fr = random.randint(1, fr_max)
    fc = random.randint(1, fc_max)
    paste(canvas, frame_img, fr, fc)
    paste(reserve, np.ones((frame_h, frame_w), dtype=int), fr, fc)

    for p in sorted(pieces, key=lambda item: -int(item["reserve_mask"].sum())):
        try:
            pr, pc = _random_reserved_location(reserve, p["reserve_mask"])
        except ValueError:
            return None
        paste(canvas, p["in_patch"], pr, pc, mask=p["in_mask"], overwrite=True)
        paste(reserve, np.ones_like(p["reserve_mask"], dtype=int), pr, pc, mask=p["reserve_mask"], overwrite=True)

    n_distractors = random.randint(*DISTRACTOR_RANGE)
    placed = 0
    for _ in range(n_distractors * DISTRACTOR_RETRY_MULT):
        if placed >= n_distractors:
            break
        rr = random.randint(0, canvas_h - 1)
        cc = random.randint(0, canvas_w - 1)
        if canvas[rr, cc] != 0 or reserve[rr, cc] != 0:
            continue
        r_lo, r_hi = max(0, rr - 1), min(canvas_h, rr + 2)
        c_lo, c_hi = max(0, cc - 1), min(canvas_w, cc + 2)
        if np.any(canvas[r_lo:r_hi, c_lo:c_hi] != 0):
            continue
        if np.any(reserve[r_lo:r_hi, c_lo:c_hi] != 0):
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
    layout_h = random.randint(*LAYOUT_H_RANGE)
    layout_w = random.randint(*LAYOUT_W_RANGE)
    n_slots = random.randint(*N_SLOTS_RANGE)
    n_units = random.randint(1, min(N_UNITS_MAX, n_slots - 1))

    palette = list(range(1, 10))
    random.shuffle(palette)
    frame_color = palette.pop()
    distractor_color = palette.pop()
    needed_piece_colors = (n_slots - n_units) + 1
    if len(palette) < needed_piece_colors:
        return None
    non_unit_colors = palette[:n_slots - n_units]
    unit_color = palette[n_slots - n_units]

    slots = _gen_layout(layout_h, layout_w, n_slots, n_units)
    if slots is None:
        return None
    unit_slots = slots[:n_units]
    non_unit_slots = slots[n_units:]

    layout_mask = np.zeros((layout_h, layout_w), dtype=bool)
    for s in slots:
        layout_mask[s["coords"][:, 0], s["coords"][:, 1]] = True
    if not (layout_mask[0].any() and layout_mask[-1].any() and layout_mask[:, 0].any() and layout_mask[:, -1].any()):
        return None

    pieces = [_make_piece(s, non_unit_colors[i], k, apply_dihedral=True) for i, s in enumerate(non_unit_slots)]
    pieces += [_make_piece(s, unit_color, k, apply_dihedral=False) for s in unit_slots]

    if not any(int(p["hollow_gap"]) > 0 for p in pieces):
        return None

    expected_out = _build_expected_output(pieces, layout_h, layout_w, k, frame_color)
    canvas = _build_input_canvas(pieces, layout_mask, layout_h, layout_w, frame_color, distractor_color)
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
