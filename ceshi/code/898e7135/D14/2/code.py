# Variant of 898e7135
# Dimension: D14 — 符号编码与图例 (symbol_instructions)
# Change: 无符号编码 — 所有信息直接通过几何和颜色表达 → 框架边框上的数字编码

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

# ---------------------------------------------------------------------------
# Design analysis for the D14 variant
#
# a) 具体化与行为追踪
# - 原题里，解题者从外部碎片的几何块大小直接推断缩放因子 k；框架只有纯色边框。
# - 本变体把 k 改为“边框数字编码”：
#   top 边有 k 个标记，bottom 有 k+2 个，left 有 k+4 个，right 有 k+6 个。
#   四边都用同一种 marker_color 标记，且只出现在框架最外层边框上。
# - 例：若 k=3，则四边标记数分别为 3/5/7/9。solver 先找到最大前景组件
#   作为框架，再统计四边边框上的 marker 像素数，得到 [3,5,7,9]，据此恢复 k=3。
#   后续槽洞提取、碎片匹配、输出拼接仍沿用原题逻辑。
# - 原逻辑在“推断 k”这一步会扫描最小实心正方形碎片；变体逻辑改为扫描框架边框。
#
# b) 复杂度校准
# - 只让 k 取 2 或 3，避免符号系统过多档位。
# - 四边计数总是相差 2，且都不同，视觉上明显，不需要额外图例。
# - 为了避免退化回原题，生成器移除 unit slot，并禁止实心正方形槽形状；
#   否则会重新出现 k×k 或 2k×2k 的几何捷径，D14 标记就不再必要。
#
# c) 隐式假设审查（对应 main 的关键前提）
# - `frame_comp = max(comps, key=area)` 仍成立：标记嵌在框架边框上，不产生更大的组件。
# - `bgc = dominant color of frame_comp` 仍成立：边框标记数受控，frame_color 仍是框架组件主色。
# - 原代码 `infer k from smallest solid square piece` 在变体中失效，因为生成器不再给出这种碎片。
# - `layout = tight crop of bg holes inside frame bbox` 仍成立：标记只在外边框，不进入洞区域。
# - `pieces = non-singleton components excluding frame` 仍成立：标记与框架连通，不会被当作碎片。
#
# d) 放置合法性
# - 标记直接替换边框像素，因此不会额外扩张 bbox，也不会破坏框架的 4 连通。
# - 四边标记只放在非角点边框上，避免一格同时属于两条边。
# - 布局尺寸收紧到 layout_h in [9,12], layout_w in [5,8]，保证 3/5/7/9 这样的计数
#   都能放进对应边的非角点长度内。
# - 外部碎片仍用 `random_free_location` 放置，和框架/彼此保持至少 1 格背景间距。
#
# e) 视觉可观察性
# - 人类在 2-4 个样例中能直接看到：输出放大比例变化时，框架四边的标记数量也同步变化。
# - top/bottom/left/right 的计数是规则的等差序列 [k, k+2, k+4, k+6]，从样例中可归纳。
# - 若仍保留 unit square，解题者可不看标记也读出 k，因此这里删除 unit slot 是必要配合修改。
#
# f) 可视化整洁度
# - 标记颜色只出现在框架外轮廓，不和洞、碎片、干扰点叠加。
# - 干扰点仍是单独颜色的孤立单像素，不会和边框标记混淆。
# - 输出图不保留标记，因此输出仍保持原题的干净“放大框架+填充碎片”形式。
#
# g) 颜色变化可推断性
# - 没有颜色映射规则；marker_color 只是计数载体。
# - 颜色只承担“框架 / 标记 / 碎片 / 干扰”四种角色，且每例都重复出现，足够可见。
#
# 9) 新 solver 的额外假设与生成器保证
# - 假设框架四边存在且仅存在一种 marker_color：生成器总是使用单一 marker_color。
# - 假设四边计数排序后为 [k, k+2, k+4, k+6]：生成器固定按该模板编码。
# - 假设没有 unit slot 或实心正方形碎片重新泄露 k：生成器显式禁止。
# ---------------------------------------------------------------------------


# --- tunables ----------------------------------------------------------------

K_CHOICES = [2, 3]
LAYOUT_H_RANGE = (9, 12)
LAYOUT_W_RANGE = (5, 8)
N_SLOTS_RANGE = (3, 6)
NON_UNIT_SIZE_CHOICES = [2, 3, 4, 5, 6, 7]

HOLE_DENSITY_RANGE = (0.20, 0.50)
NON_UNIT_SLOT_RETRIES = 40
GROW_SEED_TRIES = 10

CANVAS_DIM_RANGE = (18, 30)
AREA_BUFFER_MULT = 4
AREA_BUFFER_CONST = 60

DISTRACTOR_RANGE = (8, 13)
DISTRACTOR_RETRY_MULT = 10
MIN_DISTRACTORS = 8

GENERATE_MAX_RETRIES = 400


def _marker_counts_for_k(k: int) -> tuple[int, int, int, int]:
    """Return (top, bottom, left, right) marker counts for this k."""
    k = int(k)
    return k, k + 2, k + 4, k + 6


def _decode_k_from_frame_patch(frame_patch: np.ndarray, frame_color: int) -> int:
    """Decode k from the non-corner border marker counts."""
    top = frame_patch[0, 1:-1]
    bottom = frame_patch[-1, 1:-1]
    left = frame_patch[1:-1, 0]
    right = frame_patch[1:-1, -1]
    edges = [top, bottom, left, right]

    non_frame_vals = []
    counts = []
    for edge in edges:
        mask = edge != int(frame_color)
        counts.append(int(mask.sum()))
        non_frame_vals.extend(map(int, edge[mask].tolist()))

    assert non_frame_vals, "frame border must contain marker pixels"
    assert len(set(non_frame_vals)) == 1, "all border markers must share one color"

    sorted_counts = sorted(counts)
    k = int(sorted_counts[0])
    assert sorted_counts == [k, k + 2, k + 4, k + 6], "border code must be an arithmetic ladder"
    return k


def main(grid: np.ndarray) -> np.ndarray:
    """
    Inflate the frame's hole-layout and fill each hole with its matching piece.

    D14 change:
    - k is no longer inferred from a square piece.
    - k is decoded from numeric marker counts on the four frame edges.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    assert comps, "grid must contain foreground components"
    frame_comp = max(comps, key=lambda d: int(d["area"]))
    fr0, fc0, fr1, fc1 = map(int, frame_comp["bbox"])

    frame_cols = [int(g[int(r), int(c)]) for r, c in frame_comp["coords"]]
    frame_color = int(Counter(frame_cols).most_common(1)[0][0])

    sub = crop_bbox(g, (fr0, fc0, fr1, fc1))
    k = _decode_k_from_frame_patch(sub, frame_color)

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
        ph, pw = map(int, patch.shape)
        pm = np.zeros((ph, pw), dtype=bool)
        pm[coords[:, 0] - pr0, coords[:, 1] - pc0] = True
        pieces.append({"area": int(pm.sum()), "mask": pm, "patch": patch})
    assert pieces, "at least one non-frame piece must exist"

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
        base = []
        for kk in range(4):
            p = np.rot90(patch, int(kk))
            m = np.rot90(mask, int(kk))
            base.append((p, m))
            base.append((np.fliplr(p), np.fliplr(m)))
        uniq = []
        seen = set()
        for p, m in base:
            key = (m.shape, m.tobytes())
            if key in seen:
                continue
            seen.add(key)
            uniq.append((p, m))
        return uniq

    candidates = []
    for pi, piece in enumerate(pieces):
        for vp, vm in dihedral_variants(piece["patch"], piece["mask"]):
            um = downsample_blocks(vm)
            if um is None:
                continue
            candidates.append(
                {"pi": int(pi), "area": int(piece["area"]), "um": um, "vp": vp, "vm": vm}
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

    out_h = (int(layout.shape[0]) + 2) * int(k)
    out_w = (int(layout.shape[1]) + 2) * int(k)
    out = np.full((out_h, out_w), int(frame_color), dtype=int)

    for slot, cand in placements:
        top = (int(slot["r0"]) + 1) * int(k)
        left = (int(slot["c0"]) + 1) * int(k)
        paste(out, cand["vp"], int(top), int(left), mask=cand["vm"], overwrite=True)

    return out


# --- generator helpers -------------------------------------------------------


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
    """Grow a 4-connected slot from a random free seed."""
    h, w = forbidden.shape
    free_cells = np.argwhere(~forbidden)
    if free_cells.size == 0:
        return None
    free_list = [tuple(map(int, cell)) for cell in free_cells.tolist()]
    random.shuffle(free_list)
    for r0, c0 in free_list[:GROW_SEED_TRIES]:
        chosen = {(r0, c0)}
        frontier = {(r0 - 1, c0), (r0 + 1, c0), (r0, c0 - 1), (r0, c0 + 1)}
        while len(chosen) < int(target_size) and frontier:
            cands = [
                (rr, cc)
                for rr, cc in frontier
                if 0 <= rr < h and 0 <= cc < w and not forbidden[rr, cc] and (rr, cc) not in chosen
            ]
            if not cands:
                break
            nr, nc = random.choice(cands)
            chosen.add((nr, nc))
            frontier.discard((nr, nc))
            frontier.update({(nr - 1, nc), (nr + 1, nc), (nr, nc - 1), (nr, nc + 1)})
        if len(chosen) == int(target_size):
            return np.array(sorted(chosen), dtype=int)
    return None


def _dilate8(mask: np.ndarray) -> np.ndarray:
    """Dilate a bool mask by 1 cell (8-connectivity)."""
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
    """Mark coords and their 8-neighbors as forbidden (in-place)."""
    cell_mask = np.zeros_like(forbidden)
    cell_mask[coords[:, 0], coords[:, 1]] = True
    forbidden |= _dilate8(cell_mask)


def _is_forbidden_shape(mask: np.ndarray) -> bool:
    """Avoid shapes that would recreate the original square-based k shortcut."""
    h, w = map(int, mask.shape)
    return h == w and bool(mask.all())


def _gen_layout(h: int, w: int, n_slots: int) -> list[dict] | None:
    """
    Generate a layout with n_slots 4-connected slots.
    Slots are pairwise 8-separated and D4-unique, and none is a filled square.
    """
    forbidden = np.zeros((h, w), dtype=bool)
    slots: list[dict] = []
    used_shape_keys: set[bytes] = set()

    for _ in range(int(n_slots)):
        size = random.choice(NON_UNIT_SIZE_CHOICES)
        for _ in range(NON_UNIT_SLOT_RETRIES):
            coords = _grow_slot(forbidden, size)
            if coords is None:
                continue
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            sm = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
            sm[coords[:, 0] - r0, coords[:, 1] - c0] = True
            if _is_forbidden_shape(sm):
                continue
            key = _canonical_shape_key(sm)
            if key in used_shape_keys:
                continue
            used_shape_keys.add(key)
            slots.append({"coords": coords, "area": int(size), "mask": sm, "shape_key": key})
            _forbid_coords(forbidden, coords)
            break
        else:
            return None

    density = sum(int(slot["area"]) for slot in slots) / float(h * w)
    lo, hi = HOLE_DENSITY_RANGE
    if not (lo <= density <= hi):
        return None
    return slots


def _make_piece(slot: dict, color: int, k: int) -> dict:
    """Build output-side and input-side patches for one slot."""
    out_mask = np.kron(slot["mask"], np.ones((k, k), dtype=bool))
    out_patch = np.where(out_mask, color, 0).astype(int)

    rot = random.randint(0, 3)
    in_mask = np.rot90(out_mask, rot)
    in_patch = np.rot90(out_patch, rot)
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
    }


def _build_expected_output(
    pieces: list[dict], layout_h: int, layout_w: int, k: int, frame_color: int
) -> np.ndarray:
    """Assemble the expected output grid."""
    out = np.full(((layout_h + 2) * k, (layout_w + 2) * k), int(frame_color), dtype=int)
    for piece in pieces:
        coords = piece["slot"]["coords"]
        top = (int(coords[:, 0].min()) + 1) * int(k)
        left = (int(coords[:, 1].min()) + 1) * int(k)
        paste(out, piece["out_patch"], top, left, mask=piece["out_mask"], overwrite=True)
    return out


def _edge_marker_positions(frame_len: int, count: int) -> list[int]:
    """Choose `count` positions along one edge, excluding corners."""
    interior_len = int(frame_len) - 2
    assert 0 <= int(count) <= interior_len
    if count == 0:
        return []
    return [1 + (i * interior_len) // int(count) for i in range(int(count))]


def _build_frame_image(
    layout_mask: np.ndarray, frame_color: int, marker_color: int, k: int
) -> np.ndarray:
    """Build the frame with border markers encoding k."""
    layout_h, layout_w = map(int, layout_mask.shape)
    frame_h, frame_w = layout_h + 2, layout_w + 2
    frame = np.full((frame_h, frame_w), int(frame_color), dtype=int)
    frame[1:-1, 1:-1] = np.where(layout_mask, 0, int(frame_color))

    top_n, bottom_n, left_n, right_n = _marker_counts_for_k(k)
    for c in _edge_marker_positions(frame_w, top_n):
        frame[0, int(c)] = int(marker_color)
    for c in _edge_marker_positions(frame_w, bottom_n):
        frame[-1, int(c)] = int(marker_color)
    for r in _edge_marker_positions(frame_h, left_n):
        frame[int(r), 0] = int(marker_color)
    for r in _edge_marker_positions(frame_h, right_n):
        frame[int(r), -1] = int(marker_color)
    return frame


def _build_input_canvas(
    pieces: list[dict],
    layout_mask: np.ndarray,
    layout_h: int,
    layout_w: int,
    k: int,
    frame_color: int,
    marker_color: int,
    distractor_color: int,
) -> np.ndarray | None:
    """Build the input grid: frame + scattered pieces + singleton distractors."""
    frame_h, frame_w = layout_h + 2, layout_w + 2
    total_piece_area = sum(int(piece["in_mask"].sum()) for piece in pieces)
    needed_cells = frame_h * frame_w + AREA_BUFFER_MULT * total_piece_area + AREA_BUFFER_CONST

    min_h = max(CANVAS_DIM_RANGE[0], frame_h + 2)
    min_w = max(CANVAS_DIM_RANGE[0], frame_w + 2)
    canvas_h = random.randint(min_h, CANVAS_DIM_RANGE[1])
    canvas_w = random.randint(min_w, CANVAS_DIM_RANGE[1])
    while (
        canvas_h * canvas_w < needed_cells
        and (canvas_h < CANVAS_DIM_RANGE[1] or canvas_w < CANVAS_DIM_RANGE[1])
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

    frame_img = _build_frame_image(layout_mask, frame_color, marker_color, k)
    fr_max, fc_max = canvas_h - frame_h - 1, canvas_w - frame_w - 1
    if fr_max < 1 or fc_max < 1:
        return None
    paste(canvas, frame_img, random.randint(1, fr_max), random.randint(1, fc_max))

    for piece in sorted(pieces, key=lambda p: -int(p["in_mask"].sum())):
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
        rr, cc = random.randint(0, canvas_h - 1), random.randint(0, canvas_w - 1)
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
    layout_h = random.randint(*LAYOUT_H_RANGE)
    layout_w = random.randint(*LAYOUT_W_RANGE)
    n_slots = random.randint(*N_SLOTS_RANGE)

    palette = list(range(1, 10))
    random.shuffle(palette)
    if len(palette) < n_slots + 3:
        return None
    piece_colors = palette[:n_slots]
    frame_color = palette[n_slots]
    distractor_color = palette[n_slots + 1]
    marker_color = palette[n_slots + 2]

    slots = _gen_layout(layout_h, layout_w, n_slots)
    if slots is None:
        return None

    layout_mask = np.zeros((layout_h, layout_w), dtype=bool)
    for slot in slots:
        layout_mask[slot["coords"][:, 0], slot["coords"][:, 1]] = True
    if not (
        layout_mask[0].any()
        and layout_mask[-1].any()
        and layout_mask[:, 0].any()
        and layout_mask[:, -1].any()
    ):
        return None

    pieces = [_make_piece(slot, piece_colors[i], k) for i, slot in enumerate(slots)]
    expected_out = _build_expected_output(pieces, layout_h, layout_w, k, frame_color)

    canvas = _build_input_canvas(
        pieces,
        layout_mask,
        layout_h,
        layout_w,
        k,
        frame_color,
        marker_color,
        distractor_color,
    )
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
    raise RuntimeError(
        "generate_input: failed to generate after %d attempts" % int(GENERATE_MAX_RETRIES)
    )
