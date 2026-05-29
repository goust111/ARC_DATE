# Variant of 898e7135
# Dimension: D14 — 符号编码与图例 (symbol_instructions)
# Change: 无符号编码 — 所有信息直接通过几何和颜色表达 → 标记像素指示旋转方向
#
# Design analysis
# ---------------
# spec:
#   preserved_backbone:
#     - input 仍然是一个单色框架，内部是若干洞区域，外部散落对应精灵，背景仍为 0
#     - 输出仍然是将框架内洞布局按 k 放大后，把外部精灵放回对应洞位置
#     - k 仍限定为 2 或 3，散落精灵仍由 k×k 块放大得到
#   d14_change:
#     - 每个槽位内部不再只有背景洞；槽位中会有 1 个红色标记格（固定颜色 2）
#     - 每个外部精灵也带 1 个红色标记像素；该像素属于精灵本体，随旋转一起移动
#     - 允许的变换从原题的 D4（旋转/翻转）收缩为纯旋转 C4。原因：
#       D14 要表达“标记点给出旋转方向”，若再允许翻转，少量样例下规则会混入镜像自由度，
#       不利于人类从 2-4 组样例中稳定归纳
#   concrete_visualization:
#     - 框内每个槽位仍是 4 连通洞；某一个洞格改涂成红色，表示该槽位的“对齐点”
#     - 外部每个精灵仍是单色块状 polyomino，但其中某个 k×k 块的一个角像素改成红色
#     - 精灵在输入里会被随机旋转；解题时需要把精灵旋回去，使红点对应到槽位红格
#     - 输出中红点保留，因此旋转是否正确在输出里可见，不再退化为“转不转都一样”
#   behavior_trace_example:
#     - 原题：某个 L 形块只需通过外轮廓与 L 形洞匹配，朝向完全由洞形状决定
#     - 变体：两个槽位可以拥有相同的 L 形外轮廓，但红格位于不同拐角；外部也有两个同形
#       不同红点位置、不同颜色的 L 形精灵。此时仅靠形状无法决定“哪一块进哪一个槽位”，
#       必须比较红点在块级坐标中的相对位置，并据此确定旋转角度
#   complexity_calibration:
#     - 为了让 D14 变化不是装饰，生成器强制每题至少有一对“同形不同标记”的槽位
#     - 还会强制至少一个精灵在输入中发生非零旋转，保证标记确实控制旋转而不是长期恒等
#     - 颜色数量仍受控：红色 2 只做标记；框颜色、干扰色、各精灵主体色都与红色分离
#   implicit_assumption_review:
#     - 原 solver 的 `hole = sub == bg` 假设“洞中只有背景色”；变体下失效，因为洞里有红色标记格
#       新 solver 改为 `(sub == bg) | (sub == MARKER_COLOR)`，并额外提取每个槽位唯一标记格
#     - 原 solver 的 `dihedral_variants` 假设允许翻转；变体下改为纯旋转 `rotation_variants`
#     - 原 solver 仅按 downsample 后的布尔形状匹配；变体下这不够，因为同形槽位可能重复。
#       新 solver 必须同时比较 `(块级形状, 槽位标记格相对位置)`
#     - 原 solver 从最小实心方块推断 k；变体去掉 unit slot 以避免“单格槽位无法表达旋转”。
#       新 solver 直接在 k∈{2,3} 上尝试，并选择能够完成完整一一匹配的那个 k
#   placement_legality:
#     - 生成器仍保证框架与外部精灵之间至少隔 1 格背景，精灵彼此也至少隔 1 格背景
#     - 框组件面积被约束为大于任一单个精灵面积，从而 `largest component == frame` 仍成立
#     - 槽位总洞布局仍需触达四条边，保证 tight-crop 后的 layout 与生成时一致
#   visual_observability:
#     - 红色只承担“说明性符号”这一单一职责，视觉上非常醒目
#     - 同形槽位+同形精灵的重复让人能肉眼看出：不是颜色配色、也不是位置排序，而是“红点对齐”
#   new_solver_assumptions_to_guarantee:
#     - 每个槽位恰有 1 个红色标记格
#     - 每个精灵恰有 1 个红色标记像素
#     - 至少存在一个非零旋转精灵
#     - 至少存在一对相同旋转等价形状但不同标记位置的槽位
#     - 对错误的 k，无法完成全部槽位的一一匹配；生成器通过 oracle 校验保证这一点

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox, bbox_coords, components_mask, crop_bbox, find_components, get_bg, paste, random_free_location


MARKER_COLOR = 2
BG_COLOR = 0
K_CHOICES = (2, 3)


def _rotation_variants(patch: np.ndarray, mask: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    return [(np.rot90(patch, k), np.rot90(mask, k)) for k in range(4)]


def _downsample_blocks(mask: np.ndarray, k: int) -> np.ndarray | None:
    hh, ww = mask.shape
    if hh % k != 0 or ww % k != 0:
        return None
    out_h, out_w = hh // k, ww // k
    out = np.zeros((out_h, out_w), dtype=bool)
    for r in range(out_h):
        for c in range(out_w):
            block = mask[r * k : (r + 1) * k, c * k : (c + 1) * k]
            if bool(block.all()):
                out[r, c] = True
            elif bool(block.any()):
                return None
    return out


def _marker_block_from_patch(patch: np.ndarray, mask: np.ndarray, k: int) -> tuple[int, int] | None:
    coords = np.argwhere((patch == MARKER_COLOR) & mask)
    if coords.shape[0] != 1:
        return None
    rr, cc = map(int, coords[0])
    return rr // k, cc // k


def _shape_key_rot(mask: np.ndarray) -> bytes:
    variants = []
    for k in range(4):
        rot = np.rot90(mask, k)
        variants.append(rot.tobytes() + bytes([rot.shape[0], rot.shape[1]]))
    return min(variants)


def _rotation_state_count(mask: np.ndarray, marker: tuple[int, int]) -> int:
    states = set()
    for k in range(4):
        rm = np.rot90(mask, k)
        pts = np.argwhere(mask)
        h0, w0 = mask.shape
        coords = []
        for r, c in pts:
            rr, cc = int(r), int(c)
            if k == 0:
                coords.append((rr, cc))
            elif k == 1:
                coords.append((w0 - 1 - cc, rr))
            elif k == 2:
                coords.append((h0 - 1 - rr, w0 - 1 - cc))
            else:
                coords.append((cc, h0 - 1 - rr))
        mr, mc = marker
        if k == 0:
            new_marker = (mr, mc)
        elif k == 1:
            new_marker = (w0 - 1 - mc, mr)
        elif k == 2:
            new_marker = (h0 - 1 - mr, w0 - 1 - mc)
        else:
            new_marker = (mc, h0 - 1 - mr)
        states.add((rm.shape, rm.tobytes(), new_marker))
    return len(states)


def _match_slots(slots: list[dict], pieces: list[dict], k: int) -> list[tuple[dict, dict]] | None:
    candidates_by_slot: list[list[dict]] = []
    for s in slots:
        want_area = int(s["area"]) * k * k
        cur = []
        for pi, p in enumerate(pieces):
            if int(p["area"]) != want_area:
                continue
            for vp, vm in _rotation_variants(p["patch"], p["mask"]):
                um = _downsample_blocks(vm, k)
                if um is None or um.shape != s["mask"].shape or not np.array_equal(um, s["mask"]):
                    continue
                marker = _marker_block_from_patch(vp, vm, k)
                if marker != s["marker"]:
                    continue
                cur.append({"pi": pi, "vp": vp, "vm": vm})
        if not cur:
            return None
        candidates_by_slot.append(cur)

    order = sorted(range(len(slots)), key=lambda i: (len(candidates_by_slot[i]), -int(slots[i]["area"]), int(slots[i]["r0"]), int(slots[i]["c0"])))
    used: set[int] = set()
    ans: list[tuple[dict, dict]] = [None] * len(slots)  # type: ignore

    def dfs(idx: int) -> bool:
        if idx == len(order):
            return True
        si = order[idx]
        for cand in candidates_by_slot[si]:
            pi = int(cand["pi"])
            if pi in used:
                continue
            used.add(pi)
            ans[si] = (slots[si], cand)
            if dfs(idx + 1):
                return True
            used.remove(pi)
            ans[si] = None  # type: ignore
        return False

    return ans if dfs(0) else None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Inflate a marked frame's hole-layout and fill each marked hole with its matching marked piece.

    Input structure:
    - bg = most common color (0 in generated instances)
    - largest 4-connected multicolor component is the frame; red cells inside holes are slot markers
    - each outside piece is a k×k-block-scaled sprite with exactly one red marker pixel

    Rule:
    - treat both bg and red marker cells inside the frame as hole territory
    - extract each slot's block-level shape and its unique marker cell
    - rotate each external piece (no flips) until both the downsampled shape and marker location match a slot
    - output the k-scaled frame with the rotated pieces pasted in place, preserving the red marker pixels
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    assert comps, "expected at least one foreground component"
    frame_comp = max(comps, key=lambda d: int(d["area"]))
    fr0, fc0, fr1, fc1 = map(int, frame_comp["bbox"])

    frame_cols = [int(g[int(r), int(c)]) for r, c in frame_comp["coords"] if int(g[int(r), int(c)]) != MARKER_COLOR]
    assert frame_cols, "frame must contain non-marker color"
    frame_color = int(Counter(frame_cols).most_common(1)[0][0])

    sub = crop_bbox(g, (fr0, fc0, fr1, fc1))
    hole_like = (sub == bg) | (sub == MARKER_COLOR)
    marker_mask = sub == MARKER_COLOR
    hole_box = bbox(hole_like)
    assert hole_box is not None, "frame must enclose at least one hole-like cell"
    lr0, lc0, lr1, lc1 = map(int, hole_box)
    layout = hole_like[lr0 : lr1 + 1, lc0 : lc1 + 1]
    layout_markers = marker_mask[lr0 : lr1 + 1, lc0 : lc1 + 1]

    slots = []
    for coords in components_mask(layout, connectivity=4):
        rr0, cc0, rr1, cc1 = bbox_coords(coords)
        rr0, cc0, rr1, cc1 = map(int, (rr0, cc0, rr1, cc1))
        h = rr1 - rr0 + 1
        w = cc1 - cc0 + 1
        sm = np.zeros((h, w), dtype=bool)
        sm[coords[:, 0] - rr0, coords[:, 1] - cc0] = True
        marker_coords = [(int(r), int(c)) for r, c in coords if bool(layout_markers[int(r), int(c)])]
        assert len(marker_coords) == 1, "each slot must have exactly one marker cell"
        mr, mc = marker_coords[0]
        slots.append(
            {
                "area": int(sm.sum()),
                "r0": rr0,
                "c0": cc0,
                "mask": sm,
                "marker": (mr - rr0, mc - cc0),
            }
        )
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
        if int(np.sum((patch == MARKER_COLOR) & pm)) != 1:
            continue
        pieces.append({"area": int(pm.sum()), "mask": pm, "patch": patch})
    assert len(pieces) == len(slots), "need exactly one marked piece per slot"

    placements = None
    chosen_k = None
    for k in K_CHOICES:
        cand = _match_slots(slots, pieces, k)
        if cand is not None:
            placements = cand
            chosen_k = int(k)
            break
    assert placements is not None and chosen_k is not None, "no valid k/matching found"

    out_h = (int(layout.shape[0]) + 2) * chosen_k
    out_w = (int(layout.shape[1]) + 2) * chosen_k
    out = np.full((out_h, out_w), frame_color, dtype=int)

    for s, cand in placements:
        top = (int(s["r0"]) + 1) * chosen_k
        left = (int(s["c0"]) + 1) * chosen_k
        paste(out, cand["vp"], top, left, mask=cand["vm"], overwrite=True)

    return out


# --------------- generator ---------------

LAYOUT_H_RANGE = (5, 12)
LAYOUT_W_RANGE = (4, 8)
N_SLOTS_RANGE = (3, 6)
SLOT_SIZE_CHOICES = [2, 3, 4, 5, 6]
HOLE_DENSITY_RANGE = (0.20, 0.50)

CANVAS_DIM_RANGE = (16, 30)
AREA_BUFFER_MULT = 4
AREA_BUFFER_CONST = 60

DISTRACTOR_RANGE = (8, 13)
DISTRACTOR_RETRY_MULT = 12
MIN_DISTRACTORS = 8

PLACE_RETRIES = 80
SHAPE_RETRIES = 60
GENERATE_MAX_RETRIES = 400


def _coords_to_mask(coords: set[tuple[int, int]]) -> np.ndarray:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, c0 = min(rows), min(cols)
    h = max(rows) - r0 + 1
    w = max(cols) - c0 + 1
    out = np.zeros((h, w), dtype=bool)
    for r, c in coords:
        out[r - r0, c - c0] = True
    return out


def _grow_polyomino(area: int) -> np.ndarray:
    chosen = {(0, 0)}
    frontier = {(1, 0), (-1, 0), (0, 1), (0, -1)}
    while len(chosen) < area:
        nr, nc = random.choice(list(frontier))
        frontier.remove((nr, nc))
        chosen.add((nr, nc))
        for nxt in ((nr - 1, nc), (nr + 1, nc), (nr, nc - 1), (nr, nc + 1)):
            if nxt not in chosen:
                frontier.add(nxt)
    return _coords_to_mask(chosen)


def _marker_candidates(mask: np.ndarray) -> list[tuple[int, int]]:
    cands = []
    for r, c in np.argwhere(mask):
        marker = (int(r), int(c))
        if _rotation_state_count(mask, marker) >= 2:
            cands.append(marker)
    return cands


def _choose_markers_for_duplicate(mask: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]] | None:
    cands = _marker_candidates(mask)
    random.shuffle(cands)
    for i in range(len(cands)):
        for j in range(i + 1, len(cands)):
            if cands[i] != cands[j]:
                return cands[i], cands[j]
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


def _place_mask(forbidden: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    h, w = mask.shape
    H, W = forbidden.shape
    if h > H or w > W:
        return None
    positions = [(r0, c0) for r0 in range(H - h + 1) for c0 in range(W - w + 1)]
    random.shuffle(positions)
    coords_local = np.argwhere(mask)
    for r0, c0 in positions[:PLACE_RETRIES]:
        coords = coords_local + np.array([r0, c0])
        if np.any(forbidden[coords[:, 0], coords[:, 1]]):
            continue
        return coords
    return None


def _make_template(area: int, used_shape_keys: set[bytes]) -> tuple[np.ndarray, tuple[int, int]] | None:
    for _ in range(SHAPE_RETRIES):
        mask = _grow_polyomino(area)
        key = _shape_key_rot(mask)
        if key in used_shape_keys:
            continue
        cands = _marker_candidates(mask)
        if not cands:
            continue
        used_shape_keys.add(key)
        return mask, random.choice(cands)
    return None


def _gen_layout(h: int, w: int, n_slots: int) -> list[dict] | None:
    used_shape_keys: set[bytes] = set()

    dup_mask = None
    dup_markers = None
    for _ in range(SHAPE_RETRIES):
        area = random.choice(SLOT_SIZE_CHOICES)
        cand = _grow_polyomino(area)
        markers = _choose_markers_for_duplicate(cand)
        if markers is None:
            continue
        dup_mask = cand
        dup_markers = markers
        used_shape_keys.add(_shape_key_rot(cand))
        break
    if dup_mask is None or dup_markers is None:
        return None

    templates = [
        {"mask": dup_mask, "marker": dup_markers[0], "dup": True},
        {"mask": dup_mask, "marker": dup_markers[1], "dup": True},
    ]
    for _ in range(n_slots - 2):
        made = _make_template(random.choice(SLOT_SIZE_CHOICES), used_shape_keys)
        if made is None:
            return None
        mask, marker = made
        templates.append({"mask": mask, "marker": marker, "dup": False})

    templates.sort(key=lambda t: -int(np.sum(t["mask"])))

    forbidden = np.zeros((h, w), dtype=bool)
    slots = []
    for t in templates:
        coords = _place_mask(forbidden, t["mask"])
        if coords is None:
            return None
        coords = coords.astype(int)
        rr0, cc0 = coords.min(axis=0)
        marker = (int(rr0 + t["marker"][0]), int(cc0 + t["marker"][1]))
        slots.append(
            {
                "coords": coords,
                "area": int(coords.shape[0]),
                "mask": t["mask"].copy(),
                "shape_key": _shape_key_rot(t["mask"]),
                "marker": marker,
                "marker_rel": (int(t["marker"][0]), int(t["marker"][1])),
                "dup": bool(t["dup"]),
            }
        )
        _forbid_coords(forbidden, coords)

    layout_mask = np.zeros((h, w), dtype=bool)
    layout_markers = np.zeros((h, w), dtype=bool)
    for s in slots:
        layout_mask[s["coords"][:, 0], s["coords"][:, 1]] = True
        mr, mc = s["marker"]
        layout_markers[mr, mc] = True

    density = float(np.sum(layout_mask)) / float(h * w)
    lo, hi = HOLE_DENSITY_RANGE
    if not (lo <= density <= hi):
        return None
    if not (layout_mask[0].any() and layout_mask[-1].any() and layout_mask[:, 0].any() and layout_mask[:, -1].any()):
        return None
    if sum(1 for s in slots if s["dup"]) < 2:
        return None
    return slots


def _marker_anchor(k: int) -> tuple[int, int]:
    return 0, 0


def _make_piece(slot: dict, color: int, k: int, rot: int) -> dict:
    out_mask = np.kron(slot["mask"], np.ones((k, k), dtype=bool))
    out_patch = np.where(out_mask, color, BG_COLOR).astype(int)
    ar, ac = _marker_anchor(k)
    mr, mc = slot["marker_rel"]
    out_patch[mr * k + ar, mc * k + ac] = MARKER_COLOR

    in_mask = np.rot90(out_mask, rot)
    in_patch = np.rot90(out_patch, rot)

    return {
        "color": int(color),
        "slot": slot,
        "rotation": int(rot),
        "out_patch": out_patch,
        "out_mask": out_mask,
        "in_patch": in_patch,
        "in_mask": in_mask,
    }


def _build_expected_output(pieces: list[dict], layout_h: int, layout_w: int, k: int, frame_color: int) -> np.ndarray:
    out = np.full(((layout_h + 2) * k, (layout_w + 2) * k), frame_color, dtype=int)
    for p in pieces:
        coords = p["slot"]["coords"]
        top = (int(coords[:, 0].min()) + 1) * k
        left = (int(coords[:, 1].min()) + 1) * k
        paste(out, p["out_patch"], top, left, mask=p["out_mask"], overwrite=True)
    return out


def _build_input_canvas(
    pieces: list[dict],
    layout_mask: np.ndarray,
    layout_markers: np.ndarray,
    layout_h: int,
    layout_w: int,
    frame_color: int,
    distractor_color: int,
) -> np.ndarray | None:
    frame_h, frame_w = layout_h + 2, layout_w + 2
    total_piece_area = sum(int(np.sum(p["in_mask"])) for p in pieces)
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

    inner = np.where(layout_mask, BG_COLOR, frame_color).astype(int)
    inner[layout_markers] = MARKER_COLOR
    frame_img = np.full((frame_h, frame_w), frame_color, dtype=int)
    frame_img[1:-1, 1:-1] = inner

    fr_max, fc_max = canvas_h - frame_h - 1, canvas_w - frame_w - 1
    if fr_max < 1 or fc_max < 1:
        return None
    paste(canvas, frame_img, random.randint(1, fr_max), random.randint(1, fc_max))

    for p in sorted(pieces, key=lambda x: -int(np.sum(x["in_mask"]))):
        try:
            pr, pc = random_free_location(
                canvas,
                p["in_patch"],
                transparent=BG_COLOR,
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
        if canvas[rr, cc] != BG_COLOR:
            continue
        r0, r1 = max(0, rr - 1), min(canvas_h, rr + 2)
        c0, c1 = max(0, cc - 1), min(canvas_w, cc + 2)
        if np.any(canvas[r0:r1, c0:c1] != BG_COLOR):
            continue
        canvas[rr, cc] = distractor_color
        placed += 1
    if placed < MIN_DISTRACTORS:
        return None

    vals, counts = np.unique(canvas, return_counts=True)
    if int(vals[np.argmax(counts)]) != BG_COLOR:
        return None
    return canvas


def _gen_attempt() -> np.ndarray | None:
    k = random.choice(K_CHOICES)
    layout_h = random.randint(*LAYOUT_H_RANGE)
    layout_w = random.randint(*LAYOUT_W_RANGE)
    n_slots = random.randint(*N_SLOTS_RANGE)
    if n_slots < 3:
        return None

    palette = [c for c in range(1, 10) if c != MARKER_COLOR]
    random.shuffle(palette)
    if len(palette) < n_slots + 2:
        return None
    frame_color = palette.pop()
    distractor_color = palette.pop()
    piece_colors = [palette.pop() for _ in range(n_slots)]

    slots = _gen_layout(layout_h, layout_w, n_slots)
    if slots is None:
        return None

    hole_count = sum(int(s["area"]) for s in slots)
    frame_area = (layout_h + 2) * (layout_w + 2) - hole_count + n_slots
    max_piece_area = max(int(s["area"]) * k * k for s in slots)
    if frame_area <= max_piece_area:
        return None

    rotations = [random.randint(0, 3) for _ in slots]
    if not any(r != 0 for r in rotations):
        rotations[random.randrange(len(rotations))] = random.randint(1, 3)

    pieces = [
        _make_piece(slot=s, color=piece_colors[i], k=k, rot=rotations[i])
        for i, s in enumerate(slots)
    ]

    if not any(_rotation_state_count(s["mask"], s["marker_rel"]) >= 2 and p["rotation"] != 0 for s, p in zip(slots, pieces)):
        return None

    dup_slots = [s for s in slots if s["dup"]]
    if len(dup_slots) < 2:
        return None
    dup_signatures = {(s["shape_key"], s["marker_rel"]) for s in dup_slots}
    if len(dup_signatures) < 2:
        return None

    layout_mask = np.zeros((layout_h, layout_w), dtype=bool)
    layout_markers = np.zeros((layout_h, layout_w), dtype=bool)
    for s in slots:
        layout_mask[s["coords"][:, 0], s["coords"][:, 1]] = True
        mr, mc = s["marker"]
        layout_markers[mr, mc] = True

    expected_out = _build_expected_output(pieces, layout_h, layout_w, k, frame_color)
    canvas = _build_input_canvas(
        pieces=pieces,
        layout_mask=layout_mask,
        layout_markers=layout_markers,
        layout_h=layout_h,
        layout_w=layout_w,
        frame_color=frame_color,
        distractor_color=distractor_color,
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
    raise RuntimeError(f"generate_input: failed after {GENERATE_MAX_RETRIES} attempts")
