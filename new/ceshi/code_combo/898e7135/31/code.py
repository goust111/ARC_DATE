# Combo variant of 898e7135 (index 31)
# combo_spec: D4/3+D14/1
# Name: 标记辅助旋转缩放
#
# Design analysis
# ---------------
# spec baseline:
#   - 保留原题的大骨架：输入里有一个单色矩形框，框内是若干 4 连通洞槽位，框外散落对应精灵，背景仍为 0
#   - 输出仍然是把框内布局按 k 放大后，用外部精灵去填入各槽位；k 仍限制为 2 或 3
#   - 原题靠“缩放后的块形 + D4 形状匹配”完成放回；这次要把 D4/3 与 D14/1 融合成一个真正联动的两步规则
#
# concrete fused rule:
#   - 框内每个槽位仍有 1 个红色标记格；它属于槽位本身，求解时把它当作洞的一部分
#   - 框外每个精灵仍是 k×k 块状 polyomino，但其中恰有 1 个红色单像素；这个像素不再只是“对齐点”
#   - 红色单像素落在所属 k×k 块的四角之一；四个角编码 4 个旋转角度
#   - 先在块级读取“该转多少”，把精灵的块级形状和标记格一起旋转，再重新按 k 放大，最后把红点放回统一锚点
#   - 因此正确规则不是“把输入中的整张放大精灵直接旋转后贴回去”，而是“读符号 -> 旋转块级形状 -> 再缩放”
#
# behavior trace:
#   - 原题：外部精灵本身已经是缩放后的块图，只要在像素层面找和洞形状匹配的朝向即可
#   - 单独 D14/1：红点给旋转提示，但如果仍然把整张缩放图直接旋转，红点只是跟着图一起转
#   - 本融合版：例如某个 L 形槽位在框内右上角是红格；对应输入精灵先以另一朝向散落在外部，且红点落在其某个块的右上角。
#     这个右上角不是目标位置，而是“旋转说明书”。solver 必须先解码为“旋转 180 度”，在块级把 L 形和标记格一起转正，
#     再按 k 重建放大图，并把输出红点固定回每个标记块的左上角。若误用“先缩放后旋转”，主体形状也许仍对，但红点角落会错。
#
# complexity calibration:
#   - 去掉 unit slot。面积为 1 的槽位无法稳定承载“旋转说明 + 标记格”这两个自由度，且会削弱融合变化的可见性
#   - 每题至少造出一对“同形但标记格不同”的槽位，避免只靠形状就能唯一匹配
#   - 每题强制至少 1 个非零旋转指令，并要求至少一种旋转指令出现至少两次，给人类足够的重复支撑
#   - 只保留 4 种纯旋转，不引入翻转；否则“角标记编码角度”与镜像自由度会混在一起，2-4 个样例下不稳定
#
# implicit assumption review:
#   - 原 solver 的 `hole = sub == bg` 失效，因为槽位里现在混入红色标记格；需改成 `(sub == bg) | (sub == MARKER_COLOR)`
#   - 原 solver 把外部精灵视为“最终可直接粘贴的朝向”；本变体失效，因为输入红点是指令，输出红点要回到统一锚点
#   - 原 solver 从像素级候选里枚举 D4 变体；这里不能直接旋转整张输入 patch，而要先按 k 下采样到块级，再根据红点角解码旋转
#   - 原 solver 可由最小实心方块推断 k；这里去掉 unit slot 后改为在 k∈{2,3} 上尝试完整匹配，并由生成器 oracle 保证只有真 k 可行
#   - 新 solver 额外假设：每个有效精灵恰有 1 个红点，且该红点恰落在某个 k×k 块的四个角之一；生成器显式保证
#
# placement legality:
#   - 框、精灵、干扰点都仍放在背景上，彼此保持至少 1 格背景间隔，避免 4/8 连通粘连
#   - 槽位在 layout 内保持 8 分离，防止 tight-crop 后槽位连成一片
#   - 生成器检查框的单色边框面积大于任一单个精灵面积，维持“largest component == frame”
#
# visual observability and cleanliness:
#   - 红色只承担一种职责：框内是“目标标记格”，框外是“旋转指令像素”，输出里是“重建后的标记像素”
#   - 同形不同标记格让人能看出：不是按颜色、不是按位置顺序，而是要同时看块级形状和标记
#   - 输出中的红点统一放在标记块左上角，使“先旋转再缩放”与“整张图直接旋转”在视觉上有稳定差异，不会退化成看不见的实现细节

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


MARKER_COLOR = 2
BG_COLOR = 0
K_CHOICES = (2, 3)

OUTPUT_MARKER_ANCHOR = "tl"
ROTATION_ANCHOR_CODES = {
    0: "tl",
    1: "br",
    2: "bl",
    3: "tr",
}

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
SHAPE_RETRIES = 70
GENERATE_MAX_RETRIES = 500


def _anchor_offset(k: int, code: str) -> tuple[int, int]:
    if code == "tl":
        return 0, 0
    if code == "tr":
        return 0, k - 1
    if code == "bl":
        return k - 1, 0
    if code == "br":
        return k - 1, k - 1
    raise ValueError(f"unknown anchor code: {code}")


def _build_scaled_patch(
    mask: np.ndarray,
    marker: tuple[int, int],
    color: int,
    k: int,
    marker_code: str,
) -> tuple[np.ndarray, np.ndarray]:
    out_mask = np.kron(mask, np.ones((k, k), dtype=bool))
    patch = np.where(out_mask, color, BG_COLOR).astype(int)
    ar, ac = _anchor_offset(k, marker_code)
    mr, mc = marker
    patch[mr * k + ar, mc * k + ac] = MARKER_COLOR
    return patch, out_mask


def _rotate_marker(marker: tuple[int, int], shape: tuple[int, int], rot: int) -> tuple[int, int]:
    probe = np.zeros(shape, dtype=bool)
    probe[int(marker[0]), int(marker[1])] = True
    rr = np.argwhere(np.rot90(probe, rot))
    assert rr.shape[0] == 1
    return int(rr[0, 0]), int(rr[0, 1])


def _rotate_mask_and_marker(
    mask: np.ndarray, marker: tuple[int, int], rot: int
) -> tuple[np.ndarray, tuple[int, int]]:
    rot = int(rot) % 4
    return np.rot90(mask, rot), _rotate_marker(marker, mask.shape, rot)


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


def _decode_instruction_pixel(
    patch: np.ndarray, mask: np.ndarray, k: int
) -> tuple[tuple[int, int], int] | None:
    coords = np.argwhere((patch == MARKER_COLOR) & mask)
    if coords.shape[0] != 1:
        return None
    rr, cc = map(int, coords[0])
    marker_block = rr // k, cc // k
    corner = rr % k, cc % k
    code_to_rot = {_anchor_offset(k, code): rot for rot, code in ROTATION_ANCHOR_CODES.items()}
    rot = code_to_rot.get(corner)
    if rot is None:
        return None
    return marker_block, int(rot)


def _shape_key_rot(mask: np.ndarray) -> bytes:
    variants = []
    for rot in range(4):
        cur = np.rot90(mask, rot)
        variants.append(cur.tobytes() + bytes([cur.shape[0], cur.shape[1]]))
    return min(variants)


def _rotation_state_count(mask: np.ndarray, marker: tuple[int, int]) -> int:
    states = set()
    for rot in range(4):
        rm, rk = _rotate_mask_and_marker(mask, marker, rot)
        states.add((rm.shape, rm.tobytes(), rk))
    return len(states)


def _marker_candidates(mask: np.ndarray) -> list[tuple[int, int]]:
    out = []
    for r, c in np.argwhere(mask):
        marker = int(r), int(c)
        if _rotation_state_count(mask, marker) >= 2:
            out.append(marker)
    return out


def _choose_markers_for_duplicate(mask: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]] | None:
    cands = _marker_candidates(mask)
    random.shuffle(cands)
    for i in range(len(cands)):
        for j in range(i + 1, len(cands)):
            if cands[i] != cands[j]:
                return cands[i], cands[j]
    return None


def _piece_color_from_patch(patch: np.ndarray, mask: np.ndarray) -> int | None:
    vals = {int(patch[int(r), int(c)]) for r, c in np.argwhere(mask) if int(patch[int(r), int(c)]) != MARKER_COLOR}
    if len(vals) != 1:
        return None
    return int(next(iter(vals)))


def _match_slots(slots: list[dict], pieces: list[dict], k: int) -> list[tuple[dict, dict]] | None:
    decoded = []
    for pi, piece in enumerate(pieces):
        src_mask = _downsample_blocks(piece["mask"], k)
        if src_mask is None:
            return None
        decoded_marker = _decode_instruction_pixel(piece["patch"], piece["mask"], k)
        if decoded_marker is None:
            return None
        src_marker, rot = decoded_marker
        if not bool(src_mask[src_marker]):
            return None
        color = _piece_color_from_patch(piece["patch"], piece["mask"])
        if color is None:
            return None
        tgt_mask, tgt_marker = _rotate_mask_and_marker(src_mask, src_marker, rot)
        out_patch, out_mask = _build_scaled_patch(tgt_mask, tgt_marker, color, k, OUTPUT_MARKER_ANCHOR)
        decoded.append(
            {
                "pi": int(pi),
                "area": int(piece["area"]),
                "src_mask": src_mask,
                "src_marker": src_marker,
                "rot": int(rot),
                "tgt_mask": tgt_mask,
                "tgt_marker": tgt_marker,
                "out_patch": out_patch,
                "out_mask": out_mask,
            }
        )

    if len(decoded) != len(slots):
        return None

    candidates_by_slot: list[list[dict]] = []
    for slot in slots:
        want_area = int(slot["area"]) * k * k
        cur = []
        for cand in decoded:
            if int(cand["area"]) != want_area:
                continue
            if cand["tgt_mask"].shape != slot["mask"].shape:
                continue
            if not np.array_equal(cand["tgt_mask"], slot["mask"]):
                continue
            if cand["tgt_marker"] != slot["marker"]:
                continue
            cur.append(cand)
        if not cur:
            return None
        candidates_by_slot.append(cur)

    order = sorted(
        range(len(slots)),
        key=lambda i: (len(candidates_by_slot[i]), -int(slots[i]["area"]), int(slots[i]["r0"]), int(slots[i]["c0"])),
    )
    used: set[int] = set()
    answer: list[tuple[dict, dict] | None] = [None] * len(slots)

    def dfs(idx: int) -> bool:
        if idx == len(order):
            return True
        si = order[idx]
        for cand in candidates_by_slot[si]:
            pi = int(cand["pi"])
            if pi in used:
                continue
            used.add(pi)
            answer[si] = (slots[si], cand)
            if dfs(idx + 1):
                return True
            used.remove(pi)
            answer[si] = None
        return False

    return [x for x in answer if x is not None] if dfs(0) else None


def _recover_instance(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    assert comps, "expected at least one foreground component"
    frame_comp = max(comps, key=lambda d: int(d["area"]))
    fr0, fc0, fr1, fc1 = map(int, frame_comp["bbox"])

    frame_cols = [int(g[int(r), int(c)]) for r, c in frame_comp["coords"] if int(g[int(r), int(c)]) != MARKER_COLOR]
    assert frame_cols, "frame must contain non-marker cells"
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
        rr0, cc0, rr1, cc1 = map(int, bbox_coords(coords))
        sm = np.zeros((rr1 - rr0 + 1, cc1 - cc0 + 1), dtype=bool)
        sm[coords[:, 0] - rr0, coords[:, 1] - cc0] = True
        marker_coords = [(int(r), int(c)) for r, c in coords if bool(layout_markers[int(r), int(c)])]
        assert len(marker_coords) == 1, "each slot must have exactly one marker cell"
        mr, mc = marker_coords[0]
        slots.append(
            {
                "area": int(sm.sum()),
                "r0": int(rr0),
                "c0": int(cc0),
                "mask": sm,
                "marker": (int(mr - rr0), int(mc - cc0)),
            }
        )
    slots.sort(key=lambda s: (-int(s["area"]), int(s["r0"]), int(s["c0"])))
    assert slots, "frame must contain slots"

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
        if int(np.sum((patch == MARKER_COLOR) & pm)) != 1:
            continue
        if _piece_color_from_patch(patch, pm) is None:
            continue
        pieces.append({"area": int(pm.sum()), "mask": pm, "patch": patch})
    assert len(pieces) == len(slots), "need exactly one marked piece per slot"

    placements = None
    chosen_k = None
    for k in K_CHOICES:
        cand = _match_slots(slots, pieces, int(k))
        if cand is not None:
            placements = cand
            chosen_k = int(k)
            break
    assert placements is not None and chosen_k is not None, "no valid k / matching found"

    return {
        "frame_color": frame_color,
        "layout_shape": tuple(map(int, layout.shape)),
        "slots": slots,
        "pieces": pieces,
        "placements": placements,
        "k": chosen_k,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild a marked frame layout by decoding each piece's rotation instruction, then scaling it into place.

    Input structure:
    - bg is the most common color; the largest multicolor foreground component is the monochrome frame.
    - Each hole region inside the frame contains exactly one red marker cell.
    - Each outside piece is a k×k-block sprite with one red instruction pixel placed at a block corner.

    Rule:
    - Treat red cells inside the frame as part of the holes and record each slot's unique marker cell.
    - For each piece, downsample to block scale, read the red pixel's corner to decode a quarter-turn, and rotate the block-level shape plus marker by that amount.
    - After the rotation is resolved, rebuild the piece at scale k and place it into the slot whose shape and marker cell now match.

    Key insight:
    - The red pixel is not content to preserve verbatim from the input sprite; it is a symbolic instruction.
    - Correct reconstruction requires symbol decoding first and geometric scaling second, so directly rotating the already-scaled input sprite gives the wrong marker corner in the output.
    """
    info = _recover_instance(grid)
    layout_h, layout_w = map(int, info["layout_shape"])
    k = int(info["k"])
    out = np.full(((layout_h + 2) * k, (layout_w + 2) * k), int(info["frame_color"]), dtype=int)
    for slot, cand in info["placements"]:
        top = (int(slot["r0"]) + 1) * k
        left = (int(slot["c0"]) + 1) * k
        paste(out, cand["out_patch"], top, left, mask=cand["out_mask"], overwrite=True)
    return out


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
        return coords.astype(int)
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
        mask = _grow_polyomino(area)
        markers = _choose_markers_for_duplicate(mask)
        if markers is None:
            continue
        dup_mask = mask
        dup_markers = markers
        used_shape_keys.add(_shape_key_rot(mask))
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
    for tmpl in templates:
        coords = _place_mask(forbidden, tmpl["mask"])
        if coords is None:
            return None
        rr0, cc0 = map(int, coords.min(axis=0))
        marker = rr0 + int(tmpl["marker"][0]), cc0 + int(tmpl["marker"][1])
        slots.append(
            {
                "coords": coords,
                "area": int(coords.shape[0]),
                "mask": tmpl["mask"].copy(),
                "shape_key": _shape_key_rot(tmpl["mask"]),
                "marker": marker,
                "marker_rel": (int(tmpl["marker"][0]), int(tmpl["marker"][1])),
                "dup": bool(tmpl["dup"]),
            }
        )
        _forbid_coords(forbidden, coords)

    layout_mask = np.zeros((h, w), dtype=bool)
    layout_markers = np.zeros((h, w), dtype=bool)
    for slot in slots:
        layout_mask[slot["coords"][:, 0], slot["coords"][:, 1]] = True
        mr, mc = slot["marker"]
        layout_markers[int(mr), int(mc)] = True

    density = float(np.sum(layout_mask)) / float(h * w)
    lo, hi = HOLE_DENSITY_RANGE
    if not (lo <= density <= hi):
        return None
    if not (layout_mask[0].any() and layout_mask[-1].any() and layout_mask[:, 0].any() and layout_mask[:, -1].any()):
        return None
    return slots


def _make_piece(slot: dict, color: int, k: int, rot: int) -> dict:
    target_mask = slot["mask"]
    target_marker = slot["marker_rel"]
    src_mask, src_marker = _rotate_mask_and_marker(target_mask, target_marker, (-rot) % 4)
    out_patch, out_mask = _build_scaled_patch(target_mask, target_marker, color, k, OUTPUT_MARKER_ANCHOR)
    in_patch, in_mask = _build_scaled_patch(src_mask, src_marker, color, k, ROTATION_ANCHOR_CODES[int(rot)])
    return {
        "color": int(color),
        "slot": slot,
        "rotation": int(rot),
        "src_mask": src_mask,
        "src_marker": src_marker,
        "out_patch": out_patch,
        "out_mask": out_mask,
        "in_patch": in_patch,
        "in_mask": in_mask,
    }


def _build_expected_output(pieces: list[dict], layout_h: int, layout_w: int, k: int, frame_color: int) -> np.ndarray:
    out = np.full(((layout_h + 2) * k, (layout_w + 2) * k), frame_color, dtype=int)
    for piece in pieces:
        coords = piece["slot"]["coords"]
        top = (int(coords[:, 0].min()) + 1) * k
        left = (int(coords[:, 1].min()) + 1) * k
        paste(out, piece["out_patch"], top, left, mask=piece["out_mask"], overwrite=True)
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
    total_piece_area = sum(int(np.sum(piece["in_mask"])) for piece in pieces)
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

    for piece in sorted(pieces, key=lambda p: -int(np.sum(p["in_mask"]))):
        try:
            pr, pc = random_free_location(
                canvas,
                piece["in_patch"],
                transparent=BG_COLOR,
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
        if int(canvas[rr, cc]) != BG_COLOR:
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

    frame_area = (layout_h + 2) * (layout_w + 2) - sum(int(s["area"]) for s in slots)
    max_piece_area = max(int(s["area"]) * k * k for s in slots)
    if frame_area <= max_piece_area:
        return None

    rotations = [random.randint(0, 3) for _ in slots]
    if not any(rot != 0 for rot in rotations):
        rotations[random.randrange(len(rotations))] = random.randint(1, 3)
    if len(set(rotations)) < 2:
        rotations[0] = (rotations[0] + 1) % 4
    counts = Counter(rotations)
    if max(counts.values()) < 2:
        j = random.randrange(len(rotations))
        i = (j + 1) % len(rotations)
        rotations[i] = rotations[j]
    if not any(rot != 0 for rot in rotations):
        return None

    pieces = [_make_piece(slot=s, color=piece_colors[i], k=k, rot=rotations[i]) for i, s in enumerate(slots)]

    if not any(piece["rotation"] != 0 for piece in pieces):
        return None
    dup_slots = [s for s in slots if s["dup"]]
    if len(dup_slots) < 2:
        return None
    dup_signatures = {(s["shape_key"], s["marker_rel"]) for s in dup_slots}
    if len(dup_signatures) < 2:
        return None

    layout_mask = np.zeros((layout_h, layout_w), dtype=bool)
    layout_markers = np.zeros((layout_h, layout_w), dtype=bool)
    for slot in slots:
        layout_mask[slot["coords"][:, 0], slot["coords"][:, 1]] = True
        mr, mc = slot["marker"]
        layout_markers[int(mr), int(mc)] = True

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
