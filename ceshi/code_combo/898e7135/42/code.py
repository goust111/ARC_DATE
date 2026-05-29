# Combo variant of 898e7135 (index 42)
# combo_spec: D1/0+D3/2+D15/0
# Name: 纯颜色配对系统

from __future__ import annotations

"""
Design analysis for the combo variant
=====================================

a) Concrete behavior and trace
spec_combo:
  preserved_core:
    - 仍然有一个 1 像素厚的矩形 frame，输出仍然是其按 k 放大的裁剪图。
    - 外部仍然有一组单色 piece；每个 piece 仍是某个 slot 的 k*k 块放大图。
    - k 仍由最小的实心 k*k 方块 piece 推断，因此 unit slot 仍必须存在。
  fused_change:
    - D3/2 把 frame 内部的洞从背景 0 改成“直接着色的 slot 区域”。
    - D1/0 把 slot 和 piece 的对应关系从形状匹配改成颜色匹配。
    - D15/0 允许非单元 slot 形状重复，因此形状不再能承担身份识别职责。
  pixel_trace:
    - 例子：frame 内有两个同形 L-slot，一个红色、一个蓝色；外部也有一个红 L piece 和一个蓝 L piece。
    - 原题会把 frame 内的洞看成背景空洞，再用 D4 形状匹配把 piece 填回去。
    - 融合后，solver 直接把 frame 内的彩色 slot 当作位置标签：红 piece -> 红 slot，蓝 piece -> 蓝 slot。
    - 因为两个 L-slot 形状完全相同，形状已无法区分身份；颜色成为唯一稳定线索。
  key_difference:
    - 原题的“几何匹配”是主链路。
    - 融合题的主链路是“颜色到位置”的映射，形状只剩下同色 piece 是否仍是合法 k 块放大图的完整性检查。

b) Complexity calibration
  - 每个 slot 都使用唯一颜色，避免颜色配对再次歧义。
  - 每个样例都强制至少出现一组重复的非单元形状，而不是只靠两个 unit slot 的平凡重复。
  - 外部 piece 不再随机旋转/翻转；否则 solver 仍需额外恢复姿态，形状又会重新进入主推理链。
  - 颜色数量受控：bg + frame + distractor + 最多 6 个 slot 色，总数仍在 ARC 调色板范围内。

c) Implicit-assumption audit of the original main()
  - `hole = sub == bg` 失效：融合后 frame 内已没有背景洞，slot 本体就是彩色区域。
  - `layout = hole[...]` 失效：solver 需要保留颜色布局，而不是布尔洞掩码。
  - `components_mask(layout)` 失效：slot 不再来自一个统一 bool mask，而是来自多个单色组件。
  - “遍历全部 piece 的 D4 变体并按形状配对”失效：重复形状被刻意允许，颜色才是身份键。
  - “frame 是单色组件”也不再成立：frame 是多色连通体，但基础墙色仍应是该大组件中的主色；生成器保证这一点。

d) Placement legality
  - slot 仍然 4 连通、彼此 8 分离，因此 colored layout 清晰且不重叠。
  - pieces 与 frame、彼此之间继续保持至少 1 个背景像素间隔。
  - distractor 仍是孤立单像素，且颜色不与 frame/slot/piece 共用。
  - 生成器额外要求 frame 矩形面积严格大于任一 piece 面积，保证 solver 选到的最大前景组件一定是 frame。

e) Visual observability
  - 解题者能直接看到：frame 内每个彩色 slot 在外部都有一个同色 sprite。
  - 同形异色 slot 会反复出现，迫使人放弃“形状决定对应关系”的旧规则。
  - 输出里 frame 背景回到统一墙色，而彩色 pieces 被放回这些同色位置，颜色映射关系在 2-3 个样例中即可归纳。

f) Visual cleanliness
  - slot 颜色直接填在洞区域本体上，不再叠加额外墙边标记，视觉层次更干净。
  - frame 内只存在两层语义：基础墙色 vs slot 色；frame 外只存在 piece 与孤立噪声。
  - 强制 slot 触及内布局四条边，solver 的 tight crop 与人眼看到的有效布局一致，不会凭空裁掉静态墙区。

g) Color-change inferability
  - 颜色不做任何隐藏映射或数值排序；同色即对应。
  - 颜色种类很少且有结构支撑：颜色同时出现在 frame 内 slot 与 frame 外 piece 上。
  - 因为重复形状被显式制造，人类能看出“颜色在 disambiguate shape”而不是仅作装饰。

h) New solver assumptions introduced here
  - frame 组件中基础墙色是出现次数最多的颜色；生成器通过布局面积与颜色分配保证。
  - 每个 slot 颜色在 frame 内恰好对应一个 4 连通单色区域；生成器用唯一颜色分配保证。
  - 每个 outside piece 颜色唯一，且与某个 slot 颜色一一对应；生成器直接构造该双射。
  - 每个 piece 仍是合法的 k 块放大图，且朝向已与 slot 对齐；生成器不再做 D4 扰动并用 oracle 校验。
"""

from collections import Counter
import random
import sys

import numpy as np

if "/home/lixinyu/snap/new/solvers" not in sys.path:
    sys.path.append("/home/lixinyu/snap/new/solvers")

from arc_utils import bbox, bbox_coords, crop_bbox, find_components, get_bg, paste, random_free_location


K_CHOICES = [2, 3]
LAYOUT_H_RANGE = (4, 12)
LAYOUT_W_RANGE = (4, 8)
N_SLOTS_RANGE = (3, 6)
N_UNITS_MAX = 2
NON_UNIT_SIZE_CHOICES = [2, 3, 4, 5, 6, 7]
DUPLICATE_SIZE_CHOICES = [2, 3, 4, 5]

HOLE_DENSITY_RANGE = (0.20, 0.50)
UNIT_SLOT_RETRIES = 20
NON_UNIT_SLOT_RETRIES = 30
DUPLICATE_PAIR_RETRIES = 40
COPY_PLACEMENT_TRIES = 80
GROW_SEED_TRIES = 8

CANVAS_DIM_RANGE = (15, 30)
AREA_BUFFER_MULT = 4
AREA_BUFFER_CONST = 70

DISTRACTOR_RANGE = (8, 13)
DISTRACTOR_RETRY_MULT = 10
MIN_DISTRACTORS = 8

GENERATE_MAX_RETRIES = 400


def _slot_from_coords(coords: np.ndarray, color: int) -> dict:
    rr0, cc0, rr1, cc1 = bbox_coords(coords)
    rr0, cc0, rr1, cc1 = map(int, (rr0, cc0, rr1, cc1))
    h = int(rr1 - rr0 + 1)
    w = int(cc1 - cc0 + 1)
    mask = np.zeros((h, w), dtype=bool)
    mask[coords[:, 0] - rr0, coords[:, 1] - cc0] = True
    return {
        "color": int(color),
        "coords": coords.astype(int),
        "area": int(coords.shape[0]),
        "r0": int(rr0),
        "c0": int(cc0),
        "mask": mask,
    }


def _canonical_shape_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    best = None
    for kk in range(4):
        rot = np.rot90(mask, kk)
        for cand in (rot, np.fliplr(rot)):
            key = (tuple(map(int, cand.shape)), cand.tobytes())
            if best is None or key < best:
                best = key
    assert best is not None
    return best


def _infer_k(pieces: list[dict]) -> int:
    k = None
    for piece in pieces:
        h, w = map(int, piece["mask"].shape)
        if h == w and int(piece["mask"].sum()) == int(h * w):
            k = int(h) if k is None else min(int(k), int(h))
    return int(k or 1)


def _downsample_blocks(mask: np.ndarray, k: int) -> np.ndarray | None:
    hh, ww = mask.shape
    if hh % k != 0 or ww % k != 0:
        return None
    out_h, out_w = int(hh // k), int(ww // k)
    out = np.zeros((out_h, out_w), dtype=bool)
    for r in range(out_h):
        for c in range(out_w):
            block = mask[int(r * k) : int((r + 1) * k), int(c * k) : int((c + 1) * k)]
            if bool(block.all()):
                out[r, c] = True
            elif bool(block.any()):
                return None
    return out


def _parse_scene(grid: np.ndarray) -> dict:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    assert comps, "input must contain foreground components"
    frame_comp = max(comps, key=lambda d: int(d["area"]))
    fr0, fc0, fr1, fc1 = map(int, frame_comp["bbox"])

    frame_cols = [int(g[int(r), int(c)]) for r, c in frame_comp["coords"]]
    frame_color = int(Counter(frame_cols).most_common(1)[0][0])

    sub = crop_bbox(g, (fr0, fc0, fr1, fc1))
    marker_mask = sub != int(frame_color)
    marker_box = bbox(marker_mask)
    assert marker_box is not None, "frame must contain colored slots"
    lr0, lc0, lr1, lc1 = map(int, marker_box)
    layout = sub[lr0 : lr1 + 1, lc0 : lc1 + 1]

    slot_comps = find_components(layout, bg=frame_color, connectivity=4, monochromatic=True, return_format="dict")
    assert slot_comps, "frame must contain at least one slot"
    slots_by_color: dict[int, dict] = {}
    for comp in slot_comps:
        color = int(comp["color"])
        slot = _slot_from_coords(comp["coords"], color)
        assert color not in slots_by_color, "slot colors must be unique"
        slots_by_color[color] = slot

    mono_comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    pieces_by_color: dict[int, dict] = {}
    for comp in mono_comps:
        coords = comp["coords"]
        if int(coords.shape[0]) <= 1:
            continue
        inside_frame = bool(
            np.all(
                (coords[:, 0] >= fr0)
                & (coords[:, 0] <= fr1)
                & (coords[:, 1] >= fc0)
                & (coords[:, 1] <= fc1)
            )
        )
        if inside_frame:
            continue
        color = int(comp["color"])
        pr0, pc0, pr1, pc1 = map(int, comp["bbox"])
        patch = crop_bbox(g, (pr0, pc0, pr1, pc1))
        mask = np.zeros_like(patch, dtype=bool)
        mask[coords[:, 0] - pr0, coords[:, 1] - pc0] = True
        assert color not in pieces_by_color, "piece colors must be unique"
        pieces_by_color[color] = {
            "color": int(color),
            "area": int(coords.shape[0]),
            "mask": mask,
            "patch": patch,
        }

    assert pieces_by_color, "at least one outside piece must exist"
    assert set(slots_by_color) == set(pieces_by_color), "slot colors and piece colors must match exactly"

    pieces = list(pieces_by_color.values())
    k = _infer_k(pieces)

    return {
        "grid": g,
        "bg": int(bg),
        "frame_bbox": (int(fr0), int(fc0), int(fr1), int(fc1)),
        "frame_color": int(frame_color),
        "layout": layout,
        "slots_by_color": slots_by_color,
        "pieces_by_color": pieces_by_color,
        "k": int(k),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Inflate a color-labeled frame layout and fill each slot with the outside sprite that shares its color.

    Input structure:
    - `0` is the most common background color.
    - The largest 4-connected foreground component is a rectangular frame whose base wall color is dominant.
    - Inside that frame, each slot is already a solid colored region rather than a background hole.
    - Outside the frame, each piece is a monochrome k×k-block-scaled sprite in one of those slot colors.

    Rule:
    - Recover the inner layout by cropping the non-wall colored regions inside the frame.
    - Extract one slot per color inside the frame and one outside piece of the same color.
    - Infer `k` from the smallest solid square piece.
    - Enlarge the frame by `k` and paste each same-color piece back into its slot position.

    Key insight:
    - This variant is no longer about cross-object shape matching: repeated slot shapes are allowed on purpose.
    - Color alone identifies which piece belongs to which slot, while geometry is reduced to a local validity check.
    """
    scene = _parse_scene(np.array(grid, dtype=int))
    frame_color = int(scene["frame_color"])
    layout = scene["layout"]
    slots_by_color = scene["slots_by_color"]
    pieces_by_color = scene["pieces_by_color"]
    k = int(scene["k"])

    out_h = (int(layout.shape[0]) + 2) * int(k)
    out_w = (int(layout.shape[1]) + 2) * int(k)
    out = np.full((out_h, out_w), int(frame_color), dtype=int)

    for color, slot in slots_by_color.items():
        piece = pieces_by_color[int(color)]
        need_area = int(slot["area"]) * int(k) * int(k)
        assert int(piece["area"]) == int(need_area), "piece area must equal scaled slot area"
        ds = _downsample_blocks(piece["mask"], int(k))
        assert ds is not None and np.array_equal(ds, slot["mask"]), "piece must already align with its same-color slot"
        top = (int(slot["r0"]) + 1) * int(k)
        left = (int(slot["c0"]) + 1) * int(k)
        paste(out, piece["patch"], int(top), int(left), mask=piece["mask"], overwrite=True)

    return out


def _grow_slot(forbidden: np.ndarray, target_size: int) -> np.ndarray | None:
    h, w = forbidden.shape
    free_cells = np.argwhere(~forbidden)
    if free_cells.size == 0:
        return None
    starts = [tuple(map(int, rc)) for rc in free_cells]
    random.shuffle(starts)
    for r0, c0 in starts[:GROW_SEED_TRIES]:
        chosen = {(int(r0), int(c0))}
        frontier = {(int(r0 - 1), int(c0)), (int(r0 + 1), int(c0)), (int(r0), int(c0 - 1)), (int(r0), int(c0 + 1))}
        while len(chosen) < int(target_size) and frontier:
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


def _mask_from_coords(coords: np.ndarray) -> np.ndarray:
    rr0, cc0, rr1, cc1 = bbox_coords(coords)
    rr0, cc0, rr1, cc1 = map(int, (rr0, cc0, rr1, cc1))
    mask = np.zeros((int(rr1 - rr0 + 1), int(cc1 - cc0 + 1)), dtype=bool)
    mask[coords[:, 0] - rr0, coords[:, 1] - cc0] = True
    return mask


def _place_copy_of_mask(forbidden: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    h, w = forbidden.shape
    mh, mw = map(int, mask.shape)
    rel = np.argwhere(mask)
    positions = [(r, c) for r in range(h - mh + 1) for c in range(w - mw + 1)]
    random.shuffle(positions)
    for top, left in positions[:COPY_PLACEMENT_TRIES]:
        coords = rel + np.array([int(top), int(left)], dtype=int)
        if bool(np.all(~forbidden[coords[:, 0], coords[:, 1]])):
            return coords.astype(int)
    return None


def _gen_layout(h: int, w: int, n_slots: int, n_units: int) -> list[dict] | None:
    forbidden = np.zeros((h, w), dtype=bool)
    slots: list[dict] = []

    for _ in range(n_units):
        for _ in range(UNIT_SLOT_RETRIES):
            coords = _grow_slot(forbidden, 1)
            if coords is None:
                continue
            slots.append(_slot_from_coords(coords, color=-1))
            _forbid_coords(forbidden, coords)
            break
        else:
            return None

    n_non_units = int(n_slots - n_units)
    if n_non_units < 2:
        return None

    dup_slots = None
    for _ in range(DUPLICATE_PAIR_RETRIES):
        size = random.choice(DUPLICATE_SIZE_CHOICES)
        coords1 = _grow_slot(forbidden, size)
        if coords1 is None:
            continue
        mask1 = _mask_from_coords(coords1)
        temp_forbidden = forbidden.copy()
        _forbid_coords(temp_forbidden, coords1)
        coords2 = _place_copy_of_mask(temp_forbidden, mask1)
        if coords2 is None:
            continue
        dup_slots = [
            _slot_from_coords(coords1, color=-1),
            _slot_from_coords(coords2, color=-1),
        ]
        forbidden = temp_forbidden
        _forbid_coords(forbidden, coords2)
        break
    if dup_slots is None:
        return None
    slots.extend(dup_slots)

    used_shape_keys = {_canonical_shape_key(dup_slots[0]["mask"])}
    for _ in range(n_non_units - 2):
        size = random.choice(NON_UNIT_SIZE_CHOICES)
        for _ in range(NON_UNIT_SLOT_RETRIES):
            coords = _grow_slot(forbidden, size)
            if coords is None:
                continue
            slot = _slot_from_coords(coords, color=-1)
            shape_key = _canonical_shape_key(slot["mask"])
            if shape_key in used_shape_keys:
                continue
            used_shape_keys.add(shape_key)
            slots.append(slot)
            _forbid_coords(forbidden, coords)
            break
        else:
            return None

    density = sum(int(slot["area"]) for slot in slots) / float(h * w)
    lo, hi = HOLE_DENSITY_RANGE
    if not (float(lo) <= density <= float(hi)):
        return None

    return slots


def _make_piece(slot: dict, color: int, k: int) -> dict:
    out_mask = np.kron(slot["mask"], np.ones((k, k), dtype=bool))
    out_patch = np.where(out_mask, int(color), 0).astype(int)
    return {
        "color": int(color),
        "slot": slot,
        "out_patch": out_patch,
        "out_mask": out_mask,
        "in_patch": out_patch.copy(),
        "in_mask": out_mask.copy(),
    }


def _build_layout_colors(slots: list[dict], layout_h: int, layout_w: int, frame_color: int) -> np.ndarray:
    layout = np.full((layout_h, layout_w), int(frame_color), dtype=int)
    for slot in slots:
        coords = slot["coords"]
        layout[coords[:, 0], coords[:, 1]] = int(slot["color"])
    return layout


def _build_expected_output(pieces: list[dict], layout_h: int, layout_w: int, k: int, frame_color: int) -> np.ndarray:
    out = np.full(((layout_h + 2) * k, (layout_w + 2) * k), int(frame_color), dtype=int)
    for piece in pieces:
        slot = piece["slot"]
        top = (int(slot["r0"]) + 1) * int(k)
        left = (int(slot["c0"]) + 1) * int(k)
        paste(out, piece["out_patch"], int(top), int(left), mask=piece["out_mask"], overwrite=True)
    return out


def _build_input_canvas(
    pieces: list[dict],
    layout_colors: np.ndarray,
    layout_h: int,
    layout_w: int,
    frame_color: int,
    distractor_color: int,
) -> np.ndarray | None:
    frame_h, frame_w = layout_h + 2, layout_w + 2
    total_piece_area = sum(int(piece["in_mask"].sum()) for piece in pieces)
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
    frame_img[1:-1, 1:-1] = layout_colors
    fr_max = canvas_h - frame_h - 1
    fc_max = canvas_w - frame_w - 1
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
        paste(canvas, piece["in_patch"], int(pr), int(pc), mask=piece["in_mask"], overwrite=True)

    n_distractors = random.randint(*DISTRACTOR_RANGE)
    placed = 0
    for _ in range(n_distractors * DISTRACTOR_RETRY_MULT):
        if placed >= n_distractors:
            break
        rr = random.randint(0, canvas_h - 1)
        cc = random.randint(0, canvas_w - 1)
        if int(canvas[rr, cc]) != 0:
            continue
        r_lo, r_hi = max(0, rr - 1), min(canvas_h, rr + 2)
        c_lo, c_hi = max(0, cc - 1), min(canvas_w, cc + 2)
        if bool(np.any(canvas[r_lo:r_hi, c_lo:c_hi] != 0)):
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
    n_units = random.randint(1, min(N_UNITS_MAX, n_slots - 2))

    palette = list(range(1, 10))
    random.shuffle(palette)
    frame_color = int(palette.pop())
    distractor_color = int(palette.pop())
    if len(palette) < n_slots:
        return None
    slot_colors = [int(palette.pop()) for _ in range(n_slots)]

    slots = _gen_layout(layout_h, layout_w, n_slots, n_units)
    if slots is None:
        return None
    random.shuffle(slots)
    for slot, color in zip(slots, slot_colors):
        slot["color"] = int(color)

    layout_colors = _build_layout_colors(slots, layout_h, layout_w, frame_color)
    marker_mask = layout_colors != int(frame_color)
    if not (
        bool(marker_mask[0].any())
        and bool(marker_mask[-1].any())
        and bool(marker_mask[:, 0].any())
        and bool(marker_mask[:, -1].any())
    ):
        return None

    pieces = [_make_piece(slot, int(slot["color"]), int(k)) for slot in slots]
    if (layout_h + 2) * (layout_w + 2) <= max(int(piece["in_mask"].sum()) for piece in pieces):
        return None

    expected_out = _build_expected_output(pieces, layout_h, layout_w, int(k), frame_color)
    canvas = _build_input_canvas(pieces, layout_colors, layout_h, layout_w, frame_color, distractor_color)
    if canvas is None:
        return None

    shape_counts = Counter(_canonical_shape_key(slot["mask"]) for slot in slots if int(slot["area"]) > 1)
    if max(shape_counts.values(), default=0) < 2:
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


def variant_metrics(grid: np.ndarray) -> dict:
    """
    Diagnostic helper for step-8 validation.
    The core non-degeneracy signal is duplicate non-unit shapes combined with unique slot colors.
    """
    scene = _parse_scene(np.array(grid, dtype=int))
    slots = list(scene["slots_by_color"].values())
    non_unit_shape_counts = Counter(_canonical_shape_key(slot["mask"]) for slot in slots if int(slot["area"]) > 1)
    return {
        "slot_count": int(len(slots)),
        "unique_slot_colors": int(len(scene["slots_by_color"])),
        "unit_slot_count": int(sum(int(slot["area"]) == 1 for slot in slots)),
        "duplicate_non_unit_groups": int(sum(int(v) >= 2 for v in non_unit_shape_counts.values())),
        "max_non_unit_duplicate_group": int(max(non_unit_shape_counts.values(), default=0)),
        "frame_color_cells_in_layout": int(np.sum(scene["layout"] == int(scene["frame_color"]))),
        "k": int(scene["k"]),
    }
