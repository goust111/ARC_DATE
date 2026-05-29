# Combo variant of 898e7135 (index 28)
# combo_spec: D1/4+D3/1
# Name: 跨框架面积排序

from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, find_components, get_bg, paste

"""
Design analysis for the combo variant
=====================================

spec_delta:
  preserved_core:
    - 背景仍为 0，外部 piece 仍由 k x k 实心块拼成，frame 内的 hole 仍是 4-连通区域。
    - 输出仍然是把 frame 的局部布局按 k 放大，再把输入里看到的 piece 朝向直接贴回去。
    - distractor 仍是孤立单像素，不参与任何匹配。
  fused_changes:
    - D1/4: 不再做块级形状匹配；所有 slot 与所有 piece 统一按面积排序后配对。
    - D3/1: 画布上出现 2-3 个彼此分离的 frame；slot 列表必须跨 frame 合并后再排序。

concrete_trace:
  - 例子使用两个 frame，从左到右记为 A、B。
  - A 内有 area=6、area=3、area=1 三个洞；B 内有 area=5、area=3 两个洞。
  - 原题思路会在 A 内部找回 A 的三块、在 B 内部找回 B 的两块，并且两个 area=3 的块还会靠形状区分。
  - 现在先把所有洞合并成全局顺序: 6(A), 5(B), 3(A), 3(B), 1(A)。
  - 输入里两个 area=3 piece 被故意反向摆放，所以全局 piece 顺序变成: 6(A), 5(B), 3(B), 3(A), 1(A)。
  - 结果是 B 的三格 piece 会被贴进 A 的三格洞，A 的三格 piece 会被贴进 B 的三格洞；交换发生在跨 frame 的位置上，
    这正是“多个 frame + 全局面积排序”同时起作用的信号。

complexity_calibration:
  - 不能让所有面积都互异，否则面积排序和原题形状匹配会退化为同一结果。
  - 因此固定制造一组跨 frame 的重复面积 area=3，并要求它们的 D4 形状不同，这样交换后视觉上明显。
  - 为了避免输出出现一个空的第四象限，本变体把 2-3 个 frame 都排成单行；这让“多 frame”清楚可见，同时保持画面整洁。
  - k 固定为 2。多 frame 已经显著增加输出宽度，继续混入 k=3 会把版面放大过头，削弱规则可读性。

implicit_assumption_audit:
  - 原代码 `frame_comp = max(comps, key=area)`:
    多 frame 后不成立，solver 必须显式识别所有单色矩形 frame，而不是只取最大连通块。
  - 原代码只 tight-crop 一个 frame 的 hole 布局:
    现在每个 frame 都要各自读取完整的矩形内部 `1:-1,1:-1` 作为 layout，并把 slot 记录到全局列表里。
  - 原代码用 D4 形状匹配 candidate:
    本变体删除这条几何匹配链，只保留面积与可见位置排序。
  - 新增假设 `piece` 的排序键来自输入 bbox 左上角:
    generator 通过按目标顺序单调扫描放置 piece 保证这一点。
  - 新增假设 frame 都在同一水平带上:
    generator 固定 frame 单行放置；solver 也按这一结构把放大后的 frame 横向排开。
  - 新增假设跨 frame 的重复面积确实会交换:
    generator 强制把两个 area=3 piece 的输入顺序反转，并在诊断函数里统计跨 frame 交换数。

placement_legality:
  - frame 统一放在 30x30 画布下方的一排，彼此用 1 个背景格隔开。
  - piece 用真实 mask 做 8 邻域间隔检测，保证不与 frame、其它 piece、distractor 接触。
  - 输出中 paste 使用 clip=True；面积排序不看形状，因此 piece 可能覆盖墙体或伸出目标洞的紧框范围。

visual_observability:
  - 2-3 个 frame 并列出现，解题者肉眼能看到“候选洞分散在不同 frame 中”。
  - 两个 area=3 slot 分属不同 frame 且形状不同，交换后不只是换颜色，而是换到另一块 frame 的洞位上。
  - 排序只使用面积和空间位置，不用颜色编号等人类不可见因素。

cleanliness_and_inferability:
  - 颜色控制为: 一个 shared frame 色、一个 distractor 色、每个 piece 独立颜色。
  - 版面中只有 frame、piece、单像素噪声三层对象，没有额外标记。
  - 关键变化可用量化指标验证: `cross_frame_swaps > 0` 表示至少有一个 piece 被分配到不同来源的 frame，
    说明样本没有退化回“逐 frame 局部匹配”。
"""


K_CHOICES = [2]
N_FRAMES_CHOICES = [2, 3]
LAYOUT_H = 6
LAYOUT_W = 5
FRAME_GAP = 1
CANVAS_SIZE = 30
OUTPUT_GAP = 1
FRAME_SLOT_TEMPLATES = {
    2: [[6, 3, 1], [5, 3]],
    3: [[6, 3], [5, 3], [5, 2, 1]],
}
DUP_AREA = 3

HOLE_DENSITY_RANGE = (0.25, 0.50)
FRAME_LAYOUT_RETRIES = 80
GROW_RETRIES = 40
GROW_SEED_TRIES = 12

DISTRACTOR_RANGE = (6, 10)
DISTRACTOR_RETRY_MULT = 10
MIN_DISTRACTORS = 6

GENERATE_MAX_RETRIES = 400


def _slot_sort_key(slot: dict) -> tuple[int, int, int, int]:
    return (-int(slot["area"]), int(slot["frame_idx"]), int(slot["r0"]), int(slot["c0"]))


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
    free_cells = free_cells.tolist()
    random.shuffle(free_cells)
    for start in free_cells[:GROW_SEED_TRIES]:
        r0, c0 = map(int, start)
        chosen = {(r0, c0)}
        frontier = {(r0 - 1, c0), (r0 + 1, c0), (r0, c0 - 1), (r0, c0 + 1)}
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


def _mask_from_coords(coords: np.ndarray) -> tuple[int, int, np.ndarray]:
    r0, c0, r1, c1 = map(int, bbox_coords(coords))
    mask = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return int(r0), int(c0), mask


def _downsample_blocks(mask: np.ndarray, k: int) -> np.ndarray | None:
    hh, ww = mask.shape
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


def _is_frame_component(grid: np.ndarray, bg: int, comp: dict) -> bool:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 3 or w < 3:
        return False
    patch = crop_bbox(grid, (r0, c0, r1, c1))
    color = int(comp["color"])
    if not np.all((patch == int(color)) | (patch == int(bg))):
        return False
    mask = patch == int(color)
    if not (mask[0, :].all() and mask[-1, :].all() and mask[:, 0].all() and mask[:, -1].all()):
        return False
    if not bool(np.any(patch[1:-1, 1:-1] == int(bg))):
        return False
    return True


def _infer_k(pieces: list[dict]) -> int:
    k = None
    for piece in pieces:
        h, w = map(int, piece["mask"].shape)
        if h == w and int(piece["mask"].sum()) == int(h * w):
            k = int(h) if k is None else min(int(k), int(h))
    assert k is not None, "need a unit-slot square piece to infer k"
    return int(k)


def _extract_problem(grid: np.ndarray) -> tuple[np.ndarray, int, list[dict], list[dict], int]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")

    frame_entries: list[tuple[int, dict]] = []
    for idx, comp in enumerate(comps):
        if _is_frame_component(g, bg, comp):
            frame_entries.append((int(idx), comp))
    assert 2 <= len(frame_entries) <= 3, "input must contain 2-3 independent frames"

    frame_entries.sort(key=lambda item: (int(item[1]["bbox"][0]), int(item[1]["bbox"][1])))
    frame_lookup = {comp_idx for comp_idx, _comp in frame_entries}

    frames: list[dict] = []
    for frame_idx, (_comp_idx, comp) in enumerate(frame_entries):
        fr0, fc0, fr1, fc1 = map(int, comp["bbox"])
        patch = crop_bbox(g, (fr0, fc0, fr1, fc1))
        layout = patch[1:-1, 1:-1] == int(bg)
        assert bool(layout.any()), "frame must enclose at least one hole"
        slots = []
        for coords in components_mask(layout, connectivity=4):
            r0, c0, mask = _mask_from_coords(coords)
            slots.append(
                {
                    "frame_idx": int(frame_idx),
                    "area": int(mask.sum()),
                    "r0": int(r0),
                    "c0": int(c0),
                    "mask": mask,
                    "shape_key": _canonical_shape_key(mask),
                }
            )
        assert slots, "each frame must contain at least one slot"
        frames.append(
            {
                "frame_idx": int(frame_idx),
                "r0": int(fr0),
                "c0": int(fc0),
                "h": int(patch.shape[0]),
                "w": int(patch.shape[1]),
                "color": int(comp["color"]),
                "layout": layout,
                "slots": slots,
            }
        )

    pieces: list[dict] = []
    for idx, comp in enumerate(comps):
        if int(idx) in frame_lookup:
            continue
        coords = comp["coords"]
        if coords.size == 0 or int(coords.shape[0]) <= 1:
            continue
        pr0, pc0, pr1, pc1 = map(int, comp["bbox"])
        patch = crop_bbox(g, (pr0, pc0, pr1, pc1))
        mask = patch == int(comp["color"])
        pieces.append(
            {
                "color": int(comp["color"]),
                "area": int(mask.sum()),
                "r0": int(pr0),
                "c0": int(pc0),
                "patch": patch,
                "mask": mask,
            }
        )

    assert pieces, "input must contain non-frame pieces"
    k = _infer_k(pieces)

    total_slots = sum(len(frame["slots"]) for frame in frames)
    assert len(pieces) == int(total_slots), "piece count must equal total slot count"

    need_areas = sorted(int(slot["area"]) * int(k) * int(k) for frame in frames for slot in frame["slots"])
    have_areas = sorted(int(piece["area"]) for piece in pieces)
    assert need_areas == have_areas, "piece areas must match slot areas globally"

    return g, int(bg), frames, pieces, int(k)


def _assign_output_positions(frames: list[dict], k: int) -> tuple[int, int]:
    left = 0
    max_h = 0
    for frame in frames:
        frame["out_r"] = 0
        frame["out_c"] = int(left)
        left += int(frame["w"]) * int(k) + OUTPUT_GAP
        max_h = max(int(max_h), int(frame["h"]) * int(k))
    out_w = int(left - OUTPUT_GAP)
    out_h = int(max_h)
    return int(out_h), int(out_w)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Enlarge several independent frames and fill all of their holes by one shared size ordering.

    Input structure:
    - bg is the most common color. The non-bg picture contains 2-3 separate monochrome rectangular frames in one row.
    - Each frame encloses bg holes; outside the frames sit one-color piece sprites built from solid k x k blocks, plus singleton distractors.

    Rule:
    - Extract every frame, recover each frame's hole components, and merge all slots into a single global list.
    - Sort slots by area descending with visible left-to-right / top-to-bottom tie breaks; sort pieces by pixel area descending with input bbox position tie breaks.
    - Paste the i-th sorted piece into the i-th sorted slot at that slot's scaled anchor, then output the scaled row of frames.

    Key insight:
    - The crucial change is not just "match by area" and not just "there are multiple frames"; the solver must first pool holes across frames, then do one global ranking.
      Same-area pieces can therefore jump from one frame to another, so shape recovery is intentionally irrelevant.
    """
    _g, _bg, frames, pieces, k = _extract_problem(grid)
    frames.sort(key=lambda frame: (int(frame["r0"]), int(frame["c0"])))
    out_h, out_w = _assign_output_positions(frames, k)
    common_frame_color = int(Counter(int(frame["color"]) for frame in frames).most_common(1)[0][0])
    out = np.full((out_h, out_w), int(common_frame_color), dtype=int)

    slots = [slot for frame in frames for slot in frame["slots"]]
    slots.sort(key=_slot_sort_key)
    pieces.sort(key=_piece_sort_key)

    for slot, piece in zip(slots, pieces):
        frame = frames[int(slot["frame_idx"])]
        top = int(frame["out_r"]) + (int(slot["r0"]) + 1) * int(k)
        left = int(frame["out_c"]) + (int(slot["c0"]) + 1) * int(k)
        paste(out, piece["patch"], int(top), int(left), mask=piece["mask"], overwrite=True, clip=True)

    return out


def _gen_frame_layout(
    slot_sizes: list[int],
    forbidden_shape_keys: set[bytes],
) -> list[dict] | None:
    forbidden = np.zeros((LAYOUT_H, LAYOUT_W), dtype=bool)
    slots: list[dict] = []
    local_keys: set[bytes] = set()

    for size in sorted(slot_sizes, reverse=True):
        for _ in range(GROW_RETRIES):
            coords = _grow_slot(forbidden, int(size))
            if coords is None:
                continue
            r0, c0, mask = _mask_from_coords(coords)
            shape_key = _canonical_shape_key(mask)
            if int(size) > 1 and (shape_key in forbidden_shape_keys or shape_key in local_keys):
                continue
            slots.append(
                {
                    "coords": coords,
                    "area": int(size),
                    "r0": int(r0),
                    "c0": int(c0),
                    "mask": mask,
                    "shape_key": shape_key,
                }
            )
            if int(size) > 1:
                local_keys.add(shape_key)
            _forbid_coords(forbidden, coords)
            break
        else:
            return None

    layout_mask = np.zeros((LAYOUT_H, LAYOUT_W), dtype=bool)
    for slot in slots:
        layout_mask[slot["coords"][:, 0], slot["coords"][:, 1]] = True

    density = float(layout_mask.sum()) / float(LAYOUT_H * LAYOUT_W)
    lo, hi = HOLE_DENSITY_RANGE
    if not (float(lo) <= density <= float(hi)):
        return None

    return slots


def _build_frame_image(layout_slots: list[dict], frame_color: int) -> np.ndarray:
    layout_mask = np.zeros((LAYOUT_H, LAYOUT_W), dtype=bool)
    for slot in layout_slots:
        layout_mask[slot["coords"][:, 0], slot["coords"][:, 1]] = True
    frame = np.full((LAYOUT_H + 2, LAYOUT_W + 2), int(frame_color), dtype=int)
    frame[1:-1, 1:-1] = np.where(layout_mask, 0, int(frame_color))
    return frame


def _make_piece(slot: dict, color: int, k: int, frame_idx: int) -> dict:
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
        "source_frame": int(frame_idx),
        "slot_area": int(slot["area"]),
        "slot_shape_key": slot["shape_key"],
        "patch": patch,
        "mask": mask,
        "area": int(mask.sum()),
    }


def _desired_piece_order(pieces: list[dict], k: int) -> list[dict]:
    groups: dict[int, list[dict]] = defaultdict(list)
    for piece in pieces:
        groups[int(piece["area"])].append(piece)

    ordered: list[dict] = []
    dup_pixel_area = int(DUP_AREA) * int(k) * int(k)
    for area in sorted(groups.keys(), reverse=True):
        group = sorted(
            groups[int(area)],
            key=lambda piece: (int(piece["source_frame"]), int(piece["slot_area"]), int(piece["color"])),
        )
        if int(area) == int(dup_pixel_area) and len(group) >= 2:
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


def _frame_positions(n_frames: int) -> list[tuple[int, int]]:
    frame_h = LAYOUT_H + 2
    frame_w = LAYOUT_W + 2
    total_w = int(n_frames) * int(frame_w) + int(n_frames - 1) * int(FRAME_GAP)
    top = CANVAS_SIZE - frame_h - 1
    left = CANVAS_SIZE - total_w - 1
    assert top >= 1 and left >= 1
    return [(int(top), int(left + idx * (frame_w + FRAME_GAP))) for idx in range(int(n_frames))]


def _build_input_canvas(
    frames: list[dict],
    pieces_in_order: list[dict],
    frame_color: int,
    distractor_color: int,
) -> np.ndarray | None:
    canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=int)

    for frame, (fr, fc) in zip(frames, _frame_positions(len(frames))):
        frame["r0"] = int(fr)
        frame["c0"] = int(fc)
        frame_img = _build_frame_image(frame["slots"], frame_color)
        paste(canvas, frame_img, int(fr), int(fc))

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
            piece["r0"] = int(rr)
            piece["c0"] = int(cc)
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


def _build_expected_output(frames: list[dict], pieces: list[dict], k: int, frame_color: int) -> np.ndarray:
    frames_sorted = sorted(frames, key=lambda frame: (int(frame["r0"]), int(frame["c0"])))
    out_frames = []
    left = 0
    out_h = 0
    for frame in frames_sorted:
        out_frames.append((frame, int(left)))
        left += (LAYOUT_W + 2) * int(k) + OUTPUT_GAP
        out_h = max(int(out_h), (LAYOUT_H + 2) * int(k))
    out_w = int(left - OUTPUT_GAP)
    out = np.full((out_h, out_w), int(frame_color), dtype=int)

    slot_order = []
    for frame_idx, frame in enumerate(frames_sorted):
        for slot in frame["slots"]:
            slot_order.append(
                {
                    "frame_idx": int(frame_idx),
                    "r0": int(slot["r0"]),
                    "c0": int(slot["c0"]),
                    "area": int(slot["area"]),
                }
            )
    slot_order.sort(key=_slot_sort_key)
    piece_order = sorted(pieces, key=_piece_sort_key)

    frame_left_by_id = {int(idx): int(left_pos) for idx, (frame, left_pos) in enumerate(out_frames)}
    for slot, piece in zip(slot_order, piece_order):
        top = (int(slot["r0"]) + 1) * int(k)
        left_pos = int(frame_left_by_id[int(slot["frame_idx"])]) + (int(slot["c0"]) + 1) * int(k)
        paste(out, piece["patch"], int(top), int(left_pos), mask=piece["mask"], overwrite=True, clip=True)
    return out


def analyze_combo_behavior(grid: np.ndarray) -> dict:
    _g, _bg, frames, pieces, k = _extract_problem(grid)
    slots = [slot for frame in frames for slot in frame["slots"]]
    slots.sort(key=_slot_sort_key)
    pieces.sort(key=_piece_sort_key)

    source_frame_by_piece: dict[int, int] = {}
    source_key_to_frame = {}
    for frame in frames:
        for slot in frame["slots"]:
            key = (int(slot["area"]), slot["shape_key"])
            source_key_to_frame[key] = int(frame["frame_idx"])

    for idx, piece in enumerate(pieces):
        small = _downsample_blocks(piece["mask"], k)
        assert small is not None, "piece masks must align to k x k blocks"
        key = (int(piece["area"]) // (int(k) * int(k)), _canonical_shape_key(small))
        assert key in source_key_to_frame, "every generated piece must map back to a unique source slot"
        source_frame_by_piece[int(idx)] = int(source_key_to_frame[key])

    cross_frame_swaps = 0
    for idx, slot in enumerate(slots):
        if int(source_frame_by_piece[int(idx)]) != int(slot["frame_idx"]):
            cross_frame_swaps += 1

    frame_sequence = [int(slot["frame_idx"]) for slot in slots]
    frame_switches = sum(
        1 for i in range(1, len(frame_sequence)) if int(frame_sequence[i]) != int(frame_sequence[i - 1])
    )
    return {
        "n_frames": int(len(frames)),
        "k": int(k),
        "cross_frame_swaps": int(cross_frame_swaps),
        "frame_switches": int(frame_switches),
        "slot_areas": [int(slot["area"]) for slot in slots],
    }


def _gen_attempt() -> np.ndarray | None:
    k = random.choice(K_CHOICES)
    n_frames = random.choice(N_FRAMES_CHOICES)
    size_template = FRAME_SLOT_TEMPLATES[int(n_frames)]

    palette = list(range(1, 10))
    random.shuffle(palette)
    frame_color = int(palette.pop())
    distractor_color = int(palette.pop())
    n_pieces = sum(len(sizes) for sizes in size_template)
    if len(palette) < int(n_pieces):
        return None
    piece_colors = [int(c) for c in palette[:n_pieces]]

    frames: list[dict] = []
    used_shape_keys: set[bytes] = set()
    for frame_idx, slot_sizes in enumerate(size_template):
        slots = None
        for _ in range(FRAME_LAYOUT_RETRIES):
            slots = _gen_frame_layout(list(slot_sizes), used_shape_keys)
            if slots is None:
                continue
            break
        if slots is None:
            return None
        for slot in slots:
            if int(slot["area"]) > 1:
                used_shape_keys.add(slot["shape_key"])
        frames.append({"frame_idx": int(frame_idx), "slots": slots})

    pieces: list[dict] = []
    color_iter = iter(piece_colors)
    for frame in frames:
        for slot in frame["slots"]:
            pieces.append(_make_piece(slot, next(color_iter), k, int(frame["frame_idx"])))

    intended_piece_order = _desired_piece_order(pieces, k)
    canvas = _build_input_canvas(frames, intended_piece_order, frame_color, distractor_color)
    if canvas is None:
        return None

    actual_piece_order = sorted(pieces, key=_piece_sort_key)
    expected_out = _build_expected_output(frames, actual_piece_order, k, frame_color)

    try:
        produced = main(canvas)
    except Exception:
        return None
    if not np.array_equal(produced, expected_out):
        return None

    metrics = analyze_combo_behavior(canvas)
    if int(metrics["cross_frame_swaps"]) <= 0:
        return None
    if int(metrics["frame_switches"]) <= 0:
        return None

    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_MAX_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"generate_input: failed after {GENERATE_MAX_RETRIES} attempts")
