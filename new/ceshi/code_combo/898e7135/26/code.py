# Combo variant of 898e7135 (index 26)
# combo_spec: D1/0+D3/2
# Name: 双色对应匹配

from __future__ import annotations

"""
Design analysis for combo D1/0 + D3/2
=====================================

combo_summary:
  - 原题把 frame 内部的洞保留为背景色，再把外部 piece 通过块级形状 + D4 变换去匹配洞。
  - 本融合把“洞的身份”和“洞的位置”都显式写在颜色里：
    frame 内部每个 slot 直接填成专属颜色，外部 piece 也用同色。
  - 因为融合描述要求“推理完全转为颜色追踪”，外部 piece 不再随机旋转/翻转；
    solver 只需读取 `同色 slot -> 同色 piece` 的对应关系，然后按 slot 位置放大回填。

behavior_trace:
  small_example:
    - 假设内层布局里有两个完全相同的 2-cell 竖条 slot，一个红色、一个绿色，另有一个蓝色单元 slot。
    - 原题逻辑：
      1. 从 frame bbox 中找背景洞。
      2. 提取每个洞的布尔形状。
      3. 对每个外部 piece 枚举旋转/翻转，下采样后按形状配对。
    - 融合后逻辑：
      1. 从最大前景连通体中识别 frame。
      2. 取 frame 内部所有非 frame_color 的彩色区域，得到彩色 slot 布局。
      3. 读取每个彩色 slot 的颜色。
      4. 在 frame 外找到同色 piece，直接放回其放大后的位置。
    - 差异：
      原题的关键线索是几何；融合后的关键线索是颜色。尤其当两个 slot 形状完全相同时，
      不看颜色就无法知道哪个 piece 该回到哪个位置。

complexity_calibration:
  - slot 颜色始终一一对应，颜色数量最多 6，仍落在 ARC 常见颜色规模内。
  - 生成器强制每个样例至少存在一个“重复形状组”：
    要么有两个 unit slots，要么至少两块完全同形的非 unit slots。
    这样颜色不是可有可无的装饰，而是实际的配对依据。
  - 外部 piece 与内部 slot 保持同朝向，避免把任务又变成“颜色先配对，再额外猜旋转”。
    这与融合描述中的“推理完全转为颜色追踪”一致。

implicit_assumption_audit:
  original_code_breakpoints:
    - `hole = sub == bg` 失效：现在 frame 内部的 slot 不再是背景洞，而是彩色实体区域。
    - `components_mask(layout)` 失效：layout 不再是单一布尔洞图，而是多色区域图。
    - 原先跨 piece 做 shape search 的匹配循环失效：匹配规则已经改成颜色双射。
    - 原先 piece 可随机 D4 变换的前提失效：若继续旋转，会引入融合规格未给出的隐藏子规则。
  new_solver_assumptions:
    - 最大的多色 4 连通前景组件就是 frame；generator 通过单个矩形 frame 保证。
    - frame 组件的众数颜色是基础墙色；generator 保证 frame_color 不被外部 piece 复用，
      且墙体像素数多于任意单个 slot 颜色像素数。
    - 每个 slot 颜色在 frame 内恰好出现一次、在 frame 外恰好对应一个非单像素 piece；
      generator 直接构造这个颜色双射。
    - 每个外部 piece 都是其 slot 的 k x k 放大版本，且至少有一个 1-cell slot，
      所以 k 可由最小实心 k x k 方块恢复。

placement_legality:
  - layout 仍满足 slot 内部 4 连通、不同 slot 之间 8 分离。
  - frame 与外部 pieces、piece 与 piece 之间都留至少 1 个背景像素。
  - 干扰项仍是单独颜色的孤立单像素，不与 frame 或 piece 颜色复用。
  - 生成器要求彩色 slot 的并集触及内层布局四条边，确保 solver 的 tight crop 精确恢复布局。

visual_observability_and_cleanliness:
  - 输入中最醒目的结构是“框内彩色区域 + 框外同色精灵”，人眼能直接看到颜色对应关系。
  - frame 墙体只保留单一基础色，slot 本身纯色填充，不叠加边框标记，画面比“墙上再染色”更干净。
  - 输出中彩色区域被替换为放大后的同色 piece，且背景统一回到 frame_color，变化规律易对比。

behavior_metric:
  - `combo_metrics()` 统计：
      * `exact_duplicate_shape_groups`：完全同形 slot 组数量
      * `max_exact_duplicate_group`：最大完全同形组大小
      * `unique_slot_colors`：slot 颜色数
    若 `max_exact_duplicate_group >= 2`，则颜色配对在该样例中是显式必要信息。
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
PLACE_RETRIES = 80

CANVAS_DIM_RANGE = (15, 30)
AREA_BUFFER_MULT = 4
AREA_BUFFER_CONST = 70

DISTRACTOR_RANGE = (8, 13)
DISTRACTOR_RETRY_MULT = 10
MIN_DISTRACTORS = 8

GENERATE_MAX_RETRIES = 300


def _slot_from_coords(coords: np.ndarray, color: int) -> dict:
    rr0, cc0, rr1, cc1 = bbox_coords(coords)
    rr0, cc0, rr1, cc1 = map(int, (rr0, cc0, rr1, cc1))
    h = int(rr1 - rr0 + 1)
    w = int(cc1 - cc0 + 1)
    mask = np.zeros((h, w), dtype=bool)
    mask[coords[:, 0] - rr0, coords[:, 1] - cc0] = True
    return {
        "color": int(color),
        "coords": coords,
        "area": int(coords.shape[0]),
        "r0": int(rr0),
        "c0": int(cc0),
        "mask": mask,
    }


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
    out = np.zeros((int(hh // k), int(ww // k)), dtype=bool)
    for r in range(out.shape[0]):
        for c in range(out.shape[1]):
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
    assert comps, "input must contain a frame"
    frame_comp = max(comps, key=lambda d: int(d["area"]))
    fr0, fc0, fr1, fc1 = map(int, frame_comp["bbox"])

    frame_colors = [int(g[int(r), int(c)]) for r, c in frame_comp["coords"]]
    frame_color = int(Counter(frame_colors).most_common(1)[0][0])

    sub = crop_bbox(g, (fr0, fc0, fr1, fc1))
    marker_mask = sub != int(frame_color)
    marker_box = bbox(marker_mask)
    assert marker_box is not None, "frame must contain colored slots"
    lr0, lc0, lr1, lc1 = map(int, marker_box)
    layout = sub[lr0 : lr1 + 1, lc0 : lc1 + 1]

    slot_comps = find_components(layout, bg=frame_color, connectivity=4, monochromatic=True, return_format="dict")
    assert slot_comps, "frame must contain at least one colored slot"
    slots_by_color: dict[int, dict] = {}
    for comp in slot_comps:
        color = int(comp["color"])
        assert color != int(frame_color), "slot color must differ from frame color"
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

    assert pieces_by_color, "need at least one outside piece"
    assert set(slots_by_color) == set(pieces_by_color), "slot colors and piece colors must match"

    pieces = list(pieces_by_color.values())
    k = _infer_k(pieces)

    return {
        "grid": g,
        "bg": int(bg),
        "frame_color": int(frame_color),
        "layout": layout,
        "slots_by_color": slots_by_color,
        "pieces_by_color": pieces_by_color,
        "k": int(k),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Enlarge a color-marked frame layout and refill each slot with the outside piece of the same color.

    Input structure:
    - Background is the most common color.
    - The largest 4-connected multicolor foreground component is a rectangular frame with a 1-cell wall.
    - Inside that frame, each slot is already filled by its own non-frame marker color instead of background.
    - Outside the frame, there is exactly one monochrome k x k-block piece for each slot color; singleton noise may also appear.

    Rule:
    - Read the colored slot regions inside the frame as the destination layout.
    - Pair each slot with the outside piece that has the same color.
    - Infer k from the smallest solid square piece, then output the tight layout scaled by k with the same-color pieces pasted into their slot positions.

    Key insight:
    - The puzzle no longer asks for cross-piece shape matching: color already specifies which piece belongs to which slot,
      so the whole task reduces to tracking color correspondences from the frame interior to the scattered pieces.
    """
    scene = _parse_scene(np.array(grid, dtype=int))
    frame_color = int(scene["frame_color"])
    layout = scene["layout"]
    slots_by_color = scene["slots_by_color"]
    pieces_by_color = scene["pieces_by_color"]
    k = int(scene["k"])

    placements: list[tuple[dict, dict]] = []
    for color, slot in slots_by_color.items():
        piece = pieces_by_color[color]
        need_area = int(slot["area"]) * int(k) * int(k)
        assert int(piece["area"]) == int(need_area), "piece area must equal scaled slot area"
        down = _downsample_blocks(piece["mask"], k)
        assert down is not None and np.array_equal(down, slot["mask"]), "piece must be a clean scaled copy of its slot"
        placements.append((slot, piece))

    out_h = (int(layout.shape[0]) + 2) * int(k)
    out_w = (int(layout.shape[1]) + 2) * int(k)
    out = np.full((out_h, out_w), int(frame_color), dtype=int)

    for slot, piece in placements:
        top = (int(slot["r0"]) + 1) * int(k)
        left = (int(slot["c0"]) + 1) * int(k)
        paste(out, piece["patch"], int(top), int(left), mask=piece["mask"], overwrite=True)

    return out


def _mask_key(mask: np.ndarray) -> bytes:
    return mask.tobytes() + bytes([mask.shape[0], mask.shape[1]])


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


def _random_polyomino(target_size: int) -> np.ndarray:
    chosen = {(0, 0)}
    frontier = {(1, 0), (-1, 0), (0, 1), (0, -1)}
    while len(chosen) < int(target_size):
        nr, nc = random.choice(tuple(frontier))
        frontier.discard((nr, nc))
        chosen.add((nr, nc))
        frontier.update(
            {
                (int(nr - 1), int(nc)),
                (int(nr + 1), int(nc)),
                (int(nr), int(nc - 1)),
                (int(nr), int(nc + 1)),
            }
        )
        frontier -= chosen
    coords = np.array(sorted(chosen), dtype=int)
    r0, c0 = coords.min(axis=0)
    coords[:, 0] -= int(r0)
    coords[:, 1] -= int(c0)
    r1, c1 = coords.max(axis=0)
    mask = np.zeros((int(r1 + 1), int(c1 + 1)), dtype=bool)
    mask[coords[:, 0], coords[:, 1]] = True
    return mask


def _place_mask(mask: np.ndarray, layout_h: int, layout_w: int, forbidden: np.ndarray) -> np.ndarray | None:
    pts = np.argwhere(mask)
    mh, mw = map(int, mask.shape)
    positions = [(r0, c0) for r0 in range(layout_h - mh + 1) for c0 in range(layout_w - mw + 1)]
    random.shuffle(positions)
    for r0, c0 in positions[:PLACE_RETRIES]:
        coords = pts + np.array([[int(r0), int(c0)]], dtype=int)
        if bool(forbidden[coords[:, 0], coords[:, 1]].any()):
            continue
        return coords
    return None


def _gen_layout(layout_h: int, layout_w: int, n_slots: int, n_units: int) -> list[dict] | None:
    forbidden = np.zeros((layout_h, layout_w), dtype=bool)
    plans: list[np.ndarray] = [np.ones((1, 1), dtype=bool) for _ in range(n_units)]

    n_non_unit = int(n_slots - n_units)
    if n_non_unit >= 2:
        dup_mask = _random_polyomino(random.choice(DUPLICATE_SIZE_CHOICES))
        plans.append(dup_mask.copy())
        plans.append(dup_mask.copy())
        remaining = int(n_non_unit - 2)
    else:
        remaining = int(n_non_unit)

    for _ in range(remaining):
        plans.append(_random_polyomino(random.choice(NON_UNIT_SIZE_CHOICES)))

    random.shuffle(plans)
    plans.sort(key=lambda m: -int(m.sum()))

    slots: list[dict] = []
    for mask in plans:
        coords = _place_mask(mask, layout_h, layout_w, forbidden)
        if coords is None:
            return None
        slot = _slot_from_coords(coords, color=-1)
        if not np.array_equal(slot["mask"], mask):
            return None
        slots.append(slot)
        _forbid_coords(forbidden, coords)

    area = sum(int(slot["area"]) for slot in slots)
    density = float(area) / float(layout_h * layout_w)
    lo, hi = HOLE_DENSITY_RANGE
    if not (float(lo) <= density <= float(hi)):
        return None

    layout_mask = np.zeros((layout_h, layout_w), dtype=bool)
    for slot in slots:
        layout_mask[slot["coords"][:, 0], slot["coords"][:, 1]] = True
    if not (
        bool(layout_mask[0].any())
        and bool(layout_mask[-1].any())
        and bool(layout_mask[:, 0].any())
        and bool(layout_mask[:, -1].any())
    ):
        return None

    mask_counts = Counter(_mask_key(slot["mask"]) for slot in slots)
    if max(mask_counts.values()) < 2:
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
    frame_img[1:-1, 1:-1] = layout_colors
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
    n_units = 1 if n_slots == 3 else random.randint(1, min(N_UNITS_MAX, n_slots - 1))

    palette = list(range(1, 10))
    random.shuffle(palette)
    frame_color = palette.pop()
    distractor_color = palette.pop()
    if len(palette) < n_slots:
        return None
    slot_colors = [palette.pop() for _ in range(n_slots)]

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

    pieces = [_make_piece(slot, int(slot["color"]), k) for slot in slots]
    expected_out = _build_expected_output(pieces, layout_h, layout_w, k, frame_color)
    canvas = _build_input_canvas(pieces, layout_colors, layout_h, layout_w, frame_color, distractor_color)
    if canvas is None:
        return None

    try:
        produced = main(canvas)
    except Exception:
        return None
    if not np.array_equal(produced, expected_out):
        return None

    metrics = combo_metrics(canvas)
    if int(metrics["max_exact_duplicate_group"]) < 2:
        return None

    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_MAX_RETRIES):
        res = _gen_attempt()
        if res is not None:
            return res
    raise RuntimeError(f"generate_input: failed to generate after {GENERATE_MAX_RETRIES} attempts")


def combo_metrics(grid: np.ndarray) -> dict:
    scene = _parse_scene(np.array(grid, dtype=int))
    slots = list(scene["slots_by_color"].values())
    exact_counts = Counter(_mask_key(slot["mask"]) for slot in slots)
    return {
        "slot_count": int(len(slots)),
        "unit_slot_count": int(sum(int(slot["area"]) == 1 for slot in slots)),
        "unique_slot_colors": int(len(scene["slots_by_color"])),
        "exact_duplicate_shape_groups": int(sum(int(v) >= 2 for v in exact_counts.values())),
        "max_exact_duplicate_group": int(max(exact_counts.values())),
        "k": int(scene["k"]),
    }
