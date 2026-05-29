# Variant of 898e7135
# Dimension: D10 — 颜色分配策略 (meta)
# Change: 非单元精灵颜色互不相同，单元精灵共享一种颜色；框架用一种专用色，干扰项用另一种专用色；共需 slot_count + 2... → 精灵颜色编码洞的位置信息

from __future__ import annotations

"""
Design analysis for the D10 variant.

spec_variant:
  preserved_geometry:
    - 输入仍然有一个单色矩形框，框内是由背景色 0 形成的多个 4 连通洞槽。
    - 框外仍然散落与洞槽同形的 k*k 块放大精灵，允许旋转/翻转。
    - 输出仍然是把框内布局按 k 放大，再把对应精灵填回各洞位置。
  core_change:
    - 原题颜色只是标签：非单元精灵各不相同，单元精灵共享一种颜色。
    - 现在颜色本身编码洞的垂直位置。把所有槽按“在框内从上到下”的顺序排序，
      再按固定光谱分配颜色：顶部偏红，底部偏绿/蓝。
    - 因此颜色不再是任意标签，而是一个可见的位置信号。

  concrete_behavior:
    - 例：若 4 个槽从上到下的顺序为 A,B,C,D，则对应颜色固定为
      红, 橙, 黄, 绿。无论 A/B/C/D 的形状是否相似，框外对应 piece 的颜色
      都必须与其目标槽位的垂直顺序一致。
    - solver 的差异点：
      原逻辑：对每个槽，只看面积与形状，在全部 piece 中搜索匹配项。
      变体逻辑：先由槽位顺序推导该槽应有的颜色，再只在该颜色的 piece 中做
      形状匹配。颜色由“辅助线索”升级为“显式约束”。

  flow_trace:
    - 原题：顶部 L 形槽可能是蓝色，也可能是粉色；颜色和位置无关。
    - 变体：顶部 L 形槽必须拿到光谱最靠前的颜色；若另一个底部槽形状相近，
      颜色顺序能直接缩小搜索范围。
    - 差异落点：
      1. generator 不再自由洗牌 piece 颜色，而是按槽位行序分配。
      2. solver 不再把颜色视为可忽略属性，而是把它纳入候选过滤。

  complexity_calibration:
    - 仍保留 3-6 个槽、1-2 个 unit 槽、k∈{2,3}、二面体变换等原始复杂度。
    - 颜色只使用少量固定谱系，避免“颜色太多但无规律”。
    - 每个样例要求槽位在纵向上有可见跨度，避免所有槽都挤在同一横带导致
      颜色梯度肉眼不明显。

  implicit_assumption_review:
    - 原 solver 收集 pieces 时不读取颜色，只把 patch 原样粘回。
      这在原题无害，因为颜色不参与匹配；在变体中不够，需要显式提取 piece color。
    - 原 generator 允许 unit pieces 共色；新规则下这会破坏“颜色=位置”。
      因此 unit 与 non-unit 一样都改为按槽位顺序独立配色。
    - 原 placements 循环按面积/形状找第一个可用 candidate；
      若继续如此，solver 仍能解题，但完全没有用到 D10 的新信息。
      这里改为先检查 candidate color 是否等于该槽应有颜色。
    - 输出粘贴仍可沿用原实现，因为每个 piece 仍是单色且互不重叠；
      颜色变化不会引入覆盖冲突。

  placement_legality:
    - 框、pieces、单像素干扰项仍通过原始随机放置方式保证不重叠。
    - piece 间仍至少隔 1 个背景格；干扰项仍是孤立单像素，不会被 solver 误当成 piece。
    - frame/distractor 颜色始终从未被 row-spectrum 使用的剩余颜色中选取。

  visual_observability:
    - 解题者在 2-4 个样例中可以直接看到：越高的槽对应越“暖”的颜色，
      越低的槽对应越“冷”的颜色。
    - 颜色值不依赖数字大小，不用“颜色编号排序”这类不可见信息。
    - 即使形状本身已足够解题，颜色梯度依然是稳定、可重复观察到的附加规律。

  cleanliness:
    - 没有在同一区域叠加额外标记；只替换精灵配色策略，因此画面仍然清晰。
    - 框与干扰项使用光谱外颜色，避免与位置编码冲突。

  new_solver_assumptions:
    - 槽数量不超过 6，便于映射到固定光谱。
    - 每个非单像素 piece 是单色连通块；generator 保证这一点。
    - 槽按纵向排序后应有足够行跨度；generator 用显式约束保证这一点。
"""

import random
import sys
from collections import Counter

import numpy as np

SOLVER_DIR = "/home/lixinyu/snap/new/solvers"
if SOLVER_DIR not in sys.path:
    sys.path.append(SOLVER_DIR)

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


ROW_PALETTES = {
    1: [2],
    2: [2, 3],
    3: [2, 4, 3],
    4: [2, 7, 4, 3],
    5: [2, 7, 4, 3, 8],
    6: [2, 7, 4, 3, 8, 1],
}

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
MIN_ROW_SPAN = 2.0
MIN_ROW_BANDS = 3

GENERATE_MAX_RETRIES = 300


def _row_palette(slot_count: int) -> list[int]:
    assert int(slot_count) in ROW_PALETTES, "slot count must fit the row-color spectrum"
    return list(ROW_PALETTES[int(slot_count)])


def _slot_order_key(slot: dict) -> tuple[float, int, int]:
    coords = slot["coords"]
    return (
        float(coords[:, 0].mean()),
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
    )


def _slot_row_signal_ok(slots: list[dict]) -> bool:
    ordered = sorted(slots, key=_slot_order_key)
    anchors = [_slot_order_key(s)[0] for s in ordered]
    if anchors[-1] - anchors[0] < MIN_ROW_SPAN:
        return False
    coarse_bands = {int(round(a)) for a in anchors}
    return len(coarse_bands) >= min(MIN_ROW_BANDS, len(slots))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Inflate a frame's hole-layout and fill each hole with its matching piece.

    D10 variant:
    - Pieces are still matched by block-level shape under dihedral transforms.
    - Additionally, piece colors encode the target slot's top-to-bottom order,
      so solver first infers the expected row-spectrum and then uses color as
      a candidate filter before validating shape.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    assert comps, "input must contain foreground components"
    frame_comp = max(comps, key=lambda d: int(d["area"]))
    fr0, fc0, fr1, fc1 = map(int, frame_comp["bbox"])

    frame_cols = [int(g[int(r), int(c)]) for r, c in frame_comp["coords"]]
    frame_color = int(Counter(frame_cols).most_common(1)[0][0])

    sub = crop_bbox(g, (fr0, fc0, fr1, fc1))
    hole = sub == bg
    hole_box = bbox(hole)
    assert hole_box is not None, "frame must enclose at least one bg hole"
    lr0, lc0, lr1, lc1 = map(int, hole_box)
    layout = hole[lr0 : lr1 + 1, lc0 : lc1 + 1]

    slots = []
    for coords in components_mask(layout, connectivity=4):
        rr0, cc0, rr1, cc1 = bbox_coords(coords)
        rr0, cc0, rr1, cc1 = map(int, (rr0, cc0, rr1, cc1))
        sm = np.zeros((rr1 - rr0 + 1, cc1 - cc0 + 1), dtype=bool)
        sm[coords[:, 0] - rr0, coords[:, 1] - cc0] = True
        slots.append(
            {
                "coords": coords,
                "area": int(sm.sum()),
                "r0": rr0,
                "c0": cc0,
                "mask": sm,
            }
        )
    assert slots, "frame must contain at least one slot"

    ordered_slots = sorted(slots, key=_slot_order_key)
    row_colors = _row_palette(len(ordered_slots))
    for slot, color in zip(ordered_slots, row_colors):
        slot["expected_color"] = int(color)

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
        piece_colors = set(map(int, g[coords[:, 0], coords[:, 1]].tolist()))
        assert len(piece_colors) == 1, "each piece must be monochrome"
        pieces.append(
            {
                "area": int(pm.sum()),
                "mask": pm,
                "patch": patch,
                "color": int(next(iter(piece_colors))),
            }
        )
    assert pieces, "at least one non-frame piece must exist"

    k = None
    for p in pieces:
        h, w = map(int, p["mask"].shape)
        if h == w and int(p["mask"].sum()) == int(h * w):
            k = int(h) if k is None else min(int(k), int(h))
    k = int(k or 1)

    def downsample_blocks(mask: np.ndarray) -> np.ndarray | None:
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

    def dihedral_variants(patch: np.ndarray, mask: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        base = []
        for kk in range(4):
            p = np.rot90(patch, kk)
            m = np.rot90(mask, kk)
            base.append((p, m))
            base.append((np.fliplr(p), np.fliplr(m)))
        uniq = []
        seen = set()
        for p, m in base:
            key = m.tobytes()
            if key in seen:
                continue
            seen.add(key)
            uniq.append((p, m))
        return uniq

    candidates = []
    for pi, p in enumerate(pieces):
        for vp, vm in dihedral_variants(p["patch"], p["mask"]):
            um = downsample_blocks(vm)
            if um is None:
                continue
            candidates.append(
                {
                    "pi": int(pi),
                    "area": int(p["area"]),
                    "color": int(p["color"]),
                    "um": um,
                    "vp": vp,
                    "vm": vm,
                }
            )

    used_piece: set[int] = set()
    placements: list[tuple[dict, dict]] = []
    for slot in ordered_slots:
        need_area = int(slot["area"]) * k * k
        found = None
        for cand in candidates:
            if int(cand["pi"]) in used_piece:
                continue
            if int(cand["color"]) != int(slot["expected_color"]):
                continue
            if int(cand["area"]) != int(need_area):
                continue
            if cand["um"].shape != slot["mask"].shape:
                continue
            if np.array_equal(cand["um"], slot["mask"]):
                found = cand
                break
        assert found is not None, "every slot must match a same-color piece"
        used_piece.add(int(found["pi"]))
        placements.append((slot, found))

    out_h = (int(layout.shape[0]) + 2) * k
    out_w = (int(layout.shape[1]) + 2) * k
    out = np.full((out_h, out_w), frame_color, dtype=int)

    for slot, cand in placements:
        top = (int(slot["r0"]) + 1) * k
        left = (int(slot["c0"]) + 1) * k
        paste(out, cand["vp"], top, left, mask=cand["vm"], overwrite=True)

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
    H, W = forbidden.shape
    free_cells = np.argwhere(~forbidden)
    if free_cells.size == 0:
        return None
    free_cells = free_cells.tolist()
    random.shuffle(free_cells)
    for start in free_cells[:GROW_SEED_TRIES]:
        r0, c0 = int(start[0]), int(start[1])
        chosen = {(r0, c0)}
        frontier = {(r0 - 1, c0), (r0 + 1, c0), (r0, c0 - 1), (r0, c0 + 1)}
        while len(chosen) < target_size and frontier:
            cands = [
                p
                for p in frontier
                if 0 <= p[0] < H and 0 <= p[1] < W and not forbidden[p[0], p[1]] and p not in chosen
            ]
            if not cands:
                break
            nr, nc = random.choice(cands)
            chosen.add((nr, nc))
            frontier.discard((nr, nc))
            frontier.update({(nr - 1, nc), (nr + 1, nc), (nr, nc - 1), (nr, nc + 1)})
        if len(chosen) == target_size:
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
    forbidden = np.zeros((h, w), dtype=bool)
    slots: list[dict] = []
    used_shape_keys: set[bytes] = set()

    for _ in range(n_units):
        for _ in range(UNIT_SLOT_RETRIES):
            coords = _grow_slot(forbidden, 1)
            if coords is not None:
                slots.append({"coords": coords, "area": 1})
                _forbid_coords(forbidden, coords)
                break
        else:
            return None

    for _ in range(n_slots - n_units):
        size = random.choice(NON_UNIT_SIZE_CHOICES)
        for _ in range(NON_UNIT_SLOT_RETRIES):
            coords = _grow_slot(forbidden, size)
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
            slots.append({"coords": coords, "area": size, "mask": sm, "shape_key": key})
            _forbid_coords(forbidden, coords)
            break
        else:
            return None

    density = sum(s["area"] for s in slots) / (h * w)
    lo, hi = HOLE_DENSITY_RANGE
    if not (lo <= density <= hi):
        return None

    return slots


def _make_piece(slot: dict, color: int, k: int, apply_dihedral: bool) -> dict:
    if slot["area"] == 1:
        patch = np.full((k, k), color, dtype=int)
        mask = np.ones((k, k), dtype=bool)
        return {
            "color": int(color),
            "slot": slot,
            "out_patch": patch,
            "out_mask": mask,
            "in_patch": patch,
            "in_mask": mask,
        }

    out_mask = np.kron(slot["mask"], np.ones((k, k), dtype=bool))
    out_patch = np.where(out_mask, color, 0).astype(int)

    if apply_dihedral:
        rot = random.randint(0, 3)
        in_mask = np.rot90(out_mask, rot)
        in_patch = np.rot90(out_patch, rot)
        if random.random() < 0.5:
            in_mask = np.fliplr(in_mask)
            in_patch = np.fliplr(in_patch)
    else:
        in_mask, in_patch = out_mask, out_patch

    return {
        "color": int(color),
        "slot": slot,
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
    layout_h: int,
    layout_w: int,
    k: int,
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

    frame_img = np.full((frame_h, frame_w), frame_color, dtype=int)
    frame_img[1:-1, 1:-1] = np.where(layout_mask, 0, frame_color)
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
        if canvas[rr, cc] != 0:
            continue
        r_lo, r_hi = max(0, rr - 1), min(canvas_h, rr + 2)
        c_lo, c_hi = max(0, cc - 1), min(canvas_w, cc + 2)
        if np.any(canvas[r_lo:r_hi, c_lo:c_hi] != 0):
            continue
        canvas[rr, cc] = distractor_color
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

    slots = _gen_layout(layout_h, layout_w, n_slots, n_units)
    if slots is None:
        return None
    if not _slot_row_signal_ok(slots):
        return None

    layout_mask = np.zeros((layout_h, layout_w), dtype=bool)
    for slot in slots:
        layout_mask[slot["coords"][:, 0], slot["coords"][:, 1]] = True
    if not (layout_mask[0].any() and layout_mask[-1].any() and layout_mask[:, 0].any() and layout_mask[:, -1].any()):
        return None

    ordered_slots = sorted(slots, key=_slot_order_key)
    row_colors = _row_palette(len(ordered_slots))
    for slot, color in zip(ordered_slots, row_colors):
        slot["piece_color"] = int(color)

    remaining_colors = [c for c in range(1, 10) if c not in set(row_colors)]
    random.shuffle(remaining_colors)
    if len(remaining_colors) < 2:
        return None
    frame_color = int(remaining_colors.pop())
    distractor_color = int(remaining_colors.pop())

    pieces = [
        _make_piece(slot, int(slot["piece_color"]), k, apply_dihedral=(int(slot["area"]) > 1))
        for slot in slots
    ]

    expected_out = _build_expected_output(pieces, layout_h, layout_w, k, frame_color)
    canvas = _build_input_canvas(pieces, layout_mask, layout_h, layout_w, k, frame_color, distractor_color)
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
