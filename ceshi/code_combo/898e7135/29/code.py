# Combo variant of 898e7135 (index 29)
# combo_spec: D10/1+D14/0
# Name: 图例辅助位置推断

from __future__ import annotations

"""
Design analysis for the combo variant.

spec_variant:
  preserved_core:
    - 输入仍然有一个大框架、框外散落的 k*k 块精灵、以及输出中把框架按 k 放大后回填精灵。
    - 精灵与槽位的几何关系不变：每个精灵仍是某个槽位经 k 倍块放大后的同形体，
      输入里允许旋转/翻转，输出里要恢复到槽位朝向。
    - 单像素干扰仍保留，但它们不会参与匹配。

  concrete_fusion:
    - 采用 D14 的画面结构：框内槽位不再是背景洞，而是用独特 marker color 填色；
      左上角有固定 2 列竖直图例，每行是 (slot_marker_color, piece_color)。
    - 同时采用 D10 的配色规则：piece_color 不是任意标签，而是按槽位在框内
      “从上到下”的顺序使用固定光谱。这里用小而稳定的可见谱系：
      3 槽时 [红, 黄, 绿]，4 槽时 [红, 橙, 黄, 绿]。
    - 为了让“图例”和“颜色顺序”同时可见，图例行本身也按槽位的纵向顺序排列，
      即最上面的槽对应第一行图例，最下面的槽对应最后一行图例。

  behavior_trace:
    - 原题：
      1. 在框内找背景洞。
      2. 只用面积 + 形状 + D4 变换去匹配外部 pieces。
      3. 颜色只负责显示，不参与推理。
    - 融合后：
      1. 先读左上角图例，得到 marker -> piece_color 对应。
      2. 在框内找彩色槽位区域，而不是背景洞。
      3. 按槽位在框中的纵向顺序推导该位置应出现的光谱颜色。
      4. 检查图例给出的 piece_color 与该纵向颜色一致，再只在同色 piece 中做形状匹配。
    - 小例子：
      假设框里从上到下有 4 个槽 A/B/C/D，marker 分别是蓝/粉/灰/青。
      左上角图例四行依次是 (蓝,红), (粉,橙), (灰,黄), (青,绿)。
      框外 pieces 的颜色也正好是红/橙/黄/绿，但形状顺序被打乱且有旋转。
      solver 不再把“红 piece”当成随便哪块，而是知道它只能对应最上层槽。
      同时因为 A/B 中可能有同形 unit 槽，图例里的 marker 对应仍然是显式指令。

  complexity_calibration:
    - 槽数量限制为 3-4。这样 4 个 row-spectrum 颜色 + 4 个 marker 颜色 + 1 个 frame 颜色
      刚好仍能装进 ARC 的 1..9 色板；干扰点复用已有颜色，避免色数爆炸。
    - 每个样例固定 2 个 unit 槽，制造“几何上重复”的局部歧义，让图例不是纯装饰。
    - 非 unit 槽仍要求 D4 意义下互异，保留原题的几何匹配主体。
    - 额外要求槽位在纵向上有足够跨度，避免所有槽挤在同一水平带导致颜色谱失去可观察性。

  implicit_assumption_review:
    - 原 solver 的 `hole = sub == bg` 已失效，因为槽位不再是背景；这里改成
      `slot_occ = sub != frame_color`，并把每个连通彩色块当作槽位。
    - 原 solver 提取 pieces 时忽略颜色。融合后颜色必须进入候选过滤，否则 D10 白做。
    - 原 solver 会把所有非 frame 的非单像素前景都视为 piece。加上图例后这会误收图例，
      所以需要先构造 `legend_mask` 把左上角图例组件排除。
    - 仅靠 D10 的固定 row-spectrum，理论上可以直接按行序猜 piece；但融合规格明确要求
      “先解读图例，再用颜色顺序辅助验证或直接确定匹配”。因此 solver 显式同时使用
      `marker -> piece_color` 和 `slot_row_order -> expected_color` 这两条约束。
    - `k` 仍通过最小实心方块 piece 推断。生成器固定保留 2 个 unit 槽，所以这个假设稳定。

  placement_legality:
    - 图例固定贴在左上角；框与 pieces 用 `random_free_location(..., padding=1)` 放置，
      因此不会与图例或彼此接触，也不会误连成同一组件。
    - 槽位继续保持 8 分离，保证框内不同 marker 区域是独立 4 连通组件。
    - 仍要求 layout_mask 触碰内布局四条边，这样 solver 对槽位做 tight crop 时能恢复完整布局。

  visual_observability:
    - 人类能直接看到三个稳定信号：
      1. 左上角有专门图例区，不属于框也不属于 piece。
      2. 框内槽位是 marker 色块，而不是原题那种黑洞。
      3. 框外 pieces 的颜色沿着例子反复呈现“越上越暖、越下越绿”的顺序。
    - 颜色规则不依赖数字大小，而依赖 ARC 中可直接看见的红/橙/黄/绿顺序。
    - 图例行按纵向顺序排列，使“颜色-位置映射”在单个输入中也能被看见，而不是只能跨样例猜。

  cleanliness_and_inference:
    - 图例只占左上角 2 列窄条，不与主体叠加，视觉上整洁。
    - marker 色与 piece 色严格取自不相交集合，避免用户误以为框内颜色可直接搬到输出。
    - 量化验证时应至少检查：
      1. 图例颜色对严格对应槽位纵向顺序；
      2. row-spectrum 始终被使用；
      3. 至少存在一对同形 unit 槽，证明仍有“需要图例/颜色消歧”的结构。
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
    3: [2, 4, 3],
    4: [2, 7, 4, 3],
}

K_CHOICES = [2, 3]
LAYOUT_H_RANGE = (4, 12)
LAYOUT_W_RANGE = (4, 8)
N_SLOTS_RANGE = (3, 4)
N_UNITS = 2
NON_UNIT_SIZE_CHOICES = [2, 3, 4, 5, 6, 7]

HOLE_DENSITY_RANGE = (0.20, 0.50)
UNIT_SLOT_RETRIES = 20
NON_UNIT_SLOT_RETRIES = 30
GROW_SEED_TRIES = 8

CANVAS_DIM_RANGE = (15, 30)
AREA_BUFFER_MULT = 4
AREA_BUFFER_CONST = 60

DISTRACTOR_RANGE = (8, 13)
DISTRACTOR_RETRY_MULT = 10
MIN_DISTRACTORS = 8

MIN_ROW_SPAN = 2.0
MIN_ROW_BANDS = 3
GENERATE_MAX_RETRIES = 300


def _row_palette(slot_count: int) -> list[int]:
    assert int(slot_count) in ROW_PALETTES, "slot count must fit the row spectrum"
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
    anchors = [_slot_order_key(slot)[0] for slot in ordered]
    if anchors[-1] - anchors[0] < MIN_ROW_SPAN:
        return False
    coarse_bands = {int(round(anchor)) for anchor in anchors}
    return len(coarse_bands) >= min(MIN_ROW_BANDS, len(slots))


def _read_corner_legend(g: np.ndarray, bg: int) -> tuple[list[tuple[int, int]], np.ndarray]:
    """Read the fixed 2-column legend from the top-left corner."""
    h, w = g.shape
    assert w >= 2, "grid too narrow for legend"
    rows: list[tuple[int, int]] = []
    r = 0
    while r < h and int(g[r, 0]) != int(bg) and int(g[r, 1]) != int(bg):
        rows.append((int(g[r, 0]), int(g[r, 1])))
        r += 1
    assert len(rows) in ROW_PALETTES, "legend length must determine the row palette"
    marker_colors = [marker for marker, _piece in rows]
    assert len(set(marker_colors)) == len(marker_colors), "legend markers must be unique"
    legend_mask = np.zeros_like(g, dtype=bool)
    legend_mask[: len(rows), :2] = True
    return rows, legend_mask


def _downsample_blocks(mask: np.ndarray, k: int) -> np.ndarray | None:
    hh, ww = mask.shape
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


def _dihedral_variants(patch: np.ndarray, mask: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    base = []
    for kk in range(4):
        rp = np.rot90(patch, int(kk))
        rm = np.rot90(mask, int(kk))
        base.append((rp, rm))
        base.append((np.fliplr(rp), np.fliplr(rm)))
    uniq = []
    seen = set()
    for vp, vm in base:
        key = (vm.shape, vm.tobytes())
        if key in seen:
            continue
        seen.add(key)
        uniq.append((vp, vm))
    return uniq


def _canonical_shape_key(mask: np.ndarray) -> bytes:
    variants = []
    for kk in range(4):
        rot = np.rot90(mask, int(kk))
        variants.append(rot.tobytes() + bytes([rot.shape[0], rot.shape[1]]))
        flip = np.fliplr(rot)
        variants.append(flip.tobytes() + bytes([flip.shape[0], flip.shape[1]]))
    return min(variants)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Inflate a marked frame layout and fill each slot with the legend-linked piece.

    Input structure:
    - bg is the most common color.
    - The top-left corner contains a 2-column legend of (slot-marker color, piece color) rows.
    - The largest foreground component is a frame whose interior contains frame-color walls
      plus colored slot regions, one marker color per slot.
    - Outside the frame sit monochrome pieces made of k×k blocks; their colors follow a
      top-to-bottom row spectrum and their shapes may be rotated or mirrored.

    Rule:
    - Read the legend, then recover the colored slot regions inside the frame.
    - Sort slots by vertical position to infer the expected spectrum color for each row band.
    - For each slot, require the legend's mapped piece color to agree with that row-order color,
      then match the uniquely colored piece by block-level shape under dihedral transforms.
    - Output the k-scaled frame crop with the matched pieces pasted into their slot positions.

    Key insight:
    - The legend tells which marker color corresponds to which piece color, while the piece
      colors themselves encode the slot's top-to-bottom order. Combining both signals narrows
      matching before geometry is used only for orientation.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    legend_rows, legend_mask = _read_corner_legend(g, bg)
    marker_to_piece = {int(marker): int(piece) for marker, piece in legend_rows}
    expected_spectrum = _row_palette(len(legend_rows))

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    assert comps, "input must contain foreground components"
    frame_comp = max(comps, key=lambda comp: int(comp["area"]))
    fr0, fc0, fr1, fc1 = map(int, frame_comp["bbox"])

    frame_colors = [int(g[int(r), int(c)]) for r, c in frame_comp["coords"]]
    frame_color = int(Counter(frame_colors).most_common(1)[0][0])

    sub = crop_bbox(g, (fr0, fc0, fr1, fc1))
    slot_occ = sub != int(frame_color)
    slot_box = bbox(slot_occ)
    assert slot_box is not None, "frame must enclose at least one colored slot"
    lr0, lc0, lr1, lc1 = map(int, slot_box)
    layout = sub[lr0 : lr1 + 1, lc0 : lc1 + 1]
    layout_mask = layout != int(frame_color)

    slots = []
    for coords in components_mask(layout_mask, connectivity=4):
        rr0, cc0, rr1, cc1 = bbox_coords(coords)
        rr0, cc0, rr1, cc1 = map(int, (rr0, cc0, rr1, cc1))
        sm = np.zeros((rr1 - rr0 + 1, cc1 - cc0 + 1), dtype=bool)
        sm[coords[:, 0] - rr0, coords[:, 1] - cc0] = True
        colors = {int(layout[int(r), int(c)]) for r, c in coords}
        assert len(colors) == 1, "each slot must be monochrome"
        marker = int(next(iter(colors)))
        assert marker in marker_to_piece, "every slot marker must appear in the legend"
        slots.append(
            {
                "coords": coords,
                "area": int(sm.sum()),
                "r0": int(rr0),
                "c0": int(cc0),
                "mask": sm,
                "marker": marker,
            }
        )

    ordered_slots = sorted(slots, key=_slot_order_key)
    assert len(ordered_slots) == len(legend_rows), "legend and slot counts must match"
    for slot, expected_color in zip(ordered_slots, expected_spectrum):
        legend_color = int(marker_to_piece[int(slot["marker"])])
        assert legend_color == int(expected_color), "legend must align with the row spectrum"
        slot["expected_color"] = int(expected_color)

    pieces = []
    for comp in comps:
        if comp is frame_comp:
            continue
        coords = comp["coords"]
        if coords.size == 0 or int(coords.shape[0]) <= 1:
            continue
        if bool(np.all(legend_mask[coords[:, 0], coords[:, 1]])):
            continue
        pr0, pc0, pr1, pc1 = map(int, comp["bbox"])
        patch = crop_bbox(g, (pr0, pc0, pr1, pc1))
        pm = np.zeros(patch.shape, dtype=bool)
        pm[coords[:, 0] - pr0, coords[:, 1] - pc0] = True
        colors = {int(g[int(r), int(c)]) for r, c in coords}
        assert len(colors) == 1, "each piece must be monochrome"
        pieces.append(
            {
                "area": int(pm.sum()),
                "mask": pm,
                "patch": patch,
                "color": int(next(iter(colors))),
            }
        )
    assert len(pieces) == len(ordered_slots), "there must be one piece per slot"

    k = None
    for piece in pieces:
        h, w = map(int, piece["mask"].shape)
        if h == w and int(piece["mask"].sum()) == int(h * w):
            k = int(h) if k is None else min(int(k), int(h))
    k = int(k or 1)

    candidates = []
    for pi, piece in enumerate(pieces):
        for vp, vm in _dihedral_variants(piece["patch"], piece["mask"]):
            um = _downsample_blocks(vm, k)
            if um is None:
                continue
            candidates.append(
                {
                    "pi": int(pi),
                    "area": int(piece["area"]),
                    "color": int(piece["color"]),
                    "um": um,
                    "vp": vp,
                    "vm": vm,
                }
            )

    used_piece: set[int] = set()
    placements: list[tuple[dict, dict]] = []
    for slot in ordered_slots:
        need_area = int(slot["area"]) * int(k) * int(k)
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
        assert found is not None, "every slot must match a same-spectrum piece"
        used_piece.add(int(found["pi"]))
        placements.append((slot, found))

    out = np.full(((layout.shape[0] + 2) * k, (layout.shape[1] + 2) * k), frame_color, dtype=int)
    for slot, cand in placements:
        top = (int(slot["r0"]) + 1) * k
        left = (int(slot["c0"]) + 1) * k
        paste(out, cand["vp"], top, left, mask=cand["vm"], overwrite=True)
    return out


def _grow_slot(forbidden: np.ndarray, target_size: int) -> np.ndarray | None:
    h, w = forbidden.shape
    free_cells = np.argwhere(~forbidden)
    if free_cells.size == 0:
        return None
    order = free_cells.tolist()
    random.shuffle(order)
    for start in order[:GROW_SEED_TRIES]:
        r0, c0 = int(start[0]), int(start[1])
        chosen = {(r0, c0)}
        frontier = {(r0 - 1, c0), (r0 + 1, c0), (r0, c0 - 1), (r0, c0 + 1)}
        while len(chosen) < target_size and frontier:
            cands = [
                pos
                for pos in frontier
                if 0 <= pos[0] < h and 0 <= pos[1] < w and not forbidden[pos[0], pos[1]] and pos not in chosen
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
    mask = np.zeros_like(forbidden)
    mask[coords[:, 0], coords[:, 1]] = True
    forbidden |= _dilate8(mask)


def _gen_layout(h: int, w: int, n_slots: int) -> list[dict] | None:
    assert int(n_slots) >= int(N_UNITS)
    forbidden = np.zeros((h, w), dtype=bool)
    slots: list[dict] = []
    used_shape_keys: set[bytes] = set()

    for _ in range(N_UNITS):
        for _ in range(UNIT_SLOT_RETRIES):
            coords = _grow_slot(forbidden, 1)
            if coords is not None:
                slots.append({"coords": coords, "area": 1, "shape_key": b"unit"})
                _forbid_coords(forbidden, coords)
                break
        else:
            return None

    for _ in range(int(n_slots) - int(N_UNITS)):
        size = random.choice(NON_UNIT_SIZE_CHOICES)
        for _ in range(NON_UNIT_SLOT_RETRIES):
            coords = _grow_slot(forbidden, size)
            if coords is None:
                continue
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            sm = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
            sm[coords[:, 0] - r0, coords[:, 1] - c0] = True
            shape_key = _canonical_shape_key(sm)
            if shape_key in used_shape_keys:
                continue
            used_shape_keys.add(shape_key)
            slots.append({"coords": coords, "area": int(size), "mask": sm, "shape_key": shape_key})
            _forbid_coords(forbidden, coords)
            break
        else:
            return None

    density = sum(int(slot["area"]) for slot in slots) / float(h * w)
    if not (HOLE_DENSITY_RANGE[0] <= density <= HOLE_DENSITY_RANGE[1]):
        return None
    if not _slot_row_signal_ok(slots):
        return None
    return slots


def _make_piece(slot: dict, piece_color: int, k: int) -> dict:
    if int(slot["area"]) == 1:
        patch = np.full((k, k), int(piece_color), dtype=int)
        mask = np.ones((k, k), dtype=bool)
        return {
            "piece_color": int(piece_color),
            "slot": slot,
            "out_patch": patch,
            "out_mask": mask,
            "in_patch": patch,
            "in_mask": mask,
        }

    out_mask = np.kron(slot["mask"], np.ones((k, k), dtype=bool))
    out_patch = np.where(out_mask, int(piece_color), 0).astype(int)
    rot = random.randint(0, 3)
    in_mask = np.rot90(out_mask, rot)
    in_patch = np.rot90(out_patch, rot)
    if random.random() < 0.5:
        in_mask = np.fliplr(in_mask)
        in_patch = np.fliplr(in_patch)
    return {
        "piece_color": int(piece_color),
        "slot": slot,
        "out_patch": out_patch,
        "out_mask": out_mask,
        "in_patch": in_patch,
        "in_mask": in_mask,
    }


def _build_expected_output(
    pieces: list[dict], layout_h: int, layout_w: int, k: int, frame_color: int
) -> np.ndarray:
    out = np.full(((layout_h + 2) * k, (layout_w + 2) * k), int(frame_color), dtype=int)
    for piece in pieces:
        coords = piece["slot"]["coords"]
        top = (int(coords[:, 0].min()) + 1) * k
        left = (int(coords[:, 1].min()) + 1) * k
        paste(out, piece["out_patch"], top, left, mask=piece["out_mask"], overwrite=True)
    return out


def _build_layout_colors(slots: list[dict], layout_h: int, layout_w: int, frame_color: int) -> np.ndarray:
    layout = np.full((layout_h, layout_w), int(frame_color), dtype=int)
    for slot in slots:
        layout[slot["coords"][:, 0], slot["coords"][:, 1]] = int(slot["marker_color"])
    return layout


def _build_legend(entries: list[tuple[int, int]]) -> np.ndarray:
    legend = np.zeros((len(entries), 2), dtype=int)
    for r, (marker_color, piece_color) in enumerate(entries):
        legend[r, 0] = int(marker_color)
        legend[r, 1] = int(piece_color)
    return legend


def _build_input_canvas(
    pieces: list[dict],
    layout_colors: np.ndarray,
    layout_h: int,
    layout_w: int,
    frame_color: int,
    legend_entries: list[tuple[int, int]],
    distractor_color: int,
) -> np.ndarray | None:
    frame_h, frame_w = layout_h + 2, layout_w + 2
    legend = _build_legend(legend_entries)
    legend_h, legend_w = legend.shape
    total_piece_area = sum(int(piece["in_mask"].sum()) for piece in pieces)
    needed_cells = frame_h * frame_w + legend_h * legend_w + AREA_BUFFER_MULT * total_piece_area + AREA_BUFFER_CONST

    min_h = max(CANVAS_DIM_RANGE[0], frame_h + 2, legend_h + 4)
    min_w = max(CANVAS_DIM_RANGE[0], frame_w + 2, legend_w + 4)
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
    paste(canvas, legend, 0, 0)

    frame_img = np.full((frame_h, frame_w), int(frame_color), dtype=int)
    frame_img[1:-1, 1:-1] = layout_colors
    try:
        fr, fc = random_free_location(canvas, frame_img, transparent=0, border=0, padding=1, padding_connectivity=8)
    except ValueError:
        return None
    paste(canvas, frame_img, fr, fc)

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
        paste(canvas, piece["in_patch"], pr, pc, mask=piece["in_mask"], overwrite=True)

    target = random.randint(*DISTRACTOR_RANGE)
    placed = 0
    for _ in range(target * DISTRACTOR_RETRY_MULT):
        if placed >= target:
            break
        rr = random.randint(0, canvas_h - 1)
        cc = random.randint(0, canvas_w - 1)
        if canvas[rr, cc] != 0:
            continue
        r0, r1 = max(0, rr - 1), min(canvas_h, rr + 2)
        c0, c1 = max(0, cc - 1), min(canvas_w, cc + 2)
        if np.any(canvas[r0:r1, c0:c1] != 0):
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

    slots = _gen_layout(layout_h, layout_w, n_slots)
    if slots is None:
        return None

    layout_mask = np.zeros((layout_h, layout_w), dtype=bool)
    for slot in slots:
        layout_mask[slot["coords"][:, 0], slot["coords"][:, 1]] = True
    if not (layout_mask[0].any() and layout_mask[-1].any() and layout_mask[:, 0].any() and layout_mask[:, -1].any()):
        return None

    ordered_slots = sorted(slots, key=_slot_order_key)
    piece_colors = _row_palette(len(ordered_slots))

    all_colors = set(range(1, 10))
    remaining = list(all_colors - set(piece_colors))
    random.shuffle(remaining)
    need_remaining = 1 + len(ordered_slots)
    if len(remaining) < need_remaining:
        return None
    frame_color = int(remaining.pop())
    marker_colors = [int(remaining.pop()) for _ in range(len(ordered_slots))]

    for slot, piece_color, marker_color in zip(ordered_slots, piece_colors, marker_colors):
        slot["piece_color"] = int(piece_color)
        slot["marker_color"] = int(marker_color)

    pieces = [_make_piece(slot, int(slot["piece_color"]), k) for slot in ordered_slots]
    expected_out = _build_expected_output(pieces, layout_h, layout_w, k, frame_color)

    legend_entries = [(int(slot["marker_color"]), int(slot["piece_color"])) for slot in ordered_slots]
    layout_colors = _build_layout_colors(ordered_slots, layout_h, layout_w, frame_color)
    distractor_color = int(random.choice(piece_colors))
    canvas = _build_input_canvas(
        pieces,
        layout_colors,
        layout_h,
        layout_w,
        frame_color,
        legend_entries,
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
        canvas = _gen_attempt()
        if canvas is not None:
            return canvas
    raise RuntimeError(f"generate_input: failed after {GENERATE_MAX_RETRIES} retries")
