# Variant of 898e7135
# Dimension: D5 — 输出裁剪方式 (grid_shape)
# Change: crop — 输出为 (layout_h+2)*k × (layout_w+2)*k 的紧密裁剪，即放大后的完整框架（含边框）
#   → 输出为打包网格（精灵紧密排列）

from __future__ import annotations

"""
design_analysis:
  variant_goal:
    keep:
      - 输入仍然是一个单色框架，内部是若干 4 连通洞区域，框架外散落与洞形状对应的 k×k-block 精灵
      - 精灵与洞的匹配规则仍然是块级形状在旋转/翻转下匹配
      - k 仍由最小实心正方形精灵推断
    change:
      - 原题输出是“放大后的完整框架 + 填回洞中”
      - 变体输出不再保留框架，而是把已经匹配好的精灵按洞区域的行列顺序打包成紧凑网格
      - 网格内部只有 1 像素分隔线，没有外框；背景色仍使用框架色，保证分隔线可见

  concrete_behavior:
    slot_order_keys:
      - 每个洞区域取其 tight bbox 左上角 (r0, c0) 作为该洞在布局中的锚点
      - 输出网格的行顺序 = 不同 r0 的升序
      - 输出网格的列顺序 = 不同 c0 的升序
    packing_geometry:
      - 同一输出行中的单元高度取该行所有精灵高度的最大值
      - 同一输出列中的单元宽度取该列所有精灵宽度的最大值
      - 精灵放在对应单元左上角，行与行、列与列之间隔 1 像素框架色
    small_trace:
      original:
        - 若三个洞锚点是 (0,1), (0,4), (3,1)，原 solver 会把三个精灵放回放大框架中的对应洞位置
      variant:
        - 行键为 [0,3]，列键为 [1,4]
        - 第一行放两个精灵，第二行只放左列精灵，右下单元留框架色背景
      difference:
        - 原题保留空间绝对位置和完整边框
        - 变体只保留洞之间的相对行列顺序，裁掉框架与无关空白

  complexity_calibration:
    - 仍保留 3-6 个洞、1-2 个 unit slot、非 unit 形状在 D4 下互异，避免匹配歧义
    - 为了让 packed-grid 规则肉眼可见，generator 额外要求：
      - 洞锚点只分布在 2-3 个不同的行组、2-3 个不同的列组
      - 至少有一行包含 2 个以上洞，且至少有一列包含 2 个以上洞
    - 这样样例不会退化成单条带状输出，也不会因为行列组过多而让 packed-grid 变得稀疏、比原图更松散

  implicit_assumption_audit:
    preserved_logic:
      - frame detection via largest foreground component: 仍成立
      - layout extraction via bg holes inside frame bbox: 仍成立
      - block downsampling and D4 matching: 仍成立
    old_lines_that_break_under_D5:
      - outH/outW = (layout_h+2)*k: 依赖“输出是完整放大框架”，变体中失效
      - top/left = (slot_r0+1)*k: 依赖“输出坐标系与框架对齐”，变体中失效
    new_solver_assumptions:
      - 使用洞 bbox 的 (r0,c0) 作为 packed-grid 索引键
      - generator 必须保证这些键在样例中形成可观察的二维顺序，而不是总退化成单行/单列

  placement_legality:
    - 输入端仍沿用原题放置策略：框架与精灵在背景上分离，精灵之间至少隔 1 个背景像素
    - 输出端 packed-grid 的每行高度 / 每列宽度先整体计算，再放置精灵，天然不会重叠或越界

  observability:
    - 不用颜色值排序；排序只依赖洞在框架内的可见几何位置
    - 由于精灵是纯色块形，规则变化主要体现在输出整体形状与对象相对布局，而非不可见属性
    - 背景仍用框架色，分隔线稳定可见，空单元也可读

  cleanliness:
    - 输出只包含框架色背景和已匹配精灵，不叠加框架/洞/标记多层信息
    - 1 像素分隔线足够区分对象，同时不会让输出过大或过乱
"""

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

GENERATE_MAX_RETRIES = 600
PACK_GAP = 1


def _downsample_blocks(mask: np.ndarray, k: int) -> np.ndarray | None:
    hh, ww = map(int, mask.shape)
    if hh % k != 0 or ww % k != 0:
        return None
    out = np.zeros((hh // k, ww // k), dtype=bool)
    for r in range(out.shape[0]):
        for c in range(out.shape[1]):
            block = mask[r * k : (r + 1) * k, c * k : (c + 1) * k]
            if bool(block.all()):
                out[r, c] = True
            elif bool(block.any()):
                return None
    return out


def _dihedral_variants(
    patch: np.ndarray, mask: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray]]:
    base: list[tuple[np.ndarray, np.ndarray]] = []
    for kk in range(4):
        p = np.rot90(patch, kk)
        m = np.rot90(mask, kk)
        base.append((p, m))
        base.append((np.fliplr(p), np.fliplr(m)))
    uniq: list[tuple[np.ndarray, np.ndarray]] = []
    seen: set[bytes] = set()
    for p, m in base:
        key = m.tobytes()
        if key in seen:
            continue
        seen.add(key)
        uniq.append((p, m))
    return uniq


def _slot_anchor(slot: dict) -> tuple[int, int]:
    return int(slot["r0"]), int(slot["c0"])


def _pack_sprites_by_slot_grid(items: list[dict], bgc: int) -> np.ndarray:
    assert items, "packed grid needs at least one item"

    row_keys = sorted({int(item["r0"]) for item in items})
    col_keys = sorted({int(item["c0"]) for item in items})
    row_index = {int(r): i for i, r in enumerate(row_keys)}
    col_index = {int(c): j for j, c in enumerate(col_keys)}

    row_heights = [0] * len(row_keys)
    col_widths = [0] * len(col_keys)
    for item in items:
        h, w = map(int, item["patch"].shape)
        row_heights[row_index[int(item["r0"])]] = max(row_heights[row_index[int(item["r0"])]], h)
        col_widths[col_index[int(item["c0"])]] = max(col_widths[col_index[int(item["c0"])]], w)

    out_h = sum(row_heights) + PACK_GAP * max(0, len(row_heights) - 1)
    out_w = sum(col_widths) + PACK_GAP * max(0, len(col_widths) - 1)
    out = np.full((out_h, out_w), int(bgc), dtype=int)

    row_starts: list[int] = []
    cur = 0
    for h in row_heights:
        row_starts.append(cur)
        cur += int(h) + PACK_GAP
    col_starts: list[int] = []
    cur = 0
    for w in col_widths:
        col_starts.append(cur)
        cur += int(w) + PACK_GAP

    for item in items:
        rr = row_starts[row_index[int(item["r0"])]]
        cc = col_starts[col_index[int(item["c0"])]]
        paste(out, item["patch"], rr, cc, mask=item["mask"], overwrite=True)

    return out


def _extract_slots(grid: np.ndarray, bg: int) -> tuple[np.ndarray, list[dict], int]:
    comps = find_components(grid, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    assert comps, "expected at least one foreground component"
    frame_comp = max(comps, key=lambda d: int(d["area"]))
    fr0, fc0, fr1, fc1 = map(int, frame_comp["bbox"])

    frame_cols = [int(grid[int(r), int(c)]) for r, c in frame_comp["coords"]]
    bgc = int(Counter(frame_cols).most_common(1)[0][0])

    sub = crop_bbox(grid, (fr0, fc0, fr1, fc1))
    hole = sub == int(bg)
    hole_box = bbox(hole)
    assert hole_box is not None, "frame must enclose at least one bg hole"
    lr0, lc0, lr1, lc1 = map(int, hole_box)
    layout = hole[lr0 : lr1 + 1, lc0 : lc1 + 1]

    slots: list[dict] = []
    for coords in components_mask(layout, connectivity=4):
        rr0, cc0, rr1, cc1 = map(int, bbox_coords(coords))
        sm = np.zeros((rr1 - rr0 + 1, cc1 - cc0 + 1), dtype=bool)
        sm[coords[:, 0] - rr0, coords[:, 1] - cc0] = True
        slots.append({"area": int(sm.sum()), "r0": rr0, "c0": cc0, "mask": sm})
    slots.sort(key=lambda s: (-int(s["area"]), int(s["r0"]), int(s["c0"])))
    assert slots, "frame must contain at least one slot"

    return layout, slots, bgc


def _extract_pieces(grid: np.ndarray, bg: int) -> list[dict]:
    comps = find_components(grid, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    assert comps, "expected at least one foreground component"
    frame_comp = max(comps, key=lambda d: int(d["area"]))

    pieces: list[dict] = []
    for comp in comps:
        if comp is frame_comp:
            continue
        coords = comp["coords"]
        if coords.size == 0 or int(coords.shape[0]) <= 1:
            continue
        pr0, pc0, pr1, pc1 = map(int, comp["bbox"])
        patch = crop_bbox(grid, (pr0, pc0, pr1, pc1))
        pm = np.zeros(patch.shape, dtype=bool)
        pm[coords[:, 0] - pr0, coords[:, 1] - pc0] = True
        pieces.append({"area": int(pm.sum()), "mask": pm, "patch": patch})
    assert pieces, "at least one non-frame piece must exist"
    return pieces


def _infer_k(pieces: list[dict]) -> int:
    k = None
    for p in pieces:
        h, w = map(int, p["mask"].shape)
        if h == w and int(p["mask"].sum()) == int(h * w):
            k = int(h) if k is None else min(int(k), int(h))
    return int(k or 1)


def _match_placements(grid: np.ndarray) -> tuple[list[tuple[dict, dict]], int]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    _layout, slots, bgc = _extract_slots(g, bg)
    pieces = _extract_pieces(g, bg)
    k = _infer_k(pieces)

    candidates = []
    for pi, piece in enumerate(pieces):
        for vp, vm in _dihedral_variants(piece["patch"], piece["mask"]):
            um = _downsample_blocks(vm, k)
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
    return placements, bgc


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match the scattered pieces to frame holes, then pack the matched sprites into
    a compact grid ordered by the holes' row/column anchors.
    """
    placements, bgc = _match_placements(grid)
    packed_items = [
        {
            "r0": int(slot["r0"]),
            "c0": int(slot["c0"]),
            "patch": cand["vp"],
            "mask": cand["vm"],
        }
        for slot, cand in placements
    ]
    return _pack_sprites_by_slot_grid(packed_items, bgc)


def _canonical_shape_key(mask: np.ndarray) -> bytes:
    variants = []
    for kk in range(4):
        r = np.rot90(mask, kk)
        variants.append((tuple(map(int, r.shape)), r.tobytes()))
        fr = np.fliplr(r)
        variants.append((tuple(map(int, fr.shape)), fr.tobytes()))
    return min(variants)[1]


def _grow_slot(forbidden: np.ndarray, target_size: int) -> np.ndarray | None:
    h, w = map(int, forbidden.shape)
    free_cells = np.argwhere(~forbidden)
    if free_cells.size == 0:
        return None
    free_list = [tuple(map(int, cell)) for cell in free_cells]
    random.shuffle(free_list)
    for r0, c0 in free_list[:GROW_SEED_TRIES]:
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
            r0, c0 = map(int, coords.min(axis=0))
            r1, c1 = map(int, coords.max(axis=0))
            sm = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
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
    if int(slot["area"]) == 1:
        patch = np.full((k, k), int(color), dtype=int)
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
    out_patch = np.where(out_mask, int(color), 0).astype(int)

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


def _packed_grid_is_observable(slots: list[dict]) -> bool:
    anchors = [tuple(map(int, slot["coords"].min(axis=0))) for slot in slots]
    rows = [int(r) for r, _ in anchors]
    cols = [int(c) for _, c in anchors]
    row_counts = Counter(rows)
    col_counts = Counter(cols)
    return (
        len(row_counts) >= 2
        and len(row_counts) <= 3
        and len(col_counts) >= 2
        and len(col_counts) <= 3
        and max(row_counts.values()) >= 2
        and max(col_counts.values()) >= 2
    )


def _build_expected_output(pieces: list[dict], frame_color: int) -> np.ndarray:
    items = []
    for piece in pieces:
        coords = piece["slot"]["coords"]
        r0, c0 = map(int, coords.min(axis=0))
        items.append({"r0": r0, "c0": c0, "patch": piece["out_patch"], "mask": piece["out_mask"]})
    return _pack_sprites_by_slot_grid(items, int(frame_color))


def _build_input_canvas(
    pieces: list[dict],
    layout_mask: np.ndarray,
    layout_h: int,
    layout_w: int,
    frame_color: int,
    distractor_color: int,
) -> np.ndarray | None:
    frame_h, frame_w = int(layout_h + 2), int(layout_w + 2)
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
    n_units = random.randint(1, min(N_UNITS_MAX, n_slots - 1))

    palette = list(range(1, 10))
    random.shuffle(palette)
    frame_color = int(palette.pop())
    distractor_color = int(palette.pop())
    needed_piece_colors = (n_slots - n_units) + 1
    if len(palette) < needed_piece_colors:
        return None
    non_unit_colors = palette[: n_slots - n_units]
    unit_color = int(palette[n_slots - n_units])

    slots = _gen_layout(layout_h, layout_w, n_slots, n_units)
    if slots is None:
        return None
    if not _packed_grid_is_observable(slots):
        return None

    unit_slots = slots[:n_units]
    non_unit_slots = slots[n_units:]

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

    pieces = [
        _make_piece(slot, non_unit_colors[i], k, apply_dihedral=True)
        for i, slot in enumerate(non_unit_slots)
    ]
    pieces += [_make_piece(slot, unit_color, k, apply_dihedral=False) for slot in unit_slots]

    expected_out = _build_expected_output(pieces, frame_color)
    if expected_out.size == ((layout_h + 2) * k) * ((layout_w + 2) * k):
        # Avoid frequent accidental shape matches with the original crop-style output.
        if expected_out.shape == ((layout_h + 2) * k, (layout_w + 2) * k):
            return None

    canvas = _build_input_canvas(
        pieces, layout_mask, layout_h, layout_w, frame_color, distractor_color
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
    raise RuntimeError(
        "generate_input: failed to generate after %d attempts" % GENERATE_MAX_RETRIES
    )
