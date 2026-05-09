# Combo variant of 898e7135 (index 39)
# combo_spec: D4/2+D14/2
# Name: 编码确认缩放因子

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
# Design analysis for the combo variant D4/2 + D14/2
#
# a) 具体化与行为追踪
# - 原题：外部碎片是按 k 放大的洞形；solver 通过“最小实心正方形碎片的边长”读出 k，
#   然后把每个碎片按 k*k 块降采样，与框架内部洞形做 D4 匹配。
# - 本融合版：k 在数据分布里固定为 2，但输入里不再提供 unit 洞，也不允许任何洞形本身是
#   实心正方形，因此“最小正方形碎片”这条捷径被彻底切断。取而代之，框架边框上嵌入一圈
#   marker_color 标记，四边非角点位置的标记数分别是 k, 2k, 3k, 4k，即 2/4/6/8。
# - 具体例子：若某例有 4 个槽洞，外面散落 4 个彩色碎片；框架左、右两边各能看到 6 个和
#   8 个标记点，上、下两边看到 2 个和 4 个标记点。solver 先定位最大前景组件作为框架，
#   再仅统计其边框四边上的非框架色像素数，恢复 k=2；之后仍按块级形状匹配碎片与洞。
# - 差异：原题在“几何块大小 -> k”这一步依赖外部碎片；融合版在这一步依赖边框数字编码，
#   且生成器专门避免让碎片重新泄露 k。
#
# b) 复杂度校准
# - k 只保留单一档位 2，符合 D4/2 的“固定缩放因子”要求，也缩小搜索空间。
# - 为了让 D14 的改动仍然肉眼可见，边框四边始终呈现 2/4/6/8 的数量梯度，而不是仅放 2 个
#   标记；这样边框编码在少量样例里足够醒目。
# - 槽洞仍保留 3-6 个、面积 2-7、D4 形状唯一、8 分离，避免匹配歧义；若自由度过高，
#   外部碎片就会太乱，若自由度过低，又会和单变体样例太像。
#
# c) 隐式假设审查
# - `frame_comp = max(comps, key=area)` 仍成立：marker 嵌在边框上，与框架连为一个多色组件，
#   但面积远大于任一碎片或干扰点。
# - `frame_color = dominant color of frame_comp` 仍成立：2/4/6/8 的标记总数上限是 20，而框架
#   主体面积明显更大，因此主色仍是 frame_color。
# - 原代码“扫描最小实心正方形碎片读 k”的前提被明确打破：没有 unit 槽，也禁止实心方块槽。
# - `hole = sub == bg` 仍成立：标记只替换最外层边框像素，不进入内部洞区，紧裁洞区仍能恢复布局。
# - `pieces = non-singleton components excluding frame` 仍成立：标记附着在框架上，不会被分离成碎片。
# - `downsample_blocks` 现在依赖边框解出的 k；生成器保证所有碎片确实由 2x2 实心块拼接，因此
#   不会出现部分块。
#
# d) 放置合法性
# - 边框标记只放在非角点边框格，不改变框架 bbox，不打断 4 连通。
# - 布局尺寸设置为 inner_h 8-12、inner_w 4-8，保证 2/4/6/8 个标记都能塞进对应边的非角点长度。
# - 洞区之间依旧 8 分离；外部碎片用 `random_free_location(..., padding=1, padding_connectivity=8)`
#   放置，与框架和彼此保持至少 1 格背景间隔。
# - 干扰物仍是孤立单像素，颜色独立，不会误并入碎片组件。
#
# e) 视觉可观察性
# - 解题者在 2-4 个样例里能稳定看到：框架边框不再是纯色，而是带有一圈规则数量的计数标记。
# - 输出始终是 2 倍放大后的框架；由于没有 unit 洞或小实心方块碎片，最直接的尺度线索只剩边框编码。
# - 四边数量是 1:2:3:4 的整齐比例，容易被人眼注意到，也比随机数量更像“数字提示”。
#
# f) 可视化整洁度
# - 标记只占据最外层边框；洞、墙、碎片、干扰点各自处于不同空间层级，没有区域叠加。
# - 每个碎片单色、每个非 unit 槽形 D4 唯一，避免出现大量同色同形干扰。
# - 输出图不保留标记，只保留放大后的框架和填入后的碎片，延续原题简洁外观。
#
# g) 变化的可推断性
# - 本融合不引入颜色映射；颜色只承担“框架 / 标记 / 碎片 / 干扰”角色，推理重点是数量编码。
# - 标记颜色全边统一，解题者只需数数量，不需要猜颜色语义。
# - 变化主要来自槽洞布局与碎片朝向，而不是颜色规则，因此 2-3 组样例就足以归纳。
#
# 9) 新 solver 的额外假设与生成器保证
# - 假设四边边框上所有非 frame_color 像素都属于同一种 marker_color：
#   生成器固定使用单一标记色。
# - 假设四边非角点计数排序后恰为 [k, 2k, 3k, 4k]：
#   生成器固定按该模板编码，且本变体里 k 总是 2。
# - 假设没有任何碎片能通过“自身是实心正方形”把 k 再泄露出来：
#   生成器显式禁止 unit 槽和实心正方形槽。
# ---------------------------------------------------------------------------


K_FIXED = 2
LAYOUT_H_RANGE = (8, 12)
LAYOUT_W_RANGE = (4, 8)
N_SLOTS_RANGE = (3, 6)
NON_UNIT_SIZE_CHOICES = [2, 3, 4, 5, 6, 7]

HOLE_DENSITY_RANGE = (0.20, 0.50)
NON_UNIT_SLOT_RETRIES = 45
GROW_SEED_TRIES = 10

CANVAS_DIM_RANGE = (18, 30)
AREA_BUFFER_MULT = 4
AREA_BUFFER_CONST = 60

DISTRACTOR_RANGE = (8, 13)
DISTRACTOR_RETRY_MULT = 10
MIN_DISTRACTORS = 8

GENERATE_MAX_RETRIES = 500


def _marker_counts_for_k(k: int) -> tuple[int, int, int, int]:
    """Return the edge marker counts (top, bottom, left, right)."""
    k = int(k)
    return k, 2 * k, 3 * k, 4 * k


def _decode_k_from_frame_patch(frame_patch: np.ndarray, frame_color: int) -> int:
    """Recover k from the four border marker counts."""
    top = frame_patch[0, 1:-1]
    bottom = frame_patch[-1, 1:-1]
    left = frame_patch[1:-1, 0]
    right = frame_patch[1:-1, -1]
    edges = [top, bottom, left, right]

    marker_values: list[int] = []
    counts: list[int] = []
    for edge in edges:
        mask = edge != int(frame_color)
        counts.append(int(mask.sum()))
        marker_values.extend(map(int, edge[mask].tolist()))

    assert marker_values, "frame border must carry marker pixels"
    assert len(set(marker_values)) == 1, "all border markers must share one color"

    sorted_counts = sorted(counts)
    k = int(sorted_counts[0])
    assert sorted_counts == [k, 2 * k, 3 * k, 4 * k], "border code must follow 1:2:3:4"
    return k


def _dihedral_variants(
    patch: np.ndarray, mask: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Unique D4 variants of (patch, mask), deduplicated by mask shape+contents."""
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand the frame's hole layout by the encoded scale and place each matching piece.

    Input structure:
    - Background is the most common color. The largest foreground component is a mostly
      monochrome rectangular frame whose outer border also contains one marker color.
    - Outside the frame are monochrome piece sprites made from solid 2x2 blocks, plus
      isolated singleton distractors in a separate color.

    Rule:
    - Read the scale factor k from the counts of border markers on the frame edges rather
      than from any square-piece shortcut. Then extract the hole regions inside the frame,
      match each outside piece to a hole by block-level shape under D4 symmetry, and paste
      the matched piece into the k-scaled frame layout.

    Key insight:
    - Geometry alone no longer reveals k because the generator removes unit holes and filled
      square slots. The reliable scale cue is the 1:2:3:4 marker-count pattern on the border,
      after which the original block-downsampling shape match still solves the task.
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
        sm = np.zeros((int(rr1 - rr0 + 1), int(cc1 - cc0 + 1)), dtype=bool)
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

    candidates = []
    for pi, piece in enumerate(pieces):
        for vp, vm in _dihedral_variants(piece["patch"], piece["mask"]):
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


def _canonical_shape_key(mask: np.ndarray) -> bytes:
    """Canonical bytes representation under D4 for a tight-cropped bool mask."""
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
    """Dilate a bool mask by one cell under 8-connectivity."""
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
    """Mark coords and all 8-neighbors as forbidden in-place."""
    cell_mask = np.zeros_like(forbidden)
    cell_mask[coords[:, 0], coords[:, 1]] = True
    forbidden |= _dilate8(cell_mask)


def _is_forbidden_shape(mask: np.ndarray) -> bool:
    """Reject filled squares so geometry cannot reveal k without border decoding."""
    h, w = map(int, mask.shape)
    return h == w and bool(mask.all())


def _gen_layout(h: int, w: int, n_slots: int) -> list[dict] | None:
    """Generate D4-unique, pairwise 8-separated non-unit slots."""
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
    """Build the output-side piece and a random D4-transformed input-side piece."""
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
    """Choose distinct non-corner positions along one border edge."""
    interior_len = int(frame_len) - 2
    assert 0 <= int(count) <= interior_len
    if count == 0:
        return []
    return [1 + (i * interior_len) // int(count) for i in range(int(count))]


def _build_frame_image(
    layout_mask: np.ndarray, frame_color: int, marker_color: int, k: int
) -> np.ndarray:
    """Build the frame with a border code that reveals k."""
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
    """Build the input grid: coded frame + scattered pieces + singleton distractors."""
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
    k = K_FIXED
    layout_h = random.randint(*LAYOUT_H_RANGE)
    layout_w = random.randint(*LAYOUT_W_RANGE)
    n_slots = random.randint(*N_SLOTS_RANGE)

    palette = list(range(1, 10))
    random.shuffle(palette)
    if len(palette) < n_slots + 3:
        return None
    piece_colors = palette[:n_slots]
    frame_color = palette[n_slots]
    marker_color = palette[n_slots + 1]
    distractor_color = palette[n_slots + 2]

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
