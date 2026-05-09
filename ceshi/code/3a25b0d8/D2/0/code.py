# Variant of 3a25b0d8
# Dimension: D2 — 源对象结构 (src)
# Change: 单个 8-连通精灵：轮廓骨架 + 彩色子精灵填充空腔；LR 对称；4-9 个空腔... → 多个独立色块作为 src
#
# Design analysis
# spec:
#   core_change:
#     original_src: 单个 8 连通轮廓对象，彩色子精灵被包在框架空腔里
#     variant_src: 4-9 个彼此 8 断开的单色块，直接散落在画布背景上
#   concrete_pixels:
#     dst:
#       keep: 原题 dst 保持为单个带洞轮廓对象，洞数 4-9，输出仍是把这些洞按顺序填色后 tight-crop
#     src:
#       now: 不再有 src 框架，输入里除 dst 外剩下的所有非背景 8 连通块都属于 src
#       shape: 每个色块仍来自原 cavity 的布尔形状，因此保留不规则块、多色、大小变化
#       placement: 色块按“从上到下、同一行从左到右”的可见秩序摆放；不同样例中行列间距会变化，但块彼此不会接触
#   behavior_trace:
#     old_pipeline:
#       - 找两个大 8 连通对象
#       - 其中一个对象内部找洞，另一个对象内部找彩色子精灵
#       - 按质心秩序配对
#     new_pipeline:
#       - 找唯一一个“内部有洞”的 8 连通对象作为 dst
#       - 其余所有非背景 8 连通块全部视为 src 色块
#       - 将 src 色块和 dst 洞都按 bbox 中心的 row-major 顺序排序，再逐个填洞
#     concrete_example:
#       - 原题某行可能看到“框架内左洞是红、右洞是绿”
#       - 变体中对应输入会变成“画布另一侧有两个散开的块：左上红、右上绿”
#       - 输出仍是 dst 左洞填红、右洞填绿，但求解时必须先做全局收集，不能再在某个 src 框内分解
#   complexity_calibration:
#     keep:
#       - 仍有 4-9 个洞 / 4-9 个源块
#       - 仍有 3-7 个填充色，仍允许镜像对颜色共享
#       - dst 仍保留原题的不规则洞形与单轮廓包裹
#     simplify_for_observability:
#       - 不再引入旋转/翻转分支；原始评测样例实际也始终是 identity
#       - 匹配规则固定为可见的 top-to-bottom, left-to-right 顺序，避免“无框架 src + 浮动质心 + 多列并列”带来的隐式歧义
#   implicit_assumption_audit:
#     broken_old_assumptions:
#       - `_extract_two_objects()` 假设只有两个大对象：变体中失效，因为 src 被拆成 4-9 个组件
#       - `_colored_components(src_subgrid)` 假设所有源信息都在一个裁剪框里：变体中失效，因为没有统一 src 框架
#       - `dest_i = argmax(hole_area)` 建立在“两对象对决”上：变体中要改成“唯一带洞组件”
#       - 对 src 做旋转/翻转是基于单一 src bbox 的：变体中不再需要
#     new_assumptions:
#       - 输入里只有一个带洞组件，这是 dst
#       - 其余 8 连通组件都是单色 src 块
#       - src 块之间必须 8 断开，且数量与 dst 洞数一致
#       - row-major 顺序必须稳定可见，因此生成器要控制行间距和列间距
#   placement_legality:
#     generator_guarantees:
#       - src 每个块之间至少隔 1 个背景像素的 8 邻域距离，避免误并成一个组件
#       - src 整体 patch 与 dst 放置时使用透明随机摆放，不越界、不碰撞
#       - canvas 比原题略放大，以容纳“稀疏散点 src”而不挤乱视觉
#   visual_observability:
#     rationale:
#       - 人类能直接看到：一个轮廓目标对象 + 旁边一组散开的彩色块
#       - 源块行列顺序与 dst 洞的行列顺序对应，2-3 个样例足以归纳
#       - 没有任何基于颜色编号、面积编号等不可见属性的排序
#   cleanliness:
#     choices:
#       - src 块不与 dst 接壤
#       - 不使用额外标记、辅助符号或混合轮廓，保证画面只有“一个目标对象 + 多个散块”
#   color_inference:
#     rule:
#       - 颜色不变，只是从 source block 复制到对应 hole
#       - 颜色种类仍限制在 3-7，且通过位置顺序提供足够线索

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox, crop_bbox, get_bg, label_mask, paste, random_free_location


def main(grid: np.ndarray) -> np.ndarray:
    """
    Variant rule:
    - Find the unique 8-connected component that contains enclosed background holes: dst.
    - Treat every other non-background 8-connected component as an independent source block.
    - Sort source blocks and dst holes by visible row-major order of bbox centers.
    - Fill holes with the paired source colors and tight-crop the result.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    def _components(mask: np.ndarray, connectivity: int = 8) -> list[dict]:
        lab, n = label_mask(mask, connectivity=connectivity)
        comps: list[dict] = []
        for cid in range(1, int(n) + 1):
            pts = np.argwhere(lab == int(cid))
            if pts.size == 0:
                continue
            r0, c0 = pts.min(axis=0)
            r1, c1 = pts.max(axis=0)
            comps.append(
                {
                    "cid": int(cid),
                    "pts": pts,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "size": int(len(pts)),
                }
            )
        comps.sort(key=lambda d: (int(d["bbox"][0]), int(d["bbox"][1]), -int(d["size"])))
        return comps

    def _extract_subgrid(comp: dict) -> np.ndarray:
        r0, c0, r1, c1 = comp["bbox"]
        sub = np.full((int(r1 - r0 + 1), int(c1 - c0 + 1)), int(bg), dtype=int)
        for rr, cc in comp["pts"]:
            sub[int(rr) - int(r0), int(cc) - int(c0)] = int(g[int(rr), int(cc)])
        return sub

    def _center2_from_bbox(b: tuple[int, int, int, int]) -> tuple[int, int]:
        r0, c0, r1, c1 = b
        return int(r0 + r1), int(c0 + c1)

    def _enclosed_bg_components(sub: np.ndarray) -> list[dict]:
        mask = sub == int(bg)
        lab, n = label_mask(mask, connectivity=4)
        H, W = sub.shape
        holes: list[dict] = []
        for cid in range(1, int(n) + 1):
            pts = np.argwhere(lab == int(cid))
            if pts.size == 0:
                continue
            r0, c0 = pts.min(axis=0)
            r1, c1 = pts.max(axis=0)
            if bool(r0 == 0 or c0 == 0 or r1 == int(H - 1) or c1 == int(W - 1)):
                continue
            holes.append(
                {
                    "pts": pts,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "center2": _center2_from_bbox((int(r0), int(c0), int(r1), int(c1))),
                    "size": int(len(pts)),
                }
            )
        return holes

    comps = _components(g != int(bg), connectivity=8)
    if not comps:
        return g.copy()

    hole_infos: list[list[dict]] = []
    for comp in comps:
        hole_infos.append(_enclosed_bg_components(_extract_subgrid(comp)))

    dst_idx = None
    best_score = (-1, -1)
    for i, holes in enumerate(hole_infos):
        if not holes:
            continue
        score = (len(holes), sum(int(h["size"]) for h in holes))
        if score > best_score:
            best_score = score
            dst_idx = i

    if dst_idx is None:
        m = g != int(bg)
        b = bbox(m)
        return crop_bbox(g, b) if b is not None else g.copy()

    dest = _extract_subgrid(comps[dst_idx]).copy()
    dest_holes = hole_infos[dst_idx]
    src_comps = [comp for i, comp in enumerate(comps) if i != int(dst_idx)]

    if not dest_holes or not src_comps:
        m = dest != int(bg)
        b = bbox(m)
        return crop_bbox(dest, b) if b is not None else dest

    dest_list = sorted(
        dest_holes,
        key=lambda h: (int(h["center2"][0]), int(h["center2"][1]), int(h["bbox"][0]), int(h["bbox"][1])),
    )

    src_list = []
    for comp in src_comps:
        vals = [int(g[int(rr), int(cc)]) for rr, cc in comp["pts"]]
        color = Counter(vals).most_common(1)[0][0]
        src_list.append(
            {
                "color": int(color),
                "center2": _center2_from_bbox(comp["bbox"]),
                "bbox": comp["bbox"],
            }
        )
    src_list.sort(
        key=lambda s: (int(s["center2"][0]), int(s["center2"][1]), int(s["bbox"][0]), int(s["bbox"][1]))
    )

    for src, hole in zip(src_list, dest_list):
        for rr, cc in hole["pts"]:
            dest[int(rr), int(cc)] = int(src["color"])

    b = bbox(dest != int(bg))
    return crop_bbox(dest, b) if b is not None else dest


# --------------- generator ---------------

CANVAS_SIZE_RANGE = (26, 34)
OBJECT_H_RANGE = (7, 17)
OBJECT_W_RANGE = (7, 13)

CAVITY_COUNT_RANGE = (4, 9)
FILL_COLOR_COUNT_RANGE = (3, 7)

ROW_HEIGHT_CHOICES = [1, 2, 2, 3, 3, 3]
LAYOUT_K_CHOICES_FULL = [1, 2, 2, 3, 3]
LAYOUT_K_CHOICES_SMALL = [1, 2, 2]
ROW_GAP_CHOICES = [1, 1, 2]

CENTERED_SLOT_WIDTHS = [3, 3, 5, 5, 7]
PAIR_SLOT_WIDTHS = [2, 2, 3, 3, 4]
SIDE_SLOT_WIDTHS = [2, 2, 3]
MID_SLOT_WIDTHS = [1, 3]
PAIR_AXIS_GAP_CHOICES = [1, 1, 3]
TRIO_MID_SIDE_GAP = 1

NONRECT_PROBABILITY = 0.8
MAX_MUTATE_ATTEMPTS = 40
COLOR_MISMATCH_RATE = 0.10

PLACEMENT_BORDER = 1
PLACEMENT_PADDING = 2
SCATTER_COL_GAP_CHOICES = [2, 3, 3, 4]
SCATTER_ROW_GAP_CHOICES = [2, 3, 3, 4]
SCATTER_INNER_PAD = 1

MAX_GEN_ATTEMPTS = 400
MAX_LAYOUT_R_BUDGET = 24


def _random_rect_shape(h: int, w: int) -> np.ndarray:
    return np.ones((h, w), dtype=bool)


def _touches_all_bbox_sides(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any())


def _mutate_shape(mask: np.ndarray, *, n_remove_max: int, symmetric: bool = False) -> np.ndarray:
    """
    Remove boundary cells while preserving:
    - 4-connectivity
    - occupancy on each outer side of the original bbox
    This keeps bbox centers stable enough for visible row-major ordering.
    """
    m = mask.copy()
    H, W = m.shape
    n_remove = random.randint(1, max(1, n_remove_max))
    tried = 0
    removed = 0
    while removed < n_remove and tried < MAX_MUTATE_ATTEMPTS:
        tried += 1
        cand = []
        for r in range(H):
            for c in range(W):
                if not m[r, c]:
                    continue
                nb = 0
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W and m[nr, nc]:
                        nb += 1
                if nb <= 2:
                    cand.append((r, c))
        if not cand:
            break
        r, c = random.choice(cand)
        to_remove = [(r, c)]
        if symmetric:
            mc = W - 1 - c
            if (r, mc) != (r, c):
                to_remove.append((r, mc))
        m2 = m.copy()
        for rr, cc in to_remove:
            m2[rr, cc] = False
        if not m2.any():
            continue
        if not _touches_all_bbox_sides(m2):
            continue
        _, n_components = label_mask(m2, connectivity=4)
        if int(n_components) == 1:
            m = m2
            removed += len(to_remove)
    return m


def _gen_shape(h: int, w: int, *, nonrect: bool, symmetric: bool = False) -> np.ndarray:
    m = _random_rect_shape(h, w)
    if nonrect and h * w >= 3:
        m = _mutate_shape(m, n_remove_max=max(1, (h * w) // 2), symmetric=symmetric)
    return m


def _mirror_shape(m: np.ndarray) -> np.ndarray:
    return m[:, ::-1].copy()


def _gen_dst_src_pair(h: int, w: int, *, symmetric: bool) -> tuple[np.ndarray, np.ndarray]:
    nonrect = random.random() < NONRECT_PROBABILITY
    return (
        _gen_shape(h, w, nonrect=nonrect, symmetric=symmetric),
        _gen_shape(h, w, nonrect=nonrect, symmetric=symmetric),
    )


def _add_centered_slot(slots: list[dict], cur_r: int, pair_id: int, row_h: int) -> None:
    w = random.choice(CENTERED_SLOT_WIDTHS)
    dst, src = _gen_dst_src_pair(row_h, w, symmetric=True)
    slots.append(
        {
            "r0": cur_r,
            "col_offset": -(w // 2),
            "shape_dst": dst,
            "shape_src": src,
            "pair_id": pair_id,
        }
    )


def _add_mirror_pair_slots(slots: list[dict], cur_r: int, pair_id: int, row_h: int) -> None:
    w = random.choice(PAIR_SLOT_WIDTHS)
    dst_L, src_L = _gen_dst_src_pair(row_h, w, symmetric=False)
    dst_R = _mirror_shape(dst_L)
    src_R = _mirror_shape(src_L)
    axis_gap = random.choice(PAIR_AXIS_GAP_CHOICES)
    if axis_gap % 2 == 0:
        axis_gap += 1
    half = axis_gap // 2
    slots.append(
        {
            "r0": cur_r,
            "col_offset": -(half + w),
            "shape_dst": dst_L,
            "shape_src": src_L,
            "pair_id": pair_id,
        }
    )
    slots.append(
        {
            "r0": cur_r,
            "col_offset": half + 1,
            "shape_dst": dst_R,
            "shape_src": src_R,
            "pair_id": pair_id,
        }
    )


def _add_trio_slots(slots: list[dict], cur_r: int, pair_id: int, row_h: int) -> None:
    w_side = random.choice(SIDE_SLOT_WIDTHS)
    w_mid = random.choice(MID_SLOT_WIDTHS)
    dst_L, src_L = _gen_dst_src_pair(row_h, w_side, symmetric=False)
    dst_R = _mirror_shape(dst_L)
    src_R = _mirror_shape(src_L)
    dst_M, src_M = _gen_dst_src_pair(row_h, w_mid, symmetric=True)
    mid_half = w_mid // 2
    slots.append(
        {
            "r0": cur_r,
            "col_offset": -(mid_half + w_side + TRIO_MID_SIDE_GAP),
            "shape_dst": dst_L,
            "shape_src": src_L,
            "pair_id": pair_id,
        }
    )
    slots.append(
        {
            "r0": cur_r,
            "col_offset": -mid_half,
            "shape_dst": dst_M,
            "shape_src": src_M,
            "pair_id": pair_id + 1,
        }
    )
    slots.append(
        {
            "r0": cur_r,
            "col_offset": mid_half + TRIO_MID_SIDE_GAP + 1,
            "shape_dst": dst_R,
            "shape_src": src_R,
            "pair_id": pair_id,
        }
    )


def _pick_layout_k(budget: int) -> int:
    if budget == 1:
        return 1
    if budget == 2:
        return random.choice(LAYOUT_K_CHOICES_SMALL)
    return random.choice(LAYOUT_K_CHOICES_FULL)


def _design_layout(n_target: int) -> list[dict]:
    slots: list[dict] = []
    cur_r = 0
    pair_id = 0
    budget = n_target
    while len(slots) < n_target and cur_r < MAX_LAYOUT_R_BUDGET and budget > 0:
        row_h = random.choice(ROW_HEIGHT_CHOICES)
        k = _pick_layout_k(budget)
        if k == 1:
            _add_centered_slot(slots, cur_r, pair_id, row_h)
            pair_id += 1
            budget -= 1
        elif k == 2:
            _add_mirror_pair_slots(slots, cur_r, pair_id, row_h)
            pair_id += 1
            budget -= 2
        else:
            _add_trio_slots(slots, cur_r, pair_id, row_h)
            pair_id += 2
            budget -= 3
        cur_r += row_h + random.choice(ROW_GAP_CHOICES)
    return slots


def _render_object(
    layout: list[dict], shapes: list[np.ndarray], colors: list[int], bg: int, outline_col: int, *, filled: bool
) -> tuple[np.ndarray | None, np.ndarray | None]:
    placed_slots = []
    for slot_idx, (slot, sh) in enumerate(zip(layout, shapes)):
        rr, cc = np.where(sh)
        if len(rr) == 0:
            continue
        placed_slots.append((slot_idx, slot["r0"] + rr, slot["col_offset"] + cc))
    if not placed_slots:
        return None, None

    all_rows = np.concatenate([p[1] for p in placed_slots])
    all_cols = np.concatenate([p[2] for p in placed_slots])
    r_min, r_max = int(all_rows.min()), int(all_rows.max())
    c_min, c_max = int(all_cols.min()), int(all_cols.max())

    pad = 2
    H = (r_max - r_min + 1) + 2 * pad
    W = (c_max - c_min + 1) + 2 * pad

    canvas = np.full((H, W), bg, dtype=int)
    fill_mask = np.zeros((H, W), dtype=bool)
    for slot_idx, rows, cols in placed_slots:
        rr = rows - r_min + pad
        cc = cols - c_min + pad
        canvas[rr, cc] = colors[slot_idx]
        fill_mask[rr, cc] = True

    outline_mask = np.zeros_like(fill_mask)
    outline_mask[1:, :] |= fill_mask[:-1, :]
    outline_mask[:-1, :] |= fill_mask[1:, :]
    outline_mask[:, 1:] |= fill_mask[:, :-1]
    outline_mask[:, :-1] |= fill_mask[:, 1:]
    outline_mask &= ~fill_mask
    canvas[outline_mask] = outline_col

    filled_grid = canvas.copy()
    if not filled:
        canvas[fill_mask] = bg

    b = bbox(canvas != bg)
    if b is None:
        return None, None
    cropped = crop_bbox(canvas, b)
    r0, c0, r1, c1 = b
    return cropped, filled_grid[r0 : r1 + 1, c0 : c1 + 1]


def _count_4conn_cavities(filled_grid: np.ndarray, bg: int, outline_col: int) -> int:
    mask = (filled_grid != bg) & (filled_grid != outline_col)
    _, n = label_mask(mask, connectivity=4)
    return int(n)


def _count_enclosed_bg_holes(empty_grid: np.ndarray, bg: int) -> int:
    mask = empty_grid == bg
    lab, n = label_mask(mask, connectivity=4)
    H, W = empty_grid.shape
    count = 0
    for cid in range(1, int(n) + 1):
        pts = np.argwhere(lab == int(cid))
        if pts.size == 0:
            continue
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        if bool(r0 > 0 and c0 > 0 and r1 < H - 1 and c1 < W - 1):
            count += 1
    return count


def _is_8connected(grid: np.ndarray, bg: int) -> bool:
    mask = grid != bg
    if not mask.any():
        return False
    _, n = label_mask(mask, connectivity=8)
    return int(n) == 1


def _size_in_spec(grid: np.ndarray) -> bool:
    h, w = grid.shape
    return OBJECT_H_RANGE[0] <= h <= OBJECT_H_RANGE[1] and OBJECT_W_RANGE[0] <= w <= OBJECT_W_RANGE[1]


def _assign_pair_colors(layout: list[dict], fill_pool: list[int]) -> list[int]:
    unique_pair_ids: list[int] = []
    for slot in layout:
        if slot["pair_id"] not in unique_pair_ids:
            unique_pair_ids.append(slot["pair_id"])
    n_pairs = len(unique_pair_ids)
    n_colors = len(fill_pool)

    if n_pairs >= n_colors:
        pair_colors = list(fill_pool) + [random.choice(fill_pool) for _ in range(n_pairs - n_colors)]
    else:
        pair_colors = list(fill_pool)[:n_pairs]
    random.shuffle(pair_colors)

    pair_color_map = dict(zip(unique_pair_ids, pair_colors))
    colors = [pair_color_map[slot["pair_id"]] for slot in layout]
    for i in range(len(colors)):
        if random.random() < COLOR_MISMATCH_RATE:
            colors[i] = random.choice(fill_pool)
    return colors


def _pick_bg_and_palette() -> tuple[int, list[int]]:
    bg = random.randint(0, 9)
    palette = [c for c in range(10) if c != bg]
    random.shuffle(palette)
    return bg, palette


def _slot_center2(slot: dict, shape: np.ndarray) -> tuple[int, int]:
    h, w = shape.shape
    return int(2 * slot["r0"] + h - 1), int(2 * slot["col_offset"] + w - 1)


def _shape_to_patch(shape: np.ndarray, color: int, bg: int) -> np.ndarray:
    patch = np.full(shape.shape, bg, dtype=int)
    patch[shape] = int(color)
    return patch


def _render_scatter_src(layout: list[dict], colors: list[int], bg: int) -> np.ndarray | None:
    ordered = []
    for slot_idx, slot in enumerate(layout):
        sh = slot["shape_src"]
        ordered.append(
            {
                "slot_idx": slot_idx,
                "shape": sh,
                "color": int(colors[slot_idx]),
                "center2": _slot_center2(slot, sh),
            }
        )
    ordered.sort(key=lambda d: (int(d["center2"][0]), int(d["center2"][1]), int(d["slot_idx"])))
    if not ordered:
        return None

    rows: list[list[dict]] = []
    for item in ordered:
        if not rows or item["center2"][0] != rows[-1][0]["center2"][0]:
            rows.append([item])
        else:
            rows[-1].append(item)

    col_gaps = [random.choice(SCATTER_COL_GAP_CHOICES) for _ in rows]
    row_gaps = [random.choice(SCATTER_ROW_GAP_CHOICES) for _ in range(max(0, len(rows) - 1))]
    row_heights = [max(int(it["shape"].shape[0]) for it in row) + 2 * SCATTER_INNER_PAD for row in rows]
    row_widths = []
    for row_idx, row in enumerate(rows):
        total = sum(int(it["shape"].shape[1]) + 2 * SCATTER_INNER_PAD for it in row)
        total += col_gaps[row_idx] * max(0, len(row) - 1)
        row_widths.append(total)
    H = sum(row_heights) + sum(row_gaps)
    W = max(row_widths)
    canvas = np.full((int(H), int(W)), int(bg), dtype=int)

    cur_r = 0
    for row_idx, row in enumerate(rows):
        row_h = row_heights[row_idx]
        widths = [int(it["shape"].shape[1]) + 2 * SCATTER_INNER_PAD for it in row]
        gap = col_gaps[row_idx]
        row_w = sum(widths) + gap * max(0, len(row) - 1)
        cur_c = (W - row_w) // 2
        for item in row:
            sh = item["shape"]
            patch = _shape_to_patch(sh, item["color"], bg)
            ph, pw = patch.shape
            rr = cur_r + (row_h - ph) // 2
            cc = cur_c + SCATTER_INNER_PAD
            paste(canvas, patch, int(rr), int(cc), transparent=bg)
            cur_c += pw + 2 * SCATTER_INNER_PAD + gap
        if row_idx + 1 < len(rows):
            cur_r += row_h + row_gaps[row_idx]
        else:
            cur_r += row_h

    b = bbox(canvas != bg)
    return crop_bbox(canvas, b) if b is not None else None


def _build_object_pair(layout: list[dict], colors: list[int], bg: int, outline_col: int) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    n_cavities = len(layout)
    dst_shapes = [slot["shape_dst"] for slot in layout]

    dst_cropped, dst_filled = _render_object(layout, dst_shapes, colors, bg, outline_col, filled=False)
    if dst_cropped is None or dst_filled is None:
        return None
    if not _is_8connected(dst_cropped, bg):
        return None
    if _count_enclosed_bg_holes(dst_cropped, bg) != n_cavities:
        return None
    if not _size_in_spec(dst_cropped):
        return None

    src_scatter = _render_scatter_src(layout, colors, bg)
    if src_scatter is None:
        return None

    src_mask = src_scatter != bg
    _, n_src = label_mask(src_mask, connectivity=8)
    if int(n_src) != n_cavities:
        return None

    # Expected output = tight-cropped filled dst.
    return src_scatter, dst_cropped, dst_filled


def _place_on_canvas(src: np.ndarray, dst: np.ndarray, bg: int) -> np.ndarray | None:
    canvas_h = random.randint(*CANVAS_SIZE_RANGE)
    canvas_w = random.randint(*CANVAS_SIZE_RANGE)
    canvas = np.full((canvas_h, canvas_w), bg, dtype=int)
    try:
        r, c = random_free_location(
            canvas, src, transparent=bg, border=PLACEMENT_BORDER, padding=PLACEMENT_PADDING
        )
    except ValueError:
        return None
    paste(canvas, src, r, c, transparent=bg)
    try:
        r, c = random_free_location(
            canvas, dst, transparent=bg, border=PLACEMENT_BORDER, padding=PLACEMENT_PADDING
        )
    except ValueError:
        return None
    paste(canvas, dst, r, c, transparent=bg)
    return canvas


def _gen_attempt() -> np.ndarray | None:
    bg, palette = _pick_bg_and_palette()
    outline_col = palette[0]

    layout = _design_layout(random.randint(*CAVITY_COUNT_RANGE))
    if len(layout) < CAVITY_COUNT_RANGE[0]:
        return None
    n_cavities = len(layout)

    fc_lo, fc_hi = FILL_COLOR_COUNT_RANGE
    max_cols = min(fc_hi, n_cavities, len(palette) - 1)
    if max_cols < fc_lo:
        return None
    n_fill_cols = random.randint(fc_lo, max_cols)
    fill_pool = palette[1 : 1 + n_fill_cols]
    colors = _assign_pair_colors(layout, fill_pool)

    built = _build_object_pair(layout, colors, bg, outline_col)
    if built is None:
        return None
    src, dst, expected = built

    canvas = _place_on_canvas(src, dst, bg)
    if canvas is None:
        return None

    try:
        actual = main(canvas)
    except Exception:
        return None
    if not isinstance(actual, np.ndarray):
        return None
    if actual.shape != expected.shape or not np.array_equal(actual, expected):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        try:
            g = _gen_attempt()
        except Exception:
            g = None
        if g is not None:
            return g
    raise RuntimeError(f"generate_input: failed after {MAX_GEN_ATTEMPTS} attempts")
