from __future__ import annotations

# Combo variant of 3a25b0d8 (index 15)
# combo_spec: D4/2+D9/0
# Name: 颜色互补填充+每空腔不同色
#
# Design analysis
# =============================================================================
# a) 具体化与行为追踪
# - 原题：src 物体的每个彩色子精灵按质心行列秩序匹配到 dst 的空腔，输出时直接把 src 颜色抄到 dst。
# - 融合后：几何匹配完全不变，但每个 src 空腔颜色都唯一；输出时不再直接抄色，而是把匹配色做固定互补
#   `c -> 9-c` 后再填到 dst。
# - 小例子：若按秩序匹配得到 src 颜色序列 `[2, 7, 3, 6]`，原题输出仍是 `[2, 7, 3, 6]`；
#   现在输出改成 `[7, 2, 6, 3]`。位置匹配证据来自几何布局，颜色变化证据来自固定互补映射。
#
# b) 复杂度校准
# - D9/0 要求每个空腔不同色，因此不能再让镜像洞共享颜色。生成器把 cavity 数限制到 4-8，并对每个 slot
#   无放回分配 src 颜色，避免颜色歧义。
# - 仅仅“全唯一”还不够明显；为了让人类能从少量样例中看出互补规律，生成器会优先整对启用若干互补对，
#   例如同一图里同时出现 `{2,7}` 和 `{3,6}`，这样输出中能直接看到成对翻转。
# - 若颜色太自由，可能撞到 bg 或 outline 造成洞消失或轮廓混淆。这里强制 bg 与 outline 占用同一互补对，
#   于是剩余 8 色全部都可以安全地作为“唯一空腔色”，并且它们的互补色也仍然是剩余 8 色。
#
# c) 隐式假设审查（逐段针对原 main）
# - “目标对象 = hole area 更大的那个”仍成立，因为 dst 仍是空洞框架，src 仍是彩色填满的框架。
# - `_outline_col()` 假设轮廓色是非背景众数。D9/0 之后每个填充色只出现一次，反而更容易保证轮廓占优；
#   生成器额外验证 outline 的计数严格大于任一 fill 色。
# - `_colored_components()` 原先只提取颜色，不关心唯一性；融合后 solver 不依赖唯一性才能运行，但 generator
#   必须保证唯一性，避免任务退化回“颜色可重复”的原题。
# - 最关键失效的旧前提是最终填色：原代码默认 `dst_hole_color = src_color`。D4/2 下必须改成
#   `dst_hole_color = 9 - src_color`。
# - 质心秩序匹配、候选旋转/翻折搜索、tight crop 都不依赖颜色是否唯一，也不依赖直接抄色，因此保留。
#
# d) 放置合法性
# - 生成器仍只放两个 8-connected 大对象，使用 `random_free_location` + padding 保证不重叠、不越界、不粘连。
# - src 与 dst 可在 bbox 上投影重叠，但透明背景下不会发生像素碰撞；这与原题 spec 一致。
#
# e) 视觉可观察性
# - 每个洞都是不同色，解题者能先稳定读出“哪一个 src 子精灵匹配哪一个 dst 空腔”。
# - 颜色变化不是杂乱置换，而是固定互补 `c <-> 9-c`；同一图内通常会出现 2-4 组完整互补对，输出中反复展示
#   “某色变成它的对色”的规律，便于从 2-4 组样例归纳。
#
# f) 可视化整洁度
# - 仍保持原题的单色轮廓 + 内部色块 + 大背景三层结构，没有新增标记或覆盖层。
# - 每个空腔唯一色虽然增加了颜色种类，但洞数量上限压到 8，且轮廓始终单色，因此画面不会变成噪声拼贴。
#
# g) 变化可推断性
# - 互补映射采用最简单、可任务级复用的 palette 对称规则 `c -> 9-c`，不是依赖黑色=0 之类不可见排序。
# - 生成器拒绝任何“互补输出与直接抄色输出完全相同”的样本；同时统计时应看到：
#   1) `main(input)` 与 direct-fill 基线恒不相同；
#   2) src 的唯一颜色数恒等于空腔数。
#
# 新增 solver 假设与 generator 保证
# - 假设 1：src/dst 仍是仅有的两个大对象。generator 只生成这两个对象。
# - 假设 2：互补色固定为 `9-c`。generator 与 solver 共用同一规则。
# - 假设 3：所有 src 空腔颜色彼此不同，且它们的互补色都不是 bg / outline。generator 通过“bg 与 outline 占同一
#   互补对”来保证这一点。

from collections import Counter
import random

import numpy as np

from arc_utils import bbox, crop_bbox, get_bg, label_mask, paste, random_free_location


COMPLEMENT_MAP = {c: 9 - c for c in range(10)}
PALETTE_PAIRS = [(0, 9), (1, 8), (2, 7), (3, 6), (4, 5)]


def _comp(color: int) -> int:
    return int(COMPLEMENT_MAP[int(color)])


def _components(mask: np.ndarray) -> list[dict]:
    lab, n = label_mask(mask, connectivity=8)
    comps = []
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
                "size": int(len(pts)),
                "cent": (float(pts[:, 0].mean()), float(pts[:, 1].mean())),
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
            }
        )
    comps.sort(key=lambda d: int(d["size"]), reverse=True)
    return comps


def _bbox_of_group(group: list[dict]) -> tuple[int, int, int, int]:
    r0 = min(int(d["bbox"][0]) for d in group)
    c0 = min(int(d["bbox"][1]) for d in group)
    r1 = max(int(d["bbox"][2]) for d in group)
    c1 = max(int(d["bbox"][3]) for d in group)
    return int(r0), int(c0), int(r1), int(c1)


def _extract_group_subgrid(grid: np.ndarray, bg: int, group: list[dict]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_of_group(group)
    sub = np.full((int(r1 - r0 + 1), int(c1 - c0 + 1)), int(bg), dtype=int)
    for comp in group:
        for rr, cc in comp["pts"]:
            sub[int(rr) - int(r0), int(cc) - int(c0)] = int(grid[int(rr), int(cc)])
    return sub


def _enclosed_bg_components(sub: np.ndarray, bg: int) -> list[dict]:
    mask = sub == int(bg)
    lab, n = label_mask(mask, connectivity=4)
    H, W = sub.shape
    holes = []
    for cid in range(1, int(n) + 1):
        pts = np.argwhere(lab == int(cid))
        if pts.size == 0:
            continue
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        if bool(r0 == 0 or c0 == 0 or r1 == int(H - 1) or c1 == int(W - 1)):
            continue
        cent = (int(round(float(pts[:, 0].mean()))), int(round(float(pts[:, 1].mean()))))
        holes.append({"pts": pts, "cent": cent, "size": int(len(pts))})
    return holes


def _outline_col(sub: np.ndarray, bg: int) -> int:
    cnt = Counter(map(int, sub.flatten()))
    opts = [(c, n) for c, n in cnt.items() if int(c) != int(bg)]
    return int(max(opts, key=lambda t: t[1])[0]) if opts else int(bg)


def _colored_components(sub: np.ndarray, bg: int) -> list[dict]:
    outline_col = _outline_col(sub, bg)
    mask = (sub != int(bg)) & (sub != int(outline_col))
    lab, n = label_mask(mask, connectivity=4)
    subs = []
    for cid in range(1, int(n) + 1):
        pts = np.argwhere(lab == int(cid))
        if pts.size == 0:
            continue
        color = Counter(map(int, sub[lab == int(cid)].flatten())).most_common(1)[0][0]
        cent = (int(round(float(pts[:, 0].mean()))), int(round(float(pts[:, 1].mean()))))
        subs.append({"cent": cent, "color": int(color), "size": int(len(pts))})
    return subs


def _rank_keys(pts: list[tuple[int, int]]) -> tuple[list[tuple[int, int]], dict[int, int], dict[int, int]]:
    rows = sorted(set(int(r) for r, _c in pts))
    cols = sorted(set(int(c) for _r, c in pts))
    row_rank = {r: i for i, r in enumerate(rows)}
    col_rank = {c: i for i, c in enumerate(cols)}
    keys = [(int(row_rank[r]), int(col_rank[c])) for r, c in pts]
    return keys, row_rank, col_rank


def _transform_point(p: tuple[int, int], H0: int, W0: int, k: int, flip: bool) -> tuple[int, int]:
    r, c = int(p[0]), int(p[1])
    if int(k) == 0:
        rr, cc = int(r), int(c)
        H2, W2 = int(H0), int(W0)
    elif int(k) == 1:
        rr, cc = int(W0 - 1 - c), int(r)
        H2, W2 = int(W0), int(H0)
    elif int(k) == 2:
        rr, cc = int(H0 - 1 - r), int(W0 - 1 - c)
        H2, W2 = int(H0), int(W0)
    else:
        rr, cc = int(c), int(H0 - 1 - r)
        H2, W2 = int(W0), int(H0)
    if bool(flip):
        cc = int(W2 - 1 - cc)
    return int(rr), int(cc)


def _crop_non_bg(sub: np.ndarray, bg: int) -> np.ndarray:
    mask = sub != int(bg)
    box = bbox(mask)
    if box is None:
        return sub.copy()
    return crop_bbox(sub, box)


def _extract_src_dst_objects(grid: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    comps = _components(g != int(bg))
    if len(comps) < 2:
        raise ValueError("expected at least two foreground objects")
    groups = [[comps[0]], [comps[1]]]
    objs = [_extract_group_subgrid(g, bg, group) for group in groups]
    holes_list = [_enclosed_bg_components(obj, bg) for obj in objs]
    hole_area = [sum(int(h["size"]) for h in holes) for holes in holes_list]
    dest_i = int(np.argmax(hole_area))
    src_i = int(1 - dest_i)
    return int(bg), objs[src_i].copy(), objs[dest_i].copy()


def _solve(grid: np.ndarray, *, use_complement: bool) -> np.ndarray:
    bg, src, dest = _extract_src_dst_objects(grid)
    dest_holes = _enclosed_bg_components(dest, bg)
    src_subs = _colored_components(src, bg)
    if not dest_holes or not src_subs:
        return _crop_non_bg(dest, bg)

    dest_pts = [hole["cent"] for hole in dest_holes]
    dest_keys, d_rr, d_cc = _rank_keys(dest_pts)

    best = None
    for k in range(4):
        for flip in [False, True]:
            tpts = [_transform_point(sub["cent"], int(src.shape[0]), int(src.shape[1]), int(k), bool(flip))
                    for sub in src_subs]
            tkeys, _s_rr, _s_cc = _rank_keys(tpts)
            if sorted(tkeys) == sorted(dest_keys):
                best = (int(k), bool(flip))
                break
        if best is not None:
            break
    if best is None:
        best = (0, False)
    k, flip = best

    dest_list = []
    for hole in dest_holes:
        key = (int(d_rr[int(hole["cent"][0])]), int(d_cc[int(hole["cent"][1])]))
        dest_list.append((key, hole["cent"], hole))
    dest_list.sort(key=lambda item: (item[0][0], item[0][1], item[1][0], item[1][1]))

    tpts_all = [_transform_point(sub["cent"], int(src.shape[0]), int(src.shape[1]), int(k), bool(flip))
                for sub in src_subs]
    _keys, s_rr, s_cc = _rank_keys(tpts_all)

    src_list = []
    for sub in src_subs:
        tp = _transform_point(sub["cent"], int(src.shape[0]), int(src.shape[1]), int(k), bool(flip))
        key = (int(s_rr[int(tp[0])]), int(s_cc[int(tp[1])]))
        src_list.append((key, tp, int(sub["color"])))
    src_list.sort(key=lambda item: (item[0][0], item[0][1], item[1][0], item[1][1]))

    for (_k1, _p1, color), (_k2, _p2, hole) in zip(src_list, dest_list):
        fill_color = _comp(color) if use_complement else int(color)
        for rr, cc in hole["pts"]:
            dest[int(rr), int(cc)] = int(fill_color)

    return _crop_non_bg(dest, bg)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the target object's holes using matched src subsprites, but paint each
    hole with the fixed palette complement `c -> 9-c` of the matched src color.
    """
    return _solve(grid, use_complement=True)


CANVAS_SIZE_RANGE = (22, 30)
OBJECT_H_RANGE = (7, 17)
OBJECT_W_RANGE = (7, 13)

# Unique-color fusion: every cavity gets its own src color, so cavity count must
# fit within the 8 colors left after reserving one complement pair for bg/outline.
CAVITY_COUNT_RANGE = (4, 8)

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

PLACEMENT_BORDER = 1
PLACEMENT_PADDING = 2

MAX_GEN_ATTEMPTS = 400
MAX_LAYOUT_R_BUDGET = 24


def _random_rect_shape(h: int, w: int) -> np.ndarray:
    return np.ones((h, w), dtype=bool)


def _mutate_shape(mask: np.ndarray, *, n_remove_max: int, symmetric: bool = False) -> np.ndarray:
    m = mask.copy()
    H, W = m.shape
    n_remove = random.randint(1, max(1, n_remove_max))
    removed = 0
    tried = 0
    while removed < n_remove and tried < MAX_MUTATE_ATTEMPTS:
        tried += 1
        candidates = []
        for r in range(H):
            for c in range(W):
                if not m[r, c]:
                    continue
                nb = 0
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W and m[nr, nc]:
                        nb += 1
                if nb <= 2:
                    candidates.append((r, c))
        if not candidates:
            break
        r, c = random.choice(candidates)
        to_remove = [(r, c)]
        if bool(symmetric):
            mc = W - 1 - c
            if (r, mc) != (r, c):
                to_remove.append((r, mc))
        m2 = m.copy()
        for rr, cc in to_remove:
            m2[rr, cc] = False
        if not m2.any():
            continue
        _, n_components = label_mask(m2, connectivity=4)
        if int(n_components) == 1:
            m = m2
            removed += len(to_remove)
    return m if m.any() else mask.copy()


def _gen_shape(h: int, w: int, *, nonrect: bool, symmetric: bool = False) -> np.ndarray:
    shape = _random_rect_shape(h, w)
    if nonrect and h * w >= 3:
        limit = max(1, (h * w) // 2)
        shape = _mutate_shape(shape, n_remove_max=limit, symmetric=symmetric)
    return shape


def _mirror_shape(mask: np.ndarray) -> np.ndarray:
    return mask[:, ::-1].copy()


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
    dst_l, src_l = _gen_dst_src_pair(row_h, w, symmetric=False)
    axis_gap = random.choice(PAIR_AXIS_GAP_CHOICES)
    if axis_gap % 2 == 0:
        axis_gap += 1
    half = axis_gap // 2
    slots.append(
        {
            "r0": cur_r,
            "col_offset": -(half + w),
            "shape_dst": dst_l,
            "shape_src": src_l,
            "pair_id": pair_id,
        }
    )
    slots.append(
        {
            "r0": cur_r,
            "col_offset": half + 1,
            "shape_dst": _mirror_shape(dst_l),
            "shape_src": _mirror_shape(src_l),
            "pair_id": pair_id,
        }
    )


def _add_trio_slots(slots: list[dict], cur_r: int, pair_id: int, row_h: int) -> None:
    w_side = random.choice(SIDE_SLOT_WIDTHS)
    w_mid = random.choice(MID_SLOT_WIDTHS)
    dst_l, src_l = _gen_dst_src_pair(row_h, w_side, symmetric=False)
    dst_m, src_m = _gen_dst_src_pair(row_h, w_mid, symmetric=True)
    mid_half = w_mid // 2
    slots.append(
        {
            "r0": cur_r,
            "col_offset": -(mid_half + w_side + TRIO_MID_SIDE_GAP),
            "shape_dst": dst_l,
            "shape_src": src_l,
            "pair_id": pair_id,
        }
    )
    slots.append(
        {
            "r0": cur_r,
            "col_offset": -mid_half,
            "shape_dst": dst_m,
            "shape_src": src_m,
            "pair_id": pair_id + 1,
        }
    )
    slots.append(
        {
            "r0": cur_r,
            "col_offset": mid_half + TRIO_MID_SIDE_GAP + 1,
            "shape_dst": _mirror_shape(dst_l),
            "shape_src": _mirror_shape(src_l),
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
    slots = []
    cur_r = 0
    pair_id = 0
    budget = int(n_target)
    while len(slots) < n_target and cur_r < MAX_LAYOUT_R_BUDGET and budget > 0:
        row_h = random.choice(ROW_HEIGHT_CHOICES)
        k = _pick_layout_k(budget)
        if int(k) == 1:
            _add_centered_slot(slots, cur_r, pair_id, row_h)
            pair_id += 1
            budget -= 1
        elif int(k) == 2:
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
    layout: list[dict],
    shapes: list[np.ndarray],
    colors: list[int],
    bg: int,
    outline_col: int,
    *,
    filled: bool,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    placed_slots = []
    for slot_idx, (slot, shape) in enumerate(zip(layout, shapes)):
        rr, cc = np.where(shape)
        if len(rr) == 0:
            continue
        placed_slots.append((slot_idx, slot["r0"] + rr, slot["col_offset"] + cc))
    if not placed_slots:
        return None, None

    all_rows = np.concatenate([item[1] for item in placed_slots])
    all_cols = np.concatenate([item[2] for item in placed_slots])
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
        canvas[rr, cc] = int(colors[slot_idx])
        fill_mask[rr, cc] = True

    outline_mask = np.zeros_like(fill_mask)
    outline_mask[1:, :] |= fill_mask[:-1, :]
    outline_mask[:-1, :] |= fill_mask[1:, :]
    outline_mask[:, 1:] |= fill_mask[:, :-1]
    outline_mask[:, :-1] |= fill_mask[:, 1:]
    outline_mask &= ~fill_mask
    canvas[outline_mask] = int(outline_col)

    filled_grid = canvas.copy()
    if not filled:
        canvas[fill_mask] = int(bg)

    box = bbox(canvas != int(bg))
    if box is None:
        return None, None
    cropped = crop_bbox(canvas, box)
    r0, c0, r1, c1 = box
    filled_cropped = filled_grid[r0:r1 + 1, c0:c1 + 1]
    return cropped, filled_cropped


def _count_4conn_cavities(filled_grid: np.ndarray, bg: int, outline_col: int) -> int:
    mask = (filled_grid != int(bg)) & (filled_grid != int(outline_col))
    _, n = label_mask(mask, connectivity=4)
    return int(n)


def _count_enclosed_bg_holes(empty_grid: np.ndarray, bg: int) -> int:
    mask = empty_grid == int(bg)
    lab, n = label_mask(mask, connectivity=4)
    H, W = empty_grid.shape
    count = 0
    for cid in range(1, int(n) + 1):
        pts = np.argwhere(lab == int(cid))
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        if int(r0) > 0 and int(c0) > 0 and int(r1) < H - 1 and int(c1) < W - 1:
            count += 1
    return int(count)


def _is_8connected(grid: np.ndarray, bg: int) -> bool:
    mask = grid != int(bg)
    if not mask.any():
        return False
    _, n = label_mask(mask, connectivity=8)
    return int(n) == 1


def _size_in_spec(grid: np.ndarray) -> bool:
    h, w = grid.shape
    return OBJECT_H_RANGE[0] <= h <= OBJECT_H_RANGE[1] and OBJECT_W_RANGE[0] <= w <= OBJECT_W_RANGE[1]


def _outline_dominates(grid: np.ndarray, bg: int, outline_col: int) -> bool:
    cnt = Counter(map(int, grid.flatten()))
    outline_n = int(cnt.get(int(outline_col), 0))
    for color, n in cnt.items():
        if int(color) in (int(bg), int(outline_col)):
            continue
        if int(n) >= int(outline_n):
            return False
    return True


def _pick_bg_outline_pair() -> tuple[int, int]:
    a, b = random.choice(PALETTE_PAIRS)
    if random.random() < 0.5:
        return int(a), int(b)
    return int(b), int(a)


def _assign_unique_src_colors(n_cavities: int, bg: int, outline_col: int) -> list[int]:
    valid_pairs = [pair for pair in PALETTE_PAIRS if int(bg) not in pair and int(outline_col) not in pair]
    random.shuffle(valid_pairs)

    src_colors: list[int] = []
    n_full_pairs = min(len(valid_pairs), int(n_cavities) // 2)
    for a, b in valid_pairs[:n_full_pairs]:
        src_colors.extend([int(a), int(b)])

    remaining = int(n_cavities) - len(src_colors)
    if remaining > 0:
        leftovers = []
        for a, b in valid_pairs[n_full_pairs:]:
            leftovers.extend([int(a), int(b)])
        random.shuffle(leftovers)
        src_colors.extend(leftovers[:remaining])

    if len(src_colors) != int(n_cavities) or len(set(src_colors)) != int(n_cavities):
        raise ValueError("failed to assign enough unique source colors")
    random.shuffle(src_colors)
    return [int(c) for c in src_colors]


def _build_object_pair(
    layout: list[dict],
    src_colors: list[int],
    bg: int,
    outline_col: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    n_cavities = len(layout)
    dst_shapes = [slot["shape_dst"] for slot in layout]
    src_shapes = [slot["shape_src"] for slot in layout]
    out_colors = [_comp(color) for color in src_colors]

    dst_cropped, dst_filled = _render_object(layout, dst_shapes, out_colors, bg, outline_col, filled=False)
    if dst_cropped is None or dst_filled is None:
        return None
    if not _is_8connected(dst_cropped, bg):
        return None
    if _count_enclosed_bg_holes(dst_cropped, bg) != int(n_cavities):
        return None
    if not _size_in_spec(dst_cropped):
        return None

    src_cropped, _ = _render_object(layout, src_shapes, src_colors, bg, outline_col, filled=True)
    if src_cropped is None:
        return None
    if not _is_8connected(src_cropped, bg):
        return None
    if _count_4conn_cavities(src_cropped, bg, outline_col) != int(n_cavities):
        return None
    if not _size_in_spec(src_cropped):
        return None
    if not _outline_dominates(src_cropped, bg, outline_col):
        return None

    seen_colors = {int(sub["color"]) for sub in _colored_components(src_cropped, bg)}
    if len(seen_colors) != int(n_cavities):
        return None
    return src_cropped, dst_cropped, dst_filled


def _place_on_canvas(src: np.ndarray, dst: np.ndarray, bg: int) -> np.ndarray | None:
    H = random.randint(*CANVAS_SIZE_RANGE)
    W = random.randint(*CANVAS_SIZE_RANGE)
    canvas = np.full((H, W), bg, dtype=int)
    try:
        r, c = random_free_location(
            canvas,
            src,
            transparent=bg,
            border=PLACEMENT_BORDER,
            padding=PLACEMENT_PADDING,
        )
    except ValueError:
        return None
    paste(canvas, src, r, c, transparent=bg)
    try:
        r, c = random_free_location(
            canvas,
            dst,
            transparent=bg,
            border=PLACEMENT_BORDER,
            padding=PLACEMENT_PADDING,
        )
    except ValueError:
        return None
    paste(canvas, dst, r, c, transparent=bg)
    return canvas


def _full_pair_count(colors: list[int]) -> int:
    color_set = set(map(int, colors))
    return sum(1 for a, b in PALETTE_PAIRS if int(a) in color_set and int(b) in color_set)


def _gen_attempt() -> np.ndarray | None:
    bg, outline_col = _pick_bg_outline_pair()
    if _comp(bg) != int(outline_col):
        return None

    layout = _design_layout(random.randint(*CAVITY_COUNT_RANGE))
    if len(layout) < CAVITY_COUNT_RANGE[0] or len(layout) > CAVITY_COUNT_RANGE[1]:
        return None
    n_cavities = len(layout)

    try:
        src_colors = _assign_unique_src_colors(n_cavities, bg, outline_col)
    except ValueError:
        return None
    if _full_pair_count(src_colors) < 2:
        return None

    built = _build_object_pair(layout, src_colors, bg, outline_col)
    if built is None:
        return None
    src, dst, expected = built

    canvas = _place_on_canvas(src, dst, bg)
    if canvas is None:
        return None

    try:
        actual = main(canvas)
        direct = _solve(canvas, use_complement=False)
    except Exception:
        return None
    if not isinstance(actual, np.ndarray) or not isinstance(direct, np.ndarray):
        return None
    if actual.shape != expected.shape or not np.array_equal(actual, expected):
        return None
    if actual.shape == direct.shape and np.array_equal(actual, direct):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        try:
            grid = _gen_attempt()
        except Exception:
            grid = None
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {MAX_GEN_ATTEMPTS} attempts")
