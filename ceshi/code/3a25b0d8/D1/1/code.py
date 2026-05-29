# Variant of 3a25b0d8
# Dimension: D1 — 匹配条件：排列匹配方式 (match_cond)
# Change: arrangement_match — src 子精灵与 dst 洞按质心的行列秩序(row_rank, col_rank)匹配
#         → shape_match — 改为按子精灵/洞的形状匹配（允许旋转/翻转后视为同形）

from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np

from arc_utils import bbox, crop_bbox, get_bg, label_mask, paste, random_free_location


"""
design_analysis:
  variant_goal:
    keep:
      - 输入仍是两个分离的 8-connected 物体：一个 src（彩色子精灵填满腔体），一个 dst（同 outline 色、洞为空）
      - 输出仍是把 dst 的洞填色后紧裁
      - 背景、outline、随机放置、tight crop 等外围行为保持一致
    change:
      - 原题匹配依据是 "洞/子精灵的质心行列秩序"
      - 本变体改成 "洞形状 与 子精灵形状 完全同构（允许旋转/翻转）"

  a_concretization_and_trace:
    pixel_level_structure:
      - 每个 cavity/子精灵都是一个 3x3 槽位中的非矩形 4-connected 小形状
      - src 与 dst 使用同一批形状原型，但这些原型会被打乱到不同槽位上
      - 同一原型在 src/dst 中还会各自再随机旋转/翻转，因此肉眼可见“形状相同但方向可能不同”
    small_trace:
      - 原逻辑：dst 左上洞、右上洞、下方洞按 (row_rank,col_rank) 排序；src 里同序位置的三个子精灵颜色依次填入
      - 新逻辑：dst 左上洞是 L 形，就必须去 src 里找同一个 L 形（即便它在右下且旋转了 90 度），把那个颜色填进左上洞
      - 差异：位置序不再提供答案；真正决定配对的是轮廓本身

  b_complexity_calibration:
    - 若 src/dst 仍把同一形状放在同一槽位，样本会退化回原题；因此生成器必须显式置换形状到不同位置
    - 若允许重复形状太多，匹配会歧义；因此本变体默认每个 cavity 使用唯一的 canonical shape
    - 若全部颜色都重复，位置错配在输出上可能不可见；因此生成器会拒绝“shape-match 输出 == arrangement-match 输出”的样本

  c_implicit_assumption_audit_on_original_main:
    original_dependencies_broken_by_variant:
      - _rank_keys(...) 假设“洞集合”和“子精灵集合”在某个全局旋转/翻转后共享同一 row/col rank 多重集
      - best=(k,flip) 的搜索假设只需找一个“整体几何变换”即可恢复配对
      - 后续 src_list / dest_list 按 rank 排序后一一 zip，假设位置排序就是语义匹配
    why_invalid_now:
      - 变体中 src 与 dst 的相同形状被故意打乱到不同槽位，整体 rank 一般不再一致
      - 同形状的对应关系不依赖整个 src 物体的统一变换，而依赖每个 cavity 的局部轮廓
    replacement:
      - 对每个洞/子精灵抽取相对 bbox 内的布尔掩码
      - 对掩码取 8 个二面体变换（4 旋转 x 是否水平翻转）的最小 canonical signature
      - 用 signature 做一一匹配，再把 src 颜色写入对应 dst 洞

  d_placement_legality:
    - src/dst 仍通过 random_free_location 放到 22-30 的画布上
    - 使用 border=1, padding=2，保证不越界、不重叠、两个物体可区分
    - solver 仍只需要从全图中取最大的两个非背景 8-connected 物体

  e_visual_observability:
    - 人类能直接看到：src 中出现若干不同轮廓的小彩色块，dst 中出现同一批轮廓的空洞，但位置被打乱
    - 样例里同一种洞形状与 src 中同形状块的颜色一致，因此 2-4 个样例就足以暴露“按形状找对应”的规则
    - 禁止使用颜色编号、坐标数值等不可见信息参与匹配

  f_visual_cleanliness:
    - 仍采用规则槽位网格，洞之间保持 1-cell 间隔；形状虽变化，但整体布局整洁
    - 只存在两层显著视觉元素：outline 骨架 和 cavity/子精灵；不会叠加额外标记

  g_color_inference:
    - 颜色不是再映射，而是“沿 shape identity 搬运”
    - 颜色种类限制在 3-7，并用 shape 作为稳定线索支撑颜色归属

  h_generator_guardrails:
    - 仅采样 canonical-unique 的 3x3 非矩形连通形状
    - 仅接受 src/dst 共享完全相同 shape multiset 的样本
    - 仅接受 shape-match 输出 与 arrangement-match 输出 不同的样本，防止退化

  i_new_solver_assumptions_and_generator_support:
    solver_assumptions:
      - 恰有两个主要非背景物体
      - 每个 src 子精灵与某个 dst 洞在 canonical shape 上唯一对应
      - outline 色可由“非背景中面积最大的颜色”识别
    generator_support:
      - 通过唯一 canonical shape 保证对应唯一
      - 通过统一 outline 渲染方式让 outline 面积稳定偏大
      - 通过 oracle check main(input)==expected 和额外的退化检测共同保证样本有效
"""


def _trim_bool(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return np.zeros((0, 0), dtype=bool)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _mask_signature(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    mask = np.array(mask, dtype=bool)
    if not mask.any():
        return tuple()
    cands: list[tuple[tuple[int, ...], ...]] = []
    for k in range(4):
        rot = np.rot90(mask, k=k)
        for t in (rot, np.fliplr(rot)):
            tt = _trim_bool(t)
            cands.append(tuple(tuple(int(v) for v in row) for row in tt.astype(int)))
    return min(cands)


def _tight_crop_non_bg(grid: np.ndarray, bg: int) -> np.ndarray:
    m = grid != int(bg)
    pts = np.argwhere(m)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each dst hole with the color of the src subsprite having the same shape
    up to rotation / flip, then tight-crop the filled dst object.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

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
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                }
            )
        comps.sort(key=lambda d: int(d["size"]), reverse=True)
        return comps

    def _extract_two_objects() -> list[list[dict]]:
        comps = _components(g != int(bg))
        if len(comps) < 2:
            return [comps]
        return [[comps[0]], [comps[1]]]

    def _bbox_of(group: list[dict]) -> tuple[int, int, int, int]:
        r0 = min(int(d["bbox"][0]) for d in group)
        c0 = min(int(d["bbox"][1]) for d in group)
        r1 = max(int(d["bbox"][2]) for d in group)
        c1 = max(int(d["bbox"][3]) for d in group)
        return int(r0), int(c0), int(r1), int(c1)

    def _extract_group_subgrid(group: list[dict]) -> np.ndarray:
        r0, c0, r1, c1 = _bbox_of(group)
        sub = np.full((int(r1 - r0 + 1), int(c1 - c0 + 1)), int(bg), dtype=int)
        for d in group:
            for rr, cc in d["pts"]:
                sub[int(rr) - int(r0), int(cc) - int(c0)] = int(g[int(rr), int(cc)])
        return sub

    def _shape_record_from_pts(pts: np.ndarray) -> dict:
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        mask = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
        rel = pts - np.array([[int(r0), int(c0)]], dtype=int)
        mask[rel[:, 0], rel[:, 1]] = True
        cent = (int(round(float(pts[:, 0].mean()))), int(round(float(pts[:, 1].mean()))))
        return {
            "pts": pts,
            "cent": cent,
            "size": int(len(pts)),
            "sig": _mask_signature(mask),
        }

    def _enclosed_bg_components(sub: np.ndarray) -> list[dict]:
        m = sub == int(bg)
        lab, n = label_mask(m, connectivity=4)
        hs, ws = sub.shape
        holes = []
        for cid in range(1, int(n) + 1):
            pts = np.argwhere(lab == int(cid))
            if pts.size == 0:
                continue
            r0, c0 = pts.min(axis=0)
            r1, c1 = pts.max(axis=0)
            if bool(r0 == 0 or c0 == 0 or r1 == int(hs - 1) or c1 == int(ws - 1)):
                continue
            holes.append(_shape_record_from_pts(pts))
        return holes

    def _outline_col(sub: np.ndarray) -> int:
        cnt = Counter(map(int, sub.flatten()))
        opts = [(c, n) for c, n in cnt.items() if int(c) != int(bg)]
        return int(max(opts, key=lambda t: t[1])[0]) if opts else int(bg)

    def _colored_components(sub: np.ndarray) -> list[dict]:
        outc = _outline_col(sub)
        m = (sub != int(bg)) & (sub != int(outc))
        lab, n = label_mask(m, connectivity=4)
        subs = []
        for cid in range(1, int(n) + 1):
            pts = np.argwhere(lab == int(cid))
            if pts.size == 0:
                continue
            rec = _shape_record_from_pts(pts)
            rec["color"] = int(Counter(map(int, sub[lab == int(cid)].flatten())).most_common(1)[0][0])
            subs.append(rec)
        return subs

    groups = _extract_two_objects()
    assert len(groups) >= 2
    objs = [_extract_group_subgrid(grp) for grp in groups]

    holes_list = [_enclosed_bg_components(o) for o in objs]
    hole_area = [sum(int(h["size"]) for h in hh) for hh in holes_list]
    dest_i = int(np.argmax(hole_area))
    src_i = int(1 - dest_i)
    dest = objs[dest_i].copy()
    src = objs[src_i].copy()

    dest_holes = _enclosed_bg_components(dest)
    src_subs = _colored_components(src)
    if not dest_holes or not src_subs:
        return _tight_crop_non_bg(dest, bg)

    src_by_sig: dict[tuple[tuple[int, ...], ...], list[dict]] = defaultdict(list)
    for s in src_subs:
        src_by_sig[s["sig"]].append(s)

    dest_by_sig: dict[tuple[tuple[int, ...], ...], list[dict]] = defaultdict(list)
    for h in dest_holes:
        dest_by_sig[h["sig"]].append(h)

    if set(src_by_sig) != set(dest_by_sig):
        raise ValueError("shape multiset mismatch between src subsprites and dst holes")

    for sig, holes in dest_by_sig.items():
        subs = src_by_sig.get(sig, [])
        if len(subs) != len(holes):
            raise ValueError("shape multiplicity mismatch")
        holes.sort(key=lambda d: (int(d["cent"][0]), int(d["cent"][1]), int(d["size"])))
        subs.sort(key=lambda d: (int(d["cent"][0]), int(d["cent"][1]), int(d["color"])))
        for sub_rec, hole_rec in zip(subs, holes):
            for rr, cc in hole_rec["pts"]:
                dest[int(rr), int(cc)] = int(sub_rec["color"])

    return _tight_crop_non_bg(dest, bg)


CANVAS_SIZE_RANGE = (22, 30)
OBJECT_H_RANGE = (7, 17)
OBJECT_W_RANGE = (7, 13)
CAVITY_COUNT_RANGE = (4, 8)
FILL_COLOR_COUNT_RANGE = (3, 7)
PLACEMENT_BORDER = 1
PLACEMENT_PADDING = 2
MAX_GEN_ATTEMPTS = 400

SLOT_SIZE = 3
SLOT_STEP = 4
ROW_PATTERNS = [
    (2, 2),
    (3, 2),
    (2, 3),
    (3, 3),
    (1, 2, 2),
    (2, 1, 2),
    (1, 3, 2),
    (2, 3, 2),
    (3, 1, 3),
    (3, 2, 3),
]
COUNT_TO_COLS = {
    1: [4],
    2: [2, 6],
    3: [0, 4, 8],
}


def _is_connected(mask: np.ndarray) -> bool:
    if not mask.any():
        return False
    _, n = label_mask(mask, connectivity=4)
    return int(n) == 1


def _is_rect(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return bool(mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].all())


def _build_shape_library() -> list[np.ndarray]:
    reps: dict[tuple[tuple[int, ...], ...], np.ndarray] = {}
    for bits in range(1, 1 << (SLOT_SIZE * SLOT_SIZE)):
        arr = np.array([(bits >> i) & 1 for i in range(SLOT_SIZE * SLOT_SIZE)], dtype=bool).reshape(SLOT_SIZE, SLOT_SIZE)
        area = int(arr.sum())
        if area < 4 or area > 7:
            continue
        pts = np.argwhere(arr)
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        if (int(r0), int(c0), int(r1), int(c1)) != (0, 0, SLOT_SIZE - 1, SLOT_SIZE - 1):
            continue
        if not _is_connected(arr):
            continue
        if _is_rect(arr):
            continue
        reps.setdefault(_mask_signature(arr), arr.astype(bool))
    lib = list(reps.values())
    lib.sort(key=lambda m: (int(m.sum()), tuple(m.astype(int).flatten().tolist())))
    return lib


SHAPE_LIBRARY = _build_shape_library()


def _random_transform(mask: np.ndarray) -> np.ndarray:
    out = np.rot90(mask, k=random.randint(0, 3))
    if random.random() < 0.5:
        out = np.fliplr(out)
    return out.copy()


def _derangement(n: int) -> list[int]:
    idx = list(range(n))
    if n <= 1:
        return idx
    for _ in range(200):
        cand = idx[:]
        random.shuffle(cand)
        if all(cand[i] != i for i in range(n)):
            return cand
    return idx[1:] + idx[:1]


def _pick_bg_and_palette() -> tuple[int, list[int]]:
    bg = random.randint(0, 9)
    palette = [c for c in range(10) if c != bg]
    random.shuffle(palette)
    return bg, palette


def _slot_layout(pattern: tuple[int, ...]) -> list[dict]:
    layout = []
    for row_idx, k in enumerate(pattern):
        r0 = row_idx * SLOT_STEP
        for c0 in COUNT_TO_COLS[int(k)]:
            layout.append({"r0": int(r0), "c0": int(c0)})
    return layout


def _render_object(
    layout: list[dict],
    shapes: list[np.ndarray],
    colors: list[int],
    bg: int,
    outline_col: int,
    *,
    filled: bool,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    placed = []
    for slot, sh, color in zip(layout, shapes, colors):
        rr, cc = np.where(sh)
        if len(rr) == 0:
            continue
        placed.append((slot, rr, cc, int(color)))
    if not placed:
        return None, None

    all_rows = np.concatenate([slot["r0"] + rr for slot, rr, _cc, _color in placed])
    all_cols = np.concatenate([slot["c0"] + cc for slot, _rr, cc, _color in placed])
    r_min, r_max = int(all_rows.min()), int(all_rows.max())
    c_min, c_max = int(all_cols.min()), int(all_cols.max())

    pad = 2
    h = (r_max - r_min + 1) + 2 * pad
    w = (c_max - c_min + 1) + 2 * pad
    canvas = np.full((h, w), int(bg), dtype=int)
    fill_mask = np.zeros((h, w), dtype=bool)

    for slot, rr, cc, color in placed:
        pr = slot["r0"] + rr - r_min + pad
        pc = slot["c0"] + cc - c_min + pad
        canvas[pr, pc] = int(color)
        fill_mask[pr, pc] = True

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

    b = bbox(canvas != int(bg))
    if b is None:
        return None, None
    cropped = crop_bbox(canvas, b)
    r0b, c0b, r1b, c1b = b
    filled_cropped = filled_grid[int(r0b) : int(r1b) + 1, int(c0b) : int(c1b) + 1]
    return cropped, filled_cropped


def _count_4conn_cavities(filled_grid: np.ndarray, bg: int, outline_col: int) -> int:
    mask = (filled_grid != int(bg)) & (filled_grid != int(outline_col))
    _, n = label_mask(mask, connectivity=4)
    return int(n)


def _count_enclosed_bg_holes(empty_grid: np.ndarray, bg: int) -> int:
    mask = empty_grid == int(bg)
    lab, n = label_mask(mask, connectivity=4)
    h, w = empty_grid.shape
    count = 0
    for cid in range(1, int(n) + 1):
        pts = np.argwhere(lab == int(cid))
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        if r0 > 0 and c0 > 0 and r1 < h - 1 and c1 < w - 1:
            count += 1
    return count


def _is_8connected(grid: np.ndarray, bg: int) -> bool:
    mask = grid != int(bg)
    if not mask.any():
        return False
    _, n = label_mask(mask, connectivity=8)
    return int(n) == 1


def _size_in_spec(grid: np.ndarray) -> bool:
    h, w = grid.shape
    return OBJECT_H_RANGE[0] <= h <= OBJECT_H_RANGE[1] and OBJECT_W_RANGE[0] <= w <= OBJECT_W_RANGE[1]


def _assign_shape_colors(n_shapes: int, fill_pool: list[int]) -> list[int]:
    n_fill = min(len(fill_pool), max(FILL_COLOR_COUNT_RANGE[0], min(FILL_COLOR_COUNT_RANGE[1], n_shapes)))
    picked = fill_pool[:n_fill]
    colors = picked[:]
    while len(colors) < n_shapes:
        colors.append(random.choice(picked))
    random.shuffle(colors)
    return colors[:n_shapes]


def _build_object_pair(
    layout: list[dict],
    prototypes: list[np.ndarray],
    colors_by_shape: list[int],
    bg: int,
    outline_col: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    n = len(layout)
    perm = _derangement(n)

    dst_shapes = [_random_transform(proto) for proto in prototypes]
    src_shapes = [_random_transform(prototypes[int(perm[i])]) for i in range(n)]
    src_colors = [int(colors_by_shape[int(perm[i])]) for i in range(n)]

    dst_cropped, dst_filled = _render_object(layout, dst_shapes, colors_by_shape, bg, outline_col, filled=False)
    if dst_cropped is None or dst_filled is None:
        return None
    if not _is_8connected(dst_cropped, bg):
        return None
    if _count_enclosed_bg_holes(dst_cropped, bg) != n:
        return None
    if not _size_in_spec(dst_cropped):
        return None

    src_cropped, _src_filled = _render_object(layout, src_shapes, src_colors, bg, outline_col, filled=True)
    if src_cropped is None:
        return None
    if not _is_8connected(src_cropped, bg):
        return None
    if _count_4conn_cavities(src_cropped, bg, outline_col) != n:
        return None
    if not _size_in_spec(src_cropped):
        return None

    _rank_dst, rank_filled = _render_object(layout, dst_shapes, src_colors, bg, outline_col, filled=True)
    if rank_filled is None:
        return None
    if np.array_equal(rank_filled, dst_filled):
        return None

    return src_cropped, dst_cropped, dst_filled


def _place_on_canvas(src: np.ndarray, dst: np.ndarray, bg: int) -> np.ndarray | None:
    canvas_h = random.randint(*CANVAS_SIZE_RANGE)
    canvas_w = random.randint(*CANVAS_SIZE_RANGE)
    canvas = np.full((canvas_h, canvas_w), int(bg), dtype=int)
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
    paste(canvas, src, int(r), int(c), transparent=bg)
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
    paste(canvas, dst, int(r), int(c), transparent=bg)
    return canvas


def _gen_attempt() -> np.ndarray | None:
    bg, palette = _pick_bg_and_palette()
    outline_col = int(palette[0])

    pattern_choices = [p for p in ROW_PATTERNS if CAVITY_COUNT_RANGE[0] <= sum(p) <= CAVITY_COUNT_RANGE[1]]
    pattern = random.choice(pattern_choices)
    layout = _slot_layout(pattern)
    n = len(layout)

    if n > len(SHAPE_LIBRARY):
        return None
    prototypes = [m.copy() for m in random.sample(SHAPE_LIBRARY, n)]

    max_fill_cols = min(FILL_COLOR_COUNT_RANGE[1], n, len(palette) - 1)
    if max_fill_cols < FILL_COLOR_COUNT_RANGE[0]:
        return None
    n_fill_cols = random.randint(FILL_COLOR_COUNT_RANGE[0], max_fill_cols)
    fill_pool = palette[1 : 1 + n_fill_cols]
    colors_by_shape = _assign_shape_colors(n, fill_pool)

    built = _build_object_pair(layout, prototypes, colors_by_shape, bg, outline_col)
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
