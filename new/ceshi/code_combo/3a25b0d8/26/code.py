# Combo variant of 3a25b0d8 (index 26)
# combo_spec: D1/1+D7/2+D8/2
# Name: 形状匹配+全非矩形+无对称

from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np

from arc_utils import bbox, crop_bbox, get_bg, label_mask, paste, random_free_location


"""
design_analysis:
  variant_goal:
    keep:
      - 输入仍由两个分离的 8-connected 物体组成：src 为彩色子精灵集合，dst 为同 outline 色的空洞框架
      - 输出仍是把 dst 的洞填上对应颜色并 tight-crop
    change:
      - D1/1: 配对从 "质心行列秩序" 改为 "局部形状 canonical signature 匹配"
      - D7/2: 所有 cavity 都必须是非矩形 4-connected 形状
      - D8/2: src / dst 不再做任何 LR 对称骨架设计，并显式拒绝具有非平凡对称性的对象

  a_concretization_and_trace:
    pixel_view:
      - 物体由若干 3x3 槽位中的小型 irregular cavity 组成；每个 cavity 占满自己的 bbox，但内部不是实心矩形
      - dst 在这些槽位上只保留 outline，src 用同一批 shape identity 的彩色子精灵填满洞
      - src 与 dst 共享 shape multiset，但 src 会把这些 shape 打乱到不同槽位，因此位置不再携带答案
    small_trace:
      - 原题：dst 左上洞与 src 左上子精灵通过 row_rank/col_rank 对齐
      - 融合后：dst 左上洞如果是某个缺角 staircase 形，只能去 src 中找同 canonical 形状的子精灵；它可能被放到右下
      - 差异：位置秩序失效，唯一稳定线索变成局部轮廓；同时没有左右对称可借用，且所有洞都不是矩形

  b_complexity_calibration:
    - 若 src/dst 形状不打乱，样本会退化为按位置抄颜色；因此生成器强制使用 derangement
    - 若允许矩形洞，解题者可先排除简单洞；因此 shape library 只保留 full-bbox 非矩形连通形
    - 若对象偶然对称，人类可利用镜像捷径；因此渲染后显式拒绝任何非平凡 D4 对称
    - 若颜色重复太多，shape-match 与 arrangement-match 可能输出相同；因此保留 3-7 色并额外拒绝两种输出相同的样本

  c_implicit_assumption_audit_on_original_main:
    broken_assumptions:
      - 原 _rank_keys / _transform_point / best=(k,flip) 假设存在一个整体几何变换使 src 位置秩序与 dst 一致
      - 原 zip(src_list, dest_list) 假设排序后的第 i 个子精灵就是第 i 个洞的语义对应
      - 原生成器默认 LR 对称布局，很多 row/col rank 结构来自镜像构造
    after_fusion:
      - 同形状可以被故意放到任意别的槽位，整体 rank multiset 不能再决定配对
      - 无对称布局意味着不能依赖中心轴、镜像 pair_id、对称 gap 等生成约束
    replacement:
      - solver 对每个洞/子精灵提取相对 bbox 内的布尔 mask
      - 对 mask 取 8 个二面体变换的最小 canonical signature 作为 shape id
      - 按 signature 分组匹配并填色；仅把 centroid 排序留作无效输入时的保底 fallback

  d_placement_legality:
    - src / dst 仍用 random_free_location 放到 22-30 的背景画布中
    - 使用 border=1, padding=2，保证不越界、互不碰撞，且视觉上能区分两个对象

  e_visual_observability:
    - 人类能直接看见：dst 的每个空洞都是 irregular 小轮廓，src 中存在同轮廓彩色块，但位置被打乱
    - 多个样例里都能观察到 "按形状找颜色" 而不是 "按位置抄颜色"
    - 对称结构被移除后，解题者不能靠左右镜像推断配对

  f_visual_cleanliness:
    - 仍使用规则行列槽位，避免自由散点导致难以分辨
    - 每个洞只携带一种视觉信息：outline 包围的 irregular shape；没有额外标记层叠

  g_color_inference:
    - 颜色映射不是数值规则，而是沿 shape identity 搬运
    - distinct fill colors 控制在 3-7，并通过 derangement + 非退化检测确保 shape 线索真的必要

  i_new_solver_assumptions_and_generator_support:
    solver_assumptions:
      - 画面中恰有两个主要非背景 8-connected 物体
      - dst 的洞 shape multiset 与 src 的彩色子精灵 shape multiset 一致
      - outline 色是非背景中面积最大的单色
    generator_support:
      - 只采样 canonical-unique 的非矩形连通 shape
      - 渲染后检查 outline 推断、洞数量、8-connectivity、尺寸、无对称、非退化
      - sample_metrics() 暴露量化指标，便于验证该融合行为没有退化回原题
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
        for arr in (rot, np.fliplr(rot)):
            tt = _trim_bool(arr)
            cands.append(tuple(tuple(int(v) for v in row) for row in tt.astype(int)))
    return min(cands)


def _tight_crop_non_bg(grid: np.ndarray, bg: int) -> np.ndarray:
    pts = np.argwhere(grid != int(bg))
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _shape_record_from_pts(pts: np.ndarray) -> dict:
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    mask = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
    rel = pts - np.array([[int(r0), int(c0)]], dtype=int)
    mask[rel[:, 0], rel[:, 1]] = True
    return {
        "pts": pts,
        "cent": (float(pts[:, 0].mean()), float(pts[:, 1].mean())),
        "size": int(len(pts)),
        "mask": mask,
        "sig": _mask_signature(mask),
    }


def _components(mask: np.ndarray, *, connectivity: int) -> list[dict]:
    lab, n = label_mask(mask, connectivity=connectivity)
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


def _extract_two_object_subgrids(grid: np.ndarray, bg: int) -> list[np.ndarray]:
    comps = _components(grid != int(bg), connectivity=8)
    out = []
    for comp in comps[:2]:
        r0, c0, r1, c1 = comp["bbox"]
        sub = np.full((int(r1 - r0 + 1), int(c1 - c0 + 1)), int(bg), dtype=int)
        for rr, cc in comp["pts"]:
            sub[int(rr) - int(r0), int(cc) - int(c0)] = int(grid[int(rr), int(cc)])
        out.append(sub)
    return out


def _enclosed_bg_components(sub: np.ndarray, bg: int) -> list[dict]:
    lab, n = label_mask(sub == int(bg), connectivity=4)
    h, w = sub.shape
    holes = []
    for cid in range(1, int(n) + 1):
        pts = np.argwhere(lab == int(cid))
        if pts.size == 0:
            continue
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        if bool(r0 == 0 or c0 == 0 or r1 == int(h - 1) or c1 == int(w - 1)):
            continue
        holes.append(_shape_record_from_pts(pts))
    return holes


def _outline_col(sub: np.ndarray, bg: int) -> int:
    cnt = Counter(map(int, sub.flatten()))
    opts = [(c, n) for c, n in cnt.items() if int(c) != int(bg)]
    return int(max(opts, key=lambda t: t[1])[0]) if opts else int(bg)


def _colored_components(sub: np.ndarray, bg: int) -> list[dict]:
    outc = _outline_col(sub, bg)
    lab, n = label_mask((sub != int(bg)) & (sub != int(outc)), connectivity=4)
    subs = []
    for cid in range(1, int(n) + 1):
        pts = np.argwhere(lab == int(cid))
        if pts.size == 0:
            continue
        rec = _shape_record_from_pts(pts)
        rec["color"] = int(Counter(map(int, sub[lab == int(cid)].flatten())).most_common(1)[0][0])
        subs.append(rec)
    return subs


def _arrangement_fill_from_objects(src: np.ndarray, dest: np.ndarray, bg: int) -> np.ndarray:
    out = dest.copy()
    dest_holes = _enclosed_bg_components(dest, bg)
    src_subs = _colored_components(src, bg)
    dest_holes.sort(key=lambda d: (float(d["cent"][0]), float(d["cent"][1]), int(d["size"])))
    src_subs.sort(key=lambda d: (float(d["cent"][0]), float(d["cent"][1]), int(d["color"])))
    for s, h in zip(src_subs, dest_holes):
        for rr, cc in h["pts"]:
            out[int(rr), int(cc)] = int(s["color"])
    return _tight_crop_non_bg(out, bg)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each dst hole using the src subsprite with the same shape up to rotation / flip.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    objs = _extract_two_object_subgrids(g, bg)
    if len(objs) < 2:
        return _tight_crop_non_bg(g, bg)

    holes_list = [_enclosed_bg_components(obj, bg) for obj in objs]
    hole_area = [sum(int(h["size"]) for h in holes) for holes in holes_list]
    dest_i = int(np.argmax(hole_area))
    src_i = int(1 - dest_i)
    dest = objs[dest_i].copy()
    src = objs[src_i].copy()

    dest_holes = _enclosed_bg_components(dest, bg)
    src_subs = _colored_components(src, bg)
    if not dest_holes or not src_subs:
        return _tight_crop_non_bg(dest, bg)

    src_by_sig: dict[tuple[tuple[int, ...], ...], list[dict]] = defaultdict(list)
    for sub in src_subs:
        src_by_sig[sub["sig"]].append(sub)

    dest_by_sig: dict[tuple[tuple[int, ...], ...], list[dict]] = defaultdict(list)
    for hole in dest_holes:
        dest_by_sig[hole["sig"]].append(hole)

    matched = set(src_by_sig) == set(dest_by_sig)
    if matched:
        for sig, holes in dest_by_sig.items():
            subs = src_by_sig.get(sig, [])
            if len(subs) != len(holes):
                matched = False
                break
            holes.sort(key=lambda d: (float(d["cent"][0]), float(d["cent"][1]), int(d["size"])))
            subs.sort(key=lambda d: (float(d["cent"][0]), float(d["cent"][1]), int(d["color"])))
            for sub_rec, hole_rec in zip(subs, holes):
                for rr, cc in hole_rec["pts"]:
                    dest[int(rr), int(cc)] = int(sub_rec["color"])
    if not matched:
        return _arrangement_fill_from_objects(src, dest, bg)

    return _tight_crop_non_bg(dest, bg)


CANVAS_SIZE_RANGE = (22, 30)
OBJECT_H_RANGE = (7, 17)
OBJECT_W_RANGE = (7, 13)
CAVITY_COUNT_RANGE = (4, 9)
FILL_COLOR_COUNT_RANGE = (3, 7)
PLACEMENT_BORDER = 1
PLACEMENT_PADDING = 2
MAX_GEN_ATTEMPTS = 500

SLOT_SIZE = 3
SLOT_STEP = 4
COL_STARTS = [0, 4, 8]
ROW_OPTIONS = [
    (0,),
    (1,),
    (2,),
    (0, 1),
    (1, 2),
    (0, 2),
    (0, 1, 2),
]
ROW_SHIFT_CHOICES = [0, 0, 1]


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


def _has_any_symmetry(mask: np.ndarray) -> bool:
    base = _trim_bool(np.array(mask, dtype=bool))
    if base.size == 0:
        return False
    for k in range(4):
        rot = np.rot90(base, k=k)
        for flip in (False, True):
            arr = np.fliplr(rot) if flip else rot
            arr = _trim_bool(arr)
            if arr.shape != base.shape:
                continue
            if k == 0 and not flip:
                continue
            if np.array_equal(arr, base):
                return True
    return False


def _build_shape_library() -> list[np.ndarray]:
    reps: dict[tuple[tuple[int, ...], ...], np.ndarray] = {}
    for bits in range(1, 1 << (SLOT_SIZE * SLOT_SIZE)):
        arr = np.array(
            [(bits >> i) & 1 for i in range(SLOT_SIZE * SLOT_SIZE)],
            dtype=bool,
        ).reshape(SLOT_SIZE, SLOT_SIZE)
        area = int(arr.sum())
        if area < 4 or area > 8:
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


def _design_layout(n_target: int) -> list[dict] | None:
    for _ in range(300):
        n_rows = random.choice([2, 3, 3, 4, 4])
        rows = []
        total = 0
        for row_idx in range(n_rows):
            cols = random.choice(ROW_OPTIONS)
            shift = random.choice(ROW_SHIFT_CHOICES)
            total += len(cols)
            rows.append((row_idx, cols, shift))
        if total != n_target:
            continue
        layout = []
        seen = set()
        for row_idx, cols, shift in rows:
            for col_idx in cols:
                pos = (int(row_idx * SLOT_STEP), int(COL_STARTS[col_idx] + shift))
                if pos in seen:
                    break
                seen.add(pos)
                layout.append({"r0": pos[0], "c0": pos[1]})
            else:
                continue
            break
        else:
            layout.sort(key=lambda d: (int(d["r0"]), int(d["c0"])))
            slot_mask = np.zeros((n_rows, max(COL_STARTS) + 3), dtype=bool)
            for slot in layout:
                slot_mask[int(slot["r0"] // SLOT_STEP), int(slot["c0"])] = True
            if _has_any_symmetry(slot_mask):
                continue
            return layout
    return None


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
    lab, n = label_mask(empty_grid == int(bg), connectivity=4)
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


def _outline_is_inferable(grid: np.ndarray, bg: int, outline_col: int) -> bool:
    return int(_outline_col(grid, bg)) == int(outline_col)


def _assign_shape_colors(n_shapes: int, fill_pool: list[int]) -> list[int]:
    min_fill = max(FILL_COLOR_COUNT_RANGE[0], min(FILL_COLOR_COUNT_RANGE[1], n_shapes - 2))
    max_fill = min(FILL_COLOR_COUNT_RANGE[1], n_shapes, len(fill_pool))
    if max_fill < FILL_COLOR_COUNT_RANGE[0]:
        raise ValueError("not enough fill colors")
    n_fill = random.randint(min(min_fill, max_fill), max_fill)
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
    if not _outline_is_inferable(dst_cropped, bg, outline_col):
        return None
    if _has_any_symmetry(dst_cropped != int(bg)):
        return None

    src_cropped, _ = _render_object(layout, src_shapes, src_colors, bg, outline_col, filled=True)
    if src_cropped is None:
        return None
    if not _is_8connected(src_cropped, bg):
        return None
    if _count_4conn_cavities(src_cropped, bg, outline_col) != n:
        return None
    if not _size_in_spec(src_cropped):
        return None
    if not _outline_is_inferable(src_cropped, bg, outline_col):
        return None
    if _has_any_symmetry(src_cropped != int(bg)):
        return None

    arrangement_out = _arrangement_fill_from_objects(src_cropped, dst_cropped, bg)
    if np.array_equal(arrangement_out, dst_filled):
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

    n = random.randint(*CAVITY_COUNT_RANGE)
    layout = _design_layout(n)
    if layout is None:
        return None
    if n > len(SHAPE_LIBRARY):
        return None

    prototypes = [m.copy() for m in random.sample(SHAPE_LIBRARY, n)]
    fill_pool = palette[1:]
    colors_by_shape = _assign_shape_colors(n, fill_pool)

    built = _build_object_pair(layout, prototypes, colors_by_shape, bg, outline_col)
    if built is None:
        return None
    src, dst, expected = built

    canvas = _place_on_canvas(src, dst, bg)
    if canvas is None:
        return None

    actual = main(canvas)
    if actual.shape != expected.shape or not np.array_equal(actual, expected):
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


def sample_metrics(grid: np.ndarray) -> dict[str, int | bool]:
    """
    Quantify whether a sample really expresses the fusion:
    - all holes are non-rectangular
    - src/dst are non-symmetric
    - shape-match output differs from arrangement-match output
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    objs = _extract_two_object_subgrids(g, bg)
    if len(objs) < 2:
        raise ValueError("sample_metrics expects two objects")

    holes_list = [_enclosed_bg_components(obj, bg) for obj in objs]
    hole_area = [sum(int(h["size"]) for h in holes) for holes in holes_list]
    dest_i = int(np.argmax(hole_area))
    src_i = int(1 - dest_i)
    dest = objs[dest_i].copy()
    src = objs[src_i].copy()

    dest_holes = _enclosed_bg_components(dest, bg)
    src_subs = _colored_components(src, bg)
    shape_out = main(g)
    arrangement_out = _arrangement_fill_from_objects(src, dest, bg)

    return {
        "cavity_count": int(len(dest_holes)),
        "all_holes_nonrect": bool(all(not _is_rect(h["mask"]) for h in dest_holes)),
        "all_src_shapes_nonrect": bool(all(not _is_rect(s["mask"]) for s in src_subs)),
        "src_has_symmetry": bool(_has_any_symmetry(src != int(bg))),
        "dst_has_symmetry": bool(_has_any_symmetry(dest != int(bg))),
        "arrangement_matches_shape_output": bool(np.array_equal(arrangement_out, shape_out)),
        "arrangement_pixel_diff": int(np.sum(arrangement_out != shape_out)),
        "distinct_shape_count": int(len({h["sig"] for h in dest_holes})),
    }
