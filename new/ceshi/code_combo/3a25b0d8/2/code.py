# Combo variant of 3a25b0d8 (index 2)
# combo_spec: D1/1+D7/2
# Name: 形状吻合匹配+全非矩形空腔

from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np

from arc_utils import bbox, crop_bbox, get_bg, label_mask, paste, random_free_location


"""
design_analysis:
  variant_goal:
    keep:
      - 输入仍由两个分离的 8-connected 物体组成：src 是带彩色子精灵的轮廓体，dst 是同 outline 色但洞为空的轮廓体
      - 输出仍是把 dst 洞填色后的紧裁结果
      - 画布尺寸、随机放置、单一背景、tight crop 等外围机制保持原题风格
    change:
      - D1/1: 匹配规则从“按质心行列秩序对位”改为“按形状完全吻合配对，允许旋转/翻转”
      - D7/2: 所有洞都必须是非矩形；不再允许简单实心矩形洞混入样本

  a_concretization_and_trace:
    pixel_level_structure:
      - 每个 cavity 都是一个 3x3 满 bbox 的 4-connected 非矩形轮廓，典型外观是 L、T、十字缺角、阶梯、折线
      - src 和 dst 共享同一批 canonical shape，但会被放到不同槽位；同一 shape 在两边还会各自独立旋转/翻转
      - 因而“位置”和“朝向”都不再可靠，唯一稳定线索是 canonical contour identity
    small_trace:
      - 原题：dst 左上洞、右上洞、下方洞按 (row_rank, col_rank) 排序，src 中相同 rank 的三个彩块颜色依次填进去
      - 本融合：dst 左上洞是一个 T 形，必须去 src 的所有彩块里找“同一个 T 形”（即使它在别的位置且被旋转了 90 度），把那个颜色填回左上洞
      - 额外叠加 D7/2 后，所有洞都长成不规则轮廓，解题者不能借助简单矩形尺寸做捷径，只能精看轮廓细节

  b_complexity_calibration:
    - 若 src/dst 把同一 shape 仍放在同一槽位，则会退化回原题；因此生成器强制对 shape->slot 做 derangement
    - 若允许重复 canonical shape，shape-match 会变成多对一歧义；因此每个样本只采样 unique canonical shapes
    - 若 shape-match 与原 rank-match 恰好给出同一输出，融合行为对外不可见；因此生成器拒绝这类退化样本
    - 若混入矩形洞，人类可能只靠“长宽/面积”而非复杂轮廓来猜；因此 shape library 本身只包含非矩形满 3x3 bbox 原型

  c_implicit_assumption_audit_on_original_main:
    broken_assumptions:
      - _rank_keys(...) 假设 src 子精灵与 dst 洞在某个整体旋转/翻转后共享同一组 row/col rank
      - best=(k,flip) 的搜索假设“一个全局刚体变换”足以恢复所有配对
      - 排序后 zip 的写法假设“位置顺序就是语义身份”
    why_invalid_now:
      - 变体里相同 shape 被故意打乱到不同槽位，所以整体 rank 一般不对应
      - 同一 shape 在 src/dst 中可各自独立旋转/翻转，不能用单一全局变换解释
      - 洞的身份来自局部轮廓，不来自全局坐标
    replacement_in_main:
      - 对每个洞/子精灵抽取 bbox 内的布尔掩码
      - 枚举 8 个二面体变换，取最小 canonical signature
      - 按 signature 分组并一一配对，再把 src 颜色写入对应 dst 洞

  d_placement_legality:
    - 仍用 random_free_location 在 22-30 画布上摆放 src 和 dst
    - border=1, padding=2，保证不越界、不碰撞、视觉上可分离
    - solver 只依赖“最大的两个非背景 8-connected 物体”这一原题外层结构

  e_visual_observability:
    - 看 2-4 组样例时，人类能直接观察到：src 中若干不同的不规则彩色小块，与 dst 中若干同形但位置打乱的空洞一一对应
    - 因为全部洞都是非矩形，所以“按形状找颜色”是唯一稳定且可见的规律
    - 匹配不依赖颜色编号、坐标值等不可视属性；颜色只是被 shape identity 搬运

  f_visual_cleanliness:
    - 仍使用规则槽位布局，洞之间保持至少 1 个空列/空行，因此不会糊成无法分辨的团
    - 只保留 outline 和 cavity/fill 两层视觉信息，没有额外标记
    - 3x3 原型虽小，但全部占满 bbox 且为非矩形，轮廓差异在 ARC 像素尺度上仍然清晰

  g_color_inference:
    - 颜色种类保持 3-7，足够提供配对反馈但不会多到失控
    - 颜色映射规则不是抽象置换，而是“沿 shape identity 传递”；结构线索清晰

  h_generator_guardrails:
    - shape library 只收录：满 3x3 bbox、4-connected、无内部孔、canonical unique、非矩形 的原型
    - 生成后再次验证 dst 的所有洞都非矩形，确保 D7/2 在最终像素结果中真正发生
    - 使用 oracle check main(input)==expected，并额外要求 baseline rank solver 的输出不同，防止退化

  i_new_solver_assumptions_and_generator_support:
    solver_assumptions:
      - 恰有两个主要非背景 8-connected 物体
      - 每个 src 子精灵和某个 dst 洞在 canonical signature 上唯一对应
      - outline 色可由“非背景中面积最大的颜色”识别
      - 子精灵本身没有内部空孔，否则 src 中会残留 bg 岛破坏组件抽取
    generator_support:
      - 通过 canonical-unique library 保证唯一匹配
      - 通过满 bbox 非矩形原型保证所有洞都不规则
      - 通过 render 验证和退化过滤保证 solver 的新假设在生成样本中始终成立
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


def _extract_two_objects(grid: np.ndarray, bg: int) -> list[np.ndarray]:
    comps = _components(grid != int(bg))
    if len(comps) < 2:
        raise ValueError("expected two non-background objects")
    objs = []
    for comp in comps[:2]:
        r0, c0, r1, c1 = comp["bbox"]
        sub = np.full((int(r1 - r0 + 1), int(c1 - c0 + 1)), int(bg), dtype=int)
        for rr, cc in comp["pts"]:
            sub[int(rr) - int(r0), int(cc) - int(c0)] = int(grid[int(rr), int(cc)])
        objs.append(sub)
    return objs


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
        "mask": mask,
        "sig": _mask_signature(mask),
    }


def _enclosed_bg_components(sub: np.ndarray, bg: int) -> list[dict]:
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


def _outline_col(sub: np.ndarray, bg: int) -> int:
    cnt = Counter(map(int, sub.flatten()))
    opts = [(c, n) for c, n in cnt.items() if int(c) != int(bg)]
    return int(max(opts, key=lambda t: t[1])[0]) if opts else int(bg)


def _colored_components(sub: np.ndarray, bg: int) -> list[dict]:
    outc = _outline_col(sub, bg)
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


def _is_rect(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return False
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return bool(mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].all())


def _rank_keys(pts: list[tuple[int, int]]) -> tuple[list[tuple[int, int]], dict[int, int], dict[int, int]]:
    rows = sorted(set(int(r) for r, _c in pts))
    cols = sorted(set(int(c) for _r, c in pts))
    rrank = {r: i for i, r in enumerate(rows)}
    crank = {c: i for i, c in enumerate(cols)}
    keys = [(int(rrank[r]), int(crank[c])) for r, c in pts]
    return keys, rrank, crank


def _transform_point(p: tuple[int, int], h0: int, w0: int, k: int, flip: bool) -> tuple[int, int]:
    r, c = int(p[0]), int(p[1])
    if int(k) == 0:
        rr, cc = int(r), int(c)
        h2, w2 = int(h0), int(w0)
    elif int(k) == 1:
        rr, cc = int(w0 - 1 - c), int(r)
        h2, w2 = int(w0), int(h0)
    elif int(k) == 2:
        rr, cc = int(h0 - 1 - r), int(w0 - 1 - c)
        h2, w2 = int(h0), int(w0)
    else:
        rr, cc = int(c), int(h0 - 1 - r)
        h2, w2 = int(w0), int(h0)
    if bool(flip):
        cc = int(w2 - 1 - cc)
    return int(rr), int(cc)


def _shape_fill_main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    objs = _extract_two_objects(g, bg)
    holes_list = [_enclosed_bg_components(o, bg) for o in objs]
    hole_area = [sum(int(h["size"]) for h in hh) for hh in holes_list]
    dest_i = int(np.argmax(hole_area))
    src_i = int(1 - dest_i)
    dest = objs[dest_i].copy()
    src = objs[src_i].copy()

    dest_holes = _enclosed_bg_components(dest, bg)
    src_subs = _colored_components(src, bg)
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each dst hole with the color of the src subsprite having the same
    canonical shape up to rotation / flip, then tight-crop the filled dst object.
    """
    return _shape_fill_main(grid)


def baseline_rank_main(grid: np.ndarray) -> np.ndarray:
    """
    Baseline reproducing the original rank-based matching logic.
    Used only for generator anti-degeneration checks and post-hoc behavior metrics.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    objs = _extract_two_objects(g, bg)
    holes_list = [_enclosed_bg_components(o, bg) for o in objs]
    hole_area = [sum(int(h["size"]) for h in hh) for hh in holes_list]
    dest_i = int(np.argmax(hole_area))
    src_i = int(1 - dest_i)
    dest = objs[dest_i].copy()
    src = objs[src_i].copy()

    dest_holes = _enclosed_bg_components(dest, bg)
    src_subs = _colored_components(src, bg)
    if not dest_holes or not src_subs:
        return _tight_crop_non_bg(dest, bg)

    dest_pts = [h["cent"] for h in dest_holes]
    dest_keys, d_rr, d_cc = _rank_keys(dest_pts)

    best = None
    for k in range(4):
        for flip in (False, True):
            tpts = [_transform_point(s["cent"], int(src.shape[0]), int(src.shape[1]), int(k), bool(flip)) for s in src_subs]
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
    for h in dest_holes:
        key = (int(d_rr[int(h["cent"][0])]), int(d_cc[int(h["cent"][1])]))
        dest_list.append((key, h["cent"], h))
    dest_list.sort(key=lambda t: (t[0][0], t[0][1], t[1][0], t[1][1]))

    tpts_all = [_transform_point(s["cent"], int(src.shape[0]), int(src.shape[1]), int(k), bool(flip)) for s in src_subs]
    _keys, s_rr, s_cc = _rank_keys(tpts_all)

    src_list = []
    for s in src_subs:
        tp = _transform_point(s["cent"], int(src.shape[0]), int(src.shape[1]), int(k), bool(flip))
        key = (int(s_rr[int(tp[0])]), int(s_cc[int(tp[1])]))
        src_list.append((key, tp, int(s["color"])))
    src_list.sort(key=lambda t: (t[0][0], t[0][1], t[1][0], t[1][1]))

    for (_k1, _p1, col), (_k2, _p2, hole) in zip(src_list, dest_list):
        for rr, cc in hole["pts"]:
            dest[int(rr), int(cc)] = int(col)

    return _tight_crop_non_bg(dest, bg)


CANVAS_SIZE_RANGE = (22, 30)
OBJECT_H_RANGE = (7, 17)
OBJECT_W_RANGE = (7, 13)
CAVITY_COUNT_RANGE = (4, 8)
FILL_COLOR_COUNT_RANGE = (3, 7)
PLACEMENT_BORDER = 1
PLACEMENT_PADDING = 2
MAX_GEN_ATTEMPTS = 500

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


def _count_enclosed_bg_in_mask(mask: np.ndarray) -> int:
    inv = ~np.array(mask, dtype=bool)
    lab, n = label_mask(inv, connectivity=4)
    h, w = inv.shape
    count = 0
    for cid in range(1, int(n) + 1):
        pts = np.argwhere(lab == int(cid))
        if pts.size == 0:
            continue
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        if r0 > 0 and c0 > 0 and r1 < h - 1 and c1 < w - 1:
            count += 1
    return count


def _build_shape_library() -> list[np.ndarray]:
    reps: dict[tuple[tuple[int, ...], ...], np.ndarray] = {}
    for bits in range(1, 1 << (SLOT_SIZE * SLOT_SIZE)):
        arr = np.array([(bits >> i) & 1 for i in range(SLOT_SIZE * SLOT_SIZE)], dtype=bool).reshape(SLOT_SIZE, SLOT_SIZE)
        area = int(arr.sum())
        if area < 5 or area > 7:
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
        if _count_enclosed_bg_in_mask(arr) != 0:
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
        if pts.size == 0:
            continue
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        if r0 > 0 and c0 > 0 and r1 < h - 1 and c1 < w - 1:
            count += 1
    return count


def _all_enclosed_holes_nonrect(empty_grid: np.ndarray, bg: int) -> bool:
    for hole in _enclosed_bg_components(empty_grid, int(bg)):
        if _is_rect(hole["mask"]):
            return False
    return True


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
    if not _all_enclosed_holes_nonrect(dst_cropped, bg):
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

    if np.array_equal(src_cropped, dst_filled):
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


def variant_metrics(grid: np.ndarray) -> dict[str, int]:
    """
    Quantify whether the generated sample exhibits the combo behavior.
    Metrics:
      - all_nonrect_holes: 1 iff every dst hole is non-rectangular
      - shape_vs_rank_diff: 1 iff shape-match output differs from original rank-match output
      - cavity_count: number of dst holes / src sprites
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    objs = _extract_two_objects(g, bg)
    holes_list = [_enclosed_bg_components(o, bg) for o in objs]
    hole_area = [sum(int(h["size"]) for h in hh) for hh in holes_list]
    dest = objs[int(np.argmax(hole_area))]
    hole_count = len(_enclosed_bg_components(dest, bg))
    shape_out = main(g)
    rank_out = baseline_rank_main(g)
    return {
        "all_nonrect_holes": int(_all_enclosed_holes_nonrect(dest, bg)),
        "shape_vs_rank_diff": int(not np.array_equal(shape_out, rank_out)),
        "cavity_count": int(hole_count),
    }


def _gen_attempt() -> np.ndarray | None:
    bg, palette = _pick_bg_and_palette()
    outline_col = int(palette[0])

    target_cavities = random.randint(*CAVITY_COUNT_RANGE)
    pattern_choices = [p for p in ROW_PATTERNS if sum(p) == int(target_cavities)]
    if not pattern_choices:
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
        baseline = baseline_rank_main(canvas)
        metrics = variant_metrics(canvas)
    except Exception:
        return None
    if not isinstance(actual, np.ndarray):
        return None
    if actual.shape != expected.shape or not np.array_equal(actual, expected):
        return None
    if np.array_equal(actual, baseline):
        return None
    if metrics["all_nonrect_holes"] != 1 or metrics["shape_vs_rank_diff"] != 1:
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
