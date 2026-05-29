# Combo variant of 3a25b0d8 (index 24)
# combo_spec: D1/1+D12/2
# Name: 混合连通性+形状匹配

from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np

from arc_utils import bbox, crop_bbox, get_bg, label_mask, paste, random_free_location


"""
design_analysis:
  variant_goal:
    keep:
      - 输入仍由两个分离的 8-connected 物体组成：src 是带轮廓的彩色内容，dst 是同色轮廓中的空洞
      - 输出仍是把 dst 的洞填色后，对 dst 做 tight crop
      - 背景、outline、随机放置、对象整体 8-connected 等外围约束保持原题风格
    change:
      - D1/1: 不再按质心行列秩序配对，而是按形状 canonical signature 配对
      - D12/2: src 的可见彩色内容故意拆成多个 4-connected 碎片，再由 8 邻接斜接成一个逻辑形状；
        dst 的每个洞直接是对应的 8-connected 形状。因此 src 的 4-connected 片段数 > dst 的 8-connected 洞数

  a_concretization_and_trace:
    pixel_level_structure:
      - 每个逻辑 shape 放在一个 3x3 槽位里，shape 本身是 "8-connected 但非 4-connected" 的布尔掩码
      - src 中该 shape 被单色填满，所以肉眼看到的是几个同色碎片斜接成一个整体；按 4 连通会裂成 2-3 块
      - dst 中对应位置是同一个 shape 的空洞；按 8 连通它是 1 个洞，按 4 连通会裂成更多小洞
      - src 与 dst 共享同一批 canonical 形状，但显式打乱到不同槽位，且各自独立旋转/翻转
    small_trace:
      - 原题：先对 src 子精灵和 dst 洞做整体几何对齐，再按 (row_rank, col_rank) 一一 zip
      - 融合后：dst 左上洞可能是一个由 3 个对角像素链成的 8 形状；src 里同形状可能在右下，
        且在 src 中它不是一个 4-connected 实体，而是两个同色碎片斜接而成
      - 正确做法：先把 src 的 4-connected 彩色碎片提取出来，再按 "同色 + 8 邻接" 合并回逻辑 shape，
        对每个逻辑 shape 求 canonical signature，与 dst 的 8-connected 洞签名匹配，最后用该组颜色填洞
      - 差异：位置排序完全失效；若只用 4 连通看 dst，会把一个逻辑洞误拆成多个；若只用 8 连通看 src，
        虽能看到整体，但会掩盖“src 是由多个 4-connected 碎片组成”的核心融合行为

  b_complexity_calibration:
    - 若 src/dst 不打乱槽位，样本会退化为原题的位置匹配，因此必须使用 derangement 置换
    - 若允许重复 canonical 形状，shape-match 会歧义，因此每个样本只采样唯一形状
    - 若颜色重复太多，错误配对在输出上可能不可见，因此每个逻辑 shape 默认使用不同颜色
    - 若所有 shape 恰好也是 4-connected，则 D12/2 只剩表面参数变化；因此库中只保留 "8-connected 且非 4-connected" 形状

  c_implicit_assumption_audit_on_original_main:
    broken_assumptions:
      - `_rank_keys(...)` 假设语义对应关系由质心秩序决定；融合后不成立，因为形状被打乱到不同槽位
      - `best=(k, flip)` 假设存在一个全局刚体变换能把 src 的位置秩序对齐到 dst；融合后不成立
      - `_colored_components(..., connectivity=4)` 原本把每个彩色子精灵当成一个 4-connected 实体；
        融合后一个逻辑 src shape 会被故意拆成多个 4-connected 片段
      - `_enclosed_bg_components(..., connectivity=4)` 原本默认 dst 洞在 4 连通下计数；融合后必须改成 8 连通，
        否则会把一个逻辑洞误拆为多个
    replacement:
      - src: 先取 4-connected 彩色碎片，再根据同色且 8 邻接把碎片合并成逻辑组
      - dst: 直接取 8-connected 的内部 bg 洞
      - matching: 对 src 逻辑组和 dst 洞各自求 dihedral-invariant signature，用 signature 做一一配对

  d_placement_legality:
    - 仍通过 `random_free_location(..., border=1, padding=2)` 随机放置 src 和 dst
    - 对象先在各自 tight bbox 中渲染，再贴到 22-30 的画布上，保证不越界、不重叠
    - 两个物体都是 8-connected，所以 solver 仍可从全图最大的两个非背景 8-connected 物体中恢复它们

  e_visual_observability:
    - 人类能直接看到 src 的同色碎片是对角拼起来的，而 dst 的洞也是对应的对角轮廓
    - 由于相同 shape 在 src/dst 中位置被打乱，观察几组样例后可以排除“按位置搬色”，转而发现“按轮廓搬色”
    - 由于 dst 的洞经常包含对角连接，若尝试把它看成多个 4-connected 小洞，会与输出中的单色整洞填充矛盾，
      从而迫使解题者接受混合连通性规则

  f_visual_cleanliness:
    - 仍使用规则槽位布局和单层 outline，没有额外标记或文字噪声
    - 每个逻辑 shape 限定在 3x3 槽位内，整体对象尺寸保持在原题范围内
    - 物体间保留空隙，src 与 dst 清楚分离，不会出现覆盖叠加导致的阅读困难

  g_color_inference:
    - 颜色不是做重新映射，而是把 src 逻辑 shape 的单色直接搬到同形 dst 洞
    - 每个逻辑 shape 默认不同色，使颜色归属由可见 shape 提示支撑，而不是靠不可见编号
    - src 的一个逻辑 shape 虽可分裂成多个 4-connected 片段，但这些片段共享同色，因此颜色线索一致

  h_generator_guardrails:
    - 仅采样 canonical-unique 的 3x3 shape，且每个 shape 必须满足 n8=1 且 n4>=2
    - 仅接受 `src_fill_4 > src_fill_8 == n_shapes` 且 `dst_holes_4 > dst_holes_8 == n_shapes` 的样本
    - 仅接受 shape-match 输出 与 arrangement-style 输出不同的样本，避免退化回原题
    - 仅接受 solver oracle `main(input) == expected` 的样本

  i_new_solver_assumptions_and_generator_support:
    solver_assumptions:
      - 恰有两个主要非背景 8-connected 物体
      - src 中每个逻辑组是单色的，且由若干同色 4-connected 碎片通过 8 邻接组成
      - 不同逻辑组之间不会发生同色 8 邻接粘连
      - dst 的逻辑洞是内部 8-connected bg 组件，且 shape multiset 与 src 逻辑组完全一致
    generator_support:
      - 通过固定 3x3 槽位和步长 4，保证组间不会误粘连
      - 通过唯一 shape 采样和 derangement，保证形状匹配唯一且位置匹配失效
      - 通过单色组构造，保证填洞仍是“整洞单色”的清晰视觉规则
"""


CANVAS_SIZE_RANGE = (22, 30)
OBJECT_H_RANGE = (7, 17)
OBJECT_W_RANGE = (7, 13)
GROUP_COUNT_RANGE = (4, 6)
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
    (1, 3, 1),
    (2, 2, 2),
]
COUNT_TO_COLS = {
    1: [4],
    2: [2, 6],
    3: [0, 4, 8],
}


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
    mask = grid != int(bg)
    pts = np.argwhere(mask)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


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


def _components(mask: np.ndarray, *, connectivity: int = 8) -> list[dict]:
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


def _extract_two_objects(grid: np.ndarray, bg: int) -> list[np.ndarray]:
    comps = _components(grid != int(bg), connectivity=8)
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


def _outline_col(sub: np.ndarray, bg: int) -> int:
    cnt = Counter(map(int, sub.flatten()))
    opts = [(c, n) for c, n in cnt.items() if int(c) != int(bg)]
    return int(max(opts, key=lambda t: t[1])[0]) if opts else int(bg)


def _enclosed_bg_components(sub: np.ndarray, bg: int, *, connectivity: int) -> list[dict]:
    lab, n = label_mask(sub == int(bg), connectivity=connectivity)
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


def _find_root(parent: list[int], x: int) -> int:
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def _union(parent: list[int], a: int, b: int) -> None:
    ra = _find_root(parent, a)
    rb = _find_root(parent, b)
    if ra != rb:
        parent[rb] = ra


def _src_groups_from_fragments(sub: np.ndarray, bg: int) -> list[dict]:
    outc = _outline_col(sub, bg)
    fill_mask = (sub != int(bg)) & (sub != int(outc))
    lab4, n4 = label_mask(fill_mask, connectivity=4)
    if int(n4) == 0:
        return []

    fragments = []
    colors: list[int] = []
    for cid in range(1, int(n4) + 1):
        pts = np.argwhere(lab4 == int(cid))
        if pts.size == 0:
            continue
        col = int(Counter(map(int, sub[lab4 == int(cid)].flatten())).most_common(1)[0][0])
        fragments.append({"cid": int(cid), "pts": pts, "color": col})
        colors.append(col)

    parent = list(range(len(fragments)))
    cid_to_idx = {int(f["cid"]): i for i, f in enumerate(fragments)}
    h, w = sub.shape
    for r in range(h):
        for c in range(w):
            cid = int(lab4[r, c])
            if cid == 0:
                continue
            i = cid_to_idx[cid]
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = int(r + dr), int(c + dc)
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    ocid = int(lab4[nr, nc])
                    if ocid == 0 or ocid == cid:
                        continue
                    j = cid_to_idx[ocid]
                    if colors[i] == colors[j]:
                        _union(parent, i, j)

    merged: dict[int, dict] = {}
    for i, frag in enumerate(fragments):
        root = _find_root(parent, i)
        rec = merged.setdefault(root, {"color": int(frag["color"]), "pts": []})
        rec["pts"].append(frag["pts"])

    groups = []
    for rec in merged.values():
        pts = np.vstack(rec["pts"]).astype(int)
        shape = _shape_record_from_pts(pts)
        shape["color"] = int(rec["color"])
        groups.append(shape)
    groups.sort(key=lambda d: (int(d["cent"][0]), int(d["cent"][1]), int(d["color"])))
    return groups


def _count_fill_components(grid: np.ndarray, bg: int, outline_col: int, *, connectivity: int) -> int:
    mask = (grid != int(bg)) & (grid != int(outline_col))
    _, n = label_mask(mask, connectivity=connectivity)
    return int(n)


def _count_holes(grid: np.ndarray, bg: int, *, connectivity: int) -> int:
    return int(len(_enclosed_bg_components(grid, bg, connectivity=connectivity)))


def _is_8connected(grid: np.ndarray, bg: int) -> bool:
    mask = grid != int(bg)
    if not mask.any():
        return False
    _, n = label_mask(mask, connectivity=8)
    return int(n) == 1


def _size_in_spec(grid: np.ndarray) -> bool:
    h, w = grid.shape
    return OBJECT_H_RANGE[0] <= h <= OBJECT_H_RANGE[1] and OBJECT_W_RANGE[0] <= w <= OBJECT_W_RANGE[1]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover src logical shapes by merging 4-connected same-color fragments through 8-adjacency,
    match them to dst's 8-connected holes by shape signature, fill the holes with the matched colors,
    and tight-crop the filled dst object.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    objs = _extract_two_objects(g, bg)
    hole_areas = [sum(int(h["size"]) for h in _enclosed_bg_components(obj, bg, connectivity=8)) for obj in objs]
    dest_i = int(np.argmax(hole_areas))
    src_i = int(1 - dest_i)
    dest = objs[dest_i].copy()
    src = objs[src_i].copy()

    dest_holes = _enclosed_bg_components(dest, bg, connectivity=8)
    src_groups = _src_groups_from_fragments(src, bg)
    if not dest_holes or not src_groups:
        return _tight_crop_non_bg(dest, bg)

    src_by_sig: dict[tuple[tuple[int, ...], ...], list[dict]] = defaultdict(list)
    for rec in src_groups:
        src_by_sig[rec["sig"]].append(rec)

    dest_by_sig: dict[tuple[tuple[int, ...], ...], list[dict]] = defaultdict(list)
    for rec in dest_holes:
        dest_by_sig[rec["sig"]].append(rec)

    if set(src_by_sig) != set(dest_by_sig):
        raise ValueError("shape multiset mismatch between src groups and dst holes")

    for sig, holes in dest_by_sig.items():
        groups = src_by_sig.get(sig, [])
        if len(groups) != len(holes):
            raise ValueError("shape multiplicity mismatch")
        holes.sort(key=lambda d: (int(d["cent"][0]), int(d["cent"][1]), int(d["size"])))
        groups.sort(key=lambda d: (int(d["cent"][0]), int(d["cent"][1]), int(d["color"])))
        for src_rec, hole_rec in zip(groups, holes):
            for rr, cc in hole_rec["pts"]:
                dest[int(rr), int(cc)] = int(src_rec["color"])

    return _tight_crop_non_bg(dest, bg)


def _is_rect(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return bool(mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].all())


def _build_shape_library() -> list[np.ndarray]:
    reps: dict[tuple[tuple[int, ...], ...], np.ndarray] = {}
    for bits in range(1, 1 << (SLOT_SIZE * SLOT_SIZE)):
        arr = np.array([(bits >> i) & 1 for i in range(SLOT_SIZE * SLOT_SIZE)], dtype=bool).reshape(SLOT_SIZE, SLOT_SIZE)
        pts = np.argwhere(arr)
        if pts.size == 0:
            continue
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        if (int(r0), int(c0), int(r1), int(c1)) != (0, 0, SLOT_SIZE - 1, SLOT_SIZE - 1):
            continue
        area = int(arr.sum())
        if area < 4 or area > 7:
            continue
        if _is_rect(arr):
            continue
        _, n8 = label_mask(arr, connectivity=8)
        _, n4 = label_mask(arr, connectivity=4)
        if int(n8) != 1 or int(n4) <= 1:
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
    for row_idx, count in enumerate(pattern):
        r0 = row_idx * SLOT_STEP
        for c0 in COUNT_TO_COLS[int(count)]:
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
    for slot, shape, color in zip(layout, shapes, colors):
        rr, cc = np.where(shape)
        if len(rr) == 0:
            continue
        placed.append((slot, rr, cc, int(color)))
    if not placed:
        return None, None

    all_rows = np.concatenate([slot["r0"] + rr for slot, rr, _cc, _col in placed])
    all_cols = np.concatenate([slot["c0"] + cc for slot, _rr, cc, _col in placed])
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
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rs = slice(max(0, dr), h + min(0, dr))
            cs = slice(max(0, dc), w + min(0, dc))
            rt = slice(max(0, -dr), h + min(0, -dr))
            ct = slice(max(0, -dc), w + min(0, -dc))
            outline_mask[rt, ct] |= fill_mask[rs, cs]
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


def _analyze_object_pair(src: np.ndarray, dst: np.ndarray, bg: int) -> dict:
    src_outline = _outline_col(src, bg)
    return {
        "src_fill_4": _count_fill_components(src, bg, src_outline, connectivity=4),
        "src_fill_8": _count_fill_components(src, bg, src_outline, connectivity=8),
        "src_group_count": len(_src_groups_from_fragments(src, bg)),
        "dst_holes_4": _count_holes(dst, bg, connectivity=4),
        "dst_holes_8": _count_holes(dst, bg, connectivity=8),
    }


def analyze_sample(grid: np.ndarray) -> dict:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    src, dst = _extract_two_objects(g, bg)
    hole_areas = [sum(int(h["size"]) for h in _enclosed_bg_components(obj, bg, connectivity=8)) for obj in (src, dst)]
    dest_i = int(np.argmax(hole_areas))
    src_i = int(1 - dest_i)
    metrics = _analyze_object_pair((src, dst)[src_i], (src, dst)[dest_i], bg)
    metrics["bg"] = int(bg)
    return metrics


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
    src_colors_by_slot = [int(colors_by_shape[int(perm[i])]) for i in range(n)]

    dst_cropped, dst_filled = _render_object(layout, dst_shapes, colors_by_shape, bg, outline_col, filled=False)
    if dst_cropped is None or dst_filled is None:
        return None
    if not _is_8connected(dst_cropped, bg):
        return None
    if not _size_in_spec(dst_cropped):
        return None

    src_cropped, _src_filled = _render_object(layout, src_shapes, src_colors_by_slot, bg, outline_col, filled=True)
    if src_cropped is None:
        return None
    if not _is_8connected(src_cropped, bg):
        return None
    if not _size_in_spec(src_cropped):
        return None

    metrics = _analyze_object_pair(src_cropped, dst_cropped, bg)
    if metrics["src_fill_8"] != n or metrics["src_group_count"] != n:
        return None
    if metrics["dst_holes_8"] != n:
        return None
    if metrics["src_fill_4"] <= metrics["src_fill_8"]:
        return None
    if metrics["dst_holes_4"] <= metrics["dst_holes_8"]:
        return None

    src_groups = _src_groups_from_fragments(src_cropped, bg)
    dst_holes = _enclosed_bg_components(dst_cropped, bg, connectivity=8)
    if sorted(rec["sig"] for rec in src_groups) != sorted(rec["sig"] for rec in dst_holes):
        return None

    _arranged, arrangement_filled = _render_object(layout, dst_shapes, src_colors_by_slot, bg, outline_col, filled=True)
    if arrangement_filled is None:
        return None
    if np.array_equal(arrangement_filled, dst_filled):
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
            padding_connectivity=8,
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
            padding_connectivity=8,
        )
    except ValueError:
        return None
    paste(canvas, dst, int(r), int(c), transparent=bg)
    return canvas


def _gen_attempt() -> np.ndarray | None:
    bg, palette = _pick_bg_and_palette()
    outline_col = int(palette[0])

    patterns = [p for p in ROW_PATTERNS if GROUP_COUNT_RANGE[0] <= sum(p) <= GROUP_COUNT_RANGE[1]]
    pattern = random.choice(patterns)
    layout = _slot_layout(pattern)
    n = len(layout)
    if n > len(SHAPE_LIBRARY):
        return None

    prototypes = [mask.copy() for mask in random.sample(SHAPE_LIBRARY, n)]
    colors_by_shape = [int(c) for c in palette[1 : 1 + n]]
    if len(colors_by_shape) != n:
        return None

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
            grid = _gen_attempt()
        except Exception:
            grid = None
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {MAX_GEN_ATTEMPTS} attempts")
