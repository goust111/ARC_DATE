# Combo variant of 3a25b0d8 (index 21)
# combo_spec: D11/0+D15/1
# Name: 干扰框架+位置判角色
#
# Design analysis
# ===============
# a) Concrete behavior and trace
# - 画布上有 3-4 个 8 连通对象，并且全部使用同一种 outline 颜色。
# - 最左边的对象恒为 src：它是带彩色子精灵的轮廓体。
# - src 右侧还有 2-3 个空框；其中恰有一个框的洞质心秩签名与 src 在某个姿态下匹配，它是真 dst，
#   其余是干扰框架。
# - 小例子：原题里若只有 src 和 dst，旧 solver 会比较两者 hole_area，洞面积更大的那个当 dst。
#   现在输入可能是 `src | fake_frame | real_dst`。solver 先用“最左=src”定位彩色来源，再在右侧多个
#   框里做签名比对；只有 real_dst 通过，fake_frame 虽然同色同风格甚至洞面积更大，也不能被填。
#
# b) Complexity calibration
# - 位置规则固定为“最左对象是 src”，不在左右/上下多个档位间切换，避免样例太少时难以归纳。
# - 干扰框架保留原任务同类视觉风格：同色 outline、相近洞数、相近 hole area；不能靠“样子完全不同”
#   或“数洞即可排除”偷懒。
# - 生成器还要求至少一个干扰框的 hole_area 不小于真 dst，避免退化成“右侧框里洞面积最大者就是真 dst”。
#
# c) Implicit-assumption audit against the original main()
# - 原 `_extract_two_objects()` 假设只有两个对象；combo 下失效，必须遍历全部对象。
# - 原 `dest_i = argmax(hole_area)` 假设角色靠洞面积区分；combo 下失效，因为 D15/1 改为位置锚定 src，
#   D11 还会放入 hole_area 更大的假框。
# - 原 `src_i = 1 - dest_i` 假设剩余对象就是 src；combo 下失效，因为右侧可能有 2-3 个空框。
# - 原 `zip(src_list, dest_list)` 依赖已经找到唯一正确 dst；combo 下必须先做“左侧锁定 src，再匹配多个框”。
#
# d) Placement legality
# - 为了让“最左=src”是稳定可见规则，generator 先放 src，再把其他框限制在更靠右的列区间内。
# - 右侧各框之间允许在列投影上重叠，但其矩形 bbox 不能相交或贴边，因此不会碰撞、越界或 8 连通粘连。
#
# e) Visual observability
# - 人类能直接看见：左边只有一个彩色源对象，右边有多个同色空框，输出只会把其中一个框填满。
# - 位置线索和签名筛选都来自可见几何关系，不依赖颜色编号、面积排序中的隐藏 tie-break 等不可见因素。
#
# f) Visual cleanliness
# - 不增加文字、箭头或标记层；对象仍是标准 ARC 块状轮廓。
# - 通过 bbox 级间距约束保持对象分离，避免多框互相贴边造成“哪个框是一个整体”看不清。
#
# g) Change inferability
# - 颜色传递规则仍然只由 src 子精灵提供，颜色种类保持原题范围，不额外引入难以归纳的颜色变换。
# - 位置变化只负责“谁是 src”，真正 dst 仍需由洞排列验证，因此两个维度在推理链上是协同的。
#
# h) Post-implementation metric
# - 核心量化指标：
#   1. leftmost object carries colored subsprites.
#   2. exactly one non-leftmost empty frame matches src's transformed centroid-rank signature.
#   3. at least one distractor has hole_area >= true dst hole_area.
# - 若任一指标频繁失败，就说明融合行为在样本中退化。
#
# i) New solver assumptions and generator guarantees
# - 假设：最左对象就是 src。保证：generator 显式约束 src 的 bbox.left 严格小于其他对象。
# - 假设：其余对象里恰有一个签名匹配的真 dst。保证：干扰框生成时拒绝所有可匹配签名。
# - 假设：至少保留 1 个空框干扰。保证：D11 仍固定生成 1-2 个 distractors。

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox, crop_bbox, get_bg, label_mask, paste


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


def _bbox_of(group: list[dict]) -> tuple[int, int, int, int]:
    r0 = min(int(d["bbox"][0]) for d in group)
    c0 = min(int(d["bbox"][1]) for d in group)
    r1 = max(int(d["bbox"][2]) for d in group)
    c1 = max(int(d["bbox"][3]) for d in group)
    return int(r0), int(c0), int(r1), int(c1)


def _extract_group_subgrid(grid: np.ndarray, bg: int, group: list[dict]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_of(group)
    sub = np.full((int(r1 - r0 + 1), int(c1 - c0 + 1)), int(bg), dtype=int)
    for d in group:
        for rr, cc in d["pts"]:
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
    outc = _outline_col(sub, bg)
    mask = (sub != int(bg)) & (sub != int(outc))
    lab, n = label_mask(mask, connectivity=4)
    subs = []
    for cid in range(1, int(n) + 1):
        pts = np.argwhere(lab == int(cid))
        if pts.size == 0:
            continue
        col = Counter(map(int, sub[lab == int(cid)].flatten())).most_common(1)[0][0]
        cent = (int(round(float(pts[:, 0].mean()))), int(round(float(pts[:, 1].mean()))))
        subs.append({"cent": cent, "color": int(col)})
    return subs


def _rank_keys(pts: list[tuple[int, int]]):
    rows = sorted(set(int(r) for r, _ in pts))
    cols = sorted(set(int(c) for _, c in pts))
    rrank = {r: i for i, r in enumerate(rows)}
    crank = {c: i for i, c in enumerate(cols)}
    keys = [(int(rrank[r]), int(crank[c])) for r, c in pts]
    return keys, rrank, crank


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


def _signature_from_points(pts: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    if not pts:
        return tuple()
    keys, _rr, _cc = _rank_keys(pts)
    return tuple(sorted((int(rk), int(ck)) for rk, ck in keys))


def _all_transformed_signatures(src_subs: list[dict], src_shape: tuple[int, int]) -> set[tuple[tuple[int, int], ...]]:
    H0, W0 = map(int, src_shape)
    sigs: set[tuple[tuple[int, int], ...]] = set()
    for k in range(4):
        for flip in (False, True):
            pts = [_transform_point(s["cent"], H0, W0, int(k), bool(flip)) for s in src_subs]
            sigs.add(_signature_from_points(pts))
    return sigs


def _match_transform(
    src_subs: list[dict], src_shape: tuple[int, int], dest_holes: list[dict]
) -> tuple[int, bool] | None:
    if not src_subs or not dest_holes or len(src_subs) != len(dest_holes):
        return None
    target_sig = _signature_from_points([h["cent"] for h in dest_holes])
    H0, W0 = map(int, src_shape)
    for k in range(4):
        for flip in (False, True):
            pts = [_transform_point(s["cent"], H0, W0, int(k), bool(flip)) for s in src_subs]
            if _signature_from_points(pts) == target_sig:
                return int(k), bool(flip)
    return None


def _tight_crop_non_bg(sub: np.ndarray, bg: int) -> np.ndarray:
    mask = sub != int(bg)
    pts = np.argwhere(mask)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return sub[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _collect_object_infos(grid: np.ndarray, bg: int) -> list[dict]:
    comps = _components(grid != int(bg))
    infos = []
    for comp in comps:
        obj = _extract_group_subgrid(grid, bg, [comp])
        holes = _enclosed_bg_components(obj, bg)
        subs = _colored_components(obj, bg)
        infos.append(
            {
                "grid": obj,
                "holes": holes,
                "subs": subs,
                "outline": _outline_col(obj, bg),
                "size": int(np.count_nonzero(obj != int(bg))),
                "hole_area": int(sum(int(h["size"]) for h in holes)),
                "bbox": tuple(int(v) for v in comp["bbox"]),
            }
        )
    return infos


def _leftmost_index(infos: list[dict]) -> int:
    return min(
        range(len(infos)),
        key=lambda i: (
            int(infos[i]["bbox"][1]),
            int(infos[i]["bbox"][0]),
            -int(infos[i]["size"]),
        ),
    )


def _combo_metrics(grid: np.ndarray) -> dict:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    infos = _collect_object_infos(g, bg)
    if not infos:
        return {"num_objects": 0}
    src_i = _leftmost_index(infos)
    src_subs = infos[src_i]["subs"]
    matches: list[int] = []
    for i, info in enumerate(infos):
        if i == src_i or info["subs"]:
            continue
        if _match_transform(src_subs, infos[src_i]["grid"].shape, info["holes"]) is not None:
            matches.append(i)
    true_dst_i = matches[0] if len(matches) == 1 else None
    distractor_areas = [int(infos[i]["hole_area"]) for i in range(len(infos)) if i != src_i and i != true_dst_i]
    return {
        "bg": bg,
        "infos": infos,
        "src_i": src_i,
        "match_indices": matches,
        "true_dst_i": true_dst_i,
        "num_objects": len(infos),
        "leftmost_has_subs": bool(infos[src_i]["subs"]),
        "distractor_areas": distractor_areas,
        "true_dst_hole_area": None if true_dst_i is None else int(infos[true_dst_i]["hole_area"]),
        "left_positions": [int(info["bbox"][1]) for info in infos],
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combo solver for D11/0 + D15/1.

    Primary role cue:
    - The leftmost object is src.

    Then apply D11 filtering:
    - Among the remaining empty frames, the true dst is the unique one whose hole
      centroid-rank signature matches src's colored subsprites under some rotation/flip.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    infos = _collect_object_infos(g, bg)
    assert len(infos) >= 2

    src_i = _leftmost_index(infos)
    if not infos[src_i]["subs"]:
        src_candidates = [i for i, info in enumerate(infos) if info["subs"]]
        assert src_candidates
        src_i = min(src_candidates, key=lambda i: (infos[i]["bbox"][1], infos[i]["bbox"][0]))
    src = infos[src_i]["grid"]
    src_subs = infos[src_i]["subs"]
    matched: list[tuple[int, tuple[int, bool]]] = []
    for i, info in enumerate(infos):
        if i == src_i or info["subs"]:
            continue
        match = _match_transform(src_subs, src.shape, info["holes"])
        if match is not None:
            matched.append((i, match))

    if matched:
        dest_i, best = min(
            matched,
            key=lambda t: (
                infos[t[0]]["bbox"][1],
                infos[t[0]]["bbox"][0],
                -infos[t[0]]["hole_area"],
            ),
        )
    else:
        fallback = [i for i in range(len(infos)) if i != src_i and not infos[i]["subs"]]
        if not fallback:
            fallback = [i for i in range(len(infos)) if i != src_i]
        assert fallback
        dest_i = max(fallback, key=lambda i: (infos[i]["hole_area"], infos[i]["size"]))
        best = _match_transform(src_subs, src.shape, infos[dest_i]["holes"])
        if best is None:
            best = (0, False)

    dest = infos[dest_i]["grid"].copy()
    dest_holes = infos[dest_i]["holes"]
    if not dest_holes or not src_subs:
        return _tight_crop_non_bg(dest, bg)

    dest_pts = [h["cent"] for h in dest_holes]
    _dest_keys, d_rr, d_cc = _rank_keys(dest_pts)
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


# --------------- generator (D11 distractor-frame variant) ---------------

CANVAS_SIZE_RANGE = (22, 30)
OBJECT_H_RANGE = (7, 17)
OBJECT_W_RANGE = (7, 13)

# Cavity structure
CAVITY_COUNT_RANGE = (4, 9)
FILL_COLOR_COUNT_RANGE = (3, 7)

# Layout row patterns (weighted choices)
ROW_HEIGHT_CHOICES = [1, 2, 2, 3, 3, 3]
LAYOUT_K_CHOICES_FULL = [1, 2, 2, 3, 3]
LAYOUT_K_CHOICES_SMALL = [1, 2, 2]
ROW_GAP_CHOICES = [1, 1, 2]

# Cavity slot widths per layout type
CENTERED_SLOT_WIDTHS = [3, 3, 5, 5, 7]
PAIR_SLOT_WIDTHS = [2, 2, 3, 3, 4]
SIDE_SLOT_WIDTHS = [2, 2, 3]
MID_SLOT_WIDTHS = [1, 3]
PAIR_AXIS_GAP_CHOICES = [1, 1, 3]
TRIO_MID_SIDE_GAP = 1

# Shape mutation
NONRECT_PROBABILITY = 0.8
MAX_MUTATE_ATTEMPTS = 40

# Color assignment
COLOR_MISMATCH_RATE = 0.10

# D11 distractors
DISTRACTOR_COUNT_CHOICES = [1, 2]
MAX_DISTRACTOR_ATTEMPTS = 200
DISTRACTOR_AREA_RATIO_RANGE = (0.70, 1.35)

# Placement
PLACEMENT_BORDER = 1
PLACEMENT_PADDING = 1
CANVAS_PLACEMENT_ATTEMPTS = 30
PLACEMENT_SAMPLE_ATTEMPTS = 80
SRC_MAX_LEFT_FRACTION = 0.35
FRAME_MIN_LEFT_SHIFT = 2

# Generation retry
MAX_GEN_ATTEMPTS = 300
MAX_LAYOUT_R_BUDGET = 24


def _random_rect_shape(h: int, w: int) -> np.ndarray:
    return np.ones((h, w), dtype=bool)


def _mutate_shape(mask: np.ndarray, *, n_remove_max: int, symmetric: bool = False) -> np.ndarray:
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
        _lab, n_components = label_mask(m2, connectivity=4)
        if int(n_components) == 1:
            m = m2
            removed += len(to_remove)
    if not m.any():
        return mask.copy()
    return m


def _gen_shape(h: int, w: int, *, nonrect: bool, symmetric: bool = False) -> np.ndarray:
    m = _random_rect_shape(h, w)
    if nonrect and h * w >= 3:
        limit = max(1, (h * w) // 2)
        m = _mutate_shape(m, n_remove_max=limit, symmetric=symmetric)
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
    layout: list[dict],
    shapes: list[np.ndarray],
    colors: list[int],
    bg: int,
    outline_col: int,
    *,
    filled: bool,
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
    r0b, c0b, r1b, c1b = b
    filled_cropped = filled_grid[r0b : r1b + 1, c0b : c1b + 1]
    return cropped, filled_cropped


def _count_4conn_cavities(filled_grid: np.ndarray, bg: int, outline_col: int) -> int:
    mask = (filled_grid != bg) & (filled_grid != outline_col)
    _lab, n = label_mask(mask, connectivity=4)
    return int(n)


def _count_enclosed_bg_holes(empty_grid: np.ndarray, bg: int) -> int:
    mask = empty_grid == bg
    lab, n = label_mask(mask, connectivity=4)
    H, W = empty_grid.shape
    count = 0
    for cid in range(1, int(n) + 1):
        pts = np.argwhere(lab == cid)
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        if r0 > 0 and c0 > 0 and r1 < H - 1 and c1 < W - 1:
            count += 1
    return count


def _is_8connected(grid: np.ndarray, bg: int) -> bool:
    mask = grid != bg
    if not mask.any():
        return False
    _lab, n = label_mask(mask, connectivity=8)
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


def _build_object_pair(
    layout: list[dict], colors: list[int], bg: int, outline_col: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    n_cavities = len(layout)
    dst_shapes = [slot["shape_dst"] for slot in layout]
    src_shapes = [slot["shape_src"] for slot in layout]

    dst_cropped, dst_filled = _render_object(layout, dst_shapes, colors, bg, outline_col, filled=False)
    if dst_cropped is None:
        return None
    if not _is_8connected(dst_cropped, bg):
        return None
    if _count_enclosed_bg_holes(dst_cropped, bg) != n_cavities:
        return None
    if not _size_in_spec(dst_cropped):
        return None

    src_cropped, _filled_unused = _render_object(layout, src_shapes, colors, bg, outline_col, filled=True)
    if src_cropped is None:
        return None
    if not _is_8connected(src_cropped, bg):
        return None
    if _count_4conn_cavities(src_cropped, bg, outline_col) != n_cavities:
        return None
    if not _size_in_spec(src_cropped):
        return None
    if _match_transform(
        _colored_components(src_cropped, bg),
        src_cropped.shape,
        _enclosed_bg_components(dst_cropped, bg),
    ) is None:
        return None

    return src_cropped, dst_cropped, dst_filled


def _build_empty_frame(layout: list[dict], colors: list[int], bg: int, outline_col: int) -> np.ndarray | None:
    shapes = [slot["shape_dst"] for slot in layout]
    empty_cropped, _filled = _render_object(layout, shapes, colors, bg, outline_col, filled=False)
    if empty_cropped is None:
        return None
    if not _is_8connected(empty_cropped, bg):
        return None
    if _count_enclosed_bg_holes(empty_cropped, bg) != len(layout):
        return None
    if not _size_in_spec(empty_cropped):
        return None
    return empty_cropped


def _hole_area(grid: np.ndarray, bg: int) -> int:
    return int(sum(int(h["size"]) for h in _enclosed_bg_components(grid, bg)))


def _build_distractors(
    src: np.ndarray,
    dst: np.ndarray,
    n_distractors: int,
    colors: list[int],
    bg: int,
    outline_col: int,
) -> list[np.ndarray] | None:
    src_subs = _colored_components(src, bg)
    src_sigs = _all_transformed_signatures(src_subs, src.shape)
    target_holes = len(src_subs)
    target_area = max(1, _hole_area(dst, bg))
    distractors: list[np.ndarray] = []
    seen_sigs: set[tuple[tuple[int, int], ...]] = set()

    for _idx in range(n_distractors):
        made = None
        for _attempt in range(MAX_DISTRACTOR_ATTEMPTS):
            layout = _design_layout(target_holes)
            if len(layout) != target_holes:
                continue
            candidate = _build_empty_frame(layout, colors, bg, outline_col)
            if candidate is None:
                continue
            sig = _signature_from_points([h["cent"] for h in _enclosed_bg_components(candidate, bg)])
            if sig in src_sigs or sig in seen_sigs:
                continue
            area = _hole_area(candidate, bg)
            ratio = float(area) / float(target_area)
            if not (DISTRACTOR_AREA_RATIO_RANGE[0] <= ratio <= DISTRACTOR_AREA_RATIO_RANGE[1]):
                continue
            made = candidate
            seen_sigs.add(sig)
            break
        if made is None:
            return None
        distractors.append(made)
    if max((_hole_area(d, bg) for d in distractors), default=0) < target_area:
        return None
    return distractors


def _rect_has_gap(rect_a: tuple[int, int, int, int], rect_b: tuple[int, int, int, int], padding: int) -> bool:
    ar0, ac0, ar1, ac1 = map(int, rect_a)
    br0, bc0, br1, bc1 = map(int, rect_b)
    return (
        ar1 + int(padding) < br0
        or br1 + int(padding) < ar0
        or ac1 + int(padding) < bc0
        or bc1 + int(padding) < ac0
    )


def _sample_rect_position(
    canvas_h: int,
    canvas_w: int,
    patch: np.ndarray,
    min_left: int,
    placed_rects: list[tuple[int, int, int, int]],
) -> tuple[int, int] | None:
    h, w = map(int, patch.shape)
    if h + 2 * PLACEMENT_BORDER > canvas_h or w + 2 * PLACEMENT_BORDER > canvas_w:
        return None
    lo_c = max(int(PLACEMENT_BORDER), int(min_left))
    hi_c = int(canvas_w - w - PLACEMENT_BORDER)
    hi_r = int(canvas_h - h - PLACEMENT_BORDER)
    if lo_c > hi_c or hi_r < int(PLACEMENT_BORDER):
        return None
    for _ in range(PLACEMENT_SAMPLE_ATTEMPTS):
        r0 = random.randint(int(PLACEMENT_BORDER), hi_r)
        c0 = random.randint(lo_c, hi_c)
        rect = (r0, c0, r0 + h - 1, c0 + w - 1)
        if all(_rect_has_gap(rect, prev, PLACEMENT_PADDING) for prev in placed_rects):
            return r0, c0
    return None


def _place_on_canvas(src: np.ndarray, dst: np.ndarray, distractors: list[np.ndarray], bg: int) -> np.ndarray | None:
    frames = [dst, *distractors]
    for _ in range(CANVAS_PLACEMENT_ATTEMPTS):
        canvas_h = random.randint(*CANVAS_SIZE_RANGE)
        canvas_w = random.randint(*CANVAS_SIZE_RANGE)
        src_h, src_w = map(int, src.shape)
        src_max_left = min(
            int(canvas_w - src_w - PLACEMENT_BORDER),
            int(canvas_w * SRC_MAX_LEFT_FRACTION),
        )
        if src_max_left < int(PLACEMENT_BORDER):
            continue
        canvas = np.full((canvas_h, canvas_w), bg, dtype=int)
        src_r = random.randint(int(PLACEMENT_BORDER), int(canvas_h - src_h - PLACEMENT_BORDER))
        src_c = random.randint(int(PLACEMENT_BORDER), src_max_left)
        paste(canvas, src, src_r, src_c, transparent=bg)
        placed_rects = [(src_r, src_c, src_r + src_h - 1, src_c + src_w - 1)]

        ok = True
        shuffled_frames = list(frames)
        random.shuffle(shuffled_frames)
        min_left = src_c + FRAME_MIN_LEFT_SHIFT
        for patch in shuffled_frames:
            pos = _sample_rect_position(canvas_h, canvas_w, patch, min_left, placed_rects)
            if pos is None:
                ok = False
                break
            r, c = pos
            paste(canvas, patch, r, c, transparent=bg)
            h, w = map(int, patch.shape)
            placed_rects.append((r, c, r + h - 1, c + w - 1))
        if ok:
            return canvas
    return None


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

    n_distractors = random.choice(DISTRACTOR_COUNT_CHOICES)
    distractors = _build_distractors(src, dst, n_distractors, colors, bg, outline_col)
    if distractors is None:
        return None

    canvas = _place_on_canvas(src, dst, distractors, bg)
    if canvas is None:
        return None

    metrics = _combo_metrics(canvas)
    if not metrics.get("leftmost_has_subs", False):
        return None
    if len(metrics.get("match_indices", [])) != 1:
        return None
    true_dst_area = metrics.get("true_dst_hole_area")
    if true_dst_area is None:
        return None
    if max(metrics.get("distractor_areas", [0])) < int(true_dst_area):
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
