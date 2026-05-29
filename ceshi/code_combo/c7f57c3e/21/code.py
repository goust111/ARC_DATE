# Combo variant of c7f57c3e (index 21)
# combo_spec: D11/2+D14/1
# Name: 精灵间背景填充 + 交换路径可视化

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


# Design analysis
# ===============
# combo_rule:
#   preserved_core:
#     - 仍然只有两类多色精灵，仍然依靠 color-1 锚框恢复尺度并完成 A/B 互换。
#     - 互换后的精灵仍留在原配对位置上，因此“对象识别”部分和原题同源。
#   fused_change:
#     - D11/2 要求把配对精灵之间原本留白的背景 corridor 填成两个原型共有的颜色。
#     - D14/1 要求再在同一条 corridor 上画出交换路径；为了让双重效果都可见，路径不再占满整条带，
#       而是叠在 corridor 中央的一条单像素线，并按左右/上下来源拆成两段颜色。
#   concrete_trace:
#     - 旧题：同尺度的一对 A/B 只在原地换型，二者之间仍是纯背景。
#     - 本变体：先完成互换；若该对水平相对，则把两者输出 bbox 之间的整块竖向走廊填成 shared color，
#       再在走廊中央那一行画一条两段式水平传输线。竖直相对时做对称处理。
#     - 差异不是“多画一个独立标记”，而是把 D11 的面状填充与 D14 的线状路径叠在同一 corridor 上。
#   complexity_calibration:
#     - 如果仍允许 3-4 个对象自由散布，solver 无法知道哪两个对象之间该出现 corridor。
#     - 因此沿用 D14 的收紧方案：恰好两个尺度，每个尺度恰好一对 A/B，且每对只能水平或竖直对齐。
#     - 共有色固定为“两个原型唯一共享的非锚色”；传输色固定为“唯一与锚框 4 邻接的变量色”。
#       两种颜色来源都能直接从对象自身观察到，不依赖颜色编号排序。
#   implicit_assumptions_review:
#     - 原 solver 默认“清空旧对象并贴上新对象就结束”；现在必须在互换之后再处理 pair corridor。
#     - 原 generator 默认对象间只需互不重叠；现在还要保证同尺度 pair 之间存在一条干净 corridor，
#       并且这条 corridor 在输出中不会与别的 pair 的填充/路径相撞。
#     - 轨迹与填充若使用同一颜色，融合会退化成单层效果；因此 generator 强制：
#         1. 两个 transfer 色彼此不同；
#         2. shared fill 色与两个 transfer 色都不同；
#         3. shared fill 是两个原型唯一共享的非锚色。
#     - 若 corridor 只有 1 像素厚，中央路径会完全遮住填充；因此模板和配对几何都选成
#       “输出 bbox 在连接轴的垂直方向至少重叠 3 个像素（scale=1 时）”，放大后更明显。
#   legality_and_visibility:
#     - pair patch 在局部先同时容纳输入对象、交换后的输出对象和中间 corridor，再整体随机放入画布。
#     - reserve 直接使用输出 patch 的非背景区域，所以未来会被填充/画线的位置也提前占位。
#     - 输出视觉上分三层：精灵本体、shared-color corridor、中央双向路径线；三层边界清楚，不会糊在一起。


def _pick_anchor_bbox(arr: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    if np.any(arr == 1):
        ys, xs = np.where(arr == 1)
        return int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())
    assert False, "sprite array has no color-1 pixels"


def _scale_nn(arr: np.ndarray, scale: int) -> np.ndarray:
    if scale <= 1:
        return arr.copy()
    return np.repeat(np.repeat(arr, scale, axis=0), scale, axis=1)


def _non_bg_colors(arr: np.ndarray, bg: int) -> set[int]:
    return {int(v) for v in np.unique(arr) if int(v) != int(bg)}


def _transfer_color(arr: np.ndarray, bg: int) -> int:
    """Return the unique non-bg, non-1 color that touches the anchor in 4-neighborhood."""
    h, w = arr.shape
    colors: set[int] = set()
    for r, c in np.argwhere(arr == 1):
        rr = int(r)
        cc = int(c)
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr = rr + dr
            nc = cc + dc
            if 0 <= nr < h and 0 <= nc < w:
                v = int(arr[nr, nc])
                if v not in (bg, 1):
                    colors.add(v)
    assert len(colors) == 1, "expected exactly one anchor-adjacent transfer color"
    return next(iter(colors))


def _shared_fill_color(proto_a: np.ndarray, proto_b: np.ndarray, bg: int) -> int:
    shared = (_non_bg_colors(proto_a, bg) & _non_bg_colors(proto_b, bg)) - {1}
    assert len(shared) == 1, "expected exactly one shared non-anchor color"
    return next(iter(shared))


def _infer_scale(arr: np.ndarray, bg: int, base_anc_h: int, base_anc_w: int) -> tuple[int, tuple[int, int, int, int]]:
    anc = _pick_anchor_bbox(arr, bg)
    ar0, ac0, ar1, ac1 = anc
    inst_anc_h = int(ar1 - ar0 + 1)
    inst_anc_w = int(ac1 - ac0 + 1)
    assert inst_anc_h % max(1, base_anc_h) == 0
    assert inst_anc_w % max(1, base_anc_w) == 0
    scale_h = max(1, inst_anc_h // max(1, base_anc_h))
    scale_w = max(1, inst_anc_w // max(1, base_anc_w))
    assert scale_h == scale_w, "anchor scale must be isotropic"
    return int(scale_h), anc


def _match_proto(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    anc: tuple[int, int, int, int],
    scale: int,
    protos: list[np.ndarray],
    bg: int,
) -> int:
    """Return the best matching prototype index for a sprite instance."""
    h, w = grid.shape
    r0, c0, _r1, _c1 = bbox
    ar0, ac0, _ar1, _ac1 = anc

    best_idx = 0
    best_score = -1
    for idx, proto in enumerate(protos):
        scaled = _scale_nn(proto, scale)
        pr0, pc0, _pr1, _pc1 = _pick_anchor_bbox(proto, bg)
        top = int((r0 + ar0) - pr0 * scale)
        left = int((c0 + ac0) - pc0 * scale)

        ph, pw = scaled.shape
        rr0 = max(0, top)
        cc0 = max(0, left)
        rr1 = min(h, top + ph)
        cc1 = min(w, left + pw)
        if rr0 >= rr1 or cc0 >= cc1:
            continue

        sub_g = grid[rr0:rr1, cc0:cc1]
        sub_p = scaled[(rr0 - top) : (rr1 - top), (cc0 - left) : (cc1 - left)]
        m = sub_p != bg
        score = int(np.sum(sub_g[m] == sub_p[m])) if np.any(m) else 0
        if score > best_score:
            best_score = score
            best_idx = int(idx)

    return best_idx


def _draw_pair_connection(out: np.ndarray, a: dict, b: dict, protos: list[np.ndarray], bg: int) -> None:
    """Fill the pair corridor with the shared color, then overlay the bidirectional path line."""
    shared = _shared_fill_color(protos[0], protos[1], bg)

    if a["output_bbox"][1] <= b["output_bbox"][1]:
        left, right = a, b
    else:
        left, right = b, a
    h_gap = int(right["output_bbox"][1] - left["output_bbox"][3] - 1)
    if h_gap > 0 and left["anchor_global"][0] == right["anchor_global"][0] and left["anchor_global"][2] == right["anchor_global"][2]:
        r0 = min(int(left["output_bbox"][0]), int(right["output_bbox"][0]))
        r1 = max(int(left["output_bbox"][2]), int(right["output_bbox"][2]))
        c0 = int(left["output_bbox"][3] + 1)
        c1 = int(right["output_bbox"][1] - 1)
        out[r0 : r1 + 1, c0 : c1 + 1] = int(shared)
        mid_r = (int(left["anchor_global"][0]) + int(left["anchor_global"][2])) // 2
        split = (c0 + c1) // 2
        left_color = _transfer_color(protos[left["type"]], bg)
        right_color = _transfer_color(protos[right["type"]], bg)
        out[mid_r, c0 : split + 1] = int(left_color)
        if split + 1 <= c1:
            out[mid_r, split + 1 : c1 + 1] = int(right_color)
        return

    if a["output_bbox"][0] <= b["output_bbox"][0]:
        top_obj, bot_obj = a, b
    else:
        top_obj, bot_obj = b, a
    v_gap = int(bot_obj["output_bbox"][0] - top_obj["output_bbox"][2] - 1)
    if v_gap > 0 and top_obj["anchor_global"][1] == bot_obj["anchor_global"][1] and top_obj["anchor_global"][3] == bot_obj["anchor_global"][3]:
        c0 = min(int(top_obj["output_bbox"][1]), int(bot_obj["output_bbox"][1]))
        c1 = max(int(top_obj["output_bbox"][3]), int(bot_obj["output_bbox"][3]))
        r0 = int(top_obj["output_bbox"][2] + 1)
        r1 = int(bot_obj["output_bbox"][0] - 1)
        out[r0 : r1 + 1, c0 : c1 + 1] = int(shared)
        mid_c = (int(top_obj["anchor_global"][1]) + int(top_obj["anchor_global"][3])) // 2
        split = (r0 + r1) // 2
        top_color = _transfer_color(protos[top_obj["type"]], bg)
        bot_color = _transfer_color(protos[bot_obj["type"]], bg)
        out[r0 : split + 1, mid_c] = int(top_color)
        if split + 1 <= r1:
            out[split + 1 : r1 + 1, mid_c] = int(bot_color)
        return

    assert False, "pair corridor must be horizontal or vertical with positive gap and overlap"


def _pair_connection_valid(a: dict, b: dict) -> bool:
    if a["output_bbox"][1] <= b["output_bbox"][1]:
        left, right = a, b
    else:
        left, right = b, a
    h_gap = int(right["output_bbox"][1] - left["output_bbox"][3] - 1)
    if h_gap > 0 and left["anchor_global"][0] == right["anchor_global"][0] and left["anchor_global"][2] == right["anchor_global"][2]:
        return True

    if a["output_bbox"][0] <= b["output_bbox"][0]:
        top_obj, bot_obj = a, b
    else:
        top_obj, bot_obj = b, a
    v_gap = int(bot_obj["output_bbox"][0] - top_obj["output_bbox"][2] - 1)
    if v_gap > 0 and top_obj["anchor_global"][1] == bot_obj["anchor_global"][1] and top_obj["anchor_global"][3] == bot_obj["anchor_global"][3]:
        return True

    return False


def _solve(grid: np.ndarray, decorate: bool) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    comps = components_mask(g != bg, connectivity=8)
    assert len(comps) >= 2, "grid must contain sprite components"

    sprites = []
    for pts in comps:
        r0, c0, r1, c1 = bbox_coords(pts)
        arr = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        sprites.append({"pts": pts, "bbox": (r0, c0, r1, c1), "arr": arr})

    areas = [int((s["bbox"][2] - s["bbox"][0] + 1) * (s["bbox"][3] - s["bbox"][1] + 1)) for s in sprites]
    min_area = min(areas)
    base = [s for s, a in zip(sprites, areas, strict=False) if a == min_area]
    uniq: dict[tuple[int, ...], np.ndarray] = {}
    for s in base:
        key = tuple(s["arr"].ravel().tolist())
        if key not in uniq:
            uniq[key] = s["arr"]
    assert len(uniq) == 2, "smallest-area sprites must yield exactly 2 distinct prototypes"
    protos = [uniq[key] for key in list(uniq.keys())[:2]]

    anc0 = _pick_anchor_bbox(protos[0], bg)
    anc1 = _pick_anchor_bbox(protos[1], bg)
    assert (anc0[2] - anc0[0], anc0[3] - anc0[1]) == (anc1[2] - anc1[0], anc1[3] - anc1[1])
    base_anc_h = int(anc0[2] - anc0[0] + 1)
    base_anc_w = int(anc0[3] - anc0[1] + 1)

    out = g.copy()
    for s in sprites:
        for rr, cc in s["pts"]:
            out[int(rr), int(cc)] = int(bg)

    classified = []
    for s in sprites:
        scale, anc = _infer_scale(s["arr"], bg, base_anc_h, base_anc_w)
        best_idx = _match_proto(g, s["bbox"], anc, scale, protos, bg)
        repl_idx = 1 - best_idx

        p = protos[repl_idx]
        sp = _scale_nn(p, scale)
        pr0, pc0, _pr1, _pc1 = _pick_anchor_bbox(p, bg)
        r0, c0, _r1, _c1 = s["bbox"]
        ar0, ac0, _ar1, _ac1 = anc
        top = int((r0 + ar0) - pr0 * scale)
        left = int((c0 + ac0) - pc0 * scale)
        paste(out, sp, top, left, transparent=bg)

        classified.append(
            {
                "type": int(best_idx),
                "scale": int(scale),
                "output_bbox": (int(top), int(left), int(top + sp.shape[0] - 1), int(left + sp.shape[1] - 1)),
                "anchor_global": (
                    int(r0 + anc[0]),
                    int(c0 + anc[1]),
                    int(r0 + anc[2]),
                    int(c0 + anc[3]),
                ),
            }
        )

    if decorate:
        by_scale: dict[int, list[dict]] = {}
        for rec in classified:
            by_scale.setdefault(int(rec["scale"]), []).append(rec)

        for scale, group in by_scale.items():
            assert len(group) == 2, f"scale {scale} must contain exactly one A/B pair"
            types = {int(group[0]["type"]), int(group[1]["type"])}
            assert types == {0, 1}, f"scale {scale} pair must contain one sprite of each prototype"
            _draw_pair_connection(out, group[0], group[1], protos, bg)

    return out


def _swap_only(grid: np.ndarray) -> np.ndarray:
    return _solve(grid, decorate=False)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap every sprite to the opposite prototype, then turn each paired gap into a filled exchange corridor.

    Input structure:
    - The grid contains exactly two prototype families built from the same shape logic, with color 1 marking the anchor region.
    - Sprites appear in two same-scale A/B pairs; each pair is arranged either left-right or top-bottom with a clean empty corridor between them.

    Rule:
    - Recover the two prototypes from the smallest sprites, infer each instance's scale from the anchor box, and replace every sprite with the other prototype at the same anchored position.
    - For each same-scale pair, fill the corridor between the swapped sprites using the unique non-anchor color shared by both prototypes, then overlay a central two-part path line using the pair's two transfer colors.

    Key insight:
    - The anchor box still solves the original identification problem, while the tightened pair geometry makes the new behavior deterministic: once same-scale partners are known, the extra work is just “swap first, then decorate the unique corridor between them with shared-fill area plus a split transfer line”.
    """
    return _solve(grid, decorate=True)


# --------------- generator ---------------

# --- CFG: canvas ---
GRID_SIZE_MIN = 22
GRID_SIZE_MAX = 28
GRID_MARGIN_FOR_PAIRS = 4

# --- CFG: prototypes ---
BG_CHOICES = [0, 2, 3, 4, 5, 6, 7, 8, 9]
LABEL_TEMPLATES = [
    np.array(
        [
            [0, 2, 0],
            [1, 1, 1],
            [0, 2, 0],
            [3, 3, 3],
        ],
        dtype=int,
    ),
    np.array(
        [
            [0, 2, 2, 0],
            [1, 1, 1, 1],
            [0, 2, 2, 0],
            [3, 0, 0, 3],
        ],
        dtype=int,
    ),
    np.array(
        [
            [0, 2, 2, 0],
            [1, 1, 1, 1],
            [0, 2, 2, 0],
            [0, 2, 2, 0],
            [3, 0, 0, 3],
        ],
        dtype=int,
    ),
]

# --- CFG: pairs ---
SCALES = [1, 2]
PAIR_GAP_BY_SCALE = {1: [5, 6, 7], 2: [6, 7]}
ORIENTATIONS = ["h", "v"]

# --- CFG: placement ---
PLACEMENT_BORDER = 1
PLACEMENT_PADDING = 2
PLACEMENT_CONNECTIVITY = 8

# --- CFG: generation retries ---
MAX_RETRIES = 400


def _anchor_bbox(arr: np.ndarray) -> tuple[int, int, int, int]:
    rs, cs = np.where(arr == 1)
    return int(rs.min()), int(cs.min()), int(rs.max()), int(cs.max())


def _scale(arr: np.ndarray, k: int) -> np.ndarray:
    if k == 1:
        return arr.copy()
    return np.repeat(np.repeat(arr, k, axis=0), k, axis=1)


def _top_left_from_anchor(anc: tuple[int, int, int, int], anchor_r: int, anchor_c: int) -> tuple[int, int]:
    return int(anchor_r - anc[0]), int(anchor_c - anc[1])


def _make_prototypes(bg: int) -> tuple[np.ndarray, np.ndarray]:
    labels = random.choice(LABEL_TEMPLATES)
    pool = [c for c in range(10) if c not in (bg, 1)]
    transfer_a, transfer_b, shared_fill = random.sample(pool, 3)

    def _colorize(lab: np.ndarray, transfer: int) -> np.ndarray:
        proto = np.full(lab.shape, bg, dtype=int)
        proto[lab == 1] = 1
        proto[lab == 2] = int(transfer)
        proto[lab == 3] = int(shared_fill)
        return proto

    proto_a = _colorize(labels, transfer_a)
    proto_b = _colorize(np.flipud(labels), transfer_b)
    assert _transfer_color(proto_a, bg) != _transfer_color(proto_b, bg)
    assert _shared_fill_color(proto_a, proto_b, bg) == int(shared_fill)
    return proto_a, proto_b


def _pair_patch(
    protos: list[np.ndarray],
    bg: int,
    scale: int,
    orientation: str,
    first_type: int,
    gap: int,
) -> dict | None:
    second_type = 1 - first_type

    inp_a = _scale(protos[first_type], scale)
    inp_b = _scale(protos[second_type], scale)
    out_a = _scale(protos[second_type], scale)
    out_b = _scale(protos[first_type], scale)

    anc_a = _anchor_bbox(inp_a)
    anc_b = _anchor_bbox(inp_b)
    anc_out_a = _anchor_bbox(out_a)
    anc_out_b = _anchor_bbox(out_b)

    if orientation == "h":
        anchor_row_a = 0
        anchor_row_b = 0
        anchor_col_a = 0
        right_extent_a = max(inp_a.shape[1] - 1 - anc_a[1], out_a.shape[1] - 1 - anc_out_a[1])
        left_extent_b = max(anc_b[1], anc_out_b[1])
        anchor_col_b = int(anchor_col_a + right_extent_a + gap + left_extent_b + 1)
    else:
        anchor_col_a = 0
        anchor_col_b = 0
        anchor_row_a = 0
        bottom_extent_a = max(inp_a.shape[0] - 1 - anc_a[0], out_a.shape[0] - 1 - anc_out_a[0])
        top_extent_b = max(anc_b[0], anc_out_b[0])
        anchor_row_b = int(anchor_row_a + bottom_extent_a + gap + top_extent_b + 1)

    in_top_a, in_left_a = _top_left_from_anchor(anc_a, anchor_row_a, anchor_col_a)
    in_top_b, in_left_b = _top_left_from_anchor(anc_b, anchor_row_b, anchor_col_b)
    out_top_a, out_left_a = _top_left_from_anchor(anc_out_a, anchor_row_a, anchor_col_a)
    out_top_b, out_left_b = _top_left_from_anchor(anc_out_b, anchor_row_b, anchor_col_b)

    anchor_a = (int(anchor_row_a), int(anchor_col_a))
    anchor_b = (int(anchor_row_b), int(anchor_col_b))

    placements = [
        (inp_a, in_top_a, in_left_a),
        (inp_b, in_top_b, in_left_b),
        (out_a, out_top_a, out_left_a),
        (out_b, out_top_b, out_left_b),
    ]
    r_min = min(top for _arr, top, _left in placements)
    c_min = min(left for _arr, _top, left in placements)
    shift_r = -int(r_min)
    shift_c = -int(c_min)

    shifted = [(arr, top + shift_r, left + shift_c) for arr, top, left in placements]
    patch_h = max(top + arr.shape[0] for arr, top, _left in shifted)
    patch_w = max(left + arr.shape[1] for arr, _top, left in shifted)
    inp = np.full((patch_h, patch_w), bg, dtype=int)
    out = np.full((patch_h, patch_w), bg, dtype=int)

    paste(inp, inp_a, in_top_a + shift_r, in_left_a + shift_c, transparent=bg)
    paste(inp, inp_b, in_top_b + shift_r, in_left_b + shift_c, transparent=bg)
    paste(out, out_a, out_top_a + shift_r, out_left_a + shift_c, transparent=bg)
    paste(out, out_b, out_top_b + shift_r, out_left_b + shift_c, transparent=bg)

    rec_a = {
        "type": int(first_type),
        "output_bbox": (
            int(out_top_a + shift_r),
            int(out_left_a + shift_c),
            int(out_top_a + shift_r + out_a.shape[0] - 1),
            int(out_left_a + shift_c + out_a.shape[1] - 1),
        ),
        "anchor_global": (
            int(anchor_a[0] + shift_r),
            int(anchor_a[1] + shift_c),
            int(anchor_a[0] + shift_r + (anc_a[2] - anc_a[0])),
            int(anchor_a[1] + shift_c + (anc_a[3] - anc_a[1])),
        ),
    }
    rec_b = {
        "type": int(second_type),
        "output_bbox": (
            int(out_top_b + shift_r),
            int(out_left_b + shift_c),
            int(out_top_b + shift_r + out_b.shape[0] - 1),
            int(out_left_b + shift_c + out_b.shape[1] - 1),
        ),
        "anchor_global": (
            int(anchor_b[0] + shift_r),
            int(anchor_b[1] + shift_c),
            int(anchor_b[0] + shift_r + (anc_b[2] - anc_b[0])),
            int(anchor_b[1] + shift_c + (anc_b[3] - anc_b[1])),
        ),
    }
    if not _pair_connection_valid(rec_a, rec_b):
        return None
    _draw_pair_connection(out, rec_a, rec_b, protos, bg)

    reserve = np.where(out != bg, 1, bg)
    return {"input": inp, "output": out, "reserve": reserve}


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(BG_CHOICES)
    proto_a, proto_b = _make_prototypes(bg)
    protos = [proto_a, proto_b]

    pairs = []
    for scale in SCALES:
        gap = random.choice(PAIR_GAP_BY_SCALE[scale])
        orientation = random.choice(ORIENTATIONS)
        first_type = random.randrange(2)
        pair = _pair_patch(protos, bg, scale, orientation, first_type, gap)
        if pair is None:
            return None
        pairs.append(pair)

    pairs.sort(key=lambda item: item["reserve"].shape[0] * item["reserve"].shape[1], reverse=True)
    max_dim = max(max(item["reserve"].shape) for item in pairs)
    n_min = max(GRID_SIZE_MIN, max_dim + GRID_MARGIN_FOR_PAIRS)
    if n_min > GRID_SIZE_MAX:
        return None
    n = random.randint(max(n_min, GRID_SIZE_MAX - 2), GRID_SIZE_MAX)

    inp = np.full((n, n), bg, dtype=int)
    out = np.full((n, n), bg, dtype=int)
    reserve = np.full((n, n), bg, dtype=int)

    for pair in pairs:
        try:
            pr, pc = random_free_location(
                reserve,
                pair["reserve"],
                transparent=bg,
                border=PLACEMENT_BORDER,
                padding=PLACEMENT_PADDING,
                padding_connectivity=PLACEMENT_CONNECTIVITY,
            )
        except ValueError:
            return None

        paste(reserve, pair["reserve"], pr, pc, transparent=bg)
        paste(inp, pair["input"], pr, pc, transparent=bg)
        paste(out, pair["output"], pr, pc, transparent=bg)

    try:
        if np.array_equal(main(inp), out):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid c7f57c3e combo input")
