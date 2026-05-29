# Variant of b10624e5
# Dimension: D13 — 图案与锚块的空间关系 (meta)
# Change: 图案紧邻锚块放置（与锚块共享边或角），图案的锚定点是锚块的特定角/边... → 图案环绕锚块四周

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste


# Design analysis for the D13 variant
# -----------------------------------
# variant_spec:
#   source_task: none
#   core_change:
#     old: TL 象限里，彩色图案与红锚块是一个 4 连通组件，solver 直接取“含红像素的组件”即可。
#     new: TL 图案改成“围着锚块的四个侧片”，它们和红锚块之间保留背景留距；四个侧片互相也可断开。
#          因此，source motif 必须由“TL 象限全部非背景像素的包围框”定义，而不是由单个连通块定义。
#   concrete_layout:
#     grid: 27x27，divider 仍在 row=13 / col=13。
#     anchors: 四个象限各一个红色实心方块；TL 固定 2x2，其余尺寸改为 2..4。
#     TL motif:
#       - 最小模板固定为 6x6。
#       - 中央 2x2 是红锚块。
#       - 上/下/左/右四个彩色条分别位于模板外框四边。
#       - 红锚块与四个条片之间始终隔着 1 格背景，因此“环绕”可见，但不是旧题那种直接贴边。
#       - 四边颜色互异，镜像后左右/上下颜色交换，肉眼可见。
#   behavior_trace:
#     old_solver_path:
#       - 在 TL 象限取包含红色的 4 连通组件。
#       - 对本变体，该组件只会是 2x2 红锚块，四边条片会被漏掉。
#     new_solver_path:
#       - 在 TL 象限取全部非背景像素的 bbox，得到完整 6x6 环绕模板（含中间背景留距）。
#       - 对其他象限按锚块尺寸比缩放整个模板，并按象限做 LR / UD 反射。
#       - 将缩放后模板中的红色 bbox 对齐到目标锚块，再只盖非背景像素。
#   complexity_calibration:
#     - 四条边同时参与，避免退化成“只在一侧贴一个小块”。
#     - 四边颜色不同，反射规则在 2-3 组样例里就能看出。
#     - 目标尺寸限制为 2..4：如果允许 1x1，带留距的环绕模板缩小后会塌成不完整残片，D13 特征不稳定。
#   hidden_assumptions_review:
#     - 原 main 里 `components_mask(tl != bg)` + “选含红组件”依赖旧题的连通性；新变体不成立，必须改。
#     - 原 roundtrip 缩放断言依赖 motif 可被任意比例 NN 缩放后无损恢复；本变体保留背景留距，
#       在 2 -> 3 的比例下会发生可接受的别名，因此该断言必须移除。
#     - 原按红色 bbox 左上角对齐的放置方式仍然成立，因为模板中央红区始终是规则方块。
#   placement_legality:
#     - 6x6 模板在比例 1, 1.5, 2 下分别变成 6x6 / 9x9 / 12x12，都能完整放进 13x13 象限。
#     - generator 仅在“整块模板可完全落在象限内”的范围里选锚块位置，因此无越界、无裁切。
#   observability:
#     - TL 输入能直接看到“红块四周有四条彩边，但中间留空”的新关系。
#     - 输出中 TR 会交换左右颜色，BL 会交换上下颜色，BR 两者都交换；人眼可直接比较。
#   cleanliness:
#     - 每个非 TL 象限在输入里只剩红锚块，输出时再添四边条片；图层关系单一，不会杂乱覆盖。
#   color_inference:
#     - 不做颜色映射或排序；颜色只作为可见的侧向标签使用，规则由位置与镜像决定。


BG = 4
DIV = 1
RED = 2
GRID_SIZE = 27
MID = 13
QUADS = {
    (0, 0): (0, 12, 0, 12),
    (0, 1): (0, 12, 14, 26),
    (1, 0): (14, 26, 0, 12),
    (1, 1): (14, 26, 14, 26),
}

SOURCE_ANCHOR_SIZE = 2
TARGET_SIZE_OPTIONS = (2, 3, 4)
COLOR_POOL = [0, 3, 5, 6, 7, 8, 9]
MAX_RETRIES = 300


def _find_dividers(g: np.ndarray, bg: int) -> tuple[int, int]:
    """
    Return the unique full-row/full-col quadrant divider: (mid_row, mid_col).
    """
    h, w = g.shape
    full_rows = [(r, int(g[r, 0])) for r in range(h) if np.all(g[r, :] == g[r, 0]) and int(g[r, 0]) != bg]
    full_cols = [(c, int(g[0, c])) for c in range(w) if np.all(g[:, c] == g[0, c]) and int(g[0, c]) != bg]
    matches = []
    for r, vr in full_rows:
        for c, vc in full_cols:
            if vr == vc:
                matches.append((int(r), int(c)))
    assert len(matches) == 1, "grid must have exactly one matching full-row/full-col divider"
    return matches[0]


def _resize_nn(img: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    src_h, src_w = img.shape
    rr = np.clip(np.floor(np.arange(new_h) * src_h / new_h).astype(int), 0, src_h - 1)
    cc = np.clip(np.floor(np.arange(new_w) * src_w / new_w).astype(int), 0, src_w - 1)
    return img[rr[:, None], cc[None, :]]


def _red_bbox(img: np.ndarray) -> tuple[int, int, int, int]:
    rr, cc = np.where(img == RED)
    assert rr.size > 0, "template must contain red anchor pixels"
    return int(rr.min()), int(cc.min()), int(rr.max()), int(cc.max())


def _surrounds_anchor(patch: np.ndarray, bg: int) -> bool:
    r0, c0, r1, c1 = _red_bbox(patch)
    top = np.any(patch[:r0, :] != bg)
    bottom = np.any(patch[r1 + 1 :, :] != bg)
    left = np.any(patch[:, :c0] != bg)
    right = np.any(patch[:, c1 + 1 :] != bg)
    return bool(top and bottom and left and right)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy an anchor-centered surround template into the other quadrants using size-coded scaling and quadrant reflections.

    Input structure:
    - A full-row/full-column divider splits the grid into four quadrants.
    - Each quadrant contains exactly one red square anchor block.
    - The top-left quadrant contains the only full motif: four colored side pieces arranged around the red anchor inside one larger bbox, with background left between them and the anchor.

    Rule:
    - Take the entire non-background bbox in the top-left quadrant as the source template, not just the red-connected component.
    - Reflect that template left-right for right quadrants and up-down for bottom quadrants.
    - Scale the whole template by the ratio target_anchor_size / source_anchor_size.
    - Align the scaled template's red region to the target anchor and stamp its non-background pixels.

    Key insight: The transferable object is an anchor-centered surrounding layout, so the solver must preserve the whole bbox around the anchor rather than only the pixels connected to it.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    bg = get_bg(g)
    mid_r, mid_c = _find_dividers(g, bg)

    red = components_mask(g == RED, connectivity=4)
    assert red, "grid must contain red anchor blocks"

    anchors: dict[tuple[int, int], tuple[int, int, int]] = {}
    for pts in red:
        r0, c0, r1, c1 = bbox_coords(pts)
        rr = (r0 + r1) // 2
        cc = (c0 + c1) // 2
        qb = 0 if rr < mid_r else 1
        qr = 0 if cc < mid_c else 1
        size = max(r1 - r0 + 1, c1 - c0 + 1)
        key = (qb, qr)
        assert key not in anchors, "each quadrant must contain exactly one red anchor block"
        anchors[key] = (int(r0), int(c0), int(size))

    assert set(anchors) == set(QUADS), "there must be one red anchor in every quadrant"
    sr0, sc0, sL = anchors[(0, 0)]
    assert sL == SOURCE_ANCHOR_SIZE, "top-left red anchor must be 2x2 in this variant"

    tl = g[0:mid_r, 0:mid_c]
    tl_pts = np.argwhere(tl != bg)
    assert tl_pts.size > 0, "top-left quadrant must contain the source template"
    pr0, pc0, pr1, pc1 = bbox_coords(tl_pts)
    patch = tl[pr0 : pr1 + 1, pc0 : pc1 + 1].copy()

    assert _surrounds_anchor(patch, bg), "source template must surround the anchor from all four sides"
    rr_red, cc_red = np.where(patch == RED)
    assert int(rr_red.min()) == sr0 - pr0 and int(cc_red.min()) == sc0 - pc0

    for (qb, qr), (tr0, tc0, tL) in anchors.items():
        if (qb, qr) == (0, 0):
            continue

        patch_t = patch
        if qr == 1:
            patch_t = np.fliplr(patch_t)
        if qb == 1:
            patch_t = np.flipud(patch_t)

        ratio = float(tL) / float(sL)
        new_h = max(1, int(round(patch_t.shape[0] * ratio)))
        new_w = max(1, int(round(patch_t.shape[1] * ratio)))
        patch_s = _resize_nn(patch_t, new_h, new_w)

        rr_red, cc_red = np.where(patch_s == RED)
        assert rr_red.size > 0, "scaled template must retain a red anchor region"
        ar0, ac0 = int(rr_red.min()), int(cc_red.min())
        place_r0 = tr0 - ar0
        place_c0 = tc0 - ac0

        rr_p, cc_p = np.where(patch_s != bg)
        for rr0, cc0 in zip(rr_p.tolist(), cc_p.tolist(), strict=False):
            val = int(patch_s[rr0, cc0])
            rr_t = place_r0 + rr0
            cc_t = place_c0 + cc0
            if not (0 <= rr_t < h and 0 <= cc_t < w):
                continue
            if rr_t == mid_r or cc_t == mid_c:
                continue
            if out[rr_t, cc_t] == RED:
                continue
            out[rr_t, cc_t] = val

    return out


def _scaled_patch(patch: np.ndarray, qb: int, qr: int, target_size: int) -> np.ndarray:
    p = patch
    if qr:
        p = np.fliplr(p)
    if qb:
        p = np.flipud(p)
    ratio = target_size / SOURCE_ANCHOR_SIZE
    h = max(1, int(round(p.shape[0] * ratio)))
    w = max(1, int(round(p.shape[1] * ratio)))
    return _resize_nn(p, h, w)


def _source_patch() -> tuple[np.ndarray, int, int]:
    top, bottom, left, right = random.sample(COLOR_POOL, 4)
    patch = np.full((6, 6), BG, dtype=int)

    patch[2:4, 2:4] = RED
    patch[0, 1:5] = top
    patch[5, 1:5] = bottom
    patch[1:5, 0] = left
    patch[1:5, 5] = right

    corners = {
        (0, 0): random.choice((BG, top, left)),
        (0, 5): random.choice((BG, top, right)),
        (5, 0): random.choice((BG, bottom, left)),
        (5, 5): random.choice((BG, bottom, right)),
    }
    for (r, c), val in corners.items():
        patch[r, c] = val

    assert _surrounds_anchor(patch, BG)
    return patch, 2, 2


def _choose_anchor(qb: int, qr: int, patch_s: np.ndarray) -> tuple[int, int]:
    q_r0, q_r1, q_c0, q_c1 = QUADS[(qb, qr)]
    rr, cc = np.where(patch_s == RED)
    red_r0, red_c0 = int(rr.min()), int(cc.min())

    fit_r0 = q_r0 + red_r0
    fit_r1 = q_r1 - patch_s.shape[0] + 1 + red_r0
    fit_c0 = q_c0 + red_c0
    fit_c1 = q_c1 - patch_s.shape[1] + 1 + red_c0

    assert fit_r0 <= fit_r1 and fit_c0 <= fit_c1, "scaled surround template must fit its quadrant"
    return random.randint(fit_r0, fit_r1), random.randint(fit_c0, fit_c1)


def _stamp(out: np.ndarray, patch_s: np.ndarray, anchor_r: int, anchor_c: int) -> None:
    rr_red, cc_red = np.where(patch_s == RED)
    place_r0 = anchor_r - int(rr_red.min())
    place_c0 = anchor_c - int(cc_red.min())

    src_rr, src_cc = np.where(patch_s != BG)
    tgt_rr = src_rr + place_r0
    tgt_cc = src_cc + place_c0

    valid = (
        (tgt_rr >= 0)
        & (tgt_rr < GRID_SIZE)
        & (tgt_cc >= 0)
        & (tgt_cc < GRID_SIZE)
        & (tgt_rr != MID)
        & (tgt_cc != MID)
    )
    tgt_rr, tgt_cc = tgt_rr[valid], tgt_cc[valid]
    src_rr, src_cc = src_rr[valid], src_cc[valid]
    not_red = out[tgt_rr, tgt_cc] != RED
    out[tgt_rr[not_red], tgt_cc[not_red]] = patch_s[src_rr[not_red], src_cc[not_red]]


def _target_sizes() -> dict[tuple[int, int], int]:
    while True:
        vals = [random.choice(TARGET_SIZE_OPTIONS) for _ in range(3)]
        if len(set(vals)) >= 2:
            return {
                (0, 0): SOURCE_ANCHOR_SIZE,
                (0, 1): vals[0],
                (1, 0): vals[1],
                (1, 1): vals[2],
            }


def _gen_attempt() -> np.ndarray | None:
    target_sizes = _target_sizes()
    patch, anchor_r_in_patch, anchor_c_in_patch = _source_patch()

    scaled = {
        key: _scaled_patch(patch, key[0], key[1], size)
        for key, size in target_sizes.items()
        if key != (0, 0)
    }

    anchors: dict[tuple[int, int], tuple[int, int]] = {}
    tl_r0 = random.randint(0, MID - patch.shape[0])
    tl_c0 = random.randint(0, MID - patch.shape[1])
    anchors[(0, 0)] = (tl_r0 + anchor_r_in_patch, tl_c0 + anchor_c_in_patch)
    for key, patch_s in scaled.items():
        anchors[key] = _choose_anchor(key[0], key[1], patch_s)

    out = np.full((GRID_SIZE, GRID_SIZE), BG, dtype=int)
    out[MID, :] = DIV
    out[:, MID] = DIV
    paste(out, patch, tl_r0, tl_c0)
    for key, size in target_sizes.items():
        if key == (0, 0):
            continue
        ar, ac = anchors[key]
        paste(out, np.full((size, size), RED, dtype=int), ar, ac)
    for key, patch_s in scaled.items():
        _stamp(out, patch_s, *anchors[key])

    inp = out.copy()
    for key, size in target_sizes.items():
        if key == (0, 0):
            continue
        q_r0, q_r1, q_c0, q_c1 = QUADS[key]
        inp[q_r0 : q_r1 + 1, q_c0 : q_c1 + 1] = BG
        ar, ac = anchors[key]
        paste(inp, np.full((size, size), RED, dtype=int), ar, ac)

    return inp if np.array_equal(main(inp), out) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid b10624e5 D13 input")
