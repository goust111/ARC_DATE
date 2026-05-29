# Variant of b10624e5
# Dimension: D4 — 变换：缩放规则 (transform)
# Change: 缩放因子 = 目标红色锚块尺寸 / 源红色锚块尺寸，图案每个像素按比例放大为 factor x factor 的色块... → 非均匀缩放（水平和垂直不同倍率）
#
# Design analysis
# spec:
#   variant_core:
#     original_rule: 每个目标象限只有一个 NxN 红锚块，因此只存在一个统一倍率 N/2，图案在横纵两个方向一起等比缩放。
#     new_rule: 每个目标象限改为一个 HxW 红锚块。纵向倍率 = H / source_anchor_height，横向倍率 = W / source_anchor_width。
#     preserved_parts: 四象限结构、分隔线、TL 源图案、按象限做镜像、按红锚块对齐、非 TL 输入只保留红锚块。
#   concrete_behavior:
#     source_anchor: TL 仍保持 2x2 红锚块，这样 D4 的唯一新信息来自目标锚块的高和宽。
#     target_anchors:
#       - TR: 高固定为 2，只改宽度，呈现“只水平拉伸/压缩”。
#       - BL: 宽固定为 2，只改高度，呈现“只垂直拉伸/压缩”。
#       - BR: 高宽都改，呈现“双轴同时拉伸/压缩”。
#     source_motif: TL 图案仍是一个包含红锚块的 4 连通彩色组件，但生成时强制使用 2x2 逻辑块上采样，
#       这样非均匀最近邻缩放后，红锚块会精确变成目标矩形，且缩放信息在像素级可见。
#   behavior_trace_example:
#     source_patch: 假设 TL 图案 bbox 是 6x4，其中红锚块是内部一个 2x2 子块。
#     old_logic: 若目标锚块大小为 4，则纵横都乘 2，目标 patch 变成 12x8。
#     new_logic: 若目标锚块改为 2x4，则纵向倍率 = 2/2 = 1，横向倍率 = 4/2 = 2，目标 patch 变成 6x8。
#     visible_difference: 同一个 motif 在 TR 只变宽、在 BL 只变高、在 BR 同时变宽变高，区别肉眼可见。
#   complexity_calibration:
#     why_not_full_freeform: 原题允许 freeform patch，但非均匀缩放下 freeform 更容易产生红锚块尺寸不精确、信息混叠和 roundtrip 失败。
#       因此生成器只保留“逻辑网格 × 2x2 上采样”的 block 方案；这属于为 D4 可实现性做的最小配套修改。
#     control_of_degrees:
#       - 目标高宽均限制在 1..4。
#       - 每个样例固定同时包含“横向-only / 纵向-only / 双轴”三种情况。
#       - BR 要求高宽都不等于 2 且彼此不同，避免退化回等比缩放。
#   assumption_audit_for_main:
#     - 原代码把每个红组件记录为 (r0, c0, size)，隐含前提是锚块必为正方形；新版本必须改为 (r0, c0, h, w)。
#     - 原代码 ratio = tL / sL，把同一个倍率同时用于 new_h/new_w；新版本必须分拆为 ratio_h 与 ratio_w。
#     - 原代码允许只检查“scaled patch 含红色”；新版本还依赖“红色 bbox 的高宽恰好等于目标锚块高宽”，
#       这是由 block 生成器保证的，否则对齐会变得含糊。
#     - 原代码在放置时以红 bbox 左上角对齐，这在矩形锚块下仍成立，只要目标红组件是实心矩形。
#   placement_legality:
#     - 每个象限仍只有一个红组件。
#     - 非 TL 象限先放实心矩形红锚块，再叠加缩放后的图案；stamp 时跳过已有红像素，避免覆盖锚块。
#     - 选择锚块位置时优先让整个缩放后图案都落在象限内；如果做不到，也至少保证锚块本身完整在象限内。
#   observability_and_cleanliness:
#     - 每个样例都强制出现三类不同的非均匀缩放，避免偶然全部接近原题的等比缩放。
#     - motif 至少含 3 种非背景色，并要求逻辑图案在红锚块同行、同列都存在别的像素，且不同时水平/垂直完全对称，
#       让镜像与单轴拉伸都可见。
#     - 仍沿用单一分隔线和单一 TL 彩色组件，避免同一区域多层杂糅。
#   color_reasoning:
#     - 没有引入任何颜色映射或按颜色排序的规则，颜色只承担“组成 motif 的可见结构”作用。
#   generator_guarantees_for_main:
#     - TL 红锚块恒为 2x2，其他三个锚块是实心矩形。
#     - 源 patch 的高宽始终是 2 的倍数，因此按 H/2、W/2 做最近邻缩放时目标尺寸是精确整数。
#     - 生成器最终仍使用 output-first + oracle check，只接受 main(inp) == out 的样本。

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste


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
    """Nearest-neighbor resize that supports different row/column target sizes."""
    src_h, src_w = img.shape
    assert new_h > 0 and new_w > 0, "resized patch must stay positive"
    rr = np.clip(np.floor(np.arange(new_h) * src_h / new_h).astype(int), 0, src_h - 1)
    cc = np.clip(np.floor(np.arange(new_w) * src_w / new_w).astype(int), 0, src_w - 1)
    return img[rr[:, None], cc[None, :]]


def _red_bbox_dims(img: np.ndarray) -> tuple[int, int, int, int]:
    """Return red bbox as (r0, c0, h, w)."""
    rr, cc = np.where(img == RED)
    assert rr.size > 0, "patch must contain red pixels"
    r0, c0 = int(rr.min()), int(cc.min())
    h = int(rr.max() - rr.min() + 1)
    w = int(cc.max() - cc.min() + 1)
    return r0, c0, h, w


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy the top-left motif into the other quadrants, but let rectangular red anchors encode separate row and column stretch factors.

    Input structure:
    - A single colored divider row and divider column split the grid into four quadrants.
    - Every quadrant contains one red (2) anchor block; the TL anchor sits inside the only full motif, while the other quadrants show only their red anchor.
    - The TL motif is a single non-background connected component containing the TL red anchor.

    Rule:
    - Extract the TL motif as the source pattern.
    - For each other quadrant, reflect the source left/right for right quadrants and up/down for bottom quadrants.
    - Resize rows by (target_anchor_height / source_anchor_height) and columns by (target_anchor_width / source_anchor_width), then align the resized red bbox onto the target anchor.

    Key insight:
    - Quadrant position still tells you which mirrors to apply, but the target red block's height and width now act as two independent scaling codes.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    bg = get_bg(g)
    mid_r, mid_c = _find_dividers(g, bg)

    red = components_mask(g == 2, connectivity=4)
    assert red, "grid must contain red (color 2) anchor blocks"

    anchors: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    for pts in red:
        r0, c0, r1, c1 = bbox_coords(pts)
        ah = int(r1 - r0 + 1)
        aw = int(c1 - c0 + 1)
        assert len(pts) == ah * aw, "each anchor must be a filled rectangle"
        rr = (r0 + r1) // 2
        cc = (c0 + c1) // 2
        qb = 0 if rr < mid_r else 1
        qr = 0 if cc < mid_c else 1
        key = (qb, qr)
        assert key not in anchors, "each quadrant must contain exactly one red anchor block"
        anchors[key] = (int(r0), int(c0), ah, aw)

    assert (0, 0) in anchors, "top-left quadrant must contain a red anchor block"
    _, _, source_h, source_w = anchors[(0, 0)]
    assert source_h > 0 and source_w > 0, "top-left red anchor must have positive dimensions"

    tl = g[0:mid_r, 0:mid_c]
    comps = components_mask(tl != bg, connectivity=4)
    patch_bb = None
    for pts in comps:
        if any(tl[r, c] == RED for r, c in pts):
            r0, c0, r1, c1 = bbox_coords(pts)
            patch_bb = (int(r0), int(c0), int(r1), int(c1))
            break
    assert patch_bb is not None, "top-left quadrant must contain a non-bg component with a red pixel"

    pr0, pc0, pr1, pc1 = patch_bb
    patch = tl[pr0 : pr1 + 1, pc0 : pc1 + 1].copy()

    for (qb, qr), (tr0, tc0, target_h, target_w) in anchors.items():
        if (qb, qr) == (0, 0):
            continue

        patch_t = patch
        if qr == 1:
            patch_t = np.fliplr(patch_t)
        if qb == 1:
            patch_t = np.flipud(patch_t)

        scale_h = float(target_h) / float(source_h)
        scale_w = float(target_w) / float(source_w)
        new_h = max(1, int(round(patch_t.shape[0] * scale_h)))
        new_w = max(1, int(round(patch_t.shape[1] * scale_w)))
        patch_s = _resize_nn(patch_t, new_h, new_w)
        roundtrip = _resize_nn(patch_s, patch_t.shape[0], patch_t.shape[1])
        assert np.array_equal(roundtrip, patch_t), "non-uniform scaling must preserve all motif information"

        ar0, ac0, red_h, red_w = _red_bbox_dims(patch_s)
        assert red_h == target_h and red_w == target_w, "scaled red anchor must match the target anchor rectangle"
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


# --------------- generator (o5x) ---------------

# --- Grid constants ---
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

# --- Source motif parameters ---
SOURCE_ANCHOR_H = 2
SOURCE_ANCHOR_W = 2
BLOCK_DIMS = [(2, 2), (2, 3), (3, 2), (3, 3)]
FILL_RATIO_MIN = 0.5
FILL_RATIO_MAX = 0.8
MOTIF_COLORS_MIN = 3
MOTIF_COLORS_MAX = 5
COLOR_POOL = [0, 1, 3, 5, 6, 7, 8, 9]

# --- Target anchor parameters ---
TARGET_STRETCH_VALUES = [1, 3, 4]
BR_SHAPES = [(h, w) for h in TARGET_STRETCH_VALUES for w in TARGET_STRETCH_VALUES if h != w]

# --- Generation retry limit ---
MAX_RETRIES = 300


def _grow_4conn(h: int, w: int, seeds: set[tuple[int, int]], n_fill: int) -> set[tuple[int, int]]:
    """Grow a 4-connected region from seeds until it has n_fill cells."""
    filled = set(seeds)
    frontier = set()
    for r, c in seeds:
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in filled:
                frontier.add((nr, nc))
    while len(filled) < n_fill and frontier:
        cell = random.choice(list(frontier))
        filled.add(cell)
        frontier.discard(cell)
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = cell[0] + dr, cell[1] + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in filled:
                frontier.add((nr, nc))
    return filled


def _assign_colors(filled: set[tuple[int, int]], red_cells: set[tuple[int, int]], h: int, w: int) -> np.ndarray | None:
    """Assign random colors to filled cells; returns None if row/column coverage is incomplete."""
    rows = {r for r, _ in filled}
    cols = {c for _, c in filled}
    if rows != set(range(h)) or cols != set(range(w)):
        return None
    non_red = [p for p in filled if p not in red_cells]
    random.shuffle(non_red)
    max_colors = min(MOTIF_COLORS_MAX, len(non_red) + 1)
    if max_colors < MOTIF_COLORS_MIN:
        return None
    n_colors = random.randint(MOTIF_COLORS_MIN, max_colors)
    pool = COLOR_POOL[:]
    random.shuffle(pool)
    extras = pool[: n_colors - 1]
    grid = np.full((h, w), BG, dtype=int)
    for r, c in red_cells:
        grid[r, c] = RED
    for i, (r, c) in enumerate(non_red):
        grid[r, c] = extras[i] if i < len(extras) else random.choice(extras)
    return grid


def _logical_patch_is_observable(logical: np.ndarray, ar: int, ac: int) -> bool:
    """Reject motifs where one-axis stretching or mirroring would be too hard to see."""
    non_red = logical != BG
    same_row = any(non_red[ar, c] and logical[ar, c] != RED for c in range(logical.shape[1]))
    same_col = any(non_red[r, ac] and logical[r, ac] != RED for r in range(logical.shape[0]))
    asym_h = not np.array_equal(logical, np.fliplr(logical))
    asym_v = not np.array_equal(logical, np.flipud(logical))
    return same_row and same_col and (asym_h or asym_v)


def _source_patch() -> tuple[np.ndarray, int, int] | None:
    """Build the TL motif from a logical grid, then upscale each logical cell to a 2x2 pixel block."""
    lh, lw = random.choice(BLOCK_DIMS)
    n_cells = lh * lw
    min_fill = max(3, int(np.ceil(FILL_RATIO_MIN * n_cells)))
    max_fill = max(min_fill, int(np.floor(FILL_RATIO_MAX * n_cells)))
    ar, ac = random.randrange(lh), random.randrange(lw)
    filled = _grow_4conn(lh, lw, {(ar, ac)}, random.randint(min_fill, max_fill))
    logical = _assign_colors(filled, {(ar, ac)}, lh, lw)
    if logical is None or not _logical_patch_is_observable(logical, ar, ac):
        return None
    patch = np.repeat(np.repeat(logical, SOURCE_ANCHOR_H, axis=0), SOURCE_ANCHOR_W, axis=1)
    return patch, ar * SOURCE_ANCHOR_H, ac * SOURCE_ANCHOR_W


def _scaled_patch(patch: np.ndarray, qb: int, qr: int, target_h: int, target_w: int) -> np.ndarray:
    """Reflect and scale the source patch with independent vertical/horizontal factors."""
    p = patch
    if qr:
        p = np.fliplr(p)
    if qb:
        p = np.flipud(p)
    out_h = max(1, int(round(p.shape[0] * target_h / SOURCE_ANCHOR_H)))
    out_w = max(1, int(round(p.shape[1] * target_w / SOURCE_ANCHOR_W)))
    return _resize_nn(p, out_h, out_w)


def _choose_anchor(qb: int, qr: int, anchor_h: int, anchor_w: int, patch_s: np.ndarray) -> tuple[int, int]:
    """Pick a quadrant-local top-left for the red anchor, preferring placements where the whole motif fits."""
    q_r0, q_r1, q_c0, q_c1 = QUADS[(qb, qr)]
    red_r0, red_c0, _, _ = _red_bbox_dims(patch_s)

    fit_r0 = q_r0 + red_r0
    fit_r1 = q_r1 - patch_s.shape[0] + 1 + red_r0
    fit_c0 = q_c0 + red_c0
    fit_c1 = q_c1 - patch_s.shape[1] + 1 + red_c0

    anch_r0, anch_r1 = q_r0, q_r1 - anchor_h + 1
    anch_c0, anch_c1 = q_c0, q_c1 - anchor_w + 1

    r_lo, r_hi = max(fit_r0, anch_r0), min(fit_r1, anch_r1)
    c_lo, c_hi = max(fit_c0, anch_c0), min(fit_c1, anch_c1)
    if r_lo <= r_hi and c_lo <= c_hi:
        return random.randint(r_lo, r_hi), random.randint(c_lo, c_hi)
    return random.randint(anch_r0, anch_r1), random.randint(anch_c0, anch_c1)


def _stamp(out: np.ndarray, patch_s: np.ndarray, anchor_r: int, anchor_c: int) -> None:
    """Stamp non-bg pixels from a scaled patch onto the output grid, aligned by red anchor."""
    red_r0, red_c0, _, _ = _red_bbox_dims(patch_s)
    place_r0 = anchor_r - red_r0
    place_c0 = anchor_c - red_c0

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
    tgt_rr, tgt_cc, src_rr, src_cc = tgt_rr[valid], tgt_cc[valid], src_rr[valid], src_cc[valid]
    not_red = out[tgt_rr, tgt_cc] != RED
    out[tgt_rr[not_red], tgt_cc[not_red]] = patch_s[src_rr[not_red], src_cc[not_red]]


def _target_shapes() -> dict[tuple[int, int], tuple[int, int]]:
    """Force one horizontal-only, one vertical-only, and one fully non-uniform target."""
    return {
        (0, 0): (SOURCE_ANCHOR_H, SOURCE_ANCHOR_W),
        (0, 1): (SOURCE_ANCHOR_H, random.choice(TARGET_STRETCH_VALUES)),
        (1, 0): (random.choice(TARGET_STRETCH_VALUES), SOURCE_ANCHOR_W),
        (1, 1): random.choice(BR_SHAPES),
    }


def _gen_attempt() -> np.ndarray | None:
    """Single generation attempt: build output, derive input, and oracle-check it."""
    target_shapes = _target_shapes()
    source = _source_patch()
    if source is None:
        return None
    patch, anchor_r_in_patch, anchor_c_in_patch = source

    scaled = {
        key: _scaled_patch(patch, key[0], key[1], dims[0], dims[1])
        for key, dims in target_shapes.items()
        if key != (0, 0)
    }

    for key, patch_s in scaled.items():
        _, _, red_h, red_w = _red_bbox_dims(patch_s)
        if (red_h, red_w) != target_shapes[key]:
            return None

    anchors: dict[tuple[int, int], tuple[int, int]] = {}
    tl_r0 = random.randint(0, MID - patch.shape[0])
    tl_c0 = random.randint(0, MID - patch.shape[1])
    anchors[(0, 0)] = (tl_r0 + anchor_r_in_patch, tl_c0 + anchor_c_in_patch)
    for key, patch_s in scaled.items():
        anchor_h, anchor_w = target_shapes[key]
        anchors[key] = _choose_anchor(key[0], key[1], anchor_h, anchor_w, patch_s)

    out = np.full((GRID_SIZE, GRID_SIZE), BG, dtype=int)
    out[MID, :] = DIV
    out[:, MID] = DIV
    paste(out, patch, tl_r0, tl_c0)
    for key, (anchor_h, anchor_w) in target_shapes.items():
        if key == (0, 0):
            continue
        ar, ac = anchors[key]
        paste(out, np.full((anchor_h, anchor_w), RED, dtype=int), ar, ac)
    for key, patch_s in scaled.items():
        _stamp(out, patch_s, *anchors[key])

    inp = out.copy()
    for key, (anchor_h, anchor_w) in target_shapes.items():
        if key == (0, 0):
            continue
        q_r0, q_r1, q_c0, q_c1 = QUADS[key]
        inp[q_r0 : q_r1 + 1, q_c0 : q_c1 + 1] = BG
        ar, ac = anchors[key]
        paste(inp, np.full((anchor_h, anchor_w), RED, dtype=int), ar, ac)

    return inp if np.array_equal(main(inp), out) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid b10624e5 input")
