# Combo variant of b10624e5 (index 18)
# combo_spec: D3/3+D4/1
# Name: 图案平铺填满目标 + 非均匀缩放

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste


# Design analysis for the fused D3/3 + D4/1 variant
# a) Concrete behavior and trace:
#    - 原题里，TL 象限给出一个带 2x2 红锚块的彩色连通图案；其它象限只给红锚块。
#      solver 会先按象限做镜像，再按“目标锚块大小 / 源锚块大小”做等比缩放，然后只贴一次。
#    - 融合后，目标锚块不再限定为正方形，而是实心矩形 HxW；纵向倍率 = H / 2，横向倍率 = W / 2。
#      因此同一个图案会在不同象限被单独拉高、拉宽或双轴同时拉伸。
#    - 贴图方式也改变：先算出与目标锚块对齐的一块“基准砖块”，再以该砖块的高宽为周期平铺整个目标象限；
#      超出象限边界的部分直接裁掉。
#    - 具体小例子：若 TL patch 像素 bbox 是 4x6，内部红锚块是 2x2。
#      对 TR 若目标锚块是 2x4，则先左右镜像，再按 (1x, 2x) 缩成 4x12，红区变成 2x4；
#      原题只会在红锚块旁画这一块 4x12，融合版则会继续把这块 4x12 沿整片 TR 象限重复平铺。
#      对 BL 若目标锚块是 3x2，则会按 (1.5x, 1x) 变成 6x6 并纵向拉伸，再整区平铺。
# b) Complexity calibration:
#    - 为了让“非均匀缩放”稳定可逆，源图案只使用逻辑网格再上采样成 2x2 像素块的方案；
#      这样任意目标锚块高宽取 1/2/3/4 时，最近邻缩放后的红区仍会精确落成目标矩形。
#    - 为了让融合效果明显，每个样例固定包含三类目标：
#      TR 只改宽度，BL 只改高度，BR 同时改高度和宽度且二者不同。
#    - 为了让平铺肉眼可见，生成器拒绝“除了基准砖块几乎没有额外重复像素”的布局。
# c) Implicit-assumption audit against the original solver:
#    - 原代码把红锚块记录成单一 `size`，默认其为正方形；融合后必须记录 `(r0, c0, h, w)` 并校验是实心矩形。
#    - 原代码的 `ratio` 同时用于高和宽；融合后必须拆成 `scale_h` 与 `scale_w`。
#    - 原代码把 patch 只贴一次；融合后该假设失效，必须改成遍历整片平铺晶格。
#    - 原代码允许复制 patch 中的红像素，因为只贴一次时恰好与目标锚块重合；
#      融合后若不跳过红像素，平铺副本会制造额外红锚块，所以 tile 时只能复制非背景且非红色像素。
#    - 原代码默认只要“缩放后还有红色”即可；融合后还依赖“缩放后红 bbox 的高宽恰好等于目标锚块高宽”，
#      这由 block 型 generator 保证。
# d) Placement legality:
#    - 四个象限仍各有且仅有一个红锚块；TL 是 2x2，非 TL 是完整矩形并完全落在各自 13x13 象限中。
#    - TL 整个源 patch 完整放进 TL 象限。
#    - 非 TL 输出通过象限内裁剪平铺，不跨越分隔线，也不覆盖已有红锚块。
# e) Visual observability:
#    - TR / BL / BR 分别提供“只横向拉伸 / 只纵向拉伸 / 双轴拉伸”的对照，解题者能从少量样例里直接看到两个缩放轴是分开的。
#    - 平铺让目标象限出现重复纹理，不再像原题那样只是锚块附近的一次性贴图。
#    - 源图案要求在红锚块同行与同列都存在其它颜色，并且至少在一个镜像轴上不对称，这样镜像和单轴拉伸都可见。
# f) Visual cleanliness:
#    - 目标象限中所有重复砖块的红像素都跳过，只保留输入里原本那一个红锚块，避免“满屏红锚块”。
#    - 只保留一个 TL 彩色组件和一条十字分隔线，平铺虽密但规则单一。
# g) Color inferability:
#    - 没有新增颜色映射；颜色只继承源 motif，本变体的推理负担完全来自几何变换与平铺，不依赖颜色 id 的隐藏顺序。
# h) New solver assumptions and generator guarantees:
#    - solver 假设非 TL 象限输入里只有红锚块；generator 会先生成完整输出，再清空非 TL 象限仅保留锚块。
#    - solver 假设 TL 含红图案是单个 4 连通非背景组件；generator 只产出满足该条件的 block motif。
#    - solver 假设缩放后 patch 的红 bbox 能唯一对齐到目标矩形锚块；generator 用 2x2 block 源图案保证这一点。
#    - solver 假设平铺周期就是缩放后 patch 的整体 bbox；generator 要求每行每列都有可见像素，避免周期退化成难辨认的稀疏噪声。


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

SOURCE_ANCHOR_H = 2
SOURCE_ANCHOR_W = 2
BLOCK_DIMS = [(2, 2), (2, 3), (3, 2)]
FILL_RATIO_MIN = 0.5
FILL_RATIO_MAX = 0.8
MOTIF_COLORS_MIN = 3
MOTIF_COLORS_MAX = 5
COLOR_POOL = [0, 1, 3, 5, 6, 7, 8, 9]

TARGET_STRETCH_VALUES = [1, 3, 4]
BR_SHAPES = [(h, w) for h in TARGET_STRETCH_VALUES for w in TARGET_STRETCH_VALUES if h != w]
ANCHOR_SEARCH_TRIES = 12
MIN_EXTRA_PER_QUADRANT = 2
MIN_STRONG_EXTRA = 8
MIN_TOTAL_EXTRA = 18
MAX_RETRIES = 500


def _find_dividers(g: np.ndarray, bg: int) -> tuple[int, int]:
    """Return the unique full-row/full-col quadrant divider: (mid_row, mid_col)."""
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
    """Nearest-neighbor resize that supports different row and column target sizes."""
    src_h, src_w = img.shape
    assert new_h > 0 and new_w > 0, "resized patch must stay positive"
    rr = np.clip(np.floor(np.arange(new_h) * src_h / new_h).astype(int), 0, src_h - 1)
    cc = np.clip(np.floor(np.arange(new_w) * src_w / new_w).astype(int), 0, src_w - 1)
    return img[rr[:, None], cc[None, :]]


def _red_bbox_dims(img: np.ndarray) -> tuple[int, int, int, int]:
    """Return the red bbox as (r0, c0, h, w)."""
    rr, cc = np.where(img == RED)
    assert rr.size > 0, "patch must contain red pixels"
    r0, c0 = int(rr.min()), int(cc.min())
    h = int(rr.max() - rr.min() + 1)
    w = int(cc.max() - cc.min() + 1)
    return r0, c0, h, w


def _reflect_patch(patch: np.ndarray, qb: int, qr: int) -> np.ndarray:
    """Apply quadrant-dependent reflections."""
    out = patch
    if qr == 1:
        out = np.fliplr(out)
    if qb == 1:
        out = np.flipud(out)
    return out


def _scaled_patch(patch: np.ndarray, qb: int, qr: int, target_h: int, target_w: int) -> np.ndarray:
    """Reflect the source patch and resize it with independent vertical and horizontal factors."""
    patch_t = _reflect_patch(patch, qb, qr)
    new_h = max(1, int(round(patch_t.shape[0] * target_h / SOURCE_ANCHOR_H)))
    new_w = max(1, int(round(patch_t.shape[1] * target_w / SOURCE_ANCHOR_W)))
    patch_s = _resize_nn(patch_t, new_h, new_w)
    roundtrip = _resize_nn(patch_s, patch_t.shape[0], patch_t.shape[1])
    assert np.array_equal(roundtrip, patch_t), "non-uniform scaling must preserve all motif information"
    red_r0, red_c0, red_h, red_w = _red_bbox_dims(patch_s)
    assert (red_h, red_w) == (target_h, target_w), "scaled red anchor must match the target anchor rectangle"
    assert red_r0 >= 0 and red_c0 >= 0
    return patch_s


def _tile_quadrant(
    out: np.ndarray,
    patch_s: np.ndarray,
    anchor_r: int,
    anchor_c: int,
    q_r0: int,
    q_r1: int,
    q_c0: int,
    q_c1: int,
    bg: int,
) -> None:
    """
    Tile one transformed patch across a quadrant.

    The tile phase is defined by the placement where the patch's red bbox aligns with the target anchor.
    Only non-background, non-red motif pixels are repeated.
    """
    red_r0, red_c0, _, _ = _red_bbox_dims(patch_s)
    place_r0 = anchor_r - red_r0
    place_c0 = anchor_c - red_c0

    tile_h, tile_w = patch_s.shape
    src_rr, src_cc = np.where((patch_s != bg) & (patch_s != RED))
    vals = patch_s[src_rr, src_cc]
    if src_rr.size == 0:
        return

    start_r = place_r0 + ((q_r0 - place_r0) // tile_h) * tile_h
    if start_r + tile_h - 1 < q_r0:
        start_r += tile_h
    start_c = place_c0 + ((q_c0 - place_c0) // tile_w) * tile_w
    if start_c + tile_w - 1 < q_c0:
        start_c += tile_w

    for tile_r0 in range(start_r, q_r1 + 1, tile_h):
        for tile_c0 in range(start_c, q_c1 + 1, tile_w):
            tgt_rr = src_rr + tile_r0
            tgt_cc = src_cc + tile_c0
            valid = (
                (tgt_rr >= q_r0)
                & (tgt_rr <= q_r1)
                & (tgt_cc >= q_c0)
                & (tgt_cc <= q_c1)
            )
            if np.any(valid):
                out[tgt_rr[valid], tgt_cc[valid]] = vals[valid]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replicate the top-left motif into the other quadrants, but use rectangular red anchors to drive non-uniform scaling and then tile the result through each target quadrant.

    Input structure:
    - One full colored row and one full colored column split the grid into four 13x13 quadrants.
    - Every quadrant contains exactly one red anchor block; the TL quadrant shows the full colored motif around its 2x2 red anchor, while the other quadrants show only their red rectangular anchors.
    - The TL motif is a single non-background 4-connected component containing the TL red anchor.

    Rule:
    - Extract the TL connected motif as the source patch.
    - For each non-TL quadrant, mirror the patch according to quadrant side, resize rows by target_anchor_height / source_anchor_height and columns by target_anchor_width / source_anchor_width, and align the resized red bbox onto the target anchor.
    - Treat that aligned placement as one tile in a rectangular lattice and repeat the transformed patch across the entire target quadrant, clipping at the quadrant border and skipping repeated red pixels.

    Key insight:
    - Quadrant position still selects the reflection, but the target red anchor now carries two independent scale codes and the transformed patch is not stamped once:
      it becomes a repeating tile whose phase is locked by the anchor.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    bg = get_bg(g)
    mid_r, mid_c = _find_dividers(g, bg)

    red = components_mask(g == RED, connectivity=4)
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

    assert len(anchors) == 4 and (0, 0) in anchors, "each quadrant must contain exactly one red anchor"
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

    quads = {
        (0, 0): (0, mid_r - 1, 0, mid_c - 1),
        (0, 1): (0, mid_r - 1, mid_c + 1, g.shape[1] - 1),
        (1, 0): (mid_r + 1, g.shape[0] - 1, 0, mid_c - 1),
        (1, 1): (mid_r + 1, g.shape[0] - 1, mid_c + 1, g.shape[1] - 1),
    }

    for (qb, qr), (tr0, tc0, target_h, target_w) in anchors.items():
        if (qb, qr) == (0, 0):
            continue
        patch_s = _scaled_patch(patch, qb, qr, target_h, target_w)
        q_r0, q_r1, q_c0, q_c1 = quads[(qb, qr)]
        _tile_quadrant(out, patch_s, tr0, tc0, q_r0, q_r1, q_c0, q_c1, bg)

    return out


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


def _assign_colors(
    filled: set[tuple[int, int]],
    red_cells: set[tuple[int, int]],
    h: int,
    w: int,
) -> np.ndarray | None:
    """Assign colors while ensuring every row and column of the bbox is visibly occupied."""
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
    """Reject motifs where mirror and one-axis stretch cues would be too weak."""
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


def _count_extra_tiled_pixels(
    patch_s: np.ndarray,
    anchor_r: int,
    anchor_c: int,
    q_r0: int,
    q_r1: int,
    q_c0: int,
    q_c1: int,
) -> int:
    """
    Count visible non-red tiled pixels that lie outside the single base placement.

    This quantifies whether the target really exhibits the D3/3 behavior rather than degenerating into a one-off stamp.
    """
    red_r0, red_c0, _, _ = _red_bbox_dims(patch_s)
    place_r0 = anchor_r - red_r0
    place_c0 = anchor_c - red_c0
    tile_h, tile_w = patch_s.shape
    src_rr, src_cc = np.where((patch_s != BG) & (patch_s != RED))
    if src_rr.size == 0:
        return 0

    base_mask = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
    base_rr = src_rr + place_r0
    base_cc = src_cc + place_c0
    base_valid = (
        (base_rr >= q_r0)
        & (base_rr <= q_r1)
        & (base_cc >= q_c0)
        & (base_cc <= q_c1)
    )
    base_mask[base_rr[base_valid], base_cc[base_valid]] = True

    visible = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
    start_r = place_r0 + ((q_r0 - place_r0) // tile_h) * tile_h
    if start_r + tile_h - 1 < q_r0:
        start_r += tile_h
    start_c = place_c0 + ((q_c0 - place_c0) // tile_w) * tile_w
    if start_c + tile_w - 1 < q_c0:
        start_c += tile_w

    for tile_r0 in range(start_r, q_r1 + 1, tile_h):
        for tile_c0 in range(start_c, q_c1 + 1, tile_w):
            tgt_rr = src_rr + tile_r0
            tgt_cc = src_cc + tile_c0
            valid = (
                (tgt_rr >= q_r0)
                & (tgt_rr <= q_r1)
                & (tgt_cc >= q_c0)
                & (tgt_cc <= q_c1)
            )
            visible[tgt_rr[valid], tgt_cc[valid]] = True

    return int(np.count_nonzero(visible & ~base_mask))


def _choose_anchor(qb: int, qr: int, anchor_h: int, anchor_w: int, patch_s: np.ndarray) -> tuple[int, int]:
    """Pick a legal anchor position and prefer phases that expose more repeated tiles."""
    q_r0, q_r1, q_c0, q_c1 = QUADS[(qb, qr)]
    r_lo, r_hi = q_r0, q_r1 - anchor_h + 1
    c_lo, c_hi = q_c0, q_c1 - anchor_w + 1
    assert r_lo <= r_hi and c_lo <= c_hi, "anchor must fit in its quadrant"

    best = None
    best_score = -1
    for _ in range(ANCHOR_SEARCH_TRIES):
        cand = (random.randint(r_lo, r_hi), random.randint(c_lo, c_hi))
        score = _count_extra_tiled_pixels(patch_s, cand[0], cand[1], q_r0, q_r1, q_c0, q_c1)
        if score > best_score:
            best = cand
            best_score = score
    assert best is not None
    return best


def _target_shapes() -> dict[tuple[int, int], tuple[int, int]]:
    """Force horizontal-only, vertical-only, and fully non-uniform targets in every sample."""
    return {
        (0, 0): (SOURCE_ANCHOR_H, SOURCE_ANCHOR_W),
        (0, 1): (SOURCE_ANCHOR_H, random.choice(TARGET_STRETCH_VALUES)),
        (1, 0): (random.choice(TARGET_STRETCH_VALUES), SOURCE_ANCHOR_W),
        (1, 1): random.choice(BR_SHAPES),
    }


def _gen_attempt() -> np.ndarray | None:
    """Build a fused tiled output, derive the input, and oracle-check it."""
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

    anchors: dict[tuple[int, int], tuple[int, int]] = {}
    tl_r0 = random.randint(0, MID - patch.shape[0])
    tl_c0 = random.randint(0, MID - patch.shape[1])
    anchors[(0, 0)] = (tl_r0 + anchor_r_in_patch, tl_c0 + anchor_c_in_patch)
    for key, patch_s in scaled.items():
        anchor_h, anchor_w = target_shapes[key]
        anchors[key] = _choose_anchor(key[0], key[1], anchor_h, anchor_w, patch_s)

    extras = {
        key: _count_extra_tiled_pixels(
            patch_s,
            anchors[key][0],
            anchors[key][1],
            *QUADS[key],
        )
        for key, patch_s in scaled.items()
    }
    extra_values = list(extras.values())
    if min(extra_values) < MIN_EXTRA_PER_QUADRANT:
        return None
    if sum(v >= MIN_STRONG_EXTRA for v in extra_values) < 2:
        return None
    if sum(extra_values) < MIN_TOTAL_EXTRA:
        return None

    out = np.full((GRID_SIZE, GRID_SIZE), BG, dtype=int)
    out[MID, :] = DIV
    out[:, MID] = DIV
    paste(out, patch, tl_r0, tl_c0)
    for key, (anchor_h, anchor_w) in target_shapes.items():
        if key == (0, 0):
            continue
        paste(out, np.full((anchor_h, anchor_w), RED, dtype=int), *anchors[key])
    for key, patch_s in scaled.items():
        _tile_quadrant(out, patch_s, anchors[key][0], anchors[key][1], *QUADS[key], BG)

    inp = out.copy()
    for key, (anchor_h, anchor_w) in target_shapes.items():
        if key == (0, 0):
            continue
        q_r0, q_r1, q_c0, q_c1 = QUADS[key]
        inp[q_r0 : q_r1 + 1, q_c0 : q_c1 + 1] = BG
        paste(inp, np.full((anchor_h, anchor_w), RED, dtype=int), *anchors[key])

    return inp if np.array_equal(main(inp), out) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid fused b10624e5 input")
