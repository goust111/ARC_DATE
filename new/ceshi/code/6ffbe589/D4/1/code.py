# Variant of 6ffbe589
# Dimension: D4 — 源对象：提示精灵结构 (src)
# Change: 板外的1-3个单色小精灵，每个1-5像素，8-连通色块... → 提示精灵为线段精灵

from __future__ import annotations

import random

import numpy as np

from arc_utils import colors_present, find_components, get_bg, paste

# Design analysis for the D4 variant
# ----------------------------------
# variant_spec:
#   core_change:
#     old_src: board 外与板内颜色对应的小单色 blob; 只读面积 mod 4
#     new_src: board 外与板内颜色对应的小单色线段; 必须同时读取
#       - 长度 mod 4 -> 该颜色层的顺时针旋转量
#       - 方向 -> 额外是否左右翻转
#   concrete_visual_form:
#     - 每个 cue 都是 1xL 或 Lx1 的实心线段, L in {2,3,4,5}
#     - 水平线段: 只旋转
#     - 垂直线段: 先旋转, 再做左右翻转
#     - 颜色仍与目标层一一对应, 所以人类可直接按颜色匹配
#   behavior_trace:
#     - 原任务某色 cue 若面积=3, solver 只做该色层旋转 90deg CW
#     - 现在若 cue 是长度 3 的水平线段, 仍做 90deg CW
#     - 若 cue 是长度 3 的垂直线段, 则先做 90deg CW, 再左右翻转
#     - 差异点不在“找哪一层”, 而在“同一层上需要额外读取 cue 的方向”
#   complexity_calibration:
#     - 方向只提供二值信息, 没有引入多档离散规则, 不会过度复杂
#     - 每个样例保证至少一条垂直 cue, 且至少一个非零旋转, 避免退化回原题
#     - 当 cue_count >= 2 时, 生成器尽量同时出现水平/垂直 cue, 让两种分支同场可比
#   main_assumption_audit:
#     - crop_components_near_largest 仍假设 board 是唯一最大前景连通块
#       generator 通过面积与间距约束保证这一点
#     - outside_grid 把 crop 区域清空, 默认所有 cue 完全在 crop 外
#       generator 保持 cue bbox 到 board bbox 的 Chebyshev 间距 >= 4
#     - find_components(... colors=[col]) 默认同色板外 cue 最多一个
#       generator 对每个颜色最多放一个线段 cue
#     - 旧代码把 cue 的 area 直接当编码, 对 shape 没约束
#       新代码必须校验 cue 为轴对齐实心线段, 否则方向信息会失真
#     - 旧代码在 rot_k == 0 时直接跳过
#       新规则下“长度 4 的垂直线段”代表纯翻转, 所以不能仅凭 rot_k 决定是否跳过
#   placement_legality:
#     - cue 仍不可重叠、不可越界、不可互相贴近, 并与 board 保持足够间距
#     - 因为 cue 变成长线段, board 不再强制贴边; 改为选择一个位置, 保证
#       - 若存在水平 cue, 上下至少有一侧保留 >= 5 的边带
#       - 若存在垂直 cue, 左右至少有一侧保留 >= 5 的边带
#     - 这样长度 5 的线段也有稳定放置空间
#   visual_observability:
#     - 每个样例至少一条垂直 cue, 所以方向编码不是摆设
#     - 对垂直 cue 对应的层, 生成器额外要求镜像不对称, 否则翻转不可见
#     - 对所有有 cue 的层, 旋转不对称仍保持, 否则长度编码不可见
#   cleanliness:
#     - cue 仍散落在 board 外的背景带中, 不进入 crop, 不覆盖板面
#     - cue 只用对应颜色, 不引入额外装饰, 网格视觉仍然干净
#   color_inference:
#     - 没有颜色排序或隐藏色号逻辑; 颜色只承担“cue 与层的匹配键”
#     - 人类看到同色线段和同色层, 就能独立归纳长度规则与方向规则


def crop_components_near_largest(
    grid: np.ndarray,
    *,
    bg: int | None = None,
    margin: int = 3,
    connectivity: int = 8,
) -> tuple[np.ndarray, tuple[int, int, int, int], int]:
    """
    Local helper (only used by this solver): crop the cluster near the largest non-bg component.

    Returns:
        (crop, crop_bbox, bg) where crop_bbox is in the original grid coordinates (inclusive).
    """
    if bg is None:
        bg = get_bg(grid)
    bg = int(bg)

    comps = find_components(grid, bg=bg, connectivity=int(connectivity), monochromatic=False, return_format="dict")
    assert comps, "expected at least one foreground component"

    largest_area = max(int(comp["area"]) for comp in comps)
    largest = [comp for comp in comps if int(comp["area"]) == largest_area]
    assert len(largest) == 1, "expected a unique largest foreground component"
    mr0, mc0, mr1, mc1 = map(int, largest[0]["bbox"])

    H, W = grid.shape
    er0 = max(0, int(mr0 - margin))
    ec0 = max(0, int(mc0 - margin))
    er1 = min(int(H - 1), int(mr1 + margin))
    ec1 = min(int(W - 1), int(mc1 + margin))

    keep_mask = np.zeros((H, W), dtype=bool)
    for comp in comps:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        if r1 < er0 or r0 > er1 or c1 < ec0 or c0 > ec1:
            continue
        coords = comp["coords"]
        keep_mask[coords[:, 0], coords[:, 1]] = True

    keep_coords = np.argwhere(keep_mask)
    assert keep_coords.size > 0, "expected the expanded board crop to retain the largest component"

    r0, c0 = keep_coords.min(axis=0)
    r1, c1 = keep_coords.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))

    crop = np.full((int(r1 - r0 + 1), int(c1 - c0 + 1)), int(bg), dtype=grid.dtype)
    sub = keep_mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
    crop[sub] = grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1][sub]
    return crop, (int(r0), int(c0), int(r1), int(c1)), int(bg)


def transform_color_layer(
    grid: np.ndarray,
    color: int,
    rot_k: int,
    *,
    flip_lr: bool = False,
    bg: int | None = None,
) -> np.ndarray:
    """
    Local helper (only used by this solver): rotate one color mask inside the square crop,
    optionally mirroring it left-right after the rotation.
    """
    if bg is None:
        bg = get_bg(grid)
    bg = int(bg)

    k = int(rot_k) % 4
    do_flip = bool(flip_lr)
    if k == 0 and not do_flip:
        return grid

    col = int(color)
    mask = (grid == col).astype(int)
    tmask = np.rot90(mask, k)
    assert tmask.shape == mask.shape, "expected a square crop for layer rotation"
    if do_flip:
        tmask = np.fliplr(tmask)

    out = grid.copy()
    out[out == col] = bg
    out[tmask == 1] = col
    return out


def _decode_line_cue(comp: dict) -> tuple[int, bool]:
    """Return (rot_k, flip_lr) from one outside cue component."""
    r0, c0, r1, c1 = map(int, comp["bbox"])
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    area = int(comp["area"])
    assert (h == 1) ^ (w == 1), "expected the cue to be a non-degenerate horizontal/vertical line"

    length = int(max(h, w))
    assert length >= 2, "expected line cues to have visible orientation"
    assert area == length, "expected the cue to be a solid line segment"

    rot_k = (-length) % 4
    flip_lr = h > w
    return int(rot_k), bool(flip_lr)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the board cluster and reorient its color layers from outside line cues.

    Input structure:
    - The background is the most frequent color.
    - The largest foreground component is a square multi-color board.
    - Outside the board, some board colors have one separate monochrome cue, and each cue is a straight horizontal or vertical line segment.

    Rule:
    - Crop tightly around the board cluster.
    - For each color inside the crop, find the same-color cue outside the crop.
    - Cue length mod 4 gives the layer's clockwise rotation amount in quarter turns.
    - A vertical cue adds an extra left-right flip after that rotation, while a horizontal cue means rotation only.
    - If a color has no outside cue, keep that layer unchanged.

    Key insight: Every cue carries two visible signals at once: segment length tells how far to rotate, and segment orientation tells whether the rotated layer must also be mirrored.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    # --- STEP 1) Crop the "board" cluster (drop distant cue segments) ---
    crop, (r0, c0, r1, c1), bg = crop_components_near_largest(g, bg=bg, margin=3, connectivity=8)

    # --- STEP 2) For each layer, apply the transform encoded by its outside line cue ---
    colors = colors_present(crop, exclude={int(bg)})
    assert colors, "expected the cropped board to contain foreground colors"
    out = crop.copy()

    outside_grid = g.copy()
    outside_grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = int(bg)

    for col in colors:
        outside_comps = find_components(
            outside_grid,
            bg=int(bg),
            connectivity=8,
            monochromatic=True,
            colors=[int(col)],
            return_format="dict",
        )
        assert len(outside_comps) <= 1, f"expected at most one outside cue for color {col}"
        if not outside_comps:
            continue
        rot_k, flip_lr = _decode_line_cue(outside_comps[0])
        out = transform_color_layer(out, int(col), int(rot_k), flip_lr=bool(flip_lr), bg=int(bg))

    return out


# --------------- generator (att2) ---------------

# ---- Configuration constants ----
CANVAS_SIZE = 20
BOARD_N_MIN, BOARD_N_MAX = 10, 13
NUM_COLORS = 3

OUTER_DENSITY_LO = 0.17
OUTER_DENSITY_HI = 0.32
MID_DENSITY_LO = 0.30
MID_DENSITY_HI = 0.50
MID_OUTER_CAP = 0.32

LAYER_BAND_WIDTH = 2
MIDDLE_INSET = 2
ASYM_MARKER = (1, 2)
INNER_SPAN_MARGIN = 8

CUE_BOARD_DIST = 4
CUE_INTER_DIST = 4
OUTER_AREA_PAD = 4
EDGE_BAND_FOR_LINES = 5

GEN_RETRIES = 400

RESIDUE_TO_LINE_LEN = {0: 4, 1: 5, 2: 2, 3: 3}


def _rot_orbit(r: int, c: int, n: int) -> set[tuple[int, int]]:
    """Return the 4 positions in the 90-degree rotation orbit of (r, c) in an n*n grid."""
    pts = set()
    rr, cc = r, c
    for _ in range(4):
        pts.add((rr, cc))
        rr, cc = n - 1 - cc, rr
    return pts


def _is_rot_asym(mask: np.ndarray) -> bool:
    """True if mask is not invariant under any 90/180/270 rotation."""
    return all(not np.array_equal(mask, np.rot90(mask, k)) for k in (1, 2, 3))


def _is_mirror_asym(mask: np.ndarray) -> bool:
    """True if mask changes under both left-right and up-down mirrors."""
    return (not np.array_equal(mask, np.fliplr(mask))) and (not np.array_equal(mask, np.flipud(mask)))


def _checker_perimeter(n: int, phase: int) -> np.ndarray:
    """Checkerboard pattern restricted to the perimeter of an n*n grid."""
    rr, cc = np.indices((n, n))
    perim = (rr == 0) | (rr == n - 1) | (cc == 0) | (cc == n - 1)
    return perim & (((rr + cc + phase) % 2) == 0)


def _band_cells(n: int, width: int) -> list[tuple[int, int]]:
    """All (r, c) within `width` rows/cols of any edge in an n*n grid."""
    rr, cc = np.indices((n, n))
    inside = np.minimum(np.minimum(rr, cc), np.minimum(n - 1 - rr, n - 1 - cc)) < width
    coords = np.argwhere(inside)
    return [(int(r), int(c)) for r, c in coords]


def _fill_to_count(mask: np.ndarray, candidates: list[tuple[int, int]], target: int) -> bool:
    """Randomly set cells in mask until it reaches target count. Returns success."""
    random.shuffle(candidates)
    for r, c in candidates:
        if mask.sum() >= target:
            return True
        mask[r, c] = True
    return mask.sum() >= target


def _make_asym_layer(
    n: int,
    density_lo: float,
    density_hi: float,
    extra_lo_terms: list[int] = (),
    *,
    require_mirror_asym: bool = False,
) -> np.ndarray | None:
    """Build an n*n boolean mask with checker-perimeter base, asymmetry marker, and random fill."""
    mask = _checker_perimeter(n, random.randrange(2))

    orbit = _rot_orbit(*ASYM_MARKER, n)
    for p in orbit - {ASYM_MARKER}:
        mask[p] = False
    mask[ASYM_MARKER] = True

    lo = max(int(np.ceil(density_lo * n * n)), int(mask.sum()), *extra_lo_terms)
    hi = int(np.floor(density_hi * n * n))
    if lo > hi:
        return None
    target = random.randint(lo, hi)

    blocked = orbit - {ASYM_MARKER}
    candidates = [(r, c) for r, c in _band_cells(n, LAYER_BAND_WIDTH) if (r, c) not in blocked and not mask[r, c]]
    if not _fill_to_count(mask, candidates, target):
        return None
    if not _is_rot_asym(mask):
        return None
    if require_mirror_asym and not _is_mirror_asym(mask):
        return None
    return mask


def _make_middle_mask(n: int, *, require_mirror_asym: bool = False) -> np.ndarray | None:
    """Build the middle concentric layer, inset by MIDDLE_INSET from each edge."""
    m = n - 2 * MIDDLE_INSET
    hi_cap = int(np.floor(MID_OUTER_CAP * n * n)) - 5
    local = _make_asym_layer(
        m,
        MID_DENSITY_LO,
        min(MID_DENSITY_HI, hi_cap / (m * m)),
        require_mirror_asym=require_mirror_asym,
    )
    if local is None:
        return None
    mask = np.zeros((n, n), dtype=bool)
    mask[MIDDLE_INSET : MIDDLE_INSET + m, MIDDLE_INSET : MIDDLE_INSET + m] = local
    if not _is_rot_asym(mask):
        return None
    if require_mirror_asym and not _is_mirror_asym(mask):
        return None
    return mask


def _make_outer_mask(n: int, min_area: int, *, require_mirror_asym: bool = False) -> np.ndarray | None:
    """Build the outer frame layer spanning the full n*n grid."""
    return _make_asym_layer(
        n,
        OUTER_DENSITY_LO,
        OUTER_DENSITY_HI,
        extra_lo_terms=[min_area],
        require_mirror_asym=require_mirror_asym,
    )


def _valid_inner_spans(n: int, require_asym: bool) -> list[int]:
    """Return valid inner-sprite span sizes for a board of side n."""
    max_span = n - INNER_SPAN_MARGIN
    spans = [s for s in range(1, max_span + 1) if (s % 2) == (n % 2)]
    if require_asym:
        spans = [s for s in spans if s > 1]
    return spans


def _make_inner_mask(n: int, *, require_rot_asym: bool, require_mirror_asym: bool) -> np.ndarray | None:
    """Build the small central sprite layer, retrying until symmetry constraints are met."""
    spans = _valid_inner_spans(n, require_rot_asym or require_mirror_asym)
    for _ in range(80):
        mask = np.zeros((n, n), dtype=bool)
        span = random.choice(spans)
        top = (n - span) // 2

        if span == 1:
            local = np.ones((1, 1), dtype=bool)
        elif span == 2:
            local = np.ones((2, 2), dtype=bool)
            local[random.randrange(2), random.randrange(2)] = False
        elif span == 3:
            variants = [
                np.array([[1, 0, 0], [0, 1, 0], [1, 0, 1]], dtype=bool),
                np.array([[0, 1, 1], [0, 1, 0], [1, 0, 0]], dtype=bool),
                np.array([[1, 0, 1], [0, 1, 0], [0, 0, 1]], dtype=bool),
            ]
            local = random.choice(variants)
            local = np.rot90(local, random.randrange(4))
        else:
            local = np.zeros((span, span), dtype=bool)
            if span == 4:
                pts = [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1), (2, 2), (2, 3), (3, 2)]
            else:
                pts = [(0, 2), (1, 1), (1, 2), (1, 3), (2, 0), (2, 1), (2, 2), (2, 3), (3, 3), (4, 4)]
            for r, c in pts:
                local[r, c] = True
            if random.randrange(2):
                local = np.fliplr(local)
            local = np.rot90(local, random.randrange(4))

        mask[top : top + local.shape[0], top : top + local.shape[1]] = local
        if require_rot_asym and not _is_rot_asym(mask):
            continue
        if require_mirror_asym and not _is_mirror_asym(mask):
            continue
        return mask
    return None


def _pick_transforms() -> list[tuple[int, bool]]:
    """
    Pick (rot_k, flip_lr) for 3 colors.
    Guarantees:
    - at least one cue exists
    - at least one vertical cue exists
    - at least one non-zero rotation exists
    - when there are multiple cues, both cue orientations appear
    """
    cue_count = random.choice([1, 2, 2, 3])
    active = random.sample(range(NUM_COLORS), cue_count)
    transforms = [(0, False) for _ in range(NUM_COLORS)]

    if cue_count == 1:
        transforms[active[0]] = (random.choice([1, 2, 3]), True)
        return transforms

    vertical_idx = random.choice(active)
    horizontal_idx = random.choice([idx for idx in active if idx != vertical_idx])
    transforms[vertical_idx] = (random.randrange(4), True)
    transforms[horizontal_idx] = (random.choice([1, 2, 3]), False)

    for idx in active:
        if idx in {vertical_idx, horizontal_idx}:
            continue
        transforms[idx] = (random.randrange(4), bool(random.randrange(2)))
        if transforms[idx] == (0, False):
            transforms[idx] = (0, True)

    if not any(rot_k != 0 for rot_k, _flip in transforms):
        idx = random.choice(active)
        transforms[idx] = (random.choice([1, 2, 3]), transforms[idx][1] or idx == vertical_idx)

    if not any(flip for _rot_k, flip in transforms):
        idx = random.choice(active)
        transforms[idx] = (transforms[idx][0] or random.choice([1, 2, 3]), True)

    return transforms


def _cue_patch(color: int, rot_k: int, *, flip_lr: bool) -> np.ndarray:
    """Return a monochrome line cue whose length/orientation encodes (rot_k, flip_lr)."""
    residue = (-int(rot_k)) % 4
    length = int(RESIDUE_TO_LINE_LEN[residue])
    if flip_lr:
        return np.full((length, 1), int(color), dtype=int)
    return np.full((1, length), int(color), dtype=int)


def _bbox_chebyshev(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    """Chebyshev (L-inf) gap distance between two axis-aligned bounding boxes."""
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    dr = max(0, br0 - ar1) if ar1 < br0 else max(0, ar0 - br1)
    dc = max(0, bc0 - ac1) if ac1 < bc0 else max(0, ac0 - bc1)
    return max(dr, dc)


def _place_cue(
    canvas: np.ndarray,
    patch: np.ndarray,
    board_bbox: tuple[int, int, int, int],
    cue_bboxes: list[tuple[int, int, int, int]],
) -> bool:
    """Place a cue patch on the canvas, respecting distance constraints. Returns success."""
    h, w = patch.shape
    mask = patch != 0
    candidates = []
    for r in range(CANVAS_SIZE - h + 1):
        for c in range(CANVAS_SIZE - w + 1):
            if np.any((canvas[r : r + h, c : c + w] != 0) & mask):
                continue
            bb = (r, c, r + h - 1, c + w - 1)
            if _bbox_chebyshev(bb, board_bbox) < CUE_BOARD_DIST:
                continue
            if any(_bbox_chebyshev(bb, prev) < CUE_INTER_DIST for prev in cue_bboxes):
                continue
            candidates.append((r, c, bb))
    if not candidates:
        return False
    r, c, bb = random.choice(candidates)
    paste(canvas, patch, r, c, transparent=0)
    cue_bboxes.append(bb)
    return True


def _board_anchor(n: int, cue_items: list[tuple[int, int, bool]]) -> tuple[int, int]:
    """
    Pick a board top-left position with enough free side-band to host the required line orientations.
    Horizontal cues need a top/bottom band of >= 5; vertical cues need a left/right band of >= 5.
    """
    spare = CANVAS_SIZE - n
    need_row_band = any(not flip_lr for _col, _rot_k, flip_lr in cue_items)
    need_col_band = any(flip_lr for _col, _rot_k, flip_lr in cue_items)

    row_candidates = [
        r for r in range(spare + 1) if (not need_row_band) or max(int(r), int(spare - r)) >= EDGE_BAND_FOR_LINES
    ]
    col_candidates = [
        c for c in range(spare + 1) if (not need_col_band) or max(int(c), int(spare - c)) >= EDGE_BAND_FOR_LINES
    ]
    assert row_candidates and col_candidates, "expected at least one legal board anchor"
    return random.choice(row_candidates), random.choice(col_candidates)


def _apply_output_transform(mask: np.ndarray, rot_k: int, *, flip_lr: bool) -> np.ndarray:
    """Apply the solver-side transform to a boolean mask."""
    out = np.rot90(mask, int(rot_k) % 4)
    if flip_lr:
        out = np.fliplr(out)
    return out


def _invert_output_transform(mask: np.ndarray, rot_k: int, *, flip_lr: bool) -> np.ndarray:
    """Invert the solver-side transform for output-first generation."""
    out = np.fliplr(mask) if flip_lr else mask.copy()
    return np.rot90(out, (-int(rot_k)) % 4)


def _gen_attempt() -> np.ndarray | None:
    """Single attempt to generate a valid input grid. Returns None on failure."""
    colors = random.sample(range(1, 10), NUM_COLORS)
    transforms = _pick_transforms()
    require_rot = [rot_k != 0 for rot_k, _flip in transforms]
    require_mirror = [flip for _rot_k, flip in transforms]

    valid_ns = [
        n
        for n in range(BOARD_N_MIN, BOARD_N_MAX + 1)
        if _valid_inner_spans(n, require_rot[2] or require_mirror[2])
    ]
    n = random.choice(valid_ns)

    mid_mask = _make_middle_mask(n, require_mirror_asym=require_mirror[1])
    if mid_mask is None:
        return None
    inner_mask = _make_inner_mask(
        n,
        require_rot_asym=require_rot[2],
        require_mirror_asym=require_mirror[2],
    )
    if inner_mask is None:
        return None
    outer_mask = _make_outer_mask(n, int(mid_mask.sum()) + OUTER_AREA_PAD, require_mirror_asym=require_mirror[0])
    if outer_mask is None:
        return None

    if np.any(outer_mask & mid_mask) or np.any(outer_mask & inner_mask) or np.any(mid_mask & inner_mask):
        return None

    masks = [outer_mask, mid_mask, inner_mask]
    for mask, needs_rot, needs_mirror in zip(masks, require_rot, require_mirror):
        if needs_rot and not _is_rot_asym(mask):
            return None
        if needs_mirror and not _is_mirror_asym(mask):
            return None

    out_board = np.zeros((n, n), dtype=int)
    for col, mask in zip(colors, masks):
        out_board[mask] = col

    in_board = np.zeros((n, n), dtype=int)
    for col, mask, (rot_k, flip_lr) in zip(colors, masks, transforms):
        inv = _invert_output_transform(mask, int(rot_k), flip_lr=bool(flip_lr))
        if np.any((in_board != 0) & inv):
            return None
        in_board[inv] = int(col)

    cue_items = [(int(col), int(rot_k), bool(flip_lr)) for col, (rot_k, flip_lr) in zip(colors, transforms) if rot_k != 0 or flip_lr]
    if not cue_items:
        return None

    r0, c0 = _board_anchor(n, cue_items)
    canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=int)
    paste(canvas, in_board, r0, c0, transparent=0)
    board_bbox = (r0, c0, r0 + n - 1, c0 + n - 1)

    cue_bboxes: list[tuple[int, int, int, int]] = []
    random.shuffle(cue_items)
    for col, rot_k, flip_lr in cue_items:
        if not _place_cue(canvas, _cue_patch(col, rot_k, flip_lr=flip_lr), board_bbox, cue_bboxes):
            return None

    comps = find_components(canvas, bg=0, connectivity=8, monochromatic=False, return_format="dict")
    areas = [int(comp["area"]) for comp in comps]
    if areas.count(max(areas)) != 1:
        return None
    largest = comps[areas.index(max(areas))]
    if tuple(map(int, largest["bbox"])) != board_bbox:
        return None
    for bb in cue_bboxes:
        if _bbox_chebyshev(bb, board_bbox) < CUE_BOARD_DIST:
            return None

    if not np.array_equal(main(canvas), out_board):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 6ffbe589 D4 input")
