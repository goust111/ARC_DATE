# Variant of 6ffbe589
# Dimension: D3 — 变换类型：旋转 (transform)
# Change: np.rot90(layer, -cue_size % 4)，对每个颜色层独立执行顺时针90度整数倍旋转... → 镜像翻转代替旋转
#
# design_analysis:
#   a_specific_behavior_and_trace:
#     - 仍然只有一个最大的多色方形主板，外部散落少量同色提示精灵，solver 先裁出主板再逐色处理。
#     - 原题中，某颜色外部提示的“像素数 mod 4”决定该颜色层旋转多少个 90 度；变体里改成提示精灵的轮廓直接决定翻转类型。
#     - 这里固定 3 种提示轮廓：
#       1) `1x3` 水平线 -> `np.fliplr`，即左右镜像；
#       2) `3x1` 垂直线 -> `np.flipud`，即上下镜像；
#       3) `3x3` 主对角线三连点 -> `mask.T`，即沿主对角线翻转。
#     - 小例子追踪：
#       原逻辑：若红色 cue 面积为 3，则 solver 算出 `(-3) % 4 = 1`，把红层顺时针旋转 90 度。
#       新逻辑：若红色 cue 是 `1x3` 水平线，则 solver 识别为左右镜像，把红层执行 `fliplr`。
#       差异：控制变量不再是数量，而是几何轮廓；输出变化不再是转向，而是成对像素的镜像交换。
#   b_complexity_calibration:
#     - 只保留 3 档翻转类型，不引入反对角线等额外分支，避免样例过少时分支覆盖不足。
#     - 同时 3 个 cue 轮廓在视觉上差异足够大，不会退化成“又是数格子”的旧题体验。
#     - 由于翻转比旋转更依赖不对称性，generator 对被 cue 的层强制制造该翻转下可见的不对称。
#   c_implicit_assumption_audit_on_main:
#     - `crop_components_near_largest(...)` 依赖“主板仍是唯一最大前景组件”，变体不改这一点，generator 继续保证。
#     - `colors_present(crop, exclude={bg})` 依赖主板裁剪后仍只包含板内颜色；cue 仍需远离主板，避免被裁入。
#     - 原代码里 `cue_size -> rot_k` 假设 cue 的信息在面积里；这里必须改成“从组件 bbox 内的精确轮廓解码”。
#     - 原 `rotate_color_layer` 依赖旋转后形状仍在同一方形画布内；这里保留同一前提，尤其对主对角线翻转要求 crop 为方形。
#     - 顺序处理各颜色层仍然安全，因为 generator 保证各层目标位置两两不重叠；否则后处理颜色会覆盖先处理颜色。
#   d_placement_legality:
#     - 主板仍放在 20x20 画布某一边缘，cue 与主板 bbox 的 Chebyshev 距离 >= 4，cue 之间也保持 >= 4。
#     - cue 轮廓最大为 3x3，不越界且不会被 margin=3 的裁剪误纳入主板 crop。
#     - output-first 保留：先造期望输出板，再对每个被 cue 的颜色层施加同一种自反射得到输入板。
#   e_visual_observability:
#     - cue 的几何轮廓就是规则说明：横线/竖线/对角线分别对应三种镜像，观察者可直接从 input->output 的位移方向归纳。
#     - 为避免“翻了但看不出来”，每个被 cue 的颜色层都保证在对应翻转下发生可见变化。
#   f_visual_cleanliness:
#     - 主板仍是三个同心层级：外框层、中层、中心小精灵；cue 仍全部在板外，画面结构清楚。
#     - 不把多个编码叠加在同一 cue 上，也不使用颜色排序、坐标奇偶等隐式不可见信息。
#   g_color_inference:
#     - 颜色不发生重映射；同色 cue 只控制同色层，和原题一致，因此解题重点聚焦在“哪一层做哪种翻转”。
#   main_new_assumptions_and_generator_guarantees:
#     - 假设每种板内颜色在板外至多有一个同色 cue；generator 继续一色至多放一个 cue。
#     - 假设 cue 只会是 3 种模板之一；generator 只生成这 3 种精确模板。
#     - 假设需要主对角线翻转时，裁出的板是正方形；generator 的主板始终是 NxN，且 cue 不会污染裁剪框。

from __future__ import annotations

import random

import numpy as np

from arc_utils import colors_present, find_components, get_bg, paste


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


TRANSFORMS = ("h", "v", "d")

CUE_TEMPLATES = {
    "h": np.array([[1, 1, 1]], dtype=bool),
    "v": np.array([[1], [1], [1]], dtype=bool),
    "d": np.eye(3, dtype=bool),
}


def apply_transform(mask: np.ndarray, transform: str | None) -> np.ndarray:
    """Apply one of the three cue-encoded reflections to a square layer mask."""
    if transform is None:
        return mask.copy()
    if transform == "h":
        return np.fliplr(mask)
    if transform == "v":
        return np.flipud(mask)
    if transform == "d":
        assert mask.shape[0] == mask.shape[1], "main-diagonal reflection expects a square crop"
        return mask.T
    raise AssertionError(f"unknown transform: {transform}")


def decode_cue_transform(grid: np.ndarray, comp: dict, color: int) -> str:
    """Decode a same-color outside cue component into h/v/d reflection type."""
    r0, c0, r1, c1 = map(int, comp["bbox"])
    patch = grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] == int(color)

    if patch.shape == (1, 3) and bool(np.all(patch)):
        return "h"
    if patch.shape == (3, 1) and bool(np.all(patch)):
        return "v"
    if patch.shape == (3, 3) and np.array_equal(patch, CUE_TEMPLATES["d"]):
        return "d"
    raise AssertionError(f"unrecognized cue shape for color {color}")


def reflect_color_layer(grid: np.ndarray, color: int, transform: str | None, *, bg: int | None = None) -> np.ndarray:
    """Mirror one color layer inside the fixed board crop."""
    if transform is None:
        return grid
    if bg is None:
        bg = get_bg(grid)
    bg = int(bg)

    col = int(color)
    mask = grid == col
    tmask = apply_transform(mask, transform)

    out = grid.copy()
    out[out == col] = bg
    out[tmask] = col
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract the board cluster and mirror selected color layers from same-color cue glyphs.

    Input structure:
    - The background is the dominant color.
    - One large square multi-color board is the main foreground component.
    - Small monochrome cue glyphs sit outside the board; each cue shares a color with one board layer.

    Rule:
    - Crop tightly around the board cluster and ignore the distant cue glyphs.
    - For each color layer in the crop, look for an outside cue of the same color.
    - A horizontal 3-cell bar means left-right mirror, a vertical 3-cell bar means top-bottom mirror, and a 3-cell main diagonal means main-diagonal reflection.
    - If a color has no cue glyph, leave that layer unchanged.

    Key insight: The outside cue no longer uses its pixel count as a rotation code; its silhouette directly names the reflection to apply to the matching color mask.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    # --- STEP 1) Crop the main board cluster, leaving far-away cue glyphs behind ---
    crop, (r0, c0, r1, c1), bg = crop_components_near_largest(g, bg=bg, margin=3, connectivity=8)

    # --- STEP 2) Decode each same-color cue glyph and mirror that layer accordingly ---
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
        transform = decode_cue_transform(outside_grid, outside_comps[0], int(col)) if outside_comps else None
        out = reflect_color_layer(out, int(col), transform, bg=int(bg))

    return out


# --------------- generator ---------------

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

GEN_RETRIES = 300


def _mirror_partner(r: int, c: int, n: int, transform: str) -> tuple[int, int]:
    """Return the partner cell paired with (r, c) by the requested reflection."""
    if transform == "h":
        return r, n - 1 - c
    if transform == "v":
        return n - 1 - r, c
    if transform == "d":
        return c, r
    raise AssertionError(f"unknown transform: {transform}")


def _is_visible_under(mask: np.ndarray, transform: str | None) -> bool:
    """True iff applying the requested reflection actually changes the mask."""
    if transform is None:
        return True
    return not np.array_equal(mask, apply_transform(mask, transform))


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
        if int(mask.sum()) >= int(target):
            return True
        mask[r, c] = True
    return int(mask.sum()) >= int(target)


def _make_asym_layer(
    n: int,
    density_lo: float,
    density_hi: float,
    *,
    transform: str | None = None,
    extra_lo_terms: list[int] = (),
) -> np.ndarray | None:
    """Build a perimeter-heavy layer, forcing visibility under one requested reflection when needed."""
    mask = _checker_perimeter(n, random.randrange(2))
    blocked: set[tuple[int, int]] = set()

    if transform is not None:
        partner = _mirror_partner(*ASYM_MARKER, n, transform)
        assert partner != ASYM_MARKER, "asymmetry marker must not lie on the chosen reflection axis"
        mask[ASYM_MARKER] = True
        mask[partner] = False
        blocked.add(partner)

    lo = max(int(np.ceil(density_lo * n * n)), int(mask.sum()), *extra_lo_terms)
    hi = int(np.floor(density_hi * n * n))
    if lo > hi:
        return None
    target = random.randint(lo, hi)

    candidates = [
        (r, c)
        for r, c in _band_cells(n, LAYER_BAND_WIDTH)
        if (r, c) not in blocked and not mask[r, c]
    ]
    if not _fill_to_count(mask, candidates, target):
        return None
    if not _is_visible_under(mask, transform):
        return None
    return mask


def _make_middle_mask(n: int, transform: str | None) -> np.ndarray | None:
    """Build the middle concentric layer, inset by MIDDLE_INSET from each edge."""
    m = n - 2 * MIDDLE_INSET
    hi_cap = int(np.floor(MID_OUTER_CAP * n * n)) - 5
    local = _make_asym_layer(
        m,
        MID_DENSITY_LO,
        min(MID_DENSITY_HI, hi_cap / (m * m)),
        transform=transform,
    )
    if local is None:
        return None
    mask = np.zeros((n, n), dtype=bool)
    mask[MIDDLE_INSET : MIDDLE_INSET + m, MIDDLE_INSET : MIDDLE_INSET + m] = local
    if not _is_visible_under(mask, transform):
        return None
    return mask


def _make_outer_mask(n: int, min_area: int, transform: str | None) -> np.ndarray | None:
    """Build the outer frame layer spanning the full n*n grid."""
    return _make_asym_layer(
        n,
        OUTER_DENSITY_LO,
        OUTER_DENSITY_HI,
        transform=transform,
        extra_lo_terms=[min_area],
    )


def _valid_inner_spans(n: int, transform: str | None) -> list[int]:
    """Return valid inner-sprite span sizes for a board of side n."""
    max_span = n - INNER_SPAN_MARGIN
    spans = [s for s in range(1, max_span + 1) if (s % 2) == (n % 2)]
    if transform is not None:
        spans = [s for s in spans if s > 1]
    return spans


def _make_inner_local(span: int) -> np.ndarray:
    """Build a small centered local sprite before embedding it into the board."""
    if span == 1:
        return np.ones((1, 1), dtype=bool)
    if span == 2:
        local = np.ones((2, 2), dtype=bool)
        local[random.randrange(2), random.randrange(2)] = False
        return local
    if span == 3:
        variants = [
            np.array([[1, 0, 0], [0, 1, 0], [1, 0, 1]], dtype=bool),
            np.array([[0, 1, 1], [0, 1, 0], [1, 0, 0]], dtype=bool),
            np.array([[1, 0, 1], [0, 1, 0], [0, 0, 1]], dtype=bool),
        ]
        local = random.choice(variants).copy()
        return np.rot90(local, random.randrange(4))

    local = np.zeros((span, span), dtype=bool)
    if span == 4:
        pts = [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1), (2, 2), (2, 3), (3, 2)]
    else:
        pts = [(0, 2), (1, 1), (1, 2), (1, 3), (2, 0), (2, 1), (2, 2), (2, 3), (3, 3), (4, 4)]
    for r, c in pts:
        local[r, c] = True
    if random.randrange(2):
        local = np.fliplr(local)
    return np.rot90(local, random.randrange(4))


def _make_inner_mask(n: int, transform: str | None) -> np.ndarray | None:
    """Build the small central sprite layer and ensure cue-driven flips stay visible."""
    spans = _valid_inner_spans(n, transform)
    if not spans:
        return None

    for _ in range(30):
        span = random.choice(spans)
        local = _make_inner_local(span)
        mask = np.zeros((n, n), dtype=bool)
        top = (n - span) // 2
        mask[top : top + span, top : top + span] = local
        if _is_visible_under(mask, transform):
            return mask
    return None


def _pick_transforms() -> list[str | None]:
    """Pick cue-driven reflections for the 3 color layers; at least one must be active."""
    transforms = [random.choice([None, *TRANSFORMS]) for _ in range(3)]
    if not any(t is not None for t in transforms):
        transforms[random.randrange(3)] = random.choice(list(TRANSFORMS))
    return transforms


def _cue_patch(color: int, transform: str) -> np.ndarray:
    """Return the fixed cue glyph for one reflection type."""
    template = CUE_TEMPLATES[transform]
    patch = np.zeros(template.shape, dtype=int)
    patch[template] = int(color)
    return patch


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


def _board_anchor(n: int) -> tuple[int, int]:
    """Pick a random board top-left position, flush against one edge of the canvas."""
    spare = CANVAS_SIZE - n
    side = random.randrange(4)
    if side == 0:
        return spare, random.randint(0, spare)
    if side == 1:
        return 0, random.randint(0, spare)
    if side == 2:
        return random.randint(0, spare), spare
    return random.randint(0, spare), 0


def _gen_attempt() -> np.ndarray | None:
    """Single attempt to generate a valid input grid. Returns None on failure."""
    colors = random.sample(range(1, 10), NUM_COLORS)
    transforms = _pick_transforms()

    valid_ns = [
        n
        for n in range(BOARD_N_MIN, BOARD_N_MAX + 1)
        if _valid_inner_spans(n, transforms[2])
    ]
    n = random.choice(valid_ns)

    # Build concentric output masks that will be recovered after cue-driven reflections.
    mid_mask = _make_middle_mask(n, transforms[1])
    if mid_mask is None:
        return None
    inner_mask = _make_inner_mask(n, transforms[2])
    if inner_mask is None:
        return None
    outer_mask = _make_outer_mask(n, int(mid_mask.sum()) + OUTER_AREA_PAD, transforms[0])
    if outer_mask is None:
        return None

    if np.any(outer_mask & mid_mask) or np.any(outer_mask & inner_mask) or np.any(mid_mask & inner_mask):
        return None

    out_board = np.zeros((n, n), dtype=int)
    for color, mask in zip(colors, (outer_mask, mid_mask, inner_mask)):
        out_board[mask] = int(color)

    # Reflections are self-inverse, so the input layer is just the reflected output layer.
    in_board = np.zeros((n, n), dtype=int)
    for col, transform in zip(colors, transforms):
        inv = apply_transform(out_board == col, transform)
        if np.any((in_board != 0) & inv):
            return None
        in_board[inv] = int(col)

    r0, c0 = _board_anchor(n)
    canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=int)
    paste(canvas, in_board, r0, c0, transparent=0)
    board_bbox = (r0, c0, r0 + n - 1, c0 + n - 1)

    cue_bboxes: list[tuple[int, int, int, int]] = []
    cue_items = [(col, transform) for col, transform in zip(colors, transforms) if transform is not None]
    random.shuffle(cue_items)
    for col, transform in cue_items:
        if not _place_cue(canvas, _cue_patch(col, transform), board_bbox, cue_bboxes):
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
    raise RuntimeError("failed to generate a valid 6ffbe589 D3 input")
