# Combo variant of 6ffbe589 (index 2)
# combo_spec: D3/0+D7/1
# Name: 镜像翻转+位置决定方向
#
# design_analysis:
#   a_specific_behavior_and_trace:
#     - 仍然只有一个最大的多色方形主板，solver 先裁出主板，再按颜色去板外找同色 cue。
#     - 原题链路是：cue 面积 mod 4 -> 顺时针旋转若干个 90 度。这里改成：cue 的相对方位
#       -> 镜像轴方向。上/下侧 cue 触发沿水平轴翻转（`np.flipud`），左/右侧 cue 触发沿
#       垂直轴翻转（`np.fliplr`）。cue 的大小和具体轮廓不再携带规则信息，只要能被识别
#       为同色外部提示即可。
#     - 小例子：假设红色层在输入里是输出层做过一次上下翻转后的样子。若红色 cue 位于主板
#       下方，原题会数 cue 面积并做旋转；融合后 solver 只看“它在下方”，于是把红层执行
#       `flipud` 还原。若同样的红色 cue 改放到右侧，则同一颜色层会改做 `fliplr`。
#     - 差异在于：控制变量从“数格子”变成“看方位”，输出变化从旋转改成镜像，而且位置与
#       变换类型直接耦合，不能再把两步分开理解。
#   b_complexity_calibration:
#     - 只保留两类可见翻转：水平轴翻转和垂直轴翻转。这样不会像引入对角线那样分支过多，
#       但仍明显区别于原题的旋转。
#     - cue 继续是一色一个、放在板外，避免多 cue 竞争同一颜色造成匹配歧义。
#     - generator 在多 cue 情况下尽量同时出现两种轴向，避免样例退化成“所有东西都朝同一
#       方向翻”而让位置线索不够充分。
#   c_implicit_assumption_audit_on_main:
#     - `crop_components_near_largest(...)` 仍假设主板是唯一最大前景组件；generator 保持这一点。
#     - `colors_present(crop, exclude={bg})` 仍假设 cue 不会被裁进主板 crop；因此 cue 与主板
#       的 Chebyshev 距离继续保持至少 4，而裁剪 margin 仍是 3。
#     - 新的 `_cue_side(...)` 假设 cue 能被唯一分类为 top/bottom/left/right 之一，不能斜对角
#       歪放；generator 用 side-constrained 走廊放置保证这一点。
#     - `reflect_color_layer(...)` 假设按颜色顺序逐层翻转不会发生目标重叠覆盖；generator 继续
#       采用 output-first，再对每层做精确自反变换，保证输入层仍两两不重叠。
#     - 融合后不再读取 cue 面积或轮廓，因此 generator 必须避免无意中把“大小”做成稳定编码；
#       这里让 cue 面积在 1..5 间随机变化，仅由位置控制行为。
#   d_placement_legality:
#     - 20x20 画布配 10..13 的主板，若要求 cue 距板至少 4 格，不适合同时开放四边自由放置。
#       因此主板固定锚在四个角之一，只留下相邻的两条长走廊供 cue 使用；这两条走廊天然分别
#       对应一个水平轴翻转侧和一个垂直轴翻转侧。
#     - cue 仍是 8 连通单色小 blob，面积 1..5，不越界、不互相重叠，并与主板/其他 cue 维持
#       距离约束。
#   e_visual_observability:
#     - 人类可直接看到：同色 cue 在板的上/下侧时，该层像素上下对调；在左/右侧时，该层像素
#       左右对调。颜色不改、只改几何，因此线索集中而明确。
#     - 为避免“翻了但看不出来”，每个带 cue 的层都强制在对应翻转下发生可见变化。
#     - cue 的面积和轮廓在样本中会变化，但输出只跟位置有关，这能帮助排除旧题那种面积编码。
#   f_visual_cleanliness:
#     - 主板仍保持外框层 / 中层 / 中心小精灵三层同心结构，cue 全在板外，视觉分区清晰。
#     - 一个 cue 只承载“此颜色要变换”与“它位于哪一侧”两件事，不叠加额外编码。
#   g_color_inference:
#     - 颜色没有重映射；同色 cue 只作用于同色层。解题者只需匹配同色对象并读取其相对方位，
#       不需要推断任何颜色排序或数值含义。
#   main_new_assumptions_and_generator_guarantees:
#     - 假设每个颜色在板外至多一个同色 cue；generator 保证一色至多放一个。
#     - 假设每个 cue 与主板在正交投影上有重叠，从而 side 判定唯一；generator 的侧边放置器
#       逐个检查这一条件。
#     - 假设带 cue 的层在指定翻转下确实可见变化；generator 为外层、中层、内层都显式验证。

from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

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

    h, w = grid.shape
    er0 = max(0, int(mr0 - margin))
    ec0 = max(0, int(mc0 - margin))
    er1 = min(int(h - 1), int(mr1 + margin))
    ec1 = min(int(w - 1), int(mc1 + margin))

    keep_mask = np.zeros((h, w), dtype=bool)
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


TRANSFORMS = ("horizontal", "vertical")
SIDE_TO_TRANSFORM = {
    "top": "horizontal",
    "bottom": "horizontal",
    "left": "vertical",
    "right": "vertical",
}


def apply_transform(mask: np.ndarray, transform: str | None) -> np.ndarray:
    """Apply one cue-driven reflection to a layer mask."""
    if transform is None:
        return mask.copy()
    if transform == "horizontal":
        return np.flipud(mask)
    if transform == "vertical":
        return np.fliplr(mask)
    raise AssertionError(f"unknown transform: {transform}")


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


def _bbox_overlap_1d(a0: int, a1: int, b0: int, b1: int) -> bool:
    """True iff two inclusive 1D intervals overlap."""
    return not (a1 < b0 or b1 < a0)


def _cue_side(cue_bbox: tuple[int, int, int, int], board_bbox: tuple[int, int, int, int]) -> str:
    """Classify a cue as top/bottom/left/right relative to the board, rejecting diagonal ambiguity."""
    qr0, qc0, qr1, qc1 = map(int, cue_bbox)
    br0, bc0, br1, bc1 = map(int, board_bbox)

    if qr1 < br0 and _bbox_overlap_1d(qc0, qc1, bc0, bc1):
        return "top"
    if qr0 > br1 and _bbox_overlap_1d(qc0, qc1, bc0, bc1):
        return "bottom"
    if qc1 < bc0 and _bbox_overlap_1d(qr0, qr1, br0, br1):
        return "left"
    if qc0 > bc1 and _bbox_overlap_1d(qr0, qr1, br0, br1):
        return "right"
    raise AssertionError("expected each cue to be aligned to exactly one board side")


def _cue_transform_from_side(cue_comp: dict, board_bbox: tuple[int, int, int, int]) -> str:
    """Map a same-color cue component to the reflection implied by its side."""
    side = _cue_side(tuple(map(int, cue_comp["bbox"])), board_bbox)
    return SIDE_TO_TRANSFORM[side]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the board cluster and mirror its color layers according to cue position.

    Input structure:
    - The background is the most common color in the canvas.
    - One large square multi-color board is the unique largest foreground component.
    - Small monochrome same-color cues sit outside the board and are aligned to one board side.

    Rule:
    - Tight-crop around the board cluster and ignore the distant cue sprites.
    - For each color layer inside the crop, locate the outside cue of the same color.
    - A cue above or below the board means reflect that layer across a horizontal axis.
    - A cue to the left or right of the board means reflect that layer across a vertical axis.
    - If a color has no cue, keep that layer unchanged.

    Key insight:
    - The cue no longer encodes a rotation amount. Its only job is to point from outside the board,
      and that relative side directly selects the mirror axis for the matching color layer.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    crop, (r0, c0, r1, c1), bg = crop_components_near_largest(g, bg=bg, margin=3, connectivity=8)
    board_bbox = (int(r0), int(c0), int(r1), int(c1))

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
        transform = _cue_transform_from_side(outside_comps[0], board_bbox) if outside_comps else None
        out = reflect_color_layer(out, int(col), transform, bg=int(bg))

    return out


# --------------- generator ---------------

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

CUE_GRID = 3
CUE_BOARD_DIST = 4
CUE_INTER_DIST = 4
OUTER_AREA_PAD = 4

GEN_RETRIES = 300


def _partner_under_transform(r: int, c: int, n: int, transform: str) -> tuple[int, int]:
    """Return the cell paired with (r, c) by the requested reflection."""
    if transform == "horizontal":
        return n - 1 - r, c
    if transform == "vertical":
        return r, n - 1 - c
    raise AssertionError(f"unknown transform: {transform}")


def _is_visible_under(mask: np.ndarray, transform: str | None) -> bool:
    """True iff applying the requested reflection changes the mask."""
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
        partner = _partner_under_transform(*ASYM_MARKER, n, transform)
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

    for _ in range(40):
        span = random.choice(spans)
        local = _make_inner_local(span)
        mask = np.zeros((n, n), dtype=bool)
        top = (n - span) // 2
        mask[top : top + span, top : top + span] = local
        if _is_visible_under(mask, transform):
            return mask
    return None


def _pick_transforms() -> list[str | None]:
    """Pick layer reflections so every sample exposes both cue-direction branches."""
    transforms: list[str | None] = [None, None, None]
    active_indices = random.sample(range(3), random.choice((2, 3)))
    first, second = random.sample(active_indices, 2)
    transforms[first] = "horizontal"
    transforms[second] = "vertical"
    for idx in active_indices:
        if transforms[idx] is None:
            transforms[idx] = random.choice(list(TRANSFORMS))
    return transforms


def _cue_patch(color: int, area: int) -> np.ndarray:
    """Grow an 8-connected monochromatic blob of given area in a CUE_GRID*CUE_GRID workspace, then crop."""
    center = CUE_GRID // 2
    coords = {(center, center)}
    while len(coords) < area:
        frontier = set()
        for r, c in coords:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < CUE_GRID and 0 <= nc < CUE_GRID and (nr, nc) not in coords:
                        frontier.add((nr, nc))
        coords.add(random.choice(sorted(frontier)))

    arr = np.zeros((CUE_GRID, CUE_GRID), dtype=int)
    for r, c in coords:
        arr[r, c] = int(color)
    pos = np.argwhere(arr != 0)
    r0, c0 = pos.min(axis=0)
    r1, c1 = pos.max(axis=0)
    return arr[r0 : r1 + 1, c0 : c1 + 1]


def _bbox_chebyshev(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    """Chebyshev (L-inf) gap distance between two axis-aligned bounding boxes."""
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    dr = max(0, br0 - ar1) if ar1 < br0 else max(0, ar0 - br1)
    dc = max(0, bc0 - ac1) if ac1 < bc0 else max(0, ac0 - bc1)
    return max(dr, dc)


def _corner_anchor_and_sides(n: int) -> tuple[tuple[int, int], tuple[str, str]]:
    """Pick a corner anchor and return the two cue corridors available around that corner placement."""
    spare = CANVAS_SIZE - n
    corner = random.randrange(4)
    if corner == 0:
        return (0, 0), ("bottom", "right")
    if corner == 1:
        return (0, spare), ("bottom", "left")
    if corner == 2:
        return (spare, 0), ("top", "right")
    return (spare, spare), ("top", "left")


def _place_cue_on_side(
    canvas: np.ndarray,
    patch: np.ndarray,
    board_bbox: tuple[int, int, int, int],
    cue_bboxes: list[tuple[int, int, int, int]],
    side: str,
) -> bool:
    """Place a cue patch on a specified board side with orthogonal-span overlap."""
    h, w = patch.shape
    mask = patch != 0
    br0, bc0, br1, bc1 = map(int, board_bbox)
    candidates = []

    if side == "top":
        r_vals = range(0, br0 - h - CUE_BOARD_DIST + 1)
        c_min = max(0, bc0 - w + 1)
        c_max = min(CANVAS_SIZE - w, bc1)
        c_vals = range(c_min, c_max + 1)
    elif side == "bottom":
        r_vals = range(br1 + CUE_BOARD_DIST, CANVAS_SIZE - h + 1)
        c_min = max(0, bc0 - w + 1)
        c_max = min(CANVAS_SIZE - w, bc1)
        c_vals = range(c_min, c_max + 1)
    elif side == "left":
        c_vals = range(0, bc0 - w - CUE_BOARD_DIST + 1)
        r_min = max(0, br0 - h + 1)
        r_max = min(CANVAS_SIZE - h, br1)
        r_vals = range(r_min, r_max + 1)
    elif side == "right":
        c_vals = range(bc1 + CUE_BOARD_DIST, CANVAS_SIZE - w + 1)
        r_min = max(0, br0 - h + 1)
        r_max = min(CANVAS_SIZE - h, br1)
        r_vals = range(r_min, r_max + 1)
    else:
        raise ValueError(f"unknown side {side}")

    for r in r_vals:
        for c in c_vals:
            if np.any((canvas[r : r + h, c : c + w] != 0) & mask):
                continue
            bb = (r, c, r + h - 1, c + w - 1)
            if _bbox_chebyshev(bb, board_bbox) < CUE_BOARD_DIST:
                continue
            if any(_bbox_chebyshev(bb, prev) < CUE_INTER_DIST for prev in cue_bboxes):
                continue
            try:
                if _cue_side(bb, board_bbox) != side:
                    continue
            except AssertionError:
                continue
            candidates.append((r, c, bb))

    if not candidates:
        return False

    r, c, bb = random.choice(candidates)
    paste(canvas, patch, r, c, transparent=0)
    cue_bboxes.append(bb)
    return True


def _cue_area() -> int:
    """Choose a cue size that carries no rule information."""
    return random.randint(1, 5)


def _side_for_transform(transform: str, available_sides: tuple[str, str]) -> str:
    """Pick the one legal side in this corner layout that realizes the requested transform."""
    matches = [side for side in available_sides if SIDE_TO_TRANSFORM[side] == transform]
    assert len(matches) == 1, "corner layout should expose exactly one side for each transform family"
    return matches[0]


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

    (r0, c0), available_sides = _corner_anchor_and_sides(n)
    board_bbox = (r0, c0, r0 + n - 1, c0 + n - 1)

    in_board = np.zeros((n, n), dtype=int)
    for col, transform in zip(colors, transforms):
        inv = apply_transform(out_board == col, transform)
        if np.any((in_board != 0) & inv):
            return None
        in_board[inv] = int(col)

    canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=int)
    paste(canvas, in_board, r0, c0, transparent=0)

    cue_bboxes: list[tuple[int, int, int, int]] = []
    cue_specs = []
    for col, transform in zip(colors, transforms):
        if transform is None:
            continue
        cue_specs.append((col, _cue_area(), _side_for_transform(transform, available_sides)))
    random.shuffle(cue_specs)

    for col, area, side in cue_specs:
        if not _place_cue_on_side(canvas, _cue_patch(col, area), board_bbox, cue_bboxes, side):
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
        if _cue_side(bb, board_bbox) not in available_sides:
            return None

    if not np.array_equal(main(canvas), out_board):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 6ffbe589 combo input")
