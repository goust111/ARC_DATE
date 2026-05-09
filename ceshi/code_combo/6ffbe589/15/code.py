# Combo variant of 6ffbe589 (index 15)
# combo_spec: D8/0+D9/0+D18/0
# Name: 无提示180旋转+2层+多对一提示

from __future__ import annotations

import random

import numpy as np

from arc_utils import colors_present, find_components, get_bg, paste


# Design analysis for the combo variant
# spec:
#   fused_rule:
#     - 棋盘从原题的 3 层缩减为 2 层：外框层 + 内嵌内层。
#     - 每个颜色层都独立决定旋转量，但旋转来源分成两类：
#       - 若该颜色在板外存在提示，则把所有同色提示精灵的面积求和，rot_k = (-sum_area) % 4。
#       - 若该颜色完全没有板外提示，则该缺失本身编码 rot_k = 2。
#   traced_example:
#     - 设输出板只有红色外框和蓝色内嵌层两层，且两层都不是 90/180/270 旋转对称。
#     - 生成器若指定红层“无提示”，则输入中的红层必须先被反向旋转 180 度；solver 因看不到红色 cue，会把红层再转 180 度复原。
#     - 若蓝层有两个蓝色 cue，面积分别为 1 和 2，则 solver 不看单个 cue，而看总面积 3，于是蓝层执行 rot_k = (-3) % 4 = 1。
#     - 原题差异有两处：原题遇到“无提示”会保持不动，而且原题默认每色至多一个 cue，不需要聚合。
#   complexity_calibration:
#     - 2 层结构会降低对象数量，所以生成器反向强化线索：每个样本固定一层缺失提示、另一层固定为 2 个同色提示。
#     - 显式提示层只使用 rot_k in {1, 3}，避免和“无提示=180 度”混淆，也避免显式 cue 却不转动的弱样本。
#     - 多 cue 方案只用少数稳定拆分（如 1+2 或 2+3），并要求总面积 mod 4 与任一单个 cue 的面积 mod 4 都不同，确保求和是必要的。
#   main_assumption_audit:
#     - `crop_components_near_largest(...)` 仍依赖方板是唯一最大前景连通块；生成器继续显式验证。
#     - `rotate_color_layer(...)` 仍要求裁剪结果是方形；2 层同心方板保持这个前提。
#     - 原题 `len(outside_comps) <= 1` 的隐含假设已失效，因此 solver 改为汇总所有同色外部组件。
#     - “无提示颜色可能对称”这个旧前提也失效，因为缺失提示现在会触发 180 度旋转；因此两层都强制做成旋转非对称。
#   placement_legality:
#     - 板仍放入 20x20 画布并作为最大连通块。
#     - cue 与板 bbox 的切比雪夫距离保持 >= 4，避免被 margin=3 的裁剪吸进去。
#     - 多个 cue 之间的切比雪夫距离保持 >= 2，这足以让 8 邻接下它们仍是分离组件。
#     - 输入板由两层各自逆旋转后合成；若发生重叠或不唯一最大，整次尝试作废。
#   observability_and_cleanliness:
#     - 解题者能直接看到：一种颜色没有任何提示却稳定翻转 180 度，另一种颜色在板外重复出现两块同色小精灵，需要把像素数相加。
#     - 颜色匹配只依赖可见的同色关系，不使用颜色编号排序等不可见信息。
#     - 只有 2 层板和 2 个 cue，场景比原题更干净；内层紧贴外层内侧，保证整体仍是一个清晰的大组件。
#   inferability:
#     - 训练样例中始终同时出现“缺失提示”与“多 cue 聚合”两种现象，且两层都做成非对称形状。
#     - 因此从 2-4 个样例里，人类可以比较输入/输出后归纳：缺 cue 的同色层总会半转；有两个同色 cue 的层则按总面积编码旋转。


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


def rotate_color_layer(grid: np.ndarray, color: int, rot_k: int, *, bg: int | None = None) -> np.ndarray:
    """
    Local helper (only used by this solver): rotate a single color's mask inside a fixed-size grid.
    """
    if bg is None:
        bg = get_bg(grid)
    bg = int(bg)

    k = int(rot_k) % 4
    if k == 0:
        return grid

    col = int(color)
    mask = (grid == col).astype(int)
    rmask = np.rot90(mask, k)
    assert rmask.shape == mask.shape, "expected a square crop for layer rotation"

    out = grid.copy()
    out[out == col] = bg
    out[rmask == 1] = col
    return out


def _solve_with_rules(
    grid: np.ndarray,
    *,
    missing_rot_k: int,
    cue_mode: str,
) -> np.ndarray:
    """Shared solver core used both by main() and by generator-side anti-degeneration checks."""
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    crop, (r0, c0, r1, c1), bg = crop_components_near_largest(g, bg=bg, margin=3, connectivity=8)
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
        areas = [int(comp["area"]) for comp in outside_comps]
        if not areas:
            rot_k = int(missing_rot_k) % 4
        elif cue_mode == "sum":
            rot_k = (-sum(areas)) % 4
        elif cue_mode == "largest":
            rot_k = (-max(areas)) % 4
        else:
            raise ValueError(f"unsupported cue_mode: {cue_mode}")

        if rot_k:
            out = rotate_color_layer(out, int(col), int(rot_k), bg=int(bg))

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the 2-layer board and restore each color layer using outside same-color cues.

    Input structure:
    - Background is the most common color.
    - The largest foreground component is a square board with exactly 2 concentric color layers: an outer frame and an inset inner layer just inside it.
    - One board color may have no outside cue at all, while the other may have several disconnected same-color cue blobs.

    Rule:
    - Crop away the distant cue blobs and keep only the board.
    - For each color in the cropped board, inspect the outside region.
    - If same-color cue blobs exist, add all of their pixel counts and rotate that layer by `(-total_area) % 4` quarter-turns.
    - If a board color has no cue anywhere outside, rotate that layer by the default half-turn (`rot_k = 2`).

    Key insight:
    - Rotation is encoded in two complementary ways: repeated same-color cues must be aggregated, and the complete absence of a cue is itself a 180-degree instruction.
    """
    return _solve_with_rules(grid, missing_rot_k=2, cue_mode="sum")


# --------------- generator ---------------

CANVAS_SIZE = 20
BOARD_N_MIN, BOARD_N_MAX = 10, 13
NUM_COLORS = 2

OUTER_DENSITY_LO = 0.28
OUTER_DENSITY_HI = 0.42
INNER_DENSITY_LO = 0.50
INNER_DENSITY_HI = 0.80
LAYER_BAND_WIDTH = 3
ASYM_MARKER = (1, 2)
INNER_INSET = 3

CUE_GRID = 3
CUE_BOARD_DIST = 4
CUE_INTER_DIST = 2

MISSING_ROT_K = 2
EXPLICIT_ROT_CHOICES = (1, 3)
TOTAL_MOD_TO_SPLITS = {
    1: ((2, 3),),
    3: ((1, 2), (2, 5)),
}

GEN_RETRIES = 400


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


def _perimeter_ring(n: int) -> np.ndarray:
    """Full 1-cell border ring of an n*n grid."""
    rr, cc = np.indices((n, n))
    return (rr == 0) | (rr == n - 1) | (cc == 0) | (cc == n - 1)


def _fill_to_count(mask: np.ndarray, candidates: list[tuple[int, int]], target: int) -> bool:
    """Randomly set cells in mask until it reaches target count. Returns success."""
    random.shuffle(candidates)
    for r, c in candidates:
        if int(mask.sum()) >= target:
            return True
        mask[r, c] = True
    return int(mask.sum()) >= target


def _make_asym_band_mask(
    n: int,
    *,
    density_lo: float,
    density_hi: float,
    band_width: int,
    marker: tuple[int, int],
) -> np.ndarray | None:
    """Build a connected ring-like asymmetric mask inside an n*n workspace."""
    mask = _perimeter_ring(n)
    marker_pt = (int(marker[0]), int(marker[1]))
    orbit = _rot_orbit(marker_pt[0], marker_pt[1], n)
    if not (0 <= marker_pt[0] < n and 0 <= marker_pt[1] < n):
        return None
    if mask[marker_pt]:
        return None
    mask[marker_pt] = True

    lo = max(int(np.ceil(density_lo * n * n)), int(mask.sum()))
    hi = int(np.floor(density_hi * n * n))
    if lo > hi:
        return None
    target = random.randint(lo, hi)

    blocked = orbit - {marker_pt}
    candidates = [(r, c) for r, c in _band_cells(n, band_width) if (r, c) not in blocked and not mask[r, c]]
    if not _fill_to_count(mask, candidates, target):
        return None
    if not _is_rot_asym(mask):
        return None
    return mask


def _make_outer_mask(n: int) -> np.ndarray | None:
    """Build the outer frame layer spanning the full n*n grid."""
    return _make_asym_band_mask(
        n,
        density_lo=OUTER_DENSITY_LO,
        density_hi=OUTER_DENSITY_HI,
        band_width=LAYER_BAND_WIDTH,
        marker=ASYM_MARKER,
    )


def _make_inner_mask(n: int) -> np.ndarray | None:
    """Build an inset inner ring-like layer that stays connected to the outer layer."""
    m = n - 2 * INNER_INSET
    if m < 3:
        return None
    local = _make_asym_band_mask(
        m,
        density_lo=INNER_DENSITY_LO,
        density_hi=INNER_DENSITY_HI,
        band_width=2,
        marker=(1, 2),
    )
    if local is None:
        return None

    mask = np.zeros((n, n), dtype=bool)
    mask[INNER_INSET : INNER_INSET + m, INNER_INSET : INNER_INSET + m] = local
    if not _is_rot_asym(mask):
        return None
    return mask


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
        arr[r, c] = color
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


def _place_cue(
    canvas: np.ndarray,
    patch: np.ndarray,
    board_bbox: tuple[int, int, int, int],
    cue_bboxes: list[tuple[int, int, int, int]],
) -> bool:
    """Place a cue patch on the canvas, respecting distance constraints."""
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


def _cue_split_for_rot(rot_k: int) -> tuple[int, ...]:
    """Pick a 2-cue split whose total area mod 4 encodes the requested explicit rotation."""
    total_mod = (-int(rot_k)) % 4
    options = [areas for areas in TOTAL_MOD_TO_SPLITS[int(total_mod)] if len(areas) == 2]
    choice = tuple(int(area) for area in random.choice(options))
    assert sum(choice) % 4 == total_mod
    assert all((int(area) % 4) != total_mod for area in choice)
    return choice


def _validate_cues(
    canvas: np.ndarray,
    board_bbox: tuple[int, int, int, int],
    colors: list[int],
    missing_idx: int,
    explicit_areas: tuple[int, ...],
) -> bool:
    """Check that the missing layer truly has no cue and the explicit layer keeps two separated cues."""
    outside_grid = canvas.copy()
    r0, c0, r1, c1 = board_bbox
    outside_grid[r0 : r1 + 1, c0 : c1 + 1] = 0

    for idx, color in enumerate(colors):
        comps = find_components(
            outside_grid,
            bg=0,
            connectivity=8,
            monochromatic=True,
            colors=[int(color)],
            return_format="dict",
        )
        areas = sorted(int(comp["area"]) for comp in comps)
        if idx == missing_idx:
            if areas:
                return False
        else:
            if areas != sorted(int(a) for a in explicit_areas):
                return False
            total_mod = sum(areas) % 4
            if any((area % 4) == total_mod for area in areas):
                return False
    return True


def _fusion_signature(grid: np.ndarray) -> tuple[int, int]:
    """
    Return simple fusion metrics:
    - number of board colors without cues
    - max count of same-color cue components among board colors
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    crop, (r0, c0, r1, c1), _ = crop_components_near_largest(g, bg=bg, margin=3, connectivity=8)
    outside_grid = g.copy()
    outside_grid[r0 : r1 + 1, c0 : c1 + 1] = bg

    missing = 0
    max_group = 0
    for color in colors_present(crop, exclude={bg}):
        comps = find_components(
            outside_grid,
            bg=bg,
            connectivity=8,
            monochromatic=True,
            colors=[int(color)],
            return_format="dict",
        )
        count = len(comps)
        if count == 0:
            missing += 1
        max_group = max(max_group, count)
    return int(missing), int(max_group)


def _gen_attempt() -> np.ndarray | None:
    """Single attempt to generate a valid input grid. Returns None on failure."""
    colors = [int(c) for c in random.sample(range(1, 10), NUM_COLORS)]
    n = random.randint(BOARD_N_MIN, BOARD_N_MAX)

    outer_mask = _make_outer_mask(n)
    if outer_mask is None:
        return None
    inner_mask = _make_inner_mask(n)
    if inner_mask is None:
        return None
    if np.any(outer_mask & inner_mask):
        return None

    out_board = np.zeros((n, n), dtype=int)
    out_board[outer_mask] = colors[0]
    out_board[inner_mask] = colors[1]

    missing_idx = random.randrange(NUM_COLORS)
    explicit_idx = 1 - missing_idx
    solver_rot_ks = [0, 0]
    solver_rot_ks[missing_idx] = MISSING_ROT_K
    solver_rot_ks[explicit_idx] = int(random.choice(EXPLICIT_ROT_CHOICES))

    in_board = np.zeros((n, n), dtype=int)
    for color, solve_k in zip(colors, solver_rot_ks):
        inv = np.rot90(out_board == int(color), (-int(solve_k)) % 4)
        if np.any((in_board != 0) & inv):
            return None
        in_board[inv] = int(color)

    r0, c0 = _board_anchor(n)
    canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=int)
    paste(canvas, in_board, r0, c0, transparent=0)
    board_bbox = (r0, c0, r0 + n - 1, c0 + n - 1)

    explicit_areas = _cue_split_for_rot(solver_rot_ks[explicit_idx])
    cue_bboxes: list[tuple[int, int, int, int]] = []
    for area in sorted(explicit_areas, reverse=True):
        if not _place_cue(canvas, _cue_patch(colors[explicit_idx], int(area)), board_bbox, cue_bboxes):
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

    if not _validate_cues(canvas, board_bbox, colors, missing_idx, explicit_areas):
        return None

    if _fusion_signature(canvas) != (1, 2):
        return None

    if not np.array_equal(main(canvas), out_board):
        return None

    legacy_missing = _solve_with_rules(canvas, missing_rot_k=0, cue_mode="sum")
    if np.array_equal(legacy_missing, out_board):
        return None

    collapsed_multi = _solve_with_rules(canvas, missing_rot_k=2, cue_mode="largest")
    if np.array_equal(collapsed_multi, out_board):
        return None

    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 6ffbe589 combo input")
