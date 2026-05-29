# Combo variant of 9385bd28 (index 11)
# combo_spec: D1/4+D3/0+D11/1
# Name: 镜像对称匹配+完整矩形+多干扰

import random
from collections import Counter

import numpy as np

from arc_utils import get_bg


# Design analysis
# ===============
# concretization_and_trace:
#   - 保留原题左下角 2 列竖直图例；每行仍是 (marker_color, fill_color)，并按 bottom->top 顺序处理。
#   - 真正参与求解的对象不再是“同一 bbox 对角上的两枚 L 角标”，而是“关于整张网格中心线镜像的两枚 L 角标”。
#     具体只保留两类易观察的几何家族：
#       1) 竖直中心线版本：一个 2 行高的横向条带，左右端各有一个互为镜像的 L 角标。
#       2) 水平中心线版本：一个 2 列宽的纵向条带，上下端各有一个互为镜像的 L 角标。
#   - 小例子：若某色在 2x7 bbox 的左端是 TL 角、右端是 TR 角，并且 cmin + cmax == W-1，
#     原题会尝试判断“哪条对角线是 marker 对角线”；融合后不再做这一步，而是直接把这整个 2x7 条带视为
#     围绕网格中心竖轴的镜像配对对象，并把完整 2x7 矩形全部涂成图例映射色，连 6 个 marker 像素也吞没。
#   - 同时再加入 2-3 个非图例颜色的传统对角 L-pair 干扰项。它们在输入里看起来也像合法角标对，
#     但既不在图例里，也不应出现在输出变换链中，因此输出保持原样。
# complexity_calibration:
#   - 为了让“镜像匹配”能在 2-4 个样例中被人眼稳定归纳，所有真实对象在同一张图里共享同一条全局中心轴，
#     只在 vertical / horizontal 两档之间切换；不再混入 block/noop，避免主规则被稀释。
#   - 完整矩形填充本身会吞没 marker；若 fill 恰好等于 marker，则 D3 变化几乎不可见。因此生成器强制
#     fill != marker，并要求新输出与“保留当前 marker 的镜像版弱规则”至少相差 6*n_entries 个像素。
#   - 干扰项固定为 2-3 个，且颜色不出现在图例两列中，既能增加筛选难度，又不会把颜色线索搅乱。
# implicit_assumptions_review:
#   - 原 `main()` 的对角锚点评分 (`score_tlbr` / `score_trbl`) 依赖“真实对象一定是对角 L-pair”；
#     镜像条带在 2xN 或 Nx2 bbox 中不满足这一前提，因此整段判定必须移除。
#   - 原 `protect |= (g == marker)` 依赖“当前 marker 需要保留可见”；D3/0 明确要求 marker 被矩形吞没，
#     所以当前颜色绝不能再加入 protect。
#   - 原题允许 2x2 block / noop；融合描述只强调镜像标记对，因此若继续保留这些分支，会让 solver
#     对“一个图例色到底是镜像条带还是别的东西”产生额外假设，也会削弱视觉可推断性，故生成器全部移除。
#   - D11/1 引入多个假 L-pair 后，一个新的隐式风险是“真实矩形填充可能覆盖假对象”；这会让假对象看起来
#     像被处理过。生成器因此额外保证所有干扰 bbox 都与真实填充矩形完全不相交。
# placement_legality:
#   - 所有对象都通过 `occupied` 防碰撞，保证输入像素不重叠、不越界。
#   - 真实镜像条带还通过 `reserved_real_rects` 保证未来填充矩形彼此不重叠，并留少量间隔，避免输出糊成一片。
#   - 干扰对通过 `reserved_all_rects` 避开所有真实矩形与已放置干扰矩形，因此“未处理”在输出中肉眼可见。
# visual_observability_and_cleanliness:
#   - 一张图里多个真实对象共享同一中心轴，会形成平行条带，人眼很容易先看到中心镜像，再看到输出把条带补满。
#   - 假对象使用不在图例里的独立颜色，且保持输入形状不变，因此“先看图例过滤颜色，再看几何配对”的两层规则
#     都有直接证据。
#   - 颜色推断只依赖可见图例，不依赖颜色编号大小、黑色特殊性等不可视因素。
# new_solver_assumptions_backed_by_generator:
#   - 每个图例 marker 色恰好对应一个 6 像素镜像 L-pair，并且全部共用同一条网格中心轴。
#   - 每个非图例干扰色恰好对应一个传统对角 L-pair，且不与任何真实填充矩形相交。
#   - 真实对象之间不共享填充矩形，因此 bottom->top 顺序只承担“保护未来 marker”的职责，不承担消歧责任。


def _find_legend(g: np.ndarray, bg: int) -> tuple[int, list[tuple[int, int, int]]]:
    """
    Legend is a vertical stack of pairs at bottom-left.
    Each legend row: (marker_color at col c, fill_color at col c+1).
    Returns (legend_col, entries) where entries are ordered bottom->top as (row, marker, fill).
    """
    h, w = g.shape
    start_r = None
    start_c = None
    for r in range(h - 1, -1, -1):
        nz = np.flatnonzero(g[r, :] != bg)
        if nz.size:
            start_r = r
            start_c = int(nz[0])
            break
    assert start_r is not None and start_c is not None and start_c < w - 1, (
        "legend must exist at bottom-left with at least 2 columns"
    )

    entries: list[tuple[int, int, int]] = []
    c = start_c
    for r in range(start_r, -1, -1):
        nz = np.flatnonzero(g[r, :] != bg)
        if nz.size == 0 or int(nz[0]) != c or int(g[r, c]) == bg:
            break
        marker = int(g[r, c])
        fill = int(g[r, c + 1]) if c + 1 < w else bg
        entries.append((r, marker, fill))
    return c, entries


_L_OFFSETS = {
    "TL": [(0, 0), (0, 1), (1, 0)],
    "TR": [(0, 0), (0, -1), (1, 0)],
    "BL": [(0, 0), (0, 1), (-1, 0)],
    "BR": [(0, 0), (0, -1), (-1, 0)],
}


def _l_anchor_present(
    pts: set[tuple[int, int]],
    r: int,
    c: int,
    arm1: tuple[int, int],
    arm2: tuple[int, int],
) -> bool:
    """Check whether an L-anchor of a given orientation is present at (r,c)."""
    dr1, dc1 = arm1
    dr2, dc2 = arm2
    return (r, c) in pts and (r + dr1, c + dc1) in pts and (r + dr2, c + dc2) in pts


def _classify_mirror_pair(
    marker_pts: set[tuple[int, int]],
) -> tuple[str, tuple[int, int, int, int], int] | None:
    """
    Recognize a 6-pixel mirrored strip.

    Returns:
      (axis_kind, bbox, center_sum)
      axis_kind = 'V' for a 2-row strip mirrored about the grid's vertical centerline
      axis_kind = 'H' for a 2-col strip mirrored about the grid's horizontal centerline
    """
    if len(marker_pts) != 6:
        return None

    rs = [r for r, _ in marker_pts]
    cs = [c for _, c in marker_pts]
    rmin, rmax = min(rs), max(rs)
    cmin, cmax = min(cs), max(cs)

    if rmax - rmin == 1 and cmax - cmin >= 4:
        top = sorted(c for r, c in marker_pts if r == rmin)
        bot = sorted(c for r, c in marker_pts if r == rmax)
        if top == [cmin, cmin + 1, cmax - 1, cmax] and bot == [cmin, cmax]:
            return "V", (rmin, cmin, rmax, cmax), cmin + cmax
        if bot == [cmin, cmin + 1, cmax - 1, cmax] and top == [cmin, cmax]:
            return "V", (rmin, cmin, rmax, cmax), cmin + cmax

    if cmax - cmin == 1 and rmax - rmin >= 4:
        left = sorted(r for r, c in marker_pts if c == cmin)
        right = sorted(r for r, c in marker_pts if c == cmax)
        if left == [rmin, rmin + 1, rmax - 1, rmax] and right == [rmin, rmax]:
            return "H", (rmin, cmin, rmax, cmax), rmin + rmax
        if right == [rmin, rmin + 1, rmax - 1, rmax] and left == [rmin, rmax]:
            return "H", (rmin, cmin, rmax, cmax), rmin + rmax

    return None


def _classify_diagonal_l_pair(
    marker_pts: set[tuple[int, int]],
) -> tuple[str, tuple[int, int, int, int]] | None:
    """Recognize one classic diagonal L-pair; used only for distractor auditing."""
    if len(marker_pts) != 6:
        return None

    rs = [r for r, _ in marker_pts]
    cs = [c for _, c in marker_pts]
    rmin, rmax = min(rs), max(rs)
    cmin, cmax = min(cs), max(cs)
    if rmin == rmax or cmin == cmax:
        return None

    tl = (rmin, cmin)
    tr = (rmin, cmax)
    bl = (rmax, cmin)
    br = (rmax, cmax)

    tl_anchor = _l_anchor_present(marker_pts, tl[0], tl[1], (0, 1), (1, 0))
    tr_anchor = _l_anchor_present(marker_pts, tr[0], tr[1], (0, -1), (1, 0))
    bl_anchor = _l_anchor_present(marker_pts, bl[0], bl[1], (0, 1), (-1, 0))
    br_anchor = _l_anchor_present(marker_pts, br[0], br[1], (0, -1), (-1, 0))

    score_tlbr = int(tl_anchor) + int(br_anchor)
    score_trbl = int(tr_anchor) + int(bl_anchor)
    if score_tlbr == score_trbl or max(score_tlbr, score_trbl) == 0:
        return None
    diag = "TLBR" if score_tlbr > score_trbl else "TRBL"
    return diag, (rmin, cmin, rmax, cmax)


def _solve(grid: np.ndarray, swallow_current_markers: bool) -> np.ndarray:
    """Shared solver core; `swallow_current_markers=False` is the weaker mirror-only baseline."""
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    bg = get_bg(g)
    legend_c, entries = _find_legend(g, bg)
    assert entries, "legend must contain at least one (marker, fill) entry"

    legend_mask = np.zeros_like(g, dtype=bool)
    for r, _, _ in entries:
        legend_mask[r, legend_c] = True
        legend_mask[r, legend_c + 1] = True

    marker_order = [m for _, m, _ in entries]
    pair_info: dict[int, tuple[str, tuple[int, int, int, int]]] = {}
    axis_kinds: list[str] = []
    axis_sums: list[int] = []

    for marker in marker_order:
        marker_pts = set(map(tuple, np.argwhere((g == marker) & (~legend_mask))))
        assert marker_pts, f"legend marker color {marker} must appear outside the legend"
        info = _classify_mirror_pair(marker_pts)
        assert info is not None, f"marker color {marker} is not a valid mirrored L-pair"
        axis_kind, bbox, center_sum = info
        pair_info[marker] = (axis_kind, bbox)
        axis_kinds.append(axis_kind)
        axis_sums.append(center_sum)

    assert len(set(axis_kinds)) == 1, "all active marker pairs must share one symmetry axis"
    axis_kind = axis_kinds[0]
    expected_center_sum = w - 1 if axis_kind == "V" else h - 1
    assert all(s == expected_center_sum for s in axis_sums), (
        "mirrored marker pairs must reflect across the grid centerline"
    )

    out = g.copy()
    for i, (_, marker, fill) in enumerate(entries):
        _, (rmin, cmin, rmax, cmax) = pair_info[marker]
        future_markers = set(marker_order[i + 1 :])
        protect = legend_mask.copy()
        for fm in future_markers:
            protect |= (g == fm) & (~legend_mask)
        if not swallow_current_markers:
            protect |= (g == marker) & (~legend_mask)

        region = np.zeros_like(g, dtype=bool)
        region[rmin : rmax + 1, cmin : cmax + 1] = True
        out[region & (~protect)] = int(fill)

    return out


def _swallow_delta_count(grid: np.ndarray) -> int:
    """Cell delta caused specifically by D3/0 on top of the mirror-matching rule."""
    new_out = _solve(grid, swallow_current_markers=True)
    old_out = _solve(grid, swallow_current_markers=False)
    return int(np.count_nonzero(new_out != old_out))


def _fusion_signature(grid: np.ndarray) -> dict[str, int | str]:
    """
    Quantify whether the generated input really exercises the combo.

    Metrics:
      mirror_entries: legend colors recognized as mirrored center-axis pairs
      distractor_pairs: non-legend colors recognized as classic diagonal L-pairs
      swallow_delta: output delta vs the weaker mirror-but-keep-marker baseline
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    legend_c, entries = _find_legend(g, bg)

    legend_mask = np.zeros_like(g, dtype=bool)
    for r, _, _ in entries:
        legend_mask[r, legend_c] = True
        legend_mask[r, legend_c + 1] = True

    marker_colors = {m for _, m, _ in entries}
    mirror_entries = 0
    axis_kind = "?"
    for marker in marker_colors:
        pts = set(map(tuple, np.argwhere((g == marker) & (~legend_mask))))
        info = _classify_mirror_pair(pts)
        if info is not None:
            mirror_entries += 1
            axis_kind = info[0]

    distractor_pairs = 0
    for color in range(10):
        if color == bg or color in marker_colors:
            continue
        pts = set(map(tuple, np.argwhere(g == color)))
        if _classify_diagonal_l_pair(pts) is not None:
            distractor_pairs += 1

    return {
        "H": h,
        "W": w,
        "axis_kind": axis_kind,
        "mirror_entries": mirror_entries,
        "distractor_pairs": distractor_pairs,
        "swallow_delta": _swallow_delta_count(g),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill full rectangles between center-mirrored marker pairs using a color legend.

    Input structure:
    - The grid has one dominant background color and a 2-column legend stacked near the lower left.
    - Each legend row maps a marker color to a fill color, and legend rows are read bottom to top.
    - Away from the legend, every true marker color appears exactly once as two 3-pixel L corners that
      are mirror images about the grid centerline; extra non-legend colors may form 2-3 fake diagonal L-pairs.

    Rule:
    - Read only the marker colors listed in the legend and ignore every other color.
    - For each listed marker, verify that its six non-legend pixels form one mirrored thin strip whose
      symmetry axis is the grid's vertical or horizontal midpoint, shared by all true objects in that grid.
    - Fill the entire axis-aligned rectangle spanned by that mirrored pair with the legend's fill color,
      including the current marker pixels themselves, while still protecting legend cells and markers that
      belong to later legend rows.

    Key insight: the relevant pair is defined by global mirror symmetry, not by opposite rectangle corners.
    Once that mirrored pair is found, the output is the whole strip rectangle, and non-legend L-pairs are
    decoys that stay untouched because only legend-listed colors are actionable.
    """
    return _solve(grid, swallow_current_markers=True)


# --------------- generator (mirror + full rectangle + multi distractors) ---------------

# --- canvas & grid ---
GRID_H_RANGE = (10, 19)
GRID_W_RANGE = (10, 19)

# --- legend ---
LEGEND_COL_RANGE = (0, 2)
LEGEND_ENTRY_RANGE = (2, 4)

# --- mirrored strip geometry ---
PAIR_SPAN_MIN = 5
PAIR_SPAN_MAX = 12
AXIS_VERTICAL_PROB = 0.5
ROW_GAP = 1
COL_GAP = 1

# --- classic distractor geometry ---
BBOX_H_MAX = 19
BBOX_W_MAX = 12
BBOX_SHAPE_WEIGHTS = [0.75, 0.15, 0.10]
TLBR_PROB = 0.8
DISTRACTOR_COUNT_RANGE = (2, 3)
DISTRACTOR_PAD = 0

# --- visibility constraints ---
MIN_SWALLOW_DELTA_PER_ENTRY = 6

# --- retry budgets ---
PAIR_RETRIES = 220
DISTRACTOR_RETRIES = 120
GEN_RETRIES = 500


def _build_legend(
    H: int,
    W: int,
    bg: int,
    axis_kind: str,
    n_distractors: int,
) -> tuple[
    np.ndarray,
    list[tuple[int, int, int]],
    np.ndarray,
    int,
    int,
    list[int],
    list[int],
] | None:
    """Create legend rows plus reserved distractor colors."""
    max_legend_c = min(LEGEND_COL_RANGE[1], W - 2)
    legend_cs = list(range(LEGEND_COL_RANGE[0], max_legend_c + 1))
    if axis_kind == "V":
        max_cmin = (W - PAIR_SPAN_MIN) // 2
        legend_cs = [c for c in legend_cs if c + 1 <= max_cmin]
    if not legend_cs:
        return None

    legend_c = random.choice(legend_cs)
    legend_bot = random.choice([H - 1, H - 2])

    axis_capacity = (H + ROW_GAP) // (2 + ROW_GAP) if axis_kind == "V" else (W + COL_GAP) // (2 + COL_GAP)
    n_hi = min(LEGEND_ENTRY_RANGE[1], axis_capacity)
    if n_hi < LEGEND_ENTRY_RANGE[0]:
        return None
    n_entries = random.randint(LEGEND_ENTRY_RANGE[0], n_hi)
    if legend_bot - n_entries + 1 < 0:
        return None

    pool = [c for c in range(10) if c != bg]
    if len(pool) < n_entries + n_distractors:
        return None
    marker_colors = random.sample(pool, n_entries)
    remaining = [c for c in pool if c not in marker_colors]
    distractor_colors = random.sample(remaining, n_distractors)
    # Keep fill colors out of both marker and distractor palettes so the legend mapping
    # stays visually legible and filled regions do not masquerade as untouched markers.
    fill_pool = [
        c for c in range(10) if c not in distractor_colors and c not in marker_colors and c != bg
    ]
    if not fill_pool:
        return None
    if len(fill_pool) >= n_entries:
        fill_colors = random.sample(fill_pool, n_entries)
    else:
        fill_colors = [random.choice(fill_pool) for _ in range(n_entries)]

    grid = np.full((H, W), bg, dtype=int)
    legend_rows: list[tuple[int, int, int]] = []
    legend_mask = np.zeros((H, W), dtype=bool)
    for i in range(n_entries):
        r = legend_bot - i
        marker = marker_colors[i]
        fill = fill_colors[i]
        legend_rows.append((r, marker, fill))
        grid[r, legend_c] = marker
        grid[r, legend_c + 1] = fill
        legend_mask[r, legend_c] = True
        legend_mask[r, legend_c + 1] = True

    return grid, legend_rows, legend_mask, legend_c, legend_bot, marker_colors, distractor_colors


def _build_forbidden(
    H: int,
    W: int,
    legend_c: int,
    legend_bot: int,
    legend_rows: list[tuple[int, int, int]],
) -> np.ndarray:
    """Protect the legend scan corridor."""
    forbidden = np.zeros((H, W), dtype=bool)
    if legend_bot + 1 < H:
        forbidden[legend_bot + 1 :, :] = True
    forbidden[: legend_bot + 1, : legend_c + 1] = True
    for r, _, _ in legend_rows:
        forbidden[r, legend_c] = False
    return forbidden


def _rect_intersects(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _rect_too_close(
    rect: tuple[int, int, int, int],
    others: list[tuple[int, int, int, int]],
    pad_r: int,
    pad_c: int,
) -> bool:
    r0, c0, r1, c1 = rect
    expanded = (r0 - pad_r, c0 - pad_c, r1 + pad_r, c1 + pad_c)
    return any(_rect_intersects(expanded, other) for other in others)


def _vertical_pair_cells(
    r0: int,
    c0: int,
    c1: int,
    style: str,
) -> tuple[list[tuple[int, int]], tuple[int, int, int, int]]:
    """2-row strip mirrored about the grid's vertical centerline."""
    if style == "top":
        anchors = [("TL", r0, c0), ("TR", r0, c1)]
    else:
        anchors = [("BL", r0 + 1, c0), ("BR", r0 + 1, c1)]
    cells: list[tuple[int, int]] = []
    for name, r, c in anchors:
        for dr, dc in _L_OFFSETS[name]:
            cells.append((r + dr, c + dc))
    return cells, (r0, c0, r0 + 1, c1)


def _horizontal_pair_cells(
    r0: int,
    r1: int,
    c0: int,
    style: str,
) -> tuple[list[tuple[int, int]], tuple[int, int, int, int]]:
    """2-col strip mirrored about the grid's horizontal centerline."""
    if style == "left":
        anchors = [("TL", r0, c0), ("BL", r1, c0)]
    else:
        anchors = [("TR", r0, c0 + 1), ("BR", r1, c0 + 1)]
    cells: list[tuple[int, int]] = []
    for name, r, c in anchors:
        for dr, dc in _L_OFFSETS[name]:
            cells.append((r + dr, c + dc))
    return cells, (r0, c0, r1, c0 + 1)


def _vertical_cmins(W: int, legend_c: int) -> list[int]:
    out: list[int] = []
    max_cmin = (W - PAIR_SPAN_MIN) // 2
    for cmin in range(legend_c + 1, max_cmin + 1):
        span = W - 2 * cmin
        if PAIR_SPAN_MIN <= span <= PAIR_SPAN_MAX:
            out.append(cmin)
    return out


def _horizontal_rmins(H: int) -> list[int]:
    out: list[int] = []
    max_rmin = (H - PAIR_SPAN_MIN) // 2
    for rmin in range(0, max_rmin + 1):
        span = H - 2 * rmin
        if PAIR_SPAN_MIN <= span <= PAIR_SPAN_MAX:
            out.append(rmin)
    return out


def _try_place_mirror_pair(
    H: int,
    W: int,
    axis_kind: str,
    legend_c: int,
    color: int,
    retries: int,
    can_place,
    do_place,
    reserved_real_rects: list[tuple[int, int, int, int]],
    reserved_all_rects: list[tuple[int, int, int, int]],
) -> bool:
    """Place one true mirrored pair and reserve its future fill rectangle."""
    vertical_cmins = _vertical_cmins(W, legend_c)
    horizontal_rmins = _horizontal_rmins(H)
    for _ in range(retries):
        if axis_kind == "V":
            if not vertical_cmins:
                return False
            r0 = random.randint(0, H - 2)
            c0 = random.choice(vertical_cmins)
            c1 = W - 1 - c0
            cells, rect = _vertical_pair_cells(r0, c0, c1, random.choice(["top", "bottom"]))
            if _rect_too_close(rect, reserved_real_rects, ROW_GAP, 0):
                continue
        else:
            if not horizontal_rmins:
                return False
            r0 = random.choice(horizontal_rmins)
            r1 = H - 1 - r0
            c0 = random.randint(legend_c + 1, W - 2)
            cells, rect = _horizontal_pair_cells(r0, r1, c0, random.choice(["left", "right"]))
            if _rect_too_close(rect, reserved_real_rects, 0, COL_GAP):
                continue

        if can_place(cells):
            do_place(cells, color)
            reserved_real_rects.append(rect)
            reserved_all_rects.append(rect)
            return True
    return False


def _l_pair_cells(
    rmin: int,
    cmin: int,
    rmax: int,
    cmax: int,
    diag: str,
) -> list[tuple[int, int]]:
    """Return the 6 cells for a classic diagonal L-pair. diag in {'TLBR','TRBL'}."""
    if diag == "TLBR":
        anchors = [("TL", rmin, cmin), ("BR", rmax, cmax)]
    else:
        anchors = [("TR", rmin, cmax), ("BL", rmax, cmin)]
    cells: list[tuple[int, int]] = []
    for name, r, c in anchors:
        for dr, dc in _L_OFFSETS[name]:
            cells.append((r + dr, c + dc))
    return cells


def _l_pair_valid_bbox(h: int, w: int) -> bool:
    """True if (h, w) avoids ambiguous diagonal anchor ties."""
    if h < 2 or w < 2:
        return False
    if h == 2 and w <= 3:
        return False
    if w == 2 and h <= 3:
        return False
    return True


def _pick_bbox_size(H: int, W: int) -> tuple[int, int] | None:
    """Pick a classic distractor bbox size that fits in the grid."""
    shape_choice = random.choices(
        ["both_ge3", "h2_w_ge4", "h_ge4_w2"],
        weights=BBOX_SHAPE_WEIGHTS,
    )[0]
    if shape_choice == "both_ge3":
        bh = random.randint(3, min(BBOX_H_MAX, H))
        bw = random.randint(3, min(BBOX_W_MAX, W))
    elif shape_choice == "h2_w_ge4":
        bh = 2
        w_hi = min(BBOX_W_MAX, W)
        if w_hi < 4:
            return None
        bw = random.randint(4, w_hi)
    else:
        h_hi = min(BBOX_H_MAX, H)
        if h_hi < 4:
            return None
        bh = random.randint(4, h_hi)
        bw = 2
    if bh > H or bw > W or not _l_pair_valid_bbox(bh, bw):
        return None
    return bh, bw


def _try_place_distractor_pair(
    H: int,
    W: int,
    color: int,
    retries: int,
    can_place,
    do_place,
    reserved_all_rects: list[tuple[int, int, int, int]],
) -> bool:
    """Place one non-legend classic diagonal L-pair away from all real rectangles."""
    for _ in range(retries):
        size = _pick_bbox_size(H, W)
        if size is None:
            continue
        bh, bw = size
        rmin = random.randint(0, H - bh)
        cmin = random.randint(0, W - bw)
        rect = (rmin, cmin, rmin + bh - 1, cmin + bw - 1)
        if _rect_too_close(rect, reserved_all_rects, DISTRACTOR_PAD, DISTRACTOR_PAD):
            continue
        diag = "TLBR" if random.random() < TLBR_PROB else "TRBL"
        cells = _l_pair_cells(rmin, cmin, rect[2], rect[3], diag)
        if can_place(cells):
            do_place(cells, color)
            reserved_all_rects.append(rect)
            return True
    return False


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)
    bg = random.randint(0, 9)
    axis_kind = "V" if random.random() < AXIS_VERTICAL_PROB else "H"
    n_distractors = random.randint(*DISTRACTOR_COUNT_RANGE)

    legend_result = _build_legend(H, W, bg, axis_kind, n_distractors)
    if legend_result is None:
        return None
    (
        grid,
        legend_rows,
        legend_mask,
        legend_c,
        legend_bot,
        marker_colors,
        distractor_colors,
    ) = legend_result

    forbidden = _build_forbidden(H, W, legend_c, legend_bot, legend_rows)
    occupied = legend_mask.copy()
    reserved_real_rects: list[tuple[int, int, int, int]] = []
    reserved_all_rects: list[tuple[int, int, int, int]] = []

    def can_place(cells: list[tuple[int, int]]) -> bool:
        for r, c in cells:
            if not (0 <= r < H and 0 <= c < W):
                return False
            if occupied[r, c] or forbidden[r, c]:
                return False
        return True

    def do_place(cells: list[tuple[int, int]], color: int) -> None:
        for r, c in cells:
            grid[r, c] = color
            occupied[r, c] = True

    for _, marker, _ in legend_rows:
        placed = _try_place_mirror_pair(
            H,
            W,
            axis_kind,
            legend_c,
            marker,
            PAIR_RETRIES,
            can_place,
            do_place,
            reserved_real_rects,
            reserved_all_rects,
        )
        if not placed:
            return None

    random.shuffle(distractor_colors)
    for color in distractor_colors:
        if not _try_place_distractor_pair(
            H,
            W,
            color,
            DISTRACTOR_RETRIES,
            can_place,
            do_place,
            reserved_all_rects,
        ):
            return None

    counts = Counter(grid.flatten().tolist())
    if counts.most_common(1)[0][0] != bg:
        return None

    try:
        _ = main(grid)
        sig = _fusion_signature(grid)
        if sig["mirror_entries"] != len(marker_colors):
            return None
        if sig["distractor_pairs"] != n_distractors:
            return None
        if sig["swallow_delta"] < MIN_SWALLOW_DELTA_PER_ENTRY * len(marker_colors):
            return None
    except Exception:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError(
        "generate_input: failed to generate a valid input after %d attempts" % GEN_RETRIES
    )
