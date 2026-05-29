# Variant of 9385bd28
# Dimension: D1 — 匹配条件：标记对角线的配对方式 (match_cond)
# Change: diagonal_markers — 同色 L 形角标记位于外接矩形的对角，通过 L 形朝向判定哪条对角线是标记对角线、... → 镜像对称匹配

import random
from collections import Counter

import numpy as np

from arc_utils import get_bg


# Design analysis
# ===============
# variant_spec:
#   unchanged_core:
#     - 保留左下角 2 列竖直图例；每一行仍是 (marker_color, fill_color)，按 bottom->top 顺序处理。
#     - 保留“只根据图例中的 marker 颜色触发填充”和“后面的图例项要保护自己的标记不被前面覆盖”。
#   d1_change:
#     - 原题把两个同色 L 角标放在同一外接框的对角，并由那条“标记对角线”推出另一条“填充对角线”。
#     - 变体改成：两个同色 L 角标围绕整张网格的中心对称轴镜像摆放，不再是对角配对。
#     - 为了让镜像关系在像素层面直观可见，所有真实 marker 在同一张图里共用一条中心轴：
#       要么是竖直中心线 x=(W-1)/2，要么是水平中心线 y=(H-1)/2。
#   concrete_objects:
#     - 竖直中心线版本：每个对象是一个 2 行高的横向条带，两端各有一个镜像 L 角标。
#       可出现“上边角标”或“下边角标”两种朝向，但左右两端必定关于中心线互为镜像。
#     - 水平中心线版本：每个对象是一个 2 列宽的纵向条带，两端各有一个镜像 L 角标。
#       可出现“左边角标”或“右边角标”两种朝向，但上下两端必定关于中心线互为镜像。
#     - 输出把该条带的整块细长矩形区域（除 marker 像素与受保护像素外）涂成图例给出的 fill 色。
#   behavior_trace:
#     - 原逻辑：若某色 6 个像素落在 2x7 的 bbox 四角，solver 会检查 TL/BR 或 TR/BL 哪条对角线上有 L，
#       再去填另一条对角线对应的菱形。
#     - 新逻辑：若某色 6 个像素落在 2x7 的 bbox 两端，而且顶/底两行的像素分布呈左右镜像，
#       solver 不再问“哪条对角线是 marker”，而是先识别“这是一对围绕中心竖轴的镜像端帽”，
#       然后直接把这整个 2x7 条带内部填色。
#     - 差异：配对条件从“同 bbox 对角”改成“共用全局中心轴的镜像端点”；填充目标从“另一条对角线菱形”改成“镜像对象张成的对称条带矩形”。
#   complexity_calibration:
#     - 只保留两种全局轴向：vertical / horizontal。这样人类能从 2-4 个样例中观察到重复支撑，
#       但不会引入太多自由度。
#     - 去掉原题的 2x2 block、noop、distractor 分支，因为它们会稀释 D1 变化的可见性；
#       这里允许这类适配性调整，以保证“镜像配对”成为每个样例的主导线索。
#     - 每张图生成 2-4 个真实对象，并强制条带之间留空隙，避免视觉拥挤。
#   hidden_assumptions_review:
#     - 原 `main()` 里依赖 `rmin/rmax/cmin/cmax` 四角之一一定含有对角 L anchor，并且两条对角线得分不能打平。
#       镜像条带在 2xN 或 Nx2 bbox 中恰好会让原判定打平，因此这套打分必须整体移除。
#     - 原 `main()` 的 2x2 block 特判依赖“同色像素可能不是角标对”。变体中不再生成这种对象，否则会弱化新规则。
#     - 原 `main()` 对 future markers 的保护仍然成立，因为变体依然允许 later fills 覆盖 earlier fills，
#       只是 marker 位置改成镜像端帽。
#   placement_legality:
#     - 生成器只放置 6 像素的镜像 L 对；不允许 marker 像素与图例重叠，也不允许两个待填条带矩形相交。
#     - 竖轴版本要求 bbox 高度固定 2、宽度至少 5；横轴版本要求 bbox 宽度固定 2、高度至少 5。
#     - 所有对象都围绕网格中心线反射，因此 solver 可以先按对象几何形状分类，再核对中心轴是否一致。
#   visual_observability:
#     - 同一张图里的多个对象共享同一中心轴，且条带平行排列；肉眼很容易先看到镜像，再看到输出把两端之间补满。
#     - 颜色映射只来自左下图例，不依赖颜色 id 的大小顺序，也不使用任何不可见数值属性。
#   generator_guarantees_for_new_main:
#     - 每个非空 marker 颜色恰好对应一个 6 像素镜像 L 对。
#     - 所有非空对象的 axis type 一致，并且其中心和网格中心线一致。
#     - 条带矩形彼此不重叠，因此 `main()` 的逐条填充不会因为输入构造而产生歧义。


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


def _classify_mirror_pair(
    marker_pts: set[tuple[int, int]],
) -> tuple[str, tuple[int, int, int, int], int] | None:
    """
    Recognize one of the two mirrored strip families.

    Returns:
      (axis_kind, bbox, center_sum)
      axis_kind = 'V' for a 2-row horizontal strip mirrored about a vertical axis
      axis_kind = 'H' for a 2-col vertical strip mirrored about a horizontal axis
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the symmetric strip spanned by mirrored marker pairs using a color legend.

    Input structure:
    - The background is the most frequent color; a 2-column legend sits at bottom-left.
    - Each legend row gives (marker_color, fill_color), processed from bottom to top.
    - Outside the legend, each active marker color appears as two mirrored 3-pixel L corners.
      All active pairs in one grid share the same center symmetry axis: either the grid's
      vertical midline or its horizontal midline.

    Rule:
    - Read the legend and collect the non-legend pixels for each marker color.
    - Detect whether those pixels form a 2-row mirrored strip or a 2-column mirrored strip,
      and verify that every active marker color uses the same grid-center symmetry axis.
    - For each legend entry in order, fill the full bbox spanned by its mirrored pair with
      the legend's fill color, while preserving legend cells, the current marker cells, and
      any markers belonging to later legend entries.

    Key insight: the matching relation is no longer “opposite corners of one rectangle”.
    The two L markers are matched because they are mirror images about a shared centerline,
    and the region to paint is the thin symmetric strip stretched between them.
    """
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
        if not marker_pts:
            continue
        info = _classify_mirror_pair(marker_pts)
        assert info is not None, f"marker color {marker} is not a valid mirrored L-pair"
        axis_kind, bbox, center_sum = info
        pair_info[marker] = (axis_kind, bbox)
        axis_kinds.append(axis_kind)
        axis_sums.append(center_sum)

    assert axis_kinds, "at least one mirrored marker pair is required"
    assert len(set(axis_kinds)) == 1, "all active marker pairs must share one symmetry axis"
    axis_kind = axis_kinds[0]
    expected_center_sum = w - 1 if axis_kind == "V" else h - 1
    assert all(s == expected_center_sum for s in axis_sums), (
        "mirrored marker pairs must reflect across the grid centerline"
    )

    out = g.copy()
    for i, (_, marker, fill) in enumerate(entries):
        if marker not in pair_info:
            continue

        _, (rmin, cmin, rmax, cmax) = pair_info[marker]
        future_markers = set(marker_order[i + 1 :])
        protect = legend_mask.copy()
        for fm in future_markers:
            protect |= (g == fm) & (~legend_mask)
        protect |= (g == marker) & (~legend_mask)

        region = np.zeros_like(g, dtype=bool)
        region[rmin : rmax + 1, cmin : cmax + 1] = True
        out[region & (~protect)] = int(fill)

    return out


# --------------- generator (mirror-symmetry strip variant) ---------------

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

# --- retry budgets ---
PAIR_RETRIES = 200
GEN_RETRIES = 400


def _build_legend(
    H: int, W: int, bg: int, axis_kind: str
) -> tuple[np.ndarray, list[tuple[int, int, int]], np.ndarray, int, int, list[int]] | None:
    """Create legend entries and arrays. Returns None if the chosen layout leaves no room."""
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
    if len(pool) < n_entries:
        return None
    marker_colors = random.sample(pool, n_entries)

    grid = np.full((H, W), bg, dtype=int)
    legend_rows: list[tuple[int, int, int]] = []
    legend_mask = np.zeros((H, W), dtype=bool)
    for i in range(n_entries):
        r = legend_bot - i
        marker = marker_colors[i]
        fill = random.randint(0, 9)
        legend_rows.append((r, marker, fill))
        grid[r, legend_c] = marker
        grid[r, legend_c + 1] = fill
        legend_mask[r, legend_c] = True
        legend_mask[r, legend_c + 1] = True

    return grid, legend_rows, legend_mask, legend_c, legend_bot, marker_colors


def _build_forbidden(
    H: int, W: int, legend_c: int, legend_bot: int, legend_rows: list[tuple[int, int, int]]
) -> np.ndarray:
    """Protect the legend scan: no markers below the legend or to its left/top."""
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


def _vertical_pair_cells(r0: int, c0: int, c1: int, style: str) -> tuple[list[tuple[int, int]], tuple[int, int, int, int]]:
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


def _horizontal_pair_cells(r0: int, r1: int, c0: int, style: str) -> tuple[list[tuple[int, int]], tuple[int, int, int, int]]:
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
    reserved_rects: list[tuple[int, int, int, int]],
) -> bool:
    """Place one mirrored L-pair and reserve its future fill rectangle."""
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
            if _rect_too_close(rect, reserved_rects, ROW_GAP, 0):
                continue
        else:
            if not horizontal_rmins:
                return False
            r0 = random.choice(horizontal_rmins)
            r1 = H - 1 - r0
            c0 = random.randint(legend_c + 1, W - 2)
            cells, rect = _horizontal_pair_cells(r0, r1, c0, random.choice(["left", "right"]))
            if _rect_too_close(rect, reserved_rects, 0, COL_GAP):
                continue

        if can_place(cells):
            do_place(cells, color)
            reserved_rects.append(rect)
            return True
    return False


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)
    bg = random.randint(0, 9)
    axis_kind = "V" if random.random() < AXIS_VERTICAL_PROB else "H"

    legend_result = _build_legend(H, W, bg, axis_kind)
    if legend_result is None:
        return None
    grid, legend_rows, legend_mask, legend_c, legend_bot, _ = legend_result

    forbidden = _build_forbidden(H, W, legend_c, legend_bot, legend_rows)
    occupied = legend_mask.copy()
    reserved_rects: list[tuple[int, int, int, int]] = []

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
            reserved_rects,
        )
        if not placed:
            return None

    counts = Counter(grid.flatten().tolist())
    if counts.most_common(1)[0][0] != bg:
        return None

    try:
        _ = main(grid)
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
