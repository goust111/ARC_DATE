# Variant of 9385bd28
# Dimension: D6 — 处理顺序与重叠规则 (multi_step)
# Change: ordered_processing — 图例条目从下到上顺序处理，后处理的条目覆盖先处理的；保护后续条目的标记像素和图... → 按标记对面积排序处理

import random
from collections import Counter

import numpy as np

from arc_utils import get_bg


# Design analysis for the D6 variant
# ----------------------------------
# variant_spec:
#   visible_objects:
#     - 左下角仍然是 2 列宽的竖直图例，每行给出 (marker_color, fill_color)。
#     - 非图例区仍然是 L 形对角标记对，或少量 2x2 实心块。
#     - 每个 L 对定义一个轴对齐 bbox，实际被填充的是该 bbox 的整块区域。
#   core_change:
#     - 原题按图例自下而上处理；本变体先为每个可处理条目计算 bbox 面积，再按面积从小到大处理。
#     - 面积更大的对象更晚处理，因此在重叠区覆盖面积更小的对象。
#     - 保护规则同步改成“保护排序中尚未处理的未来条目”的标记像素和图例区。
#   concrete_trace:
#     - 例：图例底部是大矩形 A，上面是小矩形 B，且 B 完全嵌在 A 内部。
#     - 原题顺序：A 先填、B 后填，最终内层区域显示 B 的填充色。
#     - 本变体顺序：先算面积，B 小于 A，所以 B 先填、A 后填，最终整个嵌套区都被 A 的填充色盖住。
#     - 差异不依赖颜色 id，只依赖可见的几何大小与重叠关系。
#   complexity_calibration:
#     - 排序只看少量档位：每个条目只有一个几何面积值，避免引入不可见属性。
#     - 为了让 D6 真正可观察，生成器强制放置一组“图例顺序与面积顺序相反”的嵌套 L 对，并要求新旧顺序输出不同。
#     - 同时避免大量额外重叠，让人类能从少量样例直接看到“谁覆盖谁由尺寸决定”。
#   implicit_assumption_audit:
#     - 原 `for i, entry in enumerate(entries)` 隐含“处理顺序=图例顺序”；变体下不成立，必须先收集对象元数据再排序。
#     - 原 `future_markers = marker_order[i+1:]` 隐含“未来条目=图例上方条目”；变体下要改成“面积排序中位于后面的条目”。
#     - 原生成器只保证 solver 不报错，不保证顺序变化真的影响输出；变体生成器必须显式拒绝 `size_order_output == legend_order_output` 的样本。
#     - 新 solver 新增了“活动条目面积可比较”的假设；生成器据此保证活动条目面积互不相同，避免把 tie-break 变成主规则。
#   placement_legality:
#     - 图例区域仍受保护，标记像素之间不能重叠，所有坐标都必须在界内。
#     - 关键嵌套对使用严格内嵌放置，保证两个 bbox 重叠且标记本身不碰撞。
#     - 额外对象允许存在，但生成器会拒绝污染关键重叠区的杂乱布局。
#   visual_cleanliness:
#     - 关键样本固定包含一大一小两个嵌套矩形，且填充色不同、面积不同、图例顺序反转。
#     - 这样 2-3 个样例就能看出：决定覆盖层级的是矩形尺寸，不是图例上下位置。


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


def _l_anchor_present(
    pts: set[tuple[int, int]],
    r: int,
    c: int,
    arm1: tuple[int, int],
    arm2: tuple[int, int],
) -> bool:
    """
    Marker corners are L-shapes:
      TL: (r,c), (r,c+1), (r+1,c)
      TR: (r,c), (r,c-1), (r+1,c)
      BL: (r,c), (r,c+1), (r-1,c)
      BR: (r,c), (r,c-1), (r-1,c)

    This is more robust than checking only corner membership, especially when the bbox is 2-wide/2-tall
    where L arms can overlap other bbox corners.
    """
    (dr1, dc1) = arm1
    (dr2, dc2) = arm2
    return (r, c) in pts and (r + dr1, c + dc1) in pts and (r + dr2, c + dc2) in pts


def _diamond_mask(
    h: int,
    w: int,
    r0: int,
    r1: int,
    c0: int,
    c1: int,
    a: tuple[int, int],
    b: tuple[int, int],
) -> np.ndarray:
    """
    Mask for all cells inside [r0..r1]x[c0..c1] that lie on some shortest 4-neigh path between corners a and b:
      dist(a,p) + dist(p,b) == dist(a,b)
    """
    ar, ac = a
    br, bc = b
    dab = abs(ar - br) + abs(ac - bc)
    m = np.zeros((h, w), dtype=bool)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if abs(r - ar) + abs(c - ac) + abs(r - br) + abs(c - bc) == dab:
                m[r, c] = True
    return m


def _build_legend_mask(shape: tuple[int, int], legend_c: int, entries: list[tuple[int, int, int]]) -> np.ndarray:
    h, w = shape
    legend_mask = np.zeros((h, w), dtype=bool)
    for r, _, _ in entries:
        if 0 <= r < h and 0 <= legend_c < w:
            legend_mask[r, legend_c] = True
        if 0 <= r < h and 0 <= legend_c + 1 < w:
            legend_mask[r, legend_c + 1] = True
    return legend_mask


def _describe_entry(
    g: np.ndarray,
    legend_mask: np.ndarray,
    legend_index: int,
    marker: int,
    fill: int,
) -> dict[str, object]:
    h, w = g.shape
    pixel_mask = (g == marker) & (~legend_mask)
    marker_pts = set(map(tuple, np.argwhere(pixel_mask)))
    info: dict[str, object] = {
        "legend_index": int(legend_index),
        "marker": int(marker),
        "fill": int(fill),
        "pixel_mask": pixel_mask,
        "marker_pts": marker_pts,
        "mode": "noop",
        "area": None,
        "fill_mask": np.zeros((h, w), dtype=bool),
    }
    if len(marker_pts) < 2:
        return info

    rs = [r for r, _ in marker_pts]
    cs = [c for _, c in marker_pts]
    rmin, rmax = min(rs), max(rs)
    cmin, cmax = min(cs), max(cs)
    area = (rmax - rmin + 1) * (cmax - cmin + 1)

    info["bbox"] = (rmin, rmax, cmin, cmax)
    info["area"] = int(area)

    if rmax - rmin == 1 and cmax - cmin == 1 and len(marker_pts) == 4:
        info["mode"] = "block"
        info["fill_mask"] = pixel_mask.copy()
        return info

    if rmin == rmax or cmin == cmax:
        return info

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
    assert score_tlbr != 0 or score_trbl != 0, "at least one diagonal must have an L-anchor"
    assert score_tlbr != score_trbl, "diagonal scores must not tie (tie-break path is unreachable)"
    if score_tlbr > score_trbl:
        a, b = tr, bl
    else:
        a, b = tl, br

    info["mode"] = "pair"
    info["fill_mask"] = _diamond_mask(h, w, rmin, rmax, cmin, cmax, a, b)
    return info


def _processing_order(infos: list[dict[str, object]], mode: str) -> list[dict[str, object]]:
    active = [info for info in infos if info["mode"] in {"pair", "block"}]
    if mode == "legend":
        return sorted(active, key=lambda info: int(info["legend_index"]))
    if mode == "area":
        return sorted(active, key=lambda info: (int(info["area"]), int(info["legend_index"])))
    raise ValueError(f"unknown mode: {mode}")


def _apply_order(
    g: np.ndarray,
    legend_mask: np.ndarray,
    infos: list[dict[str, object]],
    mode: str,
) -> np.ndarray:
    out = g.copy()
    ordered = _processing_order(infos, mode)
    for pos, info in enumerate(ordered):
        protect = legend_mask.copy()
        for later in ordered[pos + 1 :]:
            protect |= np.asarray(later["pixel_mask"], dtype=bool)

        fill_mask = np.asarray(info["fill_mask"], dtype=bool)
        if info["mode"] == "block":
            out[fill_mask & (~protect)] = int(info["fill"])
            continue

        protect |= np.asarray(info["pixel_mask"], dtype=bool)
        out[fill_mask & (~protect)] = int(info["fill"])
    return out


def _solve_with_mode(grid: np.ndarray, mode: str) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    legend_c, entries = _find_legend(g, bg)
    assert entries, "legend must contain at least one (marker, fill) entry"
    legend_mask = _build_legend_mask(g.shape, legend_c, entries)
    infos = [
        _describe_entry(g, legend_mask, legend_index=i, marker=marker, fill=fill)
        for i, (_, marker, fill) in enumerate(entries)
    ]
    return _apply_order(g, legend_mask, infos, mode)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill rectangles between diagonal marker pairs using a color legend, but process the objects by size.

    Input structure:
    - The background contains a vertical legend near the lower-left corner, with one marker/fill pair per row.
    - Outside the legend, each marker color appears either as two diagonal L-corners of one rectangle or as a solid 2x2 block.
    - Different marker rectangles may overlap in space even though their marker pixels do not touch.

    Rule:
    - Read the legend to recover the marker/fill mapping, then locate the non-legend pixels for each marker color.
    - For every usable marker object, compute the rectangle it spans and sort all such objects by rectangle area from small to large.
    - Process fills in that size order: each step paints that object's rectangle with its legend color while preserving the legend and any marker pixels that belong to objects scheduled later.
    - A solid 2x2 marker block is handled as a degenerate case whose own four pixels are recolored when its turn arrives.

    Key insight: The legend still tells you which color each marker should paint with, but the overwrite hierarchy no longer follows legend position; it follows the visible size of the marked rectangle.
    """
    return _solve_with_mode(grid, mode="area")


# --------------- generator (D6 size-order variant) ---------------

# ── CFG: canvas & grid ──────────────────────────────────────────────
GRID_H_RANGE = (10, 19)
GRID_W_RANGE = (10, 19)

# ── CFG: legend ─────────────────────────────────────────────────────
LEGEND_COL_RANGE = (0, 2)
LEGEND_ENTRY_RANGE = (2, 5)

# ── CFG: marker types ──────────────────────────────────────────────
NOOP_PROB = 0.25
BLOCK_PROB = 0.20

# ── CFG: L-pair bbox shape ──────────────────────────────────────────
BBOX_H_MAX = 19
BBOX_W_MAX = 12
BBOX_SHAPE_WEIGHTS = [0.75, 0.15, 0.10]
TLBR_PROB = 0.8

# ── CFG: distractor ────────────────────────────────────────────────
DISTRACTOR_PROB = 0.25

# ── CFG: critical size-order witness ───────────────────────────────
CRITICAL_RETRIES = 160
CRITICAL_BIG_MIN = 5
CRITICAL_SMALL_MIN = 3

# ── CFG: retry budgets ─────────────────────────────────────────────
BLOCK_RETRIES = 100
LPAIR_RETRIES = 150
DISTRACTOR_RETRIES = 60
GEN_RETRIES = 400

_L_OFFSETS = {
    "TL": [(0, 0), (0, 1), (1, 0)],
    "TR": [(0, 0), (0, -1), (1, 0)],
    "BL": [(0, 0), (0, 1), (-1, 0)],
    "BR": [(0, 0), (0, -1), (-1, 0)],
}


def _l_pair_cells(rmin: int, cmin: int, rmax: int, cmax: int, diag: str) -> list[tuple[int, int]]:
    """Return the 6 cells for an L-pair at opposite bbox corners. diag in {'TLBR','TRBL'}."""
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
    """True if (h, w) avoids an L-anchor score tie."""
    if h < 2 or w < 2:
        return False
    if h == 2 and w <= 3:
        return False
    if w == 2 and h <= 3:
        return False
    return True


def _pick_bbox_size(H: int, W: int) -> tuple[int, int] | None:
    """Pick a random valid L-pair bbox size that fits in (H, W)."""
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


def _try_place_l_pair(
    H: int,
    W: int,
    color: int,
    retries: int,
    can_place,
    do_place,
) -> bool:
    """Try to place an L-pair of `color` on the grid within `retries` attempts."""
    for _ in range(retries):
        size = _pick_bbox_size(H, W)
        if size is None:
            continue
        bh, bw = size
        rmin = random.randint(0, H - bh)
        cmin = random.randint(0, W - bw)
        diag = "TLBR" if random.random() < TLBR_PROB else "TRBL"
        cells = _l_pair_cells(rmin, cmin, rmin + bh - 1, cmin + bw - 1, diag)
        if can_place(cells):
            do_place(cells, color)
            return True
    return False


def _build_legend(H: int, W: int, bg: int):
    """Create legend entries and the corresponding grid/mask arrays."""
    legend_c = random.randint(*LEGEND_COL_RANGE)
    legend_bot = random.choice([H - 1, H - 2])
    n_entries = random.randint(*LEGEND_ENTRY_RANGE)
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
        m = marker_colors[i]
        fill = random.randint(0, 9)
        legend_rows.append((r, m, fill))
        grid[r, legend_c] = m
        legend_mask[r, legend_c] = True
        grid[r, legend_c + 1] = fill
        legend_mask[r, legend_c + 1] = True

    return grid, legend_rows, legend_mask, legend_c, legend_bot, marker_colors


def _build_forbidden(H: int, W: int, legend_c: int, legend_bot: int, legend_rows: list[tuple[int, int, int]]) -> np.ndarray:
    """Build the forbidden-placement mask that protects legend scan integrity."""
    forbidden = np.zeros((H, W), dtype=bool)
    if legend_bot + 1 < H:
        forbidden[legend_bot + 1 :, :] = True
    forbidden[: legend_bot + 1, : legend_c + 1] = True
    for r, _, _ in legend_rows:
        forbidden[r, legend_c] = False
    return forbidden


def _pick_visible_fill_pair(bg: int, used: set[int]) -> tuple[int, int]:
    pool = [c for c in range(10) if c != bg and c not in used]
    if len(pool) >= 2:
        a, b = random.sample(pool, 2)
        return int(a), int(b)
    pool = [c for c in range(10) if c != bg]
    a, b = random.sample(pool, 2)
    return int(a), int(b)


def _try_place_nested_pair(
    H: int,
    W: int,
    big_color: int,
    small_color: int,
    can_place,
    do_place,
) -> dict[str, object] | None:
    """
    Place a large L-pair and a smaller nested L-pair.

    The large one is intended for a lower legend row, the small one for a higher row,
    so legend order and area order disagree on this critical pair.
    """
    big_h_hi = min(BBOX_H_MAX, H, 11)
    big_w_hi = min(BBOX_W_MAX, W, 11)
    if big_h_hi < CRITICAL_BIG_MIN or big_w_hi < CRITICAL_BIG_MIN:
        return None

    for _ in range(CRITICAL_RETRIES):
        bh_big = random.randint(CRITICAL_BIG_MIN, big_h_hi)
        bw_big = random.randint(CRITICAL_BIG_MIN, big_w_hi)
        if not _l_pair_valid_bbox(bh_big, bw_big):
            continue

        small_h_hi = min(bh_big - 2, 6)
        small_w_hi = min(bw_big - 2, 6)
        if small_h_hi < CRITICAL_SMALL_MIN or small_w_hi < CRITICAL_SMALL_MIN:
            continue

        bh_small = random.randint(CRITICAL_SMALL_MIN, small_h_hi)
        bw_small = random.randint(CRITICAL_SMALL_MIN, small_w_hi)
        if not _l_pair_valid_bbox(bh_small, bw_small):
            continue

        area_big = bh_big * bw_big
        area_small = bh_small * bw_small
        if area_big <= area_small:
            continue

        r_big = random.randint(0, H - bh_big)
        c_big = random.randint(0, W - bw_big)

        r_small_lo = r_big + 1
        r_small_hi = r_big + bh_big - bh_small - 1
        c_small_lo = c_big + 1
        c_small_hi = c_big + bw_big - bw_small - 1
        if r_small_lo > r_small_hi or c_small_lo > c_small_hi:
            continue

        r_small = random.randint(r_small_lo, r_small_hi)
        c_small = random.randint(c_small_lo, c_small_hi)
        diag_big = "TLBR" if random.random() < TLBR_PROB else "TRBL"
        diag_small = "TLBR" if random.random() < TLBR_PROB else "TRBL"

        big_cells = _l_pair_cells(r_big, c_big, r_big + bh_big - 1, c_big + bw_big - 1, diag_big)
        small_cells = _l_pair_cells(
            r_small,
            c_small,
            r_small + bh_small - 1,
            c_small + bw_small - 1,
            diag_small,
        )
        union = big_cells + small_cells
        if len(set(union)) != len(union):
            continue
        if not can_place(union):
            continue

        do_place(big_cells, big_color)
        do_place(small_cells, small_color)
        return {
            "big_bbox": (r_big, c_big, r_big + bh_big - 1, c_big + bw_big - 1),
            "small_bbox": (r_small, c_small, r_small + bh_small - 1, c_small + bw_small - 1),
        }
    return None


def _analyze_grid(grid: np.ndarray):
    bg = get_bg(grid)
    legend_c, entries = _find_legend(grid, bg)
    legend_mask = _build_legend_mask(grid.shape, legend_c, entries)
    infos = [
        _describe_entry(grid, legend_mask, legend_index=i, marker=marker, fill=fill)
        for i, (_, marker, fill) in enumerate(entries)
    ]
    return legend_c, entries, legend_mask, infos


def _critical_region_clean(infos: list[dict[str, object]]) -> bool:
    if len(infos) < 2:
        return False
    big = infos[0]
    small = infos[1]
    if big["mode"] != "pair" or small["mode"] != "pair":
        return False
    if int(big["area"]) <= int(small["area"]):
        return False

    overlap = np.asarray(big["fill_mask"], dtype=bool) & np.asarray(small["fill_mask"], dtype=bool)
    if not np.any(overlap):
        return False

    for info in infos[2:]:
        if info["mode"] not in {"pair", "block"}:
            continue
        if np.any(np.asarray(info["fill_mask"], dtype=bool) & overlap):
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)
    bg = random.randint(0, 9)

    legend_result = _build_legend(H, W, bg)
    if legend_result is None:
        return None
    grid, legend_rows, legend_mask, legend_c, legend_bot, marker_colors = legend_result

    # Force the first two legend entries to be a visible size-order witness:
    # bottom entry = larger rectangle, next entry above = smaller nested rectangle.
    critical_big_fill, critical_small_fill = _pick_visible_fill_pair(bg, set(marker_colors[:2]))
    r0, m0, _ = legend_rows[0]
    r1, m1, _ = legend_rows[1]
    legend_rows[0] = (r0, m0, critical_big_fill)
    legend_rows[1] = (r1, m1, critical_small_fill)
    grid[r0, legend_c + 1] = critical_big_fill
    grid[r1, legend_c + 1] = critical_small_fill

    forbidden = _build_forbidden(H, W, legend_c, legend_bot, legend_rows)
    occupied = legend_mask.copy()

    def can_place(cells):
        for r, c in cells:
            if not (0 <= r < H and 0 <= c < W):
                return False
            if occupied[r, c] or forbidden[r, c]:
                return False
        return True

    def do_place(cells, color):
        for r, c in cells:
            grid[r, c] = color
            occupied[r, c] = True

    nested = _try_place_nested_pair(H, W, marker_colors[0], marker_colors[1], can_place, do_place)
    if nested is None:
        return None

    # Assign a marker type per remaining legend entry.
    n_entries = len(legend_rows)
    types = ["L"] * n_entries
    extra_indices = list(range(2, n_entries))
    random.shuffle(extra_indices)
    if random.random() < NOOP_PROB and extra_indices:
        types[extra_indices.pop()] = "noop"
    if random.random() < BLOCK_PROB and extra_indices:
        types[extra_indices.pop()] = "block"

    for i in range(2, n_entries):
        _, m, _ = legend_rows[i]
        t = types[i]
        if t == "noop":
            continue
        if t == "block":
            placed = False
            for _ in range(BLOCK_RETRIES):
                r0 = random.randint(0, H - 2)
                c0 = random.randint(0, W - 2)
                cells = [(r0, c0), (r0, c0 + 1), (r0 + 1, c0), (r0 + 1, c0 + 1)]
                if can_place(cells):
                    do_place(cells, m)
                    placed = True
                    break
            if not placed:
                return None
        else:
            if not _try_place_l_pair(H, W, m, LPAIR_RETRIES, can_place, do_place):
                return None

    if random.random() < DISTRACTOR_PROB:
        avail = [c for c in range(10) if c != bg and c not in marker_colors]
        if avail:
            _try_place_l_pair(H, W, random.choice(avail), DISTRACTOR_RETRIES, can_place, do_place)

    counts = Counter(grid.flatten().tolist())
    if counts.most_common(1)[0][0] != bg:
        return None

    try:
        _, _, _, infos = _analyze_grid(grid)
        area_order = [int(info["legend_index"]) for info in _processing_order(infos, "area")]
        legend_order = [int(info["legend_index"]) for info in _processing_order(infos, "legend")]
        active_areas = [int(info["area"]) for info in infos if info["mode"] in {"pair", "block"}]
        size_out = _solve_with_mode(grid, "area")
        legend_out = _solve_with_mode(grid, "legend")
    except Exception:
        return None

    if len(active_areas) < 2:
        return None
    if len(set(active_areas)) != len(active_areas):
        return None
    if area_order == legend_order:
        return None
    if np.array_equal(size_out, legend_out):
        return None
    if not _critical_region_clean(infos):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError(f"generate_input: failed to generate a valid input after {GEN_RETRIES} attempts")
