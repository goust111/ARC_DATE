# Combo variant of 9385bd28 (index 8)
# combo_spec: D2/3+D6/1+D12/2
# Name: 边框标记+多尺寸变体+按面积排序

import random
from collections import Counter

import numpy as np

from arc_utils import get_bg


# Design analysis
# ---------------
# fusion_summary:
#   preserved:
#   - 左下角仍是两列宽的竖直图例；每行给出 (marker_color, fill_color)。
#   - 非图例对象仍按图例提供的颜色映射执行。
#   - 对于“成对标记”对象，仍然是先识别被标出的对角线，再填另一条对角线对应的曼哈顿菱形。
#   changed:
#   - D2/3: 成对标记不再是 L 角，而是隐藏矩形边框上的两段连续像素。两段只会出现在相对边上：
#     顶+底或左+右；每段只贴某一端，因此能指示 TLBR 或 TRBL 哪条对角线被标出。
#   - D12/2: 仍保留“实心块是 recolor-only 特例”，但尺寸不再只限 2x2，而扩展到 2x3、3x2、3x3 等。
#   - D6/1: 处理顺序不再看图例自下而上，而是先为每个活动对象算 bbox 面积，再按面积从小到大处理。
#
# concrete_trace:
#   original_flow:
#   - 旧题对某个 marker 色先收集所有同色像素；若是 2x2 实心块则直接重染，否则假定它是两枚 L 角，
#     通过四角 anchor 判断被标出的对角线，然后按图例顺序填充。
#   fusion_flow:
#   - 现在先判断同色像素是否构成一个完整实心矩形；若是，则把该矩形本身重染。
#   - 否则要求这些像素恰好分布在某个 bbox 的一对相对边上，并且各自形成一段连续边框残片。
#   - 例如：顶边左端一段 + 底边右端一段 => 触碰 TL 和 BR，表示被标出的是 TLBR，对应要填 TR-BL 菱形。
#   - 所有活动对象再按各自 bbox 面积升序处理。若图例底部是大矩形、上面是小矩形，且二者嵌套重叠，
#     则旧题会让小矩形后画覆盖大矩形；本融合版会让大矩形后画，反过来覆盖小矩形。
#
# complexity_calibration:
#   - 边框残片只保留两种朝向（顶/底、左/右），避免“任意两条边组合”带来歧义。
#   - 实心块尺寸限制在少量可见档位 {(2,2),(2,3),(3,2),(3,3)}，既能明显区别“不是成对边框标记”，
#     又不会引入过多类别。
#   - 生成器强制包含一组“图例顺序与面积顺序冲突”的嵌套边框对象，以及一个多尺寸实心块，
#     这样 2-3 个样例就能同时看到三项改动都在起作用。
#
# implicit_assumption_audit:
#   - 原 `_l_anchor_present(...)` 依赖 L 角结构；融合后不成立，必须改为“连续边段触碰哪个角”的检测。
#   - 原 `if rmax-rmin==1 and cmax-cmin==1 and len==4` 假定唯一的 dense 特例就是 2x2；融合后必须改成
#     “任意 height>=2 且 width>=2 的完整实心矩形”。
#   - 原 `for i, entry in enumerate(entries)` 默认处理顺序等于图例顺序；融合后必须先描述所有对象，再按 area 排序。
#   - 新 solver 额外假设：
#     1. 每个非 noop 图例色只有一个对象。
#     2. 稀疏对象一定是合法的边框残片对，而不是其它杂形。
#     3. 活动对象面积最好互异，否则需要 tie-break。代码用 legend_index 做稳定兜底，生成器则主动规避面积相等。
#
# placement_legality:
#   - 所有 marker 像素仍禁止与图例和其它 marker 像素重叠，且都必须在界内。
#   - 边框残片 bbox 至少 4x4，使边段能“短于整边”且仍清晰可见。
#   - 关键嵌套见证使用严格内嵌放置；其它对象允许 bbox 交叠，但生成器会拒绝污染关键重叠区的样本。
#
# visual_observability:
#   - 排序只依赖人眼可见的 bbox 大小，不依赖颜色 id 或其它不可见属性。
#   - 输入里会反复出现两类几何：边框残片对与实心块；输出则反复出现两类行为：菱形填充与原地重染。
#   - 关键大/小嵌套对象的 fill 色不同，且新旧顺序输出必须不同，保证面积排序不是“代码里有、样本里没体现”。
#
# cleanliness_and_inference:
#   - 关键重叠区只让那一对嵌套对象参与，避免多层覆盖导致看不清是谁覆盖谁。
#   - 颜色变化始终由图例两列显式给出，不要求解题者猜测“颜色数值”或额外隐变量。


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


def _build_legend_mask(shape: tuple[int, int], legend_c: int, entries: list[tuple[int, int, int]]) -> np.ndarray:
    h, w = shape
    legend_mask = np.zeros((h, w), dtype=bool)
    for r, _, _ in entries:
        if 0 <= r < h and 0 <= legend_c < w:
            legend_mask[r, legend_c] = True
        if 0 <= r < h and 0 <= legend_c + 1 < w:
            legend_mask[r, legend_c + 1] = True
    return legend_mask


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


def _is_contiguous(vals: list[int]) -> bool:
    if not vals:
        return False
    ordered = sorted(vals)
    return len(ordered) == len(set(ordered)) and ordered[-1] - ordered[0] + 1 == len(ordered)


def _solid_rect_mask(h: int, w: int, marker_pts: set[tuple[int, int]]) -> np.ndarray | None:
    if len(marker_pts) < 4:
        return None
    rs = [r for r, _ in marker_pts]
    cs = [c for _, c in marker_pts]
    rmin, rmax = min(rs), max(rs)
    cmin, cmax = min(cs), max(cs)
    bh = rmax - rmin + 1
    bw = cmax - cmin + 1
    if bh < 2 or bw < 2 or len(marker_pts) != bh * bw:
        return None
    mask = np.zeros((h, w), dtype=bool)
    for r in range(rmin, rmax + 1):
        for c in range(cmin, cmax + 1):
            if (r, c) not in marker_pts:
                return None
            mask[r, c] = True
    return mask


def _infer_fill_diagonal(
    marker_pts: set[tuple[int, int]]
) -> tuple[int, int, int, int, tuple[int, int], tuple[int, int]]:
    """
    Infer the hidden bbox and the opposite fill diagonal from two partial boundary segments.

    Valid sparse shapes are exactly one of:
    - top + bottom contiguous segments
    - left + right contiguous segments
    Each segment touches exactly one end of its side, and the two touched corners
    determine which diagonal was marked.
    """
    rs = [r for r, _ in marker_pts]
    cs = [c for _, c in marker_pts]
    rmin, rmax = min(rs), max(rs)
    cmin, cmax = min(cs), max(cs)
    assert rmin < rmax and cmin < cmax, "marker bbox must have positive area"

    horizontal_only = all(r in (rmin, rmax) for r, _ in marker_pts)
    vertical_only = all(c in (cmin, cmax) for _, c in marker_pts)
    assert horizontal_only != vertical_only, "marker must occupy exactly one pair of opposite edges"

    tl = (rmin, cmin)
    tr = (rmin, cmax)
    bl = (rmax, cmin)
    br = (rmax, cmax)

    if horizontal_only:
        top_cs = sorted(c for r, c in marker_pts if r == rmin)
        bot_cs = sorted(c for r, c in marker_pts if r == rmax)
        assert top_cs and bot_cs and _is_contiguous(top_cs) and _is_contiguous(bot_cs), (
            "top/bottom segments must be contiguous"
        )
        assert (top_cs[0] == cmin) ^ (top_cs[-1] == cmax), "top segment must touch exactly one corner"
        assert (bot_cs[0] == cmin) ^ (bot_cs[-1] == cmax), "bottom segment must touch exactly one corner"
        score_tlbr = int(top_cs[0] == cmin) + int(bot_cs[-1] == cmax)
        score_trbl = int(top_cs[-1] == cmax) + int(bot_cs[0] == cmin)
    else:
        left_rs = sorted(r for r, c in marker_pts if c == cmin)
        right_rs = sorted(r for r, c in marker_pts if c == cmax)
        assert left_rs and right_rs and _is_contiguous(left_rs) and _is_contiguous(right_rs), (
            "left/right segments must be contiguous"
        )
        assert (left_rs[0] == rmin) ^ (left_rs[-1] == rmax), "left segment must touch exactly one corner"
        assert (right_rs[0] == rmin) ^ (right_rs[-1] == rmax), "right segment must touch exactly one corner"
        score_tlbr = int(left_rs[0] == rmin) + int(right_rs[-1] == rmax)
        score_trbl = int(right_rs[0] == rmin) + int(left_rs[-1] == rmax)

    assert score_tlbr != score_trbl, "marker diagonal must be uniquely determined"
    if score_tlbr > score_trbl:
        a, b = tr, bl
    else:
        a, b = tl, br
    return rmin, rmax, cmin, cmax, a, b


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
        "bbox": None,
        "fill_mask": np.zeros((h, w), dtype=bool),
    }
    if len(marker_pts) < 2:
        return info

    block_mask = _solid_rect_mask(h, w, marker_pts)
    if block_mask is not None:
        rs = [r for r, _ in marker_pts]
        cs = [c for _, c in marker_pts]
        rmin, rmax = min(rs), max(rs)
        cmin, cmax = min(cs), max(cs)
        info["mode"] = "block"
        info["bbox"] = (rmin, rmax, cmin, cmax)
        info["area"] = int((rmax - rmin + 1) * (cmax - cmin + 1))
        info["fill_mask"] = block_mask
        return info

    rmin, rmax, cmin, cmax, a, b = _infer_fill_diagonal(marker_pts)
    info["mode"] = "pair"
    info["bbox"] = (rmin, rmax, cmin, cmax)
    info["area"] = int((rmax - rmin + 1) * (cmax - cmin + 1))
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
    Recolor legend-coded marker objects, inferring hidden rectangles from boundary fragments and applying them by size.

    Input structure:
    - A background grid contains a vertical two-column legend near the bottom-left; each row gives (marker_color, fill_color).
    - Away from the legend, a marker color appears either as two short boundary fragments on opposite sides of a hidden rectangle or as a solid rectangular block.
    - Boundary-fragment markers can have different bbox sizes, and solid blocks can be 2x2, 2x3, 3x2, 3x3, and similar small rectangles.

    Rule:
    - For each legend color, collect its non-legend pixels and classify the object.
    - If the pixels form a full rectangle of size at least 2x2, recolor that rectangle only.
    - Otherwise, recover the hidden bbox from the extreme rows/cols, determine which two corners are touched by the two boundary fragments, and fill the opposite diagonal's Manhattan diamond.
    - Execute all active objects in ascending bbox-area order, preserving the legend and the marker pixels of objects scheduled later; sparse boundary markers also preserve their own pixels.

    Key insight: The legend still supplies the color mapping, but two different visible geometries now coexist: dense rectangles mean "recolor me", while sparse edge fragments reveal a hidden rectangle whose overwrite priority is determined by its visible size, not by legend position.
    """
    return _solve_with_mode(grid, mode="area")


# --------------- generator (combo D2/3 + D6/1 + D12/2) ---------------

# ── CFG: canvas & grid ──────────────────────────────────────────────
GRID_H_RANGE = (10, 19)
GRID_W_RANGE = (10, 19)

# ── CFG: legend ─────────────────────────────────────────────────────
LEGEND_COL_RANGE = (0, 2)
LEGEND_ENTRY_RANGE = (3, 5)

# ── CFG: marker types ──────────────────────────────────────────────
NOOP_PROB = 0.20
EXTRA_EDGE_PROB = 0.65
EDGE_ORIENT_WEIGHTS = [0.55, 0.45]  # H, V
BLOCK_DIM_OPTIONS = [(2, 2), (2, 3), (3, 2), (3, 3)]
BLOCK_DIM_WEIGHTS = [0.10, 0.35, 0.20, 0.35]

# ── CFG: edge-pair bbox shape ──────────────────────────────────────
BBOX_H_MIN = 4
BBOX_W_MIN = 4
BBOX_H_MAX = 12
BBOX_W_MAX = 12
TLBR_PROB = 0.5

# ── CFG: distractor ────────────────────────────────────────────────
DISTRACTOR_PROB = 0.25

# ── CFG: critical area-order witness ───────────────────────────────
CRITICAL_RETRIES = 200
CRITICAL_BIG_MIN = 6
CRITICAL_SMALL_MIN = 4

# ── CFG: retry budgets ─────────────────────────────────────────────
BLOCK_RETRIES = 120
EDGE_RETRIES = 180
DISTRACTOR_RETRIES = 80
GEN_RETRIES = 500


def _pick_bbox_size(H: int, W: int, *, h_min: int = BBOX_H_MIN, w_min: int = BBOX_W_MIN) -> tuple[int, int] | None:
    h_hi = min(BBOX_H_MAX, H)
    w_hi = min(BBOX_W_MAX, W)
    if h_hi < h_min or w_hi < w_min:
        return None
    return random.randint(h_min, h_hi), random.randint(w_min, w_hi)


def _pick_segment_len(span: int) -> int:
    return random.randint(2, max(2, span // 2))


def _edge_pair_cells(
    rmin: int,
    cmin: int,
    rmax: int,
    cmax: int,
    orientation: str,
    diag: str,
    len1: int,
    len2: int,
) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    if orientation == "H":
        if diag == "TLBR":
            cells.extend((rmin, c) for c in range(cmin, cmin + len1))
            cells.extend((rmax, c) for c in range(cmax - len2 + 1, cmax + 1))
        else:
            cells.extend((rmin, c) for c in range(cmax - len1 + 1, cmax + 1))
            cells.extend((rmax, c) for c in range(cmin, cmin + len2))
    else:
        if diag == "TLBR":
            cells.extend((r, cmin) for r in range(rmin, rmin + len1))
            cells.extend((r, cmax) for r in range(rmax - len2 + 1, rmax + 1))
        else:
            cells.extend((r, cmax) for r in range(rmin, rmin + len1))
            cells.extend((r, cmin) for r in range(rmax - len2 + 1, rmax + 1))
    return cells


def _pick_block_size(H: int, W: int) -> tuple[int, int] | None:
    fitting = [
        (dims, weight)
        for dims, weight in zip(BLOCK_DIM_OPTIONS, BLOCK_DIM_WEIGHTS)
        if dims[0] <= H and dims[1] <= W
    ]
    if not fitting:
        return None
    dims = random.choices([d for d, _ in fitting], weights=[w for _, w in fitting])[0]
    return int(dims[0]), int(dims[1])


def _solid_block_cells(r0: int, c0: int, bh: int, bw: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(r0, r0 + bh) for c in range(c0, c0 + bw)]


def _try_place_edge_pair(H: int, W: int, color: int, retries: int, can_place, do_place) -> bool:
    for _ in range(retries):
        size = _pick_bbox_size(H, W)
        if size is None:
            continue
        bh, bw = size
        rmin = random.randint(0, H - bh)
        cmin = random.randint(0, W - bw)
        rmax = rmin + bh - 1
        cmax = cmin + bw - 1
        orientation = random.choices(["H", "V"], weights=EDGE_ORIENT_WEIGHTS)[0]
        diag = "TLBR" if random.random() < TLBR_PROB else "TRBL"
        span = bw if orientation == "H" else bh
        len1 = _pick_segment_len(span)
        len2 = _pick_segment_len(span)
        cells = _edge_pair_cells(rmin, cmin, rmax, cmax, orientation, diag, len1, len2)
        if can_place(cells):
            do_place(cells, color)
            return True
    return False


def _build_legend(H: int, W: int, bg: int):
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
        marker = marker_colors[i]
        fill = _pick_fill_color(bg, marker)
        legend_rows.append((r, marker, fill))
        grid[r, legend_c] = marker
        legend_mask[r, legend_c] = True
        grid[r, legend_c + 1] = fill
        legend_mask[r, legend_c + 1] = True

    return grid, legend_rows, legend_mask, legend_c, legend_bot, marker_colors


def _build_forbidden(
    H: int,
    W: int,
    legend_c: int,
    legend_bot: int,
    legend_rows: list[tuple[int, int, int]],
) -> np.ndarray:
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


def _pick_fill_color(bg: int, marker: int, avoid: set[int] | None = None) -> int:
    blocked = {bg, marker}
    if avoid is not None:
        blocked |= set(avoid)
    pool = [c for c in range(10) if c not in blocked]
    if pool:
        return int(random.choice(pool))
    pool = [c for c in range(10) if c != bg and c != marker]
    if pool:
        return int(random.choice(pool))
    return int((marker + 1) % 10)


def _try_place_nested_edge_witness(
    H: int,
    W: int,
    big_color: int,
    small_color: int,
    can_place,
    do_place,
) -> bool:
    big_h_hi = min(BBOX_H_MAX, H, 12)
    big_w_hi = min(BBOX_W_MAX, W, 12)
    if big_h_hi < CRITICAL_BIG_MIN or big_w_hi < CRITICAL_BIG_MIN:
        return False

    for _ in range(CRITICAL_RETRIES):
        bh_big = random.randint(CRITICAL_BIG_MIN, big_h_hi)
        bw_big = random.randint(CRITICAL_BIG_MIN, big_w_hi)
        small_h_hi = min(bh_big - 2, 8)
        small_w_hi = min(bw_big - 2, 8)
        if small_h_hi < CRITICAL_SMALL_MIN or small_w_hi < CRITICAL_SMALL_MIN:
            continue

        bh_small = random.randint(CRITICAL_SMALL_MIN, small_h_hi)
        bw_small = random.randint(CRITICAL_SMALL_MIN, small_w_hi)
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

        orient_big = random.choices(["H", "V"], weights=EDGE_ORIENT_WEIGHTS)[0]
        orient_small = random.choices(["H", "V"], weights=EDGE_ORIENT_WEIGHTS)[0]
        diag_big = "TLBR" if random.random() < TLBR_PROB else "TRBL"
        diag_small = "TLBR" if random.random() < TLBR_PROB else "TRBL"

        len_big_a = _pick_segment_len(bw_big if orient_big == "H" else bh_big)
        len_big_b = _pick_segment_len(bw_big if orient_big == "H" else bh_big)
        len_small_a = _pick_segment_len(bw_small if orient_small == "H" else bh_small)
        len_small_b = _pick_segment_len(bw_small if orient_small == "H" else bh_small)

        big_cells = _edge_pair_cells(
            r_big, c_big, r_big + bh_big - 1, c_big + bw_big - 1,
            orient_big, diag_big, len_big_a, len_big_b
        )
        small_cells = _edge_pair_cells(
            r_small, c_small, r_small + bh_small - 1, c_small + bw_small - 1,
            orient_small, diag_small, len_small_a, len_small_b
        )
        union = big_cells + small_cells
        if len(set(union)) != len(union):
            continue
        if not can_place(union):
            continue

        do_place(big_cells, big_color)
        do_place(small_cells, small_color)
        return True
    return False


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
    if len(infos) < 3:
        return False
    big = infos[0]
    small = infos[1]
    block = infos[2]
    if big["mode"] != "pair" or small["mode"] != "pair" or block["mode"] != "block":
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


def _block_has_nontrivial_size(info: dict[str, object]) -> bool:
    if info["mode"] != "block" or info["bbox"] is None:
        return False
    rmin, rmax, cmin, cmax = info["bbox"]
    bh = int(rmax - rmin + 1)
    bw = int(cmax - cmin + 1)
    return (bh, bw) != (2, 2)


def _block_is_visible_and_isolated(infos: list[dict[str, object]]) -> bool:
    if len(infos) < 3 or infos[2]["mode"] != "block":
        return False
    block_mask = np.asarray(infos[2]["fill_mask"], dtype=bool)
    other_fill = np.zeros_like(block_mask)
    for info in infos:
        if info is infos[2] or info["mode"] not in {"pair", "block"}:
            continue
        other_fill |= np.asarray(info["fill_mask"], dtype=bool)
    return not np.any(block_mask & other_fill)


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)
    bg = random.randint(0, 9)

    legend_result = _build_legend(H, W, bg)
    if legend_result is None:
        return None
    grid, legend_rows, legend_mask, legend_c, legend_bot, marker_colors = legend_result

    critical_big_fill, critical_small_fill = _pick_visible_fill_pair(bg, set(marker_colors[:2]))
    r0, m0, _ = legend_rows[0]
    r1, m1, _ = legend_rows[1]
    legend_rows[0] = (r0, m0, critical_big_fill)
    legend_rows[1] = (r1, m1, critical_small_fill)
    grid[r0, legend_c + 1] = critical_big_fill
    grid[r1, legend_c + 1] = critical_small_fill

    forbidden = _build_forbidden(H, W, legend_c, legend_bot, legend_rows)
    occupied = legend_mask.copy()

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

    if not _try_place_nested_edge_witness(H, W, marker_colors[0], marker_colors[1], can_place, do_place):
        return None

    dims = _pick_block_size(H, W)
    if dims is None:
        return None
    block_fill = _pick_fill_color(
        bg,
        marker_colors[2],
        avoid={critical_big_fill, critical_small_fill},
    )
    r2, m2, _ = legend_rows[2]
    legend_rows[2] = (r2, m2, block_fill)
    grid[r2, legend_c + 1] = block_fill

    placed_block = False
    for _ in range(BLOCK_RETRIES):
        bh, bw = dims
        r0 = random.randint(0, H - bh)
        c0 = random.randint(0, W - bw)
        cells = _solid_block_cells(r0, c0, bh, bw)
        if can_place(cells):
            do_place(cells, marker_colors[2])
            placed_block = True
            break
    if not placed_block:
        return None

    for i in range(3, len(legend_rows)):
        _, marker, _ = legend_rows[i]
        roll = random.random()
        if roll < NOOP_PROB:
            continue
        if roll < NOOP_PROB + EXTRA_EDGE_PROB:
            if not _try_place_edge_pair(H, W, marker, EDGE_RETRIES, can_place, do_place):
                return None
        else:
            dims = _pick_block_size(H, W)
            if dims is None:
                return None
            placed = False
            for _ in range(BLOCK_RETRIES):
                bh, bw = dims
                r0 = random.randint(0, H - bh)
                c0 = random.randint(0, W - bw)
                cells = _solid_block_cells(r0, c0, bh, bw)
                if can_place(cells):
                    do_place(cells, marker)
                    placed = True
                    break
            if not placed:
                return None

    if random.random() < DISTRACTOR_PROB:
        avail = [c for c in range(10) if c != bg and c not in marker_colors]
        if avail:
            _try_place_edge_pair(H, W, random.choice(avail), DISTRACTOR_RETRIES, can_place, do_place)

    counts = Counter(grid.flatten().tolist())
    if counts.most_common(1)[0][0] != bg:
        return None

    try:
        _, _, _, infos = _analyze_grid(grid)
        area_order = [int(info["legend_index"]) for info in _processing_order(infos, "area")]
        legend_order = [int(info["legend_index"]) for info in _processing_order(infos, "legend")]
        active_infos = [info for info in infos if info["mode"] in {"pair", "block"}]
        active_areas = [int(info["area"]) for info in active_infos]
        size_out = _solve_with_mode(grid, "area")
        legend_out = _solve_with_mode(grid, "legend")
    except Exception:
        return None

    if len(active_infos) < 3:
        return None
    if len(set(active_areas)) != len(active_areas):
        return None
    if area_order == legend_order:
        return None
    if np.array_equal(size_out, legend_out):
        return None
    if not _critical_region_clean(infos):
        return None
    if not _block_has_nontrivial_size(infos[2]):
        return None
    if not _block_is_visible_and_isolated(infos):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError(f"generate_input: failed to generate a valid input after {GEN_RETRIES} attempts")
