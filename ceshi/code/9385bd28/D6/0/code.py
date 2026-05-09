# Variant of 9385bd28
# Dimension: D6 — 处理顺序与重叠规则 (multi_step)
# Change: ordered_processing — 图例条目从下到上顺序处理，后处理的条目覆盖先处理的；保护后续条目的标记像素和图... → 从上到下处理（反转顺序）
"""
Design analysis
===============

spec:
  source_task: none
  kept_structure:
    - 输入仍然是左下角 2 列宽竖直图例，图例条目按底到顶读取，条目给出 (marker_color, fill_color)。
    - 非图例区仍然放置对应颜色的 L 角标记对，或少量 2x2 实心块。
    - 每个 L 对仍然由其包围盒四角中的一条对角线承载标记，真正被染色的是另一条对角线定义出的整块矩形区域。
  variant_change:
    - 唯一核心规则变化是“执行顺序”和“覆盖优先级”。
    - 原题: 图例自下而上读取并按同样顺序执行，因此更高的图例条目最后落笔。
    - 变体: 图例仍然自下而上读取颜色对应关系，但真正执行时改为自上而下，因此更低的图例条目最后落笔。
  behavior_trace:
    - 设底部条目 A 的填充矩形与顶部条目 B 的填充矩形发生交叠，且两者填充色不同。
    - 原题执行: A 先填，B 后填，所以交叠区最终呈现 B 的填充色；A 同时必须保护 B 的标记像素。
    - 变体执行: B 先填，A 后填，所以交叠区最终改为 A 的填充色；B 现在必须保护 A 的标记像素。
    - 因此可见差异不在“如何读图例”，而在“谁最后覆盖交叠区”。
  complexity_calibration:
    - 只反转顺序而不强制制造交叠，很多样本会与原题输出一致，视觉上看不出 D6。
    - 为了让规则可观察，生成器必须稳定地产生至少一对真实交叠的 L 矩形，并让它们的填充色不同。
    - 仍保留少量 noop、2x2 block、非图例干扰色，避免任务退化成“永远只有两个矩形对冲”。
  implicit_assumptions_review:
    - _find_legend 依赖“图例仍在左下角且按底到顶连续堆叠”，该前提未改动，可直接保留。
    - marker 几何识别仍应基于原输入 g，而不是中间输出 out；因为处理顺序变化只影响覆盖次序，不改变标记形状来源。
    - 原 solver 的 future_markers = marker_order[i+1:] 隐含了“未来 = 图例更高处”。变体中这个假设失效，必须改成“未来 = 当前条目之下的图例项”。
    - 对 L-pair 仍需保护当前 marker 像素；对 2x2 block 仍不保护当前像素，因为它本身应被重着色。这一点与 D6 无关，不应改动。
  placement_legality:
    - 标记像素依旧禁止互相重叠、禁止碰撞图例扫描列、禁止越界。
    - 为了制造顺序冲突，只允许填充矩形重叠，不允许 marker 重叠；否则对象身份会变得含混。
    - 生成器在放置完毕后同时运行“新顺序 solver”和“旧顺序 solver”，若两者输出相同则整样本作废。
  visual_observability:
    - 顺序不依赖颜色 id 排序，只依赖图例中的上下位置，这是可见结构线索。
    - 生成器固定让最底与最顶两个 legend 条目形成一次真实交叠，解题者能从多组样例中看出“读图例方向”和“覆盖优先级”脱钩。
  cleanliness:
    - 交叠只发生在填充矩形，不发生在 marker 本体和图例区域，避免画面过脏。
    - 其余条目仍按原任务风格随机散布，保证整体外观接近原题，而不是引入新的对象体系。
  new_main_assumptions:
    - 至少存在两个可执行 legend 条目时，生成器优先让最上/最下两项为 L-pair 且交叠。
    - generate_input 通过差分验证保证“反转顺序”确实影响结果，而不仅仅是理论上可影响。
"""

import random
from collections import Counter

import numpy as np

from arc_utils import get_bg


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


def _build_legend_mask(shape: tuple[int, int], legend_c: int,
                       entries: list[tuple[int, int, int]]) -> np.ndarray:
    h, w = shape
    legend_mask = np.zeros((h, w), dtype=bool)
    for r, _, _ in entries:
        if 0 <= r < h and 0 <= legend_c < w:
            legend_mask[r, legend_c] = True
        if 0 <= r < h and 0 <= legend_c + 1 < w:
            legend_mask[r, legend_c + 1] = True
    return legend_mask


def _l_anchor_present(pts: set[tuple[int, int]], r: int, c: int,
                      arm1: tuple[int, int], arm2: tuple[int, int]) -> bool:
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


def _diamond_mask(h: int, w: int, r0: int, r1: int, c0: int, c1: int,
                  a: tuple[int, int], b: tuple[int, int]) -> np.ndarray:
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


def _marker_effect_data(g: np.ndarray, legend_mask: np.ndarray, marker: int) -> dict[str, np.ndarray | str]:
    marker_mask = (g == marker) & (~legend_mask)
    pts_arr = np.argwhere(marker_mask)
    pts = set(map(tuple, pts_arr))
    empty_fill = np.zeros_like(g, dtype=bool)

    if len(pts) < 2:
        return {"kind": "none", "marker_mask": marker_mask, "fill_mask": empty_fill}

    rs = pts_arr[:, 0]
    cs = pts_arr[:, 1]
    rmin, rmax = int(rs.min()), int(rs.max())
    cmin, cmax = int(cs.min()), int(cs.max())

    if rmax - rmin == 1 and cmax - cmin == 1 and len(pts) == 4:
        return {"kind": "block", "marker_mask": marker_mask, "fill_mask": marker_mask}

    if rmin == rmax or cmin == cmax:
        return {"kind": "degenerate", "marker_mask": marker_mask, "fill_mask": empty_fill}

    tl = (rmin, cmin)
    tr = (rmin, cmax)
    bl = (rmax, cmin)
    br = (rmax, cmax)

    tl_anchor = _l_anchor_present(pts, tl[0], tl[1], (0, 1), (1, 0))
    tr_anchor = _l_anchor_present(pts, tr[0], tr[1], (0, -1), (1, 0))
    bl_anchor = _l_anchor_present(pts, bl[0], bl[1], (0, 1), (-1, 0))
    br_anchor = _l_anchor_present(pts, br[0], br[1], (0, -1), (-1, 0))

    score_tlbr = int(tl_anchor) + int(br_anchor)
    score_trbl = int(tr_anchor) + int(bl_anchor)
    assert score_tlbr != 0 or score_trbl != 0, "at least one diagonal must have an L-anchor"
    assert score_tlbr != score_trbl, "diagonal scores must not tie (tie-break path is unreachable)"
    if score_tlbr > score_trbl:
        a, b = tr, bl
    else:
        a, b = tl, br

    fill_mask = _diamond_mask(g.shape[0], g.shape[1], rmin, rmax, cmin, cmax, a, b)
    return {"kind": "diamond", "marker_mask": marker_mask, "fill_mask": fill_mask}


def _solve_with_order(grid: np.ndarray, process_top_to_bottom: bool) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    bg = get_bg(g)
    legend_c, entries = _find_legend(g, bg)
    assert entries, "legend must contain at least one (marker, fill) entry"

    legend_mask = _build_legend_mask((h, w), legend_c, entries)
    out = g.copy()
    marker_order = [m for _, m, _ in entries]
    index_order = range(len(entries) - 1, -1, -1) if process_top_to_bottom else range(len(entries))

    for i in index_order:
        _, marker, fill = entries[i]
        info = _marker_effect_data(g, legend_mask, marker)
        if info["kind"] == "none":
            continue

        protect = legend_mask.copy()
        future_markers = marker_order[:i] if process_top_to_bottom else marker_order[i + 1:]
        for fm in future_markers:
            protect |= (g == fm) & (~legend_mask)

        fill_mask = info["fill_mask"]
        if info["kind"] == "block":
            out[fill_mask & (~protect)] = int(fill)
            continue

        protect |= info["marker_mask"]
        if info["kind"] == "degenerate":
            continue

        out[fill_mask & (~protect)] = int(fill)

    return out


def _legacy_main(grid: np.ndarray) -> np.ndarray:
    return _solve_with_order(grid, process_top_to_bottom=False)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the regions implied by paired corner markers using a bottom-left legend.

    Input structure:
    - The lower-left corner contains a vertical 2-column legend whose rows encode (marker color, fill color).
    - Away from the legend, each marker color appears either as two opposite L-corners of a rectangle or as a solid 2x2 block.
    - Different marker rectangles may overlap in area, but their marker pixels and the legend itself stay separate.

    Rule:
    - Read the legend rows from bottom to top to recover the color pairings, but execute them in the reverse direction: top row first, bottom row last.
    - For each L-pair, detect which diagonal carries the L anchors and fill the opposite diagonal's rectangle with the legend fill color.
    - For a 2x2 block, simply recolor that block to its legend fill color.
    - While processing one entry, preserve the legend and the marker pixels of entries that will be handled later in this reversed execution order.

    Key insight: The visible twist is not where the legend is read from, but that overlap priority is flipped: upper legend rows are discovered later by reading, yet lower rows win because they are executed later.
    """
    return _solve_with_order(grid, process_top_to_bottom=True)


# --------------- generator (C1) ---------------

# ── CFG: canvas & grid ──────────────────────────────────────────────
GRID_H_RANGE = (10, 19)          # grid height bounds
GRID_W_RANGE = (10, 19)          # grid width bounds

# ── CFG: legend ─────────────────────────────────────────────────────
LEGEND_COL_RANGE = (0, 2)        # legend left column (marker at c, fill at c+1)
LEGEND_ENTRY_RANGE = (2, 5)      # number of legend entries

# ── CFG: marker types ──────────────────────────────────────────────
NOOP_PROB = 0.3                  # prob of one noop entry
BLOCK_PROB = 0.25                # prob of one 2x2 block entry

# ── CFG: L-pair bbox shape ──────────────────────────────────────────
BBOX_H_MAX = 19                  # max bbox height
BBOX_W_MAX = 12                  # max bbox width
BBOX_SHAPE_WEIGHTS = [0.75, 0.15, 0.10]  # [both>=3, h=2/w>=4, h>=4/w=2]
TLBR_PROB = 0.8                  # prob of TL-BR diagonal (vs TR-BL)
FOCUS_PAIR_MIN_SPAN = 8          # make forced-overlap pair large enough to be visible
FORCED_OVERLAP_RETRIES = 220     # retries for the guaranteed overlap pair
MIN_FORCED_OVERLAP_CELLS = 2     # overlap must contain at least this many fill cells

# ── CFG: distractor ────────────────────────────────────────────────
DISTRACTOR_PROB = 0.25           # prob of adding a non-legend L-pair

# ── CFG: retry budgets ─────────────────────────────────────────────
BLOCK_RETRIES = 100              # placement attempts for a 2x2 block
LPAIR_RETRIES = 150              # placement attempts for a standard L-pair
DISTRACTOR_RETRIES = 60          # placement attempts for distractor
GEN_RETRIES = 400                # outer generate_input retry budget

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


def _cells_to_mask(H: int, W: int, cells: list[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros((H, W), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _l_pair_fill_mask(H: int, W: int, rmin: int, cmin: int, rmax: int, cmax: int,
                      diag: str) -> np.ndarray:
    if diag == "TLBR":
        a, b = (rmin, cmax), (rmax, cmin)
    else:
        a, b = (rmin, cmin), (rmax, cmax)
    return _diamond_mask(H, W, rmin, rmax, cmin, cmax, a, b)


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


def _pick_focus_bbox_size(H: int, W: int) -> tuple[int, int] | None:
    for _ in range(25):
        size = _pick_bbox_size(H, W)
        if size is not None and size[0] + size[1] >= FOCUS_PAIR_MIN_SPAN:
            return size
    return _pick_bbox_size(H, W)


def _try_place_l_pair(H: int, W: int, color: int, retries: int,
                      can_place, do_place, accept=None, size_picker=None):
    """Try to place an L-pair of `color` on the grid within `retries` attempts."""
    size_picker = _pick_bbox_size if size_picker is None else size_picker
    for _ in range(retries):
        size = size_picker(H, W)
        if size is None:
            continue
        bh, bw = size
        rmin = random.randint(0, H - bh)
        cmin = random.randint(0, W - bw)
        rmax, cmax = rmin + bh - 1, cmin + bw - 1
        diag = "TLBR" if random.random() < TLBR_PROB else "TRBL"
        cells = _l_pair_cells(rmin, cmin, rmax, cmax, diag)
        if not can_place(cells):
            continue
        marker_mask = _cells_to_mask(H, W, cells)
        fill_mask = _l_pair_fill_mask(H, W, rmin, cmin, rmax, cmax, diag)
        layout = {
            "cells": cells,
            "marker_mask": marker_mask,
            "fill_mask": fill_mask,
            "bbox": (rmin, cmin, rmax, cmax),
            "diag": diag,
        }
        if accept is not None and not accept(layout):
            continue
        do_place(cells, color)
        return layout
    return None


def _build_legend(H, W, bg):
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
        if legend_c + 1 < W:
            grid[r, legend_c + 1] = fill
            legend_mask[r, legend_c + 1] = True

    return grid, legend_rows, legend_mask, legend_c, legend_bot, marker_colors


def _build_forbidden(H, W, legend_c, legend_bot, legend_rows):
    """Build the forbidden-placement mask that protects legend scan integrity."""
    forbidden = np.zeros((H, W), dtype=bool)
    if legend_bot + 1 < H:
        forbidden[legend_bot + 1:, :] = True
    forbidden[:legend_bot + 1, :legend_c + 1] = True
    for r, _, _ in legend_rows:
        forbidden[r, legend_c] = False
    return forbidden


def _force_distinct_focus_fills(grid: np.ndarray, legend_rows: list[tuple[int, int, int]],
                                legend_c: int) -> list[tuple[int, int, int]]:
    if len(legend_rows) < 2:
        return legend_rows
    bottom = legend_rows[0]
    top = legend_rows[-1]
    if bottom[2] != top[2]:
        return legend_rows

    choices = [c for c in range(10) if c != bottom[2]]
    new_fill = random.choice(choices)
    top_row = top[0]
    legend_rows = legend_rows.copy()
    legend_rows[-1] = (top_row, top[1], new_fill)
    grid[top_row, legend_c + 1] = new_fill
    return legend_rows


def _variant_metrics(grid: np.ndarray) -> dict[str, int]:
    new_out = main(grid)
    old_out = _legacy_main(grid)
    bg = get_bg(grid)
    legend_c, entries = _find_legend(grid, bg)
    legend_mask = _build_legend_mask(grid.shape, legend_c, entries)

    overlap_pairs = 0
    for i, (_, mi, _) in enumerate(entries):
        info_i = _marker_effect_data(grid, legend_mask, mi)
        if info_i["kind"] != "diamond":
            continue
        for _, mj, _ in entries[i + 1:]:
            info_j = _marker_effect_data(grid, legend_mask, mj)
            if info_j["kind"] != "diamond":
                continue
            overlap = info_i["fill_mask"] & info_j["fill_mask"]
            overlap &= ~(info_i["marker_mask"] | info_j["marker_mask"] | legend_mask)
            if overlap.any():
                overlap_pairs += 1

    return {
        "diff_cells": int(np.count_nonzero(new_out != old_out)),
        "overlap_pairs": overlap_pairs,
        "active_entries": int(sum(_marker_effect_data(grid, legend_mask, m)["kind"] != "none" for _, m, _ in entries)),
    }


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)
    bg = random.randint(0, 9)

    legend_result = _build_legend(H, W, bg)
    if legend_result is None:
        return None
    grid, legend_rows, legend_mask, legend_c, legend_bot, marker_colors = legend_result
    legend_rows = _force_distinct_focus_fills(grid, legend_rows, legend_c)

    forbidden = _build_forbidden(H, W, legend_c, legend_bot, legend_rows)
    occupied = legend_mask.copy()

    n_entries = len(legend_rows)
    focus_bottom = 0
    focus_top = n_entries - 1

    types = ["L"] * n_entries
    idx_pool = [i for i in range(n_entries) if i not in (focus_bottom, focus_top)]
    random.shuffle(idx_pool)
    if random.random() < NOOP_PROB and idx_pool:
        types[idx_pool.pop()] = "noop"
    if random.random() < BLOCK_PROB and idx_pool:
        types[idx_pool.pop()] = "block"

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

    focus_low = _try_place_l_pair(
        H, W, legend_rows[focus_bottom][1], FORCED_OVERLAP_RETRIES,
        can_place, do_place, size_picker=_pick_focus_bbox_size,
    )
    if focus_low is None:
        return None

    def accept_focus_high(layout) -> bool:
        overlap = focus_low["fill_mask"] & layout["fill_mask"]
        overlap &= ~(focus_low["marker_mask"] | layout["marker_mask"] | legend_mask)
        return int(overlap.sum()) >= MIN_FORCED_OVERLAP_CELLS

    focus_high = _try_place_l_pair(
        H, W, legend_rows[focus_top][1], FORCED_OVERLAP_RETRIES,
        can_place, do_place, accept=accept_focus_high, size_picker=_pick_focus_bbox_size,
    )
    if focus_high is None:
        return None

    for i, (_, m, _) in enumerate(legend_rows):
        if i in (focus_bottom, focus_top):
            continue
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
            if _try_place_l_pair(H, W, m, LPAIR_RETRIES, can_place, do_place) is None:
                return None

    if random.random() < DISTRACTOR_PROB:
        avail = [c for c in range(10) if c != bg and c not in marker_colors]
        if avail:
            _try_place_l_pair(H, W, random.choice(avail), DISTRACTOR_RETRIES, can_place, do_place)

    counts = Counter(grid.flatten().tolist())
    if counts.most_common(1)[0][0] != bg:
        return None

    try:
        metrics = _variant_metrics(grid)
    except Exception:
        return None

    if metrics["diff_cells"] <= 0 or metrics["overlap_pairs"] <= 0:
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
