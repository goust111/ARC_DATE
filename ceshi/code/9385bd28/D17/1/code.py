# Variant of 9385bd28
# Dimension: D17 — 图例的阅读方向与编码方式 (symbol_instructions)
# Change: 图例为垂直堆叠的 (marker_color, fill_color) 对，左列标记色、右列填充色，从下到上阅读确定处理...
#         → 图例为单列颜色环：自底向上读取颜色序列，相邻颜色形成映射，顶部再回到底部完成闭环。
#
# Design analysis
# a) Concrete behavior:
#    - 输入左下角不再是两列表项，而是一列 3-5 个不同颜色，按自底向上记为 [c0, c1, ..., ck].
#    - 颜色映射不是显式给出，而是由相邻关系推断：c0->c1, c1->c2, ..., ck->c0.
#    - 其余对象保持原题：每种 marker color 在网格中对应一组 L 对角角标，或一个 2x2 实心块。
#    - 例子：若图例列自底向上是 [2, 4, 7]，则 2 填 4，4 填 7，7 填 2。最后一个颜色必须回环到最底色。
#    - 原题流程在 legend 解析处直接读出 (marker, fill)；本变体则先读单列顺序，再由顺序派生 fill。
# b) Complexity calibration:
#    - 为了让“环”而非“链”在视觉上可见，图例条目数改为至少 3；2 色环太像普通互换，辨识度不足。
#    - 顶部图例颜色必须有一个真实 L 对角对象，确保闭环映射 ck->c0 在样例里被真正用到。
#    - 图例右侧额外留 1 列空白缓冲，避免被误读成旧任务的双列表。
# c) Hidden assumptions from the original solver:
#    - 原 _find_legend 假设每行都有 marker/fill 两列，因此要求 start_c < w-1；这里必须改成单列连续扫描。
#    - 原 legend_mask 覆盖两列；这里必须只保护图例列本身，否则会把右侧正常背景缓冲误当图例。
#    - 主循环按 bottom->top 顺序处理的假设仍然成立，因为“顺序”现在用于决定相邻映射与遮挡保护。
#    - fill color 现在总是别的 marker color，但 solver 仍然安全，因为 marker 检测始终基于原始输入 g，而非 out。
# d) Placement legality:
#    - 生成器继续禁止 marker 像素重叠、越界，并保护图例下方与左侧区域，保证单列扫描唯一。
#    - 顶部图例项禁止生成 noop/block，只允许 L 对角对象，确保 wrap-around 有足够面积可见。
#    - 其余项仍可出现 0-1 个 noop、0-1 个 2x2 block，与原题难度接近。
# e) Visual observability:
#    - 人类可直接看到：图例是一列颜色序列，不再是两列配对。
#    - 通过 2-4 组样例可观察到“每种颜色填成图例里上一个邻居/下一个邻居”的闭环关系，尤其顶部颜色会填成底部颜色。
#    - 规则完全依赖可见的相邻顺序，不使用颜色 id 排序等不可见因素。
# f) Visual cleanliness:
#    - 图例旁保留空白缓冲，避免 marker 贴住图例造成“这是第二列吗”的误读。
#    - marker 仍只占少量像素，填充区域是规则的 bbox 菱形，不引入额外叠层对象。
# g) Color inferability:
#    - 图例颜色数控制在 3-5，且这些颜色在对象和结果中重复出现，足以支撑少量样例下的归纳。
#    - wrap-around 由顶端颜色对象触发，因此闭环关系不是代码上的隐含分支，而是训练样例里的可见现象。
#
# Post-change assumptions audited:
# - main() 假设图例是左下角唯一的连续单列非背景串；generate_input() 通过 forbidden mask 保证这一点。
# - main() 假设每个 legend color 在非图例区最多对应一个对象组；generate_input() 仍按每色单对象生成。
# - main() 假设顶部 legend color 存在可识别 L 对角锚点；generate_input() 强制顶部条目为 L 对角对象。

from __future__ import annotations

import random
import sys
from collections import Counter

import numpy as np

if "/home/lixinyu/snap/new/solvers" not in sys.path:
    sys.path.append("/home/lixinyu/snap/new/solvers")

from arc_utils import get_bg


def _find_legend(g: np.ndarray, bg: int) -> tuple[int, list[tuple[int, int, int]]]:
    """
    Legend is a vertical single-column color cycle at bottom-left.
    Reading bottom->top yields colors [c0, c1, ..., ck], interpreted as
    c0->c1, c1->c2, ..., ck->c0.
    Returns (legend_col, entries) ordered bottom->top as (row, marker, fill).
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
    assert start_r is not None and start_c is not None, "legend must exist at bottom-left"

    rows: list[int] = []
    colors: list[int] = []
    c = start_c
    for r in range(start_r, -1, -1):
        nz = np.flatnonzero(g[r, :] != bg)
        if nz.size == 0 or int(nz[0]) != c or int(g[r, c]) == bg:
            break
        rows.append(r)
        colors.append(int(g[r, c]))

    assert len(colors) >= 2, "legend cycle must contain at least two colors"
    entries = [(rows[i], colors[i], colors[(i + 1) % len(colors)]) for i in range(len(colors))]
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
    dr1, dc1 = arm1
    dr2, dc2 = arm2
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill rectangles between diagonal marker pairs using a cyclic color legend.

    Input structure:
    - A background grid contains a single vertical legend at the bottom-left, made of one color per row.
    - Reading that legend bottom-to-top gives a color sequence rather than explicit (marker, fill) pairs.
    - Elsewhere, each legend color appears either as one L-corner diagonal pair or as one solid 2x2 block.

    Rule:
    - Convert the legend sequence [c0, c1, ..., ck] into a cycle c0->c1, c1->c2, ..., ck->c0.
    - Process entries bottom-to-top; each marker color fills its own rectangle or block using the next color in the cycle.
    - For L-corner markers, the anchored diagonal identifies the opposite diagonal whose shortest-path diamond gets filled.
    - Preserve legend pixels and all marker pixels belonging to entries that have not been processed yet.

    Key insight: The legend is no longer a table of explicit pairs; adjacency in the bottom-left color column defines the recoloring cycle, including the top-to-bottom wrap.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    bg = get_bg(g)
    legend_c, entries = _find_legend(g, bg)
    assert entries, "legend must contain at least one cycle entry"

    legend_mask = np.zeros_like(g, dtype=bool)
    for r, _, _ in entries:
        if 0 <= r < h and 0 <= legend_c < w:
            legend_mask[r, legend_c] = True

    out = g.copy()
    marker_order = [marker for _, marker, _ in entries]

    for i, (_, marker, fill) in enumerate(entries):
        eff_fill = int(fill)
        marker_pts = set(map(tuple, np.argwhere((g == marker) & (~legend_mask))))
        if len(marker_pts) < 2:
            continue

        rs = [r for r, _ in marker_pts]
        cs = [c for _, c in marker_pts]
        rmin, rmax = min(rs), max(rs)
        cmin, cmax = min(cs), max(cs)

        future_markers = set(marker_order[i + 1 :])
        protect = legend_mask.copy()
        for fm in future_markers:
            protect |= (g == fm) & (~legend_mask)

        if rmax - rmin == 1 and cmax - cmin == 1 and len(marker_pts) == 4:
            block = np.zeros_like(g, dtype=bool)
            for r, c in marker_pts:
                block[r, c] = True
            out[block & (~protect)] = eff_fill
            continue

        protect |= (g == marker) & (~legend_mask)
        if rmin == rmax or cmin == cmax:
            continue

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
        assert score_tlbr != score_trbl, "diagonal scores must not tie"

        if score_tlbr > score_trbl:
            a, b = tr, bl
        else:
            a, b = tl, br

        diamond = _diamond_mask(h, w, rmin, rmax, cmin, cmax, a, b)
        out[diamond & (~protect)] = eff_fill

    return out


# --------------- generator (C1) ---------------

# -- CFG: canvas & grid --------------------------------------------------------
GRID_H_RANGE = (10, 19)
GRID_W_RANGE = (10, 19)

# -- CFG: legend ---------------------------------------------------------------
LEGEND_COL_RANGE = (0, 2)
LEGEND_ENTRY_RANGE = (3, 5)
LEGEND_BUFFER = 1

# -- CFG: marker types ---------------------------------------------------------
NOOP_PROB = 0.25
BLOCK_PROB = 0.25

# -- CFG: L-pair bbox shape ----------------------------------------------------
BBOX_H_MAX = 19
BBOX_W_MAX = 12
BBOX_SHAPE_WEIGHTS = [0.75, 0.15, 0.10]
TLBR_PROB = 0.8

# -- CFG: distractor -----------------------------------------------------------
DISTRACTOR_PROB = 0.25

# -- CFG: retry budgets --------------------------------------------------------
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


def _try_place_l_pair(H: int, W: int, color: int, retries: int, can_place, do_place) -> bool:
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
    """Create a bottom-left single-column color cycle."""
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
    legend_rows: list[tuple[int, int]] = []
    legend_mask = np.zeros((H, W), dtype=bool)
    for i, color in enumerate(marker_colors):
        r = legend_bot - i
        legend_rows.append((r, color))
        grid[r, legend_c] = color
        legend_mask[r, legend_c] = True

    return grid, legend_rows, legend_mask, legend_c, legend_bot, marker_colors


def _build_forbidden(H: int, W: int, legend_c: int, legend_bot: int, legend_rows: list[tuple[int, int]]) -> np.ndarray:
    """Protect the legend scan path and keep the single-column legend visually isolated."""
    forbidden = np.zeros((H, W), dtype=bool)
    if legend_bot + 1 < H:
        forbidden[legend_bot + 1 :, :] = True

    keep_clear_upto = min(W, legend_c + 1 + LEGEND_BUFFER)
    forbidden[: legend_bot + 1, :keep_clear_upto] = True
    for r, _ in legend_rows:
        forbidden[r, legend_c] = False
    return forbidden


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)
    bg = random.randint(0, 9)

    legend_result = _build_legend(H, W, bg)
    if legend_result is None:
        return None
    grid, legend_rows, legend_mask, legend_c, legend_bot, marker_colors = legend_result

    forbidden = _build_forbidden(H, W, legend_c, legend_bot, legend_rows)
    occupied = legend_mask.copy()

    n_entries = len(legend_rows)
    top_idx = n_entries - 1
    types = ["L"] * n_entries
    special_pool = list(range(n_entries - 1))
    random.shuffle(special_pool)

    if random.random() < NOOP_PROB and len(special_pool) > 1:
        types[special_pool.pop()] = "noop"
    if random.random() < BLOCK_PROB and special_pool:
        types[special_pool.pop()] = "block"

    assert types[top_idx] == "L"

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

    for i, (_r, marker) in enumerate(legend_rows):
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
                    do_place(cells, marker)
                    placed = True
                    break
            if not placed:
                return None
        else:
            if not _try_place_l_pair(H, W, marker, LPAIR_RETRIES, can_place, do_place):
                return None

    if random.random() < DISTRACTOR_PROB:
        avail = [c for c in range(10) if c != bg and c not in marker_colors]
        if avail:
            _try_place_l_pair(H, W, random.choice(avail), DISTRACTOR_RETRIES, can_place, do_place)

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
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
