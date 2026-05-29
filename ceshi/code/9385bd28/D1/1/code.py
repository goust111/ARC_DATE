# Variant of 9385bd28
# Dimension: D1 — 匹配条件：标记对角线的配对方式 (match_cond)
# Change: diagonal_markers — 同色 L 形角标记位于外接矩形的对角，通过 L 形朝向判定哪条对角线是标记对角线、... → 连接器匹配替代对角匹配
#
# Design analysis
# ---------------
# concrete_rule:
#   legend:
#     keep the original bottom-left vertical 2-column legend; each row is (marker_color, fill_color)
#     and rows are processed bottom->top exactly as before.
#   regular_object:
#     replace the old "two L corners whose orientation reveals the marked diagonal"
#     with a rectangle that has:
#       - one marker-colored corner pixel at each of the 4 bbox corners
#       - one 1-cell-wide same-color rectilinear connector that links exactly one diagonal pair
#     the connected diagonal is the marker diagonal; the opposite diagonal is the fill diagonal.
#   special_cases:
#     keep the original no-op legend entry option
#     keep the original isolated 2x2 block option, which is recolored directly with no diamond fill
#
# behavior_trace:
#   old:
#     collect all pixels of one marker color -> infer which diagonal is "marked" from the two L-corner
#     orientations -> fill the opposite diagonal diamond.
#   new:
#     collect all pixels of one marker color -> compute 4-neighbor connected components -> see which
#     pair of bbox corner pixels belongs to the same connected component because of the connector ->
#     fill the opposite diagonal diamond.
#   difference:
#     corner orientation is removed as a usable cue, so the diagonal must be recovered from connectivity.
#
# complexity_calibration:
#   four corner candidates make the connector visually necessary: there are two possible diagonals, and
#   the line selects one of them. This is stronger than merely adding a decorative line between the only
#   two markers, which would not really change the reasoning.
#   the connector is restricted to a monotone rectilinear polyline with 1 bend candidate axis chosen
#   from height or width capacity. That keeps generation robust and the visual rule readable.
#
# implicit_assumptions_checked:
#   old main() assumed:
#     - a single bbox over all pixels of one marker color was enough
#     - specific L-shaped local patterns were present at bbox corners
#     - those L patterns were the only source of diagonal identity
#   these assumptions break in the variant because:
#     - a marker color can now split into 3 components: one connected diagonal pair plus two isolated
#       corner pixels on the other diagonal
#     - bbox corners are plain single pixels rather than orientation-bearing Ls
#   therefore the new solver explicitly checks corner occupancy and component membership instead of
#   anchor orientation.
#
# placement_legality:
#   regular objects are generated only when bbox size is at least 2x2 and at least one dimension is >= 5;
#   this guarantees a connector route that reaches the chosen diagonal without accidentally touching the
#   other diagonal's corner pixels.
#   marker cells from different objects never overlap and never enter the legend-protected area.
#
# visual_observability:
#   the change is visible in 2-3 examples because each regular object shows four same-colored corner
#   candidates, but only one diagonal is tied together by a same-colored path. The output consistently
#   fills the opposite diagonal. No hidden ordering uses color ids.
#
# cleanliness:
#   only 0-1 distractor object is added, and it uses a non-legend color.
#   fills may overlap in the original task style, but marker pixels and legend cells stay legible.
#
# color_inference:
#   colors still come only from the legend mapping. The connector does not introduce a second mapping;
#   it only selects which diagonal is the marker diagonal for that color.
#
# new_solver_assumptions_and_generator_guarantees:
#   - each regular marker color has all 4 bbox corners occupied
#   - exactly one diagonal pair is 4-neighbor connected
#   - the other two corner pixels stay disconnected singletons
#   - isolated 2x2 blocks remain exactly 4 cells in a 2x2 bbox
#   generate_input() enforces all of these and validates every sample by running main().

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


def _components4(pts: set[tuple[int, int]]) -> list[set[tuple[int, int]]]:
    comps: list[set[tuple[int, int]]] = []
    unseen = set(pts)
    while unseen:
        start = unseen.pop()
        stack = [start]
        comp = {start}
        while stack:
            r, c = stack.pop()
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if (nr, nc) in unseen:
                    unseen.remove((nr, nc))
                    comp.add((nr, nc))
                    stack.append((nr, nc))
        comps.append(comp)
    return comps


def _corner_pixels_present(pts: set[tuple[int, int]], rmin: int, rmax: int, cmin: int, cmax: int) -> tuple[bool, bool, bool, bool]:
    tl = (rmin, cmin) in pts
    tr = (rmin, cmax) in pts
    bl = (rmax, cmin) in pts
    br = (rmax, cmax) in pts
    return tl, tr, bl, br


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill diamonds chosen by connector-linked corner markers using a color legend.

    Input structure:
    - A background grid with a bottom-left vertical legend, where each row stores (marker_color, fill_color).
    - Regular marker objects use one pixel at each corner of an axis-aligned bbox, and a same-color rectilinear
      connector links exactly one diagonal pair of corners.
    - Some marker colors still appear only as a solid 2x2 block instead of a connector-marked rectangle.

    Rule:
    - Read the legend bottom-to-top; for each entry gather all non-legend pixels of that marker color.
    - If they form a standalone 2x2 block, recolor that block to the legend fill color.
    - Otherwise, find which bbox corner diagonal is connected into one 4-neighbor component by the same-color
      connector, and fill the diamond on the opposite diagonal with the legend fill color.
    - Preserve legend cells and marker pixels of the current object and of later legend entries.

    Key insight: The marked diagonal is no longer encoded by corner orientation; it is the diagonal whose corner
    pixels are physically joined by the same-color connector path.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    bg = get_bg(g)
    legend_c, entries = _find_legend(g, bg)
    assert entries, "legend must contain at least one (marker, fill) entry"

    legend_mask = np.zeros_like(g, dtype=bool)
    for r, _, _ in entries:
        if 0 <= r < h and 0 <= legend_c < w:
            legend_mask[r, legend_c] = True
        if 0 <= r < h and 0 <= legend_c + 1 < w:
            legend_mask[r, legend_c + 1] = True

    out = g.copy()
    marker_order = [m for _, m, _ in entries]

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

        tl_ok, tr_ok, bl_ok, br_ok = _corner_pixels_present(marker_pts, rmin, rmax, cmin, cmax)
        assert tl_ok and tr_ok and bl_ok and br_ok, "regular marker object must occupy all bbox corners"

        comps = _components4(marker_pts)
        comp_id: dict[tuple[int, int], int] = {}
        for cid, comp in enumerate(comps):
            for pt in comp:
                comp_id[pt] = cid

        tl = (rmin, cmin)
        tr = (rmin, cmax)
        bl = (rmax, cmin)
        br = (rmax, cmax)
        tlbr_linked = comp_id[tl] == comp_id[br]
        trbl_linked = comp_id[tr] == comp_id[bl]
        assert tlbr_linked != trbl_linked, "exactly one diagonal must be connector-linked"

        if tlbr_linked:
            a, b = tr, bl
        else:
            a, b = tl, br

        diamond = _diamond_mask(h, w, rmin, rmax, cmin, cmax, a, b)
        out[diamond & (~protect)] = eff_fill

    return out


# --------------- generator (D1 variant) ---------------

# ── CFG: canvas & grid ──────────────────────────────────────────────
GRID_H_RANGE = (10, 19)
GRID_W_RANGE = (10, 19)

# ── CFG: legend ─────────────────────────────────────────────────────
LEGEND_COL_RANGE = (0, 2)
LEGEND_ENTRY_RANGE = (2, 5)

# ── CFG: marker types ──────────────────────────────────────────────
NOOP_PROB = 0.30
BLOCK_PROB = 0.22

# ── CFG: connector-rectangle bbox shape ────────────────────────────
BBOX_H_MAX = 19
BBOX_W_MAX = 12
BBOX_SHAPE_WEIGHTS = [0.65, 0.175, 0.175]  # [both_long, short_h, short_w]
TLBR_PROB = 0.5

# ── CFG: distractor ────────────────────────────────────────────────
DISTRACTOR_PROB = 0.25

# ── CFG: retry budgets ─────────────────────────────────────────────
BLOCK_RETRIES = 100
PAIR_RETRIES = 160
DISTRACTOR_RETRIES = 60
GEN_RETRIES = 300


def _pair_valid_bbox(h: int, w: int) -> bool:
    return h >= 2 and w >= 2 and (h >= 5 or w >= 5)


def _pick_bbox_size(H: int, W: int) -> tuple[int, int] | None:
    shape_choice = random.choices(
        ["both_long", "short_h", "short_w"],
        weights=BBOX_SHAPE_WEIGHTS,
    )[0]
    if shape_choice == "both_long":
        if min(H, W) < 5:
            return None
        bh = random.randint(5, min(BBOX_H_MAX, H))
        bw = random.randint(5, min(BBOX_W_MAX, W))
    elif shape_choice == "short_h":
        if H < 2 or W < 5:
            return None
        bh = random.randint(2, min(4, H))
        bw = random.randint(5, min(BBOX_W_MAX, W))
    else:
        if H < 5 or W < 2:
            return None
        bh = random.randint(5, min(BBOX_H_MAX, H))
        bw = random.randint(2, min(4, W))
    if not _pair_valid_bbox(bh, bw):
        return None
    return bh, bw


def _build_connector_cells(rmin: int, cmin: int, rmax: int, cmax: int, diag: str) -> list[tuple[int, int]]:
    pts: list[tuple[int, int]] = [
        (rmin, cmin),
        (rmin, cmax),
        (rmax, cmin),
        (rmax, cmax),
    ]
    h = rmax - rmin + 1
    w = cmax - cmin + 1

    styles: list[str] = []
    if h >= 5:
        styles.append("row_bridge")
    if w >= 5:
        styles.append("col_bridge")
    assert styles, "bbox must support at least one safe connector route"
    style = random.choice(styles)

    if diag == "TLBR":
        if style == "row_bridge":
            mid_r = random.randint(rmin + 2, rmax - 2)
            for r in range(rmin, mid_r + 1):
                pts.append((r, cmin))
            for c in range(cmin, cmax + 1):
                pts.append((mid_r, c))
            for r in range(mid_r, rmax + 1):
                pts.append((r, cmax))
        else:
            mid_c = random.randint(cmin + 2, cmax - 2)
            for c in range(cmin, mid_c + 1):
                pts.append((rmin, c))
            for r in range(rmin, rmax + 1):
                pts.append((r, mid_c))
            for c in range(mid_c, cmax + 1):
                pts.append((rmax, c))
    else:
        if style == "row_bridge":
            mid_r = random.randint(rmin + 2, rmax - 2)
            for r in range(rmin, mid_r + 1):
                pts.append((r, cmax))
            for c in range(cmax, cmin - 1, -1):
                pts.append((mid_r, c))
            for r in range(mid_r, rmax + 1):
                pts.append((r, cmin))
        else:
            mid_c = random.randint(cmin + 2, cmax - 2)
            for c in range(cmax, mid_c - 1, -1):
                pts.append((rmin, c))
            for r in range(rmin, rmax + 1):
                pts.append((r, mid_c))
            for c in range(mid_c, cmin - 1, -1):
                pts.append((rmax, c))

    return sorted(set(pts))


def _try_place_connector_rect(H: int, W: int, color: int, retries: int, can_place, do_place) -> bool:
    for _ in range(retries):
        size = _pick_bbox_size(H, W)
        if size is None:
            continue
        bh, bw = size
        rmin = random.randint(0, H - bh)
        cmin = random.randint(0, W - bw)
        diag = "TLBR" if random.random() < TLBR_PROB else "TRBL"
        cells = _build_connector_cells(rmin, cmin, rmin + bh - 1, cmin + bw - 1, diag)
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
        m = marker_colors[i]
        fill = random.randint(0, 9)
        legend_rows.append((r, m, fill))
        grid[r, legend_c] = m
        legend_mask[r, legend_c] = True
        if legend_c + 1 < W:
            grid[r, legend_c + 1] = fill
            legend_mask[r, legend_c + 1] = True

    return grid, legend_rows, legend_mask, legend_c, legend_bot, marker_colors


def _build_forbidden(H: int, W: int, legend_c: int, legend_bot: int, legend_rows: list[tuple[int, int, int]]) -> np.ndarray:
    forbidden = np.zeros((H, W), dtype=bool)
    if legend_bot + 1 < H:
        forbidden[legend_bot + 1 :, :] = True
    forbidden[: legend_bot + 1, : legend_c + 1] = True
    for r, _, _ in legend_rows:
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
    types = ["conn"] * n_entries
    idx_pool = list(range(n_entries))
    random.shuffle(idx_pool)
    if random.random() < NOOP_PROB and len(idx_pool) > 1:
        types[idx_pool.pop()] = "noop"
    if random.random() < BLOCK_PROB and idx_pool:
        block_idx = idx_pool.pop()
        types[block_idx] = "block"
    if all(t != "conn" for t in types):
        for i in range(n_entries):
            if types[i] != "noop":
                types[i] = "conn"
                break

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

    for i, (_, m, _) in enumerate(legend_rows):
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
            if not _try_place_connector_rect(H, W, m, PAIR_RETRIES, can_place, do_place):
                return None

    if random.random() < DISTRACTOR_PROB:
        avail = [c for c in range(10) if c != bg and c not in marker_colors]
        if avail:
            _try_place_connector_rect(H, W, random.choice(avail), DISTRACTOR_RETRIES, can_place, do_place)

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
    raise RuntimeError(f"generate_input: failed to generate a valid input after {GEN_RETRIES} attempts")
