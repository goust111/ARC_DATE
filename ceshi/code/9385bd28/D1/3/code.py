# Variant of 9385bd28
# Dimension: D1 — 匹配条件：标记对角线的配对方式 (match_cond)
# Change: diagonal_markers — 同色 L 形角标记位于外接矩形的对角，通过 L 形朝向判定哪条对角线是标记对角线、... → 行列对齐匹配

"""
Design analysis for the D1 variant
==================================

spec_variant:
  preserved:
    - grid size, background-by-majority, bottom-left vertical 2-column legend
    - legend processing order: bottom -> top
    - optional noop entries and optional 2x2 block entries
    - fill geometry: still paint the "diamond" / shortest-path set inside an axis-aligned bbox
    - future legend entries and legend cells stay protected from earlier fills
  changed_core:
    - normal markers are still two disjoint 3-cell L-components of one legend color
    - the solver no longer reads L orientation to decide which diagonal is the marker diagonal
    - instead, each L contributes only its elbow coordinate (row, col)
    - if elbows are e1=(r1,c1), e2=(r2,c2), the fill diagonal is the crossed pair
      a=(r1,c2), b=(r2,c1)
    - this is a row/column-cross rule: one marker supplies a row, the other supplies a column

behavior_trace:
  old_logic:
    - with elbows at (2,4) and (7,9), the original solver inspects which corner-oriented Ls exist
    - the fill diagonal depends on TL/BR vs TR/BL corner evidence
  new_logic:
    - with the same elbow coordinates, ignore corner evidence entirely
    - always fill the shortest-path diamond between (2,9) and (7,4)
  concrete_divergence:
    - generator deliberately places geometrically TL/BR elbows using TR/BL-shaped Ls
    - such a sample is invalid or misleading for the old corner-orientation rule
    - it is still perfectly valid for the new row/column-cross rule because only the elbows matter

complexity_and_visibility:
  - the normal case still uses exactly two same-color components, so the task is not simplified into
    arbitrary many-to-many matching
  - the filled area remains large enough to see because bbox sizes follow the original broad ranges
  - the generator forces orientation/geometry mismatch in normal samples, so the new rule is
    observable and does not collapse back to the original diagonal-orientation behavior

implicit_assumptions_review:
  - original code assumed all same-color normal pixels formed one diagonal-corner bbox; that is replaced
    by a stronger structural assumption: each normal entry must decompose into exactly two separate
    3-cell L-components
  - original `_l_anchor_present`-style corner tests are removed; they are no longer valid because an L's
    orientation is not semantic for pairing
  - current solver now assumes each 3-cell component has a unique 4-neighbor degree-2 elbow; generator
    guarantees this by using literal L-shapes only
  - generator also enforces a 1-cell 4-neighbor gap between placed objects so two Ls do not merge into
    one component and distractors stay visually clean
"""

import random
from collections import Counter

import numpy as np

from arc_utils import components_mask, get_bg


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
    Mask for all cells inside [r0..r1]x[c0..c1] that lie on some shortest 4-neigh path between a and b.
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


def _component_points(coords: np.ndarray) -> list[tuple[int, int]]:
    return [(int(r), int(c)) for r, c in coords]


def _block_mask_from_points(shape: tuple[int, int], pts: list[tuple[int, int]]) -> np.ndarray:
    m = np.zeros(shape, dtype=bool)
    for r, c in pts:
        m[r, c] = True
    return m


def _is_block_2x2(pts: list[tuple[int, int]]) -> bool:
    if len(pts) != 4:
        return False
    rs = sorted({r for r, _ in pts})
    cs = sorted({c for _, c in pts})
    if len(rs) != 2 or len(cs) != 2 or rs[1] - rs[0] != 1 or cs[1] - cs[0] != 1:
        return False
    return set(pts) == {(rs[0], cs[0]), (rs[0], cs[1]), (rs[1], cs[0]), (rs[1], cs[1])}


def _l_elbow(pts: list[tuple[int, int]]) -> tuple[int, int] | None:
    if len(pts) != 3:
        return None
    pt_set = set(pts)
    elbow = None
    for r, c in pts:
        deg = 0
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            if (r + dr, c + dc) in pt_set:
                deg += 1
        if deg == 2:
            if elbow is not None:
                return None
            elbow = (r, c)
        elif deg != 1:
            return None
    return elbow


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill color-mapped diamonds from same-color L-markers using row/column crossing.

    Input structure:
    - A background grid contains a vertical legend at bottom-left; each legend row is (marker_color, fill_color).
    - Away from the legend, a marker color either appears as two separate 3-cell L-shapes or as one solid 2x2 block.
    - The two L-shapes are same-color components; each has one elbow cell that acts as the marker's coordinate.

    Rule:
    - Read legend rows bottom-to-top and gather the non-legend pixels of that marker color.
    - If the color forms a 2x2 block, recolor that block to the legend fill color.
    - Otherwise extract the elbow of each L-shape; cross the row from one elbow with the column from the other, and vice versa.
    - Fill the shortest-path diamond between those crossed coordinates, while preserving legend cells, the current marker pixels, and later marker colors.

    Key insight: The L orientation is no longer the pairing signal; the important data are the two elbow rows and the two elbow columns, whose row/column crossing defines the filled diagonal.
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
        marker_mask = (g == marker) & (~legend_mask)
        comps = components_mask(marker_mask, connectivity=4)
        if not comps:
            continue

        future_markers = set(marker_order[i + 1 :])
        protect = legend_mask.copy()
        for fm in future_markers:
            protect |= (g == fm) & (~legend_mask)

        if len(comps) == 1:
            pts = _component_points(comps[0])
            assert _is_block_2x2(pts), "single-component marker must be a 2x2 block"
            block = _block_mask_from_points(g.shape, pts)
            out[block & (~protect)] = int(fill)
            continue

        assert len(comps) == 2, "normal marker must consist of exactly two L-components"
        current_mask = np.zeros_like(g, dtype=bool)
        elbows: list[tuple[int, int]] = []
        for comp in comps:
            pts = _component_points(comp)
            elbow = _l_elbow(pts)
            assert elbow is not None, "normal marker components must be 3-cell L-shapes"
            elbows.append(elbow)
            for r, c in pts:
                current_mask[r, c] = True
        protect |= current_mask

        (r1, c1), (r2, c2) = elbows
        assert r1 != r2 and c1 != c2, "L elbows must differ in both row and column"
        rmin, rmax = sorted((r1, r2))
        cmin, cmax = sorted((c1, c2))

        a = (r1, c2)
        b = (r2, c1)
        diamond = _diamond_mask(h, w, rmin, rmax, cmin, cmax, a, b)
        out[diamond & (~protect)] = int(fill)

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
BLOCK_PROB = 0.25

# ── CFG: normal L-pair bbox shape ─────────────────────────────────
BBOX_H_MAX = 19
BBOX_W_MAX = 12
BBOX_SHAPE_WEIGHTS = [0.75, 0.15, 0.10]  # [both>=3, h=2/w>=4, h>=4/w=2]
GEOM_DIAG_PROB = 0.5                     # elbow positions use TL-BR vs TR-BL equally often

# ── CFG: distractor ────────────────────────────────────────────────
DISTRACTOR_PROB = 0.25

# ── CFG: retry budgets ─────────────────────────────────────────────
BLOCK_RETRIES = 100
LPAIR_RETRIES = 180
DISTRACTOR_RETRIES = 60
GEN_RETRIES = 300

_L_OFFSETS = {
    "TL": [(0, 0), (0, 1), (1, 0)],
    "TR": [(0, 0), (0, -1), (1, 0)],
    "BL": [(0, 0), (0, 1), (-1, 0)],
    "BR": [(0, 0), (0, -1), (-1, 0)],
}


def _pick_bbox_size(H: int, W: int) -> tuple[int, int] | None:
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
    if bh > H or bw > W:
        return None
    return bh, bw


def _l_cells(r: int, c: int, orient: str) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for dr, dc in _L_OFFSETS[orient]]


def _mismatched_l_pair_cells(
    rmin: int, cmin: int, rmax: int, cmax: int, geom_diag: str
) -> list[tuple[int, int]]:
    """
    Build two Ls whose elbow positions lie on one geometric diagonal, but whose orientations are intentionally
    inconsistent with the original corner-based semantics.
    """
    if geom_diag == "TLBR":
        specs = [("TR", rmin, cmin), ("BL", rmax, cmax)]
    else:
        specs = [("TL", rmin, cmax), ("BR", rmax, cmin)]
    cells: list[tuple[int, int]] = []
    for orient, r, c in specs:
        cells.extend(_l_cells(r, c, orient))
    return cells


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
        grid[r, legend_c + 1] = fill
        legend_mask[r, legend_c + 1] = True

    return grid, legend_rows, legend_mask, legend_c, legend_bot, marker_colors


def _build_forbidden(H: int, W: int, legend_c: int, legend_bot: int, legend_rows):
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
    types = ["L"] * n_entries
    idx_pool = list(range(n_entries))
    random.shuffle(idx_pool)
    if random.random() < NOOP_PROB and len(idx_pool) > 1:
        types[idx_pool.pop()] = "noop"
    if random.random() < BLOCK_PROB and idx_pool:
        types[idx_pool.pop()] = "block"

    def can_place(cells: list[tuple[int, int]]) -> bool:
        cell_set = set(cells)
        if len(cell_set) != len(cells):
            return False
        for r, c in cell_set:
            if not (0 <= r < H and 0 <= c < W):
                return False
            if occupied[r, c] or forbidden[r, c]:
                return False
        for r, c in cell_set:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and occupied[nr, nc] and (nr, nc) not in cell_set:
                    return False
        return True

    def do_place(cells: list[tuple[int, int]], color: int) -> None:
        for r, c in cells:
            grid[r, c] = color
            occupied[r, c] = True

    def try_place_block(color: int) -> bool:
        for _ in range(BLOCK_RETRIES):
            r0 = random.randint(0, H - 2)
            c0 = random.randint(0, W - 2)
            cells = [(r0, c0), (r0, c0 + 1), (r0 + 1, c0), (r0 + 1, c0 + 1)]
            if can_place(cells):
                do_place(cells, color)
                return True
        return False

    def try_place_variant_pair(color: int, retries: int) -> bool:
        for _ in range(retries):
            size = _pick_bbox_size(H, W)
            if size is None:
                continue
            bh, bw = size
            if W - bw - 1 < 1:
                continue
            rmin = random.randint(0, H - bh)
            cmin = random.randint(1, W - bw - 1)
            rmax = rmin + bh - 1
            cmax = cmin + bw - 1
            geom_diag = "TLBR" if random.random() < GEOM_DIAG_PROB else "TRBL"
            cells = _mismatched_l_pair_cells(rmin, cmin, rmax, cmax, geom_diag)
            if can_place(cells):
                do_place(cells, color)
                return True
        return False

    for i, (_, m, _) in enumerate(legend_rows):
        kind = types[i]
        if kind == "noop":
            continue
        if kind == "block":
            if not try_place_block(m):
                return None
        else:
            if not try_place_variant_pair(m, LPAIR_RETRIES):
                return None

    if random.random() < DISTRACTOR_PROB:
        avail = [c for c in range(10) if c != bg and c not in marker_colors]
        if avail:
            try_place_variant_pair(random.choice(avail), DISTRACTOR_RETRIES)

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
