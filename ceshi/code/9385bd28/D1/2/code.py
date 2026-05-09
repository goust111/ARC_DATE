# Variant of 9385bd28
# Dimension: D1 — 匹配条件：标记对角线的配对方式 (match_cond)
# Change: diagonal_markers — 同色 L 形角标记位于外接矩形的对角，通过 L 形朝向判定哪条对角线是标记对角线、... → 大小编码匹配

from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.ndimage import label

_ROOT = Path(__file__).resolve().parents[5]
_SOLVERS = _ROOT / "solvers"
if str(_SOLVERS) not in sys.path:
    sys.path.append(str(_SOLVERS))

from arc_utils import get_bg


# Design analysis for the D1 variant
# a) Concrete behavior:
#    - Keep the original legend format and legend-driven recoloring order.
#    - Replace each active same-color marker set with a 4-corner frame: all four bbox corners hold
#      inward-facing L markers of the same color, but exactly one corner is a larger L (arm length 3,
#      5 pixels) while the other three are smaller Ls (arm length 2, 3 pixels).
#    - The unique large corner selects the marker diagonal. If the large corner is on the TL/BR
#      diagonal, then that diagonal is the marker diagonal and the fill still goes on the opposite
#      diagonal; likewise for TR/BL.
#    - Concrete trace:
#      original: collect one color -> inspect which two corners contain 3-pixel L anchors -> fill the
#      opposite diagonal;
#      variant: collect one color -> split into connected components -> find the unique 5-pixel L ->
#      infer which diagonal that corner belongs to -> verify the opposite corner contains a 3-pixel L
#      of the same color -> fill the opposite diagonal.
#      The important difference is that orientation alone is now insufficient because all four corners
#      are valid L anchors; only size breaks the tie.
# b) Complexity calibration:
#    - Use only two marker sizes (small arm=2, large arm=3) so the numerical comparison is visible
#      and stable.
#    - Keep exactly one large corner per active rectangle. More size levels would make the task noisy;
#      no size variation would collapse back to the old diagonal cue.
# c) Implicit assumptions review against the original solver:
#    - Original code assumed all non-legend pixels of one marker color could be treated as a single
#      bbox with at most one valid anchor diagonal. That no longer holds because every corner now holds
#      an anchor candidate, so the old score-tie assertions would fail.
#    - Original code also protected all current-color pixels during filling; that still remains valid
#      and is necessary because the decoy small corners should stay visible.
#    - The old 2x2 block recolor special case remains harmless and is preserved as an orthogonal
#      marker subtype.
# d) Placement legality:
#    - A 4-corner frame with one large corner needs at least a 5x5 bbox so corner markers do not
#      overlap on the same row/column. The generator enforces that minimum.
#    - Markers, legend cells, and optional distractors are all placed on unoccupied cells only.
# e) Visual observability:
#    - The large-vs-small cue is directly visible in the input; no hidden ordering by color id or
#      position is used.
#    - Because all four corners exist, the large corner is the only reliable way to decide which
#      diagonal is "active", so the D1 change is visible in 2-3 examples.
# f) Visual cleanliness:
#    - Only one large corner is used per rectangle, and the other three corners are identical small
#      Ls. This keeps the object readable instead of mixing many unrelated component types.
#    - Bounding boxes are moderate in size and markers never overlap, so corners remain separable.
# g) Color-change inferability:
#    - The legend remains the sole source of fill colors. Marker colors stay distinct across legend
#      entries, and the shape/size cue is independent of the fill color itself.
# 9) New solver assumptions and generator guarantees:
#    - Active entries are either absent, a single 2x2 block, or a 4-component corner frame with
#      component sizes [3, 3, 3, 5]. The generator constructs only these cases.
#    - The unique large component always sits on one bbox corner, and the diagonally opposite corner
#      contains a small component of the same color. The generator enforces this explicitly.
#    - Future legend markers may lie inside earlier fill regions, so the solver still protects later
#      markers during bottom-to-top processing.


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
    """Cells inside [r0..r1]x[c0..c1] on a shortest 4-neigh path between corners a and b."""
    ar, ac = a
    br, bc = b
    dab = abs(ar - br) + abs(ac - bc)
    m = np.zeros((h, w), dtype=bool)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if abs(r - ar) + abs(c - ac) + abs(r - br) + abs(c - bc) == dab:
                m[r, c] = True
    return m


def _component_infos(mask: np.ndarray) -> list[dict[str, object]]:
    cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)
    labs, n = label(mask.astype(int), structure=cross)
    infos: list[dict[str, object]] = []
    for cid in range(1, int(n) + 1):
        pts = np.argwhere(labs == cid)
        if pts.size == 0:
            continue
        rows = pts[:, 0]
        cols = pts[:, 1]
        infos.append(
            {
                "size": int(len(pts)),
                "pts": {tuple(map(int, p)) for p in pts},
                "rmin": int(rows.min()),
                "rmax": int(rows.max()),
                "cmin": int(cols.min()),
                "cmax": int(cols.max()),
            }
        )
    return infos


def _component_corner(
    info: dict[str, object],
    rmin: int,
    rmax: int,
    cmin: int,
    cmax: int,
) -> str:
    ir0 = int(info["rmin"])
    ir1 = int(info["rmax"])
    ic0 = int(info["cmin"])
    ic1 = int(info["cmax"])
    matches: list[str] = []
    if ir0 == rmin and ic0 == cmin:
        matches.append("TL")
    if ir0 == rmin and ic1 == cmax:
        matches.append("TR")
    if ir1 == rmax and ic0 == cmin:
        matches.append("BL")
    if ir1 == rmax and ic1 == cmax:
        matches.append("BR")
    assert len(matches) == 1, "component must occupy exactly one bbox corner"
    return matches[0]


def _is_solid_block(info: dict[str, object]) -> bool:
    return (
        int(info["size"]) == 4
        and int(info["rmax"]) - int(info["rmin"]) == 1
        and int(info["cmax"]) - int(info["cmin"]) == 1
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill rectangles between size-coded corner markers using a color legend.

    Input structure:
    - A background grid contains a vertical 2-column legend at the bottom-left, with rows read
      bottom-to-top as (marker color, fill color).
    - Each active marker color appears either as a single solid 2x2 block or as four inward-facing
      L corners on one rectangle, where exactly one corner is a larger L than the other three.

    Rule:
    - Process legend rows bottom-to-top and collect non-legend components of the marker color.
    - A lone 2x2 block is recolored directly to the legend's fill color.
    - Otherwise, identify the unique large L corner inside the 4-corner frame; that oversized corner
      selects the marker diagonal, and the fill is drawn on the opposite diagonal of the bbox.
    - Keep legend cells, all current marker pixels, and all later marker colors protected from earlier fills.

    Key insight: All four corners are plausible anchors now, so diagonal selection is no longer encoded
    by L orientation alone; the single larger corner is the visible tie-breaker.
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

    out = g.copy()
    marker_order = [m for _, m, _ in entries]

    for i, (_, marker, fill) in enumerate(entries):
        marker_mask = (g == marker) & (~legend_mask)
        if not bool(marker_mask.any()):
            continue

        future_markers = set(marker_order[i + 1 :])
        protect = legend_mask.copy()
        for fm in future_markers:
            protect |= (g == fm) & (~legend_mask)

        comps = _component_infos(marker_mask)
        assert comps, "marker components must be non-empty"

        if len(comps) == 1 and _is_solid_block(comps[0]):
            block = np.zeros_like(g, dtype=bool)
            for r, c in comps[0]["pts"]:
                block[r, c] = True
            out[block & (~protect)] = int(fill)
            continue

        protect |= marker_mask

        pts = np.argwhere(marker_mask)
        rmin = int(pts[:, 0].min())
        rmax = int(pts[:, 0].max())
        cmin = int(pts[:, 1].min())
        cmax = int(pts[:, 1].max())
        assert rmax > rmin and cmax > cmin, "frame marker must span a non-degenerate bbox"

        by_corner: dict[str, dict[str, object]] = {}
        for info in comps:
            corner = _component_corner(info, rmin, rmax, cmin, cmax)
            by_corner[corner] = info

        assert set(by_corner.keys()) == {"TL", "TR", "BL", "BR"}, "frame must populate all four corners"
        sizes = sorted(int(info["size"]) for info in by_corner.values())
        assert sizes == [3, 3, 3, 5], "frame must contain three small Ls and one large L"

        large_corner = next(corner for corner, info in by_corner.items() if int(info["size"]) == 5)
        if large_corner in {"TL", "BR"}:
            assert int(by_corner["TL"]["size"]) != int(by_corner["BR"]["size"]), "active TL-BR pair must be mixed-size"
            a, b = (rmin, cmax), (rmax, cmin)
        else:
            assert int(by_corner["TR"]["size"]) != int(by_corner["BL"]["size"]), "active TR-BL pair must be mixed-size"
            a, b = (rmin, cmin), (rmax, cmax)

        diamond = _diamond_mask(h, w, rmin, rmax, cmin, cmax, a, b)
        out[diamond & (~protect)] = int(fill)

    return out


# --------------- generator ---------------

# ── CFG: canvas & grid ──────────────────────────────────────────────
GRID_H_RANGE = (10, 19)
GRID_W_RANGE = (10, 19)

# ── CFG: legend ─────────────────────────────────────────────────────
LEGEND_COL_RANGE = (0, 2)
LEGEND_ENTRY_RANGE = (2, 5)

# ── CFG: marker types ──────────────────────────────────────────────
NOOP_PROB = 0.25
BLOCK_PROB = 0.20

# ── CFG: size-coded frame bbox ─────────────────────────────────────
BBOX_H_MIN = 5
BBOX_W_MIN = 5
BBOX_H_MAX = 19
BBOX_W_MAX = 12
DIAG_PROB = 0.5  # TLBR vs TRBL

# ── CFG: distractor ────────────────────────────────────────────────
DISTRACTOR_PROB = 0.25

# ── CFG: retry budgets ─────────────────────────────────────────────
BLOCK_RETRIES = 100
FRAME_RETRIES = 160
DISTRACTOR_RETRIES = 60
GEN_RETRIES = 300


def _corner_l_cells(anchor_r: int, anchor_c: int, corner: str, arm_len: int) -> list[tuple[int, int]]:
    cells: set[tuple[int, int]] = {(anchor_r, anchor_c)}
    if corner == "TL":
        for k in range(1, arm_len):
            cells.add((anchor_r, anchor_c + k))
            cells.add((anchor_r + k, anchor_c))
    elif corner == "TR":
        for k in range(1, arm_len):
            cells.add((anchor_r, anchor_c - k))
            cells.add((anchor_r + k, anchor_c))
    elif corner == "BL":
        for k in range(1, arm_len):
            cells.add((anchor_r, anchor_c + k))
            cells.add((anchor_r - k, anchor_c))
    else:
        for k in range(1, arm_len):
            cells.add((anchor_r, anchor_c - k))
            cells.add((anchor_r - k, anchor_c))
    return list(cells)


def _frame_cells(rmin: int, cmin: int, rmax: int, cmax: int, diag: str) -> list[tuple[int, int]]:
    large_corner = "TL" if diag == "TLBR" else "TR"
    corner_specs = {
        "TL": ((rmin, cmin), 3 if large_corner == "TL" else 2),
        "TR": ((rmin, cmax), 3 if large_corner == "TR" else 2),
        "BL": ((rmax, cmin), 2),
        "BR": ((rmax, cmax), 2),
    }
    cells: set[tuple[int, int]] = set()
    for corner, ((r, c), arm_len) in corner_specs.items():
        cells.update(_corner_l_cells(r, c, corner, arm_len))
    return list(cells)


def _pick_bbox_size(H: int, W: int) -> tuple[int, int] | None:
    h_hi = min(BBOX_H_MAX, H)
    w_hi = min(BBOX_W_MAX, W)
    if h_hi < BBOX_H_MIN or w_hi < BBOX_W_MIN:
        return None
    bh = random.randint(BBOX_H_MIN, h_hi)
    bw = random.randint(BBOX_W_MIN, w_hi)
    return bh, bw


def _try_place_frame(H: int, W: int, color: int, retries: int, can_place, do_place) -> bool:
    for _ in range(retries):
        size = _pick_bbox_size(H, W)
        if size is None:
            continue
        bh, bw = size
        rmin = random.randint(0, H - bh)
        cmin = random.randint(0, W - bw)
        diag = "TLBR" if random.random() < DIAG_PROB else "TRBL"
        cells = _frame_cells(rmin, cmin, rmin + bh - 1, cmin + bw - 1, diag)
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
        fill = random.randint(0, 9)
        legend_rows.append((r, marker, fill))
        grid[r, legend_c] = marker
        grid[r, legend_c + 1] = fill
        legend_mask[r, legend_c] = True
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
    types = ["frame"] * n_entries
    idx_pool = list(range(n_entries))
    random.shuffle(idx_pool)
    if random.random() < NOOP_PROB and len(idx_pool) > 1:
        types[idx_pool.pop()] = "noop"
    if random.random() < BLOCK_PROB and idx_pool and types.count("frame") > 1:
        types[idx_pool.pop()] = "block"
    assert "frame" in types

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

    for i, (_, marker, _) in enumerate(legend_rows):
        kind = types[i]
        if kind == "noop":
            continue
        if kind == "block":
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
            if not _try_place_frame(H, W, marker, FRAME_RETRIES, can_place, do_place):
                return None

    if random.random() < DISTRACTOR_PROB:
        avail = [c for c in range(10) if c != bg and c not in marker_colors]
        if avail:
            _try_place_frame(H, W, random.choice(avail), DISTRACTOR_RETRIES, can_place, do_place)

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
