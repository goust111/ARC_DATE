# Combo variant of 9385bd28 (index 10)
# combo_spec: D1/3+D9/1+D14/0
# Name: 行列对齐匹配+图例编码方向+一对多标记

"""
Design analysis for combo D1/3 + D9/1 + D14/0
================================================

a) Concrete behavior and trace:
   - Original task:
     each legend row maps one marker color to one fill color; outside the legend, that marker color
     normally forms one diagonal L-pair, and the L corner orientations themselves tell the solver
     which diagonal is marked and which opposite diagonal to fill.
   - New combo:
     one legend row controls 2-3 separate pairs of the same marker color; each pair is no longer a
     diagonal-corner rectangle. Instead, the two same-color L markers are paired because their elbows
     share one axis:
       * row-pair: elbows at (r,c1) and (r,c2)
       * col-pair: elbows at (r1,c) and (r2,c)
     Their shared axis plus one-step orthogonal arms create a rectangle whose diagonal stripe is the
     filled output. The legend
     is widened to three columns and the order of colors encodes the fill diagonal:
       * [marker, bg, fill]  -> fill TL-BR
       * [fill, bg, marker]  -> fill TR-BL
     Example:
       two marker elbows on row 5 at cols 4 and 10, with one vertical arm upward and the other
       downward, define bbox rows 4..6 and cols 4..10. The old solver would fail because there are
       no diagonal corner
       markers to inspect. The new solver ignores corner orientation as a diagonal signal, matches the
       pair by shared row, reads the legend order, and paints either the TL-BR or TR-BL stripe of
       that 3x7 box.

b) Complexity calibration:
   - Every active legend entry has 2 or 3 pairs; there are no noop rows and no 2x2 block rows. This
     keeps D14 visible in every sample instead of as a rare branch.
   - Pair kinds are limited to row-pairs and col-pairs, and every generated rectangle is only 3 cells
     thick in its short axis. That reduces freedom enough to keep matching recoverable while still making D1 visibly different
     from the original diagonal-corner logic.
   - Each sample is forced to contain both row-based and column-based pairs, and if there are at
     least two legend rows, both legend direction codes are also forced to appear. That makes D1 and
     D9 observable from 2-4 examples.

c) Hidden-assumption review against the original main():
   - The original solver assumes all non-legend pixels of one marker color belong to one global bbox.
     That fails immediately for D14, because one color now owns several distant rectangles.
   - It also assumes L orientations directly encode the marked diagonal. That assumption is invalid
     under D9, because the diagonal choice now lives in the legend ordering, not in the marker shape.
   - The original code can handle arbitrary bbox heights and widths from one diagonal pair. The combo
     instead assumes each non-legend component is a literal 3-cell L, every color has an even number
     of components, and valid pair candidates are those sharing exactly one axis while defining a
     clean rectangle with no extra same-color pixels inside. The generator guarantees these assumptions.
   - Protection logic from the original solver still matters and remains: legend cells and later
     marker colors must be preserved from earlier fills.

d) Placement legality:
   - The generator keeps the legend in a clean three-column corridor near the bottom-left and forbids
     other objects from entering that scan region.
   - All marker cells stay in-bounds, non-overlapping, and 4-disconnected from unrelated objects.
   - For a given marker color, elbow rows and elbow columns are never reused across different intended
     pairs except for the within-pair shared axis. This prevents accidental alternative matchings.

e) Visual observability:
   - The same legend color visibly repeats across 2-3 separate locations, so one-to-many mapping is
     directly visible.
   - Row-pairs and col-pairs look different on the canvas, so the alignment-based pairing rule is not
     hidden behind color ids or ordering.
   - Direction is encoded by left/right legend order, which is a purely visible cue. The solver never
     depends on numeric color ids.

f) Visual cleanliness:
   - Same-color pair bboxes are separated by a margin, and a no-touch placement rule avoids marker
     components merging into larger blobs.
   - Fill colors are disjoint from marker colors and from the background, so legend rows remain easy
     to parse and recolored regions remain easy to attribute.

g) Change inferability:
   - Fill color is still read directly from the legend rather than computed arithmetically.
   - The only new abstraction is that the legend order carries the diagonal parameter. Because the
     same legend row applies to multiple repeated strips of one marker color, the direction code gets
     repeated support within a single training example.

9) New solver assumptions and generator guarantees:
   - Assumption: every non-legend marker component is a 3-cell L with one elbow, one vertical arm,
     and one horizontal arm. Guarantee: the generator places only those shapes.
   - Assumption: for one marker color, valid pairing is uniquely recoverable. Guarantee: the generator
     enforces unique elbow rows/cols per color and verifies that the recovered matching equals the
     intended one.
   - Assumption: legend decoding is unambiguous. Guarantee: fill colors never appear outside the
     legend, while marker colors always do, so the solver can infer which side of a legend row is the
     marker and which side is the fill.
"""

import random
from collections import Counter

import numpy as np

from arc_utils import get_bg


def _find_legend(g: np.ndarray, bg: int) -> tuple[int, list[tuple[int, int, int]]]:
    """
    Legend is a vertical stack of 3-column rows at bottom-left.
    Each row contains exactly two non-bg cells at columns c and c+2, with bg at c+1.
    Returns (legend_col, entries) where entries are ordered bottom->top as (row, left_color, right_color).
    """
    h, w = g.shape
    start_r = None
    start_c = None
    for r in range(h - 1, -1, -1):
        nz = np.flatnonzero(g[r, :] != bg)
        if nz.size:
            assert nz.size == 2, "legend row must contain exactly two non-bg cells"
            assert int(nz[1]) - int(nz[0]) == 2, "legend row must span exactly three columns"
            start_r = r
            start_c = int(nz[0])
            break
    assert start_r is not None and start_c is not None and start_c < w - 2, (
        "legend must exist at bottom-left with width 3"
    )

    entries: list[tuple[int, int, int]] = []
    c = start_c
    for r in range(start_r, -1, -1):
        nz = np.flatnonzero(g[r, :] != bg)
        if nz.size != 2 or int(nz[0]) != c or int(nz[1]) != c + 2 or int(g[r, c + 1]) != bg:
            break
        entries.append((r, int(g[r, c]), int(g[r, c + 2])))
    return c, entries


def _diagonal_path_mask(
    h: int,
    w: int,
    a: tuple[int, int],
    b: tuple[int, int],
) -> np.ndarray:
    """
    Mask for the rasterized diagonal stripe from a to b.
    """
    m = np.zeros((h, w), dtype=bool)
    ar, ac = a
    br, bc = b
    steps = max(abs(br - ar), abs(bc - ac))
    if steps == 0:
        m[ar, ac] = True
        return m

    prev = None
    for k in range(steps + 1):
        t = k / steps
        r = int(round(ar + (br - ar) * t))
        c = int(round(ac + (bc - ac) * t))
        m[r, c] = True
        if prev is not None and abs(prev[0] - r) == 1 and abs(prev[1] - c) == 1:
            m[prev[0], c] = True
        prev = (r, c)
    return m


def _components_from_points(pts: set[tuple[int, int]]) -> list[set[tuple[int, int]]]:
    """4-connected components for a small set of grid coordinates."""
    remaining = set(pts)
    comps: list[set[tuple[int, int]]] = []
    while remaining:
        start = remaining.pop()
        stack = [start]
        comp = {start}
        while stack:
            r, c = stack.pop()
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if (nr, nc) in remaining:
                    remaining.remove((nr, nc))
                    comp.add((nr, nc))
                    stack.append((nr, nc))
        comps.append(comp)
    return comps


def _classify_l_component(comp: set[tuple[int, int]]) -> dict | None:
    """Return elbow plus arm directions for a 3-cell L-component."""
    if len(comp) != 3:
        return None

    elbow = None
    elbow_neighbors = None
    for r, c in comp:
        neigh = []
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (nr, nc) in comp:
                neigh.append((nr, nc))
        if len(neigh) == 2:
            if elbow is not None:
                return None
            elbow = (r, c)
            elbow_neighbors = neigh
        elif len(neigh) != 1:
            return None
    if elbow is None or elbow_neighbors is None:
        return None

    vdir = None
    hdir = None
    er, ec = elbow
    for nr, nc in elbow_neighbors:
        dr = nr - er
        dc = nc - ec
        if dc == 0 and abs(dr) == 1:
            if vdir is not None:
                return None
            vdir = dr
        elif dr == 0 and abs(dc) == 1:
            if hdir is not None:
                return None
            hdir = dc
        else:
            return None
    if vdir is None or hdir is None:
        return None

    return {
        "cells": set(comp),
        "elbow": (int(er), int(ec)),
        "vdir": int(vdir),
        "hdir": int(hdir),
    }


def _decode_legend_entries(
    g: np.ndarray,
    legend_c: int,
    raw_entries: list[tuple[int, int, int]],
) -> tuple[np.ndarray, list[dict]]:
    """
    Decode legend rows into marker/fill/diag records.

    A marker color always appears outside the legend; a fill color never does.
    Direction comes from left/right order:
      marker-left  -> TLBR
      marker-right -> TRBL
    """
    h, w = g.shape
    legend_mask = np.zeros_like(g, dtype=bool)
    for r, _, _ in raw_entries:
        legend_mask[r, legend_c : legend_c + 3] = True

    outside_counts = Counter(map(int, g[~legend_mask].flatten()))
    entries: list[dict] = []
    for r, left, right in raw_entries:
        left_out = int(outside_counts.get(int(left), 0))
        right_out = int(outside_counts.get(int(right), 0))
        if left_out > 0 and right_out == 0:
            marker = int(left)
            fill = int(right)
            diag = "TLBR"
        elif right_out > 0 and left_out == 0:
            marker = int(right)
            fill = int(left)
            diag = "TRBL"
        else:
            raise AssertionError("legend row must have exactly one marker color visible outside legend")
        entries.append({"row": int(r), "marker": marker, "fill": fill, "diag": diag})

    return legend_mask, entries


def _candidate_pair(
    infos: list[dict],
    i: int,
    j: int,
    all_pts: set[tuple[int, int]],
) -> dict | None:
    """Build one valid shared-row/shared-column candidate pair, or return None."""
    a = infos[i]
    b = infos[j]
    (r1, c1) = a["elbow"]
    (r2, c2) = b["elbow"]
    pair_cells = a["cells"] | b["cells"]

    if r1 == r2 and c1 != c2:
        rmin = min(r1, r1 + a["vdir"], r2 + b["vdir"])
        rmax = max(r1, r1 + a["vdir"], r2 + b["vdir"])
        cmin, cmax = sorted((c1, c2))
        align = "row"
    elif c1 == c2 and r1 != r2:
        rmin, rmax = sorted((r1, r2))
        cmin = min(c1, c1 + a["hdir"], c2 + b["hdir"])
        cmax = max(c1, c1 + a["hdir"], c2 + b["hdir"])
        align = "col"
    else:
        return None

    inside_count = sum(1 for r, c in all_pts if rmin <= r <= rmax and cmin <= c <= cmax)
    pair_inside = sum(1 for r, c in pair_cells if rmin <= r <= rmax and cmin <= c <= cmax)
    if inside_count != pair_inside:
        return None

    return {
        "indices": (i, j),
        "bbox": (int(rmin), int(cmin), int(rmax), int(cmax)),
        "align": align,
        "area": int((rmax - rmin + 1) * (cmax - cmin + 1)),
    }


def _match_marker_pairs(marker_pts: set[tuple[int, int]]) -> list[dict]:
    """Split one marker color into L-components and recover a unique perfect matching."""
    if not marker_pts:
        return []

    comps = _components_from_points(marker_pts)
    infos: list[dict] = []
    for comp in comps:
        info = _classify_l_component(comp)
        assert info is not None, "each marker component must be a 3-cell L-shape"
        infos.append(info)

    n = len(infos)
    assert n >= 4 and n % 2 == 0, "each marker color must contain 2 or 3 L-pairs"

    adjacency: dict[int, list[dict]] = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            cand = _candidate_pair(infos, i, j, marker_pts)
            if cand is not None:
                adjacency[i].append(cand)
                adjacency[j].append(cand)

    matchings: list[list[dict]] = []

    def dfs(used: set[int], chosen: list[dict]) -> None:
        if len(used) == n:
            matchings.append(chosen.copy())
            return
        i = min(idx for idx in range(n) if idx not in used)
        for cand in adjacency[i]:
            a, b = cand["indices"]
            j = b if a == i else a
            if j in used:
                continue
            used.add(i)
            used.add(j)
            chosen.append(cand)
            dfs(used, chosen)
            chosen.pop()
            used.remove(j)
            used.remove(i)

    dfs(set(), [])
    assert matchings, "marker components must admit at least one valid matching"

    def score(matching: list[dict]) -> tuple[int, tuple[int, ...]]:
        areas = sorted(c["area"] for c in matching)
        return sum(areas), tuple(areas)

    scored = [(score(m), m) for m in matchings]
    best_score = min(s for s, _ in scored)
    best_matchings = [m for s, m in scored if s == best_score]
    assert len(best_matchings) == 1, "marker pairing must be uniquely recoverable"

    best = best_matchings[0]
    return sorted(best, key=lambda c: c["bbox"])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint repeated diagonal stripes by matching same-color L markers through shared rows or columns.

    Input structure:
    - The grid has one background color and a vertical three-column legend near the bottom-left.
    - Each legend row contains two non-background colors separated by one background cell; one is the marker color and the other is the fill color.
    - Outside the legend, every marker color appears as 2-3 separate pairs of 3-cell L-shapes, where each intended pair shares either one row or one column.

    Rule:
    - Read legend rows from bottom to top and infer, for each row, which color is the marker by checking which legend color reappears outside the legend.
    - Decode the fill diagonal from the legend order: marker-left means fill TL-BR, marker-right means fill TR-BL.
    - For each marker color, split the non-legend pixels into L-components, recover the unique same-color pairing by shared-row/shared-column alignment, and form one rectangle per pair.
    - Paint the legend-specified diagonal stripe across each recovered rectangle, while preserving the full legend strip and all current or later marker pixels.

    Key insight:
    - The L shapes no longer tell you the diagonal to fill. They only provide coordinates for axis-aligned pairing; the diagonal choice is encoded visibly in the legend's left-versus-right color order and reused across several same-colored pairs.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    bg = get_bg(g)
    legend_c, raw_entries = _find_legend(g, bg)
    assert raw_entries, "legend must contain at least one entry"
    legend_mask, entries = _decode_legend_entries(g, legend_c, raw_entries)

    out = g.copy()
    marker_order = [e["marker"] for e in entries]

    for i, entry in enumerate(entries):
        marker = int(entry["marker"])
        fill = int(entry["fill"])
        diag = entry["diag"]

        marker_pts = set(map(tuple, np.argwhere((g == marker) & (~legend_mask))))
        assert marker_pts, "every legend marker color must appear outside the legend"

        future_markers = set(marker_order[i + 1 :])
        protect = legend_mask.copy()
        for fm in future_markers:
            protect |= (g == fm) & (~legend_mask)
        protect |= (g == marker) & (~legend_mask)

        pair_infos = _match_marker_pairs(marker_pts)
        for pair in pair_infos:
            rmin, cmin, rmax, cmax = pair["bbox"]
            if diag == "TLBR":
                a, b = (rmin, cmin), (rmax, cmax)
            else:
                a, b = (rmin, cmax), (rmax, cmin)
            diamond = _diagonal_path_mask(h, w, a, b)
            out[diamond & (~protect)] = fill

    return out


# --------------- generator (combo) ---------------

# ── CFG: canvas & grid ──────────────────────────────────────────────
GRID_H_RANGE = (10, 19)
GRID_W_RANGE = (10, 19)

# ── CFG: legend ─────────────────────────────────────────────────────
LEGEND_COL_RANGE = (0, 2)
LEGEND_ENTRY_RANGE = (2, 3)

# ── CFG: per-entry multiplicity ────────────────────────────────────
PAIR_COUNT_CHOICES = (2, 3)
PAIR_COUNT_WEIGHTS = (0.7, 0.3)

# ── CFG: pair geometry ─────────────────────────────────────────────
PAIR_SPAN_MIN = 4
PAIR_SPAN_MAX = 9
PAIR_KIND_WEIGHTS = (0.5, 0.5)  # row, col
SAME_COLOR_BBOX_MARGIN = 1

# ── CFG: distractor ────────────────────────────────────────────────
DISTRACTOR_PROB = 0.2
DISTRACTOR_RETRIES = 60

# ── CFG: retry budgets ─────────────────────────────────────────────
COLOR_PAIR_RETRIES = 220
GEN_RETRIES = 500


def _boxes_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int], margin: int) -> bool:
    """True if two bboxes overlap or come within `margin` cells in both axes."""
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + margin < br0
        or br1 + margin < ar0
        or ac1 + margin < bc0
        or bc1 + margin < ac0
    )


def _build_legend(H: int, W: int, bg: int):
    """Create a 3-column legend with direction encoded by left/right order."""
    legend_c = random.randint(*LEGEND_COL_RANGE)
    legend_bot = random.choice([H - 1, H - 2])
    max_entries = min(LEGEND_ENTRY_RANGE[1], legend_bot + 1, 4)
    min_entries = min(LEGEND_ENTRY_RANGE[0], max_entries)
    if max_entries < LEGEND_ENTRY_RANGE[0] or legend_c + 2 >= W:
        return None
    n_entries = random.randint(min_entries, max_entries)

    pool = [c for c in range(10) if c != bg]
    if len(pool) < 2 * n_entries:
        return None
    marker_colors = random.sample(pool, n_entries)
    fill_pool = [c for c in pool if c not in marker_colors]
    fill_colors = random.sample(fill_pool, n_entries)

    diags = [random.choice(["TLBR", "TRBL"]) for _ in range(n_entries)]
    if n_entries >= 2 and len(set(diags)) == 1:
        diags[random.randrange(n_entries)] = "TRBL" if diags[0] == "TLBR" else "TLBR"

    grid = np.full((H, W), bg, dtype=int)
    legend_rows: list[dict] = []
    legend_mask = np.zeros((H, W), dtype=bool)
    for i in range(n_entries):
        r = legend_bot - i
        marker = marker_colors[i]
        fill = fill_colors[i]
        diag = diags[i]
        if diag == "TLBR":
            grid[r, legend_c] = marker
            grid[r, legend_c + 2] = fill
        else:
            grid[r, legend_c] = fill
            grid[r, legend_c + 2] = marker
        legend_mask[r, legend_c : legend_c + 3] = True
        legend_rows.append({"row": r, "marker": marker, "fill": fill, "diag": diag})

    return grid, legend_rows, legend_mask, legend_c, legend_bot, marker_colors


def _build_forbidden(
    H: int,
    W: int,
    legend_c: int,
    legend_bot: int,
    legend_rows: list[dict],
) -> np.ndarray:
    """Protect legend scan integrity by blocking placements below and left of the 3-col legend stack."""
    forbidden = np.zeros((H, W), dtype=bool)
    if legend_bot + 1 < H:
        forbidden[legend_bot + 1 :, :] = True
    forbidden[: legend_bot + 1, : legend_c + 3] = True
    for entry in legend_rows:
        forbidden[int(entry["row"]), :] = True
    return forbidden


def _choose_pair_counts(H: int, W: int, n_entries: int) -> list[int] | None:
    """Pick 2-3 pairs per legend entry while keeping the total density feasible."""
    area = H * W
    for _ in range(80):
        counts = list(random.choices(PAIR_COUNT_CHOICES, weights=PAIR_COUNT_WEIGHTS, k=n_entries))
        if all(c >= 2 for c in counts):
            marker_pixels = 6 * sum(counts)
            if marker_pixels + 3 * n_entries <= int(area * 0.42):
                return counts
    return None


def _l_cells_from_elbow(r: int, c: int, vdir: int, hdir: int) -> list[tuple[int, int]]:
    return [(r, c), (r + vdir, c), (r, c + hdir)]


def _row_pair_cells(
    r: int,
    c1: int,
    c2: int,
    left_vdir: int,
    right_vdir: int,
) -> tuple[list[tuple[int, int]], tuple[int, int, int, int]]:
    assert c1 < c2
    cells = _l_cells_from_elbow(r, c1, left_vdir, -1) + _l_cells_from_elbow(r, c2, right_vdir, 1)
    rmin = min(r, r + left_vdir, r + right_vdir)
    rmax = max(r, r + left_vdir, r + right_vdir)
    return cells, (rmin, c1, rmax, c2)


def _col_pair_cells(
    r1: int,
    r2: int,
    c: int,
    top_hdir: int,
    bottom_hdir: int,
) -> tuple[list[tuple[int, int]], tuple[int, int, int, int]]:
    assert r1 < r2
    cells = _l_cells_from_elbow(r1, c, -1, top_hdir) + _l_cells_from_elbow(r2, c, 1, bottom_hdir)
    cmin = min(c, c + top_hdir, c + bottom_hdir)
    cmax = max(c, c + top_hdir, c + bottom_hdir)
    return cells, (r1, cmin, r2, cmax)


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)
    bg = random.randint(0, 9)

    legend_result = _build_legend(H, W, bg)
    if legend_result is None:
        return None
    grid, legend_rows, legend_mask, legend_c, legend_bot, marker_colors = legend_result

    pair_counts = _choose_pair_counts(H, W, len(legend_rows))
    if pair_counts is None:
        return None

    forbidden = _build_forbidden(H, W, legend_c, legend_bot, legend_rows)
    occupied = legend_mask.copy()
    color_masks: dict[int, set[tuple[int, int]]] = {c: set() for c in marker_colors}
    intended_pairs: dict[int, list[tuple[tuple[int, int, int, int], str]]] = {c: [] for c in marker_colors}
    used_rows: dict[int, set[int]] = {c: set() for c in marker_colors}
    used_cols: dict[int, set[int]] = {c: set() for c in marker_colors}

    def can_place(cells: list[tuple[int, int]]) -> bool:
        cellset = set(cells)
        if len(cellset) != len(cells):
            return False
        for r, c in cellset:
            if not (0 <= r < H and 0 <= c < W):
                return False
            if occupied[r, c] or forbidden[r, c]:
                return False
        for r, c in cellset:
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < H and 0 <= nc < W and occupied[nr, nc] and (nr, nc) not in cellset:
                    return False
        return True

    def do_place(cells: list[tuple[int, int]], color: int) -> None:
        color_masks.setdefault(color, set())
        for r, c in cells:
            grid[r, c] = color
            occupied[r, c] = True
            color_masks[color].add((r, c))

    def try_place_pair(color: int, kind: str, retries: int) -> bool:
        for _ in range(retries):
            if kind == "row":
                width = random.randint(PAIR_SPAN_MIN, min(PAIR_SPAN_MAX, W - 2))
                if width < PAIR_SPAN_MIN or W - width - 1 < 1:
                    continue
                r_lo = 1
                r_hi = H - 2
                if r_lo > r_hi:
                    continue
                r = random.randint(r_lo, r_hi)
                c1 = random.randint(1, W - width - 1)
                c2 = c1 + width - 1
                if r in used_rows[color] or c1 in used_cols[color] or c2 in used_cols[color]:
                    continue
                if random.random() < 0.5:
                    left_vdir, right_vdir = -1, 1
                else:
                    left_vdir, right_vdir = 1, -1
                cells, bbox = _row_pair_cells(r, c1, c2, left_vdir, right_vdir)
            else:
                height = random.randint(PAIR_SPAN_MIN, min(PAIR_SPAN_MAX, H - 2))
                if height < PAIR_SPAN_MIN or H - height - 1 < 1:
                    continue
                c_lo = 1
                c_hi = W - 2
                if c_lo > c_hi:
                    continue
                c = random.randint(c_lo, c_hi)
                r1 = random.randint(1, H - height - 1)
                r2 = r1 + height - 1
                if c in used_cols[color] or r1 in used_rows[color] or r2 in used_rows[color]:
                    continue
                if random.random() < 0.5:
                    top_hdir, bottom_hdir = -1, 1
                else:
                    top_hdir, bottom_hdir = 1, -1
                cells, bbox = _col_pair_cells(r1, r2, c, top_hdir, bottom_hdir)

            if any(_boxes_too_close(bbox, prev_bbox, SAME_COLOR_BBOX_MARGIN) for prev_bbox, _ in intended_pairs[color]):
                continue
            if not can_place(cells):
                continue

            do_place(cells, color)
            intended_pairs[color].append((bbox, kind))
            if kind == "row":
                used_rows[color].add(r)
                used_cols[color].add(c1)
                used_cols[color].add(c2)
            else:
                used_cols[color].add(c)
                used_rows[color].add(r1)
                used_rows[color].add(r2)
            return True
        return False

    global_kinds: list[str] = []
    for idx, entry in enumerate(legend_rows):
        marker = int(entry["marker"])
        target_count = pair_counts[idx]
        forced_kinds: list[str] = []
        if idx == 0:
            forced_kinds = ["row", "col"]
        while len(forced_kinds) < target_count:
            forced_kinds.append(random.choices(["row", "col"], weights=PAIR_KIND_WEIGHTS, k=1)[0])

        for kind in forced_kinds[:target_count]:
            if not try_place_pair(marker, kind, COLOR_PAIR_RETRIES):
                return None
            global_kinds.append(kind)

        recovered = _match_marker_pairs(color_masks[marker])
        recovered_sig = sorted((pair["bbox"], pair["align"]) for pair in recovered)
        intended_sig = sorted(intended_pairs[marker])
        if recovered_sig != intended_sig:
            return None

    if "row" not in global_kinds or "col" not in global_kinds:
        return None

    if random.random() < DISTRACTOR_PROB:
        avail = [c for c in range(10) if c != bg and c not in marker_colors and all(c != e["fill"] for e in legend_rows)]
        if avail:
            distractor = random.choice(avail)
            for _ in range(DISTRACTOR_RETRIES):
                kind = random.choice(["row", "col"])
                if kind == "row":
                    width = random.randint(PAIR_SPAN_MIN, min(PAIR_SPAN_MAX, W - 2))
                    if W - width - 1 < 1:
                        continue
                    r_lo = 1
                    r_hi = H - 2
                    if r_lo > r_hi:
                        continue
                    r = random.randint(r_lo, r_hi)
                    c1 = random.randint(1, W - width - 1)
                    c2 = c1 + width - 1
                    if random.random() < 0.5:
                        left_vdir, right_vdir = -1, 1
                    else:
                        left_vdir, right_vdir = 1, -1
                    cells, _ = _row_pair_cells(r, c1, c2, left_vdir, right_vdir)
                else:
                    height = random.randint(PAIR_SPAN_MIN, min(PAIR_SPAN_MAX, H - 2))
                    if H - height - 1 < 1:
                        continue
                    c_lo = 1
                    c_hi = W - 2
                    if c_lo > c_hi:
                        continue
                    c = random.randint(c_lo, c_hi)
                    r1 = random.randint(1, H - height - 1)
                    r2 = r1 + height - 1
                    if random.random() < 0.5:
                        top_hdir, bottom_hdir = -1, 1
                    else:
                        top_hdir, bottom_hdir = 1, -1
                    cells, _ = _col_pair_cells(r1, r2, c, top_hdir, bottom_hdir)
                if can_place(cells):
                    do_place(cells, distractor)
                    break

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
