# Variant of 9385bd28
# Dimension: D14 — 标记对的基数关系 (multi_step)
# Change: 每个图例条目对应 1 个标记对（L 对或 2x2 方块），共 2-5 对... → 一个图例条目对应多个标记对

import random
from collections import Counter

import numpy as np

from arc_utils import get_bg


# Design analysis for D14 (source_task: None)
# a) Concrete behavior:
#    The legend stays as a 2-column vertical stack near the bottom-left. Each legend row still maps
#    one marker color to one fill color, but that marker color now appears in 2-3 separate L-pairs
#    across the grid. Each L-pair is still two disconnected 3-cell corner marks on one diagonal of a
#    rectangle, and the solver still fills the opposite diagonal's diamond. The behavioral change is
#    therefore not a new visual primitive; it is that one legend color now controls multiple distant
#    rectangles. Example: in the original task, all non-legend pixels of color 6 were pooled into one
#    bbox, one diagonal was detected, and one fill region was produced. In this variant, color 6 may
#    contain four or six disconnected L-corner components; the solver must first split them into
#    corner-components, then pair complementary corners into 2-3 rectangles, and fill each rectangle.
# b) Complexity calibration:
#    To make the D14 change visible in 2-4 examples, every active legend entry has multiple L-pairs.
#    I remove the original noop / 2x2-block branches, because they would let many samples collapse
#    back to "one entry, one object" behavior and visually hide the variant. Pair counts are limited
#    to 2 or 3 per color so the search space is larger than before but still small and learnable.
# c) Hidden assumptions in the original main():
#    The original code assumes all pixels of one marker color belong to one global bbox; that fails
#    immediately when the same color labels multiple rectangles. It also relies on a single diagonal
#    score comparison inside that bbox. In the variant, those lines would merge unrelated corners and
#    create one giant wrong fill. The replacement logic therefore works component-first: identify each
#    3-cell L-corner, enumerate compatible pairings, and choose the unique minimum-area perfect
#    matching. The "protect legend and later markers" logic remains valid and is kept.
# d) Placement legality:
#    The generator must guarantee that same-color corner components stay 4-disconnected, stay inside
#    bounds, and do not overlap the legend scan corridor. It also enforces same-color bbox separation
#    plus a post-check that the intended per-color pairing is exactly what the solver recovers.
# e) Visual observability:
#    The rule is visible without any hidden ordering by color id: one legend row repeatedly explains
#    several same-colored corner pairs that all turn into the same fill color. Different samples vary
#    which colors have 2 versus 3 pairs, so the "same marker color applies multiple times" pattern is
#    present in the examples rather than being a rare edge case.
# f) Visual cleanliness:
#    Same-color rectangles are kept spatially separated to prevent visually messy self-overlap.
#    Different colors may still interact through fill order, but marker cells themselves never overlap,
#    and the legend corridor remains clean so the task is readable.
# g) Color-change inferability:
#    Fill colors are still given explicitly by the adjacent legend cell; no color arithmetic or id-
#    based sorting is introduced. The only new inference is multiplicity: one legend mapping applies
#    to several repeated geometric instances of the same marker color.
# 9) New solver assumptions and generator guarantees:
#    The new matcher assumes every non-legend marker component is a 3-cell L-corner, the total number
#    of such components for one legend color is even, and the minimum-area perfect matching is unique.
#    The generator enforces all three by construction and rejects any sample whose recovered pairing
#    disagrees with the intended rectangles.


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


def _classify_l_component(comp: set[tuple[int, int]]) -> tuple[str, tuple[int, int]] | None:
    """Return (orientation, anchor) for a 3-cell L-corner component."""
    if len(comp) != 3:
        return None
    for r, c in comp:
        for name, offsets in _L_OFFSETS.items():
            cand = {(r + dr, c + dc) for dr, dc in offsets}
            if cand == comp:
                return name, (r, c)
    return None


def _candidate_pair(
    infos: list[dict],
    i: int,
    j: int,
    all_pts: set[tuple[int, int]],
) -> dict | None:
    """Build one compatible corner-pair candidate, or return None if the two components cannot pair."""
    oi = infos[i]["orient"]
    oj = infos[j]["orient"]
    ai = infos[i]["anchor"]
    aj = infos[j]["anchor"]

    if {oi, oj} == {"TL", "BR"}:
        tl = ai if oi == "TL" else aj
        br = aj if oj == "BR" else ai
        rmin, cmin = tl
        rmax, cmax = br
        diag = "TLBR"
        fill_ends = ((rmin, cmax), (rmax, cmin))
    elif {oi, oj} == {"TR", "BL"}:
        tr = ai if oi == "TR" else aj
        bl = aj if oj == "BL" else ai
        rmin, cmax = tr
        rmax, cmin = bl
        diag = "TRBL"
        fill_ends = ((rmin, cmin), (rmax, cmax))
    else:
        return None

    if rmin >= rmax or cmin >= cmax:
        return None
    bh = rmax - rmin + 1
    bw = cmax - cmin + 1
    if not _l_pair_valid_bbox(bh, bw):
        return None

    pair_cells = infos[i]["cells"] | infos[j]["cells"]
    if pair_cells != set(_l_pair_cells(rmin, cmin, rmax, cmax, diag)):
        return None

    inside_count = sum(1 for r, c in all_pts if rmin <= r <= rmax and cmin <= c <= cmax)
    if inside_count != len(pair_cells):
        return None

    return {
        "indices": (i, j),
        "bbox": (rmin, cmin, rmax, cmax),
        "diag": diag,
        "fill_ends": fill_ends,
        "area": bh * bw,
    }


def _match_marker_pairs(marker_pts: set[tuple[int, int]]) -> list[dict]:
    """Split one marker color into L-corner components and recover the unique minimum-area perfect matching."""
    if not marker_pts:
        return []

    comps = _components_from_points(marker_pts)
    infos: list[dict] = []
    for comp in comps:
        classified = _classify_l_component(comp)
        assert classified is not None, "each marker component must be a 3-cell L-corner"
        orient, anchor = classified
        infos.append({"cells": set(comp), "orient": orient, "anchor": anchor})

    n = len(infos)
    assert n % 2 == 0 and n >= 2, "each marker color must have an even positive number of L-corners"

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
    assert matchings, "marker components must admit at least one valid pairing"

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
    Use the bottom-left legend to recolor several repeated marker-pair instances.

    Input structure:
    - The grid has one background color and a vertical 2-column legend near the bottom-left.
    - Each legend row gives one marker color and its fill color, read from bottom to top.
    - Outside the legend, every legend marker color appears as 2-3 separate L-pairs, each pair made of two disconnected 3-cell corner marks.

    Rule:
    - For each legend entry, gather all non-legend pixels of that marker color and split them into 3-cell L-corner components.
    - Pair complementary corners into rectangles; for each recovered rectangle, the marked diagonal identifies the opposite diagonal to fill.
    - Fill the diamond region for every recovered rectangle with the legend's fill color, while preserving legend cells and all marker pixels for the current and later legend entries.
    - Earlier legend entries can still be overwritten later in non-protected cells, exactly as before.

    Key insight:
    - The legend mapping is reused multiple times per color: the hard part is not the fill itself, but discovering which same-colored L-corners belong together before applying one shared fill color to every matched rectangle.
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
        marker_pts = set(map(tuple, np.argwhere((g == marker) & (~legend_mask))))
        if not marker_pts:
            continue

        future_markers = set(marker_order[i + 1 :])
        protect = legend_mask.copy()
        for fm in future_markers:
            protect |= (g == fm) & (~legend_mask)
        protect |= (g == marker) & (~legend_mask)

        pair_infos = _match_marker_pairs(marker_pts)
        for pair in pair_infos:
            rmin, cmin, rmax, cmax = pair["bbox"]
            a, b = pair["fill_ends"]
            diamond = _diamond_mask(h, w, rmin, rmax, cmin, cmax, a, b)
            out[diamond & (~protect)] = int(fill)

    return out


# --------------- generator (D14) ---------------

# ── CFG: canvas & grid ──────────────────────────────────────────────
GRID_H_RANGE = (10, 19)
GRID_W_RANGE = (10, 19)

# ── CFG: legend ─────────────────────────────────────────────────────
LEGEND_COL_RANGE = (0, 2)
LEGEND_ENTRY_RANGE = (2, 4)

# ── CFG: per-entry multiplicity ────────────────────────────────────
PAIR_COUNT_CHOICES = (2, 3)
PAIR_COUNT_WEIGHTS = (0.65, 0.35)

# ── CFG: L-pair bbox shape ──────────────────────────────────────────
BBOX_H_MAX = 9
BBOX_W_MAX = 9
BBOX_SHAPE_WEIGHTS = [0.8, 0.1, 0.1]  # [both>=3, h=2/w>=4, h>=4/w=2]
TLBR_PROB = 0.8
SAME_COLOR_BBOX_MARGIN = 1

# ── CFG: distractor ────────────────────────────────────────────────
DISTRACTOR_PROB = 0.2

# ── CFG: retry budgets ─────────────────────────────────────────────
COLOR_PAIR_RETRIES = 160
DISTRACTOR_RETRIES = 60
GEN_RETRIES = 400


def _pick_bbox_size(H: int, W: int) -> tuple[int, int] | None:
    """Pick a random valid L-pair bbox size that fits in (H, W)."""
    shape_choice = random.choices(
        ["both_ge3", "h2_w_ge4", "h_ge4_w2"],
        weights=BBOX_SHAPE_WEIGHTS,
    )[0]
    if shape_choice == "both_ge3":
        h_hi = min(BBOX_H_MAX, H)
        w_hi = min(BBOX_W_MAX, W)
        if h_hi < 3 or w_hi < 3:
            return None
        bh = random.randint(3, h_hi)
        bw = random.randint(3, w_hi)
    elif shape_choice == "h2_w_ge4":
        w_hi = min(BBOX_W_MAX, W)
        if H < 2 or w_hi < 4:
            return None
        bh = 2
        bw = random.randint(4, w_hi)
    else:
        h_hi = min(BBOX_H_MAX, H)
        if h_hi < 4 or W < 2:
            return None
        bh = random.randint(4, h_hi)
        bw = 2
    if not _l_pair_valid_bbox(bh, bw):
        return None
    return bh, bw


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
    """Create legend entries and the corresponding grid/mask arrays."""
    legend_c = random.randint(*LEGEND_COL_RANGE)
    legend_bot = random.choice([H - 1, H - 2])
    max_entries = min(LEGEND_ENTRY_RANGE[1], legend_bot + 1, 9)
    min_entries = min(LEGEND_ENTRY_RANGE[0], max_entries)
    if max_entries < LEGEND_ENTRY_RANGE[0]:
        return None
    n_entries = random.randint(min_entries, max_entries)

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
        grid[r, legend_c + 1] = fill
        legend_mask[r, legend_c] = True
        legend_mask[r, legend_c + 1] = True

    return grid, legend_rows, legend_mask, legend_c, legend_bot, marker_colors


def _build_forbidden(H: int, W: int, legend_c: int, legend_bot: int, legend_rows: list[tuple[int, int, int]]) -> np.ndarray:
    """Protect legend scan integrity by blocking placements below or to the left of the legend stack."""
    forbidden = np.zeros((H, W), dtype=bool)
    if legend_bot + 1 < H:
        forbidden[legend_bot + 1 :, :] = True
    forbidden[: legend_bot + 1, : legend_c + 1] = True
    for r, _, _ in legend_rows:
        forbidden[r, legend_c] = False
    return forbidden


def _choose_pair_counts(H: int, W: int, n_entries: int) -> list[int] | None:
    """Pick 2-3 pairs per legend entry while keeping the total density feasible."""
    area = H * W
    for _ in range(80):
        counts = list(random.choices(PAIR_COUNT_CHOICES, weights=PAIR_COUNT_WEIGHTS, k=n_entries))
        if all(c >= 2 for c in counts):
            marker_pixels = 6 * sum(counts)
            if marker_pixels + 2 * n_entries <= int(area * 0.55):
                return counts
    return None


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

    def can_place(cells: list[tuple[int, int]], color: int) -> bool:
        cellset = set(cells)
        for r, c in cellset:
            if not (0 <= r < H and 0 <= c < W):
                return False
            if occupied[r, c] or forbidden[r, c]:
                return False
        for r, c in cellset:
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if (nr, nc) in color_masks[color] and (nr, nc) not in cellset:
                    return False
        return True

    def do_place(cells: list[tuple[int, int]], color: int) -> None:
        for r, c in cells:
            grid[r, c] = color
            occupied[r, c] = True
            color_masks[color].add((r, c))

    for idx, (_, marker, _) in enumerate(legend_rows):
        target_count = pair_counts[idx]
        placed = 0
        attempts = 0
        while placed < target_count and attempts < COLOR_PAIR_RETRIES:
            attempts += 1
            size = _pick_bbox_size(H, W)
            if size is None:
                continue
            bh, bw = size
            rmin = random.randint(0, H - bh)
            cmin = random.randint(0, W - bw)
            bbox = (rmin, cmin, rmin + bh - 1, cmin + bw - 1)
            if any(_boxes_too_close(bbox, prev_bbox, SAME_COLOR_BBOX_MARGIN) for prev_bbox, _ in intended_pairs[marker]):
                continue
            diag = "TLBR" if random.random() < TLBR_PROB else "TRBL"
            cells = _l_pair_cells(rmin, cmin, rmin + bh - 1, cmin + bw - 1, diag)
            if not can_place(cells, marker):
                continue
            do_place(cells, marker)
            intended_pairs[marker].append((bbox, diag))
            placed += 1
        if placed != target_count:
            return None

        recovered = _match_marker_pairs(color_masks[marker])
        recovered_sig = sorted((pair["bbox"], pair["diag"]) for pair in recovered)
        intended_sig = sorted(intended_pairs[marker])
        if recovered_sig != intended_sig:
            return None

    if random.random() < DISTRACTOR_PROB:
        avail = [c for c in range(10) if c != bg and c not in marker_colors]
        if avail:
            distractor = random.choice(avail)
            for _ in range(DISTRACTOR_RETRIES):
                size = _pick_bbox_size(H, W)
                if size is None:
                    continue
                bh, bw = size
                rmin = random.randint(0, H - bh)
                cmin = random.randint(0, W - bw)
                diag = "TLBR" if random.random() < TLBR_PROB else "TRBL"
                cells = _l_pair_cells(rmin, cmin, rmin + bh - 1, cmin + bw - 1, diag)
                cellset = set(cells)
                ok = True
                for r, c in cellset:
                    if occupied[r, c] or forbidden[r, c]:
                        ok = False
                        break
                if ok:
                    for r, c in cellset:
                        grid[r, c] = distractor
                        occupied[r, c] = True
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
    raise RuntimeError("generate_input: failed to generate a valid input after %d attempts" % GEN_RETRIES)
