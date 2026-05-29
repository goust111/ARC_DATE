# Variant of 9385bd28
# Dimension: D5 — 填充变换方式 (transform)
# Change: rectangle_fill — 用图例对应的填充色填充标记对之间的菱形区域；2x2 方块直接重新上色... → 标记色与填充色互换

import random
from collections import Counter

import numpy as np

from arc_utils import get_bg

# Design analysis:
# spec:
#   retained_structure:
#     - The legend is still a 2-column vertical stack near the bottom-left.
#     - Each active marker color still appears either as one L-pair on a bbox diagonal or as one solid 2x2 block.
#     - The bbox, anchor detection, and fill-diamond geometry stay unchanged so D5 is the only conceptual change.
#   concretized_variant:
#     - For an L-pair entry, the output still targets the diamond on the diagonal opposite the marker anchors.
#     - New behavior is a color-role exchange:
#       1. every current marker pixel changes from marker_color to fill_color;
#       2. every pre-existing non-background pixel already inside the target diamond changes to marker_color;
#       3. remaining background cells in the target diamond become fill_color.
#     - A 2x2 block has no separate interior region, so it keeps the simpler "recolor block to fill_color" behavior.
#   small_trace:
#     - Original logic: marker L-corners stay marker-colored, diamond interior becomes fill-colored, embedded payloads would be overwritten.
#     - Variant logic: the same diamond is selected, but marker corners flip to fill_color while embedded non-bg payload cells flip back to marker_color.
#     - Difference: the transformation is no longer one-way painting; it is a visible two-way swap between marker-colored anchors and interior colored payload.
#   complexity_calibration:
#     - The solver still needs the original diagonal disambiguation, legend order, and protection of later markers.
#     - To make the swap visually observable, each generated L-pair receives 1-3 fill-colored payload cells inside its diamond.
#     - Fill colors are kept different from bg and from every legend marker color so payload pixels cannot be misread as another marker object.
#   implicit_assumptions_review:
#     - `_find_legend`, `_l_anchor_present`, and diagonal scoring remain valid because geometry is unchanged.
#     - The original `protect |= current marker` assumption no longer holds; current marker pixels must now be writable.
#     - The original unconditional `out[diamond] = fill` would destroy the new swap semantics, so the diamond must be split into
#       background cells that receive fill_color and existing non-bg cells that receive marker_color.
#     - Protection of legend cells and future marker pixels is still required so earlier entries do not erase later clues.
#   placement_legality:
#     - The generator reserves each active object's full bbox/footprint so different objects do not overlap or nest.
#     - Payload cells are chosen only from the computed target diamond, excluding marker cells and already occupied cells.
#     - The solver therefore sees unambiguous L-pairs, visible interior payload, and no accidental collisions with other entries.
#   visual_observability:
#     - Every non-noop L entry gets interior payload in the legend's fill color, so the output clearly shows those cells changing to the marker color.
#     - At least one L-pair is guaranteed, preventing degeneration into an all-block/noop sample set.
#   cleanliness:
#     - Overlapping bboxes are disallowed in generation to avoid clutter from multiple swaps in the same area.
#     - Distractors stay outside reserved regions so they remain harmless decoys rather than extra swap participants.
#   color_inference:
#     - The mapping is fully local and visible: the legend names marker_color and fill_color, the L-corners supply marker pixels,
#       and the interior payload supplies fill-colored pixels that flip back to marker_color.
#     - Because payload uses the legend fill_color itself, the human-visible clue for the color swap appears repeatedly within each grid.


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
    assert start_r is not None and start_c is not None and start_c < w - 1, \
        "legend must exist at bottom-left with at least 2 columns"

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


def _l_anchor_present(pts: set[tuple[int, int]], r: int, c: int, arm1: tuple[int, int], arm2: tuple[int, int]) -> bool:
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


def _diamond_mask(h: int, w: int, r0: int, r1: int, c0: int, c1: int, a: tuple[int, int], b: tuple[int, int]) -> np.ndarray:
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


def _fill_endpoints_from_marker_diag(
    rmin: int, cmin: int, rmax: int, cmax: int, marker_diag: str
) -> tuple[tuple[int, int], tuple[int, int]]:
    if marker_diag == "TLBR":
        return (rmin, cmax), (rmax, cmin)
    return (rmin, cmin), (rmax, cmax)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap marker color with fill color inside diagonal marker constructions using a color legend.

    Input structure:
    - A background grid contains a vertical bottom-left legend whose rows give (marker_color, fill_color).
    - Away from the legend, each marker color appears either as an L-shaped diagonal corner pair or as a solid 2x2 block.
    - Some L-pairs may already contain non-background payload pixels inside the diamond that would normally be filled.

    Rule:
    - Read legend entries bottom-to-top and locate all non-legend pixels of that marker color.
    - For an L-pair, use the anchored diagonal to infer the opposite diagonal diamond target.
    - Recolor every current marker pixel to the legend fill color; inside the target diamond, pre-existing non-background pixels become the marker color, while remaining background cells become the fill color.
    - A solid 2x2 block still just recolors to the fill color.
    - Legend pixels and markers belonging to later legend entries remain protected from earlier swaps.

    Key insight: The geometry is unchanged from the base task, but the target diamond now performs a two-way color-role exchange instead of a one-way fill.
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
        current_marker_mask = (g == marker) & (~legend_mask)
        marker_pts = set(map(tuple, np.argwhere(current_marker_mask)))
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
            out[current_marker_mask & (~protect)] = eff_fill
            continue

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
        assert score_tlbr != score_trbl, "diagonal scores must not tie (tie-break path is unreachable)"

        marker_diag = "TLBR" if score_tlbr > score_trbl else "TRBL"
        a, b = _fill_endpoints_from_marker_diag(rmin, cmin, rmax, cmax, marker_diag)
        diamond = _diamond_mask(h, w, rmin, rmax, cmin, cmax, a, b)

        swap_region = diamond & (~protect) & (~current_marker_mask)
        swap_existing = swap_region & (g != bg)
        swap_background = swap_region & (g == bg)

        out[current_marker_mask & (~protect)] = eff_fill
        out[swap_background] = eff_fill
        out[swap_existing] = marker

    return out


# --------------- generator (C1) ---------------

# ── CFG: canvas & grid ──────────────────────────────────────────────
GRID_H_RANGE = (10, 19)
GRID_W_RANGE = (10, 19)

# ── CFG: legend ─────────────────────────────────────────────────────
LEGEND_COL_RANGE = (0, 2)
LEGEND_ENTRY_RANGE = (2, 5)

# ── CFG: marker types ──────────────────────────────────────────────
NOOP_PROB = 0.3
BLOCK_PROB = 0.25

# ── CFG: L-pair bbox shape ──────────────────────────────────────────
BBOX_H_MAX = 19
BBOX_W_MAX = 12
BBOX_SHAPE_WEIGHTS = [0.75, 0.15, 0.10]  # [both>=3, h=2/w>=4, h>=4/w=2]
TLBR_PROB = 0.8

# ── CFG: payload for visible color swap ─────────────────────────────
PAYLOAD_MIN_CELLS = 1
PAYLOAD_MAX_CELLS = 3
PAYLOAD_RETRIES = 30

# ── CFG: distractor ────────────────────────────────────────────────
DISTRACTOR_PROB = 0.25

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


def _try_place_l_pair(H: int, W: int, color: int, retries: int, can_place, do_place):
    """Try to place an L-pair of `color` on the grid; return placement metadata on success."""
    for _ in range(retries):
        size = _pick_bbox_size(H, W)
        if size is None:
            continue
        bh, bw = size
        rmin = random.randint(0, H - bh)
        cmin = random.randint(0, W - bw)
        rmax = rmin + bh - 1
        cmax = cmin + bw - 1
        diag = "TLBR" if random.random() < TLBR_PROB else "TRBL"
        cells = _l_pair_cells(rmin, cmin, rmax, cmax, diag)
        if can_place(cells, rmin, cmin, rmax, cmax):
            do_place(cells, color, rmin, cmin, rmax, cmax)
            return {
                "rmin": rmin,
                "cmin": cmin,
                "rmax": rmax,
                "cmax": cmax,
                "diag": diag,
                "cells": cells,
            }
    return None


def _choose_payload_cells(candidates: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not candidates:
        return []
    cand_set = set(candidates)
    max_n = min(PAYLOAD_MAX_CELLS, len(candidates))
    min_n = min(PAYLOAD_MIN_CELLS, max_n)
    if max_n >= 2:
        min_n = max(min_n, 2)
    for _ in range(PAYLOAD_RETRIES):
        target = random.randint(min_n, max_n)
        start = random.choice(candidates)
        chosen = [start]
        chosen_set = {start}
        frontier = [start]
        while frontier and len(chosen) < target:
            base = random.choice(frontier)
            nbrs = []
            br, bc = base
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nxt = (br + dr, bc + dc)
                if nxt in cand_set and nxt not in chosen_set:
                    nbrs.append(nxt)
            if not nbrs:
                frontier.remove(base)
                continue
            nxt = random.choice(nbrs)
            chosen.append(nxt)
            chosen_set.add(nxt)
            frontier.append(nxt)
        if len(chosen) >= min_n:
            return chosen
    return [random.choice(candidates)]


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
    fill_palette = [c for c in range(10) if c != bg and c not in marker_colors]
    if not fill_palette:
        return None

    for i in range(n_entries):
        r = legend_bot - i
        m = marker_colors[i]
        fill = random.choice(fill_palette)
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
        forbidden[legend_bot + 1 :, :] = True
    forbidden[: legend_bot + 1, : legend_c + 1] = True
    for r, _, _ in legend_rows:
        forbidden[r, legend_c] = False
    return forbidden


def _diamond_payload_candidates(H: int, W: int, placement: dict) -> list[tuple[int, int]]:
    rmin = int(placement["rmin"])
    cmin = int(placement["cmin"])
    rmax = int(placement["rmax"])
    cmax = int(placement["cmax"])
    diag = str(placement["diag"])
    a, b = _fill_endpoints_from_marker_diag(rmin, cmin, rmax, cmax, diag)
    diamond = _diamond_mask(H, W, rmin, rmax, cmin, cmax, a, b)
    marker_cells = set(placement["cells"])
    coords = np.argwhere(diamond)
    return [(int(r), int(c)) for r, c in coords if (int(r), int(c)) not in marker_cells]


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
    reserved = forbidden.copy()

    n_entries = len(legend_rows)
    types = ["L"] * n_entries
    idx_pool = list(range(n_entries))
    random.shuffle(idx_pool)
    if random.random() < NOOP_PROB and len(idx_pool) > 1:
        types[idx_pool.pop()] = "noop"
    if random.random() < BLOCK_PROB and idx_pool:
        block_idx = idx_pool.pop()
        if sum(1 for t in types if t == "L") > 1:
            types[block_idx] = "block"
    if all(t != "L" for t in types):
        for i, t in enumerate(types):
            if t == "block":
                types[i] = "L"
                break

    def can_place(cells, r0, c0, r1, c1):
        if reserved[r0 : r1 + 1, c0 : c1 + 1].any():
            return False
        for r, c in cells:
            if not (0 <= r < H and 0 <= c < W):
                return False
            if occupied[r, c] or forbidden[r, c]:
                return False
        return True

    def do_place(cells, color, r0, c0, r1, c1):
        for r, c in cells:
            grid[r, c] = color
            occupied[r, c] = True
        reserved[r0 : r1 + 1, c0 : c1 + 1] = True

    active_l_pairs = 0
    payload_pixels = 0

    for i, (_, m, fill) in enumerate(legend_rows):
        t = types[i]
        if t == "noop":
            continue
        if t == "block":
            placed = False
            for _ in range(BLOCK_RETRIES):
                r0 = random.randint(0, H - 2)
                c0 = random.randint(0, W - 2)
                r1, c1 = r0 + 1, c0 + 1
                cells = [(r0, c0), (r0, c1), (r1, c0), (r1, c1)]
                if can_place(cells, r0, c0, r1, c1):
                    do_place(cells, m, r0, c0, r1, c1)
                    placed = True
                    break
            if not placed:
                return None
            continue

        placement = _try_place_l_pair(H, W, m, LPAIR_RETRIES, can_place, do_place)
        if placement is None:
            return None
        candidates = [cell for cell in _diamond_payload_candidates(H, W, placement) if not occupied[cell]]
        payload = _choose_payload_cells(candidates)
        if not payload:
            return None
        for r, c in payload:
            grid[r, c] = fill
            occupied[r, c] = True
        active_l_pairs += 1
        payload_pixels += len(payload)

    if active_l_pairs == 0 or payload_pixels == 0:
        return None

    if random.random() < DISTRACTOR_PROB:
        avail = [c for c in range(10) if c != bg and c not in marker_colors]
        if avail:
            _ = _try_place_l_pair(H, W, random.choice(avail), DISTRACTOR_RETRIES, can_place, do_place)

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
