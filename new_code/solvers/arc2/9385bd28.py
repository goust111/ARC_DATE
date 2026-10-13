import numpy as np

from arc_utils import DIR8 as DIRS8
from arc_utils import get_bg as _bg_color


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
    if start_r is None or start_c is None or start_c >= w - 1:
        return 0, []

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
    (dr1, dc1) = arm1
    (dr2, dc2) = arm2
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


def transform(grid: np.ndarray) -> np.ndarray:
    """
    Fill rectangles between diagonal marker pairs using a color legend.

    Input structure:
    - bg grid with a vertical legend at the bottom-left: each row is a (marker_color, fill_color) pair.
    - Scattered L-shaped corner markers of each marker color define opposite corners of axis-aligned rectangles.
    - Some markers are solid 2×2 blocks instead of corner pairs.

    Rule:
    - Read legend bottom-to-top; for each entry find all pixels of that marker color.
    - Determine which two bbox corners carry L-shaped anchors — the *other* diagonal defines the fill region.
    - Fill the diamond (all shortest-path cells between those two corners) with the legend's fill color.
    - Solid 2×2 marker blocks are simply recolored to the fill color.
    - Later legend entries are protected from earlier fills.

    Key insight: The L-shaped anchors sit on the marker diagonal; the fill goes on the opposite diagonal of the bounding box.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    bg = _bg_color(g)
    legend_c, entries = _find_legend(g, bg)
    if not entries:
        return g.copy()

    legend_mask = np.zeros_like(g, dtype=bool)
    for r, _, _ in entries:
        if 0 <= r < h and 0 <= legend_c < w:
            legend_mask[r, legend_c] = True
        if 0 <= r < h and 0 <= legend_c + 1 < w:
            legend_mask[r, legend_c + 1] = True

    out = g.copy()

    marker_order = [m for _, m, _ in entries]
    marker_set = set(marker_order)

    # Process legend entries bottom -> top; earlier entries must not destroy later marker pixels.
    for i, (_, marker, fill) in enumerate(entries):
        eff_fill = bg if int(fill) == 0 else int(fill)
        marker_pts = set(map(tuple, np.argwhere((g == marker) & (~legend_mask))))
        if len(marker_pts) < 2:
            continue

        rs = [r for r, _ in marker_pts]
        cs = [c for _, c in marker_pts]
        rmin, rmax = min(rs), max(rs)
        cmin, cmax = min(cs), max(cs)

        # Protect legend cells and all marker pixels for entries that appear later (higher) in the legend.
        future_markers = set(marker_order[i + 1 :])
        protect = legend_mask.copy()
        for fm in future_markers:
            protect |= (g == fm) & (~legend_mask)

        # Also protect current marker pixels in the common case (so markers remain visible),
        # except when they form a solid 2x2 block (treated as a filled marker that gets recolored).
        if rmax - rmin == 1 and cmax - cmin == 1 and len(marker_pts) == 4:
            # Solid 2x2 marker: recolor these pixels directly.
            block = np.zeros_like(g, dtype=bool)
            for r, c in marker_pts:
                block[r, c] = True
            out[block & (~protect)] = eff_fill
            continue
        else:
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
        if score_tlbr == 0 and score_trbl == 0:
            continue
        if score_tlbr > score_trbl:
            a, b = tr, bl
        elif score_trbl > score_tlbr:
            a, b = tl, br
        else:
            # Tie-breaker: fall back to raw corner membership.
            tl_on = tl in marker_pts
            tr_on = tr in marker_pts
            bl_on = bl in marker_pts
            br_on = br in marker_pts
            if tl_on and br_on and (not tr_on) and (not bl_on):
                a, b = tr, bl
            elif tr_on and bl_on and (not tl_on) and (not br_on):
                a, b = tl, br
            else:
                continue

        diamond = _diamond_mask(h, w, rmin, rmax, cmin, cmax, a, b)
        out[diamond & (~protect)] = eff_fill

    return out
