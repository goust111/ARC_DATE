import numpy as np

from arc_utils import DIR4
from arc_utils import bbox_pts as _bbox_pts, cc4 as _cc4, get_bg as _mode_color


def _hole_count(mask: np.ndarray) -> int:
    """
    Count holes in a binary mask using 4-connectivity.
    A hole is a background component that does not touch the mask's bounding box border.
    """
    h, w = mask.shape
    bg = ~mask
    vis = np.zeros((h, w), dtype=bool)
    holes = 0
    for r0 in range(h):
        for c0 in range(w):
            if not bg[r0, c0] or vis[r0, c0]:
                continue
            st = [(r0, c0)]
            vis[r0, c0] = True
            touches = False
            while st:
                r, c = st.pop()
                if r == 0 or c == 0 or r == h - 1 or c == w - 1:
                    touches = True
                for dr, dc in DIR4:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < h and 0 <= cc < w and bg[rr, cc] and not vis[rr, cc]:
                        vis[rr, cc] = True
                        st.append((rr, cc))
            if not touches:
                holes += 1
    return holes


def _legend_mapping(g: np.ndarray, bg: int, grey: int) -> dict[int, int]:
    """
    Pick one 'legend' component per non-(bg,grey) color:
      - component is at least 2x2 (filters divider lines),
      - choose the top-most (then left-most) component for that color.
    Map hole_count -> legend_color.
    """
    mapping: dict[int, int] = {}
    colors = [c for c in sorted(set(map(int, np.unique(g)))) if c not in (bg, grey)]
    for c in colors:
        comps = _cc4(g == c)
        candidates = []
        for pts in comps:
            r0, c0, r1, c1 = _bbox_pts(pts)
            if (r1 - r0 + 1) < 2 or (c1 - c0 + 1) < 2:
                continue
            candidates.append((r0, c0, r1, c1))
        if not candidates:
            continue
        candidates.sort()
        r0, c0, r1, c1 = candidates[0]
        sub = (g[r0 : r1 + 1, c0 : c1 + 1] == c)
        hc = _hole_count(sub)
        mapping[hc] = c
    return mapping


def transform(grid: np.ndarray) -> np.ndarray:
    """
    Recolor grey shapes by matching their hole count to a color legend.

    Input structure:
    - bg = most common color; grey (5) sprites scattered across the grid.
    - Colored legend shapes (non-bg, non-grey, at least 2×2) near grid edge, each with a distinct number of enclosed holes.

    Rule:
    - Each colored legend shape defines a mapping: hole count → color.
    - For every grey sprite, count its holes (enclosed bg regions not touching the sprite's bounding-box border).
    - Recolor the grey sprite to the legend color whose shape has the same hole count.

    Key insight: Shape identity is encoded by topology (number of holes), not geometry or size.
    """
    g = np.asarray(grid, dtype=int)
    bg = _mode_color(g)
    grey = 5

    legend = _legend_mapping(g, bg, grey)
    if not legend:
        return g.copy()

    out = g.copy()
    for pts in _cc4(g == grey):
        r0, c0, r1, c1 = _bbox_pts(pts)
        sub = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in pts:
            sub[r - r0, c - c0] = True
        hc = _hole_count(sub)
        newc = int(legend.get(hc, bg))
        for r, c in pts:
            out[r, c] = newc

    return out
