import numpy as np
from collections import Counter, deque

from arc_utils import DIR4 as _N4, DIR8 as _N8, cc4 as _cc4, cc8 as _cc8, get_bg as _bg_color
def _components_mask(mask: np.ndarray, conn8: bool) -> list[list[tuple[int, int]]]:
    return _cc8(mask) if conn8 else _cc4(mask)


def _touches_border(pts: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts)


def _seed_pixels(g: np.ndarray, bg: int) -> list[tuple[int, int]]:
    h, w = g.shape
    vals, cnts = np.unique(g, return_counts=True)
    seeds: list[tuple[int, int]] = []
    for v, cnt in zip(vals, cnts):
        v = int(v)
        if v == bg or int(cnt) != 1:
            continue
        r, c = map(int, np.argwhere(g == v)[0])
        if r == 0 or c == 0 or r == h - 1 or c == w - 1:
            seeds.append((r, c))
    return seeds


def _nonbg_component8_from(g: np.ndarray, start: tuple[int, int], bg: int) -> set[tuple[int, int]]:
    h, w = g.shape
    st = [start]
    vis = {start}
    while st:
        r, c = st.pop()
        for dr, dc in _N8:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in vis and int(g[nr, nc]) != bg:
                vis.add((nr, nc))
                st.append((nr, nc))
    return vis


def transform(grid: np.ndarray) -> np.ndarray:
    """
    Fill enclosed regions along a line seeded by unique border pixels.

    Input structure:
    - bg = most common color.
    - Colored lines/drawings on the grid.
    - Unique single-pixel colors on the grid border mark seed endpoints of each line.

    Rule:
    - Each border seed pixel connects (8-connected) to a line drawing.
    - Find closed bg cavities whose boundary touches that line.
    - Flood-fill each such cavity with the seed pixel's color.

    Key insight: Only cavities reachable through the seed's own line get filled; the seed's unique color acts as a per-line fill label.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _bg_color(g)

    out = g.copy()

    # Precompute background regions (4-connected). Exclude those containing background corners as "outside".
    bg_comps = _components_mask(g == bg, conn8=False)
    regions = []  # list of (set_pts, boundary_set, touches_border, size)
    for pts in bg_comps:
        region = set(pts)
        touches = _touches_border(pts, h, w)
        boundary = set()
        for r, c in region:
            for dr, dc in _N4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in region and int(g[nr, nc]) != bg:
                    boundary.add((nr, nc))
        regions.append((region, boundary, touches, len(region)))

    corners = {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}
    outside_idxs = set()
    for i, (reg, _, _, _) in enumerate(regions):
        if any(p in reg for p in corners if int(g[p]) == bg):
            outside_idxs.add(i)

    cavities = []
    for i, (reg, bnd, _, _) in enumerate(regions):
        if i in outside_idxs or not bnd:
            continue
        bcols = {int(g[p]) for p in bnd}
        cavities.append((reg, bnd, bcols))

    for sr, sc in _seed_pixels(g, bg):
        seed_col = int(g[sr, sc])

        neigh_cols = []
        for dr, dc in _N4:
            nr, nc = sr + dr, sc + dc
            if 0 <= nr < h and 0 <= nc < w:
                v = int(g[nr, nc])
                if v != bg:
                    neigh_cols.append(v)
        if not neigh_cols:
            continue
        line_col = Counter(neigh_cols).most_common(1)[0][0]

        # Trace the connected "drawing" reachable from the seed (8-connected, non-background).
        traced = _nonbg_component8_from(g, (sr, sc), bg)

        for cav, boundary, bcols in cavities:
            if len(bcols) != 1:
                continue
            boundary_in_traced = boundary & traced
            if not boundary_in_traced:
                continue

            # Require that the traced boundary is adjacent (8-neigh) to at least one pixel of the line color.
            # This prevents filling regions that are only connected through other structures, not the line itself.
            ok = False
            for r, c in boundary_in_traced:
                for dr, dc in _N8:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and int(g[nr, nc]) == line_col:
                        ok = True
                        break
                if ok:
                    break
            if not ok:
                continue

            for r, c in cav:
                if int(out[r, c]) == bg:
                    out[r, c] = seed_col

    return out
