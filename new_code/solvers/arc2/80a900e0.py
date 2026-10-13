import numpy as np

from arc_utils import DIR8, bbox_pts as _bbox, cc8 as _cc8, top_k_colors as _top_k_colors
def _components8_by_color(g: np.ndarray, colors: set[int]) -> list[tuple[int, list[tuple[int, int]]]]:
    return [(int(col), pts) for col in colors for pts in _cc8(g == col)]


_DIAMOND8_OFFSETS = {
    (0, 2),
    (1, 1),
    (1, 3),
    (2, 0),
    (2, 4),
    (3, 1),
    (3, 3),
    (4, 2),
}


def _is_diamond8(comp: list[tuple[int, int]]) -> bool:
    if len(comp) != 8:
        return False
    rs = [r for r, _ in comp]
    cs = [c for _, c in comp]
    r0, r1 = min(rs), max(rs)
    c0, c1 = min(cs), max(cs)
    if (r1 - r0) != 4 or (c1 - c0) != 4:
        return False
    norm = {(r - r0, c - c0) for r, c in comp}
    return norm == _DIAMOND8_OFFSETS


def _outer_line_info(pts: list[tuple[int, int]]):
    # Returns (slope_key, endpoints_sorted_by_r) or None if not diagonal collinear.
    if len(pts) < 2:
        return None
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    if all((r - c) == (rs[0] - cs[0]) for r, c in pts):
        # slope +1, canonical direction (1,1)
        endpoints = sorted(pts, key=lambda p: (p[0], p[1]))
        return ("rc_diff", endpoints)
    if all((r + c) == (rs[0] + cs[0]) for r, c in pts):
        # slope -1, canonical direction (1,-1)
        endpoints = sorted(pts, key=lambda p: (p[0], -p[1]))
        return ("rc_sum", endpoints)
    return None


def transform(grid: np.ndarray) -> np.ndarray:
    """
    Extend diagonal rays outward from outer-border segments adjacent to diamond shapes on a checkerboard.

    Input structure:
    - Checkerboard bg (two most common colors).
    - Hollow diamond outlines (8-pixel) in a non-bg color (inner border).
    - Short diagonal line segments in another non-bg color adjacent to each diamond edge (outer border).

    Rule:
    - For each outer-border segment, determine the outward direction (perpendicular to the segment, away from the diamond).
    - From each endpoint of the segment, cast a diagonal ray outward, painting over bg pixels with the segment's color until hitting a non-bg pixel or the grid edge.

    Key insight: Rays emit only from the two endpoints of each outer-border segment, not every pixel, and travel diagonally away from the diamond.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    # The checkerboard background uses the two most frequent colors (typically 0 and 1).
    bg_colors = set(_top_k_colors(g, 2))

    # Find diamond inner borders (size-8 diamond components) in non-background colors.
    non_bg = set(int(v) for v in np.unique(g)) - bg_colors
    comps = _components8_by_color(g, non_bg)
    diamonds: list[tuple[int, list[tuple[int, int]]]] = [(col, pts) for col, pts in comps if _is_diamond8(pts)]

    out = g.copy()

    # Process diamonds in stable order (top-left by bbox).
    def bbox_key(item):
        _, pts = item
        return _bbox(pts)

    for _, dpts in sorted(diamonds, key=bbox_key):
        dset = set(dpts)

        # Collect all adjacent non-bg (non checkerboard) pixels that are not part of the diamond.
        adj = []
        for r, c in dpts:
            for dr, dc in DIR8:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in dset:
                    v = int(g[rr, cc])
                    if v not in bg_colors:
                        adj.append((rr, cc))

        if not adj:
            continue

        # Build mask for these adjacent pixels and extract their 8-connected components.
        adj_mask = np.zeros((h, w), dtype=bool)
        for r, c in adj:
            adj_mask[r, c] = True
        outer_comps = _cc8(adj_mask)

        for oc in outer_comps:
            ocol = int(g[oc[0][0], oc[0][1]])
            info = _outer_line_info(oc)
            if info is None:
                continue
            slope_key, endpoints = info

            if slope_key == "rc_diff":
                # line direction along (1,1); candidate normals are (-1,1) and (1,-1)
                normals = [(-1, 1), (1, -1)]
            else:
                # line direction along (1,-1); candidate normals are (-1,-1) and (1,1)
                normals = [(-1, -1), (1, 1)]

            # Choose inward normal by best overlap with diamond when shifting outer by that normal.
            best_n = None
            best_score = -1
            for n in normals:
                sc = sum(((r + n[0], c + n[1]) in dset) for r, c in oc)
                if sc > best_score:
                    best_score = sc
                    best_n = n
            if best_n is None or best_score <= 0:
                continue
            outward = (-best_n[0], -best_n[1])

            # Endpoints (two extremes along the diagonal line).
            p0 = endpoints[0]
            p1 = endpoints[-1]
            for pr, pc in (p0, p1):
                rr, cc = pr + outward[0], pc + outward[1]
                while 0 <= rr < h and 0 <= cc < w:
                    if int(out[rr, cc]) in bg_colors:
                        out[rr, cc] = ocol
                    else:
                        break
                    rr += outward[0]
                    cc += outward[1]

    return out
