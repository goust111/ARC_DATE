import numpy as np

from arc_utils import DIR4 as _N4, bbox_pts as _bbox, cc4 as _components_mask, get_bg as _bg_color


def _legend_directions(g: np.ndarray, bg: int) -> dict[int, tuple[int, int]]:
    """
    Detect legend blocks on the border.

    A legend is a border-touching non-background component whose most common color is the legend
    background `C` (and also the region border color it controls). The legend contains at least
    two additional colors: a "center" color (more frequent) and an "indicator" color (less frequent).

    The movement direction for regions with border color `C` is:
      sign(centroid(indicator) - centroid(center))
    """
    h, w = g.shape
    non_bg = g != bg
    best: dict[int, tuple[int, tuple[int, int]]] = {}
    for pts in _components_mask(non_bg):
        if not any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue
        # Legends are filled rectangles on the border.
        r0, c0, r1, c1 = _bbox(pts)
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        if area != len(pts):
            continue

        counts: dict[int, int] = {}
        for r, c in pts:
            v = int(g[r, c])
            counts[v] = counts.get(v, 0) + 1
        if len(counts) != 3:
            continue

        bg_col = max(counts, key=counts.get)
        others = [(cnt, col) for col, cnt in counts.items() if col != bg_col]
        if len(others) < 2:
            continue
        others.sort()  # ascending by count
        if others[0][0] == others[1][0]:
            # Tie: choose center as the color closer to the rectangle center.
            rc = np.array([(r0 + r1) / 2.0, (c0 + c1) / 2.0])
            cols = [others[0][1], others[1][1]]
            cents = []
            for col in cols:
                pts_col = np.array([(r, c) for r, c in pts if int(g[r, c]) == col], dtype=float)
                cents.append((np.sum((pts_col.mean(axis=0) - rc) ** 2), col))
            cents.sort()
            center_col = cents[0][1]
            indicator_col = cents[1][1]
        else:
            indicator_col = others[0][1]
            center_col = others[-1][1]

        center = [(r, c) for r, c in pts if int(g[r, c]) == center_col]
        indic = [(r, c) for r, c in pts if int(g[r, c]) == indicator_col]
        if not center or not indic:
            continue

        cc = np.mean(np.array(center, dtype=float), axis=0)
        ci = np.mean(np.array(indic, dtype=float), axis=0)
        vr, vc = (ci - cc).tolist()
        dr = 0 if abs(vr) < 1e-6 else (1 if vr > 0 else -1)
        dc = 0 if abs(vc) < 1e-6 else (1 if vc > 0 else -1)
        if dr == 0 and dc == 0:
            continue

        prev = best.get(bg_col)
        if prev is None or len(pts) > prev[0]:
            best[bg_col] = (len(pts), (dr, dc))

    return {k: v for k, (_, v) in best.items()}


def _slide_points(region_set: set[tuple[int, int]], points: list[tuple[int, int]], dr: int, dc: int) -> list[tuple[int, int]]:
    if not points:
        return []
    # Process from the far end (closest to destination boundary) inward.
    points_sorted = sorted(points, key=lambda p: (dr * p[0] + dc * p[1], p[0], p[1]), reverse=True)
    occupied = set()
    out_pts = []
    for r0, c0 in points_sorted:
        r, c = r0, c0
        while True:
            nr, nc = r + dr, c + dc
            if (nr, nc) not in region_set:
                break
            # For diagonal steps, disallow "corner cutting": both orthogonal intermediates must be inside the region.
            if dr != 0 and dc != 0:
                if (r + dr, c) not in region_set or (r, c + dc) not in region_set:
                    break
            if (nr, nc) in occupied:
                break
            r, c = nr, nc
        occupied.add((r, c))
        out_pts.append((r, c))
    return out_pts


def transform(grid: np.ndarray) -> np.ndarray:
    """
    Slide yellow pixels inside walled regions in the direction dictated by border-legend arrows.

    Input structure:
    - bg = most common color; yellow pixels sit inside enclosed regions bounded by colored walls.
    - Border legends: small rectangles on the grid edge whose fill color C matches a region's border color. Each legend contains a center dot and an indicator dot encoding a direction.

    Rule:
    - For each enclosed region, identify its border color C and look up the legend with fill color C.
    - The direction from the legend's center dot to its indicator dot gives the slide direction.
    - All yellow pixels in that region slide in that direction, stacking against the far wall.

    Key insight: Legend fill color links to region border color; the center→indicator offset encodes the movement vector.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _bg_color(g)

    mover = 4  # yellow in ARC
    dirs = _legend_directions(g, bg)
    if not dirs:
        return g.copy()

    passable = (g == bg) | (g == mover)
    out = g.copy()

    # Find enclosed passable regions (not connected to the outer border).
    for pts in _components_mask(passable):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue  # outside background

        movers = [(r, c) for r, c in pts if int(g[r, c]) == mover]
        if not movers:
            continue

        # Determine this region's border color by counting adjacent wall colors.
        border_counts: dict[int, int] = {}
        pset = set(pts)
        for r, c in pts:
            for ddr, ddc in _N4:
                nr, nc = r + ddr, c + ddc
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if (nr, nc) in pset:
                    continue
                v = int(g[nr, nc])
                if v == bg or v == mover:
                    continue
                border_counts[v] = border_counts.get(v, 0) + 1
        if not border_counts:
            continue
        border_col = max(border_counts, key=border_counts.get)
        if border_col not in dirs:
            continue
        dr, dc = dirs[border_col]

        # Clear old movers.
        for r, c in movers:
            out[r, c] = bg

        # Slide and repaint.
        new_pts = _slide_points(pset, movers, dr, dc)
        for r, c in new_pts:
            out[r, c] = mover

    return out
