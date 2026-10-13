import numpy as np
from collections import Counter, deque

from arc_utils import DIR4 as _N4, DIR8 as _N8, cc4 as _cc4, get_bg as _bg_color


def _flood(start: tuple[int, int], passable, h: int, w: int) -> set[tuple[int, int]]:
    q = deque([start])
    vis = {start}
    while q:
        r, c = q.popleft()
        for dr, dc in _N4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in vis and passable(nr, nc):
                vis.add((nr, nc))
                q.append((nr, nc))
    return vis


def transform(grid: np.ndarray) -> np.ndarray:
    """
    Paint an inner border around the region reachable from a corner marker pixel.

    Input structure:
    - bg = most common color; wall color (blue) forms barriers dividing the grid.
    - A unique single pixel (magenta) sits on one corner — the start marker.
    - Some wall-colored pixels are structural (border-touching, enclosing regions); others are decorations.

    Rule:
    - Flood-fill from the marker corner to find all reachable bg cells, blocked by structural walls.
    - Erase wall-colored decorations hidden behind structural walls (not reachable from marker side).
    - Paint orange (color 7) on every reachable bg cell adjacent to a wall or grid edge — forming a 1-cell inner border.

    Key insight: Only border-connected wall components act as real barriers; interior wall fragments are decorations to be removed.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _bg_color(g)

    out = g.copy()

    # Marker pixel is a unique (count==1) non-bg pixel on a corner.
    counts = Counter(int(v) for v in g.ravel())
    corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
    marker = None
    for r, c in corners:
        v = int(g[r, c])
        if v != bg and counts.get(v, 0) == 1:
            marker = (r, c, v)
            break
    if marker is None:
        return out

    mr, mc, marker_col = marker

    # Wall color: most frequent non-bg, non-marker color.
    other_cols = [c for c in counts.keys() if c not in (bg, marker_col)]
    if not other_cols:
        return out
    wall_col = max(other_cols, key=lambda c: counts[c])

    # First reach: treat all wall pixels as obstacles.
    reach0 = _flood(
        (mr, mc),
        lambda r, c: int(g[r, c]) in (bg, marker_col),
        h,
        w,
    )

    # Structural walls are wall-color components that touch the border AND are adjacent to reach0.
    walls: set[tuple[int, int]] = set()
    for comp in _cc4(g == wall_col):
        touches_border = any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp)
        if not touches_border:
            continue
        adjacent_to_reach0 = any(any((r + dr, c + dc) in reach0 for dr, dc in _N4) for r, c in comp)
        if adjacent_to_reach0:
            walls |= set(comp)

    # Second reach: allow passing through non-wall wall-color pixels (decorations) so they don't create borders.
    reach = _flood(
        (mr, mc),
        lambda r, c: int(g[r, c]) in (bg, marker_col) or (int(g[r, c]) == wall_col and (r, c) not in walls),
        h,
        w,
    )

    # Remove wall-colored decorations that are not reachable (they lie behind structural walls).
    for r in range(h):
        for c in range(w):
            if int(out[r, c]) == wall_col and (r, c) not in walls and (r, c) not in reach:
                out[r, c] = bg

    fill_col = 7
    # Paint a 1-cell-thick inner border of the reachable region (on background cells only).
    for r, c in reach:
        if int(out[r, c]) != bg:
            continue
        if r == 0 or c == 0 or r == h - 1 or c == w - 1:
            out[r, c] = fill_col
            continue
        if any((r + dr, c + dc) in walls for dr, dc in _N8):
            out[r, c] = fill_col
            continue
        if any((r + dr, c + dc) not in reach for dr, dc in _N4):
            out[r, c] = fill_col

    return out
