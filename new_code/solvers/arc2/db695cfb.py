import numpy as np

from arc_utils import get_bg as _bg_color

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Connect same-color diagonal pixel pairs, then emit perpendicular rays from differently-colored pixels on those diagonals.

    Input structure:
    - bg = most common color. Line color (e.g. blue) has pixel pairs aligned on shared diagonals. Trigger color (e.g. magenta) pixels sit along those diagonals.

    Rule:
    - Draw diagonal lines connecting each pair of same-color aligned pixels.
    - Where the diagonal crosses a trigger pixel, emit perpendicular diagonal rays (in the trigger's color) extending to the grid boundary.

    Key insight: Trigger pixels act as beam-splitters — the perpendicular rays rotate 90° from the connecting diagonal's orientation.
    """

    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()

    bg = _bg_color(grid)

    non_bg = [(r, c, int(grid[r, c])) for r in range(h) for c in range(w) if int(grid[r, c]) != bg]
    if not non_bg:
        return out

    by_color: dict[int, list[tuple[int, int]]] = {}
    for r, c, col in non_bg:
        by_color.setdefault(col, []).append((r, c))

    def diag_path(p: tuple[int, int], q: tuple[int, int]) -> list[tuple[int, int]]:
        r0, c0 = p
        r1, c1 = q
        dr = 1 if r1 > r0 else -1
        dc = 1 if c1 > c0 else -1
        steps = abs(r1 - r0)
        return [(r0 + k * dr, c0 + k * dc) for k in range(steps + 1)]

    # Choose the "line color": the non-background color with the longest diagonal-aligned pair.
    best = None  # (dist, color)
    for col, pts in by_color.items():
        if len(pts) < 2:
            continue
        for i in range(len(pts)):
            r0, c0 = pts[i]
            for j in range(i + 1, len(pts)):
                r1, c1 = pts[j]
                dr = r1 - r0
                dc = c1 - c0
                if dr == 0 or abs(dr) != abs(dc):
                    continue
                dist = abs(dr)
                cand = (dist, col)
                if best is None or cand > best:
                    best = cand

    if best is None:
        return out

    _, line_color = best
    pts = by_color.get(line_color, [])
    if not pts:
        return out

    # Collect triggers on any connecting segment. Each trigger stores which perpendicular
    # directions to extend (depends on the segment orientation).
    triggers: dict[tuple[int, int, int], set[tuple[int, int]]] = {}

    # Group line_color points by both diagonal orientations.
    groups_a: dict[int, list[tuple[int, int]]] = {}  # r - c constant (slope +1)
    groups_b: dict[int, list[tuple[int, int]]] = {}  # r + c constant (slope -1)
    for r, c in pts:
        groups_a.setdefault(r - c, []).append((r, c))
        groups_b.setdefault(r + c, []).append((r, c))

    def draw_segment(p: tuple[int, int], q: tuple[int, int]) -> None:
        path = diag_path(p, q)
        dr = 1 if q[0] > p[0] else -1
        dc = 1 if q[1] > p[1] else -1
        perp_dirs = [(dr, -dc), (-dr, dc)]
        for r, c in path:
            v = int(out[r, c])
            if v == bg:
                out[r, c] = line_color
            elif v != line_color:
                key = (r, c, v)
                triggers.setdefault(key, set()).update(perp_dirs)

    # For each diagonal that has >=2 line_color points, connect its extremes.
    for g in groups_a.values():
        if len(g) >= 2:
            g = sorted(g)
            draw_segment(g[0], g[-1])

    for g in groups_b.values():
        if len(g) >= 2:
            g = sorted(g)
            draw_segment(g[0], g[-1])

    # Emit perpendicular diagonal rays from each trigger to the grid boundary.
    for (r0, c0, tcol), dirs in triggers.items():
        for ddr, ddc in dirs:
            r, c = r0 + ddr, c0 + ddc
            while 0 <= r < h and 0 <= c < w:
                if int(out[r, c]) == bg:
                    out[r, c] = tcol
                r += ddr
                c += ddc

    return out
