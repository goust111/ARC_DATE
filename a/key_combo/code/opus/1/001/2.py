import random
from collections import deque

import numpy as np


BG = 0
AXIS = 5
COLORS = [1, 2, 3, 4]


def _find_axes(g: np.ndarray) -> tuple[int, int]:
    rows = [r for r in range(g.shape[0]) if np.all(g[r, :] == AXIS)]
    cols = [c for c in range(g.shape[1]) if np.all(g[:, c] == AXIS)]
    assert len(rows) == 1 and len(cols) == 1, "expected one grey cross"
    return rows[0], cols[0]


def _quadrant_slices(ar: int, ac: int):
    return {
        "TL": (slice(0, ar), slice(0, ac)),
        "TR": (slice(0, ar), slice(ac + 1, None)),
        "BL": (slice(ar + 1, None), slice(0, ac)),
        "BR": (slice(ar + 1, None), slice(ac + 1, None)),
    }


def _interior_holes(sub: np.ndarray) -> np.ndarray:
    h, w = sub.shape
    bg = sub == BG
    seen = np.zeros((h, w), dtype=bool)
    q = deque()
    for r in range(h):
        for c in (0, w - 1):
            if bg[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if bg[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and bg[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    return bg & ~seen


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each enclosed quadrant cavity with the color of the diagonal quadrant's ring.

    Input structure:
    - A black grid contains one complete grey row and one complete grey column.
    - The cross divides the grid into four nonempty quadrants.
    - Each quadrant contains one single-color closed ring with black interior cells.

    Rule:
    - Find the grey cross and split the grid into TL, TR, BL, and BR quadrants.
    - In each quadrant, flood-fill black cells from the quadrant boundary; remaining black cells are enclosed cavities.
    - Fill TL and BR cavities with each other's ring colors, and TR and BL cavities with each other's ring colors.
    - Keep the cross, ring outlines, and exterior black background unchanged.

    Key insight:
    - The fill color is determined by the diagonal quadrant, so all four ring colors must be identified before any cavity is solved.
    """
    g = np.asarray(grid, dtype=int)
    ar, ac = _find_axes(g)
    qs = _quadrant_slices(ar, ac)
    colors = {}
    holes = {}
    for name, sl in qs.items():
        sub = g[sl]
        vals = set(map(int, sub.ravel())) - {BG}
        assert len(vals) == 1, "each quadrant must contain one ring color"
        colors[name] = vals.pop()
        holes[name] = _interior_holes(sub)
        assert holes[name].any(), "each ring must enclose black cells"

    diag = {"TL": "BR", "BR": "TL", "TR": "BL", "BL": "TR"}
    out = g.copy()
    for name, sl in qs.items():
        sub = out[sl]
        sub[holes[name]] = colors[diag[name]]
    return out


def _place_ring(grid, rs, cs, color):
    rlo, rhi = rs.start or 0, rs.stop if rs.stop is not None else grid.shape[0]
    clo, chi = cs.start or 0, cs.stop if cs.stop is not None else grid.shape[1]
    candidates = [(3, 4), (4, 3), (4, 4), (4, 5), (5, 4)]
    candidates = [(h, w) for h, w in candidates if rhi - rlo >= h + 2 and chi - clo >= w + 2]
    if not candidates:
        return False
    h, w = random.choice(candidates)
    r0 = random.randint(rlo + 1, rhi - h - 1)
    c0 = random.randint(clo + 1, chi - w - 1)
    grid[r0 : r0 + h, c0] = color
    grid[r0 : r0 + h, c0 + w - 1] = color
    grid[r0, c0 : c0 + w] = color
    grid[r0 + h - 1, c0 : c0 + w] = color
    return True


def _gen_attempt():
    h = random.randint(15, 21)
    w = random.randint(15, 21)
    ar = random.randint(6, h - 7)
    ac = random.randint(6, w - 7)
    grid = np.zeros((h, w), dtype=int)
    grid[ar, :] = AXIS
    grid[:, ac] = AXIS
    colors = random.sample(COLORS, 4)
    for color, sl in zip(colors, _quadrant_slices(ar, ac).values()):
        if not _place_ring(grid, sl[0], sl[1], color):
            return None
    try:
        main(grid)
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
