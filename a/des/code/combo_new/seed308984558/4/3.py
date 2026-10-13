import random

import numpy as np


BG = 0
WALL = 5


def _branch(out, occ, endpoint, axis, length, color):
    dirs = [(0, 1), (0, -1)] if axis == "v" else [(1, 0), (-1, 0)]
    for dr, dc in dirs:
        cells = [(endpoint[0] + dr * k, endpoint[1] + dc * k) for k in range(1, length + 1)]
        if not all(0 <= r < out.shape[0] and 0 <= c < out.shape[1] and not occ[r, c] for r, c in cells):
            continue
        for r, c in cells:
            out[r, c] = color
            occ[r, c] = True
        _branch(out, occ, cells[-1], "h" if axis == "v" else "v", length, color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow recursive perpendicular branches of fixed length from each wall-anchored seed segment.

    Input structure:
    - Grey wall pixels are sparse obstacles.
    - Colored horizontal or vertical seed segments touch a wall at exactly one endpoint.
    - Each seed segment has a unique color and a length from two to four cells.

    Rule:
    - Read the growth direction from the endpoint touching the wall toward the free endpoint.
    - From each free endpoint, grow two perpendicular branches of the same length whenever the whole branch fits in background.
    - Recurse from every new free endpoint, alternating the growth axis, without overwriting any existing non-background pixel.

    Key insight:
    - The wall chooses the root direction, while the seed length becomes the invariant branch length for the entire recursive tree.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    occ = g != BG
    colors = [v for v in np.unique(g) if v not in (BG, WALL)]
    for color in colors:
        pts = list(zip(*np.where(g == int(color)), strict=False))
        rows = {r for r, _ in pts}
        cols = {c for _, c in pts}
        axis = "h" if len(rows) == 1 else "v"
        pts = sorted(pts)
        ends = [pts[0], pts[-1]]
        wall_end = next(e for e in ends if any(0 <= e[0] + dr < g.shape[0] and 0 <= e[1] + dc < g.shape[1] and g[e[0] + dr, e[1] + dc] == WALL for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]))
        free_end = ends[0] if ends[1] == wall_end else ends[1]
        _branch(out, occ.copy(), free_end, axis, len(pts), int(color))
    return out


def generate_input() -> np.ndarray:
    for _ in range(300):
        h, w = random.randint(18, 24), random.randint(18, 24)
        g = np.zeros((h, w), dtype=int)
        colors = random.sample([1, 2, 3, 4, 6], random.randint(1, 2))
        ok = True
        for color in colors:
            placed = False
            for _attempt in range(100):
                axis = random.choice(["h", "v"])
                length = random.randint(2, 4)
                if axis == "h":
                    r = random.randint(2, h - 3)
                    c = random.randint(2, w - length - 3)
                    cells = [(r, cc) for cc in range(c, c + length)]
                    wall = (r, c - 1)
                    free = (r, c + length - 1)
                else:
                    r = random.randint(2, h - length - 3)
                    c = random.randint(2, w - 3)
                    cells = [(rr, c) for rr in range(r, r + length)]
                    wall = (r - 1, c)
                    free = (r + length - 1, c)

                if g[wall] != BG or any(g[cell] != BG for cell in cells):
                    continue
                if any(
                    0 <= free[0] + dr < h
                    and 0 <= free[1] + dc < w
                    and g[free[0] + dr, free[1] + dc] == WALL
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                ):
                    continue

                g[wall] = WALL
                for cell in cells:
                    g[cell] = color
                placed = True
                break
            if not placed:
                ok = False
                break
        if ok:
            try:
                main(g)
                return g
            except Exception:
                continue
    raise RuntimeError("failed to generate recursive seed-branch input")
