import random
import numpy as np


BG = 8
ARROW = 6


ARROWS = {
    "r": [(1, 0), (1, 1), (1, 2), (0, 2), (2, 2)],
    "l": [(1, 2), (1, 1), (1, 0), (0, 0), (2, 0)],
    "u": [(2, 1), (1, 1), (0, 1), (0, 0), (0, 2)],
    "d": [(0, 1), (1, 1), (2, 1), (2, 0), (2, 2)],
}


def _arrow_components(grid: np.ndarray):
    seen = set()
    out = []
    for r, c in np.argwhere(grid == ARROW):
        r = int(r); c = int(c)
        if (r, c) in seen:
            continue
        stack = [(r, c)]
        seen.add((r, c))
        comp = []
        while stack:
            x, y = stack.pop()
            comp.append((x, y))
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                q = (x + dx, y + dy)
                if 0 <= q[0] < grid.shape[0] and 0 <= q[1] < grid.shape[1] and grid[q] == ARROW and q not in seen:
                    seen.add(q)
                    stack.append(q)
        out.append(comp)
    return out


def _orient(comp):
    rs = [p[0] for p in comp]; cs = [p[1] for p in comp]
    r0, c0 = min(rs), min(cs)
    norm = sorted((r - r0, c - c0) for r, c in comp)
    for k, pts in ARROWS.items():
        if norm == sorted(pts):
            return k, r0, c0
    return None, r0, c0


def main(grid: np.ndarray) -> np.ndarray:
    """
    Push each rectangle along its magenta arrow until it stops one cell short of the same-colored dock wall.

    Input structure:
    - Colored rectangles lie on an azure background.
    - A magenta arrow touches one side of each rectangle and points toward a same-colored dock wall.

    Rule:
    - Read the arrow direction from its 5-pixel shape and identify the adjacent rectangle on the tail side.
    - Translate that rectangle along the arrow direction until its front edge is exactly
      one cell before the matching dock wall.
    - Erase the arrow and keep the dock wall unchanged.

    Key insight:
    - The arrow determines both the moved object and its direction, while the stopping
      position is fixed by a separate same-color wall farther ahead.
    """
    out = grid.copy()
    for comp in _arrow_components(grid):
        d, r0, c0 = _orient(comp)
        if d is None:
            continue
        if d == "r":
            sr, sc = r0 + 1, c0 - 1
            dr, dc = 0, 1
        elif d == "l":
            sr, sc = r0 + 1, c0 + 3
            dr, dc = 0, -1
        elif d == "u":
            sr, sc = r0 + 3, c0 + 1
            dr, dc = -1, 0
        else:
            sr, sc = r0 - 1, c0 + 1
            dr, dc = 1, 0
        color = int(grid[sr, sc])
        pts = np.argwhere(grid == color)
        rmin, cmin = pts.min(axis=0)
        rmax, cmax = pts.max(axis=0)
        if d == "r":
            candidates = [c for c in range(cmax + 1, grid.shape[1]) if np.all(grid[rmin : rmax + 1, c] == color)]
            if not candidates:
                candidates = [cmax + 1 + (cmax - cmin)]
            stop = candidates[0] - 1 - (cmax - cmin)
            new_c0 = stop
            out[rmin : rmax + 1, cmin : cmax + 1] = BG
            out[rmin : rmax + 1, new_c0 : new_c0 + (cmax - cmin + 1)] = color
        elif d == "l":
            candidates = [c for c in range(cmin - 1, -1, -1) if np.all(grid[rmin : rmax + 1, c] == color)]
            if not candidates:
                candidates = [max(0, cmin - 2)]
            stop = candidates[0] + 1
            out[rmin : rmax + 1, cmin : cmax + 1] = BG
            out[rmin : rmax + 1, stop : stop + (cmax - cmin + 1)] = color
        elif d == "u":
            candidates = [r for r in range(rmin - 1, -1, -1) if np.all(grid[r, cmin : cmax + 1] == color)]
            if not candidates:
                candidates = [max(0, rmin - 2)]
            stop = candidates[0] + 1
            out[rmin : rmax + 1, cmin : cmax + 1] = BG
            out[stop : stop + (rmax - rmin + 1), cmin : cmax + 1] = color
        else:
            candidates = [r for r in range(rmax + 1, grid.shape[0]) if np.all(grid[r, cmin : cmax + 1] == color)]
            if not candidates:
                candidates = [min(grid.shape[0] - 1, rmax + 2 + (rmax - rmin))]
            stop = candidates[0] - 1 - (rmax - rmin)
            out[rmin : rmax + 1, cmin : cmax + 1] = BG
            out[stop : stop + (rmax - rmin + 1), cmin : cmax + 1] = color
        for r, c in comp:
            out[r, c] = BG
    return out


def generate_input() -> np.ndarray:
    grid = np.full((22, 22), BG, dtype=int)
    colors = random.sample([1, 2, 3, 4, 5, 7, 9], random.randint(2, 4))
    used_rows = []
    for color in colors:
        d = random.choice(["r", "l", "u", "d"])
        if d in ("r", "l"):
            h = random.randint(1, 2); w = random.randint(2, 4)
            r = random.randint(2, 18)
            c = random.randint(6, 10)
            grid[r : r + h, c : c + w] = color
            if d == "r":
                for dr, dc in ARROWS["r"]:
                    grid[r - 1 + dr, c - 3 + dc] = ARROW
                wall_c = random.randint(c + w + 3, 20)
                grid[r : r + h, wall_c] = color
            else:
                for dr, dc in ARROWS["l"]:
                    grid[r - 1 + dr, c + w + dc] = ARROW
                wall_c = random.randint(1, max(1, c - 4))
                grid[r : r + h, wall_c] = color
        else:
            h = random.randint(2, 4); w = random.randint(1, 2)
            r = random.randint(6, 10)
            c = random.randint(2, 18)
            grid[r : r + h, c : c + w] = color
            if d == "u":
                for dr, dc in ARROWS["u"]:
                    grid[r + h + dr, c - 1 + dc] = ARROW
                wall_r = random.randint(1, max(1, r - 4))
                grid[wall_r, c : c + w] = color
            else:
                for dr, dc in ARROWS["d"]:
                    grid[r - 3 + dr, c - 1 + dc] = ARROW
                wall_r = random.randint(r + h + 3, 20)
                grid[wall_r, c : c + w] = color
    return grid
