from collections import deque
import random

import numpy as np

BLACK, BLUE, RED, GREEN, YELLOW, GREY, MAGENTA, ORANGE, AZURE, MAROON = range(10)
DIRS = {"up": (-1, 0), "right": (0, 1), "down": (1, 0), "left": (0, -1)}


def _bbox(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), min(cs), max(rs), max(cs)


def _components_of(grid, color):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in DIRS.values():
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(cells)
    return comps


def _perimeter(r0, c0, r1, c1):
    cells = []
    for c in range(c0, c1 + 1):
        cells.extend([(r0, c), (r1, c)])
    for r in range(r0 + 1, r1):
        cells.extend([(r, c0), (r, c1)])
    return cells


def _gap(grid, box, color):
    r0, c0, r1, c1 = box
    missing = []
    for r, c in _perimeter(r0, c0, r1, c1):
        if int(grid[r, c]) != color:
            side = "up" if r == r0 else "down" if r == r1 else "left" if c == c0 else "right"
            missing.append((side, r, c))
    return missing[0] if len(missing) == 1 else None


def _draw_frame(grid, r0, c0, h, w, color, side):
    r1, c1 = r0 + h - 1, c0 + w - 1
    for r, c in _perimeter(r0, c0, r1, c1):
        grid[r, c] = color
    if side in ("up", "down"):
        r = r0 if side == "up" else r1
        c = (c0 + c1) // 2
    else:
        r = (r0 + r1) // 2
        c = c0 if side == "left" else c1
    grid[r, c] = BLACK


def main(grid: np.ndarray) -> np.ndarray:
    """Grow an upward column from each room seed using exit direction and room area.

    Input structure: A black grid has a four-color top legend and separated blue
    rectangular rooms.  Each room is a one-pixel wall frame with one missing wall
    cell as an exit and one colored seed on its bottom interior row.
    Rule: The top legend is read left to right as exit up, right, down, left to
    output color.  For every room, count black interior cells, find the exit side,
    and paint upward from the seed for that count using the legend color for that
    side.  The column may cross walls and is clipped at the grid edge.
    Key insight: The missing wall cell supplies color through a directional legend,
    while the room's area supplies the column height.
    """
    out = grid.copy()
    legend = {"up": int(grid[0, 0]), "right": int(grid[0, 1]), "down": int(grid[0, 2]), "left": int(grid[0, 3])}
    for cells in _components_of(grid, BLUE):
        box = _bbox(cells)
        r0, c0, r1, c1 = box
        if r1 - r0 < 3 or c1 - c0 < 3:
            continue
        g = _gap(grid, box, BLUE)
        if g is None:
            continue
        side = g[0]
        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        seeds = [(r0 + 1 + r, c0 + 1 + c, int(v)) for r, row in enumerate(interior) for c, v in enumerate(row) if int(v) not in (BLACK, BLUE)]
        if not seeds:
            continue
        sr, sc, _ = max(seeds, key=lambda x: x[0])
        height = int(np.sum(interior == BLACK))
        for k in range(height):
            rr = sr - k
            if 0 <= rr < grid.shape[0]:
                out[rr, sc] = legend[side]
    return out


def generate_input() -> np.ndarray:
    h, w = random.randint(20, 30), random.randint(20, 30)
    grid = np.zeros((h, w), dtype=int)
    legend = random.sample([2, 3, 4, 6, 7, 8], 4)
    grid[0, :4] = legend
    used = np.zeros((h, w), dtype=bool)
    used[:2, :6] = True
    seed_colors = random.sample([2, 3, 4, 6, 7, 8, 9], random.randint(2, 4))
    for color in seed_colors:
        for _ in range(300):
            rh, rw = random.randint(4, 7), random.randint(4, 7)
            r0, c0 = random.randint(3, h - rh - 1), random.randint(1, w - rw - 1)
            if not used[r0 - 1 : r0 + rh + 1, c0 - 1 : c0 + rw + 1].any():
                side = random.choice(list(DIRS))
                _draw_frame(grid, r0, c0, rh, rw, BLUE, side)
                grid[r0 + rh - 2, random.randint(c0 + 1, c0 + rw - 2)] = color
                used[r0 - 1 : r0 + rh + 1, c0 - 1 : c0 + rw + 1] = True
                break
    return grid
