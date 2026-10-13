from collections import deque
import random

import numpy as np

BLACK, BLUE, RED, GREEN, YELLOW, GREY, MAGENTA, ORANGE, AZURE, MAROON = range(10)
DIRS = {"up": (-1, 0), "right": (0, 1), "down": (1, 0), "left": (0, -1)}


def _bbox(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), min(cs), max(rs), max(cs)


def _components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not mask[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in DIRS.values():
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
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


def _gap(grid, box):
    r0, c0, r1, c1 = box
    missing = []
    for r, c in _perimeter(r0, c0, r1, c1):
        if int(grid[r, c]) != GREY:
            side = "up" if r == r0 else "down" if r == r1 else "left" if c == c0 else "right"
            missing.append((side, r, c))
    return missing[0] if len(missing) == 1 else None


def main(grid: np.ndarray) -> np.ndarray:
    """Draw outward rays from gapped grey frames using the count and color of their seeds.

    Input structure: Black background contains separate grey rectangular frames,
    each with one missing perimeter cell.  Inside each frame are one or more
    same-color seed pixels.
    Rule: For every frame, identify the gap side, count the seed pixels inside,
    and draw a ray of that length from the gap outward in the seed color, clipped
    at the grid edge.  Frames and seeds remain unchanged.
    Key insight: The gap encodes direction while the seed population encodes ray
    length; seed color is only the ray paint.
    """
    out = grid.copy()
    for cells in _components(grid == GREY):
        box = _bbox(cells)
        r0, c0, r1, c1 = box
        if r1 - r0 < 3 or c1 - c0 < 3:
            continue
        gap = _gap(grid, box)
        if not gap:
            continue
        seeds = [(r, c, int(grid[r, c])) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1) if int(grid[r, c]) not in (BLACK, GREY)]
        if not seeds:
            continue
        color = seeds[0][2]
        dr, dc = DIRS[gap[0]]
        r, c = gap[1] + dr, gap[2] + dc
        for _ in range(len(seeds)):
            if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]:
                out[r, c] = color
            r += dr
            c += dc
    return out


def _draw_frame(grid, r0, c0, h, w, side):
    r1, c1 = r0 + h - 1, c0 + w - 1
    for r, c in _perimeter(r0, c0, r1, c1):
        grid[r, c] = GREY
    if side in ("up", "down"):
        r, c = (r0 if side == "up" else r1), c0 + w // 2
    else:
        r, c = r0 + h // 2, (c0 if side == "left" else c1)
    grid[r, c] = BLACK


def generate_input() -> np.ndarray:
    h, w = random.randint(16, 28), random.randint(16, 28)
    grid = np.zeros((h, w), dtype=int)
    used = np.zeros((h, w), dtype=bool)
    n = random.randint(2, 5)
    colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], n)
    for color in colors:
        for _ in range(300):
            rh, rw = random.randint(5, 8), random.randint(5, 8)
            r0, c0 = random.randint(2, h - rh - 2), random.randint(2, w - rw - 2)
            if not used[r0 - 2 : r0 + rh + 2, c0 - 2 : c0 + rw + 2].any():
                side = random.choice(list(DIRS))
                _draw_frame(grid, r0, c0, rh, rw, side)
                for _ in range(random.randint(1, min(5, (rh - 2) * (rw - 2)))):
                    grid[random.randint(r0 + 1, r0 + rh - 2), random.randint(c0 + 1, c0 + rw - 2)] = color
                used[r0 - 2 : r0 + rh + 2, c0 - 2 : c0 + rw + 2] = True
                break
    return grid
