import random
from collections import deque
import numpy as np

BLACK = 0
GREY = 5


def components(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if mask[r, c] and not seen[r, c]:
                q = deque([(r, c)])
                seen[r, c] = True
                cells = []
                while q:
                    x, y = q.popleft()
                    cells.append((x, y))
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] and not seen[nx, ny]:
                            seen[nx, ny] = True
                            q.append((nx, ny))
                comps.append(cells)
    return comps


def arrow_dir(cells):
    cells = sorted(cells)
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    if len(set(rs)) == 2:
        top = [p for p in cells if p[0] == min(rs)]
        bot = [p for p in cells if p[0] == max(rs)]
        if len(top) == 2:
            return "up"
        if len(bot) == 2:
            return "down"
    if len(set(cs)) == 2:
        left = [p for p in cells if p[1] == min(cs)]
        right = [p for p in cells if p[1] == max(cs)]
        if len(left) == 2:
            return "left"
        if len(right) == 2:
            return "right"
    return None


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, dtype=int)
    out = grid.copy()
    mask = grid != GREY
    for comp in components(mask):
        if any(r in (0, grid.shape[0] - 1) or c in (0, grid.shape[1] - 1) for r, c in comp):
            continue
        arrow = [(r, c) for r, c in comp if grid[r, c] not in (BLACK, GREY)]
        if len(arrow) != 3:
            continue
        color = grid[arrow[0]]
        direction = arrow_dir(arrow)
        rs = [r for r, _ in comp]
        cs = [c for _, c in comp]
        r0, r1, c0, c1 = min(rs) - 1, max(rs) + 1, min(cs) - 1, max(cs) + 1
        if direction == "up":
            out[r0, c0 : c1 + 1] = color
        elif direction == "down":
            out[r1, c0 : c1 + 1] = color
        elif direction == "left":
            out[r0 : r1 + 1, c0] = color
        elif direction == "right":
            out[r0 : r1 + 1, c1] = color
        for r, c in arrow:
            out[r, c] = BLACK
    return out


def paint_arrow(patch, color, direction):
    cr, cc = patch.shape[0] // 2, patch.shape[1] // 2
    if direction == "up":
        patch[cr, cc] = color
        patch[cr - 1, cc - 1] = color
        patch[cr - 1, cc + 1] = color
    elif direction == "down":
        patch[cr, cc] = color
        patch[cr + 1, cc - 1] = color
        patch[cr + 1, cc + 1] = color
    elif direction == "left":
        patch[cr, cc] = color
        patch[cr - 1, cc - 1] = color
        patch[cr + 1, cc - 1] = color
    else:
        patch[cr, cc] = color
        patch[cr - 1, cc + 1] = color
        patch[cr + 1, cc + 1] = color


def generate_input() -> np.ndarray:
    h, w = random.randint(18, 24), random.randint(18, 24)
    grid = np.zeros((h, w), dtype=int)
    for _ in range(random.randint(2, 4)):
        rh, rw = random.randint(5, 8), random.randint(5, 8)
        r0 = random.randint(1, h - rh - 1)
        c0 = random.randint(1, w - rw - 1)
        if np.any(grid[r0 - 1 : r0 + rh + 1, c0 - 1 : c0 + rw + 1] != BLACK):
            continue
        grid[r0, c0 : c0 + rw] = GREY
        grid[r0 + rh - 1, c0 : c0 + rw] = GREY
        grid[r0 : r0 + rh, c0] = GREY
        grid[r0 : r0 + rh, c0 + rw - 1] = GREY
        paint_arrow(grid[r0 + 1 : r0 + rh - 1, c0 + 1 : c0 + rw - 1], random.choice([1, 2, 3, 4, 6, 7, 8, 9]), random.choice(["up", "down", "left", "right"]))
    return grid
