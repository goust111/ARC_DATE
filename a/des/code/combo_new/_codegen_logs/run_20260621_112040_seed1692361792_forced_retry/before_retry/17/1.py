import random
from collections import deque
import numpy as np

BLACK = 0


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


def bbox(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), max(rs), min(cs), max(cs)


def frames(grid):
    out = []
    for color in [v for v in np.unique(grid) if v != BLACK]:
        for cells in components(grid == color):
            r0, r1, c0, c1 = bbox(cells)
            if r1 - r0 < 4 or c1 - c0 < 4:
                continue
            border = {(r0, c) for c in range(c0, c1 + 1)} | {(r1, c) for c in range(c0, c1 + 1)}
            border |= {(r, c0) for r in range(r0, r1 + 1)} | {(r, c1) for r in range(r0, r1 + 1)}
            if set(cells) == border:
                out.append((color, r0, r1, c0, c1))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, dtype=int)
    out = grid.copy()
    seed_colors = sorted({grid[r, c] for _, r0, r1, c0, c1 in frames(grid) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1) if grid[r, c] not in (BLACK, grid[r0, c0])})
    width_map = {c: 2 * i + 1 for i, c in enumerate(seed_colors)}
    for frame_color, r0, r1, c0, c1 in frames(grid):
        seed = [(r, c, grid[r, c]) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1) if grid[r, c] not in (BLACK, frame_color)]
        if not seed:
            continue
        sr, sc, color = seed[0]
        ih, iw = r1 - r0 - 1, c1 - c0 - 1
        upper = (sr - (r0 + 1)) <= (ih - 1) // 2
        left = (sc - (c0 + 1)) <= (iw - 1) // 2
        if upper and left:
            side = "up"
        elif upper and not left:
            side = "right"
        elif (not upper) and left:
            side = "left"
        else:
            side = "down"
        width = min(width_map[color], (c1 - c0 - 1) if side in ("up", "down") else (r1 - r0 - 1))
        if width % 2 == 0:
            width -= 1
        if side == "up":
            start = c0 + (c1 - c0 - width + 1) // 2
            out[r0, start : start + width] = BLACK
        elif side == "down":
            start = c0 + (c1 - c0 - width + 1) // 2
            out[r1, start : start + width] = BLACK
        elif side == "left":
            start = r0 + (r1 - r0 - width + 1) // 2
            out[start : start + width, c0] = BLACK
        else:
            start = r0 + (r1 - r0 - width + 1) // 2
            out[start : start + width, c1] = BLACK
    return out


def generate_input() -> np.ndarray:
    h, w = random.randint(18, 24), random.randint(20, 28)
    grid = np.zeros((h, w), dtype=int)
    seed_palette = random.sample([1, 2, 3, 4, 6, 7, 8, 9], 3)
    seed_palette.sort()
    x = 1
    for _ in range(random.randint(2, 4)):
        fh, fw = random.randint(6, 9), random.randint(6, 10)
        r0 = random.randint(1, h - fh - 1)
        c0 = x
        if c0 + fw >= w - 1:
            break
        frame_color = random.choice([c for c in [1, 2, 3, 4, 6, 7, 8, 9] if c not in seed_palette])
        grid[r0, c0 : c0 + fw] = frame_color
        grid[r0 + fh - 1, c0 : c0 + fw] = frame_color
        grid[r0 : r0 + fh, c0] = frame_color
        grid[r0 : r0 + fh, c0 + fw - 1] = frame_color
        seed_color = random.choice(seed_palette)
        quad = random.choice(["tl", "tr", "bl", "br"])
        rr = r0 + 1 + (0 if quad[0] == "t" else (fh - 3) // 2 + 1)
        cc = c0 + 1 + (0 if quad[1] == "l" else (fw - 3) // 2 + 1)
        rr = min(rr, r0 + fh - 2)
        cc = min(cc, c0 + fw - 2)
        grid[rr, cc] = seed_color
        x += fw + 2
    return grid
