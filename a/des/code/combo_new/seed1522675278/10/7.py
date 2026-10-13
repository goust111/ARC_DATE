import random
import numpy as np


BG = 0
RED = 2
GREY = 5
LEGEND = [1, 3, 4, 7, 6]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each red path by how many hollow frame channels it passes through.

    Input structure:
    - Grey hollow rectangles with opposite-side gaps create horizontal or vertical channels.
    - Red one-pixel paths run across the grid.
    - A corner legend lists the colors for channel counts 0 through 4.

    Rule:
    - Count, for each red path, how many distinct channel frames it traverses.
    - Recolor the whole path using the legend color for that count.
    - Keep the frames and legend unchanged.

    Key insight:
    - All paths start the same color; the differentiating feature is their topological
      interaction count with the ring channels.
    """
    out = grid.copy()
    frames = []
    for pts in _components(grid == GREY):
        rs = [p[0] for p in pts]
        cs = [p[1] for p in pts]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        if r1 - r0 >= 3 and c1 - c0 >= 3:
            frames.append((r0, r1, c0, c1))
    for pts in _components(grid == RED):
        cells = set(pts)
        rows = {r for r, _ in pts}
        cols = {c for _, c in pts}
        count = 0
        for r0, r1, c0, c1 in frames:
            inside = [(r, c) for r, c in pts if r0 < r < r1 and c0 < c < c1]
            if not inside:
                continue
            if len(rows) == 1:
                row = next(iter(rows))
                if any((row, c0) in cells or (row, c1) in cells for _ in [0]):
                    count += 1
            elif len(cols) == 1:
                col = next(iter(cols))
                if any((r0, col) in cells or (r1, col) in cells for _ in [0]):
                    count += 1
        color = LEGEND[min(count, 4)]
        for r, c in pts:
            out[r, c] = color
    return out


def _components(mask: np.ndarray):
    seen = set()
    out = []
    H, W = mask.shape
    for r in range(H):
        for c in range(W):
            if not mask[r, c] or (r, c) in seen:
                continue
            stack = [(r, c)]
            seen.add((r, c))
            comp = []
            while stack:
                x, y = stack.pop()
                comp.append((x, y))
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < H and 0 <= ny < W and mask[nx, ny] and (nx, ny) not in seen:
                        seen.add((nx, ny))
                        stack.append((nx, ny))
            out.append(comp)
    return out


def generate_input() -> np.ndarray:
    H = W = random.randint(20, 26)
    grid = np.zeros((H, W), dtype=int)
    grid[:5, 0] = LEGEND
    channels = []
    for _ in range(random.randint(2, 4)):
        orient = random.choice(["h", "v"])
        if orient == "h":
            h = random.randint(5, 7)
            w = random.randint(7, 11)
            r = random.randint(6, H - h - 1)
            c = random.randint(3, W - w - 3)
            grid[r, c : c + w] = GREY
            grid[r + h - 1, c : c + w] = GREY
            grid[r : r + h, c] = GREY
            grid[r : r + h, c + w - 1] = GREY
            gap = random.randint(c + 1, c + w - 2)
            grid[r, gap] = BG
            grid[r + h - 1, gap] = BG
            channels.append(("v", gap, r, r + h - 1))
        else:
            h = random.randint(7, 11)
            w = random.randint(5, 7)
            r = random.randint(6, H - h - 1)
            c = random.randint(3, W - w - 3)
            grid[r, c : c + w] = GREY
            grid[r + h - 1, c : c + w] = GREY
            grid[r : r + h, c] = GREY
            grid[r : r + h, c + w - 1] = GREY
            gap = random.randint(r + 1, r + h - 2)
            grid[gap, c] = BG
            grid[gap, c + w - 1] = BG
            channels.append(("h", gap, c, c + w - 1))
    for orient, pos, a, b in channels:
        if orient == "v":
            grid[:, pos] = np.where(grid[:, pos] == BG, RED, grid[:, pos])
        else:
            grid[pos, :] = np.where(grid[pos, :] == BG, RED, grid[pos, :])
    return grid
