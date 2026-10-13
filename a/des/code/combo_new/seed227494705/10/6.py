import random
import numpy as np


def _components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = [(r, c)]
            seen[r, c] = True
            pts = []
            while q:
                x, y = q.pop()
                pts.append((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] and not seen[nx, ny]:
                        seen[nx, ny] = True
                        q.append((nx, ny))
            comps.append(pts)
    return comps


def _bbox(pts):
    rs = [p[0] for p in pts]
    cs = [p[1] for p in pts]
    return min(rs), max(rs), min(cs), max(cs)


def main(grid: np.ndarray) -> np.ndarray:
    """Erase or keep room contents by exit-count parity and color every exit from a legend.

    Input structure: black background with grey rectangular wall regions whose
    border lines may contain one-cell black exits, colored contents inside the
    regions, and a bottom legend for exit counts.
    Rule: infer each grey-walled region's missing border cells as exits; odd
    exit counts erase internal colored contents, even counts keep them, and
    every exit cell is painted with the legend color for that count.
    Key insight: the same counted exits control two different output actions:
    parity changes the interior, while the exact count colors the openings.
    """
    out = grid.copy()
    grey = 5
    legend = [int(x) for x in grid[-1] if x not in (0, grey)]
    for comp in _components(grid == grey):
        r1, r2, c1, c2 = _bbox(comp)
        if r2 - r1 < 3 or c2 - c1 < 3:
            continue
        border = {(r1, c) for c in range(c1, c2 + 1)}
        border |= {(r2, c) for c in range(c1, c2 + 1)}
        border |= {(r, c1) for r in range(r1, r2 + 1)}
        border |= {(r, c2) for r in range(r1, r2 + 1)}
        exits = [(r, c) for r, c in border if grid[r, c] == 0]
        e = len(exits)
        if e % 2 == 1:
            inside = (grid[r1 + 1:r2, c1 + 1:c2] != 0) & (grid[r1 + 1:r2, c1 + 1:c2] != grey)
            out[r1 + 1:r2, c1 + 1:c2][inside] = 0
        if e < len(legend):
            for r, c in exits:
                out[r, c] = legend[e]
    return out


def generate_input() -> np.ndarray:
    grid = np.zeros((22, 26), dtype=int)
    legend = random.sample([1, 2, 3, 4, 6, 7, 8, 9], 5)
    grid[-1, :5] = legend
    positions = [(2, 2), (2, 14), (11, 3), (11, 15)]
    random.shuffle(positions)
    for r, c in positions[:random.randint(2, 4)]:
        h, w = random.randint(6, 8), random.randint(6, 8)
        grid[r:r + h, c] = 5
        grid[r:r + h, c + w - 1] = 5
        grid[r, c:c + w] = 5
        grid[r + h - 1, c:c + w] = 5
        exits = random.sample(["top", "bottom", "left", "right"], random.randint(0, 3))
        for side in exits:
            if side == "top":
                grid[r, random.randint(c + 1, c + w - 2)] = 0
            elif side == "bottom":
                grid[r + h - 1, random.randint(c + 1, c + w - 2)] = 0
            elif side == "left":
                grid[random.randint(r + 1, r + h - 2), c] = 0
            else:
                grid[random.randint(r + 1, r + h - 2), c + w - 1] = 0
        color = random.choice([1, 2, 3, 4, 6, 7, 8, 9])
        grid[r + 2:r + 4, c + 2] = color
        if random.random() < 0.5:
            grid[r + 3, c + 3] = color
    return grid
