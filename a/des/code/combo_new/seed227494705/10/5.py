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
    """Color nested grey rings by the number of colored objects visible inside them.

    Input structure: black background with separated concentric grey rectangular
    rings, small colored objects inside, and a visible color legend row near the
    upper-left corner.
    Rule: sort rings from inner to outer, count independent non-grey colored
    objects inside each ring's bounding box, and recolor that ring to the
    one-based legend color selected by the cumulative count.
    Key insight: an outer ring looks inward and uses the total object count it
    encloses, not a local ring-only feature.
    """
    out = grid.copy()
    grey = 5
    legend = [int(x) for x in grid[0] if x not in (0, grey)]
    rings = []
    for comp in _components(grid == grey):
        r1, r2, c1, c2 = _bbox(comp)
        if r2 - r1 >= 2 and c2 - c1 >= 2:
            rings.append((r1, r2, c1, c2, comp))
    rings.sort(key=lambda x: (x[1] - x[0]) * (x[3] - x[2]))
    obj_mask = (grid != 0) & (grid != grey)
    obj_mask[0, :] = False
    objects = _components(obj_mask)
    for r1, r2, c1, c2, comp in rings:
        count = 0
        for pts in objects:
            pr = sum(p[0] for p in pts) / len(pts)
            pc = sum(p[1] for p in pts) / len(pts)
            if r1 < pr < r2 and c1 < pc < c2:
                count += 1
        if count and count <= len(legend):
            for r, c in comp:
                out[r, c] = legend[count - 1]
    return out


def generate_input() -> np.ndarray:
    grid = np.zeros((24, 24), dtype=int)
    legend = [1, 2, 3, 4, 6, 7, 8, 9]
    random.shuffle(legend)
    nobj = random.randint(2, 5)
    grid[0, :nobj + 1] = legend[:nobj + 1]
    layers = 2
    r1, c1, r2, c2 = 5, 5, 18, 18
    for _ in range(layers):
        grid[r1, c1:c2 + 1] = 5
        grid[r2, c1:c2 + 1] = 5
        grid[r1:r2 + 1, c1] = 5
        grid[r1:r2 + 1, c2] = 5
        r1 += 2
        c1 += 2
        r2 -= 2
        c2 -= 2
    colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], nobj)
    for color in colors:
        for _ in range(100):
            r = random.randint(r1 + 1, r2 - 1)
            c = random.randint(c1 + 1, c2 - 1)
            shape = random.choice([[(0, 0), (0, 1)], [(0, 0), (1, 0), (1, 1)], [(0, 0), (1, 0)]])
            if all(grid[r + a, c + b] == 0 for a, b in shape):
                for a, b in shape:
                    grid[r + a, c + b] = color
                break
    return grid
