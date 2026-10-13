import random
from collections import Counter
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


def _pattern(r, c, a, b, k, phase):
    return a if ((r // k + c // k + phase) % 2 == 0) else b


def _symmetric(comp):
    rs = [p[0] for p in comp]
    cs = [p[1] for p in comp]
    r1, r2, c1, c2 = min(rs), max(rs), min(cs), max(cs)
    s = {(r - r1, c - c1) for r, c in comp}
    h, w = r2 - r1 + 1, c2 - c1 + 1
    return s == {(h - 1 - r, w - 1 - c) for r, c in s}


def main(grid: np.ndarray) -> np.ndarray:
    """Repair checkerboard cells under objects, inverting phase for asymmetric objects.

    Input structure: a two-color checkerboard of block period one to three is
    partially covered by distinct-color foreground objects.
    Rule: infer the two checker colors, period, and phase from visible cells;
    remove each foreground object; fill symmetric objects with the normal
    checkerboard and asymmetric objects with the opposite checker color.
    Key insight: global checker phase is inferred from unobscured cells, while
    each object's 180-degree symmetry selects normal or inverted repair.
    """
    counts = Counter(int(x) for x in grid.ravel())
    a, b = [x for x, _ in counts.most_common(2)]
    visible = np.argwhere((grid == a) | (grid == b))
    best = (None, None, None, -1)
    for k in (1, 2, 3):
        for phase in (0, 1):
            for aa, bb in ((a, b), (b, a)):
                score = sum(grid[r, c] == _pattern(int(r), int(c), aa, bb, k, phase) for r, c in visible)
                if score > best[3]:
                    best = (aa, bb, k, phase, score)
    a, b, k, phase, _ = best
    out = grid.copy()
    for comp in _components((grid != a) & (grid != b)):
        invert = not _symmetric(comp)
        for r, c in comp:
            normal = _pattern(r, c, a, b, k, phase)
            out[r, c] = b if normal == a and invert else a if normal == b and invert else normal
    return out


def generate_input() -> np.ndarray:
    h, w = random.randint(12, 20), random.randint(12, 20)
    a, b = random.sample([1, 2, 3, 4, 6, 7, 8, 9], 2)
    k = random.randint(1, 3)
    phase = random.randint(0, 1)
    grid = np.zeros((h, w), dtype=int)
    for r in range(h):
        for c in range(w):
            grid[r, c] = _pattern(r, c, a, b, k, phase)
    shapes = [[(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)], [(0, 0), (0, 1), (1, 1), (2, 1)], [(0, 0), (0, 2), (1, 1), (2, 0), (2, 2)]]
    colors = [x for x in [1, 2, 3, 4, 6, 7, 8, 9] if x not in (a, b)]
    placed = np.zeros((h, w), bool)
    for color in random.sample(colors, random.randint(2, 4)):
        shape = random.choice(shapes)
        for _ in range(100):
            r, c = random.randint(1, h - 4), random.randint(1, w - 4)
            cells = [(r + dr, c + dc) for dr, dc in shape]
            if all(not placed[x, y] for x, y in cells):
                for x, y in cells:
                    grid[x, y] = color
                    placed[x, y] = True
                break
    return grid
