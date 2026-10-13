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


def _in_bounds(out, r, c, color):
    if 0 <= r < out.shape[0] and 0 <= c < out.shape[1]:
        out[r, c] = color


def main(grid: np.ndarray) -> np.ndarray:
    """Grow colored branch motifs from seed positions inside square grey frames.

    Input structure: black background with one or more hollow grey square frames;
    each frame has exactly one non-black, non-grey seed in one of the four inner
    quadrant positions.
    Rule: keep the frames and seeds, map the seed quadrant to the matching
    diagonal outside corner, then draw a same-colored diagonal branch whose
    length is the frame interior width and whose diagonal pixels sprout one
    horizontal and one vertical side pixel in the same outward directions.
    Key insight: the seed's tiny offset inside the frame chooses the outward
    diagonal, while the frame size chooses how long the branch grows.
    """
    out = grid.copy()
    grey = 5
    for comp in _components(grid == grey):
        rows = [p[0] for p in comp]
        cols = [p[1] for p in comp]
        r1, r2, c1, c2 = min(rows), max(rows), min(cols), max(cols)
        if r2 - r1 < 2 or r2 - r1 != c2 - c1:
            continue
        if not (np.all(grid[r1, c1:c2 + 1] == grey) and np.all(grid[r2, c1:c2 + 1] == grey)):
            continue
        if not (np.all(grid[r1:r2 + 1, c1] == grey) and np.all(grid[r1:r2 + 1, c2] == grey)):
            continue
        inner = grid[r1 + 1:r2, c1 + 1:c2]
        seeds = np.argwhere((inner != 0) & (inner != grey))
        if len(seeds) != 1:
            continue
        sr, sc = seeds[0]
        color = int(inner[sr, sc])
        dr = -1 if sr < inner.shape[0] / 2 else 1
        dc = -1 if sc < inner.shape[1] / 2 else 1
        cr = r1 if dr < 0 else r2
        cc = c1 if dc < 0 else c2
        for step in range(1, inner.shape[0] + 1):
            rr, cc2 = cr + dr * step, cc + dc * step
            _in_bounds(out, rr, cc2, color)
            _in_bounds(out, rr + dr, cc2, color)
            _in_bounds(out, rr, cc2 + dc, color)
    return out


def generate_input() -> np.ndarray:
    h = random.randint(15, 26)
    w = random.randint(15, 26)
    grid = np.zeros((h, w), dtype=int)
    colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], random.randint(2, 4))
    slots = [(3, 3), (3, w - 7), (h - 7, 3), (h - 7, w - 7), (h // 2 - 2, w // 2 - 2)]
    random.shuffle(slots)
    for color, (r, c) in zip(colors, slots):
        if r < 1 or c < 1 or r + 4 >= h - 1 or c + 4 >= w - 1:
            continue
        grid[r:r + 4, c] = 5
        grid[r:r + 4, c + 3] = 5
        grid[r, c:c + 4] = 5
        grid[r + 3, c:c + 4] = 5
        sr, sc = random.choice([(0, 0), (0, 1), (1, 0), (1, 1)])
        grid[r + 1 + sr, c + 1 + sc] = color
    return grid
