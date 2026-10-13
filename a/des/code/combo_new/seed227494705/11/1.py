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


def _paint(out, r, c, color):
    if 0 <= r < out.shape[0] and 0 <= c < out.shape[1]:
        out[r, c] = color


def main(grid: np.ndarray) -> np.ndarray:
    """Project internal marker colors to the nearest outside edge of each rectangle.

    Input structure: black background with one-pixel closed rectangular frames
    of a shared frame color and one or more colored single-pixel markers inside
    each frame.
    Rule: for every marker, compare its distance to the four inner frame edges;
    paint the marker color one cell outside every nearest edge at the aligned
    row or column, preserving the original frame and marker.
    Key insight: marker position encodes direction by nearest-edge distance, and
    ties deliberately project to multiple sides.
    """
    out = grid.copy()
    colors = [int(x) for x in np.unique(grid) if x != 0]
    if not colors:
        return out
    frame_color = max(colors, key=lambda x: int(np.sum(grid == x)))
    for comp in _components(grid == frame_color):
        rs = [p[0] for p in comp]
        cs = [p[1] for p in comp]
        r1, r2, c1, c2 = min(rs), max(rs), min(cs), max(cs)
        if r2 - r1 < 3 or c2 - c1 < 3:
            continue
        if not (np.all(grid[r1, c1:c2 + 1] == frame_color) and np.all(grid[r2, c1:c2 + 1] == frame_color)):
            continue
        for r in range(r1 + 1, r2):
            for c in range(c1 + 1, c2):
                color = int(grid[r, c])
                if color in (0, frame_color):
                    continue
                dists = {"up": r - (r1 + 1), "down": (r2 - 1) - r, "left": c - (c1 + 1), "right": (c2 - 1) - c}
                m = min(dists.values())
                if dists["up"] == m:
                    _paint(out, r1 - 1, c, color)
                if dists["down"] == m:
                    _paint(out, r2 + 1, c, color)
                if dists["left"] == m:
                    _paint(out, r, c1 - 1, color)
                if dists["right"] == m:
                    _paint(out, r, c2 + 1, color)
    return out


def generate_input() -> np.ndarray:
    grid = np.zeros((22, 24), dtype=int)
    positions = [(3, 3), (3, 14), (12, 4), (12, 14)]
    random.shuffle(positions)
    for r, c in positions[:random.randint(2, 4)]:
        h, w = random.randint(5, 7), random.randint(5, 8)
        if r + h + 1 >= grid.shape[0] or c + w + 1 >= grid.shape[1]:
            continue
        grid[r:r + h, c] = 1
        grid[r:r + h, c + w - 1] = 1
        grid[r, c:c + w] = 1
        grid[r + h - 1, c:c + w] = 1
        for color in random.sample([2, 3, 4, 6, 7, 8, 9], random.randint(1, 3)):
            rr = random.randint(r + 1, r + h - 2)
            cc = random.randint(c + 1, c + w - 2)
            grid[rr, cc] = color
    return grid
