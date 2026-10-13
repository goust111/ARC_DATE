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


def main(grid: np.ndarray) -> np.ndarray:
    """Replace each marked object by a line whose direction and color come from its L marker.

    Input structure: black background with connected foreground objects; inside
    each object a differently colored L marker has a corner, a two-pixel long
    arm, and a one-pixel short arm.
    Rule: erase the whole object, identify the L corner and long-arm endpoint,
    then draw from the corner a straight segment in that direction whose length
    equals the erased object's area and whose color is the marker color.
    Key insight: the object's area becomes a one-dimensional length, while the
    marker's asymmetric L shape supplies the direction and output color.
    """
    out = grid.copy()
    for pts in _components(grid != 0):
        vals, counts = np.unique([grid[p] for p in pts], return_counts=True)
        if len(vals) < 2:
            continue
        marker_color = int(vals[np.argmin(counts)])
        marker = [p for p in pts if grid[p] == marker_color]
        for r, c in pts:
            out[r, c] = 0
        corner = None
        mset = set(marker)
        for r, c in marker:
            deg = sum((r + dr, c + dc) in mset for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)))
            if deg >= 2:
                corner = (r, c)
                break
        if corner is None:
            corner = marker[0]
        far = max(marker, key=lambda p: abs(p[0] - corner[0]) + abs(p[1] - corner[1]))
        dr = 0 if far[0] == corner[0] else (1 if far[0] > corner[0] else -1)
        dc = 0 if far[1] == corner[1] else (1 if far[1] > corner[1] else -1)
        for i in range(len(pts)):
            nr, nc = corner[0] + dr * i, corner[1] + dc * i
            if 0 <= nr < out.shape[0] and 0 <= nc < out.shape[1]:
                out[nr, nc] = marker_color
    return out


def generate_input() -> np.ndarray:
    grid = np.zeros((22, 24), dtype=int)
    bases = random.sample([1, 3, 4, 6, 7, 8], random.randint(2, 4))
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    for i, base in enumerate(bases):
        r, c = 3 + (i // 2) * 9, 3 + (i % 2) * 11
        h, w = random.randint(4, 5), random.randint(4, 5)
        grid[r:r + h, c:c + w] = base
        marker_color = random.choice([x for x in [2, 3, 4, 6, 7, 8, 9] if x != base])
        dr, dc = random.choice(dirs)
        short = (dc, dr) if dr != 0 else (1, 0)
        cr, cc = r + h // 2, c + w // 2
        cells = [(cr, cc), (cr + dr, cc + dc), (cr + 2 * dr, cc + 2 * dc), (cr + short[0], cc + short[1])]
        if all(r <= x < r + h and c <= y < c + w for x, y in cells):
            for x, y in cells:
                grid[x, y] = marker_color
    return grid
