import random
import numpy as np


def _bbox(points):
    rs = [p[0] for p in points]
    cs = [p[1] for p in points]
    return min(rs), min(cs), max(rs), max(cs)


def _components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp = []
            while stack:
                x, y = stack.pop()
                comp.append((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] and not seen[nx, ny]:
                        seen[nx, ny] = True
                        stack.append((nx, ny))
            comps.append(comp)
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, copy=True)
    h, w = grid.shape
    sep_color = None
    sep_r = sep_c = None
    for color in np.unique(grid):
        rows = np.where(np.all(grid == color, axis=1))[0]
        cols = np.where(np.all(grid == color, axis=0))[0]
        if len(rows) and len(cols):
            sep_color = int(color)
            sep_r, sep_c = int(rows[0]), int(cols[0])
            break
    if sep_color is None:
        return grid

    row_ranges = [(0, sep_r), (sep_r + 1, h)]
    col_ranges = [(0, sep_c), (sep_c + 1, w)]
    quadrants = {}
    for qr, (r0, r1) in enumerate(row_ranges):
        for qc, (c0, c1) in enumerate(col_ranges):
            quadrants[(qr, qc)] = (r0, r1, c0, c1)

    source = None
    source_pixels = []
    for q, (r0, r1, c0, c1) in quadrants.items():
        comps = _components(grid[r0:r1, c0:c1] == 1)
        if comps:
            comp = max(comps, key=len)
            if len(comp) > 1:
                source = q
                source_pixels = [(r + r0, c + c0) for r, c in comp]
                break
    if source is None:
        return grid

    sr0, sr1, sc0, sc1 = quadrants[source]
    br0, bc0, br1, bc1 = _bbox(source_pixels)
    sprite = np.zeros((br1 - br0 + 1, bc1 - bc0 + 1), bool)
    for r, c in source_pixels:
        sprite[r - br0, c - bc0] = True
    dr, dc = br0 - sr0, bc0 - sc0
    sh, sw = sprite.shape
    qh, qw = sr1 - sr0, sc1 - sc0

    out = grid.copy()
    for q, (r0, r1, c0, c1) in quadrants.items():
        if q == source or not np.any(grid[r0:r1, c0:c1] == 2):
            continue
        qr, qc = q
        sr, sc = source
        shape = sprite
        tr, tc = dr, dc
        if qr != sr:
            shape = shape[::-1, :]
            tr = qh - dr - sh
        if qc != sc:
            shape = shape[:, ::-1]
            tc = qw - dc - sw
        for rr, cc in zip(*np.where(shape)):
            gr, gc = r0 + tr + rr, c0 + tc + cc
            if 0 <= gr < r1 and 0 <= gc < c1 and out[gr, gc] != 2:
                out[gr, gc] = 1
    return out


def _random_sprite(n):
    cells = {(0, 0)}
    while len(cells) < n:
        r, c = random.choice(tuple(cells))
        dr, dc = random.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
        cells.add((r + dr, c + dc))
    minr = min(r for r, _ in cells)
    minc = min(c for _, c in cells)
    return [(r - minr, c - minc) for r, c in cells]


def generate_input() -> np.ndarray:
    qh = random.randint(6, 8)
    qw = random.randint(6, 8)
    h, w = qh * 2 + 1, qw * 2 + 1
    grid = np.zeros((h, w), dtype=int)
    grid[qh, :] = 5
    grid[:, qw] = 5
    quadrants = {
        (0, 0): (0, qh, 0, qw),
        (0, 1): (0, qh, qw + 1, w),
        (1, 0): (qh + 1, h, 0, qw),
        (1, 1): (qh + 1, h, qw + 1, w),
    }
    source = random.choice(list(quadrants))
    while True:
        sprite = _random_sprite(random.randint(3, 6))
        sh = max(r for r, _ in sprite) + 1
        sw = max(c for _, c in sprite) + 1
        if sh <= qh - 2 and sw <= qw - 2:
            break
    r0, r1, c0, c1 = quadrants[source]
    dr = random.randint(1, qh - sh - 1)
    dc = random.randint(1, qw - sw - 1)
    for r, c in sprite:
        grid[r0 + dr + r, c0 + dc + c] = 1
    targets = [q for q in quadrants if q != source]
    k = random.randint(1, 3)
    for q in random.sample(targets, k):
        r0, r1, c0, c1 = quadrants[q]
        while True:
            rr = random.randint(r0 + 1, r1 - 2)
            cc = random.randint(c0 + 1, c1 - 2)
            if grid[rr, cc] == 0:
                grid[rr, cc] = 2
                break
    return grid
