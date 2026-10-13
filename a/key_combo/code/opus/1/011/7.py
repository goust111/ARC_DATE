import random
from collections import Counter

import numpy as np


def _find_region(grid):
    h, w = grid.shape
    for r in range(h - 10):
        for c in range(w - 10):
            block = grid[r:r + 11, c:c + 11]
            if np.all(block[5, :] == 5) and np.all(block[:, 5] == 5):
                return r, c
    raise ValueError("region not found")


def _variant(q, r, c):
    if q == 0:
        return r, c
    if q == 1:
        return r, 4 - c
    if q == 2:
        return 4 - r, c
    return 4 - r, 4 - c


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, copy=True)
    rr, cc = _find_region(grid)
    mask = None
    for r in range(grid.shape[0] - 2):
        for c in range(grid.shape[1] - 2):
            if r <= rr + 10 and rr <= r + 2 and c <= cc + 10 and cc <= c + 2:
                continue
            b = grid[r:r + 3, c:c + 3]
            vals = [int(x) for x in b.ravel() if x != 0 and x != 5]
            if len(vals) == 9:
                mask = b
                break
        if mask is not None:
            break
    counts = Counter(int(x) for x in mask.ravel())
    cm = counts.most_common(1)[0][0]
    selected = {(0, 0): 0, (0, 1): 1, (1, 0): 2, (1, 1): 3}
    chosen = {selected[(r, c)] for r in range(2) for c in range(2) if mask[r, c] == cm}
    quad_slices = [
        (slice(rr, rr + 5), slice(cc, cc + 5)),
        (slice(rr, rr + 5), slice(cc + 6, cc + 11)),
        (slice(rr + 6, rr + 11), slice(cc, cc + 5)),
        (slice(rr + 6, rr + 11), slice(cc + 6, cc + 11)),
    ]
    quads = [grid[s] for s in quad_slices]
    source = np.zeros((5, 5), dtype=int)
    for r in range(5):
        for c in range(5):
            for qi, q in enumerate(quads):
                vr, vc = _variant(qi, r, c)
                if q[vr, vc] != 0:
                    source[r, c] = q[vr, vc]
                    break
    out = np.zeros((11, 11), dtype=int)
    out[5, :] = 5
    out[:, 5] = 5
    variants = [source, np.fliplr(source), np.flipud(source), np.flipud(np.fliplr(source))]
    positions = [(0, 0), (0, 6), (6, 0), (6, 6)]
    for qi, (r0, c0) in enumerate(positions):
        tile = variants[qi].copy()
        if qi in chosen:
            tile[tile != 0] = cm
        out[r0:r0 + 5, c0:c0 + 5] = tile
    return out


def generate_input() -> np.ndarray:
    grid = np.zeros((16, 16), dtype=int)
    colors = [2, 3, 4, 6, 7, 8]
    source = np.zeros((5, 5), dtype=int)
    cells = random.sample([(r, c) for r in range(5) for c in range(5)], random.randint(9, 13))
    for r, c in cells:
        source[r, c] = random.choice(colors[:4])
    variants = [source, np.fliplr(source), np.flipud(source), np.flipud(np.fliplr(source))]
    erased = [set() for _ in range(4)]
    for qi in range(4):
        pts = [(r, c) for r in range(5) for c in range(5) if variants[qi][r, c] != 0]
        erased[qi] = set(random.sample(pts, min(random.randint(1, 3), len(pts))))
    for r in range(5):
        for c in range(5):
            if source[r, c] == 0:
                continue
            if all(_variant(qi, r, c) in erased[qi] for qi in range(4)):
                qi = random.randrange(4)
                erased[qi].remove(_variant(qi, r, c))
    rr, cc = 5, 5
    grid[rr + 5, cc:cc + 11] = 5
    grid[rr:rr + 11, cc + 5] = 5
    pos = [(rr, cc), (rr, cc + 6), (rr + 6, cc), (rr + 6, cc + 6)]
    for qi, (r0, c0) in enumerate(pos):
        tile = variants[qi].copy()
        for p in erased[qi]:
            tile[p] = 0
        grid[r0:r0 + 5, c0:c0 + 5] = tile
    cm = random.choice([1, 7, 8, 9])
    mask = np.full((3, 3), random.choice([2, 3, 4, 6]), dtype=int)
    slots = random.sample([(0, 0), (0, 1), (1, 0), (1, 1)], random.randint(2, 3))
    extra = random.sample([(r, c) for r in range(3) for c in range(3) if (r, c) not in slots], 3 - len(slots))
    for r, c in slots + extra:
        mask[r, c] = cm
    fill_colors = [x for x in [2, 3, 4, 6, 7, 8, 9] if x != cm]
    random.shuffle(fill_colors)
    fill_cycle = (fill_colors * 2)[:9]
    idx = 0
    for r in range(3):
        for c in range(3):
            if mask[r, c] != cm:
                mask[r, c] = fill_cycle[idx]
                idx += 1
    grid[0:3, 0:3] = mask
    return grid
