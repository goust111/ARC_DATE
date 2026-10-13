import random
from collections import deque

import numpy as np


BG = 0
BLUE_MARK = 2
GREEN = 3
GRAY = 5


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for sr, sc in zip(*np.where(mask)):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        q = deque([(sr, sc)])
        seen[sr, sc] = True
        comp = []
        while q:
            r, c = q.popleft()
            comp.append((r, c))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(comp)
    return comps


def _bbox(coords):
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _quadrants(cross_r, cross_c, h, w):
    return [
        (0, cross_r, 0, cross_c),
        (0, cross_r, cross_c + 1, w),
        (cross_r + 1, h, 0, cross_c),
        (cross_r + 1, h, cross_c + 1, w),
    ]


def main(grid: np.ndarray) -> np.ndarray:
    """
    In each cross-separated quadrant, count the marker's green diagonal
    neighbors and rotate that local sprite by D*90 degrees clockwise.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    cross_rows = [r for r in range(h) if np.all(g[r, :] == GRAY)]
    cross_cols = [c for c in range(w) if np.all(g[:, c] == GRAY)]
    assert len(cross_rows) == 1 and len(cross_cols) == 1, "expected one gray cross"
    cr, cc = cross_rows[0], cross_cols[0]

    out = np.zeros_like(g)
    out[cr, :] = GRAY
    out[:, cc] = GRAY
    for r0, r1, c0, c1 in _quadrants(cr, cc, h, w):
        sub = g[r0:r1, c0:c1]
        marks = list(zip(*np.where(sub == BLUE_MARK)))
        assert len(marks) == 1, "each quadrant needs one red marker"
        mr, mc = int(marks[0][0]), int(marks[0][1])
        dcount = 0
        for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            nr, nc = mr + dr, mc + dc
            if 0 <= nr < sub.shape[0] and 0 <= nc < sub.shape[1] and sub[nr, nc] == GREEN:
                dcount += 1

        mask = (sub == GREEN) | (sub == BLUE_MARK)
        coords = [(int(r), int(c)) for r, c in zip(*np.where(mask))]
        br0, bc0, br1, bc1 = _bbox(coords)
        crop = mask[br0 : br1 + 1, bc0 : bc1 + 1]
        rot = np.rot90(crop, -dcount)
        rr, cc2 = rot.shape
        tr = r0 + (sub.shape[0] - rr) // 2
        tc = c0 + (sub.shape[1] - cc2) // 2
        out[tr : tr + rr, tc : tc + cc2][rot] = GREEN
    return out


SHAPES = {
    0: [(2, 2), (2, 1), (2, 0), (1, 0), (3, 0), (2, 3)],
    1: [(2, 2), (1, 1), (2, 1), (2, 0), (3, 0), (2, 3)],
    2: [(2, 2), (1, 1), (2, 1), (2, 3), (3, 3), (3, 2), (2, 0)],
    3: [(2, 2), (1, 1), (1, 3), (3, 1), (2, 1), (2, 3), (1, 2), (3, 2)],
}


def _place_shape(grid, quadrant, dcount):
    r0, r1, c0, c1 = quadrant
    shape = SHAPES[dcount]
    max_r = max(r for r, _ in shape)
    max_c = max(c for _, c in shape)
    for _ in range(80):
        tr = random.randint(r0, r1 - max_r - 1)
        tc = random.randint(c0, c1 - max_c - 1)
        cells = [(tr + r, tc + c) for r, c in shape]
        if all(grid[r, c] == BG for r, c in cells):
            for i, (r, c) in enumerate(cells):
                grid[r, c] = BLUE_MARK if i == 0 else GREEN
            return True
    return False


def _gen_attempt():
    q = random.randint(6, 8)
    h = w = 2 * q + 1
    cr = cc = q
    grid = np.zeros((h, w), dtype=int)
    grid[cr, :] = GRAY
    grid[:, cc] = GRAY
    counts = [0, 1, 2, 3]
    random.shuffle(counts)
    for quad, dcount in zip(_quadrants(cr, cc, h, w), counts):
        if not _place_shape(grid, quad, dcount):
            return None
    try:
        main(grid)
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
