import random
from collections import deque
import numpy as np


def _components(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for sr, sc in np.argwhere(mask):
        if seen[sr, sc]:
            continue
        q = deque([(int(sr), int(sc))])
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


def _crop(grid, comp):
    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
    return grid[r0:r1 + 1, c0:c1 + 1], (r0, r1, c0, c1)


def main(grid: np.ndarray) -> np.ndarray:
    red_pts = [(int(r), int(c)) for r, c in np.argwhere(grid == 2)]
    marker = None
    for r, c in red_pts:
        if not any(
            0 <= r + dr < grid.shape[0] and 0 <= c + dc < grid.shape[1] and grid[r + dr, c + dc] not in (0, 2)
            for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc
        ):
            marker = (r, c)
            break
    mask = grid != 0
    if marker:
        mask[marker] = False
    comps = _components(mask)
    by_template = {}
    info = []
    for comp in comps:
        tile, bbox = _crop(grid, comp)
        key = (tile.shape, tuple(tile.ravel()))
        by_template.setdefault(key, []).append(len(info))
        info.append((comp, tile, bbox))
    dup_ids = next((ids for ids in by_template.values() if len(ids) == 2), None)
    if marker is None or dup_ids is None:
        return out
    (_, tile, b1), (_, _tile2, b2) = info[dup_ids[0]], info[dup_ids[1]]
    c1 = ((b1[0] + b1[1]) / 2, (b1[2] + b1[3]) / 2)
    c2 = ((b2[0] + b2[1]) / 2, (b2[2] + b2[3]) / 2)
    mr, mc = marker
    if c1[0] == c2[0]:
        target = (int(round(2 * c1[0] - mr)), mc)
    else:
        target = (mr, int(round(2 * c1[1] - mc)))
    out = grid.copy()
    for idx in dup_ids:
        for r, c in info[idx][0]:
            out[r, c] = 0
    th, tw = tile.shape
    tr = int(round(target[0] - (th - 1) / 2))
    tc = int(round(target[1] - (tw - 1) / 2))
    for r in range(th):
        for c in range(tw):
            if tile[r, c] != 0:
                rr, cc = tr + r, tc + c
                if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]:
                    out[rr, cc] = 2
    if marker:
        out[marker] = 2
    return out


def _stamp(grid, top, left, tile):
    for r in range(tile.shape[0]):
        for c in range(tile.shape[1]):
            if tile[r, c]:
                grid[top + r, left + c] = tile[r, c]


def generate_input() -> np.ndarray:
    tile = np.array([[1, 0, 3], [1, 3, 3], [0, 4, 0]])
    other_tiles = [
        np.array([[5, 5], [0, 5], [8, 0]]),
        np.array([[8, 0, 8], [8, 8, 0]]),
        np.array([[4, 0], [4, 4], [0, 4]]),
    ]
    for _ in range(200):
        grid = np.zeros((15, 15), dtype=int)
        horizontal = random.random() < 0.5
        if horizontal:
            row = random.choice([3, 7, 10])
            pos = [(row, 2), (row, 9)]
            marker = (random.choice([1, 12]), random.randint(4, 10))
        else:
            col = random.choice([3, 7, 10])
            pos = [(2, col), (9, col)]
            marker = (random.randint(4, 10), random.choice([1, 13]))
        for top, left in pos:
            _stamp(grid, top, left, tile)
        for ot in random.sample(other_tiles, 2):
            for _try in range(50):
                top = random.randint(1, 12 - ot.shape[0])
                left = random.randint(1, 12 - ot.shape[1])
                cells = [(top + r, left + c) for r in range(ot.shape[0]) for c in range(ot.shape[1]) if ot[r, c]]
                pad = {
                    (r + dr, c + dc)
                    for r, c in cells
                    for dr in (-1, 0, 1)
                    for dc in (-1, 0, 1)
                }
                if all(0 <= r < 15 and 0 <= c < 15 and grid[r, c] == 0 for r, c in pad):
                    _stamp(grid, top, left, ot)
                    break
        mr, mc = marker
        clear = all(
            0 <= mr + dr < 15 and 0 <= mc + dc < 15 and grid[mr + dr, mc + dc] == 0
            for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)
        )
        if clear:
            grid[marker] = 2
            return grid
    return generate_input()
