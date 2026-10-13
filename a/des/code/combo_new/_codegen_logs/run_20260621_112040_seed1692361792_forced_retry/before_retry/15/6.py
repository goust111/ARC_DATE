import random
from collections import Counter, deque

import numpy as np


BLACK = 0
GREY = 5


def _components(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(cells: np.ndarray):
    r0, c0 = cells.min(axis=0)
    r1, c1 = cells.max(axis=0)
    return r0, r1, c0, c1


def _mode(values):
    return Counter(values).most_common(1)[0][0]


def _scan_runs(values, baseline):
    runs = []
    start = None
    last_delta = 0
    for idx, val in enumerate(values):
        delta = val - baseline
        if delta == 0:
            if start is not None:
                runs.append((start, idx - 1, last_delta))
                start = None
            continue
        if start is None:
            start = idx
            last_delta = delta
        elif delta != last_delta:
            runs.append((start, idx - 1, last_delta))
            start = idx
            last_delta = delta
    if start is not None:
        runs.append((start, len(values) - 1, last_delta))
    return runs


def _scan_positive_runs(values):
    runs = []
    start = None
    last = 0
    for idx, val in enumerate(values):
        if val == 0:
            if start is not None:
                runs.append((start, idx - 1, last))
                start = None
            continue
        if start is None:
            start = idx
            last = val
        elif val != last:
            runs.append((start, idx - 1, last))
            start = idx
            last = val
    if start is not None:
        runs.append((start, len(values) - 1, last))
    return runs


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, dtype=int)
    out = grid.copy()

    all_grey = _components(grid == GREY)
    body = max(all_grey, key=len)
    br0, br1, bc0, bc1 = _bbox(body)
    body_mask = np.zeros_like(grid, dtype=bool)
    body_mask[body[:, 0], body[:, 1]] = True

    rows = range(br0, br1 + 1)
    row_mins = [int(np.where(body_mask[r])[0][0]) for r in rows]
    row_maxs = [int(np.where(body_mask[r])[0][-1]) for r in rows]

    left0 = min(row_mins)
    right0 = min(row_maxs)

    notches = {}
    protrusions = {}

    for s, e, delta in _scan_positive_runs([v - left0 for v in row_mins]):
        r0, r1 = br0 + s, br0 + e
        cells = [(r, c) for r in range(r0, r1 + 1) for c in range(left0, left0 + delta)]
        notches[(r1 - r0 + 1, delta)] = np.array(cells, dtype=int)
    for s, e, delta in _scan_positive_runs([v - right0 for v in row_maxs]):
        r0, r1 = br0 + s, br0 + e
        cells = [(r, c) for r in range(r0, r1 + 1) for c in range(right0 + 1, right0 + delta + 1)]
        protrusions[(r1 - r0 + 1, delta)] = np.array(cells, dtype=int)

    for color in sorted(set(grid.reshape(-1).tolist()) - {BLACK, GREY}):
        for comp in _components(grid == color):
            r0, r1, c0, c1 = _bbox(comp)
            size = (r1 - r0 + 1, c1 - c0 + 1)
            out[r0 : r1 + 1, c0 : c1 + 1] = BLACK
            notch = notches[size]
            protrusion = protrusions[size]
            out[notch[:, 0], notch[:, 1]] = color
            out[protrusion[:, 0], protrusion[:, 1]] = BLACK
    return out


def generate_input() -> np.ndarray:
    colors = [1, 2, 3, 4, 6, 7, 8, 9]
    sizes_pool = [(1, 2), (2, 1), (1, 3), (3, 1), (2, 2)]
    while True:
        h = random.randint(18, 22)
        w = random.randint(18, 22)
        grid = np.zeros((h, w), dtype=int)
        occupied = np.zeros((h, w), dtype=bool)

        top = random.randint(5, 7)
        left = random.randint(5, 7)
        bh = random.randint(8, 10)
        bw = random.randint(8, 10)
        grid[top : top + bh, left : left + bw] = GREY
        occupied[top : top + bh, left : left + bw] = True

        count = random.randint(2, 3)
        random.shuffle(sizes_pool)
        sizes = sizes_pool[:count]

        ok = True
        random.shuffle(colors)
        used_rows = []
        for color, size in zip(colors, sizes):
            sh, sw = size
            notch_placed = False
            for _ in range(200):
                nr = random.randint(top + 1, top + bh - sh - 1)
                if any(not (nr + sh + 1 <= a or nr >= b + 1) for a, b in used_rows):
                    continue
                nc = left
                area = (slice(nr, nr + sh), slice(nc, nc + sw))
                if np.any(~occupied[area]):
                    continue
                grid[area] = BLACK
                occupied[area] = False
                used_rows.append((nr, nr + sh))
                notch_placed = True
                break
            if not notch_placed:
                ok = False
                break

            placed = False
            for _ in range(100):
                pr = nr
                pc = left + bw
                if pr < 0 or pc < 0 or pr + sh > h or pc + sw > w:
                    continue
                if np.any(occupied[pr : pr + sh, pc : pc + sw]):
                    continue
                grid[pr : pr + sh, pc : pc + sw] = GREY
                occupied[pr : pr + sh, pc : pc + sw] = True
                placed = True
                break
            if not placed:
                ok = False
                break

            stamp_placed = False
            for _ in range(200):
                sr = random.randint(0, h - sh)
                sc = random.randint(0, w - sw)
                if np.any(occupied[sr : sr + sh, sc : sc + sw]):
                    continue
                grid[sr : sr + sh, sc : sc + sw] = color
                occupied[sr : sr + sh, sc : sc + sw] = True
                stamp_placed = True
                break
            if not stamp_placed:
                ok = False
                break
        if ok:
            return grid
