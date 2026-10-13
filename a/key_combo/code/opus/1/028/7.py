import random
from collections import Counter, deque
import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _repair_periodic(a):
    h, w = a.shape
    best = None
    limit = max(1, (h * w) // 8)
    for area in range(1, (h // 2) * (w // 2) + 1):
        for th in range(1, h // 2 + 1):
            if area % th:
                continue
            tw = area // th
            if tw < 1 or tw > w // 2:
                continue
            tile = np.zeros((th, tw), dtype=a.dtype)
            mismatch = 0
            for rr in range(th):
                for cc in range(tw):
                    vals = [int(a[r, c]) for r in range(rr, h, th) for c in range(cc, w, tw)]
                    val = Counter(vals).most_common(1)[0][0]
                    tile[rr, cc] = val
                    mismatch += sum(v != val for v in vals)
            if mismatch <= limit:
                best = tile
                break
        if best is not None:
            break
    if best is None:
        best = a.copy()
    th, tw = best.shape
    out = np.zeros_like(a)
    for r in range(h):
        for c in range(w):
            out[r, c] = best[r % th, c % tw]
    return out


def _yellow_components(g):
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) != 4:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                x, y = q.popleft()
                cells.append((x, y))
                for dx, dy in DIR4:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and not seen[nx, ny] and int(g[nx, ny]) == 4:
                        seen[nx, ny] = True
                        q.append((nx, ny))
            comps.append(cells)
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, copy=True)
    h, w = g.shape
    sep_cols = [c for c in range(w) if np.all(g[:, c] == 5)]
    sep = sep_cols[0] if sep_cols else w // 2
    repaired = _repair_periodic(g[:, :sep])
    out = g.copy()
    out[:, :sep] = repaired
    for comp in _yellow_components(g):
        rs = [r for r, _ in comp]
        cs = [c for _, c in comp]
        top, bottom, left, right = min(rs), max(rs), min(cs), max(cs)
        opening_rows = range(top + 1, bottom)
        for r in opening_rows:
            for c in range(left, right):
                if out[r, c] == 0:
                    out[r, c] = 4
            for c in range(0, left):
                out[r, c] = repaired[r, c % sep]
    return out


def generate_input() -> np.ndarray:
    h = random.randint(16, 20)
    left_w = random.choice([9, 10])
    w = left_w + 10
    g = np.zeros((h, w), dtype=int)
    tiles = [
        np.array([[1, 0, 2], [0, 3, 0], [2, 0, 1]], dtype=int),
        np.array([[0, 1, 0], [2, 3, 1], [0, 2, 0]], dtype=int),
        np.array([[3, 0, 1], [0, 2, 0], [1, 0, 3]], dtype=int),
    ]
    tile = random.choice(tiles)
    for r in range(h):
        for c in range(left_w):
            g[r, c] = tile[r % 3, c % 3]
    for _ in range(random.randint(3, 5)):
        r = random.randrange(h)
        c = random.randrange(left_w)
        choices = [0, 1, 2, 3]
        choices.remove(int(g[r, c]))
        g[r, c] = random.choice(choices)
    g[:, left_w] = 5
    cup_rows = random.sample(range(1, h - 5), random.choice([2, 3]))
    cup_rows.sort()
    used = []
    for top in cup_rows:
        if any(abs(top - u) < 5 for u in used):
            continue
        used.append(top)
        left = random.randint(left_w + 3, w - 5)
        hh, ww = 4, 4
        g[top, left:left + ww] = 4
        g[top + hh - 1, left:left + ww] = 4
        g[top:top + hh, left + ww - 1] = 4
    return g
