import random
from collections import deque
import numpy as np


def _components(grid, color):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or grid[r, c] != color:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp = []
            while q:
                x, y = q.popleft()
                comp.append((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and not seen[nx, ny] and grid[nx, ny] == color:
                        seen[nx, ny] = True
                        q.append((nx, ny))
            comps.append(comp)
    return comps


def _bbox(comp):
    rs = [r for r, _ in comp]
    cs = [c for _, c in comp]
    return min(rs), min(cs), max(rs), max(cs)


def _black_slots(grid, bbox):
    r0, c0, r1, c1 = bbox
    mask = grid[r0:r1 + 1, c0:c1 + 1] == 0
    comps = _components(mask.astype(int), 1)
    slots = []
    for comp in comps:
        br0, bc0, br1, bc1 = _bbox(comp)
        if br0 > 0 and bc0 > 0 and br1 < mask.shape[0] - 1 and bc1 < mask.shape[1] - 1:
            slots.append([(r0 + r, c0 + c) for r, c in comp])
    slots.sort(key=lambda comp: _bbox(comp)[0])
    return slots


def main(grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, copy=True)
    sep_cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == 5)]
    if not sep_cols:
        return out
    sep = sep_cols[0]
    left = grid[:, :sep]
    filled = []
    for comp in _components(left, 1):
        r0, c0, r1, c1 = _bbox(comp)
        gr = min(2, int(((r0 + r1) / 2) * 3 / left.shape[0]))
        gc = min(2, int(((c0 + c1) / 2) * 3 / left.shape[1]))
        filled.append(gr * 3 + gc)
    chosen_count = min(3, len(sorted(set(filled))))

    yellow = [p for comp in _components(grid, 4) for p in comp]
    if not yellow:
        return out
    ybox = _bbox(yellow)
    slots = _black_slots(grid, ybox)[:chosen_count]
    red_comps = _components(grid, 2)
    used = set()
    for slot in slots:
        sr0, sc0, sr1, sc1 = _bbox(slot)
        center = ((sr0 + sr1) / 2, (sc0 + sc1) / 2)
        best = None
        for i, comp in enumerate(red_comps):
            if i in used:
                continue
            cr0, cc0, cr1, cc1 = _bbox(comp)
            aligned = (cr0 <= center[0] <= cr1) or (cc0 <= center[1] <= cc1)
            if aligned:
                dist = abs((cr0 + cr1) / 2 - center[0]) + abs((cc0 + cc1) / 2 - center[1])
                if best is None or dist < best[0]:
                    best = (dist, i, comp)
        if best is None:
            continue
        used.add(best[1])
        comp = best[2]
        for r, c in comp:
            out[r, c] = 0
        cr0, cc0, cr1, cc1 = _bbox(comp)
        dr, dc = sr0 - cr0, sc0 - cc0
        for r, c in comp:
            nr, nc = r + dr, c + dc
            if sr0 <= nr <= sr1 and sc0 <= nc <= sc1:
                out[nr, nc] = 2
    return out


def generate_input() -> np.ndarray:
    h, w, sep = 12, 25, 12
    g = np.zeros((h, w), dtype=int)
    g[:, sep] = 5
    filled = random.sample(range(9), random.randint(3, 6))
    for idx in filled:
        r, c = divmod(idx, 3)
        rr, cc = 1 + r * 3, 1 + c * 3
        g[rr:rr + 2, cc:cc + 2] = 1
    for _ in range(random.randint(1, 3)):
        g[random.randint(0, h - 1), random.randint(0, sep - 1)] = 2

    g[1:11, 15:24] = 4
    slots = [(2, 18), (5, 18), (8, 18)]
    for r, c in slots:
        g[r:r + 2, c:c + 3] = 0
    comps = [[(0, 0), (0, 1), (1, 0)], [(0, 0), (1, 0), (1, 1)], [(0, 0), (0, 1), (1, 1)]]
    starts = [(2, 13), (5, 13), (8, 13)]
    for shape, (r0, c0) in zip(comps, starts):
        for r, c in shape:
            g[r0 + r, c0 + c] = 2
    return g
