import random
from collections import deque

import numpy as np


def _components(mask):
    seen = np.zeros(mask.shape, dtype=bool)
    comps = []
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        q = deque([(sr, sc)])
        seen[sr, sc] = True
        comp = []
        while q:
            r, c = q.popleft()
            comp.append((r, c))
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(comp)
    return comps


def _bbox(comp):
    rs = [p[0] for p in comp]
    cs = [p[1] for p in comp]
    return min(rs), max(rs) + 1, min(cs), max(cs) + 1


def main(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    non_gray = (grid != 0) & (grid != 5)
    candidates = []
    for comp in _components(non_gray):
        vals = {int(grid[r, c]) for r, c in comp}
        if len(comp) > 1 and len(vals) > 1:
            candidates.append(comp)
    source = max(candidates, key=len)
    r0, r1, c0, c1 = _bbox(source)
    template = grid[r0:r1, c0:c1]
    markers = [tuple(map(int, p)) for p in np.argwhere(non_gray) if tuple(map(int, p)) not in set(source)]

    for rect in _components(grid == 5):
        rr0, rr1, cc0, cc1 = _bbox(rect)
        center = ((rr0 + rr1 - 1) / 2, (cc0 + cc1 - 1) / 2)
        mr, mc = min(markers, key=lambda p: abs(p[0] - center[0]) + abs(p[1] - center[1]))
        if mr < rr0:
            k = 0
        elif mc >= cc1:
            k = 1
        elif mr >= rr1:
            k = 2
        else:
            k = 3
        stamp = np.rot90(template, -k)
        color = int(grid[mr, mc])
        out[rr0:rr1, cc0:cc1] = np.where(stamp != 0, color, 0)
        out[mr, mc] = 0
    return out


def _can_place(occ, r, c, h, w):
    for rr in range(r - 1, r + h + 1):
        for cc in range(c - 1, c + w + 1):
            if (rr, cc) in occ:
                return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(200):
        grid = np.zeros((16, 16), dtype=int)
        occ = set()
        template = np.array([[1, 0, 2, 0], [1, 1, 2, 0], [0, 1, 2, 2], [0, 0, 0, 2]], dtype=int)
        sr, sc = random.randint(0, 12), random.randint(0, 12)
        grid[sr : sr + 4, sc : sc + 4] = template
        occ.update((r, c) for r in range(sr - 1, sr + 5) for c in range(sc - 1, sc + 5))
        colors = random.sample([1, 2, 3, 4, 6], random.randint(3, 5))
        dirs = random.choices(["up", "right", "down", "left"], k=len(colors))
        ok = True
        for color, d in zip(colors, dirs):
            placed = False
            spots = [(r, c) for r in range(16 - 4) for c in range(16 - 4)]
            random.shuffle(spots)
            for r, c in spots:
                if d == "up":
                    mr, mc = r - 1, c + 2
                elif d == "right":
                    mr, mc = r + 2, c + 4
                elif d == "down":
                    mr, mc = r + 4, c + 2
                else:
                    mr, mc = r + 2, c - 1
                if not (0 <= mr < 16 and 0 <= mc < 16):
                    continue
                if _can_place(occ, r, c, 4, 4) and (mr, mc) not in occ:
                    grid[r : r + 4, c : c + 4] = 5
                    grid[mr, mc] = color
                    occ.update((rr, cc) for rr in range(r - 1, r + 5) for cc in range(c - 1, c + 5))
                    occ.add((mr, mc))
                    placed = True
                    break
            if not placed:
                ok = False
                break
        if ok:
            return grid
    return grid
