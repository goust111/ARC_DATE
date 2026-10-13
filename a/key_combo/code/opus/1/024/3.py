import random
from collections import deque
import numpy as np


def _components(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, copy=True)
    h, w = grid.shape
    res = grid.copy()
    markers = {}
    for r in range(h):
        for c in range(w):
            if r in (0, h - 1) or c in (0, w - 1):
                color = int(grid[r, c])
                if color:
                    if r == 0:
                        markers[color] = (1, 0)
                    elif r == h - 1:
                        markers[color] = (-1, 0)
                    elif c == 0:
                        markers[color] = (0, 1)
                    else:
                        markers[color] = (0, -1)
    for color, (dr, dc) in markers.items():
        inside = grid.copy()
        inside[0, :] = inside[-1, :] = inside[:, 0] = inside[:, -1] = 0
        comps = _components(inside == color)
        if not comps:
            continue
        comp = max(comps, key=len)
        rows = [p[0] for p in comp]
        cols = [p[1] for p in comp]
        length = len(comp)
        if dr:
            row = min(rows) if dr > 0 else max(rows)
            c0, c1 = min(cols), max(cols)
            step = dr * (2 * length - 1)
            r = row + step
            while 0 <= r < h:
                for c in range(c0, c1 + 1):
                    if res[r, c] == 0:
                        res[r, c] = color
                r += step
        else:
            col = min(cols) if dc > 0 else max(cols)
            r0, r1 = min(rows), max(rows)
            step = dc * (2 * length - 1)
            c = col + step
            while 0 <= c < w:
                for r in range(r0, r1 + 1):
                    if res[r, c] == 0:
                        res[r, c] = color
                c += step
    return res


def generate_input() -> np.ndarray:
    h = w = random.choice([12, 13, 14])
    g = np.zeros((h, w), dtype=int)
    colors = random.sample([1, 2, 3, 4], random.randint(3, 4))
    used = set()
    for color in colors:
        orient = random.choice(["h", "v"])
        length = random.randint(2, 4)
        for _ in range(200):
            if orient == "h":
                r = random.randrange(2, h - 2)
                c = random.randrange(2, w - length - 1)
                cells = [(r, cc) for cc in range(c, c + length)]
            else:
                r = random.randrange(2, h - length - 1)
                c = random.randrange(2, w - 2)
                cells = [(rr, c) for rr in range(r, r + length)]
            halo = {(rr + dr, cc + dc) for rr, cc in cells for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
            if not (halo & used):
                break
        for p in cells:
            g[p] = color
        used |= halo
        if random.random() < 0.8:
            if orient == "h":
                side = random.choice(["left", "right"])
                mr = cells[0][0]
                mc = 0 if side == "left" else w - 1
            else:
                side = random.choice(["top", "bottom"])
                mr = 0 if side == "top" else h - 1
                mc = cells[0][1]
            g[mr, mc] = color
    return g
