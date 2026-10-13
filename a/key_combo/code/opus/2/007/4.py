import random
from collections import deque
import numpy as np


def _components(mask):
    seen = np.zeros(mask.shape, dtype=bool)
    comps = []
    for r, c in np.argwhere(mask):
        if seen[r, c]:
            continue
        q = deque([(int(r), int(c))])
        seen[r, c] = True
        comp = []
        while q:
            x, y = q.popleft()
            comp.append((x, y))
            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if 0 <= nx < mask.shape[0] and 0 <= ny < mask.shape[1] and mask[nx, ny] and not seen[nx, ny]:
                    seen[nx, ny] = True
                    q.append((nx, ny))
        comps.append(comp)
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    nines = np.argwhere(grid == 9)
    dists = {
        "top": int(nines[:, 0].min()),
        "bottom": int((h - 1 - nines[:, 0]).min()),
        "left": int(nines[:, 1].min()),
        "right": int((w - 1 - nines[:, 1]).min()),
    }
    edge = min(dists, key=dists.get)
    band = np.zeros_like(grid, dtype=bool)
    if edge == "top":
        band[:2, :] = True
    elif edge == "bottom":
        band[-2:, :] = True
    elif edge == "left":
        band[:, :2] = True
    else:
        band[:, -2:] = True

    rects = []
    for comp in _components((grid == 7) & ~band):
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        rects.append((min(rows), max(rows), min(cols), max(cols)))

    counts = [0] * len(rects)
    for r, c in np.argwhere(grid == 1):
        for i, (r0, r1, c0, c1) in enumerate(rects):
            if edge in ("left", "right") and r0 <= r <= r1:
                counts[i] += 1
                break
            if edge in ("top", "bottom") and c0 <= c <= c1:
                counts[i] += 1
                break

    out = grid.copy()
    out[(grid == 1)] = 0
    out[band & (grid == 7)] = 0
    for (r0, r1, c0, c1), n in zip(rects, counts):
        if edge == "top":
            out[r0:r0 + n, c0:c1 + 1] = 1
        elif edge == "bottom":
            out[r1 - n + 1:r1 + 1, c0:c1 + 1] = 1
        elif edge == "left":
            out[r0:r1 + 1, c0:c0 + n] = 1
        else:
            out[r0:r1 + 1, c1 - n + 1:c1 + 1] = 1
    return out


def generate_input() -> np.ndarray:
    for _ in range(2000):
        h = w = random.randint(14, 16)
        edge = random.choice(["top", "bottom", "left", "right"])
        grid = np.zeros((h, w), dtype=int)
        if edge == "top":
            grid[:2, :] = 7
            spots = [(r, c) for r in range(2) for c in range(w)]
        elif edge == "bottom":
            grid[-2:, :] = 7
            spots = [(r, c) for r in range(h - 2, h) for c in range(w)]
        elif edge == "left":
            grid[:, :2] = 7
            spots = [(r, c) for r in range(h) for c in range(2)]
        else:
            grid[:, -2:] = 7
            spots = [(r, c) for r in range(h) for c in range(w - 2, w)]
        for r, c in random.sample(spots, random.randint(1, 4)):
            grid[r, c] = 9

        rects = []
        nrect = random.randint(2, 3)
        for i in range(nrect):
            ok = False
            for _j in range(200):
                rh, rw = random.randint(3, 5), random.randint(4, 6)
                r0 = random.randint(3, h - rh - 4)
                c0 = random.randint(3, w - rw - 4)
                rect = (r0, r0 + rh - 1, c0, c0 + rw - 1)
                if all(rect[1] + 1 < a or a1 + 1 < rect[0] or rect[3] + 1 < b or b1 + 1 < rect[2] for a, a1, b, b1 in rects):
                    if edge in ("left", "right") and all(rect[1] < a or a1 < rect[0] for a, a1, _, _ in rects):
                        ok = True
                    if edge in ("top", "bottom") and all(rect[3] < b or b1 < rect[2] for _, _, b, b1 in rects):
                        ok = True
                    if not rects:
                        ok = True
                    if ok:
                        rects.append(rect)
                        grid[rect[0]:rect[1] + 1, rect[2]:rect[3] + 1] = 7
                        break
            if not ok:
                break
        if len(rects) != nrect:
            continue

        trails = []
        for idx, (r0, r1, c0, c1) in enumerate(rects):
            limit = (c1 - c0 + 1) if edge in ("left", "right") else (r1 - r0 + 1)
            cnt = random.randint(1, min(limit, 5))
            for _k in range(cnt):
                for _j in range(200):
                    if edge in ("left", "right"):
                        r = random.randint(r0, r1)
                        c = random.choice(list(range(2, c0)) + list(range(c1 + 1, w - 2)))
                    else:
                        c = random.randint(c0, c1)
                        r = random.choice(list(range(2, r0)) + list(range(r1 + 1, h - 2)))
                    if grid[r, c] == 0 and all(abs(r - a) + abs(c - b) > 1 for a, b in trails):
                        grid[r, c] = 1
                        trails.append((r, c))
                        break
        if len(trails) >= nrect:
            return grid
    raise RuntimeError("failed to generate input")
