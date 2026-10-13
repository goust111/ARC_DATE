import random
from collections import deque
import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _green_components(g):
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) != 3:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                x, y = q.popleft()
                cells.append((x, y))
                for dx, dy in DIR4:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and not seen[nx, ny] and int(g[nx, ny]) == 3:
                        seen[nx, ny] = True
                        q.append((nx, ny))
            comps.append(cells)
    return comps


def _bend_count(cells):
    s = set(cells)
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    count = 0
    for r in range(min(rs), max(rs)):
        for c in range(min(cs), max(cs)):
            if sum((r + dr, c + dc) in s for dr in (0, 1) for dc in (0, 1)) == 3:
                count += 1
    return count


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, copy=True)
    h, w = g.shape
    pal_cols = [c for c in range(w) if int(g[h - 1, c]) != 0]
    palette = [int(g[h - 1, c]) for c in pal_cols[:3]] or [1, 2, 4]
    objects = []
    original_green = set()
    for cells in _green_components(g):
        original_green.update(cells)
        top = min(r for r, _ in cells)
        left = min(c for _, c in cells)
        objects.append({"cells": set(cells), "bend": _bend_count(cells), "top": top, "left": left})
    claims = {}
    for idx, obj in enumerate(objects):
        current = set(obj["cells"])
        all_seen = set(current)
        for k in range(obj["bend"] + 1):
            for p in current:
                key = (obj["bend"], -obj["top"], -obj["left"], -k, idx)
                if p not in claims or key > claims[p][0]:
                    claims[p] = (key, palette[k % len(palette)])
            nxt = set()
            if k < obj["bend"]:
                for r, c in current:
                    for dr, dc in DIR4:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h - 1 and 0 <= nc < w and (nr, nc) not in all_seen:
                            if int(g[nr, nc]) == 0 and (nr, nc) not in original_green:
                                nxt.add((nr, nc))
                all_seen |= nxt
                current = nxt
    out = np.zeros_like(g)
    for (r, c), (_, color) in claims.items():
        out[r, c] = color
    return out


def _place(g, shape):
    h, w = g.shape
    for _ in range(200):
        r0 = random.randint(0, h - 5)
        c0 = random.randint(0, w - 4)
        cells = [(r0 + r, c0 + c) for r, c in shape]
        bad = False
        for r, c in cells:
            if g[r, c] != 0:
                bad = True
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and g[nr, nc] == 3:
                        bad = True
        if not bad:
            for r, c in cells:
                g[r, c] = 3
            return True
    return False


def generate_input() -> np.ndarray:
    h = random.randint(15, 18)
    w = random.randint(15, 18)
    g = np.zeros((h, w), dtype=int)
    start = random.randint(1, w - 4)
    g[h - 1, start:start + 3] = [1, 2, 4]
    shapes = [
        [(0, 0), (1, 0), (2, 0), (2, 1)],
        [(0, 0), (1, 0), (1, 1), (2, 1), (2, 2)],
        [(0, 0), (1, 0), (1, 1), (2, 1), (2, 2), (3, 2)],
    ]
    random.shuffle(shapes)
    for shape in shapes:
        _place(g, shape)
    for _ in range(random.randint(1, 3)):
        _place(g, random.choice(shapes))
    return g
