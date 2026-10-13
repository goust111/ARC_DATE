import random
from collections import deque

import numpy as np


BG = 0


def _components(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp = []
            while q:
                x, y = q.popleft()
                comp.append((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] and not seen[nx, ny]:
                        seen[nx, ny] = True
                        q.append((nx, ny))
            out.append(comp)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, dtype=int)
    strips = []
    for comp in _components(grid != BG):
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        crop = grid[r0 : r1 + 1, c0 : c1 + 1]
        seq = []
        if crop.shape[1] > crop.shape[0]:
            for c in range(crop.shape[1]):
                seq.extend(int(v) for v in crop[:, c] if v != BG)
        else:
            for r in range(crop.shape[0]):
                seq.extend(int(v) for v in crop[r] if v != BG)
        strips.append((len(comp), seq))
    strips.sort(key=lambda x: x[0])
    width = max(len(seq) for _, seq in strips)
    out = np.zeros((len(strips), width), dtype=int)
    for r, (_, seq) in enumerate(strips):
        out[r, : len(seq)] = seq
    return out


def generate_input() -> np.ndarray:
    grid = np.zeros((24, 24), dtype=int)
    occupied = np.zeros_like(grid, dtype=bool)
    obj_count = random.randint(3, 5)
    for _ in range(obj_count):
        horiz = random.choice([True, False])
        h = random.randint(2, 4) if horiz else random.randint(4, 6)
        w = random.randint(4, 6) if horiz else random.randint(2, 4)
        arr = np.zeros((h, w), dtype=int)
        cells = {(0, 0)}
        target = random.randint(max(h, w), h * w - 1)
        while len(cells) < target:
            r, c = random.choice(list(cells))
            nr, nc = r + random.choice([-1, 0, 1]), c + random.choice([-1, 0, 1])
            if 0 <= nr < h and 0 <= nc < w and abs(nr - r) + abs(nc - c) == 1:
                cells.add((nr, nc))
        palette = random.sample([1, 2, 3, 4, 5, 6, 7, 8, 9], random.randint(2, 4))
        for r, c in cells:
            arr[r, c] = random.choice(palette)
        for _ in range(300):
            r = random.randint(1, 23 - h)
            c = random.randint(1, 23 - w)
            rr0, rr1 = max(0, r - 1), min(24, r + h + 1)
            cc0, cc1 = max(0, c - 1), min(24, c + w + 1)
            if np.any(occupied[rr0:rr1, cc0:cc1]):
                continue
            grid[r : r + h, c : c + w] = np.where(arr != BG, arr, grid[r : r + h, c : c + w])
            occupied[rr0:rr1, cc0:cc1] = True
            break
    return grid
