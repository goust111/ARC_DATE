import random
from collections import deque

import numpy as np


BG = 0
GREY = 5


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
    out = grid.copy()
    color_comps = _components((grid != BG) & (grid != GREY))
    grey_lines = _components(grid == GREY)
    center = np.array(grid.shape) / 2.0
    for line in grey_lines:
        rows = [r for r, _ in line]
        cols = [c for _, c in line]
        horiz = len(set(rows)) == 1
        if horiz:
            ends = [(rows[0], min(cols)), (rows[0], max(cols))]
        else:
            ends = [(min(rows), cols[0]), (max(rows), cols[0])]
        touched = []
        for er, ec in ends:
            for comp in color_comps:
                cells = set(comp)
                if any((er + dr, ec + dc) in cells for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))):
                    touched.append(comp)
                    break
        if len(touched) != 2:
            continue
        d0 = np.linalg.norm(np.mean(np.array(touched[0]), axis=0) - center)
        d1 = np.linalg.norm(np.mean(np.array(touched[1]), axis=0) - center)
        keep, move = (touched[0], touched[1]) if d0 <= d1 else (touched[1], touched[0])
        for r, c in line:
            out[r, c] = BG
        kr = [r for r, _ in keep]
        kc = [c for _, c in keep]
        mr = [r for r, _ in move]
        mc = [c for _, c in move]
        dr = min(kr) - min(mr)
        dc = min(kc) - min(mc)
        for r, c in move:
            out[r, c] = BG
        for r, c in keep:
            out[r, c] = grid[r, c]
        for r, c in move:
            nr, nc = r + dr, c + dc
            if out[nr, nc] == BG:
                out[nr, nc] = grid[r, c]
    return out


def generate_input() -> np.ndarray:
    grid = np.zeros((24, 24), dtype=int)
    pairs = random.randint(2, 3)
    colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], pairs * 2)
    occupied = np.zeros_like(grid, dtype=bool)
    for i in range(pairs):
        h = random.randint(3, 4)
        w = random.randint(3, 4)
        full = [(r, c) for r in range(h) for c in range(w) if random.random() < 0.55]
        if len(full) < 4:
            full = [(r, c) for r in range(h) for c in range(w) if (r + c) % 2 == 0]
        split = random.sample(full, len(full) // 2)
        left = np.zeros((h, w), dtype=int)
        right = np.zeros((h, w), dtype=int)
        ca, cb = colors[2 * i], colors[2 * i + 1]
        for cell in full:
            if cell in split:
                left[cell] = ca
            else:
                right[cell] = cb
        horiz = random.choice([True, False])
        for _ in range(300):
            if horiz:
                r = random.randint(2, 24 - h - 2)
                c1 = random.randint(1, 7)
                c2 = random.randint(c1 + w + 3, 24 - w - 2)
                line_cols = list(range(c1 + w, c2))
                if np.any(occupied[r : r + h, c1 : c1 + w]) or np.any(occupied[r : r + h, c2 : c2 + w]):
                    continue
                grid[r : r + h, c1 : c1 + w] = np.where(left != BG, left, grid[r : r + h, c1 : c1 + w])
                grid[r : r + h, c2 : c2 + w] = np.where(right != BG, right, grid[r : r + h, c2 : c2 + w])
                grid[r + h // 2, c1 + w : c2] = GREY
                occupied[r : r + h, c1 : c2 + w] |= grid[r : r + h, c1 : c2 + w] != BG
                break
            r1 = random.randint(1, 7)
            r2 = random.randint(r1 + h + 3, 24 - h - 2)
            c = random.randint(2, 24 - w - 2)
            if np.any(occupied[r1 : r1 + h, c : c + w]) or np.any(occupied[r2 : r2 + h, c : c + w]):
                continue
            grid[r1 : r1 + h, c : c + w] = np.where(left != BG, left, grid[r1 : r1 + h, c : c + w])
            grid[r2 : r2 + h, c : c + w] = np.where(right != BG, right, grid[r2 : r2 + h, c : c + w])
            grid[r1 + h : r2, c + w // 2] = GREY
            occupied[r1 : r2 + h, c : c + w] |= grid[r1 : r2 + h, c : c + w] != BG
            break
    return grid
