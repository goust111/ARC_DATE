import random
from collections import deque
import numpy as np


def _grid_slots(grid):
    grey = np.argwhere(grid == 5)
    r0, c0 = grey.min(axis=0)
    r1, c1 = grey.max(axis=0)
    sub = grid[r0:r1 + 1, c0:c1 + 1]
    line_rows = [r0 + i for i in range(sub.shape[0]) if np.all(sub[i, :] == 5)]
    line_cols = [c0 + j for j in range(sub.shape[1]) if np.all(sub[:, j] == 5)]
    slots = []
    for a, b in zip(line_rows, line_rows[1:]):
        if b - a <= 1:
            continue
        for x, y in zip(line_cols, line_cols[1:]):
            if y - x <= 1:
                continue
            slots.append((a + 1, b - 1, x + 1, y - 1))
    return (r0, r1, c0, c1), slots


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


def main(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    (gr0, gr1, gc0, gc1), slots = _grid_slots(grid)
    freq = {}
    for r0, r1, c0, c1 in slots:
        vals = [int(v) for v in grid[r0:r1 + 1, c0:c1 + 1].ravel() if v not in (0, 5)]
        if vals:
            color = max(set(vals), key=vals.count)
            freq[color] = freq.get(color, 0) + 1
    outside = np.ones(grid.shape, dtype=bool)
    outside[gr0:gr1 + 1, gc0:gc1 + 1] = False
    mask = (grid != 0) & (grid != 5) & outside
    moves = []
    for comp in _components(mask):
        color = int(grid[comp[0]])
        rows = [p[0] for p in comp]
        cols = [p[1] for p in comp]
        height = max(rows) - min(rows) + 1
        moves.append((comp, color, min(rows), max(rows), min(cols), max(cols), height))
    for comp, *_ in moves:
        for r, c in comp:
            out[r, c] = 0
    for comp, color, r0, r1, c0, c1, height in moves:
        shift = freq.get(color, 0) * height
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if grid[r, c] == color:
                    out[r - shift, c] = color
    return out


def _draw_lattice(grid, top, left, rows, cols, cell_h, cell_w):
    for rr in range(rows + 1):
        r = top + rr * (cell_h + 1)
        grid[r, left:left + cols * (cell_w + 1) + 1] = 5
    for cc in range(cols + 1):
        c = left + cc * (cell_w + 1)
        grid[top:top + rows * (cell_h + 1) + 1, c] = 5
    return top + rows * (cell_h + 1), left + cols * (cell_w + 1)


def generate_input() -> np.ndarray:
    palette = [1, 2, 3, 4, 6, 7, 8, 9]
    for _ in range(500):
        grid = np.zeros((18, 18), dtype=int)
        rows, cols = random.choice([(3, 3), (4, 3)])
        bottom, right = _draw_lattice(grid, 1, 1, rows, cols, 1, 1)
        active = random.sample(palette, random.randint(3, 4))
        slot_colors = []
        for r in range(rows):
            for c in range(cols):
                if random.random() < 0.65:
                    slot_colors.append(random.choice(active))
                else:
                    slot_colors.append(0)
        for color in active:
            if color not in slot_colors:
                slot_colors[random.randrange(len(slot_colors))] = color
        for idx, color in enumerate(slot_colors):
            if color:
                sr = 2 + (idx // cols) * 2
                sc = 2 + (idx % cols) * 2
                grid[sr, sc] = color
        counts = {c: slot_colors.count(c) for c in active}
        rects = []
        for color in random.sample(active, random.randint(2, min(4, len(active)))):
            max_rh = min(3, 15 // (counts[color] + 1))
            if max_rh < 1:
                continue
            rh = random.randint(1, max_rh)
            rw = random.randint(2, 4)
            shift = counts[color] * rh
            placed = False
            for _try in range(80):
                r0 = random.randint(shift + 1, 16 - rh)
                c0 = random.randint(max(right + 2, 9), 17 - rw)
                cells = {(r, c) for r in range(r0, r0 + rh) for c in range(c0, c0 + rw)}
                new = {(r - shift, c) for r, c in cells}
                if any(grid[r, c] != 0 for r, c in cells | new):
                    continue
                if any(cells & old or new & old for old, _ in rects):
                    continue
                rects.append((cells | new, color))
                for r, c in cells:
                    grid[r, c] = color
                placed = True
                break
            if not placed:
                break
        if len(rects) >= 2:
            return grid
    return generate_input()
