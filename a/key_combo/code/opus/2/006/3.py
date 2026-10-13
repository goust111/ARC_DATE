import random
from collections import Counter, deque

import numpy as np


def _chambers(grid):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(1, h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == 5:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                y, x = q.popleft()
                cells.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 1 <= ny < h and 0 <= nx < w and not seen[ny, nx] and int(grid[ny, nx]) != 5:
                        seen[ny, nx] = True
                        q.append((ny, nx))
            comps.append(cells)
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    legend = [int(v) for v in grid[0] if int(v) not in (0, 5)]
    pos = {color: i for i, color in enumerate(legend)}
    out = grid.copy()
    for cells in _chambers(grid):
        cell_set = set(cells)
        markers = sorted([p for p in cells if int(grid[p]) not in (0, 5)])
        votes = {}
        for p in cells:
            votes[p] = 0
        for r, c in markers:
            if int(grid[r, c]) not in pos:
                continue
            anchor = pos[int(grid[r, c])]
            start_c = c - anchor
            for i, color in enumerate(legend):
                q = (r, start_c + i)
                if q in cell_set:
                    votes[q] = color
        counts = Counter(v for v in votes.values() if v != 0)
        if not counts:
            continue
        best = min(
            counts,
            key=lambda color: (-counts[color], pos.get(color, len(legend))),
        )
        for p in cells:
            out[p] = best
    out[0] = grid[0]
    return out


def generate_input() -> np.ndarray:
    palette = [1, 2, 3, 4, 7]
    for _ in range(300):
        cols = random.choice([2, 3])
        rows = random.choice([2, 3])
        if rows * cols not in (4, 6):
            continue
        cell_h = random.randint(3, 4)
        cell_w = random.randint(3, 5)
        h = 2 + rows * cell_h + (rows - 1) + 1
        w = 2 + cols * cell_w + (cols - 1) + 1
        if h > 18 or w > 18:
            continue
        grid = np.zeros((h, w), dtype=int)
        legend = random.sample(palette, random.randint(4, 5))
        grid[0, : len(legend)] = legend
        grid[1, :] = 5
        grid[-1, :] = 5
        grid[:, 0] = 5
        grid[:, -1] = 5
        chamber_ranges = []
        r0 = 2
        for rr in range(rows):
            c0 = 1
            for cc in range(cols):
                chamber_ranges.append((range(r0, r0 + cell_h), range(c0, c0 + cell_w)))
                c0 += cell_w
                if cc < cols - 1:
                    grid[:, c0] = 5
                    c0 += 1
            r0 += cell_h
            if rr < rows - 1:
                grid[r0, :] = 5
                r0 += 1
        for rrng, crng in chamber_ranges:
            marker_count = random.choice([1, 2])
            colors = random.sample(legend, marker_count)
            spots = [(r, c) for r in rrng for c in crng]
            random.shuffle(spots)
            for color in colors:
                y, x = spots.pop()
                grid[y, x] = color
        return grid
    return generate_input()
