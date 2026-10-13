import random
import numpy as np


def _frame_cells(r0, r1, c0, c1):
    cells = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _toposort(colors, edges):
    indeg = {c: 0 for c in colors}
    adj = {c: [] for c in colors}
    for a, b in edges:
        if a != b and b not in adj[a]:
            adj[a].append(b)
            indeg[b] += 1
    order = []
    ready = sorted([c for c in colors if indeg[c] == 0])
    while ready:
        c = ready.pop(0)
        order.append(c)
        for d in adj[c]:
            indeg[d] -= 1
            if indeg[d] == 0:
                ready.append(d)
                ready.sort()
    return order if len(order) == len(colors) else list(colors)


def main(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    colors = [int(c) for c in np.unique(grid) if c not in (0, 4, 7)]
    frames = {}
    for color in colors:
        pts = np.argwhere(grid == color)
        if len(pts) == 0:
            continue
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        if r1 > r0 and c1 > c0:
            frames[color] = _frame_cells(int(r0), int(r1), int(c0), int(c1))
    colors = list(frames)
    edges = set()
    for i, a in enumerate(colors):
        for b in colors[i + 1:]:
            for r, c in frames[a] & frames[b]:
                seen = int(grid[r, c])
                if seen == a:
                    edges.add((b, a))
                    break
                if seen == b:
                    edges.add((a, b))
                    break
    order = _toposort(colors, edges)
    h, w = grid.shape
    for rank, color in enumerate(order, start=1):
        pts = np.array(list(frames[color]))
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        corners = [(r0, c0), (r0, c1), (r1, c0), (r1, c1)]
        deltas = [(-1, -1), (-1, 1), (1, -1), (1, 1)] if rank % 2 else [(-1, 0), (1, 0), (0, -1), (0, 1)]
        mark = 4 if rank % 2 else 7
        for r, c in corners:
            for dr, dc in deltas:
                rr, cc = int(r) + dr, int(c) + dc
                if 0 <= rr < h and 0 <= cc < w and out[rr, cc] == 0:
                    out[rr, cc] = mark
    return out


def generate_input() -> np.ndarray:
    h = w = random.randint(14, 18)
    grid = np.zeros((h, w), dtype=int)
    colors = random.sample([1, 2, 3, 5, 6, 8, 9], random.randint(4, 6))
    boxes = []
    base_r = random.randint(2, 4)
    base_c = random.randint(2, 4)
    for i, color in enumerate(colors):
        r0 = max(1, base_r + random.randint(-1, 1) + i % 2)
        c0 = max(1, base_c + random.randint(-1, 1) + (i // 2) % 2)
        r1 = min(h - 2, r0 + random.randint(7, 10) - i // 2)
        c1 = min(w - 2, c0 + random.randint(7, 10) - i // 3)
        if r1 - r0 < 4:
            r1 = min(h - 2, r0 + 4)
        if c1 - c0 < 4:
            c1 = min(w - 2, c0 + 4)
        boxes.append((color, r0, r1, c0, c1))
    random.shuffle(boxes)
    for color, r0, r1, c0, c1 in boxes:
        for r, c in _frame_cells(r0, r1, c0, c1):
            grid[r, c] = color
    return grid
