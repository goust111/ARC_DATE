import random
from collections import defaultdict, deque
import numpy as np


def _boundary(box):
    r1, c1, r2, c2 = box
    return {(r, c) for r in range(r1, r2 + 1) for c in range(c1, c2 + 1)
            if r in (r1, r2) or c in (c1, c2)}


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, copy=True)
    colors = [int(c) for c in np.unique(grid) if c not in (0, 5)]
    boxes = {}
    for color in colors:
        coords = np.argwhere(grid == color)
        if coords.size:
            r1, c1 = coords.min(axis=0)
            r2, c2 = coords.max(axis=0)
            boxes[color] = (int(r1), int(c1), int(r2), int(c2))
    bounds = {c: _boundary(b) for c, b in boxes.items()}
    above = defaultdict(set)
    below = defaultdict(set)
    for i, a in enumerate(colors):
        for b in colors[i + 1:]:
            for p in bounds[a] & bounds[b]:
                shown = int(grid[p])
                if shown == a:
                    above[b].add(a)
                    below[a].add(b)
                elif shown == b:
                    above[a].add(b)
                    below[b].add(a)
                elif shown in boxes:
                    above[a].add(shown)
                    above[b].add(shown)
                    below[shown].update([a, b])
    indeg = {c: len(below[c]) for c in colors}
    q = deque(sorted([c for c in colors if indeg[c] == 0]))
    order = []
    while q:
        c = q.popleft()
        order.append(c)
        for top in sorted(above[c]):
            indeg[top] -= 1
            if indeg[top] == 0:
                q.append(top)
    for c in colors:
        if c not in order:
            order.append(c)
    return np.array([order], dtype=int)


def _draw_frame(canvas, box, color):
    for p in _boundary(box):
        canvas[p] = color


def _intersections(a, b):
    return _boundary(a) & _boundary(b)


def generate_input() -> np.ndarray:
    h = w = 14
    colors = random.sample([1, 2, 3, 4, 7], random.randint(3, 5))
    order = colors[:]
    random.shuffle(order)
    templates = [
        (1, 1, 7, 8),
        (3, 0, 10, 6),
        (0, 4, 6, 12),
        (5, 5, 12, 13),
        (2, 6, 11, 10),
    ]
    random.shuffle(templates)
    boxes = dict(zip(order, templates[:len(order)]))
    for _ in range(200):
        if all(_intersections(boxes[a], boxes[b]) for i, a in enumerate(order) for b in order[i + 1:]):
            break
        random.shuffle(templates)
        boxes = dict(zip(order, templates[:len(order)]))
    g = np.zeros((h, w), dtype=int)
    for color in order:
        _draw_frame(g, boxes[color], color)
    for _ in range(random.randint(2, 5)):
        r, c = random.randrange(h), random.randrange(w)
        if g[r, c] == 0:
            g[r, c] = 5
    return g
