import random
from collections import deque

import numpy as np


def _components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                x, y = q.popleft()
                cells.append((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] and not seen[nx, ny]:
                        seen[nx, ny] = True
                        q.append((nx, ny))
            comps.append(cells)
    return comps


def _bbox(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), min(cs), max(rs) + 1, max(cs) + 1


def _center(cells):
    return sum(r for r, _ in cells) / len(cells), sum(c for _, c in cells) / len(cells)


def _anchor_bbox(g):
    return _bbox(list(zip(*np.where(g == 4))))


def _direction(cells, anchor):
    ar0, ac0, ar1, ac1 = anchor
    acen = ((ar0 + ar1 - 1) / 2, (ac0 + ac1 - 1) / 2)
    r, c = _center(cells)
    dr = acen[0] - r
    dc = acen[1] - c
    if abs(dr) > abs(dc):
        return (1, 0) if dr > 0 else (-1, 0)
    return (0, 1) if dc > 0 else (0, -1)


def _interval_overlap(a0, a1, b0, b1):
    return max(a0, b0) < min(a1, b1)


def _shift_to_contact(cells, direction, obstacles, shape):
    r0, c0, r1, c1 = _bbox(cells)
    dr, dc = direction
    best = None
    for obj in obstacles:
        or0, oc0, or1, oc1 = _bbox(obj)
        if dr == 1 and or0 >= r1 and _interval_overlap(c0, c1, oc0, oc1):
            gap = or0 - r1
        elif dr == -1 and or1 <= r0 and _interval_overlap(c0, c1, oc0, oc1):
            gap = r0 - or1
        elif dc == 1 and oc0 >= c1 and _interval_overlap(r0, r1, or0, or1):
            gap = oc0 - c1
        elif dc == -1 and oc1 <= c0 and _interval_overlap(r0, r1, or0, or1):
            gap = c0 - oc1
        else:
            continue
        if best is None or gap < best:
            best = gap
    if best is None:
        h, w = shape
        if dr == 1:
            best = h - r1
        elif dr == -1:
            best = r0
        elif dc == 1:
            best = w - c1
        else:
            best = c0
    return [(r + dr * best, c + dc * best) for r, c in cells]


def _mirror(cells, direction):
    r0, c0, r1, c1 = _bbox(cells)
    dr, dc = direction
    out = []
    for r, c in cells:
        if dr == 1:
            out.append((2 * (r1 - 1) + 1 - r, c))
        elif dr == -1:
            out.append((2 * r0 - 1 - r, c))
        elif dc == 1:
            out.append((r, 2 * (c1 - 1) + 1 - c))
        else:
            out.append((r, 2 * c0 - 1 - c))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, copy=True)
    h, w = g.shape
    anchor = _anchor_bbox(g)
    sprites = _components(g == 3)
    ar0, ac0, ar1, ac1 = anchor
    acen = ((ar0 + ar1 - 1) / 2, (ac0 + ac1 - 1) / 2)
    sprites.sort(key=lambda cells: (abs(_center(cells)[0] - acen[0]) + abs(_center(cells)[1] - acen[1]), _bbox(cells)[0], _bbox(cells)[1]))

    out = np.zeros_like(g)
    out[g == 4] = 4
    obstacles = [list(zip(*np.where(out == 4)))]
    cycle = [2, 8, 5]
    for i, cells in enumerate(sprites):
        direction = _direction(cells, anchor)
        moved = _shift_to_contact(cells, direction, obstacles, g.shape)
        mirrored = _mirror(moved, direction)
        for r, c in moved:
            if 0 <= r < h and 0 <= c < w:
                out[r, c] = 3
        color = cycle[i % len(cycle)]
        drawn_mirror = []
        for r, c in mirrored:
            if 0 <= r < h and 0 <= c < w and out[r, c] == 0:
                out[r, c] = color
                drawn_mirror.append((r, c))
        obstacles.append([p for p in moved if 0 <= p[0] < h and 0 <= p[1] < w])
        if drawn_mirror:
            obstacles.append(drawn_mirror)
    out[g == 4] = 4
    return out


SHAPES = [
    [(0, 1), (1, 0), (1, 1), (2, 1)],
    [(0, 0), (1, 0), (1, 1), (2, 1)],
    [(0, 1), (1, 1), (1, 0), (2, 0), (2, 1)],
    [(0, 0), (0, 1), (1, 1), (2, 1)],
    [(0, 1), (1, 0), (1, 1), (1, 2)],
]


def _place(g, top, left, shape):
    for r, c in shape:
        g[top + r, left + c] = 3


def generate_input() -> np.ndarray:
    g = np.zeros((18, 18), dtype=int)
    ar, ac = random.choice([(8, 8), (7, 9), (9, 7)])
    g[ar:ar + 2, ac:ac + 2] = 4
    candidates = [
        (ar - random.randint(5, 7), ac, random.choice(SHAPES)),
        (ar + random.randint(4, 6), ac, random.choice(SHAPES)),
        (ar, ac - random.randint(5, 7), random.choice(SHAPES)),
        (ar, ac + random.randint(4, 6), random.choice(SHAPES)),
    ]
    random.shuffle(candidates)
    n = random.randint(4, 5)
    used = []
    for top, left, shape in candidates[:n]:
        top = max(0, min(15, top))
        left = max(0, min(15, left))
        cells = [(top + r, left + c) for r, c in shape]
        if all(g[r, c] == 0 for r, c in cells):
            _place(g, top, left, shape)
            used.append(cells)
    if len(used) < 4:
        return generate_input()
    return g
