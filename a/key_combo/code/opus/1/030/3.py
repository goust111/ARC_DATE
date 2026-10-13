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


def _norm(cells):
    r0, c0, r1, c1 = _bbox(cells)
    arr = np.zeros((r1 - r0, c1 - c0), dtype=int)
    for r, c in cells:
        arr[r - r0, c - c0] = 1
    return arr


def _variants(arr):
    seen = []
    for k in range(4):
        rot = np.rot90(arr, k)
        for x in (rot, np.fliplr(rot)):
            if not any(np.array_equal(x, y) for y in seen):
                seen.append(x.copy())
    return seen


def _holes_for_component(grid, cells):
    r0, c0, r1, c1 = _bbox(cells)
    sub = grid[r0:r1, c0:c1]
    mask = sub == 0
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    q = deque()
    for r in range(h):
        for c in (0, w - 1):
            if mask[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if mask[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    while q:
        x, y = q.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] and not seen[nx, ny]:
                seen[nx, ny] = True
                q.append((nx, ny))
    holes = []
    for comp in _components(mask & ~seen):
        holes.append([(r + r0, c + c0) for r, c in comp])
    return holes


def _hollow_objects(grid):
    objs = []
    for comp in _components((grid != 0) & (grid != 5)):
        vals = {int(grid[r, c]) for r, c in comp}
        if len(vals) != 1:
            continue
        holes = _holes_for_component(grid, comp)
        if holes:
            objs.append({"cells": comp, "bbox": _bbox(comp), "holes": holes, "color": vals.pop()})
    return objs


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, copy=True)
    guide = list(zip(*np.where(g == 5)))
    gr0, gc0, gr1, gc1 = _bbox(guide)
    hollows = _hollow_objects(g)
    targets = [o for o in hollows if gr0 <= o["bbox"][0] and gc0 <= o["bbox"][1] and o["bbox"][2] <= gr1 and o["bbox"][3] <= gc1]
    target = targets[0]
    hole = max(target["holes"], key=len)
    hole_shape = _norm(hole)

    hollow_cells = {p for o in hollows for p in o["cells"]}
    template_comps = []
    for comp in _components((g != 0) & (g != 5)):
        if any(p in hollow_cells for p in comp):
            continue
        vals = {int(g[r, c]) for r, c in comp}
        if len(vals) == 1:
            template_comps.append((vals.pop(), comp))

    fill = None
    for color, comp in template_comps:
        shape = _norm(comp)
        if any(np.array_equal(hole_shape, v) for v in _variants(shape)):
            fill = color
            break
    if fill is None:
        fill = 6

    r0, c0, r1, c1 = target["bbox"]
    out = g[r0:r1, c0:c1].copy()
    out[out == 5] = 0
    for r, c in hole:
        out[r - r0, c - c0] = fill
    return out


HOLES = [
    [(0, 0), (0, 1)],
    [(0, 0), (1, 0), (1, 1)],
    [(0, 1), (1, 0), (1, 1), (1, 2)],
    [(0, 0), (0, 1), (1, 1), (2, 1)],
    [(0, 1), (1, 0), (1, 1), (2, 1), (1, 2)],
]


def _put_hollow(g, top, left, color, hole):
    hr0, hc0, hr1, hc1 = _bbox(hole)
    hh, hw = hr1 - hr0, hc1 - hc0
    block = np.full((hh + 2, hw + 2), color, dtype=int)
    for r, c in hole:
        block[r - hr0 + 1, c - hc0 + 1] = 0
    g[top:top + block.shape[0], left:left + block.shape[1]] = block
    return top, left, top + block.shape[0], left + block.shape[1]


def generate_input() -> np.ndarray:
    g = np.zeros((20, 20), dtype=int)
    colors = [1, 2, 3, 4, 8]
    tmpl_colors = [6, 7, 9, 2, 1]
    n = random.randint(3, 5)
    idxs = random.sample(range(len(HOLES)), n)
    positions = [(1, 1), (1, 8), (1, 14), (8, 1), (8, 8)]
    random.shuffle(positions)
    boxes = []
    for i, idx in enumerate(idxs):
        box = _put_hollow(g, positions[i][0], positions[i][1], colors[i], HOLES[idx])
        boxes.append((box, idx))

    selected = random.randrange(n)
    r0, c0, r1, c1 = boxes[selected][0]
    pad = 1
    for p in [(r0 - pad, c0 - pad), (r0 - pad, c1 + pad - 1), (r1 + pad - 1, c0 - pad), (r1 + pad - 1, c1 + pad - 1)]:
        if 0 <= p[0] < 20 and 0 <= p[1] < 20:
            g[p] = 5

    tr, tc = 14, 12
    for j, idx in enumerate(idxs):
        shape = HOLES[idx]
        color = tmpl_colors[j]
        rr = tr + (j // 2) * 3
        cc = tc + (j % 2) * 4
        for r, c in shape:
            if rr + r < 20 and cc + c < 20:
                g[rr + r, cc + c] = color
    return g
