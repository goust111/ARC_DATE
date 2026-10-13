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
    out = np.zeros((r1 - r0, c1 - c0), dtype=int)
    for r, c in cells:
        out[r - r0, c - c0] = 1
    return out


def _variants(arr):
    out = []
    for k in range(4):
        x = np.rot90(arr, k)
        for y in (x, np.fliplr(x)):
            if not any(np.array_equal(y, z) for z in out):
                out.append(y.copy())
    return out


def _find_cross(g):
    rows = [r for r in range(g.shape[0]) if np.all(g[r, :] == 5)]
    cols = [c for c in range(g.shape[1]) if np.all(g[:, c] == 5)]
    return rows[0], cols[0]


def _legend_templates(q):
    orange = list(zip(*np.where(q == 7)))
    r0, c0, r1, c1 = _bbox(orange)
    area = q[r0:r1, c0:c1]
    comps = []
    for comp in _components((area != 0) & (area != 7)):
        vals = {int(area[r, c]) for r, c in comp}
        if len(vals) == 1:
            ar0, ac0, _, _ = _bbox(comp)
            comps.append((r0 + ar0, c0 + ac0, _norm(comp)))
    comps.sort(key=lambda x: (x[0], x[1]))
    return [shape for _, _, shape in comps]


def _quad_shape(q):
    clean = q.copy()
    if np.any(clean == 7):
        orange = list(zip(*np.where(clean == 7)))
        r0, c0, r1, c1 = _bbox(orange)
        clean[r0:r1, c0:c1] = 0
    comps = []
    for comp in _components((clean != 0) & (clean != 5)):
        vals = {int(clean[r, c]) for r, c in comp}
        if len(vals) == 1:
            comps.append(comp)
    return _norm(max(comps, key=len))


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, copy=True)
    sr, sc = _find_cross(g)
    quads = [
        g[:sr, :sc].copy(),
        g[:sr, sc + 1:].copy(),
        g[sr + 1:, :sc].copy(),
        g[sr + 1:, sc + 1:].copy(),
    ]
    templates = _legend_templates(quads[0])

    priorities = []
    for q in quads:
        shp = _quad_shape(q)
        pri = None
        for i, tmpl in enumerate(templates):
            if any(np.array_equal(shp, v) for v in _variants(tmpl)):
                pri = i
                break
        priorities.append(99 if pri is None else pri)

    cleaned = []
    for q in quads:
        x = q.copy()
        if np.any(x == 7):
            orange = list(zip(*np.where(x == 7)))
            r0, c0, r1, c1 = _bbox(orange)
            x[r0:r1, c0:c1] = 0
        cleaned.append(x)

    out = np.zeros_like(cleaned[0])
    order = sorted(range(4), key=lambda i: priorities[i])
    for r in range(out.shape[0]):
        for c in range(out.shape[1]):
            for i in order:
                if cleaned[i][r, c] != 0:
                    out[r, c] = cleaned[i][r, c]
                    break
    return out


SHAPES = [
    [(0, 0), (0, 1), (0, 2)],
    [(0, 0), (1, 0), (1, 1)],
    [(0, 0), (0, 1), (1, 0), (1, 1)],
    [(0, 0), (0, 1), (0, 2), (1, 1)],
]


def _draw_shape(arr, top, left, shape, color):
    for r, c in shape:
        arr[top + r, left + c] = color


def generate_input() -> np.ndarray:
    q = 13
    g = np.zeros((q * 2 + 1, q * 2 + 1), dtype=int)
    g[q, :] = 5
    g[:, q] = 5

    # Orange L-shaped legend frame in the upper-left quadrant.
    g[0, 0:9] = 7
    g[0:9, 0] = 7
    template_order = random.sample(range(4), 4)
    slots = [(1, 1), (1, 5), (5, 1), (5, 5)]
    tmpl_colors = [2, 1, 3, 4]
    for rank, shape_idx in enumerate(template_order):
        _draw_shape(g[:q, :q], slots[rank][0], slots[rank][1], SHAPES[shape_idx], tmpl_colors[rank])

    quad_origins = [(0, 0), (0, q + 1), (q + 1, 0), (q + 1, q + 1)]
    quad_shape_order = random.sample(range(4), 4)
    shape_colors = [2, 1, 3, 4]
    base_conflicts = [(9, 2), (9, 5), (9, 8), (11, 3), (11, 6), (11, 9)]
    conflict_cells = random.sample(base_conflicts, len(base_conflicts))
    for qi, (orow, ocol) in enumerate(quad_origins):
        sub = g[orow:orow + q, ocol:ocol + q]
        for r, c in conflict_cells[:5]:
            if random.random() < 0.7:
                sub[r, c] = shape_colors[qi]
        pos = random.choice([(7, 6), (6, 7), (3, 7), (7, 3)])
        if qi == 0:
            pos = (10, 9)
        _draw_shape(sub, pos[0], pos[1], SHAPES[quad_shape_order[qi]], shape_colors[qi])
    return g
