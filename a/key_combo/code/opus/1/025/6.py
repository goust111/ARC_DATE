import random
import numpy as np


def _component(grid, color):
    pts = [tuple(map(int, p)) for p in np.argwhere(grid == color)]
    if not pts:
        return []
    remaining = set(pts)
    comps = []
    while remaining:
        start = remaining.pop()
        stack = [start]
        comp = [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                q = (r + dr, c + dc)
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
                    comp.append(q)
        comps.append(comp)
    return max(comps, key=len)


def _walk_to(path, target):
    r, c = path[-1]
    tr, tc = target
    while (r, c) != (tr, tc):
        if r != tr:
            r += (tr > r) - (tr < r)
        elif c != tc:
            c += (tc > c) - (tc < c)
        path.append((r, c))


def _route(grid, terminal, obstacle, count):
    rs = [p[0] for p in obstacle]
    cs = [p[1] for p in obstacle]
    r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
    top_len = abs(terminal[0] - (r0 - 1))
    bot_len = abs(terminal[0] - (r1 + 1))
    side_r = r0 - 1 if top_len <= bot_len else r1 + 1
    exit_dr = -1 if side_r < r0 else 1
    path = [terminal]
    _walk_to(path, (terminal[0], c0 - 1))
    _walk_to(path, (side_r, c0 - 1))
    _walk_to(path, (side_r, c1 + 1))
    r, c = path[-1]
    while len(path) <= count:
        r += exit_dr
        c += 1
        path.append((r, c))
    valid = []
    h, w = grid.shape
    red = set(obstacle)
    for p in path[1:]:
        if len(valid) == count:
            break
        if 0 <= p[0] < h and 0 <= p[1] < w and p not in red:
            valid.append(p)
    while len(valid) < count:
        r, c = valid[-1] if valid else terminal
        p = (max(0, min(h - 1, r + exit_dr)), max(0, min(w - 1, c + 1)))
        if p not in red:
            valid.append(p)
        else:
            break
    return valid[:count]


def main(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    h, w = grid.shape
    green = np.argwhere(grid == 3)
    if len(green) == 0:
        return out
    row_counts = np.bincount(green[:, 0], minlength=h)
    gr = int(row_counts.argmax())
    gcols = np.where(grid[gr] == 3)[0]
    terminal = (gr, int(gcols.max()))
    blue = _component(grid, 1)
    red = _component(grid, 2)
    if not blue or not red:
        return out
    path = _route(grid, terminal, red, len(blue))
    for r, c in blue:
        out[r, c] = 0
    for r, c in path:
        out[r, c] = 1
    prs, pcs = [p[0] for p in path], [p[1] for p in path]
    br0, br1, bc0, bc1 = min(prs), max(prs), min(pcs), max(pcs)
    width = bc1 - bc0 + 1
    red_cs = [p[1] for p in red]
    mid = (min(red_cs) + max(red_cs)) / 2
    marks = np.argwhere(grid == 4)
    if len(marks):
        mark_c = int(marks[0, 1])
        left = mark_c <= mid
        for r, c in path:
            local_c = c - bc0
            refl_c = width - 1 - local_c
            nr = r
            nc = bc0 - width + refl_c if left else bc1 + 1 + refl_c
            if 0 <= nr < h and 0 <= nc < w and out[nr, nc] == 0:
                out[nr, nc] = 4
        out[tuple(marks[0])] = 4
    return out


def generate_input() -> np.ndarray:
    h = w = random.randint(15, 18)
    g = np.zeros((h, w), dtype=int)
    row = random.randint(5, 9)
    glen = random.randint(3, 5)
    g[row, :glen] = 3
    count = random.randint(5, 7)
    for k in range(count):
        g[row + (k % 2), glen + 1 + k] = 1
    r0 = max(2, row - random.randint(1, 2))
    c0 = min(w - 6, glen + count + random.randint(2, 3))
    shape = [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)]
    if random.random() < 0.5:
        shape += [(0, 2), (2, 2)]
    for dr, dc in shape:
        g[r0 + dr, c0 + dc] = 2
    mark_left = random.random() < 0.5
    mc = c0 if mark_left else c0 + 4
    g[h - 1, max(1, min(w - 2, mc))] = 4
    return g
