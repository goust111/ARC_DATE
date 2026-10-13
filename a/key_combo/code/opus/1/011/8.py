import random
from collections import deque

import numpy as np


def _block(grid, br, bc, b):
    r, c = br * (b + 1), bc * (b + 1)
    return grid[r:r + b, c:c + b], r, c


def _is_node(block):
    return np.count_nonzero(block == 3) >= 8


def _has_door(grid, br, bc, nbr, nbc, b):
    if br == nbr:
        col = min(bc, nbc) * (b + 1) + b
        r0 = br * (b + 1)
        return np.any(grid[r0:r0 + b, col] == 4)
    row = min(br, nbr) * (b + 1) + b
    c0 = bc * (b + 1)
    return np.any(grid[row, c0:c0 + b] == 4)


def _shortest(nodes, edges, a, z):
    q = deque([[a]])
    seen = {a}
    while q:
        path = q.popleft()
        if path[-1] == z:
            return path
        for nb in sorted(edges[path[-1]]):
            if nb not in seen:
                seen.add(nb)
                q.append(path + [nb])
    return [a]


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, copy=True)
    out = grid.copy()
    b = (grid.shape[0] - 2) // 3
    nodes = []
    terminals = []
    states = {}
    sprite = None
    for br in range(3):
        for bc in range(3):
            block, r0, c0 = _block(grid, br, bc, b)
            if np.any(block == 1) and sprite is None:
                sprite = [(r, c) for r in range(b) for c in range(b) if block[r, c] == 1]
            states[(br, bc)] = "sprite" if np.any(block == 1) else "blank"
            if _is_node(block):
                nodes.append((br, bc))
                out[r0 + b // 2, c0 + b // 2] = 4
            if block[0, 0] == 7 and block[0, -1] == 7 and block[-1, 0] == 7 and block[-1, -1] == 7:
                terminals.append((br, bc))
    edges = {n: [] for n in nodes}
    for br, bc in nodes:
        for dr, dc in ((1, 0), (0, 1), (-1, 0), (0, -1)):
            nb = (br + dr, bc + dc)
            if nb in edges and _has_door(grid, br, bc, *nb, b):
                edges[(br, bc)].append(nb)
    if len(terminals) == 2:
        path = _shortest(nodes, edges, terminals[0], terminals[1])
    else:
        path = []
    if sprite is None:
        sprite = [(1, 1), (1, 2), (2, 2), (3, 2)]
    for br, bc in path:
        block, r0, c0 = _block(out, br, bc, b)
        overlay = [(r, c, int(block[r, c])) for r in range(b) for c in range(b) if block[r, c] in (3, 4, 7)]
        if states[(br, bc)] == "sprite":
            block[:, :] = 2
        else:
            block[:, :] = 0
            for r, c in sprite:
                if 0 <= r < b and 0 <= c < b:
                    block[r, c] = 1
        for r, c, val in overlay:
            block[r, c] = val
        block[b // 2, b // 2] = 4
    return out


def _draw_node(block):
    for p in [(0, 2), (0, 3), (4, 1), (4, 2), (2, 0), (3, 0), (1, 4), (2, 4)]:
        block[p] = 3


def generate_input() -> np.ndarray:
    b = 5
    grid = np.zeros((3 * b + 2, 3 * b + 2), dtype=int)
    sprite = [(1, 1), (1, 2), (2, 2), (3, 2)]
    nodes = [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2)]
    terminals = [(0, 0), (2, 2)]
    for br in range(3):
        for bc in range(3):
            block, _, _ = _block(grid, br, bc, b)
            if random.random() < 0.5:
                for r, c in sprite:
                    block[r, c] = 1
            else:
                block[:, :] = 2
            if (br, bc) in nodes:
                _draw_node(block)
            if (br, bc) in terminals:
                block[0, 0] = block[0, -1] = block[-1, 0] = block[-1, -1] = 7
    doors = [((0, 0), (0, 1)), ((0, 1), (1, 1)), ((1, 1), (1, 2)), ((1, 2), (2, 2))]
    if random.random() < 0.5:
        doors.append(((1, 1), (2, 1)))
    for (a, bpos) in doors:
        br, bc = a
        nr, nc = bpos
        if br == nr:
            col = min(bc, nc) * 6 + 5
            row = br * 6 + 2
            grid[row, col] = 4
        else:
            row = min(br, nr) * 6 + 5
            col = bc * 6 + 2
            grid[row, col] = 4
    return grid
