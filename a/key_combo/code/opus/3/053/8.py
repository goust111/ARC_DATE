import random
from collections import deque

import numpy as np


def _components(mask):
    seen = np.zeros(mask.shape, dtype=bool)
    comps = []
    h, w = mask.shape
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        q = deque([(r, c)])
        seen[r, c] = True
        comp = set()
        while q:
            rr, cc = q.popleft()
            comp.add((rr, cc))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(comp)
    return comps


def _centroid(cells):
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return (sum(rows) / len(rows), sum(cols) / len(cols))


def _direction(cells, center):
    rr, cc = _centroid(cells)
    dr, dc = rr - center[0], cc - center[1]
    if abs(dr) >= abs(dc):
        return (1, 0) if dr > 0 else (-1, 0)
    return (0, 1) if dc > 0 else (0, -1)


def _add_cap(cells, direction, shape):
    h, w = shape
    out = set(cells)
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    dr, dc = direction
    if dr:
        edge = min(rows) if dr < 0 else max(rows)
        new_r = edge + dr
        edge_cols = [c for r, c in cells if r == edge]
        mid = int(round(sum(edge_cols) / len(edge_cols)))
        for c in (mid - 1, mid, mid + 1):
            if 0 <= new_r < h and 0 <= c < w:
                out.add((new_r, c))
    else:
        edge = min(cols) if dc < 0 else max(cols)
        new_c = edge + dc
        edge_rows = [r for r, c in cells if c == edge]
        mid = int(round(sum(edge_rows) / len(edge_rows)))
        for r in (mid - 1, mid, mid + 1):
            if 0 <= r < h and 0 <= new_c < w:
                out.add((r, new_c))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Grow the largest gray object, shrink the others, then recolor by nearest row-key controls."""
    objects = _components(grid == 5)
    controls = [(int(r), int(grid[r, 0])) for r in np.where((grid[:, 0] != 0) & (grid[:, 0] != 5))[0]]
    all_rows = [r for comp in objects for r, _ in comp]
    all_cols = [c for comp in objects for _, c in comp]
    center = (sum(all_rows) / len(all_rows), sum(all_cols) / len(all_cols))
    winner_idx = max(range(len(objects)), key=lambda i: len(objects[i]))
    transformed = []
    for i, cells in enumerate(objects):
        cells = set(cells)
        if i == winner_idx:
            cells = _add_cap(cells, _direction(cells, center), grid.shape)
        else:
            tip = min(cells, key=lambda p: (abs(p[0] - center[0]) + abs(p[1] - center[1]), p[0], p[1]))
            cells.remove(tip)
        transformed.append(cells)

    out = grid.copy()
    out[grid == 5] = 0
    for r, _ in controls:
        out[r, 0] = 0
    used = set()
    for cells in transformed:
        center_row = int(sum(r for r, _ in cells) // len(cells))
        choices = sorted((abs(r - center_row), r, color) for r, color in controls if r not in used)
        if not choices:
            choices = sorted((abs(r - center_row), r, color) for r, color in controls)
        _, row, color = choices[0]
        used.add(row)
        for r, c in cells:
            out[r, c] = color
    return out


def _stamp(grid, cells):
    for r, c in cells:
        grid[r, c] = 5


def generate_input() -> np.ndarray:
    grid = np.zeros((13, 13), dtype=int)
    shapes = [
        {(1, 6), (2, 5), (2, 6), (2, 7), (3, 6), (4, 6)},
        {(11, 6), (10, 5), (10, 6), (10, 7), (9, 6), (8, 6), (9, 5)},
        {(6, 1), (5, 2), (6, 2), (7, 2), (6, 3), (6, 4)},
        {(6, 11), (5, 10), (6, 10), (7, 10), (6, 9), (6, 8)},
    ]
    big = random.randrange(4)
    if big == 0:
        shapes[0] |= {(1, 5), (1, 7), (0, 6)}
    elif big == 1:
        shapes[1] |= {(11, 5), (11, 7), (12, 6)}
    elif big == 2:
        shapes[2] |= {(5, 1), (7, 1), (4, 2)}
    else:
        shapes[3] |= {(5, 11), (7, 11), (6, 12)}
    for cells in shapes:
        _stamp(grid, cells)
    control_rows = sorted(random.sample([1, 3, 5, 7, 9, 11], 4))
    for r, color in zip(control_rows, random.sample([1, 2, 3, 4, 8], 4)):
        grid[r, 0] = color
    return grid


if __name__ == "__main__":
    for _ in range(100):
        result = main(generate_input())
        assert isinstance(result, np.ndarray)
