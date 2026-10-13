import random
from collections import deque

import numpy as np


DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _most_common(values):
    colors, counts = np.unique(values, return_counts=True)
    return colors[int(np.argmax(counts))]


def _component_neighbors(cell, cells):
    r, c = cell
    return [(r + dr, c + dc) for dr, dc in DIRS if (r + dr, c + dc) in cells]


def _ordered_path(cells):
    cells = set(cells)
    degrees = {cell: len(_component_neighbors(cell, cells)) for cell in cells}
    endpoints = [cell for cell, degree in degrees.items() if degree == 1]
    if len(endpoints) != 2:
        return None

    cur = min(endpoints)
    prev = None
    order = []
    while True:
        order.append(cur)
        next_cells = [nb for nb in _component_neighbors(cur, cells) if nb != prev]
        if not next_cells:
            break
        prev, cur = cur, next_cells[0]

    return order if len(order) == len(cells) else None


def main(grid: np.ndarray) -> np.ndarray:
    """Color each directed path segment by the most recent gate pixel.

    Input structure:
    The grid background is the most common color. Every non-background
    four-connected component is a one-pixel-wide non-branching path drawn in a
    shared base foreground color, with some path pixels replaced by colored
    gate pixels. Each valid path has exactly two endpoints.

    Rule:
    For each path, choose the endpoint with the smaller row, breaking ties by
    smaller column, as the start. Walk the path one pixel at a time from that
    start. Pixels before the first gate remain in the base path color. When a
    non-base gate pixel is reached, that pixel and every following path pixel
    are recolored to the gate color until another gate changes the active
    color. Background pixels are unchanged.

    Key insight:
    A gate is both a path pixel and a directional state change, so each segment
    is colored by the nearest earlier gate along the endpoint-to-endpoint path,
    not by the nearest gate in space.
    """
    arr = np.asarray(grid)
    out = arr.copy()
    if arr.ndim != 2 or arr.size == 0:
        return out

    bg = _most_common(arr)
    path_mask = arr != bg
    if not np.any(path_mask):
        return out

    path_color = _most_common(arr[path_mask])
    seen = np.zeros(arr.shape, dtype=bool)
    rows, cols = arr.shape

    for sr, sc in zip(*np.nonzero(path_mask)):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue

        queue = deque([(sr, sc)])
        seen[sr, sc] = True
        comp = []
        while queue:
            r, c = queue.popleft()
            comp.append((r, c))
            for dr, dc in DIRS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and path_mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    queue.append((nr, nc))

        order = _ordered_path(comp)
        if order is None:
            continue

        active = None
        for r, c in order:
            color = arr[r, c]
            if color != path_color:
                active = color
            out[r, c] = active if active is not None else path_color

    return out


def _has_turn(path):
    for a, b, c in zip(path, path[1:], path[2:]):
        if (b[0] - a[0], b[1] - a[1]) != (c[0] - b[0], c[1] - b[1]):
            return True
    return False


def _cell_ok(cell, rows, cols, blocked, used, last):
    r, c = cell
    if not (1 <= r < rows - 1 and 1 <= c < cols - 1):
        return False
    if blocked[r, c] or cell in used:
        return False
    for dr, dc in DIRS:
        nb = (r + dr, c + dc)
        if nb in used and nb != last:
            return False
    return True


def _make_walk(rows, cols, blocked, min_len, max_len):
    for _ in range(300):
        start = (random.randrange(1, rows - 1), random.randrange(1, cols - 1))
        if blocked[start]:
            continue

        target = random.randint(min_len, max_len)
        path = [start]
        used = {start}
        last_dir = None

        while len(path) < target:
            r, c = path[-1]
            candidates = []
            for direction in DIRS:
                dr, dc = direction
                nb = (r + dr, c + dc)
                if _cell_ok(nb, rows, cols, blocked, used, path[-1]):
                    candidates.append((direction, nb))
            if not candidates:
                break

            straight = [item for item in candidates if item[0] == last_dir]
            if straight and random.random() < 0.58:
                direction, nb = straight[0]
            else:
                direction, nb = random.choice(candidates)
            path.append(nb)
            used.add(nb)
            last_dir = direction

        if len(path) >= min_len and (_has_turn(path) or random.random() < 0.25):
            return path
    return None


def _block_path(blocked, path):
    rows, cols = blocked.shape
    for r, c in path:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    blocked[nr, nc] = True


def _choose_gate_indices(length, count):
    if count <= 0:
        return []

    candidates = list(range(length))
    random.shuffle(candidates)
    chosen = []
    for idx in candidates:
        if all(abs(idx - other) > 1 for other in chosen):
            chosen.append(idx)
            if len(chosen) == count:
                break
    return sorted(chosen)


def generate_input() -> np.ndarray:
    colors = list(range(10))
    for _ in range(1000):
        rows = random.randint(12, 30)
        cols = random.randint(12, 30)
        bg = random.choice(colors)
        base_choices = [c for c in colors if c != bg]
        base = random.choice(base_choices)
        gate_colors = [c for c in colors if c not in (bg, base)]

        grid = np.full((rows, cols), bg, dtype=int)
        blocked = np.zeros((rows, cols), dtype=bool)
        paths = []
        desired = random.randint(1, min(4, max(1, (rows * cols) // 85)))

        for _ in range(desired * 3):
            if len(paths) >= desired:
                break
            max_len = min(22, max(7, (rows * cols) // 18))
            min_len = min(7, max_len)
            path = _make_walk(rows, cols, blocked, min_len, max_len)
            if path is None:
                continue
            paths.append(path)
            _block_path(blocked, path)

        if not paths:
            continue

        for path in paths:
            for r, c in path:
                grid[r, c] = base

        gate_total = 0
        gate_records = []
        for path in paths:
            order = _ordered_path(path)
            if order is None:
                continue
            max_gates = min(4, max(1, len(order) // 4))
            possible = list(range(max_gates + 1))
            count = random.choice(possible)
            indices = _choose_gate_indices(len(order), count)
            for idx in indices:
                gate_records.append((order[idx], random.choice(gate_colors)))
            gate_total += len(indices)

        if gate_total == 0:
            path = random.choice(paths)
            order = _ordered_path(path)
            if order is None:
                continue
            idx = random.randrange(len(order))
            gate_records.append((order[idx], random.choice(gate_colors)))
            gate_total = 1

        used_gate_colors = []
        for cell, color in gate_records:
            if used_gate_colors and random.random() < 0.28:
                color = random.choice(used_gate_colors)
            used_gate_colors.append(color)
            grid[cell] = color

        non_bg = int(np.count_nonzero(grid != bg))
        base_count = int(np.count_nonzero(grid == base))
        if base_count > non_bg - base_count:
            return grid

    return generate_input()
