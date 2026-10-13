import numpy as np
from collections import deque


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR_NAMES = ("up", "down", "left", "right")
DIR_VECS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _normalize(cells):
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def _rotate_right_shape(cells, direction: str):
    if direction == "right":
        rot = [(r, c) for r, c in cells]
    elif direction == "down":
        rot = [(c, -r) for r, c in cells]
    elif direction == "left":
        rot = [(-r, -c) for r, c in cells]
    elif direction == "up":
        rot = [(-c, r) for r, c in cells]
    else:
        raise ValueError(direction)
    return _normalize(rot)


def _shift_cells(cells, dr: int, dc: int):
    return {(r + dr, c + dc) for r, c in cells}


def _expanded(cells, h: int, w: int):
    expanded = set()
    for r, c in cells:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if 0 <= rr < h and 0 <= cc < w:
                    expanded.add((rr, cc))
    return expanded


def _is_connected(cells):
    if not cells:
        return False
    start = next(iter(cells))
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in cells and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == len(cells)


def _find_objects(grid: np.ndarray):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    objects = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == 0 or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            objects.append((color, tuple(coords)))
    return objects


def _tail_info(coords):
    cells = set(coords)
    neigh = {
        cell: [nbr for nbr in _neighbors4(*cell) if nbr in cells]
        for cell in cells
    }
    endpoints = [cell for cell, ns in neigh.items() if len(ns) == 1]
    candidates = []

    for tip in endpoints:
        step_to_body = (neigh[tip][0][0] - tip[0], neigh[tip][0][1] - tip[1])
        prev = None
        curr = tip
        segment = [tip]

        while True:
            nexts = [p for p in neigh[curr] if p != prev]
            if len(nexts) != 1:
                body = nexts[0] if len(nexts) == 1 else None
                break

            nxt = nexts[0]
            nxt_neighbors = neigh[nxt]
            if len(nxt_neighbors) == 2:
                others = [p for p in nxt_neighbors if p != curr]
                if others and (others[0][0] - nxt[0], others[0][1] - nxt[1]) == step_to_body:
                    segment.append(nxt)
                    prev, curr = curr, nxt
                    continue

            body = nxt
            break

        seg_set = set(segment)
        remainder = cells - seg_set
        if not remainder or not _is_connected(remainder):
            continue

        body_degree = 0
        if body in cells:
            body_degree = sum(1 for p in neigh[body] if p not in seg_set)

        candidates.append((len(segment), body_degree, tip, step_to_body))

    if not candidates:
        raise AssertionError("object has no valid tail candidate")

    length, _body_degree, _tip, step_to_body = max(candidates, key=lambda item: (item[0], item[1]))
    return step_to_body, length


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each monochrome object away from its single-cell-wide tail.

    The tail direction is read from the object's asymmetric endpoint, and the
    translation length equals the tail length.
    """
    g = np.asarray(grid, dtype=int)
    out = np.zeros_like(g)

    for color, coords in _find_objects(g):
        step_to_body, tail_length = _tail_info(coords)
        dr = step_to_body[0] * tail_length
        dc = step_to_body[1] * tail_length
        for r, c in coords:
            out[r + dr, c + dc] = color

    return out


RIGHT_TEMPLATES = {
    1: [
        ((0, 0), (0, 1), (1, 0), (1, 1), (0, 2)),
        ((0, 0), (0, 1), (1, 0), (1, 1), (1, 2)),
        ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (1, 2)),
        ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (0, 2)),
    ],
    2: [
        ((0, 0), (0, 1), (1, 0), (1, 1), (0, 2), (0, 3)),
        ((0, 0), (0, 1), (1, 0), (1, 1), (1, 2), (1, 3)),
        ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (1, 2), (1, 3)),
        ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (0, 2), (0, 3)),
    ],
    3: [
        ((0, 0), (1, 0), (2, 0), (1, 1), (1, 2), (1, 3)),
        ((0, 0), (1, 0), (0, 1), (0, 2), (0, 3)),
        ((0, 0), (1, 0), (1, 1), (1, 2), (1, 3)),
        ((0, 0), (1, 0), (2, 0), (3, 0), (1, 1), (1, 2), (1, 3)),
        ((0, 0), (1, 0), (2, 0), (3, 0), (2, 1), (2, 2), (2, 3)),
    ],
}


def _make_shape(rng: np.random.Generator):
    tail_length = int(rng.integers(1, 4))
    direction = DIR_NAMES[int(rng.integers(0, 4))]
    template = RIGHT_TEMPLATES[tail_length][int(rng.integers(0, len(RIGHT_TEMPLATES[tail_length])))]
    cells = _rotate_right_shape(template, direction)
    tail_vec = DIR_VECS[direction]
    move = (-tail_vec[0] * tail_length, -tail_vec[1] * tail_length)

    # Reject a template if the structural detector would read a different tail.
    detected_step, detected_len = _tail_info(cells)
    if detected_len != tail_length or detected_step != move:
        return None

    return {
        "cells": set(cells),
        "tail_length": tail_length,
        "direction": direction,
        "move": move,
    }


def _combined_span(cells, move):
    dr, dc = move
    input_cells = {(r - dr, c - dc) for r, c in cells}
    all_cells = set(cells) | input_cells
    min_r = min(r for r, _ in all_cells)
    max_r = max(r for r, _ in all_cells)
    min_c = min(c for _, c in all_cells)
    max_c = max(c for _, c in all_cells)
    return (max_r - min_r + 1) * (max_c - min_c + 1)


def _gen_attempt(rng: np.random.Generator):
    h = int(rng.integers(12, 19))
    w = int(rng.integers(12, 19))
    max_objects = 5 if h * w >= 250 else 4
    num_objects = int(rng.integers(2, max_objects + 1))
    colors = rng.choice(np.arange(1, 10), size=num_objects, replace=False)

    specs = []
    for color in colors:
        spec = _make_shape(rng)
        if spec is None:
            return None
        spec["color"] = int(color)
        specs.append(spec)

    specs.sort(key=lambda spec: _combined_span(spec["cells"], spec["move"]), reverse=True)

    input_grid = np.zeros((h, w), dtype=int)
    output_grid = np.zeros((h, w), dtype=int)
    blocked_in = set()
    blocked_out = set()

    for spec in specs:
        cells = spec["cells"]
        dr, dc = spec["move"]
        input_rel = {(r - dr, c - dc) for r, c in cells}
        combined = cells | input_rel

        min_r = min(r for r, _ in combined)
        max_r = max(r for r, _ in combined)
        min_c = min(c for _, c in combined)
        max_c = max(c for _, c in combined)

        row_lo = -min_r
        row_hi = h - 1 - max_r
        col_lo = -min_c
        col_hi = w - 1 - max_c
        if row_lo > row_hi or col_lo > col_hi:
            return None

        placed = False
        for _ in range(160):
            base_r = int(rng.integers(row_lo, row_hi + 1))
            base_c = int(rng.integers(col_lo, col_hi + 1))

            out_cells = _shift_cells(cells, base_r, base_c)
            in_cells = _shift_cells(input_rel, base_r, base_c)
            if out_cells & blocked_out or in_cells & blocked_in:
                continue

            for r, c in out_cells:
                output_grid[r, c] = spec["color"]
            for r, c in in_cells:
                input_grid[r, c] = spec["color"]
            blocked_out |= _expanded(out_cells, h, w)
            blocked_in |= _expanded(in_cells, h, w)
            placed = True
            break

        if not placed:
            return None

    if np.count_nonzero(input_grid) == 0:
        return None
    if np.array_equal(input_grid, output_grid):
        return None
    if not np.array_equal(main(input_grid), output_grid):
        return None

    return input_grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(400):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
