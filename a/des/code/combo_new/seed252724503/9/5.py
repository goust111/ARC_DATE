from __future__ import annotations

import random

import numpy as np


BG = 0
GEN_ATTEMPTS = 300


DIRS = {
    0: (0, 1),
    1: (1, 0),
    2: (0, -1),
    3: (-1, 0),
}


def _trace_path(mask: np.ndarray) -> list[tuple[int, int]]:
    coords = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    neighbors = {
        cell: [
            (cell[0] + dr, cell[1] + dc)
            for dr, dc in DIRS.values()
            if (cell[0] + dr, cell[1] + dc) in set(coords)
        ]
        for cell in coords
    }
    endpoints = [cell for cell, nbrs in neighbors.items() if len(nbrs) == 1]
    assert len(endpoints) == 2
    start = min(endpoints, key=lambda item: (item[1], item[0]))
    path = [start]
    prev = None
    cur = start
    while True:
        nxts = [n for n in neighbors[cur] if n != prev]
        if not nxts:
            break
        assert len(nxts) == 1
        nxt = nxts[0]
        path.append(nxt)
        prev, cur = cur, nxt
    return path


def _turn_sequence(path: list[tuple[int, int]]) -> list[str]:
    seq: list[str] = []
    for i in range(1, len(path) - 1):
        dr1 = path[i][0] - path[i - 1][0]
        dc1 = path[i][1] - path[i - 1][1]
        dr2 = path[i + 1][0] - path[i][0]
        dc2 = path[i + 1][1] - path[i][1]
        if (dr1, dc1) == (dr2, dc2):
            continue
        cross = dr1 * dc2 - dc1 * dr2
        seq.append("R" if cross > 0 else "L")
    return seq


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the upper polyline's left/right turn program and replay it as a unit-step drawing from the lower marker.

    Input structure:
    - The background is dominant.
    - One foreground color forms a single one-pixel-wide orthogonal path with two endpoints and several right-angle bends.
    - A different single-pixel marker sits in a blank lower area.

    Rule:
    - Trace the upper path from its leftmost endpoint to its other endpoint and record the sequence of left and right turns.
    - Keep the original path unchanged.
    - Starting from the lower marker, use the marker's color to draw a new path whose initial direction is right.
    - Move one cell per segment; after each move apply the next recorded turn, and after the final turn draw one more final step.

    Key insight:
    - The upper drawing is not copied geometrically: only its relative turn sequence matters, and that sequence is executed again at a different scale in the lower blank area.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    colors = [int(v) for v in np.unique(g) if int(v) != BG]
    counts = {color: int(np.sum(g == color)) for color in colors}
    marker_color = min(colors, key=lambda color: counts[color])
    assert counts[marker_color] == 1
    path_color = next(color for color in colors if color != marker_color)

    path = _trace_path(g == path_color)
    turns = _turn_sequence(path)
    seed = tuple(map(int, np.argwhere(g == marker_color)[0]))

    out = g.copy()
    direction = 0
    r, c = seed
    for turn in turns:
        dr, dc = DIRS[direction]
        r += dr
        c += dc
        out[r, c] = marker_color
        direction = (direction + 1) % 4 if turn == "R" else (direction - 1) % 4
    dr, dc = DIRS[direction]
    out[r + dr, c + dc] = marker_color
    return out


def _simulate_turn_program(turns: list[str]) -> list[tuple[int, int]]:
    pts = [(0, 0)]
    direction = 0
    r = c = 0
    for turn in turns:
        dr, dc = DIRS[direction]
        r += dr
        c += dc
        pts.append((r, c))
        direction = (direction + 1) % 4 if turn == "R" else (direction - 1) % 4
    dr, dc = DIRS[direction]
    pts.append((r + dr, c + dc))
    return pts


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        turn_count = random.randint(2, 6)
        directions = [0]
        for _turn in range(turn_count):
            vertical = random.choice([1, 3])
            directions.append(vertical)
            directions.append(0)
        seg_lengths = [random.randint(2, 4) if d == 0 else random.randint(1, 3) for d in directions]

        cells = [(0, 0)]
        r = c = 0
        used = {(0, 0)}
        ok = True
        for direction, seg_len in zip(directions, seg_lengths):
            dr, dc = DIRS[direction]
            for _step in range(seg_len):
                r += dr
                c += dc
                if (r, c) in used:
                    ok = False
                    break
                cells.append((r, c))
                used.add((r, c))
            if not ok:
                break
        if not ok:
            continue

        min_r = min(r for r, _ in cells)
        max_r = max(r for r, _ in cells)
        min_c = min(c for _, c in cells)
        max_c = max(c for _, c in cells)
        path_h = max_r - min_r + 1
        path_w = max_c - min_c + 1
        turns = _turn_sequence(cells)
        small = _simulate_turn_program(turns)
        sh = max(r for r, _ in small) - min(r for r, _ in small) + 1
        sw = max(c for _, c in small) - min(c for _, c in small) + 1

        h = path_h + sh + random.randint(6, 10)
        w = max(path_w, sw) + random.randint(8, 12)
        if h > 30 or w > 30:
            continue

        grid = np.full((h, w), BG, dtype=int)
        path_color, marker_color = random.sample([1, 2, 3, 4, 6, 7, 8, 9], 2)

        top_r = random.randint(1, 3)
        left_c = random.randint(1, max(1, w - path_w - 2))
        norm_cells = [(top_r + rr - min_r, left_c + cc - min_c) for rr, cc in cells]
        for rr, cc in norm_cells:
            grid[rr, cc] = path_color

        low_r = top_r + path_h + 2
        high_r = h - sh - 2
        if low_r > high_r:
            continue
        seed_r = random.randint(low_r, high_r)
        seed_c = random.randint(1 - min(c for _, c in small), w - 2 - max(c for _, c in small))
        grid[seed_r, seed_c] = marker_color
        return grid
    raise RuntimeError("failed to generate a valid instance")
