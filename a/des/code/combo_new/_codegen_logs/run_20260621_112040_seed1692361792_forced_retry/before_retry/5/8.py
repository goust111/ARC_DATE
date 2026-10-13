from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
PIPE_COLORS = (1, 2, 3, 4, 5, 6, 7, 8, 9)
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_ATTEMPTS = 500


def _neighbors(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out = []
    for dr, dc in DIRS4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            out.append((nr, nc))
    return out


def _ball_offsets(outward: tuple[int, int], side_sign: int) -> list[tuple[int, int]]:
    dr, dc = outward
    left = (-dc, dr)
    if side_sign < 0:
        left = (-left[0], -left[1])
    return [(0, 0), (dr, dc), left, (dr + left[0], dc + left[1])]


def _find_ball_squares(grid: np.ndarray) -> list[dict[str, object]]:
    balls: list[dict[str, object]] = []
    for color in [int(v) for v in np.unique(grid) if int(v) != BG]:
        coords = np.argwhere(grid == color)
        if len(coords) != 4:
            continue
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        if (int(r1 - r0 + 1), int(c1 - c0 + 1)) != (2, 2):
            continue
        if np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color):
            balls.append({"color": color, "cells": [(int(r), int(c)) for r, c in coords]})
    return balls


def _reconstruct_mask(grid: np.ndarray, pipe_color: int, ball_mask: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    mask = g == pipe_color
    for r in range(h):
        for c in range(w):
            if mask[r, c] or ball_mask[r, c] or int(g[r, c]) == BG:
                continue
            vertical = 0 < r < h - 1 and int(g[r - 1, c]) == pipe_color and int(g[r + 1, c]) == pipe_color
            horizontal = 0 < c < w - 1 and int(g[r, c - 1]) == pipe_color and int(g[r, c + 1]) == pipe_color
            if vertical or horizontal:
                mask[r, c] = True
    return mask


def _follow_pipe(mask: np.ndarray, start: tuple[int, int], prev: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    h, w = map(int, mask.shape)
    cur = start
    before = prev
    while True:
        nxt = [(nr, nc) for nr, nc in _neighbors(cur[0], cur[1], h, w) if mask[nr, nc] and (nr, nc) != before]
        if not nxt:
            return cur, before
        if len(nxt) > 1:
            direction = (cur[0] - before[0], cur[1] - before[1])
            straight = (cur[0] + direction[0], cur[1] + direction[1])
            if straight in nxt:
                before, cur = cur, straight
                continue
        assert len(nxt) == 1
        before, cur = cur, nxt[0]


def _decode_ball_attachment(
    grid: np.ndarray,
    ball: dict[str, object],
    pipe_colors: set[int],
) -> tuple[int, int, tuple[int, int], tuple[int, int], int]:
    h, w = map(int, grid.shape)
    ball_cells = set(ball["cells"])
    for r, c in ball["cells"]:
        adjs = []
        for nr, nc in _neighbors(r, c, h, w):
            val = int(grid[nr, nc])
            if val in pipe_colors and (nr, nc) not in ball_cells:
                adjs.append((nr, nc, val))
        if len(adjs) != 1:
            continue
        (pr, pc, pipe_color) = adjs[0]
        outward = (r - pr, c - pc)
        offsets = {(rr - r, cc - c) for rr, cc in ball["cells"]}
        for side_sign in (-1, 1):
            expected = set(_ball_offsets(outward, side_sign))
            if offsets == expected:
                return pipe_color, side_sign, (r, c), (pr, pc), outward
    raise AssertionError("could not decode ball attachment")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Send each 2x2 signal ball through its colored pipe and redraw it at the opposite endpoint.

    Input structure:
    - The background is black.
    - Several one-pixel-wide colored pipe paths run through the grid; different pipes may cross orthogonally.
    - At a crossing, only one pipe color is visibly painted at the shared cell even though the other pipe continues straight through it.
    - Some pipe endpoints are covered by a monochrome 2x2 signal ball whose corner cell overlaps the endpoint.

    Rule:
    - Detect each ball and determine which pipe endpoint it covers by the corner cell that touches exactly one neighboring pipe pixel.
    - Reconstruct each pipe's hidden crossing cells by allowing a foreign-colored cell to count as part of the pipe whenever the pipe continues straight through that cell.
    - Starting from the covered endpoint, traverse the pipe to its other endpoint, always continuing straight through reconstructed crossings rather than turning into the perpendicular pipe.
    - Remove all pipes and redraw only the transported 2x2 balls at their destination endpoints, preserving each ball's handedness relative to the pipe so the square expands outward from the terminal endpoint.

    Key insight:
    - Crossings are deceptive because one pipe can visually disappear under another; the path must be recovered by straight-through continuity, and the ball's 2x2 orientation must then be transferred to the far endpoint.
    """
    g = np.asarray(grid, dtype=int)
    balls = _find_ball_squares(g)
    ball_colors = {int(ball["color"]) for ball in balls}
    pipe_colors = {int(v) for v in np.unique(g) if int(v) not in ball_colors and int(v) != BG}
    ball_mask = np.isin(g, list(ball_colors))
    masks = {color: _reconstruct_mask(g, color, ball_mask) for color in pipe_colors}

    out = np.zeros_like(g)
    for ball in balls:
        pipe_color, side_sign, overlap, start_cell, _outward = _decode_ball_attachment(g, ball, pipe_colors)
        end_cell, prev_cell = _follow_pipe(masks[pipe_color], start_cell, overlap)
        outward = (end_cell[0] - prev_cell[0], end_cell[1] - prev_cell[1])
        for dr, dc in _ball_offsets(outward, side_sign):
            rr = end_cell[0] + dr
            cc = end_cell[1] + dc
            out[rr, cc] = int(ball["color"])
    return out


def _polyline_cells(points: list[tuple[int, int]]) -> tuple[list[tuple[int, int]], dict[tuple[int, int], str]]:
    seq: list[tuple[int, int]] = []
    for (r0, c0), (r1, c1) in zip(points, points[1:]):
        dr = 0 if r0 == r1 else (1 if r1 > r0 else -1)
        dc = 0 if c0 == c1 else (1 if c1 > c0 else -1)
        length = abs(r1 - r0) + abs(c1 - c0)
        segment = [(r0 + dr * i, c0 + dc * i) for i in range(length + 1)]
        if seq:
            segment = segment[1:]
        seq.extend(segment)
    if len(set(seq)) != len(seq):
        raise AssertionError("self-intersection")
    dirs: dict[tuple[int, int], set[tuple[int, int]]] = {cell: set() for cell in seq}
    for a, b in zip(seq, seq[1:]):
        dirs[a].add((b[0] - a[0], b[1] - a[1]))
        dirs[b].add((a[0] - b[0], a[1] - b[1]))
    kinds: dict[tuple[int, int], str] = {}
    for cell, neigh_dirs in dirs.items():
        if len(neigh_dirs) == 1:
            kinds[cell] = "end"
        else:
            d0, d1 = list(neigh_dirs)
            if d0[0] == -d1[0] and d0[1] == -d1[1]:
                kinds[cell] = "h" if d0[0] == 0 else "v"
            else:
                kinds[cell] = "bend"
    return seq, kinds


def _render_pipe_grid(
    h: int,
    w: int,
    pipe_specs: list[dict[str, object]],
    balls: list[dict[str, object]],
    rng: random.Random,
) -> np.ndarray:
    cell_owners: dict[tuple[int, int], list[int]] = {}
    for idx, spec in enumerate(pipe_specs):
        for cell in spec["seq"]:
            cell_owners.setdefault(cell, []).append(idx)
    grid = np.zeros((h, w), dtype=int)
    for cell, owners in cell_owners.items():
        grid[cell] = int(pipe_specs[rng.choice(owners)]["color"])
    for ball in balls:
        for r, c in ball["cells"]:
            grid[r, c] = int(ball["color"])
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        h = rng.randint(18, 26)
        w = rng.randint(18, 26)
        pipe_count = rng.randint(2, 4)
        colors = rng.sample(PIPE_COLORS, pipe_count * 2)
        pipe_colors = colors[:pipe_count]
        ball_colors = colors[pipe_count:]

        occupied: dict[tuple[int, int], tuple[int, str]] = {}
        pipe_specs: list[dict[str, object]] = []
        ok = True
        for pipe_idx in range(pipe_count):
            placed = False
            for _try in range(300):
                if rng.random() < 0.5:
                    r0, r1 = rng.sample(range(3, h - 3), 2)
                    c0, c1, cmid = rng.sample(range(3, w - 3), 3)
                    if abs(cmid - c0) < 2 or abs(c1 - cmid) < 2 or abs(r1 - r0) < 2:
                        continue
                    points = [(r0, c0), (r0, cmid), (r1, cmid), (r1, c1)]
                else:
                    c0, c1 = rng.sample(range(3, w - 3), 2)
                    r0, r1, rmid = rng.sample(range(3, h - 3), 3)
                    if abs(rmid - r0) < 2 or abs(r1 - rmid) < 2 or abs(c1 - c0) < 2:
                        continue
                    points = [(r0, c0), (rmid, c0), (rmid, c1), (r1, c1)]
                try:
                    seq, kinds = _polyline_cells(points)
                except AssertionError:
                    continue
                legal = True
                for cell in seq:
                    if cell not in occupied:
                        continue
                    _other_idx, other_kind = occupied[cell]
                    new_kind = kinds[cell]
                    if new_kind not in ("h", "v") or other_kind not in ("h", "v") or new_kind == other_kind:
                        legal = False
                        break
                if not legal:
                    continue
                for cell in seq:
                    occupied[cell] = (pipe_idx, kinds[cell])
                pipe_specs.append({"seq": seq, "kinds": kinds, "color": pipe_colors[pipe_idx]})
                placed = True
                break
            if not placed:
                ok = False
                break
        if not ok:
            continue

        ball_specs: list[dict[str, object]] = []
        used_ball_cells: set[tuple[int, int]] = set()
        used_dest_cells: set[tuple[int, int]] = set()
        pipe_cells = set(occupied)
        chosen_pipes = rng.sample(range(pipe_count), rng.randint(1, pipe_count))
        for ball_idx, pipe_idx in enumerate(chosen_pipes):
            seq = pipe_specs[pipe_idx]["seq"]
            endpoints = [
                (seq[0], seq[1], seq[-1], seq[-2]),
                (seq[-1], seq[-2], seq[0], seq[1]),
            ]
            rng.shuffle(endpoints)
            placed = False
            for endpoint, next_cell, dest_endpoint, dest_prev in endpoints:
                outward = (endpoint[0] - next_cell[0], endpoint[1] - next_cell[1])
                dest_outward = (dest_endpoint[0] - dest_prev[0], dest_endpoint[1] - dest_prev[1])
                for side_sign in rng.sample([-1, 1], 2):
                    cells = [(endpoint[0] + dr, endpoint[1] + dc) for dr, dc in _ball_offsets(outward, side_sign)]
                    dest_cells = [
                        (dest_endpoint[0] + dr, dest_endpoint[1] + dc) for dr, dc in _ball_offsets(dest_outward, side_sign)
                    ]
                    if any(not (0 <= r < h and 0 <= c < w) for r, c in cells):
                        continue
                    if any(not (0 <= r < h and 0 <= c < w) for r, c in dest_cells):
                        continue
                    if any((r, c) in used_ball_cells for r, c in cells):
                        continue
                    if any((r, c) in used_dest_cells for r, c in dest_cells):
                        continue
                    cell_set = set(cells)
                    valid_attachment = True
                    for cell in cells:
                        adj_pipe = [(nr, nc) for nr, nc in _neighbors(cell[0], cell[1], h, w) if (nr, nc) in pipe_cells and (nr, nc) not in cell_set]
                        if cell == endpoint:
                            if adj_pipe != [next_cell]:
                                valid_attachment = False
                                break
                        elif adj_pipe:
                            valid_attachment = False
                            break
                    if not valid_attachment:
                        continue
                    collision = False
                    for r, c in cells:
                        if (r, c) != endpoint and (r, c) in occupied:
                            collision = True
                            break
                    if collision:
                        continue
                    used_ball_cells.update(cells)
                    used_dest_cells.update(dest_cells)
                    ball_specs.append({"color": ball_colors[ball_idx], "cells": cells})
                    placed = True
                    break
                if placed:
                    break
            if not placed:
                ok = False
                break
        if not ok:
            continue

        grid = _render_pipe_grid(h, w, pipe_specs, ball_specs, rng)
        try:
            solved = main(grid)
            if isinstance(solved, np.ndarray) and solved.shape == grid.shape:
                return grid
        except Exception:
            continue

    raise RuntimeError("failed to generate input")
