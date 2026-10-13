from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
YELLOW = 4
GREY = 5
PATH_COLORS = (1, 2, 3, 6, 7, 8, 9)
DIRS4 = ((-1, 0), (0, 1), (1, 0), (0, -1))
GEN_RETRIES = 300


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int, shape: tuple[int, int]):
    for dr, dc in DIRS4:
        nr, nc = r + dr, c + dc
        if _in_bounds(shape, nr, nc):
            yield nr, nc


def _path_colors(grid: np.ndarray) -> set[int]:
    return {int(v) for v in np.unique(grid) if int(v) not in (BG, YELLOW, GREY)}


def _find_grey_blocks(grid: np.ndarray) -> list[set[tuple[int, int]]]:
    shape = grid.shape
    seen: set[tuple[int, int]] = set()
    blocks: list[set[tuple[int, int]]] = []
    for r, c in map(tuple, np.argwhere(grid == GREY)):
        p = (int(r), int(c))
        if p in seen:
            continue
        q = deque([p])
        seen.add(p)
        comp: set[tuple[int, int]] = set()
        while q:
            cr, cc = q.popleft()
            comp.add((cr, cc))
            for nr, nc in _neighbors4(cr, cc, shape):
                if (nr, nc) not in seen and int(grid[nr, nc]) == GREY:
                    seen.add((nr, nc))
                    q.append((nr, nc))
        rows = [rr for rr, _ in comp]
        cols = [cc for _, cc in comp]
        if len(comp) == 4 and max(rows) - min(rows) == 1 and max(cols) - min(cols) == 1:
            blocks.append(comp)
    return blocks


def _build_membership(grid: np.ndarray) -> dict[int, set[tuple[int, int]]]:
    colors = _path_colors(grid)
    membership: dict[int, set[tuple[int, int]]] = {color: set() for color in colors}
    for color in colors:
        for r, c in map(tuple, np.argwhere(grid == color)):
            membership[color].add((int(r), int(c)))

    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            shown = int(grid[r, c])
            if shown not in colors:
                continue
            for color in colors - {shown}:
                vertical = (
                    r > 0
                    and r + 1 < h
                    and int(grid[r - 1, c]) == color
                    and int(grid[r + 1, c]) == color
                )
                horizontal = (
                    c > 0
                    and c + 1 < w
                    and int(grid[r, c - 1]) == color
                    and int(grid[r, c + 1]) == color
                )
                if vertical or horizontal:
                    membership[color].add((r, c))
    return membership


def _dirs_for(
    pos: tuple[int, int],
    color: int,
    membership: dict[int, set[tuple[int, int]]],
) -> list[tuple[int, int]]:
    cells = membership.get(color, set())
    r, c = pos
    out = []
    for dr, dc in DIRS4:
        if (r + dr, c + dc) in cells:
            out.append((dr, dc))
    return out


def _adjacent_block(
    pos: tuple[int, int], blocks: list[set[tuple[int, int]]]
) -> set[tuple[int, int]] | None:
    r, c = pos
    for block in blocks:
        for dr, dc in DIRS4:
            if (r + dr, c + dc) in block:
                return block
    return None


def _delivery_cell(
    grid: np.ndarray,
    out: np.ndarray,
    path_pos: tuple[int, int],
    block: set[tuple[int, int]],
) -> tuple[int, int]:
    pr, pc = path_pos
    side: tuple[int, int] | None = None
    for gr, gc in block:
        dr, dc = pr - gr, pc - gc
        if abs(dr) + abs(dc) == 1:
            side = (dr, dc)
            break
    if side is None:
        return path_pos

    candidates = []
    for gr, gc in sorted(block):
        rr, cc = gr + side[0], gc + side[1]
        if _in_bounds(grid.shape, rr, cc):
            candidates.append((rr, cc))

    for rr, cc in sorted(candidates, key=lambda p: abs(p[0] - pr) + abs(p[1] - pc)):
        if (rr, cc) != path_pos and int(grid[rr, cc]) == BG and int(out[rr, cc]) == BG:
            return rr, cc
    for rr, cc in candidates:
        if int(out[rr, cc]) in (BG, YELLOW):
            return rr, cc
    return path_pos


def _distance_to_block(
    start: tuple[int, int],
    direction: tuple[int, int],
    color: int,
    membership: dict[int, set[tuple[int, int]]],
    blocks: list[set[tuple[int, int]]],
    shape: tuple[int, int],
) -> int | None:
    pos = (start[0] + direction[0], start[1] + direction[1])
    prev = start
    steps = 1
    for _ in range(shape[0] * shape[1] + 5):
        if pos not in membership.get(color, set()):
            return None
        if _adjacent_block(pos, blocks) is not None:
            return steps
        dirs = [d for d in _dirs_for(pos, color, membership) if (pos[0] + d[0], pos[1] + d[1]) != prev]
        if not dirs:
            return None
        straight = direction if direction in dirs else None
        nd = straight if straight is not None else dirs[0]
        prev, pos, direction = pos, (pos[0] + nd[0], pos[1] + nd[1]), nd
        steps += 1
    return None


def _choose_direction(
    grid: np.ndarray,
    pos: tuple[int, int],
    current_color: int,
    direction: tuple[int, int],
    membership: dict[int, set[tuple[int, int]]],
    blocks: list[set[tuple[int, int]]],
) -> tuple[int, tuple[int, int]] | None:
    shown = int(grid[pos])
    colors_here = [color for color, cells in membership.items() if pos in cells]
    crossing = len(colors_here) > 1

    if crossing and shown in membership and shown != current_color:
        target_color = shown
        candidates = [
            d
            for d in _dirs_for(pos, target_color, membership)
            if d != (-direction[0], -direction[1]) and d[0] * direction[0] + d[1] * direction[1] == 0
        ]
        if not candidates:
            candidates = _dirs_for(pos, target_color, membership)
        scored = []
        for d in candidates:
            dist = _distance_to_block(pos, d, target_color, membership, blocks, grid.shape)
            if dist is not None:
                scored.append((dist, d))
        if scored:
            return target_color, min(scored)[1]
        if candidates:
            return target_color, candidates[0]
        return None

    dirs = _dirs_for(pos, current_color, membership)
    if direction in dirs:
        return current_color, direction
    back = (-direction[0], -direction[1])
    candidates = [d for d in dirs if d != back]
    if candidates:
        scored = []
        for d in candidates:
            dist = _distance_to_block(pos, d, current_color, membership, blocks, grid.shape)
            if dist is not None:
                scored.append((dist, d))
        if scored:
            return current_color, min(scored)[1]
        return current_color, candidates[0]
    return None


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    membership = _build_membership(g)
    blocks = _find_grey_blocks(g)

    starts = [tuple(map(int, p)) for p in np.argwhere(g == YELLOW)]
    for sr, sc in starts:
        out[sr, sc] = BG
        path_neighbors = [
            (nr, nc)
            for nr, nc in _neighbors4(sr, sc, g.shape)
            if int(g[nr, nc]) in membership
        ]
        if not path_neighbors:
            continue
        cr, cc = path_neighbors[0]
        color = int(g[cr, cc])
        direction = (cr - sr, cc - sc)
        prev = (sr, sc)

        for _ in range(g.shape[0] * g.shape[1] * 4):
            block = _adjacent_block((cr, cc), blocks)
            if block is not None:
                rr, cc2 = _delivery_cell(g, out, (cr, cc), block)
                out[rr, cc2] = YELLOW
                break

            choice = _choose_direction(g, (cr, cc), color, direction, membership, blocks)
            if choice is None:
                out[cr, cc] = YELLOW
                break
            color, direction = choice
            nr, nc = cr + direction[0], cc + direction[1]
            if not _in_bounds(g.shape, nr, nc) or (nr, nc) == prev:
                out[cr, cc] = YELLOW
                break
            prev = (cr, cc)
            cr, cc = nr, nc
    return out


def _draw_line(grid: np.ndarray, a: tuple[int, int], b: tuple[int, int], color: int) -> None:
    r0, c0 = a
    r1, c1 = b
    if r0 == r1:
        lo, hi = sorted((c0, c1))
        grid[r0, lo : hi + 1] = color
    elif c0 == c1:
        lo, hi = sorted((r0, r1))
        grid[lo : hi + 1, c0] = color
    else:
        raise ValueError("only orthogonal lines are supported")


def _place_warehouse(
    grid: np.ndarray,
    path_end: tuple[int, int],
    direction: tuple[int, int],
) -> bool:
    r, c = path_end
    dr, dc = direction
    if dr == 0:
        col = c + dc
        row = r - random.randint(0, 1)
        cells = [(row, col), (row + 1, col), (row, col + dc), (row + 1, col + dc)]
    else:
        row = r + dr
        col = c - random.randint(0, 1)
        cells = [(row, col), (row, col + 1), (row + dr, col), (row + dr, col + 1)]
    if not all(_in_bounds(grid.shape, rr, cc) for rr, cc in cells):
        return False
    if any(int(grid[rr, cc]) != BG for rr, cc in cells):
        return False
    for rr, cc in cells:
        grid[rr, cc] = GREY
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(18, 30)
    w = random.randint(18, 30)
    grid = np.zeros((h, w), dtype=int)
    cargo_color, signal_color = random.sample(PATH_COLORS, 2)
    horizontal = random.choice([True, False])
    turn = random.choice([True, False])

    if horizontal:
        r = random.randint(5, h - 6)
        c = random.randint(5, w - 6)
        start_left = random.choice([True, False])
        left = random.randint(2, c - 3)
        right = random.randint(c + 3, w - 3)
        top = random.randint(2, r - 3)
        bottom = random.randint(r + 3, h - 3)
        _draw_line(grid, (r, left), (r, right), cargo_color)
        _draw_line(grid, (top, c), (bottom, c), signal_color)
        start = (r, left) if start_left else (r, right)
        start_dir = (0, 1) if start_left else (0, -1)
        dest_dir = start_dir
        dest = (r, right) if start_left else (r, left)
        if turn:
            dest_dir = random.choice([(-1, 0), (1, 0)])
            dest = (top, c) if dest_dir == (-1, 0) else (bottom, c)
            grid[r, c] = signal_color
        else:
            grid[r, c] = cargo_color
    else:
        r = random.randint(5, h - 6)
        c = random.randint(5, w - 6)
        start_top = random.choice([True, False])
        top = random.randint(2, r - 3)
        bottom = random.randint(r + 3, h - 3)
        left = random.randint(2, c - 3)
        right = random.randint(c + 3, w - 3)
        _draw_line(grid, (top, c), (bottom, c), cargo_color)
        _draw_line(grid, (r, left), (r, right), signal_color)
        start = (top, c) if start_top else (bottom, c)
        start_dir = (1, 0) if start_top else (-1, 0)
        dest_dir = start_dir
        dest = (bottom, c) if start_top else (top, c)
        if turn:
            dest_dir = random.choice([(0, -1), (0, 1)])
            dest = (r, left) if dest_dir == (0, -1) else (r, right)
            grid[r, c] = signal_color
        else:
            grid[r, c] = cargo_color

    cargo_pos = (start[0] - start_dir[0], start[1] - start_dir[1])
    if not _in_bounds(grid.shape, cargo_pos[0], cargo_pos[1]):
        return None
    if int(grid[cargo_pos]) != BG:
        return None
    grid[cargo_pos] = YELLOW

    if not _place_warehouse(grid, dest, dest_dir):
        return None

    out = main(grid)
    if not isinstance(out, np.ndarray) or out.shape != grid.shape:
        return None
    if np.count_nonzero(out == YELLOW) != 1:
        return None
    if tuple(map(int, np.argwhere(out == YELLOW)[0])) == cargo_pos:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid path-transfer puzzle")
