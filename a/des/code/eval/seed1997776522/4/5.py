from __future__ import annotations

import heapq
import random
from collections import defaultdict

import numpy as np


BG = 0
GREY = 5
YELLOW = 4
MAROON = 9

DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]


def _in_bounds(r: int, c: int, h: int, w: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _uniform_2x2_blocks(g: np.ndarray, color: int | None = None) -> list[tuple[int, int, int]]:
    h, w = g.shape
    blocks = []
    for r in range(h - 1):
        for c in range(w - 1):
            vals = g[r : r + 2, c : c + 2]
            if color is not None:
                if np.all(vals == color):
                    blocks.append((r, c, int(color)))
            elif np.all(vals == int(vals[0, 0])):
                col = int(vals[0, 0])
                if col not in (BG, GREY, MAROON, YELLOW):
                    blocks.append((r, c, col))
    return blocks


def _block_cells(block: tuple[int, int, int]) -> set[tuple[int, int]]:
    r, c, _ = block
    return {(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)}


def _neighbor_pipe_cells(
    cells: set[tuple[int, int]], passable: set[tuple[int, int]]
) -> list[tuple[int, int, int]]:
    out = []
    for r, c in cells:
        for d, (dr, dc) in enumerate(DIRS):
            nr, nc = r + dr, c + dc
            if (nr, nc) in passable:
                out.append((nr, nc, d))
    return out


def _displays(g: np.ndarray) -> tuple[list[tuple[int, int, int]], set[tuple[int, int]]]:
    displays = []
    cells: set[tuple[int, int]] = set()
    for block in _uniform_2x2_blocks(g, GREY):
        b_cells = _block_cells(block)
        outside_grey = 0
        for r, c in b_cells:
            for dr, dc in DIRS:
                nr, nc = r + dr, c + dc
                if _in_bounds(nr, nc, *g.shape) and (nr, nc) not in b_cells and int(g[nr, nc]) == GREY:
                    outside_grey += 1
        if outside_grey:
            displays.append(block)
            cells.update(b_cells)
    return displays, cells


def _sources(g: np.ndarray) -> list[tuple[int, int, int]]:
    blocks = _uniform_2x2_blocks(g, None)
    # Keep only maximal-looking 2x2 source blocks; generated inputs do not use
    # larger colored rectangles, but this guard avoids duplicate nested scans.
    kept = []
    seen: set[tuple[int, int]] = set()
    for r, c, col in blocks:
        if any((rr, cc) in seen for rr in (r, r + 1) for cc in (c, c + 1)):
            continue
        kept.append((r, c, col))
        seen.update(_block_cells((r, c, col)))
    return kept


def _propagate(g: np.ndarray) -> dict[tuple[int, int], tuple[int, set[int]]]:
    displays, display_cells = _displays(g)
    del displays
    h, w = g.shape
    passable = {
        (int(r), int(c))
        for r, c in np.argwhere(g == GREY)
        if (int(r), int(c)) not in display_cells
    }

    pq: list[tuple[int, int, int, int, int]] = []
    best_state: dict[tuple[int, int, int, int], int] = {}
    for block in _sources(g):
        col = block[2]
        for nr, nc, d in _neighbor_pipe_cells(_block_cells(block), passable):
            key = (nr, nc, d, col)
            if best_state.get(key, 10**9) > 1:
                best_state[key] = 1
                heapq.heappush(pq, (1, nr, nc, d, col))

    while pq:
        cost, r, c, d, col = heapq.heappop(pq)
        if best_state.get((r, c, d, col)) != cost:
            continue
        for nd, (dr, dc) in enumerate(DIRS):
            nr, nc = r + dr, c + dc
            if (nr, nc) not in passable:
                continue
            ncost = cost + (0 if nd == d else 1)
            key = (nr, nc, nd, col)
            if ncost < best_state.get(key, 10**9):
                best_state[key] = ncost
                heapq.heappush(pq, (ncost, nr, nc, nd, col))

    cell_costs: dict[tuple[int, int], dict[int, set[int]]] = defaultdict(lambda: defaultdict(set))
    for (r, c, _d, col), cost in best_state.items():
        cell_costs[(r, c)][cost].add(col)

    result = {}
    for cell, by_cost in cell_costs.items():
        m = min(by_cost)
        result[cell] = (m, set(by_cost[m]))
    return result


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    displays, display_cells = _displays(g)
    arrivals = _propagate(g)

    for (r, c), (_cost, colors) in arrivals.items():
        if (r, c) in display_cells:
            continue
        out[r, c] = next(iter(colors)) if len(colors) == 1 else YELLOW

    for block in displays:
        b_cells = _block_cells(block)
        options: dict[int, set[int]] = defaultdict(set)
        for r, c in b_cells:
            for dr, dc in DIRS:
                nr, nc = r + dr, c + dc
                if (nr, nc) in b_cells:
                    continue
                if (nr, nc) in arrivals:
                    cost, colors = arrivals[(nr, nc)]
                    for col in colors:
                        options[cost].add(col)
        if options:
            m = min(options)
            if len(options[m]) == 1:
                fill = next(iter(options[m]))
                for r, c in b_cells:
                    out[r, c] = fill
    return out


def _draw_line(g: np.ndarray, a: tuple[int, int], b: tuple[int, int]) -> None:
    r0, c0 = a
    r1, c1 = b
    if r0 == r1:
        lo, hi = sorted((c0, c1))
        for c in range(lo, hi + 1):
            if int(g[r0, c]) == BG:
                g[r0, c] = GREY
    elif c0 == c1:
        lo, hi = sorted((r0, r1))
        for r in range(lo, hi + 1):
            if int(g[r, c0]) == BG:
                g[r, c0] = GREY
    else:
        raise ValueError("only orthogonal segments are supported")


def _put_block(g: np.ndarray, r: int, c: int, color: int) -> None:
    g[r : r + 2, c : c + 2] = color


def _straight_case(rng: random.Random, valve: bool) -> np.ndarray:
    h = rng.randint(8, 14)
    w = rng.randint(14, 24)
    r = rng.randint(1, h - 3)
    color = rng.choice([1, 2, 3, 6, 7, 8])
    g = np.full((h, w), BG, dtype=int)
    _put_block(g, r, 0, color)
    _put_block(g, r, w - 2, GREY)
    _draw_line(g, (r, 2), (r, w - 3))
    if valve:
        g[r, rng.randint(5, w - 6)] = MAROON
    return g


def _tie_or_priority_case(rng: random.Random, tie: bool) -> np.ndarray:
    h = rng.randint(16, 22)
    w = rng.randint(16, 24)
    ir = rng.randint(7, h - 6)
    ic = rng.randint(6, w - 8)
    left_color, top_color = rng.sample([1, 2, 3, 6, 7, 8], 2)
    g = np.full((h, w), BG, dtype=int)

    _put_block(g, ir, 0, left_color)
    _draw_line(g, (ir, 2), (ir, ic))

    if tie:
        _put_block(g, 0, ic, top_color)
        _draw_line(g, (2, ic), (h - 3, ic))
    else:
        sc = min(w - 2, ic + rng.randint(3, 5))
        _put_block(g, 0, sc, top_color)
        _draw_line(g, (2, sc), (ir - 2, sc))
        _draw_line(g, (ir - 2, sc), (ir - 2, ic))
        _draw_line(g, (ir - 2, ic), (h - 3, ic))

    _put_block(g, h - 2, ic, GREY)
    if rng.random() < 0.35 and ir + 2 < h - 3:
        g[rng.randint(ir + 1, h - 4), ic] = MAROON
    return g


def _multi_case(rng: random.Random) -> np.ndarray:
    top = _straight_case(rng, valve=False)
    bottom = _straight_case(rng, valve=True)
    w = max(top.shape[1], bottom.shape[1]) + 2
    h = top.shape[0] + bottom.shape[0] + 2
    g = np.full((h, w), BG, dtype=int)
    g[: top.shape[0], : top.shape[1]] = top
    g[top.shape[0] + 2 :, 2 : 2 + bottom.shape[1]] = bottom
    return g


def generate_input() -> np.ndarray:
    rng = random.Random()
    makers = [
        lambda: _straight_case(rng, False),
        lambda: _straight_case(rng, True),
        lambda: _tie_or_priority_case(rng, True),
        lambda: _tie_or_priority_case(rng, False),
        lambda: _multi_case(rng),
    ]
    for _ in range(200):
        g = rng.choice(makers)()
        try:
            out = main(g)
        except Exception:
            continue
        if isinstance(out, np.ndarray) and out.shape == g.shape:
            return g
    raise RuntimeError("failed to generate valid signal network")
