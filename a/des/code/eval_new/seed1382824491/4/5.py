from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def _neighbors4(r: int, c: int):
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _sequence_bar(grid: np.ndarray) -> tuple[list[int], set[tuple[int, int]]]:
    top = [int(x) for x in grid[0].tolist()]
    runs: list[tuple[int, int]] = []
    c = 0
    while c < len(top):
        if top[c] in (BG, GREY):
            c += 1
            continue
        c0 = c
        while c < len(top) and top[c] not in (BG, GREY):
            c += 1
        if c - c0 >= 2:
            runs.append((c0, c - 1))
    if not runs:
        return [], set()
    c0, c1 = max(runs, key=lambda x: (x[1] - x[0], x[0]))
    cells = {(0, c) for c in range(c0, c1 + 1)}
    return [int(grid[0, c]) for c in range(c0, c1 + 1)], cells


def _grey_components(grid: np.ndarray) -> list[set[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros_like(grid, dtype=bool)
    comps: list[set[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) != GREY or bool(seen[r, c]):
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp = {(r, c)}
            while q:
                rr, cc = q.popleft()
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) == GREY and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
                        comp.add((nr, nc))
            comps.append(comp)
    return comps


def _wall_lookup(grid: np.ndarray):
    lookup = {}
    for comp in _grey_components(grid):
        rows = {r for r, _ in comp}
        cols = {c for _, c in comp}
        orient = "h" if len(rows) == 1 else "v"
        for cell in comp:
            lookup[cell] = (orient, comp)
    return lookup


def _initial_direction(grid: np.ndarray, r: int, c: int) -> tuple[int, int] | None:
    h, w = grid.shape
    free = []
    for dr, dc in DIRS.values():
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) == BG:
            free.append((dr, dc))
    return free[0] if len(free) == 1 else None


def _seed_cells(grid: np.ndarray, seq_cells: set[tuple[int, int]]) -> list[tuple[int, int, int]]:
    out = []
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            val = int(grid[r, c])
            if val in (BG, GREY) or (r, c) in seq_cells:
                continue
            if _initial_direction(grid, r, c) is not None:
                out.append((r, c, val))
    return sorted(out)


def _reflect(
    wall_cell: tuple[int, int],
    current: tuple[int, int],
    direction: tuple[int, int],
    wall_info,
) -> tuple[int, int]:
    orient, comp = wall_info
    wr, wc = wall_cell
    if orient == "v":
        rs = [r for r, _ in comp]
        if wr == min(rs):
            cand = DIRS["up"]
        elif wr == max(rs):
            cand = DIRS["down"]
        else:
            cand = (-direction[0], -direction[1])
    else:
        cs = [c for _, c in comp]
        if wc == min(cs):
            cand = DIRS["left"]
        elif wc == max(cs):
            cand = DIRS["right"]
        else:
            cand = (-direction[0], -direction[1])
    return cand


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace colored rays from seeds, reflecting off gray walls and cycling colors after each reflection.

    Input structure:
    - Black background contains gray horizontal or vertical wall segments.
    - Non-gray singleton seeds each have exactly one black 4-neighbor; that cell gives the initial ray direction.
    - The top row contains a compact color sequence bar used after reflections.

    Rule:
    - From each seed, step along the unique open direction, painting the first segment in the seed color.
    - When the next cell is a gray wall, reflect: wall-middle hits reverse direction, wall-endpoint hits turn along the wall away from the wall body.
    - After every reflection, advance to the next color from the top-right sequence, cycling through it.
    - Stop a ray at the grid edge, another colored path, another seed, or when no progress is possible; keep walls and seeds unchanged.

    Key insight:
    - The local seed opening gives launch direction, while endpoint-versus-middle wall contact controls whether the ray turns or bounces.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seq, seq_cells = _sequence_bar(g)
    seq = seq or [1, 2, 3]
    walls = _wall_lookup(g)
    seeds = _seed_cells(g, seq_cells)
    seed_positions = {(r, c) for r, c, _ in seeds}
    out = g.copy()

    for sr, sc, seed_color in seeds:
        direction = _initial_direction(g, sr, sc)
        if direction is None:
            continue
        r, c = sr, sc
        color = int(seed_color)
        reflections = 0
        seen_states = set()
        for _ in range(h * w * 4):
            state = (r, c, direction, color)
            if state in seen_states:
                break
            seen_states.add(state)
            nr, nc = r + direction[0], c + direction[1]
            if not (0 <= nr < h and 0 <= nc < w):
                break
            if (nr, nc) in walls:
                direction = _reflect((nr, nc), (r, c), direction, walls[(nr, nc)])
                color = int(seq[reflections % len(seq)])
                reflections += 1
                continue
            if (nr, nc) in seed_positions:
                break
            if int(out[nr, nc]) != BG:
                break
            out[nr, nc] = color
            r, c = nr, nc
    return out


def _add_vertical_wall(grid: np.ndarray, c: int, r0: int, r1: int) -> None:
    for r in range(max(0, r0), min(grid.shape[0], r1 + 1)):
        grid[r, c] = GREY


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(12, 24)
    w = random.randint(14, 28)
    grid = np.full((h, w), BG, dtype=int)

    seq_len = random.randint(3, 4)
    seq = random.sample([c for c in COLORS if c != GREY], seq_len)
    start_c = w - seq_len - random.randint(0, 1)
    grid[0, start_c : start_c + seq_len] = seq

    seed_r = random.randint(4, h - 5)
    seed_color = random.choice([c for c in COLORS if c not in seq])
    grid[seed_r, 0] = seed_color
    _add_vertical_wall(grid, 0, seed_r - 2, seed_r - 1)
    _add_vertical_wall(grid, 0, seed_r + 1, seed_r + 2)

    wall_c = random.randint(5, min(w - seq_len - 4, w - 5))
    mode = random.choice(("middle", "turn_up", "turn_down"))
    if mode == "middle":
        _add_vertical_wall(grid, wall_c, seed_r - 2, seed_r + 2)
    elif mode == "turn_up":
        _add_vertical_wall(grid, wall_c, seed_r, min(h - 2, seed_r + random.randint(2, 5)))
    else:
        _add_vertical_wall(grid, wall_c, max(1, seed_r - random.randint(2, 5)), seed_r)

    # Optional second independent seed, separated vertically.
    if random.random() < 0.35:
        r2_choices = [r for r in range(4, h - 4) if abs(r - seed_r) >= 5]
        if r2_choices:
            r2 = random.choice(r2_choices)
            color2 = random.choice([c for c in COLORS if c not in seq and c != seed_color])
            grid[r2, w - 1] = color2
            _add_vertical_wall(grid, w - 1, r2 - 2, r2 - 1)
            _add_vertical_wall(grid, w - 1, r2 + 1, r2 + 2)
            wc2 = random.randint(max(3, start_c - 8), start_c - 2)
            _add_vertical_wall(grid, wc2, max(1, r2 - 2), min(h - 2, r2 + 2))

    try:
        out = main(grid)
    except Exception:
        return None
    if out.shape != grid.shape or max(out.shape) > 30:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid reflecting-ray input")
