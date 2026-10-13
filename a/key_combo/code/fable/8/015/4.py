from __future__ import annotations

from collections import deque

import numpy as np


BLACK = 0
YELLOW = 4
TEAL = 8
SEED_COLORS = (1, 2, 3, 5, 6, 7, 9)
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _bfs_corridor(grid: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    h, w = grid.shape
    dist = np.full((h, w), -1, dtype=int)
    q = deque([start])
    dist[start] = 0
    while q:
        r, c = q.popleft()
        base = int(dist[r, c]) + 1
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and dist[nr, nc] == -1 and int(grid[nr, nc]) == BLACK:
                dist[nr, nc] = base
                q.append((nr, nc))
    return dist


def _black_components(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != BLACK:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp = [(r, c)]
            while q:
                cr, cc = q.popleft()
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == BLACK:
                        seen[nr, nc] = True
                        q.append((nr, nc))
                        comp.append((nr, nc))
            comps.append(comp)
    return comps


def _carve_maze(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    grid = np.full((h, w), TEAL, dtype=int)
    mh = (h - 1) // 2
    mw = (w - 1) // 2
    visited = np.zeros((mh, mw), dtype=bool)
    start = (int(rng.integers(mh)), int(rng.integers(mw)))
    stack = [start]
    visited[start] = True
    grid[2 * start[0] + 1, 2 * start[1] + 1] = BLACK

    while stack:
        r, c = stack[-1]
        choices = []
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < mh and 0 <= nc < mw and not visited[nr, nc]:
                choices.append((nr, nc, dr, dc))
        if not choices:
            stack.pop()
            continue
        nr, nc, dr, dc = choices[int(rng.integers(len(choices)))]
        grid[2 * r + 1 + dr, 2 * c + 1 + dc] = BLACK
        grid[2 * nr + 1, 2 * nc + 1] = BLACK
        visited[nr, nc] = True
        stack.append((nr, nc))

    extra_links = int(rng.integers(max(1, (mh * mw) // 10), max(2, (mh * mw) // 5) + 1))
    for _ in range(extra_links):
        r = int(rng.integers(mh))
        c = int(rng.integers(mw))
        dirs = []
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < mh and 0 <= nc < mw:
                wr = 2 * r + 1 + dr
                wc = 2 * c + 1 + dc
                if int(grid[wr, wc]) == TEAL:
                    dirs.append((wr, wc))
        if dirs:
            wr, wc = dirs[int(rng.integers(len(dirs)))]
            grid[wr, wc] = BLACK
    return grid


def _add_isolated_black_cells(grid: np.ndarray, rng: np.random.Generator, count: int) -> int:
    h, w = grid.shape
    candidates = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if int(grid[r, c]) != TEAL:
                continue
            if any(int(grid[nr, nc]) != TEAL for nr, nc in _neighbors4(r, c)):
                continue
            candidates.append((r, c))
    if not candidates:
        return 0
    rng.shuffle(candidates)
    placed = 0
    for r, c in candidates:
        if any(int(grid[nr, nc]) != TEAL for nr, nc in _neighbors4(r, c)):
            continue
        grid[r, c] = BLACK
        placed += 1
        if placed >= count:
            break
    return placed


def _choose_seed_pair(
    base: np.ndarray,
    corridor: list[tuple[int, int]],
    seed_colors: tuple[int, int],
    rng: np.random.Generator,
    need_frontier: bool,
) -> np.ndarray | None:
    if len(corridor) < 12:
        return None
    cells = list(corridor)
    for _ in range(120):
        a = cells[int(rng.integers(len(cells)))]
        b = cells[int(rng.integers(len(cells)))]
        if a == b:
            continue
        if abs(a[0] - b[0]) + abs(a[1] - b[1]) < 6:
            continue
        grid = base.copy()
        grid[a] = int(seed_colors[0])
        grid[b] = int(seed_colors[1])
        out = main(grid)
        yellow = int(np.count_nonzero(out == YELLOW))
        if need_frontier and yellow == 0:
            continue
        if int(np.count_nonzero(out == int(seed_colors[0]))) <= 4:
            continue
        if int(np.count_nonzero(out == int(seed_colors[1]))) <= 4:
            continue
        return grid
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split each reachable corridor between two seeds by shortest-path distance, marking ties in yellow.

    Input structure:
    - Teal pixels are maze walls and black pixels are one-cell-wide corridors.
    - Exactly two distinct non-black, non-teal seed pixels sit on corridor cells.
    - Yellow does not appear in the input, and some black corridor cells may be disconnected from both seeds.

    Rule:
    - For every black corridor cell that can be reached from either seed by 4-connected corridor steps,
      compute its shortest-path distance to both seeds while moving only through corridor cells.
    - Recolor the cell to the nearer seed's color when one distance is strictly smaller.
    - Recolor the cell yellow when the two distances are equal; leave unreachable black cells, walls,
      and the seed pixels themselves unchanged.

    Key insight:
    - The output is a two-source geodesic Voronoi diagram drawn inside the maze corridors, with yellow
      showing the equidistant watershed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    seed_colors = [c for c in np.unique(g) if int(c) not in (BLACK, YELLOW, TEAL)]
    assert len(seed_colors) == 2, "expected exactly two seed colors"
    seeds: list[tuple[int, int]] = []
    for color in seed_colors:
        coords = np.argwhere(g == int(color))
        assert len(coords) == 1, "expected one seed pixel per seed color"
        seeds.append((int(coords[0, 0]), int(coords[0, 1])))

    d0 = _bfs_corridor(g, seeds[0])
    d1 = _bfs_corridor(g, seeds[1])
    out = g.copy()

    black_cells = np.argwhere(g == BLACK)
    for r, c in black_cells:
        a = int(d0[r, c])
        b = int(d1[r, c])
        if a == -1 and b == -1:
            continue
        if b == -1 or (a != -1 and a < b):
            out[r, c] = int(seed_colors[0])
        elif a == -1 or b < a:
            out[r, c] = int(seed_colors[1])
        else:
            out[r, c] = YELLOW
    return out


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(400):
        h = int(rng.integers(6, 15)) * 2 + 1
        w = int(rng.integers(6, 15)) * 2 + 1
        grid = _carve_maze(h, w, rng)

        need_unreachable = bool(rng.random() < 0.55)
        if need_unreachable:
            added = _add_isolated_black_cells(grid, rng, int(rng.integers(1, 5)))
            if added == 0:
                continue

        comps = _black_components(grid)
        if not comps:
            continue
        corridor = max(comps, key=len)
        seed_colors = tuple(int(c) for c in rng.choice(SEED_COLORS, size=2, replace=False))
        need_frontier = bool(rng.random() < 0.75)
        seeded = _choose_seed_pair(grid, corridor, seed_colors, rng, need_frontier)
        if seeded is None:
            continue

        out = main(seeded)
        if need_unreachable and int(np.count_nonzero((seeded == BLACK) & (out == BLACK))) == 0:
            continue
        return seeded
    raise RuntimeError("generate_input: failed to build a valid corridor watershed instance")
