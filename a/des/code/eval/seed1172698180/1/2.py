from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
WALL = 5
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_RETRIES = 300


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _traversable_degree(mask: np.ndarray, cell: tuple[int, int]) -> int:
    r, c = cell
    return sum(
        1
        for dr, dc in DIR4
        if _in_bounds(mask.shape, r + dr, c + dc) and bool(mask[r + dr, c + dc])
    )


def _distance_map(mask: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    dist = np.full(mask.shape, -1, dtype=int)
    q = deque([start])
    dist[start] = 0
    while q:
        r, c = q.popleft()
        for dr, dc in DIR4:
            nr = r + dr
            nc = c + dc
            if _in_bounds(mask.shape, nr, nc) and bool(mask[nr, nc]) and int(dist[nr, nc]) < 0:
                dist[nr, nc] = int(dist[r, c]) + 1
                q.append((nr, nc))
    return dist


def _simulate_growth(grid: np.ndarray, with_stats: bool = False):
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    h, w = g.shape

    free = np.zeros((h, w), dtype=np.int8)
    free[g == WALL] = 1

    frontiers: dict[int, set[tuple[int, int]]] = {}
    new_counts: dict[int, int] = {}
    for r, c in np.argwhere((g != BLACK) & (g != WALL)):
        color = int(g[r, c])
        frontiers.setdefault(color, set()).add((int(r), int(c)))
        new_counts.setdefault(color, 0)
        free[r, c] = 2

    collision_count = 0
    steps = 0

    while frontiers:
        attempts: dict[tuple[int, int], set[int]] = {}
        for color, cells in frontiers.items():
            for r, c in cells:
                for dr, dc in DIR4:
                    nr = r + dr
                    nc = c + dc
                    if _in_bounds(g.shape, nr, nc) and int(free[nr, nc]) == 0:
                        attempts.setdefault((nr, nc), set()).add(int(color))

        if not attempts:
            break

        next_frontiers: dict[int, set[tuple[int, int]]] = {}
        for (r, c), colors in attempts.items():
            if len(colors) == 1:
                color = next(iter(colors))
                free[r, c] = 2
                out[r, c] = int(color)
                next_frontiers.setdefault(int(color), set()).add((int(r), int(c)))
                new_counts[int(color)] = int(new_counts.get(int(color), 0)) + 1
            else:
                free[r, c] = 3
                collision_count += 1

        frontiers = next_frontiers
        steps += 1

    if with_stats:
        return out, {
            "collision_count": int(collision_count),
            "new_total": int(sum(new_counts.values())),
            "new_counts": {int(k): int(v) for k, v in new_counts.items()},
            "steps": int(steps),
        }
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Simulate synchronous multi-source vine growth inside one-cell-wide corridors.

    - Grey pixels are walls.
    - Colored singleton seeds expand simultaneously through black corridor cells.
    - A cell reached by exactly one color becomes that color.
    - A cell reached by multiple colors in the same step stays black forever.
    """
    return _simulate_growth(grid, with_stats=False)


def _build_base_network(rng: random.Random) -> np.ndarray | None:
    rows = rng.randint(4, 7)
    cols = rng.randint(4, 8)

    if rows >= 3 and cols >= 3:
        hub = (rng.randint(1, rows - 2), rng.randint(1, cols - 2))
    else:
        hub = (rng.randrange(rows), rng.randrange(cols))

    adj: dict[tuple[int, int], set[tuple[int, int]]] = {hub: set()}
    directions = list(DIR4)
    rng.shuffle(directions)

    spoke_neighbors = []
    for dr, dc in directions:
        nr = hub[0] + dr
        nc = hub[1] + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            spoke_neighbors.append((nr, nc))
    if len(spoke_neighbors) < 3:
        return None

    for nb in spoke_neighbors[: rng.choice((3, 3, 4))]:
        adj.setdefault(nb, set()).add(hub)
        adj[hub].add(nb)

    target_nodes = rng.randint(
        max(8, rows + cols - 1),
        max(9, min(rows * cols, (rows * cols) // 2 + 6)),
    )

    while len(adj) < target_nodes:
        frontier: list[tuple[tuple[int, int], tuple[int, int]]] = []
        weights: list[int] = []
        for src, neighbors in adj.items():
            for dr, dc in DIR4:
                dst = (src[0] + dr, src[1] + dc)
                if not (0 <= dst[0] < rows and 0 <= dst[1] < cols) or dst in adj:
                    continue
                frontier.append((src, dst))
                weights.append(1 + min(len(neighbors), 3) + (3 if src == hub else 0))
        if not frontier:
            break

        src, dst = rng.choices(frontier, weights=weights, k=1)[0]
        adj.setdefault(dst, set()).add(src)
        adj[src].add(dst)

        dr = dst[0] - src[0]
        dc = dst[1] - src[1]
        cur = dst
        for _ in range(rng.randint(0, 2)):
            nxt = (cur[0] + dr, cur[1] + dc)
            if not (0 <= nxt[0] < rows and 0 <= nxt[1] < cols) or nxt in adj or len(adj) >= target_nodes:
                break
            adj.setdefault(nxt, set()).add(cur)
            adj[cur].add(nxt)
            cur = nxt

    if max(len(neighbors) for neighbors in adj.values()) < 3:
        return None

    possible_cycles = []
    for a in adj:
        for dr, dc in DIR4:
            b = (a[0] + dr, a[1] + dc)
            if b in adj and b not in adj[a] and a < b:
                possible_cycles.append((a, b))
    rng.shuffle(possible_cycles)
    for a, b in possible_cycles[: rng.randint(0, min(3, len(possible_cycles)))]:
        if rng.random() < 0.35:
            adj[a].add(b)
            adj[b].add(a)

    h = 2 * rows + 1
    w = 2 * cols + 1
    grid = np.full((h, w), WALL, dtype=int)

    for (r, c), neighbors in adj.items():
        rr = 2 * r + 1
        cc = 2 * c + 1
        grid[rr, cc] = BLACK
        for nr, nc in neighbors:
            rr2 = 2 * nr + 1
            cc2 = 2 * nc + 1
            grid[(rr + rr2) // 2, (cc + cc2) // 2] = BLACK

    corridor = grid == BLACK
    if not _has_intersection(corridor):
        return None
    if int(np.count_nonzero(corridor)) < 12:
        return None
    return grid


def _has_intersection(corridor: np.ndarray) -> bool:
    return any(
        _traversable_degree(corridor, (int(r), int(c))) >= 3
        for r, c in np.argwhere(corridor)
    )


def _seed_weight(corridor: np.ndarray, cell: tuple[int, int]) -> int:
    degree = _traversable_degree(corridor, cell)
    if degree == 1:
        return 5
    if degree == 2:
        return 3
    return 1


def _sample_collision_seeds(
    base: np.ndarray,
    rng: random.Random,
    groups: list[list[tuple[int, int]]],
    dist_cache: dict[tuple[int, int], np.ndarray],
) -> np.ndarray | None:
    corridor = base == BLACK
    for _ in range(200):
        group = list(rng.choice(groups))
        possible_counts = [n for n in (2, 3, 4) if len(group) >= n]
        if not possible_counts:
            continue
        weights = [4 if n == 2 else 2 if n == 3 else 1 for n in possible_counts]
        seed_count = rng.choices(possible_counts, weights=weights, k=1)[0]
        chosen = [rng.choices(group, weights=[_seed_weight(corridor, c) for c in group], k=1)[0]]

        while len(chosen) < seed_count:
            options = [
                cell
                for cell in group
                if cell not in chosen and all(int(dist_cache[src][cell]) >= 5 for src in chosen)
            ]
            if not options:
                break
            option_weights = [
                sum(int(dist_cache[src][cell]) for src in chosen) + _seed_weight(corridor, cell)
                for cell in options
            ]
            chosen.append(rng.choices(options, weights=option_weights, k=1)[0])

        if len(chosen) != seed_count:
            continue

        grid = base.copy()
        for color, cell in zip(rng.sample(SEED_COLORS, seed_count), chosen):
            grid[cell] = int(color)

        _, stats = _simulate_growth(grid, with_stats=True)
        if int(stats["collision_count"]) <= 0:
            continue
        if int(stats["new_total"]) < 8:
            continue
        if min(int(v) for v in stats["new_counts"].values()) <= 0:
            continue
        return grid
    return None


def _sample_noncollision_seeds(
    base: np.ndarray,
    rng: random.Random,
    even_cells: list[tuple[int, int]],
    odd_cells: list[tuple[int, int]],
    dist_cache: dict[tuple[int, int], np.ndarray],
) -> np.ndarray | None:
    corridor = base == BLACK
    for _ in range(200):
        first = rng.choices(even_cells, weights=[_seed_weight(corridor, c) for c in even_cells], k=1)[0]
        options = [cell for cell in odd_cells if int(dist_cache[first][cell]) >= 5]
        if not options:
            continue
        second = rng.choices(
            options,
            weights=[int(dist_cache[first][cell]) + _seed_weight(corridor, cell) for cell in options],
            k=1,
        )[0]

        grid = base.copy()
        colors = rng.sample(SEED_COLORS, 2)
        grid[first] = int(colors[0])
        grid[second] = int(colors[1])

        _, stats = _simulate_growth(grid, with_stats=True)
        if int(stats["collision_count"]) != 0:
            continue
        if int(stats["new_total"]) < 8:
            continue
        if min(int(v) for v in stats["new_counts"].values()) <= 0:
            continue
        return grid
    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    base = _build_base_network(rng)
    if base is None:
        return None

    corridor = base == BLACK
    cells = [tuple(map(int, p)) for p in np.argwhere(corridor)]
    even_cells = [cell for cell in cells if (cell[0] + cell[1]) % 2 == 0]
    odd_cells = [cell for cell in cells if (cell[0] + cell[1]) % 2 == 1]
    if len(even_cells) < 2 or len(odd_cells) < 1:
        return None

    dist_cache = {cell: _distance_map(corridor, cell) for cell in cells}
    want_collision = rng.random() < 0.5

    if want_collision:
        parity_groups = []
        if len(even_cells) >= 2:
            parity_groups.append(even_cells)
        if len(odd_cells) >= 2:
            parity_groups.append(odd_cells)
        if not parity_groups:
            return None
        seeded = _sample_collision_seeds(base, rng, parity_groups, dist_cache)
    else:
        if not odd_cells:
            return None
        seeded = _sample_noncollision_seeds(base, rng, even_cells, odd_cells, dist_cache)

    if seeded is None:
        return None
    if not _has_intersection(seeded != WALL):
        return None
    return seeded


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid vine-maze input")
