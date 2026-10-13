from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
ORANGE = 7
TEAL = 8
INF = 10**9

_GEN_CALL_COUNT = 0
_GEN_MAX_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bfs_distances(walkable: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    dist = np.full(walkable.shape, INF, dtype=int)
    if not walkable[start]:
        return dist
    q = deque([start])
    dist[start] = 0
    while q:
        r, c = q.popleft()
        nd = dist[r, c] + 1
        for nr, nc in _neighbors4(r, c):
            if _in_bounds(walkable.shape, nr, nc) and walkable[nr, nc] and dist[nr, nc] == INF:
                dist[nr, nc] = nd
                q.append((nr, nc))
    return dist


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    seen = np.zeros(mask.shape, dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        q = deque([(rr, cc)])
        seen[rr, cc] = True
        part: list[tuple[int, int]] = []
        while q:
            cr, cc = q.popleft()
            part.append((cr, cc))
            for nr, nc in _neighbors4(cr, cc):
                if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        parts.append(part)
    return parts


def _ownership_sign(d_orange: int, d_blue: int) -> int:
    if d_orange < d_blue:
        return -1
    if d_blue < d_orange:
        return 1
    return 0


def main(grid: np.ndarray) -> np.ndarray:
    """Partition corridor cells by shortest-path distance to the two seeds and mark ties red.

    Input structure:
    - Teal cells are maze walls and black cells are corridors.
    - Exactly one orange seed and one blue seed sit on corridor cells.
    - Extra black corridor pockets may be disconnected from both seeds.

    Rule:
    - Treat every non-teal cell as walkable and compute 4-neighbor shortest-path distances from each seed.
    - Recolor each black corridor cell orange when it is closer to the orange seed, blue when it is closer to the blue seed, and red when the two distances are equal.
    - Leave teal walls fixed, keep the two seeds unchanged, and keep corridor cells black only when neither seed can reach them.

    Key insight:
    - Ownership is determined by maze-path distance through corridors, not by straight-line proximity in the image plane.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    orange_cells = np.argwhere(g == ORANGE)
    blue_cells = np.argwhere(g == BLUE)
    assert orange_cells.shape == (1, 2)
    assert blue_cells.shape == (1, 2)

    orange = tuple(map(int, orange_cells[0]))
    blue = tuple(map(int, blue_cells[0]))
    walkable = g != TEAL

    d_orange = _bfs_distances(walkable, orange)
    d_blue = _bfs_distances(walkable, blue)

    corridor = g == BLACK
    reach_orange = d_orange < INF
    reach_blue = d_blue < INF

    out = g.copy()
    orange_mask = corridor & reach_orange & (~reach_blue | (d_orange < d_blue))
    blue_mask = corridor & reach_blue & (~reach_orange | (d_blue < d_orange))
    red_mask = corridor & reach_orange & reach_blue & (d_orange == d_blue)

    out[orange_mask] = ORANGE
    out[blue_mask] = BLUE
    out[red_mask] = RED
    return out


def _carve_maze(h: int, w: int, protected: np.ndarray | None) -> np.ndarray | None:
    grid = np.full((h, w), TEAL, dtype=int)
    coarse_h = (h - 1) // 2
    coarse_w = (w - 1) // 2

    allowed: list[tuple[int, int]] = []
    for rr in range(coarse_h):
        for cc in range(coarse_w):
            ar = 2 * rr + 1
            ac = 2 * cc + 1
            if protected is None or not protected[ar, ac]:
                allowed.append((rr, cc))
    if not allowed:
        return None

    start = random.choice(allowed)
    stack = [start]
    visited = np.zeros((coarse_h, coarse_w), dtype=bool)
    visited[start] = True
    grid[2 * start[0] + 1, 2 * start[1] + 1] = BLACK

    while stack:
        rr, cc = stack[-1]
        options = []
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr = rr + dr
            nc = cc + dc
            if not (0 <= nr < coarse_h and 0 <= nc < coarse_w) or visited[nr, nc]:
                continue
            wall_r = 2 * rr + 1 + dr
            wall_c = 2 * cc + 1 + dc
            cell_r = 2 * nr + 1
            cell_c = 2 * nc + 1
            if protected is not None and (protected[wall_r, wall_c] or protected[cell_r, cell_c]):
                continue
            options.append((nr, nc, wall_r, wall_c, cell_r, cell_c))
        if not options:
            stack.pop()
            continue

        nr, nc, wall_r, wall_c, cell_r, cell_c = random.choice(options)
        visited[nr, nc] = True
        grid[wall_r, wall_c] = BLACK
        grid[cell_r, cell_c] = BLACK
        stack.append((nr, nc))

    for rr, cc in allowed:
        if not visited[rr, cc]:
            return None
    return grid


def _make_room_mask(h: int, w: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    protected = np.zeros((h, w), dtype=bool)
    pocket_h = random.randint(1, 3)
    pocket_w = random.randint(1, 3)
    room_h = pocket_h + 2
    room_w = pocket_w + 2

    top = random.randint(1, h - room_h - 1)
    left = random.randint(1, w - room_w - 1)

    protected[top : top + room_h, left : left + room_w] = True
    pocket = (top + 1, left + 1, top + room_h - 2, left + room_w - 2)
    return protected, pocket


def _branch_distance_maps(mask: np.ndarray, mid: tuple[int, int]) -> list[dict[int, list[tuple[int, int]]]]:
    seen = np.zeros(mask.shape, dtype=bool)
    seen[mid] = True
    branches: list[dict[int, list[tuple[int, int]]]] = []

    for nr, nc in _neighbors4(*mid):
        if not _in_bounds(mask.shape, nr, nc) or not mask[nr, nc] or seen[nr, nc]:
            continue
        q = deque([((nr, nc), 1)])
        seen[nr, nc] = True
        by_distance: dict[int, list[tuple[int, int]]] = {}
        while q:
            (cr, cc), dist = q.popleft()
            by_distance.setdefault(dist, []).append((cr, cc))
            for ar, ac in _neighbors4(cr, cc):
                if _in_bounds(mask.shape, ar, ac) and mask[ar, ac] and not seen[ar, ac]:
                    seen[ar, ac] = True
                    q.append(((ar, ac), dist + 1))
        branches.append(by_distance)
    return branches


def _seed_candidates(main_mask: np.ndarray, require_branchy_red: bool) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    mids = [tuple(map(int, cell)) for cell in np.argwhere(main_mask)]
    random.shuffle(mids)
    candidates: list[tuple[tuple[int, int], tuple[int, int]]] = []

    for mid in mids:
        degree = 0
        for nr, nc in _neighbors4(*mid):
            if _in_bounds(main_mask.shape, nr, nc) and main_mask[nr, nc]:
                degree += 1
        if degree < 2 or (require_branchy_red and degree < 3):
            continue

        branches = _branch_distance_maps(main_mask, mid)
        if len(branches) < 2 or (require_branchy_red and len(branches) < 3):
            continue

        for i in range(len(branches)):
            for j in range(i + 1, len(branches)):
                common = [d for d in set(branches[i]) & set(branches[j]) if d >= 2]
                random.shuffle(common)
                for d in common[:4]:
                    a_cells = branches[i][d][:]
                    b_cells = branches[j][d][:]
                    random.shuffle(a_cells)
                    random.shuffle(b_cells)
                    for a in a_cells[:2]:
                        for b in b_cells[:2]:
                            candidates.append((a, b))
        if len(candidates) >= 120:
            break

    random.shuffle(candidates)
    return candidates


def _misleading_count(
    corridor: np.ndarray,
    orange: tuple[int, int],
    blue: tuple[int, int],
    d_orange: np.ndarray,
    d_blue: np.ndarray,
) -> int:
    count = 0
    for r, c in np.argwhere(corridor & (d_orange < INF) & (d_blue < INF)):
        rr = int(r)
        cc = int(c)
        path_owner = _ownership_sign(int(d_orange[rr, cc]), int(d_blue[rr, cc]))
        geom_owner = _ownership_sign(
            abs(rr - orange[0]) + abs(cc - orange[1]),
            abs(rr - blue[0]) + abs(cc - blue[1]),
        )
        if path_owner != geom_owner:
            count += 1
    return count


def _evaluate_candidate(
    base_grid: np.ndarray,
    orange: tuple[int, int],
    blue: tuple[int, int],
    require_pocket: bool,
    require_branchy_red: bool,
) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    inp = base_grid.copy()
    inp[orange] = ORANGE
    inp[blue] = BLUE
    out = main(inp)

    walkable = inp != TEAL
    d_orange = _bfs_distances(walkable, orange)
    d_blue = _bfs_distances(walkable, blue)
    corridor = inp == BLACK

    red_count = int(np.count_nonzero(out == RED))
    orange_count = int(np.count_nonzero(out == ORANGE))
    blue_count = int(np.count_nonzero(out == BLUE))
    unreachable_count = int(np.count_nonzero(corridor & (d_orange == INF) & (d_blue == INF)))
    misleading = _misleading_count(corridor, orange, blue, d_orange, d_blue)

    if red_count == 0:
        return None
    if require_branchy_red and red_count < 2:
        return None
    if require_pocket and unreachable_count == 0:
        return None
    if orange_count < 6 or blue_count < 6:
        return None
    if misleading == 0:
        return None
    return inp, (misleading, red_count, min(orange_count, blue_count), unreachable_count)


def _mode_for_call() -> tuple[bool, bool]:
    global _GEN_CALL_COUNT
    idx = _GEN_CALL_COUNT % 3
    _GEN_CALL_COUNT += 1
    if idx == 0:
        return True, False
    if idx == 1:
        return False, True
    return random.random() < 0.5, True


def _generate_base_grid(require_pocket: bool) -> np.ndarray | None:
    h = random.choice(list(range(13, 30, 2)))
    w = random.choice(list(range(13, 30, 2)))

    protected = None
    pocket = None
    if require_pocket:
        protected, pocket = _make_room_mask(h, w)

    grid = _carve_maze(h, w, protected)
    if grid is None:
        return None

    if pocket is not None:
        r0, c0, r1, c1 = pocket
        grid[r0 : r1 + 1, c0 : c1 + 1] = BLACK
    return grid


def _validate_generated(inp: np.ndarray, require_pocket: bool, require_branchy_red: bool) -> bool:
    if inp.ndim != 2 or max(inp.shape) > 30:
        return False
    if np.count_nonzero(inp == ORANGE) != 1 or np.count_nonzero(inp == BLUE) != 1:
        return False
    if not set(map(int, np.unique(inp))).issubset({BLACK, BLUE, ORANGE, TEAL}):
        return False

    out = main(inp)
    if not set(map(int, np.unique(out))).issubset({BLACK, BLUE, RED, ORANGE, TEAL}):
        return False

    orange = tuple(map(int, np.argwhere(inp == ORANGE)[0]))
    blue = tuple(map(int, np.argwhere(inp == BLUE)[0]))
    walkable = inp != TEAL
    d_orange = _bfs_distances(walkable, orange)
    d_blue = _bfs_distances(walkable, blue)
    corridor = inp == BLACK

    unreachable = corridor & (d_orange == INF) & (d_blue == INF)
    red_count = int(np.count_nonzero(out == RED))
    misleading = _misleading_count(corridor, orange, blue, d_orange, d_blue)

    if red_count == 0 or misleading == 0:
        return False
    if require_branchy_red and red_count < 2:
        return False
    if require_pocket and np.count_nonzero(unreachable) == 0:
        return False
    if np.any(out[unreachable] != BLACK):
        return False
    return True


def generate_input() -> np.ndarray:
    require_pocket, require_branchy_red = _mode_for_call()

    for _ in range(_GEN_MAX_ATTEMPTS):
        base_grid = _generate_base_grid(require_pocket)
        if base_grid is None:
            continue

        corridor = base_grid == BLACK
        parts = _components(corridor)
        if not parts:
            continue

        main_part = max(parts, key=len)
        main_mask = np.zeros(base_grid.shape, dtype=bool)
        for r, c in main_part:
            main_mask[r, c] = True

        best: tuple[np.ndarray, tuple[int, int, int, int]] | None = None
        for orange, blue in _seed_candidates(main_mask, require_branchy_red):
            evaluated = _evaluate_candidate(
                base_grid, orange, blue, require_pocket, require_branchy_red
            )
            if evaluated is None:
                continue
            if best is None or evaluated[1] > best[1]:
                best = evaluated

        if best is None:
            continue

        inp = best[0]
        if _validate_generated(inp, require_pocket, require_branchy_red):
            return inp

    raise RuntimeError("failed to generate a valid maze-distance sample")
