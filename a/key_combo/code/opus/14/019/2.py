import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
YELLOW = 4

MAX_GENERATION_ATTEMPTS = 600
GRID_RANGE = (10, 13)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _rotate_point(pt: tuple[int, int], turns: int) -> tuple[int, int]:
    r, c = pt
    for _ in range(turns % 4):
        r, c = -c, r
    return r, c


def _transform_point(pt: tuple[int, int], turns: int, mirror: bool) -> tuple[int, int]:
    r, c = pt
    if mirror:
        c = -c
    return _rotate_point((r, c), turns)


def _transform_cells(
    cells: list[tuple[int, int]],
    turns: int,
    mirror: bool,
    shift: tuple[int, int],
) -> list[tuple[int, int]]:
    sr, sc = shift
    return [
        (r + sr, c + sc)
        for r, c in (_transform_point(pt, turns, mirror) for pt in cells)
    ]


def _count_black_regions(grid: np.ndarray) -> int:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    count = 0
    for r in range(h):
        for c in range(w):
            if grid[r, c] != BLACK or seen[r, c]:
                continue
            count += 1
            q = deque([(r, c)])
            seen[r, c] = True
            while q:
                cr, cc = q.popleft()
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and grid[nr, nc] == BLACK:
                        seen[nr, nc] = True
                        q.append((nr, nc))
    return count


def _apply_halos(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    h, w = out.shape
    seeds = np.argwhere(g == RED)
    for r, c in seeds:
        r = int(r)
        c = int(c)
        for rr in range(max(0, r - 1), min(h, r + 2)):
            for cc in range(max(0, c - 1), min(w, c + 2)):
                if out[rr, cc] == BLACK:
                    out[rr, cc] = BLUE
        out[r, c] = BLUE
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand red seeds into blue halos, count the black regions they and the yellow barriers carve out, and encode that count on a diagonal.

    Input structure:
    - A black background contains 3-5 isolated red seed pixels and 1-3 yellow horizontal or vertical barrier segments.
    - Any red seeds are separated enough that their 3x3 Chebyshev halos do not overlap, though some halos may touch after expansion.
    - Yellow barrier cells stay fixed and block some background corridors while the rest of the grid remains black.

    Rule:
    - Replace each red seed by blue and also paint every black cell in its clipped 3x3 Chebyshev-neighborhood blue.
    - Do not overwrite yellow barrier cells while expanding the halos.
    - Count the number K of remaining 4-connected black background regions.
    - Return a 5x5 black grid whose main diagonal contains K blue cells from the top-left, capped at 5.

    Key insight:
    - The answer is not a count of foreground objects: the red seeds first become larger blue walls, and only the resulting background partition count is encoded in the output.
    """
    final_grid = _apply_halos(grid)
    k = min(5, _count_black_regions(final_grid))
    out = np.zeros((5, 5), dtype=int)
    for i in range(k):
        out[i, i] = BLUE
    return out


def _base_pattern(target_k: int) -> tuple[list[tuple[int, int]], list[list[tuple[int, int]]]]:
    reds = [(-3, 0), (3, 0), (0, -3)]
    if target_k == 1:
        barriers = [[(-1, 2), (0, 2)]]
    elif target_k == 2:
        barriers = [[(-1, 2), (0, 2), (1, 2)]]
    elif target_k == 3:
        barriers = [
            [(-1, 2), (0, 2), (1, 2)],
            [(-1, 0), (0, 0), (1, 0)],
        ]
    elif target_k == 4:
        barriers = [
            [(-1, 2), (0, 2), (1, 2)],
            [(-1, 0), (0, 0), (1, 0)],
            [(0, 0), (0, 1)],
        ]
    elif target_k == 5:
        barriers = [
            [(-1, 2), (0, 2), (1, 2)],
            [(-1, 0), (0, 0), (1, 0)],
            [(0, -1), (0, 0), (0, 1)],
        ]
    else:
        raise ValueError(f"unsupported target_k: {target_k}")
    return reds, barriers


def _seed_halo_cells(seed: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = seed
    return [(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)]


def _all_relative_cells(reds: list[tuple[int, int]], barriers: list[list[tuple[int, int]]]) -> list[tuple[int, int]]:
    cells = []
    for seed in reds:
        cells.extend(_seed_halo_cells(seed))
        cells.append(seed)
    for group in barriers:
        cells.extend(group)
    return cells


def _choose_shift(
    h: int,
    w: int,
    rel_cells: list[tuple[int, int]],
    turns: int,
    mirror: bool,
) -> tuple[int, int]:
    transformed = [_transform_point(pt, turns, mirror) for pt in rel_cells]
    min_r = min(r for r, _ in transformed)
    max_r = max(r for r, _ in transformed)
    min_c = min(c for _, c in transformed)
    max_c = max(c for _, c in transformed)
    sr = random.randint(-min_r, h - 1 - max_r)
    sc = random.randint(-min_c, w - 1 - max_c)
    return sr, sc


def _paint_layout(
    h: int,
    w: int,
    reds: list[tuple[int, int]],
    barrier_groups: list[list[tuple[int, int]]],
) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for r, c in reds:
        grid[r, c] = RED
    for group in barrier_groups:
        for r, c in group:
            grid[r, c] = YELLOW
    return grid


def _all_reds(grid: np.ndarray) -> list[tuple[int, int]]:
    return [tuple(map(int, pt)) for pt in np.argwhere(grid == RED)]


def _all_yellows(grid: np.ndarray) -> list[tuple[int, int]]:
    return [tuple(map(int, pt)) for pt in np.argwhere(grid == YELLOW)]


def _is_red_yellow_clear(reds: list[tuple[int, int]], yellows: list[tuple[int, int]]) -> bool:
    yellow_set = set(yellows)
    for r, c in reds:
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in yellow_set:
                return False
    return True


def _pairwise_red_spacing_ok(reds: list[tuple[int, int]]) -> bool:
    for i, a in enumerate(reds):
        for b in reds[i + 1 :]:
            if _chebyshev(a, b) < 3:
                return False
    return True


def _place_extra_seed(
    grid: np.ndarray,
    reds: list[tuple[int, int]],
    yellow_set: set[tuple[int, int]],
    forbidden: set[tuple[int, int]],
) -> tuple[int, int] | None:
    h, w = grid.shape
    for _ in range(150):
        r = random.randint(1, h - 2)
        c = random.randint(1, w - 2)
        pt = (r, c)
        if pt in forbidden or pt in yellow_set:
            continue
        if any(_chebyshev(pt, other) < 3 for other in reds):
            continue
        if any(nei in yellow_set for nei in _neighbors4(r, c)):
            continue
        return pt
    return None


def _place_border_barrier(
    grid: np.ndarray,
    reds: list[tuple[int, int]],
    yellow_set: set[tuple[int, int]],
    forbidden: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    h, w = grid.shape
    for _ in range(200):
        length = random.randint(2, 4)
        side = random.choice(["top", "bottom", "left", "right"])
        if side == "top":
            c = random.randint(0, w - length)
            cells = [(0, c + dc) for dc in range(length)]
        elif side == "bottom":
            c = random.randint(0, w - length)
            cells = [(h - 1, c + dc) for dc in range(length)]
        elif side == "left":
            r = random.randint(0, h - length)
            cells = [(r + dr, 0) for dr in range(length)]
        else:
            r = random.randint(0, h - length)
            cells = [(r + dr, w - 1) for dr in range(length)]
        if any(cell in forbidden or cell in yellow_set for cell in cells):
            continue
        bad = False
        for cell in cells:
            if any(cell == red for red in reds):
                bad = True
                break
            if any(cell == nei for red in reds for nei in _neighbors4(*red)):
                bad = True
                break
        if bad:
            continue
        return cells
    return None


def _region_count_without_yellow(grid: np.ndarray) -> int:
    stripped = np.where(grid == YELLOW, BLACK, grid)
    return _count_black_regions(_apply_halos(stripped))


def _region_count_without_seed(grid: np.ndarray, seed: tuple[int, int]) -> int:
    stripped = grid.copy()
    stripped[seed] = BLACK
    return _count_black_regions(_apply_halos(stripped))


def _semantic_checks(grid: np.ndarray, target_k: int, base_reds: list[tuple[int, int]]) -> bool:
    reds = _all_reds(grid)
    yellows = _all_yellows(grid)
    if not (3 <= len(reds) <= 5):
        return False
    if not (1 <= len(yellows) <= 12):
        return False
    if not _pairwise_red_spacing_ok(reds):
        return False
    if not _is_red_yellow_clear(reds, yellows):
        return False
    if _count_black_regions(grid) != 1:
        return False

    final_grid = _apply_halos(grid)
    if _count_black_regions(final_grid) != target_k:
        return False

    if target_k > 1 and _region_count_without_yellow(grid) >= target_k:
        return False

    if target_k > 1:
        for seed in base_reds:
            if _region_count_without_seed(grid, seed) >= target_k:
                return False

    return True


def _gen_attempt() -> np.ndarray | None:
    target_k = random.randint(1, 5)
    h = random.randint(*GRID_RANGE)
    w = random.randint(*GRID_RANGE)
    turns = random.randint(0, 3)
    mirror = random.choice([False, True])

    rel_reds, rel_barriers = _base_pattern(target_k)
    rel_cells = _all_relative_cells(rel_reds, rel_barriers)
    shift = _choose_shift(h, w, rel_cells, turns, mirror)

    base_reds = _transform_cells(rel_reds, turns, mirror, shift)
    barrier_groups = [
        _transform_cells(group, turns, mirror, shift)
        for group in rel_barriers
    ]

    grid = _paint_layout(h, w, base_reds, barrier_groups)
    reds = list(base_reds)
    yellow_set = {cell for group in barrier_groups for cell in group}

    occupied = set(reds) | yellow_set
    min_r = min(r for r, _ in occupied)
    max_r = max(r for r, _ in occupied)
    min_c = min(c for _, c in occupied)
    max_c = max(c for _, c in occupied)
    forbidden = {
        (r, c)
        for r in range(max(0, min_r - 2), min(h, max_r + 3))
        for c in range(max(0, min_c - 2), min(w, max_c + 3))
    }

    extra_seed_goal = random.randint(0, 2)
    while len(reds) < min(5, len(base_reds) + extra_seed_goal):
        pt = _place_extra_seed(grid, reds, yellow_set, forbidden)
        if pt is None:
            break
        reds.append(pt)
        grid[pt] = RED
        occupied.add(pt)

    current_groups = list(barrier_groups)
    extra_barrier_budget = 3 - len(current_groups)
    extra_barrier_goal = random.randint(0, extra_barrier_budget)
    while extra_barrier_goal > 0:
        group = _place_border_barrier(grid, reds, yellow_set, forbidden)
        if group is None:
            break
        current_groups.append(group)
        for cell in group:
            yellow_set.add(cell)
            occupied.add(cell)
            grid[cell] = YELLOW
        extra_barrier_goal -= 1

    if not _semantic_checks(grid, target_k, base_reds):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
