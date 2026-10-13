from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
SEED = 1
ORANGE = 7
CYAN = 8

INPUT_SIDE = 5
OUTPUT_SIDE = 10
GEN_ATTEMPTS = 2000


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _diagonals(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c - 1), (r - 1, c + 1), (r + 1, c - 1), (r + 1, c + 1)]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_color_components(grid: np.ndarray, color: int) -> list[np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) != int(color):
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds((h, w), nr, nc) and not seen[nr, nc] and int(g[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(cells, dtype=int))
    return comps


def _drop_orange_components(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    components = _find_color_components(g, ORANGE)
    out[out == ORANGE] = BG

    def order_key(coords: np.ndarray) -> tuple[int, int, int]:
        r0, c0, r1, _c1 = _bbox(coords)
        return (r1, r0, c0)

    for coords in sorted(components, key=order_key, reverse=True):
        dr = 0
        while True:
            shifted = coords + np.array([dr + 1, 0], dtype=int)
            if int(shifted[:, 0].max()) >= out.shape[0]:
                break
            if np.any(out[shifted[:, 0], shifted[:, 1]] != BG):
                break
            dr += 1
        placed = coords + np.array([dr, 0], dtype=int)
        out[placed[:, 0], placed[:, 1]] = ORANGE
    return out


def _mark_seed_diagonals(grid: np.ndarray) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    seeds = np.argwhere(out == SEED)
    for r, c in seeds:
        for nr, nc in _diagonals(int(r), int(c)):
            if _in_bounds(out.shape, nr, nc) and int(out[nr, nc]) == BG:
                out[nr, nc] = CYAN
    return out


def _solve_steps(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tiled = np.tile(np.asarray(grid, dtype=int), (2, 2))
    fallen = _drop_orange_components(tiled)
    marked = _mark_seed_diagonals(fallen)
    return tiled, fallen, marked


def _drop_orange_globally(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    coords = np.argwhere(g == ORANGE)
    out = g.copy()
    out[out == ORANGE] = BG
    if coords.size == 0:
        return out

    dr = 0
    while True:
        shifted = coords + np.array([dr + 1, 0], dtype=int)
        if int(shifted[:, 0].max()) >= out.shape[0]:
            break
        if np.any(out[shifted[:, 0], shifted[:, 1]] != BG):
            break
        dr += 1

    out[coords[:, 0] + dr, coords[:, 1]] = ORANGE
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Tile the scene, let only the orange objects fall, then reveal asymmetry by diagonal cyan marks around fixed blue seeds.

    Input structure:
    - The grid is a small black-background scene containing orange 4-connected object cells and one or more blue single-pixel seeds.
    - Orange cells are the only falling objects; blue seeds stay fixed in place.
    - The same local arrangement is meant to be duplicated into four tiled quadrants before any later step is applied.

    Rule:
    - Tile the input grid into a 2x2 copy.
    - In the tiled grid, find every orange 4-connected component and drop each one straight downward.
    - Process lower orange components first so upper ones stop on the already-settled orange cells; clear every orange component from its original location before repainting it at its fallen location.
    - Finally, for every blue seed, paint each black diagonal neighbor cyan while leaving non-black diagonal neighbors unchanged.

    Key insight:
    - Tiling first creates perfect four-way symmetry, but one-direction gravity breaks the top-bottom symmetry of the orange copies, so the later diagonal seed marks become different across corresponding quadrants.
    """
    _tiled, _fallen, marked = _solve_steps(grid)
    return marked


def _normalize_patch(patch: np.ndarray) -> np.ndarray:
    coords = np.argwhere(patch)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return patch[r0 : r1 + 1, c0 : c1 + 1].copy()


def _random_connected_patch(area: int) -> np.ndarray:
    while True:
        cells = {(0, 0)}
        frontier = set(_neighbors4(0, 0))
        while len(cells) < area:
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            cells.add(cell)
            for nbr in _neighbors4(*cell):
                if nbr not in cells:
                    frontier.add(nbr)

        coords = np.asarray(list(cells), dtype=int)
        coords -= coords.min(axis=0)
        h = int(coords[:, 0].max()) + 1
        w = int(coords[:, 1].max()) + 1
        patch = np.zeros((h, w), dtype=int)
        patch[coords[:, 0], coords[:, 1]] = 1
        patch = np.rot90(patch, random.randrange(4))
        if random.random() < 0.5:
            patch = np.flipud(patch)
        if random.random() < 0.5:
            patch = np.fliplr(patch)
        patch = _normalize_patch(np.asarray(patch, dtype=int))
        h, w = patch.shape
        if h <= 4 and w <= 5 and h >= 2:
            return patch


def _place_patch(patch: np.ndarray) -> set[tuple[int, int]]:
    h, w = patch.shape
    shift_r = random.randint(0, 4 - h)
    shift_c = random.randint(0, 5 - w)
    coords = np.argwhere(patch == 1)
    return {(shift_r + int(r), shift_c + int(c)) for r, c in coords}


def _seed_candidates(shape_cells: set[tuple[int, int]]) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    all_candidates: list[tuple[int, int]] = []
    active: list[tuple[int, int]] = []
    for r in range(1, 4):
        for c in range(1, 4):
            if (r, c) in shape_cells:
                continue
            if any((nr, nc) in shape_cells for nr, nc in _neighbors4(r, c)):
                continue
            all_candidates.append((r, c))
            if any((nr, nc) in shape_cells for nr, nc in _diagonals(r, c)):
                active.append((r, c))
    return all_candidates, active


def _seed_set(candidates: list[tuple[int, int]], active: list[tuple[int, int]]) -> list[tuple[int, int]] | None:
    if not active:
        return None

    seeds = [random.choice(active)]
    want_two = random.random() < 0.5
    if not want_two:
        return seeds

    pool = [
        pos
        for pos in candidates
        if pos not in seeds and max(abs(pos[0] - seeds[0][0]), abs(pos[1] - seeds[0][1])) >= 2
    ]
    if not pool:
        return seeds
    seeds.append(random.choice(pool))
    return seeds


def _build_input(shape_cells: set[tuple[int, int]], seeds: list[tuple[int, int]]) -> np.ndarray:
    grid = np.zeros((INPUT_SIDE, INPUT_SIDE), dtype=int)
    for r, c in shape_cells:
        grid[r, c] = ORANGE
    for r, c in seeds:
        grid[r, c] = SEED
    return grid


def _valid_input_structure(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    if g.shape != (INPUT_SIDE, INPUT_SIDE):
        return False
    if not set(np.unique(g)).issubset({BG, SEED, ORANGE}):
        return False

    orange = np.argwhere(g == ORANGE)
    if orange.shape[0] < 3 or orange.shape[0] > 5:
        return False
    if int(orange[:, 0].max()) >= INPUT_SIDE - 1:
        return False
    if len(_find_color_components(g, ORANGE)) != 1:
        return False

    seeds = np.argwhere(g == SEED)
    if seeds.shape[0] < 1 or seeds.shape[0] > 2:
        return False
    orange_cells = {tuple(map(int, cell)) for cell in orange}
    for r, c in seeds:
        rr = int(r)
        cc = int(c)
        if not (1 <= rr <= 3 and 1 <= cc <= 3):
            return False
        if any((nr, nc) in orange_cells for nr, nc in _neighbors4(rr, cc)):
            return False
    return True


def _has_required_asymmetry(inp: np.ndarray, tiled: np.ndarray, out: np.ndarray) -> bool:
    seeds = np.argwhere(inp == SEED)
    for r, c in seeds:
        r0 = int(r)
        c0 = int(c)
        for dr in (-1, 1):
            for dc in (-1, 1):
                br = r0 + dr
                bc = c0 + dc
                if not _in_bounds(inp.shape, br, bc):
                    continue
                if any(int(tiled[br + row_off, bc + col_off]) != ORANGE for row_off in (0, 5) for col_off in (0, 5)):
                    continue
                for col_off in (0, 5):
                    top = int(out[br, bc + col_off])
                    bottom = int(out[br + 5, bc + col_off])
                    if top == CYAN and bottom == ORANGE:
                        return True
                    if top == ORANGE and bottom == CYAN:
                        return True
    return False


def _semantic_checks(grid: np.ndarray) -> bool:
    if not _valid_input_structure(grid):
        return False

    tiled, fallen, out = _solve_steps(grid)
    if tiled.shape != (OUTPUT_SIDE, OUTPUT_SIDE) or out.shape != (OUTPUT_SIDE, OUTPUT_SIDE):
        return False

    seed_mask = tiled == SEED
    if not np.array_equal(seed_mask, fallen == SEED):
        return False
    if not np.array_equal(seed_mask, out == SEED):
        return False

    if not np.any(out == CYAN):
        return False
    if np.array_equal(fallen, out):
        return False

    no_gravity = _mark_seed_diagonals(tiled)
    if np.array_equal(out, no_gravity):
        return False

    global_drop = _mark_seed_diagonals(_drop_orange_globally(tiled))
    if np.array_equal(out, global_drop):
        return False

    if np.array_equal(out[:5, :], out[5:, :]):
        return False

    if not _has_required_asymmetry(grid, tiled, out):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    patch = _random_connected_patch(random.randint(3, 5))
    shape_cells = _place_patch(patch)
    candidates, active = _seed_candidates(shape_cells)
    seeds = _seed_set(candidates, active)
    if seeds is None:
        return None
    grid = _build_input(shape_cells, seeds)
    if not _semantic_checks(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
