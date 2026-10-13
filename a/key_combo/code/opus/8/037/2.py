from __future__ import annotations

from collections import deque
from itertools import combinations

import numpy as np


BLACK = 0
BLUE = 1
GREY = 5
PINK = 6

GRID_SIZE = 19
CELL_SIZE = 3
STRIDE = 4
CELL_STARTS = tuple(range(0, GRID_SIZE, STRIDE))
SEPARATORS = tuple(start + CELL_SIZE for start in CELL_STARTS[:-1])
INSTRUCTION_CELLS = frozenset({(0, 0), (0, 1), (1, 0), (1, 1)})
ALL_CONTENT_CELLS = tuple(
    (r, c)
    for r in range(len(CELL_STARTS))
    for c in range(len(CELL_STARTS))
    if (r, c) not in INSTRUCTION_CELLS
)

DIR_TO_BLOCK = {
    "up": (0, 0),
    "right": (0, 1),
    "left": (1, 0),
    "down": (1, 1),
}
DIR_TO_DELTA = {
    "up": (-1, 0),
    "right": (0, 1),
    "left": (0, -1),
    "down": (1, 0),
}
MASK_OPTIONS = (
    frozenset({"up", "right"}),
    frozenset({"up", "left"}),
    frozenset({"right", "down"}),
    frozenset({"left", "down"}),
    frozenset({"up", "down"}),
    frozenset({"left", "right"}),
    frozenset({"up", "right", "left"}),
    frozenset({"up", "right", "down"}),
    frozenset({"up", "left", "down"}),
    frozenset({"right", "left", "down"}),
)


def _cell_origin(cell_r: int, cell_c: int) -> tuple[int, int]:
    return CELL_STARTS[cell_r], CELL_STARTS[cell_c]


def _cell_view(grid: np.ndarray, cell_r: int, cell_c: int) -> np.ndarray:
    r0, c0 = _cell_origin(cell_r, cell_c)
    return grid[r0 : r0 + CELL_SIZE, c0 : c0 + CELL_SIZE]


def _base_grid() -> np.ndarray:
    grid = np.full((GRID_SIZE, GRID_SIZE), BLACK, dtype=int)
    for sep in SEPARATORS:
        grid[sep, :] = GREY
        grid[:, sep] = GREY
    return grid


def _decode_allowed_dirs(grid: np.ndarray) -> frozenset[str]:
    allowed = set()
    for direction, (cell_r, cell_c) in DIR_TO_BLOCK.items():
        block = _cell_view(grid, cell_r, cell_c)
        colors = set(map(int, np.unique(block)))
        assert colors <= {BLACK, PINK}, "instruction cells must be all black or all pink"
        assert len(colors) == 1, "instruction cells must be uniform"
        if int(block[1, 1]) == PINK:
            allowed.add(direction)
    assert len(allowed) >= 2, "expected at least two allowed directions"
    return frozenset(allowed)


def _find_seed_cells(grid: np.ndarray) -> tuple[tuple[int, int], ...]:
    seeds: list[tuple[int, int]] = []
    seen_blue = 0
    for cell_r, cell_c in ALL_CONTENT_CELLS:
        block = _cell_view(grid, cell_r, cell_c)
        center_blue = int(block[1, 1]) == BLUE
        if center_blue:
            mask = np.zeros((CELL_SIZE, CELL_SIZE), dtype=bool)
            mask[1, 1] = True
            assert np.all(block[mask] == BLUE), "seed cell center must be blue"
            assert np.all(block[~mask] == BLACK), "seed cell background must stay black"
            seeds.append((cell_r, cell_c))
            seen_blue += 1
        else:
            assert np.all(block == BLACK), "non-seed content cells must be all black"
    assert 1 <= len(seeds) <= 2, "expected one or two seed cells"
    assert seen_blue == int(np.count_nonzero(grid == BLUE)), "blue may only appear as seed centers"
    return tuple(seeds)


def _reachable_cells(
    allowed_dirs: frozenset[str], seeds: tuple[tuple[int, int], ...]
) -> frozenset[tuple[int, int]]:
    reached = set(seeds)
    q: deque[tuple[int, int]] = deque(seeds)
    while q:
        cell_r, cell_c = q.popleft()
        for direction in allowed_dirs:
            dr, dc = DIR_TO_DELTA[direction]
            nr = cell_r + dr
            nc = cell_c + dc
            nxt = (nr, nc)
            if not (0 <= nr < len(CELL_STARTS) and 0 <= nc < len(CELL_STARTS)):
                continue
            if nxt in INSTRUCTION_CELLS or nxt in reached:
                continue
            reached.add(nxt)
            q.append(nxt)
    return frozenset(reached)


def _render_output(grid: np.ndarray, reached: frozenset[tuple[int, int]]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for cell_r, cell_c in reached:
        _cell_view(out, cell_r, cell_c)[:, :] = PINK
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode a 2x2 direction mask and flood-fill lattice cells reachable from the blue seed cells.

    Input structure:
    - A 19x19 black canvas is split by grey separator lines into a 5x5 lattice of 3x3 cells.
    - The top-left 2x2 lattice cells form an instruction region; each of those cells is uniformly black or pink.
    - Outside the instruction region, one or two cells contain a single blue center pixel; all other content cells are black.

    Rule:
    - Read the instruction region as a direction mask with top-left=up, top-right=right, bottom-left=left, bottom-right=down.
    - Starting from every seed cell, move from cell to adjacent cell only in the globally allowed directions, never entering the instruction cells.
    - Recolor every reachable non-instruction cell, including the seed cells themselves, into a solid 3x3 pink block; preserve the instruction region and the grey separators.

    Key insight:
    - The decoded pink/black instruction blocks do not mark pixels inside a cell; they define which inter-cell moves are legal for a flood fill over the lattice.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape == (GRID_SIZE, GRID_SIZE), "expected a 19x19 lattice"
    for sep in SEPARATORS:
        assert np.all(g[sep, :] == GREY), "separator rows must stay grey"
        assert np.all(g[:, sep] == GREY), "separator columns must stay grey"

    allowed_dirs = _decode_allowed_dirs(g)
    seeds = _find_seed_cells(g)
    reached = _reachable_cells(allowed_dirs, seeds)
    return _render_output(g, reached)


def _encode_mask(grid: np.ndarray, allowed_dirs: frozenset[str]) -> None:
    for direction, (cell_r, cell_c) in DIR_TO_BLOCK.items():
        block = _cell_view(grid, cell_r, cell_c)
        block[:, :] = PINK if direction in allowed_dirs else BLACK


def _paint_seed(grid: np.ndarray, cell: tuple[int, int]) -> None:
    cell_r, cell_c = cell
    block = _cell_view(grid, cell_r, cell_c)
    block[:, :] = BLACK
    block[1, 1] = BLUE


def _reachable_for_seed_options(allowed_dirs: frozenset[str]) -> dict[tuple[int, int], frozenset[tuple[int, int]]]:
    return {cell: _reachable_cells(allowed_dirs, (cell,)) for cell in ALL_CONTENT_CELLS}


def _is_rich_mask(allowed_dirs: frozenset[str]) -> bool:
    return allowed_dirs not in {frozenset({"up", "down"}), frozenset({"left", "right"})}


def _seed_option_score(
    allowed_dirs: frozenset[str], seeds: tuple[tuple[int, int], ...], reached: frozenset[tuple[int, int]]
) -> tuple[int, int, int, int]:
    rows = sorted({cell_r for cell_r, _ in reached})
    cols = sorted({cell_c for _, cell_c in reached})
    span = (rows[-1] - rows[0]) + (cols[-1] - cols[0]) if rows and cols else 0
    complexity = 1 if _is_rich_mask(allowed_dirs) else 0
    return (complexity, len(reached), span, -len(seeds))


def _semantic_ok(grid: np.ndarray) -> bool:
    try:
        allowed_dirs = _decode_allowed_dirs(grid)
        seeds = _find_seed_cells(grid)
    except AssertionError:
        return False

    reached = _reachable_cells(allowed_dirs, seeds)
    total_cells = len(ALL_CONTENT_CELLS)
    if len(reached) <= len(seeds):
        return False
    if len(reached) >= total_cells:
        return False

    for direction in allowed_dirs:
        if _reachable_cells(frozenset(set(allowed_dirs) - {direction}), seeds) == reached:
            return False

    if len(seeds) == 2:
        first_only = _reachable_cells(allowed_dirs, (seeds[0],))
        second_only = _reachable_cells(allowed_dirs, (seeds[1],))
        if first_only == reached or second_only == reached:
            return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False
    painted_pixels = 0
    for cell_r, cell_c in ALL_CONTENT_CELLS:
        block = _cell_view(out, cell_r, cell_c)
        if (cell_r, cell_c) in reached:
            if not np.all(block == PINK):
                return False
            painted_pixels += CELL_SIZE * CELL_SIZE
        elif not np.array_equal(block, _cell_view(grid, cell_r, cell_c)):
            return False
    if painted_pixels != len(reached) * (CELL_SIZE * CELL_SIZE):
        return False
    return True


def _candidate_seed_sets(
    allowed_dirs: frozenset[str], rng: np.random.Generator
) -> list[tuple[tuple[tuple[int, int], ...], frozenset[tuple[int, int]]]]:
    single_reach = _reachable_for_seed_options(allowed_dirs)
    options: list[tuple[tuple[tuple[int, int], ...], frozenset[tuple[int, int]]]] = []

    for cell, reached in single_reach.items():
        if len(reached) <= 1 or len(reached) >= len(ALL_CONTENT_CELLS):
            continue
        if any(_reachable_cells(frozenset(set(allowed_dirs) - {direction}), (cell,)) == reached for direction in allowed_dirs):
            continue
        options.append(((cell,), reached))

    for first, second in combinations(ALL_CONTENT_CELLS, 2):
        reached = frozenset(single_reach[first] | single_reach[second])
        if len(reached) <= 2 or len(reached) >= len(ALL_CONTENT_CELLS):
            continue
        if single_reach[first] == reached or single_reach[second] == reached:
            continue
        if any(
            _reachable_cells(frozenset(set(allowed_dirs) - {direction}), (first, second)) == reached
            for direction in allowed_dirs
        ):
            continue
        options.append(((first, second), reached))

    if not options:
        return []

    rng.shuffle(options)
    options.sort(key=lambda item: _seed_option_score(allowed_dirs, item[0], item[1]), reverse=True)
    return options


def _build_input(allowed_dirs: frozenset[str], seeds: tuple[tuple[int, int], ...]) -> np.ndarray:
    grid = _base_grid()
    _encode_mask(grid, allowed_dirs)
    for seed in seeds:
        _paint_seed(grid, seed)
    return grid


def _choose_mask(rng: np.random.Generator) -> frozenset[str]:
    weights = np.array([1 if _is_rich_mask(mask) else 0.45 for mask in MASK_OPTIONS], dtype=float)
    weights /= weights.sum()
    return MASK_OPTIONS[int(rng.choice(len(MASK_OPTIONS), p=weights))]


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    for _ in range(40):
        allowed_dirs = _choose_mask(rng)
        options = _candidate_seed_sets(allowed_dirs, rng)
        if not options:
            continue

        rich = [item for item in options if _seed_option_score(allowed_dirs, item[0], item[1])[1] >= 5]
        pool = rich if rich and rng.random() < 0.8 else options
        cut = min(len(pool), 12)
        seeds, _ = pool[int(rng.integers(0, cut))]
        grid = _build_input(allowed_dirs, seeds)
        if _semantic_ok(grid):
            return grid
    return None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(300):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
