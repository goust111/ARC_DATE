from __future__ import annotations

import random
from collections import deque

import numpy as np


SIDE = 7
HEIGHT = SIDE * 2
BG = 0
TOP_FG = 1
BOTTOM_FG = 3
OUT_FG = 2
CENTER = SIDE // 2
RAY_LEN = 3
MAX_ATTEMPTS = 400

VALID_SEEDS = ((2, 2), (2, 4), (4, 2), (4, 4))
ALL_CELLS = tuple((r, c) for r in range(SIDE) for c in range(SIDE))


def _neighbors4(r: int, c: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < SIDE:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < SIDE:
        yield r, c + 1


def _connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        for nr, nc in _neighbors4(r, c):
            if mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                queue.append((nr, nc))
    return len(seen) == int(mask.sum())


def _path_cells(seed: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    r, c = seed
    dr = -1 if r < CENTER else 1
    dc = -1 if c < CENTER else 1
    cells = []
    for step in range(RAY_LEN):
        rr = r + step * dr
        cc = c + step * dc
        cells.append((rr, cc))
    return tuple(cells)


def _seed_output(seeds: set[tuple[int, int]]) -> np.ndarray:
    out = np.zeros((SIDE, SIDE), dtype=int)
    for seed in seeds:
        for r, c in _path_cells(seed):
            out[r, c] = OUT_FG
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress the shared black cells of two stacked masks into diagonal red rays.

    Input structure:
    - The input is a 14x7 grid made of two vertically stacked 7x7 masks.
    - The top mask uses black background and blue foreground; the bottom mask uses black background and green foreground.
    - Seed cells are the positions that are black in both masks, and valid seeds stay off the central row and central column.

    Rule:
    - Compare the two 7x7 halves cell by cell and collect every position that is black in both.
    - Create a 7x7 black output canvas.
    - For each seed, choose the diagonal pointing away from the canvas center according to the seed's quadrant.
    - Paint three red cells starting at the seed and continuing two more steps along that diagonal.

    Key insight:
    - The red rays are determined by the intersection of the two background masks, not by either half alone, and each seed's global position relative to the center chooses its direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape == (HEIGHT, SIDE), f"expected a {HEIGHT}x{SIDE} grid"

    top = g[:SIDE, :]
    bottom = g[SIDE:, :]
    assert set(map(int, np.unique(top))).issubset({BG, TOP_FG}), "top half must use only black and blue"
    assert set(map(int, np.unique(bottom))).issubset({BG, BOTTOM_FG}), "bottom half must use only black and green"

    out = np.zeros((SIDE, SIDE), dtype=int)
    seed_count = 0
    for r in range(SIDE):
        for c in range(SIDE):
            if top[r, c] != BG or bottom[r, c] != BG:
                continue
            assert r != CENTER and c != CENTER, "common black seed cannot lie on the center row or center column"
            for rr, cc in _path_cells((r, c)):
                assert 0 <= rr < SIDE and 0 <= cc < SIDE, "seed ray must fit entirely inside the output canvas"
                out[rr, cc] = OUT_FG
            seed_count += 1

    assert seed_count >= 2, "expected at least two shared-background seeds"
    return out


def _choose_private_holes(
    common: set[tuple[int, int]],
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    top_only: set[tuple[int, int]] = set()
    bottom_only: set[tuple[int, int]] = set()

    remaining_valid = [cell for cell in VALID_SEEDS if cell not in common]
    random.shuffle(remaining_valid)
    for cell in remaining_valid:
        if random.random() < 0.5:
            top_only.add(cell)
        else:
            bottom_only.add(cell)

    pool = [cell for cell in ALL_CELLS if cell not in common and cell not in top_only and cell not in bottom_only]
    random.shuffle(pool)

    def fill_missing(target: set[tuple[int, int]]) -> None:
        if target:
            return
        while pool:
            target.add(pool.pop())
            return
        raise RuntimeError("hole pool exhausted")

    fill_missing(top_only)
    fill_missing(bottom_only)

    for target in (top_only, bottom_only):
        extra = random.randint(0, 2)
        for _ in range(extra):
            if not pool:
                break
            target.add(pool.pop())

    return top_only, bottom_only


def _build_half(color: int, holes: set[tuple[int, int]]) -> np.ndarray:
    half = np.full((SIDE, SIDE), color, dtype=int)
    for r, c in holes:
        half[r, c] = BG
    return half


def _validate_instance(grid: np.ndarray, common: set[tuple[int, int]]) -> bool:
    if grid.shape != (HEIGHT, SIDE):
        return False

    top = grid[:SIDE, :]
    bottom = grid[SIDE:, :]
    if set(map(int, np.unique(top))) - {BG, TOP_FG}:
        return False
    if set(map(int, np.unique(bottom))) - {BG, BOTTOM_FG}:
        return False

    top_holes = {tuple(map(int, pos)) for pos in np.argwhere(top == BG)}
    bottom_holes = {tuple(map(int, pos)) for pos in np.argwhere(bottom == BG)}
    if top_holes & bottom_holes != common:
        return False
    if len(common) < 2:
        return False
    if any(cell[0] == CENTER or cell[1] == CENTER for cell in common):
        return False

    for seed in common:
        path = _path_cells(seed)
        if any(not (0 <= r < SIDE and 0 <= c < SIDE) for r, c in path):
            return False

    path_sets = [set(_path_cells(seed)) for seed in sorted(common)]
    for i in range(len(path_sets)):
        for j in range(i + 1, len(path_sets)):
            if path_sets[i] & path_sets[j]:
                return False

    if not _connected(top != BG) or not _connected(bottom != BG):
        return False

    top_only = top_holes - common
    bottom_only = bottom_holes - common
    if not top_only or not bottom_only:
        return False
    if top_holes == common or bottom_holes == common:
        return False
    if top_holes == bottom_holes:
        return False

    expected = _seed_output(common)
    try:
        actual = main(grid)
    except AssertionError:
        return False
    return np.array_equal(actual, expected)


def _gen_attempt() -> np.ndarray | None:
    seed_count = random.choice((2, 2, 3, 3, 4))
    common = set(random.sample(VALID_SEEDS, seed_count))
    top_only, bottom_only = _choose_private_holes(common)

    top_holes = common | top_only
    bottom_holes = common | bottom_only
    top = _build_half(TOP_FG, top_holes)
    bottom = _build_half(BOTTOM_FG, bottom_holes)
    grid = np.vstack([top, bottom])

    if _validate_instance(grid, common):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
