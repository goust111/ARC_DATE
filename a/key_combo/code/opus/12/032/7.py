from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5

GRID_SIZE_CHOICES = (11, 11, 13, 13, 15)
OBSTACLE_COUNT_RANGE = (8, 15)
GREEN_COUNT_CHOICES = (0, 0, 1, 2, 2, 3, 4)
MAX_GENERATION_ATTEMPTS = 400


def _find_unique_marker(grid: np.ndarray) -> tuple[int, int]:
    coords = np.argwhere(grid == RED)
    assert coords.shape == (1, 2), "expected exactly one red marker"
    return int(coords[0, 0]), int(coords[0, 1])


def _same_parity_mask(shape: tuple[int, int], parity: tuple[int, int]) -> np.ndarray:
    rr, cc = np.indices(shape)
    return (rr % 2 == parity[0]) & (cc % 2 == parity[1])


def _step2_neighbors(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 2, c), (r + 2, c), (r, c - 2), (r, c + 2))


def _fill_on_grid(grid: np.ndarray) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    marker = _find_unique_marker(out)
    parity = (marker[0] % 2, marker[1] % 2)
    h, w = out.shape

    seen = {marker}
    queue: deque[tuple[int, int]] = deque([marker])
    while queue:
        r, c = queue.popleft()
        for nr, nc in _step2_neighbors(r, c):
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if (nr, nc) in seen:
                continue
            if nr % 2 != parity[0] or nc % 2 != parity[1]:
                continue
            if int(out[nr, nc]) != BLACK:
                continue
            seen.add((nr, nc))
            out[nr, nc] = RED
            queue.append((nr, nc))
    return out


def _paint_global_lattice(grid: np.ndarray, parity: tuple[int, int]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    mask = _same_parity_mask(out.shape, parity)
    out[(out == BLACK) & mask] = RED
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transpose the square grid, then flood-fill the marker's new parity lattice until blockers stop it.

    Input structure:
    - The grid is an odd-sized square with a full blue main diagonal on a black background.
    - Exactly one red marker sits off the diagonal; grey obstacle pixels and optional green decoration pixels occupy other off-diagonal cells.
    - The marker's row and column have opposite parity, so its coordinates change parity role after transposition.

    Rule:
    - First transpose the whole grid across the main diagonal, swapping every off-diagonal cell with its mirror position.
    - On the transposed grid, read the red marker's new coordinates and take their row/column parity as the target lattice.
    - Starting from that red marker, spread only through black cells on the same lattice using 4-neighbor moves of length 2; stop at any non-black cell, including grey obstacles, the blue diagonal, and green decorations.
    - Recolor every reached black lattice cell red and leave all other transposed cells unchanged.

    Key insight:
    - The transpose does not just move shapes; it changes the marker's coordinate parity, and that new parity determines which lattice is fillable in the transposed obstacle layout.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h == w, "expected a square grid"
    transposed = g.T.copy()
    return _fill_on_grid(transposed)


def _pick_barrier_index(indices: list[int]) -> int:
    for margin in (2, 1):
        candidates = indices[margin : len(indices) - margin]
        if candidates:
            return int(random.choice(candidates))
    raise RuntimeError("not enough parity lines to place a barrier")


def _barrier_cells(n: int, parity: tuple[int, int], orientation: str, line_index: int) -> set[tuple[int, int]]:
    pr, pc = parity
    if orientation == "row":
        return {(line_index, c) for c in range(n) if c % 2 == pc}
    return {(r, line_index) for r in range(n) if r % 2 == pr}


def _marker_candidates(
    n: int,
    parity: tuple[int, int],
    orientation: str,
    barrier_index: int,
    side: str,
) -> list[tuple[int, int]]:
    pr, pc = parity
    rows = [r for r in range(n) if r % 2 == pr]
    cols = [c for c in range(n) if c % 2 == pc]
    if orientation == "row":
        rows = [r for r in rows if (r < barrier_index if side == "low" else r > barrier_index)]
    else:
        cols = [c for c in cols if (c < barrier_index if side == "low" else c > barrier_index)]
    return [(r, c) for r in rows for c in cols]


def _obstacle_component_sizes(grid: np.ndarray) -> list[int]:
    h, w = grid.shape
    seen: set[tuple[int, int]] = set()
    sizes: list[int] = []
    for r, c in map(tuple, np.argwhere(grid == GREY)):
        if (r, c) in seen:
            continue
        queue = deque([(r, c)])
        seen.add((r, c))
        size = 0
        while queue:
            cr, cc = queue.popleft()
            size += 1
            for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) == GREY and (nr, nc) not in seen:
                    seen.add((nr, nc))
                    queue.append((nr, nc))
        sizes.append(size)
    return sizes


def _try_place_extra_obstacles(
    post: np.ndarray,
    blocked: set[tuple[int, int]],
    avoid_parity: tuple[int, int],
    target_count: int,
) -> bool:
    n = int(post.shape[0])
    attempts = 0
    while int(np.count_nonzero(post == GREY)) < target_count and attempts < 600:
        attempts += 1
        remaining = target_count - int(np.count_nonzero(post == GREY))
        make_domino = remaining >= 2 and random.random() < 0.45

        cells: list[tuple[int, int]] = []
        if make_domino:
            r = random.randrange(n)
            c = random.randrange(n)
            dr, dc = random.choice(((1, 0), (-1, 0), (0, 1), (0, -1)))
            cells = [(r, c), (r + dr, c + dc)]
        else:
            r = random.randrange(n)
            c = random.randrange(n)
            cells = [(r, c)]

        if any(not (0 <= r < n and 0 <= c < n) for r, c in cells):
            continue
        if any(r == c for r, c in cells):
            continue
        if any((r, c) in blocked for r, c in cells):
            continue
        if any(int(post[r, c]) != BLACK for r, c in cells):
            continue
        if any((r % 2, c % 2) == avoid_parity for r, c in cells):
            continue

        for r, c in cells:
            post[r, c] = GREY
        if any(size > 2 for size in _obstacle_component_sizes(post)):
            for r, c in cells:
                post[r, c] = BLACK
            continue
        blocked.update(cells)
    return int(np.count_nonzero(post == GREY)) == target_count


def _place_greens(post: np.ndarray, blocked: set[tuple[int, int]]) -> None:
    count = random.choice(GREEN_COUNT_CHOICES)
    if count == 0:
        return

    candidates = [
        (r, c)
        for r in range(post.shape[0])
        for c in range(post.shape[1])
        if int(post[r, c]) == BLACK and r != c and r % 2 == c % 2 and (r, c) not in blocked
    ]
    if not candidates:
        return

    random.shuffle(candidates)
    for r, c in candidates[:count]:
        post[r, c] = GREEN
        blocked.add((r, c))


def _build_post_grid() -> np.ndarray:
    n = random.choice(GRID_SIZE_CHOICES)
    post = np.full((n, n), BLACK, dtype=int)
    np.fill_diagonal(post, BLUE)

    parity = random.choice(((0, 1), (1, 0)))
    orientation = random.choice(("row", "col"))
    side = random.choice(("low", "high"))

    parity_lines = [i for i in range(n) if i % 2 == (parity[0] if orientation == "row" else parity[1])]
    barrier_index = _pick_barrier_index(parity_lines)
    barrier = _barrier_cells(n, parity, orientation, barrier_index)

    marker_options = _marker_candidates(n, parity, orientation, barrier_index, side)
    marker = random.choice(marker_options)
    post[marker] = RED

    blocked = {marker}
    for r, c in barrier:
        post[r, c] = GREY
        blocked.add((r, c))

    target_obstacles = random.randint(max(OBSTACLE_COUNT_RANGE[0], len(barrier)), OBSTACLE_COUNT_RANGE[1])
    if not _try_place_extra_obstacles(post, blocked, parity, target_obstacles):
        raise RuntimeError("failed to place extra obstacles")

    _place_greens(post, blocked)
    return post


def _validate_structure(inp: np.ndarray, out: np.ndarray) -> bool:
    if inp.ndim != 2 or out.ndim != 2:
        return False
    h, w = inp.shape
    if h != w or h not in {11, 13, 15}:
        return False
    if out.shape != inp.shape:
        return False

    diag = np.diag(inp)
    if not np.all(diag == BLUE):
        return False
    if np.argwhere(inp == RED).shape != (1, 2):
        return False
    mr, mc = map(int, np.argwhere(inp == RED)[0])
    if mr == mc or mr % 2 == mc % 2:
        return False
    grey_coords = np.argwhere(inp == GREY)
    if not (OBSTACLE_COUNT_RANGE[0] <= grey_coords.shape[0] <= OBSTACLE_COUNT_RANGE[1]):
        return False
    if any(int(r) == int(c) for r, c in grey_coords):
        return False
    if any(size > 2 for size in _obstacle_component_sizes(inp)):
        return False
    return True


def _validate_semantics(inp: np.ndarray) -> bool:
    post = inp.T.copy()
    correct = _fill_on_grid(post)
    actual = main(inp)
    if not np.array_equal(correct, actual):
        return False

    input_marker = _find_unique_marker(inp)
    wrong_parity = (input_marker[0] % 2, input_marker[1] % 2)
    wrong_lattice = _paint_global_lattice(post, wrong_parity)
    if np.array_equal(correct, wrong_lattice):
        return False
    if int(np.count_nonzero(correct != wrong_lattice)) < 6:
        return False

    cleared = post.copy()
    cleared[cleared == GREY] = BLACK
    no_obstacle = _fill_on_grid(cleared)
    if np.array_equal(correct, no_obstacle):
        return False
    if int(np.count_nonzero(no_obstacle == RED)) - int(np.count_nonzero(correct == RED)) < 4:
        return False

    marker = _find_unique_marker(post)
    parity = (marker[0] % 2, marker[1] % 2)
    parity_mask = _same_parity_mask(post.shape, parity)
    unrestricted = _paint_global_lattice(post, parity)
    if np.array_equal(correct, unrestricted):
        return False
    if not np.any((post == BLACK) & parity_mask & (correct == BLACK)):
        return False
    if int(np.count_nonzero(correct == RED)) < 8:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        try:
            post = _build_post_grid()
        except RuntimeError:
            continue
        inp = post.T.copy()
        out = main(inp)
        if _validate_structure(inp, out) and _validate_semantics(inp):
            return inp
    raise RuntimeError("failed to generate a valid transpose-parity-fill puzzle")
