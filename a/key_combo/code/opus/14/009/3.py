from __future__ import annotations

from collections import deque
import random

import numpy as np


BACKGROUND = 0
PIVOT = 5
MERGED = 6
PALETTE = (1, 2, 3, 4, 7, 8, 9)

GRID_MIN = 13
GRID_MAX = 18
SHAPE_COUNT_RANGE = (4, 6)
SHAPE_SIZE_RANGE = (2, 5)
MAX_GENERATION_ATTEMPTS = 500
SIDES = ("left", "right", "top", "bottom")


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds((h, w), nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reflect every foreground shape 180 degrees through the grey pivot and recolor any reflected multi-source component to magenta.

    Input structure:
    - The grid is a black background with exactly one grey pivot pixel.
    - Several small colored foreground shapes occupy ordinary cells; the pivot is not part of any shape.
    - Foreground shapes are treated as 4-connected components of all non-black, non-grey pixels.

    Rule:
    - Find the unique grey pivot at (pr, pc).
    - Reflect every non-black, non-grey pixel (r, c) to (2 * pr - r, 2 * pc - c), clearing the original foreground positions.
    - Keep the grey pivot fixed.
    - In the reflected layout, compute 4-connected components over all non-black, non-grey pixels.
    - If a reflected component contains pixels originating from two or more distinct source components, recolor that whole reflected component to magenta; otherwise keep its reflected colors.

    Key insight:
    - The output is determined by a global half-turn about the pivot, followed by a component-origin test on the reflected layout rather than on the input layout.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    pivots = np.argwhere(g == PIVOT)
    assert len(pivots) == 1, "expected exactly one grey pivot"
    pr, pc = map(int, pivots[0])

    foreground_mask = (g != BACKGROUND) & (g != PIVOT)
    source_components = _components_from_mask(foreground_mask)
    assert source_components, "expected at least one foreground shape"

    out = np.full_like(g, BACKGROUND)
    out[pr, pc] = PIVOT
    owners = np.full(g.shape, -1, dtype=int)

    for owner_id, comp in enumerate(source_components):
        for r_raw, c_raw in comp:
            r = int(r_raw)
            c = int(c_raw)
            rr = 2 * pr - r
            cc = 2 * pc - c
            assert _in_bounds(g.shape, rr, cc), "reflected pixel must stay in bounds"
            assert (rr, cc) != (pr, pc), "foreground reflection cannot overwrite the pivot"
            assert owners[rr, cc] == -1, "distinct source pixels cannot collide after reflection"
            out[rr, cc] = int(g[r, c])
            owners[rr, cc] = owner_id

    reflected_components = _components_from_mask((out != BACKGROUND) & (out != PIVOT))
    for comp in reflected_components:
        source_ids = {int(owners[r, c]) for r, c in comp}
        if len(source_ids) >= 2:
            out[comp[:, 0], comp[:, 1]] = MERGED
    return out


def _fits_side(r: int, c: int, pr: int, pc: int, side: str) -> bool:
    if side == "left":
        return c < pc
    if side == "right":
        return c > pc
    if side == "top":
        return r < pr
    if side == "bottom":
        return r > pr
    raise ValueError(f"unknown side: {side}")


def _candidate_cells(h: int, w: int, pr: int, pc: int, side: str) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for r in range(h):
        for c in range(w):
            if (r, c) == (pr, pc) or not _fits_side(r, c, pr, pc, side):
                continue
            rr = 2 * pr - r
            cc = 2 * pc - c
            if _in_bounds((h, w), rr, cc):
                cells.add((r, c))
    return cells


def _pick_pivot(h: int, w: int, side: str) -> tuple[int, int] | None:
    row_lo = 3
    row_hi = h - 4
    col_lo = 3
    col_hi = w - 4
    if side in ("left", "right"):
        col_lo = 5
        col_hi = w - 6
    else:
        row_lo = 5
        row_hi = h - 6
    if row_lo > row_hi or col_lo > col_hi:
        return None

    for _ in range(60):
        pr = random.randint(row_lo, row_hi)
        pc = random.randint(col_lo, col_hi)
        if (pr, pc) != (h // 2, w // 2):
            return pr, pc
    return None


def _shape_seed_choices(allowed: set[tuple[int, int]], occupied: set[tuple[int, int]]) -> list[tuple[int, int]]:
    seeds: list[tuple[int, int]] = []
    for r, c in allowed:
        if (r, c) in occupied:
            continue
        if any((nr, nc) in occupied for nr, nc in _neighbors4(r, c)):
            continue
        seeds.append((r, c))
    random.shuffle(seeds)
    return seeds


def _grow_shape(
    allowed: set[tuple[int, int]],
    occupied: set[tuple[int, int]],
    target_size: int,
) -> set[tuple[int, int]] | None:
    for seed in _shape_seed_choices(allowed, occupied)[:40]:
        shape = {seed}
        while len(shape) < target_size:
            frontier: set[tuple[int, int]] = set()
            for r, c in shape:
                for nr, nc in _neighbors4(r, c):
                    if (nr, nc) not in allowed or (nr, nc) in shape or (nr, nc) in occupied:
                        continue
                    if any((ar, ac) in occupied for ar, ac in _neighbors4(nr, nc)):
                        continue
                    frontier.add((nr, nc))
            if not frontier:
                break
            shape.add(random.choice(tuple(frontier)))
        if len(shape) == target_size:
            return shape
    return None


def _validate_generated_input(grid: np.ndarray) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    pivots = np.argwhere(grid == PIVOT)
    if len(pivots) != 1:
        return False
    if tuple(map(int, pivots[0])) == (grid.shape[0] // 2, grid.shape[1] // 2):
        return False

    foreground_mask = (grid != BACKGROUND) & (grid != PIVOT)
    components = _components_from_mask(foreground_mask)
    if not (SHAPE_COUNT_RANGE[0] <= len(components) <= SHAPE_COUNT_RANGE[1]):
        return False
    if not all(SHAPE_SIZE_RANGE[0] <= len(comp) <= SHAPE_SIZE_RANGE[1] for comp in components):
        return False

    colors = [int(grid[comp[0, 0], comp[0, 1]]) for comp in components]
    if len(set(colors)) != len(colors):
        return False

    out = main(grid)
    reflected_components = _components_from_mask((out != BACKGROUND) & (out != PIVOT))
    if len(reflected_components) != len(components):
        return False
    if np.any(out == MERGED):
        return False

    overlap = ((grid != BACKGROUND) & (grid != PIVOT) & (out != BACKGROUND) & (out != PIVOT)).any()
    return not bool(overlap)


def _attempt_generate() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    side = random.choice(SIDES)
    pivot = _pick_pivot(h, w, side)
    if pivot is None:
        return None
    pr, pc = pivot

    allowed = _candidate_cells(h, w, pr, pc, side)
    shape_count = random.randint(*SHAPE_COUNT_RANGE)
    sizes = [random.randint(*SHAPE_SIZE_RANGE) for _ in range(shape_count)]
    sizes.sort(reverse=True)
    if len(allowed) < sum(sizes):
        return None

    occupied: set[tuple[int, int]] = set()
    shapes: list[set[tuple[int, int]]] = []
    for size in sizes:
        shape = _grow_shape(allowed, occupied, size)
        if shape is None:
            return None
        shapes.append(shape)
        occupied |= shape

    colors = random.sample(PALETTE, shape_count)
    random.shuffle(colors)

    grid = np.full((h, w), BACKGROUND, dtype=int)
    grid[pr, pc] = PIVOT
    for color, shape in zip(colors, shapes):
        for r, c in shape:
            grid[r, c] = color

    return grid if _validate_generated_input(grid) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid pivot-reflection ARC input")
