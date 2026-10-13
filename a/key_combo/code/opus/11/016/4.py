from __future__ import annotations

from collections import Counter
import random

import numpy as np


BACKGROUND = 0
FOREGROUND = 1
SHADOW = 2
QUAD_SIZE = 5
GRID_SIZE = 2 * QUAD_SIZE + 1
SLOTS = ((0, 0), (0, QUAD_SIZE + 1), (QUAD_SIZE + 1, 0), (QUAD_SIZE + 1, QUAD_SIZE + 1))
MAX_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _is_connected(cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return False
    stack = [next(iter(cells))]
    seen = set()
    while stack:
        cell = stack.pop()
        if cell in seen:
            continue
        seen.add(cell)
        for nbr in _neighbors4(*cell):
            if nbr in cells and nbr not in seen:
                stack.append(nbr)
    return len(seen) == len(cells)


def _render(cells: set[tuple[int, int]]) -> np.ndarray:
    grid = np.zeros((QUAD_SIZE, QUAD_SIZE), dtype=int)
    for r, c in cells:
        grid[r, c] = FOREGROUND
    return grid


def _has_visible_wrap(cells: set[tuple[int, int]]) -> bool:
    return any(c == 0 and r > 0 for r, c in cells)


def _has_top_row(cells: set[tuple[int, int]]) -> bool:
    return any(r == 0 for r, _ in cells)


def _has_shadow_overlap(cells: set[tuple[int, int]]) -> bool:
    for r, c in cells:
        if r > 0 and (r - 1, (c - 1) % QUAD_SIZE) in cells:
            return True
    return False


def _grow_connected(seed: set[tuple[int, int]], target_size: int) -> set[tuple[int, int]] | None:
    cells = set(seed)
    while len(cells) < target_size:
        frontier = set()
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < QUAD_SIZE and 0 <= nc < QUAD_SIZE and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        if not frontier:
            return None
        cells.add(random.choice(sorted(frontier)))
    return cells


def _generate_pattern(
    size: int,
    *,
    need_top: bool,
    need_overlap: bool,
    require_visible_wrap: bool = True,
    avoid: set[tuple[int, int]] | None = None,
) -> set[tuple[int, int]] | None:
    for _ in range(MAX_ATTEMPTS):
        if need_overlap:
            seed = {(0, 0), (1, 0), (1, 1)}
        elif need_top:
            seed = {(0, 1), (1, 1), (1, 0)}
        else:
            seed = {(random.randint(1, QUAD_SIZE - 1), 0)}

        if len(seed) > size:
            continue
        cells = _grow_connected(seed, size)
        if cells is None:
            continue
        if require_visible_wrap and not _has_visible_wrap(cells):
            continue
        if need_top and not _has_top_row(cells):
            continue
        if need_overlap and not _has_shadow_overlap(cells):
            continue
        if avoid is not None and cells == avoid:
            continue
        return cells
    return None


def _translate_variant(cells: set[tuple[int, int]]) -> set[tuple[int, int]] | None:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)

    shifts = []
    for dr in range(-min_r, QUAD_SIZE - 1 - max_r + 1):
        for dc in range(-min_c, QUAD_SIZE - 1 - max_c + 1):
            if dr == 0 and dc == 0:
                continue
            shifted = {(r + dr, c + dc) for r, c in cells}
            if shifted != cells:
                shifts.append(shifted)
    if not shifts:
        return None
    random.shuffle(shifts)
    return shifts[0]


def _size_variant(cells: set[tuple[int, int]]) -> set[tuple[int, int]] | None:
    variants: list[set[tuple[int, int]]] = []

    if len(cells) > 3:
        removable = [cell for cell in cells if _is_connected(cells - {cell})]
        for cell in removable:
            variants.append(cells - {cell})

    if len(cells) < 7:
        frontier = set()
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < QUAD_SIZE and 0 <= nc < QUAD_SIZE and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        for cell in frontier:
            variants.append(cells | {cell})

    random.shuffle(variants)
    for variant in variants:
        if variant != cells and _is_connected(variant):
            return variant
    return None


def _reshape_variant(cells: set[tuple[int, int]]) -> set[tuple[int, int]] | None:
    need_top = random.random() < 0.25
    for _ in range(MAX_ATTEMPTS):
        variant = _generate_pattern(
            len(cells),
            need_top=need_top,
            need_overlap=False,
            require_visible_wrap=random.random() < 0.5,
            avoid=cells,
        )
        if variant is not None and variant != cells:
            return variant
    return None


def _pattern_key(block: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in row) for row in block.tolist())


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract the odd quadrant pattern and add an up-left shadow with column wrap.

    Input structure:
    - The input is an 11x11 grid with a black separator cross: the middle row and
      middle column are all 0, splitting the canvas into four 5x5 quadrants.
    - Each quadrant contains a blue-on-black 5x5 pattern.
    - Exactly three quadrants are identical and one quadrant is the unique odd one out.

    Rule:
    - Compare the four 5x5 quadrants as complete arrays and select the unique one.
    - Copy that 5x5 quadrant to the output canvas.
    - For every blue pixel at (r, c), place a red shadow at (r-1, (c-1) mod 5).
    - Drop shadows whose target row would be above the canvas, and keep blue pixels
      on top whenever a shadow and foreground would land on the same cell.

    Key insight:
    - The output depends only on the anomalous quadrant, and its left-edge pixels
      create right-edge shadows because only columns wrap while rows do not.
    """
    g = np.asarray(grid, dtype=int)
    if g.shape != (GRID_SIZE, GRID_SIZE):
        raise ValueError(f"expected {GRID_SIZE}x{GRID_SIZE} input, got {g.shape}")

    mid = QUAD_SIZE
    quadrants = [
        g[:mid, :mid],
        g[:mid, mid + 1 :],
        g[mid + 1 :, :mid],
        g[mid + 1 :, mid + 1 :],
    ]
    keys = [_pattern_key(block) for block in quadrants]
    counts = Counter(keys)
    unique_indices = [idx for idx, key in enumerate(keys) if counts[key] == 1]
    if len(unique_indices) != 1:
        raise ValueError("expected exactly one unique quadrant")

    work = quadrants[unique_indices[0]].copy()
    out = np.zeros_like(work)
    for r, c in np.argwhere(work == FOREGROUND):
        rr = int(r) - 1
        if rr >= 0:
            out[rr, (int(c) - 1) % QUAD_SIZE] = SHADOW
    out[work == FOREGROUND] = FOREGROUND
    return out


def _make_majority_and_unique() -> tuple[np.ndarray, np.ndarray] | None:
    relation = random.choice(["translate", "size", "reshape"])
    need_overlap = random.random() < 0.45
    need_top = need_overlap or (random.random() < 0.5)
    unique_size = random.randint(3, 7)

    unique_cells = _generate_pattern(unique_size, need_top=need_top, need_overlap=need_overlap)
    if unique_cells is None:
        return None

    if relation == "translate":
        majority_cells = _translate_variant(unique_cells)
    elif relation == "size":
        majority_cells = _size_variant(unique_cells)
    else:
        majority_cells = _reshape_variant(unique_cells)

    if majority_cells is None or majority_cells == unique_cells:
        return None
    if not _is_connected(majority_cells):
        return None
    if not 3 <= len(majority_cells) <= 7:
        return None

    majority = _render(majority_cells)
    unique = _render(unique_cells)
    if np.array_equal(majority, unique):
        return None
    return majority, unique


def _build_input(majority: np.ndarray, unique: np.ndarray, unique_slot: int) -> np.ndarray:
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    for slot_idx, (r0, c0) in enumerate(SLOTS):
        block = unique if slot_idx == unique_slot else majority
        grid[r0 : r0 + QUAD_SIZE, c0 : c0 + QUAD_SIZE] = block
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        pair = _make_majority_and_unique()
        if pair is None:
            continue
        majority, unique = pair
        unique_slot = random.randrange(4)
        grid = _build_input(majority, unique, unique_slot)

        if np.count_nonzero(unique == FOREGROUND) < 3 or np.count_nonzero(unique == FOREGROUND) > 7:
            continue
        unique_cells = set(map(tuple, np.argwhere(unique == FOREGROUND)))
        if not _has_visible_wrap(unique_cells):
            continue
        expected = main(grid)
        if expected.shape != (QUAD_SIZE, QUAD_SIZE):
            continue
        return grid
    raise RuntimeError("failed to generate a valid input")
