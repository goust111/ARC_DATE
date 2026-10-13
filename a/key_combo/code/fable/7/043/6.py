import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
YELLOW = 4

CELL_SIZE = 3
CELL_COUNT = 3
GRID_SIZE = CELL_COUNT * CELL_SIZE + (CELL_COUNT - 1)
SEPARATOR_POSITIONS = (CELL_SIZE, CELL_SIZE * 2 + 1)
MAX_GENERATION_ATTEMPTS = 300
ALL_CELL_COORDS = tuple((r, c) for r in range(CELL_SIZE) for c in range(CELL_SIZE))


def _segments(length: int, separators: list[int]) -> list[tuple[int, int]]:
    separators = sorted(int(x) for x in separators)
    starts = [0] + [s + 1 for s in separators]
    ends = separators + [length]
    return list(zip(starts, ends))


def _lattice_spec(grid: np.ndarray) -> tuple[int, int, list[tuple[int, int]], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    full_rows: dict[int, list[int]] = {}
    for r in range(g.shape[0]):
        if np.all(g[r, :] == g[r, 0]):
            full_rows.setdefault(int(g[r, 0]), []).append(int(r))

    full_cols: dict[int, list[int]] = {}
    for c in range(g.shape[1]):
        if np.all(g[:, c] == g[0, c]):
            full_cols.setdefault(int(g[0, c]), []).append(int(c))

    separator_colors = [
        color
        for color in set(full_rows) & set(full_cols)
        if len(full_rows[color]) == CELL_COUNT - 1 and len(full_cols[color]) == CELL_COUNT - 1
    ]
    assert len(separator_colors) == 1, "expected one separator color"
    separator = int(separator_colors[0])
    row_segments = _segments(g.shape[0], full_rows[separator])
    col_segments = _segments(g.shape[1], full_cols[separator])

    assert len(row_segments) == CELL_COUNT and len(col_segments) == CELL_COUNT
    assert all(r1 - r0 == CELL_SIZE for r0, r1 in row_segments)
    assert all(c1 - c0 == CELL_SIZE for c0, c1 in col_segments)

    content_colors = set(map(int, np.unique(g))) - {separator}
    assert BLUE in content_colors, "expected blue cell patterns"
    bg_candidates = content_colors - {BLUE}
    assert len(bg_candidates) == 1, "expected a single background color"
    bg = int(next(iter(bg_candidates)))
    return separator, bg, row_segments, col_segments


def _cell_mask(
    grid: np.ndarray,
    row_segments: list[tuple[int, int]],
    col_segments: list[tuple[int, int]],
    bg: int,
    ri: int,
    ci: int,
) -> np.ndarray:
    r0, r1 = row_segments[ri]
    c0, c1 = col_segments[ci]
    return np.asarray(grid[r0:r1, c0:c1] != bg, dtype=bool)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the unique lattice cell whose blue pattern is the whole grid's 3x3 occupancy map.

    Input structure:
    - The grid is divided by full separator lines into a 3x3 lattice of 3x3 cells.
    - Cell interiors contain only background or blue pixels, and some cells are completely empty.

    Rule:
    - Mark each cell occupied iff its 3x3 interior contains any non-background pixel, producing a 3x3 occupancy map.
    - Find the unique cell whose own 3x3 foreground pattern exactly matches that occupancy map.
    - Recolor every blue pixel in that cell red and leave everything else unchanged.

    Key insight:
    - One cell is self-referential: its local bitmap is a miniature map of which of the nine cells are occupied globally.
    """
    g = np.asarray(grid, dtype=int)
    separator, bg, row_segments, col_segments = _lattice_spec(g)
    assert separator == YELLOW

    occupancy = np.zeros((CELL_COUNT, CELL_COUNT), dtype=bool)
    for ri in range(CELL_COUNT):
        for ci in range(CELL_COUNT):
            occupancy[ri, ci] = bool(_cell_mask(g, row_segments, col_segments, bg, ri, ci).any())

    candidates: list[tuple[int, int]] = []
    for ri in range(CELL_COUNT):
        for ci in range(CELL_COUNT):
            if np.array_equal(_cell_mask(g, row_segments, col_segments, bg, ri, ci), occupancy):
                candidates.append((ri, ci))

    assert len(candidates) == 1, "expected exactly one self-describing cell"
    target_r, target_c = candidates[0]
    r0, r1 = row_segments[target_r]
    c0, c1 = col_segments[target_c]

    out = g.copy()
    cell = out[r0:r1, c0:c1]
    cell[cell == BLUE] = RED
    return out


def _mask_to_key(mask: np.ndarray) -> tuple[int, ...]:
    return tuple(int(x) for x in mask.reshape(-1))


def _indices_to_mask(indices: list[int] | tuple[int, ...] | set[int]) -> np.ndarray:
    mask = np.zeros((CELL_SIZE, CELL_SIZE), dtype=bool)
    for idx in indices:
        mask[int(idx) // CELL_SIZE, int(idx) % CELL_SIZE] = True
    return mask


def _sample_occupancy() -> tuple[tuple[int, int], np.ndarray]:
    target_idx = random.randrange(CELL_SIZE * CELL_SIZE)
    occupied_count = random.randint(3, 7)
    other_indices = [idx for idx in range(CELL_SIZE * CELL_SIZE) if idx != target_idx]
    occupied = {target_idx, *random.sample(other_indices, occupied_count - 1)}
    return (target_idx // CELL_SIZE, target_idx % CELL_SIZE), _indices_to_mask(occupied)


def _near_miss(mask: np.ndarray, used: set[tuple[int, ...]]) -> np.ndarray | None:
    candidates = []
    for r, c in ALL_CELL_COORDS:
        alt = mask.copy()
        alt[r, c] = ~alt[r, c]
        if alt.any() and _mask_to_key(alt) not in used:
            candidates.append(alt)
    if not candidates:
        return None
    return random.choice(candidates)


def _random_pattern(used: set[tuple[int, ...]]) -> np.ndarray:
    while True:
        count = random.randint(1, 8)
        indices = random.sample(range(CELL_SIZE * CELL_SIZE), count)
        mask = _indices_to_mask(indices)
        key = _mask_to_key(mask)
        if key not in used:
            return mask


def _paint_cell(grid: np.ndarray, ri: int, ci: int, mask: np.ndarray, color: int) -> None:
    r0 = ri * (CELL_SIZE + 1)
    c0 = ci * (CELL_SIZE + 1)
    grid[r0 : r0 + CELL_SIZE, c0 : c0 + CELL_SIZE][mask] = color


def _build_grid(target_cell: tuple[int, int], occupancy: np.ndarray) -> np.ndarray:
    grid = np.full((GRID_SIZE, GRID_SIZE), BLACK, dtype=int)
    for idx in SEPARATOR_POSITIONS:
        grid[idx, :] = YELLOW
        grid[:, idx] = YELLOW

    used = {_mask_to_key(occupancy)}
    occupied_cells = [(ri, ci) for ri in range(CELL_COUNT) for ci in range(CELL_COUNT) if occupancy[ri, ci]]
    random.shuffle(occupied_cells)

    near_miss_done = False
    for ri, ci in occupied_cells:
        if (ri, ci) == target_cell:
            pattern = occupancy
        else:
            pattern = None
            if not near_miss_done and random.random() < 0.75:
                pattern = _near_miss(occupancy, used)
                near_miss_done = pattern is not None
            if pattern is None:
                pattern = _random_pattern(used)
            used.add(_mask_to_key(pattern))
        _paint_cell(grid, ri, ci, pattern, BLUE)

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        target_cell, occupancy = _sample_occupancy()
        grid = _build_grid(target_cell, occupancy)
        out = main(grid)

        target_r, target_c = target_cell
        r0 = target_r * (CELL_SIZE + 1)
        c0 = target_c * (CELL_SIZE + 1)
        target_in = grid[r0 : r0 + CELL_SIZE, c0 : c0 + CELL_SIZE]
        target_out = out[r0 : r0 + CELL_SIZE, c0 : c0 + CELL_SIZE]

        if not np.any(target_in == BLUE):
            continue
        if not np.array_equal(target_out == RED, target_in == BLUE):
            continue
        if np.count_nonzero(out == RED) != np.count_nonzero(target_in == BLUE):
            continue
        return grid

    raise RuntimeError("failed to generate a valid self-referential occupancy puzzle")
