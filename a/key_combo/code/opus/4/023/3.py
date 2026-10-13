from __future__ import annotations

import itertools

import numpy as np


BACKGROUND = 0
DIVIDER = 6
CELL_INTERIOR = 3
COLOR_POOL = (1, 2, 3, 4, 5, 7, 8, 9)
COUNT_TO_TURNS = {
    0: 0,
    1: 3,  # 90 degrees clockwise
    2: 2,
    4: 1,  # 90 degrees counterclockwise
}
GENERATION_RETRIES = 500


def _ranges_between(size: int, dividers: list[int]) -> list[tuple[int, int]]:
    points = [-1] + list(dividers) + [int(size)]
    return [
        (points[i] + 1, points[i + 1] - 1)
        for i in range(len(points) - 1)
        if points[i] + 1 <= points[i + 1] - 1
    ]


def _divider_grid(grid: np.ndarray) -> tuple[list[int], list[int]]:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == DIVIDER)]
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == DIVIDER)]
    assert rows and cols, "expected full magenta divider rows and columns"
    assert len(rows) == len(cols), "row/column divider counts must match"
    assert len(rows) in {4, 5}, "expected a 3x3 or 4x4 lattice of cells"
    row_steps = {b - a for a, b in zip(rows, rows[1:])}
    col_steps = {b - a for a, b in zip(cols, cols[1:])}
    assert row_steps == {CELL_INTERIOR + 1}, "divider rows must be evenly spaced"
    assert col_steps == {CELL_INTERIOR + 1}, "divider cols must be evenly spaced"
    return rows, cols


def _cell_ranges(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    divider_rows, divider_cols = _divider_grid(grid)
    row_ranges = _ranges_between(grid.shape[0], divider_rows)
    col_ranges = _ranges_between(grid.shape[1], divider_cols)
    assert len(row_ranges) == len(col_ranges), "cell lattice must be square"
    assert len(row_ranges) in {3, 4}, "expected 3x3 or 4x4 cells"
    assert all(r1 - r0 + 1 == CELL_INTERIOR for r0, r1 in row_ranges), "cell height must be 3"
    assert all(c1 - c0 + 1 == CELL_INTERIOR for c0, c1 in col_ranges), "cell width must be 3"
    return row_ranges, col_ranges


def _corner_count(mask: np.ndarray) -> int:
    return int(mask[0, 0] + mask[0, 2] + mask[2, 0] + mask[2, 2])


def _extract_cell(cell: np.ndarray) -> tuple[int, np.ndarray]:
    vals = sorted(set(map(int, cell.ravel())) - {BACKGROUND})
    assert DIVIDER not in vals, "divider color must not appear inside cells"
    assert len(vals) == 1, "each cell must contain exactly one foreground color"
    color = vals[0]
    mask = cell == color
    assert int(mask.sum()) == 5, "each cell-local sprite must occupy exactly 5 pixels"
    count = _corner_count(mask.astype(int))
    assert count in COUNT_TO_TURNS, "corner occupancy must be 0, 1, 2, or 4"
    return color, mask


def _rotate_cell(cell: np.ndarray) -> np.ndarray:
    color, mask = _extract_cell(cell)
    turns = COUNT_TO_TURNS[_corner_count(mask.astype(int))]
    rotated_mask = np.rot90(mask.astype(int), turns)
    out = np.full_like(cell, BACKGROUND)
    out[rotated_mask.astype(bool)] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each lattice cell by the angle encoded by its own occupied corner count.

    Input structure:
    - Black is the background and magenta (6) forms complete one-pixel horizontal
      and vertical divider lines.
    - The divider lines partition the grid into a 3x3 or 4x4 square lattice of
      equal cells, each with a 3x3 interior.
    - Every cell interior contains exactly one 5-pixel monochrome foreground
      sprite using a non-black, non-magenta color.

    Rule:
    - Split the grid at the magenta divider rows and columns.
    - For each 3x3 cell, count how many of its four corners are occupied by the
      foreground sprite before any change.
    - Map that corner count to a rotation of the same cell-local sprite:
      0 corners -> keep it, 1 corner -> rotate 90 degrees clockwise,
      2 corners -> rotate 180 degrees, 4 corners -> rotate 90 degrees
      counterclockwise.
    - Write the rotated 3x3 result back into the same cell and keep the magenta
      divider lattice unchanged.

    Key insight:
    - The occupied corner pixels are self-referential: they both encode the
      rotation angle and belong to the very sprite that must be rotated.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    row_ranges, col_ranges = _cell_ranges(g)

    out = g.copy()
    for r0, r1 in row_ranges:
        for c0, c1 in col_ranges:
            out[r0 : r1 + 1, c0 : c1 + 1] = _rotate_cell(g[r0 : r1 + 1, c0 : c1 + 1])
    return out


def _build_mask_library() -> dict[int, list[np.ndarray]]:
    by_count: dict[int, list[np.ndarray]] = {0: [], 1: [], 2: [], 4: []}
    for cells in itertools.combinations(range(9), 5):
        mask = np.zeros((3, 3), dtype=int)
        for idx in cells:
            mask[idx // 3, idx % 3] = 1
        count = _corner_count(mask)
        if count not in by_count:
            continue
        if any(np.array_equal(mask, np.rot90(mask, turns)) for turns in (1, 2, 3)):
            continue
        by_count[count].append(mask)
    return by_count


MASK_LIBRARY = _build_mask_library()
REALIZABLE_COUNTS = tuple(count for count in (1, 2, 4) if MASK_LIBRARY[count])


def _choose_counts(cell_count: int, rng: np.random.Generator) -> list[int]:
    if rng.random() < 0.65:
        active = [1, 2, 4]
    else:
        active = sorted(map(int, rng.choice(np.array(REALIZABLE_COUNTS), size=2, replace=False)))

    counts = list(active)
    duplicate = int(rng.choice(active))
    if len(counts) < cell_count:
        counts.append(duplicate)
    while len(counts) < cell_count:
        counts.append(int(rng.choice(active)))
    rng.shuffle(counts)
    assert len(set(counts)) >= 2
    assert any(counts.count(count) >= 2 for count in set(counts))
    return counts


def _assign_masks(counts: list[int], rng: np.random.Generator) -> list[np.ndarray]:
    masks: list[np.ndarray] = [np.zeros((3, 3), dtype=int) for _ in counts]
    for count in sorted(set(counts)):
        idxs = [i for i, value in enumerate(counts) if value == count]
        library = MASK_LIBRARY[count]
        picks = rng.choice(len(library), size=len(idxs), replace=len(idxs) > len(library))
        if len(idxs) >= 2 and len(set(map(int, picks))) == 1 and len(library) > 1:
            picks[1] = (int(picks[0]) + 1 + int(rng.integers(len(library) - 1))) % len(library)
        for idx, pick in zip(idxs, picks):
            masks[idx] = library[int(pick)].copy()
    return masks


def _assign_colors(counts: list[int], rng: np.random.Generator) -> list[int]:
    palette_size = int(rng.integers(3, min(6, len(COLOR_POOL)) + 1))
    palette = list(map(int, rng.choice(np.array(COLOR_POOL), size=palette_size, replace=False)))
    colors = [int(rng.choice(palette)) for _ in counts]

    repeated_counts = [count for count in set(counts) if counts.count(count) >= 2]
    if repeated_counts and len(palette) >= 2:
        count = int(rng.choice(repeated_counts))
        idxs = [i for i, value in enumerate(counts) if value == count][:2]
        pair = list(map(int, rng.choice(np.array(palette), size=2, replace=False)))
        colors[idxs[0]] = pair[0]
        colors[idxs[1]] = pair[1]

    cross_pairs = [(i, j) for i in range(len(counts)) for j in range(i + 1, len(counts)) if counts[i] != counts[j]]
    assert cross_pairs, "need at least two different corner-count classes"
    i, j = cross_pairs[int(rng.integers(len(cross_pairs)))]
    shared = int(rng.choice(palette))
    colors[i] = shared
    colors[j] = shared
    return colors


def _render_grid(size: int, masks: list[np.ndarray], colors: list[int]) -> np.ndarray:
    side = 4 * size + 1
    grid = np.full((side, side), DIVIDER, dtype=int)
    idx = 0
    for cell_r in range(size):
        for cell_c in range(size):
            r0 = 1 + 4 * cell_r
            c0 = 1 + 4 * cell_c
            grid[r0 : r0 + 3, c0 : c0 + 3] = BACKGROUND
            mask = masks[idx].astype(bool)
            grid[r0 : r0 + 3, c0 : c0 + 3][mask] = int(colors[idx])
            idx += 1
    return grid


def _cell_signature(mask: np.ndarray) -> bytes:
    return np.asarray(mask, dtype=np.uint8).tobytes()


def _validate_generated(grid: np.ndarray) -> bool:
    try:
        row_ranges, col_ranges = _cell_ranges(grid)
        counts: list[int] = []
        colors: list[int] = []
        mask_sigs_by_count: dict[int, set[bytes]] = {}
        local_outputs: list[np.ndarray] = []

        for r0, r1 in row_ranges:
            for c0, c1 in col_ranges:
                cell = grid[r0 : r1 + 1, c0 : c1 + 1]
                color, mask = _extract_cell(cell)
                count = _corner_count(mask.astype(int))
                counts.append(count)
                colors.append(color)
                mask_sigs_by_count.setdefault(count, set()).add(_cell_signature(mask))
                rotated = _rotate_cell(cell)
                local_outputs.append(rotated)
                if COUNT_TO_TURNS[count] != 0 and np.array_equal(rotated, cell):
                    return False

        if len(set(counts)) < 2:
            return False
        if not any(len(sigs) >= 2 for sigs in mask_sigs_by_count.values()):
            return False
        if not any(len({counts[i] for i, color in enumerate(colors) if color == shared}) >= 2 for shared in set(colors)):
            return False
        if not any(len({colors[i] for i, count in enumerate(counts) if count == shared}) >= 2 for shared in set(counts) if counts.count(shared) >= 2):
            return False

        out = main(grid)
        if np.array_equal(out, grid):
            return False
        idx = 0
        for r0, r1 in row_ranges:
            for c0, c1 in col_ranges:
                if not np.array_equal(out[r0 : r1 + 1, c0 : c1 + 1], local_outputs[idx]):
                    return False
                idx += 1

        for wrong_turns in (0, 1, 2, 3):
            wrong = grid.copy()
            idx = 0
            for r0, r1 in row_ranges:
                for c0, c1 in col_ranges:
                    cell = grid[r0 : r1 + 1, c0 : c1 + 1]
                    if wrong_turns == COUNT_TO_TURNS[counts[idx]]:
                        idx += 1
                        continue
                    wrong[r0 : r1 + 1, c0 : c1 + 1] = np.rot90(cell, wrong_turns)
                    idx += 1
            if np.array_equal(wrong, out):
                return False
        return True
    except AssertionError:
        return False


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        rng = np.random.default_rng()
        size = int(rng.integers(3, 5))
        cell_count = size * size
        counts = _choose_counts(cell_count, rng)
        masks = _assign_masks(counts, rng)
        colors = _assign_colors(counts, rng)
        grid = _render_grid(size, masks, colors)
        if _validate_generated(grid):
            return grid
    raise RuntimeError("failed to generate a valid self-referential rotation puzzle")
