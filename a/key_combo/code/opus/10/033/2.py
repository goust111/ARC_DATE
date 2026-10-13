from __future__ import annotations

import random
from collections import deque

import numpy as np


BACKGROUND = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4

GRID_OPTIONS = (11, 13, 15, 17, 19)
ROW_COUNT_OPTIONS = (3, 5)
MAX_ATTEMPTS = 500


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _count_components(mask: np.ndarray) -> int:
    seen = np.zeros(mask.shape, dtype=bool)
    count = 0
    rows, cols = mask.shape
    for r in range(rows):
        for c in range(cols):
            if not mask[r, c] or seen[r, c]:
                continue
            count += 1
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            while q:
                rr, cc = q.popleft()
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < rows and 0 <= nc < cols and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
    return count


def _extract_structure(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    rows, cols = g.shape
    assert rows % 2 == 1 and cols % 2 == 1, "grid must have odd dimensions"
    assert rows <= 30 and cols <= 30, "grid exceeds ARC size limit"
    assert set(np.unique(g)).issubset({BACKGROUND, BLUE, RED}), "input uses unexpected colors"

    cr, cc = rows // 2, cols // 2
    assert int(g[cr, cc]) == RED, "center marker must be red"
    assert int(np.count_nonzero(g == RED)) == 1, "center marker must be unique"

    original = g == BLUE
    area = int(original.sum())
    assert 4 <= area <= 8, "blue object area must be 4-8"
    assert _count_components(original) == 1, "blue object must be 4-connected"

    coords = np.argwhere(original)
    assert np.all(coords[:, 1] < cc), "blue object must lie strictly left of center"
    assert np.all(coords[:, 0] > 0) and np.all(coords[:, 0] < rows - 1), "blue object touches top/bottom edge"
    assert np.all(coords[:, 1] > 0), "blue object touches left edge"

    row_has = np.any(original, axis=1)
    for r in np.flatnonzero(row_has):
        mirror_r = 2 * cr - int(r)
        assert row_has[mirror_r], "every occupied row must have an occupied mirror row"

    clone = np.zeros_like(original)
    for r, c in coords:
        rr = 2 * cr - int(r)
        rc = 2 * cc - int(c)
        assert 0 <= rr < rows and 0 <= rc < cols
        assert rc > cc, "reflected copy must stay strictly right of center"
        assert not original[rr, rc], "reflected copy overlaps source"
        clone[rr, rc] = True

    return g, clone, cr, cc


def _ray_masks(original: np.ndarray, clone: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = original.shape
    right_cover = np.zeros((rows, cols), dtype=bool)
    left_cover = np.zeros((rows, cols), dtype=bool)

    original_rows = [np.flatnonzero(original[r]) for r in range(rows)]
    clone_rows = [np.flatnonzero(clone[r]) for r in range(rows)]

    for r in range(rows):
        if original_rows[r].size:
            assert clone_rows[r].size, "source row must see reflected row"
            stop = int(clone_rows[r][0])
            for c in original_rows[r]:
                c = int(c)
                if c + 1 < stop:
                    right_cover[r, c + 1 : stop] = True

        if clone_rows[r].size:
            assert original_rows[r].size, "reflected row must see source row"
            stop = int(original_rows[r][-1])
            for c in clone_rows[r]:
                c = int(c)
                if stop + 1 < c:
                    left_cover[r, stop + 1 : c] = True

    return right_cover, left_cover


def _render_output(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    g, clone, cr, cc = _extract_structure(grid)
    original = g == BLUE
    right_cover, left_cover = _ray_masks(original, clone)

    out = g.copy()
    out[clone] = BLUE

    overlap = right_cover & left_cover
    single = right_cover ^ left_cover
    background_mask = out == BACKGROUND
    out[background_mask & single] = YELLOW
    out[background_mask & overlap] = GREEN
    out[cr, cc] = RED
    return out, right_cover, left_cover


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build the 180-degree reflected blue copy and connect the two blue bodies with opposing horizontal rays.

    Input structure:
    - The grid has odd height and width, black background, and exactly one red center marker at the geometric center.
    - A single 4-connected blue object of 4-8 cells lies strictly left of the center and does not touch the border.
    - The set of occupied rows is symmetric around the center row so each source row has a reflected counterpart row.

    Rule:
    - Keep the input unchanged, then add the 180-degree point-reflected blue copy around the red center marker.
    - From every source blue cell, extend a horizontal ray rightward until the first reflected blue cell in that row; from every reflected blue cell, extend a horizontal ray leftward until the first source blue cell in that row.
    - Paint cells hit by exactly one ray yellow, cells hit by both directions green, and never overwrite the red center marker or any blue cell.

    Key insight:
    - The reflected copy determines where each ray stops, so the green overlap band emerges from the interaction between point symmetry and two opposite ray families rather than from an explicitly drawn center strip.
    """
    out, _, _ = _render_output(grid)
    return out


def _intervals_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


def _cells_to_grid(rows: int, cols: int, cells: set[tuple[int, int]]) -> np.ndarray:
    grid = np.zeros((rows, cols), dtype=int)
    cr, cc = rows // 2, cols // 2
    grid[cr, cc] = RED
    for r, c in cells:
        grid[r, c] = BLUE
    return grid


def _row_columns(cells: set[tuple[int, int]]) -> dict[int, list[int]]:
    rows: dict[int, list[int]] = {}
    for r, c in cells:
        rows.setdefault(int(r), []).append(int(c))
    for cols in rows.values():
        cols.sort()
    return rows


def _has_noncontiguous_row_cells(cells: set[tuple[int, int]]) -> bool:
    for cols in _row_columns(cells).values():
        if any(cols[i + 1] - cols[i] > 1 for i in range(len(cols) - 1)):
            return True
    return False


def _frontier(cells: set[tuple[int, int]], min_row: int, max_row: int, max_col: int) -> list[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for nr, nc in _neighbors4(r, c):
            if nr < min_row or nr > max_row or nc < 1 or nc > max_col:
                continue
            if (nr, nc) not in cells:
                out.add((nr, nc))
    return list(out)


def _sample_polyomino(rows: int, cols: int, row_count: int) -> set[tuple[int, int]] | None:
    cr, cc = rows // 2, cols // 2
    k = row_count // 2
    min_row = cr - k
    max_row = cr + k
    max_col = cc - 1
    if max_col < 3:
        return None

    for _ in range(120):
        cells: set[tuple[int, int]] = set()
        center_col = random.randint(max(2, cc - 5), cc - 2)
        columns = {0: center_col}

        for d in range(1, k + 1):
            prev_top = columns[-(d - 1)]
            prev_bottom = columns[d - 1]
            columns[-d] = max(1, min(max_col, prev_top + random.choice((-1, 0, 1))))
            columns[d] = max(1, min(max_col, prev_bottom + random.choice((-1, 0, 1))))

        for d in range(-k, k + 1):
            cells.add((cr + d, columns[d]))

        target_area = random.randint(row_count + 1, 8)
        while len(cells) < target_area:
            frontier = _frontier(cells, min_row, max_row, max_col)
            if not frontier:
                break
            special = [cell for cell in frontier if _has_noncontiguous_row_cells(cells | {cell})]
            if special and random.random() < 0.8:
                cell = random.choice(special)
            else:
                cell = random.choice(frontier)
            cells.add(cell)

        if len(cells) != target_area:
            continue
        if _count_components(_cells_to_grid(rows, cols, cells) == BLUE) != 1:
            continue
        if not _has_noncontiguous_row_cells(cells):
            continue

        row_cols = _row_columns(cells)
        if any((cr + d) not in row_cols for d in range(-k, k + 1)):
            continue
        if not any(row_cols[cr - d] != row_cols[cr + d] for d in range(1, k + 1)):
            continue
        return cells

    return None


def _symmetry_only_output(grid: np.ndarray) -> np.ndarray:
    g, clone, _cr, _cc = _extract_structure(grid)
    out = g.copy()
    out[clone] = BLUE
    return out


def _has_mixed_trace_row(out: np.ndarray) -> bool:
    for r in range(out.shape[0]):
        row = out[r]
        if np.any(row == GREEN) and np.any(row == YELLOW):
            return True
    return False


def _validate_generated_case(grid: np.ndarray) -> bool:
    try:
        out, right_cover, left_cover = _render_output(grid)
    except AssertionError:
        return False

    rows, cols = grid.shape
    cr, cc = rows // 2, cols // 2

    if int(np.count_nonzero(out == GREEN)) == 0:
        return False
    if int(np.count_nonzero(out == YELLOW)) == 0:
        return False
    if not _has_mixed_trace_row(out):
        return False
    if not (right_cover[cr, cc] or left_cover[cr, cc]):
        return False
    if int(out[cr, cc]) != RED:
        return False
    if np.array_equal(out, _symmetry_only_output(grid)):
        return False
    if not np.array_equal(main(grid), out):
        return False
    if max(grid.shape) > 30 or max(out.shape) > 30:
        return False
    if not set(np.unique(out)).issubset({BACKGROUND, BLUE, RED, GREEN, YELLOW}):
        return False
    return True


def _generate_attempt() -> np.ndarray | None:
    rows = random.choice(GRID_OPTIONS)
    cols = random.choice(GRID_OPTIONS)
    row_count = random.choice(ROW_COUNT_OPTIONS)
    cells = _sample_polyomino(rows, cols, row_count)
    if cells is None:
        return None

    grid = _cells_to_grid(rows, cols, cells)
    blue = grid == BLUE
    if _count_components(blue) != 1:
        return None
    if not _validate_generated_case(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 2")
