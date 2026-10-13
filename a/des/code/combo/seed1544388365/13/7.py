from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREY = 5


def _runs(mask: np.ndarray, value: bool) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    i = 0
    n = int(mask.size)
    while i < n:
        if bool(mask[i]) != value:
            i += 1
            continue
        j = i
        while j < n and bool(mask[j]) == value:
            j += 1
        runs.append((i, j))
        i = j
    return runs


def _find_lattice(grid: np.ndarray) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]]]:
    h, w = map(int, grid.shape)
    candidates: list[tuple[int, list[tuple[int, int]], list[tuple[int, int]]]] = []

    for color in np.unique(grid):
        color = int(color)
        row_mask = np.all(grid == color, axis=1)
        col_mask = np.all(grid == color, axis=0)
        if not np.any(row_mask) or not np.any(col_mask):
            continue

        row_line_runs = _runs(row_mask, True)
        col_line_runs = _runs(col_mask, True)
        if not row_line_runs or not col_line_runs:
            continue
        if any(end - start != 1 for start, end in row_line_runs):
            continue
        if any(end - start != 1 for start, end in col_line_runs):
            continue

        row_spans = _runs(row_mask, False)
        col_spans = _runs(col_mask, False)
        if not row_spans or not col_spans:
            continue

        row_sizes = {end - start for start, end in row_spans}
        col_sizes = {end - start for start, end in col_spans}
        if len(row_sizes) != 1 or len(col_sizes) != 1:
            continue

        cell_h = next(iter(row_sizes))
        cell_w = next(iter(col_sizes))
        if not (3 <= cell_h <= 5 and 3 <= cell_w <= 5):
            continue

        candidates.append((color, row_spans, col_spans))

    assert len(candidates) == 1, "expected a unique grey lattice"
    return candidates[0]


def _invert_cell(cell: np.ndarray) -> np.ndarray:
    out = cell.copy()
    black_mask = cell == BLACK
    blue_mask = cell == BLUE
    out[black_mask] = BLUE
    out[blue_mask] = BLACK
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Partition the grid by grey separator lines and flip each cell iff its red-marker count is odd.

    The task instances place red markers on black positions, so removing a marker restores black.
    Grey separator lines remain unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    _sep_color, row_spans, col_spans = _find_lattice(g)
    out = g.copy()

    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            cell = g[r0:r1, c0:c1]
            assert np.all(np.isin(cell, (BLACK, BLUE, RED))), "cell contains unexpected colors"

            base = cell.copy()
            red_mask = base == RED
            marker_count = int(np.sum(red_mask))
            base[red_mask] = BLACK

            out[r0:r1, c0:c1] = _invert_cell(base) if marker_count % 2 == 1 else base

    return out


# --------------- generator (local) ---------------

CELL_SIZE_RANGE = (3, 5)
CELL_COUNT_RANGE = (2, 4)
PATTERN_POOL_MAX = 4
MAX_MARKERS_PER_CELL = 3
GENERATION_ATTEMPTS = 400
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _random_blob_mask(height: int, width: int, target: int) -> np.ndarray | None:
    mask = np.zeros((height, width), dtype=bool)
    start = (random.randrange(height), random.randrange(width))
    mask[start] = True

    while int(np.sum(mask)) < target:
        frontier: list[tuple[int, int]] = []
        for r, c in np.argwhere(mask):
            for dr, dc in DIR4:
                rr = int(r) + dr
                cc = int(c) + dc
                if 0 <= rr < height and 0 <= cc < width and not mask[rr, cc]:
                    frontier.append((rr, cc))
        if not frontier:
            return None
        rr, cc = random.choice(frontier)
        mask[rr, cc] = True

    return mask


def _make_pattern(height: int, width: int) -> np.ndarray:
    area = height * width
    min_blue = 2
    max_blue = area - (MAX_MARKERS_PER_CELL + 1)

    for _ in range(200):
        mode = random.choice(("rect", "plus", "stripes", "diag", "blob"))
        mask = np.zeros((height, width), dtype=bool)

        if mode == "rect":
            rect_h = random.randint(1, max(1, height - 1))
            rect_w = random.randint(1, max(1, width - 1))
            r0 = random.randint(0, height - rect_h)
            c0 = random.randint(0, width - rect_w)
            mask[r0 : r0 + rect_h, c0 : c0 + rect_w] = True
        elif mode == "plus":
            cr = random.randrange(height)
            cc = random.randrange(width)
            mask[cr, :] = True
            mask[:, cc] = True
        elif mode == "stripes":
            if random.random() < 0.5:
                count = random.randint(1, min(2, height))
                rows = random.sample(range(height), count)
                mask[rows, :] = True
            else:
                count = random.randint(1, min(2, width))
                cols = random.sample(range(width), count)
                mask[:, cols] = True
        elif mode == "diag":
            use_anti = random.random() < 0.5
            for r in range(height):
                denom = max(1, height - 1)
                c = int(round((width - 1) * r / denom))
                c = (width - 1 - c) if use_anti else c
                mask[r, c] = True
                if width >= 4 and random.random() < 0.35:
                    dc = 1 if c == 0 else (-1 if c == width - 1 else random.choice((-1, 1)))
                    mask[r, c + dc] = True
        else:
            target = random.randint(max(3, area // 4), max(3, area // 2))
            blob = _random_blob_mask(height, width, target)
            if blob is None:
                continue
            mask = blob

        if height == width and random.random() < 0.5:
            mask = np.rot90(mask, random.randrange(4))
        if random.random() < 0.5:
            mask = np.flipud(mask)
        if random.random() < 0.5:
            mask = np.fliplr(mask)
        mask = np.asarray(mask, dtype=bool)

        blue_count = int(np.sum(mask))
        black_count = area - blue_count
        if min_blue <= blue_count <= max_blue and black_count >= MAX_MARKERS_PER_CELL + 1:
            return mask.copy()

    raise RuntimeError("failed to build a non-degenerate cell pattern")


def _build_pattern_pool(height: int, width: int, pool_size: int) -> list[np.ndarray] | None:
    pool: list[np.ndarray] = []
    seen: set[bytes] = set()
    for _ in range(400):
        if len(pool) == pool_size:
            return pool
        pattern = _make_pattern(height, width)
        key = pattern.tobytes()
        if key in seen:
            continue
        seen.add(key)
        pool.append(pattern)
    return None


def _cell_origin(cell_r: int, cell_c: int, cell_h: int, cell_w: int) -> tuple[int, int]:
    return 1 + cell_r * (cell_h + 1), 1 + cell_c * (cell_w + 1)


def _render_base_cell(pattern: np.ndarray) -> np.ndarray:
    cell = np.full(pattern.shape, BLACK, dtype=int)
    cell[pattern] = BLUE
    return cell


def _place_markers(cell: np.ndarray, marker_count: int) -> np.ndarray | None:
    out = cell.copy()
    black_positions = [(int(r), int(c)) for r, c in np.argwhere(out == BLACK)]
    if marker_count > len(black_positions):
        return None
    for r, c in random.sample(black_positions, marker_count):
        out[r, c] = RED
    return out


def _extract_cell(grid: np.ndarray, index: int, n_cols: int, cell_h: int, cell_w: int) -> np.ndarray:
    cell_r = index // n_cols
    cell_c = index % n_cols
    r0, c0 = _cell_origin(cell_r, cell_c, cell_h, cell_w)
    return grid[r0 : r0 + cell_h, c0 : c0 + cell_w]


def _gen_attempt() -> np.ndarray | None:
    cell_h = random.randint(*CELL_SIZE_RANGE)
    cell_w = random.randint(*CELL_SIZE_RANGE)
    n_rows = random.randint(*CELL_COUNT_RANGE)
    n_cols = random.randint(*CELL_COUNT_RANGE)
    total_cells = n_rows * n_cols

    pool_size = random.randint(2, min(PATTERN_POOL_MAX, total_cells))
    pattern_pool = _build_pattern_pool(cell_h, cell_w, pool_size)
    if pattern_pool is None:
        return None

    pattern_ids = [0] * total_cells
    marker_counts = [0] * total_cells

    special_cells = random.sample(range(total_cells), 3)
    for idx, count in zip(special_cells, (0, 1, 2)):
        pattern_ids[idx] = 0
        marker_counts[idx] = count

    for idx in range(total_cells):
        if idx in special_cells:
            continue
        pattern_ids[idx] = random.randrange(pool_size)
        marker_counts[idx] = random.choices((0, 1, 2, 3), weights=(3, 4, 3, 2), k=1)[0]

    height = n_rows * cell_h + n_rows + 1
    width = n_cols * cell_w + n_cols + 1
    grid = np.full((height, width), GREY, dtype=int)

    for idx in range(total_cells):
        cell_r = idx // n_cols
        cell_c = idx % n_cols
        r0, c0 = _cell_origin(cell_r, cell_c, cell_h, cell_w)

        base_cell = _render_base_cell(pattern_pool[pattern_ids[idx]])
        marked = _place_markers(base_cell, marker_counts[idx])
        if marked is None:
            return None
        grid[r0 : r0 + cell_h, c0 : c0 + cell_w] = marked

    try:
        out = main(grid)
    except Exception:
        return None

    reference_base = _render_base_cell(pattern_pool[0])
    reference_inverted = _invert_cell(reference_base)
    for idx, count in zip(special_cells, (0, 1, 2)):
        cell_out = _extract_cell(out, idx, n_cols, cell_h, cell_w)
        expected = reference_inverted if count % 2 == 1 else reference_base
        if not np.array_equal(cell_out, expected):
            return None

    if np.array_equal(out, grid):
        return None
    if not np.all(out[grid == GREY] == GREY):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
