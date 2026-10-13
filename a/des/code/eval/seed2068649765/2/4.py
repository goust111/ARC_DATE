from __future__ import annotations

import random

import numpy as np


DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

MIN_CELL_SIZE = 3
MAX_CELL_SIZE = 5
MIN_GRID_CELLS = 3
MAX_GRID_CELLS = 5
SPRITE_ATTEMPTS = 800
GENERATION_ATTEMPTS = 500
PALETTE = tuple(range(10))
BASE_MASK_3 = np.array(
    [
        [0, 0, 1],
        [0, 1, 1],
        [1, 1, 0],
    ],
    dtype=bool,
)


def _neighbors4(r: int, c: int, height: int, width: int):
    for dr, dc in DIRS4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < height and 0 <= nc < width:
            yield nr, nc


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False

    h, w = mask.shape
    start = tuple(map(int, coords[0]))
    seen = {start}
    stack = [start]

    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c, h, w):
            if mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))

    return len(seen) == int(mask.sum())


def _rows_cols_mixed(mask: np.ndarray) -> bool:
    h, w = mask.shape
    row_counts = mask.sum(axis=1)
    col_counts = mask.sum(axis=0)
    return bool(
        np.all((row_counts > 0) & (row_counts < w))
        and np.all((col_counts > 0) & (col_counts < h))
    )


def _valid_sprite_mask(mask: np.ndarray) -> bool:
    h, w = mask.shape
    area = h * w
    filled = int(mask.sum())
    min_filled = max(3, h)
    max_filled = area - max(3, h)
    if not (min_filled <= filled <= max_filled):
        return False
    if not _rows_cols_mixed(mask):
        return False
    if not _is_connected(mask):
        return False
    if np.array_equal(mask, np.fliplr(mask)):
        return False
    if np.array_equal(mask, np.flipud(mask)):
        return False
    if np.array_equal(mask, np.flipud(np.fliplr(mask))):
        return False
    return True


def _sample_sprite(cell_size: int, bg: int, fg: int) -> np.ndarray:
    if cell_size == 3:
        return np.where(BASE_MASK_3, fg, bg).astype(int)

    for _ in range(SPRITE_ATTEMPTS):
        cols = list(range(cell_size))
        random.shuffle(cols)
        mask = np.ones((cell_size, cell_size), dtype=bool)
        for r, c in enumerate(cols):
            mask[r, c] = False
        if _valid_sprite_mask(mask):
            return np.where(mask, fg, bg).astype(int)
    raise AssertionError("failed to sample a non-degenerate sprite")


def _apply_cell_symmetry(cell: np.ndarray, row: int, col: int) -> np.ndarray:
    out = np.array(cell, copy=True)
    if row % 2:
        out = np.flipud(out)
    if col % 2:
        out = np.fliplr(out)
    return np.array(out, copy=True)


def _cell_variants(cell: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[bytes] = set()
    for row_parity in (0, 1):
        for col_parity in (0, 1):
            variant = _apply_cell_symmetry(cell, row_parity, col_parity)
            key = variant.tobytes()
            if key not in seen:
                seen.add(key)
                variants.append(variant)
    return variants


def _sample_wrong_cell(
    cell_size: int, bg: int, fg: int, forbidden: list[np.ndarray]
) -> np.ndarray:
    for _ in range(SPRITE_ATTEMPTS):
        density = random.uniform(0.25, 0.75)
        mask = np.array(
            [random.random() < density for _ in range(cell_size * cell_size)],
            dtype=bool,
        ).reshape(cell_size, cell_size)
        if not mask.any() or bool(mask.all()) or not _rows_cols_mixed(mask):
            continue
        candidate = np.where(mask, fg, bg).astype(int)
        if not any(np.array_equal(candidate, banned) for banned in forbidden):
            return candidate
    raise AssertionError("failed to sample a distinct tampered cell")


def _regular_intervals(lines: list[int], length: int) -> list[tuple[int, int]] | None:
    if not lines:
        return None

    lines = sorted(set(map(int, lines)))
    if any(b - a == 1 for a, b in zip(lines, lines[1:])):
        return None

    bounds = [-1, *lines, int(length)]
    intervals: list[tuple[int, int]] = []
    for left, right in zip(bounds, bounds[1:]):
        start, end = left + 1, right
        if end <= start:
            return None
        intervals.append((int(start), int(end)))

    sizes = {end - start for start, end in intervals}
    if len(sizes) != 1:
        return None
    size = next(iter(sizes))
    if not (MIN_CELL_SIZE <= size <= MAX_CELL_SIZE):
        return None
    if len(intervals) < 2:
        return None
    return intervals


def _find_layout(
    grid: np.ndarray,
) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    candidates: list[tuple[int, list[tuple[int, int]], list[tuple[int, int]]]] = []

    for color in map(int, np.unique(g)):
        row_lines = [r for r in range(h) if np.all(g[r, :] == color)]
        col_lines = [c for c in range(w) if np.all(g[:, c] == color)]
        row_intervals = _regular_intervals(row_lines, h)
        col_intervals = _regular_intervals(col_lines, w)
        if row_intervals is None or col_intervals is None:
            continue
        candidates.append((int(color), row_intervals, col_intervals))

    assert len(candidates) == 1, "expected one separator color with regular full lines"
    return candidates[0]


def _extract_cells(
    grid: np.ndarray,
    row_intervals: list[tuple[int, int]],
    col_intervals: list[tuple[int, int]],
) -> list[list[np.ndarray]]:
    g = np.asarray(grid, dtype=int)
    cells: list[list[np.ndarray]] = []
    for r0, r1 in row_intervals:
        row_cells: list[np.ndarray] = []
        for c0, c1 in col_intervals:
            row_cells.append(np.array(g[r0:r1, c0:c1], copy=True))
        cells.append(row_cells)
    return cells


def _neighbor_match_counts(cells: list[list[np.ndarray]]) -> np.ndarray:
    rows = len(cells)
    cols = len(cells[0])
    counts = np.zeros((rows, cols), dtype=int)

    for r in range(rows):
        for c in range(cols - 1):
            if np.array_equal(np.fliplr(cells[r][c]), cells[r][c + 1]):
                counts[r, c] += 1
                counts[r, c + 1] += 1

    for r in range(rows - 1):
        for c in range(cols):
            if np.array_equal(np.flipud(cells[r][c]), cells[r + 1][c]):
                counts[r, c] += 1
                counts[r + 1, c] += 1

    return counts


def _all_neighbor_relations_hold(cells: list[list[np.ndarray]]) -> bool:
    rows = len(cells)
    cols = len(cells[0])

    for r in range(rows):
        for c in range(cols - 1):
            if not np.array_equal(np.fliplr(cells[r][c]), cells[r][c + 1]):
                return False

    for r in range(rows - 1):
        for c in range(cols):
            if not np.array_equal(np.flipud(cells[r][c]), cells[r + 1][c]):
                return False

    return True


def _render_cells(
    cells: list[list[np.ndarray]],
    separator_color: int,
) -> np.ndarray:
    rows = len(cells)
    cols = len(cells[0])
    cell_h, cell_w = cells[0][0].shape
    h = rows * cell_h + (rows - 1)
    w = cols * cell_w + (cols - 1)
    out = np.full((h, w), int(separator_color), dtype=int)

    for r in range(rows):
        for c in range(cols):
            r0 = r * (cell_h + 1)
            c0 = c * (cell_w + 1)
            out[r0 : r0 + cell_h, c0 : c0 + cell_w] = cells[r][c]

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair a separator-delimited board of mirrored cell sprites by replacing the
    1-2 tampered cells with the mirror implied by a correct neighbor.
    """
    g = np.asarray(grid, dtype=int)
    separator_color, row_intervals, col_intervals = _find_layout(g)
    cells = _extract_cells(g, row_intervals, col_intervals)
    counts = _neighbor_match_counts(cells)
    bad_cells = [tuple(map(int, rc)) for rc in np.argwhere(counts == 0)]

    assert 1 <= len(bad_cells) <= 2, "expected exactly 1-2 noncompliant cells"
    repaired = [[np.array(cell, copy=True) for cell in row] for row in cells]

    for r, c in bad_cells:
        if c > 0:
            repaired[r][c] = np.array(np.fliplr(cells[r][c - 1]), copy=True)
        else:
            assert r > 0, "a first-column anomaly must have an upper neighbor"
            repaired[r][c] = np.array(np.flipud(cells[r - 1][c]), copy=True)

    assert _all_neighbor_relations_hold(repaired), "repaired board must satisfy all mirror constraints"
    out = _render_cells(repaired, separator_color)
    return out


def _ideal_cells(base_cell: np.ndarray, rows: int, cols: int) -> list[list[np.ndarray]]:
    return [
        [_apply_cell_symmetry(base_cell, r, c) for c in range(cols)]
        for r in range(rows)
    ]


def _source_of_cell(r: int, c: int) -> tuple[int, int] | None:
    if c > 0:
        return r, c - 1
    if r > 0:
        return r - 1, c
    return None


def _valid_tampered_positions(
    rows: int,
    cols: int,
    positions: list[tuple[int, int]],
) -> bool:
    pos_set = set(positions)
    if len(pos_set) != len(positions):
        return False
    if (0, 0) in pos_set:
        return False

    for r, c in pos_set:
        src = _source_of_cell(r, c)
        if src is None or src in pos_set:
            return False

    for r in range(rows):
        for c in range(cols):
            if (r, c) in pos_set:
                continue
            if not any((nr, nc) not in pos_set for nr, nc in _neighbors4(r, c, rows, cols)):
                return False

    return True


def _choose_tampered_positions(rows: int, cols: int) -> list[tuple[int, int]]:
    up_positions = [(r, 0) for r in range(1, rows)]
    left_positions = [(r, c) for r in range(rows) for c in range(1, cols)]

    if random.random() < 0.5:
        branch = random.choice(("left", "up"))
        candidates = left_positions if branch == "left" else up_positions
        candidates = [pt for pt in candidates if _valid_tampered_positions(rows, cols, [pt])]
        assert candidates, "expected at least one valid single-anomaly position"
        return [random.choice(candidates)]

    pairs = [
        [up_pt, left_pt]
        for up_pt in up_positions
        for left_pt in left_positions
        if _valid_tampered_positions(rows, cols, [up_pt, left_pt])
    ]
    assert pairs, "expected at least one valid two-anomaly pair"
    return random.choice(pairs)


def _build_instance() -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    rows = random.randint(MIN_GRID_CELLS, MAX_GRID_CELLS)
    cols = random.randint(MIN_GRID_CELLS, MAX_GRID_CELLS)
    cell_size = random.randint(MIN_CELL_SIZE, MAX_CELL_SIZE)
    separator_color, bg, fg = random.sample(PALETTE, 3)

    base_cell = _sample_sprite(cell_size, bg, fg)
    correct_cells = _ideal_cells(base_cell, rows, cols)
    tampered_positions = _choose_tampered_positions(rows, cols)
    valid_family = _cell_variants(base_cell)

    input_cells = [[np.array(cell, copy=True) for cell in row] for row in correct_cells]
    for r, c in tampered_positions:
        input_cells[r][c] = _sample_wrong_cell(cell_size, bg, fg, valid_family)

    inp = _render_cells(input_cells, separator_color)
    out = _render_cells(correct_cells, separator_color)
    return inp, out, tampered_positions


def _find_bad_cells(grid: np.ndarray) -> set[tuple[int, int]]:
    _separator_color, row_intervals, col_intervals = _find_layout(grid)
    cells = _extract_cells(grid, row_intervals, col_intervals)
    counts = _neighbor_match_counts(cells)
    return {tuple(map(int, rc)) for rc in np.argwhere(counts == 0)}


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        inp, expected, tampered_positions = _build_instance()
        found = _find_bad_cells(inp)
        if found != set(tampered_positions):
            continue
        if not np.array_equal(main(inp), expected):
            continue
        return inp

    raise AssertionError("failed to generate a valid puzzle instance")
