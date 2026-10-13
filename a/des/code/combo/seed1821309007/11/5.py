from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]

CELL_SIZE_RANGE = (4, 7)
GRID_CELLS_RANGE = (2, 4)
TARGET_COUNT_MIN = 2
MAX_GENERATION_ATTEMPTS = 400

RULE_TYPES = (
    "horizontal",
    "vertical",
    "quadrants",
    "diag_main",
    "diag_anti",
    "checker",
)


def _separator_indices(line_mask: np.ndarray) -> list[int]:
    return [int(i) for i in np.flatnonzero(line_mask)]


def _intervals_from_separators(indices: list[int], length: int) -> list[tuple[int, int]]:
    bounds = [-1, *indices, length]
    intervals = [(bounds[i] + 1, bounds[i + 1]) for i in range(len(bounds) - 1)]
    sizes = [end - start for start, end in intervals]
    assert intervals and min(sizes) > 0, "grid must contain non-empty cells"
    assert len(set(sizes)) == 1, "all cells must have the same size"
    return intervals


def _cell_slices(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    row_seps = _separator_indices(np.all(grid == GREY, axis=1))
    col_seps = _separator_indices(np.all(grid == GREY, axis=0))
    row_intervals = _intervals_from_separators(row_seps, grid.shape[0])
    col_intervals = _intervals_from_separators(col_seps, grid.shape[1])
    cell_size = row_intervals[0][1] - row_intervals[0][0]
    assert all(r1 - r0 == cell_size for r0, r1 in row_intervals)
    assert all(c1 - c0 == cell_size for c0, c1 in col_intervals)
    return row_intervals, col_intervals


def _non_bg_non_separator(cell: np.ndarray) -> np.ndarray:
    return (cell != BLACK) & (cell != GREY)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transfer the reference cell's color layout onto every grey target shape.

    The input is divided by full grey separator rows/columns into equal square
    cells. Exactly one cell is a multicolor reference pattern. Every other
    non-empty cell contains a grey target shape. Recolor each grey target pixel
    using the reference cell's color at the same local cell coordinate.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    row_intervals, col_intervals = _cell_slices(g)
    reference_pos: tuple[int, int] | None = None
    reference_cell: np.ndarray | None = None
    target_positions: list[tuple[int, int]] = []

    for ri, (r0, r1) in enumerate(row_intervals):
        for ci, (c0, c1) in enumerate(col_intervals):
            cell = g[r0:r1, c0:c1]
            colors = set(map(int, cell.ravel())) - {BLACK, GREY}
            if len(colors) >= 2:
                assert reference_pos is None, "expected exactly one multicolor reference cell"
                reference_pos = (ri, ci)
                reference_cell = cell.copy()
            elif np.any(cell == GREY):
                target_positions.append((ri, ci))
            else:
                raise AssertionError("every non-reference cell must contain a grey target shape")

    assert reference_pos is not None and reference_cell is not None, "missing reference cell"
    assert len(target_positions) >= 1, "expected at least one target cell"

    ref_mask = _non_bg_non_separator(reference_cell)
    assert bool(ref_mask.any()), "reference cell must expose at least one colored pixel"

    out = g.copy()
    for ri, ci in target_positions:
        r0, r1 = row_intervals[ri]
        c0, c1 = col_intervals[ci]
        cell = out[r0:r1, c0:c1]
        target_mask = cell == GREY
        assert np.all(ref_mask[target_mask]), "target pixels must map to visible reference coordinates"
        cell[target_mask] = reference_cell[target_mask]

    return out


def _horizontal_rule(size: int, colors: list[int]) -> np.ndarray:
    pat = np.zeros((size, size), dtype=int)
    cut_points = sorted(random.sample(range(1, size), len(colors) - 1))
    bands = [0, *cut_points, size]
    for i in range(len(bands) - 1):
        pat[bands[i] : bands[i + 1], :] = colors[i]
    return pat


def _vertical_rule(size: int, colors: list[int]) -> np.ndarray:
    pat = np.zeros((size, size), dtype=int)
    cut_points = sorted(random.sample(range(1, size), len(colors) - 1))
    bands = [0, *cut_points, size]
    for i in range(len(bands) - 1):
        pat[:, bands[i] : bands[i + 1]] = colors[i]
    return pat


def _quadrant_rule(size: int, colors: list[int]) -> np.ndarray:
    pat = np.zeros((size, size), dtype=int)
    split_r = random.randint(1, size - 1)
    split_c = random.randint(1, size - 1)
    pat[:split_r, :split_c] = colors[0]
    pat[:split_r, split_c:] = colors[1]
    pat[split_r:, :split_c] = colors[2]
    pat[split_r:, split_c:] = colors[3]
    return pat


def _diag_main_rule(size: int, colors: list[int]) -> np.ndarray:
    rr, cc = np.indices((size, size))
    diff = rr - cc
    thresholds = sorted(random.sample(range(-(size - 1), size), 2))
    pat = np.full((size, size), colors[2], dtype=int)
    pat[diff <= thresholds[1]] = colors[1]
    pat[diff <= thresholds[0]] = colors[0]
    return pat


def _diag_anti_rule(size: int, colors: list[int]) -> np.ndarray:
    rr, cc = np.indices((size, size))
    summ = rr + cc
    thresholds = sorted(random.sample(range(1, 2 * size - 2), 2))
    pat = np.full((size, size), colors[2], dtype=int)
    pat[summ <= thresholds[1]] = colors[1]
    pat[summ <= thresholds[0]] = colors[0]
    return pat


def _checker_rule(size: int, colors: list[int]) -> np.ndarray:
    rr, cc = np.indices((size, size))
    mode = random.choice((0, 1))
    pat = np.where((rr + cc + mode) % 2 == 0, colors[0], colors[1]).astype(int)
    if size >= 5 and random.random() < 0.5:
        center = size // 2
        pat[center, :] = colors[2]
        pat[:, center] = colors[2]
    return pat


def _make_reference_pattern(size: int) -> np.ndarray:
    rule = random.choice(RULE_TYPES)
    if rule == "horizontal":
        color_count = random.randint(2, min(4, size))
        return _horizontal_rule(size, random.sample(PALETTE, color_count))
    if rule == "vertical":
        color_count = random.randint(2, min(4, size))
        return _vertical_rule(size, random.sample(PALETTE, color_count))
    if rule == "quadrants":
        return _quadrant_rule(size, random.sample(PALETTE, 4))
    if rule == "diag_main":
        return _diag_main_rule(size, random.sample(PALETTE, 3))
    if rule == "diag_anti":
        return _diag_anti_rule(size, random.sample(PALETTE, 3))
    return _checker_rule(size, random.sample(PALETTE, 3))


def _plus_mask(size: int) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    cr = random.randrange(size)
    cc = random.randrange(size)
    mask[cr, :] = True
    mask[:, cc] = True
    return mask


def _rectangle_mask(size: int) -> np.ndarray:
    h = random.randint(2, size)
    w = random.randint(2, size)
    r0 = random.randint(0, size - h)
    c0 = random.randint(0, size - w)
    mask = np.zeros((size, size), dtype=bool)
    mask[r0 : r0 + h, c0 : c0 + w] = True
    return mask


def _ring_mask(size: int) -> np.ndarray | None:
    if size < 4:
        return None
    h = random.randint(3, size)
    w = random.randint(3, size)
    r0 = random.randint(0, size - h)
    c0 = random.randint(0, size - w)
    mask = np.zeros((size, size), dtype=bool)
    mask[r0 : r0 + h, c0 : c0 + w] = True
    mask[r0 + 1 : r0 + h - 1, c0 + 1 : c0 + w - 1] = False
    return mask


def _l_mask(size: int) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    h = random.randint(2, size)
    w = random.randint(2, size)
    r0 = random.randint(0, size - h)
    c0 = random.randint(0, size - w)
    corner = random.choice(("tl", "tr", "bl", "br"))
    if corner == "tl":
        mask[r0 : r0 + h, c0] = True
        mask[r0 + h - 1, c0 : c0 + w] = True
    elif corner == "tr":
        mask[r0 : r0 + h, c0 + w - 1] = True
        mask[r0 + h - 1, c0 : c0 + w] = True
    elif corner == "bl":
        mask[r0 : r0 + h, c0] = True
        mask[r0, c0 : c0 + w] = True
    else:
        mask[r0 : r0 + h, c0 + w - 1] = True
        mask[r0, c0 : c0 + w] = True
    return mask


def _t_mask(size: int) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    top = random.randint(0, size - 2)
    left = random.randint(0, size - 2)
    width = random.randint(2, size - left)
    stem_col = random.randint(left, left + width - 1)
    stem_len = random.randint(2, size - top)
    mask[top, left : left + width] = True
    mask[top : top + stem_len, stem_col] = True
    return mask


def _diag_line_mask(size: int) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    if random.random() < 0.5:
        offset = random.randint(-(size - 1), size - 1)
        for r in range(size):
            c = r + offset
            if 0 <= c < size:
                mask[r, c] = True
    else:
        offset = random.randint(0, 2 * size - 2)
        for r in range(size):
            c = offset - r
            if 0 <= c < size:
                mask[r, c] = True
    return mask


def _stair_mask(size: int) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    steps = random.randint(2, size)
    cols = sorted(random.sample(range(size), steps))
    rows = sorted(random.sample(range(size), steps))
    for i in range(steps):
        r = rows[i]
        c = cols[i]
        r_prev = rows[i - 1] if i > 0 else r
        c_prev = cols[i - 1] if i > 0 else c
        mask[min(r_prev, r) : max(r_prev, r) + 1, c] = True
        mask[r, min(c_prev, c) : max(c_prev, c) + 1] = True
    return mask


def _blob_mask(size: int) -> np.ndarray:
    target = random.randint(max(3, size), max(4, size * size // 2))
    cells = {(random.randrange(size), random.randrange(size))}
    while len(cells) < target:
        r, c = random.choice(tuple(cells))
        nbrs = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
        valid = [(rr, cc) for rr, cc in nbrs if 0 <= rr < size and 0 <= cc < size]
        cells.add(random.choice(valid))
    mask = np.zeros((size, size), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _random_target_mask(size: int) -> np.ndarray:
    builders = [
        _plus_mask,
        _rectangle_mask,
        _l_mask,
        _t_mask,
        _diag_line_mask,
        _stair_mask,
        _blob_mask,
    ]
    if size >= 4:
        builders.append(_ring_mask)

    for _ in range(40):
        builder = random.choice(builders)
        mask = builder(size)
        if mask is None:
            continue
        area = int(mask.sum())
        if 2 <= area <= size * size - 1:
            return mask
    raise RuntimeError("failed to sample a valid target mask")


def _assemble_grid(
    cell_size: int,
    rows: int,
    cols: int,
    ref_pos: tuple[int, int],
    reference: np.ndarray,
    target_masks: dict[tuple[int, int], np.ndarray],
) -> np.ndarray:
    height = rows * cell_size + (rows - 1)
    width = cols * cell_size + (cols - 1)
    grid = np.full((height, width), BLACK, dtype=int)

    for r in range(1, rows):
        grid[r * cell_size + (r - 1), :] = GREY
    for c in range(1, cols):
        grid[:, c * cell_size + (c - 1)] = GREY

    for ri in range(rows):
        for ci in range(cols):
            r0 = ri * (cell_size + 1)
            c0 = ci * (cell_size + 1)
            cell = grid[r0 : r0 + cell_size, c0 : c0 + cell_size]
            if (ri, ci) == ref_pos:
                cell[:] = reference
            else:
                cell[target_masks[(ri, ci)]] = GREY

    return grid


def _gen_attempt() -> np.ndarray | None:
    cell_size = random.randint(*CELL_SIZE_RANGE)
    rows = random.randint(*GRID_CELLS_RANGE)
    cols = random.randint(*GRID_CELLS_RANGE)
    total_cells = rows * cols
    if total_cells < TARGET_COUNT_MIN + 1:
        return None

    ref_pos = (random.randrange(rows), random.randrange(cols))
    reference = _make_reference_pattern(cell_size)
    if len(set(map(int, reference.ravel()))) < 2:
        return None

    positions = [(r, c) for r in range(rows) for c in range(cols) if (r, c) != ref_pos]
    random.shuffle(positions)
    target_count = random.randint(TARGET_COUNT_MIN, len(positions))
    active_targets = set(positions[:target_count])
    if len(active_targets) < TARGET_COUNT_MIN:
        return None

    target_masks: dict[tuple[int, int], np.ndarray] = {}
    signatures: set[bytes] = set()
    for pos in positions:
        if pos not in active_targets:
            target_masks[pos] = _random_target_mask(cell_size)
            continue
        for _ in range(30):
            mask = _random_target_mask(cell_size)
            sig = mask.tobytes()
            if sig not in signatures:
                signatures.add(sig)
                target_masks[pos] = mask
                break
        else:
            return None

    grid = _assemble_grid(cell_size, rows, cols, ref_pos, reference, target_masks)
    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
