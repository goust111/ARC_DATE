from __future__ import annotations

from collections import Counter

import numpy as np


_RNG = np.random.default_rng()

# --------------- solver helpers ---------------


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_lattice(grid: np.ndarray, bg: int) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]]]:
    height, width = map(int, grid.shape)
    candidates: list[tuple[int, list[tuple[int, int]], list[tuple[int, int]]]] = []

    for color in map(int, np.unique(grid)):
        if color == bg:
            continue

        sep_rows = np.flatnonzero(np.all(grid == color, axis=1)).tolist()
        sep_cols = np.flatnonzero(np.all(grid == color, axis=0)).tolist()
        if len(sep_rows) < 2 or len(sep_cols) < 2:
            continue
        if sep_rows[0] != 0 or sep_rows[-1] != height - 1:
            continue
        if sep_cols[0] != 0 or sep_cols[-1] != width - 1:
            continue

        row_gaps = np.diff(sep_rows)
        col_gaps = np.diff(sep_cols)
        if row_gaps.size == 0 or col_gaps.size == 0:
            continue
        if np.any(row_gaps <= 1) or np.any(col_gaps <= 1):
            continue
        if not np.all(row_gaps == row_gaps[0]):
            continue
        if not np.all(col_gaps == col_gaps[0]):
            continue

        row_bands = [(sep_rows[i] + 1, sep_rows[i + 1]) for i in range(len(sep_rows) - 1)]
        col_bands = [(sep_cols[i] + 1, sep_cols[i + 1]) for i in range(len(sep_cols) - 1)]
        if any(r0 >= r1 for r0, r1 in row_bands):
            continue
        if any(c0 >= c1 for c0, c1 in col_bands):
            continue

        row_sizes = {r1 - r0 for r0, r1 in row_bands}
        col_sizes = {c1 - c0 for c0, c1 in col_bands}
        if len(row_sizes) != 1 or len(col_sizes) != 1:
            continue

        candidates.append((color, row_bands, col_bands))

    assert len(candidates) == 1, "expected exactly one separator lattice"
    return candidates[0]


def _shift_right(cell: np.ndarray, offset: int, bg: int) -> np.ndarray:
    height, width = map(int, cell.shape)
    out = np.full((height, width), bg, dtype=int)
    if offset < width:
        out[:, offset:] = cell[:, : width - offset]
    return out


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _most_common_color(g)
    sep, row_bands, col_bands = _find_lattice(g, bg)
    row_count = len(row_bands)
    col_count = len(col_bands)
    assert row_count >= 1 and col_count >= 3, "expected at least one row and three columns"

    cell_h = row_bands[0][1] - row_bands[0][0]
    cell_w = col_bands[0][1] - col_bands[0][0]

    odd_cells: list[tuple[int, np.ndarray]] = []
    for r0, r1 in row_bands:
        row_cells = [g[r0:r1, c0:c1].copy() for c0, c1 in col_bands]
        sigs = [cell.tobytes() for cell in row_cells]
        counts = Counter(sigs)
        assert sorted(counts.values()) == [1, col_count - 1], "each row must contain one unique odd cell"
        odd_sig = next(sig for sig, count in counts.items() if count == 1)
        odd_col = sigs.index(odd_sig)
        odd_cells.append((odd_col, row_cells[odd_col]))

    out_h = row_count * cell_h + row_count + 1
    out_w = cell_w + 2
    out = np.full((out_h, out_w), sep, dtype=int)
    for row_idx, (odd_col, cell) in enumerate(odd_cells):
        shifted = _shift_right(cell, odd_col, bg)
        r0 = 1 + row_idx * (cell_h + 1)
        out[r0 : r0 + cell_h, 1 : 1 + cell_w] = shifted

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Identify the odd sprite in each lattice row, then emit those sprites in a
    single output column, shifting each one right by its original column index.
    """
    return _solve(grid)


# --------------- generator helpers ---------------


GEN_ATTEMPTS = 400
ROW_RANGE = (2, 4)
COL_RANGE = (3, 5)
CELL_H_RANGE = (4, 7)
CELL_W_RANGE = (5, 8)
PALETTE = tuple(range(10))


def _neighbors(r: int, c: int, height: int, width: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < height:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < width:
        out.append((r, c + 1))
    return out


def _random_connected_mask(max_h: int, max_w: int) -> np.ndarray:
    height = int(_RNG.integers(2, max_h + 1))
    width = int(_RNG.integers(2, max_w + 1))
    area = height * width
    target = int(_RNG.integers(max(2, area // 3), max(3, (2 * area) // 3) + 1))

    mask = np.zeros((height, width), dtype=bool)
    mask[0, 0] = True
    while int(mask.sum()) < target:
        candidates: set[tuple[int, int]] = set()
        for r, c in np.argwhere(mask):
            for nr, nc in _neighbors(int(r), int(c), height, width):
                if not mask[nr, nc]:
                    candidates.add((nr, nc))
        nr, nc = list(candidates)[int(_RNG.integers(len(candidates)))]
        mask[nr, nc] = True
    return mask


def _render_cell(mask: np.ndarray, color: int, cell_h: int, cell_w: int, bg: int) -> np.ndarray:
    cell = np.full((cell_h, cell_w), bg, dtype=int)
    h, w = map(int, mask.shape)
    cell[:h, :w][mask] = color
    return cell


def _balanced_branches(row_count: int) -> list[str]:
    half = row_count // 2
    if row_count % 2 == 0:
        branches = ["color"] * half + ["shape"] * half
    else:
        if int(_RNG.integers(2)) == 0:
            branches = ["color"] * (half + 1) + ["shape"] * half
        else:
            branches = ["color"] * half + ["shape"] * (half + 1)
    _RNG.shuffle(branches)
    return branches


def _make_row_patterns(
    cell_h: int,
    cell_w: int,
    bg: int,
    base_color: int,
    odd_color: int | None,
    branch: str,
) -> tuple[np.ndarray, np.ndarray]:
    max_sprite_h = cell_h - 1 if cell_h > 2 else cell_h
    max_sprite_w = cell_w - 1 if cell_w > 2 else cell_w

    base_mask = _random_connected_mask(max_sprite_h, max_sprite_w)
    base_cell = _render_cell(base_mask, base_color, cell_h, cell_w, bg)

    if branch == "color":
        assert odd_color is not None
        odd_cell = _render_cell(base_mask, odd_color, cell_h, cell_w, bg)
        return base_cell, odd_cell

    for _ in range(100):
        odd_mask = _random_connected_mask(max_sprite_h, max_sprite_w)
        odd_cell = _render_cell(odd_mask, base_color, cell_h, cell_w, bg)
        if not np.array_equal(odd_cell, base_cell):
            return base_cell, odd_cell
    raise AssertionError("failed to generate a distinct shape oddity")


def _gen_attempt() -> np.ndarray | None:
    row_count = int(_RNG.integers(ROW_RANGE[0], ROW_RANGE[1] + 1))
    cell_w = int(_RNG.integers(max(CELL_W_RANGE[0], row_count + 1), CELL_W_RANGE[1] + 1))
    col_low = max(COL_RANGE[0], row_count)
    col_high = min(COL_RANGE[1], cell_w)
    if col_low > col_high:
        return None
    col_count = int(_RNG.integers(col_low, col_high + 1))
    cell_h = int(_RNG.integers(CELL_H_RANGE[0], CELL_H_RANGE[1] + 1))

    bg, sep = _RNG.choice(PALETTE, size=2, replace=False).tolist()
    usable_colors = [color for color in PALETTE if color not in {bg, sep}]
    _RNG.shuffle(usable_colors)
    base_colors = usable_colors[:row_count]
    odd_color_pool = usable_colors[row_count:]

    branches = _balanced_branches(row_count)
    odd_cols = _RNG.permutation(col_count)[:row_count].tolist()

    grid_h = row_count * cell_h + row_count + 1
    grid_w = col_count * cell_w + col_count + 1
    grid = np.full((grid_h, grid_w), sep, dtype=int)

    color_branch_count = sum(branch == "color" for branch in branches)
    if color_branch_count > len(odd_color_pool):
        return None

    odd_color_idx = 0
    for row_idx in range(row_count):
        branch = branches[row_idx]
        base_color = int(base_colors[row_idx])
        odd_color = None
        if branch == "color":
            odd_color = int(odd_color_pool[odd_color_idx])
            odd_color_idx += 1

        base_cell, odd_cell = _make_row_patterns(cell_h, cell_w, bg, base_color, odd_color, branch)
        odd_col = int(odd_cols[row_idx])

        for col_idx in range(col_count):
            cell = odd_cell if col_idx == odd_col else base_cell
            r0 = 1 + row_idx * (cell_h + 1)
            c0 = 1 + col_idx * (cell_w + 1)
            grid[r0 : r0 + cell_h, c0 : c0 + cell_w] = cell

    try:
        out = _solve(grid)
    except AssertionError:
        return None

    if out.shape != (row_count * cell_h + row_count + 1, cell_w + 2):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC input")
