from __future__ import annotations

from collections import Counter

import numpy as np


BG = 0
GRID = 5
PALETTE = tuple(color for color in range(10) if color not in (BG, GRID))
MAX_GENERATION_ATTEMPTS = 400
RNG = np.random.default_rng()


def _bands(length: int, separators: list[int]) -> list[tuple[int, int]]:
    separators = sorted(int(x) for x in separators)
    bands: list[tuple[int, int]] = []
    bounds = [-1] + separators + [length]
    for left, right in zip(bounds, bounds[1:]):
        if right - left > 1:
            bands.append((int(left + 1), int(right - 1)))
    return bands


def _split_lattice(grid: np.ndarray) -> tuple[int, list[list[np.ndarray]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)

    row_colors = {int(g[r, 0]) for r in range(h) if np.all(g[r, :] == g[r, 0])}
    col_colors = {int(g[0, c]) for c in range(w) if np.all(g[:, c] == g[0, c])}
    sep_colors = row_colors & col_colors
    assert len(sep_colors) == 1, "expected exactly one separator color"
    sep = int(next(iter(sep_colors)))

    sep_rows = [r for r in range(h) if np.all(g[r, :] == sep)]
    sep_cols = [c for c in range(w) if np.all(g[:, c] == sep)]
    row_bands = _bands(h, sep_rows)
    col_bands = _bands(w, sep_cols)
    assert row_bands and col_bands, "lattice bands must exist"

    heights = {int(r1 - r0 + 1) for r0, r1 in row_bands}
    widths = {int(c1 - c0 + 1) for c0, c1 in col_bands}
    assert len(heights) == 1 and len(widths) == 1, "cell interiors must share one size"

    cells: list[list[np.ndarray]] = []
    for r0, r1 in row_bands:
        row_cells = []
        for c0, c1 in col_bands:
            row_cells.append(g[r0 : r1 + 1, c0 : c1 + 1].copy())
        cells.append(row_cells)
    return sep, cells


def _is_marker_cell(cell: np.ndarray, sep: int) -> bool:
    return bool(np.all(cell == cell[0, 0]) and int(cell[0, 0]) != sep)


def _find_marker_column(cells: list[list[np.ndarray]], sep: int) -> int:
    candidates = []
    for ci in range(len(cells[0])):
        if all(_is_marker_cell(cells[ri][ci], sep) for ri in range(len(cells))):
            candidates.append(int(ci))
    assert len(candidates) == 1, "expected exactly one marker column"
    return int(candidates[0])


def _find_odd_pattern(row_cells: list[np.ndarray]) -> np.ndarray:
    signatures = [cell.tobytes() for cell in row_cells]
    counts = Counter(signatures)
    assert len(counts) == 2, "each row must contain one odd pattern and one repeated pattern"
    odd_sigs = [sig for sig, count in counts.items() if int(count) == 1]
    repeated_counts = sorted(int(count) for count in counts.values())
    assert len(odd_sigs) == 1 and repeated_counts == [1, len(row_cells) - 1]
    odd_index = signatures.index(odd_sigs[0])
    return row_cells[int(odd_index)].copy()


def _primary_color(cell: np.ndarray, bg: int) -> int:
    vals = cell[cell != bg]
    assert vals.size > 0, "pattern must contain a visible non-background color"
    colors, counts = np.unique(vals, return_counts=True)
    best = int(np.argmax(counts))
    assert int(np.sum(counts == counts[best])) == 1, "pattern main color must be unique"
    return int(colors[best])


def _assemble_column(patterns: list[np.ndarray], sep: int) -> np.ndarray:
    assert patterns, "need at least one pattern"
    cell_h, cell_w = map(int, patterns[0].shape)
    for pat in patterns[1:]:
        assert pat.shape == (cell_h, cell_w), "all output cells must share one size"

    out = np.full((len(patterns) * cell_h + len(patterns) + 1, cell_w + 2), sep, dtype=int)
    for ri, pat in enumerate(patterns):
        r0 = 1 + ri * (cell_h + 1)
        out[r0 : r0 + cell_h, 1 : 1 + cell_w] = pat
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find the odd cell in each board row and reorder those extracted patterns by the left color-band order.
    """
    sep, cells = _split_lattice(grid)
    row_count = len(cells)
    col_count = len(cells[0])
    assert col_count >= 4, "need one marker column plus at least three board columns"

    marker_col = _find_marker_column(cells, sep)
    marker_colors = [int(cells[ri][marker_col][0, 0]) for ri in range(row_count)]

    board_cells = [
        [cells[ri][ci] for ci in range(col_count) if ci != marker_col] for ri in range(row_count)
    ]
    odd_patterns = [_find_odd_pattern(row) for row in board_cells]

    color_to_pattern: dict[int, np.ndarray] = {}
    for pat in odd_patterns:
        color = _primary_color(pat, BG)
        assert color not in color_to_pattern, "odd-pattern main colors must be unique"
        color_to_pattern[color] = pat

    assert set(marker_colors) == set(color_to_pattern), "marker colors must match odd-pattern main colors"
    ordered = [color_to_pattern[color] for color in marker_colors]
    return _assemble_column(ordered, sep)


def _rotate(mask: np.ndarray, turns: int) -> np.ndarray:
    return np.rot90(mask, int(turns))


def _candidate_masks(size: int) -> list[np.ndarray]:
    masks: list[np.ndarray] = []

    plus = np.zeros((size, size), dtype=bool)
    mid = size // 2
    plus[mid, :] = True
    plus[:, mid] = True
    masks.append(plus)

    cross = np.eye(size, dtype=bool)
    cross |= np.fliplr(np.eye(size, dtype=bool))
    masks.append(cross)

    frame = np.ones((size, size), dtype=bool)
    frame[1:-1, 1:-1] = False
    masks.append(frame)

    lshape = np.zeros((size, size), dtype=bool)
    lshape[0, :] = True
    lshape[:, 0] = True
    for turns in range(4):
        masks.append(_rotate(lshape, turns))

    tshape = np.zeros((size, size), dtype=bool)
    tshape[0, :] = True
    tshape[:, mid] = True
    for turns in range(4):
        masks.append(_rotate(tshape, turns))

    ushape = np.zeros((size, size), dtype=bool)
    ushape[:, 0] = True
    ushape[:, -1] = True
    ushape[-1, :] = True
    for turns in range(4):
        masks.append(_rotate(ushape, turns))

    unique: list[np.ndarray] = []
    seen: set[bytes] = set()
    for mask in masks:
        key = mask.tobytes()
        if key not in seen and np.any(mask) and not np.all(mask):
            seen.add(key)
            unique.append(mask)
    assert len(unique) >= 8
    return unique


def _render_pattern(size: int, mask: np.ndarray, color: int) -> np.ndarray:
    cell = np.full((size, size), BG, dtype=int)
    cell[np.asarray(mask, dtype=bool)] = int(color)
    assert np.any(cell == BG), "cell needs visible background"
    assert np.any(cell != BG), "cell needs visible foreground"
    return cell


def _choose_different_mask(mask_pool: list[np.ndarray], mask: np.ndarray) -> np.ndarray:
    choices = [candidate for candidate in mask_pool if not np.array_equal(candidate, mask)]
    assert choices, "mask pool must contain multiple distinct masks"
    return choices[int(RNG.integers(len(choices)))]


def _build_input(
    marker_colors: list[int], rows: list[list[np.ndarray]], size: int, sep: int
) -> np.ndarray:
    row_count = len(rows)
    board_cols = len(rows[0])
    total_cols = board_cols + 1
    out_h = row_count * size + row_count + 1
    out_w = total_cols * size + total_cols + 1
    grid = np.full((out_h, out_w), sep, dtype=int)

    for ri in range(row_count):
        r0 = 1 + ri * (size + 1)
        grid[r0 : r0 + size, 1 : 1 + size] = int(marker_colors[ri])
        for ci, cell in enumerate(rows[ri], start=1):
            c0 = 1 + ci * (size + 1)
            grid[r0 : r0 + size, c0 : c0 + size] = cell
    return grid


def _non_identity_permutation(length: int) -> np.ndarray:
    while True:
        perm = RNG.permutation(length)
        if not np.array_equal(perm, np.arange(length)):
            return perm


def _gen_attempt() -> np.ndarray | None:
    row_count = int(RNG.integers(3, 6))
    board_cols = int(RNG.integers(3, 5))
    size = int(RNG.integers(4, 6))
    mask_pool = _candidate_masks(size)

    odd_colors = list(map(int, RNG.choice(PALETTE, size=row_count, replace=False)))
    other_colors = [color for color in PALETTE if color not in odd_colors]

    row_cells: list[list[np.ndarray]] = []
    odd_patterns: list[np.ndarray] = []
    branches: list[str] = []

    for odd_color in odd_colors:
        branch = ("shape", "color", "both")[int(RNG.integers(3))]
        odd_mask = mask_pool[int(RNG.integers(len(mask_pool)))]

        if branch == "shape":
            base_mask = _choose_different_mask(mask_pool, odd_mask)
            base_color = int(odd_color)
        elif branch == "color":
            base_mask = odd_mask
            base_color = int(other_colors[int(RNG.integers(len(other_colors)))])
        else:
            base_mask = _choose_different_mask(mask_pool, odd_mask)
            base_color = int(other_colors[int(RNG.integers(len(other_colors)))])

        odd_pattern = _render_pattern(size, odd_mask, int(odd_color))
        base_pattern = _render_pattern(size, base_mask, int(base_color))
        if np.array_equal(odd_pattern, base_pattern):
            return None

        odd_index = int(RNG.integers(board_cols))
        row = []
        for ci in range(board_cols):
            row.append(odd_pattern.copy() if ci == odd_index else base_pattern.copy())
        row_cells.append(row)
        odd_patterns.append(odd_pattern)
        branches.append(branch)

    # Avoid the whole sample collapsing to a single branch for long stretches.
    if len(set(branches)) == 1 and row_count >= 3:
        return None

    perm = _non_identity_permutation(row_count)
    marker_colors = [odd_colors[int(idx)] for idx in perm]
    grid = _build_input(marker_colors, row_cells, size, GRID)

    expected = _assemble_column([odd_patterns[int(idx)] for idx in perm], GRID)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
