from __future__ import annotations

from collections import Counter
import random

import numpy as np


BLACK = 0
COLORS = tuple(range(10))
MAX_GENERATION_ATTEMPTS = 100
_SEEN_OUTPUT_SIGNATURES: set[tuple[tuple[int, int], bytes]] = set()


def _bands(size: int, separator_indices: list[int]) -> list[tuple[int, int]]:
    """Return inclusive non-separator intervals along one grid axis."""
    bands: list[tuple[int, int]] = []
    start = 0
    for cut in separator_indices:
        if start < cut:
            bands.append((start, cut - 1))
        start = cut + 1
    if start < size:
        bands.append((start, size - 1))
    return bands


def _lattice(
    grid: np.ndarray,
) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]]]:
    """Find the unique equal-cell square lattice made by full separator bars."""
    h, w = map(int, grid.shape)
    full_row_colors = {
        int(grid[r, 0]) for r in range(h) if bool(np.all(grid[r, :] == grid[r, 0]))
    }
    full_col_colors = {
        int(grid[0, c]) for c in range(w) if bool(np.all(grid[:, c] == grid[0, c]))
    }

    candidates = []
    for separator in full_row_colors & full_col_colors:
        separator_rows = [r for r in range(h) if bool(np.all(grid[r, :] == separator))]
        separator_cols = [c for c in range(w) if bool(np.all(grid[:, c] == separator))]
        row_bands = _bands(h, separator_rows)
        col_bands = _bands(w, separator_cols)
        if len(row_bands) != len(col_bands) or len(row_bands) < 2:
            continue
        if len({r1 - r0 + 1 for r0, r1 in row_bands}) != 1:
            continue
        if len({c1 - c0 + 1 for c0, c1 in col_bands}) != 1:
            continue
        candidates.append((int(separator), row_bands, col_bands))

    assert len(candidates) == 1, "expected one uniquely identifiable separator lattice"
    return candidates[0]


def _headers(
    grid: np.ndarray,
    row_bands: list[tuple[int, int]],
    col_bands: list[tuple[int, int]],
) -> tuple[list[int], list[int], list[tuple[int, int, int]]]:
    """Extract row and column header colors from the solid-cell permutation."""
    n = len(row_bands)
    solid_cells: list[tuple[int, int, int]] = []
    for ri, (r0, r1) in enumerate(row_bands):
        for ci, (c0, c1) in enumerate(col_bands):
            cell = grid[r0 : r1 + 1, c0 : c1 + 1]
            if bool(np.all(cell == cell[0, 0])):
                solid_cells.append((ri, ci, int(cell[0, 0])))

    assert len(solid_cells) == n, "expected exactly one solid cell per lattice row and column"
    assert len({ri for ri, _ci, _color in solid_cells}) == n, "solid cells must cover every row once"
    assert len({ci for _ri, ci, _color in solid_cells}) == n, "solid cells must cover every column once"

    row_headers = [-1] * n
    col_headers = [-1] * n
    for ri, ci, color in solid_cells:
        row_headers[ri] = color
        col_headers[ci] = color
    return row_headers, col_headers, solid_cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract hidden row and column headers from a lattice and output their color-equality table.

    Input structure:
    - Full rows and columns of one separator color divide the input into an n by n
      array of equal-sized cells.
    - Exactly one cell in every lattice row and every lattice column is completely
      solid; these solid cells form a permutation and their colors may repeat.
    - Every other cell is a multicolored distractor pattern.

    Rule:
    - The solid cell in lattice row i supplies row header i, and the solid cell in
      lattice column j supplies column header j.
    - Return an n by n grid whose (i,j) cell has that shared header color when the
      two headers match, and is black otherwise.

    Key insight:
    - Solid cells encode two ordered header sequences, not merely marked output
      positions; repeated colors require comparing every row header with every
      column header and create additional matches away from the permutation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    _separator, row_bands, col_bands = _lattice(g)
    row_headers, col_headers, _solid_cells = _headers(g, row_bands, col_bands)

    rows = np.asarray(row_headers, dtype=int)[:, None]
    cols = np.asarray(col_headers, dtype=int)[None, :]
    return np.where(rows == cols, rows, BLACK).astype(int)


def _sample_labels(n: int, separator: int) -> list[int]:
    """Choose visible, non-black labels with both repetition and contrast."""
    available = [color for color in range(1, 10) if color != separator]
    distinct = random.randint(2, min(3, n - 1))
    palette = random.sample(available, distinct)
    labels = palette.copy()
    labels.extend(random.choice(palette) for _ in range(n - distinct))
    random.shuffle(labels)
    assert 1 < len(set(labels)) < n
    return labels


def _noise_tile(
    height: int,
    width: int,
    palette: list[int],
    row_index: int,
    col_index: int,
) -> np.ndarray:
    """Make a clearly multicolored pattern with cell-specific phase and flips."""
    local_count = random.randint(3, min(5, len(palette)))
    local = random.sample(palette, local_count)
    rr, cc = np.indices((height, width))
    row_step = random.choice([1, 2])
    col_step = random.choice([1, 2])
    phase = (row_index + 2 * col_index + random.randrange(local_count)) % local_count
    tile = np.asarray(local, dtype=int)[(row_step * rr + col_step * cc + phase) % local_count]
    if random.choice([False, True]):
        tile = np.flipud(tile)
    if random.choice([False, True]):
        tile = np.fliplr(tile)

    # A few substitutions prevent the distractors from encoding another global table.
    for _ in range(random.randint(1, max(1, height * width // 5))):
        r = random.randrange(height)
        c = random.randrange(width)
        tile[r, c] = random.choice(local)

    if len(np.unique(tile)) < 3:
        for index, color in enumerate(local[:3]):
            tile.flat[index] = color
    assert len(np.unique(tile)) >= 3
    return tile.copy()


def _expected(labels: list[int], permutation: list[int]) -> tuple[np.ndarray, list[int]]:
    n = len(labels)
    column_headers = [-1] * n
    for row, col in enumerate(permutation):
        column_headers[col] = labels[row]
    expected = np.zeros((n, n), dtype=int)
    for row in range(n):
        for col in range(n):
            if labels[row] == column_headers[col]:
                expected[row, col] = labels[row]
    return expected, column_headers


def _validate_candidate(
    grid: np.ndarray,
    labels: list[int],
    permutation: list[int],
    row_bands: list[tuple[int, int]],
    col_bands: list[tuple[int, int]],
) -> bool:
    """Check the stated semantics and two anti-shortcut counterfactuals."""
    n = len(labels)
    expected, column_headers = _expected(labels, permutation)
    try:
        solved = main(grid)
    except (AssertionError, IndexError, ValueError):
        return False
    if not np.array_equal(solved, expected):
        return False

    # Repeated labels must add matches beyond the n encoded permutation cells,
    # while label contrast must leave at least one black mismatch.
    match_count = sum(a == b for a in labels for b in column_headers)
    if not (n < match_count < n * n):
        return False
    if any(expected[row, permutation[row]] != labels[row] for row in range(n)):
        return False
    if not any(
        expected[row, col] != BLACK and col != permutation[row]
        for row in range(n)
        for col in range(n)
    ):
        return False

    # Independently check that every non-header cell really is multicolored.
    for row, (r0, r1) in enumerate(row_bands):
        for col, (c0, c1) in enumerate(col_bands):
            cell = grid[r0 : r1 + 1, c0 : c1 + 1]
            unique = len(np.unique(cell))
            if col == permutation[row]:
                if unique != 1 or int(cell[0, 0]) != labels[row]:
                    return False
            elif unique < 3:
                return False

    # Counterexample to reading distractor samples: changing a distractor pixel
    # must leave the output unchanged.
    distractor_row = random.randrange(n)
    distractor_cols = [c for c in range(n) if c != permutation[distractor_row]]
    distractor_col = random.choice(distractor_cols)
    dr0, dr1 = row_bands[distractor_row]
    dc0, dc1 = col_bands[distractor_col]
    altered_noise = grid.copy()
    cell_values = list(map(int, np.unique(altered_noise[dr0 : dr1 + 1, dc0 : dc1 + 1])))
    old = int(altered_noise[dr0, dc0])
    altered_noise[dr0, dc0] = next(color for color in cell_values if color != old)
    if not np.array_equal(main(altered_noise), expected):
        return False

    # Counterexample to downsampling solid positions: recoloring one complete
    # header cell changes multiple equality-table entries, not just its position.
    changed_row = random.randrange(n)
    old_label = labels[changed_row]
    new_label = random.choice([color for color in set(labels) if color != old_label])
    changed_grid = grid.copy()
    hr0, hr1 = row_bands[changed_row]
    hc0, hc1 = col_bands[permutation[changed_row]]
    changed_grid[hr0 : hr1 + 1, hc0 : hc1 + 1] = new_label
    changed_labels = labels.copy()
    changed_labels[changed_row] = new_label
    changed_expected, _ = _expected(changed_labels, permutation)
    try:
        changed_solved = main(changed_grid)
    except (AssertionError, IndexError, ValueError):
        return False
    difference = changed_expected != expected
    if not np.array_equal(changed_solved, changed_expected) or int(difference.sum()) <= 1:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    n = random.choice([3, 4, 5])
    separator_width = random.choice([1, 2])
    max_cell_size = min(6, (30 - (n - 1) * separator_width) // n)
    cell_height = random.randint(3, max_cell_size)
    cell_width = random.randint(3, max_cell_size)

    separator = random.randint(1, 9)
    labels = _sample_labels(n, separator)
    permutation = list(range(n))
    while permutation == list(range(n)):
        random.shuffle(permutation)

    height = n * cell_height + (n - 1) * separator_width
    width = n * cell_width + (n - 1) * separator_width
    grid = np.full((height, width), separator, dtype=int)
    row_bands = [
        (row * (cell_height + separator_width), row * (cell_height + separator_width) + cell_height - 1)
        for row in range(n)
    ]
    col_bands = [
        (col * (cell_width + separator_width), col * (cell_width + separator_width) + cell_width - 1)
        for col in range(n)
    ]

    noise_palette = list(dict.fromkeys([BLACK, *labels]))
    extras = [color for color in COLORS if color != separator and color not in noise_palette]
    random.shuffle(extras)
    noise_palette.extend(extras[: max(0, 5 - len(noise_palette))])

    for row, (r0, r1) in enumerate(row_bands):
        for col, (c0, c1) in enumerate(col_bands):
            if col == permutation[row]:
                tile = np.full((cell_height, cell_width), labels[row], dtype=int)
            else:
                tile = _noise_tile(cell_height, cell_width, noise_palette, row, col)
            grid[r0 : r1 + 1, c0 : c1 + 1] = tile

    if max(grid.shape) > 30:
        return None
    if not _validate_candidate(grid, labels, permutation, row_bands, col_bands):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is None:
            continue
        output = main(grid)
        signature = (tuple(map(int, output.shape)), output.tobytes())
        if signature in _SEEN_OUTPUT_SIGNATURES:
            continue
        _SEEN_OUTPUT_SIGNATURES.add(signature)
        return grid
    raise RuntimeError("failed to generate a valid header-table lattice")
