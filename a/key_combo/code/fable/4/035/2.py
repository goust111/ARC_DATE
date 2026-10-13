from __future__ import annotations

import numpy as np


GRAY = 5
_RNG = np.random.default_rng()
_GENERATION_ATTEMPTS = 500


def _segments(length: int, separator_indices: list[int]) -> list[slice] | None:
    """Return the equal open regions between one-cell separator lines."""
    if not separator_indices:
        return None
    starts = [0] + [index + 1 for index in separator_indices]
    stops = separator_indices + [length]
    widths = [stop - start for start, stop in zip(starts, stops)]
    if min(widths) <= 0 or len(set(widths)) != 1:
        return None
    return [slice(start, stop) for start, stop in zip(starts, stops)]


def _table_layout(grid: np.ndarray) -> tuple[int, list[slice], list[slice]]:
    """Locate the unique color forming all horizontal and vertical dividers."""
    candidates: list[tuple[int, list[slice], list[slice]]] = []
    for value in np.unique(grid):
        color = int(value)
        mask = grid == color
        horizontal = [int(r) for r in np.flatnonzero(np.all(mask, axis=1))]
        vertical = [int(c) for c in np.flatnonzero(np.all(mask, axis=0))]
        if len(horizontal) < 2 or len(vertical) < 2:
            continue
        row_cells = _segments(grid.shape[0], horizontal)
        col_cells = _segments(grid.shape[1], vertical)
        if row_cells is None or col_cells is None:
            continue
        row_size = row_cells[0].stop - row_cells[0].start
        col_size = col_cells[0].stop - col_cells[0].start
        if row_size != col_size:
            continue
        candidates.append((color, row_cells, col_cells))

    assert len(candidates) == 1, "expected one regular square-cell divider lattice"
    separator, row_cells, col_cells = candidates[0]
    assert len(row_cells) >= 3 and len(col_cells) >= 3
    return separator, row_cells, col_cells


def _header_mask(cell: np.ndarray, background: int) -> tuple[int, np.ndarray]:
    colors = set(map(int, np.unique(cell))) - {background}
    assert len(colors) == 1, "each header must contain one foreground color"
    color = colors.pop()
    mask = cell == color
    assert bool(mask.any()) and bool((~mask).any()), "header masks must be non-trivial"
    return color, mask


def main(grid: np.ndarray) -> np.ndarray:
    """Complete gray cells in a divider-separated table with row-header XOR column-header masks.

    Input structure:
    - One-cell separator lines divide the grid into equal square cells; the first row and
      first column are colored mask headers, the upper-left cell is empty, and interior
      cells use one shared foreground color except for whole gray unknown cells.

    Rule:
    - In every gray interior cell, compare the row-header and column-header masks in the
      cell-local row/column coordinates and paint the shared foreground color exactly
      where one, but not both, masks are present; preserve all other cells and dividers.

    Key insight:
    - The headers are operands of a pixelwise XOR multiplication table, not templates to
      copy or unite; overlapping header pixels cancel in the generated interior cell.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    assert max(g.shape) <= 30, "ARC grids must not exceed 30x30"

    separator, row_cells, col_cells = _table_layout(g)
    corner = g[row_cells[0], col_cells[0]]
    corner_colors = np.unique(corner)
    assert len(corner_colors) == 1, "upper-left cell must be empty"
    background = int(corner_colors[0])
    assert background not in {separator, GRAY}

    row_headers: list[np.ndarray] = []
    col_headers: list[np.ndarray] = []
    header_colors: list[int] = []
    for row_slice in row_cells[1:]:
        color, mask = _header_mask(g[row_slice, col_cells[0]], background)
        header_colors.append(color)
        row_headers.append(mask)
    for col_slice in col_cells[1:]:
        color, mask = _header_mask(g[row_cells[0], col_slice], background)
        header_colors.append(color)
        col_headers.append(mask)
    assert len(set(header_colors)) == len(header_colors), "header colors must be distinct"
    assert separator not in header_colors and GRAY not in header_colors

    unknowns: list[tuple[int, int]] = []
    interior_colors: set[int] = set()
    for i, row_slice in enumerate(row_cells[1:]):
        for j, col_slice in enumerate(col_cells[1:]):
            cell = g[row_slice, col_slice]
            if bool(np.all(cell == GRAY)):
                unknowns.append((i, j))
            else:
                assert not bool(np.any(cell == GRAY)), "gray must fill an entire unknown cell"
                interior_colors.update(set(map(int, np.unique(cell))) - {background})

    assert unknowns, "expected at least one gray unknown cell"
    n_internal = len(row_headers) * len(col_headers)
    assert n_internal - len(unknowns) > len(unknowns), "most interior cells must be known"
    assert len(interior_colors) == 1, "known cells must share one foreground color"
    foreground = interior_colors.pop()
    assert foreground not in {background, separator, GRAY} and foreground not in header_colors

    unknown_set = set(unknowns)
    out = g.copy()
    for i, row_slice in enumerate(row_cells[1:]):
        for j, col_slice in enumerate(col_cells[1:]):
            xor_mask = row_headers[i] ^ col_headers[j]
            expected = np.where(xor_mask, foreground, background)
            if (i, j) in unknown_set:
                out[row_slice, col_slice] = expected
            else:
                actual = g[row_slice, col_slice]
                assert np.array_equal(actual, expected), "known cell violates the header XOR rule"
    return out


def _random_connected_mask(size: int, rng: np.random.Generator) -> np.ndarray:
    """Grow a small, readable 4-connected mask in cell-local coordinates."""
    maximum = max(3, size * size // 3)
    target = int(rng.integers(2, maximum + 1))
    start = (int(rng.integers(size)), int(rng.integers(size)))
    occupied = {start}
    while len(occupied) < target:
        frontier: set[tuple[int, int]] = set()
        for r, c in occupied:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < size and 0 <= cc < size and (rr, cc) not in occupied:
                    frontier.add((rr, cc))
        if not frontier:
            break
        choices = sorted(frontier)
        occupied.add(choices[int(rng.integers(len(choices)))])
    mask = np.zeros((size, size), dtype=bool)
    for r, c in occupied:
        mask[r, c] = True
    return mask


def _choose_semantic_cells(
    row_masks: list[np.ndarray],
    col_masks: list[np.ndarray],
    unknown_count: int,
    rng: np.random.Generator,
) -> tuple[list[tuple[int, int]], tuple[int, int], tuple[int, int]] | None:
    """Choose unknowns plus known OR-decoy and XOR-witness cells."""
    pairs = [(i, j) for i in range(len(row_masks)) for j in range(len(col_masks))]
    overlap = [(i, j) for i, j in pairs if bool((row_masks[i] & col_masks[j]).any())]
    disjoint = [(i, j) for i, j in pairs if not bool((row_masks[i] & col_masks[j]).any())]
    options = []
    for index, first in enumerate(overlap):
        for second in overlap[index + 1 :]:
            if first[0] == second[0] or first[1] == second[1]:
                continue
            first_xor = row_masks[first[0]] ^ col_masks[first[1]]
            second_xor = row_masks[second[0]] ^ col_masks[second[1]]
            if np.array_equal(first_xor, second_xor):
                continue
            blocked = {first, second}
            witnesses = [pair for pair in overlap if pair not in blocked]
            decoys = [pair for pair in disjoint if pair not in blocked]
            for witness in witnesses:
                for decoy in decoys:
                    options.append(([first, second], witness, decoy))
    if not options:
        return None

    unknowns, witness, decoy = options[int(rng.integers(len(options)))]
    unknowns = list(unknowns)
    if unknown_count > 2:
        used_outputs = [row_masks[i] ^ col_masks[j] for i, j in unknowns]
        extras = []
        for pair in overlap:
            if pair in set(unknowns) | {witness, decoy}:
                continue
            result = row_masks[pair[0]] ^ col_masks[pair[1]]
            if all(not np.array_equal(result, old) for old in used_outputs):
                extras.append(pair)
        if not extras:
            return None
        extra = extras[int(rng.integers(len(extras)))]
        unknowns.append(extra)
    return unknowns, witness, decoy


def _cell_slices(count: int, size: int) -> list[slice]:
    return [slice(i * (size + 1), i * (size + 1) + size) for i in range(count)]


def _validate_generated(
    inp: np.ndarray,
    expected: np.ndarray,
    row_slices: list[slice],
    col_slices: list[slice],
    row_masks: list[np.ndarray],
    col_masks: list[np.ndarray],
    unknowns: list[tuple[int, int]],
    witness: tuple[int, int],
    decoy: tuple[int, int],
    background: int,
    foreground: int,
) -> bool:
    """Check the puzzle semantics independently of the solver's table parser."""
    if max(inp.shape) > 30 or inp.shape != expected.shape or np.array_equal(inp, expected):
        return False
    unknown_set = set(unknowns)
    if len(unknown_set) != len(unknowns):
        return False

    for i, row_mask in enumerate(row_masks):
        for j, col_mask in enumerate(col_masks):
            cell_in = inp[row_slices[i + 1], col_slices[j + 1]]
            cell_out = expected[row_slices[i + 1], col_slices[j + 1]]
            xor_mask = row_mask ^ col_mask
            direct = np.where(xor_mask, foreground, background)
            if not np.array_equal(cell_out, direct):
                return False
            if (i, j) in unknown_set:
                if not np.all(cell_in == GRAY):
                    return False
                # Every target contains cancellation, so OR is a provably wrong shortcut.
                if not bool((row_mask & col_mask).any()):
                    return False
                union = np.where(row_mask | col_mask, foreground, background)
                if np.array_equal(direct, union):
                    return False
            elif not np.array_equal(cell_in, direct):
                return False

    wi, wj = witness
    di, dj = decoy
    if witness in unknown_set or decoy in unknown_set:
        return False
    if not bool((row_masks[wi] & col_masks[wj]).any()):
        return False
    if bool((row_masks[di] & col_masks[dj]).any()):
        return False
    witness_xor = row_masks[wi] ^ col_masks[wj]
    witness_or = row_masks[wi] | col_masks[wj]
    decoy_xor = row_masks[di] ^ col_masks[dj]
    decoy_or = row_masks[di] | col_masks[dj]
    if np.array_equal(witness_xor, witness_or) or not np.array_equal(decoy_xor, decoy_or):
        return False

    target_outputs = [row_masks[i] ^ col_masks[j] for i, j in unknowns]
    for i in range(len(target_outputs)):
        for j in range(i + 1, len(target_outputs)):
            if np.array_equal(target_outputs[i], target_outputs[j]):
                return False
    try:
        return np.array_equal(main(inp), expected)
    except (AssertionError, ValueError, IndexError):
        return False


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    layouts = ((2, 3), (3, 2), (3, 3), (2, 4), (4, 2))
    n_rows, n_cols = layouts[int(rng.integers(len(layouts)))]
    cell_size = int(rng.integers(3, 6))
    n_internal = n_rows * n_cols
    unknown_count = 2 if n_internal == 6 else int(rng.integers(2, 4))

    masks: list[np.ndarray] = []
    for _ in range(n_rows + n_cols):
        for _retry in range(80):
            candidate = _random_connected_mask(cell_size, rng)
            if all(not np.array_equal(candidate, old) for old in masks):
                masks.append(candidate)
                break
        else:
            return None
    row_masks = masks[:n_rows]
    col_masks = masks[n_rows:]

    for row_mask in row_masks:
        for col_mask in col_masks:
            xor_count = int(np.count_nonzero(row_mask ^ col_mask))
            if xor_count == 0 or xor_count == cell_size * cell_size:
                return None

    semantic_cells = _choose_semantic_cells(row_masks, col_masks, unknown_count, rng)
    if semantic_cells is None:
        return None
    unknowns, witness, decoy = semantic_cells

    palette = [color for color in range(10) if color != GRAY]
    rng.shuffle(palette)
    needed = 3 + n_rows + n_cols
    if needed > len(palette):
        return None
    background, separator, foreground = palette[:3]
    header_colors = palette[3:needed]
    row_colors = header_colors[:n_rows]
    col_colors = header_colors[n_rows:]

    height = (n_rows + 1) * cell_size + n_rows
    width = (n_cols + 1) * cell_size + n_cols
    row_slices = _cell_slices(n_rows + 1, cell_size)
    col_slices = _cell_slices(n_cols + 1, cell_size)
    solved = np.full((height, width), background, dtype=int)
    for r in [sl.stop for sl in row_slices[:-1]]:
        solved[r, :] = separator
    for c in [sl.stop for sl in col_slices[:-1]]:
        solved[:, c] = separator

    for i, (mask, color) in enumerate(zip(row_masks, row_colors)):
        cell = solved[row_slices[i + 1], col_slices[0]]
        cell[mask] = color
    for j, (mask, color) in enumerate(zip(col_masks, col_colors)):
        cell = solved[row_slices[0], col_slices[j + 1]]
        cell[mask] = color
    for i, row_mask in enumerate(row_masks):
        for j, col_mask in enumerate(col_masks):
            cell = solved[row_slices[i + 1], col_slices[j + 1]]
            cell[row_mask ^ col_mask] = foreground

    inp = solved.copy()
    for i, j in unknowns:
        inp[row_slices[i + 1], col_slices[j + 1]] = GRAY
    if not _validate_generated(
        inp,
        solved,
        row_slices,
        col_slices,
        row_masks,
        col_masks,
        unknowns,
        witness,
        decoy,
        background,
        foreground,
    ):
        return None
    return inp


def generate_input() -> np.ndarray:
    """Generate a varied, semantically witnessed XOR-table completion puzzle."""
    for _ in range(_GENERATION_ATTEMPTS):
        grid = _generate_attempt(_RNG)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {_GENERATION_ATTEMPTS} attempts")
