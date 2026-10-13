from __future__ import annotations

from collections import Counter
from itertools import permutations
import random

import numpy as np


def _constant_lines(grid: np.ndarray, axis: int) -> dict[int, list[int]]:
    """Map each color to the indices of its full constant rows or columns."""
    lines: dict[int, list[int]] = {}
    limit = grid.shape[axis]
    for index in range(limit):
        line = grid[index, :] if axis == 0 else grid[:, index]
        color = int(line[0])
        if np.all(line == color):
            lines.setdefault(color, []).append(index)
    return lines


def _content_bands(separators: list[int], limit: int) -> list[tuple[int, int]]:
    """Return half-open content bands between one-cell separator lines."""
    bands: list[tuple[int, int]] = []
    start = 0
    for stop in [*separators, limit]:
        if stop <= start:
            return []
        bands.append((start, stop))
        start = stop + 1
    if separators and separators[-1] == limit - 1:
        return []
    return bands


def _parse_lattice(grid: np.ndarray) -> tuple[int, int, list[list[np.ndarray]]]:
    """Find the common grid-line color and split an equal-cell 2-D lattice."""
    row_lines = _constant_lines(grid, axis=0)
    col_lines = _constant_lines(grid, axis=1)
    candidates = []

    for color in set(row_lines) & set(col_lines):
        rows = row_lines[color]
        cols = col_lines[color]
        row_bands = _content_bands(rows, grid.shape[0])
        col_bands = _content_bands(cols, grid.shape[1])
        if len(row_bands) < 2 or len(col_bands) < 3:
            continue
        heights = {b - a for a, b in row_bands}
        widths = {b - a for a, b in col_bands}
        if len(heights) != 1 or len(widths) != 1:
            continue
        candidates.append((int(color), row_bands, col_bands))

    assert len(candidates) == 1, "expected one uniformly divided lattice"
    separator, row_bands, col_bands = candidates[0]
    blocks = [
        [grid[r0:r1, c0:c1].copy() for c0, c1 in col_bands]
        for r0, r1 in row_bands
    ]
    return separator, len(col_bands), blocks


def _block_key(block: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(map(int, block.shape)), np.ascontiguousarray(block).tobytes()


def _row_outliers(block_rows: list[list[np.ndarray]], column_count: int) -> list[np.ndarray]:
    """Select the sole once-occurring block from every lattice row."""
    outliers: list[np.ndarray] = []
    for row in block_rows:
        keys = [_block_key(block) for block in row]
        counts = Counter(keys)
        singleton = [key for key, count in counts.items() if count == 1]
        repeated = [key for key, count in counts.items() if count == column_count - 1]
        assert len(singleton) == 1 and len(repeated) == 1 and len(counts) == 2, (
            "each row must contain one odd cell and otherwise identical copies"
        )
        outliers.append(row[keys.index(singleton[0])].copy())
    return outliers


def _background(blocks: list[np.ndarray]) -> int:
    """Find the unique dominant cell-interior color."""
    counts = Counter(int(value) for block in blocks for value in block.ravel())
    ranked = counts.most_common()
    assert ranked and (len(ranked) == 1 or ranked[0][1] > ranked[1][1])
    return int(ranked[0][0])


def _edge_graph(blocks: list[np.ndarray], background: int) -> np.ndarray:
    """Directed matches in each block's unchanged local column coordinates."""
    n = len(blocks)
    graph = np.zeros((n, n), dtype=bool)
    tops = [block[0, :] != background for block in blocks]
    bottoms = [block[-1, :] != background for block in blocks]
    for above in range(n):
        for below in range(n):
            if above != below:
                graph[above, below] = bool(np.any(bottoms[above] & tops[below]))
    return graph


def _unique_chain(graph: np.ndarray) -> tuple[int, ...]:
    """Return the only permutation covering all vertices through directed matches."""
    n = graph.shape[0]
    valid = []
    for order in permutations(range(n)):
        if all(graph[order[i], order[i + 1]] for i in range(n - 1)):
            valid.append(order)
            if len(valid) > 1:
                break
    assert len(valid) == 1, "candidate blocks must admit exactly one complete chain"
    return valid[0]


def _render_vertical(blocks: list[np.ndarray], separator: int) -> np.ndarray:
    h, w = blocks[0].shape
    out = np.full((len(blocks) * h + len(blocks) - 1, w), separator, dtype=int)
    for index, block in enumerate(blocks):
        r0 = index * (h + 1)
        out[r0:r0 + h, :] = block
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract each row's odd cell and reorder the odd cells into one vertical interlocking chain.

    Input structure:
    - One-cell solid grid lines divide the array into an R by C lattice of equal blocks.
    - Every lattice row contains one odd pattern; its other C-1 blocks are exact copies.
    - Foreground pixels occur on the top and bottom local edges of every odd block.

    Rule:
    - Select the once-occurring block independently within each lattice row.
    - Block A may precede block B exactly when A's bottom-edge foreground mask and
      B's top-edge foreground mask overlap at one or more unchanged local columns.
    - Use the unique complete top-to-bottom chain and stack its blocks vertically,
      preserving a one-cell grid line between adjacent blocks.

    Key insight:
    - Row-wise rarity chooses the material, but edge interlocks override the source
      row order; comparisons stay in each block's original local coordinate frame.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a two-dimensional grid"
    separator, column_count, block_rows = _parse_lattice(g)
    candidates = _row_outliers(block_rows, column_count)
    assert len({block.shape for block in candidates}) == 1
    background = _background([block for row in block_rows for block in row])
    assert background != separator
    graph = _edge_graph(candidates, background)
    order = _unique_chain(graph)
    return _render_vertical([candidates[index] for index in order], separator)


# Generator constraints: both the input and the vertically stacked output stay <= 30.
ROW_COUNT_OPTIONS = (3, 4, 5)
COLUMN_COUNT_OPTIONS = (3, 4)
CELL_HEIGHT_OPTIONS = (4, 5)
COLOR_POOL = tuple(range(10))
GENERATION_ATTEMPTS = 300


def _make_pattern(
    height: int,
    width: int,
    top_col: int,
    bottom_col: int,
    background: int,
    foreground: int,
) -> np.ndarray:
    """Draw a sparse connected motif with exactly one pixel on each horizontal edge."""
    # Keeping the spine between the two edge columns makes the motif sparse,
    # so the background remains visibly and numerically dominant.
    pivot = random.randint(min(top_col, bottom_col), max(top_col, bottom_col))
    mask = np.zeros((height, width), dtype=bool)
    mask[0, top_col] = True
    mask[1, min(top_col, pivot):max(top_col, pivot) + 1] = True
    mask[1:height - 1, pivot] = True
    mask[height - 2, min(bottom_col, pivot):max(bottom_col, pivot) + 1] = True
    mask[-1, bottom_col] = True

    # A small connected bump varies interiors without changing connector signatures.
    if height >= 5 and random.random() < 0.7:
        row = random.randint(2, height - 3)
        options = [col for col in (pivot - 1, pivot + 1) if 0 <= col < width]
        if options:
            mask[row, random.choice(options)] = True

    block = np.full((height, width), background, dtype=int)
    block[mask] = foreground
    return block


def _different_pattern(
    height: int,
    width: int,
    background: int,
    foreground: int,
    forbidden: set[tuple[tuple[int, int], bytes]],
) -> np.ndarray | None:
    """Sample a visible top/bottom-edge distractor not equal to any forbidden block."""
    for _ in range(40):
        block = _make_pattern(
            height,
            width,
            random.randrange(width),
            random.randrange(width),
            background,
            foreground,
        )
        if _block_key(block) not in forbidden:
            return block
    return None


def _independent_outliers(
    grid: np.ndarray,
    row_count: int,
    column_count: int,
    height: int,
    width: int,
) -> tuple[list[np.ndarray], list[int]]:
    """Extract row oddities directly from known generator geometry for validation."""
    blocks: list[np.ndarray] = []
    odd_columns: list[int] = []
    for row in range(row_count):
        row_blocks = []
        for col in range(column_count):
            r0 = row * (height + 1)
            c0 = col * (width + 1)
            row_blocks.append(grid[r0:r0 + height, c0:c0 + width])
        counts = Counter(_block_key(block) for block in row_blocks)
        singletons = [key for key, count in counts.items() if count == 1]
        if len(singletons) != 1 or sorted(counts.values()) != [1, column_count - 1]:
            return [], []
        odd_col = next(i for i, block in enumerate(row_blocks) if _block_key(block) == singletons[0])
        blocks.append(row_blocks[odd_col].copy())
        odd_columns.append(odd_col)
    return blocks, odd_columns


def _validate_generated(
    grid: np.ndarray,
    expected: np.ndarray,
    odd_by_row: list[np.ndarray],
    chain_rows: list[int],
    intended_odd_columns: list[int],
    novel_decoy: np.ndarray,
    background: int,
    separator: int,
    row_count: int,
    column_count: int,
    height: int,
    width: int,
) -> bool:
    """Check the stated semantics and anti-shortcut conditions, not only solver closure."""
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return False
    if chain_rows == list(range(row_count)) or len(set(intended_odd_columns)) < 2:
        return False

    sep_rows = list(range(height, grid.shape[0], height + 1))
    sep_cols = list(range(width, grid.shape[1], width + 1))
    if not all(np.all(grid[r, :] == separator) for r in sep_rows):
        return False
    if not all(np.all(grid[:, c] == separator) for c in sep_cols):
        return False

    extracted, found_columns = _independent_outliers(
        grid, row_count, column_count, height, width
    )
    if found_columns != intended_odd_columns:
        return False
    if any(not np.array_equal(a, b) for a, b in zip(extracted, odd_by_row)):
        return False

    graph = _edge_graph(extracted, background)
    expected_graph = np.zeros_like(graph)
    for above, below in zip(chain_rows, chain_rows[1:]):
        expected_graph[above, below] = True
    if not np.array_equal(graph, expected_graph):
        return False
    if _unique_chain(graph) != tuple(chain_rows):
        return False

    # Row order must be observably wrong, and row-wise filtering must remove a
    # genuine extra pattern that cannot be discarded by global de-duplication.
    row_order_output = _render_vertical(extracted, separator)
    if np.array_equal(row_order_output, expected):
        return False
    candidate_keys = {_block_key(block) for block in odd_by_row}
    if _block_key(novel_decoy) in candidate_keys:
        return False

    # At least one true candidate is a majority pattern elsewhere, defeating
    # global rarity as a replacement for the specified row-local selection.
    all_cells = [
        grid[
            row * (height + 1):row * (height + 1) + height,
            col * (width + 1):col * (width + 1) + width,
        ]
        for row in range(row_count)
        for col in range(column_count)
    ]
    global_counts = Counter(_block_key(block) for block in all_cells)
    if not any(global_counts[_block_key(block)] > 1 for block in odd_by_row):
        return False

    try:
        solved = main(grid)
    except (AssertionError, ValueError):
        return False
    return np.array_equal(solved, expected)


def _gen_attempt() -> np.ndarray | None:
    row_count = random.choice(ROW_COUNT_OPTIONS)
    column_count = random.choice(COLUMN_COUNT_OPTIONS)
    height_choices = [
        h for h in CELL_HEIGHT_OPTIONS if row_count * h + row_count - 1 <= 30
    ]
    height = random.choice(height_choices)
    max_width = (30 - (column_count - 1)) // column_count
    width_choices = list(range(row_count + 1, min(8, max_width) + 1))
    if not width_choices:
        return None
    width = random.choice(width_choices)

    background, foreground, separator = random.sample(COLOR_POOL, 3)
    edge_codes = random.sample(range(width), row_count + 1)
    chain_blocks = [
        _make_pattern(
            height,
            width,
            edge_codes[index],
            edge_codes[index + 1],
            background,
            foreground,
        )
        for index in range(row_count)
    ]
    if len({_block_key(block) for block in chain_blocks}) != row_count:
        return None

    chain_rows = list(range(row_count))
    while chain_rows == list(range(row_count)):
        random.shuffle(chain_rows)
    odd_by_row: list[np.ndarray | None] = [None] * row_count
    for chain_position, source_row in enumerate(chain_rows):
        odd_by_row[source_row] = chain_blocks[chain_position]
    odd_by_row = [block for block in odd_by_row if block is not None]

    forbidden = {_block_key(block) for block in odd_by_row}
    novel_decoy = _different_pattern(
        height, width, background, foreground, forbidden
    )
    if novel_decoy is None:
        return None
    forbidden.add(_block_key(novel_decoy))

    # One row repeats a true candidate from another row; all other rows repeat
    # one or more novel distractors. This makes row-local oddity indispensable.
    borrower_row, borrowed_row = random.sample(range(row_count), 2)
    repeated_by_row = []
    for row in range(row_count):
        if row == borrower_row:
            repeated_by_row.append(odd_by_row[borrowed_row])
        elif random.random() < 0.55:
            repeated_by_row.append(novel_decoy)
        else:
            extra = _different_pattern(
                height, width, background, foreground, forbidden
            )
            if extra is None:
                return None
            forbidden.add(_block_key(extra))
            repeated_by_row.append(extra)
    if any(np.array_equal(odd_by_row[row], repeated_by_row[row]) for row in range(row_count)):
        return None

    odd_columns = [random.randrange(column_count) for _ in range(row_count)]
    if len(set(odd_columns)) < 2:
        odd_columns[-1] = (odd_columns[0] + 1) % column_count

    grid_h = row_count * height + row_count - 1
    grid_w = column_count * width + column_count - 1
    grid = np.full((grid_h, grid_w), separator, dtype=int)
    for row in range(row_count):
        for col in range(column_count):
            block = odd_by_row[row] if col == odd_columns[row] else repeated_by_row[row]
            r0 = row * (height + 1)
            c0 = col * (width + 1)
            grid[r0:r0 + height, c0:c0 + width] = block

    expected = _render_vertical(chain_blocks, separator)
    if _validate_generated(
        grid,
        expected,
        odd_by_row,
        chain_rows,
        odd_columns,
        novel_decoy,
        background,
        separator,
        row_count,
        column_count,
        height,
        width,
    ):
        return grid
    return None


def generate_input() -> np.ndarray:
    """Generate a varied, non-degenerate lattice with a unique non-row-order chain."""
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GENERATION_ATTEMPTS} attempts")
