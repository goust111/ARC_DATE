from collections import Counter

import numpy as np
import itertools
import random


Block = tuple[int, int, int]  # c0, c1_exclusive, dominant color


def _unique_mode(values: np.ndarray) -> int:
    counts = Counter(map(int, values.reshape(-1)))
    ranked = counts.most_common()
    assert ranked, "expected a nonempty sequence"
    assert len(ranked) == 1 or ranked[0][1] > ranked[1][1], "expected a unique dominant color"
    return int(ranked[0][0])


def _vertical_blocks(g: np.ndarray) -> list[Block]:
    height, width = map(int, g.shape)
    assert height > 0 and width > 0, "expected a nonempty grid"

    column_colors = [_unique_mode(g[:, c]) for c in range(width)]
    blocks: list[Block] = []
    start = 0
    current = column_colors[0]
    for col, color in enumerate(column_colors[1:], start=1):
        if color != current:
            blocks.append((start, col, current))
            start = col
            current = color
    blocks.append((start, width, current))

    assert len(blocks) >= 3, "expected at least three vertical color blocks"
    base_colors = [color for _, _, color in blocks]
    assert len(set(base_colors)) == len(base_colors), "block colors must be unique"
    return blocks


def _common_marker_rows(g: np.ndarray, blocks: list[Block]) -> list[tuple[int, int]]:
    base_colors = {color for _, _, color in blocks}
    rows: list[tuple[int, int]] = []
    for r in range(g.shape[0]):
        shared = base_colors.copy()
        for c0, c1, _ in blocks:
            shared &= set(map(int, g[r, c0:c1]))
        if shared:
            assert len(shared) == 1, "expected each marker row to name one block color"
            rows.append((r, shared.pop()))
    return rows


def _block_order(g: np.ndarray, blocks: list[Block]) -> list[int]:
    block_count = len(blocks)
    base_colors = [color for _, _, color in blocks]
    marker_rows = _common_marker_rows(g, blocks)

    if len(marker_rows) == block_count:
        assert len({color for _, color in marker_rows}) == block_count, "marker rows must name unique blocks"
        row_numbers = [row for row, _ in marker_rows]
        marker_colors = [color for _, color in sorted(marker_rows)]
        if max(row_numbers) - min(row_numbers) + 1 == block_count and marker_colors == list(reversed(base_colors)):
            return list(reversed(range(block_count)))
    else:
        assert len(marker_rows) in (0, 1), "incomplete marker rows should be absent or unique"

    return list(range(1, block_count)) + [0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder the vertical color-major blocks while preserving every block's internal pixels.

    Input structure
    The grid is divided into vertical blocks, each with a unique dominant color.
    Small marker pixels from other block colors may appear inside the blocks.

    Rule
    If adjacent shared-color marker rows read exactly as the current block order
    from right to left, reverse the blocks. Otherwise move the leftmost block to
    the far right and shift the remaining blocks left.

    Key insight
    The transformation acts on whole vertical blocks, so marker pixels move with
    their source block instead of being recolored or independently rearranged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    blocks = _vertical_blocks(g)
    order = _block_order(g, blocks)
    assert sorted(order) == list(range(len(blocks))), "block order must be a permutation"

    return np.concatenate([g[:, blocks[i][0] : blocks[i][1]] for i in order], axis=1)

# --------------- generator (o5x) ---------------
# Canvas, block, and color bounds.
GEN_HEIGHT_RANGE = (7, 9)
GEN_TOTAL_WIDTH_RANGE = (10, 15)
GEN_BLOCK_COUNT_CHOICES = [3, 4]
GEN_BLOCK_WIDTH_RANGE = (3, 7)
GEN_COLOR_POOL = range(1, 10)

# Marker-layout probabilities and row-sharing limits.
GEN_REVERSE_PROB = 0.35
GEN_ALIGNED_LEFT_SHIFT_PROB = 0.7
GEN_NONCONSECUTIVE_LEFT_SHIFT_PROB = 0.5
GEN_COMMON_COLOR_COUNT_RANGE = (0, 1)

# Rejection-sampling retry limit.
GEN_MAX_ATTEMPTS = 300


def _gen_bounds(widths):
    starts = [0, *itertools.accumulate(widths[:-1])]
    return [(c0, c0 + width) for c0, width in zip(starts, widths)]


def _gen_expected(grid, widths, order):
    bounds = _gen_bounds(widths)
    return np.concatenate([grid[:, bounds[i][0] : bounds[i][1]] for i in order], axis=1)


def _gen_pick_widths(block_count):
    min_width, max_width = GEN_BLOCK_WIDTH_RANGE
    total = random.randint(
        max(GEN_TOTAL_WIDTH_RANGE[0], min_width * block_count),
        min(GEN_TOTAL_WIDTH_RANGE[1], max_width * block_count),
    )
    options = [
        widths
        for widths in itertools.product(range(min_width, max_width + 1), repeat=block_count)
        if sum(widths) == total
    ]
    return list(random.choice(options))


def _gen_marker_col(c0, c1):
    return random.randint(c0 + 1, c1 - 2)


def _gen_place_aligned(grid, widths, colors, color_rows):
    bounds = _gen_bounds(widths)
    for block_i, (c0, c1) in enumerate(bounds):
        base = colors[block_i]
        for color in colors:
            if color != base:
                grid[color_rows[color], _gen_marker_col(c0, c1)] = color


def _gen_nonconsecutive_rows(height, block_count):
    options = [
        rows
        for rows in itertools.combinations(range(height), block_count)
        if rows[-1] - rows[0] + 1 != block_count
    ]
    return list(random.choice(options))


def _gen_place_unaligned_left_shift(grid, widths, colors):
    height = grid.shape[0]
    bounds = _gen_bounds(widths)
    common_colors = random.sample(colors, random.randint(*GEN_COMMON_COLOR_COUNT_RANGE))
    row_by_common = {color: random.randrange(height) for color in common_colors}

    assignments = []
    for block_i, _ in enumerate(bounds):
        base = colors[block_i]
        used = set()
        by_color = {}
        for color in common_colors:
            if color != base:
                row = row_by_common[color]
                used.add(row)
                by_color[color] = row
        foreign = [color for color in colors if color != base and color not in common_colors]
        rows = random.sample([r for r in range(height) if r not in used], len(foreign))
        for color, row in zip(foreign, rows):
            by_color[color] = row
        assignments.append(by_color)

    for color in colors:
        if color in common_colors:
            continue
        block_ids = [i for i, base in enumerate(colors) if base != color]
        if len({assignments[i][color] for i in block_ids}) == 1:
            block_i = random.choice(block_ids)
            used = set(assignments[block_i].values())
            old_row = assignments[block_i][color]
            choices = [r for r in range(height) if r not in used and r != old_row]
            assignments[block_i][color] = random.choice(choices)

    for block_i, (c0, c1) in enumerate(bounds):
        for color, row in assignments[block_i].items():
            grid[row, _gen_marker_col(c0, c1)] = color


def _gen_base_grid():
    height = random.randint(*GEN_HEIGHT_RANGE)
    block_count = random.choice(GEN_BLOCK_COUNT_CHOICES)
    colors = random.sample(GEN_COLOR_POOL, block_count)
    widths = _gen_pick_widths(block_count)
    grid = np.zeros((height, sum(widths)), dtype=int)
    for color, (c0, c1) in zip(colors, _gen_bounds(widths)):
        grid[:, c0:c1] = color
    return grid, widths, colors


def _gen_place_reverse_markers(grid, widths, colors):
    block_count = len(colors)
    start = random.randint(0, grid.shape[0] - block_count)
    rows = list(range(start, start + block_count))
    color_rows = {color: row for color, row in zip(reversed(colors), rows)}
    _gen_place_aligned(grid, widths, colors, color_rows)
    return list(reversed(range(block_count)))


def _gen_place_left_shift_markers(grid, widths, colors):
    if random.random() < GEN_ALIGNED_LEFT_SHIFT_PROB:
        block_count = len(colors)
        if random.random() < GEN_NONCONSECUTIVE_LEFT_SHIFT_PROB:
            rows = _gen_nonconsecutive_rows(grid.shape[0], block_count)
            ordered_colors = random.sample(colors, block_count)
        else:
            start = random.randint(0, grid.shape[0] - block_count)
            rows = list(range(start, start + block_count))
            ordered_colors = list(reversed(colors))
            swap_i = random.randrange(block_count - 1)
            ordered_colors[swap_i : swap_i + 2] = reversed(ordered_colors[swap_i : swap_i + 2])
        color_rows = {color: row for color, row in zip(ordered_colors, rows)}
        _gen_place_aligned(grid, widths, colors, color_rows)
    else:
        _gen_place_unaligned_left_shift(grid, widths, colors)
    return list(range(1, len(colors))) + [0]


def _gen_place_markers(grid, widths, colors):
    if random.random() < GEN_REVERSE_PROB:
        return _gen_place_reverse_markers(grid, widths, colors)
    return _gen_place_left_shift_markers(grid, widths, colors)


def _gen_attempt():
    grid, widths, colors = _gen_base_grid()
    order = _gen_place_markers(grid, widths, colors)
    expected = _gen_expected(grid, widths, order)
    return grid if np.array_equal(main(grid), expected) else None


def generate_input():
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid f18ec8cc input")
