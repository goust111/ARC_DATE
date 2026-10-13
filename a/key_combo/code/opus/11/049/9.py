import random
from collections import Counter

import numpy as np


BACKGROUND = 0
GRID_MIN = 8
GRID_MAX = 12
LABEL_COLOR_OPTIONS = (3, 4)
LEGEND_HEIGHT_OPTIONS = (2, 3)
GEN_RETRIES = 600
COLOR_POOL = tuple(range(1, 10))


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _apply_chain_to_color(color: int, pairs: list[tuple[int, int]]) -> int:
    current = int(color)
    for src, dst in pairs:
        if current == src:
            current = dst
    return current


def _chain_steps(color: int, pairs: list[tuple[int, int]]) -> int:
    current = int(color)
    steps = 0
    for src, dst in pairs:
        if current == src:
            current = dst
            steps += 1
    return steps


def _legend_height(grid: np.ndarray, bg: int) -> int:
    height = 0
    limit = grid.shape[0] - 1
    while height < limit and int(grid[height, 0]) != bg and int(grid[height, 1]) != bg:
        height += 1
    assert 2 <= height <= 3, "expected a 2-3 row legend in the top-left interior corner"
    return height


def _parse(grid: np.ndarray) -> tuple[np.ndarray, int, np.ndarray, np.ndarray, list[tuple[int, int]], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = g.shape
    assert height >= 5 and width >= 5, "expected border labels around a non-empty interior"

    bg = _get_bg(g)
    assert int(g[-1, -1]) == bg, "expected a background bottom-right corner"

    row_labels = g[:-1, -1]
    col_labels = g[-1, :-1]
    assert np.all(row_labels != bg), "expected non-background row labels on the right edge"
    assert np.all(col_labels != bg), "expected non-background column labels on the bottom edge"

    legend_h = _legend_height(g, bg)
    interior = g[:-1, :-1]
    legend_mask = np.zeros_like(interior, dtype=bool)
    legend_mask[:legend_h, :2] = True

    assert np.all(interior[legend_mask] != bg), "expected a fully filled 2-column legend"
    assert np.all(interior[~legend_mask] == bg), "expected the rest of the interior to be background"

    pairs = [(int(g[r, 0]), int(g[r, 1])) for r in range(legend_h)]
    assert len({color for pair in pairs for color in pair}) >= 3, "expected a multicolor legend"
    return g, bg, row_labels.astype(int), col_labels.astype(int), pairs, legend_mask


def _raw_intersection(
    row_labels: np.ndarray, col_labels: np.ndarray, bg: int, legend_mask: np.ndarray
) -> np.ndarray:
    table = np.where(row_labels[:, None] == col_labels[None, :], row_labels[:, None], bg)
    table = table.astype(int)
    table[legend_mask] = bg
    return table


def _expected_from_parts(
    row_labels: np.ndarray,
    col_labels: np.ndarray,
    pairs: list[tuple[int, int]],
    legend_h: int,
) -> np.ndarray:
    interior = np.where(row_labels[:, None] == col_labels[None, :], row_labels[:, None], BACKGROUND).astype(int)
    interior[:legend_h, :2] = BACKGROUND
    for src, dst in pairs:
        interior = np.where(interior == src, dst, interior)
    return interior


def main(grid: np.ndarray) -> np.ndarray:
    """Fill the row-column equality table, then recolor it by an ordered two-column legend.

    Input structure:
    - The rightmost column holds one non-background row label per interior row.
    - The bottom row holds one non-background column label per interior column, with a background bottom-right corner.
    - The top-left interior corner contains a fully filled 2-column legend with 2-3 rows, and every other interior cell is background.

    Rule:
    - Read the right edge and bottom edge as row and column labels.
    - In the interior, fill every non-legend cell whose row label equals its column label with that shared color; leave mismatches as background.
    - Read the legend rows from top to bottom as ordered substitutions left->right, apply them sequentially to the filled interior so one cell may change multiple times, then erase the legend itself to background while keeping the border labels unchanged.

    Key insight:
    - The interior is first an equality table defined by the border labels, and the legend is not decoration: it is a chained recoloring program applied after that fill.
    """
    g, bg, row_labels, col_labels, pairs, legend_mask = _parse(grid)
    interior = _raw_intersection(row_labels, col_labels, bg, legend_mask)
    for src, dst in pairs:
        interior = np.where(interior == src, dst, interior)

    out = g.copy()
    out[:-1, :-1] = interior
    out[-1, -1] = bg
    return out


def _positive_partition(total: int, parts: int) -> list[int]:
    lengths = [1] * parts
    for _ in range(total - parts):
        lengths[random.randrange(parts)] += 1
    random.shuffle(lengths)
    return lengths


def _grouped_labels(total: int, colors: list[int], order: list[int] | None = None) -> np.ndarray:
    if order is None:
        order = random.sample(colors, len(colors))
    lengths = _positive_partition(total, len(colors))
    seq = np.concatenate(
        [np.full(length, color, dtype=int) for color, length in zip(order, lengths)]
    )
    return seq


def _build_pairs(label_colors: list[int], legend_h: int, extra_color: int | None) -> list[tuple[int, int]] | None:
    destinations = list(label_colors)
    if extra_color is not None:
        destinations.append(int(extra_color))

    for _ in range(400):
        a, b = random.sample(label_colors, 2)
        c_choices = [color for color in destinations if color != b]
        c = random.choice(c_choices)
        pairs = [(a, b), (b, c)]

        if legend_h == 3:
            src = random.choice(label_colors)
            dst = random.choice([color for color in destinations if color != src])
            insert_at = random.randrange(4)
            pairs = pairs.copy()
            pairs.insert(insert_at, (src, dst))

        final_map = {color: _apply_chain_to_color(color, pairs) for color in label_colors}
        src_colors = {src for src, _ in pairs}

        if len({color for pair in pairs for color in pair}) < 3:
            continue
        if len(set(final_map.values())) < 2:
            continue
        if not any(_chain_steps(color, pairs) >= 2 for color in label_colors):
            continue
        if any(final_map[src] == src for src in src_colors):
            continue
        if extra_color is not None and not any(dst == extra_color for _, dst in pairs):
            continue
        return pairs
    return None


def _assembled_output(
    shape: tuple[int, int],
    row_labels: np.ndarray,
    col_labels: np.ndarray,
    pairs: list[tuple[int, int]],
    legend_h: int,
) -> np.ndarray:
    out = np.full(shape, BACKGROUND, dtype=int)
    out[:-1, :-1] = _expected_from_parts(row_labels, col_labels, pairs, legend_h)
    out[:-1, -1] = row_labels
    out[-1, :-1] = col_labels
    return out


def _attempt() -> np.ndarray | None:
    height = random.randint(GRID_MIN, GRID_MAX)
    width = random.randint(GRID_MIN, GRID_MAX)
    interior_h = height - 1
    interior_w = width - 1

    label_color_count = random.choice(LABEL_COLOR_OPTIONS)
    if min(interior_h, interior_w) < label_color_count:
        return None

    label_colors = random.sample(COLOR_POOL, label_color_count)
    extra_color = None
    if label_color_count == 3 or random.random() < 0.45:
        extra_choices = [color for color in COLOR_POOL if color not in label_colors]
        extra_color = random.choice(extra_choices)

    legend_h = random.choice(LEGEND_HEIGHT_OPTIONS)
    if interior_h <= legend_h:
        return None

    pairs = _build_pairs(label_colors, legend_h, extra_color)
    if pairs is None:
        return None

    shared_order = random.sample(label_colors, len(label_colors))
    row_order = shared_order if random.random() < 0.55 else random.sample(label_colors, len(label_colors))
    if random.random() < 0.4:
        col_order = row_order[:]
    elif random.random() < 0.5:
        col_order = list(reversed(shared_order))
    else:
        col_order = random.sample(label_colors, len(label_colors))

    row_labels = _grouped_labels(interior_h, label_colors, row_order)
    col_labels = _grouped_labels(interior_w, label_colors, col_order)

    raw = _expected_from_parts(row_labels, col_labels, [], legend_h)
    for r in range(legend_h):
        raw[r, :2] = BACKGROUND
    changed = _expected_from_parts(row_labels, col_labels, pairs, legend_h)

    filled_mask = raw != BACKGROUND
    if int(filled_mask.sum()) < 6:
        return None
    if not np.any(changed != raw):
        return None
    if int(np.count_nonzero((changed != raw) & filled_mask)) < 2:
        return None
    if len(set(map(int, changed[changed != BACKGROUND].ravel()))) < 2:
        return None

    step_counts = {color: _chain_steps(color, pairs) for color in label_colors}
    multi_step_colors = {color for color, steps in step_counts.items() if steps >= 2}
    if not multi_step_colors:
        return None
    if not any(int(np.count_nonzero(raw == color)) > 0 for color in multi_step_colors):
        return None

    grid = np.full((height, width), BACKGROUND, dtype=int)
    for r, (src, dst) in enumerate(pairs):
        grid[r, 0] = src
        grid[r, 1] = dst
    grid[:-1, -1] = row_labels
    grid[-1, :-1] = col_labels

    solver_out = main(grid)
    expected_out = _assembled_output((height, width), row_labels, col_labels, pairs, legend_h)
    if not np.array_equal(solver_out, expected_out):
        return None

    parsed = _parse(grid)
    parsed_pairs = parsed[4]
    if parsed_pairs != pairs:
        return None

    raw_from_parse = _raw_intersection(row_labels, col_labels, BACKGROUND, parsed[5])
    if not np.array_equal(raw_from_parse, raw):
        return None

    assert np.all(grid[:legend_h, :2] != BACKGROUND)
    assert np.all(solver_out[:legend_h, :2] == BACKGROUND)
    assert np.array_equal(solver_out[:-1, -1], row_labels)
    assert np.array_equal(solver_out[-1, :-1], col_labels)
    assert int(solver_out[-1, -1]) == BACKGROUND
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a non-degenerate row/column-chain example")
