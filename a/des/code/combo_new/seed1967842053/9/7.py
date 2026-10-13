import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    """Reorder the interior pattern by row and column labels encoded on the border.

    Input structure:
    The outer frame is a control border and the four corner cells are black.  The
    left and right borders contain the same set of distinct row-label colors in
    different orders.  The top and bottom borders contain the same set of
    distinct column-label colors in different orders.  The inner rectangle is the
    colored pattern to transform.

    Rule:
    For each target row position, find that row label on the right border, then
    copy the interior row carrying the same label on the left border.  For each
    target column position, find that column label on the bottom border, then
    copy the interior column carrying the same label on the top border.  Apply
    these two independent reorderings to the interior and omit the border.

    Key insight:
    The border colors are not decoration: opposite border pairs encode source
    and target label orders for two independent permutations, one on rows and
    one on columns.
    """
    arr = np.asarray(grid)
    interior = arr[1:-1, 1:-1]

    left = arr[1:-1, 0]
    right = arr[1:-1, -1]
    top = arr[0, 1:-1]
    bottom = arr[-1, 1:-1]

    row_sources = {int(color): idx for idx, color in enumerate(left)}
    col_sources = {int(color): idx for idx, color in enumerate(top)}

    row_order = [row_sources[int(color)] for color in right]
    col_order = [col_sources[int(color)] for color in bottom]

    return interior[np.ix_(row_order, col_order)].copy()


def _shuffled_order(rng, values):
    values = np.asarray(values)
    for _ in range(24):
        candidate = rng.permutation(values)
        if not np.array_equal(candidate, values):
            return candidate
    return np.roll(values, 1)


def _shape_cells(rng, height, width):
    kind = int(rng.integers(0, 6))

    if kind == 0:
        return [(0, 0)], 1, 1

    if kind == 1:
        length = int(rng.integers(2, min(4, width) + 1))
        return [(0, c) for c in range(length)], 1, length

    if kind == 2:
        length = int(rng.integers(2, min(4, height) + 1))
        return [(r, 0) for r in range(length)], length, 1

    if kind == 3:
        shape_h = int(rng.integers(2, min(3, height) + 1))
        shape_w = int(rng.integers(2, min(3, width) + 1))
        return [(r, c) for r in range(shape_h) for c in range(shape_w)], shape_h, shape_w

    if kind == 4:
        arm_h = int(rng.integers(2, min(3, height) + 1))
        arm_w = int(rng.integers(2, min(3, width) + 1))
        orient = int(rng.integers(0, 4))
        if orient == 0:
            cells = [(r, 0) for r in range(arm_h)] + [(arm_h - 1, c) for c in range(arm_w)]
        elif orient == 1:
            cells = [(r, arm_w - 1) for r in range(arm_h)] + [(arm_h - 1, c) for c in range(arm_w)]
        elif orient == 2:
            cells = [(r, 0) for r in range(arm_h)] + [(0, c) for c in range(arm_w)]
        else:
            cells = [(r, arm_w - 1) for r in range(arm_h)] + [(0, c) for c in range(arm_w)]
        return sorted(set(cells)), arm_h, arm_w

    length = int(rng.integers(2, min(4, height, width) + 1))
    if bool(rng.integers(0, 2)):
        return [(i, i) for i in range(length)], length, length
    return [(i, length - 1 - i) for i in range(length)], length, length


def _fill_pattern(rng, height, width):
    pattern = np.zeros((height, width), dtype=int)
    occupied = np.zeros((height, width), dtype=bool)
    max_objects = max(2, min(7, (height * width) // 4 + 1))
    object_count = int(rng.integers(2, max_objects + 1))

    for _ in range(object_count):
        for _attempt in range(80):
            cells, shape_h, shape_w = _shape_cells(rng, height, width)
            row = int(rng.integers(0, height - shape_h + 1))
            col = int(rng.integers(0, width - shape_w + 1))
            coords = [(row + r, col + c) for r, c in cells]
            if any(occupied[r, c] for r, c in coords):
                continue

            color = int(rng.integers(1, 10))
            for r, c in coords:
                pattern[r, c] = color
                occupied[r, c] = True
            break

    return pattern


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()

    inner_h = int(rng.integers(3, 10))
    inner_w = int(rng.integers(3, 10))

    row_labels = rng.choice(np.arange(1, 10), size=inner_h, replace=False)
    col_labels = rng.choice(np.arange(1, 10), size=inner_w, replace=False)
    target_rows = _shuffled_order(rng, row_labels)
    target_cols = _shuffled_order(rng, col_labels)

    grid = np.zeros((inner_h + 2, inner_w + 2), dtype=int)
    grid[1:-1, 0] = row_labels
    grid[1:-1, -1] = target_rows
    grid[0, 1:-1] = col_labels
    grid[-1, 1:-1] = target_cols
    grid[1:-1, 1:-1] = _fill_pattern(rng, inner_h, inner_w)

    return grid
