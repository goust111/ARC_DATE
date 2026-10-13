import numpy as np


ORANGE = 7
MAX_GENERATION_ATTEMPTS = 400
GRID_ROWS_RANGE = (8, 17)
GRID_COLS_RANGE = (8, 17)
COLOR_COUNT_RANGE = (3, 6)
MIN_MINOR_COUNT = 2


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep the two most frequent colors and replace every other color with orange.

    The decision depends only on global color frequencies.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    colors, counts = np.unique(g, return_counts=True)
    assert len(colors) >= 3, "expected at least three colors"
    assert ORANGE not in set(map(int, colors)), "orange must be absent from the input"

    order = np.argsort(-counts)
    ranked_colors = colors[order]
    ranked_counts = counts[order]
    assert int(ranked_counts[1]) > int(ranked_counts[2]), "top two colors must be strictly above third place"

    keep = {int(ranked_colors[0]), int(ranked_colors[1])}
    out = np.full_like(g, ORANGE)
    for color in keep:
        out[g == color] = color
    return out


def _random_partition(total: int, parts: int, low: int, high: int) -> list[int] | None:
    if parts * low > total or parts * high < total:
        return None

    values = [low] * parts
    remaining = total - parts * low
    for i in range(parts):
        left_parts = parts - i - 1
        room_here = min(high - values[i], remaining)
        min_left = left_parts * low
        max_left = left_parts * high
        lo = max(0, remaining - max_left)
        hi = min(room_here, remaining - min_left)
        if hi < lo:
            return None
        add = int(np.random.randint(lo, hi + 1))
        values[i] += add
        remaining -= add

    if remaining != 0:
        return None
    np.random.shuffle(values)
    return values


def _pick_counts(cell_count: int, color_count: int) -> list[int] | None:
    tie_top = bool(np.random.randint(0, 2))
    gap = max(2, cell_count // 20)
    minor_count = color_count - 2

    top2_low = max(6, cell_count // 5)
    top2_high = max(top2_low, cell_count // 3)

    for _ in range(80):
        second = int(np.random.randint(top2_low, top2_high + 1))
        first = second if tie_top else int(np.random.randint(second + 1, min(cell_count, second + max(2, cell_count // 8)) + 1))
        remaining = cell_count - first - second
        max_minor = second - gap
        if max_minor < MIN_MINOR_COUNT:
            continue
        minors = _random_partition(remaining, minor_count, MIN_MINOR_COUNT, max_minor)
        if minors is None:
            continue
        counts = [first, second] + minors
        if sorted(counts, reverse=True)[1] <= sorted(counts, reverse=True)[2]:
            continue
        return counts
    return None


def _snake_path(shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    by_rows = bool(np.random.randint(0, 2))
    path: list[tuple[int, int]] = []

    if by_rows:
        row_order = list(range(h))
        if bool(np.random.randint(0, 2)):
            row_order.reverse()
        left_to_right_first = bool(np.random.randint(0, 2))
        for offset, r in enumerate(row_order):
            cols = list(range(w))
            if (offset % 2 == 0) != left_to_right_first:
                cols.reverse()
            for c in cols:
                path.append((r, c))
    else:
        col_order = list(range(w))
        if bool(np.random.randint(0, 2)):
            col_order.reverse()
        top_to_bottom_first = bool(np.random.randint(0, 2))
        for offset, c in enumerate(col_order):
            rows = list(range(h))
            if (offset % 2 == 0) != top_to_bottom_first:
                rows.reverse()
            for r in rows:
                path.append((r, c))
    return path


def _paint_regions(shape: tuple[int, int], colors: list[int], counts: list[int]) -> np.ndarray | None:
    h, w = shape
    if sum(counts) != h * w:
        return None

    grid = np.empty(shape, dtype=int)
    path = _snake_path(shape)
    start = 0
    for color, count in zip(colors, counts):
        for r, c in path[start : start + count]:
            grid[r, c] = color
        start += count
    return grid


def _gen_attempt() -> np.ndarray | None:
    h = int(np.random.randint(GRID_ROWS_RANGE[0], GRID_ROWS_RANGE[1] + 1))
    w = int(np.random.randint(GRID_COLS_RANGE[0], GRID_COLS_RANGE[1] + 1))
    color_count = int(np.random.randint(COLOR_COUNT_RANGE[0], COLOR_COUNT_RANGE[1] + 1))

    palette = [c for c in range(10) if c != ORANGE]
    chosen = list(np.random.choice(palette, size=color_count, replace=False))
    np.random.shuffle(chosen)

    counts = _pick_counts(h * w, color_count)
    if counts is None:
        return None

    grid = _paint_regions((h, w), chosen, counts)
    if grid is None:
        return None

    out = main(grid)
    if not np.any(out == ORANGE):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
