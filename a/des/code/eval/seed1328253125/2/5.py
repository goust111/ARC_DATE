import random

import numpy as np


BLACK = 0
COLOR_POOL = tuple(range(1, 10))

# Grid generation bounds.
BODY_HEIGHT_RANGE = (5, 12)
GRID_WIDTH_RANGE = (8, 18)
COLOR_COUNT_RANGE = (2, 5)
MAX_COLUMN_PIXELS = 6
MAX_GENERATION_ATTEMPTS = 300


def _template_candidates(grid: np.ndarray) -> list[int]:
    h = grid.shape[0]
    candidates: list[int] = []
    for r in (0, h - 1):
        row = grid[r]
        if np.all(row != BLACK) and len({int(v) for v in row}) >= 2:
            candidates.append(int(r))
    return candidates


def _body_bounds(height: int, template_row: int) -> tuple[int, int]:
    if template_row == 0:
        return 1, height
    return 0, height - 1


def _priority_from_template(template_row: np.ndarray) -> dict[int, int]:
    priority: dict[int, int] = {}
    for value in template_row:
        color = int(value)
        if color not in priority:
            priority[color] = len(priority)
    return priority


def _column_values(grid: np.ndarray, col: int, row_start: int, row_stop: int) -> list[int]:
    values: list[int] = []
    for r in range(row_start, row_stop):
        color = int(grid[r, col])
        if color != BLACK:
            values.append(color)
    return values


def _column_positions(grid: np.ndarray, col: int, row_start: int, row_stop: int) -> list[int]:
    positions: list[int] = []
    for r in range(row_start, row_stop):
        if int(grid[r, col]) != BLACK:
            positions.append(r - row_start)
    return positions


def _is_sorted_by_priority(values: list[int], priority: dict[int, int]) -> bool:
    return all(priority[values[i]] <= priority[values[i + 1]] for i in range(len(values) - 1))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder each column's colored pixels to match the template-row color priority.

    The unique template row sits on the top or bottom edge, stays unchanged,
    and defines a left-to-right first-occurrence priority over colors.
    Each non-template column is compacted toward the top of the non-template area
    after a stable sort by that priority.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h >= 2 and w >= 2, "grid must contain a template row and a body"

    candidates = _template_candidates(g)
    assert len(candidates) == 1, "expected exactly one edge template row"
    template_row = candidates[0]
    priority = _priority_from_template(g[template_row])
    assert len(priority) >= 2, "template row must contain at least two colors"

    row_start, row_stop = _body_bounds(h, template_row)
    out = g.copy()
    out[row_start:row_stop, :] = BLACK

    for c in range(w):
        values = _column_values(g, c, row_start, row_stop)
        for color in values:
            assert color in priority, "body colors must come from the template palette"
        values.sort(key=priority.__getitem__)
        for i, color in enumerate(values):
            out[row_start + i, c] = color

    return out


def _random_composition(total: int, parts: int, rng: random.Random) -> list[int]:
    if parts == 1:
        return [total]
    cuts = sorted(rng.sample(range(1, total), parts - 1))
    pieces: list[int] = []
    prev = 0
    for cut in cuts + [total]:
        pieces.append(cut - prev)
        prev = cut
    return pieces


def _sample_pixel_count(body_height: int, rng: random.Random) -> int:
    upper = min(body_height, MAX_COLUMN_PIXELS)
    if upper == 1:
        return 1
    roll = rng.random()
    if roll < 0.25:
        return 1
    if roll < 0.65:
        return rng.randint(2, min(3, upper))
    return rng.randint(2, upper)


def _sample_column_colors(colors: list[int], priority: dict[int, int], count: int, rng: random.Random) -> list[int]:
    if count == 1:
        return [rng.choice(colors)]

    roll = rng.random()
    if roll < 0.45 and len(colors) >= 2:
        i, j = sorted(rng.sample(range(len(colors)), 2))
        values = [colors[j], colors[i]]
        while len(values) < count:
            values.append(rng.choice(colors))
        if _is_sorted_by_priority(values, priority):
            values[0], values[1] = values[1], values[0]
        return values

    if roll < 0.75:
        indices = sorted(rng.randrange(len(colors)) for _ in range(count))
        return [colors[i] for i in indices]

    return [rng.choice(colors) for _ in range(count)]


def _ensure_all_template_colors(
    grid: np.ndarray,
    body_rows: list[int],
    colors: list[int],
    rng: random.Random,
) -> bool:
    used = {int(v) for v in grid[body_rows, :].ravel() if int(v) != BLACK}
    missing = [color for color in colors if color not in used]
    if not missing:
        return True

    available = [(r, c) for r in body_rows for c in range(grid.shape[1]) if int(grid[r, c]) == BLACK]
    rng.shuffle(available)
    if len(available) < len(missing):
        return False

    for color in missing:
        r, c = available.pop()
        grid[r, c] = color
    return True


def _column_change_stats(grid: np.ndarray, out: np.ndarray, template_row: int) -> tuple[int, bool, bool]:
    row_start, row_stop = _body_bounds(grid.shape[0], template_row)
    priority = _priority_from_template(grid[template_row])
    changed_columns = 0
    has_unsorted = False
    has_gap = False

    for c in range(grid.shape[1]):
        if not np.array_equal(grid[row_start:row_stop, c], out[row_start:row_stop, c]):
            changed_columns += 1
        before = _column_values(grid, c, row_start, row_stop)
        if len(before) >= 2 and not _is_sorted_by_priority(before, priority):
            has_unsorted = True
        positions = _column_positions(grid, c, row_start, row_stop)
        if positions and positions != list(range(len(positions))):
            has_gap = True

    return changed_columns, has_unsorted, has_gap


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    width = rng.randint(*GRID_WIDTH_RANGE)
    body_height = rng.randint(*BODY_HEIGHT_RANGE)
    total_height = body_height + 1

    color_count = rng.randint(COLOR_COUNT_RANGE[0], min(COLOR_COUNT_RANGE[1], width))
    colors = rng.sample(COLOR_POOL, color_count)
    band_widths = _random_composition(width, color_count, rng)

    template = np.zeros(width, dtype=int)
    cursor = 0
    for color, band_width in zip(colors, band_widths):
        template[cursor : cursor + band_width] = color
        cursor += band_width

    template_on_top = bool(rng.randrange(2))
    template_row = 0 if template_on_top else total_height - 1

    grid = np.zeros((total_height, width), dtype=int)
    grid[template_row, :] = template

    row_start, row_stop = _body_bounds(total_height, template_row)
    body_rows = list(range(row_start, row_stop))
    priority = _priority_from_template(template)

    min_active = min(width, max(color_count + 1, width // 2))
    active_count = rng.randint(min_active, width)
    active_cols = set(rng.sample(range(width), active_count))

    for c in range(width):
        if c not in active_cols:
            continue
        count = _sample_pixel_count(body_height, rng)
        rows = sorted(rng.sample(body_rows, count))
        values = _sample_column_colors(colors, priority, count, rng)
        for r, color in zip(rows, values):
            grid[r, c] = color

    if not _ensure_all_template_colors(grid, body_rows, colors, rng):
        return None

    opposite_edge = total_height - 1 if template_row == 0 else 0
    if np.all(grid[opposite_edge] != BLACK):
        grid[opposite_edge, rng.randrange(width)] = BLACK

    candidates = _template_candidates(grid)
    if candidates != [template_row]:
        return None

    out = main(grid)
    changed_columns, has_unsorted, has_gap = _column_change_stats(grid, out, template_row)
    if changed_columns < 2:
        return None
    if not has_unsorted:
        return None
    if not has_gap:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
