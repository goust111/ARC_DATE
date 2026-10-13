from __future__ import annotations

from collections import Counter
import random

import numpy as np


def _unique_mode(values: np.ndarray) -> int:
    counts = Counter(map(int, values.reshape(-1)))
    ranked = counts.most_common()
    assert ranked, "expected a nonempty grid"
    assert len(ranked) == 1 or ranked[0][1] > ranked[1][1], "expected a unique dominant background"
    return int(ranked[0][0])


def _quadrant_slices(shape: tuple[int, int]) -> tuple[list[tuple[slice, slice]], int, int]:
    height, width = map(int, shape)
    assert height % 2 == 1 and width % 2 == 1, "expected odd grid dimensions"
    mid_r = height // 2
    mid_c = width // 2
    assert mid_r > 0 and mid_c > 0, "quadrants must be nonempty"
    slices = [
        (slice(0, mid_r), slice(0, mid_c)),
        (slice(0, mid_r), slice(mid_c + 1, width)),
        (slice(mid_r + 1, height), slice(0, mid_c)),
        (slice(mid_r + 1, height), slice(mid_c + 1, width)),
    ]
    return slices, mid_r, mid_c


def _color_set(block: np.ndarray, bg: int) -> set[int]:
    return {int(color) for color in np.unique(block) if int(color) != bg}


def _color_pattern(block: np.ndarray, color: int) -> frozenset[tuple[int, int]]:
    rows, cols = np.where(block == color)
    coords = frozenset((int(r), int(c)) for r, c in zip(rows.tolist(), cols.tolist(), strict=False))
    assert coords, "expected every pattern color to occupy at least one cell"
    return coords


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover each quadrant's missing color by comparing color sets across quadrants,
    then copy that color's local coordinate pattern from the other quadrants.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    quadrant_slices, mid_r, mid_c = _quadrant_slices(g.shape)
    separator = int(g[mid_r, 0])
    assert np.all(g[mid_r, :] == separator), "expected a full separator row"
    assert np.all(g[:, mid_c] == separator), "expected a full separator column"

    bg = _unique_mode(g)
    quadrants = [g[row_slice, col_slice] for row_slice, col_slice in quadrant_slices]
    color_sets = [_color_set(block, bg) for block in quadrants]
    all_colors = set().union(*color_sets)
    assert all_colors, "expected at least one non-background pattern color"

    missing_by_quadrant: list[int] = []
    for colors in color_sets:
        missing = all_colors - colors
        assert len(missing) == 1, "each quadrant must be missing exactly one color"
        missing_by_quadrant.append(next(iter(missing)))
    assert len(set(missing_by_quadrant)) == len(missing_by_quadrant), "missing colors must differ by quadrant"

    patterns: dict[int, frozenset[tuple[int, int]]] = {}
    for color in all_colors:
        seen_patterns = [_color_pattern(block, color) for block, colors in zip(quadrants, color_sets) if color in colors]
        assert seen_patterns, "every color must appear somewhere"
        base = seen_patterns[0]
        assert all(pattern == base for pattern in seen_patterns[1:]), "color pattern must agree across quadrants"
        patterns[int(color)] = base

    out = g.copy()
    for (row_slice, col_slice), missing_color in zip(quadrant_slices, missing_by_quadrant):
        view = out[row_slice, col_slice]
        for r, c in patterns[missing_color]:
            assert int(view[r, c]) == bg, "missing cells should be blank in the target quadrant"
            view[r, c] = int(missing_color)

    return out


# --------------- generator ---------------

ARC_COLORS = tuple(range(10))
QUADRANT_HEIGHT_RANGE = (4, 7)
QUADRANT_WIDTH_RANGE = (4, 7)
MAX_PATTERN_COLORS = 6
MAX_SHAPE_SIZE = 5
MAX_GENERATION_ATTEMPTS = 400
SAME_AS_BG_SEPARATOR_PROB = 0.45


def _bounded_composition(total: int, parts: int, upper: int) -> list[int] | None:
    if parts == 0:
        return [] if total == 0 else None
    if not (parts <= total <= parts * upper):
        return None

    values = []
    remaining = total
    for i in range(parts):
        slots = parts - i - 1
        low = max(1, remaining - slots * upper)
        high = min(upper, remaining - slots)
        if low > high:
            return None
        values.append(random.randint(low, high))
        remaining -= values[-1]
    random.shuffle(values)
    return values


def _normalize_shape(cells: set[tuple[int, int]]) -> list[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return sorted((r - min_r, c - min_c) for r, c in cells)


def _random_polyomino(size: int) -> list[tuple[int, int]]:
    cells = {(0, 0)}
    frontier = {(1, 0), (-1, 0), (0, 1), (0, -1)}
    while len(cells) < size:
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        cells.add(cell)
        r, c = cell
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            neighbor = (r + dr, c + dc)
            if neighbor not in cells:
                frontier.add(neighbor)
    return _normalize_shape(cells)


def _reserve_with_gap(blocked: np.ndarray, coords: list[tuple[int, int]]) -> None:
    height, width = blocked.shape
    for r, c in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < height and 0 <= cc < width:
                    blocked[rr, cc] = True


def _place_shape(
    canvas: np.ndarray,
    blocked: np.ndarray,
    color: int,
    size: int,
    bg: int,
) -> frozenset[tuple[int, int]] | None:
    height, width = canvas.shape
    for _ in range(80):
        shape = _random_polyomino(size)
        shape_h = max(r for r, _ in shape) + 1
        shape_w = max(c for _, c in shape) + 1
        placements = []
        for off_r in range(height - shape_h + 1):
            for off_c in range(width - shape_w + 1):
                shifted = [(r + off_r, c + off_c) for r, c in shape]
                if all(not bool(blocked[r, c]) for r, c in shifted):
                    placements.append(shifted)
        if not placements:
            continue
        chosen = random.choice(placements)
        for r, c in chosen:
            assert int(canvas[r, c]) == bg
            canvas[r, c] = int(color)
        _reserve_with_gap(blocked, chosen)
        return frozenset((int(r), int(c)) for r, c in chosen)
    return None


def _make_template(height: int, width: int, bg: int, colors: list[int]) -> tuple[np.ndarray, dict[int, frozenset[tuple[int, int]]]] | None:
    num_colors = len(colors)
    max_shape = min(MAX_SHAPE_SIZE, max(2, (height * width) // num_colors))
    min_total = max(height + width - 1, num_colors + 2)
    max_total = min((height * width) // 2, num_colors * max_shape)
    if min_total > max_total:
        return None

    total_cells = random.randint(min_total, max_total)
    sizes = _bounded_composition(total_cells, num_colors, upper=max_shape)
    if sizes is None or sum(size > 1 for size in sizes) < 2:
        return None

    canvas = np.full((height, width), bg, dtype=int)
    blocked = np.zeros((height, width), dtype=bool)
    placements: dict[int, frozenset[tuple[int, int]]] = {}
    for color, size in sorted(zip(colors, sizes), key=lambda item: item[1], reverse=True):
        placed = _place_shape(canvas, blocked, int(color), int(size), bg)
        if placed is None:
            return None
        placements[int(color)] = placed

    if not np.all(np.any(canvas != bg, axis=1)):
        return None
    if not np.all(np.any(canvas != bg, axis=0)):
        return None
    return canvas, placements


def _quadrant_views(grid: np.ndarray) -> list[tuple[slice, slice]]:
    slices, _, _ = _quadrant_slices(grid.shape)
    return slices


def _dominant_is_background(grid: np.ndarray, bg: int) -> bool:
    counts = Counter(map(int, grid.reshape(-1)))
    ranked = counts.most_common()
    if not ranked or int(ranked[0][0]) != int(bg):
        return False
    return len(ranked) == 1 or ranked[0][1] > ranked[1][1]


def _gen_attempt() -> np.ndarray | None:
    qh = random.randint(*QUADRANT_HEIGHT_RANGE)
    qw = random.randint(*QUADRANT_WIDTH_RANGE)
    bg = random.choice(ARC_COLORS)

    if random.random() < SAME_AS_BG_SEPARATOR_PROB:
        separator = bg
    else:
        separator = random.choice([color for color in ARC_COLORS if color != bg])

    unavailable = {bg, separator}
    available = [color for color in ARC_COLORS if color not in unavailable]
    max_colors = min(MAX_PATTERN_COLORS, max(4, (qh * qw) // 4), len(available))
    if max_colors < 4:
        return None
    color_count = random.randint(4, max_colors)
    pattern_colors = random.sample(available, color_count)

    built = _make_template(qh, qw, bg, pattern_colors)
    if built is None:
        return None
    template, placements = built
    assert set(placements) == set(pattern_colors)

    missing_colors = random.sample(pattern_colors, 4)
    out = np.full((2 * qh + 1, 2 * qw + 1), bg, dtype=int)
    mid_r = qh
    mid_c = qw
    out[mid_r, :] = separator
    out[:, mid_c] = separator
    for row_slice, col_slice in _quadrant_views(out):
        out[row_slice, col_slice] = template

    inp = out.copy()
    for (row_slice, col_slice), missing_color in zip(_quadrant_views(inp), missing_colors):
        view = inp[row_slice, col_slice]
        view[template == int(missing_color)] = bg

    if not _dominant_is_background(inp, bg):
        return None

    try:
        solved = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {MAX_GENERATION_ATTEMPTS} attempts")
