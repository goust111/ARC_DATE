from __future__ import annotations

import random

import numpy as np


def _most_common_color(values: np.ndarray) -> int:
    colors, counts = np.unique(np.asarray(values, dtype=int), return_counts=True)
    best = int(np.argmax(counts))
    assert int(np.sum(counts == counts[best])) == 1, "background color must be uniquely dominant"
    return int(colors[best])


def _find_separator_cross(grid: np.ndarray) -> tuple[int, int, int]:
    height, width = map(int, grid.shape)
    solid_rows = [
        (row, int(grid[row, 0]))
        for row in range(height)
        if np.all(grid[row, :] == int(grid[row, 0]))
    ]
    solid_cols = [
        (col, int(grid[0, col]))
        for col in range(width)
        if np.all(grid[:, col] == int(grid[0, col]))
    ]
    crosses = [
        (row, col, row_color)
        for row, row_color in solid_rows
        for col, col_color in solid_cols
        if row_color == col_color
    ]
    assert len(crosses) == 1, "expected exactly one separator cross"

    cross_row, cross_col, separator = crosses[0]
    assert 0 < cross_row < height - 1 and 0 < cross_col < width - 1
    assert height == 2 * cross_row + 1 and width == 2 * cross_col + 1, "quadrants must be equal-sized"
    return int(cross_row), int(cross_col), int(separator)


def _quadrant_slices(cross_row: int, cross_col: int, height: int, width: int) -> list[tuple[slice, slice]]:
    return [
        (slice(0, cross_row), slice(0, cross_col)),
        (slice(0, cross_row), slice(cross_col + 1, width)),
        (slice(cross_row + 1, height), slice(0, cross_col)),
        (slice(cross_row + 1, height), slice(cross_col + 1, width)),
    ]


def _pattern_signature(view: np.ndarray, color: int) -> frozenset[tuple[int, int]]:
    coords = np.argwhere(view == int(color))
    return frozenset((int(r), int(c)) for r, c in coords.tolist())


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each quadrant of a cross-divided board by restoring its unique missing color.

    Input structure:
    - One solid row and one solid column of the same color divide the board into
      four equal quadrants.
    - Each quadrant uses the same background color and almost the same set of
      payload colors, but each quadrant is missing exactly one different color.
    - For any payload color that is present, its local-coordinate pattern is the
      same in every quadrant that contains it.

    Rule:
    - Find the separator cross and split the grid into quadrants.
    - Take the union of payload colors over the four quadrants.
    - For each quadrant, identify the single missing color.
    - Read that color's local-coordinate pattern from the other quadrants and
      paint it into the missing quadrant wherever the cells are background.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)

    cross_row, cross_col, separator = _find_separator_cross(g)
    specs = _quadrant_slices(cross_row, cross_col, height, width)
    quadrants = [g[rows, cols] for rows, cols in specs]

    bg = _most_common_color(np.concatenate([quadrant.ravel() for quadrant in quadrants]))
    assert bg != separator, "background and separator must differ"

    color_sets = [
        {int(color) for color in np.unique(quadrant) if int(color) != bg}
        for quadrant in quadrants
    ]
    full_colors = set().union(*color_sets)
    assert len(full_colors) >= 4, "need at least four payload colors"

    missing_sets = [full_colors - colors for colors in color_sets]
    assert all(len(missing) == 1 for missing in missing_sets), "each quadrant must miss exactly one color"
    missing_colors = [next(iter(missing)) for missing in missing_sets]
    assert len(set(missing_colors)) == 4, "quadrants must miss different colors"

    patterns: dict[int, frozenset[tuple[int, int]]] = {}
    for color in full_colors:
        observed = [_pattern_signature(quadrant, color) for quadrant in quadrants]
        observed = [pattern for pattern in observed if pattern]
        assert observed, "every payload color must appear somewhere"
        reference = observed[0]
        assert all(pattern == reference for pattern in observed[1:]), "color pattern must agree across quadrants"
        patterns[int(color)] = reference

    out = g.copy()
    for (rows, cols), missing_color in zip(specs, missing_colors, strict=True):
        quadrant = out[rows, cols]
        for r, c in patterns[missing_color]:
            assert quadrant[r, c] == bg, "missing pattern must land on background"
            quadrant[r, c] = int(missing_color)

    return out


_SHAPE_LIBRARY = [
    np.array([[1]], dtype=bool),
    np.array([[1, 1]], dtype=bool),
    np.array([[1], [1]], dtype=bool),
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1], [1], [1]], dtype=bool),
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1], [0, 1]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
]

_QUADRANT_H_RANGE = (6, 9)
_QUADRANT_W_RANGE = (6, 9)
_PAYLOAD_COLOR_COUNT_RANGE = (4, 6)
_GENERATION_ATTEMPTS = 300
_PLACEMENT_ATTEMPTS = 80
_COLOR_POOL = tuple(range(10))


def _all_transforms(shape: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for turns in range(4):
        rotated = np.rot90(shape, turns)
        for variant in (rotated, np.fliplr(rotated)):
            key = (variant.shape, variant.astype(np.uint8).tobytes())
            if key in seen:
                continue
            seen.add(key)
            variants.append(variant)
    return variants


def _random_shape(max_h: int, max_w: int) -> np.ndarray:
    candidates = [
        variant
        for base in _SHAPE_LIBRARY
        for variant in _all_transforms(base)
        if variant.shape[0] <= max_h and variant.shape[1] <= max_w
    ]
    assert candidates, "need at least one shape that fits the quadrant"
    return random.choice(candidates)


def _place_patterns(qh: int, qw: int, colors: list[int]) -> dict[int, frozenset[tuple[int, int]]] | None:
    occupied = np.zeros((qh, qw), dtype=bool)
    patterns: dict[int, frozenset[tuple[int, int]]] = {}
    order = colors[:]
    random.shuffle(order)

    for color in order:
        placed = False
        for _ in range(_PLACEMENT_ATTEMPTS):
            shape = _random_shape(qh, qw)
            sh, sw = map(int, shape.shape)
            top = random.randint(0, qh - sh)
            left = random.randint(0, qw - sw)
            cells = [
                (top + int(r), left + int(c))
                for r, c in np.argwhere(shape).tolist()
            ]
            if any(occupied[r, c] for r, c in cells):
                continue
            for r, c in cells:
                occupied[r, c] = True
            patterns[int(color)] = frozenset(cells)
            placed = True
            break
        if not placed:
            return None

    return patterns


def _render_grid(
    qh: int,
    qw: int,
    bg: int,
    separator: int,
    colors: list[int],
    patterns: dict[int, frozenset[tuple[int, int]]],
    missing_colors: list[int],
) -> np.ndarray:
    grid = np.full((2 * qh + 1, 2 * qw + 1), int(bg), dtype=int)
    grid[qh, :] = int(separator)
    grid[:, qw] = int(separator)

    specs = _quadrant_slices(qh, qw, grid.shape[0], grid.shape[1])
    for quadrant_index, (rows, cols) in enumerate(specs):
        quadrant = grid[rows, cols]
        blocked_color = int(missing_colors[quadrant_index])
        for color in colors:
            if int(color) == blocked_color:
                continue
            for r, c in patterns[int(color)]:
                quadrant[r, c] = int(color)

    return grid


def _gen_attempt() -> np.ndarray | None:
    qh = random.randint(*_QUADRANT_H_RANGE)
    qw = random.randint(*_QUADRANT_W_RANGE)

    bg, separator = random.sample(_COLOR_POOL, 2)
    payload_pool = [color for color in _COLOR_POOL if color not in {bg, separator}]
    payload_count = random.randint(*_PAYLOAD_COLOR_COUNT_RANGE)
    colors = random.sample(payload_pool, payload_count)
    missing_colors = random.sample(colors, 4)

    patterns = _place_patterns(qh, qw, colors)
    if patterns is None:
        return None

    inp = _render_grid(qh, qw, bg, separator, colors, patterns, missing_colors)
    expected = _render_grid(qh, qw, bg, separator, colors, patterns, missing_colors=[-1, -1, -1, -1])

    try:
        actual = main(inp)
    except AssertionError:
        return None

    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
