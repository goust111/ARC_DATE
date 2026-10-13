import random

import numpy as np


BACKGROUND = 0
SEPARATOR = 5
PALETTE = tuple(color for color in range(10) if color not in {BACKGROUND, SEPARATOR})

REGION_ROW_COUNT_RANGE = (2, 4)
REGION_COL_COUNT_RANGE = (2, 4)
REGION_SIZE_RANGE = (3, 6)
COMPONENT_COUNT_RANGE = (2, 3)
MAX_REGION_BUILD_ATTEMPTS = 80
MAX_GENERATION_ATTEMPTS = 300


def _mode(values):
    colors, counts = np.unique(values, return_counts=True)
    return int(colors[np.argmax(counts)])


def _spans_from_separators(separators):
    spans = []
    separators = sorted(set(int(idx) for idx in separators))
    for left, right in zip(separators, separators[1:]):
        assert right - left >= 2, "separator lines must leave a non-empty region"
        spans.append((left + 1, right - 1))
    assert spans, "expected at least one region span"
    return spans


def _find_separator_layout(grid):
    h, w = grid.shape
    candidates = []
    for color in map(int, np.unique(grid)):
        rows = [r for r in range(h) if np.all(grid[r, :] == color)]
        cols = [c for c in range(w) if np.all(grid[:, c] == color)]
        if len(rows) < 2 or len(cols) < 2:
            continue
        if 0 not in rows or h - 1 not in rows or 0 not in cols or w - 1 not in cols:
            continue
        try:
            _spans_from_separators(rows)
            _spans_from_separators(cols)
        except AssertionError:
            continue
        candidates.append((color, rows, cols))
    assert len(candidates) == 1, f"expected one separator color, got {candidates}"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """Checkerboardize each separator-bounded region using its local bottom-right marker color."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    separator, sep_rows, sep_cols = _find_separator_layout(g)
    bg = _mode(g[g != separator])
    row_spans = _spans_from_separators(sep_rows)
    col_spans = _spans_from_separators(sep_cols)

    out = g.copy()
    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            patch = g[r0 : r1 + 1, c0 : c1 + 1]
            contrast = int(patch[-1, -1])
            assert contrast not in {bg, separator}, "bottom-right cell must be a contrast marker"

            rr, cc = np.indices(patch.shape)
            foreground = (patch != bg) & (patch != separator)
            foreground[-1, -1] = False
            assert np.any(foreground), "each region must contain at least one non-marker foreground pixel"

            odd_cells = ((rr + cc) % 2 == 1) & foreground
            region_out = out[r0 : r1 + 1, c0 : c1 + 1]
            region_out[odd_cells] = contrast
            region_out[-1, -1] = bg

    return out


def _neighbors4(r, c):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _layout_from_sizes(sizes):
    spans = []
    separators = [0]
    cursor = 1
    for size in sizes:
        spans.append((cursor, cursor + size - 1))
        cursor += size
        separators.append(cursor)
        cursor += 1
    length = cursor
    return spans, separators, length


def _split_total(rng, total, parts):
    sizes = [2] * parts
    for _ in range(total - 2 * parts):
        sizes[rng.randrange(parts)] += 1
    rng.shuffle(sizes)
    return sizes


def _grow_component(rng, height, width, blocked, size):
    marker = (height - 1, width - 1)
    available = [
        (r, c)
        for r in range(height)
        for c in range(width)
        if (r, c) not in blocked and (r, c) != marker
    ]
    if len(available) < size:
        return None

    cells = {rng.choice(available)}
    while len(cells) < size:
        frontier = set()
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if not (0 <= nr < height and 0 <= nc < width):
                    continue
                if (nr, nc) in blocked or (nr, nc) in cells or (nr, nc) == marker:
                    continue
                frontier.add((nr, nc))
        if not frontier:
            return None
        cells.add(rng.choice(tuple(frontier)))
    return cells


def _make_region(rng, height, width):
    usable = height * width - 1
    max_components = min(COMPONENT_COUNT_RANGE[1], usable // 2)
    if max_components < COMPONENT_COUNT_RANGE[0]:
        return None

    for _ in range(MAX_REGION_BUILD_ATTEMPTS):
        component_count = rng.randint(COMPONENT_COUNT_RANGE[0], max_components)
        region_colors = rng.sample(PALETTE, component_count + 1)
        component_colors = region_colors[:-1]
        marker_color = region_colors[-1]

        min_fill = max(component_count * 2, usable // 3)
        max_fill = min(usable - 1, max(min_fill, (usable * 3) // 5))
        if max_fill < min_fill:
            max_fill = min_fill
        total_fill = rng.randint(min_fill, max_fill)
        component_sizes = _split_total(rng, total_fill, component_count)

        patch = np.full((height, width), BACKGROUND, dtype=int)
        blocked = set()
        ok = True
        for color, size in zip(component_colors, component_sizes):
            cells = _grow_component(rng, height, width, blocked, size)
            if cells is None:
                ok = False
                break
            for r, c in cells:
                patch[r, c] = color
            blocked.update(cells)
        if not ok:
            continue

        coords = np.argwhere(patch != BACKGROUND)
        parities = {(int(r) + int(c)) % 2 for r, c in coords}
        if parities != {0, 1}:
            continue

        patch[-1, -1] = marker_color
        return patch
    return None


def _gen_attempt(rng):
    region_rows = rng.randint(*REGION_ROW_COUNT_RANGE)
    region_cols = rng.randint(*REGION_COL_COUNT_RANGE)

    row_sizes = [rng.randint(*REGION_SIZE_RANGE) for _ in range(region_rows)]
    col_sizes = [rng.randint(*REGION_SIZE_RANGE) for _ in range(region_cols)]

    row_spans, sep_rows, height = _layout_from_sizes(row_sizes)
    col_spans, sep_cols, width = _layout_from_sizes(col_sizes)

    grid = np.full((height, width), BACKGROUND, dtype=int)
    grid[sep_rows, :] = SEPARATOR
    grid[:, sep_cols] = SEPARATOR

    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            patch = _make_region(rng, r1 - r0 + 1, c1 - c0 + 1)
            if patch is None:
                return None
            grid[r0 : r1 + 1, c0 : c1 + 1] = patch

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
