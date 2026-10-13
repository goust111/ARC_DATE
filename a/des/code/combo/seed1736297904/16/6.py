from __future__ import annotations

import random

import numpy as np


BLACK = 0
YELLOW = 4
GREY = 5

MOTHER_COLORS = (1, 2, 3, 6, 7, 8, 9)
LAYOUT_CHOICES = ((2, 2), (2, 3), (3, 2), (3, 3))
CELL_HEIGHT_RANGE = (4, 7)
CELL_WIDTH_RANGE = (4, 7)
OUTER_FRAME_PROB = 0.5
MAX_ATTEMPTS = 300


def _spans_from_separators(separators: list[int], limit: int) -> list[tuple[int, int]]:
    lines = sorted({int(x) for x in separators if 0 <= int(x) < int(limit)})
    bounds = [-1] + lines + [int(limit)]
    spans: list[tuple[int, int]] = []
    for a, b in zip(bounds, bounds[1:]):
        if b > a + 1:
            spans.append((a + 1, b))
    assert spans, "expected at least one non-separator span"
    return spans


def _extract_regions(grid: np.ndarray) -> list[dict[str, object]]:
    row_separators = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == GREY)]
    col_separators = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]
    row_spans = _spans_from_separators(row_separators, int(grid.shape[0]))
    col_spans = _spans_from_separators(col_separators, int(grid.shape[1]))

    regions: list[dict[str, object]] = []
    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            regions.append(
                {
                    "bbox": (int(r0), int(c0), int(r1 - 1), int(c1 - 1)),
                    "grid": grid[r0:r1, c0:c1],
                }
            )
    assert len(regions) >= 2, "expected one selector and at least one mother region"
    return regions


def _find_selector_index(regions: list[dict[str, object]]) -> int:
    candidates: list[int] = []
    for i, region in enumerate(regions):
        panel = np.asarray(region["grid"], dtype=int)
        colors = set(map(int, np.unique(panel)))
        if YELLOW in colors and colors <= {BLACK, YELLOW}:
            candidates.append(i)
    assert len(candidates) == 1, "expected exactly one selector region"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read a yellow row/column mask from the selector region, crop every mother
    region at those row/column indices, then concatenate the cropped pieces in
    raster-scan order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    regions = _extract_regions(g)
    selector_index = _find_selector_index(regions)
    selector = np.asarray(regions[selector_index]["grid"], dtype=int)

    shapes = {tuple(np.asarray(region["grid"]).shape) for region in regions}
    assert len(shapes) == 1, "all regions must share the same inner shape"

    row_mask = np.any(selector == YELLOW, axis=1)
    col_mask = np.any(selector == YELLOW, axis=0)
    rows = np.flatnonzero(row_mask)
    cols = np.flatnonzero(col_mask)
    assert rows.size > 0 and cols.size > 0, "selector must mark at least one row and one column"

    pieces: list[np.ndarray] = []
    for i, region in enumerate(regions):
        if i == selector_index:
            continue
        panel = np.asarray(region["grid"], dtype=int)
        pieces.append(panel[np.ix_(rows, cols)])

    assert pieces, "expected at least one mother region"
    return np.concatenate(pieces, axis=1)


def _intervals(length: int, cuts: list[int]) -> list[tuple[int, int]]:
    points = [0] + sorted(cuts) + [int(length)]
    return [(points[i], points[i + 1]) for i in range(len(points) - 1)]


def _pattern_row_bands(height: int, width: int, colors: list[int]) -> np.ndarray:
    band = random.randint(1, max(1, height // 2))
    phase = random.randrange(len(colors))
    row_colors = [colors[((r // band) + phase) % len(colors)] for r in range(height)]
    return np.repeat(np.array(row_colors, dtype=int)[:, None], width, axis=1)


def _pattern_col_bands(height: int, width: int, colors: list[int]) -> np.ndarray:
    band = random.randint(1, max(1, width // 2))
    phase = random.randrange(len(colors))
    col_colors = [colors[((c // band) + phase) % len(colors)] for c in range(width)]
    return np.repeat(np.array(col_colors, dtype=int)[None, :], height, axis=0)


def _pattern_checker(height: int, width: int, colors: list[int]) -> np.ndarray:
    rr, cc = np.indices((height, width))
    a, b = random.choice(((1, 1), (1, 2), (2, 1), (2, 2)))
    phase = random.randrange(len(colors))
    palette = np.array(colors, dtype=int)
    return palette[(a * rr + b * cc + phase) % len(colors)]


def _pattern_diagonal(height: int, width: int, colors: list[int]) -> np.ndarray:
    rr, cc = np.indices((height, width))
    base = rr + cc if random.random() < 0.5 else rr - cc
    phase = random.randrange(len(colors))
    palette = np.array(colors, dtype=int)
    return palette[(base - base.min() + phase) % len(colors)]


def _pattern_blocks(height: int, width: int, colors: list[int]) -> np.ndarray:
    row_cut_count = random.randint(1, min(2, height - 1))
    col_cut_count = random.randint(1, min(2, width - 1))
    row_cuts = random.sample(range(1, height), row_cut_count)
    col_cuts = random.sample(range(1, width), col_cut_count)
    row_spans = _intervals(height, row_cuts)
    col_spans = _intervals(width, col_cuts)

    palette = np.array(colors, dtype=int)
    grid = np.empty((height, width), dtype=int)
    phase = random.randrange(len(colors))
    for i, (r0, r1) in enumerate(row_spans):
        for j, (c0, c1) in enumerate(col_spans):
            grid[r0:r1, c0:c1] = palette[(i + 2 * j + phase) % len(colors)]
    return grid


def _pattern_rectangles(height: int, width: int, colors: list[int]) -> np.ndarray:
    grid = np.full((height, width), colors[0], dtype=int)
    for _ in range(random.randint(2, 4)):
        r0 = random.randint(0, height - 1)
        r1 = random.randint(r0, height - 1)
        c0 = random.randint(0, width - 1)
        c1 = random.randint(c0, width - 1)
        grid[r0 : r1 + 1, c0 : c1 + 1] = random.choice(colors[1:])
    return grid


def _make_mother_pattern(height: int, width: int) -> np.ndarray:
    families = (
        _pattern_row_bands,
        _pattern_col_bands,
        _pattern_checker,
        _pattern_diagonal,
        _pattern_blocks,
        _pattern_rectangles,
    )
    for _ in range(100):
        color_count = random.randint(2, min(4, len(MOTHER_COLORS)))
        colors = random.sample(MOTHER_COLORS, color_count)
        grid = random.choice(families)(height, width, colors)
        if np.unique(grid).size >= 2:
            return grid
    raise RuntimeError("failed to build a multi-color mother pattern")


def _make_selector(height: int, width: int, rows: list[int], cols: list[int]) -> np.ndarray:
    selector = np.full((height, width), BLACK, dtype=int)
    style = random.choice(("intersections", "sparse"))
    if style == "intersections":
        selector[np.ix_(rows, cols)] = YELLOW
        return selector

    marks: set[tuple[int, int]] = set()
    for r in rows:
        marks.add((int(r), random.choice(cols)))
    for c in cols:
        marks.add((random.choice(rows), int(c)))
    for r in rows:
        for c in cols:
            if random.random() < 0.35:
                marks.add((int(r), int(c)))
    for r, c in marks:
        selector[r, c] = YELLOW
    return selector


def _build_input_grid(
    cells: list[np.ndarray],
    panel_rows: int,
    panel_cols: int,
    cell_height: int,
    cell_width: int,
    bordered: bool,
) -> np.ndarray:
    height = panel_rows * cell_height + (panel_rows - 1) + 2 * int(bordered)
    width = panel_cols * cell_width + (panel_cols - 1) + 2 * int(bordered)
    grid = np.full((height, width), BLACK, dtype=int)

    if bordered:
        grid[0, :] = GREY
        grid[-1, :] = GREY
        grid[:, 0] = GREY
        grid[:, -1] = GREY

    for r in range(1, panel_rows):
        sep = int(bordered) + r * cell_height + (r - 1)
        grid[sep, :] = GREY
    for c in range(1, panel_cols):
        sep = int(bordered) + c * cell_width + (c - 1)
        grid[:, sep] = GREY

    idx = 0
    for r in range(panel_rows):
        r0 = int(bordered) + r * (cell_height + 1)
        for c in range(panel_cols):
            c0 = int(bordered) + c * (cell_width + 1)
            grid[r0 : r0 + cell_height, c0 : c0 + cell_width] = cells[idx]
            idx += 1
    return grid


def _gen_attempt() -> np.ndarray | None:
    panel_rows, panel_cols = random.choice(LAYOUT_CHOICES)
    cell_height = random.randint(*CELL_HEIGHT_RANGE)
    cell_width = random.randint(*CELL_WIDTH_RANGE)
    bordered = random.random() < OUTER_FRAME_PROB

    panel_count = panel_rows * panel_cols
    selector_index = random.randrange(panel_count)

    for _ in range(50):
        row_count = random.randint(1, cell_height)
        col_count = random.randint(1, cell_width)
        if (row_count < cell_height or col_count < cell_width) and row_count * col_count >= 2:
            break
    else:
        return None

    selected_rows = sorted(random.sample(range(cell_height), row_count))
    selected_cols = sorted(random.sample(range(cell_width), col_count))
    selector = _make_selector(cell_height, cell_width, selected_rows, selected_cols)

    mothers: list[np.ndarray] = []
    seen_patterns: set[bytes] = set()
    for _ in range(panel_count - 1):
        for _ in range(100):
            mother = _make_mother_pattern(cell_height, cell_width)
            key = mother.tobytes()
            if key not in seen_patterns:
                seen_patterns.add(key)
                mothers.append(mother)
                break
        else:
            return None

    pieces = [mother[np.ix_(selected_rows, selected_cols)] for mother in mothers]
    expected = np.concatenate(pieces, axis=1)
    if np.unique(expected).size < 2:
        return None
    if len(mothers) > 1 and len({piece.tobytes() for piece in pieces}) == 1:
        return None

    cells: list[np.ndarray] = []
    mother_index = 0
    for i in range(panel_count):
        if i == selector_index:
            cells.append(selector)
        else:
            cells.append(mothers[mother_index])
            mother_index += 1

    inp = _build_input_grid(cells, panel_rows, panel_cols, cell_height, cell_width, bordered)

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid selector-cropping ARC input")
