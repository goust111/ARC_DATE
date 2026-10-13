from __future__ import annotations

from collections import deque
import random

import numpy as np


Interval = tuple[int, int]

_COLORS = tuple(range(10))
_LAYOUTS = (
    (2, 2),
    (2, 3),
    (3, 2),
    (3, 3),
    (2, 4),
    (4, 2),
    (3, 4),
    (4, 3),
    (4, 4),
)
_MAX_GENERATION_ATTEMPTS = 300


def _runs(indices: list[int]) -> list[Interval]:
    """Return inclusive runs of consecutive integer indices."""
    ordered = sorted(set(map(int, indices)))
    if not ordered:
        return []
    result: list[Interval] = []
    start = previous = ordered[0]
    for value in ordered[1:]:
        if value == previous + 1:
            previous = value
        else:
            result.append((start, previous))
            start = previous = value
    result.append((start, previous))
    return result


def _content_intervals(length: int, separators: list[int]) -> list[Interval]:
    """Split one axis at separator runs and validate equal-sized tile spans."""
    spans: list[Interval] = []
    start = 0
    separator_runs = _runs(separators)
    assert separator_runs, "expected at least one separator run"
    assert separator_runs[0][0] > 0 and separator_runs[-1][1] < length - 1, (
        "separator runs must lie between tiles"
    )
    for sep_start, sep_end in separator_runs:
        assert start < sep_start, "empty tile span"
        spans.append((start, sep_start - 1))
        start = sep_end + 1
    assert start < length, "empty final tile span"
    spans.append((start, length - 1))

    sizes = {end - begin + 1 for begin, end in spans}
    assert len(sizes) == 1, "all tiles must have the same size on each axis"
    assert next(iter(sizes)) >= 2, "tiles must contain a visible pattern area"
    return spans


def _tile_view(
    grid: np.ndarray, row_span: Interval, col_span: Interval
) -> np.ndarray:
    r0, r1 = row_span
    c0, c1 = col_span
    return grid[r0 : r1 + 1, c0 : c1 + 1]


def _parse_tiles(
    grid: np.ndarray,
) -> tuple[int, int, list[Interval], list[Interval], np.ndarray]:
    """Recover separator, shared background, tile spans, and tile foregrounds."""
    g = np.asarray(grid)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    height, width = map(int, g.shape)
    assert height <= 30 and width <= 30, "ARC grids must be at most 30 by 30"

    candidates: list[tuple[int, list[int], list[int]]] = []
    for color in set(map(int, g.ravel())):
        rows = [r for r in range(height) if np.all(g[r, :] == color)]
        cols = [c for c in range(width) if np.all(g[:, c] == color)]
        if rows and cols:
            candidates.append((int(color), rows, cols))
    assert len(candidates) == 1, (
        "expected one visible color to form all separator rows and columns"
    )

    separator, separator_rows, separator_cols = candidates[0]
    row_spans = _content_intervals(height, separator_rows)
    col_spans = _content_intervals(width, separator_cols)
    assert len(row_spans) >= 2 and len(col_spans) >= 2, (
        "expected at least a 2 by 2 tile array"
    )

    tile_color_sets: list[set[int]] = []
    for row_span in row_spans:
        for col_span in col_spans:
            colors = set(map(int, _tile_view(g, row_span, col_span).ravel()))
            assert separator not in colors, "separator color may occur only on dividers"
            assert len(colors) == 2, (
                "each tile must contain one shared background and one foreground color"
            )
            tile_color_sets.append(colors)

    common = set.intersection(*tile_color_sets)
    assert len(common) == 1, "the common tile background must be structurally unique"
    background = int(next(iter(common)))
    assert background != separator

    tile_rows, tile_cols = len(row_spans), len(col_spans)
    foregrounds = np.empty((tile_rows, tile_cols), dtype=int)
    index = 0
    for i in range(tile_rows):
        for j in range(tile_cols):
            other = tile_color_sets[index] - {background}
            assert len(other) == 1
            foregrounds[i, j] = int(next(iter(other)))
            index += 1
    return separator, background, row_spans, col_spans, foregrounds


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace every tile by the occupancy complement of its point-symmetric partner tile.

    Input structure:
    - Full rows and columns of one separator color divide the canvas into a regular
      rectangular tile array with equal tile heights and equal tile widths.
    - Every tile contains one structurally common background color and exactly one
      foreground color; foreground colors may differ between tiles.

    Rule:
    - Keep the canvas size and all separator cells unchanged. For array position
      (i,j), use tile (R-1-i,C-1-j) as the source, without changing local tile
      coordinates. Source-foreground cells become background, while source-background
      cells receive that source tile's foreground color.

    Key insight:
    - Point symmetry acts on tile addresses, not on coordinates inside a tile, and
      the transferred occupancy must then be inverted; neither local complementation
      nor point-symmetric transport alone performs both required operations.
    """
    g = np.asarray(grid)
    _, background, row_spans, col_spans, foregrounds = _parse_tiles(g)
    tile_rows, tile_cols = len(row_spans), len(col_spans)
    out = g.copy()

    for i, target_rows in enumerate(row_spans):
        for j, target_cols in enumerate(col_spans):
            source_i = tile_rows - 1 - i
            source_j = tile_cols - 1 - j
            source = _tile_view(g, row_spans[source_i], col_spans[source_j])
            source_foreground = int(foregrounds[source_i, source_j])
            source_occupied = source == source_foreground
            target = _tile_view(out, target_rows, target_cols)
            target[:, :] = np.where(
                source_occupied, background, source_foreground
            )
    return out


def _component_count(mask: np.ndarray) -> int:
    """Count 4-connected True components without an external dependency."""
    height, width = mask.shape
    unseen = {tuple(map(int, point)) for point in np.argwhere(mask)}
    count = 0
    while unseen:
        count += 1
        start = unseen.pop()
        queue = deque([start])
        while queue:
            r, c = queue.popleft()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                point = (r + dr, c + dc)
                if 0 <= point[0] < height and 0 <= point[1] < width and point in unseen:
                    unseen.remove(point)
                    queue.append(point)
    return count


def _random_mask(
    height: int,
    width: int,
    desired_components: int,
    forbidden: list[np.ndarray],
) -> np.ndarray:
    """Make a sparse, asymmetric pattern with a requested component count."""
    area = height * width
    low = max(2, int(np.ceil(0.25 * area)), desired_components)
    high = min(area - 2, int(np.floor(0.48 * area)))
    for _ in range(500):
        count = random.randint(low, high)
        flat = np.zeros(area, dtype=bool)
        flat[random.sample(range(area), count)] = True
        mask = flat.reshape(height, width)
        if _component_count(mask) != desired_components:
            continue
        if np.array_equal(mask, mask[::-1, ::-1]):
            continue
        if any(np.array_equal(mask, old) for old in forbidden):
            continue
        return mask

    # Some very small shapes make three components rare. Preserve all essential
    # anti-shortcut constraints while relaxing only the decorative component count.
    for _ in range(500):
        count = random.randint(low, high)
        flat = np.zeros(area, dtype=bool)
        flat[random.sample(range(area), count)] = True
        mask = flat.reshape(height, width)
        if not np.array_equal(mask, mask[::-1, ::-1]) and not any(
            np.array_equal(mask, old) for old in forbidden
        ):
            return mask
    raise RuntimeError("failed to generate a distinct asymmetric tile mask")


def _make_masks(tile_rows: int, tile_cols: int, height: int, width: int) -> list[list[np.ndarray]]:
    masks: list[list[np.ndarray | None]] = [
        [None for _ in range(tile_cols)] for _ in range(tile_rows)
    ]
    made: list[np.ndarray] = []
    component_cycle = [1, 2, 1, 3]
    component_index = 0

    for i in range(tile_rows):
        for j in range(tile_cols):
            if masks[i][j] is not None:
                continue
            opposite = (tile_rows - 1 - i, tile_cols - 1 - j)
            desired = component_cycle[component_index % len(component_cycle)]
            first = _random_mask(height, width, desired, made)
            component_index += 1
            masks[i][j] = first
            made.append(first)

            if opposite != (i, j):
                desired = component_cycle[component_index % len(component_cycle)]
                second = _random_mask(height, width, desired, made)
                component_index += 1
                assert not np.array_equal(first, second)
                masks[opposite[0]][opposite[1]] = second
                made.append(second)

    return [[mask for mask in row if mask is not None] for row in masks]


def _make_foregrounds(
    tile_rows: int, tile_cols: int, palette: list[int]
) -> np.ndarray:
    """Assign visibly arbitrary colors, with different colors inside each pair."""
    foregrounds = np.full((tile_rows, tile_cols), -1, dtype=int)
    for i in range(tile_rows):
        for j in range(tile_cols):
            if foregrounds[i, j] >= 0:
                continue
            opposite = (tile_rows - 1 - i, tile_cols - 1 - j)
            if opposite == (i, j):
                foregrounds[i, j] = random.choice(palette)
            else:
                first, second = random.sample(palette, 2)
                foregrounds[i, j] = first
                foregrounds[opposite] = second
    return foregrounds


def _axis_spans(count: int, tile_size: int, separator_size: int) -> list[Interval]:
    return [
        (
            index * (tile_size + separator_size),
            index * (tile_size + separator_size) + tile_size - 1,
        )
        for index in range(count)
    ]


def _render(
    masks: list[list[np.ndarray]],
    foregrounds: np.ndarray,
    background: int,
    separator: int,
    separator_size: int,
) -> tuple[np.ndarray, list[Interval], list[Interval]]:
    tile_rows, tile_cols = foregrounds.shape
    tile_height, tile_width = masks[0][0].shape
    height = tile_rows * tile_height + (tile_rows - 1) * separator_size
    width = tile_cols * tile_width + (tile_cols - 1) * separator_size
    grid = np.full((height, width), separator, dtype=int)
    row_spans = _axis_spans(tile_rows, tile_height, separator_size)
    col_spans = _axis_spans(tile_cols, tile_width, separator_size)
    for i, row_span in enumerate(row_spans):
        for j, col_span in enumerate(col_spans):
            tile = _tile_view(grid, row_span, col_span)
            tile[:, :] = background
            tile[masks[i][j]] = int(foregrounds[i, j])
    return grid, row_spans, col_spans


def _independent_expected(
    grid: np.ndarray,
    masks: list[list[np.ndarray]],
    foregrounds: np.ndarray,
    background: int,
    row_spans: list[Interval],
    col_spans: list[Interval],
    *,
    invert: bool,
    rotate_inside_tile: bool = False,
) -> np.ndarray:
    """Metadata-based oracle, independent of the solver's scene parser."""
    out = grid.copy()
    rows, cols = foregrounds.shape
    for i, row_span in enumerate(row_spans):
        for j, col_span in enumerate(col_spans):
            source_i, source_j = rows - 1 - i, cols - 1 - j
            mask = masks[source_i][source_j]
            if rotate_inside_tile:
                mask = mask[::-1, ::-1]
            occupied = ~mask if invert else mask
            tile = _tile_view(out, row_span, col_span)
            tile[:, :] = background
            tile[occupied] = int(foregrounds[source_i, source_j])
    return out


def _local_complement(
    grid: np.ndarray,
    masks: list[list[np.ndarray]],
    foregrounds: np.ndarray,
    background: int,
    row_spans: list[Interval],
    col_spans: list[Interval],
) -> np.ndarray:
    out = grid.copy()
    for i, row_span in enumerate(row_spans):
        for j, col_span in enumerate(col_spans):
            tile = _tile_view(out, row_span, col_span)
            tile[:, :] = background
            tile[~masks[i][j]] = int(foregrounds[i, j])
    return out


def _generation_attempt() -> np.ndarray | None:
    tile_rows, tile_cols = random.choice(_LAYOUTS)
    separator_size = random.choice((1, 2))
    max_tile_height = (30 - (tile_rows - 1) * separator_size) // tile_rows
    max_tile_width = (30 - (tile_cols - 1) * separator_size) // tile_cols
    tile_height = random.randint(3, min(6, max_tile_height))
    tile_width = random.randint(3, min(6, max_tile_width))

    background, separator = random.sample(_COLORS, 2)
    available = [color for color in _COLORS if color not in {background, separator}]
    palette = random.sample(available, random.randint(3, min(6, len(available))))
    foregrounds = _make_foregrounds(tile_rows, tile_cols, palette)
    if len(set(map(int, foregrounds.ravel()))) < 3:
        return None

    masks = _make_masks(tile_rows, tile_cols, tile_height, tile_width)
    grid, row_spans, col_spans = _render(
        masks, foregrounds, background, separator, separator_size
    )
    expected = _independent_expected(
        grid,
        masks,
        foregrounds,
        background,
        row_spans,
        col_spans,
        invert=True,
    )

    # Semantic rejection tests: each tempting one-step interpretation must disagree.
    local_only = _local_complement(
        grid, masks, foregrounds, background, row_spans, col_spans
    )
    transport_only = _independent_expected(
        grid,
        masks,
        foregrounds,
        background,
        row_spans,
        col_spans,
        invert=False,
    )
    pixel_rotation_then_complement = _independent_expected(
        grid,
        masks,
        foregrounds,
        background,
        row_spans,
        col_spans,
        invert=True,
        rotate_inside_tile=True,
    )
    if any(
        np.array_equal(expected, wrong)
        for wrong in (grid, local_only, transport_only, pixel_rotation_then_complement)
    ):
        return None

    try:
        actual = main(grid)
        parsed_separator, parsed_background, parsed_rows, parsed_cols, parsed_fg = _parse_tiles(grid)
    except (AssertionError, ValueError):
        return None
    if parsed_separator != separator or parsed_background != background:
        return None
    if parsed_rows != row_spans or parsed_cols != col_spans:
        return None
    if not np.array_equal(parsed_fg, foregrounds):
        return None
    if not np.array_equal(actual, expected):
        return None
    if not np.array_equal(main(actual), grid):
        return None
    if grid.shape[0] > 30 or grid.shape[1] > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate a varied, independently validated tile-complement instance."""
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        grid = _generation_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid tile point-complement input")
