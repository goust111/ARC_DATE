from __future__ import annotations

from collections import Counter
import random

import numpy as np


Coord = tuple[int, int]
Span = tuple[int, int]

PALETTE = tuple(range(10))
MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    """Adapted from arc_utils.get_bg: return the most common grid color."""
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _uniform_row_mask(grid: np.ndarray) -> np.ndarray:
    return np.all(grid == grid[:, :1], axis=1)


def _uniform_col_mask(grid: np.ndarray) -> np.ndarray:
    return np.all(grid == grid[:1, :], axis=0)


def _non_separator_spans(mask: np.ndarray) -> list[Span]:
    spans: list[Span] = []
    idx = 0
    while idx < len(mask):
        if mask[idx]:
            idx += 1
            continue
        start = idx
        while idx < len(mask) and not mask[idx]:
            idx += 1
        spans.append((start, idx - 1))
    return spans


def _parse_partition(grid: np.ndarray) -> tuple[int, list[Span], list[Span], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    row_mask = _uniform_row_mask(g)
    col_mask = _uniform_col_mask(g)
    assert row_mask.any() and col_mask.any(), "expected both horizontal and vertical separators"

    sep_colors = set(map(int, g[row_mask, 0])) | set(map(int, g[0, col_mask]))
    assert len(sep_colors) == 1, "all separators must share one color"
    sep_color = next(iter(sep_colors))

    assert bool(np.all(g[row_mask] == sep_color)), "separator rows must be uniform"
    assert bool(np.all(g[:, col_mask] == sep_color)), "separator columns must be uniform"

    row_spans = _non_separator_spans(row_mask)
    col_spans = _non_separator_spans(col_mask)
    assert len(row_spans) >= 2 and len(col_spans) >= 2, "expected an m x n partition with m,n >= 2"

    logical = np.zeros((len(row_spans), len(col_spans)), dtype=int)
    for rr, (r0, r1) in enumerate(row_spans):
        for cc, (c0, c1) in enumerate(col_spans):
            block = g[r0 : r1 + 1, c0 : c1 + 1]
            color = int(block[0, 0])
            assert color != sep_color, "region colors must differ from the separator"
            assert bool(np.all(block == color)), "each partition cell must be monochrome"
            logical[rr, cc] = color

    return sep_color, row_spans, col_spans, logical


def _nearest_logical_corner(shape: tuple[int, int], cell: Coord) -> Coord:
    rows, cols = shape
    r, c = cell
    corners = [(0, 0), (0, cols - 1), (rows - 1, 0), (rows - 1, cols - 1)]
    distances = [abs(r - cr) + abs(c - cc) for cr, cc in corners]
    nearest = min(distances)
    assert distances.count(nearest) == 1, "expected a unique nearest logical corner"
    return corners[distances.index(nearest)]


def _unique_color_cell(logical: np.ndarray) -> tuple[int, Coord]:
    counts = Counter(map(int, logical.ravel()))
    unique_colors = [color for color, count in counts.items() if count == 1]
    assert len(unique_colors) == 1, "expected exactly one unique region color"
    unique_color = int(unique_colors[0])
    coords = np.argwhere(logical == unique_color)
    assert coords.shape[0] == 1, "expected the unique color to occupy exactly one logical cell"
    r, c = map(int, coords[0])
    return unique_color, (r, c)


def _render_output(
    grid: np.ndarray,
    row_spans: list[Span],
    col_spans: list[Span],
    unique_color: int,
    unique_cell: Coord,
    corner: Coord,
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    r0, r1 = sorted((unique_cell[0], corner[0]))
    c0, c1 = sorted((unique_cell[1], corner[1]))
    for rr in range(r0, r1 + 1):
        rs0, rs1 = row_spans[rr]
        for cc in range(c0, c1 + 1):
            cs0, cs1 = col_spans[cc]
            out[rs0 : rs1 + 1, cs0 : cs1 + 1] = unique_color
    return out


def _solve_spec(grid: np.ndarray) -> np.ndarray:
    sep_color, row_spans, col_spans, logical = _parse_partition(grid)
    _ = sep_color
    unique_color, unique_cell = _unique_color_cell(logical)
    corner = _nearest_logical_corner(logical.shape, unique_cell)
    return _render_output(grid, row_spans, col_spans, unique_color, unique_cell, corner)


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Draw the inclusive rectangle from the unique logical region to its nearest logical corner, then expand it back to the original partitioned canvas.

    Input structure:
    - Full-row and full-column separator bars partition the canvas into an m x n grid of monochrome rectangular regions.
    - Exactly one region color appears only once in that logical region grid; every other region color appears at least twice.
    - Region widths and heights are irregular, so pixel geometry can disagree with logical region distances.

    Rule:
    - Compress the partitioned canvas into its logical m x n region grid.
    - Locate the unique-color logical cell and find the uniquely nearest logical corner cell by Manhattan distance in region steps.
    - Recolor every region inside the inclusive logical rectangle between that cell and the corner to the unique color, leaving separators and all other regions unchanged.

    Key insight:
    - The target rectangle is defined in the region coordinate system, not by raw pixel distances on the unevenly scaled canvas.
    """
    return _solve_spec(np.asarray(input_grid, dtype=int))


def _separator_groups(mask: np.ndarray) -> list[Span]:
    groups: list[Span] = []
    idx = 0
    while idx < len(mask):
        if not mask[idx]:
            idx += 1
            continue
        start = idx
        while idx < len(mask) and mask[idx]:
            idx += 1
        groups.append((start, idx - 1))
    return groups


def _sample_positive_sizes(count: int, limit: int, min_size: int, max_size: int) -> list[int] | None:
    sizes = [random.randint(min_size, max_size) for _ in range(count)]
    if sum(sizes) > limit:
        return None
    slack = limit - sum(sizes)
    order = list(range(count))
    random.shuffle(order)
    for idx in order:
        if slack <= 0:
            break
        room = max_size - sizes[idx]
        if room <= 0:
            continue
        bump = random.randint(0, min(room, slack))
        sizes[idx] += bump
        slack -= bump
    return sizes


def _candidate_unique_cells(rows: int, cols: int) -> dict[str, list[Coord]]:
    candidates = {"interior": [], "edge": []}
    for r in range(rows):
        for c in range(cols):
            if (r, c) in {(0, 0), (0, cols - 1), (rows - 1, 0), (rows - 1, cols - 1)}:
                continue
            corners = [(0, 0), (0, cols - 1), (rows - 1, 0), (rows - 1, cols - 1)]
            dists = [abs(r - cr) + abs(c - cc) for cr, cc in corners]
            if dists.count(min(dists)) != 1:
                continue
            if 0 < r < rows - 1 and 0 < c < cols - 1:
                candidates["interior"].append((r, c))
            else:
                candidates["edge"].append((r, c))
    return candidates


def _pick_unique_cell(rows: int, cols: int) -> Coord | None:
    grouped = _candidate_unique_cells(rows, cols)
    weighted: list[tuple[str, float]] = []
    if grouped["interior"]:
        weighted.append(("interior", 0.6))
    if grouped["edge"]:
        weighted.append(("edge", 0.4))
    if not weighted:
        return None
    labels, weights = zip(*weighted)
    chosen_group = random.choices(labels, weights=weights, k=1)[0]
    return random.choice(grouped[chosen_group])


def _corner_center_distances(
    heights: list[int],
    widths: list[int],
    sep_h: int,
    sep_w: int,
    cell: Coord,
) -> list[float]:
    row_offsets = []
    cursor = 0
    for idx, size in enumerate(heights):
        row_offsets.append(cursor)
        cursor += size
        if idx < len(heights) - 1:
            cursor += sep_h

    col_offsets = []
    cursor = 0
    for idx, size in enumerate(widths):
        col_offsets.append(cursor)
        cursor += size
        if idx < len(widths) - 1:
            cursor += sep_w

    h_total = row_offsets[-1] + heights[-1]
    w_total = col_offsets[-1] + widths[-1]
    r, c = cell
    center_r = row_offsets[r] + (heights[r] - 1) / 2.0
    center_c = col_offsets[c] + (widths[c] - 1) / 2.0
    corners = [(0.0, 0.0), (0.0, w_total - 1.0), (h_total - 1.0, 0.0), (h_total - 1.0, w_total - 1.0)]
    return [abs(center_r - rr) + abs(center_c - cc) for rr, cc in corners]


def _pixel_decoy_exists(
    heights: list[int],
    widths: list[int],
    sep_h: int,
    sep_w: int,
    cell: Coord,
    logical_shape: tuple[int, int],
) -> bool:
    logical_corner = _nearest_logical_corner(logical_shape, cell)
    corners = [(0, 0), (0, logical_shape[1] - 1), (logical_shape[0] - 1, 0), (logical_shape[0] - 1, logical_shape[1] - 1)]
    center_dists = _corner_center_distances(heights, widths, sep_h, sep_w, cell)
    nearest = min(center_dists)
    if center_dists.count(nearest) != 1:
        return False
    pixel_corner = corners[center_dists.index(nearest)]
    return pixel_corner != logical_corner


def _random_partition_sizes(rows: int, cols: int, decoy_required: bool, cell: Coord) -> tuple[list[int], list[int], int, int] | None:
    for _ in range(120):
        sep_h = random.randint(1, 2)
        sep_w = random.randint(1, 2)
        max_height_sum = 30 - (rows - 1) * sep_h
        max_width_sum = 30 - (cols - 1) * sep_w
        if max_height_sum < rows or max_width_sum < cols:
            continue

        heights = _sample_positive_sizes(rows, max_height_sum, 1, 8)
        widths = _sample_positive_sizes(cols, max_width_sum, 1, 8)
        if heights is None or widths is None:
            continue

        if decoy_required and not _pixel_decoy_exists(heights, widths, sep_h, sep_w, cell, (rows, cols)):
            continue
        return heights, widths, sep_h, sep_w
    return None


def _random_composition(total: int, parts: int, minimum: int) -> list[int] | None:
    if total < parts * minimum:
        return None
    values = [minimum] * parts
    for _ in range(total - parts * minimum):
        values[random.randrange(parts)] += 1
    random.shuffle(values)
    return values


def _assign_logical_colors(rows: int, cols: int, unique_cell: Coord) -> tuple[np.ndarray, int, int]:
    total = rows * cols
    repeat_color_count = random.randint(2, min(4, (total - 1) // 2))
    counts = _random_composition(total - 1, repeat_color_count, 2)
    assert counts is not None

    chosen = random.sample(PALETTE, repeat_color_count + 2)
    sep_color = chosen[0]
    unique_color = chosen[1]
    repeated_colors = chosen[2:]

    logical = np.zeros((rows, cols), dtype=int)
    logical[unique_cell] = unique_color

    slots = [(r, c) for r in range(rows) for c in range(cols) if (r, c) != unique_cell]
    random.shuffle(slots)

    cursor = 0
    for color, count in zip(repeated_colors, counts):
        for _ in range(count):
            logical[slots[cursor]] = color
            cursor += 1

    return logical, sep_color, unique_color


def _render_input_from_logical(logical: np.ndarray, sep_color: int, heights: list[int], widths: list[int], sep_h: int, sep_w: int) -> np.ndarray:
    rows, cols = logical.shape
    h_total = sum(heights) + (rows - 1) * sep_h
    w_total = sum(widths) + (cols - 1) * sep_w
    grid = np.full((h_total, w_total), sep_color, dtype=int)

    row_start = 0
    for rr, height in enumerate(heights):
        col_start = 0
        for cc, width in enumerate(widths):
            grid[row_start : row_start + height, col_start : col_start + width] = logical[rr, cc]
            col_start += width
            if cc < cols - 1:
                col_start += sep_w
        row_start += height
        if rr < rows - 1:
            row_start += sep_h
    return grid


def _sample_problem_instance() -> np.ndarray | None:
    rows = random.randint(2, 5)
    cols = random.randint(2, 5)
    if rows * cols < 5:
        return None

    unique_cell = _pick_unique_cell(rows, cols)
    if unique_cell is None:
        return None
    corner = _nearest_logical_corner((rows, cols), unique_cell)
    if unique_cell == corner:
        return None

    decoy_required = random.random() < 0.55
    partition = _random_partition_sizes(rows, cols, decoy_required, unique_cell)
    if partition is None:
        return None
    heights, widths, sep_h, sep_w = partition

    logical, sep_color, unique_color = _assign_logical_colors(rows, cols, unique_cell)
    grid = _render_input_from_logical(logical, sep_color, heights, widths, sep_h, sep_w)

    output = _solve_spec(grid)
    if output.shape != grid.shape:
        return None
    if np.array_equal(output, grid):
        return None

    sep_color2, row_spans, col_spans, parsed_logical = _parse_partition(grid)
    if sep_color2 != sep_color or not np.array_equal(parsed_logical, logical):
        return None

    if unique_color != _unique_color_cell(parsed_logical)[0]:
        return None

    rr0, rr1 = sorted((unique_cell[0], corner[0]))
    cc0, cc1 = sorted((unique_cell[1], corner[1]))
    if rr0 == rr1 and cc0 == cc1:
        return None

    changed_cells = 0
    for rr, (r0, r1) in enumerate(row_spans):
        for cc, (c0, c1) in enumerate(col_spans):
            input_block = grid[r0 : r1 + 1, c0 : c1 + 1]
            output_block = output[r0 : r1 + 1, c0 : c1 + 1]
            changed = not np.array_equal(grid[r0 : r1 + 1, c0 : c1 + 1], output[r0 : r1 + 1, c0 : c1 + 1])
            in_rect = rr0 <= rr <= rr1 and cc0 <= cc <= cc1
            if in_rect:
                if not np.all(output_block == unique_color):
                    return None
            else:
                if changed:
                    return None
            if changed:
                changed_cells += 1
    if changed_cells < 1:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _sample_problem_instance()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid compressed-rectangle ARC input")
