from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
SOURCE_COLORS = (1, 2, 3, 4)
MARKER_COLORS = (6, 7, 8, 9)
MAX_GENERATION_RETRIES = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _sorted_sources(grid: np.ndarray, bg: int) -> list[tuple[int, int]]:
    sources = [(0, int(c)) for c in np.flatnonzero(grid[0] != bg)]
    assert len(sources) >= 2, "expected at least two top-row sources"
    return sorted(sources, key=lambda rc: rc[1])


def _region_bounds(source_cols: list[int], width: int) -> list[tuple[int, int]]:
    boundaries = [-1, *source_cols, width]
    return [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]


def _region_markers(grid: np.ndarray, bg: int, left: int, right: int) -> list[tuple[int, int]]:
    coords = np.argwhere(grid != bg)
    out = []
    for r, c in coords:
        rr, cc = int(r), int(c)
        if rr == 0:
            continue
        if left < cc < right:
            out.append((rr, cc))
    return out


def _region_sequence(grid: np.ndarray, coords: list[tuple[int, int]], left: int, right: int) -> list[int]:
    if not coords:
        return []

    row_min = min(r for r, _ in coords)
    region_width = right - left - 1
    assert region_width >= 1, "each region must have positive width"

    ranked = []
    for r, c in coords:
        local_r = r - row_min
        local_c = c - (left + 1)
        linear_index = local_r * region_width + local_c
        snake_row = linear_index // 3
        snake_col = linear_index % 3
        ranked.append(
            (
                snake_row,
                snake_col if snake_row % 2 == 0 else -snake_col,
                linear_index,
                int(grid[r, c]),
            )
        )

    ranked.sort()
    return [color for *_prefix, color in ranked]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress scattered markers region-by-region using top-row ray partitions and a 3-column snake order.

    Input structure:
    - A black background grid has 2 or more distinct source pixels on the top row.
    - Each source defines a vertical ray boundary through its column; the remaining foreground pixels are scattered singleton markers below the top row.
    - Markers lie strictly inside the vertical strips formed by the ordered source columns.

    Rule:
    - Sort the top-row sources from left to right and use their columns as vertical partition boundaries.
    - For each left-to-right partition, collect the markers inside that strip, convert each marker's local strip position into a row-major index over the strip width, wrap those indices into a virtual 3-column grid, and read the occupied virtual cells in boustrophedon order.
    - Concatenate the per-partition color sequences from left to right and return them as a compact N x 1 column grid.

    Key insight:
    - The source columns do not appear in the output; they only decide strip membership and strip width, and that width changes how sparse markers wrap into the virtual 3-column snake order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    sources = _sorted_sources(g, bg)
    source_cols = [c for _, c in sources]
    assert len(set(source_cols)) == len(source_cols), "source columns must be distinct"

    sequence: list[int] = []
    for left, right in _region_bounds(source_cols, g.shape[1]):
        markers = _region_markers(g, bg, left, right)
        sequence.extend(_region_sequence(g, markers, left, right))

    assert sequence, "expected at least one scattered marker"
    return np.asarray(sequence, dtype=int).reshape(-1, 1)


def _neighbors8(cell: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = cell
    return {
        (r + dr, c + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if not (dr == 0 and dc == 0)
    }


def _partition_counts(total: int, buckets: int, minimum: int) -> list[int]:
    counts = [minimum] * buckets
    for _ in range(total - buckets * minimum):
        counts[random.randrange(buckets)] += 1
    random.shuffle(counts)
    return counts


def _choose_widths(source_count: int, canvas_width: int) -> list[int]:
    region_count = source_count + 1
    minima = [1] + [2] * (region_count - 2) + [1]
    widths = minima[:]
    extra = canvas_width - source_count - sum(minima)
    assert extra >= 0
    for _ in range(extra):
        widths[random.randrange(region_count)] += 1
    return widths


def _widths_to_sources(widths: list[int]) -> list[int]:
    cols = []
    col = widths[0]
    cols.append(col)
    for width in widths[1:-1]:
        col += 1 + width
        cols.append(col)
    return cols


def _can_place(cell: tuple[int, int], used: set[tuple[int, int]]) -> bool:
    r, c = cell
    return all(max(abs(r - rr), abs(c - cc)) > 1 for rr, cc in used)


def _special_pattern(left: int, width: int, top_row: int) -> list[tuple[int, int]]:
    local = [(0, width - 1), (1, 0), (2, 2)]
    return [(top_row + dr, left + 1 + dc) for dr, dc in local]


def _place_region_markers(
    height: int,
    left: int,
    right: int,
    count: int,
    used: set[tuple[int, int]],
    force_special: bool,
) -> list[tuple[int, int]] | None:
    width = right - left - 1
    region_cells: list[tuple[int, int]] = []

    if force_special:
        if width not in (4, 5):
            return None
        top_row = random.randint(2, height - 3)
        special = _special_pattern(left, width, top_row)
        if any(not _can_place(cell, used) for cell in special):
            return None
        region_cells.extend(special)
        used.update(special)

    candidates = [(r, c) for r in range(2, height) for c in range(left + 1, right)]
    random.shuffle(candidates)
    for cell in candidates:
        if len(region_cells) == count:
            break
        if cell in used or not _can_place(cell, used):
            continue
        region_cells.append(cell)
        used.add(cell)

    if len(region_cells) != count:
        for cell in region_cells:
            used.discard(cell)
        return None
    return region_cells


def _validate_input(grid: np.ndarray) -> bool:
    bg = _get_bg(grid)
    sources = _sorted_sources(grid, bg)
    source_cols = [c for _, c in sources]
    source_colors = [int(grid[0, c]) for c in source_cols]

    if len(source_cols) not in (2, 3):
        return False
    if len(set(source_colors)) != len(source_colors):
        return False
    if any(b - a < 3 for a, b in zip(source_cols, source_cols[1:])):
        return False

    marker_coords = [tuple(map(int, rc)) for rc in np.argwhere(grid != bg) if int(rc[0]) != 0]
    if not (6 <= len(marker_coords) <= 9):
        return False
    if any(r == 0 or c in source_cols for r, c in marker_coords):
        return False
    for i, (r0, c0) in enumerate(marker_coords):
        for r1, c1 in marker_coords[i + 1 :]:
            if max(abs(r0 - r1), abs(c0 - c1)) <= 1:
                return False

    per_region = []
    for left, right in _region_bounds(source_cols, grid.shape[1]):
        per_region.append(len(_region_markers(grid, bg, left, right)))
    return all(count >= 2 for count in per_region)


def _gen_attempt() -> np.ndarray | None:
    source_count = 2 if random.random() < 0.6 else 3
    height = random.randint(10, 14)
    width = random.randint(11 if source_count == 2 else 12, 14)

    region_count = source_count + 1
    marker_total = random.randint(7, 9) if source_count == 2 else 9
    marker_counts = _partition_counts(marker_total, region_count, 2)
    region_widths = _choose_widths(source_count, width)

    special_candidates = [
        idx for idx, (count, strip_width) in enumerate(zip(marker_counts, region_widths)) if count >= 3 and strip_width in (4, 5)
    ]
    if not special_candidates:
        return None
    special_region = random.choice(special_candidates)

    source_cols = _widths_to_sources(region_widths)
    source_colors = random.sample(SOURCE_COLORS, source_count)
    grid = np.zeros((height, width), dtype=int)
    for c, color in zip(source_cols, source_colors):
        grid[0, c] = color

    used = set((0, c) for c in source_cols)
    bounds = _region_bounds(source_cols, width)
    marker_positions: list[tuple[int, int]] = []
    for idx, ((left, right), count) in enumerate(zip(bounds, marker_counts)):
        placed = _place_region_markers(
            height,
            left,
            right,
            count,
            used,
            force_special=(idx == special_region),
        )
        if placed is None:
            return None
        marker_positions.extend(placed)

    random.shuffle(marker_positions)
    marker_colors = [random.choice(MARKER_COLORS) for _ in marker_positions]
    for (r, c), color in zip(marker_positions, marker_colors):
        grid[r, c] = color

    if not _validate_input(grid):
        return None

    out = main(grid)
    if out.shape != (len(marker_positions), 1):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for arc task 2")
