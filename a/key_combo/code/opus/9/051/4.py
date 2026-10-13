from __future__ import annotations

import random

import numpy as np


BG = 0
SEPARATOR = 4
MARKER = 3
OUTPUT_COLOR = 1

COUNT_GRID_SIZE = 3
SUBBLOCK_SIZE = 3
OUTPUT_SHAPE = (COUNT_GRID_SIZE * SUBBLOCK_SIZE, COUNT_GRID_SIZE * SUBBLOCK_SIZE)

REGION_SIZE_CHOICES = (4, 5, 6)
MAX_ATTEMPTS = 500


def _bands(size: int, separators: list[int]) -> list[tuple[int, int]]:
    bounds = [-1, *separators, size]
    bands = [(left + 1, right) for left, right in zip(bounds, bounds[1:])]
    assert all(start < stop for start, stop in bands), "separator lines must split non-empty regions"
    return bands


def _parse_grid(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]], int, int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)

    colors = sorted(int(v) for v in np.unique(g))
    assert BG in colors, "expected black background"
    foreground = [color for color in colors if color != BG]
    assert len(foreground) == 2, "expected one separator color and one marker color"

    candidates = []
    for color in foreground:
        sep_rows = [r for r in range(height) if np.all(g[r, :] == color)]
        sep_cols = [c for c in range(width) if np.all(g[:, c] == color)]
        if sep_rows and sep_cols:
            candidates.append((color, sep_rows, sep_cols))

    assert len(candidates) == 1, "expected exactly one separator color spanning full rows and columns"
    separator_color, sep_rows, sep_cols = candidates[0]
    assert separator_color == SEPARATOR, "expected yellow separators"
    assert len(sep_rows) == 2 and len(sep_cols) == 2, "expected two separator rows and two separator columns"

    marker_colors = [color for color in foreground if color != separator_color]
    assert len(marker_colors) == 1 and marker_colors[0] == MARKER, "expected green marker pixels"

    row_bands = _bands(height, sep_rows)
    col_bands = _bands(width, sep_cols)
    assert len(row_bands) == 3 and len(col_bands) == 3, "expected a 3x3 partition"
    return row_bands, col_bands, separator_color, marker_colors[0]


def _count_regions(grid: np.ndarray) -> np.ndarray:
    row_bands, col_bands, _separator_color, marker_color = _parse_grid(grid)
    counts = np.zeros((COUNT_GRID_SIZE, COUNT_GRID_SIZE), dtype=int)
    allowed = {BG, marker_color}

    for out_r, (r0, r1) in enumerate(row_bands):
        for out_c, (c0, c1) in enumerate(col_bands):
            region = grid[r0:r1, c0:c1]
            assert set(map(int, region.reshape(-1))) <= allowed, "regions may contain only black background and green markers"
            marker_count = int(np.count_nonzero(region == marker_color))
            assert 0 <= marker_count <= 3, "each region must contain between zero and three markers"
            counts[out_r, out_c] = marker_count
    return counts


def _encode_counts(counts: np.ndarray) -> np.ndarray:
    counts = np.asarray(counts, dtype=int)
    assert counts.shape == (COUNT_GRID_SIZE, COUNT_GRID_SIZE)

    out = np.full(OUTPUT_SHAPE, BG, dtype=int)
    for r in range(COUNT_GRID_SIZE):
        for c in range(COUNT_GRID_SIZE):
            count = int(counts[r, c])
            assert 0 <= count <= 3
            base_r = r * SUBBLOCK_SIZE
            base_c = c * SUBBLOCK_SIZE
            for k in range(count):
                out[base_r + k, base_c + k] = OUTPUT_COLOR
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Summarize separator-delimited marker counts as diagonal codes in a 3x3 block grid.

    Input structure:
    - The background is black.
    - Yellow full-width rows and full-height columns form a 3x3 partition of the grid.
    - Each rectangular region contains only black cells plus 0-3 isolated green marker pixels.

    Rule:
    - Identify the two yellow separator rows and the two yellow separator columns, then use them to split the grid into 9 regions.
    - Count the green marker pixels inside each region.
    - Return a black 9x9 grid made of nine 3x3 sub-blocks; in sub-block (i,j), fill the first C(i,j) cells of the main diagonal with blue.

    Key insight:
    - The separators define which markers belong together; the output ignores marker positions inside a region and preserves only each region's count via diagonal length.
    """
    g = np.asarray(grid, dtype=int)
    return _encode_counts(_count_regions(g))


def _sample_axis_sizes() -> list[int]:
    profile = random.choices(
        population=("equal", "two_same", "all_distinct"),
        weights=(1, 3, 3),
        k=1,
    )[0]
    if profile == "equal":
        size = random.choice(REGION_SIZE_CHOICES)
        return [size, size, size]
    if profile == "two_same":
        repeated = random.choice(REGION_SIZE_CHOICES)
        other = random.choice([value for value in REGION_SIZE_CHOICES if value != repeated])
        sizes = [repeated, repeated, other]
        random.shuffle(sizes)
        return sizes
    sizes = list(REGION_SIZE_CHOICES)
    random.shuffle(sizes)
    return sizes


def _region_bounds(row_sizes: list[int], col_sizes: list[int]) -> list[tuple[int, int, int, int]]:
    row_starts = [0, row_sizes[0] + 1, row_sizes[0] + row_sizes[1] + 2]
    col_starts = [0, col_sizes[0] + 1, col_sizes[0] + col_sizes[1] + 2]
    bounds = []
    for r in range(COUNT_GRID_SIZE):
        for c in range(COUNT_GRID_SIZE):
            bounds.append((row_starts[r], col_starts[c], row_sizes[r], col_sizes[c]))
    return bounds


def _area_condition(row_sizes: list[int], col_sizes: list[int], counts: np.ndarray) -> bool:
    areas = np.outer(np.asarray(row_sizes, dtype=int), np.asarray(col_sizes, dtype=int))
    unique_areas = sorted(set(int(v) for v in areas.reshape(-1)))
    same_area_diff_counts = any(
        len(set(int(v) for v in counts[areas == area])) >= 2 for area in unique_areas
    )
    diff_area_same_count = any(
        len(set(int(area) for area in areas[counts == count])) >= 2 for count in range(4) if np.any(counts == count)
    )
    return same_area_diff_counts or diff_area_same_count


def _sample_counts(row_sizes: list[int], col_sizes: list[int]) -> np.ndarray | None:
    for _ in range(200):
        values = [0, 1, 2, 3]
        values.extend(random.randint(0, 3) for _ in range(5))
        random.shuffle(values)
        counts = np.asarray(values, dtype=int).reshape(COUNT_GRID_SIZE, COUNT_GRID_SIZE)

        if set(map(int, counts.reshape(-1))) != {0, 1, 2, 3}:
            continue
        if any(len(set(int(v) for v in counts[r, :])) < 2 for r in range(COUNT_GRID_SIZE)):
            continue
        if any(len(set(int(v) for v in counts[:, c])) < 2 for c in range(COUNT_GRID_SIZE)):
            continue
        if not _area_condition(row_sizes, col_sizes, counts):
            continue
        if np.array_equal(counts, counts.T):
            continue
        return counts
    return None


def _well_spaced_positions(height: int, width: int, count: int) -> tuple[tuple[int, int], ...] | None:
    cells = [(r, c) for r in range(height) for c in range(width)]
    for _ in range(200):
        choice = random.sample(cells, count)
        good = True
        for i, (r0, c0) in enumerate(choice):
            for r1, c1 in choice[i + 1 :]:
                if max(abs(r0 - r1), abs(c0 - c1)) <= 1:
                    good = False
                    break
            if not good:
                break
        if good:
            return tuple(sorted(choice))
    return None


def _sample_layouts(row_sizes: list[int], col_sizes: list[int], counts: np.ndarray) -> dict[tuple[int, int], tuple[tuple[int, int], ...]] | None:
    layouts: dict[tuple[int, int], tuple[tuple[int, int], ...]] = {}
    for r in range(COUNT_GRID_SIZE):
        for c in range(COUNT_GRID_SIZE):
            count = int(counts[r, c])
            layout = _well_spaced_positions(row_sizes[r], col_sizes[c], count)
            if layout is None:
                return None
            layouts[(r, c)] = layout

    varied_repeat = False
    for target_count in (1, 2, 3):
        cells = [(r, c) for r in range(COUNT_GRID_SIZE) for c in range(COUNT_GRID_SIZE) if counts[r, c] == target_count]
        if len(cells) < 2:
            continue
        patterns = {layouts[cell] for cell in cells}
        if len(patterns) >= 2:
            varied_repeat = True
            break
    return layouts if varied_repeat else None


def _build_grid(
    row_sizes: list[int],
    col_sizes: list[int],
    layouts: dict[tuple[int, int], tuple[tuple[int, int], ...]],
) -> np.ndarray:
    height = sum(row_sizes) + 2
    width = sum(col_sizes) + 2
    grid = np.full((height, width), BG, dtype=int)

    sep_rows = [row_sizes[0], row_sizes[0] + row_sizes[1] + 1]
    sep_cols = [col_sizes[0], col_sizes[0] + col_sizes[1] + 1]
    grid[sep_rows, :] = SEPARATOR
    grid[:, sep_cols] = SEPARATOR

    for r in range(COUNT_GRID_SIZE):
        for c in range(COUNT_GRID_SIZE):
            row_start = sum(row_sizes[:r]) + r
            col_start = sum(col_sizes[:c]) + c
            for dr, dc in layouts[(r, c)]:
                grid[row_start + dr, col_start + dc] = MARKER
    return grid


def _relayout_region(
    row_sizes: list[int],
    col_sizes: list[int],
    layouts: dict[tuple[int, int], tuple[tuple[int, int], ...]],
    target_region: tuple[int, int],
) -> dict[tuple[int, int], tuple[tuple[int, int], ...]] | None:
    r, c = target_region
    count = len(layouts[target_region])
    if count == 0:
        return None
    for _ in range(100):
        candidate = _well_spaced_positions(row_sizes[r], col_sizes[c], count)
        if candidate is not None and candidate != layouts[target_region]:
            new_layouts = dict(layouts)
            new_layouts[target_region] = candidate
            return new_layouts
    return None


def _move_marker_between_regions(
    counts: np.ndarray,
    layouts: dict[tuple[int, int], tuple[tuple[int, int], ...]],
    row_sizes: list[int],
    col_sizes: list[int],
) -> tuple[np.ndarray, dict[tuple[int, int], tuple[tuple[int, int], ...]]] | None:
    donor = next(((r, c) for r in range(3) for c in range(3) if counts[r, c] == 3), None)
    receiver = next(((r, c) for r in range(3) for c in range(3) if counts[r, c] == 0), None)
    if donor is None or receiver is None:
        return None

    new_counts = counts.copy()
    new_counts[donor] -= 1
    new_counts[receiver] += 1

    new_layouts = dict(layouts)
    donor_cells = list(layouts[donor])
    donor_cells.pop()
    new_layouts[donor] = tuple(donor_cells)

    receiver_layout = _well_spaced_positions(row_sizes[receiver[0]], col_sizes[receiver[1]], 1)
    if receiver_layout is None:
        return None
    new_layouts[receiver] = receiver_layout
    return new_counts, new_layouts


def _semantic_checks(
    row_sizes: list[int],
    col_sizes: list[int],
    counts: np.ndarray,
    layouts: dict[tuple[int, int], tuple[tuple[int, int], ...]],
) -> bool:
    grid = _build_grid(row_sizes, col_sizes, layouts)
    expected = _encode_counts(counts)
    if not np.array_equal(_count_regions(grid), counts):
        return False
    if not np.array_equal(main(grid), expected):
        return False

    relayout_target = next(
        (
            (r, c)
            for r in range(3)
            for c in range(3)
            if counts[r, c] > 0 and sum(counts.reshape(-1) == counts[r, c]) >= 2
        ),
        None,
    )
    if relayout_target is None:
        return False
    alt_layouts = _relayout_region(row_sizes, col_sizes, layouts, relayout_target)
    if alt_layouts is None:
        return False
    alt_grid = _build_grid(row_sizes, col_sizes, alt_layouts)
    if np.array_equal(alt_grid, grid):
        return False
    if not np.array_equal(main(alt_grid), expected):
        return False

    moved = _move_marker_between_regions(counts, layouts, row_sizes, col_sizes)
    if moved is None:
        return False
    moved_counts, moved_layouts = moved
    moved_grid = _build_grid(row_sizes, col_sizes, moved_layouts)
    moved_expected = _encode_counts(moved_counts)
    if np.array_equal(main(moved_grid), expected):
        return False
    if not np.array_equal(main(moved_grid), moved_expected):
        return False
    return True


def _attempt_generate() -> np.ndarray | None:
    row_sizes = _sample_axis_sizes()
    col_sizes = _sample_axis_sizes()
    counts = _sample_counts(row_sizes, col_sizes)
    if counts is None:
        return None
    layouts = _sample_layouts(row_sizes, col_sizes, counts)
    if layouts is None:
        return None
    if not _semantic_checks(row_sizes, col_sizes, counts, layouts):
        return None
    return _build_grid(row_sizes, col_sizes, layouts)


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
