from __future__ import annotations

import random

import numpy as np


DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
SPIRAL = (
    (0, 0),
    (0, 1),
    (0, 2),
    (1, 2),
    (2, 2),
    (2, 1),
    (2, 0),
    (1, 0),
    (1, 1),
)

# Fixed puzzle colors from the prompt.
BG_COLOR = 0
MARKER_COLOR = 2
SEPARATOR_COLOR = 5

# Generator controls.
GEN_MAX_ATTEMPTS = 400
GEN_REGION_SPANS = (4, 5)
GEN_VALID_COUNT_BAG = (2, 2, 3, 3, 4, 4, 5, 5, 6)
GEN_FORCE_MULTI_COMPONENT = 2


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, np.unique(np.asarray(grid, dtype=int))))
    if exclude:
        colors -= set(map(int, exclude))
    return sorted(colors)


def _bands(length: int, separators: list[int]) -> list[tuple[int, int]]:
    bands: list[tuple[int, int]] = []
    start = 0
    for sep in sorted(map(int, separators)):
        if start < sep:
            bands.append((start, sep))
        start = sep + 1
    if start < int(length):
        bands.append((start, int(length)))
    return bands


def _find_partition_spec(
    grid: np.ndarray,
) -> tuple[int, int, int, list[tuple[int, int]], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"
    height, width = map(int, g.shape)

    bg = _get_bg(g)
    foreground = _colors_present(g, exclude={bg})
    separator_candidates: list[tuple[int, list[int], list[int]]] = []
    for color in foreground:
        full_rows = [r for r in range(height) if np.all(g[r, :] == color)]
        full_cols = [c for c in range(width) if np.all(g[:, c] == color)]
        if full_rows and full_cols:
            separator_candidates.append((int(color), full_rows, full_cols))

    assert len(separator_candidates) == 1, "expected one separator color with full rows and full columns"
    separator_color, sep_rows, sep_cols = separator_candidates[0]
    row_bands = _bands(height, sep_rows)
    col_bands = _bands(width, sep_cols)
    assert len(row_bands) == 3 and len(col_bands) == 3, "expected a 3x3 partitioned board"

    marker_colors = [color for color in foreground if color != separator_color]
    assert len(marker_colors) == 1, "expected exactly one marker color besides the separator"
    marker_color = int(marker_colors[0])
    return bg, separator_color, marker_color, row_bands, col_bands


def _count_region_markers(region: np.ndarray, marker_color: int) -> tuple[int, int]:
    mask = np.asarray(region, dtype=int) == int(marker_color)
    valid = 0
    isolated = 0
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        has_neighbor = any(
            0 <= rr + dr < mask.shape[0] and 0 <= cc + dc < mask.shape[1] and mask[rr + dr, cc + dc]
            for dr, dc in DIRS4
        )
        if has_neighbor:
            valid += 1
        else:
            isolated += 1
    return int(valid), int(isolated)


def _gauge_tile(count: int, bg: int, marker_color: int) -> np.ndarray:
    assert 0 <= int(count) <= len(SPIRAL), "spiral gauge count out of range"
    tile = np.full((3, 3), int(bg), dtype=int)
    for r, c in SPIRAL[: int(count)]:
        tile[r, c] = int(marker_color)
    return tile


def _render_output(count_grid: list[list[int]], bg: int, marker_color: int) -> np.ndarray:
    out = np.full((9, 9), int(bg), dtype=int)
    for gr in range(3):
        for gc in range(3):
            tile = _gauge_tile(count_grid[gr][gc], bg, marker_color)
            out[gr * 3 : (gr + 1) * 3, gc * 3 : (gc + 1) * 3] = tile
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Filter separator-bounded marker pixels by in-region 4-connectivity and encode each region as a spiral gauge.

    Input structure:
    - The background is black, one non-background color forms full separator rows and full separator columns, and the remaining non-background color marks pixels inside the separated regions.
    - The separators partition the board into a 3x3 table of rectangular regions, each containing both isolated markers and markers that are 4-adjacent to another marker within the same region.

    Rule:
    - Identify the full separator rows and columns, then process each of the nine regions independently.
    - Within a region, discard every marker pixel that has no same-colored 4-neighbor inside that region, and count only the remaining non-isolated marker pixels.
    - Replace each region by a 3x3 black gauge whose first V cells along the clockwise spiral from the top-left are red, where V is that filtered count, and arrange the nine gauges in the same row/column order as the input regions.

    Key insight:
    - The task is not raw counting: each region must first be filtered by local 4-connectivity, and only that filtered count is compressed into the canonical 3x3 spiral meter.
    """
    g = np.asarray(grid, dtype=int)
    bg, _separator, marker_color, row_bands, col_bands = _find_partition_spec(g)

    count_grid: list[list[int]] = []
    for r0, r1 in row_bands:
        row_counts: list[int] = []
        for c0, c1 in col_bands:
            region = g[r0:r1, c0:c1]
            valid, _isolated = _count_region_markers(region, marker_color)
            row_counts.append(valid)
        count_grid.append(row_counts)

    return _render_output(count_grid, bg, marker_color)


def _neighbors4(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    return [(r + dr, c + dc) for dr, dc in DIRS4]


def _touches_occupied(cells: set[tuple[int, int]], occupied: set[tuple[int, int]]) -> bool:
    if cells & occupied:
        return True
    for cell in cells:
        for nbr in _neighbors4(cell):
            if nbr in occupied:
                return True
    return False


def _random_connected_shape(height: int, width: int, size: int) -> set[tuple[int, int]]:
    seed = (random.randrange(height), random.randrange(width))
    cells = {seed}
    while len(cells) < int(size):
        frontier: set[tuple[int, int]] = set()
        for r, c in cells:
            for nr, nc in _neighbors4((r, c)):
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        if not frontier:
            break
        cells.add(random.choice(tuple(frontier)))
    return cells


def _cluster_options(valid_count: int) -> list[tuple[int, ...]]:
    options = {
        2: [(2,)],
        3: [(3,)],
        4: [(4,), (2, 2)],
        5: [(5,), (2, 3)],
        6: [(6,), (2, 4), (3, 3)],
    }
    return options[int(valid_count)]


def _choose_cluster_sizes(valid_count: int, force_multi: bool) -> tuple[int, ...]:
    options = _cluster_options(valid_count)
    multi = [option for option in options if len(option) > 1]
    if force_multi and multi:
        return random.choice(multi)
    return random.choice(options)


def _place_region_markers(
    height: int,
    width: int,
    valid_count: int,
    isolated_count: int,
    force_multi: bool,
) -> tuple[set[tuple[int, int]], tuple[int, ...]]:
    for _ in range(120):
        cluster_sizes = _choose_cluster_sizes(valid_count, force_multi)
        occupied: set[tuple[int, int]] = set()
        success = True

        for cluster_size in sorted(cluster_sizes, reverse=True):
            placed = False
            for _ in range(120):
                cells = _random_connected_shape(height, width, cluster_size)
                if len(cells) != cluster_size:
                    continue
                if _touches_occupied(cells, occupied):
                    continue
                occupied |= cells
                placed = True
                break
            if not placed:
                success = False
                break

        if not success:
            continue

        isolated_cells: list[tuple[int, int]] = []
        candidates = [(r, c) for r in range(height) for c in range(width)]
        random.shuffle(candidates)
        for cell in candidates:
            if cell in occupied or cell in isolated_cells:
                continue
            if any(nbr in occupied or nbr in isolated_cells for nbr in _neighbors4(cell)):
                continue
            isolated_cells.append(cell)
            if len(isolated_cells) == isolated_count:
                break

        if len(isolated_cells) != isolated_count:
            continue

        all_cells = occupied | set(isolated_cells)
        if len(all_cells) != valid_count + isolated_count:
            continue
        return all_cells, cluster_sizes

    raise RuntimeError("failed to place a valid region layout")


def _sample_axis_spans() -> list[int]:
    spans = [random.choice(GEN_REGION_SPANS) for _ in range(3)]
    random.shuffle(spans)
    return spans


def _region_starts(spans: list[int]) -> list[int]:
    starts: list[int] = []
    cursor = 0
    for idx, span in enumerate(spans):
        starts.append(cursor)
        cursor += int(span)
        if idx < 2:
            cursor += 1
    return starts


def _region_stats(grid: np.ndarray) -> tuple[list[int], list[int], list[int]]:
    g = np.asarray(grid, dtype=int)
    _bg, _separator, marker_color, row_bands, col_bands = _find_partition_spec(g)

    raw_counts: list[int] = []
    valid_counts: list[int] = []
    isolated_counts: list[int] = []
    for r0, r1 in row_bands:
        for c0, c1 in col_bands:
            region = g[r0:r1, c0:c1]
            raw = int(np.sum(region == marker_color))
            valid, isolated = _count_region_markers(region, marker_color)
            raw_counts.append(raw)
            valid_counts.append(valid)
            isolated_counts.append(isolated)
    return raw_counts, valid_counts, isolated_counts


def _has_same_raw_but_different_valid(raw_counts: list[int], valid_counts: list[int]) -> bool:
    for i in range(len(raw_counts)):
        for j in range(i + 1, len(raw_counts)):
            if raw_counts[i] == raw_counts[j] and valid_counts[i] != valid_counts[j]:
                return True
    return False


def _has_same_valid_but_different_raw(raw_counts: list[int], valid_counts: list[int]) -> bool:
    for i in range(len(raw_counts)):
        for j in range(i + 1, len(raw_counts)):
            if valid_counts[i] == valid_counts[j] and raw_counts[i] != raw_counts[j]:
                return True
    return False


def _gen_attempt() -> np.ndarray | None:
    row_spans = _sample_axis_spans()
    col_spans = _sample_axis_spans()
    height = sum(row_spans) + 2
    width = sum(col_spans) + 2

    grid = np.full((height, width), BG_COLOR, dtype=int)
    row_starts = _region_starts(row_spans)
    col_starts = _region_starts(col_spans)
    sep_rows = [row_starts[1] - 1, row_starts[2] - 1]
    sep_cols = [col_starts[1] - 1, col_starts[2] - 1]
    grid[sep_rows, :] = SEPARATOR_COLOR
    grid[:, sep_cols] = SEPARATOR_COLOR

    valid_targets = list(GEN_VALID_COUNT_BAG)
    random.shuffle(valid_targets)
    iso_targets = [random.choice((1, 2)) for _ in range(9)]

    multi_remaining = GEN_FORCE_MULTI_COMPONENT
    cluster_count_record: list[int] = []
    idx = 0
    for rr in range(3):
        for cc in range(3):
            valid_target = valid_targets[idx]
            isolated_target = iso_targets[idx]
            force_multi = multi_remaining > 0 and valid_target >= 4
            try:
                cells, cluster_sizes = _place_region_markers(
                    row_spans[rr],
                    col_spans[cc],
                    valid_target,
                    isolated_target,
                    force_multi,
                )
            except RuntimeError:
                return None

            if len(cluster_sizes) > 1 and multi_remaining > 0:
                multi_remaining -= 1
            cluster_count_record.append(len(cluster_sizes))

            r0 = row_starts[rr]
            c0 = col_starts[cc]
            for dr, dc in cells:
                grid[r0 + dr, c0 + dc] = MARKER_COLOR
            idx += 1

    raw_counts, valid_counts, isolated_counts = _region_stats(grid)
    if valid_counts != valid_targets:
        return None
    if any(isolated < 1 for isolated in isolated_counts):
        return None
    if any(raw <= valid for raw, valid in zip(raw_counts, valid_counts)):
        return None
    if len(set(valid_counts)) < 5:
        return None
    if max(valid_counts) < 5 or min(valid_counts) > 2:
        return None
    if cluster_count_record.count(1) == len(cluster_count_record):
        return None
    if not _has_same_raw_but_different_valid(raw_counts, valid_counts):
        return None
    if not _has_same_valid_but_different_raw(raw_counts, valid_counts):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
