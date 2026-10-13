from __future__ import annotations

import numpy as np


WALL_COLOR = 5
REGION_COLORS = (1, 2, 3, 4, 7)
GEN_REGION_COUNT_CHOICES = (4, 5)
GEN_COARSE_ROW_CHOICES = (3, 4)
GEN_COARSE_COL_CHOICES = (4, 5)
GEN_BAND_SIZE_CHOICES = (2, 3, 4)
GEN_MAX_ATTEMPTS = 400


def _neighbors4(r: int, c: int, height: int, width: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < height:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < width:
        yield r, c + 1


def _find_regions(grid: np.ndarray, wall_color: int = WALL_COLOR) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    height, width = map(int, g.shape)
    seen = np.zeros((height, width), dtype=bool)
    regions: list[dict] = []

    for r0 in range(height):
        for c0 in range(width):
            if seen[r0, c0] or int(g[r0, c0]) == int(wall_color):
                continue

            stack = [(r0, c0)]
            seen[r0, c0] = True
            coords: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                coords.append((r, c))
                for nr, nc in _neighbors4(r, c, height, width):
                    if seen[nr, nc] or int(g[nr, nc]) == int(wall_color):
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))

            arr = np.asarray(coords, dtype=int)
            rows = arr[:, 0]
            cols = arr[:, 1]
            values = np.unique(g[rows, cols])
            assert values.size == 1, "each wall-separated region must be monochromatic"
            regions.append(
                {
                    "coords": arr,
                    "area": int(arr.shape[0]),
                    "color": int(values[0]),
                    "bbox": (
                        int(rows.min()),
                        int(cols.min()),
                        int(rows.max()),
                        int(cols.max()),
                    ),
                    "centroid": (float(rows.mean()), float(cols.mean())),
                }
            )

    return regions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate region colors forward along the small-to-large area ranking defined by grey wall partitions.

    Input structure:
    - The grid contains 1-pixel grey walls and several 4-connected non-grey regions.
    - Each wall-separated region is monochromatic, while walls stay grey.
    - The deciding property is each region's pixel area, not its position, adjacency, or color id.

    Rule:
    - Find every 4-connected non-grey region separated by the grey walls.
    - Sort the regions by area from smallest to largest and keep that full ordering.
    - Recolor each region with the next larger region's original color, wrapping the largest region to the smallest region's original color.
    - Leave all grey wall pixels unchanged.

    Key insight:
    - The recoloring is driven by a global area ranking across all wall-separated regions, so far-apart regions can exchange colors if their areas are consecutive in that ranking.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    regions = _find_regions(g, wall_color=WALL_COLOR)
    assert regions, "expected at least one non-wall region"

    areas = [region["area"] for region in regions]
    assert len(set(areas)) == len(areas), "region areas must be unique"

    ordered = sorted(regions, key=lambda region: region["area"])
    ordered_colors = [region["color"] for region in ordered]

    out = g.copy()
    for idx, region in enumerate(ordered):
        next_color = int(ordered_colors[(idx + 1) % len(ordered_colors)])
        rr = region["coords"][:, 0]
        cc = region["coords"][:, 1]
        out[rr, cc] = next_color

    return out


def _coarse_shape_options(k: int) -> list[tuple[int, int]]:
    return [
        (rows, cols)
        for rows in GEN_COARSE_ROW_CHOICES
        for cols in GEN_COARSE_COL_CHOICES
        if rows * cols >= k + 4
    ]


def _sample_connected_partition(rows: int, cols: int, k: int, rng: np.random.Generator) -> np.ndarray:
    all_cells = [(r, c) for r in range(rows) for c in range(cols)]
    seed_indices = rng.choice(len(all_cells), size=k, replace=False)
    owners = np.full((rows, cols), -1, dtype=int)
    frontiers = [set() for _ in range(k)]

    for region_id, flat_idx in enumerate(seed_indices):
        sr, sc = all_cells[int(flat_idx)]
        owners[sr, sc] = region_id

    for region_id, flat_idx in enumerate(seed_indices):
        sr, sc = all_cells[int(flat_idx)]
        for nr, nc in _neighbors4(sr, sc, rows, cols):
            if owners[nr, nc] == -1:
                frontiers[region_id].add((nr, nc))

    unassigned = rows * cols - k
    while unassigned:
        growable = [idx for idx, frontier in enumerate(frontiers) if frontier]
        assert growable, "connected growth should keep at least one frontier"
        region_id = int(growable[int(rng.integers(len(growable)))])
        frontier = list(frontiers[region_id])
        cell = frontier[int(rng.integers(len(frontier)))]
        if owners[cell] != -1:
            frontiers[region_id].discard(cell)
            continue

        owners[cell] = region_id
        unassigned -= 1
        for other in frontiers:
            other.discard(cell)
        for nr, nc in _neighbors4(cell[0], cell[1], rows, cols):
            if owners[nr, nc] == -1:
                frontiers[region_id].add((nr, nc))

    return owners


def _band_starts(sizes: list[int]) -> tuple[list[int], int]:
    starts: list[int] = []
    cursor = 0
    for idx, size in enumerate(sizes):
        starts.append(cursor)
        cursor += int(size)
        if idx + 1 < len(sizes):
            cursor += 1
    return starts, cursor


def _render_partition(
    owners: np.ndarray,
    row_heights: list[int],
    col_widths: list[int],
    region_colors: list[int],
) -> np.ndarray:
    coarse_rows, coarse_cols = map(int, owners.shape)
    row_starts, height = _band_starts(row_heights)
    col_starts, width = _band_starts(col_widths)
    grid = np.full((height, width), WALL_COLOR, dtype=int)

    for r in range(coarse_rows):
        for c in range(coarse_cols):
            color = int(region_colors[int(owners[r, c])])
            r0 = row_starts[r]
            c0 = col_starts[c]
            grid[r0 : r0 + row_heights[r], c0 : c0 + col_widths[c]] = color

    for r in range(coarse_rows):
        for c in range(coarse_cols - 1):
            if int(owners[r, c]) != int(owners[r, c + 1]):
                continue
            color = int(region_colors[int(owners[r, c])])
            wall_c = col_starts[c] + col_widths[c]
            r0 = row_starts[r]
            grid[r0 : r0 + row_heights[r], wall_c] = color

    for r in range(coarse_rows - 1):
        for c in range(coarse_cols):
            if int(owners[r, c]) != int(owners[r + 1, c]):
                continue
            color = int(region_colors[int(owners[r, c])])
            wall_r = row_starts[r] + row_heights[r]
            c0 = col_starts[c]
            grid[wall_r, c0 : c0 + col_widths[c]] = color

    return grid


def _expected_output_from_colors(grid: np.ndarray, region_colors: list[int]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    areas = {int(color): int(np.count_nonzero(grid == int(color))) for color in region_colors}
    assert len(set(areas.values())) == len(areas), "region areas must be unique"
    ordered_colors = sorted(region_colors, key=lambda color: areas[int(color)])

    for idx, color in enumerate(ordered_colors):
        out[grid == int(color)] = int(ordered_colors[(idx + 1) % len(ordered_colors)])
    return out


def _spatial_order(regions: list[dict], key_fn) -> tuple[int, ...]:
    ordered = sorted(
        regions,
        key=lambda region: (
            key_fn(region),
            region["bbox"][0],
            region["bbox"][1],
            region["color"],
        ),
    )
    return tuple(int(region["color"]) for region in ordered)


def _area_order(regions: list[dict]) -> tuple[int, ...]:
    ordered = sorted(regions, key=lambda region: (region["area"], region["color"]))
    return tuple(int(region["color"]) for region in ordered)


def _candidate_is_semantic(
    grid: np.ndarray,
    owners: np.ndarray,
    region_colors: list[int],
) -> bool:
    regions = _find_regions(grid, wall_color=WALL_COLOR)
    if len(regions) != len(region_colors):
        return False

    actual_colors = {int(region["color"]) for region in regions}
    if actual_colors != set(map(int, region_colors)):
        return False

    actual_areas = [region["area"] for region in regions]
    if len(set(actual_areas)) != len(actual_areas):
        return False

    coarse_cell_counts = [int(np.count_nonzero(owners == region_id)) for region_id in range(len(region_colors))]
    if max(coarse_cell_counts) < 3:
        return False
    if sum(count > 1 for count in coarse_cell_counts) < 2:
        return False

    non_rect_count = 0
    for region in regions:
        r0, c0, r1, c1 = region["bbox"]
        if region["area"] < (r1 - r0 + 1) * (c1 - c0 + 1):
            non_rect_count += 1
    if non_rect_count < 1:
        return False

    area_order = _area_order(regions)
    spatial_orders = {
        _spatial_order(regions, key_fn=lambda region: region["centroid"][1]),
        _spatial_order(regions, key_fn=lambda region: region["centroid"][0]),
        _spatial_order(regions, key_fn=lambda region: (region["centroid"][0], region["centroid"][1])),
        _spatial_order(regions, key_fn=lambda region: (region["centroid"][1], region["centroid"][0])),
    }
    if area_order in spatial_orders:
        return False

    numeric_asc = tuple(sorted(int(color) for color in region_colors))
    numeric_desc = tuple(sorted((int(color) for color in region_colors), reverse=True))
    if area_order in {numeric_asc, numeric_desc}:
        return False

    if len(set(coarse_cell_counts)) == len(coarse_cell_counts):
        coarse_order = tuple(
            int(region_colors[region_id])
            for region_id in np.argsort(np.asarray(coarse_cell_counts, dtype=int))
        )
        if coarse_order == area_order:
            return False

    wall_pixels = int(np.count_nonzero(grid == WALL_COLOR))
    if wall_pixels < grid.shape[0] + grid.shape[1]:
        return False

    expected = _expected_output_from_colors(grid, region_colors)
    try:
        actual = main(grid)
    except Exception:
        return False
    return np.array_equal(actual, expected)


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    k = int(GEN_REGION_COUNT_CHOICES[int(rng.integers(len(GEN_REGION_COUNT_CHOICES)))])
    coarse_rows, coarse_cols = _coarse_shape_options(k)[int(rng.integers(len(_coarse_shape_options(k))))]

    row_heights = [int(GEN_BAND_SIZE_CHOICES[int(rng.integers(len(GEN_BAND_SIZE_CHOICES)))]) for _ in range(coarse_rows)]
    col_widths = [int(GEN_BAND_SIZE_CHOICES[int(rng.integers(len(GEN_BAND_SIZE_CHOICES)))]) for _ in range(coarse_cols)]
    if len(set(row_heights + col_widths)) < 2:
        return None

    owners = _sample_connected_partition(coarse_rows, coarse_cols, k, rng)
    region_colors = [int(color) for color in rng.choice(np.asarray(REGION_COLORS, dtype=int), size=k, replace=False)]
    grid = _render_partition(owners, row_heights, col_widths, region_colors)

    if max(grid.shape) > 30:
        return None
    if not _candidate_is_semantic(grid, owners, region_colors):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_MAX_ATTEMPTS} attempts")
