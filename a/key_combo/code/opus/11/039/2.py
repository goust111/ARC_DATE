from __future__ import annotations

from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
YELLOW = 4

DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIRS8 = tuple(
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
    if (dr, dc) != (0, 0)
)

GEN_ATTEMPTS = 400
MARKER_ATTEMPTS = 120


def _trace_ray(shape: tuple[int, int], start_col: int) -> tuple[list[tuple[int, int]], int]:
    """Trace the upward bouncing diagonal ray from the bottom row."""
    height, width = map(int, shape)
    row = height - 1
    col = int(start_col)
    dc = 1
    path: list[tuple[int, int]] = []
    bounces = 0

    while True:
        path.append((row, col))
        next_row = row - 1
        if next_row < 0:
            break
        next_col = col + dc
        if next_col < 0 or next_col >= width:
            dc = -dc
            bounces += 1
            next_col = col + dc
        row, col = next_row, next_col

    return path, bounces


def _paint_path(shape: tuple[int, int], start_col: int) -> tuple[np.ndarray, list[tuple[int, int]], int]:
    mask = np.zeros(shape, dtype=bool)
    path, bounces = _trace_ray(shape, start_col)
    for row, col in path:
        mask[row, col] = True
    return mask, path, bounces


def _black_regions(grid: np.ndarray) -> list[dict[str, object]]:
    """Return 4-connected black regions sorted by (leftmost col, then topmost row)."""
    height, width = map(int, grid.shape)
    seen = np.zeros((height, width), dtype=bool)
    regions: list[dict[str, object]] = []

    for row in range(height):
        for col in range(width):
            if grid[row, col] != BLACK or seen[row, col]:
                continue
            queue: deque[tuple[int, int]] = deque([(row, col)])
            seen[row, col] = True
            coords: list[tuple[int, int]] = []
            min_row = row
            min_col = col
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                min_row = min(min_row, rr)
                min_col = min(min_col, cc)
                for dr, dc in DIRS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < height and 0 <= nc < width and grid[nr, nc] == BLACK and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            regions.append({"coords": coords, "sort_key": (min_col, min_row)})

    regions.sort(key=lambda region: region["sort_key"])
    return regions


def _analysis_from_start(shape: tuple[int, int], start_col: int) -> dict[str, object]:
    path_mask, path, bounces = _paint_path(shape, start_col)
    path_grid = np.zeros(shape, dtype=int)
    path_grid[path_mask] = BLUE
    regions = _black_regions(path_grid)
    return {
        "path_mask": path_mask,
        "path": path,
        "bounces": bounces,
        "regions": regions,
        "region_sizes": [len(region["coords"]) for region in regions],
    }


def _solve(grid: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)

    blue_cells = np.argwhere(g == BLUE)
    assert blue_cells.shape[0] == 1, "expected exactly one blue start pixel"
    start_row, start_col = map(int, blue_cells[0])
    assert start_row == height - 1, "the blue start must be on the bottom row"
    assert 0 < start_col < width - 1, "the blue start cannot be in a corner"

    k = int(np.count_nonzero(g == RED))
    assert k > 0, "expected at least one red marker"

    analysis = _analysis_from_start((height, width), start_col)
    out = g.copy()
    out[analysis["path_mask"]] = BLUE
    regions = _black_regions(out)

    for idx, region in enumerate(regions):
        if idx % k == 0:
            for row, col in region["coords"]:
                out[row, col] = YELLOW

    return out, analysis | {"k": k, "start_col": start_col, "regions": regions, "region_sizes": [len(region["coords"]) for region in regions]}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Partition the grid with a bouncing blue ray, then fill every K-th region where K is the red-marker count.

    Input structure:
    - A black grid contains exactly one blue start pixel on the bottom row and K isolated red marker pixels.
    - The blue pixel is the origin of an upward-right diagonal ray that reflects on the left and right borders.
    - Red markers never lie on the ray path; they are separate from the blue path and from the black regions.

    Rule:
    - Extend the blue ray one row upward per step, reversing only its horizontal direction when the next column would leave the grid; stop when the next row would leave the top.
    - Treat the resulting blue path as a barrier and find the remaining 4-connected black regions.
    - Sort those regions by each region's leftmost column, breaking ties by topmost row, and number them 0, 1, 2, ...
    - Count the red markers to get K, then flood-fill regions 0, K, 2K, ... with yellow while leaving blue, red, and all other cells unchanged.

    Key insight:
    - The blue ray determines the region topology, while the red markers contribute only their count, which becomes the periodic fill step.
    """
    return _solve(grid)[0]


def _selected_region_indices(region_count: int, k: int) -> list[int]:
    return list(range(0, int(region_count), int(k)))


def _valid_start_columns(shape: tuple[int, int], k: int) -> list[tuple[int, dict[str, object]]]:
    height, width = map(int, shape)
    valid: list[tuple[int, dict[str, object]]] = []
    for start_col in range(1, width - 1):
        analysis = _analysis_from_start((height, width), start_col)
        region_count = len(analysis["regions"])
        if analysis["bounces"] < 2 or region_count <= k:
            continue
        chosen = _selected_region_indices(region_count, k)
        if min(analysis["region_sizes"][idx] for idx in chosen) < 3:
            continue
        if sum(size >= 3 for size in analysis["region_sizes"]) < 3:
            continue
        valid.append((start_col, analysis))
    return valid


def _non_adjacent(cell: tuple[int, int], chosen: list[tuple[int, int]]) -> bool:
    row, col = cell
    return all(max(abs(row - rr), abs(col - cc)) > 1 for rr, cc in chosen)


def _sample_markers(
    rng: np.random.Generator,
    analysis: dict[str, object],
    k: int,
) -> list[tuple[int, int]] | None:
    regions = analysis["regions"]
    assert regions
    region_indices = np.arange(len(regions))

    for _ in range(MARKER_ATTEMPTS):
        chosen: list[tuple[int, int]] = []
        chosen_regions = []
        distinct_region_mode = bool(rng.integers(0, 2)) and len(regions) >= k
        if distinct_region_mode:
            chosen_regions = list(map(int, rng.choice(region_indices, size=k, replace=False)))
        else:
            chosen_regions = [int(rng.choice(region_indices)) for _ in range(k)]

        ok = True
        for region_idx in chosen_regions:
            cells = list(regions[region_idx]["coords"])
            order = rng.permutation(len(cells))
            picked = None
            for pos in order:
                candidate = cells[int(pos)]
                if _non_adjacent(candidate, chosen):
                    picked = candidate
                    break
            if picked is None:
                ok = False
                break
            chosen.append(picked)

        if ok and len(chosen) == k:
            return chosen
    return None


def _build_input(shape: tuple[int, int], start_col: int, markers: list[tuple[int, int]]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    grid[shape[0] - 1, start_col] = BLUE
    for row, col in markers:
        grid[row, col] = RED
    return grid


def _alternate_k_grid(
    rng: np.random.Generator,
    grid: np.ndarray,
    analysis: dict[str, object],
    markers: list[tuple[int, int]],
) -> np.ndarray | None:
    k = len(markers)
    if k == 2:
        pool = [cell for region in analysis["regions"] for cell in region["coords"]]
        order = rng.permutation(len(pool))
        extra = None
        for pos in order:
            candidate = pool[int(pos)]
            if candidate not in markers and _non_adjacent(candidate, markers):
                extra = candidate
                break
        if extra is None:
            return None
        alt = grid.copy()
        alt[extra] = RED
        return alt

    if k == 3:
        alt = grid.copy()
        alt[markers[0]] = BLACK
        return alt

    return None


def _validate_generated(grid: np.ndarray, expected_k: int) -> None:
    out, meta = _solve(grid)
    height, width = map(int, grid.shape)

    assert 0 < meta["start_col"] < width - 1
    assert meta["bounces"] >= 2
    assert meta["k"] == expected_k
    assert meta["k"] in (2, 3)
    assert len(meta["regions"]) > meta["k"]
    assert np.count_nonzero(grid == BLUE) == 1
    assert np.count_nonzero(grid == RED) == expected_k
    assert np.count_nonzero(out == YELLOW) > 0
    assert np.count_nonzero(out == BLACK) > 0

    path_set = set(meta["path"])
    red_cells = [tuple(map(int, rc)) for rc in np.argwhere(grid == RED)]
    for row, col in red_cells:
        assert (row, col) not in path_set
        for dr, dc in DIRS8:
            nr = row + dr
            nc = col + dc
            if 0 <= nr < height and 0 <= nc < width and grid[nr, nc] == RED:
                assert (nr, nc) == (row, col)

    regions_no_markers = _analysis_from_start((height, width), meta["start_col"])["regions"]
    assert len(regions_no_markers) == len(meta["regions"])
    assert [region["sort_key"] for region in regions_no_markers] == [region["sort_key"] for region in meta["regions"]]

    selected = set(_selected_region_indices(len(meta["regions"]), expected_k))
    for idx, region in enumerate(meta["regions"]):
        coords = region["coords"]
        filled_black = sum(out[row, col] == YELLOW for row, col in coords)
        if idx in selected:
            assert filled_black == len(coords), "selected black cells must all become yellow"
        else:
            assert filled_black == 0, "unselected black regions must stay black"

    assert min(meta["region_sizes"][idx] for idx in selected) >= 3

    candidate_alts = []
    for alt_start, _analysis in _valid_start_columns((height, width), expected_k):
        if alt_start != meta["start_col"]:
            alt = grid.copy()
            alt[height - 1, meta["start_col"]] = BLACK
            alt[height - 1, alt_start] = BLUE
            candidate_alts.append(alt)
    assert candidate_alts, "need another valid start column so path dependence is testable"
    assert any(not np.array_equal(main(alt), out) for alt in candidate_alts), "path must affect the output"


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    if rng.random() < 0.5:
        height = int(rng.integers(13, 19))
        width = int(rng.integers(8, 14))
    else:
        height = int(rng.integers(14, 21))
        width = int(rng.integers(5, 10))

    k = 2 if rng.random() < 0.5 else 3
    valid_starts = _valid_start_columns((height, width), k)
    if len(valid_starts) < 2:
        return None

    start_col, analysis = valid_starts[int(rng.integers(0, len(valid_starts)))]
    markers = _sample_markers(rng, analysis, k)
    if markers is None:
        return None

    grid = _build_input((height, width), start_col, markers)
    path_painted = grid.copy()
    path_painted[analysis["path_mask"]] = BLUE
    regions_with_markers = _black_regions(path_painted)
    if len(regions_with_markers) != len(analysis["regions"]):
        return None
    if [region["sort_key"] for region in regions_with_markers] != [region["sort_key"] for region in analysis["regions"]]:
        return None

    alt_k_grid = _alternate_k_grid(rng, grid, analysis, markers)
    if alt_k_grid is None:
        return None
    if np.array_equal(main(alt_k_grid), main(grid)):
        return None

    try:
        _validate_generated(grid, k)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid bouncing-ray periodic-fill input")
