import itertools
import random
from collections import defaultdict, deque

import numpy as np


BLACK = 0
GREY = 5
SOURCE_COLORS = (1, 2, 3, 4, 7)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

GEN_ATTEMPTS = 400
LAYOUT_ATTEMPTS = 120
ROWS_RANGE = (3, 6)
COLS_RANGE = (3, 6)
CELL_SIZE_RANGE = (2, 4)
MAX_REGIONS = 8
MIN_REGIONS = 5


def _label_regions(mask: np.ndarray) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=int)
    regions: dict[int, np.ndarray] = {}
    next_id = 1

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != 0:
                continue
            queue = deque([(r, c)])
            labels[r, c] = next_id
            coords: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and labels[nr, nc] == 0:
                        labels[nr, nc] = next_id
                        queue.append((nr, nc))
            regions[next_id] = np.array(coords, dtype=int)
            next_id += 1

    return labels, regions


def _analyze_regions(grid: np.ndarray) -> tuple[np.ndarray, dict[int, dict[str, object]], dict[int, dict[int, int]]]:
    labels, region_coords = _label_regions(grid != GREY)
    h, w = grid.shape
    boundary_lengths: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))

    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) != GREY:
                continue

            if 0 < c < w - 1:
                left = int(labels[r, c - 1])
                right = int(labels[r, c + 1])
                if left and right and left != right:
                    boundary_lengths[left][right] += 1
                    boundary_lengths[right][left] += 1

            if 0 < r < h - 1:
                up = int(labels[r - 1, c])
                down = int(labels[r + 1, c])
                if up and down and up != down:
                    boundary_lengths[up][down] += 1
                    boundary_lengths[down][up] += 1

    info: dict[int, dict[str, object]] = {}
    for region_id, coords in region_coords.items():
        values = grid[coords[:, 0], coords[:, 1]]
        colored = values[(values != BLACK)]
        source_color = None
        if colored.size:
            uniques = np.unique(colored)
            source_color = int(uniques[0])
        info[region_id] = {
            "coords": coords,
            "area": int(coords.shape[0]),
            "source_color": source_color,
        }

    return labels, info, boundary_lengths


def _choose_source_pixel(grid: np.ndarray, coords: np.ndarray, rng: random.Random) -> tuple[int, int]:
    cells = [(int(r), int(c)) for r, c in coords]
    preferred = [
        (r, c)
        for r, c in cells
        if all(
            0 <= r + dr < grid.shape[0]
            and 0 <= c + dc < grid.shape[1]
            and int(grid[r + dr, c + dc]) != GREY
            for dr, dc in DIR4
        )
    ]
    pool = preferred if preferred else cells
    return rng.choice(pool)


def _grid_size(parts: list[int]) -> int:
    return sum(parts) + len(parts) + 1


def _sample_cell_sizes(count: int, rng: random.Random) -> list[int] | None:
    for _ in range(80):
        sizes = [rng.randint(*CELL_SIZE_RANGE) for _ in range(count)]
        if _grid_size(sizes) <= 30:
            return sizes
    return None


def _cell_neighbors(cell: tuple[int, int], n_rows: int, n_cols: int) -> list[tuple[int, int]]:
    r, c = cell
    out = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < n_rows:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < n_cols:
        out.append((r, c + 1))
    return out


def _partition_cells(n_rows: int, n_cols: int, rng: random.Random) -> dict[tuple[int, int], int] | None:
    cells = [(r, c) for r in range(n_rows) for c in range(n_cols)]
    total = len(cells)
    min_regions = min(MIN_REGIONS, total - 1)
    max_regions = min(MAX_REGIONS, total)
    if min_regions > max_regions:
        return None

    region_count = rng.randint(min_regions, max_regions)
    seeds = rng.sample(cells, region_count)
    assignment = {cell: region_id for region_id, cell in enumerate(seeds)}
    unassigned = set(cells) - set(seeds)

    while unassigned:
        options: list[tuple[int, tuple[int, int]]] = []
        for cell in unassigned:
            neighbor_regions = {
                assignment[neighbor]
                for neighbor in _cell_neighbors(cell, n_rows, n_cols)
                if neighbor in assignment
            }
            for region_id in neighbor_regions:
                options.append((region_id, cell))
        if not options:
            return None
        region_id, cell = rng.choice(options)
        assignment[cell] = region_id
        unassigned.remove(cell)

    return assignment


def _build_partition_grid(
    row_sizes: list[int],
    col_sizes: list[int],
    assignment: dict[tuple[int, int], int],
) -> np.ndarray:
    h = _grid_size(row_sizes)
    w = _grid_size(col_sizes)
    grid = np.full((h, w), BLACK, dtype=int)

    row_bounds = [0]
    for size in row_sizes:
        row_bounds.append(row_bounds[-1] + size + 1)
    col_bounds = [0]
    for size in col_sizes:
        col_bounds.append(col_bounds[-1] + size + 1)

    grid[0, :] = GREY
    grid[-1, :] = GREY
    grid[:, 0] = GREY
    grid[:, -1] = GREY

    n_rows = len(row_sizes)
    n_cols = len(col_sizes)

    for r in range(n_rows):
        for c in range(n_cols - 1):
            if assignment[(r, c)] == assignment[(r, c + 1)]:
                continue
            divider_c = col_bounds[c + 1]
            grid[row_bounds[r] : row_bounds[r + 1] + 1, divider_c] = GREY

    for r in range(n_rows - 1):
        for c in range(n_cols):
            if assignment[(r, c)] == assignment[(r + 1, c)]:
                continue
            divider_r = row_bounds[r + 1]
            grid[divider_r, col_bounds[c] : col_bounds[c + 1] + 1] = GREY

    return grid


def _evaluate_sources(
    regions: dict[int, dict[str, object]],
    boundary_lengths: dict[int, dict[int, int]],
    source_ids: tuple[int, ...],
) -> dict[str, int] | None:
    source_set = set(source_ids)
    tie_count = 0
    unique_count = 0
    competitive_unique = 0
    competitive_tie = 0
    adjacent_counts = {region_id: 0 for region_id in source_set}

    for region_id in regions:
        if region_id in source_set:
            continue
        scores = {
            source_id: int(boundary_lengths.get(region_id, {}).get(source_id, 0))
            for source_id in source_set
            if int(boundary_lengths.get(region_id, {}).get(source_id, 0)) > 0
        }
        if not scores:
            return None
        for source_id in scores:
            adjacent_counts[source_id] += 1

        max_len = max(scores.values())
        winners = [source_id for source_id, length in scores.items() if length == max_len]
        competitive = len(scores) >= 2
        if len(winners) == 1:
            unique_count += 1
            if competitive:
                competitive_unique += 1
        else:
            tie_count += 1
            if competitive:
                competitive_tie += 1

    if not all(count > 0 for count in adjacent_counts.values()):
        return None

    return {
        "tie_count": tie_count,
        "unique_count": unique_count,
        "competitive_unique": competitive_unique,
        "competitive_tie": competitive_tie,
    }


def _pick_source_regions(
    regions: dict[int, dict[str, object]],
    boundary_lengths: dict[int, dict[int, int]],
    require_tie: bool,
    rng: random.Random,
) -> tuple[int, ...] | None:
    candidates = [region_id for region_id, meta in regions.items() if int(meta["area"]) >= 2]
    if len(candidates) < 2:
        return None

    source_count_options = [count for count in (2, 3) if count < len(regions) and count <= len(candidates)]
    rng.shuffle(source_count_options)

    valid: list[tuple[int, ...]] = []
    for count in source_count_options:
        combos = list(itertools.combinations(candidates, count))
        rng.shuffle(combos)
        for source_ids in combos:
            stats = _evaluate_sources(regions, boundary_lengths, source_ids)
            if stats is None:
                continue
            if stats["competitive_unique"] < 1:
                continue
            if require_tie:
                if stats["competitive_tie"] < 1:
                    continue
            else:
                if stats["tie_count"] != 0:
                    continue
            valid.append(source_ids)
    if not valid:
        return None
    return rng.choice(valid)


def _gen_attempt(rng: random.Random, require_tie: bool) -> np.ndarray | None:
    for _ in range(LAYOUT_ATTEMPTS):
        n_rows = rng.randint(*ROWS_RANGE)
        n_cols = rng.randint(*COLS_RANGE)
        row_sizes = _sample_cell_sizes(n_rows, rng)
        col_sizes = _sample_cell_sizes(n_cols, rng)
        if row_sizes is None or col_sizes is None:
            continue
        if _grid_size(row_sizes) > 30 or _grid_size(col_sizes) > 30:
            continue

        assignment = _partition_cells(n_rows, n_cols, rng)
        if assignment is None:
            continue

        grid = _build_partition_grid(row_sizes, col_sizes, assignment)
        _labels, regions, boundary_lengths = _analyze_regions(grid)
        if len(regions) < 4:
            continue

        source_ids = _pick_source_regions(regions, boundary_lengths, require_tie, rng)
        if source_ids is None:
            continue

        colors = rng.sample(list(SOURCE_COLORS), len(source_ids))
        source_map = dict(zip(source_ids, colors))

        for region_id, color in source_map.items():
            sr, sc = _choose_source_pixel(grid, regions[region_id]["coords"], rng)
            grid[sr, sc] = color

        try:
            out = main(grid)
        except Exception:
            continue
        if np.array_equal(out, grid):
            continue
        return grid

    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each empty compartment with the color of the neighboring source region that shares the longest grey boundary.

    Input structure:
    - Black background cells are partitioned into closed 4-connected regions by one-pixel Grey divider lines.
    - Some regions contain exactly one colored source pixel chosen from a small visible palette; all other cells in those regions stay Black.
    - Other regions are empty and contain only Black cells.

    Rule:
    - Treat every non-Grey 4-connected component as one region.
    - A region containing a colored pixel is a source region and stays unchanged.
    - For each empty region, compare every adjacent source region by counting how many Grey pixels separate them across the shared boundary.
    - If one adjacent source has the unique longest shared boundary, recolor every Black cell in the empty region to that source color; if the longest boundary is tied, leave the region Black.
    - Keep all Grey divider pixels unchanged.

    Key insight:
    - The task is decided at the region level, not by nearest pixels: the winning color comes from the neighboring source region with the longest visibly shared separator segment.
    """
    g = np.asarray(grid, dtype=int)
    _labels, regions, boundary_lengths = _analyze_regions(g)
    out = g.copy()

    source_ids = {
        region_id
        for region_id, meta in regions.items()
        if meta["source_color"] is not None
    }

    for region_id, meta in regions.items():
        if region_id in source_ids:
            continue
        scores = {
            source_id: int(boundary_lengths.get(region_id, {}).get(source_id, 0))
            for source_id in source_ids
            if int(boundary_lengths.get(region_id, {}).get(source_id, 0)) > 0
        }
        if not scores:
            continue

        max_len = max(scores.values())
        winners = [source_id for source_id, length in scores.items() if length == max_len]
        if len(winners) != 1:
            continue

        fill_color = int(regions[winners[0]]["source_color"])
        coords = meta["coords"]
        mask = out[coords[:, 0], coords[:, 1]] == BLACK
        fill_coords = coords[mask]
        out[fill_coords[:, 0], fill_coords[:, 1]] = fill_color

    return out


def generate_input() -> np.ndarray:
    rng = random.Random()
    require_tie = bool(rng.randrange(2))
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng, require_tie)
        if grid is not None:
            return grid
        require_tie = not require_tie
    raise RuntimeError("failed to generate a valid puzzle input")
