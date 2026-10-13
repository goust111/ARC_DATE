import random
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np


BG = 0
COLOR_POOL = tuple(range(1, 10))
GRID_MIN = 14
GRID_MAX = 28
COLOR_COUNT_RANGE = (2, 4)
SEEDS_PER_COLOR_RANGE = (1, 4)
GENERATION_RETRIES = 500


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _legend_info(grid: np.ndarray, bg: int) -> Tuple[List[int], Set[Tuple[int, int]]]:
    height, width = grid.shape
    legend_rows = np.flatnonzero(grid[:, width - 1] != bg)
    assert 2 <= len(legend_rows) <= 4, "legend must contain 2 to 4 pixels"
    assert int(legend_rows[-1]) == height - 1, "legend must touch the bottom edge"
    expected_rows = np.arange(int(legend_rows[0]), height)
    assert np.array_equal(legend_rows, expected_rows), "legend must be a contiguous vertical strip"

    priority = [int(grid[r, width - 1]) for r in legend_rows]
    assert len(priority) == len(set(priority)), "legend colors must be unique"
    cells = {(int(r), width - 1) for r in legend_rows}
    return priority, cells


def _cross_cells(row: int, col: int, radius: int, shape: Tuple[int, int]) -> Set[Tuple[int, int]]:
    height, width = shape
    cells: Set[Tuple[int, int]] = {(row, col)}
    for step in range(1, radius + 1):
        if row - step >= 0:
            cells.add((row - step, col))
        if row + step < height:
            cells.add((row + step, col))
        if col - step >= 0:
            cells.add((row, col - step))
        if col + step < width:
            cells.add((row, col + step))
    return cells


def _parse_seed_groups(
    grid: np.ndarray, priority: Sequence[int], legend_cells: Set[Tuple[int, int]], bg: int
) -> Dict[int, List[Tuple[int, int]]]:
    seeds_by_color: Dict[int, List[Tuple[int, int]]] = {int(color): [] for color in priority}
    for row, col in np.argwhere(grid != bg):
        pos = (int(row), int(col))
        if pos in legend_cells:
            continue
        color = int(grid[row, col])
        assert color in seeds_by_color, "every seed color must appear in the legend"
        seeds_by_color[color].append(pos)

    assert all(seeds_by_color[color] for color in seeds_by_color), "every legend color needs at least one seed"
    return seeds_by_color


def _transform(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    priority, legend_cells = _legend_info(g, bg)
    seeds_by_color = _parse_seed_groups(g, priority, legend_cells, bg)

    out = np.full(g.shape, bg, dtype=int)
    for color in reversed(priority):
        radius = len(seeds_by_color[color])
        for row, col in seeds_by_color[color]:
            for rr, cc in _cross_cells(row, col, radius, g.shape):
                out[rr, cc] = color

    for row, col in legend_cells:
        out[row, col] = bg
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each color's seed pixels into priority-resolved crosses and erase the corner legend.

    Input structure:
    - The background is black.
    - Several isolated non-background seed pixels are scattered across the grid in 2 to 4 colors.
    - The lower-right corner contains a one-column vertical legend made of the same colors, ordered from top to bottom by priority.

    Rule:
    - Ignore the legend pixels when counting seeds.
    - For each color, let R be the number of seeds of that color, then replace every seed by a cross centered on that seed and extending R cells up, down, left, and right, clipped by the grid boundary.
    - If crosses of different colors overlap, the color appearing higher in the legend wins that pixel.
    - After drawing all crosses, clear the legend column back to background.

    Key insight:
    - The legend is not decorative: it both identifies the participating colors and supplies the overlap priority, while each color's seed count determines its cross radius.
    """
    return _transform(grid)


def _fits_without_clipping(row: int, col: int, radius: int, shape: Tuple[int, int]) -> bool:
    height, width = shape
    return row - radius >= 0 and row + radius < height and col - radius >= 0 and col + radius < width


def _reserved_for_legend(row: int, col: int, height: int, width: int, legend_size: int) -> bool:
    legend_top = height - legend_size
    if col == width - 1:
        return True
    return col >= width - 2 and row >= legend_top - 1


def _touches_seed(occupied: Set[Tuple[int, int]], row: int, col: int) -> bool:
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        if (row + dr, col + dc) in occupied:
            return True
    return False


def _pick_seed_position(
    color: int,
    counts: Dict[int, int],
    shape: Tuple[int, int],
    legend_size: int,
    occupied: Set[Tuple[int, int]],
    coverage: Dict[int, Set[Tuple[int, int]]],
    require_clipping: bool,
    forbid_clipping: bool,
    avoid_overlap: bool,
) -> Optional[Tuple[int, int, Set[Tuple[int, int]]]]:
    height, width = shape
    radius = counts[color]
    candidates: List[Tuple[int, int, Set[Tuple[int, int]]]] = []

    for row in range(height):
        for col in range(width):
            if _reserved_for_legend(row, col, height, width, legend_size):
                continue
            if (row, col) in occupied or _touches_seed(occupied, row, col):
                continue

            clipped = not _fits_without_clipping(row, col, radius, shape)
            if require_clipping and not clipped:
                continue
            if forbid_clipping and clipped:
                continue

            cross = _cross_cells(row, col, radius, shape)
            if avoid_overlap and any(cross & coverage[other] for other in coverage if other != color):
                continue
            candidates.append((row, col, cross))

    if not candidates:
        return None
    return random.choice(candidates)


def _pick_overlap_pair(
    colors: Sequence[int],
    counts: Dict[int, int],
    shape: Tuple[int, int],
    legend_size: int,
    force_clipping: bool,
    forbid_clipping: bool,
) -> Optional[Tuple[int, Tuple[int, int], Set[Tuple[int, int]], int, Tuple[int, int], Set[Tuple[int, int]]]]:
    height, width = shape
    legend_cells = {(height - legend_size + offset, width - 1) for offset in range(legend_size)}
    color_pairs = [(colors[i], colors[j]) for i in range(len(colors)) for j in range(i + 1, len(colors))]
    random.shuffle(color_pairs)

    for color_a, color_b in color_pairs:
        radius_a = counts[color_a]
        radius_b = counts[color_b]
        candidates = []

        for gap_left in range(1, radius_a + 1):
            for gap_right in range(1, radius_b + 1):
                separation = gap_left + gap_right
                for row in range(height):
                    for col_a in range(width - separation):
                        col_b = col_a + separation
                        if _reserved_for_legend(row, col_a, height, width, legend_size):
                            continue
                        if _reserved_for_legend(row, col_b, height, width, legend_size):
                            continue

                        clipped_a = not _fits_without_clipping(row, col_a, radius_a, shape)
                        clipped_b = not _fits_without_clipping(row, col_b, radius_b, shape)
                        if forbid_clipping and (clipped_a or clipped_b):
                            continue
                        if force_clipping and not (clipped_a or clipped_b):
                            continue

                        cross_a = _cross_cells(row, col_a, radius_a, shape)
                        cross_b = _cross_cells(row, col_b, radius_b, shape)
                        if not ((cross_a & cross_b) - legend_cells):
                            continue
                        candidates.append((color_a, (row, col_a), cross_a, color_b, (row, col_b), cross_b))

        if candidates:
            return random.choice(candidates)
    return None


def _has_inter_color_overlap(
    coverage: Dict[int, Set[Tuple[int, int]]], legend_cells: Set[Tuple[int, int]]
) -> bool:
    colors = list(coverage)
    for i, color_a in enumerate(colors):
        for color_b in colors[i + 1 :]:
            if (coverage[color_a] & coverage[color_b]) - legend_cells:
                return True
    return False


def _has_boundary_clip(seeds_by_color: Dict[int, List[Tuple[int, int]]], shape: Tuple[int, int]) -> bool:
    for color, seeds in seeds_by_color.items():
        radius = len(seeds)
        for row, col in seeds:
            if not _fits_without_clipping(row, col, radius, shape):
                return True
    return False


def _build_grid(
    shape: Tuple[int, int], priority: Sequence[int], seeds_by_color: Dict[int, List[Tuple[int, int]]]
) -> np.ndarray:
    height, width = shape
    grid = np.full(shape, BG, dtype=int)
    legend_top = height - len(priority)
    for offset, color in enumerate(priority):
        grid[legend_top + offset, width - 1] = int(color)
    for color, seeds in seeds_by_color.items():
        for row, col in seeds:
            grid[row, col] = int(color)
    return grid


def _gen_attempt() -> Optional[np.ndarray]:
    color_count = random.randint(*COLOR_COUNT_RANGE)
    colors = random.sample(COLOR_POOL, color_count)
    priority = colors[:]
    random.shuffle(priority)

    counts = {color: random.randint(*SEEDS_PER_COLOR_RANGE) for color in colors}
    max_radius = max(counts.values())

    min_side = max(GRID_MIN, 2 * max_radius + 6)
    if random.choice((True, False)):
        min_side = max(min_side, 18)
    height = random.randint(min_side, GRID_MAX)
    width = random.randint(min_side, GRID_MAX)

    want_overlap = random.choice((True, False))
    want_clipping = random.choice((True, False))
    legend_size = len(priority)
    shape = (height, width)

    seeds_by_color: Dict[int, List[Tuple[int, int]]] = {color: [] for color in colors}
    coverage: Dict[int, Set[Tuple[int, int]]] = {color: set() for color in colors}
    occupied: Set[Tuple[int, int]] = set()

    total_seeds = sum(counts.values())
    clipping_satisfied = False

    if want_overlap:
        force_clipping = want_clipping and total_seeds == 2
        pair = _pick_overlap_pair(
            colors=colors,
            counts=counts,
            shape=shape,
            legend_size=legend_size,
            force_clipping=force_clipping,
            forbid_clipping=not want_clipping,
        )
        if pair is None:
            return None

        color_a, pos_a, cross_a, color_b, pos_b, cross_b = pair
        seeds_by_color[color_a].append(pos_a)
        seeds_by_color[color_b].append(pos_b)
        coverage[color_a] |= cross_a
        coverage[color_b] |= cross_b
        occupied.add(pos_a)
        occupied.add(pos_b)
        clipping_satisfied = force_clipping

    remaining_slots: List[int] = []
    for color in colors:
        remaining_slots.extend([color] * (counts[color] - len(seeds_by_color[color])))

    if want_clipping and not clipping_satisfied:
        eligible = [color for color in colors if counts[color] > len(seeds_by_color[color])]
        if not eligible:
            return None
        clip_color = random.choice(eligible)
        picked = _pick_seed_position(
            color=clip_color,
            counts=counts,
            shape=shape,
            legend_size=legend_size,
            occupied=occupied,
            coverage=coverage,
            require_clipping=True,
            forbid_clipping=False,
            avoid_overlap=not want_overlap,
        )
        if picked is None:
            return None
        row, col, cross = picked
        seeds_by_color[clip_color].append((row, col))
        coverage[clip_color] |= cross
        occupied.add((row, col))
        clipping_satisfied = True
        remaining_slots.remove(clip_color)

    random.shuffle(remaining_slots)
    for color in remaining_slots:
        picked = _pick_seed_position(
            color=color,
            counts=counts,
            shape=shape,
            legend_size=legend_size,
            occupied=occupied,
            coverage=coverage,
            require_clipping=False,
            forbid_clipping=not want_clipping,
            avoid_overlap=not want_overlap,
        )
        if picked is None:
            return None
        row, col, cross = picked
        seeds_by_color[color].append((row, col))
        coverage[color] |= cross
        occupied.add((row, col))

    legend_top = height - legend_size
    legend_cells = {(legend_top + offset, width - 1) for offset in range(legend_size)}
    has_overlap = _has_inter_color_overlap(coverage, legend_cells)
    has_clip = _has_boundary_clip(seeds_by_color, shape)

    if has_overlap != want_overlap:
        return None
    if has_clip != want_clipping:
        return None
    if any(len(seeds_by_color[color]) != counts[color] for color in colors):
        return None

    grid = _build_grid(shape, priority, seeds_by_color)
    transformed = _transform(grid)
    if transformed.shape != grid.shape:
        return None
    if np.any(grid[:, width - 1][:legend_top] != BG):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
