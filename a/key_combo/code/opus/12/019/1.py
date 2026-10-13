from __future__ import annotations

from collections import Counter
import random

import numpy as np


BACKGROUND = 0
ALL_NON_BG_COLORS = tuple(range(1, 10))
GENERATION_RETRIES = 500


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _unique_least_frequent_non_bg_color(grid: np.ndarray) -> int:
    bg = _background_color(grid)
    values, counts = np.unique(grid, return_counts=True)
    pairs = [(int(color), int(count)) for color, count in zip(values, counts) if int(color) != bg]
    assert pairs, "expected at least one foreground color"

    min_count = min(count for _color, count in pairs)
    candidates = [color for color, count in pairs if count == min_count]
    assert len(candidates) == 1, "expected a unique least-frequent non-background marker color"
    return int(candidates[0])


def _marker_owner_color(grid: np.ndarray, marker_color: int, r: int, c: int) -> int:
    adjacent = []
    for nr, nc in _neighbors4(r, c):
        assert _in_bounds(grid.shape, nr, nc), "marker must stay away from the grid border"
        value = int(grid[nr, nc])
        assert value != marker_color, "marker must be strictly inside a colored region"
        adjacent.append(value)

    assert len(set(adjacent)) == 1, "each marker must have a unique 4-neighbor owner color"
    return int(adjacent[0])


def _vote_counts(grid: np.ndarray) -> tuple[int, Counter[int]]:
    marker_color = _unique_least_frequent_non_bg_color(grid)
    counts: Counter[int] = Counter()
    for r, c in np.argwhere(grid == marker_color):
        owner = _marker_owner_color(grid, marker_color, int(r), int(c))
        counts[owner] += 1

    assert counts, "expected at least one marker pixel"
    tallies = list(counts.values())
    assert len(tallies) == len(set(tallies)), "marker counts must be distinct"
    return marker_color, counts


def _concentric_output(sorted_colors: list[int]) -> np.ndarray:
    n = len(sorted_colors)
    side = 2 * n - 1
    out = np.empty((side, side), dtype=int)
    for layer, color in enumerate(sorted_colors):
        out[layer : side - layer, layer : side - layer] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build a compact concentric square whose layer order is decided by marker votes inside separated color regions.

    Input structure:
    - The background is the dominant color, with 3-5 separated solid rectangular color regions.
    - One unique least-frequent non-background color appears only as single-pixel markers placed strictly inside those rectangles.
    - Every marker's 4-neighbors are all the base color of the rectangle that contains it.

    Rule:
    - Find the unique least-frequent non-background color and treat those pixels as markers.
    - Count how many markers belong to each surrounding region color by reading every marker's 4-neighbor owner color.
    - Sort the region colors by marker count from highest to lowest, then create a (2N-1) x (2N-1) square whose concentric layers use that outside-to-inside order.

    Key insight:
    - The marker pixels are only votes: they never appear in the output directly, but their per-region counts determine the spatial ordering of the concentric layers.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    _marker_color, counts = _vote_counts(g)
    ordered_colors = [
        color for color, _count in sorted(counts.items(), key=lambda item: item[1], reverse=True)
    ]
    return _concentric_output(ordered_colors)


def _choose_region_count() -> int:
    # The prompt allows 3-5 regions, but 5 regions are incompatible with
    # 5-12 total markers under the additional "all counts distinct and >=1"
    # constraint. The solver supports 5; the generator samples the feasible cases.
    return random.choice([3, 4])


def _all_count_tuples(n_regions: int) -> list[tuple[int, ...]]:
    results: list[tuple[int, ...]] = []

    def rec(start: int, remaining_regions: int, remaining_total: int, current: list[int]) -> None:
        if remaining_regions == 0:
            if remaining_total == 0:
                results.append(tuple(current))
            return

        min_rest = sum(range(start + 1, start + remaining_regions))
        for value in range(start, remaining_total + 1):
            if remaining_total - value < min_rest:
                break
            current.append(value)
            rec(value + 1, remaining_regions - 1, remaining_total - value, current)
            current.pop()

    for total in range(5, 13):
        rec(1, n_regions, total, [])

    assert results, "expected at least one feasible distinct count tuple"
    return results


def _marker_capacity(height: int, width: int) -> int:
    interior_h = height - 2
    interior_w = width - 2
    assert interior_h > 0 and interior_w > 0
    return (interior_h * interior_w + 1) // 2


def _sample_rect_size(marker_count: int) -> tuple[int, int]:
    candidates = []
    for height in range(4, 9):
        for width in range(4, 9):
            if _marker_capacity(height, width) >= marker_count:
                candidates.append((height, width))
    assert candidates

    candidates.sort(key=lambda hw: (hw[0] * hw[1], abs(hw[0] - hw[1]), hw[0]))
    near_min = candidates[: max(4, len(candidates) // 2)]
    return random.choice(near_min)


def _can_place(rect: tuple[int, int, int, int], placed: list[tuple[int, int, int, int]], shape: tuple[int, int]) -> bool:
    r0, c0, r1, c1 = rect
    if r0 < 0 or c0 < 0 or r1 > shape[0] or c1 > shape[1]:
        return False

    for pr0, pc0, pr1, pc1 in placed:
        if not (r1 + 1 < pr0 or pr1 + 1 < r0 or c1 + 1 < pc0 or pc1 + 1 < c0):
            return False
    return True


def _place_rectangles(shape: tuple[int, int], sizes: list[tuple[int, int]]) -> list[tuple[int, int, int, int]] | None:
    order = list(range(len(sizes)))
    random.shuffle(order)
    order.sort(key=lambda idx: sizes[idx][0] * sizes[idx][1], reverse=True)

    placed_by_index: list[tuple[int, int, int, int] | None] = [None] * len(sizes)
    placed_rects: list[tuple[int, int, int, int]] = []

    for idx in order:
        height, width = sizes[idx]
        candidates = []
        for r0 in range(shape[0] - height + 1):
            for c0 in range(shape[1] - width + 1):
                rect = (r0, c0, r0 + height, c0 + width)
                if _can_place(rect, placed_rects, shape):
                    candidates.append(rect)
        if not candidates:
            return None

        center_r = (shape[0] - height) / 2.0
        center_c = (shape[1] - width) / 2.0
        candidates.sort(
            key=lambda rect: (
                abs(((rect[0] + rect[2] - height) / 2.0) - center_r)
                + abs(((rect[1] + rect[3] - width) / 2.0) - center_c),
                random.random(),
            )
        )
        chosen = random.choice(candidates[: max(8, len(candidates) // 3)])
        placed_rects.append(chosen)
        placed_by_index[idx] = chosen

    return [rect for rect in placed_by_index if rect is not None]


def _choose_marker_cells(rect: tuple[int, int, int, int], count: int) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = rect
    candidates = [[], []]
    for r in range(r0 + 1, r1 - 1):
        for c in range(c0 + 1, c1 - 1):
            candidates[(r + c) & 1].append((r, c))

    valid_groups = [group for group in candidates if len(group) >= count]
    assert valid_groups, "rectangle interior must be able to host its markers"
    return random.sample(random.choice(valid_groups), count)


def _expected_output(region_colors: list[int], marker_counts: list[int]) -> np.ndarray:
    ordering = [
        color
        for color, _count in sorted(
            zip(region_colors, marker_counts),
            key=lambda item: item[1],
            reverse=True,
        )
    ]
    return _concentric_output(ordering)


def _validate_grid(
    grid: np.ndarray,
    rects: list[tuple[int, int, int, int]],
    region_colors: list[int],
    marker_color: int,
    marker_counts: list[int],
) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False

    bg = _background_color(grid)
    if bg != BACKGROUND:
        return False

    try:
        observed_marker, vote_counts = _vote_counts(grid)
    except AssertionError:
        return False

    if observed_marker != marker_color:
        return False

    if sorted(vote_counts.keys()) != sorted(region_colors):
        return False

    expected_counts = {color: count for color, count in zip(region_colors, marker_counts)}
    if any(vote_counts[color] != expected_counts[color] for color in region_colors):
        return False

    for rect, color, count in zip(rects, region_colors, marker_counts):
        r0, c0, r1, c1 = rect
        patch = grid[r0:r1, c0:c1]
        if not np.all((patch == color) | (patch == marker_color)):
            return False
        if np.count_nonzero(patch == marker_color) != count:
            return False

        expanded_r0 = max(0, r0 - 1)
        expanded_c0 = max(0, c0 - 1)
        expanded_r1 = min(grid.shape[0], r1 + 1)
        expanded_c1 = min(grid.shape[1], c1 + 1)
        halo = grid[expanded_r0:expanded_r1, expanded_c0:expanded_c1].copy()
        halo[(r0 - expanded_r0) : (r1 - expanded_r0), (c0 - expanded_c0) : (c1 - expanded_c0)] = BACKGROUND
        if np.any(halo != BACKGROUND):
            return False

    expected = _expected_output(region_colors, marker_counts)
    try:
        actual = main(grid)
    except AssertionError:
        return False
    return np.array_equal(actual, expected)


def _attempt_generate() -> np.ndarray | None:
    n_regions = _choose_region_count()
    marker_counts = list(random.choice(_all_count_tuples(n_regions)))
    random.shuffle(marker_counts)

    sizes = [_sample_rect_size(count) for count in marker_counts]
    min_height = max(14, sum(height for height, _width in sizes) // 2 + 3)
    min_width = max(14, sum(width for _height, width in sizes) // 2 + 3)
    max_height = min(24, max(min_height, 20))
    max_width = min(24, max(min_width, 20))

    if min_height > max_height or min_width > max_width:
        return None

    shape = (random.randint(min_height, max_height), random.randint(min_width, max_width))
    rects = _place_rectangles(shape, sizes)
    if rects is None:
        return None

    colors = random.sample(ALL_NON_BG_COLORS, n_regions + 1)
    marker_color = colors[0]
    region_colors = colors[1:]

    grid = np.full(shape, BACKGROUND, dtype=int)
    for rect, color in zip(rects, region_colors):
        r0, c0, r1, c1 = rect
        grid[r0:r1, c0:c1] = color

    for rect, count in zip(rects, marker_counts):
        for r, c in _choose_marker_cells(rect, count):
            grid[r, c] = marker_color

    if not _validate_grid(grid, rects, region_colors, marker_color, marker_counts):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed to produce a valid puzzle instance")
