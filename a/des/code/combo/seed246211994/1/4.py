from __future__ import annotations

import random

import numpy as np


TRANSFORM_MODES = ("lr", "ud", "rot180")
PALETTE = tuple(range(10))
MIN_REGION_ROWS = 2
MAX_REGION_ROWS = 3
MIN_REGION_COLS = 2
MAX_REGION_COLS = 4
MIN_REGION_HEIGHT = 5
MAX_CANVAS_SIZE = 30
GENERATION_ATTEMPTS = 300
REGION_ATTEMPTS = 120


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx, flag in enumerate(mask.tolist()):
        if flag:
            if start is None:
                start = idx
        elif start is not None:
            runs.append((start, idx))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def _intervals(length: int, sep_runs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    cursor = 0
    for start, end in sep_runs:
        if cursor < start:
            intervals.append((cursor, start))
        cursor = end
    if cursor < length:
        intervals.append((cursor, length))
    return intervals


def _find_partition(grid: np.ndarray) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]]]:
    h, w = grid.shape
    candidates: list[tuple[int, list[tuple[int, int]], list[tuple[int, int]]]] = []
    for color in map(int, np.unique(grid)):
        row_mask = np.all(grid == color, axis=1)
        col_mask = np.all(grid == color, axis=0)
        if not row_mask.any() or not col_mask.any():
            continue

        row_intervals = _intervals(h, _runs(row_mask))
        col_intervals = _intervals(w, _runs(col_mask))
        if len(row_intervals) < 2 or len(col_intervals) < 2:
            continue
        if any(r1 <= r0 for r0, r1 in row_intervals) or any(c1 <= c0 for c0, c1 in col_intervals):
            continue
        if len({r1 - r0 for r0, r1 in row_intervals}) != 1:
            continue
        if len({c1 - c0 for c0, c1 in col_intervals}) != 1:
            continue
        candidates.append((color, row_intervals, col_intervals))

    assert len(candidates) == 1, "expected a unique separator color"
    return candidates[0]


def _mode_from_markers(marker_points: np.ndarray) -> str:
    (r1, c1), (r2, c2) = marker_points.tolist()
    if r1 == r2:
        return "lr"
    if c1 == c2:
        return "ud"
    return "rot180"


def _apply_transform(region: np.ndarray, mode: str) -> np.ndarray:
    if mode == "lr":
        return np.fliplr(region)
    if mode == "ud":
        return np.flipud(region)
    assert mode == "rot180"
    return np.rot90(region, 2)


def main(grid: np.ndarray) -> np.ndarray:
    """
    根据每个分区内两个标记像素的相对位置关系，对区内图案做翻转或 180 度旋转，并擦除标记。
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _most_common_color(g)
    _separator, row_intervals, col_intervals = _find_partition(g)

    out = g.copy()
    for r0, r1 in row_intervals:
        for c0, c1 in col_intervals:
            region = g[r0:r1, c0:c1]
            values, counts = np.unique(region, return_counts=True)
            count_map = {
                int(color): int(count)
                for color, count in zip(values.tolist(), counts.tolist())
                if int(color) != bg
            }
            marker_candidates = [color for color, count in count_map.items() if count == 2]
            assert len(marker_candidates) == 1, "each region must expose exactly one marker color"
            marker_color = marker_candidates[0]
            marker_points = np.argwhere(region == marker_color)
            assert marker_points.shape == (2, 2)

            payload = region.copy()
            payload[payload == marker_color] = bg
            out[r0:r1, c0:c1] = _apply_transform(payload, _mode_from_markers(marker_points))

    return out


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    neighbors: list[tuple[int, int]] = []
    if r > 0:
        neighbors.append((r - 1, c))
    if r + 1 < h:
        neighbors.append((r + 1, c))
    if c > 0:
        neighbors.append((r, c - 1))
    if c + 1 < w:
        neighbors.append((r, c + 1))
    return neighbors


def _random_connected_cells(height: int, width: int, size: int) -> set[tuple[int, int]] | None:
    start = (random.randrange(height), random.randrange(width))
    cells = {start}
    frontier = set(_neighbors4(start[0], start[1], height, width))

    while len(cells) < size:
        if not frontier:
            return None
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        cells.add(cell)
        for nbr in _neighbors4(cell[0], cell[1], height, width):
            if nbr not in cells:
                frontier.add(nbr)
    return cells


def _choose_markers(region: np.ndarray, mode: str, bg: int) -> list[tuple[int, int]] | None:
    height, width = region.shape
    empties = [(r, c) for r in range(height) for c in range(width) if int(region[r, c]) == bg]
    if len(empties) < 2:
        return None

    if mode == "lr":
        row_groups: dict[int, list[int]] = {}
        for r, c in empties:
            row_groups.setdefault(r, []).append(c)
        rows = [r for r, cols in row_groups.items() if len(cols) >= 2]
        if not rows:
            return None
        row = random.choice(rows)
        c1, c2 = random.sample(row_groups[row], 2)
        return [(row, c1), (row, c2)]

    if mode == "ud":
        col_groups: dict[int, list[int]] = {}
        for r, c in empties:
            col_groups.setdefault(c, []).append(r)
        cols = [c for c, rows in col_groups.items() if len(rows) >= 2]
        if not cols:
            return None
        col = random.choice(cols)
        r1, r2 = random.sample(col_groups[col], 2)
        return [(r1, col), (r2, col)]

    pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for idx, first in enumerate(empties):
        for second in empties[idx + 1 :]:
            if first[0] != second[0] and first[1] != second[1]:
                pairs.append((first, second))
    if not pairs:
        return None
    first, second = random.choice(pairs)
    return [first, second]


def _balanced_modes(total_regions: int) -> list[str]:
    modes = list(TRANSFORM_MODES) * (total_regions // len(TRANSFORM_MODES))
    modes.extend(random.sample(TRANSFORM_MODES, total_regions % len(TRANSFORM_MODES)))
    random.shuffle(modes)
    return modes


def _make_region(height: int, width: int, bg: int, palette: list[int], mode: str) -> np.ndarray | None:
    area = height * width
    min_size = 4
    max_size = min(area // 2, area - 6)
    if max_size < min_size:
        return None

    for _ in range(REGION_ATTEMPTS):
        pattern_color, marker_color = random.sample(palette, 2)
        size = random.randint(min_size, max_size)
        cells = _random_connected_cells(height, width, size)
        if cells is None:
            continue

        region = np.full((height, width), bg, dtype=int)
        for r, c in cells:
            region[r, c] = pattern_color

        if np.array_equal(region, _apply_transform(region, mode)):
            continue

        markers = _choose_markers(region, mode, bg)
        if markers is None:
            continue
        for r, c in markers:
            region[r, c] = marker_color
        return region
    return None


def _dominant_background(grid: np.ndarray, bg: int) -> bool:
    values, counts = np.unique(grid, return_counts=True)
    count_map = {int(color): int(count) for color, count in zip(values.tolist(), counts.tolist())}
    bg_count = count_map.get(bg, 0)
    return all(bg_count > count for color, count in count_map.items() if color != bg)


def _generate_attempt() -> np.ndarray | None:
    region_rows = random.randint(MIN_REGION_ROWS, MAX_REGION_ROWS)
    region_cols = random.randint(MIN_REGION_COLS, MAX_REGION_COLS)
    max_region_height = min(MAX_CANVAS_SIZE, (MAX_CANVAS_SIZE - (region_rows - 1)) // region_rows)
    max_region_width = min(MAX_CANVAS_SIZE, (MAX_CANVAS_SIZE - (region_cols - 1)) // region_cols)
    max_region_height = min(max_region_height, 8)
    max_region_width = min(max_region_width, 8)
    if max_region_height < MIN_REGION_HEIGHT or max_region_width < MIN_REGION_HEIGHT:
        return None

    region_height = random.randint(MIN_REGION_HEIGHT, max_region_height)
    region_width = random.randint(MIN_REGION_HEIGHT, max_region_width)
    bg, separator = random.sample(PALETTE, 2)
    palette = [color for color in PALETTE if color not in (bg, separator)]

    grid_height = region_rows * region_height + (region_rows - 1)
    grid_width = region_cols * region_width + (region_cols - 1)
    grid = np.full((grid_height, grid_width), bg, dtype=int)

    for band in range(1, region_rows):
        grid[band * region_height + band - 1, :] = separator
    for band in range(1, region_cols):
        grid[:, band * region_width + band - 1] = separator

    modes = _balanced_modes(region_rows * region_cols)
    idx = 0
    for br in range(region_rows):
        for bc in range(region_cols):
            region = _make_region(region_height, region_width, bg, palette, modes[idx])
            if region is None:
                return None
            r0 = br * (region_height + 1)
            c0 = bc * (region_width + 1)
            grid[r0 : r0 + region_height, c0 : c0 + region_width] = region
            idx += 1

    if not _dominant_background(grid, bg):
        return None

    try:
        separator2, row_intervals, col_intervals = _find_partition(grid)
    except AssertionError:
        return None
    if separator2 != separator:
        return None
    if len(row_intervals) != region_rows or len(col_intervals) != region_cols:
        return None

    try:
        output = main(grid)
    except AssertionError:
        return None
    if output.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
