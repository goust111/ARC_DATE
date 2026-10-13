from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
SEPARATOR = 2
MARKER = 4
PATTERN_COLORS = [1, 3, 6, 7, 8, 9]
HORIZONTAL_DIRS = ("top", "bottom")
VERTICAL_DIRS = ("left", "right")
ALL_DIRS = HORIZONTAL_DIRS + VERTICAL_DIRS
MAX_GENERATION_ATTEMPTS = 300


def _intervals(separator_indices: list[int], length: int) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    start = 0
    for idx in sorted(separator_indices):
        if start < idx:
            intervals.append((start, idx))
        start = idx + 1
    if start < length:
        intervals.append((start, length))
    return intervals


def _half_masks(height: int, width: int) -> dict[str, np.ndarray]:
    rows = np.arange(height)[:, None]
    cols = np.arange(width)[None, :]
    return {
        "top": np.broadcast_to(rows < height // 2, (height, width)),
        "bottom": np.broadcast_to(rows >= height - height // 2, (height, width)),
        "left": np.broadcast_to(cols < width // 2, (height, width)),
        "right": np.broadcast_to(cols >= width - width // 2, (height, width)),
    }


def _orthogonal_score(pattern: np.ndarray, direction: str, masks: dict[str, np.ndarray]) -> int:
    if direction in HORIZONTAL_DIRS:
        spans_both = bool(np.any(pattern & masks["left"]) and np.any(pattern & masks["right"]))
        return 10 * spans_both + int(np.unique(np.argwhere(pattern)[:, 1]).size)
    spans_both = bool(np.any(pattern & masks["top"]) and np.any(pattern & masks["bottom"]))
    return 10 * spans_both + int(np.unique(np.argwhere(pattern)[:, 0]).size)


def _mirror_region(region: np.ndarray) -> np.ndarray:
    height, width = region.shape
    masks = _half_masks(height, width)
    pattern = (region != BACKGROUND) & (region != MARKER)
    marker_coords = np.argwhere(region == MARKER)
    assert len(marker_coords) == 1, "each region must contain exactly one marker"
    marker_r, marker_c = map(int, marker_coords[0])
    assert np.any(pattern), "each region must contain a visible pattern"

    candidates: list[tuple[int, np.ndarray, str]] = []
    for direction, source_name, axis in (
        ("top", "bottom", "horizontal"),
        ("bottom", "top", "horizontal"),
        ("left", "right", "vertical"),
        ("right", "left", "vertical"),
    ):
        target_mask = masks[direction]
        source_mask = masks[source_name]
        if not target_mask[marker_r, marker_c]:
            continue
        if not np.any(pattern & source_mask):
            continue
        if np.any(pattern & ~source_mask):
            continue

        mirrored = region.copy()
        mirrored[region == MARKER] = BACKGROUND
        for r, c in np.argwhere(pattern):
            rr = height - 1 - r if axis == "horizontal" else r
            cc = width - 1 - c if axis == "vertical" else c
            mirrored[rr, cc] = region[r, c]
        candidates.append((_orthogonal_score(pattern, direction, masks), mirrored, direction))

    assert candidates, "failed to infer a valid mirror direction"
    candidates.sort(key=lambda item: item[0], reverse=True)
    if len(candidates) >= 2:
        assert candidates[0][0] > candidates[1][0], "region direction is ambiguous"
    return candidates[0][1]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each red-separated region by mirroring its visible half into the marker half.

    Input structure:
    - The grid is split by full red separator rows and columns into rectangular regions.
    - Each region contains a small non-background pattern confined to exactly one top/bottom/left/right half.
    - The opposite empty half contains exactly one yellow marker pixel that encodes the local mirror direction.

    Rule:
    - Detect the full red separator lines and process each region independently.
    - Inside a region, infer whether the marker indicates the top, bottom, left, or right empty half.
    - Mirror the visible pattern across the corresponding central axis into that empty half and remove the marker.

    Key insight:
    - The mirror axis is not global; every region uses its own separator-bounded geometry plus marker position to choose the reflection direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = g.shape

    separator_rows = [r for r in range(height) if np.all(g[r, :] == SEPARATOR)]
    separator_cols = [c for c in range(width) if np.all(g[:, c] == SEPARATOR)]
    row_intervals = _intervals(separator_rows, height)
    col_intervals = _intervals(separator_cols, width)
    assert row_intervals and col_intervals, "expected at least one region"

    out = g.copy()
    for r0, r1 in row_intervals:
        for c0, c1 in col_intervals:
            solved = _mirror_region(g[r0:r1, c0:c1])
            out[r0:r1, c0:c1] = solved
    return out


def _source_row_range(direction: str, height: int) -> range:
    if direction == "top":
        return range(0, height // 2)
    if direction == "bottom":
        return range(height - height // 2, height)
    return range(height)


def _source_col_range(direction: str, width: int) -> range:
    if direction == "left":
        return range(0, width // 2)
    if direction == "right":
        return range(width - width // 2, width)
    return range(width)


def _target_cells(direction: str, height: int, width: int) -> list[tuple[int, int]]:
    masks = _half_masks(height, width)
    return [tuple(map(int, rc)) for rc in np.argwhere(masks[direction])]


def _build_horizontal_shape(direction: str, height: int, width: int) -> list[tuple[int, int]]:
    rows = list(_source_row_range(direction, height))
    left_cols = [c for c in range(width) if c < width // 2]
    right_cols = [c for c in range(width) if c >= width - width // 2]
    spans = [(lc, rc) for lc in left_cols for rc in right_cols if 3 <= rc - lc + 1 <= 4]
    row = random.choice(rows)
    start_c, end_c = random.choice(spans)
    cells = {(row, c) for c in range(start_c, end_c + 1)}

    extra_rows = [r for r in rows if r != row]
    if extra_rows and len(cells) < 5 and random.random() < 0.75:
        extra_row = random.choice(extra_rows)
        extra_col = random.choice([start_c, end_c, (start_c + end_c) // 2])
        cells.add((extra_row, extra_col))
    return sorted(cells)


def _build_vertical_shape(direction: str, height: int, width: int) -> list[tuple[int, int]]:
    cols = list(_source_col_range(direction, width))
    top_rows = [r for r in range(height) if r < height // 2]
    bottom_rows = [r for r in range(height) if r >= height - height // 2]
    spans = [(tr, br) for tr in top_rows for br in bottom_rows if 3 <= br - tr + 1 <= 4]
    col = random.choice(cols)
    start_r, end_r = random.choice(spans)
    cells = {(r, col) for r in range(start_r, end_r + 1)}

    extra_cols = [c for c in cols if c != col]
    if extra_cols and len(cells) < 5 and random.random() < 0.75:
        extra_col = random.choice(extra_cols)
        extra_row = random.choice([start_r, end_r, (start_r + end_r) // 2])
        cells.add((extra_row, extra_col))
    return sorted(cells)


def _mirrored_cells(cells: list[tuple[int, int]], direction: str, height: int, width: int) -> list[tuple[int, int]]:
    if direction in HORIZONTAL_DIRS:
        return [(height - 1 - r, c) for r, c in cells]
    return [(r, width - 1 - c) for r, c in cells]


def _choose_marker(
    direction: str,
    height: int,
    width: int,
    mirrored_cells: list[tuple[int, int]],
) -> tuple[int, int]:
    target = _target_cells(direction, height, width)
    if random.random() < 0.5:
        return random.choice(mirrored_cells)
    other_choices = [cell for cell in target if cell not in mirrored_cells]
    return random.choice(other_choices or target)


def _build_region(height: int, width: int, direction: str, color: int) -> np.ndarray:
    region = np.zeros((height, width), dtype=int)
    if direction in HORIZONTAL_DIRS:
        shape_cells = _build_horizontal_shape(direction, height, width)
    else:
        shape_cells = _build_vertical_shape(direction, height, width)

    for r, c in shape_cells:
        region[r, c] = color

    marker_r, marker_c = _choose_marker(
        direction,
        height,
        width,
        _mirrored_cells(shape_cells, direction, height, width),
    )
    region[marker_r, marker_c] = MARKER
    return region


def _sample_directions(count: int) -> list[str]:
    while True:
        directions = [random.choice(ALL_DIRS) for _ in range(count)]
        if not any(direction in HORIZONTAL_DIRS for direction in directions):
            continue
        if not any(direction in VERTICAL_DIRS for direction in directions):
            continue
        if len(set(directions)) < 2:
            continue
        return directions


def _gen_attempt() -> np.ndarray | None:
    row_heights = [random.randint(5, 6) for _ in range(2)]
    col_count = random.choice((2, 3))
    col_widths = [random.randint(5, 6) for _ in range(col_count)]
    total_h = sum(row_heights) + 1
    total_w = sum(col_widths) + (col_count - 1)
    if total_h > 30 or total_w > 30:
        return None

    grid = np.zeros((total_h, total_w), dtype=int)
    separator_row = row_heights[0]
    grid[separator_row, :] = SEPARATOR

    separator_cols: list[int] = []
    cursor = 0
    for idx, width in enumerate(col_widths[:-1]):
        cursor += width
        separator_cols.append(cursor)
        grid[:, cursor] = SEPARATOR
        cursor += 1

    directions = _sample_directions(len(row_heights) * col_count)
    colors = random.sample(PATTERN_COLORS, k=len(directions))

    region_idx = 0
    row_start = 0
    for region_h in row_heights:
        col_start = 0
        for region_w in col_widths:
            direction = directions[region_idx]
            color = colors[region_idx]
            region = _build_region(region_h, region_w, direction, color)
            grid[row_start : row_start + region_h, col_start : col_start + region_w] = region
            region_idx += 1
            col_start += region_w + 1
        row_start += region_h + 1

    try:
        solved = main(grid)
    except AssertionError:
        return None
    if np.array_equal(solved, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
