from __future__ import annotations

import random

import numpy as np


BG = 8
PALETTE = (1, 2, 3, 4, 6, 7, 9)
GRID_HEIGHT_RANGE = (18, 26)
GRID_WIDTH_RANGE = (18, 26)
COLOR_COUNT_RANGE = (2, 4)
POINT_COUNT_RANGE = (3, 7)
MAX_GENERATION_ATTEMPTS = 300
POINT_RETRY_LIMIT = 200

INTERIOR_PAIR_TEMPLATES: tuple[tuple[tuple[int, int], tuple[int, int]], ...] = (
    ((-2, -2), (2, 2)),
    ((-2, 2), (2, -2)),
    ((-4, 0), (4, 0)),
    ((0, -4), (0, 4)),
    ((-4, -2), (4, 2)),
    ((-4, 2), (4, -2)),
    ((-2, -4), (2, 4)),
    ((-2, 4), (2, -4)),
)
INTERIOR_EXTRA_OFFSETS: tuple[tuple[int, int], ...] = (
    (0, 2),
    (2, 0),
    (0, 4),
    (2, 2),
    (4, 0),
    (4, 2),
)
TOP_EDGE_PAIR_TEMPLATES: tuple[tuple[tuple[int, int], tuple[int, int]], ...] = (
    ((0, -2), (2, 2)),
    ((0, 2), (2, -2)),
    ((0, -4), (2, 4)),
    ((0, 4), (2, -4)),
    ((0, -2), (4, 2)),
    ((0, 2), (4, -2)),
    ((0, -4), (4, 4)),
    ((0, 4), (4, -4)),
)
TOP_EDGE_EXTRA_OFFSETS: tuple[tuple[int, int], ...] = (
    (2, 0),
    (2, 2),
    (4, 0),
    (4, 2),
)


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _lower_median(values: np.ndarray) -> int:
    ordered = np.sort(np.asarray(values, dtype=int))
    return int(ordered[(len(ordered) - 1) // 2])


def _cross_cells(center: tuple[int, int], shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = map(int, shape)
    r0, c0 = center
    cells: list[tuple[int, int]] = []
    for dr, dc in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
        r = r0 + dr
        c = c0 + dc
        if 0 <= r < h and 0 <= c < w:
            cells.append((r, c))
    return cells


def _chebyshev_far(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1])) > 1


def _points_are_clear(points: list[tuple[int, int]], occupied: list[tuple[int, int]], shape: tuple[int, int]) -> bool:
    h, w = map(int, shape)
    seen: set[tuple[int, int]] = set()
    for point in points:
        r, c = point
        if not (0 <= r < h and 0 <= c < w):
            return False
        if point in seen:
            return False
        seen.add(point)

    point_list = list(seen)
    for i, point in enumerate(point_list):
        for other in point_list[i + 1 :]:
            if not _chebyshev_far(point, other):
                return False
        for other in occupied:
            if not _chebyshev_far(point, other):
                return False
    return True


def _offset_points(
    center: tuple[int, int],
    offsets: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    cr, cc = center
    return [(cr + dr, cc + dc) for dr, dc in offsets]


def _slot_ranges(height: int, width: int) -> dict[str, tuple[int, int, int, int]]:
    half_h = height // 2
    half_w = width // 2
    return {
        "top_left": (4, max(4, half_h - 5), 4, max(4, half_w - 5)),
        "top_right": (4, max(4, half_h - 5), min(width - 5, half_w + 4), width - 5),
        "bottom_left": (half_h + 2, height - 5, 4, max(4, half_w - 5)),
        "bottom_right": (half_h + 2, height - 5, min(width - 5, half_w + 4), width - 5),
    }


def _sample_center(slot_name: str, height: int, width: int, *, edge: bool) -> tuple[int, int]:
    row_lo, row_hi, col_lo, col_hi = _slot_ranges(height, width)[slot_name]
    col = random.randint(col_lo, col_hi)
    if edge:
        return (0, col)
    row = random.randint(row_lo, row_hi)
    return (row, col)


def _choose_counts(num_colors: int) -> list[int]:
    counts = [random.randint(*POINT_COUNT_RANGE) for _ in range(num_colors)]
    if all(count % 2 == counts[0] % 2 for count in counts):
        idx = random.randrange(num_colors)
        if counts[idx] % 2 == 0:
            counts[idx] = 3 if counts[idx] == 4 else counts[idx] - 1
        else:
            counts[idx] = 4 if counts[idx] == 3 else counts[idx] - 1
    return counts


def _build_points(
    center: tuple[int, int],
    mode: str,
    count: int,
    occupied: list[tuple[int, int]],
    shape: tuple[int, int],
) -> list[tuple[int, int]] | None:
    pair_templates = INTERIOR_PAIR_TEMPLATES if mode == "interior" else TOP_EDGE_PAIR_TEMPLATES
    extra_offsets = INTERIOR_EXTRA_OFFSETS if mode == "interior" else TOP_EDGE_EXTRA_OFFSETS
    pair_count = count // 2 if count % 2 == 1 else count // 2 - 1

    for _ in range(POINT_RETRY_LIMIT):
        offsets: list[tuple[int, int]] = [(0, 0)]
        for pair in random.sample(pair_templates, pair_count):
            a, b = pair
            if random.choice((True, False)):
                offsets.extend([a, b])
            else:
                offsets.extend([b, a])
        if count % 2 == 0:
            offsets.append(random.choice(extra_offsets))

        points = _offset_points(center, offsets)
        if not _points_are_clear(points, occupied, shape):
            continue

        rows = np.array([r for r, _c in points], dtype=int)
        cols = np.array([c for _r, c in points], dtype=int)
        if _lower_median(rows) != center[0]:
            continue
        if _lower_median(cols) != center[1]:
            continue
        return points
    return None


def _render_expected(
    shape: tuple[int, int],
    assignments: list[tuple[int, tuple[int, int]]],
) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for color, center in assignments:
        for r, c in _cross_cells(center, shape):
            out[r, c] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Collapse each color's scatter of single pixels to one median-position cross.

    Input structure:
    - The background is a single dominant color.
    - Each foreground color appears only as isolated one-cell scatter points.
    - There are 2-4 foreground colors, and each color contributes several points spread across the grid.

    Rule:
    - For each foreground color, collect all of its point coordinates.
    - Compute the visual center of that color by taking the lower median row and lower median column independently.
    - Clear the whole canvas back to background, then draw a same-colored 3x3 cross at each center using the center cell plus its four cardinal neighbors.
    - If any arm would leave the grid, keep only the in-bounds part of that cross.

    Key insight:
    - The original scatter pattern matters only through per-color coordinate medians; every color is compressed to one fixed cross marker.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _dominant_color(g)

    out = np.full(g.shape, bg, dtype=int)
    for color in np.unique(g):
        color = int(color)
        if color == bg:
            continue
        coords = np.argwhere(g == color)
        assert len(coords) > 0, "foreground colors must be present"
        center = (_lower_median(coords[:, 0]), _lower_median(coords[:, 1]))
        for r, c in _cross_cells(center, g.shape):
            out[r, c] = color
    return out


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    num_colors = random.randint(*COLOR_COUNT_RANGE)
    colors = random.sample(PALETTE, num_colors)
    counts = _choose_counts(num_colors)

    edge_slot = random.choice(("top_left", "top_right"))
    interior_slot = random.choice(("bottom_left", "bottom_right"))
    selected_slots = [edge_slot, interior_slot]
    remaining_slots = [slot for slot in ("top_left", "top_right", "bottom_left", "bottom_right") if slot not in selected_slots]
    selected_slots.extend(random.sample(remaining_slots, num_colors - 2))

    assignments: list[tuple[int, tuple[int, int], list[tuple[int, int]]]] = []
    occupied: list[tuple[int, int]] = []

    for index, (slot_name, color, count) in enumerate(zip(selected_slots, colors, counts)):
        if slot_name == edge_slot:
            mode = "top_edge"
            center = _sample_center(slot_name, height, width, edge=True)
        elif slot_name.startswith("top_") and index >= 2 and random.choice((True, False)):
            mode = "top_edge"
            center = _sample_center(slot_name, height, width, edge=True)
        else:
            mode = "interior"
            center = _sample_center(slot_name, height, width, edge=False)

        points = _build_points(center, mode, count, occupied, (height, width))
        if points is None:
            return None
        occupied.extend(points)
        assignments.append((color, center, points))

    grid = np.full((height, width), BG, dtype=int)
    for color, _center, points in assignments:
        for r, c in points:
            grid[r, c] = color

    expected = _render_expected(grid.shape, [(color, center) for color, center, _points in assignments])
    if not np.array_equal(main(grid), expected):
        return None

    non_bg = np.argwhere(grid != BG)
    for i, point in enumerate(non_bg):
        a = (int(point[0]), int(point[1]))
        for other in non_bg[i + 1 :]:
            b = (int(other[0]), int(other[1]))
            if not _chebyshev_far(a, b):
                return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 2")
