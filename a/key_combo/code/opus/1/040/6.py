from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
SEED_COLOR = 5
STRIP_PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
GRID_MIN = 15
GRID_MAX = 27
MIN_RINGS = 2
MAX_RINGS = 7
MAX_RETRIES = 500


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _square_border_cells(
    shape: tuple[int, int], center: tuple[int, int], radius: int
) -> list[tuple[int, int]]:
    cr, cc = center
    top = cr - radius
    bottom = cr + radius
    left = cc - radius
    right = cc + radius
    cells: list[tuple[int, int]] = []

    for c in range(left, right + 1):
        if _in_bounds(shape, top, c):
            cells.append((top, c))
        if bottom != top and _in_bounds(shape, bottom, c):
            cells.append((bottom, c))

    for r in range(top + 1, bottom):
        if _in_bounds(shape, r, left):
            cells.append((r, left))
        if right != left and _in_bounds(shape, r, right):
            cells.append((r, right))

    return cells


def _find_seed_pattern(
    grid: np.ndarray,
) -> tuple[tuple[int, int], int, set[tuple[int, int]]]:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(grid == SEED_COLOR)]
    assert len(coords) == 3, "expected exactly three gray seed pixels"

    for idx, center in enumerate(coords):
        others = [coords[j] for j in range(3) if j != idx]
        if (
            others[0][0] + others[1][0] == 2 * center[0]
            and others[0][1] + others[1][1] == 2 * center[1]
        ):
            dr = others[0][0] - center[0]
            dc = others[0][1] - center[1]
            assert dr != 0, "seed spacing must be positive"
            if abs(dr) == abs(dc):
                return center, abs(dr), {center, others[0], others[1]}
    raise AssertionError("gray pixels must form an equally spaced 45-degree diagonal")


def _find_color_strip(
    grid: np.ndarray,
) -> tuple[list[tuple[int, int]], list[int]]:
    coords = np.argwhere((grid != BACKGROUND) & (grid != SEED_COLOR))
    assert len(coords) >= 3, "expected a color strip with length at least 3"

    r0, c0 = map(int, coords.min(axis=0))
    r1, c1 = map(int, coords.max(axis=0))
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    assert height == 1 or width == 1, "color strip must be 1xN or Nx1"
    assert len(coords) == height * width, "color strip must be tightly packed"

    strip_cells: list[tuple[int, int]] = []
    strip_colors: list[int] = []
    if height == 1:
        for c in range(c0, c1 + 1):
            strip_cells.append((r0, c))
            strip_colors.append(int(grid[r0, c]))
    else:
        for r in range(r0, r1 + 1):
            strip_cells.append((r, c0))
            strip_colors.append(int(grid[r, c0]))

    assert all(color not in (BACKGROUND, SEED_COLOR) for color in strip_colors)
    return strip_cells, strip_colors


def _ring_count(shape: tuple[int, int], center: tuple[int, int], spacing: int) -> int:
    cr, cc = center
    max_axis_distance = max(cr, shape[0] - 1 - cr, cc, shape[1] - 1 - cc)
    return max_axis_distance // spacing


def _render_output_from_spec(
    shape: tuple[int, int],
    center: tuple[int, int],
    spacing: int,
    seed_cells: set[tuple[int, int]],
    strip_colors: list[int],
) -> np.ndarray:
    out = np.full(shape, BACKGROUND, dtype=int)
    out_colors = [int(color) for color in strip_colors]
    assert out_colors, "need at least one strip color"

    for r, c in seed_cells:
        out[r, c] = out_colors[0]

    ring_total = _ring_count(shape, center, spacing)
    for k in range(1, ring_total + 1):
        color = out_colors[k % len(out_colors)]
        for r, c in _square_border_cells(shape, center, k * spacing):
            if (r, c) not in seed_cells:
                out[r, c] = color
    return out


def _painted_cells(
    shape: tuple[int, int], center: tuple[int, int], spacing: int, seed_cells: set[tuple[int, int]]
) -> set[tuple[int, int]]:
    painted = set(seed_cells)
    for k in range(1, _ring_count(shape, center, spacing) + 1):
        for cell in _square_border_cells(shape, center, k * spacing):
            if cell not in seed_cells:
                painted.add(cell)
    return painted


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor concentric square layers from diagonal gray seeds using a separate control strip.

    Input structure:
    - The grid has a black background with exactly three gray seed pixels on an equally spaced 45-degree diagonal.
    - A separate horizontal or vertical 1-cell-thick color strip is the only other foreground object.
    - The middle gray seed gives the square center, while the strip order gives the layer colors.

    Rule:
    - Read the middle seed as the center and the diagonal seed spacing as the ring step size.
    - Treat the three seed pixels as layer 0 and recolor them with the first strip color.
    - For each k >= 1, draw the axis-aligned square border at Chebyshev radius k * spacing around the center, color it with the k-th strip color, cycle through the strip if needed, clip at the grid boundary, and keep earlier layers unchanged where they intersect a later border.
    - Remove the input color strip from the output.

    Key insight:
    - The geometry and the color program come from two disjoint structures: the diagonal gray seeds encode the layer spacing, while the strip indexes colors by layer number rather than by cell position.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    center, spacing, seed_cells = _find_seed_pattern(g)
    _strip_cells, strip_colors = _find_color_strip(g)
    return _render_output_from_spec(g.shape, center, spacing, seed_cells, strip_colors)


def _center_candidates(
    shape: tuple[int, int], spacing: int, clipped: bool
) -> list[tuple[int, int, int]]:
    height, width = shape
    candidates: list[tuple[int, int, int]] = []
    for cr in range(spacing, height - spacing):
        for cc in range(spacing, width - spacing):
            ring_total = _ring_count(shape, (cr, cc), spacing)
            if not (MIN_RINGS <= ring_total <= MAX_RINGS):
                continue
            fully_inside = (
                cr - ring_total * spacing >= 0
                and cr + ring_total * spacing < height
                and cc - ring_total * spacing >= 0
                and cc + ring_total * spacing < width
            )
            if clipped == fully_inside:
                continue
            candidates.append((cr, cc, ring_total))
    return candidates


def _bar_candidates(shape: tuple[int, int], length: int) -> list[list[tuple[int, int]]]:
    height, width = shape
    if length > max(height, width):
        return []

    candidates: list[list[tuple[int, int]]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    row_offsets = list(range(min(3, height)))
    col_offsets = list(range(min(3, width)))

    def add(coords: list[tuple[int, int]]) -> None:
        key = tuple(coords)
        if key not in seen:
            seen.add(key)
            candidates.append(coords)

    if length <= width:
        for row in row_offsets:
            add([(row, c) for c in range(length)])
            add([(row, width - length + c) for c in range(length)])
        for row in range(max(0, height - 3), height):
            add([(row, c) for c in range(length)])
            add([(row, width - length + c) for c in range(length)])

    if length <= height:
        for col in col_offsets:
            add([(r, col) for r in range(length)])
            add([(height - length + r, col) for r in range(length)])
        for col in range(max(0, width - 3), width):
            add([(r, col) for r in range(length)])
            add([(height - length + r, col) for r in range(length)])

    return candidates


def _is_visually_separate(
    bar_cells: list[tuple[int, int]], painted: set[tuple[int, int]], shape: tuple[int, int]
) -> bool:
    rows = [r for r, _ in bar_cells]
    cols = [c for _, c in bar_cells]
    r0 = max(0, min(rows) - 1)
    r1 = min(shape[0] - 1, max(rows) + 1)
    c0 = max(0, min(cols) - 1)
    c1 = min(shape[1] - 1, max(cols) + 1)
    for r, c in painted:
        if r0 <= r <= r1 and c0 <= c <= c1:
            return False
    return True


def _build_input_from_spec(
    shape: tuple[int, int],
    center: tuple[int, int],
    spacing: int,
    diag_slope: int,
    strip_cells: list[tuple[int, int]],
    strip_colors: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    cr, cc = center
    seed_cells = {
        (cr, cc),
        (cr - spacing, cc - diag_slope * spacing),
        (cr + spacing, cc + diag_slope * spacing),
    }

    inp = np.full(shape, BACKGROUND, dtype=int)
    for r, c in seed_cells:
        inp[r, c] = SEED_COLOR
    for (r, c), color in zip(strip_cells, strip_colors):
        inp[r, c] = int(color)

    expected = _render_output_from_spec(shape, center, spacing, seed_cells, strip_colors)
    return inp, expected


def _generate_case() -> tuple[np.ndarray, np.ndarray] | None:
    height = random.randint(GRID_MIN, GRID_MAX)
    width = random.randint(GRID_MIN, GRID_MAX)
    shape = (height, width)
    spacing = random.randint(2, 5)
    clipped = random.choice([False, True])
    diag_slope = random.choice([-1, 1])

    centers = _center_candidates(shape, spacing, clipped)
    if not centers:
        return None

    center_r, center_c, ring_total = random.choice(centers)
    center = (center_r, center_c)
    layer_total = ring_total + 1
    max_strip_len = min(len(STRIP_PALETTE), layer_total + 2)
    length_choices = list(range(layer_total, max_strip_len + 1))
    random.shuffle(length_choices)

    seed_cells = {
        center,
        (center_r - spacing, center_c - diag_slope * spacing),
        (center_r + spacing, center_c + diag_slope * spacing),
    }
    painted = _painted_cells(shape, center, spacing, seed_cells)

    for strip_len in length_choices:
        strip_positions = _bar_candidates(shape, strip_len)
        random.shuffle(strip_positions)
        for strip_cells in strip_positions:
            if any(cell in painted for cell in strip_cells):
                continue
            if not _is_visually_separate(strip_cells, painted, shape):
                continue
            strip_colors = random.sample(STRIP_PALETTE, strip_len)
            inp, expected = _build_input_from_spec(
                shape, center, spacing, diag_slope, strip_cells, strip_colors
            )
            try:
                solved = main(inp)
            except AssertionError:
                return None
            if not np.array_equal(solved, expected):
                return None
            return inp, expected

    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        case = _generate_case()
        if case is not None:
            return case[0]
    raise RuntimeError("failed to generate a valid concentric-layer control-strip input")
