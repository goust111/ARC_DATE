from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np


BLACK = 0
GREY = 5
FILL_COLORS = tuple(color for color in range(10) if color not in (BLACK, GREY))

DIR_VECTORS: dict[str, tuple[int, int]] = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
PERPENDICULARS: dict[str, tuple[tuple[int, int], tuple[int, int]]] = {
    "up": ((0, -1), (0, 1)),
    "down": ((0, -1), (0, 1)),
    "left": ((-1, 0), (1, 0)),
    "right": ((-1, 0), (1, 0)),
}

MIN_GRID_PARTS = 2
MAX_GRID_PARTS = 4
MIN_CELL_SIZE = 5
MAX_CELL_SIZE_BY_COUNT = {2: 8, 3: 7, 4: 6}
MAX_CANVAS_SIZE = 30
MAX_MARGIN = 2
MAX_GENERATE_ATTEMPTS = 400


@dataclass(frozen=True)
class Region:
    component_id: int
    coords: tuple[tuple[int, int], ...]
    fill_color: int
    direction: str
    tip: tuple[int, int]


def _components(mask: np.ndarray) -> tuple[list[list[tuple[int, int]]], np.ndarray]:
    h, w = mask.shape
    labels = np.full((h, w), -1, dtype=int)
    components: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != -1:
                continue

            comp_id = len(components)
            stack = [(r, c)]
            labels[r, c] = comp_id
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and labels[nr, nc] == -1:
                        labels[nr, nc] = comp_id
                        stack.append((nr, nc))

            components.append(coords)

    return components, labels


def _detect_arrow(black_cells: list[tuple[int, int]]) -> tuple[str, tuple[int, int]]:
    cell_set = set(black_cells)
    assert len(cell_set) == 4, "each region must contain a 4-pixel arrow"

    for direction, (dr, dc) in DIR_VECTORS.items():
        for tip in cell_set:
            tail_1 = (tip[0] - dr, tip[1] - dc)
            tail_2 = (tip[0] - 2 * dr, tip[1] - 2 * dc)
            if tail_1 not in cell_set or tail_2 not in cell_set:
                continue
            for pr, pc in PERPENDICULARS[direction]:
                extra = (tip[0] + pr, tip[1] + pc)
                if {tip, tail_1, tail_2, extra} == cell_set:
                    return direction, tip

    raise AssertionError("failed to recognize arrow direction")


def _extract_regions(grid: np.ndarray) -> tuple[list[Region], np.ndarray, dict[int, int]]:
    h, w = grid.shape
    components, labels = _components(grid != GREY)
    regions: list[Region] = []
    component_to_region: dict[int, int] = {}

    for component_id, coords in enumerate(components):
        touches_border = any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in coords)
        if touches_border:
            continue

        fill_colors = {
            int(grid[r, c])
            for r, c in coords
            if int(grid[r, c]) not in (BLACK, GREY)
        }
        assert len(fill_colors) == 1, "each region must have a single non-black fill color"
        fill_color = next(iter(fill_colors))

        black_cells = [(r, c) for r, c in coords if int(grid[r, c]) == BLACK]
        direction, tip = _detect_arrow(black_cells)

        component_to_region[component_id] = len(regions)
        regions.append(
            Region(
                component_id=component_id,
                coords=tuple(coords),
                fill_color=fill_color,
                direction=direction,
                tip=tip,
            )
        )

    assert regions, "expected at least one enclosed region"
    return regions, labels, component_to_region


def _target_region_index(
    grid: np.ndarray,
    labels: np.ndarray,
    component_to_region: dict[int, int],
    region: Region,
) -> int:
    h, w = grid.shape
    dr, dc = DIR_VECTORS[region.direction]
    r, c = region.tip

    while True:
        nr, nc = r + dr, c + dc
        assert 0 <= nr < h and 0 <= nc < w, "arrow path must hit a grey divider before leaving the grid"
        if int(grid[nr, nc]) == GREY:
            over_r, over_c = nr + dr, nc + dc
            break
        r, c = nr, nc

    assert 0 <= over_r < h and 0 <= over_c < w, "arrow must point to another region"
    assert int(grid[over_r, over_c]) != GREY, "neighbouring region must start immediately after the divider"

    component_id = int(labels[over_r, over_c])
    assert component_id in component_to_region, "arrow must cross into another enclosed region"
    target_index = component_to_region[component_id]
    assert target_index != component_to_region[region.component_id], "arrow may not target its own region"
    return target_index


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy each region's neighbour color according to the black arrow inside the region.
    Grey divider lines stay fixed and every region reads neighbour colors from the
    original input, not from already-updated regions.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    regions, labels, component_to_region = _extract_regions(g)
    new_colors = [
        regions[_target_region_index(g, labels, component_to_region, region)].fill_color
        for region in regions
    ]

    out = g.copy()
    for region, color in zip(regions, new_colors):
        for r, c in region.coords:
            out[r, c] = color
    return out


def _choose_sizes(count: int) -> list[int]:
    max_size = MAX_CELL_SIZE_BY_COUNT[count]
    return [random.randint(MIN_CELL_SIZE, max_size) for _ in range(count)]


def _line_positions(offset: int, sizes: list[int]) -> list[int]:
    positions = [offset]
    cursor = offset
    for size in sizes:
        cursor += size + 1
        positions.append(cursor)
    return positions


def _cell_ranges(offset: int, sizes: list[int]) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    cursor = offset + 1
    for size in sizes:
        ranges.append((cursor, cursor + size - 1))
        cursor += size + 1
    return ranges


def _valid_directions(row: int, col: int, n_rows: int, n_cols: int) -> list[str]:
    directions: list[str] = []
    if row > 0:
        directions.append("up")
    if row + 1 < n_rows:
        directions.append("down")
    if col > 0:
        directions.append("left")
    if col + 1 < n_cols:
        directions.append("right")
    return directions


def _assign_directions(n_rows: int, n_cols: int) -> dict[tuple[int, int], str]:
    directions: dict[tuple[int, int], str] = {
        (0, 0): "right",
        (0, n_cols - 1): "down",
        (n_rows - 1, 0): "up",
        (n_rows - 1, n_cols - 1): "left",
    }
    counts = {name: 1 for name in DIR_VECTORS}

    remaining = [
        (row, col)
        for row in range(n_rows)
        for col in range(n_cols)
        if (row, col) not in directions
    ]
    random.shuffle(remaining)

    for row, col in remaining:
        valid = _valid_directions(row, col, n_rows, n_cols)
        best = min(counts[name] for name in valid)
        candidates = [name for name in valid if counts[name] == best]
        direction = random.choice(candidates)
        directions[(row, col)] = direction
        counts[direction] += 1

    return directions


def _choose_colors() -> list[int]:
    return random.sample(list(FILL_COLORS), 4)


def _cell_color(row: int, col: int, palette: list[int]) -> int:
    return palette[(row % 2) * 2 + (col % 2)]


def _arrow_pixels(
    tip: tuple[int, int], direction: str, extra_offset: tuple[int, int]
) -> tuple[tuple[int, int], ...]:
    dr, dc = DIR_VECTORS[direction]
    pr, pc = extra_offset
    return (
        tip,
        (tip[0] - dr, tip[1] - dc),
        (tip[0] - 2 * dr, tip[1] - 2 * dc),
        (tip[0] + pr, tip[1] + pc),
    )


def _place_arrow(
    grid: np.ndarray,
    row_bounds: tuple[int, int],
    col_bounds: tuple[int, int],
    direction: str,
) -> None:
    r0, r1 = row_bounds
    c0, c1 = col_bounds
    options: list[tuple[tuple[int, int], ...]] = []

    for tip_r in range(r0, r1 + 1):
        for tip_c in range(c0, c1 + 1):
            tip = (tip_r, tip_c)
            for extra in PERPENDICULARS[direction]:
                pixels = _arrow_pixels(tip, direction, extra)
                rows = [r for r, _ in pixels]
                cols = [c for _, c in pixels]
                if not all(r0 <= r <= r1 and c0 <= c <= c1 for r, c in pixels):
                    continue
                if min(rows) <= r0 or max(rows) >= r1 or min(cols) <= c0 or max(cols) >= c1:
                    continue
                options.append(pixels)

    assert options, "cell is too small for an inset arrow"
    for r, c in random.choice(options):
        grid[r, c] = BLACK


def _generate_attempt() -> np.ndarray | None:
    n_rows = random.randint(MIN_GRID_PARTS, MAX_GRID_PARTS)
    n_cols = random.randint(MIN_GRID_PARTS, MAX_GRID_PARTS)
    row_sizes = _choose_sizes(n_rows)
    col_sizes = _choose_sizes(n_cols)

    content_h = sum(row_sizes) + n_rows + 1
    content_w = sum(col_sizes) + n_cols + 1
    if content_h > MAX_CANVAS_SIZE or content_w > MAX_CANVAS_SIZE:
        return None

    remaining_h = MAX_CANVAS_SIZE - content_h
    remaining_w = MAX_CANVAS_SIZE - content_w
    top = random.randint(0, min(MAX_MARGIN, remaining_h))
    bottom = random.randint(0, min(MAX_MARGIN, remaining_h - top))
    left = random.randint(0, min(MAX_MARGIN, remaining_w))
    right = random.randint(0, min(MAX_MARGIN, remaining_w - left))
    if top + bottom + left + right == 0:
        return None

    grid = np.full((content_h + top + bottom, content_w + left + right), BLACK, dtype=int)

    row_lines = _line_positions(top, row_sizes)
    col_lines = _line_positions(left, col_sizes)
    for r in row_lines:
        grid[r, left : col_lines[-1] + 1] = GREY
    for c in col_lines:
        grid[top : row_lines[-1] + 1, c] = GREY

    row_ranges = _cell_ranges(top, row_sizes)
    col_ranges = _cell_ranges(left, col_sizes)
    directions = _assign_directions(n_rows, n_cols)
    palette = _choose_colors()

    for row in range(n_rows):
        rr0, rr1 = row_ranges[row]
        for col in range(n_cols):
            cc0, cc1 = col_ranges[col]
            grid[rr0 : rr1 + 1, cc0 : cc1 + 1] = _cell_color(row, col, palette)

    for row in range(n_rows):
        for col in range(n_cols):
            _place_arrow(grid, row_ranges[row], col_ranges[col], directions[(row, col)])

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATE_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
