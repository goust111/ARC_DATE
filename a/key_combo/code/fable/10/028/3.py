from __future__ import annotations

from collections import Counter
import random

import numpy as np


CORNERS = ("tl", "tr", "bl", "br")
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_MAX_ATTEMPTS = 500
GEN_COLOR_POOL = tuple(range(1, 10))
_GEN_CALL_COUNT = 0


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _legend_origin(shape: tuple[int, int], corner: str) -> tuple[int, int]:
    h, w = map(int, shape)
    r0 = 0 if corner[0] == "t" else h - 2
    c0 = 0 if corner[1] == "l" else w - 2
    return r0, c0


def _corner_index(corner: str) -> tuple[int, int]:
    return (0 if corner[0] == "t" else 1, 0 if corner[1] == "l" else 1)


def _corner_patch(grid: np.ndarray, corner: str) -> np.ndarray:
    r0, c0 = _legend_origin(grid.shape, corner)
    return grid[r0 : r0 + 2, c0 : c0 + 2]


def _legend_mask(shape: tuple[int, int], corner: str) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    r0, c0 = _legend_origin(shape, corner)
    mask[r0 : r0 + 2, c0 : c0 + 2] = True
    return mask


def _swap_map_from_legend(legend: np.ndarray) -> dict[int, int]:
    assert legend.shape == (2, 2)
    row0 = [int(legend[0, 0]), int(legend[0, 1])]
    row1 = [int(legend[1, 0]), int(legend[1, 1])]
    assert len(set(row0 + row1)) == 4, "legend must contain four distinct colors"
    assert row0[0] != row0[1] and row1[0] != row1[1], "each legend row must be a color pair"
    return {
        row0[0]: row0[1],
        row0[1]: row0[0],
        row1[0]: row1[1],
        row1[1]: row1[0],
    }


def _swap_non_bg(grid: np.ndarray, swap_map: dict[int, int], bg: int) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for src, dst in swap_map.items():
        out[(grid == int(src)) & (grid != int(bg))] = int(dst)
    return out


def _transform_for_quadrant(
    grid: np.ndarray,
    corner: str,
    quadrant_row: int,
    quadrant_col: int,
    swap_map: dict[int, int],
    bg: int,
) -> np.ndarray:
    base_row, base_col = _corner_index(corner)
    out = np.asarray(grid, dtype=int)
    if quadrant_row != base_row:
        out = np.flipud(out)
    if quadrant_col != base_col:
        out = np.fliplr(out)
    if (quadrant_row != base_row) + (quadrant_col != base_col) == 1:
        out = _swap_non_bg(out, swap_map, bg)
    return np.array(out, dtype=int, copy=True)


def _expected_output_from_corner(grid: np.ndarray, corner: str) -> np.ndarray:
    bg = _get_bg(grid)
    legend = _corner_patch(grid, corner)
    swap_map = _swap_map_from_legend(legend)
    quads = [
        [
            _transform_for_quadrant(grid, corner, 0, 0, swap_map, bg),
            _transform_for_quadrant(grid, corner, 0, 1, swap_map, bg),
        ],
        [
            _transform_for_quadrant(grid, corner, 1, 0, swap_map, bg),
            _transform_for_quadrant(grid, corner, 1, 1, swap_map, bg),
        ],
    ]
    return np.block(quads)


def _find_legend_corner(grid: np.ndarray, bg: int) -> str:
    non_bg_colors = {int(v) for v in np.unique(grid) if int(v) != int(bg)}
    candidates: list[str] = []
    for corner in CORNERS:
        legend = _corner_patch(grid, corner)
        colors = [int(v) for v in legend.ravel()]
        if int(bg) in colors:
            continue
        if len(set(colors)) != 4:
            continue
        if legend[0, 0] == legend[0, 1] or legend[1, 0] == legend[1, 1]:
            continue
        if not non_bg_colors <= set(colors):
            continue
        candidates.append(corner)
    assert len(candidates) == 1, "expected exactly one corner legend"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand the input into a 2x2 mirrored layout, keeping the original in the corner named by the legend.

    Input structure:
    - The grid has a dominant background and one solid 2x2 legend block flush with exactly one corner.
    - Each legend row contains a pair of different colors, and every other non-background pixel uses one of those four colors.

    Rule:
    - Double the height and width and fill the four quadrants with the input and its horizontal and vertical reflections.
    - Place the unchanged input in the output quadrant that matches the legend corner.
    - In the two quadrants reached by exactly one reflection, swap every non-background color with the other color from the same legend row.
    - In the diagonally opposite quadrant, which uses two reflections, keep the original colors.

    Key insight:
    - The corner legend is both the position anchor and the color key: odd-parity reflections flip handedness and trigger the row-wise swap, while even-parity reflections do not.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 2 and g.shape[1] >= 2, "expected a 2D grid"

    bg = _get_bg(g)
    corner = _find_legend_corner(g, bg)
    legend = _corner_patch(g, corner)
    legend_colors = {int(v) for v in legend.ravel()}
    assert bg not in legend_colors, "legend must be foreground"
    assert len(legend_colors) == 4, "legend must expose four colors"

    non_bg = (g != bg)
    assert np.any(non_bg), "expected foreground"
    assert {int(v) for v in g[non_bg]} <= legend_colors, "foreground must use legend colors only"

    return _expected_output_from_corner(g, corner)


def _other_corner_block_cells(shape: tuple[int, int], legend_corner: str) -> set[tuple[int, int]]:
    forbidden: set[tuple[int, int]] = set()
    for corner in CORNERS:
        if corner == legend_corner:
            continue
        r0, c0 = _legend_origin(shape, corner)
        for r in range(r0, r0 + 2):
            for c in range(c0, c0 + 2):
                forbidden.add((r, c))
    return forbidden


def _legend_neighbor_cells(shape: tuple[int, int], corner: str) -> list[tuple[int, int]]:
    h, w = shape
    r0, c0 = _legend_origin(shape, corner)
    cells: list[tuple[int, int]] = []

    neighbor_row = 2 if corner[0] == "t" else h - 3
    neighbor_col = 2 if corner[1] == "l" else w - 3

    if 0 <= neighbor_row < h:
        cells.extend([(neighbor_row, c0), (neighbor_row, c0 + 1)])
    if 0 <= neighbor_col < w:
        cells.extend([(r0, neighbor_col), (r0 + 1, neighbor_col)])

    unique: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for cell in cells:
        if cell not in seen:
            seen.add(cell)
            unique.append(cell)
    return unique


def _bbox_of_cells(cells: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _allowed_component_cells(shape: tuple[int, int], corner: str) -> set[tuple[int, int]]:
    h, w = shape
    allowed = {(r, c) for r in range(h) for c in range(w)}
    legend_cells = {
        (int(r), int(c))
        for r, c in np.argwhere(_legend_mask(shape, corner))
    }
    allowed -= legend_cells
    allowed -= _other_corner_block_cells(shape, corner)
    return {(int(r), int(c)) for r, c in allowed}


def _pick_growth_cell(
    frontier: list[tuple[int, int]],
    cells: set[tuple[int, int]],
    anchor: tuple[int, int],
) -> tuple[int, int]:
    r0, c0, r1, c1 = _bbox_of_cells(cells)
    expanding = [
        cell
        for cell in frontier
        if cell[0] < r0 or cell[0] > r1 or cell[1] < c0 or cell[1] > c1
    ]
    pool = expanding if expanding and random.random() < 0.7 else frontier
    weights = [
        1 + abs(cell[0] - anchor[0]) + abs(cell[1] - anchor[1])
        for cell in pool
    ]
    return random.choices(pool, weights=weights, k=1)[0]


def _grow_component(shape: tuple[int, int], corner: str, target_area: int) -> set[tuple[int, int]] | None:
    allowed = _allowed_component_cells(shape, corner)
    seeds = [cell for cell in _legend_neighbor_cells(shape, corner) if cell in allowed]
    if not seeds:
        return None

    cells = {random.choice(seeds)}
    anchor = _legend_origin(shape, corner)

    while len(cells) < target_area:
        frontier_set: set[tuple[int, int]] = set()
        for r, c in cells:
            for dr, dc in NEIGHBORS4:
                nr, nc = r + dr, c + dc
                cell = (nr, nc)
                if cell in allowed and cell not in cells:
                    frontier_set.add(cell)
        if not frontier_set:
            return None
        chosen = _pick_growth_cell(sorted(frontier_set), cells, anchor)
        cells.add(chosen)

    return cells


def _assign_component_colors(cells: set[tuple[int, int]], legend: np.ndarray) -> dict[tuple[int, int], int] | None:
    coords = list(cells)
    if len(coords) < 4:
        return None
    random.shuffle(coords)
    colors = [int(v) for v in legend.ravel()]
    values = colors[:]
    for _ in range(len(coords) - len(colors)):
        values.append(random.choice(colors))
    random.shuffle(values)
    return {coord: int(color) for coord, color in zip(coords, values)}


def _build_input_grid(
    shape: tuple[int, int],
    corner: str,
    legend: np.ndarray,
    comp_colors: dict[tuple[int, int], int],
) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    r0, c0 = _legend_origin(shape, corner)
    grid[r0 : r0 + 2, c0 : c0 + 2] = legend
    for (r, c), color in comp_colors.items():
        grid[r, c] = int(color)
    return grid


def _quadrant_slice(shape: tuple[int, int], quadrant_row: int, quadrant_col: int) -> tuple[slice, slice]:
    h, w = shape
    rs = slice(quadrant_row * h, (quadrant_row + 1) * h)
    cs = slice(quadrant_col * w, (quadrant_col + 1) * w)
    return rs, cs


def _validate_generated_input(grid: np.ndarray) -> str:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert 7 <= h <= 15 and 7 <= w <= 15, "generator must keep the doubled output within 30x30"

    bg = _get_bg(g)
    assert int(np.count_nonzero(g == bg)) > (h * w) // 2, "background must stay dominant"

    corner = _find_legend_corner(g, bg)
    legend = _corner_patch(g, corner)
    legend_colors = {int(v) for v in legend.ravel()}
    assert len(legend_colors) == 4

    legend_mask = _legend_mask(g.shape, corner)
    outside_mask = (g != bg) & (~legend_mask)
    assert np.any(outside_mask), "need non-legend foreground"
    assert {int(v) for v in g[outside_mask]} == legend_colors, "all legend colors must appear outside the legend"

    for other in CORNERS:
        if other == corner:
            continue
        other_patch = _corner_patch(g, other)
        assert np.any(other_patch == bg), "other corners must not host a competing 2x2 legend"

    variants = []
    swap_map = _swap_map_from_legend(legend)
    for qr in range(2):
        for qc in range(2):
            variants.append(_transform_for_quadrant(g, corner, qr, qc, swap_map, bg))
    sigs = {
        (variant.shape, variant.tobytes())
        for variant in variants
    }
    assert len(sigs) == 4, "all reflected/recolored quadrants must be visibly different"

    out = _expected_output_from_corner(g, corner)
    for qr in range(2):
        for qc in range(2):
            rs, cs = _quadrant_slice(g.shape, qr, qc)
            quad = out[rs, cs]
            expected = _transform_for_quadrant(g, corner, qr, qc, swap_map, bg)
            assert np.array_equal(quad, expected)

    return corner


def _sample_shape(aspect_mode: int) -> tuple[int, int]:
    if aspect_mode == 0:
        return random.randint(8, 15), random.randint(8, 15)
    if aspect_mode == 1:
        return random.randint(10, 15), random.randint(7, 12)
    return random.randint(7, 12), random.randint(10, 15)


def _gen_attempt(corner: str, aspect_mode: int) -> np.ndarray | None:
    h, w = _sample_shape(aspect_mode)
    colors = random.sample(GEN_COLOR_POOL, 4)
    legend = np.array(
        [
            [colors[0], colors[1]],
            [colors[2], colors[3]],
        ],
        dtype=int,
    )

    max_component = min(24, (h * w) // 3)
    if max_component < 8:
        return None
    target_area = random.randint(8, max_component)
    cells = _grow_component((h, w), corner, target_area)
    if cells is None:
        return None

    r0, c0, r1, c1 = _bbox_of_cells(cells)
    if r1 - r0 + 1 < 4 or c1 - c0 + 1 < 4:
        return None

    comp_colors = _assign_component_colors(cells, legend)
    if comp_colors is None:
        return None

    grid = _build_input_grid((h, w), corner, legend, comp_colors)

    try:
        found_corner = _validate_generated_input(grid)
    except AssertionError:
        return None
    if found_corner != corner:
        return None

    expected = _expected_output_from_corner(grid, corner)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT

    call_id = _GEN_CALL_COUNT
    _GEN_CALL_COUNT += 1
    corner = CORNERS[call_id % len(CORNERS)]
    aspect_mode = call_id % 3

    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt(corner, aspect_mode)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid legend-corner mirror input")
