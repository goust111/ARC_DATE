from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
RED = 2
YELLOW = 4
FILL_COLORS = (1, 3, 5, 6, 7, 8, 9)

MIN_ROWS = 2
MAX_ROWS = 5
MIN_COLS = 2
MAX_COLS = 5
CELL_SIZE_RANGE = (2, 5)
GRID_SIZE_RANGE = (9, 25)
GEN_ATTEMPTS = 300


def _is_boundary_value(value: int) -> bool:
    return int(value) in (BLACK, RED)


def _boundary_rows(g: np.ndarray) -> list[int]:
    rows = []
    for r in range(g.shape[0]):
        if all(_is_boundary_value(v) for v in g[r]):
            rows.append(int(r))
    return rows


def _boundary_cols(g: np.ndarray) -> list[int]:
    cols = []
    for c in range(g.shape[1]):
        if all(_is_boundary_value(v) for v in g[:, c]):
            cols.append(int(c))
    return cols


def _bands(lines: list[int], limit: int) -> list[tuple[int, int]]:
    assert len(lines) >= 2, "expected outer border lines"
    assert lines[0] == 0 and lines[-1] == limit - 1, "grid must be enclosed by boundary lines"
    out: list[tuple[int, int]] = []
    for a, b in zip(lines, lines[1:]):
        assert b - a >= 2, "grid lines must have width 1 and leave room for a region"
        out.append((int(a + 1), int(b - 1)))
    return out


def _region_fill(rect: np.ndarray) -> tuple[int, bool]:
    yellow_count = int(np.count_nonzero(rect == YELLOW))
    colors = {int(v) for v in np.unique(rect)}
    colors.discard(YELLOW)
    assert BLACK not in colors and RED not in colors, "region interior must not contain boundary colors"
    assert len(colors) == 1, "each region interior must have one fill color"
    return next(iter(colors)), yellow_count > 0


def _parse_regions(
    g: np.ndarray,
) -> tuple[
    list[tuple[int, int]],
    list[tuple[int, int]],
    dict[tuple[int, int], int],
    tuple[int, int],
]:
    row_lines = _boundary_rows(g)
    col_lines = _boundary_cols(g)
    row_bands = _bands(row_lines, g.shape[0])
    col_bands = _bands(col_lines, g.shape[1])

    fills: dict[tuple[int, int], int] = {}
    source_region: tuple[int, int] | None = None
    yellow_total = 0
    for rr, (r0, r1) in enumerate(row_bands):
        for cc, (c0, c1) in enumerate(col_bands):
            fill, has_yellow = _region_fill(g[r0 : r1 + 1, c0 : c1 + 1])
            fills[(rr, cc)] = int(fill)
            if has_yellow:
                yellow_total += int(np.count_nonzero(g[r0 : r1 + 1, c0 : c1 + 1] == YELLOW))
                assert source_region is None, "yellow marker must be unique"
                source_region = (int(rr), int(cc))

    assert yellow_total == 1, "expected exactly one yellow marker pixel"
    assert source_region is not None, "yellow marker must lie inside a region"
    return row_bands, col_bands, fills, source_region


def _doors(
    g: np.ndarray,
    row_bands: list[tuple[int, int]],
    col_bands: list[tuple[int, int]],
) -> dict[frozenset[tuple[int, int]], list[tuple[int, int]]]:
    row_lines = [r0 - 1 for r0, _r1 in row_bands] + [row_bands[-1][1] + 1]
    col_lines = [c0 - 1 for c0, _c1 in col_bands] + [col_bands[-1][1] + 1]
    edge_doors: dict[frozenset[tuple[int, int]], list[tuple[int, int]]] = {}

    for rr, (r0, r1) in enumerate(row_bands):
        for cc in range(len(col_bands) - 1):
            boundary_col = int(col_lines[cc + 1])
            segment = g[r0 : r1 + 1, boundary_col]
            assert set(map(int, np.unique(segment))) <= {BLACK, RED}, "vertical boundary must be black/red only"
            coords = [(int(r0 + k), boundary_col) for k in np.flatnonzero(segment == RED)]
            if coords:
                edge = frozenset({(int(rr), int(cc)), (int(rr), int(cc + 1))})
                edge_doors[edge] = coords

    for rr in range(len(row_bands) - 1):
        boundary_row = int(row_lines[rr + 1])
        for cc, (c0, c1) in enumerate(col_bands):
            segment = g[boundary_row, c0 : c1 + 1]
            assert set(map(int, np.unique(segment))) <= {BLACK, RED}, "horizontal boundary must be black/red only"
            coords = [(boundary_row, int(c0 + k)) for k in np.flatnonzero(segment == RED)]
            if coords:
                edge = frozenset({(int(rr), int(cc)), (int(rr + 1), int(cc))})
                edge_doors[edge] = coords

    return edge_doors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor all regions reachable from the yellow-marked source through red one-pixel doors.

    The black boundary rows/columns define a rectangular region grid. A red pixel on a shared
    boundary segment is a door connecting the two adjacent regions. Starting from the unique
    region containing the yellow marker, propagate through doors and repaint every reachable
    region to the source region's original fill color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    row_bands, col_bands, fills, source = _parse_regions(g)
    doors = _doors(g, row_bands, col_bands)
    source_color = int(fills[source])

    seen = {source}
    queue: deque[tuple[int, int]] = deque([source])
    while queue:
        region = queue.popleft()
        rr, cc = region
        neighbors = []
        if rr > 0:
            neighbors.append((rr - 1, cc))
        if rr + 1 < len(row_bands):
            neighbors.append((rr + 1, cc))
        if cc > 0:
            neighbors.append((rr, cc - 1))
        if cc + 1 < len(col_bands):
            neighbors.append((rr, cc + 1))
        for nxt in neighbors:
            edge = frozenset({region, nxt})
            if edge in doors and nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)

    out = g.copy()
    out[out == RED] = BLACK
    out[out == YELLOW] = source_color

    for rr, cc in seen:
        r0, r1 = row_bands[rr]
        c0, c1 = col_bands[cc]
        out[r0 : r1 + 1, c0 : c1 + 1] = source_color

    for edge, coords in doors.items():
        a, b = tuple(edge)
        if a in seen and b in seen:
            for r, c in coords:
                out[r, c] = source_color

    return out


def _region_neighbors(region: tuple[int, int], rows: int, cols: int) -> list[tuple[int, int]]:
    rr, cc = region
    out = []
    if rr > 0:
        out.append((rr - 1, cc))
    if rr + 1 < rows:
        out.append((rr + 1, cc))
    if cc > 0:
        out.append((rr, cc - 1))
    if cc + 1 < cols:
        out.append((rr, cc + 1))
    return out


def _grid_edges(rows: int, cols: int) -> list[frozenset[tuple[int, int]]]:
    edges: list[frozenset[tuple[int, int]]] = []
    for rr in range(rows):
        for cc in range(cols):
            if rr + 1 < rows:
                edges.append(frozenset({(rr, cc), (rr + 1, cc)}))
            if cc + 1 < cols:
                edges.append(frozenset({(rr, cc), (rr, cc + 1)}))
    return edges


def _choose_connected_subset(
    rows: int,
    cols: int,
    size: int,
) -> tuple[set[tuple[int, int]], set[frozenset[tuple[int, int]]]]:
    start = (random.randrange(rows), random.randrange(cols))
    subset = {start}
    tree_edges: set[frozenset[tuple[int, int]]] = set()
    frontier: list[tuple[tuple[int, int], tuple[int, int]]] = [
        (start, nxt) for nxt in _region_neighbors(start, rows, cols)
    ]

    while len(subset) < size:
        choices = [(src, dst) for src, dst in frontier if dst not in subset]
        assert choices, "connected subset growth got stuck"
        src, dst = random.choice(choices)
        subset.add(dst)
        tree_edges.add(frozenset({src, dst}))
        frontier.extend((dst, nxt) for nxt in _region_neighbors(dst, rows, cols))

    return subset, tree_edges


def _pick_layout() -> tuple[int, int, list[int], list[int]] | None:
    rows = random.randint(MIN_ROWS, MAX_ROWS)
    cols = random.randint(MIN_COLS, MAX_COLS)
    heights = [random.randint(*CELL_SIZE_RANGE) for _ in range(rows)]
    widths = [random.randint(*CELL_SIZE_RANGE) for _ in range(cols)]
    grid_h = sum(heights) + rows + 1
    grid_w = sum(widths) + cols + 1
    if not (GRID_SIZE_RANGE[0] <= grid_h <= GRID_SIZE_RANGE[1]):
        return None
    if not (GRID_SIZE_RANGE[0] <= grid_w <= GRID_SIZE_RANGE[1]):
        return None
    return rows, cols, heights, widths


def _line_positions(sizes: list[int]) -> list[int]:
    pos = 0
    lines = [0]
    for size in sizes:
        pos += int(size) + 1
        lines.append(pos)
    return lines


def _fill_assignments(count: int) -> list[int]:
    palette_size = random.randint(2, min(6, len(FILL_COLORS)))
    palette = random.sample(FILL_COLORS, palette_size)
    colors = [random.choice(palette) for _ in range(count)]
    if len(set(colors)) == 1:
        alt = random.choice([c for c in FILL_COLORS if c != colors[0]])
        colors[random.randrange(count)] = alt
    return colors


def _subset_mode(region_count: int) -> str:
    assert region_count >= 4
    return random.choice(["single", "partial", "full"])


def _choose_doors(
    rows: int,
    cols: int,
    subset: set[tuple[int, int]],
    tree_edges: set[frozenset[tuple[int, int]]],
    mode: str,
) -> set[frozenset[tuple[int, int]]]:
    all_edges = _grid_edges(rows, cols)
    subset_edges = {edge for edge in all_edges if all(node in subset for node in edge)}
    outside = {(rr, cc) for rr in range(rows) for cc in range(cols)} - subset
    outside_edges = {edge for edge in all_edges if all(node in outside for node in edge)}

    doors = set(tree_edges)
    for edge in subset_edges - tree_edges:
        if random.random() < 0.35:
            doors.add(edge)

    outside_selected = {edge for edge in outside_edges if random.random() < 0.35}
    if mode != "full" and outside_edges and not outside_selected:
        outside_selected.add(random.choice(list(outside_edges)))
    doors |= outside_selected

    if not doors:
        doors.add(random.choice(all_edges))
    return doors


def _paint_layout(
    rows: int,
    cols: int,
    heights: list[int],
    widths: list[int],
    fills: list[int],
    source: tuple[int, int],
    door_edges: set[frozenset[tuple[int, int]]],
) -> np.ndarray:
    row_lines = _line_positions(heights)
    col_lines = _line_positions(widths)
    grid = np.full((row_lines[-1] + 1, col_lines[-1] + 1), BLACK, dtype=int)

    row_bands = [(row_lines[i] + 1, row_lines[i + 1] - 1) for i in range(rows)]
    col_bands = [(col_lines[i] + 1, col_lines[i + 1] - 1) for i in range(cols)]

    for rr in range(rows):
        for cc in range(cols):
            fill = fills[rr * cols + cc]
            r0, r1 = row_bands[rr]
            c0, c1 = col_bands[cc]
            grid[r0 : r1 + 1, c0 : c1 + 1] = int(fill)

    for edge in door_edges:
        a, b = sorted(edge)
        (r1, c1), (r2, c2) = a, b
        if r1 == r2:
            rr = r1
            boundary_col = col_lines[min(c1, c2) + 1]
            r0, r1_band = row_bands[rr]
            door_row = random.randint(r0, r1_band)
            grid[door_row, boundary_col] = RED
        else:
            cc = c1
            boundary_row = row_lines[min(r1, r2) + 1]
            c0, c1_band = col_bands[cc]
            door_col = random.randint(c0, c1_band)
            grid[boundary_row, door_col] = RED

    sr, sc = source
    r0, r1 = row_bands[sr]
    c0, c1 = col_bands[sc]
    marker_r = random.randint(r0, r1)
    marker_c = random.randint(c0, c1)
    grid[marker_r, marker_c] = YELLOW
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        layout = _pick_layout()
        if layout is None:
            continue
        rows, cols, heights, widths = layout
        region_count = rows * cols
        mode = _subset_mode(region_count)

        if mode == "single":
            subset_size = 1
        elif mode == "full":
            subset_size = region_count
        else:
            subset_size = random.randint(2, region_count - 2)

        subset, tree_edges = _choose_connected_subset(rows, cols, subset_size)
        source = random.choice(list(subset))
        door_edges = _choose_doors(rows, cols, subset, tree_edges, mode)
        fills = _fill_assignments(region_count)
        grid = _paint_layout(rows, cols, heights, widths, fills, source, door_edges)

        try:
            out = main(grid)
        except AssertionError:
            continue

        if out.shape != grid.shape:
            continue
        if np.array_equal(out, grid):
            continue
        return grid

    raise RuntimeError("failed to generate a valid puzzle instance")
