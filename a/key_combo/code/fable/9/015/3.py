from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
TEAL = 8
MAROON = 9

MARKER_PALETTE = (BLUE, GREEN, YELLOW, GREY, ORANGE, MAROON)
MAX_GENERATION_ATTEMPTS = 800
TIE_PROBABILITY = 0.6
THREE_COLOR_PROBABILITY = 0.45
DUPLICATE_COLOR_PROBABILITY = 0.55


Cell = tuple[int, int]


def _layout_from_sizes(sizes: list[int]) -> tuple[list[tuple[int, int]], list[int], int]:
    bands: list[tuple[int, int]] = []
    lines: list[int] = []
    pos = 0
    for idx, size in enumerate(sizes):
        bands.append((pos, pos + size - 1))
        pos += size
        if idx < len(sizes) - 1:
            lines.append(pos)
            pos += 1
    return bands, lines, pos


def _band_index(pos: int, bands: list[tuple[int, int]]) -> int:
    for idx, (lo, hi) in enumerate(bands):
        if lo <= pos <= hi:
            return idx
    raise AssertionError("position must belong to exactly one cell band")


def _cell_ranges(length: int, line_indices: list[int]) -> list[tuple[int, int]]:
    assert all(0 < line < length - 1 for line in line_indices), "lattice lines must be internal"
    ranges = []
    start = 0
    for line in sorted(line_indices):
        assert start <= line - 1, "adjacent lattice lines would erase a cell band"
        ranges.append((start, line - 1))
        start = line + 1
    assert start <= length - 1, "missing trailing cell band"
    ranges.append((start, length - 1))
    return ranges


def _normalize_edge(a: Cell, b: Cell) -> tuple[Cell, Cell]:
    return (a, b) if a <= b else (b, a)


def _graph_neighbors(
    r: int,
    c: int,
    n_rows: int,
    n_cols: int,
    blocked_h: set[tuple[int, int]],
    blocked_v: set[tuple[int, int]],
) -> list[Cell]:
    out: list[Cell] = []
    if r > 0 and (r - 1, c) not in blocked_h:
        out.append((r - 1, c))
    if r + 1 < n_rows and (r, c) not in blocked_h:
        out.append((r + 1, c))
    if c > 0 and (r, c - 1) not in blocked_v:
        out.append((r, c - 1))
    if c + 1 < n_cols and (r, c) not in blocked_v:
        out.append((r, c + 1))
    return out


def _coarse_distance(
    n_rows: int,
    n_cols: int,
    blocked_h: set[tuple[int, int]],
    blocked_v: set[tuple[int, int]],
    sources: set[Cell],
) -> np.ndarray:
    inf = n_rows * n_cols + 10
    dist = np.full((n_rows, n_cols), inf, dtype=int)
    q: deque[Cell] = deque()
    for cell in sources:
        dist[cell] = 0
        q.append(cell)
    while q:
        r, c = q.popleft()
        step = dist[r, c] + 1
        for nr, nc in _graph_neighbors(r, c, n_rows, n_cols, blocked_h, blocked_v):
            if step < dist[nr, nc]:
                dist[nr, nc] = step
                q.append((nr, nc))
    return dist


def _solve_coarse(
    n_rows: int,
    n_cols: int,
    blocked_h: set[tuple[int, int]],
    blocked_v: set[tuple[int, int]],
    sources_by_color: dict[int, set[Cell]],
) -> tuple[np.ndarray, np.ndarray]:
    colors = sorted(sources_by_color)
    assert len(colors) >= 2, "expected at least two marker colors"
    dist_stack = np.stack(
        [
            _coarse_distance(n_rows, n_cols, blocked_h, blocked_v, sources_by_color[color])
            for color in colors
        ],
        axis=0,
    )
    best = dist_stack.min(axis=0)
    assert np.all(best < n_rows * n_cols + 10), "every cell must reach at least one marker"

    fill = np.full((n_rows, n_cols), -1, dtype=int)
    for r in range(n_rows):
        for c in range(n_cols):
            winners = [
                colors[idx]
                for idx in range(len(colors))
                if int(dist_stack[idx, r, c]) == int(best[r, c])
            ]
            fill[r, c] = PINK if len(winners) > 1 else winners[0]
    return fill, best


def _render_input(
    row_sizes: list[int],
    col_sizes: list[int],
    blocked_h: set[tuple[int, int]],
    blocked_v: set[tuple[int, int]],
    sources_by_color: dict[int, set[Cell]],
) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]], dict[Cell, tuple[int, int, int]]]:
    row_bands, row_lines, height = _layout_from_sizes(row_sizes)
    col_bands, col_lines, width = _layout_from_sizes(col_sizes)
    grid = np.full((height, width), BLACK, dtype=int)
    for row in row_lines:
        grid[row, :] = TEAL
    for col in col_lines:
        grid[:, col] = TEAL

    for r, c in blocked_h:
        wall_row = row_lines[r]
        c0, c1 = col_bands[c]
        grid[wall_row, c0 : c1 + 1] = RED
    for r, c in blocked_v:
        wall_col = col_lines[c]
        r0, r1 = row_bands[r]
        grid[r0 : r1 + 1, wall_col] = RED

    for row in row_lines:
        for col in col_lines:
            grid[row, col] = TEAL

    marker_pixels: dict[Cell, tuple[int, int, int]] = {}
    for color, cells in sources_by_color.items():
        for r, c in cells:
            r0, r1 = row_bands[r]
            c0, c1 = col_bands[c]
            mr = (r0 + r1) // 2
            mc = (c0 + c1) // 2
            grid[mr, mc] = color
            marker_pixels[(r, c)] = (mr, mc, color)
    return grid, row_bands, col_bands, marker_pixels


def _render_output(
    grid: np.ndarray,
    row_bands: list[tuple[int, int]],
    col_bands: list[tuple[int, int]],
    fill_map: np.ndarray,
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for r, (r0, r1) in enumerate(row_bands):
        for c, (c0, c1) in enumerate(col_bands):
            out[r0 : r1 + 1, c0 : c1 + 1] = int(fill_map[r, c])
    return out


def _coarse_components(
    n_rows: int,
    n_cols: int,
    blocked_h: set[tuple[int, int]],
    blocked_v: set[tuple[int, int]],
) -> list[set[Cell]]:
    unseen = {(r, c) for r in range(n_rows) for c in range(n_cols)}
    comps: list[set[Cell]] = []
    while unseen:
        start = unseen.pop()
        comp = {start}
        q = deque([start])
        while q:
            r, c = q.popleft()
            for nbr in _graph_neighbors(r, c, n_rows, n_cols, blocked_h, blocked_v):
                if nbr in unseen:
                    unseen.remove(nbr)
                    comp.add(nbr)
                    q.append(nbr)
        comps.append(comp)
    return comps


def _sample_wall_sets(n_rows: int, n_cols: int) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    blocked_h: set[tuple[int, int]] = set()
    blocked_v: set[tuple[int, int]] = set()

    fence_count = random.randint(1, 3)
    for _ in range(fence_count):
        if random.random() < 0.5:
            boundary = random.randrange(n_cols - 1)
            length = random.randint(2, max(2, n_rows - 1))
            start = random.randint(0, n_rows - length)
            for r in range(start, start + length):
                blocked_v.add((r, boundary))
        else:
            boundary = random.randrange(n_rows - 1)
            length = random.randint(2, max(2, n_cols - 1))
            start = random.randint(0, n_cols - length)
            for c in range(start, start + length):
                blocked_h.add((boundary, c))

    all_h = [(r, c) for r in range(n_rows - 1) for c in range(n_cols)]
    all_v = [(r, c) for r in range(n_rows) for c in range(n_cols - 1)]
    extra = random.randint(0, max(1, (n_rows + n_cols) // 3))
    for _ in range(extra):
        if random.random() < 0.5 and all_h:
            blocked_h.add(random.choice(all_h))
        elif all_v:
            blocked_v.add(random.choice(all_v))
    return blocked_h, blocked_v


def _assign_marker_sources(
    n_rows: int,
    n_cols: int,
    colors: list[int],
    want_duplicates: bool,
) -> dict[int, set[Cell]]:
    counts = [1 for _ in colors]
    if want_duplicates:
        extras = 1 + int(random.random() < 0.45)
        for _ in range(extras):
            counts[random.randrange(len(counts))] += 1

    total = sum(counts)
    all_cells = [(r, c) for r in range(n_rows) for c in range(n_cols)]
    chosen = random.sample(all_cells, total)
    random.shuffle(chosen)

    sources_by_color: dict[int, set[Cell]] = {}
    idx = 0
    for color, count in zip(colors, counts):
        sources_by_color[color] = set(chosen[idx : idx + count])
        idx += count
    return sources_by_color


def _nearest_pixel_map(
    row_bands: list[tuple[int, int]],
    col_bands: list[tuple[int, int]],
    sources_by_color: dict[int, set[Cell]],
    marker_pixels: dict[Cell, tuple[int, int, int]],
) -> np.ndarray:
    n_rows = len(row_bands)
    n_cols = len(col_bands)
    out = np.full((n_rows, n_cols), -1, dtype=int)
    markers_by_color: dict[int, list[tuple[int, int]]] = {}
    for _cell, (mr, mc, color) in marker_pixels.items():
        markers_by_color.setdefault(color, []).append((mr, mc))

    for r, (r0, r1) in enumerate(row_bands):
        center_r2 = r0 + r1
        for c, (c0, c1) in enumerate(col_bands):
            center_c2 = c0 + c1
            per_color = {}
            for color in sources_by_color:
                best = min(
                    (center_r2 - 2 * mr) ** 2 + (center_c2 - 2 * mc) ** 2
                    for mr, mc in markers_by_color[color]
                )
                per_color[color] = best
            min_dist = min(per_color.values())
            winners = [color for color, dist in per_color.items() if dist == min_dist]
            out[r, c] = PINK if len(winners) > 1 else winners[0]
    return out


def _validate_instance(
    grid: np.ndarray,
    row_sizes: list[int],
    col_sizes: list[int],
    blocked_h: set[tuple[int, int]],
    blocked_v: set[tuple[int, int]],
    sources_by_color: dict[int, set[Cell]],
    want_tie: bool,
) -> bool:
    n_rows = len(row_sizes)
    n_cols = len(col_sizes)
    if len(blocked_h) + len(blocked_v) < 3:
        return False
    if len(set(row_sizes)) == 1 and len(set(col_sizes)) == 1:
        return False

    comps = _coarse_components(n_rows, n_cols, blocked_h, blocked_v)
    if len(comps) != 1:
        return False

    remaining_edges = (n_rows - 1) * n_cols + n_rows * (n_cols - 1) - len(blocked_h) - len(blocked_v)
    if remaining_edges < n_rows * n_cols:
        return False

    row_bands, _row_lines, _height = _layout_from_sizes(row_sizes)
    col_bands, _col_lines, _width = _layout_from_sizes(col_sizes)
    marker_grid, _, _, marker_pixels = _render_input(row_sizes, col_sizes, blocked_h, blocked_v, sources_by_color)
    fill_map, best_dist = _solve_coarse(n_rows, n_cols, blocked_h, blocked_v, sources_by_color)
    expected = _render_output(marker_grid, row_bands, col_bands, fill_map)
    if not np.array_equal(grid, marker_grid):
        return False
    if not np.array_equal(main(grid), expected):
        return False

    distinct_output_colors = set(map(int, fill_map.ravel()))
    if len(distinct_output_colors) < 2:
        return False
    if int(best_dist.max()) < 2:
        return False

    pink_cells = {
        (r, c)
        for r in range(n_rows)
        for c in range(n_cols)
        if int(fill_map[r, c]) == PINK
    }
    marker_cells = {cell for cells in sources_by_color.values() for cell in cells}
    has_non_marker_tie = bool(pink_cells - marker_cells)
    if want_tie and not has_non_marker_tie:
        return False
    if not want_tie and pink_cells:
        return False

    no_wall_map, _ = _solve_coarse(n_rows, n_cols, set(), set(), sources_by_color)
    if np.array_equal(fill_map, no_wall_map):
        return False

    pixel_map = _nearest_pixel_map(row_bands, col_bands, sources_by_color, marker_pixels)
    if np.array_equal(fill_map, pixel_map):
        return False

    if any(len(cells) > 1 for cells in sources_by_color.values()):
        per_color_territory = {
            color: int(np.count_nonzero(fill_map == color))
            for color in sources_by_color
        }
        if max(per_color_territory.values()) < 2:
            return False

    return True


def _generate_candidate(want_tie: bool, want_three_colors: bool, want_duplicates: bool) -> np.ndarray | None:
    n_rows = random.randint(4, 6)
    n_cols = random.randint(4, 6)

    row_sizes = [random.randint(2, 4) for _ in range(n_rows)]
    col_sizes = [random.randint(2, 4) for _ in range(n_cols)]
    if len(set(row_sizes)) == 1 and len(set(col_sizes)) == 1:
        if random.random() < 0.5:
            row_sizes[random.randrange(n_rows)] += 1
        else:
            col_sizes[random.randrange(n_cols)] += 1

    _row_bands, _row_lines, height = _layout_from_sizes(row_sizes)
    _col_bands, _col_lines, width = _layout_from_sizes(col_sizes)
    if height > 30 or width > 30:
        return None

    color_count = 3 if want_three_colors else 2
    colors = random.sample(MARKER_PALETTE, color_count)
    sources_by_color = _assign_marker_sources(n_rows, n_cols, colors, want_duplicates)

    blocked_h, blocked_v = _sample_wall_sets(n_rows, n_cols)
    grid, _row_bands, _col_bands, _marker_pixels = _render_input(
        row_sizes, col_sizes, blocked_h, blocked_v, sources_by_color
    )
    if not _validate_instance(
        grid,
        row_sizes,
        col_sizes,
        blocked_h,
        blocked_v,
        sources_by_color,
        want_tie,
    ):
        return None
    return grid


def main(grid: np.ndarray) -> np.ndarray:
    """
    Color each lattice cell by the wall-aware nearest marker color, turning equal-distance frontiers pink.

    Input structure:
    - A teal lattice of full rows and full columns partitions the canvas into rectangular cells with possibly unequal sizes.
    - Some lattice segments between neighboring cells are recolored red, forming walls on those shared borders.
    - A few cell interiors contain single marker pixels of non-red, non-teal colors; pink does not appear in the input.

    Rule:
    - Treat each rectangle as one coarse cell. Two cells are adjacent exactly when they share a teal lattice segment that is not replaced by a red wall segment.
    - For every cell, measure the 4-neighbor shortest-path distance on that coarse cell graph to the marker cells of each color.
    - Fill the whole cell with the uniquely nearest marker color; if different colors attain the same minimum distance, fill the whole cell pink. Marker cells keep their own color, and the teal lattice plus red wall segments stay unchanged.

    Key insight:
    - The visible geometry is a coarse maze: distances are counted in cell steps after deleting adjacencies where red wall segments sit, so pixel proximity and unequal cell sizes are both decoys.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    height, width = map(int, g.shape)

    line_palette = np.array([TEAL, RED], dtype=int)
    row_lines = [
        r
        for r in range(height)
        if np.all(np.isin(g[r, :], line_palette))
    ]
    col_lines = [
        c
        for c in range(width)
        if np.all(np.isin(g[:, c], line_palette))
    ]
    assert row_lines and col_lines, "expected an internal lattice"
    assert all(0 < r < height - 1 for r in row_lines), "lattice rows must be internal"
    assert all(0 < c < width - 1 for c in col_lines), "lattice cols must be internal"
    assert all(g[r, c] == TEAL for r in row_lines for c in col_lines), "lattice intersections must stay teal"

    row_bands = _cell_ranges(height, row_lines)
    col_bands = _cell_ranges(width, col_lines)
    n_rows = len(row_bands)
    n_cols = len(col_bands)

    blocked_h: set[tuple[int, int]] = set()
    for r, line_row in enumerate(row_lines):
        for c, (c0, c1) in enumerate(col_bands):
            segment = g[line_row, c0 : c1 + 1]
            if np.all(segment == RED):
                blocked_h.add((r, c))
            else:
                assert np.all(segment == TEAL), "horizontal lattice segment must be uniformly teal or red"

    blocked_v: set[tuple[int, int]] = set()
    for r, (r0, r1) in enumerate(row_bands):
        for c, line_col in enumerate(col_lines):
            segment = g[r0 : r1 + 1, line_col]
            if np.all(segment == RED):
                blocked_v.add((r, c))
            else:
                assert np.all(segment == TEAL), "vertical lattice segment must be uniformly teal or red"

    sources_by_color: dict[int, set[Cell]] = {}
    cell_to_color: dict[Cell, int] = {}
    marker_mask = ~np.isin(g, np.array([BLACK, TEAL, RED], dtype=int))
    assert not np.any(g == PINK), "pink should be reserved for output ties"
    for rr, cc in np.argwhere(marker_mask):
        r = _band_index(int(rr), row_bands)
        c = _band_index(int(cc), col_bands)
        cell = (r, c)
        color = int(g[int(rr), int(cc)])
        if cell in cell_to_color:
            assert cell_to_color[cell] == color, "a cell cannot contain markers of different colors"
        cell_to_color[cell] = color
        sources_by_color.setdefault(color, set()).add(cell)
    assert len(sources_by_color) >= 2, "expected at least two marker colors"

    fill_map, _best = _solve_coarse(n_rows, n_cols, blocked_h, blocked_v, sources_by_color)
    return _render_output(g, row_bands, col_bands, fill_map)


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        want_tie = random.random() < TIE_PROBABILITY
        want_three_colors = random.random() < THREE_COLOR_PROBABILITY
        want_duplicates = random.random() < DUPLICATE_COLOR_PROBABILITY
        candidate = _generate_candidate(want_tie, want_three_colors, want_duplicates)
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid wall-aware nearest-color lattice")
