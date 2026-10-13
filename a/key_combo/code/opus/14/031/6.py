from __future__ import annotations

from collections import deque

import numpy as np


BG = 0
LINE = 1
NOISE_COLORS = (2, 3, 4, 5)

N_RANGE = (3, 5)
CELL_SIDE_RANGE = (3, 4)
FG_COUNT_MIN = 2
FG_COUNT_MAX = 4
EXTRA_REACHABLE_EDGE_PROB = 0.35
EXTRA_UNREACHABLE_EDGE_PROB = 0.22
MAX_GENERATION_ATTEMPTS = 400


Cell = tuple[int, int]
Edge = tuple[Cell, Cell]


def _normalize_edge(a: Cell, b: Cell) -> Edge:
    return (a, b) if a <= b else (b, a)


def _bands(size: int, separators: list[int]) -> list[tuple[int, int]]:
    edges = [-1, *separators, size]
    bands = [(a + 1, b) for a, b in zip(edges, edges[1:])]
    assert all(start < end for start, end in bands), bands
    widths = {end - start for start, end in bands}
    assert len(widths) == 1, bands
    return bands


def _separator_indices(grid: np.ndarray, *, rows: bool) -> list[int]:
    counts = np.count_nonzero(grid == LINE, axis=1 if rows else 0)
    threshold = (grid.shape[1] if rows else grid.shape[0]) // 2
    separators = [int(i) for i, count in enumerate(counts) if int(count) > threshold]
    assert separators, "failed to detect separator lines"
    if len(separators) > 1:
        gaps = {b - a for a, b in zip(separators, separators[1:])}
        assert len(gaps) == 1, separators
    return separators


def _parse_layout(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]], int]:
    row_separators = _separator_indices(grid, rows=True)
    col_separators = _separator_indices(grid, rows=False)
    row_bands = _bands(grid.shape[0], row_separators)
    col_bands = _bands(grid.shape[1], col_separators)
    cell_sizes = {row_bands[0][1] - row_bands[0][0], col_bands[0][1] - col_bands[0][0]}
    assert len(cell_sizes) == 1, cell_sizes
    cell_side = next(iter(cell_sizes))
    return row_bands, col_bands, int(cell_side)


def _cell_foreground_colors(cell: np.ndarray) -> set[int]:
    return set(map(int, np.unique(cell))) - {BG, LINE}


def _find_source_cell(
    grid: np.ndarray, row_bands: list[tuple[int, int]], col_bands: list[tuple[int, int]]
) -> tuple[Cell, int]:
    candidates: list[tuple[Cell, int]] = []
    for br, (r0, r1) in enumerate(row_bands):
        for bc, (c0, c1) in enumerate(col_bands):
            fg_colors = _cell_foreground_colors(grid[r0:r1, c0:c1])
            assert fg_colors, "every cell must contain foreground pixels"
            if len(fg_colors) == 1:
                candidates.append(((br, bc), next(iter(fg_colors))))
    assert len(candidates) == 1, f"expected one monochrome source cell, got {len(candidates)}"
    return candidates[0]


def _build_graph(
    grid: np.ndarray, row_bands: list[tuple[int, int]], col_bands: list[tuple[int, int]]
) -> dict[Cell, set[Cell]]:
    n_rows = len(row_bands)
    n_cols = len(col_bands)
    graph = {(r, c): set() for r in range(n_rows) for c in range(n_cols)}

    for br, (r0, r1) in enumerate(row_bands):
        for bc in range(n_cols - 1):
            separator_col = col_bands[bc][1]
            if np.any(grid[r0:r1, separator_col] == BG):
                a, b = (br, bc), (br, bc + 1)
                graph[a].add(b)
                graph[b].add(a)

    for br in range(n_rows - 1):
        separator_row = row_bands[br][1]
        for bc, (c0, c1) in enumerate(col_bands):
            if np.any(grid[separator_row, c0:c1] == BG):
                a, b = (br, bc), (br + 1, bc)
                graph[a].add(b)
                graph[b].add(a)

    return graph


def _reachable(graph: dict[Cell, set[Cell]], source: Cell) -> set[Cell]:
    seen = {source}
    queue = deque([source])
    while queue:
        cur = queue.popleft()
        for nxt in graph[cur]:
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return seen


def _recolor_reachable(
    grid: np.ndarray,
    row_bands: list[tuple[int, int]],
    col_bands: list[tuple[int, int]],
    reachable_cells: set[Cell],
    color: int,
) -> np.ndarray:
    out = grid.copy()
    for br, bc in reachable_cells:
        r0, r1 = row_bands[br]
        c0, c1 = col_bands[bc]
        cell = out[r0:r1, c0:c1]
        mask = (cell != BG) & (cell != LINE)
        cell[mask] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor every cell reachable from the unique monochrome source through separator gaps.

    Input structure:
    - The grid has black background with blue separator lines that partition it into equal square cells.
    - Each cell contains a few non-blue foreground pixels, and exactly one cell is monochrome.
    - Some separator segments contain black one-pixel gaps that open passages between adjacent cells.

    Rule:
    - Detect the equal cell partition from the nearly complete blue rows and columns.
    - Find the unique cell whose non-black, non-blue foreground pixels are all one color; this cell supplies the flood color and start cell.
    - Build a cell-level adjacency graph where two orthogonally adjacent cells are connected iff their shared separator segment contains a black gap.
    - Breadth-first flood from the monochrome source on that cell graph, and in every reachable cell recolor all foreground pixels to the source color while leaving black background, blue separators, and gap pixels unchanged.

    Key insight:
    - The flood is not a pixel flood: pixel gaps encode edges of a higher-level graph whose nodes are the separator-defined cells.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    row_bands, col_bands, _cell_side = _parse_layout(g)
    source_cell, source_color = _find_source_cell(g, row_bands, col_bands)
    graph = _build_graph(g, row_bands, col_bands)
    reachable_cells = _reachable(graph, source_cell)
    return _recolor_reachable(g, row_bands, col_bands, reachable_cells, source_color)


def _cell_origin(cell: Cell, cell_side: int) -> tuple[int, int]:
    br, bc = cell
    return br * (cell_side + 1), bc * (cell_side + 1)


def _all_cells(n: int) -> list[Cell]:
    return [(r, c) for r in range(n) for c in range(n)]


def _grid_neighbors(cell: Cell, n: int) -> list[Cell]:
    r, c = cell
    out = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < n:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < n:
        out.append((r, c + 1))
    return out


def _all_lattice_edges(n: int) -> list[Edge]:
    edges: list[Edge] = []
    for cell in _all_cells(n):
        for nbr in _grid_neighbors(cell, n):
            edge = _normalize_edge(cell, nbr)
            if edge not in edges:
                edges.append(edge)
    return edges


def _distance_map(source: Cell, open_edges: set[Edge], n: int) -> dict[Cell, int]:
    graph = {cell: set() for cell in _all_cells(n)}
    for a, b in open_edges:
        graph[a].add(b)
        graph[b].add(a)
    dist = {source: 0}
    queue = deque([source])
    while queue:
        cur = queue.popleft()
        for nxt in graph[cur]:
            if nxt not in dist:
                dist[nxt] = dist[cur] + 1
                queue.append(nxt)
    return dist


def _grow_reachable_component(rng: np.random.Generator, n: int, source: Cell, size: int) -> tuple[set[Cell], set[Edge]]:
    reachable = {source}
    tree_edges: set[Edge] = set()
    while len(reachable) < size:
        frontier: list[tuple[Cell, Cell]] = []
        for cell in sorted(reachable):
            for nbr in _grid_neighbors(cell, n):
                if nbr not in reachable:
                    frontier.append((cell, nbr))
        if not frontier:
            raise AssertionError("connected growth frontier vanished")
        parent, child = frontier[int(rng.integers(0, len(frontier)))]
        reachable.add(child)
        tree_edges.add(_normalize_edge(parent, child))
    return reachable, tree_edges


def _blank_partitioned_grid(n: int, cell_side: int) -> np.ndarray:
    side = n * cell_side + (n - 1)
    grid = np.zeros((side, side), dtype=int)
    for band in range(1, n):
        separator = band * cell_side + (band - 1)
        grid[separator, :] = LINE
        grid[:, separator] = LINE
    return grid


def _carve_gap(grid: np.ndarray, edge: Edge, cell_side: int, rng: np.random.Generator) -> None:
    (r1, c1), (r2, c2) = edge
    if r1 == r2:
        left = (r1, min(c1, c2))
        base_r, base_c = _cell_origin(left, cell_side)
        separator_col = base_c + cell_side
        gap_row = int(base_r + rng.integers(0, cell_side))
        grid[gap_row, separator_col] = BG
        return

    top = (min(r1, r2), c1)
    base_r, base_c = _cell_origin(top, cell_side)
    separator_row = base_r + cell_side
    gap_col = int(base_c + rng.integers(0, cell_side))
    grid[separator_row, gap_col] = BG


def _sample_positions(rng: np.random.Generator, cell_side: int, count: int) -> np.ndarray:
    area = cell_side * cell_side
    return rng.choice(area, size=count, replace=False)


def _make_source_cell(rng: np.random.Generator, cell_side: int, color: int) -> np.ndarray:
    cell = np.zeros((cell_side, cell_side), dtype=int)
    count = int(rng.integers(FG_COUNT_MIN, min(FG_COUNT_MAX, cell_side * cell_side) + 1))
    pos = _sample_positions(rng, cell_side, count)
    cell.flat[pos] = int(color)
    return cell


def _make_distractor_cell(
    rng: np.random.Generator,
    cell_side: int,
    source_color: int,
    *,
    force_include_source: bool,
) -> np.ndarray:
    cell = np.zeros((cell_side, cell_side), dtype=int)
    count = int(rng.integers(FG_COUNT_MIN, min(FG_COUNT_MAX, cell_side * cell_side) + 1))
    max_colors = min(3, count, len(NOISE_COLORS))
    color_count = 2 if max_colors == 2 else int(rng.integers(2, max_colors + 1))

    palette = list(NOISE_COLORS)
    if force_include_source:
        others = [color for color in palette if color != source_color]
        chosen = [source_color]
        extra = rng.choice(others, size=color_count - 1, replace=False)
        chosen.extend(int(x) for x in extra)
    else:
        chosen = [int(x) for x in rng.choice(palette, size=color_count, replace=False)]

    rng.shuffle(chosen)
    pos = _sample_positions(rng, cell_side, count)
    for idx, color in enumerate(chosen):
        cell.flat[int(pos[idx])] = int(color)
    for idx in range(len(chosen), count):
        cell.flat[int(pos[idx])] = int(chosen[int(rng.integers(0, len(chosen)))])
    assert len(_cell_foreground_colors(cell)) >= 2
    return cell


def _render_expected(
    grid: np.ndarray,
    n: int,
    cell_side: int,
    reachable_cells: set[Cell],
    source_color: int,
) -> np.ndarray:
    row_bands = [(r * (cell_side + 1), r * (cell_side + 1) + cell_side) for r in range(n)]
    col_bands = [(c * (cell_side + 1), c * (cell_side + 1) + cell_side) for c in range(n)]
    return _recolor_reachable(grid, row_bands, col_bands, reachable_cells, source_color)


def _find_source_with_known_layout(grid: np.ndarray, n: int, cell_side: int) -> tuple[Cell, int] | None:
    hits: list[tuple[Cell, int]] = []
    for br in range(n):
        for bc in range(n):
            r0, c0 = _cell_origin((br, bc), cell_side)
            fg_colors = _cell_foreground_colors(grid[r0 : r0 + cell_side, c0 : c0 + cell_side])
            if not fg_colors:
                return None
            if len(fg_colors) == 1:
                hits.append(((br, bc), next(iter(fg_colors))))
    if len(hits) != 1:
        return None
    return hits[0]


def _infer_open_edges(grid: np.ndarray, n: int, cell_side: int) -> set[Edge]:
    edges: set[Edge] = set()
    for br in range(n):
        for bc in range(n - 1):
            r0, c0 = _cell_origin((br, bc), cell_side)
            separator_col = c0 + cell_side
            if np.any(grid[r0 : r0 + cell_side, separator_col] == BG):
                edges.add(_normalize_edge((br, bc), (br, bc + 1)))
    for br in range(n - 1):
        for bc in range(n):
            r0, c0 = _cell_origin((br, bc), cell_side)
            separator_row = r0 + cell_side
            if np.any(grid[separator_row, c0 : c0 + cell_side] == BG):
                edges.add(_normalize_edge((br, bc), (br + 1, bc)))
    return edges


def _validate_instance(
    grid: np.ndarray,
    expected: np.ndarray,
    *,
    n: int,
    cell_side: int,
    source: Cell,
    source_color: int,
    open_edges: set[Edge],
    reachable_cells: set[Cell],
) -> bool:
    if grid.shape != (n * cell_side + (n - 1), n * cell_side + (n - 1)):
        return False
    if set(map(int, np.unique(grid))) - ({BG, LINE} | set(NOISE_COLORS)):
        return False

    parsed = _find_source_with_known_layout(grid, n, cell_side)
    if parsed != (source, source_color):
        return False
    if _infer_open_edges(grid, n, cell_side) != open_edges:
        return False

    distances = _distance_map(source, open_edges, n)
    if set(distances) != reachable_cells:
        return False
    if len(reachable_cells) < 3 or len(reachable_cells) >= n * n:
        return False
    if max(distances.values()) < 2:
        return False
    if len({r for r, _ in reachable_cells}) < 2 or len({c for _, c in reachable_cells}) < 2:
        return False

    changed = expected != grid
    if int(np.count_nonzero(changed)) == 0:
        return False

    unreachable = [cell for cell in _all_cells(n) if cell not in reachable_cells]
    reachable_distractors = [cell for cell in reachable_cells if cell != source]
    if not unreachable or not reachable_distractors:
        return False

    saw_reachable_with_source = False
    saw_unreachable_with_source = False
    saw_unchanged_unreachable_fg = False
    all_fg_colors = set()
    for br, bc in _all_cells(n):
        r0, c0 = _cell_origin((br, bc), cell_side)
        inp_cell = grid[r0 : r0 + cell_side, c0 : c0 + cell_side]
        out_cell = expected[r0 : r0 + cell_side, c0 : c0 + cell_side]
        fg_colors = _cell_foreground_colors(inp_cell)
        all_fg_colors |= fg_colors
        if (br, bc) in reachable_cells:
            if (br, bc) != source and source_color in fg_colors:
                saw_reachable_with_source = True
            if (br, bc) == source:
                if not np.array_equal(inp_cell, out_cell):
                    return False
            else:
                if np.array_equal(inp_cell, out_cell):
                    return False
                out_colors = _cell_foreground_colors(out_cell)
                if out_colors != {source_color}:
                    return False
        else:
            if source_color in fg_colors:
                saw_unreachable_with_source = True
            if np.array_equal(inp_cell, out_cell):
                saw_unchanged_unreachable_fg = True
            else:
                return False

    if all_fg_colors != set(NOISE_COLORS):
        return False
    if not saw_reachable_with_source or not saw_unreachable_with_source:
        return False
    if not saw_unchanged_unreachable_fg:
        return False

    oracle = _render_expected(grid, n, cell_side, reachable_cells, source_color)
    return np.array_equal(oracle, expected)


def _generate_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    n = int(rng.integers(N_RANGE[0], N_RANGE[1] + 1))
    cell_side = int(rng.integers(CELL_SIDE_RANGE[0], CELL_SIDE_RANGE[1] + 1))
    total_cells = n * n
    source = (int(rng.integers(0, n)), int(rng.integers(0, n)))

    reachable_size = int(rng.integers(3, total_cells))
    reachable_cells, tree_edges = _grow_reachable_component(rng, n, source, reachable_size)
    open_edges = set(tree_edges)

    for edge in _all_lattice_edges(n):
        if edge in open_edges:
            continue
        a, b = edge
        if a in reachable_cells and b in reachable_cells and rng.random() < EXTRA_REACHABLE_EDGE_PROB:
            open_edges.add(edge)

    unreachable_cells = [cell for cell in _all_cells(n) if cell not in reachable_cells]
    unreachable_set = set(unreachable_cells)
    for edge in _all_lattice_edges(n):
        if edge in open_edges:
            continue
        a, b = edge
        if a in unreachable_set and b in unreachable_set and rng.random() < EXTRA_UNREACHABLE_EDGE_PROB:
            open_edges.add(edge)

    source_color = int(NOISE_COLORS[int(rng.integers(0, len(NOISE_COLORS)))])
    grid = _blank_partitioned_grid(n, cell_side)
    for edge in open_edges:
        _carve_gap(grid, edge, cell_side, rng)

    reachable_distractors = sorted(cell for cell in reachable_cells if cell != source)
    if not reachable_distractors or not unreachable_cells:
        return None
    force_reachable = reachable_distractors[int(rng.integers(0, len(reachable_distractors)))]
    force_unreachable = unreachable_cells[int(rng.integers(0, len(unreachable_cells)))]

    for cell in _all_cells(n):
        r0, c0 = _cell_origin(cell, cell_side)
        if cell == source:
            cell_grid = _make_source_cell(rng, cell_side, source_color)
        else:
            force_include_source = cell in {force_reachable, force_unreachable}
            cell_grid = _make_distractor_cell(
                rng,
                cell_side,
                source_color,
                force_include_source=force_include_source,
            )
        grid[r0 : r0 + cell_side, c0 : c0 + cell_side] = cell_grid

    expected = _render_expected(grid, n, cell_side, reachable_cells, source_color)
    if not _validate_instance(
        grid,
        expected,
        n=n,
        cell_side=cell_side,
        source=source,
        source_color=source_color,
        open_edges=open_edges,
        reachable_cells=reachable_cells,
    ):
        return None

    try:
        solved = main(grid)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
