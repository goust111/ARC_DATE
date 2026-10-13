from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7
TEAL = 8

SOURCE_COLORS = (BLUE, RED, GREEN, YELLOW)
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
GRID_BANDS = 4
GEN_MAX_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _full_divider_indices(grid: np.ndarray, color: int) -> tuple[list[int], list[int]]:
    row_dividers = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == color)]
    col_dividers = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == color)]
    return row_dividers, col_dividers


def _non_divider_runs(length: int, divider_indices: list[int]) -> list[tuple[int, int]]:
    dividers = set(int(i) for i in divider_indices)
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx in range(length):
        if idx in dividers:
            if start is not None:
                runs.append((start, idx - 1))
                start = None
        elif start is None:
            start = idx
    if start is not None:
        runs.append((start, length - 1))
    return runs


def _partition_table(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    row_dividers, col_dividers = _full_divider_indices(grid, GREY)
    row_runs = _non_divider_runs(grid.shape[0], row_dividers)
    col_runs = _non_divider_runs(grid.shape[1], col_dividers)
    assert len(row_runs) >= 2 and len(col_runs) >= 2, "expected a divider table"
    return row_runs, col_runs


def _classify_cell(region: np.ndarray) -> dict[str, object]:
    colors = set(map(int, np.unique(region)))
    assert GREY not in colors, "divider color must not appear inside a cell"

    if colors <= {BLACK, ORANGE, TEAL}:
        orange_pts = np.argwhere(region == ORANGE)
        teal_pts = np.argwhere(region == TEAL)
        assert orange_pts.shape[0] == 1 and teal_pts.shape[0] == 1, "pointer cell must contain one orange and one teal pixel"
        assert int(np.count_nonzero(region != BLACK)) == 2, "pointer cell must contain exactly one domino"
        orange = tuple(map(int, orange_pts[0]))
        teal = tuple(map(int, teal_pts[0]))
        direction = (teal[0] - orange[0], teal[1] - orange[1])
        assert direction in DIRS4, "domino must be 4-connected"
        return {"kind": "pointer", "direction": direction}

    assert len(colors) == 1, "source cell must be uniformly filled"
    color = next(iter(colors))
    assert color in SOURCE_COLORS, "source cell color must be one of the four foreground colors"
    return {"kind": "source", "color": color}


def _extract_cells(grid: np.ndarray):
    row_runs, col_runs = _partition_table(grid)
    cells: dict[tuple[int, int], dict[str, object]] = {}
    for ri, (r0, r1) in enumerate(row_runs):
        for ci, (c0, c1) in enumerate(col_runs):
            region = grid[r0 : r1 + 1, c0 : c1 + 1]
            cells[(ri, ci)] = _classify_cell(region)
    return row_runs, col_runs, cells


def _resolve_final_colors(cells: dict[tuple[int, int], dict[str, object]], n_rows: int, n_cols: int) -> dict[tuple[int, int], int]:
    resolved: dict[tuple[int, int], int] = {}
    visiting: set[tuple[int, int]] = set()

    def resolve(cell: tuple[int, int]) -> int:
        if cell in resolved:
            return resolved[cell]
        assert cell not in visiting, "pointer graph must be acyclic"
        visiting.add(cell)

        info = cells[cell]
        if info["kind"] == "source":
            color = int(info["color"])
        else:
            dr, dc = info["direction"]
            nr = cell[0] + int(dr)
            nc = cell[1] + int(dc)
            assert 0 <= nr < n_rows and 0 <= nc < n_cols, "pointer must stay inside the table"
            color = resolve((nr, nc))

        visiting.remove(cell)
        resolved[cell] = color
        return color

    for cell in cells:
        resolve(cell)
    return resolved


def _paint_merged_dividers(
    out: np.ndarray,
    row_runs: list[tuple[int, int]],
    col_runs: list[tuple[int, int]],
    final_colors: dict[tuple[int, int], int],
) -> None:
    for ri, (r0, r1) in enumerate(row_runs):
        for ci in range(len(col_runs) - 1):
            color = final_colors[(ri, ci)]
            if color != final_colors[(ri, ci + 1)]:
                continue
            left_c1 = col_runs[ci][1]
            right_c0 = col_runs[ci + 1][0]
            out[r0 : r1 + 1, left_c1 + 1 : right_c0] = color

    for ri in range(len(row_runs) - 1):
        top_r1 = row_runs[ri][1]
        bottom_r0 = row_runs[ri + 1][0]
        for ci, (c0, c1) in enumerate(col_runs):
            color = final_colors[(ri, ci)]
            if color != final_colors[(ri + 1, ci)]:
                continue
            out[top_r1 + 1 : bottom_r0, c0 : c1 + 1] = color

    for ri in range(len(row_runs) - 1):
        top_r1 = row_runs[ri][1]
        bottom_r0 = row_runs[ri + 1][0]
        for ci in range(len(col_runs) - 1):
            colors = {
                final_colors[(ri, ci)],
                final_colors[(ri, ci + 1)],
                final_colors[(ri + 1, ci)],
                final_colors[(ri + 1, ci + 1)],
            }
            if len(colors) != 1:
                continue
            left_c1 = col_runs[ci][1]
            right_c0 = col_runs[ci + 1][0]
            out[top_r1 + 1 : bottom_r0, left_c1 + 1 : right_c0] = next(iter(colors))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate source colors through domino pointers across a divider table, then erase separators between equal-colored neighbors.

    Input structure:
    - Gray full-row and full-column lines partition the grid into rectangular table cells.
    - Each cell is either a solid source cell colored blue, red, green, or yellow, or a pointer cell with black background plus exactly one orange-teal domino.
    - The domino direction from orange to teal points to one orthogonally adjacent cell in the table.

    Rule:
    - Split the grid into cells using the gray divider lines.
    - For every pointer cell, follow its domino direction to the neighboring cell; if that cell is another pointer, continue recursively until reaching a source cell.
    - Fill each pointer cell completely with the resolved source color while keeping gray divider lines initially intact.
    - After all cells are colored, recolor each divider segment between same-colored neighboring cells to that shared color; keep different-color divider segments gray.

    Key insight:
    - The domino does not encode a color directly; it encodes which neighboring cell to inherit from, so the solution requires tracing pointer chains through the table before deciding which separators disappear.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    row_runs, col_runs, cells = _extract_cells(g)
    final_colors = _resolve_final_colors(cells, len(row_runs), len(col_runs))

    out = g.copy()
    for (ri, ci), color in final_colors.items():
        r0, r1 = row_runs[ri]
        c0, c1 = col_runs[ci]
        out[r0 : r1 + 1, c0 : c1 + 1] = color

    _paint_merged_dividers(out, row_runs, col_runs, final_colors)
    return out


def _cell_runs_from_sizes(sizes: list[int]) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = 0
    for size in sizes:
        runs.append((start, start + size - 1))
        start += size + 1
    return runs


def _choose_sizes(rng: random.Random) -> tuple[list[int], list[int]]:
    row_sizes = [rng.randint(2, 5) for _ in range(GRID_BANDS)]
    col_sizes = [rng.randint(2, 5) for _ in range(GRID_BANDS)]
    return row_sizes, col_sizes


def _sample_sources(rng: random.Random) -> list[tuple[int, int]]:
    cells = [(r, c) for r in range(GRID_BANDS) for c in range(GRID_BANDS)]
    source_count = rng.choice([4, 4, 5])
    return rng.sample(cells, source_count)


def _build_pointer_forest(
    rng: random.Random, sources: list[tuple[int, int]]
) -> tuple[dict[tuple[int, int], tuple[int, int]], dict[tuple[int, int], int]]:
    cells = [(r, c) for r in range(GRID_BANDS) for c in range(GRID_BANDS)]
    assigned = set(sources)
    parent: dict[tuple[int, int], tuple[int, int]] = {}
    depth = {cell: 0 for cell in sources}

    while len(assigned) < len(cells):
        frontier = [
            cell
            for cell in cells
            if cell not in assigned and any(nei in assigned for nei in _neighbors4(*cell))
        ]
        child = rng.choice(frontier)
        candidates = [nei for nei in _neighbors4(*child) if nei in assigned]
        chosen_parent = rng.choice(candidates)
        parent[child] = chosen_parent
        depth[child] = depth[chosen_parent] + 1
        assigned.add(child)

    return parent, depth


def _assign_source_colors(
    rng: random.Random, sources: list[tuple[int, int]]
) -> dict[tuple[int, int], int]:
    n_distinct = rng.choice([2, 3, 4])
    n_distinct = min(n_distinct, len(sources))
    palette = rng.sample(list(SOURCE_COLORS), n_distinct)
    color_list = palette[:]
    while len(color_list) < len(sources):
        color_list.append(rng.choice(palette))
    rng.shuffle(color_list)
    return dict(zip(sources, color_list))


def _compute_final_colors_from_metadata(
    sources: list[tuple[int, int]],
    parent: dict[tuple[int, int], tuple[int, int]],
    source_colors: dict[tuple[int, int], int],
) -> dict[tuple[int, int], int]:
    final = {cell: int(color) for cell, color in source_colors.items()}

    def resolve(cell: tuple[int, int]) -> int:
        if cell in final:
            return final[cell]
        final[cell] = resolve(parent[cell])
        return final[cell]

    for cell in [(r, c) for r in range(GRID_BANDS) for c in range(GRID_BANDS)]:
        resolve(cell)
    return final


def _place_pointer_domino(
    region: np.ndarray, direction: tuple[int, int], rng: random.Random
) -> None:
    h, w = region.shape
    dr, dc = direction
    if dr == -1:
        c = rng.randrange(w)
        r = rng.randrange(1, h)
        orange = (r, c)
        teal = (r - 1, c)
    elif dr == 1:
        c = rng.randrange(w)
        r = rng.randrange(h - 1)
        orange = (r, c)
        teal = (r + 1, c)
    elif dc == -1:
        r = rng.randrange(h)
        c = rng.randrange(1, w)
        orange = (r, c)
        teal = (r, c - 1)
    else:
        assert dc == 1
        r = rng.randrange(h)
        c = rng.randrange(w - 1)
        orange = (r, c)
        teal = (r, c + 1)

    region[orange] = ORANGE
    region[teal] = TEAL


def _render_input(
    row_sizes: list[int],
    col_sizes: list[int],
    sources: list[tuple[int, int]],
    parent: dict[tuple[int, int], tuple[int, int]],
    source_colors: dict[tuple[int, int], int],
    rng: random.Random,
) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    height = sum(row_sizes) + GRID_BANDS - 1
    width = sum(col_sizes) + GRID_BANDS - 1
    grid = np.full((height, width), BLACK, dtype=int)
    row_runs = _cell_runs_from_sizes(row_sizes)
    col_runs = _cell_runs_from_sizes(col_sizes)

    for divider_r in [run[1] + 1 for run in row_runs[:-1]]:
        grid[divider_r, :] = GREY
    for divider_c in [run[1] + 1 for run in col_runs[:-1]]:
        grid[:, divider_c] = GREY

    source_set = set(sources)
    for ri, (r0, r1) in enumerate(row_runs):
        for ci, (c0, c1) in enumerate(col_runs):
            region = grid[r0 : r1 + 1, c0 : c1 + 1]
            if (ri, ci) in source_set:
                region[:, :] = source_colors[(ri, ci)]
            else:
                pr, pc = parent[(ri, ci)]
                direction = (pr - ri, pc - ci)
                _place_pointer_domino(region, direction, rng)

    return grid, row_runs, col_runs


def _render_expected_output(
    inp: np.ndarray,
    row_runs: list[tuple[int, int]],
    col_runs: list[tuple[int, int]],
    final_colors: dict[tuple[int, int], int],
) -> np.ndarray:
    out = inp.copy()
    for (ri, ci), color in final_colors.items():
        r0, r1 = row_runs[ri]
        c0, c1 = col_runs[ci]
        out[r0 : r1 + 1, c0 : c1 + 1] = color
    _paint_merged_dividers(out, row_runs, col_runs, final_colors)
    return out


def _adjacency_stats(final_colors: dict[tuple[int, int], int]) -> tuple[int, int]:
    equal_edges = 0
    different_edges = 0
    for r in range(GRID_BANDS):
        for c in range(GRID_BANDS):
            if r + 1 < GRID_BANDS:
                if final_colors[(r, c)] == final_colors[(r + 1, c)]:
                    equal_edges += 1
                else:
                    different_edges += 1
            if c + 1 < GRID_BANDS:
                if final_colors[(r, c)] == final_colors[(r, c + 1)]:
                    equal_edges += 1
                else:
                    different_edges += 1
    return equal_edges, different_edges


def _is_valid_instance(
    inp: np.ndarray,
    row_runs: list[tuple[int, int]],
    col_runs: list[tuple[int, int]],
    sources: list[tuple[int, int]],
    parent: dict[tuple[int, int], tuple[int, int]],
    depth: dict[tuple[int, int], int],
    source_colors: dict[tuple[int, int], int],
) -> bool:
    if inp.ndim != 2 or max(inp.shape) > 30:
        return False
    if len(sources) not in {4, 5}:
        return False
    if max(depth.values()) < 2:
        return False
    if not any(d > 1 for d in depth.values()):
        return False

    directions = {(pr - cell[0], pc - cell[1]) for cell, (pr, pc) in parent.items()}
    has_vertical = any(dr != 0 for dr, _ in directions)
    has_horizontal = any(dc != 0 for _, dc in directions)
    if not (has_vertical and has_horizontal):
        return False

    final_colors = _compute_final_colors_from_metadata(sources, parent, source_colors)
    if len(set(final_colors.values())) < 2:
        return False

    equal_edges, different_edges = _adjacency_stats(final_colors)
    if equal_edges == 0 or different_edges == 0:
        return False

    expected = _render_expected_output(inp, row_runs, col_runs, final_colors)
    try:
        actual = main(inp)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False
    if np.array_equal(actual, inp):
        return False
    return True


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    row_sizes, col_sizes = _choose_sizes(rng)
    sources = _sample_sources(rng)
    parent, depth = _build_pointer_forest(rng, sources)
    source_colors = _assign_source_colors(rng, sources)
    inp, row_runs, col_runs = _render_input(row_sizes, col_sizes, sources, parent, source_colors, rng)

    if not _is_valid_instance(inp, row_runs, col_runs, sources, parent, depth, source_colors):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        rng = random.Random()
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid domino-pointer propagation input")
