from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_RETRIES = 400


def _neighbors(cell: tuple[int, int], rows: int, cols: int) -> list[tuple[int, int]]:
    r, c = cell
    out: list[tuple[int, int]] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < rows:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < cols:
        out.append((r, c + 1))
    return out


def _component_count(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    count = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            count += 1
            stack = [(r, c)]
            seen[r, c] = True
            while stack:
                rr, cc = stack.pop()
                for dr, dc in NEIGHBORS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
    return count


def _fragment_counts(grid: np.ndarray) -> dict[int, int]:
    counts: dict[int, int] = {}
    for color in np.unique(grid):
        color = int(color)
        if color in (BLACK, GREY):
            continue
        counts[color] = _component_count(grid == color)
    return counts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort colors by how many 4-connected fragments remain after grey cut lines split them,
    then encode that order as 2x2 color blocks along the bottom of the grid.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert h >= 2, "expected two reserved output rows"

    counts = _fragment_counts(g)
    assert counts, "expected at least one foreground color"

    ordered = sorted(counts, key=lambda color: counts[color])
    assert len({counts[color] for color in ordered}) == len(ordered), "fragment counts must be unique"
    assert 3 * len(ordered) - 1 <= w, "bottom code does not fit the grid width"

    out = g.copy()
    out[h - 2 :, :] = BLACK
    for i, color in enumerate(ordered):
        c0 = 3 * i
        out[h - 2 : h, c0 : c0 + 2] = int(color)
    return out


def _sample_partition(mode: str) -> tuple[list[int], list[int]] | None:
    if mode == "vertical":
        row_count = 1
        col_count = random.randint(6, 8)
    elif mode == "horizontal":
        row_count = random.randint(6, 8)
        col_count = 1
    else:
        row_count = random.randint(2, 4)
        col_count = random.randint(3, 6)

    if row_count == 1:
        row_heights = [random.randint(10, 20)]
    else:
        row_heights = [random.randint(2, 5) for _ in range(row_count)]

    if col_count == 1:
        col_widths = [random.randint(12, 24)]
    else:
        col_widths = [random.randint(2, 4) for _ in range(col_count)]

    active_h = sum(row_heights) + row_count - 1
    width = sum(col_widths) + col_count - 1
    height = active_h + 2
    if not (12 <= height <= 30 and 8 <= width <= 30):
        return None
    return row_heights, col_widths


def _cell_bounds(lengths: list[int]) -> tuple[list[tuple[int, int]], list[int]]:
    bounds: list[tuple[int, int]] = []
    lines: list[int] = []
    cursor = 0
    for i, size in enumerate(lengths):
        start = cursor
        end = start + size - 1
        bounds.append((start, end))
        cursor = end + 1
        if i + 1 < len(lengths):
            lines.append(cursor)
            cursor += 1
    return bounds, lines


def _sample_counts(total_cells: int, max_colors_by_width: int) -> tuple[int, list[int]] | None:
    max_colors_by_cells = 0
    while (max_colors_by_cells + 1) * (max_colors_by_cells + 2) // 2 <= total_cells:
        max_colors_by_cells += 1
    max_colors = min(5, max_colors_by_width, max_colors_by_cells)
    if max_colors < 3:
        return None

    n_colors = random.randint(3, max_colors)
    choices = list(range(1, min(total_cells, 8) + 1))
    slack = 1 if total_cells >= 8 else 0
    for _ in range(200):
        counts = sorted(random.sample(choices, n_colors))
        if sum(counts) <= total_cells - slack:
            return n_colors, counts

    counts = list(range(1, n_colors + 1))
    if sum(counts) <= total_cells:
        return n_colors, counts
    return None


def _available_components(
    available: set[tuple[int, int]], rows: int, cols: int
) -> list[set[tuple[int, int]]]:
    unseen = set(available)
    components: list[set[tuple[int, int]]] = []
    while unseen:
        start = unseen.pop()
        comp = {start}
        stack = [start]
        while stack:
            cell = stack.pop()
            for nxt in _neighbors(cell, rows, cols):
                if nxt in unseen:
                    unseen.remove(nxt)
                    comp.add(nxt)
                    stack.append(nxt)
        components.append(comp)
    return components


def _grow_connected_subset(
    component: set[tuple[int, int]], size: int, rows: int, cols: int
) -> set[tuple[int, int]] | None:
    cells = list(component)
    for _ in range(80):
        start = random.choice(cells)
        chosen = {start}
        frontier = {nxt for nxt in _neighbors(start, rows, cols) if nxt in component}
        while frontier and len(chosen) < size:
            cell = random.choice(tuple(frontier))
            chosen.add(cell)
            frontier.discard(cell)
            for nxt in _neighbors(cell, rows, cols):
                if nxt in component and nxt not in chosen:
                    frontier.add(nxt)
        if len(chosen) == size:
            return chosen
    return None


def _place_shapes(
    rows: int, cols: int, counts: list[int]
) -> list[set[tuple[int, int]]] | None:
    available = {(r, c) for r in range(rows) for c in range(cols)}
    placements: dict[int, set[tuple[int, int]]] = {}

    for count in sorted(counts, reverse=True):
        candidates = [comp for comp in _available_components(available, rows, cols) if len(comp) >= count]
        random.shuffle(candidates)
        chosen_cells = None
        for comp in candidates:
            chosen_cells = _grow_connected_subset(comp, count, rows, cols)
            if chosen_cells is not None:
                break
        if chosen_cells is None:
            return None
        placements[count] = chosen_cells
        available -= chosen_cells

    return [placements[count] for count in counts]


def _build_grid(
    row_heights: list[int],
    col_widths: list[int],
    colors: list[int],
    shapes: list[set[tuple[int, int]]],
) -> np.ndarray:
    row_bounds, line_rows = _cell_bounds(row_heights)
    col_bounds, line_cols = _cell_bounds(col_widths)
    active_h = row_bounds[-1][1] + 1
    width = col_bounds[-1][1] + 1
    grid = np.full((active_h + 2, width), BLACK, dtype=int)

    if line_rows:
        grid[np.asarray(line_rows), :] = GREY
    if line_cols:
        grid[:active_h, np.asarray(line_cols)] = GREY

    for color, cells in zip(colors, shapes):
        for rr, cc in cells:
            r0, r1 = row_bounds[rr]
            c0, c1 = col_bounds[cc]
            grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)

    return grid


def _is_valid_scene(grid: np.ndarray, expected_counts: dict[int, int]) -> bool:
    if not np.all(grid[-2:, :] == BLACK):
        return False

    counts = _fragment_counts(grid)
    if counts != expected_counts:
        return False
    if len(set(counts.values())) != len(counts):
        return False

    h, w = grid.shape
    if 3 * len(counts) - 1 > w:
        return False
    if not np.any(grid[:-2, :] == GREY):
        return False

    try:
        out = main(grid)
    except AssertionError:
        return False

    if not np.array_equal(out[:-2, :], grid[:-2, :]):
        return False
    ordered = [color for color, _count in sorted(counts.items(), key=lambda item: item[1])]
    expected = grid.copy()
    expected[-2:, :] = BLACK
    for i, color in enumerate(ordered):
        expected[-2:, 3 * i : 3 * i + 2] = int(color)
    return np.array_equal(out, expected)


def _gen_attempt() -> np.ndarray | None:
    mode = random.choice(("vertical", "horizontal", "mixed"))
    sampled = _sample_partition(mode)
    if sampled is None:
        return None
    row_heights, col_widths = sampled
    total_cells = len(row_heights) * len(col_widths)
    width = sum(col_widths) + len(col_widths) - 1

    count_info = _sample_counts(total_cells, (width + 1) // 3)
    if count_info is None:
        return None
    n_colors, counts = count_info
    shapes = _place_shapes(len(row_heights), len(col_widths), counts)
    if shapes is None:
        return None

    colors = random.sample(PALETTE, n_colors)
    expected_counts = {color: count for color, count in zip(colors, counts)}
    grid = _build_grid(row_heights, col_widths, colors, shapes)
    if _is_valid_scene(grid, expected_counts):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
