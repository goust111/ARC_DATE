import random
from collections import defaultdict, deque

import numpy as np


BG = 0
GREEN = 3
GREY = 5
ORANGE = 7
PATH_COLOR_POOL = (1, 2, 4, 6, 8, 9)
GEN_RETRIES = 400


def _spaced_positions(count: int, lo: int, hi: int, min_gap: int) -> list[int] | None:
    span = hi - lo + 1
    needed = count + (count - 1) * min_gap
    if count < 0 or span < needed:
        return None
    picks = sorted(random.sample(range(span - (count - 1) * min_gap), count))
    return [lo + pick + i * min_gap for i, pick in enumerate(picks)]


def _neighbors(shape: tuple[int, int], r: int, c: int):
    h, w = shape
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _find_start_blocks(grid: np.ndarray) -> dict[int, set[tuple[int, int]]]:
    h, w = grid.shape
    blocks: dict[int, set[tuple[int, int]]] = {}
    for r in range(h - 1):
        for c in range(w - 1):
            color = int(grid[r, c])
            if color in (BG, GREEN, GREY, ORANGE):
                continue
            if (
                int(grid[r, c + 1]) == color
                and int(grid[r + 1, c]) == color
                and int(grid[r + 1, c + 1]) == color
            ):
                coords = {(r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)}
                prev = blocks.get(color)
                assert prev is None or prev == coords, "expected one 2x2 start block per path color"
                blocks[color] = coords
    assert blocks, "expected at least one start block"
    return blocks


def _find_priority_table(
    grid: np.ndarray, path_colors: set[int]
) -> tuple[dict[int, int], set[tuple[int, int]]]:
    h, w = grid.shape
    for c in range(w - 1, -1, -1):
        rows = [r for r in range(h) if int(grid[r, c]) in path_colors]
        if len(rows) != len(path_colors):
            continue
        colors = [int(grid[r, c]) for r in rows]
        if len(set(colors)) != len(path_colors) or set(colors) != path_colors:
            continue
        if rows[-1] - rows[0] + 1 != len(rows):
            continue
        if any(c > 0 and int(grid[r, c - 1]) != BG for r in rows):
            continue
        if any(c + 1 < w and int(grid[r, c + 1]) != BG for r in rows):
            continue
        return {color: rank for rank, color in enumerate(colors)}, {(r, c) for r in rows}
    raise AssertionError("priority table not found")


def _find_lamps(
    grid: np.ndarray, path_colors: set[int], table_cells: set[tuple[int, int]]
) -> tuple[dict[int, tuple[int, int]], dict[int, tuple[int, int]]]:
    lamps: dict[int, tuple[int, int]] = {}
    endpoints: dict[int, tuple[int, int]] = {}
    for r, c in np.argwhere(grid == ORANGE):
        hits = []
        for rr, cc in _neighbors(grid.shape, int(r), int(c)):
            if (rr, cc) in table_cells:
                continue
            color = int(grid[rr, cc])
            if color in path_colors:
                hits.append((color, (rr, cc)))
        assert len(hits) == 1, "each orange lamp must touch exactly one path endpoint"
        color, endpoint = hits[0]
        assert color not in lamps, "expected one lamp per path color"
        lamps[color] = (int(r), int(c))
        endpoints[color] = endpoint
    assert set(lamps) == path_colors, "missing lamps for some paths"
    return lamps, endpoints


def _hidden_cells_for_color(
    grid: np.ndarray, visible_cells: set[tuple[int, int]], table_cells: set[tuple[int, int]]
) -> set[tuple[int, int]]:
    hidden: set[tuple[int, int]] = set()
    for r, c in visible_cells:
        for dr, dc in ((0, 1), (1, 0)):
            rr, cc = r + 2 * dr, c + 2 * dc
            mr, mc = r + dr, c + dc
            if (rr, cc) not in visible_cells:
                continue
            if not (0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1]):
                continue
            if (mr, mc) in visible_cells or (mr, mc) in table_cells:
                continue
            middle = int(grid[mr, mc])
            if middle not in (BG, ORANGE):
                hidden.add((mr, mc))
    return hidden


def _reachable(
    starts: set[tuple[int, int]], target: tuple[int, int], allowed: set[tuple[int, int]]
) -> bool:
    frontier = deque([cell for cell in starts if cell in allowed])
    seen = set(frontier)
    while frontier:
        cell = frontier.popleft()
        if cell == target:
            return True
        r, c = cell
        for nxt in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if nxt in allowed and nxt not in seen:
                seen.add(nxt)
                frontier.append(nxt)
    return False


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate each colored path from its 2x2 source block to its orange-lit terminal.

    Higher-priority colors keep control of shared crossing cells. A lower-priority
    path is broken at every such crossing and therefore fails to reach its lamp.
    Successful paths keep their color and turn their lamp green. Failed paths turn
    grey. The priority table and all 2x2 source blocks are erased.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    start_blocks = _find_start_blocks(g)
    path_colors = set(start_blocks)
    priority_rank, table_cells = _find_priority_table(g, path_colors)
    lamps, endpoints = _find_lamps(g, path_colors, table_cells)

    visible_by_color: dict[int, set[tuple[int, int]]] = {}
    full_by_color: dict[int, set[tuple[int, int]]] = {}
    for color in path_colors:
        visible = {tuple(map(int, rc)) for rc in np.argwhere(g == color)} - table_cells
        visible_by_color[color] = visible
        full_by_color[color] = visible | _hidden_cells_for_color(g, visible, table_cells)

    cell_to_colors: dict[tuple[int, int], set[int]] = defaultdict(set)
    for color, cells in full_by_color.items():
        for cell in cells:
            cell_to_colors[cell].add(color)

    success: dict[int, bool] = {}
    for color in path_colors:
        allowed = {
            cell
            for cell in full_by_color[color]
            if min(cell_to_colors[cell], key=priority_rank.get) == color
        }
        success[color] = _reachable(start_blocks[color], endpoints[color], allowed)

    out = g.copy()
    for r, c in table_cells:
        out[r, c] = BG
    for cells in start_blocks.values():
        for r, c in cells:
            out[r, c] = BG

    for color, visible in visible_by_color.items():
        paint = color if success[color] else GREY
        for r, c in visible - start_blocks[color]:
            out[r, c] = paint

    for color, lamp in lamps.items():
        out[lamp] = GREEN if success[color] else ORANGE

    return out


def _make_horizontal_path(color: int, row: int, end_c: int, height: int) -> dict:
    block_above = random.choice((True, False))
    r0 = row - 1 if block_above else row
    block = {(r0, 1), (r0, 2), (r0 + 1, 1), (r0 + 1, 2)}
    lamp_options = [("right", (row, end_c + 1))]
    if row - 1 >= 0:
        lamp_options.append(("up", (row - 1, end_c)))
    if row + 1 < height:
        lamp_options.append(("down", (row + 1, end_c)))
    _side, lamp = random.choice(lamp_options)
    cells = {(row, c) for c in range(2, end_c + 1)}
    return {
        "color": color,
        "cells": cells,
        "block": block,
        "lamp": lamp,
    }


def _make_vertical_path(color: int, col: int, end_r: int, width: int) -> dict:
    block_left = random.choice((True, False))
    c0 = col - 1 if block_left else col
    block = {(1, c0), (1, c0 + 1), (2, c0), (2, c0 + 1)}
    lamp_options = [("down", (end_r + 1, col))]
    if col - 1 >= 0:
        lamp_options.append(("left", (end_r, col - 1)))
    if col + 1 < width:
        lamp_options.append(("right", (end_r, col + 1)))
    _side, lamp = random.choice(lamp_options)
    cells = {(r, col) for r in range(2, end_r + 1)}
    return {
        "color": color,
        "cells": cells,
        "block": block,
        "lamp": lamp,
    }


def _render_grid(height: int, width: int, paths: list[dict], priority_order: list[int]) -> np.ndarray:
    grid = np.full((height, width), BG, dtype=int)

    table_col = width - 2
    table_row0 = random.randint(1, height - len(priority_order) - 1)
    for i, color in enumerate(priority_order):
        grid[table_row0 + i, table_col] = color

    priority_rank = {color: rank for rank, color in enumerate(priority_order)}
    occupants: dict[tuple[int, int], list[int]] = defaultdict(list)
    for path in paths:
        for cell in path["block"] | path["cells"]:
            occupants[cell].append(int(path["color"]))

    for cell, colors in occupants.items():
        grid[cell] = min(colors, key=priority_rank.get)

    for path in paths:
        lr, lc = path["lamp"]
        assert grid[lr, lc] == BG, "lamp placement must stay clear"
        grid[lr, lc] = ORANGE

    return grid


def _gen_attempt() -> np.ndarray | None:
    n_horizontal = random.randint(1, 3)
    n_vertical = random.randint(1, 3)
    total = n_horizontal + n_vertical

    height = random.randint(19, 26)
    width = random.randint(19, 30)

    rows = _spaced_positions(n_horizontal, 4, height - 7, 3)
    cols = _spaced_positions(n_vertical, 4, width - 7, 3)
    if rows is None or cols is None:
        return None

    end_c_lo = max(cols) + 2
    end_r_lo = max(rows) + 2
    if end_c_lo > width - 5 or end_r_lo > height - 5:
        return None

    colors = random.sample(PATH_COLOR_POOL, total)
    random.shuffle(rows)
    random.shuffle(cols)

    paths: list[dict] = []
    for color, row in zip(colors[:n_horizontal], rows):
        end_c = random.randint(end_c_lo, width - 5)
        paths.append(_make_horizontal_path(color, row, end_c, height))
    for color, col in zip(colors[n_horizontal:], cols):
        end_r = random.randint(end_r_lo, height - 5)
        paths.append(_make_vertical_path(color, col, end_r, width))

    priority_order = random.sample(colors, len(colors))
    grid = _render_grid(height, width, paths, priority_order)

    try:
        out = main(grid)
    except AssertionError:
        return None

    green_count = sum(int(out[path["lamp"]]) == GREEN for path in paths)
    if not (0 < green_count < len(paths)):
        return None
    if not np.any(out == GREY):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for task 3")
