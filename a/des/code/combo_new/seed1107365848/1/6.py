from __future__ import annotations

import random
from collections import Counter

import numpy as np


GREY = 5
CELL_COLORS = (1, 2, 3, 4)  # blue, red, green, yellow
CELL_SIZE = 3
LINE_SIZE = 1
GRID_CELL_MIN = 4
GRID_CELL_MAX = 7
GEN_RETRIES = 300


def _runs(mask: np.ndarray, value: bool) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i, flag in enumerate(mask.tolist()):
        if bool(flag) == value:
            if start is None:
                start = i
        elif start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def _extract_cells(grid: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep_rows = np.all(g == GREY, axis=1)
    sep_cols = np.all(g == GREY, axis=0)
    row_blocks = _runs(sep_rows, False)
    col_blocks = _runs(sep_cols, False)

    assert row_blocks and col_blocks, "expected at least one cell block"
    assert all(r1 - r0 + 1 == CELL_SIZE for r0, r1 in row_blocks), "cell row height must be 3"
    assert all(c1 - c0 + 1 == CELL_SIZE for c0, c1 in col_blocks), "cell column width must be 3"

    for prev, curr in zip(row_blocks, row_blocks[1:]):
        assert curr[0] - prev[1] - 1 == LINE_SIZE, "separator rows must be single-pixel grey lines"
    for prev, curr in zip(col_blocks, col_blocks[1:]):
        assert curr[0] - prev[1] - 1 == LINE_SIZE, "separator columns must be single-pixel grey lines"

    assert row_blocks[0][0] in (0, LINE_SIZE), "unexpected top margin"
    assert col_blocks[0][0] in (0, LINE_SIZE), "unexpected left margin"
    assert g.shape[0] - 1 - row_blocks[-1][1] in (0, LINE_SIZE), "unexpected bottom margin"
    assert g.shape[1] - 1 - col_blocks[-1][1] in (0, LINE_SIZE), "unexpected right margin"

    cells = np.empty((len(row_blocks), len(col_blocks)), dtype=int)
    for rr, (r0, r1) in enumerate(row_blocks):
        for cc, (c0, c1) in enumerate(col_blocks):
            patch = g[r0 : r1 + 1, c0 : c1 + 1]
            colors = np.unique(patch)
            assert len(colors) == 1, "each 3x3 cell must be monochrome"
            color = int(colors[0])
            assert color in CELL_COLORS, "unexpected cell color"
            cells[rr, cc] = color

    return cells, row_blocks, col_blocks


def _voted_color(cells: np.ndarray, r: int, c: int) -> int:
    neighbors: list[int] = []
    if r > 0:
        neighbors.append(int(cells[r - 1, c]))
    if r + 1 < cells.shape[0]:
        neighbors.append(int(cells[r + 1, c]))
    if c > 0:
        neighbors.append(int(cells[r, c - 1]))
    if c + 1 < cells.shape[1]:
        neighbors.append(int(cells[r, c + 1]))

    counts = Counter(neighbors)
    best = max(counts.values())
    winners = [color for color, count in counts.items() if count == best]
    return winners[0] if len(winners) == 1 else int(cells[r, c])


def _transform_cells(cells: np.ndarray) -> np.ndarray:
    out = np.array(cells, copy=True)
    for r in range(cells.shape[0]):
        for c in range(cells.shape[1]):
            out[r, c] = _voted_color(cells, r, c)
    return out


def _render_grid(cells: np.ndarray) -> np.ndarray:
    rows, cols = map(int, cells.shape)
    grid = np.full((rows * (CELL_SIZE + LINE_SIZE) + LINE_SIZE, cols * (CELL_SIZE + LINE_SIZE) + LINE_SIZE), GREY, dtype=int)
    for r in range(rows):
        for c in range(cols):
            r0 = LINE_SIZE + r * (CELL_SIZE + LINE_SIZE)
            c0 = LINE_SIZE + c * (CELL_SIZE + LINE_SIZE)
            grid[r0 : r0 + CELL_SIZE, c0 : c0 + CELL_SIZE] = int(cells[r, c])
    return grid


def _branch_stats(cells: np.ndarray) -> tuple[int, int]:
    changed = 0
    ties = 0
    transformed = _transform_cells(cells)
    for r in range(cells.shape[0]):
        for c in range(cells.shape[1]):
            neighbors: list[int] = []
            if r > 0:
                neighbors.append(int(cells[r - 1, c]))
            if r + 1 < cells.shape[0]:
                neighbors.append(int(cells[r + 1, c]))
            if c > 0:
                neighbors.append(int(cells[r, c - 1]))
            if c + 1 < cells.shape[1]:
                neighbors.append(int(cells[r, c + 1]))
            counts = Counter(neighbors)
            best = max(counts.values())
            if sum(count == best for count in counts.values()) > 1:
                ties += 1
            if int(transformed[r, c]) != int(cells[r, c]):
                changed += 1
    return changed, ties


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each lattice cell by the unique majority color among its orthogonal neighboring cells.

    Input structure:
    - The input is a rectangular lattice split by single-pixel grey separator rows and columns.
    - Each lattice cell is a 3x3 monochrome block colored red, blue, green, or yellow.
    - The separator lines may also form an outer border around the cell matrix.

    Rule:
    - Treat every 3x3 block between grey lines as one logical cell and read its single fill color.
    - For each logical cell, inspect the colors of its orthogonal neighboring cells (up, down, left, right when present).
    - If one neighbor color is the unique most frequent color, repaint the whole 3x3 cell to that color; if the neighbor vote is tied, keep the cell's original color.
    - Keep all grey separator lines unchanged.

    Key insight:
    - A cell's output color is determined by a local vote from surrounding cells rather than by the cell itself, with ties resolving to "no change".
    """
    g = np.asarray(grid, dtype=int)
    cells, row_blocks, col_blocks = _extract_cells(g)
    out_cells = _transform_cells(cells)

    out = np.array(g, copy=True)
    for rr, (r0, _r1) in enumerate(row_blocks):
        for cc, (c0, _c1) in enumerate(col_blocks):
            out[r0 : r0 + CELL_SIZE, c0 : c0 + CELL_SIZE] = int(out_cells[rr, cc])
    return out


def _make_patch_pattern(rows: int, cols: int, rng: random.Random) -> np.ndarray:
    cells = np.empty((rows, cols), dtype=int)
    row_cuts = sorted(rng.sample(range(1, rows), rng.randint(1, min(2, rows - 1))))
    col_cuts = sorted(rng.sample(range(1, cols), rng.randint(1, min(2, cols - 1))))
    row_edges = [0] + row_cuts + [rows]
    col_edges = [0] + col_cuts + [cols]
    palette = list(CELL_COLORS)
    rng.shuffle(palette)
    color_idx = 0
    for rs, re in zip(row_edges, row_edges[1:]):
        for cs, ce in zip(col_edges, col_edges[1:]):
            cells[rs:re, cs:ce] = palette[color_idx % len(palette)]
            color_idx += 1
    return cells


def _make_stripe_pattern(rows: int, cols: int, rng: random.Random) -> np.ndarray:
    cells = np.empty((rows, cols), dtype=int)
    palette = list(CELL_COLORS)
    rng.shuffle(palette)
    mode = rng.choice(("row", "col", "checker", "diagonal"))
    if mode == "row":
        offset = rng.randrange(len(palette))
        for r in range(rows):
            cells[r, :] = palette[(r + offset) % len(palette)]
    elif mode == "col":
        offset = rng.randrange(len(palette))
        for c in range(cols):
            cells[:, c] = palette[(c + offset) % len(palette)]
    elif mode == "checker":
        offset = rng.randrange(len(palette))
        for r in range(rows):
            for c in range(cols):
                cells[r, c] = palette[(r + c + offset) % len(palette)]
    else:
        offset = rng.randrange(len(palette))
        for r in range(rows):
            for c in range(cols):
                cells[r, c] = palette[(2 * r + c + offset) % len(palette)]
    return cells


def _make_voronoi_pattern(rows: int, cols: int, rng: random.Random) -> np.ndarray:
    cells = np.empty((rows, cols), dtype=int)
    n_seeds = rng.randint(3, min(6, rows * cols))
    positions = rng.sample([(r, c) for r in range(rows) for c in range(cols)], n_seeds)
    palette = list(CELL_COLORS)
    rng.shuffle(palette)
    seed_colors = [palette[i % len(palette)] for i in range(n_seeds)]
    for r in range(rows):
        for c in range(cols):
            best_dist = min(abs(r - sr) + abs(c - sc) for sr, sc in positions)
            nearest = [i for i, (sr, sc) in enumerate(positions) if abs(r - sr) + abs(c - sc) == best_dist]
            chosen = rng.choice(nearest)
            cells[r, c] = seed_colors[chosen]
    return cells


def _base_pattern(rows: int, cols: int, rng: random.Random) -> np.ndarray:
    maker = rng.choice((_make_patch_pattern, _make_stripe_pattern, _make_voronoi_pattern))
    cells = maker(rows, cols, rng)
    mutations = rng.randint(max(2, rows * cols // 8), max(3, rows * cols // 4))
    for _ in range(mutations):
        r = rng.randrange(rows)
        c = rng.randrange(cols)
        current = int(cells[r, c])
        cells[r, c] = rng.choice([color for color in CELL_COLORS if color != current])
    return cells


def _force_change_example(cells: np.ndarray, rng: random.Random, reserved: set[tuple[int, int]]) -> None:
    rows, cols = map(int, cells.shape)
    candidates = [
        (r, c)
        for r in range(1, rows - 1)
        for c in range(1, cols - 1)
        if (r, c) not in reserved
        and all(nei not in reserved for nei in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)))
    ]
    center = rng.choice(candidates)
    r, c = center
    reserved.add(center)

    palette = list(CELL_COLORS)
    rng.shuffle(palette)
    majority, minority, center_color = palette[:3]
    cells[r, c] = center_color

    neighbors = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
    rng.shuffle(neighbors)
    for nr, nc in neighbors[:3]:
        cells[nr, nc] = majority
        reserved.add((nr, nc))
    nr, nc = neighbors[3]
    cells[nr, nc] = minority
    reserved.add((nr, nc))


def _force_tie_example(cells: np.ndarray, rng: random.Random, reserved: set[tuple[int, int]]) -> None:
    rows, cols = map(int, cells.shape)
    corners = [
        ((0, 0), [(0, 1), (1, 0)]),
        ((0, cols - 1), [(0, cols - 2), (1, cols - 1)]),
        ((rows - 1, 0), [(rows - 2, 0), (rows - 1, 1)]),
        ((rows - 1, cols - 1), [(rows - 2, cols - 1), (rows - 1, cols - 2)]),
    ]
    choices = [item for item in corners if item[0] not in reserved and all(nei not in reserved for nei in item[1])]
    corner, neighbors = rng.choice(choices)

    palette = list(CELL_COLORS)
    rng.shuffle(palette)
    cells[corner] = palette[0]
    cells[neighbors[0]] = palette[1]
    cells[neighbors[1]] = palette[2]
    reserved.add(corner)
    reserved.update(neighbors)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    rows = rng.randint(GRID_CELL_MIN, GRID_CELL_MAX)
    cols = rng.randint(GRID_CELL_MIN, GRID_CELL_MAX)
    cells = _base_pattern(rows, cols, rng)

    reserved: set[tuple[int, int]] = set()
    _force_tie_example(cells, rng, reserved)
    _force_change_example(cells, rng, reserved)

    if len({int(color) for color in cells.flatten()}) < 3:
        return None

    changed, ties = _branch_stats(cells)
    if changed < 2 or ties < 2:
        return None

    grid = _render_grid(cells)
    expected = _render_grid(_transform_cells(cells))

    try:
        actual = main(grid)
    except Exception:
        return None

    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
