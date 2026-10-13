from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
SEPARATOR = 5
CONTROL = 7
CORNER = 9
TOP_ROW_COLORS = (1, 2, 3, 4, 6, 8)
CORNERS = ("BL", "BR")
GEN_RETRY_LIMIT = 400


def _triangle_cells(shape: tuple[int, int], corner: str, side: int) -> list[tuple[int, int]]:
    """Return triangle cells in row-major order within rows below the separator."""
    height, width = map(int, shape)
    cells: list[tuple[int, int]] = []
    for r in range(2, height):
        dr = height - 1 - r
        reach = int(side) - int(dr)
        if reach <= 0:
            continue
        if corner == "BL":
            for c in range(min(width, reach)):
                cells.append((int(r), int(c)))
        else:
            start = max(0, width - reach)
            for c in range(start, width):
                cells.append((int(r), int(c)))
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill a corner right triangle whose side is the orange-point count, using the top-row color sequence cyclically.

    Input structure:
    - The grid has a non-black top-row color template and a full gray separator row directly below it.
    - Rows below the separator are a black work area containing exactly one maroon corner marker in the lower-left or lower-right corner.
    - The work area also contains several scattered orange singleton control points.

    Rule:
    - Read the entire top row as the cyclic color template.
    - Count the orange control points; that count is the side length of a right isosceles triangle anchored at the marked lower corner.
    - Fill the triangle's interior in row-major order with the top-row colors cycling by index, while every non-triangle work-area cell becomes black.
    - Preserve the top row and gray separator row.

    Key insight:
    - The triangle geometry comes only from counting distant orange controls, while the fill colors come only from the separate top-row sequence.
    """
    g = np.asarray(grid, dtype=int)
    height, width = map(int, g.shape)

    top_row = g[0].copy()
    side = int(np.count_nonzero(g[2:] == CONTROL))

    marked = []
    if int(g[height - 1, 0]) == CORNER:
        marked.append("BL")
    if int(g[height - 1, width - 1]) == CORNER:
        marked.append("BR")
    assert len(marked) == 1, "expected exactly one lower-corner marker"
    corner = marked[0]

    out = np.full_like(g, BACKGROUND)
    out[0] = top_row
    out[1] = g[1].copy()

    fill_cells = _triangle_cells((height, width), corner, side)
    cycle = [int(v) for v in top_row]
    for i, (r, c) in enumerate(fill_cells):
        out[r, c] = int(cycle[i % len(cycle)])
    return out


def _build_top_row(width: int) -> np.ndarray | None:
    n_colors = random.randint(2, 4)
    palette = random.sample(list(TOP_ROW_COLORS), n_colors)
    motif_len = random.randint(n_colors, min(width, n_colors + 4))

    motif: list[int] = []
    unused = set(int(c) for c in palette)
    for idx in range(motif_len):
        prev = motif[-1] if motif else None
        slots_left = motif_len - idx
        choices = [int(c) for c in palette if int(c) != prev]
        must_cover = [c for c in choices if c in unused]
        if len(unused) >= slots_left:
            choices = must_cover
        elif must_cover and random.random() < 0.7:
            choices = must_cover
        if not choices:
            return None
        color = int(random.choice(choices))
        motif.append(color)
        unused.discard(color)

    start = random.randrange(motif_len)
    row = [int(motif[(start + i) % motif_len]) for i in range(width)]
    return np.array(row, dtype=int)


def _spread_out_sample(
    candidates: list[tuple[int, int]], count: int, min_chebyshev_gap: int
) -> list[tuple[int, int]] | None:
    shuffled = candidates[:]
    random.shuffle(shuffled)
    chosen: list[tuple[int, int]] = []
    for r, c in shuffled:
        if all(
            max(abs(int(r) - int(rr)), abs(int(c) - int(cc))) >= int(min_chebyshev_gap)
            for rr, cc in chosen
        ):
            chosen.append((int(r), int(c)))
            if len(chosen) == count:
                return chosen
    return None


def _sample_controls(
    height: int, width: int, side: int, corner: str
) -> list[tuple[int, int]] | None:
    all_cells = [(r, c) for r in range(2, height) for c in range(width)]
    corner_cell = (height - 1, 0) if corner == "BL" else (height - 1, width - 1)
    triangle = set(_triangle_cells((height, width), corner, side))

    non_triangle = [
        (r, c)
        for r, c in all_cells
        if (r, c) != corner_cell and (r, c) not in triangle
    ]
    all_work = [(r, c) for r, c in all_cells if (r, c) != corner_cell]

    prefer_outside = random.random() < 0.65
    primary = non_triangle if prefer_outside and len(non_triangle) >= side else all_work
    secondary = all_work if primary is non_triangle else []

    for gap in (2, 1):
        sample = _spread_out_sample(primary, side, gap)
        if sample is not None:
            return sample
        if secondary:
            sample = _spread_out_sample(secondary, side, gap)
            if sample is not None:
                return sample
    return None


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(8, 15)
    width = random.randint(8, 15)
    work_height = height - 2
    max_side = min(10, work_height, width)
    if max_side < 3:
        return None

    corner = random.choice(CORNERS)
    side = random.randint(3, max_side)
    top_row = _build_top_row(width)
    if top_row is None:
        return None

    controls = _sample_controls(height, width, side, corner)
    if controls is None or len(controls) != side:
        return None

    grid = np.full((height, width), BACKGROUND, dtype=int)
    grid[0] = top_row
    grid[1] = SEPARATOR
    if corner == "BL":
        grid[height - 1, 0] = CORNER
    else:
        grid[height - 1, width - 1] = CORNER
    for r, c in controls:
        grid[r, c] = CONTROL

    if len(set(map(int, top_row.tolist()))) < 2:
        return None
    if np.count_nonzero(grid == CONTROL) != side:
        return None
    if np.count_nonzero(grid == CORNER) != 1:
        return None
    if max(grid.shape) > 30:
        return None

    out = main(grid)
    if out.shape != grid.shape:
        return None
    if not np.array_equal(out[0], top_row):
        return None
    if not np.all(out[1] == SEPARATOR):
        return None
    if not np.all(np.isin(out[2:], np.array([BACKGROUND, *TOP_ROW_COLORS], dtype=int))):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
