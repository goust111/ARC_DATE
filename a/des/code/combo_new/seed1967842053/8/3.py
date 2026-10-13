import random

import numpy as np


BLACK = 0
GREY = 5
COLORS = [1, 2, 3, 4, 6, 7, 8, 9]


def _solid_grey_rows(grid):
    return [r for r in range(grid.shape[0]) if np.all(grid[r, :] == GREY)]


def _solid_grey_cols(grid):
    return [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]


def _interior_spans(lines, limit):
    boundaries = [-1] + sorted(lines) + [limit]
    return [(a + 1, b) for a, b in zip(boundaries, boundaries[1:]) if b - a > 1]


def main(grid: np.ndarray) -> np.ndarray:
    """Expand each colored seed into quadrant-coded diagonal striping inside its cell.

    Input structure:
    The grid has a black background and one-pixel grey grid lines that divide it
    into equal square cells. Some cell interiors contain exactly one non-black,
    non-grey colored seed pixel; empty cells contain only black pixels.

    Rule:
    For each seeded cell, use the seed color for the drawn pattern and compare
    the seed position with the center pixel of that cell. A seed in the upper
    left quadrant fills pixels whose row offset minus column offset is even. A
    seed in the lower right quadrant fills the same main-diagonal family with
    odd parity. A seed in the upper right quadrant fills pixels whose row offset
    plus column offset is even. A seed in the lower left quadrant fills that
    anti-diagonal family with odd parity. Grey grid lines and empty cells stay
    unchanged.

    Key insight:
    The seed location encodes both the diagonal family and the parity phase of
    the stripe pattern to draw in its own cell.
    """
    out = np.array(grid, copy=True)
    row_spans = _interior_spans(_solid_grey_rows(grid), grid.shape[0])
    col_spans = _interior_spans(_solid_grey_cols(grid), grid.shape[1])

    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            cell = grid[r0:r1, c0:c1]
            seed_mask = (cell != BLACK) & (cell != GREY)
            ys, xs = np.where(seed_mask)
            if len(ys) != 1:
                continue

            sr, sc = int(ys[0]), int(xs[0])
            color = int(cell[sr, sc])
            center_r = (r1 - r0) // 2
            center_c = (c1 - c0) // 2

            if sr < center_r and sc < center_c:
                use_difference = True
                parity = 0
            elif sr > center_r and sc > center_c:
                use_difference = True
                parity = 1
            elif sr < center_r and sc > center_c:
                use_difference = False
                parity = 0
            elif sr > center_r and sc < center_c:
                use_difference = False
                parity = 1
            else:
                continue

            height, width = cell.shape
            rr, cc = np.indices((height, width))
            if use_difference:
                stripe_mask = ((rr - cc) % 2) == parity
            else:
                stripe_mask = ((rr + cc) % 2) == parity

            target = out[r0:r1, c0:c1]
            target[:, :] = BLACK
            target[stripe_mask] = color

    return out


def generate_input() -> np.ndarray:
    cell_size = random.choice([3, 5, 7, 9])
    max_cells = (30 - 1) // (cell_size + 1)
    cell_rows = random.randint(2, max_cells)
    cell_cols = random.randint(2, max_cells)

    height = cell_rows * (cell_size + 1) + 1
    width = cell_cols * (cell_size + 1) + 1
    grid = np.zeros((height, width), dtype=int)

    for r in range(0, height, cell_size + 1):
        grid[r, :] = GREY
    for c in range(0, width, cell_size + 1):
        grid[:, c] = GREY

    cells = [(r, c) for r in range(cell_rows) for c in range(cell_cols)]
    random.shuffle(cells)
    total_cells = len(cells)
    min_seeds = min(4, total_cells)
    max_seeds = min(total_cells, max(4, total_cells // 2))
    seed_count = random.randint(min_seeds, max_seeds)

    quadrants = ["ul", "ur", "dr", "dl"] * ((seed_count + 3) // 4)
    quadrants = quadrants[:seed_count]
    random.shuffle(quadrants)

    center = cell_size // 2
    for (cell_r, cell_c), quadrant in zip(cells[:seed_count], quadrants):
        if quadrant in ("ul", "ur"):
            local_r = random.randrange(0, center)
        else:
            local_r = random.randrange(center + 1, cell_size)

        if quadrant in ("ul", "dl"):
            local_c = random.randrange(0, center)
        else:
            local_c = random.randrange(center + 1, cell_size)

        abs_r = cell_r * (cell_size + 1) + 1 + local_r
        abs_c = cell_c * (cell_size + 1) + 1 + local_c
        grid[abs_r, abs_c] = random.choice(COLORS)

    return grid
