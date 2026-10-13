import random

import numpy as np


BACKGROUND = 0


def _dominant_non_background_color(grid):
    values = grid[grid != BACKGROUND]
    if values.size == 0:
        return None
    colors, counts = np.unique(values, return_counts=True)
    return int(colors[int(np.argmax(counts))])


def _border_components(grid, border_color):
    rows, cols = grid.shape
    seen = np.zeros((rows, cols), dtype=bool)
    components = []

    for start_r in range(rows):
        for start_c in range(cols):
            if seen[start_r, start_c] or grid[start_r, start_c] != border_color:
                continue

            stack = [(start_r, start_c)]
            seen[start_r, start_c] = True
            cells = []

            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if (
                        0 <= nr < rows
                        and 0 <= nc < cols
                        and not seen[nr, nc]
                        and grid[nr, nc] == border_color
                    ):
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            components.append(cells)

    return components


def _perimeter_cells(r0, c0, r1, c1):
    cells = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        cells.append((r, c1))
    return cells


def _ray_path(rows, cols, gap, direction):
    r, c = gap
    dr, dc = direction
    path = []
    r += dr
    c += dc
    while 0 <= r < rows and 0 <= c < cols:
        path.append((r, c))
        r += dr
        c += dc
    return path


def main(grid: np.ndarray) -> np.ndarray:
    """Close each gapped rectangular frame and emit the interior seed color outward through its gap.

    Input structure:
    The grid has a black background, several same-colored 1-pixel rectangular
    frames, one black 1-pixel gap on each frame perimeter, and exactly one
    non-background, non-frame seed pixel inside each frame. Seeds are not
    adjacent to their frame, and frames are separate.

    Rule:
    For every frame, find its missing perimeter pixel, read the unique seed
    color inside the frame, repair the gap with the frame color, then draw that
    seed color from the cell just outside the repaired gap in the outward
    cardinal direction until the grid edge. Existing non-background pixels are
    preserved.

    Key insight:
    The gap gives the ray direction while the spatially separate interior seed
    gives the ray color, so each frame must be interpreted as one paired object.
    """
    out = np.array(grid, copy=True)
    border_color = _dominant_non_background_color(grid)
    if border_color is None:
        return out

    rows, cols = grid.shape
    for component in _border_components(grid, border_color):
        rs = [r for r, _ in component]
        cs = [c for _, c in component]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue

        missing = [
            (r, c)
            for r, c in _perimeter_cells(r0, c0, r1, c1)
            if grid[r, c] != border_color
        ]
        if len(missing) != 1:
            continue
        gap = missing[0]

        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        seed_locs = np.argwhere((interior != BACKGROUND) & (interior != border_color))
        if len(seed_locs) != 1:
            continue
        sr, sc = seed_locs[0]
        seed_color = int(interior[sr, sc])

        gr, gc = gap
        if gr == r0:
            direction = (-1, 0)
        elif gr == r1:
            direction = (1, 0)
        elif gc == c0:
            direction = (0, -1)
        elif gc == c1:
            direction = (0, 1)
        else:
            continue

        out[gr, gc] = border_color
        for rr, cc in _ray_path(rows, cols, gap, direction):
            if out[rr, cc] == BACKGROUND:
                out[rr, cc] = seed_color

    return out


def _candidate_cells(r0, c0, r1, c1, gap, seed):
    cells = []
    for cell in _perimeter_cells(r0, c0, r1, c1):
        if cell != gap:
            cells.append(cell)
    cells.append(seed)
    return cells


def _try_place(grid, occupied, ray_mask, seed_color):
    rows, cols = grid.shape
    border_color = 5
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for _ in range(300):
        height = random.randint(5, min(9, rows - 2))
        width = random.randint(5, min(9, cols - 2))
        r0 = random.randint(1, rows - height - 1)
        c0 = random.randint(1, cols - width - 1)
        r1 = r0 + height - 1
        c1 = c0 + width - 1
        dr, dc = random.choice(directions)

        if dr == -1:
            gap = (r0, random.randint(c0 + 1, c1 - 1))
        elif dr == 1:
            gap = (r1, random.randint(c0 + 1, c1 - 1))
        elif dc == -1:
            gap = (random.randint(r0 + 1, r1 - 1), c0)
        else:
            gap = (random.randint(r0 + 1, r1 - 1), c1)

        seed = (random.randint(r0 + 2, r1 - 2), random.randint(c0 + 2, c1 - 2))
        object_cells = _candidate_cells(r0, c0, r1, c1, gap, seed)
        path = _ray_path(rows, cols, gap, (dr, dc))

        mr0, mr1 = max(0, r0 - 1), min(rows - 1, r1 + 1)
        mc0, mc1 = max(0, c0 - 1), min(cols - 1, c1 + 1)
        if occupied[mr0 : mr1 + 1, mc0 : mc1 + 1].any():
            continue
        if ray_mask[r0 : r1 + 1, c0 : c1 + 1].any():
            continue
        if any(ray_mask[r, c] for r, c in object_cells):
            continue
        if any(occupied[r, c] or ray_mask[r, c] for r, c in path):
            continue

        for r, c in _perimeter_cells(r0, c0, r1, c1):
            if (r, c) != gap:
                grid[r, c] = border_color
                occupied[r, c] = True
        sr, sc = seed
        grid[sr, sc] = seed_color
        occupied[sr, sc] = True
        for r, c in path:
            ray_mask[r, c] = True
        return True

    return False


def generate_input() -> np.ndarray:
    seed_colors = [1, 2, 3, 4, 6, 7, 8, 9]

    for _ in range(50):
        target_count = random.randint(1, 4)
        low = min(30, 12 + target_count * 4)
        rows = random.randint(low, 30)
        cols = random.randint(low, 30)
        grid = np.zeros((rows, cols), dtype=int)
        occupied = np.zeros((rows, cols), dtype=bool)
        ray_mask = np.zeros((rows, cols), dtype=bool)
        colors = random.sample(seed_colors, target_count)

        placed = 0
        for color in colors:
            if _try_place(grid, occupied, ray_mask, color):
                placed += 1

        if placed == target_count or placed > 0:
            return grid

    grid = np.zeros((16, 16), dtype=int)
    grid[4, 4:11] = 5
    grid[10, 4:11] = 5
    grid[5:10, 4] = 5
    grid[5:10, 10] = 5
    grid[4, 7] = 0
    grid[7, 7] = 2
    return grid
