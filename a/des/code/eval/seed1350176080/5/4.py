from __future__ import annotations

import random

import numpy as np


BG = 0
GRID = 8
MARKER = 9
ARROW_COLORS = (1, 2, 3, 4)

UP, DOWN, LEFT, RIGHT = range(4)
DIR_VECS = {
    UP: (-1, 0),
    DOWN: (1, 0),
    LEFT: (0, -1),
    RIGHT: (0, 1),
}

TL, TR, BL, BR = range(4)

ARROW_PATTERNS = {
    UP: ((0, 1), (1, 0), (1, 2)),
    DOWN: ((0, 0), (0, 2), (1, 1)),
    LEFT: ((0, 1), (1, 0), (2, 1)),
    RIGHT: ((0, 0), (1, 1), (2, 0)),
}

ARROW_BBOXES = {
    UP: (2, 3),
    DOWN: (2, 3),
    LEFT: (3, 2),
    RIGHT: (3, 2),
}

NORMALIZED_ARROW_PATTERNS = {
    frozenset(ARROW_PATTERNS[UP]): UP,
    frozenset(ARROW_PATTERNS[DOWN]): DOWN,
    frozenset(ARROW_PATTERNS[LEFT]): LEFT,
    frozenset(ARROW_PATTERNS[RIGHT]): RIGHT,
}

CORNER_OFFSETS = {
    TL: lambda n: (0, 0),
    TR: lambda n: (0, n - 1),
    BL: lambda n: (n - 1, 0),
    BR: lambda n: (n - 1, n - 1),
}

GEN_RETRIES = 300


def _mode(values: np.ndarray) -> int:
    vals, counts = np.unique(values, return_counts=True)
    return int(vals[np.argmax(counts)])


def _infer_grid_geometry(g: np.ndarray) -> tuple[int, int, list[int], list[int], int]:
    h, w = g.shape
    rows_by_color: dict[int, list[int]] = {}
    cols_by_color: dict[int, list[int]] = {}

    for r in range(h):
        if np.all(g[r, :] == g[r, 0]):
            rows_by_color.setdefault(int(g[r, 0]), []).append(r)
    for c in range(w):
        if np.all(g[:, c] == g[0, c]):
            cols_by_color.setdefault(int(g[0, c]), []).append(c)

    candidates = []
    for color, rows in rows_by_color.items():
        cols = cols_by_color.get(color, [])
        if len(rows) >= 2 and len(cols) >= 2:
            candidates.append((color, rows, cols))

    assert len(candidates) == 1, "expected exactly one grid-line color"
    grid_color, row_lines, col_lines = candidates[0]

    assert row_lines[0] == 0 and row_lines[-1] == h - 1
    assert col_lines[0] == 0 and col_lines[-1] == w - 1

    row_steps = np.diff(row_lines)
    col_steps = np.diff(col_lines)
    assert row_steps.size > 0 and col_steps.size > 0
    assert np.all(row_steps == row_steps[0])
    assert np.all(col_steps == col_steps[0])
    assert int(row_steps[0]) == int(col_steps[0])

    cell_size = int(row_steps[0] - 1)
    assert cell_size >= 3

    non_grid = g[g != grid_color]
    assert non_grid.size > 0
    bg = _mode(non_grid)
    return int(bg), int(grid_color), list(map(int, row_lines)), list(map(int, col_lines)), cell_size


def _corner_from_point(point: tuple[int, int], size: int) -> int:
    mapping = {
        (0, 0): TL,
        (0, size - 1): TR,
        (size - 1, 0): BL,
        (size - 1, size - 1): BR,
    }
    assert point in mapping, "marker must sit on a cell corner"
    return mapping[point]


def _decode_arrow(coords: np.ndarray) -> int:
    coords = np.asarray(coords, dtype=int)
    assert coords.shape == (3, 2)
    mins = coords.min(axis=0)
    norm = frozenset((int(r - mins[0]), int(c - mins[1])) for r, c in coords)
    assert norm in NORMALIZED_ARROW_PATTERNS, "invalid arrow sprite"
    return NORMALIZED_ARROW_PATTERNS[norm]


def _parse_cell(cell: np.ndarray, bg: int) -> tuple[int, int, int]:
    n = int(cell.shape[0])
    assert cell.shape == (n, n)

    fg = cell[cell != bg]
    assert fg.size == 4, "each cell should contain exactly 4 non-background pixels"
    colors, counts = np.unique(fg, return_counts=True)
    assert len(colors) == 2

    arrow_color = None
    marker_color = None
    for color, count in zip(colors, counts):
        if int(count) == 3:
            arrow_color = int(color)
        elif int(count) == 1:
            marker_color = int(color)
    assert arrow_color is not None and marker_color is not None

    marker_pos = tuple(map(int, np.argwhere(cell == marker_color)[0]))
    corner = _corner_from_point(marker_pos, n)
    direction = _decode_arrow(np.argwhere(cell == arrow_color))
    return arrow_color, direction, corner


def _decode_grid(grid: np.ndarray) -> tuple[int, int, list[int], list[int], int, np.ndarray, np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg, grid_color, row_lines, col_lines, cell_size = _infer_grid_geometry(g)
    nr = len(row_lines) - 1
    nc = len(col_lines) - 1

    colors = np.zeros((nr, nc), dtype=int)
    directions = np.zeros((nr, nc), dtype=int)
    corners = np.zeros((nr, nc), dtype=int)

    for r in range(nr):
        for c in range(nc):
            r0 = row_lines[r] + 1
            c0 = col_lines[c] + 1
            cell = g[r0 : r0 + cell_size, c0 : c0 + cell_size]
            color, direction, corner = _parse_cell(cell, bg)
            colors[r, c] = color
            directions[r, c] = direction
            corners[r, c] = corner

    return bg, grid_color, row_lines, col_lines, cell_size, colors, directions, corners


def _same_color_visible(colors: np.ndarray, r: int, c: int, direction: int) -> int:
    dr, dc = DIR_VECS[direction]
    nr, nc = colors.shape
    color = int(colors[r, c])
    seen = 0
    rr = r + dr
    cc = c + dc
    while 0 <= rr < nr and 0 <= cc < nc:
        if int(colors[rr, cc]) != color:
            break
        seen += 1
        rr += dr
        cc += dc
    return seen


def _fill_patch(size: int, corner: int, layers: int, color: int, bg: int) -> np.ndarray:
    rr, cc = np.indices((size, size))
    ar, ac = CORNER_OFFSETS[corner](size)
    dist = np.maximum(np.abs(rr - ar), np.abs(cc - ac))
    patch = np.full((size, size), bg, dtype=int)
    patch[dist < int(layers)] = int(color)
    return patch


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each cell's arrow and corner marker by a corner-anchored fill size
    determined by how many same-colored arrows are visible straight ahead.
    """
    g = np.asarray(grid, dtype=int)
    bg, grid_color, row_lines, col_lines, cell_size, colors, directions, corners = _decode_grid(g)
    nr, nc = colors.shape

    out = np.full_like(g, bg)
    out[row_lines, :] = grid_color
    out[:, col_lines] = grid_color

    for r in range(nr):
        for c in range(nc):
            layers = 1 + _same_color_visible(colors, r, c, int(directions[r, c]))
            patch = _fill_patch(cell_size, int(corners[r, c]), layers, int(colors[r, c]), bg)
            r0 = row_lines[r] + 1
            c0 = col_lines[c] + 1
            out[r0 : r0 + cell_size, c0 : c0 + cell_size] = patch

    return out


def _corner_coord(corner: int, size: int) -> tuple[int, int]:
    return CORNER_OFFSETS[corner](size)


def _valid_arrow_origins(size: int, direction: int, corner: int) -> list[tuple[int, int]]:
    h, w = ARROW_BBOXES[direction]
    marker = _corner_coord(corner, size)
    valid: list[tuple[int, int]] = []
    interior: list[tuple[int, int]] = []

    for r0 in range(size - h + 1):
        for c0 in range(size - w + 1):
            coords = [(r0 + dr, c0 + dc) for dr, dc in ARROW_PATTERNS[direction]]
            if marker in coords:
                continue
            valid.append((r0, c0))
            if all(0 < r < size - 1 and 0 < c < size - 1 for r, c in coords):
                interior.append((r0, c0))

    return interior if interior else valid


def _row_run_colors(nr: int, nc: int, palette: tuple[int, ...]) -> np.ndarray:
    grid = np.zeros((nr, nc), dtype=int)
    for r in range(nr):
        c = 0
        current = random.choice(palette)
        while c < nc:
            run = random.randint(1, min(3, nc - c))
            grid[r, c : c + run] = current
            c += run
            if c < nc:
                choices = [color for color in palette if color != current]
                current = random.choice(choices)
    if nr >= 2 and random.random() < 0.6:
        for r in range(1, nr):
            if random.random() < 0.35:
                grid[r, :] = grid[r - 1, :]
                flip_c = random.randrange(nc)
                choices = [color for color in palette if color != int(grid[r, flip_c])]
                if choices:
                    grid[r, flip_c] = random.choice(choices)
    return grid


def _col_run_colors(nr: int, nc: int, palette: tuple[int, ...]) -> np.ndarray:
    return _row_run_colors(nc, nr, palette).T


def _blob_colors(nr: int, nc: int, palette: tuple[int, ...]) -> np.ndarray:
    grid = np.zeros((nr, nc), dtype=int)
    copy_bias = random.uniform(0.45, 0.8)
    for r in range(nr):
        for c in range(nc):
            neighbors = []
            if r > 0:
                neighbors.append(int(grid[r - 1, c]))
            if c > 0:
                neighbors.append(int(grid[r, c - 1]))
            if neighbors and random.random() < copy_bias:
                grid[r, c] = random.choice(neighbors)
            else:
                grid[r, c] = random.choice(palette)
    return grid


def _make_color_grid(nr: int, nc: int, palette: tuple[int, ...]) -> np.ndarray:
    mode = random.choice(("row_runs", "col_runs", "blobs"))
    if mode == "row_runs":
        grid = _row_run_colors(nr, nc, palette)
    elif mode == "col_runs":
        grid = _col_run_colors(nr, nc, palette)
    else:
        grid = _blob_colors(nr, nc, palette)

    cells = [(r, c) for r in range(nr) for c in range(nc)]
    random.shuffle(cells)
    for color, (r, c) in zip(palette, cells):
        grid[r, c] = color
    return grid


def _all_direction_counts(colors: np.ndarray) -> np.ndarray:
    nr, nc = colors.shape
    counts = np.zeros((nr, nc, 4), dtype=int)
    for r in range(nr):
        for c in range(nc):
            for direction in range(4):
                counts[r, c, direction] = _same_color_visible(colors, r, c, direction)
    return counts


def _assign_directions(option_counts: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    nr, nc, _ = option_counts.shape
    total = nr * nc
    cells = [(r, c) for r in range(nr) for c in range(nc)]
    random.shuffle(cells)

    positive_capable = [(r, c) for r, c in cells if np.any(option_counts[r, c] > 0)]
    if not positive_capable:
        return None

    low = max(1, total // 3)
    high = min(len(positive_capable), max(1, (2 * total) // 3))
    if low > high:
        return None
    target_positive = random.randint(low, high)
    positive_targets = set(positive_capable[:target_positive])

    directions = np.zeros((nr, nc), dtype=int)
    for r, c in cells:
        pos_dirs = [d for d in range(4) if int(option_counts[r, c, d]) > 0]
        zero_dirs = [d for d in range(4) if int(option_counts[r, c, d]) == 0]
        want_positive = (r, c) in positive_targets
        if want_positive and pos_dirs:
            directions[r, c] = random.choice(pos_dirs)
        elif not want_positive and zero_dirs:
            directions[r, c] = random.choice(zero_dirs)
        elif pos_dirs:
            directions[r, c] = random.choice(pos_dirs)
        else:
            directions[r, c] = random.choice(zero_dirs)

    if total >= 4:
        for missing in [d for d in range(4) if d not in set(map(int, directions.flat))]:
            random.shuffle(cells)
            replaced = False
            for r, c in cells:
                current_pos = int(option_counts[r, c, int(directions[r, c])]) > 0
                target_pos = int(option_counts[r, c, missing]) > 0
                if current_pos == target_pos:
                    directions[r, c] = missing
                    replaced = True
                    break
            if not replaced:
                r, c = random.choice(cells)
                directions[r, c] = missing

    chosen_counts = np.zeros((nr, nc), dtype=int)
    for r in range(nr):
        for c in range(nc):
            chosen_counts[r, c] = int(option_counts[r, c, int(directions[r, c])])

    positives = int(np.count_nonzero(chosen_counts > 0))
    if positives == 0 or positives == total:
        return None
    ratio = positives / float(total)
    if not (0.25 <= ratio <= 0.75):
        return None
    if int(chosen_counts.max()) < 2:
        return None

    return directions, chosen_counts


def _assign_corners(nr: int, nc: int) -> np.ndarray:
    total = nr * nc
    corners = np.zeros((nr, nc), dtype=int)
    cells = [(r, c) for r in range(nr) for c in range(nc)]
    random.shuffle(cells)

    idx = 0
    if total >= 4:
        for corner in (TL, TR, BL, BR):
            r, c = cells[idx]
            corners[r, c] = corner
            idx += 1
    for r, c in cells[idx:]:
        corners[r, c] = random.choice((TL, TR, BL, BR))
    return corners


def _render_input(colors: np.ndarray, directions: np.ndarray, corners: np.ndarray, cell_size: int) -> np.ndarray:
    nr, nc = colors.shape
    h = nr * (cell_size + 1) + 1
    w = nc * (cell_size + 1) + 1
    grid = np.full((h, w), BG, dtype=int)

    row_lines = list(range(0, h, cell_size + 1))
    col_lines = list(range(0, w, cell_size + 1))
    grid[row_lines, :] = GRID
    grid[:, col_lines] = GRID

    for r in range(nr):
        for c in range(nc):
            patch = np.full((cell_size, cell_size), BG, dtype=int)
            mr, mc = _corner_coord(int(corners[r, c]), cell_size)
            patch[mr, mc] = MARKER

            origins = _valid_arrow_origins(cell_size, int(directions[r, c]), int(corners[r, c]))
            assert origins
            r0, c0 = random.choice(origins)
            for dr, dc in ARROW_PATTERNS[int(directions[r, c])]:
                patch[r0 + dr, c0 + dc] = int(colors[r, c])

            rr = row_lines[r] + 1
            cc = col_lines[c] + 1
            grid[rr : rr + cell_size, cc : cc + cell_size] = patch

    return grid


def _candidate_dimensions() -> tuple[int, int, int]:
    cell_size = random.choice((5, 6, 7))
    if cell_size == 5:
        choices = (3, 4, 5)
    elif cell_size == 6:
        choices = (3, 4)
    else:
        choices = (3, 4)
    nr = random.choice(choices)
    nc = random.choice(choices)
    return cell_size, nr, nc


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        cell_size, nr, nc = _candidate_dimensions()
        palette_size = random.choice((2, 3, 4))
        palette = tuple(random.sample(ARROW_COLORS, palette_size))

        colors = _make_color_grid(nr, nc, palette)
        if len(set(map(int, colors.flat))) < 2:
            continue

        option_counts = _all_direction_counts(colors)
        assigned = _assign_directions(option_counts)
        if assigned is None:
            continue
        directions, chosen_counts = assigned
        corners = _assign_corners(nr, nc)

        grid = _render_input(colors, directions, corners, cell_size)

        decoded = _decode_grid(grid)
        _, _, _, _, decoded_size, dec_colors, dec_dirs, dec_corners = decoded
        if decoded_size != cell_size:
            continue
        if not np.array_equal(dec_colors, colors):
            continue
        if not np.array_equal(dec_dirs, directions):
            continue
        if not np.array_equal(dec_corners, corners):
            continue

        recomputed = np.zeros_like(chosen_counts)
        for r in range(nr):
            for c in range(nc):
                recomputed[r, c] = _same_color_visible(dec_colors, r, c, int(dec_dirs[r, c]))
        if not np.array_equal(recomputed, chosen_counts):
            continue

        if int(np.count_nonzero(chosen_counts == 0)) == 0 or int(np.count_nonzero(chosen_counts > 0)) == 0:
            continue

        out = main(grid)
        if out.shape != grid.shape or np.array_equal(out, grid):
            continue

        return grid

    raise RuntimeError("failed to generate a valid puzzle instance")
