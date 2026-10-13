import numpy as np


BLACK = 0
GREEN = 3
GEN_ATTEMPTS = 300


def _separator_structure(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    row_groups: dict[int, list[int]] = {}
    col_groups: dict[int, list[int]] = {}
    for r in range(h):
        vals, counts = np.unique(g[r], return_counts=True)
        idx = int(np.argmax(counts))
        if int(counts[idx]) >= w - 2:
            row_groups.setdefault(int(vals[idx]), []).append(int(r))
    for c in range(w):
        vals, counts = np.unique(g[:, c], return_counts=True)
        idx = int(np.argmax(counts))
        if int(counts[idx]) >= h - 2:
            col_groups.setdefault(int(vals[idx]), []).append(int(c))

    for color, rows in row_groups.items():
        cols = col_groups.get(color, [])
        if len(rows) < 2 or len(cols) < 2:
            continue
        row_gaps = [rows[i + 1] - rows[i] - 1 for i in range(len(rows) - 1)]
        col_gaps = [cols[i + 1] - cols[i] - 1 for i in range(len(cols) - 1)]
        if not row_gaps or not col_gaps:
            continue
        if min(row_gaps) <= 0 or min(col_gaps) <= 0:
            continue
        if len(set(row_gaps)) != 1 or len(set(col_gaps)) != 1:
            continue
        if len(row_gaps) != len(col_gaps):
            continue
        return int(color), rows, cols, int(row_gaps[0]), int(col_gaps[0])

    raise AssertionError("failed to detect regular separator grid")


def _cell_bounds(rows: list[int], cols: list[int]):
    bounds = []
    for i in range(len(rows) - 1):
        for j in range(len(cols) - 1):
            r0, r1 = int(rows[i]), int(rows[i + 1])
            c0, c1 = int(cols[j]), int(cols[j + 1])
            bounds.append((i, j, r0, r1, c0, c1))
    return bounds


def _marker_color(grid: np.ndarray, sep: int, r0: int, r1: int, c0: int, c1: int) -> int:
    coords = set()
    for c in range(c0 + 1, c1):
        if int(grid[r0, c]) not in (BLACK, sep):
            coords.add((r0, c))
        if int(grid[r1, c]) not in (BLACK, sep):
            coords.add((r1, c))
    for r in range(r0 + 1, r1):
        if int(grid[r, c0]) not in (BLACK, sep):
            coords.add((r, c0))
        if int(grid[r, c1]) not in (BLACK, sep):
            coords.add((r, c1))

    assert len(coords) == 1, "each cell must have exactly one border marker pixel"
    r, c = next(iter(coords))
    return int(grid[r, c])


def _vote_template(cells: list[np.ndarray]) -> np.ndarray:
    cell_h, cell_w = cells[0].shape
    template = np.zeros((cell_h, cell_w), dtype=int)
    for r in range(cell_h):
        for c in range(cell_w):
            counts: dict[int, int] = {}
            for cell in cells:
                val = int(cell[r, c])
                if val != BLACK:
                    counts[val] = counts.get(val, 0) + 1
            if not counts:
                continue
            best = max(counts.values())
            winners = [color for color, count in counts.items() if count == best]
            assert len(winners) == 1, "template vote must be unambiguous"
            template[r, c] = int(winners[0])
    return template


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover a shared template by cell-wise voting, then fill each cell's masked
    pixels with that cell's unique border marker color.
    """
    g = np.asarray(grid, dtype=int)
    sep, rows, cols, cell_h, cell_w = _separator_structure(g)
    assert cell_h > 0 and cell_w > 0

    cells = []
    markers = []
    bounds = _cell_bounds(rows, cols)
    for _i, _j, r0, r1, c0, c1 in bounds:
        cell = g[r0 + 1 : r1, c0 + 1 : c1]
        assert cell.shape == (cell_h, cell_w)
        cells.append(cell.copy())
        markers.append(_marker_color(g, sep, r0, r1, c0, c1))

    template = _vote_template(cells)
    out = g.copy()
    for cell, marker, (_i, _j, r0, r1, c0, c1) in zip(cells, markers, bounds):
        missing = (cell == BLACK) & (template != BLACK)
        filled = cell.copy()
        filled[missing] = int(marker)
        out[r0 + 1 : r1, c0 + 1 : c1] = filled

    return out


def _random_connected_mask(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    target = int(rng.integers(max(4, (h * w) // 4), max(5, (2 * h * w) // 3)))
    target = min(target, h * w - 1)
    mask = np.zeros((h, w), dtype=bool)
    start = (int(rng.integers(0, h)), int(rng.integers(0, w)))
    mask[start] = True
    frontier = [start]
    while int(mask.sum()) < target:
        br, bc = frontier[int(rng.integers(0, len(frontier)))]
        neighbors = []
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = br + dr, bc + dc
            if 0 <= rr < h and 0 <= cc < w and not mask[rr, cc]:
                neighbors.append((rr, cc))
        if neighbors:
            pick = neighbors[int(rng.integers(0, len(neighbors)))]
            mask[pick] = True
            frontier.append(pick)
        else:
            frontier = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    return mask


def _paint_template(
    rng: np.random.Generator, mask: np.ndarray, colors: list[int]
) -> np.ndarray:
    h, w = mask.shape
    template = np.zeros((h, w), dtype=int)
    mode = int(rng.integers(0, 3))

    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            if len(colors) == 1:
                val = colors[0]
            elif mode == 0:
                val = colors[(r + c) % len(colors)]
            elif mode == 1:
                val = colors[r % len(colors)]
            else:
                band = 0 if c < w // 2 else 1
                val = colors[(band + r) % len(colors)]
            template[r, c] = int(val)
    return template


def _marker_positions(rng: np.random.Generator, r0: int, r1: int, c0: int, c1: int):
    side = int(rng.integers(0, 4))
    if side == 0:
        return r0, int(rng.integers(c0 + 1, c1))
    if side == 1:
        return r1, int(rng.integers(c0 + 1, c1))
    if side == 2:
        return int(rng.integers(r0 + 1, r1)), c0
    return int(rng.integers(r0 + 1, r1)), c1


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    cell_h = int(rng.integers(4, 9))
    cell_w = int(rng.integers(4, 9))
    palette = [1, 2, 4, 5, 6, 7, 8, 9]
    marker_colors = list(rng.choice(palette, size=4, replace=False))
    template_palette = [c for c in palette if c not in marker_colors]
    n_template_colors = int(rng.integers(1, min(4, len(template_palette)) + 1))
    template_colors = list(rng.choice(template_palette, size=n_template_colors, replace=False))

    mask = _random_connected_mask(rng, cell_h, cell_w)
    if int(mask.sum()) < 4:
        return None
    template = _paint_template(rng, mask, template_colors)

    fg_positions = [(int(r), int(c)) for r, c in np.argwhere(template != BLACK)]
    if len(fg_positions) < 4:
        return None

    cells = [template.copy() for _ in range(4)]
    assigned = [0, 1, 2, 3]
    rng.shuffle(assigned)
    for idx, (r, c) in zip(assigned, fg_positions[:4]):
        cells[int(idx)][r, c] = BLACK
    for r, c in fg_positions[4:]:
        if bool(rng.random() < 0.7):
            idx = int(rng.integers(0, 4))
            cells[idx][r, c] = BLACK

    if not all(np.any((cell == BLACK) & (template != BLACK)) for cell in cells):
        return None

    h = 2 * cell_h + 3
    w = 2 * cell_w + 3
    grid = np.full((h, w), BLACK, dtype=int)
    for r in (0, cell_h + 1, h - 1):
        grid[r, :] = GREEN
    for c in (0, cell_w + 1, w - 1):
        grid[:, c] = GREEN

    for idx, cell in enumerate(cells):
        row = idx // 2
        col = idx % 2
        r0 = 0 if row == 0 else cell_h + 1
        c0 = 0 if col == 0 else cell_w + 1
        r1 = r0 + cell_h + 1
        c1 = c0 + cell_w + 1
        grid[r0 + 1 : r1, c0 + 1 : c1] = cell
        mr, mc = _marker_positions(rng, r0, r1, c0, c1)
        grid[mr, mc] = int(marker_colors[idx])

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
