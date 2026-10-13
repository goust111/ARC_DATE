from __future__ import annotations

import random

import numpy as np


BG = 0
SEP = 5
MARK = 9
PATTERN_COLORS = (1, 2, 3, 4, 6, 7, 8)

GEN_ATTEMPTS = 400
CELL_ATTEMPTS = 80
MASK_ATTEMPTS = 60


def _interval_slices(lines: list[int], size: int) -> tuple[list[slice], int]:
    bounds = [-1] + sorted(int(x) for x in lines) + [int(size)]
    spans = [slice(a + 1, b) for a, b in zip(bounds, bounds[1:]) if b - a > 1]
    assert spans, "expected at least one cell interval"
    lengths = {sl.stop - sl.start for sl in spans}
    assert len(lengths) == 1, "cell intervals must be equal-sized"
    return spans, lengths.pop()


def _find_layout(grid: np.ndarray) -> tuple[int, list[int], list[int], list[slice], list[slice], int]:
    h, w = grid.shape
    candidates: list[tuple[int, int, list[int], list[int], list[slice], list[slice], int]] = []
    for color in map(int, np.unique(grid)):
        row_lines = [r for r in range(h) if np.all(grid[r, :] == color)]
        col_lines = [c for c in range(w) if np.all(grid[:, c] == color)]
        if not row_lines or not col_lines:
            continue
        try:
            row_slices, cell_h = _interval_slices(row_lines, h)
            col_slices, cell_w = _interval_slices(col_lines, w)
        except AssertionError:
            continue
        if len(row_slices) < 2 or len(col_slices) < 2 or cell_h != cell_w:
            continue
        score = len(row_lines) + len(col_lines)
        candidates.append((score, color, row_lines, col_lines, row_slices, col_slices, cell_h))

    assert candidates, "could not detect the grey separator lattice"
    _score, sep, row_lines, col_lines, row_slices, col_slices, cell_size = max(candidates, key=lambda item: item[0])
    return sep, row_lines, col_lines, row_slices, col_slices, cell_size


def _reflect_diagonal(patch: np.ndarray, use_main_diagonal: bool) -> np.ndarray:
    return patch.T.copy() if use_main_diagonal else patch[::-1, ::-1].T.copy()


def _transform_cell(cell: np.ndarray) -> tuple[np.ndarray, int]:
    patch = np.asarray(cell, dtype=int)
    h, w = patch.shape
    assert h == w, "diagonal reflection requires square cells"

    marks = np.argwhere(patch == MARK)
    assert marks.shape == (1, 2), "each cell must contain exactly one maroon marker"
    mr, mc = map(int, marks[0])
    corners = {
        (0, 0): True,
        (h - 1, w - 1): True,
        (0, w - 1): False,
        (h - 1, 0): False,
    }
    assert (mr, mc) in corners, "marker must sit on one of the four cell corners"

    payload = patch.copy()
    payload[mr, mc] = BG
    reflected = _reflect_diagonal(payload, corners[(mr, mc)])

    colors = set(map(int, np.unique(reflected))) - {BG}
    assert len(colors) == 1, "each cell must contain exactly one pattern color"
    return reflected, int(next(iter(colors)))


def main(grid: np.ndarray) -> np.ndarray:
    """
    In each lattice cell, reflect the colored pattern across a diagonal chosen by the maroon corner marker.
    After all local reflections, swap horizontally adjacent cell contents whenever the two cells have the same
    pattern color. Grey separator lines remain unchanged and maroon markers are removed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    sep, row_lines, col_lines, row_slices, col_slices, _cell_size = _find_layout(g)
    cell_patches: list[list[np.ndarray]] = []
    cell_colors: list[list[int]] = []

    for rs in row_slices:
        patch_row: list[np.ndarray] = []
        color_row: list[int] = []
        for cs in col_slices:
            transformed, color = _transform_cell(g[rs, cs])
            patch_row.append(transformed)
            color_row.append(color)
        cell_patches.append(patch_row)
        cell_colors.append(color_row)

    for r in range(len(cell_patches)):
        c = 0
        while c + 1 < len(cell_patches[r]):
            if cell_colors[r][c] == cell_colors[r][c + 1]:
                cell_patches[r][c], cell_patches[r][c + 1] = cell_patches[r][c + 1], cell_patches[r][c]
                c += 2
            else:
                c += 1

    out = g.copy()
    for rs, patch_row in zip(row_slices, cell_patches):
        for cs, patch in zip(col_slices, patch_row):
            out[rs, cs] = patch
    for r in row_lines:
        out[r, :] = sep
    for c in col_lines:
        out[:, c] = sep
    return out


def _neighbors4(size: int, r: int, c: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < size and 0 <= nc < size:
            out.append((nr, nc))
    return out


def _random_connected_mask(size: int, forbidden: tuple[int, int], target: int, rng: random.Random) -> np.ndarray | None:
    available = [(r, c) for r in range(size) for c in range(size) if (r, c) != forbidden]
    if target < 1 or target > len(available):
        return None

    seed = rng.choice(available)
    cells = {seed}
    frontier = {pt for pt in _neighbors4(size, *seed) if pt != forbidden}
    while len(cells) < target and frontier:
        pt = rng.choice(tuple(frontier))
        frontier.remove(pt)
        if pt == forbidden or pt in cells:
            continue
        cells.add(pt)
        for nb in _neighbors4(size, *pt):
            if nb != forbidden and nb not in cells:
                frontier.add(nb)
    if len(cells) != target:
        return None

    mask = np.zeros((size, size), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _mask_signature(mask: np.ndarray) -> tuple[int, ...]:
    return tuple(int(x) for x in mask.astype(np.uint8).ravel())


def _make_cell_patch(
    size: int,
    color: int,
    use_main_diagonal: bool,
    rng: random.Random,
    avoid_reflected_signature: tuple[int, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    corner_choices = [(0, 0), (size - 1, size - 1)] if use_main_diagonal else [(0, size - 1), (size - 1, 0)]
    corner = rng.choice(corner_choices)

    max_pixels = min(size + 2, size * size - 1)
    min_pixels = min(3, max_pixels)
    for _ in range(CELL_ATTEMPTS):
        target = rng.randint(min_pixels, max_pixels)
        mask = None
        for _ in range(MASK_ATTEMPTS):
            candidate = _random_connected_mask(size, corner, target, rng)
            if candidate is not None:
                mask = candidate
                break
        if mask is None:
            continue

        reflected_mask = _reflect_diagonal(mask.astype(int), use_main_diagonal).astype(bool)
        if np.array_equal(mask, reflected_mask):
            continue
        if avoid_reflected_signature is not None and _mask_signature(reflected_mask) == avoid_reflected_signature:
            continue

        patch = np.full((size, size), BG, dtype=int)
        patch[mask] = color
        patch[corner] = MARK
        reflected = np.full((size, size), BG, dtype=int)
        reflected[reflected_mask] = color
        return patch, reflected
    return None


def _sample_color_grid(rows: int, cols: int, palette: list[int], rng: random.Random) -> list[list[int]] | None:
    for _ in range(120):
        color_grid: list[list[int]] = []
        for _r in range(rows):
            row: list[int] = []
            while len(row) < cols:
                prev = row[-1] if row else None
                remaining = cols - len(row)
                make_pair = remaining >= 2 and rng.random() < 0.5
                options = [color for color in palette if color != prev]
                if not options:
                    return None
                color = rng.choice(options)
                if make_pair:
                    row.extend([color, color])
                else:
                    row.append(color)
            color_grid.append(row)

        same = sum(color_grid[r][c] == color_grid[r][c + 1] for r in range(rows) for c in range(cols - 1))
        diff = rows * (cols - 1) - same
        if same > 0 and diff > 0:
            return color_grid
    return None


def _sample_diagonal_grid(rows: int, cols: int, rng: random.Random) -> list[list[bool]]:
    flags = [[bool(rng.getrandbits(1)) for _ in range(cols)] for _ in range(rows)]
    all_flags = [flag for row in flags for flag in row]
    if all(all_flags):
        flags[-1][-1] = False
    elif not any(all_flags):
        flags[0][0] = True
    return flags


def _axis_layout(count: int, inner: int, border: bool) -> tuple[int, list[slice], list[int]]:
    pos = 0
    lines: list[int] = []
    if border:
        lines.append(0)
        pos = 1

    spans: list[slice] = []
    for idx in range(count):
        spans.append(slice(pos, pos + inner))
        pos += inner
        if idx + 1 < count:
            lines.append(pos)
            pos += 1

    if border:
        lines.append(pos)
        pos += 1
    return pos, spans, lines


def _build_base_grid(rows: int, cols: int, inner: int, border: bool) -> tuple[np.ndarray, list[slice], list[slice], list[int], list[int]]:
    height, row_slices, row_lines = _axis_layout(rows, inner, border)
    width, col_slices, col_lines = _axis_layout(cols, inner, border)
    grid = np.full((height, width), BG, dtype=int)
    for r in row_lines:
        grid[r, :] = SEP
    for c in col_lines:
        grid[:, c] = SEP
    return grid, row_slices, col_slices, row_lines, col_lines


def _render_output(
    base: np.ndarray,
    row_slices: list[slice],
    col_slices: list[slice],
    reflected_cells: list[list[np.ndarray]],
    color_grid: list[list[int]],
) -> np.ndarray:
    out = base.copy()
    final_cells = [[patch.copy() for patch in row] for row in reflected_cells]
    for r in range(len(final_cells)):
        c = 0
        while c + 1 < len(final_cells[r]):
            if color_grid[r][c] == color_grid[r][c + 1]:
                final_cells[r][c], final_cells[r][c + 1] = final_cells[r][c + 1], final_cells[r][c]
                c += 2
            else:
                c += 1
    for rs, patch_row in zip(row_slices, final_cells):
        for cs, patch in zip(col_slices, patch_row):
            out[rs, cs] = patch
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    rows = rng.randint(2, 4)
    cols = rng.randint(3, 5)
    inner = rng.randint(4, 6)
    border = rng.random() < 0.5
    palette = rng.sample(PATTERN_COLORS, rng.randint(3, min(5, len(PATTERN_COLORS))))

    color_grid = _sample_color_grid(rows, cols, palette, rng)
    if color_grid is None:
        return None
    diagonal_grid = _sample_diagonal_grid(rows, cols, rng)

    base, row_slices, col_slices, _row_lines, _col_lines = _build_base_grid(rows, cols, inner, border)
    inp = base.copy()
    reflected_cells: list[list[np.ndarray]] = [[None for _ in range(cols)] for _ in range(rows)]  # type: ignore[list-item]

    for r in range(rows):
        c = 0
        while c < cols:
            color = color_grid[r][c]
            use_main = diagonal_grid[r][c]
            if c + 1 < cols and color_grid[r][c + 1] == color:
                first = _make_cell_patch(inner, color, use_main, rng)
                if first is None:
                    return None
                second = _make_cell_patch(
                    inner,
                    color,
                    diagonal_grid[r][c + 1],
                    rng,
                    avoid_reflected_signature=_mask_signature(first[1] == color),
                )
                if second is None:
                    return None
                inp[row_slices[r], col_slices[c]] = first[0]
                inp[row_slices[r], col_slices[c + 1]] = second[0]
                reflected_cells[r][c] = first[1]
                reflected_cells[r][c + 1] = second[1]
                c += 2
            else:
                cell = _make_cell_patch(inner, color, use_main, rng)
                if cell is None:
                    return None
                inp[row_slices[r], col_slices[c]] = cell[0]
                reflected_cells[r][c] = cell[1]
                c += 1

    expected = _render_output(base, row_slices, col_slices, reflected_cells, color_grid)
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
