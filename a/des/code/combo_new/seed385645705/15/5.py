from __future__ import annotations

import numpy as np


BACKGROUND = 0
SEPARATOR = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]

MIN_TOTAL_ROWS = 3
MAX_TOTAL_ROWS = 5
MIN_TOTAL_COLS = 3
MAX_TOTAL_COLS = 5
MIN_CELL_SIDE = 4
MAX_CELL_SIDE = 7

SHAPE_ATTEMPTS = 150
GENERATION_ATTEMPTS = 400


def _full_line_indices(grid: np.ndarray, color: int, axis: int) -> list[int]:
    if axis == 0:
        return [r for r in range(grid.shape[0]) if np.all(grid[r, :] == color)]
    return [c for c in range(grid.shape[1]) if np.all(grid[:, c] == color)]


def _regular_step(indices: list[int], length: int) -> int | None:
    if len(indices) < 4 or indices[0] != 0 or indices[-1] != length - 1:
        return None
    steps = [b - a for a, b in zip(indices, indices[1:])]
    if len(set(steps)) != 1:
        return None
    step = steps[0]
    if step < 3:
        return None
    expected = list(range(0, length, step))
    if expected != indices or expected[-1] != length - 1:
        return None
    return int(step)


def _find_board(grid: np.ndarray) -> tuple[int, list[int], list[int], int, int]:
    candidates: list[tuple[int, list[int], list[int], int, int]] = []
    h, w = map(int, grid.shape)

    for color in np.unique(grid):
        color = int(color)
        row_lines = _full_line_indices(grid, color, axis=0)
        col_lines = _full_line_indices(grid, color, axis=1)
        row_step = _regular_step(row_lines, h)
        col_step = _regular_step(col_lines, w)
        if row_step is None or col_step is None:
            continue

        total_rows = len(row_lines) - 1
        total_cols = len(col_lines) - 1
        if total_rows < MIN_TOTAL_ROWS or total_cols < MIN_TOTAL_COLS:
            continue
        candidates.append((color, row_lines, col_lines, row_step - 1, col_step - 1))

    assert len(candidates) == 1, "expected exactly one regular separator grid"
    return candidates[0]


def _cell_spans(lines: list[int]) -> list[tuple[int, int]]:
    return [(lines[i] + 1, lines[i + 1]) for i in range(len(lines) - 1)]


def _foreground_color(cell: np.ndarray) -> int:
    colors = set(map(int, np.unique(cell))) - {BACKGROUND}
    assert len(colors) == 1, "each cell must contain exactly one foreground color"
    return next(iter(colors))


def _shape_signature(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    coords = np.argwhere(mask)
    assert coords.size > 0
    coords = coords - coords.min(axis=0)
    return tuple(sorted((int(r), int(c)) for r, c in coords))


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _random_connected_shape(cell_h: int, cell_w: int, rng: np.random.Generator) -> np.ndarray | None:
    bbox_h = int(rng.integers(2, cell_h + 1))
    bbox_w = int(rng.integers(2, cell_w + 1))
    bbox_area = bbox_h * bbox_w
    min_area = max(3, (bbox_area + 3) // 4)
    max_area = bbox_area - 1 if bbox_area > 4 else bbox_area
    if min_area > max_area:
        return None
    target_area = int(rng.integers(min_area, max_area + 1))

    mask = np.zeros((bbox_h, bbox_w), dtype=bool)
    seed = (int(rng.integers(bbox_h)), int(rng.integers(bbox_w)))
    mask[seed] = True

    while int(mask.sum()) < target_area:
        frontier: list[tuple[int, int]] = []
        for r, c in zip(*np.where(mask), strict=False):
            for rr, cc in _neighbors4(int(r), int(c), bbox_h, bbox_w):
                if not mask[rr, cc]:
                    frontier.append((rr, cc))
        if not frontier:
            return None
        rr, cc = frontier[int(rng.integers(len(frontier)))]
        mask[rr, cc] = True

    if int(np.unique(np.where(mask)[0]).size) < 2 or int(np.unique(np.where(mask)[1]).size) < 2:
        return None

    k = int(rng.integers(4))
    if k:
        mask = np.rot90(mask, k)
    if bool(rng.integers(2)):
        mask = np.fliplr(mask)
    if bool(rng.integers(2)):
        mask = np.flipud(mask)

    shape_h, shape_w = map(int, mask.shape)
    if shape_h > cell_h or shape_w > cell_w:
        return None
    top = int(rng.integers(0, cell_h - shape_h + 1))
    left = int(rng.integers(0, cell_w - shape_w + 1))
    placed = np.zeros((cell_h, cell_w), dtype=bool)
    placed[top : top + shape_h, left : left + shape_w] = mask
    return placed


def _connected_keep(mask: np.ndarray, keep_count: int, rng: np.random.Generator) -> np.ndarray:
    coords = [tuple(map(int, xy)) for xy in np.argwhere(mask)]
    seed = coords[int(rng.integers(len(coords)))]
    kept = {seed}
    while len(kept) < keep_count:
        frontier: list[tuple[int, int]] = []
        for r, c in kept:
            for rr, cc in _neighbors4(r, c, *mask.shape):
                if mask[rr, cc] and (rr, cc) not in kept:
                    frontier.append((rr, cc))
        assert frontier
        kept.add(frontier[int(rng.integers(len(frontier)))])

    out = np.zeros_like(mask, dtype=bool)
    for r, c in kept:
        out[r, c] = True
    return out


def _make_partial_mask(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    area = int(mask.sum())
    assert area >= 2
    mode = int(rng.integers(3))

    if mode == 0:
        keep_count = int(rng.integers(max(1, area // 3), area))
        chosen = rng.choice(area, size=keep_count, replace=False)
        coords = np.argwhere(mask)
        out = np.zeros_like(mask, dtype=bool)
        for idx in np.atleast_1d(chosen):
            r, c = coords[int(idx)]
            out[int(r), int(c)] = True
        return out

    if mode == 1:
        keep_count = int(rng.integers(1, area))
        return _connected_keep(mask, keep_count, rng)

    out = mask.copy()
    rows = [int(r) for r in np.unique(np.where(mask)[0])]
    cols = [int(c) for c in np.unique(np.where(mask)[1])]
    use_rows = bool(rng.integers(2))
    lines = rows if use_rows else cols
    start = int(rng.integers(len(lines)))
    stop = int(rng.integers(start + 1, len(lines) + 1))
    for line in lines[start:stop]:
        if use_rows:
            out[line, :] &= ~mask[line, :]
        else:
            out[:, line] &= ~mask[:, line]

    remaining = int(out.sum())
    if 0 < remaining < area:
        return out

    keep_count = int(rng.integers(1, area))
    return _connected_keep(mask, keep_count, rng)


def _paint_cell(grid: np.ndarray, row_spans: list[tuple[int, int]], col_spans: list[tuple[int, int]], row: int, col: int, cell: np.ndarray) -> None:
    r0, r1 = row_spans[row]
    c0, c1 = col_spans[col]
    grid[r0:r1, c0:c1] = cell


def main(grid: np.ndarray) -> np.ndarray:
    """Complete each bottom-row puzzle cell using the same-column template with the matching color.

    Input structure:
    - A black background is partitioned by one separator color into an equal-size board of rectangular cells.
    - Every row except the bottom one is a template row; within each column those template cells share one shape but use different foreground colors.
    - The bottom row contains partial copies of the column templates, each keeping only some pixels from one template color.

    Rule:
    - Detect the regular separator lattice and slice the grid into cells.
    - For each bottom-row cell, use its remaining foreground color to identify the matching template row in the same column.
    - Copy the matched full template only into the missing background pixels of that answer cell, leaving all template cells and preserved pixels unchanged.

    Key insight:
    - Color is the lookup key that selects which same-column template row to use, while the shared shape within a column tells you how to restore the erased pixels.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    _, row_lines, col_lines, _, _ = _find_board(g)
    row_spans = _cell_spans(row_lines)
    col_spans = _cell_spans(col_lines)
    total_rows = len(row_spans)
    total_cols = len(col_spans)
    template_rows = total_rows - 1
    assert template_rows >= 2

    templates: list[dict[int, np.ndarray]] = []
    for col in range(total_cols):
        lookup: dict[int, np.ndarray] = {}
        for row in range(template_rows):
            r0, r1 = row_spans[row]
            c0, c1 = col_spans[col]
            cell = g[r0:r1, c0:c1]
            color = _foreground_color(cell)
            assert color not in lookup, "template colors must be unique within a column"
            lookup[color] = cell.copy()
        templates.append(lookup)

    out = g.copy()
    answer_row = total_rows - 1
    for col in range(total_cols):
        r0, r1 = row_spans[answer_row]
        c0, c1 = col_spans[col]
        cell = g[r0:r1, c0:c1]
        color = _foreground_color(cell)
        assert color in templates[col], "answer color must match one template row in the same column"
        template = templates[col][color]
        assert np.all((cell != BACKGROUND) <= (template != BACKGROUND))
        filled = cell.copy()
        fill_mask = (filled == BACKGROUND) & (template != BACKGROUND)
        filled[fill_mask] = template[fill_mask]
        out[r0:r1, c0:c1] = filled
    return out


def _cell_side_choices(total: int) -> list[int]:
    upper = min(MAX_CELL_SIDE, (29 // total) - 1)
    return list(range(MIN_CELL_SIDE, upper + 1))


def _sample_column_shape(cell_h: int, cell_w: int, used: set[tuple[tuple[int, int], ...]], rng: np.random.Generator) -> np.ndarray | None:
    for _ in range(SHAPE_ATTEMPTS):
        mask = _random_connected_shape(cell_h, cell_w, rng)
        if mask is None:
            continue
        sig = _shape_signature(mask)
        if sig in used:
            continue
        used.add(sig)
        return mask
    return None


def _render_colored_cell(mask: np.ndarray, color: int) -> np.ndarray:
    cell = np.full(mask.shape, BACKGROUND, dtype=int)
    cell[mask] = int(color)
    return cell


def _build_example(
    total_rows: int,
    total_cols: int,
    cell_h: int,
    cell_w: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray] | None:
    height = total_rows * (cell_h + 1) + 1
    width = total_cols * (cell_w + 1) + 1
    if height > 30 or width > 30:
        return None

    row_lines = list(range(0, height, cell_h + 1))
    col_lines = list(range(0, width, cell_w + 1))
    row_spans = _cell_spans(row_lines)
    col_spans = _cell_spans(col_lines)

    inp = np.full((height, width), BACKGROUND, dtype=int)
    out = inp.copy()
    inp[row_lines, :] = SEPARATOR
    inp[:, col_lines] = SEPARATOR
    out[row_lines, :] = SEPARATOR
    out[:, col_lines] = SEPARATOR

    template_rows = total_rows - 1
    used_shapes: set[tuple[tuple[int, int], ...]] = set()
    chosen_rows: list[int] = []
    missing_modes: list[int] = []

    for col in range(total_cols):
        mask = _sample_column_shape(cell_h, cell_w, used_shapes, rng)
        if mask is None:
            return None

        colors = [int(c) for c in rng.choice(COLOR_POOL, size=template_rows, replace=False)]
        source_row = int(rng.integers(template_rows))
        chosen_rows.append(source_row)

        for row, color in enumerate(colors):
            full_cell = _render_colored_cell(mask, color)
            _paint_cell(inp, row_spans, col_spans, row, col, full_cell)
            _paint_cell(out, row_spans, col_spans, row, col, full_cell)

        answer_color = colors[source_row]
        partial_mask = _make_partial_mask(mask, rng)
        missing_modes.append(int(partial_mask.sum()))
        partial_cell = _render_colored_cell(partial_mask, answer_color)
        full_answer = _render_colored_cell(mask, answer_color)
        _paint_cell(inp, row_spans, col_spans, template_rows, col, partial_cell)
        _paint_cell(out, row_spans, col_spans, template_rows, col, full_answer)

    if len(set(chosen_rows)) == 1 and template_rows > 2:
        return None
    if min(missing_modes) <= 0:
        return None
    return inp, out


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GENERATION_ATTEMPTS):
        total_rows = int(rng.integers(MIN_TOTAL_ROWS, MAX_TOTAL_ROWS + 1))
        total_cols = int(rng.integers(MIN_TOTAL_COLS, MAX_TOTAL_COLS + 1))
        cell_h = int(rng.choice(_cell_side_choices(total_rows)))
        cell_w = int(rng.choice(_cell_side_choices(total_cols)))

        pair = _build_example(total_rows, total_cols, cell_h, cell_w, rng)
        if pair is None:
            continue
        inp, expected = pair
        try:
            actual = main(inp)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError(f"generate_input: failed after {GENERATION_ATTEMPTS} attempts")
