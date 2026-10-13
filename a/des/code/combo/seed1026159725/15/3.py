from __future__ import annotations

import random

import numpy as np


MAX_GENERATION_ATTEMPTS = 400
CELL_SIZE_RANGE = (4, 7)
INTERNAL_LINE_COUNT_RANGE = (1, 3)
OBJECT_COUNT_RANGE = (1, 6)
PALETTE = tuple(range(10))


def _most_common_color(grid: np.ndarray) -> int:
    return int(np.bincount(grid.ravel()).argmax())


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            part: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                part.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(part)
    return parts


def _split_projection(counts: np.ndarray) -> list[int]:
    positive = sorted({int(v) for v in counts if int(v) > 0})
    if len(positive) < 2:
        return []
    best_idx = max(range(len(positive) - 1), key=lambda i: positive[i + 1] - positive[i])
    if positive[best_idx + 1] == positive[best_idx]:
        return []
    threshold = positive[best_idx]
    positions = [int(i) for i in np.flatnonzero(counts > threshold)]
    low_seen = any(0 < int(v) <= threshold for v in counts)
    if not positions or not low_seen:
        return []
    return positions


def _cell_bounds(index: int, lines: list[int], length: int) -> tuple[int, int]:
    prev = -1
    for line in lines:
        if index < line:
            return prev + 1, line - 1
        prev = line
    return prev + 1, length - 1


def _validate_line_candidate(
    grid: np.ndarray, bg: int, line_color: int
) -> tuple[list[int], list[int]] | None:
    h, w = grid.shape
    mask = grid == line_color
    row_lines = _split_projection(mask.sum(axis=1))
    col_lines = _split_projection(mask.sum(axis=0))
    if not row_lines or not col_lines:
        return None
    if any(r in {0, h - 1} for r in row_lines) or any(c in {0, w - 1} for c in col_lines):
        return None

    row_set = set(row_lines)
    col_set = set(col_lines)
    pixels = np.argwhere(mask)
    if not all(int(r) in row_set or int(c) in col_set for r, c in pixels):
        return None
    has_horizontal = any(int(r) in row_set and int(c) not in col_set for r, c in pixels)
    has_vertical = any(int(c) in col_set and int(r) not in row_set for r, c in pixels)
    if not (has_horizontal and has_vertical):
        return None

    for color in np.unique(grid):
        color = int(color)
        if color in {bg, line_color}:
            continue
        for comp in _components(grid == color):
            if any(r in row_set or c in col_set for r, c in comp):
                return None
            r0, c0 = comp[0]
            rb0, rb1 = _cell_bounds(r0, row_lines, h)
            cb0, cb1 = _cell_bounds(c0, col_lines, w)
            if not all(rb0 <= r <= rb1 and cb0 <= c <= cb1 for r, c in comp):
                return None

    return row_lines, col_lines


def _detect_grid_lines(grid: np.ndarray, bg: int) -> tuple[int, list[int], list[int]]:
    best: tuple[int, list[int], list[int], int] | None = None
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        lines = _validate_line_candidate(grid, bg, color)
        if lines is None:
            continue
        row_lines, col_lines = lines
        score = int((grid == color).sum()) + 8 * (len(row_lines) + len(col_lines))
        if best is None or score > best[3]:
            best = (color, row_lines, col_lines, score)
    assert best is not None, "expected a unique dashed-line color"
    return best[0], best[1], best[2]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer hidden grid lines from dashed cue pixels, then expand each object to fill
    its containing grid cell while leaving line positions as background.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _most_common_color(g)
    line_color, row_lines, col_lines = _detect_grid_lines(g, bg)
    row_set = set(row_lines)
    col_set = set(col_lines)

    out = np.full_like(g, bg)
    cell_to_color: dict[tuple[int, int], int] = {}

    for color in np.unique(g):
        color = int(color)
        if color in {bg, line_color}:
            continue
        for comp in _components(g == color):
            assert comp, "object component must be non-empty"
            r0, c0 = comp[0]
            rb0, rb1 = _cell_bounds(r0, row_lines, h)
            cb0, cb1 = _cell_bounds(c0, col_lines, w)
            assert all(r not in row_set and c not in col_set for r, c in comp)
            assert all(rb0 <= r <= rb1 and cb0 <= c <= cb1 for r, c in comp)
            cell_key = (rb0, cb0)
            if cell_key in cell_to_color:
                assert cell_to_color[cell_key] == color, "conflicting colors in one cell"
            cell_to_color[cell_key] = color
            out[rb0 : rb1 + 1, cb0 : cb1 + 1] = color

    out[row_lines, :] = bg
    out[:, col_lines] = bg
    return out


def _line_positions(cell_sizes: list[int]) -> list[int]:
    positions: list[int] = []
    cursor = cell_sizes[0]
    for size in cell_sizes[1:]:
        positions.append(cursor)
        cursor += 1 + size
    return positions


def _cell_spans(lines: list[int], length: int) -> list[tuple[int, int]]:
    bounds = [-1] + list(lines) + [length]
    return [(bounds[i] + 1, bounds[i + 1] - 1) for i in range(len(bounds) - 1)]


def _pick_dash_positions(start: int, end: int, rng: random.Random) -> list[int]:
    choices = list(range(start, end + 1))
    count = 2 if len(choices) <= 5 else rng.choice([2, 3])
    return sorted(rng.sample(choices, count))


def _is_connected(mask: np.ndarray) -> bool:
    points = np.argwhere(mask)
    if points.size == 0:
        return False
    return len(_components(mask)) == 1


def _random_walk_shape(h: int, w: int, rng: random.Random) -> np.ndarray:
    target = rng.randint(max(2, (h * w + 2) // 3), h * w)
    start = (rng.randrange(h), rng.randrange(w))
    filled = {start}
    frontier = [start]
    while len(filled) < target:
        base = frontier[rng.randrange(len(frontier))]
        nbrs = []
        for nr, nc in _neighbors4(base[0], base[1], h, w):
            if (nr, nc) not in filled:
                nbrs.append((nr, nc))
        if nbrs:
            nxt = rng.choice(nbrs)
            filled.add(nxt)
            frontier.append(nxt)
        else:
            frontier.pop(rng.randrange(len(frontier)))
            if not frontier:
                frontier = list(filled)
    mask = np.zeros((h, w), dtype=bool)
    for r, c in filled:
        mask[r, c] = True
    return mask


def _random_shape(h: int, w: int, rng: random.Random) -> np.ndarray:
    if h == 1 and w == 1:
        return np.ones((1, 1), dtype=bool)
    if h == 1:
        length = rng.randint(1, w)
        start = rng.randint(0, w - length)
        mask = np.zeros((1, w), dtype=bool)
        mask[0, start : start + length] = True
        return mask
    if w == 1:
        length = rng.randint(1, h)
        start = rng.randint(0, h - length)
        mask = np.zeros((h, 1), dtype=bool)
        mask[start : start + length, 0] = True
        return mask

    modes = ["solid", "ell", "walk"]
    if h >= 3 and w >= 3:
        modes.append("plus")
    mode = rng.choice(modes)

    if mode == "solid":
        mask = np.ones((h, w), dtype=bool)
        if h * w >= 4 and rng.random() < 0.65:
            rr, cc = rng.randrange(h), rng.randrange(w)
            mask[rr, cc] = False
            if not _is_connected(mask):
                mask[rr, cc] = True
        return mask

    if mode == "ell":
        mask = np.zeros((h, w), dtype=bool)
        mask[0, :] = True
        mask[:, 0] = True
        if rng.random() < 0.5:
            mask = np.flipud(mask)
        if rng.random() < 0.5:
            mask = np.fliplr(mask)
        return mask.copy()

    if mode == "plus":
        mask = np.zeros((h, w), dtype=bool)
        mask[h // 2, :] = True
        mask[:, w // 2] = True
        return mask

    return _random_walk_shape(h, w, rng)


def _build_expected(
    shape: tuple[int, int],
    bg: int,
    row_lines: list[int],
    col_lines: list[int],
    placements: list[tuple[tuple[int, int], int]],
) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    row_spans = _cell_spans(row_lines, shape[0])
    col_spans = _cell_spans(col_lines, shape[1])
    for (ri, ci), color in placements:
        r0, r1 = row_spans[ri]
        c0, c1 = col_spans[ci]
        out[r0 : r1 + 1, c0 : c1 + 1] = color
    out[row_lines, :] = bg
    out[:, col_lines] = bg
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_row_lines = rng.randint(*INTERNAL_LINE_COUNT_RANGE)
    n_col_lines = rng.randint(*INTERNAL_LINE_COUNT_RANGE)
    row_cell_sizes = [rng.randint(*CELL_SIZE_RANGE) for _ in range(n_row_lines + 1)]
    col_cell_sizes = [rng.randint(*CELL_SIZE_RANGE) for _ in range(n_col_lines + 1)]
    row_lines = _line_positions(row_cell_sizes)
    col_lines = _line_positions(col_cell_sizes)
    h = sum(row_cell_sizes) + n_row_lines
    w = sum(col_cell_sizes) + n_col_lines

    n_cells = (n_row_lines + 1) * (n_col_lines + 1)
    max_objects = min(OBJECT_COUNT_RANGE[1], n_cells, len(PALETTE) - 2)
    n_objects = rng.randint(OBJECT_COUNT_RANGE[0], max_objects)

    colors = list(PALETTE)
    rng.shuffle(colors)
    bg = colors.pop()
    line_color = colors.pop()
    object_colors = colors[:n_objects]

    grid = np.full((h, w), bg, dtype=int)
    row_spans = _cell_spans(row_lines, h)
    col_spans = _cell_spans(col_lines, w)

    for r in row_lines:
        for c0, c1 in col_spans:
            grid[r, _pick_dash_positions(c0, c1, rng)] = line_color
    for c in col_lines:
        for r0, r1 in row_spans:
            grid[_pick_dash_positions(r0, r1, rng), c] = line_color

    all_cells = [(ri, ci) for ri in range(len(row_spans)) for ci in range(len(col_spans))]
    chosen_cells = rng.sample(all_cells, n_objects)
    placements: list[tuple[tuple[int, int], int]] = []

    for (ri, ci), color in zip(chosen_cells, object_colors):
        r0, r1 = row_spans[ri]
        c0, c1 = col_spans[ci]
        cell_h = r1 - r0 + 1
        cell_w = c1 - c0 + 1
        obj_h = rng.randint(1, cell_h - 2)
        obj_w = rng.randint(1, cell_w - 2)
        top = rng.randint(r0 + 1, r1 - obj_h)
        left = rng.randint(c0 + 1, c1 - obj_w)
        shape = _random_shape(obj_h, obj_w, rng)
        coords = np.argwhere(shape)
        for dr, dc in coords:
            grid[top + int(dr), left + int(dc)] = color
        placements.append(((ri, ci), color))

    expected = _build_expected(grid.shape, bg, row_lines, col_lines, placements)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GENERATION_ATTEMPTS} attempts")
