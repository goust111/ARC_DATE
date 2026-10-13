import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


BG = 0
SEP = 7
COLOR_POOL = [1, 2, 3, 4, 5, 6, 8, 9]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_RETRIES = 400


def _mode(values: np.ndarray) -> int:
    vals, counts = np.unique(values, return_counts=True)
    return int(vals[np.argmax(counts)])


def _step_candidates(size: int) -> List[int]:
    return [step for step in range(2, size) if (size - 1) % step == 0 and (size - 1) // step >= 2]


def _infer_lattice(grid: np.ndarray) -> Tuple[int, List[int], List[int]]:
    h, w = grid.shape
    best = None

    for row_step in _step_candidates(h):
        row_lines = list(range(0, h, row_step))
        row_set = set(row_lines)
        row_non = [r for r in range(h) if r not in row_set]
        for col_step in _step_candidates(w):
            col_lines = list(range(0, w, col_step))
            col_set = set(col_lines)
            col_non = [c for c in range(w) if c not in col_set]

            intersections = grid[np.ix_(row_lines, col_lines)]
            inter_vals, inter_counts = np.unique(intersections, return_counts=True)
            sep = int(inter_vals[np.argmax(inter_counts)])

            row_good = float(np.mean([(grid[r] == sep).mean() for r in row_lines]))
            col_good = float(np.mean([(grid[:, c] == sep).mean() for c in col_lines]))
            if row_good < 0.6 or col_good < 0.6:
                continue

            row_bad = float(np.mean([(grid[r] == sep).mean() for r in row_non])) if row_non else 0.0
            col_bad = float(np.mean([(grid[:, c] == sep).mean() for c in col_non])) if col_non else 0.0
            inter_good = float((intersections == sep).mean())
            score = inter_good + row_good + col_good - 0.4 * (row_bad + col_bad)

            if best is None or score > best[0]:
                best = (score, sep, row_lines, col_lines)

    assert best is not None, "failed to infer lattice"
    _score, sep, row_lines, col_lines = best
    return int(sep), list(map(int, row_lines)), list(map(int, col_lines))


def _line_mask(shape: Tuple[int, int], row_lines: Sequence[int], col_lines: Sequence[int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[list(row_lines), :] = True
    mask[:, list(col_lines)] = True
    return mask


def _cell_inner_bounds(row_lines: Sequence[int], col_lines: Sequence[int], i: int, j: int) -> Tuple[int, int, int, int]:
    return row_lines[i] + 1, row_lines[i + 1] - 1, col_lines[j] + 1, col_lines[j + 1] - 1


def _crop_pattern(arr: np.ndarray, bg: int) -> Optional[np.ndarray]:
    pts = np.argwhere(arr != bg)
    if pts.size == 0:
        return None
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return arr[r0 : r1 + 1, c0 : c1 + 1].copy()


def _cell_border_key(
    grid: np.ndarray, row_lines: Sequence[int], col_lines: Sequence[int], i: int, j: int, sep: int
) -> Optional[int]:
    r0, r1, c0, c1 = _cell_inner_bounds(row_lines, col_lines, i, j)
    border = np.concatenate(
        [
            grid[row_lines[i], c0 : c1 + 1],
            grid[row_lines[i + 1], c0 : c1 + 1],
            grid[r0 : r1 + 1, col_lines[j]],
            grid[r0 : r1 + 1, col_lines[j + 1]],
        ]
    )
    colors = {int(v) for v in border if int(v) != sep}
    if len(colors) != 1:
        return None
    return int(next(iter(colors)))


def _paste_centered(out: np.ndarray, bounds: Tuple[int, int, int, int], patch: np.ndarray, bg: int) -> None:
    r0, r1, c0, c1 = bounds
    cell_h = r1 - r0 + 1
    cell_w = c1 - c0 + 1
    ph, pw = patch.shape
    dr = r0 + (cell_h - ph) // 2
    dc = c0 + (cell_w - pw) // 2
    out[r0 : r1 + 1, c0 : c1 + 1] = bg
    out[dr : dr + ph, dc : dc + pw] = patch


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each seeded partition with the pattern from the border legend whose colored border segment matches the seed color.

    Input structure:
    - The background is black, and 1-pixel-wide orange lattice lines partition the canvas into equal rectangular cells.
    - Some outer-ring cells are legends: they contain a compact pattern inside the cell, and one segment of that cell's orange border is replaced by a key color.
    - Other cells are either empty or contain exactly one colored seed pixel somewhere inside the black interior.

    Rule:
    - Infer the lattice spacing from the repeated separator rows and columns, then inspect every outer-ring cell for a non-orange border segment.
    - Use that border-segment color as a lookup key, and crop the legend's interior pattern by its tight non-background bounding box.
    - For every non-legend cell containing a single seed pixel of color C, erase the seed and copy the matched legend pattern into that cell, centered within the cell interior.

    Key insight:
    - The seed color is only an index into the border legends; the actual painted content comes from the legend interior, so the solve path is seed color -> legend border key -> copied pattern.
    """
    g = np.asarray(grid, dtype=int)
    sep, row_lines, col_lines = _infer_lattice(g)
    bg = BG

    cell_rows = len(row_lines) - 1
    cell_cols = len(col_lines) - 1
    legend_patterns: Dict[int, np.ndarray] = {}
    legend_cells = set()

    for i in range(cell_rows):
        for j in range(cell_cols):
            if i not in (0, cell_rows - 1) and j not in (0, cell_cols - 1):
                continue
            key = _cell_border_key(g, row_lines, col_lines, i, j, sep)
            if key is None:
                continue
            r0, r1, c0, c1 = _cell_inner_bounds(row_lines, col_lines, i, j)
            patch = _crop_pattern(g[r0 : r1 + 1, c0 : c1 + 1], bg)
            if patch is None:
                continue
            legend_patterns[int(key)] = patch
            legend_cells.add((i, j))

    out = g.copy()
    for i in range(cell_rows):
        for j in range(cell_cols):
            if (i, j) in legend_cells:
                continue
            r0, r1, c0, c1 = _cell_inner_bounds(row_lines, col_lines, i, j)
            interior = g[r0 : r1 + 1, c0 : c1 + 1]
            pts = np.argwhere(interior != bg)
            if len(pts) != 1:
                continue
            seed_color = int(interior[tuple(pts[0])])
            patch = legend_patterns.get(seed_color)
            if patch is None:
                continue
            _paste_centered(out, (r0, r1, c0, c1), patch, bg)

    return out


def _outer_cells(cell_rows: int, cell_cols: int) -> List[Tuple[int, int]]:
    return [(i, j) for i in range(cell_rows) for j in range(cell_cols) if i in (0, cell_rows - 1) or j in (0, cell_cols - 1)]


def _outer_sides(i: int, j: int, cell_rows: int, cell_cols: int) -> List[str]:
    sides = []
    if i == 0:
        sides.append("top")
    if i == cell_rows - 1:
        sides.append("bottom")
    if j == 0:
        sides.append("left")
    if j == cell_cols - 1:
        sides.append("right")
    return sides


def _random_connected_mask(rng: random.Random, height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    start = (rng.randrange(height), rng.randrange(width))
    mask[start] = True
    max_cells = height * width
    target = 1 if max_cells == 1 else rng.randint(2, max_cells)

    while int(mask.sum()) < target:
        frontier = []
        for r, c in np.argwhere(mask):
            for dr, dc in DIR4:
                rr, cc = int(r + dr), int(c + dc)
                if 0 <= rr < height and 0 <= cc < width and not mask[rr, cc]:
                    frontier.append((rr, cc))
        if not frontier:
            break
        rr, cc = rng.choice(frontier)
        mask[rr, cc] = True

    return mask


def _random_pattern(rng: random.Random, cell_h: int, cell_w: int, color: int) -> np.ndarray:
    height = rng.randint(1, min(4, cell_h))
    width = rng.randint(1, min(4, cell_w))
    mask = _random_connected_mask(rng, height, width)
    patch = np.full((height, width), BG, dtype=int)
    patch[mask] = int(color)
    return patch


def _paint_lattice(grid: np.ndarray, cell_rows: int, cell_cols: int, cell_h: int, cell_w: int) -> None:
    row_step = cell_h + 1
    col_step = cell_w + 1
    grid[::row_step, :] = SEP
    grid[:, ::col_step] = SEP


def _paint_key_segment(
    rng: random.Random,
    grid: np.ndarray,
    row_lines: Sequence[int],
    col_lines: Sequence[int],
    cell: Tuple[int, int],
    side: str,
    color: int,
) -> None:
    i, j = cell
    r0, r1, c0, c1 = _cell_inner_bounds(row_lines, col_lines, i, j)
    if side in ("top", "bottom"):
        row = row_lines[i] if side == "top" else row_lines[i + 1]
        span = list(range(c0, c1 + 1))
        seg_len = min(len(span), 1 + (len(span) > 2) + (len(span) > 4))
        if len(span) > seg_len:
            start = rng.randint(0, len(span) - seg_len)
            cols = span[start : start + seg_len]
        else:
            cols = span
        grid[row, cols] = color
    else:
        col = col_lines[j] if side == "left" else col_lines[j + 1]
        span = list(range(r0, r1 + 1))
        seg_len = min(len(span), 1 + (len(span) > 2) + (len(span) > 4))
        if len(span) > seg_len:
            start = rng.randint(0, len(span) - seg_len)
            rows = span[start : start + seg_len]
        else:
            rows = span
        grid[rows, col] = color


def _paint_patch(grid: np.ndarray, bounds: Tuple[int, int, int, int], patch: np.ndarray) -> None:
    r0, r1, c0, c1 = bounds
    cell_h = r1 - r0 + 1
    cell_w = c1 - c0 + 1
    ph, pw = patch.shape
    dr = r0 + (cell_h - ph) // 2
    dc = c0 + (cell_w - pw) // 2
    grid[dr : dr + ph, dc : dc + pw] = patch


def _gen_attempt(rng: random.Random) -> Optional[np.ndarray]:
    cell_rows = rng.choice([3, 4])
    cell_cols = rng.choice([3, 4])

    max_h = min(6, (29 // cell_rows) - 1)
    max_w = min(6, (29 // cell_cols) - 1)
    if max_h < 3 or max_w < 3:
        return None
    cell_h = rng.randint(3, max_h)
    cell_w = rng.randint(3, max_w)

    h = cell_rows * (cell_h + 1) + 1
    w = cell_cols * (cell_w + 1) + 1
    grid = np.full((h, w), BG, dtype=int)
    _paint_lattice(grid, cell_rows, cell_cols, cell_h, cell_w)

    row_lines = list(range(0, h, cell_h + 1))
    col_lines = list(range(0, w, cell_w + 1))
    outer_cells = _outer_cells(cell_rows, cell_cols)

    max_legends = min(4, len(outer_cells) - 1, len(COLOR_POOL) // 2)
    legend_count = rng.randint(2, max_legends)
    legend_cells = rng.sample(outer_cells, legend_count)

    colors = rng.sample(COLOR_POOL, 2 * legend_count)
    key_colors = colors[:legend_count]
    pattern_colors = colors[legend_count:]

    legend_patterns: Dict[int, np.ndarray] = {}
    for cell, key_color, pattern_color in zip(legend_cells, key_colors, pattern_colors):
        patch = _random_pattern(rng, cell_h, cell_w, pattern_color)
        bounds = _cell_inner_bounds(row_lines, col_lines, cell[0], cell[1])
        _paint_patch(grid, bounds, patch)
        side = rng.choice(_outer_sides(cell[0], cell[1], cell_rows, cell_cols))
        _paint_key_segment(rng, grid, row_lines, col_lines, cell, side, key_color)
        legend_patterns[int(key_color)] = patch

    candidate_cells = [(i, j) for i in range(cell_rows) for j in range(cell_cols) if (i, j) not in legend_cells]
    if len(candidate_cells) < legend_count:
        return None

    seed_total = rng.randint(legend_count, min(len(candidate_cells), legend_count + 4))
    seed_cells = rng.sample(candidate_cells, seed_total)
    seed_keys = key_colors.copy()
    while len(seed_keys) < seed_total:
        seed_keys.append(rng.choice(key_colors))
    rng.shuffle(seed_keys)

    seed_specs = []
    for cell, key_color in zip(seed_cells, seed_keys):
        r0, r1, c0, c1 = _cell_inner_bounds(row_lines, col_lines, cell[0], cell[1])
        sr = rng.randint(r0, r1)
        sc = rng.randint(c0, c1)
        grid[sr, sc] = key_color
        seed_specs.append((cell, key_color))

    out = main(grid)
    if np.array_equal(out, grid):
        return None

    for cell in legend_cells:
        r0, r1, c0, c1 = _cell_inner_bounds(row_lines, col_lines, cell[0], cell[1])
        if not np.array_equal(out[r0 : r1 + 1, c0 : c1 + 1], grid[r0 : r1 + 1, c0 : c1 + 1]):
            return None

    for cell, key_color in seed_specs:
        r0, r1, c0, c1 = _cell_inner_bounds(row_lines, col_lines, cell[0], cell[1])
        expected = np.full((cell_h, cell_w), BG, dtype=int)
        patch = legend_patterns[int(key_color)]
        ph, pw = patch.shape
        dr = (cell_h - ph) // 2
        dc = (cell_w - pw) // 2
        expected[dr : dr + ph, dc : dc + pw] = patch
        if not np.array_equal(out[r0 : r1 + 1, c0 : c1 + 1], expected):
            return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
