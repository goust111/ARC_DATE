from __future__ import annotations

from collections import Counter
import random

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    colors = [int(color) for color in np.unique(grid)]
    if exclude is None:
        return colors
    return [color for color in colors if color not in exclude]


def _is_regular_indices(indices: np.ndarray) -> bool:
    if len(indices) < 2:
        return False
    diffs = np.diff(indices)
    return bool(np.all(diffs == diffs[0]) and diffs[0] >= 2)


def _bands(length: int, lines: np.ndarray) -> list[tuple[int, int]]:
    bands: list[tuple[int, int]] = []
    start = 0
    for line in map(int, lines):
        if start <= line - 1:
            bands.append((start, line - 1))
        start = line + 1
    if start <= length - 1:
        bands.append((start, length - 1))
    return bands


def _validate_lattice_candidate(
    grid: np.ndarray,
    mask: np.ndarray,
    bg: int,
    color: int,
    row_lines: np.ndarray,
    col_lines: np.ndarray,
    non_line_pattern: np.ndarray,
) -> bool:
    height, width = mask.shape
    if len(row_lines) < 2 or len(col_lines) < 2:
        return False
    if not _is_regular_indices(row_lines) or not _is_regular_indices(col_lines):
        return False

    all_rows = np.arange(height)
    all_cols = np.arange(width)
    non_rows = all_rows[~np.isin(all_rows, row_lines)]
    non_cols = all_cols[~np.isin(all_cols, col_lines)]
    if len(non_rows) < 2 or len(non_cols) < 2:
        return False

    if not np.array_equal(non_line_pattern.astype(bool), np.isin(all_cols, col_lines)):
        return False
    if not np.all(mask[np.ix_(row_lines, non_cols)]):
        return False
    if not np.all(mask[np.ix_(non_rows, col_lines)]):
        return False
    if np.any(mask[np.ix_(non_rows, non_cols)]):
        return False
    if not np.all(mask[non_rows] == non_line_pattern):
        return False

    intersections = grid[np.ix_(row_lines, col_lines)]
    if not np.isin(intersections, (bg, color)).all():
        return False
    return True


def _candidates_from_orientation(
    grid: np.ndarray, bg: int, color: int, *, transposed: bool
) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    work_grid = grid.T if transposed else grid
    mask = work_grid == color
    counter = Counter(tuple(int(x) for x in row) for row in mask)
    found: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    for pattern, count in counter.most_common():
        pattern_arr = np.array(pattern, dtype=bool)
        if count < 2 or pattern_arr.sum() < 2:
            continue

        col_lines = np.flatnonzero(pattern_arr)
        non_cols = np.flatnonzero(~pattern_arr)
        if len(col_lines) < 2 or len(non_cols) < 2:
            continue

        row_lines = np.flatnonzero(np.any(mask[:, non_cols], axis=1))
        if not _validate_lattice_candidate(
            work_grid, mask, bg, color, row_lines, col_lines, pattern_arr
        ):
            continue

        if transposed:
            rows = tuple(int(x) for x in col_lines)
            cols = tuple(int(x) for x in row_lines)
        else:
            rows = tuple(int(x) for x in row_lines)
            cols = tuple(int(x) for x in col_lines)
        found.append((rows, cols))

    return found


def _find_lattice(grid: np.ndarray) -> tuple[int, int, np.ndarray, np.ndarray]:
    bg = _get_bg(grid)
    candidates: set[tuple[int, tuple[int, ...], tuple[int, ...]]] = set()

    for color in _colors_present(grid, exclude={bg}):
        for rows, cols in _candidates_from_orientation(grid, bg, color, transposed=False):
            candidates.add((color, rows, cols))
        for rows, cols in _candidates_from_orientation(grid, bg, color, transposed=True):
            candidates.add((color, rows, cols))

    assert len(candidates) == 1, "expected exactly one lattice color and geometry"
    color, row_lines, col_lines = next(iter(candidates))
    return bg, int(color), np.array(row_lines, dtype=int), np.array(col_lines, dtype=int)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the missing lattice intersections as a coarse image.

    Input structure:
    - One non-background color forms equally spaced horizontal and vertical lattice lines.
    - Some line intersections are replaced by the background color, while the rest stay intact.
    - Other colored pixels may appear only inside the lattice cells as distractor markers.

    Rule:
    - Recover the lattice row and column coordinates from the repeated line template.
    - Build an output grid whose shape is the number of lattice rows by the number of lattice columns.
    - Write the lattice color where an intersection is missing in the input, and the background color where the intersection is intact.

    Key insight:
    - The picture is encoded in the negative space of the lattice: the meaningful pattern lives at the missing intersections after downsampling to lattice coordinates.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg, line_color, row_lines, col_lines = _find_lattice(g)
    intersections = g[np.ix_(row_lines, col_lines)]

    out = np.full((len(row_lines), len(col_lines)), bg, dtype=int)
    out[intersections == bg] = line_color
    return out


_MINI_SPRITES = (
    np.array([[1]], dtype=bool),
    np.array([[1, 1]], dtype=bool),
    np.array([[1], [1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1], [0, 1]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
)

_PALETTE = tuple(range(10))
_GRID_MAX_SIDE = 30
_GEN_RETRIES = 400


def _draw_segment(mask: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> None:
    r0, c0 = start
    r1, c1 = end
    steps = max(abs(r1 - r0), abs(c1 - c0))
    if steps == 0:
        mask[r0, c0] = True
        return
    for step in range(steps + 1):
        t = step / steps
        r = int(round(r0 + (r1 - r0) * t))
        c = int(round(c0 + (c1 - c0) * t))
        mask[r, c] = True


def _draw_rect_border(mask: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True


def _stair_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    corners = [((0, 0), (height - 1, width - 1)), ((0, width - 1), (height - 1, 0))]
    (r, c), (tr, tc) = random.choice(corners)
    mask[r, c] = True
    while (r, c) != (tr, tc):
        moves: list[tuple[int, int]] = []
        if r != tr:
            moves.append((r + (1 if tr > r else -1), c))
        if c != tc:
            moves.append((r, c + (1 if tc > c else -1)))
        r, c = random.choice(moves)
        mask[r, c] = True
    return mask


def _random_walk_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    r = random.randrange(height)
    c = random.randrange(width)
    mask[r, c] = True
    steps = random.randint(max(3, min(height, width)), max(4, min(height * width - 1, height + width + 2)))
    for _ in range(steps):
        neighbors = [
            (nr, nc)
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
            if 0 <= nr < height and 0 <= nc < width
        ]
        r, c = random.choice(neighbors)
        mask[r, c] = True
    return mask


def _shape_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    kind = random.choice(("cross", "x", "box", "l", "t", "u", "stairs", "walk"))

    if kind == "cross":
        mask[random.randrange(height), :] = True
        mask[:, random.randrange(width)] = True
    elif kind == "x":
        _draw_segment(mask, (0, 0), (height - 1, width - 1))
        if random.random() < 0.7:
            _draw_segment(mask, (0, width - 1), (height - 1, 0))
    elif kind == "box":
        r0 = random.randrange(max(1, height - 1))
        r1 = random.randrange(r0 + 1, height)
        c0 = random.randrange(max(1, width - 1))
        c1 = random.randrange(c0 + 1, width)
        _draw_rect_border(mask, r0, c0, r1, c1)
    elif kind == "l":
        corner = random.choice(((0, 0), (0, width - 1), (height - 1, 0), (height - 1, width - 1)))
        r0, c0 = corner
        r1 = random.randrange(height)
        c1 = random.randrange(width)
        _draw_segment(mask, (r0, c0), (r1, c0))
        _draw_segment(mask, (r0, c0), (r0, c1))
    elif kind == "t":
        orientation = random.choice(("up", "down", "left", "right"))
        if orientation in {"up", "down"}:
            row = 0 if orientation == "up" else height - 1
            stem_col = random.randrange(width)
            mask[row, :] = True
            _draw_segment(mask, (row, stem_col), (height - 1 - row, stem_col))
        else:
            col = 0 if orientation == "left" else width - 1
            stem_row = random.randrange(height)
            mask[:, col] = True
            _draw_segment(mask, (stem_row, col), (stem_row, width - 1 - col))
    elif kind == "u":
        left = random.randrange(max(1, width - 1))
        right = random.randrange(left + 1, width)
        top = random.randrange(max(1, height - 1))
        bottom = random.randrange(top + 1, height)
        side = random.choice(("top", "bottom"))
        _draw_segment(mask, (top, left), (bottom, left))
        _draw_segment(mask, (top, right), (bottom, right))
        if side == "top":
            _draw_segment(mask, (top, left), (top, right))
        else:
            _draw_segment(mask, (bottom, left), (bottom, right))
    elif kind == "stairs":
        mask = _stair_mask(height, width)
    else:
        mask = _random_walk_mask(height, width)

    if random.random() < 0.35:
        extra = random.choice(
            (
                ((0, random.randrange(width)), (height - 1, random.randrange(width))),
                ((random.randrange(height), 0), (random.randrange(height), width - 1)),
            )
        )
        _draw_segment(mask, extra[0], extra[1])
    return mask


def _sample_symbol_mask(height: int, width: int, *, min_fill: int, max_fill_ratio: float) -> np.ndarray:
    for _ in range(200):
        mask = _shape_mask(height, width)
        filled = int(mask.sum())
        if min_fill <= filled < height * width and filled <= max(1, int(round(height * width * max_fill_ratio))):
            if len(np.unique(np.argwhere(mask)[:, 0])) >= 2 or len(np.unique(np.argwhere(mask)[:, 1])) >= 2:
                return mask
    fallback = np.zeros((height, width), dtype=bool)
    fallback[height // 2, :] = True
    fallback[:, width // 2] = True
    return fallback


def _sample_geometry() -> tuple[int, int, np.ndarray, np.ndarray] | None:
    for _ in range(200):
        rows = random.randint(3, 7)
        cols = random.randint(3, 7)
        cell_h = random.randint(2, 5)
        cell_w = random.randint(2, 5)

        mode = random.choice(("interior", "one_edge", "mixed"))

        def pick_margin(limit: int, zero_ok: bool) -> int:
            if zero_ok:
                return random.randint(0, limit)
            return random.randint(1, limit)

        top = pick_margin(cell_h, mode != "interior")
        bottom = pick_margin(cell_h, mode == "mixed")
        left = pick_margin(cell_w, mode != "interior")
        right = pick_margin(cell_w, mode == "mixed")

        if mode == "one_edge":
            forced = random.choice(("top", "bottom", "left", "right"))
            if forced == "top":
                top = 0
            elif forced == "bottom":
                bottom = 0
            elif forced == "left":
                left = 0
            else:
                right = 0

        height = top + (rows - 1) * (cell_h + 1) + 1 + bottom
        width = left + (cols - 1) * (cell_w + 1) + 1 + right
        if height > _GRID_MAX_SIDE or width > _GRID_MAX_SIDE:
            continue

        row_lines = np.array([top + i * (cell_h + 1) for i in range(rows)], dtype=int)
        col_lines = np.array([left + i * (cell_w + 1) for i in range(cols)], dtype=int)
        return height, width, row_lines, col_lines
    return None


def _render_input(
    height: int,
    width: int,
    row_lines: np.ndarray,
    col_lines: np.ndarray,
    bg: int,
    line_color: int,
    missing: np.ndarray,
) -> np.ndarray:
    grid = np.full((height, width), bg, dtype=int)
    grid[row_lines, :] = line_color
    grid[:, col_lines] = line_color
    for rr, row in enumerate(row_lines):
        for cc, col in enumerate(col_lines):
            if missing[rr, cc]:
                grid[int(row), int(col)] = bg
    return grid


def _paint_sprite_in_band(
    grid: np.ndarray,
    band_rows: tuple[int, int],
    band_cols: tuple[int, int],
    color: int,
) -> None:
    r0, r1 = band_rows
    c0, c1 = band_cols
    band_h = r1 - r0 + 1
    band_w = c1 - c0 + 1
    choices = [sprite for sprite in _MINI_SPRITES if sprite.shape[0] <= band_h and sprite.shape[1] <= band_w]
    sprite = random.choice(choices)
    sr = r0 + (band_h - sprite.shape[0]) // 2
    sc = c0 + (band_w - sprite.shape[1]) // 2
    patch = grid[sr : sr + sprite.shape[0], sc : sc + sprite.shape[1]]
    patch[sprite] = color


def _paint_decoys(
    grid: np.ndarray,
    row_lines: np.ndarray,
    col_lines: np.ndarray,
    distractor_colors: list[int],
) -> None:
    row_bands = _bands(grid.shape[0], row_lines)
    col_bands = _bands(grid.shape[1], col_lines)
    if len(row_bands) < 2 or len(col_bands) < 2:
        return

    decoy_mask = _sample_symbol_mask(
        len(row_bands),
        len(col_bands),
        min_fill=max(2, (len(row_bands) + len(col_bands)) // 3),
        max_fill_ratio=0.45,
    )
    color_shift = random.randrange(len(distractor_colors))
    for br, bc in np.argwhere(decoy_mask):
        color = distractor_colors[(int(br) + 2 * int(bc) + color_shift) % len(distractor_colors)]
        _paint_sprite_in_band(grid, row_bands[int(br)], col_bands[int(bc)], color)


def _oracle_output(bg: int, line_color: int, missing: np.ndarray) -> np.ndarray:
    out = np.full(missing.shape, bg, dtype=int)
    out[missing] = line_color
    return out


def _gen_attempt() -> np.ndarray | None:
    geometry = _sample_geometry()
    if geometry is None:
        return None
    height, width, row_lines, col_lines = geometry
    rows = len(row_lines)
    cols = len(col_lines)

    missing = _sample_symbol_mask(rows, cols, min_fill=2, max_fill_ratio=0.6)
    if int(missing.sum()) == rows * cols:
        return None

    color_count = random.randint(4, 5)
    palette = random.sample(_PALETTE, color_count)
    bg = palette[0]
    line_color = palette[1]
    distractor_colors = palette[2:]

    inp = _render_input(height, width, row_lines, col_lines, bg, line_color, missing)
    _paint_decoys(inp, row_lines, col_lines, distractor_colors)

    expected = _oracle_output(bg, line_color, missing)
    if _get_bg(inp) != bg:
        return None
    if np.count_nonzero((inp != bg) & (inp != line_color)) < 3:
        return None

    try:
        got = main(inp)
    except Exception:
        return None
    if not np.array_equal(got, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid lattice-missing-intersections input")
