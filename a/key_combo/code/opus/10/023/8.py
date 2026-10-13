import random

import numpy as np


BLACK = 0
GREY = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]

SIZE_OPTIONS = (4, 5)
MAX_ATTEMPTS = 500


def _shift_rows(grid: np.ndarray, shift: int) -> np.ndarray:
    return np.roll(grid, int(shift), axis=0)


def _expected_output(grid: np.ndarray) -> np.ndarray:
    h, w = map(int, grid.shape)
    out = np.zeros((h * h, w * w), dtype=int)
    for r in range(h):
        shifted = _shift_rows(grid, r)
        for c in range(w):
            if int(grid[r, c]) == GREY:
                out[r * h : (r + 1) * h, c * w : (c + 1) * w] = shifted
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp row-shifted copies of the input pattern into blocks selected by grey pixels.

    Input structure:
    - The input is a small 2D grid containing grey control pixels, black background cells,
      and non-grey colored motif pixels.
    - Each input cell corresponds to one output block of the same height and width as the input.

    Rule:
    - The output has shape (H*H, W*W) for an input of shape (H, W).
    - For a grey pixel at input position (r, c), stamp the whole input grid into output block
      (r, c) after cyclically shifting the input downward by exactly r rows.
    - For every non-grey input pixel, leave the corresponding output block black.
    - Grey pixels inside stamped copies remain grey and do not trigger recursive stamping.

    Key insight:
    - A grey pixel carries two visible roles at once: it selects which block receives a stamp,
      and its row index parameterizes the cyclic row offset of that stamped copy.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    assert np.all((0 <= g) & (g <= 9)), "expected ARC color ids"

    return _expected_output(g)


def _grid_signature(grid: np.ndarray) -> tuple[int, ...]:
    return tuple(int(x) for x in grid.ravel())


def _choose_grey_layout(h: int, w: int) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)

    same_row = random.randrange(h)
    same_cols = random.sample(range(w), k=random.randint(2, min(3, w)))
    for c in same_cols:
        grid[same_row, c] = GREY

    other_count = random.randint(1, min(2, h - 1))
    other_rows = random.sample([r for r in range(h) if r != same_row], k=other_count)
    for r in other_rows:
        grey_cols = random.sample(range(w), k=random.randint(1, min(2, w - 1)))
        for c in grey_cols:
            grid[r, c] = GREY

    if random.random() < 0.5:
        extra = random.randint(0, max(0, h - 2))
        empties = [(r, c) for r in range(h) for c in range(w) if grid[r, c] == BLACK]
        random.shuffle(empties)
        for r, c in empties[:extra]:
            grid[r, c] = GREY

    return grid


def _paint_colored_rows(base: np.ndarray) -> np.ndarray | None:
    h, w = map(int, base.shape)
    grid = base.copy()

    palette = random.sample(COLOR_POOL, k=random.randint(3, min(5, len(COLOR_POOL))))
    used_colors: set[int] = set()
    seen_rows: set[tuple[int, ...]] = set()

    for r in range(h):
        free_cols = [c for c in range(w) if grid[r, c] == BLACK]
        if not free_cols:
            return None

        min_cells = 1 if len(free_cols) == 1 else 2
        colored_count = random.randint(min_cells, min(3, len(free_cols)))
        cols = random.sample(free_cols, k=colored_count)

        row_values = [random.choice(palette) for _ in range(colored_count)]
        if colored_count >= 2 and len(set(row_values)) == 1 and len(palette) >= 2:
            a, b = random.sample(palette, 2)
            row_values[0] = a
            row_values[-1] = b

        for c, value in zip(cols, row_values):
            grid[r, c] = value
            used_colors.add(int(value))

        row_sig = tuple(int(x) for x in grid[r])
        if row_sig in seen_rows:
            return None
        seen_rows.add(row_sig)

    if len(used_colors) < 2:
        return None
    return grid


def _validate_semantics(grid: np.ndarray, output: np.ndarray | None = None) -> bool:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        return False

    h, w = map(int, g.shape)
    if h not in SIZE_OPTIONS or w not in SIZE_OPTIONS:
        return False
    if h * h > 30 or w * w > 30:
        return False

    grey_pos = np.argwhere(g == GREY)
    if grey_pos.shape[0] < 3:
        return False

    row_counts = np.bincount(grey_pos[:, 0], minlength=h)
    grey_rows = np.flatnonzero(row_counts)
    if grey_rows.size < 2 or int(row_counts.max()) < 2:
        return False

    colored_mask = (g != BLACK) & (g != GREY)
    if int(colored_mask.sum()) < max(6, h):
        return False
    if len({int(x) for x in g[colored_mask]}) < 2:
        return False
    if any(int(np.count_nonzero(colored_mask[r])) == 0 for r in range(h)):
        return False

    row_sigs = [tuple(int(x) for x in g[r]) for r in range(h)]
    if len(set(row_sigs)) != h:
        return False

    roll_sigs = {_grid_signature(_shift_rows(g, shift)) for shift in range(h)}
    if len(roll_sigs) != h:
        return False

    expected = _expected_output(g)
    if output is None:
        output = main(g)
    out = np.asarray(output, dtype=int)
    if out.shape != expected.shape or not np.array_equal(out, expected):
        return False

    for r in range(h):
        shifted = _shift_rows(g, r)
        for c in range(w):
            block = out[r * h : (r + 1) * h, c * w : (c + 1) * w]
            if int(g[r, c]) == GREY:
                if not np.array_equal(block, shifted):
                    return False
            elif np.any(block != BLACK):
                return False

    same_row = int(np.flatnonzero(row_counts >= 2)[0])
    same_cols = np.flatnonzero(g[same_row] == GREY)
    ref_block = out[same_row * h : (same_row + 1) * h, same_cols[0] * w : (same_cols[0] + 1) * w]
    for c in same_cols[1:]:
        other_block = out[same_row * h : (same_row + 1) * h, c * w : (c + 1) * w]
        if not np.array_equal(ref_block, other_block):
            return False

    row_a = int(grey_rows[0])
    row_b = int(grey_rows[1])
    col_a = int(np.flatnonzero(g[row_a] == GREY)[0])
    col_b = int(np.flatnonzero(g[row_b] == GREY)[0])
    block_a = out[row_a * h : (row_a + 1) * h, col_a * w : (col_a + 1) * w]
    block_b = out[row_b * h : (row_b + 1) * h, col_b * w : (col_b + 1) * w]
    if np.array_equal(block_a, block_b):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.choice(SIZE_OPTIONS)
    w = random.choice(SIZE_OPTIONS)

    for _ in range(100):
        grey_layout = _choose_grey_layout(h, w)
        grid = _paint_colored_rows(grey_layout)
        if grid is None:
            continue
        if _validate_semantics(grid):
            return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle-8 input")
