import random

import numpy as np


BLACK = 0
PINK = 6
LINE_PALETTE = (1, 2, 3, 4, 7, 8, 9)
QUADRANT_SIDES = (4, 5, 6)
MIN_OUTPUT_DIM = 6
MAX_OUTPUT_DIM = 12
GEN_ATTEMPTS = 400


def _split_quadrants(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert h == w and h % 2 == 0, "expected an even square grid"
    q = h // 2
    return g[:q, :q], g[:q, q:], g[q:, :q], g[q:, q:]


def _first_appearance_colors(line: np.ndarray) -> list[int]:
    seq: list[int] = []
    seen: set[int] = set()
    for value in map(int, line):
        if value != BLACK and value not in seen:
            seen.add(value)
            seq.append(value)
    return seq


def _candidate_line_starts(row: np.ndarray) -> list[int]:
    row = np.asarray(row, dtype=int)
    width = int(row.shape[0])
    starts: list[int] = []
    if width < 4 or not np.any(row != BLACK):
        return starts
    for start in range(width - 3):
        stop = start + 4
        if np.any(row[:start] != BLACK) or np.any(row[stop:] != BLACK):
            continue
        window = row[start:stop]
        positions = np.flatnonzero(window != BLACK)
        if len(positions) < 2:
            continue
        if np.all(np.diff(positions) == 1):
            continue
        if not (1 <= int(np.count_nonzero(window == BLACK)) <= 2):
            continue
        colors = _first_appearance_colors(window)
        if 2 <= len(colors) <= 3:
            starts.append(start)
    return starts


def _extract_sparse_line(lower_left: np.ndarray) -> np.ndarray:
    lower_left = np.asarray(lower_left, dtype=int)
    active_rows = np.flatnonzero(np.any(lower_left != BLACK, axis=1))
    assert len(active_rows) == 1, "expected exactly one active row in the lower-left quadrant"
    row = lower_left[int(active_rows[0])]
    starts = _candidate_line_starts(row)
    assert len(starts) == 1, "expected a unique 4-cell sparse line"
    start = int(starts[0])
    return row[start : start + 4].copy()


def _tile_diagonally(line: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    line = np.asarray(line, dtype=int)
    out = np.full((int(out_h), int(out_w)), BLACK, dtype=int)
    for step in range(out_h):
        r = out_h - 1 - step
        c0 = step
        if r < 0 or c0 >= out_w:
            break
        for offset, value in enumerate(map(int, line)):
            c = c0 + offset
            if value != BLACK and 0 <= c < out_w and out[r, c] == BLACK:
                out[r, c] = value
    return out


def _periodic_reverse_recolor(base: np.ndarray, seq: list[int]) -> np.ndarray:
    out = np.asarray(base, dtype=int).copy()
    period = len(seq)
    reverse_seq = list(reversed(seq))
    mapping = {src: dst for src, dst in zip(seq, reverse_seq)}
    for c in range(0, out.shape[1], period):
        original = out[:, c].copy()
        recolored = original.copy()
        for src, dst in mapping.items():
            recolored[original == src] = dst
        out[:, c] = recolored
    return out


def _global_reverse_recolor(base: np.ndarray, seq: list[int]) -> np.ndarray:
    out = np.asarray(base, dtype=int).copy()
    reverse_seq = list(reversed(seq))
    mapping = {src: dst for src, dst in zip(seq, reverse_seq)}
    original = out.copy()
    for src, dst in mapping.items():
        out[original == src] = dst
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Size an output from upper control counts, tile the lower sparse line along rising diagonals, then reverse-recolor only periodic columns.

    Input structure:
    - The grid is an even square split into four equal quadrants on a black background.
    - The upper-left and upper-right quadrants are control regions containing only black and pink pixels.
    - The lower-left quadrant contains exactly one active row with a unique length-4 sparse colored line.
    - The lower-right quadrant is entirely black.

    Rule:
    - Count pink pixels in the upper-left quadrant to get output height H, and pink pixels in the upper-right quadrant to get output width W.
    - Extract the unique 4-cell sparse line L from the single active row in the lower-left quadrant.
    - Place L on the bottom row of an HxW black canvas starting at the left edge, then repeat it one row up and one column right at each step, clipping at the right edge and painting only non-black cells.
    - Read the distinct non-black colors of L in first-appearance order, reverse that order, and use the resulting color mapping only on columns 0, K, 2K, ... where K is the number of distinct colors.
    - Leave every other output pixel unchanged.

    Key insight:
    - The top quadrants provide only the output dimensions, while the single sparse line controls both the diagonal trajectories and the reversed color substitution pattern applied on periodic columns.
    """
    g = np.asarray(grid, dtype=int)
    upper_left, upper_right, lower_left, lower_right = _split_quadrants(g)
    assert set(map(int, np.unique(upper_left))) <= {BLACK, PINK}
    assert set(map(int, np.unique(upper_right))) <= {BLACK, PINK}
    assert np.all(lower_right == BLACK), "expected an empty lower-right quadrant"

    out_h = int(np.count_nonzero(upper_left == PINK))
    out_w = int(np.count_nonzero(upper_right == PINK))
    assert out_h > 0 and out_w > 0, "control counts must be positive"

    line = _extract_sparse_line(lower_left)
    seq = _first_appearance_colors(line)
    base = _tile_diagonally(line, out_h, out_w)
    return _periodic_reverse_recolor(base, seq)


def _random_control_quadrant(q: int, count: int) -> np.ndarray:
    quad = np.full((q, q), BLACK, dtype=int)
    cells = random.sample(range(q * q), count)
    for idx in cells:
        r, c = divmod(idx, q)
        quad[r, c] = PINK
    return quad


def _pattern_k3(colors: list[int]) -> np.ndarray:
    zero_idx = random.choice((1, 2))
    pattern = []
    color_iter = iter(colors)
    for i in range(4):
        if i == zero_idx:
            pattern.append(BLACK)
        else:
            pattern.append(next(color_iter))
    return np.array(pattern, dtype=int)


def _pattern_k2(q: int, colors: list[int]) -> np.ndarray:
    a, b = colors
    patterns = [
        [a, BLACK, b, BLACK],
        [a, BLACK, BLACK, b],
        [BLACK, a, BLACK, b],
        [a, BLACK, b, a],
        [a, BLACK, a, b],
        [a, b, BLACK, a],
        [a, b, BLACK, b],
    ]
    if q > 4:
        patterns = [p for p in patterns if p[0] != BLACK and p[3] != BLACK]
    return np.array(random.choice(patterns), dtype=int)


def _sample_line(q: int) -> np.ndarray:
    k = random.choices((2, 3), weights=(9, 11), k=1)[0]
    colors = random.sample(LINE_PALETTE, k)
    if k == 3:
        return _pattern_k3(colors)
    return _pattern_k2(q, colors)


def _build_instance(q: int, out_h: int, out_w: int, line: np.ndarray, row_idx: int, start_col: int) -> np.ndarray:
    grid = np.full((2 * q, 2 * q), BLACK, dtype=int)
    grid[:q, :q] = _random_control_quadrant(q, out_h)
    grid[:q, q:] = _random_control_quadrant(q, out_w)
    grid[q + row_idx, start_col : start_col + 4] = line
    return grid


def _validate_instance(grid: np.ndarray) -> bool:
    upper_left, upper_right, lower_left, lower_right = _split_quadrants(grid)
    if not (set(map(int, np.unique(upper_left))) <= {BLACK, PINK}):
        return False
    if not (set(map(int, np.unique(upper_right))) <= {BLACK, PINK}):
        return False
    if not np.all(lower_right == BLACK):
        return False

    active_rows = np.flatnonzero(np.any(lower_left != BLACK, axis=1))
    if len(active_rows) != 1:
        return False
    row = lower_left[int(active_rows[0])]
    starts = _candidate_line_starts(row)
    if len(starts) != 1:
        return False

    line = _extract_sparse_line(lower_left)
    seq = _first_appearance_colors(line)
    out_h = int(np.count_nonzero(upper_left == PINK))
    out_w = int(np.count_nonzero(upper_right == PINK))
    if not (MIN_OUTPUT_DIM <= out_h <= MAX_OUTPUT_DIM and MIN_OUTPUT_DIM <= out_w <= MAX_OUTPUT_DIM):
        return False

    base = _tile_diagonally(line, out_h, out_w)
    out = _periodic_reverse_recolor(base, seq)
    if not np.array_equal(main(grid), out):
        return False
    if np.array_equal(base, out):
        return False
    if np.count_nonzero((base != BLACK) & (base != out)) == 0:
        return False
    if np.count_nonzero((base != BLACK) & (base == out)) == 0:
        return False

    period_cols = np.zeros(out_w, dtype=bool)
    period_cols[:: len(seq)] = True
    if np.count_nonzero(base[:, period_cols] != BLACK) == 0:
        return False
    if np.count_nonzero(base[:, ~period_cols] != BLACK) == 0:
        return False

    if np.array_equal(_global_reverse_recolor(base, seq), out):
        return False

    changed_sources = set(map(int, np.unique(base[(base != BLACK) & (base != out)])))
    if len(seq) == 2 and changed_sources != set(seq):
        return False
    if len(seq) == 3 and changed_sources != {seq[0], seq[2]}:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    q = random.choice(QUADRANT_SIDES)
    max_dim = min(MAX_OUTPUT_DIM, q * q)
    out_h = random.randint(MIN_OUTPUT_DIM, max_dim)
    out_w = random.randint(MIN_OUTPUT_DIM, max_dim)
    line = _sample_line(q)
    row_idx = random.randrange(q)
    start_col = random.randrange(q - 3)
    grid = _build_instance(q, out_h, out_w, line, row_idx, start_col)
    if _validate_instance(grid):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
