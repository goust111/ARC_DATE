from __future__ import annotations
import numpy as np
import random


def _run_starts(g: np.ndarray, axis: int) -> list[int]:
    length = g.shape[axis]
    assert length > 0, "cannot compress an empty axis"

    starts = [0]
    for i in range(1, length):
        if axis == 0:
            prev = g[i - 1, :]
            cur = g[i, :]
        else:
            prev = g[:, i - 1]
            cur = g[:, i]
        if not np.array_equal(cur, prev):
            starts.append(i)
    return starts


def main(grid: np.ndarray) -> np.ndarray:
    """Collapse each run of identical neighboring rows and columns to one representative.

    Input structure:
    - The grid is a stretched small pattern: foreground bars make some rows and columns
      distinctive, while the blank bands between them duplicate adjacent row or column
      vectors.
    - Repeated bands are contiguous; the colors themselves do not define special cases.

    Rule:
    - Scan rows from top to bottom and keep the first row of each consecutive run of
      identical rows.
    - Scan columns from left to right and keep the first column of each consecutive run of
      identical columns.
    - The output is the intersection of those kept row and column representatives.

    Key insight:
    - The large image is just a run-length-expanded version of the target pattern along
      both axes, so removing adjacent duplicate row and column vectors recovers it.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "input must be a 2-D grid"
    assert g.shape[0] > 0 and g.shape[1] > 0, "input grid must be nonempty"

    row_starts = _run_starts(g, axis=0)
    col_starts = _run_starts(g, axis=1)
    assert len(row_starts) < g.shape[0], "input must contain repeated row bands"
    assert len(col_starts) < g.shape[1], "input must contain repeated column bands"

    out = g[np.ix_(row_starts, col_starts)].copy()
    assert out.shape == (len(row_starts), len(col_starts))
    return out

# --------------- generator (o5x) ---------------

# Bar count ranges.
H_BAR_MIN, H_BAR_MAX = 2, 3
V_BAR_MIN, V_BAR_MAX = 1, 4

# Expanded input dimensions and separator run lengths.
ROW_DIM_MIN, ROW_DIM_MAX = 10, 12
COL_DIM_MIN, COL_DIM_MAX = 9, 13
SEP_RUN_MIN, SEP_RUN_MAX = 1, 6

# Palette and retry policy.
BG_COLOR = 0
BAR_COLORS = tuple(range(1, 9))
GEN_MAX_ATTEMPTS = 300


def _gen_bounded_composition(
    total: int, parts: int, *, lo: int = SEP_RUN_MIN, hi: int = SEP_RUN_MAX
) -> list[int]:
    """Sample positive run lengths summing to total within inclusive bounds."""
    vals: list[int] = []
    remaining = total
    for i in range(parts):
        left = parts - i - 1
        low = max(lo, remaining - hi * left)
        high = min(hi, remaining - lo * left)
        vals.append(random.randint(low, high))
        remaining -= vals[-1]
    random.shuffle(vals)
    return vals


def _gen_separator_runs(n_bars: int, *, min_dim: int, max_dim: int) -> list[int]:
    """Choose separator run lengths for one axis with at least one stretched gap."""
    n_separators = n_bars + 1
    valid_dims = [
        dim
        for dim in range(min_dim, max_dim + 1)
        if n_separators * SEP_RUN_MIN < dim - n_bars <= SEP_RUN_MAX * n_separators
    ]
    if not valid_dims:
        raise ValueError("no valid expanded dimension")
    total_dim = random.choice(valid_dims)
    return _gen_bounded_composition(total_dim - n_bars, n_separators)


def _gen_seed(n_h: int, n_v: int) -> np.ndarray:
    """Build the compressed bar grid before row/column expansion."""
    colors = random.sample(BAR_COLORS, n_h + n_v)
    h_colors = colors[:n_h]
    v_colors = colors[n_h:]

    bars = [("h", i) for i in range(n_h)] + [("v", i) for i in range(n_v)]
    random.shuffle(bars)
    depth = {bar: rank for rank, bar in enumerate(bars)}

    seed = np.full((2 * n_h + 1, 2 * n_v + 1), BG_COLOR, dtype=int)
    for i, color in enumerate(v_colors):
        seed[:, 2 * i + 1] = color

    for i, h_color in enumerate(h_colors):
        r = 2 * i + 1
        seed[r, :] = h_color
        for j, v_color in enumerate(v_colors):
            if depth[("v", j)] > depth[("h", i)]:
                seed[r, 2 * j + 1] = v_color

    return seed


def _gen_expand_seed(seed: np.ndarray, row_runs: list[int], col_runs: list[int]) -> np.ndarray:
    """Repeat only separator rows/columns; bar rows/columns stay one pixel wide."""
    row_repeats = [row_runs[r // 2] if r % 2 == 0 else 1 for r in range(seed.shape[0])]
    col_repeats = [col_runs[c // 2] if c % 2 == 0 else 1 for c in range(seed.shape[1])]

    return np.repeat(np.repeat(seed, row_repeats, axis=0), col_repeats, axis=1)


def _gen_attempt() -> np.ndarray | None:
    n_h = random.randint(H_BAR_MIN, H_BAR_MAX)
    n_v = random.randint(V_BAR_MIN, V_BAR_MAX)

    seed = _gen_seed(n_h, n_v)
    row_runs = _gen_separator_runs(n_h, min_dim=ROW_DIM_MIN, max_dim=ROW_DIM_MAX)
    col_runs = _gen_separator_runs(n_v, min_dim=COL_DIM_MIN, max_dim=COL_DIM_MAX)
    inp = _gen_expand_seed(seed, row_runs, col_runs)

    if not np.array_equal(main(inp), seed):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate valid input")
