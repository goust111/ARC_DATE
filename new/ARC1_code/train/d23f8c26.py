import numpy as np

from arc_utils import get_bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep only the input's center vertical column and erase every other cell to the background.

    Input structure
    - A 2-D grid with an odd number of columns and a dominant background color.
    - Foreground pixels may appear anywhere, but only the middle column is relevant.

    Rule
    - Create a same-size canvas filled with the background color.
    - Copy the input middle column unchanged into the output.

    Key insight
    - The examples preserve every value in the vertical center column and delete all off-center foreground pixels.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h > 0 and w > 0
    assert w % 2 == 1

    bg = get_bg(g)
    mid = w // 2

    out = np.full_like(g, bg)
    out[:, mid] = g[:, mid]
    return out

# --------------- generator (o5x) ---------------

# Grid, color, and palette controls.
GRID_SIZES, BACKGROUND_COLOR = [3, 5, 7], 0
FOREGROUND_COLORS, COLOR_COUNT_OPTIONS = np.arange(1, 10), [5, 6]
# Foreground count and density controls.
TOTAL_DENSITY_RANGE, OFF_DENSITY_RANGE = (0.35, 0.56), (0.30, 0.50)
MIN_TOTAL_FG, MIN_OFF_FG = 5, 3
MID_FG_RANGE = (2, 4)
# Retry controls.
GENERATION_ATTEMPTS = 300

def _count_plan(n: int) -> tuple[int, int, int] | None:
    """Choose total, middle-column, and off-center foreground counts."""
    cells = n * n
    total_min = max(int(np.ceil(TOTAL_DENSITY_RANGE[0] * cells)), MIN_TOTAL_FG, MID_FG_RANGE[0] + MIN_OFF_FG)
    total_max = int(np.floor(TOTAL_DENSITY_RANGE[1] * cells))
    if total_min > total_max:
        return None
    total_fg = int(np.random.randint(total_min, total_max + 1))
    off_cells = n * (n - 1)
    off_pref_min = int(np.ceil(OFF_DENSITY_RANGE[0] * off_cells))
    off_pref_max = int(np.floor(OFF_DENSITY_RANGE[1] * off_cells))
    mid_min, mid_max = MID_FG_RANGE
    mid_options = [k for k in range(mid_min, min(mid_max, n - 1) + 1) if MIN_OFF_FG <= total_fg - k <= off_cells]
    if not mid_options:
        return None
    preferred = [k for k in mid_options if off_pref_min <= total_fg - k <= off_pref_max]
    mid_count = (
        int(np.random.choice(preferred)) if preferred else
        min(mid_options, key=lambda k: min(abs(total_fg - k - off_pref_min), abs(total_fg - k - off_pref_max)))
    )
    return total_fg, mid_count, total_fg - mid_count

def _sample_coords(n: int, mid: int, mid_count: int, off_count: int) -> list[tuple[int, int]]:
    """Sample and shuffle the foreground coordinates for the planned counts."""
    mid_rows = np.random.choice(n, size=mid_count, replace=False)
    off_positions = [(r, c) for r in range(n) for c in range(n) if c != mid]
    off_idx = np.random.choice(len(off_positions), size=off_count, replace=False)
    coords = [(int(r), mid) for r in mid_rows]
    coords.extend(off_positions[int(i)] for i in off_idx)
    np.random.shuffle(coords)
    return coords

def _sample_values(total_fg: int) -> np.ndarray:
    """Sample and shuffle foreground colors while preserving 5/6 color coverage."""
    n_colors = MIN_TOTAL_FG if total_fg == MIN_TOTAL_FG else int(np.random.choice(COLOR_COUNT_OPTIONS))
    palette = np.random.choice(FOREGROUND_COLORS, size=n_colors, replace=False)
    values = np.array([palette[i % n_colors] for i in range(total_fg)], dtype=int)
    np.random.shuffle(values)
    return values

def _gen_attempt() -> np.ndarray | None:
    n = int(np.random.choice(GRID_SIZES))
    plan = _count_plan(n)
    if plan is None:
        return None
    total_fg, mid_count, off_count = plan
    mid = n // 2

    coords = _sample_coords(n, mid, mid_count, off_count)
    values = _sample_values(total_fg)
    grid = np.full((n, n), BACKGROUND_COLOR, dtype=int)
    rows, cols = np.array(coords, dtype=int).T
    grid[rows, cols] = values

    expected = np.full_like(grid, BACKGROUND_COLOR)
    expected[:, mid] = grid[:, mid]
    return grid if np.array_equal(main(grid), expected) else None

def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid d23f8c26 input")
