import numpy as np

from arc_utils import Color, get_bg


def main(grid: np.ndarray) -> np.ndarray:
    """Move each red vertical bar upward in its column until it rests directly under the blue pixels above it.

    Input structure
    - The grid has a black background, an upper blue shape, and lower red vertical bars.
    - Every column containing red has one contiguous red segment below at least one blue pixel.

    Rule
    - Clear the red cells from their starting positions.
    - In each red column, preserve the red segment height and repaint it starting one row below the lowest blue pixel in that same column.

    Key insight
    - Each red bar slides straight upward along its own column; the blue pixels in that column define the stopping position.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    assert bg == Color.BLACK
    assert set(map(int, np.unique(g))) == {Color.BLACK, Color.BLUE, Color.RED}

    h, _w = g.shape
    red_cols = np.flatnonzero(np.any(g == Color.RED, axis=0))
    assert red_cols.size > 0

    out = g.copy()
    out[g == Color.RED] = bg

    for c in red_cols:
        red_rows = np.flatnonzero(g[:, c] == Color.RED)
        assert red_rows.size > 0
        assert np.all(np.diff(red_rows) == 1)

        blue_rows = np.flatnonzero(g[:, c] == Color.BLUE)
        assert blue_rows.size > 0
        assert blue_rows.max() < red_rows[0]

        dest_top = int(blue_rows.max()) + 1
        dest_bottom = dest_top + int(red_rows.size)
        assert dest_bottom <= h
        assert np.all(g[dest_top:dest_bottom, c] != Color.BLUE)

        out[dest_top:dest_bottom, c] = Color.RED

    return out

# --------------- generator (o5x) ---------------

# Canvas and palette.
GRID_H, GRID_W = 10, 10
BLUE, RED, BG = Color.BLUE, Color.RED, Color.BLACK

# Blue ceiling geometry.
BLUE_WIDTH_RANGE, BLUE_MAX_HEIGHT_RANGE = (5, 10), (3, 6)
SIDE_MARGIN_MAX = 3

# Red bars and rejection sampling.
RED_COUNT_MIN, RED_COUNT_CAP = 1, 5
RED_BAR_HEIGHT_RANGE, INPUT_GAP_MIN = (1, 7), 3
GEN_RETRIES = 300

def _sample_scene():
    """Return sampled columns and heights, or reject impossible notch/gap constraints."""
    width = int(np.random.randint(*BLUE_WIDTH_RANGE))
    last_start = GRID_W - width
    starts = [s for s in range(last_start + 1) if max(s, last_start - s) <= SIDE_MARGIN_MAX]
    c0 = int(np.random.choice(starts))
    cols = list(range(c0, c0 + width))

    max_blue_h = int(np.random.randint(*BLUE_MAX_HEIGHT_RANGE))
    red_count = int(np.random.randint(RED_COUNT_MIN, min(RED_COUNT_CAP, width - 1) + 1))
    red_cols = sorted(np.random.choice(cols, size=red_count, replace=False).tolist())
    red_set = set(red_cols)
    non_red_cols = [c for c in cols if c not in red_set]
    blue_heights, bar_heights = {}, {}
    for c in red_cols:
        bar_h = int(np.random.randint(*RED_BAR_HEIGHT_RANGE))
        max_notch_h = min(max_blue_h - 1, GRID_H - INPUT_GAP_MIN - bar_h)
        if max_notch_h < 1:
            return None
        blue_heights[c] = int(np.random.randint(1, max_notch_h + 1))
        bar_heights[c] = bar_h

    for c in non_red_cols:
        blue_heights[c] = int(np.random.randint(1, max_blue_h + 1))
    blue_heights[int(np.random.choice(non_red_cols))] = max_blue_h
    return cols, red_cols, blue_heights, bar_heights

def _render_pair(cols, red_cols, blue_heights, bar_heights):
    """Build input/output grids, rejecting bottom-aligned bars with too-small gaps."""
    out = np.full((GRID_H, GRID_W), BG, dtype=int)
    for c in cols:
        out[: blue_heights[c], c] = BLUE
    for c in red_cols:
        top = blue_heights[c]
        out[top : top + bar_heights[c], c] = RED

    inp = out.copy()
    inp[inp == RED] = BG
    for c in red_cols:
        top = GRID_H - bar_heights[c]
        if top - blue_heights[c] < INPUT_GAP_MIN:
            return None
        inp[top:GRID_H, c] = RED
    return inp, out

def _gen_attempt() -> np.ndarray | None:
    scene = _sample_scene()
    if scene is None:
        return None
    grids = _render_pair(*scene)
    if grids is None:
        return None
    inp, out = grids
    try:
        return inp if np.array_equal(main(inp), out) else None
    except Exception:
        return None

def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        if (grid := _gen_attempt()) is not None:
            return grid
    raise RuntimeError("failed to generate valid 3906de3d input")
