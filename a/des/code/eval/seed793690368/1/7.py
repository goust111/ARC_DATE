import numpy as np


BLACK = 0
YELLOW = 4
GREY = 5
BAR_COLORS = (1, 2, 3, 6, 7, 8, 9)

MIN_BAR_HEIGHT = 1
MAX_BAR_HEIGHT = 6
MIN_COLUMN_GAP = 3
GEN_RETRY_LIMIT = 300

_RNG = np.random.default_rng()
_GEN_COUNTER = 0


def _light_col(grid: np.ndarray) -> int:
    cols = np.flatnonzero(grid[0] == YELLOW)
    assert cols.size == 1, "expected exactly one yellow light on the top row"
    assert int(np.count_nonzero(grid == YELLOW)) == 1, "light must be unique"
    return int(cols[0])


def _extract_pillars(grid: np.ndarray, light_col: int) -> list[dict[str, int]]:
    height, width = grid.shape
    pillars: list[dict[str, int]] = []

    for col in range(width):
        rows = np.flatnonzero(grid[:, col] != BLACK)
        if rows.size == 0:
            continue
        if col == light_col and rows[0] == 0:
            rows = rows[1:]
        if rows.size == 0:
            continue

        colors = np.unique(grid[rows, col])
        assert colors.size == 1, "each pillar must be monochrome"
        color = int(colors[0])
        assert color not in (BLACK, YELLOW), "pillar color must differ from ground and light"

        pillar_height = int(rows.size)
        expected_rows = np.arange(height - pillar_height, height)
        assert np.array_equal(rows, expected_rows), "pillar must be bottom-aligned and contiguous"
        pillars.append({"col": col, "height": pillar_height, "color": color})

    pillars.sort(key=lambda pillar: pillar["col"])
    assert pillars, "expected at least one pillar"
    return pillars


def main(grid: np.ndarray) -> np.ndarray:
    """Draw ground shadows whose direction comes from the yellow light and whose length equals pillar height."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    height, width = g.shape
    light_col = _light_col(g)
    pillars = _extract_pillars(g, light_col)

    out = g.copy()
    ground_row = height - 1
    for pillar in pillars:
        col = pillar["col"]
        shadow_len = pillar["height"]

        if col < light_col:
            target_cols = range(col - 1, col - shadow_len - 1, -1)
        elif col > light_col:
            target_cols = range(col + 1, col + shadow_len + 1)
        else:
            continue

        for target_col in target_cols:
            assert 0 <= target_col < width, "shadow must stay inside the grid"
            if g[ground_row, target_col] == BLACK:
                out[ground_row, target_col] = GREY

    return out


def _rand_int(low: int, high: int) -> int:
    assert low <= high
    return int(_RNG.integers(low, high + 1))


def _sample_bar_color_order(count: int) -> list[int]:
    return list(map(int, _RNG.choice(np.array(BAR_COLORS), size=count, replace=False)))


def _build_grid(height: int, width: int, light_col: int, bars: list[tuple[int, int, int]]) -> np.ndarray:
    grid = np.full((height, width), BLACK, dtype=int)
    grid[0, light_col] = YELLOW
    for col, bar_height, color in bars:
        assert 0 <= col < width
        assert MIN_BAR_HEIGHT <= bar_height <= min(MAX_BAR_HEIGHT, height)
        top = height - bar_height
        assert not (col == light_col and top == 0), "pillar under the light must leave room for the light pixel"
        grid[top:, col] = color
    return grid


def _validate_generated_grid(grid: np.ndarray) -> np.ndarray:
    light_col = _light_col(grid)
    pillars = _extract_pillars(grid, light_col)
    assert len({pillar["color"] for pillar in pillars}) == len(pillars), "pillar colors must be distinct"

    cols = [pillar["col"] for pillar in pillars]
    for left, right in zip(cols, cols[1:]):
        assert right - left >= MIN_COLUMN_GAP, "pillars must stay visually separated"

    relations = {int(np.sign(pillar["col"] - light_col)) for pillar in pillars}
    assert relations == {-1, 0, 1}, "each sample must include left, aligned, and right pillars"

    out = main(grid)
    left_visible = np.any(out[-1, :light_col] == GREY)
    right_visible = np.any(out[-1, light_col + 1 :] == GREY)
    assert left_visible and right_visible, "both shadow directions must remain visible"
    return grid


def _gen_basic_attempt() -> np.ndarray | None:
    height = _rand_int(8, 14)
    width = _rand_int(13, 21)
    light_low = 6
    light_high = width - 7
    if light_low > light_high:
        return None

    light_col = _rand_int(light_low, light_high)
    left_low, left_high = 3, light_col - MIN_COLUMN_GAP
    right_low, right_high = light_col + MIN_COLUMN_GAP, width - 4
    if left_low > left_high or right_low > right_high:
        return None

    left_col = _rand_int(left_low, left_high)
    right_col = _rand_int(right_low, right_high)

    left_height = _rand_int(1, min(MAX_BAR_HEIGHT, left_col))
    same_height = _rand_int(1, min(MAX_BAR_HEIGHT, height - 1))
    right_height = _rand_int(1, min(MAX_BAR_HEIGHT, width - 1 - right_col))
    colors = _sample_bar_color_order(3)

    bars = [
        (left_col, left_height, colors[0]),
        (light_col, same_height, colors[1]),
        (right_col, right_height, colors[2]),
    ]
    return _validate_generated_grid(_build_grid(height, width, light_col, bars))


def _gen_overlap_attempt() -> np.ndarray | None:
    height = _rand_int(8, 14)
    width = _rand_int(19, 25)

    left_reach = _rand_int(3, 5)
    right_reach = _rand_int(3, 5)
    light_low = left_reach + 5
    light_high = width - right_reach - 6
    if light_low > light_high:
        return None

    light_col = _rand_int(light_low, light_high)
    left_inner_low = left_reach + 1
    left_inner_high = light_col - MIN_COLUMN_GAP
    right_inner_low = light_col + MIN_COLUMN_GAP
    right_inner_high = width - right_reach - 2
    if left_inner_low > left_inner_high or right_inner_low > right_inner_high:
        return None

    left_inner_col = _rand_int(left_inner_low, left_inner_high)
    right_inner_col = _rand_int(right_inner_low, right_inner_high)
    left_outer_col = left_inner_col - left_reach
    right_outer_col = right_inner_col + right_reach
    if left_outer_col < 1 or right_outer_col > width - 2:
        return None

    left_outer_height = _rand_int(1, min(MAX_BAR_HEIGHT, left_outer_col))
    left_inner_height = _rand_int(left_reach, min(MAX_BAR_HEIGHT, left_inner_col))
    same_height = _rand_int(1, min(MAX_BAR_HEIGHT, height - 1))
    right_inner_height = _rand_int(right_reach, min(MAX_BAR_HEIGHT, width - 1 - right_inner_col))
    right_outer_height = _rand_int(1, min(MAX_BAR_HEIGHT, width - 1 - right_outer_col))
    colors = _sample_bar_color_order(5)

    bars = [
        (left_outer_col, left_outer_height, colors[0]),
        (left_inner_col, left_inner_height, colors[1]),
        (light_col, same_height, colors[2]),
        (right_inner_col, right_inner_height, colors[3]),
        (right_outer_col, right_outer_height, colors[4]),
    ]
    return _validate_generated_grid(_build_grid(height, width, light_col, bars))


def generate_input() -> np.ndarray:
    global _GEN_COUNTER

    for _ in range(GEN_RETRY_LIMIT):
        mode = _GEN_COUNTER % 2
        _GEN_COUNTER += 1
        grid = _gen_basic_attempt() if mode == 0 else _gen_overlap_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
