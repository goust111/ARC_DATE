import numpy as np

from arc_utils import Color, colors_present, get_bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split two side-by-side masks and mark the cells that are empty in both.

    Input structure
    - The input is a 2D grid with black background.
    - The grid has even width and splits into two equal-size panels.
    - The left panel contains a pink mask and the right panel contains a grey mask.
    Rule
    - Align the two panels cell by cell.
    - Output one panel-sized black grid.
    - Paint a yellow pixel wherever both aligned input cells are background.
    Key insight
    - The output is the intersection of the two background masks after horizontal alignment.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height > 0 and width > 0, "expected a nonempty grid"
    assert width % 2 == 0, "expected two equal-width panels"

    bg = get_bg(g)
    assert bg == Color.BLACK, "expected black background"

    half = width // 2
    left = g[:, :half]
    right = g[:, half:]
    assert left.shape == right.shape, "expected aligned left and right panels"

    left_colors = set(colors_present(left, exclude={bg}))
    right_colors = set(colors_present(right, exclude={bg}))
    assert left_colors == {Color.PINK}, "expected a pink mask in the left panel"
    assert right_colors == {Color.GREY}, "expected a grey mask in the right panel"

    out = np.full(left.shape, Color.BLACK, dtype=int)
    out[(left == bg) & (right == bg)] = Color.YELLOW
    return out

# --------------- generator (o5x) ---------------
# Canvas and panel bounds.
MIN_HEIGHT, MAX_HEIGHT, MIN_PANEL_WIDTH, MAX_PANEL_WIDTH = 3, 6, 3, 6
# Output and foreground density bounds, expressed as integer percentages.
MIN_YELLOW_PCT, MAX_YELLOW_PCT, MIN_FOREGROUND_PCT, MAX_FOREGROUND_PCT = 15, 45, 30, 70
PCT_DENOMINATOR = 100
# Retry budget.
MAX_GENERATION_ATTEMPTS = 300


def _ceil_pct(total: int, pct: int) -> int:
    return (pct * total + PCT_DENOMINATOR - 1) // PCT_DENOMINATOR


def _feasible_foreground_counts(total: int, active_count: int) -> list[tuple[int, int]]:
    """Return all left/right foreground counts that satisfy panel density bounds."""
    min_fg = _ceil_pct(total, MIN_FOREGROUND_PCT)
    max_fg = min((MAX_FOREGROUND_PCT * total) // PCT_DENOMINATOR, active_count)
    return [
        (left_count, right_count)
        for left_count in range(min_fg, max_fg + 1)
        for right_count in range(min_fg, max_fg + 1)
        if left_count + right_count >= active_count
        and 2 * total - left_count - right_count > max(left_count, right_count)
    ]


def _gen_attempt() -> np.ndarray | None:
    height = int(np.random.randint(MIN_HEIGHT, MAX_HEIGHT + 1))
    width = int(np.random.randint(MIN_PANEL_WIDTH, MAX_PANEL_WIDTH + 1))
    total = height * width

    min_yellow = max(1, _ceil_pct(total, MIN_YELLOW_PCT))
    max_yellow = max(min_yellow, (MAX_YELLOW_PCT * total) // PCT_DENOMINATOR)
    yellow_count = int(np.random.randint(min_yellow, max_yellow + 1))

    order = np.arange(total)
    np.random.shuffle(order)
    yellow_cells = order[:yellow_count]
    active_cells = order[yellow_count:]
    active_count = total - yellow_count

    feasible_counts = _feasible_foreground_counts(total, active_count)
    if not feasible_counts:
        return None

    left_count, right_count = feasible_counts[int(np.random.randint(0, len(feasible_counts)))]
    both_count = left_count + right_count - active_count

    np.random.shuffle(active_cells)
    both_cells = active_cells[:both_count]
    left_only_count = left_count - both_count
    left_only_end = both_count + left_only_count
    left_only_cells = active_cells[both_count:left_only_end]
    right_only_cells = active_cells[left_only_end:]

    out_flat = np.full(total, Color.BLACK, dtype=int)
    out_flat[yellow_cells] = Color.YELLOW
    expected = out_flat.reshape(height, width)

    left_flat = np.full(total, Color.BLACK, dtype=int)
    right_flat = np.full(total, Color.BLACK, dtype=int)
    left_flat[both_cells] = Color.PINK
    right_flat[both_cells] = Color.GREY
    left_flat[left_only_cells] = Color.PINK
    right_flat[right_only_cells] = Color.GREY

    left, right = left_flat.reshape(height, width), right_flat.reshape(height, width)
    inp = np.concatenate([left, right], axis=1)

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a valid e345f17b input")
