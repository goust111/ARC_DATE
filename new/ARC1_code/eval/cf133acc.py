import numpy as np

from arc_utils import get_bg
import random


def _gap_columns(g: np.ndarray, bg: int) -> dict[int, list[tuple[int, int]]]:
    H, W = g.shape
    gaps: dict[int, list[tuple[int, int]]] = {}
    for r in range(H):
        for c in range(1, W - 1):
            left = int(g[r, c - 1])
            if int(g[r, c]) == bg and left != bg and left == int(g[r, c + 1]):
                gaps.setdefault(c, []).append((r, left))
    return gaps


def main(grid: np.ndarray) -> np.ndarray:
    """Extend broken horizontal bars vertically through their gap columns.

    Input structure
    - The background is empty color, and each control cell is a one-pixel gap
      inside a horizontal bar: its immediate left and right neighbors have the
      same non-background color.
    - Gap cells in the same column form a top-to-bottom stack, with a
      non-background vertical marker color somewhere below the last gap.
    Rule
    - For each gap column, sort its gaps by row. Paint from the top through each
      gap row with that row's horizontal-bar color, then paint the remaining
      lower part of the column with the marker color found below the last gap.
    Key insight
    - A gap column is an implicit vertical color ruler: horizontal bars provide
      the upper breakpoints, and the lower marker provides the final segment.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = get_bg(g)
    gaps = _gap_columns(g, bg)
    assert gaps

    out = g.copy()
    H, _ = g.shape
    for c, entries in gaps.items():
        entries = sorted(entries)
        rows = [r for r, _ in entries]
        assert len(rows) == len(set(rows))

        last_gap_row = rows[-1]
        assert last_gap_row + 1 < H
        terminal_colors = {int(x) for x in g[last_gap_row + 1 :, c] if int(x) != bg}
        assert len(terminal_colors) == 1
        terminal_color = terminal_colors.pop()

        start = 0
        for r, color in entries:
            assert all(int(x) in (bg, color) for x in g[start : r + 1, c])
            out[start : r + 1, c] = color
            start = r + 1

        assert all(int(x) in (bg, terminal_color) for x in g[start:, c])
        out[start:, c] = terminal_color

    return out

# --------------- generator (o5x) ---------------

# Canvas, palette, and retry limits.
GRID_H, GRID_W, BG = 15, 15, 0
COLOR_POOL = range(1, 10)
MIN_COLORS, MAX_COLORS, MAX_ATTEMPTS = 3, 7, 300

# Gap columns, bar rows, and side order.
GAP_COUNT_MIN, GAP_COUNT_MAX = 1, 3
GAP_DISTANCE_MIN, GAP_DISTANCE_MAX = 4, 8
INTERIOR_COL_MIN, INTERIOR_COL_MAX = 1, 13
BOTTOM_BELOW_MIN, BOTTOM_BELOW_MAX = 1, 8
SIDE_LEFT, SIDE_RIGHT = "L", "R"

# Horizontal span offsets from the two gap columns.
LEFT_END_MIN, LEFT_END_BEFORE_C2 = 2, 1
RIGHT_START_AFTER_C1, RIGHT_START_BEFORE_C2 = 1, 2
SPLIT_AFTER_C1, SPLIT_BEFORE_C2 = 1, 2

# Terminal-marker gap and length bounds.
MARKER_GAP_MIN, MARKER_GAP_MAX = 0, 2
MARKER_LEN_MIN, MARKER_LEN_MAX = 1, 6

def _gen_color_sequences(n: int) -> tuple[list[int], list[int]]:
    total = random.randint(max(MIN_COLORS, n + 1), min(MAX_COLORS, 2 * n + 2))
    palette = random.sample(COLOR_POOL, total)

    left_seq = random.sample(palette, n + 1)
    missing_from_left = [color for color in palette if color not in left_seq]
    right_seq = missing_from_left + random.sample(left_seq, n + 1 - len(missing_from_left))
    random.shuffle(right_seq)
    return left_seq, right_seq


def _gen_marker_start(bottom_row: int) -> int:
    total_below = GRID_H - 1 - bottom_row
    gaps = [
        gap
        for gap in range(MARKER_GAP_MIN, MARKER_GAP_MAX + 1)
        if MARKER_LEN_MIN <= total_below - gap <= MARKER_LEN_MAX
    ]
    gap = random.choice(gaps)
    return bottom_row + 1 + gap


def _gen_fill_column(
    out: np.ndarray, col: int, gap_rows: list[int], colors: list[int]
) -> None:
    start = 0
    for row, color in zip(gap_rows, colors[:-1]):
        out[start : row + 1, col] = color
        start = row + 1
    out[start:, col] = colors[-1]


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(GAP_COUNT_MIN, GAP_COUNT_MAX)
    distance = random.randint(GAP_DISTANCE_MIN, GAP_DISTANCE_MAX)
    c1 = random.randint(INTERIOR_COL_MIN, INTERIOR_COL_MAX - distance)
    c2 = c1 + distance

    below_count = random.randint(BOTTOM_BELOW_MIN, BOTTOM_BELOW_MAX)
    bottom_row = GRID_H - 1 - below_count
    above_rows = sorted(random.sample(range(bottom_row), 2 * (n - 1)))

    first_side = random.choice([SIDE_LEFT, SIDE_RIGHT])
    second_side = SIDE_RIGHT if first_side == SIDE_LEFT else SIDE_LEFT
    sides = [first_side if i % 2 == 0 else second_side for i in range(len(above_rows))]
    left_gap_rows = [row for row, side in zip(above_rows, sides) if side == SIDE_LEFT]
    right_gap_rows = [row for row, side in zip(above_rows, sides) if side == SIDE_RIGHT]
    left_gap_rows.append(bottom_row)
    right_gap_rows.append(bottom_row)

    left_colors, right_colors = _gen_color_sequences(n)

    out = np.zeros((GRID_H, GRID_W), dtype=int)
    _gen_fill_column(out, c1, left_gap_rows, left_colors)
    _gen_fill_column(out, c2, right_gap_rows, right_colors)

    for row, color in zip(left_gap_rows[:-1], left_colors[:-2]):
        end_col = random.randint(c1 + LEFT_END_MIN, c2 - LEFT_END_BEFORE_C2)
        out[row, : end_col + 1] = color

    for row, color in zip(right_gap_rows[:-1], right_colors[:-2]):
        start_col = random.randint(c1 + RIGHT_START_AFTER_C1, c2 - RIGHT_START_BEFORE_C2)
        out[row, start_col:] = color

    bottom_left = left_colors[-2]
    bottom_right = right_colors[-2]
    if bottom_left == bottom_right:
        out[bottom_row, :] = bottom_left
    else:
        split_col = random.randint(c1 + SPLIT_AFTER_C1, c2 - SPLIT_BEFORE_C2)
        out[bottom_row, : split_col + 1] = bottom_left
        out[bottom_row, split_col + 1 :] = bottom_right

    inp = out.copy()
    inp[:, c1] = BG
    inp[:, c2] = BG

    left_marker_start = _gen_marker_start(bottom_row)
    right_marker_start = _gen_marker_start(bottom_row)
    inp[left_marker_start:, c1] = left_colors[-1]
    inp[right_marker_start:, c2] = right_colors[-1]

    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        generated = _gen_attempt()
        if generated is not None:
            return generated
    raise RuntimeError("failed to generate a valid cf133acc input")
