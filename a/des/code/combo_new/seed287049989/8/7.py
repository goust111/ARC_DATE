import random

import numpy as np


BLACK = 0
SEPARATOR_COLOR = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_RETRIES = 300


def _nonblack_runs(row: np.ndarray) -> list[tuple[int, int, int]]:
    runs: list[tuple[int, int, int]] = []
    c = 0
    width = int(row.shape[0])
    while c < width:
        if int(row[c]) == BLACK:
            c += 1
            continue
        color = int(row[c])
        start = c
        while c < width and int(row[c]) == color:
            c += 1
        runs.append((start, c, color))
    return runs


def _parse_legend_row(row: np.ndarray) -> tuple[int, int] | None:
    nonblack = row[row != BLACK]
    if nonblack.size == 0:
        return None
    colors = {int(v) for v in nonblack}
    if len(colors) != 1:
        return None
    runs = _nonblack_runs(row)
    if len(runs) != 2:
        return None
    (s1, e1, color1), (s2, e2, color2) = runs
    if color1 != color2 or e1 - s1 != 1:
        return None
    if s2 <= e1:
        return None
    count = e2 - s2
    if not (1 <= count <= 5):
        return None
    return count, color1


def _extract_legend_mapping(region: np.ndarray) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for r in range(int(region.shape[0])):
        parsed = _parse_legend_row(region[r])
        if parsed is None:
            continue
        count, color = parsed
        if count in mapping and mapping[count] != color:
            raise AssertionError("ambiguous legend count")
        mapping[count] = color
    return mapping


def _find_stripe_rows(region: np.ndarray) -> list[tuple[int, int]]:
    rows: list[tuple[int, int]] = []
    for r in range(int(region.shape[0])):
        vals = {int(v) for v in region[r] if int(v) != BLACK}
        if not vals:
            continue
        if len(vals) != 1:
            return []
        rows.append((r, vals.pop()))
    if not rows:
        return []
    row_ids = [r for r, _color in rows]
    if len(row_ids) >= 2 and any(b - a != 2 for a, b in zip(row_ids, row_ids[1:])):
        return []
    for r in range(row_ids[0], row_ids[-1] + 1):
        if r in row_ids:
            continue
        if np.any(region[r] != BLACK):
            return []
    return rows


def _find_separator(grid: np.ndarray) -> int:
    height, width = map(int, grid.shape)
    best_col = -1
    best_score = -1
    for c in range(1, width - 1):
        col = grid[:, c]
        if not np.all(col == col[0]):
            continue
        if int(col[0]) == BLACK:
            continue
        left = grid[:, :c]
        right = grid[:, c + 1 :]
        stripes = _find_stripe_rows(left)
        if not stripes:
            continue
        legend = _extract_legend_mapping(right)
        if not legend:
            continue
        score = len(stripes) + len(legend)
        if score > best_score:
            best_score = score
            best_col = c
    if best_col < 0:
        raise AssertionError("could not locate separator column")
    return best_col


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor horizontal stripe rows by matching their hole counts to a separate color legend.

    Input structure:
    - The grid is split into left and right regions by one full-height grey separator column.
    - The left region contains several 1-pixel-tall colored horizontal stripes, with 1 black row between consecutive stripes.
    - Each stripe spans the full left-region width and has a distinct number of black holes punched into it.
    - The right region contains a legend: each active row has one colored pixel, a black gap, then N pixels of that same color, where N is between 1 and 5.

    Rule:
    - For each stripe on the left, count how many black holes it contains.
    - Find the legend row on the right whose trailing same-color run has that exact count.
    - Replace the entire stripe, including its holes, with the legend row's color.
    - Leave the grey separator and the right-side legend unchanged.

    Key insight:
    - The stripe's original color is irrelevant: only the number of black holes matters, and that count indexes a color lookup in the separate legend.
    """
    g = np.asarray(grid, dtype=int)
    sep = _find_separator(g)
    left = g[:, :sep]
    right = g[:, sep + 1 :]

    mapping = _extract_legend_mapping(right)
    if not mapping:
        raise AssertionError("legend mapping must be non-empty")

    stripes = _find_stripe_rows(left)
    if not stripes:
        raise AssertionError("left region must contain stripe rows")

    out = g.copy()
    for r, _color in stripes:
        hole_count = int(np.count_nonzero(left[r] == BLACK))
        if hole_count not in mapping:
            raise AssertionError("missing legend entry for stripe hole count")
        out[r, :sep] = mapping[hole_count]
    return out


def _build_attempt(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    stripe_count = rng.randint(2, 4)
    max_legend_rows = min(5, len(PALETTE) - stripe_count)
    legend_count = rng.randint(stripe_count, max_legend_rows)

    legend_counts = rng.sample([1, 2, 3, 4, 5], legend_count)
    stripe_holes = rng.sample(legend_counts, stripe_count)

    all_colors = list(PALETTE)
    rng.shuffle(all_colors)
    stripe_colors = all_colors[:stripe_count]
    legend_colors = all_colors[stripe_count : stripe_count + legend_count]

    left_width = rng.randint(max(6, max(stripe_holes) + 1), 14)
    top_left = rng.randint(0, 4)
    bottom_left = rng.randint(0, 3)
    left_block_h = 2 * stripe_count - 1

    legend_gap_rows = rng.randint(0, 1)
    top_right = rng.randint(0, 4)
    bottom_right = rng.randint(0, 3)
    right_rows_h = legend_count + (legend_count - 1) * legend_gap_rows

    legend_specs: list[tuple[int, int, int, int]] = []
    right_width = 0
    for count, color in zip(rng.sample(legend_counts, len(legend_counts)), legend_colors):
        offset = rng.randint(0, 2)
        gap = rng.randint(1, 3)
        trail = rng.randint(0, 2)
        need = offset + 1 + gap + count + trail
        right_width = max(right_width, need)
        legend_specs.append((count, color, offset, gap))
    right_width += rng.randint(0, 2)

    height = max(top_left + left_block_h + bottom_left, top_right + right_rows_h + bottom_right)
    width = left_width + 1 + right_width
    if height > 30 or width > 30:
        return None

    grid = np.full((height, width), BLACK, dtype=int)
    grid[:, left_width] = SEPARATOR_COLOR

    legend_map = {count: color for count, color, _offset, _gap in legend_specs}

    stripe_rows = [top_left + 2 * i for i in range(stripe_count)]
    stripe_specs = list(zip(rng.sample(stripe_holes, len(stripe_holes)), stripe_colors, stripe_rows))
    for holes, color, row in stripe_specs:
        grid[row, :left_width] = color
        hole_cols = rng.sample(range(left_width), holes)
        grid[row, hole_cols] = BLACK

    legend_rows = [top_right + i * (legend_gap_rows + 1) for i in range(legend_count)]
    for row, (count, color, offset, gap) in zip(legend_rows, legend_specs):
        start = left_width + 1 + offset
        grid[row, start] = color
        run_start = start + 1 + gap
        grid[row, run_start : run_start + count] = color

    expected = grid.copy()
    for holes, _color, row in stripe_specs:
        expected[row, :left_width] = legend_map[holes]

    if np.array_equal(grid, expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_RETRIES):
        attempt = _build_attempt(rng)
        if attempt is None:
            continue
        grid, expected = attempt
        actual = main(grid)
        if np.array_equal(actual, expected):
            return grid
    raise RuntimeError("failed to generate a valid stripe-legend sample")
