from __future__ import annotations

from collections import Counter
import itertools
import random

import numpy as np


BG = 0
BLUE = 1
GREEN = 3
GREY = 5
MAIN_COLORS = (2, 4, 7)
BAR_COLORS = {BLUE, GREEN}

SIZE_RETRIES = 300
HEIGHT_RANGE = (8, 15)
WIDTH_RANGE = (12, 18)


def _sign(n: int) -> int:
    if n > 0:
        return 1
    if n < 0:
        return -1
    return 0


def _bar_info(row: np.ndarray) -> tuple[int, tuple[int, int]]:
    cols = np.flatnonzero(np.isin(row, [BLUE, GREEN]))
    assert cols.size == 2, "each row must contain exactly one blue/green bar"
    start = int(cols[0])
    assert int(cols[1]) == start + 1, "bar pixels must be adjacent"
    return start, (int(row[start]), int(row[start + 1]))


def _row_main_color(row: np.ndarray) -> int:
    counts = Counter(int(v) for v in row if int(v) not in (BG, BLUE, GREEN))
    assert counts, "each data row must contain a dominant non-bar color"
    top = max(counts.values())
    winners = [color for color, count in counts.items() if count == top]
    assert len(winners) == 1, "dominant color must be structurally unique"
    return int(winners[0])


def _row_direction(row: np.ndarray) -> str:
    start, _values = _bar_info(row)
    dominant = _row_main_color(row)

    left = sum(int(row[c]) != dominant for c in range(start))
    right = sum(int(row[c]) != dominant for c in range(start + 2, row.size))

    if left > right:
        return "left"
    if right > left:
        return "right"
    return "stay"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Align each two-color row bar to the reference columns using direction encoded by row-wise inversion.

    Input structure:
    - A black grid has a two-pixel blue/green reference bar on the top row.
    - Every lower row contains one adjacent blue/green bar plus several pixels of one dominant non-bar color and optional grey distractor pixels.
    - In each lower row, the dominant color is the most frequent color among the non-black, non-blue, non-green pixels.

    Rule:
    - Read the top-row blue and green columns as the target columns for those colors.
    - For each lower row independently, conceptually invert the row by treating dominant-color cells as empty and black or grey cells as solid, then compare the solid-cell counts to the left and right of the bar.
    - If the left side has more solids the row bar must slide left, if the right side has more solids it must slide right, and if the counts tie it stays; after that, orient the bar so blue and green match the reference order and place it on the reference columns.

    Key insight:
    - The dominant and distractor pixels disappear from the output; they only encode the bar's movement direction through a row-wise inversion count.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    ref_start, ref_values = _bar_info(g[0])
    assert set(ref_values) == BAR_COLORS, "reference bar must use blue and green"
    col_blue = int(np.flatnonzero(g[0] == BLUE)[0])
    col_green = int(np.flatnonzero(g[0] == GREEN)[0])
    target_left = min(col_blue, col_green)

    out = np.full_like(g, BG)
    out[0, col_blue] = BLUE
    out[0, col_green] = GREEN

    for r in range(1, g.shape[0]):
        start, values = _bar_info(g[r])
        assert set(values) == BAR_COLORS, "every row bar must use blue and green"

        direction = _row_direction(g[r])
        shift_sign = _sign(target_left - start)
        expected = {1: "right", -1: "left", 0: "stay"}[shift_sign]
        assert direction == expected, "encoded direction must match the slide needed for alignment"

        out[r, col_blue] = BLUE
        out[r, col_green] = GREEN

    return out


def _central_reference_left(width: int) -> int:
    middle = (width - 2) // 2
    candidates = list(range(max(2, middle - 1), min(width - 3, middle + 1) + 1))
    assert candidates
    return random.choice(candidates)


def _movement_plan(row_count: int) -> list[str]:
    moves = ["left", "right"]
    if row_count >= 3:
        moves.append("stay")
    while len(moves) < row_count:
        moves.append(random.choice(["left", "right", "stay"]))
    random.shuffle(moves)
    return moves[:row_count]


def _orientation_plan(row_count: int) -> list[str]:
    orders = ["same", "reversed"]
    while len(orders) < row_count:
        orders.append(random.choice(["same", "reversed"]))
    random.shuffle(orders)
    return orders[:row_count]


def _color_plan(row_count: int) -> list[int]:
    colors = list(MAIN_COLORS)
    while len(colors) < row_count:
        colors.append(random.choice(MAIN_COLORS))
    random.shuffle(colors)
    return colors[:row_count]


def _bar_start(width: int, ref_left: int, move: str) -> int:
    if move == "left":
        return random.randint(ref_left + 1, width - 2)
    if move == "right":
        return random.randint(0, ref_left - 1)
    return ref_left


def _find_main_positions(width: int, bar_start: int, move: str) -> list[int] | None:
    open_cols = [c for c in range(width) if c not in (bar_start, bar_start + 1)]
    want = {"left": 1, "right": -1, "stay": 0}[move]
    sizes = [2, 3, 4]
    random.shuffle(sizes)

    for count in sizes:
        valid: list[tuple[int, ...]] = []
        for cols in itertools.combinations(open_cols, count):
            main_left = sum(c < bar_start for c in cols)
            main_right = sum(c > bar_start + 1 for c in cols)
            left_solid = bar_start - main_left
            right_solid = (width - bar_start - 2) - main_right
            cmp = _sign(left_solid - right_solid)
            if cmp == want:
                valid.append(cols)
        if valid:
            return list(random.choice(valid))
    return None


def _place_row(
    width: int,
    ref_left: int,
    move: str,
    order: str,
    main_color: int,
    force_grey: bool,
) -> np.ndarray | None:
    row = np.full(width, BG, dtype=int)
    start = _bar_start(width, ref_left, move)
    row[start : start + 2] = [BLUE, GREEN] if order == "same" else [GREEN, BLUE]

    main_positions = _find_main_positions(width, start, move)
    if main_positions is None:
        return None
    row[main_positions] = main_color

    remaining = [c for c in range(width) if c not in (start, start + 1) and c not in main_positions]
    max_grey = min(2, len(main_positions) - 1, len(remaining))
    min_grey = 1 if force_grey and max_grey >= 1 else 0
    grey_count = random.randint(min_grey, max_grey) if max_grey else 0
    if grey_count:
        for c in random.sample(remaining, grey_count):
            row[c] = GREY

    return row


def _build_candidate() -> np.ndarray | None:
    height = random.randint(*HEIGHT_RANGE)
    width = random.randint(*WIDTH_RANGE)
    ref_left = _central_reference_left(width)
    row_count = height - 1

    moves = _movement_plan(row_count)
    orders = _orientation_plan(row_count)
    colors = _color_plan(row_count)
    grey_rows = set(random.sample(range(row_count), k=random.randint(1, max(1, row_count // 3))))

    grid = np.full((height, width), BG, dtype=int)
    grid[0, ref_left : ref_left + 2] = [BLUE, GREEN]

    seen_grey = False
    for idx in range(row_count):
        row = _place_row(
            width=width,
            ref_left=ref_left,
            move=moves[idx],
            order=orders[idx],
            main_color=colors[idx],
            force_grey=idx in grey_rows,
        )
        if row is None:
            return None
        seen_grey = seen_grey or bool(np.any(row == GREY))
        grid[idx + 1] = row

    if not seen_grey:
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    expected = np.full_like(grid, BG)
    expected[0, ref_left] = BLUE
    expected[0, ref_left + 1] = GREEN
    expected[1:, ref_left] = BLUE
    expected[1:, ref_left + 1] = GREEN
    if not np.array_equal(out, expected):
        return None

    found_moves = {_row_direction(grid[r]) for r in range(1, height)}
    if "left" not in found_moves or "right" not in found_moves:
        return None
    if row_count >= 3 and "stay" not in found_moves:
        return None

    row_orders = {tuple(grid[r, _bar_info(grid[r])[0] : _bar_info(grid[r])[0] + 2]) for r in range(1, height)}
    if row_orders != {(BLUE, GREEN), (GREEN, BLUE)}:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(SIZE_RETRIES):
        grid = _build_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 9")
