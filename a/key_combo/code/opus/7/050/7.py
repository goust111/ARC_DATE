from __future__ import annotations

import itertools
import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
SEP = 3
YELLOW = 4
BAR = 5
ORANGE = 7


def _separator_column(grid: np.ndarray) -> int:
    height, width = map(int, grid.shape)
    assert width >= 3 and width % 2 == 1, "expected two equal panels plus a divider"

    candidates: list[int] = []
    for col in range(1, width - 1):
        if col != width - col - 1:
            continue
        color = int(grid[0, col])
        if color != SEP or not np.all(grid[:, col] == color):
            continue
        if np.any(grid[:, :col] == color) or np.any(grid[:, col + 1 :] == color):
            continue
        candidates.append(col)

    assert len(candidates) == 1, "expected one central separator column"
    return candidates[0]


def _bar_rows(panel: np.ndarray) -> list[int]:
    rows = [r for r in range(panel.shape[0]) if np.all(panel[r, :] == BAR)]
    assert 1 <= len(rows) <= 2, "expected one or two horizontal full-width bars"
    return rows


def _stamp_hit(out: np.ndarray, row: int, col: int, color: int) -> None:
    height, width = out.shape
    r0 = max(0, row - 1)
    r1 = min(height, row + 2)
    c0 = max(0, col - 1)
    c1 = min(width, col + 2)
    out[r0:r1, c0:c1] = BAR
    out[row, col] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compare the two panels, convert only differing pixels into active ray sources, and stamp each bar hit.

    Input structure:
    - The input is a black background grid split into two equal panels by one solid green separator column.
    - Each panel contains one or two gray full-width horizontal bars at the same rows and several isolated orange pixels away from the bars.
    - The orange pixels may be shared between panels or appear in only one panel.

    Rule:
    - Remove the separator and compare the aligned panels by orange-pixel presence only.
    - Shared orange pixels become yellow inert markers; left-only pixels become red active sources; right-only pixels become blue active sources; gray bars stay gray.
    - Each active source casts vertical rays upward and downward until the first bar in that direction, coloring the source-to-bar segment with the source color and then stamping a gray 3x3 block centered on the hit bar cell while restoring the center to the source color.

    Key insight:
    - The panel comparison is only a filter stage: it decides which aligned pixels stay inert and which ones gain ray-casting behavior before the bar-hit stamps are applied.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep_col = _separator_column(g)
    left = g[:, :sep_col]
    right = g[:, sep_col + 1 :]
    assert left.shape == right.shape, "separator must split two equal panels"

    left_bars = _bar_rows(left)
    right_bars = _bar_rows(right)
    assert left_bars == right_bars, "both panels must share the same bar rows"
    bar_set = set(left_bars)

    left_marks = (left != BG) & (left != BAR)
    right_marks = (right != BG) & (right != BAR)
    assert set(map(int, np.unique(left[left_marks]))) <= {ORANGE}
    assert set(map(int, np.unique(right[right_marks]))) <= {ORANGE}

    out = np.full(left.shape, BG, dtype=int)
    out[left_bars, :] = BAR

    shared = left_marks & right_marks
    left_only = left_marks & ~right_marks
    right_only = right_marks & ~left_marks
    out[shared] = YELLOW
    out[left_only] = RED
    out[right_only] = BLUE

    controls = [
        (int(r), int(c), int(out[r, c]))
        for r, c in np.argwhere((out == RED) | (out == BLUE))
    ]
    controls.sort()

    height = out.shape[0]
    for row, col, color in controls:
        for step in (-1, 1):
            hit_row = None
            rr = row + step
            while 0 <= rr < height:
                if rr in bar_set:
                    hit_row = rr
                    break
                rr += step
            if hit_row is None:
                continue

            rr = row
            while rr != hit_row:
                out[rr, col] = color
                rr += step
            _stamp_hit(out, hit_row, col, color)

    return out


PANEL_HEIGHT_RANGE = (11, 15)
PANEL_WIDTH_RANGE = (9, 14)
MIN_BAR_GAP = 6
MIN_PIXEL_TO_BAR_DIST = 3
MAX_ATTEMPTS = 500
OVERLAP_BRANCH_PROB = 0.35

COUNT_TRIPLES = [
    (shared, red_count, blue_count)
    for shared in range(1, 4)
    for red_count in range(1, 4)
    for blue_count in range(1, 4)
    if 3 <= shared + red_count <= 6 and 3 <= shared + blue_count <= 6
]


def _row_bands(height: int, bars: list[int]) -> dict[str, list[int]]:
    top = min(bars)
    bottom = max(bars)
    above = list(range(0, top - MIN_PIXEL_TO_BAR_DIST + 1))
    below = list(range(bottom + MIN_PIXEL_TO_BAR_DIST, height))
    between: list[int] = []
    if len(bars) == 2:
        between = list(range(top + MIN_PIXEL_TO_BAR_DIST, bottom - MIN_PIXEL_TO_BAR_DIST + 1))
    return {"above": above, "between": between, "below": below}


def _choose_bars(height: int) -> list[int]:
    one_bar = list(range(3, height - 3))
    two_bar: list[tuple[int, int]] = []
    for pair in itertools.combinations(range(2, height - 2), 2):
        if pair[1] - pair[0] < MIN_BAR_GAP:
            continue
        bands = _row_bands(height, list(pair))
        if not bands["between"]:
            continue
        if not (bands["above"] or bands["below"]):
            continue
        two_bar.append(pair)

    if two_bar and random.random() < 0.5:
        return list(random.choice(two_bar))
    return [random.choice(one_bar)]


def _isolated_pixels(panel: np.ndarray) -> bool:
    mask = panel == ORANGE
    return not (np.any(mask[1:, :] & mask[:-1, :]) or np.any(mask[:, 1:] & mask[:, :-1]))


def _can_place(cell: tuple[int, int], used: set[tuple[int, int]], used_cols: set[int]) -> bool:
    row, col = cell
    if col in used_cols or cell in used:
        return False
    for rr, cc in used:
        if abs(row - rr) + abs(col - cc) == 1:
            return False
    return True


def _pick_cell(
    rows: list[int],
    width: int,
    used: set[tuple[int, int]],
    used_cols: set[int],
) -> tuple[int, int] | None:
    candidates = [
        (row, col)
        for row in rows
        for col in range(width)
        if _can_place((row, col), used, used_cols)
    ]
    if not candidates:
        return None
    return random.choice(candidates)


def _build_positions(
    height: int,
    width: int,
    bars: list[int],
    counts: tuple[int, int, int],
) -> dict[str, list[tuple[int, int]]] | None:
    shared_count, red_count, blue_count = counts
    bands = _row_bands(height, bars)
    all_rows = sorted(set(bands["above"] + bands["between"] + bands["below"]))

    positions = {"shared": [], "red": [], "blue": []}
    used: set[tuple[int, int]] = set()
    used_cols: set[int] = set()

    def place_exact(kind: str, row_choices: list[int], *, fixed_col: int | None = None) -> bool:
        if fixed_col is None:
            cell = _pick_cell(row_choices, width, used, used_cols)
        else:
            cells = [
                (row, fixed_col)
                for row in row_choices
                if _can_place((row, fixed_col), used, used_cols - {fixed_col})
            ]
            cell = random.choice(cells) if cells else None
        if cell is None:
            return False
        positions[kind].append(cell)
        used.add(cell)
        used_cols.add(cell[1])
        return True

    if len(bars) == 1:
        want_overlap = bool(bands["above"] and bands["below"] and random.random() < OVERLAP_BRANCH_PROB)
        if want_overlap:
            overlap_cols = list(range(width))
            random.shuffle(overlap_cols)
            placed = False
            for col in overlap_cols:
                upper_choices = [(row, col) for row in bands["above"] if _can_place((row, col), used, used_cols)]
                lower_choices = [(row, col) for row in bands["below"] if _can_place((row, col), used, used_cols)]
                if not upper_choices or not lower_choices:
                    continue
                upper = random.choice(upper_choices)
                lower = random.choice(lower_choices)
                if random.random() < 0.5:
                    positions["red"].append(upper)
                    positions["blue"].append(lower)
                else:
                    positions["blue"].append(upper)
                    positions["red"].append(lower)
                used.update({upper, lower})
                used_cols.add(col)
                placed = True
                break
            if not placed:
                return None
        else:
            if random.random() < 0.5:
                order = [("red", bands["above"]), ("blue", bands["below"])]
            else:
                order = [("blue", bands["above"]), ("red", bands["below"])]
            for kind, rows in order:
                if not place_exact(kind, rows):
                    return None
    else:
        outside_rows = bands["above"] + bands["below"]
        if random.random() < 0.5:
            mandatory = [("red", bands["between"]), ("blue", outside_rows)]
        else:
            mandatory = [("blue", bands["between"]), ("red", outside_rows)]
        for kind, rows in mandatory:
            if not place_exact(kind, rows):
                return None

    remaining = {
        "shared": shared_count - len(positions["shared"]),
        "red": red_count - len(positions["red"]),
        "blue": blue_count - len(positions["blue"]),
    }
    if min(remaining.values()) < 0:
        return None

    fill_order = ["shared", "red", "blue"]
    random.shuffle(fill_order)
    for kind in fill_order:
        for _ in range(remaining[kind]):
            cell = _pick_cell(all_rows, width, used, used_cols)
            if cell is None:
                return None
            positions[kind].append(cell)
            used.add(cell)
            used_cols.add(cell[1])

    return positions


def _render_input(height: int, width: int, bars: list[int], positions: dict[str, list[tuple[int, int]]]) -> np.ndarray:
    left = np.full((height, width), BG, dtype=int)
    right = np.full((height, width), BG, dtype=int)
    left[bars, :] = BAR
    right[bars, :] = BAR

    for row, col in positions["shared"]:
        left[row, col] = ORANGE
        right[row, col] = ORANGE
    for row, col in positions["red"]:
        left[row, col] = ORANGE
    for row, col in positions["blue"]:
        right[row, col] = ORANGE

    sep = np.full((height, 1), SEP, dtype=int)
    return np.concatenate([left, sep, right], axis=1)


def _valid_generator_case(
    grid: np.ndarray,
    bars: list[int],
    positions: dict[str, list[tuple[int, int]]],
) -> bool:
    sep_col = grid.shape[1] // 2
    left = grid[:, :sep_col]
    right = grid[:, sep_col + 1 :]

    left_count = len(positions["shared"]) + len(positions["red"])
    right_count = len(positions["shared"]) + len(positions["blue"])
    if not (3 <= left_count <= 6 and 3 <= right_count <= 6):
        return False
    if len(positions["shared"]) < 1:
        return False
    if len(positions["red"]) + len(positions["blue"]) < 2:
        return False
    if not _isolated_pixels(left) or not _isolated_pixels(right):
        return False
    if np.any(left[bars, :] == ORANGE) or np.any(right[bars, :] == ORANGE):
        return False

    active_rows = [row for row, _col in positions["red"] + positions["blue"]]
    if len(bars) == 2:
        top, bottom = sorted(bars)
        if not any(top < row < bottom for row in active_rows):
            return False
        if not any(row < top or row > bottom for row in active_rows):
            return False

    try:
        out = main(grid)
    except AssertionError:
        return False
    return out.shape == left.shape


def _generate_attempt() -> np.ndarray | None:
    height = random.randint(*PANEL_HEIGHT_RANGE)
    width = random.randint(*PANEL_WIDTH_RANGE)
    if 2 * width + 1 > 30:
        return None

    bars = _choose_bars(height)
    counts = random.choice(COUNT_TRIPLES)
    positions = _build_positions(height, width, bars, counts)
    if positions is None:
        return None

    grid = _render_input(height, width, bars, positions)
    if not _valid_generator_case(grid, bars, positions):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
