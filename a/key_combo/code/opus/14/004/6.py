import random
from collections import Counter

import numpy as np


BLACK = 0
RED = 2
ORANGE = 7
SAFE_ROW_COLORS = (1, 3, 4, 5, 6, 8, 9)

GEN_ATTEMPTS = 500
ROW_ATTEMPTS = 120


def _row_main_color(row: np.ndarray) -> int:
    counts: Counter[int] = Counter()
    first_seen: dict[int, int] = {}
    for c, value in enumerate(np.asarray(row, dtype=int)):
        color = int(value)
        if color == BLACK:
            continue
        counts[color] += 1
        first_seen.setdefault(color, c)
    if not counts:
        return BLACK
    return max(counts, key=lambda color: (counts[color], -first_seen[color]))


def _has_tied_main(row: np.ndarray, main_color: int) -> bool:
    counts: Counter[int] = Counter(int(v) for v in row if int(v) != BLACK)
    if main_color == BLACK or main_color not in counts:
        return False
    best = counts[main_color]
    return any(color != main_color and count == best for color, count in counts.items())


def _has_unique_main(row: np.ndarray, main_color: int) -> bool:
    counts = [count for color, count in Counter(int(v) for v in row if int(v) != BLACK).items() if color != main_color]
    return main_color != BLACK and (not counts or Counter(int(v) for v in row if int(v) != BLACK)[main_color] > max(counts))


def _active_cells(grid: np.ndarray) -> set[tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    active: set[tuple[int, int]] = set()
    start_cols = [c for c in range(w) if int(g[h - 1, c]) == RED]

    for start_c in start_cols:
        r = h - 1
        c = start_c
        while r > 0:
            next_r = r - 1
            if int(g[next_r, c]) == ORANGE:
                c += 1
                if c >= w:
                    raise AssertionError("route shifted out of bounds")
                r = next_r
                active.add((r, c))
            else:
                r = next_r
                active.add((r, c))
    return active


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use upward obstacle-turn routes to decide which row-wise columns get inverted.

    Input structure:
    - The grid uses black as background and contains 1-2 red marker pixels on the bottom edge.
    - Orange singleton pixels act as obstacles above the bottom row.
    - Every row has its own dominant non-black color, plus optional black cells and non-dominant distractor colors.

    Rule:
    - From each bottom red marker, move upward one row at a time in the current column.
    - If the next cell upward is orange, do not enter it; instead move to that row one column to the right and continue upward from there.
    - The visited cells across all routes form an activation mask.
    - For each row independently, find the most frequent non-black color, breaking ties by the leftmost first appearance.
    - On activated cells only, flip that row's dominant color with black: dominant-color pixels become black, and every other non-orange color becomes the row's dominant color.
    - Orange obstacle pixels always stay orange.

    Key insight:
    - The route itself is invisible in the output; it only selects per-row columns where a row-local dominant-color inversion should happen.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise AssertionError("expected a 2D grid")

    out = g.copy()
    active = _active_cells(g)

    active_by_row: list[set[int]] = [set() for _ in range(g.shape[0])]
    for r, c in active:
        active_by_row[r].add(c)

    for r, cols in enumerate(active_by_row):
        if not cols:
            continue
        row_main = _row_main_color(g[r])
        if row_main == BLACK:
            continue
        for c in cols:
            if int(g[r, c]) == ORANGE:
                continue
            out[r, c] = BLACK if int(g[r, c]) == row_main else row_main
    return out


def _distribute_total(total: int, parts: int) -> list[int]:
    values = [1] * parts
    for _ in range(total - parts):
        values[random.randrange(parts)] += 1
    random.shuffle(values)
    return values


def _band_starts(width: int, turn_counts: list[int]) -> list[int] | None:
    band_widths = [turns + 1 for turns in turn_counts]
    min_gaps = [1] + [2] * (len(turn_counts) - 1) + [1]
    min_total = sum(band_widths) + sum(min_gaps)
    if min_total > width:
        return None
    gaps = min_gaps[:]
    extra = width - min_total
    for _ in range(extra):
        gaps[random.randrange(len(gaps))] += 1

    starts: list[int] = []
    cursor = gaps[0]
    for idx, band_width in enumerate(band_widths):
        starts.append(cursor)
        cursor += band_width + gaps[idx + 1]
    return starts


def _route_active_cells(height: int, start_col: int, turn_rows_desc: list[int]) -> list[tuple[int, int]]:
    turn_rows = set(turn_rows_desc)
    cells: list[tuple[int, int]] = []
    r = height - 1
    c = start_col
    while r > 0:
        next_r = r - 1
        if next_r in turn_rows:
            c += 1
            r = next_r
            cells.append((r, c))
        else:
            r = next_r
            cells.append((r, c))
    return cells


def _generate_row(
    width: int,
    fixed: dict[int, int],
    active_cols: list[int],
    require_tie: bool,
) -> np.ndarray | None:
    available = [c for c in range(width) if c not in fixed]
    if len(available) < 5:
        return None

    for _ in range(ROW_ATTEMPTS):
        main_color = random.choice(SAFE_ROW_COLORS)
        other_choices = [color for color in SAFE_ROW_COLORS if color != main_color]
        rival_color = random.choice(other_choices)
        third_color = random.choice([None] + [color for color in other_choices if color != rival_color])

        row = np.zeros(width, dtype=int)
        for c, color in fixed.items():
            row[c] = color

        active_free = [c for c in active_cols if c not in fixed]
        state_by_active: dict[int, str] = {}
        for c in active_free:
            state = random.choices(("main", "rival", "black"), weights=(4, 3, 3), k=1)[0]
            state_by_active[c] = state
            if state == "main":
                row[c] = main_color
            elif state == "rival":
                row[c] = rival_color

        current_main = int(np.count_nonzero(row == main_color))
        current_rival = int(np.count_nonzero(row == rival_color))
        remaining = [c for c in available if c not in active_free]

        if require_tie:
            options = list(range(3, 6))
            random.shuffle(options)
            target_main = None
            third_count = 0
            for candidate in options:
                for extra_third in random.sample([0, 1], k=2):
                    need_main = max(0, candidate - current_main)
                    need_rival = max(0, candidate - current_rival)
                    if need_main + need_rival + extra_third <= len(remaining):
                        target_main = candidate
                        third_count = extra_third
                        break
                if target_main is not None:
                    break
            if target_main is None:
                continue
            target_rival = target_main
        else:
            combos: list[tuple[int, int, int]] = []
            for target_main in range(4, 8):
                for target_rival in range(0, min(4, target_main - 1) + 1):
                    for third_count in range(0, 3):
                        need_main = max(0, target_main - current_main)
                        need_rival = max(0, target_rival - current_rival)
                        if need_main + need_rival + third_count <= len(remaining):
                            combos.append((target_main, target_rival, third_count))
            if not combos:
                continue
            target_main, target_rival, third_count = random.choice(combos)

        need_main = max(0, target_main - current_main)
        need_rival = max(0, target_rival - current_rival)

        pool = remaining[:]
        random.shuffle(pool)
        main_positions = set(pool[:need_main])
        pool = [c for c in pool if c not in main_positions]
        rival_positions = set(pool[:need_rival])
        pool = [c for c in pool if c not in rival_positions]
        third_positions = set(pool[:third_count])

        for c in main_positions:
            row[c] = main_color
        for c in rival_positions:
            row[c] = rival_color
        if third_color is not None:
            for c in third_positions:
                row[c] = third_color

        actual_main = _row_main_color(row)
        if actual_main != main_color:
            continue
        if require_tie and not _has_tied_main(row, actual_main):
            continue
        if not require_tie and not _has_unique_main(row, actual_main):
            continue
        if np.count_nonzero(row) < 4:
            continue
        return row

    return None


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(13, 18)
    width = random.randint(14, 18)
    marker_count = random.choice((1, 2))
    obstacle_total = random.randint(max(2, marker_count), 5)
    turn_counts = _distribute_total(obstacle_total, marker_count)

    starts = _band_starts(width, turn_counts)
    if starts is None:
        return None

    fixed_by_row = [dict() for _ in range(height)]
    active_by_row: list[set[int]] = [set() for _ in range(height)]

    route_starts = list(zip(starts, turn_counts))
    for start_col, turn_count in route_starts:
        turn_rows_desc = sorted(random.sample(range(1, height - 1), turn_count), reverse=True)
        for idx, obstacle_row in enumerate(turn_rows_desc):
            fixed_by_row[obstacle_row][start_col + idx] = ORANGE
        fixed_by_row[height - 1][start_col] = RED
        for r, c in _route_active_cells(height, start_col, turn_rows_desc):
            active_by_row[r].add(c)

    if random.random() < 0.7:
        tie_row_count = random.randint(1, min(3, height))
        tie_rows = set(random.sample(range(height), tie_row_count))
    else:
        tie_rows = set()

    rows: list[np.ndarray] = []
    for r in range(height):
        row = _generate_row(
            width=width,
            fixed=fixed_by_row[r],
            active_cols=sorted(active_by_row[r]),
            require_tie=r in tie_rows,
        )
        if row is None:
            return None
        rows.append(row)

    grid = np.stack(rows, axis=0)

    actual_active = _active_cells(grid)
    if actual_active != {(r, c) for r in range(height) for c in active_by_row[r]}:
        return None

    actual_obstacles = int(np.count_nonzero(grid == ORANGE))
    if not (2 <= actual_obstacles <= 5):
        return None

    if int(np.count_nonzero(grid[height - 1] == RED)) != marker_count:
        return None

    actual_tie_rows = sum(1 for r in range(height) if _has_tied_main(grid[r], _row_main_color(grid[r])))
    if tie_rows and actual_tie_rows == 0:
        return None

    active_main = 0
    active_nonmain = 0
    for r, c in actual_active:
        row_main = _row_main_color(grid[r])
        if int(grid[r, c]) == row_main:
            active_main += 1
        else:
            active_nonmain += 1
    if active_main == 0 or active_nonmain == 0:
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    if int(np.count_nonzero(out != grid)) < height - 1:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
