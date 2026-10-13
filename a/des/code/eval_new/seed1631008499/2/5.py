import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GRAY = 5
MAGENTA = 6
ORANGE = 7
CYAN = 8
BROWN = 9

TOWER_COLORS = [BLUE, RED, GREEN, YELLOW, MAGENTA, CYAN, BROWN]
GRID_H_RANGE = (7, 12)
GRID_W_RANGE = (16, 28)
HEIGHT_RANGE = (2, 6)
EXTRA_TOWERS_BY_BRANCH = {
    "disjoint": (0, 1),
    "unequal": (0, 2),
    "equal": (0, 2),
}
GEN_RETRIES = 400


def _coverage_interval(col: int, height: int, width: int) -> tuple[int, int]:
    return max(0, col - height), min(width - 1, col + height)


def _extract_towers(grid: np.ndarray) -> tuple[list[tuple[int, int, int]], int, int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height >= 3, "grid must leave room for a tower row above the baseline"

    baseline_color = int(g[-1, 0])
    assert np.all(g[-1, :] == baseline_color), "bottom row must be a uniform baseline"

    values, counts = np.unique(g[:-1], return_counts=True)
    bg = int(values[np.argmax(counts)])
    assert bg != baseline_color, "background and baseline colors must differ"

    signal_row = height - 2
    towers: list[tuple[int, int, int]] = []
    tower_mask = np.zeros_like(g, dtype=bool)

    for col in range(width):
        color = int(g[signal_row, col])
        if color in {bg, baseline_color}:
            continue

        row = signal_row
        while row >= 0 and int(g[row, col]) == color:
            row -= 1
        top = row + 1
        tower_height = signal_row - top + 1

        assert 2 <= tower_height <= signal_row + 1, "tower height must be positive and vertical"
        assert np.all(g[top : signal_row + 1, col] == color), "tower must be a solid monochrome column"
        if top > 0:
            assert np.all(g[:top, col] == bg), "tower column must be empty above the tower"

        clear_mask = np.ones(height, dtype=bool)
        clear_mask[top : signal_row + 1] = False
        clear_mask[-1] = False
        assert np.all(g[clear_mask, col] == bg), "tower column cannot contain extra foreground pixels"

        tower_mask[top : signal_row + 1, col] = True
        towers.append((col, tower_height, color))

    assert towers, "expected at least one tower"
    tower_cols = sorted(col for col, _tower_height, _color in towers)
    assert len(tower_cols) == len(set(tower_cols)), "tower columns must be unique"
    if len(tower_cols) > 1:
        assert min(b - a for a, b in zip(tower_cols, tower_cols[1:])) >= 3, "towers must have gaps"

    above_baseline = g[:-1]
    assert np.all(above_baseline[~tower_mask[:-1]] == bg), "only towers may appear above the baseline"
    return towers, bg, baseline_color


def _paint_signal_row(
    towers: list[tuple[int, int, int]],
    width: int,
    original_row: np.ndarray,
) -> np.ndarray:
    row = np.array(original_row, copy=True)
    tower_cols = {col for col, _tower_height, _color in towers}

    for col in range(width):
        if col in tower_cols:
            continue

        best_height = -1
        best_color = int(row[col])
        tie = False

        for tower_col, tower_height, tower_color in towers:
            left, right = _coverage_interval(tower_col, tower_height, width)
            if not (left <= col <= right):
                continue
            if tower_height > best_height:
                best_height = tower_height
                best_color = tower_color
                tie = False
            elif tower_height == best_height:
                tie = True

        if best_height >= 0:
            row[col] = ORANGE if tie else best_color

    return row


def _visible_overlap_columns(
    col_a: int,
    height_a: int,
    col_b: int,
    height_b: int,
    width: int,
) -> list[int]:
    left_a, right_a = _coverage_interval(col_a, height_a, width)
    left_b, right_b = _coverage_interval(col_b, height_b, width)
    left = max(left_a, left_b)
    right = min(right_a, right_b)
    if left > right:
        return []
    return [col for col in range(left, right + 1) if col not in {col_a, col_b}]


def _pairwise_intervals_overlap(
    towers: list[tuple[int, int]],
    width: int,
) -> bool:
    for i in range(len(towers)):
        col_a, height_a = towers[i]
        for j in range(i + 1, len(towers)):
            col_b, height_b = towers[j]
            if _visible_overlap_columns(col_a, height_a, col_b, height_b, width):
                return True
    return False


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw each tower's horizontal signal reach on the row above the baseline.

    Input structure:
    - The background is black.
    - The bottom row is a full-width gray baseline.
    - Above the baseline stand several single-column towers, each a solid vertical run of one distinct color.
    - Towers are separated by at least two empty columns, and all non-tower cells above the baseline are black.

    Rule:
    - Measure each tower's height in pixels; that height is also its signal radius to the left and right.
    - On the row directly above the baseline, fill every covered black cell with the tower's color, clipping at the grid edges.
    - Tower body cells on that row stay unchanged even if another tower's signal also reaches them.
    - If multiple towers cover the same non-tower column, use the color of the tallest covering tower; if the tallest height is tied, paint that column orange.

    Key insight:
    - A tower's visible height is the hidden numeric parameter that controls how far its signal spreads, and overlap resolution depends on comparing those heights rather than on drawing order.
    """
    g = np.asarray(grid, dtype=int)
    towers, _bg, _baseline_color = _extract_towers(g)
    out = g.copy()
    signal_row = g.shape[0] - 2
    out[signal_row, :] = _paint_signal_row(towers, g.shape[1], g[signal_row, :])
    return out


def _can_place_disjoint(col: int, height: int, towers: list[tuple[int, int]], width: int) -> bool:
    for other_col, other_height in towers:
        if abs(col - other_col) < 3:
            return False
        if _visible_overlap_columns(col, height, other_col, other_height, width):
            return False
        left_a, right_a = _coverage_interval(col, height, width)
        left_b, right_b = _coverage_interval(other_col, other_height, width)
        if not (right_a < left_b or right_b < left_a):
            return False
    return True


def _sample_core_pair(branch: str, width: int, max_height: int) -> list[tuple[int, int]] | None:
    for _ in range(80):
        if branch == "equal":
            shared_height = random.randint(HEIGHT_RANGE[0], max_height)
            height_a = shared_height
            height_b = shared_height
        elif branch == "unequal":
            height_a = random.randint(HEIGHT_RANGE[0], max_height)
            choices = [h for h in range(HEIGHT_RANGE[0], max_height + 1) if h != height_a]
            height_b = random.choice(choices)
        else:
            height_a = random.randint(HEIGHT_RANGE[0], max_height)
            height_b = random.randint(HEIGHT_RANGE[0], max_height)

        if branch == "disjoint":
            min_distance = max(3, height_a + height_b + 1)
            max_distance = min(width - 1, min_distance + 8)
        else:
            min_distance = max(3, max(height_a, height_b) + 1)
            max_distance = min(width - 1, height_a + height_b)

        if min_distance > max_distance:
            continue

        distance = random.randint(min_distance, max_distance)
        left_col = random.randint(0, width - distance - 1)
        right_col = left_col + distance
        return [(left_col, height_a), (right_col, height_b)]
    return None


def _candidate_extra_columns(
    width: int,
    height: int,
    towers: list[tuple[int, int]],
) -> list[int]:
    return [col for col in range(width) if _can_place_disjoint(col, height, towers, width)]


def _shift_towers(towers: list[tuple[int, int]], width: int) -> list[tuple[int, int]]:
    cols = [col for col, _height in towers]
    min_col = min(cols)
    max_col = max(cols)
    mode = random.random()

    if mode < 0.35 and min_col > 0:
        target = random.randint(0, min(1, width - 1))
        delta = target - min_col
    elif mode < 0.70 and max_col < width - 1:
        target = random.randint(max(width - 2, 0), width - 1)
        delta = target - max_col
    else:
        return towers

    shifted = [(col + delta, height) for col, height in towers]
    assert min(col for col, _height in shifted) >= 0
    assert max(col for col, _height in shifted) < width
    return shifted


def _branch_matches(branch: str, towers: list[tuple[int, int]], width: int) -> bool:
    visible_pairs: list[tuple[int, int]] = []
    for i in range(len(towers)):
        col_a, height_a = towers[i]
        for j in range(i + 1, len(towers)):
            col_b, height_b = towers[j]
            if _visible_overlap_columns(col_a, height_a, col_b, height_b, width):
                visible_pairs.append((height_a, height_b))

    if branch == "disjoint":
        return not visible_pairs
    if branch == "equal":
        return any(height_a == height_b for height_a, height_b in visible_pairs)
    return any(height_a != height_b for height_a, height_b in visible_pairs)


def _build_grid(height: int, width: int, towers: list[tuple[int, int, int]]) -> np.ndarray:
    grid = np.full((height, width), BLACK, dtype=int)
    grid[-1, :] = GRAY
    signal_row = height - 2
    for col, tower_height, color in towers:
        top = signal_row - tower_height + 1
        grid[top : signal_row + 1, col] = color
    return grid


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_H_RANGE)
    width = random.randint(*GRID_W_RANGE)
    max_height = min(HEIGHT_RANGE[1], height - 1)
    if max_height < HEIGHT_RANGE[0]:
        return None

    branch = random.choice(["disjoint", "unequal", "equal"])
    core = _sample_core_pair(branch, width, max_height)
    if core is None:
        return None

    towers = core[:]
    extra_min, extra_max = EXTRA_TOWERS_BY_BRANCH[branch]
    target_count = 2 + random.randint(extra_min, extra_max)
    for _ in range(target_count - 2):
        tower_height = random.randint(HEIGHT_RANGE[0], max_height)
        candidates = _candidate_extra_columns(width, tower_height, towers)
        if not candidates:
            return None
        towers.append((random.choice(candidates), tower_height))

    towers = sorted(_shift_towers(towers, width))
    if not _branch_matches(branch, towers, width):
        return None

    colors = random.sample(TOWER_COLORS, len(towers))
    colored_towers = [(col, tower_height, color) for (col, tower_height), color in zip(towers, colors)]
    grid = _build_grid(height, width, colored_towers)
    out = main(grid)

    if branch == "equal" and not np.any(out[-2, :] == ORANGE):
        return None
    if branch == "disjoint" and np.any(out[-2, :] == ORANGE):
        return None

    parsed_towers, bg, baseline_color = _extract_towers(grid)
    if bg != BLACK or baseline_color != GRAY:
        return None
    if sorted(parsed_towers) != sorted(colored_towers):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid signal-tower input after {GEN_RETRIES} attempts")
