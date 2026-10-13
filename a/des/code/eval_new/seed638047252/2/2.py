import random
from collections import defaultdict

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
CYAN = 8
MAROON = 9

SEED_COLORS = (BLUE, RED, YELLOW, CYAN)
GEN_ATTEMPTS = 300
LINE_COUNT_RANGE = (2, 4)
GAP_RANGE = (2, 6)
MARGIN_RANGE = (1, 4)
MAX_EXTRA_SEEDS = 2


def _axis_membership(mask: np.ndarray, r: int, c: int) -> tuple[bool, bool]:
    h, w = mask.shape
    horizontal = (c > 0 and mask[r, c - 1]) or (c + 1 < w and mask[r, c + 1])
    vertical = (r > 0 and mask[r - 1, c]) or (r + 1 < h and mask[r + 1, c])
    return horizontal, vertical


def _blend(color_a: int, color_b: int) -> int:
    if color_a == color_b:
        return int(color_a)
    pair = frozenset((int(color_a), int(color_b)))
    if pair == frozenset((RED, BLUE)):
        return MAGENTA
    if pair == frozenset((RED, YELLOW)):
        return ORANGE
    if pair == frozenset((BLUE, YELLOW)):
        return GREEN
    return MAROON


def _nearest_target(position: int, targets: list[int]) -> int:
    return min(targets, key=lambda target: (abs(target - position), target))


def _candidate_map(points: list[int]) -> dict[int, dict[str, list[int]]]:
    point_set = set(points)
    result = {point: {"tie": [], "plain": []} for point in points}
    for pos in range(points[0], points[-1] + 1):
        if pos in point_set:
            continue
        distances = [abs(point - pos) for point in points]
        best_distance = min(distances)
        nearest = [point for point, distance in zip(points, distances) if distance == best_distance]
        chosen = min(nearest)
        bucket = "tie" if len(nearest) > 1 else "plain"
        result[chosen][bucket].append(pos)
    return result


def main(grid: np.ndarray) -> np.ndarray:
    """Slide colored seed pixels along grey row/column channels into the nearest intersections and mix crossing arrivals there.

    Input structure:
    - The background is black, with one-pixel-wide grey horizontal and vertical channels forming an orthogonal lattice.
    - Non-grey, non-black singleton seed pixels sit on channel cells that are not intersections.
    - Grey cells that belong to both a horizontal and a vertical channel are intersections.

    Rule:
    - Each seed moves only along its own channel axis toward the nearest intersection on that row or column.
    - If a seed is equally close to two intersections, it chooses the left one on horizontal channels or the upper one on vertical channels.
    - The seed's original cell becomes grey again, and an intersection reached by one seed takes that seed's color.
    - If one horizontal seed and one vertical seed reach the same intersection, red+blue becomes magenta, red+yellow becomes orange, blue+yellow becomes green, and every other distinct pair becomes maroon.

    Key insight:
    - The task is solved by recovering the hidden channel network from the visible grey paths, then routing each seed to its nearest cross-point before applying a small unordered color-mixing table at shared destinations.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    occupied = g != BLACK
    grey_cells = g == GREY

    intersections: list[tuple[int, int]] = []
    for r, c in np.argwhere(grey_cells):
        horizontal, vertical = _axis_membership(occupied, int(r), int(c))
        if horizontal and vertical:
            intersections.append((int(r), int(c)))

    assert intersections, "expected at least one intersection"

    row_targets: dict[int, list[int]] = defaultdict(list)
    col_targets: dict[int, list[int]] = defaultdict(list)
    for r, c in intersections:
        row_targets[r].append(c)
        col_targets[c].append(r)
    for cols in row_targets.values():
        cols.sort()
    for rows in col_targets.values():
        rows.sort()

    out = g.copy()
    arrivals: dict[tuple[int, int], list[int]] = defaultdict(list)
    seed_cells = np.argwhere((g != BLACK) & (g != GREY))
    for r, c in seed_cells:
        r = int(r)
        c = int(c)
        color = int(g[r, c])
        horizontal, vertical = _axis_membership(occupied, r, c)
        assert horizontal ^ vertical, "seed must lie on exactly one channel axis"

        out[r, c] = GREY
        if horizontal:
            targets = row_targets.get(r, [])
            assert targets, "horizontal seed row must contain intersections"
            target = (r, _nearest_target(c, targets))
        else:
            targets = col_targets.get(c, [])
            assert targets, "vertical seed column must contain intersections"
            target = (_nearest_target(r, targets), c)
        arrivals[target].append(color)

    for (r, c), colors in arrivals.items():
        assert len(colors) <= 2, "at most one horizontal and one vertical seed may share an intersection"
        out[r, c] = colors[0] if len(colors) == 1 else _blend(colors[0], colors[1])

    return out


def _sample_axis_points(count: int) -> tuple[list[int], int]:
    start = random.randint(*MARGIN_RANGE)
    points = [start]
    for _ in range(count - 1):
        points.append(points[-1] + random.randint(*GAP_RANGE))
    end_margin = random.randint(*MARGIN_RANGE)
    size = points[-1] + end_margin + 1
    return points, size


def _pair_colors() -> tuple[int, int]:
    pair_type = random.randrange(4)
    if pair_type == 0:
        return RED, BLUE
    if pair_type == 1:
        return RED, YELLOW
    if pair_type == 2:
        return BLUE, YELLOW
    return random.choice(((RED, CYAN), (BLUE, CYAN), (YELLOW, CYAN)))


def _choose_position(candidates: dict[str, list[int]]) -> tuple[int, bool] | None:
    choices: list[tuple[int, bool]] = []
    choices.extend((pos, False) for pos in candidates["plain"])
    choices.extend((pos, True) for pos in candidates["tie"])
    if not choices:
        return None
    plain_weight = max(1, len(candidates["plain"]))
    tie_weight = max(1, len(candidates["tie"]))
    if candidates["plain"] and candidates["tie"]:
        bucket = random.choices(("plain", "tie"), weights=(plain_weight, tie_weight), k=1)[0]
        pos = random.choice(candidates[bucket])
        return pos, bucket == "tie"
    if candidates["plain"]:
        return random.choice(candidates["plain"]), False
    return random.choice(candidates["tie"]), True


def _place_horizontal_seed(
    grid: np.ndarray,
    row: int,
    target_col: int,
    color: int,
    row_candidates: dict[int, dict[str, list[int]]],
) -> bool:
    choice = _choose_position(row_candidates[target_col])
    if choice is None:
        return False
    col, _is_tie = choice
    grid[row, col] = color
    return True


def _place_vertical_seed(
    grid: np.ndarray,
    col: int,
    target_row: int,
    color: int,
    col_candidates: dict[int, dict[str, list[int]]],
) -> bool:
    choice = _choose_position(col_candidates[target_row])
    if choice is None:
        return False
    row, _is_tie = choice
    grid[row, col] = color
    return True


def _gen_attempt() -> np.ndarray | None:
    row_count = random.randint(*LINE_COUNT_RANGE)
    col_count = random.randint(*LINE_COUNT_RANGE)
    rows, height = _sample_axis_points(row_count)
    cols, width = _sample_axis_points(col_count)
    if max(height, width) > 30:
        return None

    row_candidates = _candidate_map(cols)
    col_candidates = _candidate_map(rows)
    if not any(bucket["tie"] for bucket in row_candidates.values()) and not any(
        bucket["tie"] for bucket in col_candidates.values()
    ):
        return None

    grid = np.full((height, width), BLACK, dtype=int)
    r0, r1 = rows[0], rows[-1]
    c0, c1 = cols[0], cols[-1]
    for row in rows:
        grid[row, c0 : c1 + 1] = GREY
    for col in cols:
        grid[r0 : r1 + 1, col] = GREY

    max_duals = min(row_count, col_count, 2)
    dual_count = 1 if max_duals == 1 else random.randint(1, max_duals)
    if dual_count == min(row_count, col_count) and min(row_count, col_count) == 2:
        dual_count = 1

    dual_rows = random.sample(rows, dual_count)
    dual_cols = random.sample(cols, dual_count)
    random.shuffle(dual_rows)
    random.shuffle(dual_cols)
    dual_pairs = list(zip(dual_rows, dual_cols))

    horizontal_targets: dict[int, int] = {}
    vertical_targets: dict[int, int] = {}
    for row, col in dual_pairs:
        horizontal_targets[row] = col
        vertical_targets[col] = row

    remaining_rows = [row for row in rows if row not in horizontal_targets]
    remaining_cols = [col for col in cols if col not in vertical_targets]

    extra_h = random.randint(0, min(MAX_EXTRA_SEEDS, len(remaining_rows)))
    extra_v = random.randint(0, min(MAX_EXTRA_SEEDS, len(remaining_cols)))
    if extra_h + extra_v == 0:
        if remaining_rows and (not remaining_cols or random.random() < 0.5):
            extra_h = 1
        elif remaining_cols:
            extra_v = 1

    random.shuffle(remaining_rows)
    random.shuffle(remaining_cols)

    vertical_target_cells = {(target_row, col) for col, target_row in vertical_targets.items()}
    for row in remaining_rows[:extra_h]:
        eligible_cols = [col for col in cols if (row, col) not in vertical_target_cells]
        if not eligible_cols:
            return None
        horizontal_targets[row] = random.choice(eligible_cols)

    horizontal_target_cells = {(row, target_col) for row, target_col in horizontal_targets.items()}
    for col in remaining_cols[:extra_v]:
        eligible_rows = [row for row in rows if (row, col) not in horizontal_target_cells]
        if not eligible_rows:
            return None
        vertical_targets[col] = random.choice(eligible_rows)

    for row, col in dual_pairs:
        horizontal_color, vertical_color = _pair_colors()
        if not _place_horizontal_seed(grid, row, col, horizontal_color, row_candidates):
            return None
        if not _place_vertical_seed(grid, col, row, vertical_color, col_candidates):
            return None

    for row, target_col in horizontal_targets.items():
        if row in {pair_row for pair_row, _pair_col in dual_pairs}:
            continue
        if not _place_horizontal_seed(grid, row, target_col, random.choice(SEED_COLORS), row_candidates):
            return None

    for col, target_row in vertical_targets.items():
        if col in {pair_col for _pair_row, pair_col in dual_pairs}:
            continue
        if not _place_vertical_seed(grid, col, target_row, random.choice(SEED_COLORS), col_candidates):
            return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid sliding-seed channel input")
