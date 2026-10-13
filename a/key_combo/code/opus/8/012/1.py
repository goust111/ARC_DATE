import random
from collections import Counter

import numpy as np


class Color:
    BLACK = 0
    BLUE = 1
    RED = 2
    GREEN = 3
    YELLOW = 4
    GREY = 5
    ORANGE = 7


SOURCE_COLORS = (Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.ORANGE)
GEN_RETRY_LIMIT = 600
GEN_HEIGHT_RANGE = (13, 18)
GEN_WIDTH_RANGE = (12, 16)
GEN_SOURCE_COUNT_RANGE = (3, 5)
GEN_OBSTACLE_COUNT_RANGE = (2, 4)


def get_bg(grid: np.ndarray) -> int:
    values = np.asarray(grid, dtype=int).ravel()
    return int(Counter(map(int, values)).most_common(1)[0][0])


def _contiguous_runs(cols: np.ndarray) -> list[tuple[int, int]]:
    cols = [int(c) for c in cols]
    if not cols:
        return []
    runs: list[tuple[int, int]] = []
    start = prev = cols[0]
    for col in cols[1:]:
        if col == prev + 1:
            prev = col
            continue
        runs.append((start, prev))
        start = prev = col
    runs.append((start, prev))
    return runs


def _nearest_outside(col: int, left: int, right: int, width: int) -> int:
    candidates: list[tuple[int, int, int]] = []
    if left - 1 >= 0:
        candidates.append((col - (left - 1), 0, left - 1))
    if right + 1 < width:
        candidates.append(((right + 1) - col, 1, right + 1))
    assert candidates
    return min(candidates, key=lambda item: (item[0], -item[1]))[2]


def _find_obstacles(grid: np.ndarray) -> list[dict[str, int]]:
    obstacles: list[dict[str, int]] = []
    for row in range(grid.shape[0]):
        runs = _contiguous_runs(np.flatnonzero(grid[row] == Color.GREY))
        for left, right in runs:
            obstacles.append({"row": int(row), "left": int(left), "right": int(right)})
    return obstacles


def _trace_path(
    height: int,
    width: int,
    source_col: int,
    obstacle_by_cell: dict[tuple[int, int], tuple[int, int]],
) -> tuple[list[tuple[int, int]], int]:
    r = 0
    c = int(source_col)
    cells = [(0, c)]
    hits = 0
    seen_states = {(r, c, hits)}

    while r < height - 1:
        below = (r + 1, c)
        if below not in obstacle_by_cell:
            r += 1
            cells.append((r, c))
        else:
            hits += 1
            left, right = obstacle_by_cell[below]
            side_col = _nearest_outside(c, left, right, width)
            step = 1 if side_col > c else -1
            for cc in range(c + step, side_col + step, step):
                cells.append((r, cc))
            c = side_col

        state = (r, c, hits)
        if state in seen_states:
            raise AssertionError("path simulation looped")
        seen_states.add(state)

    return cells, hits


def _path_lengths_and_colors(grid: np.ndarray) -> list[tuple[int, int, int]]:
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    bg = get_bg(g)

    source_cols = np.flatnonzero((g[0] != bg) & (g[0] != Color.GREY))
    assert 3 <= len(source_cols) <= 5

    source_info: list[tuple[int, int]] = []
    for col in source_cols:
        color = int(g[0, int(col)])
        source_info.append((int(col), color))
    colors = [color for _, color in source_info]
    assert len(colors) == len(set(colors))

    obstacles = _find_obstacles(g)
    assert 2 <= len(obstacles) <= 4
    obstacle_by_cell: dict[tuple[int, int], tuple[int, int]] = {}
    for obstacle in obstacles:
        row = obstacle["row"]
        left = obstacle["left"]
        right = obstacle["right"]
        assert row > 0
        assert 1 <= left <= right <= width - 2
        assert 2 <= right - left + 1 <= 5
        for col in range(left, right + 1):
            obstacle_by_cell[(row, col)] = (left, right)

    specs: list[tuple[int, int, int]] = []
    for col, color in source_info:
        cells, _hits = _trace_path(height, width, col, obstacle_by_cell)
        specs.append((len(cells), col, color))
    return specs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Order the top source colors by the total length of their obstacle-detour paths.

    Input structure:
    - The background is black.
    - Row 0 contains 3 to 5 isolated source pixels, all with different colors chosen from blue, red, green, yellow, and orange.
    - The middle rows contain 2 to 4 grey horizontal obstacle segments of length 2 to 5 that do not touch the left or right border.

    Rule:
    - For each source pixel, follow its path from the top downward one cell at a time.
    - When the next cell below is part of a grey obstacle segment, move horizontally on the row just above that segment to the nearer outside column; exact midpoint ties detour to the right.
    - Continue downward from that outside column, count every visited path cell, and sort the source colors by increasing total path length.
    - If two paths have the same length, break the tie by the source column from left to right.

    Key insight:
    - The grey barriers encode hidden numeric path lengths: the output is not a traced picture, but the source colors reordered by how much detouring each path must do.
    """
    ordered = sorted(_path_lengths_and_colors(grid), key=lambda item: (item[0], item[1]))
    return np.array([[color for _, _, color in ordered]], dtype=int)


def _sample_source_columns(width: int, count: int) -> list[int]:
    usable = width - 2
    compressed = sorted(random.sample(range(usable - count + 1), count))
    return [1 + value + idx for idx, value in enumerate(compressed)]


def _segments_for_move(source_col: int, dest_col: int, width: int) -> list[tuple[int, int, bool]]:
    segments: list[tuple[int, int, bool]] = []
    if dest_col < source_col:
        dist = source_col - dest_col
        left = dest_col + 1
        right_min = source_col + dist
        right_max = min(width - 2, left + 4)
        for right in range(right_min, right_max + 1):
            length = right - left + 1
            if 2 <= length <= 5:
                segments.append((left, right, False))
        return segments

    dist = dest_col - source_col
    right = dest_col - 1
    left_min = max(1, right - 4)
    left_max = min(source_col, source_col - dist + 1)
    for left in range(left_min, left_max + 1):
        length = right - left + 1
        if not 2 <= length <= 5:
            continue
        left_dist = source_col - left + 1
        if dist <= left_dist:
            segments.append((left, right, dist == left_dist))
    return segments


def _choose_blocked_specs(
    width: int,
    source_cols: list[int],
    blocked_cols: list[int],
    need_midpoint_tie: bool,
) -> list[dict[str, int | bool]] | None:
    source_set = set(source_cols)
    used_dests: set[int] = set()
    specs: list[dict[str, int | bool]] = []
    midpoint_tie_used = False

    for source_col in random.sample(blocked_cols, len(blocked_cols)):
        candidates: list[dict[str, int | bool]] = []
        for delta in (-3, -2, -1, 1, 2, 3):
            dest_col = source_col + delta
            if not 0 <= dest_col < width:
                continue
            if dest_col in source_set or dest_col in used_dests:
                continue
            lo, hi = sorted((source_col, dest_col))
            if any(other != source_col and lo <= other <= hi for other in source_cols):
                continue
            for left, right, is_midpoint_tie in _segments_for_move(source_col, dest_col, width):
                if any(other != source_col and left <= other <= right for other in source_cols):
                    continue
                candidates.append(
                    {
                        "source": source_col,
                        "dest": dest_col,
                        "left": left,
                        "right": right,
                        "dist": abs(dest_col - source_col),
                        "midpoint_tie": is_midpoint_tie,
                    }
                )

        if not candidates:
            return None

        if need_midpoint_tie and not midpoint_tie_used:
            tie_candidates = [candidate for candidate in candidates if candidate["midpoint_tie"]]
            if tie_candidates:
                candidates = tie_candidates
        candidate = random.choice(candidates)
        midpoint_tie_used = midpoint_tie_used or bool(candidate["midpoint_tie"])
        used_dests.add(int(candidate["dest"]))
        specs.append(candidate)

    if need_midpoint_tie and not midpoint_tie_used:
        return None
    return specs


def _sample_rows(height: int, count: int) -> list[int] | None:
    choices = list(range(2, height - 1))
    for _ in range(200):
        rows = sorted(random.sample(choices, count))
        if all(rows[i + 1] - rows[i] >= 2 for i in range(count - 1)):
            return rows
    return None


def _build_grid(
    height: int,
    width: int,
    source_cols: list[int],
    source_colors: list[int],
    blocked_specs: list[dict[str, int | bool]],
    obstacle_rows: list[int],
) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    for col, color in zip(source_cols, source_colors):
        grid[0, col] = color
    for row, spec in zip(obstacle_rows, blocked_specs):
        left = int(spec["left"])
        right = int(spec["right"])
        grid[row, left : right + 1] = Color.GREY
    return grid


def _validate_generated_case(
    grid: np.ndarray,
    blocked_specs: list[dict[str, int | bool]],
    require_length_tie: bool,
) -> bool:
    height, width = grid.shape
    source_cols = [int(col) for col in np.flatnonzero((grid[0] != 0) & (grid[0] != Color.GREY))]
    obstacles = _find_obstacles(grid)
    if len(obstacles) != len(blocked_specs):
        return False

    spans_by_row = {(ob["row"], col) for ob in obstacles for col in range(ob["left"], ob["right"] + 1)}
    obstacle_by_cell = {
        (ob["row"], col): (ob["left"], ob["right"])
        for ob in obstacles
        for col in range(ob["left"], ob["right"] + 1)
    }

    final_cols: set[int] = set()
    for col in source_cols:
        final_cols.add(col)
    for spec in blocked_specs:
        final_cols.discard(int(spec["source"]))
        final_cols.add(int(spec["dest"]))

    if len(final_cols) != len(source_cols):
        return False

    for final_col in final_cols:
        if any((ob["row"], final_col) in spans_by_row for ob in obstacles):
            return False

    path_cells: list[set[tuple[int, int]]] = []
    lengths: list[int] = []
    blocked_offsets: list[int] = []
    hit_counts: list[int] = []
    for col in source_cols:
        cells, hits = _trace_path(height, width, col, obstacle_by_cell)
        cell_set = set(cells)
        if len(cell_set) != len(cells):
            return False
        path_cells.append(cell_set)
        lengths.append(len(cells))
        hit_counts.append(hits)
        blocked_offsets.append(len(cells) - height)

    total = sum(len(cells) for cells in path_cells)
    if len(set().union(*path_cells)) != total:
        return False

    if sorted(hit_counts).count(0) == len(hit_counts):
        return False
    if sorted(hit_counts).count(1) != len(blocked_specs):
        return False
    if len(set(lengths)) < 2:
        return False

    blocked_dists = [int(spec["dist"]) for spec in blocked_specs]
    if len(set(blocked_dists)) < 2:
        return False
    if require_length_tie and len(set(lengths)) == len(lengths):
        return False

    source_left_to_right = [int(grid[0, col]) for col in source_cols]
    source_right_to_left = list(reversed(source_left_to_right))
    output_colors = list(main(grid)[0])
    if output_colors == source_left_to_right or output_colors == source_right_to_left:
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GEN_HEIGHT_RANGE)
    width = random.randint(*GEN_WIDTH_RANGE)
    source_count = random.randint(*GEN_SOURCE_COUNT_RANGE)
    source_cols = _sample_source_columns(width, source_count)
    source_colors = random.sample(list(SOURCE_COLORS), source_count)

    obstacle_count = random.randint(GEN_OBSTACLE_COUNT_RANGE[0], min(GEN_OBSTACLE_COUNT_RANGE[1], source_count))
    blocked_cols = random.sample(source_cols, obstacle_count)
    require_length_tie = random.random() < 0.7
    require_midpoint_tie = random.random() < 0.5

    blocked_specs = _choose_blocked_specs(width, source_cols, blocked_cols, require_midpoint_tie)
    if blocked_specs is None:
        return None

    obstacle_rows = _sample_rows(height, obstacle_count)
    if obstacle_rows is None:
        return None

    random.shuffle(blocked_specs)
    grid = _build_grid(height, width, source_cols, source_colors, blocked_specs, obstacle_rows)
    if not _validate_generated_case(grid, blocked_specs, require_length_tie):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid path-sorting ARC input")
