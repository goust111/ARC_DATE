import random

import numpy as np


BACKGROUND = 0
OBSTACLE = 2
MARKER_COLORS = (1, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 400


def _obstacle_segment(grid: np.ndarray, row: int, col: int) -> tuple[int, int]:
    left = int(col)
    while left - 1 >= 0 and int(grid[row, left - 1]) == OBSTACLE:
        left -= 1

    right = int(col)
    width = grid.shape[1]
    while right + 1 < width and int(grid[row, right + 1]) == OBSTACLE:
        right += 1
    return left, right


def _choose_bypass_column(left: int, right: int, col: int, width: int) -> int | None:
    options = []
    if left - 1 >= 0:
        options.append((int(col - left), 1, left - 1))
    if right + 1 < width:
        options.append((int(right - col), 0, right + 1))
    if not options:
        return None
    return int(min(options)[2])


def _propagate_from_marker(src: np.ndarray, out: np.ndarray, start_col: int, color: int) -> None:
    height, width = src.shape
    row = height - 1
    col = int(start_col)
    current_color = int(color)

    while 0 <= row < height:
        cell = int(src[row, col])
        if cell == BACKGROUND:
            out[row, col] = current_color
        elif cell == current_color:
            out[row, col] = current_color
        else:
            break

        if row == 0:
            break

        next_row = row - 1
        next_cell = int(src[next_row, col])
        if next_cell == OBSTACLE:
            left, right = _obstacle_segment(src, next_row, col)
            target_col = _choose_bypass_column(left, right, col, width)
            if target_col is None:
                break

            step = 1 if target_col > col else -1
            for cc in range(col, target_col + step, step):
                if int(src[row, cc]) not in (BACKGROUND, current_color):
                    return
                out[row, cc] = current_color

            col = target_col
            row = next_row - 1
            continue

        if next_cell not in (BACKGROUND, current_color):
            break
        row = next_row


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate each bottom-row marker color upward, detouring around blocking red horizontal segments.

    Input structure:
    - The background is black, and the bottom row contains 2-3 non-black singleton markers that select the columns to process.
    - Each selected column contains several scattered support pixels of the same color as its bottom marker.
    - Some rows contain red horizontal obstacle segments that may cross a selected column.

    Rule:
    - For every marked bottom-row column, scan upward from the marker while keeping the current propagation color equal to the marker color.
    - Replace background cells in that active column with the propagation color, and preserve same-color support pixels.
    - If the next cell above is red, find that obstacle segment's nearer end, draw a horizontal bypass on the row just below the obstacle to one column beyond that end, then continue upward from the bypass column starting one row above the obstacle.
    - Stop a propagation when it reaches the top row or when the next position would be a non-background, non-matching color; leave unmarked columns and red obstacles unchanged.

    Key insight:
    - The selected bottom markers define vertical color bands, but a band can jump sideways exactly once it meets a red horizontal blocker, continuing upward from the nearest free side.
    """
    src = np.asarray(grid, dtype=int)
    assert src.ndim == 2

    out = src.copy()
    marker_cols = np.flatnonzero(src[-1] != BACKGROUND)
    for col in marker_cols:
        _propagate_from_marker(src, out, int(col), int(src[-1, col]))
    return out


def _sample_marker_columns(width: int, count: int) -> list[int] | None:
    candidates = list(range(2, width - 2))
    for _ in range(200):
        cols = sorted(random.sample(candidates, count))
        if all(b - a >= 4 for a, b in zip(cols, cols[1:])):
            return cols
    return None


def _candidate_bypasses(
    width: int,
    source_col: int,
    all_source_cols: list[int],
    chosen_targets: list[int],
    reserved_ranges: list[tuple[int, int]],
) -> list[dict]:
    occupied_columns = set(all_source_cols) | set(chosen_targets)
    candidates = []

    for left_dist in range(0, 4):
        for right_dist in range(0, 4):
            seg_len = left_dist + right_dist + 1
            if not (3 <= seg_len <= 6):
                continue

            left = source_col - left_dist
            right = source_col + right_dist
            if left < 1 or right > width - 2:
                continue

            if any(left <= col <= right for col in occupied_columns if col != source_col):
                continue

            left_target = left - 1
            if left_target >= 0 and right_dist > left_dist:
                if left_target not in occupied_columns and not any(
                    lo <= left_target <= hi for lo, hi in reserved_ranges
                ):
                    candidates.append(
                        {
                            "side": "left",
                            "left": left,
                            "right": right,
                            "target": left_target,
                        }
                    )

            right_target = right + 1
            if right_target < width and right_dist <= left_dist:
                if right_target not in occupied_columns and not any(
                    lo <= right_target <= hi for lo, hi in reserved_ranges
                ):
                    candidates.append(
                        {
                            "side": "right",
                            "left": left,
                            "right": right,
                            "target": right_target,
                        }
                    )

    return candidates


def _choose_obstacle_rows(height: int, count: int) -> list[int] | None:
    rows = []
    for _ in range(count):
        candidates = [r for r in range(3, height - 5) if all(abs(r - prev) >= 2 for prev in rows)]
        if not candidates:
            return None
        rows.append(random.choice(candidates))
    return rows


def _place_support_pixels(grid: np.ndarray, spec: dict) -> bool:
    height = grid.shape[0]
    source_col = spec["source"]
    color = spec["color"]

    if spec["blocked"]:
        candidate_rows = list(range(spec["obstacle_row"] + 2, height - 1))
    else:
        candidate_rows = list(range(1, height - 1))

    if len(candidate_rows) < 2:
        return False

    random.shuffle(candidate_rows)
    count = random.randint(2, min(4, len(candidate_rows)))
    for row in sorted(candidate_rows[:count]):
        if int(grid[row, source_col]) != BACKGROUND:
            return False
        grid[row, source_col] = color
    return True


def _validate_generated(grid: np.ndarray, specs: list[dict]) -> bool:
    out = main(grid)
    if np.array_equal(out, grid):
        return False
    if not np.array_equal(out[grid == OBSTACLE], grid[grid == OBSTACLE]):
        return False
    if not any(spec["blocked"] for spec in specs):
        return False
    if not any(not spec["blocked"] for spec in specs):
        return False

    for spec in specs:
        source_col = spec["source"]
        color = spec["color"]

        if spec["blocked"]:
            obstacle_row = spec["obstacle_row"]
            target_col = spec["target"]
            changed_above = np.count_nonzero(
                (grid[:obstacle_row, target_col] == BACKGROUND)
                & (out[:obstacle_row, target_col] == color)
            )
            if changed_above == 0:
                return False

            bypass_row = obstacle_row + 1
            lo, hi = sorted((source_col, target_col))
            if not np.all(out[bypass_row, lo : hi + 1] == color):
                return False
        else:
            changed_in_col = np.count_nonzero(
                (grid[:, source_col] == BACKGROUND) & (out[:, source_col] == color)
            )
            if changed_in_col == 0:
                return False
    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(16, 24)
    width = random.randint(14, 22)
    marker_count = random.choice([2, 3])
    blocked_count = random.randint(1, marker_count - 1)

    marker_cols = _sample_marker_columns(width, marker_count)
    if marker_cols is None:
        return None

    blocked_indices = set(random.sample(range(marker_count), blocked_count))
    colors = random.sample(MARKER_COLORS, marker_count)

    specs: list[dict] = []
    chosen_targets: list[int] = []
    reserved_ranges: list[tuple[int, int]] = []
    for idx, source_col in enumerate(marker_cols):
        spec = {
            "source": int(source_col),
            "color": int(colors[idx]),
            "blocked": idx in blocked_indices,
            "target": int(source_col),
        }

        if spec["blocked"]:
            candidates = _candidate_bypasses(
                width,
                int(source_col),
                marker_cols,
                chosen_targets,
                reserved_ranges,
            )
            if not candidates:
                return None

            bypass = random.choice(candidates)
            spec.update(bypass)
            chosen_targets.append(int(spec["target"]))
            reserved_ranges.append((int(spec["left"]), int(spec["right"])))
            reserved_ranges.append(
                (min(int(spec["source"]), int(spec["target"])), max(int(spec["source"]), int(spec["target"])))
            )

        specs.append(spec)

    blocked_specs = [spec for spec in specs if spec["blocked"]]
    obstacle_rows = _choose_obstacle_rows(height, len(blocked_specs))
    if obstacle_rows is None:
        return None

    for spec, row in zip(blocked_specs, obstacle_rows):
        spec["obstacle_row"] = int(row)

    grid = np.zeros((height, width), dtype=int)
    for spec in specs:
        grid[height - 1, spec["source"]] = spec["color"]

    for spec in blocked_specs:
        grid[spec["obstacle_row"], spec["left"] : spec["right"] + 1] = OBSTACLE

    for spec in specs:
        if not _place_support_pixels(grid, spec):
            return None

    if _validate_generated(grid, specs):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
