import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
TEAL = 8
MAROON = 9

UPPER_COLORS = (TEAL, GREEN, GREY)
OBSTACLE_COLORS = (PINK, MAROON)
PAIRED_COLOR = {TEAL: RED, GREEN: BLUE, GREY: YELLOW}
MAX_GENERATION_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _find_separator_row(grid: np.ndarray) -> int:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r] == GREY)]
    assert len(rows) == 1, "expected exactly one full-width grey separator row"
    return int(rows[0])


def _basin_mask(grid: np.ndarray, separator_row: int) -> tuple[np.ndarray, list[np.ndarray]]:
    h, w = grid.shape
    lower = grid[separator_row + 1 :] == BLACK
    comps = _components_mask(lower)
    basin = np.zeros((h, w), dtype=bool)
    basin_components: list[np.ndarray] = []
    for comp in comps:
        global_comp = comp.copy()
        global_comp[:, 0] += separator_row + 1
        touches_bottom_or_side = bool(
            np.any(global_comp[:, 0] == h - 1)
            or np.any(global_comp[:, 1] == 0)
            or np.any(global_comp[:, 1] == w - 1)
        )
        if not touches_bottom_or_side:
            basin[global_comp[:, 0], global_comp[:, 1]] = True
            basin_components.append(global_comp)
    return basin, basin_components


def _simulate_fall(grid: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    sep = _find_separator_row(g)
    out = g.copy()

    falling: list[tuple[int, int, int]] = []
    for r in range(sep):
        for c in range(w):
            color = int(out[r, c])
            if color in UPPER_COLORS:
                falling.append((r, c, color))
                out[r, c] = BLACK

    out[sep, :] = BLACK
    landed: list[tuple[int, int, int, int]] = []

    by_col: dict[int, list[tuple[int, int]]] = {}
    for r, c, color in falling:
        by_col.setdefault(c, []).append((r, color))

    for c, entries in by_col.items():
        for r, color in sorted(entries, reverse=True):
            rr = r
            while rr + 1 < h and out[rr + 1, c] == BLACK:
                rr += 1
            out[rr, c] = color
            landed.append((r, c, rr, color))

    return out, landed


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop the upper colored pixels through a removed divider, then recolor only the ones that settle inside precomputed enclosed basins.
    
    Input structure:
    - A black background grid contains a single full-width grey divider row that separates an upper source area from a lower obstacle area.
    - Above the divider are isolated teal, green, and grey pixels.
    - Below the divider, pink and maroon obstacle cells form walls, shelves, and pocket-like structures over black empty space.
    
    Rule:
    - Before moving anything, inspect only the black cells below the grey divider and split them into 4-connected regions.
    - Any such region that does not touch the grid bottom edge or the left/right border is a basin; all other lower black regions are open.
    - Remove the grey divider to black, then let each upper colored pixel fall straight down within its own column until blocked by an obstacle, a previously settled pixel, or the grid bottom; process lower source pixels before higher ones so same-column stacks form correctly.
    - A settled pixel changes by the fixed pair mapping teal->red, green->blue, grey->yellow iff its final cell belonged to a precomputed basin cell; otherwise it keeps its original color.
    - Obstacles stay unchanged.
    
    Key insight:
    - The color change is determined by the topology of the lower empty-space component computed before the divider is removed, not merely by how deep the pixel falls or by the obstacle color below it.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    sep = _find_separator_row(g)
    basin_mask, _ = _basin_mask(g, sep)
    out, landed = _simulate_fall(g)

    for _src_r, c, rr, color in landed:
        if basin_mask[rr, c]:
            out[rr, c] = PAIRED_COLOR[color]

    return out


def _paint_obstacles(grid: np.ndarray, obstacle_mask: np.ndarray) -> None:
    rows, cols = np.nonzero(obstacle_mask)
    for r, c in zip(rows.tolist(), cols.tolist()):
        grid[r, c] = PINK if (r + 2 * c) % 3 else MAROON


def _draw_left_open_pocket(mask: np.ndarray, sep: int, width: int, floor_row: int) -> list[int]:
    right_wall = width - 1
    mask[sep + 1 :, right_wall] = True
    mask[floor_row, :width] = True
    if floor_row + 1 < mask.shape[0] and width >= 4:
        mask[floor_row + 1 :, max(0, right_wall - 2)] = True
    return list(range(0, max(1, width - 1)))


def _draw_right_open_pocket(mask: np.ndarray, sep: int, start: int, floor_row: int) -> list[int]:
    left_wall = start
    mask[sep + 1 :, left_wall] = True
    mask[floor_row, start:] = True
    if floor_row + 1 < mask.shape[0] and mask.shape[1] - start >= 4:
        mask[floor_row + 1 :, min(mask.shape[1] - 1, left_wall + 2)] = True
    return list(range(min(mask.shape[1] - 1, start + 1), mask.shape[1]))


def _draw_basin(mask: np.ndarray, sep: int, left: int, right: int, floor_row: int) -> list[int]:
    mask[sep + 1 :, left] = True
    mask[sep + 1 :, right] = True
    mask[floor_row, left : right + 1] = True
    if floor_row + 1 < mask.shape[0] and right - left >= 4:
        if random.random() < 0.5:
            mask[floor_row + 1 :, left + 1] = True
        else:
            mask[floor_row + 1 :, right - 1] = True
    return list(range(left + 1, right))


def _structure_columns(mask: np.ndarray) -> set[int]:
    return {int(c) for c in np.nonzero(mask)[1].tolist()}


def _choose_spaced_columns(candidates: list[int], used: set[int], count: int) -> list[int] | None:
    pool = candidates[:]
    random.shuffle(pool)
    chosen = []
    for c in pool:
        if all(abs(c - u) >= 2 for u in used | set(chosen)):
            chosen.append(c)
            if len(chosen) == count:
                return chosen
    return None


def _place_upper_pixels(
    grid: np.ndarray,
    sep: int,
    column_counts: dict[int, int],
    colors_by_column: dict[int, list[int]],
) -> None:
    for c, count in column_counts.items():
        rows = list(range(sep - 1, -1, -2))[:count]
        assert len(rows) == count
        for r, color in zip(rows, colors_by_column[c]):
            grid[r, c] = color


def _independent_output(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int, int, int]]]:
    sep = _find_separator_row(grid)
    basin_mask, basin_components = _basin_mask(grid, sep)
    assert 2 <= len(basin_components) <= 3

    h, w = grid.shape
    work = grid.copy()
    source_cells = [(r, c, int(grid[r, c])) for r in range(sep) for c in range(w) if int(grid[r, c]) in UPPER_COLORS]
    for r, c, _color in source_cells:
        work[r, c] = BLACK
    work[sep, :] = BLACK

    landed: list[tuple[int, int, int, int]] = []
    for c in range(w):
        entries = [(r, color) for r, cc, color in source_cells if cc == c]
        for r, color in sorted(entries, reverse=True):
            rr = r
            while rr + 1 < h and work[rr + 1, c] == BLACK:
                rr += 1
            work[rr, c] = color
            landed.append((r, c, rr, color))

    out = work.copy()
    for _src_r, c, rr, color in landed:
        if basin_mask[rr, c]:
            out[rr, c] = PAIRED_COLOR[color]
    return out, basin_mask, landed


def _validate_example(grid: np.ndarray, expect_overflow: bool) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False

    h, w = grid.shape
    sep = _find_separator_row(grid)
    if sep not in (4, 5):
        return False

    if not np.all(grid[sep] == GREY):
        return False

    upper = grid[:sep]
    lower = grid[sep + 1 :]

    for r in range(sep):
        for c in range(w):
            color = int(upper[r, c])
            if color == BLACK:
                continue
            if color not in UPPER_COLORS:
                return False
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < sep and 0 <= nc < w and upper[nr, nc] != BLACK:
                    return False

    lower_colors = set(np.unique(lower).tolist())
    if not lower_colors.issubset({BLACK, PINK, MAROON}):
        return False

    source_colors = {TEAL: 0, GREEN: 0, GREY: 0}
    for color in upper.flatten().tolist():
        if color in source_colors:
            source_colors[int(color)] += 1
    if any(not (2 <= cnt <= 4) for cnt in source_colors.values()):
        return False

    expected, basin_mask, landed = _independent_output(grid)
    actual = main(grid)
    if not np.array_equal(expected, actual):
        return False

    basin_hits = []
    open_hits = []
    overflow_hits = []
    for _src_r, c, rr, color in landed:
        if basin_mask[rr, c]:
            basin_hits.append((c, rr, color))
        else:
            open_hits.append((c, rr, color))
            if rr <= sep or (rr + 1 < h and basin_mask[rr + 1, c]):
                overflow_hits.append((c, rr, color))

    if not basin_hits or not open_hits:
        return False

    basin_input_colors = {color for _c, _rr, color in basin_hits}
    open_input_colors = {color for _c, _rr, color in open_hits}
    if not (basin_input_colors & open_input_colors):
        return False

    if expect_overflow and not overflow_hits:
        return False

    output_colors = expected.flatten().tolist()
    if not any(color in (RED, BLUE, YELLOW) for color in output_colors):
        return False

    return True


def _build_attempt() -> tuple[np.ndarray, bool] | None:
    h = random.randint(14, 18)
    w = random.randint(15, 20)
    sep = random.randint(4, 5)
    overflow = sep == 5 and random.random() < 0.45

    basin_count = 3 if w >= 17 and random.random() < 0.45 else 2
    pocket_side = random.choice(("left", "right"))
    pocket_width = random.randint(4, 5 if basin_count == 2 else 4)
    pocket_floor = random.randint(sep + 3, h - 3)

    if basin_count == 3:
        basin_widths = [random.randint(3, 4) for _ in range(basin_count)]
        gap_after_pocket = 1
        basin_gaps = [1 for _ in range(basin_count - 1)]
    else:
        basin_widths = [random.randint(4, 5) for _ in range(basin_count)]
        gap_after_pocket = random.randint(1, 2)
        basin_gaps = [random.randint(1, 2) for _ in range(basin_count - 1)]

    placements: list[tuple[int, int]] = []
    if pocket_side == "left":
        cursor = pocket_width + gap_after_pocket
        for i, bw in enumerate(basin_widths):
            left = cursor
            right = left + bw - 1
            placements.append((left, right))
            cursor = right + 1 + (basin_gaps[i] if i < len(basin_gaps) else 0)
        if placements[-1][1] > w - 2:
            return None
        pocket_start = 0
    else:
        cursor = 1
        for i, bw in enumerate(basin_widths):
            left = cursor
            right = left + bw - 1
            placements.append((left, right))
            cursor = right + 1 + (basin_gaps[i] if i < len(basin_gaps) else 0)
        pocket_start = w - pocket_width
        if cursor + gap_after_pocket - 1 > pocket_start - 1:
            return None

    obstacle_mask = np.zeros((h, w), dtype=bool)
    if pocket_side == "left":
        open_columns = _draw_left_open_pocket(obstacle_mask, sep, pocket_width, pocket_floor)
    else:
        open_columns = _draw_right_open_pocket(obstacle_mask, sep, pocket_start, pocket_floor)

    basin_columns: list[list[int]] = []
    chosen_overflow_index = random.randrange(basin_count) if overflow else -1
    basin_floors = []
    for i, (left, right) in enumerate(placements):
        if right >= w - 1 or left <= 0:
            return None
        floor_row = sep + 3 if i == chosen_overflow_index else random.randint(sep + 3, h - 3)
        basin_floors.append(floor_row)
        basin_columns.append(_draw_basin(obstacle_mask, sep, left, right, floor_row))

    grid = np.zeros((h, w), dtype=int)
    grid[sep, :] = GREY
    _paint_obstacles(grid, obstacle_mask)

    basin_mask, basin_components = _basin_mask(grid, sep)
    if len(basin_components) != basin_count:
        return None

    used_columns = set()
    pocket_candidates = [c for c in open_columns if all(abs(c - u) >= 2 for u in used_columns)]
    if len(pocket_candidates) < 1:
        return None

    open_choice = random.choice(pocket_candidates)
    used_columns.add(open_choice)

    basin_choice_by_index: dict[int, int] = {}
    for idx in range(basin_count):
        cols = _choose_spaced_columns(basin_columns[idx], used_columns, 1)
        if cols is None:
            return None
        basin_choice_by_index[idx] = cols[0]
        used_columns.update(cols)

    if len(open_columns) >= 3:
        extra_open = _choose_spaced_columns(open_columns, used_columns, 1)
        open_columns_used = [open_choice] + (extra_open or [])
        used_columns.update(extra_open or [])
    else:
        open_columns_used = [open_choice]

    column_counts: dict[int, int] = {}
    target_categories: list[str] = []

    column_counts[open_columns_used[0]] = 1
    target_categories.append("open")
    for idx in range(basin_count):
        c = basin_choice_by_index[idx]
        column_counts[c] = 1
        target_categories.append("basin")
    for c in open_columns_used[1:]:
        column_counts[c] = 1
        target_categories.append("open")

    total_pixels = 6 + random.randint(0, 2)
    if overflow:
        overflow_col = basin_choice_by_index[chosen_overflow_index]
        column_counts[overflow_col] = 3
    current_total = sum(column_counts.values())

    max_capacity: dict[int, int] = {}
    for c in column_counts:
        capacity = len(list(range(sep - 1, -1, -2)))
        for basin_idx, basin_col in basin_choice_by_index.items():
            if basin_col == c and basin_idx != chosen_overflow_index:
                basin_height = basin_floors[basin_idx] - (sep + 1)
                capacity = min(capacity, basin_height)
                break
        max_capacity[c] = capacity

    expandable = list(column_counts.keys())
    while current_total < total_pixels:
        c = random.choice(expandable)
        if column_counts[c] >= max_capacity[c]:
            if all(column_counts[col] >= max_capacity[col] for col in expandable):
                break
            continue
        column_counts[c] += 1
        current_total += 1

    total_pixels = sum(column_counts.values())
    if not (6 <= total_pixels <= 9):
        return None

    counts = {TEAL: 2, GREEN: 2, GREY: 2}
    while sum(counts.values()) < total_pixels:
        color = random.choice([TEAL, GREEN, GREY])
        if counts[color] < 4:
            counts[color] += 1

    shared_color = random.choice([TEAL, GREEN, GREY])
    if counts[shared_color] < 2:
        return None

    basin_column_list = [basin_choice_by_index[idx] for idx in range(basin_count)]
    open_column_list = open_columns_used[:]
    colors_by_column: dict[int, list[int]] = {c: [] for c in column_counts}
    colors_by_column[open_column_list[0]].append(shared_color)
    counts[shared_color] -= 1
    colors_by_column[basin_column_list[0]].append(shared_color)
    counts[shared_color] -= 1

    remaining = []
    for color, count in counts.items():
        remaining.extend([color] * count)
    random.shuffle(remaining)

    fill_order = []
    for c, count in sorted(column_counts.items()):
        reserved = len(colors_by_column[c])
        fill_order.extend([c] * (count - reserved))
    random.shuffle(fill_order)
    if len(fill_order) != len(remaining):
        return None
    for c, color in zip(fill_order, remaining):
        colors_by_column[c].append(color)

    for c, colors in colors_by_column.items():
        if len(colors) != column_counts[c]:
            return None

    _place_upper_pixels(grid, sep, column_counts, colors_by_column)

    if not _validate_example(grid, overflow):
        return None
    return grid, overflow


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        built = _build_attempt()
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate valid input for puzzle 6")
