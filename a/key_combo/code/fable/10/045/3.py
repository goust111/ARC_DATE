import random
from collections import Counter, defaultdict

import numpy as np


BG_COLOR = 0
OBSTACLE_COLOR = 5
SOURCE_COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
RECT_DIMS = tuple((h, w) for h in range(2, 7) for w in range(2, 5))
MAX_GEN_ATTEMPTS = 600


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _colors_present(grid: np.ndarray, exclude=()) -> list[int]:
    colors = sorted(set(map(int, grid.ravel())) - set(map(int, exclude)))
    return colors


def _nearest_side(col: int, left: int, right: int, width: int) -> int:
    candidates = []
    if left - 1 >= 0:
        candidates.append((col - (left - 1), 0, left - 1))
    if right + 1 < width:
        candidates.append(((right + 1) - col, 1, right + 1))
    assert candidates, "obstacle segment must have an outside side"
    return min(candidates, key=lambda item: (item[0], -item[1]))[2]


def _parse_scene(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)

    top_non_bg = np.flatnonzero(g[0] != bg)
    assert len(top_non_bg) >= 2, "expected at least two source pixels on the top row"

    sources = []
    source_colors = []
    for c in top_non_bg:
        color = int(g[0, c])
        assert color not in source_colors, "source colors must be distinct"
        assert np.count_nonzero(g[0] == color) == 1, "each source color must appear once in row 0"
        sources.append((color, int(c)))
        source_colors.append(color)

    extra_colors = _colors_present(g, exclude={bg, *source_colors})
    assert len(extra_colors) == 1, "expected exactly one obstacle color"
    obstacle_color = extra_colors[0]

    fuel_pixels_by_color = {}
    for color, _ in sources:
        coords = [tuple(map(int, rc)) for rc in np.argwhere(g == color) if int(rc[0]) != 0]
        assert coords, "each source color must have a fuel blob"
        fuel_pixels_by_color[color] = sorted(coords)

    return g, bg, obstacle_color, sources, fuel_pixels_by_color


def _trace_full_path(grid: np.ndarray, start_col: int, obstacle_color: int) -> list[tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    r, c = 0, int(start_col)
    path = [(r, c)]

    while r < height - 1:
        if g[r + 1, c] == obstacle_color:
            left = c
            while left - 1 >= 0 and g[r + 1, left - 1] == obstacle_color:
                left -= 1
            right = c
            while right + 1 < width and g[r + 1, right + 1] == obstacle_color:
                right += 1
            side_col = _nearest_side(c, left, right, width)
            step = 1 if side_col > c else -1
            for nc in range(c + step, side_col + step, step):
                path.append((r, nc))
            c = side_col
            continue

        r += 1
        path.append((r, c))

    return path


def _actual_path(full_path: list[tuple[int, int]], fuel_count: int) -> list[tuple[int, int]]:
    steps = min(int(fuel_count), len(full_path) - 1)
    return full_path[: steps + 1]


def main(grid: np.ndarray) -> np.ndarray:
    """Extend each top-row source downward while spending same-colored fuel and detouring around horizontal obstacles.

    Input structure:
    - The dominant color is the background. Row 0 contains several single source pixels of distinct colors.
    - A separate foreground color forms one or more horizontal obstacle segments in the middle of the path region.
    - In a blank side area, each source color also appears as one solid fuel blob that is disconnected from the path region.

    Rule:
    - Copy the input. For each top-row source, grow a same-colored trail downward.
    - Each time the trail enters a new cell, erase one pixel from the matching fuel blob in top-to-bottom, left-to-right order.
    - If the next downward cell is an obstacle, move horizontally on the row above that obstacle to the nearer outside side of the obstacle segment, using the right side on exact ties, then continue downward from there.
    - When fuel runs out, the trail stops in its current cell. When the trail reaches the bottom row, it also stops, and any unspent fuel pixels stay unchanged. Obstacles stay unchanged throughout.

    Key insight:
    - The side blobs are not decoration: their pixel counts cap each path, and obstacle detours consume extra fuel, so the obstacle layout indirectly decides which trails can still reach the bottom.
    """
    g, bg, obstacle_color, sources, fuel_pixels_by_color = _parse_scene(grid)
    out = g.copy()

    for color, start_col in sources:
        fuel_pixels = fuel_pixels_by_color[color]
        full_path = _trace_full_path(g, start_col, obstacle_color)
        spent = min(len(fuel_pixels), len(full_path) - 1)

        for r, c in fuel_pixels[:spent]:
            out[r, c] = bg
        for r, c in full_path[: spent + 1]:
            out[r, c] = color

    out[g == obstacle_color] = obstacle_color
    return out


def _pick_source_columns(path_left: int, path_right: int, count: int) -> list[int] | None:
    cols = list(range(path_left + 1, path_right))
    if len(cols) < count:
        return None
    for _ in range(150):
        chosen = sorted(random.sample(cols, count))
        if all(b - a >= 5 for a, b in zip(chosen, chosen[1:])):
            return chosen
    return None


def _lane_bounds(source_cols: list[int], path_left: int, path_right: int) -> list[tuple[int, int]]:
    bounds = []
    for i, col in enumerate(source_cols):
        left = path_left if i == 0 else (source_cols[i - 1] + col) // 2 + 1
        right = path_right if i == len(source_cols) - 1 else (col + source_cols[i + 1]) // 2
        bounds.append((left, right))
    return bounds


def _segment_candidates(col: int, lane_left: int, lane_right: int, kind: str | None) -> list[tuple[int, int, int]]:
    candidates = []
    for left in range(max(lane_left + 1, col - 3), min(col, lane_right - 1) + 1):
        for right in range(max(col, left), min(col + 3, lane_right - 1) + 1):
            length = right - left + 1
            if length < 2 or length > 4:
                continue

            left_side = left - 1 if left - 1 >= lane_left else None
            right_side = right + 1 if right + 1 <= lane_right else None
            if left_side is None and right_side is None:
                continue

            if left_side is None:
                side_col = right_side
            elif right_side is None:
                side_col = left_side
            else:
                dl = col - left_side
                dr = right_side - col
                side_col = left_side if dl < dr else right_side

            if kind == "tie":
                if left_side is None or right_side is None:
                    continue
                if (col - left_side) != (right_side - col):
                    continue
                if side_col != right_side:
                    continue
            elif kind == "left" and not side_col < col:
                continue
            elif kind == "right" and not side_col > col:
                continue

            candidates.append((left, right, side_col))
    return candidates


def _choose_rows(height: int, count: int) -> list[int] | None:
    if count == 0:
        return []
    choices = list(range(3, height - 1))
    for _ in range(100):
        rows = sorted(random.sample(choices, count))
        if all(b - a >= 2 for a, b in zip(rows, rows[1:])):
            return rows
    return None


def _path_has_horizontal(path: list[tuple[int, int]]) -> bool:
    return any(r0 == r1 and c0 != c1 for (r0, c0), (r1, c1) in zip(path, path[1:]))


def _pick_blob_dims(full_steps: int, height: int) -> tuple[int, int] | None:
    candidates = list(RECT_DIMS)
    random.shuffle(candidates)
    for h, w in candidates:
        area = h * w
        if random.random() < 0.35 and full_steps > height - 1 and height - 1 <= area < full_steps:
            return h, w
        if random.random() < 0.35 and area > full_steps:
            return h, w
        if area < full_steps:
            return h, w
    return random.choice(RECT_DIMS) if RECT_DIMS else None


def _place_rectangles(
    grid: np.ndarray,
    fuel_side_cols: tuple[int, int],
    source_colors: list[int],
    dims_by_color: dict[int, tuple[int, int]],
) -> bool:
    col0, col1 = fuel_side_cols
    occupied = np.zeros_like(grid, dtype=bool)
    for color in source_colors:
        h, w = dims_by_color[color]
        row_choices = list(range(2, grid.shape[0] - h + 1))
        col_choices = list(range(col0, col1 - w + 2))
        random.shuffle(row_choices)
        placed = False
        for r in row_choices:
            random.shuffle(col_choices)
            for c in col_choices:
                r0 = max(0, r - 1)
                c0 = max(0, c - 1)
                r1 = min(grid.shape[0], r + h + 1)
                c1 = min(grid.shape[1], c + w + 1)
                if occupied[r0:r1, c0:c1].any():
                    continue
                grid[r : r + h, c : c + w] = color
                occupied[r : r + h, c : c + w] = True
                placed = True
                break
            if placed:
                break
        if not placed:
            return False
    return True


def _validate_generated(grid: np.ndarray) -> bool:
    try:
        g, bg, obstacle_color, sources, fuel_pixels_by_color = _parse_scene(grid)
    except AssertionError:
        return False

    used_cells = set()
    has_detour = False
    has_visible_detour = False
    has_short_stop = False
    has_bottom_reach = False
    has_obstacle_sensitive = False

    for color, start_col in sources:
        path = _trace_full_path(g, start_col, obstacle_color)
        fuel = len(fuel_pixels_by_color[color])
        actual = _actual_path(path, fuel)
        full_steps = len(path) - 1

        for cell in actual:
            if cell in used_cells:
                return False
            used_cells.add(cell)

        has_detour |= _path_has_horizontal(path)
        has_visible_detour |= _path_has_horizontal(actual)
        has_short_stop |= fuel < full_steps
        has_bottom_reach |= actual[-1][0] == g.shape[0] - 1
        has_obstacle_sensitive |= full_steps > g.shape[0] - 1 and g.shape[0] - 1 <= fuel < full_steps

    if not has_detour or not has_visible_detour or not has_short_stop or not has_bottom_reach or not has_obstacle_sensitive:
        return False

    out = main(g)
    if np.array_equal(out, g):
        return False
    if _get_bg(g) != bg:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(13, 18)
    path_width = random.randint(12, 15)
    fuel_width = random.randint(7, 8)
    gap = 2
    width = path_width + fuel_width + gap
    if width > 30:
        return None

    fuel_on_left = random.choice([False, True])
    if fuel_on_left:
        fuel_cols = (0, fuel_width - 1)
        path_left, path_right = fuel_width + gap, width - 1
    else:
        path_left, path_right = 0, path_width - 1
        fuel_cols = (path_width + gap, width - 1)

    source_count = random.randint(2, 3)
    source_cols = _pick_source_columns(path_left, path_right, source_count)
    if source_cols is None:
        return None
    lane_bounds = _lane_bounds(source_cols, path_left, path_right)
    source_colors = random.sample(SOURCE_COLOR_POOL, source_count)

    events_by_source = {}
    tie_requested = random.random() < 0.45
    tie_done = False
    any_event = False

    for i, (color, src_col) in enumerate(zip(source_colors, source_cols)):
        lane_left, lane_right = lane_bounds[i]
        max_events = 2 if height >= 15 else 1
        event_count = random.choices([0, 1, 2][: max_events + 1], weights=[1, 3, 2][: max_events + 1], k=1)[0]
        rows = _choose_rows(height, event_count)
        if rows is None:
            return None

        current_col = src_col
        source_events = []
        for event_row in rows:
            preferred = []
            if tie_requested and not tie_done:
                preferred.append("tie")
            preferred.extend(random.sample(["left", "right"], k=2))
            preferred.append(None)

            chosen = None
            for kind in preferred:
                candidates = _segment_candidates(current_col, lane_left, lane_right, kind)
                if candidates:
                    chosen = random.choice(candidates)
                    if kind == "tie":
                        tie_done = True
                    break
            if chosen is None:
                return None

            left, right, side_col = chosen
            source_events.append((event_row, left, right))
            current_col = side_col

        events_by_source[color] = source_events
        any_event |= bool(source_events)

    if not any_event:
        picked = random.randrange(source_count)
        color = source_colors[picked]
        lane_left, lane_right = lane_bounds[picked]
        row = random.randint(3, height - 2)
        candidates = _segment_candidates(source_cols[picked], lane_left, lane_right, "right")
        if not candidates:
            return None
        left, right, _ = random.choice(candidates)
        events_by_source[color] = [(row, left, right)]

    grid = np.full((height, width), BG_COLOR, dtype=int)
    for color, col in zip(source_colors, source_cols):
        grid[0, col] = color

    obstacle_runs = defaultdict(list)
    for color in source_colors:
        for row, left, right in events_by_source[color]:
            obstacle_runs[row].append((left, right))

    for row, runs in obstacle_runs.items():
        runs = sorted(runs)
        for (_, prev_right), (next_left, _) in zip(runs, runs[1:]):
            if prev_right >= next_left - 1:
                return None
        for left, right in runs:
            grid[row, left : right + 1] = OBSTACLE_COLOR

    full_paths = {}
    for color, col in zip(source_colors, source_cols):
        path = _trace_full_path(grid, col, OBSTACLE_COLOR)
        full_paths[color] = path

    dims_by_color = {}
    for color in source_colors:
        full_steps = len(full_paths[color]) - 1
        dims = _pick_blob_dims(full_steps, height)
        if dims is None:
            return None
        dims_by_color[color] = dims

    if not _place_rectangles(grid, fuel_cols, source_colors, dims_by_color):
        return None

    return grid if _validate_generated(grid) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid fuel-guided detour instance")
