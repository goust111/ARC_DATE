import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5

# Fixed visible priority chain used across all generated tasks.
# The solver should not compare raw color ids numerically; it only tests
# membership against this explicit chain.
PRIORITY_ORDER = (2, 8, 3, 6, 1, 4, 7, 9)
PRIORITY_RANK = {color: rank for rank, color in enumerate(PRIORITY_ORDER)}


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _touches_edge(points: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in points)


def _uniform_color(arr: np.ndarray) -> int:
    vals = np.unique(arr)
    assert vals.size == 1, "expected a uniform wall strip"
    color = int(vals[0])
    assert color not in (BLACK, GREY), "wall strips must use non-black, non-grey colors"
    assert color in PRIORITY_RANK, "unexpected wall color"
    return color


def _room_side_colors(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    h, w = grid.shape
    assert r0 >= 2 and c0 >= 2 and r1 + 2 < h and c1 + 2 < w, "room too close to edge"

    top = _uniform_color(grid[r0 - 1, c0 : c1 + 1])
    bottom = _uniform_color(grid[r1 + 1, c0 : c1 + 1])
    left = _uniform_color(grid[r0 : r1 + 1, c0 - 1])
    right = _uniform_color(grid[r0 : r1 + 1, c1 + 1])

    assert np.all(grid[r0 - 2, c0 - 2 : c1 + 3] == GREY), "missing top grey boundary"
    assert np.all(grid[r1 + 2, c0 - 2 : c1 + 3] == GREY), "missing bottom grey boundary"
    assert np.all(grid[r0 - 2 : r1 + 3, c0 - 2] == GREY), "missing left grey boundary"
    assert np.all(grid[r0 - 2 : r1 + 3, c1 + 2] == GREY), "missing right grey boundary"

    return top, bottom, left, right


def _dominant_color(colors: set[int]) -> int:
    return min(colors, key=lambda color: PRIORITY_RANK[color])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each enclosed room interior with the highest-priority wall color.

    Input structure:
    - Black background.
    - Grey lines partition the scene into closed rectangular rooms.
    - Each room has one-pixel colored wall bands just inside the grey boundary.
    - Room interiors are solid black rectangles.

    Rule:
    - There is a fixed global priority chain over the wall colors.
    - A room's dominant color is the highest-priority color present on any of its
      four walls.
    - Fill the room's black interior with that dominant color, keeping grey
      boundaries and colored wall bands unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    h, w = g.shape
    for pts in _components(g == BLACK):
        if _touches_edge(pts, h, w):
            continue
        r0, c0, r1, c1 = _bbox(pts)
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        assert len(pts) == area, "room interior must be a solid rectangle"
        top, bottom, left, right = _room_side_colors(g, (r0, c0, r1, c1))
        dominant = _dominant_color({top, bottom, left, right})
        out[r0 : r1 + 1, c0 : c1 + 1] = dominant

    return out


# ---------------------------- generator ----------------------------

ACTIVE_COUNT_RANGE = (3, 5)
GRID_ROWS_RANGE = (2, 4)
GRID_COLS_RANGE = (2, 4)
ROOM_GAP_RANGE = (4, 8)  # cells between grey lines; interior size is gap - 2
PADDING_RANGE = (1, 3)
MAX_GRID_SIZE = 30
GEN_RETRIES = 400


def _sorted_active_subset(rng: random.Random) -> list[int]:
    count = rng.randint(*ACTIVE_COUNT_RANGE)
    chosen = set(rng.sample(PRIORITY_ORDER, count))
    return [color for color in PRIORITY_ORDER if color in chosen]


def _distribute_sides(rng: random.Random, colors: list[int]) -> tuple[int, int, int, int]:
    assert 2 <= len(colors) <= 4
    sides = list(colors)
    while len(sides) < 4:
        sides.append(rng.choice(colors))
    rng.shuffle(sides)
    return tuple(sides)


def _random_room_spec(rng: random.Random, active_colors: list[int]) -> tuple[int, int, int, int]:
    max_distinct = min(4, len(active_colors))
    distinct = rng.randint(2, max_distinct)
    colors = rng.sample(active_colors, distinct)
    return _distribute_sides(rng, colors)


def _required_room_specs(rng: random.Random, active_colors: list[int]) -> list[tuple[int, int, int, int]]:
    specs: list[tuple[int, int, int, int]] = []

    if len(active_colors) >= 4:
        specs.append(_distribute_sides(rng, rng.sample(active_colors, 4)))
    if len(active_colors) >= 3:
        specs.append(_distribute_sides(rng, rng.sample(active_colors, 3)))

    for i in range(len(active_colors) - 1):
        pair = [active_colors[i], active_colors[i + 1]]
        specs.append(_distribute_sides(rng, pair))

    return specs


def _choose_room_grid(rng: random.Random, min_rooms: int) -> tuple[int, int]:
    choices = []
    for rows in range(GRID_ROWS_RANGE[0], GRID_ROWS_RANGE[1] + 1):
        for cols in range(GRID_COLS_RANGE[0], GRID_COLS_RANGE[1] + 1):
            if rows * cols >= min_rooms:
                choices.append((rows, cols))
    assert choices
    return rng.choice(choices)


def _sample_layout(rng: random.Random, rows: int, cols: int):
    pad_t = rng.randint(*PADDING_RANGE)
    pad_b = rng.randint(*PADDING_RANGE)
    pad_l = rng.randint(*PADDING_RANGE)
    pad_r = rng.randint(*PADDING_RANGE)

    row_gaps = [rng.randint(*ROOM_GAP_RANGE) for _ in range(rows)]
    col_gaps = [rng.randint(*ROOM_GAP_RANGE) for _ in range(cols)]

    r_lines = [pad_t]
    for gap in row_gaps:
        r_lines.append(r_lines[-1] + gap + 1)
    c_lines = [pad_l]
    for gap in col_gaps:
        c_lines.append(c_lines[-1] + gap + 1)

    height = r_lines[-1] + pad_b + 1
    width = c_lines[-1] + pad_r + 1
    if height > MAX_GRID_SIZE or width > MAX_GRID_SIZE:
        return None
    return r_lines, c_lines, height, width


def _paint_room(
    grid: np.ndarray,
    r_top: int,
    r_bottom: int,
    c_left: int,
    c_right: int,
    sides: tuple[int, int, int, int],
) -> None:
    top_color, bottom_color, left_color, right_color = sides

    top_row = r_top + 1
    bottom_row = r_bottom - 1
    left_col = c_left + 1
    right_col = c_right - 1
    inner_r0 = r_top + 2
    inner_r1 = r_bottom - 2
    inner_c0 = c_left + 2
    inner_c1 = c_right - 2

    grid[top_row, inner_c0 : inner_c1 + 1] = top_color
    grid[bottom_row, inner_c0 : inner_c1 + 1] = bottom_color
    grid[inner_r0 : inner_r1 + 1, left_col] = left_color
    grid[inner_r0 : inner_r1 + 1, right_col] = right_color

    grid[top_row, left_col] = left_color
    grid[bottom_row, left_col] = left_color
    grid[top_row, right_col] = right_color
    grid[bottom_row, right_col] = right_color


def _generate_candidate(rng: random.Random) -> np.ndarray | None:
    active_colors = _sorted_active_subset(rng)
    required_specs = _required_room_specs(rng, active_colors)
    extra_rooms = rng.randint(0, 4)
    rows, cols = _choose_room_grid(rng, len(required_specs) + extra_rooms)

    layout = _sample_layout(rng, rows, cols)
    if layout is None:
        return None
    r_lines, c_lines, height, width = layout

    room_total = rows * cols
    specs = list(required_specs)
    while len(specs) < room_total:
        specs.append(_random_room_spec(rng, active_colors))
    rng.shuffle(specs)

    grid = np.zeros((height, width), dtype=int)
    top, bottom = r_lines[0], r_lines[-1]
    left, right = c_lines[0], c_lines[-1]

    for r in r_lines:
        grid[r, left : right + 1] = GREY
    for c in c_lines:
        grid[top : bottom + 1, c] = GREY

    idx = 0
    for rr in range(rows):
        for cc in range(cols):
            _paint_room(
                grid,
                r_lines[rr],
                r_lines[rr + 1],
                c_lines[cc],
                c_lines[cc + 1],
                specs[idx],
            )
            idx += 1

    return grid


def _valid_candidate(grid: np.ndarray) -> bool:
    try:
        out = main(grid)
    except AssertionError:
        return False

    if np.array_equal(grid, out):
        return False

    enclosed_rooms = [
        pts
        for pts in _components(grid == BLACK)
        if not _touches_edge(pts, grid.shape[0], grid.shape[1])
    ]
    if len(enclosed_rooms) < 4:
        return False

    dominant_colors = set()
    distinct_wall_counts = set()
    for pts in enclosed_rooms:
        bbox = _bbox(pts)
        sides = _room_side_colors(grid, bbox)
        distinct_wall_counts.add(len(set(sides)))
        dominant_colors.add(_dominant_color(set(sides)))

    if len(dominant_colors) < 2:
        return False
    if 2 not in distinct_wall_counts:
        return False
    if len(PRIORITY_ORDER) >= 3 and 3 not in distinct_wall_counts and len(dominant_colors) >= 3:
        return False

    return True


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _generate_candidate(rng)
        if grid is not None and _valid_candidate(grid):
            return grid
    raise RuntimeError("generate_input: exhausted attempts")
