from __future__ import annotations

import random

import numpy as np


BG = 0
WALL = 5
OBSTACLE = 9
FADE_CHAIN = (4, 7, 2, 9)  # yellow -> orange -> red -> maroon
SIDES = ("top", "bottom", "left", "right")
WINDOW_COLORS = (1, 2, 3, 4, 6, 7, 8)
FADEABLE = (4, 7, 2)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _room_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    pts = np.argwhere(grid == WALL)
    assert pts.size > 0, "room border must exist"
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    r0, c0, r1, c1 = int(r0), int(c0), int(r1), int(c1)
    assert r1 - r0 >= 4 and c1 - c0 >= 4, "room must have an interior"

    border_mask = np.zeros_like(grid, dtype=bool)
    border_mask[r0, c0 : c1 + 1] = True
    border_mask[r1, c0 : c1 + 1] = True
    border_mask[r0 : r1 + 1, c0] = True
    border_mask[r0 : r1 + 1, c1] = True

    assert np.all(grid[r0, c0 : c1 + 1] == WALL)
    assert np.all(grid[r1, c0 : c1 + 1] == WALL)
    assert np.all(grid[r0 : r1 + 1, c0] == WALL)
    assert np.all(grid[r0 : r1 + 1, c1] == WALL)
    assert not np.any((grid == WALL) & ~border_mask), "grey pixels should only form the room border"
    return r0, c0, r1, c1


def _window_side(
    room: tuple[int, int, int, int], bbox: tuple[int, int, int, int]
) -> str:
    rr0, rc0, rr1, rc1 = room
    r0, c0, r1, c1 = bbox
    inner_top = rr0 + 1
    inner_bottom = rr1 - 1
    inner_left = rc0 + 1
    inner_right = rc1 - 1
    flags = {
        "top": r0 == inner_top and c0 > inner_left and c1 < inner_right,
        "bottom": r1 == inner_bottom and c0 > inner_left and c1 < inner_right,
        "left": c0 == inner_left and r0 > inner_top and r1 < inner_bottom,
        "right": c1 == inner_right and r0 > inner_top and r1 < inner_bottom,
    }
    sides = [side for side, ok in flags.items() if ok]
    assert len(sides) == 1, "each window must attach to exactly one wall"
    return sides[0]


def _destination_bbox(
    room: tuple[int, int, int, int], bbox: tuple[int, int, int, int], side: str
) -> tuple[int, int, int, int]:
    rr0, rc0, rr1, rc1 = room
    r0, c0, r1, c1 = bbox
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    inner_top = rr0 + 1
    inner_bottom = rr1 - 1
    inner_left = rc0 + 1
    inner_right = rc1 - 1

    if side == "top":
        return inner_bottom - height + 1, c0, inner_bottom, c1
    if side == "bottom":
        return inner_top, c0, inner_top + height - 1, c1
    if side == "left":
        return r0, inner_right - width + 1, r1, inner_right
    if side == "right":
        return r0, inner_left, r1, inner_left + width - 1
    raise AssertionError(f"unknown side {side}")


def _corridor_ranges(
    bbox: tuple[int, int, int, int],
    dest: tuple[int, int, int, int],
    side: str,
) -> tuple[range, range]:
    r0, c0, r1, c1 = bbox
    dr0, dc0, dr1, dc1 = dest
    if side == "top":
        return range(r1 + 1, dr0), range(c0, c1 + 1)
    if side == "bottom":
        return range(dr1 + 1, r0), range(c0, c1 + 1)
    if side == "left":
        return range(r0, r1 + 1), range(c1 + 1, dc0)
    if side == "right":
        return range(r0, r1 + 1), range(dc1 + 1, c0)
    raise AssertionError(f"unknown side {side}")


def _count_obstacles(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    room: tuple[int, int, int, int],
    side: str,
) -> int:
    dest = _destination_bbox(room, bbox, side)
    rows, cols = _corridor_ranges(bbox, dest, side)
    if len(rows) == 0 or len(cols) == 0:
        return 0
    sub = grid[rows.start : rows.stop, cols.start : cols.stop]
    return int(np.sum(sub == OBSTACLE))


def _fade_value(color: int, count: int) -> int:
    if color not in FADE_CHAIN:
        return int(color)
    idx = FADE_CHAIN.index(int(color))
    return int(FADE_CHAIN[min(idx + int(count), len(FADE_CHAIN) - 1)])


def _fade_pattern(pattern: np.ndarray, count: int) -> np.ndarray:
    out = pattern.copy()
    for color in np.unique(pattern):
        out[pattern == color] = _fade_value(int(color), count)
    return out


def _extract_windows(
    grid: np.ndarray, room: tuple[int, int, int, int]
) -> list[dict[str, object]]:
    rr0, rc0, rr1, rc1 = room
    interior = np.zeros_like(grid, dtype=bool)
    interior[rr0 + 1 : rr1, rc0 + 1 : rc1] = True
    mask = interior & (grid != BG) & (grid != WALL) & (grid != OBSTACLE)

    windows: list[dict[str, object]] = []
    for comp in _components(mask):
        bbox = _bbox(comp)
        r0, c0, r1, c1 = bbox
        rect = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        rect[comp[:, 0] - r0, comp[:, 1] - c0] = True
        assert bool(rect.all()), "window mask must be a solid rectangle"
        side = _window_side(room, bbox)
        windows.append(
            {
                "bbox": bbox,
                "side": side,
                "pattern": grid[r0 : r1 + 1, c0 : c1 + 1].copy(),
            }
        )

    assert windows, "at least one window is required"
    windows.sort(key=lambda item: tuple(int(v) for v in item["bbox"]))  # deterministic
    return windows


def _project_from_windows(
    grid: np.ndarray,
    room: tuple[int, int, int, int],
    windows: list[dict[str, object]],
) -> np.ndarray:
    out = grid.copy()
    for window in windows:
        bbox = tuple(int(v) for v in window["bbox"])
        side = str(window["side"])
        pattern = np.asarray(window["pattern"], dtype=int)
        dest = _destination_bbox(room, bbox, side)
        count = _count_obstacles(grid, bbox, room, side)
        faded = _fade_pattern(pattern, count)
        dr0, dc0, dr1, dc1 = dest
        out[dr0 : dr1 + 1, dc0 : dc1 + 1] = faded
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project each wall-attached colored window straight across the room and fade it by obstacle count.

    Input structure:
    - The background is black.
    - Grey pixels form one large rectangular room border.
    - Inside the room, several small multicolor solid rectangles touch exactly
      one inner wall and act as windows.
    - Maroon single pixels may appear in the empty room interior as obstacles.

    Rule:
    - Identify which wall each window touches; that wall determines the
      projection direction toward the opposite wall.
    - Copy the full window pattern to the opposite inner wall at the matching
      translated position, without changing the original window.
    - Count how many maroon obstacles lie in the rectangular projection
      corridor between the source window and its copy.
    - Fade every copied color by that count along the fixed chain
      yellow -> orange -> red -> maroon; colors outside the chain stay unchanged.

    Key insight:
    - The transformation combines wall-normal projection with an independent
      obstacle-count-controlled recoloring rule applied only to the projected copy.
    """
    g = np.asarray(grid, dtype=int)
    room = _room_bbox(g)
    windows = _extract_windows(g, room)
    return _project_from_windows(g, room, windows)


GEN_ATTEMPTS = 400
ROOM_SIZE_RANGE = (12, 22)
MARGIN_RANGE = (1, 4)
WINDOW_SIDE_COUNT_RANGE = (2, 4)
WINDOW_SHORT_RANGE = (2, 4)
WINDOW_LONG_RANGE = (2, 5)
WINDOW_TRIES = 100
OBSTACLE_TRIES = 120
MAX_OBSTACLES = 7


def _rand_margin_pair(max_total: int) -> tuple[int, int]:
    pairs = [
        (a, b)
        for a in range(MARGIN_RANGE[0], MARGIN_RANGE[1] + 1)
        for b in range(MARGIN_RANGE[0], MARGIN_RANGE[1] + 1)
        if a + b <= max_total
    ]
    return random.choice(pairs)


def _random_pattern(height: int, width: int) -> np.ndarray:
    cell_count = height * width
    color_count = random.randint(2, min(4, cell_count))
    colors = [random.choice(FADEABLE)]
    pool = [color for color in WINDOW_COLORS if color not in colors]
    while len(colors) < color_count:
        color = random.choice(pool)
        if color not in colors:
            colors.append(color)

    pattern = np.array(
        [[random.choice(colors) for _ in range(width)] for _ in range(height)],
        dtype=int,
    )
    positions = [(r, c) for r in range(height) for c in range(width)]
    random.shuffle(positions)
    for color, (r, c) in zip(colors, positions):
        pattern[r, c] = color
    return pattern


def _sample_window_bbox(
    room: tuple[int, int, int, int], side: str
) -> tuple[int, int, int, int] | None:
    rr0, rc0, rr1, rc1 = room
    inner_h = rr1 - rr0 - 1
    inner_w = rc1 - rc0 - 1
    if side in ("top", "bottom"):
        max_h = min(WINDOW_SHORT_RANGE[1], inner_h - 2)
        max_w = min(WINDOW_LONG_RANGE[1], inner_w - 2)
        if max_h < WINDOW_SHORT_RANGE[0] or max_w < WINDOW_LONG_RANGE[0]:
            return None
        height = random.randint(WINDOW_SHORT_RANGE[0], max_h)
        width = random.randint(WINDOW_LONG_RANGE[0], max_w)
        c0 = random.randint(rc0 + 2, rc1 - width - 1)
        r0 = rr0 + 1 if side == "top" else rr1 - height
    else:
        max_h = min(WINDOW_LONG_RANGE[1], inner_h - 2)
        max_w = min(WINDOW_SHORT_RANGE[1], inner_w - 2)
        if max_h < WINDOW_LONG_RANGE[0] or max_w < WINDOW_SHORT_RANGE[0]:
            return None
        height = random.randint(WINDOW_LONG_RANGE[0], max_h)
        width = random.randint(WINDOW_SHORT_RANGE[0], max_w)
        r0 = random.randint(rr0 + 2, rr1 - height - 1)
        c0 = rc0 + 1 if side == "left" else rc1 - width
    return r0, c0, r0 + height - 1, c0 + width - 1


def _sample_windows(room: tuple[int, int, int, int]) -> list[dict[str, object]] | None:
    side_count = random.randint(*WINDOW_SIDE_COUNT_RANGE)
    sides = random.sample(SIDES, side_count)
    random.shuffle(sides)
    windows: list[dict[str, object]] = []
    blocked: list[tuple[int, int, int, int]] = []

    for side in sides:
        placed = False
        for _ in range(WINDOW_TRIES):
            bbox = _sample_window_bbox(room, side)
            if bbox is None:
                return None
            dest = _destination_bbox(room, bbox, side)
            if any(_boxes_overlap(bbox, other) or _boxes_overlap(dest, other) for other in blocked):
                continue
            r0, c0, r1, c1 = bbox
            windows.append(
                {
                    "side": side,
                    "bbox": bbox,
                    "dest": dest,
                    "pattern": _random_pattern(r1 - r0 + 1, c1 - c0 + 1),
                }
            )
            blocked.extend((bbox, dest))
            placed = True
            break
        if not placed:
            return None
    return windows


def _paint_room(grid: np.ndarray, room: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = room
    grid[r0, c0 : c1 + 1] = WALL
    grid[r1, c0 : c1 + 1] = WALL
    grid[r0 : r1 + 1, c0] = WALL
    grid[r0 : r1 + 1, c1] = WALL


def _paint_windows(grid: np.ndarray, windows: list[dict[str, object]]) -> None:
    for window in windows:
        r0, c0, r1, c1 = (int(v) for v in window["bbox"])
        grid[r0 : r1 + 1, c0 : c1 + 1] = np.asarray(window["pattern"], dtype=int)


def _corridor_cells(
    room: tuple[int, int, int, int], window: dict[str, object]
) -> set[tuple[int, int]]:
    bbox = tuple(int(v) for v in window["bbox"])
    side = str(window["side"])
    dest = _destination_bbox(room, bbox, side)
    rows, cols = _corridor_ranges(bbox, dest, side)
    return {(r, c) for r in rows for c in cols}


def _valid_obstacle_cell(grid: np.ndarray, room: tuple[int, int, int, int], r: int, c: int) -> bool:
    rr0, rc0, rr1, rc1 = room
    if not (rr0 + 1 < r < rr1 - 1 and rc0 + 1 < c < rc1 - 1):
        return False
    if int(grid[r, c]) != BG:
        return False
    for nr, nc in _neighbors4(r, c):
        if int(grid[nr, nc]) != BG:
            return False
    for nr in range(r - 1, r + 2):
        for nc in range(c - 1, c + 2):
            if (nr != r or nc != c) and int(grid[nr, nc]) == OBSTACLE:
                return False
    return True


def _place_obstacles(
    grid: np.ndarray,
    room: tuple[int, int, int, int],
    windows: list[dict[str, object]],
) -> bool:
    corridor_sets = [_corridor_cells(room, window) for window in windows]
    union = sorted(set().union(*corridor_sets))
    if not union:
        return False

    for _ in range(OBSTACLE_TRIES):
        trial = grid.copy()
        order = union[:]
        random.shuffle(order)
        target = random.randint(1, min(MAX_OBSTACLES, len(order)))
        placed = 0
        for r, c in order:
            if placed >= target:
                break
            if _valid_obstacle_cell(trial, room, r, c):
                trial[r, c] = OBSTACLE
                placed += 1
        if placed == 0:
            continue

        counts = [
            _count_obstacles(trial, tuple(int(v) for v in window["bbox"]), room, str(window["side"]))
            for window in windows
        ]
        if max(counts) == 0:
            continue
        if len(windows) > 1 and len(set(counts)) < 2:
            continue
        grid[:] = trial
        return True
    return False


def _build_example() -> tuple[np.ndarray, np.ndarray] | None:
    room_h = random.randint(*ROOM_SIZE_RANGE)
    room_w = random.randint(*ROOM_SIZE_RANGE)
    row_margin_total = 30 - room_h
    col_margin_total = 30 - room_w
    if row_margin_total < 2 * MARGIN_RANGE[0] or col_margin_total < 2 * MARGIN_RANGE[0]:
        return None
    top_margin, bottom_margin = _rand_margin_pair(row_margin_total)
    left_margin, right_margin = _rand_margin_pair(col_margin_total)

    rows = room_h + top_margin + bottom_margin
    cols = room_w + left_margin + right_margin
    if rows > 30 or cols > 30:
        return None

    room = (top_margin, left_margin, top_margin + room_h - 1, left_margin + room_w - 1)
    windows = _sample_windows(room)
    if windows is None:
        return None

    grid = np.full((rows, cols), BG, dtype=int)
    _paint_room(grid, room)
    _paint_windows(grid, windows)
    if not _place_obstacles(grid, room, windows):
        return None

    expected = _project_from_windows(grid, room, windows)
    if np.array_equal(expected, grid):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        built = _build_example()
        if built is None:
            continue
        grid, expected = built
        try:
            actual = main(grid)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return grid
    raise RuntimeError("failed to generate a valid input")
