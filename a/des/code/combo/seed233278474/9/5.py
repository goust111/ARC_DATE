import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
SEED_COLORS = (3, 4, 5, 6, 7, 8, 9)

MISSING_TO_DIRECTION = {
    (0, 0): "up",
    (0, 1): "right",
    (1, 0): "left",
    (1, 1): "down",
}

DIRECTION_TO_ARROW = {
    "up": ((0, 1), (1, 0), (1, 1)),
    "right": ((0, 0), (1, 0), (1, 1)),
    "left": ((0, 0), (0, 1), (1, 1)),
    "down": ((0, 0), (0, 1), (1, 0)),
}

ROOM_LAYOUTS = ((1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2))
MIN_ROOM_SIZE = 4
MAX_ROOM_SIZE = 8
GEN_RETRIES = 400


def _room_spans(separator_idx: list[int]) -> list[tuple[int, int]]:
    spans = []
    for a, b in zip(separator_idx, separator_idx[1:]):
        if b - a > 1:
            spans.append((a + 1, b - 1))
    return spans


def _decode_direction(red_pts: np.ndarray) -> str:
    r0 = int(red_pts[:, 0].min())
    c0 = int(red_pts[:, 1].min())
    r1 = int(red_pts[:, 0].max())
    c1 = int(red_pts[:, 1].max())
    assert r1 == r0 + 1 and c1 == c0 + 1, "arrow must occupy a 2x2 bbox"

    mask = np.zeros((2, 2), dtype=bool)
    for r, c in red_pts:
        mask[int(r - r0), int(c - c0)] = True

    missing = np.argwhere(~mask)
    assert missing.shape == (1, 2), "arrow must have exactly one missing corner"
    return MISSING_TO_DIRECTION[tuple(map(int, missing[0]))]


def _fill_room(out: np.ndarray, room: tuple[int, int, int, int], seed_r: int, seed_c: int, color: int, direction: str) -> None:
    top, bottom, left, right = room

    if direction == "right":
        out[top : bottom + 1, seed_c : right + 1] = color
        out[seed_r, seed_c : right + 1] = BLACK
    elif direction == "left":
        out[top : bottom + 1, left : seed_c + 1] = color
        out[seed_r, left : seed_c + 1] = BLACK
    elif direction == "down":
        out[seed_r : bottom + 1, left : right + 1] = color
        out[seed_r : bottom + 1, seed_c] = BLACK
    elif direction == "up":
        out[top : seed_r + 1, left : right + 1] = color
        out[top : seed_r + 1, seed_c] = BLACK
    else:
        raise AssertionError(f"unknown direction: {direction}")


def main(grid: np.ndarray) -> np.ndarray:
    """Grow each room's seed toward its red L-arrow, keeping the seed row/column as a cut line."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    h, w = g.shape
    wall_rows = [r for r in range(h) if np.all(g[r] == BLUE)]
    wall_cols = [c for c in range(w) if np.all(g[:, c] == BLUE)]
    assert len(wall_rows) >= 2 and len(wall_cols) >= 2
    assert wall_rows[0] == 0 and wall_rows[-1] == h - 1
    assert wall_cols[0] == 0 and wall_cols[-1] == w - 1

    row_spans = _room_spans(wall_rows)
    col_spans = _room_spans(wall_cols)
    assert row_spans and col_spans

    out = g.copy()
    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            room = g[r0 : r1 + 1, c0 : c1 + 1]

            seed_pts = np.argwhere((room != BLACK) & (room != BLUE) & (room != RED))
            red_pts = np.argwhere(room == RED)
            assert seed_pts.shape == (1, 2), "each room must contain exactly one seed"
            assert red_pts.shape == (3, 2), "each room must contain exactly one 3-pixel arrow"

            seed_rr, seed_cc = map(int, seed_pts[0])
            seed_r = r0 + seed_rr
            seed_c = c0 + seed_cc
            seed_color = int(room[seed_rr, seed_cc])
            direction = _decode_direction(red_pts)

            _fill_room(out, (r0, r1, c0, c1), seed_r, seed_c, seed_color, direction)

    out[g == RED] = BLACK
    return out


def _fill_mask(height: int, width: int, seed_r: int, seed_c: int, direction: str) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    if direction == "right":
        mask[:, seed_c:] = True
        mask[seed_r, seed_c:] = False
    elif direction == "left":
        mask[:, : seed_c + 1] = True
        mask[seed_r, : seed_c + 1] = False
    elif direction == "down":
        mask[seed_r:, :] = True
        mask[seed_r:, seed_c] = False
    elif direction == "up":
        mask[: seed_r + 1, :] = True
        mask[: seed_r + 1, seed_c] = False
    else:
        raise AssertionError(f"unknown direction: {direction}")
    return mask


def _arrow_candidates(height: int, width: int, seed_r: int, seed_c: int, direction: str) -> list[tuple[int, int]]:
    fill_mask = _fill_mask(height, width, seed_r, seed_c, direction)
    pattern = DIRECTION_TO_ARROW[direction]
    candidates = []
    for top in range(height - 1):
        for left in range(width - 1):
            pts = [(top + dr, left + dc) for dr, dc in pattern]
            if (seed_r, seed_c) in pts:
                continue
            if any(fill_mask[r, c] for r, c in pts):
                continue
            if min(abs(r - seed_r) + abs(c - seed_c) for r, c in pts) < 2:
                continue
            candidates.append((top, left))
    return candidates


def _place_room_objects(grid: np.ndarray, room: tuple[int, int, int, int], color: int, direction: str) -> bool:
    top, bottom, left, right = room
    height = bottom - top + 1
    width = right - left + 1

    seed_choices = [(r, c) for r in range(1, height - 1) for c in range(1, width - 1)]
    random.shuffle(seed_choices)

    for seed_r, seed_c in seed_choices:
        arrow_boxes = _arrow_candidates(height, width, seed_r, seed_c, direction)
        if not arrow_boxes:
            continue

        grid[top + seed_r, left + seed_c] = color
        arrow_top, arrow_left = random.choice(arrow_boxes)
        for dr, dc in DIRECTION_TO_ARROW[direction]:
            grid[top + arrow_top + dr, left + arrow_left + dc] = RED
        return True

    return False


def _build_grid(row_heights: list[int], col_widths: list[int]) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    total_h = sum(row_heights) + len(row_heights) + 1
    total_w = sum(col_widths) + len(col_widths) + 1
    grid = np.full((total_h, total_w), BLACK, dtype=int)

    wall_rows = [0]
    for height in row_heights:
        wall_rows.append(wall_rows[-1] + height + 1)
    wall_cols = [0]
    for width in col_widths:
        wall_cols.append(wall_cols[-1] + width + 1)

    grid[wall_rows, :] = BLUE
    grid[:, wall_cols] = BLUE

    rooms = []
    for i, height in enumerate(row_heights):
        for j, width in enumerate(col_widths):
            top = wall_rows[i] + 1
            bottom = top + height - 1
            left = wall_cols[j] + 1
            right = left + width - 1
            rooms.append((top, bottom, left, right))

    return grid, rooms


def _gen_attempt() -> np.ndarray | None:
    room_rows, room_cols = random.choice(ROOM_LAYOUTS)
    room_count = room_rows * room_cols

    row_heights = [random.randint(MIN_ROOM_SIZE, MAX_ROOM_SIZE) for _ in range(room_rows)]
    col_widths = [random.randint(MIN_ROOM_SIZE, MAX_ROOM_SIZE) for _ in range(room_cols)]
    grid, rooms = _build_grid(row_heights, col_widths)

    colors = random.sample(SEED_COLORS, room_count)
    directions = [random.choice(("up", "right", "down", "left")) for _ in range(room_count)]
    if room_count > 1 and len(set(directions)) == 1:
        directions[random.randrange(room_count)] = random.choice([d for d in ("up", "right", "down", "left") if d != directions[0]])

    for room, color, direction in zip(rooms, colors, directions):
        if not _place_room_objects(grid, room, color, direction):
            return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(grid, out):
        return None

    if np.count_nonzero(out == RED) != 0:
        return None

    for color in colors:
        if np.count_nonzero(out == color) < 2:
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
