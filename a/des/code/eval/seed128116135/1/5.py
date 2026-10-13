import random

import numpy as np


BG = 0
GREY = 5
MAROON = 9
PAYLOAD_COLORS = (1, 2, 3, 4, 6, 7, 8)
DIRS = ("up", "down", "left", "right")

GEN_RETRIES = 300
GRID_H_RANGE = (10, 18)
GRID_W_RANGE = (10, 18)
FRAME_H_RANGE = (6, 11)
FRAME_W_RANGE = (6, 11)
PAYLOAD_COUNT_RANGE = (2, 6)

ARROW_SHAPES = {
    "up": {(1, 0), (1, 1), (1, 2), (0, 1)},
    "down": {(0, 0), (0, 1), (0, 2), (1, 1)},
    "left": {(0, 1), (1, 1), (2, 1), (1, 0)},
    "right": {(0, 0), (1, 0), (2, 0), (1, 1)},
}
ARROW_SIZES = {
    "up": (2, 3),
    "down": (2, 3),
    "left": (3, 2),
    "right": (3, 2),
}


def _frame_and_gap(grid: np.ndarray) -> tuple[int, int, int, int, tuple[int, int], str]:
    grey_pts = np.argwhere(grid == GREY)
    assert len(grey_pts) >= 8, "expected a grey rectangular frame"

    top = int(grey_pts[:, 0].min())
    bottom = int(grey_pts[:, 0].max())
    left = int(grey_pts[:, 1].min())
    right = int(grey_pts[:, 1].max())
    assert bottom - top >= 2 and right - left >= 2, "frame must enclose an interior"

    missing: list[tuple[int, int]] = []
    grey_count = 0
    for r in range(top, bottom + 1):
        for c in range(left, right + 1):
            on_border = r in (top, bottom) or c in (left, right)
            if not on_border:
                continue
            if int(grid[r, c]) == GREY:
                grey_count += 1
            else:
                missing.append((r, c))

    expected_border = 2 * ((bottom - top + 1) + (right - left + 1)) - 4
    assert grey_count == expected_border - 1, "frame must have exactly one missing border pixel"
    assert len(missing) == 1, "expected exactly one gap in the frame"
    gap_r, gap_c = missing[0]
    assert int(grid[gap_r, gap_c]) == BG, "gap must be black"

    if gap_r == top:
        direction = "up"
    elif gap_r == bottom:
        direction = "down"
    elif gap_c == left:
        direction = "left"
    else:
        direction = "right"

    assert (gap_r not in (top, bottom)) or (gap_c not in (left, right)), "gap cannot be a corner"
    return top, bottom, left, right, (gap_r, gap_c), direction


def _arrow_direction(grid: np.ndarray) -> str | None:
    pts = np.argwhere(grid == MAROON)
    if len(pts) != 4:
        return None
    rel = {(int(r - pts[:, 0].min()), int(c - pts[:, 1].min())) for r, c in pts}
    for direction, shape in ARROW_SHAPES.items():
        if rel == shape:
            return direction
    return None


def _payload_pixels(
    grid: np.ndarray, top: int, bottom: int, left: int, right: int
) -> list[tuple[int, int, int]]:
    payload: list[tuple[int, int, int]] = []
    for r in range(top + 1, bottom):
        for c in range(left + 1, right):
            color = int(grid[r, c])
            if color not in (BG, GREY, MAROON):
                payload.append((r, c, color))
    return payload


def _sort_payload(payload: list[tuple[int, int, int]], direction: str) -> list[tuple[int, int, int]]:
    if direction == "up":
        key = lambda item: (item[0], item[1])
    elif direction == "down":
        key = lambda item: (-item[0], item[1])
    elif direction == "left":
        key = lambda item: (item[1], item[0])
    else:
        key = lambda item: (-item[1], item[0])
    return sorted(payload, key=key)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort interior colored pixels by depth toward the frame gap, then eject them outside.

    The grey frame has one 1-cell gap. A maroon arrow inside points toward the gap side.
    Non-black, non-grey, non-maroon 1x1 pixels inside the frame are sorted by distance to
    that side, then placed outside the gap along the perpendicular axis.
    """
    g = np.asarray(grid, dtype=int)
    top, bottom, left, right, gap, gap_dir = _frame_and_gap(g)

    arrow_dir = _arrow_direction(g)
    direction = gap_dir
    if arrow_dir is not None:
        assert arrow_dir == gap_dir, "arrow direction must match the gap side"
        direction = arrow_dir

    payload = _sort_payload(_payload_pixels(g, top, bottom, left, right), direction)

    out = g.copy()
    out[top + 1 : bottom, left + 1 : right] = BG

    gap_r, gap_c = gap
    if direction == "up":
        start_r, start_c = top - 1, gap_c
        cells = [(start_r, start_c + i) for i in range(len(payload))]
    elif direction == "down":
        start_r, start_c = bottom + 1, gap_c
        cells = [(start_r, start_c + i) for i in range(len(payload))]
    elif direction == "left":
        start_r, start_c = gap_r, left - 1
        cells = [(start_r + i, start_c) for i in range(len(payload))]
    else:
        start_r, start_c = gap_r, right + 1
        cells = [(start_r + i, start_c) for i in range(len(payload))]

    for (r, c), (_, _, color) in zip(cells, payload):
        assert 0 <= r < g.shape[0] and 0 <= c < g.shape[1], "output placement must fit in-grid"
        out[r, c] = color

    return out


def _place_arrow(
    grid: np.ndarray,
    occupied: set[tuple[int, int]],
    inner_top: int,
    inner_bottom: int,
    inner_left: int,
    inner_right: int,
    direction: str,
    rng: random.Random,
) -> bool:
    arrow_h, arrow_w = ARROW_SIZES[direction]
    row_lo = inner_top
    row_hi = inner_bottom - arrow_h + 1
    col_lo = inner_left
    col_hi = inner_right - arrow_w + 1
    if row_lo > row_hi or col_lo > col_hi:
        return False

    anchors = [(r, c) for r in range(row_lo, row_hi + 1) for c in range(col_lo, col_hi + 1)]
    rng.shuffle(anchors)
    for ar, ac in anchors:
        pts = {(ar + dr, ac + dc) for dr, dc in ARROW_SHAPES[direction]}
        if pts & occupied:
            continue
        for r, c in pts:
            grid[r, c] = MAROON
        occupied.update(pts)
        return True
    return False


def _choose_layout(
    height: int, width: int, count: int, direction: str, rng: random.Random
) -> tuple[int, int, int, int, tuple[int, int]] | None:
    frame_h_max = min(FRAME_H_RANGE[1], height - 2)
    frame_w_max = min(FRAME_W_RANGE[1], width - 2)
    frame_h_min = FRAME_H_RANGE[0]
    frame_w_min = FRAME_W_RANGE[0]
    if frame_h_max < frame_h_min or frame_w_max < frame_w_min:
        return None

    frame_heights = list(range(frame_h_min, frame_h_max + 1))
    frame_widths = list(range(frame_w_min, frame_w_max + 1))
    rng.shuffle(frame_heights)
    rng.shuffle(frame_widths)

    for frame_h in frame_heights:
        for frame_w in frame_widths:
            inner_h = frame_h - 2
            inner_w = frame_w - 2
            if inner_h * inner_w < count + 4:
                continue

            if direction in ("up", "down"):
                top_min = 1 if direction == "up" else 0
                top_max = height - frame_h - (0 if direction == "up" else 1)
                left_min = 0
                left_max = width - frame_w
                if top_min > top_max or left_min > left_max:
                    continue
                top = rng.randint(top_min, top_max)
                left = rng.randint(left_min, left_max)
                gap_c_min = left + 1
                gap_c_max = min(left + frame_w - 2, width - count)
                if gap_c_min > gap_c_max:
                    continue
                gap_c = rng.randint(gap_c_min, gap_c_max)
                gap_r = top if direction == "up" else top + frame_h - 1
            else:
                top_min = 0
                top_max = height - frame_h
                left_min = 1 if direction == "left" else 0
                left_max = width - frame_w - (0 if direction == "left" else 1)
                if top_min > top_max or left_min > left_max:
                    continue
                top = rng.randint(top_min, top_max)
                left = rng.randint(left_min, left_max)
                gap_r_min = top + 1
                gap_r_max = min(top + frame_h - 2, height - count)
                if gap_r_min > gap_r_max:
                    continue
                gap_r = rng.randint(gap_r_min, gap_r_max)
                gap_c = left if direction == "left" else left + frame_w - 1

            bottom = top + frame_h - 1
            right = left + frame_w - 1
            return top, bottom, left, right, (gap_r, gap_c)
    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(*GRID_H_RANGE)
    width = rng.randint(*GRID_W_RANGE)
    direction = rng.choice(DIRS)
    count = rng.randint(*PAYLOAD_COUNT_RANGE)

    layout = _choose_layout(height, width, count, direction, rng)
    if layout is None:
        return None
    top, bottom, left, right, gap = layout
    gap_r, gap_c = gap

    grid = np.full((height, width), BG, dtype=int)
    grid[top, left : right + 1] = GREY
    grid[bottom, left : right + 1] = GREY
    grid[top : bottom + 1, left] = GREY
    grid[top : bottom + 1, right] = GREY
    grid[gap_r, gap_c] = BG

    occupied = set()
    inner_top, inner_bottom = top + 1, bottom - 1
    inner_left, inner_right = left + 1, right - 1

    if not _place_arrow(
        grid, occupied, inner_top, inner_bottom, inner_left, inner_right, direction, rng
    ):
        return None

    free_cells = [
        (r, c)
        for r in range(inner_top, inner_bottom + 1)
        for c in range(inner_left, inner_right + 1)
        if (r, c) not in occupied
    ]
    if len(free_cells) < count:
        return None

    rng.shuffle(free_cells)
    colors = list(PAYLOAD_COLORS)
    rng.shuffle(colors)
    for (r, c), color in zip(free_cells[:count], colors[:count]):
        grid[r, c] = color
        occupied.add((r, c))

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted retries")
