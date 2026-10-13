import random

import numpy as np


BG = 0
DIRS = ("up", "down", "left", "right")
STEP_BY_DIR = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def _side_cells(r0: int, c0: int) -> dict[str, list[tuple[int, int]]]:
    return {
        "up": [(r0, c0 + dc) for dc in range(5)],
        "down": [(r0 + 4, c0 + dc) for dc in range(5)],
        "left": [(r0 + dr, c0) for dr in range(1, 4)],
        "right": [(r0 + dr, c0 + 4) for dr in range(1, 4)],
    }


def _gap_cell(r0: int, c0: int, direction: str) -> tuple[int, int]:
    return {
        "up": (r0, c0 + 2),
        "down": (r0 + 4, c0 + 2),
        "left": (r0 + 2, c0),
        "right": (r0 + 2, c0 + 4),
    }[direction]


def _extract_window(grid: np.ndarray, r0: int, c0: int) -> dict | None:
    h, w = grid.shape
    if r0 < 0 or c0 < 0 or r0 + 4 >= h or c0 + 4 >= w:
        return None

    patch = grid[r0 : r0 + 5, c0 : c0 + 5]
    if np.any(patch[1:4, 1:4] != BG):
        return None

    side_map = _side_cells(r0, c0)
    colors: dict[str, int] = {}
    gap_dir = None

    for direction in DIRS:
        cells = side_map[direction]
        gap = _gap_cell(r0, c0, direction)
        values = [int(grid[r, c]) for r, c in cells]
        non_bg = [v for (r, c), v in zip(cells, values) if (r, c) != gap]
        if not non_bg or any(v == BG for v in non_bg):
            return None
        color = non_bg[0]
        if any(v != color for v in non_bg):
            return None

        gap_value = int(grid[gap])
        if gap_value == BG:
            if gap_dir is not None:
                return None
            gap_dir = direction
        else:
            if gap_value != color:
                return None
        colors[direction] = color

    if gap_dir is None:
        return None
    if len(set(colors.values())) != 4:
        return None

    perimeter = set()
    for cells in side_map.values():
        perimeter.update(cells)
    fg_count = sum(int(grid[r, c] != BG) for r, c in perimeter)
    if fg_count != 15:
        return None

    return {"top": r0, "left": c0, "gap_dir": gap_dir, "colors": colors}


def _find_windows(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    windows: list[dict] = []
    used = np.zeros((h, w), dtype=bool)

    for r0 in range(h - 4):
        for c0 in range(w - 4):
            if used[r0 : r0 + 5, c0 : c0 + 5].any():
                continue
            window = _extract_window(g, r0, c0)
            if window is None:
                continue
            used[r0 : r0 + 5, c0 : c0 + 5] = True
            windows.append(window)

    assert windows, "expected at least one window"
    return windows


def main(grid: np.ndarray) -> np.ndarray:
    """
    Identify each 5x5 window frame, find the side with the missing center pixel,
    and emit a ray in that side's color from the gap outward until the grid edge
    or the next window frame pixel.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    out = g.copy()
    h, w = g.shape

    for window in _find_windows(g):
        r0 = int(window["top"])
        c0 = int(window["left"])
        direction = str(window["gap_dir"])
        color = int(window["colors"][direction])
        gap_r, gap_c = _gap_cell(r0, c0, direction)
        dr, dc = STEP_BY_DIR[direction]

        r = gap_r + dr
        c = gap_c + dc
        while 0 <= r < h and 0 <= c < w:
            if int(g[r, c]) != BG:
                break
            if out[r, c] == BG:
                out[r, c] = color
            r += dr
            c += dc

    return out


GRID_MIN = 14
GRID_MAX = 24
WINDOW_MIN = 1
WINDOW_MAX = 5
MAX_ATTEMPTS = 400
PLACEMENT_TRIES = 120
PALETTE = tuple(range(1, 10))


def _expanded_box(r0: int, c0: int, margin: int = 1) -> tuple[int, int, int, int]:
    return r0 - margin, c0 - margin, r0 + 4 + margin, c0 + 4 + margin


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return a[0] <= b[2] and a[2] >= b[0] and a[1] <= b[3] and a[3] >= b[1]


def _paint_window(grid: np.ndarray, r0: int, c0: int, colors: dict[str, int], gap_dir: str) -> None:
    for direction, cells in _side_cells(r0, c0).items():
        color = int(colors[direction])
        for r, c in cells:
            grid[r, c] = color
    gap_r, gap_c = _gap_cell(r0, c0, gap_dir)
    grid[gap_r, gap_c] = BG


def _balanced_dirs(count: int) -> list[str]:
    base = list(DIRS)
    random.shuffle(base)
    dirs = [base[i % 4] for i in range(count)]
    random.shuffle(dirs)
    return dirs


def _random_colors() -> dict[str, int]:
    picked = random.sample(PALETTE, 4)
    return {direction: picked[i] for i, direction in enumerate(DIRS)}


def _can_place(box: tuple[int, int, int, int], placed: list[tuple[int, int, int, int]]) -> bool:
    return all(not _boxes_overlap(box, other) for other in placed)


def _place_blocker_pair(h: int, w: int) -> list[tuple[int, int, str]]:
    direction = random.choice(DIRS)
    tries = 100
    for _ in range(tries):
        if direction == "down":
            c0 = random.randint(0, w - 5)
            top1 = random.randint(0, h - 11)
            top2 = random.randint(top1 + 6, h - 5)
            if top2 - (top1 + 4) < 2:
                continue
            return [(top1, c0, "down"), (top2, c0, random.choice(DIRS))]
        if direction == "up":
            c0 = random.randint(0, w - 5)
            top2 = random.randint(0, h - 11)
            top1 = random.randint(top2 + 6, h - 5)
            if top1 - (top2 + 4) < 2:
                continue
            return [(top1, c0, "up"), (top2, c0, random.choice(DIRS))]
        if direction == "right":
            r0 = random.randint(0, h - 5)
            left1 = random.randint(0, w - 11)
            left2 = random.randint(left1 + 6, w - 5)
            if left2 - (left1 + 4) < 2:
                continue
            return [(r0, left1, "right"), (r0, left2, random.choice(DIRS))]
        r0 = random.randint(0, h - 5)
        left2 = random.randint(0, w - 11)
        left1 = random.randint(left2 + 6, w - 5)
        if left1 - (left2 + 4) < 2:
            continue
        return [(r0, left1, "left"), (r0, left2, random.choice(DIRS))]
    raise RuntimeError("failed to place blocker pair")


def _generate_layout() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    grid = np.full((h, w), BG, dtype=int)

    blocker_case = random.random() < 0.5 and h >= 11 and w >= 11
    min_windows = 2 if blocker_case else WINDOW_MIN
    max_windows = min(WINDOW_MAX, max(2, (h * w) // 45))
    if max_windows < min_windows:
        return None
    count = random.randint(min_windows, max_windows)

    placements: list[tuple[int, int, str]] = []
    reserved: list[tuple[int, int, int, int]] = []

    if blocker_case:
        try:
            pair = _place_blocker_pair(h, w)
        except RuntimeError:
            return None
        for r0, c0, direction in pair:
            box = _expanded_box(r0, c0, margin=1)
            if not _can_place(box, reserved):
                return None
            reserved.append(box)
            placements.append((r0, c0, direction))

    remaining = count - len(placements)
    extra_dirs = _balanced_dirs(count)
    used_seed_dirs = [direction for _r0, _c0, direction in placements]
    extras = [d for d in extra_dirs if d not in used_seed_dirs]
    while len(extras) < remaining:
        extras.append(random.choice(DIRS))
    extras = extras[:remaining]

    for direction in extras:
        placed = False
        for _ in range(PLACEMENT_TRIES):
            r0 = random.randint(0, h - 5)
            c0 = random.randint(0, w - 5)
            box = _expanded_box(r0, c0, margin=1)
            if not _can_place(box, reserved):
                continue
            reserved.append(box)
            placements.append((r0, c0, direction))
            placed = True
            break
        if not placed:
            return None

    random.shuffle(placements)
    for r0, c0, direction in placements:
        _paint_window(grid, r0, c0, _random_colors(), direction)

    windows = _find_windows(grid)
    if len(windows) != len(placements):
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _generate_layout()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
