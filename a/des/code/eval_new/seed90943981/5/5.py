import random

import numpy as np


BG = 8
GREY = 5
MAROON = 9
PATH_COLORS = (1, 2, 3, 4, 6, 7)
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
DIR_INDEX = {delta: idx for idx, delta in enumerate(DIRS)}

# --------------- solver helpers ---------------


def _find_summary_frame(grid: np.ndarray) -> tuple[int, int, int, int]:
    grey_pts = np.argwhere(grid == GREY)
    assert grey_pts.size > 0, "expected a grey summary frame"

    r0, c0 = map(int, grey_pts.min(axis=0))
    r1, c1 = map(int, grey_pts.max(axis=0))
    assert r1 - r0 >= 2 and c1 - c0 >= 2, "summary frame must enclose an interior"

    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            on_border = r in (r0, r1) or c in (c0, c1)
            if on_border:
                assert int(grid[r, c]) == GREY, "grey frame border must be solid"
            else:
                assert int(grid[r, c]) == BG, "summary frame interior must start blank"

    expected_border = 2 * (r1 - r0 + 1) + 2 * (c1 - c0 + 1) - 4
    assert int(np.count_nonzero(grid == GREY)) == expected_border, "grey pixels must belong only to the summary frame"
    return r0, c0, r1, c1


def _scan_path_colors(grid: np.ndarray) -> list[int]:
    colors: list[int] = []
    seen: set[int] = set()
    for value in grid.ravel():
        color = int(value)
        if color in (BG, GREY) or color in seen:
            continue
        seen.add(color)
        colors.append(color)
    return colors


def _trace_path(grid: np.ndarray, color: int) -> list[tuple[int, int]]:
    pts = [tuple(map(int, pt)) for pt in np.argwhere(grid == color)]
    assert pts, f"missing path color {color}"

    cell_set = set(pts)
    if len(cell_set) == 1:
        return [pts[0]]

    degrees = {
        cell: sum((cell[0] + dr, cell[1] + dc) in cell_set for dr, dc in DIRS)
        for cell in cell_set
    }
    assert all(degree in (1, 2) for degree in degrees.values()), "path cannot branch"

    endpoints = sorted(cell for cell, degree in degrees.items() if degree == 1)
    assert len(endpoints) == 2, "path must have exactly two endpoints"

    path = [endpoints[0]]
    prev = None
    cur = endpoints[0]
    while True:
        nxt = [
            (cur[0] + dr, cur[1] + dc)
            for dr, dc in DIRS
            if (cur[0] + dr, cur[1] + dc) in cell_set and (cur[0] + dr, cur[1] + dc) != prev
        ]
        if not nxt:
            break
        assert len(nxt) == 1, "path traversal must stay unambiguous"
        prev, cur = cur, nxt[0]
        path.append(cur)

    assert len(path) == len(cell_set), "path traversal must visit every cell exactly once"
    return path


def _turn_sequence(path: list[tuple[int, int]]) -> list[str]:
    if len(path) < 3:
        return []

    step_dirs = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:])]
    turns: list[str] = []
    for prev_dir, next_dir in zip(step_dirs, step_dirs[1:]):
        a = DIR_INDEX[prev_dir]
        b = DIR_INDEX[next_dir]
        if a == b:
            continue
        diff = (b - a) % 4
        if diff == 1:
            turns.append("R")
        elif diff == 3:
            turns.append("L")
        else:
            raise AssertionError("path may only continue straight or turn 90 degrees")
    return turns


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace each colored orthogonal path and encode its left/right turns inside the grey summary frame.

    Input structure:
    - The background is azure.
    - Several non-grey colors form one-pixel-wide horizontal/vertical path fragments; all cells of one color belong to one non-branching path.
    - A single grey rectangular frame sits in the lower-right corner, with an azure interior whose height equals the number of path colors and whose width is 6.

    Rule:
    - Ignore the grey frame and order the path colors by the first time each color appears in row-major scan order.
    - For each color, follow its full path from the row-major earlier endpoint to the other endpoint and record only the turns: left turns become the path color, right turns become maroon.
    - Write those symbols from left to right in the matching row inside the grey frame; unused cells stay azure and the original paths plus frame remain unchanged.

    Key insight:
    - The same-color cells are not independent marks: together they define one recoverable polyline, and the output is a compact turn-signature of that polyline rather than a geometric redraw.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    r0, c0, r1, c1 = _find_summary_frame(g)
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    colors = _scan_path_colors(g)

    assert inner_h == len(colors), "summary frame height must match the number of colors"
    assert inner_w == 6, "summary frame interior width must be 6"

    out[r0 + 1 : r1, c0 + 1 : c1] = BG
    for row, color in enumerate(colors):
        turns = _turn_sequence(_trace_path(g, color))
        assert len(turns) <= inner_w, "turn signature must fit inside the summary frame"
        for col, turn in enumerate(turns):
            out[r0 + 1 + row, c0 + 1 + col] = color if turn == "L" else MAROON

    return out


# --------------- generator ---------------

MAX_RETRIES = 500
PLACE_TRIES = 400
SHAPE_TRIES = 1000
COLOR_COUNT_RANGE = (2, 5)
MIN_GRID_SIDE = 15
FRAME_INNER_W = 6


def _normalize_path(path: list[tuple[int, int]]) -> list[tuple[int, int]]:
    r0 = min(r for r, _ in path)
    c0 = min(c for _, c in path)
    return [(r - r0, c - c0) for r, c in path]


def _dilate(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    for r, c in np.argwhere(mask):
        for rr in range(max(0, r - 1), min(h, r + 2)):
            for cc in range(max(0, c - 1), min(w, c + 2)):
                out[rr, cc] = True
    return out


def _random_path_shape(rng: random.Random, max_turns: int, max_box: int) -> list[tuple[int, int]] | None:
    for _ in range(SHAPE_TRIES):
        turn_count = rng.randint(0, max_turns)
        seg_len_hi = 4 if turn_count >= 3 else 5
        lr_steps = [rng.choice((-1, 1)) for _ in range(turn_count)]

        directions = [rng.randrange(4)]
        for step in lr_steps:
            directions.append((directions[-1] + step) % 4)
        lengths = [rng.randint(2, seg_len_hi) for _ in range(turn_count + 1)]

        path = [(0, 0)]
        r = c = 0
        for direction, length in zip(directions, lengths):
            dr, dc = DIRS[direction]
            for _ in range(length):
                r += dr
                c += dc
                path.append((r, c))

        if len(set(path)) != len(path):
            continue

        path = _normalize_path(path)
        h = 1 + max(r for r, _ in path)
        w = 1 + max(c for _, c in path)
        if h > max_box or w > max_box:
            continue

        cell_set = set(path)
        degrees = {
            cell: sum((cell[0] + dr, cell[1] + dc) in cell_set for dr, dc in DIRS)
            for cell in cell_set
        }
        if any(degree not in (1, 2) for degree in degrees.values()):
            continue
        if sum(degree == 1 for degree in degrees.values()) != 2:
            continue
        if len(_turn_sequence(path)) > FRAME_INNER_W:
            continue
        return path

    return None


def _place_path(
    grid: np.ndarray,
    reserved: np.ndarray,
    path: list[tuple[int, int]],
    color: int,
    rng: random.Random,
) -> bool:
    ph = 1 + max(r for r, _ in path)
    pw = 1 + max(c for _, c in path)
    blocked = _dilate(reserved)

    positions = [(r, c) for r in range(grid.shape[0] - ph + 1) for c in range(grid.shape[1] - pw + 1)]
    rng.shuffle(positions)
    for top, left in positions[:PLACE_TRIES]:
        coords = [(top + r, left + c) for r, c in path]
        if any(blocked[r, c] for r, c in coords):
            continue
        for r, c in coords:
            grid[r, c] = color
            reserved[r, c] = True
        return True
    return False


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    color_count = rng.randint(*COLOR_COUNT_RANGE)
    frame_inner_h = color_count
    frame_h = frame_inner_h + 2
    frame_w = FRAME_INNER_W + 2

    height = rng.randint(max(MIN_GRID_SIDE, frame_h + 6), 30)
    width = rng.randint(max(MIN_GRID_SIDE + 1, frame_w + 8), 30)

    grid = np.full((height, width), BG, dtype=int)
    reserved = np.zeros((height, width), dtype=bool)

    fr0 = height - frame_h
    fc0 = width - frame_w
    grid[fr0, fc0:width] = GREY
    grid[height - 1, fc0:width] = GREY
    grid[fr0:height, fc0] = GREY
    grid[fr0:height, width - 1] = GREY
    reserved[fr0:height, fc0:width] = True

    colors = list(PATH_COLORS)
    rng.shuffle(colors)
    colors = colors[:color_count]

    for color in colors:
        max_turns = 5 if color_count <= 3 else 4
        max_box = 13 if color_count <= 3 else 11
        path = _random_path_shape(rng, max_turns=max_turns, max_box=max_box)
        if path is None or not _place_path(grid, reserved, path, color, rng):
            return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
