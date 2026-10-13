from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
ORANGE = 7

GEN_ATTEMPTS = 500
HEIGHT_RANGE = (10, 16)
WIDTH_RANGE = (12, 18)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors8(shape: tuple[int, int], r: int, c: int) -> set[tuple[int, int]]:
    h, w = shape
    out: set[tuple[int, int]] = set()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            rr = r + dr
            cc = c + dc
            if 0 <= rr < h and 0 <= cc < w:
                out.add((rr, cc))
    return out


def _find_anchor_row(grid: np.ndarray) -> int:
    h, w = grid.shape
    candidates = []
    for r in range(h):
        row = grid[r, :]
        if np.count_nonzero(row == BLACK) != 0:
            continue
        if np.all(np.isin(row, (BLUE, GREEN))) and np.count_nonzero(row == BLUE) >= w - 1:
            candidates.append(int(r))
    assert len(candidates) == 1, "expected exactly one blue horizontal anchor row"
    return candidates[0]


def _find_anchor_col(grid: np.ndarray) -> int:
    h, w = grid.shape
    candidates = []
    for c in range(w):
        col = grid[:, c]
        if np.count_nonzero(col == BLACK) != 0:
            continue
        if np.all(np.isin(col, (BLUE, GREEN))) and np.count_nonzero(col == GREEN) >= h - 1:
            candidates.append(int(c))
    assert len(candidates) == 1, "expected exactly one green vertical anchor column"
    return candidates[0]


def _draw_short_diagonal(
    out: np.ndarray,
    inp: np.ndarray,
    color: int,
    obstacle: tuple[int, int],
    slide: tuple[int, int],
) -> None:
    h, w = inp.shape
    r, c = obstacle
    dr, dc = slide
    if dr != 0:
        side_offsets = ((0, -1), (0, 1))
    else:
        side_offsets = ((-1, 0), (1, 0))

    for sr, sc in side_offsets:
        wr = r + sr
        wc = c + sc
        er = wr + dr
        ec = wc + dc
        if not _in_bounds((h, w), wr, wc):
            continue
        if int(inp[wr, wc]) != BLACK or int(out[wr, wc]) != BLACK:
            continue
        out[wr, wc] = color
        if _in_bounds((h, w), er, ec) and int(inp[er, ec]) == BLACK and int(out[er, ec]) == BLACK:
            out[er, ec] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide loose blue/green pixels toward their full-line anchors and turn obstacle collisions into side diagonal marks.

    Input structure:
    - Black background contains one blue horizontal full row anchor and one green vertical full column anchor; their crossing cell may show either anchor color.
    - Blue loose pixels lie off the anchor row and slide vertically toward the blue row; green loose pixels lie off the anchor column and slide horizontally toward the green column.
    - Red single-pixel obstacles may lie on some slide paths, and orange loose pixels have no matching anchor.

    Rule:
    - Copy the two anchors and all red obstacles unchanged.
    - For each blue or green loose pixel, move step by step along the unique orthogonal direction toward its same-color anchor.
    - If no red obstacle appears first, place the pixel in the cell adjacent to the anchor; if a red obstacle appears first, stop one cell before it and also draw same-color two-pixel side marks on the obstacle's perpendicular flanks, skipping any flank whose side cell is already occupied.
    - Drop unmatched loose pixels such as orange ones.

    Key insight:
    - Color identifies both whether a pixel is movable and which full-line anchor defines its slide axis; only obstacle hits create the extra diagonal-style side marks.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    blue_row = _find_anchor_row(g)
    green_col = _find_anchor_col(g)
    h, w = g.shape

    out = np.zeros_like(g)
    out[blue_row, :] = g[blue_row, :]
    out[:, green_col] = g[:, green_col]
    out[g == RED] = RED

    loose_pixels: list[tuple[int, int, int]] = []
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == BLACK or color == RED:
                continue
            if r == blue_row or c == green_col:
                continue
            loose_pixels.append((color, r, c))

    for color, r, c in sorted(loose_pixels):
        if color == BLUE:
            dr = 1 if blue_row > r else -1
            stop = (blue_row - dr, c)
            obstacle: tuple[int, int] | None = None
            for rr in range(r + dr, blue_row, dr):
                if int(g[rr, c]) == RED:
                    obstacle = (rr, c)
                    stop = (rr - dr, c)
                    break
            sr, sc = stop
            if int(out[sr, sc]) == BLACK:
                out[sr, sc] = BLUE
            if obstacle is not None:
                _draw_short_diagonal(out, g, BLUE, obstacle, (dr, 0))
        elif color == GREEN:
            dc = 1 if green_col > c else -1
            stop = (r, green_col - dc)
            obstacle = None
            for cc in range(c + dc, green_col, dc):
                if int(g[r, cc]) == RED:
                    obstacle = (r, cc)
                    stop = (r, cc - dc)
                    break
            sr, sc = stop
            if int(out[sr, sc]) == BLACK:
                out[sr, sc] = GREEN
            if obstacle is not None:
                _draw_short_diagonal(out, g, GREEN, obstacle, (0, dc))

    return out


def _empty_segment_vertical(grid: np.ndarray, col: int, r0: int, r1: int) -> bool:
    lo = min(r0, r1)
    hi = max(r0, r1)
    return bool(np.all(grid[lo : hi + 1, col] == BLACK))


def _empty_segment_horizontal(grid: np.ndarray, row: int, c0: int, c1: int) -> bool:
    lo = min(c0, c1)
    hi = max(c0, c1)
    return bool(np.all(grid[row, lo : hi + 1] == BLACK))


def _place_blue_task(
    grid: np.ndarray,
    blue_row: int,
    green_col: int,
    mode: str,
    visual_reserved: set[tuple[int, int]],
    halo_reserved: set[tuple[int, int]],
) -> dict | None:
    h, w = grid.shape
    candidates: list[dict] = []
    for c in range(1, w - 1):
        if c == green_col or abs(c - green_col) <= 1:
            continue
        for dr in (-1, 1):
            if dr == 1:
                clear_starts = range(0, blue_row - 1)
                block_obstacles = range(2, blue_row - 1)
            else:
                clear_starts = range(blue_row + 2, h)
                block_obstacles = range(blue_row + 2, h - 1)

            if mode == "clear":
                for start_r in clear_starts:
                    stop_r = blue_row - dr
                    if not _empty_segment_vertical(grid, c, start_r, stop_r):
                        continue
                    start = (start_r, c)
                    stop = (stop_r, c)
                    if start in visual_reserved or stop in visual_reserved:
                        continue
                    if start in halo_reserved:
                        continue
                    candidates.append(
                        {
                            "color": BLUE,
                            "start": start,
                            "stop": stop,
                            "obstacle": None,
                            "marks": set(),
                        }
                    )
            else:
                for obs_r in block_obstacles:
                    if dr == 1:
                        start_rows = range(0, obs_r - 1)
                    else:
                        start_rows = range(obs_r + 2, h)
                    for start_r in start_rows:
                        stop_r = obs_r - dr
                        start = (start_r, c)
                        obstacle = (obs_r, c)
                        stop = (stop_r, c)
                        marks = {
                            (obs_r, c - 1),
                            (obs_r, c + 1),
                            (obs_r + dr, c - 1),
                            (obs_r + dr, c + 1),
                        }
                        if not all(_in_bounds((h, w), rr, cc) for rr, cc in marks):
                            continue
                        if any(rr == blue_row or cc == green_col for rr, cc in marks):
                            continue
                        if not _empty_segment_vertical(grid, c, start_r, blue_row - dr):
                            continue
                        if int(grid[obstacle]) != BLACK:
                            continue
                        if any(cell in visual_reserved for cell in {start, obstacle, stop, *marks}):
                            continue
                        if start in halo_reserved or obstacle in halo_reserved:
                            continue
                        candidates.append(
                            {
                                "color": BLUE,
                                "start": start,
                                "stop": stop,
                                "obstacle": obstacle,
                                "marks": marks,
                            }
                        )
    if not candidates:
        return None
    return random.choice(candidates)


def _place_green_task(
    grid: np.ndarray,
    blue_row: int,
    green_col: int,
    mode: str,
    visual_reserved: set[tuple[int, int]],
    halo_reserved: set[tuple[int, int]],
) -> dict | None:
    h, w = grid.shape
    candidates: list[dict] = []
    for r in range(1, h - 1):
        if r == blue_row or abs(r - blue_row) <= 1:
            continue
        for dc in (-1, 1):
            if dc == 1:
                clear_starts = range(0, green_col - 1)
                block_obstacles = range(2, green_col - 1)
            else:
                clear_starts = range(green_col + 2, w)
                block_obstacles = range(green_col + 2, w - 1)

            if mode == "clear":
                for start_c in clear_starts:
                    stop_c = green_col - dc
                    if not _empty_segment_horizontal(grid, r, start_c, stop_c):
                        continue
                    start = (r, start_c)
                    stop = (r, stop_c)
                    if start in visual_reserved or stop in visual_reserved:
                        continue
                    if start in halo_reserved:
                        continue
                    candidates.append(
                        {
                            "color": GREEN,
                            "start": start,
                            "stop": stop,
                            "obstacle": None,
                            "marks": set(),
                        }
                    )
            else:
                for obs_c in block_obstacles:
                    if dc == 1:
                        start_cols = range(0, obs_c - 1)
                    else:
                        start_cols = range(obs_c + 2, w)
                    for start_c in start_cols:
                        stop_c = obs_c - dc
                        start = (r, start_c)
                        obstacle = (r, obs_c)
                        stop = (r, stop_c)
                        marks = {
                            (r - 1, obs_c),
                            (r + 1, obs_c),
                            (r - 1, obs_c + dc),
                            (r + 1, obs_c + dc),
                        }
                        if not all(_in_bounds((h, w), rr, cc) for rr, cc in marks):
                            continue
                        if any(rr == blue_row or cc == green_col for rr, cc in marks):
                            continue
                        if not _empty_segment_horizontal(grid, r, start_c, green_col - dc):
                            continue
                        if int(grid[obstacle]) != BLACK:
                            continue
                        if any(cell in visual_reserved for cell in {start, obstacle, stop, *marks}):
                            continue
                        if start in halo_reserved or obstacle in halo_reserved:
                            continue
                        candidates.append(
                            {
                                "color": GREEN,
                                "start": start,
                                "stop": stop,
                                "obstacle": obstacle,
                                "marks": marks,
                            }
                        )
    if not candidates:
        return None
    return random.choice(candidates)


def _place_orange(
    grid: np.ndarray,
    blue_row: int,
    green_col: int,
    visual_reserved: set[tuple[int, int]],
    halo_reserved: set[tuple[int, int]],
) -> tuple[int, int] | None:
    h, w = grid.shape
    candidates = []
    for r in range(h):
        for c in range(w):
            if r == blue_row or c == green_col:
                continue
            if int(grid[r, c]) != BLACK:
                continue
            cell = (r, c)
            if cell in visual_reserved or cell in halo_reserved:
                continue
            candidates.append(cell)
    if not candidates:
        return None
    return random.choice(candidates)


def _apply_task(
    grid: np.ndarray,
    task: dict,
    visual_reserved: set[tuple[int, int]],
    halo_reserved: set[tuple[int, int]],
) -> None:
    color = int(task["color"])
    sr, sc = task["start"]
    grid[sr, sc] = color
    visual_reserved.add((sr, sc))
    halo_reserved.update(_neighbors8(grid.shape, sr, sc))

    if task["obstacle"] is not None:
        orow, ocol = task["obstacle"]
        grid[orow, ocol] = RED
        visual_reserved.add((orow, ocol))
        halo_reserved.update(_neighbors8(grid.shape, orow, ocol))

    visual_reserved.add(tuple(task["stop"]))
    visual_reserved.update(task["marks"])


def _classify_loose_pixels(grid: np.ndarray) -> tuple[int, int, int]:
    blue_row = _find_anchor_row(grid)
    green_col = _find_anchor_col(grid)
    blocked = 0
    clear = 0
    unmatched = 0
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color in (BLACK, RED) or r == blue_row or c == green_col:
                continue
            if color == BLUE:
                dr = 1 if blue_row > r else -1
                hit = any(int(grid[rr, c]) == RED for rr in range(r + dr, blue_row, dr))
                blocked += int(hit)
                clear += int(not hit)
            elif color == GREEN:
                dc = 1 if green_col > c else -1
                hit = any(int(grid[r, cc]) == RED for cc in range(c + dc, green_col, dc))
                blocked += int(hit)
                clear += int(not hit)
            else:
                unmatched += 1
    return blocked, clear, unmatched


def _validate_generated(grid: np.ndarray, tasks: list[dict], oranges: list[tuple[int, int]]) -> bool:
    h, w = grid.shape
    if h > 30 or w > 30:
        return False

    blue_row = _find_anchor_row(grid)
    green_col = _find_anchor_col(grid)
    if not (2 <= blue_row < h - 2 and 2 <= green_col < w - 2):
        return False

    blocked, clear, unmatched = _classify_loose_pixels(grid)
    if blocked < 2 or clear < 1 or unmatched < 1:
        return False

    blue_tasks = sum(task["color"] == BLUE for task in tasks)
    green_tasks = sum(task["color"] == GREEN for task in tasks)
    if blue_tasks < 2 or green_tasks < 2:
        return False

    if not any(task["color"] == BLUE and task["obstacle"] is not None for task in tasks):
        return False
    if not any(task["color"] == GREEN and task["obstacle"] is not None for task in tasks):
        return False
    if not any(task["obstacle"] is None for task in tasks):
        return False

    non_bar_non_obstacles = [
        (r, c)
        for r in range(h)
        for c in range(w)
        if int(grid[r, c]) not in (BLACK, RED) and r != blue_row and c != green_col
    ]
    for r, c in non_bar_non_obstacles:
        for rr, cc in _neighbors8((h, w), r, c):
            if (rr, cc) == (r, c):
                continue
            if rr == blue_row or cc == green_col:
                continue
            if int(grid[rr, cc]) != BLACK:
                return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False
    if int(np.count_nonzero(out == RED)) != int(np.count_nonzero(grid == RED)):
        return False
    if len(oranges) != unmatched:
        return False
    return True


def _attempt_generate() -> np.ndarray | None:
    h = random.randint(*HEIGHT_RANGE)
    w = random.randint(*WIDTH_RANGE)
    grid = np.zeros((h, w), dtype=int)

    blue_row = random.randint(2, h - 3)
    green_col = random.randint(2, w - 3)
    intersection_color = random.choice((BLUE, GREEN))

    grid[blue_row, :] = BLUE
    grid[:, green_col] = GREEN
    grid[blue_row, green_col] = intersection_color

    visual_reserved = {(blue_row, c) for c in range(w)}
    visual_reserved.update((r, green_col) for r in range(h))
    halo_reserved: set[tuple[int, int]] = set()

    blue_modes = ["blocked", "clear"]
    green_modes = ["blocked", "clear"]
    if random.random() < 0.5:
        blue_modes.append(random.choice(("blocked", "clear")))
    if random.random() < 0.5:
        green_modes.append(random.choice(("blocked", "clear")))
    random.shuffle(blue_modes)
    random.shuffle(green_modes)

    tasks: list[dict] = []
    for mode in sorted(blue_modes, key=lambda x: x != "blocked"):
        task = _place_blue_task(grid, blue_row, green_col, mode, visual_reserved, halo_reserved)
        if task is None:
            return None
        _apply_task(grid, task, visual_reserved, halo_reserved)
        tasks.append(task)

    for mode in sorted(green_modes, key=lambda x: x != "blocked"):
        task = _place_green_task(grid, blue_row, green_col, mode, visual_reserved, halo_reserved)
        if task is None:
            return None
        _apply_task(grid, task, visual_reserved, halo_reserved)
        tasks.append(task)

    oranges: list[tuple[int, int]] = []
    for _ in range(random.randint(1, 2)):
        cell = _place_orange(grid, blue_row, green_col, visual_reserved, halo_reserved)
        if cell is None:
            return None
        r, c = cell
        grid[r, c] = ORANGE
        oranges.append(cell)
        visual_reserved.add(cell)
        halo_reserved.update(_neighbors8(grid.shape, r, c))

    if not _validate_generated(grid, tasks, oranges):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
