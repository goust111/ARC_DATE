from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

Coord = tuple[int, int]
BBox = tuple[int, int, int, int]

GEN_RETRY_LIMIT = 500


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int) -> tuple[Coord, Coord, Coord, Coord]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _frame_cells(box: BBox) -> set[Coord]:
    r0, c0, r1, c1 = box
    cells: set[Coord] = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _touch_color_map(grid: np.ndarray, frame: set[Coord], pos: Coord) -> set[int]:
    colors = set()
    for rr, cc in _neighbors4(*pos):
        if (rr, cc) in frame:
            colors.add(int(grid[rr, cc]))
    return colors


def _is_valid_frame(grid: np.ndarray, box: BBox) -> bool:
    r0, c0, r1, c1 = box
    if r1 - r0 + 1 < 5 or c1 - c0 + 1 < 5:
        return False
    if not np.all(grid[r0 + 1 : r1, c0 + 1 : c1] == BG):
        return False
    if not np.all(grid[r0, c0 + 1 : c1] == RED):
        return False
    if not np.all(grid[r1, c0 + 1 : c1] == YELLOW):
        return False
    if not np.all(grid[r0 + 1 : r1, c0] == BLUE):
        return False
    if not np.all(grid[r0 + 1 : r1, c1] == GREEN):
        return False
    tl = int(grid[r0, c0])
    tr = int(grid[r0, c1])
    bl = int(grid[r1, c0])
    br = int(grid[r1, c1])
    return tl in {BLUE, RED} and tr in {RED, GREEN} and bl in {BLUE, YELLOW} and br in {GREEN, YELLOW}


def _find_frame(grid: np.ndarray) -> BBox:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    best: BBox | None = None
    best_area = -1
    for r0 in range(h - 4):
        for c0 in range(w - 4):
            if int(g[r0, c0 + 1]) != RED:
                continue
            for r1 in range(r0 + 4, h):
                if int(g[r1, c0 + 1]) != YELLOW:
                    continue
                for c1 in range(c0 + 4, w):
                    if int(g[r0 + 1, c1]) != GREEN:
                        continue
                    if _is_valid_frame(g, (r0, c0, r1, c1)):
                        area = (r1 - r0 + 1) * (c1 - c0 + 1)
                        if area > best_area:
                            best = (r0, c0, r1, c1)
                            best_area = area
    if best is None:
        raise AssertionError("frame not found")
    return best


def _simulate_one(grid: np.ndarray, start: Coord, frame_box: BBox) -> tuple[list[Coord], int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    frame = _frame_cells(frame_box)
    cur = start
    path: list[Coord] = []
    seen = {start}

    for _ in range(h * w * 4):
        touch_colors = _touch_color_map(g, frame, cur)
        if touch_colors:
            if len(touch_colors) != 1:
                raise AssertionError("touch color is ambiguous")
            return path, next(iter(touch_colors))

        r, c = cur
        if r == 0:
            raise AssertionError("path reached top border without touching the frame")
        up_blocked = (r - 1, c) in frame or int(g[r - 1, c]) == GREY
        if not up_blocked:
            nxt = (r - 1, c)
        else:
            if c + 1 >= w or (r, c + 1) in frame or int(g[r, c + 1]) == GREY:
                nxt = (r, c - 1)
            else:
                nxt = (r, c + 1)

        nr, nc = nxt
        if not _in_bounds((h, w), nr, nc):
            raise AssertionError("path left the grid")
        if nxt in frame:
            raise AssertionError("path stepped onto the frame")
        if nxt in seen:
            raise AssertionError("path looped")
        seen.add(nxt)
        path.append(nxt)
        cur = nxt

    raise AssertionError("path did not terminate")


def _simulate_all(grid: np.ndarray, frame_box: BBox) -> tuple[dict[Coord, list[Coord]], dict[Coord, int]]:
    g = np.asarray(grid, dtype=int)
    frame = _frame_cells(frame_box)
    starts = []
    for r, c in np.argwhere(g == YELLOW):
        pt = (int(r), int(c))
        if pt not in frame:
            starts.append(pt)
    starts.sort()
    if len(starts) < 2:
        raise AssertionError("expected 2-3 start pixels")

    paths: dict[Coord, list[Coord]] = {}
    colors: dict[Coord, int] = {}
    for start in starts:
        path, color = _simulate_one(g, start, frame_box)
        paths[start] = path
        colors[start] = color
    return paths, colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each yellow start upward until it touches the colored frame, then recolor the whole trail by the touched frame side color.

    Input structure:
    - Black background with one hollow rectangle near the center.
    - The frame uses fixed side colors: top red, left blue, right green, bottom yellow.
    - Two or three yellow start pixels sit below the frame, and gray obstacle pixels lie between the starts and the frame.

    Rule:
    - For each yellow start, repeatedly try to move one cell upward.
    - If the cell above is a gray obstacle, move right instead; if right is blocked or at the right border, move left instead.
    - Stop as soon as the current trail cell is 4-neighbor adjacent to a frame pixel, without overwriting the frame or the start.
    - Recolor every visited trail cell by the color of the touched frame side pixel; keep the starts and the frame unchanged.

    Key insight:
    - The obstacles determine both the bent path geometry and which frame side is touched, and that touched side color is only known after the full growth path has finished.
    """
    g = np.asarray(grid, dtype=int)
    frame_box = _find_frame(g)
    paths, colors = _simulate_all(g, frame_box)

    out = g.copy()
    for start, path in paths.items():
        _ = start
        color = colors[start]
        for r, c in path:
            out[r, c] = color
    return out


def _draw_frame(grid: np.ndarray, box: BBox) -> None:
    r0, c0, r1, c1 = box
    grid[r0, c0 : c1 + 1] = RED
    grid[r1, c0 : c1 + 1] = YELLOW
    grid[r0 : r1 + 1, c0] = BLUE
    grid[r0 : r1 + 1, c1] = GREEN
    grid[r0, c0] = RED
    grid[r0, c1] = RED
    grid[r1, c0] = BLUE
    grid[r1, c1] = GREEN


def _build_path_from_moves(start: Coord, moves: Iterable[str]) -> tuple[list[Coord], list[Coord]]:
    cur = start
    path = [start]
    obstacle_needs: list[Coord] = []
    for step in moves:
        r, c = cur
        if step == "U":
            nxt = (r - 1, c)
        elif step == "R":
            obstacle_needs.append((r - 1, c))
            nxt = (r, c + 1)
        elif step == "L":
            obstacle_needs.append((r - 1, c))
            obstacle_needs.append((r, c + 1))
            nxt = (r, c - 1)
        else:
            raise ValueError(f"unknown step: {step}")
        path.append(nxt)
        cur = nxt
    return path, obstacle_needs


def _path_for_left(frame_box: BBox, start: Coord) -> tuple[list[Coord], list[Coord], str]:
    _r0, c0, r1, _c1 = frame_box
    sr, sc = start
    left_steps = sc - (c0 - 1)
    moves: list[str] = []
    initial_up = random.randint(0, 1)
    moves.extend("U" for _ in range(initial_up))
    left_row = sr - initial_up
    if left_row - (left_steps - 1) < r1 + 2:
        raise AssertionError("left path too short for staircase")
    for i in range(left_steps):
        moves.append("L")
        if i != left_steps - 1:
            moves.append("U")
    cur_row = sr - initial_up - (left_steps - 1)
    moves.extend("U" for _ in range(cur_row - r1))
    path, obstacles = _build_path_from_moves(start, moves)
    return path, obstacles, "left"


def _path_for_top(frame_box: BBox, start: Coord) -> tuple[list[Coord], list[Coord], str]:
    r0, c0, _r1, c1 = frame_box
    sr, sc = start
    outer_col = c0 - 2
    top_col = random.randint(c0, c1)
    left_steps = sc - outer_col
    moves: list[str] = []
    initial_up = random.randint(0, 1)
    moves.extend("U" for _ in range(initial_up))
    left_row = sr - initial_up
    if left_row - (left_steps - 1) < frame_box[2] + 2:
        raise AssertionError("top path cannot clear frame bottom")
    for i in range(left_steps):
        moves.append("L")
        if i != left_steps - 1:
            moves.append("U")
    cur_row = sr - initial_up - (left_steps - 1)
    moves.extend("U" for _ in range(cur_row - (r0 - 1)))
    right_steps = top_col - outer_col
    moves.extend("R" for _ in range(right_steps))
    path, obstacles = _build_path_from_moves(start, moves)
    return path, obstacles, "top"


def _path_for_right(frame_box: BBox, start: Coord) -> tuple[list[Coord], list[Coord], str]:
    _r0, _c0, r1, c1 = frame_box
    sr, sc = start
    detour_row = random.randint(r1 + 2, sr)
    moves: list[str] = []
    moves.extend("U" for _ in range(sr - detour_row))
    right_steps = (c1 + 1) - sc
    moves.extend("R" for _ in range(right_steps))
    moves.extend("U" for _ in range(detour_row - r1))
    path, obstacles = _build_path_from_moves(start, moves)
    return path, obstacles, "right"


def _make_scene() -> tuple[np.ndarray, dict[Coord, dict[str, object]]]:
    width = random.randint(14, 20)
    frame_w = random.randint(7, min(10, width - 6))
    c0 = random.randint(3, width - frame_w - 3)
    c1 = c0 + frame_w - 1

    frame_h = random.randint(6, 8)
    r0 = random.randint(2, 4)
    r1 = r0 + frame_h - 1

    base_start_row = r1 + random.randint(5, 7)
    height = base_start_row + random.randint(2, 4)
    if height > 30:
        raise AssertionError("scene too tall")

    grid = np.zeros((height, width), dtype=int)
    frame_box = (r0, c0, r1, c1)
    _draw_frame(grid, frame_box)

    side_sets = [
        ("top", "right"),
        ("left", "top"),
        ("left", "top", "right"),
        ("left", "top", "right"),
        ("left", "top", "right"),
    ]
    sides = list(random.choice(side_sets))
    start_rows = {
        "top": base_start_row,
        "left": base_start_row - (1 if "top" in sides else 0),
        "right": base_start_row - random.choice([0, 1]),
    }
    start_cols = {"top": c0, "left": c0 + 1, "right": c1 - random.choice([1, 2])}
    starts = {side: (start_rows[side], start_cols[side]) for side in sides}
    if len(set(starts.values())) != len(starts):
        raise AssertionError("duplicate starts")

    plans: dict[Coord, dict[str, object]] = {}
    reserved_paths: set[Coord] = set()
    reserved_obstacles: set[Coord] = set()
    reserved_starts = set(starts.values())

    for side in sides:
        start = starts[side]
        if side == "left":
            path, obstacles, target = _path_for_left(frame_box, start)
        elif side == "top":
            path, obstacles, target = _path_for_top(frame_box, start)
        else:
            path, obstacles, target = _path_for_right(frame_box, start)

        if len(path) < 2:
            raise AssertionError("degenerate path")
        body = path[1:]
        if len(body) != len(set(body)):
            raise AssertionError("path self-overlap")
        if any(cell in reserved_paths for cell in body):
            raise AssertionError("paths overlap")
        if any(cell in reserved_starts - {start} for cell in body):
            raise AssertionError("path crossed another start")
        if any(cell in reserved_paths for cell in obstacles):
            raise AssertionError("obstacle on earlier path")
        if any(cell in reserved_obstacles for cell in obstacles):
            raise AssertionError("obstacles overlap")
        if any(cell in reserved_starts for cell in obstacles):
            raise AssertionError("obstacle on a start")

        for cell in path:
            if not _in_bounds(grid.shape, *cell):
                raise AssertionError("path out of bounds")
            if cell in _frame_cells(frame_box):
                raise AssertionError("path on frame")
        for cell in obstacles:
            if not _in_bounds(grid.shape, *cell):
                raise AssertionError("obstacle out of bounds")
            if cell in _frame_cells(frame_box):
                raise AssertionError("obstacle on frame")
            if cell in path:
                raise AssertionError("obstacle on path")

        reserved_paths.update(body)
        reserved_obstacles.update(obstacles)
        plans[start] = {
            "path": body,
            "obstacles": obstacles,
            "target": target,
        }

    if len(sides) != len(plans):
        raise AssertionError("duplicate starts")

    for start in plans:
        if start in reserved_obstacles:
            raise AssertionError("start overlapped obstacle")
        grid[start] = YELLOW

    for cell in reserved_obstacles:
        if int(grid[cell]) != BG:
            raise AssertionError("obstacle overwrote object")
        grid[cell] = GREY

    return grid, plans


def _side_from_color(color: int) -> str:
    if color == BLUE:
        return "left"
    if color == RED:
        return "top"
    if color == GREEN:
        return "right"
    raise AssertionError("unexpected terminal color")


def _validate_scene(grid: np.ndarray, plans: dict[Coord, dict[str, object]]) -> bool:
    frame_box = _find_frame(grid)
    frame = _frame_cells(frame_box)
    if np.max(grid.shape) > 30:
        return False

    starts = sorted(plans)
    if len(starts) not in {2, 3}:
        return False

    paths, colors = _simulate_all(grid, frame_box)
    reached_sides = set()
    used_horizontal = False
    occupied: set[Coord] = set()

    for start in starts:
        expected_path = plans[start]["path"]
        expected_target = plans[start]["target"]
        if paths[start] != expected_path:
            return False
        side = _side_from_color(colors[start])
        if side != expected_target:
            return False
        reached_sides.add(side)
        if any(p[0] == q[0] for p, q in zip([start] + paths[start][:-1], paths[start])):
            used_horizontal = True
        for cell in paths[start]:
            if cell in occupied:
                return False
            occupied.add(cell)
            if cell in frame or int(grid[cell]) != BG:
                return False

    if len(reached_sides) < 2 or not used_horizontal:
        return False

    for start in starts:
        obstacles = list(plans[start]["obstacles"])
        if not obstacles:
            return False
        probe = grid.copy()
        rr, cc = obstacles[0]
        probe[rr, cc] = BG
        try:
            new_path, new_color = _simulate_one(probe, start, frame_box)
        except AssertionError:
            continue
        if new_path == paths[start] and new_color == colors[start]:
            return False

    try:
        out = main(grid)
    except AssertionError:
        return False

    for start in starts:
        color = colors[start]
        for r, c in paths[start]:
            if int(out[r, c]) != color:
                return False
        if int(out[start]) != YELLOW:
            return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        try:
            grid, plans = _make_scene()
        except AssertionError:
            continue
        if _validate_scene(grid, plans):
            return grid
    raise RuntimeError("failed to generate a valid ARC input")
