from __future__ import annotations

import random

import numpy as np


BG = 0
GREY = 5
AZURE = 8
MAROON = 9
COLOR_POOL = (1, 2, 3, 4, 6, 7)
DIRECTIONS = ("up", "down", "left", "right")
POINTER_BY_DIRECTION = {
    "up": (0, 1),
    "down": (2, 1),
    "left": (1, 0),
    "right": (1, 2),
}
GEN_RETRIES = 300


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _detect_clocks(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = grid.shape
    clocks: list[dict[str, object]] = []
    for r0 in range(h - 2):
        for c0 in range(w - 2):
            patch = grid[r0 : r0 + 3, c0 : c0 + 3]
            center = int(patch[1, 1])
            if center in (bg, GREY, MAROON):
                continue
            direction = None
            for name, pointer in POINTER_BY_DIRECTION.items():
                ok = True
                for rr in range(3):
                    for cc in range(3):
                        value = int(patch[rr, cc])
                        if (rr, cc) == (1, 1):
                            if value != center:
                                ok = False
                                break
                        elif (rr, cc) == pointer:
                            if value != MAROON:
                                ok = False
                                break
                        elif value != GREY:
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    direction = name
                    break
            if direction is None:
                continue
            clocks.append(
                {
                    "color": center,
                    "direction": direction,
                    "center": (r0 + 1, c0 + 1),
                }
            )
    return clocks


def _edge_hit_base(grid: np.ndarray, direction: str, lane: int) -> int:
    h, w = grid.shape
    if direction == "up":
        return 1 if int(grid[0, lane]) == AZURE else 0
    if direction == "down":
        return h - 2 if int(grid[h - 1, lane]) == AZURE else h - 1
    if direction == "left":
        return 1 if int(grid[lane, 0]) == AZURE else 0
    return w - 2 if int(grid[lane, w - 1]) == AZURE else w - 1


def _distance_to_base(start: tuple[int, int], direction: str, base: int) -> int:
    r, c = start
    if direction == "up":
        return int(r - base)
    if direction == "down":
        return int(base - r)
    if direction == "left":
        return int(c - base)
    return int(base - c)


def _stack_position(direction: str, lane: int, base: int, index: int) -> tuple[int, int]:
    if direction == "up":
        return (base + index, lane)
    if direction == "down":
        return (base - index, lane)
    if direction == "left":
        return (lane, base + index)
    return (lane, base - index)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read each 3x3 clock, decode its maroon pointer direction, and push the
    matching cargo pixel to the wall or edge along that lane.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    clocks = _detect_clocks(g, bg)
    if not clocks:
        return g.copy()

    color_to_clock = {int(clock["color"]): clock for clock in clocks}
    center_positions = {tuple(clock["center"]) for clock in clocks}

    cargos: dict[int, tuple[int, int]] = {}
    for color in color_to_clock:
        positions = [
            (int(r), int(c))
            for r, c in np.argwhere(g == color)
            if (int(r), int(c)) not in center_positions
        ]
        if len(positions) == 1:
            cargos[color] = positions[0]

    out = g.copy()
    groups: dict[tuple[str, int], list[dict[str, object]]] = {}
    for color, start in cargos.items():
        direction = str(color_to_clock[color]["direction"])
        lane = int(start[1] if direction in ("up", "down") else start[0])
        base = _edge_hit_base(g, direction, lane)
        dist = _distance_to_base(start, direction, base)
        groups.setdefault((direction, lane), []).append(
            {"color": color, "start": start, "base": base, "dist": dist}
        )
        out[start] = bg

    for direction, lane in sorted(groups):
        items = groups[(direction, lane)]
        items.sort(key=lambda item: int(item["dist"]))
        base = int(items[0]["base"])
        for index, item in enumerate(items):
            rr, cc = _stack_position(direction, lane, base, index)
            out[rr, cc] = int(item["color"])

    return out


def _choose_wall_segment(length: int) -> tuple[int, int]:
    max_len = min(length // 2, length - 10)
    seg_len = random.randint(3, max_len)
    start = random.randint(2, length - seg_len - 2)
    return start, seg_len


def _wall_cells(shape: tuple[int, int], walls: dict[str, tuple[int, int]]) -> set[tuple[int, int]]:
    h, w = shape
    cells: set[tuple[int, int]] = set()
    top_start, top_len = walls["top"]
    bottom_start, bottom_len = walls["bottom"]
    left_start, left_len = walls["left"]
    right_start, right_len = walls["right"]
    for c in range(top_start, top_start + top_len):
        cells.add((0, c))
    for c in range(bottom_start, bottom_start + bottom_len):
        cells.add((h - 1, c))
    for r in range(left_start, left_start + left_len):
        cells.add((r, 0))
    for r in range(right_start, right_start + right_len):
        cells.add((r, w - 1))
    return cells


def _paint_walls(grid: np.ndarray, walls: dict[str, tuple[int, int]]) -> None:
    h, w = grid.shape
    top_start, top_len = walls["top"]
    bottom_start, bottom_len = walls["bottom"]
    left_start, left_len = walls["left"]
    right_start, right_len = walls["right"]
    grid[0, top_start : top_start + top_len] = AZURE
    grid[h - 1, bottom_start : bottom_start + bottom_len] = AZURE
    grid[left_start : left_start + left_len, 0] = AZURE
    grid[right_start : right_start + right_len, w - 1] = AZURE


def _lane_candidates(length: int, segment: tuple[int, int], hit_wall: bool) -> list[int]:
    start, seg_len = segment
    wall_coords = set(range(start, start + seg_len))
    return [
        coord
        for coord in range(4, length - 4)
        if (coord in wall_coords) == bool(hit_wall)
    ]


def _base_from_walls(
    shape: tuple[int, int], walls: dict[str, tuple[int, int]], direction: str, lane: int
) -> int:
    h, w = shape
    if direction == "up":
        start, seg_len = walls["top"]
        return 1 if start <= lane < start + seg_len else 0
    if direction == "down":
        start, seg_len = walls["bottom"]
        return h - 2 if start <= lane < start + seg_len else h - 1
    if direction == "left":
        start, seg_len = walls["left"]
        return 1 if start <= lane < start + seg_len else 0
    start, seg_len = walls["right"]
    return w - 2 if start <= lane < start + seg_len else w - 1


def _final_positions_from_specs(
    shape: tuple[int, int], walls: dict[str, tuple[int, int]], specs: list[dict[str, object]]
) -> dict[int, tuple[int, int]]:
    groups: dict[tuple[str, int], list[dict[str, object]]] = {}
    for spec in specs:
        direction = str(spec["direction"])
        start = tuple(spec["start"])
        lane = int(start[1] if direction in ("up", "down") else start[0])
        base = _base_from_walls(shape, walls, direction, lane)
        dist = _distance_to_base(start, direction, base)
        groups.setdefault((direction, lane), []).append(
            {"color": int(spec["color"]), "start": start, "base": base, "dist": dist}
        )

    finals: dict[int, tuple[int, int]] = {}
    for direction, lane in groups:
        items = groups[(direction, lane)]
        items.sort(key=lambda item: int(item["dist"]))
        base = int(items[0]["base"])
        for index, item in enumerate(items):
            finals[int(item["color"])] = _stack_position(direction, lane, base, index)
    return finals


def _expected_output_from_specs(
    grid: np.ndarray, walls: dict[str, tuple[int, int]], specs: list[dict[str, object]]
) -> np.ndarray:
    out = np.array(grid, copy=True)
    finals = _final_positions_from_specs(grid.shape, walls, specs)
    for spec in specs:
        out[tuple(spec["start"])] = BG
    for spec in specs:
        out[finals[int(spec["color"])]] = int(spec["color"])
    return out


def _path_cells(
    start: tuple[int, int], end: tuple[int, int]
) -> set[tuple[int, int]]:
    if start[0] == end[0]:
        r = start[0]
        c0, c1 = sorted((start[1], end[1]))
        return {(r, c) for c in range(c0, c1 + 1)}
    c = start[1]
    r0, r1 = sorted((start[0], end[0]))
    return {(r, c) for r in range(r0, r1 + 1)}


def _clock_patch(color: int, direction: str) -> np.ndarray:
    patch = np.full((3, 3), GREY, dtype=int)
    patch[1, 1] = color
    pr, pc = POINTER_BY_DIRECTION[direction]
    patch[pr, pc] = MAROON
    return patch


def _sample_rows(h: int, blocked_rows: set[int], count: int, reverse: bool) -> list[int] | None:
    candidates = [r for r in range(4, h - 4) if r not in blocked_rows]
    if len(candidates) < count:
        return None
    rows = random.sample(candidates, count)
    rows.sort(reverse=reverse)
    return rows


def _sample_cols(w: int, blocked_cols: set[int], count: int, reverse: bool) -> list[int] | None:
    candidates = [c for c in range(4, w - 4) if c not in blocked_cols]
    if len(candidates) < count:
        return None
    cols = random.sample(candidates, count)
    cols.sort(reverse=reverse)
    return cols


def _pick_lane_pair(
    length: int,
    first_segment: tuple[int, int],
    first_hit: bool,
    second_segment: tuple[int, int],
    second_hit: bool,
) -> tuple[int, int] | None:
    first_candidates = _lane_candidates(length, first_segment, first_hit)
    second_candidates = _lane_candidates(length, second_segment, second_hit)
    random.shuffle(first_candidates)
    random.shuffle(second_candidates)
    for first in first_candidates:
        for second in second_candidates:
            if first != second:
                return first, second
    return None


def _place_clocks(
    grid: np.ndarray,
    blocked: np.ndarray,
    clock_specs: list[tuple[int, str]],
) -> bool:
    h, w = grid.shape
    positions = [(r0, c0) for r0 in range(1, h - 3) for c0 in range(1, w - 3)]
    random.shuffle(positions)
    for color, direction in clock_specs:
        patch = _clock_patch(color, direction)
        placed = False
        random.shuffle(positions)
        for r0, c0 in positions:
            rr0 = r0 - 1
            cc0 = c0 - 1
            rr1 = r0 + 3
            cc1 = c0 + 3
            if blocked[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
                continue
            grid[r0 : r0 + 3, c0 : c0 + 3] = patch
            blocked[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
            placed = True
            break
        if not placed:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(16, 22)
    w = random.randint(16, 22)
    grid = np.full((h, w), BG, dtype=int)

    walls = {
        "top": _choose_wall_segment(w),
        "bottom": _choose_wall_segment(w),
        "left": _choose_wall_segment(h),
        "right": _choose_wall_segment(h),
    }
    _paint_walls(grid, walls)

    wall_dirs = set(random.sample(DIRECTIONS, 2))
    hits_wall = {direction: direction in wall_dirs for direction in DIRECTIONS}

    vertical_pair = _pick_lane_pair(
        w, walls["top"], hits_wall["up"], walls["bottom"], hits_wall["down"]
    )
    horizontal_pair = _pick_lane_pair(
        h, walls["left"], hits_wall["left"], walls["right"], hits_wall["right"]
    )
    if vertical_pair is None or horizontal_pair is None:
        return None
    up_col, down_col = vertical_pair
    left_row, right_row = horizontal_pair

    counts = {direction: 1 for direction in DIRECTIONS}
    extras = random.randint(1, 2)
    for direction in random.choices(DIRECTIONS, k=extras):
        counts[direction] += 1

    up_rows = _sample_rows(h, {left_row, right_row}, counts["up"], reverse=False)
    down_rows = _sample_rows(h, {left_row, right_row}, counts["down"], reverse=True)
    left_cols = _sample_cols(w, {up_col, down_col}, counts["left"], reverse=False)
    right_cols = _sample_cols(w, {up_col, down_col}, counts["right"], reverse=True)
    if None in (up_rows, down_rows, left_cols, right_cols):
        return None

    color_count = sum(counts.values())
    colors = random.sample(COLOR_POOL, color_count)
    random.shuffle(colors)
    color_iter = iter(colors)

    specs: list[dict[str, object]] = []
    for row in up_rows:
        specs.append({"color": next(color_iter), "direction": "up", "start": (row, up_col)})
    for row in down_rows:
        specs.append({"color": next(color_iter), "direction": "down", "start": (row, down_col)})
    for col in left_cols:
        specs.append({"color": next(color_iter), "direction": "left", "start": (left_row, col)})
    for col in right_cols:
        specs.append({"color": next(color_iter), "direction": "right", "start": (right_row, col)})
    random.shuffle(specs)

    for spec in specs:
        grid[tuple(spec["start"])] = int(spec["color"])

    finals = _final_positions_from_specs(grid.shape, walls, specs)
    if len(set(finals.values())) != len(specs):
        return None

    blocked = np.zeros((h, w), dtype=bool)
    for r, c in _wall_cells((h, w), walls):
        blocked[r, c] = True
    for spec in specs:
        color = int(spec["color"])
        start = tuple(spec["start"])
        end = finals[color]
        for r, c in _path_cells(start, end):
            blocked[r, c] = True

    clock_specs = [(int(spec["color"]), str(spec["direction"])) for spec in specs]
    random.shuffle(clock_specs)
    if not _place_clocks(grid, blocked, clock_specs):
        return None

    expected = _expected_output_from_specs(grid, walls, specs)
    actual = main(grid)
    if np.array_equal(actual, grid):
        return None
    if not np.array_equal(actual, expected):
        return None
    if len(_detect_clocks(grid, BG)) != len(specs):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
