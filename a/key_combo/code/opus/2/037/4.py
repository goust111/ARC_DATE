import random
from collections import Counter

import numpy as np


BG = 0
RED = 2
GREY = 5
PAIR_COLORS = [1, 3, 4, 6, 7, 8, 9]

PAIR_COUNT_CHOICES = [2, 3]
PAIR_COUNT_WEIGHTS = [4, 6]
GRID_MIN = 17
GRID_MAX = 22
RECT_SIDE_CHOICES = [2, 3]
DELTA_MIN = 4
DELTA_MAX = 8
OBSTACLE_LEN_MIN = 2
OBSTACLE_LEN_MAX = 5
NOISE_MAX = 4
GEN_ATTEMPTS = 400
PAIR_ATTEMPTS = 300
OBSTACLE_ATTEMPTS = 200


BBox = tuple[int, int, int, int]
Coord = tuple[int, int]


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def colors_present(grid: np.ndarray, *, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, np.unique(grid)))
    if exclude is not None:
        colors -= set(map(int, exclude))
    return sorted(colors)


def bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_of_color(grid: np.ndarray, color: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    out = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            coords_arr = np.array(coords, dtype=int)
            out.append(
                {
                    "coords": coords_arr,
                    "bbox": bbox_coords(coords_arr),
                    "area": int(coords_arr.shape[0]),
                }
            )
    return out


def _is_solid_rectangle(grid: np.ndarray, color: int, bbox: BBox) -> bool:
    r0, c0, r1, c1 = bbox
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    return bool(np.all(patch == int(color)))


def _sort_pair_components(components: list[dict]) -> tuple[dict, dict]:
    assert len(components) == 2
    comps = sorted(
        components,
        key=lambda comp: ((comp["bbox"][0] + comp["bbox"][2]), (comp["bbox"][1] + comp["bbox"][3])),
    )
    return comps[0], comps[1]


def _red_run(row: np.ndarray, col: int) -> tuple[int, int]:
    c = int(col)
    assert int(row[c]) == RED
    left = c
    while left - 1 >= 0 and int(row[left - 1]) == RED:
        left -= 1
    right = c
    while right + 1 < row.shape[0] and int(row[right + 1]) == RED:
        right += 1
    return left, right


def _nearest_outside_col(current_col: int, left: int, right: int, width: int) -> int:
    candidates: list[tuple[int, int, int]] = []
    if left - 1 >= 0:
        side_col = left - 1
        candidates.append((abs(int(current_col) - side_col), 0, side_col))
    if right + 1 < width:
        side_col = right + 1
        candidates.append((abs(side_col - int(current_col)), 1, side_col))
    assert candidates
    return min(candidates, key=lambda item: (item[0], -item[1]))[2]


def _trace_route(grid: np.ndarray, start: Coord, target: Coord) -> tuple[list[Coord], list[dict], str]:
    _h, w = grid.shape
    sr, sc = map(int, start)
    tr, tc = map(int, target)
    assert sr < tr and sc < tc

    route: list[Coord] = []
    detours: list[dict] = []
    r, c = sr, sc

    while True:
        if r == tr or c == tc:
            break

        dc = 1 if tc > c else -1
        nr, nc = r + 1, c + dc

        if nr > tr:
            break
        if (nr, nc) == (tr, tc):
            break

        if 0 <= nc < w and int(grid[nr, nc]) == RED:
            left, right = _red_run(grid[nr], nc)
            side_col = _nearest_outside_col(c, left, right, w)
            step = 1 if side_col > c else -1
            for cc in range(c + step, side_col + step, step):
                route.append((r, cc))
            route.append((nr, side_col))
            detours.append(
                {
                    "row_before": r,
                    "obstacle_row": nr,
                    "current_col": c,
                    "left": left,
                    "right": right,
                    "side_col": side_col,
                }
            )
            r, c = nr, side_col
            continue

        route.append((nr, nc))
        r, c = nr, nc
        if r == tr or c == tc:
            break

    correction = "none"
    if r == tr and c != tc:
        correction = "horizontal"
        step = 1 if tc > c else -1
        for cc in range(c + step, tc, step):
            route.append((r, cc))
    elif c == tc and r != tr:
        correction = "vertical"
        for rr in range(r + 1, tr):
            route.append((rr, c))
    return route, detours, correction


def _paint_route(out: np.ndarray, clean: np.ndarray, color: int, route: list[Coord]) -> None:
    for r, c in route:
        val = int(out[r, c])
        if val in (BG, int(color)):
            out[r, c] = int(color)
            continue
        if int(clean[r, c]) == int(color):
            out[r, c] = int(color)
            continue
        raise AssertionError("route overlaps a foreign object")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each same-color rectangle pair with a diagonal path that detours around red horizontal blockers.

    Input structure:
    - The background is black.  There are 2-3 pairs of same-color solid rectangles, plus red horizontal obstacle segments.
    - Each color other than black, red, and grey appears as exactly two 4-connected solid rectangles.
    - The upper rectangle of a pair sits up-left of the lower rectangle so their facing corners define a down-right 45-degree connector.
    - Grey singleton noise pixels may be scattered anywhere on background cells.

    Rule:
    - Remove every grey noise pixel by turning it into background.
    - For each foreground color, find its upper rectangle and lower rectangle, then use the upper rectangle's bottom-right corner and the lower rectangle's top-left corner as connector endpoints.
    - Draw the connector on background cells along the down-right diagonal until the next diagonal step would hit a red obstacle pixel.
    - When blocked, stop on the previous row, move horizontally on that row to the nearer outside side of the red segment, with exact ties choosing the right side, step down into the obstacle row at that outside column, then continue diagonally from the new column toward the target.
    - Stop the resumed diagonal when it reaches the target row or target column; if it reaches the target row first, add a final horizontal correction to the target column.  Red obstacle pixels stay unchanged.

    Key insight:
    - The rectangles still define the desired diagonal endpoints, but the red segment changes only the connector's local routing: choose the obstacle side nearest to the current column, step one cell past that endpoint, then continue the diagonal from the shifted column.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    clean = g.copy()
    clean[clean == GREY] = BG
    out = clean.copy()

    for color in colors_present(clean, exclude={BG, RED, GREY}):
        components = _components_of_color(clean, int(color))
        assert len(components) == 2
        for comp in components:
            assert _is_solid_rectangle(clean, int(color), comp["bbox"])

        upper, lower = _sort_pair_components(components)
        ur0, uc0, ur1, uc1 = upper["bbox"]
        lr0, lc0, lr1, lc1 = lower["bbox"]
        start = (ur1, uc1)
        target = (lr0, lc0)
        assert start[0] < target[0] and start[1] < target[1]

        route, _detours, correction = _trace_route(clean, start, target)
        assert correction != "vertical", "valid inputs should not need a vertical final correction"
        _paint_route(out, clean, int(color), route)

    out[clean == RED] = RED
    return out


def _box_cells(box: BBox) -> set[Coord]:
    r0, c0, r1, c1 = box
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _expanded_box(box: BBox, h: int, w: int, pad: int = 1) -> BBox:
    r0, c0, r1, c1 = box
    return max(0, r0 - pad), max(0, c0 - pad), min(h - 1, r1 + pad), min(w - 1, c1 + pad)


def _boxes_clear(rect_guard: np.ndarray, *boxes: BBox) -> bool:
    for box in boxes:
        r0, c0, r1, c1 = box
        if np.any(rect_guard[r0 : r1 + 1, c0 : c1 + 1]):
            return False
    return True


def _mark_box(mask: np.ndarray, box: BBox, value: bool = True) -> None:
    r0, c0, r1, c1 = box
    mask[r0 : r1 + 1, c0 : c1 + 1] = value


def _place_rect(grid: np.ndarray, box: BBox, color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def _candidate_boxes(h: int, w: int) -> tuple[BBox, BBox, Coord, Coord] | None:
    h1 = random.choice(RECT_SIDE_CHOICES)
    w1 = random.choice(RECT_SIDE_CHOICES)
    h2 = random.choice(RECT_SIDE_CHOICES)
    w2 = random.choice(RECT_SIDE_CHOICES)
    delta = random.randint(DELTA_MIN, DELTA_MAX)

    max_ur = h - (h1 + delta + h2)
    max_uc = w - (w1 + delta + w2)
    if max_ur < 0 or max_uc < 0:
        return None

    ur = random.randint(0, max_ur)
    uc = random.randint(0, max_uc)
    upper = (ur, uc, ur + h1 - 1, uc + w1 - 1)
    start = (upper[2], upper[3])
    target = (start[0] + delta, start[1] + delta)
    lower = (target[0], target[1], target[0] + h2 - 1, target[1] + w2 - 1)
    return upper, lower, start, target


def _obstacle_choices(start: Coord, target: Coord, width: int) -> list[tuple[int, int, int]]:
    sr, sc = start
    tr, tc = target
    choices = []
    delta = tr - sr
    for step_idx in range(1, delta):
        obstacle_row = sr + step_idx
        hit_col = sc + step_idx
        for length in range(OBSTACLE_LEN_MIN, OBSTACLE_LEN_MAX + 1):
            for left in range(hit_col - length + 1, hit_col + 1):
                right = left + length - 1
                if left < 1 or right >= width - 1:
                    continue
                if left <= hit_col <= right:
                    choices.append((obstacle_row, left, right))
    random.shuffle(choices)
    return choices


def _route_cells_conflict(
    route: list[Coord],
    own_rects: set[Coord],
    occupied: np.ndarray,
    obstacle_cells: set[Coord],
) -> bool:
    for r, c in route:
        if (r, c) in own_rects or (r, c) in obstacle_cells:
            continue
        if occupied[r, c]:
            return True
    return False


def _sample_pair(
    h: int,
    w: int,
    color: int,
    blocked: bool,
    occupied: np.ndarray,
    rect_guard: np.ndarray,
    obstacle_grid: np.ndarray,
) -> dict | None:
    for _ in range(PAIR_ATTEMPTS):
        candidate = _candidate_boxes(h, w)
        if candidate is None:
            return None
        upper, lower, start, target = candidate
        if not _boxes_clear(rect_guard, _expanded_box(upper, h, w), _expanded_box(lower, h, w)):
            continue

        own_rects = _box_cells(upper) | _box_cells(lower)
        if any(occupied[r, c] for r, c in own_rects):
            continue

        if not blocked:
            route, detours, correction = _trace_route(obstacle_grid, start, target)
            if detours or correction == "vertical":
                continue
            if _route_cells_conflict(route, own_rects, occupied, set()):
                continue
            return {
                "color": int(color),
                "upper": upper,
                "lower": lower,
                "route": route,
                "obstacle": None,
                "rect_cells": own_rects,
            }

        for obstacle_row, left, right in _obstacle_choices(start, target, w):
            obstacle_cells = {(obstacle_row, c) for c in range(left, right + 1)}
            if any(occupied[r, c] for r, c in obstacle_cells):
                continue

            temp_grid = obstacle_grid.copy()
            temp_grid[obstacle_row, left : right + 1] = RED
            route, detours, correction = _trace_route(temp_grid, start, target)
            if len(detours) != 1 or correction == "vertical":
                continue
            detour = detours[0]
            if detour["obstacle_row"] != obstacle_row or detour["left"] != left or detour["right"] != right:
                continue
            if _route_cells_conflict(route, own_rects, occupied, obstacle_cells):
                continue
            return {
                "color": int(color),
                "upper": upper,
                "lower": lower,
                "route": route,
                "obstacle": (obstacle_row, left, right),
                "rect_cells": own_rects,
            }
    return None


def _add_noise(inp: np.ndarray, occupied: np.ndarray) -> None:
    bg_cells = [(r, c) for r in range(inp.shape[0]) for c in range(inp.shape[1]) if not occupied[r, c] and int(inp[r, c]) == BG]
    if not bg_cells:
        return
    noise_count = random.randint(0, min(NOISE_MAX, len(bg_cells)))
    for r, c in random.sample(bg_cells, noise_count):
        inp[r, c] = GREY


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    n_pairs = random.choices(PAIR_COUNT_CHOICES, weights=PAIR_COUNT_WEIGHTS, k=1)[0]
    blocked_count = 1 if n_pairs == 2 else random.choice([1, 2])
    blocked_flags = [True] * blocked_count + [False] * (n_pairs - blocked_count)
    random.shuffle(blocked_flags)
    colors = random.sample(PAIR_COLORS, n_pairs)

    out = np.full((h, w), BG, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    rect_guard = np.zeros((h, w), dtype=bool)
    obstacle_grid = np.full((h, w), BG, dtype=int)
    specs: list[dict] = []

    order = list(range(n_pairs))
    order.sort(key=lambda idx: (blocked_flags[idx],), reverse=True)
    for idx in order:
        spec = _sample_pair(h, w, colors[idx], blocked_flags[idx], occupied, rect_guard, obstacle_grid)
        if spec is None:
            return None
        specs.append(spec)

        _place_rect(out, spec["upper"], spec["color"])
        _place_rect(out, spec["lower"], spec["color"])
        if spec["obstacle"] is not None:
            rr, left, right = spec["obstacle"]
            out[rr, left : right + 1] = RED
            obstacle_grid[rr, left : right + 1] = RED
            occupied[rr, left : right + 1] = True
        for r, c in spec["route"]:
            if int(out[r, c]) in (BG, spec["color"]):
                out[r, c] = spec["color"]
                occupied[r, c] = True
            else:
                return None
        for r, c in spec["rect_cells"]:
            occupied[r, c] = True
        _mark_box(rect_guard, _expanded_box(spec["upper"], h, w))
        _mark_box(rect_guard, _expanded_box(spec["lower"], h, w))

    inp = out.copy()
    rect_mask = np.zeros((h, w), dtype=bool)
    for spec in specs:
        for r, c in spec["rect_cells"]:
            rect_mask[r, c] = True
    for spec in specs:
        for r, c in spec["route"]:
            if not rect_mask[r, c]:
                inp[r, c] = BG

    _add_noise(inp, occupied)
    expected = out.copy()
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
