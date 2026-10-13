import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
ORANGE = 7

ORTHO_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_components(grid: np.ndarray, color: int) -> list[dict]:
    mask = np.asarray(grid == color, dtype=bool)
    seen = np.zeros_like(mask, dtype=bool)
    components = []
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in ORTHO_DIRS:
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds(grid.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            coords.sort()
            components.append(
                {
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "area": len(coords),
                }
            )
    return components


def _component_mask(component: dict) -> np.ndarray:
    r0, c0, r1, c1 = component["bbox"]
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in component["coords"]:
        mask[r - r0, c - c0] = True
    return mask


def _symmetry_kind(component: dict) -> str:
    mask = _component_mask(component)
    horizontal = bool(np.array_equal(mask, np.flipud(mask)))
    vertical = bool(np.array_equal(mask, np.fliplr(mask)))
    if horizontal == vertical:
        raise AssertionError("each obstacle must have exactly one reflection symmetry axis")
    return "H" if horizontal else "V"


def _start_and_direction(grid: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    starts = np.argwhere(grid == RED)
    if starts.shape != (1, 2):
        raise AssertionError("expected exactly one red border marker")
    r, c = map(int, starts[0])
    h, w = grid.shape
    if r == 0:
        return (r, c), (1, 0)
    if r == h - 1:
        return (r, c), (-1, 0)
    if c == 0:
        return (r, c), (0, 1)
    if c == w - 1:
        return (r, c), (0, -1)
    raise AssertionError("red marker must be on the border")


def _left_of(direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = direction
    return -dc, dr


def _right_of(direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = direction
    return dc, -dr


def _scan_hit(
    grid: np.ndarray,
    start: tuple[int, int],
    direction: tuple[int, int],
    target_color: int,
) -> tuple[int, int] | None:
    r, c = start
    dr, dc = direction
    rr, cc = r + dr, c + dc
    while _in_bounds(grid.shape, rr, cc):
        if int(grid[rr, cc]) == target_color:
            return rr, cc
        rr += dr
        cc += dc
    return None


def _paint_segment(
    out: np.ndarray,
    a: tuple[int, int],
    b: tuple[int, int],
    color: int,
) -> None:
    ar, ac = a
    br, bc = b
    if ar == br:
        step = 1 if bc >= ac else -1
        for cc in range(ac, bc + step, step):
            out[ar, cc] = color
    elif ac == bc:
        step = 1 if br >= ar else -1
        for rr in range(ar, br + step, step):
            out[rr, ac] = color
    else:
        raise AssertionError("path segments must be axis-aligned")


def _segment_is_clear(grid: np.ndarray, a: tuple[int, int], b: tuple[int, int]) -> bool:
    ar, ac = a
    br, bc = b
    if ar == br:
        step = 1 if bc >= ac else -1
        return all(int(grid[ar, cc]) == BLACK for cc in range(ac, bc + step, step))
    if ac == bc:
        step = 1 if br >= ar else -1
        return all(int(grid[rr, ac]) == BLACK for rr in range(ar, br + step, step))
    return False


def _detour_points(
    component: dict,
    hit: tuple[int, int],
    direction: tuple[int, int],
    kind: str,
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], int]:
    dr, dc = direction
    turn_start = (hit[0] - dr, hit[1] - dc)
    lateral = _left_of(direction) if kind == "H" else _right_of(direction)
    detour_color = BLUE if kind == "H" else GREEN
    side_entry = (turn_start[0] + lateral[0], turn_start[1] + lateral[1])

    r0, c0, r1, c1 = component["bbox"]
    if dr == 1:
        side_exit = (r1 + 1, side_entry[1])
    elif dr == -1:
        side_exit = (r0 - 1, side_entry[1])
    elif dc == 1:
        side_exit = (side_entry[0], c1 + 1)
    else:
        side_exit = (side_entry[0], c0 - 1)
    rejoin = (side_exit[0] - lateral[0], side_exit[1] - lateral[1])
    return turn_start, side_entry, rejoin, detour_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw a straight red border path that detours around orange obstacles based on their symmetry.

    Input structure:
    - The grid has a black background, one red marker pixel on the border, and 2-4 orange
      4-connected obstacle shapes inside the canvas.
    - Every orange obstacle has exactly one reflection symmetry type: horizontal or vertical.
    - Each obstacle intersects the marker's straight inward channel, and there is free space on
      the needed detour side.

    Rule:
    - Start from the red border marker and move straight inward, coloring background cells red.
    - Whenever the path first hits an orange obstacle, determine that obstacle's symmetry type.
    - Horizontally symmetric obstacles force a left detour colored blue; vertically symmetric
      obstacles force a right detour colored green.
    - The detour begins one cell before the hit, shifts sideways by one cell, passes just beyond
      the obstacle's far edge, shifts back to the original channel, and then the red straight path resumes.
    - Orange obstacle pixels stay unchanged, and the path stops when it reaches the opposite border.

    Key insight:
    - Obstacle color alone is uninformative; the visible reflection symmetry of each orange shape
      simultaneously determines both the detour side and the detour color.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    start, direction = _start_and_direction(g)
    components = _find_components(g, ORANGE)
    if not 2 <= len(components) <= 4:
        raise AssertionError("expected 2-4 orange obstacles")

    component_at = {}
    for component in components:
        if not 3 <= component["area"] <= 5:
            raise AssertionError("obstacle area must be 3-5")
        _symmetry_kind(component)
        for cell in component["coords"]:
            component_at[cell] = component

    pos = start
    dr, dc = direction
    while True:
        hit = _scan_hit(g, pos, direction, ORANGE)
        if hit is None:
            end = pos
            rr, cc = pos[0] + dr, pos[1] + dc
            while _in_bounds(g.shape, rr, cc):
                end = (rr, cc)
                rr += dr
                cc += dc
            if end != pos:
                _paint_segment(out, (pos[0] + dr, pos[1] + dc), end, RED)
            break

        component = component_at[hit]
        kind = _symmetry_kind(component)
        turn_start, side_entry, rejoin, detour_color = _detour_points(component, hit, direction, kind)

        if turn_start != pos:
            _paint_segment(out, (pos[0] + dr, pos[1] + dc), turn_start, RED)

        if not _in_bounds(g.shape, *side_entry) or not _in_bounds(g.shape, *rejoin):
            raise AssertionError("detour stepped out of bounds")
        if int(g[side_entry]) != BLACK or int(g[rejoin]) != BLACK:
            raise AssertionError("detour endpoints must be background")

        lateral = _left_of(direction) if kind == "H" else _right_of(direction)
        side_exit = (rejoin[0] + lateral[0], rejoin[1] + lateral[1])
        if not _segment_is_clear(g, side_entry, side_exit):
            raise AssertionError("detour corridor must stay on background")
        if not _segment_is_clear(g, side_exit, rejoin):
            raise AssertionError("detour corridor must stay on background")

        _paint_segment(out, side_entry, side_exit, detour_color)
        _paint_segment(out, side_exit, rejoin, detour_color)
        pos = rejoin

    return out


H_SHAPES = [
    {
        "kind": "H",
        "lane_offset": 0,
        "coords": [(0, 1), (1, 0), (1, 1), (2, 1)],
        "shape": (3, 2),
    },
    {
        "kind": "H",
        "lane_offset": 1,
        "coords": [(0, 1), (0, 2), (1, 1), (2, 1), (2, 2)],
        "shape": (3, 3),
    },
    {
        "kind": "H",
        "lane_offset": 0,
        "coords": [(0, 0), (0, 1), (1, 1), (2, 0), (2, 1)],
        "shape": (3, 2),
    },
]

V_SHAPES = [
    {
        "kind": "V",
        "lane_offset": 2,
        "coords": [(0, 0), (0, 1), (0, 2), (1, 1)],
        "shape": (2, 3),
    },
    {
        "kind": "V",
        "lane_offset": 2,
        "coords": [(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)],
        "shape": (3, 3),
    },
    {
        "kind": "V",
        "lane_offset": 2,
        "coords": [(0, 1), (1, 0), (1, 1), (1, 2)],
        "shape": (2, 3),
    },
]


def _rotated_grid(grid: np.ndarray, turns: int) -> np.ndarray:
    return np.rot90(grid, turns).copy()


def _orient_grid(grid: np.ndarray) -> np.ndarray:
    choice = random.choice(["down", "up", "right", "left"])
    if choice == "down":
        return np.fliplr(grid).copy()
    if choice == "up":
        return np.flipud(grid).copy()
    if choice == "right":
        return _rotated_grid(grid, 1)
    return _rotated_grid(grid, -1)


def _build_canonical() -> np.ndarray | None:
    n_obstacles = random.choices([2, 3, 4], weights=[3, 4, 2], k=1)[0]
    labels = ["H", "V"] + [random.choice(["H", "V"]) for _ in range(n_obstacles - 2)]
    random.shuffle(labels)

    shapes = [random.choice(H_SHAPES if label == "H" else V_SHAPES) for label in labels]
    lead_gap = random.randint(2, 3)
    between_gaps = [random.randint(1, 2) for _ in range(n_obstacles - 1)]
    tail_gap = random.randint(1, 3)
    total_height = 1 + lead_gap + sum(spec["shape"][0] for spec in shapes) + sum(between_gaps) + tail_gap
    if not 13 <= total_height <= 22:
        return None

    height = total_height + random.randint(0, min(2, 22 - total_height))
    width = random.randint(13, 18)
    lane = random.randint(2, width - 3)
    grid = np.full((height, width), BLACK, dtype=int)
    grid[0, lane] = RED

    top = lead_gap
    for index, spec in enumerate(shapes):
        h, w = spec["shape"]
        left = lane - spec["lane_offset"]
        if left < 0 or left + w > width:
            return None
        for dr, dc in spec["coords"]:
            rr, cc = top + dr, left + dc
            if grid[rr, cc] != BLACK:
                return None
            grid[rr, cc] = ORANGE
        top += h
        if index < len(shapes) - 1:
            top += between_gaps[index]

    return _orient_grid(grid)


def _validate_generated(grid: np.ndarray) -> bool:
    try:
        start, direction = _start_and_direction(grid)
        components = _find_components(grid, ORANGE)
        if not 2 <= len(components) <= 4:
            return False
        kinds = [_symmetry_kind(component) for component in components]
        if "H" not in kinds or "V" not in kinds:
            return False

        sr, sc = start
        dr, dc = direction
        if dr:
            if any(not any(c == sc for _, c in component["coords"]) for component in components):
                return False
            ordered = sorted(components, key=lambda comp: comp["bbox"][0] if dr > 0 else -comp["bbox"][2])
        else:
            if any(not any(r == sr for r, _ in component["coords"]) for component in components):
                return False
            ordered = sorted(components, key=lambda comp: comp["bbox"][1] if dc > 0 else -comp["bbox"][3])

        pos = start
        for component in ordered:
            hit = _scan_hit(grid, pos, direction, ORANGE)
            if hit is None or hit not in component["coords"]:
                return False
            kind = _symmetry_kind(component)
            turn_start, side_entry, rejoin, _ = _detour_points(component, hit, direction, kind)
            if turn_start == pos:
                return False
            if not _in_bounds(grid.shape, *side_entry) or not _in_bounds(grid.shape, *rejoin):
                return False
            if int(grid[side_entry]) != BLACK or int(grid[rejoin]) != BLACK:
                return False
            lateral = _left_of(direction) if kind == "H" else _right_of(direction)
            side_exit = (rejoin[0] + lateral[0], rejoin[1] + lateral[1])
            if not _in_bounds(grid.shape, *side_exit):
                return False
            if not _segment_is_clear(grid, side_entry, side_exit):
                return False
            if not _segment_is_clear(grid, side_exit, rejoin):
                return False
            pos = rejoin

        out = main(grid)
        if not np.array_equal(out[grid == ORANGE], grid[grid == ORANGE]):
            return False
        if int((out == BLUE).sum()) == 0 or int((out == GREEN).sum()) == 0:
            return False
        if np.array_equal(out, grid):
            return False
        return True
    except Exception:
        return False


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _build_canonical()
        if grid is not None and _validate_generated(grid):
            return grid
    raise RuntimeError("failed to generate a valid input")
