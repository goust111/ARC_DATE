from __future__ import annotations

import random

import numpy as np


BG = 0
RIGHT_COLOR = 1
UP_COLOR = 2
DOWN_COLOR = 3
LEFT_COLOR = 4
OBJ = 5
OBSTACLE = 7

DIRS = {
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
}
DIR_COLOR = {
    "up": UP_COLOR,
    "right": RIGHT_COLOR,
    "down": DOWN_COLOR,
    "left": LEFT_COLOR,
}
FRAME_COLORS = {RIGHT_COLOR, UP_COLOR, DOWN_COLOR, LEFT_COLOR}
FRAME_BY_DIR = {
    "up": UP_COLOR,
    "right": RIGHT_COLOR,
    "down": DOWN_COLOR,
    "left": LEFT_COLOR,
}


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


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
            pts = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _frame_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(np.isin(grid, list(FRAME_COLORS)))
    assert len(coords) > 0, "expected a colored frame"
    return _bbox_coords(coords)


def _direction_distances(r: int, c: int, frame_bbox: tuple[int, int, int, int]) -> dict[str, int]:
    fr0, fc0, fr1, fc1 = frame_bbox
    return {
        "up": r - fr0,
        "down": fr1 - r,
        "left": c - fc0,
        "right": fc1 - c,
    }


def _notch_pattern(center: tuple[int, int], direction: str) -> dict[str, tuple[tuple[int, int], ...]]:
    r, c = center
    dr, dc = DIRS[direction]
    pr, pc = -dc, dr
    cross = (
        (r, c),
        (r + dr, c + dc),
        (r - dr, c - dc),
        (r + pr, c + pc),
        (r - pr, c - pc),
    )
    support = (
        (r - 2 * dr, c - 2 * dc),
        (r - dr + pr, c - dc + pc),
        (r - dr - pr, c - dc - pc),
        (r + 2 * pr, c + 2 * pc),
        (r - 2 * pr, c - 2 * pc),
    )
    open_side = (
        (r + 2 * dr, c + 2 * dc),
        (r + dr + pr, c + dc + pc),
        (r + dr - pr, c + dc - pc),
    )
    return {"cross": cross, "support": support, "open_side": open_side}


def _notch_directions(grid: np.ndarray, r: int, c: int) -> list[str]:
    out = []
    for direction in DIRS:
        pattern = _notch_pattern((r, c), direction)
        if not all(_in_bounds(grid.shape, rr, cc) and int(grid[rr, cc]) == OBJ for rr, cc in pattern["support"]):
            continue
        if not all(_in_bounds(grid.shape, rr, cc) and int(grid[rr, cc]) == BG for rr, cc in pattern["open_side"]):
            continue
        out.append(direction)
    return out


def _find_notches(grid: np.ndarray) -> list[dict]:
    found = []
    h, w = grid.shape
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            cross = _notch_pattern((r, c), "up")["cross"]
            if not all(int(grid[rr, cc]) == BG for rr, cc in cross):
                continue
            dirs = _notch_directions(grid, r, c)
            assert len(dirs) <= 1, "expected each notch to have a unique opening direction"
            if dirs:
                found.append({"center": (r, c), "direction": dirs[0]})
    return found


def _ray_cells_until_stop(
    grid: np.ndarray,
    center: tuple[int, int],
    direction: str,
    frame_bbox: tuple[int, int, int, int],
) -> list[tuple[int, int]]:
    fr0, fc0, fr1, fc1 = frame_bbox
    dr, dc = DIRS[direction]
    r, c = center
    out = []
    while True:
        r += dr
        c += dc
        if r <= fr0 or r >= fr1 or c <= fc0 or c >= fc1:
            break
        if int(grid[r, c]) in FRAME_COLORS or int(grid[r, c]) == OBJ or int(grid[r, c]) == OBSTACLE:
            break
        out.append((r, c))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Fill each plus-shaped notch with the color of its opening side and extend a ray toward that frame side until blocked.

    Input structure:
    - Black background contains one one-cell-thick rectangular frame whose top/right/bottom/left sides are red/blue/green/yellow.
    - Inside the frame is one 4-connected grey object whose boundary contains two or three black plus-shaped notches.
    - One to three orange singleton obstacle pixels lie in the frame interior outside the grey object.

    Rule:
    - Detect every five-cell black plus notch whose supported side is grey and whose open side points toward one frame side.
    - Recolor the five notch cells with the color of that facing frame side.
    - From the notch center, cast a one-cell-wide ray in the same direction using the same color, stopping before an obstacle, grey object pixel, or frame pixel.
    - Leave the frame, grey object, obstacles, and all other cells unchanged.

    Key insight:
    - The notch opening encodes both the ray direction and the ray color, while orange obstacles only truncate rays that actually pass through them.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frame_bbox = _frame_bbox(g)
    out = g.copy()
    for notch in _find_notches(g):
        color = FRAME_BY_DIR[notch["direction"]]
        pattern = _notch_pattern(notch["center"], notch["direction"])
        for r, c in pattern["cross"]:
            out[r, c] = color
        for r, c in _ray_cells_until_stop(g, notch["center"], notch["direction"], frame_bbox):
            out[r, c] = color
    return out


PATTERNS = (
    {
        "name": "vertical_pair",
        "mask": np.array(
            [
                [1, 0, 0, 0, 1],
                [1, 1, 0, 1, 1],
                [0, 1, 1, 1, 0],
                [1, 1, 0, 1, 1],
                [1, 0, 0, 0, 1],
            ],
            dtype=int,
        ),
        "notches": (
            {"direction": "up", "center": (0, 2)},
            {"direction": "down", "center": (4, 2)},
        ),
    },
    {
        "name": "horizontal_pair",
        "mask": np.array(
            [
                [1, 1, 0, 1, 1],
                [0, 1, 1, 1, 0],
                [0, 0, 1, 0, 0],
                [0, 1, 1, 1, 0],
                [1, 1, 0, 1, 1],
            ],
            dtype=int,
        ),
        "notches": (
            {"direction": "left", "center": (2, 0)},
            {"direction": "right", "center": (2, 4)},
        ),
    },
    {
        "name": "up_down_left",
        "mask": np.array(
            [
                [1, 1, 0, 0, 0, 1],
                [0, 1, 1, 0, 1, 1],
                [0, 0, 1, 1, 1, 0],
                [0, 1, 1, 0, 1, 1],
                [1, 1, 0, 0, 0, 1],
            ],
            dtype=int,
        ),
        "notches": (
            {"direction": "up", "center": (0, 3)},
            {"direction": "left", "center": (2, 0)},
            {"direction": "down", "center": (4, 3)},
        ),
    },
    {
        "name": "up_down_right",
        "mask": np.array(
            [
                [1, 0, 0, 0, 1, 1],
                [1, 1, 0, 1, 1, 0],
                [0, 1, 1, 1, 0, 0],
                [1, 1, 0, 1, 1, 0],
                [1, 0, 0, 0, 1, 1],
            ],
            dtype=int,
        ),
        "notches": (
            {"direction": "up", "center": (0, 2)},
            {"direction": "right", "center": (2, 5)},
            {"direction": "down", "center": (4, 2)},
        ),
    },
    {
        "name": "left_right_up",
        "mask": np.array(
            [
                [1, 0, 0, 0, 1],
                [1, 1, 0, 1, 1],
                [0, 1, 1, 1, 0],
                [0, 0, 1, 0, 0],
                [0, 1, 1, 1, 0],
                [1, 1, 0, 1, 1],
            ],
            dtype=int,
        ),
        "notches": (
            {"direction": "up", "center": (0, 2)},
            {"direction": "left", "center": (3, 0)},
            {"direction": "right", "center": (3, 4)},
        ),
    },
    {
        "name": "left_right_down",
        "mask": np.array(
            [
                [1, 1, 0, 1, 1],
                [0, 1, 1, 1, 0],
                [0, 0, 1, 0, 0],
                [0, 1, 1, 1, 0],
                [1, 1, 0, 1, 1],
                [1, 0, 0, 0, 1],
            ],
            dtype=int,
        ),
        "notches": (
            {"direction": "left", "center": (2, 0)},
            {"direction": "right", "center": (2, 4)},
            {"direction": "down", "center": (5, 2)},
        ),
    },
)


def _draw_frame(grid: np.ndarray, frame_bbox: tuple[int, int, int, int]) -> None:
    fr0, fc0, fr1, fc1 = frame_bbox
    grid[fr0, fc0 : fc1 + 1] = UP_COLOR
    grid[fr1, fc0 : fc1 + 1] = DOWN_COLOR
    grid[fr0 + 1 : fr1, fc0] = LEFT_COLOR
    grid[fr0 + 1 : fr1, fc1] = RIGHT_COLOR


def _place_object(
    grid: np.ndarray,
    mask: np.ndarray,
    top: int,
    left: int,
) -> set[tuple[int, int]]:
    cells = set()
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if int(mask[r, c]) == 1:
                grid[top + r, left + c] = OBJ
                cells.add((top + r, left + c))
    return cells


def _global_notches(pattern: dict, top: int, left: int) -> list[dict]:
    return [
        {
            "direction": notch["direction"],
            "center": (top + notch["center"][0], left + notch["center"][1]),
        }
        for notch in pattern["notches"]
    ]


def _global_crosses(pattern: dict, top: int, left: int) -> set[tuple[int, int]]:
    cells = set()
    for notch in _global_notches(pattern, top, left):
        cells.update(_notch_pattern(notch["center"], notch["direction"])["cross"])
    return cells


def _path_without_obstacles(
    object_cells: set[tuple[int, int]],
    center: tuple[int, int],
    direction: str,
    frame_bbox: tuple[int, int, int, int],
) -> list[tuple[int, int]]:
    fr0, fc0, fr1, fc1 = frame_bbox
    dr, dc = DIRS[direction]
    r, c = center
    cells = []
    while True:
        r += dr
        c += dc
        if r <= fr0 or r >= fr1 or c <= fc0 or c >= fc1:
            break
        if (r, c) in object_cells:
            return []
        cells.append((r, c))
    return cells


def _independent_expected(
    inp: np.ndarray,
    notches: list[dict],
    frame_bbox: tuple[int, int, int, int],
) -> np.ndarray:
    out = inp.copy()
    for notch in notches:
        color = DIR_COLOR[notch["direction"]]
        pattern = _notch_pattern(notch["center"], notch["direction"])
        for r, c in pattern["cross"]:
            out[r, c] = color
        dr, dc = DIRS[notch["direction"]]
        r, c = notch["center"]
        while True:
            r += dr
            c += dc
            if r <= frame_bbox[0] or r >= frame_bbox[2] or c <= frame_bbox[1] or c >= frame_bbox[3]:
                break
            if int(inp[r, c]) in FRAME_COLORS or int(inp[r, c]) == OBJ or int(inp[r, c]) == OBSTACLE:
                break
            out[r, c] = color
    return out


def _candidate_placements(
    frame_bbox: tuple[int, int, int, int],
    pattern: dict,
) -> list[dict]:
    fr0, fc0, fr1, fc1 = frame_bbox
    mask = pattern["mask"]
    h, w = mask.shape
    placements = []
    for top in range(fr0 + 1, fr1 - h + 1):
        for left in range(fc0 + 1, fc1 - w + 1):
            object_cells = {
                (top + r, left + c)
                for r in range(h)
                for c in range(w)
                if int(mask[r, c]) == 1
            }
            notches = _global_notches(pattern, top, left)
            crosses = _global_crosses(pattern, top, left)
            if any(not _in_bounds((fr1 + 1, fc1 + 1), r, c) for r, c in crosses):
                continue
            if any(r <= fr0 or r >= fr1 or c <= fc0 or c >= fc1 for r, c in crosses):
                continue

            nearest_ok = True
            paths = []
            for notch in notches:
                dists = _direction_distances(notch["center"][0], notch["center"][1], frame_bbox)
                if any(dists[notch["direction"]] >= dists[other] for other in DIRS if other != notch["direction"]):
                    nearest_ok = False
                    break
                path = _path_without_obstacles(object_cells, notch["center"], notch["direction"], frame_bbox)
                if not path:
                    nearest_ok = False
                    break
                paths.append(path)
            if not nearest_ok:
                continue

            union = set()
            disjoint = True
            for path in paths:
                if union & set(path):
                    disjoint = False
                    break
                union.update(path)
            if not disjoint:
                continue
            if max(len(path) for path in paths) < 2:
                continue

            placements.append(
                {
                    "top": top,
                    "left": left,
                    "object_cells": object_cells,
                    "notches": notches,
                    "paths": paths,
                }
            )
    return placements


def _all_black_interior_cells(
    shape: tuple[int, int],
    frame_bbox: tuple[int, int, int, int],
    object_cells: set[tuple[int, int]],
    forbidden: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    fr0, fc0, fr1, fc1 = frame_bbox
    cells = []
    for r in range(fr0 + 1, fr1):
        for c in range(fc0 + 1, fc1):
            if (r, c) in object_cells or (r, c) in forbidden:
                continue
            cells.append((r, c))
    return cells


def _isolated_from_obstacles(cell: tuple[int, int], placed: set[tuple[int, int]]) -> bool:
    r, c = cell
    for rr, cc in placed:
        if max(abs(r - rr), abs(c - cc)) <= 1:
            return False
    return True


def _build_candidate() -> np.ndarray | None:
    h = random.randint(16, 24)
    w = random.randint(16, 24)
    grid = np.zeros((h, w), dtype=int)

    top_margin = random.randint(1, 3)
    left_margin = random.randint(1, 3)
    bottom_margin = random.randint(1, 3)
    right_margin = random.randint(1, 3)
    frame_bbox = (top_margin, left_margin, h - 1 - bottom_margin, w - 1 - right_margin)
    fr0, fc0, fr1, fc1 = frame_bbox
    if fr1 - fr0 + 1 < 12 or fc1 - fc0 + 1 < 12:
        return None

    pattern = random.choice(PATTERNS)
    placements = _candidate_placements(frame_bbox, pattern)
    if not placements:
        return None
    placement = random.choice(placements)

    _draw_frame(grid, frame_bbox)
    object_cells = _place_object(grid, pattern["mask"], placement["top"], placement["left"])
    notches = placement["notches"]
    crosses = _global_crosses(pattern, placement["top"], placement["left"])

    path_choices = [path for path in placement["paths"] if len(path) >= 2]
    if not path_choices:
        return None
    blocking_path = random.choice(path_choices)
    obstacles = {random.choice(blocking_path[1:])}

    interior_forbidden = object_cells | crosses | set().union(*placement["paths"])
    free_cells = _all_black_interior_cells(grid.shape, frame_bbox, object_cells, object_cells | crosses)
    random.shuffle(free_cells)
    total_obstacles = random.randint(1, 3)
    for cell in free_cells:
        if len(obstacles) >= total_obstacles:
            break
        if cell in obstacles or cell in interior_forbidden:
            continue
        if not _isolated_from_obstacles(cell, obstacles):
            continue
        obstacles.add(cell)

    if len(obstacles) != total_obstacles:
        return None

    for r, c in obstacles:
        grid[r, c] = OBSTACLE

    if not _semantic_check(grid, pattern, placement, frame_bbox, obstacles):
        return None
    return grid


def _semantic_check(
    inp: np.ndarray,
    pattern: dict,
    placement: dict,
    frame_bbox: tuple[int, int, int, int],
    obstacles: set[tuple[int, int]],
) -> bool:
    object_comps = _components(inp == OBJ)
    if len(object_comps) != 1:
        return False

    object_area = int(np.count_nonzero(inp == OBJ))
    if not (10 <= object_area <= 18):
        return False

    dirs = [notch["direction"] for notch in placement["notches"]]
    if len(set(dirs)) == 1:
        return False

    expected_notches = {(notch["center"], notch["direction"]) for notch in placement["notches"]}
    actual_notches = {(item["center"], item["direction"]) for item in _find_notches(inp)}
    if actual_notches != expected_notches:
        return False

    for notch in placement["notches"]:
        dists = _direction_distances(notch["center"][0], notch["center"][1], frame_bbox)
        if any(dists[notch["direction"]] >= dists[other] for other in DIRS if other != notch["direction"]):
            return False

    if not all(_isolated_from_obstacles(cell, obstacles - {cell}) for cell in obstacles):
        return False

    blocked = False
    for notch in placement["notches"]:
        path = _path_without_obstacles(
            placement["object_cells"],
            notch["center"],
            notch["direction"],
            frame_bbox,
        )
        if any(cell in obstacles for cell in path[1:]):
            blocked = True
            break
    if not blocked:
        return False

    expected = _independent_expected(inp, placement["notches"], frame_bbox)
    try:
        actual = main(inp)
    except Exception:
        return False
    return np.array_equal(actual, expected)


def generate_input() -> np.ndarray:
    for _ in range(500):
        candidate = _build_candidate()
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid puzzle instance")
