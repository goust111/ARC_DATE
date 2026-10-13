import random
from collections import deque

import numpy as np


BG = 0
WALL = 5
PALETTE_CHOICES = [1, 2, 3, 4, 6, 7, 8, 9]
DIRS = ("top", "right", "bottom", "left")

GRID_H_RANGE = (18, 26)
GRID_W_RANGE = (18, 26)
ROOM_COUNT_RANGE = (4, 5)
ROOM_H_RANGE = (4, 7)
ROOM_W_RANGE = (4, 8)
PLACEMENT_RETRIES = 200
GEN_RETRIES = 300


def _mode(values: np.ndarray) -> int:
    vals = np.asarray(values, dtype=int).ravel()
    if vals.size == 0:
        raise ValueError("mode of empty array")
    return int(np.bincount(vals).argmax())


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _infer_wall_color(grid: np.ndarray) -> int:
    interior = grid[1:-1, 1:-1]
    vals = interior[interior != BG]
    assert vals.size > 0, "expected non-background room walls in the interior"
    return _mode(vals)


def _edge_color(line: np.ndarray, wall: int) -> int:
    vals = np.asarray(line, dtype=int)
    vals = vals[(vals != BG) & (vals != wall)]
    assert vals.size > 0, "each outer edge must contain a palette strip"
    return _mode(vals)


def _edge_palette(grid: np.ndarray, wall: int) -> dict[str, int]:
    palette = {
        "top": _edge_color(grid[0, :], wall),
        "right": _edge_color(grid[:, -1], wall),
        "bottom": _edge_color(grid[-1, :], wall),
        "left": _edge_color(grid[:, 0], wall),
    }
    assert len(set(palette.values())) == 4, "direction palette colors must be distinct"
    return palette


def _room_direction(patch: np.ndarray, wall: int) -> str | None:
    h, w = patch.shape
    if h < 3 or w < 3:
        return None
    if np.any(patch[1:-1, 1:-1] == wall):
        return None
    if np.any(patch[1:-1, 1:-1] != BG):
        return None

    expected = 2 * (h + w) - 5
    if int(np.count_nonzero(patch == wall)) != expected:
        return None

    side_gaps = {
        "top": np.flatnonzero(patch[0, :] != wall),
        "right": np.flatnonzero(patch[:, -1] != wall),
        "bottom": np.flatnonzero(patch[-1, :] != wall),
        "left": np.flatnonzero(patch[:, 0] != wall),
    }

    total_missing = sum(len(gaps) for gaps in side_gaps.values())
    if total_missing != 1:
        return None

    if len(side_gaps["top"]) == 1:
        gap = int(side_gaps["top"][0])
        if 0 < gap < w - 1 and int(patch[0, gap]) == BG:
            return "top"
    if len(side_gaps["right"]) == 1:
        gap = int(side_gaps["right"][0])
        if 0 < gap < h - 1 and int(patch[gap, -1]) == BG:
            return "right"
    if len(side_gaps["bottom"]) == 1:
        gap = int(side_gaps["bottom"][0])
        if 0 < gap < w - 1 and int(patch[-1, gap]) == BG:
            return "bottom"
    if len(side_gaps["left"]) == 1:
        gap = int(side_gaps["left"][0])
        if 0 < gap < h - 1 and int(patch[gap, 0]) == BG:
            return "left"
    return None


def _find_rooms(grid: np.ndarray, wall: int) -> list[tuple[int, int, int, int, str]]:
    rooms: list[tuple[int, int, int, int, str]] = []
    for comp in _components(grid == wall):
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        direction = _room_direction(patch, wall)
        if direction is None:
            continue
        rooms.append((r0, c0, r1, c1, direction))
    return rooms


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each rectangular room according to the direction of its one-cell wall gap.

    The outer border provides a direction→color palette:
    top edge, right edge, bottom edge, left edge.
    Every gray rectangular room has exactly one one-cell gap on one side.
    The room interior is filled with the color from the matching outer edge.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    wall = _infer_wall_color(g)
    palette = _edge_palette(g, wall)
    rooms = _find_rooms(g, wall)
    assert rooms, "expected at least one valid room"

    out = g.copy()
    for r0, c0, r1, c1, direction in rooms:
        out[r0 + 1 : r1, c0 + 1 : c1] = palette[direction]
    return out


def _paint_palette_bands(grid: np.ndarray, palette: dict[str, int]) -> None:
    grid[0, 1:-1] = palette["top"]
    grid[1:-1, -1] = palette["right"]
    grid[-1, 1:-1] = palette["bottom"]
    grid[1:-1, 0] = palette["left"]


def _rect_is_clear(
    rects: list[tuple[int, int, int, int]],
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    pad: int = 1,
) -> bool:
    for ar0, ac0, ar1, ac1 in rects:
        if not (r1 + pad < ar0 or ar1 + pad < r0 or c1 + pad < ac0 or ac1 + pad < c0):
            return False
    return True


def _draw_room(
    grid: np.ndarray,
    r0: int,
    c0: int,
    height: int,
    width: int,
    direction: str,
    rng: random.Random,
) -> None:
    r1 = r0 + height - 1
    c1 = c0 + width - 1
    grid[r0, c0 : c1 + 1] = WALL
    grid[r1, c0 : c1 + 1] = WALL
    grid[r0 : r1 + 1, c0] = WALL
    grid[r0 : r1 + 1, c1] = WALL

    if direction == "top":
        grid[r0, rng.randint(c0 + 1, c1 - 1)] = BG
    elif direction == "right":
        grid[rng.randint(r0 + 1, r1 - 1), c1] = BG
    elif direction == "bottom":
        grid[r1, rng.randint(c0 + 1, c1 - 1)] = BG
    else:
        grid[rng.randint(r0 + 1, r1 - 1), c0] = BG


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    room_count = rng.randint(*ROOM_COUNT_RANGE)
    min_h = GRID_H_RANGE[0] + max(0, room_count - 4)
    min_w = GRID_W_RANGE[0] + max(0, room_count - 4)
    h = rng.randint(min_h, GRID_H_RANGE[1])
    w = rng.randint(min_w, GRID_W_RANGE[1])

    grid = np.full((h, w), BG, dtype=int)
    palette_colors = rng.sample(PALETTE_CHOICES, 4)
    palette = dict(zip(DIRS, palette_colors))
    _paint_palette_bands(grid, palette)

    room_dirs = list(DIRS)
    while len(room_dirs) < room_count:
        room_dirs.append(rng.choice(DIRS))
    rng.shuffle(room_dirs)

    rects: list[tuple[int, int, int, int]] = []
    for direction in room_dirs:
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            height = rng.randint(*ROOM_H_RANGE)
            width = rng.randint(*ROOM_W_RANGE)
            max_r0 = h - height - 2
            max_c0 = w - width - 2
            if max_r0 < 2 or max_c0 < 2:
                return None
            r0 = rng.randint(2, max_r0)
            c0 = rng.randint(2, max_c0)
            r1 = r0 + height - 1
            c1 = c0 + width - 1
            if not _rect_is_clear(rects, r0, c0, r1, c1, pad=1):
                continue
            _draw_room(grid, r0, c0, height, width, direction, rng)
            rects.append((r0, c0, r1, c1))
            placed = True
            break
        if not placed:
            return None

    rooms = _find_rooms(grid, WALL)
    if len(rooms) != room_count:
        return None

    found_dirs = {direction for _, _, _, _, direction in rooms}
    if found_dirs != set(DIRS):
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None

    for r0, c0, r1, c1, direction in rooms:
        fill = palette[direction]
        if not np.all(out[r0 + 1 : r1, c0 + 1 : c1] == fill):
            return None
        gap_ok = {
            "top": np.count_nonzero(grid[r0, c0 : c1 + 1] == BG) == 1,
            "right": np.count_nonzero(grid[r0 : r1 + 1, c1] == BG) == 1,
            "bottom": np.count_nonzero(grid[r1, c0 : c1 + 1] == BG) == 1,
            "left": np.count_nonzero(grid[r0 : r1 + 1, c0] == BG) == 1,
        }[direction]
        if not gap_ok:
            return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted retries")
