from __future__ import annotations

import random
from collections import deque

import numpy as np


WALL_COLOR = 5
MARKER_COLOR = 9
PALETTE_COLOR_POOL = [1, 2, 3, 4, 6, 7, 8]
DIRS_CLOCKWISE = [(-1, 0), (0, 1), (1, 0), (0, -1)]


def _clockwise(direction: tuple[int, int]) -> tuple[int, int]:
    idx = DIRS_CLOCKWISE.index(direction)
    return DIRS_CLOCKWISE[(idx + 1) % 4]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q: deque[tuple[int, int]] = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _extract_palette(grid: np.ndarray, bg: int) -> list[int]:
    h, w = grid.shape
    candidates: list[tuple[int, int, int, list[int]]] = []
    for row in (0, h - 1):
        c = 0
        while c < w:
            if int(grid[row, c]) == bg:
                c += 1
                continue
            start = c
            values: list[int] = []
            while c < w and int(grid[row, c]) != bg:
                values.append(int(grid[row, c]))
                c += 1
            end = c - 1
            if (start == 0 or end == w - 1) and len(values) >= 2 and len(set(values)) == len(values):
                edge_bias = 1 if row == 0 else 0
                candidates.append((len(values), edge_bias, start, values))
    if not candidates:
        raise AssertionError("palette row not found")
    candidates.sort(reverse=True)
    return candidates[0][3]


def _border_coords(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    border = set()
    for c in range(c0, c1 + 1):
        border.add((r0, c))
        border.add((r1, c))
    for r in range(r0, r1 + 1):
        border.add((r, c0))
        border.add((r, c1))
    return border


def _find_rooms(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    rooms: list[dict[str, object]] = []
    for comp in _components(grid == WALL_COLOR):
        rs = [r for r, _ in comp]
        cs = [c for _, c in comp]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)
        if r1 - r0 + 1 < 5 or c1 - c0 + 1 < 5:
            continue
        border = _border_coords(r0, c0, r1, c1)
        comp_set = set(comp)
        if not comp_set.issubset(border):
            continue
        missing = list(border - comp_set)
        if len(missing) != 1:
            continue
        marker = missing[0]
        marker_color = int(grid[marker])
        if marker_color in (bg, WALL_COLOR):
            continue

        interior_non_bg = [
            (r, c)
            for r in range(r0 + 1, r1)
            for c in range(c0 + 1, c1)
            if int(grid[r, c]) != bg
        ]
        if len(interior_non_bg) != 1:
            continue
        seed = interior_non_bg[0]
        dr = seed[0] - marker[0]
        dc = seed[1] - marker[1]
        if (dr, dc) not in DIRS_CLOCKWISE:
            continue

        rooms.append(
            {
                "bbox": (r0, c0, r1, c1),
                "seed": seed,
                "marker": marker,
                "direction": (dr, dc),
            }
        )
    rooms.sort(key=lambda room: room["bbox"])  # deterministic traversal
    return rooms


def _trace_path(
    interior_shape: tuple[int, int],
    start: tuple[int, int],
    direction: tuple[int, int],
    palette: list[int],
) -> tuple[list[tuple[int, int, int]], int]:
    ih, iw = interior_shape
    r, c = start
    dr, dc = direction
    color_idx = 0
    visited: set[tuple[int, int]] = set()
    painted: list[tuple[int, int, int]] = []
    turns = 0

    while True:
        if (r, c) in visited:
            raise AssertionError("path should never revisit a painted cell")
        visited.add((r, c))
        painted.append((r, c, palette[color_idx]))

        nr, nc = r + dr, c + dc
        if 0 <= nr < ih and 0 <= nc < iw and (nr, nc) not in visited:
            r, c = nr, nc
            continue

        # The prompt only mentions walls, but without treating the already
        # painted trail as occupied the trajectory inside a rectangle would not
        # terminate. The generated tasks use this finite spiral interpretation.
        turns += 1
        color_idx = (color_idx + 1) % len(palette)
        dr, dc = _clockwise((dr, dc))
        nr, nc = r + dr, c + dc
        if not (0 <= nr < ih and 0 <= nc < iw) or (nr, nc) in visited:
            break
        r, c = nr, nc

    return painted, turns


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw a turning path inside each hollow rectangular room.

    Each room contains one seed adjacent to one marked wall cell. The wall mark
    determines the inward start direction. A horizontal border palette gives the
    cyclic segment colors. The path moves cell by cell, turns clockwise whenever
    it would hit the room wall or its already drawn trail, and terminates if the
    post-turn step is still blocked.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(np.bincount(g.ravel()).argmax())
    palette = _extract_palette(g, bg)
    rooms = _find_rooms(g, bg)

    out = g.copy()
    for room in rooms:
        r0, c0, r1, c1 = room["bbox"]
        seed_r, seed_c = room["seed"]
        start = (seed_r - (r0 + 1), seed_c - (c0 + 1))
        direction = room["direction"]
        painted, _turns = _trace_path((r1 - r0 - 1, c1 - c0 - 1), start, direction, palette)
        for pr, pc, color in painted:
            out[r0 + 1 + pr, c0 + 1 + pc] = color
    return out


def _boxes_separated(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    gap: int = 1,
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _seed_marker_direction(
    bbox: tuple[int, int, int, int],
    side: str,
    rng: random.Random,
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    if side == "top":
        col = rng.randint(c0 + 1, c1 - 1)
        return (r0 + 1, col), (r0, col), (1, 0)
    if side == "bottom":
        col = rng.randint(c0 + 1, c1 - 1)
        return (r1 - 1, col), (r1, col), (-1, 0)
    if side == "left":
        row = rng.randint(r0 + 1, r1 - 1)
        return (row, c0 + 1), (row, c0), (0, 1)
    row = rng.randint(r0 + 1, r1 - 1)
    return (row, c1 - 1), (row, c1), (0, -1)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    bg = 0
    h = rng.randint(20, 30)
    w = rng.randint(20, 30)
    grid = np.full((h, w), bg, dtype=int)

    palette_len = rng.randint(2, 5)
    palette = rng.sample(PALETTE_COLOR_POOL, palette_len)
    palette_row = 0 if rng.random() < 0.5 else h - 1
    palette_start = 0 if rng.random() < 0.5 else w - palette_len
    grid[palette_row, palette_start : palette_start + palette_len] = palette

    area = h * w
    max_rooms = 4 if area >= 650 else 3 if area >= 500 else 2
    target_rooms = rng.randint(1, max_rooms)
    size_hi = 10 if target_rooms <= 2 else 9

    rooms: list[dict[str, object]] = []
    bboxes: list[tuple[int, int, int, int]] = []
    for _ in range(200):
        if len(rooms) == target_rooms:
            break
        room_h = rng.randint(5, min(size_hi, h - 4))
        room_w = rng.randint(5, min(size_hi, w - 4))
        r0 = rng.randint(2, h - room_h - 2)
        c0 = rng.randint(2, w - room_w - 2)
        bbox = (r0, c0, r0 + room_h - 1, c0 + room_w - 1)
        if any(not _boxes_separated(bbox, other, gap=1) for other in bboxes):
            continue
        side = rng.choice(["top", "bottom", "left", "right"])
        seed, marker, direction = _seed_marker_direction(bbox, side, rng)
        trace, turns = _trace_path((room_h - 2, room_w - 2), (seed[0] - r0 - 1, seed[1] - c0 - 1), direction, palette)
        rooms.append(
            {
                "bbox": bbox,
                "seed": seed,
                "marker": marker,
                "turns": turns,
                "trace_len": len(trace),
            }
        )
        bboxes.append(bbox)

    if len(rooms) != target_rooms:
        return None

    total_trace = sum(int(room["trace_len"]) for room in rooms)
    max_turns = max(int(room["turns"]) for room in rooms)
    if total_trace < 12 or max_turns < 2:
        return None

    for room in rooms:
        r0, c0, r1, c1 = room["bbox"]
        grid[r0, c0 : c1 + 1] = WALL_COLOR
        grid[r1, c0 : c1 + 1] = WALL_COLOR
        grid[r0 : r1 + 1, c0] = WALL_COLOR
        grid[r0 : r1 + 1, c1] = WALL_COLOR
        mr, mc = room["marker"]
        sr, sc = room["seed"]
        grid[mr, mc] = MARKER_COLOR
        grid[sr, sc] = palette[0]

    counts = np.bincount(grid.ravel(), minlength=10)
    if counts[bg] <= max(counts[color] for color in range(len(counts)) if color != bg):
        return None

    if _extract_palette(grid, bg) != palette:
        return None
    found_rooms = _find_rooms(grid, bg)
    if len(found_rooms) != target_rooms:
        return None
    if np.array_equal(main(grid), grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(400):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for task 6")
