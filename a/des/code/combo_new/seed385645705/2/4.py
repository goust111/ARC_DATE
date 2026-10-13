from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
WALL = 1
LEGEND_COLORS = (2, 3, 4, 7)
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in DIR4:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r0, c0 in np.argwhere(mask):
        r0, c0 = int(r0), int(c0)
        if seen[r0, c0]:
            continue
        seen[r0, c0] = True
        q = deque([(r0, c0)])
        comp: list[tuple[int, int]] = []
        while q:
            r, c = q.popleft()
            comp.append((r, c))
            for rr, cc in _neighbors4(r, c, h, w):
                if mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    q.append((rr, cc))
        comps.append(comp)
    return comps


def _touches_edge(comp: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp)


def _legend_palette(g: np.ndarray) -> dict[int, int]:
    """Read the four non-blue color blocks by their visible left-to-right order."""
    h, w = g.shape
    mask = (g != BG) & (g != WALL)
    entries: list[tuple[int, int, int, int]] = []
    for comp in _components(mask):
        colors = {int(g[r, c]) for r, c in comp}
        if len(colors) != 1:
            continue
        rs = [r for r, _ in comp]
        cs = [c for _, c in comp]
        edge_rank = 0 if _touches_edge(comp, h, w) else 1
        entries.append((edge_rank, min(cs), min(rs), colors.pop()))

    edge_entries = [e for e in entries if e[0] == 0]
    if len(edge_entries) >= 4:
        entries = edge_entries
    entries.sort(key=lambda x: (x[1], x[2]))
    assert len(entries) >= 4, "expected four legend color blocks"
    return {i + 1: int(entries[i][3]) for i in range(4)}


def _door_mask(g: np.ndarray) -> np.ndarray:
    """A door is a black cell that exactly bridges two collinear blue wall cells."""
    h, w = g.shape
    blue = g == WALL
    doors = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if int(g[r, c]) != BG:
                continue
            horizontal = c > 0 and c + 1 < w and blue[r, c - 1] and blue[r, c + 1]
            vertical = r > 0 and r + 1 < h and blue[r - 1, c] and blue[r + 1, c]
            if horizontal or vertical:
                doors[r, c] = True
    return doors


def _region_door_count(region: list[tuple[int, int]], doors: np.ndarray) -> int:
    h, w = doors.shape
    adj: set[tuple[int, int]] = set()
    for r, c in region:
        for rr, cc in _neighbors4(r, c, h, w):
            if doors[rr, cc]:
                adj.add((rr, cc))
    return len(adj)


def _paint_layers(out: np.ndarray, region: list[tuple[int, int]], layers: int, color: int) -> None:
    h, w = out.shape
    region_mask = np.zeros((h, w), dtype=bool)
    for r, c in region:
        region_mask[r, c] = True

    dist = np.zeros((h, w), dtype=int)
    q: deque[tuple[int, int]] = deque()
    for r, c in region:
        if any(not region_mask[rr, cc] for rr, cc in _neighbors4(r, c, h, w)):
            dist[r, c] = 1
            q.append((r, c))

    while q:
        r, c = q.popleft()
        if dist[r, c] <= layers:
            out[r, c] = color
        if dist[r, c] >= layers:
            continue
        for rr, cc in _neighbors4(r, c, h, w):
            if region_mask[rr, cc] and dist[rr, cc] == 0:
                dist[rr, cc] = dist[r, c] + 1
                q.append((rr, cc))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each blue-walled room inward using the color and depth named by its door count.

    Input structure:
    - Black is the background and blue pixels form a connected boundary network that partitions several rooms.
    - Room interiors are black, while each room boundary has one to four single black gaps between collinear blue wall pixels.
    - Four small non-blue legend blocks sit on the outer margin from left to right and give the colors for counts 1, 2, 3, and 4.

    Rule:
    - Detect every one-pixel black door gap and temporarily close those gaps to recover the individual enclosed room interiors.
    - For each recovered room, count the distinct door cells touching its boundary and read that count's color from the legend.
    - Paint exactly that many inward erosion layers of the room with the legend color, leaving any deeper interior black and preserving blue walls and black door gaps.

    Key insight:
    The same visible door count is bound twice: it selects both the legend color and the number of inward layers to fill.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    palette = _legend_palette(g)
    doors = _door_mask(g)
    closed = g.copy()
    closed[doors] = WALL

    out = g.copy()
    h, w = g.shape
    for region in _components(closed == BG):
        if _touches_edge(region, h, w):
            continue
        n_doors = _region_door_count(region, doors)
        if n_doors in palette:
            _paint_layers(out, region, n_doors, palette[n_doors])
    return out


# --------------- generator ---------------

ROOM_GRID_CHOICES = ((2, 2), (2, 3), (3, 2))
ROOM_SIZE_RANGE = (5, 7)
WALL_THICKNESS_RANGE = (3, 4)
GENERATION_RETRIES = 300


def _draw_legend(grid: np.ndarray, block: int, rng: random.Random) -> None:
    legend_w = 4 * int(block) + 3
    start = rng.randint(0, grid.shape[1] - legend_w)
    c = start
    for color in LEGEND_COLORS:
        grid[0:block, c : c + block] = int(color)
        c += int(block) + 1


def _target_counts(rows: int, cols: int, rng: random.Random) -> dict[tuple[int, int], int]:
    rooms = [(i, j) for i in range(rows) for j in range(cols)]
    counts = [1, 2, 3, 4]
    while len(counts) < len(rooms):
        counts.append(rng.randint(1, 4))
    rng.shuffle(counts)
    rng.shuffle(rooms)
    return {room: int(count) for room, count in zip(rooms, counts)}


def _blue_is_connected(grid: np.ndarray) -> bool:
    comps = _components(grid == WALL)
    return len(comps) == 1


def _room_rectangles(
    top: int, left: int, room_h: list[int], room_w: list[int], wall_t: int
) -> dict[tuple[int, int], tuple[int, int, int, int]]:
    rects: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    r = int(top) + int(wall_t)
    for i, height in enumerate(room_h):
        c = int(left) + int(wall_t)
        for j, width in enumerate(room_w):
            rects[(i, j)] = (r, c, r + int(height) - 1, c + int(width) - 1)
            c += int(width) + int(wall_t)
        r += int(height) + int(wall_t)
    return rects


def _carve_rooms(grid: np.ndarray, rects: dict[tuple[int, int], tuple[int, int, int, int]]) -> None:
    for r0, c0, r1, c1 in rects.values():
        grid[r0 : r1 + 1, c0 : c1 + 1] = BG


def _choose_room_doors(
    rect: tuple[int, int, int, int], count: int, rng: random.Random
) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = rect
    sides = [
        [(r0 - 1, c) for c in range(c0 + 1, c1)],
        [(r1 + 1, c) for c in range(c0 + 1, c1)],
        [(r, c0 - 1) for r in range(r0 + 1, r1)],
        [(r, c1 + 1) for r in range(r0 + 1, r1)],
    ]
    assert all(sides), "room must be large enough for side-centered doors"
    chosen: set[tuple[int, int]] = set()
    for side in rng.sample(sides, int(count)):
        chosen.add(rng.choice(side))
    return chosen


def _rect_room_counts(
    grid: np.ndarray, rects: dict[tuple[int, int], tuple[int, int, int, int]]
):
    doors = _door_mask(grid)
    closed = grid.copy()
    closed[doors] = WALL
    h, w = grid.shape

    cell_to_count: dict[tuple[int, int], int] = {}
    regions = 0
    for region in _components(closed == BG):
        if _touches_edge(region, h, w):
            continue
        regions += 1
        count = _region_door_count(region, doors)
        for cell in region:
            cell_to_count[cell] = count

    counts: dict[tuple[int, int], int] = {}
    for room, rect in rects.items():
        r0, c0, _, _ = rect
        if (r0, c0) not in cell_to_count:
            return None
        counts[room] = cell_to_count[(r0, c0)]
    return regions, counts


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    rows, cols = rng.choice(ROOM_GRID_CHOICES)
    room_h = [rng.randint(*ROOM_SIZE_RANGE) for _ in range(rows)]
    room_w = [rng.randint(*ROOM_SIZE_RANGE) for _ in range(cols)]
    wall_t = rng.randint(*WALL_THICKNESS_RANGE)
    plan_h = sum(room_h) + wall_t * (rows + 1)
    plan_w = sum(room_w) + wall_t * (cols + 1)

    legend_block = rng.choice([1, 2])
    min_top = legend_block + 1
    max_top = min(4, 30 - plan_h - 1)
    if max_top < min_top:
        return None
    top = rng.randint(min_top, max_top)

    max_left = min(4, 30 - plan_w - 1)
    if max_left < 1:
        return None
    left = rng.randint(1, max_left)

    min_h = top + plan_h + 1
    min_w = max(left + plan_w + 1, 4 * legend_block + 5)
    if min_h > 30 or min_w > 30:
        return None
    height = rng.randint(min_h, 30)
    width = rng.randint(min_w, 30)

    grid = np.zeros((height, width), dtype=int)
    _draw_legend(grid, legend_block, rng)

    grid[top : top + plan_h, left : left + plan_w] = WALL
    rects = _room_rectangles(top, left, room_h, room_w, wall_t)
    _carve_rooms(grid, rects)
    targets = _target_counts(rows, cols, rng)

    door_cells: set[tuple[int, int]] = set()
    for room, count in targets.items():
        room_doors = _choose_room_doors(rects[room], count, rng)
        if door_cells & room_doors:
            return None
        door_cells |= room_doors
    for r, c in door_cells:
        grid[r, c] = BG

    detected = {tuple(map(int, rc)) for rc in np.argwhere(_door_mask(grid))}
    if detected != door_cells:
        return None
    if not _blue_is_connected(grid):
        return None

    counted = _rect_room_counts(grid, rects)
    if counted is None:
        return None
    regions, counts = counted
    if regions != rows * cols or counts != targets:
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None
    for color in LEGEND_COLORS:
        if not np.any((out == color) & (grid == BG)):
            return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GENERATION_RETRIES} attempts")
