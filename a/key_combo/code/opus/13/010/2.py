import random
from collections import deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREY = 5
MAROON = 9
CONTROL_COLORS = (BLUE, RED)

GEN_ATTEMPTS = 400
GRID_H_RANGE = (15, 20)
GRID_W_RANGE = (18, 24)
ROOM_COUNT_CHOICES = (2, 3)
ROOM_INNER_H_RANGE = (2, 4)
ROOM_INNER_W_RANGE = (2, 3)
ROOM_GAP_RANGE = (2, 4)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _zero_components(grid: np.ndarray) -> list[np.ndarray]:
    h, w = grid.shape
    outside = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if grid[r, c] == BG and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if grid[r, c] == BG and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == BG and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))

    seen = np.zeros((h, w), dtype=bool)
    regions: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if grid[r, c] != BG or outside[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and grid[nr, nc] == BG
                        and not outside[nr, nc]
                        and not seen[nr, nc]
                    ):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            regions.append(np.array(cells, dtype=int))
    return regions


def _drop_columns(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()
    out[:2, :] = BG

    assert np.all(g[0, :] == MAROON), "expected a full maroon top row"
    assert set(np.unique(g[1, :])).issubset(set(CONTROL_COLORS)), "expected blue/red control row"

    for c in range(w):
        below = np.flatnonzero(g[2:, c] == GREY)
        if len(below):
            obstacle_r = int(2 + below[0])
            bottom_r = obstacle_r - 1
        else:
            bottom_r = h - 1
        top_r = bottom_r - 1
        assert top_r >= 0, "falling 2x1 block must fit inside the grid"
        out[top_r, c] = MAROON
        out[bottom_r, c] = int(g[1, c])

    return out


def _region_color(grid: np.ndarray, region: np.ndarray) -> int:
    h, w = grid.shape
    colors = set()
    for r, c in region:
        for nr, nc in _neighbors4(int(r), int(c)):
            if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] in CONTROL_COLORS:
                if nr > 0 and grid[nr - 1, nc] == MAROON:
                    colors.add(int(grid[nr, nc]))
    assert len(colors) == 1, "each enclosed region must have a unique control color"
    return next(iter(colors))


def _paint_room(grid: np.ndarray, r0: int, c0: int, inner_h: int, inner_w: int, side: str):
    r1 = r0 + inner_h + 1
    c1 = c0 + inner_w + 1

    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY

    plug_c = c0 if side == "left" else c1
    grid[r0, plug_c] = BG
    grid[r0 + 1, plug_c] = BG

    interior = (slice(r0 + 1, r1), slice(c0 + 1, c1))
    return {"interior": interior, "plug_c": plug_c, "bbox": (r0, c0, r1, c1)}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop the top stripe column by column, then fill each newly sealed chamber by the matching control color.

    Input structure:
    - The background is black.
    - Row 0 is a full maroon stripe and row 1 is a full blue/red control stripe.
    - Lower grey obstacle segments form short rectilinear walls; after the stripe falls, some side gaps become plugged and create enclosed zero regions.

    Rule:
    - Erase the top two rows.
    - Treat each column of the stripe as a 2x1 falling block with maroon above and its control color below; drop it until the lower cell sits just above the first grey obstacle in that column, or on the bottom row if no obstacle exists.
    - On the landed grid, find every 4-connected zero region that cannot reach the outer border.
    - For each such region, inspect the adjacent control pixel(s) that sit directly under a landed maroon cell on that region's boundary; fill the whole region with that unique blue/red control color.

    Key insight:
    - The enclosed regions do not exist in the input: grey walls are deliberately left open, and only the gravity-settled stripe columns seal those gaps and simultaneously label each chamber's fill color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    landed = _drop_columns(g)
    out = landed.copy()
    for region in _zero_components(landed):
        color = _region_color(landed, region)
        out[region[:, 0], region[:, 1]] = color
    return out


def _choose_room_colors(room_count: int) -> list[int]:
    colors = [random.choice(CONTROL_COLORS) for _ in range(room_count)]
    if room_count >= 2 and len(set(colors)) == 1:
        colors[random.randrange(room_count)] = BLUE if colors[0] == RED else RED
    return colors


def _valid_room_positions(h: int, w: int, room_count: int):
    x = random.randint(1, 3)
    rooms = []
    for _ in range(room_count):
        inner_h = random.randint(*ROOM_INNER_H_RANGE)
        inner_w = random.randint(*ROOM_INNER_W_RANGE)
        outer_w = inner_w + 2
        r0_max = h - inner_h - 3
        if x + outer_w >= w - 1 or r0_max < 4:
            return None
        r0 = random.randint(4, r0_max)
        rooms.append((r0, x, inner_h, inner_w, random.choice(("left", "right"))))
        x += outer_w + random.randint(*ROOM_GAP_RANGE)
    return rooms


def _generate_once() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    room_count = random.choice(ROOM_COUNT_CHOICES)
    room_specs = _valid_room_positions(h, w, room_count)
    if room_specs is None:
        return None

    grid = np.full((h, w), BG, dtype=int)
    grid[0, :] = MAROON
    grid[1, :] = np.array([random.choice(CONTROL_COLORS) for _ in range(w)], dtype=int)

    room_colors = _choose_room_colors(room_count)
    rooms = []
    for color, (r0, c0, inner_h, inner_w, side) in zip(room_colors, room_specs):
        room = _paint_room(grid, r0, c0, inner_h, inner_w, side)
        grid[1, room["plug_c"]] = color
        room["color"] = color
        rooms.append(room)

    landed = _drop_columns(grid)
    regions = _zero_components(landed)
    if len(regions) != room_count:
        return None

    out = main(grid)
    changed = out != landed
    expected_area = 0
    for room in rooms:
        interior = room["interior"]
        fill = out[interior]
        if fill.size == 0 or not np.all(fill == room["color"]):
            return None
        expected_area += fill.size

    if int(np.count_nonzero(changed)) != expected_area:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_once()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for the gravity-chamber ARC task")
