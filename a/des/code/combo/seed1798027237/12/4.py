import random
from collections import deque

import numpy as np


FRAME_COLOR = 1
ROOM_COUNT_RANGE = (2, 5)
GRID_MIN = 12
GRID_MAX = 30
SIDE_CHOICES = (5, 7, 9)
GAP_RANGE = (1, 4)
MAX_GENERATION_ATTEMPTS = 300
PATTERN_ATTEMPTS = 80
LAYOUT_ATTEMPTS = 80
COLOR_POOL = [0, 2, 3, 4, 5, 6, 7, 8, 9]

ROTATIONS_TO_TOP = {
    "top": 0,
    "right": 1,
    "bottom": 2,
    "left": 3,
}
GAP_CELLS = {
    "top": lambda side: (0, side // 2),
    "right": lambda side: (side // 2, side - 1),
    "bottom": lambda side: (side - 1, side // 2),
    "left": lambda side: (side // 2, 0),
}


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            parts.append(np.asarray(coords, dtype=int))
    return parts


def _parse_room(grid: np.ndarray, bg: int, coords: np.ndarray) -> dict:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    side = int(r1 - r0 + 1)
    assert side == int(c1 - c0 + 1), "frame bbox must be square"
    assert side >= 5 and side % 2 == 1, "frame side must be odd and >= 5"

    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    ring = np.zeros((side, side), dtype=bool)
    ring[0, :] = True
    ring[-1, :] = True
    ring[:, 0] = True
    ring[:, -1] = True

    blue = patch == FRAME_COLOR
    assert not np.any(blue & ~ring), "blue pixels may only lie on the border"

    missing = np.argwhere(ring & ~blue)
    assert missing.shape[0] == 1, "frame must have exactly one border gap"
    gap_r, gap_c = map(int, missing[0])

    side_name = None
    for name, fn in GAP_CELLS.items():
        if (gap_r, gap_c) == fn(side):
            side_name = name
            break
    assert side_name is not None, "gap must be centered on one side"
    assert int(patch[gap_r, gap_c]) == bg, "gap cell must be background"

    interior = patch[1:-1, 1:-1].copy()
    assert not np.any(interior == FRAME_COLOR), "interior must not contain the frame color"

    return {
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "gap_side": side_name,
        "interior": interior,
    }


def _extract_rooms(grid: np.ndarray) -> tuple[int, list[dict]]:
    bg = _most_common_color(grid)
    parts = _components(grid == FRAME_COLOR)
    assert parts, "no blue frame found"
    rooms = [_parse_room(grid, bg, coords) for coords in parts]
    shapes = {room["interior"].shape for room in rooms}
    assert len(shapes) == 1, "all rooms must share one interior size"
    rooms.sort(key=lambda room: (room["bbox"][0], room["bbox"][1]))
    return bg, rooms


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    bg, rooms = _extract_rooms(g)
    inner_h, inner_w = rooms[0]["interior"].shape
    assert inner_h == inner_w

    out = np.full((inner_h, len(rooms) * inner_w + len(rooms) - 1), bg, dtype=int)
    for i, room in enumerate(rooms):
        tile = np.rot90(room["interior"], ROTATIONS_TO_TOP[room["gap_side"]]).copy()
        left = i * (inner_w + 1)
        out[:, left : left + inner_w] = tile
    return out


def _positive_composition(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(1, total), parts - 1))
    values = []
    prev = 0
    for cut in cuts + [total]:
        values.append(cut - prev)
        prev = cut
    return values


def _non_symmetric(tile: np.ndarray) -> bool:
    return all(not np.array_equal(tile, np.rot90(tile, k)) for k in (1, 2, 3))


def _grow_connected_mask(size: int, target: int) -> np.ndarray | None:
    mask = np.zeros((size, size), dtype=bool)
    start = (random.randrange(size), random.randrange(size))
    mask[start] = True
    frontier = set()

    def add_frontier(cell: tuple[int, int]) -> None:
        r, c = cell
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < size and 0 <= nc < size and not mask[nr, nc]:
                frontier.add((nr, nc))

    add_frontier(start)
    while int(mask.sum()) < target:
        if not frontier:
            return None
        cell = random.choice(tuple(frontier))
        frontier.discard(cell)
        if mask[cell]:
            continue
        mask[cell] = True
        add_frontier(cell)
    return mask


def _make_canonical_tile(inner: int, bg: int) -> np.ndarray | None:
    palette = [color for color in COLOR_POOL if color != bg]
    max_cells = max(3, int(round(inner * inner * 0.55)))
    min_cells = min(max_cells, max(3, inner))

    for _ in range(PATTERN_ATTEMPTS):
        target = random.randint(min_cells, max_cells)
        mask = _grow_connected_mask(inner, target)
        if mask is None:
            continue

        colors = random.sample(palette, random.choice((1, 2)))
        tile = np.full((inner, inner), bg, dtype=int)
        coords = [tuple(map(int, cell)) for cell in np.argwhere(mask)]
        random.shuffle(coords)
        for r, c in coords:
            tile[r, c] = colors[0]
        if len(colors) == 2:
            accent_count = random.randint(1, max(1, len(coords) // 3))
            for r, c in coords[:accent_count]:
                tile[r, c] = colors[1]

        if _non_symmetric(tile):
            return tile
    return None


def _choose_side(room_count: int) -> int:
    feasible = [
        side
        for side in SIDE_CHOICES
        if room_count * (side - 2) + room_count - 1 <= GRID_MAX
    ]
    return random.choice(feasible)


def _choose_gap_sides(room_count: int) -> list[str]:
    base = ["top", "right", "bottom", "left"]
    random.shuffle(base)
    picked = base[: min(room_count, 4)]
    while len(picked) < room_count:
        picked.append(random.choice(base))
    random.shuffle(picked)
    return picked


def _arrange_rooms(room_count: int, side: int) -> tuple[int, int, list[tuple[int, int]]] | None:
    for _ in range(LAYOUT_ATTEMPTS):
        row_count = random.randint(1, min(3, room_count))
        per_row = _positive_composition(room_count, row_count)
        row_widths = []
        h_gaps = []
        for count in per_row:
            gaps = [random.randint(*GAP_RANGE) for _ in range(max(0, count - 1))]
            h_gaps.append(gaps)
            row_widths.append(count * side + sum(gaps))
        v_gaps = [random.randint(*GAP_RANGE) for _ in range(max(0, row_count - 1))]

        min_h = row_count * side + sum(v_gaps)
        min_w = max(row_widths)
        if min_h > GRID_MAX or min_w > GRID_MAX:
            continue

        h = random.randint(max(GRID_MIN, min_h), GRID_MAX)
        w = random.randint(max(GRID_MIN, min_w), GRID_MAX)
        top = random.randint(0, h - min_h)

        positions = []
        cur_r = top
        for row_idx, count in enumerate(per_row):
            left = random.randint(0, w - row_widths[row_idx])
            cur_c = left
            for col_idx in range(count):
                positions.append((cur_r, cur_c))
                if col_idx < count - 1:
                    cur_c += side + h_gaps[row_idx][col_idx]
            if row_idx < row_count - 1:
                cur_r += side + v_gaps[row_idx]
        return h, w, positions
    return None


def _build_room_patch(canonical: np.ndarray, gap_side: str, bg: int) -> np.ndarray:
    inner = canonical.shape[0]
    side = inner + 2
    patch = np.full((side, side), bg, dtype=int)
    patch[0, :] = FRAME_COLOR
    patch[-1, :] = FRAME_COLOR
    patch[:, 0] = FRAME_COLOR
    patch[:, -1] = FRAME_COLOR

    gap_r, gap_c = GAP_CELLS[gap_side](side)
    patch[gap_r, gap_c] = bg

    undo = (-ROTATIONS_TO_TOP[gap_side]) % 4
    patch[1:-1, 1:-1] = np.rot90(canonical, undo)
    return patch


def _expected_output(bg: int, ordered_tiles: list[np.ndarray]) -> np.ndarray:
    inner = ordered_tiles[0].shape[0]
    out = np.full((inner, len(ordered_tiles) * inner + len(ordered_tiles) - 1), bg, dtype=int)
    for i, tile in enumerate(ordered_tiles):
        left = i * (inner + 1)
        out[:, left : left + inner] = tile
    return out


def _gen_attempt() -> np.ndarray | None:
    room_count = random.randint(*ROOM_COUNT_RANGE)
    side = _choose_side(room_count)
    arranged = _arrange_rooms(room_count, side)
    if arranged is None:
        return None
    h, w, positions = arranged

    bg = random.choice(COLOR_POOL)
    inner = side - 2
    gap_sides = _choose_gap_sides(room_count)

    canonical_tiles = []
    for _ in range(room_count):
        tile = _make_canonical_tile(inner, bg)
        if tile is None:
            return None
        canonical_tiles.append(tile)

    grid = np.full((h, w), bg, dtype=int)
    rooms = []
    for (r, c), gap_side, canonical in zip(positions, gap_sides, canonical_tiles):
        patch = _build_room_patch(canonical, gap_side, bg)
        grid[r : r + side, c : c + side] = patch
        rooms.append({"pos": (r, c), "canonical": canonical})

    if np.count_nonzero(grid != bg) >= grid.size // 2:
        return None

    ordered_tiles = [room["canonical"] for room in sorted(rooms, key=lambda room: room["pos"])]
    expected = _expected_output(bg, ordered_tiles)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
