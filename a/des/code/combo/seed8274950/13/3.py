import random
from collections import defaultdict

import numpy as np


ORTHO_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MODES = ("vertical_only", "horizontal_only", "mixed")
ROOM_SIZE_RANGE = (3, 6)
MARGIN_RANGE = (1, 3)
MAX_GENERATION_ATTEMPTS = 400


def _border_background(grid: np.ndarray) -> int:
    border = np.concatenate(
        [
            grid[0, :],
            grid[-1, :],
            grid[1:-1, 0],
            grid[1:-1, -1],
        ]
    )
    values, counts = np.unique(border, return_counts=True)
    return int(values[np.argmax(counts)])


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
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in ORTHO_DIRS:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _is_room_component(grid: np.ndarray, coords: np.ndarray) -> bool:
    colors = {int(grid[r, c]) for r, c in coords}
    if len(colors) != 1:
        return False
    r0, c0, r1, c1 = _bbox(coords)
    return len(coords) == (r1 - r0 + 1) * (c1 - c0 + 1)


def _find_doors(grid: np.ndarray, bg: int, wall: int) -> list[dict]:
    h, w = grid.shape
    doors: list[dict] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if int(grid[r, c]) != bg:
                continue

            up = int(grid[r - 1, c])
            down = int(grid[r + 1, c])
            left = int(grid[r, c - 1])
            right = int(grid[r, c + 1])

            vertical = (
                up == wall
                and down == wall
                and left not in (bg, wall)
                and right not in (bg, wall)
            )
            horizontal = (
                left == wall
                and right == wall
                and up not in (bg, wall)
                and down not in (bg, wall)
            )

            if vertical == horizontal:
                continue

            if vertical:
                doors.append(
                    {
                        "pos": (r, c),
                        "rays": [((0, -1), right), ((0, 1), left)],
                    }
                )
            else:
                doors.append(
                    {
                        "pos": (r, c),
                        "rays": [((-1, 0), down), ((1, 0), up)],
                    }
                )
    return doors


def _infer_wall_color(grid: np.ndarray, bg: int) -> int:
    candidates: list[tuple[tuple[int, int], int]] = []
    for raw_color in np.unique(grid):
        wall = int(raw_color)
        if wall == bg:
            continue
        room_mask = (grid != bg) & (grid != wall)
        room_comps = _components(room_mask)
        if len(room_comps) < 2:
            continue
        if not all(_is_room_component(grid, comp) for comp in room_comps):
            continue
        doors = _find_doors(grid, bg, wall)
        if not doors:
            continue
        score = (len(doors), int(np.count_nonzero(grid == wall)))
        candidates.append((score, wall))

    assert candidates, "failed to infer wall color"
    candidates.sort(reverse=True)
    best_score = candidates[0][0]
    best = [wall for score, wall in candidates if score == best_score]
    assert len(best) == 1, "ambiguous wall color"
    return int(best[0])


def _cast_rays(grid: np.ndarray, bg: int, wall: int, doors: list[dict]) -> np.ndarray:
    h, w = grid.shape
    out = grid.copy()
    painted = np.zeros((h, w), dtype=bool)

    active: list[tuple[int, int, int, int, int]] = []
    for door in doors:
        r, c = door["pos"]
        for (dr, dc), color in door["rays"]:
            active.append((int(r), int(c), int(dr), int(dc), int(color)))

    while active:
        proposals: dict[tuple[int, int], list[int]] = defaultdict(list)
        for idx, (r, c, dr, dc, _color) in enumerate(active):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            cell = int(grid[nr, nc])
            if cell in (bg, wall) or painted[nr, nc]:
                continue
            proposals[(nr, nc)].append(idx)

        next_active: list[tuple[int, int, int, int, int]] = []
        for (nr, nc), ids in proposals.items():
            if len(ids) != 1:
                continue
            idx = ids[0]
            _r, _c, dr, dc, color = active[idx]
            out[nr, nc] = color
            painted[nr, nc] = True
            next_active.append((nr, nc, dr, dc, color))

        active = next_active

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find single-cell doors in one-cell-thick walls, then emit cross-colored rays
    perpendicular to each wall into the adjacent rooms.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    bg = _border_background(g)
    wall = _infer_wall_color(g, bg)
    doors = _find_doors(g, bg, wall)
    assert doors, "expected at least one door"

    return _cast_rays(g, bg, wall, doors)


def _sample_counts(mode: str) -> tuple[int, int]:
    if mode == "vertical_only":
        options = [(1, 2), (1, 3), (2, 2), (2, 3), (3, 2)]
    elif mode == "horizontal_only":
        options = [(2, 1), (3, 1), (2, 2), (2, 3), (3, 2)]
    else:
        options = [(2, 2), (2, 3), (3, 2)]
    return random.choice(options)


def _build_layout(
    room_rows: int, room_cols: int, room_heights: list[int], room_widths: list[int]
) -> tuple[list[int], list[int], dict[tuple[int, int], tuple[int, int, int, int]]]:
    row_walls = [0]
    for height in room_heights:
        row_walls.append(row_walls[-1] + height + 1)

    col_walls = [0]
    for width in room_widths:
        col_walls.append(col_walls[-1] + width + 1)

    rooms: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    for i in range(room_rows):
        for j in range(room_cols):
            rooms[(i, j)] = (
                row_walls[i] + 1,
                col_walls[j] + 1,
                row_walls[i + 1] - 1,
                col_walls[j + 1] - 1,
            )
    return row_walls, col_walls, rooms


def _fill_walls(
    grid: np.ndarray, wall: int, top: int, left: int, row_walls: list[int], col_walls: list[int]
) -> None:
    bottom = top + row_walls[-1]
    right = left + col_walls[-1]
    for rr in row_walls:
        grid[top + rr, left : right + 1] = wall
    for cc in col_walls:
        grid[top : bottom + 1, left + cc] = wall


def _place_rooms(
    grid: np.ndarray,
    rooms: dict[tuple[int, int], tuple[int, int, int, int]],
    top: int,
    left: int,
    room_colors: dict[tuple[int, int], int],
) -> None:
    for room, (r0, c0, r1, c1) in rooms.items():
        grid[top + r0 : top + r1 + 1, left + c0 : left + c1 + 1] = room_colors[room]


def _door_site_vertical(
    room_a: tuple[int, int],
    room_b: tuple[int, int],
    rooms: dict[tuple[int, int], tuple[int, int, int, int]],
    col_walls: list[int],
    kinds: dict[tuple[int, int], str | None],
    used_rows: dict[tuple[int, int], set[int]],
) -> tuple[int, int] | None:
    if kinds[room_a] not in (None, "horizontal") or kinds[room_b] not in (None, "horizontal"):
        return None

    i, j = room_a
    wall_col = col_walls[j + 1]
    rows = list(range(rooms[room_a][0], rooms[room_a][2] + 1))
    random.shuffle(rows)
    for row in rows:
        if row not in used_rows[room_a] and row not in used_rows[room_b]:
            return row, wall_col
    return None


def _door_site_horizontal(
    room_a: tuple[int, int],
    room_b: tuple[int, int],
    rooms: dict[tuple[int, int], tuple[int, int, int, int]],
    row_walls: list[int],
    kinds: dict[tuple[int, int], str | None],
    used_cols: dict[tuple[int, int], set[int]],
) -> tuple[int, int] | None:
    if kinds[room_a] not in (None, "vertical") or kinds[room_b] not in (None, "vertical"):
        return None

    i, j = room_a
    wall_row = row_walls[i + 1]
    cols = list(range(rooms[room_a][1], rooms[room_a][3] + 1))
    random.shuffle(cols)
    for col in cols:
        if col not in used_cols[room_a] and col not in used_cols[room_b]:
            return wall_row, col
    return None


def _gen_attempt() -> np.ndarray | None:
    mode = random.choice(MODES)
    room_rows, room_cols = _sample_counts(mode)
    room_count = room_rows * room_cols

    palette = random.sample(range(10), 2 + room_count)
    bg = palette[0]
    wall = palette[1]
    room_palette = palette[2:]

    room_heights = [random.randint(*ROOM_SIZE_RANGE) for _ in range(room_rows)]
    room_widths = [random.randint(*ROOM_SIZE_RANGE) for _ in range(room_cols)]
    row_walls, col_walls, rooms = _build_layout(room_rows, room_cols, room_heights, room_widths)

    margin_top = random.randint(*MARGIN_RANGE)
    margin_left = random.randint(*MARGIN_RANGE)
    margin_bottom = random.randint(*MARGIN_RANGE)
    margin_right = random.randint(*MARGIN_RANGE)
    height = row_walls[-1] + 1 + margin_top + margin_bottom
    width = col_walls[-1] + 1 + margin_left + margin_right

    grid = np.full((height, width), bg, dtype=int)
    _fill_walls(grid, wall, margin_top, margin_left, row_walls, col_walls)

    room_colors = {
        room: room_palette[idx]
        for idx, room in enumerate(sorted(rooms))
    }
    _place_rooms(grid, rooms, margin_top, margin_left, room_colors)

    kinds = {room: None for room in rooms}
    used_rows = {room: set() for room in rooms}
    used_cols = {room: set() for room in rooms}
    placed: list[tuple[str, int, int]] = []
    placed_segments: set[tuple[str, int, int]] = set()

    def place_vertical(i: int, j: int) -> bool:
        key = ("v", i, j)
        if key in placed_segments:
            return False
        room_a = (i, j)
        room_b = (i, j + 1)
        site = _door_site_vertical(room_a, room_b, rooms, col_walls, kinds, used_rows)
        if site is None:
            return False
        row, col = site
        grid[margin_top + row, margin_left + col] = bg
        kinds[room_a] = "horizontal"
        kinds[room_b] = "horizontal"
        used_rows[room_a].add(row)
        used_rows[room_b].add(row)
        placed.append(("v", row, col))
        placed_segments.add(key)
        return True

    def place_horizontal(i: int, j: int) -> bool:
        key = ("h", i, j)
        if key in placed_segments:
            return False
        room_a = (i, j)
        room_b = (i + 1, j)
        site = _door_site_horizontal(room_a, room_b, rooms, row_walls, kinds, used_cols)
        if site is None:
            return False
        row, col = site
        grid[margin_top + row, margin_left + col] = bg
        kinds[room_a] = "vertical"
        kinds[room_b] = "vertical"
        used_cols[room_a].add(col)
        used_cols[room_b].add(col)
        placed.append(("h", row, col))
        placed_segments.add(key)
        return True

    vertical_segments = [(i, j) for i in range(room_rows) for j in range(room_cols - 1)]
    horizontal_segments = [(i, j) for i in range(room_rows - 1) for j in range(room_cols)]
    random.shuffle(vertical_segments)
    random.shuffle(horizontal_segments)

    if mode in ("vertical_only", "mixed"):
        if not any(place_vertical(i, j) for i, j in vertical_segments):
            return None
    if mode in ("horizontal_only", "mixed"):
        if not any(place_horizontal(i, j) for i, j in horizontal_segments):
            return None

    remaining: list[tuple[str, int, int]] = []
    if mode != "horizontal_only":
        remaining.extend(("v", i, j) for i, j in vertical_segments)
    if mode != "vertical_only":
        remaining.extend(("h", i, j) for i, j in horizontal_segments)
    random.shuffle(remaining)

    for kind, i, j in remaining:
        if random.random() < 0.55:
            if kind == "v":
                place_vertical(i, j)
            else:
                place_horizontal(i, j)

    doors = _find_doors(grid, bg, wall)
    if len(doors) != len(placed):
        return None

    door_has_vertical = any(
        int(grid[r - 1, c]) == wall and int(grid[r + 1, c]) == wall
        for r, c, _rays in [(door["pos"][0], door["pos"][1], door["rays"]) for door in doors]
    )
    door_has_horizontal = any(
        int(grid[r, c - 1]) == wall and int(grid[r, c + 1]) == wall
        for r, c, _rays in [(door["pos"][0], door["pos"][1], door["rays"]) for door in doors]
    )

    if mode == "vertical_only" and door_has_horizontal:
        return None
    if mode == "horizontal_only" and door_has_vertical:
        return None
    if mode == "mixed" and (not door_has_vertical or not door_has_horizontal):
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
