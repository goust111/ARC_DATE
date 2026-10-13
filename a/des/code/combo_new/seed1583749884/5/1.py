import random

import numpy as np


BG = 0
WALL = 9
COLOR_POOL = tuple(range(1, 9))
GENERATION_ATTEMPTS = 300
ROOM_LAYOUTS = ((2, 2), (2, 3), (3, 2), (2, 4), (4, 2))
ROOM_SIZE_RANGE = (2, 5)
MARGIN_RANGE = (1, 2)

DIRECTION_STEPS = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
OPPOSITE_DIRECTION = {
    "top": "bottom",
    "bottom": "top",
    "left": "right",
    "right": "left",
}


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            components.append(cells)

    return components


def _line_info(
    grid: np.ndarray,
    bg: int,
    wall: int,
    bbox: tuple[int, int, int, int],
    direction: str,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    r0, r1, c0, c1 = bbox

    if direction == "top":
        line = grid[r0 - 1, c0 : c1 + 1]
        if np.any((line != wall) & (line != bg)):
            raise AssertionError("invalid top wall")
        gaps = np.flatnonzero(line == bg)
        if gaps.size == 0:
            return None
        if gaps.size != 1:
            raise AssertionError("expected a single top gap")
        gap = (r0 - 1, c0 + int(gaps[0]))
    elif direction == "bottom":
        line = grid[r1 + 1, c0 : c1 + 1]
        if np.any((line != wall) & (line != bg)):
            raise AssertionError("invalid bottom wall")
        gaps = np.flatnonzero(line == bg)
        if gaps.size == 0:
            return None
        if gaps.size != 1:
            raise AssertionError("expected a single bottom gap")
        gap = (r1 + 1, c0 + int(gaps[0]))
    elif direction == "left":
        line = grid[r0 : r1 + 1, c0 - 1]
        if np.any((line != wall) & (line != bg)):
            raise AssertionError("invalid left wall")
        gaps = np.flatnonzero(line == bg)
        if gaps.size == 0:
            return None
        if gaps.size != 1:
            raise AssertionError("expected a single left gap")
        gap = (r0 + int(gaps[0]), c0 - 1)
    else:
        line = grid[r0 : r1 + 1, c1 + 1]
        if np.any((line != wall) & (line != bg)):
            raise AssertionError("invalid right wall")
        gaps = np.flatnonzero(line == bg)
        if gaps.size == 0:
            return None
        if gaps.size != 1:
            raise AssertionError("expected a single right gap")
        gap = (r0 + int(gaps[0]), c1 + 1)

    dr, dc = DIRECTION_STEPS[direction]
    target = (gap[0] + dr, gap[1] + dc)
    return gap, target


def _extract_rooms(grid: np.ndarray, bg: int, wall: int) -> list[dict]:
    h, w = grid.shape
    mask = (grid != bg) & (grid != wall)
    components = _components(mask)
    if not components:
        raise AssertionError("expected colored room interiors")

    rooms: list[dict] = []
    used_colors: set[int] = set()

    for cells in components:
        coords = np.array(cells, dtype=int)
        rs = coords[:, 0]
        cs = coords[:, 1]
        r0, r1 = int(rs.min()), int(rs.max())
        c0, c1 = int(cs.min()), int(cs.max())
        color = int(grid[r0, c0])

        if color in used_colors:
            raise AssertionError("room colors must be unique")
        used_colors.add(color)

        if not np.all(grid[rs, cs] == color):
            raise AssertionError("room component must be monochrome")
        if len(cells) != (r1 - r0 + 1) * (c1 - c0 + 1):
            raise AssertionError("room component must be a solid rectangle")
        if not np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color):
            raise AssertionError("room rectangle must be fully filled")

        if not (0 < r0 and r1 < h - 1 and 0 < c0 and c1 < w - 1):
            raise AssertionError("room must have surrounding wall cells")

        corners = (
            grid[r0 - 1, c0 - 1],
            grid[r0 - 1, c1 + 1],
            grid[r1 + 1, c0 - 1],
            grid[r1 + 1, c1 + 1],
        )
        if any(int(value) != wall for value in corners):
            raise AssertionError("room corners must stay walled")

        gap_candidates: list[tuple[str, tuple[int, int], tuple[int, int]]] = []
        for direction in ("top", "bottom", "left", "right"):
            info = _line_info(grid, bg, wall, (r0, r1, c0, c1), direction)
            if info is not None:
                gap, target = info
                tr, tc = target
                if not (0 <= tr < h and 0 <= tc < w):
                    raise AssertionError("gap must point to another room")
                if int(grid[tr, tc]) in (bg, wall):
                    raise AssertionError("gap must face another room interior")
                gap_candidates.append((direction, gap, target))

        if len(gap_candidates) != 1:
            raise AssertionError("each room must have exactly one gapped wall")

        direction, gap, target = gap_candidates[0]
        rooms.append(
            {
                "bbox": (r0, r1, c0, c1),
                "color": color,
                "direction": direction,
                "gap": gap,
                "target_sample": target,
            }
        )

    room_at = np.full(grid.shape, -1, dtype=int)
    for idx, room in enumerate(rooms):
        r0, r1, c0, c1 = room["bbox"]
        if np.any(room_at[r0 : r1 + 1, c0 : c1 + 1] != -1):
            raise AssertionError("rooms must not overlap")
        room_at[r0 : r1 + 1, c0 : c1 + 1] = idx

    for room in rooms:
        tr, tc = room["target_sample"]
        target_idx = int(room_at[tr, tc])
        if target_idx < 0:
            raise AssertionError("gap target must land inside another room")
        room["target_idx"] = target_idx

    for idx, room in enumerate(rooms):
        target = rooms[room["target_idx"]]
        if target["target_idx"] != idx:
            raise AssertionError("gaps must pair rooms bidirectionally")
        if target["gap"] != room["gap"]:
            raise AssertionError("paired rooms must share the same wall gap")
        if target["direction"] != OPPOSITE_DIRECTION[room["direction"]]:
            raise AssertionError("paired rooms must face each other")

    return rooms


def _parse_grid(grid: np.ndarray) -> tuple[int, int, list[dict]]:
    bg = BG if np.any(grid == BG) else int(np.bincount(grid.ravel()).argmax())
    wall_candidates = [int(color) for color in np.unique(grid) if int(color) != bg]
    parsed: list[tuple[tuple[int, int], int, list[dict]]] = []

    for wall in wall_candidates:
        try:
            rooms = _extract_rooms(grid, bg, wall)
        except AssertionError:
            continue
        score = (len(rooms), int(np.count_nonzero(grid == wall)))
        parsed.append((score, wall, rooms))

    if not parsed:
        raise AssertionError("failed to identify the wall color")

    _, wall, rooms = max(parsed, key=lambda item: item[0])
    return bg, wall, rooms


def _paint_frame(out: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, r1, c0, c1 = bbox
    out[r0 : r1 + 1, c0] = color
    out[r0 : r1 + 1, c1] = color
    out[r0, c0 : c1 + 1] = color
    out[r1, c0 : c1 + 1] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each room's color through its unique wall gap into the neighboring room as an inner border.

    Input structure:
    - Background is black.
    - Maroon one-cell-thick orthogonal walls partition the canvas into solid rectangular rooms.
    - Every room interior is a unique non-background, non-wall color.
    - Each room has exactly one wall side containing a single-cell background gap, and that gap is shared with one neighboring room.

    Rule:
    - Detect every rectangular room and find the wall side containing its unique gap.
    - Follow that gap to the adjacent room across the shared wall.
    - Clear the source room interior to background, then draw the source room's color as a one-cell-thick border along the inside of the target room.
    - Keep all maroon walls and all gap cells unchanged.

    Key insight:
    - The gap is a directional pairing marker between two rooms: the room's own color leaves its interior and reappears as a border inside the room on the other side of that shared gap.
    """
    g = np.asarray(grid, dtype=int)
    _, _, rooms = _parse_grid(g)

    out = g.copy()
    for room in rooms:
        r0, r1, c0, c1 = room["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = BG

    for room in rooms:
        target = rooms[room["target_idx"]]
        _paint_frame(out, target["bbox"], room["color"])

    return out


def _room_intervals(lengths: list[int], margin_before: int) -> tuple[list[tuple[int, int]], list[int]]:
    intervals: list[tuple[int, int]] = []
    walls = [margin_before]
    pos = margin_before + 1

    for length in lengths:
        start = pos
        end = pos + length - 1
        intervals.append((start, end))
        walls.append(end + 1)
        pos = end + 2

    return intervals, walls


def _grid_neighbors(cell: tuple[int, int], rows: int, cols: int) -> list[tuple[int, int]]:
    r, c = cell
    neighbors: list[tuple[int, int]] = []
    if r > 0:
        neighbors.append((r - 1, c))
    if r + 1 < rows:
        neighbors.append((r + 1, c))
    if c > 0:
        neighbors.append((r, c - 1))
    if c + 1 < cols:
        neighbors.append((r, c + 1))
    return neighbors


def _random_matching(rows: int, cols: int) -> list[tuple[tuple[int, int], tuple[int, int]]] | None:
    cells = [(r, c) for r in range(rows) for c in range(cols)]
    unpaired = set(cells)
    pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []

    def search() -> bool:
        if not unpaired:
            return True

        cell = min(unpaired)
        candidates = [nbr for nbr in _grid_neighbors(cell, rows, cols) if nbr in unpaired]
        random.shuffle(candidates)
        unpaired.remove(cell)
        for nbr in candidates:
            unpaired.remove(nbr)
            pairs.append((cell, nbr))
            if search():
                return True
            pairs.pop()
            unpaired.add(nbr)
        unpaired.add(cell)
        return False

    if search():
        random.shuffle(pairs)
        return pairs
    return None


def _apply_random_symmetry(grid: np.ndarray) -> np.ndarray:
    out = np.rot90(grid, random.randrange(4))
    if random.random() < 0.5:
        out = np.fliplr(out)
    return np.array(out, copy=True)


def _gen_attempt() -> np.ndarray | None:
    room_rows, room_cols = random.choice(ROOM_LAYOUTS)
    row_sizes = [random.randint(*ROOM_SIZE_RANGE) for _ in range(room_rows)]
    col_sizes = [random.randint(*ROOM_SIZE_RANGE) for _ in range(room_cols)]
    top_margin = random.randint(*MARGIN_RANGE)
    bottom_margin = random.randint(*MARGIN_RANGE)
    left_margin = random.randint(*MARGIN_RANGE)
    right_margin = random.randint(*MARGIN_RANGE)

    row_intervals, row_walls = _room_intervals(row_sizes, top_margin)
    col_intervals, col_walls = _room_intervals(col_sizes, left_margin)

    height = row_walls[-1] + 1 + bottom_margin
    width = col_walls[-1] + 1 + right_margin
    if height > 30 or width > 30:
        return None

    matching = _random_matching(room_rows, room_cols)
    if matching is None:
        return None

    grid = np.full((height, width), BG, dtype=int)
    top_wall, bottom_wall = row_walls[0], row_walls[-1]
    left_wall, right_wall = col_walls[0], col_walls[-1]

    for wall_row in row_walls:
        grid[wall_row, left_wall : right_wall + 1] = WALL
    for wall_col in col_walls:
        grid[top_wall : bottom_wall + 1, wall_col] = WALL

    colors = random.sample(COLOR_POOL, room_rows * room_cols)
    room_colors: dict[tuple[int, int], int] = {}
    for idx, (r, c) in enumerate((rr, cc) for rr in range(room_rows) for cc in range(room_cols)):
        color = colors[idx]
        room_colors[(r, c)] = color
        r0, r1 = row_intervals[r]
        c0, c1 = col_intervals[c]
        grid[r0 : r1 + 1, c0 : c1 + 1] = color

    pair_orientations = set()
    for a, b in matching:
        if a[0] == b[0]:
            room_r = a[0]
            left_c = min(a[1], b[1])
            shared_col = col_walls[left_c + 1]
            r0, r1 = row_intervals[room_r]
            gap_row = random.randint(r0, r1)
            grid[gap_row, shared_col] = BG
            pair_orientations.add("horizontal")
        else:
            upper_r = min(a[0], b[0])
            room_c = a[1]
            shared_row = row_walls[upper_r + 1]
            c0, c1 = col_intervals[room_c]
            gap_col = random.randint(c0, c1)
            grid[shared_row, gap_col] = BG
            pair_orientations.add("vertical")

    if len(matching) > 1 and len(pair_orientations) == 1 and random.random() < 0.6:
        return None

    grid = _apply_random_symmetry(grid)

    try:
        _, wall, rooms = _parse_grid(grid)
    except AssertionError:
        return None

    if wall != WALL or len(rooms) != room_rows * room_cols:
        return None

    out = main(grid)
    if out.shape != grid.shape or np.array_equal(out, grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid room-gap transfer puzzle")
