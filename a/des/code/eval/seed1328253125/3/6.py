from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GRAY = 5
MAGENTA = 6
ORANGE = 7
CYAN = 8
BROWN = 9

PALETTE_POOL = (BLUE, GREEN, YELLOW, MAGENTA, ORANGE, CYAN, BROWN)
PALETTE_LENGTH = 5
CORNERS = ("tl", "tr", "bl", "br")
PALETTE_ORIENTATIONS = ("horizontal", "vertical")

GRID_HEIGHT_RANGE = (18, 25)
GRID_WIDTH_RANGE = (18, 25)
FRAME_INSET_OPTIONS = (2, 3)
MIN_ROOM_HEIGHT = 3
MIN_ROOM_WIDTH = 3
TARGET_ROOM_RANGE = (4, 7)
GENERATION_ATTEMPTS = 400
COUNT_RETRIES = 200


def _neighbors4(r: int, c: int, height: int, width: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < height:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < width:
        out.append((r, c + 1))
    return out


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    height, width = map(int, mask.shape)
    labels = np.zeros((height, width), dtype=int)
    components: list[np.ndarray] = []
    comp_id = 0

    for r in range(height):
        for c in range(width):
            if not bool(mask[r, c]) or int(labels[r, c]) != 0:
                continue
            comp_id += 1
            stack = [(r, c)]
            labels[r, c] = comp_id
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, height, width):
                    if bool(mask[nr, nc]) and int(labels[nr, nc]) == 0:
                        labels[nr, nc] = comp_id
                        stack.append((nr, nc))
            components.append(np.array(coords, dtype=int))

    return labels, components


def _touches_edge(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    height, width = map(int, shape)
    return bool(
        (coords[:, 0] == 0).any()
        or (coords[:, 0] == height - 1).any()
        or (coords[:, 1] == 0).any()
        or (coords[:, 1] == width - 1).any()
    )


def _detect_palette(grid: np.ndarray) -> tuple[list[int], np.ndarray]:
    palette_mask = ~np.isin(grid, (BLACK, RED, GRAY))
    coords = np.argwhere(palette_mask)
    assert int(coords.shape[0]) == PALETTE_LENGTH, "expected exactly five palette cells"
    assert len({int(grid[r, c]) for r, c in coords}) == PALETTE_LENGTH, "palette colors must be distinct"

    height, width = map(int, grid.shape)
    rows = sorted({int(r) for r, _c in coords})
    cols = sorted({int(c) for _r, c in coords})

    if len(rows) == 1:
        r = rows[0]
        assert r in {0, height - 1}, "horizontal palette must lie on a border row"
        assert cols == list(range(cols[0], cols[0] + PALETTE_LENGTH)), "palette cells must be consecutive"
        if cols[0] == 0:
            order = [(r, c) for c in cols]
        else:
            assert cols[-1] == width - 1, "horizontal palette must start from a corner"
            order = [(r, c) for c in cols[::-1]]
    else:
        assert len(cols) == 1, "palette must be a straight line"
        c = cols[0]
        assert c in {0, width - 1}, "vertical palette must lie on a border column"
        assert rows == list(range(rows[0], rows[0] + PALETTE_LENGTH)), "palette cells must be consecutive"
        if rows[0] == 0:
            order = [(r, c) for r in rows]
        else:
            assert rows[-1] == height - 1, "vertical palette must start from a corner"
            order = [(r, c) for r in rows[::-1]]

    return [int(grid[r, c]) for r, c in order], palette_mask


def _room_color_lookup(grid: np.ndarray, palette: list[int]) -> tuple[np.ndarray, dict[int, int]]:
    passable = grid != GRAY
    labels, components = _label_components(passable)
    room_colors: dict[int, int] = {}
    for comp_id, coords in enumerate(components, start=1):
        if _touches_edge(coords, grid.shape):
            continue
        count = int(np.sum(grid[labels == comp_id] == RED))
        assert 1 <= count <= PALETTE_LENGTH, "each room must contain 1..5 furniture pixels"
        room_colors[comp_id] = int(palette[count - 1])
    assert room_colors, "expected enclosed rooms"
    return labels, room_colors


def _wall_orientation(wall: np.ndarray, r: int, c: int) -> str:
    up = r > 0 and bool(wall[r - 1, c])
    down = r + 1 < wall.shape[0] and bool(wall[r + 1, c])
    left = c > 0 and bool(wall[r, c - 1])
    right = c + 1 < wall.shape[1] and bool(wall[r, c + 1])

    if left and right and not up and not down:
        return "horizontal"
    if up and down and not left and not right:
        return "vertical"
    return ""


def _room_color_at(labels: np.ndarray, room_colors: dict[int, int], r: int, c: int) -> int:
    if not (0 <= r < labels.shape[0] and 0 <= c < labels.shape[1]):
        return 0
    comp_id = int(labels[r, c])
    return int(room_colors.get(comp_id, 0))


def _consensus_wall_color(side_a: int, side_b: int) -> int:
    if side_a != 0 and side_b != 0:
        return int(side_a) if side_a == side_b else GRAY
    if side_a != 0:
        return int(side_a)
    if side_b != 0:
        return int(side_b)
    return GRAY


def _node_color(labels: np.ndarray, room_colors: dict[int, int], r: int, c: int) -> int:
    seen = {
        _room_color_at(labels, room_colors, r + dr, c + dc)
        for dr in (-1, 1)
        for dc in (-1, 1)
        if 0 <= r + dr < labels.shape[0] and 0 <= c + dc < labels.shape[1]
    }
    seen.discard(0)
    if len(seen) == 1:
        return int(next(iter(seen)))
    return GRAY


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor room walls according to furniture counts and a corner palette.

    Input structure:
    - black background with gray one-cell-thick rectilinear walls
    - red single-pixel furniture inside enclosed rooms
    - five distinct palette colors arranged from a corner outward on one edge

    Rule:
    - count red furniture in each enclosed room and map 1..5 to the palette order
    - a shared wall segment recolors only when the two adjacent rooms map to the same color
    - a boundary wall with only one adjacent room recolors to that room color
    - red furniture disappears; the palette stays unchanged
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    palette, _palette_mask = _detect_palette(g)
    labels, room_colors = _room_color_lookup(g, palette)
    wall = g == GRAY

    out = g.copy()
    out[g == RED] = BLACK

    height, width = map(int, g.shape)
    for r in range(height):
        for c in range(width):
            if not bool(wall[r, c]):
                continue
            orient = _wall_orientation(wall, r, c)
            if orient == "horizontal":
                side_a = _room_color_at(labels, room_colors, r - 1, c)
                side_b = _room_color_at(labels, room_colors, r + 1, c)
                out[r, c] = _consensus_wall_color(side_a, side_b)
            elif orient == "vertical":
                side_a = _room_color_at(labels, room_colors, r, c - 1)
                side_b = _room_color_at(labels, room_colors, r, c + 1)
                out[r, c] = _consensus_wall_color(side_a, side_b)
            else:
                out[r, c] = _node_color(labels, room_colors, r, c)

    return out


def _palette_cells(corner: str, orientation: str, height: int, width: int) -> list[tuple[int, int]]:
    if corner == "tl":
        if orientation == "horizontal":
            return [(0, c) for c in range(PALETTE_LENGTH)]
        return [(r, 0) for r in range(PALETTE_LENGTH)]
    if corner == "tr":
        if orientation == "horizontal":
            return [(0, width - 1 - i) for i in range(PALETTE_LENGTH)]
        return [(r, width - 1) for r in range(PALETTE_LENGTH)]
    if corner == "bl":
        if orientation == "horizontal":
            return [(height - 1, c) for c in range(PALETTE_LENGTH)]
        return [(height - 1 - i, 0) for i in range(PALETTE_LENGTH)]
    assert corner == "br"
    if orientation == "horizontal":
        return [(height - 1, width - 1 - i) for i in range(PALETTE_LENGTH)]
    return [(height - 1 - i, width - 1) for i in range(PALETTE_LENGTH)]


def _outer_frame(corner: str, height: int, width: int) -> tuple[int, int, int, int]:
    top = random.choice(FRAME_INSET_OPTIONS) if "t" in corner else 0
    bottom = height - 1 - random.choice(FRAME_INSET_OPTIONS) if "b" in corner else height - 1
    left = random.choice(FRAME_INSET_OPTIONS) if "l" in corner else 0
    right = width - 1 - random.choice(FRAME_INSET_OPTIONS) if "r" in corner else width - 1
    return int(top), int(bottom), int(left), int(right)


def _draw_frame(grid: np.ndarray, frame: tuple[int, int, int, int]) -> None:
    top, bottom, left, right = frame
    grid[top : bottom + 1, left] = GRAY
    grid[top : bottom + 1, right] = GRAY
    grid[top, left : right + 1] = GRAY
    grid[bottom, left : right + 1] = GRAY


def _split_candidates(room: tuple[int, int, int, int]) -> list[str]:
    top, bottom, left, right = room
    options: list[str] = []
    if top + MIN_ROOM_HEIGHT + 1 <= bottom - MIN_ROOM_HEIGHT - 1:
        options.append("horizontal")
    if left + MIN_ROOM_WIDTH + 1 <= right - MIN_ROOM_WIDTH - 1:
        options.append("vertical")
    return options


def _split_rooms(frame: tuple[int, int, int, int]) -> tuple[list[tuple[int, int, int, int]], list[tuple[str, int, int, int]]]:
    target_rooms = random.randint(*TARGET_ROOM_RANGE)
    rooms = [frame]
    splits: list[tuple[str, int, int, int]] = []

    for _ in range(target_rooms - 1):
        options: list[tuple[int, str]] = []
        for idx, room in enumerate(rooms):
            for orientation in _split_candidates(room):
                options.append((idx, orientation))
        if not options:
            break

        idx, orientation = random.choice(options)
        top, bottom, left, right = rooms.pop(idx)
        if orientation == "horizontal":
            row = random.choice(list(range(top + MIN_ROOM_HEIGHT + 1, bottom - MIN_ROOM_HEIGHT)))
            splits.append(("horizontal", int(row), int(left), int(right)))
            rooms.append((top, row, left, right))
            rooms.append((row, bottom, left, right))
        else:
            col = random.choice(list(range(left + MIN_ROOM_WIDTH + 1, right - MIN_ROOM_WIDTH)))
            splits.append(("vertical", int(col), int(top), int(bottom)))
            rooms.append((top, bottom, left, col))
            rooms.append((top, bottom, col, right))

    return rooms, splits


def _adjacent_pairs(rooms: list[tuple[int, int, int, int]]) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for i in range(len(rooms)):
        t1, b1, l1, r1 = rooms[i]
        for j in range(i + 1, len(rooms)):
            t2, b2, l2, r2 = rooms[j]
            if r1 == l2 or r2 == l1:
                overlap = min(b1, b2) - max(t1, t2) + 1
                if overlap >= 3:
                    pairs.append((i, j))
            elif b1 == t2 or b2 == t1:
                overlap = min(r1, r2) - max(l1, l2) + 1
                if overlap >= 3:
                    pairs.append((i, j))
    return pairs


def _sample_counts(rooms: list[tuple[int, int, int, int]]) -> list[int] | None:
    pairs = _adjacent_pairs(rooms)
    if len(pairs) < 2:
        return None
    for _ in range(COUNT_RETRIES):
        counts = [random.randint(1, PALETTE_LENGTH) for _ in rooms]
        same = any(counts[i] == counts[j] for i, j in pairs)
        diff = any(counts[i] != counts[j] for i, j in pairs)
        if same and diff:
            return counts
    return None


def _paint_splits(grid: np.ndarray, splits: list[tuple[str, int, int, int]]) -> None:
    for orientation, axis, start, end in splits:
        if orientation == "horizontal":
            grid[axis, start : end + 1] = GRAY
        else:
            grid[start : end + 1, axis] = GRAY


def _place_palette(grid: np.ndarray, corner: str, orientation: str, palette: list[int]) -> None:
    for color, (r, c) in zip(palette, _palette_cells(corner, orientation, *grid.shape), strict=False):
        grid[r, c] = int(color)


def _room_cells(room: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    top, bottom, left, right = room
    return [
        (r, c)
        for r in range(top + 1, bottom)
        for c in range(left + 1, right)
    ]


def _place_furniture(grid: np.ndarray, rooms: list[tuple[int, int, int, int]], counts: list[int]) -> None:
    for room, count in zip(rooms, counts, strict=False):
        cells = _room_cells(room)
        assert len(cells) >= count
        for r, c in random.sample(cells, count):
            grid[r, c] = RED


def _valid_frame(frame: tuple[int, int, int, int]) -> bool:
    top, bottom, left, right = frame
    return bool(
        bottom - top - 1 >= 2 * MIN_ROOM_HEIGHT + 1
        and right - left - 1 >= 2 * MIN_ROOM_WIDTH + 1
    )


def _branch_flags(grid: np.ndarray, palette: list[int]) -> tuple[bool, bool, bool]:
    labels, room_colors = _room_color_lookup(grid, palette)
    wall = grid == GRAY
    shared_same = False
    shared_diff = False
    boundary = False

    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if not bool(wall[r, c]):
                continue
            orient = _wall_orientation(wall, r, c)
            if orient == "horizontal":
                side_a = _room_color_at(labels, room_colors, r - 1, c)
                side_b = _room_color_at(labels, room_colors, r + 1, c)
            elif orient == "vertical":
                side_a = _room_color_at(labels, room_colors, r, c - 1)
                side_b = _room_color_at(labels, room_colors, r, c + 1)
            else:
                continue

            if side_a != 0 and side_b != 0:
                if side_a == side_b:
                    shared_same = True
                else:
                    shared_diff = True
            elif side_a != 0 or side_b != 0:
                boundary = True

    return shared_same, shared_diff, boundary


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    corner = random.choice(CORNERS)
    orientation = random.choice(PALETTE_ORIENTATIONS)
    palette = random.sample(PALETTE_POOL, PALETTE_LENGTH)

    frame = _outer_frame(corner, height, width)
    if not _valid_frame(frame):
        return None

    rooms, splits = _split_rooms(frame)
    if len(rooms) < TARGET_ROOM_RANGE[0]:
        return None

    counts = _sample_counts(rooms)
    if counts is None:
        return None

    grid = np.full((height, width), BLACK, dtype=int)
    _draw_frame(grid, frame)
    _paint_splits(grid, splits)
    _place_furniture(grid, rooms, counts)
    _place_palette(grid, corner, orientation, palette)

    try:
        detected_palette, _ = _detect_palette(grid)
        if detected_palette != palette:
            return None
        output = main(grid)
    except Exception:
        return None

    shared_same, shared_diff, boundary = _branch_flags(grid, palette)
    if not (shared_same and shared_diff and boundary):
        return None
    if np.array_equal(output, grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        candidate = _gen_attempt()
        if candidate is not None:
            return candidate
    raise RuntimeError(f"failed to generate a valid puzzle after {GENERATION_ATTEMPTS} attempts")
