from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ARC_UTILS = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS) not in sys.path:
    sys.path.append(str(ARC_UTILS))

from arc_utils import Color, bbox_coords, find_components


BG = Color.BLACK
SEPARATOR = Color.GREY
WALL = Color.RED
MARKER_COLORS = (
    Color.BLUE,
    Color.GREEN,
    Color.YELLOW,
    Color.PINK,
    Color.ORANGE,
    Color.TEAL,
    Color.MAROON,
)
DOOR_SIDES = ("top", "bottom", "left", "right")
PQ_CHOICES = ((2, 2), (2, 3), (3, 2), (3, 3))
MAX_ATTEMPTS = 500


def _bands_from_separator(subgrid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    sep_rows = [r for r in range(subgrid.shape[0]) if np.all(subgrid[r, :] == SEPARATOR)]
    sep_cols = [c for c in range(subgrid.shape[1]) if np.all(subgrid[:, c] == SEPARATOR)]
    assert len(sep_rows) >= 2 and len(sep_cols) >= 2

    row_mask = np.zeros(subgrid.shape[0], dtype=bool)
    col_mask = np.zeros(subgrid.shape[1], dtype=bool)
    row_mask[sep_rows] = True
    col_mask[sep_cols] = True
    expected_sep = row_mask[:, None] | col_mask[None, :]
    assert np.array_equal(subgrid == SEPARATOR, expected_sep), "separator board must be a clean line grid"

    row_bands: list[tuple[int, int]] = []
    col_bands: list[tuple[int, int]] = []
    for a, b in zip(sep_rows, sep_rows[1:]):
        assert b - a >= 2, "separator rows must enclose nonempty blueprint cells"
        row_bands.append((a + 1, b - 1))
    for a, b in zip(sep_cols, sep_cols[1:]):
        assert b - a >= 2, "separator cols must enclose nonempty blueprint cells"
        col_bands.append((a + 1, b - 1))
    return row_bands, col_bands


def _extract_blueprint(grid: np.ndarray) -> dict:
    coords = np.argwhere(grid == SEPARATOR)
    assert coords.size > 0, "expected grey blueprint separators"
    r0, c0, r1, c1 = bbox_coords(coords)
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    row_bands_local, col_bands_local = _bands_from_separator(sub)
    row_bands = [(r0 + a, r0 + b) for a, b in row_bands_local]
    col_bands = [(c0 + a, c0 + b) for a, b in col_bands_local]
    return {"bbox": (r0, c0, r1, c1), "row_bands": row_bands, "col_bands": col_bands}


def _room_border_positions(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    border: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        border.append((r0, c))
        border.append((r1, c))
    for r in range(r0 + 1, r1):
        border.append((r, c0))
        border.append((r, c1))
    return border


def _room_side_from_gap(bbox: tuple[int, int, int, int], gap: tuple[int, int]) -> str:
    r0, c0, r1, c1 = bbox
    r, c = gap
    if r == r0 and c0 < c < c1:
        return "top"
    if r == r1 and c0 < c < c1:
        return "bottom"
    if c == c0 and r0 < r < r1:
        return "left"
    if c == c1 and r0 < r < r1:
        return "right"
    raise AssertionError("door gap must lie on exactly one non-corner wall position")


def _extract_rooms(grid: np.ndarray) -> list[list[dict]]:
    components = find_components(grid, bg=BG, colors=[WALL], connectivity=4)
    assert components, "expected red wall rooms"

    rooms: list[dict] = []
    for obj in components:
        bbox = tuple(map(int, obj["bbox"]))
        r0, c0, r1, c1 = bbox
        assert r1 - r0 >= 4 and c1 - c0 >= 4, "room interior must be larger than blueprint cells"

        missing = [(r, c) for r, c in _room_border_positions(bbox) if int(grid[r, c]) != WALL]
        assert len(missing) == 1, "each room must have exactly one door gap"
        side = _room_side_from_gap(bbox, missing[0])

        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        assert interior.size > 0
        allowed = {BG}
        assert set(map(int, np.unique(interior))) <= allowed, "room interiors must start empty"
        rooms.append({"bbox": bbox, "door": missing[0], "side": side})

    row_starts = sorted({room["bbox"][0] for room in rooms})
    col_starts = sorted({room["bbox"][1] for room in rooms})
    assert len(rooms) == len(row_starts) * len(col_starts), "rooms must form a full p x q array"

    grid_rooms: list[list[dict | None]] = [[None for _ in col_starts] for _ in row_starts]
    row_index = {value: i for i, value in enumerate(row_starts)}
    col_index = {value: j for j, value in enumerate(col_starts)}
    for room in rooms:
        i = row_index[room["bbox"][0]]
        j = col_index[room["bbox"][1]]
        assert grid_rooms[i][j] is None, "duplicate room position"
        grid_rooms[i][j] = room

    assert all(room is not None for row in grid_rooms for room in row)
    return [[room for room in row if room is not None] for row in grid_rooms]


def _read_blueprint_cell(grid: np.ndarray, row_band: tuple[int, int], col_band: tuple[int, int]) -> tuple[int | None, int]:
    r0, r1 = row_band
    c0, c1 = col_band
    cell = grid[r0 : r1 + 1, c0 : c1 + 1]
    colors = set(map(int, np.unique(cell))) - {BG}
    assert SEPARATOR not in colors and WALL not in colors, "blueprint cells may only hold markers"
    if not colors:
        return None, 0

    assert len(colors) == 1, "a blueprint cell may use at most one marker color"
    color = colors.pop()
    assert color in MARKER_COLORS
    count = int(np.count_nonzero(cell == color))
    assert count > 0
    assert set(map(int, np.unique(cell))) <= {BG, color}
    return color, count


def _room_depth(room: dict) -> int:
    r0, c0, r1, c1 = room["bbox"]
    if room["side"] in ("top", "bottom"):
        return r1 - r0 - 1
    return c1 - c0 - 1


def _fill_room(out: np.ndarray, room: dict, count: int, color: int) -> None:
    if count == 0:
        return

    r0, c0, r1, c1 = room["bbox"]
    side = room["side"]
    if side == "top":
        out[r0 + 1 : r0 + 1 + count, c0 + 1 : c1] = color
    elif side == "bottom":
        out[r1 - count : r1, c0 + 1 : c1] = color
    elif side == "left":
        out[r0 + 1 : r1, c0 + 1 : c0 + 1 + count] = color
    elif side == "right":
        out[r0 + 1 : r1, c1 - count : c1] = color
    else:
        raise AssertionError("unknown door side")


def _instructions_from_grid(grid: np.ndarray, blueprint: dict) -> list[list[tuple[int | None, int]]]:
    instructions: list[list[tuple[int | None, int]]] = []
    for row_band in blueprint["row_bands"]:
        row_data = []
        for col_band in blueprint["col_bands"]:
            row_data.append(_read_blueprint_cell(grid, row_band, col_band))
        instructions.append(row_data)
    return instructions


def _apply_instructions(grid: np.ndarray, rooms: list[list[dict]], instructions: list[list[tuple[int | None, int]]]) -> np.ndarray:
    out = grid.copy()
    for i, row in enumerate(rooms):
        for j, room in enumerate(row):
            color, count = instructions[i][j]
            depth = _room_depth(room)
            assert 0 <= count <= depth
            if count > 0:
                assert color is not None
                _fill_room(out, room, count, int(color))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each red room with as many door-side layers as the matching blueprint cell's marker count, using that marker color.

    Input structure:
    - The left side contains a grey separator grid whose p x q cells form a small blueprint; each cell has either no marker or several same-colored marker pixels.
    - The right side contains a p x q array of larger red-walled rectangular rooms; each room has exactly one one-cell door gap on one wall and an empty interior.
    - The two arrays correspond by row and column position only, while black background separates the two boards.

    Rule:
    - Partition the grey blueprint into its cells and, for each cell, count its marker pixels k and record their unique marker color c.
    - In the correspondingly positioned red room, leave the room unchanged if k = 0; otherwise fill exactly k complete interior rows or columns with color c, starting from the wall that contains that room's door gap and moving inward.
    - Preserve the grey blueprint, the red walls, the door gaps, and all untouched background cells.

    Key insight:
    - Quantity and color come from the distant blueprint cell, but the fill direction comes only from the local room door, so the two aligned boards must be matched by shared p x q layout.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    blueprint = _extract_blueprint(g)
    rooms = _extract_rooms(g)
    assert len(blueprint["row_bands"]) == len(rooms)
    assert len(blueprint["col_bands"]) == len(rooms[0])

    instructions = _instructions_from_grid(g, blueprint)
    return _apply_instructions(g, rooms, instructions)


def _choose_layout() -> dict | None:
    p, q = random.choice(PQ_CHOICES)
    cell_h = random.randint(2, 3)
    cell_w = random.randint(2, 3)
    room_h = random.randint(cell_h + 1, 5)
    room_w = random.randint(cell_w + 1, 5)
    room_gap_r = random.randint(1, 2)
    room_gap_c = random.randint(1, 2)
    board_gap = random.randint(2, 4)

    blueprint_h = p * cell_h + (p + 1)
    blueprint_w = q * cell_w + (q + 1)
    rooms_h = p * (room_h + 2) + (p - 1) * room_gap_r
    rooms_w = q * (room_w + 2) + (q - 1) * room_gap_c

    max_h = max(blueprint_h, rooms_h) + 2
    max_w = blueprint_w + board_gap + rooms_w
    if max_h > 30 or max_w > 30:
        return None

    bp_top = random.randint(0, 2)
    room_top = random.randint(0, 2)
    height = max(bp_top + blueprint_h, room_top + rooms_h)
    width = blueprint_w + board_gap + rooms_w
    if height > 30 or width > 30:
        return None

    return {
        "p": p,
        "q": q,
        "cell_h": cell_h,
        "cell_w": cell_w,
        "room_h": room_h,
        "room_w": room_w,
        "room_gap_r": room_gap_r,
        "room_gap_c": room_gap_c,
        "board_gap": board_gap,
        "bp_top": bp_top,
        "bp_left": 0,
        "room_top": room_top,
        "room_left": blueprint_w + board_gap,
        "height": height,
        "width": width,
    }


def _draw_blueprint(grid: np.ndarray, spec: dict) -> list[list[tuple[int, int, int, int]]]:
    p, q = spec["p"], spec["q"]
    cell_h, cell_w = spec["cell_h"], spec["cell_w"]
    top, left = spec["bp_top"], spec["bp_left"]
    total_h = p * cell_h + (p + 1)
    total_w = q * cell_w + (q + 1)
    for i in range(p + 1):
        r = top + i * (cell_h + 1)
        grid[r, left : left + total_w] = SEPARATOR
    for j in range(q + 1):
        c = left + j * (cell_w + 1)
        grid[top : top + total_h, c] = SEPARATOR

    cells: list[list[tuple[int, int, int, int]]] = []
    for i in range(p):
        row_cells = []
        for j in range(q):
            r0 = top + 1 + i * (cell_h + 1)
            c0 = left + 1 + j * (cell_w + 1)
            row_cells.append((r0, c0, r0 + cell_h - 1, c0 + cell_w - 1))
        cells.append(row_cells)
    return cells


def _door_gap_for_side(bbox: tuple[int, int, int, int], side: str) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    if side == "top":
        return r0, random.randint(c0 + 1, c1 - 1)
    if side == "bottom":
        return r1, random.randint(c0 + 1, c1 - 1)
    if side == "left":
        return random.randint(r0 + 1, r1 - 1), c0
    if side == "right":
        return random.randint(r0 + 1, r1 - 1), c1
    raise AssertionError("unknown side")


def _draw_rooms(grid: np.ndarray, spec: dict) -> list[list[dict]]:
    p, q = spec["p"], spec["q"]
    room_h, room_w = spec["room_h"], spec["room_w"]
    gap_r, gap_c = spec["room_gap_r"], spec["room_gap_c"]
    top, left = spec["room_top"], spec["room_left"]

    used_dirs = random.sample(DOOR_SIDES, random.randint(2, 4))
    sides = used_dirs[:]
    while len(sides) < p * q:
        sides.append(random.choice(used_dirs))
    random.shuffle(sides)

    rooms: list[list[dict]] = []
    for i in range(p):
        row_rooms = []
        for j in range(q):
            r0 = top + i * (room_h + 2 + gap_r)
            c0 = left + j * (room_w + 2 + gap_c)
            bbox = (r0, c0, r0 + room_h + 1, c0 + room_w + 1)
            rr0, cc0, rr1, cc1 = bbox
            grid[rr0 : rr1 + 1, cc0] = WALL
            grid[rr0 : rr1 + 1, cc1] = WALL
            grid[rr0, cc0 : cc1 + 1] = WALL
            grid[rr1, cc0 : cc1 + 1] = WALL

            side = sides[i * q + j]
            gap = _door_gap_for_side(bbox, side)
            grid[gap] = BG
            row_rooms.append({"bbox": bbox, "door": gap, "side": side})
        rooms.append(row_rooms)
    return rooms


def _choose_instruction_grid(rooms: list[list[dict]]) -> list[list[tuple[int | None, int]]]:
    flat_rooms = [room for row in rooms for room in row]
    n = len(flat_rooms)
    assert n >= 4

    by_side: dict[str, list[int]] = {}
    for idx, room in enumerate(flat_rooms):
        by_side.setdefault(room["side"], []).append(idx)

    side_choices = [side for side, ids in by_side.items() if ids]
    side_a, side_b = random.sample(side_choices, 2)
    a = random.choice(by_side[side_a])
    b = random.choice(by_side[side_b])
    remaining = [idx for idx in range(n) if idx not in {a, b}]
    d, z = random.sample(remaining, 2)

    counts = [None] * n
    colors: list[int | None] = [None] * n

    depth_a = _room_depth(flat_rooms[a])
    depth_b = _room_depth(flat_rooms[b])
    shared_count = random.randint(1, min(depth_a, depth_b))
    color_a = random.choice(MARKER_COLORS)
    other_colors = [color for color in MARKER_COLORS if color != color_a]
    color_b = random.choice(other_colors)

    counts[a] = shared_count
    colors[a] = color_a
    counts[b] = shared_count
    colors[b] = color_b

    depth_d = _room_depth(flat_rooms[d])
    diff_options = [value for value in range(1, depth_d + 1) if value != shared_count]
    assert diff_options
    diff_count = depth_d if depth_d != shared_count else random.choice(diff_options)
    counts[d] = diff_count
    colors[d] = color_a

    counts[z] = 0
    colors[z] = None

    full_fill_present = shared_count == depth_a or shared_count == depth_b or diff_count == depth_d
    for idx, room in enumerate(flat_rooms):
        if counts[idx] is not None:
            continue
        depth = _room_depth(room)
        count = random.randint(0, depth)
        if idx == remaining[-1] and not full_fill_present:
            count = depth
        counts[idx] = count
        colors[idx] = None if count == 0 else random.choice(MARKER_COLORS)
        full_fill_present = full_fill_present or count == depth

    if not full_fill_present:
        return []

    positive_colors = {int(color) for color, count in zip(colors, counts) if count and color is not None}
    if len(positive_colors) < 2:
        return []
    if not any(count == 0 for count in counts):
        return []
    if not any(count > 1 for count in counts):
        return []
    if not any(
        counts[i] == counts[j] and counts[i] > 0 and flat_rooms[i]["side"] != flat_rooms[j]["side"]
        for i in range(n)
        for j in range(i + 1, n)
    ):
        return []
    if not any(
        colors[i] == colors[j]
        and colors[i] is not None
        and counts[i] != counts[j]
        and counts[i] > 0
        and counts[j] > 0
        for i in range(n)
        for j in range(i + 1, n)
    ):
        return []

    q = len(rooms[0])
    instructions = []
    for i, row in enumerate(rooms):
        row_items = []
        for j, _room in enumerate(row):
            idx = i * q + j
            row_items.append((colors[idx], int(counts[idx])))
        instructions.append(row_items)
    return instructions


def _place_markers(grid: np.ndarray, cells: list[list[tuple[int, int, int, int]]], instructions: list[list[tuple[int | None, int]]]) -> None:
    for i, row in enumerate(cells):
        for j, bbox in enumerate(row):
            color, count = instructions[i][j]
            if count == 0:
                continue
            r0, c0, r1, c1 = bbox
            coords = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
            positions = random.sample(coords, count)
            for r, c in positions:
                grid[r, c] = int(color)


def _all_bottom_variant(grid: np.ndarray, rooms: list[list[dict]], instructions: list[list[tuple[int | None, int]]]) -> np.ndarray:
    fake_rooms = [[{**room, "side": "bottom"} for room in row] for row in rooms]
    adjusted: list[list[tuple[int | None, int]]] = []
    for room_row, inst_row in zip(fake_rooms, instructions):
        row_items = []
        for room, (color, count) in zip(room_row, inst_row):
            row_items.append((color, min(count, _room_depth(room))))
        adjusted.append(row_items)
    return _apply_instructions(grid, fake_rooms, adjusted)


def _reversed_columns_variant(grid: np.ndarray, rooms: list[list[dict]], instructions: list[list[tuple[int | None, int]]]) -> np.ndarray:
    adjusted: list[list[tuple[int | None, int]]] = []
    for room_row, inst_row in zip(rooms, instructions):
        reversed_row = list(reversed(inst_row))
        row_items = []
        for room, (color, count) in zip(room_row, reversed_row):
            row_items.append((color, min(count, _room_depth(room))))
        adjusted.append(row_items)
    return _apply_instructions(grid, rooms, adjusted)


def _unit_count_variant(grid: np.ndarray, rooms: list[list[dict]], instructions: list[list[tuple[int | None, int]]]) -> np.ndarray:
    shrunk = []
    for row in instructions:
        shrunk_row = []
        for color, count in row:
            shrunk_row.append((color, 0 if count == 0 else 1))
        shrunk.append(shrunk_row)
    return _apply_instructions(grid, rooms, shrunk)


def _gen_attempt() -> np.ndarray | None:
    spec = _choose_layout()
    if spec is None:
        return None

    grid = np.full((spec["height"], spec["width"]), BG, dtype=int)
    cells = _draw_blueprint(grid, spec)
    rooms = _draw_rooms(grid, spec)
    instructions = _choose_instruction_grid(rooms)
    if not instructions:
        return None
    max_cell_area = spec["cell_h"] * spec["cell_w"]
    if any(count > max_cell_area for row in instructions for _color, count in row):
        return None
    _place_markers(grid, cells, instructions)

    try:
        expected = _apply_instructions(grid, rooms, instructions)
        actual = main(grid)
        parsed_blueprint = _extract_blueprint(grid)
        parsed_instructions = _instructions_from_grid(grid, parsed_blueprint)
    except AssertionError:
        return None

    if parsed_instructions != instructions:
        return None
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(expected, _all_bottom_variant(grid, rooms, instructions)):
        return None
    if np.array_equal(expected, _reversed_columns_variant(grid, rooms, instructions)):
        return None
    if np.array_equal(expected, _unit_count_variant(grid, rooms, instructions)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
