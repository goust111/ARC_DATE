import random
from collections import Counter

import numpy as np


BLACK = 0
GREY = 5
GEN_ATTEMPTS = 300
ROOM_COUNT_RANGE = (3, 5)
ROOM_SIZE_RANGE = (2, 5)
EXTRA_SOURCE_RANGE = (0, 3)
EXTRA_PIPE_RANGE = (0, 3)
COLOR_POOL = tuple(c for c in range(10) if c not in {BLACK, GREY})


def _most_common_non_black(grid: np.ndarray) -> int:
    values = grid[grid != BLACK]
    assert values.size > 0, "expected at least one non-black pixel"
    return Counter(map(int, values)).most_common(1)[0][0]


def _find_wall_rows(grid: np.ndarray) -> list[int]:
    rows = []
    for r in range(grid.shape[0]):
        row = grid[r]
        if row.size >= 2 and np.any((row[:-1] == BLACK) & (row[1:] == BLACK)):
            rows.append(int(r))
    return rows


def _find_wall_cols(grid: np.ndarray) -> list[int]:
    cols = []
    for c in range(grid.shape[1]):
        col = grid[:, c]
        if col.size >= 2 and np.any((col[:-1] == BLACK) & (col[1:] == BLACK)):
            cols.append(int(c))
    return cols


def _room_slices(wall_rows: list[int], wall_cols: list[int]) -> dict[tuple[int, int], tuple[slice, slice]]:
    rooms = {}
    for rr in range(len(wall_rows) - 1):
        for cc in range(len(wall_cols) - 1):
            rs = slice(wall_rows[rr] + 1, wall_rows[rr + 1])
            cs = slice(wall_cols[cc] + 1, wall_cols[cc + 1])
            assert rs.start < rs.stop and cs.start < cs.stop, "rooms must have non-empty interiors"
            rooms[(rr, cc)] = (rs, cs)
    return rooms


def _parse_structure(
    grid: np.ndarray,
) -> tuple[int, list[int], list[int], dict[tuple[int, int], tuple[slice, slice]], dict[tuple[int, int], int], dict[tuple[int, int], set[tuple[int, int]]]]:
    bg = _most_common_non_black(grid)
    wall_rows = _find_wall_rows(grid)
    wall_cols = _find_wall_cols(grid)

    assert len(wall_rows) >= 2 and len(wall_cols) >= 2, "expected a walled room grid"
    assert wall_rows[0] == 0 and wall_rows[-1] == grid.shape[0] - 1
    assert wall_cols[0] == 0 and wall_cols[-1] == grid.shape[1] - 1

    rooms = _room_slices(wall_rows, wall_cols)
    source_rooms: dict[tuple[int, int], int] = {}
    adjacency = {room: set() for room in rooms}

    for room, (rs, cs) in rooms.items():
        interior = grid[rs, cs]
        assert np.all(interior != BLACK), "room interior must not contain wall pixels"
        source_mask = interior != bg
        if not np.any(source_mask):
            continue
        colors = np.unique(interior[source_mask])
        assert len(colors) == 1, "source room must contain a single source color"
        assert int(source_mask.sum()) == 1, "source room must contain exactly one source pixel"
        source_rooms[room] = int(colors[0])

    room_rows = len(wall_rows) - 1
    room_cols = len(wall_cols) - 1
    for rr in range(room_rows):
        for cc in range(room_cols - 1):
            sep_c = wall_cols[cc + 1]
            top = wall_rows[rr] + 1
            bottom = wall_rows[rr + 1]
            if np.any(grid[top:bottom, sep_c] != BLACK):
                left_room = (rr, cc)
                right_room = (rr, cc + 1)
                adjacency[left_room].add(right_room)
                adjacency[right_room].add(left_room)

    for rr in range(room_rows - 1):
        for cc in range(room_cols):
            sep_r = wall_rows[rr + 1]
            left = wall_cols[cc] + 1
            right = wall_cols[cc + 1]
            if np.any(grid[sep_r, left:right] != BLACK):
                upper_room = (rr, cc)
                lower_room = (rr + 1, cc)
                adjacency[upper_room].add(lower_room)
                adjacency[lower_room].add(upper_room)

    return bg, wall_rows, wall_cols, rooms, source_rooms, adjacency


def main(grid: np.ndarray) -> np.ndarray:
    """
    In a grid of rectangular rooms separated by black walls, color each source
    room with its source color and propagate source colors through one pipe hop.

    Empty rooms only consider directly adjacent source rooms:
    - one adjacent source color -> copy it
    - multiple adjacent sources with the same color -> copy it
    - multiple adjacent sources with different colors -> grey
    - no adjacent source rooms -> stay background
    """
    g = np.asarray(grid, dtype=int)
    bg, _wall_rows, _wall_cols, rooms, source_rooms, adjacency = _parse_structure(g)

    out = g.copy()
    for room, color in source_rooms.items():
        rs, cs = rooms[room]
        interior = out[rs, cs]
        interior[interior == bg] = color

    for room, (rs, cs) in rooms.items():
        if room in source_rooms:
            continue
        neighbor_colors = [source_rooms[nbr] for nbr in adjacency[room] if nbr in source_rooms]
        if not neighbor_colors:
            continue
        fill_color = neighbor_colors[0] if len(set(neighbor_colors)) == 1 else GREY
        interior = out[rs, cs]
        interior[interior == bg] = fill_color

    return out


def _wall_positions(sizes: list[int]) -> list[int]:
    positions = [0]
    cursor = 0
    for size in sizes:
        cursor += size + 1
        positions.append(cursor)
    return positions


def _room_bounds(
    wall_rows: list[int], wall_cols: list[int], room: tuple[int, int]
) -> tuple[int, int, int, int]:
    rr, cc = room
    top = wall_rows[rr] + 1
    bottom = wall_rows[rr + 1] - 1
    left = wall_cols[cc] + 1
    right = wall_cols[cc + 1] - 1
    return top, bottom, left, right


def _transform_coord(coord: tuple[int, int], op: int) -> tuple[int, int]:
    r, c = coord
    transforms = [
        lambda x, y: (x, y),
        lambda x, y: (y, 2 - x),
        lambda x, y: (2 - x, 2 - y),
        lambda x, y: (2 - y, x),
        lambda x, y: (x, 2 - y),
        lambda x, y: (2 - x, y),
        lambda x, y: (y, x),
        lambda x, y: (2 - y, 2 - x),
    ]
    return transforms[op](r, c)


def _canonical_edge(a: tuple[int, int], b: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    return (a, b) if a <= b else (b, a)


def _build_branch_block(offset_r: int, offset_c: int) -> tuple[dict[str, tuple[int, int]], set[tuple[tuple[int, int], tuple[int, int]]], set[tuple[int, int]]]:
    base_rooms = {
        "single_source": (0, 0),
        "single_target": (0, 1),
        "none_target": (0, 2),
        "same_source_a": (1, 0),
        "same_target": (1, 1),
        "same_source_b": (1, 2),
        "diff_source_a": (2, 0),
        "diff_target": (2, 1),
        "diff_source_b": (2, 2),
    }
    base_edges = {
        _canonical_edge((0, 0), (0, 1)),
        _canonical_edge((0, 1), (0, 2)),
        _canonical_edge((1, 0), (1, 1)),
        _canonical_edge((1, 1), (1, 2)),
        _canonical_edge((2, 0), (2, 1)),
        _canonical_edge((2, 1), (2, 2)),
    }
    op = random.randrange(8)
    rooms = {
        name: (
            offset_r + _transform_coord(coord, op)[0],
            offset_c + _transform_coord(coord, op)[1],
        )
        for name, coord in base_rooms.items()
    }
    edges = {
        _canonical_edge(
            (
                offset_r + _transform_coord(a, op)[0],
                offset_c + _transform_coord(a, op)[1],
            ),
            (
                offset_r + _transform_coord(b, op)[0],
                offset_c + _transform_coord(b, op)[1],
            ),
        )
        for a, b in base_edges
    }
    occupied = set(rooms.values())
    return rooms, edges, occupied


def _layout_output(
    grid: np.ndarray,
    wall_rows: list[int],
    wall_cols: list[int],
    bg: int,
    source_rooms: dict[tuple[int, int], int],
    pipes: set[tuple[tuple[int, int], tuple[int, int]]],
) -> np.ndarray:
    rooms = _room_slices(wall_rows, wall_cols)
    adjacency = {room: set() for room in rooms}
    for a, b in pipes:
        adjacency[a].add(b)
        adjacency[b].add(a)

    out = grid.copy()
    for room, color in source_rooms.items():
        rs, cs = rooms[room]
        interior = out[rs, cs]
        interior[interior == bg] = color

    for room, (rs, cs) in rooms.items():
        if room in source_rooms:
            continue
        colors = [source_rooms[nbr] for nbr in adjacency[room] if nbr in source_rooms]
        if not colors:
            continue
        fill_color = colors[0] if len(set(colors)) == 1 else GREY
        interior = out[rs, cs]
        interior[interior == bg] = fill_color

    return out


def _gen_attempt() -> np.ndarray | None:
    room_rows = random.randint(*ROOM_COUNT_RANGE)
    room_cols = random.randint(*ROOM_COUNT_RANGE)
    heights = [random.randint(*ROOM_SIZE_RANGE) for _ in range(room_rows)]
    widths = [random.randint(*ROOM_SIZE_RANGE) for _ in range(room_cols)]
    wall_rows = _wall_positions(heights)
    wall_cols = _wall_positions(widths)

    bg = random.choice(COLOR_POOL)
    remaining_colors = [c for c in COLOR_POOL if c != bg]
    if len(remaining_colors) < 4:
        return None
    single_color, same_color, diff_a_color, diff_b_color = random.sample(remaining_colors, 4)

    shape = (wall_rows[-1] + 1, wall_cols[-1] + 1)
    grid = np.full(shape, bg, dtype=int)
    grid[wall_rows, :] = BLACK
    grid[:, wall_cols] = BLACK

    block_r = random.randint(0, room_rows - 3)
    block_c = random.randint(0, room_cols - 3)
    branch_rooms, branch_edges, occupied_branch = _build_branch_block(block_r, block_c)

    source_rooms = {
        branch_rooms["single_source"]: single_color,
        branch_rooms["same_source_a"]: same_color,
        branch_rooms["same_source_b"]: same_color,
        branch_rooms["diff_source_a"]: diff_a_color,
        branch_rooms["diff_source_b"]: diff_b_color,
    }
    pipes = set(branch_edges)

    all_rooms = [(r, c) for r in range(room_rows) for c in range(room_cols)]
    free_rooms = [room for room in all_rooms if room not in occupied_branch]
    extra_source_count = min(len(free_rooms), random.randint(*EXTRA_SOURCE_RANGE))
    if extra_source_count:
        extra_rooms = random.sample(free_rooms, extra_source_count)
        source_palette = [c for c in remaining_colors]
        for room in extra_rooms:
            source_rooms[room] = random.choice(source_palette)
        free_rooms = [room for room in free_rooms if room not in extra_rooms]

    candidate_extra_edges = []
    free_set = set(free_rooms)
    for r, c in free_rooms:
        for dr, dc in ((1, 0), (0, 1)):
            nr, nc = r + dr, c + dc
            nbr = (nr, nc)
            if nbr in free_set:
                candidate_extra_edges.append(_canonical_edge((r, c), nbr))
    if candidate_extra_edges:
        extra_pipe_count = min(len(candidate_extra_edges), random.randint(*EXTRA_PIPE_RANGE))
        for edge in random.sample(candidate_extra_edges, extra_pipe_count):
            pipes.add(edge)

    for room, color in source_rooms.items():
        top, bottom, left, right = _room_bounds(wall_rows, wall_cols, room)
        sr = random.randint(top, bottom)
        sc = random.randint(left, right)
        grid[sr, sc] = color

    for a, b in pipes:
        if a[0] == b[0]:
            rr = a[0]
            left_room = a if a[1] < b[1] else b
            sep_c = wall_cols[left_room[1] + 1]
            top = wall_rows[rr] + 1
            bottom = wall_rows[rr + 1] - 1
            gap_r = random.randint(top, bottom)
            grid[gap_r, sep_c] = bg
        else:
            cc = a[1]
            upper_room = a if a[0] < b[0] else b
            sep_r = wall_rows[upper_room[0] + 1]
            left = wall_cols[cc] + 1
            right = wall_cols[cc + 1] - 1
            gap_c = random.randint(left, right)
            grid[sep_r, gap_c] = bg

    expected = _layout_output(grid, wall_rows, wall_cols, bg, source_rooms, pipes)
    if not np.array_equal(main(grid), expected):
        return None

    rooms = _room_slices(wall_rows, wall_cols)
    branch_checks = {
        "single_target": single_color,
        "same_target": same_color,
        "diff_target": GREY,
        "none_target": bg,
    }
    for name, expected_color in branch_checks.items():
        rs, cs = rooms[branch_rooms[name]]
        interior = expected[rs, cs]
        actual_colors = set(map(int, np.unique(interior)))
        if expected_color == bg:
            if actual_colors != {bg}:
                return None
        else:
            if actual_colors != {expected_color}:
                return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid room-pipe propagation input")
