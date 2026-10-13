import random
from collections import Counter

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
YELLOW = 4
GREY = 5
FILTER_COLORS = (3, 6, 7, 8, 9)

NORTH = (-1, 0)
SOUTH = (1, 0)
WEST = (0, -1)
EAST = (0, 1)
DIRS = (NORTH, SOUTH, WEST, EAST)
ORTHO_DIRS = {
    NORTH: [EAST, WEST],
    SOUTH: [EAST, WEST],
    EAST: [NORTH, SOUTH],
    WEST: [NORTH, SOUTH],
}
OPPOSITE = {NORTH: SOUTH, SOUTH: NORTH, EAST: WEST, WEST: EAST}

# A 2x2 mirror block uses exactly one corner. Each corner pairs two outer
# attachment cells: one horizontal and one vertical.
CORNER_OFFSETS = [
    {WEST: (0, -1), NORTH: (-1, 0)},
    {EAST: (0, 2), NORTH: (-1, 1)},
    {WEST: (1, -1), SOUTH: (2, 0)},
    {EAST: (1, 2), SOUTH: (2, 1)},
]

GRID_MIN = 12
GRID_MAX = 24
MAX_GENERATION_ATTEMPTS = 400


def _add(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return (a[0] + b[0], a[1] + b[1])


def _sub(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return (a[0] - b[0], a[1] - b[1])


def _neighbors4(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _in_bounds(cell: tuple[int, int], shape: tuple[int, int]) -> bool:
    r, c = cell
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _in_inner(cell: tuple[int, int], shape: tuple[int, int]) -> bool:
    r, c = cell
    h, w = shape
    return 1 <= r <= h - 2 and 1 <= c <= w - 2


def _is_filter_color(value: int) -> bool:
    return value not in {BLACK, BLUE, RED, YELLOW, GREY}


def _is_channel_color(value: int) -> bool:
    return value == GREY or _is_filter_color(value)


def _segment_cells(
    start: tuple[int, int], direction: tuple[int, int], length: int
) -> list[tuple[int, int]]:
    dr, dc = direction
    return [(start[0] + dr * i, start[1] + dc * i) for i in range(length)]


def _block_cells(block: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = block
    return [(r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)]


def _turn_block(
    endpoint: tuple[int, int],
    in_dir: tuple[int, int],
    out_dir: tuple[int, int],
) -> tuple[tuple[int, int], tuple[int, int]]:
    entry_side = OPPOSITE[in_dir]
    for corner in CORNER_OFFSETS:
        if entry_side in corner and out_dir in corner:
            block = _sub(endpoint, corner[entry_side])
            next_start = _add(block, corner[out_dir])
            return block, next_start
    raise AssertionError("invalid turn geometry")


def _mirror_corner_pairs(block: tuple[int, int]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    r, c = block
    return [
        ((r, c - 1), (r - 1, c)),
        ((r, c + 2), (r - 1, c + 1)),
        ((r + 1, c - 1), (r + 2, c)),
        ((r + 1, c + 2), (r + 2, c + 1)),
    ]


def _count_paths(
    graph: dict[tuple[int, int], set[tuple[int, int]]],
    start: tuple[int, int],
    end: tuple[int, int],
    limit: int = 2,
) -> tuple[int, list[tuple[int, int]] | None]:
    count = 0
    found_path = None
    stack: list[tuple[tuple[int, int], list[tuple[int, int]]]] = [(start, [start])]
    while stack:
        node, path = stack.pop()
        if node == end:
            count += 1
            found_path = path
            if count >= limit:
                return count, None
            continue
        for neighbor in graph[node]:
            if neighbor in path:
                continue
            stack.append((neighbor, path + [neighbor]))
    return count, found_path


def _build_connectivity(
    grid: np.ndarray,
) -> tuple[
    dict[tuple[int, int], set[tuple[int, int]]],
    dict[tuple[int, int], set[tuple[int, int]]],
]:
    h, w = grid.shape
    channel_mask = (grid != BLACK) & (grid != BLUE) & (grid != RED) & (grid != YELLOW)
    channel_cells = {(int(r), int(c)) for r, c in np.argwhere(channel_mask)}
    graph = {cell: set() for cell in channel_cells}
    direct = {cell: set() for cell in channel_cells}

    for cell in channel_cells:
        for neighbor in _neighbors4(cell):
            if neighbor in channel_cells:
                graph[cell].add(neighbor)
                direct[cell].add(neighbor)

    blocks: list[tuple[int, int]] = []
    for r in range(h - 1):
        for c in range(w - 1):
            if np.all(grid[r : r + 2, c : c + 2] == YELLOW):
                blocks.append((r, c))

    cover = Counter()
    for block in blocks:
        for cell in _block_cells(block):
            cover[cell] += 1
    yellow_cells = {(int(r), int(c)) for r, c in np.argwhere(grid == YELLOW)}
    assert yellow_cells == set(cover), "yellow pixels must form disjoint 2x2 mirror blocks"
    assert all(v == 1 for v in cover.values()), "mirror blocks may not overlap"

    for block in blocks:
        used_pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
        attached_cells: set[tuple[int, int]] = set()
        for a, b in _mirror_corner_pairs(block):
            a_used = a in channel_cells
            b_used = b in channel_cells
            if a_used != b_used:
                raise AssertionError("mirror attachments must come in orthogonal pairs")
            if a_used:
                used_pairs.append((a, b))
                attached_cells.add(a)
                attached_cells.add(b)
                graph[a].add(b)
                graph[b].add(a)

        all_adjacent = {
            neighbor
            for cell in _block_cells(block)
            for neighbor in _neighbors4(cell)
            if neighbor in channel_cells
        }
        assert len(used_pairs) == 1, "each mirror must connect exactly one channel corner"
        assert all_adjacent == attached_cells, "mirror has unexpected channel contacts"

    return graph, direct


def _marker_channel_neighbor(
    grid: np.ndarray, marker_color: int
) -> tuple[tuple[int, int], tuple[int, int]]:
    h, w = grid.shape
    markers = [
        (int(r), int(c))
        for r, c in np.argwhere(grid == marker_color)
        if r in {0, h - 1} or c in {0, w - 1}
    ]
    assert len(markers) == 1, "expected exactly one edge marker of each role"
    marker = markers[0]
    neighbors = [
        cell
        for cell in _neighbors4(marker)
        if _in_bounds(cell, grid.shape) and _is_channel_color(int(grid[cell]))
    ]
    assert len(neighbors) == 1, "edge marker must touch exactly one channel endpoint"
    return marker, neighbors[0]


def _trace_signal_path(grid: np.ndarray) -> list[tuple[int, int]]:
    graph, direct = _build_connectivity(grid)
    _red_marker, start = _marker_channel_neighbor(grid, RED)
    _blue_marker, end = _marker_channel_neighbor(grid, BLUE)

    for cell, neighbors in direct.items():
        assert len(neighbors) <= 2, "straight channels cannot branch directly"
        if len(neighbors) == 2:
            rows = {cell[0]} | {pt[0] for pt in neighbors}
            cols = {cell[1]} | {pt[1] for pt in neighbors}
            assert len(rows) == 1 or len(cols) == 1, "direct bends must use a mirror block"

    for neighbors in graph.values():
        assert len(neighbors) <= 2, "the channel network must be a collection of simple paths"

    path_count, path = _count_paths(graph, start, end)
    assert path_count == 1 and path is not None, "entry and exit must have a unique route"
    return path


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate the entry signal through a mirror-guided channel network and recolor only the traversed route.

    Input structure:
    - The background is black.
    - Grey one-cell-wide horizontal and vertical channel segments are connected by yellow 2x2 mirror blocks.
    - Some channel cells are single non-grey, non-black, non-yellow filter pixels that change the signal color.
    - A red edge pixel marks the entry signal and a blue edge pixel marks a different channel endpoint as the exit.

    Rule:
    - Follow the unique channel route from the red marker's adjacent endpoint to the blue marker's adjacent endpoint.
    - Grey cells on that route are recolored to the current signal color as the signal advances.
    - When the route passes through a filter pixel, the signal color changes to that filter color for all later grey cells.
    - Yellow mirror blocks, the red entry marker, the blue exit marker, and untouched channels stay unchanged.

    Key insight:
    - The puzzle separates topology from color: first recover the unique entry-to-exit path through straight channels and mirror turns, then replay the path while applying filter-driven color changes in order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    path = _trace_signal_path(g)

    out = g.copy()
    signal = RED
    for r, c in path:
        value = int(out[r, c])
        if value == GREY:
            out[r, c] = signal
        elif _is_filter_color(value):
            signal = value
    return out


def _start_for_edge(
    shape: tuple[int, int], edge: str
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    h, w = shape
    if edge == "left":
        r = random.randint(2, h - 3)
        return (r, 1), (r, 0), EAST
    if edge == "right":
        r = random.randint(2, h - 3)
        return (r, w - 2), (r, w - 1), WEST
    if edge == "top":
        c = random.randint(2, w - 3)
        return (1, c), (0, c), SOUTH
    c = random.randint(2, w - 3)
    return (h - 2, c), (h - 1, c), NORTH


def _forced_exit(
    current: tuple[int, int],
    direction: tuple[int, int],
    shape: tuple[int, int],
) -> tuple[int, tuple[int, int]]:
    h, w = shape
    r, c = current
    if direction == EAST:
        endpoint = (r, w - 2)
    elif direction == WEST:
        endpoint = (r, 1)
    elif direction == SOUTH:
        endpoint = (h - 2, c)
    else:
        endpoint = (1, c)
    length = abs(endpoint[0] - r) + abs(endpoint[1] - c) + 1
    return length, _add(endpoint, direction)


def _build_main_path(
    shape: tuple[int, int], turns: int
) -> tuple[
    tuple[int, int],
    list[tuple[int, int]],
    list[tuple[int, int]],
    tuple[int, int],
] | None:
    edge = random.choice(["left", "right", "top", "bottom"])
    start, red_marker, start_dir = _start_for_edge(shape, edge)
    occupied_channel = {start}
    occupied_mirror: set[tuple[int, int]] = set()

    def dfs(
        current: tuple[int, int],
        direction: tuple[int, int],
        turns_left: int,
        path_cells: list[tuple[int, int]],
        mirrors: list[tuple[int, int]],
    ) -> tuple[list[tuple[int, int]], list[tuple[int, int]], tuple[int, int]] | None:
        if turns_left == 0:
            final_length, blue_marker = _forced_exit(current, direction, shape)
            if final_length < 2:
                return None
            segment = _segment_cells(current, direction, final_length)
            if any(not _in_inner(cell, shape) for cell in segment):
                return None
            body = segment[1:]
            if any(cell in occupied_channel or cell in occupied_mirror for cell in body):
                return None
            return path_cells + body, mirrors, blue_marker

        next_dirs = ORTHO_DIRS[direction][:]
        random.shuffle(next_dirs)
        lengths = list(range(2, 9))
        random.shuffle(lengths)
        for next_dir in next_dirs:
            for length in lengths:
                segment = _segment_cells(current, direction, length)
                endpoint = segment[-1]
                block, next_start = _turn_block(endpoint, direction, next_dir)
                mirror_cells = _block_cells(block)
                if any(not _in_inner(cell, shape) for cell in segment):
                    continue
                if any(not _in_inner(cell, shape) for cell in mirror_cells):
                    continue
                if not _in_inner(next_start, shape):
                    continue
                body = segment[1:]
                if any(cell in occupied_channel or cell in occupied_mirror for cell in body):
                    continue
                if any(cell in occupied_channel or cell in occupied_mirror for cell in mirror_cells):
                    continue
                if next_start in occupied_channel or next_start in occupied_mirror:
                    continue

                occupied_channel.update(body)
                occupied_channel.add(next_start)
                occupied_mirror.update(mirror_cells)
                result = dfs(
                    next_start,
                    next_dir,
                    turns_left - 1,
                    path_cells + body + [next_start],
                    mirrors + [block],
                )
                if result is not None:
                    return result
                occupied_channel.difference_update(body)
                occupied_channel.remove(next_start)
                occupied_mirror.difference_update(mirror_cells)
        return None

    result = dfs(start, start_dir, turns, [start], [])
    if result is None:
        return None
    path_cells, mirrors, blue_marker = result
    return red_marker, path_cells, mirrors, blue_marker


def _pick_filter_cells(cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not cells:
        return []
    max_count = min(4, len(cells))
    if max_count == 0:
        return []

    mode = random.choice(["none", "one", "many"])
    if mode == "none":
        target = 0
    elif mode == "one":
        target = 1
    else:
        target = random.randint(2, max_count) if max_count >= 2 else 1

    shuffled = cells[:]
    random.shuffle(shuffled)
    chosen: list[tuple[int, int]] = []
    for cell in shuffled:
        if all(abs(cell[0] - other[0]) + abs(cell[1] - other[1]) >= 3 for other in chosen):
            chosen.append(cell)
            if len(chosen) == target:
                break
    return chosen


def _paint_filters(grid: np.ndarray, cells: list[tuple[int, int]]) -> None:
    chosen = _pick_filter_cells(cells)
    if not chosen:
        return
    palette = list(FILTER_COLORS)
    random.shuffle(palette)
    for index, cell in enumerate(chosen):
        grid[cell] = palette[index % len(palette)]


def _candidate_is_clear(
    grid: np.ndarray,
    channel_cells: list[tuple[int, int]],
    mirror_blocks: list[tuple[int, int]],
) -> bool:
    channel_set = set(channel_cells)
    mirror_set = {cell for block in mirror_blocks for cell in _block_cells(block)}
    occupied = channel_set | mirror_set

    if any(not _in_inner(cell, grid.shape) for cell in occupied):
        return False
    if any(int(grid[cell]) != BLACK for cell in occupied):
        return False

    for cell in occupied:
        for neighbor in _neighbors4(cell):
            if not _in_bounds(neighbor, grid.shape):
                continue
            if neighbor in occupied:
                continue
            if int(grid[neighbor]) != BLACK:
                return False
    return True


def _place_decoy_segment(grid: np.ndarray) -> None:
    h, w = grid.shape
    direction = random.choice([EAST, SOUTH])
    length = random.randint(2, 6)
    for _ in range(80):
        if direction == EAST:
            r = random.randint(2, h - 3)
            c = random.randint(2, w - length - 2)
            cells = [(r, c + i) for i in range(length)]
        else:
            r = random.randint(2, h - length - 2)
            c = random.randint(2, w - 3)
            cells = [(r + i, c) for i in range(length)]
        if not _candidate_is_clear(grid, cells, []):
            continue
        for cell in cells:
            grid[cell] = GREY
        _paint_filters(grid, cells[1:-1])
        return


def _place_decoy_elbow(grid: np.ndarray) -> None:
    h, w = grid.shape
    for _ in range(120):
        start = (random.randint(2, h - 3), random.randint(2, w - 3))
        start_dir = random.choice(DIRS)
        next_dir = random.choice(ORTHO_DIRS[start_dir])
        len_a = random.randint(2, 5)
        len_b = random.randint(2, 5)
        first = _segment_cells(start, start_dir, len_a)
        endpoint = first[-1]
        block, second_start = _turn_block(endpoint, start_dir, next_dir)
        second = _segment_cells(second_start, next_dir, len_b)
        channels = first + second
        if len(set(channels)) != len(channels):
            continue
        if not _candidate_is_clear(grid, channels, [block]):
            continue
        for cell in channels:
            grid[cell] = GREY
        r, c = block
        grid[r : r + 2, c : c + 2] = YELLOW
        _paint_filters(grid, channels[1:-1])
        return


def _add_decoys(grid: np.ndarray) -> None:
    for _ in range(random.randint(0, 2)):
        if random.random() < 0.5:
            _place_decoy_segment(grid)
        else:
            _place_decoy_elbow(grid)


def _gen_attempt() -> np.ndarray | None:
    shape = (random.randint(GRID_MIN, GRID_MAX), random.randint(GRID_MIN, GRID_MAX))
    turns = random.randint(1, 4)
    built = _build_main_path(shape, turns)
    if built is None:
        return None

    red_marker, path_cells, mirrors, blue_marker = built
    grid = np.full(shape, BLACK, dtype=int)
    for cell in path_cells:
        grid[cell] = GREY
    for block in mirrors:
        r, c = block
        grid[r : r + 2, c : c + 2] = YELLOW
    grid[red_marker] = RED
    grid[blue_marker] = BLUE

    mirror_adjacent = {
        cell
        for block in mirrors
        for pair in _mirror_corner_pairs(block)
        for cell in pair
        if _in_bounds(cell, shape)
    }
    filter_candidates = [
        cell for cell in path_cells[1:-1] if cell not in mirror_adjacent and grid[cell] == GREY
    ]
    _paint_filters(grid, filter_candidates)
    _add_decoys(grid)

    try:
        out = main(grid)
    except AssertionError:
        return None
    if out.shape != grid.shape:
        return None
    if max(grid.shape) > 30 or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirror-channel input")
