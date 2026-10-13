from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _most_common_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _infer_palette(grid: np.ndarray) -> tuple[int, int, int]:
    counts = Counter(map(int, grid.ravel()))
    bg = _most_common_color(grid)

    singletons = [color for color, n in counts.items() if color != bg and n == 1]
    assert len(singletons) == 1, "expected exactly one singleton source color"
    source = int(singletons[0])

    candidates = [color for color in counts if color not in {bg, source}]
    assert candidates, "expected a separator color besides background and source"

    h, w = grid.shape
    scored = []
    for color in candidates:
        row_peak = max(int(np.count_nonzero(grid[r] == color)) for r in range(h))
        col_peak = max(int(np.count_nonzero(grid[:, c] == color)) for c in range(w))
        scored.append((max(row_peak, col_peak), row_peak + col_peak, counts[color], int(color)))

    scored.sort()
    sep = scored[-1][3]
    assert scored[-1][0] > max(2, min(h, w) // 2), "failed to identify separator color"
    return bg, sep, source


def _separator_indices(grid: np.ndarray, sep: int) -> tuple[list[int], list[int]]:
    h, w = grid.shape
    sep_rows = [r for r in range(h) if int(np.count_nonzero(grid[r] == sep)) > w // 2]
    sep_cols = [c for c in range(w) if int(np.count_nonzero(grid[:, c] == sep)) > h // 2]
    assert sep_rows and sep_cols, "expected both horizontal and vertical separators"
    return sep_rows, sep_cols


def _spans_from_cuts(size: int, cuts: list[int]) -> list[tuple[int, int]]:
    spans = []
    start = 0
    for cut in cuts:
        assert start <= cut - 1, "separator cuts produced an empty span"
        spans.append((start, cut - 1))
        start = cut + 1
    assert start <= size - 1, "separator cuts consumed the full axis"
    spans.append((start, size - 1))
    return spans


def _find_span_index(spans: list[tuple[int, int]], value: int) -> int:
    for idx, (lo, hi) in enumerate(spans):
        if lo <= value <= hi:
            return idx
    raise AssertionError("coordinate is not inside any room span")


def _room_graph(
    grid: np.ndarray,
    sep: int,
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
    sep_rows: list[int],
    sep_cols: list[int],
) -> dict[tuple[int, int], set[tuple[int, int]]]:
    rooms = {
        (i, j): set()
        for i in range(len(row_spans))
        for j in range(len(col_spans))
    }

    for i, cut_r in enumerate(sep_rows):
        for j, (c0, c1) in enumerate(col_spans):
            if np.any(grid[cut_r, c0 : c1 + 1] != sep):
                a = (i, j)
                b = (i + 1, j)
                rooms[a].add(b)
                rooms[b].add(a)

    for j, cut_c in enumerate(sep_cols):
        for i, (r0, r1) in enumerate(row_spans):
            if np.any(grid[r0 : r1 + 1, cut_c] != sep):
                a = (i, j)
                b = (i, j + 1)
                rooms[a].add(b)
                rooms[b].add(a)

    return rooms


def _reachable_rooms(
    graph: dict[tuple[int, int], set[tuple[int, int]]],
    start: tuple[int, int],
) -> set[tuple[int, int]]:
    seen = {start}
    q = deque([start])
    while q:
        node = q.popleft()
        for nxt in graph[node]:
            if nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return seen


def main(input_grid: np.ndarray) -> np.ndarray:
    """Output a room-by-room thumbnail colored by door reachability from the unique source pixel.

    Input structure:
    - A 2-D grid contains one separator color forming nearly full horizontal and vertical divider lines.
    - Single-cell gaps in those dividers act as doors between rectangular rooms, and exactly one room contains a single source-colored pixel.
    - Extra noise pixels of another color may appear inside rooms but never on divider cells.

    Rule:
    - Infer the room partition from the horizontal and vertical divider lines, and let the output size be the number of room bands by the number of room columns.
    - Treat every non-separator cell on a divider segment as an open door joining the two adjacent rooms across that divider.
    - Starting from the room containing the unique source pixel, mark every room reachable through doors with the source color, and leave all other output cells as the background color.

    Key insight:
    - The output is not a geometric downscale of the picture; it is a room-level connectivity summary where only the divider topology and the source room matter.
    """
    grid = np.asarray(input_grid, dtype=int)
    assert grid.ndim == 2

    bg, sep, source = _infer_palette(grid)
    sep_rows, sep_cols = _separator_indices(grid, sep)
    row_spans = _spans_from_cuts(grid.shape[0], sep_rows)
    col_spans = _spans_from_cuts(grid.shape[1], sep_cols)

    src_coords = np.argwhere(grid == source)
    assert src_coords.shape == (1, 2), "expected exactly one source pixel"
    src_r, src_c = map(int, src_coords[0])
    src_room = (
        _find_span_index(row_spans, src_r),
        _find_span_index(col_spans, src_c),
    )

    graph = _room_graph(grid, sep, row_spans, col_spans, sep_rows, sep_cols)
    reachable = _reachable_rooms(graph, src_room)

    out = np.full((len(row_spans), len(col_spans)), bg, dtype=int)
    for r, c in reachable:
        out[r, c] = source
    return out


_COLORS = tuple(range(10))
_MIN_ROOM_SIZE = 3
_MAX_ROOM_SIZE = 6
_MAX_TRIES = 400


def _random_room_sizes(count: int) -> list[int]:
    for _ in range(100):
        sizes = [random.randint(_MIN_ROOM_SIZE, _MAX_ROOM_SIZE) for _ in range(count)]
        if sum(sizes) + (count - 1) <= 30:
            return sizes
    raise RuntimeError("failed to sample room sizes within ARC bounds")


def _separator_positions(sizes: list[int]) -> list[int]:
    pos = []
    cursor = sizes[0]
    for size in sizes[1:]:
        pos.append(cursor)
        cursor += 1 + size
    return pos


def _sample_door_arrays(rows: int, cols: int) -> tuple[np.ndarray, np.ndarray]:
    h_doors = np.zeros((rows - 1, cols), dtype=bool)
    v_doors = np.zeros((rows, cols - 1), dtype=bool)

    for i in range(rows - 1):
        max_open = max(1, min(cols - 1 if cols > 1 else 1, 3))
        open_count = random.randint(1, max_open)
        for j in random.sample(range(cols), open_count):
            h_doors[i, j] = True

    for j in range(cols - 1):
        max_open = max(1, min(rows - 1 if rows > 1 else 1, 3))
        open_count = random.randint(1, max_open)
        for i in random.sample(range(rows), open_count):
            v_doors[i, j] = True

    return h_doors, v_doors


def _graph_from_doors(
    rows: int,
    cols: int,
    h_doors: np.ndarray,
    v_doors: np.ndarray,
) -> dict[tuple[int, int], set[tuple[int, int]]]:
    graph = {(i, j): set() for i in range(rows) for j in range(cols)}
    for i in range(rows - 1):
        for j in range(cols):
            if h_doors[i, j]:
                a = (i, j)
                b = (i + 1, j)
                graph[a].add(b)
                graph[b].add(a)
    for i in range(rows):
        for j in range(cols - 1):
            if v_doors[i, j]:
                a = (i, j)
                b = (i, j + 1)
                graph[a].add(b)
                graph[b].add(a)
    return graph


def _door_layout_is_rich(
    rows: int,
    cols: int,
    source_room: tuple[int, int],
    h_doors: np.ndarray,
    v_doors: np.ndarray,
) -> tuple[bool, set[tuple[int, int]]]:
    graph = _graph_from_doors(rows, cols, h_doors, v_doors)
    reachable = _reachable_rooms(graph, source_room)
    total = rows * cols

    if not (3 <= len(reachable) <= total - 1):
        return False, reachable

    if not any(r != source_room[0] and c != source_room[1] for r, c in reachable):
        return False, reachable

    closed_frontier = False
    for r in range(rows):
        for c in range(cols):
            here = (r, c)
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < rows and 0 <= nc < cols:
                    there = (nr, nc)
                    if (here in reachable) ^ (there in reachable):
                        if there not in graph[here]:
                            closed_frontier = True
                            break
            if closed_frontier:
                break
        if closed_frontier:
            break

    return closed_frontier, reachable


def _pick_room_cell(
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
    room: tuple[int, int],
    used: set[tuple[int, int]],
) -> tuple[int, int]:
    r0, r1 = row_spans[room[0]]
    c0, c1 = col_spans[room[1]]
    cells = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1) if (r, c) not in used]
    assert cells, "expected a free cell inside the room"
    return random.choice(cells)


def _build_case(return_meta: bool = False):
    rows = random.randint(2, 5)
    cols = random.randint(2, 5)
    if rows * cols < 6:
        rows, cols = random.choice([(2, 3), (3, 2), (2, 4), (4, 2), (3, 3)])

    row_sizes = _random_room_sizes(rows)
    col_sizes = _random_room_sizes(cols)
    h = sum(row_sizes) + rows - 1
    w = sum(col_sizes) + cols - 1
    assert h <= 30 and w <= 30

    bg, sep, source, noise = random.sample(_COLORS, 4)
    grid = np.full((h, w), bg, dtype=int)

    sep_rows = _separator_positions(row_sizes)
    sep_cols = _separator_positions(col_sizes)
    row_spans = _spans_from_cuts(h, sep_rows)
    col_spans = _spans_from_cuts(w, sep_cols)

    for r in sep_rows:
        grid[r, :] = sep
    for c in sep_cols:
        grid[:, c] = sep

    source_room = (random.randrange(rows), random.randrange(cols))
    for _ in range(120):
        h_doors, v_doors = _sample_door_arrays(rows, cols)
        ok, reachable = _door_layout_is_rich(rows, cols, source_room, h_doors, v_doors)
        if ok:
            break
    else:
        raise RuntimeError("failed to sample a rich door layout")

    for i, cut_r in enumerate(sep_rows):
        for j, is_open in enumerate(h_doors[i]):
            if is_open:
                c0, c1 = col_spans[j]
                grid[cut_r, random.randint(c0, c1)] = bg

    for i in range(rows):
        for j, cut_c in enumerate(sep_cols):
            if v_doors[i, j]:
                r0, r1 = row_spans[i]
                grid[random.randint(r0, r1), cut_c] = bg

    used = set()
    src_cell = _pick_room_cell(row_spans, col_spans, source_room, used)
    used.add(src_cell)
    grid[src_cell] = source

    all_rooms = [(i, j) for i in range(rows) for j in range(cols)]
    reachable_others = [room for room in all_rooms if room in reachable and room != source_room]
    blocked_rooms = [room for room in all_rooms if room not in reachable]
    noise_rooms = []
    if reachable_others:
        noise_rooms.append(random.choice(reachable_others))
    if blocked_rooms:
        noise_rooms.append(random.choice(blocked_rooms))
    if random.random() < 0.5:
        noise_rooms.append(source_room)

    remaining_rooms = [room for room in all_rooms if room not in noise_rooms]
    random.shuffle(remaining_rooms)
    extra_count = random.randint(0, min(3, len(remaining_rooms)))
    noise_rooms.extend(remaining_rooms[:extra_count])
    noise_rooms = list(dict.fromkeys(noise_rooms))

    noise_pixels = 0
    for room in noise_rooms:
        repeats = random.randint(1, 2)
        for _ in range(repeats):
            cell = _pick_room_cell(row_spans, col_spans, room, used)
            used.add(cell)
            grid[cell] = noise
            noise_pixels += 1
    if noise_pixels < 2:
        room = random.choice(noise_rooms or blocked_rooms or reachable_others or [source_room])
        cell = _pick_room_cell(row_spans, col_spans, room, used)
        used.add(cell)
        grid[cell] = noise
        noise_pixels += 1

    expected = np.full((rows, cols), bg, dtype=int)
    for room in reachable:
        expected[room] = source

    if return_meta:
        return grid, {
            "bg": bg,
            "sep": sep,
            "source": source,
            "noise": noise,
            "row_sizes": row_sizes,
            "col_sizes": col_sizes,
            "source_room": source_room,
            "reachable": reachable,
            "expected": expected,
        }
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_TRIES):
        grid, meta = _build_case(return_meta=True)
        try:
            out = main(grid)
        except AssertionError:
            continue
        if np.array_equal(out, meta["expected"]):
            return grid
    raise RuntimeError("failed to generate a valid input")
