import random
from collections import defaultdict

import numpy as np


BG = 0
LINE = 5
CONFLICT = 9
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8)
GEN_ATTEMPTS = 400


def _find_separator_color(grid: np.ndarray) -> tuple[int, list[int], list[int]]:
    h, w = grid.shape
    best: tuple[tuple[int, int], int, list[int], list[int]] | None = None
    for color in np.unique(grid):
        color = int(color)
        if color == BG:
            continue
        sep_rows = [r for r in range(h) if int(np.count_nonzero(grid[r] == color)) > w // 2]
        sep_cols = [c for c in range(w) if int(np.count_nonzero(grid[:, c] == color)) > h // 2]
        if len(sep_rows) < 2 or len(sep_cols) < 2:
            continue
        score = (len(sep_rows) + len(sep_cols), int(np.count_nonzero(grid == color)))
        if best is None or score > best[0]:
            best = (score, color, sep_rows, sep_cols)
    assert best is not None, "failed to identify separator color"
    _, color, sep_rows, sep_cols = best
    return color, sep_rows, sep_cols


def _room_bounds(sep_rows: list[int], sep_cols: list[int]) -> list[tuple[int, int, int, int]]:
    rooms: list[tuple[int, int, int, int]] = []
    for i in range(len(sep_rows) - 1):
        r0 = sep_rows[i] + 1
        r1 = sep_rows[i + 1]
        assert r0 < r1
        for j in range(len(sep_cols) - 1):
            c0 = sep_cols[j] + 1
            c1 = sep_cols[j + 1]
            assert c0 < c1
            rooms.append((r0, r1, c0, c1))
    return rooms


def _parse_grid(
    grid: np.ndarray,
) -> tuple[int, list[tuple[int, int, int, int]], list[set[int]], dict[int, int], int, int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = map(int, g.shape)

    line_color, sep_rows, sep_cols = _find_separator_color(g)
    assert sep_rows[0] == 0 and sep_rows[-1] == h - 1
    assert sep_cols[0] == 0 and sep_cols[-1] == w - 1
    assert all(b - a >= 2 for a, b in zip(sep_rows, sep_rows[1:]))
    assert all(b - a >= 2 for a, b in zip(sep_cols, sep_cols[1:]))

    room_rows = len(sep_rows) - 1
    room_cols = len(sep_cols) - 1
    rooms = _room_bounds(sep_rows, sep_cols)
    adjacency = [set() for _ in rooms]
    seeds: dict[int, int] = {}

    for i in range(room_rows):
        for j in range(room_cols):
            room_id = i * room_cols + j
            r0, r1, c0, c1 = rooms[room_id]
            patch = g[r0:r1, c0:c1]
            seed_mask = (patch != BG) & (patch != line_color)
            seed_pixels = np.argwhere(seed_mask)
            assert len(seed_pixels) <= 1, "each room may contain at most one seed"
            if len(seed_pixels) == 1:
                sr, sc = map(int, seed_pixels[0])
                seeds[room_id] = int(patch[sr, sc])

            if j + 1 < room_cols:
                wall_col = sep_cols[j + 1]
                wall_segment = g[r0:r1, wall_col]
                assert np.all((wall_segment == BG) | (wall_segment == line_color))
                if bool(np.any(wall_segment == BG)):
                    other_id = room_id + 1
                    adjacency[room_id].add(other_id)
                    adjacency[other_id].add(room_id)

            if i + 1 < room_rows:
                wall_row = sep_rows[i + 1]
                wall_segment = g[wall_row, c0:c1]
                assert np.all((wall_segment == BG) | (wall_segment == line_color))
                if bool(np.any(wall_segment == BG)):
                    other_id = room_id + room_cols
                    adjacency[room_id].add(other_id)
                    adjacency[other_id].add(room_id)

    assert seeds, "expected at least one seed room"
    return line_color, rooms, adjacency, seeds, room_rows, room_cols


def _propagate(adjacency: list[set[int]], seeds: dict[int, int]) -> list[int | None]:
    labels: list[int | None] = [None] * len(adjacency)
    frontier: list[int] = []
    for room_id, color in seeds.items():
        labels[room_id] = int(color)
        frontier.append(int(room_id))

    while frontier:
        arrivals: dict[int, set[int]] = defaultdict(set)
        for room_id in frontier:
            color = int(labels[room_id])
            for neighbor in adjacency[room_id]:
                if labels[neighbor] is None:
                    arrivals[neighbor].add(color)

        frontier = []
        for room_id, colors in arrivals.items():
            labels[room_id] = next(iter(colors)) if len(colors) == 1 else CONFLICT
            frontier.append(int(room_id))

    return labels


def _paint_rooms(
    grid: np.ndarray, rooms: list[tuple[int, int, int, int]], labels: list[int | None]
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for room_id, color in enumerate(labels):
        if color is None:
            continue
        r0, r1, c0, c1 = rooms[room_id]
        patch = out[r0:r1, c0:c1]
        patch[patch == BG] = int(color)
    return out


def _solve(grid: np.ndarray) -> np.ndarray:
    _line_color, rooms, adjacency, seeds, _room_rows, _room_cols = _parse_grid(grid)
    labels = _propagate(adjacency, seeds)
    return _paint_rooms(grid, rooms, labels)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each table-like room by synchronously propagating seed colors through one-pixel doors, using maroon for same-round conflicts.

    Input structure:
    - The background is black, and grey one-pixel separator rows and columns form
      a rectangular room grid with a grey outer border.
    - Some separator segments contain one-pixel black gaps that act as doors
      between two adjacent rooms.
    - A subset of rooms contains exactly one isolated non-black, non-grey seed
      pixel; every other room interior is otherwise black.

    Rule:
    - Reconstruct the room rectangles from the long grey separator rows and columns.
    - Build a room-adjacency graph by detecting which shared walls contain black
      door pixels.
    - Mark all seed rooms in the first round, then propagate colors one graph step
      per round into still-empty neighboring rooms.
    - If an empty room receives two different colors in the same round, mark that
      room maroon; marked rooms never change, and maroon rooms keep propagating.
    - In the output, fill every black interior pixel of each reached room with its
      final room color, while preserving seed pixels, grey separators, and door gaps.

    Key insight:
    - The task is solved at the room-graph level: first infer rooms and doors from
      the separator lattice, then run a synchronous multi-source BFS with tie-based
      conflict rooms instead of pixel-wise flood fill.
    """
    return _solve(np.asarray(grid, dtype=int))


def _random_lengths(count: int, max_total: int) -> list[int] | None:
    for _ in range(100):
        lengths = [random.randint(2, 6) for _ in range(count)]
        if sum(lengths) + count + 1 <= max_total:
            return lengths
    return None


def _build_layout():
    room_rows = random.randint(2, 4)
    room_cols = random.randint(2, 4)
    heights = _random_lengths(room_rows, 30)
    widths = _random_lengths(room_cols, 30)
    if heights is None or widths is None:
        return None

    sep_rows = [0]
    for height in heights:
        sep_rows.append(sep_rows[-1] + height + 1)
    sep_cols = [0]
    for width in widths:
        sep_cols.append(sep_cols[-1] + width + 1)

    h = sep_rows[-1] + 1
    w = sep_cols[-1] + 1
    rooms = _room_bounds(sep_rows, sep_cols)
    return room_rows, room_cols, sep_rows, sep_cols, rooms, h, w


def _build_walls(room_rows: int, room_cols: int, sep_rows: list[int], sep_cols: list[int]):
    walls = []
    for i in range(room_rows):
        r0 = sep_rows[i] + 1
        r1 = sep_rows[i + 1]
        for j in range(room_cols - 1):
            walls.append(
                {
                    "kind": "v",
                    "a": i * room_cols + j,
                    "b": i * room_cols + j + 1,
                    "pos": sep_cols[j + 1],
                    "start": r0,
                    "stop": r1,
                }
            )
    for i in range(room_rows - 1):
        wall_row = sep_rows[i + 1]
        for j in range(room_cols):
            c0 = sep_cols[j] + 1
            c1 = sep_cols[j + 1]
            walls.append(
                {
                    "kind": "h",
                    "a": i * room_cols + j,
                    "b": (i + 1) * room_cols + j,
                    "pos": wall_row,
                    "start": c0,
                    "stop": c1,
                }
            )
    return walls


def _mode_seed_colors(mode: str, count: int) -> list[int]:
    if mode == "conflict":
        colors = random.sample(SEED_COLORS, 2)
        while len(colors) < count:
            colors.append(random.choice(SEED_COLORS))
        random.shuffle(colors)
        if len(set(colors)) < 2:
            colors[0], colors[-1] = random.sample(SEED_COLORS, 2)
        return colors

    if random.random() < 0.7:
        color = random.choice(SEED_COLORS)
        return [color] * count

    return [random.choice(SEED_COLORS) for _ in range(count)]


def _room_graph(
    room_count: int, walls: list[dict], mode: str
) -> tuple[list[set[int]], list[dict]] | None:
    if mode == "conflict":
        door_prob = random.uniform(0.45, 0.8)
    elif mode == "disconnected":
        door_prob = random.uniform(0.15, 0.45)
    else:
        door_prob = random.uniform(0.3, 0.7)

    adjacency = [set() for _ in range(room_count)]
    open_walls = []
    for wall in walls:
        if random.random() < door_prob:
            open_walls.append(wall)
            a = int(wall["a"])
            b = int(wall["b"])
            adjacency[a].add(b)
            adjacency[b].add(a)
    if not open_walls:
        return None
    return adjacency, open_walls


def _render_input(
    rooms: list[tuple[int, int, int, int]],
    sep_rows: list[int],
    sep_cols: list[int],
    h: int,
    w: int,
    seeds: dict[int, int],
    open_walls: list[dict],
) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    grid[sep_rows, :] = LINE
    grid[:, sep_cols] = LINE

    for wall in open_walls:
        if wall["kind"] == "v":
            door_r = random.randint(int(wall["start"]), int(wall["stop"]) - 1)
            grid[door_r, int(wall["pos"])] = BG
        else:
            door_c = random.randint(int(wall["start"]), int(wall["stop"]) - 1)
            grid[int(wall["pos"]), door_c] = BG

    for room_id, color in seeds.items():
        r0, r1, c0, c1 = rooms[room_id]
        sr = random.randint(r0, r1 - 1)
        sc = random.randint(c0, c1 - 1)
        grid[sr, sc] = int(color)

    return grid


def _acceptable(mode: str, labels: list[int | None], seeds: dict[int, int]) -> bool:
    reached = sum(color is not None for color in labels)
    has_growth = any(labels[i] is not None and i not in seeds for i in range(len(labels)))
    has_conflict = any(color == CONFLICT for color in labels)
    has_unreached = reached < len(labels)

    if mode == "conflict":
        return has_growth and has_conflict
    if mode == "disconnected":
        return has_growth and has_unreached and not has_conflict
    return has_growth and not has_conflict


def _gen_attempt() -> np.ndarray | None:
    mode = random.choice(("conflict", "no_conflict", "disconnected"))
    layout = _build_layout()
    if layout is None:
        return None
    room_rows, room_cols, sep_rows, sep_cols, rooms, h, w = layout
    room_count = len(rooms)

    walls = _build_walls(room_rows, room_cols, sep_rows, sep_cols)
    graph = _room_graph(room_count, walls, mode)
    if graph is None:
        return None
    adjacency, open_walls = graph

    max_seed_count = min(4, room_count - 1)
    if max_seed_count < 1:
        return None
    if mode == "conflict":
        seed_count = random.randint(2, max_seed_count)
    elif mode == "no_conflict" and random.random() < 0.45:
        seed_count = 1
    else:
        seed_count = random.randint(1, min(3, max_seed_count))

    seed_rooms = random.sample(range(room_count), seed_count)
    seed_colors = _mode_seed_colors(mode, seed_count)
    if mode == "conflict" and len(set(seed_colors)) < 2:
        return None
    seeds = {room_id: color for room_id, color in zip(seed_rooms, seed_colors)}

    labels = _propagate(adjacency, seeds)
    if not _acceptable(mode, labels, seeds):
        return None

    grid = _render_input(rooms, sep_rows, sep_cols, h, w, seeds, open_walls)
    expected = _paint_rooms(grid, rooms, labels)
    if not np.array_equal(_solve(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
