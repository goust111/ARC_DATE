from __future__ import annotations

from collections import Counter, deque
from pathlib import Path
import random
import sys

import numpy as np

_ARC_ROOT = Path("/home/lixinyu/snap/new")
if str(_ARC_ROOT) not in sys.path:
    sys.path.insert(0, str(_ARC_ROOT))

from solvers.arc_utils import get_bg


GEN_RETRIES = 500
STATE_NONE = "none"
STATE_OPEN = "open"
STATE_BLOCKED = "blocked"


def _partition_ranges(size: int, separators: list[int]) -> list[tuple[int, int]]:
    bounds = [-1] + list(separators) + [size]
    parts: list[tuple[int, int]] = []
    for i in range(len(bounds) - 1):
        start = bounds[i] + 1
        end = bounds[i + 1] - 1
        assert start <= end
        parts.append((start, end))
    return parts


def _room_of_cell(r: int, c: int, row_ranges: list[tuple[int, int]], col_ranges: list[tuple[int, int]]) -> tuple[int, int]:
    room_r = next(i for i, (r0, r1) in enumerate(row_ranges) if r0 <= r <= r1)
    room_c = next(j for j, (c0, c1) in enumerate(col_ranges) if c0 <= c <= c1)
    return room_r, room_c


def _bfs(adj: dict[tuple[int, int], set[tuple[int, int]]], start: tuple[int, int]) -> dict[tuple[int, int], int]:
    dist = {start: 0}
    q: deque[tuple[int, int]] = deque([start])
    while q:
        cur = q.popleft()
        for nxt in adj[cur]:
            if nxt not in dist:
                dist[nxt] = dist[cur] + 1
                q.append(nxt)
    return dist


def _fill_rooms(
    grid: np.ndarray,
    row_ranges: list[tuple[int, int]],
    col_ranges: list[tuple[int, int]],
    dist: dict[tuple[int, int], int],
    even_color: int,
    odd_color: int,
) -> np.ndarray:
    out = np.array(grid, copy=True, dtype=int)
    for room_r, (r0, r1) in enumerate(row_ranges):
        for room_c, (c0, c1) in enumerate(col_ranges):
            room = (room_r, room_c)
            if room not in dist:
                continue
            fill = even_color if dist[room] % 2 == 0 else odd_color
            out[r0 : r1 + 1, c0 : c1 + 1] = fill
    return out


def _find_separator_layout(grid: np.ndarray, bg: int) -> tuple[int, list[int], list[int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    colors = [int(c) for c in np.unique(g) if int(c) != bg]
    best: tuple[int, list[int], list[int], int] | None = None

    for color in colors:
        mask = g == color
        row_counts = mask.sum(axis=1)
        row_base = int(np.median(row_counts))
        row_seps = [int(r) for r in np.flatnonzero(row_counts > row_base)]
        col_seps: list[int] = []

        for _ in range(3):
            keep_rows = [r for r in range(h) if r not in set(row_seps)]
            col_counts = mask[keep_rows, :].sum(axis=0) if keep_rows else np.zeros(w, dtype=int)
            col_base = int(np.median(col_counts))
            col_seps = [int(c) for c in np.flatnonzero(col_counts > col_base)]
            if not row_seps or not col_seps:
                break

            keep_cols = [c for c in range(w) if c not in set(col_seps)]
            row_counts = mask[:, keep_cols].sum(axis=1) if keep_cols else np.zeros(h, dtype=int)
            row_base = int(np.median(row_counts))
            row_seps = [int(r) for r in np.flatnonzero(row_counts > row_base)]

        if not row_seps or not col_seps or len(row_seps) >= h or len(col_seps) >= w:
            continue

        row_set = set(row_seps)
        col_set = set(col_seps)
        coords = np.argwhere(mask)
        if any(int(r) not in row_set and int(c) not in col_set for r, c in coords):
            continue
        if any(g[r, c] != color for r in row_seps for c in col_seps):
            continue

        row_counts_core = mask[:, [c for c in range(w) if c not in col_set]].sum(axis=1)
        col_counts_core = mask[[r for r in range(h) if r not in row_set], :].sum(axis=0)
        other_row_max = max((int(row_counts_core[r]) for r in range(h) if r not in row_set), default=-1)
        other_col_max = max((int(col_counts_core[c]) for c in range(w) if c not in col_set), default=-1)
        if min(int(row_counts_core[r]) for r in row_seps) <= other_row_max:
            continue
        if min(int(col_counts_core[c]) for c in col_seps) <= other_col_max:
            continue

        score = int(mask.sum())
        if best is None or score > best[3]:
            best = (color, row_seps, col_seps, score)

    assert best is not None, "failed to identify separator color"
    return best[0], best[1], best[2]


def _parse_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    bg = int(get_bg(g))
    sep, row_seps, col_seps = _find_separator_layout(g, bg)
    row_ranges = _partition_ranges(h, row_seps)
    col_ranges = _partition_ranges(w, col_seps)

    counts = Counter(map(int, g.flat))
    singleton_seeds: list[tuple[tuple[int, int], int]] = []
    for color, count in sorted(counts.items()):
        if color in (bg, sep) or count != 1:
            continue
        pos = tuple(map(int, np.argwhere(g == color)[0]))
        singleton_seeds.append((pos, int(color)))
    assert len(singleton_seeds) == 2, "expected exactly two singleton seed colors"
    singleton_seeds.sort(key=lambda item: item[0])
    (seed_a_pos, even_color), (seed_b_pos, odd_color) = singleton_seeds
    assert abs(seed_a_pos[0] - seed_b_pos[0]) + abs(seed_a_pos[1] - seed_b_pos[1]) == 1

    seed_room = _room_of_cell(seed_a_pos[0], seed_a_pos[1], row_ranges, col_ranges)
    assert seed_room == _room_of_cell(seed_b_pos[0], seed_b_pos[1], row_ranges, col_ranges)

    rooms = [(ri, cj) for ri in range(len(row_ranges)) for cj in range(len(col_ranges))]
    adj = {room: set() for room in rooms}
    blocked_edges: set[frozenset[tuple[int, int]]] = set()

    for i, sr in enumerate(row_seps):
        for j, (c0, c1) in enumerate(col_ranges):
            special = [int(g[sr, c]) for c in range(c0, c1 + 1) if int(g[sr, c]) != sep]
            assert len(special) <= 1
            if not special:
                continue
            room_u = (i, j)
            room_v = (i + 1, j)
            if special[0] == bg:
                adj[room_u].add(room_v)
                adj[room_v].add(room_u)
            else:
                blocked_edges.add(frozenset((room_u, room_v)))

    for j, sc in enumerate(col_seps):
        for i, (r0, r1) in enumerate(row_ranges):
            special = [int(g[r, sc]) for r in range(r0, r1 + 1) if int(g[r, sc]) != sep]
            assert len(special) <= 1
            if not special:
                continue
            room_u = (i, j)
            room_v = (i, j + 1)
            if special[0] == bg:
                adj[room_u].add(room_v)
                adj[room_v].add(room_u)
            else:
                blocked_edges.add(frozenset((room_u, room_v)))

    dist = _bfs(adj, seed_room)
    return {
        "bg": bg,
        "sep": sep,
        "row_ranges": row_ranges,
        "col_ranges": col_ranges,
        "seed_room": seed_room,
        "even_color": even_color,
        "odd_color": odd_color,
        "adj": adj,
        "blocked_edges": blocked_edges,
        "dist": dist,
    }


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Color whole rooms by parity of door-count distance from the seed room.

    Input structure:
    - One color forms full separator rows and columns that partition the canvas into rectangular rooms; some separator segments contain a single-cell door opening.
    - Some door openings are occupied by obstacle pixels, which visibly block that doorway.
    - Exactly one room contains two adjacent singleton seed pixels of different colors; in row-major order the first seed is color A and the second seed is color B.

    Rule:
    - Treat each room as a node and each unblocked doorway as an edge; a blocked doorway and a separator segment with no opening are both walls.
    - Compute the minimum number of door crossings from the seed room to every reachable room.
    - Fill each reachable room uniformly with color A for even distance and color B for odd distance; unreachable rooms stay background.
    - Preserve separator lines, open door cells, and obstacle pixels; the two seed pixels are absorbed into their room fill.

    Key insight:
    - The alternation happens on the room graph rather than on per-pixel paths: room size and door position do not matter, only reachability and the parity of door crossings.
    """
    parsed = _parse_grid(np.asarray(input_grid, dtype=int))
    return _fill_rooms(
        np.asarray(input_grid, dtype=int),
        parsed["row_ranges"],
        parsed["col_ranges"],
        parsed["dist"],
        parsed["even_color"],
        parsed["odd_color"],
    )


def _sample_spans(count: int, limit: int) -> list[int]:
    while True:
        spans = [random.randint(2, 6) for _ in range(count)]
        if sum(spans) + count - 1 <= limit:
            return spans


def _render_case(case: dict) -> np.ndarray:
    bg = case["bg"]
    sep = case["sep"]
    obstacle = case["obstacle"]
    grid = np.full((case["height"], case["width"]), bg, dtype=int)

    for r in case["row_seps"]:
        grid[r, :] = sep
    for c in case["col_seps"]:
        grid[:, c] = sep

    for (i, j), door in case["h_doors"].items():
        if door["state"] == STATE_NONE:
            continue
        rr = case["row_seps"][i]
        cc = door["coord"]
        grid[rr, cc] = bg if door["state"] == STATE_OPEN else obstacle

    for (i, j), door in case["v_doors"].items():
        if door["state"] == STATE_NONE:
            continue
        rr = door["coord"]
        cc = case["col_seps"][j]
        grid[rr, cc] = bg if door["state"] == STATE_OPEN else obstacle

    (a_pos, a_color), (b_pos, b_color) = case["seeds"]
    grid[a_pos] = a_color
    grid[b_pos] = b_color
    return grid


def _build_case_graph(case: dict, open_blocked: bool = False) -> dict[tuple[int, int], set[tuple[int, int]]]:
    row_count = len(case["row_ranges"])
    col_count = len(case["col_ranges"])
    rooms = [(ri, cj) for ri in range(row_count) for cj in range(col_count)]
    adj = {room: set() for room in rooms}

    for (i, j), door in case["h_doors"].items():
        if door["state"] == STATE_OPEN or (open_blocked and door["state"] == STATE_BLOCKED):
            u = (i, j)
            v = (i + 1, j)
            adj[u].add(v)
            adj[v].add(u)

    for (i, j), door in case["v_doors"].items():
        if door["state"] == STATE_OPEN or (open_blocked and door["state"] == STATE_BLOCKED):
            u = (i, j)
            v = (i, j + 1)
            adj[u].add(v)
            adj[v].add(u)

    return adj


def _expected_output_from_case(case: dict, grid: np.ndarray) -> np.ndarray:
    adj = _build_case_graph(case, open_blocked=False)
    dist = _bfs(adj, case["seed_room"])
    even_color = case["seeds"][0][1]
    odd_color = case["seeds"][1][1]
    return _fill_rooms(grid, case["row_ranges"], case["col_ranges"], dist, even_color, odd_color)


def _sample_seed_positions(room_r: tuple[int, int], room_c: tuple[int, int], even_color: int, odd_color: int) -> list[tuple[tuple[int, int], int]]:
    r0, r1 = room_r
    c0, c1 = room_c
    options: list[list[tuple[tuple[int, int], int]]] = []

    for r in range(r0, r1 + 1):
        for c in range(c0, c1):
            options.append([((r, c), even_color), ((r, c + 1), odd_color)])
    for r in range(r0, r1):
        for c in range(c0, c1 + 1):
            options.append([((r, c), even_color), ((r + 1, c), odd_color)])

    assert options
    return random.choice(options)


def _gen_attempt() -> np.ndarray | None:
    room_rows = random.randint(2, 4)
    room_cols = random.randint(2, 4)
    row_spans = _sample_spans(room_rows, 30)
    col_spans = _sample_spans(room_cols, 30)
    height = sum(row_spans) + room_rows - 1
    width = sum(col_spans) + room_cols - 1
    if max(height, width) > 30:
        return None

    bg = 0
    sep, even_color, odd_color, obstacle = random.sample(range(1, 10), 4)

    row_seps = []
    cursor = row_spans[0]
    for span in row_spans[:-1]:
        cursor = cursor if not row_seps else row_seps[-1] + 1 + span
        row_seps.append(cursor)
    col_seps = []
    cursor = col_spans[0]
    for span in col_spans[:-1]:
        cursor = cursor if not col_seps else col_seps[-1] + 1 + span
        col_seps.append(cursor)
    row_ranges = _partition_ranges(height, row_seps)
    col_ranges = _partition_ranges(width, col_seps)

    case = {
        "bg": bg,
        "sep": sep,
        "obstacle": obstacle,
        "height": height,
        "width": width,
        "row_seps": row_seps,
        "col_seps": col_seps,
        "row_ranges": row_ranges,
        "col_ranges": col_ranges,
        "h_doors": {},
        "v_doors": {},
    }

    for i, sr in enumerate(row_seps):
        _ = sr
        for j, (c0, c1) in enumerate(col_ranges):
            case["h_doors"][(i, j)] = {
                "state": random.choices(
                    [STATE_NONE, STATE_OPEN, STATE_BLOCKED],
                    weights=[0.25, 0.45, 0.30],
                    k=1,
                )[0],
                "coord": random.randint(c0, c1),
            }

    for j, sc in enumerate(col_seps):
        _ = sc
        for i, (r0, r1) in enumerate(row_ranges):
            case["v_doors"][(i, j)] = {
                "state": random.choices(
                    [STATE_NONE, STATE_OPEN, STATE_BLOCKED],
                    weights=[0.25, 0.45, 0.30],
                    k=1,
                )[0],
                "coord": random.randint(r0, r1),
            }

    blocked_count = sum(door["state"] == STATE_BLOCKED for door in case["h_doors"].values())
    blocked_count += sum(door["state"] == STATE_BLOCKED for door in case["v_doors"].values())
    if blocked_count < 2:
        return None

    seed_room = (random.randrange(room_rows), random.randrange(room_cols))
    case["seed_room"] = seed_room
    case["seeds"] = _sample_seed_positions(
        row_ranges[seed_room[0]],
        col_ranges[seed_room[1]],
        even_color,
        odd_color,
    )

    adj = _build_case_graph(case, open_blocked=False)
    dist = _bfs(adj, seed_room)
    if len(dist) < 3:
        return None
    if len(dist) == room_rows * room_cols:
        return None
    if max(dist.values()) < 2:
        return None

    all_open_dist = _bfs(_build_case_graph(case, open_blocked=True), seed_room)
    if set(all_open_dist) == set(dist):
        return None

    grid = _render_case(case)
    expected = _expected_output_from_case(case, grid)
    actual = main(grid)
    if not np.array_equal(expected, actual):
        return None

    parsed = _parse_grid(grid)
    if set(parsed["dist"]) != set(dist):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
