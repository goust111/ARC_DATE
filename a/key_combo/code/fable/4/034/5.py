from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


Coord = tuple[int, int]
Room = tuple[int, int]


def _unique_mode(values: np.ndarray, role: str) -> int:
    counts = Counter(map(int, values.ravel()))
    largest = max(counts.values())
    winners = [color for color, count in counts.items() if count == largest]
    assert len(winners) == 1, f"expected a unique {role} color"
    return int(winners[0])


def _wall_lines(g: np.ndarray, bg: int) -> tuple[list[int], list[int]]:
    """Find one-cell full-span walls, allowing a few one-cell door gaps."""
    h, w = g.shape
    row_min = max(4, int(np.ceil(0.60 * w)))
    col_min = max(4, int(np.ceil(0.60 * h)))
    rows = [r for r in range(h) if int(np.count_nonzero(g[r, :] != bg)) >= row_min]
    cols = [c for c in range(w) if int(np.count_nonzero(g[:, c] != bg)) >= col_min]

    assert len(rows) >= 3 and len(cols) >= 3, "expected outer and internal wall lines"
    assert rows[0] == 0 and rows[-1] == h - 1, "horizontal walls must bound the board"
    assert cols[0] == 0 and cols[-1] == w - 1, "vertical walls must bound the board"
    assert all(b - a >= 3 for a, b in zip(rows, rows[1:])), "rooms must have visible interiors"
    assert all(b - a >= 3 for a, b in zip(cols, cols[1:])), "rooms must have visible interiors"
    return rows, cols


def _line_colors(
    g: np.ndarray, bg: int, wall_rows: list[int], wall_cols: list[int]
) -> tuple[dict[int, int], dict[int, int]]:
    """Read a wall's color away from crossings and background door cells."""
    row_colors: dict[int, int] = {}
    col_colors: dict[int, int] = {}
    col_set = set(wall_cols)
    row_set = set(wall_rows)

    for r in wall_rows:
        values = [int(g[r, c]) for c in range(g.shape[1]) if c not in col_set and int(g[r, c]) != bg]
        assert values, "each horizontal wall needs visible colored cells"
        counts = Counter(values)
        top = max(counts.values())
        winners = [color for color, count in counts.items() if count == top]
        assert len(winners) == 1, "horizontal wall color must be unambiguous"
        row_colors[r] = int(winners[0])

    for c in wall_cols:
        values = [int(g[r, c]) for r in range(g.shape[0]) if r not in row_set and int(g[r, c]) != bg]
        assert values, "each vertical wall needs visible colored cells"
        counts = Counter(values)
        top = max(counts.values())
        winners = [color for color, count in counts.items() if count == top]
        assert len(winners) == 1, "vertical wall color must be unambiguous"
        col_colors[c] = int(winners[0])

    return row_colors, col_colors


def _interval_index(value: int, walls: list[int]) -> int:
    for i, (lo, hi) in enumerate(zip(walls, walls[1:])):
        if lo < value < hi:
            return i
    raise AssertionError("cell must lie strictly inside one room")


def _room_slice(room: Room, wall_rows: list[int], wall_cols: list[int]) -> tuple[slice, slice]:
    i, j = room
    return (
        slice(wall_rows[i] + 1, wall_rows[i + 1]),
        slice(wall_cols[j] + 1, wall_cols[j + 1]),
    )


def _doors(
    g: np.ndarray,
    bg: int,
    wall_rows: list[int],
    wall_cols: list[int],
    row_colors: dict[int, int],
    col_colors: dict[int, int],
) -> list[tuple[Room, Room, int, Coord]]:
    """Convert each background gap in an internal wall into a room-graph edge."""
    edges: list[tuple[Room, Room, int, Coord]] = []
    row_pos = {r: i for i, r in enumerate(wall_rows)}
    col_pos = {c: j for j, c in enumerate(wall_cols)}
    wall_row_set = set(wall_rows)
    wall_col_set = set(wall_cols)
    seen_gaps: set[Coord] = set()

    for r in wall_rows[1:-1]:
        i = row_pos[r]
        for c in range(1, g.shape[1] - 1):
            if c in wall_col_set or int(g[r, c]) != bg:
                continue
            j = _interval_index(c, wall_cols)
            assert r - 1 not in wall_row_set and r + 1 not in wall_row_set
            edges.append(((i - 1, j), (i, j), row_colors[r], (r, c)))
            seen_gaps.add((r, c))

    for c in wall_cols[1:-1]:
        j = col_pos[c]
        for r in range(1, g.shape[0] - 1):
            if r in wall_row_set or int(g[r, c]) != bg:
                continue
            i = _interval_index(r, wall_rows)
            edges.append(((i, j - 1), (i, j), col_colors[c], (r, c)))
            seen_gaps.add((r, c))

    # Every wall cell is colored except a recognized single-cell door.
    for r in wall_rows:
        for c in range(g.shape[1]):
            if int(g[r, c]) == bg:
                assert (r, c) in seen_gaps, "unexpected break in a horizontal wall"
    for c in wall_cols:
        for r in range(g.shape[0]):
            if int(g[r, c]) == bg:
                assert (r, c) in seen_gaps, "unexpected break in a vertical wall"

    pairs: set[frozenset[Room]] = set()
    for a, b, _color, _gap in edges:
        pair = frozenset((a, b))
        assert pair not in pairs, "expected at most one door between a pair of rooms"
        pairs.add(pair)
    assert edges, "expected at least one door"
    return edges


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate from the seeded room through a tree of wall gaps, recording each entry wall's color in the new room.

    Input structure:
    - A dominant background is divided into rectangular rooms by one-cell full-span horizontal and vertical walls.
    - Internal walls may have single-cell door gaps, whose room adjacency graph is one tree plus optional isolated rooms.
    - Exactly one non-wall foreground pixel is the seed inside a room; wall colors and the seed color are visible labels.

    Rule:
    - Fill the seed room with the seed color, then traverse the door tree from that room.
    - Fill every newly reached room with the color of the wall containing the door used to enter it.
    - Leave isolated unreachable rooms as background, and replace every door gap with its own wall's color.

    Key insight:
    - A room touching several differently colored walls is labeled by its parent edge in the seed-rooted door tree,
      so the entry wall can only be identified by following the propagation order from the seed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2-D grid"
    assert max(g.shape) <= 30, "ARC grids may not exceed 30 in either dimension"

    bg = _unique_mode(g, "background")
    wall_rows, wall_cols = _wall_lines(g, bg)
    row_colors, col_colors = _line_colors(g, bg, wall_rows, wall_cols)

    wall_row_set = set(wall_rows)
    wall_col_set = set(wall_cols)
    seed_cells = [
        (r, c)
        for r in range(g.shape[0])
        for c in range(g.shape[1])
        if r not in wall_row_set and c not in wall_col_set and int(g[r, c]) != bg
    ]
    assert len(seed_cells) == 1, "expected exactly one seed pixel inside the rooms"
    seed_r, seed_c = seed_cells[0]
    seed_color = int(g[seed_r, seed_c])
    root = (_interval_index(seed_r, wall_rows), _interval_index(seed_c, wall_cols))

    edges = _doors(g, bg, wall_rows, wall_cols, row_colors, col_colors)
    adjacency: dict[Room, list[tuple[Room, int]]] = {}
    active: set[Room] = set()
    for a, b, color, _gap in edges:
        adjacency.setdefault(a, []).append((b, color))
        adjacency.setdefault(b, []).append((a, color))
        active.update((a, b))

    assert root in active, "the seed room must belong to the door tree"
    reached_active = {root}
    q = deque([root])
    while q:
        room = q.popleft()
        for neighbor, _color in adjacency.get(room, []):
            if neighbor not in reached_active:
                reached_active.add(neighbor)
                q.append(neighbor)
    assert reached_active == active, "all doors must form one connected tree"
    assert len(edges) == len(active) - 1, "door connectivity must be acyclic"

    room_colors: dict[Room, int] = {root: seed_color}
    q = deque([root])
    while q:
        room = q.popleft()
        for neighbor, entry_color in adjacency.get(room, []):
            if neighbor in room_colors:
                continue
            room_colors[neighbor] = int(entry_color)
            q.append(neighbor)

    out = g.copy()
    for room, color in room_colors.items():
        rs, cs = _room_slice(room, wall_rows, wall_cols)
        out[rs, cs] = int(color)
    for _a, _b, color, (r, c) in edges:
        out[r, c] = int(color)
    return out


# ------------------------------ generator ------------------------------

BG = 0
NON_BG = list(range(1, 10))
MAX_ATTEMPTS = 500
_generation_count = 0


def _wall_positions(sizes: list[int]) -> list[int]:
    positions = [0]
    for size in sizes:
        positions.append(positions[-1] + int(size) + 1)
    return positions


def _neighbors(room: Room, n_rows: int, n_cols: int) -> list[Room]:
    i, j = room
    result = []
    for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        ni, nj = i + di, j + dj
        if 0 <= ni < n_rows and 0 <= nj < n_cols:
            result.append((ni, nj))
    return result


def _entry_color(
    a: Room,
    b: Room,
    wall_rows: list[int],
    wall_cols: list[int],
    row_colors: dict[int, int],
    col_colors: dict[int, int],
) -> tuple[int, str]:
    if a[0] != b[0]:
        wall = wall_rows[max(a[0], b[0])]
        return int(row_colors[wall]), "horizontal"
    wall = wall_cols[max(a[1], b[1])]
    return int(col_colors[wall]), "vertical"


def _grow_tree(
    n_rows: int,
    n_cols: int,
    target_size: int,
    wall_rows: list[int],
    wall_cols: list[int],
    row_colors: dict[int, int],
    col_colors: dict[int, int],
) -> tuple[Room, set[Room], list[tuple[Room, Room, int, str]], dict[Room, Room]] | None:
    for _ in range(160):
        root = (random.randrange(n_rows), random.randrange(n_cols))
        selected = {root}
        parents: dict[Room, Room] = {}
        raw_edges: list[tuple[Room, Room]] = []
        while len(selected) < target_size:
            frontier = [
                (a, b)
                for a in selected
                for b in _neighbors(a, n_rows, n_cols)
                if b not in selected
            ]
            if not frontier:
                break
            parent, child = random.choice(frontier)
            selected.add(child)
            parents[child] = parent
            raw_edges.append((parent, child))
        if len(selected) != target_size:
            continue

        edges = []
        depths = {root: 0}
        children = Counter()
        for parent, child in raw_edges:
            color, orientation = _entry_color(
                parent, child, wall_rows, wall_cols, row_colors, col_colors
            )
            edges.append((parent, child, color, orientation))
            depths[child] = depths[parent] + 1
            children[parent] += 1

        if max(depths.values()) < 2 or max(children.values(), default=0) < 2:
            continue
        if {orientation for *_rest, orientation in edges} != {"horizontal", "vertical"}:
            continue
        if len({color for _a, _b, color, _orientation in edges}) < 2:
            continue

        incident: dict[Room, list[int]] = {}
        for a, b, color, _orientation in edges:
            incident.setdefault(a, []).append(color)
            incident.setdefault(b, []).append(color)
        if not any(room != root and len(colors) >= 2 and len(set(colors)) >= 2 for room, colors in incident.items()):
            continue
        return root, selected, edges, parents
    return None


def _door_coordinate(a: Room, b: Room, wall_rows: list[int], wall_cols: list[int]) -> Coord:
    if a[0] != b[0]:
        r = wall_rows[max(a[0], b[0])]
        j = a[1]
        c = random.randint(wall_cols[j] + 1, wall_cols[j + 1] - 1)
        return r, c
    c = wall_cols[max(a[1], b[1])]
    i = a[0]
    r = random.randint(wall_rows[i] + 1, wall_rows[i + 1] - 1)
    return r, c


def _render_expected(
    inp: np.ndarray,
    root: Room,
    seed_color: int,
    wall_rows: list[int],
    wall_cols: list[int],
    edges: list[tuple[Room, Room, int, Coord]],
) -> np.ndarray:
    adjacency: dict[Room, list[tuple[Room, int]]] = {}
    for a, b, color, _gap in edges:
        adjacency.setdefault(a, []).append((b, color))
        adjacency.setdefault(b, []).append((a, color))

    colors = {root: int(seed_color)}
    q = deque([root])
    while q:
        room = q.popleft()
        for neighbor, entry_color in adjacency.get(room, []):
            if neighbor not in colors:
                colors[neighbor] = int(entry_color)
                q.append(neighbor)

    out = inp.copy()
    for room, color in colors.items():
        rs, cs = _room_slice(room, wall_rows, wall_cols)
        out[rs, cs] = color
    for _a, _b, color, gap in edges:
        out[gap] = color
    return out


def _validate_generated(
    inp: np.ndarray,
    expected: np.ndarray,
    root: Room,
    selected: set[Room],
    seed_cell: Coord,
    seed_color: int,
    wall_rows: list[int],
    wall_cols: list[int],
    edges: list[tuple[Room, Room, int, Coord]],
) -> bool:
    n_rows, n_cols = len(wall_rows) - 1, len(wall_cols) - 1
    if inp.ndim != 2 or max(inp.shape) > 30 or len(edges) != len(selected) - 1:
        return False
    if np.array_equal(inp, expected):
        return False
    if any(int(inp[gap]) != BG or int(expected[gap]) != color for _a, _b, color, gap in edges):
        return False
    if any(
        np.any(expected[_room_slice((i, j), wall_rows, wall_cols)] != BG)
        for i in range(n_rows)
        for j in range(n_cols)
        if (i, j) not in selected
    ):
        return False

    try:
        if not np.array_equal(main(inp), expected):
            return False

        # Counterexample to a seed-independent interpretation: root the same visible
        # door tree at another room and independently derive its new entry colors.
        alt_root = max(selected, key=lambda room: abs(room[0] - root[0]) + abs(room[1] - root[1]))
        if alt_root == root:
            return False
        alt_inp = inp.copy()
        alt_inp[seed_cell] = BG
        rs, cs = _room_slice(alt_root, wall_rows, wall_cols)
        alt_cell = (rs.start, cs.start)
        alt_inp[alt_cell] = seed_color
        alt_expected = _render_expected(
            alt_inp, alt_root, seed_color, wall_rows, wall_cols, edges
        )
        if np.array_equal(alt_expected, expected) or not np.array_equal(main(alt_inp), alt_expected):
            return False
    except (AssertionError, ValueError):
        return False
    return True


def _generate_attempt(want_unreachable: bool) -> np.ndarray | None:
    n_rows = random.randint(2, 4)
    n_cols = random.randint(2, 4)
    if n_rows * n_cols < 6:
        return None
    room_heights = [random.randint(4, 6) for _ in range(n_rows)]
    room_widths = [random.randint(4, 6) for _ in range(n_cols)]
    wall_rows = _wall_positions(room_heights)
    wall_cols = _wall_positions(room_widths)
    h, w = wall_rows[-1] + 1, wall_cols[-1] + 1
    if max(h, w) > 30:
        return None

    seed_color = random.choice(NON_BG)
    wall_palette = random.sample([color for color in NON_BG if color != seed_color], 3)
    row_colors = {r: random.choice(wall_palette) for r in wall_rows}
    col_colors = {c: random.choice(wall_palette) for c in wall_cols}

    total_rooms = n_rows * n_cols
    if want_unreachable:
        target_size = random.randint(4, total_rooms - 1)
    else:
        target_size = total_rooms
    grown = _grow_tree(
        n_rows, n_cols, target_size, wall_rows, wall_cols, row_colors, col_colors
    )
    if grown is None:
        return None
    root, selected, grown_edges, _parents = grown

    inp = np.full((h, w), BG, dtype=int)
    for c in wall_cols:
        inp[:, c] = col_colors[c]
    for r in wall_rows:
        inp[r, :] = row_colors[r]

    edges: list[tuple[Room, Room, int, Coord]] = []
    for a, b, color, _orientation in grown_edges:
        gap = _door_coordinate(a, b, wall_rows, wall_cols)
        inp[gap] = BG
        edges.append((a, b, color, gap))

    root_rs, root_cs = _room_slice(root, wall_rows, wall_cols)
    seed_cell = (
        random.randrange(root_rs.start, root_rs.stop),
        random.randrange(root_cs.start, root_cs.stop),
    )
    inp[seed_cell] = seed_color
    expected = _render_expected(inp, root, seed_color, wall_rows, wall_cols, edges)

    if _unique_mode(inp, "background") != BG:
        return None
    if not _validate_generated(
        inp,
        expected,
        root,
        selected,
        seed_cell,
        seed_color,
        wall_rows,
        wall_cols,
        edges,
    ):
        return None
    return inp


def generate_input() -> np.ndarray:
    """Generate varied boards while cycling the reachable-all / unreachable-room branch."""
    global _generation_count
    want_unreachable = (_generation_count % 3) != 1
    for _ in range(MAX_ATTEMPTS):
        grid = _generate_attempt(want_unreachable)
        if grid is not None:
            _generation_count += 1
            return grid
    raise RuntimeError("failed to generate a valid door-tree propagation input")
