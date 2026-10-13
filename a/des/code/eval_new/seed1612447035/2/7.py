import random
from collections import Counter, deque

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _most_common_color(grid: np.ndarray) -> int:
    return Counter(map(int, grid.ravel())).most_common(1)[0][0]


def _largest_non_bg_color(grid: np.ndarray, bg: int) -> int:
    counts = Counter(map(int, grid.ravel()))
    for color, _ in counts.most_common():
        if int(color) != bg:
            return int(color)
    raise ValueError("expected at least one non-background color")


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while queue:
                cr, cc = queue.popleft()
                cells.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(cells)
    return components


def _detect_doors(grid: np.ndarray, bg: int, wall: int) -> dict[tuple[int, int], str]:
    h, w = grid.shape
    doors = {}
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            color = int(grid[r, c])
            if color == bg or color == wall:
                continue
            up = int(grid[r - 1, c])
            down = int(grid[r + 1, c])
            left = int(grid[r, c - 1])
            right = int(grid[r, c + 1])
            vertical = up == wall and down == wall and left != wall and right != wall
            horizontal = left == wall and right == wall and up != wall and down != wall
            if vertical ^ horizontal:
                doors[(r, c)] = "vertical" if vertical else "horizontal"
    return doors


def _room_groups(
    grid: np.ndarray, bg: int, wall: int, doors: dict[tuple[int, int], str]
) -> tuple[list[list[tuple[int, int]]], np.ndarray]:
    barrier = grid == wall
    for r, c in doors:
        barrier[r, c] = True
    passable = ~barrier

    room_id = -np.ones(grid.shape, dtype=int)
    rooms = []
    for cells in _components(passable):
        touches_border = any(
            r == 0 or c == 0 or r == grid.shape[0] - 1 or c == grid.shape[1] - 1 for r, c in cells
        )
        if touches_border:
            continue
        idx = len(rooms)
        rooms.append(cells)
        for r, c in cells:
            room_id[r, c] = idx
    return rooms, room_id


def _choose_fill_color(
    members: list[int],
    room_color_counts: list[Counter],
    room_first_pos: list[dict[int, int]],
    bg: int,
) -> int:
    total = Counter()
    first_pos = {}
    for rid in members:
        total.update(room_color_counts[rid])
        for color, pos in room_first_pos[rid].items():
            if color not in first_pos or pos < first_pos[color]:
                first_pos[color] = pos
    if not total:
        return bg
    best = max(total.values())
    candidates = [color for color, count in total.items() if count == best]
    return min(candidates, key=lambda color: first_pos[color])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Open doors between equally populated rooms and recolor each open-connected room group uniformly.

    Input structure:
    - bg is the most common color, with grey wall pixels forming a rectangular room layout.
    - Each room is a wall-enclosed rectangular interior; outside the wall structure may also be bg.
    - Shared walls between adjacent rooms may contain a one-pixel colored door, and rooms may contain scattered one-pixel colored items.

    Rule:
    - Identify every room while treating colored door pixels as wall breaks that do not yet connect rooms.
    - Count the number of item pixels inside each room.
    - A door opens iff the two rooms it separates contain the same number of items; open doors become traversable, closed doors become wall.
    - For each connected component of rooms linked by open doors, fill all member rooms and opened door pixels with the most frequent item color in that component, or keep bg if the component has no items.

    Key insight:
    - The colored door pixels are not classified by color value but by their wall-gap geometry, and the real hidden rule is that door state depends only on equality of item counts across the two adjacent rooms.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    wall = _largest_non_bg_color(g, bg)
    doors = _detect_doors(g, bg, wall)
    rooms, room_id = _room_groups(g, bg, wall, doors)

    room_item_counts = [0 for _ in rooms]
    room_color_counts = [Counter() for _ in rooms]
    room_first_pos = [{} for _ in rooms]
    h, w = g.shape

    for rid, cells in enumerate(rooms):
        for r, c in cells:
            color = int(g[r, c])
            if color == bg:
                continue
            room_item_counts[rid] += 1
            room_color_counts[rid][color] += 1
            pos = r * w + c
            if color not in room_first_pos[rid] or pos < room_first_pos[rid][color]:
                room_first_pos[rid][color] = pos

    graph = [[] for _ in rooms]
    door_edges = []
    for (r, c), orient in doors.items():
        if orient == "vertical":
            a = int(room_id[r, c - 1])
            b = int(room_id[r, c + 1])
        else:
            a = int(room_id[r - 1, c])
            b = int(room_id[r + 1, c])
        if a < 0 or b < 0 or a == b:
            continue
        open_door = room_item_counts[a] == room_item_counts[b]
        door_edges.append((r, c, a, b, open_door))
        if open_door:
            graph[a].append((b, (r, c)))
            graph[b].append((a, (r, c)))

    out = g.copy()
    for r, c, _a, _b, open_door in door_edges:
        out[r, c] = bg if open_door else wall

    seen = [False for _ in rooms]
    for start in range(len(rooms)):
        if seen[start]:
            continue
        queue = deque([start])
        seen[start] = True
        members = []
        open_doors = set()
        while queue:
            rid = queue.popleft()
            members.append(rid)
            for other, door_pos in graph[rid]:
                open_doors.add(door_pos)
                if not seen[other]:
                    seen[other] = True
                    queue.append(other)

        fill = _choose_fill_color(members, room_color_counts, room_first_pos, bg)
        for rid in members:
            for r, c in rooms[rid]:
                out[r, c] = fill
        for r, c in open_doors:
            out[r, c] = fill

    return out


def _partition_intervals(sizes: list[int], offset: int) -> tuple[list[int], list[tuple[int, int]]]:
    lines = [offset]
    intervals = []
    cursor = offset
    for size in sizes:
        start = cursor + 1
        end = start + size - 1
        intervals.append((start, end))
        cursor = end + 1
        lines.append(cursor)
    return lines, intervals


def _room_graph(n_rows: int, n_cols: int) -> list[tuple[tuple[int, int], tuple[int, int], str]]:
    edges = []
    for i in range(n_rows):
        for j in range(n_cols):
            if j + 1 < n_cols:
                edges.append(((i, j), (i, j + 1), "vertical"))
            if i + 1 < n_rows:
                edges.append(((i, j), (i + 1, j), "horizontal"))
    return edges


def _assign_room_counts(
    rng: random.Random,
    room_ids: list[tuple[int, int]],
    edges: list[tuple[tuple[int, int], tuple[int, int], str]],
) -> dict[tuple[int, int], int] | None:
    allow_zero = rng.random() < 0.5
    choices = [0, 1, 2, 3] if allow_zero else [1, 2, 3]
    for _ in range(200):
        counts = {room: rng.choice(choices) for room in room_ids}
        if allow_zero and 0 not in counts.values():
            continue
        if not any(value > 0 for value in counts.values()):
            continue
        equal_edges = [edge for edge in edges if counts[edge[0]] == counts[edge[1]]]
        diff_edges = [edge for edge in edges if counts[edge[0]] != counts[edge[1]]]
        if equal_edges and diff_edges:
            return counts
    return None


def _choose_doors(
    rng: random.Random,
    counts: dict[tuple[int, int], int],
    edges: list[tuple[tuple[int, int], tuple[int, int], str]],
) -> dict[tuple[tuple[int, int], tuple[int, int]], bool]:
    equal_edges = [edge for edge in edges if counts[edge[0]] == counts[edge[1]]]
    diff_edges = [edge for edge in edges if counts[edge[0]] != counts[edge[1]]]
    forced_open = rng.choice(equal_edges)
    forced_closed = rng.choice(diff_edges)

    present = {}
    for a, b, _orient in edges:
        key = tuple(sorted((a, b)))
        if (a, b, _orient) == forced_open or (a, b, _orient) == forced_closed:
            present[key] = True
        else:
            present[key] = rng.random() < 0.55
    return present


def _open_components(
    room_ids: list[tuple[int, int]],
    edges: list[tuple[tuple[int, int], tuple[int, int], str]],
    counts: dict[tuple[int, int], int],
    present: dict[tuple[tuple[int, int], tuple[int, int]], bool],
) -> list[list[tuple[int, int]]]:
    graph = {room: [] for room in room_ids}
    for a, b, _orient in edges:
        key = tuple(sorted((a, b)))
        if present.get(key, False) and counts[a] == counts[b]:
            graph[a].append(b)
            graph[b].append(a)

    seen = set()
    groups = []
    for room in room_ids:
        if room in seen:
            continue
        queue = deque([room])
        seen.add(room)
        group = []
        while queue:
            cur = queue.popleft()
            group.append(cur)
            for nxt in graph[cur]:
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        groups.append(group)
    return groups


def _assign_group_item_colors(
    rng: random.Random,
    groups: list[list[tuple[int, int]]],
    counts: dict[tuple[int, int], int],
    item_colors: list[int],
) -> dict[tuple[int, int], list[int]]:
    per_room = {room: [] for group in groups for room in group}
    for group in groups:
        total_items = sum(counts[room] for room in group)
        if total_items == 0:
            continue
        dominant = rng.choice(item_colors)
        dominant_count = rng.randint(total_items // 2 + 1, total_items)
        palette = [dominant] * dominant_count
        others = [color for color in item_colors if color != dominant]
        for _ in range(total_items - dominant_count):
            palette.append(rng.choice(others))
        rng.shuffle(palette)

        cursor = 0
        for room in group:
            amount = counts[room]
            per_room[room] = palette[cursor : cursor + amount]
            cursor += amount
    return per_room


def _paint_structure(
    n_rows: int,
    n_cols: int,
    row_sizes: list[int],
    col_sizes: list[int],
    counts: dict[tuple[int, int], int],
    door_present: dict[tuple[tuple[int, int], tuple[int, int]], bool],
    room_items: dict[tuple[int, int], list[int]],
    door_color: int,
    item_colors: list[int],
    rng: random.Random,
) -> np.ndarray:
    wall = 5
    bg = 0
    pad = 2

    part_h = sum(row_sizes) + n_rows + 1
    part_w = sum(col_sizes) + n_cols + 1
    grid = np.full((part_h + 2 * pad, part_w + 2 * pad), bg, dtype=int)

    row_lines, row_intervals = _partition_intervals(row_sizes, pad)
    col_lines, col_intervals = _partition_intervals(col_sizes, pad)
    top, bottom = row_lines[0], row_lines[-1]
    left, right = col_lines[0], col_lines[-1]

    for r in row_lines:
        grid[r, left : right + 1] = wall
    for c in col_lines:
        grid[top : bottom + 1, c] = wall

    for i in range(n_rows):
        for j in range(n_cols):
            room = (i, j)
            r0, r1 = row_intervals[i]
            c0, c1 = col_intervals[j]
            cells = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
            amount = counts[room]
            if amount:
                positions = rng.sample(cells, amount)
                colors = list(room_items[room])
                rng.shuffle(colors)
                for (r, c), color in zip(positions, colors):
                    grid[r, c] = color

    for i in range(n_rows):
        for j in range(n_cols):
            if j + 1 < n_cols:
                a = (i, j)
                b = (i, j + 1)
                key = tuple(sorted((a, b)))
                if door_present.get(key, False):
                    wall_c = col_lines[j + 1]
                    r0, r1 = row_intervals[i]
                    door_r = rng.randint(r0, r1)
                    grid[door_r, wall_c] = door_color
            if i + 1 < n_rows:
                a = (i, j)
                b = (i + 1, j)
                key = tuple(sorted((a, b)))
                if door_present.get(key, False):
                    wall_r = row_lines[i + 1]
                    c0, c1 = col_intervals[j]
                    door_c = rng.randint(c0, c1)
                    grid[wall_r, door_c] = door_color

    assert set(item_colors).isdisjoint({bg, wall, door_color})
    return grid


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_rows = rng.randint(2, 4)
    n_cols = rng.randint(2, 4)
    row_sizes = [rng.randint(2, 5) for _ in range(n_rows)]
    col_sizes = [rng.randint(2, 5) for _ in range(n_cols)]

    part_h = sum(row_sizes) + n_rows + 1
    part_w = sum(col_sizes) + n_cols + 1
    if part_h + 4 > 30 or part_w + 4 > 30:
        return None

    room_ids = [(i, j) for i in range(n_rows) for j in range(n_cols)]
    edges = _room_graph(n_rows, n_cols)
    counts = _assign_room_counts(rng, room_ids, edges)
    if counts is None:
        return None

    door_present = _choose_doors(rng, counts, edges)
    groups = _open_components(room_ids, edges, counts, door_present)

    palette = [color for color in range(1, 10) if color != 5]
    door_color = rng.choice(palette)
    item_colors = rng.sample([color for color in palette if color != door_color], 3)
    room_items = _assign_group_item_colors(rng, groups, counts, item_colors)
    grid = _paint_structure(
        n_rows,
        n_cols,
        row_sizes,
        col_sizes,
        counts,
        door_present,
        room_items,
        door_color,
        item_colors,
        rng,
    )

    if max(grid.shape) > 30:
        return None

    color_counts = Counter(map(int, grid.ravel()))
    if color_counts[0] <= color_counts[5]:
        return None

    door_states = []
    for a, b, orient in edges:
        key = tuple(sorted((a, b)))
        if not door_present.get(key, False):
            continue
        door_states.append(counts[a] == counts[b])
    if not door_states or not any(door_states) or all(door_states):
        return None

    has_zero_group = any(sum(counts[room] for room in group) == 0 for group in groups)
    if (0 in counts.values()) != has_zero_group:
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if out.shape != grid.shape:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(400):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid room-and-door puzzle")
