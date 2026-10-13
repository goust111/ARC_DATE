from __future__ import annotations

import random

import numpy as np


def _mode_color(values: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(values, dtype=int), return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _find_dividers(g: np.ndarray, bg: int) -> tuple[list[int], list[int], int]:
    row_divs = [r for r in range(g.shape[0]) if bool(np.all(g[r, :] != bg))]
    col_divs = [c for c in range(g.shape[1]) if bool(np.all(g[:, c] != bg))]
    assert len(row_divs) >= 3 and len(col_divs) >= 3

    divider_samples = []
    for r in row_divs:
        divider_samples.extend(int(v) for v in g[r, :])
    for c in col_divs:
        divider_samples.extend(int(v) for v in g[:, c])
    divider_samples = np.asarray([v for v in divider_samples if v != bg], dtype=int)
    grey = _mode_color(divider_samples)
    return row_divs, col_divs, grey


def _room_bounds(row_divs: list[int], col_divs: list[int]) -> dict[tuple[int, int], dict[str, int]]:
    rooms: dict[tuple[int, int], dict[str, int]] = {}
    for ri in range(len(row_divs) - 1):
        r0 = row_divs[ri] + 1
        r1 = row_divs[ri + 1] - 1
        assert r0 <= r1
        for ci in range(len(col_divs) - 1):
            c0 = col_divs[ci] + 1
            c1 = col_divs[ci + 1] - 1
            assert c0 <= c1
            rooms[(ri, ci)] = {
                "r0": r0,
                "r1": r1,
                "c0": c0,
                "c1": c1,
                "top": row_divs[ri],
                "bottom": row_divs[ri + 1],
                "left": col_divs[ci],
                "right": col_divs[ci + 1],
            }
    return rooms


def _collect_doors(
    g: np.ndarray,
    bg: int,
    grey: int,
    row_divs: list[int],
    col_divs: list[int],
) -> tuple[dict[tuple[int, int], list[dict[str, object]]], set[tuple[int, int]]]:
    rooms = _room_bounds(row_divs, col_divs)
    room_doors = {room: [] for room in rooms}
    door_cells: set[tuple[int, int]] = set()
    row_set = set(row_divs)
    col_set = set(col_divs)

    def _room_col_index(c: int) -> int:
        for ci in range(len(col_divs) - 1):
            if col_divs[ci] < c < col_divs[ci + 1]:
                return ci
        raise AssertionError("door column must lie inside one room band")

    def _room_row_index(r: int) -> int:
        for ri in range(len(row_divs) - 1):
            if row_divs[ri] < r < row_divs[ri + 1]:
                return ri
        raise AssertionError("door row must lie inside one room band")

    for r, c in np.argwhere((g != bg) & (g != grey)):
        rr = int(r)
        cc = int(c)
        on_row = rr in row_set
        on_col = cc in col_set
        if on_row or on_col:
            assert not (on_row and on_col), "doors may not sit on divider intersections"
            door_cells.add((rr, cc))
            color = int(g[rr, cc])
            if on_row:
                ridx = row_divs.index(rr)
                assert 0 < ridx < len(row_divs) - 1, "doors must connect two rooms"
                cidx = _room_col_index(cc)
                above = (ridx - 1, cidx)
                below = (ridx, cidx)
                room_doors[above].append(
                    {"color": color, "side": "down", "target": below, "coord": (rr, cc)}
                )
                room_doors[below].append(
                    {"color": color, "side": "up", "target": above, "coord": (rr, cc)}
                )
            else:
                cidx = col_divs.index(cc)
                assert 0 < cidx < len(col_divs) - 1, "doors must connect two rooms"
                ridx = _room_row_index(rr)
                left = (ridx, cidx - 1)
                right = (ridx, cidx)
                room_doors[left].append(
                    {"color": color, "side": "right", "target": right, "coord": (rr, cc)}
                )
                room_doors[right].append(
                    {"color": color, "side": "left", "target": left, "coord": (rr, cc)}
                )

    return room_doors, door_cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each room's key to its colored wall gap, then pour that room's pattern into the adjacent room in the door's direction.

    Input structure:
    - The background is black, and one-pixel-wide grey divider rows and columns partition the grid into rectangular rooms.
    - Interior rooms contain colored pattern pixels plus one singleton key pixel.
    - Some internal divider cells are recolored as one-pixel doors; each room's key color matches exactly one door on that room's boundary.

    Rule:
    - For each room, find the boundary door whose color matches the room's key.
    - Remove all keys and all door pixels.
    - Translate every other non-black, non-grey pixel in the room straight through the matched door into the adjacent room.
    - The translation keeps the perpendicular coordinate fixed and pushes the pattern until its leading edge is flush with the far wall of the destination room; moved pixels overwrite earlier painted pixels.

    Key insight:
    - The key color selects which boundary gap belongs to the room, and the wall containing that gap encodes the movement direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _mode_color(g)
    row_divs, col_divs, grey = _find_dividers(g, bg)
    rooms = _room_bounds(row_divs, col_divs)
    room_doors, _door_cells = _collect_doors(g, bg, grey, row_divs, col_divs)

    out = np.full_like(g, bg)
    out[g == grey] = grey

    for room_id in sorted(rooms):
        room = rooms[room_id]
        sub = g[room["r0"] : room["r1"] + 1, room["c0"] : room["c1"] + 1]
        boundary_doors = room_doors[room_id]
        assert boundary_doors, "each room must see at least one boundary door"
        boundary_colors = {int(door["color"]) for door in boundary_doors}

        key_rel = np.argwhere(np.isin(sub, list(boundary_colors)))
        assert key_rel.shape == (1, 2), "each room must contain exactly one matching key pixel"
        key_rr, key_cc = map(int, key_rel[0])
        key_color = int(sub[key_rr, key_cc])

        matched = [door for door in boundary_doors if int(door["color"]) == key_color]
        assert len(matched) == 1, "key color must match exactly one boundary door"
        door = matched[0]

        content_mask = (sub != bg) & (sub != grey)
        content_mask[key_rr, key_cc] = False
        payload_rel = np.argwhere(content_mask)
        if payload_rel.size == 0:
            continue

        payload_abs = payload_rel + np.array([room["r0"], room["c0"]], dtype=int)
        payload_colors = sub[payload_rel[:, 0], payload_rel[:, 1]].astype(int)
        target = rooms[door["target"]]
        side = str(door["side"])

        if side == "up":
            delta = target["r0"] - int(payload_abs[:, 0].min())
            shifted = payload_abs + np.array([delta, 0], dtype=int)
        elif side == "down":
            delta = target["r1"] - int(payload_abs[:, 0].max())
            shifted = payload_abs + np.array([delta, 0], dtype=int)
        elif side == "left":
            delta = target["c0"] - int(payload_abs[:, 1].min())
            shifted = payload_abs + np.array([0, delta], dtype=int)
        else:
            assert side == "right"
            delta = target["c1"] - int(payload_abs[:, 1].max())
            shifted = payload_abs + np.array([0, delta], dtype=int)

        assert bool(
            np.all(
                (target["r0"] <= shifted[:, 0])
                & (shifted[:, 0] <= target["r1"])
                & (target["c0"] <= shifted[:, 1])
                & (shifted[:, 1] <= target["c1"])
            )
        )
        for (rr, cc), color in zip(shifted.tolist(), payload_colors.tolist(), strict=False):
            out[int(rr), int(cc)] = int(color)

    return out


BG = 0
GREY = 5
DOOR_COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
ROOM_COUNT_CHOICES = [
    (2, 2),
    (2, 3),
    (2, 4),
    (3, 2),
    (3, 4),
    (4, 2),
    (4, 3),
    (4, 4),
]
ROOM_SIZE_RANGE = (4, 6)
GEN_ATTEMPTS = 300

_CYCLE_CACHE: dict[tuple[int, int], list[tuple[int, int]]] = {}


def _neighbors(m: int, n: int, cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    out = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr = r + dr
        nc = c + dc
        if 0 <= nr < m and 0 <= nc < n:
            out.append((nr, nc))
    return out


def _base_cycle(m: int, n: int) -> list[tuple[int, int]]:
    key = (m, n)
    if key in _CYCLE_CACHE:
        return list(_CYCLE_CACHE[key])

    assert (m * n) % 2 == 0
    if m % 2 == 0:
        cycle: list[tuple[int, int]] = []
        for r in range(m):
            cols = range(1, n) if r % 2 == 0 else range(n - 1, 0, -1)
            for c in cols:
                cycle.append((r, c))
        for r in range(m - 1, -1, -1):
            cycle.append((r, 0))
    else:
        assert n % 2 == 0
        transposed = _base_cycle(n, m)
        cycle = [(c, r) for r, c in transposed]

    assert len(cycle) == m * n
    assert len(set(cycle)) == m * n
    assert all(
        abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1
        for a, b in zip(cycle, cycle[1:] + cycle[:1], strict=False)
    )
    _CYCLE_CACHE[key] = list(cycle)
    return list(cycle)


def _transform_cycle(cycle: list[tuple[int, int]], m: int, n: int) -> list[tuple[int, int]]:
    out = list(cycle)
    if random.random() < 0.5:
        out = list(reversed(out))
    if random.random() < 0.5:
        out = [(r, n - 1 - c) for r, c in out]
    if random.random() < 0.5:
        out = [(m - 1 - r, c) for r, c in out]
    return out


def _divider_positions(sizes: list[int]) -> list[int]:
    pos = [0]
    cur = 0
    for size in sizes:
        cur += size + 1
        pos.append(cur)
    return pos


def _choose_room_sizes(count: int) -> list[int]:
    return [random.randint(*ROOM_SIZE_RANGE) for _ in range(count)]


def _room_grid(m: int, n: int) -> tuple[np.ndarray, list[int], list[int], dict[tuple[int, int], dict[str, int]]]:
    heights = _choose_room_sizes(m)
    widths = _choose_room_sizes(n)
    row_divs = _divider_positions(heights)
    col_divs = _divider_positions(widths)
    grid = np.full((row_divs[-1] + 1, col_divs[-1] + 1), BG, dtype=int)
    grid[row_divs, :] = GREY
    grid[:, col_divs] = GREY
    rooms = _room_bounds(row_divs, col_divs)
    return grid, row_divs, col_divs, rooms


def _edge_side(src: tuple[int, int], dst: tuple[int, int]) -> str:
    dr = dst[0] - src[0]
    dc = dst[1] - src[1]
    if dr == -1:
        return "up"
    if dr == 1:
        return "down"
    if dc == -1:
        return "left"
    assert dc == 1
    return "right"


def _place_doors(
    grid: np.ndarray,
    cycle: list[tuple[int, int]],
    rooms: dict[tuple[int, int], dict[str, int]],
    row_divs: list[int],
    col_divs: list[int],
) -> dict[tuple[int, int], dict[str, object]]:
    door_colors = random.sample(DOOR_COLOR_POOL, 2)
    edge_info: dict[tuple[int, int], dict[str, object]] = {}
    used: set[tuple[int, int]] = set()
    N = len(cycle)

    for i, src in enumerate(cycle):
        dst = cycle[(i + 1) % N]
        side = _edge_side(src, dst)
        color = int(door_colors[i % 2])
        room = rooms[src]
        if side == "up":
            r = row_divs[src[0]]
            slots = [(r, c) for c in range(room["c0"], room["c1"] + 1)]
        elif side == "down":
            r = row_divs[src[0] + 1]
            slots = [(r, c) for c in range(room["c0"], room["c1"] + 1)]
        elif side == "left":
            c = col_divs[src[1]]
            slots = [(r, c) for r in range(room["r0"], room["r1"] + 1)]
        else:
            c = col_divs[src[1] + 1]
            slots = [(r, c) for r in range(room["r0"], room["r1"] + 1)]

        slots = [coord for coord in slots if coord not in used]
        assert slots
        door = random.choice(slots)
        used.add(door)
        grid[door] = color
        edge_info[src] = {"target": dst, "side": side, "door_color": color, "door": door}

    return edge_info


def _grow_polyomino(h: int, w: int, size: int) -> list[tuple[int, int]]:
    start = (random.randrange(h), random.randrange(w))
    cells = {start}
    while len(cells) < size:
        frontier = []
        for r, c in cells:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        assert frontier
        cells.add(random.choice(frontier))
    return sorted(cells)


def _shape_cells(max_h: int, max_w: int) -> list[tuple[int, int]]:
    kind = random.choice(["poly", "rect", "l", "t"])
    if kind == "rect":
        h = random.randint(1, min(max_h, 3))
        w = random.randint(2, min(max_w, 4))
        return [(r, c) for r in range(h) for c in range(w)]
    if kind == "l":
        h = random.randint(2, min(max_h, 4))
        w = random.randint(2, min(max_w, 4))
        arm_r = random.randint(1, h)
        arm_c = random.randint(1, w)
        cells = {(r, 0) for r in range(arm_r)} | {(0, c) for c in range(arm_c)}
        return sorted(cells)
    if kind == "t" and max_w >= 3:
        h = random.randint(2, min(max_h, 4))
        w_choices = [w for w in range(3, min(max_w, 5) + 1) if w % 2 == 1]
        if w_choices:
            w = random.choice(w_choices)
            mid = w // 2
            cells = {(0, c) for c in range(w)} | {(r, mid) for r in range(h)}
            return sorted(cells)

    h = random.randint(2, min(max_h, 4))
    w = random.randint(2, min(max_w, 4))
    size = random.randint(3, min(7, h * w))
    return _grow_polyomino(h, w, size)


def _paint_room_contents(
    grid: np.ndarray,
    room: dict[str, int],
    target: dict[str, int],
    door_color: int,
    forbidden_colors: set[int],
) -> None:
    max_h = min(room["r1"] - room["r0"] + 1, target["r1"] - target["r0"] + 1)
    max_w = min(room["c1"] - room["c0"] + 1, target["c1"] - target["c0"] + 1)
    cells = _shape_cells(max_h, max_w)
    shape_h = max(r for r, _ in cells) + 1
    shape_w = max(c for _, c in cells) + 1

    top = random.randint(room["r0"], room["r1"] - shape_h + 1)
    left = random.randint(room["c0"], room["c1"] - shape_w + 1)
    abs_cells = [(top + r, left + c) for r, c in cells]

    payload_palette = [c for c in DOOR_COLOR_POOL if c not in forbidden_colors]
    base_color = random.choice(payload_palette)
    accent_color = random.choice([c for c in payload_palette if c != base_color]) if random.random() < 0.45 else base_color
    accent_count = random.randint(1, max(1, len(abs_cells) // 2))
    accent_cells = set(random.sample(abs_cells, accent_count)) if accent_color != base_color else set()

    for cell in abs_cells:
        grid[cell] = accent_color if cell in accent_cells else base_color

    empty = [
        (r, c)
        for r in range(room["r0"], room["r1"] + 1)
        for c in range(room["c0"], room["c1"] + 1)
        if grid[r, c] == BG
    ]
    assert empty
    isolated = []
    for r, c in empty:
        if all(
            not (0 <= r + dr < grid.shape[0] and 0 <= c + dc < grid.shape[1] and grid[r + dr, c + dc] != BG)
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
        ):
            isolated.append((r, c))
    key_cell = random.choice(isolated or empty)
    grid[key_cell] = door_color


def _generate_attempt() -> np.ndarray | None:
    m, n = random.choice(ROOM_COUNT_CHOICES)
    grid, row_divs, col_divs, rooms = _room_grid(m, n)
    if max(grid.shape) > 30:
        return None

    cycle = _transform_cycle(_base_cycle(m, n), m, n)
    edge_info = _place_doors(grid, cycle, rooms, row_divs, col_divs)
    cycle_index = {room_id: idx for idx, room_id in enumerate(cycle)}

    for room_id, info in edge_info.items():
        room = rooms[room_id]
        target = rooms[info["target"]]
        forbidden = {GREY, BG}
        forbidden.add(int(info["door_color"]))
        prev_idx = cycle_index[room_id]
        prev_room = cycle[prev_idx - 1]
        forbidden.add(int(edge_info[prev_room]["door_color"]))
        _paint_room_contents(
            grid,
            room,
            target,
            int(info["door_color"]),
            forbidden,
        )

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid room-door ARC input")
