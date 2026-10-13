from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
AZURE = 8

WALL = GREY
SOURCE_COLORS = (RED, GREEN, BLUE)
ROTATE_COLOR = {RED: GREEN, GREEN: BLUE, BLUE: RED}
COLOR_TO_BIT = {RED: 1, GREEN: 2, BLUE: 4}
BIT_TO_COLOR = {
    1: RED,
    2: GREEN,
    3: YELLOW,
    4: BLUE,
    5: MAGENTA,
    6: AZURE,
    7: GREY,
}

MAX_GENERATION_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _detect_doors(grid: np.ndarray) -> list[dict[str, int | str]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    doors: list[dict[str, int | str]] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if int(g[r, c]) != BLACK:
                continue
            vertical = (
                int(g[r - 1, c]) == WALL
                and int(g[r + 1, c]) == WALL
                and int(g[r, c - 1]) != WALL
                and int(g[r, c + 1]) != WALL
            )
            horizontal = (
                int(g[r, c - 1]) == WALL
                and int(g[r, c + 1]) == WALL
                and int(g[r - 1, c]) != WALL
                and int(g[r + 1, c]) != WALL
            )
            if vertical:
                doors.append({"r": r, "c": c, "orient": "vertical"})
            elif horizontal:
                doors.append({"r": r, "c": c, "orient": "horizontal"})
    return doors


def _label_rooms(grid: np.ndarray, doors: list[dict[str, int | str]]) -> tuple[np.ndarray, dict[int, dict[str, object]]]:
    closed = np.asarray(grid, dtype=int).copy()
    for door in doors:
        closed[int(door["r"]), int(door["c"])] = WALL

    mask = closed != WALL
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=int)
    rooms: dict[int, dict[str, object]] = {}
    next_id = 0

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != 0:
                continue
            next_id += 1
            q: deque[tuple[int, int]] = deque([(r, c)])
            labels[r, c] = next_id
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and labels[nr, nc] == 0:
                        labels[nr, nc] = next_id
                        q.append((nr, nc))

            coords = np.asarray(pts, dtype=int)
            r0, c0, r1, c1 = _bbox(coords)
            assert bool(np.all(mask[r0 : r1 + 1, c0 : c1 + 1]))
            rooms[next_id] = {
                "bbox": (r0, c0, r1, c1),
                "source": None,
                "doors": [],
            }

    return labels, rooms


def _find_sources(grid: np.ndarray, labels: np.ndarray, rooms: dict[int, dict[str, object]]) -> None:
    g = np.asarray(grid, dtype=int)
    for color in SOURCE_COLORS:
        for coords in _components(g == color):
            r0, c0, r1, c1 = _bbox(coords)
            assert coords.shape[0] == 4
            assert (r1 - r0 + 1, c1 - c0 + 1) == (2, 2)
            assert bool(np.all(g[r0 : r1 + 1, c0 : c1 + 1] == color))
            room_id = int(labels[r0, c0])
            assert room_id != 0
            assert rooms[room_id]["source"] is None
            rooms[room_id]["source"] = {"bbox": (r0, c0, r1, c1), "color": color}


def _build_door_graph(
    doors: list[dict[str, int | str]],
    labels: np.ndarray,
    rooms: dict[int, dict[str, object]],
) -> dict[tuple[int, int], dict[str, int | str]]:
    door_map: dict[tuple[int, int], dict[str, int | str]] = {}
    for door in doors:
        r = int(door["r"])
        c = int(door["c"])
        orient = str(door["orient"])
        if orient == "vertical":
            left = int(labels[r, c - 1])
            right = int(labels[r, c + 1])
            assert left != 0 and right != 0 and left != right
            info = {"orient": "vertical", "left": left, "right": right}
            rooms[left]["doors"].append({"r": r, "c": c, "direction": "right", "other": right})
            rooms[right]["doors"].append({"r": r, "c": c, "direction": "left", "other": left})
        else:
            up = int(labels[r - 1, c])
            down = int(labels[r + 1, c])
            assert up != 0 and down != 0 and up != down
            info = {"orient": "horizontal", "up": up, "down": down}
            rooms[up]["doors"].append({"r": r, "c": c, "direction": "down", "other": down})
            rooms[down]["doors"].append({"r": r, "c": c, "direction": "up", "other": up})
        door_map[(r, c)] = info
    return door_map


def _mark_cell(contrib: np.ndarray, grid: np.ndarray, r: int, c: int, color: int) -> None:
    if int(grid[r, c]) == BLACK:
        contrib[r, c] |= COLOR_TO_BIT[color]


def _trace_room(
    grid: np.ndarray,
    contrib: np.ndarray,
    rooms: dict[int, dict[str, object]],
    door_map: dict[tuple[int, int], dict[str, int | str]],
    room_id: int,
    direction: str,
    line: int,
    color: int,
) -> None:
    r0, c0, r1, c1 = rooms[room_id]["bbox"]  # type: ignore[misc]

    if direction == "right":
        assert r0 <= line <= r1
        for c in range(c0, c1 + 1):
            _mark_cell(contrib, grid, line, c, color)
        exit_pos = (line, c1 + 1)
        exit_info = door_map.get(exit_pos)
        if exit_info is not None and str(exit_info["orient"]) == "vertical" and int(exit_info["left"]) == room_id:
            _mark_cell(contrib, grid, exit_pos[0], exit_pos[1], color)
            _trace_room(
                grid,
                contrib,
                rooms,
                door_map,
                int(exit_info["right"]),
                direction,
                line,
                ROTATE_COLOR[color],
            )
        return

    if direction == "left":
        assert r0 <= line <= r1
        for c in range(c1, c0 - 1, -1):
            _mark_cell(contrib, grid, line, c, color)
        exit_pos = (line, c0 - 1)
        exit_info = door_map.get(exit_pos)
        if exit_info is not None and str(exit_info["orient"]) == "vertical" and int(exit_info["right"]) == room_id:
            _mark_cell(contrib, grid, exit_pos[0], exit_pos[1], color)
            _trace_room(
                grid,
                contrib,
                rooms,
                door_map,
                int(exit_info["left"]),
                direction,
                line,
                ROTATE_COLOR[color],
            )
        return

    if direction == "down":
        assert c0 <= line <= c1
        for r in range(r0, r1 + 1):
            _mark_cell(contrib, grid, r, line, color)
        exit_pos = (r1 + 1, line)
        exit_info = door_map.get(exit_pos)
        if exit_info is not None and str(exit_info["orient"]) == "horizontal" and int(exit_info["up"]) == room_id:
            _mark_cell(contrib, grid, exit_pos[0], exit_pos[1], color)
            _trace_room(
                grid,
                contrib,
                rooms,
                door_map,
                int(exit_info["down"]),
                direction,
                line,
                ROTATE_COLOR[color],
            )
        return

    assert direction == "up"
    assert c0 <= line <= c1
    for r in range(r1, r0 - 1, -1):
        _mark_cell(contrib, grid, r, line, color)
    exit_pos = (r0 - 1, line)
    exit_info = door_map.get(exit_pos)
    if exit_info is not None and str(exit_info["orient"]) == "horizontal" and int(exit_info["down"]) == room_id:
        _mark_cell(contrib, grid, exit_pos[0], exit_pos[1], color)
        _trace_room(
            grid,
            contrib,
            rooms,
            door_map,
            int(exit_info["up"]),
            direction,
            line,
            ROTATE_COLOR[color],
        )


def main(grid: np.ndarray) -> np.ndarray:
    """Project axis-aligned light through room doors, rotating color at each door crossing."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    doors = _detect_doors(g)
    labels, rooms = _label_rooms(g, doors)
    _find_sources(g, labels, rooms)
    door_map = _build_door_graph(doors, labels, rooms)

    contrib = np.zeros(g.shape, dtype=np.uint8)
    for room_id, room in rooms.items():
        source = room["source"]
        if source is None:
            continue

        sr0, sc0, sr1, sc1 = source["bbox"]  # type: ignore[index]
        color = int(source["color"])  # type: ignore[index]
        source_rows = {int(sr0), int(sr1)}
        source_cols = {int(sc0), int(sc1)}

        for door in room["doors"]:
            r = int(door["r"])
            c = int(door["c"])
            direction = str(door["direction"])
            next_room = int(door["other"])

            if direction == "right" and r in source_rows:
                for cc in range(sc1 + 1, c + 1):
                    _mark_cell(contrib, g, r, cc, color)
                _trace_room(g, contrib, rooms, door_map, next_room, direction, r, ROTATE_COLOR[color])
            elif direction == "left" and r in source_rows:
                for cc in range(sc0 - 1, c - 1, -1):
                    _mark_cell(contrib, g, r, cc, color)
                _trace_room(g, contrib, rooms, door_map, next_room, direction, r, ROTATE_COLOR[color])
            elif direction == "down" and c in source_cols:
                for rr in range(sr1 + 1, r + 1):
                    _mark_cell(contrib, g, rr, c, color)
                _trace_room(g, contrib, rooms, door_map, next_room, direction, c, ROTATE_COLOR[color])
            elif direction == "up" and c in source_cols:
                for rr in range(sr0 - 1, r - 1, -1):
                    _mark_cell(contrib, g, rr, c, color)
                _trace_room(g, contrib, rooms, door_map, next_room, direction, c, ROTATE_COLOR[color])

    out = g.copy()
    black_mask = g == BLACK
    painted = contrib != 0
    for bitmask, color in BIT_TO_COLOR.items():
        out[black_mask & painted & (contrib == bitmask)] = color
    return out


def _weighted_room_count() -> int:
    return random.choices([2, 3, 4], weights=[2, 4, 3], k=1)[0]


def _room_starts(lengths: list[int]) -> list[int]:
    starts: list[int] = []
    cursor = 1
    for size in lengths:
        starts.append(cursor)
        cursor += size + 1
    return starts


def _choose_source_position(
    height: int,
    width: int,
    desired_row: int | None = None,
    desired_col: int | None = None,
    avoid_rows: set[int] | None = None,
    avoid_cols: set[int] | None = None,
) -> tuple[int, int]:
    avoid_rows = set() if avoid_rows is None else set(avoid_rows)
    avoid_cols = set() if avoid_cols is None else set(avoid_cols)

    top_options = list(range(height - 1))
    left_options = list(range(width - 1))
    if desired_row is not None:
        top_options = [top for top in (desired_row - 1, desired_row) if 0 <= top <= height - 2]
    if desired_col is not None:
        left_options = [left for left in (desired_col - 1, desired_col) if 0 <= left <= width - 2]

    strict: list[tuple[int, int]] = []
    loose: list[tuple[int, int]] = []
    for top in top_options:
        row_set = {top, top + 1}
        for left in left_options:
            col_set = {left, left + 1}
            loose.append((top, left))
            if row_set.isdisjoint(avoid_rows) and col_set.isdisjoint(avoid_cols):
                strict.append((top, left))

    candidates = strict if strict else loose
    assert candidates
    return random.choice(candidates)


def _render_case(
    room_h: list[int],
    room_w: list[int],
    vdoors: list[list[int]],
    hdoors: list[list[int]],
    sources: dict[tuple[int, int], dict[str, int]],
) -> np.ndarray:
    row_starts = _room_starts(room_h)
    col_starts = _room_starts(room_w)
    height = row_starts[-1] + room_h[-1] + 1
    width = col_starts[-1] + room_w[-1] + 1

    grid = np.full((height, width), BLACK, dtype=int)
    grid[0, :] = WALL
    grid[-1, :] = WALL
    grid[:, 0] = WALL
    grid[:, -1] = WALL

    for i, rh in enumerate(room_h[:-1]):
        wall_r = row_starts[i] + rh
        grid[wall_r, :] = WALL
    for j, rw in enumerate(room_w[:-1]):
        wall_c = col_starts[j] + rw
        grid[:, wall_c] = WALL

    for i in range(len(room_h)):
        room_top = row_starts[i]
        for j in range(len(room_w) - 1):
            wall_c = col_starts[j] + room_w[j]
            grid[room_top + vdoors[i][j], wall_c] = BLACK

    for i in range(len(room_h) - 1):
        wall_r = row_starts[i] + room_h[i]
        for j in range(len(room_w)):
            grid[wall_r, col_starts[j] + hdoors[i][j]] = BLACK

    for (ri, cj), source in sources.items():
        top = row_starts[ri] + source["top"]
        left = col_starts[cj] + source["left"]
        grid[top : top + 2, left : left + 2] = source["color"]

    return grid


def _validate_structure(
    grid: np.ndarray,
    room_h: list[int],
    room_w: list[int],
    vdoors: list[list[int]],
    hdoors: list[list[int]],
    sources: dict[tuple[int, int], dict[str, int]],
) -> bool:
    row_starts = _room_starts(room_h)
    col_starts = _room_starts(room_w)

    if not (
        np.all(grid[0, :] == WALL)
        and np.all(grid[-1, :] == WALL)
        and np.all(grid[:, 0] == WALL)
        and np.all(grid[:, -1] == WALL)
    ):
        return False

    for i, rh in enumerate(room_h[:-1]):
        wall_r = row_starts[i] + rh
        black_count = 0
        for j in range(len(room_w)):
            left = col_starts[j]
            right = left + room_w[j] - 1
            segment = grid[wall_r, left : right + 1]
            if int(np.sum(segment == BLACK)) != 1:
                return False
            black_count += 1
        if black_count != len(room_w):
            return False

    for j, rw in enumerate(room_w[:-1]):
        wall_c = col_starts[j] + rw
        black_count = 0
        for i in range(len(room_h)):
            top = row_starts[i]
            bottom = top + room_h[i] - 1
            segment = grid[top : bottom + 1, wall_c]
            if int(np.sum(segment == BLACK)) != 1:
                return False
            black_count += 1
        if black_count != len(room_h):
            return False

    for (ri, cj), source in sources.items():
        top = row_starts[ri] + source["top"]
        left = col_starts[cj] + source["left"]
        if not np.all(grid[top : top + 2, left : left + 2] == source["color"]):
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    room_rows = _weighted_room_count()
    room_cols = _weighted_room_count()
    room_h = [random.randint(4, 7) for _ in range(room_rows)]
    room_w = [random.randint(4, 7) for _ in range(room_cols)]

    h_row = random.randrange(room_rows)
    v_col = random.randrange(room_cols)
    h_line = random.randint(1, room_h[h_row] - 2)
    v_line = random.randint(1, room_w[v_col] - 2)

    vdoors = [
        [random.randint(0, room_h[i] - 1) for _ in range(room_cols - 1)]
        for i in range(room_rows)
    ]
    hdoors = [
        [random.randint(0, room_w[j] - 1) for j in range(room_cols)]
        for _ in range(room_rows - 1)
    ]

    for j in range(room_cols - 1):
        vdoors[h_row][j] = h_line
    for i in range(room_rows - 1):
        hdoors[i][v_col] = v_line

    colors = list(SOURCE_COLORS)
    random.shuffle(colors)
    h_color, v_color = colors[:2]
    extra_color = colors[2]

    h_src_col = random.randrange(room_cols)
    v_src_row_choices = [r for r in range(room_rows) if (r, v_col) != (h_row, h_src_col)]
    if not v_src_row_choices:
        return None
    v_src_row = random.choice(v_src_row_choices)

    source_specs: list[tuple[str, int, int, int]] = [
        ("h", h_row, h_src_col, h_color),
        ("v", v_src_row, v_col, v_color),
    ]

    add_extra = random.random() < 0.55
    if add_extra:
        use_horizontal = random.random() < 0.5 or room_rows == 1
        if use_horizontal:
            extra_cols = [c for c in range(room_cols) if (h_row, c) not in {(h_row, h_src_col), (v_src_row, v_col)}]
            if extra_cols:
                source_specs.append(("h", h_row, random.choice(extra_cols), extra_color))
        else:
            extra_rows = [r for r in range(room_rows) if (r, v_col) not in {(h_row, h_src_col), (v_src_row, v_col)}]
            if extra_rows:
                source_specs.append(("v", random.choice(extra_rows), v_col, extra_color))

    sources: dict[tuple[int, int], dict[str, int]] = {}
    for orient, ri, cj, color in source_specs:
        desired_row = h_line if ri == h_row else None
        desired_col = v_line if cj == v_col else None

        avoid_rows: set[int] = set()
        avoid_cols: set[int] = set()
        if desired_row is None:
            if cj > 0:
                avoid_rows.add(vdoors[ri][cj - 1])
            if cj < room_cols - 1:
                avoid_rows.add(vdoors[ri][cj])
        if desired_col is None:
            if ri > 0:
                avoid_cols.add(hdoors[ri - 1][cj])
            if ri < room_rows - 1:
                avoid_cols.add(hdoors[ri][cj])

        top, left = _choose_source_position(
            room_h[ri],
            room_w[cj],
            desired_row=desired_row,
            desired_col=desired_col,
            avoid_rows=avoid_rows,
            avoid_cols=avoid_cols,
        )
        sources[(ri, cj)] = {"top": top, "left": left, "color": color}

    grid = _render_case(room_h, room_w, vdoors, hdoors, sources)
    if not _validate_structure(grid, room_h, room_w, vdoors, hdoors, sources):
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    if len(sources) < 2:
        return None
    if int(np.sum((grid == BLACK) & (out != grid))) < 10:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid light-room ARC input")
